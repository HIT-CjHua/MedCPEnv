"""
Judger 重测脚本 (并发版)

使用当前的 JUDGE_PROMPT 对已有 checkpoint 结果重新做诊断/治疗/安全评分。
不重新跑 Agent，只重新调用 Judger.evaluate()。

Usage:
    # 全量重测所有模型 (并发10)
    python exp/rejudge.py --all

    # 指定模型
    python exp/rejudge.py --models qwen3.5-plus,gpt-5.4 --n 50

    # 调整并发数
    python exp/rejudge.py --all --workers 20

    # 干跑
    python exp/rejudge.py --all --dry-run
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.medagent import Judger

RESULTS_DIR = "exp/output"
DATA_PATH = "exp/data/benchmark_1000.jsonl"

ALL_MODELS = [
    "qwen3.5-plus",
    "qwen3-max-2026-01-23",
    "glm-5",
    "kimi-k2.5",
    "MiniMax-M2.5",
    "deepseek-v3.2",
    "qwen3.5-35b-a3b",
    "gpt-5.4",
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
]


def load_benchmark_complaints() -> Dict[str, str]:
    """加载 benchmark 数据的 chief_complaint"""
    complaints = {}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                complaints[data["case_id"]] = data.get("chief_complaint", "")
    return complaints


def load_checkpoint_entries(model_name: str) -> list:
    """加载 checkpoint 文件"""
    path = os.path.join(RESULTS_DIR, f"checkpoint_{model_name}.jsonl")
    if not os.path.exists(path):
        return []
    entries_by_case = {}
    order = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                case_id = entry.get("case_id")
                if not case_id:
                    continue
                if case_id not in entries_by_case:
                    order.append(case_id)
                entries_by_case[case_id] = entry
    return [entries_by_case[case_id] for case_id in order]


def save_rejudge_results(model_name: str, results: list, output_dir: str):
    """保存重测结果"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"rejudge_checkpoint_{model_name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def compute_stats(results: list) -> Dict:
    """计算统计指标"""
    if not results:
        return {}

    total = len(results)
    valid = [r for r in results if r.get("total_score", 0) > 0]
    valid_count = len(valid)

    diagnosis_correct = sum(1 for r in valid if r.get("diagnosis_correct", False))
    treatment_correct = sum(1 for r in valid if r.get("treatment_correct", False))
    avoid_violated = sum(1 for r in valid if r.get("avoid_violated", False))

    avg_diagnosis = sum(r.get("diagnosis_score", 0) for r in valid) / valid_count if valid_count else 0
    avg_treatment = sum(r.get("treatment_score", 0) for r in valid) / valid_count if valid_count else 0
    avg_avoid = sum(r.get("avoid_score", 0) for r in valid) / valid_count if valid_count else 0
    avg_total = sum(r.get("total_score", 0) for r in valid) / valid_count if valid_count else 0

    score_dist = {
        "excellent": sum(1 for r in valid if r.get("total_score", 0) >= 4),
        "good": sum(1 for r in valid if 3 <= r.get("total_score", 0) < 4),
        "medium": sum(1 for r in valid if 2 <= r.get("total_score", 0) < 3),
        "poor": sum(1 for r in valid if r.get("total_score", 0) < 2),
    }

    safety_avg = avg_avoid

    return {
        "total_cases": total,
        "valid_cases": valid_count,
        "diagnosis": {
            "accuracy": diagnosis_correct / valid_count if valid_count else 0,
            "avg_score": avg_diagnosis,
        },
        "treatment": {
            "accuracy": treatment_correct / valid_count if valid_count else 0,
            "avg_score": avg_treatment,
        },
        "safety": {
            "violation_rate": avoid_violated / valid_count if valid_count else 0,
            "avg_score": safety_avg,
        },
        "total_avg_score": avg_total,
        "score_distribution": score_dist,
    }


def judge_one_entry(entry: dict, chief_complaint: str, judger: Judger) -> dict:
    """单条重测 (用于并发)"""
    case_id = entry.get("case_id", "unknown")
    trajectory = entry.get("trajectory", [])
    agent_diagnosis = entry.get("agent_diagnosis", "")
    agent_treatment = entry.get("agent_treatment", "")
    ground_truth = entry.get("ground_truth", {})

    eval_result = judger.evaluate(
        case_id=case_id,
        chief_complaint=chief_complaint,
        ground_truth=ground_truth,
        trajectory=trajectory,
        agent_diagnosis=agent_diagnosis,
        agent_treatment=agent_treatment,
    )

    # 更新字段
    entry["diagnosis_correct"] = eval_result.diagnosis_correct
    entry["diagnosis_score"] = eval_result.diagnosis_score
    entry["diagnosis_reason"] = eval_result.diagnosis_reason
    entry["treatment_correct"] = eval_result.treatment_correct
    entry["treatment_score"] = eval_result.treatment_score
    entry["treatment_reason"] = eval_result.treatment_reason
    entry["avoid_violated"] = eval_result.avoid_violated
    entry["avoid_score"] = eval_result.avoid_score
    entry["avoid_reason"] = eval_result.avoid_reason
    entry["avoid_violations"] = eval_result.avoid_violations
    entry["total_score"] = eval_result.total_score

    return entry


def rejudge_model_concurrent(
    model_name: str,
    entries: list,
    complaints: Dict[str, str],
    judger: Judger,
    max_workers: int = 10,
    dry_run: bool = False,
) -> list:
    """并发重测单个模型"""
    results = [None] * len(entries)
    success = 0
    failed = 0
    completed = 0

    if dry_run:
        for i, entry in enumerate(entries):
            results[i] = entry
        print(f"  {model_name} 完成: {len(entries)} 条 (dry-run)")
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, entry in enumerate(entries):
            case_id = entry.get("case_id", "unknown")
            chief_complaint = complaints.get(case_id, "")
            future = executor.submit(judge_one_entry, entry, chief_complaint, judger)
            futures[future] = i

        with tqdm(total=len(entries), desc=f"  {model_name}", unit="条") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                    success += 1
                except Exception as e:
                    failed += 1
                    results[idx] = entries[idx]

                completed += 1
                pbar.update(1)
                pbar.set_postfix({"成功": success, "失败": failed})

    print(f"  {model_name} 完成: {success} 成功, {failed} 失败")
    return results


def save_summary(all_stats: Dict, output_dir: str):
    """保存 summary 文件"""
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "rejudge_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
    }
    for model, stats in sorted(all_stats.items(), key=lambda x: x[1].get("total_avg_score", 0), reverse=True):
        summary["models"][model] = {
            "total_cases": stats["total_cases"],
            "diagnosis_accuracy": round(stats["diagnosis"]["accuracy"] * 100, 1),
            "diagnosis_avg": round(stats["diagnosis"]["avg_score"], 2),
            "treatment_accuracy": round(stats["treatment"]["accuracy"] * 100, 1),
            "treatment_avg": round(stats["treatment"]["avg_score"], 2),
            "safety_violation_rate": round(stats["safety"]["violation_rate"] * 100, 1),
            "safety_avg": round(stats["safety"]["avg_score"], 2),
            "total_avg_score": round(stats["total_avg_score"], 2),
            "score_distribution": stats["score_distribution"],
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary 已保存至: {summary_path}")


def print_table(all_stats: Dict):
    """打印汇总表"""
    print(f"\n{'='*60}")
    print("重测汇总")
    print(f"{'='*60}")
    print(f"\n| 模型 | 诊断准确率 | 诊断分数 | 治疗准确率 | 治疗分数 | 安全违反率 | 安全分数 | 综合分数 |")
    print(f"|------|-----------|---------|-----------|---------|-----------|---------|---------|")
    for model, stats in sorted(all_stats.items(), key=lambda x: x[1].get("total_avg_score", 0), reverse=True):
        print(f"| {model} | {stats['diagnosis']['accuracy']*100:.1f}% | {stats['diagnosis']['avg_score']:.2f} | {stats['treatment']['accuracy']*100:.1f}% | {stats['treatment']['avg_score']:.2f} | {stats['safety']['violation_rate']*100:.1f}% | {stats['safety']['avg_score']:.2f} | **{stats['total_avg_score']:.2f}** |")


def main():
    parser = argparse.ArgumentParser(description="Judger 重测脚本 (并发版)")
    parser.add_argument("--all", action="store_true", help="全量重测所有模型")
    parser.add_argument("--models", type=str, default=None, help="指定模型 (逗号分隔)")
    parser.add_argument("--n", type=int, default=None, help="每个模型重测数量")
    parser.add_argument("--output", type=str, default="exp/results", help="输出目录")
    parser.add_argument("--dry-run", action="store_true", help="只计数不调用 API")
    parser.add_argument("--workers", type=int, default=10, help="并发数 (默认10)")
    parser.add_argument("--summary-only", action="store_true", help="仅汇总已有重测结果，不重新评测")
    args = parser.parse_args()

    # 仅汇总模式
    if args.summary_only:
        models = ALL_MODELS if args.all else ([m.strip() for m in args.models.split(",")] if args.models else ALL_MODELS)
        print(f"[Summary] 汇总已有重测结果...")
        all_stats = {}
        for m in models:
            path = os.path.join(RESULTS_DIR, f"rejudge_checkpoint_{m}.jsonl")
            if not os.path.exists(path):
                print(f"  [Skip] {m}: 文件不存在")
                continue
            with open(path, "r", encoding="utf-8") as f:
                entries = [json.loads(line) for line in f if line.strip()]
            if entries:
                stats = compute_stats(entries)
                all_stats[m] = stats
                print(f"  {m}: {stats['total_cases']} 条")
        print_table(all_stats)
        save_summary(all_stats, args.output)
        return

    if args.all:
        models = ALL_MODELS
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        print("请使用 --all 或 --models 指定模型")
        return

    # 加载 chief_complaint
    print(f"[1/3] 加载 benchmark 数据...")
    complaints = load_benchmark_complaints()
    print(f"  加载 {len(complaints)} 条 chief_complaint")

    # 加载 checkpoint
    print(f"\n[2/3] 加载 checkpoint 并筛选...")
    model_entries = {}
    for m in models:
        entries = load_checkpoint_entries(m)
        if not entries:
            print(f"  [Skip] {m}: 无 checkpoint")
            continue
        if args.n:
            entries = entries[:args.n]
        model_entries[m] = entries
        print(f"  {m}: {len(entries)} 条待重测")

    if not model_entries:
        print("没有需要重测的模型")
        return

    # 初始化 Judger
    judger = Judger()

    # 重测
    print(f"\n[3/3] 开始重测{' [DRY-RUN]' if args.dry_run else ''} (workers={args.workers})...")
    start_time = time.time()

    all_stats = {}
    for model_name, entries in model_entries.items():
        results = rejudge_model_concurrent(
            model_name=model_name,
            entries=entries,
            complaints=complaints,
            judger=judger,
            max_workers=args.workers,
            dry_run=args.dry_run,
        )

        if not args.dry_run:
            save_rejudge_results(model_name, results, args.output)
            stats = compute_stats(results)
            all_stats[model_name] = stats

        elapsed = time.time() - start_time
        print(f"\n  {model_name} 耗时: {elapsed:.1f}s")
        if stats and not args.dry_run:
            print(f"    诊断准确率: {stats['diagnosis']['accuracy']*100:.1f}% (avg={stats['diagnosis']['avg_score']:.2f})")
            print(f"    治疗准确率: {stats['treatment']['accuracy']*100:.1f}% (avg={stats['treatment']['avg_score']:.2f})")
            print(f"    安全违反率: {stats['safety']['violation_rate']*100:.1f}% (avg={stats['safety']['avg_score']:.2f})")
            print(f"    综合分数:   {stats['total_avg_score']:.2f}")
            dist = stats["score_distribution"]
            print(f"    分数分布: 优秀={dist['excellent']}, 良好={dist['good']}, 中等={dist['medium']}, 较差={dist['poor']}")

    if all_stats:
        print_table(all_stats)
        save_summary(all_stats, args.output)

    print(f"\n总耗时: {time.time() - start_time:.1f}s")
    print("完成!")


if __name__ == "__main__":
    main()
