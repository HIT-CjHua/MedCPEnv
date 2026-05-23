"""
Benchmark Cost 补算脚本

为所有 checkpoint 中 total_cost 为 0 的条目计算费用。

Usage:
    python exp/compute_missing_costs.py --all
    python exp/compute_missing_costs.py --models qwen3.5-plus,qwen3.6-plus
    python exp/compute_missing_costs.py --dry-run
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.medagent.cost import CostEvaluator
from src.medagent.llm import api_counter

RESULTS_DIR = "exp/results"

ALL_MODELS = [
    "qwen3.5-plus",
    "qwen3.6-plus",
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


def load_checkpoint_entries(model_name: str) -> list:
    """加载 checkpoint 文件中的所有条目"""
    path = os.path.join(RESULTS_DIR, f"checkpoint_{model_name}.jsonl")
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def save_checkpoint_entries(model_name: str, entries: list):
    """覆盖保存 checkpoint 文件"""
    path = os.path.join(RESULTS_DIR, f"checkpoint_{model_name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def compute_costs_for_model(
    model_name: str,
    cost_evaluator: CostEvaluator,
    dry_run: bool = False,
    save_every: int = 10,
) -> dict:
    """
    为单个模型补算缺失的 cost

    Returns:
        dict: 统计信息 {total, computed, skipped, errors}
    """
    entries = load_checkpoint_entries(model_name)
    if not entries:
        print(f"  [Skip] {model_name}: 无 checkpoint 文件")
        return {"total": 0, "computed": 0, "skipped": 0, "errors": 0}

    # 找出没有 cost 的条目
    missing_indices = []
    for i, entry in enumerate(entries):
        cost = entry.get("total_cost", 0)
        if not cost or cost == 0:
            missing_indices.append(i)

    total_missing = len(missing_indices)
    if total_missing == 0:
        print(f"  [Done] {model_name}: 全部已有 cost (1000/1000)")
        return {"total": 1000, "computed": 0, "skipped": 1000, "errors": 0}

    print(f"  [Computing] {model_name}: {total_missing}/{len(entries)} 条缺失 cost")

    stats = {"total": len(entries), "computed": 0, "skipped": 0, "errors": 0}
    batch = []

    for idx_pos, entry_idx in enumerate(missing_indices):
        entry = entries[entry_idx]
        case_id = entry.get("case_id", "unknown")

        if dry_run:
            print(f"  [Dry-run] {case_id}: 将计算 cost")
            stats["computed"] += 1
            continue

        # 构建 agent_result 供 CostEvaluator 使用
        agent_result = {
            "case_id": case_id,
            "chief_complaint": "",
            "trajectory": entry.get("trajectory", []),
            "diagnosis": entry.get("agent_diagnosis", ""),
            "treatment": entry.get("agent_treatment", ""),
        }

        try:
            cost_result = cost_evaluator.estimate_from_agent_result(agent_result)
            cost_val = int(round(cost_result.total_cost))

            # 更新条目
            entry["total_cost"] = cost_val
            entry["cost_detail"] = {
                "service_cost": cost_result.service_cost,
                "medicine_cost": cost_result.medicine_cost,
                "matched_count": cost_result.matched_count,
                "generated_count": cost_result.generated_count,
                "service_items": [
                    {
                        "name": item.item_name,
                        "price": item.price,
                        "quantity": item.quantity,
                        "source": item.source,
                    }
                    for item in cost_result.service_items
                ],
                "medicine_items": [
                    {
                        "name": item.item_name,
                        "price": item.price,
                        "quantity": item.quantity,
                        "source": item.source,
                    }
                    for item in cost_result.medicine_items
                ],
            }
            entries[entry_idx] = entry
            stats["computed"] += 1

            if cost_val > 0:
                print(f"  [{idx_pos+1}/{total_missing}] {case_id}: {cost_val}元", flush=True)
            else:
                print(f"  [{idx_pos+1}/{total_missing}] {case_id}: 0元 (无项目)", flush=True)

        except Exception as e:
            stats["errors"] += 1
            print(f"  [{idx_pos+1}/{total_missing}] {case_id}: ERROR - {e}", flush=True)
            # 429 限流，等待后重试
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 60
                print(f"    等待 {wait}s...", flush=True)
                time.sleep(wait)
                # 重新尝试一次
                try:
                    cost_result = cost_evaluator.estimate_from_agent_result(agent_result)
                    cost_val = int(round(cost_result.total_cost))
                    entry["total_cost"] = cost_val
                    entry["cost_detail"] = {
                        "service_cost": cost_result.service_cost,
                        "medicine_cost": cost_result.medicine_cost,
                        "matched_count": cost_result.matched_count,
                        "generated_count": cost_result.generated_count,
                        "service_items": [
                            {"name": i.item_name, "price": i.price, "quantity": i.quantity, "source": i.source}
                            for i in cost_result.service_items
                        ],
                        "medicine_items": [
                            {"name": i.item_name, "price": i.price, "quantity": i.quantity, "source": i.source}
                            for i in cost_result.medicine_items
                        ],
                    }
                    entries[entry_idx] = entry
                    stats["computed"] += 1
                    stats["errors"] -= 1
                except Exception as e2:
                    print(f"    重试仍然失败: {e2}", flush=True)

        # 定期保存
        batch.append(entry_idx)
        if len(batch) >= save_every:
            if not dry_run:
                save_checkpoint_entries(model_name, entries)
            batch.clear()

        # 每次请求间短暂等待，避免限流
        time.sleep(0.5)

    # 最终保存
    if not dry_run and batch:
        save_checkpoint_entries(model_name, entries)

    return stats


def main():
    parser = argparse.ArgumentParser(description="补算 Benchmark 缺失的 cost 数据")
    parser.add_argument("--all", action="store_true", help="处理所有模型")
    parser.add_argument("--models", type=str, default=None, help="指定模型 (逗号分隔)")
    parser.add_argument("--dry-run", action="store_true", help="只统计不实际计算")
    parser.add_argument("--save-every", type=int, default=10, help="每 N 条保存一次")
    args = parser.parse_args()

    if args.all:
        models = ALL_MODELS
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        print("请使用 --all 或 --models 指定要处理的模型")
        return

    # 先统计各模型缺失情况
    print("=" * 60)
    print("Benchmark Cost 补算")
    print("=" * 60)
    print("\n缺失情况统计:")
    total_missing_all = 0
    for m in ALL_MODELS:
        entries = load_checkpoint_entries(m)
        missing = sum(1 for e in entries if not e.get("total_cost", 0))
        if missing > 0:
            print(f"  {m}: {missing}/{len(entries)} 缺失")
            total_missing_all += missing
        else:
            print(f"  {m}: 全部已有 cost")
    print(f"\n总计缺失: {total_missing_all} 条")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry-run 模式] 不会实际调用 API")

    # 初始化 CostEvaluator
    cost_evaluator = CostEvaluator()

    overall_stats = {}
    start_time = time.time()

    for model_name in models:
        entries = load_checkpoint_entries(model_name)
        missing = sum(1 for e in entries if not e.get("total_cost", 0))
        if missing == 0:
            continue

        stats = compute_costs_for_model(
            model_name=model_name,
            cost_evaluator=cost_evaluator,
            dry_run=args.dry_run,
            save_every=args.save_every,
        )
        overall_stats[model_name] = stats
        elapsed = time.time() - start_time
        print(f"\n  {model_name} 完成: {stats['computed']} 已算, {stats['errors']} 错误, 耗时 {elapsed:.1f}s")

    # 打印汇总
    if overall_stats:
        print("\n" + "=" * 60)
        print("补算汇总")
        print("=" * 60)
        total_computed = sum(s["computed"] for s in overall_stats.values())
        total_errors = sum(s["errors"] for s in overall_stats.values())
        print(f"  已计算: {total_computed} 条")
        print(f"  错误: {total_errors} 条")
        print(f"  总耗时: {time.time() - start_time:.1f}s")
        print(f"  总调用次数: {api_counter.get_count()}")

    # 保存价格清单
    cost_evaluator._save_price_list()

    print("\n完成!")


if __name__ == "__main__":
    main()
