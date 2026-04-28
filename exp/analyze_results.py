"""
分析 benchmark 结果脚本

正确处理 checkpoint 文件中的重复记录，统计各模型的详细指标。
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

RESULTS_DIR = Path("exp/results")

def load_unique_results(checkpoint_file: Path) -> Dict[str, dict]:
    """
    加载 checkpoint 文件，只保留每个 case_id 的最后一条记录

    Returns:
        Dict[case_id, record]: 去重后的结果字典
    """
    results = {}

    if not checkpoint_file.exists():
        return results

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                case_id = data.get("case_id")
                if case_id:
                    # 后面的记录覆盖前面的（保留最后一次运行结果）
                    results[case_id] = data

    return results

def compute_stats(results: Dict[str, dict]) -> dict:
    """
    计算单个模型的统计指标
    """
    if not results:
        return {}

    total = len(results)

    # 状态统计
    success_count = sum(1 for r in results.values() if r.get("status") == "success" or r.get("total_score", 0) > 0)
    failed_count = total - success_count

    # 只统计有分数的记录
    scored_results = [r for r in results.values() if r.get("total_score", 0) > 0]
    scored_count = len(scored_results)

    if scored_count == 0:
        return {
            "total_cases": total,
            "success_count": success_count,
            "failed_count": failed_count,
            "scored_count": 0,
        }

    # 诊断指标
    diagnosis_correct = sum(1 for r in scored_results if r.get("diagnosis_correct", False))
    diagnosis_scores = [r.get("diagnosis_score", 0) for r in scored_results]

    # 治疗指标
    treatment_correct = sum(1 for r in scored_results if r.get("treatment_correct", False))
    treatment_scores = [r.get("treatment_score", 0) for r in scored_results]

    # 安全指标
    avoid_violated = sum(1 for r in scored_results if r.get("avoid_violated", False))
    avoid_scores = [r.get("avoid_score", 0) for r in scored_results]

    # 综合分数
    total_scores = [r.get("total_score", 0) for r in scored_results]

    # 效率指标（从 trajectory 计算）
    total_steps = 0
    total_tokens = 0
    total_latency = 0
    step_counts = 0

    for r in scored_results:
        trajectory = r.get("trajectory", [])
        if trajectory:
            total_steps += len(trajectory)
            step_counts += 1
            for step in trajectory:
                total_tokens += step.get("estimated_tokens", 0)
                total_latency += step.get("latency", 0)

    # 分数分布
    score_dist = {
        "excellent (>=4)": sum(1 for s in total_scores if s >= 4),
        "good (3-4)": sum(1 for s in total_scores if 3 <= s < 4),
        "medium (2-3)": sum(1 for s in total_scores if 2 <= s < 3),
        "poor (<2)": sum(1 for s in total_scores if s < 2),
    }

    return {
        "total_cases": total,
        "success_count": success_count,
        "failed_count": failed_count,
        "scored_count": scored_count,

        # 诊断
        "diagnosis_accuracy": diagnosis_correct / scored_count,
        "diagnosis_avg_score": sum(diagnosis_scores) / scored_count,

        # 治疗
        "treatment_accuracy": treatment_correct / scored_count,
        "treatment_avg_score": sum(treatment_scores) / scored_count,

        # 安全
        "avoid_violation_rate": avoid_violated / scored_count,
        "avoid_avg_score": sum(avoid_scores) / scored_count,

        # 综合
        "total_avg_score": sum(total_scores) / scored_count,
        "score_distribution": score_dist,

        # 效率
        "avg_steps": total_steps / step_counts if step_counts > 0 else 0,
        "avg_tokens": total_tokens / step_counts if step_counts > 0 else 0,
        "avg_latency": total_latency / step_counts if step_counts > 0 else 0,
    }

def main():
    print("=" * 80)
    print("MedAgent Benchmark 结果分析 (去重统计)")
    print("=" * 80)

    # 查找所有 checkpoint 文件
    checkpoint_files = list(RESULTS_DIR.glob("checkpoint_*.jsonl"))

    all_stats = {}

    for checkpoint_file in checkpoint_files:
        model_name = checkpoint_file.stem.replace("checkpoint_", "")

        print(f"\n处理: {model_name}")
        results = load_unique_results(checkpoint_file)
        stats = compute_stats(results)
        all_stats[model_name] = stats

        print(f"  总数: {stats.get('total_cases', 0)}")
        print(f"  成功: {stats.get('success_count', 0)}, 失败: {stats.get('failed_count', 0)}")
        if stats.get('scored_count', 0) > 0:
            print(f"  综合分数: {stats.get('total_avg_score', 0):.2f}")

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("汇总表格")
    print("=" * 80)

    # 按综合分数排序
    sorted_models = sorted(
        all_stats.items(),
        key=lambda x: x[1].get("total_avg_score", 0),
        reverse=True
    )

    print("\n| 模型 | 成功数 | 失败数 | 综合分数 | 诊断准确率 | 治疗准确率 | 禁忌违反率 |")
    print("|------|--------|--------|---------|-----------|-----------|-----------|")

    for model, stats in sorted_models:
        success = stats.get("success_count", 0)
        failed = stats.get("failed_count", 0)
        total_score = stats.get("total_avg_score", 0)
        diag_acc = stats.get("diagnosis_accuracy", 0) * 100
        treat_acc = stats.get("treatment_accuracy", 0) * 100
        avoid_rate = stats.get("avoid_violation_rate", 0) * 100

        print(f"| {model} | {success} | {failed} | **{total_score:.2f}** | {diag_acc:.1f}% | {treat_acc:.1f}% | {avoid_rate:.1f}% |")

    # 效率对比
    print("\n" + "=" * 80)
    print("效率对比")
    print("=" * 80)

    print("\n| 模型 | 平均步数 | 平均Token数 | 平均耗时(s) |")
    print("|------|---------|------------|------------|")

    for model, stats in sorted_models:
        avg_steps = stats.get("avg_steps", 0)
        avg_tokens = stats.get("avg_tokens", 0)
        avg_latency = stats.get("avg_latency", 0)

        print(f"| {model} | {avg_steps:.1f} | {avg_tokens:.0f} | {avg_latency:.1f} |")

    # 分数分布
    print("\n" + "=" * 80)
    print("分数分布")
    print("=" * 80)

    for model, stats in sorted_models[:5]:
        dist = stats.get("score_distribution", {})
        scored = stats.get("scored_count", 0)
        if scored > 0:
            print(f"\n{model}:")
            for level, count in dist.items():
                pct = count / scored * 100
                print(f"  {level}: {count} ({pct:.1f}%)")

    # 保存详细 JSON
    output_file = RESULTS_DIR / "detailed_stats.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print(f"\n\n详细统计已保存到: {output_file}")

    # 找出最佳模型
    best_model = sorted_models[0][0]
    best_score = sorted_models[0][1].get("total_avg_score", 0)
    print(f"\n最佳模型: {best_model} (综合分数: {best_score:.2f})")

if __name__ == "__main__":
    main()