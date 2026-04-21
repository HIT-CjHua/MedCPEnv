"""
Benchmark 汇总脚本

读取已完成的模型评测结果，生成对比报告。

每个模型评测完成后会保存独立的结果文件 (model_stats_{model}.json)，
本脚本读取所有结果文件并生成汇总对比报告。

Usage:
    # 汇报所有已完成的模型结果
    python exp/summarize_benchmark.py

    # 指定结果目录
    python exp/summarize_benchmark.py --results-dir exp/results

    # 指定输出报告名称
    python exp/summarize_benchmark.py --output exp/results/final_report.md
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# 读取模型结果
# =============================================================================

def load_model_stats(results_dir: str) -> Dict[str, Dict]:
    """读取所有模型的结果文件"""
    all_stats = {}

    # 查找所有 model_stats_*.json 文件
    for filename in os.listdir(results_dir):
        if filename.startswith("model_stats_") and filename.endswith(".json"):
            model_name = filename.replace("model_stats_", "").replace(".json", "")
            filepath = os.path.join(results_dir, filename)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                    all_stats[model_name] = stats
                    print(f"  加载: {model_name} ({stats.get('total_cases', 0)} 条)")
            except Exception as e:
                print(f"  [Error] 加载 {filename} 失败: {e}")

    return all_stats


def load_checkpoint_stats(results_dir: str) -> Dict[str, Dict]:
    """从 checkpoint 文件计算统计 (备选方案)"""
    all_stats = {}

    # 查找所有 checkpoint_*.jsonl 文件
    for filename in os.listdir(results_dir):
        if filename.startswith("checkpoint_") and filename.endswith(".jsonl"):
            model_name = filename.replace("checkpoint_", "").replace(".jsonl", "")
            filepath = os.path.join(results_dir, filename)

            try:
                results = []
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            results.append(data)

                if results:
                    stats = compute_stats_from_results(results)
                    all_stats[model_name] = stats
                    print(f"  从 checkpoint 计算: {model_name} ({len(results)} 条)")

            except Exception as e:
                print(f"  [Error] 处理 {filename} 失败: {e}")

    return all_stats


def compute_stats_from_results(results: List[Dict]) -> Dict:
    """从结果列表计算统计指标"""
    total = len(results)

    if total == 0:
        return {}

    # 准确率统计
    diagnosis_correct = sum(1 for r in results if r.get("diagnosis_correct", False))
    treatment_correct = sum(1 for r in results if r.get("treatment_correct", False))
    avoid_violated = sum(1 for r in results if r.get("avoid_violated", False))

    # 平均分数
    avg_diagnosis = sum(r.get("diagnosis_score", 0) for r in results) / total
    avg_treatment = sum(r.get("treatment_score", 0) for r in results) / total
    avg_avoid = sum(r.get("avoid_score", 0) for r in results) / total
    avg_total = sum(r.get("total_score", 0) for r in results) / total

    # 分数分布
    score_dist = {
        "excellent": sum(1 for r in results if r.get("total_score", 0) >= 4),
        "good": sum(1 for r in results if 3 <= r.get("total_score", 0) < 4),
        "medium": sum(1 for r in results if 2 <= r.get("total_score", 0) < 3),
        "poor": sum(1 for r in results if r.get("total_score", 0) < 2),
    }

    return {
        "total_cases": total,
        "diagnosis": {
            "accuracy": diagnosis_correct / total,
            "avg_score": avg_diagnosis,
        },
        "treatment": {
            "accuracy": treatment_correct / total,
            "avg_score": avg_treatment,
        },
        "safety": {
            "violation_rate": avoid_violated / total,
            "avg_score": avg_avoid,
        },
        "efficiency": {
            "avg_steps": 0,  # checkpoint 中可能没有
            "avg_exam_items": 0,
            "avg_tokens": 0,
            "avg_latency": 0,
        },
        "total_avg_score": avg_total,
        "score_distribution": score_dist,
    }


# =============================================================================
# 报告生成
# =============================================================================

def generate_comparison_table(all_stats: Dict[str, Dict]) -> str:
    """生成对比表格 (Markdown)"""
    lines = []
    lines.append("\n## 模型对比表格\n")
    lines.append("| 模型 | 病例数 | 诊断准确率 | 诊断分数 | 治疗准确率 | 治疗分数 | 禁忌违反率 | 安全分数 | 综合分数 |")
    lines.append("|------|-------|-----------|---------|-----------|---------|-----------|---------|---------|")

    # 按综合分数排序
    sorted_models = sorted(
        all_stats.items(),
        key=lambda x: x[1].get("total_avg_score", 0),
        reverse=True
    )

    for model, stats in sorted_models:
        total_cases = stats.get("total_cases", 0)
        diag_acc = stats["diagnosis"]["accuracy"] * 100
        diag_score = stats["diagnosis"]["avg_score"]
        treat_acc = stats["treatment"]["accuracy"] * 100
        treat_score = stats["treatment"]["avg_score"]
        avoid_rate = stats["safety"]["violation_rate"] * 100
        avoid_score = stats["safety"]["avg_score"]
        total_score = stats["total_avg_score"]

        lines.append(f"| {model} | {total_cases} | {diag_acc:.1f}% | {diag_score:.2f} | {treat_acc:.1f}% | {treat_score:.2f} | {avoid_rate:.1f}% | {avoid_score:.2f} | **{total_score:.2f}** |")

    return "\n".join(lines)


def generate_efficiency_table(all_stats: Dict[str, Dict]) -> str:
    """生成效率对比表格"""
    lines = []
    lines.append("\n## 效率指标对比\n")
    lines.append("| 模型 | 平均轮次 | 平均检查项 | 平均Token数 | 平均耗时(s) |")
    lines.append("|------|---------|-----------|------------|------------|")

    for model, stats in sorted(all_stats.items(), key=lambda x: x[1].get("total_avg_score", 0), reverse=True):
        eff = stats.get("efficiency", {})
        if eff.get("avg_steps", 0) > 0:
            lines.append(f"| {model} | {eff.get('avg_steps', 0):.1f} | {eff.get('avg_exam_items', 0):.1f} | {eff.get('avg_tokens', 0):.0f} | {eff.get('avg_latency', 0):.2f} |")
        else:
            lines.append(f"| {model} | - | - | - | - |")

    return "\n".join(lines)


def generate_ranking(all_stats: Dict[str, Dict]) -> str:
    """生成排名列表"""
    lines = []
    lines.append("\n## 模型排名\n")

    sorted_models = sorted(all_stats.items(), key=lambda x: x[1].get("total_avg_score", 0), reverse=True)

    for i, (model, stats) in enumerate(sorted_models, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        score = stats["total_avg_score"]
        cases = stats.get("total_cases", 0)
        lines.append(f"{medal} **{model}**: {score:.2f} 分 ({cases} 条病例)")

    return "\n".join(lines)


def generate_score_distribution(all_stats: Dict[str, Dict]) -> str:
    """生成分数分布统计"""
    lines = []
    lines.append("\n## 分数分布\n")

    sorted_models = sorted(all_stats.items(), key=lambda x: x[1].get("total_avg_score", 0), reverse=True)

    for model, stats in sorted_models[:5]:  # 只显示前5个
        dist = stats.get("score_distribution", {})
        total = stats.get("total_cases", 0)

        if total > 0 and dist:
            lines.append(f"\n### {model}")
            lines.append(f"- 优秀 (4-5分): {dist.get('excellent', 0)} ({dist.get('excellent', 0)/total*100:.1f}%)")
            lines.append(f"- 良好 (3-4分): {dist.get('good', 0)} ({dist.get('good', 0)/total*100:.1f}%)")
            lines.append(f"- 中等 (2-3分): {dist.get('medium', 0)} ({dist.get('medium', 0)/total*100:.1f}%)")
            lines.append(f"- 较差 (<2分): {dist.get('poor', 0)} ({dist.get('poor', 0)/total*100:.1f}%)")

    return "\n".join(lines)


def generate_markdown_report(all_stats: Dict[str, Dict], results_dir: str) -> str:
    """生成完整的 Markdown 报告"""
    lines = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# MedAgent Benchmark 评测汇总报告\n")
    lines.append(f"**生成时间**: {timestamp}")
    lines.append(f"**结果目录**: {results_dir}")
    lines.append(f"**模型数量**: {len(all_stats)}")

    # 排名
    lines.append(generate_ranking(all_stats))

    # 对比表格
    lines.append(generate_comparison_table(all_stats))

    # 效率表格
    lines.append(generate_efficiency_table(all_stats))

    # 分数分布
    lines.append(generate_score_distribution(all_stats))

    # 模型详情
    lines.append("\n## 各模型详情\n")
    sorted_models = sorted(all_stats.items(), key=lambda x: x[1].get("total_avg_score", 0), reverse=True)

    for model, stats in sorted_models:
        lines.append(f"\n### {model}\n")
        lines.append(f"- **评测病例数**: {stats.get('total_cases', 0)}")
        lines.append(f"- **综合分数**: {stats['total_avg_score']:.2f}")
        lines.append(f"- **诊断准确率**: {stats['diagnosis']['accuracy']*100:.1f}%")
        lines.append(f"- **诊断平均分**: {stats['diagnosis']['avg_score']:.2f}")
        lines.append(f"- **治疗准确率**: {stats['treatment']['accuracy']*100:.1f}%")
        lines.append(f"- **治疗平均分**: {stats['treatment']['avg_score']:.2f}")
        lines.append(f"- **禁忌违反率**: {stats['safety']['violation_rate']*100:.1f}%")
        lines.append(f"- **安全平均分**: {stats['safety']['avg_score']:.2f}")

    lines.append("\n---")
    lines.append(f"*报告由 summarize_benchmark.py 自动生成 ({timestamp})*")

    return "\n".join(lines)


def generate_console_summary(all_stats: Dict[str, Dict]):
    """生成控制台摘要"""
    print("\n" + "=" * 70)
    print("Benchmark 汇总结果")
    print("=" * 70)

    sorted_models = sorted(all_stats.items(), key=lambda x: x[1].get("total_avg_score", 0), reverse=True)

    print("\n| 排名 | 模型 | 病例数 | 综合分数 | 诊断准确率 | 治疗准确率 | 禁忌违反率 |")
    print("|------|------|-------|---------|-----------|-----------|-----------|")
    for i, (model, stats) in enumerate(sorted_models, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)
        print(f"| {medal} | {model} | {stats.get('total_cases', 0)} | **{stats['total_avg_score']:.2f}** | {stats['diagnosis']['accuracy']*100:.1f}% | {stats['treatment']['accuracy']*100:.1f}% | {stats['safety']['violation_rate']*100:.1f}% |")

    print("\n最佳模型: " + sorted_models[0][0] + f" ({sorted_models[0][1]['total_avg_score']:.2f} 分)")
    print("=" * 70)


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark 汇总脚本")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="exp/results",
        help="结果文件目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出报告路径 (默认保存到 results-dir)"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="从 checkpoint 文件计算统计 (如果没有 model_stats 文件)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Benchmark 汇总")
    print("=" * 60)
    print(f"结果目录: {args.results_dir}")

    if not os.path.exists(args.results_dir):
        print(f"  [Error] 目录不存在: {args.results_dir}")
        return

    # 读取结果
    print("\n[1/2] 读取模型结果...")
    all_stats = load_model_stats(args.results_dir)

    # 如果没有 model_stats 文件，尝试从 checkpoint 计算
    if not all_stats and args.use_checkpoint:
        print("\n  未找到 model_stats 文件，尝试从 checkpoint 计算...")
        all_stats = load_checkpoint_stats(args.results_dir)

    if not all_stats:
        print("\n  [Error] 未找到任何模型结果")
        return

    print(f"\n  共 {len(all_stats)} 个模型")

    # 生成报告
    print("\n[2/2] 生成汇总报告...")

    # 控台摘要
    generate_console_summary(all_stats)

    # Markdown 报告
    report_content = generate_markdown_report(all_stats, args.results_dir)

    # 确定输出路径
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.results_dir, f"benchmark_summary_{timestamp}.md")

    # 保存报告
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else args.results_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n  报告已保存: {output_file}")

    # 保存 JSON 汇总
    summary_json = os.path.join(args.results_dir, "all_models_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print(f"  JSON 汇总: {summary_json}")
    print("\n" + "=" * 60)
    print("汇总完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()