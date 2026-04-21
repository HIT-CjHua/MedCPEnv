"""
数据集分割脚本

将 passed_cases.jsonl 分割为 train/eval/test + benchmark 子集。

分割方案:
- Train: 70% (~33,600) - Agent 训练
- Eval: 15% (~7,200) - 训练验证
- Test: 15% (~7,200) - 包含 Benchmark 子集
- Benchmark: 1000条 - 从 Test 中分层采样，用于多模型对比

分层采样确保难度平衡 (easy/medium/hard)。

Usage:
    python exp/split_dataset.py
"""

import os
import json
import random
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))


def load_data(filepath: str) -> List[Dict]:
    """加载数据"""
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    case = json.loads(line)
                    if "case_id" in case and "chief_complaint" in case:
                        cases.append(case)
                except:
                    continue
    return cases


def analyze_distribution(cases: List[Dict]) -> Dict:
    """分析数据分布"""
    difficulty_count = Counter()
    department_count = Counter()
    source_count = Counter()

    for case in cases:
        difficulty = case.get("difficulty", "unknown")
        difficulty_count[difficulty] += 1

        tags = case.get("tags", [])
        if tags:
            dept = tags[0] if tags else "unknown"
            department_count[dept] += 1

        source = case.get("source", "unknown")
        source_count[source] += 1

    return {
        "total": len(cases),
        "difficulty": dict(difficulty_count),
        "department": dict(department_count),
        "source": dict(source_count),
    }


def print_distribution(name: str, dist: Dict):
    """打印分布信息"""
    print(f"\n{name}:")
    print(f"  总数: {dist['total']}")
    print(f"  难度分布:")
    for diff in ["easy", "medium", "hard"]:
        count = dist['difficulty'].get(diff, 0)
        pct = count / dist['total'] * 100 if dist['total'] > 0 else 0
        print(f"    {diff}: {count} ({pct:.1f}%)")


def split_by_difficulty(cases: List[Dict]) -> Dict[str, List[Dict]]:
    """按难度分组"""
    by_diff = {}
    for case in cases:
        diff = case.get("difficulty", "medium")
        if diff not in by_diff:
            by_diff[diff] = []
        by_diff[diff].append(case)
    return by_diff


def stratified_split(
    cases: List[Dict],
    train_ratio: float = 0.70,
    eval_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    分层分割数据集

    确保 train/eval/test 中难度分布一致
    """
    random.seed(seed)

    # 按难度分组
    by_diff = split_by_difficulty(cases)

    train_set, eval_set, test_set = [], [], []

    for diff, diff_cases in by_diff.items():
        # 打乱每组内的顺序
        random.shuffle(diff_cases)

        n = len(diff_cases)
        n_train = int(n * train_ratio)
        n_eval = int(n * eval_ratio)
        n_test = n - n_train - n_eval  # 剩余给 test

        train_set.extend(diff_cases[:n_train])
        eval_set.extend(diff_cases[n_train:n_train + n_eval])
        test_set.extend(diff_cases[n_train + n_eval:])

    # 再次打乱各集合内部顺序
    random.shuffle(train_set)
    random.shuffle(eval_set)
    random.shuffle(test_set)

    return train_set, eval_set, test_set


def stratified_sample(cases: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    """
    分层采样

    保持难度比例，从 test 中采样 benchmark
    """
    random.seed(seed + 1000)  # 不同种子

    # 按难度分组
    by_diff = split_by_difficulty(cases)

    # 计算原始难度比例
    total = len(cases)
    ratios = {}
    for diff, diff_cases in by_diff.items():
        ratios[diff] = len(diff_cases) / total

    # 按比例采样
    sampled = []
    remaining_cases = {diff: list(diff_cases) for diff, diff_cases in by_diff.items()}

    for diff, ratio in ratios.items():
        target = int(n * ratio)
        available = remaining_cases.get(diff, [])

        if len(available) >= target:
            selected = random.sample(available, target)
            sampled.extend(selected)
            # 从可用列表移除已选
            for s in selected:
                remaining_cases[diff].remove(s)
        else:
            sampled.extend(available)
            remaining_cases[diff] = []

    # 补齐到目标数量
    while len(sampled) < n:
        # 从任意难度补充
        for diff in ["medium", "easy", "hard"]:
            if remaining_cases.get(diff):
                sampled.append(remaining_cases[diff].pop())
                if len(sampled) >= n:
                    break

    random.shuffle(sampled)
    return sampled[:n]


def save_dataset(cases: List[Dict], filepath: str):
    """保存数据集"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"  已保存: {filepath} ({len(cases)} 条)")


def main():
    parser = argparse.ArgumentParser(description="数据集分割")
    parser.add_argument(
        "--input",
        type=str,
        default="output/m2_mimic/passed_cases.jsonl",
        help="输入数据路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/data",
        help="输出目录"
    )
    parser.add_argument(
        "--benchmark-size",
        type=int,
        default=1000,
        help="Benchmark 集大小"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细分布"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MedAgent 数据集分割")
    print("=" * 60)
    print(f"输入: {args.input}")
    print(f"输出目录: {args.output_dir}")
    print(f"Benchmark 大小: {args.benchmark_size}")
    print(f"随机种子: {args.seed}")
    print(f"分割比例: Train 70% / Eval 15% / Test 15%")

    # 加载数据
    print("\n[1/5] 加载数据...")
    cases = load_data(args.input)
    print(f"  加载 {len(cases)} 条病例")

    if len(cases) == 0:
        print("  [Error] 未找到任何数据")
        return

    # 分析分布
    print("\n[2/5] 分析数据分布...")
    full_dist = analyze_distribution(cases)
    print_distribution("原始数据", full_dist)

    if args.verbose:
        print(f"\n  数据来源分布:")
        for source, count in sorted(full_dist['source'].items()):
            pct = count / full_dist['total'] * 100
            print(f"    {source}: {count} ({pct:.1f}%)")
        print(f"\n  科室分布 (前 10):")
        for dept, count in sorted(full_dist['department'].items(), key=lambda x: -x[1])[:10]:
            print(f"    {dept}: {count}")

    # 分层分割
    print("\n[3/5] 分层分割 (70/15/15)...")
    train_set, eval_set, test_set = stratified_split(
        cases,
        train_ratio=0.70,
        eval_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed
    )

    train_dist = analyze_distribution(train_set)
    eval_dist = analyze_distribution(eval_set)
    test_dist = analyze_distribution(test_set)

    print_distribution("Train 集", train_dist)
    print_distribution("Eval 集", eval_dist)
    print_distribution("Test 集", test_dist)

    # 从 Test 中采样 Benchmark
    print("\n[4/5] 分层采样 Benchmark...")
    benchmark_set = stratified_sample(test_set, args.benchmark_size, seed=args.seed)
    benchmark_dist = analyze_distribution(benchmark_set)
    print_distribution("Benchmark 集", benchmark_dist)

    # 验证 Benchmark 与 Test 的难度分布一致性
    print("\n  验证分布一致性:")
    for diff in ["easy", "medium", "hard"]:
        test_pct = test_dist['difficulty'].get(diff, 0) / test_dist['total'] * 100
        bench_pct = benchmark_dist['difficulty'].get(diff, 0) / benchmark_dist['total'] * 100
        diff_pct = abs(test_pct - bench_pct)
        print(f"    {diff}: Test {test_pct:.1f}% vs Benchmark {bench_pct:.1f}% (差异 {diff_pct:.1f}%)")

    # 保存数据集
    print("\n[5/5] 保存数据集...")
    output_dir = args.output_dir

    save_dataset(train_set, os.path.join(output_dir, "train.jsonl"))
    save_dataset(eval_set, os.path.join(output_dir, "eval.jsonl"))
    save_dataset(test_set, os.path.join(output_dir, "test.jsonl"))
    save_dataset(benchmark_set, os.path.join(output_dir, "benchmark_1000.jsonl"))

    # 保存统计信息
    stats = {
        "input_file": args.input,
        "seed": args.seed,
        "split_ratio": {"train": 0.70, "eval": 0.15, "test": 0.15},
        "benchmark_size": args.benchmark_size,
        "counts": {
            "total": len(cases),
            "train": len(train_set),
            "eval": len(eval_set),
            "test": len(test_set),
            "benchmark": len(benchmark_set),
        },
        "difficulty_distribution": {
            "full": full_dist['difficulty'],
            "train": train_dist['difficulty'],
            "eval": eval_dist['difficulty'],
            "test": test_dist['difficulty'],
            "benchmark": benchmark_dist['difficulty'],
        },
    }

    stats_file = os.path.join(output_dir, "split_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  统计信息: {stats_file}")

    print("\n" + "=" * 60)
    print("分割完成!")
    print("=" * 60)
    print(f"\n数据集位置: {output_dir}/")
    print(f"  - train.jsonl      ({len(train_set)} 条) - 用于 Agent 训练")
    print(f"  - eval.jsonl       ({len(eval_set)} 条) - 用于训练验证")
    print(f"  - test.jsonl       ({len(test_set)} 条) - 用于模型评估")
    print(f"  - benchmark_1000.jsonl ({len(benchmark_set)} 条) - 用于多模型对比测试")


if __name__ == "__main__":
    main()