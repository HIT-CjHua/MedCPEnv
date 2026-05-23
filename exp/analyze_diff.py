"""
诊断-治疗分数分布差异分析

1. 抽样案例分析：每个模型抽 1 条"诊断不过但治疗过了"的典型 case
2. 分数组合统计：(诊断分, 治疗分) 的联合分布
"""
import json
import os
from collections import Counter

RESULTS_DIR = "exp/results"
DATA_PATH = "exp/data/benchmark_1000.jsonl"

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


def load_benchmark():
    """加载 benchmark 原始数据"""
    data = {}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                data[entry["case_id"]] = entry
    return data


def load_rejudge(model_name):
    path = os.path.join(RESULTS_DIR, f"rejudge_checkpoint_{model_name}.jsonl")
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


# =====================================================================
# Part 1: 分数组合联合分布
# =====================================================================
def analyze_score_combinations():
    """统计 (诊断分, 治疗分) 的联合分布"""
    print("=" * 70)
    print("1. 诊断-治疗分数联合分布 (所有模型合并)")
    print("=" * 70)

    all_combos = Counter()
    model_combos = {}

    for model_name in ALL_MODELS:
        entries = load_rejudge(model_name)
        if not entries:
            continue

        valid = [e for e in entries if e.get("total_score", 0) > 0]
        combos = Counter()
        for e in valid:
            d = round(e["diagnosis_score"])
            t = round(e["treatment_score"])
            combo = (d, t)
            combos[combo] += 1
            all_combos[combo] += 1

        model_combos[model_name] = combos

    # 打印矩阵 (诊断分=行, 治疗分=列)
    print(f"\n{'诊断\\治疗':>12} | {'1分':>4} {'2分':>4} {'3分':>4} {'4分':>4} {'5分':>4} | 合计")
    print("-" * 50)
    for d_score in [1, 2, 3, 4, 5]:
        row = [all_combos.get((d_score, t), 0) for t in [1, 2, 3, 4, 5]]
        row_total = sum(row)
        print(f"  {d_score} 分       | {row[0]:>4} {row[1]:>4} {row[2]:>4} {row[3]:>4} {row[4]:>4} | {row_total:>4}")
    print("-" * 50)
    col_totals = [sum(all_combos.get((d, t), 0) for d in [1,2,3,4,5]) for t in [1,2,3,4,5]]
    grand = sum(all_combos.values())
    print(f"  合计       | {col_totals[0]:>4} {col_totals[1]:>4} {col_totals[2]:>4} {col_totals[3]:>4} {col_totals[4]:>4} | {grand:>4}")

    # 各模型的分布矩阵
    print(f"\n\n各模型详细分布:")
    for model_name, combos in model_combos.items():
        print(f"\n  {model_name} (n={sum(combos.values())})")
        print(f"  {'诊断\\治疗':>12} | {'1':>3} {'2':>3} {'3':>3} {'4':>3} {'5':>3} | 合计")
        print(f"  {'-'*38}")
        for d_score in [1, 2, 3, 4, 5]:
            row = [combos.get((d_score, t), 0) for t in [1, 2, 3, 4, 5]]
            row_total = sum(row)
            if row_total > 0:
                print(f"    {d_score} 分       | {row[0]:>3} {row[1]:>3} {row[2]:>3} {row[3]:>3} {row[4]:>3} | {row_total:>3}")


# =====================================================================
# Part 2: 典型案例抽样
# =====================================================================
def sample_cases():
    """每个模型抽样 1 条典型 case"""
    print("\n" + "=" * 70)
    print("2. 典型案例分析 (每模型抽样)")
    print("=" * 70)

    benchmark = load_benchmark()

    for model_name in ALL_MODELS:
        entries = load_rejudge(model_name)
        if not entries:
            continue

        valid = [e for e in entries if e.get("total_score", 0) > 0]

        # 抽样优先级:
        # 1. 诊断不过但治疗过了 (0, 4) 或 (0, 5)
        # 2. 诊断过但治疗不过 (1, 0-2)
        # 3. 诊断 low 治疗 high (1-2, 4-5)
        # 4. 诊断 high 治疗 low (4-5, 1-2)
        samples = []

        # 类型 A: 诊断 incorrect, 治疗 correct, 治疗分 >= 4
        type_a = [e for e in valid if not e["diagnosis_correct"] and e["treatment_correct"] and e["treatment_score"] >= 4]
        # 类型 B: 诊断 correct, 治疗 incorrect, 治疗分 <= 2
        type_b = [e for e in valid if e["diagnosis_correct"] and not e["treatment_correct"]]
        # 类型 C: 诊断分 <= 2, 治疗分 >= 4
        type_c = [e for e in valid if e["diagnosis_score"] <= 2 and e["treatment_score"] >= 4]
        # 类型 D: 诊断分 >= 4, 治疗分 <= 2
        type_d = [e for e in valid if e["diagnosis_score"] >= 4 and e["treatment_score"] <= 2]

        # 优先选 A, 其次 C, 再 B, 最后 D
        if type_a:
            samples.append(("A: 诊断不过但治疗过了(分>=4)", type_a[0]))
        elif type_c:
            samples.append(("C: 诊断分<=2但治疗分>=4", type_c[0]))
        elif type_b:
            samples.append(("B: 诊断过了但治疗不过", type_b[0]))
        elif type_d:
            samples.append(("D: 诊断分>=4但治疗分<=2", type_d[0]))
        else:
            # 选最异常的
            worst = max(valid, key=lambda e: abs(e["diagnosis_score"] - e["treatment_score"]))
            samples.append(("差异最大", worst))

        for label, entry in samples:
            case_id = entry["case_id"]
            gt_data = benchmark.get(case_id, {})
            chief_complaint = gt_data.get("chief_complaint", entry.get("trajectory", [{}])[0].get("observation", "")[:100] if entry.get("trajectory") else "")

            print(f"\n{'='*70}")
            print(f"模型: {model_name} | 类型: {label}")
            print(f"Case ID: {case_id}")
            print(f"主诉: {chief_complaint}")
            print(f"")
            print(f"  诊断: correct={entry['diagnosis_correct']}, score={entry['diagnosis_score']}")
            print(f"    原因: {entry.get('diagnosis_reason', '')[:150]}")
            print(f"  治疗: correct={entry['treatment_correct']}, score={entry['treatment_score']}")
            print(f"    原因: {entry.get('treatment_reason', '')[:150]}")
            print(f"  安全: violated={entry['avoid_violated']}, violations={entry.get('avoid_violations', [])}")
            print(f"")

            # Ground truth
            gt = entry.get("ground_truth", gt_data)
            gt_diagnosis = gt.get("diagnosis", [])
            gt_treatment = gt.get("treatment", [])
            gt_avoid = gt.get("avoid", [])
            print(f"  Ground Truth:")
            print(f"    诊断: {gt_diagnosis}")
            print(f"    治疗: {gt_treatment}")
            print(f"    禁忌: {gt_avoid}")

            # Agent 输出
            agent_diag = entry.get("agent_diagnosis", "")
            agent_treat = entry.get("agent_treatment", "")
            print(f"  Agent 输出:")
            print(f"    诊断: {agent_diag[:200]}")
            print(f"    治疗: {agent_treat[:200]}")

            # 检查 substring 匹配
            diag_match = any(d.lower() in agent_diag.lower() for d in gt_diagnosis if d) if gt_diagnosis else False
            treat_match = any(t.lower() in agent_treat.lower() for t in gt_treatment if t) if gt_treatment else False
            print(f"  Substring 匹配: 诊断={'Y' if diag_match else 'N'}, 治疗={'Y' if treat_match else 'N'}")


# =====================================================================
# Part 3: 分数差值分布
# =====================================================================
def analyze_score_diff():
    """诊断-治疗分差分布"""
    print("\n" + "=" * 70)
    print("3. 诊断-治疗分差分布 (诊断分 - 治疗分)")
    print("=" * 70)

    print(f"\n{'Model':<25} | 平均差 | 诊断>治疗 | 诊断=治疗 | 诊断<治疗")
    print("-" * 70)

    for model_name in ALL_MODELS:
        entries = load_rejudge(model_name)
        if not entries:
            continue

        valid = [e for e in entries if e.get("total_score", 0) > 0]
        diffs = [e["diagnosis_score"] - e["treatment_score"] for e in valid]
        avg_diff = sum(diffs) / len(diffs)

        d_gt = sum(1 for d in diffs if d > 0)
        d_eq = sum(1 for d in diffs if d == 0)
        d_lt = sum(1 for d in diffs if d < 0)

        print(f"  {model_name:<23} | {avg_diff:+.2f}  | {d_gt:>3} ({d_gt/len(valid)*100:.0f}%) | {d_eq:>3} ({d_eq/len(valid)*100:.0f}%) | {d_lt:>3} ({d_lt/len(valid)*100:.0f}%)")


if __name__ == "__main__":
    analyze_score_combinations()
    analyze_score_diff()
    sample_cases()
