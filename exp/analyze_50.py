"""
50条 pilot 数据分析脚本
"""
import json
import os
from collections import defaultdict

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


def load_rejudge(model_name):
    path = os.path.join(RESULTS_DIR, f"rejudge_checkpoint_{model_name}.jsonl")
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def analyze():
    all_data = {}
    for m in ALL_MODELS:
        entries = load_rejudge(m)
        if not entries:
            print(f"[Skip] {m}: 无数据")
            continue
        all_data[m] = entries

    if not all_data:
        print("无数据可分析")
        return

    # ============================================================
    # 1. 各维度分数分布直方图
    # ============================================================
    print("=" * 70)
    print("1. 各维度分数分布 (1-5分)")
    print("=" * 70)

    for model_name, entries in all_data.items():
        valid = [e for e in entries if e.get("total_score", 0) > 0]
        if not valid:
            continue

        diag_scores = [e["diagnosis_score"] for e in valid]
        treat_scores = [e["treatment_score"] for e in valid]
        safe_scores = [e["avoid_score"] for e in valid]

        def hist(scores, bins=5):
            counts = [0] * bins
            for s in scores:
                idx = min(int(s), bins - 1)
                if idx < 0: idx = 0
                counts[idx] += 1
            return counts

        d_h = hist(diag_scores)
        t_h = hist(treat_scores)
        s_h = hist(safe_scores)

        avg_d = sum(diag_scores) / len(diag_scores)
        avg_t = sum(treat_scores) / len(treat_scores)
        avg_s = sum(safe_scores) / len(safe_scores)

        print(f"\n{model_name} (n={len(valid)})")
        print(f"  诊断: [{', '.join(str(c) for c in d_h)}]  avg={avg_d:.2f}")
        print(f"  治疗: [{', '.join(str(c) for c in t_h)}]  avg={avg_t:.2f}")
        print(f"  安全: [{', '.join(str(c) for c in s_h)}]  avg={avg_s:.2f}")

    # ============================================================
    # 2. 诊断-治疗相关性
    # ============================================================
    print("\n" + "=" * 70)
    print("2. 诊断-治疗分数相关性 (Pearson)")
    print("=" * 70)

    import statistics

    for model_name, entries in all_data.items():
        valid = [e for e in entries if e.get("total_score", 0) > 0]
        if len(valid) < 3:
            continue

        ds = [e["diagnosis_score"] for e in valid]
        ts = [e["treatment_score"] for e in valid]

        # Pearson correlation
        n = len(ds)
        mean_d = sum(ds) / n
        mean_t = sum(ts) / n

        cov = sum((d - mean_d) * (t - mean_t) for d, t in zip(ds, ts)) / n
        std_d = (sum((d - mean_d)**2 for d in ds) / n) ** 0.5
        std_t = (sum((t - mean_t)**2 for t in ts) / n) ** 0.5

        if std_d > 0 and std_t > 0:
            r = cov / (std_d * std_t)
        else:
            r = 0

        # 同时 pass / fail 的比例
        both_pass = sum(1 for e in valid if e["diagnosis_correct"] and e["treatment_correct"])
        both_fail = sum(1 for e in valid if not e["diagnosis_correct"] and not e["treatment_correct"])
        diag_only = sum(1 for e in valid if e["diagnosis_correct"] and not e["treatment_correct"])
        treat_only = sum(1 for e in valid if not e["diagnosis_correct"] and e["treatment_correct"])

        print(f"  {model_name}: r={r:.3f} | 双过={both_pass} 双不过={both_fail} 仅诊断={diag_only} 仅治疗={treat_only}")

    # ============================================================
    # 3. 诊断正确 vs 错误的治疗分数差异
    # ============================================================
    print("\n" + "=" * 70)
    print("3. 诊断正确/错误的治疗分数差异")
    print("=" * 70)

    for model_name, entries in all_data.items():
        valid = [e for e in entries if e.get("total_score", 0) > 0]

        diag_pass_t = [e["treatment_score"] for e in valid if e["diagnosis_correct"]]
        diag_fail_t = [e["treatment_score"] for e in valid if not e["diagnosis_correct"]]

        avg_tp = sum(diag_pass_t) / len(diag_pass_t) if diag_pass_t else 0
        avg_tf = sum(diag_fail_t) / len(diag_fail_t) if diag_fail_t else 0

        print(f"  {model_name}: 诊断过→治疗分={avg_tp:.2f}(n={len(diag_pass_t)}) | 诊断不过→治疗分={avg_tf:.2f}(n={len(diag_fail_t)}) | 差={avg_tp-avg_tf:.2f}")

    # ============================================================
    # 4. 安全违反详情
    # ============================================================
    print("\n" + "=" * 70)
    print("4. 安全违反汇总")
    print("=" * 70)

    for model_name, entries in all_data.items():
        valid = [e for e in entries if e.get("total_score", 0) > 0]
        violated = [e for e in valid if e.get("avoid_violated", False)]

        if violated:
            violations = []
            for e in violated:
                violations.extend(e.get("avoid_violations", []))

            # 统计最常见的违反项
            from collections import Counter
            viol_counts = Counter(violations)

            print(f"\n  {model_name}: {len(violated)}/{len(valid)} 条有违反 ({len(violated)/len(valid)*100:.0f}%)")
            for item, cnt in viol_counts.most_common(5):
                print(f"    - {item}: {cnt}次")
        else:
            print(f"  {model_name}: 0/{len(valid)} 违反")

    # ============================================================
    # 5. 各模型综合排名对比
    # ============================================================
    print("\n" + "=" * 70)
    print("5. 综合排名表")
    print("=" * 70)

    rows = []
    for model_name, entries in all_data.items():
        valid = [e for e in entries if e.get("total_score", 0) > 0]
        if not valid:
            continue

        n = len(valid)
        avg_d = sum(e["diagnosis_score"] for e in valid) / n
        avg_t = sum(e["treatment_score"] for e in valid) / n
        avg_s = sum(e["avoid_score"] for e in valid) / n
        total = (avg_d + avg_t + avg_s) / 3

        diag_acc = sum(1 for e in valid if e["diagnosis_correct"]) / n
        treat_acc = sum(1 for e in valid if e["treatment_correct"]) / n
        safe_viol = sum(1 for e in valid if e["avoid_violated"]) / n

        excellent = sum(1 for e in valid if e["total_score"] >= 4)
        good = sum(1 for e in valid if 3 <= e["total_score"] < 4)
        medium = sum(1 for e in valid if 2 <= e["total_score"] < 3)
        poor = sum(1 for e in valid if e["total_score"] < 2)

        rows.append({
            "model": model_name,
            "n": n,
            "diag_acc": diag_acc,
            "avg_d": avg_d,
            "treat_acc": treat_acc,
            "avg_t": avg_t,
            "safe_viol": safe_viol,
            "avg_s": avg_s,
            "total": total,
            "excellent": excellent,
            "good": good,
            "medium": medium,
            "poor": poor,
        })

    rows.sort(key=lambda x: x["total"], reverse=True)

    print(f"\n{'Rank':<5} {'Model':<25} {'D-Acc':>6} {'D-Avg':>5} {'T-Acc':>6} {'T-Avg':>5} {'S-Viol':>6} {'S-Avg':>5} {'Total':>5} {'E':>3} {'G':>3} {'M':>3} {'P':>3}")
    print("-" * 110)
    for i, r in enumerate(rows, 1):
        print(f"{i:<5} {r['model']:<25} {r['diag_acc']*100:>5.1f}% {r['avg_d']:>5.2f} {r['treat_acc']*100:>5.1f}% {r['avg_t']:>5.2f} {r['safe_viol']*100:>5.1f}% {r['avg_s']:>5.2f} {r['total']:>5.2f} {r['excellent']:>3} {r['good']:>3} {r['medium']:>3} {r['poor']:>3}")

    # ============================================================
    # 6. Judger 打分一致性检查
    # ============================================================
    print("\n" + "=" * 70)
    print("6. 0-1判定与1-5分数的一致性")
    print("=" * 70)

    for model_name, entries in all_data.items():
        valid = [e for e in entries if e.get("total_score", 0) > 0]

        # diagnosis: correct 但 score < 3 的异常案例
        diag_anomaly = [e for e in valid if e["diagnosis_correct"] and e["diagnosis_score"] < 3]
        diag_anomaly2 = [e for e in valid if not e["diagnosis_correct"] and e["diagnosis_score"] >= 3]

        treat_anomaly = [e for e in valid if e["treatment_correct"] and e["treatment_score"] < 3]
        treat_anomaly2 = [e for e in valid if not e["treatment_correct"] and e["treatment_score"] >= 3]

        safe_anomaly = [e for e in valid if not e["avoid_violated"] and e["avoid_score"] < 4]
        safe_anomaly2 = [e for e in valid if e["avoid_violated"] and e["avoid_score"] >= 4]

        total_anomaly = len(diag_anomaly) + len(diag_anomaly2) + len(treat_anomaly) + len(treat_anomaly2) + len(safe_anomaly) + len(safe_anomaly2)

        if total_anomaly > 0:
            print(f"  {model_name}: {total_anomaly} 个不一致 ({total_anomaly/(len(valid)*3)*100:.1f}%)")
            if diag_anomaly:
                print(f"    诊断correct但score<3: {len(diag_anomaly)}例")
            if diag_anomaly2:
                print(f"    诊断incorrect但score>=3: {len(diag_anomaly2)}例")
            if treat_anomaly:
                print(f"    治疗correct但score<3: {len(treat_anomaly)}例")
            if treat_anomaly2:
                print(f"    治疗incorrect但score>=3: {len(treat_anomaly2)}例")
            if safe_anomaly:
                print(f"    安全无违反但score<4: {len(safe_anomaly)}例")
            if safe_anomaly2:
                print(f"    安全有违反但score>=4: {len(safe_anomaly2)}例")
        else:
            print(f"  {model_name}: 无不一致")


if __name__ == "__main__":
    analyze()
