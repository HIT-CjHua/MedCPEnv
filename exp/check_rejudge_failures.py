"""检查 rejudge 结果中是否有 Judger API 调用失败导致只有兜底分数的记录"""
import json
from pathlib import Path

RESULTS_DIR = Path("exp/results")

models = [
    "qwen3.5-plus",
    "qwen3.6-plus",
    "qwen3.max-2026-01-23",
    "glm-5",
    "kimi-k2.5",
    "MiniMax-M2.5",
    "deepseek-v3.2",
    "qwen3.5-35b-a3b",
    "gpt-5.4",
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
]

for model in models:
    fpath = RESULTS_DIR / f"rejudge_checkpoint_{model}.jsonl"
    if not fpath.exists():
        print(f"{model}: 文件不存在")
        continue

    total = 0
    failed = 0
    failed_ids = []

    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            rec = json.loads(line)
            rejudge = rec.get("rejudge", rec)
            d_score = rejudge.get("diagnosis_score", -1)
            t_score = rejudge.get("treatment_score", -1)
            s_score = rejudge.get("avoid_score", -1)
            error = rejudge.get("error", "")

            # 兜底最低分: 诊断和治疗都是 1.0, 安全是 0 或者全是 0
            is_fallback = (
                (d_score == 1.0 and t_score == 1.0)
                or error
            )
            if is_fallback:
                failed += 1
                failed_ids.append(rec.get("case_id", total))

    rate = failed / total * 100 if total else 0
    print(f"{model:30s} | total={total:5d} | failed={failed:5d} ({rate:5.1f}%)")
    if failed > 0 and failed <= 5:
        print(f"  -> failed case_ids: {failed_ids}")
    elif failed > 5:
        print(f"  -> first 5 failed case_ids: {failed_ids[:5]}")
