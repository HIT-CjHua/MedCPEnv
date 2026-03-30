#!/usr/bin/env python3
"""
数据合成 Pipeline 优化对比测试

对比两组配置:
- 配置A (当前): Review 4模型 + Judge 4模型 (10次调用)
- 配置B (优化): Review 1模型 + Judge 3模型 (6次调用)

在小规模数据上验证优化后质量是否可接受
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import MedicalCase
from src.utils import extract_tag_content

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
SEED_DATA_PATH = "/home/jiazixiao.jzx/MedAgent/data/seed_dataset/format_data.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "output" / "pipeline_comparison"

# 测试样本数
N_SAMPLES = 30

# 配置A: 当前配置
CONFIG_A = {
    "name": "当前配置(10次调用)",
    "generate_model": "qwen3.5-plus",
    "review_models": ["qwen3.5-plus", "glm-5", "kimi-k2.5", "MiniMax-M2.5"],
    "rewrite_model": "qwen3.5-plus",
    "judge_models": ["qwen3.5-plus", "glm-5", "kimi-k2.5", "MiniMax-M2.5"],
    "min_passing_judges": 4,
}

# 配置B: 优化配置
CONFIG_B = {
    "name": "优化配置(6次调用)",
    "generate_model": "qwen3.5-plus",
    "review_models": ["glm-5"],  # 只用异构模型
    "rewrite_model": "qwen3.5-plus",
    "judge_models": ["qwen3.5-plus", "glm-5", "MiniMax-M2.5"],  # 去掉kimi
    "min_passing_judges": 3,  # 降低阈值
}

BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
API_KEY = os.getenv("DASHSCOPE_API_KEY_CP")

# -----------------------------------------------------------------------------
# Prompts (简化版)
# -----------------------------------------------------------------------------

GENERATE_PROMPT = """你是一位专业的医疗数据合成专家。请根据种子信息生成一条完整的病例数据。

输出一个JSON，包含以下字段:
- case_id: SYN_xxx
- difficulty: easy/medium/hard
- tags: ["科室", "疾病类型"]
- chief_complaint: 主诉
- subjective: [{keywords: [], content: "", necessity: bool}] 至少2项
- objective: [{keywords: [], content: "", necessity: bool}] 至少2项
- ground_truth: {diagnosis: [], treatment: [], avoid: []}
- source: "synthetic"

把JSON放在<result></result>标签中。"""

REVIEW_PROMPT = """审核病例数据质量，检查:
1. 结构完整性
2. 主观/客观信息是否充分
3. 诊断治疗是否合理

输出JSON: {"pass": true/false, "issues": [], "suggestions": []}
放在<review></review>标签中。"""

REWRITE_PROMPT = """根据review意见修订病例数据。
输出完整修订后的JSON，放在<rewrite></rewrite>标签中。"""

JUDGE_PROMPT = """判断病例是否可用于评测。
输出JSON: {"pass": true/false, "reason": ""}
放在<judge></judge>标签中。"""


def make_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_llm(prompt: str, model: str) -> str:
    client = make_client()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content or ""
    finally:
        try:
            client.close()
        except:
            pass


def parse_tagged_json(text: str, tag: str) -> Dict:
    content = extract_tag_content(text, tag)
    if content:
        return json.loads(content)
    raise ValueError(f"未找到标签 {tag}")


def validate_schema(data: Dict) -> Tuple[bool, str]:
    """简化的schema校验"""
    required = ["case_id", "chief_complaint", "subjective", "objective", "ground_truth"]
    for f in required:
        if f not in data:
            return False, f"缺少字段: {f}"
    for side in ["subjective", "objective"]:
        if not isinstance(data.get(side), list) or len(data[side]) < 2:
            return False, f"{side}需至少2项"
    gt = data.get("ground_truth", {})
    for f in ["diagnosis", "treatment", "avoid"]:
        if f not in gt:
            return False, f"ground_truth缺少{f}"
    return True, "ok"


def process_one_sample(seed: Dict, config: Dict) -> Dict:
    """处理单条样本"""
    seed_id = seed.get("id", "unknown")
    result = {
        "seed_id": seed_id,
        "config_name": config["name"],
        "status": "failed",
        "calls": 0,
        "final_case": None,
        "error": None,
    }

    try:
        # 1. Generate
        gen_prompt = f"{GENERATE_PROMPT}\n\n种子: {json.dumps(seed, ensure_ascii=False)}"
        gen_resp = call_llm(gen_prompt, config["generate_model"])
        result["calls"] += 1
        case_data = parse_tagged_json(gen_resp, "result")
        case_data["source"] = "synthetic"

        ok, msg = validate_schema(case_data)
        if not ok:
            result["error"] = f"generate schema: {msg}"
            return result

        # 2. Review (多模型)
        all_issues = []
        all_suggestions = []
        for model in config["review_models"]:
            rev_prompt = f"{REVIEW_PROMPT}\n\n病例: {json.dumps(case_data, ensure_ascii=False)}"
            rev_resp = call_llm(rev_prompt, model)
            result["calls"] += 1
            try:
                rev_data = parse_tagged_json(rev_resp, "review")
                all_issues.extend(rev_data.get("issues", []))
                all_suggestions.extend(rev_data.get("suggestions", []))
            except:
                pass

        # 3. Rewrite
        rewrite_prompt = f"{REWRITE_PROMPT}\n\n原始病例: {json.dumps(case_data, ensure_ascii=False)}\n\n问题: {all_issues}\n建议: {all_suggestions}"
        rewrite_resp = call_llm(rewrite_prompt, config["rewrite_model"])
        result["calls"] += 1
        rewritten = parse_tagged_json(rewrite_resp, "rewrite")
        rewritten["source"] = "synthetic"

        ok, msg = validate_schema(rewritten)
        if not ok:
            result["error"] = f"rewrite schema: {msg}"
            return result

        # 4. Judge (多模型)
        passing = 0
        for model in config["judge_models"]:
            judge_prompt = f"{JUDGE_PROMPT}\n\n病例: {json.dumps(rewritten, ensure_ascii=False)}"
            judge_resp = call_llm(judge_prompt, model)
            result["calls"] += 1
            try:
                judge_data = parse_tagged_json(judge_resp, "judge")
                if judge_data.get("pass"):
                    passing += 1
            except:
                pass

        result["final_case"] = rewritten
        result["judge_passing"] = passing
        result["judge_total"] = len(config["judge_models"])

        if passing >= config["min_passing_judges"]:
            result["status"] = "passed"
        else:
            result["error"] = f"judge未通过: {passing}/{len(config['judge_models'])}"

    except Exception as e:
        result["error"] = str(e)

    return result


def run_comparison():
    """运行对比测试"""
    print("=" * 60)
    print("Pipeline 优化对比测试")
    print("=" * 60)

    # 加载种子数据
    with open(SEED_DATA_PATH, "r", encoding="utf-8") as f:
        all_seeds = [json.loads(line) for line in f if line.strip()]

    # 随机抽样
    random.seed(42)
    seeds = random.sample(all_seeds, N_SAMPLES)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {"config_a": [], "config_b": []}

    # 并发运行配置A
    print(f"\n[配置A] {CONFIG_A['name']}")
    print(f"  Review模型: {CONFIG_A['review_models']}")
    print(f"  Judge模型: {CONFIG_A['judge_models']}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_one_sample, seed, CONFIG_A): seed for seed in seeds}
        for future in tqdm(as_completed(futures), total=len(futures), desc="配置A"):
            r = future.result()
            results["config_a"].append(r)

    # 并发运行配置B
    print(f"\n[配置B] {CONFIG_B['name']}")
    print(f"  Review模型: {CONFIG_B['review_models']}")
    print(f"  Judge模型: {CONFIG_B['judge_models']}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_one_sample, seed, CONFIG_B): seed for seed in seeds}
        for future in tqdm(as_completed(futures), total=len(futures), desc="配置B"):
            r = future.result()
            results["config_b"].append(r)

    # 统计对比
    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)

    for name, res in [("配置A", results["config_a"]), ("配置B", results["config_b"])]:
        passed = sum(1 for r in res if r["status"] == "passed")
        total_calls = sum(r["calls"] for r in res)
        avg_calls = total_calls / len(res) if res else 0

        print(f"\n{name}:")
        print(f"  通过率: {passed}/{len(res)} ({passed/len(res)*100:.1f}%)")
        print(f"  总调用: {total_calls}")
        print(f"  平均调用: {avg_calls:.1f} 次/样本")

    # 保存结果
    output_file = OUTPUT_DIR / "comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_file}")

    # 保存通过样本
    for config_name, config_key in [("A", "config_a"), ("B", "config_b")]:
        passed_cases = [r["final_case"] for r in results[config_key] if r["status"] == "passed" and r["final_case"]]
        out_file = OUTPUT_DIR / f"passed_config_{config_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for case in passed_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"配置{config_name}通过样本: {out_file}")

    return results


if __name__ == "__main__":
    run_comparison()