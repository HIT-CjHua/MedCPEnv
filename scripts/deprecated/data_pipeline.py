# MedAgent/scripts/data_pipeline.py
"""
数据合成 Pipeline 脚本

功能：
1. 从种子数据集中按 label 尽量均匀抽取 N 条数据
2. 将抽取到的种子数据写出到 sampled_seeds.jsonl
3. 按 shard 组织任务，每个 shard 默认 500 条
4. 进程启动后每 5 小时启动一个 shard
5. 每条样本执行：
   - 单模型 generate
   - 多模型 review
   - 单模型 rewrite
   - 多模型 judge
6. 输出：
   - sampled_seeds.jsonl：均匀抽样后的种子数据
   - shard_xxxx/result.json：每条样本四步中间结果，JSON 更易读
   - shard_xxxx/selected.jsonl：仅该 shard 成功样本
   - shard_xxxx/stats.json：该 shard 统计
   - merged_selected.jsonl：所有 shard 成功样本合并
   - merged_stats.json：所有 shard 统计合并

说明：
- 使用 run_concurrent_task 进行并发处理
- 使用 extract_tag_content 提取模型在 tag 中返回的结果
- 当前 schema 为简化版：
  ground_truth 仅包含 diagnosis / treatment / avoid
"""

import os
import sys
import json
import time
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# -----------------------------------------------------------------------------
# 路径与导入
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/jiazixiao.jzx/MedAgent")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import MedicalCase
from src.utils import run_concurrent_task, extract_tag_content

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
SEED_DATA_PATH = "/home/jiazixiao.jzx/MedAgent/data/seed_dataset/format_data.jsonl"
OUTPUT_ROOT = Path("/home/jiazixiao.jzx/MedAgent/output")

# 总目标条数
N_SAMPLES = 2000

# 每个 shard 处理条数
SHARD_SIZE = 500

# run_concurrent_task 的并发
MAX_CONCURRENT_REQUESTS = 10

# 每个 shard 启动间隔
SHARD_INTERVAL_SECONDS = 0

# 优化配置: 减少40%调用，通过率更高
SYNTHESIS_MODEL = "qwen3.5-plus"
REWRITE_MODEL = "qwen3.5-plus"
REVIEW_MODELS = [
    "glm-5",  # 只用异构模型进行review
]
JUDGE_MODELS = [
    "qwen3.5-plus",
    "glm-5",
    "MiniMax-M2.5",  # 去掉kimi-k2.5
]

# 最终 judge 要求 3/3 全通过
MIN_PASSING_MODELS = 3

BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
API_KEY = os.getenv("DASHSCOPE_API_KEY_CP")

# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------

GENERATE_PROMPT = """你是一位专业的医疗数据合成专家。请根据用户提供的医疗问题，生成一条符合 MedicalCase schema 的病例数据。

【目标数据结构】
你必须输出一个完整 JSON，对应如下结构：
{
  "case_id": "SYN_xxx",
  "difficulty": "easy/medium/hard",
  "tags": ["科室", "疾病类型"],
  "chief_complaint": "主诉",
  "subjective": [
    {
      "keywords": ["关键词1", "关键词2"],
      "content": "患者主观描述",
      "necessity": true
    }
  ],
  "objective": [
    {
      "keywords": ["检查关键词1", "检查关键词2"],
      "content": "客观检查结果",
      "necessity": true
    }
  ],
  "ground_truth": {
    "diagnosis": ["诊断1"],
    "treatment": ["治疗方案1"],
    "avoid": ["禁忌1"]
  },
  "source": "synthetic"
}

【硬性要求】
1. subjective 和 objective 都必须是列表，且每个至少 2 项
2. 每个 item 必须包含：
   - keywords: 字符串数组
   - content: 非空字符串
   - necessity: 布尔值
3. ground_truth 必须包含：
   - diagnosis
   - treatment
   - avoid
4. 诊断、检查、治疗之间要医学上基本合理
5. 使用中文
6. 不要输出任何解释

【输出要求】
请把最终 JSON 严格放在 <result></result> 标签中。
"""

REVIEW_PROMPT = """你是一位资深临床医生，负责审核医疗病例数据质量。

请审核输入病例，重点检查：
1. 数据结构是否符合 MedicalCase schema
2. subjective 和 objective 是否各至少 2 项
3. 主诉、症状、检查、诊断、治疗之间是否存在明显冲突
4. 数据是否适合作为模型评测样本

请输出 JSON：
{
  "pass": true/false,
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"],
  "summary": "一句话总结"
}

【输出要求】
不要输出任何额外说明，只把最终 JSON 放在 <review></review> 标签中。
"""

REWRITE_PROMPT = """你是一位专业的医疗数据修订专家。请根据原始病例数据和 review 意见，对病例进行重写修正。

目标：
- 输出一条完整的修正后 MedicalCase JSON
- 修复 review 中指出的问题
- 保持结构符合 MedicalCase schema
- 保证内容医学上更合理

【硬性要求】
1. 输出完整 JSON，而不是 diff
2. subjective 和 objective 都必须至少 2 项
3. ground_truth 必须包含 diagnosis、treatment、avoid
4. source 必须为 "synthetic"
5. 使用中文
6. 不要输出解释性文字

【输出要求】
只把最终修正后的 JSON 放在 <rewrite></rewrite> 标签中。
"""

JUDGE_PROMPT = """你是一位资深临床专家，负责最终判断一条病例数据是否可用于模型评测。

评判标准：
1. 数据结构完整
2. subjective 和 objective 信息充分
3. 症状、检查、诊断、治疗之间无明显冲突
4. 样本具有一定评测价值

请输出 JSON：
{
  "pass": true/false,
  "reason": "简要原因"
}

【输出要求】
只把最终 JSON 放在 <judge></judge> 标签中。
"""

# -----------------------------------------------------------------------------
# 基础函数
# -----------------------------------------------------------------------------

def make_client() -> OpenAI:
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )


def load_all_seed_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def balanced_sample_by_label(data: List[Dict[str, Any]], n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    按 label 尽量均匀抽样。

    策略：
    1. 按 label 分组
    2. 每个 label 先分配基础配额 n // num_labels
    3. 余数尽量再均匀补
    4. 若某些 label 不足，从剩余池中补齐
    """
    rng = random.Random(seed)

    label_to_items = defaultdict(list)
    for item in data:
        label = item.get("label", "未知")
        label_to_items[label].append(item)

    labels = sorted(label_to_items.keys())
    if not labels:
        return []

    for label in labels:
        rng.shuffle(label_to_items[label])

    num_labels = len(labels)
    base_quota = n // num_labels
    remainder = n % num_labels

    selected = []
    leftovers = []

    # 第一轮：每类基础配额
    for label in labels:
        items = label_to_items[label]
        take = min(base_quota, len(items))
        selected.extend(items[:take])
        leftovers.extend(items[take:])

    # 第二轮：补余数，尽量一类一个
    if remainder > 0:
        labels_with_remaining = [label for label in labels if len(label_to_items[label]) > base_quota]
        rng.shuffle(labels_with_remaining)

        extra = []
        for label in labels_with_remaining:
            if len(extra) >= remainder:
                break
            items = label_to_items[label]
            idx = base_quota
            if idx < len(items):
                extra.append(items[idx])

        selected.extend(extra)
        selected_ids = {id(x) for x in selected}
        leftovers = [x for x in leftovers if id(x) not in selected_ids]

    # 第三轮：若仍不足，从剩余池随机补齐
    if len(selected) < n:
        need = n - len(selected)
        rng.shuffle(leftovers)
        selected.extend(leftovers[:need])

    # 最终截断
    if len(selected) > n:
        rng.shuffle(selected)
        selected = selected[:n]

    return selected


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_list(items: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def call_llm_with_system(
    prompt: str,
    model: str,
    system_prompt: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    client = make_client()
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model,
            "messages": messages,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        completion = client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content
    finally:
        try:
            client.close()
        except Exception:
            pass


def parse_tagged_json(text: str, tag_name: str) -> Dict[str, Any]:
    content = extract_tag_content(text, tag_name)
    if content is None:
        raise ValueError(f"未找到标签 <{tag_name}></{tag_name}>")
    return json.loads(content)


# -----------------------------------------------------------------------------
# Schema 校验
# -----------------------------------------------------------------------------

def validate_medical_case_dict(data: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    校验 dict 是否符合简化后的 MedicalCase schema，并归一化
    """
    try:
        required_fields = [
            "case_id", "difficulty", "tags", "chief_complaint",
            "subjective", "objective", "ground_truth", "source"
        ]
        for field in required_fields:
            if field not in data:
                return False, f"缺少字段: {field}", None

        if data["difficulty"] not in ["easy", "medium", "hard"]:
            return False, f"difficulty 非法: {data['difficulty']}", None

        if data["source"] != "synthetic":
            return False, f"source 必须为 synthetic，当前为: {data['source']}", None

        if not isinstance(data["tags"], list) or len(data["tags"]) == 0:
            return False, "tags 必须为非空 list", None

        if not isinstance(data["chief_complaint"], str) or not data["chief_complaint"].strip():
            return False, "chief_complaint 必须为非空字符串", None

        for side in ["subjective", "objective"]:
            items = data.get(side)
            if not isinstance(items, list) or len(items) < 2:
                return False, f"{side} 必须是至少包含 2 项的 list", None

            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    return False, f"{side}[{i}] 必须是 dict", None
                if "keywords" not in item or "content" not in item or "necessity" not in item:
                    return False, f"{side}[{i}] 缺少 keywords/content/necessity", None
                if not isinstance(item["keywords"], list) or len(item["keywords"]) == 0:
                    return False, f"{side}[{i}].keywords 必须为非空 list", None
                if not isinstance(item["content"], str) or not item["content"].strip():
                    return False, f"{side}[{i}].content 必须为非空字符串", None
                if not isinstance(item["necessity"], bool):
                    return False, f"{side}[{i}].necessity 必须为 bool", None

        gt = data["ground_truth"]
        if not isinstance(gt, dict):
            return False, "ground_truth 必须为 dict", None

        gt_required = ["diagnosis", "treatment", "avoid"]
        for field in gt_required:
            if field not in gt:
                return False, f"ground_truth 缺少字段: {field}", None
            if not isinstance(gt[field], list):
                return False, f"ground_truth.{field} 必须为 list", None

        case_obj = MedicalCase.dict_to_case(data)
        normalized = case_obj.case_to_dict()
        normalized["source"] = data.get("source", "synthetic")

        return True, "ok", normalized
    except Exception as e:
        return False, f"schema 校验异常: {e}", None


# -----------------------------------------------------------------------------
# Prompt 构造
# -----------------------------------------------------------------------------

def build_generate_prompt(seed: Dict[str, Any]) -> str:
    return f"""{GENERATE_PROMPT}

【种子信息】
- 种子ID: {seed.get("id")}
- 科室: {seed.get("label", "")}
- 相关疾病: {seed.get("related_diseases", "")}
- 用户问题:
{seed.get("question", "")}
"""


def build_review_prompt(seed: Dict[str, Any], case_data: Dict[str, Any]) -> str:
    return f"""{REVIEW_PROMPT}

【种子信息】
- 种子ID: {seed.get("id")}
- 科室: {seed.get("label", "")}
- 相关疾病: {seed.get("related_diseases", "")}
- 用户问题:
{seed.get("question", "")}

【待审核病例】
{json.dumps(case_data, ensure_ascii=False, indent=2)}
"""


def build_rewrite_prompt(seed: Dict[str, Any], case_data: Dict[str, Any], review_summary: Dict[str, Any]) -> str:
    return f"""{REWRITE_PROMPT}

【种子信息】
- 种子ID: {seed.get("id")}
- 科室: {seed.get("label", "")}
- 相关疾病: {seed.get("related_diseases", "")}
- 用户问题:
{seed.get("question", "")}

【原始病例】
{json.dumps(case_data, ensure_ascii=False, indent=2)}

【review 汇总意见】
{json.dumps(review_summary, ensure_ascii=False, indent=2)}
"""


def build_judge_prompt(seed: Dict[str, Any], case_data: Dict[str, Any]) -> str:
    return f"""{JUDGE_PROMPT}

【种子信息】
- 种子ID: {seed.get("id")}
- 科室: {seed.get("label", "")}
- 相关疾病: {seed.get("related_diseases", "")}
- 用户问题:
{seed.get("question", "")}

【待评判病例】
{json.dumps(case_data, ensure_ascii=False, indent=2)}
"""


# -----------------------------------------------------------------------------
# 四步处理
# -----------------------------------------------------------------------------

def do_generate(seed: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_generate_prompt(seed)
    raw_response = call_llm_with_system(
        prompt=prompt,
        model=SYNTHESIS_MODEL,
        system_prompt=None,
        response_format=None,
    )
    parsed = parse_tagged_json(raw_response, "result")
    if not parsed.get("case_id"):
        parsed["case_id"] = f"SYN_{seed.get('id')}"
    parsed["source"] = "synthetic"
    return {
        "model": SYNTHESIS_MODEL,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed": parsed,
    }


def do_review(seed: Dict[str, Any], case_data: Dict[str, Any]) -> Dict[str, Any]:
    details = []

    for model in REVIEW_MODELS:
        prompt = build_review_prompt(seed, case_data)
        try:
            raw_response = call_llm_with_system(
                prompt=prompt,
                model=model,
                system_prompt=None,
                response_format=None,
            )
            parsed = parse_tagged_json(raw_response, "review")
            details.append({
                "model": model,
                "prompt": prompt,
                "raw_response": raw_response,
                "parsed": parsed,
                "success": True,
            })
        except Exception as e:
            details.append({
                "model": model,
                "prompt": prompt,
                "raw_response": None,
                "parsed": {
                    "pass": False,
                    "issues": [f"review 调用失败: {e}"],
                    "suggestions": ["检查模型调用或输出格式"],
                    "summary": "review 失败",
                },
                "success": False,
                "error": str(e),
            })

    passing_models = []
    failing_models = []
    issues = []
    suggestions = []

    for item in details:
        parsed = item.get("parsed", {})
        if parsed.get("pass", False):
            passing_models.append(item["model"])
        else:
            failing_models.append(item["model"])

        issues.extend(parsed.get("issues", []))
        suggestions.extend(parsed.get("suggestions", []))

    def dedup_keep_order(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    summary = {
        "passing_models": passing_models,
        "failing_models": failing_models,
        "passing_count": len(passing_models),
        "total_models": len(REVIEW_MODELS),
        "pass": len(passing_models) >= 3,
        "issues": dedup_keep_order(issues),
        "suggestions": dedup_keep_order(suggestions),
    }

    return {
        "details": details,
        "summary": summary,
    }


def do_rewrite(seed: Dict[str, Any], case_data: Dict[str, Any], review_summary: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_rewrite_prompt(seed, case_data, review_summary)
    raw_response = call_llm_with_system(
        prompt=prompt,
        model=REWRITE_MODEL,
        system_prompt=None,
        response_format=None,
    )
    parsed = parse_tagged_json(raw_response, "rewrite")
    if not parsed.get("case_id"):
        parsed["case_id"] = case_data.get("case_id", f"SYN_{seed.get('id')}")
    parsed["source"] = "synthetic"

    return {
        "model": REWRITE_MODEL,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed": parsed,
    }


def do_judge(seed: Dict[str, Any], case_data: Dict[str, Any]) -> Dict[str, Any]:
    details = []

    for model in JUDGE_MODELS:
        prompt = build_judge_prompt(seed, case_data)
        try:
            raw_response = call_llm_with_system(
                prompt=prompt,
                model=model,
                system_prompt=None,
                response_format=None,
            )
            parsed = parse_tagged_json(raw_response, "judge")
            details.append({
                "model": model,
                "prompt": prompt,
                "raw_response": raw_response,
                "parsed": parsed,
                "success": True,
            })
        except Exception as e:
            details.append({
                "model": model,
                "prompt": prompt,
                "raw_response": None,
                "parsed": {
                    "pass": False,
                    "reason": f"judge 调用失败: {e}",
                },
                "success": False,
                "error": str(e),
            })

    passing_models = [x["model"] for x in details if x.get("parsed", {}).get("pass", False)]
    failing_models = [x["model"] for x in details if not x.get("parsed", {}).get("pass", False)]

    summary = {
        "passing_models": passing_models,
        "failing_models": failing_models,
        "passing_count": len(passing_models),
        "total_models": len(JUDGE_MODELS),
        "pass": len(passing_models) >= MIN_PASSING_MODELS,
    }

    return {
        "details": details,
        "summary": summary,
    }


# -----------------------------------------------------------------------------
# handler：单条数据完整处理
# -----------------------------------------------------------------------------

def handler(item: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
    seed_id = item.get("id", "unknown")

    record = {
        "id": str(seed_id),
        "seed": item,
        "status": "failed",
        "failed_stage": None,
        "error": None,
        "generate": None,
        "generate_schema_check": None,
        "review": None,
        "rewrite": None,
        "rewrite_schema_check": None,
        "judge": None,
        "final_case": None,
    }

    try:
        tqdm.write(f"[Seed {seed_id}] generate")
        gen_result = None
        gen_error = None
        for _ in range(max_retries):
            try:
                gen_result = do_generate(item)
                break
            except Exception as e:
                gen_error = e

        if gen_result is None:
            record["failed_stage"] = "generate"
            record["error"] = f"generate failed: {gen_error}"
            tqdm.write(f"[Seed {seed_id}] generate failed: {gen_error}")
            return record

        record["generate"] = gen_result

        ok, msg, normalized_gen = validate_medical_case_dict(gen_result["parsed"])
        record["generate_schema_check"] = {
            "pass": ok,
            "message": msg,
            "normalized": normalized_gen if ok else None,
        }
        if not ok:
            record["failed_stage"] = "generate_schema_check"
            record["error"] = msg
            tqdm.write(f"[Seed {seed_id}] generate_schema_check failed: {msg}")
            return record

        tqdm.write(f"[Seed {seed_id}] review")
        review_result = do_review(item, normalized_gen)
        record["review"] = review_result

        tqdm.write(f"[Seed {seed_id}] rewrite")
        rewrite_result = None
        rewrite_error = None
        for _ in range(max_retries):
            try:
                rewrite_result = do_rewrite(item, normalized_gen, review_result["summary"])
                break
            except Exception as e:
                rewrite_error = e

        if rewrite_result is None:
            record["failed_stage"] = "rewrite"
            record["error"] = f"rewrite failed: {rewrite_error}"
            tqdm.write(f"[Seed {seed_id}] rewrite failed: {rewrite_error}")
            return record

        record["rewrite"] = rewrite_result

        ok2, msg2, normalized_rewrite = validate_medical_case_dict(rewrite_result["parsed"])
        record["rewrite_schema_check"] = {
            "pass": ok2,
            "message": msg2,
            "normalized": normalized_rewrite if ok2 else None,
        }
        if not ok2:
            record["failed_stage"] = "rewrite_schema_check"
            record["error"] = msg2
            tqdm.write(f"[Seed {seed_id}] rewrite_schema_check failed: {msg2}")
            return record

        tqdm.write(f"[Seed {seed_id}] judge")
        judge_result = do_judge(item, normalized_rewrite)
        record["judge"] = judge_result
        record["final_case"] = normalized_rewrite

        if judge_result["summary"]["pass"]:
            record["status"] = "passed"
            tqdm.write(f"[Seed {seed_id}] passed")
        else:
            record["status"] = "failed"
            record["failed_stage"] = "judge"
            record["error"] = (
                f"judge 未通过: "
                f"{judge_result['summary']['passing_count']}/{judge_result['summary']['total_models']}"
            )
            tqdm.write(f"[Seed {seed_id}] judge failed")

        return record

    except Exception as e:
        record["status"] = "failed"
        record["failed_stage"] = record["failed_stage"] or "pipeline_exception"
        record["error"] = str(e)
        tqdm.write(f"[Seed {seed_id}] exception: {e}")
        return record


# -----------------------------------------------------------------------------
# 结果后处理
# -----------------------------------------------------------------------------

def select(result_file_path: str, output_file_path: str) -> List[Dict[str, Any]]:
    """
    从 result.json / result.jsonl 中选择成功样本，输出 selected.jsonl
    每条仅保留 final_case（MedicalCase dict）
    """
    selected_cases = []

    if result_file_path.endswith(".json"):
        with open(result_file_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            if row.get("status") == "passed" and row.get("final_case") is not None:
                selected_cases.append(row["final_case"])
    else:
        with open(result_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") == "passed" and row.get("final_case") is not None:
                    selected_cases.append(row["final_case"])

    with open(output_file_path, "w", encoding="utf-8") as f:
        for case in selected_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"[SELECT] Selected {len(selected_cases)} passed cases -> {output_file_path}")
    return selected_cases


def stats(result_file_path: str, output_file_path: str) -> Dict[str, Any]:
    """
    对 result.json / result.jsonl 做统计，输出 stats.json
    """
    total = 0
    passed = 0
    failed = 0
    failed_by_stage = {}

    if result_file_path.endswith(".json"):
        with open(result_file_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        iterable = rows
    else:
        with open(result_file_path, "r", encoding="utf-8") as f:
            iterable = [json.loads(line) for line in f if line.strip()]

    for row in iterable:
        total += 1
        if row.get("status") == "passed":
            passed += 1
        else:
            failed += 1
            stage = row.get("failed_stage", "unknown")
            failed_by_stage[stage] = failed_by_stage.get(stage, 0) + 1

    result = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total * 100, 2) if total > 0 else 0.0,
        "failed_by_stage": failed_by_stage,
        "judge_models": JUDGE_MODELS,
        "min_passing_models": MIN_PASSING_MODELS,
    }

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[STATS] {result}")
    return result


def run_one_shard(shard_data: List[Dict[str, Any]], shard_output_dir: Path) -> Dict[str, Any]:
    shard_output_dir.mkdir(parents=True, exist_ok=True)

    result_file = shard_output_dir / "result.json"
    selected_file = shard_output_dir / "selected.jsonl"
    stats_file = shard_output_dir / "stats.json"

    print(f"\n[SHARD] 开始处理: {shard_output_dir.name}")
    print(f"[SHARD] 数据量: {len(shard_data)}")
    print(f"[SHARD] 输出目录: {shard_output_dir}")

    run_concurrent_task(
        input_data=shard_data,
        output_file=str(result_file),
        handler=handler,
        max_workers=MAX_CONCURRENT_REQUESTS,
        output_format="json",
        checkpoint_every=10,
        return_results=False,
    )

    selected = select(str(result_file), str(selected_file))
    shard_stats = stats(str(result_file), str(stats_file))

    return {
        "shard_dir": str(shard_output_dir),
        "result_file": str(result_file),
        "selected_file": str(selected_file),
        "stats_file": str(stats_file),
        "selected_count": len(selected),
        "stats": shard_stats,
    }


def merge_selected_files(shard_dirs: List[Path], merged_output_file: Path) -> int:
    total = 0
    with open(merged_output_file, "w", encoding="utf-8") as fout:
        for shard_dir in shard_dirs:
            selected_file = shard_dir / "selected.jsonl"
            if not selected_file.exists():
                continue
            with open(selected_file, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
                        total += 1
    return total


def merge_stats(shard_dirs: List[Path], output_file: Path) -> Dict[str, Any]:
    merged = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "failed_by_stage": {},
        "num_shards": len(shard_dirs),
        "judge_models": JUDGE_MODELS,
        "min_passing_models": MIN_PASSING_MODELS,
    }

    for shard_dir in shard_dirs:
        stats_file = shard_dir / "stats.json"
        if not stats_file.exists():
            continue
        with open(stats_file, "r", encoding="utf-8") as f:
            s = json.load(f)

        merged["total"] += s.get("total", 0)
        merged["passed"] += s.get("passed", 0)
        merged["failed"] += s.get("failed", 0)

        for k, v in s.get("failed_by_stage", {}).items():
            merged["failed_by_stage"][k] = merged["failed_by_stage"].get(k, 0) + v

    merged["pass_rate"] = round(merged["passed"] / merged["total"] * 100, 2) if merged["total"] else 0.0

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    return merged


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------

def run_pipeline(
    n_samples: int = N_SAMPLES,
    output_dir_override: Optional[Path] = None,
    sample_seed: int = 42,
) -> Path:
    print("=" * 80)
    print("医疗数据合成 Pipeline")
    print("=" * 80)

    output_dir = output_dir_override or (OUTPUT_ROOT / f"generate_{n_samples}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {output_dir}")
    print(f"种子数据: {SEED_DATA_PATH}")
    print(f"目标样本数: {n_samples}")
    print(f"shard 大小: {SHARD_SIZE}")
    print(f"shard 间隔: {SHARD_INTERVAL_SECONDS} 秒")
    print(f"最大并发: {MAX_CONCURRENT_REQUESTS}")

    # 1) 读取并按 label 尽量均匀抽样
    all_seed_data = load_all_seed_data(SEED_DATA_PATH)
    sampled_seed_data = balanced_sample_by_label(all_seed_data, n_samples, seed=sample_seed)

    sampled_seed_file = output_dir / "sampled_seeds.jsonl"
    write_jsonl(sampled_seed_file, sampled_seed_data)

    print(f"均匀抽样完成，共 {len(sampled_seed_data)} 条")
    print(f"抽样结果已写出: {sampled_seed_file}")

    # 2) shard 切分
    shards = chunk_list(sampled_seed_data, SHARD_SIZE)
    shard_dirs = []

    for shard_idx, shard_data in enumerate(shards):
        shard_dir = output_dir / f"shard_{shard_idx:04d}"
        shard_dirs.append(shard_dir)

        if shard_idx > 0:
            print(f"\n[WAIT] 等待 {SHARD_INTERVAL_SECONDS} 秒后启动下一个 shard...")
            time.sleep(SHARD_INTERVAL_SECONDS)

        run_one_shard(shard_data, shard_dir)

    # 3) 合并成功数据
    merged_selected_file = output_dir / "merged_selected.jsonl"
    merged_selected_count = merge_selected_files(shard_dirs, merged_selected_file)

    # 4) 合并统计
    merged_stats_file = output_dir / "merged_stats.json"
    merged_stats = merge_stats(shard_dirs, merged_stats_file)

    print("\n" + "=" * 80)
    print("Pipeline 完成")
    print("=" * 80)
    print(f"sampled_seeds.jsonl   : {sampled_seed_file}")
    print(f"merged_selected.jsonl : {merged_selected_file}")
    print(f"merged_stats.json     : {merged_stats_file}")
    print(f"合并成功样本数        : {merged_selected_count}")
    print(f"总体通过率            : {merged_stats['pass_rate']}%")
    print("=" * 80)

    return output_dir


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="医疗数据合成 Pipeline")
    parser.add_argument("--n", type=int, default=N_SAMPLES, help=f"抽取种子数据数量 (默认: {N_SAMPLES})")
    parser.add_argument("--output", type=str, default=None, help="输出目录 (默认: output/generate_{N})")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output) if args.output else None
    final_dir = run_pipeline(
        n_samples=args.n,
        output_dir_override=output_dir,
        sample_seed=args.seed,
    )

    print(f"\n🎉 Pipeline 完成！结果保存在: {final_dir}")
