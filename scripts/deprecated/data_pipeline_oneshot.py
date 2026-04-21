#!/usr/bin/env python3
"""
MedAgent One-Shot 数据合成 Pipeline

特点:
1. 单次LLM调用完成信息分析+数据合成
2. 根据问题类型自适应调整合成策略
3. 成本降低50%+，流程简化

使用方式:
    # 默认训练
    python scripts/data_pipeline_oneshot.py

    # 指定参数
    python scripts/data_pipeline_oneshot.py --start-idx 0 --end-idx 1000

    # 指定输出路径
    python scripts/data_pipeline_oneshot.py --output output/synthesis_oneshot
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import extract_tag_content


# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
SEED_DATA_PATH = PROJECT_ROOT / "data" / "seed_dataset" / "format_data.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "synthesis_oneshot"

MAX_CONCURRENT_REQUESTS = 10
BATCH_SIZE = 50  # 每50条保存一次checkpoint

# 模型配置
MODEL = "qwen3.5-plus"
BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
API_KEY = os.getenv("DASHSCOPE_API_KEY_CP")

# Judge 配置（用于数据质量验证）
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "/dev/shm/models/baichuan-inc/Baichuan-M3-235B-GPTQ-INT4")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "http://localhost:8100/v1")


# -----------------------------------------------------------------------------
# One-Shot Prompt
# -----------------------------------------------------------------------------
ONE_SHOT_PROMPT_TEMPLATE = '''你是一位专业的医疗数据合成专家。请分析原始医疗问答并合成结构化病例数据。

## 任务说明

你需要完成两个步骤：
1. **信息分析**：分析原始问题中包含的信息类型
2. **数据合成**：根据分析结果生成MedicalCase格式数据

## 第一步：信息分析

分析原始问题(Q)，判断：

- **has_subjective**: 是否包含患者主观症状描述？
  - 如：疼痛、瘙痒、头晕、恶心等患者主观感受
  - 注意：仅描述症状存在不算，需要有具体描述内容

- **has_objective**: 是否包含检查/检验结果？
  - 如：B超结果、化验报告、CT影像描述、具体数值等
  - 注意：仅提及"检查过"不算，需要有具体结果

- **is_simple_consultation**: 是否为简单咨询问题？
  - 特征：信息完整、无需渐进披露、可直接回答
  - 如："糖尿病饮食注意事项"、"感冒吃什么药"

## 第二步：数据合成

根据分析结果，采用不同合成策略：

### 策略A: 简单咨询型 (is_simple_consultation=true)
- chief_complaint: 从Q提取核心诉求
- subjective: 1-2项，简短描述
- objective: 0-1项（如有相关信息）
- ground_truth: 基于A生成

### 策略B: 症状主导型 (has_subjective=true, has_objective=false)
- chief_complaint: 主诉
- subjective: 2-4项，详细症状描述
- objective: 1-2项，基础检查（如体温、血压）
- ground_truth: 诊断+治疗+禁忌

### 策略C: 检查主导型 (has_subjective=false, has_objective=true)
- chief_complaint: 主诉
- subjective: 1-2项，基础问诊
- objective: 2-4项，详细检查结果
- ground_truth: 诊断+治疗+禁忌

### 策略D: 综合病例型 (两者都为true)
- chief_complaint: 主诉
- subjective: 2-4项，完整症状描述
- objective: 2-4项，完整检查结果
- ground_truth: 诊断+治疗+禁忌

## MedicalCase 数据结构

- case_id: SYN_xxx
- difficulty: easy/medium/hard
- tags: 列表形式，如 [科室, 疾病类型]
- chief_complaint: 主诉（患者核心诉求）
- subjective: 列表形式，每项包含 keywords/content/necessity
- objective: 列表形式，每项包含 keywords/content/necessity
- ground_truth: 包含 diagnosis/treatment/avoid 三个列表
- source: synthetic

## 硬性要求

1. subjective和objective必须为数组，每项至少2项（策略A除外）
2. 每项必须包含：keywords（数组）、content（字符串）、necessity（布尔）
3. ground_truth必须包含：diagnosis、treatment、avoid（数组）
4. 禁忌项应与诊断/治疗相关，避免通用建议
5. 使用中文
6. 保持医学合理性

## 输出格式

请直接输出JSON，不需要代码块包裹，不需要额外解释：

<result>
{"analysis": {"has_subjective": bool, "has_objective": bool, "is_simple_consultation": bool, "strategy": "A/B/C/D", "reason": "简要分析原因"}, "medical_case": {"case_id": "SYN_xxx", "difficulty": "medium", "tags": ["科室"], "chief_complaint": "主诉", "subjective": [{"keywords": ["关键词"], "content": "描述内容", "necessity": true}], "objective": [{"keywords": ["检查关键词"], "content": "检查结果", "necessity": true}], "ground_truth": {"diagnosis": ["诊断"], "treatment": ["治疗方案"], "avoid": ["禁忌项"]}, "source": "synthetic"}}
</result>'''


# -----------------------------------------------------------------------------
# Few-shot 示例（可选，用于提升质量）
# -----------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {
        "question": "糖尿病饮食注意事项",
        "answer": "糖尿病患者饮食应注意：控制总热量、少食多餐、低糖低脂、高纤维饮食。避免含糖饮料、甜点。定期监测血糖。",
        "label": "内科",
        "expected_analysis": {
            "has_subjective": False,
            "has_objective": False,
            "is_simple_consultation": True,
            "strategy": "A"
        }
    },
    {
        "question": "我头痛3天了，主要是额头和太阳穴胀痛，伴有恶心，没有呕吐。请问这是什么原因？",
        "answer": "根据您的描述，可能是紧张型头痛或偏头痛。建议：1.注意休息；2.避免过度用眼；3.如症状持续建议做头颅CT排除器质性病变。",
        "label": "神经科学",
        "expected_analysis": {
            "has_subjective": True,
            "has_objective": False,
            "is_simple_consultation": False,
            "strategy": "B"
        }
    },
    {
        "question": "B超显示子宫肌瘤3cm，需要手术吗？",
        "answer": "子宫肌瘤3cm属于较小肌瘤，如无症状可定期观察。如月经量多或有压迫症状，可考虑手术。建议每6个月复查B超。",
        "label": "妇产科",
        "expected_analysis": {
            "has_subjective": False,
            "has_objective": True,
            "is_simple_consultation": False,
            "strategy": "C"
        }
    },
]


# -----------------------------------------------------------------------------
# 核心函数
# -----------------------------------------------------------------------------
def make_client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_llm(prompt: str, model: str = MODEL) -> str:
    client = make_client()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        raise e
    finally:
        try:
            client.close()
        except:
            pass


def parse_result(response: str) -> Dict:
    """解析LLM响应"""
    content = extract_tag_content(response, "result")
    if content:
        try:
            # 尝试直接解析
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试提取代码块中的JSON
        import re
        code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试找到第一个完整的JSON对象
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    raise ValueError("未能解析有效结果")


def validate_medical_case(data: Dict) -> tuple:
    """校验MedicalCase格式"""
    # 检查必要字段
    required = ["case_id", "chief_complaint", "subjective", "objective", "ground_truth"]
    for f in required:
        if f not in data:
            return False, f"缺少字段: {f}"

    # 获取分析结果判断策略
    analysis = data.get("_analysis", {})
    is_simple = analysis.get("is_simple_consultation", False)

    # 检查subjective
    subjective = data.get("subjective", [])
    if not isinstance(subjective, list):
        return False, "subjective必须是数组"
    if not is_simple and len(subjective) < 2:
        return False, "subjective需至少2项"

    # 检查objective
    objective = data.get("objective", [])
    if not isinstance(objective, list):
        return False, "objective必须是数组"
    if not is_simple and len(objective) < 2:
        return False, "objective需至少2项"

    # 检查ground_truth
    gt = data.get("ground_truth", {})
    for f in ["diagnosis", "treatment", "avoid"]:
        if f not in gt:
            return False, f"ground_truth缺少{f}"
        if not isinstance(gt.get(f), list):
            return False, f"ground_truth.{f}必须是数组"

    return True, "ok"


def process_one_sample(seed: Dict, idx: int) -> Dict:
    """处理单条样本"""
    result = {
        "seed_id": seed.get("id", f"unknown_{idx}"),
        "seed_idx": idx,
        "status": "failed",
        "analysis": None,
        "medical_case": None,
        "error": None,
    }

    try:
        # 构建prompt - 使用拼接避免format问题
        prompt = ONE_SHOT_PROMPT_TEMPLATE + f'''

---

## 原始数据

**问题(Q)**: {seed.get("question", "")}

**回答(A)**: {seed.get("answer", "")}

**标签**: {seed.get("label", "未知")}

---

请开始分析和合成，直接输出结果JSON。'''

        # 调用LLM
        response = call_llm(prompt)

        # 解析结果
        parsed = parse_result(response)

        analysis = parsed.get("analysis", {})
        medical_case = parsed.get("medical_case", {})

        # 补充字段
        medical_case["case_id"] = f"SYN_{seed.get('id', idx)}"
        medical_case["source"] = "synthetic"
        medical_case["_analysis"] = analysis

        # 校验
        ok, msg = validate_medical_case(medical_case)
        if not ok:
            result["error"] = f"validation: {msg}"
            return result

        # 移除内部字段
        del medical_case["_analysis"]

        result["analysis"] = analysis
        result["medical_case"] = medical_case
        result["status"] = "passed"

    except Exception as e:
        result["error"] = str(e)

    return result


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def run_pipeline(
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = BATCH_SIZE,
    max_workers: int = MAX_CONCURRENT_REQUESTS,
):
    """运行One-Shot数据合成Pipeline"""

    print("=" * 70)
    print("MedAgent One-Shot 数据合成 Pipeline")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {MODEL}")
    print(f"输出目录: {output_dir}")
    print(f"样本范围: {start_idx} - {end_idx or '全部'}")
    print("=" * 70)

    # 加载种子数据
    print("\n[1/3] 加载种子数据...")
    with open(SEED_DATA_PATH, "r", encoding="utf-8") as f:
        all_seeds = [json.loads(line) for line in f if line.strip()]
    print(f"  加载 {len(all_seeds)} 条种子数据")

    # 确定处理范围
    end_idx = end_idx or len(all_seeds)
    seeds_to_process = all_seeds[start_idx:end_idx]
    print(f"  将处理 {len(seeds_to_process)} 条")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "checkpoint.jsonl"
    result_file = output_dir / "results.jsonl"
    stats_file = output_dir / "stats.json"

    # 加载已有checkpoint
    processed_ids = set()
    results = []

    if checkpoint_file.exists():
        print(f"\n[恢复] 从checkpoint恢复...")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    processed_ids.add(r["seed_idx"])
                    results.append(r)
        print(f"  已恢复 {len(processed_ids)} 条结果")

    # 过滤已处理的样本
    remaining = [(start_idx + i, s) for i, s in enumerate(seeds_to_process)
                 if (start_idx + i) not in processed_ids]
    print(f"  剩余待处理: {len(remaining)}/{len(seeds_to_process)}")

    if not remaining:
        print("\n所有样本已处理完成")
    else:
        print(f"\n[2/3] 开始合成...")

        # 并发处理
        batch_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_one_sample, seed, idx): (idx, seed)
                       for idx, seed in remaining}

            for future in tqdm(as_completed(futures), total=len(futures), desc="合成进度"):
                r = future.result()
                results.append(r)
                batch_results.append(r)

                # 定期保存checkpoint
                if len(batch_results) >= batch_size:
                    with open(checkpoint_file, "a", encoding="utf-8") as f:
                        for res in batch_results:
                            f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    batch_results.clear()

        # 保存剩余checkpoint
        if batch_results:
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                for res in batch_results:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")

    # 统计
    print(f"\n[3/3] 统计结果...")
    passed = [r for r in results if r["status"] == "passed"]
    failed = [r for r in results if r["status"] != "passed"]

    # 统计分析结果分布
    strategy_dist = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in passed:
        strategy = r.get("analysis", {}).get("strategy", "D")
        strategy_dist[strategy] = strategy_dist.get(strategy, 0) + 1

    stats = {
        "total": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": round(len(passed) / len(results) * 100, 2) if results else 0,
        "strategy_distribution": strategy_dist,
        "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # 保存结果
    with open(result_file, "w", encoding="utf-8") as f:
        for r in passed:
            if r.get("medical_case"):
                f.write(json.dumps(r["medical_case"], ensure_ascii=False) + "\n")

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 输出结果
    print("\n" + "=" * 70)
    print("合成完成!")
    print("=" * 70)
    print(f"总处理: {stats['total']}")
    print(f"通过: {stats['passed']}")
    print(f"失败: {stats['failed']}")
    print(f"通过率: {stats['pass_rate']}%")
    print(f"\n策略分布:")
    for strategy, count in strategy_dist.items():
        pct = count / len(passed) * 100 if passed else 0
        print(f"  策略{strategy}: {count} ({pct:.1f}%)")
    print(f"\n输出文件:")
    print(f"  结果: {result_file}")
    print(f"  统计: {stats_file}")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedAgent One-Shot 数据合成 Pipeline")
    parser.add_argument("--start-idx", type=int, default=0, help="起始索引")
    parser.add_argument("--end-idx", type=int, default=None, help="结束索引")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="输出目录")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="批次大小")
    parser.add_argument("--max-workers", type=int, default=MAX_CONCURRENT_REQUESTS, help="并发数")

    args = parser.parse_args()

    run_pipeline(
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        output_dir=Path(args.output),
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )