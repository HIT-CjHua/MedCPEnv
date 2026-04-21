#!/usr/bin/env python3
"""
MedAgent 大规模数据合成 Pipeline

特性:
1. 按label均匀采样，划分为size=500的shard
2. 每个shard处理完后暂停指定时间（默认2小时）
3. 断点续跑机制
4. 支持指定shard范围

使用方式:
    # 处理全部shard
    python scripts/data_pipeline_v2.py

    # 处理shard 1-10
    python scripts/data_pipeline_v2.py --end-shard 10

    # 处理shard 5-10
    python scripts/data_pipeline_v2.py --start-shard 5 --end-shard 10

    # 指定暂停间隔（秒）
    python scripts/data_pipeline_v2.py --interval 7200
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
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
SEED_DATA_PATH = PROJECT_ROOT / "data" / "seed_dataset" / "format_data.jsonl"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "synthesis_v2"

SHARD_SIZE = 500
SHARD_INTERVAL_SECONDS = 7200  # 2小时
MAX_CONCURRENT_REQUESTS = 10

# 优化后的模型配置 (6次调用/样本)
SYNTHESIS_MODEL = "qwen3.5-plus"
REWRITE_MODEL = "qwen3.5-plus"
REVIEW_MODELS = ["glm-5"]
JUDGE_MODELS = ["qwen3.5-plus", "glm-5", "MiniMax-M2.5"]
MIN_PASSING_JUDGES = 3

BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
API_KEY = os.getenv("DASHSCOPE_API_KEY_CP")

# -----------------------------------------------------------------------------
# Prompts
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
2. 每个 item 必须包含：keywords, content, necessity
3. ground_truth 必须包含：diagnosis, treatment, avoid
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
  "suggestions": ["建议1", "建议2"]
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
    """校验dict是否符合MedicalCase schema"""
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


# -----------------------------------------------------------------------------
# 核心处理逻辑
# -----------------------------------------------------------------------------
def process_one_sample(seed: Dict, seed_idx: int) -> Dict:
    """处理单条样本"""
    seed_id = seed.get("id", f"unknown_{seed_idx}")
    result = {
        "seed_id": seed_id,
        "seed_idx": seed_idx,
        "status": "failed",
        "calls": 0,
        "final_case": None,
        "error": None,
    }

    try:
        # 1. Generate
        gen_prompt = f"{GENERATE_PROMPT}\n\n【种子信息】\n种子ID: {seed_id}\n科室: {seed.get('label', '')}\n相关疾病: {seed.get('related_diseases', '')}\n用户问题:\n{seed.get('question', '')}"
        gen_resp = call_llm(gen_prompt, SYNTHESIS_MODEL)
        result["calls"] += 1
        case_data = parse_tagged_json(gen_resp, "result")
        case_data["source"] = "synthetic"

        ok, msg = validate_schema(case_data)
        if not ok:
            result["error"] = f"generate schema: {msg}"
            return result

        # 2. Review (单模型)
        all_issues = []
        all_suggestions = []
        for model in REVIEW_MODELS:
            rev_prompt = f"{REVIEW_PROMPT}\n\n【待审核病例】\n{json.dumps(case_data, ensure_ascii=False, indent=2)}"
            rev_resp = call_llm(rev_prompt, model)
            result["calls"] += 1
            try:
                rev_data = parse_tagged_json(rev_resp, "review")
                all_issues.extend(rev_data.get("issues", []))
                all_suggestions.extend(rev_data.get("suggestions", []))
            except:
                pass

        # 3. Rewrite
        rewrite_prompt = f"{REWRITE_PROMPT}\n\n【原始病例】\n{json.dumps(case_data, ensure_ascii=False, indent=2)}\n\n【问题】\n{all_issues}\n\n【建议】\n{all_suggestions}"
        rewrite_resp = call_llm(rewrite_prompt, REWRITE_MODEL)
        result["calls"] += 1
        rewritten = parse_tagged_json(rewrite_resp, "rewrite")
        rewritten["source"] = "synthetic"

        ok, msg = validate_schema(rewritten)
        if not ok:
            result["error"] = f"rewrite schema: {msg}"
            return result

        # 4. Judge (3模型)
        passing = 0
        for model in JUDGE_MODELS:
            judge_prompt = f"{JUDGE_PROMPT}\n\n【待评判病例】\n{json.dumps(rewritten, ensure_ascii=False, indent=2)}"
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
        result["judge_total"] = len(JUDGE_MODELS)

        if passing >= MIN_PASSING_JUDGES:
            result["status"] = "passed"
        else:
            result["error"] = f"judge未通过: {passing}/{len(JUDGE_MODELS)}"

    except Exception as e:
        result["error"] = str(e)

    return result


# -----------------------------------------------------------------------------
# Shard 处理
# -----------------------------------------------------------------------------
def process_shard(shard_idx: int, shard_data: List[Dict], output_dir: Path) -> Dict:
    """处理单个shard"""
    shard_dir = output_dir / f"shard_{shard_idx:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = shard_dir / "checkpoint.jsonl"
    result_file = shard_dir / "result.json"
    selected_file = shard_dir / "selected.jsonl"
    stats_file = shard_dir / "stats.json"

    print(f"\n{'='*60}")
    print(f"处理 Shard {shard_idx}")
    print(f"数据量: {len(shard_data)}")
    print(f"输出目录: {shard_dir}")
    print(f"{'='*60}")

    # 加载已处理的checkpoint
    processed_ids = set()
    results = []

    if checkpoint_file.exists():
        print(f"  从checkpoint恢复...")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    processed_ids.add(r["seed_idx"])
                    results.append(r)
        print(f"  已恢复 {len(processed_ids)} 条结果")

    # 过滤未处理的样本
    remaining = [(i, s) for i, s in enumerate(shard_data) if i not in processed_ids]
    print(f"  剩余待处理: {len(remaining)}/{len(shard_data)}")

    if not remaining:
        print(f"  Shard {shard_idx} 已完成")
    else:
        # 并发处理
        batch_results = []
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
            futures = {executor.submit(process_one_sample, seed, idx): (idx, seed) for idx, seed in remaining}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Shard {shard_idx}"):
                r = future.result()
                results.append(r)
                batch_results.append(r)

                # 定期保存checkpoint
                if len(batch_results) >= 50:
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
    passed = [r for r in results if r["status"] == "passed"]
    total_calls = sum(r["calls"] for r in results)

    stats = {
        "shard_idx": shard_idx,
        "total": len(shard_data),
        "processed": len(results),
        "passed": len(passed),
        "failed": len(results) - len(passed),
        "pass_rate": round(len(passed) / len(results) * 100, 2) if results else 0,
        "total_calls": total_calls,
        "avg_calls": round(total_calls / len(results), 2) if results else 0,
    }

    # 保存结果
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(selected_file, "w", encoding="utf-8") as f:
        for r in passed:
            if r.get("final_case"):
                f.write(json.dumps(r["final_case"], ensure_ascii=False) + "\n")

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n  Shard {shard_idx} 完成:")
    print(f"    通过: {len(passed)}/{len(results)} ({stats['pass_rate']}%)")
    print(f"    API调用: {total_calls}")

    return stats


# -----------------------------------------------------------------------------
# 数据划分
# -----------------------------------------------------------------------------
def balanced_shard_split(data: List[Dict], shard_size: int = 500, seed: int = 42) -> List[List[Dict]]:
    """
    按label均匀采样，划分为指定大小的shard
    """
    rng = random.Random(seed)

    # 按label分组
    label_to_items = defaultdict(list)
    for item in data:
        label = item.get("label", "未知")
        label_to_items[label].append(item)

    labels = sorted(label_to_items.keys())
    for label in labels:
        rng.shuffle(label_to_items[label])

    print(f"Label分布:")
    for label in labels:
        print(f"  {label}: {len(label_to_items[label])}")

    # 轮询采样，确保每个shard中label分布均匀
    all_shards = []
    current_shard = []

    # 计算总样本数
    total = sum(len(items) for items in label_to_items.values())
    num_shards = (total + shard_size - 1) // shard_size

    print(f"\n总样本数: {total}")
    print(f"Shard大小: {shard_size}")
    print(f"Shard数量: {num_shards}")

    # 轮询采样
    label_indices = {label: 0 for label in labels}
    finished_labels = set()

    while len(finished_labels) < len(labels):
        for label in labels:
            if label in finished_labels:
                continue

            items = label_to_items[label]
            idx = label_indices[label]

            if idx < len(items):
                current_shard.append(items[idx])
                label_indices[label] = idx + 1

                if len(current_shard) >= shard_size:
                    all_shards.append(current_shard)
                    current_shard = []

            if label_indices[label] >= len(items):
                finished_labels.add(label)

    # 添加最后一个不完整的shard
    if current_shard:
        all_shards.append(current_shard)

    return all_shards


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def run_pipeline(
    start_shard: int = 1,
    end_shard: Optional[int] = None,
    interval_seconds: int = SHARD_INTERVAL_SECONDS,
    shard_size: int = SHARD_SIZE,
    seed: int = 42,
):
    """运行数据合成pipeline"""

    print("=" * 60)
    print("MedAgent 大规模数据合成 Pipeline V2")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Shard范围: {start_shard} - {end_shard or '全部'}")
    print(f"Shard大小: {shard_size}")
    print(f"Shard间隔: {interval_seconds}秒 ({interval_seconds/3600:.1f}小时)")
    print(f"模型配置:")
    print(f"  Generate: {SYNTHESIS_MODEL}")
    print(f"  Review: {REVIEW_MODELS}")
    print(f"  Rewrite: {REWRITE_MODEL}")
    print(f"  Judge: {JUDGE_MODELS} (阈值: {MIN_PASSING_JUDGES}/{len(JUDGE_MODELS)})")
    print("=" * 60)

    # 加载种子数据
    print("\n加载种子数据...")
    with open(SEED_DATA_PATH, "r", encoding="utf-8") as f:
        all_seeds = [json.loads(line) for line in f if line.strip()]
    print(f"  加载 {len(all_seeds)} 条种子数据")

    # 划分shard
    print("\n划分Shard...")
    shards = balanced_shard_split(all_seeds, shard_size, seed)

    # 确定处理范围 (shard从1开始编号)
    start_idx = start_shard - 1  # 转为0-based
    end_idx = end_shard if end_shard else len(shards)

    shards_to_process = shards[start_idx:end_idx]
    print(f"\n将处理 {len(shards_to_process)} 个Shard (Shard {start_shard}-{start_shard + len(shards_to_process) - 1})")

    # 创建输出目录
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 保存shard划分信息
    shard_info = {
        "total_shards": len(shards),
        "shard_size": shard_size,
        "total_samples": len(all_seeds),
        "labels": list(set(s.get("label", "未知") for s in all_seeds)),
        "processing_range": [start_shard, start_shard + len(shards_to_process) - 1],
    }
    with open(OUTPUT_ROOT / "shard_info.json", "w", encoding="utf-8") as f:
        json.dump(shard_info, f, ensure_ascii=False, indent=2)

    # 处理每个shard
    all_stats = []
    for i, shard_data in enumerate(shards_to_process):
        shard_idx = start_shard + i

        stats = process_shard(shard_idx, shard_data, OUTPUT_ROOT)
        all_stats.append(stats)

        # 保存汇总统计
        with open(OUTPUT_ROOT / "all_stats.json", "w", encoding="utf-8") as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)

        # 暂停（最后一个shard不暂停）
        if i < len(shards_to_process) - 1:
            next_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            wake_time = datetime.fromtimestamp(time.time() + interval_seconds).strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n暂停 {interval_seconds/3600:.1f} 小时...")
            print(f"  当前时间: {next_time}")
            print(f"  预计继续: {wake_time}")
            time.sleep(interval_seconds)

    # 合并所有selected
    print("\n合并结果...")
    merged_file = OUTPUT_ROOT / "merged_selected.jsonl"
    total_passed = 0
    with open(merged_file, "w", encoding="utf-8") as fout:
        for stats in all_stats:
            shard_dir = OUTPUT_ROOT / f"shard_{stats['shard_idx']:04d}"
            selected_file = shard_dir / "selected.jsonl"
            if selected_file.exists():
                with open(selected_file, "r", encoding="utf-8") as fin:
                    for line in fin:
                        if line.strip():
                            fout.write(line)
                            total_passed += 1

    # 最终统计
    final_stats = {
        "total_shards_processed": len(all_stats),
        "total_processed": sum(s["processed"] for s in all_stats),
        "total_passed": total_passed,
        "total_failed": sum(s["failed"] for s in all_stats),
        "overall_pass_rate": round(total_passed / sum(s["processed"] for s in all_stats) * 100, 2) if all_stats else 0,
        "total_calls": sum(s["total_calls"] for s in all_stats),
        "end_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(OUTPUT_ROOT / "final_stats.json", "w", encoding="utf-8") as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Pipeline 完成!")
    print("=" * 60)
    print(f"处理Shard数: {final_stats['total_shards_processed']}")
    print(f"总处理数: {final_stats['total_processed']}")
    print(f"通过数: {final_stats['total_passed']}")
    print(f"通过率: {final_stats['overall_pass_rate']}%")
    print(f"总API调用: {final_stats['total_calls']}")
    print(f"完成时间: {final_stats['end_time']}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedAgent 大规模数据合成 Pipeline V2")
    parser.add_argument("--start-shard", type=int, default=1, help="起始Shard编号 (从1开始)")
    parser.add_argument("--end-shard", type=int, default=None, help="结束Shard编号 (包含)")
    parser.add_argument("--interval", type=int, default=SHARD_INTERVAL_SECONDS, help="Shard间隔时间(秒)")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE, help="每个Shard的样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    run_pipeline(
        start_shard=args.start_shard,
        end_shard=args.end_shard,
        interval_seconds=args.interval,
        shard_size=args.shard_size,
        seed=args.seed,
    )