#!/usr/bin/env python3
"""
M3 Generate Pipeline - 生成10k条病例数据

从 format_data.jsonl 抽样10k条，使用M3生成符合schema的病例数据
支持断点续跑
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------------------
# Prompt - 与 data_pipeline.py 保持一致
# -----------------------------------------------------------------------------

GENERATE_PROMPT = """你是一位专业的医疗数据合成专家。请根据用户提供的医疗问题，生成一条符合 MedicalCase schema 的病例数据。

【Schema 结构】
- case_id: 字符串，病例唯一标识
- difficulty: 字符串，难度级别 easy/medium/hard
- tags: 字符串列表，标签（科室、疾病类型等）
- chief_complaint: 字符串，主诉（患者主动描述的症状）
- subjective: 列表，问诊项，每项包含:
  - keywords: 触发关键词列表
  - content: 问诊回答内容
  - necessity: 是否必要
- objective: 列表，检查项，每项包含:
  - keywords: 触发关键词列表
  - content: 检查结果
  - necessity: 是否必要
- ground_truth: 评测标准答案
  - diagnosis: 正确诊断列表
  - treatment: 标准治疗方案列表
  - avoid: 禁忌项列表
- source: 数据来源标记为 "synthetic"

【要求】
1. 病例要有合理的医疗逻辑
2. subjective 至少 3 项（现病史、既往史、家族史等）
3. objective 至少 2 项（体格检查、实验室检查等）
4. 禁忌项要有医学依据
5. 直接输出 JSON，不要其他文字

【种子信息】
- 种子ID: {seed_id}
- 科室: {label}
- 相关疾病: {related_diseases}
- 用户问题:
{question}

请生成病例数据："""


def load_seed_data(file_path: str) -> List[Dict]:
    """加载种子数据"""
    seeds = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line))
    return seeds


def sample_seeds(seeds: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    """随机抽样"""
    random.seed(seed)
    if len(seeds) <= n:
        return seeds
    return random.sample(seeds, n)


def generate_case(
    client: OpenAI,
    model_name: str,
    seed: Dict,
    max_tokens: int = 16384,
) -> Dict[str, Any]:
    """生成单个病例"""
    start_time = time.time()

    prompt = GENERATE_PROMPT.format(
        seed_id=seed.get("id", "unknown"),
        label=seed.get("label", ""),
        related_diseases=seed.get("related_diseases", ""),
        question=seed.get("question", ""),
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )

        latency = time.time() - start_time
        content = response.choices[0].message.content

        # 解析JSON - M3会先分析再输出JSON
        case_data = None
        try:
            # 方法1: 找```json块
            if "```json" in content:
                parts = content.split("```json")
                for part in reversed(parts):
                    if "```" in part:
                        json_str = part.split("```")[0].strip()
                        try:
                            case_data = json.loads(json_str)
                            break
                        except:
                            continue
            # 方法2: 找{}块
            if not case_data:
                import re
                # 找最大的JSON对象
                brace_count = 0
                start_idx = -1
                for i, c in enumerate(content):
                    if c == '{':
                        if brace_count == 0:
                            start_idx = i
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_idx >= 0:
                            json_str = content[start_idx:i+1]
                            try:
                                case_data = json.loads(json_str)
                                if "case_id" in case_data or "chief_complaint" in case_data:
                                    break
                            except:
                                pass
        except:
            pass

        return {
            "seed_id": seed.get("id"),
            "success": True,
            "latency": latency,
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
            "case_data": case_data,
            "json_parsed": case_data is not None,
            "raw_content": content,
        }

    except Exception as e:
        return {
            "seed_id": seed.get("id"),
            "success": False,
            "latency": time.time() - start_time,
            "error": str(e),
        }


def load_checkpoint(checkpoint_file: str) -> Dict[int, Dict]:
    """加载checkpoint"""
    processed = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    seed_id = data.get("seed_id")
                    if seed_id is not None:
                        processed[seed_id] = data
    return processed


def save_checkpoint(checkpoint_file: str, result: Dict):
    """保存单条checkpoint"""
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def run_pipeline(
    client: OpenAI,
    model_name: str,
    seeds: List[Dict],
    output_dir: str,
    max_workers: int = 16,
    max_tokens: int = 16384,
    checkpoint_every: int = 10,
) -> Dict:
    """运行生成pipeline"""

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, "checkpoint.jsonl")

    # 加载已有checkpoint
    processed = load_checkpoint(checkpoint_file)
    processed_ids = set(processed.keys())

    # 过滤已处理的
    remaining = [s for s in seeds if s.get("id") not in processed_ids]
    print(f"已处理: {len(processed)}, 待处理: {len(remaining)}/{len(seeds)}")

    if not remaining:
        print("所有数据已处理完成")
        return {"total": len(seeds), "processed": len(processed)}

    # 统计
    stats = {
        "total": len(seeds),
        "processed": len(processed),
        "success": 0,
        "failed": 0,
        "json_parsed": 0,
        "total_tokens": 0,
        "total_output_tokens": 0,
        "latencies": [],
        "start_time": time.time(),
    }

    # 并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_case, client, model_name, seed, max_tokens): seed
            for seed in remaining
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generate"):
            seed = futures[future]
            result = future.result()

            # 更新统计
            if result["success"]:
                stats["success"] += 1
                stats["latencies"].append(result["latency"])
                stats["total_tokens"] += result["total_tokens"]
                stats["total_output_tokens"] += result["output_tokens"]
                if result["json_parsed"]:
                    stats["json_parsed"] += 1
            else:
                stats["failed"] += 1

            # 保存checkpoint
            save_checkpoint(checkpoint_file, result)

    stats["total_time"] = time.time() - stats["start_time"]

    return stats


def main():
    parser = argparse.ArgumentParser(description="M3 Generate Pipeline - 生成10k病例")
    parser.add_argument("--url", type=str, default="http://localhost:8100/v1", help="M3服务地址")
    parser.add_argument("--seed-data", type=str, default="data/seed_dataset/format_data.jsonl", help="种子数据路径")
    parser.add_argument("--n", type=int, default=10000, help="生成数量")
    parser.add_argument("--workers", type=int, default=16, help="并发数")
    parser.add_argument("--max-tokens", type=int, default=16384, help="最大生成长度")
    parser.add_argument("--output", type=str, default="output/m3_generate_10k", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    print(f"{'='*60}")
    print("M3 Generate Pipeline")
    print(f"{'='*60}")
    print(f"服务地址: {args.url}")
    print(f"种子数据: {args.seed_data}")
    print(f"生成数量: {args.n}")
    print(f"并发数: {args.workers}")
    print(f"最大生成长度: {args.max_tokens}")
    print(f"输出目录: {args.output}")
    print(f"{'='*60}")

    client = OpenAI(base_url=args.url, api_key="EMPTY")

    # 等待服务
    print("\n等待M3服务...")
    for i in range(30):
        try:
            models = client.models.list()
            model_name = models.data[0].id
            print(f"就绪: {model_name}")
            break
        except:
            time.sleep(3)

    # 加载种子数据
    print(f"\n加载种子数据...")
    all_seeds = load_seed_data(args.seed_data)
    print(f"总种子数: {len(all_seeds)}")

    # 抽样
    seeds = sample_seeds(all_seeds, args.n, args.seed)
    print(f"抽样: {len(seeds)} 条")

    # 运行pipeline
    print(f"\n开始生成...")
    stats = run_pipeline(
        client=client,
        model_name=model_name,
        seeds=seeds,
        output_dir=args.output,
        max_workers=args.workers,
        max_tokens=args.max_tokens,
    )

    # 输出统计
    print(f"\n{'='*60}")
    print("生成完成")
    print(f"{'='*60}")
    print(f"总数: {stats['total']}")
    print(f"成功: {stats['success']}")
    print(f"失败: {stats['failed']}")
    print(f"JSON解析成功: {stats['json_parsed']}")
    print(f"总耗时: {stats['total_time']:.2f}s")
    if stats['latencies']:
        print(f"平均延迟: {sum(stats['latencies'])/len(stats['latencies']):.2f}s")
    if stats['total_time'] > 0:
        print(f"生成速度: {stats['total_output_tokens']/stats['total_time']:.2f} tokens/s")

    # 保存最终结果
    checkpoint_file = os.path.join(args.output, "checkpoint.jsonl")
    output_file = os.path.join(args.output, "generated_cases.jsonl")

    # 从checkpoint提取成功解析的病例
    success_count = 0
    with open(checkpoint_file, "r", encoding="utf-8") as fin:
        with open(output_file, "w", encoding="utf-8") as fout:
            for line in fin:
                if line.strip():
                    data = json.loads(line)
                    if data.get("success") and data.get("json_parsed") and data.get("case_data"):
                        fout.write(json.dumps(data["case_data"], ensure_ascii=False) + "\n")
                        success_count += 1

    print(f"\n成功病例已保存: {output_file} ({success_count} 条)")

    # 保存统计
    stats_file = os.path.join(args.output, "stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "n": args.n,
                "workers": args.workers,
                "max_tokens": args.max_tokens,
                "model": model_name,
            },
            "stats": stats,
            "final_count": success_count,
        }, f, ensure_ascii=False, indent=2)
    print(f"统计已保存: {stats_file}")


if __name__ == "__main__":
    main()