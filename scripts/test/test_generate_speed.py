#!/usr/bin/env python3
"""
简单生成速度测试脚本
测试本地 vLLM 服务在指定数据上的生成速度
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_cases(data_path: str, n: int = 100) -> List[Dict]:
    """加载测试数据"""
    cases = []
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cases.append(json.loads(line))
    elif data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            cases = json.load(f)
    return cases[:n]


def build_prompt(case: Dict) -> str:
    """构建诊断 prompt"""
    chief_complaint = case.get("chief_complaint", "")
    history = case.get("history_of_present_illness", "")

    prompt = f"""请根据以下病例信息进行分析和诊断。

主诉：{chief_complaint}

现病史：{history}

请简要分析可能的诊断方向，给出诊断建议。"""

    return prompt


def generate_single(
    client: OpenAI,
    model_name: str,
    prompt: str,
    idx: int,
    max_tokens: int = 8192,
) -> Dict:
    """单次生成"""
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )

        latency = time.time() - start_time

        return {
            "idx": idx,
            "success": True,
            "latency": latency,
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
            "content": response.choices[0].message.content,
        }
    except Exception as e:
        return {
            "idx": idx,
            "success": False,
            "latency": time.time() - start_time,
            "error": str(e),
        }


def run_test(
    client: OpenAI,
    model_name: str,
    prompts: List[str],
    max_workers: int = 8,
    max_tokens: int = 8192,
) -> Dict:
    """运行并发测试"""

    print(f"\n开始生成测试...")
    print(f"  测试样本数: {len(prompts)}")
    print(f"  并发数: {max_workers}")
    print(f"  最大生成长度: {max_tokens}")

    results = {
        "total_time": 0,
        "total_tokens": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "success": 0,
        "failed": 0,
        "latencies": [],
        "details": [],
    }

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_single, client, model_name, prompt, idx, max_tokens): idx
            for idx, prompt in enumerate(prompts)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="生成进度"):
            result = future.result()
            results["details"].append(result)

            if result["success"]:
                results["success"] += 1
                results["latencies"].append(result["latency"])
                results["total_tokens"] += result["total_tokens"]
                results["total_input_tokens"] += result["input_tokens"]
                results["total_output_tokens"] += result["output_tokens"]
            else:
                results["failed"] += 1

    results["total_time"] = time.time() - start_time

    # 计算统计指标
    if results["success"] > 0:
        results["avg_latency"] = sum(results["latencies"]) / results["success"]
        results["tokens_per_second"] = results["total_output_tokens"] / results["total_time"]
        results["samples_per_second"] = results["success"] / results["total_time"]
        results["avg_output_tokens"] = results["total_output_tokens"] / results["success"]

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="生成速度测试")
    parser.add_argument("--url", type=str, default="http://localhost:8100/v1", help="服务地址")
    parser.add_argument("--data", type=str, default="output/generate_2000/sampled_seeds.jsonl", help="数据路径")
    parser.add_argument("--n", type=int, default=100, help="样本数量")
    parser.add_argument("--workers", type=int, default=8, help="并发数")
    parser.add_argument("--max-tokens", type=int, default=8192, help="最大生成长度")
    parser.add_argument("--output", type=str, default="results/baichuan_test", help="结果输出目录")

    args = parser.parse_args()

    print(f"{'='*60}")
    print("Baichuan M2 生成速度测试")
    print(f"{'='*60}")
    print(f"服务地址: {args.url}")
    print(f"数据路径: {args.data}")
    print(f"样本数量: {args.n}")
    print(f"并发数: {args.workers}")
    print(f"{'='*60}")

    # 创建客户端
    client = OpenAI(base_url=args.url, api_key="EMPTY")

    # 获取模型名称
    models = client.models.list()
    model_name = models.data[0].id
    print(f"模型名称: {model_name}")

    # 加载测试数据
    print(f"\n加载测试数据...")
    cases = load_cases(args.data, args.n)
    prompts = [build_prompt(c) for c in cases]
    print(f"已加载 {len(prompts)} 条测试数据")

    # 运行测试
    results = run_test(
        client=client,
        model_name=model_name,
        prompts=prompts,
        max_workers=args.workers,
        max_tokens=args.max_tokens,
    )

    # 添加元信息
    results["model_name"] = model_name
    results["test_samples"] = args.n
    results["max_workers"] = args.workers
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 输出结果
    print(f"\n{'='*60}")
    print("测试结果")
    print(f"{'='*60}")
    print(f"成功: {results['success']}/{results['success'] + results['failed']}")
    print(f"失败: {results['failed']}")
    print(f"\n时间统计:")
    print(f"  总耗时: {results['total_time']:.2f}s")
    print(f"  平均延迟: {results['avg_latency']:.2f}s")
    print(f"  吞吐速度: {results['samples_per_second']:.2f} samples/s")
    print(f"\nToken统计:")
    print(f"  输入Token: {results['total_input_tokens']}")
    print(f"  输出Token: {results['total_output_tokens']}")
    print(f"  平均输出: {results['avg_output_tokens']:.1f} tokens")
    print(f"  生成速度: {results['tokens_per_second']:.2f} tokens/s")
    print(f"{'='*60}")

    # 保存结果
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"speed_test_{timestamp}.json"
    filepath = os.path.join(args.output, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {filepath}")


if __name__ == "__main__":
    main()