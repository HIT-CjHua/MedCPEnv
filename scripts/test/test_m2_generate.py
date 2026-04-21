#!/usr/bin/env python3
"""
Baichuan M2 生成速度测试
生成符合schema的病例数据并测试生成速度
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_case_prompt(idx: int) -> str:
    """生成病例生成prompt"""
    idx_str = f"{idx:04d}"

    prompt = """请生成一个完整的医疗病例数据，严格按照以下JSON格式输出：

```json
{
  "case_id": "test_""" + idx_str + """",
  "difficulty": "medium",
  "tags": ["内科", "常见病"],
  "chief_complaint": "主诉内容（患者主动描述的症状）",
  "subjective": [
    {
      "keywords": ["问诊触发关键词"],
      "content": "问诊回答内容",
      "necessity": true
    }
  ],
  "objective": [
    {
      "keywords": ["检查触发关键词"],
      "content": "检查结果",
      "necessity": true
    }
  ],
  "ground_truth": {
    "diagnosis": ["诊断1", "诊断2"],
    "treatment": ["治疗方案1", "治疗方案2"],
    "avoid": ["禁忌项1", "禁忌项2"]
  },
  "source": "synthetic"
}
```

要求：
1. 病例要有合理的医疗逻辑，主诉、问诊、检查、诊断、治疗要连贯
2. subjective包含至少3个问诊项（现病史、既往史、家族史等）
3. objective包含至少2个检查项（体格检查、实验室检查等）
4. ground_truth的诊断要明确，治疗方案要具体，禁忌项要有医学依据
5. 必须输出完整的JSON，不要遗漏任何字段
6. 只输出JSON，不要有其他文字

请生成病例 #""" + idx_str + """："""

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
        content = response.choices[0].message.content

        # 尝试解析JSON
        json_data = None
        try:
            # 提取JSON部分
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            json_data = json.loads(json_str)
        except:
            pass

        return {
            "idx": idx,
            "success": True,
            "latency": latency,
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
            "content": content,
            "json_parsed": json_data is not None,
            "json_data": json_data,
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
    n: int,
    max_workers: int = 16,
    max_tokens: int = 8192,
) -> Dict:
    """运行并发测试"""

    print(f"\n开始生成测试...")
    print(f"  测试样本数: {n}")
    print(f"  并发数: {max_workers}")
    print(f"  最大生成长度: {max_tokens}")

    prompts = [generate_case_prompt(i) for i in range(n)]

    results = {
        "total_time": 0,
        "total_tokens": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "success": 0,
        "failed": 0,
        "json_parsed": 0,
        "latencies": [],
        "cases": [],
    }

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_single, client, model_name, prompt, idx, max_tokens): idx
            for idx, prompt in enumerate(prompts)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="生成进度"):
            result = future.result()

            if result["success"]:
                results["success"] += 1
                results["latencies"].append(result["latency"])
                results["total_tokens"] += result["total_tokens"]
                results["total_input_tokens"] += result["input_tokens"]
                results["total_output_tokens"] += result["output_tokens"]

                if result["json_parsed"]:
                    results["json_parsed"] += 1
                    results["cases"].append(result["json_data"])
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
    parser = argparse.ArgumentParser(description="Baichuan M2 病例生成速度测试")
    parser.add_argument("--url", type=str, default="http://localhost:8100/v1", help="服务地址")
    parser.add_argument("--n", type=int, default=50, help="样本数量")
    parser.add_argument("--workers", type=int, default=16, help="并发数")
    parser.add_argument("--max-tokens", type=int, default=8192, help="最大生成长度")
    parser.add_argument("--output", type=str, default="results/baichuan_test", help="结果输出目录")

    args = parser.parse_args()

    print(f"{'='*60}")
    print("Baichuan M2 病例生成速度测试")
    print(f"{'='*60}")
    print(f"服务地址: {args.url}")
    print(f"样本数量: {args.n}")
    print(f"并发数: {args.workers}")
    print(f"最大生成长度: {args.max_tokens}")
    print(f"{'='*60}")

    # 创建客户端并获取模型名称
    client = OpenAI(base_url=args.url, api_key="EMPTY")

    # 等待服务就绪
    print("\n等待服务就绪...")
    for i in range(60):
        try:
            models = client.models.list()
            model_name = models.data[0].id
            print(f"模型已就绪: {model_name}")
            break
        except Exception as e:
            if i < 59:
                time.sleep(5)
            else:
                print(f"服务未就绪: {e}")
                return

    # 运行测试
    results = run_test(
        client=client,
        model_name=model_name,
        n=args.n,
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
    print(f"JSON解析成功: {results['json_parsed']}")
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

    # 保存统计结果
    stats_file = os.path.join(args.output, f"speed_test_{timestamp}.json")
    stats_to_save = {k: v for k, v in results.items() if k != "cases"}
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_to_save, f, ensure_ascii=False, indent=2)
    print(f"\n统计结果已保存: {stats_file}")

    # 保存生成的病例数据
    if results["cases"]:
        cases_file = os.path.join(args.output, f"generated_cases_{timestamp}.jsonl")
        with open(cases_file, "w", encoding="utf-8") as f:
            for case in results["cases"]:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"病例数据已保存: {cases_file}")


if __name__ == "__main__":
    main()