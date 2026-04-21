#!/usr/bin/env python3
"""
Test M2 model as judge with different max_tokens settings
"""

import json
import time
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

JUDGE_PROMPT = """/no_think
You are a senior clinical expert responsible for determining whether a case can be used for model evaluation.

[Evaluation Criteria]
1. Data structure complete (contains required fields: case_id, chief_complaint, subjective, objective, ground_truth)
2. Information sufficient (subjective and objective each have at least 2 items)
3. No obvious conflicts between symptoms, examinations, diagnoses, and treatments
4. Sample has evaluation value

[Output Requirement]
Output only one word: PASS or FAIL"""


def parse_pass_fail(content: str) -> tuple:
    """Parse PASS/FAIL from response content"""
    content_upper = content.upper()

    # Method 1: Check last line
    lines = content.strip().split('\n')
    for line in reversed(lines):
        line_upper = line.strip().upper()
        if line_upper == "PASS":
            return True, content
        elif line_upper == "FAIL":
            return False, content

    # Method 2: Find last PASS/FAIL keyword
    pass_positions = []
    fail_positions = []

    for i in range(len(content_upper)):
        if content_upper[i:i+4] == "PASS" and (i+4 >= len(content_upper) or not content_upper[i+4].isalpha()):
            pass_positions.append(i)
        if content_upper[i:i+4] == "FAIL" and (i+4 >= len(content_upper) or not content_upper[i+4].isalpha()):
            fail_positions.append(i)

    all_positions = [(p, "PASS") for p in pass_positions] + [(p, "FAIL") for p in fail_positions]
    if all_positions:
        last_result = max(all_positions, key=lambda x: x[0])[1]
        return (last_result == "PASS"), content

    return False, content


def judge_single_case(client, model_name, case_data, idx, max_tokens):
    """Judge a single case"""
    prompt = f"{JUDGE_PROMPT}\n\n[Case Data]\n{json.dumps(case_data, ensure_ascii=False, indent=2)}"

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )

        latency = time.time() - start_time
        content = response.choices[0].message.content

        passed, reason = parse_pass_fail(content)

        # Check if truncated
        truncated = False
        if response.usage:
            if response.usage.completion_tokens >= max_tokens * 0.95:
                truncated = True

        return {
            "idx": idx,
            "success": True,
            "latency": latency,
            "passed": passed,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "content_length": len(content),
            "truncated": truncated,
        }
    except Exception as e:
        return {
            "idx": idx,
            "success": False,
            "latency": time.time() - start_time,
            "error": str(e),
        }


def run_judge_test(client, model_name, cases, max_tokens, max_workers):
    """Run judge test with given configuration"""
    print(f"\n[Judge Test] max_tokens={max_tokens}, workers={max_workers}, n={len(cases)}")

    results = {
        "max_tokens": max_tokens,
        "workers": max_workers,
        "total": len(cases),
        "success": 0,
        "failed": 0,
        "passed": 0,
        "truncated": 0,
        "latencies": [],
        "output_tokens": [],
        "total_time": 0,
    }

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(judge_single_case, client, model_name, case, idx, max_tokens): idx
            for idx, case in enumerate(cases)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Judge(mt={max_tokens})"):
            result = future.result()

            if result["success"]:
                results["success"] += 1
                results["latencies"].append(result["latency"])
                results["output_tokens"].append(result["output_tokens"])

                if result["passed"]:
                    results["passed"] += 1

                if result.get("truncated"):
                    results["truncated"] += 1
            else:
                results["failed"] += 1

    results["total_time"] = time.time() - start_time

    if results["success"] > 0:
        results["pass_rate"] = results["passed"] / results["success"] * 100
        results["avg_latency"] = sum(results["latencies"]) / results["success"]
        results["avg_output_tokens"] = sum(results["output_tokens"]) / results["success"]
        results["samples_per_second"] = results["success"] / results["total_time"]
    else:
        results["pass_rate"] = 0
        results["avg_latency"] = 0
        results["avg_output_tokens"] = 0
        results["samples_per_second"] = 0

    return results


def main():
    parser = argparse.ArgumentParser(description="Test M2 as judge with different max_tokens")
    parser.add_argument("--url", type=str, default="http://localhost:8200/v1", help="M2 service URL")
    parser.add_argument("--cases-file", type=str, default="output/m2_mimic_shard_0/medical_cases.jsonl", help="Cases file")
    parser.add_argument("--limit", type=int, default=100, help="Number of cases to test")
    parser.add_argument("--workers", type=int, default=64, help="Concurrency")
    parser.add_argument("--output", type=str, default="results/m2_judge_test", help="Output directory")

    args = parser.parse_args()

    # Load cases
    cases = []
    with open(args.cases_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            cases.append(json.loads(line.strip()))

    print(f"Loaded {len(cases)} cases from {args.cases_file}")

    # Connect to M2
    client = OpenAI(base_url=args.url, api_key="EMPTY")

    print(f"\nWaiting for M2 service at {args.url}...")
    for i in range(30):
        try:
            models = client.models.list()
            model_name = models.data[0].id
            print(f"Ready: {model_name}")
            break
        except:
            time.sleep(3)

    print(f"\n{'='*60}")
    print("M2 Judge Test - Different max_tokens")
    print(f"{'='*60}")
    print(f"URL: {args.url}")
    print(f"Model: {model_name}")
    print(f"Cases: {len(cases)}")
    print(f"Workers: {args.workers}")
    print(f"{'='*60}")

    # Test different max_tokens
    max_tokens_values = [512, 1024, 2048, 4096, 8192]
    all_results = {}

    for max_tokens in max_tokens_values:
        result = run_judge_test(client, model_name, cases, max_tokens, args.workers)
        all_results[max_tokens] = result

        print(f"\n[Results for max_tokens={max_tokens}]")
        print(f"  Success: {result['success']}/{result['total']}")
        print(f"  Passed: {result['passed']}/{result['success']} ({result['pass_rate']:.1f}%)")
        print(f"  Truncated: {result['truncated']}")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Avg latency: {result['avg_latency']:.2f}s")
        print(f"  Avg output tokens: {result['avg_output_tokens']:.1f}")
        print(f"  Throughput: {result['samples_per_second']:.2f} samples/s")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'max_tokens':<12} {'pass_rate':<12} {'truncated':<12} {'avg_tokens':<12} {'avg_latency':<12} {'throughput':<12}")
    print("-" * 72)
    for mt in max_tokens_values:
        r = all_results[mt]
        print(f"{mt:<12} {r['pass_rate']:<12.1f}% {r['truncated']:<12} {r['avg_output_tokens']:<12.1f} {r['avg_latency']:<12.2f}s {r['samples_per_second']:<12.2f}")

    # Save results
    import os
    os.makedirs(args.output, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output, f"m2_judge_{timestamp}.json")

    # Convert to serializable
    serializable = {}
    for mt, r in all_results.items():
        serializable[str(mt)] = {
            "max_tokens": r["max_tokens"],
            "workers": r["workers"],
            "total": r["total"],
            "success": r["success"],
            "failed": r["failed"],
            "passed": r["passed"],
            "truncated": r["truncated"],
            "pass_rate": r["pass_rate"],
            "total_time": r["total_time"],
            "avg_latency": r["avg_latency"],
            "avg_output_tokens": r["avg_output_tokens"],
            "samples_per_second": r["samples_per_second"],
        }

    serializable["config"] = {
        "url": args.url,
        "model": model_name,
        "cases_file": args.cases_file,
        "limit": args.limit,
        "workers": args.workers,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()