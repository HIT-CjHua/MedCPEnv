#!/usr/bin/env python3
"""
Test different max_tokens settings for judge_single_case
"""

import json
import time
from openai import OpenAI
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
    # Method 1: Check last line
    last_line = content.strip().split('\n')[-1].upper().strip()
    if last_line == "PASS":
        return True, content
    elif last_line == "FAIL":
        return False, content

    # Method 2: Find last PASS/FAIL keyword
    content_upper = content.upper()
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


def judge_single_case(client, model_name, case_data, max_tokens):
    """Judge a single case with given max_tokens"""
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

        return {
            "success": True,
            "latency": latency,
            "passed": passed,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "content_length": len(content),
            "content": content,
            "truncated": content.endswith("...") or len(content) >= max_tokens * 4 * 0.9,
        }
    except Exception as e:
        return {
            "success": False,
            "latency": time.time() - start_time,
            "error": str(e),
        }


def test_max_tokens(client, model_name, cases, max_tokens_values):
    """Test different max_tokens settings"""
    results = {}

    for max_tokens in max_tokens_values:
        print(f"\n{'='*60}")
        print(f"Testing max_tokens = {max_tokens}")
        print(f"{'='*60}")

        test_results = {
            "max_tokens": max_tokens,
            "total": len(cases),
            "success": 0,
            "failed": 0,
            "passed": 0,
            "truncated": 0,
            "latencies": [],
            "output_tokens": [],
            "content_lengths": [],
            "pass_rate": 0,
        }

        for case in tqdm(cases, desc=f"max_tokens={max_tokens}"):
            result = judge_single_case(client, model_name, case, max_tokens)

            if result["success"]:
                test_results["success"] += 1
                test_results["latencies"].append(result["latency"])
                test_results["output_tokens"].append(result["output_tokens"])
                test_results["content_lengths"].append(result["content_length"])

                if result["passed"]:
                    test_results["passed"] += 1

                if result.get("truncated"):
                    test_results["truncated"] += 1
            else:
                test_results["failed"] += 1

        if test_results["success"] > 0:
            test_results["pass_rate"] = test_results["passed"] / test_results["success"] * 100
            test_results["avg_latency"] = sum(test_results["latencies"]) / test_results["success"]
            test_results["avg_output_tokens"] = sum(test_results["output_tokens"]) / test_results["success"]
            test_results["avg_content_length"] = sum(test_results["content_lengths"]) / test_results["success"]

        results[max_tokens] = test_results

        print(f"\n[Results for max_tokens={max_tokens}]")
        print(f"  Success: {test_results['success']}/{test_results['total']}")
        print(f"  Passed: {test_results['passed']}/{test_results['success']} ({test_results['pass_rate']:.1f}%)")
        print(f"  Truncated: {test_results['truncated']}")
        print(f"  Avg latency: {test_results.get('avg_latency', 0):.2f}s")
        print(f"  Avg output tokens: {test_results.get('avg_output_tokens', 0):.1f}")
        print(f"  Avg content length: {test_results.get('avg_content_length', 0):.1f} chars")

    return results


def main():
    # Load 30 cases
    cases_file = "output/m2_mimic_shard_0/medical_cases.jsonl"
    cases = []
    with open(cases_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 30:
                break
            cases.append(json.loads(line.strip()))

    print(f"Loaded {len(cases)} cases from {cases_file}")

    # Connect to M3
    client = OpenAI(base_url="http://localhost:8100/v1", api_key="EMPTY")
    models = client.models.list()
    model_name = models.data[0].id
    print(f"Model: {model_name}")

    # Test different max_tokens values
    max_tokens_values = [1024, 2048, 4096, 8192, 16384]

    results = test_max_tokens(client, model_name, cases, max_tokens_values)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'max_tokens':<12} {'pass_rate':<12} {'truncated':<12} {'avg_tokens':<12} {'avg_latency':<12}")
    print("-" * 60)
    for mt in max_tokens_values:
        r = results[mt]
        print(f"{mt:<12} {r['pass_rate']:<12.1f}% {r['truncated']:<12} {r.get('avg_output_tokens', 0):<12.1f} {r.get('avg_latency', 0):<12.2f}s")

    # Save results
    output_file = "results/max_tokens_test.json"
    import os
    os.makedirs("results", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        # Convert to serializable format
        serializable_results = {}
        for mt, r in results.items():
            serializable_results[str(mt)] = {
                "max_tokens": r["max_tokens"],
                "total": r["total"],
                "success": r["success"],
                "failed": r["failed"],
                "passed": r["passed"],
                "truncated": r["truncated"],
                "pass_rate": r["pass_rate"],
                "avg_latency": r.get("avg_latency", 0),
                "avg_output_tokens": r.get("avg_output_tokens", 0),
                "avg_content_length": r.get("avg_content_length", 0),
            }
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()