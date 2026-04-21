#!/usr/bin/env python3
"""
M3模型评判系统（可选双模型）

合成数据质量评估流程：
1. M2模型生成病例数据
2. M3模型评判（可选：+ 通用模型双评判）
3. 双模式时：双模型都通过才算通过；单模式时：M3通过即可

使用方式:
    # 只使用M3模型评判（推荐，节省API调用）
    python scripts/dual_model_judge.py --input results/generated_cases.jsonl

    # 双模型评判（M3 + qwen3.6-plus）
    python scripts/dual_model_judge.py --input results/generated_cases.jsonl --dual

    # 指定M3服务地址
    python scripts/dual_model_judge.py --input results/generated_cases.jsonl --m3-url http://localhost:8100/v1
"""

import os
import sys
import json
import time
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI


# 评判Prompt - 用于M3模型
M3_JUDGE_PROMPT = """你是一位资深医学专家，负责判断合成医疗数据是否符合医学知识。

## 判断标准

请判断以下病例数据是否：
1. 主诉、症状描述符合医学常识
2. 诊断与症状/检查结果逻辑一致
3. 治疗方案针对诊断且医学合理
4. 禁忌项有医学依据，不是随意设定
5. 整体病例可用于医疗AI训练

## 病例数据

{case_data}

## 输出要求

请用XML标签输出判断结果：

如果通过，输出：
<judge_result>passed</judge_result>

如果不通过，输出：
<judge_result>rejected</judge_result>
<reject_reason>简要说明不符合之处</reject_reason>

只输出XML标签，不要输出其他内容。"""


# 评判Prompt - 用于通用模型(qwen3.6-plus)
GENERAL_JUDGE_PROMPT = """你是一位资深医学专家，负责判断合成医疗数据是否符合医学事实。

## 判断标准

请判断以下病例数据是否符合医学事实：
1. 症状描述是否真实可信
2. 诊断逻辑是否正确
3. 治疗方案是否合理
4. 禁忌项是否有医学依据
5. 整体是否有明显医学错误

## 病例数据

{case_data}

## 输出要求

请用XML标签输出判断结果：

如果通过，输出：
<judge_result>passed</judge_result>

如果不通过，输出：
<judge_result>rejected</judge_result>
<reject_reason>简要说明问题</reject_reason>

只输出XML标签，不要输出其他内容。"""


class M3Judger:
    """M3评判器（可选双模型模式）"""

    def __init__(
        self,
        m3_url: str = "http://localhost:8100/v1",
        general_model: str = "qwen3.6-plus",
        general_url: str = "https://coding.dashscope.aliyuncs.com/v1",
        m3_model_name: str = None,
        use_dual: bool = False,
    ):
        """
        初始化评判器

        Args:
            m3_url: M3模型服务地址（本地vLLM）
            general_model: 通用评判模型名称
            general_url: 通用模型API地址
            m3_model_name: M3模型名称（自动获取）
            use_dual: 是否使用双模型模式
        """
        # M3客户端（本地vLLM服务）
        self.m3_client = OpenAI(
            base_url=m3_url,
            api_key="EMPTY",
        )
        self.m3_model_name = m3_model_name or "baichuan-m3"
        self.m3_url = m3_url
        self.use_dual = use_dual

        # 通用模型客户端（仅在双模型模式下使用）
        self.general_client = None
        self.general_model = None

        if use_dual:
            import os
            self.general_client = OpenAI(
                base_url=general_url,
                api_key=os.getenv("DASHSCOPE_API_KEY_CP", "EMPTY"),
            )
            self.general_model = general_model

        # 等待M3服务就绪并获取模型名称
        self._wait_for_m3()

    def _wait_for_m3(self, max_wait: int = 60):
        """等待M3服务就绪"""
        client = OpenAI(base_url=self.m3_url, api_key="EMPTY")

        print(f"等待M3服务: {self.m3_url}")
        for i in range(max_wait):
            try:
                models = client.models.list()
                if models.data:
                    self.m3_model_name = models.data[0].id
                    print(f"M3服务就绪: {self.m3_model_name}")
                    return True
            except:
                time.sleep(2)

        print(f"警告: M3服务未就绪，继续尝试...")
        return False

    def judge_with_m3(self, case_data: Dict) -> Tuple[Dict, float]:
        """
        M3模型评判

        Returns:
            Tuple[Dict, float]: (评判结果, 耗时)
        """
        start_time = time.time()

        try:
            case_json = json.dumps(case_data, ensure_ascii=False, indent=2)
            prompt = M3_JUDGE_PROMPT.format(case_data=case_json)

            response = self.m3_client.chat.completions.create(
                model=self.m3_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=16384,
            )

            latency = time.time() - start_time
            content = response.choices[0].message.content

            # 解析XML
            result = self._parse_xml_response(content)

            if result and "passed" in result:
                return {
                    "success": True,
                    "passed": result.get("passed", False),
                    "reason": result.get("reason", ""),
                    "latency": latency,
                    "raw_output": content[:500],
                }, latency
            else:
                return {
                    "success": False,
                    "error": "XML解析失败",
                    "raw_output": content[:500],
                    "latency": latency,
                }, latency

        except Exception as e:
            latency = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "latency": latency,
            }, latency

    def judge_with_general(self, case_data: Dict) -> Tuple[Dict, float]:
        """
        通用模型评判 (qwen3.6-plus)

        Returns:
            Tuple[Dict, float]: (评判结果, 耗时)
        """
        start_time = time.time()

        try:
            case_json = json.dumps(case_data, ensure_ascii=False, indent=2)
            prompt = GENERAL_JUDGE_PROMPT.format(case_data=case_json)

            response = self.general_client.chat.completions.create(
                model=self.general_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )

            latency = time.time() - start_time
            content = response.choices[0].message.content

            # 解析XML
            result = self._parse_xml_response(content)

            if result and "passed" in result:
                return {
                    "success": True,
                    "passed": result.get("passed", False),
                    "reason": result.get("reason", ""),
                    "latency": latency,
                    "raw_output": content[:500],
                }, latency
            else:
                return {
                    "success": False,
                    "error": "XML解析失败",
                    "raw_output": content[:500],
                    "latency": latency,
                }, latency

        except Exception as e:
            latency = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "latency": latency,
            }, latency

    def _parse_xml_response(self, response: str) -> Dict:
        """解析XML格式响应"""
        # 提取 judge_result
        result_match = re.search(r'<judge_result>(passed|rejected)</judge_result>', response)
        if result_match:
            passed = result_match.group(1) == 'passed'

            # 提取 reject_reason（如果存在）
            reason_match = re.search(r'<reject_reason>(.*?)</reject_reason>', response)
            reason = reason_match.group(1) if reason_match else ""

            return {
                "passed": passed,
                "reason": reason,
            }

        return None

    def dual_judge(self, case_data: Dict) -> Dict:
        """
        评判（单模型或双模型）

        Returns:
            Dict: 包含M3和通用模型（如果启用）的评判结果，以及最终判定
        """
        case_id = case_data.get("case_id", "unknown")

        # M3评判
        m3_result, m3_latency = self.judge_with_m3(case_data)

        m3_passed = (
            m3_result.get("success", False) and
            m3_result.get("passed", False)
        )

        # 如果是单模型模式，直接返回M3结果
        if not self.use_dual:
            return {
                "case_id": case_id,
                "m3_result": m3_result,
                "m3_passed": m3_passed,
                "m3_latency": m3_latency,
                "general_result": None,
                "general_passed": None,
                "general_latency": None,
                "final_passed": m3_passed,
                "pass_reason": "M3模型通过" if m3_passed else f"M3模型拒绝: {m3_result.get('reason', '')}",
                "mode": "single",
            }

        # 双模型模式：通用模型评判
        general_result, general_latency = self.judge_with_general(case_data)

        general_passed = (
            general_result.get("success", False) and
            general_result.get("passed", False)
        )

        # 最终判定：双模型都通过才算通过
        final_passed = m3_passed and general_passed

        return {
            "case_id": case_id,
            "m3_result": m3_result,
            "m3_passed": m3_passed,
            "m3_latency": m3_latency,
            "general_result": general_result,
            "general_passed": general_passed,
            "general_latency": general_latency,
            "final_passed": final_passed,
            "pass_reason": self._get_pass_reason(m3_passed, general_passed, m3_result, general_result),
            "mode": "dual",
        }

    def _get_pass_reason(self, m3_passed: bool, general_passed: bool, m3_result: Dict, general_result: Dict) -> str:
        """获取通过/拒绝原因"""
        if m3_passed and general_passed:
            return "双模型都通过"
        elif not m3_passed and not general_passed:
            m3_reason = m3_result.get("reason", "")
            general_reason = general_result.get("reason", "")
            return f"双模型都拒绝: M3({m3_reason}) 通用({general_reason})"
        elif not m3_passed:
            return f"M3模型拒绝: {m3_result.get('reason', '')}"
        else:
            return f"通用模型拒绝: {general_result.get('reason', '')}"

    def batch_judge(
        self,
        cases: List[Dict],
        max_workers: int = 8,
    ) -> List[Dict]:
        """
        批量评判

        Args:
            cases: 病例数据列表
            max_workers: 并发数（注意：通用模型有速率限制，建议不超过8）

        Returns:
            List[Dict]: 评判结果列表
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.dual_judge, case): idx
                for idx, case in enumerate(cases)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="M3评判"):
                results.append(future.result())

        # 按原始顺序排序
        results.sort(key=lambda x: cases.index(
            next(c for c in cases if c.get("case_id") == x["case_id"])
        ) if x["case_id"] != "unknown" else 0)

        return results


def main():
    parser = argparse.ArgumentParser(description="双模型评判系统")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入数据文件路径 (jsonl)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出结果文件路径",
    )

    parser.add_argument(
        "--dual",
        action="store_true",
        help="启用双模型模式（M3 + 通用模型）",
    )

    parser.add_argument(
        "--m3-url",
        type=str,
        default="http://localhost:8100/v1",
        help="M3模型服务地址",
    )

    parser.add_argument(
        "--general-model",
        type=str,
        default="qwen3.6-plus",
        help="通用评判模型名称（仅双模型模式）",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并发数（单模型建议8-16，双模型建议4）",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="抽样数量（默认全部评判）",
    )

    args = parser.parse_args()

    print("=" * 60)
    mode_str = "双模型评判" if args.dual else "M3单模型评判"
    print(mode_str)
    print("=" * 60)
    print(f"M3服务: {args.m3_url}")
    if args.dual:
        print(f"通用模型: {args.general_model}")
    print(f"输入文件: {args.input}")
    print(f"并发数: {args.workers}")
    print(f"模式: {mode_str}")
    print("=" * 60)

    # 加载数据
    print(f"\n加载数据: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    print(f"加载 {len(cases)} 条数据")

    if args.sample and args.sample < len(cases):
        import random
        cases = random.sample(cases, args.sample)
        print(f"随机抽样 {args.sample} 条")

    # 初始化评判器
    judger = M3Judger(
        m3_url=args.m3_url,
        general_model=args.general_model if args.dual else None,
        use_dual=args.dual,
    )

    # 执行评判
    start_time = time.time()
    results = judger.batch_judge(cases, max_workers=args.workers)
    total_time = time.time() - start_time

    # 统计
    m3_success = sum(1 for r in results if r["m3_result"].get("success", False))
    general_success = sum(1 for r in results if r.get("general_result") and r["general_result"].get("success", False))
    m3_passed = sum(1 for r in results if r["m3_passed"])
    general_passed = sum(1 for r in results if r.get("general_passed", False))
    final_passed = sum(1 for r in results if r["final_passed"])

    # 输出结果
    print("\n" + "=" * 60)
    print("评判结果统计")
    print("=" * 60)
    print(f"总样本数: {len(results)}")
    print(f"\nM3模型:")
    print(f"  评判成功: {m3_success}/{len(results)}")
    print(f"  数据通过: {m3_passed}/{m3_success} ({m3_passed/max(m3_success,1)*100:.1f}%)")

    if args.dual:
        print(f"\n通用模型 ({args.general_model}):")
        print(f"  评判成功: {general_success}/{len(results)}")
        print(f"  数据通过: {general_passed}/{general_success} ({general_passed/max(general_success,1)*100:.1f}%)")
        print(f"\n双模型综合:")
        print(f"  最终通过: {final_passed}/{len(results)} ({final_passed/len(results)*100:.1f}%)")
    else:
        print(f"\n单模型模式:")
        print(f"  最终通过: {final_passed}/{len(results)} ({final_passed/len(results)*100:.1f}%)")

    print(f"\n耗时: {total_time:.1f}s")

    # 通过原因分析
    pass_reasons = {}
    for r in results:
        reason = r["pass_reason"]
        pass_reasons[reason] = pass_reasons.get(reason, 0) + 1

    print("\n通过/拒绝原因分布:")
    for reason, count in sorted(pass_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/len(results)*100:.1f}%)")

    # 保存结果
    if args.output is None:
        input_path = Path(args.input)
        suffix = "dual_judge" if args.dual else "m3_judge"
        args.output = str(input_path.parent / f"{suffix}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # 分离通过和未通过的数据
    passed_cases = [cases[i] for i, r in enumerate(results) if r["final_passed"]]

    output_data = {
        "summary": {
            "total": len(results),
            "m3_success": m3_success,
            "m3_passed": m3_passed,
            "general_success": general_success if args.dual else 0,
            "general_passed": general_passed if args.dual else 0,
            "final_passed": final_passed,
            "pass_rate": final_passed / len(results),
            "pass_reasons": pass_reasons,
            "total_time": total_time,
        },
        "results": results,
        "passed_cases": passed_cases,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {args.output}")

    # 同时保存通过的数据为jsonl
    if passed_cases:
        passed_file = args.output.replace(".json", "_passed.jsonl")
        with open(passed_file, "w", encoding="utf-8") as f:
            for case in passed_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"通过数据已保存: {passed_file} ({len(passed_cases)} 条)")


if __name__ == "__main__":
    main()