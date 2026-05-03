# MedAgent/scripts/eval.py

"""
MedAgent 评测脚本

功能：
1. 加载病例数据
2. 运行 Agent 进行诊断
3. 使用 Judger 评估结果
4. 生成评测报告

使用方式：
    python scripts/eval.py --data data/cases.jsonl --output results/eval_report.json

    # 指定模型
    python scripts/eval.py --model qwen3.5-plus --n 100

    # 使用知识库
    python scripts/eval.py --kb data/knowledge_dataset/ResponseMed.json
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import MedicalCase
from src.medagent import LLMClient, MedAgent, Judger, EvalResult, EfficiencyStats, CostEvaluator
from src.medagent.knowledge_tool_v2 import KeywordKnowledgeBase
from src.medagent.llm import api_counter


def load_cases(data_path: str) -> List[MedicalCase]:
    """加载病例数据"""
    cases = []

    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    cases.append(MedicalCase.dict_to_case(data))

    elif data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
            for data in data_list:
                cases.append(MedicalCase.dict_to_case(data))

    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    return cases


def run_single_case(
    agent: MedAgent,
    judger: Judger,
    case: MedicalCase,
    verbose: bool = False,
) -> EvalResult:
    """运行单个病例的评测"""

    if verbose:
        print(f"\n{'='*60}")
        print(f"病例ID: {case.case_id}")
        print(f"主诉: {case.chief_complaint}")
        print(f"{'='*60}")

    # 运行 Agent
    result = agent.run()

    # 评估
    eval_result = judger.evaluate(
        case_id=case.case_id,
        chief_complaint=case.chief_complaint,
        ground_truth={
            "diagnosis": case.ground_truth.diagnosis,
            "treatment": case.ground_truth.treatment,
            "avoid": case.ground_truth.avoid,
        },
        trajectory=result.get("trajectory", []),
        agent_diagnosis=result.get("diagnosis", ""),
        agent_treatment=result.get("treatment", ""),
    )

    if verbose:
        print(f"\n诊断结果: {result.get('diagnosis', '')}")
        print(f"治疗建议: {result.get('treatment', '')}")
        print(f"标准诊断: {case.ground_truth.diagnosis}")
        print(f"评分: {eval_result.total_score:.1f}/10")

    return eval_result


def run_evaluation(
    cases: List[MedicalCase],
    llm_client: LLMClient,
    knowledge_base: Optional[KnowledgeBase],
    judger: Judger,
    max_cases: Optional[int] = None,
    max_steps: int = 20,
    top_k: int = 10,
    max_workers: int = 16,
    verbose: bool = False,
    checkpoint_file: Optional[str] = None,
    checkpoint_every: int = 10,
) -> List[EvalResult]:
    """
    运行完整评测（并发 + checkpoint）

    Args:
        checkpoint_file: checkpoint 文件路径，支持断点续跑
        checkpoint_every: 每 N 条保存一次 checkpoint
    """

    if max_cases:
        cases = cases[:max_cases]

    # 加载已处理的结果（断点续跑）
    processed_ids = set()
    results_dict = {}

    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"  从 checkpoint 恢复: {checkpoint_file}")
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        case_id = data.get("case_id")
                        if case_id:
                            processed_ids.add(case_id)
                            results_dict[case_id] = EvalResult(
                                case_id=case_id,
                                diagnosis_correct=data.get("diagnosis_correct", False),
                                diagnosis_score=data.get("diagnosis_score", 0),
                                diagnosis_reason=data.get("diagnosis_reason", ""),
                                treatment_correct=data.get("treatment_correct", False),
                                treatment_score=data.get("treatment_score", 0),
                                treatment_reason=data.get("treatment_reason", ""),
                                avoid_violated=data.get("avoid_violated", False),
                                avoid_score=data.get("avoid_score", 0),
                                avoid_reason=data.get("avoid_reason", ""),
                                avoid_violations=data.get("avoid_violations", []),
                                total_score=data.get("total_score", 0),
                                trajectory=data.get("trajectory", []),
                                ground_truth=data.get("ground_truth", {}),
                                agent_diagnosis=data.get("agent_diagnosis", ""),
                                agent_treatment=data.get("agent_treatment", ""),
                                total_cost=data.get("total_cost", 0),
                            )
                            # 恢复efficiency字段
                            eff_data = data.get("efficiency", {})
                            results_dict[case_id].efficiency.total_steps = eff_data.get("total_steps", 0)
                            results_dict[case_id].efficiency.ask_count = eff_data.get("ask_count", 0)
                            results_dict[case_id].efficiency.exam_count = eff_data.get("exam_count", 0)
                            results_dict[case_id].efficiency.knowledge_count = eff_data.get("knowledge_count", 0)
                            results_dict[case_id].efficiency.exam_items = eff_data.get("exam_items", 0)
                            results_dict[case_id].efficiency.ask_items = eff_data.get("ask_items", 0)
                            results_dict[case_id].efficiency.total_tokens = eff_data.get("total_tokens", 0)
                            results_dict[case_id].efficiency.total_latency = eff_data.get("total_latency", 0)
                            results_dict[case_id].efficiency.avg_tokens_per_step = eff_data.get("avg_tokens_per_step", 0)
                            results_dict[case_id].efficiency.avg_latency_per_step = eff_data.get("avg_latency_per_step", 0)
                            results_dict[case_id].efficiency.tokens_per_second = eff_data.get("tokens_per_second", 0)
            print(f"  已恢复 {len(processed_ids)} 条结果")
        except Exception as e:
            print(f"  checkpoint 加载失败: {e}")

    # 过滤未处理的病例
    remaining_cases = [c for c in cases if c.case_id not in processed_ids]
    print(f"  剩余待处理: {len(remaining_cases)}/{len(cases)}")

    if not remaining_cases:
        print("  所有病例已处理完成")
        return [results_dict.get(c.case_id) for c in cases]

    # 保存 checkpoint 的函数
    def save_checkpoint(batch_results: List[EvalResult]):
        if not checkpoint_file:
            return
        try:
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            mode = "a" if os.path.exists(checkpoint_file) else "w"
            with open(checkpoint_file, mode, encoding="utf-8") as f:
                for r in batch_results:
                    f.write(json.dumps({
                        "case_id": r.case_id,
                        "diagnosis_correct": r.diagnosis_correct,
                        "diagnosis_score": r.diagnosis_score,
                        "diagnosis_reason": r.diagnosis_reason,
                        "treatment_correct": r.treatment_correct,
                        "treatment_score": r.treatment_score,
                        "treatment_reason": r.treatment_reason,
                        "avoid_violated": r.avoid_violated,
                        "avoid_score": r.avoid_score,
                        "avoid_reason": r.avoid_reason,
                        "avoid_violations": r.avoid_violations,
                        "total_score": r.total_score,
                        "trajectory": r.trajectory,
                        "ground_truth": r.ground_truth,
                        "agent_diagnosis": r.agent_diagnosis,
                        "agent_treatment": r.agent_treatment,
                        "total_cost": r.total_cost,
                        "efficiency": {
                            "total_steps": r.efficiency.total_steps,
                            "ask_count": r.efficiency.ask_count,
                            "exam_count": r.efficiency.exam_count,
                            "knowledge_count": r.efficiency.knowledge_count,
                            "exam_items": r.efficiency.exam_items,
                            "ask_items": r.efficiency.ask_items,
                            "total_tokens": r.efficiency.total_tokens,
                            "total_latency": r.efficiency.total_latency,
                            "avg_tokens_per_step": r.efficiency.avg_tokens_per_step,
                            "avg_latency_per_step": r.efficiency.avg_latency_per_step,
                            "tokens_per_second": r.efficiency.tokens_per_second,
                        },
                    }, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  checkpoint 保存失败: {e}")

    def process_case(idx: int, case: MedicalCase) -> tuple:
        """处理单个病例"""
        try:
            # 为每个病例创建新的 Agent
            agent = MedAgent(
                llm_client=llm_client,
                case=case,
                knowledge_base=knowledge_base,
                max_steps=max_steps,
                top_k=top_k,
                verbose=verbose,
            )

            eval_result = run_single_case(agent, judger, case, verbose)
            return idx, case.case_id, eval_result

        except Exception as e:
            print(f"\n[Error] Case {case.case_id} failed: {e}")
            return idx, case.case_id, EvalResult(
                case_id=case.case_id,
                total_score=0,
                trajectory=[],
                ground_truth=case.case_to_dict().get("ground_truth", {}),
            )

    # 并发执行
    batch_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_case, idx, case): idx
            for idx, case in enumerate(remaining_cases)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="评测进度"):
            idx, case_id, eval_result = future.result()
            results_dict[case_id] = eval_result
            batch_results.append(eval_result)

            # 定期保存 checkpoint
            if len(batch_results) >= checkpoint_every:
                save_checkpoint(batch_results)
                batch_results.clear()

            # 定期打印API调用统计
            total_processed = len(processed_ids) + len(results_dict) - len(batch_results)
            if total_processed % 50 == 0 and total_processed > 0:
                status = api_counter.get_status()
                print(f"\n  [API统计] 已处理: {total_processed}, 总调用: {status['total_calls']}, "
                      f"速率: {status['calls_per_hour']:.0f}次/小时")

    # 保存剩余结果
    if batch_results:
        save_checkpoint(batch_results)

    # 按原始顺序返回结果
    return [results_dict.get(c.case_id) for c in cases]


def generate_report(
    results: List[EvalResult],
    output_dir: str,
    model_name: str,
    timestamp: str,
    enable_cost: bool = False,
) -> Dict[str, Any]:
    """生成评测报告"""

    # 计算统计
    total = len(results)
    if total == 0:
        return {}

    diagnosis_correct = sum(1 for r in results if r.diagnosis_correct)
    treatment_correct = sum(1 for r in results if r.treatment_correct)
    avoid_violated = sum(1 for r in results if r.avoid_violated)

    avg_scores = {
        "diagnosis": sum(r.diagnosis_score for r in results) / total,
        "treatment": sum(r.treatment_score for r in results) / total,
        "avoid": sum(r.avoid_score for r in results) / total,
        "total": sum(r.total_score for r in results) / total,
    }

    # 效率统计
    avg_steps = sum(r.efficiency.total_steps for r in results) / total
    avg_exam_items = sum(r.efficiency.exam_items for r in results) / total
    avg_tokens = sum(r.efficiency.total_tokens for r in results) / total
    avg_latency = sum(r.efficiency.total_latency for r in results) / total
    avg_tokens_per_second = sum(r.efficiency.tokens_per_second for r in results) / total

    # 分数分布 (1-5分制)
    score_distribution = {
        "excellent_4_5": sum(1 for r in results if r.total_score >= 4),
        "good_3_4": sum(1 for r in results if 3 <= r.total_score < 4),
        "medium_2_3": sum(1 for r in results if 2 <= r.total_score < 3),
        "poor_1_2": sum(1 for r in results if r.total_score < 2),
    }

    # 费用统计
    cost_stats = None
    if enable_cost:
        valid_costs = [r.total_cost for r in results if r.total_cost > 0]
        if valid_costs:
            cost_stats = {
                "avg_cost": sum(valid_costs) / len(valid_costs),
                "max_cost": max(valid_costs),
                "min_cost": min(valid_costs),
                "total_cost": sum(valid_costs),
            }

    report = {
        "meta": {
            "model": model_name,
            "timestamp": timestamp,
            "total_cases": total,
            "enable_cost": enable_cost,
        },
        "summary": {
            "diagnosis": {
                "accuracy": diagnosis_correct / total,
                "avg_score": avg_scores["diagnosis"],
            },
            "treatment": {
                "accuracy": treatment_correct / total,
                "avg_score": avg_scores["treatment"],
            },
            "safety": {
                "violation_rate": avoid_violated / total,
                "avg_score": avg_scores["avoid"],
            },
            "efficiency": {
                "avg_steps": avg_steps,
                "avg_exam_items": avg_exam_items,
                "avg_tokens": avg_tokens,
                "avg_latency": avg_latency,
                "avg_tokens_per_second": avg_tokens_per_second,
            },
            "total_avg_score": avg_scores["total"],
        },
        "cost_stats": cost_stats,
        "score_distribution": score_distribution,
        "details": [
            {
                "case_id": r.case_id,
                "diagnosis_correct": r.diagnosis_correct,
                "diagnosis_score": r.diagnosis_score,
                "diagnosis_reason": r.diagnosis_reason,
                "treatment_correct": r.treatment_correct,
                "treatment_score": r.treatment_score,
                "treatment_reason": r.treatment_reason,
                "avoid_violated": r.avoid_violated,
                "avoid_score": r.avoid_score,
                "avoid_reason": r.avoid_reason,
                "total_score": r.total_score,
                "efficiency": {
                    "total_steps": r.efficiency.total_steps,
                    "exam_items": r.efficiency.exam_items,
                },
                "total_cost": r.total_cost if enable_cost else None,
            }
            for r in results
        ],
    }

    # 保存报告
    os.makedirs(output_dir, exist_ok=True)

    report_file = os.path.join(output_dir, f"eval_report_{timestamp}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 保存简要结果
    summary_file = os.path.join(output_dir, f"eval_summary_{timestamp}.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"MedAgent 评测报告\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"时间: {timestamp}\n")
        f.write(f"病例数: {total}\n\n")
        f.write(f"{'='*50}\n")
        f.write(f"评测结果\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"诊断准确率: {report['summary']['diagnosis']['accuracy']*100:.1f}%\n")
        f.write(f"诊断平均分: {avg_scores['diagnosis']:.2f}/5\n")
        f.write(f"治疗准确率: {report['summary']['treatment']['accuracy']*100:.1f}%\n")
        f.write(f"治疗平均分: {avg_scores['treatment']:.2f}/5\n")
        f.write(f"禁忌违反率: {report['summary']['safety']['violation_rate']*100:.1f}%\n")
        f.write(f"安全平均分: {avg_scores['avoid']:.2f}/5\n\n")
        f.write(f"效率统计:\n")
        f.write(f"  平均轮次: {avg_steps:.1f}\n")
        f.write(f"  平均检查项: {avg_exam_items:.1f}\n")
        f.write(f"  平均生成token: {avg_tokens:.0f}\n")
        f.write(f"  平均生成耗时: {avg_latency:.2f}s\n")
        f.write(f"  平均生成速度: {avg_tokens_per_second:.1f} tokens/s\n\n")
        f.write(f"综合平均分: {avg_scores['total']:.2f}/5\n\n")

        if enable_cost and cost_stats:
            f.write(f"费用评估:\n")
            f.write(f"  平均费用: {cost_stats['avg_cost']:.2f}元\n")
            f.write(f"  最高费用: {cost_stats['max_cost']:.2f}元\n")
            f.write(f"  最低费用: {cost_stats['min_cost']:.2f}元\n")
            f.write(f"  总费用: {cost_stats['total_cost']:.2f}元\n\n")

        f.write(f"分数分布:\n")
        f.write(f"  优秀 (4-5): {score_distribution['excellent_4_5']}\n")
        f.write(f"  良好 (3-4):  {score_distribution['good_3_4']}\n")
        f.write(f"  中等 (2-3):  {score_distribution['medium_2_3']}\n")
        f.write(f"  较差 (1-2):  {score_distribution['poor_1_2']}\n")

    print(f"\n报告已保存:")
    print(f"  {report_file}")
    print(f"  {summary_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description="MedAgent 评测脚本")

    parser.add_argument(
        "--data",
        type=str,
        default="output/generate_2000/merged_selected.jsonl",
        help="病例数据路径 (jsonl/json)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/eval",
        help="输出目录",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5-plus",
        help="被评测的 Agent LLM 模型名称",
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default="baichuan-m2",
        help="评测器使用的 Judge 模型名称（默认百川M2）",
    )

    parser.add_argument(
        "--judge-url",
        type=str,
        default="http://localhost:8200/v1",
        help="Judge 模型的 API 地址（默认本地M2服务）",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="评测病例数量 (None 表示全部)",
    )

    parser.add_argument(
        "--kb",
        type=str,
        default="data/knowledge_dataset/ResponseMed.json",
        help="知识库路径 (ResponseMed.json)",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Agent 最大步数",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="知识库检索 top_k 数量",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="并发评测线程数",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="每 N 条保存一次 checkpoint",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细信息",
    )

    parser.add_argument(
        "--no-kb",
        action="store_true",
        help="不使用知识库",
    )

    parser.add_argument(
        "--enable-cost",
        action="store_true",
        help="启用费用评估",
    )

    args = parser.parse_args()

    # 时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("MedAgent 评测系统")
    print("=" * 60)
    print(f"数据: {args.data}")
    print(f"模型: {args.model}")
    print(f"知识库: {'不使用' if args.no_kb else args.kb}")
    print(f"病例数: {'全部' if args.n is None else args.n}")
    print(f"最大轮次: {args.max_steps}")
    print(f"Top-K: {args.topk}")
    print("=" * 60)

    # 加载病例
    print("\n[1/4] 加载病例数据...")
    cases = load_cases(args.data)
    print(f"  加载 {len(cases)} 条病例")

    if args.n:
        cases = cases[:args.n]
        print(f"  选择前 {args.n} 条进行评测")

    # 初始化组件
    print("\n[2/4] 初始化组件...")
    llm_client = LLMClient(model_name=args.model)
    print(f"  LLM 客户端已初始化 ({args.model})")

    knowledge_base = None
    if not args.no_kb and os.path.exists(args.kb):
        knowledge_base = KeywordKnowledgeBase()
        knowledge_base.load(args.kb)
        print(f"  知识库已加载 ({len(knowledge_base.records)} 条)")
    else:
        print("  知识库未加载")

    judger = Judger(
        base_url=args.judge_url,
        enable_cost=args.enable_cost,
    )
    print(f"  评测器已初始化 (M2: {args.judge_url})")
    if args.enable_cost:
        print(f"  费用评估已启用")

    # 运行评测
    print("\n[3/4] 开始评测...")
    start_time = time.time()

    # 设置 checkpoint 文件
    checkpoint_file = os.path.join(args.output, f"checkpoint_{args.model}.jsonl")

    results = run_evaluation(
        cases=cases,
        llm_client=llm_client,
        knowledge_base=knowledge_base,
        judger=judger,
        max_steps=args.max_steps,
        top_k=args.topk,
        max_workers=args.max_workers,
        verbose=args.verbose,
        checkpoint_file=checkpoint_file,
        checkpoint_every=args.checkpoint_every,
    )

    elapsed = time.time() - start_time
    print(f"\n  评测完成，耗时 {elapsed:.1f} 秒")

    # 打印最终API调用统计
    api_status = api_counter.get_status()
    print(f"\n  [API调用统计]")
    print(f"    总调用次数: {api_status['total_calls']}")
    print(f"    运行时间: {api_status['elapsed_hours']:.2f} 小时")
    print(f"    平均速率: {api_status['calls_per_hour']:.1f} 次/小时")
    print(f"    各模型调用:")
    for model, count in api_status['model_counts'].items():
        print(f"      {model}: {count}")

    # 生成报告
    print("\n[4/4] 生成报告...")
    report = generate_report(
        results=results,
        output_dir=args.output,
        model_name=args.model,
        timestamp=timestamp,
        enable_cost=args.enable_cost,
    )

    # 打印摘要
    print("\n" + "=" * 60)
    print("评测结果摘要")
    print("=" * 60)
    print(f"诊断准确率: {report['summary']['diagnosis']['accuracy']*100:.1f}%")
    print(f"诊断平均分: {report['summary']['diagnosis']['avg_score']:.2f}/5")
    print(f"治疗准确率: {report['summary']['treatment']['accuracy']*100:.1f}%")
    print(f"治疗平均分: {report['summary']['treatment']['avg_score']:.2f}/5")
    print(f"禁忌违反率: {report['summary']['safety']['violation_rate']*100:.1f}%")
    print(f"安全平均分: {report['summary']['safety']['avg_score']:.2f}/5")
    print(f"综合平均分: {report['summary']['total_avg_score']:.2f}/5")

    if args.enable_cost and report.get('cost_stats'):
        print(f"平均费用: {report['cost_stats']['avg_cost']:.2f}元")
    print("=" * 60)


if __name__ == "__main__":
    main()