"""
MedAgent Benchmark 多模型评测脚本

评测多个 LLM 模型在医疗问诊任务上的表现，生成对比报告。

支持的模型:
- 302API: gpt-5.4, claude-opus-4-6, gemini-3.1-pro-preview
- 百炼codingplan: qwen3.5-plus, qwen3-max-2026-01-23, glm-5, kimi-k2.5, MiniMax-M2.5
- 百炼normal: deepseek-v3.2, qwen3.5-35b-a3b

Usage:
    # 评测所有模型 (各10条)
    python exp/benchmark.py --n 10

    # 指定模型评测
    python exp/benchmark.py --models qwen3.5-plus,gpt-5.4,deepseek-v3.2 --n 100

    # 分批评测 (先测500条，再测后500条)
    python exp/benchmark.py --n 500 --start 0
    python exp/benchmark.py --n 500 --start 500

    # 重跑失败的病例 (断点续跑)
    python exp/benchmark.py --models qwen3.5-plus --retry-failed

    # 使用benchmark数据集
    python exp/benchmark.py --data exp/data/benchmark_1000.jsonl --n 1000

    # 完整评测 (所有模型 + 100条)
    python exp/benchmark.py --all --n 100

容错机制:
- 429限流错误自动等待重试 (最多3次，递增等待时间)
- checkpoint 记录每条病例状态 (success/failed)
- --retry-failed 参数只重跑失败的病例
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.schema import MedicalCase
from src.medagent import LLMClient, MedAgent, Judger, EvalResult
from src.medagent.knowledge_tool_v2 import KeywordKnowledgeBase
from src.medagent.llm import api_counter

# =============================================================================
# 模型配置
# =============================================================================

MODEL_CONFIGS = {
    # 302API 模型 (全部使用 302AI)
    "gpt-5.4": {
        "provider": "302",
        "base_url": "https://api.302.ai/v1",
        "api_key_env": "302_API_KEY",
    },
    "claude-opus-4-6": {
        "provider": "302",
        "base_url": "https://api.302.ai/v1",
        "api_key_env": "302_API_KEY",
    },
    "gemini-3.1-pro-preview": {
        "provider": "302",
        "base_url": "https://api.302.ai/v1",
        "api_key_env": "302_API_KEY",
    },
    "qwen3.5-plus": {
        "provider": "302",
        "base_url": "https://api.302.ai/v1",
        "api_key_env": "302_API_KEY",
    },
    "qwen3-max-2026-01-23": {
        "provider": "302",
        "base_url": "https://api.302.ai/v1",
        "api_key_env": "302_API_KEY",
    },
    "glm-5": {
        "provider": "302",
        "base_url": "https://api.302.ai/v1",
        "api_key_env": "302_API_KEY",
    },
    "kimi-k2.5": {
        "provider": "302",
        "base_url": "https://api.302.ai/v1",
        "api_key_env": "302_API_KEY",
    },
    "MiniMax-M2.5": {
        "provider": "302",
        "base_url": "https://api.302.ai/v1",
        "api_key_env": "302_API_KEY",
    },
    "deepseek-v3.2": {
        "provider": "302",
        "base_url": "https://api.302.ai/v1",
        "api_key_env": "302_API_KEY",
    },
}

# 默认评测模型列表 (全部使用 302AI)
DEFAULT_MODELS = [
    "gpt-5.4",
    "claude-opus-4-6",
    "gemini-3.1-pro-preview",
    "qwen3.5-plus",
    "qwen3-max-2026-01-23",
    "glm-5",
    "kimi-k2.5",
    "MiniMax-M2.5",
    "deepseek-v3.2",
]


# =============================================================================
# 数据加载
# =============================================================================

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


# =============================================================================
# 单模型评测
# =============================================================================

def run_single_model_eval(
    model_name: str,
    cases: List[MedicalCase],
    knowledge_base: Optional[KeywordKnowledgeBase],
    judger: Judger,
    max_steps: int = 20,
    top_k: int = 3,
    max_workers: int = 5,
    checkpoint_file: Optional[str] = None,
    checkpoint_every: int = 10,
    retry_failed: bool = False,  # 是否重跑失败的病例
) -> List[EvalResult]:
    """
    运行单个模型的评测
    """
    print(f"\n{'='*60}")
    print(f"  评测模型: {model_name}")
    print(f"{'='*60}")

    # 获取模型配置
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        print(f"  [Error] 未找到模型配置: {model_name}")
        return []

    # 检查 API key
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        print(f"  [Error] 环境变量 {config['api_key_env']} 未设置")
        return []

    # 创建 LLM 客户端
    llm_client = LLMClient(
        model_name=model_name,
        base_url=config["base_url"],
        api_key=api_key,
    )

    # 加载 checkpoint
    processed_ids = set()
    failed_ids = set()  # 记录失败的病例ID
    results_dict = {}

    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"  从 checkpoint 恢复...")
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        case_id = data.get("case_id")
                        status = data.get("status", "success")  # 默认成功
                        if case_id:
                            if status == "success":
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
                                )
                            elif status == "failed":
                                failed_ids.add(case_id)  # 失败的病例需要重跑
            print(f"  已恢复 {len(processed_ids)} 条成功, {len(failed_ids)} 条失败")
        except Exception as e:
            print(f"  checkpoint 加载失败: {e}")

    # 过滤未处理病例
    if retry_failed:
        # 只重跑失败的病例
        remaining_cases = [c for c in cases if c.case_id in failed_ids]
        print(f"  重跑失败病例: {len(remaining_cases)} 条")
    else:
        # 正常流程：跳过已成功的，重跑失败的和未处理的
        remaining_cases = [c for c in cases if c.case_id not in processed_ids]
        print(f"  剩余待处理: {len(remaining_cases)}/{len(cases)} (含 {len(failed_ids)} 条失败待重跑)")

    if not remaining_cases:
        return [results_dict.get(c.case_id) for c in cases]

    # 保存 checkpoint (带状态标记)
    def save_checkpoint(batch_results: List[tuple]):
        """保存 checkpoint，每项为 (EvalResult, status)"""
        if not checkpoint_file:
            return
        try:
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
            mode = "a" if os.path.exists(checkpoint_file) else "w"
            with open(checkpoint_file, mode, encoding="utf-8") as f:
                for r, status in batch_results:
                    f.write(json.dumps({
                        "case_id": r.case_id,
                        "status": status,  # "success" 或 "failed"
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
                    }, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  checkpoint 保存失败: {e}")

    # 处理单个病例 (带重试机制)
    def process_case(case: MedicalCase, max_retries: int = 3) -> tuple:
        last_error = None
        for retry in range(max_retries):
            try:
                agent = MedAgent(
                    llm_client=llm_client,
                    case=case,
                    knowledge_base=knowledge_base,
                    max_steps=max_steps,
                    top_k=top_k,
                    verbose=False,
                )

                result = agent.run()

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

                return eval_result, "success"  # 成功标记

            except Exception as e:
                last_error = e
                error_str = str(e)
                # 429 限流错误，等待后重试
                if "429" in error_str or "rate" in error_str.lower() or "throttl" in error_str.lower():
                    wait_time = 60 * (retry + 1)  # 递增等待时间: 60s, 120s, 180s
                    print(f"\n[Retry {retry+1}/{max_retries}] Case {case.case_id}: 429限流，等待{wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"\n[Retry {retry+1}/{max_retries}] Case {case.case_id}: {e}")
                    if retry < max_retries - 1:
                        time.sleep(5)  # 其他错误等待5秒

        # 重试全部失败
        print(f"\n[Failed] Case {case.case_id}: 所有重试失败 - {last_error}")
        return EvalResult(
            case_id=case.case_id,
            total_score=0,
            trajectory=[],
            ground_truth=case.case_to_dict().get("ground_truth", {}),
        ), "failed"  # 返回失败标记

    # 并发执行
    batch_results = []
    success_count = 0
    failed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_case, case): case for case in remaining_cases}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{model_name}"):
            result_tuple = future.result()
            eval_result, status = result_tuple if isinstance(result_tuple, tuple) else (result_tuple, "success")
            results_dict[eval_result.case_id] = eval_result
            batch_results.append((eval_result, status))

            if status == "success":
                success_count += 1
            else:
                failed_count += 1

            if len(batch_results) >= checkpoint_every:
                save_checkpoint(batch_results)
                batch_results.clear()

    if batch_results:
        save_checkpoint(batch_results)

    print(f"  完成: {success_count} 成功, {failed_count} 失败")
    return [results_dict.get(c.case_id) for c in cases]


# =============================================================================
# 结果分析与报告生成
# =============================================================================

def compute_model_stats(results: List[EvalResult]) -> Dict:
    """计算单个模型的统计指标"""
    if not results:
        return {}

    total = len(results)

    # 准确率统计
    diagnosis_correct = sum(1 for r in results if r.diagnosis_correct)
    treatment_correct = sum(1 for r in results if r.treatment_correct)
    avoid_violated = sum(1 for r in results if r.avoid_violated)

    # 平均分数
    avg_diagnosis = sum(r.diagnosis_score for r in results) / total
    avg_treatment = sum(r.treatment_score for r in results) / total
    avg_avoid = sum(r.avoid_score for r in results) / total
    avg_total = sum(r.total_score for r in results) / total

    # 效率统计
    avg_steps = sum(r.efficiency.total_steps for r in results) / total
    avg_exam_items = sum(r.efficiency.exam_items for r in results) / total
    avg_tokens = sum(r.efficiency.total_tokens for r in results) / total
    avg_latency = sum(r.efficiency.total_latency for r in results) / total

    # 分数分布
    score_dist = {
        "excellent": sum(1 for r in results if r.total_score >= 4),
        "good": sum(1 for r in results if 3 <= r.total_score < 4),
        "medium": sum(1 for r in results if 2 <= r.total_score < 3),
        "poor": sum(1 for r in results if r.total_score < 2),
    }

    return {
        "total_cases": total,
        "diagnosis": {
            "accuracy": diagnosis_correct / total,
            "avg_score": avg_diagnosis,
        },
        "treatment": {
            "accuracy": treatment_correct / total,
            "avg_score": avg_treatment,
        },
        "safety": {
            "violation_rate": avoid_violated / total,
            "avg_score": avg_avoid,
        },
        "efficiency": {
            "avg_steps": avg_steps,
            "avg_exam_items": avg_exam_items,
            "avg_tokens": avg_tokens,
            "avg_latency": avg_latency,
        },
        "total_avg_score": avg_total,
        "score_distribution": score_dist,
    }


def generate_comparison_table(all_stats: Dict[str, Dict]) -> str:
    """生成对比表格 (Markdown)"""
    lines = []
    lines.append("\n## 模型对比表格\n")
    lines.append("| 模型 | 诊断准确率 | 诊断分数 | 治疗准确率 | 治疗分数 | 禁忌违反率 | 安全分数 | 综合分数 |")
    lines.append("|------|-----------|---------|-----------|---------|-----------|---------|---------|")

    # 按综合分数排序
    sorted_models = sorted(
        all_stats.items(),
        key=lambda x: x[1].get("total_avg_score", 0),
        reverse=True
    )

    for model, stats in sorted_models:
        diag_acc = stats["diagnosis"]["accuracy"] * 100
        diag_score = stats["diagnosis"]["avg_score"]
        treat_acc = stats["treatment"]["accuracy"] * 100
        treat_score = stats["treatment"]["avg_score"]
        avoid_rate = stats["safety"]["violation_rate"] * 100
        avoid_score = stats["safety"]["avg_score"]
        total_score = stats["total_avg_score"]

        lines.append(f"| {model} | {diag_acc:.1f}% | {diag_score:.2f} | {treat_acc:.1f}% | {treat_score:.2f} | {avoid_rate:.1f}% | {avoid_score:.2f} | **{total_score:.2f}** |")

    return "\n".join(lines)


def generate_efficiency_table(all_stats: Dict[str, Dict]) -> str:
    """生成效率对比表格"""
    lines = []
    lines.append("\n## 效率指标对比\n")
    lines.append("| 模型 | 平均轮次 | 平均检查项 | 平均Token数 | 平均耗时(s) |")
    lines.append("|------|---------|-----------|------------|------------|")

    for model, stats in sorted(all_stats.items(), key=lambda x: x[1].get("total_avg_score", 0), reverse=True):
        eff = stats["efficiency"]
        lines.append(f"| {model} | {eff['avg_steps']:.1f} | {eff['avg_exam_items']:.1f} | {eff['avg_tokens']:.0f} | {eff['avg_latency']:.2f} |")

    return "\n".join(lines)


def generate_markdown_report(
    all_stats: Dict[str, Dict],
    output_dir: str,
    timestamp: str,
    n_cases: int,
    data_path: str,
) -> str:
    """生成 Markdown 报告"""
    lines = []

    lines.append("# MedAgent Benchmark 评测报告\n")
    lines.append(f"**时间**: {timestamp}")
    lines.append(f"**数据**: {data_path}")
    lines.append(f"**病例数**: {n_cases}")
    lines.append(f"**模型数**: {len(all_stats)}")

    # 排名
    lines.append("\n## 模型排名\n")
    sorted_models = sorted(all_stats.items(), key=lambda x: x[1].get("total_avg_score", 0), reverse=True)
    for i, (model, stats) in enumerate(sorted_models, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        lines.append(f"{medal} **{model}**: {stats['total_avg_score']:.2f} 分")

    # 对比表格
    lines.append(generate_comparison_table(all_stats))

    # 效率表格
    lines.append(generate_efficiency_table(all_stats))

    # 分数分布
    lines.append("\n## 分数分布\n")
    for model, stats in sorted_models[:5]:
        dist = stats["score_distribution"]
        total = stats["total_cases"]
        lines.append(f"\n### {model}")
        lines.append(f"- 优秀 (4-5分): {dist['excellent']} ({dist['excellent']/total*100:.1f}%)")
        lines.append(f"- 良好 (3-4分): {dist['good']} ({dist['good']/total*100:.1f}%)")
        lines.append(f"- 中等 (2-3分): {dist['medium']} ({dist['medium']/total*100:.1f}%)")
        lines.append(f"- 较差 (<2分): {dist['poor']} ({dist['poor']/total*100:.1f}%)")

    lines.append("\n---")
    lines.append("*报告由 MedAgent Benchmark 自动生成*")

    report_content = "\n".join(lines)

    # 保存报告
    report_file = os.path.join(output_dir, f"benchmark_report_{timestamp}.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_file


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MedAgent Benchmark 多模型评测")

    parser.add_argument(
        "--data",
        type=str,
        default="exp/data/benchmark_1000.jsonl",
        help="评测数据路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exp/output",
        help="输出目录"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="评测模型列表 (逗号分隔，如 qwen3.5-plus,gpt-5.4)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="评测所有可用模型"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="评测病例数量 (None 表示全部)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="起始病例索引 (默认从0开始)"
    )
    parser.add_argument(
        "--kb",
        type=str,
        default="data/knowledge_dataset/ResponseMed.json",
        help="知识库路径 (ResponseMed.json)"
    )
    parser.add_argument(
        "--no-kb",
        action="store_true",
        help="不使用知识库"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Agent 最大步数"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="并发线程数"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="每 N 条保存 checkpoint"
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="只重跑 checkpoint 中标记为失败的病例"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="自定义模型 base URL (用于本地部署模型，如 http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="自定义模型 API key (与 --base-url 配合使用)"
    )

    args = parser.parse_args()

    # 时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 如果指定了 --base-url，添加自定义模型配置
    if args.base_url:
        custom_model_name = args.models.split(",")[0].strip() if args.models else "custom"
        MODEL_CONFIGS[custom_model_name] = {
            "provider": "custom",
            "base_url": args.base_url,
            "api_key_env": None,
        }
        if args.api_key:
            # 临时设置环境变量
            os.environ["CUSTOM_API_KEY"] = args.api_key
            MODEL_CONFIGS[custom_model_name]["api_key_env"] = "CUSTOM_API_KEY"
        else:
            # vLLM 本地部署可能不需要 API key
            MODEL_CONFIGS[custom_model_name]["api_key_env"] = None

    # 确定评测模型
    if args.all:
        models = DEFAULT_MODELS
    elif args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = DEFAULT_MODELS[:5]  # 默认评测前5个

    print("=" * 60)
    print("MedAgent Benchmark 多模型评测")
    print("=" * 60)
    print(f"数据: {args.data}")
    print(f"输出: {args.output}")
    print(f"模型: {len(models)} 个")
    for m in models:
        print(f"  - {m}")
    print(f"知识库: {'不使用' if args.no_kb else args.kb}")
    print(f"病例范围: [{args.start}:{args.start + args.n if args.n else 'end'}]")
    print("=" * 60)

    # 加载病例
    print("\n[1/3] 加载病例数据...")
    cases = load_cases(args.data)
    print(f"  加载 {len(cases)} 条病例")

    if args.n:
        cases = cases[args.start:args.start + args.n]
        print(f"  选择 [{args.start}:{args.start + args.n}] 共 {len(cases)} 条")
    elif args.start > 0:
        cases = cases[args.start:]
        print(f"  选择 [{args.start}:] 共 {len(cases)} 条")

    # 加载知识库
    knowledge_base = None
    if not args.no_kb and os.path.exists(args.kb):
        print("\n[2/3] 加载知识库...")
        knowledge_base = KeywordKnowledgeBase()
        knowledge_base.load(args.kb)
        print(f"  加载 {len(knowledge_base.records)} 条知识")

    # 初始化 Judger
    judger = Judger()

    # 评测各模型
    print("\n[3/3] 开始评测...")
    start_time = time.time()

    all_results = {}
    all_stats = {}

    os.makedirs(args.output, exist_ok=True)

    for model_name in models:
        # 检查模型配置
        config = MODEL_CONFIGS.get(model_name)
        if not config:
            print(f"\n[Skip] 未找到模型配置: {model_name}")
            continue

        # 检查 API key (本地部署可能不需要)
        api_key_env = config.get("api_key_env")
        api_key = os.getenv(api_key_env) if api_key_env else "dummy"
        if not api_key_env:
            print(f"\n[Info] 使用本地部署模型: {model_name} ({config['base_url']})")
        elif not api_key:
            print(f"\n[Skip] 环境变量 {api_key_env} 未设置")
            continue

        # checkpoint 文件
        # checkpoint 文件名包含 start 参数，避免不同范围的测试互相干扰
        checkpoint_suffix = f"_start{args.start}" if args.start > 0 else ""
        checkpoint_file = os.path.join(args.output, f"checkpoint_{model_name}{checkpoint_suffix}.jsonl")

        # 运行评测
        results = run_single_model_eval(
            model_name=model_name,
            cases=cases,
            knowledge_base=knowledge_base,
            judger=judger,
            max_steps=args.max_steps,
            max_workers=args.max_workers,
            checkpoint_file=checkpoint_file,
            checkpoint_every=args.checkpoint_every,
            retry_failed=args.retry_failed,
        )

        if results:
            all_results[model_name] = results
            all_stats[model_name] = compute_model_stats(results)

    elapsed = time.time() - start_time
    print(f"\n评测完成，耗时 {elapsed:.1f} 秒")

    # 保存各模型结果
    if all_stats:
        print("\n保存结果...")

        for model_name, stats in all_stats.items():
            # 保存单个模型结果
            stats_file = os.path.join(args.output, f"model_stats_{model_name}.json")
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"  {model_name}: {stats_file}")

        # 打印摘要
        print("\n" + "=" * 60)
        print("评测结果摘要")
        print("=" * 60)

        sorted_models = sorted(all_stats.items(), key=lambda x: x[1].get("total_avg_score", 0), reverse=True)

        print("\n| 模型 | 综合分数 | 诊断准确率 | 治疗准确率 | 禁忌违反率 |")
        print("|------|---------|-----------|-----------|-----------|")
        for model, stats in sorted_models:
            print(f"| {model} | {stats['total_avg_score']:.2f} | {stats['diagnosis']['accuracy']*100:.1f}% | {stats['treatment']['accuracy']*100:.1f}% | {stats['safety']['violation_rate']*100:.1f}% |")

        print("\n最佳模型: " + sorted_models[0][0])
        print("\n提示: 运行 python exp/summarize_benchmark.py 生成对比报告")

    print("=" * 60)


if __name__ == "__main__":
    main()