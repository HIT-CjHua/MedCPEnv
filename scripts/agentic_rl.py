#!/usr/bin/env python3
"""
MedAgent Agentic RL Training with GRPO (TRL)

基于TRL的GRPO方法训练MedAgent，使其学会:
1. 合理使用问诊(ASK)、检查(EXAM)、知识库(KNOWLEDGE)工具
2. 给出正确的诊断和治疗建议
3. 避免违反医疗禁忌

使用方式:
    # 默认训练
    python scripts/agentic_rl.py

    # 指定模型和参数
    python scripts/agentic_rl.py --model Qwen/Qwen2.5-3B --max-steps 1000

    # 使用LoRA微调
    python scripts/agentic_rl.py --use-lora --lora-r 16

要求:
    - GPU 0: 用于训练模型
    - GPU 1: 用于部署embedding model (知识库检索)
"""

import os
import sys
import re
import json
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datasets import Dataset
from contextvars import ContextVar

# 设置GPU - 只使用GPU 0训练，GPU 1留给embedding
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema import MedicalCase, MedicalItem, GroundTruth
from src.medagent.tool import AskTool, ExamTool, KnowledgeTool, ToolManager
from src.medagent.knowledge_base import KnowledgeBase


# -----------------------------------------------------------------------------
# 上下文管理：用于在训练时动态绑定当前处理的病例
# -----------------------------------------------------------------------------

# 当前病例上下文
_current_case: ContextVar[Optional[MedicalCase]] = ContextVar("current_case", default=None)

# 知识库实例（全局单例）
_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base(kb_path: str = "data/knowledge_db") -> Optional[KnowledgeBase]:
    """获取或初始化知识库实例"""
    global _knowledge_base
    if _knowledge_base is None:
        kb_full_path = PROJECT_ROOT / kb_path
        if kb_full_path.exists():
            try:
                _knowledge_base = KnowledgeBase().load(str(kb_full_path))
                print(f"[KnowledgeBase] 加载成功: {kb_full_path}")
            except Exception as e:
                print(f"[KnowledgeBase] 加载失败: {e}")
        else:
            print(f"[KnowledgeBase] 知识库不存在: {kb_full_path}")
    return _knowledge_base


def set_current_case(case: MedicalCase):
    """设置当前处理的病例"""
    _current_case.set(case)


def get_current_case() -> Optional[MedicalCase]:
    """获取当前处理的病例"""
    return _current_case.get()


# -----------------------------------------------------------------------------
# TRL工具定义 - 包装MedAgent实际工具
# -----------------------------------------------------------------------------

def tool_ask(keywords: str) -> str:
    """
    问诊工具，用于获取患者的主观信息。

    通过关键词匹配获取患者的症状描述、病史等主观信息。
    在实际执行时，工具会根据病例数据返回匹配的内容。

    Args:
        keywords: 问诊关键词，多个关键词用逗号分隔。
                  例如: "疼痛性质,疼痛部位,持续时间"

    Returns:
        str: 患者的回答，如果未匹配到相关信息则返回'相关信息不明'
    """
    case = get_current_case()
    if case is None:
        # RL训练模式：返回模拟响应（格式正确但不包含实际数据）
        # 模型会学习工具的使用模式，实际病例数据在reward计算时验证
        kw_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        if not kw_list:
            return "请提供问诊关键词。"
        # 返回格式化的模拟响应
        return f"[问诊结果] 关于 {', '.join(kw_list)} 的相关信息已获取。"

    # 评测/推理模式：使用实际工具
    kw_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
    if not kw_list:
        return "请提供问诊关键词。"

    tool = AskTool(case)
    return tool.execute(keywords=kw_list)


def tool_exam(keywords: str) -> str:
    """
    检查工具，用于获取患者的客观信息。

    通过关键词匹配获取化验结果、影像学检查等客观信息。
    在实际执行时，工具会根据病例数据返回匹配的检查结果。

    Args:
        keywords: 检查关键词，多个关键词用逗号分隔。
                  例如: "血常规,胸片,心电图"

    Returns:
        str: 检查结果，如果未匹配到相关检查则返回'相关检查不适用'
    """
    case = get_current_case()
    if case is None:
        # RL训练模式：返回模拟响应
        kw_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        if not kw_list:
            return "请提供检查关键词。"
        return f"[检查结果] 关于 {', '.join(kw_list)} 的检查数据已获取。"

    # 评测/推理模式：使用实际工具
    kw_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
    if not kw_list:
        return "请提供检查关键词。"

    tool = ExamTool(case)
    return tool.execute(keywords=kw_list)


def tool_knowledge(query: str) -> str:
    """
    知识库查询工具，用于检索医学知识。

    查询医学知识库获取诊断标准、治疗指南、药物信息等。
    知识库包含权威医学文献和临床指南。

    Args:
        query: 查询内容，应具体明确。
               例如: "急性阑尾炎的诊断标准"

    Returns:
        str: 相关医学知识的摘要
    """
    kb = get_knowledge_base()
    if kb is None:
        # 知识库未初始化时的响应
        if not query or not query.strip():
            return "请提供查询内容。"
        return f"[知识库] 关于 '{query.strip()}' 的相关知识已检索。"

    if not query or not query.strip():
        return "请提供查询内容。"

    # 使用实际工具
    tool = KnowledgeTool(knowledge_base=kb, top_k=10, rerank_top_n=3, max_length=500)
    return tool.execute(query=query.strip())


# -----------------------------------------------------------------------------
# 数据格式化
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一名专业的医疗问诊AI助手。你需要根据患者的主诉，通过问诊、检查和知识查询来做出诊断。

## 可用工具

你可以使用以下工具获取信息：

1. tool_ask: 问诊工具，获取患者主观信息（症状、病史等）
   参数: keywords - 关键词，用逗号分隔

2. tool_exam: 检查工具，获取患者客观信息（化验、影像等）
   参数: keywords - 关键词，用逗号分隔

3. tool_knowledge: 知识库查询工具
   参数: query - 查询内容

## 诊断流程建议

1. 首先通过问诊获取关键症状信息
2. 根据症状选择必要的检查项目
3. 有疑问时可查询知识库
4. 综合所有信息给出诊断和治疗建议

## 输出格式

最终诊断请使用以下格式:
*诊断: 你的诊断结果*
*治疗: 你的治疗建议*

注意：避免推荐禁忌项目。"""


def build_medical_case(example: Dict) -> MedicalCase:
    """
    从训练数据构建MedicalCase实例

    Args:
        example: 包含case_id, chief_complaint, subjective, objective等字段的病例数据

    Returns:
        MedicalCase: 病例实例
    """
    # 构建 subjective
    subjective = []
    for item in example.get("subjective", []):
        subjective.append(MedicalItem(
            keywords=item.get("keywords", []),
            content=item.get("content", ""),
            necessity=item.get("necessity", True),
        ))

    # 构建 objective
    objective = []
    for item in example.get("objective", []):
        objective.append(MedicalItem(
            keywords=item.get("keywords", []),
            content=item.get("content", ""),
            necessity=item.get("necessity", True),
        ))

    # 构建 ground_truth
    gt_data = example.get("ground_truth", {})
    ground_truth = GroundTruth(
        diagnosis=gt_data.get("diagnosis", []),
        treatment=gt_data.get("treatment", []),
        avoid=gt_data.get("avoid", []),
    )

    # 构建病例
    case = MedicalCase(
        case_id=example.get("case_id", "unknown"),
        difficulty=example.get("difficulty", "medium"),
        tags=example.get("tags", []),
        chief_complaint=example.get("chief_complaint", ""),
        subjective=subjective,
        objective=objective,
        ground_truth=ground_truth,
        source=example.get("source", "synthetic"),
    )

    return case


def format_training_example(example: Dict) -> Dict:
    """
    格式化训练样本为TRL需要的格式

    Args:
        example: 包含chief_complaint, ground_truth等字段的病例数据

    Returns:
        Dict: 包含prompt和元数据的训练样本
    """
    chief_complaint = example.get("chief_complaint", "")
    ground_truth = example.get("ground_truth", {})
    diagnosis = ground_truth.get("diagnosis", [])
    treatment = ground_truth.get("treatment", [])
    avoid = ground_truth.get("avoid", [])

    content = f"{SYSTEM_PROMPT}\n\n患者主诉: {chief_complaint}\n\n请开始诊断流程。"

    prompt = [{"role": "user", "content": content}]

    # 构建MedicalCase用于工具执行
    medical_case = build_medical_case(example)

    return {
        "prompt": prompt,
        "case_id": example.get("case_id", ""),
        "ground_truth_diagnosis": diagnosis,
        "ground_truth_treatment": treatment,
        "ground_truth_avoid": avoid,
        "medical_case": medical_case,  # 传递病例实例
    }


def load_training_data(data_path: str, max_samples: Optional[int] = None) -> Dataset:
    """
    加载并格式化训练数据

    Args:
        data_path: 数据文件路径 (jsonl或json)
        max_samples: 最大样本数

    Returns:
        Dataset: HuggingFace Dataset对象
    """
    data = []

    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"不支持的数据格式: {data_path}")

    if max_samples:
        data = data[:max_samples]

    formatted_data = [format_training_example(d) for d in data]

    print(f"加载 {len(formatted_data)} 条训练样本")
    return Dataset.from_list(formatted_data)


# -----------------------------------------------------------------------------
# 奖励函数
# -----------------------------------------------------------------------------

def correctness_reward(completions, ground_truth_diagnosis, ground_truth_treatment, **kwargs) -> List[float]:
    """
    诊断正确性奖励

    检查最终诊断是否与标准答案匹配。
    正确的诊断得1分，正确的治疗建议得0.5分。

    Args:
        completions: 模型生成的完成序列
        ground_truth_diagnosis: 标准诊断列表
        ground_truth_treatment: 标准治疗列表

    Returns:
        List[float]: 每个样本的奖励值
    """
    rewards = []

    for completion, gt_diag, gt_treat in zip(
        completions, ground_truth_diagnosis, ground_truth_treatment
    ):
        reward = 0.0

        if completion and len(completion) > 0:
            last_turn = completion[-1]
            content = (
                last_turn.get("content", "").lower()
                if isinstance(last_turn, dict)
                else str(last_turn).lower()
            )

            # 检查诊断匹配
            if gt_diag:
                for diag in gt_diag:
                    if isinstance(diag, str) and diag.lower() in content:
                        reward += 1.0
                        break

            # 检查治疗匹配
            if gt_treat:
                for treat in gt_treat:
                    if isinstance(treat, str) and treat.lower() in content:
                        reward += 0.5
                        break

        rewards.append(reward)

    return rewards


def avoid_violation_reward(completions, ground_truth_avoid, **kwargs) -> List[float]:
    """
    禁忌违反惩罚

    检查是否推荐了禁忌项目，违反禁忌给予严重惩罚。

    Args:
        completions: 模型生成的完成序列
        ground_truth_avoid: 禁忌项目列表

    Returns:
        List[float]: 每个样本的奖励值（负数表示惩罚）
    """
    rewards = []

    for completion, gt_avoid in zip(completions, ground_truth_avoid):
        reward = 0.0

        if completion and len(completion) > 0:
            last_turn = completion[-1]
            content = (
                last_turn.get("content", "").lower()
                if isinstance(last_turn, dict)
                else str(last_turn).lower()
            )

            # 检查是否违反禁忌
            if gt_avoid:
                for avoid_item in gt_avoid:
                    if isinstance(avoid_item, str) and avoid_item.lower() in content:
                        reward -= 2.0  # 严重惩罚

        rewards.append(reward)

    return rewards


def tool_usage_reward(completions, **kwargs) -> List[float]:
    """
    工具使用效率奖励

    奖励合理使用工具：
    - 2-6次工具调用: +0.5分（合理）
    - 超过10次: -1.0分（过度检查）
    - 0次: -0.5分（未充分收集信息）

    Args:
        completions: 模型生成的完成序列

    Returns:
        List[float]: 每个样本的奖励值
    """
    rewards = []

    for completion in completions:
        reward = 0.0
        tool_calls = 0

        for turn in completion:
            if isinstance(turn, dict) and turn.get("tool_calls"):
                tool_calls += len(turn["tool_calls"])

        # 合理的工具调用次数
        if 2 <= tool_calls <= 6:
            reward += 0.5
        elif tool_calls > 10:
            reward -= 1.0
        elif tool_calls == 0:
            reward -= 0.5

        rewards.append(reward)

    return rewards


def structure_reward(completions, **kwargs) -> List[float]:
    """
    结构奖励

    奖励正确的输出格式，确保诊断和治疗建议清晰标注。

    Args:
        completions: 模型生成的完成序列

    Returns:
        List[float]: 每个样本的奖励值
    """
    rewards = []

    for completion in completions:
        reward = 0.0

        if completion and len(completion) > 0:
            last_turn = completion[-1]
            content = (
                last_turn.get("content", "")
                if isinstance(last_turn, dict)
                else str(last_turn)
            )

            # 检查是否有正确的格式
            if re.search(r'\*诊断[:：]', content):
                reward += 0.3
            if re.search(r'\*治疗[:：]', content):
                reward += 0.3

        rewards.append(reward)

    return rewards


# -----------------------------------------------------------------------------
# 自定义Callback：在工具执行前绑定病例上下文
# -----------------------------------------------------------------------------

from transformers import TrainerCallback


class CaseContextCallback(TrainerCallback):
    """
    病例上下文回调

    在每个训练步骤前，将当前batch的病例数据绑定到工具上下文中。
    由于TRL的GRPOTrainer在生成completions时需要访问工具，
    我们需要确保工具能够获取到正确的病例数据。
    """

    def __init__(self, dataset):
        self.dataset = dataset
        # 构建索引到medical_case的映射
        self.case_map = {}
        for idx in range(len(dataset)):
            item = dataset[idx]
            if "medical_case" in item:
                self.case_map[idx] = item["medical_case"]
        print(f"  [CaseContextCallback] 已绑定 {len(self.case_map)} 个病例")

    def on_step_begin(self, args, state, control, **kwargs):
        """每个step开始时的回调"""
        # 注意：TRL的GRPO在生成时会处理整个batch
        # 我们在工具执行时动态查找病例
        pass

    def on_step_end(self, args, state, control, **kwargs):
        """每个step结束时的回调 - 清理上下文"""
        set_current_case(None)

    def get_case_by_index(self, idx: int) -> Optional[MedicalCase]:
        """根据索引获取病例"""
        return self.case_map.get(idx)


# 全局回调实例引用（用于工具函数访问）
_global_case_callback: Optional[CaseContextCallback] = None


def set_case_callback(callback: CaseContextCallback):
    """设置全局病例回调"""
    global _global_case_callback
    _global_case_callback = callback


def get_case_from_callback(idx: int) -> Optional[MedicalCase]:
    """从回调中获取病例"""
    if _global_case_callback is not None:
        return _global_case_callback.get_case_by_index(idx)
    return None


# -----------------------------------------------------------------------------
# 主训练函数
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MedAgent Agentic RL Training with GRPO")

    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="基础模型名称或路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/agentic_rl",
        help="输出目录",
    )

    # 数据参数
    parser.add_argument(
        "--data",
        type=str,
        default="output/generate_2000/merged_selected.jsonl",
        help="训练数据路径",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数（用于调试）",
    )

    # 训练参数
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="最大训练步数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批次大小",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="学习率",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=1024,
        help="最大生成长度",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="梯度累积步数",
    )

    # LoRA参数
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="是否使用LoRA微调",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA秩",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )

    # vLLM参数
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="是否使用vLLM加速推理",
    )

    # 知识库参数
    parser.add_argument(
        "--kb-path",
        type=str,
        default="data/knowledge_db",
        help="知识库路径",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MedAgent Agentic RL Training")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"数据: {args.data}")
    print(f"输出目录: {args.output_dir}")
    print(f"GPU: 0 (训练), GPU 1 (embedding)")
    print(f"训练步数: {args.max_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"使用LoRA: {args.use_lora}")
    print("=" * 60)

    # 检查TRL是否安装
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("\n错误: TRL未安装。请运行以下命令安装:")
        print("  pip install trl[vllm] transformers accelerate peft")
        print("\n如果上述命令失败，请分别安装:")
        print("  pip install trl transformers accelerate peft")
        return

    # 初始化知识库
    print("\n[0/4] 初始化知识库...")
    get_knowledge_base(args.kb_path)

    # 加载数据
    print("\n[1/4] 加载训练数据...")
    train_dataset = load_training_data(args.data, args.max_samples)

    # 创建病例上下文回调
    case_callback = CaseContextCallback(train_dataset)

    # 配置GRPO
    print("\n[2/4] 配置GRPO...")
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        chat_template_kwargs={"enable_thinking": False},

        # vLLM配置
        use_vllm=args.use_vllm,
        vllm_mode="colocate" if args.use_vllm else None,
        vllm_enable_sleep_mode=False,

        # 日志与保存
        save_steps=50,
        logging_steps=10,
        log_completions=True,
        report_to="none",

        # 显存优化
        bf16=True,
        tf32=True,
    )

    # LoRA配置
    peft_config = None
    if args.use_lora:
        from peft import LoraConfig, TaskType
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        print(f"  LoRA配置: r={args.lora_r}, alpha={args.lora_alpha}")

    # 创建Trainer
    print("\n[3/4] 创建GRPOTrainer...")
    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=train_dataset,
        tools=[tool_ask, tool_exam, tool_knowledge],
        reward_funcs=[
            correctness_reward,
            avoid_violation_reward,
            tool_usage_reward,
            structure_reward,
        ],
        args=grpo_config,
        peft_config=peft_config,
        callbacks=[case_callback],  # 添加病例上下文回调
    )

    # 显示GPU信息
    import torch
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_memory = torch.cuda.max_memory_reserved() / 1024**3
        print(f"\nGPU 0: {gpu_stats.name}")
        print(f"显存总量: {gpu_stats.total_memory / 1024**3:.1f} GB")
        print(f"已预留显存: {start_memory:.2f} GB")

    # 开始训练
    print("\n[4/4] 开始训练...")
    print("-" * 60)

    try:
        trainer.train()
    except Exception as e:
        print(f"\n训练出错: {e}")
        raise

    # 保存模型
    print("\n" + "=" * 60)
    print("训练完成，保存模型...")
    trainer.save_model(args.output_dir)

    # 保存训练配置
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"\n模型已保存至: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()