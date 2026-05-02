#!/usr/bin/env python3
"""
MedAgent Agentic RL Training with GRPO (TRL)

基于TRL的GRPO方法训练MedAgent，使其学会:
1. 合理使用问诊(ASK)、检查(EXAM)、知识库(KNOWLEDGE)工具
2. 给出正确的诊断和治疗建议
3. 避免违反医疗禁忌

参考实现: MedAgent/reference/agentic_rl_trl.ipynb

使用方式:
    # 默认训练
    python scripts/agentic_rl.py

    # 指定模型和参数
    python scripts/agentic_rl.py --model Qwen/Qwen2.5-3B --max-steps 1000

    # 使用LoRA微调
    python scripts/agentic_rl.py --use-lora --lora-r 16

要求:
    - GPU: 用于训练模型和推理
    - 知识库: data/knowledge_db (可选，用于 KNOWLEDGE 工具)
"""

import os
import sys
import re
import json
import time
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datasets import Dataset

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置默认 API Key (Judger 使用 Baichuan-M2)
os.environ.setdefault("BAICHUAN_API_KEY", "sk-aeadfa5f7122ad2358623ab8ab04")

from src.schema import MedicalCase, MedicalItem, GroundTruth
from src.medagent.tool import AskTool, ExamTool, KnowledgeTool
from src.medagent.knowledge_tool_v2 import KeywordKnowledgeBase
from src.medagent.judger import Judger


# -----------------------------------------------------------------------------
# Reward 配置
# -----------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """
    Reward 函数配置

    用于控制各个奖励项的启用与否以及权重

    核心奖励使用 Judger (Baichuan-M2 + ground_truth) 进行评分:
    - diagnosis_score (1-5): 诊断准确性
    - treatment_score (1-5): 治疗合理性
    - 安全分已去除（区分度极低，97%+样本为5分，对梯度无贡献）
    """
    # Judger 评分权重（核心奖励）
    enable_judger: bool = True
    diagnosis_weight: float = 1.0  # 诊断得分权重
    treatment_weight: float = 1.0  # 治疗得分权重（与诊断等权）

    # 工具效率奖励
    enable_tool_efficiency: bool = True
    min_tool_calls: int = 2            # 最低要求: ASK + EXAM
    tool_base_reward: float = 0.5      # 满足最低要求的基准奖励
    efficiency_scale: float = 1.0      # 调用数越少奖励越大，缩放因子

    # 成本奖励
    enable_cost_reward: bool = False
    cost_reward_scale: float = 1.0  # 组内排序奖励的缩放因子

    # Judger 配置
    judger_model: str = "Baichuan-M2"
    judger_base_url: str = "https://api.baichuan-ai.com/v1"  # Baichuan 官方 API


# -----------------------------------------------------------------------------
# 知识库初始化（全局单例）
# -----------------------------------------------------------------------------

_knowledge_base: Optional[KeywordKnowledgeBase] = None


def get_knowledge_base(kb_path: str = "data/knowledge_dataset/ResponseMed.json") -> Optional[KeywordKnowledgeBase]:
    """获取或初始化知识库实例"""
    global _knowledge_base
    if _knowledge_base is None:
        kb_full_path = PROJECT_ROOT / kb_path
        if kb_full_path.exists():
            try:
                _knowledge_base = KeywordKnowledgeBase()
                _knowledge_base.load(str(kb_full_path))
                print(f"[KnowledgeBase] 加载成功: {kb_full_path} ({len(_knowledge_base.records)} 条记录)")
            except Exception as e:
                print(f"[KnowledgeBase] 加载失败: {e}")
        else:
            print(f"[KnowledgeBase] 知识库不存在: {kb_full_path}")
    return _knowledge_base


# -----------------------------------------------------------------------------
# TRL Tool 定义
# 
# 参考 agentic_rl_trl.ipynb:
# - 标准函数签名，带 type hints
# - Google-style docstring 描述工具用途、参数和返回值
# - TRL 会自动解析函数签名和 docstring 生成工具描述
# -----------------------------------------------------------------------------

def tool_ask(keywords: str) -> str:
    """
    问诊工具，用于获取患者的主观信息。
    
    通过关键词匹配获取患者的症状描述、病史等主观信息。
    在 RL 训练时返回模拟响应，在评测时返回真实病例数据。
    
    Args:
        keywords: 问诊关键词，多个关键词用逗号分隔。
                  例如: "疼痛性质,疼痛部位,持续时间"
    
    Returns:
        str: 患者的回答，如果未匹配到相关信息则返回"相关信息不明"。
    """
    # RL 训练模式：返回模拟响应（TRL 在生成时不执行真实工具）
    # 格式化的响应帮助模型学习工具使用模式
    kw_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
    if not kw_list:
        return "请提供问诊关键词。"
    
    # 模拟问诊响应
    simulated_responses = {
        "疼痛": "患者描述疼痛为压榨样，位于胸骨后，向左肩放射。",
        "发热": "患者发热3天，最高体温38.5°C，伴有咳嗽。",
        "头痛": "患者头痛位于前额，持续性，程度中等。",
        "恶心": "患者有恶心感，无呕吐。",
        "呕吐": "患者呕吐2次，为胃内容物。",
        "腹泻": "患者腹泻3次，为水样便。",
        "乏力": "患者感到乏力，精神欠佳。",
        "出汗": "患者伴有大汗。",
        "持续时间": "症状持续约2小时。",
        "既往史": "患者有高血压病史5年，糖尿病病史3年。",
        "过敏史": "患者无药物过敏史。",
        "用药史": "患者目前服用降压药和降糖药。",
    }
    
    # 根据关键词生成模拟响应
    responses = []
    for kw in kw_list:
        for key, resp in simulated_responses.items():
            if key in kw.lower() or kw.lower() in key:
                responses.append(resp)
                break
    
    if responses:
        return "\n".join(responses)
    return f"[问诊结果] 关于 {', '.join(kw_list)} 的相关信息已获取。"


def tool_exam(keywords: str) -> str:
    """
    检查工具，用于获取患者的客观信息。
    
    通过关键词匹配获取化验结果、影像学检查等客观信息。
    在 RL 训练时返回模拟响应，在评测时返回真实病例数据。
    
    Args:
        keywords: 检查关键词，多个关键词用逗号分隔。
                  例如: "血常规,胸片,心电图"
    
    Returns:
        str: 检查结果，如果未匹配到相关检查则返回"相关检查不适用"。
    """
    kw_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
    if not kw_list:
        return "请提供检查关键词。"
    
    # 模拟检查响应
    simulated_results = {
        "血常规": "白细胞 10.5×10^9/L，中性粒细胞 78%，血红蛋白 140g/L。",
        "心电图": "心电图示窦性心律，心率 85次/分，ST段未见明显异常。",
        "胸片": "胸片示双肺纹理清晰，心影大小正常。",
        "心肌酶": "肌钙蛋白T 0.1ng/mL（正常范围），CK-MB 25U/L。",
        "肝功能": "ALT 35U/L，AST 28U/L，总胆红素 15μmol/L。",
        "肾功能": "肌酐 80μmol/L，尿素氮 6.5mmol/L。",
        "血糖": "空腹血糖 6.2mmol/L。",
        "血压": "血压 140/90mmHg。",
        "体温": "体温 37.5°C。",
        "心率": "心率 88次/分。",
        "腹部B超": "肝胆脾胰未见明显异常。",
        "头颅CT": "头颅CT未见明显异常。",
        "MRI": "MRI检查未见明显异常。",
    }
    
    responses = []
    for kw in kw_list:
        for key, resp in simulated_results.items():
            if key in kw.lower() or kw.lower() in key:
                responses.append(resp)
                break
    
    if responses:
        return "\n".join(responses)
    return f"[检查结果] 关于 {', '.join(kw_list)} 的检查数据已获取。"


def tool_knowledge(keywords: str) -> str:
    """
    知识库查询工具，用于检索医学知识。

    通过关键词匹配查询医学知识库，获取诊断标准、治疗指南、药物信息等。
    知识库包含 ResponseMed.json 中的 37 万+条 QA 数据。

    Args:
        keywords: 查询关键词，多个关键词用逗号分隔。
                  例如: "appendicitis, diagnostic criteria, treatment"

    Returns:
        str: 相关医学知识摘要，包含匹配的 QA 记录。
    """
    kw_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
    if not kw_list:
        return "请提供查询关键词。"

    # 尝试使用真实知识库
    kb = get_knowledge_base()
    if kb is not None:
        try:
            tool = KnowledgeTool(knowledge_base=kb, top_k=3)
            return tool.execute(keywords=kw_list)
        except Exception as e:
            print(f"[KnowledgeTool] 查询失败: {e}")
    
    # 模拟知识库响应（用于 RL 训练）
    simulated_knowledge = {
        "心肌梗死": "急性心肌梗死诊断标准：1) 典型胸痛症状；2) 心电图ST段抬高或新发Q波；3) 心肌酶升高。治疗：急诊PCI或溶栓，抗血小板，抗凝。",
        "阑尾炎": "急性阑尾炎诊断标准：转移性右下腹痛，麦氏点压痛，白细胞升高。治疗：急诊手术切除阑尾。",
        "肺炎": "肺炎诊断：发热、咳嗽、咳痰，胸片示肺部浸润影。治疗：抗生素抗感染，对症支持。",
        "胃炎": "急性胃炎：上腹痛、恶心呕吐，多有饮食不当诱因。治疗：抑酸、保护胃黏膜、对症。",
        "高血压": "高血压诊断：收缩压≥140或舒张压≥90mmHg。治疗：生活方式干预+降压药物。",
        "糖尿病": "糖尿病诊断：空腹血糖≥7.0mmol/L或随机血糖≥11.1mmol/L。治疗：饮食控制+运动+降糖药物。",
        "脑卒中": "急性脑卒中：突发神经功能缺损，CT/MRI确诊。治疗：溶栓（缺血性）或手术（出血性），康复。",
        "骨折": "骨折诊断：外伤史、疼痛肿胀畸形，X线确诊。治疗：复位固定，康复训练。",
    }
    
    query_lower = query.strip().lower()
    for key, resp in simulated_knowledge.items():
        if key in query_lower or query_lower in key:
            return resp
    
    return f"[知识库] 关于 '{query.strip()}' 的相关知识已检索。"


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


def format_training_example(example: Dict) -> Dict:
    """
    格式化训练样本为TRL需要的格式

    Args:
        example: 包含 chief_complaint, ground_truth 等字段的病例数据

    Returns:
        Dict: 包含 prompt 和元数据的训练样本
    """
    chief_complaint = example.get("chief_complaint", "")
    ground_truth = example.get("ground_truth", {})

    # 提取标准答案
    diagnosis = ground_truth.get("diagnosis", [])
    treatment = ground_truth.get("treatment", [])
    avoid = ground_truth.get("avoid", [])

    # 构建用户消息
    content = f"{SYSTEM_PROMPT}\n\n患者主诉: {chief_complaint}\n\n请开始诊断流程。"
    prompt = [{"role": "user", "content": content}]

    # 构建标准答案（用于 judger_reward）
    # 将诊断列表转换为字符串，便于奖励函数匹配
    answer = diagnosis[0] if diagnosis else ""

    return {
        "prompt": prompt,
        "answer": answer,                          # 用于 fallback 匹配
        "ground_truth_diagnosis": diagnosis,       # 用于 judger_reward
        "ground_truth_treatment": treatment,       # 用于 judger_reward
        "ground_truth_avoid": avoid,               # 用于 judger_reward
        "chief_complaint": chief_complaint,        # 用于 judger_reward
        "case_id": example.get("case_id", ""),     # 用于 judger_reward
    }


def load_training_data(data_path: str, max_samples: Optional[int] = None) -> Dataset:
    """
    加载并格式化训练数据
    
    Args:
        data_path: 数据文件路径 (jsonl 或 json)
        max_samples: 最大样本数（用于调试）
    
    Returns:
        Dataset: HuggingFace Dataset 对象
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
    
    print(f"[数据] 加载 {len(formatted_data)} 条训练样本")
    return Dataset.from_list(formatted_data)


# -----------------------------------------------------------------------------
# Judger 全局实例
# -----------------------------------------------------------------------------

_judger: Optional[Judger] = None


def get_judger(reward_config: Optional[RewardConfig] = None) -> Judger:
    """获取或初始化 Judger 实例"""
    global _judger
    if _judger is None:
        config = reward_config or RewardConfig()
        _judger = Judger(
            model_name=config.judger_model,
            base_url=config.judger_base_url,
        )
        print(f"[Judger] 初始化完成: {config.judger_model} @ {config.judger_base_url}")
    return _judger


# -----------------------------------------------------------------------------
# Reward 函数定义
#
# 参考 agentic_rl_trl.ipynb:
# - 每个函数接收 completions 和其他参数
# - 返回 List[float]，每个样本一个奖励值
# -----------------------------------------------------------------------------

def judger_reward(
    completions: List[List[Dict]],
    ground_truth_diagnosis: List[List[str]],
    ground_truth_treatment: List[List[str]],
    ground_truth_avoid: List[List[str]],
    chief_complaint: List[str],
    case_id: List[str],
    reward_config: Optional[RewardConfig] = None,
    **kwargs
) -> List[float]:
    """
    使用 Judger (Baichuan-M2 + ground_truth) 进行综合评分

    对诊断和治疗两个维度进行评估（安全分已去除）:
    - diagnosis_score (1-5): 诊断准确性
    - treatment_score (1-5): 治疗合理性

    综合奖励 = diagnosis_weight * diagnosis_score +
               treatment_weight * treatment_score

    Args:
        completions: 模型生成的完成序列（每个样本是一个消息列表）
        ground_truth_diagnosis: 标准诊断列表
        ground_truth_treatment: 标准治疗列表
        ground_truth_avoid: 禁忌项目列表（保留用于兼容性，不参与奖励）
        chief_complaint: 主诉列表
        case_id: 病例ID列表
        reward_config: 奖励配置

    Returns:
        List[float]: 每个样本的奖励值
    """
    config = reward_config or RewardConfig()
    judger = get_judger(config)
    rewards = []

    for completion, gt_diag, gt_treat, gt_avoid, complaint, cid in zip(
        completions, ground_truth_diagnosis, ground_truth_treatment,
        ground_truth_avoid, chief_complaint, case_id
    ):
        reward = 0.0

        if completion and len(completion) > 0:
            # 提取轨迹（从工具调用构建）
            trajectory = _extract_trajectory_from_completion(completion)

            # 提取最终诊断和治疗建议
            agent_diagnosis, agent_treatment = _extract_final_output(completion)

            # 构建 ground_truth dict
            ground_truth = {
                "diagnosis": gt_diag if gt_diag else [],
                "treatment": gt_treat if gt_treat else [],
                "avoid": gt_avoid if gt_avoid else [],
            }

            # 调用 Judger 进行评分
            try:
                eval_result = judger.evaluate(
                    case_id=cid,
                    chief_complaint=complaint,
                    ground_truth=ground_truth,
                    trajectory=trajectory,
                    agent_diagnosis=agent_diagnosis,
                    agent_treatment=agent_treatment,
                )

                # 计算综合奖励（仅诊断+治疗，不含安全分）
                reward += config.diagnosis_weight * eval_result.diagnosis_score
                reward += config.treatment_weight * eval_result.treatment_score

            except Exception as e:
                # Judger 调用失败，使用 fallback 简单匹配
                print(f"[Judger Reward Warning] Evaluation failed for {cid}: {e}")
                reward = _fallback_reward(
                    gt_diag, gt_treat, gt_avoid,
                    agent_diagnosis, agent_treatment,
                    config
                )

        rewards.append(reward)

    return rewards


def _extract_trajectory_from_completion(completion: List[Dict]) -> List[Dict]:
    """
    从 completion 消息列表提取轨迹

    将工具调用转换为轨迹格式
    """
    trajectory = []

    for turn in completion:
        if turn.get("role") == "assistant":
            # 检查是否有工具调用
            if turn.get("tool_calls"):
                for call in turn["tool_calls"]:
                    func_name = call.get("function", {}).get("name", "")
                    func_args = call.get("function", {}).get("arguments", "{}")

                    # 解析参数
                    try:
                        args = json.loads(func_args) if isinstance(func_args, str) else func_args
                    except:
                        args = {}

                    # 构建轨迹步骤
                    if func_name == "tool_ask":
                        keywords = args.get("keywords", "").split(",")
                        trajectory.append({
                            "parsed": {"action": "ASK", "keywords": keywords},
                            "observation": ""  # RL 训练时无真实响应
                        })

                    elif func_name == "tool_exam":
                        keywords = args.get("keywords", "").split(",")
                        trajectory.append({
                            "parsed": {"action": "EXAM", "keywords": keywords},
                            "observation": ""
                        })

                    elif func_name == "tool_knowledge":
                        query = args.get("query", "")
                        trajectory.append({
                            "parsed": {"action": "KNOWLEDGE", "query": query},
                            "observation": ""
                        })

            # 检查是否是 FINAL 输出（包含诊断和治疗）
            content = turn.get("content", "")
            if content:
                diag_match = re.search(r'\*诊断[:：]?\s*(.+?)(?:\n|$|\*治疗)', content)
                treat_match = re.search(r'\*治疗[:：]?\s*(.+?)(?:\n|$)', content)

                if diag_match or treat_match:
                    trajectory.append({
                        "parsed": {
                            "action": "FINAL",
                            "diagnosis": diag_match.group(1).strip() if diag_match else "",
                            "treatment": treat_match.group(1).strip() if treat_match else "",
                        },
                        "observation": ""
                    })

    return trajectory


def _extract_final_output(completion: List[Dict]) -> Tuple[str, str]:
    """
    从 completion 提取最终的诊断和治疗建议

    Returns:
        (agent_diagnosis, agent_treatment)
    """
    agent_diagnosis = ""
    agent_treatment = ""

    # 从最后一轮 assistant 内容提取
    for turn in reversed(completion):
        if turn.get("role") == "assistant":
            content = turn.get("content", "")
            if content:
                # 提取诊断
                diag_match = re.search(r'\*诊断[:：]?\s*(.+?)(?:\n|$|\*治疗)', content)
                if diag_match:
                    agent_diagnosis = diag_match.group(1).strip()

                # 提取治疗
                treat_match = re.search(r'\*治疗[:：]?\s*(.+?)(?:\n|$)', content)
                if treat_match:
                    agent_treatment = treat_match.group(1).strip()

                # 如果没有找到格式化的输出，使用整个内容
                if not agent_diagnosis and not agent_treatment:
                    agent_diagnosis = content[:200]
                    agent_treatment = content[:200]

            break

    return agent_diagnosis, agent_treatment


def _fallback_reward(
    gt_diag: List[str],
    gt_treat: List[str],
    gt_avoid: List[str],
    agent_diagnosis: str,
    agent_treatment: str,
    config: RewardConfig,
) -> float:
    """
    当 Judger 调用失败时的备选奖励计算

    使用简单的字符串匹配（仅诊断+治疗，不含安全分）
    """
    reward = 0.0

    # 诊断匹配
    if gt_diag:
        for diag in gt_diag:
            if isinstance(diag, str) and diag.lower() in agent_diagnosis.lower():
                reward += config.diagnosis_weight * 4.0  # 相当于 4/5 分
                break
        else:
            reward += config.diagnosis_weight * 1.0  # 最低分

    # 治疗匹配
    if gt_treat:
        for treat in gt_treat:
            if isinstance(treat, str) and treat.lower() in agent_treatment.lower():
                reward += config.treatment_weight * 4.0
                break
        else:
            reward += config.treatment_weight * 1.0

    return reward


def tool_efficiency_reward(
    completions: List[List[Dict]],
    reward_config: Optional[RewardConfig] = None,
    **kwargs
) -> List[float]:
    """
    工具使用效率奖励

    规则（基于 Python 实现）:
    1. 至少要有 ASK + EXAM 调用（>=2 次不同的工具类型）
    2. 满足最低要求的基础上，调用总数越低奖励越大
    3. 无工具调用或只有单一工具类型：给负奖励

    Args:
        completions: 模型生成的完成序列
        reward_config: 奖励配置

    Returns:
        List[float]: 每个样本的奖励值
    """
    config = reward_config or RewardConfig()
    rewards = []

    for completion in completions:
        reward = 0.0
        tool_types_used = set()
        total_tool_calls = 0

        for turn in completion:
            if turn.get("role") == "assistant" and turn.get("tool_calls"):
                for call in turn["tool_calls"]:
                    total_tool_calls += 1
                    name = call.get("function", {}).get("name", "")
                    if name == "tool_ask":
                        tool_types_used.add("ASK")
                    elif name == "tool_exam":
                        tool_types_used.add("EXAM")
                    elif name == "tool_knowledge":
                        tool_types_used.add("KNOWLEDGE")

        has_ask_exam = ("ASK" in tool_types_used) and ("EXAM" in tool_types_used)

        if has_ask_exam and total_tool_calls >= config.min_tool_calls:
            # 满足最低要求：基准奖励 + 调用数越少的额外奖励
            # 额外奖励 = scale / (total_tool_calls + 1)
            reward = config.tool_base_reward + config.efficiency_scale / (total_tool_calls + 1)
        else:
            # 未满足最低要求：负奖励
            reward = -config.tool_base_reward

        rewards.append(reward)

    return rewards


def cost_reward(
    completions: List[List[Dict]],
    reward_config: Optional[RewardConfig] = None,
    **kwargs
) -> List[float]:
    """
    成本奖励（组内排序）

    对于同组（batch）样本：
    1. 从 completion 提取轨迹
    2. 使用 CostEvaluator 计算每条轨迹的费用
    3. 组内排序：cost 越低排名越靠前
    4. 奖励 = scale * (1 - rank / (N-1))，最低 cost 得满分，最高得 0

    Args:
        completions: 模型生成的完成序列
        reward_config: 奖励配置

    Returns:
        List[float]: 每个样本的奖励值
    """
    config = reward_config or RewardConfig()
    n = len(completions)

    if n <= 1:
        return [0.0] * n

    # 计算每条轨迹的 cost
    costs = []
    for completion in completions:
        trajectory = _extract_trajectory_from_completion(completion)
        if trajectory:
            # 从轨迹中估算 cost（基于规则：检查项和问诊项计数）
            cost = _estimate_trajectory_cost(trajectory)
        else:
            cost = float("inf")  # 无轨迹的给最高 cost
        costs.append(cost)

    # 组内排序，计算排名奖励
    # rank 0 = 最低 cost，rank N-1 = 最高 cost
    sorted_indices = sorted(range(n), key=lambda i: costs[i])
    ranks = [0] * n
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank

    # 奖励归一化到 [0, scale]
    max_rank = n - 1
    rewards = [config.cost_reward_scale * (1.0 - r / max_rank) for r in ranks]

    return rewards


def _estimate_trajectory_cost(trajectory: List[Dict]) -> float:
    """
    基于轨迹估算费用（规则实现，不调用 API）

    估算策略：
    - 每个 EXAM 项目: 50 元（平均检查费用）
    - 每个 ASK 项目: 10 元（问诊成本，时间等）
    - 每个 KNOWLEDGE 查询: 5 元

    这只是一个 proxy cost，用于训练时的组内排序。
    精确的费用评估需要在评测阶段用 CostEvaluator + Baichuan API。
    """
    cost = 0.0
    for step in trajectory:
        action = step.get("parsed", {}).get("action", "")
        keywords = step.get("parsed", {}).get("keywords", [])

        if action == "EXAM":
            cost += len(keywords) * 50.0
        elif action == "ASK":
            cost += len(keywords) * 10.0
        elif action == "KNOWLEDGE":
            cost += 5.0

    return cost


def build_reward_functions(reward_config: RewardConfig) -> List:
    """
    根据配置构建奖励函数列表

    Args:
        reward_config: 奖励配置

    Returns:
        List: 奖励函数列表
    """
    reward_funcs = []

    # 核心奖励：Judger 评分（诊断、治疗、安全）
    if reward_config.enable_judger:
        reward_funcs.append(judger_reward)

    # 工具效率奖励
    if reward_config.enable_tool_efficiency:
        reward_funcs.append(tool_efficiency_reward)

    # 成本奖励
    if reward_config.enable_cost_reward:
        reward_funcs.append(cost_reward)

    print(f"[Reward] 已启用 {len(reward_funcs)} 个奖励函数: judger={reward_config.enable_judger}(诊断+治疗), efficiency={reward_config.enable_tool_efficiency}, cost={reward_config.enable_cost_reward}")
    return reward_funcs


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
        default="data/train.jsonl",
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
    
    # LoRA 参数
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="是否使用 LoRA 微调",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA 秩",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    
    # vLLM 参数
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="是否使用 vLLM 加速推理",
    )
    
    # 知识库参数
    parser.add_argument(
        "--kb-path",
        type=str,
        default="data/knowledge_db",
        help="知识库路径",
    )
    parser.add_argument(
        "--disable-kb",
        action="store_true",
        help="禁用知识库（使用模拟响应，适用于 RL 训练）",
    )
    
    # 奖励配置参数
    parser.add_argument(
        "--enable-cost-reward",
        action="store_true",
        help="是否启用成本奖励",
    )
    parser.add_argument(
        "--disable-tool-efficiency",
        action="store_true",
        help="是否禁用工具效率奖励",
    )
    parser.add_argument(
        "--disable-judger",
        action="store_true",
        help="是否禁用 Judger 评分（使用简单匹配作为备选）",
    )
    parser.add_argument(
        "--judger-model",
        type=str,
        default="Baichuan-M2",
        help="Judger 使用的模型名称",
    )
    parser.add_argument(
        "--judger-base-url",
        type=str,
        default="https://api.baichuan-ai.com/v1",
        help="Judger API 地址",
    )

    # SwanLab 日志参数
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="SwanLab run 名称（不指定则自动生成）",
    )
    parser.add_argument(
        "--use-swanlab",
        action="store_true",
        help="是否使用 SwanLab 记录训练日志",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MedAgent Agentic RL Training with GRPO")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"数据: {args.data}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练步数: {args.max_steps}")
    print(f"批次大小: {args.batch_size}")
    print(f"使用 LoRA: {args.use_lora}")
    print(f"使用 vLLM: {args.use_vllm}")
    print("=" * 60)
    
    # 检查 TRL 是否安装
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("\n错误: TRL 未安装。请运行以下命令安装:")
        print("  pip install trl[vllm] transformers accelerate peft")
        print("\n如果上述命令失败，请分别安装:")
        print("  pip install trl transformers accelerate peft")
        return
    
    # 初始化知识库
    print("\n[0/4] 初始化知识库...")
    if not args.disable_kb:
        kb = get_knowledge_base(args.kb_path)
        if kb is None:
            print("  知识库未加载，将使用模拟响应")
    else:
        print("  知识库已禁用，使用模拟响应进行 RL 训练")
    
    # 加载数据
    print("\n[1/4] 加载训练数据...")
    train_dataset = load_training_data(args.data, args.max_samples)
    
    # 配置奖励
    print("\n[2/4] 配置奖励函数...")
    reward_config = RewardConfig(
        enable_cost_reward=args.enable_cost_reward,
        enable_tool_efficiency=not args.disable_tool_efficiency,
        enable_judger=not args.disable_judger,
        judger_model=args.judger_model,
        judger_base_url=args.judger_base_url,
    )

    # 预初始化 Judger（如果启用）
    if reward_config.enable_judger:
        get_judger(reward_config)

    reward_funcs = build_reward_functions(reward_config)

    # 配置 GRPO
    print("\n[3/4] 配置 GRPO...")

    # SwanLab 配置
    use_swanlab = args.use_swanlab
    if use_swanlab:
        # 从环境变量获取 project name，默认 MedAgentEnv
        swanlab_project_name = os.getenv("SWANLAB_PROJECT_NAME", "MedAgentEnv")

        # 生成 run_name（如果未手动指定）
        if args.run_name:
            run_name = args.run_name
        else:
            # 自动生成：模型名+时间戳
            model_short_name = args.model.split("/")[-1]  # 如 Qwen2.5-3B
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_name = f"{model_short_name}_{timestamp}"

        print(f"  SwanLab Project: {swanlab_project_name}")
        print(f"  SwanLab Run Name: {run_name}")

        # 设置环境变量供 SwanLab 使用
        os.environ["SWANLAB_PROJECT_NAME"] = swanlab_project_name
        os.environ["SWANLAB_RUN_NAME"] = run_name

        report_to = "swanlab"
    else:
        report_to = "none"
        run_name = None

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        chat_template_kwargs={"enable_thinking": False},

        # vLLM 配置
        use_vllm=args.use_vllm,
        vllm_mode="colocate" if args.use_vllm else None,
        vllm_enable_sleep_mode=False,

        # 日志与保存
        save_steps=50,
        logging_steps=10,
        log_completions=True,
        report_to=report_to,

        # SwanLab run name (通过 run_name 参数)
        run_name=run_name,

        # 显存优化
        bf16=True,
        tf32=True,
    )
    
    # LoRA 配置
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
        print(f"  LoRA 配置: r={args.lora_r}, alpha={args.lora_alpha}")
    
    # 创建 Trainer
    print("\n[4/4] 创建 GRPOTrainer...")
    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=train_dataset,
        tools=[tool_ask, tool_exam, tool_knowledge],
        reward_funcs=reward_funcs,
        args=grpo_config,
        peft_config=peft_config,
    )
    
    # 显示 GPU 信息
    import torch
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_memory = torch.cuda.max_memory_reserved() / 1024**3
        print(f"\nGPU 0: {gpu_stats.name}")
        print(f"显存总量: {gpu_stats.total_memory / 1024**3:.1f} GB")
        print(f"已预留显存: {start_memory:.2f} GB")
    
    # 开始训练
    print("\n" + "-" * 60)
    print("开始训练...")
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