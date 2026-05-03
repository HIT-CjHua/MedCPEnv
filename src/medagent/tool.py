# MedAgent/src/medagent/tool.py

"""
医疗问诊工具类

Tool基类 + Ask/Exam/Knowledge 工具实现

每个工具包含:
- name: 工具名称
- description: 工具描述
- input_desc: 输入参数描述
- output_desc: 输出描述
- execute(): 执行方法
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.schema import MedicalCase, MedicalItem
from .knowledge_tool_v2 import KeywordKnowledgeBase


@dataclass
class ToolInfo:
    """工具信息"""
    name: str
    description: str
    input_desc: str
    output_desc: str


class BaseTool(ABC):
    """工具基类"""

    @property
    @abstractmethod
    def info(self) -> ToolInfo:
        """返回工具信息"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具"""
        pass

    def get_prompt_desc(self) -> str:
        """生成用于 LLM prompt 的工具描述"""
        info = self.info
        return f"Name: {info.name}\nDescription: {info.description}\nInput: {info.input_desc}\nOutput: {info.output_desc}"


class AskTool(BaseTool):
    """
    问诊工具

    匹配 subjective 中的 keywords，返回对应 content
    """

    def __init__(self, case: MedicalCase):
        self.case = case

    @property
    def info(self) -> ToolInfo:
        return ToolInfo(
            name="ASK",
            description="Consultation tool to obtain patient's subjective information (symptoms, medical history, etc.)",
            input_desc="keywords (List[str]) - consultation keyword list, e.g. ['pain character', 'pain location', 'duration']",
            output_desc="String - patient's response, or 'Information unclear' if no match found",
        )

    def execute(self, keywords: List[str]) -> str:
        """
        执行问诊

        Args:
            keywords: 问诊关键词列表

        Returns:
            str: 匹配到的内容或"相关信息不明"
        """
        if not keywords:
            return "Please provide consultation keywords."

        matched_contents = []
        matched_items = []

        for item in self.case.subjective:
            # 检查是否有关键词匹配
            for kw in keywords:
                for item_kw in item.keywords:
                    if kw.lower() in item_kw.lower() or item_kw.lower() in kw.lower():
                        matched_contents.append(item.content)
                        matched_items.append(item)
                        break
                else:
                    continue
                break

        if not matched_contents:
            return "Information unclear."

        # 去重并返回
        unique_contents = list(dict.fromkeys(matched_contents))
        return "\n".join(unique_contents)


class ExamTool(BaseTool):
    """
    检查工具

    匹配 objective 中的 keywords，返回对应 content
    """

    def __init__(self, case: MedicalCase):
        self.case = case

    @property
    def info(self) -> ToolInfo:
        return ToolInfo(
            name="EXAM",
            description="Examination tool to obtain patient's objective information (lab tests, imaging, etc.)",
            input_desc="keywords (List[str]) - examination keyword list, e.g. ['CBC', 'chest X-ray', 'ECG']",
            output_desc="String - examination results, or 'Examination not applicable' if no match found",
        )

    def execute(self, keywords: List[str]) -> str:
        """
        执行检查

        Args:
            keywords: 检查关键词列表

        Returns:
            str: 匹配到的内容或"相关检查不适用"
        """
        if not keywords:
            return "Please provide examination keywords."

        matched_contents = []

        for item in self.case.objective:
            # 检查是否有关键词匹配
            for kw in keywords:
                for item_kw in item.keywords:
                    if kw.lower() in item_kw.lower() or item_kw.lower() in kw.lower():
                        matched_contents.append(item.content)
                        break
                else:
                    continue
                break

        if not matched_contents:
            return "Examination not applicable."

        # 去重并返回
        unique_contents = list(dict.fromkeys(matched_contents))
        return "\n".join(unique_contents)


class KnowledgeTool(BaseTool):
    """
    知识库查询工具 (v2 - 关键词匹配)
    
    使用多个关键词在 ResponseMed.json 中检索完整 QA 数据
    返回 topk 条结果
    """

    def __init__(
        self,
        knowledge_base: Optional[KeywordKnowledgeBase] = None,
        top_k: int = 3,
    ):
        self.knowledge_base = knowledge_base
        self.top_k = top_k

    @property
    def info(self) -> ToolInfo:
        return ToolInfo(
            name="KNOWLEDGE",
            description="Knowledge base query tool using keyword matching to retrieve medical QA data",
            input_desc="keywords (List[str]) - keyword list for querying medical knowledge, e.g. ['appendicitis', 'diagnostic criteria', 'treatment']",
            output_desc="String - top-k matched QA records from knowledge base with full question and answer",
        )

    def execute(self, keywords: List[str]) -> str:
        """
        执行知识库查询

        Args:
            keywords: 查询关键词列表

        Returns:
            str: 匹配到的 QA 数据摘要
        """
        if not keywords:
            return "Please provide keywords for knowledge base query."

        if self.knowledge_base is None:
            return "Knowledge base not initialized."

        try:
            results = self.knowledge_base.search(
                keywords=keywords,
                top_k=self.top_k
            )
            return self.knowledge_base.format_results(results)
        except Exception as e:
            return f"Knowledge base query failed: {str(e)}"


class ToolManager:
    """
    工具管理器

    管理所有可用工具，提供统一的调用接口
    """

    def __init__(
        self,
        case: MedicalCase,
        knowledge_base: Optional[KeywordKnowledgeBase] = None,
        top_k: int = 3,
    ):
        self.tools = {
            "ASK": AskTool(case),
            "EXAM": ExamTool(case),
            "KNOWLEDGE": KnowledgeTool(knowledge_base, top_k=top_k),
        }

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self.tools.get(name.upper())

    def execute(self, name: str, **kwargs) -> str:
        """执行工具"""
        tool = self.get_tool(name)
        if tool is None:
            return f"Unknown tool: {name}"
        return tool.execute(**kwargs)

    def get_all_tools_prompt(self) -> str:
        """生成所有工具的 prompt 描述"""
        prompts = []
        for tool in self.tools.values():
            prompts.append(tool.get_prompt_desc())
        return "\n\n".join(prompts)

    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self.tools.keys())


if __name__ == "__main__":
    # 测试
    from src.schema import MedicalCase, MedicalItem, GroundTruth

    # 创建测试病例
    case = MedicalCase(
        case_id="test_001",
        difficulty="medium",
        tags=["内科"],
        chief_complaint="胸痛2小时",
        subjective=[
            MedicalItem(
                keywords=["疼痛性质", "胸痛特点"],
                content="胸痛为压榨样，位于胸骨后，向左肩放射",
                necessity=True,
            ),
            MedicalItem(
                keywords=["持续时间", "发病时间"],
                content="疼痛持续约2小时，休息后不缓解",
                necessity=True,
            ),
        ],
        objective=[
            MedicalItem(
                keywords=["心电图", "ECG"],
                content="心电图示ST段抬高，提示急性心肌梗死",
                necessity=True,
            ),
            MedicalItem(
                keywords=["心肌酶", "肌钙蛋白"],
                content="肌钙蛋白T 0.5ng/mL（升高）",
                necessity=True,
            ),
        ],
        ground_truth=GroundTruth(
            diagnosis=["急性心肌梗死"],
            treatment=["急诊PCI"],
            avoid=["运动负荷试验"],
        ),
    )

    # 测试 AskTool
    print("=== AskTool 测试 ===")
    ask_tool = AskTool(case)
    print(f"Info: {ask_tool.info}")
    print(f"Result: {ask_tool.execute(['疼痛性质'])}\n")

    # 测试 ExamTool
    print("=== ExamTool 测试 ===")
    exam_tool = ExamTool(case)
    print(f"Result: {exam_tool.execute(['心电图'])}\n")

    # 测试 ToolManager
    print("=== ToolManager 测试 ===")
    manager = ToolManager(case)
    print(f"Tools: {manager.list_tools()}")
    print(f"Result: {manager.execute('ASK', keywords=['持续时间'])}")