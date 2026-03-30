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
from .knowledge_base import KnowledgeBase


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
            description="问诊工具，用于获取患者的主观信息（症状描述、病史等）",
            input_desc="keywords (List[str]) - 问诊关键词列表，如 ['疼痛性质', '疼痛部位', '持续时间']",
            output_desc="String - 患者的回答，如果未匹配到相关信息则返回'相关信息不明'",
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
            return "请提供问诊关键词。"

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
            return "相关信息不明。"

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
            description="检查工具，用于获取患者的客观信息（化验结果、影像学检查等）",
            input_desc="keywords (List[str]) - 检查关键词列表，如 ['血常规', '胸片', '心电图']",
            output_desc="String - 检查结果，如果未匹配到相关检查则返回'相关检查不适用'",
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
            return "请提供检查关键词。"

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
            return "相关检查不适用。"

        # 去重并返回
        unique_contents = list(dict.fromkeys(matched_contents))
        return "\n".join(unique_contents)


class KnowledgeTool(BaseTool):
    """
    知识库查询工具

    使用 query 在知识库中检索、重排、摘要
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        top_k: int = 10,
        rerank_top_n: int = 3,
        max_length: int = 500,
    ):
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.max_length = max_length

    @property
    def info(self) -> ToolInfo:
        return ToolInfo(
            name="KNOWLEDGE",
            description="知识库查询工具，用于检索医学知识",
            input_desc="query (str) - 查询内容，如 '急性阑尾炎的诊断标准'",
            output_desc="String - 相关知识的摘要",
        )

    def execute(self, query: str) -> str:
        """
        执行知识库查询

        Args:
            query: 查询内容

        Returns:
            str: 相关知识摘要
        """
        if not query:
            return "请提供查询内容。"

        if self.knowledge_base is None:
            return "知识库未初始化。"

        try:
            result = self.knowledge_base.search_with_summary(
                query=query,
                top_k=self.top_k,
                rerank_top_n=self.rerank_top_n,
                max_length=self.max_length,
            )
            return result
        except Exception as e:
            return f"知识库查询失败: {str(e)}"


class ToolManager:
    """
    工具管理器

    管理所有可用工具，提供统一的调用接口
    """

    def __init__(
        self,
        case: MedicalCase,
        knowledge_base: Optional[KnowledgeBase] = None,
        top_k: int = 10,
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
            return f"未知工具: {name}"
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