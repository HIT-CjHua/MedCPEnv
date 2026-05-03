# MedAgent/src/medagent/knowledge_tool_v2.py

"""
关键词匹配式知识库查询工具 v2

从 ResponseMed.json (alpaca格式) 中通过关键词匹配检索 QA 数据
- 不使用 embedding + reranker
- 纯关键词匹配
- 返回 topk 条完整 QA 数据
"""

import json
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class ToolInfo:
    """工具信息"""
    name: str
    description: str
    input_desc: str
    output_desc: str


class BaseTool:
    """工具基类（简化版，避免循环导入）"""
    
    @property
    def info(self) -> ToolInfo:
        raise NotImplementedError
    
    def execute(self, **kwargs) -> str:
        raise NotImplementedError
    
    def get_prompt_desc(self) -> str:
        info = self.info
        return f"Name: {info.name}\nDescription: {info.description}\nInput: {info.input_desc}\nOutput: {info.output_desc}"


@dataclass
class QARecord:
    """QA 记录"""
    instruction: str  # 问题
    output: str       # 回答
    full_text: str    # 完整文本 (用于匹配)


class KeywordKnowledgeBase:
    """
    关键词匹配知识库
    
    加载 ResponseMed.json，支持关键词匹配检索
    """
    
    # 模板前缀模式（用于清洗 instruction）
    TEMPLATE_PATTERNS = [
        r"^Please answer the following multiple-choice question:\n?",
        r"^Please answer the following question:\n?",
    ]
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "knowledge_dataset", "ResponseMed.json"
        )
        self.records: List[QARecord] = []
        self.is_loaded = False
    
    def load(self, data_path: Optional[str] = None) -> int:
        """
        加载知识库数据
        
        Args:
            data_path: ResponseMed.json 路径，可选
            
        Returns:
            int: 加载的记录数
        """
        path = data_path or self.data_path
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Knowledge data file not found: {path}")
        
        print(f"Loading knowledge data from: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.records = []
        for item in data:
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            
            # 清洗 instruction（去掉模板前缀）
            cleaned_instruction = self._clean_template(instruction)
            
            # 构建完整文本（用于匹配）
            full_text = f"{cleaned_instruction} {output}".lower()
            
            self.records.append(QARecord(
                instruction=cleaned_instruction,
                output=output,
                full_text=full_text
            ))
        
        self.is_loaded = True
        print(f"Loaded {len(self.records)} QA records")
        return len(self.records)
    
    def _clean_template(self, text: str) -> str:
        """去掉模板前缀"""
        for pattern in self.TEMPLATE_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text.strip()
    
    def search(
        self,
        keywords: List[str],
        top_k: int = 3
    ) -> List[Dict[str, str]]:
        """
        关键词匹配检索
        
        Args:
            keywords: 关键词列表
            top_k: 返回前 k 条结果
            
        Returns:
            List[Dict]: 匹配的 QA 记录列表
        """
        if not self.is_loaded:
            raise RuntimeError("Knowledge base not loaded. Call load() first.")
        
        if not keywords:
            return []
        
        # 计算每条记录的匹配分数
        scored_records = []
        for idx, record in enumerate(self.records):
            score = 0
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in record.full_text:
                    score += 1
            
            if score > 0:
                scored_records.append((idx, score, record))
        
        # 按分数降序排序
        scored_records.sort(key=lambda x: x[1], reverse=True)
        
        # 取 top_k
        top_records = scored_records[:top_k]
        
        # 构建返回结果
        results = []
        for idx, score, record in top_records:
            results.append({
                "instruction": record.instruction,
                "output": record.output,
                "match_score": score,
                "matched_keywords": [
                    kw for kw in keywords 
                    if kw.lower() in record.full_text
                ]
            })
        
        return results
    
    def format_results(self, results: List[Dict[str, str]]) -> str:
        """
        格式化检索结果为字符串
        
        Args:
            results: search() 返回的结果
            
        Returns:
            str: 格式化的结果字符串
        """
        if not results:
            return "未找到相关知识。"
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"--- 知识 {i} (匹配度: {result['match_score']}) ---\n"
                f"问题: {result['instruction']}\n\n"
                f"回答: {result['output']}"
            )
        
        return "\n\n".join(formatted)


class KeywordKnowledgeTool(BaseTool):
    """
    关键词匹配知识库查询工具 v2
    
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


if __name__ == "__main__":
    # 测试
    kb = KeywordKnowledgeBase()
    kb.load()
    
    # 测试查询
    print("\n=== 测试查询: appendicitis diagnosis ===")
    results = kb.search(["appendicitis", "diagnosis"], top_k=3)
    print(kb.format_results(results))
    
    print("\n=== 测试查询: hypertension treatment ===")
    results = kb.search(["hypertension", "treatment"], top_k=2)
    print(kb.format_results(results))
