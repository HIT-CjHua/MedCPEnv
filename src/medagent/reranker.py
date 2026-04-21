# MedAgent/src/medagent/reranker.py

"""
Reranker 客户端

使用 DashScope API 的 qwen3-rerank 模型进行文档重排序

参考代码来自 exp/model_list.md:
    import dashscope
    from http import HTTPStatus
    resp = dashscope.TextReRank.call(
        model="qwen3-rerank",
        query="什么是文本排序模型",
        documents=["文档1", "文档2", ...],
        top_n=10,
        return_documents=True,
        instruct="Given a web search query, retrieve relevant passages that answer the query."
    )
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

try:
    import dashscope
    from dashscope import TextReRank
    from http import HTTPStatus
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("[Reranker Warning] dashscope package not installed, reranker will not work")

load_dotenv()


class RerankerClient:
    """
    Reranker 客户端 (DashScope API)

    使用示例:
        client = RerankerClient()

        # 重排序
        results = client.rerank("什么是AI?", ["文档1", "文档2", "文档3"])

        # 单条打分 (通过 rerank 实现)
        score = client.compute_score("什么是AI?", "人工智能是...")
    """

    def __init__(
        self,
        # vLLM 本地部署配置 (已弃用，改用 DashScope API)
        # base_url: str = "http://localhost:8201/v1",
        # api_key: str = "EMPTY",
        # model_name: str = "Qwen3-reranker",
        model_name: str = "qwen3-rerank",
        api_key: str = None,  # 使用 DASHSCOPE_API_KEY 环境变量
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", None)
        self.instruction = instruction

        if self.api_key:
            dashscope.api_key = self.api_key

        if not DASHSCOPE_AVAILABLE:
            raise ImportError("dashscope package is required for RerankerClient. Install with: pip install dashscope")

    def compute_score(self, query: str, document: str) -> Optional[float]:
        """
        计算单条 query-document 相关性分数

        Args:
            query: 查询文本
            document: 文档文本

        Returns:
            float: 相关性分数 [0, 1]
        """
        if not query or not document:
            return 0.0

        try:
            resp = TextReRank.call(
                model=self.model_name,
                query=query,
                documents=[document],
                top_n=1,
                return_documents=True,
                instruct=self.instruction,
            )

            if resp.status_code == HTTPStatus.OK:
                results = resp.output.results
                if results:
                    # results[0] 是 ReRankResult 对象，有 relevance_score 属性
                    return results[0].relevance_score
                return 0.0
            else:
                print(f"[Reranker Error] {resp.message}")
                return None

        except Exception as e:
            print(f"[Reranker Error] {e}")
            return None

    def compute_score_batch(
        self,
        queries: List[str],
        documents: List[str],
    ) -> List[Optional[float]]:
        """
        批量计算相关性分数

        Args:
            queries: 查询文本列表
            documents: 文档文本列表

        Returns:
            List[float]: 相关性分数列表
        """
        if len(queries) != len(documents):
            raise ValueError("queries 和 documents 长度必须相同")

        scores = []
        for q, d in zip(queries, documents):
            score = self.compute_score(q, d)
            scores.append(score)
        return scores

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[tuple]:
        """
        重排序文档列表

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前 k 个结果（None 则返回全部）

        Returns:
            List[tuple]: [(index, score, document), ...] 按分数降序
        """
        if not query or not documents:
            return []

        try:
            top_n = top_k if top_k is not None else len(documents)

            resp = TextReRank.call(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=True,
                instruct=self.instruction,
            )

            if resp.status_code == HTTPStatus.OK:
                results = resp.output.results
                formatted_results = []
                for item in results:
                    # ReRankResult 对象支持 .get() 方法
                    index = item.get("index", 0)
                    score = item.get("relevance_score", 0.0)
                    # document 是包含 text 的字典
                    doc_dict = item.get("document", {})
                    doc = doc_dict.get("text", documents[index] if index < len(documents) else "")
                    formatted_results.append((index, score, doc))
                return formatted_results
            else:
                print(f"[Reranker Error] {resp.message}")
                return []

        except Exception as e:
            print(f"[Reranker Error] {e}")
            return []


if __name__ == "__main__":
    # 测试
    print("=== 测试 RerankerClient (DashScope qwen3-rerank) ===")
    client = RerankerClient()

    print("\n--- 单条打分测试 ---")
    score = client.compute_score(
        query="高血压的症状有哪些?",
        document="高血压的常见症状包括头痛、头晕、心悸等。",
    )
    print(f"Score: {score:.4f}")

    print("\n--- 重排序测试 ---")
    query = "高血压的症状有哪些?"
    docs = [
        "高血压的常见症状包括头痛、头晕、心悸等。",
        "糖尿病是一种代谢性疾病。",
        "低血压可能导致头晕和乏力。",
    ]
    results = client.rerank(query, docs, top_k=2)
    for idx, score, doc in results:
        print(f"[{idx}] {score:.4f}: {doc[:50]}...")