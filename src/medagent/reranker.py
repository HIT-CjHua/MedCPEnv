# MedAgent/src/medagent/reranker.py

"""
极简 Reranker 客户端

只负责调用 reranker API，不加载模型
支持 vLLM 部署的 Qwen3-Reranker（OpenAI 兼容接口）

参考 vllm reranker 实现:
- 添加 suffix (chat template 结束部分)
- 使用 allowed_token_ids 限制只输出 yes/no
- 从 logprobs 计算相关性分数
"""

import os
import math
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class RerankerClient:
    """
    Reranker 客户端

    使用示例:
        client = RerankerClient(base_url="http://localhost:8201/v1")

        # 单条打分
        score = client.compute_score("什么是AI?", "人工智能是...")

        # 批量重排序
        scores = client.rerank("什么是AI?", ["文档1", "文档2", "文档3"])
    """

    # Qwen3 chat template 的 suffix
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

    def __init__(
        self,
        base_url: str = "http://localhost:8201/v1",
        api_key: str = "EMPTY",
        model_name: str = "Qwen3-reranker",
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
        max_length: int = 8192,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.instruction = instruction
        self.max_length = max_length

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        # 初始化时获取 token id
        self._true_token_id = None
        self._false_token_id = None
        self._suffix_tokens = None
        self._init_token_ids()

    def _init_token_ids(self):
        """初始化 yes/no 的 token id 和 suffix tokens"""
        try:
            # 通过 tokenize API 获取 token id
            # vllm 支持 /tokenize 端点
            response = self._client.post(
                "/tokenize",
                json={"text": "yes"}
            )
            self._true_token_id = response.json().get("tokens", [0])[0]

            response = self._client.post(
                "/tokenize",
                json={"text": "no"}
            )
            self._false_token_id = response.json().get("tokens", [0])[0]

            response = self._client.post(
                "/tokenize",
                json={"text": self.SUFFIX}
            )
            self._suffix_tokens = response.json().get("tokens", [])

            print(f"[Reranker] true_token_id={self._true_token_id}, false_token_id={self._false_token_id}")
        except Exception as e:
            print(f"[Reranker] Warning: Could not get token ids, using defaults: {e}")
            self._true_token_id = 9453  # "yes" 的常见 token id
            self._false_token_id = 2732  # "no" 的常见 token id
            self._suffix_tokens = []

    def _build_messages(self, query: str, document: str) -> List[dict]:
        """构建 reranker prompt"""
        system_msg = (
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
        )
        user_msg = f"<Instruct>: {self.instruction}\n\n<Query>: {query}\n\n<Document>: {document}"
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

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

        messages = self._build_messages(query, document)

        try:
            # 添加 suffix 到最后一个消息
            messages_with_suffix = messages.copy()
            if messages_with_suffix and self._suffix_tokens:
                # 将 suffix 追加到 user 消息后，作为 assistant 响应的前缀
                pass  # OpenAI API 的 messages 不支持直接添加 suffix

            # 使用 extra_body 传递 vllm 特定参数
            extra_body = {}
            if self._true_token_id and self._false_token_id:
                extra_body["guided_decoding_backend"] = "tokens"
                extra_body["guided_allowed_tokens"] = [self._true_token_id, self._false_token_id]

            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=20,
                extra_body=extra_body if extra_body else None,
            )

            # 从 logprobs 提取 yes/no 概率
            last_token_logprobs = response.choices[0].logprobs.content[-1].top_logprobs
            yes_logprob = no_logprob = None

            for item in last_token_logprobs:
                token = item.token.strip().lower()
                if token == "yes":
                    yes_logprob = item.logprob
                elif token == "no":
                    no_logprob = item.logprob

            # 计算分数: softmax over yes/no
            yes_prob = math.exp(yes_logprob) if yes_logprob is not None else 1e-10
            no_prob = math.exp(no_logprob) if no_logprob is not None else 1e-10
            score = yes_prob / (yes_prob + no_prob)

            return min(max(score, 0.0), 1.0)

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

        scores = []
        for i, doc in enumerate(documents):
            score = self.compute_score(query, doc)
            if score is not None:
                scores.append((i, score, doc))

        scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scores = scores[:top_k]

        return scores


if __name__ == "__main__":
    # 测试
    client = RerankerClient()

    print("=== 测试单条打分 ===")
    score = client.compute_score(
        query="What is the capital of China?",
        document="Beijing is the capital of China.",
    )
    print(f"Score: {score:.4f}")

    print("\n=== 测试重排序 ===")
    query = "What is the capital of China?"
    docs = [
        "Beijing is the capital of China.",
        "Gravity is a force that attracts two bodies towards each other.",
        "Shanghai is the largest city in China.",
    ]
    results = client.rerank(query, docs, top_k=2)
    for idx, score, doc in results:
        print(f"[{idx}] {score:.4f}: {doc}")