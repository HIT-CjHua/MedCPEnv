# MedAgent/src/medagent/embedding.py

"""
极简 Embedding 客户端

只负责调用 embedding API，不加载模型
"""

import os
from typing import List, Optional
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class EmbeddingClient:
    """
    Embedding 客户端

    使用示例:
        client = EmbeddingClient()

        # 单条
        emb = client.embed("你好")

        # 批量
        embs = client.embed_batch(["你好", "世界"])

        # 相似度
        score = client.cosine_similarity(emb1, emb2)

        # 检索
        results = client.search("水果", ["苹果是水果", "北京是首都"])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8300/v1",
        api_key: str = "EMPTY",
        model_name: str = "Qwen3-embedding",
        batch_size: int = 32,
    ):
        self.base_url = base_url
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY_CP", "EMPTY")
        self.model_name = model_name
        self.batch_size = batch_size

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def embed(self, text: str) -> List[float]:
        """
        单条文本 embedding

        Args:
            text: 输入文本

        Returns:
            List[float]: embedding 向量
        """
        if not text or not text.strip():
            return []

        response = self._client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量 embedding

        Args:
            texts: 文本列表

        Returns:
            List[List[float]]: embedding 向量列表
        """
        if not texts:
            return []

        # 过滤空文本
        valid_texts = [t if t and t.strip() else " " for t in texts]

        all_embeddings = []

        for i in range(0, len(valid_texts), self.batch_size):
            batch = valid_texts[i : i + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            # 按 index 排序
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([item.embedding for item in sorted_data])

        return all_embeddings

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[tuple]:
        """
        基于相似度的文档检索

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前 k 个结果

        Returns:
            List[tuple]: [(index, score, document), ...]
        """
        query_emb = self.embed(query)
        doc_embs = self.embed_batch(documents)

        scores = [
            (i, self.cosine_similarity(query_emb, doc_emb), documents[i])
            for i, doc_emb in enumerate(doc_embs)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


if __name__ == "__main__":
    # 测试
    client = EmbeddingClient()

    print("=== 测试单条 embedding ===")
    emb = client.embed("你好")
    print(f"维度: {len(emb)}")

    print("\n=== 测试批量 embedding ===")
    embs = client.embed_batch(["你好", "世界", "测试"])
    print(f"数量: {len(embs)}, 维度: {len(embs[0])}")

    print("\n=== 测试相似度 ===")
    score = client.cosine_similarity(embs[0], embs[1])
    print(f"相似度: {score:.4f}")

    print("\n=== 测试检索 ===")
    docs = ["苹果是一种水果", "北京是中国的首都", "今天天气不错"]
    results = client.search("水果", docs, top_k=2)
    for idx, score, doc in results:
        print(f"[{idx}] {score:.4f}: {doc}")