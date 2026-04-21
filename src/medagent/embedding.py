# MedAgent/src/medagent/embedding.py

"""
Embedding 客户端

使用 DashScope API 的 text-embedding-v4 模型进行文本向量嵌入

参考代码来自 exp/model_list.md:
    import dashscope
    from http import HTTPStatus
    resp = dashscope.TextEmbedding.call(
        model="text-embedding-v4",
        input=input_texts
    )
"""

import os
from typing import List
from dotenv import load_dotenv

try:
    import dashscope
    from dashscope import TextEmbedding
    from http import HTTPStatus
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("[Embedding Warning] dashscope package not installed")

import numpy as np

load_dotenv()


class EmbeddingClient:
    """
    Embedding 客户端 (DashScope API)

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
        # vLLM 本地部署配置 (已弃用，改用 DashScope API)
        # base_url: str = "http://localhost:8300/v1",
        # api_key: str = "EMPTY",
        # model_name: str = "Qwen3-embedding",
        model_name: str = "text-embedding-v4",
        api_key: str = None,  # 使用 DASHSCOPE_API_KEY 环境变量
        batch_size: int = 25,  # DashScope API 批量限制
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", None)
        self.batch_size = batch_size

        if self.api_key:
            dashscope.api_key = self.api_key

        if not DASHSCOPE_AVAILABLE:
            raise ImportError("dashscope package is required for EmbeddingClient. Install with: pip install dashscope")

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

        try:
            resp = TextEmbedding.call(
                model=self.model_name,
                input=text,
            )

            if resp.status_code == HTTPStatus.OK:
                embeddings = resp.output.get("embeddings", [])
                if embeddings:
                    return embeddings[0].get("embedding", [])
                return []
            else:
                print(f"[Embedding Error] {resp.message}")
                return []

        except Exception as e:
            print(f"[Embedding Error] {e}")
            return []

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

            try:
                resp = TextEmbedding.call(
                    model=self.model_name,
                    input=batch,
                )

                if resp.status_code == HTTPStatus.OK:
                    embeddings = resp.output.get("embeddings", [])
                    # 按 index 排序
                    sorted_embeddings = sorted(embeddings, key=lambda x: x.get("text_index", 0))
                    all_embeddings.extend([item.get("embedding", []) for item in sorted_embeddings])
                else:
                    print(f"[Embedding Error] Batch {i}: {resp.message}")
                    # 填充空 embedding
                    all_embeddings.extend([[] for _ in batch])

            except Exception as e:
                print(f"[Embedding Error] Batch {i}: {e}")
                all_embeddings.extend([[] for _ in batch])

        return all_embeddings

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0
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
            for i, doc_emb in enumerate(doc_embs) if doc_emb
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]


if __name__ == "__main__":
    # 测试
    print("=== 测试 EmbeddingClient (DashScope text-embedding-v4) ===")
    client = EmbeddingClient()

    print("\n--- 单条 embedding 测试 ---")
    emb = client.embed("你好")
    print(f"维度: {len(emb)}")

    print("\n--- 批量 embedding 测试 ---")
    embs = client.embed_batch(["你好", "世界", "测试"])
    print(f"数量: {len(embs)}, 维度: {len(embs[0]) if embs else 0}")

    print("\n--- 相似度测试 ---")
    score = client.cosine_similarity(embs[0], embs[1])
    print(f"相似度: {score:.4f}")

    print("\n--- 检索测试 ---")
    docs = ["苹果是一种水果", "北京是中国的首都", "今天天气不错"]
    results = client.search("水果", docs, top_k=2)
    for idx, score, doc in results:
        print(f"[{idx}] {score:.4f}: {doc}")