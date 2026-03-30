# MedAgent/src/medagent/knowledge_base.py

"""
RAG 知识库类

功能:
1. 建库: 加载 ResponseMed.json，去掉模板，存入 Chroma 向量数据库
2. 检索: Embedding + Chroma 查询
3. 重排: Reranker 精排

使用示例:
    from src.medagent import KnowledgeBase

    # 建库（首次运行）
    kb = KnowledgeBase()
    kb.build(save_path="data/knowledge_db")

    # 加载已有知识库
    kb = KnowledgeBase.load("data/knowledge_db")

    # 检索 + 重排
    results = kb.search("高血压的治疗方案", top_k=10, rerank_top_n=3)
"""

import os
import json
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings

from .embedding import EmbeddingClient
from .reranker import RerankerClient


@dataclass
class KnowledgeChunk:
    """知识块"""
    id: str
    content: str
    metadata: dict


class KnowledgeBase:
    """
    知识库类

    支持:
    - 从 ResponseMed.json 建库
    - Embedding 检索
    - Reranker 重排
    - 本地持久化
    """

    # 需要去掉的统一模板
    TEMPLATE_PATTERNS = [
        r"^Please answer the following multiple-choice question:\n?",
        r"^Please answer the following question:\n?",
    ]

    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
        reranker_client: Optional[RerankerClient] = None,
        collection_name: str = "medical_knowledge",
    ):
        self.embedding_client = embedding_client or EmbeddingClient()
        self.reranker_client = reranker_client  # 可选，不传则不做精排
        self.collection_name = collection_name
        self._chroma_client = None
        self._collection = None

    def _clean_template(self, text: str) -> str:
        """去掉统一模板"""
        for pattern in self.TEMPLATE_PATTERNS:
            text = re.sub(pattern, "", text)
        return text.strip()

    def _load_response_med(self, file_path: str) -> List[KnowledgeChunk]:
        """加载 ResponseMed.json 数据"""
        print(f"[Loading] {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        for idx, item in enumerate(data):
            # 合并 instruction + output，去掉模板
            instruction = self._clean_template(item.get("instruction", ""))
            output = item.get("output", "")

            # 构建内容
            content = f"问题: {instruction}\n\n回答: {output}"
            if not instruction:
                content = output

            if content.strip():
                chunks.append(KnowledgeChunk(
                    id=str(idx),
                    content=content,
                    metadata={
                        "source": "ResponseMed",
                        "has_question": bool(instruction),
                    }
                ))

            if (idx + 1) % 10000 == 0:
                print(f"  Loaded {idx + 1} chunks...")

        print(f"[Done] Total {len(chunks)} chunks loaded")
        return chunks

    def build(
        self,
        data_path: str = "data/knowledge_dataset/ResponseMed.json",
        save_path: str = "data/knowledge_db",
        batch_size: int = 10,
    ):
        """
        构建知识库

        Args:
            data_path: 数据文件路径
            save_path: 向量库保存路径
            batch_size: 批量处理大小
        """
        # 加载数据
        chunks = self._load_response_med(data_path)

        # 初始化 Chroma
        os.makedirs(save_path, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(
            path=save_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # 删除已有 collection
        try:
            self._chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass

        # 创建 collection
        self._collection = self._chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # 批量添加
        print(f"[Building] Adding to Chroma...")

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # 计算 embedding
            texts = [c.content for c in batch]
            embeddings = self.embedding_client.embed_batch(texts)

            # 添加到 Chroma
            self._collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c.metadata for c in batch],
            )

            if (i + batch_size) % 1000 == 0 or i + batch_size >= len(chunks):
                print(f"  Added {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

        print(f"[Done] Knowledge base saved to {save_path}")

    def load(self, save_path: str = "data/knowledge_db") -> "KnowledgeBase":
        """加载已有知识库"""
        self._chroma_client = chromadb.PersistentClient(
            path=save_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._chroma_client.get_collection(self.collection_name)
        print(f"[Loaded] Collection '{self.collection_name}' with {self._collection.count()} chunks")
        return self

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_n: Optional[int] = None,
    ) -> List[Tuple[str, float, dict]]:
        """
        检索知识

        Args:
            query: 查询文本
            top_k: 初始检索数量
            rerank_top_n: 重排后返回数量（None 则不重排）

        Returns:
            List[Tuple[document, score, metadata]]
        """
        if self._collection is None:
            raise ValueError("Knowledge base not loaded. Call build() or load() first.")

        # Embedding 检索
        query_embedding = self.embedding_client.embed(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )

        # 整理结果
        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        # Chroma 返回的是 distance，转换为 similarity
        candidates = [
            (doc, 1 - dist, meta)
            for doc, dist, meta in zip(documents, distances, metadatas)
        ]

        # 不需要重排
        if rerank_top_n is None:
            return candidates

        # Reranker 重排
        if self.reranker_client is None:
            print("[Warning] Reranker not available, returning embedding results only")
            return candidates[:rerank_top_n]

        reranked = self.reranker_client.rerank(
            query=query,
            documents=[c[0] for c in candidates],
            top_k=rerank_top_n,
        )

        # 返回重排结果
        return [
            (candidates[idx][0], score, candidates[idx][2])
            for idx, score, _ in reranked
        ]

    def search_with_summary(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_n: int = 3,
        max_length: int = 500,
    ) -> str:
        """
        检索知识并摘要

        Args:
            query: 查询文本
            top_k: 初始检索数量
            rerank_top_n: 重排后保留数量
            max_length: 摘要最大长度

        Returns:
            str: 摘要后的知识内容
        """
        results = self.search(query, top_k=top_k, rerank_top_n=rerank_top_n)

        if not results:
            return "未找到相关知识。"

        # 拼接内容
        content_parts = []
        total_length = 0

        for doc, score, meta in results:
            if total_length + len(doc) > max_length:
                # 截断
                remaining = max_length - total_length
                if remaining > 100:
                    content_parts.append(doc[:remaining] + "...")
                break
            content_parts.append(doc)
            total_length += len(doc)

        return "\n\n---\n\n".join(content_parts)

    def count(self) -> int:
        """返回知识库中的文档数量"""
        if self._collection is None:
            return 0
        return self._collection.count()