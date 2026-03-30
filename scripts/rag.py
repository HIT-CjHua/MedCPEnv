# MedAgent/scripts/rag.py

"""
RAG 知识库命令行工具

使用方式:
    # 建库
    python scripts/rag.py --build --data data/knowledge_dataset/ResponseMed.json --db data/knowledge_db

    # 检索
    python scripts/rag.py --search "高血压的治疗方案" --db data/knowledge_db --top-k 10 --rerank 3
"""

import os
import sys
import argparse

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.medagent.knowledge_base import KnowledgeBase


def main():
    parser = argparse.ArgumentParser(description="RAG Knowledge Base CLI")
    parser.add_argument("--build", action="store_true", help="Build knowledge base")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--data", type=str, default="data/knowledge_dataset/ResponseMed.json")
    parser.add_argument("--db", type=str, default="data/knowledge_db")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank", type=int, default=3)
    args = parser.parse_args()

    kb = KnowledgeBase()

    if args.build:
        kb.build(data_path=args.data, save_path=args.db)
    elif args.search:
        kb.load(args.db)
        results = kb.search(args.search, top_k=args.top_k, rerank_top_n=args.rerank)
        print(f"\n=== Search: {args.search} ===\n")
        for i, (doc, score, meta) in enumerate(results):
            print(f"[{i+1}] Score: {score:.4f}")
            print(f"{doc[:200]}...\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()