"""
RAG 子包：财报 PDF 向量化检索。

- 语料目录默认：<仓库根>/doc/caibao（见 rag.config.DOC_CAIBAO_DIR）。
- 嵌入 + 分块后写入：FAISS 本地索引（rag.faiss_store）与可选 Elasticsearch（rag.es_store）。
- 对外编排：rag.service（构建索引、检索、把上下文注入对话）。
"""

from rag.service import RagService, get_rag_service

__all__ = ["RagService", "get_rag_service"]
