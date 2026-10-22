"""路径与运行参数（可通过环境变量覆盖）。"""

from __future__ import annotations

import os
from pathlib import Path

# 仓库根目录（backServer 的上一级）
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 待向量化的财报 PDF 目录
DOC_CAIBAO_DIR = Path(
    os.environ.get("RAG_DOC_CAIBAO_DIR", str(REPO_ROOT / "doc" / "caibao"))
).expanduser().resolve()

# 向量索引与元数据缓存目录（默认在 backServer/.rag_cache）
_RAG_CACHE_DEFAULT = Path(__file__).resolve().parent.parent / ".rag_cache"
RAG_CACHE_DIR = Path(os.environ.get("RAG_CACHE_DIR", str(_RAG_CACHE_DEFAULT))).expanduser().resolve()

# 句向量模型（BGE 小模型，中文检索常用；维度 512）
# 原默认（多语言 MiniLM，384 维）：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RAG_EMBEDDING_MODEL = os.environ.get(
    "RAG_EMBEDDING_MODEL",
    "BAAI/bge-small-zh-v1.5",
).strip()

# 文本分块：单块最大字符数、块间重叠（仅 legacy 滑窗使用）
RAG_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "900"))
RAG_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "120"))

# 财报分块（chunk_text_v2）：目标块最大字符、块间重叠、过短原子块合并阈值（越小越碎）
RAG_CHUNK_V2_MAX_CHARS = int(os.environ.get("RAG_CHUNK_V2_MAX_CHARS", str(RAG_CHUNK_SIZE)))
RAG_CHUNK_V2_OVERLAP = int(os.environ.get("RAG_CHUNK_V2_OVERLAP", str(RAG_CHUNK_OVERLAP)))
RAG_CHUNK_V2_MIN_ATOMIC_CHARS = int(os.environ.get("RAG_CHUNK_V2_MIN_ATOMIC_CHARS", "80"))

# Elasticsearch
ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://127.0.0.1:9200").strip()
# 索引名：财报块 + 稠密向量
ES_INDEX_NAME = os.environ.get("RAG_ES_INDEX", "caibao_rag_chunks").strip()

# PDF 单页最大字符（防止异常页拖垮内存）
RAG_PDF_PAGE_MAX_CHARS = int(os.environ.get("RAG_PDF_PAGE_MAX_CHARS", "12000"))
