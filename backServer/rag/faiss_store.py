"""FAISS 本地向量索引：构建、持久化、Top-K 检索。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag.chunking import TextChunk
from rag.config import RAG_CACHE_DIR
from rag.embedder import encode_texts

FAISS_DIR = RAG_CACHE_DIR / "faiss"
INDEX_PATH = FAISS_DIR / "index.faiss"
META_PATH = FAISS_DIR / "meta.json"


def _ensure_dir() -> None:
    FAISS_DIR.mkdir(parents=True, exist_ok=True)


def faiss_index_ready() -> bool:
    """磁盘上是否已有可加载的 FAISS 索引。"""
    return INDEX_PATH.is_file() and META_PATH.is_file()


def read_faiss_vector_dim() -> int | None:
    """不加载 SentenceTransformer，仅从已存在索引读取向量维度。"""
    if not faiss_index_ready():
        return None
    try:
        idx = faiss.read_index(str(INDEX_PATH))
        return int(idx.d)
    except Exception:
        return None


def build_faiss_index(chunks: list[TextChunk], vectors: np.ndarray) -> None:
    """用已算好的向量矩阵构建 Flat 内积索引（向量已 L2 归一化时等价于余弦相似度）。"""
    _ensure_dir()
    if vectors.ndim != 2:
        raise ValueError("vectors 须为二维矩阵")
    n, d = vectors.shape
    if n != len(chunks):
        raise ValueError("chunks 与 vectors 行数不一致")
    index = faiss.IndexFlatIP(d)
    index.add(vectors.astype(np.float32, copy=False))
    faiss.write_index(index, str(INDEX_PATH))
    payload = {
        "embedding_model": None,  # 由 meta 外层写入
        "chunks": [asdict(c) for c in chunks],
    }
    META_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class FaissChunkStore:
    """运行时持有 FAISS index 与块元数据。"""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self._chunks: list[dict[str, Any]] = []

    def load(self) -> None:
        """从磁盘加载；不存在则抛出 FileNotFoundError。"""
        if not faiss_index_ready():
            raise FileNotFoundError(f"未找到 FAISS 索引：{INDEX_PATH}")
        self._index = faiss.read_index(str(INDEX_PATH))
        data = json.loads(META_PATH.read_text(encoding="utf-8"))
        self._chunks = list(data.get("chunks") or [])

    @property
    def ready(self) -> bool:
        return self._index is not None and bool(self._chunks)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        """返回命中列表，含 score / text / source / chunk_index。"""
        if self._index is None:
            raise RuntimeError("FAISS 未加载")
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        k = min(top_k, len(self._chunks))
        if k <= 0:
            return []
        scores, ids = self._index.search(q, k)
        out: list[dict[str, Any]] = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0 or idx >= len(self._chunks):
                continue
            row = self._chunks[idx]
            out.append(
                {
                    "score": float(score),
                    "text": row.get("text", ""),
                    "source": row.get("source", ""),
                    "chunk_index": int(row.get("chunk_index", -1)),
                    "vector_id": int(idx),
                }
            )
        return out


def rebuild_faiss_from_chunks(chunks: list[TextChunk], *, embedding_model: str) -> None:
    """抽取文本 → 编码 → 写入 FAISS + meta.json。"""
    texts = [c.text for c in chunks]
    # 并且 自动做归一化（余弦相似度必备）
    vecs = encode_texts(texts, batch_size=16)
    build_faiss_index(chunks, vecs)
    # 写入模型名，便于排查版本不一致
    data = json.loads(META_PATH.read_text(encoding="utf-8"))
    data["embedding_model"] = embedding_model
    META_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_or_none() -> FaissChunkStore | None:
    """若索引存在则加载，否则返回 None。"""
    if not faiss_index_ready():
        return None
    st = FaissChunkStore()
    st.load()
    return st
