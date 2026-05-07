"""句向量编码：懒加载 SentenceTransformer，批推理。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rag.config import RAG_EMBEDDING_MODEL

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """单例加载嵌入模型（首次调用会下载权重，请保持网络或使用本地缓存）。"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer as ST

        _model = ST(RAG_EMBEDDING_MODEL)
    return _model


def embedding_dim() -> int:
    """当前模型的向量维度。"""
    m = get_embedder()
    return int(m.get_sentence_embedding_dimension())


def encode_texts(texts: list[str], *, batch_size: int = 32) -> np.ndarray:
    """将一批字符串编码为 float32 矩阵 (N, D)，并已 L2 归一化（便于余弦 / 内积）。"""
    if not texts:
        return np.zeros((0, embedding_dim()), dtype=np.float32)
    m = get_embedder()
    emb = m.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    arr = np.asarray(emb, dtype=np.float32)
    return arr
