"""
Elasticsearch 向量索引（与 FAISS 共用同一套分块与向量）。

- 建索引：dense_vector + cosine，bulk 写入每块文本与 embedding。
- 检索：kNN 查询，返回 hits 供 service 拼上下文与调试 JSON。
- 需本机或远程 ES 8+；URL 见 rag.config.ELASTICSEARCH_URL。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ApiError
from elasticsearch.helpers import bulk

from rag.chunking import TextChunk
from rag.config import ELASTICSEARCH_URL, ES_INDEX_NAME, RAG_CACHE_DIR
from rag.embedder import encode_texts

ES_META_PATH = RAG_CACHE_DIR / "elasticsearch" / "build_meta.json"


def _client() -> Elasticsearch:
    return Elasticsearch(ELASTICSEARCH_URL, request_timeout=60)


def es_ping() -> bool:
    """探测本机 ES 是否可达。"""
    try:
        return bool(_client().ping())
    except Exception:
        return False


def es_index_ready() -> bool:
    """索引是否已存在（且 mapping 合理）。"""
    if not es_ping():
        return False
    try:
        c = _client()
        if not c.indices.exists(index=ES_INDEX_NAME):
            return False
        return True
    except Exception:
        return False


def _mapping_body(dim: int) -> dict[str, Any]:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                # 与 ES 8 knn 检索字段一致：cosine
                "embedding": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "cosine",
                },
                "text": {"type": "text"},
                "source": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
            }
        },
    }


def rebuild_elasticsearch_index(chunks: list[TextChunk], vectors: np.ndarray, *, embedding_model: str) -> None:
    """批量写入 ES：先删后建索引，再 bulk 文档。"""
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("vectors 与 chunks 数量不一致")
    c = _client()
    if c.indices.exists(index=ES_INDEX_NAME):
        c.indices.delete(index=ES_INDEX_NAME)
    dim = int(vectors.shape[1])
    mb = _mapping_body(dim)
    c.indices.create(index=ES_INDEX_NAME, settings=mb["settings"], mappings=mb["mappings"])

    actions: list[dict[str, Any]] = []
    for i, (ch, row) in enumerate(zip(chunks, vectors, strict=True)):
        actions.append(
            {
                "_index": ES_INDEX_NAME,
                "_id": str(i),
                "_source": {
                    "embedding": row.astype(float).tolist(),
                    "text": ch.text,
                    "source": ch.source,
                    "chunk_index": ch.chunk_index,
                },
            }
        )
    bulk(c, actions, refresh=True, request_timeout=120)

    ES_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    ES_META_PATH.write_text(
        json.dumps(
            {"embedding_model": embedding_model, "chunks": len(chunks), "index": ES_INDEX_NAME},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def search_elasticsearch(query_vec: np.ndarray, top_k: int) -> list[dict[str, Any]]:
    """kNN 查询，返回与 FAISS 侧结构尽量一致的列表。"""
    c = _client()
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1).tolist()
    num_candidates = max(50, top_k * 20)
    try:
        resp = c.search(
            index=ES_INDEX_NAME,
            knn={
                "field": "embedding",
                "query_vector": q,
                "k": top_k,
                "num_candidates": num_candidates,
            },
            source=["text", "source", "chunk_index"],
            size=top_k,
        )
    except ApiError as e:
        raise RuntimeError(f"Elasticsearch 查询失败: {e}") from e

    hits = resp.get("hits", {}).get("hits", []) or []
    out: list[dict[str, Any]] = []
    for h in hits:
        src = h.get("_source") or {}
        out.append(
            {
                "score": float(h.get("_score") or 0.0),
                "text": src.get("text", ""),
                "source": src.get("source", ""),
                "chunk_index": int(src.get("chunk_index", -1)),
                "vector_id": h.get("_id"),
            }
        )
    return out


def build_es_from_chunks(chunks: list[TextChunk], *, embedding_model: str) -> None:
    """仅编码并写入 ES（向量矩阵由调用方传入亦可；此处内部编码）。"""
    texts = [c.text for c in chunks]
    vecs = encode_texts(texts, batch_size=16)
    rebuild_elasticsearch_index(chunks, vecs, embedding_model=embedding_model)
