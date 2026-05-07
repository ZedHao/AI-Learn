"""
RAG 编排：扫描 doc/caibao 下 PDF → 分块（chunk_text_v2）→ 句向量 → 写入 FAISS / 可选 ES。

检索结果在 main._generate_sse 中注入首条 system，并通过 SSE type=rag 把 request/response 推给前端。
"""

from __future__ import annotations

import time
from typing import Any, Literal

from rag.chunking import TextChunk, chunk_text_v2
from rag.config import (
    DOC_CAIBAO_DIR,
    ELASTICSEARCH_URL,
    ES_INDEX_NAME,
    RAG_CHUNK_V2_MAX_CHARS,
    RAG_EMBEDDING_MODEL,
)
from rag.embedder import encode_texts
from rag.es_store import build_es_from_chunks, es_index_ready, es_ping, search_elasticsearch
from rag.faiss_store import (
    FaissChunkStore,
    faiss_index_ready,
    load_or_none,
    read_faiss_vector_dim,
    rebuild_faiss_from_chunks,
)
from rag.pdf_text import extract_pdf_text, iter_caibao_pdfs

RagBackend = Literal["faiss", "elasticsearch"]

_rag_singleton: RagService | None = None


def _rag_index_log(msg: str) -> None:
    """索引重建过程日志（控制台，与 [qwen-chat] 风格一致）。"""
    print(f"[rag-index] {msg}", flush=True)


class RagService:
    """对外暴露：构建索引、状态、检索上下文。"""

    def __init__(self) -> None:
        self._faiss: FaissChunkStore | None = None

    def _ensure_faiss_loaded(self) -> None:
        if self._faiss is not None and self._faiss.ready:
            return
        self._faiss = load_or_none()

    def status(self) -> dict[str, Any]:
        """供 /api/rag/status 使用：各后端是否就绪、块数量等。"""
        self._ensure_faiss_loaded()
        faiss_chunks = self._faiss.chunk_count if self._faiss and self._faiss.ready else 0
        return {
            "doc_dir": str(DOC_CAIBAO_DIR),
            "embedding_model": RAG_EMBEDDING_MODEL,
            "chunk_max_chars": RAG_CHUNK_V2_MAX_CHARS,
            # 仅从磁盘索引读取维度，避免 /api/rag/status 触发下载嵌入模型
            "embedding_dim": read_faiss_vector_dim(),
            "faiss": {
                "index_path_ok": faiss_index_ready(),
                "loaded": bool(self._faiss and self._faiss.ready),
                "chunk_count": faiss_chunks,
            },
            "elasticsearch": {
                "url": ELASTICSEARCH_URL,
                "ping": es_ping(),
                "index_ready": es_index_ready(),
            },
        }

    def build_all(self, *, write_es: bool) -> dict[str, Any]:
        """从 doc/caibao 重建向量索引；write_es=True 且 ES 可达时同步写入 ES。"""
        t_all = time.perf_counter()
        _rag_index_log(
            f"开始重建索引 | doc_dir={DOC_CAIBAO_DIR} | write_elasticsearch={write_es} | "
            f"embedding_model={RAG_EMBEDDING_MODEL}"
        )
        pdfs = iter_caibao_pdfs(DOC_CAIBAO_DIR)
        if not pdfs:
            _rag_index_log(f"失败：目录下无 PDF — {DOC_CAIBAO_DIR}")
            raise FileNotFoundError(f"目录下未发现 PDF：{DOC_CAIBAO_DIR}")

        _rag_index_log(f"发现 PDF {len(pdfs)} 个，开始抽取文本与分块（chunk_text_v2）…")
        all_chunks: list[TextChunk] = []
        per_file: list[dict[str, Any]] = []
        for p in pdfs:
            text = extract_pdf_text(p)
            chs = chunk_text_v2(text, source=p.name)
            all_chunks.extend(chs)
            per_file.append({"file": p.name, "chars": len(text), "chunks": len(chs)})
            _rag_index_log(f"  · {p.name} | 字符 {len(text)} | 块 {len(chs)}")

        if not all_chunks:
            _rag_index_log("失败：所有 PDF 均未解析出有效文本")
            raise RuntimeError("所有 PDF 均未解析出有效文本，无法建立索引")

        _rag_index_log(
            f"合计块数 {len(all_chunks)}，开始句向量编码并写入 FAISS（首次会加载嵌入模型，可能较慢）…"
        )
        t_embed = time.perf_counter()
        # 先写 FAISS（同时触发嵌入模型加载）
        rebuild_faiss_from_chunks(all_chunks, embedding_model=RAG_EMBEDDING_MODEL)
        _rag_index_log(f"FAISS 已落盘 | 嵌入+建索引耗时 {time.perf_counter() - t_embed:.1f}s")
        self._faiss = load_or_none()

        es_note = "skipped"
        if write_es:
            if not es_ping():
                es_note = "skipped_es_unreachable"
                _rag_index_log(f"Elasticsearch 不可达（{ELASTICSEARCH_URL}），跳过写入 ES")
            else:
                _rag_index_log(f"开始写入 Elasticsearch 索引 {ES_INDEX_NAME!r}…")
                t_es = time.perf_counter()
                build_es_from_chunks(all_chunks, embedding_model=RAG_EMBEDDING_MODEL)
                es_note = "ok"
                _rag_index_log(f"Elasticsearch 写入完成 | 耗时 {time.perf_counter() - t_es:.1f}s")
        else:
            _rag_index_log("未请求写入 Elasticsearch（write_elasticsearch=false）")

        _rag_index_log(
            f"索引重建结束 | 总耗时 {time.perf_counter() - t_all:.1f}s | "
            f"pdf={len(pdfs)} chunk={len(all_chunks)} | es={es_note}"
        )
        return {
            "pdf_count": len(pdfs),
            "chunk_count": len(all_chunks),
            "files": per_file,
            "faiss": "ok",
            "elasticsearch": es_note,
        }

    def retrieve(
        self,
        *,
        query: str,
        backend: RagBackend,
        top_k: int,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """执行检索，返回 (拼好的上下文字符串, 请求侧调试字典, 响应侧调试字典)。"""
        q = (query or "").strip()
        req_dbg: dict[str, Any] = {
            "backend": backend,
            "top_k": top_k,
            "query_text": q,
        }
        if not q:
            return "", req_dbg, {"hits": [], "error": "empty_query"}

        t0 = time.perf_counter()
        qv = encode_texts([q], batch_size=1)[0]

        if backend == "faiss":
            self._ensure_faiss_loaded()
            if not self._faiss or not self._faiss.ready:
                return "", req_dbg, {"hits": [], "error": "faiss_index_missing"}
            hits = self._faiss.search(qv, top_k=top_k)
        else:
            if not es_index_ready():
                return "", req_dbg, {"hits": [], "error": "elasticsearch_index_missing"}
            try:
                hits = search_elasticsearch(qv, top_k=top_k)
            except Exception as e:
                return "", req_dbg, {"hits": [], "error": str(e)}

        dt_ms = int((time.perf_counter() - t0) * 1000)
        resp_dbg: dict[str, Any] = {
            "hits": hits,
            "latency_ms": dt_ms,
            "embedding_model": RAG_EMBEDDING_MODEL,
            "vector_dim": int(qv.shape[0]),
        }

        # 组装进模型的上下文（控制长度，避免撑爆上下文）
        max_chars = 12000
        buf: list[str] = []
        used = 0
        for i, h in enumerate(hits, start=1):
            header = f"[片段{i} | 来源:{h.get('source','')} | score={h.get('score',0):.4f}]\n"
            piece = header + (h.get("text") or "").strip()
            if used + len(piece) + 2 > max_chars:
                break
            buf.append(piece)
            used += len(piece) + 2
        context = "\n\n".join(buf)
        return context, req_dbg, resp_dbg


def get_rag_service() -> RagService:
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = RagService()
    return _rag_singleton


def inject_rag_into_messages(messages: list[dict[str, Any]], context: str) -> list[dict[str, Any]]:
    """在「已合并默认 system」的 messages 上追加检索到的资料块（首条 system 为字符串时直接拼接）。"""
    if not (context or "").strip():
        return messages
    block = (
        "\n\n【检索到的本地财报资料（RAG）】\n"
        + context.strip()
        + "\n\n请结合上述资料作答；若资料与问题无关或不足，请明确说明并基于常识谨慎回答。"
    )
    out: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        if i == 0 and m.get("role") == "system":
            mm = dict(m)
            c = mm.get("content")
            if isinstance(c, str):
                mm["content"] = c + block
            elif isinstance(c, list):
                mm["content"] = [*c, {"type": "text", "text": block.strip()}]
            else:
                mm["content"] = str(c) + block
            out.append(mm)
        else:
            out.append(dict(m))
    return out if out else list(messages)


def last_user_plain_text(messages: list[dict[str, Any]]) -> str:
    """从 OpenAI 风格 messages 中取最近一条用户文本（忽略历史图片）。"""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            texts: list[str] = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "text":
                    texts.append(str(p.get("text") or ""))
            return " ".join(texts).strip()
    return ""
