"""文本分块：索引管线仅用 chunk_text_v2；chunk_text 为历史滑窗实现，保留未引用。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.config import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_V2_MAX_CHARS,
    RAG_CHUNK_V2_MIN_ATOMIC_CHARS,
    RAG_CHUNK_V2_OVERLAP,
)


@dataclass(frozen=True)
class TextChunk:
    """单个文本块及其来源。"""

    text: str
    source: str
    chunk_index: int


def chunk_text(text: str, *, source: str) -> list[TextChunk]:
    """固定长度滑动窗口 + 重叠（历史实现，当前索引管线不再调用，仅保留作对照）。"""
    t = (text or "").strip()
    if not t:
        return []
    size = max(200, RAG_CHUNK_SIZE)
    overlap = max(0, min(RAG_CHUNK_OVERLAP, size // 2))
    step = max(1, size - overlap)
    out: list[TextChunk] = []
    i = 0
    idx = 0
    while i < len(t):
        piece = t[i : i + size].strip()
        if piece:
            out.append(TextChunk(text=piece, source=source, chunk_index=idx))
            idx += 1
        i += step
    return out


# ---------------------------------------------------------------------------
# v2：财报 PDF — 归一化 → 段落 → 句末标点拆长段 → 原子块合并 → 定长组块 + 重叠
# ---------------------------------------------------------------------------


def _normalize_report_text(text: str) -> str:
    """统一换行、压缩空白，保留空行作为段落边界。"""
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t\u3000]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r'^\s*(\d+)\s*$', '', t, flags=re.MULTILINE)  # 去页码

    return t.strip()


def _split_paragraphs(text: str) -> list[str]:
    """双换行分段；若几乎无段落则退化为按行拆（表格/目录多单行）。"""
    parts = re.split(r"\n\s*\n+", text)
    paras = [p.strip() for p in parts if p.strip()]
    if len(paras) <= 1 and text:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) >= 8:
            return lines
    return paras if paras else ([text] if text.strip() else [])


def _hard_split_oversized(segment: str, max_chars: int) -> list[str]:
    """单句/单行仍超长时在 max_chars 附近找软边界，否则硬切。"""
    s = segment.strip()
    if not s:
        return []
    if len(s) <= max_chars:
        return [s]
    out: list[str] = []
    start = 0
    while start < len(s):
        end = min(start + max_chars, len(s))
        if end < len(s):
            window = s[start:end]
            cut = max(
                window.rfind("，"),
                window.rfind("、"),
                window.rfind("；"),
                window.rfind(" "),
                window.rfind("%"),    # 百分比
                window.rfind("％"),   # 全角百分比
                window.rfind("亿"),   # 亿
                window.rfind("万"),   # 万
                window.rfind("元"),   # 元
                window.rfind("）"),   # 右括号
                window.rfind("."),    # 小数点
                window.rfind("\n"),
            )
            if cut > max_chars // 3:
                end = start + cut + 1
        piece = s[start:end].strip()
        if piece:
            out.append(piece)
        start = end if end > start else start + max_chars
    return out


def _atomic_units_from_paragraph(para: str, max_chars: int) -> list[str]:
    """单段内按中文句末标点拆成不超过 max_chars 的原子串（尽量不切半句）。"""
    p = para.strip()
    if not p:
        return []
    if len(p) <= max_chars:
        return [p]
    # 按句末切分（保留标点在前一片末尾）
    pieces: list[str] = []
    buf = ""
    i = 0
    n = len(p)
    while i < n:
        ch = p[i]
        buf += ch
        i += 1
        if ch in "。！？；\n" or i == n:
            seg = buf.strip()
            buf = ""
            if not seg:
                continue
            if len(seg) <= max_chars:
                pieces.append(seg)
            else:
                pieces.extend(_hard_split_oversized(seg, max_chars))
    # 合并过短碎片，减少 embedding 噪声（在不超过 max_chars 前提下拼接）
    merged: list[str] = []
    acc = ""
    min_atom = max(20, min(RAG_CHUNK_V2_MIN_ATOMIC_CHARS, max_chars // 4))
    for seg in pieces:
        if not acc:
            acc = seg
            continue
        if len(acc) + len(seg) <= max_chars and (len(acc) < min_atom or len(seg) < min_atom):
            acc = acc + seg
        else:
            merged.append(acc)
            acc = seg
    if acc:
        merged.append(acc)
    return merged


def _pack_with_overlap(units: list[str], max_chars: int, overlap: int) -> list[str]:
    """贪心拼接原子块为不超过 max_chars 的块；块间带尾部重叠利于检索连贯。"""
    if not units:
        return []
    overlap = max(0, min(overlap, max_chars // 4))
    chunks: list[str] = []
    i = 0
    carry = ""
    while i < len(units):
        parts: list[str] = []
        if carry:
            parts.append(carry)
        cur = sum(len(x) + 1 for x in parts) - 1 if parts else 0
        while i < len(units):
            u = units[i]
            sep = 1 if parts else 0
            if cur + sep + len(u) <= max_chars:
                parts.append(u)
                cur += sep + len(u)
                i += 1
            else:
                break
        if not parts:
            # 单个 unit 仍超长（理论上 atomic 已压过）
            chunks.append(units[i][:max_chars].strip())
            i += 1
            carry = chunks[-1][-overlap:] if overlap and chunks else ""
            continue
        text_c = "\n".join(parts).strip()
        if text_c:
            chunks.append(text_c)
            carry = text_c[-overlap:] if overlap and len(text_c) > overlap else ""
        else:
            carry = ""
    return [c for c in chunks if c]


def chunk_text_v2(text: str, *, source: str) -> list[TextChunk]:
    """财报 PDF 分块：段落边界 → 句末标点 → 超长软切/硬切 → 定长打包 + 重叠（索引重建唯一入口）。"""
    max_chars = max(400, RAG_CHUNK_V2_MAX_CHARS)
    overlap = max(0, min(RAG_CHUNK_V2_OVERLAP, max_chars // 4))

    t = _normalize_report_text(text)
    if not t:
        return []

    units: list[str] = []
    for para in _split_paragraphs(t):
        units.extend(_atomic_units_from_paragraph(para, max_chars))

    raw = _pack_with_overlap(units, max_chars=max_chars, overlap=overlap)
    out: list[TextChunk] = []
    for idx, piece in enumerate(raw):
        if piece.strip():
            out.append(TextChunk(text=piece.strip(), source=source, chunk_index=idx))
    return out
