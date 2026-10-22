"""
从 PDF 抽取纯文本（供 RAG 分块与向量化）。

使用 pypdf 按页 extract_text：无表格结构化解析，复杂排版/扫描件效果因文件而异。
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from rag.config import RAG_PDF_PAGE_MAX_CHARS


def extract_pdf_text(path: Path) -> str:
    """读取单个 PDF 的全部页面文本并简单清洗。

    说明：部分扫描版 PDF 可能几乎无文本层，此时返回内容会偏少，属正常现象。
    """
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        t = t.strip()
        if len(t) > RAG_PDF_PAGE_MAX_CHARS:
            t = t[:RAG_PDF_PAGE_MAX_CHARS]
        if t:
            parts.append(t)
    body = "\n\n".join(parts)
    # 合并多余空白，减轻分块噪声
    lines = [ln.strip() for ln in body.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def iter_caibao_pdfs(root: Path) -> list[Path]:
    """列出目录下所有 .pdf（不递归子目录）。"""
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
