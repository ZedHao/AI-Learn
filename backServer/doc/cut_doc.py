#!/usr/bin/env python3
"""
财报 PDF 截断转纯文本（UTF-8）。

输出（均在 cut_doc_dir/<公司名>/）：
- {文件名}_目录.txt：仅目录页合并文本（逻辑与原先一致）。
- {文件名}_报告提取.txt：目录中「公司简介/主要财务指标」类条目所印页 x 至「环境/社会/ESG」类条目所印页 y 之间的正文（多别名子串 + 规范化 + difflib 相似度兜底；含 x、y 两页）。
- {文件名}_财务报告.txt：目录中「财务报告」类条目所印页 z 起（须出现在「公司治理」类条目之后），至首次出现
  「4. 四、财务报表的编制基础 … 1、编制基础」类锚点所在页 m 止（含 m；m 之后页丢弃）；
  再在拼接结果上按锚点做与原先一致的尾部截取（编制基础段结束处截断）。

源 PDF：优先 <仓库根>/doc/caibao，否则 backServer/doc/caibao；仅处理文件名含「比亚迪」或「立讯精密」。
"""

from __future__ import annotations

import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from pypdf import PdfReader

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent


def _default_pdf_input_dir() -> Path:
    """与 RAG 对齐：优先 <仓库根>/doc/caibao，否则退回 backServer/doc/caibao。"""
    root_caibao = _REPO_ROOT / "doc" / "caibao"
    local_caibao = _SCRIPT_DIR / "caibao"
    if root_caibao.is_dir():
        return root_caibao
    return local_caibao


PDF_DIR = Path(
    sys.argv[1] if len(sys.argv) > 1 else str(_default_pdf_input_dir())
).expanduser().resolve()
OUT_ROOT = _SCRIPT_DIR / "cut_doc_dir"

ALLOW_NAME_KEYWORDS = ("比亚迪", "立讯精密")

# ---------------------------------------------------------------------------
# 目录 / 章节 / 结束标记
# ---------------------------------------------------------------------------
_TOC_LINE_LOOSE = re.compile(
    r"第[一二三四五六七八九十百千两〇零0-9]+节\s+.+\.{4,}\s*\d+",
)


def _safe_stem(name: str) -> str:
    s = Path(name).stem
    for c in '<>:"/\\|?*':
        s = s.replace(c, "_")
    return s.strip() or "unnamed"


def _company_subdir(pdf_name: str) -> str:
    if "比亚迪" in pdf_name:
        return "比亚迪"
    if "立讯精密" in pdf_name:
        return "立讯精密"
    return "其他"


def _page_text(reader: PdfReader, idx: int) -> str:
    try:
        return (reader.pages[idx].extract_text() or "").replace("\r\n", "\n").replace("\r", "\n")
    except Exception:
        return ""


def _count_toc_like_lines(text: str) -> int:
    n = 0
    for ln in text.splitlines():
        if _TOC_LINE_LOOSE.search(ln):
            n += 1
    return n


def _is_toc_page(text: str) -> bool:
    t = text.strip()
    if "目录" not in t:
        return False
    if "第一节" not in t and "第" not in t:
        return False
    return _count_toc_like_lines(text) >= 2 or ("目录" in t and _TOC_LINE_LOOSE.search(t))


def extract_toc_pages(reader: PdfReader) -> tuple[list[int], str]:
    """识别目录所在页（可连续多页），返回页索引列表与合并文本。"""
    n = len(reader.pages)
    toc_indices: list[int] = []
    for i in range(min(60, n)):
        t = _page_text(reader, i)
        if not _is_toc_page(t):
            continue
        if not toc_indices:
            toc_indices.append(i)
        elif i == toc_indices[-1] + 1:
            toc_indices.append(i)
    if not toc_indices:
        for i in range(min(40, n)):
            t = _page_text(reader, i)
            if "目录" in t and _count_toc_like_lines(t) >= 2:
                toc_indices = [i]
                break
    if not toc_indices:
        return [], ""
    last = toc_indices[-1]
    j = last + 1
    while j < n and j < last + 4:
        t = _page_text(reader, j)
        if _count_toc_like_lines(t) >= 2:
            toc_indices.append(j)
            j += 1
            last = j - 1
        else:
            break
    merged = "\n\n".join(_page_text(reader, i) for i in toc_indices)
    return toc_indices, merged


# 「四、财务报表的编制基础」+「1、编制基础」类结束锚点
_FIN_END_RE = re.compile(
    r"(?:4[、,．\.]\s*)?"
    r"四[、,．\.]\s*财务报表的编制基础\s*[\s\S]{0,160}?"
    r"(?:1|１|①)\s*[、,．\.]\s*编制基础",
    re.MULTILINE,
)


def _slice_financial_until_marker(full: str) -> tuple[str, bool]:
    """从财务报告拼接正文中截取到「编制基础」小节结束（锚点 + 其后一段说明）。"""
    m = _FIN_END_RE.search(full)
    if not m:
        return full, False
    tail_start = m.end()
    rest = full[tail_start:]
    end_rel = len(rest)
    for sep in ("\n\n\n", "\n\n二、", "\n\n三、", "\n二、", "\n三、", "\n（二）", "\n（三）"):
        p = rest.find(sep)
        if p != -1:
            end_rel = min(end_rel, p + (2 if sep.startswith("\n\n") else 1))
    min_tail = min(200, len(rest))
    cut = tail_start + max(min_tail, min(end_rel, 2800))
    cut = min(cut, len(full))
    return full[:cut].rstrip() + "\n", True


def _write_utf8_txt(path_txt: Path, body: str) -> None:
    path_txt.parent.mkdir(parents=True, exist_ok=True)
    out = body.rstrip("\n") + "\n"
    path_txt.write_text(out, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# 目录行解析：x / y / z（不依赖「第几节」序号；多别名 + 规范化 + 模糊相似兜底）
# ---------------------------------------------------------------------------
_TOC_TAIL_PAGE = re.compile(r"\.{3,}\s*(\d+)\s*$")
_TOC_TITLE_CORE = re.compile(
    r"^第[一二三四五六七八九十百千两〇零0-9\s]+节\s*(.+?)\s*\.{2,}",
)


def _normalize_toc_line(s: str) -> str:
    """全角半角统一、空白折叠，便于子串与模糊匹配。"""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u3000", " ")
    return re.sub(r"\s+", " ", s.strip())


def _toc_title_segment(line: str) -> str:
    """去掉「第…节」前缀与尾部点阵页码，得到章节标题核心（用于相似度）。"""
    n = _normalize_toc_line(line)
    m = _TOC_TITLE_CORE.match(n)
    if m:
        return m.group(1).strip()
    n2 = re.sub(r"^第[一二三四五六七八九十百千两〇零0-9\s]+节\s*", "", n).strip()
    return re.sub(r"\.{2,}\s*\d+\s*$", "", n2).strip()


def _best_fuzzy_ratio(title: str, references: tuple[str, ...]) -> float:
    if len(title) < 2:
        return 0.0
    t = _normalize_toc_line(title)
    best = 0.0
    for ref in references:
        r = SequenceMatcher(None, t, _normalize_toc_line(ref)).ratio()
        if r > best:
            best = r
    return best


# --- x：公司简介 / 主要财务指标（子串优先，相似度兜底）---
_PROFILE_KEYWORDS: tuple[str, ...] = (
    "公司简介和主要财务指标",
    "公司简介及主要财务指标",
    "公司简介与主要财务指标",
    "公司简介、主要财务指标",
    "公司简介、主要业务",
    "公司简介及主要业务",
    "公司简介",
    "公司概况",
    "公司基本情况",
    "公司基本情况及主要财务指标",
    "公司基本情况与主要财务指标",
    "主要财务指标",
    "主要会计数据和财务指标",
    "主要会计数据及财务指标",
    "主要会计数据和指标",
    "会计数据和财务指标摘要",
    "会计数据及财务指标摘要",
    "主要会计数据",
    "主要会计数据与财务指标",
    "近三年主要会计数据",
    "近三年主要会计数据和指标",
    "财务摘要",
    "核心财务指标",
    "主要业务概要",
    "基本情况简介",
)
_PROFILE_FUZZ_REF: tuple[str, ...] = (
    "公司简介和主要财务指标",
    "公司基本情况及主要财务指标",
    "主要会计数据和财务指标",
)


def _line_matches_profile(line: str) -> bool:
    n = _normalize_toc_line(line)
    for kw in _PROFILE_KEYWORDS:
        if kw in n:
            return True
    title = _toc_title_segment(line)
    if len(title) < 5:
        return False
    if _best_fuzzy_ratio(title, _PROFILE_FUZZ_REF) >= 0.58:
        return any(k in title for k in ("公司", "财务", "会计", "业务", "概况", "简介", "指标", "数据"))
    return False


# --- y：环境 / 社会 / ESG / 可持续（子串优先；避免单独「社会责任」误触）---
_ENV_KEYWORDS: tuple[str, ...] = (
    "环境和社会责任",
    "环境与社会责任",
    "环境、社会及公司治理",
    "环境、社会与公司治理",
    "环境、社会及治理",
    "环境、社会和治理",
    "环境社会及治理",
    "环境社会与公司治理",
    "环境及社会责任",
    "环境保护与社会责任",
    "环境保护和社会责任",
    "环境与社会",
    "环境和社会",
    "环境、社会",
    "绿色低碳发展",
    "绿色低碳",
    "低碳发展",
    "碳达峰碳中和",
    "可持续发展",
    "ESG报告",
)
_ENV_CONTEXT = ("环境", "社会", "ESG", "可持续", "绿色", "低碳", "碳", "气候", "生态", "公益", "管治")


def _line_matches_env_social(line: str) -> bool:
    n = _normalize_toc_line(line)
    for kw in _ENV_KEYWORDS:
        if kw in n:
            return True
    nu = n.upper()
    if "ESG" in nu and any(c in n for c in ("环境", "社会", "治理", "可持续", "责任")):
        return True
    title = _toc_title_segment(line)
    if "社会责任" in n and any(c in n for c in _ENV_CONTEXT):
        return True
    if len(title) >= 3:
        if _best_fuzzy_ratio(title, ("环境和社会责任", "环境社会及治理", "环境和社会", "环境、社会及治理")) >= 0.52:
            if any(c in title for c in ("环境", "社会", "ESG", "可持续", "绿色", "低碳", "碳")):
                return True
    return False


# --- 公司治理（用于 z 的先后次序）---
_GOV_KEYWORDS: tuple[str, ...] = (
    "公司治理",
    "公司治理结构",
    "公司治理制度",
    "企业管治",
    "企业管治报告",
    "管治报告",
    "公司治理及内部控制",
    "公司治理与内部控制",
)


def _line_has_governance(line: str) -> bool:
    n = _normalize_toc_line(line)
    return any(k in n for k in _GOV_KEYWORDS)


# --- z：财务报告章节（须出现在公司治理之后；子串 + 模糊）---
_FIN_KEYWORDS: tuple[str, ...] = (
    "财务报告",
    "财务会计报告",
    "财务报表及附注",
    "财务报表和附注",
    "财务报表及审计报告",
    "审计报告及财务报表",
    "合并财务报表",
    "合并财务会计报表",
    "会计报表及附注",
    "会计报表和附注",
    "财务报表附注",
)
_FIN_FUZZ_REF: tuple[str, ...] = (
    "财务报告",
    "财务会计报告",
    "财务报表及附注",
)


def _line_matches_financial_report(line: str) -> bool:
    n = _normalize_toc_line(line)
    for kw in _FIN_KEYWORDS:
        if kw in n:
            return True
    title = _toc_title_segment(line)
    if len(title) < 4:
        return False
    if _best_fuzzy_ratio(title, _FIN_FUZZ_REF) >= 0.55:
        return any(k in title for k in ("财务", "会计", "报表", "审计", "附注"))
    return False


def _line_toc_tail_page(line: str) -> int | None:
    s = line.strip()
    m = _TOC_TAIL_PAGE.search(s)
    return int(m.group(1)) if m else None


def _iter_toc_rows_in_order(toc_text: str) -> list[tuple[str, int]]:
    """按目录文本自上而下，收集带点阵行尾页码的「第…节」类目录行。"""
    rows: list[tuple[str, int]] = []
    for raw in toc_text.splitlines():
        line = raw.strip()
        if "节" not in line:
            continue
        if not (_TOC_LINE_LOOSE.search(line) or _TOC_TAIL_PAGE.search(line)):
            continue
        page = _line_toc_tail_page(line)
        if page is None:
            continue
        rows.append((line, page))
    return rows


def parse_toc_xyz(toc_text: str) -> tuple[int | None, int | None, int | None, bool]:
    """
    从合并后的目录文本解析目录所印页码（章节序号可变，顺序不变）：
    - x：首条命中「公司简介 / 主要财务指标」等别名或模糊相似目录行尾页码。
    - y：首条命中「环境 / 社会 / ESG / 可持续」等别名或模糊相似目录行尾页码。
    - z：在首条「公司治理」类目录行**之后**首条「财务报告」类目录行尾页码。
      若无公司治理行，则 z 退化为全文首条财务报告类行。

    匹配策略：NFKC 规范化 + 多关键词子串 + 章节标题对典型全称的 difflib 相似度兜底（无额外模型依赖）。

    返回值: (x, y, z, governance_row_seen)
    """
    rows = _iter_toc_rows_in_order(toc_text)
    x: int | None = None
    y: int | None = None
    z: int | None = None
    gov_idx: int | None = None

    for i, (line, page) in enumerate(rows):
        if x is None and _line_matches_profile(line):
            x = page
        if y is None and _line_matches_env_social(line):
            y = page
        if gov_idx is None and _line_has_governance(line):
            gov_idx = i

    governance_row_seen = gov_idx is not None

    if gov_idx is not None:
        for j in range(gov_idx + 1, len(rows)):
            line, page = rows[j]
            if _line_matches_financial_report(line):
                z = page
                break
    else:
        for line, page in rows:
            if _line_matches_financial_report(line):
                z = page
                break

    return x, y, z, governance_row_seen


# ---------------------------------------------------------------------------
# 目录所印页码 ↔ PDF 页索引（0-based）
# ---------------------------------------------------------------------------
_DASH_PAGE = re.compile(r"[—\-]\s*(\d{1,4})\s*[—\-]")


def _guess_printed_page_one_page(page_text: str, n_pages: int) -> int | None:
    """从单页正文猜「页脚所印页码」。"""
    lines = page_text.replace("\r", "\n").split("\n")
    tail = "\n".join(lines[-28:])
    dm = list(_DASH_PAGE.finditer(tail))
    if dm:
        v = int(dm[-1].group(1))
        if 1 <= v <= min(2000, n_pages + 80):
            return v
    for ln in reversed(lines[-28:]):
        s = ln.strip()
        if re.fullmatch(r"\d{1,4}", s):
            v = int(s)
            if 1 <= v <= min(2000, n_pages + 80):
                return v
    return None


def build_printed_to_pdf(reader: PdfReader) -> dict[int, int]:
    """
    所印页码 -> 首次出现的 PDF 页索引。
    若页脚识别率过低则返回空 dict，调用方用「所印页码 = PDF 1-based 页序」回退。
    """
    n = len(reader.pages)
    buckets: dict[int, list[int]] = {}
    for j in range(n):
        g = _guess_printed_page_one_page(_page_text(reader, j), n)
        if g is not None:
            buckets.setdefault(g, []).append(j)
    if len(buckets) < max(5, int(n * 0.12)):
        return {}
    return {printed: min(indices) for printed, indices in buckets.items()}


def _pdf_index_for_printed(
    printed: int,
    *,
    pmap: dict[int, int],
    n_pages: int,
) -> int:
    if printed in pmap:
        return pmap[printed]
    return max(0, min(n_pages - 1, printed - 1))


def _printed_for_pdf_index(
    pdf_idx: int,
    *,
    pmap: dict[int, int],
    n_pages: int,
) -> int:
    """给定 PDF 页索引，反推所印页码（用于确定 m）。"""
    if pmap:
        for printed, idx in sorted(pmap.items(), key=lambda kv: kv[1]):
            if idx == pdf_idx:
                return printed
    return pdf_idx + 1


def join_printed_page_range(
    reader: PdfReader,
    p_lo: int,
    p_hi: int,
    pmap: dict[int, int],
) -> str:
    """按所印页码闭区间 [p_lo, p_hi] 拼接正文（页与页之间空行分隔）。"""
    if p_lo > p_hi:
        p_lo, p_hi = p_hi, p_lo
    n = len(reader.pages)
    parts: list[str] = []
    for p in range(p_lo, p_hi + 1):
        idx = _pdf_index_for_printed(p, pmap=pmap, n_pages=n)
        parts.append(_page_text(reader, idx))
    return "\n\n".join(parts)


def find_end_printed_m(
    reader: PdfReader,
    z_printed: int,
    pmap: dict[int, int],
) -> tuple[int | None, int | None, bool]:
    """
    从财务报告节起始页 z 起向后扫，直到拼接文本中出现财务结束锚。
    返回 (m_printed, m_pdf_idx, hit_anchor)。
    m_printed 为锚点首次完整出现时当前 PDF 页所对应的所印页码；m 之后页丢弃。
    """
    n = len(reader.pages)
    iz = _pdf_index_for_printed(z_printed, pmap=pmap, n_pages=n)
    if iz >= n:
        return None, None, False
    acc = ""
    last_j = iz
    for j in range(iz, n):
        last_j = j
        chunk = _page_text(reader, j)
        acc = f"{acc}\n\n{chunk}" if acc else chunk
        if _FIN_END_RE.search(acc):
            m_printed = _printed_for_pdf_index(j, pmap=pmap, n_pages=n)
            return m_printed, j, True
    m_printed = _printed_for_pdf_index(last_j, pmap=pmap, n_pages=n)
    return m_printed, last_j, False


def process_one_pdf(pdf_path: Path) -> None:
    name = pdf_path.name
    if not any(k in name for k in ALLOW_NAME_KEYWORDS):
        return
    company = _company_subdir(name)
    stem = _safe_stem(name)
    out_dir = OUT_ROOT / company
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(pdf_path))
    n = len(reader.pages)
    print(f"[cut_doc] 处理: {name} | 页数={n}", flush=True)

    toc_indices, toc_text = extract_toc_pages(reader)
    if not toc_text.strip():
        print(f"[cut_doc] 警告: 未识别到目录页，跳过: {name}", flush=True)
        return

    toc_out = out_dir / f"{stem}_目录.txt"
    _write_utf8_txt(toc_out, toc_text)
    print(f"[cut_doc] 已写: {toc_out}", flush=True)

    x, y, z, governance_row_seen = parse_toc_xyz(toc_text)
    pmap = build_printed_to_pdf(reader)
    if pmap:
        print(f"[cut_doc] 页脚映射: 已识别 {len(pmap)} 个所印页码 <-> PDF 页", flush=True)
    else:
        print("[cut_doc] 页脚映射: 不足，回退为「所印页码 = PDF 从首页起的 1-based 页序」", flush=True)

    if x is not None and y is not None:
        body_xy = join_printed_page_range(reader, x, y, pmap)
        extract_out = out_dir / f"{stem}_报告提取.txt"
        _write_utf8_txt(extract_out, body_xy)
        print(f"[cut_doc] 已写: {extract_out} | 所印页码区间 [{x}..{y}]", flush=True)
    else:
        print(
            f"[cut_doc] 警告: 目录未解析到公司简介/主要财务指标或环境社会责任页码 (x={x}, y={y})，跳过 _报告提取.txt: {name}",
            flush=True,
        )

    if z is None:
        if governance_row_seen:
            print(
                f"[cut_doc] 警告: 目录中「公司治理」之后未找到「财务报告」行，跳过 _财务报告.txt: {name}",
                flush=True,
            )
        else:
            print(
                f"[cut_doc] 警告: 目录未找到「财务报告」行（且无「公司治理」可供定位），跳过 _财务报告.txt: {name}",
                flush=True,
            )
        return

    if z is not None and not governance_row_seen:
        print(
            "[cut_doc] 提示: 目录中未识别「公司治理」行，z 取为全文首条「财务报告」目录行，请核对是否与正文一致。",
            flush=True,
        )

    m_printed, m_pdf_idx, anchor_hit = find_end_printed_m(reader, z, pmap)
    if m_printed is None or m_pdf_idx is None:
        print(f"[cut_doc] 警告: 财务报告起始页 z={z} 超出 PDF 页数，跳过 _财务报告.txt: {name}", flush=True)
        return

    if m_printed < z:
        print(f"[cut_doc] 警告: 所印 m={m_printed} < z={z}，已令 m=z（单页）: {name}", flush=True)
        m_printed = z

    fin_raw = join_printed_page_range(reader, z, m_printed, pmap)
    fin_body, slice_hit = _slice_financial_until_marker(fin_raw)
    if not anchor_hit:
        print(
            f"[cut_doc] 警告: 未在 z..文末 命中结束锚，m 取为最后一页所印≈{m_printed}（PDF idx={m_pdf_idx}）: {name}",
            flush=True,
        )
    elif not slice_hit:
        print(f"[cut_doc] 警告: z..m 拼接中未命中截取锚，财务报告写入该区间的全文: {name}", flush=True)

    fin_out = out_dir / f"{stem}_财务报告.txt"
    _write_utf8_txt(fin_out, fin_body)
    print(
        f"[cut_doc] 已写: {fin_out} | 所印页码区间 [{z}..{m_printed}] | 锚点命中={anchor_hit} 截取={slice_hit}",
        flush=True,
    )


def main() -> None:
    if not PDF_DIR.is_dir():
        print(f"[cut_doc] 错误: PDF 目录不存在: {PDF_DIR}", flush=True)
        sys.exit(1)
    pdfs = sorted(p for p in PDF_DIR.iterdir() if p.suffix.lower() == ".pdf")
    targets = [p for p in pdfs if any(k in p.name for k in ALLOW_NAME_KEYWORDS)]
    if not targets:
        print(f"[cut_doc] 未找到文件名含 {ALLOW_NAME_KEYWORDS} 的 PDF: {PDF_DIR}", flush=True)
        return
    print(f"[cut_doc] PDF 目录: {PDF_DIR} | 输出: {OUT_ROOT}", flush=True)
    for p in targets:
        process_one_pdf(p)
    print("[cut_doc] 全部完成。", flush=True)


if __name__ == "__main__":
    1. # 临时导出或添加到系统环境变量
2. # 临时导出只能会话级使用，添加到系统环境变量可以长期使用
setx ANTHROPIC_BASE_URL=http://ada-cli-golang.ctripcorp.com/coding-plan
setx ANTHROPIC_AUTH_TOKEN=ada_250e8560bf6a41559218c0c9f35cd252
setx API_TIMEOUT_MS=3000000
setx CLAUDE_CODE_AUTO_COMPACT_WINDOW=200000
setx CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

{
    "env": {
        "ANTHROPIC_AUTH_TOKEN": "ada_250e8560bf6a41559218c0c9f35cd252",
        "ANTHROPIC_BASE_URL": "http://ada-cli-golang.ctripcorp.com/coding-plan",
        "API_TIMEOUT_MS": "3000000",

        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "200000",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",

    }
}

    main()
dccdcdccdcdccdcdccdc