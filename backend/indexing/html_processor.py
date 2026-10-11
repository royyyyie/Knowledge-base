"""
HTML 知识库处理：编码检测、去脚本/样式、结构线性化，并按章节拆成多「页」供三层分块复用。

设计要点：
- 优先解析 <main>/<article>，否则 <body>，避免全站导航噪声。
- 将 h1–h4 与段落、列表项按文档顺序线性化，表格按行保留表头和值的关系。
- 只为有正文的章节生成 page_number；标题路径随章节传给分块器。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import Comment
from langchain_core.documents import Document


_HEADING_RE = re.compile(r"^(#{1,4}) (.+)$")


def _read_html_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8")
    head = raw[:8192].decode("utf-8", errors="ignore")
    m = re.search(r'charset\s*=\s*["\']?([^"\'>\s]+)', head, re.I)
    if m:
        enc = m.group(1).strip().lower()
        if enc == "utf8":
            enc = "utf-8"
        try:
            return raw.decode(enc)
        except (LookupError, UnicodeDecodeError):
            pass
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gb18030", errors="replace")


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "template", "iframe", "svg"]):
        tag.decompose()
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()


def _pick_root(soup: BeautifulSoup) -> Any:
    return soup.find("main") or soup.find("article") or soup.body or soup


def _br_to_newlines(root: Any) -> None:
    for br in root.find_all("br"):
        br.replace_with("\n")


def _table_facts(table: Any) -> list[str]:
    """Render each data row with its column names, without repeating cell text."""

    rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    if not rows:
        plain = table.get_text(" ", strip=True)
        return [plain] if plain else []

    caption_tag = table.find("caption", recursive=False)
    caption = caption_tag.get_text(" ", strip=True) if caption_tag else ""
    first_cells = rows[0].find_all(["th", "td"], recursive=False)
    has_header = bool(first_cells) and (
        all(cell.name == "th" for cell in first_cells)
        or rows[0].find_parent("thead") is not None
    )
    headers = (
        [cell.get_text(" ", strip=True) for cell in first_cells] if has_header else []
    )
    header_merged = any(
        cell.get("rowspan", "1") != "1" or cell.get("colspan", "1") != "1"
        for cell in first_cells
    )
    facts: list[str] = []
    for row in rows[1:] if has_header else rows:
        cells = row.find_all(["th", "td"], recursive=False)
        values = [cell.get_text(" ", strip=True) for cell in cells]
        if not any(values):
            continue
        merged = any(
            cell.get("rowspan", "1") != "1" or cell.get("colspan", "1") != "1"
            for cell in cells
        )
        aligned = (
            bool(headers)
            and len(headers) == len(values)
            and not header_merged
            and not merged
        )
        parts = [
            f"{(headers[index] if aligned and headers[index] else f'列{index + 1}')}={value}"
            for index, value in enumerate(values)
            if value
        ]
        fact = "；".join(parts)
        facts.append(f"{caption}：{fact}" if caption else fact)
    if facts:
        return facts
    plain = table.get_text(" ", strip=True)
    return [plain] if plain else []


def _linearize_blocks(root: Any) -> str:
    _br_to_newlines(root)

    block_tags = [
        "h1",
        "h2",
        "h3",
        "h4",
        "p",
        "li",
        "blockquote",
        "pre",
        "table",
        "dd",
        "dt",
        "figcaption",
    ]
    parts: list[str] = []
    for el in root.find_all(block_tags, limit=8000):
        if el.find_parent(["script", "style", "noscript"]):
            continue
        if el.find_parent("table"):
            continue
        if el.name == "table":
            parts.extend(_table_facts(el))
            continue
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        name = el.name
        if name == "h1":
            parts.append(f"\n# {text}\n")
        elif name == "h2":
            parts.append(f"\n## {text}\n")
        elif name == "h3":
            parts.append(f"\n### {text}\n")
        elif name == "h4":
            parts.append(f"\n#### {text}\n")
        else:
            parts.append(text)

    structured = "\n".join(parts).strip()
    plain = root.get_text("\n", strip=True)

    if not plain:
        return structured
    if not structured:
        return plain

    if not root.find(["h1", "h2", "h3", "h4", "table"]) and len(structured) < max(
        120, int(0.22 * len(plain))
    ):
        return plain

    return structured


def _doc_title(soup: BeautifulSoup) -> str:
    t = soup.find("title")
    if t and t.string:
        return t.string.strip()
    return ""


def _split_into_sections(linear: str, page_title: str) -> list[dict[str, Any]]:
    text = linear.strip()
    if not text:
        return []

    headings: dict[int, str] = {}
    current_lines: list[str] = []
    sections: list[dict[str, Any]] = []

    def heading_path() -> tuple[str, ...]:
        path = [headings[level] for level in sorted(headings)]
        if page_title and (not path or path[0] != page_title):
            path.insert(0, page_title)
        return tuple(path)

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if not body:
            return
        path = heading_path()
        sections.append(
            {
                "page": len(sections) + 1,
                "title": path[-1] if path else "",
                "heading_path": path,
                "text": body,
            }
        )

    for line in text.splitlines():
        match = _HEADING_RE.fullmatch(line.strip())
        if match:
            flush()
            current_lines.clear()
            level = len(match.group(1))
            headings = {
                existing_level: title
                for existing_level, title in headings.items()
                if existing_level < level
            }
            headings[level] = match.group(2).strip()
        else:
            current_lines.append(line)
    flush()
    return sections


def parse_html_file_to_sections(file_path: str | Path) -> list[dict[str, Any]]:
    path = Path(file_path)
    html = _read_html_text(path)
    soup = BeautifulSoup(html, "html.parser")
    _strip_noise(soup)
    root = _pick_root(soup)
    page_title = _doc_title(soup)
    linear = _linearize_blocks(root)
    return _split_into_sections(linear, page_title)


def load_html_for_document_loader(file_path: str, filename: str) -> list[Document]:
    """供 DocumentLoader 使用：返回 LangChain Document 列表，metadata.page 为章节序号。"""
    sections = parse_html_file_to_sections(file_path)
    docs: list[Document] = []
    for sec in sections:
        docs.append(
            Document(
                page_content=sec["text"],
                metadata={
                    "page": sec["page"],
                    "source": filename,
                    "section_title": sec.get("title") or "",
                    "heading_path": sec["heading_path"],
                },
            )
        )
    return docs
