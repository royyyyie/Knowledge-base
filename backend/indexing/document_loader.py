"""文档加载和分片服务"""

import hashlib
import re
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.settings import get_settings
from backend.indexing.pdf_parser import parse_pdf
from backend.indexing.structured_parser import (
    ParsedElement,
    convert_legacy_office,
    parse_docx,
    parse_xlsx,
)

# 编译非打印 C0/C1 控制字符的正则（保留常规排版字：\t, \n, \r）
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 编译零宽字符和不可见格式化控制字符（零宽空白、BOM 标记、左右强排标志等）
_INVISIBLE_CHAR_RE = re.compile(r"[\u200b-\u200d\ufeff\u200f\u202a-\u202e]")


def sanitize_text(text: str) -> str:
    """
    企业级标准文本净化器 (Text Sanitizer)。
    1. 规范化 (Normalization)：统一转换为标准 NFC 格式，合并分离变音符和音调，确保多端字符表示合一。
    2. 剔除/替换不合法及不可见字节：过滤 NUL (0x00) 空字节、零宽字符、BOM 标签和不可见强排标记。
    3. 清洗非打印字符及乱码：剔除 C0/C1 控制符号，剥离 Unicode PUA 私有使用区乱码符号。
    4. 编码收敛防爆：利用 utf-8 ignore 安全剥离任何不完整的、孤立的 UTF-16 代理项（Surrogates）。
    """
    if not text:
        return ""

    # 1. 规范化为 Unicode NFC 格式
    text = unicodedata.normalize("NFC", text)

    # 2. 清除不可见零宽字符、BOM 及格式控制符
    text = _INVISIBLE_CHAR_RE.sub("", text)

    # 3. 清洗非打印控制符及 PUA 乱码框区字符
    text = _CONTROL_CHAR_RE.sub("", text)
    text = re.sub(r"[\ue000-\uf8ff]", "", text)

    # 4. 彻底擦除孤立代理项 (Surrogates)，收敛至 100% 合规的 UTF-8 (对应 PostgreSQL 的 utf8mb4 标准)
    try:
        cleaned = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    except Exception:
        chars = []
        for char in text:
            if 0xD800 <= ord(char) <= 0xDFFF:
                continue
            chars.append(char)
        cleaned = "".join(chars)

    return cleaned


def _normalize_metadata_text(value: object) -> str:
    if value is None:
        return ""
    return sanitize_text(str(value)).strip()


def _normalize_acl_tags(values: Sequence[str] | str | None) -> list[str]:
    candidates: Sequence[str]
    if values is None:
        candidates = ()
    elif isinstance(values, str):
        candidates = (values,)
    else:
        candidates = values

    normalized: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        tag = _normalize_metadata_text(value)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


@dataclass(frozen=True, slots=True)
class DocumentArtifactMetadata:
    """一次版本化入库所需的稳定文档身份与访问范围。"""

    tenant_id: str
    knowledge_base_id: str
    document_id: str
    document_version_id: str
    section_id: str = ""
    acl_tags: Sequence[str] = ()
    index_version: str = "v1"

    def __post_init__(self) -> None:
        for field_name in (
            "tenant_id",
            "knowledge_base_id",
            "document_id",
            "document_version_id",
            "section_id",
            "index_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_metadata_text(getattr(self, field_name)),
            )
        for field_name in (
            "tenant_id",
            "knowledge_base_id",
            "document_id",
            "document_version_id",
            "index_version",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be a non-empty string")
        object.__setattr__(self, "acl_tags", tuple(_normalize_acl_tags(self.acl_tags)))


class DocumentLoader:
    """文档加载和分片服务"""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        *,
        max_pages: int | None = None,
        max_page_characters: int | None = None,
    ):
        storage_settings = get_settings().storage
        self.max_pages = max_pages or storage_settings.max_document_pages
        self.max_page_characters = (
            max_page_characters or storage_settings.max_page_characters
        )
        level_1_size = max(2000, chunk_size * 3)
        level_1_overlap = max(400, chunk_overlap * 3)
        level_2_size = max(1000, chunk_size * 2)
        level_2_overlap = max(200, chunk_overlap * 2)
        level_3_size = max(600, chunk_size)
        level_3_overlap = max(100, chunk_overlap)

        self._splitter_level_1 = RecursiveCharacterTextSplitter(
            chunk_size=level_1_size,
            chunk_overlap=level_1_overlap,
            add_start_index=True,
            separators=["\n\n", "。", "！", "？", "\n", "，", "、", " ", ""],
        )
        self._splitter_level_2 = RecursiveCharacterTextSplitter(
            chunk_size=level_2_size,
            chunk_overlap=level_2_overlap,
            add_start_index=True,
            separators=["\n\n", "。", "！", "？", "\n", "，", "、", " ", ""],
        )
        self._splitter_level_3 = RecursiveCharacterTextSplitter(
            chunk_size=level_3_size,
            chunk_overlap=level_3_overlap,
            add_start_index=True,
            separators=["\n\n", "。", "！", "？", "\n", "，", "、", " ", ""],
        )

    @staticmethod
    def _build_chunk_id(
        filename: str,
        page_number: int,
        level: int,
        index: int,
        document_version_id: str = "",
        source_unit_id: str = "",
    ) -> str:
        unit = f"::u{source_unit_id}" if source_unit_id else ""
        chunk_path = f"{filename}{unit}::p{page_number}::l{level}::{index}"
        if not document_version_id:
            raise ValueError("document_version_id must not be empty")
        return f"{document_version_id}::{chunk_path}"

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _artifact_metadata_for_page(
        raw_metadata: dict,
        metadata: DocumentArtifactMetadata,
        page_number: int,
    ) -> dict:
        tenant_id = metadata.tenant_id
        knowledge_base_id = metadata.knowledge_base_id
        document_id = metadata.document_id
        document_version_id = metadata.document_version_id
        index_version = metadata.index_version
        acl_tags = list(metadata.acl_tags)
        explicit_section_id = metadata.section_id

        section_id = (
            explicit_section_id
            or _normalize_metadata_text(raw_metadata.get("section_id"))
            or _normalize_metadata_text(raw_metadata.get("section_title"))
            or f"page:{page_number}"
        )
        return {
            "tenant_id": tenant_id,
            "knowledge_base_id": knowledge_base_id,
            "document_id": document_id,
            "document_version_id": document_version_id,
            "section_id": section_id,
            "acl_tags": acl_tags,
            "index_version": index_version,
        }

    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
        *,
        heading_path: tuple[str, ...] = (),
    ) -> List[Dict]:
        if not text:
            return []

        heading_prefix = (
            f"[标题路径: {' > '.join(heading_path)}]\n\n" if heading_path else ""
        )

        def with_heading_path(chunk_text: str) -> str:
            return f"{heading_prefix}{chunk_text}" if heading_prefix else chunk_text

        root_chunks: List[Dict] = []
        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]
        document_version_id = base_doc.get("document_version_id", "")
        source_unit_id = base_doc.get("source_unit_id", "")

        level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])
        level_1_counter = 0
        level_2_counter = 0
        level_3_counter = 0

        for level_1_doc in level_1_docs:
            level_1_text = (level_1_doc.page_content or "").strip()
            if not level_1_text:
                continue
            level_1_index_text = with_heading_path(level_1_text)
            level_1_id = self._build_chunk_id(
                filename,
                page_number,
                1,
                level_1_counter,
                document_version_id,
                source_unit_id,
            )
            level_1_counter += 1

            level_1_chunk = {
                **base_doc,
                "text": level_1_index_text,
                "chunk_id": level_1_id,
                "parent_chunk_id": "",
                "root_chunk_id": level_1_id,
                "chunk_level": 1,
                "chunk_idx": page_global_chunk_idx,
                "content_hash": self._content_hash(level_1_index_text),
            }
            page_global_chunk_idx += 1
            root_chunks.append(level_1_chunk)

            level_2_docs = self._splitter_level_2.create_documents(
                [level_1_text], [base_doc]
            )
            for level_2_doc in level_2_docs:
                level_2_text = (level_2_doc.page_content or "").strip()
                if not level_2_text:
                    continue
                level_2_index_text = with_heading_path(level_2_text)
                level_2_id = self._build_chunk_id(
                    filename,
                    page_number,
                    2,
                    level_2_counter,
                    document_version_id,
                    source_unit_id,
                )
                level_2_counter += 1

                level_2_chunk = {
                    **base_doc,
                    "text": level_2_index_text,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": page_global_chunk_idx,
                    "content_hash": self._content_hash(level_2_index_text),
                }
                page_global_chunk_idx += 1
                root_chunks.append(level_2_chunk)

                level_3_docs = self._splitter_level_3.create_documents(
                    [level_2_text], [base_doc]
                )
                for level_3_doc in level_3_docs:
                    level_3_text = (level_3_doc.page_content or "").strip()
                    if not level_3_text:
                        continue
                    level_3_index_text = with_heading_path(level_3_text)
                    level_3_id = self._build_chunk_id(
                        filename,
                        page_number,
                        3,
                        level_3_counter,
                        document_version_id,
                        source_unit_id,
                    )
                    level_3_counter += 1
                    root_chunks.append(
                        {
                            **base_doc,
                            "text": level_3_index_text,
                            "chunk_id": level_3_id,
                            "parent_chunk_id": level_2_id,
                            "root_chunk_id": level_1_id,
                            "chunk_level": 3,
                            "chunk_idx": page_global_chunk_idx,
                            "content_hash": self._content_hash(level_3_index_text),
                        }
                    )
                    page_global_chunk_idx += 1

        return root_chunks

    def _load_from_langchain_docs(
        self,
        raw_docs: list,
        file_path: str,
        filename: str,
        doc_type: str,
        metadata: DocumentArtifactMetadata,
        *,
        enforce_page_count: bool = True,
    ) -> list[dict]:
        if enforce_page_count and len(raw_docs) > self.max_pages:
            raise ValueError(f"文档页数超过限制（最多 {self.max_pages} 页）")
        documents: list[dict] = []
        page_global_chunk_idx = 0
        for doc in raw_docs:
            meta = getattr(doc, "metadata", None) or {}
            page_num = meta.get("page", 0)
            if page_num is None:
                page_num = 0
            try:
                page_num = int(page_num)
            except (TypeError, ValueError):
                page_num = 0
            page_content = sanitize_text((doc.page_content or "").strip())
            if len(page_content) > self.max_page_characters:
                raise ValueError(
                    f"单页字符数超过限制（最多 {self.max_page_characters} 字符）"
                )
            base_doc = {
                "filename": sanitize_text(filename),
                "file_path": sanitize_text(file_path),
                "file_type": sanitize_text(doc_type),
                "page_number": page_num,
                **self._artifact_metadata_for_page(meta, metadata, page_num),
            }
            source_unit_id = _normalize_metadata_text(meta.get("source_unit_id"))
            if source_unit_id:
                if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", source_unit_id):
                    raise ValueError("source_unit_id 格式无效")
                base_doc["source_unit_id"] = source_unit_id
            raw_heading_path = meta.get("heading_path", ())
            heading_path = (
                tuple(
                    normalized
                    for part in raw_heading_path
                    if isinstance(part, str)
                    if (normalized := sanitize_text(part).strip())
                )
                if isinstance(raw_heading_path, (list, tuple))
                else ()
            )
            page_chunks = self._split_page_to_three_levels(
                text=page_content,
                base_doc=base_doc,
                page_global_chunk_idx=page_global_chunk_idx,
                heading_path=heading_path,
            )
            page_global_chunk_idx += len(page_chunks)
            documents.extend(page_chunks)
        return documents

    def _load_structured_elements(
        self,
        elements: list[ParsedElement],
        file_path: str,
        filename: str,
        doc_type: str,
        metadata: DocumentArtifactMetadata,
    ) -> list[dict]:
        raw_docs = [
            Document(
                page_content=element.text,
                metadata={
                    "page": element.page_number,
                    "section_id": element.section_id,
                    "heading_path": element.heading_path,
                    "source_unit_id": element.unit_id,
                },
            )
            for element in elements
        ]
        return self._load_from_langchain_docs(
            raw_docs,
            file_path,
            filename,
            doc_type,
            metadata,
            enforce_page_count=False,
        )

    def load_document(
        self,
        file_path: str,
        filename: str,
        *,
        metadata: DocumentArtifactMetadata,
    ) -> list[dict]:
        file_lower = filename.lower()

        if file_lower.endswith((".html", ".htm")):
            doc_type = "HTML"
            from backend.indexing.html_processor import load_html_for_document_loader

            raw_docs = load_html_for_document_loader(file_path, filename)
            return self._load_from_langchain_docs(
                raw_docs,
                file_path,
                filename,
                doc_type,
                metadata,
            )
        storage = get_settings().storage
        deadline = time.monotonic() + storage.parser_timeout_seconds

        def remaining_time() -> float:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("文档解析超时")
            return remaining

        max_elements = min(self.max_pages * 5, 10_000)
        if file_lower.endswith(".pdf"):
            elements = parse_pdf(
                file_path,
                max_pages=self.max_pages,
                max_page_characters=self.max_page_characters,
                max_elements=max_elements,
                timeout_seconds=remaining_time(),
            )
            doc_type = "PDF"
        elif file_lower.endswith(".docx"):
            elements = parse_docx(
                file_path,
                max_elements=max_elements,
                max_element_characters=self.max_page_characters,
                timeout_seconds=remaining_time(),
            )
            doc_type = "Word"
        elif file_lower.endswith(".xlsx"):
            elements = parse_xlsx(
                file_path,
                max_rows=storage.max_workbook_rows,
                max_cells=storage.max_workbook_cells,
                max_elements=max_elements,
                max_element_characters=self.max_page_characters,
                timeout_seconds=remaining_time(),
            )
            doc_type = "Excel"
        elif file_lower.endswith((".doc", ".xls")):
            is_word = file_lower.endswith(".doc")
            temporary, converted = convert_legacy_office(
                file_path,
                target_extension="docx" if is_word else "xlsx",
                timeout_seconds=remaining_time(),
            )
            try:
                if is_word:
                    elements = parse_docx(
                        converted,
                        max_elements=max_elements,
                        max_element_characters=self.max_page_characters,
                        timeout_seconds=remaining_time(),
                    )
                else:
                    elements = parse_xlsx(
                        converted,
                        max_rows=storage.max_workbook_rows,
                        max_cells=storage.max_workbook_cells,
                        max_elements=max_elements,
                        max_element_characters=self.max_page_characters,
                        timeout_seconds=remaining_time(),
                    )
            finally:
                temporary.cleanup()
            doc_type = "Word" if is_word else "Excel"
        else:
            raise ValueError(f"不支持的文件类型: {filename}")
        if not elements:
            raise ValueError("文档没有可检索的结构化正文")
        return self._load_structured_elements(
            elements, file_path, filename, doc_type, metadata
        )
