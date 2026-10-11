from __future__ import annotations

import hashlib

from backend.indexing.document_loader import (
    DocumentArtifactMetadata,
    DocumentLoader,
)
from backend.indexing.html_processor import parse_html_file_to_sections


def _write_html(tmp_path, body: str, *, name: str = "guide.html"):
    path = tmp_path / name
    path.write_text(f"<html><main>{body}</main></html>", encoding="utf-8")
    return path


def _loader() -> DocumentLoader:
    return DocumentLoader(
        chunk_size=600,
        chunk_overlap=100,
        max_pages=20,
        max_page_characters=20_000,
    )


def _metadata(version_id: str, *, section_id: str = "") -> DocumentArtifactMetadata:
    return DocumentArtifactMetadata(
        tenant_id=" tenant-a ",
        knowledge_base_id="kb-a",
        document_id="doc-a",
        document_version_id=version_id,
        section_id=section_id,
        acl_tags=(" reader ", "reader", "\u200b研发", "", "研发"),
        index_version="index-v3",
    )


def test_versioned_chunk_ids_are_stable_and_do_not_collide(tmp_path):
    path = _write_html(tmp_path, "<h1>Guide</h1><p>Stable content.</p>")
    loader = _loader()

    first = loader.load_document(str(path), path.name, metadata=_metadata("version-a"))
    repeated = loader.load_document(
        str(path), path.name, metadata=_metadata("version-a")
    )
    replacement = loader.load_document(
        str(path), path.name, metadata=_metadata("version-b")
    )

    first_ids = [document["chunk_id"] for document in first]
    repeated_ids = [document["chunk_id"] for document in repeated]
    replacement_ids = [document["chunk_id"] for document in replacement]
    assert first_ids == repeated_ids
    assert not set(first_ids).intersection(replacement_ids)
    assert first_ids == [
        "version-a::guide.html::p1::l1::0",
        "version-a::guide.html::p1::l2::0",
        "version-a::guide.html::p1::l3::0",
    ]
    assert first[-1]["text"].startswith("[标题路径: Guide]\n\nStable content.")


def test_every_chunk_carries_normalized_artifact_metadata_and_content_hash(tmp_path):
    path = _write_html(tmp_path, "<h1>Guide</h1><p>Stable content.</p>")

    documents = _loader().load_document(
        str(path),
        path.name,
        metadata=_metadata("version-a", section_id=" section-explicit "),
    )

    assert {document["chunk_level"] for document in documents} == {1, 2, 3}
    for document in documents:
        assert document["tenant_id"] == "tenant-a"
        assert document["knowledge_base_id"] == "kb-a"
        assert document["document_id"] == "doc-a"
        assert document["document_version_id"] == "version-a"
        assert document["section_id"] == "section-explicit"
        assert document["acl_tags"] == ["reader", "研发"]
        assert document["index_version"] == "index-v3"
        assert (
            document["content_hash"]
            == hashlib.sha256(document["text"].encode("utf-8")).hexdigest()
        )


def test_section_id_prefers_source_section_then_stable_page_fallback(tmp_path):
    intro = "First section detail. " * 20
    details = "Second section detail. " * 20
    sectioned_path = _write_html(
        tmp_path,
        f"<h2>Intro</h2><p>{intro}</p><h2>Details</h2><p>{details}</p>",
        name="sectioned.html",
    )
    plain_path = _write_html(tmp_path, "<p>Without a heading.</p>", name="plain.html")
    loader = _loader()

    sectioned = loader.load_document(
        str(sectioned_path),
        sectioned_path.name,
        metadata=_metadata("version-a"),
    )
    plain = loader.load_document(
        str(plain_path),
        plain_path.name,
        metadata=_metadata("version-a"),
    )

    section_ids_by_page = {
        document["page_number"]: document["section_id"] for document in sectioned
    }
    assert section_ids_by_page == {1: "Intro", 2: "Details"}
    assert {document["section_id"] for document in plain} == {"page:1"}


def test_html_headings_only_index_sections_with_body_and_prefix_every_leaf(tmp_path):
    path = _write_html(
        tmp_path,
        "<h1>Warehouse guide</h1>"
        "<h2>Network</h2><p>Gateway uses TCP 9201.</p>"
        "<h2>Maintenance</h2><h3>Battery</h3>"
        f"<p>{'Replace after 700 cycles. ' * 90}</p>",
    )
    sections = parse_html_file_to_sections(path)
    assert len(sections) == 2
    assert [item["page"] for item in sections] == [1, 2]
    assert [item["heading_path"] for item in sections] == [
        ("Warehouse guide", "Network"),
        ("Warehouse guide", "Maintenance", "Battery"),
    ]

    leaves = [
        chunk
        for chunk in _loader().load_document(
            str(path), path.name, metadata=_metadata("version-a")
        )
        if chunk["chunk_level"] == 3
    ]
    assert len(leaves) > 2
    assert all("[标题路径: Warehouse guide >" in chunk["text"] for chunk in leaves)
    assert all(
        "Gateway uses" in chunk["text"] or "Replace after" in chunk["text"]
        for chunk in leaves
    )
    assert all(chunk["section_id"] in {"Network", "Battery"} for chunk in leaves)


def test_html_table_rows_keep_header_value_pairs_without_duplicate_cells(tmp_path):
    path = _write_html(
        tmp_path,
        "<h1>Orion 支持套餐</h1>"
        "<table><caption>响应承诺</caption>"
        "<tr><th>套餐</th><th>设备上限</th><th>响应时间</th></tr>"
        "<tr><td>Starter</td><td>10 台</td><td>下一个工作日</td></tr>"
        "<tr><td>Pro</td><td>50 台</td><td>4 小时</td></tr></table>",
    )
    sections = parse_html_file_to_sections(path)
    assert len(sections) == 1
    assert sections[0]["text"].splitlines() == [
        "响应承诺：套餐=Starter；设备上限=10 台；响应时间=下一个工作日",
        "响应承诺：套餐=Pro；设备上限=50 台；响应时间=4 小时",
    ]

    leaves = [
        chunk
        for chunk in _loader().load_document(
            str(path), path.name, metadata=_metadata("version-a")
        )
        if chunk["chunk_level"] == 3
    ]
    assert len(leaves) == 1
    assert leaves[0]["text"].startswith("[标题路径: Orion 支持套餐]")
    assert leaves[0]["text"].count("套餐=Pro") == 1
