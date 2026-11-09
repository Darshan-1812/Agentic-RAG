"""Unit tests for Docling normalization, chunking, storage, and pipeline flow."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from retrieval.config import RetrievalSettings
from retrieval.ingestion.chunker import chunk_blocks
from retrieval.ingestion.docling_parser import DoclingParser
from retrieval.ingestion.metadata import sha256_file
from retrieval.ingestion.pipeline import IngestionError, IngestionPipeline
from retrieval.ingestion.storage import DocumentStorage
from retrieval.schemas import (
    DocumentChunk,
    DocumentStatus,
    IngestionJobStatus,
)


class _Prov:
    def __init__(self, page_no):
        self.page_no = page_no


class _Item:
    def __init__(self, label, text="", page_no=1, table_markdown=None):
        self.label = label
        self.text = text
        self.prov = [_Prov(page_no)]
        self._table_markdown = table_markdown

    def export_to_markdown(self, document=None):
        return self._table_markdown or ""


class _Document:
    def __init__(self, entries):
        self.entries = entries
        self.pages = {1: object(), 2: object()}

    def iterate_items(self):
        return iter(self.entries)


class _Result:
    def __init__(self, document):
        self.document = document
        self.status = "success"
        self.errors = None


class _Converter:
    def __init__(self, document):
        self.document = document
        self.calls = 0

    def convert(self, path):
        self.calls += 1
        return _Result(self.document)


class _FailingConverter:
    def __init__(self):
        self.calls = 0

    def convert(self, path):
        self.calls += 1
        raise RuntimeError("simulated converter failure")


class _FakeRepository:
    def __init__(self):
        self.workspaces = set()
        self.documents = {}
        self.chunk_store = {}
        self.jobs = {}

    async def ensure_workspace(self, workspace_id, name=None, config=None):
        self.workspaces.add(workspace_id)

    async def find_document_by_hash(self, workspace_id, content_hash):
        return self.documents.get((workspace_id, content_hash))

    async def get_chunks(self, workspace_id, doc_id, limit=None):
        return list(self.chunk_store.get((workspace_id, doc_id), []))

    async def create_job(self, workspace_id, source_filename, doc_id=None):
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "workspace_id": workspace_id,
            "source_filename": source_filename,
            "status": IngestionJobStatus.QUEUED.value,
            "doc_id": doc_id,
            "stats": {},
        }
        return job_id

    async def update_job(
        self,
        workspace_id,
        job_id,
        status,
        error=None,
        stats=None,
        doc_id=None,
    ):
        job = self.jobs[job_id]
        assert job["workspace_id"] == workspace_id
        job["status"] = getattr(status, "value", status)
        job["error"] = error
        if stats is not None:
            job["stats"].update(stats)
        if doc_id:
            job["doc_id"] = doc_id

    async def save_document_with_chunks(self, document, chunks, job_id=None):
        self.documents[(document.workspace_id, document.content_hash)] = document
        self.chunk_store[(document.workspace_id, document.doc_id)] = list(chunks)
        return document


def _settings(tmp_path, **overrides):
    values = {
        "database_url": "postgresql+asyncpg://postgres:postgres@localhost:5432/calibrag",
        "storage_root": tmp_path / "storage",
        "max_upload_bytes": 5 * 1024 * 1024,
        "chunk_target_chars": 120,
        "chunk_max_chars": 180,
        "chunk_overlap_chars": 20,
    }
    values.update(overrides)
    return RetrievalSettings(**values)


def _fixture_document():
    return _Document(
        [
            (_Item("title", "Vendor Policy", 1), 1),
            (_Item("text", "This policy governs supplier delivery obligations.", 1), 1),
            (_Item("section_header", "Delivery Penalties", 1), 2),
            (
                _Item(
                    "text",
                    "Late delivery beyond seven business days triggers a five percent penalty.",
                    1,
                ),
                2,
            ),
            (
                _Item(
                    "table",
                    "",
                    2,
                    table_markdown="| Supplier | Penalty |\n|---|---|\n| Apex | 5% |",
                ),
                2,
            ),
        ]
    )


def _write_source(tmp_path: Path, name="policy.md") -> Path:
    path = tmp_path / name
    path.write_text("# Vendor Policy\n\nLate delivery penalties apply.\n", encoding="utf-8")
    return path


def test_docling_adapter_preserves_pages_sections_and_tables(tmp_path):
    parser = DoclingParser(
        settings=_settings(tmp_path), converter=_Converter(_fixture_document())
    )
    parsed = parser.parse(_write_source(tmp_path), "tenant_a")

    assert parsed.doc_title == "Vendor Policy"
    assert parsed.page_count == 2
    headings = [block.text for block in parsed.blocks if block.block_type == "heading"]
    assert headings == ["Vendor Policy", "Delivery Penalties"]
    table = next(block for block in parsed.blocks if block.block_type == "table")
    assert table.page_number == 2
    assert "| Apex | 5% |" in table.table_markdown
    assert table.section_hierarchy == ["Vendor Policy", "Delivery Penalties"]


def test_docling_pdf_options_use_pdf_format_option(tmp_path):
    pytest.importorskip("docling")
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    parser = DoclingParser(settings=_settings(tmp_path))
    converter = parser._get_converter()

    assert isinstance(converter, DocumentConverter)
    option = converter.format_to_options[InputFormat.PDF]
    assert isinstance(option, PdfFormatOption)
    assert option.pipeline_options.do_ocr is parser.settings.docling_ocr


def test_chunker_is_deterministic_and_keeps_tables_intact(tmp_path):
    parser = DoclingParser(
        settings=_settings(tmp_path), converter=_Converter(_fixture_document())
    )
    parsed = parser.parse(_write_source(tmp_path), "tenant_a")

    chunks = chunk_blocks(
        parsed.blocks,
        doc_id="11111111-1111-4111-8111-111111111111",
        workspace_id="tenant_a",
        doc_title=parsed.doc_title,
        target_chars=120,
        max_chars=180,
        overlap_chars=20,
        source_type="markdown",
    )
    repeat = chunk_blocks(
        parsed.blocks,
        doc_id="11111111-1111-4111-8111-111111111111",
        workspace_id="tenant_a",
        doc_title=parsed.doc_title,
        target_chars=120,
        max_chars=180,
        overlap_chars=20,
        source_type="markdown",
    )

    assert chunks
    assert [chunk.chunk_id for chunk in chunks] == [
        chunk.chunk_id for chunk in repeat
    ]
    assert all(chunk.workspace_id == "tenant_a" for chunk in chunks)
    assert all(chunk.chunk_id.startswith("11111111-1111-4111-8111-111111111111:c") for chunk in chunks)
    tables = [chunk for chunk in chunks if chunk.has_table]
    assert len(tables) == 1
    assert tables[0].table_markdown == "| Supplier | Penalty |\n|---|---|\n| Apex | 5% |"
    assert tables[0].content == tables[0].table_markdown


def test_storage_is_content_addressed_and_workspace_isolated(tmp_path):
    source = _write_source(tmp_path)
    content_hash = sha256_file(source)
    storage = DocumentStorage(tmp_path / "objects", max_upload_bytes=1024 * 1024)

    first = storage.store_file(source, "tenant:one", content_hash)
    second = storage.store_file(source, "tenant:one", content_hash)
    other = storage.store_file(source, "tenant_two", content_hash)

    assert first.created is True
    assert second.created is False
    assert first.relative_path == second.relative_path
    assert other.relative_path != first.relative_path
    assert "tenant:one" not in first.relative_path
    assert storage.read_bytes(first.relative_path) == source.read_bytes()


def test_pipeline_deduplicates_within_workspace_only(tmp_path):
    source = _write_source(tmp_path)
    converter = _Converter(_fixture_document())
    settings = _settings(tmp_path)
    repository = _FakeRepository()
    pipeline = IngestionPipeline(
        repository,
        settings=settings,
        storage=DocumentStorage(settings.resolved_storage_root, settings.max_upload_bytes),
        parser=DoclingParser(settings=settings, converter=converter),
    )

    first = asyncio.run(pipeline.ingest_file("tenant_a", source))
    duplicate = asyncio.run(pipeline.ingest_file("tenant_a", source))
    other_workspace = asyncio.run(pipeline.ingest_file("tenant_b", source))

    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert duplicate.doc_id == first.doc_id
    assert other_workspace.duplicate is False
    assert other_workspace.doc_id != first.doc_id
    assert converter.calls == 2
    assert first.chunks
    assert all(chunk.workspace_id == "tenant_a" for chunk in first.chunks)
    assert all(chunk.workspace_id == "tenant_b" for chunk in other_workspace.chunks)
    assert duplicate.document.status == DocumentStatus.READY


def test_pipeline_marks_job_failed_and_preserves_original_error(tmp_path):
    source = _write_source(tmp_path)
    settings = _settings(tmp_path)
    repository = _FakeRepository()
    pipeline = IngestionPipeline(
        repository,
        settings=settings,
        storage=DocumentStorage(settings.resolved_storage_root, settings.max_upload_bytes),
        parser=DoclingParser(settings=settings, converter=_FailingConverter()),
    )

    with pytest.raises(IngestionError):
        asyncio.run(pipeline.ingest_file("tenant_a", source))

    assert len(repository.jobs) == 1
    job = next(iter(repository.jobs.values()))
    assert job["status"] == IngestionJobStatus.FAILED.value
    assert "simulated converter failure" in job["error"]


def test_pipeline_requires_explicit_workspace(tmp_path):
    source = _write_source(tmp_path)
    settings = _settings(tmp_path)
    pipeline = IngestionPipeline(
        _FakeRepository(),
        settings=settings,
        storage=DocumentStorage(settings.resolved_storage_root, settings.max_upload_bytes),
        parser=DoclingParser(settings=settings, converter=_Converter(_fixture_document())),
    )

    with pytest.raises(ValueError):
        asyncio.run(pipeline.ingest_file("", source))
