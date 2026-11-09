"""Phase 5.2 tests for multi-chunk LightRAG ingestion identity and retries."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pytest

from retrieval.config import RetrievalSettings
from retrieval.lightrag_indexer import GraphConfigurationError, LightRAGGraphEngine


class _StatusRow:
    def __init__(self, status: str, error: Optional[str] = None):
        self.status = status
        self.error_msg = error


class _DocStatus:
    def __init__(self, statuses: Optional[Dict[str, _StatusRow]] = None):
        self.statuses = dict(statuses or {})

    async def get_by_id(self, doc_id: str) -> Optional[_StatusRow]:
        return self.statuses.get(doc_id)


class _MultiChunkLightRAG:
    def __init__(
        self,
        statuses: Optional[Dict[str, _StatusRow]] = None,
        fail_ids: Iterable[str] = (),
        delete_status: str = "success",
    ):
        self.doc_status = _DocStatus(statuses)
        self.fail_ids = set(fail_ids)
        self.delete_status = delete_status
        self.inserted: List[Dict[str, Any]] = []
        self.deleted: List[str] = []
        self.process_calls = 0
        self.finalized = 0

    async def initialize_storages(self) -> None:
        return None

    async def finalize_storages(self) -> None:
        self.finalized += 1

    async def ainsert(
        self,
        input: Any,
        split_by_character: Optional[str] = None,
        split_by_character_only: bool = False,
        ids: Any = None,
        file_paths: Any = None,
        track_id: Optional[str] = None,
    ) -> str:
        self.inserted.append(
            {
                "input": list(input) if isinstance(input, list) else input,
                "ids": list(ids) if ids is not None else None,
                "file_paths": list(file_paths) if file_paths is not None else None,
                "track_id": track_id,
            }
        )
        for doc_id in ids or []:
            if doc_id in self.fail_ids:
                self.doc_status.statuses[doc_id] = _StatusRow(
                    "failed", f"forced failure for {doc_id}"
                )
            else:
                self.doc_status.statuses[doc_id] = _StatusRow("processed")
        return track_id or "track-test"

    async def adelete_by_doc_id(
        self, doc_id: str, delete_llm_cache: bool = False
    ) -> SimpleNamespace:
        self.deleted.append(doc_id)
        if self.delete_status == "success":
            self.doc_status.statuses.pop(doc_id, None)
            self.fail_ids.discard(doc_id)
        return SimpleNamespace(status=self.delete_status)

    async def apipeline_process_enqueue_documents(self) -> None:
        self.process_calls += 1
        for doc_id, row in list(self.doc_status.statuses.items()):
            if row.status == "pending":
                self.doc_status.statuses[doc_id] = _StatusRow("processed")


def _settings(tmp_path: Any) -> RetrievalSettings:
    return RetrievalSettings(
        graph_storage_root=tmp_path,
        graph_llm_api_key="test-key",
        graph_llm_base_url="http://localhost:9999/v1",
    )


def _chunks(
    *,
    doc_id: str,
    workspace_id: str = "tenant_a",
    count: int = 2,
    filename: str = "report.md",
    title: str = "Supplier report",
    start: int = 0,
) -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": f"{doc_id}:c{index:05d}",
            "doc_id": doc_id,
            "doc_title": title,
            "workspace_id": workspace_id,
            "content": f"Controlled fact {index} for {doc_id}.",
            "metadata": {"file_path": filename},
        }
        for index in range(start, start + count)
    ]


def _engine(tmp_path: Any, fake: _MultiChunkLightRAG, workspace_id: str = "tenant_a"):
    return LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id=workspace_id,
        lightrag_factory=lambda _: fake,
    )


def _run(engine: LightRAGGraphEngine, chunks: Sequence[Any], track_id: str = "track"):
    async def scenario() -> Dict[str, Any]:
        return await engine.aindex_chunks(chunks, track_id=track_id)

    try:
        return asyncio.run(scenario())
    finally:
        engine.close()


def test_multichunk_same_filename_gets_unique_stable_sources(tmp_path: Any) -> None:
    fake = _MultiChunkLightRAG()
    engine = _engine(tmp_path, fake)
    chunks = _chunks(doc_id="doc-a", start=0) + _chunks(doc_id="doc-b", start=2)

    result = _run(engine, chunks)

    assert result["status"] == "completed"
    assert result["indexed_documents"] == 4
    assert result["submitted_chunks"] == 4
    assert result["source_documents"] == 2
    assert len(fake.inserted) == 1
    assert fake.inserted[0]["ids"] == [chunk["chunk_id"] for chunk in chunks]
    assert len(set(fake.inserted[0]["file_paths"])) == 4
    assert all(
        path.startswith("report.md::chunk::") for path in fake.inserted[0]["file_paths"]
    )
    assert [mapping["chunk_id"] for mapping in result["chunk_mappings"]] == [
        chunk["chunk_id"] for chunk in chunks
    ]


def test_multichunk_idempotent_second_call_submits_nothing(tmp_path: Any) -> None:
    chunks = _chunks(doc_id="doc-a", count=4)
    statuses = {
        chunks[0]["chunk_id"]: _StatusRow("processed"),
        chunks[1]["chunk_id"]: _StatusRow("processed"),
    }
    fake = _MultiChunkLightRAG(statuses=statuses)
    engine = _engine(tmp_path, fake)

    first = _run(engine, chunks)
    second = _run(engine, chunks)

    assert first["indexed_documents"] == 2
    assert first["skipped_chunks"] == [chunks[0]["chunk_id"], chunks[1]["chunk_id"]]
    assert fake.inserted[0]["ids"] == [chunk["chunk_id"] for chunk in chunks[2:]]
    assert len(fake.inserted) == 1
    assert second["status"] == "completed"
    assert second["indexed_documents"] == 0
    assert second["submitted_chunks"] == 0
    assert second["skipped_chunks"] == [chunk["chunk_id"] for chunk in chunks]
    assert len(fake.inserted) == 1


def test_multichunk_retries_failed_chunk_without_reinserting_processed(tmp_path: Any) -> None:
    chunks = _chunks(doc_id="doc-a", count=4)
    statuses = {
        chunks[0]["chunk_id"]: _StatusRow("processed"),
        chunks[1]["chunk_id"]: _StatusRow("failed", "earlier extraction failure"),
    }
    fake = _MultiChunkLightRAG(statuses=statuses)
    engine = _engine(tmp_path, fake)

    result = _run(engine, chunks)

    assert result["status"] == "completed"
    assert result["retried_failed_chunks"] == [chunks[1]["chunk_id"]]
    assert result["skipped_chunks"] == [chunks[0]["chunk_id"]]
    assert fake.deleted == [chunks[1]["chunk_id"]]
    assert fake.inserted[0]["ids"] == [chunk["chunk_id"] for chunk in chunks[1:]]
    assert result["indexed_documents"] == 3
    assert result["processed_chunks"] == [chunk["chunk_id"] for chunk in chunks]


def test_multichunk_reports_partial_failure_and_recovers_on_retry(tmp_path: Any) -> None:
    chunks = _chunks(doc_id="doc-a", count=4)
    fake = _MultiChunkLightRAG(fail_ids={chunks[1]["chunk_id"]})
    engine = _engine(tmp_path, fake)

    first = _run(engine, chunks)
    second = _run(engine, chunks)

    assert first["status"] == "partial_failure"
    assert first["indexed_documents"] == 3
    assert first["failed_chunks"] == [
        {
            "chunk_id": chunks[1]["chunk_id"],
            "doc_id": "doc-a",
            "error": f"forced failure for {chunks[1]['chunk_id']}",
        }
    ]
    assert second["status"] == "completed"
    assert second["retried_failed_chunks"] == [chunks[1]["chunk_id"]]
    assert second["skipped_chunks"] == [
        chunks[0]["chunk_id"],
        chunks[2]["chunk_id"],
        chunks[3]["chunk_id"],
    ]
    assert fake.deleted == [chunks[1]["chunk_id"]]
    assert fake.inserted[0]["ids"] == [chunk["chunk_id"] for chunk in chunks]
    assert fake.inserted[1]["ids"] == [chunks[1]["chunk_id"]]
    assert second["indexed_documents"] == 1


def test_multichunk_reports_empty_and_duplicate_inputs(tmp_path: Any) -> None:
    chunks = _chunks(doc_id="doc-a", count=1)
    empty = {
        "chunk_id": "doc-a:c99999",
        "doc_id": "doc-a",
        "workspace_id": "tenant_a",
        "content": "   ",
    }
    fake = _MultiChunkLightRAG()
    engine = _engine(tmp_path, fake)

    result = _run(engine, [chunks[0], chunks[0], empty])

    assert result["empty_chunks"] == ["doc-a:c99999"]
    assert result["duplicate_input_chunks"] == [chunks[0]["chunk_id"]]
    assert fake.inserted[0]["ids"] == [chunks[0]["chunk_id"]]
    assert result["indexed_documents"] == 1


def test_multichunk_delete_failure_stops_retry(tmp_path: Any) -> None:
    chunks = _chunks(doc_id="doc-a", count=1)
    statuses = {chunks[0]["chunk_id"]: _StatusRow("failed", "old failure")}
    fake = _MultiChunkLightRAG(statuses=statuses, delete_status="fail")
    engine = _engine(tmp_path, fake)

    with pytest.raises(GraphConfigurationError, match="failed to delete"):
        _run(engine, chunks)

    assert fake.inserted == []
