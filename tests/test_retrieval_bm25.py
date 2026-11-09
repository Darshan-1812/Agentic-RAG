"""Unit tests for ParadeDB BM25 retrieval and query sanitization."""
from __future__ import annotations

import json

import pytest

from retrieval.bm25 import (
    BM25RetrievalError,
    ParadeDBBM25Retriever,
    normalize_bm25_query,
)
from retrieval.config import RetrievalSettings


class _AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.calls = []

    async def fetch(self, sql, *params):
        self.calls.append((sql, params))
        return self.rows


class _FakeDatabase:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return _AsyncContext(self.connection)


def _row():
    return {
        "chunk_id": "doc-1:c00000",
        "doc_id": "doc-1",
        "content": "Maximum defect tolerance is 0.5 percent.",
        "page_number": 2,
        "section_hierarchy": ["Quality"],
        "has_table": False,
        "table_markdown": None,
        "source_type": "pdf",
        "metadata": json.dumps({"filename": "policy.pdf"}),
        "bm25_score": 3.75,
    }


def test_bm25_query_sanitizes_operators_and_punctuation():
    assert normalize_bm25_query("  ") == ""
    assert normalize_bm25_query("!!! --- ???") == ""
    assert normalize_bm25_query('defect OR "tolerance" --') == "defect OR tolerance"
    assert normalize_bm25_query("EV-Pack-V2 v0.5") == "EV-Pack-V2 v0.5"
    assert normalize_bm25_query("C++ and C#") == "C and C"
    malicious = normalize_bm25_query("x'; DROP TABLE document_chunks; --")
    assert "'" not in malicious
    assert ";" not in malicious
    assert "--" not in malicious


def test_bm25_query_is_capped_to_64_tokens():
    normalized = normalize_bm25_query(" ".join(f"word{index}" for index in range(80)))
    assert len(normalized.split()) == 64


async def test_bm25_retriever_maps_scores_and_uses_bound_parameters():
    connection = _FakeConnection(rows=[_row()])
    retriever = ParadeDBBM25Retriever(
        _FakeDatabase(connection), RetrievalSettings()
    )

    results = await retriever.search(
        'defect OR "tolerance" --',
        top_k=5,
        workspace_id="tenant_a",
    )
    assert len(results) == 1
    result = results[0]
    assert result.chunk_id == "doc-1:c00000"
    assert result.bm25_score == pytest.approx(3.75)
    assert result.bm25_rank == 1
    assert result.retrieval_sources == ("bm25",)
    assert result.metadata == {"filename": "policy.pdf"}

    sql, params = connection.calls[0]
    assert "c.content @@@ paradedb.match('content', $2)" in sql
    assert "paradedb.score(c) AS bm25_score" in sql
    assert "c.workspace_id = $1" in sql
    assert params == ("tenant_a", "defect OR tolerance", 50)


async def test_bm25_retriever_empty_query_skips_database():
    connection = _FakeConnection(rows=[_row()])
    retriever = ParadeDBBM25Retriever(
        _FakeDatabase(connection), RetrievalSettings()
    )

    assert await retriever.search("???", workspace_id="tenant_a") == []
    assert connection.calls == []

    with pytest.raises(BM25RetrievalError):
        await retriever.search("query", top_k=0, workspace_id="tenant_a")
