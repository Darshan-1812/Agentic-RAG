"""Unit tests for pgvector persistence and HNSW retrieval."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from retrieval.config import RetrievalSettings
from retrieval.vector import (
    PGVectorRetriever,
    PGVectorStore,
    VectorIndexError,
    VectorRetrievalError,
)


class _AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self, rows=None, existing=None):
        self.rows = rows or []
        self.existing = existing or []
        self.calls = []
        self.executemany_rows = []

    async def fetch(self, sql, *params):
        self.calls.append(("fetch", sql, params))
        if "FROM document_chunks" in sql and "ANY($2::text[])" in sql:
            return self.existing
        return self.rows

    async def execute(self, sql, *params):
        self.calls.append(("execute", sql, params))
        return "DELETE 1"

    async def executemany(self, sql, rows):
        self.calls.append(("executemany", sql, rows))
        self.executemany_rows.extend(rows)
        return None

    def transaction(self):
        return _AsyncContext(self)


class _FakeDatabase:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return _AsyncContext(self.connection)


class _FakeEmbedder:
    model_name = "test-model"
    model_revision = "test-revision"

    def __init__(self):
        self.document_calls = 0
        self.query_calls = 0

    async def embed_documents(self, texts, batch_size=None):
        self.document_calls += 1
        return np.tile(
            np.ones(1024, dtype=np.float32) / np.sqrt(1024),
            (len(texts), 1),
        )

    async def embed_query(self, query):
        self.query_calls += 1
        return np.ones(1024, dtype=np.float32) / np.sqrt(1024)


def _settings():
    return RetrievalSettings()


def _row(distance=0.25):
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
        "model_name": "test-model",
        "model_revision": "test-revision",
        "distance": distance,
    }


async def test_vector_store_upserts_workspace_scoped_embeddings():
    connection = _FakeConnection(existing=[{"chunk_id": "doc-1:c00000"}])
    embedder = _FakeEmbedder()
    store = PGVectorStore(_FakeDatabase(connection), embedder, _settings())

    chunks = [
        SimpleNamespace(
            chunk_id="doc-1:c00000",
            content="Maximum defect tolerance is 0.5 percent.",
        )
    ]
    count = await store.upsert_embeddings("tenant_a", chunks)
    assert count == 1
    assert embedder.document_calls == 1

    sql, rows = connection.calls[-1][1], connection.calls[-1][2]
    assert "INSERT INTO chunk_embeddings" in sql
    assert "$5::vector" in sql
    assert rows[0][0] == "doc-1:c00000"
    assert rows[0][1] == "tenant_a"
    assert rows[0][2] == "test-model"
    assert rows[0][3] == "test-revision"


async def test_vector_store_rejects_missing_or_duplicate_chunks():
    connection = _FakeConnection(existing=[])
    store = PGVectorStore(_FakeDatabase(connection), _FakeEmbedder(), _settings())

    with pytest.raises(VectorIndexError):
        await store.upsert_embeddings(
            "tenant_a",
            [SimpleNamespace(chunk_id="missing:c00000", content="text")],
        )

    with pytest.raises(VectorIndexError):
        await store.upsert_embeddings(
            "tenant_a",
            [
                SimpleNamespace(chunk_id="doc-1:c00000", content="one"),
                SimpleNamespace(chunk_id="doc-1:c00000", content="two"),
            ],
        )


async def test_vector_retriever_maps_distance_and_ranks():
    connection = _FakeConnection(rows=[_row(0.25)])
    embedder = _FakeEmbedder()
    retriever = PGVectorRetriever(
        _FakeDatabase(connection), embedder, _settings()
    )

    results = await retriever.retrieve("defect tolerance", top_k=5, workspace_id="tenant_a")
    assert len(results) == 1
    result = results[0]
    assert result.chunk_id == "doc-1:c00000"
    assert result.vector_rank == 1
    assert result.vector_distance == pytest.approx(0.25)
    assert result.vector_similarity == pytest.approx(0.75)
    assert result.metadata == {"filename": "policy.pdf"}
    assert result.retrieval_sources == ("vector",)
    assert result.embedding_model == "test-model"

    fetch_calls = [call for call in connection.calls if call[0] == "fetch"]
    sql, params = fetch_calls[0][1], fetch_calls[0][2]
    assert "e.workspace_id = $2" in sql
    assert "e.model_name = $3" in sql
    assert "e.model_revision = $4" in sql
    assert params[1:] == ("tenant_a", "test-model", "test-revision", 50)
    assert any(
        call[0] == "execute" and "hnsw.ef_search" in call[1]
        for call in connection.calls
    )


async def test_vector_retriever_empty_query_skips_database_and_embedding():
    connection = _FakeConnection(rows=[_row()])
    embedder = _FakeEmbedder()
    retriever = PGVectorRetriever(
        _FakeDatabase(connection), embedder, _settings()
    )

    assert await retriever.retrieve("   ", workspace_id="tenant_a") == []
    assert connection.calls == []
    assert embedder.query_calls == 0

    with pytest.raises(VectorRetrievalError):
        await retriever.retrieve("query", top_k=0, workspace_id="tenant_a")
