"""Database-foundation tests that do not require a live PostgreSQL server."""
from __future__ import annotations

import asyncio
import uuid

import pytest

from retrieval.config import (
    ConfigurationError,
    DEFAULT_STORAGE_ROOT,
    RetrievalSettings,
    normalize_asyncpg_dsn,
)
from retrieval.db import IngestionRepository, WorkspaceRequiredError
from retrieval.db.migrate import discover_migrations, migration_checksums
from retrieval.schemas import DocumentChunk, DocumentMetadata


class _AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self, fetchrow_result=None):
        self.fetchrow_result = fetchrow_result
        self.calls = []

    async def execute(self, sql, *params):
        self.calls.append(("execute", sql, params))
        return "OK"

    async def fetchrow(self, sql, *params):
        self.calls.append(("fetchrow", sql, params))
        return self.fetchrow_result

    async def fetchval(self, sql, *params):
        self.calls.append(("fetchval", sql, params))
        return 0

    async def fetch(self, sql, *params):
        self.calls.append(("fetch", sql, params))
        return []

    async def executemany(self, sql, params):
        self.calls.append(("executemany", sql, params))
        return None

    def transaction(self):
        return _AsyncContext(self)


class _FakeDatabase:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return _AsyncContext(self.connection)


def _settings(tmp_path, **overrides):
    values = {
        "database_url": "postgresql+asyncpg://postgres:postgres@localhost:5432/calibrag",
        "storage_root": tmp_path,
    }
    values.update(overrides)
    return RetrievalSettings(**values)


def _document(tmp_path, workspace_id="tenant_a"):
    return DocumentMetadata(
        doc_id="11111111-1111-4111-8111-111111111111",
        workspace_id=workspace_id,
        filename="policy.md",
        doc_title="Policy",
        mime_type="text/markdown",
        content_hash="a" * 64,
        byte_size=12,
        page_count=1,
        storage_path="tenant/aa/" + "a" * 64 + ".md",
        metadata={},
    )


def _chunk(workspace_id="tenant_a", doc_id="11111111-1111-4111-8111-111111111111"):
    return DocumentChunk(
        chunk_id=f"{doc_id}:c00000",
        doc_id=doc_id,
        doc_title="Policy",
        content="A policy paragraph with enough text for a chunk.",
        page_number=1,
        section_hierarchy=["Policy"],
        char_count=48,
        token_count_est=12,
        workspace_id=workspace_id,
        chunk_index=0,
        source_type="markdown",
    )


def test_normalize_asyncpg_dsn():
    assert (
        normalize_asyncpg_dsn("postgresql+asyncpg://user:pass@db:5432/calibrag")
        == "postgresql://user:pass@db:5432/calibrag"
    )
    assert (
        normalize_asyncpg_dsn("postgres://user:pass@db:5432/calibrag")
        == "postgresql://user:pass@db:5432/calibrag"
    )
    with pytest.raises(ConfigurationError):
        normalize_asyncpg_dsn("sqlite:///local.db")


def test_settings_reject_invalid_chunk_budget(tmp_path):
    with pytest.raises(ConfigurationError):
        _settings(tmp_path, chunk_target_chars=3000, chunk_max_chars=2000)
    assert DEFAULT_STORAGE_ROOT.name == "storage"


def test_migrations_are_discoverable_and_contain_foundation_schema():
    migrations = discover_migrations()
    versions = [migration.version for migration in migrations]
    assert versions == ["0001", "0002", "0003"]

    combined = "\n".join(migration.sql for migration in migrations)
    for expected in (
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE EXTENSION IF NOT EXISTS pg_search",
        "CREATE TABLE IF NOT EXISTS workspaces",
        "CREATE TABLE IF NOT EXISTS documents",
        "CREATE TABLE IF NOT EXISTS document_chunks",
        "CREATE TABLE IF NOT EXISTS ingestion_jobs",
        "CREATE TABLE IF NOT EXISTS chunk_embeddings",
        "USING bm25",
        "USING hnsw",
    ):
        assert expected in combined
    assert len(set(dict(migration_checksums()).values())) == 3


def test_repository_requires_workspace_and_filters_sql(tmp_path):
    connection = _FakeConnection()
    repository = IngestionRepository(_FakeDatabase(connection))

    with pytest.raises(WorkspaceRequiredError):
        asyncio.run(repository.find_document_by_hash("", "a" * 64))

    result = asyncio.run(repository.find_document_by_hash("tenant_a", "a" * 64))
    assert result is None
    sql, params = connection.calls[-1][1], connection.calls[-1][2]
    assert "workspace_id = $1" in sql
    assert params == ("tenant_a", "a" * 64)


def test_repository_chunk_lookup_is_tenant_scoped(tmp_path):
    connection = _FakeConnection()
    repository = IngestionRepository(_FakeDatabase(connection))
    doc_id = str(uuid.uuid4())

    asyncio.run(repository.get_chunks("tenant_a", doc_id))
    sql, params = connection.calls[-1][1], connection.calls[-1][2]
    assert "c.workspace_id = $1" in sql
    assert "c.doc_id = $2" in sql
    assert params[0] == "tenant_a"


def test_repository_rejects_cross_workspace_chunk(tmp_path):
    connection = _FakeConnection()
    repository = IngestionRepository(_FakeDatabase(connection))

    with pytest.raises(WorkspaceRequiredError):
        asyncio.run(
            repository.save_document_with_chunks(
                _document(tmp_path, "tenant_a"),
                [_chunk("tenant_b")],
            )
        )
    assert connection.calls == []
