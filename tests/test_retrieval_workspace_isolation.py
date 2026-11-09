"""Workspace-isolation tests.

The live PostgreSQL test is opt-in because CI may not have the ParadeDB image
running. Set ``RETRIEVAL_TEST_DATABASE_URL`` to a disposable database to run it.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from retrieval.config import RetrievalSettings
from retrieval.db.connection import Database
from retrieval.db.migrate import apply_migrations
from retrieval.db.repository import IngestionRepository
from retrieval.schemas import (
    DocumentChunk,
    DocumentMetadata,
    compute_document_id,
    validate_workspace_id,
)


def test_workspace_ids_reject_traversal():
    for invalid in ("", "../tenant", "tenant/name", "tenant\\name", ".hidden"):
        with pytest.raises(ValueError):
            validate_workspace_id(invalid)
    assert validate_workspace_id("tenant_a") == "tenant_a"
    assert validate_workspace_id("tenant:one") == "tenant:one"


def test_document_ids_are_scoped_by_workspace():
    content_hash = "a" * 64
    first = compute_document_id("tenant_a", content_hash)
    second = compute_document_id("tenant_b", content_hash)
    assert first != second
    assert compute_document_id("tenant_a", content_hash) == first


_TEST_DSN = os.getenv("RETRIEVAL_TEST_DATABASE_URL")


@pytest.mark.skipif(not _TEST_DSN, reason="RETRIEVAL_TEST_DATABASE_URL is not set")
def test_live_postgres_workspace_isolation(tmp_path):
    async def scenario():
        settings = RetrievalSettings(
            database_url=_TEST_DSN,
            storage_root=tmp_path,
        )
        database = Database(settings)
        workspace_a = f"iso_a_{uuid.uuid4().hex[:8]}"
        workspace_b = f"iso_b_{uuid.uuid4().hex[:8]}"
        await database.connect()
        try:
            await database.assert_healthy()
            await apply_migrations(database)
            repository = IngestionRepository(database)

            await repository.ensure_workspace(workspace_a)
            await repository.ensure_workspace(workspace_b)

            content_hash = uuid.uuid4().hex + uuid.uuid4().hex
            doc_id = compute_document_id(workspace_a, content_hash)
            document = DocumentMetadata(
                doc_id=doc_id,
                workspace_id=workspace_a,
                filename="policy.md",
                doc_title="Policy",
                mime_type="text/markdown",
                content_hash=content_hash,
                byte_size=100,
                page_count=1,
                storage_path="test/policy.md",
                metadata={},
            )
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}:c00000",
                doc_id=doc_id,
                doc_title="Policy",
                content="Tenant A confidential policy text.",
                page_number=1,
                section_hierarchy=["Policy"],
                char_count=35,
                token_count_est=9,
                workspace_id=workspace_a,
                chunk_index=0,
                source_type="markdown",
            )
            await repository.save_document_with_chunks(document, [chunk])

            assert await repository.get_document(workspace_a, doc_id) is not None
            assert await repository.get_document(workspace_b, doc_id) is None
            assert len(await repository.get_chunks(workspace_a, doc_id)) == 1
            assert await repository.get_chunks(workspace_b, doc_id) == []
            assert await repository.find_document_by_hash(workspace_b, content_hash) is None
        finally:
            async with database.acquire() as connection:
                await connection.execute(
                    "DELETE FROM workspaces WHERE workspace_id = ANY($1::text[])",
                    [workspace_a, workspace_b],
                )
            await database.close()

    asyncio.run(scenario())
