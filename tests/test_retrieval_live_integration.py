"""Opt-in live Phase 3 integration tests against ParadeDB PostgreSQL.

Set ``RETRIEVAL_TEST_DATABASE_URL`` to a disposable database to run these tests.
They use deterministic fake models so no Hugging Face download is required.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import replace

import numpy as np
import pytest

from retrieval.bm25 import ParadeDBBM25Retriever
from retrieval.config import RetrievalSettings
from retrieval.db.connection import Database
from retrieval.db.migrate import apply_migrations
from retrieval.db.repository import IngestionRepository
from retrieval.schemas import DocumentChunk, DocumentMetadata, compute_document_id
from retrieval.service import RetrievalService
from retrieval.vector import PGVectorRetriever, PGVectorStore


_TEST_DSN = os.getenv("RETRIEVAL_TEST_DATABASE_URL")


class _DeterministicEmbedder:
    model_name = "phase3-test-embedder"
    model_revision = "phase3-test-revision"

    @staticmethod
    def _vector(text):
        vector = np.zeros(1024, dtype=np.float32)
        lowered = text.lower()
        if "alpha" in lowered:
            vector[0] = 1.0
        elif "beta" in lowered:
            vector[1] = 1.0
        else:
            vector[2] = 1.0
        return vector

    async def embed_documents(self, texts, batch_size=None):
        return np.stack([self._vector(text) for text in texts])

    async def embed_query(self, query):
        return self._vector(query)


class _FakeReranker:
    async def rerank(self, query, chunks, top_k=5):
        scored = [
            replace(
                chunk,
                cross_encoder_logit=(
                    5.0 if "tenant a" in chunk.content.lower() else -5.0
                ),
            )
            for chunk in chunks
        ]
        scored.sort(
            key=lambda chunk: (-float(chunk.cross_encoder_logit), chunk.chunk_id)
        )
        return [
            replace(chunk, rerank_rank=rank)
            for rank, chunk in enumerate(scored[:top_k], start=1)
        ]


def _document(workspace_id, suffix, content):
    content_hash = uuid.uuid4().hex + uuid.uuid4().hex
    doc_id = compute_document_id(workspace_id, content_hash)
    document = DocumentMetadata(
        doc_id=doc_id,
        workspace_id=workspace_id,
        filename=f"policy-{suffix}.md",
        doc_title=f"Policy {suffix}",
        mime_type="text/markdown",
        content_hash=content_hash,
        byte_size=len(content),
        page_count=1,
        storage_path=f"test/{workspace_id}/policy-{suffix}.md",
        metadata={"suffix": suffix},
    )
    chunk = DocumentChunk(
        chunk_id=f"{doc_id}:c00000",
        doc_id=doc_id,
        doc_title=document.doc_title,
        content=content,
        page_number=1,
        section_hierarchy=["Policy"],
        char_count=len(content),
        token_count_est=max(1, len(content.split())),
        workspace_id=workspace_id,
        chunk_index=0,
        source_type="markdown",
        metadata={"suffix": suffix},
    )
    return document, chunk


@pytest.mark.skipif(not _TEST_DSN, reason="RETRIEVAL_TEST_DATABASE_URL is not set")
def test_live_phase3_vector_bm25_and_hybrid_workspace_isolation():
    async def scenario():
        settings = RetrievalSettings(database_url=_TEST_DSN)
        database = Database(settings)
        workspace_a = f"p3_a_{uuid.uuid4().hex[:8]}"
        workspace_b = f"p3_b_{uuid.uuid4().hex[:8]}"
        await database.connect()
        try:
            await database.assert_healthy()
            await apply_migrations(database)
            repository = IngestionRepository(database)
            await repository.ensure_workspace(workspace_a)
            await repository.ensure_workspace(workspace_b)

            document_a, chunk_a = _document(
                workspace_a,
                "a",
                "Alpha tenant A confidential defect tolerance is 0.5 percent.",
            )
            document_b, chunk_b = _document(
                workspace_b,
                "b",
                "Alpha tenant B confidential defect tolerance is 2.0 percent.",
            )
            await repository.save_document_with_chunks(document_a, [chunk_a])
            await repository.save_document_with_chunks(document_b, [chunk_b])

            embedder = _DeterministicEmbedder()
            store = PGVectorStore(database, embedder, settings)
            assert await store.upsert_embeddings(workspace_a, [chunk_a]) == 1
            assert await store.upsert_embeddings(workspace_b, [chunk_b]) == 1

            vector = PGVectorRetriever(database, embedder, settings)
            vector_a = await vector.retrieve("alpha", top_k=5, workspace_id=workspace_a)
            vector_b = await vector.retrieve("alpha", top_k=5, workspace_id=workspace_b)
            assert [item.chunk_id for item in vector_a] == [chunk_a.chunk_id]
            assert [item.chunk_id for item in vector_b] == [chunk_b.chunk_id]

            bm25 = ParadeDBBM25Retriever(database, settings)
            bm25_a = await bm25.search(
                "alpha confidential", top_k=5, workspace_id=workspace_a
            )
            bm25_b = await bm25.search(
                "alpha confidential", top_k=5, workspace_id=workspace_b
            )
            assert [item.chunk_id for item in bm25_a] == [chunk_a.chunk_id]
            assert [item.chunk_id for item in bm25_b] == [chunk_b.chunk_id]

            service = RetrievalService(
                database,
                settings=settings,
                embedder=embedder,
                reranker=_FakeReranker(),
                vector_retriever=vector,
                bm25_retriever=bm25,
                vector_store=store,
            )
            evidence_a = await service.retrieve(
                "alpha confidential", top_k=5, workspace_id=workspace_a
            )
            evidence_b = await service.retrieve(
                "alpha confidential", top_k=5, workspace_id=workspace_b
            )
            assert [item.chunk_id for item in evidence_a] == [chunk_a.chunk_id]
            assert [item.chunk_id for item in evidence_b] == [chunk_b.chunk_id]
            assert evidence_a[0].cross_encoder_logit == pytest.approx(5.0)
            assert evidence_b[0].cross_encoder_logit == pytest.approx(-5.0)
            assert evidence_a[0].metadata["retrieval_sources"] == ["vector", "bm25"]
            assert evidence_b[0].metadata["retrieval_sources"] == ["vector", "bm25"]
        finally:
            async with database.acquire() as connection:
                await connection.execute(
                    "DELETE FROM workspaces WHERE workspace_id = ANY($1::text[])",
                    [workspace_a, workspace_b],
                )
            await database.close()

    asyncio.run(scenario())
