"""Opt-in Phase 5 end-to-end ingestion, hybrid retrieval, and graph context test.

Set ``RETRIEVAL_TEST_DATABASE_URL`` to a disposable ParadeDB PostgreSQL database.
The test uses deterministic fake embedding/reranking models so it exercises the
real ingestion pipeline, PostgreSQL schema, pgvector, ParadeDB BM25, RRF, and the
unified service without downloading Hugging Face models or calling an LLM.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import replace

import numpy as np
import pytest

from retrieval.config import RetrievalSettings
from retrieval.db.connection import Database
from retrieval.db.migrate import apply_migrations
from retrieval.db.repository import IngestionRepository
from retrieval.ingestion.pipeline import IngestionPipeline
from retrieval.ingestion.storage import DocumentStorage
from retrieval.schemas import ParsedBlock, ParsedDocument
from retrieval.service import RetrievalService


_TEST_DSN = os.getenv("RETRIEVAL_TEST_DATABASE_URL")


class _DeterministicEmbedder:
    model_name = "phase5-test-embedder"
    model_revision = "phase5-test-revision"

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


class _StaticParser:
    """Return deterministic parsed blocks without requiring Docling."""

    def __init__(self, text: str):
        self.text = text

    def parse(self, file_path, workspace_id):
        return ParsedDocument(
            doc_title="Supplier Quality Policy",
            page_count=1,
            blocks=[
                ParsedBlock(
                    block_type="heading",
                    text="Supplier Quality Policy",
                    page_number=1,
                    section_hierarchy=["Supplier Quality Policy"],
                    level=1,
                ),
                ParsedBlock(
                    block_type="text",
                    text=self.text,
                    page_number=1,
                    section_hierarchy=["Supplier Quality Policy"],
                ),
            ],
            metadata={"parser": "phase5-static"},
        )


class _FakeGraphEngine:
    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id
        self.calls = []

    def traverse_entities(self, query, entity_seeds, max_hops=2):
        self.calls.append((query, list(entity_seeds), max_hops))
        return {
            "query": query,
            "seed_entities": list(entity_seeds),
            "max_hops": max_hops,
            "total_nodes_found": 2,
            "nodes": [
                {"id": "VoltCore", "entity_name": "VoltCore", "entity_type": "Company"},
                {"id": "EV-Pack-V2", "entity_name": "EV-Pack-V2", "entity_type": "Product"},
            ],
            "edges": [
                {
                    "source_id": "VoltCore",
                    "target_id": "EV-Pack-V2",
                    "relation_type": "SUPPLIES",
                    "weight": 1.0,
                }
            ],
            "context_summary": "Graph Link: VoltCore -> [SUPPLIES] -> EV-Pack-V2",
            "truncated": False,
            "workspace_id": self.workspace_id,
        }


def _write_source(tmp_path, name: str) -> str:
    path = tmp_path / name
    path.write_text("# Supplier Quality Policy\n\nAlpha defect tolerance policy.\n", encoding="utf-8")
    return str(path)


@pytest.mark.skipif(not _TEST_DSN, reason="RETRIEVAL_TEST_DATABASE_URL is not set")
def test_phase5_ingestion_hybrid_retrieval_and_graph_context(tmp_path):
    async def scenario():
        settings = RetrievalSettings(
            database_url=_TEST_DSN,
            storage_root=tmp_path / "storage",
            graph_storage_root=tmp_path / "graph",
            chunk_target_chars=80,
            chunk_max_chars=120,
            chunk_overlap_chars=10,
            retrieval_top_k=3,
            vector_candidate_depth=10,
            bm25_candidate_depth=10,
            hybrid_candidate_depth=10,
            rerank_candidate_depth=5,
        )
        database = Database(settings)
        workspace_a = f"p5_a_{uuid.uuid4().hex[:8]}"
        workspace_b = f"p5_b_{uuid.uuid4().hex[:8]}"
        await database.connect()
        try:
            await database.assert_healthy()
            await apply_migrations(database)
            repository = IngestionRepository(database)

            text_a = "Alpha tenant A confidential defect tolerance is 0.5 percent."
            text_b = "Alpha tenant B confidential defect tolerance is 2.0 percent."
            source_a = _write_source(tmp_path, "tenant-a.md")
            source_b = _write_source(tmp_path, "tenant-b.md")

            pipeline_a = IngestionPipeline(
                repository,
                settings=settings,
                storage=DocumentStorage(
                    settings.resolved_storage_root, settings.max_upload_bytes
                ),
                parser=_StaticParser(text_a),
            )
            pipeline_b = IngestionPipeline(
                repository,
                settings=settings,
                storage=DocumentStorage(
                    settings.resolved_storage_root, settings.max_upload_bytes
                ),
                parser=_StaticParser(text_b),
            )

            ingested_a = await pipeline_a.ingest_file(workspace_a, source_a)
            ingested_b = await pipeline_b.ingest_file(workspace_b, source_b)
            assert ingested_a.chunks
            assert ingested_b.chunks
            assert ingested_a.duplicate is False
            assert ingested_b.duplicate is False

            duplicate = await pipeline_a.ingest_file(workspace_a, source_a)
            assert duplicate.duplicate is True
            assert duplicate.doc_id == ingested_a.doc_id

            embedder = _DeterministicEmbedder()
            service = RetrievalService(
                database,
                settings=settings,
                embedder=embedder,
                reranker=_FakeReranker(),
            )
            assert await service.index_document(workspace_a, ingested_a.doc_id) == len(
                ingested_a.chunks
            )
            assert await service.index_document(workspace_b, ingested_b.doc_id) == len(
                ingested_b.chunks
            )

            graph_a = _FakeGraphEngine(workspace_a)
            graph_b = _FakeGraphEngine(workspace_b)
            result_a = await service.retrieve_with_graph(
                "alpha confidential defect tolerance",
                top_k=3,
                workspace_id=workspace_a,
                entity_seeds=["VoltCore", "EV-Pack-V2"],
                graph_engine=graph_a,
            )
            result_b = await service.retrieve_with_graph(
                "alpha confidential defect tolerance",
                top_k=3,
                workspace_id=workspace_b,
                entity_seeds=["VoltCore", "EV-Pack-V2"],
                graph_engine=graph_b,
            )

            assert result_a.graph_error is None
            assert result_b.graph_error is None
            assert result_a.evidence
            assert result_b.evidence
            assert any("tenant a" in item.content.lower() for item in result_a.evidence)
            assert any("tenant b" in item.content.lower() for item in result_b.evidence)
            assert all("tenant b" not in item.content.lower() for item in result_a.evidence)
            assert all("tenant a" not in item.content.lower() for item in result_b.evidence)
            assert result_a.evidence[0].cross_encoder_logit == pytest.approx(5.0)
            assert result_b.evidence[0].cross_encoder_logit == pytest.approx(-5.0)
            assert result_a.evidence[0].metadata["retrieval_sources"] == [
                "vector",
                "bm25",
            ]
            assert result_a.graph_context == (
                "Graph Link: VoltCore -> [SUPPLIES] -> EV-Pack-V2"
            )
            assert graph_a.calls and graph_b.calls
            assert not set(item.chunk_id for item in result_a.evidence).intersection(
                item.chunk_id for item in result_b.evidence
            )
        finally:
            async with database.acquire() as connection:
                await connection.execute(
                    "DELETE FROM workspaces WHERE workspace_id = ANY($1::text[])",
                    [workspace_a, workspace_b],
                )
            await database.close()

    asyncio.run(scenario())
