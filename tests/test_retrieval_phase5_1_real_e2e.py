"""Opt-in Phase 5.1 real hybrid + GraphRAG end-to-end verification.

Set ``RETRIEVAL_PHASE5_1_REAL_TESTS=1``, ``RETRIEVAL_TEST_DATABASE_URL``, and
provide ``retrieval/.env`` with a real Groq key. The test uses real Docling,
BGE-M3, PostgreSQL/pgvector, ParadeDB BM25, RRF, BGE reranking, and LightRAG
extraction through Groq. It is skipped unless explicitly enabled.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from retrieval.bm25.paradedb import ParadeDBBM25Retriever
from retrieval.config import RetrievalSettings
from retrieval.db.connection import Database
from retrieval.db.migrate import apply_migrations
from retrieval.db.repository import IngestionRepository
from retrieval.embeddings.bge_m3 import BgeM3Embedder
from retrieval.hybrid.rrf import reciprocal_rank_fusion
from retrieval.ingestion.docling_parser import DoclingParser
from retrieval.ingestion.pipeline import IngestionPipeline
from retrieval.ingestion.storage import DocumentStorage
from retrieval.lightrag_indexer import LightRAGGraphEngine
from retrieval.reranking.bge_reranker import BgeReranker
from retrieval.service import RetrievalService
from retrieval.vector.pgvector import PGVectorRetriever

_ENV_FILE = Path(__file__).resolve().parents[1] / "retrieval" / ".env"
_TEST_DSN = os.getenv("RETRIEVAL_TEST_DATABASE_URL")
_REAL_ENABLED = os.getenv("RETRIEVAL_PHASE5_1_REAL_TESTS") == "1"
_GRAPH_PAUSE = float(os.getenv("RETRIEVAL_PHASE5_1_GRAPH_PAUSE_SECONDS", "65"))

pytestmark = pytest.mark.skipif(
    not (_REAL_ENABLED and _TEST_DSN and _ENV_FILE.exists()),
    reason=(
        "set RETRIEVAL_PHASE5_1_REAL_TESTS=1, RETRIEVAL_TEST_DATABASE_URL, "
        "and retrieval/.env to run the real Phase 5.1 E2E test"
    ),
)

_DOC_A = """Acme Technologies is a private software company. Alice works at Acme Technologies as the engineering lead.

Acme Technologies is headquartered in Pune, India. The Pune headquarters houses the core platform team.

Alice leads the Fraud Detection project at Acme Technologies. The Fraud Detection project is a strategic initiative.

The Fraud Detection project uses PostgreSQL as its primary database. PostgreSQL stores fraud signals and investigation records.
"""

_DOC_B = """Beta Systems is a logistics software company. Bob works at Beta Systems as the product lead.

Beta Systems is headquartered in Mumbai, India. The Mumbai headquarters houses the logistics platform team.

Bob leads the Inventory Sync project at Beta Systems. The Inventory Sync project is a strategic initiative.

The Inventory Sync project uses MySQL as its primary database. MySQL stores inventory snapshots and warehouse events.
"""

_DISTRACTOR_A = "The Acme cafeteria opens at 8:00 AM. Visitor parking is available in Building B.\n"
_DISTRACTOR_B = "The Beta cafeteria opens at 9:00 AM. Visitor parking is available in Building C.\n"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _evidence_text(result) -> str:
    return "\n".join(item.content for item in result.evidence).lower()


def test_phase5_1_real_hybrid_graphrag_e2e(tmp_path: Path) -> None:
    async def scenario() -> None:
        base = RetrievalSettings.from_env(env_file=str(_ENV_FILE))
        settings = replace(
            base,
            database_url=_TEST_DSN,
            storage_root=tmp_path / "storage",
            graph_storage_root=tmp_path / "graph",
            chunk_target_chars=1200,
            chunk_max_chars=1800,
            chunk_overlap_chars=0,
            retrieval_top_k=5,
            vector_candidate_depth=20,
            bm25_candidate_depth=20,
            hybrid_candidate_depth=20,
            rerank_candidate_depth=10,
            graph_traversal_timeout_seconds=180.0,
        )
        ws_a = f"p51a_{uuid.uuid4().hex[:10]}"
        ws_b = f"p51b_{uuid.uuid4().hex[:10]}"
        empty_ws = f"p51e_{uuid.uuid4().hex[:10]}"
        database = Database(settings)
        embedder = BgeM3Embedder(settings)
        reranker = BgeReranker(settings)
        service = RetrievalService(
            database, settings=settings, embedder=embedder, reranker=reranker
        )
        graph_a = LightRAGGraphEngine(settings=settings, workspace_id=ws_a, embedder=embedder)
        graph_b = LightRAGGraphEngine(settings=settings, workspace_id=ws_b, embedder=embedder)
        empty_graph = LightRAGGraphEngine(
            settings=settings, workspace_id=empty_ws, embedder=embedder
        )
        reload_a = None
        reload_b = None
        try:
            await database.connect()
            await database.assert_healthy()
            await apply_migrations(database)
            repository = IngestionRepository(database)
            parser = DoclingParser(settings)
            storage = DocumentStorage(
                settings.resolved_storage_root, settings.max_upload_bytes
            )
            pipeline_a = IngestionPipeline(
                repository, settings=settings, storage=storage, parser=parser
            )
            pipeline_b = IngestionPipeline(
                repository, settings=settings, storage=storage, parser=parser
            )

            source_a = _write(tmp_path, "acme.md", _DOC_A)
            source_b = _write(tmp_path, "beta.md", _DOC_B)
            distractor_a = _write(tmp_path, "acme_notes.md", _DISTRACTOR_A)
            distractor_b = _write(tmp_path, "beta_notes.md", _DISTRACTOR_B)

            ingested_a = await pipeline_a.ingest_file(ws_a, source_a)
            ingested_b = await pipeline_b.ingest_file(ws_b, source_b)
            ingested_a_extra = await pipeline_a.ingest_file(ws_a, distractor_a)
            ingested_b_extra = await pipeline_b.ingest_file(ws_b, distractor_b)

            assert ingested_a.document.metadata["docling_used"] is True
            assert ingested_b.document.metadata["docling_used"] is True
            assert len(ingested_a.chunks) == 1
            assert len(ingested_b.chunks) == 1
            assert len(ingested_a_extra.chunks) == 1
            assert len(ingested_b_extra.chunks) == 1

            assert await service.index_document(ws_a, ingested_a.doc_id) == 1
            assert await service.index_document(ws_b, ingested_b.doc_id) == 1
            assert await service.index_document(ws_a, ingested_a_extra.doc_id) == 1
            assert await service.index_document(ws_b, ingested_b_extra.doc_id) == 1

            query_a = (
                "Where is Acme Technologies headquartered and which database "
                "does the Fraud Detection project use?"
            )
            query_b = (
                "Where is Beta Systems headquartered and which database "
                "does the Inventory Sync project use?"
            )
            vector = PGVectorRetriever(database, embedder, settings)
            bm25 = ParadeDBBM25Retriever(database, settings)
            vector_a = await vector.retrieve(query_a, top_k=20, workspace_id=ws_a)
            bm25_a = await bm25.search(query_a, top_k=20, workspace_id=ws_a)
            vector_b = await vector.retrieve(query_b, top_k=20, workspace_id=ws_b)
            bm25_b = await bm25.search(query_b, top_k=20, workspace_id=ws_b)
            assert len(vector_a) == 2
            assert len(bm25_a) == 2
            assert len(vector_b) == 2
            assert len(bm25_b) == 2
            assert vector_a[0].chunk_id == ingested_a.chunks[0].chunk_id
            assert bm25_a[0].chunk_id == ingested_a.chunks[0].chunk_id
            assert vector_b[0].chunk_id == ingested_b.chunks[0].chunk_id
            assert bm25_b[0].chunk_id == ingested_b.chunks[0].chunk_id
            fused_a = reciprocal_rank_fusion(vector_a, bm25_a, k=settings.rrf_k, top_k=20)
            fused_b = reciprocal_rank_fusion(vector_b, bm25_b, k=settings.rrf_k, top_k=20)
            assert len(fused_a) == 2
            assert len(fused_b) == 2

            graph_index_a = await graph_a.aindex_chunks(ingested_a.chunks)
            assert graph_index_a["indexed_documents"] == 1
            if _GRAPH_PAUSE > 0:
                await asyncio.sleep(_GRAPH_PAUSE)
            graph_index_b = await graph_b.aindex_chunks(ingested_b.chunks)
            assert graph_index_b["indexed_documents"] == 1

            seeds_a = ["Alice", "Acme", "Fraud Detection", "PostgreSQL", "Pune"]
            seeds_b = ["Bob", "Beta Systems", "Inventory Sync", "MySQL", "Mumbai"]
            result_a = await service.retrieve_with_graph(
                query_a,
                top_k=5,
                workspace_id=ws_a,
                entity_seeds=seeds_a,
                graph_engine=graph_a,
                max_hops=2,
            )
            result_b = await service.retrieve_with_graph(
                query_b,
                top_k=5,
                workspace_id=ws_b,
                entity_seeds=seeds_b,
                graph_engine=graph_b,
                max_hops=2,
            )

            assert result_a.graph_error is None
            assert result_b.graph_error is None
            assert result_a.evidence
            assert result_b.evidence
            text_a = _evidence_text(result_a)
            text_b = _evidence_text(result_b)
            assert "pune" in text_a and "postgresql" in text_a
            assert "mumbai" in text_b and "mysql" in text_b
            for forbidden in ("mumbai", "mysql", "bob", "beta systems"):
                assert forbidden not in text_a
            for forbidden in ("pune", "postgresql", "alice", "acme technologies"):
                assert forbidden not in text_b
            assert all(isinstance(item.cross_encoder_logit, float) for item in result_a.evidence)
            assert all(isinstance(item.cross_encoder_logit, float) for item in result_b.evidence)
            assert result_a.evidence[0].chunk_id == ingested_a.chunks[0].chunk_id
            assert result_b.evidence[0].chunk_id == ingested_b.chunks[0].chunk_id
            assert result_a.evidence[0].cross_encoder_logit > result_a.evidence[1].cross_encoder_logit
            assert result_b.evidence[0].cross_encoder_logit > result_b.evidence[1].cross_encoder_logit
            assert result_a.evidence[0].metadata["retrieval_sources"] == ["vector", "bm25"]
            assert result_b.evidence[0].metadata["retrieval_sources"] == ["vector", "bm25"]

            nodes_a = {node["entity_name"] for node in result_a.graph["nodes"]}
            nodes_b = {node["entity_name"] for node in result_b.graph["nodes"]}
            assert result_a.graph["edges"]
            assert result_b.graph["edges"]
            assert {"Acme Technologies", "Alice", "PostgreSQL"} <= nodes_a
            assert any("fraud detection" in node.lower() for node in nodes_a)
            assert {"Beta Systems", "Bob", "MySQL"} <= nodes_b
            assert any("inventory sync" in node.lower() for node in nodes_b)
            assert "Graph Link:" in result_a.graph_context
            assert "Graph Link:" in result_b.graph_context
            assert nodes_a.isdisjoint(nodes_b)
            assert not set(item.chunk_id for item in result_a.evidence).intersection(
                item.chunk_id for item in result_b.evidence
            )

            graph_a.close()
            graph_b.close()
            reload_a = LightRAGGraphEngine(
                settings=settings, workspace_id=ws_a, embedder=embedder
            )
            reload_b = LightRAGGraphEngine(
                settings=settings, workspace_id=ws_b, embedder=embedder
            )
            reload_result_a = await asyncio.to_thread(
                reload_a.traverse_entities, query_a, ["Alice", "Acme", "PostgreSQL"], 2
            )
            reload_result_b = await asyncio.to_thread(
                reload_b.traverse_entities, query_b, ["Bob", "Beta Systems", "MySQL"], 2
            )
            assert reload_result_a["workspace_id"] == ws_a
            assert reload_result_b["workspace_id"] == ws_b
            assert reload_result_a["total_nodes_found"] >= 3
            assert reload_result_b["total_nodes_found"] >= 3
            assert reload_result_a["edges"]
            assert reload_result_b["edges"]

            empty_result = await service.retrieve_with_graph(
                "unrelated query about lunar geology",
                top_k=5,
                workspace_id=empty_ws,
                entity_seeds=["Nothing"],
                graph_engine=empty_graph,
                max_hops=1,
            )
            assert empty_result.chunks == []
            assert empty_result.evidence == []
            assert empty_result.graph_error is None
            assert empty_result.graph["nodes"] == []
        finally:
            for engine in (graph_a, graph_b, empty_graph, reload_a, reload_b):
                if engine is not None:
                    engine.close()
            if database.is_connected:
                async with database.acquire() as connection:
                    await connection.execute(
                        "DELETE FROM workspaces WHERE workspace_id = ANY($1::text[])",
                        [ws_a, ws_b, empty_ws],
                    )
                await database.close()

    asyncio.run(scenario())
