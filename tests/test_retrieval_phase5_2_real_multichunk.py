"""Opt-in Phase 5.2 real multi-chunk GraphRAG verification.

Set ``RETRIEVAL_PHASE5_2_REAL_TESTS=1``, ``RETRIEVAL_TEST_DATABASE_URL``, and
provide ``retrieval/.env`` with a real Groq key. The test uses real Docling,
BGE-M3, PostgreSQL, and LightRAG extraction through Groq. It verifies four
chunks per document, later-chunk facts, provenance, idempotency, GraphML
reload, and workspace isolation.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from retrieval.config import RetrievalSettings
from retrieval.db.connection import Database
from retrieval.db.migrate import apply_migrations
from retrieval.db.repository import IngestionRepository
from retrieval.embeddings.bge_m3 import BgeM3Embedder
from retrieval.ingestion.docling_parser import DoclingParser
from retrieval.ingestion.pipeline import IngestionPipeline
from retrieval.ingestion.storage import DocumentStorage
from retrieval.lightrag_indexer import LightRAGGraphEngine

_ENV_FILE = Path(__file__).resolve().parents[1] / "retrieval" / ".env"
_TEST_DSN = os.getenv("RETRIEVAL_TEST_DATABASE_URL")
_REAL_ENABLED = os.getenv("RETRIEVAL_PHASE5_2_REAL_TESTS") == "1"
_GRAPH_PAUSE = float(os.getenv("RETRIEVAL_PHASE5_2_GRAPH_PAUSE_SECONDS", "65"))

pytestmark = pytest.mark.skipif(
    not (_REAL_ENABLED and _TEST_DSN and _ENV_FILE.exists()),
    reason=(
        "set RETRIEVAL_PHASE5_2_REAL_TESTS=1, RETRIEVAL_TEST_DATABASE_URL, "
        "and retrieval/.env to run the real Phase 5.2 multi-chunk test"
    ),
)

_DOC_A = """# Acme Fraud Platform

Alice works at Acme Technologies. Acme Technologies is headquartered in Pune, India. Alice is the engineering lead for the fraud platform.

## Fraud Detection Project

Alice leads the Fraud Detection project. The project is part of Acme Technologies and focuses on detecting suspicious card transactions.

## PostgreSQL Storage

The Fraud Detection project uses PostgreSQL as its primary database. PostgreSQL stores transaction records, fraud signals, and investigation notes.

## Risk Engineering Team

The Risk Engineering team maintains the Fraud Detection project. The team reports to Alice and works with the transaction review group.
"""

_DOC_B = """# Beta Inventory Platform

Bob works at Beta Systems. Beta Systems is headquartered in Mumbai, India. Bob is the product lead for the inventory platform.

## Inventory Sync Project

Bob leads the Inventory Sync project. The project is part of Beta Systems and focuses on synchronizing warehouse inventory.

## MySQL Storage

The Inventory Sync project uses MySQL as its primary database. MySQL stores inventory snapshots and warehouse events.

## Logistics Operations Team

The Logistics Operations team maintains the Inventory Sync project. The team reports to Bob and works with the warehouse operations group.
"""


def _write(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "controlled.md"
    path.write_text(text, encoding="utf-8")
    return path


def _node_names(result) -> set[str]:
    return {node["entity_name"].lower() for node in result["nodes"]}


def _edge_keys(result) -> set[tuple[str, str, str]]:
    return {
        (
            edge["source_id"].lower(),
            edge["target_id"].lower(),
            edge["relation_type"].lower(),
        )
        for edge in result["edges"]
    }


async def _index_with_retries(
    engine: LightRAGGraphEngine,
    chunks,
    *,
    attempts: int = 5,
) -> tuple[dict, list[dict]]:
    results: list[dict] = []
    result = await engine.aindex_chunks(chunks)
    results.append(result)
    for _ in range(1, attempts):
        if result["status"] == "completed" and not result["failed_chunks"]:
            break
        if _GRAPH_PAUSE > 0:
            await asyncio.sleep(_GRAPH_PAUSE)
        result = await engine.aindex_chunks(chunks)
        results.append(result)
    return result, results


def test_phase5_2_real_multichunk_graphrag_e2e(tmp_path: Path) -> None:
    async def scenario() -> None:
        base = RetrievalSettings.from_env(env_file=str(_ENV_FILE))
        settings = replace(
            base,
            database_url=_TEST_DSN,
            storage_root=tmp_path / "storage",
            graph_storage_root=tmp_path / "graph",
            chunk_target_chars=500,
            chunk_max_chars=800,
            chunk_overlap_chars=0,
            graph_llm_max_async=1,
            graph_traversal_timeout_seconds=180.0,
        )
        ws_a = f"p52a_{uuid.uuid4().hex[:10]}"
        ws_b = f"p52b_{uuid.uuid4().hex[:10]}"
        database = Database(settings)
        embedder = BgeM3Embedder(settings)
        graph_a = LightRAGGraphEngine(settings=settings, workspace_id=ws_a, embedder=embedder)
        graph_b = LightRAGGraphEngine(settings=settings, workspace_id=ws_b, embedder=embedder)
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

            source_a = _write(tmp_path / "a", _DOC_A)
            source_b = _write(tmp_path / "b", _DOC_B)
            ingested_a = await pipeline_a.ingest_file(ws_a, source_a)
            ingested_b = await pipeline_b.ingest_file(ws_b, source_b)
            ingested_a_again = await pipeline_a.ingest_file(ws_a, source_a)

            assert ingested_a.document.metadata["docling_used"] is True
            assert ingested_b.document.metadata["docling_used"] is True
            assert ingested_a_again.duplicate is True
            assert ingested_a_again.doc_id == ingested_a.doc_id
            assert [chunk.chunk_id for chunk in ingested_a_again.chunks] == [
                chunk.chunk_id for chunk in ingested_a.chunks
            ]
            assert len(ingested_a.chunks) == 4
            assert len(ingested_b.chunks) == 4
            assert ingested_a.doc_id != ingested_b.doc_id
            assert all(
                chunk.chunk_id not in {item.chunk_id for item in ingested_b.chunks}
                for chunk in ingested_a.chunks
            )

            first_a, attempts_a = await _index_with_retries(
                graph_a, ingested_a.chunks
            )
            assert first_a["status"] == "completed"
            assert first_a["source_documents"] == 1
            assert len(first_a["processed_chunks"]) == 4
            assert first_a["failed_chunks"] == []
            assert sum(result["submitted_chunks"] for result in attempts_a) >= 4
            assert len(set(mapping["source_path"] for mapping in first_a["chunk_mappings"])) == 4
            source_prefix_a = f"{ingested_a.document.doc_title}::chunk::"
            assert all(
                mapping["source_path"].startswith(source_prefix_a)
                for mapping in first_a["chunk_mappings"]
            )

            query_a = (
                "Who reports to Alice, which database stores fraud signals, "
                "and where is Acme Technologies headquartered?"
            )
            seeds_a = ["Risk Engineering", "Alice", "PostgreSQL", "Fraud Detection"]
            before_a = await asyncio.to_thread(
                graph_a.traverse_entities, query_a, seeds_a, 2
            )
            assert before_a["workspace_id"] == ws_a
            assert before_a["total_nodes_found"] >= 4
            names_before_a = _node_names(before_a)
            edges_before_a = _edge_keys(before_a)
            assert any("risk engineering" in name for name in names_before_a)
            assert "alice" in names_before_a
            assert any("postgresql" in name for name in names_before_a)
            assert any("fraud detection" in name for name in names_before_a)
            assert "risk engineering" in before_a["context_summary"].lower()
            assert "postgresql" in before_a["context_summary"].lower()

            later_source = first_a["chunk_mappings"][3]["source_path"]
            later_edges = [
                edge
                for edge in before_a["edges"]
                if "risk engineering" in edge["source_id"].lower()
                or "risk engineering" in edge["target_id"].lower()
            ]
            assert later_edges
            assert any(
                later_source in (edge.get("provenance_doc") or "")
                for edge in later_edges
            )

            second_a = await graph_a.aindex_chunks(ingested_a.chunks)
            assert second_a["status"] == "completed"
            assert second_a["indexed_documents"] == 0
            assert second_a["submitted_chunks"] == 0
            assert second_a["skipped_chunks"] == [
                chunk.chunk_id for chunk in ingested_a.chunks
            ]
            after_a = await asyncio.to_thread(
                graph_a.traverse_entities, query_a, seeds_a, 2
            )
            assert _node_names(after_a) == names_before_a
            assert _edge_keys(after_a) == edges_before_a

            if _GRAPH_PAUSE > 0:
                await asyncio.sleep(_GRAPH_PAUSE)
            first_b, attempts_b = await _index_with_retries(
                graph_b, ingested_b.chunks
            )
            assert first_b["status"] == "completed"
            assert first_b["source_documents"] == 1
            assert len(first_b["processed_chunks"]) == 4
            assert first_b["failed_chunks"] == []
            assert sum(result["submitted_chunks"] for result in attempts_b) >= 4
            source_prefix_b = f"{ingested_b.document.doc_title}::chunk::"
            assert all(
                mapping["source_path"].startswith(source_prefix_b)
                for mapping in first_b["chunk_mappings"]
            )

            query_b = (
                "Who reports to Bob, which database stores inventory snapshots, "
                "and where is Beta Systems headquartered?"
            )
            seeds_b = [
                "Logistics Operations",
                "Bob",
                "MySQL",
                "Inventory Sync",
            ]
            result_b = await asyncio.to_thread(
                graph_b.traverse_entities, query_b, seeds_b, 2
            )
            assert result_b["workspace_id"] == ws_b
            assert result_b["total_nodes_found"] >= 4
            names_b = _node_names(result_b)
            assert any("logistics operations" in name for name in names_b)
            assert "bob" in names_b
            assert any("mysql" in name for name in names_b)
            assert any("inventory sync" in name for name in names_b)
            assert "logistics operations" in result_b["context_summary"].lower()
            assert "mysql" in result_b["context_summary"].lower()
            assert names_before_a.isdisjoint(names_b)
            for forbidden in ("beta", "mysql", "bob", "mumbai"):
                assert forbidden not in before_a["context_summary"].lower()
            for forbidden in ("acme", "postgresql", "alice", "pune"):
                assert forbidden not in result_b["context_summary"].lower()

            graph_file_a = graph_a.graph_file_path
            graph_file_b = graph_b.graph_file_path
            assert graph_file_a.is_file()
            assert graph_file_b.is_file()
            assert graph_file_a != graph_file_b
            graph_a.close()
            graph_b.close()
            reload_a = LightRAGGraphEngine(
                settings=settings, workspace_id=ws_a, embedder=embedder
            )
            reload_b = LightRAGGraphEngine(
                settings=settings, workspace_id=ws_b, embedder=embedder
            )
            reload_result_a = await asyncio.to_thread(
                reload_a.traverse_entities, query_a, seeds_a, 2
            )
            reload_result_b = await asyncio.to_thread(
                reload_b.traverse_entities, query_b, seeds_b, 2
            )
            assert _node_names(reload_result_a) == names_before_a
            assert _node_names(reload_result_b) == names_b
            assert reload_result_a["edges"]
            assert reload_result_b["edges"]
        finally:
            for engine in (graph_a, graph_b, reload_a, reload_b):
                if engine is not None:
                    engine.close()
            if database.is_connected:
                async with database.acquire() as connection:
                    await connection.execute(
                        "DELETE FROM workspaces WHERE workspace_id = ANY($1::text[])",
                        [ws_a, ws_b],
                    )
                await database.close()

    asyncio.run(scenario())
