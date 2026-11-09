"""Phase 5 tests for hybrid retrieval combined with graph context."""
from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from retrieval.config import RetrievalSettings
from retrieval.lightrag_indexer import LightRAGGraphEngine
from retrieval.retrieval_types import RetrievedChunk
from retrieval.service import RetrievalInputError, RetrievalService


def _chunk(chunk_id: str, content: str | None = None, **kwargs):
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        content=content or f"content for {chunk_id}",
        **kwargs,
    )


class _FakeVectorRetriever:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def retrieve(self, query, top_k=5, workspace_id="default"):
        self.calls.append((query, top_k, workspace_id))
        return self.results[:top_k]


class _FakeBM25Retriever:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def search(self, query, top_k=5, workspace_id="default"):
        self.calls.append((query, top_k, workspace_id))
        return self.results[:top_k]


class _FakeReranker:
    async def rerank(self, query, chunks, top_k=5):
        scored = [
            replace(
                chunk,
                cross_encoder_logit=5.0 if chunk.chunk_id == "c2" else -5.0,
            )
            for chunk in chunks
        ]
        scored.sort(key=lambda chunk: (-chunk.cross_encoder_logit, chunk.chunk_id))
        return [
            replace(chunk, rerank_rank=rank)
            for rank, chunk in enumerate(scored[:top_k], start=1)
        ]


class _FakeGraphEngine:
    def __init__(self, *, workspace_id="tenant_a", fail=None, delay=0.0):
        self.workspace_id = workspace_id
        self.fail = fail
        self.delay = delay
        self.calls = []

    def traverse_entities(self, query, entity_seeds, max_hops=2):
        self.calls.append((query, list(entity_seeds), max_hops))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
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


class _FakeLightRAGClient:
    def __init__(self):
        self.initialized = 0
        self.finalized = 0

    async def initialize_storages(self):
        self.initialized += 1

    async def finalize_storages(self):
        self.finalized += 1

    async def get_graph_labels(self):
        return ["VoltCore", "EV-Pack-V2"]

    async def get_knowledge_graph(self, node_label, max_depth=3, max_nodes=None):
        nodes = [
            SimpleNamespace(
                id="VoltCore",
                labels=["VoltCore"],
                properties={"entity_name": "VoltCore", "entity_type": "Company"},
            ),
            SimpleNamespace(
                id="EV-Pack-V2",
                labels=["EV-Pack-V2"],
                properties={"entity_name": "EV-Pack-V2", "entity_type": "Product"},
            ),
        ]
        edges = [
            SimpleNamespace(
                source="VoltCore",
                target="EV-Pack-V2",
                properties={"description": "supplies cells"},
            )
        ]
        return SimpleNamespace(nodes=nodes, edges=edges, is_truncated=False)


def _service(*, graph_engine=None, timeout=1.0):
    settings = RetrievalSettings(
        retrieval_top_k=2,
        vector_candidate_depth=5,
        bm25_candidate_depth=5,
        hybrid_candidate_depth=5,
        rerank_candidate_depth=4,
        graph_traversal_timeout_seconds=timeout,
    )
    return RetrievalService(
        settings=settings,
        vector_retriever=_FakeVectorRetriever(
            [
                _chunk("c1", vector_rank=1, vector_distance=0.1),
                _chunk("c2", vector_rank=2, vector_distance=0.2),
            ]
        ),
        bm25_retriever=_FakeBM25Retriever(
            [
                _chunk("c2", bm25_score=5.0, bm25_rank=1),
                _chunk("c3", bm25_score=2.0, bm25_rank=2),
            ]
        ),
        reranker=_FakeReranker(),
        vector_store=object(),
        graph_engine=graph_engine,
    )


async def test_retrieve_with_graph_returns_hybrid_evidence_and_context():
    graph = _FakeGraphEngine()
    service = _service(graph_engine=graph)

    result = await service.retrieve_with_graph(
        "which supplier provides the battery pack",
        top_k=2,
        workspace_id="tenant_a",
        entity_seeds=["VoltCore", "EV-Pack-V2"],
        max_hops=2,
    )

    assert result.workspace_id == "tenant_a"
    assert result.graph_error is None
    assert [chunk.chunk_id for chunk in result.chunks] == ["c2", "c1"]
    assert [item.chunk_id for item in result.evidence] == ["c2", "c1"]
    assert result.evidence[0].cross_encoder_logit == pytest.approx(5.0)
    assert result.graph_context == "Graph Link: VoltCore -> [SUPPLIES] -> EV-Pack-V2"
    assert result.graph["total_nodes_found"] == 2
    assert graph.calls == [
        ("which supplier provides the battery pack", ["VoltCore", "EV-Pack-V2"], 2)
    ]


async def test_graph_failure_does_not_discard_hybrid_evidence():
    graph = _FakeGraphEngine(fail=RuntimeError("graph backend unavailable"))
    service = _service(graph_engine=graph)

    result = await service.retrieve_with_graph(
        "defect tolerance",
        top_k=2,
        workspace_id="tenant_a",
        entity_seeds=["VoltCore"],
    )

    assert [chunk.chunk_id for chunk in result.chunks] == ["c2", "c1"]
    assert result.graph_error == "RuntimeError: graph backend unavailable"
    assert result.graph["nodes"] == []
    assert result.graph_context == ""


async def test_graph_timeout_is_reported_and_non_fatal():
    graph = _FakeGraphEngine(delay=0.25)
    service = _service(graph_engine=graph, timeout=0.05)

    result = await service.retrieve_with_graph(
        "defect tolerance",
        top_k=2,
        workspace_id="tenant_a",
        entity_seeds=["VoltCore"],
    )

    assert [chunk.chunk_id for chunk in result.chunks] == ["c2", "c1"]
    assert result.graph_error is not None
    assert "timed out" in result.graph_error


async def test_graph_workspace_mismatch_is_rejected():
    graph = _FakeGraphEngine(workspace_id="tenant_b")
    service = _service(graph_engine=graph)

    result = await service.retrieve_with_graph(
        "defect tolerance",
        top_k=2,
        workspace_id="tenant_a",
        entity_seeds=["VoltCore"],
    )

    assert [chunk.chunk_id for chunk in result.chunks] == ["c2", "c1"]
    assert result.graph_error == "graph traversal returned a different workspace"
    assert result.graph["nodes"] == []


async def test_empty_seeds_skip_graph_engine():
    graph = _FakeGraphEngine()
    service = _service(graph_engine=graph)

    result = await service.retrieve_with_graph(
        "defect tolerance",
        top_k=2,
        workspace_id="tenant_a",
        entity_seeds=[],
    )

    assert [chunk.chunk_id for chunk in result.chunks] == ["c2", "c1"]
    assert result.graph_error is None
    assert result.graph["nodes"] == []
    assert graph.calls == []


async def test_invalid_max_hops_and_blank_query():
    graph = _FakeGraphEngine()
    service = _service(graph_engine=graph)

    with pytest.raises(RetrievalInputError):
        await service.retrieve_with_graph(
            "query", workspace_id="tenant_a", entity_seeds=["VoltCore"], max_hops=-1
        )

    result = await service.retrieve_with_graph(
        "   ", workspace_id="tenant_a", entity_seeds=["VoltCore"]
    )
    assert result.chunks == []
    assert result.graph["nodes"] == []
    assert graph.calls == []


async def test_service_uses_real_lightrag_adapter_with_injected_client(tmp_path):
    fake_client = _FakeLightRAGClient()
    engine = LightRAGGraphEngine(
        settings=RetrievalSettings(graph_storage_root=tmp_path),
        workspace_id="tenant_a",
        lightrag_factory=lambda _: fake_client,
    )
    service = _service(graph_engine=engine)
    try:
        result = await service.retrieve_with_graph(
            "which supplier provides the battery pack",
            top_k=2,
            workspace_id="tenant_a",
            entity_seeds=["VoltCore"],
            max_hops=1,
        )
    finally:
        engine.close()

    assert result.graph_error is None
    assert [chunk.chunk_id for chunk in result.chunks] == ["c2", "c1"]
    assert result.graph["workspace_id"] == "tenant_a"
    assert [node["entity_name"] for node in result.graph["nodes"]] == [
        "VoltCore",
        "EV-Pack-V2",
    ]
    assert "Graph Link: VoltCore -> [RELATED_TO] -> EV-Pack-V2" in result.graph_context
    assert fake_client.initialized == 1
    assert fake_client.finalized == 1
