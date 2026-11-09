"""Unit tests for RRF fusion and the unified hybrid retrieval service."""
from __future__ import annotations

from dataclasses import replace

import pytest

from retrieval.config import RetrievalSettings
from retrieval.hybrid import FusionError, reciprocal_rank_fusion
from retrieval.retrieval_types import MissingCrossEncoderLogit, RetrievedChunk
from retrieval.service import RetrievalInputError, RetrievalService


def _chunk(chunk_id, content=None, **kwargs):
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
    def __init__(self, logits):
        self.logits = dict(logits)
        self.calls = []

    async def rerank(self, query, chunks, top_k=5):
        self.calls.append((query, [chunk.chunk_id for chunk in chunks], top_k))
        scored = [
            replace(
                chunk,
                cross_encoder_logit=float(self.logits.get(chunk.chunk_id, 0.0)),
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


def test_rrf_fuses_duplicate_chunks_and_preserves_provenance():
    c1 = _chunk("c1", vector_rank=1, vector_distance=0.1, metadata={"vector": True})
    c2 = _chunk("c2", vector_rank=2, vector_distance=0.2, metadata={"vector": True})
    c3 = _chunk("c3", bm25_score=2.0, bm25_rank=2, metadata={"bm25": True})
    bm25_c2 = _chunk("c2", bm25_score=5.0, bm25_rank=1, metadata={"bm25": True})

    fused = reciprocal_rank_fusion([c1, c2], [bm25_c2, c3], k=60, top_k=3)
    assert [chunk.chunk_id for chunk in fused] == ["c2", "c1", "c3"]
    assert fused[0].retrieval_sources == ("vector", "bm25")
    assert fused[0].metadata == {"vector": True, "bm25": True}
    assert fused[0].vector_distance == pytest.approx(0.2)
    assert fused[0].bm25_score == pytest.approx(5.0)
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].rrf_score == pytest.approx(1 / 61)
    assert fused[2].rrf_score == pytest.approx(1 / 62)
    assert [chunk.rrf_rank for chunk in fused] == [1, 2, 3]


def test_rrf_rejects_invalid_inputs():
    chunk = _chunk("c1")
    with pytest.raises(FusionError):
        reciprocal_rank_fusion([chunk], [], k=0)
    with pytest.raises(FusionError):
        reciprocal_rank_fusion([chunk], [], top_k=0)
    with pytest.raises(FusionError):
        reciprocal_rank_fusion([chunk], [], vector_weight=0, bm25_weight=0)
    with pytest.raises(FusionError):
        reciprocal_rank_fusion([object()], [], k=60)


def test_retrieved_chunk_requires_real_reranker_logit_for_evidence():
    chunk = _chunk(
        "c1",
        vector_rank=1,
        vector_distance=0.1,
        retrieval_sources=("vector",),
    )
    with pytest.raises(MissingCrossEncoderLogit):
        chunk.to_chunk_evidence()

    evidence = replace(chunk, cross_encoder_logit=3.25).to_chunk_evidence()
    assert evidence.cross_encoder_logit == pytest.approx(3.25)
    assert evidence.metadata["retrieval_vector_rank"] == 1
    assert evidence.metadata["retrieval_sources"] == ["vector"]


async def test_service_runs_vector_bm25_rrf_and_reranker():
    vector = _FakeVectorRetriever(
        [
            _chunk("c1", vector_rank=1, vector_distance=0.1),
            _chunk("c2", vector_rank=2, vector_distance=0.2),
            _chunk("c3", vector_rank=3, vector_distance=0.3),
        ]
    )
    bm25 = _FakeBM25Retriever(
        [
            _chunk("c2", bm25_score=5.0, bm25_rank=1),
            _chunk("c4", bm25_score=2.0, bm25_rank=2),
        ]
    )
    reranker = _FakeReranker({"c1": 1.0, "c2": 4.5, "c3": -1.0, "c4": 0.2})
    settings = RetrievalSettings(
        retrieval_top_k=2,
        vector_candidate_depth=5,
        bm25_candidate_depth=5,
        hybrid_candidate_depth=5,
        rerank_candidate_depth=4,
    )
    service = RetrievalService(
        settings=settings,
        vector_retriever=vector,
        bm25_retriever=bm25,
        reranker=reranker,
        vector_store=object(),
    )

    evidence = await service.retrieve(
        "defect tolerance", top_k=2, workspace_id="tenant_a"
    )
    assert [item.chunk_id for item in evidence] == ["c2", "c1"]
    assert evidence[0].cross_encoder_logit == pytest.approx(4.5)
    assert evidence[0].metadata["retrieval_sources"] == ["vector", "bm25"]
    assert evidence[0].metadata["retrieval_cross_encoder_logit"] == pytest.approx(4.5)
    assert vector.calls == [("defect tolerance", 5, "tenant_a")]
    assert bm25.calls == [("defect tolerance", 5, "tenant_a")]
    assert reranker.calls[0][0] == "defect tolerance"
    assert reranker.calls[0][2] == 2
    assert len(reranker.calls[0][1]) == 4


async def test_service_empty_query_and_invalid_top_k():
    vector = _FakeVectorRetriever([])
    bm25 = _FakeBM25Retriever([])
    reranker = _FakeReranker({})
    service = RetrievalService(
        vector_retriever=vector,
        bm25_retriever=bm25,
        reranker=reranker,
        vector_store=object(),
    )

    assert await service.retrieve("   ", workspace_id="tenant_a") == []
    assert vector.calls == []
    assert bm25.calls == []
    assert reranker.calls == []

    with pytest.raises(RetrievalInputError):
        await service.retrieve("query", top_k=0, workspace_id="tenant_a")
