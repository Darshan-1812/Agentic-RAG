"""Unit tests for raw-logit BGE cross-encoder reranking."""
from __future__ import annotations

import numpy as np
import pytest

from retrieval.config import RetrievalSettings
from retrieval.reranking import BgeReranker, RerankerError
from retrieval.retrieval_types import RetrievedChunk


class _FakeRerankerModel:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.calls = []

    def compute_score(self, pairs, **kwargs):
        self.calls.append((list(pairs), kwargs))
        return self.scores


def _chunk(chunk_id, content):
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        content=content,
        vector_rank=1,
        vector_distance=0.1,
    )


async def test_reranker_preserves_raw_logits_and_orders_by_score():
    model = _FakeRerankerModel([1.25, 9.5, -3.0])
    settings = RetrievalSettings(
        reranker_batch_size=2,
        reranker_max_length=512,
    )
    reranker = BgeReranker(settings, model=model)
    chunks = [
        _chunk("c1", "first passage"),
        _chunk("c2", "second passage"),
        _chunk("c3", "third passage"),
    ]

    results = await reranker.rerank("which passage?", chunks, top_k=2)
    assert [result.chunk_id for result in results] == ["c2", "c1"]
    assert results[0].cross_encoder_logit == pytest.approx(9.5)
    assert results[1].cross_encoder_logit == pytest.approx(1.25)
    assert [result.rerank_rank for result in results] == [1, 2]

    pairs, kwargs = model.calls[0]
    assert pairs[0] == ("which passage?", "first passage")
    assert kwargs == {
        "batch_size": 2,
        "max_length": 512,
        "normalize": False,
    }


async def test_reranker_does_not_apply_sigmoid():
    model = _FakeRerankerModel([10.0, -10.0])
    reranker = BgeReranker(RetrievalSettings(), model=model)
    results = await reranker.rerank(
        "query",
        [_chunk("c1", "positive"), _chunk("c2", "negative")],
        top_k=2,
    )
    assert results[0].cross_encoder_logit == pytest.approx(10.0)
    assert results[1].cross_encoder_logit == pytest.approx(-10.0)


async def test_reranker_empty_chunks_does_not_load_model():
    model = _FakeRerankerModel([])
    reranker = BgeReranker(RetrievalSettings(), model=model)
    assert await reranker.rerank("query", [], top_k=5) == []
    assert model.calls == []


async def test_reranker_rejects_invalid_scores_and_inputs():
    reranker = BgeReranker(
        RetrievalSettings(), model=_FakeRerankerModel([1.0, 2.0])
    )
    with pytest.raises(RerankerError):
        await reranker.rerank(
            "query",
            [_chunk("c1", "one"), _chunk("c2", "two"), _chunk("c3", "three")],
            top_k=2,
        )

    nan_reranker = BgeReranker(
        RetrievalSettings(), model=_FakeRerankerModel([np.nan])
    )
    with pytest.raises(RerankerError):
        await nan_reranker.rerank("query", [_chunk("c1", "one")], top_k=1)

    empty_reranker = BgeReranker(RetrievalSettings(), model=_FakeRerankerModel([]))
    with pytest.raises(RerankerError):
        await empty_reranker.rerank("   ", [_chunk("c1", "one")], top_k=1)
    with pytest.raises(RerankerError):
        await empty_reranker.rerank("query", [_chunk("c1", "one")], top_k=0)
