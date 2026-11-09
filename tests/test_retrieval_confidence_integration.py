"""Compatibility test between Phase 3 evidence and Member 1 confidence gate."""
from __future__ import annotations

from dataclasses import replace

import pytest

from core.config import DEFAULT_POLICY_PROFILES, SLAPolicyMode
from core.schemas import RetrievalTier
from retrieval.retrieval_types import RetrievedChunk
from verifier.confidence_gate import compute_evidence_confidence, sigmoid


def _evidence(chunk_id, content, logit, rank):
    return replace(
        RetrievedChunk(
            chunk_id=chunk_id,
            doc_id="doc-1",
            content=content,
            vector_rank=rank,
            vector_distance=0.1 * rank,
        ),
        cross_encoder_logit=logit,
        rerank_rank=rank,
    ).to_chunk_evidence()


def test_confidence_gate_consumes_raw_reranker_logits():
    chunks = [
        _evidence(
            "c1",
            "EV-Pack-V2 has a defect tolerance of 0.5 percent.",
            4.5,
            1,
        ),
        _evidence(
            "c2",
            "EV-Pack-V2 inspection records are retained for five years.",
            1.0,
            2,
        ),
        _evidence(
            "c3",
            "The cafeteria closes at 18:00.",
            -2.0,
            3,
        ),
    ]
    policy = DEFAULT_POLICY_PROFILES[SLAPolicyMode.BALANCED]
    result = compute_evidence_confidence(
        query="What is the EV-Pack-V2 defect tolerance?",
        chunks=chunks,
        p_jev_noul_sufficient=0.9,
        policy=policy,
        current_tier=RetrievalTier.TIER2_HYBRID,
    )

    expected_delta = 4.5 - (-2.0)
    assert result.sigma_delta_rerank == pytest.approx(round(sigmoid(expected_delta), 4))
    assert result.sigma_delta_rerank > 0.99
    assert result.c_evidence > 0.9
    assert result.cov_entities == pytest.approx(1.0)
    assert result.escalate_triggered is False
    assert chunks[0].cross_encoder_logit == pytest.approx(4.5)
    assert chunks[0].metadata["retrieval_cross_encoder_logit"] == pytest.approx(4.5)

def test_confidence_gate_uses_last_available_candidate_when_fewer_than_five():
    chunks = [
        _evidence("c1", "EV-Pack-V2 defect tolerance is 0.5 percent.", 4.0, 1),
        _evidence("c2", "The cafeteria closes at 18:00.", 1.0, 2),
    ]
    policy = DEFAULT_POLICY_PROFILES[SLAPolicyMode.BALANCED]
    result = compute_evidence_confidence(
        query="What is the EV-Pack-V2 defect tolerance?",
        chunks=list(reversed(chunks)),
        p_jev_noul_sufficient=0.9,
        policy=policy,
        current_tier=RetrievalTier.TIER2_HYBRID,
    )

    expected_delta = 4.0 - 1.0
    assert result.sigma_delta_rerank == pytest.approx(round(sigmoid(expected_delta), 4))
    assert result.sigma_delta_rerank < 0.99
