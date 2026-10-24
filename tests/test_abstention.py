"""Unit tests for Stage 5 Calibrated Abstention Engine."""
import pytest
from core.schemas import EvidenceConfidenceResult, RetrievalTier, ChunkEvidence
from core.config import SLAPolicyMode, DEFAULT_POLICY_PROFILES
from verifier.abstention_engine import AbstentionEngine


def test_abstention_decision():
    engine = AbstentionEngine()
    policy = DEFAULT_POLICY_PROFILES[SLAPolicyMode.BALANCED]

    # Confidence below tau_abstain (0.50)
    low_conf = EvidenceConfidenceResult(
        c_evidence=0.14,  # as in Blueprint Page 7 for Query 4
        p_jev_noul_sufficient=0.11,
        sigma_delta_rerank=0.20,
        cov_entities=0.10,
        escalate_triggered=True,
        target_tier=RetrievalTier.TIER3_AGENTIC,
        tau_escalate=0.65,
        tau_abstain=0.50,
        policy_mode=SLAPolicyMode.BALANCED,
    )

    assert engine.should_abstain(low_conf, policy) is True

    # High confidence above tau_abstain
    high_conf = EvidenceConfidenceResult(
        c_evidence=0.88,
        p_jev_noul_sufficient=0.92,
        sigma_delta_rerank=0.85,
        cov_entities=0.85,
        escalate_triggered=False,
        target_tier=None,
        tau_escalate=0.65,
        tau_abstain=0.50,
        policy_mode=SLAPolicyMode.BALANCED,
    )

    assert engine.should_abstain(high_conf, policy) is False


def test_build_structured_abstention_response():
    engine = AbstentionEngine()
    policy = DEFAULT_POLICY_PROFILES[SLAPolicyMode.BALANCED]

    low_conf = EvidenceConfidenceResult(
        c_evidence=0.14,
        p_jev_noul_sufficient=0.11,
        sigma_delta_rerank=0.20,
        cov_entities=0.10,
        escalate_triggered=True,
        target_tier=RetrievalTier.TIER3_AGENTIC,
        tau_escalate=0.65,
        tau_abstain=0.50,
        policy_mode=SLAPolicyMode.BALANCED,
    )

    query = "Which supplier will win our Q3 2027 cobalt tender and at what price per ton?"
    chunks = [
        ChunkEvidence(chunk_id="c1", doc_id="Tender_2025", content="Past tender archives for 2025 cobalt shipments.", cross_encoder_logit=0.1)
    ]

    resp = engine.build_abstention_response(query=query, confidence_result=low_conf, chunks=chunks)

    assert resp.status == "abstained"
    assert resp.confidence_score == 0.14
    assert resp.threshold_required == 0.50
    assert len(resp.missing_facts) > 0
    assert any("2027" in f for f in resp.missing_facts)
    assert len(resp.missing_documents) > 0
    assert "ACR-RAG halted generation" in resp.explanation
