"""Unit tests for Mid-Pipeline Evidence Sufficiency Gate (C_evidence)."""
import pytest
from core.schemas import ChunkEvidence, RetrievalTier
from core.config import SLAPolicyMode, DEFAULT_POLICY_PROFILES
from verifier.confidence_gate import (
    sigmoid,
    extract_query_entities,
    compute_entity_coverage,
    compute_evidence_confidence,
    EvidenceSufficiencyGate,
)


def test_sigmoid_properties():
    assert sigmoid(0.0) == 0.5
    assert sigmoid(100.0) == 1.0
    assert sigmoid(-100.0) == 0.0
    assert 0.70 < sigmoid(1.0) < 0.75


def test_entity_extraction_and_coverage():
    query = "Why was the EV-Pack-V2 line halted yesterday, and did VoltCore breach SLA Clause 8.2?"
    entities = extract_query_entities(query)
    assert any("ev-pack-v2" in e for e in entities)
    assert any("voltcore" in e for e in entities)

    context_full = "The EV-Pack-V2 line was halted yesterday. Supplier VoltCore breached Clause 8.2."
    cov_full = compute_entity_coverage(query, context_full)
    assert cov_full >= 0.80

    context_empty = "General safety instructions and lunch breaks."
    cov_empty = compute_entity_coverage(query, context_empty)
    assert cov_empty < 0.30


def test_composite_evidence_confidence_formula():
    policy = DEFAULT_POLICY_PROFILES[SLAPolicyMode.BALANCED]
    
    # Chunks with high rerank margin
    chunks = [
        ChunkEvidence(chunk_id="c1", doc_id="QA_SOP", content="Maximum defect tolerance percentage is 0.5% for lithium cells.", cross_encoder_logit=4.5),
        ChunkEvidence(chunk_id="c2", doc_id="QA_SOP", content="Lithium cell batch testing guidelines.", cross_encoder_logit=1.0),
        ChunkEvidence(chunk_id="c3", doc_id="QA_SOP", content="Battery packaging standards.", cross_encoder_logit=-1.2),
    ]

    # High Noul sufficiency (0.95), High reranker margin, High entity coverage
    res = compute_evidence_confidence(
        query="What is the maximum defect tolerance percentage for lithium cells?",
        chunks=chunks,
        p_jev_noul_sufficient=0.95,
        policy=policy,
        current_tier=RetrievalTier.TIER1_VECTOR,
    )

    assert res.c_evidence >= 0.65
    assert res.escalate_triggered is False
    assert res.target_tier is None


def test_confidence_gated_escalation_trigger():
    policy = DEFAULT_POLICY_PROFILES[SLAPolicyMode.BALANCED]
    
    # Missing information: low Noul sufficiency (0.34) and low margin
    chunks = [
        ChunkEvidence(chunk_id="c1", doc_id="VoltCore_SLA", content="Clause 8.2 delivery delays.", cross_encoder_logit=0.8),
        ChunkEvidence(chunk_id="c2", doc_id="VoltCore_SLA", content="General supplier terms.", cross_encoder_logit=0.6),
    ]

    res = compute_evidence_confidence(
        query="Why was EV-Pack-V2 line halted yesterday, and did VoltCore breach SLA?",
        chunks=chunks,
        p_jev_noul_sufficient=0.34,  # as in Blueprint Page 7
        policy=policy,
        current_tier=RetrievalTier.TIER2_HYBRID,
    )

    # In Balanced mode, tau_escalate is 0.65.
    assert res.c_evidence < 0.65
    assert res.escalate_triggered is True
    assert res.target_tier == RetrievalTier.TIER3_AGENTIC
