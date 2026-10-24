"""Unit tests for Stage 6 Post-Generation Claim Verifier & Noul Entailment."""
import pytest
from core.config import DEFAULT_POLICY_PROFILES, SLAPolicyMode
from verifier.claim_verifier import (
    split_into_sentences,
    extract_inline_citations,
    ClaimVerifier,
)


def test_sentence_splitting_and_citations():
    text = "The maximum defect tolerance is 0.5% [QA_SOP, p.12]. Delivery was delayed by 9 days [MCP-ERP #402]. VoltCore breached Clause 8.2 [VoltCore_SLA, p.6]."
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    citations = extract_inline_citations(sentences[0])
    assert citations == ["QA_SOP, p.12"]


def test_verify_draft_with_grounded_content():
    verifier = ClaimVerifier()
    policy = DEFAULT_POLICY_PROFILES[SLAPolicyMode.BALANCED]

    draft = "The maximum defect tolerance percentage allowed for lithium cell batches is 0.5% [QA_SOP, p.12]."
    sources = {
        "QA_SOP, p.12": "Section 4.2: The maximum defect tolerance percentage allowed for lithium cell batches is strictly capped at 0.5%."
    }

    res = verifier.verify_draft(draft, sources, policy)
    assert res.all_claims_verified is True
    assert res.stripped_claims_count == 0
    assert res.mean_noul_score >= 0.75
    assert res.is_abstention_triggered is False
    assert len(res.claims) == 1
    assert res.claims[0].is_entailed is True


def test_verify_draft_with_hallucinated_sentence_stripped():
    verifier = ClaimVerifier()
    policy = DEFAULT_POLICY_PROFILES[SLAPolicyMode.BALANCED]

    # Two sentences: first is grounded, second is a hallucinated speculation
    draft = (
        "The defect tolerance is 0.5% [QA_SOP, p.12]. "
        "VoltCore announced yesterday that all future contracts will give 50% discounts."
    )
    sources = {
        "QA_SOP, p.12": "Defect tolerance is strictly 0.5% for all cell batches."
    }

    res = verifier.verify_draft(draft, sources, policy)
    # The second sentence is unsupported by the sources context
    assert res.all_claims_verified is False
    assert res.stripped_claims_count >= 1
    assert "50% discounts" not in res.verified_text
