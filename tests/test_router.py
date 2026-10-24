"""Unit tests for Member 1 Pre-Retrieval System 1 Router & Jev Client."""
import pytest
from core.schemas import RetrievalTier
from core.config import SLAPolicyMode
from router.jev_client import JevClient, JevChoice, JevScore, JevNoul
from router.policy_manager import PolicyManager
from router.pre_retrieval_router import PreRetrievalRouter
from router.fallback_router import FallbackRouter


def test_jev_client_primitives_parallel():
    client = JevClient()
    prims = {
        "tier": JevChoice(options=["tier1_vector", "tier2_hybrid", "tier3_agentic"]),
        "score": JevScore(rubric="1 to 5 complexity"),
        "noul": JevNoul(question="Does answering require live database?"),
    }
    res = client.evaluate_parallel(
        query="What is the maximum defect tolerance percentage allowed for lithium cell batches?",
        primitives=prims
    )
    assert "tier" in res
    assert res["tier"].value == "tier1_vector"
    assert res["tier"].calibrated_probability >= 0.90
    assert res["score"].value <= 2.0
    assert res["noul"].calibrated_probability < 0.10


def test_pre_retrieval_router_query_1_simple():
    router = PreRetrievalRouter()
    query = "What is the maximum defect tolerance percentage allowed for lithium cell batches?"
    decision = router.route(query, sla_policy=SLAPolicyMode.BALANCED)
    
    assert decision.selected_tier == RetrievalTier.TIER1_VECTOR
    assert decision.needs_live_mcp is False
    assert decision.p_needs_live_mcp < 0.10
    assert decision.reasoning_depth <= 2.0
    assert decision.latency_ms > 0
    assert decision.estimated_cost_usd > 0


def test_pre_retrieval_router_query_3_complex_multihop():
    router = PreRetrievalRouter()
    query = "Which open GitHub firmware issues affect projects using VoltCore's delayed batches, and who is the lead engineer?"
    decision = router.route(query, sla_policy=SLAPolicyMode.BALANCED)
    
    assert decision.selected_tier == RetrievalTier.TIER3_AGENTIC
    assert decision.needs_live_mcp is True
    assert decision.p_needs_live_mcp > 0.90
    assert decision.reasoning_depth >= 4.0


def test_policy_manager_adjustments():
    pm = PolicyManager()
    balanced = pm.get_policy(override_mode=SLAPolicyMode.BALANCED)
    economy = pm.get_policy(override_mode=SLAPolicyMode.ECONOMY)
    strict = pm.get_policy(override_mode=SLAPolicyMode.STRICT)

    # Economy policy demotes borderline tier3 with low live mcp requirement
    adjusted = pm.adjust_tier_for_policy(
        base_tier=RetrievalTier.TIER3_AGENTIC,
        reasoning_score=4.0,
        p_needs_mcp=0.40,
        policy=economy
    )
    assert adjusted == RetrievalTier.TIER2_HYBRID

    # Strict policy escalates simple lookup with minor complexity
    strict_adjusted = pm.adjust_tier_for_policy(
        base_tier=RetrievalTier.TIER1_VECTOR,
        reasoning_score=2.5,
        p_needs_mcp=0.25,
        policy=strict
    )
    assert strict_adjusted == RetrievalTier.TIER2_HYBRID


def test_fallback_router():
    fallback = FallbackRouter()
    simple_dec = fallback.route("What is our late delivery penalty?", SLAPolicyMode.BALANCED)
    assert simple_dec.selected_tier == RetrievalTier.TIER1_VECTOR

    complex_dec = fallback.route("Why was batch #402 delayed yesterday in PostgreSQL ERP and which GitHub issue tracks it?", SLAPolicyMode.BALANCED)
    assert complex_dec.selected_tier == RetrievalTier.TIER3_AGENTIC
    assert complex_dec.needs_live_mcp is True
