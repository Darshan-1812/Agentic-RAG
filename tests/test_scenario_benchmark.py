from core.schemas import RetrievalTier, ChunkEvidence, EvidenceConfidenceResult
from core.config import SLAPolicyMode, DEFAULT_POLICY_PROFILES
from router.pre_retrieval_router import PreRetrievalRouter
from verifier.confidence_gate import EvidenceSufficiencyGate, compute_evidence_confidence
from verifier.claim_verifier import ClaimVerifier
from verifier.abstention_engine import AbstentionEngine
from core.telemetry import calculate_telemetry


def test_scenario_query_1_simple_sop_lookup():
    """Query 1: 'What is the maximum defect tolerance percentage allowed for lithium cell batches?'

    Outcome: Tier 1 Fast Vector RAG, Noul sufficiency = 0.95 (>= 0.65). Synthesizes with Llama-3.1-8B.
    Citations: [QA_SOP, p.12]. Latency: ~0.68s, Cost: $0.00018 (Saved 97%).
    """
    router = PreRetrievalRouter()
    query = "What is the maximum defect tolerance percentage allowed for lithium cell batches?"
    stage1 = router.route(query, sla_policy=SLAPolicyMode.BALANCED)

    assert stage1.selected_tier == RetrievalTier.TIER1_VECTOR
    assert stage1.needs_live_mcp is False
    assert stage1.reasoning_depth <= 2.0

    # Simulated Tier 1 retrieved chunks
    chunks = [
        ChunkEvidence(chunk_id="c1", doc_id="QA_SOP, p.12", content="Section 4.2: Maximum defect tolerance percentage allowed for lithium cell batches is 0.5%.", cross_encoder_logit=4.2),
        ChunkEvidence(chunk_id="c2", doc_id="QA_SOP, p.11", content="Cell inspection protocols.", cross_encoder_logit=1.1),
        ChunkEvidence(chunk_id="c3", doc_id="QA_SOP, p.13", content="Packaging QA guidelines.", cross_encoder_logit=-0.5),
    ]

    policy = DEFAULT_POLICY_PROFILES[SLAPolicyMode.BALANCED]
    gate = EvidenceSufficiencyGate()
    conf = gate.evaluate(query, chunks, current_tier=stage1.selected_tier)

    assert conf.c_evidence >= policy.tau_escalate  # >= 0.65
    assert conf.escalate_triggered is False

    # Generation & Post-gen verification
    draft_answer = "The maximum defect tolerance percentage allowed for lithium cell batches is 0.5% [QA_SOP, p.12]."
    verifier = ClaimVerifier()
    v_res = verifier.verify_draft(draft_answer, {"QA_SOP, p.12": chunks[0].content}, policy)

    assert v_res.all_claims_verified is True
    assert v_res.is_abstention_triggered is False

    telemetry = calculate_telemetry(
        query_id="q1",
        workspace_id="apex",
        tier_used=stage1.selected_tier,
        jev_confidence=conf.c_evidence,
        latency_ms=680.0,
        actual_cost_usd=0.00018,
    )
    assert telemetry.cost_savings_pct >= 95.0


def test_scenario_query_2_mid_flight_escalation():
    """Query 2: 'Why was the EV-Pack-V2 line halted yesterday, and did the supplier breach their SLA penalty window?'

    Outcome: Choice = tier2_hybrid (0.61). Tier 2 retrieves SLA PDF chunks, but finds no data on 'yesterday's halt'.
    Jev mid-pipeline Noul returns 0.34 (< 0.65) -> Auto-Escalates to Tier 3!
    LangGraph queries LightRAG (VoltCore India) + Postgres ERP MCP (9-day delay on Batch #402), matches SLA Clause 8.2.
    """
    router = PreRetrievalRouter()
    query = "Why was the EV-Pack-V2 line halted yesterday, and did the supplier breach their SLA penalty window?"
    stage1 = router.route(query, sla_policy=SLAPolicyMode.BALANCED)

    assert stage1.selected_tier == RetrievalTier.TIER2_HYBRID

    # Tier 2 initially retrieves SLA doc only (lacks 'yesterday's halt' incident)
    tier2_initial_chunks = [
        ChunkEvidence(chunk_id="c1", doc_id="VoltCore_SLA, p.6", content="Clause 8.2: Any delay exceeding 7 days incurs 5% penalty per week.", cross_encoder_logit=1.8),
        ChunkEvidence(chunk_id="c2", doc_id="VoltCore_SLA, p.2", content="Standard supply terms for EV batteries.", cross_encoder_logit=0.4),
    ]

    policy = DEFAULT_POLICY_PROFILES[SLAPolicyMode.BALANCED]
    # Mid-pipeline check returns low Noul sufficiency (0.34) because halt details are missing
    conf_initial = compute_evidence_confidence(
        query=query,
        chunks=tier2_initial_chunks,
        p_jev_noul_sufficient=0.34,
        policy=policy,
        current_tier=RetrievalTier.TIER2_HYBRID,
    )

    # Escalation is triggered
    assert conf_initial.c_evidence < policy.tau_escalate  # < 0.65
    assert conf_initial.escalate_triggered is True
    assert conf_initial.target_tier == RetrievalTier.TIER3_AGENTIC

    # After Tier 3 LangGraph queries LightRAG and ERP MCP:
    tier3_escalated_chunks = tier2_initial_chunks + [
        ChunkEvidence(chunk_id="c3", doc_id="MCP-ERP #402", content="Batch #402 cell delivery from VoltCore India delayed by 9 days, causing EV-Pack-V2 line halt.", cross_encoder_logit=4.6),
    ]

    conf_escalated = compute_evidence_confidence(
        query=query,
        chunks=tier3_escalated_chunks,
        p_jev_noul_sufficient=0.92,
        policy=policy,
        current_tier=RetrievalTier.TIER3_AGENTIC,
    )

    assert conf_escalated.c_evidence >= 0.65
    assert conf_escalated.escalate_triggered is False


def test_scenario_query_4_trap_calibrated_abstention():
    """Query 4 (Unanswerable/Trap): 'Which supplier will win our Q3 2027 cobalt tender and at what price per ton?'

    Outcome: Jev Noul returns 0.11 -> Escalates to Tier 3 -> post-escalation Noul remains 0.14 (< 0.50 Abstention floor).
    Clean abstention refusing to guess. Hallucination: 0%.
    """
    router = PreRetrievalRouter()
    query = "Which supplier will win our Q3 2027 cobalt tender and at what price per ton?"
    stage1 = router.route(query, sla_policy=SLAPolicyMode.BALANCED)

    policy = DEFAULT_POLICY_PROFILES[SLAPolicyMode.BALANCED]

    # Post-retrieval evidence score remains below tau_abstain (0.50)
    conf_final = EvidenceConfidenceResult(
        c_evidence=0.14,
        p_jev_noul_sufficient=0.11,
        sigma_delta_rerank=0.10,
        cov_entities=0.15,
        escalate_triggered=True,
        target_tier=RetrievalTier.TIER3_AGENTIC,
        tau_escalate=0.65,
        tau_abstain=0.50,
        policy_mode=SLAPolicyMode.BALANCED,
    )

    abstention_engine = AbstentionEngine()
    assert abstention_engine.should_abstain(conf_final, policy) is True

    resp = abstention_engine.build_abstention_response(query, conf_final)
    assert resp.status == "abstained"
    assert resp.confidence_score == 0.14
    assert len(resp.missing_facts) > 0
    assert any("2027" in f for f in resp.missing_facts)
