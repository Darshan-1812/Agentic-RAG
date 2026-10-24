"""Benchmark test queries and ground-truth expectations from Blueprint Section 6."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from core.schemas import RetrievalTier


class BenchmarkScenario(BaseModel):
    id: str
    name: str
    query: str
    expected_stage1_tier: RetrievalTier
    expected_stage1_mcp_need: bool
    expected_reasoning_depth_min: float
    expected_reasoning_depth_max: float
    requires_mid_pipeline_escalation: bool
    expected_final_outcome: str  # "answered", "escalated_and_answered", "abstained"
    expected_citations: List[str]


BENCHMARK_SCENARIOS: List[BenchmarkScenario] = [
    BenchmarkScenario(
        id="query_1_simple",
        name="Query 1 (Simple Fact Lookup)",
        query="What is the maximum defect tolerance percentage allowed for lithium cell batches?",
        expected_stage1_tier=RetrievalTier.TIER1_VECTOR,
        expected_stage1_mcp_need=False,
        expected_reasoning_depth_min=1.0,
        expected_reasoning_depth_max=2.0,
        requires_mid_pipeline_escalation=False,
        expected_final_outcome="answered",
        expected_citations=["QA_SOP, p.12"],
    ),
    BenchmarkScenario(
        id="query_2_escalation",
        name="Query 2 (Medium + Mid-Flight Escalation)",
        query="Why was the EV-Pack-V2 line halted yesterday, and did the supplier breach their SLA penalty window?",
        expected_stage1_tier=RetrievalTier.TIER2_HYBRID,
        expected_stage1_mcp_need=False,  # Initially routes to hybrid because query mentions contract clauses
        expected_reasoning_depth_min=2.5,
        expected_reasoning_depth_max=3.5,
        requires_mid_pipeline_escalation=True,  # Triggers escalation when SLA doc lacks "yesterday's halt"
        expected_final_outcome="escalated_and_answered",
        expected_citations=["MCP-ERP #402", "VoltCore_SLA, p.6"],
    ),
    BenchmarkScenario(
        id="query_3_multihop",
        name="Query 3 (Complex Multi-Hop)",
        query="Which open GitHub firmware issues affect projects using VoltCore's delayed batches, and who is the lead engineer?",
        expected_stage1_tier=RetrievalTier.TIER3_AGENTIC,
        expected_stage1_mcp_need=True,
        expected_reasoning_depth_min=4.0,
        expected_reasoning_depth_max=5.0,
        requires_mid_pipeline_escalation=False,
        expected_final_outcome="answered",
        expected_citations=["Graph: BMS-v4", "GitHub #114, #119"],
    ),
    BenchmarkScenario(
        id="query_4_trap",
        name="Query 4 (Unanswerable / Trap)",
        query="Which supplier will win our Q3 2027 cobalt tender and at what price per ton?",
        expected_stage1_tier=RetrievalTier.TIER2_HYBRID,
        expected_stage1_mcp_need=False,
        expected_reasoning_depth_min=2.0,
        expected_reasoning_depth_max=3.0,
        requires_mid_pipeline_escalation=True,
        expected_final_outcome="abstained",
        expected_citations=[],
    ),
]
