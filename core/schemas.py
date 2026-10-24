"""Core schemas and type definitions for ACR-RAG Platform."""
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from core.config import SLAPolicyMode


class RetrievalTier(str, Enum):
    TIER1_VECTOR = "tier1_vector"
    TIER2_HYBRID = "tier2_hybrid"
    TIER3_AGENTIC = "tier3_agentic"


class QueryRequest(BaseModel):
    """User incoming query request to ACR-RAG."""
    query: str = Field(..., description="The user's prompt or question")
    workspace_id: str = Field(default="default_workspace", description="Tenant or workspace ID")
    sla_policy: Optional[SLAPolicyMode] = Field(default=None, description="Override SLA policy profile")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkEvidence(BaseModel):
    """A retrieved knowledge chunk with cross-encoder score and metadata."""
    chunk_id: str
    doc_id: str
    content: str
    cross_encoder_logit: float = Field(..., description="Raw logit score from cross-encoder (e.g. bge-reranker-v2-m3)")
    page_number: Optional[int] = None
    source_type: str = Field(default="pdf", description="pdf, graph, sql, github")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PreRetrievalRouteDecision(BaseModel):
    """System 1 Pre-Retrieval Triage decision evaluated in ~90ms."""
    query: str
    selected_tier: RetrievalTier
    tier_probabilities: Dict[str, float] = Field(..., description="Calibrated probabilities across tiers")
    needs_live_mcp: bool = Field(..., description="Noul check: requires live DB/API")
    p_needs_live_mcp: float = Field(..., description="Calibrated probability for live tool need")
    reasoning_depth: float = Field(..., ge=1.0, le=5.0, description="Complexity score (1: lookup, 3: comparison, 5: synthesis)")
    sla_policy: SLAPolicyMode
    latency_ms: float = Field(..., description="Execution latency of System 1 triage in milliseconds")
    estimated_cost_usd: float = Field(..., description="Jev token cost for the triage evaluation")
    explanation: str


class EvidenceConfidenceResult(BaseModel):
    """Mid-Pipeline Evidence Sufficiency Gate result (C_evidence formula)."""
    c_evidence: float = Field(..., ge=0.0, le=1.0, description="Composite evidence confidence score")
    p_jev_noul_sufficient: float = Field(..., ge=0.0, le=1.0, description="Jev Noul probability that passages answer query")
    sigma_delta_rerank: float = Field(..., ge=0.0, le=1.0, description="Normalized sigmoid logit margin between top-1 and top-5")
    cov_entities: float = Field(..., ge=0.0, le=1.0, description="Entity coverage fraction in retrieved context")
    escalate_triggered: bool = Field(..., description="Whether mid-flight escalation was triggered")
    target_tier: Optional[RetrievalTier] = Field(default=None, description="Escalation destination tier if triggered")
    tau_escalate: float = Field(..., description="Threshold applied")
    tau_abstain: float = Field(..., description="Abstention floor applied")
    policy_mode: SLAPolicyMode


class ClaimVerification(BaseModel):
    """Verification check for a single claim or sentence in the generated draft."""
    claim_index: int
    sentence: str
    cited_source: Optional[str] = None
    noul_entailment_prob: float = Field(..., ge=0.0, le=1.0, description="Calibrated entailment probability")
    is_entailed: bool = Field(..., description="Whether claim passes the minimum Noul entailment threshold")
    unsupported_reason: Optional[str] = None


class VerificationResult(BaseModel):
    """Post-Generation verification report across all sentences."""
    original_text: str
    verified_text: str
    claims: List[ClaimVerification]
    all_claims_verified: bool
    stripped_claims_count: int
    mean_noul_score: float
    is_abstention_triggered: bool


class InsufficientEvidenceResponse(BaseModel):
    """Calibrated Abstention structured response when evidence is below tau_abstain."""
    status: str = "abstained"
    query: str
    confidence_score: float
    threshold_required: float
    missing_facts: List[str] = Field(default_factory=list, description="Specific factual assertions missing")
    missing_documents: List[str] = Field(default_factory=list, description="Knowledge sources, tables or tools needed")
    reason: str
    explanation: str
