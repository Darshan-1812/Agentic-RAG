"""Telemetry and cost accounting engine for ACR-RAG."""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from core.schemas import RetrievalTier


# Estimated baseline costs per query tier in enterprise environments
BASELINE_COSTS = {
    # Always-On Agentic RAG baseline runs frontier model + graph traversal + tools every time
    "always_on_agentic_baseline": 0.0085,
    # ACR-RAG Tiers (Retrieval + Model generation)
    RetrievalTier.TIER1_VECTOR: 0.00018,
    RetrievalTier.TIER2_HYBRID: 0.00120,
    RetrievalTier.TIER3_AGENTIC: 0.00790,
}


class ACRTelemetry(BaseModel):
    """Runtime execution telemetry for Agent Control Room and HTTP response headers."""
    query_id: str
    workspace_id: str
    tier_used: RetrievalTier
    jev_confidence: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float
    total_cost_usd: float
    cost_saved_usd: float
    cost_savings_pct: float
    is_escalated: bool = False
    is_abstained: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_http_headers(self) -> Dict[str, str]:
        """Convert telemetry to standard ACR HTTP response headers as per blueprint Section 3.2."""
        return {
            "X-ACR-Tier-Used": self.tier_used.value,
            "X-ACR-Jev-Confidence": f"{self.jev_confidence:.4f}",
            "X-ACR-Latency-Ms": f"{self.latency_ms:.2f}",
            "X-ACR-Cost-Saved-USD": f"{self.cost_saved_usd:.6f}",
        }


def calculate_telemetry(
    query_id: str,
    workspace_id: str,
    tier_used: RetrievalTier,
    jev_confidence: float,
    latency_ms: float,
    actual_cost_usd: Optional[float] = None,
    is_escalated: bool = False,
    is_abstained: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> ACRTelemetry:
    """Computes runtime telemetry including exact dollar savings relative to Always-On Agentic RAG."""
    baseline = BASELINE_COSTS["always_on_agentic_baseline"]
    cost = actual_cost_usd if actual_cost_usd is not None else BASELINE_COSTS[tier_used]
    cost_saved = max(0.0, baseline - cost)
    savings_pct = (cost_saved / baseline) * 100.0 if baseline > 0 else 0.0

    return ACRTelemetry(
        query_id=query_id,
        workspace_id=workspace_id,
        tier_used=tier_used,
        jev_confidence=jev_confidence,
        latency_ms=latency_ms,
        total_cost_usd=cost,
        cost_saved_usd=cost_saved,
        cost_savings_pct=savings_pct,
        is_escalated=is_escalated,
        is_abstained=is_abstained,
        metadata=metadata or {},
    )
