"""Pre-Retrieval System 1 Router (Member 1: Control Plane & Router Lead).

Executes Stage 1 Pre-Retrieval Triage in ~90ms via TypeSafe AI Jev:
- Choice: ["tier1_vector", "tier2_hybrid", "tier3_agentic"]
- Noul: "Does answering this question require live database records, issue status, or external tool state?"
- Score: "1: Single fact lookup; 3: Multi-document comparison; 5: Multi-hop causal or relational synthesis"
"""
import time
from typing import List, Dict, Any, Optional
from core.schemas import RetrievalTier, PreRetrievalRouteDecision
from core.config import SLAPolicyMode, settings
from router.jev_client import JevClient, JevChoice, JevScore, JevNoul
from router.policy_manager import PolicyManager
from router.fallback_router import FallbackRouter


class PreRetrievalRouter:
    """Stage 1 Pre-Retrieval System 1 Triage Orchestrator."""

    def __init__(
        self,
        jev_client: Optional[JevClient] = None,
        policy_manager: Optional[PolicyManager] = None,
        fallback_router: Optional[FallbackRouter] = None,
    ):
        self.jev = jev_client or JevClient()
        self.policy_manager = policy_manager or PolicyManager()
        self.fallback = fallback_router or FallbackRouter()

    def route(
        self,
        query: str,
        workspace_id: str = "default_workspace",
        sla_policy: Optional[SLAPolicyMode] = None,
        tools_manifest: Optional[List[Dict[str, Any]]] = None,
    ) -> PreRetrievalRouteDecision:
        """Evaluates incoming user query with Jev in a single parallel request (~90 ms)."""
        start_time = time.perf_counter()
        active_policy = self.policy_manager.get_policy(workspace_id=workspace_id, override_mode=sla_policy)

        # Build atomic typed System 1 primitives for parallel evaluation
        primitives = {
            "tier_choice": JevChoice(
                options=[
                    RetrievalTier.TIER1_VECTOR.value,
                    RetrievalTier.TIER2_HYBRID.value,
                    RetrievalTier.TIER3_AGENTIC.value,
                ],
                description="Select minimum viable retrieval tier for query complexity"
            ),
            "needs_live_mcp": JevNoul(
                question="Does answering this question require live database records, issue status, or external tool state?"
            ),
            "reasoning_depth": JevScore(
                rubric="1: Single fact lookup; 3: Multi-document comparison; 5: Multi-hop causal or relational synthesis",
                min_val=1.0,
                max_val=5.0
            ),
        }

        try:
            results = self.jev.evaluate_parallel(
                query=query,
                context=None,
                primitives=primitives,
                tools_manifest=tools_manifest
            )

            choice_res = results.get("tier_choice")
            mcp_res = results.get("needs_live_mcp")
            score_res = results.get("reasoning_depth")

            base_tier = RetrievalTier(choice_res.value)
            dist = choice_res.distribution or {base_tier.value: 1.0}
            p_needs_mcp = mcp_res.calibrated_probability if mcp_res else 0.0
            needs_mcp = bool(mcp_res.value) if mcp_res else False
            reasoning_score = float(score_res.value) if score_res else 1.0

            # Apply governance policy adjustments
            final_tier = self.policy_manager.adjust_tier_for_policy(
                base_tier=base_tier,
                reasoning_score=reasoning_score,
                p_needs_mcp=p_needs_mcp,
                policy=active_policy
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            # Jev pricing: $0.042 per MTok; typical request ~250 input tokens, 0 output token cost
            est_cost = (250 / 1_000_000.0) * settings.jev_price_per_million_input_tokens

            return PreRetrievalRouteDecision(
                query=query,
                selected_tier=final_tier,
                tier_probabilities=dist,
                needs_live_mcp=needs_mcp,
                p_needs_live_mcp=round(p_needs_mcp, 4),
                reasoning_depth=round(reasoning_score, 2),
                sla_policy=active_policy.mode,
                latency_ms=round(elapsed_ms, 2),
                estimated_cost_usd=round(est_cost, 7),
                explanation=f"Jev System 1 routed query to {final_tier.value} (prob: {dist.get(final_tier.value, 1.0):.2f}) with reasoning depth {reasoning_score:.1f} and live tool probability {p_needs_mcp:.2f}."
            )

        except Exception as err:
            # Automatic graceful fallback to RouteLLM / DeBERTa calibrated classifier
            fallback_decision = self.fallback.route(query=query, sla_policy=active_policy.mode)
            fallback_decision.explanation += f" (Note: Jev fallback engaged due to: {str(err)})"
            return fallback_decision
