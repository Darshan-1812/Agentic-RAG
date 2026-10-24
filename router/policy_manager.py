"""SLA & Cost Governance Policy Manager.

Provides dynamic thresholding and routing governance across Economy, Balanced,
and Strict Zero-Hallucination modes.
"""
from typing import Dict, Optional
from core.config import SLAPolicyMode, SLAPolicyConfig, DEFAULT_POLICY_PROFILES, settings
from core.schemas import RetrievalTier


class PolicyManager:
    """Manages workspace SLA governance profiles and threshold adjustments."""

    def __init__(self, default_mode: SLAPolicyMode = SLAPolicyMode.BALANCED):
        self.default_mode = default_mode
        self._workspace_policies: Dict[str, SLAPolicyConfig] = {}

    def get_policy(self, workspace_id: Optional[str] = None, override_mode: Optional[SLAPolicyMode] = None) -> SLAPolicyConfig:
        """Resolves the active SLA policy config given workspace and potential override."""
        if override_mode:
            return DEFAULT_POLICY_PROFILES.get(override_mode, DEFAULT_POLICY_PROFILES[self.default_mode])
        if workspace_id and workspace_id in self._workspace_policies:
            return self._workspace_policies[workspace_id]
        return DEFAULT_POLICY_PROFILES[self.default_mode]

    def set_workspace_policy(self, workspace_id: str, mode: SLAPolicyMode) -> SLAPolicyConfig:
        """Sets or updates the SLA policy for a specific workspace."""
        config = DEFAULT_POLICY_PROFILES[mode]
        self._workspace_policies[workspace_id] = config
        return config

    def adjust_tier_for_policy(
        self,
        base_tier: RetrievalTier,
        reasoning_score: float,
        p_needs_mcp: float,
        policy: SLAPolicyConfig
    ) -> RetrievalTier:
        """Adjusts selected tier based on SLA constraints."""
        if policy.mode == SLAPolicyMode.ECONOMY:
            # Demote to lower tier if borderline to save tokens
            if base_tier == RetrievalTier.TIER3_AGENTIC and p_needs_mcp < 0.80 and reasoning_score < 4.5:
                return RetrievalTier.TIER2_HYBRID
            if base_tier == RetrievalTier.TIER2_HYBRID and reasoning_score < 3.0:
                return RetrievalTier.TIER1_VECTOR
            return base_tier

        elif policy.mode == SLAPolicyMode.STRICT:
            # Escalate aggressively to ensure zero hallucination
            if base_tier == RetrievalTier.TIER1_VECTOR and (reasoning_score > 2.0 or p_needs_mcp > 0.20):
                return RetrievalTier.TIER2_HYBRID
            if base_tier == RetrievalTier.TIER2_HYBRID and (reasoning_score > 3.0 or p_needs_mcp > 0.40):
                return RetrievalTier.TIER3_AGENTIC
            return base_tier

        # Balanced: keep base tier
        return base_tier
