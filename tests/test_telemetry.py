"""Unit tests for Telemetry, cost accounting, and HTTP header generation."""
import pytest
from core.schemas import RetrievalTier
from core.telemetry import calculate_telemetry


def test_telemetry_cost_savings_tier1():
    telemetry = calculate_telemetry(
        query_id="q-001",
        workspace_id="apex_autocomponents",
        tier_used=RetrievalTier.TIER1_VECTOR,
        jev_confidence=0.95,
        latency_ms=680.0,
        actual_cost_usd=0.00018,
    )

    assert telemetry.tier_used == RetrievalTier.TIER1_VECTOR
    assert telemetry.total_cost_usd == 0.00018
    assert telemetry.cost_saved_usd > 0.0080
    assert telemetry.cost_savings_pct >= 95.0

    headers = telemetry.to_http_headers()
    assert headers["X-ACR-Tier-Used"] == "tier1_vector"
    assert "0.9500" in headers["X-ACR-Jev-Confidence"]
    assert "680.00" in headers["X-ACR-Latency-Ms"]
    assert float(headers["X-ACR-Cost-Saved-USD"]) > 0.008


def test_telemetry_escalated():
    telemetry = calculate_telemetry(
        query_id="q-002",
        workspace_id="apex_autocomponents",
        tier_used=RetrievalTier.TIER3_AGENTIC,
        jev_confidence=0.64,
        latency_ms=3900.0,
        actual_cost_usd=0.0064,
        is_escalated=True,
    )
    assert telemetry.is_escalated is True
    assert telemetry.cost_saved_usd > 0
