"""ACR-RAG Verifier Package (Member 1: Control Plane & Router Lead)."""
from verifier.confidence_gate import EvidenceSufficiencyGate, compute_evidence_confidence
from verifier.claim_verifier import ClaimVerifier
from verifier.abstention_engine import AbstentionEngine
from verifier.calibration_metrics import (
    calculate_ece,
    calculate_brier_score,
    generate_reliability_diagram,
)

__all__ = [
    "EvidenceSufficiencyGate",
    "compute_evidence_confidence",
    "ClaimVerifier",
    "AbstentionEngine",
    "calculate_ece",
    "calculate_brier_score",
    "generate_reliability_diagram",
]
