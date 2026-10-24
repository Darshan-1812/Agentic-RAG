"""ACR-RAG Core Package."""
from core.config import settings, SLAPolicyConfig
from core.schemas import (
    RetrievalTier,
    SLAPolicyMode,
    PreRetrievalRouteDecision,
    ChunkEvidence,
    EvidenceConfidenceResult,
    ClaimVerification,
    VerificationResult,
    InsufficientEvidenceResponse,
    QueryRequest,
)

__all__ = [
    "settings",
    "SLAPolicyConfig",
    "RetrievalTier",
    "SLAPolicyMode",
    "PreRetrievalRouteDecision",
    "ChunkEvidence",
    "EvidenceConfidenceResult",
    "ClaimVerification",
    "VerificationResult",
    "InsufficientEvidenceResponse",
    "QueryRequest",
]
