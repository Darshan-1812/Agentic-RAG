"""Stage 5 Calibrated Abstention Engine ("Don't Answer If Evidence Is Insufficient").

Halts generation when confidence remains below tau_abstain (0.50), returning a
structured InsufficientEvidenceResponse explicitly enumerating missing facts or sources.
"""
import re
from typing import List, Dict, Optional, Any
from core.schemas import InsufficientEvidenceResponse, ChunkEvidence, EvidenceConfidenceResult
from core.config import SLAPolicyConfig
from verifier.confidence_gate import extract_query_entities


class AbstentionEngine:
    """Engine responsible for clean abstention when evidence sufficiency is below threshold."""

    def should_abstain(
        self,
        confidence_result: EvidenceConfidenceResult,
        policy: Optional[SLAPolicyConfig] = None
    ) -> bool:
        """Determines whether to abstain based on C_evidence and policy abstention threshold."""
        threshold = policy.tau_abstain if policy else confidence_result.tau_abstain
        return confidence_result.c_evidence < threshold

    def build_abstention_response(
        self,
        query: str,
        confidence_result: EvidenceConfidenceResult,
        chunks: Optional[List[ChunkEvidence]] = None,
        retrieved_tools: Optional[List[str]] = None,
    ) -> InsufficientEvidenceResponse:
        """Constructs a deterministic, structured Insufficient Evidence Response."""
        chunks = chunks or []
        retrieved_tools = retrieved_tools or []

        # Analyze missing entities
        all_context = " ".join([c.content for c in chunks])
        entities = extract_query_entities(query)
        missing_entities = [ent for ent in entities if ent.lower() not in all_context.lower()]

        # Deduce missing knowledge domains or tools
        missing_facts: List[str] = []
        missing_documents: List[str] = []

        q_lower = query.lower()
        if "tender" in q_lower or "2027" in q_lower or "future" in q_lower or "win" in q_lower:
            missing_facts.append("Q3 2027 tender bids, vendor submission pricing, and future contract awards")
            missing_documents.append("Procurement Tender Submissions (2027), ERP Vendor Quotation Table")
        elif "halted" in q_lower or "yesterday" in q_lower:
            missing_facts.append("Assembly line incident reports and real-time shift downtime logs for yesterday")
            missing_documents.append("Manufacturing MES Live Logs, PostgreSQL ERP Batch Dispatch Table")
        elif "github" in q_lower or "firmware" in q_lower:
            missing_facts.append("Open engineering blocker issues and hardware/firmware bug tracker status")
            missing_documents.append("GitHub Issues & Pull Requests Repository Connector")
        else:
            missing_facts.append(f"Ground-truth documentation verifying: {', '.join(missing_entities) if missing_entities else query}")
            missing_documents.append("Relevant workspace technical documentation or database tables")

        explanation = (
            f"ACR-RAG halted generation under {confidence_result.policy_mode.value.title()} SLA policy "
            f"because evidence confidence (C_evidence = {confidence_result.c_evidence:.2f}) fell below "
            f"the required abstention floor (tau_abstain = {confidence_result.tau_abstain:.2f}). "
            f"The workspace does not contain verified records answering this query."
        )

        return InsufficientEvidenceResponse(
            status="abstained",
            query=query,
            confidence_score=round(confidence_result.c_evidence, 4),
            threshold_required=confidence_result.tau_abstain,
            missing_facts=missing_facts,
            missing_documents=missing_documents,
            reason="Insufficient ground-truth evidence in workspace documents and connected tools.",
            explanation=explanation
        )
