"""Mid-Pipeline Mathematical Evidence Sufficiency Gate (C_evidence).

Implements Stage 3 of the ACR-RAG runtime flow:
    C_evidence = 0.50 * P_Jev_Noul(Sufficient) + 0.30 * sigma(Delta L_rerank) + 0.20 * Cov_entities
"""
import re
import math
from typing import List, Optional, Set
from core.schemas import ChunkEvidence, EvidenceConfidenceResult, RetrievalTier
from core.config import SLAPolicyMode, SLAPolicyConfig
from router.jev_client import JevClient, JevNoul
from router.policy_manager import PolicyManager


def sigmoid(x: float) -> float:
    """Standard logistic sigmoid function mapping (-inf, inf) to (0, 1)."""
    if x >= 20.0:
        return 1.0
    if x <= -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def extract_query_entities(query: str) -> Set[str]:
    """Extracts named entities, model codes, acronyms, and key nominal tokens from query."""
    # Matches words with capital letters, hyphenated codes (e.g. EV-Pack-V2, BMS-Controller-v4, Batch #402), numbers
    raw_tokens = re.findall(r'\b(?:[A-Z][a-zA-Z0-9_-]+|\d+(?:\.\d+)?|#[0-9]+)\b', query)
    stopwords = {"what", "when", "where", "which", "whose", "why", "does", "did", "have", "with", "from", "that", "this", "about", "the"}
    
    entities = set()
    for t in raw_tokens:
        clean = t.strip().lower()
        if len(clean) > 1 and clean not in stopwords:
            entities.add(clean)
    return entities


def compute_entity_coverage(query: str, context_text: str) -> float:
    """Computes Cov_entities: fraction of query named entities present in retrieved context."""
    entities = extract_query_entities(query)
    if not entities:
        # If no explicit named entities, fallback to major nouns/tokens (>3 chars)
        words = {w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', query)}
        stopwords = {"what", "when", "where", "which", "whose", "does", "have", "with", "from", "that", "this", "about"}
        keywords = words - stopwords
        if not keywords:
            return 1.0
        c_lower = context_text.lower()
        matched = sum(1 for kw in keywords if kw in c_lower)
        return matched / len(keywords)

    c_lower = context_text.lower()
    matched = sum(1 for ent in entities if ent in c_lower)
    return matched / len(entities)


def compute_evidence_confidence(
    query: str,
    chunks: List[ChunkEvidence],
    p_jev_noul_sufficient: float,
    policy: SLAPolicyConfig,
    current_tier: RetrievalTier = RetrievalTier.TIER1_VECTOR,
) -> EvidenceConfidenceResult:
    """Calculates the composite evidence confidence score C_evidence and escalation decision."""
    # 1. Delta L rerank
    if not chunks:
        delta_l = 0.0
        sigma_rerank = 0.0
        combined_text = ""
    else:
        # Sort chunks by cross_encoder_logit descending
        sorted_chunks = sorted(chunks, key=lambda c: c.cross_encoder_logit, reverse=True)
        top1_logit = sorted_chunks[0].cross_encoder_logit
        top5_logit = sorted_chunks[min(4, len(sorted_chunks) - 1)].cross_encoder_logit
        delta_l = top1_logit - top5_logit
        sigma_rerank = sigmoid(delta_l)
        combined_text = " ".join([c.content for c in sorted_chunks[:5]])

    # 2. Entity coverage
    cov_entities = compute_entity_coverage(query, combined_text)

    # 3. Composite Formula: C_evidence = 0.50 * P_Jev_Noul + 0.30 * sigma(Delta L) + 0.20 * Cov_entities
    c_evidence = (
        0.50 * p_jev_noul_sufficient +
        0.30 * sigma_rerank +
        0.20 * cov_entities
    )
    c_evidence = max(0.0, min(1.0, c_evidence))

    # Escalation decision: If C_evidence < tau_escalate
    escalate_triggered = c_evidence < policy.tau_escalate

    target_tier = None
    if escalate_triggered:
        if current_tier == RetrievalTier.TIER1_VECTOR:
            target_tier = RetrievalTier.TIER2_HYBRID
        elif current_tier == RetrievalTier.TIER2_HYBRID:
            target_tier = RetrievalTier.TIER3_AGENTIC
        else:
            target_tier = RetrievalTier.TIER3_AGENTIC

    return EvidenceConfidenceResult(
        c_evidence=round(c_evidence, 4),
        p_jev_noul_sufficient=round(p_jev_noul_sufficient, 4),
        sigma_delta_rerank=round(sigma_rerank, 4),
        cov_entities=round(cov_entities, 4),
        escalate_triggered=escalate_triggered,
        target_tier=target_tier,
        tau_escalate=policy.tau_escalate,
        tau_abstain=policy.tau_abstain,
        policy_mode=policy.mode,
    )


class EvidenceSufficiencyGate:
    """Stage 3 Gate evaluating retrieved chunks with Jev Noul and mathematical confidence."""

    def __init__(self, jev_client: Optional[JevClient] = None, policy_manager: Optional[PolicyManager] = None):
        self.jev = jev_client or JevClient()
        self.policy_manager = policy_manager or PolicyManager()

    def evaluate(
        self,
        query: str,
        chunks: List[ChunkEvidence],
        current_tier: RetrievalTier,
        workspace_id: str = "default_workspace",
        sla_override: Optional[SLAPolicyMode] = None,
    ) -> EvidenceConfidenceResult:
        """Evaluates chunks against query using Jev Noul sufficiency check and composite formula."""
        policy = self.policy_manager.get_policy(workspace_id=workspace_id, override_mode=sla_override)
        combined_text = "\n\n".join([f"[{c.doc_id}] {c.content}" for c in chunks[:5]])

        # Query Jev Noul primitive: "Do passages contain complete facts to answer?"
        primitives = {
            "is_sufficient": JevNoul(
                question="Do passages contain complete facts to answer the question without guessing?"
            )
        }
        res = self.jev.evaluate_parallel(query=query, context=combined_text, primitives=primitives)
        noul_res = res.get("is_sufficient")
        p_sufficient = noul_res.calibrated_probability if noul_res else 0.50

        return compute_evidence_confidence(
            query=query,
            chunks=chunks,
            p_jev_noul_sufficient=p_sufficient,
            policy=policy,
            current_tier=current_tier,
        )
