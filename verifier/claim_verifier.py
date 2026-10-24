"""Stage 6 Post-Generation Claim Verifier & Sentence-Level Noul Entailment Guard.

Evaluates every sentence in a draft answer against cited sources using parallel
Jev Noul checks: "Is claim X directly entailed by source Y?"
Strips or flags unsupported claims and triggers abstention if evidence is insufficient.
"""
import re
from typing import List, Dict, Optional, Tuple
from core.schemas import ClaimVerification, VerificationResult
from core.config import SLAPolicyConfig
from router.jev_client import JevClient, JevNoul
from router.policy_manager import PolicyManager


def split_into_sentences(text: str) -> List[str]:
    """Splits markdown or text into clean sentences while preserving bracket citations."""
    # Split on period followed by space, or newlines, ignoring abbreviations like e.g., i.e., vs.
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\[])|\n\n+', text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences


def extract_inline_citations(sentence: str) -> List[str]:
    """Finds inline citations like [Doc-1, p.4] or [MCP-ERP #402] or [GitHub #114]."""
    return re.findall(r'\[([^\]]+)\]', sentence)


class ClaimVerifier:
    """Post-generation claim verifier using parallel Jev Noul primitives."""

    def __init__(self, jev_client: Optional[JevClient] = None, policy_manager: Optional[PolicyManager] = None):
        self.jev = jev_client or JevClient()
        self.policy_manager = policy_manager or PolicyManager()

    def verify_draft(
        self,
        draft_text: str,
        sources_context: Dict[str, str],
        policy: SLAPolicyConfig,
    ) -> VerificationResult:
        """Evaluates draft text sentence-by-sentence in parallel against cited context."""
        sentences = split_into_sentences(draft_text)
        if not sentences:
            return VerificationResult(
                original_text=draft_text,
                verified_text=draft_text,
                claims=[],
                all_claims_verified=True,
                stripped_claims_count=0,
                mean_noul_score=1.0,
                is_abstention_triggered=False,
            )

        # Build parallel Noul primitives for each sentence
        primitives = {}
        for idx, sentence in enumerate(sentences):
            primitives[f"claim_{idx}"] = JevNoul(
                question=f"Is claim '{sentence}' directly entailed by cited source context without hallucination?"
            )

        # Combined context string from sources
        combined_source_context = "\n\n".join([f"[{k}] {v}" for k, v in sources_context.items()])

        # Parallel evaluation via Jev
        eval_results = self.jev.evaluate_parallel(
            query="Verify claims",
            context=combined_source_context,
            primitives=primitives
        )

        claims: List[ClaimVerification] = []
        verified_sentences: List[str] = []
        stripped_count = 0
        noul_scores: List[float] = []

        for idx, sentence in enumerate(sentences):
            res_key = f"claim_{idx}"
            jev_res = eval_results.get(res_key)
            prob = jev_res.calibrated_probability if jev_res else 0.50
            noul_scores.append(prob)

            citations = extract_inline_citations(sentence)
            cited_src = citations[0] if citations else None

            # Entailment check against policy min_sentence_noul threshold
            is_entailed = prob >= policy.min_sentence_noul

            reason = None if is_entailed else f"Noul score ({prob:.2f}) fell below required threshold ({policy.min_sentence_noul:.2f})"

            claim = ClaimVerification(
                claim_index=idx,
                sentence=sentence,
                cited_source=cited_src,
                noul_entailment_prob=round(prob, 4),
                is_entailed=is_entailed,
                unsupported_reason=reason,
            )
            claims.append(claim)

            if is_entailed:
                verified_sentences.append(sentence)
            else:
                stripped_count += 1

        mean_noul = sum(noul_scores) / len(noul_scores) if noul_scores else 0.0
        all_verified = (stripped_count == 0)

        # Trigger abstention if no sentences passed or mean score is below tau_abstain
        is_abstain = (len(verified_sentences) == 0) or (mean_noul < policy.tau_abstain)

        return VerificationResult(
            original_text=draft_text,
            verified_text=" ".join(verified_sentences),
            claims=claims,
            all_claims_verified=all_verified,
            stripped_claims_count=stripped_count,
            mean_noul_score=round(mean_noul, 4),
            is_abstention_triggered=is_abstain,
        )
