"""Deterministic ranking metrics for the retrieval evaluation harness.

All functions accept ordered chunk identifiers and a collection of relevant
chunk identifiers. They raise ``ValueError`` for empty relevance sets so a
missing label cannot silently inflate an aggregate score.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Set, Tuple


def _ordered_ids(values: Iterable[object]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in values:
        chunk_id = str(value)
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            ordered.append(chunk_id)
    return ordered


def _relevant_set(values: Iterable[object]) -> Set[str]:
    relevant = {str(value) for value in values if str(value)}
    if not relevant:
        raise ValueError("relevance labels must contain at least one chunk id")
    return relevant


def _validate_k(k: int) -> int:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be an integer >= 1")
    return k


def recall_at_k(
    ranked_ids: Sequence[object], relevant_ids: Iterable[object], k: int
) -> float:
    """Return the fraction of relevant chunks found in the first ``k`` results."""
    k = _validate_k(k)
    ranked = _ordered_ids(ranked_ids)[:k]
    relevant = _relevant_set(relevant_ids)
    return len(relevant.intersection(ranked)) / len(relevant)


def precision_at_k(
    ranked_ids: Sequence[object], relevant_ids: Iterable[object], k: int
) -> float:
    """Return the fraction of the first ``k`` results that are relevant."""
    k = _validate_k(k)
    ranked = _ordered_ids(ranked_ids)[:k]
    relevant = _relevant_set(relevant_ids)
    return len(relevant.intersection(ranked)) / k


def hit_at_k(
    ranked_ids: Sequence[object], relevant_ids: Iterable[object], k: int
) -> float:
    """Return ``1.0`` when at least one relevant chunk appears in the top ``k``."""
    k = _validate_k(k)
    ranked = _ordered_ids(ranked_ids)[:k]
    relevant = _relevant_set(relevant_ids)
    return 1.0 if relevant.intersection(ranked) else 0.0


def reciprocal_rank_at_k(
    ranked_ids: Sequence[object], relevant_ids: Iterable[object], k: int
) -> float:
    """Return the reciprocal rank of the first relevant chunk within ``k``."""
    k = _validate_k(k)
    ranked = _ordered_ids(ranked_ids)[:k]
    relevant = _relevant_set(relevant_ids)
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[object], relevant_ids: Iterable[object], k: int
) -> float:
    """Return binary-relevance nDCG@k."""
    k = _validate_k(k)
    ranked = _ordered_ids(ranked_ids)[:k]
    relevant = _relevant_set(relevant_ids)

    dcg = 0.0
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_ranking(
    ranked_ids: Sequence[object],
    relevant_ids: Iterable[object],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Tuple[dict[str, float], List[int]]:
    """Compute all supported metrics for one ranked result list."""
    relevant = _relevant_set(relevant_ids)
    ranked = _ordered_ids(ranked_ids)
    ks = sorted({_validate_k(k) for k in k_values})
    if not ks:
        raise ValueError("k_values must contain at least one value")

    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"Recall@{k}"] = recall_at_k(ranked, relevant, k)
        metrics[f"Precision@{k}"] = precision_at_k(ranked, relevant, k)
        metrics[f"Hit@{k}"] = hit_at_k(ranked, relevant, k)
        metrics[f"nDCG@{k}"] = ndcg_at_k(ranked, relevant, k)
        metrics[f"MRR@{k}"] = reciprocal_rank_at_k(ranked, relevant, k)
    return metrics, ks
