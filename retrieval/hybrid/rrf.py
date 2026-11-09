"""Reciprocal Rank Fusion for dense and lexical retrieval results."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence

from retrieval.retrieval_types import RetrievedChunk


class FusionError(ValueError):
    """Raised when RRF inputs are invalid."""


def reciprocal_rank_fusion(
    vector_results: Sequence[RetrievedChunk],
    bm25_results: Sequence[RetrievedChunk],
    *,
    k: int = 60,
    top_k: Optional[int] = None,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> List[RetrievedChunk]:
    """Fuse ranked lists with ``weight / (k + rank)`` and merge duplicates."""
    if k < 1:
        raise FusionError("rrf k must be >= 1")
    if top_k is not None and top_k < 1:
        raise FusionError("top_k must be >= 1")
    if vector_weight < 0 or bm25_weight < 0:
        raise FusionError("RRF weights must be non-negative")
    if vector_weight == 0 and bm25_weight == 0:
        raise FusionError("at least one RRF weight must be positive")

    scores: Dict[str, float] = {}
    chunks: Dict[str, RetrievedChunk] = {}

    ranked_lists = (
        (vector_results, float(vector_weight), "vector"),
        (bm25_results, float(bm25_weight), "bm25"),
    )
    for results, weight, source in ranked_lists:
        for rank, chunk in enumerate(results, start=1):
            if not isinstance(chunk, RetrievedChunk):
                raise FusionError(
                    f"{source} results must contain RetrievedChunk objects"
                )
            if not chunk.chunk_id:
                raise FusionError("fused chunks must have a chunk_id")
            existing = chunks.get(chunk.chunk_id)
            if existing is None:
                current = chunk
            else:
                if existing.doc_id != chunk.doc_id:
                    raise FusionError(
                        f"chunk {chunk.chunk_id} appears under multiple doc_ids"
                    )
                current = existing.merged_with(chunk)
            if source not in current.retrieval_sources:
                current = replace(
                    current,
                    retrieval_sources=current.retrieval_sources + (source,),
                )
            chunks[chunk.chunk_id] = current
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                weight / (k + rank)
            )

    ordered = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], chunk_id),
    )
    if top_k is not None:
        ordered = ordered[:top_k]

    fused: List[RetrievedChunk] = []
    for rank, chunk_id in enumerate(ordered, start=1):
        fused.append(
            replace(
                chunks[chunk_id],
                rrf_score=scores[chunk_id],
                rrf_rank=rank,
            )
        )
    return fused


def rrf_fuse(
    vector_results: Sequence[RetrievedChunk],
    bm25_results: Sequence[RetrievedChunk],
    *,
    k: int = 60,
    top_k: Optional[int] = None,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> List[RetrievedChunk]:
    """Alias kept explicit for callers that prefer a short fusion name."""
    return reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        k=k,
        top_k=top_k,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )
