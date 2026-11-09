"""Phase 5 retrieval evaluation, benchmarking, and labeled-case utilities."""
from __future__ import annotations

from retrieval.evaluation.benchmark import (
    RetrievalBenchmarkResult,
    benchmark_retriever,
)
from retrieval.evaluation.harness import (
    RetrievalEvalCase,
    RetrievalEvalReport,
    evaluate_retrieval,
    evaluate_service,
    load_eval_cases,
    write_eval_report,
)
from retrieval.evaluation.metrics import (
    evaluate_ranking,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

__all__ = [
    "RetrievalBenchmarkResult",
    "RetrievalEvalCase",
    "RetrievalEvalReport",
    "benchmark_retriever",
    "evaluate_ranking",
    "evaluate_retrieval",
    "evaluate_service",
    "hit_at_k",
    "load_eval_cases",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
    "write_eval_report",
]
