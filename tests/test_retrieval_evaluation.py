"""Phase 5 tests for the real retrieval evaluation harness and metrics."""
from __future__ import annotations

import json

import pytest

from retrieval.evaluation import (
    RetrievalEvalCase,
    benchmark_retriever,
    evaluate_ranking,
    evaluate_retrieval,
    load_eval_cases,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_ranking_metrics_match_hand_computed_values():
    ranked = ["a", "b", "c", "d"]
    relevant = ["b", "d"]

    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 2) == pytest.approx(0.5)
    assert recall_at_k(ranked, relevant, 4) == pytest.approx(1.0)
    assert precision_at_k(ranked, relevant, 2) == pytest.approx(0.5)
    assert reciprocal_rank_at_k(ranked, relevant, 2) == pytest.approx(0.5)
    assert ndcg_at_k(ranked, relevant, 2) == pytest.approx(0.386852807, rel=1e-6)
    assert ndcg_at_k(ranked, relevant, 4) == pytest.approx(0.65092093, rel=1e-6)


def test_evaluate_ranking_reports_all_requested_cutoffs():
    metrics, ks = evaluate_ranking(
        ["c1", "c2", "c3"], ["c2"], k_values=(1, 3)
    )

    assert ks == [1, 3]
    assert metrics["Recall@1"] == 0.0
    assert metrics["Recall@3"] == 1.0
    assert metrics["Precision@1"] == 0.0
    assert metrics["Precision@3"] == pytest.approx(1 / 3)
    assert metrics["Hit@1"] == 0.0
    assert metrics["Hit@3"] == 1.0
    assert metrics["MRR@1"] == 0.0
    assert metrics["MRR@3"] == pytest.approx(0.5)


def test_metrics_reject_empty_relevance_and_duplicate_ids_are_ignored():
    with pytest.raises(ValueError):
        recall_at_k(["c1"], [], 1)

    metrics, _ = evaluate_ranking(["c1", "c1", "c2"], ["c2"], k_values=(2,))
    assert metrics["Recall@2"] == 1.0
    assert metrics["Precision@2"] == pytest.approx(0.5)


async def test_evaluation_harness_uses_actual_rankings():
    calls = []

    async def retriever(query, workspace_id="default", top_k=10):
        calls.append((query, workspace_id, top_k))
        return [{"chunk_id": "c2"}, {"chunk_id": "c1"}]

    report = await evaluate_retrieval(
        retriever,
        [
            RetrievalEvalCase(
                query="defect tolerance",
                workspace_id="tenant_a",
                relevant_chunk_ids=("c2",),
            )
        ],
        top_k=2,
        k_values=(1, 2),
    )

    assert calls == [("defect tolerance", "tenant_a", 2)]
    assert report.ok
    assert report.successful_cases == 1
    assert report.metrics["Recall@1"] == 1.0
    assert report.metrics["MRR@2"] == 1.0
    assert report.per_case[0]["ranked_chunk_ids"] == ["c2", "c1"]


async def test_evaluation_harness_records_failures_without_inventing_scores():
    async def retriever(query, workspace_id="default", top_k=10):
        if query == "broken":
            raise RuntimeError("database unavailable")
        return ["c1"]

    report = await evaluate_retrieval(
        retriever,
        [
            RetrievalEvalCase(query="ok", relevant_chunk_ids=("c1",)),
            RetrievalEvalCase(query="broken", relevant_chunk_ids=("c1",)),
            RetrievalEvalCase(query="unlabeled", relevant_chunk_ids=()),
        ],
        k_values=(1,),
    )

    assert report.successful_cases == 1
    assert report.failed_cases == 2
    assert report.metrics["Recall@1"] == 1.0
    assert len(report.failures) == 2


def test_load_eval_cases_supports_json_and_jsonl(tmp_path):
    json_path = tmp_path / "cases.json"
    json_path.write_text(
        json.dumps(
            [
                {"query": "q1", "workspace_id": "tenant_a", "relevant_chunk_ids": ["c1"]},
                {"query": "q2", "relevant_ids": ["c2"]},
            ]
        ),
        encoding="utf-8",
    )
    cases = load_eval_cases(json_path)
    assert [case.query for case in cases] == ["q1", "q2"]
    assert cases[0].workspace_id == "tenant_a"
    assert cases[1].relevant_chunk_ids == ("c2",)

    jsonl_path = tmp_path / "cases.jsonl"
    jsonl_path.write_text(
        '{"query": "q3", "relevant_chunk_ids": ["c3"]}\n',
        encoding="utf-8",
    )
    cases = load_eval_cases(jsonl_path)
    assert len(cases) == 1
    assert cases[0].relevant_chunk_ids == ("c3",)


async def test_benchmark_reports_latency_and_memory_for_async_retriever():
    async def retriever(query, workspace_id="default", top_k=5):
        return [{"chunk_id": "c1"}]

    result = await benchmark_retriever(
        retriever,
        "defect tolerance",
        workspace_id="tenant_a",
        top_k=1,
        iterations=3,
        warmup=1,
    )

    assert result.successes == 3
    assert result.failures == 0
    assert len(result.latencies_ms) == 3
    assert result.p50_ms >= 0
    assert result.p95_ms >= result.p50_ms
    assert result.peak_memory_mb is not None
