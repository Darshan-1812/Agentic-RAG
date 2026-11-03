"""Unit tests for Member 4: 5-Baseline Benchmark & Pareto analysis."""
import pytest
from benchmarks.run_5_baseline_benchmark import run_5_baseline_evaluation, format_research_paper_table
from benchmarks.generate_pareto_curves import render_ascii_pareto_frontier


def test_5_baseline_evaluation_run():
    data = run_5_baseline_evaluation(n_queries=300)
    assert data["dataset_metadata"]["total_queries"] == 300
    assert len(data["benchmarks"]) == 5

    calibrag = data["benchmarks"]["Proposed: CalibRAG (Jev System 1 + Cascaded System 2)"]
    always_on = data["benchmarks"]["B4: Always-On Agentic RAG"]

    # Verify CalibRAG beats or matches Always-On Agentic RAG with higher faithfulness and ~70% lower cost
    assert calibrag["Answer Faithfulness (%)"] > always_on["Answer Faithfulness (%)"]
    assert calibrag["Cost Savings vs Always-On (%)"] >= 65.0
    assert calibrag["P50 Latency (ms)"] < always_on["P50 Latency (ms)"]
    assert calibrag["Abstention F1 (Traps)"] >= 0.90


def test_markdown_table_and_pareto_diagram():
    data = run_5_baseline_evaluation(n_queries=300)
    table = format_research_paper_table(data)
    assert "Proposed: CalibRAG" in table
    assert "Always-On Agentic RAG" in table

    diagram = render_ascii_pareto_frontier()
    assert "Expected Research Evaluation Pareto Frontier" in diagram
    assert "Proposed: CalibRAG" in diagram
