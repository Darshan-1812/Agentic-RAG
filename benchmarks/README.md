# Benchmarks Module (Member 4: BYOK Studio UI & Benchmark Lead)

## 1. Overview
The `benchmarks/` package implements the 5-baseline comparative evaluation framework defined in Section 7 & 8 of the CalibRAG blueprint for IEEE/ACM/Springer publication.

## 2. Baselines Evaluated:
- **(B1) Basic Vector RAG**: pgvector HNSW single-hop dense lookup.
- **(B2) Hybrid BM25+Vector RAG**: pgvector + ParadeDB BM25 with Reciprocal Rank Fusion.
- **(B3) Hybrid + LightRAG**: Dual-level graph context traversal without tool calling.
- **(B4) Always-On LangGraph Agentic RAG**: Cyclic multi-agent graph with tool calling on every query.
- **(Proposed) CalibRAG**: System 1 Jev Control Plane (~90ms triage) with confidence-gated escalation and post-generation claim verification.

## 3. Running the Benchmark:
```bash
python3 -m benchmarks.run_5_baseline_benchmark
```

## 4. Viewing the Pareto Frontier:
```bash
python3 -m benchmarks.generate_pareto_curves
```
