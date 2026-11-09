# Phase 5 Final Integration Report

## 1. Executive Summary

Phase 5 is complete on branch `member-2-retrieval`. The ingestion pipeline, real PostgreSQL hybrid retrieval service, LightRAG graph adapter, unified graph-aware retrieval API, evaluation harness, benchmark utility, live end-to-end test, and documentation updates are in place.

The full retrieval suite passes with the live database enabled: **104 passed, 2 skipped in 20.18s**. The two skips are the opt-in real-model confidence test and the opt-in real LightRAG test. The real-model integration smoke evaluation retrieved the correct labeled chunk at rank 1 for all four cases. Real graph extraction and traversal evidence from Phase 4.3 remains the authoritative real-LLM graph verification.

No commit, push, merge, rebase, or branch switch was performed.

## 2. Scope and Constraints

Phase 5 was limited to Member 2 retrieval integration work:

- Integrate ingestion, hybrid retrieval, and GraphRAG behind one service.
- Add a real evaluation harness with honest metrics.
- Add end-to-end, error-path, and performance tests.
- Update retrieval documentation.
- Produce this report.

The following constraints were preserved:

- Member 1, Member 3, and Member 4 owned modules were not modified.
- Shared interfaces and schemas, including `ChunkEvidence`, were not changed.
- Synchronous `traverse_entities(query, entity_seeds, max_hops=2)` was preserved.
- Member 1 `cross_encoder_logit` semantics were preserved.
- Existing uncommitted work and existing tests/reports were preserved.
- No work from phases after Phase 5 was started.

## 3. Branch and Repository State

- Branch: `member-2-retrieval`
- No commit was created.
- No push was performed.
- No merge, rebase, or branch switch was performed.
- Legacy `retrieval/pg_hybrid_search.py` remains untouched and is still treated as a simulation.
- Legacy `retrieval/recall_ablation.py` and `retrieval/recall_ablation_results.json` remain untouched.

## 4. Architecture Audit Findings

The audit confirmed that `retrieval/service.py` is the real unified retrieval service. It uses:

- PostgreSQL 17 with pgvector for dense retrieval.
- ParadeDB `pg_search` for BM25.
- Reciprocal Rank Fusion with `k=60`.
- BGE reranker for cross-encoder logit margins.
- BGE-M3 embeddings with 1024 dimensions.

The audit also confirmed that `retrieval/pg_hybrid_search.py` is a legacy in-memory simulation and must not be used as the production integration path. The README was corrected to state this explicitly.

The audit found that the legacy `recall_ablation.py` output is synthetic and must not be cited as Phase 5 empirical evidence. It was left untouched.

## 5. Unified Retrieval API

`RetrievalService.retrieve(...)` and `RetrievalService.retrieve_chunks(...)` remain the primary hybrid retrieval paths. Their contracts are unchanged.

`HybridGraphRetrievalResult` was added with:

- `query`
- `workspace_id`
- `chunks`
- `graph`
- `graph_error`
- `evidence` property returning `List[ChunkEvidence]`
- `graph_context` property returning the graph context string

This preserves Member 1 compatibility while adding graph context without breaking existing callers.

## 6. Graph Integration API

`RetrievalService.retrieve_with_graph(...)` was added. It accepts:

- `query`
- `top_k`
- `workspace_id`
- optional `entity_seeds`
- optional `graph_engine`
- optional `max_hops`

It runs hybrid retrieval and graph traversal concurrently. Hybrid results are never discarded because of a graph failure. Graph errors, timeouts, and workspace mismatches are returned through `graph_error`.

When no graph engine is supplied, the service can own a per-workspace `LightRAGGraphEngine`. Owned engines are closed by `RetrievalService.close()`.

## 7. Configuration Changes

`RetrievalSettings` gained:

- `graph_traversal_timeout_seconds: float = 120.0`
- Environment variable: `RETRIEVAL_GRAPH_TRAVERSAL_TIMEOUT`
- Validation: the value must be greater than zero.

The existing graph LLM configuration is preserved:

- `RETRIEVAL_GRAPH_LLM_MODEL`
- `RETRIEVAL_GRAPH_LLM_BASE_URL`
- `RETRIEVAL_GRAPH_LLM_API_KEY`
- `RETRIEVAL_GRAPH_LLM_TIMEOUT`
- `RETRIEVAL_GRAPH_LLM_MAX_ASYNC`

API-key fallback order is `RETRIEVAL_GRAPH_LLM_API_KEY`, then `GROQ_API_KEY`, then `OPENAI_API_KEY`.

## 8. Reliability and Error Handling

The graph-aware retrieval path handles:

- Graph traversal failure: returns hybrid chunks with a sanitized `graph_error`.
- Graph traversal timeout: returns hybrid chunks with a timeout `graph_error`.
- Workspace mismatch: rejects graph context and reports the mismatch.
- Empty entity seeds: returns hybrid chunks with an empty graph.
- Invalid `max_hops`: raises an input error before retrieval.
- Blank query: raises an input error before retrieval.

Error messages are sanitized by redacting `sk-...` and `Bearer ...` values and truncating long messages. API keys are not logged or included in reports.

## 9. Evaluation Metrics

`retrieval/evaluation/metrics.py` implements real ranking metrics:

- `recall_at_k`
- `precision_at_k`
- `hit_at_k`
- `reciprocal_rank_at_k`
- `ndcg_at_k`
- `evaluate_ranking`

Metrics are computed from actual returned chunk IDs and labeled relevance IDs. No synthetic scores are substituted.

## 10. Evaluation Harness

`retrieval/evaluation/harness.py` provides:

- `RetrievalEvalCase`
- `RetrievalEvalReport`
- `evaluate_retrieval`
- `evaluate_service`
- `load_eval_cases`
- `write_eval_report`

The harness supports JSON array and JSONL case files. Per-case failures are recorded without inventing scores. A failed case does not corrupt successful cases.

## 11. Benchmark Utility

`retrieval/evaluation/benchmark.py` provides `benchmark_retriever`, which reports:

- iteration count
- success and failure counts
- mean latency
- p50 latency
- p95 latency
- minimum and maximum latency
- peak traced Python memory
- individual latencies
- sanitized errors

The benchmark uses `tracemalloc`. Model weights loaded before tracing are excluded from peak traced memory, so model-load time must be reported separately.

## 12. Test Coverage Added

New test files:

- `tests/test_retrieval_graph_integration.py`
- `tests/test_retrieval_evaluation.py`
- `tests/test_retrieval_phase5_e2e.py`

Coverage includes:

- Graph-aware retrieval success.
- Graph failure and timeout behavior.
- Workspace mismatch rejection.
- Empty seeds and invalid inputs.
- Real `LightRAGGraphEngine` integration with an injected fake LightRAG client.
- Hand-computed metric checks.
- Cutoffs, empty relevance sets, duplicate IDs, async evaluation, and per-case failures.
- JSON and JSONL case loading.
- Benchmark latency and memory reporting.
- Live ingestion, duplicate detection, pgvector, BM25, RRF, reranking, graph integration, and workspace isolation.

## 13. Full Test Suite Results

Command:

```powershell
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:5432/calibrag'
python -m pytest -q
```

Result:

```text
104 passed, 2 skipped in 20.18s
```

The two skips are:

```text
tests/test_retrieval_confidence_live.py: RETRIEVAL_RUN_REAL_MODELS is not set to 1
tests/test_retrieval_lightrag_real.py: set LIGHTRAG_REAL_TESTS=1 to run real LightRAG verification
```

`python -m compileall -q retrieval tests` completed with no errors.

The targeted graph integration suite also passed independently:

```text
7 passed in 1.80s
```

## 14. Live End-to-End Test

`tests/test_retrieval_phase5_e2e.py` is an opt-in live test gated by `RETRIEVAL_TEST_DATABASE_URL`.

It exercises:

- real database connection and health check
- migrations
- real ingestion pipeline
- duplicate detection
- document and chunk persistence
- real pgvector and BM25 indexing
- RRF fusion
- deterministic fake reranker
- real graph integration path with a fake graph engine
- workspace isolation

The test passed independently in **2.12s** and is included in the full suite.

## 15. Real-Model Evaluation Setup

A real-model smoke evaluation was run against the live PostgreSQL database using:

- real BGE-M3 embeddings
- real BGE reranker
- real ingestion pipeline
- real pgvector and ParadeDB BM25 retrieval
- real RRF fusion
- four labeled documents
- one relevant chunk per case

The corpus contained four short documents covering:

- defect tolerance for `EV-Pack-V2`
- the lead architect for `BMS-Controller-v4`
- late-delivery penalties
- factory cafeteria location

This is an integration smoke test, not a retrieval-quality benchmark.

## 16. Real-Model Evaluation Results

The harness evaluated four labeled cases with `k_values=(1, 3, 5)`.

Results:

| Metric | @1 | @3 | @5 |
|---|---:|---:|---:|
| Recall | 1.000 | 1.000 | 1.000 |
| Precision | 1.000 | 0.333 | 0.200 |
| Hit | 1.000 | 1.000 | 1.000 |
| nDCG | 1.000 | 1.000 | 1.000 |
| MRR | 1.000 | 1.000 | 1.000 |

All four cases retrieved the labeled relevant chunk at rank 1. There were no case failures.

The perfect rank-1 results are expected for a four-document corpus and must not be presented as evidence of production retrieval quality.

## 17. Real-Model Benchmark Results

The benchmark ran one warmup plus three measured retrieval calls:

| Field | Value |
|---|---:|
| Iterations | 3 |
| Successes | 3 |
| Failures | 0 |
| Mean | 20,879.12 ms |
| p50 | 21,631.36 ms |
| p95 | 22,114.998 ms |
| Min | 18,890.99 ms |
| Max | 22,114.998 ms |
| Peak traced memory | 0.1985 MB |

Measured latencies:

```text
18890.99 ms
21631.36 ms
22114.998 ms
```

These are CPU-side real-model latencies and include BGE-M3 query embedding plus BGE reranking. The peak traced memory excludes model weights loaded before tracing.

## 18. Graph Integration Verification

Graph integration was verified with the real `LightRAGGraphEngine` and an injected fake LightRAG client. This validates the actual adapter path without requiring an external LLM call during the unit test.

The test confirms:

- the service calls the real graph adapter
- graph nodes and edges are normalized correctly
- `Graph Link: VoltCore -> [RELATED_TO] -> EV-Pack-V2` appears in `graph_context`
- the LightRAG client is initialized and finalized
- hybrid evidence remains available with graph context

## 19. Real Graph Extraction Evidence

Phase 4.3 remains the real-LLM graph extraction verification:

- Groq model probe: `models_status=200`, latency `0.463s`, 11 models available.
- Groq chat probe: `chat_status=200`, latency `0.190s`, non-empty `OK`.
- Workspace A: 1 document indexed, warmup `44.102s`, extraction `49.497s`, 6 nodes and 4 edges.
- Workspace B: 1 document indexed, extraction `5.911s`, 5 nodes and 4 edges.
- Reload: `0.0247s`, 6 nodes and 4 edges.
- All six query modes succeeded: local, global, hybrid, naive, mix, and bypass.
- Duplicate entity forms `PostgreSQL` and `Postgresql` were observed and are a known extraction-quality issue.

## 20. Workspace Isolation Verification

Phase 4.3 verified cross-workspace graph isolation:

```text
a_contains_b_entities = []
b_contains_a_entities = []
```

The Phase 5 live end-to-end test also verifies retrieval workspace isolation.

## 21. Legacy Artifact Audit

The audit found that:

- `retrieval/pg_hybrid_search.py` is a legacy simulation.
- `retrieval/recall_ablation.py` and `retrieval/recall_ablation_results.json` contain hard-coded synthetic metrics.

These artifacts were not modified and were not used for any Phase 5 claim. The README now states that they must not be cited as empirical Phase 5 results.

## 22. PostgreSQL LightRAG Storage Status

PostgreSQL-backed LightRAG storage remains unvalidated end-to-end. Phase 4.3 and Phase 5 verification used file-backed GraphML storage at:

```text
<RETRIEVAL_GRAPH_STORAGE_ROOT>/<workspace_id>/graph_chunk_entity_relation.graphml
```

Storage class names remain configurable, but configuration alone is not evidence of a working PostgreSQL backend. This is an explicit Phase 5 limitation.

## 23. Groq/Qwen Reliability Observations

Phase 4.3 used:

```text
RETRIEVAL_GRAPH_LLM_MODEL=qwen/qwen3.8-27b
RETRIEVAL_GRAPH_LLM_BASE_URL=https://api.groq.com/openai/v1
```

Groq intermittently returned `429` ITPM and `503` over-capacity responses. Retries succeeded in the final Phase 4.3 run. Phase 5 did not run a new real graph extraction because the Phase 4.3 evidence already validates that path and Groq availability is intermittent.

No API key was printed, logged, or included in this report.

## 24. Documentation Updates

`retrieval/README.md` was updated with:

- corrected architecture ownership
- warning that `pg_hybrid_search.py` is legacy
- unified hybrid and graph retrieval examples
- evaluation harness usage
- reliability and performance behavior
- updated LightRAG section numbering
- test commands
- environment variable reference
- known limitations
- warning that legacy recall ablation output is synthetic

## 25. Files Changed

Phase 5 changes:

- `retrieval/config.py`
- `retrieval/service.py`
- `retrieval/__init__.py`
- `retrieval/evaluation/__init__.py`
- `retrieval/evaluation/metrics.py`
- `retrieval/evaluation/harness.py`
- `retrieval/evaluation/benchmark.py`
- `tests/test_retrieval_graph_integration.py`
- `tests/test_retrieval_evaluation.py`
- `tests/test_retrieval_phase5_e2e.py`
- `retrieval/README.md`
- `retrieval/reports/PHASE_5_FINAL_INTEGRATION_REPORT.md`

No Member 1, Member 3, Member 4, or shared schema files were changed.

## 26. Known Limitations

- The real-model evaluation used only four short documents and four cases.
- Perfect rank-1 metrics on that corpus are not a production quality claim.
- PostgreSQL-backed LightRAG storage was not validated.
- Real graph extraction depends on an external LLM and may hit Groq rate limits or capacity errors.
- The benchmark measures CPU-side model inference and excludes model-load time from traced peak memory.
- The legacy recall ablation artifacts are synthetic and remain unused.
- A single duplicate entity form issue was observed in Phase 4.3 extraction.

## 27. Recommendations

1. Run the evaluation harness on a larger labeled corpus with realistic distractors before making retrieval-quality claims.
2. Validate PostgreSQL-backed LightRAG storage in a dedicated integration environment.
3. Add retry/backoff telemetry for Groq `429` and `503` responses.
4. Normalize entity aliases such as `PostgreSQL` and `Postgresql` before graph traversal.
5. Add a CLI entry point for the evaluation harness if repeated benchmark runs are needed.
6. Report real-model latency and model-load time separately in future performance reports.

## 28. Final Acceptance Statement

Phase 5 integration is complete within the stated scope. The unified hybrid and graph-aware retrieval API, real evaluation harness, benchmark utility, live end-to-end test, error-path tests, and documentation are implemented and verified.

The work is accepted with the documented limitations: the real-model evaluation is a small smoke test, PostgreSQL-backed LightRAG storage remains unvalidated, and real graph extraction depends on an intermittent external provider.
