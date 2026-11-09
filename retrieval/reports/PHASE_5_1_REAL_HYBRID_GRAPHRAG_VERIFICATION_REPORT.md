# Phase 5.1 Real Hybrid + GraphRAG Verification Report

## 1. Executive Summary

Phase 5.1 is complete on branch `member-2-retrieval`. The controlled real end-to-end test passed with real Docling parsing, real BGE-M3 embeddings, live PostgreSQL/pgvector, live ParadeDB BM25, real RRF fusion, real BGE reranking, real Groq/Qwen LightRAG extraction, the actual `LightRAGGraphEngine`, and `RetrievalService.retrieve_with_graph(...)`.

Overall status: **PASS**.

The main Phase 5.1 test passed in **271.80s**. The full retrieval suite passed with the live database enabled: **104 passed, 4 skipped in 38.11s**. The four skips are the existing opt-in real-model confidence test, the existing opt-in real LightRAG test, and the two new Phase 5.1 opt-in tests when their gates are not enabled.

The real controlled run verified:

- Workspace A answers: Acme Technologies is headquartered in **Pune**; the Fraud Detection project uses **PostgreSQL**.
- Workspace B answers: Beta Systems is headquartered in **Mumbai**; the Inventory Sync project uses **MySQL**.
- Graph context contains the expected entity relationships for both workspaces.
- Cross-workspace chunk and graph leakage was not observed.
- LightRAG GraphML persistence and reload worked.
- A real Groq `429` was observed and retried successfully.
- `PGTableGraphStorage` worked in a disposable isolated PostgreSQL database and persisted/reloaded correctly.

No commit, push, merge, rebase, or branch switch was performed.

## 2. Scope and Constraints

Phase 5.1 was limited to real end-to-end verification of Member 2 hybrid retrieval and GraphRAG:

- Use real ingestion, real embeddings, real PostgreSQL/pgvector, real BM25, real fusion, real reranking, and real LightRAG extraction.
- Use the actual `LightRAGGraphEngine` and `RetrievalService.retrieve_with_graph(...)`.
- Use two isolated workspaces with unambiguous controlled documents.
- Verify final `ChunkEvidence`, graph context, workspace isolation, persistence, and failure behavior.
- Investigate PostgreSQL-backed LightRAG storage without changing production storage configuration.
- Add a gated real integration test and produce this report.

Constraints preserved:

- Member 1, Member 3, and Member 4 owned modules were not modified.
- `core/schemas.py` and `ChunkEvidence` were not changed.
- Synchronous `traverse_entities(query, entity_seeds, max_hops=2)` was preserved.
- Member 1 `cross_encoder_logit` semantics were preserved.
- Production LightRAG storage remains `NetworkXStorage`.
- Existing uncommitted work, tests, and reports were preserved.
- `ainsert_custom_kg()` was not used as proof of real extraction.
- No commit or push was performed.

## 3. Branch and Repository State

- Branch: `member-2-retrieval`
- No commit created.
- No push performed.
- No merge, rebase, or branch switch performed.
- Phase 5.1 added two opt-in tests and this report.
- No production retrieval code was changed during Phase 5.1.
- Existing legacy artifacts remain untouched:
  - `retrieval/pg_hybrid_search.py`
  - `retrieval/recall_ablation.py`
  - `retrieval/recall_ablation_results.json`

## 4. Exact Environment

| Item | Value |
|---|---|
| Date | 2026-11-09 |
| Platform | Windows `win32`, PowerShell 5.1 |
| Branch | `member-2-retrieval` |
| Python | 3.11.6 |
| Virtualenv | `C:\Users\LENOVO\AppData\Local\Temp\opencode\calibrag-phase2-venv` |
| LightRAG | 1.5.7 |
| Docling | 2.148.0 |
| FlagEmbedding | 1.4.2 |
| sentence-transformers | 6.2.0 |
| PyTorch | 2.15.0+cpu |
| asyncpg | 0.32.0 |
| NumPy | 2.4.6 |
| PostgreSQL container | `calibrag-retrieval-postgres-1` |
| PostgreSQL image | `paradedb/paradedb:0.26.5-pg17` |
| PostgreSQL status | Up 3 days, healthy |
| PostgreSQL version | 17.11 |
| `pg_search` | 0.26.5 |
| `vector` | 0.8.6 |
| Test database | local `calibrag` database, disposable test workspaces |
| Embedding model | `BAAI/bge-m3`, 1024 dimensions |
| Reranker model | `BAAI/bge-reranker-v2-m3` |
| Graph LLM | `qwen/qwen3.8-27b` via Groq OpenAI-compatible endpoint |
| Production graph storage | `NetworkXStorage` GraphML |

Real credentials were loaded from `retrieval/.env`. Secret values are not reproduced in this report.

## 5. Real vs Mocked vs Skipped Components

| Component | Phase 5.1 Main Test |
|---|---|
| Docling parser | Real |
| Chunking | Real |
| PostgreSQL schema/migrations | Real |
| BGE-M3 embeddings | Real |
| pgvector retrieval | Real |
| ParadeDB BM25 | Real |
| RRF fusion | Real |
| BGE reranker | Real |
| Groq/Qwen LightRAG extraction | Real |
| LightRAG graph storage | Real GraphML `NetworkXStorage` |
| `LightRAGGraphEngine` | Real |
| `retrieve_with_graph` | Real |
| Final `ChunkEvidence` | Real |
| Graph failure/timeout tests | Deterministic test doubles, separate from the main real flow |
| PGTableGraphStorage test | Real PostgreSQL, isolated disposable database |
| 503 response | Not induced; retry path inspected |
| Existing opt-in real-model confidence test | Skipped by default |
| Existing opt-in real LightRAG test | Skipped by default |
| New Phase 5.1 real E2E test | Skipped unless `RETRIEVAL_PHASE5_1_REAL_TESTS=1` |
| New PGTable test | Skipped unless `RETRIEVAL_PHASE5_1_PGTABLE_TESTS=1` |

The main integration proof did not substitute fake graph engines, fake LightRAG clients, fake embeddings, or fake rerankers.

## 6. Code Changes

Phase 5.1 added:

- `tests/test_retrieval_phase5_1_real_e2e.py`
  - Gated by `RETRIEVAL_PHASE5_1_REAL_TESTS=1`, `RETRIEVAL_TEST_DATABASE_URL`, and the presence of `retrieval/.env`.
  - Uses real Docling, BGE-M3, PostgreSQL/pgvector, BM25, RRF, BGE reranker, Groq/Qwen LightRAG extraction, actual `LightRAGGraphEngine`, and `retrieve_with_graph`.
  - Covers two workspaces, cross-workspace isolation, GraphML reload, empty-workspace behavior, and final `ChunkEvidence`.
- `tests/test_retrieval_lightrag_pgtable.py`
  - Gated by `RETRIEVAL_PHASE5_1_PGTABLE_TESTS=1` and `RETRIEVAL_TEST_DATABASE_URL`.
  - Creates a disposable PostgreSQL database, verifies `PGTableGraphStorage` persistence and workspace isolation, then drops the database.
- `retrieval/reports/PHASE_5_1_REAL_HYBRID_GRAPHRAG_VERIFICATION_REPORT.md`
  - This report.

No production code, shared schema, or production storage configuration was changed.

## 7. Ingestion Results

Controlled Workspace A document:

- Acme Technologies is a private software company.
- Alice works at Acme Technologies as the engineering lead.
- Acme Technologies is headquartered in Pune, India.
- Alice leads the Fraud Detection project.
- The Fraud Detection project uses PostgreSQL as its primary database.

Controlled Workspace B document:

- Beta Systems is a logistics software company.
- Bob works at Beta Systems as the product lead.
- Beta Systems is headquartered in Mumbai, India.
- Bob leads the Inventory Sync project.
- The Inventory Sync project uses MySQL as its primary database.

A one-chunk distractor document was also ingested into each workspace to exercise multi-candidate vector/BM25/RRF/reranker behavior without changing the controlled facts.

| Ingestion stage | Workspace A | Workspace B |
|---|---:|---:|
| Controlled document chunks | 1 | 1 |
| Distractor chunks | 1 | 1 |
| Docling used | Yes | Yes |
| Page count | 1 | 1 |
| Controlled ingestion latency | 23.4465s | 0.1270s |
| Distractor ingestion latency | 0.0959s | 0.0866s |
| Controlled embedding/index latency | 36.1987s | 2.9695s |
| Distractor embedding/index latency | 0.8812s | 0.9744s |

The first ingestion and embedding calls include Docling and BGE-M3 model-load cost. Later calls are much faster.

## 8. Dense Retrieval Results

Real BGE-M3 embeddings and live pgvector retrieval were used.

| Workspace | Vector candidates | Top chunk | Latency |
|---|---:|---|---:|
| A | 2 | Controlled Acme document | 0.8253s |
| B | 2 | Controlled Beta document | 0.7699s |

The controlled document ranked first in both workspaces. The distractor ranked second.

## 9. BM25 Results

Live ParadeDB BM25 retrieval was used against `document_chunks` with the `document_chunks_bm25_idx` index.

| Workspace | BM25 candidates | Top chunk | Latency |
|---|---:|---|---:|
| A | 2 | Controlled Acme document | 0.0829s |
| B | 2 | Controlled Beta document | 0.0061s |

The controlled document ranked first in both workspaces.

## 10. RRF Fusion Results

Real `reciprocal_rank_fusion(...)` was run with the configured `rrf_k=60`.

| Workspace | Vector candidates | BM25 candidates | Fused candidates |
|---|---:|---:|---:|
| A | 2 | 2 | 2 |
| B | 2 | 2 | 2 |

The final evidence metadata reported `retrieval_sources: ["vector", "bm25"]` for the top controlled chunk in both workspaces, confirming that both retrieval paths contributed to the fused result.

## 11. BGE Reranker Results

Real `BAAI/bge-reranker-v2-m3` reranking produced the final `cross_encoder_logit` values.

| Workspace | Controlled chunk logit | Distractor logit | Final order |
|---|---:|---:|---|
| A | 10.194250106811523 | -7.787367820739746 | Controlled first |
| B | 9.612455368041992 | -8.947998046875 | Controlled first |

The final `ChunkEvidence` objects contained real float `cross_encoder_logit` values, and the controlled chunk had the higher logit in both workspaces.

## 12. LightRAG Graph Extraction Results

Real Groq/Qwen extraction was performed by the actual `LightRAGGraphEngine.aindex_chunks(...)`. No custom knowledge graph was inserted.

| Workspace | Indexed documents | Nodes | Edges | Extraction latency | Track ID |
|---|---:|---:|---:|---:|---|
| A | 1 | 5 | 5 | 19.7508s | `insert_20261109_120015_8c4ac31b` |
| B | 1 | 7 | 8 | 14.0732s | `insert_20261109_120130_fca2f63d` |

Workspace A extracted entities:

- Acme Technologies
- Alice
- Fraud Detection Project
- PostgreSQL
- Pune

Workspace A extracted relationships included:

- Acme Technologies -> Alice
- Acme Technologies -> Fraud Detection Project
- Acme Technologies -> Pune
- Alice -> Fraud Detection Project
- Fraud Detection Project -> PostgreSQL

Workspace B extracted entities:

- Beta Systems
- Bob
- Inventory Sync Project
- Mumbai
- MySQL
- India
- Logistics Platform Team

Workspace B extracted relationships included:

- Beta Systems -> Bob
- Beta Systems -> Inventory Sync Project
- Beta Systems -> Mumbai
- Bob -> Inventory Sync Project
- Inventory Sync Project -> MySQL
- Beta Systems -> Logistics Platform Team
- Bob -> Logistics Platform Team
- India -> Mumbai

LightRAG persisted the graphs as GraphML under the per-workspace graph storage root.

## 13. Hybrid + Graph Integration Results

`RetrievalService.retrieve_with_graph(...)` was executed with the real graph engines and real hybrid retrieval.

| Workspace | Evidence count | Graph nodes | Graph edges | Graph error | End-to-end latency |
|---|---:|---:|---:|---|---:|
| A | 2 | 5 | 5 | None | 29.0126s |
| B | 2 | 7 | 8 | None | 28.4959s |

Workspace A final answer-bearing evidence contained:

- Acme Technologies is headquartered in Pune.
- The Fraud Detection project uses PostgreSQL.

Workspace B final answer-bearing evidence contained:

- Beta Systems is headquartered in Mumbai.
- The Inventory Sync project uses MySQL.

Workspace A graph context included:

- Acme Technologies -> Alice
- Acme Technologies -> Fraud Detection Project
- Acme Technologies -> Pune
- Alice -> Fraud Detection Project
- Fraud Detection Project -> PostgreSQL

Workspace B graph context included the expected Beta Systems, Bob, Inventory Sync, Mumbai, and MySQL relationships.

The final result contained real `ChunkEvidence` objects with real reranker logits, real content, real source metadata, and workspace-scoped graph context.

GraphML reload verification:

| Workspace | Reload traversal latency | Persisted nodes | Persisted edges |
|---|---:|---:|---:|
| A | 1.0257s | 5 | 5 |
| B | 0.3744s | 7 | 8 |

Reloaded traversal returned the same workspaces and expected entities/relationships.

## 14. Workspace Isolation Results

Workspace A evidence contained no Beta Systems, Bob, Mumbai, or MySQL content. Workspace B evidence contained no Acme Technologies, Alice, Pune, or PostgreSQL content.

| Isolation check | Result |
|---|---|
| A evidence contains B facts | No |
| B evidence contains A facts | No |
| A graph contains B entities | No |
| B graph contains A entities | No |
| A/B evidence chunk overlap | None |
| Graph traversal workspace IDs | Correct for each workspace |
| Vector/BM25 workspace filter | Correct for each workspace |

The real test also confirmed that `ChunkEvidence` from Workspace A and Workspace B had no shared chunk IDs.

## 15. Failure Handling and Retry Behavior

Real and deterministic failure checks:

| Scenario | Behavior | Evidence |
|---|---|---|
| Graph backend raises an exception | Hybrid evidence is preserved; sanitized `graph_error` returned | `tests/test_retrieval_graph_integration.py` |
| Graph traversal exceeds timeout | Hybrid evidence is preserved; timeout reported | `tests/test_retrieval_graph_integration.py` |
| Graph returns a different workspace | Graph context rejected; mismatch reported | `tests/test_retrieval_graph_integration.py` |
| Empty entity seeds | Graph skipped; hybrid evidence preserved | `tests/test_retrieval_graph_integration.py` |
| Invalid `max_hops` | `RetrievalInputError` raised | `tests/test_retrieval_graph_integration.py` |
| Blank query | Empty result; no graph traversal | `tests/test_retrieval_graph_integration.py` |
| Empty vector/BM25 workspace | 0 chunks, 0 evidence, 0 graph nodes, no fabricated result | Real Phase 5.1 test |
| Groq `429` rate limit | Retried and completed successfully | Real Phase 5.1 run |
| Groq `503` | Not induced; transient 5xx retry path inspected | LightRAG 1.5.7 retry policy |

The first real E2E run observed Groq `429 Too Many Requests` during Workspace B extraction. LightRAG retried and the extraction completed. LightRAG 1.5.7 uses a tenacity retry policy with 3 attempts and exponential backoff, retrying rate limits, transient 5xx responses, connection errors, timeouts, and invalid responses (`lightrag/llm/openai.py:288-301`).

Targeted graph failure/timeout suite result: **7 passed in 2.25s**.

Empty-workspace real result:

- chunks: 0
- evidence: 0
- graph nodes: 0
- graph_error: `None`

## 16. PostgreSQL-Backed LightRAG Investigation

LightRAG 1.5.7 provides:

- `PGKVStorage`
- `PGVectorStorage`
- `PGDocStatusStorage`
- `PGGraphStorage`
- `PGTableGraphStorage`

Findings:

- `PGGraphStorage` depends on Apache AGE/Cypher and is not practical with the current ParadeDB image without adding AGE.
- `PGTableGraphStorage` uses plain PostgreSQL tables:
  - `lightrag_graph_nodes`
  - `lightrag_graph_edges`
- It requires no Apache AGE and no pgvector for graph storage.
- It uses JSONB properties and B-tree indexes.
- It reads `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DATABASE`, and `POSTGRES_WORKSPACE`.
- Direct use requires LightRAG shared-data initialization via `initialize_share_data()`.

Feasibility test:

- Created a disposable isolated PostgreSQL database.
- Initialized two `PGTableGraphStorage` instances in separate workspaces.
- Wrote nodes and edges.
- Queried knowledge graphs.
- Finalized and recreated the storages.
- Verified persistence and reload.
- Verified workspace isolation.
- Dropped the data and database.

Result: **PASS**, `1 passed in 7.64s`.

| Check | Result |
|---|---|
| Workspace A nodes before reload | Acme Technologies, Alice, Pune |
| Workspace B nodes before reload | Beta Systems, Bob |
| Workspace A edges before reload | 2 |
| Workspace B edges before reload | 1 |
| Cross-workspace nodes | None |
| Reload match | Yes |
| Labels after drop | Empty for both workspaces |
| Production storage changed | No |

Production graph storage remains `NetworkXStorage`. `PGTableGraphStorage` is feasible and was verified in isolation, but was not enabled for production.

## 17. Performance Measurements

| Stage | Workspace A | Workspace B |
|---|---:|---:|
| Database connect | 0.3565s | shared |
| Controlled ingestion | 23.4465s | 0.1270s |
| Distractor ingestion | 0.0959s | 0.0866s |
| Controlled embedding/index | 36.1987s | 2.9695s |
| Distractor embedding/index | 0.8812s | 0.9744s |
| Vector retrieval | 0.8253s | 0.7699s |
| BM25 retrieval | 0.0829s | 0.0061s |
| Graph extraction | 19.7508s | 14.0732s |
| Graph reload traversal | 1.0257s | 0.3744s |
| End-to-end hybrid + graph retrieval | 29.0126s | 28.4959s |
| Empty workspace retrieval | 10.9181s | shared |

The 65s pause between Workspace A and Workspace B graph extraction was intentional to stay below the Groq input-tokens-per-minute limit. It is not part of production retrieval latency.

The first model load dominates Docling and BGE-M3 latency. Warm graph traversal is sub-second to roughly one second.

## 18. Security and Secret Handling

- API keys were loaded from `retrieval/.env`.
- API keys were never printed, logged, or included in this report.
- Only a boolean key-presence check was performed during environment inspection.
- Database credentials are local disposable test credentials.
- Commands in this report use a redacted test DSN placeholder.
- Graph error messages are sanitized by `RetrievalService`.
- No secrets were written to test files, reports, or temporary evidence.
- Observed Groq `429` output contained provider metadata; the organization identifier is intentionally omitted from this report.

## 19. Known Limitations

- Multi-chunk LightRAG indexing limitation: when multiple chunks share the same fallback `doc_title` file path, LightRAG 1.5.7 filename deduplication can process only the first chunk. The controlled Phase 5.1 documents were intentionally one chunk each so the real extraction path could be verified without this limitation. A production fix would require unique per-chunk source paths or a deliberate document-level extraction strategy.
- Groq `qwen/qwen3.8-27b` has a 7000 input-token-per-minute on-demand limit. The real test pauses 65s between workspace extractions to avoid repeated `429` responses.
- LightRAG extraction output is model-dependent. In this run, extracted edge types were `RELATED_TO`; semantic meaning was carried in edge descriptions.
- `PGTableGraphStorage` was verified directly in isolation, not integrated into the production `LightRAGGraphEngine` factory.
- Groq `503` was not induced. The retry path was inspected rather than exercised against a live 503 response.
- The controlled verification used two workspaces and one chunk per controlled document; it is not a large-corpus benchmark.
- Legacy simulation and synthetic ablation artifacts remain in the repository and must not be cited as Phase 5.1 evidence.

## 20. Commands and Test Evidence

Full suite with live database and Phase 5.1 gates disabled:

```powershell
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:***@localhost:5432/calibrag'
$env:RETRIEVAL_PHASE5_1_REAL_TESTS='0'
$env:RETRIEVAL_PHASE5_1_PGTABLE_TESTS='0'
& $py -m pytest -q
```

Result:

```text
104 passed, 4 skipped in 38.11s
```

Skip reasons:

```text
tests/test_retrieval_confidence_live.py: RETRIEVAL_RUN_REAL_MODELS is not set to 1
tests/test_retrieval_lightrag_pgtable.py: RETRIEVAL_PHASE5_1_PGTABLE_TESTS and RETRIEVAL_TEST_DATABASE_URL not enabled
tests/test_retrieval_lightrag_real.py: LIGHTRAG_REAL_TESTS is not set to 1
tests/test_retrieval_phase5_1_real_e2e.py: RETRIEVAL_PHASE5_1_REAL_TESTS and RETRIEVAL_TEST_DATABASE_URL not enabled
```

Real Phase 5.1 end-to-end test:

```powershell
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:***@localhost:5432/calibrag'
$env:RETRIEVAL_PHASE5_1_REAL_TESTS='1'
$env:RETRIEVAL_PHASE5_1_GRAPH_PAUSE_SECONDS='65'
& $py -m pytest tests/test_retrieval_phase5_1_real_e2e.py -q
```

Result:

```text
1 passed in 271.80s (0:04:31)
```

PostgreSQL-native LightRAG storage test:

```powershell
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:***@localhost:5432/calibrag'
$env:RETRIEVAL_PHASE5_1_PGTABLE_TESTS='1'
& $py -m pytest tests/test_retrieval_lightrag_pgtable.py -q
```

Result:

```text
1 passed in 7.64s
```

Graph failure/timeout suite:

```powershell
& $py -m pytest tests/test_retrieval_graph_integration.py -q
```

Result:

```text
7 passed in 2.25s
```

Compilation check:

```powershell
& $py -m compileall -q retrieval tests
```

Result:

```text
compileall ok
```

Live database check:

```powershell
docker ps --filter "name=calibrag-retrieval-postgres-1"
```

Result:

```text
calibrag-retrieval-postgres-1 | paradedb/paradedb:0.26.5-pg17 | Up 3 days (healthy)
```

## 21. Final Acceptance Status

Phase 5.1 acceptance status: **PASS**.

Accepted evidence:

- Real Docling ingestion completed.
- Real BGE-M3 embeddings and pgvector retrieval completed.
- Real ParadeDB BM25 retrieval completed.
- Real RRF fusion completed.
- Real BGE reranking completed.
- Real Groq/Qwen LightRAG extraction completed through the actual `LightRAGGraphEngine`.
- `RetrievalService.retrieve_with_graph(...)` returned hybrid evidence plus workspace-scoped graph context.
- Final `ChunkEvidence` contained real reranker logits and metadata.
- Workspace A and Workspace B remained isolated.
- GraphML persistence and reload worked.
- Empty vector/BM25 workspace returned no fabricated evidence.
- A real Groq `429` was observed and successfully retried.
- `PGTableGraphStorage` feasibility was verified in an isolated disposable PostgreSQL database.
- Full suite passed with the live database enabled.
- No secrets were exposed.
- No commit or push was performed.

Phase 5.1 is ready for review.
