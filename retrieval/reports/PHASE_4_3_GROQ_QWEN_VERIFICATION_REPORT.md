# Phase 4.3 - Groq/Qwen Real LightRAG Verification Report

- Date: 2026-11-08
- Branch: `member-2-retrieval`
- Status: **PASS for Phase 4.3 scope**
- Supersedes: blocked real-extraction status in `PHASE_4_2_REAL_EXTRACTION_REPORT.md`
- Commit/push: **None**
- Phase 5: **Not started**

## 1. Objective

Verify that the Member 2 retrieval layer works end-to-end with Groq's OpenAI-compatible API using `qwen/qwen3.8-27b`, including:

- real LightRAG entity and relationship extraction
- GraphML persistence and reload
- real graph queries
- all six LightRAG query modes
- workspace isolation
- graceful behavior under provider failures

## 2. Scope and Constraints

Respected constraints:

- Worked only on `member-2-retrieval`.
- Did not merge, rebase, commit, or push.
- Did not modify Member 1/3/4 modules, shared schemas, BGE-M3 configuration, pgvector indexes/dimensions, ParadeDB BM25, RRF, or reranker behavior.
- Preserved synchronous `traverse_entities(query, entity_seeds, max_hops=2)` for Member 3.
- Used normal `ainsert()`/`aindex_text()`, not `ainsert_custom_kg()`.
- Did not start Phase 5.
- Did not print, log, or report the API key.

## 3. Working Tree Status

Branch confirmed:

```text
member-2-retrieval
```

Tracked modifications at verification time:

```text
M retrieval/README.md
M retrieval/__init__.py
M retrieval/docling_parser.py
M retrieval/lightrag_indexer.py
M tests/test_retrieval_member2.py
```

Relevant untracked Phase 2-4 files remain present, including `retrieval/config.py`, `retrieval/reports/`, `tests/test_retrieval_groq_config.py`, `tests/test_retrieval_lightrag.py`, and the rest of the existing retrieval implementation. Pre-existing `myenv/` is also untracked and was preserved.

No files were staged. No commits or pushes were made.

## 4. Files Created or Modified

### Created

- `tests/test_retrieval_groq_config.py`
- `retrieval/reports/PHASE_4_3_GROQ_QWEN_VERIFICATION_REPORT.md`

### Modified

- `retrieval/config.py`
  - `RetrievalSettings.from_env()` now resolves the graph LLM key from `RETRIEVAL_GRAPH_LLM_API_KEY`, then `GROQ_API_KEY`, then `OPENAI_API_KEY`.
- `retrieval/lightrag_indexer.py`
  - `_llm_api_key()` now supports the same `GROQ_API_KEY`/`OPENAI_API_KEY` fallback.
  - Missing-credential error text now mentions the fallback variables.
  - LLM configuration validation now requires an API key; a base URL alone is rejected early.
  - Added default-embedder warmup before LightRAG initialization to avoid the BGE-M3 cold-start embedding timeout.
- `retrieval/.env.ingestion.example`
  - Documented `GROQ_API_KEY` as a supported fallback.
- `retrieval/README.md`
  - Added Groq/Qwen configuration and first-use BGE-M3 warmup note.
- `tests/test_retrieval_lightrag.py`
  - Added tests for embedder warmup behavior.

### Local-only secret file

- `retrieval/.env`
  - Contains the Groq API key. Not committed and not reproduced in this report.

## 5. API Configuration

Verified configuration, with credentials redacted:

```text
RETRIEVAL_GRAPH_LLM_MODEL=qwen/qwen3.8-27b
RETRIEVAL_GRAPH_LLM_BASE_URL=https://api.groq.com/openai/v1
RETRIEVAL_GRAPH_LLM_API_KEY=<empty>
GROQ_API_KEY=<configured, value redacted>
RETRIEVAL_GRAPH_LLM_TIMEOUT=60
RETRIEVAL_GRAPH_LLM_MAX_ASYNC=4
```

Key resolution works through the new fallback:

```text
RETRIEVAL_GRAPH_LLM_API_KEY -> GROQ_API_KEY -> OPENAI_API_KEY
```

Configuration validation requires an API key. A base URL alone is rejected with `GraphConfigurationError` instead of failing later inside the OpenAI client.

No unsupported provider-specific parameters were added. LightRAG uses its normal OpenAI-compatible chat completion path.

## 6. Groq Connection Test

Probe script:

```text
C:\Users\LENOVO\AppData\Local\Temp\opencode\groq_probe_openai.py
```

Result:

```text
model=qwen/qwen3.8-27b
base_url=https://api.groq.com/openai/v1
api_key=SET
models_status=200
models_latency_s=0.463
model_count=11
model_available=True
chat_status=200
chat_latency_s=0.190
chat_non_empty=True
chat_content=OK
response_model=qwen/qwen3.8-27b
```

The model list contained `qwen/qwen3.8-27b`, and the chat completion returned a non-empty `OK` response.

Provider compatibility note: a raw `urllib` probe was blocked by Cloudflare (`403 Error 1010`), while the OpenAI-compatible client used by LightRAG succeeded. The implementation therefore uses the normal OpenAI-compatible client path.

## 7. Real Extraction Run

Real verification script:

```text
C:\Users\LENOVO\AppData\Local\Temp\opencode\phase43_real_extraction.py
C:\Users\LENOVO\AppData\Local\Temp\opencode\phase43_resume.py
```

Sanitized result file:

```text
C:\Users\LENOVO\AppData\Local\Temp\opencode\phase43_results.json
```

Controlled workspace A source text:

```text
Alice works at Acme Technologies. Acme Technologies is headquartered in Pune.
Alice leads the Fraud Detection project. The Fraud Detection project uses PostgreSQL.
```

Controlled workspace B source text:

```text
Bob works at Beta Systems. Beta Systems is headquartered in Mumbai.
Bob leads the Inventory Sync project. The Inventory Sync project uses Redis.
```

Workspace A extraction:

```text
workspace_id=phase43a
indexed_documents=1
embedder_warmup_s=44.102158800000325
elapsed_s=49.49686480005039
```

Workspace B extraction:

```text
workspace_id=phase43b
indexed_documents=1
elapsed_s=5.911266200011596
```

Workspace B was extracted after the embedder was already loaded, so its elapsed time was much shorter.

## 8. Real Relationship Extraction Results

### Workspace A

Final persisted graph:

```text
node_count=6
edge_count=4
```

Persisted entities:

```text
Acme Technologies
Alice
Fraud Detection
PostgreSQL
Postgresql
Pune
```

Persisted relationships:

```text
Acme Technologies -> Alice
  Alice works at Acme Technologies.

Acme Technologies -> Pune
  Acme Technologies is headquartered in Pune.

Alice -> Fraud Detection
  Alice leads the Fraud Detection project.

Fraud Detection -> Postgresql
  The Fraud Detection project uses PostgreSQL.
```

All four source facts are present. The extraction produced a casing duplicate (`PostgreSQL` and `Postgresql`) and one additional candidate relation that was not persisted in the final graph. No required fact or relationship is missing.

Fact hit check:

```text
alice=true
acme technologies=true
pune=true
fraud detection=true
postgresql=true
```

### Workspace B

Final persisted graph:

```text
node_count=5
edge_count=4
```

Persisted entities:

```text
Beta Systems
Bob
Inventory Sync Project
Mumbai
Redis
```

Persisted relationships:

```text
Beta Systems -> Mumbai
  Beta Systems is headquartered in Mumbai.

Bob -> Beta Systems
  Bob works at Beta Systems.

Bob -> Inventory Sync Project
  Bob leads the Inventory Sync project.

Inventory Sync Project -> Redis
  The Inventory Sync project uses Redis.
```

All four source facts are present.

Fact hit check:

```text
bob=true
beta systems=true
mumbai=true
inventory sync=true
redis=true
```

## 9. Graph Persistence and Reload

Workspace A graph file:

```text
C:\Users\LENOVO\AppData\Local\Temp\phase43-lightrag-13qfnxhr\phase43a\graph_chunk_entity_relation.graphml
```

Workspace B graph file:

```text
C:\Users\LENOVO\AppData\Local\Temp\phase43-lightrag-13qfnxhr\phase43b\graph_chunk_entity_relation.graphml
```

Both files existed after extraction.

Workspace A reload result:

```text
graph_file_exists=true
reload_latency_s=0.024697399989236146
node_count=6
edge_count=4
fact_hits=all true
```

Reload traversal through `traverse_entities(..., max_hops=2)` returned:

```text
total_nodes_found=6
edge_count=4
workspace_id=phase43a
truncated=false
```

The reloaded graph preserves entity descriptions, relationship descriptions, source IDs, file paths, and workspace identity.

Temporary Phase 4.3 test data was deleted after capture. The temp root no longer exists.

## 10. Real Graph Query Results

All real answers were generated by `qwen/qwen3.8-27b` through LightRAG using `mode="mix"` and explicit high-level/low-level keywords.

### Workspace A answers

Question: `Where is Acme Technologies headquartered?`

```text
Acme Technologies is headquartered in **Pune**. The organization employs Alice, who leads the Fraud Detection project, a project that utilizes PostgreSQL as its software solution.
```

Question: `Which project does Alice lead?`

```text
Alice leads the **Fraud Detection** project. There is no mention of any other projects she leads in the provided knowledge base.
```

Question: `Which database does the Fraud Detection project use?`

```text
Based on the provided information, the Fraud Detection project utilizes **PostgreSQL** as its database. Additionally, the project is led by Alice, who is an employee at Acme Technologies, a company headquartered in Pune.
```

Observed answer latencies:

```text
Where is Acme Technologies headquartered?       1.9943069000146352s
Which project does Alice lead?                  1.0960909000132233s
Which database does the Fraud Detection project use?  1.1669832000043243s
```

### Workspace B answers

Question: `Where is Beta Systems headquartered?`

```text
Based on the provided information, **Beta Systems** is headquartered in **Mumbai**.
```

Question: `Which project does Bob lead?`

```text
Bob leads the **Inventory Sync project**. He is an employee at Beta Systems, which is headquartered in Mumbai, and the project he leads utilizes Redis technology.
```

Question: `Which database does the Inventory Sync project use?`

```text
The Inventory Sync project utilizes **Redis** as its database technology. This project is led by Bob, an employee at Beta Systems, which is headquartered in Mumbai.
```

Observed answer latencies:

```text
Where is Beta Systems headquartered?             5.184397500008345s
Which project does Bob lead?                     9.098906800034456s
Which database does the Inventory Sync project use?  9.29772370000137s
```

All answers are grounded in the controlled documents and match the expected facts.

## 11. All Query Modes

Workspace A was queried through `aquery_data()` in all six modes using explicit keywords.

| Mode | Status | Entities | Relationships | Chunks | References |
| --- | --- | ---: | ---: | ---: | ---: |
| `local` | success | 5 | 4 | 1 | 1 |
| `global` | success | 5 | 4 | 1 | 1 |
| `hybrid` | success | 6 | 4 | 1 | 1 |
| `naive` | success | 0 | 0 | 1 | 1 |
| `mix` | success | 6 | 4 | 1 | 1 |
| `bypass` | success | 0 | 0 | 0 | 0 |

Notes:

- `naive` returned chunk context without graph entities, as expected.
- `bypass` returned no graph or chunk context, as expected for bypass mode.
- All six modes completed without adapter errors.

## 12. Workspace Isolation Results

Workspace A labels:

```text
acme technologies
alice
fraud detection
postgresql
pune
```

Workspace B labels:

```text
beta systems
bob
inventory sync project
mumbai
redis
```

Cross-contamination checks:

```text
a_contains_b_entities=[]
b_contains_a_entities=[]
```

Cross-workspace queries:

- Workspace A queried with Bob/Beta Systems/Inventory Sync/Redis keywords returned only workspace A entities: `Fraud Detection`, `Alice`, `PostgreSQL`, `Acme Technologies`, `Postgresql`, `Pune`.
- Workspace B queried with Alice/Acme Technologies/Fraud Detection/PostgreSQL keywords returned only workspace B entities: `Redis`, `Beta Systems`, `Bob`, `Mumbai`, `Inventory Sync Project`.

Both cross-workspace queries succeeded and returned no entities from the other workspace. Workspace isolation passes.

## 13. Automated Test Results

Targeted tests:

```text
python -m pytest tests\test_retrieval_groq_config.py tests\test_retrieval_lightrag.py -q
21 passed in 1.20s
```

Opt-in real LightRAG structured-query test:

```text
$env:LIGHTRAG_REAL_TESTS='1'
python -m pytest tests\test_retrieval_lightrag_real.py -q
1 passed in 7.92s
```

Full suite:

```text
python -m pytest -q -rs
87 passed, 4 skipped in 22.01s
```

Skipped tests and reasons:

```text
tests\test_retrieval_confidence_live.py:78
  RETRIEVAL_RUN_REAL_MODELS is not set to 1

tests\test_retrieval_lightrag_real.py:77
  set LIGHTRAG_REAL_TESTS=1 to run real LightRAG verification

tests\test_retrieval_live_integration.py:104
  RETRIEVAL_TEST_DATABASE_URL is not set

tests\test_retrieval_workspace_isolation.py:45
  RETRIEVAL_TEST_DATABASE_URL is not set
```

Compilation check:

```text
python -m py_compile retrieval\config.py retrieval\lightrag_indexer.py tests\test_retrieval_groq_config.py tests\test_retrieval_lightrag.py tests\test_retrieval_lightrag_real.py
```

Result: no errors.

## 14. Real vs Mocked vs Skipped

### Real

- Groq model-list and chat-completion connection probe.
- BGE-M3 embedding model load and inference.
- LightRAG real extraction for workspace A and workspace B.
- GraphML persistence and reload from disk.
- Real LLM answers for three workspace A questions and three workspace B questions.
- Real `aquery_data()` calls for all six query modes.
- Real cross-workspace isolation queries.

### Mocked

- `tests/test_retrieval_groq_config.py` verifies environment fallback and engine key resolution without calling Groq.
- `tests/test_retrieval_lightrag.py` uses fake LightRAG clients and fake embedders for adapter behavior and warmup tests.
- Existing Phase 4 unit tests remain fake-client based.

### Skipped

- Live confidence-model test requires `RETRIEVAL_RUN_REAL_MODELS=1`.
- Opt-in real LightRAG test is skipped by default and was run separately with `LIGHTRAG_REAL_TESTS=1`.
- Live database integration tests require `RETRIEVAL_TEST_DATABASE_URL`.

## 15. Production Reliability Improvement

The first real extraction attempt exposed a cold-start defect: LightRAG's embedding worker timed out while BGE-M3 was still loading. BGE-M3 cold start took roughly 44 seconds in this environment.

Fix implemented in `retrieval/lightrag_indexer.py`:

- Added `_warm_default_embedder_async()`.
- It checks `embedder.is_loaded`.
- If the default embedder is not loaded, it embeds one warmup text before LightRAG initializes.
- The warmup text is not indexed into the graph.
- The warmup path is skipped when a custom `lightrag_factory` is used, preserving fake-client tests.
- The warmup is also skipped when the embedder reports it is already loaded.

Tests added in `tests/test_retrieval_lightrag.py`:

- unloaded default embedder receives one warmup call
- loaded default embedder receives no warmup call

This prevents the first production extraction from failing solely because of embedder cold start. First use is still slower because the model must load.

## 16. Limitations and Risks

- Groq on-demand limits caused transient `429` input-token-per-minute responses and one `503` over-capacity response during earlier attempts. Retries succeeded in the final verification run. Production should use backoff, request queueing, or a higher Groq tier.
- LightRAG produced a casing duplicate for PostgreSQL (`PostgreSQL` and `Postgresql`). Entity canonicalization is not implemented in this phase. The duplicate did not break extraction, queries, or isolation.
- The real verification used file-based GraphML storage. PostgreSQL-backed LightRAG storage backends were not validated against the live CalibRAG database.
- BGE-M3 cold start is still approximately 44 seconds. The engine now warms it automatically, but deployment readiness/liveness timeouts must account for the first operation.
- The real Groq extraction verification is currently captured by temporary scripts and the sanitized result JSON, not by a committed real-API pytest. The committed tests cover configuration, adapter behavior, and warmup logic.

## 17. Final Acceptance Checklist

- [x] Groq base URL, model, and API key connection works.
- [x] `GROQ_API_KEY` fallback works when `RETRIEVAL_GRAPH_LLM_API_KEY` is empty.
- [x] Real entity extraction works with `qwen/qwen3.8-27b`.
- [x] Real relationship extraction works with `qwen/qwen3.8-27b`.
- [x] Required facts are present in workspace A and workspace B graphs.
- [x] GraphML persistence succeeds under the workspace-specific path.
- [x] Graph reload and traversal succeed after persistence.
- [x] Real graph answers are grounded in the controlled documents.
- [x] All six query modes complete successfully.
- [x] Workspace A and B graphs are isolated.
- [x] Cross-workspace queries return no foreign entities.
- [x] Targeted tests pass: `21 passed`.
- [x] Opt-in real LightRAG test passes: `1 passed`.
- [x] Full suite passes: `87 passed, 4 skipped`.
- [x] `py_compile` passes.
- [x] Phase 4.3 temporary test data cleaned up.
- [x] No API key exposed in code, logs, or report.
- [x] Member 3 synchronous traversal interface unchanged.
- [x] No commit or push performed.
- [x] Phase 5 not started.

## 18. Recommendation and Stop Point

Phase 4.3 is complete for the tested file-backed LightRAG workflow. The Groq/Qwen provider path, real extraction, persistence, reload, real queries, query modes, and workspace isolation all passed.

Recommended follow-up before production use:

- validate PostgreSQL-backed LightRAG storage against the real database
- add Groq retry/backoff and rate-limit handling at the service boundary
- add entity casing/canonicalization if duplicate entities become a retrieval quality issue
- decide whether the real Groq verification should become a committed opt-in pytest

Stop point: Phase 4.3 complete. Do not start Phase 5 without explicit approval.
