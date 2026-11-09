# Phase 4.1 LightRAG Verification Report - Real Query and Extraction Audit

**Branch:** `member-2-retrieval`  
**Date:** 2026-11-08  
**Owner:** Member 2 - Data, Hybrid RAG & GraphRAG Lead  
**Status:** Phase 4.1 query adapter implementation and structured-query verification complete. Real LLM extraction and LLM-generated graph answers remain blocked by missing provider credentials. Phase 5 was not started.

---

## A. Executive Summary

Phase 4.1 added the missing LightRAG query surface to `LightRAGGraphEngine` and verified the real `lightrag-hku==1.5.7` query path:

```text
workspace-scoped LightRAG graph
  -> QueryParam (local/global/hybrid/mix/naive/bypass)
  -> aquery_data structured retrieval
  -> aquery_llm structured retrieval + LLM response
  -> aquery answer text
  -> structured query metadata with workspace_id
```

Verification results:

| Verification | Result |
| :--- | :--- |
| Full test suite | `79 passed`, `4 skipped` |
| Phase 4.1 targeted tests | `20 passed` |
| Opt-in real LightRAG test | `1 passed` |
| Static compilation | Passed |
| Real LightRAG structured query | Passed for all six query modes |
| Real LightRAG workspace isolation | Passed for two workspaces |
| Real LLM entity/relationship extraction | Blocked: no provider credentials |
| Real LLM-generated graph answer | Blocked: no provider credentials |

The real structured-query verification used caller-supplied custom KG data and a deterministic fake embedder. It verifies the real LightRAG query engine, `QueryParam` handling, vector/graph retrieval, references, and workspace isolation. It is **not** evidence of real LLM entity extraction.

---

## B. Scope and Constraints

Implemented scope:

- Added `aquery_data` / `query_data` for structured retrieval without LLM generation.
- Added `aquery_llm` / `query_llm` for structured retrieval plus LLM response.
- Added `aquery` / `query` for generated answer text.
- Added `QueryParam` construction for all LightRAG 1.5.7 modes.
- Added mode, limit, query, and empty-graph validation.
- Added workspace metadata to structured query results.
- Added optional `query_param_factory` for deterministic tests without importing LightRAG.
- Added fake-client tests for query modes, limits, keywords, answers, errors, empty graphs, and running-loop safety.
- Added README query documentation.

Constraints preserved:

- Work stayed on `member-2-retrieval`.
- No merge, rebase, commit, or push was performed.
- No changes to `core/schemas.py`, `verifier/confidence_gate.py`, Member 1/3/4 files, or shared interfaces.
- `BaseGraphRAGRetriever.traverse_entities` remains unchanged.
- `traverse_entities` remains synchronous for Member 3.
- Graph storage remains workspace-bound.
- Heavy LightRAG imports remain lazy.
- Structured query without explicit keywords requires an LLM for keyword extraction; generated answers always require an LLM provider.
- No real extraction or answer results were fabricated.

---

## C. LightRAG 1.5.7 Query API

Verified API facts:

- `QueryParam.mode` supports `local`, `global`, `hybrid`, `naive`, `mix`, and `bypass`.
- `aquery_data(query, param)` returns structured entities, relationships, chunks, references, and metadata without LLM generation.
- `aquery_llm(query, param, system_prompt=None)` returns structured data plus `llm_response`.
- `aquery(query, param, system_prompt=None)` returns answer text or an async stream.
- `ainsert` is the normal LLM-backed extraction/indexing path.
- `ainsert_custom_kg` writes caller-supplied chunks, entities, relationships, and vectors without LLM extraction.

Mode behavior observed in the real LightRAG run:

| Mode | Result shape |
| :--- | :--- |
| `local` | 3 entities, 2 relationships, 1 chunk, 1 reference |
| `global` | 3 entities, 2 relationships, 1 chunk, 1 reference |
| `hybrid` | 3 entities, 2 relationships, 1 chunk, 1 reference |
| `mix` | 3 entities, 2 relationships, 1 chunk, 1 reference |
| `naive` | 0 entities, 0 relationships, 1 chunk, 1 reference |
| `bypass` | 0 entities, 0 relationships, 0 chunks, 0 references |

`bypass` skips retrieval and is only useful with LLM generation. `naive` performs chunk-vector retrieval only.

---

## D. Implementation Details

### Query methods

`retrieval/lightrag_indexer.py` now exposes:

- `aquery_data(query, *, mode="mix", top_k=None, chunk_top_k=None, hl_keywords=None, ll_keywords=None, include_references=True, enable_rerank=False)`
- `aquery_llm(...)`
- `aquery(...) -> str`
- `query_data(...)`
- `query_llm(...)`
- `query(...) -> str`

All methods are workspace-bound through the engine instance. Structured results include:

```text
query
mode
workspace_id
status
message
data.entities
data.relationships
data.chunks
data.references
metadata.query_mode
```

### QueryParam construction

The adapter validates the mode and positive limits, normalizes keyword lists, disables streaming, and lazily imports `lightrag.QueryParam`. Tests can inject `query_param_factory` so the fake-client suite does not require LightRAG.

### Empty workspace behavior

If the default NetworkX graph file is absent and no client/factory is injected, structured query returns a deterministic failure payload:

```text
status = "failure"
metadata.failure_reason = "no_graph"
data = {}
workspace_id = <bound workspace>
```

No empty result is treated as a successful retrieval.

### LLM requirements

- `aquery_data` does not require LLM credentials when explicit `hl_keywords` and `ll_keywords` are supplied.
- Without explicit keywords, LightRAG's graph query modes call the LLM for keyword extraction.
- `aquery` and `aquery_llm` always require an LLM provider and raise `GraphConfigurationError` when none is configured.

---

## E. Files Created or Modified

- `retrieval/lightrag_indexer.py` - query methods, `QueryParam` handling, validation, normalization, workspace metadata, optional query-param factory.
- `tests/test_retrieval_lightrag.py` - fake-client query coverage.
- `tests/test_retrieval_lightrag_real.py` - opt-in real LightRAG structured-query verification.
- `retrieval/README.md` - query mode usage and no-credential behavior.
- `retrieval/reports/PHASE_4_1_REAL_LIGHTRAG_VERIFICATION_REPORT.md` - this report.

No Member 1/3/4 files, shared schemas, or confidence-gate files were modified.

---

## F. Verification

Commands run with the Phase 2 temporary virtual environment:

```bash
python -m py_compile retrieval/lightrag_indexer.py tests/test_retrieval_lightrag.py tests/test_retrieval_lightrag_real.py
python -m pytest tests/test_retrieval_lightrag.py tests/test_retrieval_member2.py tests/test_agents_member3.py -q
LIGHTRAG_REAL_TESTS=1 python -m pytest tests/test_retrieval_lightrag_real.py -q
python -m pytest -q
```

Results:

- Targeted Phase 4.1/Member 3 tests: `20 passed`.
- Opt-in real LightRAG structured-query test: `1 passed`.
- Full suite: `79 passed, 4 skipped` (the real test is skipped unless `LIGHTRAG_REAL_TESTS=1`).
- Static compilation: passed.
- Real LightRAG structured query smoke test: all six modes returned expected shapes.
- Real LightRAG workspace isolation: `tenant_a` results contained only `tenant_a` graph data; `tenant_b` results were separate.

Temporary verification script:

```text
C:\Users\LENOVO\AppData\Local\Temp\opencode\test_real_lightrag_query.py
```

The script used:

- `lightrag-hku==1.5.7`
- real `LightRAG.aquery_data`
- real NetworkX graph storage
- real NanoVectorDB vector storage
- deterministic fake embeddings
- caller-supplied custom KG data
- no LLM calls

---

## G. Blockers and Limitations

- Real LLM entity/relationship extraction was not executed because no verified provider credentials are configured.
- Real LLM-generated graph answers were not executed for the same reason.
- No root `.env` exists; only `.env.example` and `retrieval/.env.ingestion.example` are present. No `OPENAI_API_KEY`, `RETRIEVAL_GRAPH_LLM_API_KEY`, `OPENAI_BASE_URL`, or `RETRIEVAL_GRAPH_LLM_BASE_URL` was available.
- The custom KG query verification proves the query/storage path, not extraction quality.
- PostgreSQL-backed LightRAG storage classes remain configurable but were not verified against the live CalibRAG database.
- LightRAG's NetworkX graph response does not preserve semantic edge direction or relation labels; `RELATED_TO` remains the truthful fallback in `traverse_entities`.

To complete real extraction verification, configure an OpenAI-compatible provider:

```bash
RETRIEVAL_GRAPH_LLM_MODEL=gpt-4o-mini
RETRIEVAL_GRAPH_LLM_BASE_URL=https://api.openai.com/v1
RETRIEVAL_GRAPH_LLM_API_KEY=...
```

Then run `aindex_workspace`/`aindex_text` and compare extracted graph nodes and edges with source chunks.

---

## H. Recommendation

Phase 4.1 query implementation is complete and verified against real LightRAG storage and retrieval. Phase 4.1 real LLM extraction verification should remain open until credentials are supplied. Phase 5 was not started, and no commit or push was performed.
