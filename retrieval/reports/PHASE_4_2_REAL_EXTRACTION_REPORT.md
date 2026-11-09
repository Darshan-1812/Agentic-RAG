# Phase 4.2 Real Extraction Verification Report

**Branch:** `member-2-retrieval`  
**Date:** 2026-11-08  
**Owner:** Member 2 - Data, Hybrid RAG & GraphRAG Lead  
**Status:** Blocked by missing provider credentials. Safe static and mocked checks passed. Real LLM extraction, real graph answers, and post-extraction persistence/reload were not executed. Phase 5 was not started.

---

## A. Executive Summary

Phase 4.2 was intended to verify the complete real LightRAG path:

```text
real document/chunks
  -> LLM entity extraction
  -> LLM relationship extraction
  -> persisted LightRAG graph
  -> close/reopen persistence check
  -> real aquery modes
  -> grounded graph answer
  -> workspace isolation
```

**Real LLM-based entity extraction and graph answer generation were not verified because provider credentials were unavailable.**

No credentials were invented, no API keys were printed, and no mocked LLM was used as a substitute for real-model verification.

| Verification Area | Result |
| :--- | :--- |
| Branch and repository state | Verified |
| LightRAG runtime/version | `lightrag-hku==1.5.7` installed |
| Provider credentials | Missing |
| Static compilation | Passed |
| Mocked/unit regression suite | Passed |
| Safe empty/error behavior | Passed |
| Opt-in real LightRAG structured-query test | `1 passed` |
| Full test suite | `79 passed`, `4 skipped` |
| Real LLM entity extraction | Blocked |
| Real LLM relationship extraction | Blocked |
| Extraction grounding against source chunks | Blocked |
| Graph persistence/reload after real extraction | Blocked |
| Real LLM-generated graph answer | Blocked |
| Real extraction workspace isolation | Blocked |

The passing real LightRAG test uses caller-supplied custom KG data and deterministic fake embeddings. It verifies real LightRAG storage/query APIs, not automatic LLM extraction.

---

## B. Scope and Constraints

Preserved constraints:

- Work stayed on `member-2-retrieval`.
- No merge, rebase, commit, or push was performed.
- No Member 1/3/4 files were modified.
- No changes were made to `core/schemas.py`, `verifier/confidence_gate.py`, or shared interfaces.
- `BaseGraphRAGRetriever.traverse_entities` remains unchanged.
- `traverse_entities` remains synchronous for Member 3.
- Existing Phase 2/3/4 functionality was preserved.
- No credentials, secrets, or API keys were written to tracked files or printed.
- No real extraction results or answers were fabricated.

Phase 4.2 verification scope was limited to safe static checks, mocked/unit checks, empty/error behavior, and the existing opt-in real LightRAG structured-query test because the required provider configuration was absent.

---

## C. Provider and Credential Audit

Credential availability was checked without printing values:

```text
RETRIEVAL_GRAPH_LLM_MODEL=UNSET
RETRIEVAL_GRAPH_LLM_BASE_URL=UNSET
RETRIEVAL_GRAPH_LLM_API_KEY=UNSET
OPENAI_API_KEY=UNSET
OPENAI_BASE_URL=UNSET
.env=MISSING
retrieval\.env=MISSING
myenv\.env=MISSING
```

Provider model used for real extraction: **none**. No provider connection was attempted and no model calls were made.

The configuration supports:

```bash
RETRIEVAL_GRAPH_LLM_MODEL=gpt-4o-mini
RETRIEVAL_GRAPH_LLM_BASE_URL=https://api.openai.com/v1
RETRIEVAL_GRAPH_LLM_API_KEY=...
```

`retrieval/config.py` also accepts `OPENAI_API_KEY` and `OPENAI_BASE_URL` as fallbacks. No real values were available in the environment or local `.env` files.

---

## D. LightRAG Runtime and API Audit

Installed runtime:

```text
lightrag-hku==1.5.7
```

Verified API signatures:

```text
LightRAG.ainsert(input, split_by_character=None, split_by_character_only=False,
                 ids=None, file_paths=None, track_id=None)
LightRAG.aquery_data(query, param) -> dict[str, Any]
LightRAG.aquery_llm(query, param, system_prompt=None,
                    progress_callback=None) -> dict[str, Any]
LightRAG.aquery(query, param, system_prompt=None) -> str | AsyncIterator[str]
```

`QueryParam` defaults to `mode='mix'`. Supported modes are `local`, `global`, `hybrid`, `naive`, `mix`, and `bypass`.

The normal LLM-backed extraction path is `ainsert`. `ainsert_custom_kg` bypasses extraction and therefore cannot be used as evidence of real LLM extraction.

---

## E. Safe Static and Mocked Verification

Commands were run with:

```text
C:\Users\LENOVO\AppData\Local\Temp\opencode\calibrag-phase2-venv\Scripts\python.exe
```

Results:

| Command | Result |
| :--- | :--- |
| `python -m py_compile retrieval\config.py retrieval\lightrag_indexer.py tests\test_retrieval_lightrag.py tests\test_retrieval_lightrag_real.py tests\test_retrieval_member2.py tests\test_agents_member3.py` | Passed |
| `python -m pytest tests\test_retrieval_lightrag.py tests\test_retrieval_lightrag_real.py tests\test_retrieval_member2.py tests\test_agents_member3.py -q` | `20 passed, 1 skipped` |
| `python -m pytest tests\test_retrieval_lightrag.py -q -k "requires_llm_config"` | `2 passed, 11 deselected` |
| `LIGHTRAG_REAL_TESTS=1 python -m pytest tests\test_retrieval_lightrag_real.py -q` | `1 passed` |
| `python -m pytest -q` | `79 passed, 4 skipped` |

The opt-in real test verifies real LightRAG storage and structured query APIs with custom KG data and deterministic fake embeddings. It does not call an LLM and does not verify automatic extraction.

---

## F. Safe Error and Empty-State Behavior

A no-credential runtime check produced:

```text
{
  'empty_text': 'ValueError',
  'empty_chunks': 0,
  'invalid_mode': 'ValueError',
  'empty_graph_query': 'no_graph',
  'cross_workspace': 'GraphWorkspaceError'
}
```

Behavior matrix:

| Scenario | Expected Behavior | Result |
| :--- | :--- | :--- |
| Missing LLM API key for indexing | `GraphConfigurationError` | Passed in unit tests |
| Missing LLM API key for answer generation | `GraphConfigurationError` | Passed in unit tests |
| Empty/whitespace text indexing | `ValueError` | Passed |
| Empty chunk list | Zero indexed documents | Passed |
| Invalid query mode | `ValueError` | Passed |
| Query against missing graph | `status="failure"`, `failure_reason="no_graph"` | Passed |
| Cross-workspace chunk indexing | `GraphWorkspaceError` | Passed |
| Invalid API key | Provider-specific error propagation | Blocked: no credentials |
| Provider connection failure | Provider-specific error propagation | Blocked: no credentials |
| LLM timeout | Timeout/error propagation | Blocked: no credentials |
| Malformed provider response | Error propagation | Blocked: no credentials |
| Duplicate indexing with real provider | Idempotent behavior | Not verified |
| Graph storage write failure | Actionable storage error | Not exercised |

---

## G. Real Extraction Verification Checklist

| Requirement | Status | Evidence |
| :--- | :--- | :--- |
| Real LLM entity extraction from actual text/chunks | Not run | No provider credentials |
| Real LLM relationship extraction | Not run | No provider credentials |
| Entities grounded in source text | Not run | No extraction output |
| Relationships grounded in source text | Not run | No extraction output |
| Graph persisted to workspace storage after extraction | Not run | No extraction output |
| Graph reloads after `aclose()`/`close()` and engine recreation | Not run | No extraction output |
| Real `aquery` modes against extracted graph | Not run | No provider credentials |
| Real graph answer with citations/evidence | Not run | No provider credentials |
| Workspace isolation with real extraction | Not run | No provider credentials |
| Duplicate document indexing behavior | Not verified | Requires provider-backed extraction |

Default persistence path remains:

```text
RETRIEVAL_GRAPH_STORAGE_ROOT/<workspace_id>/graph_chunk_entity_relation.graphml
```

---

## H. Files Updated

- `retrieval/reports/PHASE_4_2_REAL_EXTRACTION_REPORT.md` - this report.
- `retrieval/README.md` - added real LLM extraction verification workflow, graph inspection, persistence/reload guidance, and current test limitation.

No production code was changed during Phase 4.2 because no actual defect was found and real provider verification could not be performed.

---

## I. Repository State

Branch:

```text
member-2-retrieval
```

Tracked modifications remain uncommitted:

```text
 M retrieval/README.md
 M retrieval/__init__.py
 M retrieval/docling_parser.py
 M retrieval/lightrag_indexer.py
 M tests/test_retrieval_member2.py
```

Numerous Phase 2/3/4 files and `myenv/` remain untracked and were preserved. No commit, merge, rebase, or push was performed.

---

## J. Blockers and Limitations

- No `RETRIEVAL_GRAPH_LLM_API_KEY` or `OPENAI_API_KEY` was available.
- No `RETRIEVAL_GRAPH_LLM_BASE_URL` or `OPENAI_BASE_URL` was available.
- No local `.env` file supplied provider configuration.
- Real extraction quality, graph grounding, graph persistence after extraction, real graph answers, and real extraction workspace isolation remain unverified.
- PostgreSQL-backed LightRAG storage classes remain unverified against the live CalibRAG database.
- The existing opt-in real test proves the query/storage path with custom KG data, not LLM extraction.
- Phase 4 cannot be described as fully verified for real extraction until credentials are supplied and the checklist in Section G is executed.

---

## K. Recommendation

Treat Phase 4.2 as **blocked, not passed**. Keep real extraction verification open. When a valid OpenAI-compatible provider is configured, rerun the workflow documented in `retrieval/README.md` and update this report with actual extracted entities, relationships, persistence/reload evidence, query results, and workspace-isolation results.

Phase 5 was not started. No commit or push was performed.
