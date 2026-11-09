# Phase 3 Hybrid RAG Report - Real Embeddings, BM25, RRF, and Reranking

**Branch:** `member-2-retrieval`  
**Date:** 2026-11-07  
**Owner:** Member 2 - Data, Hybrid RAG & GraphRAG Lead  
**Status:** Phase 3 implementation and verification complete. Phase 4 was not started.

---

## A. Executive Summary

Phase 3 adds a real hybrid retrieval pipeline on top of the verified Phase 2 ingestion foundation:

```text
query
  -> BGE-M3 dense embedding
  -> pgvector HNSW cosine search
  -> ParadeDB BM25 lexical search
  -> reciprocal rank fusion (RRF)
  -> BGE cross-encoder reranking with raw logits
  -> core.schemas.ChunkEvidence
```

All required behavior was implemented without modifying shared Member 1/3/4 files, without changing the database schema, and without committing or pushing changes.

Final verification results:

| Verification | Result |
| :--- | :--- |
| Full test suite | `67 passed`, 1 unrelated deprecation warning |
| Member 2 suite with live PostgreSQL | `43 passed` |
| New Phase 3 tests | `23 passed` |
| Real BGE-M3 embedding | 1024-dimensional vectors, unit normalized |
| Real BGE reranker | Raw logits, no sigmoid conversion |
| Live real-model end-to-end retrieval | Passed; correct top chunk, both retrieval sources, `c_evidence=0.95` |
| Static compilation | Passed |

---

## B. Scope and Constraints

Implemented scope:

- Real `BAAI/bge-m3` dense embeddings.
- Workspace-scoped pgvector persistence and HNSW cosine retrieval.
- Workspace-scoped ParadeDB BM25 retrieval.
- Reciprocal rank fusion of dense and lexical candidates.
- Real `BAAI/bge-reranker-v2-m3` cross-encoder reranking.
- Final `ChunkEvidence` output compatible with Member 1 confidence calculation.
- Unit, live database, and real-model verification.
- Dependency documentation for the retrieval runtime.

Constraints preserved:

- Work stayed on `member-2-retrieval`.
- No merge, rebase, commit, or push was performed.
- No changes to `pyproject.toml`, root Docker configuration, root `.env.example`, `core/schemas.py`, `verifier/confidence_gate.py`, `api/`, or Member 1/3/4 modules.
- No database migration was added; the existing `0003_search_indexes` schema was sufficient.
- Heavy ML imports remain lazy; importing `retrieval` does not load models.
- Cross-encoder scores are raw logits. No sigmoid probabilities or fabricated scores are emitted as `cross_encoder_logit`.
- Existing Phase 2 files and untracked workspace files were preserved.

---

## C. Architecture

### Dense path

`BgeM3Embedder` lazily resolves `BAAI/bge-m3` to an immutable Hugging Face snapshot, constructs `BGEM3FlagModel` on the configured device, encodes documents or queries, validates shape and finite values, and normalizes vectors.

`PGVectorStore` validates chunk ownership, embeds chunks, and upserts 1024-dimensional vectors into `chunk_embeddings` with model name and resolved model revision provenance.

`PGVectorRetriever` embeds the query, applies a workspace and model/revision filter, sets `hnsw.ef_search`, and orders candidates by pgvector cosine distance.

### Lexical path

`ParadeDBBM25Retriever` normalizes the query into bounded plain lexical terms, then executes a parameterized `paradedb.match('content', $2)` query with `paradedb.score(c)`. Workspace and non-deleted document filters are always applied.

### Fusion path

`reciprocal_rank_fusion` combines ranked dense and lexical lists using:

```text
score = weight / (k + rank)
```

Duplicate chunks are merged without losing vector scores, BM25 scores, metadata, or retrieval-source provenance. Ties are deterministic by chunk ID.

### Reranking path

`BgeReranker` lazily resolves `BAAI/bge-reranker-v2-m3`, computes raw cross-encoder logits with `normalize=False`, sorts by logit descending, and assigns rerank ranks.

### Service path

`RetrievalService` runs vector and BM25 retrieval concurrently with `asyncio.gather`, fuses results with RRF, reranks the configured candidate depth, and converts final internal results into `ChunkEvidence`.

---

## D. Files Created or Modified

Phase 3 implementation files:

- `retrieval/config.py` - Phase 3 settings, validation, and environment parsing.
- `retrieval/retrieval_types.py` - `RetrievedChunk`, provenance metadata, and final evidence conversion.
- `retrieval/model_cache.py` - immutable Hugging Face snapshot and revision resolution.
- `retrieval/embeddings/base.py` - embedding contracts, device resolution, validation.
- `retrieval/embeddings/bge_m3.py` - BGE-M3 embedder.
- `retrieval/embeddings/__init__.py` - embedding exports.
- `retrieval/vector/pgvector.py` - pgvector store, HNSW retriever, vector errors.
- `retrieval/vector/__init__.py` - vector exports.
- `retrieval/bm25/paradedb.py` - ParadeDB BM25 retriever and query normalization.
- `retrieval/bm25/__init__.py` - BM25 exports.
- `retrieval/hybrid/rrf.py` - production RRF implementation.
- `retrieval/hybrid/__init__.py` - hybrid exports.
- `retrieval/reranking/bge_reranker.py` - raw-logit BGE reranker.
- `retrieval/reranking/__init__.py` - reranker exports.
- `retrieval/service.py` - unified hybrid retrieval service.
- `retrieval/requirements-retrieval.txt` - Phase 3 runtime dependencies.
- `retrieval/__init__.py` - lazy Phase 3 exports.

Phase 3 test files:

- `tests/test_retrieval_embeddings.py`
- `tests/test_retrieval_pgvector.py`
- `tests/test_retrieval_bm25.py`
- `tests/test_retrieval_hybrid.py`
- `tests/test_retrieval_reranker.py`
- `tests/test_retrieval_confidence_integration.py`
- `tests/test_retrieval_live_integration.py`

Phase 2 files preserved:

- `retrieval/ingestion/docling_parser.py` remains modified from the Phase 2.1 PDF fix.
- Existing Phase 2 ingestion, database, migration, and test files were not reverted.

No shared files were modified.

---

## E. Configuration

New retrieval settings and defaults:

| Setting | Default |
| :--- | :--- |
| `embedding_model_name` | `BAAI/bge-m3` |
| `embedding_model_revision` | `main` |
| `embedding_dimension` | `1024` |
| `embedding_device` | `auto` |
| `embedding_batch_size` | `16` |
| `embedding_max_length` | `8192` |
| `embedding_query_max_length` | `8192` |
| `embedding_fp16` | `true` |
| `embedding_normalize` | `true` |
| `model_cache_dir` | `None` |
| `model_local_files_only` | `false` |
| `retrieval_top_k` | `5` |
| `vector_candidate_depth` | `50` |
| `hnsw_ef_search` | `100` |
| `bm25_candidate_depth` | `50` |
| `rrf_k` | `60` |
| `hybrid_candidate_depth` | `50` |
| `rerank_candidate_depth` | `20` |
| `reranker_model_name` | `BAAI/bge-reranker-v2-m3` |
| `reranker_model_revision` | `main` |
| `reranker_device` | `auto` |
| `reranker_batch_size` | `16` |
| `reranker_max_length` | `1024` |
| `reranker_fp16` | `true` |

Validation enforces:

- Embedding dimension is exactly 1024 to match `chunk_embeddings.embedding vector(1024)`.
- Candidate depths are at least `retrieval_top_k`.
- Rerank depth does not exceed hybrid depth.
- Device values are `auto`, `cpu`, `mps`, `cuda`, or `cuda:<index>`.
- Model names and revisions are non-empty.
- Database DSNs are valid PostgreSQL URLs.

Environment variable names use the `RETRIEVAL_` prefix, including `RETRIEVAL_EMBEDDING_*`, `RETRIEVAL_VECTOR_CANDIDATE_DEPTH`, `RETRIEVAL_BM25_CANDIDATE_DEPTH`, `RETRIEVAL_RRF_K`, `RETRIEVAL_RERANK_CANDIDATE_DEPTH`, and `RETRIEVAL_RERANKER_*`.

---

## F. Embeddings

`BgeM3Embedder` implementation details:

- Lazy model loading with a thread lock.
- Optional injected model for tests.
- Device resolution before model construction because FlagEmbedding treats a string device literally.
- Snapshot resolution through `huggingface_hub.snapshot_download`.
- Resolved commit hash exposed as `model_revision` after loading.
- Document and query encodings use separate configured maximum lengths.
- Dense output must contain `dense_vecs`.
- Output is validated for 1024 dimensions, expected count, finite values, and zero vectors.
- Optional L2 normalization is applied and verified.

Real-model verification:

| Check | Result |
| :--- | :--- |
| Model | `BAAI/bge-m3` |
| Resolved revision | `5617a9f61b028005a4858fdac845db406aefb181` |
| Document matrix shape | `(2, 1024)` |
| Query shape | `(1024,)` |
| Vector norms | `[1.0, 1.0]` |
| Query similarity to relevant passage | `0.875985` |
| Device | CPU |
| First load plus inference | `107.76s` including download |

---

## G. Vector Retrieval

`PGVectorStore.upsert_embeddings`:

- Validates workspace ID.
- Rejects empty chunk batches, missing chunk IDs, empty content, and duplicate chunk IDs.
- Verifies every chunk exists in `document_chunks` for the same workspace before embedding.
- Requires embedder model name and revision.
- Enforces exact `(chunk_count, 1024)` output.
- Upserts with bound parameters and `$5::vector`.
- Preserves model provenance in `chunk_embeddings`.

`PGVectorRetriever.retrieve`:

- Returns an empty list for blank queries without calling the embedder or database.
- Filters by `e.workspace_id`, `e.model_name`, and `e.model_revision`.
- Joins `document_chunks` and `documents`, excluding `d.status = 'deleted'`.
- Uses pgvector cosine distance `<=>`.
- Sets transaction-local `hnsw.ef_search`.
- Returns internal `RetrievedChunk` objects with distance, similarity, rank, metadata, and model provenance.
- Does not emit `ChunkEvidence` before reranking.

Live database verification confirmed workspace isolation: two workspaces with identical query terms returned only their own chunk.

---

## H. BM25 Retrieval

`normalize_bm25_query`:

- Accepts only string input.
- Extracts Unicode word tokens with optional internal dots, underscores, or hyphens.
- Removes quotes, semicolons, SQL comment markers, and other punctuation.
- Preserves model codes such as `EV-Pack-V2` and versions such as `v0.5`.
- Caps the query at 64 tokens.

`ParadeDBBM25Retriever.search`:

- Returns an empty list when normalization produces no terms.
- Uses `paradedb.match('content', $2)` with the normalized query as a bound parameter.
- Uses `paradedb.score(c)` for ranking.
- Applies workspace and non-deleted document filters.
- Orders by BM25 score, chunk index, and chunk ID for deterministic output.
- Returns internal `RetrievedChunk` objects with BM25 score and rank.

Live database verification confirmed BM25 workspace isolation and correct retrieval of the relevant chunk.

---

## I. RRF Fusion

`reciprocal_rank_fusion`:

- Uses `weight / (k + rank)` for each ranked list.
- Supports independent vector and BM25 weights.
- Rejects invalid `k`, `top_k`, negative weights, and both weights zero.
- Rejects non-`RetrievedChunk` inputs and chunks without IDs.
- Rejects the same chunk ID under different document IDs.
- Merges duplicate chunks while preserving vector scores, BM25 scores, metadata, and source provenance.
- Applies deterministic tie-breaking by chunk ID.
- Assigns final `rrf_score` and `rrf_rank`.

Unit tests cover duplicate merging, source provenance, score ordering, rank assignment, and invalid inputs.

---

## J. Reranking

`BgeReranker`:

- Lazy-loads `BAAI/bge-reranker-v2-m3`.
- Uses `resolve_model_source` for immutable snapshot and revision provenance.
- Calls `compute_score(..., normalize=False)`.
- Never applies sigmoid to logits.
- Validates score count, finite values, empty queries, empty passages, and invalid `top_k`.
- Sorts by raw logit descending and assigns rerank ranks.
- Runs inference in a worker thread so the async service is not blocked.

Real-model verification:

| Check | Result |
| :--- | :--- |
| Model | `BAAI/bge-reranker-v2-m3` |
| Resolved revision | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| Raw logits | `[9.445046, -11.034733]` |
| Relevant passage ranked first | Yes |
| Device | CPU |
| First load plus inference | `93.26s` including download |

The raw logits are retained unchanged in final `ChunkEvidence`.

---

## K. Unified Service and Member 1 Compatibility

`RetrievalService.retrieve`:

1. Validates query and `top_k`.
2. Validates workspace ID.
3. Runs vector and BM25 retrieval concurrently.
4. Fuses candidates with RRF.
5. Selects the rerank candidate depth.
6. Reranks with raw logits.
7. Converts final results to `core.schemas.ChunkEvidence`.

`RetrievalService.index_chunks` indexes pre-chunked content, and `index_document` loads chunks through `IngestionRepository` and indexes them.

Member 1 compatibility:

- `ChunkEvidence.cross_encoder_logit` contains the raw cross-encoder logit.
- `RetrievedChunk.to_chunk_evidence` raises `MissingCrossEncoderLogit` if reranking has not run.
- `score_metadata` exposes namespaced provenance:
  - `retrieval_sources`
  - `retrieval_vector_distance`
  - `retrieval_vector_similarity`
  - `retrieval_vector_rank`
  - `retrieval_bm25_score`
  - `retrieval_bm25_rank`
  - `retrieval_rrf_score`
  - `retrieval_rrf_rank`
  - `retrieval_cross_encoder_logit`
  - `retrieval_rerank_rank`
  - `retrieval_embedding_model`
  - `retrieval_embedding_model_revision`
- The confidence integration test passes final evidence into `compute_evidence_confidence` and confirms the raw-logit delta is used.

Live real-model end-to-end result:

```text
top chunk: The EV-Pack-V2 maximum defect tolerance is 0.5 percent.
raw logits: [10.0116, -11.0394]
retrieval sources: ['vector', 'bm25']
c_evidence: 0.95
sigma_delta_rerank: 1.0
```

---

## L. Tests and Commands

### New Phase 3 tests with live PostgreSQL

```powershell
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:5432/calibrag'
python -m pytest tests/test_retrieval_embeddings.py tests/test_retrieval_pgvector.py tests/test_retrieval_bm25.py tests/test_retrieval_hybrid.py tests/test_retrieval_reranker.py tests/test_retrieval_confidence_integration.py tests/test_retrieval_live_integration.py -q
```

Result:

```text
23 passed in 2.39s
```

### Member 2 suite with live PostgreSQL

```powershell
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:5432/calibrag'
python -m pytest tests/test_retrieval_db.py tests/test_retrieval_ingestion.py tests/test_retrieval_workspace_isolation.py tests/test_retrieval_member2.py tests/test_retrieval_embeddings.py tests/test_retrieval_pgvector.py tests/test_retrieval_bm25.py tests/test_retrieval_hybrid.py tests/test_retrieval_reranker.py tests/test_retrieval_confidence_integration.py tests/test_retrieval_live_integration.py -q
```

Result:

```text
43 passed in 21.97s
```

### Full project suite

`fastapi` was installed only in the temporary verification virtual environment to unblock Member 3 collection. No project dependency file was changed.

```powershell
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:5432/calibrag'
python -m pytest -q
```

Result:

```text
67 passed, 1 warning in 65.96s
```

The warning is an unrelated Starlette deprecation notice from `fastapi.testclient` about `httpx`.

### Static compilation

```powershell
python -m compileall -q retrieval tests/test_retrieval_embeddings.py tests/test_retrieval_pgvector.py tests/test_retrieval_bm25.py tests/test_retrieval_hybrid.py tests/test_retrieval_reranker.py tests/test_retrieval_confidence_integration.py tests/test_retrieval_live_integration.py
```

Result: `compile-ok`.

---

## M. Environment and Live Verification

| Component | Verified value |
| :--- | :--- |
| OS | Windows (`win32`) |
| Python | 3.11.6 |
| Branch | `member-2-retrieval` |
| PostgreSQL | 17.11 |
| Container | `calibrag-retrieval-postgres-1` |
| pgvector | 0.8.6 |
| pg_search / ParadeDB | 0.26.5 |
| Schema version | `0003` |
| FlagEmbedding | 1.4.2 |
| transformers | 5.22.1 |
| PyTorch | 2.15.0+cpu |
| NumPy | 2.4.6 |
| asyncpg | 0.32.0 |
| pytest | 8.4.2 |

Live verification exercised:

- Real PostgreSQL migrations and health checks.
- Real pgvector upsert and HNSW cosine retrieval.
- Real ParadeDB BM25 retrieval.
- Workspace isolation for vector, BM25, and the unified service.
- Real BGE-M3 inference.
- Real BGE reranker inference.
- Real end-to-end hybrid retrieval and Member 1 confidence calculation.

Hugging Face emitted Windows symlink-cache warnings because the cache is not running with developer mode. The downloads and inference still succeeded; this is a cache-storage warning, not a retrieval failure.

---

## N. Performance Observations

Measured on CPU with cold lazy model loading and first-time model download:

| Operation | Observed time |
| :--- | :--- |
| BGE-M3 first load plus two short document embeddings and one query | `107.76s` |
| BGE reranker first load plus two pair scores | `93.26s` |
| Live real-model indexing of two chunks | `44.71s` |
| Live real-model hybrid retrieval including reranking | `15.89s` |

These are verification timings, not production latency claims. They include model download, model load, and CPU inference. Warm-process steady-state latency was not benchmarked.

Database-side performance was not benchmarked at scale. The HNSW and BM25 indexes exist, but recall, throughput, and p95 latency on a large corpus remain unmeasured.

---

## O. Limitations

- Real-model verification used short passages, not full 8192-token documents.
- CPU-only inference is slow; GPU performance was not tested.
- HNSW recall and BM25 quality were not evaluated against a labeled benchmark corpus.
- BM25 normalization intentionally removes punctuation and boolean syntax. This is safe for SQL/query parsing but may reduce recall for some punctuation-sensitive searches.
- Stored embeddings are filtered by exact model name and resolved revision. Changing the configured revision requires reindexing.
- `model_revision="main"` resolves to the current commit at first load. The resolved commit hash is persisted for indexed embeddings, which prevents accidental mixing of revisions.
- Hugging Face cache symlinks are unavailable on the current Windows setup; cache storage may use more disk space.
- The full suite required installing `fastapi` into the temporary verification environment. This did not modify repository dependency files.
- No large-scale load, concurrency, or failure-injection testing was performed.
- Phase 4 work, including LightRAG/GraphRAG and agentic escalation, was not started.

---

## P. Git Summary, Checklist, and Phase 4 Readiness

### Git summary

- Branch: `member-2-retrieval`
- Latest commit at verification time: `2ddd481`
- No commit was created.
- No push was performed.
- No merge or rebase was performed.
- Modified tracked files: `retrieval/__init__.py`, plus the pre-existing Phase 2.1 `retrieval/docling_parser.py` modification.
- New untracked Phase 3 files are listed in section D.
- Existing untracked `myenv/`, `retrieval/IMPLEMENTATION_AUDIT.md`, Phase 2 reports, and Phase 2 files were preserved.

### Completion checklist

- [x] BGE-M3 embedding provider implemented.
- [x] Embedding dimension, normalization, and finite-value validation implemented.
- [x] pgvector workspace-scoped upsert implemented.
- [x] HNSW cosine retrieval implemented.
- [x] Model name and resolved revision filtering implemented.
- [x] ParadeDB BM25 retrieval implemented.
- [x] Safe bounded BM25 query normalization implemented.
- [x] RRF fusion implemented.
- [x] Duplicate chunk merging and provenance implemented.
- [x] BGE cross-encoder reranking implemented.
- [x] Raw logits preserved end to end.
- [x] Unified retrieval service implemented.
- [x] `ChunkEvidence` compatibility verified with Member 1 confidence gate.
- [x] Unit tests added.
- [x] Live PostgreSQL integration test added and passed.
- [x] Real BGE-M3 verification passed.
- [x] Real BGE reranker verification passed.
- [x] Live real-model end-to-end verification passed.
- [x] Full project test suite passed.
- [x] No shared files modified.
- [x] No commit or push performed.
- [x] Phase 4 not started.

### Phase 4 readiness

Phase 3 is complete and verified. The retrieval layer can now supply workspace-scoped hybrid candidates and raw cross-encoder logits to Member 1's confidence gate and to later agentic stages.

Recommended next step: obtain explicit approval before beginning Phase 4 or any LightRAG/GraphRAG work.
