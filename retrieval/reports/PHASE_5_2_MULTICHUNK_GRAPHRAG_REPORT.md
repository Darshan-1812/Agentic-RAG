# Phase 5.2 Multi-Chunk GraphRAG Verification Report

## 1. Executive Summary

Phase 5.2 fixed the Member 2 LightRAG adapter so every intended ingested chunk is submitted as a stable, separately identified LightRAG document instead of relying on LightRAG's filename-based deduplication.

The fix was verified with six mocked multi-chunk tests and one opt-in real test using four real Docling chunks per document, real BGE-M3 embeddings, PostgreSQL ingestion, real LightRAG 1.5.7 GraphML storage, and real Groq/Qwen extraction. The real test passed in 397.47 seconds. It verified later-chunk facts, chunk-level provenance, idempotent re-indexing, GraphML reload, and two-workspace isolation.

Real Groq `429` ITPM responses were observed during testing. The adapter reported failed chunks as `partial_failure` rather than silently treating the run as complete. Retrying the same chunk set skipped already processed chunks, deleted failed documents, resubmitted only the failed chunks, and completed without duplicate graph data.

No commit, push, merge, or rebase was performed.

## 2. Scope and Constraints

In scope:

- Fix multi-chunk LightRAG ingestion in `retrieval/lightrag_indexer.py`.
- Preserve stable chunk identity and original document/chunk provenance.
- Add idempotent preflight behavior for already processed chunks.
- Report partial failures and retry failed chunks without duplicating processed chunks.
- Verify real GraphRAG extraction, persistence, reload, provenance, and workspace isolation.
- Add mocked tests, a gated real test, README documentation, and this report.

Out of scope:

- Production PostgreSQL-backed LightRAG storage integration.
- Member 1, Member 3, Member 4, or shared schema changes.
- Changing the synchronous `traverse_entities` contract.
- Changing `ChunkEvidence` or Member 1 `cross_encoder_logit` semantics.
- Using `ainsert_custom_kg()` as proof of extraction.
- Hardcoded graph entities or relationships.
- Committing, pushing, merging, or rebasing.

Constraints preserved:

- Work was performed only on branch `member-2-retrieval`.
- Existing uncommitted work, tests, reports, and file-backed GraphML production configuration were preserved.
- API keys and secrets were not printed, logged, or written to this report.
- The real verification used the normal LLM-backed extraction path, not synthetic graph injection.

## 3. Branch and Repository State

- Branch: `member-2-retrieval`
- Commit/push status: no commit and no push performed.
- Working tree contains many pre-existing modified and untracked Member 2 files from prior phases.
- Phase 5.2 production change: `retrieval/lightrag_indexer.py`.
- Phase 5.2 documentation change: `retrieval/README.md`.
- Phase 5.2 new tests:
  - `tests/test_retrieval_lightrag_multichunk.py`
  - `tests/test_retrieval_phase5_2_real_multichunk.py`
- Phase 5.2 test adjustment:
  - `tests/test_retrieval_phase5_1_real_e2e.py` was made tolerant of equivalent LLM entity names such as `Inventory Sync` versus `Inventory Sync Project`. This was a regression-test robustness fix, not a production behavior change.
- Existing LightRAG unit test expectation updated:
  - `tests/test_retrieval_lightrag.py` now expects unique per-chunk source paths.

## 4. Exact Environment

- OS: Windows
- Shell: PowerShell 5.1
- Python: 3.11.6
- Python environment: `C:\Users\LENOVO\AppData\Local\Temp\opencode\calibrag-phase2-venv`
- LightRAG: `lightrag-hku 1.5.7`
- Docling: `2.148.0`
- FlagEmbedding: `1.4.2`
- sentence-transformers: `6.2.0`
- PyTorch: `2.15.0+cpu`
- asyncpg: `0.32.0`
- NumPy: `2.4.6`
- PostgreSQL: 17.11
- Docker service: `calibrag-retrieval-postgres-1`
- Docker image: `paradedb/paradedb:0.26.5-pg17`
- PostgreSQL extensions: `pg_search 0.26.5`, `vector 0.8.6`
- Test DSN: `postgresql://postgres:postgres@localhost:5432/calibrag`
- Real LLM provider: Groq OpenAI-compatible endpoint
- Real extraction model: `qwen/qwen3.8-27b`
- Graph storage used by the real test: file-backed GraphML under a temporary `RETRIEVAL_GRAPH_STORAGE_ROOT`

## 5. Real vs Mocked vs Skipped Components

Real components exercised by the Phase 5.2 opt-in test:

- Real Docling parsing of controlled Markdown documents.
- Real ingestion pipeline and PostgreSQL persistence.
- Real four-chunk documents per workspace.
- Real BGE-M3 embeddings.
- Real LightRAG 1.5.7 client.
- Real LightRAG GraphML persistence.
- Real Groq/Qwen entity and relationship extraction.
- Real GraphML reload in a new LightRAG engine instance.
- Real two-workspace isolation.

Mocked components exercised by the Phase 5.2 unit tests:

- Fake LightRAG `doc_status` storage.
- Fake `ainsert` behavior.
- Fake `adelete_by_doc_id` behavior.
- Forced processed, failed, pending, empty, and duplicate input states.

Skipped or not validated:

- PostgreSQL-backed LightRAG storage classes were not used for Phase 5.2.
- Real `503` over-capacity behavior was not induced.
- Large-corpus graph extraction quality was not benchmarked.
- Existing GraphML data from earlier phases was not migrated to the new source-path format.
- Production deployment and commit/push were intentionally not performed.

## 6. Code Changes

### `retrieval/lightrag_indexer.py`

The `aindex_chunks` path now builds stable per-chunk entries before contacting LightRAG:

- Each chunk is keyed by the original `chunk_id`.
- Each chunk gets a unique source path:
  - `<filename-or-doc-title>::chunk::<safe-chunk-id>`
- Non-empty chunks are deduplicated by `chunk_id`; conflicting content for the same ID raises `GraphConfigurationError`.
- Empty chunks are reported in `empty_chunks`.
- Duplicate inputs with identical content are reported in `duplicate_input_chunks`.

The adapter now preflights LightRAG `doc_status`:

- `PROCESSED` chunks are skipped.
- `FAILED` chunks are deleted with `adelete_by_doc_id` and resubmitted.
- Pending states are resumed through the LightRAG processing pipeline.
- Missing or failed deletion raises `GraphConfigurationError` rather than silently duplicating data.

The result payload now reports:

- `status`
- `indexed_documents`
- `submitted_chunks`
- `source_documents`
- `processed_chunks`
- `failed_chunks`
- `pending_chunks`
- `skipped_chunks`
- `retried_failed_chunks`
- `empty_chunks`
- `duplicate_input_chunks`
- `chunk_mappings`
- `track_id`
- `workspace_id`

Each chunk mapping contains:

- `chunk_id`
- `doc_id`
- `source_path`
- `lightrag_document_id`
- `status`
- `error`

Key locations:

- `_PROCESSED_STATUSES` and related status constants: `retrieval/lightrag_indexer.py:29`
- `_chunk_source_path`: `retrieval/lightrag_indexer.py:880`
- `_build_chunk_entries`: `retrieval/lightrag_indexer.py:903`
- `_delete_failed_document`: `retrieval/lightrag_indexer.py:962`
- `_process_pending_documents`: `retrieval/lightrag_indexer.py:978`
- `_empty_index_result`: `retrieval/lightrag_indexer.py:996`
- `aindex_chunks`: `retrieval/lightrag_indexer.py:1020`

### Tests and documentation

- Added `tests/test_retrieval_lightrag_multichunk.py` with six focused tests.
- Added `tests/test_retrieval_phase5_2_real_multichunk.py` with an opt-in real multi-chunk verification.
- Updated `tests/test_retrieval_lightrag.py` for the new unique source-path expectation.
- Updated `tests/test_retrieval_phase5_1_real_e2e.py` to accept equivalent LLM entity names.
- Updated `retrieval/README.md` with multi-chunk identity, provenance, retry behavior, result fields, test commands, and limitations.

No shared schemas, Member 1/3/4 interfaces, or production graph-storage configuration were changed.

## 7. Ingestion Results

Real Phase 5.2 ingestion:

- Two controlled Markdown documents were ingested into two separate workspaces.
- Both files used the display filename `controlled.md`.
- Each document produced exactly four real Docling chunks.
- The document IDs differed because the workspaces differed.
- The chunk ID sets were disjoint across workspaces.
- Re-ingesting the same source into the same workspace returned `duplicate=True`.
- The duplicate re-ingest returned the same document ID and the same four chunk IDs.

Chunk source-path behavior:

- Real ingestion chunks do not currently carry `file_path` metadata.
- The adapter therefore used the document title as the source-path prefix.
- Example form: `<doc-title>::chunk::<safe-chunk-id>`.
- Chunk IDs remained unique and stable, so same-filename ingestion did not collapse chunks.

## 8. Dense Retrieval Results

Phase 5.2 did not change dense retrieval, embeddings, or pgvector behavior.

Regression evidence:

- The existing Phase 5.1 real hybrid + GraphRAG test passed after the Phase 5.2 adapter change.
- That test exercises real Docling, BGE-M3 embeddings, PostgreSQL/pgvector retrieval, and graph integration.
- Result: `1 passed in 279.66s`.

No new dense-retrieval quality benchmark was performed in Phase 5.2.

## 9. BM25 Results

Phase 5.2 did not change BM25 or ParadeDB behavior.

Regression evidence:

- The existing Phase 5.1 real hybrid test passed after the Phase 5.2 change.
- That test exercises real ParadeDB BM25 retrieval against the live PostgreSQL service.

No new BM25 quality benchmark was performed in Phase 5.2.

## 10. RRF Fusion Results

Phase 5.2 did not change reciprocal rank fusion.

Regression evidence:

- The existing Phase 5.1 real hybrid test passed after the Phase 5.2 change.
- That test exercises real vector and BM25 result fusion through `reciprocal_rank_fusion`.

No new RRF parameter sweep was performed in Phase 5.2.

## 11. BGE Reranker Results

Phase 5.2 did not change reranker behavior or `ChunkEvidence` semantics.

Regression evidence:

- The existing Phase 5.1 real hybrid test passed after the Phase 5.2 change.
- That test exercises the real BGE reranker and asserts `cross_encoder_logit` values and ordering.

No new reranker quality benchmark was performed in Phase 5.2.

## 12. LightRAG Graph Extraction Results

Real Phase 5.2 verification:

- Each workspace indexed one four-chunk document.
- Every chunk was submitted as a separate stable LightRAG document.
- All four chunks reached `processed` status after retry handling.
- The graph contained entities and edges extracted by the real Groq/Qwen model.
- Later-chunk facts were present:
  - Workspace A: `Risk Engineering` team, `Alice`, `PostgreSQL`, and fraud-platform facts.
  - Workspace B: `Logistics Operations` team, `Bob`, `MySQL`, and inventory-platform facts.
- The test asserted that later-chunk provenance appeared on edges incident to the later-chunk entity.
- The graph was persisted as GraphML.
- Closing and recreating the engine for the same workspace preserved the expected node set and edges.
- Graph files for the two workspaces were distinct.

Observed real provider behavior:

- Groq returned multiple `429` ITPM responses during the first real run.
- LightRAG marked some chunks `FAILED`.
- The adapter returned `partial_failure` with failed chunk details rather than reporting success.
- Retrying the same chunk set skipped processed chunks, deleted failed documents, and resubmitted only failed chunks.
- The final real run completed with all four chunks processed and no failed chunks.

## 13. Hybrid + Graph Integration Results

Phase 5.2 did not change `retrieve_with_graph` or the synchronous traversal contract.

Regression evidence:

- The existing Phase 5.1 real hybrid + GraphRAG test passed after the Phase 5.2 change.
- It covers:
  - real hybrid evidence
  - real graph context
  - `Graph Link:` context lines
  - `graph_error is None`
  - GraphML reload
  - empty-workspace behavior
  - cross-workspace evidence isolation
- Result: `1 passed in 279.66s`.

## 14. Workspace Isolation Results

Real Phase 5.2 verification:

- Workspace A and workspace B ingested documents with the same display filename.
- Document IDs differed.
- Chunk IDs were disjoint.
- Graph node sets were disjoint.
- Workspace A graph context contained no Beta/MySQL/Bob/Mumbai facts.
- Workspace B graph context contained no Acme/PostgreSQL/Alice/Pune facts.
- Each workspace wrote a separate GraphML file.
- Reloaded engines preserved workspace-specific node sets.

Mocked and existing coverage:

- Existing cross-workspace chunk rejection tests still pass.
- The full suite includes workspace isolation tests.

## 15. Failure Handling and Retry Behavior

Unit test coverage:

- `test_multichunk_same_filename_gets_unique_stable_sources`
  - Verifies unique source paths and stable chunk mappings for chunks from two documents.
- `test_multichunk_idempotent_second_call_submits_nothing`
  - Verifies a second index call skips all processed chunks and submits nothing.
- `test_multichunk_retries_failed_chunk_without_reinserting_processed`
  - Verifies processed chunks are skipped, failed chunks are deleted, and only failed chunks are resubmitted.
- `test_multichunk_reports_partial_failure_and_recovers_on_retry`
  - Verifies `partial_failure` reporting and successful recovery on the next call.
- `test_multichunk_reports_empty_and_duplicate_inputs`
  - Verifies empty and duplicate inputs are reported and not resubmitted.
- `test_multichunk_delete_failure_stops_retry`
  - Verifies a failed deletion raises `GraphConfigurationError` instead of duplicating data.

Real failure behavior:

- Real Groq `429` ITPM responses caused some chunks to fail during the first attempt.
- The adapter reported `partial_failure` with chunk IDs and errors.
- Retrying the same chunk set did not reinsert processed chunks.
- Failed documents were deleted before resubmission.
- The final result was `completed` with all four chunks processed.

Known boundary:

- If LightRAG raises before recording per-chunk status, the exception propagates. A later retry still uses `doc_status` preflight and does not duplicate chunks already marked processed.

## 16. PostgreSQL-Backed LightRAG Investigation

Phase 5.2 did not implement or validate production PostgreSQL-backed LightRAG storage.

Status:

- File-backed GraphML remains the validated configuration.
- Phase 5.1's controlled PGTable feasibility test passed, but it is not production graph-storage integration.
- Phase 5.2 real verification used GraphML.
- No production storage configuration was changed.

## 17. Performance Measurements

- Phase 5.2 real multi-chunk test: `1 passed in 397.47s`.
  - Includes real Docling, BGE-M3 model load, PostgreSQL operations, LightRAG initialization, Groq extraction, rate-limit retries, a 65-second inter-workspace pause, GraphML persistence, and reload.
- Phase 5.1 real hybrid + GraphRAG regression: `1 passed in 279.66s`.
- Full test suite with live DB and real gates disabled: `110 passed, 5 skipped in 39.25s`.
- Focused LightRAG unit tests: `22 passed in 2.02s`.
- Compile check: `compileall ok`.

These are integration timings, not retrieval-quality or throughput benchmarks. Groq rate limiting materially affects real graph indexing time.

## 18. Security and Secret Handling

- API keys were not printed, logged, or written to this report.
- The real test loaded provider configuration from `retrieval/.env`.
- No secret values are included in commands or evidence.
- The local test DSN is a development-only PostgreSQL credential already used by prior Phase 5 reports.
- No commit or push was performed.
- Error messages returned by the adapter are sanitized through the existing graph error path.

## 19. Known Limitations

- LightRAG still applies its internal text chunker to each submitted chunk. The original chunk ID is the outer document/provenance identity, but internal LightRAG subchunks use LightRAG's own chunking settings.
- Real ingestion chunks currently lack `file_path` metadata, so real source paths fall back to the document title plus chunk ID.
- Existing GraphML files created before Phase 5.2 are not migrated to the new per-chunk source paths. Already-`PROCESSED` chunk IDs are skipped on retry.
- Groq's `429` ITPM limit can cause partial graph indexing. Callers should retry failed chunks with pauses or a retry helper.
- `aindex_text` remains a single-document path and does not provide multi-chunk preflight or per-chunk failure reporting.
- PostgreSQL-backed LightRAG storage is still not validated end-to-end.
- The real Phase 5.2 corpus is a small controlled integration smoke test, not a retrieval-quality benchmark.
- The Phase 5.1 real test's exact entity-name assertion was relaxed because the LLM produced semantically equivalent names. Other Phase 5.1 assertions remain intact.

## 20. Commands and Test Evidence

Focused mocked tests:

```powershell
& $py -m pytest tests/test_retrieval_lightrag_multichunk.py tests/test_retrieval_lightrag.py -q
```

Result:

```text
22 passed in 2.02s
```

Full suite with live database and real gates disabled:

```powershell
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:5432/calibrag'
& $py -m pytest tests -q
```

Result:

```text
110 passed, 5 skipped in 39.25s
```

Compile check:

```powershell
& $py -m compileall -q retrieval tests
```

Result:

```text
compileall ok
```

Real Phase 5.2 multi-chunk test:

```powershell
$env:RETRIEVAL_PHASE5_2_REAL_TESTS='1'
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:5432/calibrag'
& $py -m pytest tests/test_retrieval_phase5_2_real_multichunk.py -q
```

Result:

```text
1 passed in 397.47s (0:06:37)
```

Real Phase 5.1 regression after the Phase 5.2 adapter change:

```powershell
$env:RETRIEVAL_PHASE5_1_REAL_TESTS='1'
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:5432/calibrag'
$env:RETRIEVAL_PHASE5_1_GRAPH_PAUSE_SECONDS='90'
& $py -m pytest tests/test_retrieval_phase5_1_real_e2e.py -q
```

Result:

```text
1 passed in 279.66s (0:04:39)
```

## 21. Final Acceptance Status

Phase 5.2 acceptance criteria status:

- Multi-chunk LightRAG ingestion fixed: passed.
- Every intended chunk submitted with stable identity: passed.
- Unique source paths avoid filename deduplication: passed.
- Idempotent re-indexing skips processed chunks: passed.
- Failed chunks reported accurately: passed.
- Failed chunks retried without duplicating processed chunks: passed.
- Later-chunk facts extracted from real GraphRAG: passed.
- Original document/chunk provenance retained in chunk mappings and graph edge provenance: passed.
- GraphML persistence and reload: passed.
- Two-workspace isolation: passed.
- Existing hybrid, graph traversal, and API regression: passed.
- Mocked multi-chunk coverage: passed.
- README updated: passed.
- Report created with 21 sections: passed.
- No commit, push, merge, or rebase: confirmed.

Phase 5.2 is complete and ready for review. Work should stop here.
