# Phase 2 Report — Database Foundation and Real Ingestion

**Branch:** `member-2-retrieval`  
**Base commit:** `2ddd481`  
**Date:** 2026-11-06  
**Owner:** Member 2 — Data, Hybrid RAG & GraphRAG Lead  
**Status:** Phase 2 implementation complete; live PostgreSQL and real-Docling verification blocked by the local environment.

---

## 1. Scope Delivered

This phase implements only the two approved Phase 2 workstreams:

1. PostgreSQL 17 database foundation for ingestion and future retrieval.
2. Real IBM Docling ingestion for PDF, DOCX, and Markdown.

The following were deliberately **not** implemented: BGE-M3 embeddings, pgvector retrieval, ParadeDB BM25 queries, RRF orchestration, cross-encoder reranking, LightRAG, API routes, and frontend work.

No shared files were modified: `pyproject.toml`, `.env.example`, root-level Docker configuration, `core/schemas.py`, and `api/` are unchanged.

---

## 2. Database Foundation

### 2.1 Pinned infrastructure

| Component | Pin | Source |
| :--- | :--- | :--- |
| PostgreSQL image | `paradedb/paradedb:0.26.5-pg17` | Docker Hub tag verified on 2026-11-06 |
| Docling | `docling==2.148.0` | PyPI metadata checked on 2026-11-06 |
| asyncpg | `asyncpg==0.32.0` | PyPI metadata checked on 2026-11-06 |

The ParadeDB image is required because vanilla `postgres:17` does not include `pg_search`. The image includes PostgreSQL 17, pgvector, and ParadeDB `pg_search`.

### 2.2 Local service

`retrieval/docker-compose.yml` starts a pinned PostgreSQL 17 + pgvector + `pg_search` service with:

- Persistent named volume.
- `pg_isready` health check.
- Configurable credentials/database via environment variables.
- No hard-coded production secret.

### 2.3 Migrations

Numbered SQL migrations are applied by a checksummed, advisory-locked migration runner:

- `0001_extensions.sql` — creates `vector` and `pg_search`.
- `0002_core_schema.sql` — core tenant/document/chunk/job tables.
- `0003_search_indexes.sql` — pgvector embedding table, HNSW index, and ParadeDB BM25 index.

Implemented tables:

- `workspaces`
- `documents`
- `document_chunks`
- `ingestion_jobs`
- `chunk_embeddings`
- `schema_migrations`

Implemented indexes include:

- `document_chunks_bm25_idx` using ParadeDB `USING bm25` with `key_field = 'chunk_id'`.
- `chunk_embeddings_hnsw_idx` using pgvector `vector_cosine_ops`.
- Workspace/document/chunk and job-status lookup indexes.

`chunk_embeddings` and the search indexes establish the Phase 3/4 schema foundation only. No embedding generation or retrieval queries were added.

### 2.4 Workspace isolation

Workspace isolation is enforced at multiple layers:

- `workspace_id` is required on all repository calls; empty/missing values raise `WorkspaceRequiredError`.
- Every repository SQL query filters by `workspace_id`.
- `document_chunks` uses a composite foreign key `(doc_id, workspace_id)` to prevent cross-tenant document/chunk pairing.
- `chunk_embeddings` uses a composite foreign key `(chunk_id, workspace_id)`.
- Document IDs are deterministic UUIDv5 values scoped by `workspace_id + content_hash`, so identical content in two workspaces produces different document IDs.
- Source storage paths include a sanitized workspace component plus a workspace hash.
- A live cross-tenant test is provided but skipped unless `RETRIEVAL_TEST_DATABASE_URL` is set.

### 2.5 Database utilities

- `retrieval/db/connection.py` — lazy asyncpg import, pool lifecycle, extension/schema health report, readiness wait.
- `retrieval/db/migrate.py` — migration discovery, checksum validation, advisory lock, per-migration transaction.
- `retrieval/db/repository.py` — workspace/document/chunk/job persistence.
- `python -m retrieval.db` — CLI for applying migrations and printing health.

Example:

```bash
python -m retrieval.db --migrate --health
```

---

## 3. Ingestion Pipeline

### 3.1 Docling adapter

`retrieval/ingestion/docling_parser.py` uses IBM Docling through a lazy `DocumentConverter` import.

It extracts:

- Normalized text and headings.
- Section hierarchy.
- Page numbers from Docling provenance.
- Table Markdown via `export_to_markdown`.
- Document title, page count, and conversion metadata.

The adapter accepts an injected converter, which allowed parser/chunker tests to run without installing the heavy Docling/ML stack.

### 3.2 Normalization and chunking

- Unicode NFKC normalization, line-ending normalization, control-character removal, and blank-line cleanup.
- Structure-aware chunking with configurable target/max/overlap character budgets.
- Headings start new sections.
- Tables are emitted as standalone chunks and are never split mid-table.
- Stable deterministic chunk IDs: `<doc_id>:c00000`, `<doc_id>:c00001`, ...
- Page ranges and block types are retained in chunk metadata.

### 3.3 Storage and deduplication

- Content-addressed storage: `<workspace>/<hash-prefix>/<sha256>.<ext>`.
- Sanitized workspace directory and Windows reserved-name handling.
- Symlink uploads rejected; path traversal rejected on write and resolve.
- Atomic temporary-file write followed by `os.replace`.
- SHA-256 rechecked while copying to detect source changes.
- Duplicate detection by `(workspace_id, content_hash)`.
- Duplicate ingestion returns the existing document and chunks without re-parsing.
- Failed persistence deletes a newly-created storage object; pre-existing objects are not deleted.

### 3.4 Job tracking

`ingestion_jobs` records queued/running/completed/failed status, source filename, document ID, error text, timestamps, and JSON statistics. Jobs are workspace-scoped.

---

## 4. Compatibility

- `retrieval/docling_parser.py` remains import-compatible and now delegates to the real Docling adapter/chunker.
- `retrieval.docling_parser.DocumentChunk` remains available via re-export.
- The old `_generate_simulated_chunks` helper remains only as a deprecated test fixture and is not used by `parse_document`.
- `retrieval/__init__.py` now lazily exports heavy modules, avoiding eager Docling/asyncpg/ML imports.
- `core.schemas.ChunkEvidence` was not changed.
- Member 3's `LightRAGGraphEngine` contract was not changed.

---

## 5. Verification Performed

### 5.1 Static checks

```text
python -m compileall -q retrieval tests
compileall: OK

git diff --check
No whitespace errors; only Git's CRLF conversion warnings on Windows.
```

### 5.2 Unit tests

A temporary virtual environment outside the repository was used with `pydantic 2.14.3`, `pydantic-settings`, and `pytest 8.4.2`.

```bash
python -m pytest \
  tests/test_retrieval_db.py \
  tests/test_retrieval_ingestion.py \
  tests/test_retrieval_workspace_isolation.py \
  tests/test_retrieval_member2.py -q
```

Result:

```text
18 passed, 1 skipped
```

The skipped test is the opt-in live PostgreSQL workspace-isolation test. It is skipped because `RETRIEVAL_TEST_DATABASE_URL` is not set.

### 5.3 What the tests cover

- DSN normalization.
- Migration discovery, ordering, checksums, and required schema/index text.
- Repository workspace-filter SQL.
- Rejection of cross-workspace chunks.
- Docling adapter page/section/table extraction using an injected fake converter.
- Deterministic chunking and no table splitting.
- Content-addressed storage, duplicate object reuse, and workspace-separated paths.
- Pipeline deduplication within one workspace and isolation between workspaces.
- Failed conversion marks the job failed and raises `IngestionError`.
- Missing workspace is rejected.
- Workspace ID traversal rejection.
- Workspace-scoped deterministic document IDs.

---

## 6. Environment Limitations / Not Verified

These items are **not** claimed as verified:

1. **No live PostgreSQL run.** Docker CLI is installed, but the daemon is not running:
   `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.`
   Therefore migrations, extensions, BM25 index creation, HNSW index creation, and live workspace isolation were not executed against a real server.
2. **No real Docling conversion.** Docling was not installed in this environment. The adapter was tested with an injected converter; real PDF/DOCX/table extraction still needs a Docling-installed integration run.
3. **No binary fixtures.** PDF/DOCX fixtures were not added because validating them requires the heavy Docling stack.
4. **Full repository test suite not run.** Other members' runtime dependencies are not installed in the current environments.
5. **`pg_search` query behavior not exercised.** The index is created by migration, but no BM25 query code exists in Phase 2 (by design).
6. **`docling_ocr` is best-effort.** The setting is applied through `PdfPipelineOptions` when the installed Docling version supports it; otherwise the parser falls back to Docling defaults.

---

## 7. Files Added / Changed

### Added

- `retrieval/config.py`
- `retrieval/schemas.py`
- `retrieval/db/__init__.py`
- `retrieval/db/__main__.py`
- `retrieval/db/connection.py`
- `retrieval/db/migrate.py`
- `retrieval/db/repository.py`
- `retrieval/db/migrations/0001_extensions.sql`
- `retrieval/db/migrations/0002_core_schema.sql`
- `retrieval/db/migrations/0003_search_indexes.sql`
- `retrieval/ingestion/__init__.py`
- `retrieval/ingestion/chunker.py`
- `retrieval/ingestion/docling_parser.py`
- `retrieval/ingestion/metadata.py`
- `retrieval/ingestion/normalizer.py`
- `retrieval/ingestion/pipeline.py`
- `retrieval/ingestion/storage.py`
- `retrieval/requirements-ingestion.txt`
- `retrieval/requirements-ingestion-dev.txt`
- `retrieval/docker-compose.yml`
- `retrieval/.env.ingestion.example`
- `retrieval/.gitignore`
- `tests/test_retrieval_db.py`
- `tests/test_retrieval_ingestion.py`
- `tests/test_retrieval_workspace_isolation.py`
- `retrieval/reports/PHASE_2_FOUNDATION_INGESTION_REPORT.md`

### Modified

- `retrieval/__init__.py` — lazy exports; no eager heavy imports.
- `retrieval/docling_parser.py` — compatibility shim over the real ingestion path.

### Untouched / preserved

- `myenv/`
- `retrieval/IMPLEMENTATION_AUDIT.md`
- All shared Member 1/3/4 files and root configuration.

---

## 8. Recommended Phase 2 Verification on a Working Host

```bash
# 1. Start the pinned database service.
docker compose -f retrieval/docker-compose.yml up -d

# 2. Install the ingestion stack.
python -m pip install -r retrieval/requirements-ingestion.txt

# 3. Apply migrations and inspect health.
python -m retrieval.db --migrate --health

# 4. Run the opt-in live isolation test against a disposable database.
$env:RETRIEVAL_TEST_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/calibrag"
python -m pytest tests/test_retrieval_workspace_isolation.py -q
```

Then add one real PDF, DOCX, and Markdown fixture and run the ingestion pipeline with Docling installed.

---

## 9. Next Phase

Phase 3 should implement BGE-M3 embeddings and pgvector persistence/retrieval using the existing `chunk_embeddings` table and HNSW index. Phase 4 should add `pg_search` BM25 queries and reuse the existing correct RRF logic.

No commit or push was performed in this phase.
