# Phase 2.1 Verification Report - Real Database, Docling, and End-to-End Ingestion

**Branch:** `member-2-retrieval`  
**Date:** 2026-11-06  
**Owner:** Member 2 - Data, Hybrid RAG & GraphRAG Lead  
**Status:** Phase 2.1 verification complete. Real PostgreSQL, real Docling, and real end-to-end ingestion all passed.

---

## 1. Environment

| Component | Verified value |
| :--- | :--- |
| OS | Windows (`win32`) |
| Python | 3.11.6 |
| Docker Engine | 28.4.0 |
| Docker Compose | v2.39.2-desktop.1 |
| PostgreSQL | 17.11 (Debian package, x86_64) |
| Docker image | `paradedb/paradedb:0.26.5-pg17` |
| Container | `calibrag-retrieval-postgres-1` |
| Database | `calibrag` at `localhost:5432` |
| Docling | 2.148.0 |
| PyTorch | 2.15.0+cpu |
| asyncpg | 0.32.0 |
| pytest | 8.4.2 |

Docker Desktop was started successfully. Direct non-elevated start of `com.docker.service` failed, but launching Docker Desktop started the engine and the compose stack worked normally. First-run Docling model downloads used a local Hugging Face cache outside the repository.

No Phase 3 work was started. No embeddings, pgvector retrieval, BM25 queries, RRF, reranking, or LightRAG code was added.

---

## 2. PostgreSQL Verification

The compose service was started with:

```powershell
docker compose -f retrieval/docker-compose.yml up -d
```

The container reached `health=healthy`. Real database connectivity was verified with:

```powershell
python -m retrieval.db --health
```

Actual output confirmed:

- `connected: true`
- `database: calibrag`
- `schema_version: 0003`
- `ok: true`
- `missing_extensions: []`

The real database CLI migrated the schema and the repository tests performed real inserts, upserts, chunk reads, duplicate checks, job updates, and workspace-scoped lookups.

---

## 3. Extensions

Verified from the live database:

| Extension | Installed version |
| :--- | :--- |
| `vector` (pgvector) | 0.8.6 |
| `pg_search` (ParadeDB) | 0.26.5 |

Both extensions are installed and available in the `calibrag` database.

---

## 4. Migrations

The real migration runner applied:

| Version | Name |
| :--- | :--- |
| 0001 | extensions |
| 0002 | core_schema |
| 0003 | search_indexes |

The first real run reported migrations `0001`, `0002`, and `0003` as applied. Subsequent verification runs reported them as already applied, confirming checksummed migration state.

Core tables verified: `workspaces`, `documents`, `document_chunks`, `ingestion_jobs`, `chunk_embeddings`, and `schema_migrations`.

Search indexes verified with their actual definitions:

```sql
CREATE INDEX chunk_embeddings_hnsw_idx
ON public.chunk_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m='16', ef_construction='64');

CREATE INDEX document_chunks_bm25_idx
ON public.document_chunks
USING bm25 (chunk_id, content)
WITH (key_field=chunk_id);
```

This confirms the Phase 3 schema foundation exists, but no retrieval query or embedding generation was exercised.

---

## 5. PDF Conversion

Real Docling conversion of `verification_report.pdf` succeeded after the parser fix.

| Check | Result |
| :--- | :--- |
| Conversion status | Success |
| Title | `Quarterly Operations Review` |
| Page count | 2 |
| Structural blocks | 7 |
| Block types | heading, table, text |
| Headings | `Quarterly Operations Review`, `Revenue Summary`, `Risk Factors` |
| Tables | 2 |
| Table output | Markdown tables preserved |
| Page numbers | 1 and 2 |
| Chunks | 5 |
| Table chunks | 2 |
| Chunk pages | 1 and 2 |
| Metadata | `docling_used=true`, source extension `.pdf`, 7 blocks, success status |

The extracted tables included the regional revenue table and the risk table as valid Markdown. Chunk metadata preserved `page_start`, `page_end`, and block types. Section hierarchies were present for all non-table chunks.

---

## 6. DOCX Conversion

Real Docling conversion of `verification_report.docx` succeeded.

| Check | Result |
| :--- | :--- |
| Conversion status | Success |
| Title | `Quarterly Operations Review` |
| Page count | 1 |
| Structural blocks | 7 |
| Headings | `Quarterly Operations Review`, `Revenue Summary`, `Risk Factors` |
| Tables | 2 |
| Chunks | 5 |
| Table chunks | 2 |
| Metadata | `docling_used=true`, source extension `.docx`, 7 blocks, success status |

DOCX chunk hierarchies included the document title plus section headings. The two tables were preserved as Markdown and emitted as dedicated table chunks.

---

## 7. Markdown Conversion

Real Docling conversion of `verification_report.md` succeeded.

| Check | Result |
| :--- | :--- |
| Conversion status | Success |
| Title | `Quarterly Operations Review` |
| Page count | 1 |
| Structural blocks | 7 |
| Headings | `Quarterly Operations Review`, `Revenue Summary`, `Risk Factors` |
| Tables | 2 |
| Chunks | 5 |
| Table chunks | 2 |
| Metadata | `docling_used=true`, source extension `.md`, 7 blocks, success status |

Markdown tables were preserved as Markdown and emitted as dedicated table chunks. Section hierarchies were present for all non-table chunks.

---

## 8. End-to-End Ingestion

A real end-to-end verification run used the actual ingestion pipeline, real Docling conversion, real filesystem storage, and the live PostgreSQL database. It executed 53 checks, all of which passed.

Flow exercised: file validation, SHA-256 content hashing, workspace setup, job creation, real Docling conversion, normalization, deterministic chunking, content-addressed storage, document/chunk persistence, duplicate detection, workspace isolation, and failure recording.

Results from the successful run:

| Check | Result |
| :--- | :--- |
| Documents persisted | 4 |
| Chunks persisted | 20 |
| Completed ingestion jobs | 5 |
| Failed ingestion jobs | 1 |
| PDF pages | 2 |
| PDF chunks | 5 |
| DOCX chunks | 5 |
| Markdown chunks | 5 |
| Duplicate PDF chunks after re-ingest | 5, unchanged |
| Cross-workspace PDF chunks | 5 |

Additional verified behavior:

- Document metadata included `docling_used`, `source_type`, `source_filename`, `workspace_id`, `chunk_count`, and source file details.
- Stored source objects existed at their workspace-scoped storage paths.
- Chunk records preserved workspace ID, document ID, chunk index, page numbers, section hierarchy, table flags, table Markdown, character counts, token estimates, source type, and metadata.
- Re-ingesting the same PDF in the same workspace detected a duplicate, reused the original document ID, left the chunk count unchanged, and completed an explicitly created job with `duplicate=true`.
- A corrupt PDF raised `IngestionError`, recorded a failed job with phase `converting`, preserved the original `DocumentParseError`, and created no document row.
- All temporary verification workspaces were removed afterward; direct SQL confirmed 0 remaining `e2e_%` workspaces and 0 rows in documents, chunks, and jobs.

---

## 9. Workspace Isolation

Real database verification confirmed tenant isolation:

- The same PDF content produced different document IDs in workspace A and workspace B.
- Both workspaces stored the same content hash but resolved to their own document records.
- Looking up workspace A's document ID from workspace B returned `None`.
- Reading chunks for workspace A's document ID from workspace B returned an empty list.
- Workspace B returned only its own 5 chunks, all scoped to workspace B.
- Cross-workspace hash lookups resolved only the requesting workspace's document.
- Direct SQL counts confirmed workspace-scoped documents and chunks.
- The live pytest test `test_live_postgres_workspace_isolation` passed against real PostgreSQL.

---

## 10. Tests Passed and Failed

### Repository test suite

Command:

```powershell
$env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:5432/calibrag'
python -m pytest tests/test_retrieval_db.py tests/test_retrieval_ingestion.py tests/test_retrieval_workspace_isolation.py tests/test_retrieval_member2.py -q -p no:cacheprovider
```

Result:

```text
20 passed in 16.12s
```

Breakdown:

| Test file | Tests |
| :--- | :--- |
| `tests/test_retrieval_db.py` | 6 |
| `tests/test_retrieval_ingestion.py` | 7 |
| `tests/test_retrieval_workspace_isolation.py` | 3, including 1 live PostgreSQL test |
| `tests/test_retrieval_member2.py` | 4 |

The live PostgreSQL isolation test ran against the real container and passed. The new Docling PDF-option regression test also ran against the installed Docling package and passed.

### Real Docling verification

- PDF: passed
- DOCX: passed
- Markdown: passed
- Format failures: 0

### Real end-to-end verification

- Checks passed: 53
- Checks failed: 0

### Static verification

`python -m compileall -q retrieval tests` completed without errors.

---

## 11. Bugs and Fixes

### Product bug: PDF pipeline option type

The first real PDF conversion failed with:

```text
DocumentParseError: Docling failed to convert verification_report.pdf:
AttributeError: 'PdfPipelineOptions' object has no attribute 'backend'
```

Root cause: `format_options={InputFormat.PDF: pdf_options}` passed a raw `PdfPipelineOptions` object. Docling expects a `PdfFormatOption` wrapper containing the pipeline options.

Fix in `retrieval/ingestion/docling_parser.py`:

```python
self._converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
    }
)
```

A regression test was added to `tests/test_retrieval_ingestion.py`:

```text
test_docling_pdf_options_use_pdf_format_option
```

After the fix, real PDF conversion, the repository suite, and the full end-to-end run all passed.

### Verification harness corrections

The temporary end-to-end verification script required three harness-only corrections: serializing `MigrationRun`, decoding asyncpg JSONB stats, and correcting the expected number of completed jobs from 4 to 5. These were verification-script bugs, not application defects, and required no production code change.

---

## 12. Files Modified

Changes made during Phase 2.1 verification:

- `retrieval/ingestion/docling_parser.py` - fixed PDF format option construction.
- `tests/test_retrieval_ingestion.py` - added the PDF-option regression test.
- `retrieval/reports/PHASE_2_VERIFICATION_REPORT.md` - this report.

No shared files were modified: `pyproject.toml`, root `.env.example`, root Docker configuration, `core/schemas.py`, `api/`, and other members' files are unchanged. No commit or push was performed.

---

## 13. Remaining Limitations

- The verification PDF is text-based. OCR is enabled by default, but a scanned/image-only PDF was not specifically tested.
- Docling reports DOCX and Markdown documents as one page; PDF page provenance is multi-page and was verified.
- The live end-to-end verifier and its generated fixtures live in a temporary directory and are not committed to the repository.
- Live verification requires Docker Desktop and network access for the first Docling model download; subsequent runs can use the local cache.
- Phase 3 retrieval behavior remains intentionally unimplemented and unverified in this phase.

No unresolved Phase 2.1 blocker remains.

---

## 14. Final Phase 3 Recommendation

Phase 2.1 verification is complete and successful:

- Real PostgreSQL 17.11 is healthy.
- Required extensions are installed.
- Migrations are applied at version `0003`.
- HNSW and BM25 indexes exist with the intended definitions.
- Real PDF, DOCX, and Markdown conversion works.
- Real end-to-end ingestion persists documents and chunks correctly.
- Duplicate handling, workspace isolation, job success, and job failure behavior are verified.
- The repository suite passes 20/20 tests, including the live PostgreSQL isolation test.

Recommendation: proceed to Phase 3 only after explicit approval. The next phase can safely build embedding generation and hybrid retrieval on the verified schema and ingestion contract.
