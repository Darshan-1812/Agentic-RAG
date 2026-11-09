# CalibRAG — Member 2 Implementation Audit

**Scope:** Data, BYOK Ingestion, Hybrid RAG, GraphRAG, Retrieval Evaluation
**Phase:** Phase 1 — Repository Audit and Implementation Planning (no implementation)
**Assigned branch:** `member-2-retrieval`
**Audit date:** 2026-11-06
**Repository:** https://github.com/Darshan-1812/Agentic-RAG
**Auditor:** Member 2 implementation agent

---

## Executive Summary

The repository currently contains a **scaffolded, demonstration-oriented Member 2 package**. The public
interfaces, Pydantic schemas, an RRF algorithm, a simple text chunker, an in-memory hybrid-search stub,
a hard-coded graph traversal, and a synthetic Recall@K report exist. **No real IBM Docling parsing,
no PostgreSQL connection, no pgvector/HNSW index, no ParadeDB BM25 index, no BGE-M3 embeddings, no
cross-encoder model, no LightRAG library integration, no persistence, and no workspace isolation are
implemented.** The current tests are mock-level smoke tests and cannot be executed in this environment
because no project dependencies are installed.

The branch `member-2-retrieval` exists on the remote and is currently **identical to `main`
(commit `2ddd481`)**. No Member 2-specific commits exist beyond the shared architecture commit.

---

## 1. Current Branch and Git Status

### 1.1 Pre-flight state (before work)

| Item | Value |
| :--- | :--- |
| Initial branch | `main` |
| Untracked files | `myenv/` (empty Python venv: `Include/`, `Lib/`, `Scripts/`, `pyvenv.cfg`) |
| Tracked modifications | None |
| Staged changes | None |
| Conflicts | None |
| Remote | `origin` → `https://github.com/Darshan-1812/Agentic-RAG.git` |

The untracked `myenv/` was reported before switching. The user explicitly authorized switching to
`member-2-retrieval` while leaving `myenv/` untouched and untracked. Nothing was discarded, stashed,
reset, or overwritten.

### 1.2 Post-switch state

| Item | Value |
| :--- | :--- |
| Active branch | `member-2-retrieval` |
| Tracking | `origin/member-2-retrieval` |
| HEAD commit | `2ddd481` — `feat(calibrag): update whole project architecture to CalibRAG across all 4 members` |
| Working tree | Only untracked `myenv/` |
| `origin/member-2-retrieval` | Exists, at `2ddd481` |
| `origin/main` | `2ddd481` (same commit) |
| `origin/member-1-control-plane` | `2ddd481` (same commit) |
| `origin/member-4-ui-eval` | `2ddd481` (same commit) |
| `origin/member-3-agents-mcp` | `e094f2c` — **one commit ahead** of the shared baseline |

### 1.3 Branch history

```
e094f2c  origin/member-3-agents-mcp  feat(member-3): LangGraph supervisor, FastMCP tools, SSE proxy
2ddd481  HEAD -> member-2-retrieval, origin/member-2-retrieval, origin/main,
         origin/member-1-control-plane, origin/member-4-ui-eval,
         origin/member-3-agents-mcp^  feat(calibrag): update whole project architecture to CalibRAG
6b48e77  feat(acr-rag): initial architecture scaffolding and Member 1 Control Plane implementation
```

**Implications:**

- The Member 2 branch contains **no Member 2 implementation commit**; all `retrieval/*.py` files were
  introduced by the shared `2ddd481` commit.
- Member 3 has an additional unpushed-relative-to-main commit (`e094f2c`). Do **not** merge or rebase
  it into `member-2-retrieval` without team coordination; its interface changes may affect retrieval.
- `main` must not be modified. All Member 2 work belongs on `member-2-retrieval`.

---

## 2. Complete Repository Structure

Tracked files at `2ddd481`:

```
Agentic-RAG/
├── .env.example
├── .gitignore
├── ACR_RAG_Product_and_Architecture_Blueprint.pdf
├── CalibRAG_Product_and_Architecture_Blueprint.pdf
├── pyproject.toml
├── README.md
├── agents/                          # Member 3
│   ├── __init__.py
│   ├── interfaces.py
│   ├── langgraph_supervisor.py
│   └── README.md
├── api/                             # Member 3
│   ├── __init__.py
│   ├── openai_proxy.py
│   ├── server.py
│   └── README.md
├── benchmarks/                      # Member 4
│   ├── __init__.py
│   ├── 5_baseline_results.json
│   ├── generate_pareto_curves.py
│   ├── run_5_baseline_benchmark.py
│   └── README.md
├── core/                            # Shared
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   └── telemetry.py
├── evals/                           # Member 1 / shared
│   ├── __init__.py
│   ├── calibration_results.json
│   ├── run_calibration_eval.py
│   └── test_queries.py
├── frontend/                        # Member 4
│   ├── components/
│   │   ├── BYOKWorkspaceStudio.tsx
│   │   └── SplitScreenPlayground.tsx
│   ├── package.json
│   └── README.md
├── mcp_servers/                     # Member 3
│   ├── github_issues/server.py
│   ├── postgres_erp/server.py
│   └── README.md
├── retrieval/                       # Member 2  ← ownership
│   ├── __init__.py
│   ├── docling_parser.py
│   ├── interfaces.py
│   ├── lightrag_indexer.py
│   ├── pg_hybrid_search.py
│   ├── recall_ablation.py
│   ├── recall_ablation_results.json
│   └── README.md
├── router/                          # Member 1
│   ├── __init__.py
│   ├── fallback_router.py
│   ├── jev_client.py
│   ├── policy_manager.py
│   └── pre_retrieval_router.py
├── tests/
│   ├── __init__.py
│   ├── test_abstention.py
│   ├── test_agents_member3.py
│   ├── test_benchmarks_member4.py
│   ├── test_confidence_gate.py
│   ├── test_retrieval_member2.py
│   ├── test_router.py
│   ├── test_scenario_benchmark.py
│   ├── test_telemetry.py
│   └── test_verifier.py
└── verifier/                        # Member 1
    ├── __init__.py
    ├── abstention_engine.py
    ├── calibration_metrics.py
    ├── claim_verifier.py
    └── confidence_gate.py
```

**Not present at all:**

- `Dockerfile`, `docker-compose.yml`, container orchestration
- Any `.sql` schema, migration, Alembic config, or seed SQL
- Any `requirements*.txt`, lockfile, or environment specification beyond `pyproject.toml`
- Any local document corpus / fixtures for real ingestion tests
- Any `models/`, `data/`, `storage/`, `migrations/`, `scripts/` directories
- Frontend Next.js app directories (`app/`, `pages/`, `tsconfig.json`, `next.config.*`, lockfile)
- `.env` (gitignored; not present in working tree)

> **Note:** `retrieval/` exists, so the report location is valid.

---

## 3. Existing Member 2 Files and Responsibilities

| File | Current responsibility | Reality |
| :--- | :--- | :--- |
| `retrieval/__init__.py` | Aggregates and re-exports all retrieval classes/functions | Eager imports; adding heavy optional dependencies here will affect every `import retrieval` (including Member 3 agents/API) |
| `retrieval/interfaces.py` | Four ABCs: `BaseVectorRetriever`, `BaseBM25Retriever`, `BaseHybridReranker`, `BaseGraphRAGRetriever` | Contracts only; signatures omit workspace scoping for graph traversal; no async variants; imports `ChunkEvidence` from `core.schemas` |
| `retrieval/docling_parser.py` | `DocumentChunk` schema + `DoclingDocumentParser` | Does **not** use Docling. Reads files as UTF-8 text and regex-splits paragraphs. Non-existent paths return hard-coded demo chunks. PDF/DOCX binary input produces garbage. `workspace_id` argument is ignored. |
| `retrieval/pg_hybrid_search.py` | `PGHybridSearchEngine` implementing all three retrieval ABCs | Purely in-memory. "Vector search" = word-overlap heuristic. "BM25" = `str.count` weighting. "Cross-encoder" = token-match heuristic plus demo keyword boosts. RRF algorithm is genuine and reusable. No SQL, no PostgreSQL, no embeddings. `workspace_id` is ignored. |
| `retrieval/lightrag_indexer.py` | `GraphNode`, `GraphEdge`, `LightRAGGraphEngine` | Does **not** use the LightRAG library. Graph is a hard-coded 9-node/6-edge demo seeded in `__init__`. Traversal is a basic BFS over the in-memory list. No entity/relation extraction, no persistence, no workspace partitioning. |
| `retrieval/recall_ablation.py` | Recall@K/MRR ablation report producer | Returns hard-coded metric dictionaries. `n_queries` and `seed` are unused. No retrieval is executed, no ground-truth labels exist. Not a valid evaluation. |
| `retrieval/recall_ablation_results.json` | Saved output of the above | Synthetic numbers copied from the same constants. |
| `retrieval/README.md` | Member 2 scope documentation | Describes intended features as if implemented; useful as a requirements statement only. |
| `tests/test_retrieval_member2.py` | Four smoke tests | Calls `_generate_simulated_chunks` (private mock path), in-memory hybrid engine, hard-coded graph, and synthetic ablation thresholds. Does not exercise any real system. |

### 3.1 Public API surface currently exported

From `retrieval/__init__.py`:

- `BaseVectorRetriever.retrieve(query, top_k=5, workspace_id="default") -> List[ChunkEvidence]`
- `BaseBM25Retriever.search(query, top_k=5, workspace_id="default") -> List[ChunkEvidence]`
- `BaseHybridReranker.rerank(query, chunks, top_k=5) -> List[ChunkEvidence]`
- `BaseGraphRAGRetriever.traverse_entities(query, entity_seeds, max_hops=2) -> Dict[str, Any]`
- `DoclingDocumentParser.parse_document(file_path, doc_id=None, workspace_id="default_workspace") -> List[DocumentChunk]`
- `PGHybridSearchEngine` with `index_chunks`, `retrieve`, `search`, `rerank`, `reciprocal_rank_fusion`
- `LightRAGGraphEngine` with `add_node`, `add_edge`, `traverse_entities`
- `run_retrieval_recall_ablation`, `format_markdown_table`

---

## 4. Implemented vs. Partially Implemented vs. Missing

Legend: **DONE** = production-ready; **PARTIAL** = skeleton/demo/mock; **MISSING** = absent.

### A. BYOK Ingestion

| Capability | Status | Evidence / gap |
| :--- | :--- | :--- |
| IBM Docling integration | **MISSING** | `docling_parser.py:43-45` only contains commented-out Docling usage. `docling` is not a dependency. |
| PDF support | **MISSING** | `docling_parser.py:53-57` opens every path in text mode with `errors="ignore"`; real PDF bytes are not parsed. |
| DOCX support | **MISSING** | Same text-mode read; DOCX is a ZIP container and will not parse. |
| Markdown support | **PARTIAL** | Plain `.md` text can be split by blank lines, but headings/tables/front-matter are not preserved. |
| Document normalization | **MISSING** | No unicode normalization, whitespace cleanup, header/footer removal, table normalization, or language handling. |
| Structure-aware chunking | **PARTIAL** | Naive paragraph accumulation with `chunk_size`/`overlap` fields. `overlap` is never used. No heading hierarchy, table-aware splitting, sentence boundaries, or token-based sizing. |
| Page provenance | **PARTIAL** | `page_number` exists, but is inferred from paragraphs containing the word "page" (`docling_parser.py:71-75`). No real page extraction. |
| Document metadata | **PARTIAL** | Only `chunk_id`, `doc_id`, `doc_title`, counts. No filename, MIME type, hash, size, created/modified time, author, version, or ingestion status. |
| Workspace-level isolation | **MISSING** | `DocumentChunk` has no `workspace_id`; parser ignores the parameter; no storage layer enforces tenancy. |
| File persistence / original storage | **MISSING** | No upload directory, object store, dedupe, or lifecycle management. |
| Ingestion job tracking / idempotency | **MISSING** | No job table, content hashing, re-ingestion, deletion, or versioning. |
| Ingestion API endpoint | **MISSING** | No upload endpoint exists in `api/` or `retrieval/`. Frontend drop zone is static (`BYOKWorkspaceStudio.tsx:34-45`). |

### B. Hybrid Retrieval

| Capability | Status | Evidence / gap |
| :--- | :--- | :--- |
| PostgreSQL 17 connectivity | **MISSING** | `DATABASE_URL` exists in config (`core/config.py:63`, `.env.example:18`) but is never read by retrieval. No driver (`asyncpg`/`psycopg`) installed or declared. |
| Database schema | **MISSING** | No tables for workspaces, documents, chunks, embeddings, or graph. |
| pgvector + HNSW | **MISSING** | No `vector` extension, no `vector(1024)` column, no HNSW index, no cosine operator query. `retrieve()` is word overlap (`pg_hybrid_search.py:32-42`). |
| ParadeDB `pg_search` BM25 | **MISSING** | No `pg_search` extension, no `USING bm25` index, no `@@@` query. `search()` is `str.count` (`pg_hybrid_search.py:49-57`). |
| BAAI/bge-m3 embeddings | **MISSING** | No embedding model, tokenizer, batching, normalization, or model download. No `sentence-transformers`/`FlagEmbedding` dependency. |
| Reciprocal Rank Fusion | **PARTIAL / REUSABLE** | `reciprocal_rank_fusion` (`pg_hybrid_search.py:90-111`) is a correct RRF implementation with `k=60`; it operates on already-retrieved `ChunkEvidence` lists and ignores workspace. Candidate retrieval depth is not configurable. |
| Cross-encoder reranking | **MISSING** | `rerank()` (`pg_hybrid_search.py:59-88`) is a hand-written token-overlap heuristic with demo keyword boosts. No `bge-reranker-v2-m3`, no model inference, no batched logits. |
| Unified hybrid orchestration | **MISSING** | No single method performs `vector -> BM25 -> RRF -> rerank -> top_k`. Callers must invoke three methods manually. |
| Workspace-filtered retrieval | **MISSING** | `workspace_id` accepted by `retrieve`/`search` but never used. Cross-tenant leakage risk. |
| Async / connection pooling | **MISSING** | All methods are synchronous and in-memory. `DATABASE_URL` uses `postgresql+asyncpg://`, implying async, but no async code exists. |
| Score semantics for Member 1 | **PARTIAL** | `ChunkEvidence.cross_encoder_logit` is populated by the heuristic and range-clamped implicitly; no calibration of raw logits. |

### C. GraphRAG (LightRAG)

| Capability | Status | Evidence / gap |
| :--- | :--- | :--- |
| LightRAG library integration | **MISSING** | No `lightrag` dependency or import. `lightrag_indexer.py` is a custom in-memory graph. |
| Entity and relationship extraction | **MISSING** | Graph is hand-seeded (`lightrag_indexer.py:36-55`). No LLM extraction, prompt templates, gleaning, or entity resolution. |
| Knowledge graph indexing | **MISSING** | No persistent node/edge storage; graph resets on process restart. |
| Workspace-specific graph retrieval | **MISSING** | `traverse_entities` has no `workspace_id`; interface lacks it; single global graph shared by all tenants. |
| 2-hop traversal | **PARTIAL** | BFS over hard-coded edges works for the demo (`lightrag_indexer.py:63-121`), but only for seeded data. Unknown IDs in edges can raise `KeyError` when building summaries. |
| Integration with Member 3 supervisor | **PARTIAL** | `LangGraphSupervisor` instantiates `LightRAGGraphEngine()` with default seeded graph and uses `context_summary`/`nodes` (`agents/langgraph_supervisor.py:77,110-112`). Must remain backward compatible or be coordinated. |
| Graph context into retrieval ranking | **MISSING** | Graph output is a text summary only; it is not fused/reranked with chunk retrieval. |

### D. Evaluation

| Capability | Status | Evidence / gap |
| :--- | :--- | :--- |
| Retrieval Recall@K implementation | **MISSING** | `recall_ablation.py` returns constants; no `qrels`, no relevance matching, no retrieved-ID comparison. |
| MRR@10 | **MISSING** | Constant only. |
| Ablation configurations | **MISSING** | Config names are hard-coded; no execution of Vector/BM25/Hybrid/Rerank/Graph variants. |
| Reproducible labeled dataset | **MISSING** | No corpus, queries, or relevance labels. |
| Real vs synthetic separation | **MISSING** | Synthetic "published" numbers are written to `recall_ablation_results.json`, which could be mistaken for empirical results. |

---

## 5. Existing Dependencies and Configuration

### 5.1 Declared dependencies (`pyproject.toml`)

```toml
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0",
    "httpx>=0.24.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
]
optional dev = ["pytest>=7.0.0", "pytest-asyncio>=0.20.0"]
```

**Missing dependencies required for Member 2 target features:**

- Ingestion: `docling` (which pulls layout/OCR/table models and likely `torch`), `python-docx` if a lighter DOCX path is chosen.
- Database: `asyncpg` (or `psycopg[binary]`), optionally `sqlalchemy>=2.0`, `pgvector` Python helper (optional if raw SQL).
- Embeddings/reranking: `sentence-transformers`, `FlagEmbedding` (if using BGE-M3 sparse/dense/colbert), `torch`, `transformers`, `huggingface-hub`.
- Graph: `lightrag-hku` (confirm exact package name and pinned version), plus its storage backend dependencies.
- Evaluation/tokenization: `tiktoken` or model tokenizer, `scikit-learn` (optional), `pandas` (optional for reports).

No version pins exist. There is no lockfile, so builds are not reproducible.

### 5.2 Runtime configuration

| Setting | Location | Current use |
| :--- | :--- | :--- |
| `DATABASE_URL` | `.env.example:18`, `core/config.py:63` | Declared as `postgresql+asyncpg://...`; **unused by retrieval**. |
| `POSTGRES_ERP_READONLY_URI` | `.env.example:21`, `core/config.py:64` | Member 3 MCP concern; separate ERP database. |
| `CALIBRAG_DEFAULT_SLA_POLICY`, thresholds | `.env.example:6-15` | Member 1 routing/verifier. |
| `NEXT_PUBLIC_CALIBRAG_API_URL` | `.env.example:27` | Frontend; no ingestion endpoint exists. |
| Embedding model name | **absent** | No `BGE_M3_MODEL`, dimension, batch size, device, cache path. |
| Reranker model name | **absent** | No `BGE_RERANKER_MODEL`, batch size, max length, device. |
| HNSW parameters | **absent** | No `m`, `ef_construction`, `ef_search`, distance metric. |
| LightRAG settings | **absent** | No storage backend, LLM, embedding function, working directory, graph namespace. |
| Document storage root | **absent** | No upload directory or size limits. |

### 5.3 Environment/execution status

- System Python: **3.11.6**. The untracked `myenv` also uses **3.11.6**.
- Neither interpreter has the project dependencies installed (`pydantic`, `fastapi`, `pytest`, etc. are
  absent). Only `pip` and `setuptools` are present in `myenv`; system Python likewise lacks project deps.
- **No tests were run** in this phase because the environment is not provisioned. This audit is based on
  static inspection, not execution.
- No `Dockerfile` / `docker-compose.yml` exists, so PostgreSQL 17 + pgvector + ParadeDB cannot be started
  from the repository.

---

## 6. Database Schema and Connection Status

### 6.1 Current status

- **Connection:** none. `DATABASE_URL` is a string in settings only.
- **Schema:** none. No SQL files, no migrations, no ORM models.
- **Persistence:** none. All retrieval state is process-local in `PGHybridSearchEngine._document_store`
  and `LightRAGGraphEngine.nodes/edges`.
- **Transaction/concurrency:** not applicable; no database code.
- **Workspace isolation:** not implemented at any layer.

### 6.2 Required database capabilities (recommendation, not implemented)

The chosen PostgreSQL image must provide **both** extensions:

- `vector` (pgvector) for dense embeddings and HNSW.
- `pg_search` (ParadeDB) for BM25. ParadeDB ships a PostgreSQL-compatible distribution; a vanilla
  `postgres:17` image will **not** have `pg_search`. This is a deployment decision.

Recommended logical schema (to be finalized with the team before coding):

| Table | Key columns | Notes |
| :--- | :--- | :--- |
| `workspaces` | `workspace_id PK`, `name`, `created_at`, `config JSONB` | Tenant root. |
| `documents` | `doc_id PK`, `workspace_id FK`, `filename`, `mime_type`, `content_hash`, `byte_size`, `page_count`, `status`, `created_at`, `metadata JSONB` | Dedupe by `(workspace_id, content_hash)`. |
| `document_pages` | `doc_id FK`, `page_number`, `text`, `layout JSONB` | Optional but useful for provenance. |
| `chunks` | `chunk_id PK`, `doc_id FK`, `workspace_id`, `content`, `page_number`, `section_hierarchy TEXT[]`, `chunk_index`, `has_table`, `table_markdown`, `metadata JSONB` | `workspace_id` denormalized for fast filtering. |
| `chunk_embeddings` | `chunk_id FK`, `embedding VECTOR(1024)` | BGE-M3 dense dimension is **1024**; verify before finalizing. |
| BM25 index | `USING bm25 (content)` on `chunks` | ParadeDB `pg_search`; query with `@@@ paradedb.match(...)`. |
| `graph_nodes` | `node_id PK`, `workspace_id`, `entity_name`, `entity_type`, `properties JSONB` | Unique per workspace. |
| `graph_edges` | `edge_id PK`, `workspace_id`, `source_id`, `target_id`, `relation_type`, `weight`, `provenance_doc` | Index `(workspace_id, source_id)`, `(workspace_id, target_id)`. |
| `entity_mentions` | `node_id`, `chunk_id`, `workspace_id` | Links graph back to evidence. |
| `ingestion_jobs` | `job_id PK`, `workspace_id`, `doc_id`, `status`, `error`, `started_at`, `finished_at` | Async job tracking. |

Additional requirements:

- Every retrieval query must filter `workspace_id` **in SQL**, not in Python.
- HNSW index example: `CREATE INDEX ... ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)`.
- BM25 index and text search must be validated against the deployed `pg_search` version.
- Optional PostgreSQL Row-Level Security can add defense-in-depth; not a substitute for mandatory filters.
- Schema changes must be versioned via migrations (Alembic or plain numbered SQL).

---

## 7. Existing API and Integration Interfaces

### 7.1 Retrieval interfaces (`retrieval/interfaces.py`)

- Synchronous only; no async support.
- `workspace_id` is present for vector/BM25 but absent for graph traversal.
- No single high-level retrieval facade. `ChunkEvidence` is the shared currency with Members 1 and 3.

### 7.2 Core schemas (`core/schemas.py`)

`ChunkEvidence` (shared, Member 1/3 depend on it):

```python
chunk_id: str
doc_id: str
content: str
cross_encoder_logit: float   # REQUIRED, no default
page_number: Optional[int]
source_type: str = "pdf"
metadata: Dict[str, Any]
```

Constraints and gaps:

- `cross_encoder_logit` is mandatory. Any ingestion/vector-only path must supply a value; current tests
  pass `0.0`. Its semantics must remain **raw logit** because Member 1's confidence gate computes
  `sigmoid(top1_logit - top5_logit)` (`verifier/confidence_gate.py:72-76`).
- No `workspace_id` field. Workspace scoping must happen before constructing `ChunkEvidence`, or the
  schema must be extended (shared change requiring coordination).
- `DocumentChunk` is not convertible to `ChunkEvidence`; no mapping function exists.

### 7.3 Member 1 integration (Router & Verifier)

- `EvidenceSufficiencyGate.evaluate(query, chunks, current_tier, workspace_id, sla_override)` consumes
  `List[ChunkEvidence]` and reads `.content`, `.doc_id`, `.cross_encoder_logit`
  (`verifier/confidence_gate.py:122-150`).
- It assumes chunks are already ordered/scored by a cross-encoder and that at least a top-5 margin is
  meaningful. Real reranking must return **raw logits** and preserve enough candidates.
- `AbstentionEngine` and `ClaimVerifier` also consume chunk content / cited doc IDs.

### 7.4 Member 3 integration (Agents & API)

- `LangGraphSupervisor` imports `LightRAGGraphEngine`, instantiates it with defaults, and calls
  `traverse_entities(query, entity_seeds=..., max_hops=2)` (`agents/langgraph_supervisor.py:10,77,110-112`).
- It expects a dict with `context_summary` and `nodes`; `nodes` entries contain `entity_name`.
- Seeds are hard-coded to `["VoltCore", "EV-Pack-V2", "BMS-Controller-v4"]`. No workspace ID is passed.
- `api/openai_proxy.py` **does not call retrieval at all**; Tier 1/2/3 answers are hard-coded
  (`api/openai_proxy.py:76-119`). Therefore there is currently no end-to-end path from the API to
  Member 2 retrieval.
- `api/server.py` mounts only the OpenAI proxy and `/health`; no ingestion routes.

### 7.5 Member 4 integration (Frontend & Benchmarks)

- `BYOKWorkspaceStudio.tsx` is a static component: no `fetch`, no file input handler, no API call.
  It displays hard-coded filenames.
- `benchmarks/run_5_baseline_benchmark.py` returns hard-coded metrics; it does not call retrieval.
- No ingestion API contract, upload payload, job-status endpoint, or workspace CRUD exists.

### 7.6 Interfaces that must be frozen before implementation

1. **Ingestion API contract:** endpoint path, multipart payload, workspace identity, async job vs.
   synchronous response, status codes, document/chunk response schema.
2. **Retrieval facade contract:** method name/signature for hybrid retrieval, async vs sync, return type,
   candidate depth, workspace parameter, and error behavior.
3. **`ChunkEvidence` evolution:** workspace ID, provenance fields, score semantics. Shared schema change.
4. **Graph interface:** add `workspace_id` while preserving Member 3's call signature (optional kwarg with
   backward-compatible default is the safest path).
5. **Ownership of ingestion routes:** Member 2 service logic vs. Member 3 `api/` route mounting.

---

## 8. Missing Tests

### 8.1 Current tests (Member 2)

`tests/test_retrieval_member2.py`:

- `test_docling_parser_provenance` — calls the private mock generator, not `parse_document`.
- `test_hybrid_search_rrf_and_cross_encoder` — in-memory heuristic only.
- `test_lightrag_2hop_traversal` — hard-coded graph only.
- `test_recall_ablation_metrics` — asserts hard-coded thresholds; passes without retrieval.

These tests encode the mock behavior. They must be rewritten/replaced, not merely extended, when real
implementations land. They also cannot run in the current environment (no dependencies installed).

### 8.2 Missing test coverage

**Ingestion**

- Real PDF, DOCX, and Markdown fixtures; assert text, tables, headings, and page numbers.
- Corrupt/encrypted/oversized files; unsupported MIME types; empty documents.
- Chunk boundary invariants: max size, overlap correctness, no mid-table splits, stable chunk IDs.
- Unicode/encoding normalization and metadata extraction.
- Duplicate upload idempotency via content hash.
- Workspace isolation: identical filenames in two workspaces do not collide or leak.
- Deletion/re-ingestion/versioning behavior.

**Database / vector / BM25**

- Schema migration idempotency and extension checks.
- Insert/read round-trip for documents, chunks, embeddings.
- HNSW cosine ordering vs. brute-force reference on a small fixture.
- `pg_search` BM25 result correctness and ranking sanity.
- Workspace filter enforcement on every query (negative cross-tenant tests).
- Dimension mismatch, null embedding, and transaction rollback behavior.
- Connection failure / retry / pool exhaustion handling.

**Hybrid / reranking**

- RRF with ties, disjoint lists, duplicate IDs, empty lists, and `top_k > candidates`.
- Candidate-depth behavior before reranking.
- Cross-encoder logits: monotonic relevance, raw-logit range, deterministic batching, CPU/GPU fallback.
- `top_k` contract and stable ordering/deduplication by document/page.
- End-to-end hybrid retrieval with `ChunkEvidence` feeding Member 1's gate.

**GraphRAG**

- Entity/relation extraction on fixture text (precision/recall expectations).
- Workspace-scoped graph isolation.
- 1-hop/2-hop traversal correctness, cycles, missing nodes, dangling edges.
- Graph persistence across process restart.
- Backward compatibility with `LangGraphSupervisor` call signature and return keys.

**Evaluation**

- Real labeled corpus (queries + qrels), Recall@K and MRR computation unit tests.
- Ablation configs actually execute the corresponding retrievers.
- Determinism given a seed; no synthetic fallback silently substituting results.

**Integration**

- API endpoint -> ingestion -> DB -> retrieval -> `ChunkEvidence` -> Member 1 gate.
- Member 3 supervisor -> LightRAG with workspace context.
- Concurrency test: two workspaces ingesting/querying simultaneously.

---

## 9. Technical Risks and Dependency Conflicts

| # | Risk | Impact | Mitigation |
| :-: | :--- | :--- | :--- |
| 1 | **No real implementation behind feature names** | Demo tests pass while production behavior is absent; research results are not defensible | Treat all current retrieval modules as scaffolding; replace internals behind stable interfaces |
| 2 | **Synthetic evaluation presented as empirical** | Academic/reporting integrity risk | Clearly separate synthetic fixtures from real evaluation; require qrels and executed retrievals |
| 3 | **ParadeDB is not vanilla PostgreSQL** | `postgres:17` image lacks `pg_search`; BM25 queries fail | Standardize on a ParadeDB image that bundles PG17 + pgvector + pg_search; pin image version |
| 4 | **pgvector dimension mismatch** | Wrong dimension silently breaks inserts or similarity | Confirm BGE-M3 dense dimension (1024) and enforce `VECTOR(1024)` |
| 5 | **Heavy ML dependencies** | `docling`, `torch`, `sentence-transformers`, `lightrag` are large; Windows/CPU installs are slow and fragile | Put them in optional extras/groups; lazy-load models; separate ingestion worker from lightweight API path |
| 6 | **Model download / offline CI** | Tests fail without Hugging Face network access | Cache models locally; mark model tests as integration/slow; provide deterministic fakes for unit tests |
| 7 | **LightRAG API/version drift** | Package name/API and storage backend behavior may change | Pin an exact version; isolate behind `retrieval/graph/` adapter; document supported version |
| 8 | **LightRAG storage backend choice** | Default may be NetworkX/JSON, not PostgreSQL; workspace isolation may be weak | Configure a workspace-scoped namespace/storage; verify isolation before claiming GraphRAG over Postgres |
| 9 | **Eager imports in `retrieval/__init__.py`** | Importing heavy models breaks `import retrieval` for API/agents/tests | Use lazy imports or keep heavy modules out of package `__init__` |
| 10 | **`ChunkEvidence.cross_encoder_logit` is required** | Vector/BM25-only paths and ingestion cannot emit chunks without inventing a score | Always produce a calibrated score; or extend schema carefully with team approval |
| 11 | **Shared `core.schemas` / `pyproject.toml` changes** | Cross-member breakage | Coordinate schema/dependency changes; prefer additive optional fields; do not edit Member 1/3 logic |
| 12 | **Member 3 branch is ahead (`e094f2c`)** | Agent/API interface changes may conflict with retrieval integration | Review Member 3 diff before integration; rebase/merge only with coordination |
| 13 | **No workspace isolation anywhere** | Cross-tenant data leakage; severe security/compliance issue | Mandatory `workspace_id` in schema, DTOs, SQL, and tests; deny missing workspace IDs |
| 14 | **No authentication/authorization** | Client-supplied `workspace_id` can be spoofed | Flag as a cross-team requirement; at minimum validate workspace existence; add auth before production |
| 15 | **Upload security** | Path traversal, zip bombs, malicious PDFs, oversized files, PII | Store outside web root; sanitize filenames; enforce size/type limits; sandbox Docling; define retention |
| 16 | **Windows development environment** | Path separators, file locks, model build issues | Use `pathlib`, UTF-8 explicitly, and document Windows-specific setup |
| 17 | **No lockfile / environment provisioning** | Irreproducible installs; tests cannot run | Add a lockfile or pinned constraints; document setup; avoid installing during audit |
| 18 | **Async inconsistency** | `DATABASE_URL` says `asyncpg` but code is sync/in-memory | Choose one execution model (recommend async for FastAPI; sync worker for ingestion) and document it |
| 19 | **No migrations** | Schema drift between developers/deployments | Adopt Alembic or numbered SQL migrations from the first commit |
| 20 | **Demo data/keyword boosts leak into production** | Incorrect behavior on real corpora | Remove hard-coded keyword boosts and simulated chunks from production paths; move to test fixtures |
| 21 | **Frontend is a static scaffold** | No real upload/status UX; Member 4 blocked on API | Provide a minimal ingestion contract early; keep frontend changes with Member 4 |
| 22 | **Backward compatibility with Member 3** | Changing `LightRAGGraphEngine` defaults/signature breaks `agents` and its tests | Keep current method/keys; add optional `workspace_id`; coordinate constructor changes |

---

## 10. Recommended Implementation Order

Implementation must proceed bottom-up so that each phase is independently testable and does not break
Members 1/3. Do **not** implement all of this at once; each phase should be a separate commit/PR on
`member-2-retrieval`.

### Phase 0 — Coordination and interface freeze (no code)

- Agree on ingestion API contract with Member 3 and Member 4.
- Agree on retrieval facade signature and async model.
- Agree on `ChunkEvidence` extension (workspace/provenance/score semantics) with Member 1 and Member 3.
- Agree on ParadeDB image and model pinning.
- Decide ownership of `POST /v1/workspaces/{workspace_id}/documents` route mounting.

### Phase 1 — Configuration, schema, and database foundation

- Add Member 2 configuration (without editing `core/config.py` if possible): model names, dimensions,
  DB pool sizes, storage root, HNSW/BM25 parameters.
- Add DB connection layer and migration for all tables/indexes.
- Add a reproducible local PostgreSQL 17 + pgvector + `pg_search` service.
- Health check for extensions and schema version.

### Phase 2 — Real ingestion pipeline

- Integrate IBM Docling for PDF/DOCX/Markdown.
- Add normalization, structure-aware chunking, page/section provenance, metadata, hashing.
- Add document/chunk persistence, workspace isolation, dedupe, ingestion jobs.
- Replace the simulated parser path; keep the old private method only as a test fixture if needed.

### Phase 3 — Embeddings and dense retrieval

- Implement BGE-M3 embedding provider with batching, normalization, caching, and lazy loading.
- Persist embeddings to pgvector with HNSW index.
- Implement workspace-filtered dense retrieval returning `ChunkEvidence`.

### Phase 4 — BM25 and hybrid orchestration

- Implement ParadeDB `pg_search` BM25 retrieval.
- Extract and reuse the existing correct RRF logic; add configurable candidate depth.
- Implement a single hybrid retrieval facade (`vector + BM25 -> RRF`).
- Add deduplication and deterministic ordering.

### Phase 5 — Cross-encoder reranking and Member 1 compatibility

- Integrate `bge-reranker-v2-m3` with batching and raw-logit output.
- Populate `ChunkEvidence.cross_encoder_logit` with true logits.
- Add end-to-end test feeding real retrieved chunks into Member 1's confidence gate.
- Validate latency on CPU and define GPU/CPU fallback.

### Phase 6 — LightRAG GraphRAG

- Integrate the pinned LightRAG library.
- Implement workspace-scoped entity/relation extraction and graph indexing.
- Persist graph data; verify restart durability.
- Adapt `traverse_entities` with optional `workspace_id`, preserving Member 3's existing call pattern.
- Optionally blend graph context into the hybrid retrieval facade.

### Phase 7 — Real retrieval evaluation

- Build a small labeled corpus with queries and relevance judgments.
- Implement true Recall@K and MRR@10.
- Execute the five ablation configurations against the same corpus.
- Replace the synthetic report with reproducible, clearly labeled results (keep synthetic data only as
  a fixture/demo artifact, not as reported results).

### Phase 8 — Hardening, tests, and API handoff

- Complete unit/integration/negative tests listed in Section 8.
- Add ingestion service functions and hand the route contract to Member 3 (or mount a Member 2 router with
  coordination).
- Document setup, model cache, DB bootstrap, and operations in `retrieval/README.md`.
- Performance benchmark and tuning of HNSW/BM25/reranker.

---

## Appendix A — Exact Files to Implement per Phase

> Existing files should be **refactored in place or moved with compatibility shims**, not duplicated.
> Suggested layout uses subpackages; keep `retrieval/docling_parser.py`, `retrieval/pg_hybrid_search.py`,
> and `retrieval/lightrag_indexer.py` as thin re-export shims if external code imports them.

### Phase 0 (documentation only)

- `retrieval/IMPLEMENTATION_AUDIT.md` (this file)
- `retrieval/INTEGRATION_CONTRACT.md` *(new; optional, recommended for team coordination)*

### Phase 1 — Foundation

- `retrieval/config.py` *(new; retrieval-scoped settings)*
- `retrieval/db/__init__.py` *(new)*
- `retrieval/db/connection.py` *(new; pool, health check, extension checks)*
- `retrieval/db/migrations/0001_init.sql` *(new)*
- `retrieval/db/migrations/0002_indexes.sql` *(new; HNSW + BM25)*
- `retrieval/db/repository.py` *(new; workspace/document/chunk access)*
- `retrieval/db/models.py` *(new; only if SQLAlchemy is chosen)*
- `docker-compose.yml` *(root; shared/infra — coordinate)*
- `pyproject.toml` *(add optional dependency groups — shared change, coordinate)*
- `.env.example` *(add Member 2 variables — shared change, coordinate)*
- `tests/test_retrieval_db.py` *(new)*

### Phase 2 — Ingestion

- `retrieval/ingestion/__init__.py` *(new)*
- `retrieval/ingestion/docling_parser.py` *(new; real Docling)*
- `retrieval/ingestion/normalizer.py` *(new)*
- `retrieval/ingestion/chunker.py` *(new; structure-aware)*
- `retrieval/ingestion/metadata.py` *(new)*
- `retrieval/ingestion/storage.py` *(new; file persistence, hashing, dedupe)*
- `retrieval/ingestion/pipeline.py` *(new; orchestration + job tracking)*
- `retrieval/schemas.py` *(new; `IngestedDocument`, extended chunk metadata; keep `core.schemas` unchanged pending agreement)*
- `retrieval/docling_parser.py` *(refactor to compatibility shim)*
- `tests/fixtures/` *(new; sample PDF/DOCX/MD)*
- `tests/test_retrieval_ingestion.py` *(new)*
- `tests/test_retrieval_workspace_isolation.py` *(new)*

### Phase 3 — Embeddings + pgvector

- `retrieval/embeddings/__init__.py` *(new)*
- `retrieval/embeddings/bge_m3.py` *(new; lazy model loader, batching)*
- `retrieval/vector/__init__.py` *(new)*
- `retrieval/vector/pgvector_store.py` *(new; upsert/search)*
- `retrieval/vector/retriever.py` *(new; implements `BaseVectorRetriever`)*
- `tests/test_retrieval_embeddings.py` *(new)*
- `tests/test_retrieval_pgvector.py` *(new)*

### Phase 4 — BM25 + Hybrid

- `retrieval/bm25/__init__.py` *(new)*
- `retrieval/bm25/pg_search.py` *(new; implements `BaseBM25Retriever`)*
- `retrieval/hybrid/__init__.py` *(new)*
- `retrieval/hybrid/rrf.py` *(new; move existing correct RRF logic here)*
- `retrieval/hybrid/engine.py` *(new; `BaseHybridReranker` orchestration without reranker yet)*
- `retrieval/service.py` *(new; high-level `RetrievalService.hybrid_search`)*
- `retrieval/pg_hybrid_search.py` *(refactor to shim or remove after migration)*
- `retrieval/__init__.py` *(update exports; avoid eager heavy imports)*
- `tests/test_retrieval_bm25.py` *(new)*
- `tests/test_retrieval_hybrid.py` *(new)*

### Phase 5 — Reranking

- `retrieval/reranking/__init__.py` *(new)*
- `retrieval/reranking/bge_reranker.py` *(new; lazy loader, batched raw logits)*
- `retrieval/reranking/service.py` *(new; score + map to `ChunkEvidence`)*
- `retrieval/hybrid/engine.py` *(extend to apply reranker)*
- `tests/test_retrieval_reranker.py` *(new)*
- `tests/test_retrieval_confidence_integration.py` *(new; feeds Member 1 gate)*

### Phase 6 — GraphRAG / LightRAG

- `retrieval/graph/__init__.py` *(new)*
- `retrieval/graph/config.py` *(new)*
- `retrieval/graph/lightrag_engine.py` *(new; real LightRAG adapter)*
- `retrieval/graph/entity_extractor.py` *(new)*
- `retrieval/graph/workspace_store.py` *(new; namespace/isolation)*
- `retrieval/graph/traversal.py` *(new; compatibility wrapper)*
- `retrieval/lightrag_indexer.py` *(refactor to compatibility shim implementing `BaseGraphRAGRetriever`)*
- `retrieval/interfaces.py` *(add optional `workspace_id` to graph interface — coordinate)*
- `tests/test_retrieval_graph.py` *(new)*
- `tests/test_retrieval_graph_workspace.py` *(new)*

### Phase 7 — Evaluation

- `retrieval/evaluation/__init__.py` *(new)*
- `retrieval/evaluation/dataset.py` *(new; corpus + qrels loader)*
- `retrieval/evaluation/recall.py` *(new; Recall@K, MRR@K)*
- `retrieval/evaluation/ablation.py` *(new; configs executed against retrievers)*
- `retrieval/evaluation/run_ablation.py` *(new; CLI entry point)*
- `retrieval/evaluation/qrels.sample.json` *(new; clearly labeled sample)*
- `retrieval/recall_ablation.py` *(refactor to delegate to `retrieval.evaluation`)*
- `retrieval/recall_ablation_results.json` *(regenerate only from real runs; label provenance)*
- `tests/test_retrieval_evaluation.py` *(new)*

### Phase 8 — Hardening and handoff

- `retrieval/api.py` *(new; ingestion/retrieval router — mount by Member 3 only after agreement)*
- `tests/test_retrieval_api.py` *(new)*
- `tests/test_retrieval_end_to_end.py` *(new)*
- `retrieval/README.md` *(update with real setup, model cache, DB bootstrap, API contract)*
- `retrieval/IMPLEMENTATION_AUDIT.md` *(update status after each phase)*

---

## Appendix B — Integration Decisions Requiring Team Coordination

| # | Decision | Affects | Recommendation |
| :-: | :--- | :--- | :--- |
| 1 | Add `workspace_id`/provenance to `ChunkEvidence` or keep it in `metadata` | Members 1, 3 | Prefer additive optional fields or namespaced `metadata`; Member 1 must approve score semantics |
| 2 | Ingestion route ownership (`retrieval/api.py` vs `api/`) | Members 2, 3, 4 | Member 2 implements service + router; Member 3 mounts it and owns auth/middleware |
| 3 | Sync vs async retrieval facade | Members 2, 3 | Recommend async facade for FastAPI path; allow sync worker for CLI/eval |
| 4 | `LightRAGGraphEngine` signature/constructor changes | Member 3 | Preserve `traverse_entities(query, entity_seeds, max_hops)` and return keys; add optional `workspace_id` |
| 5 | ParadeDB image and extension versions | All backend | Pin a ParadeDB distribution supporting PG17 + pgvector + pg_search |
| 6 | Embedding/reranker model versions and dimensions | Members 1, 2, 4 | Pin exact model revisions; store model version with embeddings |
| 7 | Document storage location and retention | Members 2, 3, 4 | Define local/object storage, size limits, deletion, and backup policy |
| 8 | Authentication and workspace authorization | All | Current `workspace_id` is client-supplied and unauthenticated; schedule before production |
| 9 | Dependency additions to `pyproject.toml` | All | Coordinate optional extras; avoid forcing heavy ML deps on Members 1/3 |
| 10 | Member 3's extra commit `e094f2c` | Members 2, 3 | Review before integration; do not merge/rebase without agreement |
| 11 | Real evaluation corpus ownership | Members 2, 4 | Member 2 defines qrels; Member 4 may reuse for benchmark consistency |
| 12 | Error/empty-result contract | Members 1, 3 | Define behavior for no documents, DB unavailable, model unavailable, and workspace not found |

---

## Appendix C — Audit Limitations

- The blueprint PDFs (`CalibRAG_Product_and_Architecture_Blueprint.pdf`,
  `ACR_RAG_Product_and_Architecture_Blueprint.pdf`) could not be machine-read by the available tooling in
  this environment. Findings are based on source code, `README.md`, module READMEs, `.env.example`, and
  configuration. Any blueprint-specific schema or math not reflected in code should be re-checked by the
  team.
- No tests were executed because project dependencies are not installed. Static inspection was used
  throughout.
- `myenv/` is an untracked, essentially empty virtual environment; it is not part of the repository and
  was left untouched.
- No files outside `retrieval/IMPLEMENTATION_AUDIT.md` were created or modified in this phase.

---

## Final Summary

### What already exists

- Package structure and `__init__` exports for `retrieval/`.
- Abstract interfaces for vector, BM25, hybrid/rerank, and graph retrieval.
- `DocumentChunk` and a naive text chunker with a simulated-document fallback.
- A correct RRF implementation operating on in-memory `ChunkEvidence`.
- In-memory heuristic stand-ins for vector search, BM25, and cross-encoder reranking.
- A hard-coded demo knowledge graph with 1–2 hop traversal.
- Synthetic Recall@K/MRR ablation output.
- Mock-level Member 2 tests; shared `ChunkEvidence` schema used by Members 1 and 3.
- Configuration placeholders for `DATABASE_URL` and related environment variables.

### What is missing

- Real IBM Docling parsing for PDF/DOCX/Markdown, normalization, structure-aware chunking, and provenance.
- Any PostgreSQL 17 connection, schema, migration, pgvector/HNSW index, or ParadeDB `pg_search` BM25 index.
- BGE-M3 embedding generation and `bge-reranker-v2-m3` cross-encoder inference with true logits.
- Persistence, deduplication, ingestion jobs, and workspace-level isolation.
- Real LightRAG integration, entity/relation extraction, persistent workspace-scoped graphs.
- A high-level retrieval facade and an ingestion API contract.
- Real Recall@K evaluation with labeled data and executed ablations.
- Production-grade tests, dependency pinning, Docker setup, and environment provisioning.

### Exact files to implement in subsequent phases

See **Appendix A**. In short: `retrieval/db/*`, `retrieval/ingestion/*`, `retrieval/embeddings/*`,
`retrieval/vector/*`, `retrieval/bm25/*`, `retrieval/hybrid/*`, `retrieval/reranking/*`,
`retrieval/graph/*`, `retrieval/evaluation/*`, `retrieval/service.py`, `retrieval/api.py`, plus
corresponding `tests/*` and coordinated edits to `pyproject.toml`, `.env.example`, `docker-compose.yml`,
and `retrieval/__init__.py`.

### Decisions requiring team coordination

See **Appendix B**. Highest priority: `ChunkEvidence` schema/score semantics, ingestion route ownership,
sync vs async retrieval, LightRAG interface compatibility with Member 3, ParadeDB image choice, and
dependency/model pinning.
