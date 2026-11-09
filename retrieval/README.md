# Retrieval Module (Member 2: Data, Hybrid RAG & GraphRAG Lead)

## 1. Overview
The `retrieval/` module provides CalibRAG's unified data ingestion and retrieval infrastructure, replacing brittle multi-database stacks (Elasticsearch + Neo4j) with a clean, high-performance PostgreSQL 17 + LightRAG architecture.

## 2. Core Components Owned:
1. **BYOK Ingestion Pipeline (`retrieval/ingestion/`)**:
   - Parses multi-format documents (PDFs, DOCX, Markdown) via IBM Docling.
   - Preserves complex tables, section hierarchy, and page-level provenance.
   - Persists documents, chunks, ingestion jobs, and content-addressed source files in PostgreSQL 17.
2. **Unified Hybrid Retrieval (`retrieval/service.py`)**:
   - **pgvector (HNSW)**: Fast dense vector semantic similarity with `BAAI/bge-m3`.
   - **ParadeDB pg_search**: Rust-based BM25 lexical keyword matching inside PostgreSQL.
   - **Reciprocal Rank Fusion (RRF)**: Merges dense + sparse rankings using $k=60$.
   - **BGE Cross-Encoder Reranking (`bge-reranker-v2-m3`)**: Computes calibrated logit margins ($\Delta L_{rerank}$) for Member 1's $C_{evidence}$ formula.
   - `retrieval/pg_hybrid_search.py` is a legacy in-memory simulation retained for compatibility; production code must use `RetrievalService`.
3. **LightRAG Graph Engine (`retrieval/lightrag_indexer.py`)**:
   - Uses the real `lightrag-hku` package to extract entities and relationships from ingested chunks.
   - Stores graph state under `RETRIEVAL_GRAPH_STORAGE_ROOT/<workspace_id>/` by default; storage class names remain configurable for PostgreSQL backends.
   - Binds one engine instance to one workspace and validates chunk/document workspace IDs before indexing.
   - Traverses real entity links with a bounded BFS and returns `GraphNode`, `GraphEdge`, and `context_summary` data for Member 3's LangGraph supervisor.
   - Creates the LightRAG client lazily. If no graph exists for a workspace, traversal returns an empty result instead of seeded/demo entities.
4. **Retrieval Evaluation Harness (`retrieval/evaluation/`)**:
   - Computes Recall@K, Precision@K, Hit@K, nDCG@K, and MRR@K from actual returned chunk IDs.
   - Loads labeled JSON or JSONL cases and writes sanitized JSON reports.
   - Includes a latency/peak-memory benchmark helper.
   - `retrieval/recall_ablation.py` and `recall_ablation_results.json` are legacy synthetic artifacts and must not be cited as empirical results.

## 3. Unified Hybrid + Graph Retrieval

Use `retrieve` for Member 1-compatible evidence, or `retrieve_with_graph` when graph context is also needed:

```python
import asyncio

from retrieval.config import get_settings
from retrieval.db import Database
from retrieval.lightrag_indexer import LightRAGGraphEngine
from retrieval.service import RetrievalService

async def main() -> None:
    settings = get_settings()
    async with Database(settings) as database:
        service = RetrievalService(database, settings=settings)
        engine = LightRAGGraphEngine(settings=settings, workspace_id="tenant_a")
        try:
            result = await service.retrieve_with_graph(
                "Who supplies the BMS controller?",
                top_k=5,
                workspace_id="tenant_a",
                entity_seeds=["VoltCore", "BMS-Controller-v4"],
                graph_engine=engine,
                max_hops=2,
            )
            evidence = result.evidence          # List[ChunkEvidence]
            graph_context = result.graph_context  # str
            graph_error = result.graph_error      # None or sanitized message
        finally:
            engine.close()
            service.close()

asyncio.run(main())
```

Graph traversal is bounded by `RETRIEVAL_GRAPH_TRAVERSAL_TIMEOUT` (default `120` seconds). Graph failures and timeouts are reported through `graph_error` and never discard valid hybrid evidence. `retrieve_with_graph` keeps the existing `retrieve` and `retrieve_chunks` contracts unchanged.

## 4. Retrieval Evaluation Harness

The harness requires labeled relevance IDs and evaluates the actual ranked output:

```python
import asyncio
import json

from retrieval.evaluation import RetrievalEvalCase, evaluate_service

async def main(service) -> None:
    cases = [
        RetrievalEvalCase(
            query="What is the maximum defect tolerance?",
            workspace_id="tenant_a",
            relevant_chunk_ids=("doc-id:c00000",),
        )
    ]
    report = await evaluate_service(service, cases, top_k=5, k_values=(1, 3, 5))
    print(json.dumps(report.to_dict(), indent=2))

asyncio.run(main(service))
```

JSON array and JSONL case files can be loaded with `load_eval_cases(path)`. A missing label or retriever failure is recorded as a failure; the harness never substitutes synthetic metrics.

## 5. Reliability and Performance Checks

- Graph traversal runs in a worker thread with a configurable timeout.
- Graph workspace mismatches are rejected before context is returned.
- Error messages are sanitized and truncated before being exposed.
- `benchmark_retriever` reports mean, p50, p95, min, max latency and peak traced Python memory.
- Peak traced memory excludes model weights loaded before tracing; report model-load time separately.

## 6. LightRAG Graph Indexing and Traversal
Install the retrieval runtime, including `lightrag-hku==1.5.7`:

```bash
pip install -r retrieval/requirements-retrieval.txt
```

Graph indexing requires an OpenAI-compatible LLM. Traversal of an existing graph does not call the LLM:

```bash
RETRIEVAL_GRAPH_LLM_MODEL=gpt-4o-mini
RETRIEVAL_GRAPH_LLM_BASE_URL=https://api.openai.com/v1
RETRIEVAL_GRAPH_LLM_API_KEY=...
```

`RETRIEVAL_GRAPH_LLM_API_KEY` takes precedence; `GROQ_API_KEY` and `OPENAI_API_KEY` are supported fallbacks. For Groq:

```bash
RETRIEVAL_GRAPH_LLM_MODEL=qwen/qwen3.8-27b
RETRIEVAL_GRAPH_LLM_BASE_URL=https://api.groq.com/openai/v1
GROQ_API_KEY=...
```

The default BGE-M3 embedder is preloaded before LightRAG initializes on first use, avoiding the embedding worker cold-start timeout. Expect roughly 45 seconds for the initial model load.

Index existing PostgreSQL chunks and traverse the resulting graph:

```python
from retrieval.config import get_settings
from retrieval.db import Database, IngestionRepository
from retrieval.lightrag_indexer import LightRAGGraphEngine

settings = get_settings()
async with Database(settings) as database:
    engine = LightRAGGraphEngine(settings=settings, workspace_id="tenant_a")
    try:
        await engine.aindex_workspace(IngestionRepository(database))
        result = engine.traverse_entities(
            query="Who is the lead architect?",
            entity_seeds=["VoltCore", "BMS-Controller-v4"],
            max_hops=2,
        )
    finally:
        await engine.aclose()
```

`traverse_entities` is synchronous for the existing `BaseGraphRAGRetriever` contract and runs the async LightRAG client on a private event-loop thread, so it is safe to call from inside an async supervisor. Use `aindex_text`, `aindex_chunks`, `aindex_document`, or `aindex_workspace` for indexing, and `aclose()`/`close()` to finalize LightRAG storages.

### Multi-chunk indexing and provenance

`aindex_chunks` submits every non-empty chunk as its own stable LightRAG document keyed by the original `chunk_id`. It does not rely on LightRAG's filename-based deduplication:

- Each chunk receives a unique source path of the form `<filename-or-doc-title>::chunk::<safe-chunk-id>`.
- Chunks are preflighted against LightRAG `doc_status`. Already `PROCESSED` chunks are skipped, so retrying an indexing call does not duplicate graph data.
- `FAILED` chunks are deleted with `adelete_by_doc_id` and resubmitted. If deletion is unavailable or fails, indexing raises `GraphConfigurationError` instead of silently duplicating data.
- Pending documents are resumed with the LightRAG processing pipeline.
- The result reports `status`, `indexed_documents`, `submitted_chunks`, `processed_chunks`, `skipped_chunks`, `failed_chunks`, `pending_chunks`, `retried_failed_chunks`, and per-chunk `chunk_mappings` with `source_path` and `lightrag_document_id`.

LightRAG still applies its internal text chunker to each submitted chunk, so the outer original chunk ID remains the provenance identity while LightRAG may create internal subchunks. Existing GraphML files created before this change are not migrated; their already-`PROCESSED` chunk IDs are still skipped on retry.

### Graph query modes

The engine also exposes the LightRAG 1.5.7 query APIs through `QueryParam`:

```python
from retrieval.lightrag_indexer import LightRAGGraphEngine

engine = LightRAGGraphEngine(workspace_id="tenant_a")
try:
    data = engine.query_data(
        "Who supplies the BMS controller?",
        mode="hybrid",
        hl_keywords=["supply", "modules"],
        ll_keywords=["VoltCore India", "BMS-Controller-v4"],
    )
    answer = engine.query(
        "Who supplies the BMS controller?",
        mode="hybrid",
        hl_keywords=["supply", "modules"],
        ll_keywords=["VoltCore India", "BMS-Controller-v4"],
    )
finally:
    engine.close()
```

- Supported modes: `local`, `global`, `hybrid`, `naive`, `mix`, and `bypass`.
- `query_data` / `aquery_data` return LightRAG structured entities, relationships, chunks, references, and metadata without LLM generation.
- `query` / `aquery` return generated answer text and require an LLM provider.
- `query_llm` / `aquery_llm` return both structured retrieval data and the LLM response.
- When no LLM credentials are available, pass explicit `hl_keywords` and `ll_keywords` to `query_data` to skip LightRAG keyword extraction. Without explicit keywords, graph query modes need an LLM for keyword extraction.

Run the opt-in real LightRAG structured-query test after installing the graph runtime:

```bash
LIGHTRAG_REAL_TESTS=1 pytest tests/test_retrieval_lightrag_real.py -q
```

### Real LLM extraction verification

Real entity/relationship extraction uses the normal LLM-backed `ainsert` path. Configure an OpenAI-compatible provider as shown above, then index a document with `aindex_text` or `aindex_workspace` and inspect the persisted graph:

```python
import asyncio

from retrieval.config import get_settings
from retrieval.lightrag_indexer import LightRAGGraphEngine

DOCUMENT = (
    "Company Alpha develops Product X. Product X uses Technology Y. "
    "Technology Y is maintained by Team Z. Company Alpha was founded in 2018."
)

async def main() -> None:
    settings = get_settings()
    engine = LightRAGGraphEngine(settings=settings, workspace_id="phase42_alpha")
    try:
        await engine.aindex_text(
            DOCUMENT,
            doc_id="phase42-alpha-doc",
            file_path="phase42-alpha.txt",
        )
        graph = engine.traverse_entities(
            query="Which product does Company Alpha develop?",
            entity_seeds=["Company Alpha", "Product X", "Technology Y", "Team Z"],
            max_hops=2,
        )
        print(graph["nodes"])
        print(graph["edges"])
        print(graph["context_summary"])
    finally:
        await engine.aclose()

asyncio.run(main())
```

Persistence is file-backed by default at `RETRIEVAL_GRAPH_STORAGE_ROOT/<workspace_id>/graph_chunk_entity_relation.graphml`. After `aclose()`/`close()`, create a new engine for the same workspace and call `traverse_entities` again to verify reload.

The current opt-in `LIGHTRAG_REAL_TESTS` test verifies real LightRAG storage and query APIs with custom KG data and deterministic fake embeddings. It does not prove LLM extraction. Real extraction verification requires provider credentials; see `retrieval/reports/PHASE_4_2_REAL_EXTRACTION_REPORT.md`.

## 7. Test Commands

```bash
# Full retrieval suite (live DB tests are skipped unless the DSN is set)
pytest -q

# Live PostgreSQL integration tests
RETRIEVAL_TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/calibrag pytest -q

# Phase 5 live end-to-end ingestion + hybrid + graph integration
RETRIEVAL_TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/calibrag \
  pytest tests/test_retrieval_phase5_e2e.py -q

# Opt-in real confidence models
RETRIEVAL_RUN_REAL_MODELS=1 pytest tests/test_retrieval_confidence_integration.py -q

# Opt-in real LightRAG query APIs
LIGHTRAG_REAL_TESTS=1 pytest tests/test_retrieval_lightrag_real.py -q

# Opt-in real Phase 5.2 multi-chunk GraphRAG verification
RETRIEVAL_PHASE5_2_REAL_TESTS=1 \
RETRIEVAL_TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/calibrag \
  pytest tests/test_retrieval_phase5_2_real_multichunk.py -q
```

The Phase 5.2 test uses four real chunks per document, checks later-chunk facts and provenance, verifies idempotent re-indexing and GraphML reload, and isolates two workspaces. Set `RETRIEVAL_PHASE5_2_GRAPH_PAUSE_SECONDS` to tune the pause between workspaces when the LLM provider is rate-limited.

The live database tests expect PostgreSQL 17 with `pgvector` and ParadeDB `pg_search`. The Docker service used during Phase 5 was `calibrag-retrieval-postgres-1` (`paradedb/paradedb:0.26.5-pg17`) on port `5432`.

## 8. Environment Variables

Core retrieval and storage:

- `RETRIEVAL_DATABASE_URL` or `DATABASE_URL`
- `RETRIEVAL_STORAGE_ROOT`
- `RETRIEVAL_MAX_UPLOAD_BYTES`
- `RETRIEVAL_CHUNK_TARGET_CHARS`, `RETRIEVAL_CHUNK_MAX_CHARS`, `RETRIEVAL_CHUNK_OVERLAP_CHARS`
- `RETRIEVAL_EMBEDDING_MODEL`, `RETRIEVAL_EMBEDDING_DIMENSION`, `RETRIEVAL_EMBEDDING_DEVICE`
- `RETRIEVAL_TOP_K`, `RETRIEVAL_VECTOR_CANDIDATE_DEPTH`, `RETRIEVAL_BM25_CANDIDATE_DEPTH`
- `RETRIEVAL_RRF_K`, `RETRIEVAL_HYBRID_CANDIDATE_DEPTH`, `RETRIEVAL_RERANK_CANDIDATE_DEPTH`
- `RETRIEVAL_RERANKER_MODEL`, `RETRIEVAL_RERANKER_DEVICE`

Graph and evaluation:

- `RETRIEVAL_GRAPH_STORAGE_ROOT`
- `RETRIEVAL_GRAPH_KV_STORAGE`, `RETRIEVAL_GRAPH_VECTOR_STORAGE`, `RETRIEVAL_GRAPH_GRAPH_STORAGE`, `RETRIEVAL_GRAPH_DOC_STATUS_STORAGE`
- `RETRIEVAL_GRAPH_LLM_MODEL`, `RETRIEVAL_GRAPH_LLM_BASE_URL`, `RETRIEVAL_GRAPH_LLM_API_KEY`
- `RETRIEVAL_GRAPH_LLM_TIMEOUT`, `RETRIEVAL_GRAPH_LLM_MAX_ASYNC`
- `RETRIEVAL_GRAPH_MAX_NODES`, `RETRIEVAL_GRAPH_TRAVERSAL_MAX_NODES`, `RETRIEVAL_GRAPH_TRAVERSAL_TIMEOUT`
- `RETRIEVAL_TEST_DATABASE_URL`, `RETRIEVAL_RUN_REAL_MODELS`, `LIGHTRAG_REAL_TESTS`
- `RETRIEVAL_PHASE5_2_REAL_TESTS`, `RETRIEVAL_PHASE5_2_GRAPH_PAUSE_SECONDS`

API-key fallback order is `RETRIEVAL_GRAPH_LLM_API_KEY`, then `GROQ_API_KEY`, then `OPENAI_API_KEY`. Never print or commit API keys.

## 9. Known Limitations

- PostgreSQL-backed LightRAG storage classes are configurable but were not validated end-to-end. Phase 4.3 and Phase 5 verification used file-backed GraphML under `RETRIEVAL_GRAPH_STORAGE_ROOT`.
- Real graph extraction depends on an external OpenAI-compatible LLM. Groq may return `429` ITPM or `503` over-capacity responses; retries are required. The Phase 5.2 real multi-chunk test retries failed chunks after a configurable pause.
- LightRAG's internal text chunker still runs inside each submitted chunk. The adapter preserves the original chunk ID as the outer document/provenance identity, but internal LightRAG subchunks use LightRAG's own chunking settings.
- Existing GraphML files created before Phase 5.2 are not migrated to the new per-chunk source paths. Already-`PROCESSED` chunk IDs are skipped, so re-indexing does not duplicate them.
- The Phase 5 real-model evaluation used a small four-document corpus and should be treated as an integration smoke test, not a retrieval-quality benchmark.
- `benchmark_retriever` traces Python allocations during retrieval calls. Model weights loaded before tracing are excluded from peak memory.
- `retrieval/recall_ablation.py` and `retrieval/recall_ablation_results.json` contain legacy synthetic values and are not used for Phase 5 claims.
