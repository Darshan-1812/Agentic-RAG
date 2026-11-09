# Phase 4 LightRAG Report - Real Graph Indexing and Traversal

**Branch:** `member-2-retrieval`  
**Date:** 2026-11-07  
**Owner:** Member 2 - Data, Hybrid RAG & GraphRAG Lead  
**Status:** Phase 4 implementation complete. Phase 5 was not started.

---

## A. Executive Summary

Phase 4 replaces the previous in-memory/demo graph engine with a real `lightrag-hku` adapter:

```text
ingested PostgreSQL chunks
  -> LightRAG entity/relationship extraction
  -> workspace-scoped graph storage
  -> real graph label lookup and bounded BFS
  -> GraphNode / GraphEdge / context_summary
  -> Member 3 LangGraph supervisor
```

The adapter is workspace-bound, creates LightRAG lazily, preserves the synchronous `traverse_entities` contract, and returns an empty result when a workspace has no graph instead of seeded demo entities.

Final verification results:

| Verification | Result |
| :--- | :--- |
| Full test suite | `72 passed`, `3 skipped` |
| Phase 4 targeted tests | `13 passed` |
| Real LightRAG client construction | Passed with `lightrag-hku==1.5.7` |
| Real graph file traversal smoke test | Passed: 2 nodes, 1 edge, valid context summary |
| Static compilation | Passed |

---

## B. Scope and Constraints

Implemented scope:

- Real LightRAG client integration in `retrieval/lightrag_indexer.py`.
- Workspace-scoped graph storage and validation.
- Real graph label lookup, knowledge-graph retrieval, bounded BFS, and context formatting.
- Graph indexing from chunks, text, documents, and existing PostgreSQL ingestion data.
- Lazy client creation and explicit storage finalization.
- Configuration for graph storage, LLM provider, node limits, and traversal limits.
- Unit tests with a fake LightRAG client and a real LightRAG smoke test.

Constraints preserved:

- Work stayed on `member-2-retrieval`.
- No merge, rebase, commit, or push was performed.
- No changes to `core/schemas.py`, `verifier/confidence_gate.py`, Member 1/3/4 files, or shared interfaces.
- `BaseGraphRAGRetriever.traverse_entities` remains unchanged.
- No hardcoded enterprise graph data remains in the graph engine.
- Heavy LightRAG imports are lazy; importing `retrieval` does not require LightRAG.
- Traversal does not require an LLM API key. Indexing does.

---

## C. Architecture

### Workspace binding

Each `LightRAGGraphEngine` instance is bound to one validated `workspace_id`. The existing shared interface does not pass a workspace ID into `traverse_entities`, so binding at engine construction avoids a cross-team interface change. Chunk and document workspace IDs are validated before indexing.

Default graph storage:

```text
<RETRIEVAL_GRAPH_STORAGE_ROOT>/<workspace_id>/graph_chunk_entity_relation.graphml
```

Storage class names are configurable for optional PostgreSQL backends. The default remains file-backed LightRAG storage.

### Lazy client and event loop

LightRAG is imported and constructed only when graph data exists or indexing is requested. A private daemon event-loop thread runs the async LightRAG client. The synchronous `traverse_entities` method submits work to that loop and waits, so it is safe to call from inside Member 3's already-running asyncio loop. `aclose()` and `close()` finalize LightRAG storages and stop the thread.

### Traversal

Traversal performs:

1. `get_graph_labels()` to find real labels.
2. Case-insensitive exact/substring seed matching.
3. `get_knowledge_graph(label, max_depth=max_hops, max_nodes=...)`.
4. Merging subgraphs from multiple seeds.
5. BFS distance calculation from matched seeds.
6. Conversion to `GraphNode` and `GraphEdge`.
7. Context lines in the existing `Graph Link: source -> [relation] -> target` format.

LightRAG's NetworkX backend does not expose a semantic relation type in its graph response, so edges use `RELATED_TO` unless a `relation_type` property is present. The source description and file path remain in edge properties.

### Empty graph behavior

If the default NetworkX graph file does not exist, traversal returns an empty result without importing LightRAG. This keeps Member 3 tests and local development working without a graph or LLM credentials. The previous hardcoded `R. Kulkarni`/VoltCore graph is removed.

### Indexing

The adapter supports:

- `aindex_chunks` / `index_chunks`
- `aindex_text` / `index_text`
- `aindex_document` / `index_document`
- `aindex_workspace` / `index_workspace`

`aindex_workspace` loads existing documents and chunks through `IngestionRepository`, so Phase 2 ingestion remains the source of truth. Indexing requires an OpenAI-compatible LLM configuration and raises `GraphConfigurationError` if none is available.

---

## D. Files Created or Modified

- `retrieval/lightrag_indexer.py` - real workspace-scoped LightRAG adapter, traversal, indexing, loop runner, lifecycle.
- `retrieval/config.py` - graph storage, LLM, and traversal settings plus validation and environment parsing.
- `retrieval/requirements-retrieval.txt` - added `lightrag-hku==1.5.7`.
- `retrieval/.env.ingestion.example` - documented graph environment variables.
- `retrieval/README.md` - documented graph indexing, traversal, and runtime usage.
- `tests/test_retrieval_lightrag.py` - new fake-client tests for traversal, indexing, isolation, configuration, and loop safety.
- `tests/test_retrieval_member2.py` - replaced the old hardcoded graph test with an empty-index regression test.
- `retrieval/reports/PHASE_4_LIGHTRAG_REPORT.md` - this report.

---

## E. Configuration

New retrieval settings:

| Setting | Default | Purpose |
| :--- | :--- | :--- |
| `RETRIEVAL_GRAPH_STORAGE_ROOT` | `retrieval/storage/graph` | LightRAG working directory |
| `RETRIEVAL_GRAPH_KV_STORAGE` | `JsonKVStorage` | LightRAG KV backend |
| `RETRIEVAL_GRAPH_VECTOR_STORAGE` | `NanoVectorDBStorage` | LightRAG vector backend |
| `RETRIEVAL_GRAPH_GRAPH_STORAGE` | `NetworkXStorage` | LightRAG graph backend |
| `RETRIEVAL_GRAPH_DOC_STATUS_STORAGE` | `JsonDocStatusStorage` | LightRAG document status backend |
| `RETRIEVAL_GRAPH_LLM_MODEL` | `gpt-4o-mini` | Extraction/query model |
| `RETRIEVAL_GRAPH_LLM_BASE_URL` | unset / `OPENAI_BASE_URL` | OpenAI-compatible endpoint |
| `RETRIEVAL_GRAPH_LLM_API_KEY` | unset / `OPENAI_API_KEY` | LLM API key |
| `RETRIEVAL_GRAPH_LLM_TIMEOUT` | `60` | LLM timeout in seconds |
| `RETRIEVAL_GRAPH_LLM_MAX_ASYNC` | `4` | Maximum concurrent LLM calls |
| `RETRIEVAL_GRAPH_MAX_NODES` | `1000` | LightRAG maximum graph nodes |
| `RETRIEVAL_GRAPH_TRAVERSAL_MAX_NODES` | `200` | Per-traversal node limit |

The API key is excluded from `repr`.

---

## F. Verification

Commands run with the Phase 2 temporary virtual environment:

```bash
python -m py_compile retrieval/config.py retrieval/lightrag_indexer.py tests/test_retrieval_lightrag.py tests/test_retrieval_member2.py
python -m pytest tests/test_retrieval_lightrag.py tests/test_retrieval_member2.py tests/test_agents_member3.py -q
python -m pytest -q
```

Results:

- Targeted Phase 4/Member 3 tests: `13 passed`.
- Full suite: `72 passed, 3 skipped`.
- Real LightRAG smoke test: constructed `LightRAG` against a temporary NetworkX graph file, initialized storages, returned 2 nodes and 1 edge, produced `Graph Link:` context, and finalized storages successfully.

---

## G. Limitations and Follow-Up

- Real entity extraction and graph querying were not executed against an external LLM because no verified production credentials were available. The adapter fails clearly when indexing without LLM configuration.
- PostgreSQL LightRAG storage backends are configurable but were not verified against the live CalibRAG database. File-backed NetworkX storage is the tested default.
- Graph indexing currently indexes chunk content as separate LightRAG documents keyed by `chunk_id`. This preserves chunk-level provenance and deduplication, but document-level graph tuning may be added later.
- The LightRAG NetworkX response does not preserve semantic edge direction or relation labels; `RELATED_TO` is used as a truthful fallback and descriptions remain available in edge properties.

Phase 5, orchestration changes, commits, and pushes were not started.
