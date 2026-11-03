# Retrieval Module (Member 2: Data, Hybrid RAG & GraphRAG Lead)

## 1. Overview
The `retrieval/` module provides CalibRAG's unified data ingestion and retrieval infrastructure, replacing brittle multi-database stacks (Elasticsearch + Neo4j) with a clean, high-performance PostgreSQL 17 + LightRAG architecture.

## 2. Core Components Owned:
1. **BYOK Ingestion Pipeline (`docling_parser.py`)**:
   - Parses multi-format documents (PDFs, DOCX, Markdown) via IBM Docling.
   - Preserves complex tables, section hierarchy, and page-level provenance.
2. **Unified PostgreSQL 17 Vector + BM25 Database (`pg_hybrid_search.py`)**:
   - **pgvector (HNSW)**: Fast dense vector semantic similarity with `BAAI/bge-m3`.
   - **ParadeDB pg_search**: Rust-based BM25 lexical keyword matching inside PostgreSQL.
   - **Reciprocal Rank Fusion (RRF)**: Merges dense + sparse rankings using $k=60$.
   - **BGE Cross-Encoder Reranking (`bge-reranker-v2-m3`)**: Computes calibrated logit margins ($\Delta L_{rerank}$) for Member 1's $C_{evidence}$ formula.
3. **Dual-Level GraphRAG Engine (`lightrag_indexer.py`)**:
   - Extracts entity relationships at 10x lower token cost than Microsoft GraphRAG.
   - Traverses 2-hop entity links (e.g., Projects $\to$ Modules $\to$ Suppliers $\to$ Penalty Clauses) for Member 3's LangGraph supervisor.
4. **Research Paper Section 4 Contributions (`recall_ablation.py`)**:
   - Authors Section 4: *Hybrid + Graph Retrieval Architecture*.
   - Generates empirical Recall@K ($K \in \{1, 3, 5, 10\}$) and MRR@10 ablation tables.

## 3. Running Retrieval Recall Ablation
```bash
python3 -m retrieval.recall_ablation
```
