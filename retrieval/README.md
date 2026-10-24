# Retrieval Module (Member 2: Data, Hybrid RAG & GraphRAG Lead)

## Core Responsibilities:
- **BYOK Ingestion Pipeline**: Document parsing via IBM Docling into markdown, tables, and page-level provenance.
- **Unified Vector + BM25 Database**: PostgreSQL 17 + `pgvector` (HNSW) + ParadeDB `pg_search` (Rust-based BM25).
- **Embeddings & Cross-Encoder**: BAAI `bge-m3` dense/sparse embeddings + `bge-reranker-v2-m3` for cross-encoder logits used in $C_{evidence}$.
- **LightRAG Graph Indexer**: Dual-level GraphRAG extracting entity-relationship graphs stored over Postgres/KV.

## Interface Contracts:
Defined in `retrieval/interfaces.py`:
- `BaseVectorRetriever`
- `BaseBM25Retriever`
- `BaseHybridReranker`
- `BaseGraphRAGRetriever`
