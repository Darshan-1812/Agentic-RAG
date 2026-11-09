"""CalibRAG Retrieval Package (Member 2: Data, Hybrid RAG & GraphRAG Lead).

Exports are lazy so importing ``retrieval`` does not pull in Docling, asyncpg, or
later-phase ML dependencies. Existing import paths continue to work.
"""
from __future__ import annotations

import importlib
from typing import Dict, Tuple


_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Existing contracts and scaffolds.
    "BaseVectorRetriever": ("retrieval.interfaces", "BaseVectorRetriever"),
    "BaseBM25Retriever": ("retrieval.interfaces", "BaseBM25Retriever"),
    "BaseHybridReranker": ("retrieval.interfaces", "BaseHybridReranker"),
    "BaseGraphRAGRetriever": ("retrieval.interfaces", "BaseGraphRAGRetriever"),
    "DoclingDocumentParser": ("retrieval.docling_parser", "DoclingDocumentParser"),
    "DocumentChunk": ("retrieval.schemas", "DocumentChunk"),
    "PGHybridSearchEngine": ("retrieval.pg_hybrid_search", "PGHybridSearchEngine"),
    "LightRAGGraphEngine": ("retrieval.lightrag_indexer", "LightRAGGraphEngine"),
    "GraphNode": ("retrieval.lightrag_indexer", "GraphNode"),
    "GraphEdge": ("retrieval.lightrag_indexer", "GraphEdge"),
    "run_retrieval_recall_ablation": (
        "retrieval.recall_ablation",
        "run_retrieval_recall_ablation",
    ),
    "format_markdown_table": ("retrieval.recall_ablation", "format_markdown_table"),
    # Phase 2 foundation.
    "RetrievalSettings": ("retrieval.config", "RetrievalSettings"),
    "get_settings": ("retrieval.config", "get_settings"),
    "Database": ("retrieval.db", "Database"),
    "IngestionRepository": ("retrieval.db", "IngestionRepository"),
    "apply_migrations": ("retrieval.db", "apply_migrations"),
    "DoclingParser": ("retrieval.ingestion", "DoclingParser"),
    "chunk_blocks": ("retrieval.ingestion", "chunk_blocks"),
    "IngestionPipeline": ("retrieval.ingestion", "IngestionPipeline"),
    # Phase 3 hybrid retrieval.
    "RetrievedChunk": ("retrieval.retrieval_types", "RetrievedChunk"),
    "BgeM3Embedder": ("retrieval.embeddings", "BgeM3Embedder"),
    "BaseEmbedder": ("retrieval.embeddings", "BaseEmbedder"),
    "PGVectorStore": ("retrieval.vector", "PGVectorStore"),
    "PGVectorRetriever": ("retrieval.vector", "PGVectorRetriever"),
    "ParadeDBBM25Retriever": ("retrieval.bm25", "ParadeDBBM25Retriever"),
    "normalize_bm25_query": ("retrieval.bm25", "normalize_bm25_query"),
    "reciprocal_rank_fusion": ("retrieval.hybrid", "reciprocal_rank_fusion"),
    "rrf_fuse": ("retrieval.hybrid", "rrf_fuse"),
    "BgeReranker": ("retrieval.reranking", "BgeReranker"),
    "RetrievalService": ("retrieval.service", "RetrievalService"),
    "HybridRetrievalEngine": ("retrieval.service", "HybridRetrievalEngine"),
    "HybridGraphRetrievalResult": (
        "retrieval.service",
        "HybridGraphRetrievalResult",
    ),
    # Phase 5 evaluation harness.
    "RetrievalEvalCase": ("retrieval.evaluation", "RetrievalEvalCase"),
    "RetrievalEvalReport": ("retrieval.evaluation", "RetrievalEvalReport"),
    "evaluate_retrieval": ("retrieval.evaluation", "evaluate_retrieval"),
    "evaluate_service": ("retrieval.evaluation", "evaluate_service"),
    "load_eval_cases": ("retrieval.evaluation", "load_eval_cases"),
    "benchmark_retriever": ("retrieval.evaluation", "benchmark_retriever"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'retrieval' has no attribute {name!r}") from exc
    module = importlib.import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
