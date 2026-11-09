"""Unit tests for Member 2: Retrieval, Docling parsing, Hybrid Search & LightRAG."""
import pytest
from core.schemas import ChunkEvidence
from retrieval.docling_parser import DoclingDocumentParser
from retrieval.pg_hybrid_search import PGHybridSearchEngine
from retrieval.config import RetrievalSettings
from retrieval.lightrag_indexer import LightRAGGraphEngine
from retrieval.recall_ablation import run_retrieval_recall_ablation


def test_docling_parser_provenance():
    parser = DoclingDocumentParser(target_chunk_size=200)
    chunks = parser._generate_simulated_chunks("QA_SOP_2026.pdf", "QA_SOP")
    assert len(chunks) == 1
    assert chunks[0].page_number == 12
    assert "0.5%" in chunks[0].content


def test_hybrid_search_rrf_and_cross_encoder():
    engine = PGHybridSearchEngine(rrf_k=60)
    c1 = ChunkEvidence(chunk_id="c1", doc_id="QA_SOP", content="Maximum defect tolerance percentage is 0.5% for lithium cell batches.", cross_encoder_logit=0.0)
    c2 = ChunkEvidence(chunk_id="c2", doc_id="General", content="General factory cafeteria guidelines and shift schedules.", cross_encoder_logit=0.0)
    engine.index_chunks([c1, c2])

    # Vector search
    vec_results = engine.retrieve("defect tolerance percentage", top_k=2)
    assert vec_results[0].chunk_id == "c1"

    # BM25 search
    bm25_results = engine.search("defect tolerance", top_k=2)
    assert bm25_results[0].chunk_id == "c1"

    # RRF merge
    rrf_merged = engine.reciprocal_rank_fusion(vec_results, bm25_results, top_k=1)
    assert rrf_merged[0].chunk_id == "c1"

    # Cross-encoder rerank
    reranked = engine.rerank("defect tolerance percentage", [c1, c2], top_k=2)
    assert reranked[0].chunk_id == "c1"
    assert reranked[0].cross_encoder_logit > reranked[1].cross_encoder_logit


def test_lightrag_without_index_returns_empty(tmp_path):
    settings = RetrievalSettings(graph_storage_root=tmp_path)
    graph = LightRAGGraphEngine(settings=settings, workspace_id="member2-empty")
    res = graph.traverse_entities(
        query="Who is lead architect for VoltCore modules?",
        entity_seeds=["VoltCore"],
        max_hops=2,
    )
    graph.close()

    assert res["total_nodes_found"] == 0
    assert res["nodes"] == []
    assert res["edges"] == []


def test_recall_ablation_metrics():
    ablation = run_retrieval_recall_ablation()
    assert "Dense Vector Only (pgvector HNSW)" in ablation
    assert "Hybrid RRF + Cross-Encoder + LightRAG (Proposed CalibRAG Retrieval)" in ablation
    calibrag = ablation["Hybrid RRF + Cross-Encoder + LightRAG (Proposed CalibRAG Retrieval)"]
    assert calibrag["Recall@5"] >= 0.95
    assert calibrag["MRR@10"] >= 0.90
