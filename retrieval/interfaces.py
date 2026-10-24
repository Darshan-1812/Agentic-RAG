"""Abstract interfaces for Member 2: Data, Hybrid RAG & GraphRAG Lead."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from core.schemas import ChunkEvidence


class BaseVectorRetriever(ABC):
    """Interface for pgvector (HNSW) dense semantic retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, workspace_id: str = "default") -> List[ChunkEvidence]:
        """Performs dense vector cosine similarity lookup."""
        pass


class BaseBM25Retriever(ABC):
    """Interface for ParadeDB pg_search Rust-based BM25 lexical retrieval."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5, workspace_id: str = "default") -> List[ChunkEvidence]:
        """Performs BM25 keyword search."""
        pass


class BaseHybridReranker(ABC):
    """Interface for Reciprocal Rank Fusion (RRF) + BGE-Reranker-v2-m3."""

    @abstractmethod
    def rerank(self, query: str, chunks: List[ChunkEvidence], top_k: int = 5) -> List[ChunkEvidence]:
        """Merges vector + BM25 results with RRF and computes Cross-Encoder logits."""
        pass


class BaseGraphRAGRetriever(ABC):
    """Interface for LightRAG dual-level entity-relationship graph traversal."""

    @abstractmethod
    def traverse_entities(self, query: str, entity_seeds: List[str], max_hops: int = 2) -> Dict[str, Any]:
        """Traverses entity relationships and returns graph topology context."""
        pass
