"""Internal retrieval result types shared by dense, lexical, and reranking stages."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from core.schemas import ChunkEvidence


class MissingCrossEncoderLogit(ValueError):
    """Raised when final evidence is requested before cross-encoder reranking."""


def _first_not_none(current: Any, incoming: Any) -> Any:
    return current if current is not None else incoming


@dataclass
class RetrievedChunk:
    """A chunk moving through dense, BM25, RRF, and reranking stages.

    Intermediate stages intentionally keep ``cross_encoder_logit`` optional so
    callers cannot mistake a vector distance or BM25 score for a raw logit.
    """

    chunk_id: str
    doc_id: str
    content: str
    page_number: Optional[int] = None
    source_type: str = "pdf"
    metadata: Dict[str, Any] = field(default_factory=dict)
    section_hierarchy: List[str] = field(default_factory=list)
    has_table: bool = False
    table_markdown: Optional[str] = None

    vector_distance: Optional[float] = None
    vector_similarity: Optional[float] = None
    vector_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rrf_rank: Optional[int] = None
    cross_encoder_logit: Optional[float] = None
    rerank_rank: Optional[int] = None

    retrieval_sources: Tuple[str, ...] = ()
    embedding_model: Optional[str] = None
    embedding_model_revision: Optional[str] = None

    @property
    def has_cross_encoder_logit(self) -> bool:
        return self.cross_encoder_logit is not None

    def merged_with(self, other: "RetrievedChunk") -> "RetrievedChunk":
        """Merge duplicate chunk representations without losing stage scores."""
        if self.chunk_id != other.chunk_id or self.doc_id != other.doc_id:
            raise ValueError("only results for the same chunk and document can be merged")

        metadata = dict(self.metadata)
        metadata.update(other.metadata)
        sources: Tuple[str, ...] = tuple(
            dict.fromkeys(self.retrieval_sources + other.retrieval_sources)
        )
        return replace(
            self,
            page_number=_first_not_none(self.page_number, other.page_number),
            source_type=self.source_type or other.source_type,
            metadata=metadata,
            section_hierarchy=self.section_hierarchy or list(other.section_hierarchy),
            has_table=bool(self.has_table or other.has_table),
            table_markdown=_first_not_none(
                self.table_markdown, other.table_markdown
            ),
            vector_distance=_first_not_none(
                self.vector_distance, other.vector_distance
            ),
            vector_similarity=_first_not_none(
                self.vector_similarity, other.vector_similarity
            ),
            vector_rank=_first_not_none(self.vector_rank, other.vector_rank),
            bm25_score=_first_not_none(self.bm25_score, other.bm25_score),
            bm25_rank=_first_not_none(self.bm25_rank, other.bm25_rank),
            rrf_score=_first_not_none(self.rrf_score, other.rrf_score),
            rrf_rank=_first_not_none(self.rrf_rank, other.rrf_rank),
            cross_encoder_logit=_first_not_none(
                self.cross_encoder_logit, other.cross_encoder_logit
            ),
            rerank_rank=_first_not_none(self.rerank_rank, other.rerank_rank),
            retrieval_sources=sources,
            embedding_model=_first_not_none(
                self.embedding_model, other.embedding_model
            ),
            embedding_model_revision=_first_not_none(
                self.embedding_model_revision, other.embedding_model_revision
            ),
        )

    def score_metadata(self) -> Dict[str, Any]:
        """Return provenance plus namespaced retrieval scores for ``ChunkEvidence``."""
        metadata = dict(self.metadata)
        metadata["retrieval_sources"] = list(self.retrieval_sources)
        if self.section_hierarchy:
            metadata.setdefault("section_hierarchy", list(self.section_hierarchy))
        if self.has_table:
            metadata.setdefault("has_table", True)
        if self.table_markdown:
            metadata.setdefault("table_markdown", self.table_markdown)

        optional_scores = {
            "retrieval_vector_distance": self.vector_distance,
            "retrieval_vector_similarity": self.vector_similarity,
            "retrieval_vector_rank": self.vector_rank,
            "retrieval_bm25_score": self.bm25_score,
            "retrieval_bm25_rank": self.bm25_rank,
            "retrieval_rrf_score": self.rrf_score,
            "retrieval_rrf_rank": self.rrf_rank,
            "retrieval_cross_encoder_logit": self.cross_encoder_logit,
            "retrieval_rerank_rank": self.rerank_rank,
            "retrieval_embedding_model": self.embedding_model,
            "retrieval_embedding_model_revision": self.embedding_model_revision,
        }
        for key, value in optional_scores.items():
            if value is not None:
                metadata[key] = value
        return metadata

    def to_chunk_evidence(self) -> ChunkEvidence:
        """Build the shared final evidence DTO; requires a real reranker logit."""
        if self.cross_encoder_logit is None:
            raise MissingCrossEncoderLogit(
                f"chunk {self.chunk_id} has no cross-encoder logit; reranking is required"
            )
        return ChunkEvidence(
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            content=self.content,
            cross_encoder_logit=float(self.cross_encoder_logit),
            page_number=self.page_number,
            source_type=self.source_type,
            metadata=self.score_metadata(),
        )
