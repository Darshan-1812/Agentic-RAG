"""Unified dense + BM25 + RRF + cross-encoder retrieval pipeline."""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.schemas import ChunkEvidence
from retrieval.bm25.paradedb import ParadeDBBM25Retriever
from retrieval.config import RetrievalSettings, get_settings
from retrieval.embeddings.bge_m3 import BgeM3Embedder
from retrieval.hybrid.rrf import reciprocal_rank_fusion
from retrieval.reranking.bge_reranker import BgeReranker
from retrieval.retrieval_types import RetrievedChunk
from retrieval.schemas import validate_workspace_id
from retrieval.vector.pgvector import PGVectorRetriever, PGVectorStore


class RetrievalInputError(ValueError):
    """Raised when a retrieval request is invalid."""


@dataclass
class HybridGraphRetrievalResult:
    """Combined hybrid evidence and workspace-scoped graph context."""

    query: str
    workspace_id: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    graph: Dict[str, Any] = field(default_factory=dict)
    graph_error: Optional[str] = None

    @property
    def evidence(self) -> List[ChunkEvidence]:
        """Convert reranked chunks into the shared evidence DTO."""
        return [chunk.to_chunk_evidence() for chunk in self.chunks]

    @property
    def graph_context(self) -> str:
        """Return the graph context summary, or an empty string."""
        value = self.graph.get("context_summary") if self.graph else None
        return str(value or "")


class RetrievalService:
    """Run vector and BM25 retrieval concurrently, fuse with RRF, then rerank."""

    def __init__(
        self,
        database: Any = None,
        *,
        settings: Optional[RetrievalSettings] = None,
        embedder: Any = None,
        reranker: Any = None,
        vector_retriever: Any = None,
        bm25_retriever: Any = None,
        vector_store: Any = None,
        graph_engine: Any = None,
    ):
        self.settings = settings or get_settings()
        self.database = database
        self.embedder = embedder or BgeM3Embedder(self.settings)
        self.reranker = reranker or BgeReranker(self.settings)
        self.vector_retriever = vector_retriever or PGVectorRetriever(
            database, self.embedder, self.settings
        )
        self.bm25_retriever = bm25_retriever or ParadeDBBM25Retriever(
            database, self.settings
        )
        self.vector_store = vector_store or PGVectorStore(
            database, self.embedder, self.settings
        )
        self.graph_engine = graph_engine
        self._owned_graph_engines: Dict[str, Any] = {}

    async def index_chunks(
        self,
        workspace_id: str,
        chunks: Sequence[Any],
        batch_size: Optional[int] = None,
    ) -> int:
        return await self.vector_store.upsert_embeddings(
            workspace_id, chunks, batch_size=batch_size
        )

    async def index_document(self, workspace_id: str, doc_id: str) -> int:
        from retrieval.db.repository import IngestionRepository

        repository = IngestionRepository(self.database)
        chunks = await repository.get_chunks(workspace_id, doc_id)
        return await self.index_chunks(workspace_id, chunks)

    async def retrieve_chunks(
        self,
        query: str,
        top_k: Optional[int] = None,
        workspace_id: str = "default",
    ) -> List[RetrievedChunk]:
        if not isinstance(query, str):
            raise RetrievalInputError("query must be a string")
        if not query.strip():
            return []
        tenant = validate_workspace_id(workspace_id)
        limit = top_k if top_k is not None else self.settings.retrieval_top_k
        if limit < 1:
            raise RetrievalInputError("top_k must be >= 1")

        vector_depth = max(self.settings.vector_candidate_depth, limit)
        bm25_depth = max(self.settings.bm25_candidate_depth, limit)
        hybrid_depth = max(self.settings.hybrid_candidate_depth, limit)

        vector_results, bm25_results = await asyncio.gather(
            self.vector_retriever.retrieve(
                query, top_k=vector_depth, workspace_id=tenant
            ),
            self.bm25_retriever.search(
                query, top_k=bm25_depth, workspace_id=tenant
            ),
        )
        fused = reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            k=self.settings.rrf_k,
            top_k=hybrid_depth,
        )
        if not fused:
            return []

        rerank_depth = min(
            max(self.settings.rerank_candidate_depth, limit), len(fused)
        )
        return await self.reranker.rerank(
            query, fused[:rerank_depth], top_k=limit
        )

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        workspace_id: str = "default",
    ) -> List[ChunkEvidence]:
        chunks = await self.retrieve_chunks(
            query, top_k=top_k, workspace_id=workspace_id
        )
        return [chunk.to_chunk_evidence() for chunk in chunks]

    async def retrieve_with_graph(
        self,
        query: str,
        top_k: Optional[int] = None,
        workspace_id: str = "default",
        *,
        entity_seeds: Sequence[str] = (),
        graph_engine: Any = None,
        max_hops: int = 2,
    ) -> HybridGraphRetrievalResult:
        """Run hybrid retrieval and workspace-scoped graph traversal together.

        Graph failures are reported through ``graph_error`` and never discard
        valid hybrid evidence. The existing ``retrieve``/``retrieve_chunks``
        methods remain unchanged for callers that do not need graph context.
        """
        if not isinstance(query, str):
            raise RetrievalInputError("query must be a string")
        if not isinstance(max_hops, int) or isinstance(max_hops, bool) or max_hops < 0:
            raise RetrievalInputError("max_hops must be an integer >= 0")

        tenant = validate_workspace_id(workspace_id)
        seeds = self._normalise_seeds(entity_seeds)
        if not query.strip():
            return HybridGraphRetrievalResult(
                query=query,
                workspace_id=tenant,
                graph=self._empty_graph_result(query, seeds, max_hops, tenant),
            )

        if seeds:
            graph_engine = graph_engine or self._get_graph_engine(tenant)
        else:
            graph_engine = None
        chunks, graph_result = await asyncio.gather(
            self.retrieve_chunks(query, top_k=top_k, workspace_id=tenant),
            self._traverse_graph_safe(
                graph_engine,
                query=query,
                entity_seeds=seeds,
                max_hops=max_hops,
                workspace_id=tenant,
            ),
        )
        graph, graph_error = graph_result
        return HybridGraphRetrievalResult(
            query=query,
            workspace_id=tenant,
            chunks=chunks,
            graph=graph,
            graph_error=graph_error,
        )

    def _get_graph_engine(self, workspace_id: str) -> Any:
        if self.graph_engine is not None:
            return self.graph_engine
        engine = self._owned_graph_engines.get(workspace_id)
        if engine is None:
            from retrieval.lightrag_indexer import LightRAGGraphEngine

            engine = LightRAGGraphEngine(
                settings=self.settings, workspace_id=workspace_id
            )
            self._owned_graph_engines[workspace_id] = engine
        return engine

    async def _traverse_graph_safe(
        self,
        graph_engine: Any,
        *,
        query: str,
        entity_seeds: Sequence[str],
        max_hops: int,
        workspace_id: str,
    ) -> tuple[Dict[str, Any], Optional[str]]:
        empty = self._empty_graph_result(query, entity_seeds, max_hops, workspace_id)
        if not entity_seeds:
            return empty, None
        if graph_engine is None:
            return empty, "graph engine is not configured"

        traverse = getattr(graph_engine, "traverse_entities", None)
        if not callable(traverse):
            return empty, "graph engine does not implement traverse_entities"

        timeout = self.settings.graph_traversal_timeout_seconds
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    traverse,
                    query=query,
                    entity_seeds=list(entity_seeds),
                    max_hops=max_hops,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return empty, f"graph traversal timed out after {timeout:g}s"
        except Exception as exc:  # noqa: BLE001 - graph failure must be non-fatal
            return empty, self._sanitise_graph_error(
                f"{type(exc).__name__}: {exc}"
            )

        if not isinstance(result, dict):
            return empty, "graph traversal returned a non-dict result"
        result_workspace = result.get("workspace_id")
        if result_workspace not in (None, workspace_id):
            return empty, "graph traversal returned a different workspace"
        return self._normalise_graph_result(result, empty), None

    @staticmethod
    def _normalise_seeds(entity_seeds: Sequence[str]) -> List[str]:
        if entity_seeds is None:
            return []
        if isinstance(entity_seeds, str):
            entity_seeds = [entity_seeds]
        seeds: List[str] = []
        for seed in entity_seeds:
            if not isinstance(seed, str):
                continue
            value = seed.strip()
            if value and value not in seeds:
                seeds.append(value)
        return seeds

    @staticmethod
    def _empty_graph_result(
        query: str,
        entity_seeds: Sequence[str],
        max_hops: int,
        workspace_id: str,
    ) -> Dict[str, Any]:
        return {
            "query": query,
            "seed_entities": list(entity_seeds),
            "max_hops": max_hops,
            "total_nodes_found": 0,
            "nodes": [],
            "edges": [],
            "context_summary": "",
            "truncated": False,
            "workspace_id": workspace_id,
        }

    @staticmethod
    def _normalise_graph_result(
        result: Dict[str, Any], empty: Dict[str, Any]
    ) -> Dict[str, Any]:
        graph = dict(empty)
        graph.update(result)
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        graph["nodes"] = list(nodes) if isinstance(nodes, (list, tuple)) else []
        graph["edges"] = list(edges) if isinstance(edges, (list, tuple)) else []
        graph["context_summary"] = str(graph.get("context_summary") or "")
        graph["seed_entities"] = list(graph.get("seed_entities") or [])
        graph["truncated"] = bool(graph.get("truncated", False))
        graph["total_nodes_found"] = int(
            graph.get("total_nodes_found") or len(graph["nodes"])
        )
        graph["workspace_id"] = empty["workspace_id"]
        return graph

    @staticmethod
    def _sanitise_graph_error(message: str) -> str:
        value = " ".join(str(message).split())
        value = re.sub(r"sk-[A-Za-z0-9_\-]+", "sk-***", value)
        value = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer ***", value)
        return value[:500]

    def close(self) -> None:
        for component in (self.embedder, self.reranker):
            close = getattr(component, "close", None)
            if callable(close):
                close()
        for engine in self._owned_graph_engines.values():
            close = getattr(engine, "close", None)
            if callable(close):
                close()
        self._owned_graph_engines.clear()


HybridRetrievalEngine = RetrievalService
