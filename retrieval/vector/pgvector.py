"""PostgreSQL pgvector persistence and HNSW dense retrieval."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from retrieval.config import RetrievalSettings, get_settings
from retrieval.retrieval_types import RetrievedChunk
from retrieval.schemas import validate_workspace_id


class VectorStoreError(RuntimeError):
    """Base class for pgvector persistence and retrieval failures."""


class VectorIndexError(VectorStoreError):
    """Raised when chunks cannot be indexed safely."""


class VectorRetrievalError(VectorStoreError):
    """Raised when pgvector retrieval cannot run."""


def _json_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _vector_literal(vector: object) -> str:
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1:
        raise VectorIndexError(f"expected a 1-dimensional vector, got {values.shape}")
    if values.size == 0 or not np.isfinite(values).all():
        raise VectorIndexError("vector must be non-empty and contain finite values")
    return "[" + ",".join(f"{float(value):.9g}" for value in values) + "]"


def _retrieved_from_record(record: Any, rank: int) -> RetrievedChunk:
    data = dict(record)
    distance = float(data["distance"])
    return RetrievedChunk(
        chunk_id=str(data["chunk_id"]),
        doc_id=str(data["doc_id"]),
        content=str(data["content"]),
        page_number=int(data["page_number"]) if data.get("page_number") else None,
        source_type=str(data.get("source_type") or "pdf"),
        metadata=_json_dict(data.get("metadata")),
        section_hierarchy=list(data.get("section_hierarchy") or []),
        has_table=bool(data.get("has_table")),
        table_markdown=data.get("table_markdown"),
        vector_distance=distance,
        vector_similarity=1.0 - distance,
        vector_rank=rank,
        retrieval_sources=("vector",),
        embedding_model=str(data.get("model_name") or ""),
        embedding_model_revision=str(data.get("model_revision") or ""),
    )


class PGVectorStore:
    """Workspace-scoped embedding indexer for ``chunk_embeddings``."""

    def __init__(
        self,
        database: Any,
        embedder: Any,
        settings: Optional[RetrievalSettings] = None,
    ):
        if not hasattr(database, "acquire"):
            raise TypeError("database must expose an async acquire() context manager")
        if not callable(getattr(embedder, "embed_documents", None)):
            raise TypeError("embedder must expose async embed_documents()")
        self.database = database
        self.embedder = embedder
        self.settings = settings or get_settings()

    async def upsert_embeddings(
        self,
        workspace_id: str,
        chunks: Sequence[Any],
        batch_size: Optional[int] = None,
    ) -> int:
        """Embed and upsert chunks atomically, replacing same-model vectors."""
        tenant = validate_workspace_id(workspace_id)
        items = list(chunks)
        if not items:
            return 0

        chunk_ids: List[str] = []
        texts: List[str] = []
        seen = set()
        for item in items:
            chunk_id = str(getattr(item, "chunk_id", "") or "")
            content = getattr(item, "content", None)
            if not chunk_id:
                raise VectorIndexError("every chunk must have a chunk_id")
            if not isinstance(content, str) or not content.strip():
                raise VectorIndexError(f"chunk {chunk_id} has no content to embed")
            if chunk_id in seen:
                raise VectorIndexError(f"duplicate chunk_id in indexing batch: {chunk_id}")
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)
            texts.append(content)

        async with self.database.acquire() as connection:
            existing = await connection.fetch(
                "SELECT chunk_id FROM document_chunks "
                "WHERE workspace_id = $1 AND chunk_id = ANY($2::text[])",
                tenant,
                chunk_ids,
            )
        existing_ids = {str(row["chunk_id"]) for row in existing}
        missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in existing_ids]
        if missing:
            preview = ", ".join(missing[:5])
            raise VectorIndexError(
                f"{len(missing)} chunk(s) do not belong to workspace {tenant!r}; "
                f"first missing: {preview}"
            )

        vectors = await self.embedder.embed_documents(texts, batch_size=batch_size)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(chunk_ids), self.settings.embedding_dimension):
            raise VectorIndexError(
                f"expected {(len(chunk_ids), self.settings.embedding_dimension)} "
                f"embeddings, got {vectors.shape}"
            )

        model_name = str(getattr(self.embedder, "model_name", "") or "")
        model_revision = str(getattr(self.embedder, "model_revision", "") or "")
        if not model_name or not model_revision:
            raise VectorIndexError("embedder must expose model_name and model_revision")

        rows = [
            (
                chunk_id,
                tenant,
                model_name,
                model_revision,
                _vector_literal(vector),
            )
            for chunk_id, vector in zip(chunk_ids, vectors)
        ]
        async with self.database.acquire() as connection:
            async with connection.transaction():
                await connection.executemany(
                    """
                    INSERT INTO chunk_embeddings
                        (chunk_id, workspace_id, model_name, model_revision, embedding)
                    VALUES ($1, $2, $3, $4, $5::vector)
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET model_name = EXCLUDED.model_name,
                        model_revision = EXCLUDED.model_revision,
                        embedding = EXCLUDED.embedding,
                        created_at = now()
                    WHERE chunk_embeddings.workspace_id = EXCLUDED.workspace_id
                    """,
                    rows,
                )
        return len(rows)

    async def delete_embeddings(
        self, workspace_id: str, chunk_ids: Iterable[str]
    ) -> int:
        tenant = validate_workspace_id(workspace_id)
        ids = [str(value) for value in chunk_ids]
        if not ids:
            return 0
        async with self.database.acquire() as connection:
            result = await connection.execute(
                "DELETE FROM chunk_embeddings "
                "WHERE workspace_id = $1 AND chunk_id = ANY($2::text[])",
                tenant,
                ids,
            )
        try:
            return int(result.split()[-1])
        except (IndexError, ValueError):
            return 0

    async def count_embeddings(self, workspace_id: str) -> int:
        tenant = validate_workspace_id(workspace_id)
        async with self.database.acquire() as connection:
            value = await connection.fetchval(
                "SELECT count(*) FROM chunk_embeddings WHERE workspace_id = $1",
                tenant,
            )
        return int(value or 0)


class PGVectorRetriever:
    """Dense HNSW retriever over pgvector cosine distance."""

    def __init__(
        self,
        database: Any,
        embedder: Any,
        settings: Optional[RetrievalSettings] = None,
    ):
        if not hasattr(database, "acquire"):
            raise TypeError("database must expose an async acquire() context manager")
        if not callable(getattr(embedder, "embed_query", None)):
            raise TypeError("embedder must expose async embed_query()")
        self.database = database
        self.embedder = embedder
        self.settings = settings or get_settings()

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        workspace_id: str = "default",
        *,
        candidate_depth: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not isinstance(query, str):
            raise VectorRetrievalError("query must be a string")
        if not query.strip():
            return []
        tenant = validate_workspace_id(workspace_id)
        limit = top_k if top_k is not None else self.settings.retrieval_top_k
        if limit < 1:
            raise VectorRetrievalError("top_k must be >= 1")
        depth = candidate_depth or self.settings.vector_candidate_depth
        if depth < limit:
            depth = limit
        ef_search = max(self.settings.hnsw_ef_search, depth)

        vector = await self.embedder.embed_query(query)
        literal = _vector_literal(vector)
        model_name = str(getattr(self.embedder, "model_name", "") or "")
        model_revision = str(getattr(self.embedder, "model_revision", "") or "")
        if not model_name or not model_revision:
            raise VectorRetrievalError(
                "embedder must expose model_name and model_revision"
            )

        sql = """
            WITH candidates AS (
                SELECT c.chunk_id,
                       c.doc_id,
                       c.content,
                       c.page_number,
                       c.section_hierarchy,
                       c.has_table,
                       c.table_markdown,
                       c.source_type,
                       c.metadata,
                       e.model_name,
                       e.model_revision,
                       e.embedding <=> $1::vector AS distance
                FROM chunk_embeddings e
                JOIN document_chunks c
                  ON c.chunk_id = e.chunk_id
                 AND c.workspace_id = e.workspace_id
                JOIN documents d
                  ON d.doc_id = c.doc_id
                 AND d.workspace_id = c.workspace_id
                WHERE e.workspace_id = $2
                  AND e.model_name = $3
                  AND e.model_revision = $4
                  AND d.status <> 'deleted'
                ORDER BY e.embedding <=> $1::vector
                LIMIT $5
            )
            SELECT *, 1.0 - distance AS distance_rank
            FROM candidates
            ORDER BY distance
        """
        try:
            async with self.database.acquire() as connection:
                async with connection.transaction():
                    await connection.execute(
                        f"SET LOCAL hnsw.ef_search = {int(ef_search)}"
                    )
                    rows = await connection.fetch(
                        sql,
                        literal,
                        tenant,
                        model_name,
                        model_revision,
                        depth,
                    )
        except Exception as exc:
            raise VectorRetrievalError(
                f"pgvector retrieval failed: {type(exc).__name__}: {exc}"
            ) from exc

        return [
            _retrieved_from_record(row, rank)
            for rank, row in enumerate(rows[:limit], start=1)
        ]
