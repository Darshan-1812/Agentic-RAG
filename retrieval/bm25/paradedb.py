"""ParadeDB pg_search BM25 lexical retrieval."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from retrieval.config import RetrievalSettings, get_settings
from retrieval.retrieval_types import RetrievedChunk
from retrieval.schemas import validate_workspace_id


class BM25RetrievalError(RuntimeError):
    """Raised when ParadeDB BM25 retrieval cannot run."""


_QUERY_TOKEN_RE = re.compile(r"[^\W_]+(?:[._-][^\W_]+)*", re.UNICODE)
_MAX_QUERY_TOKENS = 64


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


def normalize_bm25_query(query: str) -> str:
    """Extract plain lexical terms so syntax cannot reach ParadeDB.

    ParadeDB receives the result through a bound parameter. Removing query
    operators and punctuation prevents malformed boolean/phrase syntax while
    preserving words, numbers, model codes, and dotted versions.
    """
    if not isinstance(query, str):
        return ""
    tokens = _QUERY_TOKEN_RE.findall(query)
    return " ".join(tokens[:_MAX_QUERY_TOKENS])


def _retrieved_from_record(record: Any, rank: int) -> RetrievedChunk:
    data = dict(record)
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
        bm25_score=float(data.get("bm25_score") or 0.0),
        bm25_rank=rank,
        retrieval_sources=("bm25",),
    )


class ParadeDBBM25Retriever:
    """Workspace-scoped BM25 search using ``paradedb.match`` and ``paradedb.score``."""

    def __init__(
        self,
        database: Any,
        settings: Optional[RetrievalSettings] = None,
    ):
        if not hasattr(database, "acquire"):
            raise TypeError("database must expose an async acquire() context manager")
        self.database = database
        self.settings = settings or get_settings()

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        workspace_id: str = "default",
        *,
        candidate_depth: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not isinstance(query, str):
            raise BM25RetrievalError("query must be a string")
        lexical_query = normalize_bm25_query(query)
        if not lexical_query:
            return []

        tenant = validate_workspace_id(workspace_id)
        limit = top_k if top_k is not None else self.settings.retrieval_top_k
        if limit < 1:
            raise BM25RetrievalError("top_k must be >= 1")
        depth = candidate_depth or self.settings.bm25_candidate_depth
        if depth < limit:
            depth = limit

        sql = """
            SELECT c.chunk_id,
                   c.doc_id,
                   c.content,
                   c.page_number,
                   c.section_hierarchy,
                   c.has_table,
                   c.table_markdown,
                   c.source_type,
                   c.metadata,
                   paradedb.score(c) AS bm25_score
            FROM document_chunks c
            JOIN documents d
              ON d.doc_id = c.doc_id
             AND d.workspace_id = c.workspace_id
            WHERE c.workspace_id = $1
              AND c.content @@@ paradedb.match('content', $2)
              AND d.status <> 'deleted'
            ORDER BY bm25_score DESC, c.chunk_index ASC, c.chunk_id ASC
            LIMIT $3
        """
        try:
            async with self.database.acquire() as connection:
                rows = await connection.fetch(
                    sql, tenant, lexical_query, depth
                )
        except Exception as exc:
            raise BM25RetrievalError(
                f"ParadeDB BM25 retrieval failed: {type(exc).__name__}: {exc}"
            ) from exc

        return [
            _retrieved_from_record(row, rank)
            for rank, row in enumerate(rows[:limit], start=1)
        ]
