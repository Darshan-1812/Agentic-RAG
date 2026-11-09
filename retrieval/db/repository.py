"""Async persistence for workspaces, documents, chunks, and ingestion jobs.

All queries include ``workspace_id`` in the SQL predicate. Callers cannot use
these helpers to perform a global (cross-tenant) lookup.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Sequence

from retrieval.schemas import (
    DocumentChunk,
    DocumentMetadata,
    DocumentStatus,
    IngestionJobStatus,
    validate_workspace_id,
)


class WorkspaceRequiredError(ValueError):
    """Raised when a repository call is missing its tenant boundary."""


def _workspace(workspace_id: str) -> str:
    try:
        return validate_workspace_id(workspace_id)
    except ValueError as exc:
        raise WorkspaceRequiredError(str(exc)) from exc


def _uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)


def _json_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _status_value(status: DocumentStatus | IngestionJobStatus | str) -> str:
    if isinstance(status, (DocumentStatus, IngestionJobStatus)):
        return status.value
    return str(status)


def _document_from_record(record: Any) -> DocumentMetadata:
    data = dict(record)
    return DocumentMetadata(
        doc_id=str(data["doc_id"]),
        workspace_id=str(data["workspace_id"]),
        filename=str(data["filename"]),
        doc_title=str(data["doc_title"]),
        mime_type=str(data["mime_type"]),
        content_hash=str(data["content_hash"]),
        byte_size=int(data["byte_size"]),
        page_count=int(data["page_count"]),
        status=DocumentStatus(str(data["status"])),
        storage_path=str(data["storage_path"]),
        metadata=_json_dict(data.get("metadata")),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
    )


def _chunk_from_record(record: Any) -> DocumentChunk:
    data = dict(record)
    return DocumentChunk(
        chunk_id=str(data["chunk_id"]),
        doc_id=str(data["doc_id"]),
        doc_title=str(data["doc_title"]),
        content=str(data["content"]),
        page_number=int(data.get("page_number") or 1),
        section_hierarchy=list(data.get("section_hierarchy") or []),
        has_table=bool(data.get("has_table")),
        table_markdown=data.get("table_markdown"),
        char_count=int(data["char_count"]),
        token_count_est=int(data["token_count_est"]),
        workspace_id=str(data["workspace_id"]),
        chunk_index=int(data["chunk_index"]),
        source_type=str(data.get("source_type") or "pdf"),
        metadata=_json_dict(data.get("metadata")),
    )


class IngestionRepository:
    """Workspace-scoped persistence used by the ingestion pipeline."""

    def __init__(self, database: Any):
        if not hasattr(database, "acquire"):
            raise TypeError("database must expose an async acquire() context manager")
        self.database = database

    async def ensure_workspace(
        self,
        workspace_id: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        tenant = _workspace(workspace_id)
        async with self.database.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO workspaces (workspace_id, name, config)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (workspace_id) DO UPDATE
                SET name = COALESCE(EXCLUDED.name, workspaces.name),
                    config = CASE
                        WHEN EXCLUDED.config = '{}'::jsonb THEN workspaces.config
                        ELSE EXCLUDED.config
                    END,
                    updated_at = now()
                """,
                tenant,
                name or tenant,
                _json_dumps(config or {}),
            )

    async def get_workspace(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        tenant = _workspace(workspace_id)
        async with self.database.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT workspace_id, name, config, created_at, updated_at "
                "FROM workspaces WHERE workspace_id = $1",
                tenant,
            )
        if row is None:
            return None
        data = dict(row)
        data["config"] = _json_dict(data.get("config"))
        return data

    async def find_document_by_hash(
        self, workspace_id: str, content_hash: str
    ) -> Optional[DocumentMetadata]:
        tenant = _workspace(workspace_id)
        async with self.database.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT doc_id, workspace_id, filename, doc_title, mime_type,
                       content_hash, byte_size, page_count, status, storage_path,
                       metadata, created_at, updated_at
                FROM documents
                WHERE workspace_id = $1
                  AND content_hash = $2
                  AND status <> 'deleted'
                """,
                tenant,
                content_hash,
            )
        return _document_from_record(row) if row else None

    async def get_document(
        self, workspace_id: str, doc_id: str
    ) -> Optional[DocumentMetadata]:
        tenant = _workspace(workspace_id)
        async with self.database.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT doc_id, workspace_id, filename, doc_title, mime_type,
                       content_hash, byte_size, page_count, status, storage_path,
                       metadata, created_at, updated_at
                FROM documents
                WHERE workspace_id = $1 AND doc_id = $2
                """,
                tenant,
                _uuid(doc_id),
            )
        return _document_from_record(row) if row else None

    async def list_documents(
        self,
        workspace_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DocumentMetadata]:
        tenant = _workspace(workspace_id)
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        async with self.database.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT doc_id, workspace_id, filename, doc_title, mime_type,
                       content_hash, byte_size, page_count, status, storage_path,
                       metadata, created_at, updated_at
                FROM documents
                WHERE workspace_id = $1 AND status <> 'deleted'
                ORDER BY created_at DESC, doc_id
                LIMIT $2 OFFSET $3
                """,
                tenant,
                limit,
                offset,
            )
        return [_document_from_record(row) for row in rows]

    async def get_chunks(
        self,
        workspace_id: str,
        doc_id: str,
        limit: Optional[int] = None,
    ) -> List[DocumentChunk]:
        tenant = _workspace(workspace_id)
        if limit is not None and (limit < 1 or limit > 5000):
            raise ValueError("limit must be between 1 and 5000")
        sql = """
            SELECT c.chunk_id, c.doc_id, c.workspace_id, c.chunk_index,
                   c.content, c.page_number, c.section_hierarchy, c.has_table,
                   c.table_markdown, c.char_count, c.token_count_est,
                   c.source_type, c.metadata, d.doc_title
            FROM document_chunks c
            JOIN documents d
              ON d.doc_id = c.doc_id
             AND d.workspace_id = c.workspace_id
            WHERE c.workspace_id = $1 AND c.doc_id = $2 AND d.status <> 'deleted'
            ORDER BY c.chunk_index
        """
        params: List[Any] = [tenant, _uuid(doc_id)]
        if limit is not None:
            sql += " LIMIT $3"
            params.append(limit)
        async with self.database.acquire() as connection:
            rows = await connection.fetch(sql, *params)
        return [_chunk_from_record(row) for row in rows]

    async def create_job(
        self,
        workspace_id: str,
        source_filename: str,
        doc_id: Optional[str] = None,
    ) -> str:
        tenant = _workspace(workspace_id)
        job_id = uuid.uuid4()
        async with self.database.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO ingestion_jobs
                    (job_id, workspace_id, doc_id, source_filename, status, stats)
                VALUES ($1, $2, $3, $4, $5, '{}'::jsonb)
                """,
                job_id,
                tenant,
                _uuid(doc_id) if doc_id else None,
                source_filename,
                IngestionJobStatus.QUEUED.value,
            )
        return str(job_id)

    async def update_job(
        self,
        workspace_id: str,
        job_id: str,
        status: IngestionJobStatus | str,
        error: Optional[str] = None,
        stats: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> None:
        tenant = _workspace(workspace_id)
        status_value = _status_value(status)
        async with self.database.acquire() as connection:
            await connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = $3,
                    error = $4,
                    stats = CASE
                        WHEN $5::jsonb IS NULL THEN stats
                        ELSE $5::jsonb
                    END,
                    doc_id = COALESCE($6, doc_id),
                    started_at = CASE
                        WHEN $3 = 'running' AND started_at IS NULL THEN now()
                        ELSE started_at
                    END,
                    finished_at = CASE
                        WHEN $3 IN ('completed', 'failed', 'cancelled') THEN now()
                        ELSE finished_at
                    END
                WHERE workspace_id = $1 AND job_id = $2
                """,
                tenant,
                _uuid(job_id),
                status_value,
                error,
                _json_dumps(stats) if stats is not None else None,
                _uuid(doc_id) if doc_id else None,
            )

    async def save_document_with_chunks(
        self,
        document: DocumentMetadata,
        chunks: Sequence[DocumentChunk],
        job_id: Optional[str] = None,
    ) -> DocumentMetadata:
        """Atomically upsert a document and replace all of its chunks.

        Re-ingesting identical content replaces the previous chunk rows in the
        same transaction. Existing embeddings cascade-delete with their chunks.
        """
        tenant = _workspace(document.workspace_id)
        if document.workspace_id != tenant:
            raise WorkspaceRequiredError("document workspace_id is invalid")

        for chunk in chunks:
            if chunk.workspace_id != tenant:
                raise WorkspaceRequiredError(
                    f"chunk {chunk.chunk_id} belongs to a different workspace"
                )
            if chunk.doc_id != document.doc_id:
                raise WorkspaceRequiredError(
                    f"chunk {chunk.chunk_id} belongs to a different document"
                )

        async with self.database.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    INSERT INTO documents
                        (doc_id, workspace_id, filename, doc_title, mime_type,
                         content_hash, byte_size, page_count, status, storage_path,
                         metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
                    ON CONFLICT (workspace_id, content_hash) DO UPDATE
                    SET filename = EXCLUDED.filename,
                        doc_title = EXCLUDED.doc_title,
                        mime_type = EXCLUDED.mime_type,
                        byte_size = EXCLUDED.byte_size,
                        page_count = EXCLUDED.page_count,
                        status = EXCLUDED.status,
                        storage_path = EXCLUDED.storage_path,
                        metadata = EXCLUDED.metadata,
                        updated_at = now()
                    RETURNING doc_id, workspace_id, filename, doc_title, mime_type,
                              content_hash, byte_size, page_count, status,
                              storage_path, metadata, created_at, updated_at
                    """,
                    _uuid(document.doc_id),
                    tenant,
                    document.filename,
                    document.doc_title,
                    document.mime_type,
                    document.content_hash,
                    document.byte_size,
                    document.page_count,
                    _status_value(document.status),
                    document.storage_path,
                    _json_dumps(document.metadata),
                )
                if row is None:
                    raise RuntimeError("Document upsert returned no row")
                persisted = _document_from_record(row)

                await connection.execute(
                    "DELETE FROM document_chunks WHERE workspace_id = $1 AND doc_id = $2",
                    tenant,
                    _uuid(persisted.doc_id),
                )

                if chunks:
                    await connection.executemany(
                        """
                        INSERT INTO document_chunks
                            (chunk_id, doc_id, workspace_id, chunk_index, content,
                             page_number, section_hierarchy, has_table,
                             table_markdown, char_count, token_count_est,
                             source_type, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                                $12, $13::jsonb)
                        """,
                        [
                            (
                                chunk.chunk_id,
                                _uuid(persisted.doc_id),
                                tenant,
                                chunk.chunk_index,
                                chunk.content,
                                chunk.page_number,
                                list(chunk.section_hierarchy),
                                chunk.has_table,
                                chunk.table_markdown,
                                chunk.char_count,
                                chunk.token_count_est,
                                chunk.source_type,
                                _json_dumps(chunk.metadata),
                            )
                            for chunk in chunks
                        ],
                    )

                if job_id:
                    await connection.execute(
                        """
                        UPDATE ingestion_jobs
                        SET doc_id = $3
                        WHERE workspace_id = $1 AND job_id = $2
                        """,
                        tenant,
                        _uuid(job_id),
                        _uuid(persisted.doc_id),
                    )

                return persisted

    async def soft_delete_document(self, workspace_id: str, doc_id: str) -> bool:
        tenant = _workspace(workspace_id)
        async with self.database.acquire() as connection:
            result = await connection.execute(
                "UPDATE documents SET status = 'deleted', updated_at = now() "
                "WHERE workspace_id = $1 AND doc_id = $2",
                tenant,
                _uuid(doc_id),
            )
        return result.endswith(" 1")

    async def count_chunks(self, workspace_id: str, doc_id: str) -> int:
        tenant = _workspace(workspace_id)
        async with self.database.acquire() as connection:
            value = await connection.fetchval(
                "SELECT count(*) FROM document_chunks "
                "WHERE workspace_id = $1 AND doc_id = $2",
                tenant,
                _uuid(doc_id),
            )
        return int(value or 0)
