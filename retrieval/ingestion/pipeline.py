"""End-to-end BYOK ingestion orchestration."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from retrieval.config import RetrievalSettings, get_settings
from retrieval.ingestion.chunker import chunk_blocks
from retrieval.ingestion.docling_parser import DoclingParser
from retrieval.ingestion.metadata import (
    build_document_metadata,
    detect_mime_type,
    sha256_file,
    validate_source_file,
)
from retrieval.ingestion.storage import DocumentStorage
from retrieval.schemas import (
    DocumentStatus,
    IngestionJobStatus,
    IngestionResult,
    compute_document_id,
    validate_workspace_id,
)


class IngestionError(RuntimeError):
    """Raised when a document cannot be parsed, stored, or persisted."""


_SOURCE_TYPES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".md": "markdown",
    ".markdown": "markdown",
}


class IngestionPipeline:
    """Parse, chunk, store, and persist one document at a time."""

    def __init__(
        self,
        repository: Any,
        settings: Optional[RetrievalSettings] = None,
        storage: Optional[DocumentStorage] = None,
        parser: Optional[DoclingParser] = None,
    ):
        if not hasattr(repository, "save_document_with_chunks"):
            raise TypeError("repository must implement the ingestion repository contract")
        self.settings = settings or get_settings()
        self.repository = repository
        self.storage = storage or DocumentStorage(
            self.settings.resolved_storage_root,
            self.settings.max_upload_bytes,
        )
        self.parser = parser or DoclingParser(self.settings)

    async def ingest_file(
        self,
        workspace_id: str,
        file_path: Path | str,
        *,
        original_filename: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
    ) -> IngestionResult:
        """Ingest a PDF, DOCX, or Markdown file into a single workspace."""
        tenant = validate_workspace_id(workspace_id)
        source = Path(file_path)
        stat = validate_source_file(source, self.settings.max_upload_bytes)
        mime_type = detect_mime_type(source)
        content_hash = await asyncio.to_thread(sha256_file, source)
        doc_id = compute_document_id(tenant, content_hash)
        display_filename = Path(original_filename).name if original_filename else source.name
        if not display_filename:
            display_filename = source.name
        source_type = _SOURCE_TYPES.get(source.suffix.lower(), "document")

        await self.repository.ensure_workspace(tenant)
        existing = await self.repository.find_document_by_hash(tenant, content_hash)
        if (
            existing is not None
            and existing.status == DocumentStatus.READY
            and self.storage.exists(existing.storage_path)
        ):
            chunks = await self.repository.get_chunks(tenant, existing.doc_id)
            if job_id:
                await self.repository.update_job(
                    tenant,
                    job_id,
                    IngestionJobStatus.COMPLETED,
                    stats={
                        "duplicate": True,
                        "doc_id": existing.doc_id,
                        "chunk_count": len(chunks),
                    },
                    doc_id=existing.doc_id,
                )
            return IngestionResult(
                job_id=job_id,
                workspace_id=tenant,
                doc_id=existing.doc_id,
                document=existing,
                chunks=chunks,
                duplicate=True,
                status=IngestionJobStatus.COMPLETED,
            )

        active_job_id = job_id or await self.repository.create_job(
            tenant, display_filename, doc_id=doc_id
        )
        await self.repository.update_job(
            tenant,
            active_job_id,
            IngestionJobStatus.RUNNING,
            stats={"phase": "converting", "doc_id": doc_id},
            doc_id=doc_id,
        )

        stored_relative: Optional[str] = None
        stored_created = False
        persisted = None
        phase = "converting"
        try:
            parsed = await asyncio.to_thread(self.parser.parse, source, tenant)
            phase = "chunking"
            chunks = chunk_blocks(
                parsed.blocks,
                doc_id=doc_id,
                workspace_id=tenant,
                doc_title=parsed.doc_title,
                target_chars=self.settings.chunk_target_chars,
                max_chars=self.settings.chunk_max_chars,
                overlap_chars=self.settings.chunk_overlap_chars,
                source_type=source_type,
            )
            if not chunks:
                raise IngestionError("Document produced no chunks")

            phase = "storing"
            stored = await asyncio.to_thread(
                self.storage.store_file, source, tenant, content_hash
            )
            stored_relative = stored.relative_path
            stored_created = stored.created

            phase = "persisting"
            metadata = dict(parsed.metadata)
            if extra_metadata:
                metadata.update(extra_metadata)
            metadata.update(
                {
                    "chunk_count": len(chunks),
                    "source_type": source_type,
                    "ingestion_job_id": active_job_id,
                }
            )
            document = build_document_metadata(
                source_path=source,
                workspace_id=tenant,
                content_hash=content_hash,
                storage_path=stored.relative_path,
                doc_title=parsed.doc_title,
                mime_type=mime_type,
                page_count=parsed.page_count,
                filename=display_filename,
                extra_metadata=metadata,
            )
            persisted = await self.repository.save_document_with_chunks(
                document, chunks, job_id=active_job_id
            )

            phase = "finalizing"
            await self.repository.update_job(
                tenant,
                active_job_id,
                IngestionJobStatus.COMPLETED,
                stats={
                    "duplicate": False,
                    "doc_id": persisted.doc_id,
                    "chunk_count": len(chunks),
                    "page_count": parsed.page_count,
                    "byte_size": stat.st_size,
                    "storage_path": stored.relative_path,
                },
                doc_id=persisted.doc_id,
            )
            return IngestionResult(
                job_id=active_job_id,
                workspace_id=tenant,
                doc_id=persisted.doc_id,
                document=persisted,
                chunks=chunks,
                duplicate=False,
                status=IngestionJobStatus.COMPLETED,
            )
        except Exception as exc:
            if stored_created and stored_relative and persisted is None:
                try:
                    self.storage.delete(stored_relative)
                except Exception:
                    pass
            try:
                await self.repository.update_job(
                    tenant,
                    active_job_id,
                    IngestionJobStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                    stats={"phase": phase, "doc_id": doc_id},
                    doc_id=doc_id,
                )
            except Exception:
                pass
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError(
                f"Ingestion failed for {display_filename}: {type(exc).__name__}: {exc}"
            ) from exc
