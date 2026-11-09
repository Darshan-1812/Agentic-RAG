"""Pydantic schemas for Member 2 ingestion and persistence.

The shared ``core.schemas.ChunkEvidence`` type is intentionally left unchanged
until the cross-member schema review. These schemas describe documents and
chunks while they are being ingested or read back from PostgreSQL.
"""
from __future__ import annotations

import math
import re
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


_WORKSPACE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DOC_ID_NAMESPACE = uuid.UUID("7cfbe9c2-3b36-4e03-9d0f-9b4f4a5e2e90")


class RetrievalSchemaError(ValueError):
    """Raised when an ingestion DTO is invalid."""


def validate_workspace_id(workspace_id: str) -> str:
    """Validate a tenant/workspace identifier.

    Workspace identifiers are used in SQL filters and storage paths, so they
    must be explicit, non-empty, and free of path traversal characters.
    """
    if not isinstance(workspace_id, str):
        raise RetrievalSchemaError("workspace_id must be a string")
    normalized = workspace_id.strip()
    if not normalized:
        raise RetrievalSchemaError("workspace_id must not be empty")
    if not _WORKSPACE_ID_RE.fullmatch(normalized):
        raise RetrievalSchemaError(
            "workspace_id must start with an alphanumeric character and contain "
            "only letters, numbers, '.', '_', ':', or '-' (max 128 chars)"
        )
    return normalized


def compute_document_id(workspace_id: str, content_hash: str) -> str:
    """Compute a deterministic document ID from tenant and content hash."""
    workspace = validate_workspace_id(workspace_id)
    if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
        raise RetrievalSchemaError("content_hash must be a lowercase SHA-256 hex digest")
    return str(uuid.uuid5(_DOC_ID_NAMESPACE, f"calibrag:doc:{workspace}:{content_hash}"))


def compute_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Compute the stable chunk ID used by ingestion and retrieval."""
    if not doc_id or not doc_id.strip():
        raise RetrievalSchemaError("doc_id must not be empty")
    if chunk_index < 0:
        raise RetrievalSchemaError("chunk_index must be >= 0")
    return f"{doc_id}:c{chunk_index:05d}"


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate used for metadata only."""
    return max(1, math.ceil(len(text) / 4))


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class IngestionJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentChunk(BaseModel):
    """Parsed document chunk with rich provenance metadata.

    The first ten fields preserve the original ``DocumentChunk`` contract used
    by the old ``retrieval.docling_parser`` module. The remaining fields are
    additive and are populated by the real ingestion pipeline.
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    content: str
    page_number: int = 1
    section_hierarchy: List[str] = Field(default_factory=list)
    has_table: bool = False
    table_markdown: Optional[str] = None
    char_count: int
    token_count_est: int

    workspace_id: Optional[str] = None
    chunk_index: int = 0
    source_type: str = "pdf"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("workspace_id")
    @classmethod
    def _validate_optional_workspace(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return validate_workspace_id(value)

    @field_validator("page_number")
    @classmethod
    def _validate_page_number(cls, value: int) -> int:
        return max(1, int(value))


class ParsedBlock(BaseModel):
    """A normalized structural block emitted by the Docling adapter."""

    block_type: Literal["text", "heading", "table", "list", "caption", "code", "formula"]
    text: str
    page_number: int = 1
    section_hierarchy: List[str] = Field(default_factory=list)
    level: int = 0
    table_markdown: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("page_number")
    @classmethod
    def _validate_page_number(cls, value: int) -> int:
        return max(1, int(value))


class ParsedDocument(BaseModel):
    """The normalized result of parsing one source file with Docling."""

    doc_title: str
    page_count: int = 0
    blocks: List[ParsedBlock] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Persisted document metadata (without chunk bodies)."""

    doc_id: str
    workspace_id: str
    filename: str
    doc_title: str
    mime_type: str
    content_hash: str
    byte_size: int
    page_count: int = 0
    status: DocumentStatus = DocumentStatus.READY
    storage_path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("workspace_id")
    @classmethod
    def _validate_workspace(cls, value: str) -> str:
        return validate_workspace_id(value)

    @field_validator("content_hash")
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("content_hash must be a lowercase SHA-256 hex digest")
        return value


class IngestionResult(BaseModel):
    """Result returned by the ingestion pipeline."""

    job_id: Optional[str] = None
    workspace_id: str
    doc_id: str
    document: DocumentMetadata
    chunks: List[DocumentChunk] = Field(default_factory=list)
    duplicate: bool = False
    status: IngestionJobStatus = IngestionJobStatus.COMPLETED

    @field_validator("workspace_id")
    @classmethod
    def _validate_workspace(cls, value: str) -> str:
        return validate_workspace_id(value)
