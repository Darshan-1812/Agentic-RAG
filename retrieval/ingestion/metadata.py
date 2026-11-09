"""File metadata, hashing, and validation helpers for ingestion."""
from __future__ import annotations

import hashlib
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from retrieval.config import SUPPORTED_EXTENSIONS
from retrieval.schemas import (
    DocumentMetadata,
    compute_document_id,
    validate_workspace_id,
)


class UnsupportedDocumentError(ValueError):
    """Raised for files whose extension/MIME type is not supported."""


class DocumentTooLargeError(ValueError):
    """Raised when a source file exceeds the configured upload limit."""


class InvalidDocumentError(ValueError):
    """Raised when a source file is missing, empty, or unsafe to read."""


def detect_mime_type(path: Path, declared_mime_type: Optional[str] = None) -> str:
    """Return the canonical MIME type for a supported document extension.

    The extension is authoritative for this BYOK ingestion path; client-provided
    MIME types are not trusted. ``declared_mime_type`` is accepted for API
    compatibility and is deliberately not used for validation.
    """
    extension = path.suffix.lower()
    expected = SUPPORTED_EXTENSIONS.get(extension)
    if expected is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedDocumentError(
            f"Unsupported document extension {extension or '<none>'}; supported: {supported}"
        )
    guessed, _ = mimetypes.guess_type(path.name)
    return expected or guessed or "application/octet-stream"


def validate_source_file(path: Path, max_upload_bytes: int) -> os.stat_result:
    """Validate a source path and return its stat record."""
    if not path.exists():
        raise InvalidDocumentError(f"Document does not exist: {path}")
    if path.is_symlink():
        raise InvalidDocumentError("Symbolic-link uploads are not accepted")
    if not path.is_file():
        raise InvalidDocumentError(f"Document path is not a regular file: {path}")
    stat = path.stat()
    if stat.st_size <= 0:
        raise InvalidDocumentError("Empty documents are not supported")
    if stat.st_size > max_upload_bytes:
        raise DocumentTooLargeError(
            f"Document is {stat.st_size} bytes; limit is {max_upload_bytes} bytes"
        )
    detect_mime_type(path)
    return stat


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a streaming SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def build_document_metadata(
    *,
    source_path: Path,
    workspace_id: str,
    content_hash: str,
    storage_path: str,
    doc_title: str,
    mime_type: str,
    page_count: int,
    filename: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> DocumentMetadata:
    """Construct the persisted document metadata record."""
    tenant = validate_workspace_id(workspace_id)
    stat = source_path.stat()
    display_filename = Path(filename).name if filename else source_path.name
    if not display_filename:
        display_filename = source_path.name
    metadata: Dict[str, Any] = {
        "original_filename": display_filename,
        "file_created_at": _utc_timestamp(stat.st_ctime),
        "file_modified_at": _utc_timestamp(stat.st_mtime),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return DocumentMetadata(
        doc_id=compute_document_id(tenant, content_hash),
        workspace_id=tenant,
        filename=display_filename,
        doc_title=doc_title or Path(display_filename).stem,
        mime_type=mime_type,
        content_hash=content_hash,
        byte_size=stat.st_size,
        page_count=max(0, int(page_count)),
        storage_path=storage_path,
        metadata=metadata,
    )
