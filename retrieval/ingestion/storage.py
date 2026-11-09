"""Content-addressed document storage with workspace and path isolation."""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from retrieval.config import SUPPORTED_EXTENSIONS
from retrieval.ingestion.metadata import InvalidDocumentError, sha256_file
from retrieval.schemas import validate_workspace_id


_UNSAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


class StorageError(RuntimeError):
    """Raised when a stored file cannot be written or resolved safely."""


class StorageConflictError(StorageError):
    """Raised when content-addressed storage finds a hash mismatch."""


@dataclass(frozen=True)
class StorageResult:
    relative_path: str
    absolute_path: Path
    created: bool
    byte_size: int


def _safe_workspace_component(workspace_id: str) -> str:
    tenant = validate_workspace_id(workspace_id)
    sanitized = _UNSAFE_COMPONENT_RE.sub("_", tenant).strip("._-")
    if not sanitized:
        sanitized = "workspace"
    lowered = sanitized.lower()
    if lowered in _WINDOWS_RESERVED:
        sanitized = f"workspace_{sanitized}"
    digest = hashlib.sha256(tenant.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized[:72]}-{digest}"


def _safe_extension(path: Path) -> str:
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise StorageError(f"Unsupported file extension for storage: {extension!r}")
    return extension


class DocumentStorage:
    """Store source files outside the web root using their content hash."""

    def __init__(self, root: Path | str, max_upload_bytes: int):
        self.root = Path(root).expanduser().resolve()
        self.max_upload_bytes = max_upload_bytes

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def relative_path(
        self,
        workspace_id: str,
        content_hash: str,
        extension: str,
    ) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
            raise StorageError("content_hash must be a lowercase SHA-256 digest")
        extension = extension.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise StorageError(f"Unsupported file extension: {extension!r}")
        workspace = _safe_workspace_component(workspace_id)
        relative = Path(workspace) / content_hash[:2] / f"{content_hash}{extension}"
        return relative.as_posix()

    def resolve(self, relative_path: str) -> Path:
        """Resolve a stored relative path and reject traversal."""
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise StorageError("Stored paths must be relative")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise StorageError("Stored path escapes the storage root") from exc
        return resolved

    def store_file(
        self,
        source_path: Path | str,
        workspace_id: str,
        content_hash: str,
    ) -> StorageResult:
        source = Path(source_path)
        if not source.is_file():
            raise InvalidDocumentError(f"Source file does not exist: {source}")
        size = source.stat().st_size
        if size <= 0:
            raise InvalidDocumentError("Empty documents are not supported")
        if size > self.max_upload_bytes:
            raise InvalidDocumentError(
                f"Document is {size} bytes; limit is {self.max_upload_bytes} bytes"
            )

        extension = _safe_extension(source)
        relative = self.relative_path(workspace_id, content_hash, extension)
        destination = self.resolve(relative)

        if destination.exists():
            if destination.stat().st_size != size or sha256_file(destination) != content_hash:
                raise StorageConflictError(
                    f"Stored object collision or corruption at {relative}"
                )
            return StorageResult(relative, destination, False, size)

        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=".ingest-", suffix=".tmp", dir=str(destination.parent)
        )
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        try:
            with os.fdopen(fd, "wb") as output, source.open("rb") as input_handle:
                for block in iter(lambda: input_handle.read(1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())

            actual_hash = digest.hexdigest()
            if actual_hash != content_hash:
                raise StorageConflictError(
                    "Source file changed while it was being stored "
                    f"(expected {content_hash}, got {actual_hash})"
                )

            os.replace(temporary, destination)
            return StorageResult(relative, destination, True, size)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def delete(self, relative_path: str) -> bool:
        path = self.resolve(relative_path)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def read_bytes(self, relative_path: str) -> bytes:
        return self.resolve(relative_path).read_bytes()

    def exists(self, relative_path: str) -> bool:
        return self.resolve(relative_path).is_file()

    @staticmethod
    def _copy_stream(source: Path, destination: Path) -> Tuple[str, int]:
        """Compatibility helper retained for tests and tooling."""
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as input_handle, destination.open("wb") as output:
            for block in iter(lambda: input_handle.read(1024 * 1024), b""):
                digest.update(block)
                output.write(block)
                size += len(block)
        return digest.hexdigest(), size
