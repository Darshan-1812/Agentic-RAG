"""Member 2 document ingestion package."""
from retrieval.ingestion.chunker import chunk_blocks
from retrieval.ingestion.docling_parser import (
    DoclingParser,
    DoclingUnavailableError,
    DocumentParseError,
)
from retrieval.ingestion.metadata import (
    DocumentTooLargeError,
    InvalidDocumentError,
    UnsupportedDocumentError,
    build_document_metadata,
    detect_mime_type,
    sha256_file,
    validate_source_file,
)
from retrieval.ingestion.normalizer import (
    normalize_heading,
    normalize_table_markdown,
    normalize_text,
    split_paragraphs,
)
from retrieval.ingestion.pipeline import IngestionError, IngestionPipeline
from retrieval.ingestion.storage import (
    DocumentStorage,
    StorageConflictError,
    StorageError,
    StorageResult,
)

__all__ = [
    "chunk_blocks",
    "DoclingParser",
    "DoclingUnavailableError",
    "DocumentParseError",
    "DocumentTooLargeError",
    "InvalidDocumentError",
    "UnsupportedDocumentError",
    "build_document_metadata",
    "detect_mime_type",
    "sha256_file",
    "validate_source_file",
    "normalize_heading",
    "normalize_table_markdown",
    "normalize_text",
    "split_paragraphs",
    "IngestionError",
    "IngestionPipeline",
    "DocumentStorage",
    "StorageConflictError",
    "StorageError",
    "StorageResult",
]
