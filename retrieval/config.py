"""Retrieval-scoped configuration for Member 2.

This module deliberately avoids importing the shared application settings so that
the ingestion worker can be started without loading the FastAPI/Member 1/3 stack.
Configuration is read from environment variables and, optionally, a local
``.env`` file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlsplit, urlunsplit


DEFAULT_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/calibrag"
DEFAULT_STORAGE_ROOT = Path(__file__).resolve().parent / "storage"
DEFAULT_GRAPH_STORAGE_ROOT = DEFAULT_STORAGE_ROOT / "graph"

SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}


class ConfigurationError(ValueError):
    """Raised when retrieval configuration is invalid."""


def normalize_asyncpg_dsn(url: str) -> str:
    """Convert SQLAlchemy-style URLs to the URL accepted by ``asyncpg``.

    ``asyncpg`` understands ``postgresql://`` but not the SQLAlchemy driver
    suffix used in the shared ``.env.example``.
    """
    if not url or not url.strip():
        raise ConfigurationError("DATABASE_URL must not be empty")

    value = url.strip()
    if value.startswith("postgresql+asyncpg://"):
        value = "postgresql://" + value[len("postgresql+asyncpg://") :]
    elif value.startswith("postgres+asyncpg://"):
        value = "postgresql://" + value[len("postgres+asyncpg://") :]
    elif value.startswith("postgres://"):
        value = "postgresql://" + value[len("postgres://") :]

    parsed = urlsplit(value)
    if parsed.scheme not in {"postgresql"}:
        raise ConfigurationError(
            "DATABASE_URL must use postgresql:// or postgresql+asyncpg://"
        )
    if not parsed.hostname or not parsed.path or parsed.path == "/":
        raise ConfigurationError("DATABASE_URL must include a host and database name")
    return urlunsplit(parsed)


def _as_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean value, got {value!r}")


def _as_int(value: str, *, name: str, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _as_optional_path(value: Optional[str]) -> Optional[Path]:
    if value is None or not value.strip():
        return None
    return Path(value.strip()).expanduser()


def _as_device(value: str, *, name: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu", "mps"}:
        return normalized
    if normalized == "cuda" or normalized.startswith("cuda:"):
        suffix = normalized[len("cuda") :]
        if not suffix or (suffix.startswith(":") and suffix[1:].isdigit()):
            return normalized
    raise ConfigurationError(
        f"{name} must be 'auto', 'cpu', 'mps', 'cuda', or 'cuda:<index>', got {value!r}"
    )


def _load_env_file(path: Path) -> None:
    """Load a small, dependency-free subset of dotenv syntax.

    Existing process environment variables always win. Only ``KEY=VALUE`` lines,
    optional ``export`` prefixes, blank lines, and ``#`` comments are supported.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigurationError(f"Unable to read env file {path}: {exc}") from exc

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class RetrievalSettings:
    """Settings used by the database and ingestion layers."""

    database_url: str = field(default=DEFAULT_DATABASE_URL, repr=False)
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10
    db_command_timeout: float = 30.0
    db_statement_cache_size: int = 100

    storage_root: Path = field(default=DEFAULT_STORAGE_ROOT)
    max_upload_bytes: int = 100 * 1024 * 1024

    graph_storage_root: Path = field(default=DEFAULT_GRAPH_STORAGE_ROOT)
    graph_kv_storage: str = "JsonKVStorage"
    graph_vector_storage: str = "NanoVectorDBStorage"
    graph_graph_storage: str = "NetworkXStorage"
    graph_doc_status_storage: str = "JsonDocStatusStorage"
    graph_llm_model: str = "gpt-4o-mini"
    graph_llm_base_url: Optional[str] = None
    graph_llm_api_key: Optional[str] = field(default=None, repr=False)
    graph_llm_timeout: int = 60
    graph_llm_max_async: int = 4
    graph_max_nodes: int = 1000
    graph_traversal_max_nodes: int = 200
    graph_traversal_timeout_seconds: float = 120.0

    chunk_target_chars: int = 1800
    chunk_max_chars: int = 2600
    chunk_overlap_chars: int = 200

    docling_ocr: bool = True
    docling_include_page_furniture: bool = False

    # Phase 3: BGE-M3 dense embeddings.
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_model_revision: str = "main"
    embedding_dimension: int = 1024
    embedding_device: str = "auto"
    embedding_batch_size: int = 16
    embedding_max_length: int = 8192
    embedding_query_max_length: int = 8192
    embedding_fp16: bool = True
    embedding_normalize: bool = True
    model_cache_dir: Optional[Path] = None
    model_local_files_only: bool = False

    # Phase 3: dense, lexical, fusion, and reranking candidate sizes.
    retrieval_top_k: int = 5
    vector_candidate_depth: int = 50
    hnsw_ef_search: int = 100
    bm25_candidate_depth: int = 50
    rrf_k: int = 60
    hybrid_candidate_depth: int = 50
    rerank_candidate_depth: int = 20

    # Phase 3: BGE cross-encoder reranker.
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_model_revision: str = "main"
    reranker_device: str = "auto"
    reranker_batch_size: int = 16
    reranker_max_length: int = 1024
    reranker_fp16: bool = True

    def __post_init__(self) -> None:
        normalize_asyncpg_dsn(self.database_url)
        if self.db_pool_min_size < 0:
            raise ConfigurationError("db_pool_min_size must be >= 0")
        if self.db_pool_max_size < 1:
            raise ConfigurationError("db_pool_max_size must be >= 1")
        if self.db_pool_min_size > self.db_pool_max_size:
            raise ConfigurationError("db_pool_min_size cannot exceed db_pool_max_size")
        if self.db_command_timeout <= 0:
            raise ConfigurationError("db_command_timeout must be > 0")
        if self.max_upload_bytes <= 0:
            raise ConfigurationError("max_upload_bytes must be > 0")

        for name, value in (
            ("graph_kv_storage", self.graph_kv_storage),
            ("graph_vector_storage", self.graph_vector_storage),
            ("graph_graph_storage", self.graph_graph_storage),
            ("graph_doc_status_storage", self.graph_doc_status_storage),
            ("graph_llm_model", self.graph_llm_model),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{name} must not be empty")
        for name, value in (
            ("graph_llm_timeout", self.graph_llm_timeout),
            ("graph_llm_max_async", self.graph_llm_max_async),
            ("graph_max_nodes", self.graph_max_nodes),
            ("graph_traversal_max_nodes", self.graph_traversal_max_nodes),
        ):
            if value < 1:
                raise ConfigurationError(f"{name} must be >= 1")
        if self.graph_traversal_max_nodes > self.graph_max_nodes:
            raise ConfigurationError(
                "graph_traversal_max_nodes cannot exceed graph_max_nodes"
            )
        if self.graph_traversal_timeout_seconds <= 0:
            raise ConfigurationError("graph_traversal_timeout_seconds must be > 0")
        if self.graph_llm_base_url is not None and not self.graph_llm_base_url.strip():
            object.__setattr__(self, "graph_llm_base_url", None)
        if self.graph_llm_api_key is not None and not self.graph_llm_api_key.strip():
            object.__setattr__(self, "graph_llm_api_key", None)

        if self.chunk_max_chars < 1:
            raise ConfigurationError("chunk_max_chars must be >= 1")
        if self.chunk_target_chars < 1:
            raise ConfigurationError("chunk_target_chars must be >= 1")
        if self.chunk_target_chars > self.chunk_max_chars:
            raise ConfigurationError("chunk_target_chars cannot exceed chunk_max_chars")
        if self.chunk_overlap_chars < 0:
            raise ConfigurationError("chunk_overlap_chars must be >= 0")
        if self.chunk_overlap_chars >= self.chunk_target_chars:
            raise ConfigurationError("chunk_overlap_chars must be smaller than chunk_target_chars")

        if not self.embedding_model_name.strip():
            raise ConfigurationError("embedding_model_name must not be empty")
        if not self.embedding_model_revision.strip():
            raise ConfigurationError("embedding_model_revision must not be empty")
        if self.embedding_dimension != 1024:
            raise ConfigurationError(
                "embedding_dimension must be 1024 to match the chunk_embeddings schema"
            )
        _as_device(self.embedding_device, name="embedding_device")
        if self.embedding_batch_size < 1:
            raise ConfigurationError("embedding_batch_size must be >= 1")
        if self.embedding_max_length < 8:
            raise ConfigurationError("embedding_max_length must be >= 8")
        if self.embedding_query_max_length < 8:
            raise ConfigurationError("embedding_query_max_length must be >= 8")

        if not self.reranker_model_name.strip():
            raise ConfigurationError("reranker_model_name must not be empty")
        if not self.reranker_model_revision.strip():
            raise ConfigurationError("reranker_model_revision must not be empty")
        _as_device(self.reranker_device, name="reranker_device")
        if self.reranker_batch_size < 1:
            raise ConfigurationError("reranker_batch_size must be >= 1")
        if self.reranker_max_length < 8:
            raise ConfigurationError("reranker_max_length must be >= 8")

        for name, value in (
            ("retrieval_top_k", self.retrieval_top_k),
            ("vector_candidate_depth", self.vector_candidate_depth),
            ("hnsw_ef_search", self.hnsw_ef_search),
            ("bm25_candidate_depth", self.bm25_candidate_depth),
            ("rrf_k", self.rrf_k),
            ("hybrid_candidate_depth", self.hybrid_candidate_depth),
            ("rerank_candidate_depth", self.rerank_candidate_depth),
        ):
            if value < 1:
                raise ConfigurationError(f"{name} must be >= 1")
        if self.vector_candidate_depth < self.retrieval_top_k:
            raise ConfigurationError(
                "vector_candidate_depth must be >= retrieval_top_k"
            )
        if self.bm25_candidate_depth < self.retrieval_top_k:
            raise ConfigurationError(
                "bm25_candidate_depth must be >= retrieval_top_k"
            )
        if self.hybrid_candidate_depth < self.retrieval_top_k:
            raise ConfigurationError(
                "hybrid_candidate_depth must be >= retrieval_top_k"
            )
        if self.rerank_candidate_depth > self.hybrid_candidate_depth:
            raise ConfigurationError(
                "rerank_candidate_depth cannot exceed hybrid_candidate_depth"
            )
        if self.rerank_candidate_depth < self.retrieval_top_k:
            raise ConfigurationError(
                "rerank_candidate_depth must be >= retrieval_top_k"
            )
        if self.model_cache_dir is not None:
            object.__setattr__(self, "model_cache_dir", self.model_cache_dir.expanduser())

    @property
    def database_dsn(self) -> str:
        """Return the DSN in the format accepted by ``asyncpg``."""
        return normalize_asyncpg_dsn(self.database_url)

    @property
    def resolved_storage_root(self) -> Path:
        return self.storage_root.expanduser().resolve()

    @property
    def resolved_graph_storage_root(self) -> Path:
        return self.graph_storage_root.expanduser().resolve()

    def ensure_storage_root(self) -> Path:
        root = self.resolved_storage_root
        root.mkdir(parents=True, exist_ok=True)
        return root

    def ensure_graph_storage_root(self) -> Path:
        root = self.resolved_graph_storage_root
        root.mkdir(parents=True, exist_ok=True)
        return root

    @classmethod
    def from_env(
        cls,
        env_file: Optional[Path | str] = None,
        overrides: Optional[Dict[str, object]] = None,
    ) -> "RetrievalSettings":
        """Build settings from environment variables.

        ``env_file`` defaults to ``.env`` in the current working directory when
        that file exists. Pass an explicit path to use a different file, or
        ``False`` to skip dotenv loading.
        """
        if env_file is None:
            candidate = Path.cwd() / ".env"
            if candidate.is_file():
                _load_env_file(candidate)
        elif env_file is not False:
            path = Path(env_file)
            if not path.is_absolute():
                path = Path.cwd() / path
            _load_env_file(path)

        values: Dict[str, object] = {
            "database_url": os.getenv("RETRIEVAL_DATABASE_URL")
            or os.getenv("DATABASE_URL")
            or DEFAULT_DATABASE_URL,
            "db_pool_min_size": _as_int(
                os.getenv("RETRIEVAL_DB_POOL_MIN_SIZE", "1"),
                name="RETRIEVAL_DB_POOL_MIN_SIZE",
            ),
            "db_pool_max_size": _as_int(
                os.getenv("RETRIEVAL_DB_POOL_MAX_SIZE", "10"),
                name="RETRIEVAL_DB_POOL_MAX_SIZE",
                minimum=1,
            ),
            "db_command_timeout": float(os.getenv("RETRIEVAL_DB_COMMAND_TIMEOUT", "30")),
            "db_statement_cache_size": _as_int(
                os.getenv("RETRIEVAL_DB_STATEMENT_CACHE_SIZE", "100"),
                name="RETRIEVAL_DB_STATEMENT_CACHE_SIZE",
            ),
            "storage_root": Path(
                os.getenv("RETRIEVAL_STORAGE_ROOT", str(DEFAULT_STORAGE_ROOT))
            ),
            "max_upload_bytes": _as_int(
                os.getenv("RETRIEVAL_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)),
                name="RETRIEVAL_MAX_UPLOAD_BYTES",
                minimum=1,
            ),
            "graph_storage_root": Path(
                os.getenv(
                    "RETRIEVAL_GRAPH_STORAGE_ROOT", str(DEFAULT_GRAPH_STORAGE_ROOT)
                )
            ),
            "graph_kv_storage": os.getenv(
                "RETRIEVAL_GRAPH_KV_STORAGE", "JsonKVStorage"
            ),
            "graph_vector_storage": os.getenv(
                "RETRIEVAL_GRAPH_VECTOR_STORAGE", "NanoVectorDBStorage"
            ),
            "graph_graph_storage": os.getenv(
                "RETRIEVAL_GRAPH_GRAPH_STORAGE", "NetworkXStorage"
            ),
            "graph_doc_status_storage": os.getenv(
                "RETRIEVAL_GRAPH_DOC_STATUS_STORAGE", "JsonDocStatusStorage"
            ),
            "graph_llm_model": os.getenv("RETRIEVAL_GRAPH_LLM_MODEL")
            or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "graph_llm_base_url": os.getenv("RETRIEVAL_GRAPH_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL"),
            "graph_llm_api_key": os.getenv("RETRIEVAL_GRAPH_LLM_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
            "graph_llm_timeout": _as_int(
                os.getenv("RETRIEVAL_GRAPH_LLM_TIMEOUT", "60"),
                name="RETRIEVAL_GRAPH_LLM_TIMEOUT",
                minimum=1,
            ),
            "graph_llm_max_async": _as_int(
                os.getenv("RETRIEVAL_GRAPH_LLM_MAX_ASYNC", "4"),
                name="RETRIEVAL_GRAPH_LLM_MAX_ASYNC",
                minimum=1,
            ),
            "graph_max_nodes": _as_int(
                os.getenv("RETRIEVAL_GRAPH_MAX_NODES", "1000"),
                name="RETRIEVAL_GRAPH_MAX_NODES",
                minimum=1,
            ),
            "graph_traversal_max_nodes": _as_int(
                os.getenv("RETRIEVAL_GRAPH_TRAVERSAL_MAX_NODES", "200"),
                name="RETRIEVAL_GRAPH_TRAVERSAL_MAX_NODES",
                minimum=1,
            ),
            "graph_traversal_timeout_seconds": float(
                os.getenv("RETRIEVAL_GRAPH_TRAVERSAL_TIMEOUT", "120")
            ),
            "chunk_target_chars": _as_int(
                os.getenv("RETRIEVAL_CHUNK_TARGET_CHARS", "1800"),
                name="RETRIEVAL_CHUNK_TARGET_CHARS",
                minimum=1,
            ),
            "chunk_max_chars": _as_int(
                os.getenv("RETRIEVAL_CHUNK_MAX_CHARS", "2600"),
                name="RETRIEVAL_CHUNK_MAX_CHARS",
                minimum=1,
            ),
            "chunk_overlap_chars": _as_int(
                os.getenv("RETRIEVAL_CHUNK_OVERLAP_CHARS", "200"),
                name="RETRIEVAL_CHUNK_OVERLAP_CHARS",
            ),
            "docling_ocr": _as_bool(
                os.getenv("RETRIEVAL_DOCLING_OCR", "true"), name="RETRIEVAL_DOCLING_OCR"
            ),
            "docling_include_page_furniture": _as_bool(
                os.getenv("RETRIEVAL_DOCLING_INCLUDE_PAGE_FURNITURE", "false"),
                name="RETRIEVAL_DOCLING_INCLUDE_PAGE_FURNITURE",
            ),
            "embedding_model_name": os.getenv(
                "RETRIEVAL_EMBEDDING_MODEL", "BAAI/bge-m3"
            ),
            "embedding_model_revision": os.getenv(
                "RETRIEVAL_EMBEDDING_REVISION", "main"
            ),
            "embedding_dimension": _as_int(
                os.getenv("RETRIEVAL_EMBEDDING_DIMENSION", "1024"),
                name="RETRIEVAL_EMBEDDING_DIMENSION",
                minimum=1,
            ),
            "embedding_device": _as_device(
                os.getenv("RETRIEVAL_EMBEDDING_DEVICE", "auto"),
                name="RETRIEVAL_EMBEDDING_DEVICE",
            ),
            "embedding_batch_size": _as_int(
                os.getenv("RETRIEVAL_EMBEDDING_BATCH_SIZE", "16"),
                name="RETRIEVAL_EMBEDDING_BATCH_SIZE",
                minimum=1,
            ),
            "embedding_max_length": _as_int(
                os.getenv("RETRIEVAL_EMBEDDING_MAX_LENGTH", "8192"),
                name="RETRIEVAL_EMBEDDING_MAX_LENGTH",
                minimum=8,
            ),
            "embedding_query_max_length": _as_int(
                os.getenv("RETRIEVAL_EMBEDDING_QUERY_MAX_LENGTH", "8192"),
                name="RETRIEVAL_EMBEDDING_QUERY_MAX_LENGTH",
                minimum=8,
            ),
            "embedding_fp16": _as_bool(
                os.getenv("RETRIEVAL_EMBEDDING_FP16", "true"),
                name="RETRIEVAL_EMBEDDING_FP16",
            ),
            "embedding_normalize": _as_bool(
                os.getenv("RETRIEVAL_EMBEDDING_NORMALIZE", "true"),
                name="RETRIEVAL_EMBEDDING_NORMALIZE",
            ),
            "model_cache_dir": _as_optional_path(
                os.getenv("RETRIEVAL_MODEL_CACHE_DIR")
            ),
            "model_local_files_only": _as_bool(
                os.getenv("RETRIEVAL_MODEL_LOCAL_FILES_ONLY", "false"),
                name="RETRIEVAL_MODEL_LOCAL_FILES_ONLY",
            ),
            "retrieval_top_k": _as_int(
                os.getenv("RETRIEVAL_TOP_K", "5"),
                name="RETRIEVAL_TOP_K",
                minimum=1,
            ),
            "vector_candidate_depth": _as_int(
                os.getenv("RETRIEVAL_VECTOR_CANDIDATE_DEPTH", "50"),
                name="RETRIEVAL_VECTOR_CANDIDATE_DEPTH",
                minimum=1,
            ),
            "hnsw_ef_search": _as_int(
                os.getenv("RETRIEVAL_HNSW_EF_SEARCH", "100"),
                name="RETRIEVAL_HNSW_EF_SEARCH",
                minimum=1,
            ),
            "bm25_candidate_depth": _as_int(
                os.getenv("RETRIEVAL_BM25_CANDIDATE_DEPTH", "50"),
                name="RETRIEVAL_BM25_CANDIDATE_DEPTH",
                minimum=1,
            ),
            "rrf_k": _as_int(
                os.getenv("RETRIEVAL_RRF_K", "60"),
                name="RETRIEVAL_RRF_K",
                minimum=1,
            ),
            "hybrid_candidate_depth": _as_int(
                os.getenv("RETRIEVAL_HYBRID_CANDIDATE_DEPTH", "50"),
                name="RETRIEVAL_HYBRID_CANDIDATE_DEPTH",
                minimum=1,
            ),
            "rerank_candidate_depth": _as_int(
                os.getenv("RETRIEVAL_RERANK_CANDIDATE_DEPTH", "20"),
                name="RETRIEVAL_RERANK_CANDIDATE_DEPTH",
                minimum=1,
            ),
            "reranker_model_name": os.getenv(
                "RETRIEVAL_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
            ),
            "reranker_model_revision": os.getenv(
                "RETRIEVAL_RERANKER_REVISION", "main"
            ),
            "reranker_device": _as_device(
                os.getenv("RETRIEVAL_RERANKER_DEVICE", "auto"),
                name="RETRIEVAL_RERANKER_DEVICE",
            ),
            "reranker_batch_size": _as_int(
                os.getenv("RETRIEVAL_RERANKER_BATCH_SIZE", "16"),
                name="RETRIEVAL_RERANKER_BATCH_SIZE",
                minimum=1,
            ),
            "reranker_max_length": _as_int(
                os.getenv("RETRIEVAL_RERANKER_MAX_LENGTH", "1024"),
                name="RETRIEVAL_RERANKER_MAX_LENGTH",
                minimum=8,
            ),
            "reranker_fp16": _as_bool(
                os.getenv("RETRIEVAL_RERANKER_FP16", "true"),
                name="RETRIEVAL_RERANKER_FP16",
            ),
        }

        if overrides:
            valid_fields = {item.name for item in fields(cls)}
            unknown = set(overrides) - valid_fields
            if unknown:
                raise ConfigurationError(
                    "Unknown retrieval settings: " + ", ".join(sorted(unknown))
                )
            values.update(overrides)

        return cls(**values)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def get_settings() -> RetrievalSettings:
    """Return process-wide cached retrieval settings."""
    return RetrievalSettings.from_env()
