"""Base contracts and validation for dense embedding providers."""
from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np


class EmbeddingError(RuntimeError):
    """Base class for embedding failures."""


class EmbeddingUnavailableError(EmbeddingError):
    """Raised when the embedding runtime or model is unavailable."""


class EmbeddingModelError(EmbeddingError):
    """Raised when the model returns an unusable response."""


class EmbeddingInputError(EmbeddingError):
    """Raised when input text is empty or malformed."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when vector dimensions do not match the database schema."""


class EmbeddingValueError(EmbeddingError):
    """Raised when a vector contains non-finite or zero-magnitude values."""


def resolve_device(configured: str) -> str:
    """Resolve ``auto`` without importing torch until model construction."""
    value = configured.strip().lower()
    if value != "auto":
        if value == "cuda":
            return "cuda:0"
        return value
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - retrieval env includes torch
        raise EmbeddingUnavailableError(
            "torch is required when embedding_device='auto'; install "
            "retrieval/requirements-retrieval.txt"
        ) from exc
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class BaseEmbedder(ABC):
    """Async embedding interface with strict dimension and value validation."""

    def __init__(self, dimension: int, normalize: bool = True):
        if dimension < 1:
            raise EmbeddingDimensionError("embedding dimension must be >= 1")
        self._dimension = dimension
        self._normalize = normalize
        self._load_lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_revision(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def embed_documents_sync(
        self, texts: Sequence[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_query_sync(self, text: str) -> np.ndarray:
        raise NotImplementedError

    async def embed_documents(
        self, texts: Sequence[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        prepared = self._validate_texts(texts, allow_empty=False)
        return await asyncio.to_thread(
            self.embed_documents_sync, prepared, batch_size
        )

    async def embed_query(self, text: str) -> np.ndarray:
        prepared = self._validate_texts([text], allow_empty=False)[0]
        return await asyncio.to_thread(self.embed_query_sync, prepared)

    def validate_document_vectors(
        self, vectors: object, expected_count: Optional[int] = None
    ) -> np.ndarray:
        return self._validate_vectors(vectors, expected_count=expected_count)

    def validate_query_vector(self, vector: object) -> np.ndarray:
        matrix = self._validate_vectors(vector, expected_count=1)
        return matrix[0]

    def _validate_texts(
        self, texts: Sequence[str], *, allow_empty: bool
    ) -> list[str]:
        if isinstance(texts, (str, bytes)):
            raise EmbeddingInputError("texts must be a sequence of strings")
        prepared = list(texts)
        if not prepared and not allow_empty:
            raise EmbeddingInputError("at least one text is required")
        for index, value in enumerate(prepared):
            if not isinstance(value, str):
                raise EmbeddingInputError(f"text at index {index} must be a string")
            if not value.strip() and not allow_empty:
                raise EmbeddingInputError(f"text at index {index} must not be empty")
        return prepared

    def _validate_vectors(
        self, values: object, expected_count: Optional[int]
    ) -> np.ndarray:
        try:
            matrix = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise EmbeddingModelError(f"embedding output is not numeric: {exc}") from exc

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise EmbeddingDimensionError(
                f"embedding output must be 2-dimensional, got shape {matrix.shape}"
            )
        if matrix.shape[1] != self._dimension:
            raise EmbeddingDimensionError(
                f"expected {self._dimension}-dimensional embeddings, "
                f"got shape {matrix.shape}"
            )
        if expected_count is not None and matrix.shape[0] != expected_count:
            raise EmbeddingDimensionError(
                f"expected {expected_count} embeddings, got {matrix.shape[0]}"
            )
        if matrix.shape[0] == 0:
            raise EmbeddingModelError("embedding output is empty")
        if not np.isfinite(matrix).all():
            raise EmbeddingValueError("embedding output contains NaN or infinity")

        if self._normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms <= 1e-12):
                raise EmbeddingValueError("embedding output contains a zero vector")
            matrix = matrix / norms
        return np.ascontiguousarray(matrix, dtype=np.float32)
