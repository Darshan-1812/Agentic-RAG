"""BAAI/bge-m3 dense embedding provider."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from retrieval.config import RetrievalSettings, get_settings
from retrieval.embeddings.base import (
    BaseEmbedder,
    EmbeddingInputError,
    EmbeddingModelError,
    EmbeddingUnavailableError,
    resolve_device,
)
from retrieval.model_cache import ModelResolutionError, resolve_model_source


class BgeM3Embedder(BaseEmbedder):
    """Lazy BGE-M3 embedder producing 1024-dimensional dense vectors."""

    def __init__(
        self,
        settings: Optional[RetrievalSettings] = None,
        model: object = None,
    ):
        self.settings = settings or get_settings()
        super().__init__(
            dimension=self.settings.embedding_dimension,
            normalize=self.settings.embedding_normalize,
        )
        self._model = model
        self._resolved = None

    @property
    def model_name(self) -> str:
        return self.settings.embedding_model_name

    @property
    def model_revision(self) -> str:
        if self._resolved is not None:
            return self._resolved.revision
        return self.settings.embedding_model_revision

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from FlagEmbedding import BGEM3FlagModel  # type: ignore
            except ImportError as exc:
                raise EmbeddingUnavailableError(
                    "FlagEmbedding is not installed. Install "
                    "retrieval/requirements-retrieval.txt before embedding."
                ) from exc

            try:
                resolved = resolve_model_source(
                    self.settings.embedding_model_name,
                    self.settings.embedding_model_revision,
                    cache_dir=self.settings.model_cache_dir,
                    local_files_only=self.settings.model_local_files_only,
                )
            except ModelResolutionError as exc:
                raise EmbeddingUnavailableError(str(exc)) from exc

            device = resolve_device(self.settings.embedding_device)
            use_fp16 = self.settings.embedding_fp16 and device.startswith("cuda")
            try:
                model = BGEM3FlagModel(
                    resolved.path,
                    normalize_embeddings=self.settings.embedding_normalize,
                    use_fp16=use_fp16,
                    devices=device,
                    batch_size=self.settings.embedding_batch_size,
                    query_max_length=self.settings.embedding_query_max_length,
                    passage_max_length=self.settings.embedding_max_length,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
            except Exception as exc:
                raise EmbeddingUnavailableError(
                    f"Unable to load BGE-M3 model {self.settings.embedding_model_name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            self._resolved = resolved
            self._model = model
            return model

    def _encode(
        self,
        texts: Sequence[str],
        *,
        max_length: int,
        batch_size: Optional[int],
    ) -> np.ndarray:
        model = self._ensure_model()
        actual_batch_size = batch_size or self.settings.embedding_batch_size
        if actual_batch_size < 1:
            raise EmbeddingInputError("batch_size must be >= 1")
        try:
            output = model.encode(
                list(texts),
                batch_size=actual_batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except Exception as exc:
            raise EmbeddingModelError(
                f"BGE-M3 inference failed: {type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(output, dict) or "dense_vecs" not in output:
            raise EmbeddingModelError(
                "BGE-M3 did not return dense_vecs; check the FlagEmbedding version"
            )
        return self._validate_vectors(output["dense_vecs"], expected_count=len(texts))

    def embed_documents_sync(
        self, texts: Sequence[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        prepared = self._validate_texts(texts, allow_empty=False)
        return self._encode(
            prepared,
            max_length=self.settings.embedding_max_length,
            batch_size=batch_size,
        )

    def embed_query_sync(self, text: str) -> np.ndarray:
        prepared = self._validate_texts([text], allow_empty=False)[0]
        vectors = self._encode(
            [prepared],
            max_length=self.settings.embedding_query_max_length,
            batch_size=self.settings.embedding_batch_size,
        )
        return vectors[0]

    def close(self) -> None:
        """Release the model reference; the next call reloads lazily."""
        self._model = None
