"""BAAI/bge-reranker-v2-m3 cross-encoder reranking."""
from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from typing import List, Optional, Sequence

import numpy as np

from retrieval.config import RetrievalSettings, get_settings
from retrieval.embeddings.base import resolve_device
from retrieval.model_cache import ModelResolutionError, resolve_model_source
from retrieval.retrieval_types import RetrievedChunk


class RerankerError(RuntimeError):
    """Raised when cross-encoder reranking cannot run."""


class RerankerUnavailableError(RerankerError):
    """Raised when the reranker runtime or model is unavailable."""


class BgeReranker:
    """Lazy BGE cross-encoder returning raw logits, never probabilities."""

    def __init__(
        self,
        settings: Optional[RetrievalSettings] = None,
        model: object = None,
    ):
        self.settings = settings or get_settings()
        self._model = model
        self._resolved = None
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self.settings.reranker_model_name

    @property
    def model_revision(self) -> str:
        if self._resolved is not None:
            return self._resolved.revision
        return self.settings.reranker_model_revision

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
                from FlagEmbedding import FlagReranker  # type: ignore
            except ImportError as exc:
                raise RerankerUnavailableError(
                    "FlagEmbedding is not installed. Install "
                    "retrieval/requirements-retrieval.txt before reranking."
                ) from exc

            try:
                resolved = resolve_model_source(
                    self.settings.reranker_model_name,
                    self.settings.reranker_model_revision,
                    cache_dir=self.settings.model_cache_dir,
                    local_files_only=self.settings.model_local_files_only,
                )
            except ModelResolutionError as exc:
                raise RerankerUnavailableError(str(exc)) from exc

            device = resolve_device(self.settings.reranker_device)
            use_fp16 = self.settings.reranker_fp16 and device.startswith("cuda")
            try:
                model = FlagReranker(
                    resolved.path,
                    use_fp16=use_fp16,
                    devices=device,
                    batch_size=self.settings.reranker_batch_size,
                    max_length=self.settings.reranker_max_length,
                    normalize=False,
                )
            except Exception as exc:
                raise RerankerUnavailableError(
                    f"Unable to load reranker {self.settings.reranker_model_name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            self._resolved = resolved
            self._model = model
            return model

    def score_pairs_sync(
        self,
        query: str,
        passages: Sequence[str],
        *,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise RerankerError("query must be a non-empty string")
        if isinstance(passages, (str, bytes)):
            raise RerankerError("passages must be a sequence of strings")
        items = list(passages)
        if not items:
            return np.empty((0,), dtype=np.float32)
        for index, passage in enumerate(items):
            if not isinstance(passage, str) or not passage.strip():
                raise RerankerError(f"passage at index {index} must not be empty")

        model = self._ensure_model()
        actual_batch_size = batch_size or self.settings.reranker_batch_size
        if actual_batch_size < 1:
            raise RerankerError("batch_size must be >= 1")
        pairs = [(query, passage) for passage in items]
        try:
            raw_scores = model.compute_score(
                pairs,
                batch_size=actual_batch_size,
                max_length=self.settings.reranker_max_length,
                normalize=False,
            )
        except Exception as exc:
            raise RerankerError(
                f"Cross-encoder inference failed: {type(exc).__name__}: {exc}"
            ) from exc

        scores = np.asarray(raw_scores, dtype=np.float32)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        elif scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores.reshape(-1)
        if scores.shape != (len(items),):
            raise RerankerError(
                f"expected {len(items)} raw logits, got shape {scores.shape}"
            )
        if not np.isfinite(scores).all():
            raise RerankerError("reranker returned NaN or infinite logits")
        return scores

    async def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        items = list(chunks)
        if not items:
            return []
        limit = top_k if top_k is not None else self.settings.retrieval_top_k
        if limit < 1:
            raise RerankerError("top_k must be >= 1")

        scores = await asyncio.to_thread(
            self.score_pairs_sync, query, [chunk.content for chunk in items]
        )
        scored = [
            replace(
                chunk,
                cross_encoder_logit=float(score),
            )
            for chunk, score in zip(items, scores)
        ]
        scored.sort(
            key=lambda chunk: (-float(chunk.cross_encoder_logit), chunk.chunk_id)
        )
        return [
            replace(chunk, rerank_rank=rank)
            for rank, chunk in enumerate(scored[:limit], start=1)
        ]

    def close(self) -> None:
        """Release the model reference; the next call reloads lazily."""
        self._model = None
