"""Unit tests for Phase 3 BGE-M3 embedding integration."""
from __future__ import annotations

import numpy as np
import pytest

from retrieval.config import ConfigurationError, RetrievalSettings
from retrieval.embeddings import (
    BgeM3Embedder,
    EmbeddingDimensionError,
    EmbeddingInputError,
    EmbeddingValueError,
)


class _FakeModel:
    def __init__(self, builder):
        self.builder = builder
        self.calls = []

    def encode(self, sentences, **kwargs):
        self.calls.append({"sentences": list(sentences), **kwargs})
        return {"dense_vecs": self.builder(len(sentences))}


def _settings(**overrides):
    values = {"embedding_batch_size": 2}
    values.update(overrides)
    return RetrievalSettings(**values)


def _ramp(count):
    return np.tile(
        np.arange(1, 1025, dtype=np.float32),
        (count, 1),
    )


async def test_bge_m3_embeds_documents_and_query_with_configured_lengths():
    model = _FakeModel(_ramp)
    settings = _settings(embedding_max_length=4096, embedding_query_max_length=1024)
    embedder = BgeM3Embedder(settings, model=model)

    vectors = await embedder.embed_documents(["alpha", "beta"], batch_size=2)
    assert vectors.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert model.calls[0]["batch_size"] == 2
    assert model.calls[0]["max_length"] == 4096
    assert model.calls[0]["return_dense"] is True

    query = await embedder.embed_query("alpha")
    assert query.shape == (1024,)
    assert np.isclose(np.linalg.norm(query), 1.0)
    assert model.calls[1]["max_length"] == 1024


async def test_bge_m3_rejects_dimension_mismatch_and_zero_vectors():
    wrong_dimension = BgeM3Embedder(
        _settings(), model=_FakeModel(lambda count: np.ones((count, 3)))
    )
    with pytest.raises(EmbeddingDimensionError):
        await wrong_dimension.embed_documents(["alpha"])

    zero_vectors = BgeM3Embedder(
        _settings(), model=_FakeModel(lambda count: np.zeros((count, 1024)))
    )
    with pytest.raises(EmbeddingValueError):
        await zero_vectors.embed_query("alpha")


async def test_bge_m3_rejects_empty_inputs():
    embedder = BgeM3Embedder(_settings(), model=_FakeModel(_ramp))
    with pytest.raises(EmbeddingInputError):
        await embedder.embed_documents([])
    with pytest.raises(EmbeddingInputError):
        await embedder.embed_query("   ")


def test_embedding_configuration_is_validated(monkeypatch):
    with pytest.raises(ConfigurationError):
        RetrievalSettings(embedding_dimension=768)
    with pytest.raises(ConfigurationError):
        RetrievalSettings(rerank_candidate_depth=100, hybrid_candidate_depth=50)

    monkeypatch.setenv("RETRIEVAL_EMBEDDING_MODEL", "local/test-embedder")
    monkeypatch.setenv("RETRIEVAL_EMBEDDING_REVISION", "abc123")
    monkeypatch.setenv("RETRIEVAL_EMBEDDING_BATCH_SIZE", "4")
    monkeypatch.setenv("RETRIEVAL_RRF_K", "42")
    settings = RetrievalSettings.from_env(env_file=False)
    assert settings.embedding_model_name == "local/test-embedder"
    assert settings.embedding_model_revision == "abc123"
    assert settings.embedding_batch_size == 4
    assert settings.rrf_k == 42
