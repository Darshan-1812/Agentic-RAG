"""Phase 4.3 Groq configuration tests."""
from __future__ import annotations

from typing import Any

from retrieval.config import RetrievalSettings
from retrieval.lightrag_indexer import LightRAGGraphEngine


def _clear_graph_credentials(monkeypatch: Any) -> None:
    for name in (
        "RETRIEVAL_GRAPH_LLM_API_KEY",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_graph_llm_api_key_prefers_retrieval_specific(monkeypatch: Any) -> None:
    _clear_graph_credentials(monkeypatch)
    monkeypatch.setenv("RETRIEVAL_GRAPH_LLM_API_KEY", "retrieval-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    settings = RetrievalSettings.from_env(env_file=False)

    assert settings.graph_llm_api_key == "retrieval-key"


def test_graph_llm_api_key_falls_back_to_groq(monkeypatch: Any) -> None:
    _clear_graph_credentials(monkeypatch)
    monkeypatch.setenv("RETRIEVAL_GRAPH_LLM_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    settings = RetrievalSettings.from_env(env_file=False)

    assert settings.graph_llm_api_key == "groq-key"


def test_engine_api_key_falls_back_to_groq_environment(monkeypatch: Any) -> None:
    _clear_graph_credentials(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")

    engine = LightRAGGraphEngine(
        settings=RetrievalSettings(graph_llm_api_key=None, graph_llm_base_url=None),
        workspace_id="tenant_a",
    )

    assert engine._llm_api_key() == "groq-key"


def test_graph_llm_api_key_falls_back_to_openai(monkeypatch: Any) -> None:
    _clear_graph_credentials(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    settings = RetrievalSettings.from_env(env_file=False)

    assert settings.graph_llm_api_key == "openai-key"


def test_graph_llm_api_key_is_none_without_credentials(monkeypatch: Any) -> None:
    _clear_graph_credentials(monkeypatch)

    settings = RetrievalSettings.from_env(env_file=False)

    assert settings.graph_llm_api_key is None
