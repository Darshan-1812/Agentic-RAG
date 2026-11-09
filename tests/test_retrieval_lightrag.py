"""Phase 4 tests for the real LightRAG graph adapter."""
from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from retrieval.config import RetrievalSettings
from retrieval.lightrag_indexer import (
    GraphConfigurationError,
    GraphWorkspaceError,
    LightRAGGraphEngine,
)


class _KGNode:
    def __init__(self, node_id: str, properties: Dict[str, Any]):
        self.id = node_id
        self.labels = [node_id]
        self.properties = properties


class _KGEdge:
    def __init__(self, source: str, target: str, properties: Dict[str, Any]):
        self.id = f"{source}-{target}"
        self.type = "DIRECTED"
        self.source = source
        self.target = target
        self.properties = properties


class _KnowledgeGraph:
    def __init__(
        self,
        nodes: Sequence[_KGNode],
        edges: Sequence[_KGEdge],
        is_truncated: bool = False,
    ):
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.is_truncated = is_truncated


class _FakeLightRAG:
    def __init__(self, nodes: Dict[str, Dict[str, Any]], edges: Sequence[Tuple[str, str, Dict[str, Any]]]):
        self.nodes = nodes
        self.edges = list(edges)
        self.adjacency: Dict[str, set[str]] = {node_id: set() for node_id in nodes}
        for source, target, _ in edges:
            self.adjacency.setdefault(source, set()).add(target)
            self.adjacency.setdefault(target, set()).add(source)
        self.initialized = 0
        self.finalized = 0
        self.inserted: List[Dict[str, Any]] = []
        self.query_data_calls: List[Dict[str, Any]] = []
        self.query_llm_calls: List[Dict[str, Any]] = []
        self.query_calls: List[Dict[str, Any]] = []

    async def initialize_storages(self) -> None:
        self.initialized += 1

    async def finalize_storages(self) -> None:
        self.finalized += 1

    async def get_graph_labels(self) -> List[str]:
        return sorted(self.nodes)

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: Optional[int] = None,
    ) -> _KnowledgeGraph:
        if node_label not in self.nodes:
            return _KnowledgeGraph([], [])

        limit = max_nodes or 1000
        distances = {node_label: 0}
        queue = deque([node_label])
        while queue:
            current = queue.popleft()
            if distances[current] >= max_depth:
                continue
            for neighbor in sorted(self.adjacency.get(current, ())):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

        selected = sorted(distances, key=lambda item: (distances[item], item.lower()))[:limit]
        selected_set = set(selected)
        nodes = [
            _KGNode(node_id, dict(self.nodes[node_id]))
            for node_id in selected
        ]
        edges = [
            _KGEdge(source, target, dict(properties))
            for source, target, properties in self.edges
            if source in selected_set and target in selected_set
        ]
        return _KnowledgeGraph(nodes, edges, is_truncated=len(distances) > limit)

    async def ainsert(
        self,
        input: Any,
        split_by_character: Optional[str] = None,
        split_by_character_only: bool = False,
        ids: Any = None,
        file_paths: Any = None,
        track_id: Optional[str] = None,
    ) -> str:
        self.inserted.append(
            {
                "input": input,
                "ids": ids,
                "file_paths": file_paths,
                "track_id": track_id,
            }
        )
        return track_id or "track-test"

    @staticmethod
    def _query_data(param: Any) -> Dict[str, Any]:
        return {
            "status": "success",
            "message": "Query executed successfully",
            "data": {
                "entities": [
                    {
                        "entity_name": "VoltCore India",
                        "entity_type": "Supplier",
                        "description": "supplies modules",
                        "source_id": "chunk-1",
                        "file_path": "supplier.pdf",
                        "reference_id": "1",
                    }
                ],
                "relationships": [
                    {
                        "src_id": "VoltCore India",
                        "tgt_id": "BMS-Controller-v4",
                        "description": "supplies modules",
                        "keywords": "supply, modules",
                        "weight": 1.0,
                        "source_id": "chunk-1",
                        "file_path": "supplier.pdf",
                        "reference_id": "1",
                    }
                ],
                "chunks": [
                    {
                        "content": "VoltCore India supplies BMS-Controller-v4.",
                        "file_path": "supplier.pdf",
                        "chunk_id": "chunk-1",
                        "reference_id": "1",
                    }
                ],
                "references": [
                    {"reference_id": "1", "file_path": "supplier.pdf"}
                ],
            },
            "metadata": {
                "query_mode": param.mode,
                "keywords": {
                    "high_level": list(param.hl_keywords),
                    "low_level": list(param.ll_keywords),
                },
            },
        }

    async def aquery_data(self, query: str, param: Any) -> Dict[str, Any]:
        self.query_data_calls.append({"query": query, "param": param})
        return self._query_data(param)

    async def aquery_llm(
        self,
        query: str,
        param: Any,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.query_llm_calls.append(
            {"query": query, "param": param, "system_prompt": system_prompt}
        )
        result = self._query_data(param)
        result["llm_response"] = {
            "content": f"answer: {query}",
            "response_iterator": None,
            "is_streaming": False,
            "llm_generated": True,
        }
        return result

    async def aquery(
        self,
        query: str,
        param: Any,
        system_prompt: Optional[str] = None,
    ) -> str:
        self.query_calls.append(
            {"query": query, "param": param, "system_prompt": system_prompt}
        )
        return f"answer: {query}"


def _graph() -> _FakeLightRAG:
    return _FakeLightRAG(
        nodes={
            "VoltCore India": {"entity_type": "Supplier", "entity_name": "VoltCore India"},
            "BMS-Controller-v4": {"entity_type": "BatteryModule", "entity_name": "BMS-Controller-v4"},
            "R. Kulkarni": {"entity_type": "Engineer", "entity_name": "R. Kulkarni"},
            "EV-Pack-V2": {"entity_type": "BatteryModule", "entity_name": "EV-Pack-V2"},
        },
        edges=[
            (
                "VoltCore India",
                "BMS-Controller-v4",
                {"description": "supplies modules", "file_path": "supplier.pdf"},
            ),
            (
                "BMS-Controller-v4",
                "R. Kulkarni",
                {"description": "lead architect"},
            ),
            (
                "EV-Pack-V2",
                "VoltCore India",
                {"description": "uses cells"},
            ),
        ],
    )


def _settings(tmp_path: Any, **overrides: Any) -> RetrievalSettings:
    values = {
        "graph_storage_root": tmp_path,
        "graph_llm_api_key": "test-key",
        "graph_llm_base_url": "http://localhost:9999/v1",
    }
    values.update(overrides)
    return RetrievalSettings(**values)


def test_lightrag_traversal_uses_real_client_shape(tmp_path: Any) -> None:
    fake = _graph()
    engine = LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id="tenant_a",
        lightrag_factory=lambda _: fake,
    )
    try:
        result = engine.traverse_entities(
            query="Who is the lead architect for VoltCore modules?",
            entity_seeds=["VoltCore"],
            max_hops=2,
        )
    finally:
        engine.close()

    assert result["workspace_id"] == "tenant_a"
    assert result["total_nodes_found"] == 4
    names = [node["entity_name"] for node in result["nodes"]]
    assert names == [
        "VoltCore India",
        "BMS-Controller-v4",
        "EV-Pack-V2",
        "R. Kulkarni",
    ]
    assert "Graph Link:" in result["context_summary"]
    assert "supplies modules" in result["context_summary"]
    assert fake.initialized == 1
    assert fake.finalized == 1
    assert result["edges"][0]["relation_type"] == "RELATED_TO"


def test_lightrag_traversal_from_running_loop(tmp_path: Any) -> None:
    fake = _graph()
    engine = LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id="tenant_a",
        lightrag_factory=lambda _: fake,
    )

    async def scenario() -> Dict[str, Any]:
        return engine.traverse_entities(
            query="Find BMS owner",
            entity_seeds=["BMS-Controller-v4"],
            max_hops=1,
        )

    try:
        result = asyncio.run(scenario())
    finally:
        engine.close()

    assert result["total_nodes_found"] == 3
    assert "R. Kulkarni" in [node["entity_name"] for node in result["nodes"]]


def test_lightrag_without_index_is_empty(tmp_path: Any) -> None:
    engine = LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id="empty_workspace",
    )
    result = engine.traverse_entities(
        query="anything",
        entity_seeds=["VoltCore"],
        max_hops=2,
    )
    engine.close()

    assert result["total_nodes_found"] == 0
    assert result["nodes"] == []
    assert result["edges"] == []
    assert result["context_summary"] == ""


def test_lightrag_index_chunks_uses_lightrag(tmp_path: Any) -> None:
    fake = _graph()
    engine = LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id="tenant_a",
        lightrag_factory=lambda _: fake,
    )

    chunks = [
        {
            "chunk_id": "chunk-1",
            "doc_id": "doc-1",
            "doc_title": "Supplier report",
            "workspace_id": "tenant_a",
            "content": "VoltCore India supplies BMS-Controller-v4.",
            "metadata": {"file_path": "supplier.pdf"},
        },
        {
            "chunk_id": "chunk-2",
            "doc_id": "doc-1",
            "doc_title": "Supplier report",
            "workspace_id": "tenant_a",
            "content": "R. Kulkarni is the lead architect.",
            "metadata": {},
        },
    ]

    async def scenario() -> Dict[str, Any]:
        return await engine.aindex_chunks(chunks, track_id="track-42")

    try:
        result = asyncio.run(scenario())
    finally:
        engine.close()

    assert result["indexed_documents"] == 2
    assert result["track_id"] == "track-42"
    assert fake.inserted[0]["ids"] == ["chunk-1", "chunk-2"]
    assert fake.inserted[0]["file_paths"] == [
        "supplier.pdf::chunk::chunk-1",
        "Supplier report::chunk::chunk-2",
    ]


def test_lightrag_index_requires_llm_config(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.delenv("RETRIEVAL_GRAPH_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    engine = LightRAGGraphEngine(
        settings=_settings(
            tmp_path,
            graph_llm_api_key=None,
            graph_llm_base_url=None,
        ),
        workspace_id="tenant_a",
        lightrag_factory=lambda _: _graph(),
    )

    with pytest.raises(GraphConfigurationError):
        asyncio.run(engine.aindex_text("VoltCore supplies modules."))

    engine.close()


def test_lightrag_index_requires_api_key_with_base_url(
    tmp_path: Any, monkeypatch: Any
) -> None:
    monkeypatch.delenv("RETRIEVAL_GRAPH_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9999/v1")
    engine = LightRAGGraphEngine(
        settings=_settings(
            tmp_path,
            graph_llm_api_key=None,
            graph_llm_base_url=None,
        ),
        workspace_id="tenant_a",
        lightrag_factory=lambda _: _graph(),
    )

    with pytest.raises(GraphConfigurationError):
        asyncio.run(engine.aindex_text("VoltCore supplies modules."))

    engine.close()


def test_lightrag_rejects_cross_workspace_chunks(tmp_path: Any) -> None:
    engine = LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id="tenant_a",
        lightrag_factory=lambda _: _graph(),
    )
    chunks = [
        {
            "chunk_id": "chunk-1",
            "doc_id": "doc-1",
            "workspace_id": "tenant_b",
            "content": "cross-workspace content",
        }
    ]

    with pytest.raises(GraphWorkspaceError):
        asyncio.run(engine.aindex_chunks(chunks))

    engine.close()


def _query_param_factory(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def test_lightrag_query_data_uses_query_param(tmp_path: Any) -> None:
    fake = _graph()
    engine = LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id="tenant_a",
        lightrag_factory=lambda _: fake,
        query_param_factory=_query_param_factory,
    )

    async def scenario() -> Dict[str, Any]:
        return await engine.aquery_data(
            "Who supplies the BMS controller?",
            mode="hybrid",
            top_k=7,
            chunk_top_k=3,
            hl_keywords=["supply"],
            ll_keywords=["VoltCore India"],
        )

    try:
        result = asyncio.run(scenario())
    finally:
        engine.close()

    assert result["status"] == "success"
    assert result["workspace_id"] == "tenant_a"
    assert result["mode"] == "hybrid"
    assert result["data"]["entities"][0]["entity_name"] == "VoltCore India"
    assert result["data"]["relationships"][0]["src_id"] == "VoltCore India"
    assert result["data"]["chunks"][0]["chunk_id"] == "chunk-1"
    param = fake.query_data_calls[0]["param"]
    assert param.mode == "hybrid"
    assert param.top_k == 7
    assert param.chunk_top_k == 3
    assert param.hl_keywords == ["supply"]
    assert param.ll_keywords == ["VoltCore India"]
    assert param.stream is False


def test_lightrag_query_data_without_graph_is_empty(tmp_path: Any) -> None:
    engine = LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id="empty_workspace",
        query_param_factory=_query_param_factory,
    )

    result = asyncio.run(
        engine.aquery_data(
            "Who supplies the BMS controller?",
            ll_keywords=["VoltCore India"],
        )
    )
    engine.close()

    assert result["status"] == "failure"
    assert result["metadata"]["failure_reason"] == "no_graph"
    assert result["data"] == {}
    assert result["workspace_id"] == "empty_workspace"


def test_lightrag_query_requires_llm_config(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.delenv("RETRIEVAL_GRAPH_LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    engine = LightRAGGraphEngine(
        settings=_settings(
            tmp_path,
            graph_llm_api_key=None,
            graph_llm_base_url=None,
        ),
        workspace_id="tenant_a",
        lightrag_factory=lambda _: _graph(),
        query_param_factory=_query_param_factory,
    )

    with pytest.raises(GraphConfigurationError):
        asyncio.run(engine.aquery("Who supplies the BMS controller?"))

    engine.close()


def test_lightrag_query_returns_answer_and_param(tmp_path: Any) -> None:
    fake = _graph()
    engine = LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id="tenant_a",
        lightrag_factory=lambda _: fake,
        query_param_factory=_query_param_factory,
    )

    answer = asyncio.run(
        engine.aquery(
            "Who supplies the BMS controller?",
            mode="local",
            ll_keywords=["VoltCore India"],
            system_prompt="Use the graph.",
        )
    )
    engine.close()

    assert answer == "answer: Who supplies the BMS controller?"
    param = fake.query_calls[0]["param"]
    assert param.mode == "local"
    assert param.ll_keywords == ["VoltCore India"]
    assert fake.query_calls[0]["system_prompt"] == "Use the graph."


def test_lightrag_query_llm_returns_structured_and_answer(tmp_path: Any) -> None:
    fake = _graph()
    engine = LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id="tenant_a",
        lightrag_factory=lambda _: fake,
        query_param_factory=_query_param_factory,
    )

    result = asyncio.run(
        engine.aquery_llm(
            "Who supplies the BMS controller?",
            mode="global",
            hl_keywords=["supply"],
            system_prompt="Use the graph.",
        )
    )
    engine.close()

    assert result["status"] == "success"
    assert result["workspace_id"] == "tenant_a"
    assert result["llm_response"]["content"].startswith("answer:")
    call = fake.query_llm_calls[0]
    assert call["param"].mode == "global"
    assert call["system_prompt"] == "Use the graph."


def test_lightrag_query_rejects_invalid_mode(tmp_path: Any) -> None:
    engine = LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id="tenant_a",
        lightrag_factory=lambda _: _graph(),
        query_param_factory=_query_param_factory,
    )

    with pytest.raises(ValueError):
        asyncio.run(engine.aquery_data("query", mode="unknown"))

    engine.close()


def test_lightrag_query_data_from_running_loop(tmp_path: Any) -> None:
    fake = _graph()
    engine = LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id="tenant_a",
        lightrag_factory=lambda _: fake,
        query_param_factory=_query_param_factory,
    )

    async def scenario() -> Dict[str, Any]:
        return engine.query_data(
            "Who supplies the BMS controller?",
            ll_keywords=["VoltCore India"],
        )

    try:
        result = asyncio.run(scenario())
    finally:
        engine.close()

    assert result["data"]["entities"][0]["entity_name"] == "VoltCore India"


class _WarmupEmbedder:
    def __init__(self, loaded: bool = False):
        self.is_loaded = loaded
        self.calls: List[List[str]] = []

    async def embed_documents(self, texts: Sequence[str], batch_size: Any = None) -> Any:
        self.calls.append(list(texts))


def test_lightrag_warms_unloaded_default_embedder(tmp_path: Any) -> None:
    embedder = _WarmupEmbedder(loaded=False)
    engine = LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id="tenant_a",
        embedder=embedder,
    )

    asyncio.run(engine._warm_default_embedder_async())
    engine.close()

    assert embedder.calls == [["LightRAG embedding warmup"]]


def test_lightrag_skips_warmup_for_loaded_embedder(tmp_path: Any) -> None:
    embedder = _WarmupEmbedder(loaded=True)
    engine = LightRAGGraphEngine(
        settings=_settings(tmp_path),
        workspace_id="tenant_a",
        embedder=embedder,
    )

    asyncio.run(engine._warm_default_embedder_async())
    engine.close()

    assert embedder.calls == []
