"""LightRAG-backed entity-relationship graph retrieval for Member 2.

The engine binds a LightRAG instance to one workspace and lazily creates the
underlying client only when graph data or indexing is actually needed. Traversal
is exposed synchronously for the existing ``BaseGraphRAGRetriever`` contract and
is executed on a dedicated background event loop so callers may invoke it from
inside an already-running asyncio loop.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from retrieval.config import RetrievalSettings, get_settings
from retrieval.interfaces import BaseGraphRAGRetriever
from retrieval.schemas import validate_workspace_id

GRAPH_NAMESPACE = "chunk_entity_relation"
GRAPH_FILENAME = f"graph_{GRAPH_NAMESPACE}.graphml"
DEFAULT_WORKSPACE_ID = "default"
QUERY_MODES = frozenset({"local", "global", "hybrid", "naive", "mix", "bypass"})
_PROCESSED_STATUSES = frozenset({"processed"})
_FAILED_STATUSES = frozenset({"failed"})
_PENDING_STATUSES = frozenset(
    {"pending", "processing", "parsing", "analyzing", "preprocessed"}
)


class GraphConfigurationError(RuntimeError):
    """Raised when LightRAG graph configuration or runtime dependencies are invalid."""


class GraphWorkspaceError(ValueError):
    """Raised when graph data crosses a workspace boundary."""


class GraphNode(BaseModel):
    """Entity node in the knowledge graph."""

    id: str
    entity_name: str
    entity_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Directed relation between two entity nodes."""

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    provenance_doc: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class _AsyncLoopRunner:
    """Run coroutines on a private event loop owned by one thread."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._closed = False

    @property
    def started(self) -> bool:
        return self._thread is not None

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._closed:
                raise RuntimeError("LightRAG graph event loop is closed")
            if self._loop is not None:
                return self._loop

            ready = threading.Event()

            def _run() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                ready.set()
                try:
                    loop.run_forever()
                finally:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()

            thread = threading.Thread(
                target=_run,
                name="lightrag-graph-loop",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            ready.wait()
            if self._loop is None:
                raise RuntimeError("LightRAG graph event loop failed to start")
            return self._loop

    def submit(self, coro: Any) -> Any:
        loop = self._ensure_started()
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        if current is loop:
            raise RuntimeError(
                "Cannot synchronously wait on the LightRAG graph event loop "
                "from within that same loop"
            )
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def run(self, coro: Any) -> Any:
        return self.submit(coro).result()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread

        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10)


class LightRAGGraphEngine(BaseGraphRAGRetriever):
    """Workspace-scoped LightRAG graph traversal and indexing adapter."""

    def __init__(
        self,
        settings: Optional[RetrievalSettings] = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        lightrag_factory: Optional[Any] = None,
        embedder: Optional[Any] = None,
        query_param_factory: Optional[Callable[..., Any]] = None,
    ):
        self.settings = settings or get_settings()
        self.workspace_id = validate_workspace_id(workspace_id)
        self._lightrag_factory = lightrag_factory
        self._embedder = embedder
        self._query_param_factory = query_param_factory
        self._client: Optional[Any] = None
        self._runner: Optional[_AsyncLoopRunner] = None
        self._indexed = False

    @property
    def graph_storage_root(self) -> Path:
        return self.settings.resolved_graph_storage_root

    @property
    def graph_file_path(self) -> Path:
        return self.graph_storage_root / self.workspace_id / GRAPH_FILENAME

    def _get_runner(self) -> _AsyncLoopRunner:
        if self._runner is None:
            self._runner = _AsyncLoopRunner()
        return self._runner

    def _run_sync(self, coro: Any) -> Any:
        return self._get_runner().run(coro)

    async def _run_async(self, coro: Any) -> Any:
        runner = self._get_runner()
        return await asyncio.wrap_future(runner.submit(coro))

    def _llm_api_key(self) -> Optional[str]:
        return (
            self.settings.graph_llm_api_key
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

    def _llm_base_url(self) -> Optional[str]:
        return self.settings.graph_llm_base_url or os.getenv("OPENAI_BASE_URL")

    def _llm_config_available(self) -> bool:
        return bool(self._llm_api_key())

    def _require_llm_config(self) -> None:
        if not self._llm_config_available():
            raise GraphConfigurationError(
                "LightRAG indexing requires an LLM provider. Set "
                "RETRIEVAL_GRAPH_LLM_API_KEY (or GROQ_API_KEY/OPENAI_API_KEY) "
                "and optionally RETRIEVAL_GRAPH_LLM_BASE_URL/OPENAI_BASE_URL."
            )

    def _should_try_client(self) -> bool:
        if self._client is not None or self._lightrag_factory is not None:
            return True
        if self._indexed:
            return True
        if self.settings.graph_graph_storage != "NetworkXStorage":
            return True
        return self.graph_file_path.is_file()

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from retrieval.embeddings.bge_m3 import BgeM3Embedder

            self._embedder = BgeM3Embedder(self.settings)
        return self._embedder

    def _build_default_client(self) -> Any:
        try:
            from lightrag import LightRAG
            from lightrag.llm.openai import openai_complete_if_cache
            from lightrag.utils import EmbeddingFunc
        except ImportError as exc:
            raise GraphConfigurationError(
                "LightRAG is not installed. Install the retrieval graph runtime "
                "with `pip install -r retrieval/requirements-retrieval.txt`."
            ) from exc

        embedder = self._get_embedder()

        async def _embed(texts: Sequence[str], context: Optional[str] = None, **_: Any) -> Any:
            if context == "query" and len(texts) == 1:
                vector = await embedder.embed_query(texts[0])
                return vector.reshape(1, -1)
            return await embedder.embed_documents(texts)

        embedding_func = EmbeddingFunc(
            embedding_dim=self.settings.embedding_dimension,
            func=_embed,
            max_token_size=self.settings.embedding_max_length,
            model_name=self.settings.embedding_model_name,
            supports_asymmetric=True,
        )

        async def _llm(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[List[Dict[str, Any]]] = None,
            **kwargs: Any,
        ) -> Any:
            self._require_llm_config()
            kwargs.pop("model", None)
            kwargs.pop("prompt", None)
            kwargs.pop("system_prompt", None)
            kwargs.pop("history_messages", None)
            kwargs.pop("base_url", None)
            kwargs.pop("api_key", None)
            kwargs.pop("timeout", None)
            return await openai_complete_if_cache(
                model=self.settings.graph_llm_model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=self._llm_base_url(),
                api_key=self._llm_api_key(),
                timeout=self.settings.graph_llm_timeout,
                **kwargs,
            )

        self.settings.ensure_graph_storage_root()
        return LightRAG(
            working_dir=str(self.graph_storage_root),
            workspace=self.workspace_id,
            kv_storage=self.settings.graph_kv_storage,
            vector_storage=self.settings.graph_vector_storage,
            graph_storage=self.settings.graph_graph_storage,
            doc_status_storage=self.settings.graph_doc_status_storage,
            embedding_func=embedding_func,
            llm_model_func=_llm,
            llm_model_name=self.settings.graph_llm_model,
            llm_model_max_async=self.settings.graph_llm_max_async,
            default_llm_timeout=self.settings.graph_llm_timeout,
            max_graph_nodes=self.settings.graph_max_nodes,
        )

    async def _warm_default_embedder_async(self) -> None:
        embedder = self._get_embedder()
        if getattr(embedder, "is_loaded", True):
            return
        await embedder.embed_documents(["LightRAG embedding warmup"])

    async def _ensure_client_async(self) -> Any:
        if self._client is not None:
            return self._client

        if self._lightrag_factory is not None:
            client = self._lightrag_factory(self)
        else:
            await self._warm_default_embedder_async()
            client = self._build_default_client()

        initialize = getattr(client, "initialize_storages", None)
        if initialize is not None:
            result = initialize()
            if inspect.isawaitable(result):
                await result

        self._client = client
        return client

    async def _client_labels(self, client: Any) -> List[str]:
        getter = getattr(client, "get_graph_labels", None)
        if getter is None:
            graph = getattr(client, "chunk_entity_relation_graph", None)
            getter = getattr(graph, "get_all_labels", None)
        if getter is None:
            raise GraphConfigurationError(
                "LightRAG client does not expose graph label retrieval"
            )
        result = getter()
        if inspect.isawaitable(result):
            result = await result
        return [str(label) for label in (result or [])]

    async def _client_knowledge_graph(
        self, client: Any, label: str, max_hops: int
    ) -> Any:
        getter = getattr(client, "get_knowledge_graph", None)
        if getter is None:
            graph = getattr(client, "chunk_entity_relation_graph", None)
            getter = getattr(graph, "get_knowledge_graph", None)
        if getter is None:
            raise GraphConfigurationError(
                "LightRAG client does not expose knowledge graph retrieval"
            )
        result = getter(
            label,
            max_depth=max_hops,
            max_nodes=self.settings.graph_traversal_max_nodes,
        )
        if inspect.isawaitable(result):
            result = await result
        return result

    @staticmethod
    def _match_labels(labels: Sequence[str], entity_seeds: Sequence[str]) -> List[str]:
        matched: List[str] = []
        for seed in entity_seeds:
            if not isinstance(seed, str) or not seed.strip():
                continue
            seed_lower = seed.strip().lower()
            for label in labels:
                label_lower = label.lower()
                if seed_lower == label_lower or seed_lower in label_lower:
                    if label not in matched:
                        matched.append(label)
        return matched

    @staticmethod
    def _empty_result(
        query: str,
        entity_seeds: Sequence[str],
        max_hops: int,
        workspace_id: str,
    ) -> Dict[str, Any]:
        return {
            "query": query,
            "seed_entities": list(entity_seeds),
            "max_hops": max_hops,
            "total_nodes_found": 0,
            "nodes": [],
            "edges": [],
            "context_summary": "",
            "truncated": False,
            "workspace_id": workspace_id,
        }

    @staticmethod
    def _kg_value(kg: Any, name: str, default: Any) -> Any:
        if isinstance(kg, Mapping):
            return kg.get(name, default)
        return getattr(kg, name, default)

    @staticmethod
    def _item_value(item: Any, name: str, default: Any = None) -> Any:
        if isinstance(item, Mapping):
            return item.get(name, default)
        return getattr(item, name, default)

    @staticmethod
    def _relation_type(properties: Mapping[str, Any]) -> str:
        value = properties.get("relation_type")
        if value:
            return str(value)
        return "RELATED_TO"

    @staticmethod
    def _weight(properties: Mapping[str, Any]) -> float:
        try:
            return float(properties.get("weight", 1.0))
        except (TypeError, ValueError):
            return 1.0

    def _node_model(self, node: Any, node_id: str, hop: int) -> GraphNode:
        properties = dict(self._item_value(node, "properties", {}) or {})
        labels = list(self._item_value(node, "labels", []) or [])
        entity_name = str(properties.get("entity_name") or node_id)
        entity_type = properties.get("entity_type")
        if isinstance(entity_type, (list, tuple)):
            entity_type = ", ".join(str(item) for item in entity_type)
        if not entity_type:
            candidate = labels[0] if labels else ""
            entity_type = candidate if candidate and candidate != entity_name else "Entity"
        properties.setdefault("labels", labels)
        properties.setdefault("hop", hop)
        return GraphNode(
            id=node_id,
            entity_name=entity_name,
            entity_type=str(entity_type),
            properties=properties,
        )

    def _edge_model(
        self,
        edge: Any,
        node_names: Mapping[str, str],
    ) -> Optional[GraphEdge]:
        source_id = str(self._item_value(edge, "source", ""))
        target_id = str(self._item_value(edge, "target", ""))
        if not source_id or not target_id:
            return None
        properties = dict(self._item_value(edge, "properties", {}) or {})
        relation_type = self._relation_type(properties)
        provenance = properties.get("file_path")
        return GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=self._weight(properties),
            provenance_doc=str(provenance) if provenance else None,
            properties=properties,
        )

    async def _atraverse_entities(
        self,
        query: str,
        entity_seeds: Sequence[str],
        max_hops: int,
    ) -> Dict[str, Any]:
        if not self._should_try_client():
            return self._empty_result(query, entity_seeds, max_hops, self.workspace_id)

        client = await self._ensure_client_async()
        labels = await self._client_labels(client)
        matched_labels = self._match_labels(labels, entity_seeds)
        if not matched_labels:
            return self._empty_result(query, entity_seeds, max_hops, self.workspace_id)

        node_map: Dict[str, Any] = {}
        edge_map: Dict[Tuple[str, str], Any] = {}
        truncated = False

        for label in matched_labels:
            kg = await self._client_knowledge_graph(client, label, max_hops)
            truncated = truncated or bool(self._kg_value(kg, "is_truncated", False))
            for node in self._kg_value(kg, "nodes", []) or []:
                node_id = str(self._item_value(node, "id", ""))
                if node_id and node_id not in node_map:
                    node_map[node_id] = node
            for edge in self._kg_value(kg, "edges", []) or []:
                source_id = str(self._item_value(edge, "source", ""))
                target_id = str(self._item_value(edge, "target", ""))
                if not source_id or not target_id:
                    continue
                key = tuple(sorted((source_id, target_id)))
                edge_map[key] = edge

        adjacency: Dict[str, set[str]] = defaultdict(set)
        for edge in edge_map.values():
            source_id = str(self._item_value(edge, "source", ""))
            target_id = str(self._item_value(edge, "target", ""))
            if source_id and target_id:
                adjacency[source_id].add(target_id)
                adjacency[target_id].add(source_id)

        distances: Dict[str, int] = {}
        queue: deque[str] = deque()
        for label in matched_labels:
            if label in node_map:
                distances[label] = 0
                queue.append(label)

        while queue:
            current = queue.popleft()
            current_hop = distances[current]
            if current_hop >= max_hops:
                continue
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor not in distances:
                    distances[neighbor] = current_hop + 1
                    queue.append(neighbor)

        included_ids = [
            node_id
            for node_id, hop in distances.items()
            if node_id in node_map and hop <= max_hops
        ]
        included_ids.sort(key=lambda node_id: (distances[node_id], node_id.lower()))

        graph_nodes = [
            self._node_model(node_map[node_id], node_id, distances[node_id])
            for node_id in included_ids
        ]
        node_names = {node.id: node.entity_name for node in graph_nodes}

        graph_edges: List[GraphEdge] = []
        for edge in edge_map.values():
            source_id = str(self._item_value(edge, "source", ""))
            target_id = str(self._item_value(edge, "target", ""))
            if source_id not in node_names or target_id not in node_names:
                continue
            if max(distances[source_id], distances[target_id]) > max_hops:
                continue
            model = self._edge_model(edge, node_names)
            if model is not None:
                graph_edges.append(model)

        graph_edges.sort(
            key=lambda item: (
                max(distances[item.source_id], distances[item.target_id]),
                item.source_id.lower(),
                item.target_id.lower(),
            )
        )

        facts: List[str] = []
        for edge in graph_edges:
            source_name = node_names.get(edge.source_id, edge.source_id)
            target_name = node_names.get(edge.target_id, edge.target_id)
            line = (
                f"Graph Link: {source_name} -> [{edge.relation_type}] -> "
                f"{target_name}"
            )
            description = edge.properties.get("description")
            if description:
                line = f"{line} ({str(description)[:500]})"
            facts.append(line)

        return {
            "query": query,
            "seed_entities": list(entity_seeds),
            "max_hops": max_hops,
            "total_nodes_found": len(graph_nodes),
            "nodes": [node.model_dump() for node in graph_nodes],
            "edges": [edge.model_dump() for edge in graph_edges],
            "context_summary": "\n".join(facts),
            "truncated": truncated,
            "workspace_id": self.workspace_id,
        }

    def traverse_entities(
        self,
        query: str,
        entity_seeds: List[str],
        max_hops: int = 2,
    ) -> Dict[str, Any]:
        """Traverse real LightRAG entity relationships from the supplied seeds."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if max_hops < 0:
            raise ValueError("max_hops must be >= 0")
        seeds = list(entity_seeds or [])
        return self._run_sync(self._atraverse_entities(query, seeds, max_hops))

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        return query

    @staticmethod
    def _validate_query_mode(mode: str) -> str:
        if mode not in QUERY_MODES:
            allowed = ", ".join(sorted(QUERY_MODES))
            raise ValueError(f"mode must be one of: {allowed}")
        return mode

    @staticmethod
    def _validate_query_limit(value: int, *, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
        return value

    def _build_query_param(
        self,
        *,
        mode: str,
        top_k: int,
        chunk_top_k: int,
        hl_keywords: Optional[Sequence[str]],
        ll_keywords: Optional[Sequence[str]],
        include_references: bool,
        enable_rerank: bool,
    ) -> Any:
        mode = self._validate_query_mode(mode)
        top_k = self._validate_query_limit(top_k, name="top_k")
        chunk_top_k = self._validate_query_limit(chunk_top_k, name="chunk_top_k")
        kwargs = {
            "mode": mode,
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
            "hl_keywords": list(hl_keywords or []),
            "ll_keywords": list(ll_keywords or []),
            "include_references": include_references,
            "enable_rerank": enable_rerank,
            "stream": False,
        }
        if self._query_param_factory is not None:
            return self._query_param_factory(**kwargs)
        try:
            from lightrag import QueryParam
        except ImportError as exc:
            raise GraphConfigurationError(
                "LightRAG is not installed. Install the retrieval graph runtime "
                "with `pip install -r retrieval/requirements-retrieval.txt`."
            ) from exc
        return QueryParam(**kwargs)

    def _empty_query_result(
        self,
        query: str,
        mode: str,
        *,
        message: str,
    ) -> Dict[str, Any]:
        return {
            "status": "failure",
            "message": message,
            "data": {},
            "metadata": {"failure_reason": "no_graph", "query_mode": mode},
            "query": query,
            "mode": mode,
            "workspace_id": self.workspace_id,
        }

    def _normalize_query_result(
        self,
        payload: Any,
        query: str,
        mode: str,
    ) -> Dict[str, Any]:
        if isinstance(payload, Mapping):
            result = dict(payload)
        else:
            result = {"status": "success", "message": "", "data": payload or {}}
        result.setdefault("status", "success")
        result.setdefault("message", "")
        data = result.get("data")
        result["data"] = data if isinstance(data, Mapping) else (data or {})
        metadata = result.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        metadata.setdefault("query_mode", mode)
        result["metadata"] = metadata
        result["query"] = query
        result["mode"] = mode
        result["workspace_id"] = self.workspace_id
        return result

    async def _aquery_payload(
        self,
        query: str,
        *,
        mode: str,
        top_k: int,
        chunk_top_k: int,
        hl_keywords: Optional[Sequence[str]],
        ll_keywords: Optional[Sequence[str]],
        include_references: bool,
        enable_rerank: bool,
        method_name: str,
        system_prompt: Optional[str] = None,
    ) -> Any:
        query = self._validate_query(query)
        mode = self._validate_query_mode(mode)
        if mode != "bypass" and not self._should_try_client():
            return None
        client = await self._ensure_client_async()
        method = getattr(client, method_name, None)
        if method is None:
            raise GraphConfigurationError(
                f"LightRAG client does not expose {method_name}"
            )
        param = self._build_query_param(
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            hl_keywords=hl_keywords,
            ll_keywords=ll_keywords,
            include_references=include_references,
            enable_rerank=enable_rerank,
        )
        if method_name in {"aquery", "aquery_llm"}:
            result = method(query, param, system_prompt=system_prompt)
        else:
            result = method(query, param)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def aquery_data(
        self,
        query: str,
        *,
        mode: str = "mix",
        top_k: Optional[int] = None,
        chunk_top_k: Optional[int] = None,
        hl_keywords: Optional[Sequence[str]] = None,
        ll_keywords: Optional[Sequence[str]] = None,
        include_references: bool = True,
        enable_rerank: bool = False,
    ) -> Dict[str, Any]:
        """Run LightRAG structured retrieval without invoking LLM generation."""
        query = self._validate_query(query)
        mode = self._validate_query_mode(mode)
        top_k = top_k if top_k is not None else self.settings.retrieval_top_k
        chunk_top_k = (
            chunk_top_k if chunk_top_k is not None else self.settings.retrieval_top_k
        )
        payload = await self._aquery_payload(
            query,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            hl_keywords=hl_keywords,
            ll_keywords=ll_keywords,
            include_references=include_references,
            enable_rerank=enable_rerank,
            method_name="aquery_data",
        )
        if payload is None:
            return self._empty_query_result(
                query,
                mode,
                message=f"No graph data found for workspace {self.workspace_id!r}",
            )
        return self._normalize_query_result(payload, query, mode)

    async def aquery_llm(
        self,
        query: str,
        *,
        mode: str = "mix",
        top_k: Optional[int] = None,
        chunk_top_k: Optional[int] = None,
        hl_keywords: Optional[Sequence[str]] = None,
        ll_keywords: Optional[Sequence[str]] = None,
        include_references: bool = True,
        enable_rerank: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run LightRAG retrieval and LLM generation, returning both."""
        query = self._validate_query(query)
        mode = self._validate_query_mode(mode)
        self._require_llm_config()
        top_k = top_k if top_k is not None else self.settings.retrieval_top_k
        chunk_top_k = (
            chunk_top_k if chunk_top_k is not None else self.settings.retrieval_top_k
        )
        payload = await self._aquery_payload(
            query,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            hl_keywords=hl_keywords,
            ll_keywords=ll_keywords,
            include_references=include_references,
            enable_rerank=enable_rerank,
            method_name="aquery_llm",
            system_prompt=system_prompt,
        )
        if payload is None:
            return self._empty_query_result(
                query,
                mode,
                message=f"No graph data found for workspace {self.workspace_id!r}",
            )
        return self._normalize_query_result(payload, query, mode)

    async def aquery(
        self,
        query: str,
        *,
        mode: str = "mix",
        top_k: Optional[int] = None,
        chunk_top_k: Optional[int] = None,
        hl_keywords: Optional[Sequence[str]] = None,
        ll_keywords: Optional[Sequence[str]] = None,
        include_references: bool = True,
        enable_rerank: bool = False,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Run a LightRAG query and return the generated answer text."""
        query = self._validate_query(query)
        mode = self._validate_query_mode(mode)
        self._require_llm_config()
        top_k = top_k if top_k is not None else self.settings.retrieval_top_k
        chunk_top_k = (
            chunk_top_k if chunk_top_k is not None else self.settings.retrieval_top_k
        )
        result = await self._aquery_payload(
            query,
            mode=mode,
            top_k=top_k,
            chunk_top_k=chunk_top_k,
            hl_keywords=hl_keywords,
            ll_keywords=ll_keywords,
            include_references=include_references,
            enable_rerank=enable_rerank,
            method_name="aquery",
            system_prompt=system_prompt,
        )
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if hasattr(result, "__aiter__"):
            chunks: List[str] = []
            async for chunk in result:
                chunks.append(str(chunk))
            return "".join(chunks)
        return str(result)

    def query_data(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run_sync(self.aquery_data(query, **kwargs))

    def query_llm(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run_sync(self.aquery_llm(query, **kwargs))

    def query(self, query: str, **kwargs: Any) -> str:
        return self._run_sync(self.aquery(query, **kwargs))

    @staticmethod
    def _chunk_value(chunk: Any, name: str, default: Any = None) -> Any:
        if isinstance(chunk, Mapping):
            return chunk.get(name, default)
        return getattr(chunk, name, default)

    @staticmethod
    def _status_value(value: Any) -> str:
        if value is None:
            return ""
        status = getattr(value, "value", value)
        return str(status).strip().lower()

    @staticmethod
    def _result_status(result: Any) -> str:
        if isinstance(result, Mapping):
            return LightRAGGraphEngine._status_value(result.get("status"))
        return LightRAGGraphEngine._status_value(getattr(result, "status", None))

    @staticmethod
    def _document_error(row: Any) -> Optional[str]:
        if row is None:
            return None
        value = getattr(row, "error_msg", None)
        if value is None:
            value = getattr(row, "error", None)
        if value is None and isinstance(row, Mapping):
            value = row.get("error_msg") or row.get("error")
        return str(value) if value else None

    @staticmethod
    def _chunk_source_path(
        file_path: Any,
        doc_title: Any,
        doc_id: Any,
        chunk_id: Any,
    ) -> str:
        raw_source = file_path or doc_title or doc_id or chunk_id
        source_name = Path(str(raw_source)).name or str(raw_source)
        safe_chunk = "".join(
            char if char.isalnum() or char in "._-" else "_"
            for char in str(chunk_id)
        )
        return f"{source_name}::chunk::{safe_chunk}"

    def _validate_chunk_workspaces(self, chunks: Iterable[Any]) -> None:
        for chunk in chunks:
            workspace_id = self._chunk_value(chunk, "workspace_id")
            if workspace_id is not None and str(workspace_id) != self.workspace_id:
                raise GraphWorkspaceError(
                    f"chunk belongs to workspace {workspace_id!r}, expected "
                    f"{self.workspace_id!r}"
                )

    def _build_chunk_entries(
        self, chunks: Sequence[Any]
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        entries: List[Dict[str, Any]] = []
        by_id: Dict[str, Dict[str, Any]] = {}
        empty_chunks: List[str] = []
        duplicate_input_chunks: List[str] = []
        for position, chunk in enumerate(chunks):
            content = self._chunk_value(chunk, "content")
            chunk_id = self._chunk_value(chunk, "chunk_id")
            if not isinstance(content, str) or not content.strip():
                if chunk_id:
                    empty_chunks.append(str(chunk_id))
                continue
            if not chunk_id:
                raise GraphConfigurationError("every chunk must have a chunk_id")
            chunk_id = str(chunk_id)
            doc_id_value = self._chunk_value(chunk, "doc_id")
            doc_id = str(doc_id_value) if doc_id_value else chunk_id
            doc_title = self._chunk_value(chunk, "doc_title")
            metadata = self._chunk_value(chunk, "metadata", {}) or {}
            file_path = (
                metadata.get("file_path") if isinstance(metadata, Mapping) else None
            )
            existing = by_id.get(chunk_id)
            if existing is not None:
                if existing["content"] != content:
                    raise GraphConfigurationError(
                        f"chunk_id {chunk_id!r} was supplied with different content"
                    )
                duplicate_input_chunks.append(chunk_id)
                continue
            entry = {
                "position": position,
                "content": content,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source_path": self._chunk_source_path(
                    file_path, doc_title, doc_id, chunk_id
                ),
                "status": "new",
                "error": None,
            }
            by_id[chunk_id] = entry
            entries.append(entry)
        return entries, empty_chunks, duplicate_input_chunks

    async def _document_status(self, client: Any, doc_id: str) -> Any:
        storage = getattr(client, "doc_status", None)
        if storage is None:
            return None
        getter = getattr(storage, "get_by_id", None)
        if getter is None:
            return None
        result = getter(doc_id)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def _delete_failed_document(self, client: Any, doc_id: str) -> None:
        deleter = getattr(client, "adelete_by_doc_id", None)
        if deleter is None:
            raise GraphConfigurationError(
                f"LightRAG cannot delete failed document {doc_id!r} before retry"
            )
        result = deleter(doc_id)
        if inspect.isawaitable(result):
            result = await result
        status = self._result_status(result)
        if status and status not in {"success", "not_found"}:
            raise GraphConfigurationError(
                f"LightRAG failed to delete document {doc_id!r} before retry: "
                f"{status}"
            )

    async def _process_pending_documents(self, client: Any) -> None:
        processor = getattr(client, "apipeline_process_enqueue_documents", None)
        if processor is None:
            return
        result = processor()
        if inspect.isawaitable(result):
            await result

    async def _has_pending_status(
        self, client: Any, entries: Sequence[Dict[str, Any]]
    ) -> bool:
        for entry in entries:
            row = await self._document_status(client, entry["chunk_id"])
            status = self._status_value(self._item_value(row, "status", None))
            if status in _PENDING_STATUSES:
                return True
        return False

    def _empty_index_result(
        self,
        track_id: Optional[str],
        *,
        empty_chunks: Optional[List[str]] = None,
        duplicate_input_chunks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "status": "completed",
            "indexed_documents": 0,
            "submitted_chunks": 0,
            "source_documents": 0,
            "processed_chunks": [],
            "failed_chunks": [],
            "pending_chunks": [],
            "skipped_chunks": [],
            "retried_failed_chunks": [],
            "empty_chunks": empty_chunks or [],
            "duplicate_input_chunks": duplicate_input_chunks or [],
            "chunk_mappings": [],
            "track_id": track_id,
            "workspace_id": self.workspace_id,
        }

    async def aindex_chunks(
        self,
        chunks: Sequence[Any],
        *,
        track_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        chunk_list = list(chunks)
        if not chunk_list:
            return self._empty_index_result(track_id)

        self._validate_chunk_workspaces(chunk_list)
        self._require_llm_config()
        entries, empty_chunks, duplicate_input_chunks = self._build_chunk_entries(
            chunk_list
        )
        if not entries:
            return self._empty_index_result(
                track_id,
                empty_chunks=empty_chunks,
                duplicate_input_chunks=duplicate_input_chunks,
            )

        client = await self._ensure_client_async()
        submitted: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        pending: List[Dict[str, Any]] = []
        retried_failed_chunks: List[str] = []

        for entry in entries:
            row = await self._document_status(client, entry["chunk_id"])
            status = self._status_value(self._item_value(row, "status", None))
            entry["error"] = self._document_error(row)
            if status in _PROCESSED_STATUSES:
                entry["status"] = "processed"
                skipped.append(entry)
            elif status in _FAILED_STATUSES:
                await self._delete_failed_document(client, entry["chunk_id"])
                entry["status"] = "submitted"
                retried_failed_chunks.append(entry["chunk_id"])
                submitted.append(entry)
            elif status in _PENDING_STATUSES:
                entry["status"] = "pending"
                pending.append(entry)
            else:
                entry["status"] = "submitted"
                submitted.append(entry)

        result: Any = track_id
        if submitted:
            result = client.ainsert(
                [entry["content"] for entry in submitted],
                ids=[entry["chunk_id"] for entry in submitted],
                file_paths=[entry["source_path"] for entry in submitted],
                track_id=track_id,
            )
            if inspect.isawaitable(result):
                result = await result

        if pending or (
            submitted and await self._has_pending_status(client, submitted)
        ):
            await self._process_pending_documents(client)

        processed_chunks: List[str] = []
        failed_chunks: List[Dict[str, Any]] = []
        pending_chunks: List[str] = []
        chunk_mappings: List[Dict[str, Any]] = []
        indexed_documents = 0
        for entry in entries:
            status = entry["status"]
            error = entry["error"]
            if status in {"submitted", "pending"}:
                row = await self._document_status(client, entry["chunk_id"])
                row_status = self._status_value(
                    self._item_value(row, "status", None)
                )
                if row_status:
                    status = row_status
                error = self._document_error(row) or error
            if status in _PROCESSED_STATUSES:
                processed_chunks.append(entry["chunk_id"])
                if entry["status"] != "processed":
                    indexed_documents += 1
            elif status in _FAILED_STATUSES:
                failed_chunks.append(
                    {
                        "chunk_id": entry["chunk_id"],
                        "doc_id": entry["doc_id"],
                        "error": error or "LightRAG marked the chunk as failed",
                    }
                )
            elif status in _PENDING_STATUSES:
                pending_chunks.append(entry["chunk_id"])
            elif status == "submitted":
                indexed_documents += 1
            chunk_mappings.append(
                {
                    "chunk_id": entry["chunk_id"],
                    "doc_id": entry["doc_id"],
                    "source_path": entry["source_path"],
                    "lightrag_document_id": entry["chunk_id"],
                    "status": status,
                    "error": error,
                }
            )

        if indexed_documents or processed_chunks or pending_chunks:
            self._indexed = True
        if failed_chunks:
            status = "partial_failure" if processed_chunks or pending_chunks else "failed"
        elif pending_chunks:
            status = "pending"
        else:
            status = "completed"
        return {
            "status": status,
            "indexed_documents": indexed_documents,
            "submitted_chunks": len(submitted),
            "source_documents": len({entry["doc_id"] for entry in entries}),
            "processed_chunks": processed_chunks,
            "failed_chunks": failed_chunks,
            "pending_chunks": pending_chunks,
            "skipped_chunks": [entry["chunk_id"] for entry in skipped],
            "retried_failed_chunks": retried_failed_chunks,
            "empty_chunks": empty_chunks,
            "duplicate_input_chunks": duplicate_input_chunks,
            "chunk_mappings": chunk_mappings,
            "track_id": result,
            "workspace_id": self.workspace_id,
        }

    async def aindex_text(
        self,
        text: str,
        *,
        doc_id: Optional[str] = None,
        file_path: Optional[str] = None,
        track_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        self._require_llm_config()
        client = await self._ensure_client_async()
        result = client.ainsert(
            text,
            ids=doc_id,
            file_paths=file_path,
            track_id=track_id,
        )
        if inspect.isawaitable(result):
            result = await result
        self._indexed = True
        return {
            "indexed_documents": 1,
            "track_id": result,
            "workspace_id": self.workspace_id,
        }

    async def aindex_document(
        self,
        document: Any,
        chunks: Sequence[Any],
        *,
        track_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        workspace_id = self._chunk_value(document, "workspace_id")
        if workspace_id is not None and str(workspace_id) != self.workspace_id:
            raise GraphWorkspaceError(
                f"document belongs to workspace {workspace_id!r}, expected "
                f"{self.workspace_id!r}"
            )
        return await self.aindex_chunks(chunks, track_id=track_id)

    async def aindex_workspace(
        self,
        repository: Any,
        *,
        document_ids: Optional[Sequence[str]] = None,
        limit: int = 100,
        track_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if document_ids:
            documents = []
            for doc_id in document_ids:
                document = await repository.get_document(self.workspace_id, doc_id)
                if document is not None:
                    documents.append(document)
        else:
            documents = await repository.list_documents(self.workspace_id, limit=limit)

        chunks: List[Any] = []
        for document in documents:
            chunks.extend(
                await repository.get_chunks(self.workspace_id, document.doc_id)
            )
        return await self.aindex_chunks(chunks, track_id=track_id)

    def index_chunks(self, chunks: Sequence[Any], **kwargs: Any) -> Dict[str, Any]:
        return self._run_sync(self.aindex_chunks(chunks, **kwargs))

    def index_text(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        return self._run_sync(self.aindex_text(text, **kwargs))

    def index_document(self, document: Any, chunks: Sequence[Any], **kwargs: Any) -> Dict[str, Any]:
        return self._run_sync(self.aindex_document(document, chunks, **kwargs))

    def index_workspace(self, repository: Any, **kwargs: Any) -> Dict[str, Any]:
        return self._run_sync(self.aindex_workspace(repository, **kwargs))

    async def _afinalize_client(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        finalize = getattr(client, "finalize_storages", None)
        if finalize is not None:
            result = finalize()
            if inspect.isawaitable(result):
                await result

    async def aclose(self) -> None:
        runner = self._runner
        if runner is not None and runner.started:
            await asyncio.wrap_future(runner.submit(self._afinalize_client()))
        else:
            await self._afinalize_client()
        if runner is not None:
            runner.close()
        self._runner = None

    def close(self) -> None:
        runner = self._runner
        if runner is not None and runner.started:
            runner.run(self._afinalize_client())
            runner.close()
        self._runner = None

    def __enter__(self) -> "LightRAGGraphEngine":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
