"""Opt-in PostgreSQL-native LightRAG graph storage verification.

Set ``RETRIEVAL_PHASE5_1_PGTABLE_TESTS=1`` and ``RETRIEVAL_TEST_DATABASE_URL``.
The test creates a disposable database, verifies ``PGTableGraphStorage``
persistence and workspace isolation, then drops the database.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from urllib.parse import urlparse

import asyncpg
import pytest

from lightrag.kg.pgtable_impl import PGTableGraphStorage
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import EmbeddingFunc

_TEST_DSN = os.getenv("RETRIEVAL_TEST_DATABASE_URL")
_ENABLED = os.getenv("RETRIEVAL_PHASE5_1_PGTABLE_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not (_ENABLED and _TEST_DSN),
    reason=(
        "set RETRIEVAL_PHASE5_1_PGTABLE_TESTS=1 and "
        "RETRIEVAL_TEST_DATABASE_URL to run the PGTableGraphStorage test"
    ),
)


async def _embed(texts):
    return [[0.0] * 1024 for _ in texts]


def _connection_parts() -> dict[str, object]:
    parsed = urlparse(_TEST_DSN)
    if not parsed.hostname or not parsed.username:
        raise ValueError("RETRIEVAL_TEST_DATABASE_URL must include host and user")
    return {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "user": parsed.username,
        "password": parsed.password or "",
    }


def test_pgtable_graph_storage_persists_and_isolates_workspaces() -> None:
    async def scenario() -> None:
        parts = _connection_parts()
        database_name = f"p51pg_{uuid.uuid4().hex[:10]}"
        admin = await asyncpg.connect(database="postgres", **parts)
        initialize_share_data()
        try:
            await admin.execute(f'CREATE DATABASE "{database_name}"')
            os.environ["POSTGRES_HOST"] = str(parts["host"])
            os.environ["POSTGRES_PORT"] = str(parts["port"])
            os.environ["POSTGRES_USER"] = str(parts["user"])
            os.environ["POSTGRES_PASSWORD"] = str(parts["password"])
            os.environ["POSTGRES_DATABASE"] = database_name
            os.environ.pop("POSTGRES_WORKSPACE", None)

            workspace_a = f"p51a_{uuid.uuid4().hex[:8]}"
            workspace_b = f"p51b_{uuid.uuid4().hex[:8]}"
            namespace = "chunk_entity_relation"
            embedding_func = EmbeddingFunc(embedding_dim=1024, func=_embed)

            storage_a = PGTableGraphStorage(
                namespace=namespace,
                workspace=workspace_a,
                global_config={"vector_storage": None},
                embedding_func=embedding_func,
            )
            storage_b = PGTableGraphStorage(
                namespace=namespace,
                workspace=workspace_b,
                global_config={"vector_storage": None},
                embedding_func=embedding_func,
            )
            try:
                await storage_a.initialize()
                await storage_b.initialize()
                await storage_a.upsert_node(
                    "Alice",
                    {"entity_id": "Alice", "entity_type": "Person"},
                )
                await storage_a.upsert_node(
                    "Acme Technologies",
                    {"entity_id": "Acme Technologies", "entity_type": "Organization"},
                )
                await storage_a.upsert_node(
                    "Pune",
                    {"entity_id": "Pune", "entity_type": "Location"},
                )
                await storage_a.upsert_edge(
                    "Alice", "Acme Technologies", {"description": "works at"}
                )
                await storage_a.upsert_edge(
                    "Acme Technologies", "Pune", {"description": "headquartered in"}
                )
                await storage_b.upsert_node(
                    "Bob",
                    {"entity_id": "Bob", "entity_type": "Person"},
                )
                await storage_b.upsert_node(
                    "Beta Systems",
                    {"entity_id": "Beta Systems", "entity_type": "Organization"},
                )
                await storage_b.upsert_edge(
                    "Bob", "Beta Systems", {"description": "works at"}
                )

                graph_a = await storage_a.get_knowledge_graph(
                    "Alice", max_depth=2, max_nodes=50
                )
                graph_b = await storage_b.get_knowledge_graph(
                    "Bob", max_depth=2, max_nodes=50
                )
                nodes_a = {node.id for node in graph_a.nodes}
                nodes_b = {node.id for node in graph_b.nodes}
                assert nodes_a == {"Alice", "Acme Technologies", "Pune"}
                assert nodes_b == {"Bob", "Beta Systems"}
                assert nodes_a.isdisjoint(nodes_b)
                assert len(graph_a.edges) == 2
                assert len(graph_b.edges) == 1
            finally:
                await storage_a.finalize()
                await storage_b.finalize()

            reload_a = PGTableGraphStorage(
                namespace=namespace,
                workspace=workspace_a,
                global_config={"vector_storage": None},
                embedding_func=embedding_func,
            )
            reload_b = PGTableGraphStorage(
                namespace=namespace,
                workspace=workspace_b,
                global_config={"vector_storage": None},
                embedding_func=embedding_func,
            )
            try:
                await reload_a.initialize()
                await reload_b.initialize()
                reload_graph_a = await reload_a.get_knowledge_graph(
                    "Alice", max_depth=2, max_nodes=50
                )
                reload_graph_b = await reload_b.get_knowledge_graph(
                    "Bob", max_depth=2, max_nodes=50
                )
                assert {node.id for node in reload_graph_a.nodes} == {
                    "Alice",
                    "Acme Technologies",
                    "Pune",
                }
                assert {node.id for node in reload_graph_b.nodes} == {
                    "Bob",
                    "Beta Systems",
                }
                assert len(reload_graph_a.edges) == 2
                assert len(reload_graph_b.edges) == 1
                await reload_a.drop()
                await reload_b.drop()
                assert await reload_a.get_all_labels() == []
                assert await reload_b.get_all_labels() == []
            finally:
                await reload_a.finalize()
                await reload_b.finalize()
        finally:
            await admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = $1 AND pid <> pg_backend_pid()",
                database_name,
            )
            await admin.execute(f'DROP DATABASE IF EXISTS "{database_name}"')
            await admin.close()

    asyncio.run(scenario())
