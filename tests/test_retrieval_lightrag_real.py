"""Opt-in real LightRAG query verification.

Run with ``LIGHTRAG_REAL_TESTS=1`` after installing
``retrieval/requirements-retrieval.txt``. This test uses real LightRAG storage
and query APIs with deterministic fake embeddings; it does not call an LLM.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

pytest.importorskip("lightrag")

from retrieval.config import RetrievalSettings
from retrieval.lightrag_indexer import LightRAGGraphEngine

pytestmark = pytest.mark.skipif(
    os.getenv("LIGHTRAG_REAL_TESTS") != "1",
    reason="set LIGHTRAG_REAL_TESTS=1 to run real LightRAG verification",
)


class _FakeEmbedder:
    dimension = 1024

    async def embed_documents(self, texts: Any, batch_size: Any = None) -> Any:
        return np.ones((len(texts), self.dimension), dtype=np.float32)

    async def embed_query(self, text: str) -> Any:
        return np.ones(self.dimension, dtype=np.float32)


CUSTOM_KG = {
    "chunks": [
        {
            "content": "VoltCore India supplies BMS-Controller-v4 modules.",
            "source_id": "chunk-1",
            "file_path": "supplier.pdf",
            "chunk_order_index": 0,
        }
    ],
    "entities": [
        {
            "entity_name": "VoltCore India",
            "entity_type": "Supplier",
            "description": "Supplies BMS controller modules.",
            "source_id": "chunk-1",
            "file_path": "supplier.pdf",
        },
        {
            "entity_name": "BMS-Controller-v4",
            "entity_type": "BatteryModule",
            "description": "Battery management controller.",
            "source_id": "chunk-1",
            "file_path": "supplier.pdf",
        },
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
        }
    ],
}


def test_real_lightrag_structured_query(tmp_path: Path) -> None:
    engine = LightRAGGraphEngine(
        settings=RetrievalSettings(graph_storage_root=tmp_path),
        workspace_id="tenant_a",
        embedder=_FakeEmbedder(),
    )

    async def scenario() -> Dict[str, Any]:
        client = await engine._ensure_client_async()
        await client.ainsert_custom_kg(CUSTOM_KG, full_doc_id="doc-1")
        return await engine.aquery_data(
            "Who supplies the BMS controller?",
            mode="hybrid",
            hl_keywords=["supply", "modules"],
            ll_keywords=["VoltCore India", "BMS-Controller-v4"],
        )

    try:
        result = asyncio.run(scenario())
    finally:
        asyncio.run(engine.aclose())

    assert result["status"] == "success"
    assert result["workspace_id"] == "tenant_a"
    assert result["mode"] == "hybrid"
    assert result["data"]["entities"]
    assert result["data"]["relationships"]
    assert result["data"]["chunks"]
