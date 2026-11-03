"""Unit tests for Member 3: LangGraph supervisor, FastMCP tools, and OpenAI Drop-In Proxy."""
import pytest
import asyncio
from core.schemas import QueryRequest, RetrievalTier
from agents.langgraph_supervisor import LangGraphSupervisor, MockFastMCPConnector
from mcp_servers.postgres_erp.server import query_shipment_delays, get_vendor_defect_rate
from mcp_servers.github_issues.server import search_open_blockers
from fastapi.testclient import TestClient
from api.server import app


@pytest.mark.asyncio
async def test_langgraph_supervisor_execution():
    supervisor = LangGraphSupervisor()
    req = QueryRequest(query="Which open GitHub firmware issues affect projects using VoltCore's delayed batches, and who is the lead engineer?")
    res = await supervisor.run(req)

    assert res["status"] == "success"
    assert res["tier_used"] == RetrievalTier.TIER3_AGENTIC.value
    assert len(res["sub_goals"]) >= 2
    assert any("GitHub" in src or "Graph:" in src for src in res["cited_sources"])
    assert "R. Kulkarni" in res["synthesis"]


def test_fastmcp_tool_functions():
    # Test Postgres ERP tools
    erp_res = query_shipment_delays("402")
    assert erp_res["delay_days"] == 9
    assert erp_res["supplier"] == "VoltCore India"

    qa_res = get_vendor_defect_rate("VoltCore")
    assert qa_res["latest_batch_defect_percentage"] == 0.42

    # Test GitHub MCP tools
    gh_res = search_open_blockers("BMS-Controller-v4")
    assert gh_res["open_blockers_count"] == 2
    assert gh_res["lead_engineer"] == "R. Kulkarni"


def test_openai_proxy_chat_completions():
    client = TestClient(app)
    payload = {
        "model": "calibrag-adaptive",
        "messages": [
            {"role": "user", "content": "What is the maximum defect tolerance percentage allowed for lithium cell batches?"}
        ]
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "choices" in data
    assert "0.5%" in data["choices"][0]["message"]["content"]
    assert "[QA_SOP, p.12]" in data["choices"][0]["message"]["content"]

    # Verify injected CalibRAG HTTP response headers
    assert resp.headers.get("X-CalibRAG-Tier") == "tier1_vector"
    assert float(resp.headers.get("X-CalibRAG-Confidence")) >= 0.90
    assert float(resp.headers.get("X-CalibRAG-Cost-Saved-USD")) > 0.005
    assert float(resp.headers.get("X-CalibRAG-Latency-Ms")) > 0
