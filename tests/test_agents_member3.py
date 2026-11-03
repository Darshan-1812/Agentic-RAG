"""Comprehensive unit and integration tests for Member 3:
- Tier 3 LangGraph Multi-Agent Supervisor (StateGraph & Dynamic Planning)
- FastMCP Tool Servers (Generic Schema Discovery & Safe Read-Only SQL)
- Issue Tracker Tool Server (Multi-Domain Repositories)
- OpenAI Drop-In Proxy with SSE streaming & CalibRAG telemetry headers
"""
import pytest
import asyncio
import json
from core.schemas import QueryRequest, RetrievalTier
from agents.langgraph_supervisor import (
    LangGraphSupervisor,
    FastMCPConnector,
    extract_entities_from_query,
)
from mcp_servers.postgres_erp.server import (
    list_tables,
    describe_table,
    execute_read_only_query,
    query_shipment_delays,
    get_vendor_defect_rate,
    list_active_line_stoppages,
)
from mcp_servers.github_issues.server import (
    list_repositories,
    search_issues,
    search_open_blockers,
    get_issue_details,
)
from fastapi.testclient import TestClient
from api.server import app


@pytest.mark.asyncio
async def test_langgraph_supervisor_execution():
    """Validates multi-hop LangGraph StateGraph execution, tool calls, and citations."""
    supervisor = LangGraphSupervisor()
    req = QueryRequest(query="Which open GitHub firmware issues affect projects using VoltCore's delayed batches, and who is the lead engineer?")
    res = await supervisor.run(req)

    assert res["status"] == "success"
    assert res["tier_used"] == RetrievalTier.TIER3_AGENTIC.value
    assert len(res["sub_goals"]) >= 2
    assert any("GitHub" in src or "Graph:" in src for src in res["cited_sources"])
    assert "R. Kulkarni" in res["synthesis"]


def test_dynamic_entity_extraction():
    """Validates domain-agnostic dynamic entity and identifier extraction from arbitrary queries."""
    query_1 = "Why is Batch #402 delayed for VoltCore India at Pune facility?"
    entities_1 = extract_entities_from_query(query_1)
    assert any("402" in e for e in entities_1)
    assert "VoltCore India" in entities_1 or "VoltCore" in entities_1

    query_2 = "Investigate patient PAT-8092 admission under Dr. Sarah Connor in Cardiology."
    entities_2 = extract_entities_from_query(query_2)
    assert any("PAT-8092" in e for e in entities_2)
    assert any("Sarah Connor" in e for e in entities_2)


def test_generic_database_mcp_tools():
    """Validates schema-adaptive database discovery and safe read-only SQL execution."""
    # 1. Dynamic Table Listing
    tables = list_tables()
    assert "erp_shipments" in tables
    assert "vendor_quality" in tables
    assert "operational_incidents" in tables

    # 2. Table Schema Description
    schema = describe_table("operational_incidents")
    assert schema["table_name"] == "operational_incidents"
    column_names = [col["name"] for col in schema["columns"]]
    assert "incident_id" in column_names
    assert "severity" in column_names
    assert schema["total_rows"] >= 1

    # 3. Safe SELECT query
    res = execute_read_only_query("SELECT * FROM operational_incidents WHERE severity = 'CRITICAL'")
    assert res["success"] is True
    assert res["row_count"] >= 1
    assert "INC-801" in [r["incident_id"] for r in res["rows"]]

    # 4. Destructive query guard (Security Violation)
    bad_res = execute_read_only_query("DROP TABLE operational_incidents")
    assert bad_res["success"] is False
    assert "Security Violation" in bad_res["error"]

    bad_delete = execute_read_only_query("DELETE FROM erp_shipments WHERE delay_days > 0")
    assert bad_delete["success"] is False
    assert "Security Violation" in bad_delete["error"]


def test_generic_issue_tracker_tools():
    """Validates issue tracker search across multi-domain repositories."""
    repos = list_repositories()
    assert "apex-ev/firmware-bms" in repos
    assert "company/operations-platform" in repos

    # Generic search across second repository
    res = search_issues("timeout", repository="company/operations-platform")
    assert res["total_found"] >= 1
    assert any("timeout" in i["title"].lower() for i in res["issues"])


def test_fastmcp_tool_functions():
    """Validates real database and GitHub issue queries on FastMCP servers."""
    erp_res = query_shipment_delays("402")
    assert erp_res["delay_days"] == 9
    assert erp_res["supplier"] == "VoltCore India"
    assert erp_res["found_in_db"] is True

    qa_res = get_vendor_defect_rate("VoltCore")
    assert qa_res["latest_batch_defect_percentage"] == 0.42
    assert qa_res["found_in_db"] is True

    stoppages = list_active_line_stoppages()
    assert len(stoppages) >= 1
    assert any("EV-Pack-V2" in s["impact"] for s in stoppages)

    gh_res = search_open_blockers("BMS-Controller-v4")
    assert gh_res["open_blockers_count"] >= 2
    assert gh_res["lead_engineer"] == "R. Kulkarni"

    detail = get_issue_details("#114")
    assert detail["found"] is True
    assert "clock drift" in detail["title"].lower()


def test_openai_proxy_chat_completions():
    """Validates drop-in OpenAI JSON proxy completions and header injection."""
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

    assert resp.headers.get("X-CalibRAG-Tier") == "tier1_vector"
    assert float(resp.headers.get("X-CalibRAG-Confidence")) >= 0.90
    assert float(resp.headers.get("X-CalibRAG-Cost-Saved-USD")) > 0.005
    assert float(resp.headers.get("X-CalibRAG-Latency-Ms")) > 0


def test_openai_proxy_mid_flight_escalation():
    """Validates mid-flight escalation from Tier 2 to Tier 3 when data is missing from static docs."""
    client = TestClient(app)
    payload = {
        "model": "calibrag-adaptive",
        "messages": [
            {"role": "user", "content": "Why was the EV-Pack-V2 line halted yesterday, and did the supplier breach their SLA penalty window?"}
        ]
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert resp.headers.get("X-CalibRAG-Tier") == "tier3_agentic"
    assert data["calibrag_metadata"]["is_escalated"] is True
    assert "MCP-ERP #402" in data["choices"][0]["message"]["content"]
    assert "VoltCore_SLA" in data["choices"][0]["message"]["content"]


def test_openai_proxy_sse_streaming():
    """Validates Server-Sent Events (SSE) streaming format and [DONE] termination."""
    client = TestClient(app)
    payload = {
        "model": "calibrag-adaptive",
        "stream": True,
        "messages": [
            {"role": "user", "content": "What is the maximum defect tolerance percentage allowed for lithium cell batches?"}
        ]
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("Content-Type", "")

    lines = resp.text.strip().split("\n\n")
    assert len(lines) >= 3

    chunks = []
    has_done = False
    for line in lines:
        if line.startswith("data: "):
            payload_str = line[len("data: "):].strip()
            if payload_str == "[DONE]":
                has_done = True
            else:
                chunk = json.loads(payload_str)
                chunks.append(chunk)

    assert has_done is True
    assert len(chunks) > 0
    assert chunks[0]["object"] == "chat.completion.chunk"
