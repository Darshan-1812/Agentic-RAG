# FastMCP Tool Servers (Member 3: Agentic Workflow & FastMCP Lead)

## 1. Overview
CalibRAG adopts the open **Model Context Protocol (MCP 2026 Spec)** via FastMCP to expose external systems to the Tier 3 LangGraph multi-agent supervisor in under 15 lines of Python.

## 2. Servers Available:
1. **PostgreSQL ERP Server (`postgres_erp/server.py`)**:
   - `query_shipment_delays(batch_number)`: Queries real-time ERP tables for supplier delays and line stoppage.
   - `get_vendor_defect_rate(vendor_name)`: Queries historical quality assurance records.
2. **GitHub Issues Server (`github_issues/server.py`)**:
   - `search_open_blockers(component)`: Queries GitHub issues and PR status to identify assignees, blocker severities, and lead architects.
