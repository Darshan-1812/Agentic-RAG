# Agents Module (Member 3: Agentic Workflow & FastMCP Lead)

## Core Responsibilities:
- **Tier 3 LangGraph Supervisor**: Cyclic multi-agent supervisor orchestrating sub-goals, LightRAG graph lookups, and MCP tool execution.
- **Mid-Flight Escalation Receiver**: Triggered when Member 1's Mid-Pipeline Confidence Gate determines $C_{evidence} < \tau_{escalate}$.
- **FastMCP Tool Integration**: Read-only PostgreSQL ERP database server and GitHub Issues/PRs server.
- **OpenAI Proxy API**: `POST /v1/chat/completions` drop-in proxy with streaming SSE and telemetry header injection.
