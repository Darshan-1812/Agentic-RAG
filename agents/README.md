# Agents Module (Member 3: Agentic Workflow & FastMCP Lead)

## 1. Overview
The `agents/` module implements CalibRAG's Tier 3 System 2 Multi-Agent Orchestration using LangGraph and the FastMCP protocol.

## 2. Core Responsibilities Owned:
1. **Tier 3 LangGraph Supervisor (`langgraph_supervisor.py`)**:
   - Decomposes complex multi-hop queries into atomic sub-goals.
   - Provides cyclic state control for mid-flight escalation triggered by Member 1's Confidence Gate ($C_{evidence} < \tau_{escalate}$).
   - Traverses 2-hop entity links in Member 2's LightRAG knowledge graph.
   - Invokes FastMCP tool servers dynamically.
2. **Research Paper Section 5 Contribution**:
   - Authors Section 5: *Agentic Escalation & MCP Tool Orchestration*.
   - Evaluates multi-hop query resolution and tool invocation latency vs. monolithic chains.
