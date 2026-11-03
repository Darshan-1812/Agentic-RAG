# CalibRAG API Module (Member 3: Agentic Workflow & FastMCP Lead)

## 1. Overview
The `api/` module provides a drop-in OpenAI-compatible proxy interface (`POST /v1/chat/completions`) allowing any existing application, chatbot, or workflow to gain adaptive System 1 routing, confidence gating, and cost savings without changing SDKs.

## 2. Injected CalibRAG Response Headers:
- `X-CalibRAG-Tier`: The minimum viable tier executed (`tier1_vector`, `tier2_hybrid`, `tier3_agentic`).
- `X-CalibRAG-Confidence`: Mathematical evidence confidence score ($C_{evidence}$).
- `X-CalibRAG-Latency-Ms`: Total round-trip latency in milliseconds.
- `X-CalibRAG-Cost-Saved-USD`: Exact dollars saved compared to running Always-On Agentic RAG.

## 3. Running the Server:
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```
