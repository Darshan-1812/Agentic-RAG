# ACR-RAG: Adaptive Control & Retrieval Platform
### Dual-System (System 1 Jev + System 2 Agentic RAG) Cost-Aware, Self-Evaluating AI Platform
**Department of AI & Data Science, VIIT Pune | Major Project (2026–27)**

---

## 1. System Overview
ACR-RAG replaces monolithic, expensive RAG pipelines with an ultra-fast **System 1 Calibrated Control Plane** (TypeSafe AI's Jev) that dynamically governs when and how an expensive **System 2 Retrieval & Multi-Agent Pipeline** (LangGraph + LightRAG + FastMCP) is invoked—cutting token costs by 60–80%, eliminating unnecessary latency, and cleanly refusing to answer when verified evidence is absent.

```
Incoming Query + Workspace SLA Policy
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Pre-Retrieval System 1 Triage (~90 ms)         │
│   • Choice(["tier1_vector", "tier2_hybrid", "tier3"])   │
│   • Noul("Requires live database, issues, or tools?")    │
│   • Score(rubric="1: Fact -> 3: Comparison -> 5: Graph")│
└────────────────┬────────────────────────────────────────┘
                 │
                 ├── Low Complexity ────────► Tier 1: Fast Dense Vector RAG (pgvector HNSW)
                 │
                 ├── Medium Complexity ─────► Tier 2: Hybrid BM25 (ParadeDB) + Vector + Cross-Encoder
                 │
                 └── High Complexity ───────► Tier 3: Agentic RAG (LangGraph + LightRAG + FastMCP)
                                                ▲
                                                │ (Mid-Flight Escalation if C_evidence < 0.65)
┌───────────────────────────────────────────────┴─────────┐
│ Stage 3: Mid-Pipeline Evidence Sufficiency Gate         │
│   C_evidence = 0.50*P_Noul + 0.30*σ(ΔL) + 0.20*Cov_ent  │
└────────────────┬────────────────────────────────────────┘
                 │
                 ├── C_evidence < tau_abstain (0.50) ─────► Stage 5: Calibrated Abstention (0% Hallucination)
                 │
                 └── C_evidence >= tau_escalate ──────────► Stage 6: Post-Gen Claim Verifier & Telemetry
                                                            (Parallel Sentence-Level Noul Checks)
```

---

## 2. Team Architecture & 4-Member Responsibility Split

| Member | Role | Core Modules Owned | Key Technologies & Files |
| :--- | :--- | :--- | :--- |
| **Member 1** *(Current)* | **Control Plane & Router Lead** | Pre-Retrieval Jev Router, Mid-Pipeline Confidence Gate ($C_{evidence}$), Post-Gen Noul Verifier & Abstention Engine | TypeSafe AI Jev SDK, RouteLLM/DeBERTa fallback, calibration formulas (`router/`, `verifier/`) |
| **Member 2** | **Data, Hybrid RAG & GraphRAG Lead** | BYOK Ingestion Pipeline, Unified Postgres Vector + BM25 RRF Search, Cross-Encoder Reranking, LightRAG Graph Indexer | IBM Docling, PostgreSQL 17, `pgvector`, ParadeDB `pg_search`, `bge-m3`, LightRAG (`retrieval/`) |
| **Member 3** | **Agentic Workflow & FastMCP Lead** | Tier 3 LangGraph Supervisor, Read-Only Postgres SQL MCP Server, GitHub Issues MCP Server, OpenAI Proxy API | LangGraph, FastMCP, FastAPI SSE streaming, Pydantic v2 (`agents/`, `mcp_servers/`, `api/`) |
| **Member 4** | **Product UI, Control Room & Eval Lead** | Next.js BYOK Studio, Split-Screen Shadow Mode Playground, Agent Control Room Dashboard, Auto-Benchmark Suite | Next.js 15, Tailwind, shadcn/ui, Recharts, Arize Phoenix, RAGAS (`frontend/`, `evals/`) |

---

## 3. Member 1 Implemented Modules

### 3.1 Pre-Retrieval System 1 Router (`router/`)
- **Parallel Primitives**: Evaluates `Choice`, `Noul`, and `Score` concurrently in ~90 ms.
- **SLA Governance Policies**:
  - `Economy`: Biases toward Tier 1; escalates only when $C_{evidence} < 0.35$.
  - `Balanced`: Standard enterprise tiering; escalates when $< 0.65$; abstains when $< 0.50$.
  - `Strict Zero-Hallucination`: Conservative routing; requires sentence Noul $\ge 0.88$.
- **Fallback Router (`router/fallback_router.py`)**: Local calibrated classifier fallback ensuring zero downtime if external APIs are unreachable.

### 3.2 Mid-Pipeline Confidence Gate (`verifier/confidence_gate.py`)
Computes the blueprint composite evidence confidence formula:
$$C_{evidence} = 0.50 \cdot P_{\text{Jev\_Noul}}(\text{Sufficient}) + 0.30 \cdot \sigma(\Delta L_{\text{rerank}}) + 0.20 \cdot \text{Cov}_{\text{entities}}$$
Triggers mid-flight escalation to Tier 2 or Tier 3 if $C_{evidence} < \tau_{escalate}$.

### 3.3 Post-Generation Sentence-Level Noul Verifier (`verifier/claim_verifier.py`)
- Evaluates individual sentences against cited chunks or tool outputs in parallel.
- Strips hallucinated claims and ensures grounding before responses stream to the client.

### 3.4 Calibrated Abstention Engine (`verifier/abstention_engine.py`)
- If final post-retrieval evidence score remains below $\tau_{abstain}$ (0.50), halts text generation and returns a structured `InsufficientEvidenceResponse` enumerating missing facts and sources.

### 3.5 Research Paper Calibration Evaluation (`verifier/calibration_metrics.py`, `evals/run_calibration_eval.py`)
- Implements Expected Calibration Error (ECE), Brier Score, Maximum Calibration Error (MCE), and reliability diagrams for Section 3 of the research paper.

---

## 4. Running Tests and Calibration Evaluation

### Run Test Suite
```bash
pytest tests/ -v
```

### Run Calibration Evaluation & Generate Paper Curves
```bash
python3 -m evals.run_calibration_eval
```
