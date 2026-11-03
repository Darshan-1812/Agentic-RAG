# Frontend Module (Member 4: BYOK Studio UI & Benchmark Lead)

## 1. Overview
The `frontend/` package provides CalibRAG's Next.js 15 product interface designed to turn complex agentic RAG into an accessible self-serve developer experience.

## 2. Core Views Owned:
1. **3-Click BYOK Workspace Studio (`components/BYOKWorkspaceStudio.tsx`)**:
   - Workspace creation and isolated tenant configuration.
   - Drag-and-drop document upload (parsed via IBM Docling into pgvector, BM25, and LightRAG).
   - FastMCP connector toggles (PostgreSQL ERP & GitHub issues).
2. **Split-Screen Comparison Playground (`components/SplitScreenPlayground.tsx`)**:
   - Side-by-side execution: **Standard Fixed RAG vs. CalibRAG Adaptive Routing**.
   - Live telemetry indicators: minimum viable tier chosen, confidence score ($C_{evidence}$), token dollars saved, and exact inline citations.
3. **Embeddable JS Widget**:
   - Lightweight script allowing client websites or internal wikis to embed the CalibRAG chat experience via the drop-in proxy API.
