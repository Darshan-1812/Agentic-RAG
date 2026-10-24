# API Module (Member 3: Agentic Workflow & FastMCP Lead)

## Endpoints:
- `POST /v1/chat/completions`: OpenAI-compatible drop-in proxy.
  - Injects telemetry headers:
    - `X-ACR-Tier-Used`
    - `X-ACR-Jev-Confidence`
    - `X-ACR-Latency-Ms`
    - `X-ACR-Cost-Saved-USD`
- `POST /api/v1/route`: Stage 1 System 1 direct triage endpoint.
- `POST /api/v1/verify`: Stage 6 Post-generation claim verifier endpoint.
