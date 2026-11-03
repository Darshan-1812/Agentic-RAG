"""FastAPI Application Server for CalibRAG (Member 3)."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.openai_proxy import router as openai_router

app = FastAPI(
    title="CalibRAG API",
    description="Adaptive Dual-System RAG & Agent Platform (System 1 Jev + System 2 Cascaded RAG)",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-CalibRAG-Tier",
        "X-CalibRAG-Confidence",
        "X-CalibRAG-Latency-Ms",
        "X-CalibRAG-Cost-Saved-USD",
        "X-ACR-Tier-Used",
        "X-ACR-Jev-Confidence",
        "X-ACR-Latency-Ms",
        "X-ACR-Cost-Saved-USD",
    ]
)

# Mount OpenAI-compatible drop-in proxy under /v1
app.include_router(openai_router, prefix="/v1")


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "CalibRAG", "version": "0.2.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
