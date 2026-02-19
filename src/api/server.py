"""
server.py
---------
FastAPI application exposing the RAG pipeline as a REST API.
Clients send a query + sender_id and receive a generated answer.

Run with:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8001 --reload
"""

import os
import logging

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.retrieval.retriever import generate_answer

# ── Force CPU-only mode for Ollama ────────────────────────────────────────────
os.environ.setdefault("OLLAMA_NO_GPU", "1")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Document Assistant API",
    description="Query your documents using a local LLM (Ollama) and FAISS vector search.",
    version="1.0.0"
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query:     str
    sender_id: str

class QueryResponse(BaseModel):
    answer: str | None
    error:  str | None = None


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    gpu_mode = os.environ.get("OLLAMA_NO_GPU")
    if gpu_mode == "1":
        logger.info("🧠 Ollama running in CPU-only mode (OLLAMA_NO_GPU=1).")
    else:
        logger.warning("⚠️ OLLAMA_NO_GPU is not set — Ollama may use GPU.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def handle_query(req: QueryRequest):
    """
    Accept a natural-language query and return an LLM-generated answer
    grounded in the ingested document context.
    """
    logger.info(f"📥 Query from '{req.sender_id}': {req.query}")

    answer = await run_in_threadpool(generate_answer, req.query, req.sender_id)

    if answer is None:
        logger.error("❌ Answer generation returned None.")
        raise HTTPException(status_code=500, detail="Failed to generate an answer.")

    return QueryResponse(answer=answer)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting RAG Document Assistant API...")
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8001, reload=False)
