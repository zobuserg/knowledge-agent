"""
Knowledge Agent API
-------------------
FastAPI server exposing the ingestion and query pipelines as REST endpoints.

Endpoints:
    POST /ingest        → upload and index a document
    POST /query         → ask a question, get an answer with citations
    GET  /documents     → list all indexed documents
    GET  /health        → check if the service is running
"""

from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.ingestion.pipeline import ingest_single_file, get_vector_store
from app.query.agent import query
from app.config import settings

app = FastAPI(
    title="Knowledge Agent",
    description="Agentic RAG — ask questions about your documents",
    version="0.1.0",
)


# ── Request/Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"   # optional: for multi-turn conversations

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick check that the server is up."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a document and add it to the knowledge base.
    Accepts: PDF, Markdown (.md), plain text (.txt)
    """
    allowed = {".pdf", ".md", ".txt"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed}"
        )

    # Save uploaded file to data/raw/ temporarily
    dest = Path("data/raw") / file.filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run ingestion pipeline
    result = ingest_single_file(dest)

    return JSONResponse(content={
        "message": f"Successfully ingested: {file.filename}",
        "stats": result,
    })


@app.post("/query", response_model=QueryResponse)
def ask(request: QueryRequest):
    """
    Ask a question. The agent searches the knowledge base and
    returns an answer with citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = query(
        question=request.question,
        session_id=request.session_id,
    )
    return result


@app.get("/documents")
def list_documents():
    """List all documents currently in the knowledge base."""
    collection, _ = get_vector_store()
    count = collection.count()

    # Get unique filenames from metadata
    if count == 0:
        return {"total_chunks": 0, "documents": []}

    results = collection.get(include=["metadatas"])
    files = set()
    for meta in results["metadatas"]:
        fname = meta.get("file_name") or meta.get("file_path", "unknown")
        files.add(fname)

    return {
        "total_chunks": count,
        "documents": sorted(files),
    }
