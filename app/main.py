# app/main.py
"""
FastAPI API for HealthyPartner v2.

Changes from previous version:
- Orchestrator is now the single entry point for all inference
  (replaces the manual route_by_keywords + direct LLM calls that bypassed it)
- KnowledgeGraph initialised at startup and passed to Orchestrator
- Added /chat endpoint for conversational use without a document
- Document Q&A endpoint now uses orchestrator.process() for the full 7-step pipeline
"""

import os
import uuid
import base64
from typing import List, Optional
from contextlib import asynccontextmanager

import requests as http_requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .llm_engine import LLMEngine, detect_system
from .orchestrator import Orchestrator
from .ingestion import process_and_get_retriever
from .knowledge.graph import KnowledgeGraph

# ── Configuration ───────────────────────────────────────────────────────────────

DOWNLOAD_PATH = "./downloaded_files"
DB_BASE_PATH = "./db"
KG_DB_PATH = "./data/knowledge.db"

# ── Request / Response models ───────────────────────────────────────────────────


class RunRequest(BaseModel):
    documents: str = Field(..., description="URL or base64-encoded PDF")
    questions: List[str] = Field(..., description="List of questions to answer")
    is_base64: Optional[bool] = Field(default=False)


class RunResponse(BaseModel):
    answers: List[str]


class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = None  # [{"role": "user"|"assistant", "content": "..."}]


class ChatResponse(BaseModel):
    response: str
    intent: str
    route: str
    language: str
    is_emergency: bool
    latency_ms: float


# ── App lifespan ────────────────────────────────────────────────────────────────

engine: Optional[LLMEngine] = None
orchestrator: Optional[Orchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, orchestrator
    print("🏥 HealthyPartner v2 starting up...")
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)
    os.makedirs(DB_BASE_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(KG_DB_PATH), exist_ok=True)

    # Local LLM engine (no API keys needed)
    engine = LLMEngine()
    status = engine.ensure_models_available()
    print(f"   Engine: {engine}")
    print(f"   Models: main={'✅' if status['main_model'] else '❌'} "
          f"fast={'✅' if status['fast_model'] else '❌'}")

    # Knowledge graph (Indian healthcare data — instant lookups, no LLM needed)
    kg = KnowledgeGraph(db_path=KG_DB_PATH)
    print("   Knowledge graph: ✅")

    # Orchestrator wires engine + KG into the 7-step pipeline
    orchestrator = Orchestrator(engine=engine, knowledge_graph=kg)
    print("   Orchestrator: ✅")
    print("   Ready.\n")
    yield
    print("Server shutting down.")


app = FastAPI(
    title="HealthyPartner v2 API",
    description="Privacy-first local healthcare AI — no cloud, no API keys",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Health & system endpoints ───────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health_check():
    """Health check — returns Ollama connectivity, model availability, system info."""
    if engine is None:
        return {"status": "starting", "message": "Engine not yet initialised"}
    return {
        "status": "healthy",
        "message": "HealthyPartner v2 running locally",
        **engine.health_check(),
    }


@app.get("/system/info", tags=["System"])
async def system_info():
    """Return hardware detection and model tier recommendation."""
    info = detect_system()
    result = info.to_dict()
    if engine:
        result["current_tier"] = engine.tier
        result["main_model"] = engine.main_model
        result["fast_model"] = engine.fast_model
        result["local_models"] = engine.list_local_models()
    return result


@app.get("/models", tags=["System"])
async def list_models():
    """List locally available Ollama models."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return {
        "current": {
            "main": engine.main_model,
            "fast": engine.fast_model,
            "tier": engine.tier,
        },
        "available": engine.list_local_models(),
        "status": engine.ensure_models_available(),
    }


# ── Chat endpoint (no document required) ───────────────────────────────────────


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Free-form healthcare chat.

    Runs the full 7-step orchestration pipeline:
    language detection → emergency check → intent classify →
    knowledge graph lookup → routing → retrieval → generation → safety guardrails.
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")

    # Format conversation history as a string for the LLM context window
    history_str = ""
    is_new_user = not bool(request.conversation_history)
    if request.conversation_history:
        lines = []
        for turn in request.conversation_history[-6:]:  # last 3 turns (6 messages)
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role and content:
                lines.append(f"{role.capitalize()}: {content}")
        history_str = "\n".join(lines)

    result = orchestrator.process(
        message=request.message,
        conversation_history=history_str,
        has_document=False,
        is_new_user=is_new_user,
    )

    return ChatResponse(
        response=result.response,
        intent=result.intent,
        route=result.route,
        language=result.language,
        is_emergency=result.is_emergency,
        latency_ms=result.latency_ms,
    )


# ── Document Q&A endpoints ──────────────────────────────────────────────────────


@app.post("/hackrx/run", response_model=RunResponse, tags=["Q&A"])
async def run_submission(request: RunRequest):
    """Backward-compatible endpoint (original hackathon route)."""
    return await _process_document_qa(request)


@app.post("/healthypartner/run", response_model=RunResponse, tags=["Q&A"])
async def healthypartner_run(request: RunRequest):
    """Answer questions about a PDF document using the full orchestration pipeline."""
    return await _process_document_qa(request)


async def _process_document_qa(request: RunRequest) -> RunResponse:
    """
    Core document Q&A logic shared by both endpoints.

    For each question, runs the full orchestrator pipeline with:
    - retrieved_chunks from HybridRetriever (BM25 + vector + cross-encoder rerank)
    - full_document_text for specific fact queries
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")

    # Download or decode document
    local_filename = os.path.join(DOWNLOAD_PATH, str(uuid.uuid4()) + ".pdf")
    try:
        if request.is_base64:
            file_content = base64.b64decode(request.documents)
            with open(local_filename, "wb") as f:
                f.write(file_content)
        else:
            response = http_requests.get(request.documents, timeout=30)
            response.raise_for_status()
            with open(local_filename, "wb") as f:
                f.write(response.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process document: {e}")

    # Ingest document → HybridRetriever + full text
    document_id = os.path.splitext(os.path.basename(local_filename))[0]
    retriever, full_text = process_and_get_retriever(local_filename, document_id)
    if not retriever:
        raise HTTPException(status_code=500, detail="Failed to process document.")

    # Answer each question via orchestrator (full 7-step pipeline)
    answers = []
    for question in request.questions:
        print(f"--- Answering: {question} ---")
        try:
            retrieved_docs = retriever.invoke(question)
            retrieved_chunks = [doc.page_content for doc in retrieved_docs]

            result = orchestrator.process(
                message=question,
                retrieved_chunks=retrieved_chunks,
                full_document_text=full_text,
                has_document=True,
            )
            print(
                f"    Route: {result.route} | Intent: {result.intent} | {result.latency_ms:.0f}ms"
            )
            answers.append(result.response)
        except Exception as e:
            answers.append(f"Error processing question: {e}")
            print(f"    Error: {e}")

    return RunResponse(answers=answers)
