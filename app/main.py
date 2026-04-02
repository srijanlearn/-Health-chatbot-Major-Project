# app/main.py
"""
FastAPI RAG endpoint for HealthyPartner v2.

Changes from v1:
- Removed Google Gemini dependency entirely
- All inference runs locally via LLMEngine (Ollama)
- Added system info and model management endpoints
- Kept dual-path architecture (full text for specific facts, RAG for general)
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
from .prompts.insurance_qa import (
    INSURANCE_SYSTEM_PROMPT,
    INSURANCE_SPECIFIC_FACT_PROMPT,
    INSURANCE_GENERAL_PROMPT,
)
from .prompts.router import ROUTER_SYSTEM_PROMPT, route_by_keywords

# ── Configuration ──────────────────────────────────────────────────────────────

DOWNLOAD_PATH = "./downloaded_files"
DB_BASE_PATH = "./db"

# ── Request / Response models ──────────────────────────────────────────────────


class RunRequest(BaseModel):
    documents: str = Field(..., description="URL or base64-encoded PDF")
    questions: List[str] = Field(..., description="List of questions to answer")
    is_base64: Optional[bool] = Field(default=False)


class RunResponse(BaseModel):
    answers: List[str]


# ── App lifespan ───────────────────────────────────────────────────────────────

engine: Optional[LLMEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("🏥 HealthyPartner v2 starting up...")
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)
    os.makedirs(DB_BASE_PATH, exist_ok=True)

    # Initialise local LLM engine (no API keys needed)
    engine = LLMEngine()
    status = engine.ensure_models_available()
    print(f"   Engine: {engine}")
    print(f"   Models: main={'✅' if status['main_model'] else '❌'} "
          f"fast={'✅' if status['fast_model'] else '❌'}")
    print("   Ready.\n")
    yield
    print("Server shutting down.")


app = FastAPI(
    title="HealthyPartner v2 API",
    description="Privacy-first local healthcare AI — no cloud, no API keys",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Health & system endpoints ──────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health_check():
    """Health check — returns system and model status."""
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
    """List locally available models."""
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


# ── Main RAG endpoint ─────────────────────────────────────────────────────────


@app.post("/hackrx/run", response_model=RunResponse, tags=["Q&A"])
async def run_submission(request: RunRequest):
    """
    Answer questions about a PDF document.

    Kept at /hackrx/run for backward compatibility.
    Also available at /healthypartner/run.
    """
    return await _process_document_qa(request)


@app.post("/healthypartner/run", response_model=RunResponse, tags=["Q&A"])
async def healthypartner_run(request: RunRequest):
    """Answer questions about a PDF document."""
    return await _process_document_qa(request)


async def _process_document_qa(request: RunRequest) -> RunResponse:
    """Core document Q&A logic shared by both endpoints."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")

    document_data = request.documents
    questions = request.questions
    is_base64 = request.is_base64

    # Download or decode document
    local_filename = os.path.join(DOWNLOAD_PATH, str(uuid.uuid4()) + ".pdf")
    try:
        if is_base64:
            file_content = base64.b64decode(document_data)
            with open(local_filename, "wb") as f:
                f.write(file_content)
        else:
            response = http_requests.get(document_data, timeout=30)
            response.raise_for_status()
            with open(local_filename, "wb") as f:
                f.write(response.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process document: {e}")

    # Ingest document
    document_id = os.path.splitext(os.path.basename(local_filename))[0]
    retriever, full_text = process_and_get_retriever(local_filename, document_id)
    if not retriever:
        raise HTTPException(status_code=500, detail="Failed to process document.")

    # Answer each question
    answers = []
    for question in questions:
        print(f"--- Answering: {question} ---")
        try:
            # Route question
            route = route_by_keywords(question)
            if route is None:
                route_raw = engine.classify(question, system_prompt=ROUTER_SYSTEM_PROMPT)
                route = "specific_fact" if "specific" in route_raw.lower() else "general_rag"
            print(f"    Route: {route}")

            if route == "specific_fact":
                # Path A: Full document context for precise extraction
                doc_context = full_text[:12_000]
                prompt = INSURANCE_SPECIFIC_FACT_PROMPT.format(
                    context=doc_context, question=question
                )
                answer = engine.generate(
                    prompt=prompt,
                    system_prompt=INSURANCE_SYSTEM_PROMPT,
                    think=True,
                )
            else:
                # Path B: RAG with retrieved chunks
                docs = retriever.invoke(question)
                chunk_context = "\n\n".join(doc.page_content for doc in docs[:5])
                prompt = INSURANCE_GENERAL_PROMPT.format(
                    context=chunk_context, question=question
                )
                answer = engine.generate(
                    prompt=prompt,
                    system_prompt=INSURANCE_SYSTEM_PROMPT,
                )

            answers.append(answer)
        except Exception as e:
            answers.append(f"Error processing question: {e}")
            print(f"    Error: {e}")

    return RunResponse(answers=answers)
