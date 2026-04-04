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

# Load .env before any module reads os.environ
try:
    from dotenv import load_dotenv
    from pathlib import Path as _Path
    load_dotenv(_Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import requests as http_requests
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audit import AuditLog
from .llm_engine import LLMEngine, detect_system
from .tenant import TenantManager, TenantConfig, validate_tenant_id, DEFAULT_TENANT_ID
from .ingestion import process_and_get_retriever
from .admin import router as admin_router
from .session_manager import SessionManager

# ── Configuration ───────────────────────────────────────────────────────────────

DOWNLOAD_PATH = "./downloaded_files"

# ── Request / Response models ───────────────────────────────────────────────────


class RunRequest(BaseModel):
    documents: str = Field(..., description="URL or base64-encoded PDF")
    questions: List[str] = Field(..., description="List of questions to answer")
    is_base64: Optional[bool] = Field(default=False)


class RunResponse(BaseModel):
    answers: List[str]


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    conversation_history: Optional[List[dict]] = None  # [{"role": "user"|"assistant", "content": "..."}]


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    route: str
    language: str
    is_emergency: bool
    confidence: float
    latency_ms: float


class GenerateRequest(BaseModel):
    prompt: str
    n: int = Field(default=5, ge=1, le=20)


class GenerateResponse(BaseModel):
    questions: List[str]


# ── App lifespan ────────────────────────────────────────────────────────────────

engine: Optional[LLMEngine] = None
tenant_manager: Optional[TenantManager] = None
session_manager: Optional[SessionManager] = None


def _warm_recent_caches(session_store: SessionManager, manager: TenantManager) -> None:
    """Hydrate per-tenant response caches from persisted session history."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    for pair in session_store.recent_query_pairs(limit=200):
        grouped.setdefault(pair["tenant_id"], []).append((pair["message"], pair["response"]))

    for tenant_id, entries in grouped.items():
        manager.get_orchestrator(tenant_id).warm_response_cache(entries)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, tenant_manager, session_manager
    print("🏥 HealthyPartner v2 starting up...")
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    # Audit log — shared across all tenants, single SQLite file
    audit_log = AuditLog()
    app.state.audit_log = audit_log
    print("   AuditLog: ✅")

    # Local LLM engine — shared across all tenants
    engine = LLMEngine()
    status = engine.ensure_models_available()
    print(f"   Engine: {engine}")
    print(f"   Models: main={'✅' if status['main_model'] else '❌'} "
          f"fast={'✅' if status['fast_model'] else '❌'}")

    # Tenant manager — one Orchestrator + KG per tenant, lazily initialised
    tenant_manager = TenantManager(engine=engine, audit_log=audit_log)
    session_manager = SessionManager()
    app.state.session_manager = session_manager
    # Warm up the default tenant so the first real request isn't slow
    tenant_manager.get_orchestrator(DEFAULT_TENANT_ID)
    _warm_recent_caches(session_manager, tenant_manager)
    # Expose via app.state so admin router can access without circular imports
    app.state.tenant_manager = tenant_manager
    print(f"   TenantManager: ✅ (default tenant warmed up)")
    print(f"   SessionManager: ✅ ({session_manager.get_active_session_count()} active in memory)")
    print("   Ready.\n")
    yield
    print("Server shutting down.")


app = FastAPI(
    title="HealthyPartner v2 API",
    description="Privacy-first local healthcare AI — no cloud, no API keys",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — open to all origins (local dev only; restrict before any network-exposed deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Admin router — KB management at /admin/*
app.include_router(admin_router)

# Serve the web frontend at /app (e.g. http://localhost:8000/app/)
_FRONTEND_DIR = str(_Path(__file__).parent.parent / "frontend_web")
if _Path(_FRONTEND_DIR).exists():
    app.mount("/app", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")


# ── Tenant dependency ───────────────────────────────────────────────────────────


def get_orchestrator(x_tenant_id: str = Header(default=DEFAULT_TENANT_ID)):
    """
    FastAPI dependency that resolves the correct Orchestrator for the request.

    Reads the X-Tenant-ID header (defaults to "default" for backward compat).
    Returns 400 on invalid tenant_id, 503 if the server isn't ready yet.
    """
    if tenant_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    if not validate_tenant_id(x_tenant_id):
        raise HTTPException(status_code=400, detail="Invalid X-Tenant-ID header")
    return tenant_manager.get_orchestrator(x_tenant_id)


def get_tenant_config(x_tenant_id: str = Header(default=DEFAULT_TENANT_ID)) -> TenantConfig:
    """Resolves the TenantConfig for the request."""
    if tenant_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    if not validate_tenant_id(x_tenant_id):
        raise HTTPException(status_code=400, detail="Invalid X-Tenant-ID header")
    return tenant_manager.get_config(x_tenant_id)


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
async def chat(
    request: ChatRequest,
    orchestrator=Depends(get_orchestrator),
    x_tenant_id: str = Header(default=DEFAULT_TENANT_ID),
):
    """
    Free-form healthcare chat.

    Runs the full 7-step orchestration pipeline:
    language detection → emergency check → intent classify →
    knowledge graph lookup → routing → retrieval → generation → safety guardrails.

    Pass X-Tenant-ID header to route to a specific tenant's knowledge base.
    Defaults to "default" tenant for backward compatibility.
    """
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not ready")

    session_id = request.session_id or uuid.uuid4().hex
    session = session_manager.get_or_create_session(session_id, tenant_id=x_tenant_id)
    if request.conversation_history and not session["conversation_history"]:
        session_manager.import_conversation_history(
            session_id,
            request.conversation_history,
            tenant_id=x_tenant_id,
        )

    history_str = session_manager.format_history_for_llm(session_id, last_n=6)
    is_new_user = len(session_manager.get_conversation_history(session_id)) == 0

    result = orchestrator.process(
        message=request.message,
        conversation_history=history_str,
        has_document=False,
        is_new_user=is_new_user,
        tenant_id=x_tenant_id,
        session_id=session_id,
    )

    session_manager.add_message(session_id, "user", request.message)
    session_manager.add_message(session_id, "assistant", result.response)

    return ChatResponse(
        session_id=session_id,
        response=result.response,
        intent=result.intent,
        route=result.route,
        language=result.language,
        is_emergency=result.is_emergency,
        confidence=round(result.confidence, 2),
        latency_ms=result.latency_ms,
    )


# ── Document Q&A endpoints ──────────────────────────────────────────────────────


@app.post("/hackrx/run", response_model=RunResponse, tags=["Q&A"])
async def run_submission(
    request: RunRequest,
    orchestrator=Depends(get_orchestrator),
    tenant_config: TenantConfig = Depends(get_tenant_config),
):
    """Backward-compatible endpoint (original hackathon route)."""
    return await _process_document_qa(request, orchestrator, tenant_config)


@app.post("/healthypartner/run", response_model=RunResponse, tags=["Q&A"])
async def healthypartner_run(
    request: RunRequest,
    orchestrator=Depends(get_orchestrator),
    tenant_config: TenantConfig = Depends(get_tenant_config),
):
    """Answer questions about a PDF document using the full orchestration pipeline."""
    return await _process_document_qa(request, orchestrator, tenant_config)


async def _process_document_qa(request: RunRequest, orchestrator, tenant_config: TenantConfig) -> RunResponse:
    """
    Core document Q&A logic shared by both endpoints.

    Documents are stored under the tenant's download directory.
    Vector stores and BM25 indexes are scoped to the tenant's vector_store_base.
    """
    # Store uploaded document under tenant-scoped directory
    local_filename = os.path.join(tenant_config.download_dir, str(uuid.uuid4()) + ".pdf")
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

    # Ingest into tenant-scoped vector store
    document_id = os.path.splitext(os.path.basename(local_filename))[0]
    retriever, full_text = process_and_get_retriever(
        local_filename,
        document_id,
        base_path=tenant_config.resolved_vector_store_base,
    )
    if not retriever:
        raise HTTPException(status_code=500, detail="Failed to process document.")

    # Answer each question via orchestrator (full 7-step pipeline)
    answers = []
    for question in request.questions:
        print(f"--- Answering [{tenant_config.tenant_id}]: {question} ---")
        try:
            retrieved_docs = retriever.invoke(question)
            retrieved_chunks = [doc.page_content for doc in retrieved_docs]

            result = orchestrator.process(
                message=question,
                retrieved_chunks=retrieved_chunks,
                full_document_text=full_text,
                has_document=True,
            )
            print(f"    Route: {result.route} | Intent: {result.intent} | {result.latency_ms:.0f}ms")
            answers.append(result.response)
        except Exception as e:
            answers.append(f"Error processing question: {e}")
            print(f"    Error: {e}")

    return RunResponse(answers=answers)


# ── Question generation ─────────────────────────────────────────────────────────


@app.post("/healthypartner/generate", response_model=GenerateResponse, tags=["Q&A"])
async def generate_questions(req: GenerateRequest):
    """Generate question templates from a short topic prompt."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    if not req.prompt.strip():
        return GenerateResponse(questions=[f"Generic question #{i + 1}" for i in range(req.n)])

    sys_prompt = (
        f"Generate exactly {req.n} specific questions someone might ask about "
        f"a health insurance policy related to: {req.prompt}. "
        f"Return only the questions, one per line, numbered."
    )
    raw = engine.generate(prompt=f"Topic: {req.prompt}", system_prompt=sys_prompt)
    questions = [
        line.strip().lstrip("0123456789.-) ")
        for line in raw.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ][: req.n]

    return GenerateResponse(questions=questions)


# ── Tenant management endpoints ─────────────────────────────────────────────────


@app.get("/tenants", tags=["Tenants"])
async def list_tenants():
    """List all registered tenants."""
    if tenant_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    return {"tenants": tenant_manager.list_tenants()}


@app.post("/tenants/{tenant_id}/reload", tags=["Tenants"])
async def reload_tenant(tenant_id: str):
    """
    Evict a tenant's orchestrator from cache and force re-initialisation.

    Call this after updating a tenant's knowledge base or config file.
    """
    if tenant_manager is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    if not validate_tenant_id(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant_id")
    tenant_manager.reload_tenant(tenant_id)
    return {"status": "reloaded", "tenant_id": tenant_id}


# ── Integrations ────────────────────────────────────────────────────────────────


@app.post("/webhook", tags=["Integrations"])
async def webhook(request: Request):
    """
    Twilio WhatsApp/SMS webhook entrypoint (Phase 7 — stub).

    Accepts both form-encoded (Twilio default) and JSON payloads.
    Full implementation deferred until GAP-010.
    """
    try:
        data = await request.json()
    except Exception:
        form = await request.form()
        data = dict(form)
    return {"status": "received", "data": data}


# ── Debug ───────────────────────────────────────────────────────────────────────


@app.post("/test", tags=["System"])
async def test_endpoint(request: Request):
    """Echo endpoint for local debugging — returns request body + engine info."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    return {
        "ok": True,
        "echo": body,
        "engine": str(engine),
        "tier": engine.tier if engine else None,
    }
