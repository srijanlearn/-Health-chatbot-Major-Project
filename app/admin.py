# app/admin.py
"""
Admin API router — KB management for HealthyPartner.

Endpoints:
  GET  /admin                          → Admin UI (HTML)
  GET  /admin/kb/stats                 → Record counts per domain per tenant
  POST /admin/kb/upload/medicines      → CSV upload → medicines table
  POST /admin/kb/upload/interactions   → CSV upload → drug_interactions table
  POST /admin/kb/upload/facts          → CSV upload → facts table
  POST /admin/kb/upload/icd10          → CSV upload → icd10_map table
  POST /admin/kb/rebuild               → Rebuild FTS indexes
  POST /admin/kb/reset                 → Clear KB and reload from JSON files

Auth: X-Admin-Key header must match HP_ADMIN_KEY env var.
      If HP_ADMIN_KEY is not set, all admin endpoints are accessible (local dev).
"""

from __future__ import annotations

import csv
import io
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse

from .tenant import validate_tenant_id, DEFAULT_TENANT_ID
from .knowledge.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

_ADMIN_UI_PATH = Path(__file__).parent.parent / "frontend_web" / "admin.html"


# ── Auth dependency ────────────────────────────────────────────────────────────


def _require_admin(x_admin_key: str = Header(default="")):
    """Validate admin key. No-op if HP_ADMIN_KEY is not configured."""
    expected = os.getenv("HP_ADMIN_KEY", "")
    if expected and x_admin_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Admin-Key header")


# ── KG dependency ──────────────────────────────────────────────────────────────


def _get_kg(
    request: Request,
    x_tenant_id: str = Header(default=DEFAULT_TENANT_ID),
    _auth=Depends(_require_admin),
) -> KnowledgeGraph:
    """Resolve the KnowledgeGraph for the target tenant."""
    tm = getattr(request.app.state, "tenant_manager", None)
    if tm is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    if not validate_tenant_id(x_tenant_id):
        raise HTTPException(status_code=400, detail="Invalid X-Tenant-ID")
    orch = tm.get_orchestrator(x_tenant_id)
    if orch.knowledge_graph is None:
        raise HTTPException(status_code=404, detail=f"No knowledge graph for tenant '{x_tenant_id}'")
    return orch.knowledge_graph


# ── CSV parsing helper ─────────────────────────────────────────────────────────


async def _parse_csv(file: UploadFile) -> list[dict[str, Any]]:
    content = await file.read()
    try:
        text = content.decode("utf-8-sig")  # handle BOM from Excel exports
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    reader = csv.DictReader(io.StringIO(text))
    return [row for row in reader]


# ── Admin UI ───────────────────────────────────────────────────────────────────


@router.get("", response_class=HTMLResponse)
async def admin_ui():
    """Serve the admin panel HTML."""
    if _ADMIN_UI_PATH.exists():
        return HTMLResponse(content=_ADMIN_UI_PATH.read_text())
    return HTMLResponse(content="<h1>Admin UI not found</h1><p>Ensure frontend_web/admin.html exists.</p>", status_code=404)


# ── Stats ──────────────────────────────────────────────────────────────────────


@router.get("/kb/stats")
async def kb_stats(
    x_tenant_id: str = Header(default=DEFAULT_TENANT_ID),
    kg: KnowledgeGraph = Depends(_get_kg),
):
    """Return record counts per domain for the tenant's knowledge base."""
    return {
        "tenant_id": x_tenant_id,
        "stats": kg.get_stats(),
    }


# ── CSV Uploads ────────────────────────────────────────────────────────────────


@router.post("/kb/upload/medicines")
async def upload_medicines(
    file: UploadFile = File(..., description="CSV with columns: brand_name, generic_name, generic_name_hi, category, jan_aushadhi_price, market_price, savings_percent, usage"),
    x_tenant_id: str = Header(default=DEFAULT_TENANT_ID),
    kg: KnowledgeGraph = Depends(_get_kg),
):
    """
    Upload a medicines CSV.

    Required columns: brand_name, generic_name
    Optional columns: generic_name_hi, category, jan_aushadhi_price, market_price, savings_percent, usage
    """
    rows = await _parse_csv(file)
    inserted = kg.import_csv_medicines(rows)
    logger.info("Admin: inserted %d medicines for tenant=%s", inserted, x_tenant_id)
    return {"status": "ok", "inserted": inserted, "total_rows": len(rows)}


@router.post("/kb/upload/interactions")
async def upload_interactions(
    file: UploadFile = File(..., description="CSV with columns: drug_a, drug_b, severity, description, recommendation"),
    x_tenant_id: str = Header(default=DEFAULT_TENANT_ID),
    kg: KnowledgeGraph = Depends(_get_kg),
):
    """
    Upload a drug interactions CSV.

    Required columns: drug_a, drug_b, description
    Optional columns: severity (mild/moderate/severe), recommendation
    """
    rows = await _parse_csv(file)
    inserted = kg.import_csv_interactions(rows)
    logger.info("Admin: inserted %d interactions for tenant=%s", inserted, x_tenant_id)
    return {"status": "ok", "inserted": inserted, "total_rows": len(rows)}


@router.post("/kb/upload/facts")
async def upload_facts(
    file: UploadFile = File(..., description="CSV with columns: category_id, category_name, key, value, key_hi, value_hi, source, tags"),
    x_tenant_id: str = Header(default=DEFAULT_TENANT_ID),
    kg: KnowledgeGraph = Depends(_get_kg),
):
    """
    Upload a facts/schemes CSV (IRDAI, PMJAY, hospital policies, etc).

    Required columns: category_id, key, value
    Optional columns: category_name, key_hi, value_hi, source, tags (comma-separated)
    """
    rows = await _parse_csv(file)
    inserted = kg.import_csv_facts(rows)
    logger.info("Admin: inserted %d facts for tenant=%s", inserted, x_tenant_id)
    return {"status": "ok", "inserted": inserted, "total_rows": len(rows)}


@router.post("/kb/upload/icd10")
async def upload_icd10(
    file: UploadFile = File(..., description="CSV with columns: symptom, icd10_code, condition_name, symptom_hi, condition_name_hi, severity, see_doctor_urgency"),
    x_tenant_id: str = Header(default=DEFAULT_TENANT_ID),
    kg: KnowledgeGraph = Depends(_get_kg),
):
    """
    Upload an ICD-10 symptom→condition CSV.

    Required columns: symptom, icd10_code, condition_name
    Optional columns: symptom_hi, condition_name_hi, severity, see_doctor_urgency (immediate/within_24h/within_week/routine)
    """
    rows = await _parse_csv(file)
    inserted = kg.import_csv_icd10(rows)
    logger.info("Admin: inserted %d ICD-10 entries for tenant=%s", inserted, x_tenant_id)
    return {"status": "ok", "inserted": inserted, "total_rows": len(rows)}


# ── Maintenance ────────────────────────────────────────────────────────────────


@router.post("/kb/rebuild")
async def rebuild_fts(
    x_tenant_id: str = Header(default=DEFAULT_TENANT_ID),
    kg: KnowledgeGraph = Depends(_get_kg),
):
    """Rebuild all FTS5 search indexes for the tenant's KB."""
    kg._rebuild_fts()
    return {"status": "ok", "message": "FTS indexes rebuilt"}


@router.post("/kb/reset")
async def reset_kb(
    x_tenant_id: str = Header(default=DEFAULT_TENANT_ID),
    kg: KnowledgeGraph = Depends(_get_kg),
):
    """
    Clear all KB data and reload from the default JSON source files.

    WARNING: This deletes all uploaded data for this tenant.
    Only use to restore from the baseline JSON files.
    """
    kg.reset_and_reload()
    logger.info("Admin: KB reset for tenant=%s", x_tenant_id)
    return {"status": "ok", "message": "KB cleared and reloaded from JSON files", "stats": kg.get_stats()}


# ── Audit endpoints ────────────────────────────────────────────────────────────


def _get_audit_log(request: Request):
    """Resolve the shared AuditLog from app.state."""
    audit = getattr(request.app.state, "audit_log", None)
    if audit is None:
        raise HTTPException(status_code=503, detail="Audit log not initialised")
    return audit


@router.get("/audit/stats")
async def audit_stats(
    request: Request,
    x_tenant_id: str = Header(default=DEFAULT_TENANT_ID),
    _auth=Depends(_require_admin),
):
    """Aggregate query stats for the tenant — totals, route breakdown, intent breakdown."""
    audit = _get_audit_log(request)
    return {"tenant_id": x_tenant_id, "stats": audit.get_stats(tenant_id=x_tenant_id)}


@router.get("/audit/recent")
async def audit_recent(
    request: Request,
    limit: int = 20,
    x_tenant_id: str = Header(default=DEFAULT_TENANT_ID),
    _auth=Depends(_require_admin),
):
    """Most recent queries (no PII — message hash only)."""
    audit = _get_audit_log(request)
    rows = audit.get_recent(limit=min(limit, 100), tenant_id=x_tenant_id)
    return {"tenant_id": x_tenant_id, "rows": rows, "count": len(rows)}
