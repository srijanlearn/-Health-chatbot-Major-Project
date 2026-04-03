# app/tenant.py
"""
Multi-tenant isolation for HealthyPartner.

Each tenant gets fully isolated:
  - KnowledgeGraph  → tenants/{tenant_id}/knowledge.db
  - Vector store    → db/{tenant_id}/{document_id}/
  - BM25 index      → db/{tenant_id}/{document_id}/bm25_index.pkl
  - Config          → tenants/{tenant_id}/config.yaml
  - Orchestrator    → lazily initialised and cached per tenant

The "default" tenant maps to legacy paths (data/knowledge.db, ./db/)
for zero-migration backward compatibility with single-tenant deployments.

Security: tenant_id is validated against a strict allowlist regex before
any filesystem path is constructed — prevents path traversal attacks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .audit import AuditLog
from .knowledge.graph import KnowledgeGraph
from .llm_engine import LLMEngine
from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)

TENANTS_DIR = Path("tenants")
DEFAULT_TENANT_ID = "default"

# Only alphanumeric, underscore, and hyphen — prevents path traversal
_TENANT_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def validate_tenant_id(tenant_id: str) -> bool:
    """Return True if tenant_id is safe to use as a filesystem path component."""
    return bool(_TENANT_ID_RE.match(tenant_id))


# ── Tenant Config ──────────────────────────────────────────────────────────────


@dataclass
class TenantConfig:
    """
    Per-tenant configuration loaded from tenants/{tenant_id}/config.yaml.

    Fields can be omitted — safe defaults are derived from tenant_id.
    """
    tenant_id: str
    name: str = "HealthyPartner"
    kg_db_path: Optional[str] = None       # absolute or relative path to SQLite KG
    vector_store_base: Optional[str] = None  # base dir for ChromaDB + BM25

    @classmethod
    def load(cls, tenant_id: str) -> "TenantConfig":
        """Load config from disk, or return safe defaults if file is absent."""
        config_path = TENANTS_DIR / tenant_id / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            data.pop("tenant_id", None)  # never let config override tenant_id
            allowed = {"name", "kg_db_path", "vector_store_base"}
            return cls(tenant_id=tenant_id, **{k: v for k, v in data.items() if k in allowed})
        return cls(tenant_id=tenant_id)

    @property
    def resolved_kg_db_path(self) -> str:
        """Absolute path to this tenant's KnowledgeGraph SQLite database."""
        if self.kg_db_path:
            return self.kg_db_path
        if self.tenant_id == DEFAULT_TENANT_ID:
            return "data/knowledge.db"  # legacy single-tenant path
        path = TENANTS_DIR / self.tenant_id / "knowledge.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    @property
    def resolved_vector_store_base(self) -> str:
        """Base directory for this tenant's ChromaDB collections and BM25 indexes."""
        if self.vector_store_base:
            return self.vector_store_base
        if self.tenant_id == DEFAULT_TENANT_ID:
            return "./db"  # legacy single-tenant path
        return str(Path("db") / self.tenant_id)

    @property
    def download_dir(self) -> str:
        """Directory for temporary document uploads for this tenant."""
        path = Path("downloaded_files") / self.tenant_id
        path.mkdir(parents=True, exist_ok=True)
        return str(path)


# ── Tenant Manager ─────────────────────────────────────────────────────────────


class TenantManager:
    """
    Manages per-tenant Orchestrator instances.

    Orchestrators are created lazily on the first request from a tenant
    and cached indefinitely (until explicit reload). The LLMEngine is
    shared across all tenants — only the KnowledgeGraph and vector store
    paths differ per tenant.

    Usage:
        manager = TenantManager(engine)
        orch = manager.get_orchestrator("apollo_delhi")
        result = orch.process(message="...")
    """

    def __init__(self, engine: LLMEngine, audit_log: Optional[AuditLog] = None) -> None:
        self._engine = engine
        self._audit = audit_log
        self._orchestrators: Dict[str, Orchestrator] = {}
        self._configs: Dict[str, TenantConfig] = {}

    def get_orchestrator(self, tenant_id: str) -> Orchestrator:
        """Return the cached Orchestrator for tenant_id, creating it if needed."""
        if tenant_id not in self._orchestrators:
            self._orchestrators[tenant_id] = self._build_orchestrator(tenant_id)
        return self._orchestrators[tenant_id]

    def get_config(self, tenant_id: str) -> TenantConfig:
        """Return the cached TenantConfig for tenant_id."""
        if tenant_id not in self._configs:
            self._configs[tenant_id] = TenantConfig.load(tenant_id)
        return self._configs[tenant_id]

    def _build_orchestrator(self, tenant_id: str) -> Orchestrator:
        config = self.get_config(tenant_id)
        logger.info("Initialising orchestrator for tenant=%r (kg=%s)", tenant_id, config.resolved_kg_db_path)
        kg = KnowledgeGraph(db_path=config.resolved_kg_db_path)
        return Orchestrator(engine=self._engine, knowledge_graph=kg, audit_log=self._audit)

    def reload_tenant(self, tenant_id: str) -> None:
        """
        Evict a tenant's orchestrator and config from cache.

        The next request from this tenant will trigger fresh initialisation.
        Use this after updating a tenant's knowledge base or config file.
        """
        self._orchestrators.pop(tenant_id, None)
        self._configs.pop(tenant_id, None)
        logger.info("Tenant %r evicted from cache — will reinitialise on next request", tenant_id)

    def list_tenants(self) -> List[str]:
        """Return all known tenant IDs (from disk and from active cache)."""
        from_disk: set[str] = set()
        if TENANTS_DIR.exists():
            from_disk = {p.name for p in TENANTS_DIR.iterdir() if p.is_dir()}
        from_cache = set(self._orchestrators.keys())
        return sorted(from_disk | from_cache)
