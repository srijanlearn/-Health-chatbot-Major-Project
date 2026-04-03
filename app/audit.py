"""
Audit logging — every query pipeline run logged to SQLite.

No PII is stored: the message is hashed (SHA-256, first 16 hex chars).
The log is safe to export, inspect, or hand to a customer without
revealing what any individual actually typed.

Schema
------
query_log(id, tenant_id, session_id, message_hash, intent, route,
          latency_ms, timestamp)

Usage
-----
    audit = AuditLog()                     # or AuditLog("path/to/audit.db")
    audit.log_query(
        tenant_id="default",
        message="original user text",      # hashed internally
        intent="insurance_query",
        route="knowledge_graph",
        latency_ms=42.5,
        session_id="abc123",               # optional
    )
    stats = audit.get_stats(tenant_id="default")
    rows  = audit.get_recent(limit=20, tenant_id="default")
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

AUDIT_DB_PATH = "data/audit.db"

_DDL = """
CREATE TABLE IF NOT EXISTS query_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id    TEXT    NOT NULL,
    session_id   TEXT,
    message_hash TEXT    NOT NULL,
    intent       TEXT,
    route        TEXT,
    latency_ms   REAL,
    timestamp    TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ql_tenant ON query_log(tenant_id);
CREATE INDEX IF NOT EXISTS idx_ql_ts     ON query_log(timestamp);
"""


class AuditLog:
    """Thread-safe, append-only query audit log backed by SQLite."""

    def __init__(self, db_path: str = AUDIT_DB_PATH) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        logger.info("AuditLog initialised at %s", db_path)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_DDL)

    @staticmethod
    def _hash_message(message: str) -> str:
        """One-way hash — first 16 hex chars of SHA-256."""
        return hashlib.sha256(message.lower().strip().encode()).hexdigest()[:16]

    # ── Write ──────────────────────────────────────────────────────────────────

    def log_query(
        self,
        *,
        tenant_id: str,
        message: str,
        intent: Optional[str] = None,
        route: Optional[str] = None,
        latency_ms: float = 0.0,
        session_id: Optional[str] = None,
    ) -> None:
        """Append one row to the audit log.  Never raises — failures are logged."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """INSERT INTO query_log
                               (tenant_id, session_id, message_hash, intent,
                                route, latency_ms, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            tenant_id,
                            session_id,
                            self._hash_message(message),
                            intent,
                            route,
                            round(latency_ms, 2),
                            ts,
                        ),
                    )
            except Exception:
                logger.exception("AuditLog.log_query failed — query not recorded")

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_stats(self, tenant_id: Optional[str] = None) -> Dict:
        """Aggregate stats.  Pass tenant_id to filter to one tenant."""
        where = "WHERE tenant_id = ?" if tenant_id else ""
        params: tuple = (tenant_id,) if tenant_id else ()

        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM query_log {where}", params
            ).fetchone()[0]

            by_route = {
                r["route"] or "unknown": r["cnt"]
                for r in conn.execute(
                    f"SELECT route, COUNT(*) as cnt FROM query_log {where} GROUP BY route",
                    params,
                ).fetchall()
            }

            by_intent = {
                r["intent"] or "unknown": r["cnt"]
                for r in conn.execute(
                    f"SELECT intent, COUNT(*) as cnt FROM query_log {where} GROUP BY intent",
                    params,
                ).fetchall()
            }

            avg_row = conn.execute(
                f"SELECT AVG(latency_ms) FROM query_log {where}", params
            ).fetchone()[0]

        return {
            "total_queries": total,
            "by_route": by_route,
            "by_intent": by_intent,
            "avg_latency_ms": round(avg_row, 1) if avg_row else 0.0,
        }

    def get_recent(
        self, limit: int = 20, tenant_id: Optional[str] = None
    ) -> List[Dict]:
        """Most recent rows, newest first.  No PII — message_hash only."""
        where = "WHERE tenant_id = ?" if tenant_id else ""
        p: list = [tenant_id] if tenant_id else []
        p.append(min(limit, 200))  # cap to prevent runaway queries

        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT tenant_id, session_id, message_hash, intent,
                           route, latency_ms, timestamp
                    FROM query_log {where}
                    ORDER BY id DESC LIMIT ?""",
                p,
            ).fetchall()

        return [dict(r) for r in rows]

    def total(self, tenant_id: Optional[str] = None) -> int:
        """Quick count — used in tests."""
        where = "WHERE tenant_id = ?" if tenant_id else ""
        params: tuple = (tenant_id,) if tenant_id else ()
        with self._connect() as conn:
            return conn.execute(
                f"SELECT COUNT(*) FROM query_log {where}", params
            ).fetchone()[0]
