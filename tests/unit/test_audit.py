"""
Unit tests for AuditLog.

Verifies:
- No PII is stored (message is hashed)
- Log is append-only and thread-safe
- Stats aggregate correctly
- get_recent respects limit and tenant filter
- Failures in log_query never raise to the caller
"""

from __future__ import annotations

import threading
import pytest

from app.audit import AuditLog


@pytest.fixture
def audit(tmp_path):
    return AuditLog(db_path=str(tmp_path / "audit.db"))


# ── Basic write / read ─────────────────────────────────────────────────────────

class TestLogQuery:
    def test_single_entry_is_recorded(self, audit):
        audit.log_query(tenant_id="t1", message="hello", intent="greeting", route="static", latency_ms=10)
        assert audit.total(tenant_id="t1") == 1

    def test_message_is_hashed_not_stored_in_plaintext(self, audit):
        original = "what is my blood sugar level"
        audit.log_query(tenant_id="t1", message=original)
        rows = audit.get_recent(limit=5, tenant_id="t1")
        assert len(rows) == 1
        row = rows[0]
        assert "message_hash" in row
        assert original not in row.values()          # plaintext must NOT appear
        assert len(row["message_hash"]) == 16        # 16 hex chars

    def test_different_messages_produce_different_hashes(self, audit):
        audit.log_query(tenant_id="t1", message="query one")
        audit.log_query(tenant_id="t1", message="query two")
        rows = audit.get_recent(limit=10, tenant_id="t1")
        hashes = [r["message_hash"] for r in rows]
        assert hashes[0] != hashes[1]

    def test_identical_messages_produce_same_hash(self, audit):
        audit.log_query(tenant_id="t1", message="same query")
        audit.log_query(tenant_id="t1", message="same query")
        rows = audit.get_recent(limit=10, tenant_id="t1")
        assert rows[0]["message_hash"] == rows[1]["message_hash"]

    def test_hash_is_case_insensitive(self, audit):
        audit.log_query(tenant_id="t1", message="What is Paracetamol")
        audit.log_query(tenant_id="t1", message="what is paracetamol")
        rows = audit.get_recent(limit=10, tenant_id="t1")
        assert rows[0]["message_hash"] == rows[1]["message_hash"]

    def test_optional_fields_stored(self, audit):
        audit.log_query(
            tenant_id="t1",
            message="test",
            intent="lab_results",
            route="knowledge_graph",
            latency_ms=42.5,
            session_id="sess-abc",
        )
        row = audit.get_recent(limit=1, tenant_id="t1")[0]
        assert row["intent"] == "lab_results"
        assert row["route"] == "knowledge_graph"
        assert row["session_id"] == "sess-abc"
        assert row["latency_ms"] == 42.5

    def test_log_query_never_raises(self, audit, monkeypatch):
        """Even if the DB write fails, the caller must not see an exception."""
        monkeypatch.setattr(audit, "_connect", lambda: (_ for _ in ()).throw(RuntimeError("DB full")))
        # Should not raise
        audit.log_query(tenant_id="t1", message="safe")


# ── Tenant isolation ───────────────────────────────────────────────────────────

class TestTenantIsolation:
    def test_counts_are_isolated_per_tenant(self, audit):
        audit.log_query(tenant_id="clinic_a", message="q1")
        audit.log_query(tenant_id="clinic_a", message="q2")
        audit.log_query(tenant_id="clinic_b", message="q3")
        assert audit.total(tenant_id="clinic_a") == 2
        assert audit.total(tenant_id="clinic_b") == 1

    def test_get_recent_filters_by_tenant(self, audit):
        audit.log_query(tenant_id="t1", message="only for t1")
        audit.log_query(tenant_id="t2", message="only for t2")
        rows = audit.get_recent(limit=10, tenant_id="t1")
        assert all(r["tenant_id"] == "t1" for r in rows)

    def test_total_without_filter_counts_all(self, audit):
        audit.log_query(tenant_id="a", message="m1")
        audit.log_query(tenant_id="b", message="m2")
        assert audit.total() == 2


# ── Stats ──────────────────────────────────────────────────────────────────────

class TestGetStats:
    def test_total_queries_matches_log_count(self, audit):
        for i in range(5):
            audit.log_query(tenant_id="t1", message=f"q{i}", route="direct_llm")
        stats = audit.get_stats(tenant_id="t1")
        assert stats["total_queries"] == 5

    def test_by_route_counts_correctly(self, audit):
        audit.log_query(tenant_id="t1", message="a", route="knowledge_graph")
        audit.log_query(tenant_id="t1", message="b", route="knowledge_graph")
        audit.log_query(tenant_id="t1", message="c", route="direct_llm")
        stats = audit.get_stats(tenant_id="t1")
        assert stats["by_route"]["knowledge_graph"] == 2
        assert stats["by_route"]["direct_llm"] == 1

    def test_avg_latency_is_correct(self, audit):
        audit.log_query(tenant_id="t1", message="a", latency_ms=100.0)
        audit.log_query(tenant_id="t1", message="b", latency_ms=200.0)
        stats = audit.get_stats(tenant_id="t1")
        assert stats["avg_latency_ms"] == 150.0

    def test_empty_db_returns_zero_stats(self, audit):
        stats = audit.get_stats(tenant_id="nobody")
        assert stats["total_queries"] == 0
        assert stats["by_route"] == {}
        assert stats["avg_latency_ms"] == 0.0


# ── get_recent ─────────────────────────────────────────────────────────────────

class TestGetRecent:
    def test_limit_is_respected(self, audit):
        for i in range(10):
            audit.log_query(tenant_id="t1", message=f"msg{i}")
        rows = audit.get_recent(limit=3, tenant_id="t1")
        assert len(rows) == 3

    def test_newest_first(self, audit):
        audit.log_query(tenant_id="t1", message="first")
        audit.log_query(tenant_id="t1", message="second")
        rows = audit.get_recent(limit=2, tenant_id="t1")
        # Rows are ordered newest first (ORDER BY id DESC)
        assert rows[0]["timestamp"] >= rows[1]["timestamp"]


# ── Thread safety ──────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_writes_all_recorded(self, audit):
        n = 50
        errors: list = []

        def write(i: int):
            try:
                audit.log_query(tenant_id="t1", message=f"concurrent query {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent writes raised: {errors}"
        assert audit.total(tenant_id="t1") == n
