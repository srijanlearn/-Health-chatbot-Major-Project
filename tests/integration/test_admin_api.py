"""
Integration tests for the Admin API endpoints.
Uses FastAPI TestClient — no running server needed.
"""

import io
import csv
import pytest
from fastapi.testclient import TestClient


# ── App fixture ────────────────────────────────────────────────────────────────

@pytest.fixture
def client(mock_engine, tmp_path, monkeypatch):
    """
    Returns a TestClient with a fully wired FastAPI app.
    Uses mock engine and a temp KG — no Ollama, no production data.
    """
    # Point all data paths to tmp_path
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "db").mkdir()
    (tmp_path / "downloaded_files").mkdir()

    # Isolate KG from production JSON files
    import app.knowledge.graph as graph_module
    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()
    monkeypatch.setattr(graph_module, "DATA_DIR", empty_data_dir)

    # Patch LLMEngine constructor to return the mock
    import app.main as main_module
    import app.llm_engine as engine_module
    monkeypatch.setattr(engine_module, "LLMEngine", lambda: mock_engine)

    from app.main import app
    with TestClient(app) as c:
        yield c


def _make_csv(rows: list[dict]) -> bytes:
    """Helper: serialize a list of dicts to CSV bytes."""
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


# ── Health check ───────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_returns_ok_status(self, client):
        data = r = client.get("/health").json()
        assert "status" in data


# ── KB Stats ───────────────────────────────────────────────────────────────────

class TestKBStats:
    def test_stats_returns_200(self, client):
        r = client.get("/admin/kb/stats")
        assert r.status_code == 200

    def test_stats_contains_all_domains(self, client):
        data = client.get("/admin/kb/stats").json()
        stats = data["stats"]
        assert "medicines" in stats
        assert "drug_interactions" in stats
        assert "facts" in stats
        assert "icd10_map" in stats

    def test_stats_initial_counts_are_zero(self, client):
        data = client.get("/admin/kb/stats").json()
        stats = data["stats"]
        assert stats["medicines"] == 0
        assert stats["drug_interactions"] == 0


# ── Medicines Upload ───────────────────────────────────────────────────────────

class TestUploadMedicines:
    def test_valid_csv_returns_200(self, client):
        rows = [{"brand_name": "Crocin", "generic_name": "Paracetamol", "usage": "fever"}]
        r = client.post(
            "/admin/kb/upload/medicines",
            files={"file": ("medicines.csv", _make_csv(rows), "text/csv")},
        )
        assert r.status_code == 200

    def test_inserted_count_matches_valid_rows(self, client):
        rows = [
            {"brand_name": "Crocin", "generic_name": "Paracetamol"},
            {"brand_name": "Brufen", "generic_name": "Ibuprofen"},
        ]
        data = client.post(
            "/admin/kb/upload/medicines",
            files={"file": ("medicines.csv", _make_csv(rows), "text/csv")},
        ).json()
        assert data["inserted"] == 2
        assert data["total_rows"] == 2

    def test_rows_missing_brand_name_skipped(self, client):
        rows = [
            {"brand_name": "Valid", "generic_name": "GenericValid"},
            {"brand_name": "", "generic_name": "NoName"},
        ]
        data = client.post(
            "/admin/kb/upload/medicines",
            files={"file": ("medicines.csv", _make_csv(rows), "text/csv")},
        ).json()
        assert data["inserted"] == 1

    def test_stats_updated_after_upload(self, client):
        rows = [{"brand_name": "X", "generic_name": "Y"}]
        client.post(
            "/admin/kb/upload/medicines",
            files={"file": ("medicines.csv", _make_csv(rows), "text/csv")},
        )
        stats = client.get("/admin/kb/stats").json()["stats"]
        assert stats["medicines"] == 1


# ── Interactions Upload ────────────────────────────────────────────────────────

class TestUploadInteractions:
    def test_valid_interaction_uploaded(self, client):
        rows = [{"drug_a": "Warfarin", "drug_b": "Aspirin",
                 "severity": "severe", "description": "Bleeding risk"}]
        data = client.post(
            "/admin/kb/upload/interactions",
            files={"file": ("interactions.csv", _make_csv(rows), "text/csv")},
        ).json()
        assert data["inserted"] == 1


# ── Facts Upload ───────────────────────────────────────────────────────────────

class TestUploadFacts:
    def test_valid_fact_uploaded(self, client):
        rows = [{"category_id": "pmjay", "key": "Coverage", "value": "5 lakh"}]
        data = client.post(
            "/admin/kb/upload/facts",
            files={"file": ("facts.csv", _make_csv(rows), "text/csv")},
        ).json()
        assert data["inserted"] == 1


# ── ICD-10 Upload ──────────────────────────────────────────────────────────────

class TestUploadICD10:
    def test_valid_icd10_uploaded(self, client):
        rows = [{"symptom": "chest pain", "icd10_code": "R07.9", "condition_name": "Chest pain"}]
        data = client.post(
            "/admin/kb/upload/icd10",
            files={"file": ("icd10.csv", _make_csv(rows), "text/csv")},
        ).json()
        assert data["inserted"] == 1


# ── Rebuild FTS ────────────────────────────────────────────────────────────────

class TestRebuildFTS:
    def test_rebuild_returns_ok(self, client):
        r = client.post("/admin/kb/rebuild")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ── Reset KB ───────────────────────────────────────────────────────────────────

class TestResetKB:
    def test_reset_clears_uploaded_data(self, client):
        # Upload some data first
        rows = [{"brand_name": "X", "generic_name": "Y"}]
        client.post(
            "/admin/kb/upload/medicines",
            files={"file": ("medicines.csv", _make_csv(rows), "text/csv")},
        )
        stats_before = client.get("/admin/kb/stats").json()["stats"]
        assert stats_before["medicines"] == 1

        # Reset
        r = client.post("/admin/kb/reset")
        assert r.status_code == 200

        # Medicines should be gone (JSON files not present in tmp_path, so reload inserts 0)
        stats_after = client.get("/admin/kb/stats").json()["stats"]
        assert stats_after["medicines"] == 0


# ── Admin Key Auth ─────────────────────────────────────────────────────────────

class TestAdminKeyAuth:
    def test_no_key_allowed_when_env_not_set(self, client, monkeypatch):
        monkeypatch.delenv("HP_ADMIN_KEY", raising=False)
        r = client.get("/admin/kb/stats")
        assert r.status_code == 200

    def test_wrong_key_returns_403_when_env_set(self, client, monkeypatch):
        monkeypatch.setenv("HP_ADMIN_KEY", "secret123")
        r = client.get("/admin/kb/stats", headers={"X-Admin-Key": "wrongkey"})
        assert r.status_code == 403

    def test_correct_key_allowed(self, client, monkeypatch):
        monkeypatch.setenv("HP_ADMIN_KEY", "secret123")
        r = client.get("/admin/kb/stats", headers={"X-Admin-Key": "secret123"})
        assert r.status_code == 200


# ── Audit Endpoints ────────────────────────────────────────────────────────────

class TestAuditStats:
    def test_stats_returns_200(self, client):
        r = client.get("/admin/audit/stats")
        assert r.status_code == 200

    def test_stats_shape(self, client):
        data = client.get("/admin/audit/stats").json()
        assert "stats" in data
        stats = data["stats"]
        assert "total_queries" in stats
        assert "by_route" in stats
        assert "by_intent" in stats
        assert "avg_latency_ms" in stats

    def test_stats_start_at_zero(self, client):
        data = client.get("/admin/audit/stats").json()
        assert data["stats"]["total_queries"] == 0

    def test_stats_reflect_chat_calls(self, client):
        # Fire two chat requests — both should be recorded in the audit log
        client.post("/chat", json={"message": "what is paracetamol"})
        client.post("/chat", json={"message": "what is ibuprofen"})
        data = client.get("/admin/audit/stats").json()
        assert data["stats"]["total_queries"] == 2


class TestAuditRecent:
    def test_recent_returns_200(self, client):
        r = client.get("/admin/audit/recent")
        assert r.status_code == 200

    def test_recent_shape(self, client):
        data = client.get("/admin/audit/recent").json()
        assert "rows" in data
        assert "count" in data

    def test_recent_no_plaintext_message(self, client):
        client.post("/chat", json={"message": "secret patient query"})
        rows = client.get("/admin/audit/recent").json()["rows"]
        assert len(rows) >= 1
        for row in rows:
            assert "secret patient query" not in str(row)
            assert "message_hash" in row

    def test_limit_param_respected(self, client):
        for i in range(5):
            client.post("/chat", json={"message": f"query number {i}"})
        data = client.get("/admin/audit/recent?limit=2").json()
        assert data["count"] <= 2
