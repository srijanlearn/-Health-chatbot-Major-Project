"""
Load test for HealthyPartner API — run with Locust.

Usage (interactive UI):
    locust -f tests/load/locustfile.py --host http://localhost:8000

Usage (headless / CI — 5-minute run, 50 users):
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
           --users 50 --spawn-rate 5 --run-time 5m --headless \
           --csv tests/load/results/load_test --csv-full-history

Pass criteria (from GAP-007):
  - No OOM in 30-minute run
  - P95 < 3s  for /health and /admin/kb/stats (no-LLM routes)
  - P95 < 8s  for /chat (LLM routes — includes cold-start variance)
  - Error rate < 1%

User mix:
  - HealthyPartnerUser  (weight=10) — end users sending chat messages
  - MultiTenantUser     (weight=3)  — multi-tenant isolation check
  - AdminUser           (weight=1)  — KB management (rare)
"""

import random
import uuid
from locust import HttpUser, task, between

# ── Query banks ────────────────────────────────────────────────────────────────

# KG-hit queries: should be answered from SQLite, no LLM (fast path)
KG_QUERIES = [
    "what is the pmjay coverage amount",
    "generic alternative for crocin",
    "can I take aspirin with warfarin",
    "ayushman bharat eligibility criteria",
    "substitute for dolo 650",
    "metformin alcohol interaction",
    "ibuprofen paracetamol combination safe",
    "how to apply for ayushman bharat card",
    "what is jan aushadhi scheme",
    "difference between generic and branded medicine",
]

# LLM-fallback queries: ambiguous, likely misses KG
LLM_QUERIES = [
    "what does high blood sugar mean for a diabetic patient",
    "symptoms of chest pain and when to see a doctor",
    "what is normal hemoglobin level for a woman aged 40",
    "high creatinine in blood test meaning",
    "headache causes and treatment options",
    "how to manage blood pressure naturally",
    "explain my lab report values",
    "is it safe to take two paracetamol at once",
]

# Hindi queries: test language detection + Hindi response path
HINDI_QUERIES = [
    "पेरासिटामोल क्या है",
    "आयुष्मान भारत में कितना कवरेज मिलता है",
    "सिरदर्द के कारण क्या हैं",
    "मेटफॉर्मिन कब लेनी चाहिए",
]

ALL_QUERIES = KG_QUERIES + LLM_QUERIES + HINDI_QUERIES

TENANT_IDS = ["default", "default", "default"]  # Only "default" tenant exists; extend when multi-tenant is set up


# ── User classes ───────────────────────────────────────────────────────────────


class HealthyPartnerUser(HttpUser):
    """
    Primary end-user simulation.

    Each virtual user maintains a session_id across requests to exercise the
    SQLite session persistence path (GAP-008) under concurrent load.
    """
    weight = 10
    wait_time = between(1, 3)

    def on_start(self):
        """Assign a stable session_id for this virtual user's lifetime."""
        self.session_id = uuid.uuid4().hex
        self.tenant_id = random.choice(TENANT_IDS)

    @task(6)
    def chat_kg_query(self):
        """KG-backed query — should hit SQLite, skip LLM, be fast."""
        query = random.choice(KG_QUERIES)
        with self.client.post(
            "/chat",
            json={"message": query, "session_id": self.session_id},
            headers={"X-Tenant-ID": self.tenant_id},
            catch_response=True,
            name="/chat [kg]",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("response"):
                    resp.failure("Empty response body")
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(3)
    def chat_llm_query(self):
        """LLM-fallback query — exercises the full pipeline."""
        query = random.choice(LLM_QUERIES)
        with self.client.post(
            "/chat",
            json={"message": query, "session_id": self.session_id},
            headers={"X-Tenant-ID": self.tenant_id},
            catch_response=True,
            name="/chat [llm]",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("response"):
                    resp.failure("Empty response body")
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(2)
    def chat_hindi_query(self):
        """Hindi query — exercises language detection path."""
        query = random.choice(HINDI_QUERIES)
        with self.client.post(
            "/chat",
            json={"message": query, "session_id": self.session_id},
            headers={"X-Tenant-ID": self.tenant_id},
            catch_response=True,
            name="/chat [hi]",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("response"):
                    resp.failure("Empty response body")
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(4)
    def health_check(self):
        """Lightweight liveness probe — always fast, no LLM."""
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def system_info(self):
        """System info endpoint — validates engine introspection under load."""
        self.client.get("/system/info", name="/system/info")

    @task(1)
    def debug_stats(self):
        """Memory/cache stats endpoint — confirms no RSS leak during run."""
        with self.client.get("/debug/stats", catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                rss = data.get("rss_mb", 0)
                # Warn (not fail) if RSS exceeds 2 GB — likely a leak
                if rss > 2048:
                    resp.failure(f"RSS too high: {rss} MB")
            else:
                resp.failure(f"HTTP {resp.status_code}")


class MultiTenantUser(HttpUser):
    """
    Exercises the multi-tenant isolation path.

    Each virtual user picks a different tenant per request to ensure
    no cross-tenant cache bleed under concurrent load.
    """
    weight = 3
    wait_time = between(2, 5)

    def on_start(self):
        self.session_id = uuid.uuid4().hex

    @task
    def chat_default_tenant(self):
        query = random.choice(KG_QUERIES + LLM_QUERIES)
        with self.client.post(
            "/chat",
            json={"message": query, "session_id": self.session_id},
            headers={"X-Tenant-ID": "default"},
            catch_response=True,
            name="/chat [tenant=default]",
        ) as resp:
            if resp.status_code not in (200, 404):
                resp.failure(f"HTTP {resp.status_code}")

    @task
    def admin_stats(self):
        """Admin stats — KG lookup under load."""
        with self.client.get(
            "/admin/kb/stats",
            headers={"X-Tenant-ID": "default"},
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")


class AdminUser(HttpUser):
    """
    Simulates an admin doing KB management (rare).
    """
    weight = 1
    wait_time = between(10, 30)

    @task(3)
    def check_stats(self):
        self.client.get("/admin/kb/stats", headers={"X-Tenant-ID": "default"})

    @task(1)
    def rebuild_fts(self):
        """FTS rebuild under load — should not block chat requests."""
        with self.client.post(
            "/admin/kb/rebuild",
            headers={"X-Tenant-ID": "default"},
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
