"""
Load test for HealthyPartner API — run with Locust.

Usage:
    pip install locust
    locust -f tests/load/locustfile.py --host http://localhost:8000

Or headless (CI):
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
           --users 50 --spawn-rate 5 --run-time 5m --headless \
           --csv results/load_test

Pass criteria (from GAP-007):
  - No OOM in 30-minute run
  - P95 < 3s for /health and /admin/kb/stats (cache/KG routes)
  - P95 < 8s for /chat (LLM routes)
"""

import random
from locust import HttpUser, task, between

# Realistic Indian healthcare queries — mix of cache-able and non-cache-able
HEALTH_QUERIES = [
    "what is the pmjay coverage amount",
    "generic alternative for crocin",
    "can I take aspirin with warfarin",
    "what does high blood sugar mean",
    "ayushman bharat eligibility criteria",
    "symptoms of chest pain",
    "substitute for dolo 650",
    "metformin alcohol interaction",
    "what is normal hemoglobin level",
    "ibuprofen paracetamol combination safe",
    "how to apply for ayushman bharat card",
    "what is jan aushadhi scheme",
    "high creatinine in blood test meaning",
    "difference between generic and branded medicine",
    "headache causes and treatment",
]

CONVERSATION_HISTORY_SAMPLE = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Hello! I am your healthcare assistant."},
]


class HealthyPartnerUser(HttpUser):
    """
    Simulates a real user interacting with the HealthyPartner API.
    Mix of chat queries, health checks, and admin stats reads.
    """
    wait_time = between(1, 3)  # think time between requests

    @task(5)
    def chat_query(self):
        """Most common operation — free-form chat."""
        query = random.choice(HEALTH_QUERIES)
        with self.client.post(
            "/chat",
            json={"message": query},
            headers={"X-Tenant-ID": "default"},
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("response"):
                    resp.failure("Empty response body")
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(2)
    def chat_with_history(self):
        """Chat with conversation history — tests history formatting path."""
        query = random.choice(HEALTH_QUERIES)
        with self.client.post(
            "/chat",
            json={"message": query, "conversation_history": CONVERSATION_HISTORY_SAMPLE},
            headers={"X-Tenant-ID": "default"},
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")

    @task(3)
    def health_check(self):
        """Lightweight liveness probe — should always be fast."""
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def admin_stats(self):
        """Admin stats — tests KG query path under load."""
        with self.client.get(
            "/admin/kb/stats",
            headers={"X-Tenant-ID": "default"},
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def system_info(self):
        """System info — validates engine introspection under concurrent load."""
        self.client.get("/system/info")


class AdminUser(HttpUser):
    """
    Simulates an admin doing KB management tasks.
    Lower weight — admins are rare compared to end users.
    weight = 1 vs HealthyPartnerUser weight = 10
    """
    weight = 1
    wait_time = between(5, 15)

    @task
    def check_stats(self):
        self.client.get("/admin/kb/stats", headers={"X-Tenant-ID": "default"})

    @task
    def rebuild_fts(self):
        self.client.post("/admin/kb/rebuild", headers={"X-Tenant-ID": "default"})
