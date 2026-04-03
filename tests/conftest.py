"""
Shared pytest fixtures for HealthyPartner test suite.

All fixtures are designed to be:
- Fast (no Ollama, no network)
- Isolated (tmp_path for every DB)
- Deterministic (mock LLM returns fixed strings)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.knowledge.graph import KnowledgeGraph
from app.orchestrator import Orchestrator


# ── Seed data ──────────────────────────────────────────────────────────────────

SEED_MEDICINES = [
    {"brand_name": "Crocin", "generic_name": "Paracetamol", "category": "Analgesic",
     "jan_aushadhi_price": 10.0, "market_price": 45.0, "savings_percent": 78.0,
     "usage": "fever and pain relief"},
    {"brand_name": "Dolo 650", "generic_name": "Paracetamol 650mg", "category": "Analgesic",
     "jan_aushadhi_price": 15.0, "market_price": 60.0, "savings_percent": 75.0,
     "usage": "fever and pain"},
    {"brand_name": "Brufen", "generic_name": "Ibuprofen", "category": "NSAID",
     "jan_aushadhi_price": 12.0, "market_price": 55.0, "savings_percent": 78.0,
     "usage": "inflammation and pain"},
]

SEED_INTERACTIONS = [
    {"drug_a": "Warfarin", "drug_b": "Aspirin", "severity": "severe",
     "description": "Concurrent use increases bleeding risk significantly.",
     "recommendation": "Avoid combination; consult doctor."},
    {"drug_a": "Metformin", "drug_b": "Alcohol", "severity": "moderate",
     "description": "Increased risk of lactic acidosis.",
     "recommendation": "Limit alcohol consumption."},
]

SEED_FACTS = [
    {"category_id": "pmjay", "category_name": "PMJAY", "key": "PMJAY Coverage",
     "value": "5 lakh per family per year under Ayushman Bharat.",
     "source": "NHA", "tags": "pmjay,ayushman,coverage"},
    {"category_id": "pmjay", "category_name": "PMJAY", "key": "PMJAY Eligibility",
     "value": "Families listed in SECC 2011 database are eligible.",
     "source": "NHA", "tags": "pmjay,eligibility"},
]

SEED_ICD10 = [
    {"symptom": "chest pain", "icd10_code": "R07.9", "condition_name": "Chest pain unspecified",
     "severity": "severe", "see_doctor_urgency": "immediate"},
    {"symptom": "headache", "icd10_code": "R51", "condition_name": "Headache",
     "severity": "mild", "see_doctor_urgency": "routine"},
    {"symptom": "high blood sugar", "icd10_code": "R73.09", "condition_name": "Hyperglycaemia",
     "severity": "moderate", "see_doctor_urgency": "within_24h"},
]


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def test_kg(tmp_path, monkeypatch):
    """
    A KnowledgeGraph backed by a fresh temp SQLite DB, pre-seeded with
    representative data. DATA_DIR is patched to empty so no JSON files load.
    """
    import app.knowledge.graph as graph_module
    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()
    monkeypatch.setattr(graph_module, "DATA_DIR", empty_data_dir)

    kg = KnowledgeGraph(db_path=str(tmp_path / "test.db"))
    kg.import_csv_medicines(SEED_MEDICINES)
    kg.import_csv_interactions(SEED_INTERACTIONS)
    kg.import_csv_facts(SEED_FACTS)
    kg.import_csv_icd10(SEED_ICD10)
    return kg


@pytest.fixture
def empty_kg(tmp_path, monkeypatch):
    """
    A KnowledgeGraph with schema but no data.
    DATA_DIR is monkeypatched to an empty directory so no JSON files are loaded.
    """
    import app.knowledge.graph as graph_module
    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()
    monkeypatch.setattr(graph_module, "DATA_DIR", empty_data_dir)
    return KnowledgeGraph(db_path=str(tmp_path / "empty.db"))


@pytest.fixture
def mock_engine():
    """
    A MagicMock that mimics LLMEngine without starting Ollama.
    Override return values per-test as needed.
    """
    engine = MagicMock()
    engine.classify.return_value = "general_health"
    engine.generate.return_value = "This is a test response from the mock LLM."
    engine.ensure_models_available.return_value = {"main_model": True, "fast_model": True}
    engine.health_check.return_value = {"status": "ok", "ollama": True}
    engine.tier = "balanced"
    engine.main_model = "mock-main"
    engine.fast_model = "mock-fast"
    engine.list_local_models.return_value = ["mock-main", "mock-fast"]
    return engine


@pytest.fixture
def orchestrator(mock_engine, test_kg):
    """A fully wired Orchestrator using the mock engine and seeded KG."""
    return Orchestrator(engine=mock_engine, knowledge_graph=test_kg)


@pytest.fixture
def orchestrator_no_kg(mock_engine):
    """Orchestrator with no knowledge graph — forces all queries to LLM."""
    return Orchestrator(engine=mock_engine, knowledge_graph=None)
