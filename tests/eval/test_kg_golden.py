"""
Golden test suite for KnowledgeGraph.query() — pytest edition.

Each test represents a real-world query that the system must answer correctly.
Tests are parametrized so failures name the exact case that broke.

Add cases here whenever:
  - A bug is fixed (regression guard)
  - New data is loaded into the KB
  - A new intent/domain is added

Mark a case with pytest.mark.skip if a data gap is known and being fixed.
"""

import pytest
from dataclasses import dataclass, field
from typing import List


@dataclass
class GoldenCase:
    question: str
    intent: str
    must_contain: List[str]
    must_not: List[str] = field(default_factory=list)
    category: str = "general"


GOLDEN_CASES = [
    # ── Drug Interactions ──────────────────────────────────────────────────────
    GoldenCase(
        question="can I take aspirin with warfarin",
        intent="prescription_info",
        must_contain=["warfarin", "aspirin"],
        category="drug_interactions",
    ),
    GoldenCase(
        question="is it safe to mix ibuprofen with paracetamol",
        intent="prescription_info",
        must_contain=["ibuprofen", "paracetamol"],
        category="drug_interactions",
    ),
    GoldenCase(
        question="interaction between metformin and alcohol",
        intent="prescription_info",
        must_contain=["metformin"],
        category="drug_interactions",
    ),
    GoldenCase(
        question="combine aspirin warfarin safe",
        intent="general_health",  # interaction keyword should override intent ordering
        must_contain=["aspirin", "warfarin"],
        category="drug_interactions",
    ),

    # ── Medicine Alternatives ──────────────────────────────────────────────────
    GoldenCase(
        question="generic alternative for crocin",
        intent="prescription_info",
        must_contain=["paracetamol"],
        must_not=["pmjay", "ayushman"],
        category="medicines",
    ),
    GoldenCase(
        question="cheap substitute for dolo 650",
        intent="prescription_info",
        must_contain=["paracetamol"],
        category="medicines",
    ),

    # ── Lab Results / ICD-10 ───────────────────────────────────────────────────
    GoldenCase(
        question="what does high blood sugar mean",
        intent="lab_results",
        must_contain=["blood sugar"],
        must_not=["ayushman", "pmjay"],
        category="lab_results",
    ),
    GoldenCase(
        question="symptoms of chest pain",
        intent="symptom_check",
        must_contain=["chest pain"],
        category="lab_results",
    ),

    # ── Government Schemes ─────────────────────────────────────────────────────
    GoldenCase(
        question="pmjay coverage amount",
        intent="insurance_query",
        must_contain=["5 lakh", "pmjay"],
        category="schemes",
    ),
    GoldenCase(
        question="ayushman bharat eligibility",
        intent="insurance_query",
        must_contain=["pmjay"],
        category="schemes",
    ),
]


def _case_id(case: GoldenCase) -> str:
    return f"[{case.category}] {case.question[:50]}"


@pytest.fixture(scope="module")
def seeded_kg(tmp_path_factory):
    """One seeded KG shared across all golden tests (module scope = fast)."""
    from tests.conftest import (
        SEED_MEDICINES, SEED_INTERACTIONS, SEED_FACTS, SEED_ICD10
    )
    from app.knowledge.graph import KnowledgeGraph

    tmp = tmp_path_factory.mktemp("golden_kg")
    kg = KnowledgeGraph(db_path=str(tmp / "golden.db"))
    kg.import_csv_medicines(SEED_MEDICINES)
    kg.import_csv_interactions(SEED_INTERACTIONS)
    kg.import_csv_facts(SEED_FACTS)
    kg.import_csv_icd10(SEED_ICD10)
    return kg


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[_case_id(c) for c in GOLDEN_CASES])
def test_golden_case(seeded_kg, case: GoldenCase):
    response = seeded_kg.query(case.question, intent=case.intent)
    assert response is not None, f"No response returned for: {case.question!r}"

    resp_lower = response.lower()
    for token in case.must_contain:
        assert token.lower() in resp_lower, (
            f"Expected {token!r} in response for: {case.question!r}\n"
            f"Got: {response[:200]!r}"
        )
    for token in case.must_not:
        assert token.lower() not in resp_lower, (
            f"Unexpected {token!r} found in response for: {case.question!r}\n"
            f"(false-positive guard)\nGot: {response[:200]!r}"
        )
