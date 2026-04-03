"""
Knowledge Graph Eval Harness
=============================
Golden test suite for KnowledgeGraph.query().

Run:
    python -m tests.eval_knowledge_graph
    python -m tests.eval_knowledge_graph --verbose
    python -m tests.eval_knowledge_graph --fail-fast

Each test case specifies:
  - question    : raw user query
  - intent      : orchestrator-classified intent for this query
  - must_contain: list of substrings that MUST appear in the response (case-insensitive)
  - must_not    : list of substrings that must NOT appear (false-positive guard)
  - category    : which domain this case exercises

Add cases here whenever a bug is fixed or a new data source is loaded.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.knowledge.graph import KnowledgeGraph

# ── Test Cases ────────────────────────────────────────────────────────────────


@dataclass
class Case:
    question: str
    intent: str
    must_contain: List[str]
    must_not: List[str] = field(default_factory=list)
    category: str = "general"
    skip: bool = False  # set True while a data gap is being fixed


CASES: List[Case] = [
    # ── Drug Interactions ─────────────────────────────────────────────────────
    Case(
        question="can i take aspirin with warfarin",
        intent="prescription_info",
        must_contain=["warfarin", "aspirin"],
        must_not=[],
        category="drug_interactions",
    ),
    Case(
        question="is it safe to mix ibuprofen with paracetamol",
        intent="prescription_info",
        must_contain=["ibuprofen", "paracetamol"],
        category="drug_interactions",
    ),
    Case(
        question="drug interaction metformin and alcohol",
        intent="prescription_info",
        must_contain=["metformin"],
        category="drug_interactions",
    ),

    # ── Medicine Alternatives ─────────────────────────────────────────────────
    Case(
        question="generic alternative for crocin",
        intent="prescription_info",
        must_contain=["paracetamol"],  # Crocin → Paracetamol generic
        category="medicines",
    ),
    Case(
        question="cheap substitute for dolo 650",
        intent="prescription_info",
        must_contain=["paracetamol"],
        category="medicines",
    ),

    # ── Lab Results / ICD-10 ──────────────────────────────────────────────────
    Case(
        question="what is normal range for hemoglobin",
        intent="lab_results",
        must_contain=["hemoglobin"],
        must_not=["ayushman", "pmjay"],  # should NOT return insurance facts
        category="lab_results",
    ),
    Case(
        question="high creatinine in blood test what does it mean",
        intent="lab_results",
        must_contain=["creatinine"],
        category="lab_results",
    ),
    Case(
        question="symptoms of diabetes",
        intent="symptom_check",
        must_contain=["diabetes"],
        category="lab_results",
    ),

    # ── Government Schemes / Insurance ───────────────────────────────────────
    Case(
        question="what is ayushman bharat scheme",
        intent="insurance_query",
        must_contain=["ayushman"],
        category="schemes",
    ),
    Case(
        question="pmjay eligibility criteria",
        intent="insurance_query",
        must_contain=["pmjay"],
        category="schemes",
    ),
    Case(
        question="cashless hospitalisation coverage",
        intent="insurance_query",
        must_contain=["cashless"],
        category="schemes",
    ),

    # ── Interaction-signal detection (safety priority) ────────────────────────
    Case(
        question="can I combine atorvastatin with amlodipine",
        intent="general_health",  # keyword signal should override intent ordering
        must_contain=["atorvastatin", "amlodipine"],
        category="drug_interactions",
    ),
]


# ── Runner ────────────────────────────────────────────────────────────────────


@dataclass
class Result:
    case: Case
    response: Optional[str]
    passed: bool
    failures: List[str]
    latency_ms: float


def run_case(kg: KnowledgeGraph, case: Case) -> Result:
    t0 = time.time()
    try:
        response = kg.query(case.question, intent=case.intent)
    except Exception as exc:
        latency = round((time.time() - t0) * 1000, 1)
        return Result(case, None, False, [f"Exception: {exc}"], latency)

    latency = round((time.time() - t0) * 1000, 1)

    if response is None:
        return Result(case, None, False, ["No response (None returned)"], latency)

    resp_lower = response.lower()
    failures: List[str] = []

    for token in case.must_contain:
        if token.lower() not in resp_lower:
            failures.append(f"must_contain '{token}' — not found")

    for token in case.must_not:
        if token.lower() in resp_lower:
            failures.append(f"must_not '{token}' — found unexpectedly")

    return Result(case, response, len(failures) == 0, failures, latency)


def run_all(verbose: bool = False, fail_fast: bool = False) -> bool:
    db_path = Path(__file__).parent.parent / "db" / "healthcare_knowledge.db"
    if not db_path.exists():
        print(f"[ERROR] Knowledge DB not found at {db_path}")
        print("        Run the ingestion script to build the database first.")
        return False

    kg = KnowledgeGraph(str(db_path))

    total = skipped = passed = failed = 0
    results: List[Result] = []

    for case in CASES:
        total += 1
        if case.skip:
            skipped += 1
            print(f"  SKIP  [{case.category}] {case.question[:60]}")
            continue

        result = run_case(kg, case)
        results.append(result)

        if result.passed:
            passed += 1
            status = "  PASS"
        else:
            failed += 1
            status = "  FAIL"

        print(f"{status}  [{case.category}] {case.question[:60]}  ({result.latency_ms}ms)")

        if verbose or not result.passed:
            if result.response:
                print(f"        response: {result.response[:120].strip()!r}")
            for failure in result.failures:
                print(f"        → {failure}")

        if fail_fast and not result.passed:
            break

    print()
    print(f"Results: {passed}/{total - skipped} passed, {failed} failed, {skipped} skipped")

    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KnowledgeGraph golden tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print all responses")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at first failure")
    args = parser.parse_args()

    ok = run_all(verbose=args.verbose, fail_fast=args.fail_fast)
    sys.exit(0 if ok else 1)
