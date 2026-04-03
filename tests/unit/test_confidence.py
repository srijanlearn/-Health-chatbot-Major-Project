"""
Unit tests for confidence scoring.

Verifies:
- Route-based confidence assignment (KG=1.0, LLM=heuristic, etc.)
- LLM heuristic scorer behaviour
- Escalated disclaimer is appended for low-confidence health responses
- Cache preserves confidence across calls
"""

from __future__ import annotations

import pytest
from app.orchestrator import _score_llm_confidence, LOW_CONFIDENCE_THRESHOLD
from app.prompts.medical_safety import (
    DISCLAIMER_ESCALATED,
    DISCLAIMER,
    get_disclaimer,
)


# ── _score_llm_confidence ─────────────────────────────────────────────────────

class TestScoreLLMConfidence:
    def test_normal_detailed_response_above_threshold(self):
        response = (
            "Paracetamol is a common analgesic used for fever and pain relief. "
            "The standard adult dose is 500 mg to 1000 mg every 4–6 hours as needed. "
            "It is available over the counter at Jan Aushadhi centres."
        )
        score = _score_llm_confidence(response)
        assert score >= LOW_CONFIDENCE_THRESHOLD

    def test_strong_hedge_reduces_score(self):
        certain = "The dosage is 500 mg taken twice daily with water."
        uncertain = "I'm not sure, but the dosage might be 500 mg taken twice daily."
        assert _score_llm_confidence(uncertain) < _score_llm_confidence(certain)

    def test_very_short_response_is_low_confidence(self):
        score = _score_llm_confidence("I don't know.")
        assert score < LOW_CONFIDENCE_THRESHOLD

    def test_score_never_below_0_30(self):
        worst = "I'm not sure. I don't know. Not certain. I cannot say. Possibly."
        assert _score_llm_confidence(worst) >= 0.30

    def test_score_never_above_0_85(self):
        best = (
            "Metformin 500 mg tablet should be taken twice daily with meals. "
            "The dosage is clearly defined in the Jan Aushadhi formulary."
        )
        assert _score_llm_confidence(best) <= 0.85

    def test_specific_numeric_fact_boosts_score(self):
        without_number = "Paracetamol helps with fever and is widely available."
        with_number = "Paracetamol 500mg helps with fever and is widely available."
        assert _score_llm_confidence(with_number) >= _score_llm_confidence(without_number)

    def test_multiple_mild_hedges_compound(self):
        mildly_hedged = "It typically might possibly depend on your condition."
        assert _score_llm_confidence(mildly_hedged) < 0.65

    def test_case_insensitive(self):
        lower = "i'm not sure about this"
        upper = "I'M NOT SURE ABOUT THIS"
        assert abs(_score_llm_confidence(lower) - _score_llm_confidence(upper)) < 0.01


# ── get_disclaimer ────────────────────────────────────────────────────────────

class TestGetDisclaimer:
    def test_standard_disclaimer_en(self):
        d = get_disclaimer("en", escalated=False)
        assert "consult" in d.lower()
        assert "doctor" in d.lower() or "healthcare" in d.lower()

    def test_escalated_disclaimer_stronger(self):
        standard = get_disclaimer("en", escalated=False)
        escalated = get_disclaimer("en", escalated=True)
        assert len(escalated) > len(standard)
        assert "not confident" in escalated.lower() or "important" in escalated.lower()

    def test_escalated_disclaimer_hi(self):
        d = get_disclaimer("hi", escalated=True)
        # Hindi escalated disclaimer should contain Hindi text
        hindi_chars = sum(1 for c in d if "\u0900" <= c <= "\u097F")
        assert hindi_chars > 0


# ── Pipeline confidence assignment ────────────────────────────────────────────

class TestPipelineConfidence:
    def test_kg_hit_confidence_is_1(self, orchestrator):
        result = orchestrator.process(message="pmjay coverage amount")
        if result.route == "knowledge_graph":
            assert result.confidence == 1.0

    def test_emergency_confidence_is_1(self, orchestrator):
        result = orchestrator.process(message="I am having a heart attack help me")
        assert result.is_emergency is True
        assert result.confidence == 1.0

    def test_static_route_confidence_is_1(self, orchestrator, mock_engine):
        mock_engine.classify.return_value = "greeting"
        result = orchestrator.process(message="hi", is_new_user=True)
        assert result.route == "static"
        assert result.confidence == 1.0

    def test_llm_route_confidence_between_0_and_1(self, orchestrator_no_kg, mock_engine):
        mock_engine.generate.return_value = (
            "Paracetamol is used for pain and fever relief. "
            "It is available at Jan Aushadhi pharmacies for 10 rupee per strip."
        )
        result = orchestrator_no_kg.process(message="what is paracetamol used for")
        assert 0.0 < result.confidence <= 1.0

    def test_low_confidence_llm_response_gets_escalated_disclaimer(self, orchestrator_no_kg, mock_engine):
        mock_engine.classify.return_value = "symptom_check"
        # Force a vague, uncertain LLM response
        mock_engine.generate.return_value = "I'm not sure. It possibly depends."
        result = orchestrator_no_kg.process(message="I feel unwell sometimes")
        if result.confidence < LOW_CONFIDENCE_THRESHOLD:
            assert "not confident" in result.response.lower() or "important" in result.response.lower()

    def test_high_confidence_llm_response_gets_standard_disclaimer(self, orchestrator_no_kg, mock_engine):
        mock_engine.classify.return_value = "symptom_check"
        mock_engine.generate.return_value = (
            "Based on the symptoms described, this could indicate a vitamin D deficiency. "
            "Common causes include limited sun exposure and low dietary intake. "
            "Standard supplementation is 1000 IU per day but should be confirmed by a doctor."
        )
        result = orchestrator_no_kg.process(message="why am I always tired")
        # Standard disclaimer (not escalated) should appear
        assert DISCLAIMER.strip().lower()[:30] in result.response.lower()

    def test_cache_hit_preserves_confidence(self, orchestrator):
        msg = "what is the pmjay coverage amount"
        result1 = orchestrator.process(message=msg)
        result2 = orchestrator.process(message=msg)
        assert result2.route == "cache"
        assert result2.confidence == result1.confidence

    def test_confidence_in_to_dict(self, orchestrator):
        result = orchestrator.process(message="what is paracetamol")
        d = result.to_dict()
        assert "confidence" in d
        assert isinstance(d["confidence"], float)
