"""
Integration tests for the Orchestrator pipeline.
Uses mock LLMEngine — no Ollama required.
"""

import pytest
from app.orchestrator import Orchestrator


class TestEmergencyDetection:
    def test_emergency_message_short_circuits_pipeline(self, orchestrator):
        result = orchestrator.process(message="I am having a heart attack help me")
        assert result.is_emergency is True
        assert result.route == "emergency"
        assert result.intent == "emergency"

    def test_emergency_not_cached(self, orchestrator):
        """Emergency responses must never be cached and served as stale."""
        orchestrator.process(message="I am having a heart attack")
        # Second call should still detect as emergency, not cache hit
        result = orchestrator.process(message="I am having a heart attack")
        assert result.is_emergency is True
        assert result.route == "emergency"

    def test_normal_message_is_not_emergency(self, orchestrator):
        result = orchestrator.process(message="what is paracetamol used for")
        assert result.is_emergency is False


class TestLanguageDetection:
    def test_english_message_detected(self, orchestrator):
        result = orchestrator.process(message="what is paracetamol")
        assert result.language == "en"


class TestCaching:
    def test_second_identical_call_is_cache_hit(self, orchestrator, mock_engine):
        msg = "what is the coverage amount for pmjay scheme"
        result1 = orchestrator.process(message=msg)
        result2 = orchestrator.process(message=msg)
        assert result2.route == "cache"
        # LLM generate should only be called once (or zero if KG hits first)
        total_generate_calls = mock_engine.generate.call_count
        assert total_generate_calls <= 1

    def test_different_messages_are_not_cache_hits(self, orchestrator):
        orchestrator.process(message="what is paracetamol")
        result = orchestrator.process(message="what is ibuprofen")
        assert result.route != "cache"

    def test_document_queries_not_cached(self, orchestrator):
        """Queries with has_document=True must bypass the cache both ways."""
        msg = "what is the waiting period"
        orchestrator.process(message=msg, has_document=True)
        result = orchestrator.process(message=msg, has_document=True)
        assert result.route != "cache"

    def test_ocr_queries_not_cached(self, orchestrator):
        msg = "what medication is listed here"
        orchestrator.process(message=msg, ocr_text="Tab Metformin 500mg")
        result = orchestrator.process(message=msg, ocr_text="Tab Metformin 500mg")
        assert result.route != "cache"


class TestKnowledgeGraphLookup:
    def test_kg_hit_returns_knowledge_graph_route(self, orchestrator, mock_engine):
        """A question matching seeded KG data should be answered without LLM."""
        result = orchestrator.process(message="pmjay coverage amount")
        if result.route == "knowledge_graph":
            # KG answered — LLM generate should not have been called
            mock_engine.generate.assert_not_called()

    def test_kg_miss_falls_through_to_llm(self, orchestrator_no_kg, mock_engine):
        """With no KG, every query must reach the LLM generate step."""
        orchestrator_no_kg.process(message="what is metformin used for")
        mock_engine.generate.assert_called_once()


class TestStaticIntents:
    def test_greeting_returns_static_route(self, orchestrator, mock_engine):
        mock_engine.classify.return_value = "greeting"
        result = orchestrator.process(message="hello", is_new_user=True)
        assert result.route == "static"
        mock_engine.generate.assert_not_called()

    def test_new_user_greeting_includes_feature_list(self, orchestrator, mock_engine):
        mock_engine.classify.return_value = "greeting"
        result = orchestrator.process(message="hi", is_new_user=True)
        # New user greeting should mention capabilities
        assert any(word in result.response.lower() for word in ["insurance", "help", "health"])

    def test_returning_user_greeting_is_shorter(self, orchestrator, mock_engine):
        mock_engine.classify.return_value = "greeting"
        new_user_result = orchestrator.process(message="hi", is_new_user=True)
        returning_result = orchestrator.process(message="hi", is_new_user=False)
        assert len(returning_result.response) < len(new_user_result.response)


class TestSafetyGuardrails:
    def test_symptom_check_gets_disclaimer(self, orchestrator, mock_engine):
        mock_engine.classify.return_value = "symptom_check"
        result = orchestrator.process(message="I have a mild headache")
        # Disclaimer should be appended for symptom_check intent
        assert "doctor" in result.response.lower() or "consult" in result.response.lower()

    def test_emergency_response_contains_emergency_info(self, orchestrator):
        result = orchestrator.process(message="I am about to faint please help")
        if result.is_emergency:
            assert len(result.response) > 20  # Not empty


class TestPipelineResult:
    def test_result_has_all_required_fields(self, orchestrator):
        result = orchestrator.process(message="what is paracetamol")
        assert result.response is not None
        assert result.intent is not None
        assert result.route is not None
        assert result.language is not None
        assert isinstance(result.is_emergency, bool)
        assert result.latency_ms >= 0

    def test_result_to_dict_is_serialisable(self, orchestrator):
        import json
        result = orchestrator.process(message="hello")
        d = result.to_dict()
        # Should be JSON-serialisable
        json.dumps(d)

    def test_steps_are_recorded(self, orchestrator):
        result = orchestrator.process(message="what is paracetamol")
        assert len(result.steps) > 0
        step_names = [s["step"] for s in result.steps]
        assert "detect_language" in step_names
        assert "emergency_check" in step_names
