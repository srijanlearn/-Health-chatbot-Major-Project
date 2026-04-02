# app/orchestrator.py
"""
Multi-step reasoning pipeline for HealthyPartner v2.

This is the key to matching 8B quality with a 4B model.
Instead of one monolithic prompt, we break complex queries
into focused steps — each step is narrow, well-defined,
and optimised for small model performance.

Pipeline:
  1. Detect language (rule-based, instant)
  2. Check for emergencies (keyword-based, instant)
  3. Classify intent (fast model, <100ms)
  4. Check knowledge graph (SQLite, instant — skips LLM if found)
  5. Route to handler (keyword rules, LLM fallback)
  6. Retrieve + Rerank context (hybrid BM25+vector, cross-encoder)
  7. Generate answer (quality model, optimised prompt)
  8. Apply safety guardrails (disclaimers, warnings)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .llm_engine import LLMEngine
from .knowledge.graph import KnowledgeGraph
from .prompts.intent_classifier import (
    INTENT_CATEGORIES,
    INTENT_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT_HI,
)
from .prompts.insurance_qa import (
    INSURANCE_QA_PROMPT,
    INSURANCE_GENERAL_PROMPT,
    INSURANCE_SPECIFIC_FACT_PROMPT,
    INSURANCE_SYSTEM_PROMPT,
)
from .prompts.medical_safety import (
    DISCLAIMER,
    check_emergency,
    detect_language,
    get_disclaimer,
    get_emergency_response,
)
from .prompts.router import ROUTER_SYSTEM_PROMPT, route_by_keywords

logger = logging.getLogger(__name__)


# ── Pipeline Result ────────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """Complete result from the orchestration pipeline."""

    response: str
    intent: str = "unknown"
    route: str = "direct_llm"
    language: str = "en"
    is_emergency: bool = False
    confidence: float = 0.0
    latency_ms: float = 0.0
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "intent": self.intent,
            "route": self.route,
            "language": self.language,
            "is_emergency": self.is_emergency,
            "latency_ms": round(self.latency_ms, 1),
            "steps": self.steps,
        }


# ── Orchestrator ───────────────────────────────────────────────────────────────


class Orchestrator:
    """
    Multi-step reasoning pipeline that coordinates intent detection,
    routing, retrieval, generation, and safety checks.
    """

    # Intents that need document context
    DOCUMENT_INTENTS = {"insurance_query", "prescription_info", "lab_results"}

    # Intents that get direct responses (no LLM needed)
    STATIC_INTENTS = {"greeting", "document_upload"}

    def __init__(self, engine: LLMEngine, knowledge_graph: Optional[KnowledgeGraph] = None) -> None:
        self.engine = engine
        self.knowledge_graph = knowledge_graph
        logger.info("Orchestrator initialised with %r", engine)

    # ── Main entry point ───────────────────────────────────────────────────────

    def process(
        self,
        message: str,
        context: str = "",
        conversation_history: str = "",
        retrieved_chunks: Optional[List[str]] = None,
        full_document_text: Optional[str] = None,
        ocr_text: Optional[str] = None,
        has_document: bool = False,
        is_new_user: bool = True,
    ) -> PipelineResult:
        """
        Run the full orchestration pipeline on a user message.

        Args:
            message: Raw user input.
            context: Pre-formatted conversation context string.
            conversation_history: History for LLM context window.
            retrieved_chunks: Pre-retrieved document chunks (if available).
            full_document_text: Full document text (for specific fact queries).
            ocr_text: Text extracted from an uploaded image.
            has_document: Whether the user has an active document.
            is_new_user: Whether this is the user's first message.

        Returns:
            PipelineResult with response, metadata, and step timings.
        """
        start = time.time()
        result = PipelineResult(response="")
        steps: List[Dict[str, Any]] = []

        # ── Step 1: Language detection (rule-based, instant) ───────────────────
        t0 = time.time()
        result.language = detect_language(message)
        steps.append({"step": "detect_language", "result": result.language, "ms": _ms(t0)})

        # ── Step 2: Emergency check (keyword-based, instant) ──────────────────
        t0 = time.time()
        if check_emergency(message):
            result.is_emergency = True
            result.intent = "emergency"
            result.response = get_emergency_response(result.language)
            result.latency_ms = _ms(start)
            steps.append({"step": "emergency_check", "result": True, "ms": _ms(t0)})
            result.steps = steps
            return result
        steps.append({"step": "emergency_check", "result": False, "ms": _ms(t0)})

        # ── Step 3: Intent classification (fast model) ─────────────────────────
        t0 = time.time()
        sys_prompt = (
            INTENT_SYSTEM_PROMPT_HI if result.language == "hi" else INTENT_SYSTEM_PROMPT
        )
        raw_intent = self.engine.classify(message, system_prompt=sys_prompt)
        intent = self._parse_intent(raw_intent)
        result.intent = intent
        steps.append({
            "step": "classify_intent",
            "raw": raw_intent,
            "parsed": intent,
            "ms": _ms(t0),
        })

        # ── Step 4: Check knowledge graph (instant, no LLM) ───────────────────
        t0 = time.time()
        if self.knowledge_graph and intent not in self.STATIC_INTENTS:
            kg_answer = self.knowledge_graph.query(message)
            if kg_answer:
                steps.append({"step": "knowledge_graph", "hit": True, "ms": _ms(t0)})
                result.response = kg_answer
                result.route = "knowledge_graph"
                result.latency_ms = _ms(start)
                result.steps = steps
                return result
        steps.append({"step": "knowledge_graph", "hit": False, "ms": _ms(t0)})

        # ── Step 5: Handle static intents (no LLM needed) ─────────────────────
        if intent == "greeting":
            result.response = self._greeting_response(is_new_user, result.language)
            result.route = "static"
            result.latency_ms = _ms(start)
            result.steps = steps
            return result

        if intent == "document_upload":
            result.response = self._document_upload_response(result.language)
            result.route = "static"
            result.latency_ms = _ms(start)
            result.steps = steps
            return result

        # ── Step 5: Route question (keywords first, LLM fallback) ──────────────
        t0 = time.time()
        route = route_by_keywords(message)
        if route is None:
            # Ambiguous — use LLM router
            route_raw = self.engine.classify(message, system_prompt=ROUTER_SYSTEM_PROMPT)
            route = self._parse_route(route_raw)
        result.route = route
        steps.append({"step": "route_question", "result": route, "ms": _ms(t0)})

        # ── Step 6: Generate answer based on route ─────────────────────────────
        t0 = time.time()
        response = self._generate_by_route(
            message=message,
            route=route,
            intent=intent,
            context=context,
            conversation_history=conversation_history,
            retrieved_chunks=retrieved_chunks,
            full_document_text=full_document_text,
            ocr_text=ocr_text,
            has_document=has_document,
            language=result.language,
        )
        steps.append({"step": "generate_answer", "route": route, "ms": _ms(t0)})

        # ── Step 7: Safety guardrails ──────────────────────────────────────────
        t0 = time.time()
        # Add medical disclaimer for health-related intents
        if intent in ("symptom_check", "prescription_info", "lab_results", "general_health"):
            response += get_disclaimer(result.language)
        steps.append({"step": "safety_guardrails", "ms": _ms(t0)})

        result.response = response
        result.latency_ms = _ms(start)
        result.steps = steps
        return result

    # ── Route-based generation ─────────────────────────────────────────────────

    def _generate_by_route(
        self,
        message: str,
        route: str,
        intent: str,
        context: str,
        conversation_history: str,
        retrieved_chunks: Optional[List[str]],
        full_document_text: Optional[str],
        ocr_text: Optional[str],
        has_document: bool,
        language: str,
    ) -> str:
        """Generate answer based on the routing decision."""

        # ── Insurance / document queries ───────────────────────────────────────
        if intent in self.DOCUMENT_INTENTS and has_document:
            if route == "specific_fact" and full_document_text:
                # Path A: Full document for precise fact extraction
                doc_context = full_document_text[:12_000]  # Cap for 4B context window
                prompt = INSURANCE_SPECIFIC_FACT_PROMPT.format(
                    context=doc_context, question=message
                )
                return self.engine.generate(
                    prompt=prompt,
                    system_prompt=INSURANCE_SYSTEM_PROMPT,
                    think=True,  # Enable step-by-step reasoning for precise facts
                )
            elif retrieved_chunks:
                # Path B: RAG with retrieved chunks
                chunk_context = "\n\n---\n\n".join(retrieved_chunks[:5])
                prompt = INSURANCE_GENERAL_PROMPT.format(
                    context=chunk_context, question=message
                )
                return self.engine.generate(
                    prompt=prompt,
                    system_prompt=INSURANCE_SYSTEM_PROMPT,
                )
            else:
                return (
                    "I have your document loaded, but I'm having trouble retrieving "
                    "the relevant sections. Could you rephrase your question?"
                )

        # ── Prescription analysis (with OCR text) ─────────────────────────────
        if intent == "prescription_info" and ocr_text:
            sys_prompt = (
                "You are a healthcare assistant analyzing a prescription. "
                "Provide information about medications, dosages, and usage. "
                "Always advise consulting a healthcare provider."
            )
            prompt = (
                f"PRESCRIPTION TEXT (from image):\n{ocr_text}\n\n"
                f"CONVERSATION HISTORY:\n{conversation_history}\n\n"
                f"USER QUESTION:\n{message}"
            )
            return self.engine.generate(prompt=prompt, system_prompt=sys_prompt)

        # ── Lab results analysis (with OCR text) ──────────────────────────────
        if intent == "lab_results" and ocr_text:
            sys_prompt = (
                "You are a healthcare assistant analyzing lab results. "
                "Explain what each test measures and whether values appear "
                "within normal ranges. Always recommend consulting a doctor."
            )
            prompt = (
                f"LAB RESULTS (from image):\n{ocr_text}\n\n"
                f"CONVERSATION HISTORY:\n{conversation_history}\n\n"
                f"USER QUESTION:\n{message}"
            )
            return self.engine.generate(prompt=prompt, system_prompt=sys_prompt)

        # ── Direct LLM (general health, no document needed) ───────────────────
        sys_prompt = (
            "You are a helpful healthcare assistant. Provide accurate health information. "
            "For medical conditions, always recommend consulting a healthcare professional. "
            "Be concise and friendly."
        )
        return self.engine.generate(
            prompt=message,
            system_prompt=sys_prompt,
            context=conversation_history if conversation_history else "",
        )

    # ── Parsing helpers ────────────────────────────────────────────────────────

    def _parse_intent(self, raw: str) -> str:
        """Parse LLM intent output to a valid intent category."""
        raw_lower = raw.strip().lower().replace(" ", "_")
        for key in INTENT_CATEGORIES:
            if key in raw_lower:
                return key
        return "unknown"

    def _parse_route(self, raw: str) -> str:
        """Parse LLM router output to a valid route."""
        raw_lower = raw.strip().lower().replace(" ", "_")
        valid_routes = {"specific_fact", "general_rag", "direct_llm", "knowledge_graph"}
        for route in valid_routes:
            if route in raw_lower:
                return route
        return "general_rag"  # safe default

    # ── Static responses ───────────────────────────────────────────────────────

    def _greeting_response(self, is_new_user: bool, lang: str) -> str:
        if lang == "hi":
            if is_new_user:
                return (
                    "नमस्ते! मैं आपका AI स्वास्थ्य सहायक हूं। मैं आपकी मदद कर सकता हूं:\n"
                    "• बीमा पॉलिसी के सवाल\n"
                    "• प्रिस्क्रिप्शन की जानकारी\n"
                    "• लैब रिपोर्ट समझना\n"
                    "• सामान्य स्वास्थ्य सलाह\n"
                    "• आयुष्मान भारत / सरकारी योजनाएं\n\n"
                    "आप दस्तावेजों की तस्वीरें भी भेज सकते हैं। कैसे मदद करूं?"
                )
            return "फिर से नमस्ते! आज मैं कैसे मदद कर सकता हूं?"

        if is_new_user:
            return (
                "Hello! I'm your AI Healthcare Assistant. I can help you with:\n"
                "• Insurance policy questions\n"
                "• Prescription information\n"
                "• Lab report interpretation\n"
                "• General health queries\n"
                "• Ayushman Bharat / Government schemes\n\n"
                "You can also send images of documents for analysis. "
                "How can I help you today?"
            )
        return "Hello again! How can I help you today?"

    def _document_upload_response(self, lang: str) -> str:
        if lang == "hi":
            return (
                "कृपया वह दस्तावेज़ या तस्वीर भेजें जिसका आप विश्लेषण चाहते हैं। "
                "मैं बीमा पॉलिसी, प्रिस्क्रिप्शन, लैब रिपोर्ट और अन्य "
                "चिकित्सा दस्तावेज़ों में मदद कर सकता हूं।"
            )
        return (
            "Please send the document or image you'd like me to analyse. "
            "I can help with insurance policies, prescriptions, lab reports, "
            "and other medical documents."
        )


# ── Utilities ──────────────────────────────────────────────────────────────────


def _ms(start: float) -> float:
    """Milliseconds elapsed since *start*."""
    return round((time.time() - start) * 1000, 1)
