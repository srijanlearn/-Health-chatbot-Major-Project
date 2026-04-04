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

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .audit import AuditLog
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
            "confidence": round(self.confidence, 2),
            "latency_ms": round(self.latency_ms, 1),
            "steps": self.steps,
        }


# ── Response Cache ─────────────────────────────────────────────────────────────


class _ResponseCache:
    """
    In-memory TTL cache for LLM responses.

    Caches (message → response) for queries that don't depend on an uploaded
    document or OCR text.  Keyed on a SHA-256 prefix of the lowercased,
    stripped message so near-identical phrasing reuses cached answers.

    Eviction policy: expired entries first, then FIFO when at capacity.
    Thread safety: not needed — Flask/FastAPI run each request sequentially
    against the same engine process.
    """

    def __init__(self, ttl_seconds: int = 1800, max_size: int = 500) -> None:
        # store: key → (response_text, confidence, expires_at)
        self._store: Dict[str, tuple[str, float, float]] = {}
        self._ttl = ttl_seconds
        self._max = max_size

    def _key(self, message: str) -> str:
        return hashlib.sha256(message.lower().strip().encode()).hexdigest()[:20]

    def get(self, message: str) -> Optional[tuple[str, float]]:
        """Return (response, confidence) or None on miss/expiry."""
        key = self._key(message)
        entry = self._store.get(key)
        if entry is None:
            return None
        response, confidence, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        return response, confidence

    def set(self, message: str, response: str, confidence: float = 1.0) -> None:
        if len(self._store) >= self._max:
            now = time.time()
            # Remove all expired entries first
            expired = [k for k, (_, _c, exp) in self._store.items() if exp < now]
            for k in expired:
                del self._store[k]
            # If still at capacity, drop the oldest inserted entry
            if len(self._store) >= self._max:
                del self._store[next(iter(self._store))]
        self._store[self._key(message)] = (response, confidence, time.time() + self._ttl)

    def warm(self, entries: List[tuple[str, str, float]]) -> int:
        """Preload recent query/response pairs into the cache."""
        added = 0
        for message, response, confidence in entries:
            if not message or not response:
                continue
            self.set(message, response, confidence=confidence)
            added += 1
        return added

    @property
    def size(self) -> int:
        return len(self._store)


# ── Confidence scoring ────────────────────────────────────────────────────────

# Responses below this threshold get a stronger safety disclaimer appended.
LOW_CONFIDENCE_THRESHOLD = 0.60

# Hedging phrases that indicate the LLM is uncertain.
_HEDGE_STRONG = (
    "i'm not sure", "i am not sure", "i don't know", "i do not know",
    "i cannot say", "i can't say", "not certain", "unclear",
    "no information", "limited information",
)
_HEDGE_MILD = (
    "it depends", "depending on", "may vary", "could be", "might be",
    "possibly", "perhaps", "generally", "usually", "typically",
    "approximately", "around", "about",
)


def _score_llm_confidence(response: str) -> float:
    """
    Heuristic confidence score for an LLM-generated response.

    Range: 0.30 – 0.85  (1.0 is reserved for KG-backed facts)

    Scoring:
      - Base: 0.65
      - Very short response (< 80 chars): -0.20
      - Short response (< 160 chars): -0.08
      - Each strong hedge phrase: -0.12  (capped at -0.24)
      - Each mild hedge phrase:   -0.04  (capped at -0.12)
      - Contains a number (specific fact): +0.05
    """
    text = response.lower()
    score = 0.65

    # Length penalty
    if len(response) < 80:
        score -= 0.20
    elif len(response) < 160:
        score -= 0.08

    # Strong hedging
    strong_hits = sum(1 for p in _HEDGE_STRONG if p in text)
    score -= min(strong_hits * 0.12, 0.24)

    # Mild hedging
    mild_hits = sum(1 for p in _HEDGE_MILD if p in text)
    score -= min(mild_hits * 0.04, 0.12)

    # Specific number → small boost (concrete answer)
    import re as _re
    if _re.search(r"\b\d+(\.\d+)?\s*(mg|mcg|ml|lakh|year|day|hour|%|rupee)", text):
        score += 0.05

    return round(max(0.30, min(0.85, score)), 2)


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

    def __init__(
        self,
        engine: LLMEngine,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        audit_log: Optional[AuditLog] = None,
    ) -> None:
        self.engine = engine
        self.knowledge_graph = knowledge_graph
        self._audit = audit_log
        self._cache = _ResponseCache(ttl_seconds=1800, max_size=500)
        logger.info("Orchestrator initialised with %r", engine)

    def warm_response_cache(
        self,
        entries: List[tuple[str, str, float]] | List[tuple[str, str]],
    ) -> int:
        """
        Preload the in-memory cache from persisted recent conversations.

        Entries may be ``(message, response)`` or ``(message, response, confidence)``.
        """
        prepared: List[tuple[str, str, float]] = []
        for entry in entries:
            if len(entry) == 2:
                message, response = entry
                prepared.append((message, response, _score_llm_confidence(response)))
            else:
                message, response, confidence = entry
                prepared.append((message, response, confidence))
        warmed = self._cache.warm(prepared)
        if warmed:
            logger.info("Warmed response cache with %d entries", warmed)
        return warmed

    # ── Audit helper ──────────────────────────────────────────────────────────

    def _log_audit(
        self,
        result: "PipelineResult",
        message: str,
        tenant_id: str,
        session_id: Optional[str],
    ) -> None:
        if self._audit is not None:
            self._audit.log_query(
                tenant_id=tenant_id,
                message=message,
                intent=result.intent,
                route=result.route,
                latency_ms=result.latency_ms,
                session_id=session_id,
            )

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
        tenant_id: str = "default",
        session_id: Optional[str] = None,
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
            result.route = "emergency"
            result.confidence = 1.0
            result.response = get_emergency_response(result.language)
            result.latency_ms = _ms(start)
            steps.append({"step": "emergency_check", "result": True, "ms": _ms(t0)})
            result.steps = steps
            self._log_audit(result, message, tenant_id, session_id)
            return result
        steps.append({"step": "emergency_check", "result": False, "ms": _ms(t0)})

        # ── Cache lookup (after safety checks, before expensive LLM steps) ──────
        # Only cache stateless queries: no uploaded document, no OCR image.
        _cacheable = not has_document and not ocr_text
        if _cacheable:
            cached_response = self._cache.get(message)
            if cached_response is not None:
                cached_text, cached_conf = cached_response
                result.response = cached_text
                result.confidence = cached_conf
                result.route = "cache"
                result.latency_ms = _ms(start)
                steps.append({"step": "cache", "hit": True, "ms": _ms(t0)})
                result.steps = steps
                logger.debug("Cache hit for message (cache_size=%d)", self._cache.size)
                self._log_audit(result, message, tenant_id, session_id)
                return result

        # ── Steps 3 + 4: Intent classification and KG lookup — run in parallel ──
        # Intent classify (fast model, ~100ms) and KG lookup (SQLite, ~2ms) are
        # independent — neither depends on the other's result.  Running them
        # concurrently shaves ~100ms from every non-cached request.
        t0 = time.time()
        sys_prompt = (
            INTENT_SYSTEM_PROMPT_HI if result.language == "hi" else INTENT_SYSTEM_PROMPT
        )

        with ThreadPoolExecutor(max_workers=2) as _pool:
            _intent_fut: Future = _pool.submit(
                self.engine.classify, message, sys_prompt
            )
            # KG lookup with intent="unknown" first; we refine below if needed
            _kg_fut: Optional[Future] = (
                _pool.submit(self.knowledge_graph.query, message, "unknown")
                if self.knowledge_graph else None
            )
            raw_intent = _intent_fut.result()
            _kg_answer_unknown = _kg_fut.result() if _kg_fut is not None else None

        intent = self._parse_intent(raw_intent)
        result.intent = intent
        steps.append({
            "step": "classify_intent",
            "raw": raw_intent,
            "parsed": intent,
            "ms": _ms(t0),
        })

        # ── Reconcile KG result with resolved intent ───────────────────────────
        # If the KG already answered for "unknown" intent, accept it.
        # If not, retry with the resolved intent (catches intent-gated KG paths).
        t0 = time.time()
        if self.knowledge_graph and intent not in self.STATIC_INTENTS:
            kg_answer = _kg_answer_unknown or self.knowledge_graph.query(
                message, intent=intent
            )
            if kg_answer:
                steps.append({"step": "knowledge_graph", "hit": True, "ms": _ms(t0)})
                result.response = kg_answer
                result.route = "knowledge_graph"
                result.confidence = 1.0
                result.latency_ms = _ms(start)
                result.steps = steps
                if _cacheable:
                    self._cache.set(message, kg_answer, confidence=1.0)
                self._log_audit(result, message, tenant_id, session_id)
                return result
        steps.append({"step": "knowledge_graph", "hit": False, "ms": _ms(t0)})

        # ── Step 5: Handle static intents (no LLM needed) ─────────────────────
        if intent == "greeting":
            result.response = self._greeting_response(is_new_user, result.language)
            result.route = "static"
            result.confidence = 1.0
            result.latency_ms = _ms(start)
            result.steps = steps
            self._log_audit(result, message, tenant_id, session_id)
            return result

        if intent == "document_upload":
            result.response = self._document_upload_response(result.language)
            result.route = "static"
            result.confidence = 1.0
            result.latency_ms = _ms(start)
            result.steps = steps
            self._log_audit(result, message, tenant_id, session_id)
            return result

        # ── Step 5: Route question (keywords first, LLM fallback) ──────────────
        t0 = time.time()
        route = route_by_keywords(message)
        if route is None:
            # Ambiguous — use LLM router
            route_raw = self.engine.classify(message, system_prompt=ROUTER_SYSTEM_PROMPT)
            route = self._parse_route(route_raw)
        # If keyword/LLM router says knowledge_graph but step 4 already missed,
        # fall back to direct_llm so the route badge reflects what actually answers.
        if route == "knowledge_graph":
            route = "direct_llm"
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

        # ── Confidence scoring ────────────────────────────────────────────────
        confidence = _score_llm_confidence(response)
        result.confidence = confidence
        steps.append({"step": "confidence_score", "score": confidence, "ms": 0})

        # ── Step 7: Safety guardrails ──────────────────────────────────────────
        t0 = time.time()
        health_intents = ("symptom_check", "prescription_info", "lab_results", "general_health")
        if intent in health_intents:
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                # Low-confidence answer — escalate the disclaimer
                response += get_disclaimer(result.language, escalated=True)
            else:
                response += get_disclaimer(result.language)
        steps.append({"step": "safety_guardrails", "confidence": confidence, "ms": _ms(t0)})

        result.response = response
        result.latency_ms = _ms(start)
        result.steps = steps

        # Store in cache for future identical queries
        if _cacheable:
            self._cache.set(message, response, confidence=confidence)

        self._log_audit(result, message, tenant_id, session_id)
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
