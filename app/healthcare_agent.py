# app/healthcare_agent.py
"""
Healthcare conversation agent — v2 (local LLM).

Integrates the Orchestrator pipeline with session management,
document retrieval, and OCR processing.

Changes from v1:
- Replaced all ChatOllama/ChatGoogleGenerativeAI with LLMEngine
- Replaced per-chain prompt logic with Orchestrator pipeline
- Added Hindi/regional language support
- Added emergency detection
- Added Indian healthcare intents
"""

from __future__ import annotations

import logging
from typing import Optional

from .llm_engine import LLMEngine
from .orchestrator import Orchestrator
from .session_manager import SessionManager
from .ingestion import process_and_get_retriever

logger = logging.getLogger(__name__)


class HealthcareAgent:
    """
    Healthcare conversation agent with intent detection and context-aware responses.
    Integrates with the RAG pipeline and orchestration layer.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        engine: Optional[LLMEngine] = None,
    ) -> None:
        """
        Initialise the healthcare agent.

        Args:
            session_manager: Session tracking instance.
            engine: LLMEngine instance. If None, creates one with auto-detected tier.
        """
        self.session_manager = session_manager
        self.engine = engine or LLMEngine()
        self.orchestrator = Orchestrator(self.engine)

        # Per-document retriever cache (document_id → (retriever, full_text))
        self._doc_retriever_cache: dict[str, tuple] = {}

        logger.info("✅ HealthcareAgent initialised with %r", self.engine)

    # ── Main entry point ───────────────────────────────────────────────────────

    def process_user_message(
        self,
        user_id: str,
        message: str,
        ocr_results: Optional[dict] = None,
    ) -> dict[str, str]:
        """
        Process a user message end-to-end through the orchestration pipeline.

        Args:
            user_id: User identifier.
            message: Raw user message.
            ocr_results: Optional OCR extraction result dict.

        Returns:
            ``{"intent": str, "response": str}``
        """
        # Get conversation context
        conversation_history = self.session_manager.format_history_for_llm(
            user_id, last_n=5
        )
        history = self.session_manager.get_conversation_history(user_id)
        is_new_user = len(history) <= 2

        # Get document context if available
        recent_docs = self.session_manager.get_recent_documents(user_id, limit=1)
        has_document = bool(recent_docs)
        retrieved_chunks = None
        full_document_text = None

        if has_document:
            doc_info = recent_docs[0]
            retriever, full_text = self._get_retriever(
                doc_info["path"], doc_info["document_id"]
            )
            if retriever and full_text:
                full_document_text = full_text
                try:
                    docs = retriever.invoke(message)
                    retrieved_chunks = [doc.page_content for doc in docs]
                except Exception:
                    logger.exception("Retrieval failed for doc %s", doc_info["document_id"])

        # Handle OCR text
        ocr_text: Optional[str] = None
        if ocr_results and ocr_results.get("success"):
            ocr_text = ocr_results.get("text")
            if ocr_results.get("image_url"):
                self.session_manager.add_extracted_image(
                    user_id,
                    ocr_results["image_url"],
                    ocr_text,
                )

        # Run the orchestration pipeline
        result = self.orchestrator.process(
            message=message,
            conversation_history=conversation_history,
            retrieved_chunks=retrieved_chunks,
            full_document_text=full_document_text,
            ocr_text=ocr_text,
            has_document=has_document,
            is_new_user=is_new_user,
        )

        # Persist conversation
        self.session_manager.add_message(user_id, "user", message)
        self.session_manager.add_message(user_id, "assistant", result.response)

        logger.info(
            "Processed message for user=%s intent=%s route=%s latency=%.0fms",
            user_id,
            result.intent,
            result.route,
            result.latency_ms,
        )

        return {"intent": result.intent, "response": result.response}

    # ── Document retriever cache ───────────────────────────────────────────────

    def _get_retriever(self, document_path: str, document_id: str):
        """Get or create a retriever for a document, with caching."""
        if document_id not in self._doc_retriever_cache:
            retriever, full_text = process_and_get_retriever(document_path, document_id)
            if retriever is None:
                return None, None
            self._doc_retriever_cache[document_id] = (retriever, full_text)

        return self._doc_retriever_cache[document_id]

    # ── Health check ───────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """Return agent and engine health status."""
        return {
            "agent": "ready",
            "engine": self.engine.health_check(),
            "cached_documents": len(self._doc_retriever_cache),
        }
