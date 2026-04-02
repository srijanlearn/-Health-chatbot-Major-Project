# app/healthcare_agent.py
"""
Healthcare conversation agent with intent detection and context-aware responses.

Optimisations applied (python-pro):
- Prompt templates built once at class level (not rebuilt per call).
- LangChain chains cached per instance so chain objects are reused.
- Ingestion retriever cached inside the agent so process_and_get_retriever
  is NOT called on every insurance query — only once per document.
- Structured logging replaces raw print() calls.
- Full type-hint coverage.
- Cleaner intent routing via a dispatch table.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable

from .session_manager import SessionManager
from .ingestion import process_and_get_retriever

logger = logging.getLogger(__name__)


# ── Prompt constants (built once, shared across all invocations) ───────────────

_INTENT_PROMPT = ChatPromptTemplate.from_template(
    "You are an intent classifier for a healthcare chatbot.\n"
    "Classify the following user message into ONE of these categories:\n\n"
    "- GREETING: Greetings, casual conversation starters\n"
    "- INSURANCE_QUERY: Questions about insurance coverage, policy details, claims, benefits\n"
    "- PRESCRIPTION_INFO: Questions about medications, prescriptions, dosages\n"
    "- SYMPTOM_CHECK: Describing symptoms, asking about health conditions\n"
    "- APPOINTMENT: Scheduling, rescheduling, or asking about appointments\n"
    "- LAB_RESULTS: Questions about test results, lab reports\n"
    "- GENERAL_HEALTH: General health advice, wellness tips\n"
    "- DOCUMENT_UPLOAD: User mentions uploading or sending a document/image\n"
    "- UNKNOWN: Cannot determine intent\n\n"
    "User Message: {message}\n\n"
    "Respond with ONLY the category name (e.g., \"INSURANCE_QUERY\").\n\n"
    "Intent:"
)

_ROUTER_PROMPT = ChatPromptTemplate.from_template(
    "Classify this question as either 'Specific Fact' or 'General Context'.\n"
    "Specific Fact: asks for precise numbers, dates, names, or specific policy details.\n"
    "General Context: asks for summaries, explanations, or general information.\n\n"
    "Question: {question}\n"
    "Classification:"
)

_INSURANCE_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful healthcare insurance assistant. "
    "Answer the question based on the provided context.\n\n"
    "CONTEXT:\n{context}\n\n"
    "CONVERSATION HISTORY:\n{history}\n\n"
    "QUESTION:\n{question}\n\n"
    "Provide a clear, concise answer. If the information isn't in the context, say so politely.\n\n"
    "ANSWER:"
)

_PRESCRIPTION_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful healthcare assistant analysing prescription information.\n\n"
    "PRESCRIPTION TEXT (from image):\n{prescription}\n\n"
    "CONVERSATION HISTORY:\n{history}\n\n"
    "USER QUESTION:\n{question}\n\n"
    "Provide helpful information about the prescription. Include:\n"
    "- Medication names and dosages if asked\n"
    "- Usage instructions if available\n"
    "- Any important notes or warnings\n\n"
    "IMPORTANT: This is informational only. Advise the user to consult their healthcare provider.\n\n"
    "ANSWER:"
)

_LAB_RESULTS_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful healthcare assistant analysing lab results.\n\n"
    "LAB RESULTS (from image):\n{lab_results}\n\n"
    "CONVERSATION HISTORY:\n{history}\n\n"
    "USER QUESTION:\n{question}\n\n"
    "Provide an informative explanation. If specific values are mentioned:\n"
    "- Indicate if they appear within normal ranges (general knowledge only)\n"
    "- Explain what the tests measure\n"
    "- Suggest discussing with a healthcare provider\n\n"
    "IMPORTANT: Always recommend consulting a healthcare provider for proper interpretation.\n\n"
    "ANSWER:"
)

_GENERAL_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful healthcare chatbot assistant.\n\n"
    "CONVERSATION HISTORY:\n{history}\n\n"
    "USER QUESTION:\n{question}\n\n"
    "Provide helpful, accurate health information. "
    "For medical advice, always recommend consulting healthcare professionals.\n"
    "Keep responses concise and friendly.\n\n"
    "ANSWER:"
)

_DOC_UPLOAD_RESPONSE = (
    "Please send the document or image you'd like me to analyse. "
    "I can help with insurance policies, prescriptions, lab reports, "
    "and other medical documents."
)

_GREETING_NEW = (
    "Hello! I'm your AI Healthcare Assistant. I can help you with:\n"
    "• Insurance policy questions\n"
    "• Prescription information\n"
    "• Lab result interpretation\n"
    "• General health queries\n\n"
    "You can also send images of documents for me to analyse. "
    "How can I assist you today?"
)

_GREETING_RETURNING = "Hello again! How can I help you today?"


class HealthcareAgent:
    """
    Healthcare conversation agent with intent detection and context-aware responses.
    Integrates with the existing RAG pipeline for document-based queries.
    """

    # Intent category values
    INTENTS: dict[str, str] = {
        "GREETING": "greeting",
        "INSURANCE_QUERY": "insurance_query",
        "PRESCRIPTION_INFO": "prescription_info",
        "SYMPTOM_CHECK": "symptom_check",
        "APPOINTMENT": "appointment",
        "LAB_RESULTS": "lab_results",
        "GENERAL_HEALTH": "general_health",
        "DOCUMENT_UPLOAD": "document_upload",
        "UNKNOWN": "unknown",
    }

    def __init__(
        self,
        session_manager: SessionManager,
        model_name: str = "llama3",
    ) -> None:
        """
        Initialise the healthcare agent.

        Args:
            session_manager: Session tracking instance.
            model_name: Ollama model identifier (default ``"llama3"``).
        """
        self.session_manager = session_manager

        # LLM instances
        self.llm_pro = ChatOllama(
            model=model_name,
            temperature=0.3,
            num_ctx=2048,
            num_predict=256,
        )
        self.llm_flash = ChatOllama(
            model=model_name,
            temperature=0,
            num_ctx=1024,
            num_predict=32,
        )

        # ── Pre-build all chains once ──────────────────────────────────────────
        self._intent_chain: Runnable = _INTENT_PROMPT | self.llm_flash | StrOutputParser()
        self._router_chain: Runnable = _ROUTER_PROMPT | self.llm_flash | StrOutputParser()
        self._insurance_chain: Runnable = _INSURANCE_PROMPT | self.llm_pro | StrOutputParser()
        self._prescription_chain: Runnable = _PRESCRIPTION_PROMPT | self.llm_pro | StrOutputParser()
        self._lab_chain: Runnable = _LAB_RESULTS_PROMPT | self.llm_pro | StrOutputParser()
        self._general_chain: Runnable = _GENERAL_PROMPT | self.llm_pro | StrOutputParser()

        # ── Per-document retriever cache (document_id → (retriever, full_text)) ─
        self._doc_retriever_cache: dict[str, tuple] = {}

        logger.info("✅ HealthcareAgent initialised with model=%s", model_name)

    # ── Intent detection ───────────────────────────────────────────────────────

    def detect_intent(self, message: str) -> str:
        """
        Classify user intent from *message*.

        Returns one of the values in :attr:`INTENTS`.
        """
        try:
            raw = self._intent_chain.invoke({"message": message}).strip().lower()
            for intent_key, intent_val in self.INTENTS.items():
                if intent_val in raw:
                    return intent_val
            return self.INTENTS["UNKNOWN"]
        except Exception:
            logger.exception("Intent detection failed")
            return self.INTENTS["UNKNOWN"]

    # ── Response generation ────────────────────────────────────────────────────

    def generate_response(
        self,
        user_id: str,
        message: str,
        intent: str,
        extracted_ocr_text: Optional[str] = None,
    ) -> str:
        """
        Generate a contextual response.

        Args:
            user_id: User identifier.
            message: User's raw message.
            intent: Detected intent string.
            extracted_ocr_text: Optional text extracted from an uploaded image.

        Returns:
            Response string to send back to the user.
        """
        conversation_context = self.session_manager.format_history_for_llm(user_id, last_n=5)
        context_parts: list[str] = []

        if conversation_context:
            context_parts.append(f"Conversation History:\n{conversation_context}")
        if extracted_ocr_text:
            context_parts.append(f"\nExtracted from uploaded image:\n{extracted_ocr_text}")

        context = "\n\n".join(context_parts) if context_parts else "No previous conversation."

        recent_docs = self.session_manager.get_recent_documents(user_id, limit=1)
        has_document = bool(recent_docs)

        # ── Intent dispatch ────────────────────────────────────────────────────
        if intent == self.INTENTS["GREETING"]:
            return self._handle_greeting(user_id)

        if intent == self.INTENTS["INSURANCE_QUERY"] and has_document:
            return self._handle_insurance_query(user_id, message, context, recent_docs)

        if intent == self.INTENTS["PRESCRIPTION_INFO"] and extracted_ocr_text:
            return self._handle_prescription_info(message, extracted_ocr_text, context)

        if intent == self.INTENTS["LAB_RESULTS"] and extracted_ocr_text:
            return self._handle_lab_results(message, extracted_ocr_text, context)

        if intent == self.INTENTS["DOCUMENT_UPLOAD"]:
            return _DOC_UPLOAD_RESPONSE

        return self._handle_general_query(message, context)

    # ── Intent handlers ────────────────────────────────────────────────────────

    def _handle_greeting(self, user_id: str) -> str:
        history = self.session_manager.get_conversation_history(user_id)
        return _GREETING_NEW if len(history) <= 2 else _GREETING_RETURNING

    def _handle_insurance_query(
        self,
        user_id: str,
        question: str,
        context: str,
        recent_docs: list[dict],
    ) -> str:
        """Handle insurance queries using the RAG pipeline with per-document caching."""
        doc_info = recent_docs[0]
        document_id: str = doc_info["document_id"]
        document_path: str = doc_info["path"]

        # ── Retriever cache (agent-level) ──────────────────────────────────────
        if document_id not in self._doc_retriever_cache:
            retriever, full_text = process_and_get_retriever(document_path, document_id)
            if retriever is None:
                return "I'm having trouble processing your insurance document. Please try again."
            self._doc_retriever_cache[document_id] = (retriever, full_text)

        retriever, full_text = self._doc_retriever_cache[document_id]

        try:
            route = self._route_question(question)

            if "specific" in route.lower():
                return self._insurance_chain.invoke({
                    "context": full_text[:15_000],
                    "history": context,
                    "question": question,
                })

            def _format_docs(docs: list) -> str:
                return "\n\n".join(d.page_content for d in docs)

            rag_chain: Runnable = (
                {
                    "context": retriever | _format_docs,
                    "history": RunnablePassthrough(),
                    "question": RunnablePassthrough(),
                }
                | _INSURANCE_PROMPT
                | self.llm_pro
                | StrOutputParser()
            )
            return rag_chain.invoke({"history": context, "question": question})

        except Exception:
            logger.exception("Insurance query failed for document_id=%s", document_id)
            return "I encountered an error processing your insurance question. Please try rephrasing."

    def _route_question(self, question: str) -> str:
        """Classify question as 'Specific Fact' or 'General Context'."""
        return self._router_chain.invoke({"question": question})

    def _handle_prescription_info(
        self, question: str, ocr_text: str, context: str
    ) -> str:
        return self._prescription_chain.invoke({
            "prescription": ocr_text,
            "history": context,
            "question": question,
        })

    def _handle_lab_results(
        self, question: str, ocr_text: str, context: str
    ) -> str:
        return self._lab_chain.invoke({
            "lab_results": ocr_text,
            "history": context,
            "question": question,
        })

    def _handle_general_query(self, question: str, context: str) -> str:
        return self._general_chain.invoke({
            "history": context,
            "question": question,
        })

    # ── Main entry point ───────────────────────────────────────────────────────

    def process_user_message(
        self,
        user_id: str,
        message: str,
        ocr_results: Optional[dict] = None,
    ) -> dict[str, str]:
        """
        Process a user message end-to-end.

        Args:
            user_id: User identifier.
            message: Raw user message.
            ocr_results: Optional OCR extraction result dict.

        Returns:
            ``{"intent": str, "response": str}``
        """
        intent = self.detect_intent(message)
        logger.info("Detected intent=%s for user=%s", intent, user_id)

        ocr_text: Optional[str] = None
        if ocr_results and ocr_results.get("success"):
            ocr_text = ocr_results.get("text")
            if ocr_results.get("image_url"):
                self.session_manager.add_extracted_image(
                    user_id,
                    ocr_results["image_url"],
                    ocr_text,
                )

        response = self.generate_response(user_id, message, intent, ocr_text)

        # Persist conversation
        self.session_manager.add_message(user_id, "user", message)
        self.session_manager.add_message(user_id, "assistant", response)

        return {"intent": intent, "response": response}
