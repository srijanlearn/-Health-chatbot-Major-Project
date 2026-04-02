# app/prompts/router.py
"""
Hybrid question router: keyword rules FIRST, LLM fallback SECOND.

90% of questions can be routed accurately by keywords alone,
saving an LLM inference call (~100ms). Only ambiguous queries
fall through to the LLM-based router.

Route categories:
- "knowledge_graph" → answer from structured data (instant, skip LLM)
- "specific_fact"   → full document context + LLM
- "general_rag"     → retrieved chunks + LLM
- "direct_llm"      → no document context needed, general health Q&A
"""

from typing import Optional

# ── Keyword-based routing rules ────────────────────────────────────────────────

_KNOWLEDGE_GRAPH_KEYWORDS = [
    # Government schemes
    "ayushman bharat", "pmjay", "pm-jay", "jan aushadhi",
    "आयुष्मान भारत", "जन औषधि",
    # Generic medicines
    "generic alternative", "generic medicine", "generic for",
    "जेनेरिक दवा", "जेनेरिक विकल्प",
    # Regulatory
    "irdai", "irda", "regulation",
]

_SPECIFIC_FACT_KEYWORDS = [
    # Numbers, amounts, limits
    "waiting period", "how much", "what is the limit", "room rent",
    "sum insured", "co-pay", "copay", "deductible", "sub-limit",
    "premium", "claim amount", "maximum", "minimum",
    "कितना", "प्रतीक्षा अवधि", "सीमा",
    # Exact values
    "percentage", "number of days", "how many", "how long",
]

_GENERAL_RAG_KEYWORDS = [
    # Summaries, explanations
    "does this cover", "explain", "summarize", "summary",
    "what are the", "tell me about", "describe", "overview",
    "covered", "excluded", "exclusion", "inclusion",
    "क्या कवर", "बताइए", "समझाइए",
]

_DIRECT_LLM_KEYWORDS = [
    # General health (no document needed)
    "how to reduce", "symptoms of", "what causes", "treatment for",
    "home remedy", "diet for", "exercise for", "prevention",
    "कैसे कम करें", "लक्षण", "उपचार", "इलाज",
]


def route_by_keywords(question: str) -> Optional[str]:
    """
    Route a question using keyword matching.
    Returns the route category, or None if ambiguous (needs LLM).
    """
    q_lower = question.lower()

    # Check in priority order
    for kw in _KNOWLEDGE_GRAPH_KEYWORDS:
        if kw in q_lower:
            return "knowledge_graph"

    for kw in _SPECIFIC_FACT_KEYWORDS:
        if kw in q_lower:
            return "specific_fact"

    for kw in _DIRECT_LLM_KEYWORDS:
        if kw in q_lower:
            return "direct_llm"

    for kw in _GENERAL_RAG_KEYWORDS:
        if kw in q_lower:
            return "general_rag"

    # Ambiguous — needs LLM router
    return None


# ── LLM-based router (fallback) ───────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """Classify this question into ONE routing category:

- specific_fact: asks for precise numbers, dates, names, amounts, limits, waiting periods
- general_rag: asks for summaries, explanations, coverage details, general information about a document
- direct_llm: general health advice that doesn't need a document (symptoms, diet, exercise)
- knowledge_graph: asks about government schemes (Ayushman Bharat, PMJAY), generic medicines, or IRDAI regulations

Reply with ONLY the category name. Nothing else.

Examples:
"What is the waiting period for cataracts?" → specific_fact
"Does this policy cover maternity?" → general_rag
"How to reduce blood pressure?" → direct_llm
"Am I eligible for Ayushman Bharat?" → knowledge_graph"""
