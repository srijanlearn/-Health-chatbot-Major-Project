# app/prompts/__init__.py
"""
Optimized prompt library for HealthyPartner v2.

All prompts are designed for small models (3-4B parameters):
- Concise system prompts (<300 tokens)
- Few-shot examples where needed
- Structured output format for reliable parsing
"""

from .intent_classifier import INTENT_SYSTEM_PROMPT, INTENT_CATEGORIES
from .insurance_qa import INSURANCE_SYSTEM_PROMPT, INSURANCE_QA_PROMPT
from .medical_safety import SAFETY_CHECK_PROMPT, EMERGENCY_KEYWORDS, DISCLAIMER
from .router import ROUTER_SYSTEM_PROMPT, route_by_keywords

__all__ = [
    "INTENT_SYSTEM_PROMPT",
    "INTENT_CATEGORIES",
    "INSURANCE_SYSTEM_PROMPT",
    "INSURANCE_QA_PROMPT",
    "SAFETY_CHECK_PROMPT",
    "EMERGENCY_KEYWORDS",
    "DISCLAIMER",
    "ROUTER_SYSTEM_PROMPT",
    "route_by_keywords",
]
