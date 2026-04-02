# app/knowledge/__init__.py
"""
Indian Healthcare Knowledge Graph.

Pre-built structured data for instant, 100% accurate answers
without needing LLM inference. Covers:
- IRDAI health insurance regulations
- Ayushman Bharat (PMJAY) scheme
- Generic medicine alternatives (Jan Aushadhi)
- Common drug interactions
- ICD-10 symptom-to-condition mapping
"""

from .graph import KnowledgeGraph

__all__ = ["KnowledgeGraph"]
