# app/ingestion.py
"""
Document ingestion pipeline using ParentDocumentRetriever for RAG.
Embeddings and retrievers are cached per document_id to avoid
redundant expensive reinstantiation on repeated calls.
"""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Optional

from langchain_core.stores import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_unstructured import UnstructuredLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores.utils import filter_complex_metadata

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DB_BASE_PATH = "./db"
PARENT_CHUNK_SIZE = 2000
CHILD_CHUNK_SIZE = 400


# ── Cached helpers ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a single shared embeddings instance (expensive to create)."""
    logger.info("Initialising HuggingFace embeddings model: %s", EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# Cache up to 8 document retrievers so repeated queries on the same document
# don't re-ingest from scratch.
_retriever_cache: dict[str, tuple[ParentDocumentRetriever, str]] = {}


def process_and_get_retriever(
    file_path: str,
    document_id: str,
) -> tuple[Optional[ParentDocumentRetriever], Optional[str]]:
    """
    Process a document and return a (retriever, full_text) tuple.

    Results are cached per ``document_id`` — subsequent calls with the same
    document_id are instant and skip re-ingestion entirely.

    Args:
        file_path: Absolute or relative path to the source document.
        document_id: Unique identifier used for the Chroma collection and cache key.

    Returns:
        ``(retriever, full_text)`` on success, ``(None, None)`` on failure.
    """
    # ── Cache hit ──────────────────────────────────────────────────────────────
    if document_id in _retriever_cache:
        logger.debug("Cache hit for document_id=%s", document_id)
        return _retriever_cache[document_id]

    db_path = os.path.join(DB_BASE_PATH, document_id)

    try:
        logger.info("Loading document (incl. OCR): %s", file_path)
        docs = UnstructuredLoader(file_path).load()

        filtered_docs = filter_complex_metadata(docs)

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE)

        vectorstore = Chroma(
            collection_name=document_id,
            embedding_function=_get_embeddings(),   # reuse shared instance
            persist_directory=db_path,
        )

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=InMemoryStore(),
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        logger.info("Adding documents to ParentDocumentRetriever …")
        retriever.add_documents(filtered_docs, ids=None, add_to_docstore=True)

        full_text = "\n\n".join(doc.page_content for doc in filtered_docs)

        result = (retriever, full_text)
        _retriever_cache[document_id] = result          # populate cache

        logger.info("✅ Ingestion complete for document_id=%s", document_id)
        return result

    except Exception:
        logger.exception("❌ Ingestion failed for %s (id=%s)", file_path, document_id)
        return None, None


def invalidate_cache(document_id: str) -> None:
    """Remove a cached retriever (e.g. after the user replaces a document)."""
    _retriever_cache.pop(document_id, None)
    logger.debug("Cache invalidated for document_id=%s", document_id)
