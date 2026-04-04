# app/ingestion.py
"""
Document ingestion pipeline — v2 (Hybrid Search).

Upgrades from v1:
- Hybrid BM25 + vector search (Reciprocal Rank Fusion)
- Cross-encoder reranker: 20 chunks → top 5 precision filtered
- Multilingual embeddings (multilingual-e5-small) for Hindi+English docs
- Optimised chunk sizes for small model context windows (1500/300)
- BM25 index persisted alongside ChromaDB
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from functools import lru_cache
from typing import List, Optional, Tuple

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

DB_BASE_PATH = "./db"

# v2: multilingual embeddings for Hindi+English documents
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Chunk sizes tuned for hybrid retrieval + cross-encoder rerank on 4B context window.
# Smaller child chunks = finer-grained retrieval precision; reranker handles ordering.
# Smaller parent = fits cleaner inside 3072-token context budget.
PARENT_CHUNK_SIZE = 1000   # was 1500
CHILD_CHUNK_SIZE = 200     # was 300
CHUNK_OVERLAP = 40         # was 50 (~20% of child size)

# Retrieval settings
INITIAL_RETRIEVE_K = 20    # broad net
RERANK_TOP_K = 5           # precision filter

# ── Embeddings (singleton) ─────────────────────────────────────────────────────

_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embeddings model: %s", EMBEDDING_MODEL)
        # Use MPS (Apple Metal) for 3-5x faster document ingestion on Apple Silicon.
        # For single-query retrieval the MPS transfer overhead can exceed the speedup,
        # so we keep MPS only — benchmark your hardware if queries feel slower.
        import torch as _torch
        _device = "mps" if _torch.backends.mps.is_available() else "cpu"
        logger.info("Embeddings device: %s", _device)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": _device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
    return _embeddings


# ── Reranker (lazy load) ───────────────────────────────────────────────────────

_reranker = None


def get_reranker():
    """Load cross-encoder reranker lazily (adds ~80MB memory)."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("✅ Cross-encoder reranker loaded")
        except Exception:
            logger.warning("Cross-encoder not available — skipping rerank step")
    return _reranker


# ── Hybrid Search ──────────────────────────────────────────────────────────────


def _reciprocal_rank_fusion(
    vector_docs: List[Document],
    bm25_docs: List[Document],
    k: int = 40,
) -> List[Document]:
    """
    Merge vector and BM25 results using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank)) across all result lists.
    Deduplicates by content and preserves original Document objects.

    k=40 (was 60): slightly more aggressive top-rank weighting, better for
    healthcare text where exact medical terms (drug names, ICD codes) dominate.

    Dedup key uses source + content prefix to avoid false merges on docs
    that share boilerplate headers (e.g. repeated policy preamble text).
    """
    import hashlib as _hashlib

    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    def _key(doc: Document) -> str:
        source = doc.metadata.get("source", "")
        raw = f"{source}|{doc.page_content[:150]}"
        return _hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()

    for rank, doc in enumerate(vector_docs):
        key = _key(doc)
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = _key(doc)
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


def _rerank(query: str, docs: List[Document], top_k: int = RERANK_TOP_K) -> List[Document]:
    """
    Re-rank documents using cross-encoder for precision filtering.
    Falls back to first top_k documents if reranker unavailable.
    """
    reranker = get_reranker()
    if reranker is None or not docs:
        return docs[:top_k]

    try:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
    except Exception:
        logger.exception("Reranker failed — using top-k fallback")
        return docs[:top_k]


# ── BM25 Index ─────────────────────────────────────────────────────────────────


def _build_bm25_index(documents: List[Document], index_path: str):
    """Build and persist a BM25 index from documents."""
    try:
        from rank_bm25 import BM25Okapi

        corpus = [doc.page_content.lower().split() for doc in documents]
        bm25 = BM25Okapi(corpus)

        with open(index_path, "wb") as f:
            pickle.dump({"bm25": bm25, "docs": documents}, f)

        logger.info("BM25 index saved to %s (%d docs)", index_path, len(documents))
        return bm25, documents
    except ImportError:
        logger.warning("rank_bm25 not installed — BM25 search disabled")
        return None, []


def _load_bm25_index(index_path: str):
    """Load a persisted BM25 index."""
    try:
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        return data["bm25"], data["docs"]
    except Exception:
        return None, []


def _bm25_search(bm25, docs: List[Document], query: str, k: int = 10) -> List[Document]:
    """Search using BM25 index."""
    if bm25 is None:
        return []
    try:
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [docs[i] for i in top_indices if scores[i] > 0]
    except Exception:
        return []


# ── Hybrid Retriever ───────────────────────────────────────────────────────────


class HybridRetriever:
    """
    Combines vector search (ChromaDB) + BM25 keyword search
    with Reciprocal Rank Fusion and optional cross-encoder reranking.
    """

    def __init__(self, parent_retriever: ParentDocumentRetriever, bm25, bm25_docs: List[Document]):
        self._parent = parent_retriever
        self._bm25 = bm25
        self._bm25_docs = bm25_docs

    def invoke(self, query: str) -> List[Document]:
        """Retrieve and rerank documents for a query."""
        # Vector search
        try:
            vector_docs = self._parent.invoke(query)
        except Exception:
            vector_docs = []

        # BM25 keyword search
        bm25_docs = _bm25_search(self._bm25, self._bm25_docs, query, k=10)

        # Merge with RRF
        merged = _reciprocal_rank_fusion(vector_docs, bm25_docs)

        # Rerank for precision
        return _rerank(query, merged, top_k=RERANK_TOP_K)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """LangChain backward-compatible alias."""
        return self.invoke(query)


# ── Main ingestion function ────────────────────────────────────────────────────

# Cache key: (base_path, document_id) — ensures tenant isolation
_retriever_cache: dict[tuple[str, str], Tuple[HybridRetriever, str]] = {}


def process_and_get_retriever(
    document_path: str,
    document_id: str,
    base_path: str = DB_BASE_PATH,
) -> Tuple[Optional[HybridRetriever], Optional[str]]:
    """
    Process a PDF and return a HybridRetriever + full text.

    Caches both the vector store and BM25 index on disk.
    Subsequent calls with the same (base_path, document_id) reuse existing indexes.

    Args:
        document_path: Absolute path to the PDF file.
        document_id:   Unique identifier for the document (used as storage key).
        base_path:     Root directory for this tenant's vector store and BM25 indexes.
                       Defaults to DB_BASE_PATH for backward compatibility.

    Returns:
        (HybridRetriever, full_document_text) or (None, None) on failure.
    """
    cache_key = (base_path, document_id)

    # In-memory cache
    if cache_key in _retriever_cache:
        logger.info("Cache hit for document_id=%s (base=%s)", document_id, base_path)
        return _retriever_cache[cache_key]

    db_path = os.path.join(base_path, document_id)
    bm25_path = os.path.join(db_path, "bm25_index.pkl")
    full_text_path = os.path.join(db_path, "full_text.txt")
    os.makedirs(db_path, exist_ok=True)

    embeddings = get_embeddings()

    try:
        # ── Load or create vector store ────────────────────────────────────────
        vector_store = Chroma(
            collection_name=document_id,
            embedding_function=embeddings,
            persist_directory=db_path,
        )
        doc_store = InMemoryStore()

        # Splitters
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        parent_retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=doc_store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        # ── Check if already ingested ──────────────────────────────────────────
        existing = vector_store.get()
        already_ingested = len(existing.get("ids", [])) > 0

        all_documents: List[Document] = []

        if not already_ingested:
            logger.info("Ingesting document: %s", document_path)
            loader = UnstructuredFileLoader(document_path, mode="single", strategy="fast")
            documents = loader.load()

            if not documents:
                logger.error("No content extracted from %s", document_path)
                return None, None

            all_documents = documents

            # Filter metadata
            for doc in all_documents:
                doc.metadata = {
                    k: v for k, v in doc.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                }

            parent_retriever.add_documents(all_documents, ids=None)
            logger.info("✅ Vector store built with %d documents", len(all_documents))

            # Save full text
            full_text = "\n\n".join(doc.page_content for doc in all_documents)
            with open(full_text_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            # Build BM25 index
            _bm25, _bm25_docs = _build_bm25_index(all_documents, bm25_path)

        else:
            logger.info("Document already ingested — loading from cache")

            # Load full text
            full_text = ""
            if os.path.exists(full_text_path):
                with open(full_text_path, "r", encoding="utf-8") as f:
                    full_text = f.read()

            # Load BM25 index
            _bm25, _bm25_docs = _load_bm25_index(bm25_path)

            if not full_text:
                return None, None

        # ── Build hybrid retriever ─────────────────────────────────────────────
        full_text_out = full_text if full_text else "\n\n".join(
            doc.page_content for doc in all_documents
        )

        retriever = HybridRetriever(parent_retriever, _bm25, _bm25_docs)
        _retriever_cache[cache_key] = (retriever, full_text_out)

        logger.info(
            "✅ HybridRetriever ready for document_id=%s (BM25=%s, Reranker=%s)",
            document_id,
            "enabled" if _bm25 else "disabled",
            "enabled" if get_reranker() else "disabled",
        )
        return retriever, full_text_out

    except Exception:
        logger.exception("Failed to process document: %s", document_path)
        return None, None
