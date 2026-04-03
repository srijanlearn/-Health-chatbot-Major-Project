"""
Unit tests for ingestion helpers — RRF merge, BM25 search.
No ChromaDB or Ollama required.
"""

import pytest
from langchain_core.documents import Document
from app.ingestion import _reciprocal_rank_fusion, _bm25_search, _build_bm25_index


def _doc(content: str, meta: dict = None) -> Document:
    return Document(page_content=content, metadata=meta or {})


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

class TestRRF:
    def test_empty_inputs_returns_empty(self):
        assert _reciprocal_rank_fusion([], []) == []

    def test_single_list_preserves_order(self):
        docs = [_doc("a"), _doc("b"), _doc("c")]
        result = _reciprocal_rank_fusion(docs, [])
        contents = [d.page_content for d in result]
        assert contents == ["a", "b", "c"]

    def test_doc_in_both_lists_ranks_higher(self):
        shared = _doc("shared doc")
        vector_docs = [_doc("only vector"), shared]
        bm25_docs = [shared, _doc("only bm25")]
        result = _reciprocal_rank_fusion(vector_docs, bm25_docs)
        # shared doc appears in both lists — should be ranked first
        assert result[0].page_content == "shared doc"

    def test_deduplication(self):
        doc = _doc("duplicate")
        result = _reciprocal_rank_fusion([doc, doc], [doc])
        # Content-keyed dedup: same doc content appears only once
        contents = [d.page_content for d in result]
        assert contents.count("duplicate") == 1

    def test_all_results_present(self):
        v = [_doc("v1"), _doc("v2")]
        b = [_doc("b1"), _doc("b2")]
        result = _reciprocal_rank_fusion(v, b)
        contents = {d.page_content for d in result}
        assert contents == {"v1", "v2", "b1", "b2"}


# ── BM25 Search ───────────────────────────────────────────────────────────────

class TestBM25Search:
    @pytest.fixture
    def bm25_index(self, tmp_path):
        """Build a real BM25 index from sample documents."""
        docs = [
            _doc("Warfarin is an anticoagulant blood thinner medication"),
            _doc("Aspirin is used for pain relief and fever reduction"),
            _doc("Paracetamol is a common painkiller sold as Crocin"),
            _doc("Metformin is used to treat type 2 diabetes"),
        ]
        index_path = str(tmp_path / "bm25_test.pkl")
        bm25, stored_docs = _build_bm25_index(docs, index_path)
        return bm25, stored_docs

    def test_relevant_query_returns_results(self, bm25_index):
        bm25, docs = bm25_index
        results = _bm25_search(bm25, docs, "warfarin anticoagulant", k=2)
        assert len(results) >= 1
        assert "warfarin" in results[0].page_content.lower()

    def test_no_match_returns_empty(self, bm25_index):
        bm25, docs = bm25_index
        results = _bm25_search(bm25, docs, "xyznonexistentterm999", k=5)
        assert results == []

    def test_none_bm25_returns_empty(self):
        results = _bm25_search(None, [], "any query")
        assert results == []

    def test_respects_k_limit(self, bm25_index):
        bm25, docs = bm25_index
        results = _bm25_search(bm25, docs, "medication", k=1)
        assert len(results) <= 1
