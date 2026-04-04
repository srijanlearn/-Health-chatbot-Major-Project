"""
Unit tests for SessionManager — SQLite-backed session persistence.

Key scenarios:
  - Sessions survive a "restart" (new SessionManager instance, same db_path)
  - Conversation history, documents, and context all round-trip correctly
  - Expired sessions are cleaned up
  - recent_query_pairs() returns paired turns for cache warmup
"""

import tempfile
import os
import time
import pytest
from datetime import timedelta

from app.session_manager import SessionManager


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_sessions.db")


@pytest.fixture
def sm(tmp_db):
    return SessionManager(session_timeout_minutes=30, db_path=tmp_db)


# ── Basic CRUD ─────────────────────────────────────────────────────────────────


class TestGetOrCreateSession:
    def test_creates_new_session(self, sm):
        session = sm.get_or_create_session("user1")
        assert session["user_id"] == "user1"
        assert session["conversation_history"] == []

    def test_returns_existing_session(self, sm):
        sm.add_message("user1", "user", "hello")
        session = sm.get_or_create_session("user1")
        assert len(session["conversation_history"]) == 1

    def test_tenant_id_stored(self, sm):
        session = sm.get_or_create_session("user1", tenant_id="clinic_a")
        assert session["tenant_id"] == "clinic_a"

    def test_different_users_isolated(self, sm):
        sm.add_message("alice", "user", "msg_alice")
        sm.add_message("bob", "user", "msg_bob")
        alice_hist = sm.get_conversation_history("alice")
        bob_hist = sm.get_conversation_history("bob")
        assert len(alice_hist) == 1
        assert alice_hist[0]["content"] == "msg_alice"
        assert bob_hist[0]["content"] == "msg_bob"


class TestAddMessage:
    def test_adds_user_message(self, sm):
        sm.add_message("u1", "user", "What is Paracetamol?")
        history = sm.get_conversation_history("u1")
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What is Paracetamol?"

    def test_adds_assistant_message(self, sm):
        sm.add_message("u1", "assistant", "Paracetamol is a pain reliever.")
        history = sm.get_conversation_history("u1")
        assert history[0]["role"] == "assistant"

    def test_message_order_preserved(self, sm):
        sm.add_message("u1", "user", "first")
        sm.add_message("u1", "assistant", "second")
        sm.add_message("u1", "user", "third")
        history = sm.get_conversation_history("u1")
        assert [m["content"] for m in history] == ["first", "second", "third"]

    def test_last_n_messages(self, sm):
        for i in range(10):
            sm.add_message("u1", "user", f"msg {i}")
        last5 = sm.get_conversation_history("u1", last_n=5)
        assert len(last5) == 5
        assert last5[0]["content"] == "msg 5"


class TestDocumentTracking:
    def test_add_document(self, sm):
        sm.add_document("u1", "doc_123", "/path/policy.pdf", "pdf")
        docs = sm.get_recent_documents("u1")
        assert len(docs) == 1
        assert docs[0]["document_id"] == "doc_123"
        assert docs[0]["type"] == "pdf"

    def test_recent_documents_limit(self, sm):
        for i in range(10):
            sm.add_document("u1", f"doc_{i}", f"/path/{i}.pdf")
        recent = sm.get_recent_documents("u1", limit=3)
        assert len(recent) == 3
        assert recent[-1]["document_id"] == "doc_9"


class TestContext:
    def test_set_and_get_context(self, sm):
        sm.set_context("u1", "language", "hi")
        assert sm.get_context("u1", "language") == "hi"

    def test_get_missing_context_returns_default(self, sm):
        assert sm.get_context("u1", "missing_key", default="en") == "en"

    def test_context_overwrite(self, sm):
        sm.set_context("u1", "lang", "en")
        sm.set_context("u1", "lang", "hi")
        assert sm.get_context("u1", "lang") == "hi"


class TestOCRExtraction:
    def test_add_extracted_image(self, sm):
        sm.add_extracted_image("u1", "http://img/scan.jpg", "Paracetamol 500mg")
        texts = sm.get_recent_extracted_text("u1")
        assert texts == ["Paracetamol 500mg"]

    def test_recent_limit(self, sm):
        for i in range(5):
            sm.add_extracted_image("u1", f"url_{i}", f"text_{i}")
        texts = sm.get_recent_extracted_text("u1", limit=2)
        assert len(texts) == 2
        assert texts[-1] == "text_4"


# ── Persistence (survives restart) ────────────────────────────────────────────


class TestPersistence:
    """Core GAP-008 tests: conversation history must survive server restart."""

    def test_history_survives_restart(self, tmp_db):
        # Simulate first server run
        sm1 = SessionManager(db_path=tmp_db)
        sm1.add_message("user1", "user", "What is metformin?")
        sm1.add_message("user1", "assistant", "Metformin is a diabetes medication.")

        # Simulate restart — new SessionManager instance, same db
        sm2 = SessionManager(db_path=tmp_db)
        history = sm2.get_conversation_history("user1")
        assert len(history) == 2
        assert history[0]["content"] == "What is metformin?"
        assert history[1]["content"] == "Metformin is a diabetes medication."

    def test_documents_survive_restart(self, tmp_db):
        sm1 = SessionManager(db_path=tmp_db)
        sm1.add_document("user1", "doc_abc", "/path/policy.pdf")

        sm2 = SessionManager(db_path=tmp_db)
        docs = sm2.get_recent_documents("user1")
        assert len(docs) == 1
        assert docs[0]["document_id"] == "doc_abc"

    def test_context_survives_restart(self, tmp_db):
        sm1 = SessionManager(db_path=tmp_db)
        sm1.set_context("user1", "preferred_lang", "hi")

        sm2 = SessionManager(db_path=tmp_db)
        assert sm2.get_context("user1", "preferred_lang") == "hi"

    def test_tenant_id_survives_restart(self, tmp_db):
        sm1 = SessionManager(db_path=tmp_db)
        sm1.get_or_create_session("user1", tenant_id="hospital_mumbai")

        sm2 = SessionManager(db_path=tmp_db)
        # load_session restores the persisted row without overriding tenant_id
        sm2.load_session("user1")
        assert sm2.sessions["user1"]["tenant_id"] == "hospital_mumbai"

    def test_multiple_sessions_survive_restart(self, tmp_db):
        sm1 = SessionManager(db_path=tmp_db)
        sm1.add_message("alice", "user", "hello alice")
        sm1.add_message("bob", "user", "hello bob")

        sm2 = SessionManager(db_path=tmp_db)
        assert sm2.get_conversation_history("alice")[0]["content"] == "hello alice"
        assert sm2.get_conversation_history("bob")[0]["content"] == "hello bob"

    def test_in_memory_cache_loaded_on_access(self, tmp_db):
        sm1 = SessionManager(db_path=tmp_db)
        sm1.add_message("user1", "user", "persisted message")

        sm2 = SessionManager(db_path=tmp_db)
        # Session not yet in memory cache; get_or_create_session should load from DB
        assert sm2.get_active_session_count() == 0
        sm2.get_or_create_session("user1")
        assert sm2.get_active_session_count() == 1


# ── Session lifecycle ──────────────────────────────────────────────────────────


class TestClearAndCleanup:
    def test_clear_session_removes_from_memory(self, sm):
        sm.add_message("u1", "user", "hello")
        sm.clear_session("u1")
        assert sm.get_active_session_count() == 0

    def test_clear_session_removes_from_db(self, tmp_db):
        sm1 = SessionManager(db_path=tmp_db)
        sm1.add_message("u1", "user", "hello")
        sm1.clear_session("u1")

        sm2 = SessionManager(db_path=tmp_db)
        # load_session should return False — row is gone
        assert not sm2.load_session("u1")

    def test_cleanup_expired_sessions(self, tmp_db):
        sm = SessionManager(session_timeout_minutes=0, db_path=tmp_db)  # 0-min timeout = immediately expired
        sm.get_or_create_session("u1")
        # Manually backdate last_activity so the session looks expired
        sm.sessions["u1"]["last_activity"] = sm._now() - timedelta(minutes=1)
        sm.cleanup_expired_sessions()
        assert "u1" not in sm.sessions


# ── Cache warmup data ──────────────────────────────────────────────────────────


class TestRecentQueryPairs:
    def test_returns_user_assistant_pairs(self, sm):
        sm.add_message("u1", "user", "What is ibuprofen?")
        sm.add_message("u1", "assistant", "Ibuprofen is an NSAID pain reliever.")
        pairs = sm.recent_query_pairs(limit=10)
        assert len(pairs) == 1
        assert pairs[0]["message"] == "What is ibuprofen?"
        assert pairs[0]["response"] == "Ibuprofen is an NSAID pain reliever."

    def test_skips_unpaired_turns(self, sm):
        # Two user messages in a row — no valid pair
        sm.add_message("u1", "user", "msg1")
        sm.add_message("u1", "user", "msg2")
        assert sm.recent_query_pairs() == []

    def test_filters_by_tenant_id(self, sm):
        sm.get_or_create_session("u1", tenant_id="tenant_a")
        sm.add_message("u1", "user", "question a")
        sm.add_message("u1", "assistant", "answer a")

        sm.get_or_create_session("u2", tenant_id="tenant_b")
        sm.add_message("u2", "user", "question b")
        sm.add_message("u2", "assistant", "answer b")

        pairs_a = sm.recent_query_pairs(tenant_id="tenant_a")
        assert all(p["tenant_id"] == "tenant_a" for p in pairs_a)

        pairs_b = sm.recent_query_pairs(tenant_id="tenant_b")
        assert all(p["tenant_id"] == "tenant_b" for p in pairs_b)

    def test_respects_limit(self, sm):
        for i in range(10):
            sm.add_message("u1", "user", f"question {i}")
            sm.add_message("u1", "assistant", f"answer {i}")
        pairs = sm.recent_query_pairs(limit=3)
        assert len(pairs) == 3

    def test_empty_when_no_sessions(self, sm):
        assert sm.recent_query_pairs() == []


# ── Import conversation history ────────────────────────────────────────────────


class TestImportConversationHistory:
    def test_imports_into_empty_session(self, sm):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        sm.import_conversation_history("u1", history)
        imported = sm.get_conversation_history("u1")
        assert len(imported) == 2

    def test_does_not_overwrite_existing_history(self, sm):
        sm.add_message("u1", "user", "existing message")
        sm.import_conversation_history("u1", [{"role": "user", "content": "import"}])
        history = sm.get_conversation_history("u1")
        assert len(history) == 1
        assert history[0]["content"] == "existing message"

    def test_skips_invalid_roles(self, sm):
        history = [
            {"role": "system", "content": "ignore this"},
            {"role": "user", "content": "valid"},
        ]
        sm.import_conversation_history("u1", history)
        imported = sm.get_conversation_history("u1")
        assert len(imported) == 1
        assert imported[0]["content"] == "valid"

    def test_imported_history_survives_restart(self, tmp_db):
        sm1 = SessionManager(db_path=tmp_db)
        sm1.import_conversation_history(
            "u1",
            [{"role": "user", "content": "pre-existing msg"}],
        )
        sm2 = SessionManager(db_path=tmp_db)
        history = sm2.get_conversation_history("u1")
        assert len(history) == 1


# ── format_history_for_llm ─────────────────────────────────────────────────────


class TestFormatHistoryForLlm:
    def test_formats_correctly(self, sm):
        sm.add_message("u1", "user", "Hello")
        sm.add_message("u1", "assistant", "Hi there")
        formatted = sm.format_history_for_llm("u1")
        assert "User: Hello" in formatted
        assert "Assistant: Hi there" in formatted

    def test_last_n_respected(self, sm):
        for i in range(10):
            sm.add_message("u1", "user", f"msg {i}")
        formatted = sm.format_history_for_llm("u1", last_n=2)
        assert "msg 8" in formatted
        assert "msg 9" in formatted
        assert "msg 0" not in formatted
