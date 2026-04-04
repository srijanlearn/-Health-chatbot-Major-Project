"""
SQLite-backed session persistence for conversational state.

Sessions remain cached in memory for fast request handling, but every
mutation is flushed to SQLite so conversation history survives restarts.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_SESSION_DB_PATH = "data/sessions.db"

_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    user_id               TEXT PRIMARY KEY,
    tenant_id             TEXT NOT NULL,
    created_at            TEXT NOT NULL,
    last_activity         TEXT NOT NULL,
    conversation_history  TEXT NOT NULL,
    documents             TEXT NOT NULL,
    extracted_images      TEXT NOT NULL,
    context               TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity);
"""


class SessionManager:
    """
    Manages user sessions, conversation history, and document context.

    Public methods intentionally stay close to the original implementation so
    existing call sites do not need a larger refactor.
    """

    def __init__(
        self,
        session_timeout_minutes: int = 30,
        db_path: str = DEFAULT_SESSION_DB_PATH,
    ) -> None:
        self.sessions: Dict[str, dict] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_DDL)

    @staticmethod
    def _now() -> datetime:
        return datetime.now()

    def _new_session(self, user_id: str, tenant_id: str = "default") -> dict:
        now = self._now()
        return {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "created_at": now,
            "last_activity": now,
            "conversation_history": [],
            "documents": [],
            "extracted_images": [],
            "context": {},
        }

    @staticmethod
    def _serialize_session(session: dict) -> tuple:
        return (
            session["user_id"],
            session.get("tenant_id", "default"),
            session["created_at"].isoformat(),
            session["last_activity"].isoformat(),
            json.dumps(session["conversation_history"]),
            json.dumps(session["documents"]),
            json.dumps(session["extracted_images"]),
            json.dumps(session["context"]),
        )

    @staticmethod
    def _deserialize_row(row: sqlite3.Row) -> dict:
        return {
            "user_id": row["user_id"],
            "tenant_id": row["tenant_id"],
            "created_at": datetime.fromisoformat(row["created_at"]),
            "last_activity": datetime.fromisoformat(row["last_activity"]),
            "conversation_history": json.loads(row["conversation_history"]),
            "documents": json.loads(row["documents"]),
            "extracted_images": json.loads(row["extracted_images"]),
            "context": json.loads(row["context"]),
        }

    def _persist_session(self, session: dict) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    user_id, tenant_id, created_at, last_activity,
                    conversation_history, documents, extracted_images, context
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    tenant_id=excluded.tenant_id,
                    created_at=excluded.created_at,
                    last_activity=excluded.last_activity,
                    conversation_history=excluded.conversation_history,
                    documents=excluded.documents,
                    extracted_images=excluded.extracted_images,
                    context=excluded.context
                """,
                self._serialize_session(session),
            )

    def get_or_create_session(self, user_id: str, tenant_id: str = "default") -> dict:
        """Get existing session or create a new one for user_id."""
        session = self.sessions.get(user_id)
        if session is None and self.load_session(user_id):
            session = self.sessions[user_id]

        if session is None:
            session = self._new_session(user_id, tenant_id=tenant_id)
            self.sessions[user_id] = session
        else:
            session["last_activity"] = self._now()
            session["tenant_id"] = tenant_id or session.get("tenant_id", "default")

        self._persist_session(session)
        return session

    def add_message(self, user_id: str, role: str, content: str) -> None:
        """Add a message to conversation history."""
        session = self.get_or_create_session(user_id)
        session["conversation_history"].append(
            {
                "role": role,
                "content": content,
                "timestamp": self._now().isoformat(),
            }
        )
        session["last_activity"] = self._now()
        self._persist_session(session)

    def get_conversation_history(self, user_id: str, last_n: Optional[int] = None) -> List[dict]:
        """Get conversation history, optionally limited to last N messages."""
        session = self.get_or_create_session(user_id)
        history = session["conversation_history"]
        if last_n:
            return history[-last_n:]
        return history

    def add_document(
        self,
        user_id: str,
        document_id: str,
        document_path: str,
        doc_type: str = "pdf",
    ) -> None:
        """Track uploaded documents for user."""
        session = self.get_or_create_session(user_id)
        session["documents"].append(
            {
                "document_id": document_id,
                "path": document_path,
                "type": doc_type,
                "uploaded_at": self._now().isoformat(),
            }
        )
        session["last_activity"] = self._now()
        self._persist_session(session)

    def add_extracted_image(self, user_id: str, image_url: str, extracted_text: str) -> None:
        """Store OCR extracted text from images."""
        session = self.get_or_create_session(user_id)
        session["extracted_images"].append(
            {
                "image_url": image_url,
                "extracted_text": extracted_text,
                "extracted_at": self._now().isoformat(),
            }
        )
        session["last_activity"] = self._now()
        self._persist_session(session)

    def set_context(self, user_id: str, key: str, value: Any) -> None:
        """Store custom context data for user."""
        session = self.get_or_create_session(user_id)
        session["context"][key] = value
        session["last_activity"] = self._now()
        self._persist_session(session)

    def get_context(self, user_id: str, key: str, default: Any = None) -> Any:
        """Retrieve custom context data."""
        session = self.get_or_create_session(user_id)
        return session["context"].get(key, default)

    def get_recent_documents(self, user_id: str, limit: int = 5) -> List[dict]:
        """Get recently uploaded documents."""
        session = self.get_or_create_session(user_id)
        return session["documents"][-limit:]

    def get_recent_extracted_text(self, user_id: str, limit: int = 3) -> List[str]:
        """Get recently extracted OCR text snippets."""
        session = self.get_or_create_session(user_id)
        recent = session["extracted_images"][-limit:]
        return [item["extracted_text"] for item in recent]

    def clear_session(self, user_id: str) -> None:
        """Clear user session data from memory and SQLite."""
        self.sessions.pop(user_id, None)
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))

    def cleanup_expired_sessions(self) -> None:
        """Remove sessions that have exceeded timeout."""
        current_time = self._now()
        expired = []
        for user_id, session in list(self.sessions.items()):
            if current_time - session["last_activity"] > self.session_timeout:
                expired.append(user_id)

        for user_id in expired:
            logger.info("Cleaning up expired session for user=%s", user_id)
            self.clear_session(user_id)

    def save_session(self, user_id: str) -> None:
        """Persist one cached session to SQLite."""
        if user_id in self.sessions:
            self._persist_session(self.sessions[user_id])

    def load_session(self, user_id: str) -> bool:
        """Load session from SQLite if it exists."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return False
        self.sessions[user_id] = self._deserialize_row(row)
        return True

    def get_active_session_count(self) -> int:
        """Get count of active sessions currently cached in memory."""
        return len(self.sessions)

    def format_history_for_llm(self, user_id: str, last_n: int = 5) -> str:
        """Format conversation history as context for LLM."""
        history = self.get_conversation_history(user_id, last_n)
        formatted = []
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role_label}: {msg['content']}")
        return "\n".join(formatted)

    def import_conversation_history(
        self,
        user_id: str,
        conversation_history: List[dict],
        tenant_id: str = "default",
    ) -> None:
        """Seed a session from client-supplied history when no local state exists yet."""
        session = self.get_or_create_session(user_id, tenant_id=tenant_id)
        if session["conversation_history"]:
            return
        imported = []
        for turn in conversation_history:
            role = turn.get("role")
            content = turn.get("content")
            if role in {"user", "assistant"} and content:
                imported.append(
                    {
                        "role": role,
                        "content": content,
                        "timestamp": turn.get("timestamp", self._now().isoformat()),
                    }
                )
        if imported:
            session["conversation_history"].extend(imported)
            session["last_activity"] = self._now()
            self._persist_session(session)

    def recent_query_pairs(
        self,
        limit: int = 100,
        tenant_id: Optional[str] = None,
    ) -> List[dict]:
        """
        Return recent user-query/assistant-response pairs for cache warmup.

        Each row has: user_id, tenant_id, message, response.
        """
        where = "WHERE tenant_id = ?" if tenant_id else ""
        params: tuple = (tenant_id,) if tenant_id else ()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT user_id, tenant_id, conversation_history
                FROM sessions
                {where}
                ORDER BY last_activity DESC
                LIMIT ?
                """,
                params + (min(limit, 200),),
            ).fetchall()

        pairs: List[dict] = []
        for row in rows:
            history = json.loads(row["conversation_history"])
            for idx, turn in enumerate(history[:-1]):
                next_turn = history[idx + 1]
                if turn.get("role") != "user" or next_turn.get("role") != "assistant":
                    continue
                message = turn.get("content", "").strip()
                response = next_turn.get("content", "").strip()
                if not message or not response:
                    continue
                pairs.append(
                    {
                        "user_id": row["user_id"],
                        "tenant_id": row["tenant_id"],
                        "message": message,
                        "response": response,
                    }
                )
        return pairs[:limit]
