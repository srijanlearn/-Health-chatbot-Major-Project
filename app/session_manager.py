# app/session_manager.py

import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import os

class SessionManager:
    """
    Manages user sessions, conversation history, and document context.
    Stores data in memory with optional file-based persistence.
    """
    
    def __init__(self, session_timeout_minutes: int = 30, persist_dir: str = "./sessions"):
        self.sessions: Dict[str, dict] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
    
    def get_or_create_session(self, user_id: str) -> dict:
        """Get existing session or create new one for user."""
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                "user_id": user_id,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "conversation_history": [],
                "documents": [],
                "extracted_images": [],
                "context": {}
            }
        else:
            # Update last activity
            self.sessions[user_id]["last_activity"] = datetime.now()
        
        return self.sessions[user_id]
    
    def add_message(self, user_id: str, role: str, content: str):
        """Add a message to conversation history."""
        session = self.get_or_create_session(user_id)
        session["conversation_history"].append({
            "role": role,  # 'user' or 'assistant'
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_conversation_history(self, user_id: str, last_n: Optional[int] = None) -> List[dict]:
        """Get conversation history, optionally limited to last N messages."""
        session = self.get_or_create_session(user_id)
        history = session["conversation_history"]
        if last_n:
            return history[-last_n:]
        return history
    
    def add_document(self, user_id: str, document_id: str, document_path: str, doc_type: str = "pdf"):
        """Track uploaded documents for user."""
        session = self.get_or_create_session(user_id)
        session["documents"].append({
            "document_id": document_id,
            "path": document_path,
            "type": doc_type,
            "uploaded_at": datetime.now().isoformat()
        })
    
    def add_extracted_image(self, user_id: str, image_url: str, extracted_text: str):
        """Store OCR extracted text from images."""
        session = self.get_or_create_session(user_id)
        session["extracted_images"].append({
            "image_url": image_url,
            "extracted_text": extracted_text,
            "extracted_at": datetime.now().isoformat()
        })
    
    def set_context(self, user_id: str, key: str, value: any):
        """Store custom context data for user."""
        session = self.get_or_create_session(user_id)
        session["context"][key] = value
    
    def get_context(self, user_id: str, key: str, default: any = None) -> any:
        """Retrieve custom context data."""
        session = self.get_or_create_session(user_id)
        return session["context"].get(key, default)
    
    def get_recent_documents(self, user_id: str, limit: int = 5) -> List[dict]:
        """Get recently uploaded documents."""
        session = self.get_or_create_session(user_id)
        return session["documents"][-limit:]
    
    def get_recent_extracted_text(self, user_id: str, limit: int = 3) -> List[str]:
        """Get recently extracted text from images."""
        session = self.get_or_create_session(user_id)
        recent = session["extracted_images"][-limit:]
        return [item["extracted_text"] for item in recent]
    
    def clear_session(self, user_id: str):
        """Clear user session data."""
        if user_id in self.sessions:
            del self.sessions[user_id]
    
    def cleanup_expired_sessions(self):
        """Remove sessions that have exceeded timeout."""
        current_time = datetime.now()
        expired = [
            user_id for user_id, session in self.sessions.items()
            if current_time - session["last_activity"] > self.session_timeout
        ]
        for user_id in expired:
            print(f"Cleaning up expired session for user: {user_id}")
            del self.sessions[user_id]
    
    def save_session(self, user_id: str):
        """Persist session to disk."""
        if user_id in self.sessions:
            session_file = os.path.join(self.persist_dir, f"{user_id}.json")
            session_data = self.sessions[user_id].copy()
            # Convert datetime objects to strings
            session_data["created_at"] = session_data["created_at"].isoformat()
            session_data["last_activity"] = session_data["last_activity"].isoformat()
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
    
    def load_session(self, user_id: str) -> bool:
        """Load session from disk if exists."""
        session_file = os.path.join(self.persist_dir, f"{user_id}.json")
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            # Convert string timestamps back to datetime
            session_data["created_at"] = datetime.fromisoformat(session_data["created_at"])
            session_data["last_activity"] = datetime.fromisoformat(session_data["last_activity"])
            self.sessions[user_id] = session_data
            return True
        return False
    
    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self.sessions)
    
    def format_history_for_llm(self, user_id: str, last_n: int = 5) -> str:
        """Format conversation history as context for LLM."""
        history = self.get_conversation_history(user_id, last_n)
        formatted = []
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role_label}: {msg['content']}")
        return "\n".join(formatted)
