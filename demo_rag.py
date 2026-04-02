#!/usr/bin/env python
# demo_rag.py

import os
from app.session_manager import SessionManager
from app.healthcare_agent import HealthcareAgent

def main():
    print("==================================================")
    print("🧠 RAG (Knowledge Base) Demonstration for Llama 3")
    print("==================================================")
    
    # 1. Initialize components
    print("\n[1/4] Starting Session Manager and ChatOllama...")
    session_manager = SessionManager()
    agent = HealthcareAgent(session_manager=session_manager, model_name="llama3")
    user_id = "rag_test_user"

    # 2. Add document to the user's session
    doc_path = os.path.abspath("dummy_policy.md")
    print(f"\n[2/4] Uploading Document to Session: {doc_path}")
    session_manager.add_document(user_id, document_id="doc_001", document_path=doc_path, doc_type="md")

    # 3. Ask a question that requires the document context
    question = "Does my insurance cover maternity care, and what is the deductible?"
    print(f"\n[3/4] Asking Question: '{question}'")
    
    # 4. Process the message
    # The healthcare agent will detect "INSURANCE_QUERY", see the document,
    # and automatically run "app.ingestion.process_and_get_retriever" to chunk it into ChromaDB!
    print("      (Please wait, Llama 3 is reading the document via ChromaDB...)")
    result = agent.process_user_message(user_id=user_id, message=question)
    
    print("\n[4/4] Result Generated!")
    print(f"Detected Intent: {result['intent']}")
    print(f"\n🤖 Llama 3 Answer:\n{result['response']}")

if __name__ == "__main__":
    main()
