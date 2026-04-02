"""
HealthyPartner backend compatibility shim

- Replaces previous /hackrx/* endpoints with /healthypartner/*
- Provides example handlers for:
    GET  /health
    POST /healthypartner/run       -> expects payload with 'questions' and 'documents' (base64 or document_url)
    POST /healthypartner/generate  -> expects {"prompt": "...", "n": 5} and returns {"questions": [...]}
- IMPORTANT: Replace the placeholder `process_questions_for_document` and `generate_questions_from_prompt`
  with your real implementations (ingestion + LLM calls).

Run:
    python healthypartner_backend.py
"""

from flask import Flask, request, jsonify
import base64
import os
import traceback
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.ingestion import process_and_get_retriever

app = Flask(__name__)

# --- Configuration (tweak) ---
DOWNLOAD_DIR = os.environ.get("DOWNLOAD_PATH", "./downloaded_files")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set in the environment")

# Lazy-initialised global LLMs
llm_pro: ChatGoogleGenerativeAI | None = None
llm_flash: ChatGoogleGenerativeAI | None = None


def get_llms():
    """Initialise and cache Gemini LLM clients (mirrors app/main.py)."""
    global llm_pro, llm_flash
    if llm_pro is None or llm_flash is None:
        llm_pro = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest", google_api_key=API_KEY, temperature=0
        )
        llm_flash = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", google_api_key=API_KEY, temperature=0
        )
    return llm_pro, llm_flash



def save_base64_pdf(b64: str, filename_prefix: str = "doc") -> str:
    """Save base64-encoded PDF to disk and return filepath"""
    if not b64:
        raise ValueError("no base64 content provided")
    data = base64.b64decode(b64)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.pdf"
    path = os.path.join(DOWNLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(data)
    return path



def _format_docs(docs):
    """Same as format_docs from app/main.py"""
    return "\n\n".join(doc.page_content for doc in docs)



def process_questions_for_document(
    doc_path: str,
    questions: List[str],
    page_ranges: str | None = None,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the full ingestion + router + Gemini pipeline for a single PDF.

    Returns
    -------
    dict with keys {"answers": [...], "sources": [...], "meta": {...}}
    """
    # Ingest and build retriever/full text
    document_id = os.path.splitext(os.path.basename(doc_path))[0]
    retriever, full_text = process_and_get_retriever(doc_path, document_id)
    if not retriever:
        raise RuntimeError("Failed to process document for retrieval")

    llm_pro, llm_flash = get_llms()

    # Router chain (copied from app/main.py)
    router_template = (
        "You are an expert at routing a user's question. Based on the question, "
        "determine if it is a 'Specific Fact' question or a 'General Context' question.\n"
        "- 'Specific Fact' questions ask for a precise number, date, name, or a waiting "
        "period for a named item (e.g., \"What is the waiting period for cataracts?\", "
        "\"What is the limit for room rent?\").\n"
        "- 'General Context' questions are broader and ask for summaries or conditions "
        "(e.g., \"Does this policy cover maternity?\", \"Summarize the organ donor rules.\").\n"
        "Return only the single word 'Specific Fact' or 'General Context'.\n"
        "Question: {question}\n"
        "Classification:"
    )
    router_prompt = ChatPromptTemplate.from_template(router_template)
    router_chain = router_prompt | llm_flash | StrOutputParser()

    # Final answer prompt (copied from app/main.py)
    final_prompt_template = """You are a highly precise Q&A engine that answers questions based ONLY on the provided CONTEXT.

CONTEXT:
{context}

QUESTION:
{question}

**INSTRUCTIONS FOR YOUR ANSWER:**
1.  **Strictly Adhere to Context:** Your answer MUST be based exclusively on the information within the provided CONTEXT.
2.  **CRITICAL REASONING RULE:** To answer the question, you may need to synthesize scattered information. For questions about a 'waiting period' for a specific procedure, you MUST find where the procedure is listed and what waiting period category it falls under.
3.  **Conciseness vs. Completeness Rule:** Your answer MUST be a single, concise paragraph. 
    - For **definitional questions** (e.g., "How is a 'Hospital' defined?"), you MUST be comprehensive and include all specific criteria listed in the context (like bed counts, staff, etc.).
    - For **all other questions**, you MUST be ruthlessly concise and include ONLY the information that directly answers the question. Do not include extra details.
4.  **Format:** If the question is objective, you MUST begin your answer with "Yes," or "No,".
5.  **Data Extraction:** You MUST extract precise numerical values and percentages from the context.
6.  **Missing Information:** If the information is not in the context, state only: "This information is not available in the provided document."

**ANSWER:**"""
    final_prompt = ChatPromptTemplate.from_template(final_prompt_template)

    answers: List[str] = []
    sources: List[Any] = []

    for q in questions:
        try:
            route = router_chain.invoke({"question": q})
            print(f"--- Answering question: {q} ---")
            print(f"Router choice: {route}")

            if "Specific Fact" in route:
                # Path A: use the full text for precise extraction
                context = full_text
                chain = final_prompt | llm_pro | StrOutputParser()
                answer = chain.invoke({"context": context, "question": q})
                # For sources, still retrieve a few chunks as evidence
                retrieved_docs = retriever.get_relevant_documents(q)
                snippet = _format_docs(retrieved_docs[:1]) if retrieved_docs else ""
                sources.append({"type": "full_text", "snippet": snippet})
            else:
                # Path B: retrieval-augmented generation
                chain = (
                    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
                    | final_prompt
                    | llm_pro
                    | StrOutputParser()
                )
                answer = chain.invoke(q)
                retrieved_docs = retriever.get_relevant_documents(q)
                snippet = _format_docs(retrieved_docs[:2]) if retrieved_docs else ""
                sources.append({"type": "retrieval", "snippet": snippet})

            answers.append(answer)
        except Exception as e:  # keep pipeline robust per-question
            print(f"Error on question '{q}': {e}")
            answers.append(f"Error processing question: {e}")
            sources.append({"type": "error", "message": str(e)})

    meta_out = {
        "model_used": meta.get("model") if meta else "gemini-1.5-pro-latest",
        "temperature": meta.get("temperature") if meta else 0,
        "document_id": document_id,
    }

    return {"answers": answers, "sources": sources, "meta": meta_out}


def generate_questions_from_prompt_backend(prompt: str, n: int = 5) -> List[str]:
    """Simple backend-side question generator.

    You can later replace this with a Gemini call if you want smarter generation.
    """
    if not prompt:
        return [f"Generic question #{i+1}" for i in range(n)]
    tokens = [t.strip() for t in prompt.split() if len(t.strip()) > 3][:n]
    if not tokens:
        return [f"What does the document say about {prompt}?"]
    return [f"What information is provided about {tok}?" for tok in tokens[:n]]


# --- Routes ---


@app.get("/health")
def health():
    """Simple healthcheck returning component status"""
    info = {
        "status": "ok",
        "service": "HealthyPartner backend",
        "time": datetime.utcnow().isoformat(),
        "routes": [
            "/healthypartner/run",
            "/healthypartner/generate",
            "/webhook",
            "/test",
        ],
    }
    return jsonify(info)


@app.post("/healthypartner/run")
def hp_run():
    """Main document Q&A endpoint."""
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "missing json body"}), 400

        questions = payload.get("questions") or []
        if not questions:
            return jsonify({"error": "no questions provided"}), 400

        # accept base64 document or document_url
        doc_b64 = payload.get("documents")
        doc_url = payload.get("document_url")
        page_ranges = payload.get("page_ranges")
        model = payload.get("model")
        meta = {"model": model, "temperature": payload.get("temperature")}

        # Save base64 to disk if provided
        doc_path = None
        if doc_b64:
            doc_path = save_base64_pdf(doc_b64, filename_prefix="hp_doc")
        elif doc_url:
            # Option A: download doc here (simple approach)
            import requests

            r = requests.get(doc_url, timeout=30)
            if r.status_code != 200:
                return (
                    jsonify(
                        {
                            "error": f"failed to fetch document_url (status {r.status_code})"
                        }
                    ),
                    400,
                )
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"remote_{ts}.pdf"
            doc_path = os.path.join(DOWNLOAD_DIR, filename)
            with open(doc_path, "wb") as f:
                f.write(r.content)
        else:
            return (
                jsonify(
                    {"error": "no document content provided (documents or document_url)"}
                ),
                400,
            )

        # Call your processing function (replace with production pipeline)
        result = process_questions_for_document(
            doc_path, questions, page_ranges=page_ranges, meta=meta
        )

        answers = result.get("answers", [])
        sources = result.get("sources", [])
        meta_out = result.get("meta", {})

        response = {"answers": answers, "sources": sources, "meta": meta_out}
        return jsonify(response)

    except Exception as e:  # pragma: no cover - defensive
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/healthypartner/generate")
def hp_generate():
    """Generate question templates from a short prompt."""
    try:
        payload = request.get_json(force=True) or {}
        prompt = payload.get("prompt", "")
        n = int(payload.get("n", 5))
        questions = generate_questions_from_prompt_backend(prompt, n)
        return jsonify({"questions": questions})
    except Exception as e:  # pragma: no cover - defensive
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.post("/webhook")
def webhook():
    """Twilio webhook / other webhook entrypoint."""
    # For compatibility, simply acknowledge if not integrated
    data = request.values.to_dict() or request.get_json(silent=True) or {}
    # TODO: plug into your intent detection / agent pipeline
    return jsonify({"status": "received", "data": data})


@app.post("/test")
def test():
    payload = request.get_json(force=True) or {}
    # simple echo for local testing
    return jsonify({"ok": True, "echo": payload})


if __name__ == "__main__":
    # Flask dev server (use gunicorn / uvicorn for production)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
