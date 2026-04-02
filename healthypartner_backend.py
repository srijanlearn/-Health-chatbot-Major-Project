"""
HealthyPartner v2 — Unified Backend Server

Combines all endpoints into a single Flask server:
- /health              → System health check
- /healthypartner/run  → Document Q&A (main endpoint)
- /healthypartner/generate → Question template generation
- /webhook             → Twilio WhatsApp/SMS webhook
- /test                → Local testing endpoint
- /system/info         → Hardware & model info

No cloud API keys required. All inference runs locally via Ollama.
"""

import os
import base64
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, request, jsonify

from app.llm_engine import LLMEngine, detect_system
from app.ingestion import process_and_get_retriever
from app.prompts.insurance_qa import (
    INSURANCE_SYSTEM_PROMPT,
    INSURANCE_SPECIFIC_FACT_PROMPT,
    INSURANCE_GENERAL_PROMPT,
)
from app.prompts.router import ROUTER_SYSTEM_PROMPT, route_by_keywords

load_dotenv()

app = Flask(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

DOWNLOAD_DIR = os.environ.get("DOWNLOAD_PATH", "./downloaded_files")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ── Initialise local LLM engine (no API keys) ─────────────────────────────────

print("🏥 HealthyPartner v2 — Initialising local LLM engine...")
engine = LLMEngine()
status = engine.ensure_models_available()
print(f"   Engine: {engine}")
print(f"   Models: main={'✅' if status['main_model'] else '❌'} "
      f"fast={'✅' if status['fast_model'] else '❌'}")
print("   Ready.\n")


# ── Helper functions ───────────────────────────────────────────────────────────


def save_base64_pdf(b64: str, prefix: str = "doc") -> str:
    """Decode base64 PDF and save to disk."""
    data = base64.b64decode(b64)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts}.pdf"
    path = os.path.join(DOWNLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(data)
    return path


def process_questions_for_document(
    doc_path: str,
    questions: List[str],
    page_ranges: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full ingestion + routing + local LLM pipeline for a PDF."""

    document_id = os.path.splitext(os.path.basename(doc_path))[0]
    retriever, full_text = process_and_get_retriever(doc_path, document_id)
    if not retriever:
        raise RuntimeError("Failed to process document for retrieval")

    answers: List[str] = []
    sources: List[Any] = []

    for q in questions:
        try:
            # Route question
            route = route_by_keywords(q)
            if route is None:
                route_raw = engine.classify(q, system_prompt=ROUTER_SYSTEM_PROMPT)
                route = "specific_fact" if "specific" in route_raw.lower() else "general_rag"

            print(f"--- Answering: {q} (route: {route}) ---")

            if route == "specific_fact":
                doc_context = full_text[:12_000]
                prompt = INSURANCE_SPECIFIC_FACT_PROMPT.format(
                    context=doc_context, question=q
                )
                answer = engine.generate(
                    prompt=prompt,
                    system_prompt=INSURANCE_SYSTEM_PROMPT,
                    think=True,
                )
                sources.append({"type": "full_text"})
            else:
                docs = retriever.invoke(q)
                chunk_context = "\n\n".join(d.page_content for d in docs[:5])
                prompt = INSURANCE_GENERAL_PROMPT.format(
                    context=chunk_context, question=q
                )
                answer = engine.generate(
                    prompt=prompt,
                    system_prompt=INSURANCE_SYSTEM_PROMPT,
                )
                snippet = "\n\n".join(d.page_content for d in docs[:2])
                sources.append({"type": "retrieval", "snippet": snippet})

            answers.append(answer)
        except Exception as e:
            print(f"Error on question '{q}': {e}")
            answers.append(f"Error processing question: {e}")
            sources.append({"type": "error", "message": str(e)})

    return {
        "answers": answers,
        "sources": sources,
        "meta": {
            "engine": str(engine),
            "tier": engine.tier,
            "document_id": document_id,
        },
    }


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    """System health check with model and hardware info."""
    return jsonify({
        "status": "ok",
        "service": "HealthyPartner v2",
        "time": datetime.utcnow().isoformat(),
        "engine": engine.health_check(),
    })


@app.get("/system/info")
def system_info():
    """Hardware detection and model tier recommendation."""
    info = detect_system().to_dict()
    info["current_tier"] = engine.tier
    info["main_model"] = engine.main_model
    info["fast_model"] = engine.fast_model
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

        doc_b64 = payload.get("documents")
        doc_url = payload.get("document_url")
        page_ranges = payload.get("page_ranges")
        meta = {"model": engine.main_model, "temperature": payload.get("temperature")}

        # Get document
        doc_path = None
        if doc_b64:
            doc_path = save_base64_pdf(doc_b64, prefix="hp_doc")
        elif doc_url:
            import requests as http_req

            r = http_req.get(doc_url, timeout=30)
            if r.status_code != 200:
                return jsonify({"error": f"Failed to fetch URL (status {r.status_code})"}), 400
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            doc_path = os.path.join(DOWNLOAD_DIR, f"remote_{ts}.pdf")
            with open(doc_path, "wb") as f:
                f.write(r.content)
        else:
            return jsonify({"error": "No document provided (documents or document_url)"}), 400

        result = process_questions_for_document(doc_path, questions, page_ranges, meta)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.post("/healthypartner/generate")
def hp_generate():
    """Generate question templates from a short prompt."""
    try:
        payload = request.get_json(force=True) or {}
        prompt = payload.get("prompt", "")
        n = int(payload.get("n", 5))

        if not prompt:
            return jsonify({"questions": [f"Generic question #{i+1}" for i in range(n)]})

        # Use LLM to generate relevant questions
        sys_prompt = (
            f"Generate exactly {n} specific questions someone might ask about "
            f"a health insurance policy related to: {prompt}. "
            f"Return only the questions, one per line, numbered."
        )
        raw = engine.generate(prompt=f"Topic: {prompt}", system_prompt=sys_prompt)
        questions = [
            line.strip().lstrip("0123456789.-) ")
            for line in raw.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ][:n]

        return jsonify({"questions": questions})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.post("/webhook")
def webhook():
    """Twilio webhook / other webhook entrypoint."""
    data = request.values.to_dict() or request.get_json(silent=True) or {}
    return jsonify({"status": "received", "data": data})


@app.post("/test")
def test():
    """Local testing endpoint — echo + model info."""
    payload = request.get_json(force=True) or {}
    return jsonify({
        "ok": True,
        "echo": payload,
        "engine": str(engine),
        "tier": engine.tier,
    })


# ── Error handlers ─────────────────────────────────────────────────────────────


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(_):
    return jsonify({"error": "Internal server error"}), 500


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🏥 HealthyPartner v2 — Local Healthcare AI")
    print("=" * 50)
    print(f"  Engine: {engine}")
    print(f"  Tier:   {engine.tier}")
    print(f"  Server: http://0.0.0.0:5000")
    print(f"  No API keys required. Running 100% locally.")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
