# healthcare_chatbot.py
"""
Flask server for Twilio WhatsApp/SMS webhook integration.

v2: Uses local LLMEngine instead of cloud APIs.
No API keys required for AI functionality.
Twilio credentials only needed for WhatsApp/SMS messaging.
"""

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator

from app.llm_engine import LLMEngine
from app.session_manager import SessionManager
from app.ocr_processor import OCRProcessor
from app.healthcare_agent import HealthcareAgent

# Load environment variables
load_dotenv()

# Configuration (Twilio is optional — only for WhatsApp/SMS)
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")

# Initialize Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

# Initialize components — all local, no cloud keys needed
print("🏥 Initializing HealthyPartner v2 chatbot...")
engine = LLMEngine()
session_manager = SessionManager(session_timeout_minutes=30)
ocr_processor = OCRProcessor(languages=["en"], gpu=False)
healthcare_agent = HealthcareAgent(session_manager, engine=engine)

status = engine.ensure_models_available()
print(f"   Engine: {engine}")
print(f"   Models: main={'✅' if status['main_model'] else '❌'} "
      f"fast={'✅' if status['fast_model'] else '❌'}")
print("✅ All components initialized\n")


# ── Helper Functions ───────────────────────────────────────────────────────────


def validate_twilio_request(request_obj):
    """Validate that the request came from Twilio."""
    if not TWILIO_AUTH_TOKEN:
        return True  # Skip in dev if token not set
    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    return validator.validate(
        request_obj.url,
        request_obj.form,
        request_obj.headers.get("X-TWILIO-SIGNATURE", ""),
    )


def extract_user_id(request_obj):
    """Extract user identifier from Twilio request."""
    return request_obj.values.get("From", "unknown_user")


def get_media_urls(request_obj):
    """Extract media URLs from Twilio MMS message."""
    num_media = int(request_obj.values.get("NumMedia", 0))
    return [
        request_obj.values.get(f"MediaUrl{i}")
        for i in range(num_media)
        if request_obj.values.get(f"MediaUrl{i}")
    ]


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.route("/")
def home():
    return jsonify({
        "service": "HealthyPartner v2 — AI Healthcare Chatbot",
        "status": "running",
        "version": "2.0.0",
        "engine": str(engine),
        "tier": engine.tier,
        "cloud_dependency": False,
        "capabilities": [
            "Insurance policy Q&A",
            "Prescription analysis (OCR)",
            "Lab report interpretation (OCR)",
            "General health queries",
            "Hindi / multilingual support",
            "Ayushman Bharat / PMJAY guidance",
        ],
    })


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "HealthyPartner v2",
        "engine": engine.health_check(),
        "active_sessions": session_manager.get_active_session_count(),
        "twilio_configured": bool(TWILIO_AUTH_TOKEN),
    })


@app.route("/webhook", methods=["POST"])
def webhook():
    """Main Twilio webhook for incoming WhatsApp/SMS messages."""
    if not validate_twilio_request(request):
        print("⚠️ Invalid Twilio signature")
        return "Forbidden", 403

    user_id = extract_user_id(request)
    incoming_msg = request.values.get("Body", "").strip()

    print(f"\n--- Message from {user_id}: {incoming_msg} ---")

    resp = MessagingResponse()

    try:
        # Handle media attachments (OCR)
        ocr_results = None
        media_urls = get_media_urls(request)
        if media_urls:
            print(f"📷 Processing {len(media_urls)} attachment(s)")
            auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None
            ocr_results = ocr_processor.process_medical_document(
                media_urls[0], is_url=True, auth=auth
            )
            if ocr_results.get("success"):
                print(f"✅ OCR: {len(ocr_results['text'])} chars")
                if not incoming_msg:
                    doc_type = ocr_results.get("document_type", "document")
                    incoming_msg = f"What can you tell me about this {doc_type}?"

        # Process through orchestration pipeline
        result = healthcare_agent.process_user_message(
            user_id=user_id, message=incoming_msg, ocr_results=ocr_results
        )

        print(f"   Intent: {result['intent']}")
        print(f"   Response: {result['response'][:100]}...")

        msg = resp.message()
        msg.body(result["response"])

        session_manager.cleanup_expired_sessions()
        return str(resp)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        resp.message("I'm sorry, I encountered an error. Please try again.")
        return str(resp)


@app.route("/test", methods=["POST"])
def test_endpoint():
    """Test endpoint for local development without Twilio."""
    data = request.get_json()
    user_id = data.get("user_id", "test_user")
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        result = healthcare_agent.process_user_message(
            user_id=user_id, message=message, ocr_results=None
        )
        return jsonify({
            "success": True,
            "user_id": user_id,
            "intent": result["intent"],
            "response": result["response"],
            "engine": str(engine),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/session/<user_id>", methods=["GET"])
def get_session(user_id):
    history = session_manager.get_conversation_history(user_id)
    return jsonify({
        "user_id": user_id,
        "message_count": len(history),
        "recent_messages": history[-5:] if history else [],
    })


@app.route("/session/<user_id>", methods=["DELETE"])
def clear_session(user_id):
    session_manager.clear_session(user_id)
    return jsonify({"success": True, "message": f"Session cleared for {user_id}"})


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
    print("🏥 HealthyPartner v2 — Healthcare Chatbot")
    print("=" * 50)
    print(f"  Engine: {engine}")
    print(f"  Server: http://0.0.0.0:5000")
    print(f"  Webhook: http://your-domain.com/webhook")
    print(f"  Test:    http://localhost:5000/test")
    print(f"  No cloud API keys required.")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=True)
