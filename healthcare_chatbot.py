# healthcare_chatbot.py

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator

from app.session_manager import SessionManager
from app.ocr_processor import OCRProcessor
from app.healthcare_agent import HealthcareAgent

# Load environment variables
load_dotenv()

# Configuration
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize components
print("Initializing Healthcare Chatbot components...")
session_manager = SessionManager(session_timeout_minutes=30)
ocr_processor = OCRProcessor(languages=['en'], gpu=False)
healthcare_agent = HealthcareAgent(session_manager)
print("✅ All components initialized successfully")


# --- Helper Functions ---

def validate_twilio_request(request_obj):
    """
    Validate that the request came from Twilio.
    Returns True if valid, False otherwise.
    """
    if not TWILIO_AUTH_TOKEN:
        # Skip validation in development if token not set
        return True
    
    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    request_valid = validator.validate(
        request_obj.url,
        request_obj.form,
        request_obj.headers.get('X-TWILIO-SIGNATURE', '')
    )
    return request_valid


def extract_user_id(request_obj):
    """Extract user identifier from Twilio request."""
    # Use phone number as user ID
    return request_obj.values.get('From', 'unknown_user')


def has_media(request_obj):
    """Check if message contains media attachments."""
    num_media = int(request_obj.values.get('NumMedia', 0))
    return num_media > 0


def get_media_urls(request_obj):
    """Extract media URLs from Twilio MMS message."""
    num_media = int(request_obj.values.get('NumMedia', 0))
    media_urls = []
    
    for i in range(num_media):
        media_url = request_obj.values.get(f'MediaUrl{i}')
        if media_url:
            media_urls.append(media_url)
    
    return media_urls


# --- Routes ---

@app.route('/')
def home():
    """Home endpoint - basic info."""
    return jsonify({
        "service": "AI-Driven Healthcare Chatbot",
        "status": "running",
        "version": "1.0",
        "capabilities": [
            "Insurance policy Q&A",
            "Prescription analysis (OCR)",
            "Lab report interpretation (OCR)",
            "General health queries",
            "Multi-turn conversations"
        ],
        "endpoints": {
            "/": "Service information",
            "/health": "Health check",
            "/webhook": "Twilio webhook (POST only)"
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    agent_status = "initialized" if healthcare_agent else "not initialized"
    
    return jsonify({
        "status": "healthy",
        "service": "Healthcare Chatbot",
        "components": {
            "session_manager": "active",
            "ocr_processor": "active",
            "healthcare_agent": agent_status
        },
        "active_sessions": session_manager.get_active_session_count(),
        "local_llm_configured": True,
        "twilio_configured": bool(TWILIO_AUTH_TOKEN)
    })


@app.route('/webhook', methods=['POST'])
def webhook():
    """
    Main Twilio webhook endpoint for incoming messages.
    Handles both text messages and MMS with images.
    """
    # Validate request is from Twilio
    if not validate_twilio_request(request):
        print("⚠️  Invalid Twilio signature")
        return "Forbidden", 403
    
    # Check if healthcare agent is initialized
    if not healthcare_agent:
        resp = MessagingResponse()
        resp.message("Sorry, the chatbot is not properly configured. Please contact support.")
        return str(resp)
    
    # Extract message data
    user_id = extract_user_id(request)
    incoming_msg = request.values.get('Body', '').strip()
    
    print(f"\n--- Incoming message from {user_id} ---")
    print(f"Message: {incoming_msg}")
    
    # Initialize response
    resp = MessagingResponse()
    
    try:
        # Check for media attachments (images)
        ocr_results = None
        if has_media(request):
            media_urls = get_media_urls(request)
            print(f"📷 Processing {len(media_urls)} media attachment(s)")
            
            # Process first image (handle multiple in future enhancement)
            if media_urls:
                # Twilio requires authentication to download media
                auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None
                
                ocr_results = ocr_processor.process_medical_document(
                    media_urls[0],
                    is_url=True,
                    auth=auth
                )
                
                if ocr_results.get("success"):
                    print(f"✅ OCR extracted {len(ocr_results['text'])} characters")
                    print(f"📄 Document type: {ocr_results.get('document_type', 'unknown')}")
                    
                    # If no message text, provide default based on document type
                    if not incoming_msg:
                        doc_type = ocr_results.get('document_type', 'document')
                        incoming_msg = f"What can you tell me about this {doc_type}?"
                else:
                    print(f"❌ OCR failed: {ocr_results.get('error')}")
        
        # Process message with healthcare agent
        result = healthcare_agent.process_user_message(
            user_id=user_id,
            message=incoming_msg,
            ocr_results=ocr_results
        )
        
        response_text = result["response"]
        intent = result["intent"]
        
        print(f"Intent: {intent}")
        print(f"Response: {response_text[:100]}...")
        
        # Send response via Twilio
        msg = resp.message()
        msg.body(response_text)
        
        # Cleanup expired sessions periodically
        session_manager.cleanup_expired_sessions()
        
        return str(resp)
        
    except Exception as e:
        print(f"❌ Error processing message: {e}")
        import traceback
        traceback.print_exc()
        
        resp.message("I'm sorry, I encountered an error processing your message. Please try again.")
        return str(resp)


@app.route('/test', methods=['POST'])
def test_endpoint():
    """
    Test endpoint for local development without Twilio.
    Send JSON: {"user_id": "test_user", "message": "Hello"}
    """
    if not healthcare_agent:
        return jsonify({"error": "Healthcare agent not initialized"}), 500
    
    data = request.get_json()
    user_id = data.get('user_id', 'test_user')
    message = data.get('message', '')
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    try:
        result = healthcare_agent.process_user_message(
            user_id=user_id,
            message=message,
            ocr_results=None
        )
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "intent": result["intent"],
            "response": result["response"],
            "session_info": {
                "message_count": len(session_manager.get_conversation_history(user_id))
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/session/<user_id>', methods=['GET'])
def get_session(user_id):
    """Get session information for a user (for debugging)."""
    history = session_manager.get_conversation_history(user_id)
    documents = session_manager.get_recent_documents(user_id, limit=10)
    
    return jsonify({
        "user_id": user_id,
        "message_count": len(history),
        "recent_messages": history[-5:] if history else [],
        "documents": documents
    })


@app.route('/session/<user_id>', methods=['DELETE'])
def clear_session(user_id):
    """Clear session for a user."""
    session_manager.clear_session(user_id)
    return jsonify({
        "success": True,
        "message": f"Session cleared for {user_id}"
    })


# --- Error Handlers ---

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# --- Run Application ---

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🏥 AI-Driven Healthcare Chatbot")
    print("="*50)
    print(f"\nServer starting on http://127.0.0.1:5000")
    print(f"Twilio webhook URL: http://your-domain.com/webhook")
    print(f"Test endpoint: http://127.0.0.1:5000/test")
    print(f"\nActive sessions: {session_manager.get_active_session_count()}")
    print("="*50 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
