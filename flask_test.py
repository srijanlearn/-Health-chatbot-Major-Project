"""
Simple Flask server to test Twilio webhook integration
Run with: python flask_test.py
"""
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

@app.route('/')
def hello():
    return "Flask + Twilio Healthcare Chatbot - Server is Running! ✓"

@app.route('/webhook', methods=['POST'])
def webhook():
    """Twilio webhook endpoint for incoming messages"""
    incoming_msg = request.values.get('Body', '').strip()
    resp = MessagingResponse()
    
    # Simple echo response for testing
    msg = resp.message()
    msg.body(f"Echo: {incoming_msg}")
    
    return str(resp)

@app.route('/health', methods=['GET'])
def health():
    return {"status": "healthy", "service": "Flask + Twilio", "ocr": "EasyOCR ready"}

if __name__ == '__main__':
    print("Starting Flask test server...")
    print("Test endpoints:")
    print("  - http://localhost:5000/")
    print("  - http://localhost:5000/health")
    print("  - http://localhost:5000/webhook (POST)")
    app.run(debug=True, port=5000)
