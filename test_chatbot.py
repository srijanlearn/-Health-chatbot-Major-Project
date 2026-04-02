#!/usr/bin/env python
# test_chatbot.py

"""
Local testing script for Healthcare Chatbot.
Tests the chatbot without requiring Twilio integration.
"""

import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:5000"
TEST_USER_ID = "test_user_001"


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_health_check():
    """Test the health check endpoint."""
    print_section("Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


def send_message(user_id, message):
    """Send a test message to the chatbot."""
    payload = {
        "user_id": user_id,
        "message": message
    }
    response = requests.post(f"{BASE_URL}/test", json=payload)
    return response.json()


def test_conversation():
    """Test a multi-turn conversation."""
    print_section("Conversation Test")
    
    messages = [
        "Hello!",
        "What can you help me with?",
        "I have questions about my insurance policy",
        "What are lab results?",
        "Thank you!"
    ]
    
    for i, msg in enumerate(messages, 1):
        print(f"\n[Turn {i}]")
        print(f"User: {msg}")
        
        result = send_message(TEST_USER_ID, msg)
        
        if result.get("success"):
            print(f"Intent: {result['intent']}")
            print(f"Assistant: {result['response']}")
        else:
            print(f"Error: {result.get('error')}")
            break


def test_intent_detection():
    """Test various intent classifications."""
    print_section("Intent Detection Test")
    
    test_cases = [
        "Hi there!",
        "What does my insurance policy cover for maternity?",
        "Can you explain my prescription?",
        "I have a headache and fever",
        "I want to schedule an appointment",
        "What do my lab results mean?",
        "I want to upload a document"
    ]
    
    for msg in test_cases:
        result = send_message(f"intent_test_{hash(msg) % 1000}", msg)
        if result.get("success"):
            print(f"\nMessage: {msg}")
            print(f"Intent: {result['intent']}")
        else:
            print(f"\nError for '{msg}': {result.get('error')}")


def get_session_info(user_id):
    """Get session information."""
    response = requests.get(f"{BASE_URL}/session/{user_id}")
    return response.json()


def clear_session(user_id):
    """Clear session for user."""
    response = requests.delete(f"{BASE_URL}/session/{user_id}")
    return response.json()


def test_session_management():
    """Test session management."""
    print_section("Session Management Test")
    
    # Send some messages
    print("\nSending messages...")
    for i in range(3):
        send_message(TEST_USER_ID, f"Test message {i+1}")
    
    # Get session info
    print("\nSession Info:")
    session = get_session_info(TEST_USER_ID)
    print(json.dumps(session, indent=2))
    
    # Clear session
    print("\nClearing session...")
    result = clear_session(TEST_USER_ID)
    print(json.dumps(result, indent=2))
    
    # Verify cleared
    print("\nSession Info After Clear:")
    session = get_session_info(TEST_USER_ID)
    print(json.dumps(session, indent=2))


def test_ocr_simulation():
    """Simulate OCR functionality with text."""
    print_section("OCR Simulation Test")
    
    print("\nNote: To test actual OCR, you need to:")
    print("1. Have a medical image (prescription, lab report, etc.)")
    print("2. Use the actual Twilio webhook or upload via MMS")
    print("\nFor now, testing with simulated OCR text...")
    
    # This would normally come from OCR, but we're simulating
    msg = "I have a prescription image. What medications are listed?"
    result = send_message("ocr_test_user", msg)
    
    if result.get("success"):
        print(f"\nUser: {msg}")
        print(f"Intent: {result['intent']}")
        print(f"Assistant: {result['response']}")


def interactive_mode():
    """Interactive chat mode."""
    print_section("Interactive Mode")
    print("\nType 'quit' to exit, 'clear' to clear session\n")
    
    user_id = "interactive_user"
    
    while True:
        try:
            message = input("\nYou: ").strip()
            
            if message.lower() == 'quit':
                print("Goodbye!")
                break
            
            if message.lower() == 'clear':
                clear_session(user_id)
                print("Session cleared!")
                continue
            
            if not message:
                continue
            
            result = send_message(user_id, message)
            
            if result.get("success"):
                print(f"Assistant: {result['response']}")
            else:
                print(f"Error: {result.get('error')}")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main test runner."""
    print("\n" + "="*60)
    print("  Healthcare Chatbot - Test Suite")
    print("="*60)
    
    # Check if server is running
    try:
        response = requests.get(BASE_URL, timeout=2)
        print("✅ Server is running")
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running!")
        print(f"\nPlease start the server first:")
        print("  python healthcare_chatbot.py")
        return
    
    # Check Google API Key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key or google_api_key == "your_api_key_here":
        print("\n⚠️  WARNING: GOOGLE_API_KEY not configured")
        print("The chatbot will not work without a valid Google API key.")
        print("Please set it in your .env file.\n")
    
    # Menu
    while True:
        print("\n" + "-"*60)
        print("Test Options:")
        print("-"*60)
        print("1. Health Check")
        print("2. Simple Conversation Test")
        print("3. Intent Detection Test")
        print("4. Session Management Test")
        print("5. OCR Simulation")
        print("6. Interactive Mode")
        print("0. Exit")
        print("-"*60)
        
        choice = input("\nSelect option (0-6): ").strip()
        
        if choice == '0':
            print("\nGoodbye!")
            break
        elif choice == '1':
            test_health_check()
        elif choice == '2':
            test_conversation()
        elif choice == '3':
            test_intent_detection()
        elif choice == '4':
            test_session_management()
        elif choice == '5':
            test_ocr_simulation()
        elif choice == '6':
            interactive_mode()
        else:
            print("Invalid option!")


if __name__ == '__main__':
    main()
