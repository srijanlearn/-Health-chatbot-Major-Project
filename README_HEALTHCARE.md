# AI-Driven Healthcare Chatbot

An intelligent healthcare assistant that integrates with Twilio for SMS/WhatsApp messaging, uses EasyOCR for medical document processing, and leverages Google Gemini LLM for context-aware responses.

## 🌟 Features

- **Multi-Channel Communication**: SMS and WhatsApp via Twilio
- **Medical Document OCR**: Extract text from prescriptions, lab reports, insurance cards
- **Insurance Policy Q&A**: RAG-powered document analysis using existing ChromaDB infrastructure
- **Intent Detection**: Automatically classify and route healthcare queries
- **Multi-Turn Conversations**: Session management with conversation history
- **Context-Aware Responses**: Healthcare-specific prompts and templates

## 🏗️ Architecture

### Tech Stack
- **Backend**: Flask (Python)
- **LLM**: Google Gemini (1.5 Pro & Flash)
- **OCR**: EasyOCR
- **Messaging**: Twilio API
- **Vector Store**: ChromaDB (from existing infrastructure)
- **Document Processing**: LangChain + UnstructuredLoader
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)

### Core Components

1. **Flask Application** (`healthcare_chatbot.py`)
   - Twilio webhook endpoints
   - Request validation and routing
   - Health monitoring

2. **Healthcare Agent** (`app/healthcare_agent.py`)
   - Intent classification (9 categories)
   - Context-aware response generation
   - Integration with RAG pipeline

3. **OCR Processor** (`app/ocr_processor.py`)
   - Medical image processing
   - Document type classification
   - Batch processing support

4. **Session Manager** (`app/session_manager.py`)
   - User conversation tracking
   - Document history
   - Context persistence

## 📋 Intent Categories

The chatbot can handle:
- **GREETING**: Welcome messages
- **INSURANCE_QUERY**: Policy coverage and benefits questions
- **PRESCRIPTION_INFO**: Medication information from OCR
- **SYMPTOM_CHECK**: Health condition queries
- **APPOINTMENT**: Scheduling assistance
- **LAB_RESULTS**: Lab report interpretation
- **GENERAL_HEALTH**: Health tips and advice
- **DOCUMENT_UPLOAD**: Document submission requests
- **UNKNOWN**: Fallback for unclear intents

## 🚀 Setup

### 1. Install Dependencies

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install all requirements
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
# Google API Key (REQUIRED)
GOOGLE_API_KEY=your_actual_google_api_key

# Twilio Credentials (REQUIRED)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890
```

**Get Your API Keys:**
- Google API Key: https://makersuite.google.com/app/apikey
- Twilio Credentials: https://console.twilio.com/

### 3. Run the Application

```bash
# Start the Flask server
python healthcare_chatbot.py
```

The server will start on `http://0.0.0.0:5000`

## 🧪 Testing

### Local Testing (Without Twilio)

```bash
# Run the test script
python test_chatbot.py
```

**Test Options:**
1. Health Check - Verify all components are running
2. Conversation Test - Multi-turn dialogue simulation
3. Intent Detection - Test intent classification
4. Session Management - Verify session persistence
5. OCR Simulation - Test document processing flow
6. Interactive Mode - Chat with the bot directly

### Testing with Twilio

1. **Set up ngrok** (for local development):
   ```bash
   ngrok http 5000
   ```

2. **Configure Twilio Webhook**:
   - Go to Twilio Console → Phone Numbers
   - Select your number
   - Set "A MESSAGE COMES IN" webhook to: `https://your-ngrok-url.ngrok.io/webhook`

3. **Send a test message** to your Twilio number

## 📱 Usage Examples

### Text-Based Queries

**User**: "Hello!"  
**Bot**: "Hello! I'm your AI Healthcare Assistant. I can help you with: • Insurance policy questions • Prescription information..."

**User**: "What does my insurance cover for maternity?"  
**Bot**: *Analyzes uploaded insurance policy and provides specific coverage details*

### Image-Based Queries (MMS)

1. Send an image of a prescription
2. Bot automatically:
   - Performs OCR extraction
   - Classifies document type
   - Provides relevant information

**User**: *[Sends prescription image]*  
**Bot**: "I can see this is a prescription. The medication listed is..."

## 🔌 API Endpoints

### `GET /`
Service information and capabilities

### `GET /health`
Health check with component status

### `POST /webhook`
Main Twilio webhook for incoming messages

### `POST /test`
Local testing endpoint (no Twilio required)
```json
{
  "user_id": "test_user",
  "message": "Hello"
}
```

### `GET /session/<user_id>`
Get session information and conversation history

### `DELETE /session/<user_id>`
Clear session data for a user

## 🗂️ Project Structure

```
turingntesla2/
├── app/
│   ├── healthcare_agent.py      # Intent detection & response generation
│   ├── ocr_processor.py          # EasyOCR medical image processing
│   ├── session_manager.py        # Conversation & session tracking
│   ├── ingestion.py              # Document processing (existing)
│   ├── parsers.py                # Utility parsers (existing)
│   └── main.py                   # Original FastAPI app (reference)
│
├── healthcare_chatbot.py         # Main Flask application
├── test_chatbot.py               # Local testing script
├── flask_test.py                 # Simple Flask test server
│
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (not in git)
├── .env.example                  # Environment template
│
├── db/                           # ChromaDB vector store
├── downloaded_files/             # Uploaded PDFs
├── downloaded_images/            # OCR images from MMS
├── sessions/                     # User session persistence
│
└── README_HEALTHCARE.md          # This file
```

## 🔄 Workflow

1. **User sends message** via SMS/WhatsApp
2. **Twilio forwards** to `/webhook` endpoint
3. **If image attached**: OCR extracts text
4. **Intent detection**: Classify user's query
5. **Context retrieval**: Get conversation history & documents
6. **Response generation**: Use appropriate LLM strategy
7. **Reply sent** back via Twilio

## 🎯 RAG Pipeline Integration

For insurance queries, the chatbot uses the existing dual-path architecture:

- **Specific Fact Path**: Full document context for precise queries
- **General Context Path**: Vector search for summaries

This leverages:
- Parent-child document retrieval
- ChromaDB vector store
- HuggingFace embeddings
- Gemini 1.5 Pro for generation

## 🛠️ Extending the Bot

### Adding New Intents

Edit `app/healthcare_agent.py`:

```python
INTENTS = {
    "YOUR_NEW_INTENT": "your_new_intent",
    # ...
}
```

Add handler method:
```python
def _handle_your_new_intent(self, message: str, context: str) -> str:
    # Your logic here
    pass
```

### Customizing OCR Classification

Edit `app/ocr_processor.py`:

```python
def _classify_medical_document(self, text: str) -> str:
    keywords = {
        "your_doc_type": ["keyword1", "keyword2"],
        # ...
    }
```

## 🔐 Security Notes

- Never commit `.env` file
- Use Twilio signature validation in production
- Sanitize user inputs
- Implement rate limiting for production
- Use HTTPS for webhook URLs

## 📊 Monitoring

The chatbot provides:
- Active session count
- Component health status
- API configuration status
- Per-user conversation history

Access via `/health` endpoint.

## 🐛 Troubleshooting

### Bot not responding
- Check GOOGLE_API_KEY is set correctly
- Verify Twilio credentials
- Check webhook URL configuration

### OCR not working
- Ensure image format is supported (jpg, png)
- Check Twilio authentication credentials
- Verify EasyOCR model downloaded successfully

### Intent detection issues
- Check conversation history for context
- Review intent classification prompts
- Test with clear, specific messages

## 📚 Resources

- [Twilio SMS Documentation](https://www.twilio.com/docs/sms)
- [Google Gemini API](https://ai.google.dev/)
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [LangChain Docs](https://python.langchain.com/)

## 📄 License

MIT License

## 🙏 Acknowledgments

Built upon the HackRx 6.0 Document Q&A System, enhanced with healthcare-specific features and multi-channel messaging capabilities.

---

**For the original document Q&A system, see the FastAPI implementation in `app/main.py` and `frontend.py`**
