# 🏥 AI-Driven Healthcare Chatbot & Document Q&A System

An intelligent, multi-purpose healthcare assistant combining **RAG-powered insurance document Q&A** with a **Twilio-integrated chatbot** for SMS/WhatsApp communication.

---

## 🌟 Features

| Feature | Description |
|---|---|
| 📄 **Insurance Policy Q&A** | RAG pipeline with dual-path routing (vector search + full-context) |
| 💬 **Multi-Channel Messaging** | SMS & WhatsApp via Twilio |
| 🔍 **Medical Document OCR** | Extract text from prescriptions, lab reports, insurance cards |
| 🧠 **Intent Detection** | 9-category auto-classification for routing health queries |
| 🗂️ **Session Management** | Multi-turn conversation history per user |
| 🖥️ **Streamlit Frontend** | Interactive web UI for document Q&A |

---

## 🏗️ Architecture

```
turingntesla2/
├── app/
│   ├── main.py                  # FastAPI RAG endpoint (/hackrx/run)
│   ├── ingestion.py             # PDF ingestion → ChromaDB vector store
│   ├── healthcare_agent.py      # Intent detection & response generation
│   ├── ocr_processor.py         # EasyOCR medical image processing
│   ├── session_manager.py       # Conversation & session tracking
│   └── parsers.py               # Utility parsers
│
├── healthcare_chatbot.py        # Flask app — Twilio webhook server
├── healthypartner_backend.py    # Healthy Partner backend API
├── frontend.py                  # Streamlit web UI
├── build_knowledge_base.py      # Pre-build ChromaDB from local PDFs
├── demo_rag.py                  # Quick RAG demo script
├── test_chatbot.py              # Local chatbot testing (no Twilio needed)
├── flask_test.py                # Minimal Flask test server
│
├── Dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
└── constraints.txt              # Dependency version pins
```

### Tech Stack

- **LLM**: Google Gemini 1.5 Pro & Flash via LangChain
- **Vector Store**: ChromaDB + HuggingFace `all-MiniLM-L6-v2` embeddings
- **Document Loading**: LangChain + UnstructuredLoader
- **OCR**: EasyOCR
- **Messaging**: Twilio API (SMS/WhatsApp)
- **Web UI**: Streamlit
- **APIs**: FastAPI + Flask

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Srijan1419/turingntesla2.git
cd turingntesla2

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and fill in your credentials
```

Required keys in `.env`:

```env
GOOGLE_API_KEY=your_google_api_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=+1234567890
```

- **Google API Key**: https://makersuite.google.com/app/apikey
- **Twilio Credentials**: https://console.twilio.com/

### 3. Run

**RAG API (FastAPI):**
```bash
uvicorn app.main:app --reload
# Endpoint: POST http://localhost:8000/hackrx/run
```

**Healthcare Chatbot (Flask/Twilio):**
```bash
python healthcare_chatbot.py
# Webhook: POST http://localhost:5000/webhook
```

**Streamlit UI:**
```bash
streamlit run frontend.py
```

---

## 🔌 API Reference

### RAG Endpoint — `POST /hackrx/run`

```json
{
  "documents": "https://url-to-pdf.com/policy.pdf",
  "questions": ["What is the waiting period for cataracts?"],
  "is_base64": false
}
```

**Response:**
```json
{
  "answers": ["The waiting period for cataracts is 2 years."]
}
```

### Chatbot Endpoints (Flask)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `POST` | `/webhook` | Twilio incoming message handler |
| `POST` | `/test` | Local test (no Twilio) |
| `GET` | `/session/<user_id>` | View session history |
| `DELETE` | `/session/<user_id>` | Clear session |

**Local test:**
```bash
curl -X POST http://localhost:5000/test \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "message": "What does my policy cover?"}'
```

---

## 🧪 Testing

```bash
# Test chatbot locally (no Twilio required)
python test_chatbot.py

# Quick RAG demo
python demo_rag.py
```

---

## 🐳 Docker

```bash
docker build -t turingntesla2 .
docker run -p 8000:8000 --env-file .env turingntesla2
```

---

## 📋 Intent Categories

The chatbot auto-classifies queries into:
`GREETING` · `INSURANCE_QUERY` · `PRESCRIPTION_INFO` · `SYMPTOM_CHECK` · `APPOINTMENT` · `LAB_RESULTS` · `GENERAL_HEALTH` · `DOCUMENT_UPLOAD` · `UNKNOWN`

---

## 🔐 Security

- `.env` is **never committed** — use `.env.example` as template
- Twilio signature validation on all webhook requests
- Sanitize all user inputs before LLM calls

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
