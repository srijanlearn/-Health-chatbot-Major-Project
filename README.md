# HealthyPartner v2

**Privacy-first, locally-running healthcare AI platform for the Indian market.**

Zero cloud dependency. Zero API cost. 100% on-device inference. Runs on an 8 GB laptop.

---

## What It Does

HealthyPartner is an AI-powered healthcare assistant that runs **entirely on your local device** — no data ever leaves your machine. It handles:

- 🏥 **Insurance Policy Q&A** — Upload any health insurance PDF, ask questions in Hindi or English
- 💊 **Prescription Analysis** — Photograph a prescription, get medication info and drug interaction warnings
- 🔬 **Lab Report Interpretation** — Upload lab results, get plain-language explanations
- 🩺 **Symptom Guidance** — Describe symptoms, get ICD-10 mapped guidance (always with "consult a doctor")
- 📋 **Government Schemes** — Built-in knowledge of Ayushman Bharat (PMJAY), IRDAI regulations

Accessible via **Web UI**, **WhatsApp**, or **REST API**.

---

## Architecture

```
User (WhatsApp / Web UI / API)
        │
        ▼
┌─ Orchestration Layer ──────────────────┐
│  Intent Classifier (Qwen3.5-0.6B)     │
│  Knowledge Router (rules + LLM)       │
│  Confidence Scorer                     │
└──────────┬──────────────┬──────────────┘
           │              │
    ┌──────▼──┐    ┌──────▼──────────────┐
    │Knowledge│    │  3-Stage RAG        │
    │Graph    │    │  1. Hybrid Retrieve  │
    │(instant)│    │  2. Cross-Encoder    │
    │         │    │  3. Qwen3-4B Gen    │
    └─────────┘    └─────────────────────┘
           │              │
           ▼              ▼
    ┌────────────────────────────┐
    │  Safety & Compliance       │
    │  Medical disclaimers       │
    │  Drug interaction warnings │
    └────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Main LLM** | Qwen3-4B Q4 (~3 GB RAM) via Ollama |
| **Fast LLM** | Qwen3.5-0.6B Q4 (~0.5 GB RAM) via Ollama |
| **Embeddings** | multilingual-e5-small (Hindi + English) |
| **Vector Store** | ChromaDB |
| **Keyword Search** | BM25 (rank-bm25) |
| **Reranker** | Cross-encoder (ms-marco-MiniLM-L-6-v2) |
| **OCR** | EasyOCR |
| **API** | FastAPI + Flask |
| **Frontend** | Streamlit |
| **WhatsApp** | Twilio (production) / Baileys (local) |

## Project Structure

```
healthypartner/
├── app/
│   ├── main.py               # FastAPI — RAG endpoint
│   ├── llm_engine.py         # Unified local LLM engine (Ollama)
│   ├── orchestrator.py       # Multi-step reasoning pipeline
│   ├── healthcare_agent.py   # Intent detection & response generation
│   ├── ingestion.py          # PDF → ChromaDB + BM25 hybrid search
│   ├── ocr_processor.py      # EasyOCR medical image processing
│   ├── session_manager.py    # Conversation & session tracking
│   ├── parsers.py            # Utility parsers
│   ├── prompts/              # Optimized prompt library
│   │   ├── intent_classifier.py
│   │   ├── insurance_qa.py
│   │   ├── medical_safety.py
│   │   └── router.py
│   └── knowledge/            # Indian healthcare knowledge graph
│       ├── graph.py
│       └── data/             # IRDAI, PMJAY, drugs, ICD-10
│
├── installer/                # One-click deployment
│   ├── setup.sh              # macOS/Linux installer
│   ├── setup.bat             # Windows installer
│   └── config.yaml           # Default configuration
│
├── training/                 # Fine-tuning pipeline
│   ├── prepare_data.py
│   ├── finetune.py           # Unsloth + QLoRA
│   ├── eval.py
│   └── datasets/             # Training data
│
├── whatsapp/                 # WhatsApp local bridge
│   └── bridge.py
│
├── healthcare_chatbot.py     # Flask — Twilio webhook server
├── healthypartner_backend.py # HealthyPartner backend API
├── frontend.py               # Streamlit web UI
├── requirements.txt
├── Dockerfile
├── .env.example
└── PROJECT.md                # Full project context & memory
```

## Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed
- 8 GB RAM minimum

### Install & Run

```bash
# 1. Clone
git clone https://github.com/srijanlearn/-Health-chatbot-Major-Project.git
cd -Health-chatbot-Major-Project

# 2. Setup
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Pull models
ollama pull qwen3:4b
ollama pull qwen3.5:0.6b

# 4. Run
python healthypartner_backend.py     # API on http://localhost:5000
streamlit run frontend.py            # UI on http://localhost:8501
```

No API keys needed. No cloud account. Just run.

## License

MIT
