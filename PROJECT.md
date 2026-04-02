# PROJECT.md — HealthyPartner v2 Complete Project Context

> **This file is the single source of truth for the project.**
> It captures architecture decisions, what's been done, what's pending, ideas, and context
> so any developer (or AI) can pick up where work left off.

---

## 🔖 Project Identity

| Field | Value |
|-------|-------|
| **Name** | HealthyPartner |
| **Version** | 2.0.0 (in development) |
| **Tagline** | Privacy-first local healthcare AI for India |
| **Repo** | https://github.com/srijanlearn/-Health-chatbot-Major-Project |
| **License** | MIT |
| **Target market** | Indian hospitals, clinics, pharmacies, practitioners (B2B + D2C) |

---

## 🧭 Vision

Build the **#1 locally-running healthcare AI platform** for the Indian market. Any hospital, medical shop, or local practitioner installs it on their own PC — their data never leaves their device, it costs near-zero to operate, and it works offline. Future: mobile app that runs entirely on-device.

### Key Principles
1. **Privacy by default** — All inference local, no cloud telemetry, DPDPA-ready
2. **Works offline** — Full functionality without internet after initial setup
3. **Low-power hardware** — 8 GB RAM laptop is the target minimum
4. **Indian-first** — Hindi/regional languages, IRDAI regulations, PMJAY, Jan Aushadhi generics
5. **Product, not project** — One-click installer, admin panel, WhatsApp integration

---

## 🏗️ Architecture Decisions (ADR Log)

### ADR-001: Qwen3-4B over Phi-4-mini as main LLM
- **Decision**: Use Qwen3-4B Q4 (~3 GB RAM) as the primary generation model
- **Rationale**: 
  - #1 ranked fine-tunable model across 8 tasks
  - Native support for 200+ languages including Hindi, Tamil, Telugu, Bengali
  - Dual reasoning mode (think/no-think) matches our router pattern
  - Apache 2.0 license (commercial OK)
- **Alternatives considered**: 
  - Phi-4-mini (better raw reasoning but English-heavy — no Hindi)
  - BitNet 2B4T (lighter but lower quality — kept as optional ultra-light tier)
  - LLaMA 3.2 3B (good but Qwen3-4B outperforms in fine-tuned benchmarks)

### ADR-002: Dual-model architecture (fast + quality)
- **Decision**: Qwen3.5-0.6B for classification/routing, Qwen3-4B for generation
- **Rationale**: Same architecture family enables speculative decoding (2-3x speedup). Fast model handles ~60% of pipeline work at 100+ tok/s, quality model only runs for final answer generation.

### ADR-003: 3-stage RAG over naive retrieve+generate
- **Decision**: Hybrid BM25+Vector retrieve → Cross-encoder rerank → LLM generate
- **Rationale**: 
  - BM25 handles exact-match queries ("waiting period for cataracts") that vector search misses
  - Cross-encoder reranking filters 20 noisy chunks to 5 precise ones
  - Small models perform dramatically better with 5 focused chunks vs 20 noisy ones
  - This is enterprise-standard RAG (used by Cohere, LlamaIndex production deployments)

### ADR-004: Knowledge graph for structured Indian healthcare data
- **Decision**: SQLite-backed knowledge graph for IRDAI, PMJAY, drugs, ICD-10
- **Rationale**: 
  - 70% of common queries are better answered by structured lookup (instant, 100% accurate)
  - LLM only handles the 30% that need reasoning
  - This is the core competitive moat — any competitor can download Qwen3, but not the curated data

### ADR-005: multilingual-e5-small over all-MiniLM-L6-v2
- **Decision**: Switch embeddings to multilingual-e5-small (118 MB, 100+ languages)
- **Rationale**: Indian insurance documents mix Hindi and English. English-only embeddings miss half the content.

### ADR-006: Ollama as primary inference server
- **Decision**: Use Ollama for model serving with llama-cpp-python as fallback
- **Rationale**: Best developer experience (167K GitHub stars, `ollama pull` workflow), automatic GPU detection, OpenAI-compatible API, Docker-like UX for model management. Already partially integrated in v1 (`healthcare_agent.py` uses `ChatOllama`).

---

## 📦 Current State — What Exists (v1 → v2 migration)

### Core Files (KEEP — will be modified)

| File | Purpose | Migration Status |
|------|---------|-----------------|
| `app/main.py` | FastAPI RAG endpoint | ⬜ Needs Gemini→local swap |
| `app/healthcare_agent.py` | Intent detection + response gen | ⬜ Needs LLMEngine integration |
| `app/ingestion.py` | PDF → ChromaDB vectors | ⬜ Needs hybrid search upgrade |
| `app/ocr_processor.py` | EasyOCR medical images | ✅ Already local — no changes |
| `app/session_manager.py` | Conversation tracking | ⬜ Minor: SQLite persistence upgrade |
| `app/parsers.py` | Utility parsers | ✅ No changes needed |
| `healthcare_chatbot.py` | Flask + Twilio webhook | ⬜ Needs LLMEngine integration |
| `healthypartner_backend.py` | HealthyPartner API | ⬜ Needs full Gemini removal |
| `frontend.py` | Streamlit web UI | ⬜ Needs local-first UI updates |

### Files Removed (College project cleanup)

| File | Why Removed |
|------|-------------|
| `B25 Phase 1 PPT.pdf` | College presentation — not production |
| `Major project report B25.docx/pdf` | College report — not production |
| `CLAUDE.md` | Old AI assistant notes — replaced by PROJECT.md |
| `WARP.md` | Old AI assistant notes — replaced by PROJECT.md |
| `dummy_policy.md` | Demo data — will use real test fixtures |
| `constraints.txt` | Outdated version pins (chromadb 0.3.29, etc.) |
| `flask_test.py` | Echo-only test server — superseded by test suite |
| `demo_rag.py` | Demo script — not production |
| `test_chatbot.py` | Old test file — will build proper test suite |
| `build_knowledge_base.py` | Old ingestion script — superseded by new pipeline |
| `README_HEALTHCARE.md` | Duplicate README — consolidated into README.md |

### Files To Create (v2 new components)

| File/Directory | Purpose | Phase |
|---------------|---------|-------|
| `app/llm_engine.py` | Unified local LLM engine | Phase 1 |
| `app/orchestrator.py` | Multi-step reasoning pipeline | Phase 2 |
| `app/prompts/` | Optimized prompt library | Phase 2 |
| `app/knowledge/` | Indian healthcare knowledge graph | Phase 3 |
| `installer/` | One-click cross-platform installer | Phase 4 |
| `app/admin_panel.py` | B2B admin dashboard | Phase 5 |
| `training/` | Fine-tuning pipeline (Unsloth+QLoRA) | Phase 6 |
| `whatsapp/bridge.py` | Local WhatsApp bridge | Phase 7 |

---

## 🗺️ Execution Phases

| # | Phase | Priority | Est. Time | Status |
|---|-------|----------|-----------|--------|
| 1 | **Local LLM Engine** — `llm_engine.py`, Gemini removal, requirements | 🔴 Critical | 2-3 days | ✅ Completed |
| 2 | **Prompt Engineering + Orchestration** — prompts/, orchestrator | 🔴 Critical | 2-3 days | ✅ Completed |
| 3 | **Knowledge Graph + Hybrid Search** — knowledge/, ingestion upgrade | 🟡 High | 2-3 days | ✅ Completed |
| 4 | **One-Click Installer** — installer/, Dockerfile | 🟡 High | 2 days | ⬜ Not started |
| 5 | **Frontend** — local-first UI, admin panel | 🟡 High | 1-2 days | ✅ Completed |
| 6 | **Fine-Tuning Pipeline** — training/ (Unsloth+QLoRA) | 🟠 Medium | 3-5 days | ⬜ Not started |
| 7 | **WhatsApp Local Bridge** — whatsapp/ | 🟢 Future | 2-3 days | ⬜ Not started |

---

## 💡 Ideas & Backlog

### Near-term
- [ ] Speculative decoding: Qwen3.5-0.6B as draft model for Qwen3-4B (2-3x speedup, same family)
- [ ] Response streaming in Streamlit UI (token-by-token display)
- [ ] Session persistence to SQLite (survive server restarts)
- [ ] Multi-document comparison (compare 2 insurance policies side-by-side)
- [ ] PDF preview thumbnails in UI

### Medium-term
- [ ] React Native mobile app using Cactus SDK or MLC LLM
- [ ] Voice input/output (speech-to-text + TTS for accessibility)
- [ ] Multi-tenant B2B SaaS mode (isolated data per organization)
- [ ] Automated insurance claim form filling
- [ ] Integration with hospital management systems (HIS/HMS APIs)

### Long-term
- [ ] On-device model fine-tuning (personalize to each hospital's documents)
- [ ] Federated learning across hospitals (improve model without sharing data)
- [ ] ABDM (Ayushman Bharat Digital Mission) integration
- [ ] Digital Health ID (ABHA) integration
- [ ] Medical image analysis beyond OCR (X-ray, dermatology using vision models)

---

## 💰 Business Model Ideas

| Tier | Target | Price | Features |
|------|--------|-------|----------|
| **Free** | Individual practitioners | ₹0 | Single user, English, basic intents |
| **Pro** | Clinics / pharmacies | ₹999/month | Multi-user, Hindi+English, WhatsApp, admin panel |
| **Enterprise** | Hospitals / chains | ₹4,999/month | Custom fine-tuning, API access, priority support, audit logs, all languages |

### Revenue levers
- Installation & setup service (one-time ₹5,000-10,000)
- Custom model fine-tuning service
- Hospital-specific knowledge base curation
- Premium support SLA

---

## ⚠️ Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| WhatsApp bans general AI bots (Jan 2026 policy) | Can't use WhatsApp | Already compliant — task-specific healthcare assistant, not general chatbot |
| Small model hallucination on medical topics | Patient safety | Knowledge graph for factual queries + mandatory disclaimers + confidence scoring |
| Indian language quality not good enough | Poor UX in Hindi | Qwen3 has native Hindi training; fine-tune on Indian medical corpus |
| Qwen3-4B doesn't fit on target hardware | Can't deploy | Tiered models: fall back to BitNet 2B4T for ultra-light, or Qwen3.5-0.6B only |
| DPDPA (India's data protection act) compliance | Legal risk | 100% local by design — no data leaves device. Add audit logging. |

---

## 📝 Changelog

### 2026-04-02 — Project Reset
- Cleaned all college project filler files (PPTs, reports, demo scripts)
- Rewrote README.md as professional product documentation
- Created PROJECT.md as single source of truth
- Updated .gitignore for v2
- Finalized architecture: Qwen3-4B + Qwen3.5-0.6B, 3-stage RAG, knowledge graph
- Created implementation plan (7 phases)

---

### 2026-04-03 — Orchestrator wired in + Frontend rewrite

- **Fixed critical gap**: `app/main.py` was bypassing the Orchestrator entirely (doing its own routing). Now both `/chat` and `/healthypartner/run` run the full 7-step pipeline.
- **Added `/chat` endpoint**: Free-form healthcare chat without a document — intent classify → KG lookup → LLM generate.
- **`KnowledgeGraph` initialised at startup** and passed to Orchestrator (Phase 3 data now live).
- **`frontend.py` complete rewrite** (962 lines → 327 lines):
  - `st.chat_message` chat interface replacing batch Q&A tabs
  - Sidebar: live Ollama status (🟢/🔴), model names, RAM/GPU/arch, PDF uploader
  - Per-message route badges: 🗂 Knowledge Graph / 📄 Document Fact / 🔍 RAG / 🤖 LLM / 🚨 Emergency
  - Removed all Gemini/GPT4 model references
  - API base fixed from `localhost:5000` → `localhost:8000`

*Last updated: 2026-04-03*
