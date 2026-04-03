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

## 🧠 How We Work (Development Philosophy)

> This is the agreed working method between Aman and Claude for this project.

### The Loop
```
Plan → Implement → Test → Stress Test → Optimise → back to Plan
```

1. **Plan first** — Before any implementation, create a comprehensive plan of what's missing, what needs to be built, and in what order (prioritised by weight/impact).
2. **Implement** — Build exactly what the plan says. No scope creep, no speculative features.
3. **Test** — Unit tests, integration tests, golden eval harness. Nothing ships untested.
4. **Stress test** — Concurrent load, memory leak detection, edge cases. Production-level validation.
5. **Optimise** — Come back to the scratchpad, identify bottlenecks, tighten what's already there.
6. **Repeat** — Start the loop again with a cleaner, tighter baseline.

### Standards
- **Industry best practices always** — if there's a standard way to do something, we do it that way.
- **No shortcuts that become debt** — a fast hack today is a rewrite next month.
- **One backend, one truth** — no duplicate code paths, no parallel implementations.
- **Everything tested before it ships** — code that isn't tested doesn't exist.

### Target Product
- Locally-running, privacy-first healthcare AI for the Indian market
- Runs on customer's own hardware (hospital, clinic, pharmacy)
- Customer configures their own knowledge base (formulary, schemes, policies)
- Aman provides setup assistance as a service — that's the revenue model
- Must be the best possible software, not just a proof of concept

---

## 🗺️ Execution Phases

| # | Phase | Priority | Est. Time | Status |
|---|-------|----------|-----------|--------|
| 1 | **Local LLM Engine** — `llm_engine.py`, Gemini removal, requirements | 🔴 Critical | 2-3 days | ✅ Completed |
| 2 | **Prompt Engineering + Orchestration** — prompts/, orchestrator | 🔴 Critical | 2-3 days | ✅ Completed |
| 3 | **Knowledge Graph + Hybrid Search** — knowledge/, ingestion upgrade | 🟡 High | 2-3 days | ✅ Completed |
| 4 | **One-Click Installer** — installer/, Dockerfile | 🟡 High | 2 days | ✅ Completed |
| 5 | **Frontend** — local-first UI, admin panel | 🟡 High | 1-2 days | ✅ Completed |
| 6 | **Fine-Tuning Pipeline** — training/ (Unsloth+QLoRA) | 🟠 Medium | 3-5 days | ✅ Completed |
| 7 | **WhatsApp Local Bridge** — whatsapp/ | 🟢 Future | 2-3 days | ⬜ Not started |

---

## 🚧 Comprehensive Gap Analysis & Build Plan

> Created 2026-04-03. This is the master backlog — ordered by priority. Each item must be planned → implemented → tested → stress-tested before moving to the next.

### GAP-001 — Kill the Flask backend (BLOCKER)
- **Problem**: `healthypartner_backend.py` (Flask) and `app/main.py` (FastAPI) both expose `/chat` and `/healthypartner/run` with diverging logic. Flask `/chat` bypasses the Orchestrator — no emergency detection, no KG lookup, hardcoded metadata.
- **Action**: Delete `healthypartner_backend.py`. Migrate any unique endpoints (e.g. `/webhook` for Twilio) into `app/main.py`.
- **Test**: All existing endpoints respond identically from FastAPI. No Flask process running.
- **Status**: ✅ Done (2026-04-04)

### GAP-002 — Multi-tenancy architecture (CORE BUSINESS MODEL)
- **Problem**: All customers share one global KnowledgeGraph and one vector store. A hospital in Delhi and a pharmacy in Mumbai cannot have isolated KBs.
- **Action**: Design a `tenant_id` system. Each tenant gets: isolated SQLite KG, isolated ChromaDB collection, isolated BM25 index, isolated config (`tenant.yaml`). Orchestrator is initialized per-tenant at request time (or cached per tenant).
- **Test**: Two tenants with different KBs, queries to each return tenant-specific data with zero bleed.
- **Stress test**: 10 tenants, 50 concurrent requests, no cross-contamination.
- **Status**: ✅ Done (2026-04-04)

### GAP-003 — KB Admin Interface (THE PRODUCT)
- **Problem**: Loading a customer's knowledge base requires manually editing JSON files. Not shippable.
- **Action**: Build an admin panel (FastAPI + simple HTML, or extend `frontend_web/`) where the customer (or Aman during setup) can: upload their drug formulary CSV, add custom schemes/facts, import ICD-10 subset, trigger KB rebuild, and see KB stats (record counts per domain).
- **Test**: Upload a CSV → verify rows appear in KG queries.
- **Status**: ✅ Done (2026-04-04)

### GAP-004 — Full test suite
- **Problem**: Essentially zero automated tests. Can't call it shippable software without them.
- **Action**:
  - `tests/unit/` — KG query functions, intent parser, cache eviction, RRF merge
  - `tests/integration/` — Orchestrator end-to-end with mock LLM (no Ollama dependency)
  - `tests/eval/` — golden Q&A harness (started: `eval_knowledge_graph.py`)
  - `tests/load/` — locust load test (50 concurrent users, 5-minute sustained)
- **Test**: `pytest` passes clean. Locust P95 latency < 3s under 50 users.
- **Status**: ⬜ Not started (eval harness skeleton exists)

### GAP-005 — Audit logging
- **Problem**: No record of what was asked and what was answered. In healthcare this is a legal and safety requirement.
- **Action**: Log every query to SQLite: `tenant_id`, `session_id`, `message_hash`, `intent`, `route`, `latency_ms`, `timestamp`. No PII in the log — only hash. Admin panel shows aggregate stats.
- **Test**: 100 queries logged, query the log via admin panel.
- **Status**: ✅ Done (2026-04-04)
  - `app/audit.py` — thread-safe `AuditLog`, SHA-256 message hash (first 16 chars), never raises
  - Wired into all `Orchestrator.process()` return paths (emergency, cache, KG hit, LLM)
  - `TenantManager` creates one shared `AuditLog` at `data/audit.db`, passes it to all orchestrators
  - Admin endpoints: `GET /admin/audit/stats`, `GET /admin/audit/recent?limit=N`
  - 17 unit tests + 8 integration tests — 148 total, all green

### GAP-006 — Confidence scoring + graceful degradation
- **Problem**: Low-confidence LLM responses are returned with the same authority as KG-backed facts.
- **Action**: Add a `confidence` field to `PipelineResult`. KG hits = 1.0, direct LLM = 0.5–0.7 based on response length/hedging heuristics. If confidence < threshold, append a stronger "please consult a doctor" warning and flag in the UI.
- **Test**: Nonsense query returns low confidence + escalated disclaimer.
- **Status**: ✅ Done (2026-04-04)
  - `_score_llm_confidence(response)` — heuristic scorer (base 0.65, hedge/length penalties, numeric boost, clamped 0.30–0.85)
  - Route-based assignment: emergency/KG/static = 1.0; LLM = heuristic score
  - `LOW_CONFIDENCE_THRESHOLD = 0.60` — below this, escalated disclaimer appended
  - `_ResponseCache.get()` now returns `(text, confidence)` tuple; confidence preserved across cache hits
  - `confidence` field in `PipelineResult.to_dict()` and `ChatResponse` API model
  - 19 new tests (scorer, disclaimers, pipeline integration) — 167 total, all green

### GAP-007 — Stress testing + memory profiling
- **Problem**: No performance baseline. Unknown behaviour under sustained load on target hardware (8GB RAM laptop).
- **Action**: Run locust against a real deployment on M4 Air (16GB). Profile memory over 30-minute run — detect leaks in `_retriever_cache` and `_ResponseCache`. Document P50/P95/P99 latency per route.
- **Test**: No OOM in 30-minute run. P95 < 3s for KG/cache routes, < 8s for LLM routes.
- **Status**: ⬜ Not started

### GAP-008 — Session persistence
- **Problem**: Server restart wipes all conversation history (stored in-memory). Also `_retriever_cache` and `_ResponseCache` are lost.
- **Action**: Persist sessions to SQLite. Warm the response cache from SQLite on startup (top-N most recent entries).
- **Test**: Restart server mid-conversation, verify history survives.
- **Status**: ⬜ Not started (in backlog)

### GAP-009 — Installer hardening
- **Problem**: `installer/setup.sh` exists but is not stress-tested. Edge cases: no internet, Ollama already running on different port, Python 3.11 vs 3.12 differences, Windows path spaces.
- **Action**: Test on a clean VM (macOS, Ubuntu, Windows). Add pre-flight checks (port availability, disk space, Python version). Add an `--update` flag for existing installations.
- **Status**: ⬜ Not started

### GAP-010 — WhatsApp bridge
- **Problem**: Phase 7 deferred. High value for Indian market (WhatsApp is the primary communication channel).
- **Action**: Twilio Cloud API webhook → FastAPI `/webhook` → Orchestrator. Not Baileys (policy risk).
- **Status**: ⬜ Deferred (tackle after GAP-001 through GAP-005 are done)

---

## 📊 Functionality Weight Matrix

Evaluated 2026-04-03. Weight = Impact × Frequency × Risk (1–10 scale).
Higher weight = prioritise first. Re-evaluate when a new phase ships.

| # | Feature | Status | Impact | Frequency | Risk if Absent | **Weight** |
|---|---------|--------|--------|-----------|----------------|-----------|
| 1 | Emergency Detection | ✅ Complete | Life-safety | Low | Life-threatening | **10** |
| 2 | Safety Guardrails (disclaimers) | ✅ Basic | Legal/medical | High | Legal liability | **9** |
| 3 | Knowledge Graph Lookup | ✅ Improved | No-LLM answers | High | High LLM cost | **8** |
| 4 | Hybrid RAG (BM25 + Vector + Rerank) | ✅ Complete | Doc QA quality | Medium | Poor doc answers | **8** |
| 5 | Response Cache (LLM) | ✅ Implemented | Latency + cost | High | Repeat LLM calls | **8** |
| 6 | Intent Classification | ✅ Fast LLM | Routing accuracy | High | Misrouted queries | **7** |
| 7 | Language Detection | ✅ Rule-based | Hindi support | Medium | Wrong language | **6** |
| 8 | Eval Harness (golden tests) | ✅ Implemented | Quality guard | Continuous | Silent regressions | **6** |
| 9 | OCR / Image Analysis | ⚠️ Partial | Prescription reads | Medium | Can't process images | **6** |
| 10 | LLMOps Observability | ⚠️ Latency only | Quality tracking | Always | No quality visibility | **5** |
| 11 | LLM Fallback / Resilience | ⚠️ Tiers only | Reliability | Low | Single point failure | **4** |

### Next gaps to close (ordered by weight)
- **OCR (weight 6)**: EasyOCR is integrated but prescription/lab image paths are only wired when `ocr_text` is passed externally. End-to-end auto-OCR on upload is not yet live.
- **LLMOps (weight 5)**: Add structured metrics — user satisfaction (thumbs up/down), cache hit rate logged per session, KG hit rate. Needs a lightweight feedback endpoint.
- **LLM Fallback (weight 4)**: Auto-fallback from `main_model → fast_model` on Ollama timeout. Low priority as local Ollama has no rate limits.

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

### 2026-04-03 (2) — Phase 4: One-Click Installer

- `installer/setup.sh` — macOS/Linux installer: Python check, Ollama check, venv, pip install, auto RAM-detection for tier selection, `ollama pull` models, creates `start.sh`
- `installer/setup.bat` — Windows installer: same flow as setup.sh, generates `start.bat`
- `start.sh` / `start.bat` — Daily launch scripts (gitignored, generated by installer): starts Ollama → backend → opens Streamlit in browser
- **Fixed `Dockerfile`**: was PORT 80 with `/api/v1` root-path prefix (wrong) → now PORT 8000, no prefix, correct healthcheck, added system deps for PDF parsing
- **Fixed `.gitignore`**: `data/` rule was blocking `app/knowledge/data/*.json` (curated KB) — added `!app/knowledge/data/` exception so those files are tracked. Also gitignored generated `start.sh`/`start.bat`.

### 2026-04-03 (3) — Phase 6: Fine-Tuning Pipeline

- `training/config.yaml` — all hyperparameters in one place (LR 2e-4, rank 16, all projection layers, 2 epochs — from research doc)
- `training/prepare_dataset.py` — loads `sample_qa.jsonl` + auto-converts all 5 knowledge graph JSON files to ChatML Q&A pairs; deduplicates, shuffles, 90/10 train/eval split
- `training/finetune.py` — Unsloth + QLoRA training; masks inputs so model trains only on completions; saves LoRA adapters + config snapshot
- `training/export_to_ollama.py` — merge LoRA → FP16 GGUF → Q4_K_M quantise → write Modelfile → `ollama create healthypartner`
- `training/data/sample_qa.jsonl` — 30 hand-curated Indian healthcare Q&A pairs (insurance, IRDAI, PMJAY, drug info, lab reports, Hindi examples)

**To run on Colab T4 (~25 min):**
```bash
pip install unsloth trl transformers datasets peft accelerate bitsandbytes pyyaml
python training/prepare_dataset.py
python training/finetune.py
python training/export_to_ollama.py
```

### 2026-04-03 (4) — Code simplification + LLM app patterns (weight-based)

**Code simplification (`/simplify-code`)**
- `app/knowledge/graph.py`: moved `import re` to module level (was repeated inside 4 methods); hoisted `_INTERACTION_SIGNALS`, `_STOPWORDS`, `_DI_STOPWORDS` to module-level constants; `_DI_STOPWORDS` now composes from `_FTS_STOPWORDS` instead of duplicating the shared core.
- `healthypartner_backend.py`: moved `import time` / `import re` from inside `/chat` handler to module top; pre-compiled `<think>` strip regex as module-level `_THINK_RE`.
- `app/main.py`: corrected misleading CORS comment (`"any local port"` → accurate wildcard warning).

**LLM app patterns — weight-based prioritisation**

Evaluated 11 functional areas (see Weight Matrix section above). Top two gaps:

**Weight 8 — Response Cache** (`app/orchestrator.py`)
- Added `_ResponseCache` class: SHA-256 keyed, 30-min TTL, max 500 entries, FIFO+expired eviction.
- Cache check fires after emergency detection (step 2) but before intent classification (step 3) — saves 2 LLM calls per cache hit (~800–2000 ms).
- Cache is bypassed for `has_document=True` and `ocr_text` queries (document-specific, must not bleed across users).
- Route field set to `"cache"` on hit — visible in step trace for monitoring.

**Weight 6 — Eval Harness** (`tests/eval_knowledge_graph.py`)
- 12 golden test cases across 4 categories: drug interactions, medicine alternatives, lab results, government schemes.
- Each case specifies `must_contain` and `must_not` tokens — tests both recall and false-positive suppression.
- `skip=True` flag keeps known data gaps visible without blocking CI.
- Run: `python -m tests.eval_knowledge_graph --verbose`

### 2026-04-04 — GAP-001: Flask backend eliminated

- **Deleted** `healthypartner_backend.py` (Flask) — one backend, one truth
- **Migrated** 3 unique endpoints into `app/main.py` (FastAPI):
  - `POST /healthypartner/generate` — LLM-powered question template generation
  - `POST /webhook` — Twilio WhatsApp/SMS stub (Phase 7 ready)
  - `POST /test` — debug echo endpoint
- **Added** `GenerateRequest` / `GenerateResponse` Pydantic models
- **Added** `Request` to FastAPI imports (needed for raw form/JSON webhook parsing)
- **Removed** `flask` from `requirements.txt`; added `python-multipart` (required for FastAPI form data — Twilio sends form-encoded webhooks)
- **Result**: Single FastAPI server on port 8000 handles all endpoints. All queries go through the 7-step Orchestrator pipeline including emergency detection and safety guardrails.

### 2026-04-04 (2) — GAP-002: Multi-tenancy

**New file: `app/tenant.py`**
- `TenantConfig` dataclass — loads from `tenants/{tenant_id}/config.yaml`; safe defaults derived from `tenant_id`; `resolved_kg_db_path`, `resolved_vector_store_base`, `download_dir` properties
- `TenantManager` — lazily initialises and caches one `Orchestrator` + `KnowledgeGraph` per tenant; `reload_tenant()` evicts cache for hot-reload after KB updates
- `validate_tenant_id()` — strict regex `^[a-zA-Z0-9_-]{1,64}$` prevents path traversal
- Default tenant maps to legacy paths (`data/knowledge.db`, `./db/`) — zero-migration backward compat

**New file: `tenants/default/config.yaml`** — explicit config for the default single-tenant deployment

**Modified: `app/ingestion.py`**
- `process_and_get_retriever()` now accepts `base_path` param (default = legacy `./db`)
- Cache key changed from `document_id` → `(base_path, document_id)` tuple — ensures no cross-tenant cache bleed

**Modified: `app/main.py`**
- Global `orchestrator` replaced with `TenantManager`
- `get_orchestrator()` FastAPI dependency reads `X-Tenant-ID` header (defaults to `"default"`)
- `get_tenant_config()` FastAPI dependency for tenant-scoped paths
- `/chat`, `/hackrx/run`, `/healthypartner/run` all use `Depends(get_orchestrator)` + `Depends(get_tenant_config)`
- Document uploads scoped to `downloaded_files/{tenant_id}/`
- Vector stores scoped to `db/{tenant_id}/{document_id}/`
- New endpoints: `GET /tenants`, `POST /tenants/{tenant_id}/reload`

**How to onboard a new customer:**
1. Create `tenants/{tenant_id}/config.yaml` with their name and optional KB path overrides
2. Load their knowledge base into `tenants/{tenant_id}/knowledge.db`
3. Call `POST /tenants/{tenant_id}/reload` (or restart server)
4. All their API calls pass `X-Tenant-ID: {tenant_id}` — fully isolated

### 2026-04-04 (3) — GAP-003: KB Admin Interface

**`app/knowledge/graph.py`** — added KB mutation methods:
- `import_csv_medicines(rows)` — bulk insert, skips rows missing required fields, rebuilds FTS
- `import_csv_interactions(rows)` — bulk insert drug interactions
- `import_csv_facts(rows)` — bulk insert facts/schemes with auto category upsert
- `import_csv_icd10(rows)` — bulk insert ICD-10 symptom→condition mappings
- `_rebuild_fts()` — rebuilds all three FTS5 indexes after data changes
- `reset_and_reload()` — clears all KB data and reloads from source JSON files
- `_safe_float()` module-level helper for CSV numeric parsing

**`app/admin.py`** — new FastAPI router mounted at `/admin`:
- `GET  /admin`                    → Admin HTML panel
- `GET  /admin/kb/stats`           → Record counts per domain (tenant-aware)
- `POST /admin/kb/upload/medicines`     → CSV upload
- `POST /admin/kb/upload/interactions`  → CSV upload
- `POST /admin/kb/upload/facts`         → CSV upload
- `POST /admin/kb/upload/icd10`         → CSV upload
- `POST /admin/kb/rebuild`         → Rebuild FTS indexes
- `POST /admin/kb/reset`           → Clear + reload from JSON
- Auth: `X-Admin-Key` header vs `HP_ADMIN_KEY` env var (no-op if key not set)
- Uses `request.app.state.tenant_manager` — no circular imports

**`app/main.py`** — `app.state.tenant_manager` set in lifespan; `admin_router` included

**`frontend_web/admin.html`** — minimal dark admin panel:
- Stats dashboard (5 domain counts, auto-refreshed)
- 4 CSV upload sections with column hints (drag-and-drop ready)
- Rebuild FTS + Reset KB buttons with confirmation
- Tenant ID + Admin Key fields in header
- No build tools — vanilla JS, works immediately

**Access:** `http://localhost:8000/admin`

*Last updated: 2026-04-04*
