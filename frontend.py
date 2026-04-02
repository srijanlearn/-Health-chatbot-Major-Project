"""
HealthyPartner v2 — Streamlit Frontend

Healthcare chat interface with local AI, document analysis, and knowledge graph.
No cloud dependencies — all inference runs locally via Ollama.

Endpoints used:
  GET  /health             — Ollama connectivity + model status
  GET  /system/info        — RAM, GPU, tier
  POST /chat               — Free-form chat (no document)
  POST /healthypartner/run — Document Q&A (PDF base64 + question)
"""

import base64
import time

import requests
import streamlit as st

# ── Configuration ───────────────────────────────────────────────────────────────

DEFAULT_API_BASE = "http://localhost:8000"

# Badge labels and colors for each pipeline route
ROUTE_BADGES = {
    "knowledge_graph": ("🗂 Knowledge Graph", "#16a34a", "#dcfce7"),
    "specific_fact":   ("📄 Document Fact",   "#2563eb", "#dbeafe"),
    "general_rag":     ("🔍 Document RAG",    "#7c3aed", "#ede9fe"),
    "direct_llm":      ("🤖 AI Generated",    "#d97706", "#fef3c7"),
    "static":          ("⚡ Instant",          "#64748b", "#f1f5f9"),
    "emergency":       ("🚨 Emergency",        "#dc2626", "#fef2f2"),
}

INTENT_LABELS = {
    "insurance_query":   "Insurance",
    "symptom_check":     "Symptoms",
    "prescription_info": "Prescription",
    "lab_results":       "Lab Results",
    "general_health":    "General Health",
    "ayushman_bharat":   "Ayushman Bharat",
    "drug_info":         "Drug Info",
    "greeting":          "Greeting",
    "emergency":         "Emergency",
    "unknown":           "General",
}

# ── Page setup ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HealthyPartner",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #f0fdf4; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #15803d; }

.route-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
}
.doc-banner {
    background: #dcfce7;
    border: 1px solid #86efac;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
    color: #15803d;
}
.latency-tag {
    font-size: 0.7rem;
    color: #94a3b8;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "messages": [],       # [{"role": str, "content": str, "meta": dict}]
        "active_doc": None,   # {"name": str, "b64": str, "size_kb": int}
        "api_base": DEFAULT_API_BASE,
        "health_cache": {},
        "health_ts": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── API helpers ─────────────────────────────────────────────────────────────────

def _get_health(force: bool = False) -> dict:
    """Cached health check — re-fetches at most once every 30 s."""
    now = time.time()
    if force or now - st.session_state.health_ts > 30:
        try:
            r = requests.get(f"{st.session_state.api_base}/health", timeout=3)
            if r.status_code == 200:
                st.session_state.health_cache = r.json()
        except Exception:
            st.session_state.health_cache = {}
        st.session_state.health_ts = now
    return st.session_state.health_cache


def _get_system_info() -> dict:
    try:
        r = requests.get(f"{st.session_state.api_base}/system/info", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def _chat(message: str) -> dict:
    """Call /chat — no document required."""
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-10:]
    ]
    r = requests.post(
        f"{st.session_state.api_base}/chat",
        json={"message": message, "conversation_history": history},
        timeout=90,
    )
    r.raise_for_status()
    return r.json()  # ChatResponse fields


def _document_qa(message: str, doc_b64: str) -> dict:
    """Call /healthypartner/run — PDF document Q&A."""
    r = requests.post(
        f"{st.session_state.api_base}/healthypartner/run",
        json={"documents": doc_b64, "questions": [message], "is_base64": True},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return {
        "response": data["answers"][0] if data.get("answers") else "No answer returned.",
        "route": "general_rag",
        "intent": "insurance_query",
        "language": "en",
        "is_emergency": False,
        "latency_ms": 0,
    }

# ── UI helpers ──────────────────────────────────────────────────────────────────

def _badge_html(route: str, intent: str, latency_ms: float) -> str:
    label, color, bg = ROUTE_BADGES.get(route, ("🤖 AI Generated", "#d97706", "#fef3c7"))
    intent_label = INTENT_LABELS.get(intent, intent or "General")
    latency_str = f"{latency_ms:.0f} ms" if latency_ms else ""
    html = (
        f'<span class="route-badge" style="background:{bg}; color:{color}; '
        f'border:1px solid {color}40">{label}</span>'
        f'<span class="route-badge" style="background:#f1f5f9; color:#475569; '
        f'border:1px solid #e2e8f0">{intent_label}</span>'
    )
    if latency_str:
        html += f'<span class="latency-tag">{latency_str}</span>'
    return html


def _status_indicator(ok: bool, label: str) -> str:
    dot = "🟢" if ok else "🔴"
    return f"{dot} {label}"

# ── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 💚 HealthyPartner")
    st.caption("Privacy-first local healthcare AI")
    st.divider()

    # Connection status
    health = _get_health()
    ollama_ok = health.get("ollama_connected", False)
    main_ok   = health.get("models", {}).get("main_model", False)
    fast_ok   = health.get("models", {}).get("fast_model", False)

    st.markdown(_status_indicator(ollama_ok, "Ollama"))
    st.markdown(_status_indicator(main_ok, f"`{health.get('models', {}).get('main_model_name', 'main model')}`"))
    st.markdown(_status_indicator(fast_ok, f"`{health.get('models', {}).get('fast_model_name', 'fast model')}`"))

    if not ollama_ok:
        st.warning("Start Ollama: `ollama serve`", icon="⚠️")
    elif not main_ok:
        models = health.get("models", {})
        st.warning(f"`ollama pull {models.get('main_model_name', 'qwen3:4b')}`", icon="⚠️")

    if st.button("↺ Refresh status", use_container_width=True):
        _get_health(force=True)
        st.rerun()

    st.divider()

    # Document upload
    st.markdown("### 📎 Document")

    if st.session_state.active_doc:
        doc = st.session_state.active_doc
        st.success(f"📄 {doc['name']}  ({doc['size_kb']} KB)", icon="✅")
        if st.button("Remove document", use_container_width=True):
            st.session_state.active_doc = None
            st.rerun()
    else:
        uploaded = st.file_uploader(
            "Upload PDF (insurance policy, prescription, lab report)",
            type=["pdf"],
            label_visibility="collapsed",
        )
        if uploaded:
            raw = uploaded.read()
            st.session_state.active_doc = {
                "name": uploaded.name,
                "b64": base64.b64encode(raw).decode("utf-8"),
                "size_kb": round(len(raw) / 1024),
            }
            st.rerun()
        st.caption("No document — general health chat mode")

    st.divider()

    # System info
    tier = health.get("tier", "")
    if tier:
        labels = {"ultra_light": "Ultra-light", "balanced": "Balanced", "quality": "Quality"}
        st.caption(f"**Tier:** {labels.get(tier, tier)}")

    sys_info = _get_system_info()
    if sys_info:
        ram  = sys_info.get("total_ram_gb", 0)
        gpu  = sys_info.get("gpu_name") or ("None" if not sys_info.get("gpu_available") else "")
        arch = sys_info.get("arch", "")
        st.caption(f"RAM {ram:.0f} GB  ·  GPU: {gpu}  ·  {arch}")

    st.divider()

    with st.expander("⚙️ Settings"):
        new_base = st.text_input("API endpoint", value=st.session_state.api_base)
        if new_base != st.session_state.api_base:
            st.session_state.api_base = new_base
            _get_health(force=True)

        if st.button("Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ── Main chat area ──────────────────────────────────────────────────────────────

st.markdown("## 🏥 HealthyPartner")

# Active document banner
if st.session_state.active_doc:
    st.markdown(
        f'<div class="doc-banner">📄 <strong>{st.session_state.active_doc["name"]}</strong>'
        f" ({st.session_state.active_doc['size_kb']} KB) — "
        f"questions will be answered from this document</div>",
        unsafe_allow_html=True,
    )

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:3rem 0; color:#94a3b8;">
        <div style="font-size:3rem; margin-bottom:1rem">💚</div>
        <p style="font-size:1.1rem; color:#475569; font-weight:500;">
            Ask about health insurance, prescriptions, lab reports, or general health.
        </p>
        <p style="font-size:0.85rem;">
            Upload a PDF in the sidebar for document-specific questions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            meta = msg.get("meta", {})
            st.markdown(
                _badge_html(
                    meta.get("route", "direct_llm"),
                    meta.get("intent", "unknown"),
                    meta.get("latency_ms", 0),
                ),
                unsafe_allow_html=True,
            )

# Chat input
user_input = st.chat_input("Ask a health question…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                if st.session_state.active_doc:
                    result = _document_qa(user_input, st.session_state.active_doc["b64"])
                else:
                    result = _chat(user_input)

                st.write(result["response"])
                st.markdown(
                    _badge_html(
                        result.get("route", "direct_llm"),
                        result.get("intent", "unknown"),
                        result.get("latency_ms", 0),
                    ),
                    unsafe_allow_html=True,
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "meta": {
                        "route":        result.get("route", "direct_llm"),
                        "intent":       result.get("intent", "unknown"),
                        "language":     result.get("language", "en"),
                        "is_emergency": result.get("is_emergency", False),
                        "latency_ms":   result.get("latency_ms", 0),
                    },
                })

            except requests.exceptions.ConnectionError:
                msg = (
                    "Cannot connect to the backend. Start it with:\n\n"
                    "```bash\nuvicorn app.main:app --reload\n```"
                )
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg, "meta": {}})

            except requests.exceptions.HTTPError as e:
                msg = f"Backend error {e.response.status_code}: {e.response.text[:200]}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg, "meta": {}})

            except Exception as e:
                msg = f"Unexpected error: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg, "meta": {}})
