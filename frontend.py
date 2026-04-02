"""
HealthyPartner — Enhanced Streamlit Frontend for LLM Document Q&A System

Features:
- Multi-document management with smart caching
- Real-time response streaming
- Evidence/sources display with citations
- Multi-turn conversations with context
- Advanced settings and model selection
- Bulk question generation
- Export capabilities (Markdown, JSON, CSV)
"""

import streamlit as st
import requests
import json
import base64
import time
import hashlib
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# =============================================================================
# Configuration & Constants
# =============================================================================

class ModelType(Enum):
    GEMINI_PRO = "gemini-1.5-pro"
    GEMINI_FLASH = "gemini-1.5-flash"
    GPT4 = "openai-gpt-4o"
    LOCAL_LLAMA = "local-llama"

@dataclass
class Document:
    id: str
    name: str
    bytes: Optional[bytes]
    url: Optional[str]
    uploaded_at: str
    size: int
    page_count: Optional[int] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "uploaded_at": self.uploaded_at,
            "size": self.size,
            "page_count": self.page_count
        }

@dataclass
class QAPair:
    question: str
    answer: str
    sources: Optional[List[Any]]
    timestamp: str
    model: str
    doc_id: str
    
# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="HealthyPartner • Document Q&A",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .doc-card {
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        background: #f9fafb;
    }
    .doc-card:hover {
        border-color: #22c55e;
        background: #f0fdf4;
    }
    .answer-box {
        background: #f9fafb;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #22c55e;
        margin: 1rem 0;
    }
    .source-box {
        background: #fffbeb;
        padding: 1rem;
        border-radius: 6px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize all session state variables with defaults"""
    defaults = {
        'documents': {},  # id -> Document
        'selected_doc_id': None,
        'questions': [],
        'qa_history': [],  # List of QAPair
        'processing_log': [],
        'cache': {},  # cache_key -> answers
        'api_endpoint': 'http://localhost:5000/healthypartner/run',
        'selected_model': ModelType.GEMINI_PRO.value,
        'streaming': True,
        'temperature': 0.5,
        'max_tokens': 1000,
        'chunk_size': 1000,
        'retries': 2,
        'show_debug': False,
        'conversation_context': [],  # For multi-turn conversations
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# Utility Functions
# =============================================================================

def make_doc_id(name: str, content: bytes) -> str:
    """Generate stable document ID from content hash"""
    hash_str = hashlib.sha256(content).hexdigest()[:12]
    clean_name = "".join(c for c in name if c.isalnum() or c in "._-")[:30]
    return f"{clean_name}_{hash_str}"

def make_cache_key(doc_id: str, questions: List[str], model: str) -> str:
    """Generate cache key for question set"""
    q_hash = hashlib.md5(json.dumps(questions, sort_keys=True).encode()).hexdigest()[:8]
    return f"{doc_id}_{model}_{q_hash}"

def format_bytes(size: int) -> str:
    """Format byte size to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def log_message(msg: str, level: str = "INFO"):
    """Add timestamped message to processing log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.processing_log.append(f"[{timestamp}] {level}: {msg}")

def extract_preview_text(pdf_bytes: bytes, max_chars: int = 500) -> str:
    """Extract preview text from PDF (placeholder implementation)"""
    # In production, use PyPDF2 or similar
    return f"PDF document ({format_bytes(len(pdf_bytes))}). Preview requires backend processing."

# =============================================================================
# API Communication
# =============================================================================

def build_request_payload(
    doc_id: str,
    questions: List[str],
    page_ranges: Optional[str] = None,
    include_context: bool = False
) -> Dict[str, Any]:
    """Build request payload for backend API"""
    doc = st.session_state.documents[doc_id]
    
    payload = {
        "questions": questions,
        "model": st.session_state.selected_model,
        "temperature": float(st.session_state.temperature),
        "max_tokens": int(st.session_state.max_tokens),
        "chunk_size": int(st.session_state.chunk_size),
        "is_base64": True,
    }
    
    # Add document content
    if doc.bytes:
        payload["documents"] = base64.b64encode(doc.bytes).decode('utf-8')
    else:
        payload["documents"] = None
        payload["document_url"] = doc.url
    
    # Add optional parameters
    if page_ranges:
        payload["page_ranges"] = page_ranges
    
    if include_context and st.session_state.conversation_context:
        payload["conversation_context"] = st.session_state.conversation_context[-5:]
    
    return payload

def call_backend_api(
    doc_id: str,
    questions: List[str],
    page_ranges: Optional[str] = None,
    progress_callback=None
) -> Dict[str, Any]:
    """Call backend API with retry logic and progress updates"""
    payload = build_request_payload(doc_id, questions, page_ranges, include_context=True)
    endpoint = st.session_state.api_endpoint
    retries = int(st.session_state.retries)
    
    log_message(f"Calling API: {endpoint}")
    
    for attempt in range(retries + 1):
        try:
            if progress_callback:
                progress_callback(30 + (attempt * 10), f"Attempt {attempt + 1}/{retries + 1}")
            
            response = requests.post(
                endpoint,
                json=payload,
                timeout=120,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            log_message(f"API call successful (status {response.status_code})")
            
            return result
            
        except requests.exceptions.Timeout:
            log_message(f"Timeout on attempt {attempt + 1}", "WARNING")
            if attempt < retries:
                time.sleep(2 ** attempt)  # Exponential backoff
        except requests.exceptions.RequestException as e:
            log_message(f"Request error on attempt {attempt + 1}: {str(e)}", "ERROR")
            if attempt < retries:
                time.sleep(2 ** attempt)
        except Exception as e:
            log_message(f"Unexpected error: {str(e)}", "ERROR")
            raise
    
    raise Exception(f"Failed after {retries + 1} attempts")

def generate_questions_from_prompt(prompt: str, count: int = 5) -> List[str]:
    """Generate questions using backend /generate endpoint"""
    try:
        gen_endpoint = st.session_state.api_endpoint.replace("/run", "/generate")
        payload = {
            "action": "generate_questions",
            "prompt": prompt,
            "n": count
        }
        
        response = requests.post(gen_endpoint, json=payload, timeout=30)
        response.raise_for_status()
        
        return response.json().get("questions", [])
    except Exception as e:
        log_message(f"Question generation failed: {e}", "WARNING")
        # Fallback: generate basic questions
        terms = [t.strip() for t in prompt.split() if len(t.strip()) > 3][:5]
        return [f"What information is provided about {term}?" for term in terms]

# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render sidebar with settings and document management"""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # API Configuration
        with st.expander("🔌 API Configuration", expanded=False):
            api = st.text_input(
                "Backend Endpoint",
                value=st.session_state.api_endpoint,
                help="URL of your HealthyPartner backend API"
            )
            st.session_state.api_endpoint = api
            
            if st.button("🔍 Test Connection"):
                try:
                    health_url = api.replace("/run", "/health")
                    resp = requests.get(health_url, timeout=5)
                    if resp.status_code == 200:
                        st.success("✅ Connection successful")
                    else:
                        st.error(f"❌ Server returned {resp.status_code}")
                except:
                    st.error("❌ Cannot connect to backend")
        
        # Model Settings
        with st.expander("🤖 Model Settings", expanded=True):
            models = [m.value for m in ModelType]
            current_idx = models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0
            
            st.selectbox(
                "Model",
                models,
                index=current_idx,
                key="selected_model",
                help="Select the LLM model for processing"
            )
            
            st.slider(
                "Temperature",
                0.0, 1.0,
                float(st.session_state.temperature),
                0.05,
                key="temperature",
                help="Higher = more creative, Lower = more focused"
            )
            
            st.number_input(
                "Max Tokens",
                64, 4000,
                int(st.session_state.max_tokens),
                100,
                key="max_tokens",
                help="Maximum response length"
            )
        
        # Processing Settings
        with st.expander("⚡ Processing Settings"):
            st.number_input(
                "Chunk Size (chars)",
                200, 5000,
                int(st.session_state.chunk_size),
                100,
                key="chunk_size"
            )
            
            st.number_input(
                "Retry Attempts",
                0, 5,
                int(st.session_state.retries),
                key="retries"
            )
            
            st.checkbox(
                "Enable Streaming UI",
                value=st.session_state.streaming,
                key="streaming"
            )
        
        st.markdown("---")
        
        # Document Summary
        st.markdown("### 📚 Documents")
        doc_count = len(st.session_state.documents)
        st.metric("Total Uploaded", doc_count)
        
        if doc_count > 0:
            total_size = sum(d.size for d in st.session_state.documents.values())
            st.metric("Total Size", format_bytes(total_size))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.documents = {}
                st.session_state.selected_doc_id = None
                st.session_state.cache = {}
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        
        # Debug Tools
        st.markdown("### 🐛 Debug")
        st.checkbox("Show Debug Info", key="show_debug")
        
        if st.button("📋 View Logs"):
            with st.expander("Processing Logs", expanded=True):
                for log in st.session_state.processing_log[-50:]:
                    st.text(log)
        
        if st.button("🧹 Clear Cache"):
            st.session_state.cache = {}
            st.session_state.processing_log = []
            st.success("Cache cleared")

def render_document_manager():
    """Render document upload and management interface"""
    st.markdown("### 📎 Document Management")
    
    tab1, tab2 = st.tabs(["📤 Upload", "📋 Manage"])
    
    with tab1:
        # File Upload
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload PDF Documents",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more PDF documents for analysis"
            )
            
            if uploaded_files:
                for file in uploaded_files:
                    file_bytes = file.read()
                    doc_id = make_doc_id(file.name, file_bytes)
                    
                    if doc_id not in st.session_state.documents:
                        doc = Document(
                            id=doc_id,
                            name=file.name,
                            bytes=file_bytes,
                            url=None,
                            uploaded_at=datetime.now().isoformat(),
                            size=len(file_bytes)
                        )
                        st.session_state.documents[doc_id] = doc
                        st.success(f"✅ Added: {file.name}")
                        log_message(f"Document uploaded: {file.name}")
                    else:
                        st.info(f"ℹ️ Already uploaded: {file.name}")
                
                if uploaded_files:
                    st.session_state.selected_doc_id = doc_id
        
        with col2:
            st.markdown("**Quick Stats**")
            if st.session_state.documents:
                st.metric("Documents", len(st.session_state.documents))
        
        # URL Input
        st.markdown("**Or fetch from URL:**")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            url = st.text_input(
                "PDF URL",
                placeholder="https://example.com/document.pdf",
                label_visibility="collapsed"
            )
        
        with col2:
            if st.button("📥 Fetch", use_container_width=True):
                if url:
                    try:
                        filename = url.split("/")[-1] or "remote_doc.pdf"
                        doc_id = f"{filename}_{hash(url) & 0xffffffff:08x}"
                        
                        doc = Document(
                            id=doc_id,
                            name=filename,
                            bytes=None,
                            url=url,
                            uploaded_at=datetime.now().isoformat(),
                            size=0
                        )
                        st.session_state.documents[doc_id] = doc
                        st.session_state.selected_doc_id = doc_id
                        st.success(f"✅ Registered: {filename}")
                        log_message(f"Remote document registered: {url}")
                    except Exception as e:
                        st.error(f"Failed to register URL: {e}")
    
    with tab2:
        # Document List
        if not st.session_state.documents:
            st.info("📭 No documents uploaded yet")
        else:
            for doc_id, doc in st.session_state.documents.items():
                is_selected = doc_id == st.session_state.selected_doc_id
                
                with st.container():
                    col1, col2, col3 = st.columns([6, 2, 1])
                    
                    with col1:
                        label = f"{'✅' if is_selected else '📄'} **{doc.name}**"
                        if st.button(label, key=f"sel_{doc_id}", use_container_width=True):
                            st.session_state.selected_doc_id = doc_id
                            st.rerun()
                    
                    with col2:
                        size_str = format_bytes(doc.size) if doc.size > 0 else "Remote"
                        st.caption(size_str)
                    
                    with col3:
                        if st.button("🗑️", key=f"del_{doc_id}"):
                            del st.session_state.documents[doc_id]
                            if st.session_state.selected_doc_id == doc_id:
                                st.session_state.selected_doc_id = None
                            st.rerun()
                
                if is_selected:
                    with st.expander("📋 Document Details", expanded=False):
                        st.json(doc.to_dict())

def render_question_interface():
    """Render question input and submission interface"""
    st.markdown("### ❓ Ask Questions")
    
    if not st.session_state.selected_doc_id:
        st.warning("⚠️ Please select a document first")
        return
    
    # Selected document info
    doc = st.session_state.documents[st.session_state.selected_doc_id]
    st.info(f"📄 Selected: **{doc.name}** ({format_bytes(doc.size)})")
    
    # Question templates
    with st.expander("📝 Question Templates"):
        templates = [
            "What are the main coverage details for [condition/procedure]?",
            "Are there any exclusions related to [topic]?",
            "What documentation is required for [claim/procedure]?",
            "What is the policy on [specific scenario]?",
            "Summarize the section about [topic].",
        ]
        
        cols = st.columns(2)
        for idx, template in enumerate(templates):
            with cols[idx % 2]:
                if st.button(template, key=f"tmpl_{idx}", use_container_width=True):
                    st.session_state.questions.append(template)
                    st.rerun()
    
    # Dynamic question inputs
    st.markdown("**Your Questions:**")
    
    num_questions = st.number_input(
        "Number of questions",
        min_value=1,
        max_value=10,
        value=max(len(st.session_state.questions), 1)
    )
    
    questions = []
    for i in range(num_questions):
        default = st.session_state.questions[i] if i < len(st.session_state.questions) else ""
        q = st.text_area(
            f"Question {i + 1}",
            value=default,
            key=f"q_input_{i}",
            height=100,
            placeholder="Enter your question here..."
        )
        if q.strip():
            questions.append(q.strip())
    
    st.session_state.questions = questions
    
    # Page range filter
    page_range = st.text_input(
        "Page Range (optional)",
        placeholder="e.g., 1-5, 8, 10-12",
        help="Restrict analysis to specific pages"
    )
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        submit = st.button(
            "🚀 Get Answers",
            type="primary",
            use_container_width=True,
            disabled=len(questions) == 0
        )
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.questions = []
            st.rerun()
    
    with col3:
        if st.button("💾 Save Q's", use_container_width=True):
            if questions:
                q_text = "\n".join(questions)
                st.download_button(
                    "Download",
                    q_text,
                    file_name="questions.txt",
                    use_container_width=True
                )
    
    # Handle submission
    if submit:
        process_questions(questions, page_range)

def process_questions(questions: List[str], page_range: Optional[str] = None):
    """Process questions and get answers from backend"""
    doc_id = st.session_state.selected_doc_id
    model = st.session_state.selected_model
    
    # Check cache
    cache_key = make_cache_key(doc_id, questions, model)
    
    if cache_key in st.session_state.cache:
        st.success("⚡ Using cached results")
        qa_pairs = st.session_state.cache[cache_key]
        st.session_state.qa_history.extend(qa_pairs)
        return
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(pct, msg):
        progress_bar.progress(pct)
        status_text.text(msg)
    
    try:
        update_progress(10, "📄 Preparing document...")
        time.sleep(0.5)
        
        update_progress(20, "🔄 Calling backend API...")
        result = call_backend_api(doc_id, questions, page_range, update_progress)
        
        update_progress(90, "📝 Processing responses...")
        
        # Parse results
        answers = result.get("answers") or result.get("responses") or []
        sources = result.get("sources") or result.get("evidence") or [None] * len(answers)
        
        # Create QA pairs
        qa_pairs = []
        for q, a, s in zip(questions, answers, sources):
            qa_pair = QAPair(
                question=q,
                answer=a or "No answer provided",
                sources=s,
                timestamp=datetime.now().isoformat(),
                model=model,
                doc_id=doc_id
            )
            qa_pairs.append(qa_pair)
        
        # Update session state
        st.session_state.qa_history.extend(qa_pairs)
        st.session_state.cache[cache_key] = qa_pairs
        
        # Add to conversation context
        for qa in qa_pairs:
            st.session_state.conversation_context.append({
                "question": qa.question,
                "answer": qa.answer
            })
        
        update_progress(100, "✅ Complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Processed {len(qa_pairs)} questions successfully")
        log_message(f"Successfully processed {len(qa_pairs)} questions")
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error: {str(e)}")
        log_message(f"Processing failed: {str(e)}", "ERROR")

def render_results():
    """Render Q&A results and history"""
    st.markdown("### 💡 Answers & Insights")
    
    if not st.session_state.qa_history:
        st.info("📭 No answers yet. Submit questions in the Questions tab.")
        return
    
    # Show recent answers
    for idx, qa in enumerate(reversed(st.session_state.qa_history[-10:]), start=1):
        with st.expander(f"**Q{idx}:** {qa.question[:100]}...", expanded=(idx == 1)):
            # Answer
            st.markdown("#### 💬 Answer")
            st.markdown(f'<div class="answer-box">{qa.answer}</div>', unsafe_allow_html=True)
            
            # Sources
            if qa.sources:
                st.markdown("#### 📚 Sources & Evidence")
                sources_list = qa.sources if isinstance(qa.sources, list) else [qa.sources]
                for s_idx, source in enumerate(sources_list, 1):
                    st.markdown(f'<div class="source-box">**Source {s_idx}:** {source}</div>', unsafe_allow_html=True)
            
            # Metadata
            with st.expander("ℹ️ Metadata"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"**Model:** {qa.model}")
                with col2:
                    st.caption(f"**Time:** {qa.timestamp[:19]}")
                with col3:
                    doc = st.session_state.documents.get(qa.doc_id)
                    st.caption(f"**Doc:** {doc.name if doc else 'Unknown'}")
            
            # Follow-up
            follow_up = st.text_input(f"Follow-up question", key=f"follow_{idx}")
            if st.button(f"Ask Follow-up", key=f"ask_follow_{idx}"):
                if follow_up.strip():
                    st.session_state.questions = [follow_up.strip()]
                    st.rerun()

def render_export():
    """Render export functionality"""
    st.markdown("### 📥 Export Results")
    
    if not st.session_state.qa_history:
        st.info("No data to export")
        return
    
    col1, col2, col3 = st.columns(3)
    
    # Markdown export
    with col1:
        md_content = "# HealthyPartner Q&A Export\n\n"
        md_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md_content += "---\n\n"
        
        for idx, qa in enumerate(st.session_state.qa_history, 1):
            md_content += f"## Question {idx}\n\n"
            md_content += f"**Q:** {qa.question}\n\n"
            md_content += f"**A:** {qa.answer}\n\n"
            if qa.sources:
                md_content += f"**Sources:** {qa.sources}\n\n"
            md_content += "---\n\n"
        
        st.download_button(
            "📄 Download Markdown",
            md_content,
            file_name=f"healthypartner_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    # JSON export
    with col2:
        json_content = json.dumps(
            [asdict(qa) for qa in st.session_state.qa_history],
            indent=2,
            default=str
        )
        
        st.download_button(
            "📊 Download JSON",
            json_content,
            file_name=f"healthypartner_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # CSV export
    with col3:
        csv_content = "Question,Answer,Model,Timestamp,Document\n"
        for qa in st.session_state.qa_history:
            doc = st.session_state.documents.get(qa.doc_id)
            doc_name = doc.name if doc else "Unknown"
            csv_content += f'"{qa.question}","{qa.answer}","{qa.model}","{qa.timestamp}","{doc_name}"\n'
        
        st.download_button(
            "📑 Download CSV",
            csv_content,
            file_name=f"healthypartner_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

def render_bulk_tools():
    """Render bulk import and question generation tools"""
    st.markdown("### 🗂️ Bulk Operations")
    
    tab1, tab2 = st.tabs(["📝 Bulk Import", "⚡ Auto-Generate"])
    
    with tab1:
        st.markdown("**Import Multiple Questions**")
        bulk_text = st.text_area(
            "Paste questions (one per line)",
            height=200,
            placeholder="What is the coverage for X?\nAre there exclusions for Y?\nHow to claim Z?"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ Import Questions", use_container_width=True):
                if bulk_text.strip():
                    new_questions = [q.strip() for q in bulk_text.split('\n') if q.strip()]
                    st.session_state.questions = new_questions
                    st.success(f"✅ Replaced with {len(new_questions)} questions")
                    st.rerun()
    
    with tab2:
        st.markdown("**AI-Powered Question Generation**")
        
        seed_prompt = st.text_area(
            "Describe what you want to know",
            height=100,
            placeholder="e.g., 'insurance coverage for maternity care, including prenatal visits, delivery, and postpartum care'"
        )
        
        num_questions = st.slider("Number of questions to generate", 3, 15, 5)
        
        if st.button("✨ Generate Questions", use_container_width=True):
            if seed_prompt.strip():
                with st.spinner("Generating questions..."):
                    try:
                        generated = generate_questions_from_prompt(seed_prompt, num_questions)
                        st.session_state.questions.extend(generated)
                        st.success(f"✅ Generated {len(generated)} questions")
                        
                        # Show preview
                        with st.expander("📋 Generated Questions", expanded=True):
                            for i, q in enumerate(generated, 1):
                                st.write(f"{i}. {q}")
                        
                        log_message(f"Generated {len(generated)} questions from prompt")
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">💚 HealthyPartner</h1>
        <p style="color: white; margin: 0; opacity: 0.9;">
            Intelligent Document Q&A System • Upload PDFs, ask questions, get evidence-backed answers
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Documents & Questions",
        "💡 Results",
        "📥 Export",
        "🗂️ Bulk Tools"
    ])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            render_document_manager()
        
        with col2:
            render_question_interface()
    
    with tab2:
        render_results()
    
    with tab3:
        render_export()
    
    with tab4:
        render_bulk_tools()
    
    # Debug info
    if st.session_state.show_debug:
        with st.expander("🐛 Debug Information", expanded=False):
            st.markdown("**Session State:**")
            debug_state = {
                "documents": len(st.session_state.documents),
                "selected_doc": st.session_state.selected_doc_id,
                "questions": len(st.session_state.questions),
                "qa_history": len(st.session_state.qa_history),
                "cache_size": len(st.session_state.cache),
                "api_endpoint": st.session_state.api_endpoint,
                "model": st.session_state.selected_model,
            }
            st.json(debug_state)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem;">
        <p><strong>HealthyPartner</strong> • Powered by Advanced LLMs</p>
        <p style="font-size: 0.9rem;">
            Need help? Check the <a href="#" style="color: #22c55e;">documentation</a> or 
            <a href="#" style="color: #22c55e;">contact support</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
