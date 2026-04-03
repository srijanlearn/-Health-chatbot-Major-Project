// State
let state = {
    apiBase: 'http://localhost:8000',
    messages: [],
    activeDoc: null, // { name: str, b64: str, size: int }
    health: { ollama_connected: false, models: {} }
};

// DOM Elements
const els = {
    apiInput: document.getElementById('api-base-url'),
    btnRefresh: document.getElementById('btn-refresh'),
    btnClassChat: document.getElementById('btn-clear-chat'),
    
    // Status
    statOllama: document.getElementById('status-ollama'),
    statModMain: document.getElementById('status-model-main'),
    statModFast: document.getElementById('status-model-fast'),
    lblModMain: document.getElementById('label-model-main'),
    lblModFast: document.getElementById('label-model-fast'),
    
    // Upload
    fileInput: document.getElementById('file-upload'),
    dropZone: document.getElementById('drop-zone'),
    docInfo: document.getElementById('active-doc-info'),
    docName: document.getElementById('doc-name'),
    btnRemoveDoc: document.getElementById('btn-remove-doc'),
    chatSubtitle: document.getElementById('chat-subtitle'),
    
    // Sys Info
    sysTier: document.getElementById('sys-tier'),
    sysHardware: document.getElementById('sys-hardware'),
    
    // Chat
    chatForm: document.getElementById('chat-form'),
    chatInput: document.getElementById('chat-input'),
    chatMessages: document.getElementById('chat-messages'),
    emptyState: document.getElementById('empty-state'),
    typingIndicator: document.getElementById('typing-indicator')
};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    els.apiInput.value = state.apiBase;
    
    setupEventListeners();
    fetchHealth();
    fetchSystemInfo();
});

function setupEventListeners() {
    // API change
    els.apiInput.addEventListener('change', (e) => {
        state.apiBase = e.target.value.replace(/\/$/, "");
        fetchHealth();
        fetchSystemInfo();
    });

    els.btnRefresh.addEventListener('click', fetchHealth);
    
    els.btnClassChat.addEventListener('click', () => {
        state.messages = [];
        renderMessages();
    });

    // File Upload
    els.fileInput.addEventListener('change', handleFileUpload);
    els.btnRemoveDoc.addEventListener('click', removeDocument);
    
    // Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => {
        els.dropZone.addEventListener(ev, preventDefaults, false);
    });
    
    els.dropZone.addEventListener('dragover', () => els.dropZone.classList.add('dragover'));
    els.dropZone.addEventListener('dragleave', () => els.dropZone.classList.remove('dragover'));
    els.dropZone.addEventListener('drop', (e) => {
        els.dropZone.classList.remove('dragover');
        const dt = e.dataTransfer;
        if(dt.files && dt.files.length) {
            els.fileInput.files = dt.files;
            handleFileUpload({ target: els.fileInput });
        }
    });

    // Chat
    els.chatForm.addEventListener('submit', handleChatSubmit);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// System API Calls
async function fetchHealth() {
    try {
        const res = await fetch(`${state.apiBase}/health`);
        const data = await res.json();
        // Backend returns: { status, engine: { ollama_connected, models: {...}, tier, system: {...} } }
        state.health = data.engine || data;
        updateStatusUI();
    } catch (e) {
        state.health = { ollama_connected: false, models: {} };
        updateStatusUI();
    }
}

async function fetchSystemInfo() {
    try {
        const res = await fetch(`${state.apiBase}/system/info`);
        const data = await res.json();
        // /system/info returns flat object with hardware fields
        const tierLabels = { ultra_light: 'Ultra-light', balanced: 'Balanced', quality: 'Quality' };
        const tier = data.current_tier || data.recommended_tier || '';
        els.sysTier.textContent = `Tier: ${tierLabels[tier] || tier || 'Detecting...'}`;
        const gpuStr = data.gpu_name || (data.gpu_available ? 'Yes' : 'None');
        els.sysHardware.textContent = `RAM: ${Math.round(data.total_ram_gb || 0)} GB · GPU: ${gpuStr} · ${data.arch || ''}`;
    } catch (e) {
        console.warn('System info not available', e);
    }
}

function updateStatusUI() {
    const isOk = (el, val) => { el.className = `status-dot ${val ? 'ok' : ''}`; };

    // state.health is the engine sub-object: { ollama_connected, models: { main_model, fast_model, ... }, tier }
    const h = state.health;
    isOk(els.statOllama, h.ollama_connected);
    isOk(els.statModMain, h.models?.main_model);
    isOk(els.statModFast, h.models?.fast_model);

    els.lblModMain.textContent = h.models?.main_model_name || 'Main Model';
    els.lblModFast.textContent = h.models?.fast_model_name || 'Fast Model';
}

// Document Handling
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    if (file.type !== "application/pdf") {
         alert("Only PDF documents are supported.");
         return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
        const base64Str = event.target.result.split(',')[1];
        state.activeDoc = {
            name: file.name,
            size: Math.round(file.size / 1024),
            b64: base64Str
        };
        updateDocUI();
    };
    reader.readAsDataURL(file);
}

function removeDocument() {
    state.activeDoc = null;
    els.fileInput.value = "";
    updateDocUI();
}

function updateDocUI() {
    if (state.activeDoc) {
        els.docName.textContent = `${state.activeDoc.name} (${state.activeDoc.size} KB)`;
        els.dropZone.classList.add('hidden');
        els.docInfo.classList.remove('hidden');
        els.chatSubtitle.textContent = `Document Mode: Analyzing ${state.activeDoc.name}`;
    } else {
        els.dropZone.classList.remove('hidden');
        els.docInfo.classList.add('hidden');
        els.chatSubtitle.textContent = "General Health Mode (No Document)";
    }
}

// Chat Handling
async function handleChatSubmit(e) {
    e.preventDefault();
    const msg = els.chatInput.value.trim();
    if (!msg) return;
    
    els.chatInput.value = "";
    addMessage("user", msg);
    
    els.typingIndicator.classList.remove('hidden');
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;

    try {
        let answerData;
        if (state.activeDoc) {
            answerData = await _doDocQuery(msg);
        } else {
            answerData = await _doChatQuery(msg);
        }
        
        addMessage("assistant", answerData.response, {
            route: answerData.route,
            intent: answerData.intent,
            latency_ms: answerData.latency_ms
        });
        
    } catch (e) {
        addMessage("assistant", `Error connecting to backend: ${e.message}`, { route: "error", intent: "error" });
    } finally {
        els.typingIndicator.classList.add('hidden');
    }
}

async function _doChatQuery(msg) {
    const history = state.messages.slice(-10).map(m => ({ role: m.role, content: m.content }));

    const res = await fetch(`${state.apiBase}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, conversation_history: history })
    });

    if (res.status === 404) {
        // /chat not yet on backend — fall through a basic message
        throw new Error('The /chat endpoint is not available on the backend. Please upload a document to use Document Q&A mode, or ask the developer to add the /chat route.');
    }
    if (!res.ok) throw new Error(`Backend error ${res.status}: ${res.statusText}`);
    return await res.json();
}

async function _doDocQuery(msg) {
    const res = await fetch(`${state.apiBase}/healthypartner/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
            questions: [msg], 
            documents: state.activeDoc.b64, 
            is_base64: true 
        })
    });
    
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    
    return {
        response: data.answers && data.answers.length > 0 ? data.answers[0] : "No answer found in document.",
        route: "general_rag",
        intent: "document_qa",
        latency_ms: 0
    };
}

function addMessage(role, content, meta = null) {
    state.messages.push({ role, content, meta });
    renderMessages();
}

function renderMessages() {
    if (state.messages.length > 0) {
        els.emptyState.style.display = 'none';
    } else {
        els.emptyState.style.display = 'block';
    }
    
    // Keep nodes we already rendered, just re-render to be simple
    // Select all messages and remove them
    const existing = els.chatMessages.querySelectorAll('.message');
    existing.forEach(e => e.remove());
    
    state.messages.forEach(msg => {
        const div = document.createElement('div');
        div.className = `message ${msg.role === 'user' ? 'user' : 'bot'}`;
        
        let metaHtml = '';
        if (msg.role === 'assistant' && msg.meta) {
            const r = msg.meta.route || 'direct_llm';
            const i = msg.meta.intent || 'general';
            metaHtml = `
            <div class="msg-meta">
                <span class="badge intent">${r.replace('_', ' ').toUpperCase()}</span>
                <span class="badge intent">${i.replace('_', ' ').toUpperCase()}</span>
                ${msg.meta.latency_ms ? `<span>${Math.round(msg.meta.latency_ms)}ms</span>` : ''}
            </div>`;
        }
        
        div.innerHTML = `
            <div class="msg-bubble">
                ${escapeHTML(msg.content).replace(/\n/g, '<br>')}
            </div>
            ${metaHtml}
        `;
        
        els.chatMessages.insertBefore(div, els.typingIndicator);
    });
    
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
}

function escapeHTML(str) {
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag]));
}
