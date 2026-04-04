#!/usr/bin/env bash
# HealthyPartner v2 — One-Click Installer (macOS / Linux)
#
# Usage:
#   bash installer/setup.sh             # fresh install
#   bash installer/setup.sh --update    # upgrade existing install
#   bash installer/setup.sh --dry-run   # pre-flight checks only, no changes
#
set -euo pipefail

# ── Colours ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BLUE}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
step()    { echo -e "\n${BOLD}── $* ──────────────────────────────────────${RESET}"; }

# ── Flags ────────────────────────────────────────────────────────────────────────
UPDATE_MODE=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --update)   UPDATE_MODE=true ;;
        --dry-run)  DRY_RUN=true ;;
        --help|-h)
            echo "Usage: bash installer/setup.sh [--update] [--dry-run]"
            echo ""
            echo "  (no flags)   Fresh install"
            echo "  --update     Re-install deps and re-pull models without recreating venv"
            echo "  --dry-run    Run pre-flight checks only — make no changes"
            exit 0 ;;
        *) error "Unknown option: $arg  (use --help for usage)" ;;
    esac
done

# ── Resolve project root ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo ""
echo -e "${BOLD}╔════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║       HealthyPartner v2 — Installer        ║${RESET}"
echo -e "${BOLD}║   Privacy-first local healthcare AI        ║${RESET}"
if $UPDATE_MODE; then
echo -e "${BOLD}║            [ UPDATE MODE ]                 ║${RESET}"
fi
if $DRY_RUN; then
echo -e "${BOLD}║            [ DRY RUN ]                     ║${RESET}"
fi
echo -e "${BOLD}╚════════════════════════════════════════════╝${RESET}"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────────────────

PREFLIGHT_ERRORS=0

step "Pre-flight checks"

# 1. Python version
info "Checking Python..."
PYTHON_CMD=""
for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VERSION=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        MAJOR=$(echo "$VERSION" | cut -d. -f1)
        MINOR=$(echo "$VERSION" | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "  ${RED}[FAIL]${RESET} Python 3.10+ not found"
    echo "         macOS:  brew install python@3.11"
    echo "         Linux:  sudo apt install python3.11"
    PREFLIGHT_ERRORS=$((PREFLIGHT_ERRORS + 1))
else
    success "Python $VERSION ($PYTHON_CMD)"
fi

# 2. Disk space — need at least 5 GB free (models ~3 GB + venv ~1 GB + buffer)
info "Checking disk space..."
REQUIRED_GB=5
FREE_KB=$(df -k "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
FREE_GB=$(( FREE_KB / 1024 / 1024 ))
if [ "$FREE_GB" -lt "$REQUIRED_GB" ]; then
    echo -e "  ${RED}[FAIL]${RESET} Only ${FREE_GB} GB free at $PROJECT_ROOT (need ${REQUIRED_GB} GB)"
    echo "         Free up disk space before installing."
    PREFLIGHT_ERRORS=$((PREFLIGHT_ERRORS + 1))
else
    success "${FREE_GB} GB free (need ${REQUIRED_GB} GB)"
fi

# 3. Port 8000 availability
info "Checking port 8000 (backend)..."
if lsof -iTCP:8000 -sTCP:LISTEN &>/dev/null 2>&1; then
    PROC=$(lsof -iTCP:8000 -sTCP:LISTEN -Fp 2>/dev/null | head -1 | tr -d 'p')
    PROC_NAME=$(ps -p "$PROC" -o comm= 2>/dev/null || echo "unknown")
    warn "Port 8000 is in use by $PROC_NAME (PID $PROC)"
    echo "       Stop it before starting HealthyPartner, or it won't bind."
else
    success "Port 8000 is free"
fi

# 4. Port 8501 availability
info "Checking port 8501 (frontend)..."
if lsof -iTCP:8501 -sTCP:LISTEN &>/dev/null 2>&1; then
    PROC=$(lsof -iTCP:8501 -sTCP:LISTEN -Fp 2>/dev/null | head -1 | tr -d 'p')
    PROC_NAME=$(ps -p "$PROC" -o comm= 2>/dev/null || echo "unknown")
    warn "Port 8501 is in use by $PROC_NAME (PID $PROC)"
    echo "       The Streamlit frontend may fail to start."
else
    success "Port 8501 is free"
fi

# 5. Ollama available
info "Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    echo -e "  ${YELLOW}[WARN]${RESET} Ollama is not installed — you will be prompted to install it"
    # Not a hard failure during pre-flight; handled interactively below
else
    OLLAMA_VER=$(ollama --version 2>/dev/null | head -1 || echo "installed")
    success "Ollama: $OLLAMA_VER"

    # Check if Ollama API is reachable on the standard port
    if pgrep -x ollama &>/dev/null || pgrep -x "Ollama" &>/dev/null; then
        if curl -sf --max-time 3 http://localhost:11434/api/tags &>/dev/null; then
            success "Ollama API reachable on port 11434"
        else
            warn "Ollama process is running but not answering on port 11434"
            echo "       Check if OLLAMA_HOST is set to a different address."
            echo "       HealthyPartner expects http://localhost:11434"
        fi
    fi
fi

# 6. Internet connectivity (soft check — models may already be cached)
info "Checking internet connectivity..."
if curl -sf --max-time 5 https://ollama.com > /dev/null 2>&1; then
    success "Internet reachable"
    OFFLINE=false
else
    warn "Cannot reach ollama.com — running in offline mode"
    echo "       Model download will be skipped if models are already cached."
    OFFLINE=true
fi

# ── Abort if hard pre-flight failures ────────────────────────────────────────────
if [ "$PREFLIGHT_ERRORS" -gt 0 ]; then
    echo ""
    echo -e "${RED}Pre-flight checks failed ($PREFLIGHT_ERRORS issue(s)). Fix them and re-run.${RESET}"
    exit 1
fi

if $DRY_RUN; then
    echo ""
    success "Dry run complete — no changes made."
    exit 0
fi

# ── Step 1: Python (already verified above) ──────────────────────────────────────
# PYTHON_CMD is set from pre-flight

# ── Step 2: Ollama ───────────────────────────────────────────────────────────────
step "Ollama"
if ! command -v ollama &>/dev/null; then
    warn "Ollama is not installed."
    echo ""
    echo -e "  Install it from: ${BOLD}https://ollama.com/download${RESET}"
    echo ""
    echo "  macOS:  Download the .dmg from the website, or:"
    echo "          brew install ollama"
    echo ""
    echo "  Linux:  curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    read -r -p "  Press Enter after installing Ollama, or Ctrl+C to cancel... "

    if ! command -v ollama &>/dev/null; then
        error "Ollama still not found. Install it and re-run this script."
    fi
fi
success "Ollama ready"

# ── Step 3: Virtual environment ──────────────────────────────────────────────────
step "Python virtual environment"
VENV_DIR="$PROJECT_ROOT/.venv"

if $UPDATE_MODE; then
    info "Update mode — skipping venv recreation"
elif [ -d "$VENV_DIR" ]; then
    info "Virtual environment already exists — skipping creation"
else
    info "Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    success "Virtual environment created at .venv/"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PIP="$VENV_DIR/bin/pip"

# ── Step 4: Python dependencies ──────────────────────────────────────────────────
step "Python dependencies"
info "Installing Python dependencies (this may take 2-5 minutes)..."
"$PIP" install --upgrade pip --quiet
"$PIP" install -r "$PROJECT_ROOT/requirements.txt" --quiet
success "Python dependencies installed"

# ── Step 5: Detect RAM tier ──────────────────────────────────────────────────────
TOTAL_RAM_GB=8
if command -v python3 &>/dev/null; then
    TOTAL_RAM_GB=$("$VENV_DIR/bin/python" \
        -c "import psutil; print(int(psutil.virtual_memory().total / (1024**3)))" \
        2>/dev/null || echo 8)
fi

if [ "$TOTAL_RAM_GB" -ge 16 ]; then
    MAIN_MODEL="qwen3:4b"
    FAST_MODEL="qwen2.5:0.5b"
    TIER="quality (16 GB+ RAM)"
elif [ "$TOTAL_RAM_GB" -ge 8 ]; then
    MAIN_MODEL="qwen3:4b"
    FAST_MODEL="qwen2.5:0.5b"
    TIER="balanced (8 GB+ RAM)"
else
    MAIN_MODEL="qwen2.5:1.5b"
    FAST_MODEL="qwen2.5:0.5b"
    TIER="ultra-light (< 8 GB RAM)"
fi

# ── Step 6: Pull Ollama models ───────────────────────────────────────────────────
step "AI models"
echo -e "  Detected ${TOTAL_RAM_GB} GB RAM → ${BOLD}${TIER}${RESET} tier"
echo "  Main model : $MAIN_MODEL"
echo "  Fast model : $FAST_MODEL"
echo ""

# Start Ollama in background if not running
if ! pgrep -x ollama &>/dev/null; then
    info "Starting Ollama service..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    # Wait up to 10 seconds for it to be ready
    for i in $(seq 1 10); do
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then
            break
        fi
        sleep 1
    done
    success "Ollama service started (PID $OLLAMA_PID)"
fi

# Check if models are already present
_model_exists() {
    ollama list 2>/dev/null | grep -q "^$1"
}

if $OFFLINE; then
    if _model_exists "$MAIN_MODEL" && _model_exists "$FAST_MODEL"; then
        success "Models already cached — skipping download (offline)"
    else
        warn "Offline and models not cached. Connect to internet and re-run."
        warn "Continuing install — you must pull models manually:"
        echo "  ollama pull $MAIN_MODEL"
        echo "  ollama pull $FAST_MODEL"
    fi
else
    # Pull only if not already present (idempotent)
    if _model_exists "$MAIN_MODEL"; then
        success "$MAIN_MODEL already cached"
    else
        info "Pulling $MAIN_MODEL (~2-3 GB)..."
        ollama pull "$MAIN_MODEL"
        success "$MAIN_MODEL ready"
    fi

    if _model_exists "$FAST_MODEL"; then
        success "$FAST_MODEL already cached"
    else
        info "Pulling $FAST_MODEL (~400 MB)..."
        ollama pull "$FAST_MODEL"
        success "$FAST_MODEL ready"
    fi
fi

# ── Step 7: Data directories ─────────────────────────────────────────────────────
step "Data directories"
mkdir -p "$PROJECT_ROOT/data" "$PROJECT_ROOT/db" "$PROJECT_ROOT/downloaded_files" \
         "$PROJECT_ROOT/tenants/default"
success "Directories ready"

# ── Step 8: .env ─────────────────────────────────────────────────────────────────
step "Configuration"
ENV_FILE="$PROJECT_ROOT/.env"
if $UPDATE_MODE && [ -f "$ENV_FILE" ]; then
    info ".env already exists — skipping (use --update to preserve your config)"
elif [ ! -f "$ENV_FILE" ]; then
    info "Writing .env configuration..."
    cat > "$ENV_FILE" <<EOF
# HealthyPartner v2 — Auto-generated by installer
HP_MODEL_TIER=balanced
HP_MAIN_MODEL=$MAIN_MODEL
HP_FAST_MODEL=$FAST_MODEL
EOF
    success ".env written"
else
    info ".env already exists — not overwritten"
fi

# ── Step 9: launch script ────────────────────────────────────────────────────────
step "Launch script"
cat > "$PROJECT_ROOT/start.sh" <<'LAUNCH'
#!/usr/bin/env bash
# HealthyPartner v2 — Launch Script
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'; BOLD='\033[1m'; RESET='\033[0m'

echo ""
echo -e "${BOLD}Starting HealthyPartner v2...${RESET}"
echo ""

# Start Ollama if not running
if ! pgrep -x ollama &>/dev/null; then
    echo -e "${GREEN}[1/3]${RESET} Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 2
else
    echo -e "${GREEN}[1/3]${RESET} Ollama already running"
fi

source .venv/bin/activate

echo -e "${GREEN}[2/3]${RESET} Starting backend (http://localhost:8000)..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 3

echo -e "${GREEN}[3/3]${RESET} Starting frontend (http://localhost:8501)..."
echo ""
echo -e "  ${BOLD}HealthyPartner is running!${RESET}"
echo "  Open your browser at: http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo ""

trap "kill $BACKEND_PID 2>/dev/null; pkill -f 'uvicorn app.main' 2>/dev/null" EXIT
streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0
LAUNCH

chmod +x "$PROJECT_ROOT/start.sh"
success "start.sh created"

# ── Done ──────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔═══════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}${BOLD}║        Installation Complete!             ║${RESET}"
echo -e "${GREEN}${BOLD}╚═══════════════════════════════════════════╝${RESET}"
echo ""
if $UPDATE_MODE; then
    echo "  HealthyPartner has been updated."
else
    echo "  To start HealthyPartner, run:"
    echo ""
    echo -e "    ${BOLD}bash start.sh${RESET}"
    echo ""
fi
