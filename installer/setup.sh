#!/usr/bin/env bash
# HealthyPartner v2 — One-Click Installer (macOS / Linux)
# Run once: bash installer/setup.sh
set -euo pipefail

# ── Colours ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BLUE}[INFO]${RESET} $*"; }
success() { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }

# ── Resolve project root (works from any directory) ─────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo ""
echo -e "${BOLD}╔════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║       HealthyPartner v2 — Installer        ║${RESET}"
echo -e "${BOLD}║   Privacy-first local healthcare AI        ║${RESET}"
echo -e "${BOLD}╚════════════════════════════════════════════╝${RESET}"
echo ""

# ── Step 1: Python 3.10+ ─────────────────────────────────────────────────────────
info "Checking Python..."

PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
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
    error "Python 3.10 or higher is required.\n  macOS:  brew install python@3.11\n  Linux:  sudo apt install python3.11"
fi
success "Python found: $PYTHON_CMD ($VERSION)"

# ── Step 2: Ollama ───────────────────────────────────────────────────────────────
info "Checking Ollama..."

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
        error "Ollama still not found. Please install it and re-run this script."
    fi
fi
success "Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"

# ── Step 3: Virtual environment ──────────────────────────────────────────────────
VENV_DIR="$PROJECT_ROOT/.venv"

if [ -d "$VENV_DIR" ]; then
    info "Virtual environment already exists — skipping creation"
else
    info "Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    success "Virtual environment created at .venv/"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PIP="$VENV_DIR/bin/pip"

# ── Step 4: Python dependencies ──────────────────────────────────────────────────
info "Installing Python dependencies (this may take 2-5 minutes)..."
"$PIP" install --upgrade pip --quiet
"$PIP" install -r "$PROJECT_ROOT/requirements.txt" --quiet
success "Python dependencies installed"

# ── Step 5: Pull Ollama models ───────────────────────────────────────────────────
info "Pulling AI models via Ollama..."
echo ""
echo "  This downloads ~3 GB of model weights. Only needed once."
echo "  Make sure you have a stable internet connection."
echo ""

# Detect available RAM to choose tier
TOTAL_RAM_GB=8
if command -v python3 &>/dev/null; then
    TOTAL_RAM_GB=$("$VENV_DIR/bin/python" -c "import psutil; print(int(psutil.virtual_memory().total / (1024**3)))" 2>/dev/null || echo 8)
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
    TIER="ultra-light (4 GB RAM)"
fi

echo -e "  Detected ${TOTAL_RAM_GB} GB RAM → using ${BOLD}${TIER}${RESET} tier"
echo "  Main model : $MAIN_MODEL"
echo "  Fast model : $FAST_MODEL"
echo ""

# Start Ollama in background if not running
if ! pgrep -x ollama &>/dev/null; then
    info "Starting Ollama service..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    sleep 3
    success "Ollama service started (PID $OLLAMA_PID)"
fi

info "Pulling $MAIN_MODEL..."
ollama pull "$MAIN_MODEL"
success "$MAIN_MODEL ready"

info "Pulling $FAST_MODEL..."
ollama pull "$FAST_MODEL"
success "$FAST_MODEL ready"

# ── Step 6: Create data directories ─────────────────────────────────────────────
info "Creating data directories..."
mkdir -p "$PROJECT_ROOT/data" "$PROJECT_ROOT/db" "$PROJECT_ROOT/downloaded_files"
success "Directories ready"

# ── Step 7: Write .env with detected tier ────────────────────────────────────────
ENV_FILE="$PROJECT_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
    info "Writing .env configuration..."
    cat > "$ENV_FILE" <<EOF
# HealthyPartner v2 — Auto-generated by installer
HP_MODEL_TIER=balanced
HP_MAIN_MODEL=$MAIN_MODEL
HP_FAST_MODEL=$FAST_MODEL
EOF
    success ".env written"
else
    info ".env already exists — skipping"
fi

# ── Step 8: Write start.sh ───────────────────────────────────────────────────────
info "Creating launch script..."
cat > "$PROJECT_ROOT/start.sh" <<'LAUNCH'
#!/usr/bin/env bash
# HealthyPartner v2 — Launch Script
# Double-click or run: bash start.sh

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

# Activate venv
source .venv/bin/activate

# Start FastAPI backend
echo -e "${GREEN}[2/3]${RESET} Starting backend (http://localhost:8000)..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 3

# Start Streamlit frontend
echo -e "${GREEN}[3/3]${RESET} Starting frontend (http://localhost:8501)..."
echo ""
echo -e "  ${BOLD}HealthyPartner is running!${RESET}"
echo "  Open your browser at: http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo ""

streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null; pkill -f 'uvicorn app.main' 2>/dev/null" EXIT
LAUNCH

chmod +x "$PROJECT_ROOT/start.sh"
success "start.sh created"

# ── Done ─────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔═══════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}${BOLD}║        Installation Complete!             ║${RESET}"
echo -e "${GREEN}${BOLD}╚═══════════════════════════════════════════╝${RESET}"
echo ""
echo "  To start HealthyPartner, run:"
echo ""
echo -e "    ${BOLD}bash start.sh${RESET}"
echo ""
echo "  Or open start.sh in Finder / Files and double-click it."
echo ""
