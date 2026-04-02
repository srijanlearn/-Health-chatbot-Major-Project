#!/usr/bin/env bash
# setup.sh

set -e

# Colors for UI
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 🏥 HealthyPartner v2 Setup ===${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python 3 is not installed. Please install Python 3.11+.${NC}"
    exit 1
fi

# Setup Virtual Env (Optional but recommended)
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
fi
source venv/bin/activate

# Install requirements
echo -e "${BLUE}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Create .env if missing
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo -e "${BLUE}Creating .env file from .env.example...${NC}"
    cp .env.example .env
fi

# Ensure data directories exist
mkdir -p app/data/chroma
mkdir -p app/data/uploads

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama is not installed. Please install it from https://ollama.com before continuing.${NC}"
else
    echo -e "${BLUE}Ollama found! Checking for local models...${NC}"
    # Start ollama daemon silently if it's not running
    if ! pgrep -x "ollama" > /dev/null; then
        ollama serve >/dev/null 2>&1 &
        sleep 3
    fi
    
    echo -e "${GREEN}Verifying qwen3:4b (Generator)...${NC}"
    ollama pull qwen3:4b || echo -e "${YELLOW}qwen3:4b might not be available exactly with this tag or wait for network.${NC}"
    
    echo -e "${GREEN}Verifying qwen2.5:0.5b (Router)...${NC}"
    ollama pull qwen2.5:0.5b || echo -e "${YELLOW}qwen2.5:0.5b fallback..${NC}"
fi

echo -e "${GREEN}✅ Setup complete! You can now run ./start.sh${NC}"
