#!/usr/bin/env bash
# start.sh

set -e

# Colors for UI
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 🚀 Starting HealthyPartner v2 ===${NC}"

# Source virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Cleanup function to kill background processes on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    echo -e "${GREEN}Services stopped successfully.${NC}"
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM to run the cleanup function
trap cleanup SIGINT SIGTERM

echo -e "${GREEN}[1/2] Starting Flask Backend (healthypartner_backend.py) on Port 5000...${NC}"
python3 healthypartner_backend.py &
BACKEND_PID=$!

# Wait a brief moment to ensure backend starts before hitting it with frontend
sleep 2

# Check if backend crashed immediately
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}❌ Backend failed to start. Check logs above.${NC}"
    exit 1
fi

echo -e "${GREEN}[2/2] Starting Streamlit Frontend (frontend.py) on Port 8501...${NC}"
streamlit run frontend.py

# If Streamlit exits normally, cleanup
cleanup
