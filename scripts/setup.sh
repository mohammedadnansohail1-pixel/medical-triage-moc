#!/bin/bash
set -e

echo "========================================"
echo "Medical Triage AI - Setup Script"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}[1/6] Checking Python version...${NC}"
python3 --version || { echo -e "${RED}Python 3 not found!${NC}"; exit 1; }

# Create virtual environment
echo -e "\n${YELLOW}[2/6] Creating virtual environment...${NC}"
if [ ! -d "backend/venv" ]; then
    python3 -m venv backend/venv
    echo -e "${GREEN}Created backend/venv${NC}"
else
    echo "Virtual environment already exists"
fi

# Activate and install dependencies
echo -e "\n${YELLOW}[3/6] Installing dependencies...${NC}"
source backend/venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt

# Create necessary directories
echo -e "\n${YELLOW}[4/6] Creating directories...${NC}"
mkdir -p backend/data/classifier
mkdir -p data/ddxplus

# Check for DDXPlus data
echo -e "\n${YELLOW}[5/6] Checking DDXPlus data...${NC}"
if [ ! -f "data/ddxplus/release_evidences.json" ]; then
    echo -e "${RED}DDXPlus data not found!${NC}"
    echo "Please ensure these files exist in data/ddxplus/:"
    echo "  - release_evidences.json"
    echo "  - release_conditions.json"
    echo ""
    echo "You can download DDXPlus from: https://github.com/mila-iqia/ddxplus"
    exit 1
else
    echo -e "${GREEN}DDXPlus data found${NC}"
fi

# Train models
echo -e "\n${YELLOW}[6/6] Training models...${NC}"
cd backend
python -m scripts.train_models
cd ..

echo -e "\n${GREEN}========================================"
echo "Setup complete!"
echo "========================================"
echo -e "${NC}"
echo "To start the API server:"
echo "  source backend/venv/bin/activate"
echo "  cd backend"
echo "  uvicorn app.main:app --reload"
echo ""
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
