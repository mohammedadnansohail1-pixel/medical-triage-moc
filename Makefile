.PHONY: setup install train run test clean help

# Variables
PYTHON := python3
VENV := backend/venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# Colors
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

help:
	@echo "Medical Triage AI - Available Commands"
	@echo "======================================="
	@echo "  make setup     - Complete first-time setup (venv + deps + train)"
	@echo "  make install   - Install dependencies only"
	@echo "  make train     - Train ML models from scratch"
	@echo "  make run       - Start the API server"
	@echo "  make test      - Run tests"
	@echo "  make evaluate  - Run evaluation metrics"
	@echo "  make clean     - Remove generated files"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make setup"
	@echo "  2. make run"
	@echo "  3. Open http://localhost:8000/docs"

setup: venv install train
	@echo "$(GREEN)Setup complete!$(NC)"
	@echo "Run 'make run' to start the server"

venv:
	@echo "$(YELLOW)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)

install: venv
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements.txt

train:
	@echo "$(YELLOW)Training models...$(NC)"
	cd backend && $(PYTHON_VENV) -m scripts.train_models

run:
	@echo "$(YELLOW)Starting API server...$(NC)"
	@echo "API docs: http://localhost:8000/docs"
	cd backend && $(PYTHON_VENV) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	@echo "$(YELLOW)Running tests...$(NC)"
	cd backend && $(PYTHON_VENV) -m pytest tests/ -v

evaluate:
	@echo "$(YELLOW)Running evaluation...$(NC)"
	cd backend && $(PYTHON_VENV) -c "\
from pathlib import Path; \
import sys; sys.path.insert(0, '.'); \
from app.core.triage_pipeline_v2 import TriagePipelineV2; \
p = TriagePipelineV2(); \
p.load(Path('../data/ddxplus/release_evidences.json'), Path('data/classifier/model.pkl'), Path('data/classifier/vocabulary.pkl'), enable_explanations=False); \
result = p.predict(['chest pain', 'shortness of breath']); \
print('Test prediction:', result['specialty'], result['confidence']); \
p.unload()"

clean:
	@echo "$(YELLOW)Cleaning generated files...$(NC)"
	rm -rf backend/data/classifier/*.pkl
	rm -rf backend/__pycache__
	rm -rf backend/app/__pycache__
	rm -rf backend/app/core/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Clean complete$(NC)"

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
