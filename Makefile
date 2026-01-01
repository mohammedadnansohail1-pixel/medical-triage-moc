.PHONY: setup start stop logs status clean health dev-backend dev-frontend

# ===================
# MAIN COMMANDS
# ===================

# First-time setup
setup:
	@echo "🔧 Setting up Medical Triage MOC..."
	@test -f .env || cp .env.example .env
	docker compose pull
	docker compose build
	@echo "⬇️  Pulling Ollama model (this takes a few minutes)..."
	docker compose up -d ollama
	@sleep 15
	docker exec triage-ollama ollama pull mistral:7b-instruct-q8_0
	docker compose down
	@echo "✅ Setup complete! Run 'make start' to begin."

# Start all services
start:
	@echo "🚀 Starting Medical Triage MOC..."
	docker compose up -d
	@echo ""
	@echo "✅ Services starting..."
	@echo "   Frontend: http://localhost:3000"
	@echo "   Backend:  http://localhost:8000/docs"
	@echo "   Neo4j:    http://localhost:7474"

# Stop all services (frees all resources)
stop:
	@echo "🛑 Stopping all services..."
	docker compose down
	@echo "✅ All resources freed"

# ===================
# MONITORING
# ===================

# View logs (all services)
logs:
	docker compose logs -f

# View specific service logs
logs-backend:
	docker compose logs -f backend

logs-ollama:
	docker compose logs -f ollama

# Check status and resource usage
status:
	@echo "📊 Service Status:"
	@docker compose ps
	@echo ""
	@echo "📊 Resource Usage:"
	@docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || true

# Health check
health:
	@echo "🏥 Health Check:"
	@curl -sf http://localhost:8000/health 2>/dev/null && echo "Backend: ✅" || echo "Backend: ❌"
	@curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && echo "Ollama:  ✅" || echo "Ollama:  ❌"
	@curl -sf http://localhost:7474 >/dev/null 2>&1 && echo "Neo4j:   ✅" || echo "Neo4j:   ❌"
	@curl -sf http://localhost:5432 >/dev/null 2>&1 || echo "Postgres: (check via psql)"

# ===================
# DEVELOPMENT
# ===================

# Run backend tests
test:
	cd backend && python -m pytest tests/ -v

# Format code
format:
	cd backend && python -m black app/ tests/

# Lint code
lint:
	cd backend && python -m ruff check app/ tests/

# ===================
# CLEANUP
# ===================

# Stop and remove volumes (DELETES DATA)
clean:
	@echo "⚠️  This will DELETE ALL DATA!"
	@read -p "Are you sure? [y/N] " confirm && [ "$${confirm}" = "y" ] || exit 1
	docker compose down -v
	@echo "✅ Cleaned"
