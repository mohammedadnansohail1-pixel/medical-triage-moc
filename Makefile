.PHONY: setup start stop logs status clean health

# First-time setup
setup:
	@echo "🔧 Setting up Medical Triage MOC..."
	cp -n .env.example .env || true
	docker compose pull
	docker compose build
	@echo "⬇️  Pulling Ollama model..."
	docker compose up -d ollama
	@sleep 15
	docker exec triage-ollama ollama pull mistral:7b-instruct-q8_0
	docker compose down
	@echo "✅ Setup complete! Run 'make start' to begin."

# Start all services
start:
	@echo "🚀 Starting Medical Triage MOC..."
	docker compose up -d
	@echo "✅ Services starting..."
	@echo "   Frontend: http://localhost:3000"
	@echo "   Backend:  http://localhost:8000/docs"
	@echo "   Neo4j:    http://localhost:7474"

# Stop all services (frees resources)
stop:
	@echo "🛑 Stopping all services..."
	docker compose down
	@echo "✅ All resources freed"

# View logs
logs:
	docker compose logs -f

# Check status
status:
	@docker compose ps
	@echo ""
	@docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null || true

# Health check
health:
	@echo "🏥 Health Check:"
	@curl -sf http://localhost:8000/health && echo "Backend: ✅" || echo "Backend: ❌"
	@curl -sf http://localhost:11434/api/tags > /dev/null && echo "Ollama: ✅" || echo "Ollama: ❌"
	@curl -sf http://localhost:7474 > /dev/null && echo "Neo4j: ✅" || echo "Neo4j: ❌"

# Clean everything
clean:
	@echo "⚠️  This will DELETE ALL DATA!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	docker compose down -v
	@echo "✅ Cleaned"
