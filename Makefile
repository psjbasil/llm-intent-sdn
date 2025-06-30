# Makefile for LLM Intent-based SDN system
# Provides convenient commands for development and deployment

.PHONY: help setup install clean dev test lint format run run-ryu run-mininet run-all examples docs docker

# Default target
help:
	@echo "🚀 LLM Intent-based SDN System - Available Commands:"
	@echo ""
	@echo "📦 Setup and Installation:"
	@echo "  make setup      - Setup development environment"
	@echo "  make install    - Install dependencies"
	@echo "  make clean      - Clean temporary files"
	@echo ""
	@echo "🔧 Development:"
	@echo "  make dev        - Start development server"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code"
	@echo ""
	@echo "🏃 Running Services:"
	@echo "  make run        - Start API server"
	@echo "  make run-ryu    - Start RYU controller"
	@echo "  make run-mininet - Start Mininet topology (requires sudo)"
	@echo "  make run-all    - Start all services"
	@echo ""
	@echo "📚 Examples and Documentation:"
	@echo "  make examples   - Run intent examples"
	@echo "  make docs       - Generate documentation"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker     - Build and run with Docker"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test-system - Run system tests"
	@echo "  make test-api   - Test API endpoints"

# Setup development environment
setup:
	@echo "🔧 Setting up development environment..."
	python setup.py

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -e .
	pip install -e .[dev]

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Start development server with auto-reload
dev:
	@echo "🔥 Starting development server..."
	uvicorn llm_intent_sdn.api.main:app --host 0.0.0.0 --port 8000 --reload

# Run tests
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

# Run system tests
test-system:
	@echo "🔍 Running system tests..."
	python scripts/test_system.py

# Test API endpoints
test-api:
	@echo "📡 Testing API endpoints..."
	curl -s http://localhost:8000/health | python -m json.tool
	curl -s http://localhost:8000/api/v1/network/topology | python -m json.tool

# Run code linting
lint:
	@echo "🔍 Running code linting..."
	ruff check src/
	mypy src/

# Format code
format:
	@echo "✨ Formatting code..."
	ruff format src/
	ruff check --fix src/

# Start API server
run:
	@echo "🚀 Starting API server..."
	python run.py

# Start RYU controller
run-ryu:
	@echo "🎛️  Starting RYU controller..."
	python scripts/start_ryu.py

# Start Mininet topology (requires sudo)
run-mininet:
	@echo "🌐 Starting Mininet topology..."
	@echo "⚠️  This requires sudo privileges..."
	sudo python scripts/start_mininet.py

# Start all services (in background)
run-all:
	@echo "🚀 Starting all services..."
	@echo "Starting RYU controller in background..."
	python scripts/start_ryu.py &
	@sleep 3
	@echo "Starting API server in background..."
	python run.py &
	@echo "✅ All services started!"
	@echo "💡 Use 'make stop-all' to stop all services"

# Stop all background services
stop-all:
	@echo "🛑 Stopping all services..."
	pkill -f ryu-manager || true
	pkill -f "python run.py" || true
	pkill -f uvicorn || true
	@echo "✅ All services stopped!"

# Run intent examples
examples:
	@echo "📚 Running intent examples..."
	python examples/intent_examples.py

# Interactive intent testing
examples-interactive:
	@echo "🎯 Starting interactive intent testing..."
	python examples/intent_examples.py --interactive

# Generate documentation
docs:
	@echo "📚 Generating documentation..."
	@mkdir -p docs/api
	@echo "Generating API documentation..."
	curl -s http://localhost:8000/openapi.json > docs/api/openapi.json
	@echo "✅ Documentation generated in docs/"

# Check system status
status:
	@echo "📊 System Status:"
	@echo "=================="
	@echo "🔗 API Server:"
	@curl -s http://localhost:8000/health > /dev/null && echo "  ✅ Running" || echo "  ❌ Not running"
	@echo "🎛️  RYU Controller:"
	@curl -s http://localhost:8080/stats/switches > /dev/null && echo "  ✅ Running" || echo "  ❌ Not running"
	@echo "🌐 Network Topology:"
	@curl -s http://localhost:8000/api/v1/network/topology > /dev/null && echo "  ✅ Available" || echo "  ❌ Not available"

# Environment setup check
check-env:
	@echo "🔍 Checking environment setup..."
	@echo "Python version:"
	@python --version
	@echo "Dependencies:"
	@pip list | grep -E "(fastapi|ryu|uvicorn|httpx)" || echo "  ⚠️  Some dependencies missing"
	@echo "Environment file:"
	@test -f .env && echo "  ✅ .env file exists" || echo "  ❌ .env file missing"

# Create sample network topology file
create-topo:
	@echo "🌐 Creating sample topology file..."
	@mkdir -p data
	@cat > data/sample_topology.json << 'EOF'
	{
	  "switches": [
	    {"dpid": "0000000000000001", "name": "s1"},
	    {"dpid": "0000000000000002", "name": "s2"},
	    {"dpid": "0000000000000003", "name": "s3"}
	  ],
	  "hosts": [
	    {"ip": "10.0.0.1", "mac": "00:00:00:00:00:01", "switch": "s1"},
	    {"ip": "10.0.0.2", "mac": "00:00:00:00:00:02", "switch": "s1"},
	    {"ip": "10.0.0.3", "mac": "00:00:00:00:00:03", "switch": "s2"},
	    {"ip": "10.0.0.4", "mac": "00:00:00:00:00:04", "switch": "s3"}
	  ],
	  "links": [
	    {"src": "s1", "dst": "s2", "bandwidth": 5},
	    {"src": "s2", "dst": "s3", "bandwidth": 5},
	    {"src": "s1", "dst": "s3", "bandwidth": 8}
	  ]
	}
	EOF
	@echo "✅ Sample topology created at data/sample_topology.json"

# Docker build and run
docker:
	@echo "🐳 Building Docker image..."
	docker build -t llm-intent-sdn .
	@echo "🚀 Running Docker container..."
	docker run -p 8000:8000 -v $(PWD)/.env:/app/.env llm-intent-sdn

# Development workflow
dev-workflow: setup format lint test
	@echo "✅ Development workflow completed!"

# Production deployment check
prod-check:
	@echo "🔍 Production deployment check..."
	@echo "Checking required files:"
	@test -f .env && echo "  ✅ .env file" || echo "  ❌ .env file missing"
	@test -f pyproject.toml && echo "  ✅ pyproject.toml" || echo "  ❌ pyproject.toml missing"
	@echo "Checking Python dependencies:"
	@python -c "import fastapi, ryu, uvicorn" && echo "  ✅ Core dependencies" || echo "  ❌ Missing dependencies"
	@echo "Checking configuration:"
	@python -c "from src.llm_intent_sdn.config import settings; print(f'  API: {settings.api_host}:{settings.api_port}'); print(f'  RYU: {settings.ryu_host}:{settings.ryu_port}')"

# Quick start for new users
quickstart:
	@echo "🚀 Quick Start Guide:"
	@echo "===================="
	@echo "1. Setup environment: make setup"
	@echo "2. Configure API key: edit .env file"
	@echo "3. Start RYU:        make run-ryu     (Terminal 1)"
	@echo "4. Start Mininet:    make run-mininet (Terminal 2, sudo required)"
	@echo "5. Start API:        make run         (Terminal 3)"
	@echo "6. Test system:      make test-system"
	@echo "7. Try examples:     make examples"
	@echo ""
	@echo "📚 Visit http://localhost:8000/docs for API documentation" 