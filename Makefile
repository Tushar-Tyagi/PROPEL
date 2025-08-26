# Makefile for LLM Position Bias Analysis Framework
# Provides common development tasks and shortcuts

.PHONY: help install install-dev test lint format clean docs run-example setup-env

# Default target
help:
	@echo "🎯 LLM Position Bias Analysis Framework - Development Commands"
	@echo "================================================================"
	@echo ""
	@echo "📦 Installation:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  setup-env      Set up environment configuration"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  test           Run test suite"
	@echo "  test-cov       Run tests with coverage"
	@echo "  lint           Run code linting (flake8)"
	@echo "  format         Format code (black, isort)"
	@echo "  check          Run all quality checks"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  docs           Build documentation"
	@echo "  docs-serve     Serve documentation locally"
	@echo ""
	@echo "🔄 Development:"
	@echo "  run-example    Run basic usage example"
	@echo "  clean          Clean generated files"
	@echo "  dist           Build distribution package"
	@echo ""
	@echo "🔧 Utilities:"
	@echo "  check-deps     Check dependency conflicts"
	@echo "  update-deps    Update dependencies"
	@echo "  freeze-deps    Freeze current dependency versions"

# Installation
install:
	@echo "📦 Installing production dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "🔧 Installing development dependencies..."
	pip install -r requirements.txt
	pip install -e .[dev]

setup-env:
	@echo "🔑 Setting up environment configuration..."
	@if [ ! -f .env ]; then \
		cp env.example .env; \
		echo "✅ Created .env file from env.example"; \
		echo "⚠️  Please edit .env with your API keys"; \
	else \
		echo "✅ .env file already exists"; \
	fi

# Testing
test:
	@echo "🧪 Running test suite..."
	python -m pytest tests/ -v

test-cov:
	@echo "📊 Running tests with coverage..."
	python -m pytest tests/ --cov=LLM_debias --cov-report=html --cov-report=term

# Code Quality
lint:
	@echo "🔍 Running code linting..."
	flake8 LLM_debias.py tests/ examples/ --max-line-length=100 --ignore=E203,W503

format:
	@echo "🎨 Formatting code..."
	black LLM_debias.py tests/ examples/ --line-length=100
	isort LLM_debias.py tests/ examples/ --profile=black

check: lint format test
	@echo "✅ All quality checks passed!"

# Documentation
docs:
	@echo "📚 Building documentation..."
	@if command -v sphinx-build >/dev/null 2>&1; then \
		mkdir -p docs/_build; \
		sphinx-build -b html docs docs/_build/html; \
		echo "✅ Documentation built in docs/_build/html/"; \
	else \
		echo "❌ sphinx-build not found. Install with: pip install sphinx sphinx-rtd-theme"; \
	fi

docs-serve:
	@echo "🌐 Serving documentation locally..."
	@if [ -d "docs/_build/html" ]; then \
		cd docs/_build/html && python -m http.server 8000; \
	else \
		echo "❌ Documentation not built. Run 'make docs' first."; \
	fi

# Development
run-example:
	@echo "🚀 Running basic usage example..."
	python examples/basic_usage.py

clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf docs/_build/
	@echo "✅ Cleanup complete!"

dist: clean
	@echo "📦 Building distribution package..."
	python setup.py sdist bdist_wheel

# Utilities
check-deps:
	@echo "🔍 Checking for dependency conflicts..."
	pip check

update-deps:
	@echo "🔄 Updating dependencies..."
	pip install --upgrade -r requirements.txt

freeze-deps:
	@echo "📌 Freezing current dependency versions..."
	pip freeze > requirements-frozen.txt
	@echo "✅ Dependencies frozen to requirements-frozen.txt"

# Quick start for new developers
quickstart: setup-env install-dev
	@echo ""
	@echo "🚀 Quick start completed!"
	@echo "Next steps:"
	@echo "1. Edit .env with your API keys"
	@echo "2. Run 'make test' to verify installation"
	@echo "3. Run 'make run-example' to see the framework in action"
	@echo "4. Check README.md for detailed usage instructions"

# Development workflow
dev: install-dev setup-env
	@echo "🔧 Development environment ready!"
	@echo "Run 'make check' before committing changes"

# CI/CD pipeline
ci: install-dev lint format test
	@echo "✅ CI pipeline completed successfully!"

# Help for specific targets
install-help:
	@echo "Installation Options:"
	@echo "  make install      - Install production dependencies only"
	@echo "  make install-dev  - Install development dependencies (recommended)"
	@echo "  make quickstart   - Complete setup for new developers"

test-help:
	@echo "Testing Options:"
	@echo "  make test         - Run basic test suite"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make check        - Run all quality checks (lint + format + test)"

# Show current environment status
status:
	@echo "🔍 Environment Status:"
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@if [ -f .env ]; then echo "✅ .env file exists"; else echo "❌ .env file missing"; fi
	@if [ -d "venv" ] || [ -d ".venv" ]; then echo "✅ Virtual environment detected"; else echo "⚠️  No virtual environment detected"; fi
	@echo "Installed packages:"
	@pip list --format=columns | grep -E "(pandas|numpy|openai|pytest)" || echo "   No key packages found"
