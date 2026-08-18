# Makefile for PROPEL (PROpensity-based-Position-bias-Elimination-for-LLMs)
# Provides common development tasks and shortcuts

.PHONY: help install install-dev test test-cov lint format clean run-example setup-env

# Default target
help:
	@echo "🎯 PROPEL - Development Commands"
	@echo "================================="
	@echo ""
	@echo "📦 Installation:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install in editable mode with development dependencies"
	@echo "  setup-env      Set up environment configuration"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  test           Run test suite"
	@echo "  test-cov       Run tests with coverage"
	@echo "  lint           Run code linting (flake8)"
	@echo "  format         Format code (black, isort)"
	@echo "  check          Run all quality checks"
	@echo ""
	@echo "🔄 Development:"
	@echo "  run-example    Run quickstart example"
	@echo "  clean          Clean generated files"
	@echo "  dist           Build distribution package"

# Installation
install:
	@echo "📦 Installing production dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "🔧 Installing development dependencies..."
	pip install -e ".[dev]"

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
	python -m pytest tests/ --cov=propel --cov-report=term

# Code Quality
lint:
	@echo "🔍 Running code linting..."
	flake8 propel/ tests/ examples/ --max-line-length=120 --ignore=E203,W503

format:
	@echo "🎨 Formatting code..."
	black propel/ tests/ examples/ --line-length=120
	isort propel/ tests/ examples/ --profile=black

check: lint format test
	@echo "✅ All quality checks passed!"

# Development
run-example:
	@echo "🚀 Running quickstart example..."
	python examples/quickstart.py

clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf __pycache__/ propel/__pycache__/ tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	@echo "✅ Cleanup complete!"

dist: clean
	@echo "📦 Building distribution package..."
	python -m build
