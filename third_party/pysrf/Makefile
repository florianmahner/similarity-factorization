.PHONY: install dev test clean compile lint format help

help:
	@echo "pysrf development commands:"
	@echo "  make install      - Install package with poetry"
	@echo "  make dev          - Install with dev dependencies"
	@echo "  make compile      - Compile Cython extensions"
	@echo "  make test         - Run test suite"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Run linters (ruff)"
	@echo "  make format       - Format code (black)"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make build        - Build distribution package"
	@echo "  make docs         - Build documentation"
	@echo "  make docs-serve   - Serve documentation locally"
	@echo "  make docs-deploy  - Deploy docs to GitHub Pages"

install:
	poetry install --only main

dev:
	poetry install
	poetry run pysrf-compile

compile:
	poetry run pysrf-compile

test:
	poetry run pytest tests/ -v

test-cov:
	poetry run pytest tests/ -v --cov=pysrf --cov-report=html --cov-report=term

lint:
	poetry run ruff check pysrf/ tests/
	poetry run black --check pysrf/ tests/

format:
	poetry run black pysrf/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .coverage htmlcov/
	rm -rf pysrf/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete
	find . -name "*.so" -delete
	find . -name "*.cpp" -delete

build: clean
	poetry build

docs:
	poetry run mkdocs build

docs-serve:
	poetry run mkdocs serve

docs-deploy:
	poetry run mkdocs gh-deploy --force

all: dev test

