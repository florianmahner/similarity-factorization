.PHONY: install dev test test-cov lint format clean docs docs-serve

install:
	pip install .

dev:
	poetry install --no-root --all-extras
	poetry run pip install -e . --no-build-isolation

test:
	poetry run pytest tests/ -v

test-cov:
	poetry run pytest tests/ -v --cov=pysrf --cov-report=html --cov-report=term

lint:
	poetry run ruff check pysrf/ tests/

format:
	poetry run ruff format pysrf/ tests/

clean:
	rm -rf build/ builddir/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -name "__pycache__" -type d -exec rm -rf {} +

docs:
	poetry run zensical build

docs-serve:
	poetry run zensical serve
