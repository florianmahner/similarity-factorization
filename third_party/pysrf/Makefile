.PHONY: install dev test test-cov lint format clean compile docs docs-serve

install:
	poetry install --only main

dev:
	poetry install
	$(MAKE) compile

compile:
	poetry run python setup.py build_ext --inplace

test:
	poetry run pytest tests/ -v

test-cov:
	poetry run pytest tests/ -v --cov=pysrf --cov-report=html --cov-report=term

lint:
	poetry run ruff check pysrf/ tests/

format:
	poetry run ruff format pysrf/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov/
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.so" -delete
	find pysrf/ -name "*.c" -not -name "*.cfg" -delete
	find pysrf/ -name "*.cpp" -delete

docs:
	poetry run zensical build

docs-serve:
	poetry run zensical serve
