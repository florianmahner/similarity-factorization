# Development Guide

## Setup Development Environment

### Using the Setup Script

The easiest way to set up the development environment:

```bash
chmod +x setup.sh
./setup.sh
```

This script will:
1. Install `pyenv` (if not present)
2. Install Python 3.12.4
3. Install Poetry
4. Install all dependencies
5. Compile Cython extensions
6. Run tests

### Manual Setup

```bash
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Compile Cython extensions
poetry run pysrf-compile

# Run tests
poetry run pytest
```

## Makefile Commands

The project includes a Makefile for common development tasks:

```bash
make help          # Show all available commands
make dev           # Install with dev dependencies + compile Cython
make compile       # Compile Cython extensions
make test          # Run test suite
make test-cov      # Run tests with coverage report
make lint          # Run linters (ruff + black)
make format        # Format code with black
make clean         # Remove build artifacts
make build         # Build distribution package
make docs          # Build documentation
make docs-serve    # Serve documentation locally
```

## Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
poetry run pytest tests/test_model.py -v

# Run specific test
poetry run pytest tests/test_model.py::test_srf_fit_complete_data -v
```

## Code Quality

### Linting

```bash
# Check code quality
make lint

# Auto-format code
make format
```

### Type Checking

The codebase uses Python 3.10+ type hints:

```python
from __future__ import annotations

def my_function(x: np.ndarray, rank: int = 10) -> tuple[np.ndarray, float]:
    ...
```

## Cython Extensions

The performance-critical inner loop is implemented in Cython (`_bsum.pyx`):

```bash
# Compile Cython extensions
make compile

# Or directly
poetry run pysrf-compile
```

### Testing Cython vs Python

```python
from pysrf.model import _get_update_w_function

update_w = _get_update_w_function()
print(f"Using: {update_w.__module__}")
# Cython: _bsum
# Python fallback: pysrf.model
```

## Documentation

### Building Docs

```bash
# Build documentation
make docs

# Serve locally at http://127.0.0.1:8000
make docs-serve
```

### Writing Docstrings

Use NumPy-style docstrings:

```python
def my_function(x: np.ndarray, param: int = 10) -> float:
    """
    Brief description.

    Longer description explaining the function's purpose and behavior.

    Parameters
    ----------
    x : ndarray
        Description of x
    param : int, default=10
        Description of param

    Returns
    -------
    result : float
        Description of return value

    Examples
    --------
    >>> result = my_function(np.array([1, 2, 3]))
    >>> print(result)
    0.123
    """
    ...
```

## Contributing

### Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `make test`
5. Format code: `make format`
6. Commit: `git commit -m "Add feature"`
7. Push: `git push origin feature-name`
8. Open a Pull Request

### Guidelines

- Write tests for new features
- Maintain type hints
- Update documentation
- Keep changes focused
- Follow existing code style

## Publishing

### PyPI Release

```bash
# Update version in pyproject.toml
poetry version patch  # or minor, major

# Build package
make build

# Publish to PyPI
poetry publish
```

### Development Release

```bash
# Build with dev version
poetry version prerelease

# Publish to TestPyPI
poetry publish -r testpypi
```

## Project Structure

```
pysrf/
├── pysrf/              # Main package
│   ├── __init__.py    # Public API
│   ├── model.py       # SRF class
│   ├── cross_validation.py
│   ├── bounds.py      # Sampling bound estimation
│   ├── utils.py       # Helper functions
│   └── _bsum.pyx      # Cython extension
├── tests/             # Test suite
├── docs/              # Documentation
├── build.py           # Cython build script
├── Makefile           # Development commands
├── pyproject.toml     # Poetry config
└── README.md          # Project overview
```

