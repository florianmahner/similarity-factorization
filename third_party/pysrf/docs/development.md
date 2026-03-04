# Development

## Set up the development environment

### Automated setup

Run the setup script to install all tools and dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

The script installs pyenv, Python 3.12.4, Poetry, all dependencies, compiles
the Cython extension, and runs the test suite.

### Manual setup

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
make compile
poetry run pytest
```

## Makefile targets

The Makefile provides shortcuts for common tasks:

```bash
make dev           # install dev dependencies and compile Cython
make compile       # compile Cython extension
make test          # run test suite
make test-cov      # run tests with coverage report
make lint          # run ruff linter
make format        # format code with ruff
make clean         # remove build artifacts
make docs          # build documentation with zensical
make docs-serve    # serve documentation locally with zensical
```

## Run tests

Run the full suite:

```bash
make test
```

Run tests with coverage:

```bash
make test-cov
```

Run a specific test file or test function:

```bash
poetry run pytest tests/test_model.py -v
poetry run pytest tests/test_model.py::test_srf_fit_complete_data -v
```

## Code quality

### Lint and format

Check code quality and auto-format:

```bash
make lint
make format
```

### Type hints

Use Python 3.10+ style type hints in all public functions:

```python
from __future__ import annotations

def my_function(x: np.ndarray, rank: int = 10) -> tuple[np.ndarray, float]:
    ...
```

## Cython extension

The performance-critical inner loop lives in `_bsum.pyx`. Compile it with:

```bash
make compile

# or
poetry run python setup.py build_ext --inplace
```

Verify which implementation is active:

```python
from pysrf.model import _get_update_w_function

update_w = _get_update_w_function()
print(f"Using: {update_w.__module__}")
# _bsum    → Cython (fast)
# pysrf.model → pure Python fallback
```

## Documentation

### Build and preview

pysrf uses [zensical](https://zensical.org) for documentation. Build and
preview locally:

```bash
make docs          # build docs
make docs-serve    # serve locally at http://127.0.0.1:8000
```

### Docstring format

Use NumPy-style docstrings:

```python
def my_function(x: np.ndarray, param: int = 10) -> float:
    """
    Brief description.

    Longer description explaining the function's purpose and behavior.

    Parameters
    ----------
    x : ndarray
        Description of x.
    param : int, default=10
        Description of param.

    Returns
    -------
    result : float
        Description of return value.

    Examples
    --------
    >>> result = my_function(np.array([1, 2, 3]))
    >>> print(result)
    0.123
    """
    ...
```

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes.
4. Run tests: `make test`
5. Format code: `make format`
6. Commit: `git commit -m "Add feature"`
7. Push: `git push origin feature-name`
8. Open a pull request.

### Guidelines

- Write tests for new features.
- Maintain type hints.
- Update documentation.
- Keep changes focused.
- Follow existing code style.

## Publish to PyPI

Update the version in `pyproject.toml`, then build and publish:

```bash
poetry version patch  # or minor, major
make build
poetry publish
```

For a pre-release version:

```bash
poetry version prerelease
poetry publish -r testpypi
```

## Project structure

```
pysrf/
├── pysrf/              # main package
│   ├── __init__.py     # public API
│   ├── model.py        # SRF class
│   ├── cross_validation.py
│   ├── bounds.py       # sampling-bound estimation
│   ├── utils.py        # helper functions
│   └── _bsum.pyx       # Cython extension
├── tests/              # test suite
├── docs/               # documentation
├── setup.py            # Cython build (setuptools)
├── Makefile            # development targets
├── pyproject.toml      # project config
└── README.md           # project overview
```
