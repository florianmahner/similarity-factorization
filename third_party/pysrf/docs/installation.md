# Installation

!!! warning "Cython Compilation"
    For optimal performance (10-50x speedup), ensure Cython extensions are compiled during installation. You may need development tools installed.

## Automated Setup (Recommended)

The easiest way to set up the complete development environment:

```bash
git clone https://github.com/fmahner/pysrf.git
cd pysrf
./setup.sh
```

This script will:
1. Check for and install `pyenv` (if missing)
2. Install `poetry` (if missing)
3. Install Python 3.12.4 via `pyenv`
4. Set the local Python version
5. Install all dependencies via `poetry`
6. Compile Cython extensions for 10-50x speedup
7. Run the test suite

Activate the environment with `poetry shell`.

## Manual Installation

If you prefer manual setup or need more control:

### Step 1: Install Prerequisites

```bash
# Install pyenv (if not already installed)
curl https://pyenv.run | bash

# Install poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -
```

### Step 2: Set Up Python Environment

```bash
# Install Python 3.12.4 (or your preferred version >=3.10)
pyenv install 3.12.4
pyenv local 3.12.4
```

### Step 3: Install Dependencies

```bash
# Install dependencies with poetry
poetry install
```

### Step 4: Compile Cython Extensions

Cython compilation is critical for performance (10-50x speedup). Without it, a pure Python fallback is used.

```bash
# Compile via poetry script
poetry run pysrf-compile

# Or use Makefile (includes additional options)
make compile
```

The Makefile also provides other useful commands:
- `make dev` - Install with dev dependencies and compile
- `make test` - Run test suite
- `make format` - Format code
- `make clean` - Remove build artifacts
- `make docs` - Build documentation

Run `make help` for all available commands.

## Alternative Installation Methods

### From PyPI (Future)

Once published to PyPI:

```bash
# Stable release
pip install pysrf

# Development version
pip install --pre pysrf
```

### As Git Subtree (For Development Integration)

```bash
# Add as subtree in your project
git subtree add --prefix=pysrf https://github.com/fmahner/pysrf.git master --squash

# Update subtree
git subtree pull --prefix=pysrf https://github.com/fmahner/pysrf.git master --squash

# Install from subtree
cd pysrf && poetry install && make compile
```

## Verify Installation

```python
import pysrf
print(pysrf.__version__)

# Check Cython compilation
from pysrf.model import _get_update_w_function
update_w = _get_update_w_function()
print(f"Using: {update_w.__module__}")  # Should show _bsum if compiled
```
