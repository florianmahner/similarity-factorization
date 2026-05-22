# Installation

!!! warning "Cython compilation"
    Compiling the Cython extension gives a 10-50x speedup. Without it, pysrf
    falls back to a pure Python implementation. Make sure you have a C compiler
    installed before proceeding.

## Quick install

```bash
git clone https://github.com/florianmahner/pysrf.git
cd pysrf
pip install .
```

This builds the Cython extension automatically using meson-python.

## Developer setup

For development with an editable install:

```bash
git clone https://github.com/florianmahner/pysrf.git
cd pysrf
make dev
```

This installs all dependencies (including build tools) via Poetry and
compiles the Cython extension into `build/`, keeping the source tree clean.

### Makefile targets

- `make dev`: install dependencies and compile Cython (editable install).
- `make install`: install the package (non-editable).
- `make test`: run the test suite.
- `make format`: format code with ruff.
- `make clean`: remove build artifacts.
- `make docs`: build documentation.

### Manual steps

If you prefer not to use Make:

```bash
poetry install --no-root --all-extras
poetry run pip install -e . --no-build-isolation
```

## Verify the installation

```python
import pysrf
print(pysrf.__version__)

# Check whether the Cython extension is active
from pysrf.model import _w_solver_backend
print(f"Backend: {_w_solver_backend}")  # 'cython' if compiled
```

## Performance

### Multi-threaded BLAS

SRF uses BLAS routines (via scipy) that benefit from multi-threading.
Set `OMP_NUM_THREADS` **before** importing pysrf to enable parallel
linear algebra:

```bash
export OMP_NUM_THREADS=4
python my_script.py
```

Or at the top of a script, before any imports:

```python
import os
os.environ["OMP_NUM_THREADS"] = "4"

from pysrf import SRF
```

When `verbose=1` is set, SRF logs the current thread count at fit time.
