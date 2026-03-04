# Installation

!!! warning "Cython compilation"
    Compiling the Cython extension gives a 10-50x speedup. Without it, pysrf
    falls back to a pure Python implementation. Make sure you have a C compiler
    installed before proceeding.

## Automated setup (recommended)

Clone the repository and run the setup script:

```bash
git clone https://github.com/fmahner/pysrf.git
cd pysrf
./setup.sh
```

The script performs the following steps:

1. Installs `pyenv` if it is not already present.
2. Installs `poetry` if it is not already present.
3. Installs Python 3.12.4 through `pyenv`.
4. Sets the local Python version.
5. Installs all dependencies through `poetry`.
6. Compiles the Cython extension.
7. Runs the test suite.

After the script finishes, activate the environment:

```bash
poetry shell
```

## Manual installation

### Install prerequisites

Install pyenv and poetry if you do not have them:

```bash
curl https://pyenv.run | bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Set up python

Install Python 3.12.4 (or any version >=3.10) and pin it for this project:

```bash
pyenv install 3.12.4
pyenv local 3.12.4
```

### Install dependencies

```bash
poetry install
```

### Compile the Cython extension

Use the Makefile target or run the build command directly:

```bash
make compile

# or
poetry run python setup.py build_ext --inplace
```

The Makefile provides additional targets:

- `make dev`: install dev dependencies and compile Cython.
- `make test`: run the test suite.
- `make format`: format code with ruff.
- `make clean`: remove build artifacts.
- `make docs`: build documentation.

## Alternative methods

### From PyPI (planned)

```bash
pip install pysrf

# development version
pip install --pre pysrf
```

### As a git subtree

Add pysrf as a subtree inside another project:

```bash
git subtree add --prefix=pysrf https://github.com/fmahner/pysrf.git master --squash
```

Update the subtree:

```bash
git subtree pull --prefix=pysrf https://github.com/fmahner/pysrf.git master --squash
```

Then install and compile:

```bash
cd pysrf && poetry install && make compile
```

## Verify the installation

```python
import pysrf
print(pysrf.__version__)

# Check whether the Cython extension is active
from pysrf.model import _get_update_w_function
update_w = _get_update_w_function()
print(f"Using: {update_w.__module__}")  # _bsum if compiled
```
