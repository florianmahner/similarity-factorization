# Installation

!!! warning "Cython Compilation"
    For optimal performance (10-50x speedup), ensure Cython extensions are compiled during installation. You may need development tools installed.

## From Source

### Using Poetry (Recommended)

```bash
git clone https://github.com/fmahner/pysrf.git
cd pysrf
poetry install
poetry run pysrf-compile  # Compile Cython extensions
```

### Using pip

```bash
git clone https://github.com/fmahner/pysrf.git
cd pysrf
pip install -e .
```

## As a Git Subtree

For development, you can add `pysrf` as a subtree in your project:

```bash
# Add as subtree
git subtree add --prefix=pysrf https://github.com/fmahner/pysrf.git main --squash

# Update later
git subtree pull --prefix=pysrf https://github.com/fmahner/pysrf.git main --squash
```

## From PyPI (coming soon...)

```bash
pip install pysrf
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


