# pysrf

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Symmetric non-negative matrix factorization using ADMM optimization. Handles missing data, supports bounded constraints, and includes rank estimation via Random Matrix Theory.

## Installation

### From Source

```bash
git clone https://github.com/fmahner/pysrf.git
cd pysrf
poetry install
make compile  # Strongly recommended for performance (10-50x speedup)
```

### As Git Subtree (for development integration)

```bash
# Add as subtree in your project
git subtree add --prefix=pysrf https://github.com/fmahner/pysrf.git main --squash

# Update subtree
git subtree pull --prefix=pysrf https://github.com/fmahner/pysrf.git main --squash

# Install from subtree
cd pysrf && poetry install && make compile
```

### From PyPI (when released)

```bash
# Stable release
pip install pysrf

# Development version
pip install --pre pysrf
```

> **⚠️ Performance Note:** Cython compilation is **critical for speed**. Without it, a pure Python fallback is used but will be 10-50x slower. Requires `g++` compiler.

## Quick Start

```python
import numpy as np
from pysrf import SRF

# Generate data
n, rank = 100, 10
w_true = np.random.rand(n, rank)
s = w_true @ w_true.T

# Fit model
model = SRF(rank=10, random_state=42)
w = model.fit_transform(s)
s_hat = model.reconstruct()
```

### With Missing Data

```python
s[mask] = np.nan  # Mark missing entries
model = SRF(rank=10, missing_values=np.nan)
w = model.fit_transform(s)
```

### Cross-Validation & Rank Selection

```python
from pysrf import cross_val_score, estimate_sampling_bounds_fast

# Estimate optimal sampling rate
pmin, pmax, _ = estimate_sampling_bounds_fast(s, n_jobs=-1)
sampling_rate = 0.5 * (pmin + pmax)

# Find best rank
grid = cross_val_score(
    s,
    param_grid={'rank': [5, 10, 15, 20]},
    sampling_fraction=sampling_rate,
    n_repeats=5,
    n_jobs=-1
)
print(f"Best rank: {grid.best_params_['rank']}")
```

## Features

- **ADMM optimization** with missing data support
- **Cython-optimized** inner loop (10-50x faster than pure Python)
- **Cross-validation** for hyperparameter tuning
- **Rank estimation** via Random Matrix Theory (sampling bounds)
- **Scikit-learn compatible** API
- Full type hints (Python 3.10+)

## Algorithm

Solves symmetric NMF via ADMM:
```
min_{W≥0, V} ||M ⊙ (S - V)||²_F + ρ/2 ||V - WW^T||²_F
```

Based on: Shi et al. (2016) "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization"

## Development

```bash
make dev           # Setup environment
make test          # Run tests (32 tests)
make compile       # Compile Cython
make format        # Format code
```

### Publishing to PyPI

```bash
# Update version in pyproject.toml, then:
poetry build                    # Build package
poetry publish                  # Release stable version

# For development releases (e.g., 0.1.0a1, 0.1.0b2)
# Set version in pyproject.toml to "0.1.0a1" then:
poetry build && poetry publish  # Users install with: pip install --pre pysrf
```

## License

MIT
