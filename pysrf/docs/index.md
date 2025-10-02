# pysrf

**Symmetric Representation Factorization (SRF)** - Fast matrix completion and rank estimation using ADMM optimization.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

`pysrf` implements symmetric non-negative matrix factorization using the Alternating Direction Method of Multipliers (ADMM). It's designed for:

- **Matrix completion** with missing entries
- **Rank estimation** via Random Matrix Theory
- **Cross-validation** for hyperparameter tuning
- **High performance** through Cython optimization (10-50x speedup)

## Key Features

- ✅ **ADMM optimization** with missing data support
- ✅ **Cython-optimized** inner loop for speed
- ✅ **Automatic sampling bound estimation** for cross-validation
- ✅ **Scikit-learn compatible** API
- ✅ Full type hints (Python 3.10+)

## Quick Example

```python
import numpy as np
from pysrf import SRF

# Your similarity matrix
s = np.random.rand(100, 100)
s = (s + s.T) / 2  # make symmetric

# Fit model
model = SRF(rank=10, max_outer=20, random_state=42)
w = model.fit_transform(s)

# Reconstruct
s_reconstructed = w @ w.T
```

## Installation

See the [Installation Guide](installation.md) for detailed instructions.

## Next Steps

- [Quick Start Guide](quickstart.md) - Learn the basics
- [API Reference](api/model.md) - Detailed documentation
- [Development Guide](development.md) - Contributing guidelines

