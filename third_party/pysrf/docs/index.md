# pysrf

**Similarity-Based Representation Factorization (SRF)** 

## Overview

`pysrf` implements symmetric non-negative matrix factorization using the Alternating Direction Method of Multipliers (SRF). 

## Installation

See the [Installation Guide](installation.md) for detailed instructions.

## Quick Example

```python
import numpy as np
from pysrf import SRF

# Your similarity matrix
s = np.random.rand(100, 100)
s = (s + s.T) / 2  # make symmetric

# Fit model
model = SRF(rank=10, max_outer=20, random_state=42)
embedding = model.fit_transform(s)

# Reconstruct
s_reconstructed = model.reconstruct()
```

## Next Steps

- [Installation Guide](installation.md) - Setup instructions
- [Examples](examples.md) - Usage examples and workflows
- [Quick Start Guide](quickstart.md) - Detailed tutorials
- [API Reference](api/model.md) - Complete API documentation
- [Development Guide](development.md) - Contributing guidelines
