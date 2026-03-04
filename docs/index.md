---
hide:
  - navigation
  - toc
---

# PySRF

**Discover interpretable dimensions from representational similarities.**

Representational similarity is a widely used tool in cognitive science and machine learning. PySRF decomposes similarity matrices in sparse, non-negative dimensions that reveal the latent structure underlying similarities.

Say we have a similarity matrix $S$, PySRF finds a non-negative embedding $W$ such that $S \approx WW^\top$. Each column of $W$ is a dimension, and each row gives for an item a numeric weight alongside each dimensions. Because dimensions are non-negative and sparse (e.g. many entries drive toward zero), the embedding is an additive, compositional representation where dimensions contribute positively without canceling each other out.

## When to use PySRF

PySRF works very broadly on in principle any (symmetric) similarity matix. These can come from different domains, for example:

- **Behavioral data**: any task that yields a measure of similarity between items.
- **Neural data**: representational similarity matrices derived from fMRI, electrophysiology, or other neural recordings.
- **Machine learning**: similarity matrices from deep neural network activations

## Key capabilities

- **Missing data**: real-world similarity matrices are sometimes incomplete and some entries $i,j$ in a similarity matrix not observed. PySRF can handle missing entries naturally.
- **Dimensionality estimation**: the number of optimal dimensions can be estimated via cross-validation
- **Fast solver**: a Cython-accelerated solver provides 10-50x
  speedup over pure Python.

## Quick example

```python
import numpy as np
from pysrf import SRF

# Your similarity matrix (e.g., from behavioral judgments or neural data)
s = np.random.rand(100, 100)
s = (s + s.T) / 2

# Decompose into 10 interpretable dimensions
model = SRF(rank=10, max_outer=20, random_state=42)
w = model.fit_transform(s)

# Reconstruct the similarity matrix from the dimensions
s_reconstructed = model.reconstruct()
```

## Reference

Mahner, F. P.\*, Lam, K. C.\*, & Hebart, M. N. (2025). Interpretable
dimensions from sparse representational similarities. *In preparation*.

## Next steps

- [Installation](installation.md): set up pysrf and compile Cython extensions.
- [Quick start](quickstart.md): walk through the core workflow step by step.
- [Examples](examples.md): explore advanced features and full pipelines.
- [API reference](api/model.md): browse the complete API.
- [Development](development.md): contribute to pysrf.
