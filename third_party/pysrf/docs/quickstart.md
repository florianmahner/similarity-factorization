# Quick start

This guide walks through the core pysrf workflow: starting from a
similarity matrix, recovering interpretable dimensions, handling missing
data, and selecting the number of dimensions with cross-validation.

## Decompose a similarity matrix

A similarity matrix $S$ captures pairwise relationships between items.
It might come from behavioral judgments, neural recordings, word
associations, or model activations. pysrf factorizes $S$ into a
non-negative embedding $W$ such that $S \approx WW^\top$, where each column
of $W$ is an interpretable dimension.

```python
import numpy as np
from pysrf import SRF

# Load or construct your similarity matrix
s = np.random.rand(100, 100)
s = (s + s.T) / 2  # ensure symmetry

# Decompose into 10 dimensions
model = SRF(rank=10, max_outer=20, random_state=42)
w = model.fit_transform(s)

# Reconstruct the similarity matrix from the embedding
s_reconstructed = model.reconstruct()

# Evaluate reconstruction quality
score = model.score(s)
print(f"Reconstruction error: {score:.4f}")
```

Each row of `w` gives an item's coordinates along the recovered dimensions.
Items that load on the same dimension are similar for the same reason,
making the result directly interpretable.

## Handle missing data

In behavioral experiments, the number of pairwise comparisons grows
quadratically with the number of items, so most similarity matrices are
incomplete. pysrf handles this natively: mark missing entries as `NaN` and
the model learns only from the observed pairs.

```python
import numpy as np
from pysrf import SRF

# Similarity matrix with 30% of entries unobserved
s = np.random.rand(100, 100)
s = (s + s.T) / 2
s[np.random.rand(100, 100) < 0.3] = np.nan

# Missing entries are excluded during fitting
model = SRF(rank=10, missing_values=np.nan, random_state=42)
w = model.fit_transform(s)

# The reconstruction fills in the missing entries
s_completed = model.reconstruct()
```

The completed matrix `s_completed` contains the model's predictions for
both the observed and the previously missing entries.

## Select the number of dimensions

Choosing the right number of dimensions (rank) is critical. Too few
dimensions miss important structure; too many overfit. pysrf provides
entry-wise cross-validation: it holds out individual similarity values,
fits the model on the remaining entries, and evaluates prediction accuracy
on the held-out values.

```python
from pysrf import cross_val_score

s = np.random.rand(100, 100)
s = (s + s.T) / 2

# Evaluate candidate ranks
cv = cross_val_score(s, param_grid={"rank": [5, 10, 15, 20]})

print(f"Best rank: {cv.best_params_}")
print(f"Best score: {cv.best_score_:.4f}")
```

This approach respects the dependency structure of similarity matrices,
unlike standard cross-validation methods designed for independent
observations.

For consensus embeddings, sampling-bound estimation, and the full analysis
pipeline, see the [Examples](examples.md) page.
