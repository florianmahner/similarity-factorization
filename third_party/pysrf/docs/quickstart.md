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
dimensions miss important structure; too many overfit. `estimate_rank`
estimates the rank and the sampling fraction used for CV; `cross_val_score`
then gives a confirmation curve around that estimate.

```python
from pysrf import SRF, cross_val_score, estimate_rank

s = np.random.rand(100, 100)
s = (s + s.T) / 2

estimate = estimate_rank(s, random_state=42)
ranks = range(max(1, estimate.rank - 3), estimate.rank + 4)
curve = cross_val_score(
    s,
    ranks=ranks,
    sampling_fraction=estimate.sampling_fraction,
    random_state=42,
)

cv_mean = curve.groupby("rank")["val_mse"].mean()
print(f"Estimated rank: {estimate.rank}")
print(f"CV minimum: {int(cv_mean.idxmin())}")

model = SRF(rank=estimate.rank, random_state=42)
w = model.fit_transform(s)
```

Cross-validation starts from the observed entries and artificially hides
some of them as validation entries. Entries that were not measured in the
input remain missing throughout fitting.

For consensus embeddings and the full analysis pipeline, see the
[Examples](examples.md) page.
