# Examples

These examples show how to use pysrf for increasingly complex analyses,
from basic factorization to the full pipeline with rank selection,
ensemble embeddings, and evaluation.

## Basic factorization

Generate a similarity matrix from a known low-rank embedding and recover
the latent dimensions:

```python
import numpy as np
from pysrf import SRF

# Ground-truth low-rank structure
n, rank = 100, 10
w_true = np.random.rand(n, rank)
s = w_true @ w_true.T

# Recover the dimensions
model = SRF(rank=10, random_state=42)
w = model.fit_transform(s)
s_hat = model.reconstruct()
```

Because the input has exact rank 10, the reconstruction error is near
zero. In practice, real similarity matrices contain noise and the optimal
rank must be estimated (see below).

## Missing data

Behavioral similarity experiments typically sample only a fraction of all
possible item pairs. pysrf treats unobserved entries as missing rather
than imputing them:

```python
import numpy as np
from pysrf import SRF

n, rank = 100, 10
w_true = np.random.rand(n, rank)
s = w_true @ w_true.T

# Remove 30% of entries to simulate sparse sampling
mask = np.random.rand(n, n) < 0.3
s[mask] = np.nan

model = SRF(rank=10, missing_values=np.nan, random_state=42)
w = model.fit_transform(s)
s_completed = model.reconstruct()
```

The model learns from the observed entries and predicts the held-out ones
in the reconstruction.

## Cross-validation for rank selection

The number of dimensions is a free parameter. Cross-validation selects
the rank that best generalizes to unseen similarities. Set
`estimate_sampling_fraction=True` to automatically derive the hold-out
fraction from the data:

```python
from pysrf import cross_val_score, SRF

cv = cross_val_score(
    s,
    estimate_sampling_fraction=True,
    param_grid={"rank": [5, 10, 15, 20]},
    n_repeats=5,
    n_jobs=-1,
    random_state=42,
)

print(f"Best rank: {cv.best_params_['rank']}")
print(f"Best score: {cv.best_score_:.4f}")
```

## Consensus embeddings

Symmetric NMF is non-convex. Different random initializations can
converge to different solutions. To obtain a stable embedding, fit
multiple runs and align them. `EnsembleEmbedding` does this
automatically: it runs the factorization `n_runs` times, aligns the
columns using the Hungarian algorithm, and selects the most central
solution. `ClusterEmbedding` then applies consensus clustering to find
stable groups of items.

```python
from sklearn import pipeline
from pysrf.consensus import EnsembleEmbedding, ClusterEmbedding
from pysrf import SRF, cross_val_score

# 1. Select the rank
cv = cross_val_score(
    s,
    estimate_sampling_fraction=True,
    param_grid={"rank": [5, 10, 15, 20]},
    n_repeats=5,
    n_jobs=-1,
)

# 2. Build the ensemble-clustering pipeline
pipe = pipeline.Pipeline(
    [
        ("ensemble", EnsembleEmbedding(SRF(cv.best_params_), n_runs=50)),
        ("cluster", ClusterEmbedding(min_clusters=2, max_clusters=6, step=1)),
    ]
)

consensus_embedding = pipe.fit_transform(s)
```

## Value bounds

When the similarity measure has a known range, constrain the
reconstruction accordingly. For example, similarities derived from cosine
distance lie in [0, 1]:

```python
from pysrf import SRF

model = SRF(rank=10, bounds=(0, 1), random_state=42)
w = model.fit_transform(s)
s_reconstructed = model.reconstruct()

assert s_reconstructed.min() >= 0
assert s_reconstructed.max() <= 1
```

## Sampling-bound estimation

Before running cross-validation, you can estimate the range of hold-out
fractions that produce reliable scores. This is useful for very sparse
matrices where holding out too many entries leaves insufficient data for
fitting:

```python
from pysrf import estimate_sampling_bounds_fast

pmin, pmax, s_denoised = estimate_sampling_bounds_fast(
    s,
    n_jobs=-1,
    random_state=42,
)

print(f"Minimum sampling rate: {pmin:.4f}")
print(f"Maximum sampling rate: {pmax:.4f}")

# Use the midpoint for cross-validation
sampling_rate = 0.5 * (pmin + pmax)
```

## Complete workflow

A full analysis combines rank selection, ensemble fitting, and
evaluation:

```python
import numpy as np
from pysrf import SRF, cross_val_score, estimate_sampling_bounds_fast

# 1. Generate a noisy, incomplete similarity matrix
np.random.seed(42)
n, true_rank = 100, 8
w_true = np.random.rand(n, true_rank)
s = w_true @ w_true.T
s += 0.1 * np.random.randn(n, n)
s = (s + s.T) / 2
mask = np.random.rand(n, n) < 0.2
s[mask] = np.nan

# 2. Estimate sampling bounds
pmin, pmax, _ = estimate_sampling_bounds_fast(s, n_jobs=-1)
print(f"Sampling bounds: [{pmin:.3f}, {pmax:.3f}]")

# 3. Cross-validate to find the best rank
result = cross_val_score(
    s,
    param_grid={"rank": range(5, 21)},
    estimate_sampling_fraction=True,
    n_repeats=3,
    n_jobs=-1,
    random_state=42,
)
best_rank = result.best_params_["rank"]
print(f"Best rank: {best_rank} (true rank: {true_rank})")

# 4. Fit the final model
model = SRF(rank=best_rank, max_outer=20, random_state=42)
w = model.fit_transform(s)
s_completed = model.reconstruct()

# 5. Evaluate
score = model.score(s)
print(f"Reconstruction error: {score:.4f}")
```
