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

## Rank estimation and cross-validation

The number of dimensions is estimated first. Cross-validation then checks
whether nearby ranks produce a lower held-out error.

```python
from pysrf import cross_val_score, estimate_rank

estimate = estimate_rank(s, random_state=42)
ranks = range(max(1, estimate.rank - 3), estimate.rank + 4)
curve = cross_val_score(
    s,
    ranks=ranks,
    sampling_fraction=estimate.sampling_fraction,
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

mean_mse = curve.groupby("rank")["val_mse"].mean()
print(f"Estimated rank: {estimate.rank}")
print(f"CV minimum: {int(mean_mse.idxmin())}")
```

## Consensus embeddings

Symmetric NMF is non-convex. Different random initializations can
converge to different solutions. To obtain a stable embedding, fit
multiple runs and align them. `EnsembleFit` runs the base model from
several random starts; `AlignedConsensus` aligns runs and returns the most
central embedding.

```python
from sklearn import pipeline
from pysrf import AlignedConsensus, EnsembleFit, SRF, estimate_rank

estimate = estimate_rank(s, random_state=42)

pipe = pipeline.Pipeline(
    [
        ("ensemble", EnsembleFit(SRF(rank=estimate.rank), n_runs=50)),
        ("align", AlignedConsensus(rank=estimate.rank)),
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

## Complete workflow

A full analysis combines rank selection, ensemble fitting, and
evaluation:

```python
import numpy as np
from pysrf import SRF, cross_val_score, estimate_rank

# 1. Generate a noisy, incomplete similarity matrix
np.random.seed(42)
n, true_rank = 100, 8
w_true = np.random.rand(n, true_rank)
s = w_true @ w_true.T
s += 0.1 * np.random.randn(n, n)
s = (s + s.T) / 2
mask = np.random.rand(n, n) < 0.2
s[mask] = np.nan

estimate = estimate_rank(s, random_state=42, n_jobs=-1)
ranks = range(max(1, estimate.rank - 3), estimate.rank + 4)

curve = cross_val_score(
    s,
    ranks=ranks,
    sampling_fraction=estimate.sampling_fraction,
    n_repeats=3,
    random_state=42,
    n_jobs=-1,
)
mean_mse = curve.groupby("rank")["val_mse"].mean()
print(f"Estimated rank: {estimate.rank}; CV minimum: {int(mean_mse.idxmin())}")

# 3. Fit the final model
model = SRF(rank=estimate.rank, max_outer=20, random_state=42)
w = model.fit_transform(s)
s_completed = model.reconstruct()

# 4. Evaluate
score = model.score(s)
print(f"Reconstruction error: {score:.4f}")
```
