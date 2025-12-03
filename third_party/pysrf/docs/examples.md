# Examples

## Basic Usage

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

## Handling Missing Data

```python
import numpy as np
from pysrf import SRF

# Generate data with missing entries
n, rank = 100, 10
w_true = np.random.rand(n, rank)
s = w_true @ w_true.T

# Mark missing entries
mask = np.random.rand(n, n) < 0.3
s[mask] = np.nan

# Fit model with missing data
model = SRF(rank=10, missing_values=np.nan, random_state=42)
w = model.fit_transform(s)
s_completed = model.reconstruct()
```

## Cross-Validation for Rank Selection

```python
from pysrf import cross_val_score, SRF

# Auto-estimate sampling fraction
cv = cross_val_score(
    s,
    estimate_sampling_fraction=True,
    param_grid={"rank": [5, 10, 15, 20]},
    n_repeats=5,
    n_jobs=-1,
    random_state=42
)

print(f"Best rank: {cv.best_params_['rank']}")
print(f"Best score: {cv.best_score_:.4f}")
```

## Ensemble and Consensus Clustering

```python
from sklearn import pipeline
from pysrf.consensus import EnsembleEmbedding, ClusterEmbedding
from pysrf import SRF, cross_val_score

# 1. Rank selection
cv = cross_val_score(
    s,
    estimate_sampling_fraction=True,
    param_grid={"rank": [5, 10, 15, 20]},
    n_repeats=5,
    n_jobs=-1,
)

# 2. Stable ensemble + consensus clustering
pipe = pipeline.Pipeline(
    [
        ("ensemble", EnsembleEmbedding(SRF(cv.best_params_), n_runs=50)),
        ("cluster", ClusterEmbedding(min_clusters=2, max_clusters=6, step=1)),
    ]
)

consensus_embedding = pipe.fit_transform(s)
```

## Value Bounds

```python
from pysrf import SRF

# Constrain reconstructed values to [0, 1] (e.g., for cosine similarity)
model = SRF(rank=10, bounds=(0, 1), random_state=42)
w = model.fit_transform(s)
s_reconstructed = model.reconstruct()

# Verify bounds
assert s_reconstructed.min() >= 0
assert s_reconstructed.max() <= 1
```

## Sampling Bound Estimation

```python
from pysrf import estimate_sampling_bounds_fast

# Estimate sampling rate bounds for reliable matrix completion
pmin, pmax, s_denoised = estimate_sampling_bounds_fast(
    s,
    n_jobs=-1,
    random_state=42
)

print(f"Minimum sampling rate: {pmin:.4f}")
print(f"Maximum sampling rate: {pmax:.4f}")

# Use mid-point for cross-validation
sampling_rate = 0.5 * (pmin + pmax)
```

## Complete Workflow

```python
import numpy as np
from pysrf import SRF, cross_val_score, estimate_sampling_bounds_fast

# 1. Generate data
np.random.seed(42)
n, true_rank = 100, 8
w_true = np.random.rand(n, true_rank)
s = w_true @ w_true.T

# 2. Add noise and missing data
s += 0.1 * np.random.randn(n, n)
s = (s + s.T) / 2
mask = np.random.rand(n, n) < 0.2
s[mask] = np.nan

# 3. Estimate sampling bounds
pmin, pmax, _ = estimate_sampling_bounds_fast(s, n_jobs=-1)
print(f"Sampling bounds: [{pmin:.3f}, {pmax:.3f}]")

# 4. Cross-validate to find best rank
result = cross_val_score(
    s,
    param_grid={'rank': range(5, 21)},
    estimate_sampling_fraction=True,
    n_repeats=3,
    n_jobs=-1,
    random_state=42
)
best_rank = result.best_params_['rank']
print(f"Best rank: {best_rank} (true rank: {true_rank})")

# 5. Fit final model
model = SRF(rank=best_rank, max_outer=20, random_state=42)
w = model.fit_transform(s)
s_completed = model.reconstruct()

# 6. Evaluate
score = model.score(s)
print(f"Reconstruction error: {score:.4f}")
```

