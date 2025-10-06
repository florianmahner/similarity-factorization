# Quick Start

## Basic Usage

### Matrix Factorization

```python
import numpy as np
from pysrf import SRF

# Generate or load your similarity matrix
s = np.random.rand(100, 100)
s = (s + s.T) / 2  # ensure symmetry

# Fit the model
model = SRF(rank=10, max_outer=20, random_state=42)
w = model.fit_transform(s)

# Reconstruct the matrix
s_reconstructed = model.reconstruct()
# or equivalently: s_reconstructed = w @ w.T

# Evaluate fit
score = model.score(s)
print(f"Reconstruction error: {score:.4f}")
```

## Handling Missing Data

```python
import numpy as np
from pysrf import SRF

# Matrix with missing values (NaN)
s = np.random.rand(100, 100)
s = (s + s.T) / 2
s[np.random.rand(100, 100) < 0.3] = np.nan  # 30% missing

# Model handles missing data automatically
model = SRF(rank=10, missing_values=np.nan, random_state=42)
w = model.fit_transform(s)
s_completed = model.reconstruct()
```

## Cross-Validation to estimate the rank

### Manual Sampling Fraction

```python
from pysrf import cross_val_score

# Define parameter grid
param_grid = {
    'rank': [5, 10, 15, 20],
    'rho': [2.0, 3.0, 4.0]
}

# Run cross-validation
result = cross_val_score(
    s,
    param_grid=param_grid,
    sampling_fraction=0.8,  # 80/20 train/test split
    n_repeats=5,
    n_jobs=-1,
    random_state=42
)

print(f"Best parameters: {result.best_params_}")
print(f"Best score: {result.best_score_:.4f}")
```

### Automatic Sampling Fraction Estimation

The optimal sampling fraction can be automatically estimated:

```python
from pysrf import cross_val_score

# Automatically estimate optimal sampling fraction
result = cross_val_score(
    s,
    param_grid={'rank': [5, 10, 15, 20]},
    estimate_sampling_fraction=True,  # ✨ New feature!
    n_repeats=5,
    n_jobs=-1,
    random_state=42,
    verbose=1  # Shows estimated bounds
)

print(f"Best rank: {result.best_params_['rank']}")
```

## Sampling Bound Estimation

Estimate the sampling rate bounds required for reliable matrix completion:

```python
from pysrf import estimate_sampling_bounds_fast

# Estimate bounds
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

## Value Bounds

Constrain reconstructed values to known ranges:

```python
from pysrf import SRF

# For cosine similarity (range: [0, 1])
model = SRF(rank=10, bounds=(0, 1), random_state=42)
w = model.fit_transform(s)
s_reconstructed = model.reconstruct()

# Verify bounds
assert s_reconstructed.min() >= 0
assert s_reconstructed.max() <= 1
```

## Complete Example

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

## Next Steps

- Check the [API Reference](api/model.md) for detailed parameter descriptions
- See [Development](development.md) for contributing guidelines

