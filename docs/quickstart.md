# Quick Start

## Basic Usage

### Factorize a symmetric matrix

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

s = np.random.rand(100, 100)
s = (s + s.T) / 2

# Define parameter grid
cv = cross_val_score(s, param_grid = {'rank': [5, 10, 15, 20]})

print(f"Best parameters: {result.best_params_}")
print(f"Best score: {result.best_score_:.4f}")
```