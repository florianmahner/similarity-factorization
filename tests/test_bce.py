import numpy as np
from pysrf import SRF

print("Generating Synthetic Data...")
np.random.seed(42)
N = 50  # Small scale for Python-based W update speed
Rank = 3

# Create ground truth factors
W_true = np.random.rand(N, Rank)
# Create probabilities (Sigmoid of dot product to ensure 0-1 range for BCE)
Logits = W_true @ W_true.T
# Normalize to 0-1 roughly for simulation
X_true = 1 / (1 + np.exp(-Logits))  # Sigmoid

# Fill diagonal for symmetry properties
np.fill_diagonal(X_true, 1.0)

# Add some missing values (NaN)
mask = np.random.rand(N, N) > 0.8  # 20% missing
X_miss = X_true.copy()
X_miss[mask] = np.nan
# Enforce symmetry on missingness
X_miss = np.triu(X_miss) + np.triu(X_miss, 1).T

print(f"Data Shape: {X_miss.shape}, Rank: {Rank}")
print("Testing SRF with Frobenius Loss...")
model_frob = SRF(rank=Rank, loss="frobenius", max_outer=15, verbose=1)
model_frob.fit(X_miss)

print("\nTesting SRF with BCE Loss (Your Concern)...")
# For BCE, input data essentially represents probabilities
model_bce = SRF(rank=Rank, loss="bce", max_outer=20, rho=1.0, verbose=1)
model_bce.fit(X_miss)

# Validate Results
W_pred = model_bce.w_
X_rec = W_pred @ W_pred.T

# Check reconstruction on observed data
obs_mask = ~np.isnan(X_miss)
# Clip reconstruction for BCE calc
X_rec_safe = np.clip(X_rec, 1e-9, 1 - 1e-9)

bce_error = -np.mean(
    X_miss[obs_mask] * np.log(X_rec_safe[obs_mask])
    + (1 - X_miss[obs_mask]) * np.log(1 - X_rec_safe[obs_mask])
)

print(f"\nFinal BCE Error on Observed Data: {bce_error:.5f}")

if bce_error < 0.7:  # Random guessing is usually ~0.69 (ln 2)
    print("SUCCESS: Algorithm converged better than random guess.")
else:
    print("WARNING: Convergence issues detected.")
