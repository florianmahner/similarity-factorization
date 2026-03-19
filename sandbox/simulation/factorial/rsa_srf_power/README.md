# RSA vs SRF Power Comparison

## Goal

Compare statistical power of RSA vs SRF for detecting factorial structure in similarity data, as a function of SNR.

## Experimental Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Create Factorial Design X                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  X = one-hot factorial design  (N × k)                          │
│      N = 48 items (2 × 3 × 2 × 4 factorial)                     │
│      k = 11 dimensions                                          │
│                                                                 │
│  Signal RSM:  S = X @ X.T                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Create Noise Column & Scale for SNR                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  noise = abs(randn(N))      (N × 1)                             │
│                                                                 │
│  Noise RSM:  E = noise @ noise.T                                │
│                                                                 │
│  Scale noise so that:                                           │
│                                                                 │
│      SNR = var(S) / var(S + E_scaled)                           │
│                                                                 │
│  where var() = variance of off-diagonal elements                │
│                                                                 │
│  SNR = 1.0  →  no noise                                         │
│  SNR = 0.5  →  signal is 50% of total variance                  │
│  SNR → 0    →  mostly noise                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Build Combined RSM                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  X_combined = [X, noise_scaled]    (N × k+1)                    │
│                                                                 │
│  RSM = X_combined @ X_combined.T                                │
│      = S + E_scaled                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: RSA Test (Mantel)                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  H = X @ X.T  (hypothesis = signal RSM)                         │
│                                                                 │
│  obs_rsa = corr(offdiag(H), offdiag(RSM))                       │
│                                                                 │
│  NULL DISTRIBUTION:                                             │
│    For p in 1..n_perms:                                         │
│      H_perm = permute rows/cols of H                            │
│      null_rsa[p] = corr(offdiag(H_perm), offdiag(RSM))          │
│                                                                 │
│  p_rsa = (sum(null_rsa >= obs_rsa) + 1) / (n_perms + 1)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: SRF Test (Matching LOO)                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Fit SRF(rank=k) ONCE on RSM → W (N × k)                        │
│                                                                 │
│  OBSERVED ACCURACY:                                             │
│    For each item i in 1..N (LOO):                               │
│      1. Hungarian match W[~i] to X[~i] on all k columns         │
│         → get column permutation                                │
│      2. Apply permutation to W[i] → W_aligned[i]                │
│      3. For each factor f:                                      │
│         pred[f] = argmax(W_aligned[i, factor_f_cols])           │
│         true[f] = argmax(X[i, factor_f_cols])                   │
│         correct if pred[f] == true[f]                           │
│    obs_acc = total correct / (N × n_factors)                    │
│                                                                 │
│  NULL DISTRIBUTION:                                             │
│    For p in 1..n_perms:                                         │
│      X_perm = shuffle rows of X                                 │
│      For each item i in 1..N (LOO):                             │
│        1. Hungarian match W[~i] to X_perm[~i]                   │
│        2. Apply permutation to W[i]                             │
│        3. Predict each factor via argmax, compare to X_perm[i]  │
│      null_acc[p] = total correct / (N × n_factors)              │
│                                                                 │
│  p_srf = (sum(null_acc >= obs_acc) + 1) / (n_perms + 1)         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Repeat & Compute Power                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each SNR in [0.1, 0.2, 0.3, ..., 0.9, 1.0]:                │
│    For each repeat in 1..n_repeats:                             │
│      - Run Steps 2-5                                            │
│      - Record p_rsa and p_srf                                   │
│                                                                 │
│  power_rsa[snr] = fraction of repeats with p_rsa < 0.05         │
│  power_srf[snr] = fraction of repeats with p_srf < 0.05         │
│                                                                 │
│  Plot: Power vs SNR for RSA and SRF                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Insight

- **RSA** correlates full RSMs, so the noise dimension dilutes the signal
- **SRF** with rank=k ignores the noise dimension (k+1th column), recovering only factorial structure
- At low SNR, SRF should maintain higher power than RSA

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 48 | Number of items (2×3×2×4 factorial) |
| k | 11 | Factorial dimensions |
| n_repeats | 50 | Simulation repeats per SNR |
| n_perms | 500 | Permutations per test |
| SNR | 0.1 - 1.0 | Signal-to-noise ratio |
