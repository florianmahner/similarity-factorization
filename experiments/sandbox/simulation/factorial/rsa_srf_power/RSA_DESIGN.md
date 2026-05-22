# RSA vs SRF Power Analysis: Design Document

## Problem Statement

We want to understand when SRF (Similarity-based Representation Factorization) provides higher statistical power than RSA (Representational Similarity Analysis) for detecting factorial structure in measured similarity data.

## Simple Noise Dimension Approach

Supervisor's approach: Add a noise dimension as a column to the factorial design, then factorize.

### Setup

```python
# Factorial design (hypothesis)
X_factorial = create_factorial_design(levels)  # N × k (48 × 11)

# Add noise as an extra column
noise = randn(N)
X_noisy = hstack([X_factorial, noise * weight])  # N × (k+1)

# Build RSM
RSM = X_noisy @ X_noisy.T  # Contains factorial + noise component
```

### Key Insight

The RSM decomposes as:
```
RSM = X @ X.T + weight² * (noise @ noise.T)
    = H_factorial + H_noise
```

- RSA correlates full RSMs: `corr(H_factorial, RSM)` is diluted by noise component
- SRF with rank k ignores the noise dimension (rank 1), recovering only factorial structure

## Results

### Power Comparison

| Noise Weight | RSA Power | SRF-Hungarian Power | RSA Effect Size | SRF Effect Size |
|--------------|-----------|---------------------|-----------------|-----------------|
| 0.0          | 100%      | 100%                | 1.00            | 0.99            |
| 0.5          | 100%      | 100%                | 0.87            | 0.90            |
| 1.0          | 100%      | 100%                | 0.39            | 0.83            |
| 1.5          | 100%      | 100%                | 0.19            | 0.72            |
| 2.0          | 100%      | 100%                | 0.11            | 0.59            |
| 3.0          | 93.3%     | 100%                | 0.05            | 0.45            |

### Key Findings

1. **RSA effect size drops faster than SRF** as noise increases
   - At noise=3.0: RSA correlation = 0.05 (near zero), SRF correlation = 0.45

2. **RSA power drops before SRF** at high noise
   - RSA: 93.3% power at noise=3.0
   - SRF: 100% power at noise=3.0

3. **SRF filters the noise dimension** through low-rank factorization
   - With rank=k (correct), SRF ignores the k+1th dimension (noise)
   - RSA has no such filtering mechanism

## When SRF Beats RSA

SRF has higher power when:
1. **Noise is in a separate dimension** (not corrupting existing signal dimensions)
2. **Correct rank is known** (so SRF uses rank=k, not k+1)
3. **Noise has high variance** (large noise weight)

RSA is robust when:
1. **Signal is strong** (low noise weight)
2. **Noise corrupts existing dimensions** (not separable by rank)

## Statistical Tests

### RSA: Mantel Test
```python
H = X_factorial @ X_factorial.T  # Hypothesis RSM
obs = corr(upper_tri(H), upper_tri(RSM))
null = [corr(upper_tri(permute(H)), upper_tri(RSM)) for _ in range(n_perms)]
p_value = mean(null >= obs)
```

### SRF: Hungarian Matching Test
```python
W = SRF(rank=k).fit_transform(RSM)  # Embedding
obs = hungarian_correlation(X_factorial, W)  # Optimal column alignment
null = [hungarian_correlation(permute(X_factorial), W) for _ in range(n_perms)]
p_value = mean(null >= obs)
```

### SRF: LOO k-NN Test
```python
W = SRF(rank=k).fit_transform(RSM)
for each item i:
    find nearest neighbor j in W (j ≠ i)
    check if factor_label[i] == factor_label[j]
obs_acc = fraction correct
null = [same with permuted labels for _ in range(n_perms)]
p_value = mean(null >= obs_acc)
```

## Conclusions

1. **SRF provides higher power than RSA when noise is in a separable dimension**
2. **The advantage comes from low-rank filtering**, not from the test statistic
3. **Knowing the correct rank is critical** for SRF to filter effectively
4. **RSA remains robust** at low-to-moderate noise levels
