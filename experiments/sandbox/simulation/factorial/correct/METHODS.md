# Factorial Design: RSA vs SRF

## Setup

Let $X \in \mathbb{R}^{N \times D}$ be a one-hot encoded design matrix for a factorial design with $F$ factors. Each row indicates which factor levels were present for that item. We have $N = \prod_f L_f$ items and $D = \sum_f L_f$ columns.

We add noise to $X$ and compute the similarity matrix $S = \tilde{X}\tilde{X}^T$. The goal is to test whether individual factors explain variance in $S$.

## Methods

### RSA (Mantel Test)

For factor $f$, extract columns $X_f$ and compute hypothesis matrix $S_f = X_f X_f^T$. Test statistic: $r = \text{corr}(S, S_f)$ using upper triangular entries. P-value via permutation of rows/columns of $S_f$.

### SRF-Denoise

Factorize $S \approx WW^T$ via SRF. Test statistic: $r = \text{corr}(WW^T, S_f)$. Same permutation scheme as RSA.

### SRF-LOO

From the embedding $W$, predict factor labels via leave-one-out linear regression. Test statistic: correlation between predicted and true labels. P-value via permutation of true labels.

## Simulation

- **Design:** 4 factors with [2, 3, 2, 4] levels = 48 items, 11 dimensions
- **Noise models:** Column noise (replace columns) and element-wise noise
- **SNR:** 0.0 to 1.0 (0 = pure noise, 1 = pure signal)
- **Validation:** Type I error ~5% at SNR=0, power reaches 100% at SNR=1

## Results

See `outputs/` for power curves. All methods show:
- Proper Type I error control (~5%) at SNR=0
- Monotonically increasing power with SNR
- 100% power at SNR=1 for all factors
