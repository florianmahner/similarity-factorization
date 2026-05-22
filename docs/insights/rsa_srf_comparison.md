# RSA vs SRF Comparison: Results

## Experimental designs

**Factorial (controlled):** Stimuli explicitly constructed to vary along specific factors. Each factor has maximum variance—objects evenly distributed across levels. This is the best-case scenario for detecting structure.

**SPoSE (naturalistic):** Stimuli randomly sampled from a semantic embedding. Different dimensions have varying relevance to any given sample. A dimension like "vehicle-related" has low variance if the sample contains few vehicles. This tests robustness under realistic exploratory conditions.

## Why dimension variance matters

Similarity is computed as S = XX'. Dimensions with low variance contribute less to S because they don't differentiate objects—if everything scores ~0.9 on "is_tangible", that dimension doesn't affect which objects are similar.

For RSA, which tests each dimension marginally against the full S, low-variance dimensions produce weak correlations even when truly present. SRF factorizes S first, separating dimensions before testing, improving detection of low-variance dimensions.

## The linear combination problem

The similarity matrix is a mixture of all factor contributions:

S = x₁x₁' + x₂x₂' + x₃x₃' + x₄x₄'

**RSA tests marginally:** Each hypothesis H_j = x_j x_j' is tested against the full S. Even with perfect signal, cor(H_j, S) ≠ 1 because H_j explains only ~1/k of S's variance. The other factors act as "noise."

With k equal-variance factors: cor(H_j, S) ≈ √(1/k)
- 4 factors → cor ≈ 0.50
- 10 factors → cor ≈ 0.32

**SRF factorizes first:** S ≈ WW' separates the factors into distinct dimensions. Each recovered w_j captures one factor's contribution, so cor(w_j, x_j) ≈ 1 after alignment.

## Results

### Factorial design

| SNR | RSA | SRF-LOO | SRF-Global |
|-----|-----|---------|------------|
| 0.0 | ~1% | ~1% | ~1% |
| 0.5 | ~8% | ~94% | ~97% |
| 1.0 | 100% | ~99% | ~99% |

- All methods well-calibrated (FP ≤ 5% at SNR=0)
- SRF dramatically outperforms RSA at intermediate SNR
- At high SNR, all methods converge to full power

### SPoSE design

| SNR | RSA | SRF-LOO |
|-----|-----|---------|
| 0.0 | ~1% | ~1% |
| 0.5 | ~51% | ~80% |
| 1.0 | ~69% | ~97% |

RSA plateaus at ~70% even at perfect signal because low-variance dimensions remain hard to detect.

**Power by variance quartile (SNR=1.0):**

| Variance | RSA | SRF-LOO |
|----------|-----|---------|
| Q1 (low) | 26% | 82% |
| Q2 | 62% | 98% |
| Q3 | 82% | 99% |
| Q4 (high) | 94% | 100% |

SRF is robust across variance quartiles. RSA struggles with low-variance dimensions.

## Practical implications

- **Controlled experiments** (factorial stimuli): RSA works if SNR is high; SRF provides earlier detection at lower SNR
- **Exploratory experiments** (naturalistic stimuli): SRF preferred because it detects dimensions regardless of their variance in the sample
- **Unknown structure**: SRF advantage when researchers don't know a priori which dimensions will be relevant
