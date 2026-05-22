# PPMI Zeros Should Be Treated as Missing Data

## Summary

In the SWOW word association dataset, the PPMI similarity matrix has 98.7% zero entries. These zeros represent word pairs that were never sampled together, not pairs observed to have no association. Treating them as missing (NaN) rather than observed zeros is more principled and improves downstream prediction.

## Background

PPMI (Positive Pointwise Mutual Information) is defined as:

    PPMI(w, c) = max(log2(P(w,c) / P(w)P(c)), 0)

When count(w, c) = 0, the joint probability P(w,c) = 0, making PMI undefined (log of zero). The standard convention truncates this to PPMI = 0, but this conflates two fundamentally different cases:

1. **Unobserved pairs**: Words that were never presented together in the association task. No data exists. This is the vast majority (98.7% of entries).
2. **Observed but below-chance pairs**: Words that did co-occur but not above chance expectation. These have genuine zero PPMI. This is extremely rare in the SWOW data.

## Evidence

### Entry breakdown (SWOW, 8647 words)

| Category | Count | Percentage |
|----------|-------|-----------|
| NaN (diagonal only) | 8,647 | 0.01% |
| Zero (truncated PPMI) | 73,809,196 | 98.71% |
| Positive (above chance) | 952,766 | 1.27% |
| Directed edges in graph | 532,492 | 1.42% of pairs |

Only 532k of 37.4M unique word pairs have any co-occurrence data at all. The remaining 98.6% of zeros are from pairs that were never sampled.

### Sandbox comparison (100 outer x 30 inner iterations)

| Property | Zeros observed | Zeros missing | Delta |
|----------|---------------|---------------|-------|
| Arousal | 0.70 | 0.72 | +0.02 |
| Valence | 0.80 | 0.82 | +0.02 |
| Concreteness | 0.74 | 0.79 | +0.05 |
| Imageability | 0.72 | 0.77 | +0.05 |
| Mean (8/9 improved) | 0.71 | 0.73 | +0.02 |

R² on observed entries: 0.32 (still converging at 100 iterations).

## Literature support

- **GloVe** (Pennington et al., 2014): Uses a weighting function f(X_ij) where f(0) = 0, meaning zero co-occurrence pairs contribute nothing to the loss. This is equivalent to treating zeros as missing.
- **Levy & Goldberg (2014)**: Showed word2vec implicitly factorizes a shifted PMI matrix. Negative sampling handles unobserved pairs differently from observed zeros.
- **PMI definition**: PMI is undefined when P(w,c) = 0. The PPMI = 0 convention is practical, not principled.

## Tentative conclusions

- Treating PPMI zeros as missing is the correct approach for word association data where most pairs were never sampled.
- SRF's ADMM missing-data mechanism handles this naturally: it fits only the observed positive associations and predicts the unobserved entries from the factored structure.
- The consensus embedding should be recomputed with zeros as NaN.

## Reproducibility

- Sandbox: `experiments/sandbox/swow/missing_zeros/run.py`
- Stable: `experiments/analyses/swow/missing_zeros/run.py`
