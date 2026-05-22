# Triplet-to-RSM Conversion: Softmax Bias and Denoising Limits

## Summary

The count-based RSM from triplets (`compute_similarity_matrix_from_triplets`) estimates the **average choice probability**, not the underlying similarity. This creates a systematic bias that cannot be removed by any matrix-level denoising. The best achievable test accuracy with pure matrix operations is ~63.7%, vs SPoSE/VICE at ~64.9-65.0%.

## Background

Given odd-one-out triplets (i, j, k) where (i,j) is the similar pair, the standard approach builds:

```
S[i,j] = (count_chosen + alpha) / (times_shown + 2*alpha)
```

The generative model (used by SPoSE/VICE) is softmax:

```
P(choose i,j | i,j,k) = exp(s_ij) / (exp(s_ij) + exp(s_ik) + exp(s_jk))
```

The count RSM estimates `E_k[P(choose i,j | random k)]`, not `s_ij`. The mapping from `s_ij` to `E[P]` is nonlinear and context-dependent (depends on which k appears).

## Evidence

### Experiment 1: Denoising the count RSM (train_90.txt -> test_10.txt)

The count RSM is ~50% noise (911/1854 eigenvalues are negative). Low-rank PSD projection dramatically improves generalization:

| Method | Train | Test |
|--------|-------|------|
| Count RSM raw (alpha=1) | 76.0% | 59.1% |
| Count RSM rank-20 | 63.9% | 63.5% |
| Count RSM rank-30 | 64.3% | 63.6% |
| **Count RSM rank-35** | **64.4%** | **63.7%** |
| Count RSM rank-50 | 64.8% | 63.4% |
| Count RSM rank-200 | 67.2% | 61.2% |
| **SPoSE** | **64.9%** | **64.9%** |
| **VICE** | **~65.0%** | **~65.0%** |

Optimal rank is ~30-35. Higher ranks overfit; lower ranks lose signal.

### Experiment 2: Alternative denoising methods

All static denoising methods converge to the same ~63.5-63.7% ceiling:

| Method | Test |
|--------|------|
| Hard rank truncation (rank 35) | 63.7% |
| Gavish-Donoho hard threshold | 63.4% |
| Gavish-Donoho soft shrinkage | 63.4% |
| Variance-stabilizing transform + rank 30 | 63.6% |
| Double-centering + rank 30 | 63.6% |
| Log-odds transform + rank 30 | 63.7% |

### Experiment 3: MLE refinement (embedding-level, = SPoSE)

Parameterizing S = LL^T and optimizing the triplet softmax loss directly closes the gap:

| Method | Train | Test |
|--------|-------|------|
| Rank-35 PSD (matrix only) | 64.4% | 63.7% |
| Rank-35 + MLE refinement (5 epochs) | 65.2% | 64.3% |
| Rank-35 + MLE refinement (10 epochs) | 65.3% | 64.3% |

But this IS essentially SPoSE (optimizes the same softmax loss). Not a matrix-level improvement.

### Experiment 4: Kleindessner kernels (k1, k2)

Kleindessner & von Luxburg (NeurIPS 2017) build kernels from triplets by comparing rankings. Failed on THINGS because the triplet coverage per anchor is too sparse (0.2%):

| Method | Test |
|--------|------|
| K1 (Kendall tau) raw | 46.7% |
| K2 (ranked by others) raw | 43.2% |

### Experiment 5: Iterative softmax bias correction (in progress)

Attempt to invert the softmax aggregation at the matrix level:
1. Compute P_hat[i,j] = mean_k softmax(s_ij, s_ik, s_jk) from current s
2. Adjust s += eta * (P_obs - P_hat)
3. Repeat

Unconstrained version (v1): negligible improvement (~0.1pp after 15 iterations). The correction goes into the noise subspace that rank projection discards.

Low-rank constrained version (v2): results pending.

## Tentative Conclusions

1. **The count RSM is a sufficient statistic for ~97% of the triplet signal.** Rank-35 denoised count RSM achieves 63.7% vs SPoSE's 64.9%. Given the noise ceiling of ~67.2%, the gap is (64.9-63.7)/(67.2-33.3) = 3.5% of recoverable signal.

2. **The remaining gap is due to softmax bias, not noise.** All denoising methods hit the same ~63.7% ceiling. The bias arises because the count RSM estimates the average choice probability (context-dependent) rather than the underlying similarity.

3. **Fixing the bias requires triplet-level information.** The softmax denominator varies per context k. The count RSM loses which k's appeared with each pair, so no matrix-level operation can fully recover the underlying similarity.

4. **Observation-count weighting could help marginally** by making SRF trust well-observed entries more. But this addresses noise heteroscedasticity, not bias. Expected improvement: <0.2pp.

## Open Questions

- Can iterative bias correction with a low-rank constraint (alternating projection) close part of the gap?
- Is there a different "sufficient statistic" from triplets that preserves more context information than pairwise counts?
- How does the gap change with data volume (does it shrink with more triplets)?

## Reproducibility

- Data: `data/things/triplets_47/train_90.txt`, `test_10.txt`
- Scripts: `experiments/sandbox/things/rsm_diagnosis/run.py`, `run_advanced_denoise.py`, `run_kleindessner.py`, `run_bias_correction.py`
- n_objects = 1854, SPoSE/VICE embeddings from `data/things/`
