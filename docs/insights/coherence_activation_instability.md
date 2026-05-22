# Coherence Activation Instability: k* Depends on k_max

**Date:** 2026-03-12
**Context:** Testing projected incremental coherence (`sandbox/coherence/kachun/v2/`) on THINGS behavioral similarity (1854x1854, triplet-derived, 8 NaN)

## Summary

The null-calibrated activation diagnostic (`activation_p` / `activation_idx` in `_compute_coherence.py`) is not invariant to the maximum rank tested (`k_max`). Increasing k_max systematically shifts k* upward because the activation metric uses count-based pooling across all tested dimensions.

## Background

The coherence method has two stages:
1. **Stage 1** computes projected incremental coherence I^proj_k(p) with bootstrap CIs and null calibration
2. **Stage 2** (`analyze_iproj_signal_alignment_v3`) uses kappa changepoint to detect the signal/noise boundary

Stage 1 produces an `activation_p[r]` diagnostic: the first sampling fraction p where at least r+1 of the tested dimensions pass the null test. The run scripts (`run_things_behavior.py`) use this to report k* as the largest k where activation occurs.

## Evidence

Three runs on the same THINGS behavioral matrix with identical parameters (B=100, p in [0.05, 0.95], 25 grid points) but different k_max:

| k_max | k* (activation) | k=70 status | k=85 status | k=90 status | k=100 status |
|-------|-----------------|-------------|-------------|-------------|--------------|
| 70    | 65              | not activated | -         | -           | -            |
| 100   | 85              | p=0.838     | p=0.950    | not activated | not activated |
| 120   | 100             | p=0.725     | p=0.838   | p=0.875     | p=0.950      |

k=70 was NOT activated when k_max=70 but IS activated (at p=0.838) when k_max=100 and at p=0.725 when k_max=120. The reported k* shifted from 65 to 85 to 100 just by changing the search range.

## Mechanism

The instability comes from count-based pooling in `_compute_coherence.py` lines 729-747:

```python
A = (x_ci_lo >= tau_kp)  # (K_count, P) — which (k, p) cells pass null test
N = A.sum(axis=0)         # (P,) — count of ALL dimensions passing at each p
for r in range(K_count):
    idxs = np.where(N >= (r + 1))[0]  # first p where count >= r+1
```

`activation_p[r]` answers: "what is the first p where at least r+1 dimensions (out of any k in k_list) pass the null test?" When k_max increases, more dimensions enter the pool. Even noisy high-k dimensions occasionally pass at high p, inflating N(p) and making it easier to reach the count threshold r+1. This shifts all activation thresholds downward (lower p needed) and extends k* upward.

Concretely, if k_max=70 gives N(p=0.838)=65, adding dimensions k=75...100 might contribute a few more passes at p=0.838, pushing N above 70 and "activating" k=70.

There is a secondary mechanism: the bootstrap eigensolve uses `_topk_eigenvectors(A, Kmax, eigsh=...)`. Iterative eigensolvers (eigsh) can give slightly different results for the same low-rank eigenvalues when Kmax changes. For THINGS with only 8 NaN (obs rate 99.9998%), both the reference eigenspace (randomized SVD, line 603) and bootstrap eigenspaces depend on Kmax, but this effect is negligible compared to the count-based pooling.

## Per-component activation (stable but not saved)

The code also computes per-component activation (`activation_p_component`), which tests each dimension independently:

```python
for kk in range(K_count):
    idxs = np.where(A[kk, :])[0]
    if len(idxs) > 0:
        activation_idx_component[kk] = int(idxs[0])
        activation_p_component[kk] = float(p_list[idxs[0]])
```

This answers: "what is the first p where dimension kk individually passes the null test?" Since `tau_kp[kk, :]` and `x_ci_lo[kk, :]` for a given kk do not depend on other k values in the list (they only depend on `Kmax` through the eigensolver, which is negligible for dense matrices), this metric should be nearly invariant to k_max. However, `activation_p_component` is NOT saved in the diagnostics dict (lines 984-985 only save `activation_idx` and `activation_p`).

## Implications

1. **Do not use `activation_p[r]` for rank selection.** The reported k* is an artifact of the search range, not a property of the data.

2. **The kappa changepoint (stage 2) is more appropriate.** It operates on the per-dimension coherence profiles and detects the largest drop in leakage difficulty kappa_k. This should be invariant to k_max as long as k_max covers the true signal range.

3. **The coherence CV curve** (I_k at p_max vs null threshold tau_k) provides a per-dimension test analogous to cross-validation. Since tau_k is computed per-dimension from the null distribution, it does not suffer from count-based pooling.

4. **For continuous-spectrum data like THINGS**, there may not be a sharp rank cutoff. The spectral gap between signal and noise is gradual, making any rank selection method somewhat arbitrary. Model selection criteria (AIC/BIC on reconstruction error) or the coherence CV curve with a threshold may be more principled.

## Tentative Conclusions

- The count-based activation is a useful diagnostic for block-structured data with clear spectral gaps (e.g., Ka Chun's synthetic 10-block example), where the count reaches a plateau and adding more dimensions does not inflate it
- For continuous-spectrum data, prefer kappa changepoint or coherence CV curve for rank selection
- If per-component activation is needed, modify `_compute_coherence.py` to include `activation_p_component` in the diagnostics dict

## Open Questions

- Does kappa changepoint show similar (milder) instability with k_max on continuous-spectrum data?
- Would the per-component activation be stable in practice for THINGS, or does the eigsh sensitivity at high k introduce noise?
- How does the coherence CV curve compare to standard SRF cross-validation for rank selection accuracy?

## Reproducibility

```bash
# Three runs with different k_max (modify k_list in run_things_behavior.py):
# k_list = list(range(5, 71, 5))   -> k_max=70
# k_list = list(range(5, 101, 5))  -> k_max=100
# k_list = list(range(5, 121, 5))  -> k_max=120
./scripts/submit sandbox/coherence/kachun/v2/run_things_behavior.py --bg
```
