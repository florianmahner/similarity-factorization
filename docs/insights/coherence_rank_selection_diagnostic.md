# Coherence Rank Selection: Cross-Dataset Diagnostic

**Date:** 2026-03-16
**Context:** Systematic investigation of why the v2 coherence framework fails to give a single k* on continuous-spectrum data.

## Summary

The v2 projected coherence framework (I^proj_k with random-direction null) works correctly at the per-dimension level. The per-component activation (first p where CI of I^proj_k exceeds null tau_k) is the right diagnostic. The issue is that continuous-spectrum matrices have no sharp signal/noise boundary, making any single k* inherently threshold-dependent.

## Evidence: Cross-dataset activation profiles

### Peterson animals (120x120, neural RSM)
- Sharp cutoff: k=2-6 activate (p=0.16-0.69), k=8 barely (p=0.95), k=10+ never
- Fraction above null: k=2: 88%, k=4: 60%, k=6: 32%, k=8: 4%, k=10+: 0%
- Interpretation: clear low-rank structure with ~6 recoverable dimensions

### Peterson various (120x120, neural RSM)
- Sharp cutoff: k=2-4 activate (p=0.31-0.73), k=6-8 barely (p=0.95), k=10+ never
- Interpretation: even sparser than animals, ~4-6 recoverable dimensions

### THINGS behavioral (1854x1854, triplet-derived)
- Gradual ramp: activation_p increases from 0.12 (k=5) to 0.95 (k=100), then never (k=105+)
- Fraction above null: k=5: 92%, k=25: 48%, k=50: 36%, k=80: 20%, k=100: 4%, k=105+: 0%
- At p=0.50: only k=5-20 significantly above null (excess > 0.04)
- At p=0.80: k=5-80 significantly above null
- Kappa changepoint: k=25 (transition from strong to weak signal)
- SRF CV optimal: k=80
- Interpretation: continuous spectrum, no sharp boundary. Rank depends on threshold.

### SWOW (8647x8647, random walk similarity)
- ALL 100 tested dimensions activate at p=0.05 with 100% of p-grid above null
- I^proj at p=0.50 ranges from 0.99 (k=5) to 0.54 (k=100)
- Running extended diagnostic with k_max=300 to find where activation drops
- Interpretation: extremely rich structure, likely >100 recoverable dimensions

### NSD subject 1 (neural features)
- Running (results pending)

## Key observations

1. **Dataset size matters**: Peterson (n=120) has tau ~ k/n ~ 0.05-0.50, large relative to I^proj for noise. THINGS (n=1854) has tau ~ 0.005-0.065, much smaller. SWOW (n=8647) has tau ~ 0.001-0.035, tiny. The null gets weaker as n grows because random projections onto a k/n fraction of dimensions become negligible.

2. **Spectral structure matters more**: Peterson has a clear spectral gap. THINGS has continuous decay. SWOW has very strong structure at all tested ranks. The coherence test correctly reflects these differences.

3. **The test is valid but the question is ill-posed**: "What is the optimal k?" assumes a sharp answer. For continuous spectra, the answer is a function k*(p) -- how many dimensions are recoverable at observation density p.

## Practical implications

For datasets with sharp spectral gaps (Peterson, block matrices): the per-component activation gives a clean k* as the last dimension that activates at any p < 1.

For continuous-spectrum datasets (THINGS, likely SWOW): rank selection requires an additional criterion:
- Conservative (kappa changepoint): finds the steepest spectral transition
- Moderate (fraction-above > 0.20): includes all dimensions passing a binomial test
- Liberal (last to activate at any p): includes everything marginally above null

The fraction-above-null f_k is the most interpretable single statistic:
- f > 0.50: robust signal (recoverable at most observation densities)
- 0.20 < f < 0.50: moderate signal (needs 50-80% of data)
- 0.05 < f < 0.20: marginal (needs >80% of data)
- f = 0: noise

## Reproducibility

```bash
# THINGS diagnostic
./scripts/submit sandbox/coherence/subspace_rank/run_diagnostic.py --bg

# Peterson datasets
./scripts/submit sandbox/coherence/subspace_rank/run_diagnostic_multi.py --bg

# SWOW k=300 + NSD
./scripts/submit sandbox/coherence/subspace_rank/run_diagnostic_large.py --bg
```

## Open questions

- Does SWOW show a transition below k=300?
- What does NSD look like? (neural data, intermediate n)
- Can the fraction-above-null be turned into a formal hypothesis test with known power?
- Should rank selection report a range [k_conservative, k_liberal] rather than a single k*?
