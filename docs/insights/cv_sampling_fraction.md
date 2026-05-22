# CV-Based Rank Selection: Sampling Fraction Requirements

**Date:** 2026-01-23
**Context:** Investigating why rank detection experiment fails for high-alpha Dirichlet simulations

## Summary

Cross-validation based rank selection for SRF appears to require higher sampling fractions (p) than the bounds estimation provides, particularly for data with small spectral gaps.

## Background

The rank detection experiment uses CV to select the optimal rank by:
1. Masking a fraction (1-p) of entries for validation
2. Fitting SRF on training entries
3. Scoring on held-out entries
4. Selecting rank with lowest validation error

The sampling fraction p is estimated from matrix spectral properties using `estimate_sampling_bounds_ultra()`.

## Evidence

### Experiment 1: Alpha variation with bounds-estimated p

For n=1000, k=20 Dirichlet simulation, varying alpha:

| alpha | spectral gap | bounds p | CV selected | correct? |
|-------|-------------|----------|-------------|----------|
| 0.1 | 13.73 | 0.30 | 20 | yes |
| 0.5 | 3.40 | 0.11-0.39 | 26 | no |
| 2.0 | 0.93 | 0.38 | 31 | no |
| 5.0 | 0.36 | 0.46 | 23 | no |

*Source: sandbox/simulation/rank_selection/dirichlet/run_k20_alphas.py*

### Experiment 2: Fixed p=0.7 vs bounds-estimated p

For alpha=0.5 (gap=3.40), testing 5 seeds:

| seed | bounds p | selected (bounds) | selected (p=0.7) |
|------|----------|-------------------|------------------|
| 0 | 0.107 | 26 | 20 |
| 1 | 0.108 | 26 | 20 |
| 2 | 0.108 | 26 | 20 |
| 3 | 0.389 | 26 | 20 |
| 4 | 0.396 | 26 | 20 |

**Accuracy:** bounds p = 0%, fixed p=0.7 = 100%

### Experiment 3: p_comparison across obs_per_dof

From `sandbox/simulation/rank_selection/detection_viz/outputs/251217/131405/p_comparison.csv`:

| obs_per_dof | bounds accuracy | p=0.5 accuracy | p=0.8 accuracy |
|-------------|-----------------|----------------|----------------|
| 33.2 (k=3) | 100% | 100% | 90% |
| 19.9 (k=5) | 100% | 100% | 100% |
| 9.95 (k=10) | 100% | 90% | 100% |
| 7.11 (k=14) | 100% | 90% | 90% |
| 4.97 (k=20) | 60% | 50% | 10% |
| 3.02 (k=33) | 20% | 10% | 0% |

*Note: This was tested with alpha=1.0 only.*

## Tentative Conclusions

1. **Bounds estimation may underestimate required p**: For alpha=0.5, bounds gives p~0.1-0.4 but CV needs p~0.7 to work.

2. **Spectral gap appears critical**: CV works well when gap > 10, struggles when gap < 1, regardless of p.

3. **obs_per_dof threshold**: Evidence suggests obs_per_dof >= 7 needed for reliable CV with bounds-estimated p.

4. **Higher fixed p may help intermediate cases**: For alpha=0.5 (gap~3.4), fixed p=0.7 achieves 100% vs 0% with bounds p.

## Open Questions

- Why does bounds estimation give lower p than needed?
- Is there a principled way to adjust p based on spectral gap?
- Does this issue affect real data or only simulations?
- Would different CV strategies (e.g., more repeats, different masking) help?

## Recommendations for Experiments

For simulation rank detection:
- Consider using fixed p=0.7 instead of bounds-estimated p
- Or use `max(bounds_p, 0.6)` as a minimum threshold
- Document that CV-based rank selection has limited applicability when spectral gap < 1

## Reproducibility

```bash
# Test bounds vs fixed p
python3 << 'EOF'
from scipy.stats import dirichlet
from pysrf import cross_val_score

n, k, alpha = 1000, 20, 0.5
W = dirichlet.rvs(alpha=[alpha] * k, size=n, random_state=42)
S = W @ W.T

# Fixed p=0.7
cv = cross_val_score(S, param_grid={'rank': list(range(12, 28, 2))},
                    sampling_fraction=0.7, n_repeats=10, n_jobs=-1, verbose=0)
print(f"Selected: {cv.best_params_['rank']}")  # Should be 20
EOF
```
