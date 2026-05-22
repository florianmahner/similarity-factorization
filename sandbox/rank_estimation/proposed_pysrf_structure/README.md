# Proposed `pysrf` source layout after the coherence integration

This folder is a **non-functional skeleton** — empty files with docstrings
and signatures only. Use it to inspect the final tree shape and the public
API surface before we actually edit `third_party/pysrf/`.

## What changes

```
third_party/pysrf/pysrf/
├── __init__.py                   UPDATED   imports flip
├── model.py                      (kept, untouched)
├── consensus.py                  (kept, untouched)
├── _bsum.py                      (kept, untouched)
├── cross_validation.py           UPDATED   EntryKFold replaces EntryMaskSplit;
│                                            cv_sampling_fraction sourced from
│                                            RankEstimator (no bounds.py call)
├── coherence/                    NEW       4-file subpackage
│   ├── __init__.py
│   ├── _bootstrap.py
│   ├── _rank_selection.py
│   ├── _sampling_fraction.py
│   └── estimator.py
└── bounds.py                     REMOVED   no longer used
```

## What each new file does

| File | Public name(s) | Responsibility |
|---|---|---|
| `coherence/__init__.py` | `RankEstimator` | single export |
| `coherence/_bootstrap.py` | `symmetrize`, `observation_mask`, `reference_eigenpairs`, `bootstrap_coherence` | symmetrize → mask → reference eigvecs → Bernoulli replicates → per-rank Iproj + Rayleigh-trace numerator |
| `coherence/_rank_selection.py` | `leakage_profile`, `changepoint` | scaled-leakage per dim + F-stat 2-segment changepoint → `k_cut` |
| `coherence/_sampling_fraction.py` | `recovery_curve`, `invert_recovery`, `detectability_floor` | deficit curve at `k_cut` → PAV monotone → invert at δ → BBP floor |
| `coherence/estimator.py` | `RankEstimator` | sklearn-style class glueing the three steps; attributes `rank_`, `sampling_fraction_`, plus `cv_sampling_fraction(n_folds)` |
| `cross_validation.py` | `EntryKFold`, `cross_val_score`, `GridSearchCV`, `fit_and_score` | k-fold splitter with adaptive cap; `cross_val_score(estimate_sampling_fraction=True)` → `RankEstimator(s).cv_sampling_fraction(k_cv)` |

## What goes away

- `pysrf/bounds.py` — 481 lines, no longer reachable after the rewire.
- `pysrf/coherence.py` (old single-file, 602 lines) — replaced by the subpackage.
- `EntryMaskSplit`, `create_train_val_split`, `estimate_rank` — old public API.

## How to inspect

Each stub file below has the **public signatures + docstrings** of what will
land there — no implementation. Read top-to-bottom for the API surface, or
diff against the sandbox files (`sandbox/rank_estimation/coherence/*.py`,
`sandbox/rank_estimation/cross_validation.py`) which already contain the
working implementation that will be copied over verbatim (modulo
import-path tweaks).
