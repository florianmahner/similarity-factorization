# Integration review: rank-estimation pipeline in `pysrf`

Three commits land on `master`: a coherence subpackage with `estimate_rank() -> RankEstimate`, a minimal `cross_val_score`, and rewritten tests. I read the public surface (`pysrf/__init__.py`), the four-file `pysrf/coherence/` subpackage, the rewritten `pysrf/cross_validation.py`, both test modules, the unchanged `model.py` style reference, and the build manifest. 103 tests reportedly pass. Comments below are concrete; where I push back on the previous reviewer or call out residual debt I say so.

---

## 1. Public API surface

Seven names: `SRF`, `RankEstimate`, `estimate_rank`, `cross_val_score`, `EnsembleFit`, `ClusterConsensus`, `AlignedConsensus`. This is genuinely minimal and a big improvement over the pre-refactor surface. `bounds.py` is gone, `GridSearchCV` is gone, `EntryMaskSplit` is gone, `fit_and_score` is gone, `create_train_val_split` is gone, the old dict-returning `estimate_rank` is gone. Good.

`RankEstimate` at top-level alongside `estimate_rank` is the right call. It's the return type, users pickle/serialize it, they isinstance-check it (see `test_returns_rank_estimate`), and they introspect its fields. Demoting it to `pysrf.coherence.RankEstimate` would force the typical `est = estimate_rank(s); isinstance(est, pysrf.coherence.RankEstimate)` import dance for zero benefit. Co-locating a dataclass with its factory function is normal Python practice (see e.g. `subprocess.CompletedProcess`, `urllib.parse.SplitResult`). Keep as-is.

`adaptive_cap` is correctly private (lives in `pysrf.coherence._sampling_fraction`). It's an implementation detail that only `cross_val_score` and `estimate_rank` need; exposing it would create an API contract on the `0.95 / 2000` constants that the library should be free to change. Good.

Nit: `RankEstimate` in `__init__.py` is imported `from .coherence` (the public package alias), but `cross_val_score` imports `adaptive_cap` from `.coherence._sampling_fraction` (the private path). That asymmetry is correct — public clients route through `coherence.__init__`, internal cross-module use bypasses it — but worth a one-line comment in `__init__.py` to flag the convention.

---

## 2. Subpackage layout (4 files inside `coherence/`)

The previous reviewer was wrong on this one, in my opinion. The split *is* semantically meaningful and the user's call to keep it stands up. Look at what each file actually owns:

- `_bootstrap.py` (221 lines): masked-bootstrap engine, eigenpair extraction, the entire numerical heavy-lifting layer. Pure linear algebra. No knowledge of "rank" or "sampling fraction" as concepts.
- `_rank_selection.py` (74 lines): F-statistic changepoint on a leakage profile. No bootstrap, no eigendecomp; just operates on the median-overlap matrix.
- `_sampling_fraction.py` (126 lines): recovery-curve inversion, detectability floor, the shared `adaptive_cap` helper. Independent of bootstrap and changepoint.
- `_estimate.py` (168 lines): the orchestrator. Composes the three.

These really are three independent algorithmic concerns. The previous reviewer's argument ("they import in a linear chain into estimator.py") was based on the older stub. Now `adaptive_cap` is consumed from `cross_validation.py` as well — `_sampling_fraction` already has two consumers, falsifying the "single consumer" critique. The split also makes future work much cleaner: swap the changepoint method without touching the bootstrap; swap the bootstrap engine without touching downstream calibration.

That said, two observations:

1. `coherence/__init__.py` re-exports only two names. The previous reviewer's "package re-exporting a single name is a smell" is now "two names" — still mild — but it's defensible because `RankEstimate` is the public dataclass and `estimate_rank` is the function. Fine.
2. 600 lines could fit one file, but `model.py` is 715 lines and arguably _should_ have been split. Following the wrong precedent isn't a virtue.

Keep the subpackage. The split is real, not theatre.

---

## 3. Internal consistency / leaks

`_CV_CAP_FLOOR` and `_CV_HOLDOUT_BUDGET` exist in exactly one place (`pysrf/coherence/_sampling_fraction.py`). `adaptive_cap` is the only consumer, and `cross_validation.py` calls it via a single import. The previous review's "copy-pasted constants will drift in six months" issue is fully fixed. Verified by `grep -rn` for the literal `0.95` and `2000` in `pysrf/` — only the constants and a docstring example mention them.

The docstring of `cross_val_score` repeats the formula `max(0.95, 1 - 2000 / N_pairs)` verbatim instead of referring to `adaptive_cap`. This is a minor doc-rot risk: if you change `adaptive_cap`, the docstring lies silently. Replace with "(see :func:`pysrf.coherence._sampling_fraction.adaptive_cap`)" or just "(capped to retain a held-out budget; see source)" — but the private path in a docstring is awkward. I'd just delete the formula from the prose, since the warning message itself reports the cap value at runtime.

The warning message uses `stacklevel=3`, which is correct (warning fires inside `_kfold_entry_splits` called from `cross_val_score` called by the user). Good attention to detail.

---

## 4. `_kfold_entry_splits` design

Now a private generator, no class. This is correct. The previous reviewer's call to drop `BaseCrossValidator` was right and the implementation followed through cleanly.

Reading the body for edge cases:

- **Empty pool** (`p_outer` is tiny, or `s` is fully missing): `eligible.size == 0` or the Bernoulli draw returns nothing. `np.array_split(rng.permutation([]), n_folds)` yields `n_folds` empty arrays; downstream `setdiff1d(pool, [])` returns the empty pool; `train_mask` and `val_mask` are both all-zero (except possibly the diagonal). `_fit_score` would call `SRF.fit` on an all-NaN matrix, which `SRF.fit` rejects with "No observed entries found". **Bug-ish:** the user gets a confusing `ValueError` from inside SRF rather than a clear "no eligible pairs" from CV. Worth catching and re-raising, or returning all-NaN scores with a warning.
- **Fewer eligible pairs than folds**: `np.array_split` handles this fine (yields some empty arrays among the folds). `_fit_score` returns `nan` for empty val_mask. Acceptable, though a warning would be friendlier.
- **All-NaN diagonals**: `_observed_diagonal_indices` returns an empty array; `train_mask[[], []]` is a no-op. Fine.
- **Partially-observed but symmetric input**: handled correctly — eligible entries are scanned from the upper triangle only, then mirrored via the `cols[..]`/`rows[..]` write. No double-counting.
- **`pool` membership for `setdiff1d`**: `pool` contains positions in upper-triangle order, `val_positions` is a permutation slice of `pool`. `assume_unique=False` is conservative (it's actually unique). Could be `True`, marginal speed win.

**Cleaner alternative for the diagonal handling**: rather than computing `_observed_diagonal_indices` and writing them in after the val_mask is set, build the diagonal mask once outside the loop. The current code re-writes `train_mask[diag, diag]` for every fold, which is correct but slightly wasteful — `train_mask` is freshly allocated per fold anyway, so this is only `n_folds * n` index assignments. Not worth optimizing.

**Real-API concern**: the function takes `random_state: int | None` but the docstring doesn't say what `None` means. `check_random_state(None)` uses the global numpy RNG, which breaks the determinism contract `cross_val_score` claims. Either drop `None` from the signature or document the behavior.

**Type hint nit**: `random_state: int | None` is inconsistent with `cross_val_score`'s `random_state: int = 0` (no `None` allowed). Either align the types or document why they differ.

---

## 5. Style compliance vs the rest of pysrf

Cross-checked against `model.py`:

| Aspect | `model.py` | New code | Verdict |
|---|---|---|---|
| Module docstring | Multi-paragraph with Reference section | Single one-liner | **Inconsistent.** New modules should have at least a paragraph explaining the role. `_bootstrap.py`'s "Masked-bootstrap engine for eigenspace coherence." is a one-liner; everywhere else in pysrf the module docstring is fuller. |
| Author/License block | After docstring | After docstring | Matches. |
| `from __future__ import annotations` | Present | Present | Matches. |
| Logger setup (`logger = logging.getLogger(__name__)`) | Yes | **None in any new file** | **Inconsistent.** `consensus.py` has none either, so the inconsistency predates this PR — but if anything's worth a log line, it's the "inflated outer mask exceeds cap" warning currently going to `warnings.warn`. Mixed convention, not a regression. |
| Numpy-style Parameters/Returns | Throughout, very thorough | Mixed. `estimate_rank` has full Parameters/Returns; `cross_val_score` has full Parameters/Returns; the private helpers in `_bootstrap.py`, `_rank_selection.py`, `_sampling_fraction.py` have abbreviated docstrings with no Parameters block. | **Mostly consistent.** Private helpers in `model.py` (e.g. `_frobenius_residual`, `_initialize_w`) DO have full numpy-style. New private helpers don't. Tighten this. |
| Lowercase variable names | `w`, `x`, `s` | `s`, `w`, `a` | Matches. |
| `np.asarray(x, dtype=np.float64)` for input coercion | Yes | Yes in `symmetrize` and `estimate_rank` | Matches. |
| Function naming (`snake_case`, `_` prefix for private) | Consistent | Consistent | Matches. |

Specific style nits:

- `_bootstrap.py` line 121: `# ---- internal helpers ---------------------------------------------------` is fine but `consensus.py` uses `# -----...-----` with no leading `----`. Pick one.
- `_estimate.py` line 100: `n_bootstrap : int, default=20` has no description line. Every other Parameter in the same docstring does. Add one.
- `_rank_selection.py:38` `changepoint(leakage, min_rank=2, min_segment=2)` — the public function is `changepoint` (no underscore), but it's not in any `__init__.py`, so it's effectively private. Either prefix with `_` or expose it. Currently it's a phantom public name.
- Similarly `leakage_profile`, `recovery_curve`, `invert_recovery`, `detectability_floor`, `bootstrap_coherence`, `symmetrize`, `observation_mask`, `reference_eigenpairs` are all undocumented-private (no `_` prefix, not in any `__init__.py`). Either underscore-prefix them or accept that any `pysrf.coherence._bootstrap.symmetrize` import is fair game.

This last point is the largest style problem. The convention in pysrf is `_` for private; the new code uses _filename underscore_ to mean private and assumes nobody will reach into the module. **Recommendation:** prefix all module-level helpers in `coherence/*.py` with `_`, OR add an explicit `__all__` to each file listing only what `_estimate.py` consumes. Pick one to mean "private".

---

## 6. Determinism / reproducibility

This is the real footgun. `_top_eigenpairs` calls `eigsh(a, k=k, which="LA", tol=1e-6)` with no `v0=`, which falls back to ARPACK's default starting vector — drawn from numpy's global RNG. So two `estimate_rank` calls with the same `random_state` can return different eigenvectors (within the same subspace, modulo signs and degenerate-eigenvalue rotation), which propagates into `overlap`, the leakage profile, and ultimately `rank` and `sampling_fraction` for borderline cases.

The test `test_deterministic` has been weakened to tolerate this: it requires equal `rank`, `pytest.approx(rel=1e-6)` for `sampling_fraction`, and `rtol=1e-5` for eigenvalues. That hides the bug. The fix is one of:

1. Pass `v0=` explicitly: `eigsh(a, k=k, v0=rng.standard_normal(n), which="LA", tol=1e-6)`. Threads `random_state` into a per-call deterministic v0.
2. Drop `eigsh` for the bootstrap inner loop entirely. The bootstrap calls `_top_eigenpairs` once per replicate (~B × P = 400 calls for default settings), and for `n=60, k=15` the speedup vs dense `eigh` is small. Dense is deterministic. This is what `_dense_top_eigenpairs` already does.
3. Seed numpy's global state at function entry. Ugly, leaks global side effects, not recommended.

Option 1 is cleanest and preserves the perf. The signature change is contained:

```python
def _top_eigenpairs(a, k, rng=None):
    ...
    v0 = rng.standard_normal(a.shape[0]) if rng is not None else None
    values, vectors = eigsh(a, k=k, v0=v0, which="LA", tol=1e-6)
```

Threading `rng` through `_bootstrap_one_grid_point` is already there (`rng = np.random.default_rng(seed)`). Just pass it down. Then tighten `test_deterministic` to `rtol=1e-12` and `sampling_fraction == approx(rel=1e-12)`.

**This is a must-fix before opening the PR.** A doc note ("results may vary across runs due to ARPACK internals") is not acceptable; the function takes `random_state` as a parameter, that's a contract.

---

## 7. Tests — sufficient?

Eight tests in `test_coherence.py`, six in `test_cross_validation.py`. Coverage of happy paths is good. What's missing:

- **`estimate_rank` with no signal** (pure-noise input): does it return `rank=1` or some sane fallback? Currently untested.
- **Pathological inputs**: `n=2`, fully-missing matrix, all-identical similarities (rank-1 trivial). The `_resolve_max_rank(n=2)` returns `max(min(0, 100), 2) = 2`, which then drives `_top_eigenpairs(a, k=2)` on a 2x2 — fine, but untested.
- **Detectability floor activation**: when `detectability_floor > raw_fraction`, the floor kicks in. `test_sampling_fraction_in_unit_interval` checks `floor <= sampling_fraction` but doesn't construct a case where the floor is the binding constraint. Worth a regression test.
- **`cross_val_score` with `missing_values != np.nan`**: the code branches in `_eligible_pair_positions` and `_observed_diagonal_indices` on the missing-marker, but no test passes `missing_values=-1` or similar. The dead branch is at risk of bit-rotting.
- **`cross_val_score` with all-NaN matrix**: would error with the confusing message in (4). Worth a regression test that asserts a clean error.
- **Diagonal-in-training contract**: write a test that confirms `val_mask[diag, diag]` is always False and `train_mask[diag, diag]` is True where observed. This is a core correctness claim that's currently implicit.
- **Symmetry of masks**: assert `(train_mask == train_mask.T).all()`, `(val_mask == val_mask.T).all()`.
- **Disjointness**: `(train_mask & val_mask).sum() == 0`. The whole partition-CV story rests on this.

The last three are one-line asserts in `test_returns_long_dataframe` and would catch any future refactor that breaks the contract. Add them.

Running `pytest --cov pysrf.coherence pysrf.cross_validation` is worth the 10 seconds. I'd bet line coverage is ~85% with the missing-values branch and `invert_recovery` boundary cases (`monotone[0] <= tolerance`, `monotone[-1] >= tolerance`) being the gaps.

---

## 8. Build manifest

`meson.build` lists the four `coherence/*.py` files explicitly. This is brittle: add a fifth file, forget to update `meson.build`, and the file silently doesn't ship in the wheel. Tests pass because pytest reads from the source tree.

Meson does support globs via `fs.module()` but they're discouraged because the build doesn't reconfigure when the glob result changes. The pragmatic answer for a pure-Python subpackage is `py.install_subdir`:

```meson
py.install_subdir('pysrf/coherence',
  install_dir: py.get_install_dir() / 'pysrf',
  exclude_files: ['__pycache__'],
)
```

That installs whatever `pysrf/coherence/` contains. One line replaces the seven-line block. Worth doing.

(For `pysrf/__init__.py`, `_bsum.py`, `consensus.py`, `cross_validation.py`, `model.py` the existing explicit list is fine — flat directory, low churn.)

---

## 9. Other pre-PR items

- **`Generator` type hint** in `cross_validation.py` line 9 uses `typing.Generator`. With `from __future__ import annotations` you can use `collections.abc.Generator` (Python 3.9+) which is the modern form. Trivial.
- **`stacklevel=3` for the cap warning** — verified correct in (3).
- **`warnings.warn` for the cap-clip** is the right level (not an error, user might be running with deliberately aggressive `sampling_fraction`), but the warning could include a suggested fix: "consider passing `sampling_fraction <= {cap * (n_folds-1)/n_folds:.3f}`". Minor UX.
- **`__init__.py` import order** of `pysrf/__init__.py`: alphabetical except `SRF` is last. The convention is unclear — `model.py` is logically first (others depend on it), so import-order-by-dependency is defensible. Either alphabetize or comment why.
- **`_estimate.py` `_resolve_n_jobs`** clamps to `cpu - 1`. This is unusual — sklearn convention is `n_jobs=-1` means all cores, `n_jobs=None` means 1. Here `None` means `cpu - 1`. Likely intentional (be a good citizen on shared machines) but inconsistent with the rest of the ecosystem; document it in the parameter docstring.
- **`_estimate.py` line 145** stores `sampling_grid=p_sorted` in the returned `RankEstimate`, not the original input `grid`. This is fine (sorted is more useful for plotting), but the docstring of `RankEstimate.sampling_grid` says "Sampling probabilities at which the bootstrap was evaluated." which is ambiguous — was it sorted before evaluation or after? Clarify: "...sorted ascending after the bootstrap."
- **`bootstrap_coherence`'s `n_jobs=1` shortcut** (line 109) is a nice optimization for tests. The Parallel-vs-serial branch is symmetric, so any change to one needs to change the other. Worth a `# keep in sync` comment.
- **The `_bernoulli_replicate` rescaling** (`scale = 1.0 / max(p, 1e-12)`) silently caps `p` at `1e-12` — meaningful only if a user passes `sampling_grid=[0.0, ...]`. The default grid starts at 0.05 so this never bites in practice, but if exposed to users via `sampling_grid` kwarg, a `p == 0` entry would produce a `1e12`-scaled garbage replicate without error. Either reject zero values in `_resolve_sampling_grid` or document the floor.

---

## Bottom line

This is a real improvement. Public surface shrank from 9 names to 7 (the four new ones plus three pre-existing consensus classes), 1100 lines of `bounds.py` + old `coherence.py` collapsed to ~600 lines, the duplicated cap constants are deduplicated, and `EntryKFold` correctly disappeared as a public class. Two of the three "must-fix" items the previous reviewer raised are fixed; the third (the subpackage split) was a judgment call where the user's instinct turned out to be defensible.

**Must-fix before PR:**

1. **ARPACK non-determinism (§6).** `eigsh` is called without `v0=`, so `random_state` doesn't actually control the bootstrap. Either thread a per-call `v0` from the RNG or fall back to dense `eigh` (which is deterministic). The relaxed `test_deterministic` thresholds papering over this need to be tightened after the fix.
2. **Inconsistent private/public in `coherence/*.py` (§5).** Module-level helpers like `symmetrize`, `bootstrap_coherence`, `changepoint` are not underscore-prefixed despite being implementation details. Add `__all__` per file or prefix with `_`.

**Should-fix:**

3. Edge-case test for `_kfold_entry_splits` when `pool` is empty (currently produces a confusing `ValueError` from inside SRF rather than from CV).
4. Switch `meson.build` to `py.install_subdir('pysrf/coherence', ...)` (§8).
5. Tighten private-helper docstrings to match `model.py`'s numpy-style (§5).
6. Add disjointness / symmetry assertions to CV tests (§7).

**Nice-to-have:**

7. Detectability-floor regression test (§7).
8. `missing_values != np.nan` test for `cross_val_score` (§7).
9. Document that `_resolve_n_jobs(None) == cpu - 1`, not `1`, since it diverges from sklearn (§9).

After items 1 and 2 are addressed, this is PR-ready. The integration is clean enough that I'd merge it with the must-fixes done in the same branch rather than as follow-ups — both are small and the determinism issue in particular is a regression in user-facing contract that shouldn't ship at v0.1.x even briefly.
