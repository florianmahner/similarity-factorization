# Design review: rank-estimation pipeline integration into `pysrf`

This review covers the proposed integration of `RankEstimator` / `EntryKFold` /
`cross_val_score` into `pysrf`, replacing the current `coherence.estimate_rank()`,
`bounds.py`, and most of the existing `cross_validation.py`. I read all the files
in the requested order; my comments below are concrete and sharp where warranted.

A short summary up front so the rest can be skimmed: the design is good and
substantially cleaner than what it replaces, but it is over-engineered in a
couple of specific places. The 4-file `coherence/` subpackage is real overkill
for ~1000 lines of code with a single public class. The stub for
`cross_validation.py` also re-exports `GridSearchCV` and `fit_and_score` which
no longer have any reason to exist after the rewrite. Below, question by question.

---

## 1. Is the API surface minimal?

Stated public surface: `RankEstimator`, `EntryKFold`, `cross_val_score`, plus
`SRF` and the consensus things you already had. That part is good — it's a real
reduction from the previous surface (`estimate_rank`, `GridSearchCV`,
`EntryMaskSplit`, `create_train_val_split`, `fit_and_score`,
`estimate_sampling_bounds`, `estimate_sampling_bounds_fast`, `pmin_bound`,
`p_upper_only_k`). Roughly 9 names collapse to 3. That's the right direction.

**But the proposed `__init__.py` still exports `GridSearchCV` and
`fit_and_score`.** Read the stub at `proposed_pysrf_structure/pysrf/__init__.py`
lines 11–16: those are still on the public surface even though your prose says
they aren't. The sandbox `cross_validation.py` doesn't even define them anymore
— it has just `EntryKFold`, `_fit_score_one` (private), and `cross_val_score`.
So either the README is right and the stub `__init__` is stale, or the stub is
right and the surface isn't actually 3 names. I'd strongly suggest dropping
both — `GridSearchCV` was a pysrf-specific clone of a sklearn class that the
real `cross_val_score` no longer uses, and `fit_and_score` is a one-line helper
no caller will ever need. **Real public surface should be exactly 3 names.**

Beyond that: `EntryKFold` and `cross_val_score` overlap a lot. See (4).

---

## 2. Is the 4-file `coherence/` subpackage justified?

**No, not at the current size.** The old `coherence.py` was 602 lines of one
file with a clean "Layer 1/2/3/4/5" comment section pattern. The new code is
similar size (~600 lines across the four files) and does substantively the same
work. Splitting it into `_bootstrap.py`, `_rank_selection.py`,
`_sampling_fraction.py`, `estimator.py` buys you very little:

- All four files import from each other in a single linear chain
  (estimator → all three). There are no alternative consumers.
- All "private" helpers (`_top_eigenpairs`, `_bernoulli_replicate`,
  `_segment_sse`, `_f_statistic`, `_monotone_decreasing`, ...) are still
  private to their file. Splitting doesn't widen any contract.
- The three "phase" modules each export 2–4 functions that are called from
  exactly one place: `estimator.py`. So the split adds 3 import statements
  and 3 file-headers for zero re-use benefit.
- The `__init__.py` re-exports a single name. That's a strong code-smell
  for "this should be a module, not a package."

The single-file convention in pysrf elsewhere (`model.py` is 716 lines,
`consensus.py` is similar) is the right precedent. **Recommendation: keep this
as `coherence.py`, one file, with the same Layer 1/2/3/4 comment-separator
pattern the old file used.** When does a subpackage make sense? When (a) there
are multiple independent public entrypoints from the package, or (b) the files
have meaningfully different dependency surfaces (e.g., one needs `numba`, one
needs `scipy.sparse`), or (c) sub-modules will plausibly be tested or imported
independently. None of those apply here.

A reasonable middle ground if you genuinely want some structure: `coherence.py`
single file with internal `# --- Bootstrap ---`, `# --- Rank selection ---`,
`# --- Sampling fraction ---`, `# --- Estimator ---` banners. That's what the
current code already does, and it works.

---

## 3. Is `cross_val_score`'s auto-fit of `RankEstimator` the right magic level?

The stub signature has `sampling_fraction: float | None = None` AND
`estimate_sampling_fraction: bool | dict = False`. That's two ways to do the
same thing and is more confusing than the original. Reading the implementation
in `sandbox/cross_validation.py`, the actual function is much cleaner: it just
takes `sampling_fraction: float` (required, positional). No auto-fit, no flag.

**The actual sandbox code is the right design. The stub is regressing.** Keep
`cross_val_score` honest:

```python
def cross_val_score(s, ranks, sampling_fraction, n_folds=5, ...):
```

If a user wants to auto-fit, they write:

```python
est = RankEstimator().fit(s)
curve = cross_val_score(s, ranks=[...], sampling_fraction=est.sampling_fraction_)
```

That's two lines of code, fully explicit, and removes the most surprising
behavior: silently re-running a non-trivial bootstrap (~20 reps × 20 grid
points × top-k eigh) inside what looks like a cheap convenience function. The
auto-fit also makes the result non-deterministic w.r.t. `random_state` in a
sneaky way (which seed flows where?), and it duplicates the eigendecomp the
user almost certainly already did.

Concretely: **drop `estimate_sampling_fraction` and `rank_estimator=` from the
signature.** Make `sampling_fraction` required. The README in the prompt
already shows the user calling `RankEstimator` explicitly; the docstring of
`RankEstimator` (estimator.py L139–144) does the same. The auto-fit branch
serves no audience.

---

## 4. Is `EntryKFold` worth exposing?

This is the most interesting question. Look at the actual code in
`sandbox/cross_validation.py`:

- `EntryKFold` yields `(train_mask, val_mask)` (boolean ndarrays, not
  indices).
- `cross_val_score` is the only caller. It builds the splits once
  (`splits = list(cv.split(s))`) and passes them to `_fit_score_one`.
- sklearn's `GridSearchCV` cannot use `EntryKFold` because it expects
  index arrays, not masks. So the supposed "sklearn pipeline use" doesn't
  actually work.

So `EntryKFold` is an exported class with exactly one consumer
(`cross_val_score`) and zero possibility of plugging into the broader sklearn
ecosystem (mask vs. index convention, plus the symmetric pre-mask logic).

**Recommendation: don't expose it.** Move the splitter logic into a private
function — even a simple iterator `_kfold_entry_splits(s, sampling_fraction,
n_folds, ...)` is enough. The class form (`BaseCrossValidator` subclass) is
sklearn cosplay that confers no real interop.

If you do keep it exposed (e.g., for someone writing a custom training loop
with a different estimator than `SRF`), at least be honest in the docstring:
"This splitter is for use with `pysrf.cross_val_score`. It is not compatible
with `sklearn.model_selection.cross_val_score` because it yields masks rather
than indices." Currently neither the stub nor the sandbox code says this and a
sklearn-fluent user will burn an hour figuring it out.

---

## 5. sklearn-convention compliance

`RankEstimator(BaseEstimator)`: looks correct. `fit(x, y=None)` signature,
trailing-underscore fitted attributes (`rank_`, `sampling_fraction_`,
`eigenvalues_`, `leakage_`, ...), `n_features_in_` set in `fit`,
`_parameter_constraints` populated, `_validate_params()` called, `check_is_fitted`
in `cv_sampling_fraction`. Good citizens. One nit: there's no `transform` or
`predict`, so it's a "pure estimator", which is fine — sklearn allows that, but
in that case you might consider inheriting from `TransformerMixin` only if
`transform()` would return something useful (e.g., the rank as a scalar, or the
eigenvalues). I think it's fine as-is.

`EntryKFold(BaseCrossValidator)`: this is the real violation. Sklearn's
`BaseCrossValidator.split()` is documented to yield `(train_idx, test_idx)` —
1-D arrays of integer row indices. You yield two `(n, n)` boolean masks. That
breaks every sklearn-aware caller (`cross_validate`, `cross_val_score`,
`GridSearchCV`, `RFECV`, anything from `sklearn.model_selection`). It also
breaks the implied contract that `get_n_splits` returns the number of splits
and `split` yields that many — that part you get right.

**You either need a different base class or a sklearn-shaped interface.** The
cleanest fix is to not inherit from `BaseCrossValidator` at all (it confers no
behavior; it's just a marker), and either (a) inherit from nothing and document
your protocol, or (b) collapse the splitter to a generator function. Option (b)
goes with my recommendation in (4).

Other minor sklearn violations:

- `cross_val_score` returns a `pd.DataFrame`. sklearn's same-named function
  returns a numpy array. This is a name collision risk — `from pysrf import
  cross_val_score` will shadow `from sklearn.model_selection import
  cross_val_score` and the return types differ. Either rename
  (`rank_cv_curve`, `cv_rank_score`, ...) or be very explicit in the
  docstring. The current name is a footgun.

---

## 6. Could this collapse to ONE function?

A live possibility, but **no, I don't think so, and the reason is the CV use
case.** Consider:

```python
def select_rank(s) -> dict:
    return {"rank": ..., "sampling_fraction": ..., "cv_curve": ...}
```

This collapses the spectrum-based estimator and the CV in one shot. It would
be nice if rank estimation were a single concept — and the bootstrap result
IS, by itself, a rank estimate. But the CV is an *independent verification*
that does completely different work (fitting SRF at multiple ranks, measuring
held-out reconstruction loss). They share `sampling_fraction` but nothing
else. Fusing them into one function would:

- Force every caller who already has a `sampling_fraction` (e.g., from
  domain knowledge, or a previous run) to skip the bootstrap.
- Make the CV's `ranks=` argument needed up front, but the natural
  ranks-to-sweep depend on `rank_` from the bootstrap (e.g., `[k-2, k,
  k+2]`). So you'd either run the bootstrap to choose `ranks`, or accept a
  ranks-selection heuristic, both of which add complexity.
- Hide the fact that bootstrap is one method, CV is another, and they
  can disagree (which is when you learn something).

So **two functions stays right**, and the boundary between them is the right
boundary. What CAN collapse is the class vs. function question (see 7) and the
removal of magic auto-fit (see 3).

A version of the collapse that I would endorse: make `RankEstimator` a
function `estimate_rank(s, **kwargs) -> RankResult` returning a dataclass with
the same fields you currently expose as attributes. That keeps the two-step
shape but drops the sklearn-estimator overhead. See (7).

---

## 7. Other simplifications

**`RankEstimator` doesn't need to be a sklearn estimator.** This is the
sharpest version of the same point. What does the `BaseEstimator` inheritance
buy you?

- `get_params` / `set_params` — for sklearn pipelines and `GridSearchCV`.
  Nobody is going to grid-search hyperparameters of the bootstrap.
- `clone()` — same use case, irrelevant here.
- `_parameter_constraints` validation — nice but trivially replaced by
  inline checks at function entry.
- `check_is_fitted` — used only to guard `cv_sampling_fraction`, which
  goes away if it's a free function.

What you actually want is "a thing that takes `s` and returns rank, sampling
fraction, and diagnostics". That's a function returning a dataclass:

```python
@dataclass(frozen=True)
class RankEstimate:
    rank: int
    sampling_fraction: float
    eigenvalues: np.ndarray
    leakage: np.ndarray
    sampling_grid: np.ndarray
    recovery_raw: np.ndarray
    recovery_monotone: np.ndarray
    detectability_floor: float

    def cv_sampling_fraction(self, n_folds: int, n_pairs: int) -> float:
        ...

def estimate_rank(s, recovery_tolerance=0.10, ...) -> RankEstimate:
    ...
```

This is **less code, less ceremony, exactly as discoverable, and
deterministic**. The user does:

```python
est = estimate_rank(s)
curve = cross_val_score(s, ranks=[est.rank-2, est.rank, est.rank+2],
                        sampling_fraction=est.sampling_fraction)
```

Same ergonomics, no `fit()`, no `_validate_params`, no `check_is_fitted`, no
`BaseEstimator`. The dataclass is reproducible and trivially
pickle/JSON-serializable for caching. I'd push for this.

(The one tradeoff: `cv_sampling_fraction` needs `n_pairs` since you can't
store `n_features_in_` on a free function. Easy — pass it. Or make it a free
function `inflated_outer_mask(sampling_fraction, n_folds, n)` and have the
dataclass not carry the method at all.)

**`cv_sampling_fraction` is fine as a method/standalone, but the cap logic
duplicates between `RankEstimator._cv_cap` and `cross_validation._adaptive_cap`.**
The sandbox code has both implementations of the same `max(0.95, 1 - 2000/N)`
rule. Move that to one place. If you keep the class, the method should defer
to the same helper that `EntryKFold` uses (or vice versa). Currently they're
copy-pasted with the same magic constants (`_CV_CAP_FLOOR = 0.95`,
`_CV_HOLDOUT_BUDGET = 2000`), which is exactly how those constants drift apart
six months from now.

**The reproduce-reference and cv-diagnostics sandboxes have already validated
this code.** Good — that means the integration risk is low and the design
choices above are about long-term ergonomics, not correctness.

---

## Bottom line

The integration is a clear win versus the status quo (smaller surface, removes
`bounds.py`, removes overlapping CV machinery). Three concrete pushes to
simplify further:

1. **Kill the 4-file `coherence/` subpackage.** Make it `coherence.py`, one
   file. Three internal "Layer X" banners. The split adds no reuse, no
   testability, no clarity. It's organizational theatre.
2. **Drop the auto-fit in `cross_val_score` and drop `GridSearchCV` /
   `fit_and_score` from the public API.** The stub `__init__.py` doesn't
   match the README's promise; the sandbox code is closer to right. Make
   `sampling_fraction` a required positional, document the explicit two-step
   pattern, done.
3. **Reconsider whether `RankEstimator` should be a sklearn estimator at
   all.** A function returning a frozen dataclass is honest, less code,
   equally usable, and avoids the `EntryKFold(BaseCrossValidator)` mask-vs-
   index protocol violation downstream. If you keep the class form, at
   minimum (a) don't subclass `BaseCrossValidator` for `EntryKFold` (it's
   not sklearn-compatible anyway), and (b) deduplicate the
   `_adaptive_cap` constants between the estimator and the splitter.

If you only do one of these, do #1 — it's the highest-cost / lowest-value
piece of the proposal. If you do all three, the public API collapses to
`SRF`, `estimate_rank`, `cross_val_score` and the implementation drops
~30% of its code volume without losing capability. That is, I think, the
design you actually want.
