from __future__ import annotations

import numpy as np

from pysrf import cross_val_score, estimate_rank


def _make_low_rank_similarity(n: int, rank: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    w = np.abs(rng.standard_normal((n, rank)))
    s = w @ w.T
    e = rng.standard_normal((n, n))
    return s + noise * (e + e.T) / 2


def main():
    seed = 42
    true_rank = 8
    s = _make_low_rank_similarity(n=150, rank=true_rank, noise=0.02, seed=seed)

    estimate = estimate_rank(
        s,
        max_rank=20,
        sampling_grid=np.linspace(0.1, 0.95, 20),
        n_bootstrap=30,
        random_state=seed,
        n_jobs=-1,
    )
    ranks = range(max(1, estimate.rank - 4), estimate.rank + 5)
    curve = cross_val_score(
        s,
        ranks=ranks,
        sampling_fraction=estimate.sampling_fraction,
        n_repeats=3,
        random_state=seed,
        n_jobs=-1,
        srf_kwargs={"max_outer": 50},
    )
    cv_mean = curve.groupby("rank")["val_mse"].mean()
    cv_rank = int(cv_mean.idxmin())

    print(f"true rank:      {true_rank}")
    print(f"estimate_rank:  {estimate.rank}")
    print(f"sampling frac:  {estimate.sampling_fraction:.3f}")
    print(f"cv minimum:     {cv_rank}")
    print(cv_mean.to_string())


if __name__ == "__main__":
    main()
