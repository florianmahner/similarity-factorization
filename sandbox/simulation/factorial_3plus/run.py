"""Factorial design with only 3-4 level factors to verify LOO works."""

from __future__ import annotations

import itertools as it
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pysrf import SRF

from src.tools.rsa import mantel_test, loo_alignment_test_multi, global_alignment_test_multi
from src.utils.helpers import add_positive_noise_with_snr
from src.utils import get_output_dir

OUTPUT_DIR = get_output_dir()

LEVELS = {
    "shape": ["circle", "square", "triangle"],      # 3 levels
    "color": ["red", "green", "blue"],              # 3 levels
    "size": ["small", "medium", "large", "xlarge"], # 4 levels
}

SNRS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_REPEATS = 50
N_PERM = 200


def create_factorial(levels: dict[str, list[str]]) -> tuple[np.ndarray, dict]:
    items = list(it.product(*levels.values()))
    features, col_ranges = [], {}
    col = 0
    for i, (factor, lvls) in enumerate(levels.items()):
        idx = [lvls.index(item[i]) for item in items]
        features.append(np.eye(len(lvls))[idx])
        col_ranges[factor] = (col, col + len(lvls))
        col += len(lvls)
    return np.hstack(features), col_ranges


def run_single(x, col_ranges, snr, n_perm, seed):
    rng_seed = 10000 + 97 * (seed + 1)
    x_noisy = add_positive_noise_with_snr(x, snr, rng=rng_seed)
    s = x_noisy @ x_noisy.T

    w = SRF(rank=x.shape[1], verbose=False, tol=0.0, random_state=seed).fit_transform(s)

    # RSA (two-sided)
    rsa_ps, rsa_rs = [], []
    for i in range(x.shape[1]):
        h = x[:, [i]] @ x[:, [i]].T
        p, _, r = mantel_test(h, s, permutations=n_perm, random_state=rng_seed + i, two_sided=True)
        rsa_ps.append(p)
        rsa_rs.append(r)

    # SRF methods
    srf_loo = loo_alignment_test_multi(w, x, permutations=n_perm, random_state=rng_seed + 1000)
    srf_global = global_alignment_test_multi(w, x, permutations=n_perm, random_state=rng_seed + 2000)

    rows = []
    for col in range(len(rsa_ps)):
        factor = next(f for f, (s, e) in col_ranges.items() if s <= col < e)
        n_levels = col_ranges[factor][1] - col_ranges[factor][0]
        base = {"snr": snr, "repeat": seed, "factor": factor, "column": col, "n_levels": n_levels}
        rows.append({**base, "method": "RSA", "r_obs": rsa_rs[col], "raw_p": rsa_ps[col]})
        rows.append({**base, "method": "SRF-LOO", "r_obs": srf_loo["r_obs"][col], "raw_p": srf_loo["raw_p"][col]})
        rows.append({**base, "method": "SRF-Global", "r_obs": srf_global["r_obs"][col], "raw_p": srf_global["raw_p"][col]})
    return rows


def main():
    x, col_ranges = create_factorial(LEVELS)
    print(f"Factorial: {x.shape[0]} items, {x.shape[1]} columns")
    print(f"Factors: {', '.join(f'{k} ({v[1]-v[0]} levels)' for k, v in col_ranges.items())}")

    rows = []
    for snr in SNRS:
        print(f"SNR={snr:.2f}")
        results = Parallel(n_jobs=-1)(
            delayed(run_single)(x, col_ranges, snr, N_PERM, seed)
            for seed in range(N_REPEATS)
        )
        for r in results:
            rows.extend(r)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    # Quick summary
    from statsmodels.stats.multitest import multipletests

    def add_fdr(g):
        g = g.copy()
        _, corrected, _, _ = multipletests(g["raw_p"], alpha=0.05, method="fdr_bh")
        g["significant"] = corrected < 0.05
        return g

    df = df.groupby(["snr", "repeat", "method"], group_keys=False).apply(add_fdr)

    print("\nPower by factor at SNR=1.0:")
    snr1 = df[df["snr"] == 1.0]
    by_factor = snr1.groupby(["factor", "method"])["significant"].mean().unstack() * 100
    cols = [c for c in ["RSA", "SRF-LOO", "SRF-Global"] if c in by_factor.columns]
    print(by_factor[cols].round(1).to_string())


if __name__ == "__main__":
    main()
