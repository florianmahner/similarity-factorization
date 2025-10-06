#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from pysrf.bounds import estimate_p_bound, estimate_p_bound_fast


def gen_sym(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return a + a.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--method", type=str, default="dyson", choices=["dyson", "mc"])
    parser.add_argument("--omega", type=float, default=0.8)
    parser.add_argument("--eta_pmax", type=float, default=1e-3)
    parser.add_argument("--jump_frac", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--gap", type=float, default=0.05)
    parser.add_argument("--random_state", type=int, default=31213)
    parser.add_argument("--n_jobs_fast", type=int, default=1)
    args = parser.parse_args()

    S = gen_sym(args.n, args.seed)

    t0 = perf_counter()
    pmin_o, pmax_o, Sno_o = estimate_p_bound(
        S,
        method=args.method,
        omega=args.omega,
        eta_pmax=args.eta_pmax,
        jump_frac=args.jump_frac,
        tol=args.tol,
        gap=args.gap,
        random_state=args.random_state,
        verbose=False,
    )
    t1 = perf_counter()

    pmin_f, pmax_f, Sno_f = estimate_p_bound_fast(
        S,
        method=args.method,
        omega=args.omega,
        eta_pmax=args.eta_pmax,
        jump_frac=args.jump_frac,
        tol=args.tol,
        gap=args.gap,
        random_state=args.random_state,
        verbose=False,
        n_jobs=args.n_jobs_fast,
    )
    t2 = perf_counter()

    assert pmin_o == pmin_f
    assert pmax_o == pmax_f
    assert np.array_equal(Sno_o, Sno_f)

    print(f"OK n={args.n} method={args.method} t_orig={t1-t0:.3f}s t_fast={t2-t1:.3f}s")


if __name__ == "__main__":
    main()
