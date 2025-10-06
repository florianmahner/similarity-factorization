import numpy as np
import pytest

from pysrf.bounds import (
    estimate_p_bound,
    estimate_p_bound_fast,
    estimate_p_bound_ultra,
)


def gen_sym(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return a + a.T


@pytest.mark.parametrize("n,seed", [(32, 123), (48, 777)])
@pytest.mark.parametrize("method", ["dyson", "mc"])
def test_ultra_matches_existing(n, seed, method):
    S = gen_sym(n, seed)
    args = dict(
        method=method,
        omega=0.8,
        eta_pmax=1e-3,
        jump_frac=0.1,
        tol=1e-4,
        gap=0.05,
        random_state=31213,
        verbose=False,
    )

    pmin_o, pmax_o, Sno_o = estimate_p_bound(S, **args)
    pmin_f, pmax_f, Sno_f = estimate_p_bound_fast(S, **args, n_jobs=2)
    pmin_u, pmax_u, Sno_u = estimate_p_bound_ultra(S, **args, n_jobs=2)

    assert pmin_o == pmin_f
    assert pmax_o == pmax_f
    assert np.array_equal(Sno_o, Sno_f)

    assert pmin_u == pmin_f
    assert pmax_u == pmax_f
    assert np.array_equal(Sno_u, Sno_f)
