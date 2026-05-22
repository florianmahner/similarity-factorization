"""Numerical equivalence tests: src/coherence/analysis vs v2."""
import sys
from pathlib import Path

import numpy as np
import pytest

_V2 = str(Path(__file__).resolve().parents[2] / "sandbox" / "coherence" / "kachun" / "v2")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
from _compute_coherence import (
    _baseline_correct_array as v2_baseline,
    _bh_fdr_reject as v2_bh_fdr,
    _estimate_kappa_hat as v2_kappa,
    _smooth_median as v2_smooth,
    kappa_changepoint as v2_changepoint,
    _find_first_run_ge_threshold as v2_first_run,
)

from src.coherence.analysis import (
    baseline_correct,
    bh_fdr,
    estimate_kappa,
    smooth_median,
    kappa_changepoint,
    find_first_run_above,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_baseline_correct(rng):
    arr = rng.random((5, 20, 10))
    k_list = np.array([1, 5, 10, 20, 30])
    n = 100
    new = baseline_correct(arr, k_list, n)
    old = v2_baseline(arr, k_list, n)
    assert np.allclose(new, old, atol=1e-12)


def test_bh_fdr(rng):
    pvals = rng.random(20)
    assert np.array_equal(bh_fdr(pvals, 0.05), v2_bh_fdr(pvals, 0.05))


def test_estimate_kappa(rng):
    i_med = rng.random((10, 20))
    p_list = np.linspace(0.05, 0.95, 20)
    new_k, new_info = estimate_kappa(i_med, p_list, 0.85)
    old_k, old_info = v2_kappa(i_med, p_list, 0.85)
    assert np.allclose(new_k, old_k, atol=1e-12)


def test_smooth_median(rng):
    x = rng.random(30)
    assert np.allclose(smooth_median(x, 5), v2_smooth(x, 5), atol=1e-12)


def test_kappa_changepoint(rng):
    kappa = np.sort(rng.random(20))
    k_list = np.arange(1, 21)
    new_k, _ = kappa_changepoint(kappa, k_list)
    old_k, _ = v2_changepoint(kappa, k_list)
    assert new_k == old_k


def test_find_first_run_above():
    p_list = np.linspace(0.1, 1.0, 10)
    curve = np.array([0.0, 0.0, 0.5, 0.9, 0.95, 0.97, 0.98, 0.99, 0.99, 1.0])
    new = find_first_run_above(p_list, curve, 0.9, run=2)
    old = v2_first_run(p_list, curve, 0.9, run=2)
    assert new == old
