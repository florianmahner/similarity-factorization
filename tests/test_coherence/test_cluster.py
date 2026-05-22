"""Numerical equivalence tests: src/coherence/cluster vs v2."""
import sys
from pathlib import Path

import numpy as np
import pytest

_V2 = str(Path(__file__).resolve().parents[2] / "sandbox" / "coherence" / "kachun" / "v2")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
from _compute_coherence import (
    compute_trend_correlation as v2_trend_corr,
    compute_cumulative_trend_correlation as v2_cumul_trend,
)

from src.coherence.cluster import (
    compute_trend_correlation,
    compute_cumulative_trend_correlation,
)


@pytest.fixture
def bootstrap_data():
    rng = np.random.default_rng(88)
    k_count, p_count, b_count = 8, 15, 5
    iproj_boot = rng.random((k_count, p_count, b_count))
    k_list = np.arange(1, k_count + 1)
    p_list = np.linspace(0.1, 0.9, p_count)
    return iproj_boot, k_list, p_list


def test_trend_correlation(bootstrap_data):
    iproj, k_list, p_list = bootstrap_data
    new = compute_trend_correlation(iproj, k_list, p_list)
    old = v2_trend_corr(iproj, k_list, p_list)
    for key in ("pearson", "spearman", "mutual_info"):
        assert np.allclose(
            new["corr_matrices"][key], old["corr_matrices"][key], atol=1e-10,
        ), f"Mismatch on {key}"
    assert np.allclose(new["x_trend"], old["x_trend"], atol=1e-12)


def test_cumulative_trend_correlation(bootstrap_data):
    iproj, k_list, p_list = bootstrap_data
    new = compute_cumulative_trend_correlation(iproj, k_list, p_list)
    old = v2_cumul_trend(iproj, k_list, p_list)
    for key in ("pearson", "spearman", "mutual_info"):
        assert np.allclose(
            new["corr_matrices"][key], old["corr_matrices"][key], atol=1e-10,
        ), f"Mismatch on cumulative {key}"
