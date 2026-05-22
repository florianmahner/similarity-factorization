"""Numerical equivalence tests: src/coherence/_correlation vs v2."""
import sys
from pathlib import Path

import numpy as np
import pytest

_V2 = str(Path(__file__).resolve().parents[2] / "sandbox" / "coherence" / "kachun" / "v2")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
from _compute_coherence import (
    _rankdata_average_1d as v2_rankdata,
    _pearson_corr_matrix as v2_pearson_mat,
    _spearman_corr_matrix as v2_spearman_mat,
    _mutual_info_matrix as v2_mi_mat,
    _pearson_corr as v2_pearson,
    _spearman_corr as v2_spearman,
    _fd_bins as v2_fd_bins,
    _mutual_info_discrete as v2_mi_discrete,
    _mutual_info_matrix_pairwise as v2_mi_pairwise,
    _get_pair_metric as v2_get_pair_metric,
    _compute_all_trend_similarity_matrices as v2_compute_trend,
)

from src.coherence._correlation import (
    rankdata_average,
    pearson_corr_matrix,
    spearman_corr_matrix,
    mutual_info_matrix,
    pearson_corr,
    spearman_corr,
    fd_bins,
    mutual_info_discrete,
    mutual_info_matrix_pairwise,
    get_pair_metric,
    compute_trend_similarity,
)


@pytest.fixture
def trend_data():
    rng = np.random.default_rng(55)
    return rng.standard_normal((10, 20))


def test_rankdata(trend_data):
    row = trend_data[0]
    assert np.allclose(rankdata_average(row), v2_rankdata(row), atol=1e-12)


def test_pearson_matrix(trend_data):
    assert np.allclose(pearson_corr_matrix(trend_data), v2_pearson_mat(trend_data), atol=1e-12)


def test_spearman_matrix(trend_data):
    assert np.allclose(spearman_corr_matrix(trend_data), v2_spearman_mat(trend_data), atol=1e-12)


def test_mi_matrix(trend_data):
    assert np.allclose(mutual_info_matrix(trend_data), v2_mi_mat(trend_data), atol=1e-10)


def test_pearson_corr(trend_data):
    a, b = trend_data[0], trend_data[1]
    assert abs(pearson_corr(a, b) - v2_pearson(a, b)) < 1e-12


def test_spearman_corr(trend_data):
    a, b = trend_data[0], trend_data[1]
    assert abs(spearman_corr(a, b) - v2_spearman(a, b)) < 1e-12


def test_fd_bins(trend_data):
    assert fd_bins(trend_data[0]) == v2_fd_bins(trend_data[0])


def test_mi_discrete(trend_data):
    a, b = trend_data[0], trend_data[1]
    assert abs(mutual_info_discrete(a, b) - v2_mi_discrete(a, b)) < 1e-10


def test_mi_pairwise(trend_data):
    assert np.allclose(
        mutual_info_matrix_pairwise(trend_data),
        v2_mi_pairwise(trend_data),
        atol=1e-10,
    )


def test_compute_trend_similarity(trend_data):
    new = compute_trend_similarity(trend_data)
    old = v2_compute_trend(trend_data)
    for key in ("pearson", "spearman", "mutual_info"):
        assert np.allclose(new[key], old[key], atol=1e-10)


def test_get_pair_metric():
    fn_new, range_new = get_pair_metric("spearman")
    fn_old, range_old = v2_get_pair_metric("spearman")
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([4.0, 3.0, 2.0, 1.0])
    assert abs(fn_new(a, b) - fn_old(a, b)) < 1e-12
    assert range_new == range_old
