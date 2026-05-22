"""Numerical equivalence tests: src/coherence/eigensolve vs v2."""
import sys
from pathlib import Path

import numpy as np
import pytest

_V2 = str(Path(__file__).resolve().parents[2] / "sandbox" / "coherence" / "kachun" / "v2")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
from _compute_coherence import (
    _topk_eigenvectors as v2_topk,
    _randomized_topk_eigenspace_symmetric as v2_randomized,
    _orthonormalize_columns as v2_orthonorm,
    _topk_eigenvectors_exact_warm as v2_warm,
)

from src.coherence.eigensolve import (
    topk_eigenvectors,
    randomized_topk_eigenspace,
    orthonormalize_columns,
    topk_eigenvectors_warm,
)


@pytest.fixture
def spd_matrix():
    rng = np.random.default_rng(123)
    a = rng.standard_normal((50, 50))
    return a @ a.T / 50


def test_topk_eigenvectors(spd_matrix):
    vals_new, vecs_new = topk_eigenvectors(spd_matrix, 5)
    vals_old, vecs_old = v2_topk(spd_matrix, 5)
    assert np.allclose(vals_new, vals_old, atol=1e-10)
    dots = np.abs(np.sum(vecs_new * vecs_old, axis=0))
    assert np.allclose(dots, 1.0, atol=1e-10)


def test_randomized_topk(spd_matrix):
    vals_new, vecs_new = randomized_topk_eigenspace(spd_matrix, 5)
    vals_old, vecs_old = v2_randomized(spd_matrix, 5)
    assert np.allclose(vals_new, vals_old, atol=1e-10)


def test_orthonormalize(spd_matrix):
    x = spd_matrix[:, :5]
    new = orthonormalize_columns(x)
    old = v2_orthonorm(x)
    assert np.allclose(new, old, atol=1e-12)


def test_warm_start(spd_matrix):
    vals_new, vecs_new, info_new = topk_eigenvectors_warm(spd_matrix, 5)
    vals_old, vecs_old, info_old = v2_warm(spd_matrix, 5)
    assert np.allclose(vals_new, vals_old, atol=1e-10)
    dots = np.abs(np.sum(vecs_new * vecs_old, axis=0))
    assert np.allclose(dots, 1.0, atol=1e-10)
