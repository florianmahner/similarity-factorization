"""Integration test: compute_coherence produces numerically equivalent output to v2."""
import sys
from pathlib import Path

import numpy as np
import pytest

_V2 = str(Path(__file__).resolve().parents[2] / "sandbox" / "coherence" / "kachun" / "v2")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
from _compute_coherence import (
    compute_incremental_coherence_multi_k_eig_anisotropic as v2_compute,
)

from src.coherence import compute_coherence


@pytest.fixture
def small_problem():
    rng = np.random.default_rng(42)
    n = 30
    a = rng.standard_normal((n, n))
    s = a @ a.T / n
    k_list = np.array([1, 3, 5, 10])
    p_list = np.linspace(0.1, 0.9, 5)
    return s, k_list, p_list


def test_compute_coherence_equivalence(small_problem):
    """Verify new package matches v2 exactly.

    Uses atol=1e-12 because both implementations use identical RNG seeding
    and call order, so outputs should be bitwise-identical up to float noise.
    If this fails, the most likely cause is a change in function call order
    that shifts the RNG state.
    """
    s, k_list, p_list = small_problem
    new = compute_coherence(
        s, k_list, p_list, b=3, random_state=42, n_jobs=1,
        show_progress=False, compute_null=True, b_null=5,
    )
    old = v2_compute(
        s, k_list, p_list, B=3, random_state=42, n_jobs=1,
        show_progress=False, compute_null=True, B_null=5,
        visualize=False,
    )

    for key in ("Iproj_boot", "C_boot", "I_boot", "Iproj_mean", "C_mean", "I_mean"):
        assert np.allclose(new[key], old[key], atol=1e-12), f"Mismatch on {key}"

    assert np.allclose(new["evals_ref"], old["evals_ref"], atol=1e-10)
    assert np.allclose(
        new["diagnostics"]["tau_kp"],
        old["diagnostics"]["tau_kp"],
        atol=1e-10,
    )
