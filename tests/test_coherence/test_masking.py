"""Numerical equivalence tests: src/coherence/masking vs v2 _compute_coherence."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Import v2 originals via sys.path (temporary, for testing only)
# In a git worktree, sandbox/ is untracked and lives only in the main repo.
# Walk up from the worktree's .git file to find the main repo root.
def _find_main_repo_root() -> Path:
    worktree_root = Path(__file__).resolve().parents[2]
    git_path = worktree_root / ".git"
    if git_path.is_file():
        # .git is a file in a worktree: "gitdir: <path>"
        gitdir = Path(git_path.read_text().split(":", 1)[1].strip())
        commondir_file = gitdir / "commondir"
        if commondir_file.exists():
            common = (gitdir / commondir_file.read_text().strip()).resolve()
            return common.parent
    return worktree_root

_V2 = str(_find_main_repo_root() / "sandbox" / "coherence" / "kachun" / "v2")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)
from _compute_coherence import (
    _symmetrize_with_nan as v2_symmetrize,
    _prepare_observation_mask as v2_prepare_mask,
    _masked_unbiased_spd_missing as v2_masked_bootstrap,
    _masked_unbiased_spd_missing_from_uniform as v2_masked_from_uniform,
)

from src.coherence.masking import (
    symmetrize_with_nan,
    prepare_observation_mask,
    masked_bootstrap_sample,
    masked_bootstrap_from_uniform,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sim_with_nan(rng):
    """30x30 similarity matrix with ~20% NaN off-diagonal."""
    n = 30
    s = rng.standard_normal((n, n))
    s = s @ s.T / n
    mask = rng.random((n, n)) < 0.2
    np.fill_diagonal(mask, False)
    s[mask] = np.nan
    return s


def test_symmetrize_with_nan(sim_with_nan):
    new = symmetrize_with_nan(sim_with_nan)
    old = v2_symmetrize(sim_with_nan)
    assert np.allclose(new, old, atol=1e-12, equal_nan=True)


def test_prepare_observation_mask(sim_with_nan):
    s_sym = symmetrize_with_nan(sim_with_nan)
    s0_new, w_new, q_new = prepare_observation_mask(s_sym)
    s0_old, w_old, q_old = v2_prepare_mask(s_sym)
    assert np.allclose(s0_new, s0_old, atol=1e-12)
    assert np.allclose(w_new, w_old, atol=1e-12)
    assert abs(q_new - q_old) < 1e-12


def test_masked_bootstrap_sample(sim_with_nan):
    s_sym = symmetrize_with_nan(sim_with_nan)
    s0, w, q = prepare_observation_mask(s_sym)
    p = 0.5
    rng_new = np.random.default_rng(99)
    rng_old = np.random.default_rng(99)
    new = masked_bootstrap_sample(s0, w, q, p, rng_new)
    old = v2_masked_bootstrap(s0, w, q, p, rng_old)
    assert np.allclose(new, old, atol=1e-12)


def test_masked_bootstrap_from_uniform(sim_with_nan):
    s_sym = symmetrize_with_nan(sim_with_nan)
    s0, w, q = prepare_observation_mask(s_sym)
    n = s0.shape[0]
    iu = np.triu_indices(n, k=1)
    rng = np.random.default_rng(77)
    edge_u = rng.random(iu[0].size)
    p = 0.4
    new = masked_bootstrap_from_uniform(s0, w, p, edge_u, iu)
    old = v2_masked_from_uniform(s0, w, p, edge_u, iu)
    assert np.allclose(new, old, atol=1e-12)
