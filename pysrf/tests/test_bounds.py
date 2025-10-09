import numpy as np
import pytest
from pysrf.bounds import (
    pmin_bound,
    p_upper_only_k,
    estimate_sampling_bounds,
    estimate_sampling_bounds_fast,
    lambda_bulk_dyson_raw,
)
from pysrf import SRF, cross_val_score
from pysrf.cross_validation import mask_missing_entries


def generate_test_matrix(n=30, rank=5, random_state=42):
    rng = np.random.RandomState(random_state)
    w = rng.rand(n, rank)
    s = w @ w.T
    s = (s + s.T) / 2
    return s


def test_pmin_bound():
    s = generate_test_matrix(n=30, rank=5)

    pmin, _, _, _, _ = pmin_bound(s, random_state=42, verbose=False)

    assert isinstance(pmin, (float, np.floating))
    assert 0 <= pmin <= 1


def test_pmin_bound_verbose():
    s = generate_test_matrix(n=30, rank=5)

    pmin, pmin_bern, pmin_lower, pmin_alt, mc_norms = pmin_bound(
        s, random_state=42, verbose=False
    )

    assert isinstance(pmin, (float, np.floating))
    assert isinstance(pmin_bern, (float, np.floating))
    assert isinstance(pmin_lower, (float, np.floating))
    assert isinstance(pmin_alt, (float, np.floating))
    assert isinstance(mc_norms, np.ndarray)


def test_lambda_bulk_dyson_raw():
    s = generate_test_matrix(n=30, rank=5)

    p = 0.5
    edge = lambda_bulk_dyson_raw(s, p)

    assert isinstance(edge, float)
    assert edge >= 0


def test_lambda_bulk_dyson_raw_edge_cases():
    s = generate_test_matrix(n=30, rank=5)

    edge_low = lambda_bulk_dyson_raw(s, 0.0)
    assert edge_low == 0.0

    edge_high = lambda_bulk_dyson_raw(s, 1.0)
    assert edge_high == 0.0


def test_p_upper_only_k():
    s = generate_test_matrix(n=30, rank=5)

    k = 5
    pmax = p_upper_only_k(s, k=k, method="dyson", verbose=False, seed=42)

    assert isinstance(pmax, (float, np.floating))
    assert 0 <= pmax <= 1


def test_p_upper_only_k_invalid_k():
    s = generate_test_matrix(n=30, rank=5)

    with pytest.raises(ValueError):
        p_upper_only_k(s, k=0)

    with pytest.raises(ValueError):
        p_upper_only_k(s, k=100)


def test_estimate_sampling_bounds():
    s = generate_test_matrix(n=30, rank=5)

    pmin, pmax, s_noise = estimate_sampling_bounds(s, verbose=False, random_state=42)

    assert isinstance(pmin, (float, np.floating))
    assert isinstance(pmax, (float, np.floating))
    assert isinstance(s_noise, np.ndarray)
    assert s_noise.shape == s.shape


def test_estimate_sampling_bounds_fast():
    s = generate_test_matrix(n=30, rank=5)

    pmin, pmax, s_noise = estimate_sampling_bounds_fast(
        s, verbose=False, random_state=42, n_jobs=1
    )

    assert isinstance(pmin, (float, np.floating))
    assert isinstance(pmax, (float, np.floating))
    assert isinstance(s_noise, np.ndarray)
    assert s_noise.shape == s.shape


def test_estimate_sampling_bounds_parameters():
    s = generate_test_matrix(n=30, rank=5)

    pmin, pmax, s_noise = estimate_sampling_bounds(
        s,
        gamma=1.1,
        eta=0.1,
        rho=0.9,
        method="dyson",
        omega=0.85,
        verbose=False,
        random_state=42,
    )

    assert isinstance(pmin, (float, np.floating))
    assert isinstance(pmax, (float, np.floating))


def test_rank_recovery_with_cv():
    """Test that CV with sampling bound estimates produces reasonable rank selection."""
    n, k, seed = 200, 10, 42

    s = generate_test_matrix(n=n, rank=k, random_state=seed)
    rank_range = list(range(max(1, k - 5), k + 5))

    grid = cross_val_score(
        s,
        estimator=SRF(rho=3.0, max_outer=50, max_inner=20, random_state=seed),
        param_grid={"rank": rank_range},
        n_repeats=10,
        estimate_sampling_fraction=True,
        sampling_selection="mean",
        n_jobs=1,
        random_state=seed,
        verbose=0,
        fit_final_estimator=False,
    )

    best_rank = grid.best_params_["rank"]

    assert best_rank >= 7, "CV should select a reasonable rank (at least 3)"
    assert best_rank <= k + 1, f"CV should not overestimate rank too much"
    assert grid.best_score_ < 1.0, "Best CV score should indicate good fit"


def test_sampling_bounds_scale_with_rank():
    """Test that sampling bounds change appropriately with matrix rank."""
    n = 80
    seed = 42

    s_low_rank = generate_test_matrix(n=n, rank=5, random_state=seed)
    s_high_rank = generate_test_matrix(n=n, rank=20, random_state=seed)

    pmin_low, pmax_low, _ = estimate_sampling_bounds_fast(
        s_low_rank, random_state=seed, verbose=False, n_jobs=1
    )
    pmin_high, pmax_high, _ = estimate_sampling_bounds_fast(
        s_high_rank, random_state=seed, verbose=False, n_jobs=1
    )

    assert pmin_low < 1.0 and pmax_low < 1.0
    assert pmin_high < 1.0 and pmax_high < 1.0

    assert (
        pmax_high > pmax_low
    ), "Higher rank matrices should need higher sampling rates"


def test_reconstruction_quality_at_estimated_bounds():
    """Test that reconstruction at estimated sampling rate recovers structure."""
    true_rank = 6
    n = 60
    seed = 42

    s = generate_test_matrix(n=n, rank=true_rank, random_state=seed)

    pmin, pmax, _ = estimate_sampling_bounds_fast(
        s, random_state=seed, verbose=False, n_jobs=1
    )

    sampling_fraction = 0.5 * (pmin + pmax)

    rng = np.random.RandomState(seed)
    missing_mask = mask_missing_entries(s, sampling_fraction, rng, np.nan)

    s_masked = s.copy()
    s_masked[missing_mask] = np.nan

    model = SRF(rank=true_rank, max_outer=20, random_state=seed)
    model.fit(s_masked)
    s_reconstructed = model.reconstruct()

    observed_mask = ~missing_mask
    mse_observed = np.mean((s[observed_mask] - s_reconstructed[observed_mask]) ** 2)

    assert (
        mse_observed < 0.1
    ), "Reconstruction should be accurate at estimated sampling rate"
