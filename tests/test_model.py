import numpy as np
import pytest
from pysrf import SRF


def generate_symmetric_matrix(n, rank, noise_level=0.1, random_state=None):
    rng = np.random.RandomState(random_state)
    w_true = rng.rand(n, rank)
    s = w_true @ w_true.T
    s += noise_level * rng.randn(n, n)
    s = (s + s.T) / 2
    return s, w_true


def test_srf_initialization():
    model = SRF(rank=5, random_state=42)
    assert model.rank == 5
    assert model.rho == 3.0
    assert model.max_outer == 10
    assert model.max_inner == 30
    assert model.tol == 1e-4


def test_srf_fit_complete_data():
    np.random.seed(42)
    n, rank = 20, 5
    s, _ = generate_symmetric_matrix(n, rank, noise_level=0.01, random_state=42)

    model = SRF(rank=rank, max_outer=20, random_state=42)
    model.fit(s)

    assert hasattr(model, "w_")
    assert model.w_.shape == (n, rank)
    assert hasattr(model, "components_")
    assert hasattr(model, "n_iter_")
    assert hasattr(model, "history_")


def test_srf_fit_transform():
    np.random.seed(42)
    n, rank = 20, 5
    s, _ = generate_symmetric_matrix(n, rank, noise_level=0.01, random_state=42)

    model = SRF(rank=rank, max_outer=20, random_state=42)
    w = model.fit_transform(s)

    assert w.shape == (n, rank)
    assert np.all(w >= 0)


def test_srf_reconstruct():
    np.random.seed(42)
    n, rank = 20, 5
    s, _ = generate_symmetric_matrix(n, rank, noise_level=0.01, random_state=42)

    model = SRF(rank=rank, max_outer=20, random_state=42)
    model.fit(s)
    s_hat = model.reconstruct()

    assert s_hat.shape == s.shape
    assert np.allclose(s_hat, s_hat.T)


def test_srf_missing_data():
    np.random.seed(42)
    n, rank = 20, 5
    s, _ = generate_symmetric_matrix(n, rank, noise_level=0.01, random_state=42)

    mask = np.random.rand(n, n) < 0.3
    mask = mask | mask.T
    s[mask] = np.nan

    model = SRF(rank=rank, max_outer=20, missing_values=np.nan, random_state=42)
    model.fit(s)

    assert hasattr(model, "w_")
    assert model.w_.shape == (n, rank)


def test_srf_score():
    np.random.seed(42)
    n, rank = 20, 5
    s, _ = generate_symmetric_matrix(n, rank, noise_level=0.01, random_state=42)

    model = SRF(rank=rank, max_outer=20, random_state=42)
    model.fit(s)
    score = model.score(s)

    assert isinstance(score, float)
    assert score >= 0


def test_srf_with_bounds():
    np.random.seed(42)
    n, rank = 20, 5
    s, _ = generate_symmetric_matrix(n, rank, noise_level=0.01, random_state=42)
    s = np.clip(s, 0, 1)

    mask = np.random.rand(n, n) < 0.2
    mask = mask | mask.T
    s[mask] = np.nan

    model = SRF(
        rank=rank, max_outer=20, bounds=(0, 1), missing_values=np.nan, random_state=42
    )
    model.fit(s)

    assert hasattr(model, "w_")
    reconstruction = model.reconstruct()
    assert np.all(reconstruction >= 0)
    assert np.min(reconstruction) >= 0
    assert np.max(reconstruction) <= 1.1


def test_srf_invalid_inputs():
    model = SRF(rank=-1)
    with pytest.raises(ValueError):
        model.fit(np.eye(10))

    model = SRF(rank=5, rho=-1)
    with pytest.raises(ValueError):
        model.fit(np.eye(10))

    model = SRF(rank=5, bounds=(1, 0))
    with pytest.raises(ValueError):
        model.fit(np.eye(10))


def test_srf_all_missing():
    n = 10
    s = np.full((n, n), np.nan)

    model = SRF(rank=5, missing_values=np.nan)
    with pytest.raises(ValueError):
        model.fit(s)


def test_srf_different_init_methods():
    np.random.seed(42)
    n, rank = 20, 5
    s, _ = generate_symmetric_matrix(n, rank, noise_level=0.01, random_state=42)

    for init in ["random", "random_sqrt", "nndsvd", "nndsvdar", "eigenspectrum"]:
        model = SRF(rank=rank, max_outer=10, init=init, random_state=42)
        model.fit(s)
        assert hasattr(model, "w_")
        assert model.w_.shape == (n, rank)
