import numpy as np
import pytest


def test_scalar_cython_import():
    try:
        from pysrf._bsum import update_w

        assert callable(update_w)
    except ImportError:
        pytest.skip("Cython module not available")


def test_blas_cython_import():
    try:
        from pysrf._bsum import update_w_blas

        assert callable(update_w_blas)
    except ImportError:
        pytest.skip("Cython module not available")


def test_blocked_cython_import():
    try:
        from pysrf._bsum import update_w_blas_blocked

        assert callable(update_w_blas_blocked)
    except ImportError:
        pytest.skip("Cython module not available")


def test_scalar_cython_compilation():
    try:
        from pysrf._bsum import update_w
    except ImportError:
        pytest.skip("Cython module not available")

    n, r = 10, 3
    rng = np.random.default_rng(0)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))

    result = update_w(m, x0, max_iter=10, tol=1e-6)

    assert result.shape == (n, r)
    assert np.all(result >= 0)
    assert not np.any(np.isnan(result))


def test_blas_cython_compilation():
    try:
        from pysrf._bsum import update_w_blas
    except ImportError:
        pytest.skip("Cython module not available")

    n, r = 10, 3
    rng = np.random.default_rng(0)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))

    result = update_w_blas(m, x0, max_iter=10, tol=1e-6)

    assert result.shape == (n, r)
    assert np.all(result >= 0)
    assert not np.any(np.isnan(result))


def test_blocked_cython_compilation():
    try:
        from pysrf._bsum import update_w_blas_blocked
    except ImportError:
        pytest.skip("Cython module not available")

    n, r = 10, 3
    rng = np.random.default_rng(0)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))

    result = update_w_blas_blocked(m, x0, max_iter=10, tol=1e-6)

    assert result.shape == (n, r)
    assert np.all(result >= 0)
    assert not np.any(np.isnan(result))


def test_python_fallback():
    from pysrf.model import update_w

    n, r = 10, 3
    rng = np.random.default_rng(0)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))

    result = update_w(m, x0, max_iter=10, tol=1e-6)

    assert result.shape == (n, r)
    assert np.all(result >= 0)
    assert not np.any(np.isnan(result))


def test_cython_vs_python():
    try:
        from pysrf._bsum import update_w as update_w_cython
    except ImportError:
        pytest.skip("Cython module not available")

    from pysrf.model import update_w as update_w_python

    rng = np.random.default_rng(42)
    n, r = 10, 3
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))

    result_cython = update_w_cython(m, x0.copy(), max_iter=50, tol=1e-6)
    result_python = update_w_python(m, x0.copy(), max_iter=50, tol=1e-6)

    np.testing.assert_allclose(result_cython, result_python, rtol=1e-5, atol=1e-5)


def test_fallback_chain_prefers_blocked():
    try:
        from pysrf._bsum import update_w_blas_blocked  # noqa: F401
    except ImportError:
        pytest.skip("Cython module not available")

    from pysrf.model import _update_w_source

    assert _update_w_source == "cython", (
        f"Expected 'cython' but got '{_update_w_source}'"
    )
