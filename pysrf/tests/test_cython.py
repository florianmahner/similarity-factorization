import numpy as np
import pytest


def test_cython_import():
    try:
        from pysrf._bsum import update_w

        assert callable(update_w)
    except ImportError:
        pytest.skip("Cython module not available")


def test_cython_compilation():
    try:
        from pysrf._bsum import update_w

        n, r = 10, 3
        m = np.random.rand(n, n)
        m = (m + m.T) / 2
        x0 = np.random.rand(n, r)

        result = update_w(m, x0, max_iter=10, tol=1e-6)

        assert result.shape == (n, r)
        assert np.all(result >= 0)
        assert not np.any(np.isnan(result))

    except ImportError:
        pytest.skip("Cython module not available")


def test_python_fallback():
    from pysrf.model import update_w_python

    n, r = 10, 3
    m = np.random.rand(n, n)
    m = (m + m.T) / 2
    x0 = np.random.rand(n, r)

    result = update_w_python(m, x0, max_iter=10, tol=1e-6)

    assert result.shape == (n, r)
    assert np.all(result >= 0)
    assert not np.any(np.isnan(result))


def test_cython_vs_python():
    try:
        from pysrf._bsum import update_w as update_w_cython
        from pysrf.model import update_w_python

        np.random.seed(42)
        n, r = 10, 3
        m = np.random.rand(n, n)
        m = (m + m.T) / 2
        x0 = np.random.rand(n, r)

        result_cython = update_w_cython(m, x0.copy(), max_iter=50, tol=1e-6)
        result_python = update_w_python(m, x0.copy(), max_iter=50, tol=1e-6)

        np.testing.assert_allclose(result_cython, result_python, rtol=1e-5, atol=1e-5)

    except ImportError:
        pytest.skip("Cython module not available")
