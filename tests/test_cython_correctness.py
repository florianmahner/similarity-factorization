import numpy as np
import pytest
import os

# Set up pyximport for Cython compilation
os.environ["CPPFLAGS"] = (
    f"-I{np.get_include()} -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"
)
import pyximport
pyximport.install(
    setup_args={
        "include_dirs": [np.get_include()],
        "script_args": ["--quiet"],
    },
    language_level=3,
)

from pysrf.model import update_w_python
from pysrf._bsum import update_w as update_w_cython


def make_data(n: int = 100, r: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    m = rng.random((n, n))
    m = (m + m.T) * 0.5
    x0 = rng.random((n, r))
    return m, x0


def test_cython_python_equivalence():
    """Test that Cython and Python implementations give identical results."""
    m, x0 = make_data(seed=1234)
    
    cython_result = update_w_cython(m, x0, tol=0.0)
    python_result = update_w_python(m, x0, tol=0.0)
    
    error = np.max(np.abs(cython_result - python_result))
    assert error < 1e-10, f"Large discrepancy between implementations: {error:.4e}"


def test_cython_python_equivalence_multiple_seeds():
    """Test equivalence across multiple random seeds."""
    for seed in [0, 42, 123, 456, 789]:
        m, x0 = make_data(seed=seed)
        
        cython_result = update_w_cython(m, x0, tol=0.0)
        python_result = update_w_python(m, x0, tol=0.0)
        
        error = np.max(np.abs(cython_result - python_result))
        assert error < 1e-10, f"Large discrepancy for seed {seed}: {error:.4e}"


def test_cython_python_equivalence_different_sizes():
    """Test equivalence for different matrix sizes."""
    sizes = [(10, 3), (50, 5), (100, 10), (200, 15)]
    
    for n, r in sizes:
        m, x0 = make_data(n=n, r=r, seed=42)
        
        cython_result = update_w_cython(m, x0, tol=0.0)
        python_result = update_w_python(m, x0, tol=0.0)
        
        error = np.max(np.abs(cython_result - python_result))
        assert error < 1e-10, f"Large discrepancy for size ({n}, {r}): {error:.4e}"
