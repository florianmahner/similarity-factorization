# distutils: language = c++, define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow, fabs

ctypedef np.float64_t DTYPE_t

cdef inline double _cbrt(double x) nogil:
    """Cube root that handles negative numbers correctly like numpy.cbrt"""
    if x >= 0.0:
        return pow(x, 1.0/3.0)
    else:
        return -pow(-x, 1.0/3.0)

cdef inline double _cubic_root(double a, double b, double c, double d) nogil:
    cdef double p = (3.0 * a * c - b * b) / (3.0 * a * a)
    cdef double q = (9.0 * a * b * c - 27.0 * a * a * d - 2.0 * b * b * b) / (27.0 * a * a * a)
    cdef double x_new
    cdef double delta
    cdef double stmp
    if c > b * b / (3.0 * a):
        delta = sqrt(q * q / 4.0 + p * p * p / 27.0)
        x_new = _cbrt(q / 2.0 - delta) + _cbrt(q / 2.0 + delta)
    else:
        stmp = b * b * b / (27.0 * a * a * a) - d / a
        x_new = _cbrt(stmp)
    if x_new < 0.0:
        x_new = 0.0
    return x_new

cdef inline double _dot_product(double[:] a, double[:] b) noexcept nogil:
    cdef int n = a.shape[0]
    cdef int i
    cdef double result = 0.0
    
    for i in range(n):
        result += a[i] * b[i]
    return result

cdef inline void _update_xtx_row_col_correct(double[:, :] xtx, double[:] x_row, double delta, int j) noexcept nogil:
    cdef int r = xtx.shape[0]
    cdef int k
    cdef double update_val
    
    # Update row and column, but handle diagonal separately to avoid double counting
    for k in range(r):
        update_val = delta * x_row[k]
        xtx[j, k] += update_val  # Update row j
        if k != j:  # Skip diagonal here
            xtx[k, j] += update_val  # Update column j
    
    # Handle diagonal separately - it should get updated exactly twice
    xtx[j, j] += delta * x_row[j]  # Second update for diagonal

cpdef np.ndarray[DTYPE_t, ndim=2] update_w_fast(np.ndarray[DTYPE_t, ndim=2] m,
                                                np.ndarray[DTYPE_t, ndim=2] x0,
                                                int max_iter=100,
                                                double tol=1e-6,
                                                bint verbose=False):
    cdef int n = x0.shape[0]
    cdef int r = x0.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] x = x0.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] xtx = np.dot(x.T, x)
    cdef np.ndarray[DTYPE_t, ndim=1] diag = np.einsum("ij,ij->i", x, x)
    
    cdef double[:, :] x_view = x
    cdef double[:, :] m_view = m
    cdef double[:, :] xtx_view = xtx
    cdef double[:] diag_view = diag
    
    cdef double a = 4.0
    cdef int it, i, j
    cdef double max_delta, old, b, c, d, new, delta
    
    for it in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            for j in range(r):
                old = x_view[i, j]
                b = 12.0 * old
                c = 4.0 * ((diag_view[i] - m_view[i, i]) + xtx_view[j, j] + old * old)
                
                d = 4.0 * _dot_product(x_view[i, :], xtx_view[:, j]) - 4.0 * _dot_product(m_view[i, :], x_view[:, j])
                
                new = _cubic_root(a, b, c, d)
                delta = new - old
                
                if abs(delta) > max_delta:
                    max_delta = abs(delta)
                
                diag_view[i] += new * new - old * old
                
                _update_xtx_row_col_correct(xtx_view, x_view[i, :], delta, j)
                xtx_view[j, j] += delta * delta
                
                x_view[i, j] = new
        
        if max_delta < tol and tol > 0.0:
            break
    
    return x

cpdef np.ndarray[DTYPE_t, ndim=2] update_w_orig(np.ndarray[DTYPE_t, ndim=2] m,
                                                np.ndarray[DTYPE_t, ndim=2] x0,
                                                int max_iter=100,
                                                double tol=1e-6,
                                                bint verbose=False):
    cdef int n = x0.shape[0]
    cdef int r = x0.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] x = x0.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] xtx = np.dot(x.T, x)
    cdef np.ndarray[DTYPE_t, ndim=1] diag = np.einsum("ij,ij->i", x, x)
    
    cdef double[:, :] x_view = x
    cdef double[:, :] m_view = m
    cdef double[:, :] xtx_view = xtx
    cdef double[:] diag_view = diag
    
    cdef double a = 4.0
    cdef int it, i, j
    cdef double max_delta, old, b, c, d, new, delta
    cdef np.ndarray[DTYPE_t, ndim=1] update_row
    
    for it in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            for j in range(r):
                old = x_view[i, j]
                b = 12.0 * old
                c = 4.0 * ((diag_view[i] - m_view[i, i]) + xtx_view[j, j] + old * old)
                d = 4.0 * np.dot(x[i], xtx[:, j]) - 4.0 * np.dot(m[i], x[:, j])
                new = _cubic_root(a, b, c, d)
                delta = new - old
                if abs(delta) > max_delta:
                    max_delta = abs(delta)
                diag_view[i] += new * new - old * old
                update_row = delta * x[i]
                xtx[j, :] += update_row
                xtx[:, j] += update_row
                xtx[j, j] += delta * delta
                x_view[i, j] = new
        if max_delta < tol and tol > 0.0:
            break
    return x
