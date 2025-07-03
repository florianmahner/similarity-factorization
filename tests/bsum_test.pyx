# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow, fabs, cbrt

ctypedef np.float64_t DTYPE_t


cpdef double _quartic_root(double a, double b, double c, double d) nogil:
    cdef double bb  = b * b
    cdef double a2  = a * a
    cdef double p   = (3.0 * a * c - bb) / (3.0 * a2)
    cdef double q   = (9.0 * a * b * c - 27.0 * a2 * d - 2.0 * b * bb) / (27.0 * a2 * a)
    cdef double root, delta, tmp

    if c > bb / (3.0 * a):
        delta = sqrt(q * q * 0.25 + p * p * p / 27.0)
        root  = cbrt(q * 0.5 - delta) + cbrt(q * 0.5 + delta)
    else:
        tmp   = b * bb / (27.0 * a2 * a) - d / a
        root  = cbrt(tmp)

    return 0.0 if root < 0.0 else root


cdef inline double _dot_product(double[:] a, double[:] b) nogil:
    cdef int n = a.shape[0]
    cdef int i
    cdef double result = 0.0
    
    for i in range(n):
        result += a[i] * b[i]
    return result


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
    cdef int it, i, j, k
    cdef double max_delta, old, b, c, d, new, delta, update_val
    
    for it in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            for j in range(r):
                old = x_view[i, j]
                b = 12.0 * old
                c = 4.0 * ((diag_view[i] - m_view[i, i]) + xtx_view[j, j] + old * old)
                
                d = 4.0 * _dot_product(x_view[i, :], xtx_view[:, j]) - 4.0 * _dot_product(m_view[i, :], x_view[:, j])
                new = _quartic_root(a, b, c, d)
                delta = new - old
                
                if abs(delta) > max_delta:
                    max_delta = abs(delta)
                
                diag_view[i] += new * new - old * old

                for k in range(r):
                    update_val = delta * x_view[i, k]
                    xtx_view[j, k] += update_val
                    xtx_view[k, j] += update_val
                
                xtx_view[j, j] += delta * delta
                
                x_view[i, j] = new
        
        if max_delta < tol and tol > 0.0:
            break
    
    return x
