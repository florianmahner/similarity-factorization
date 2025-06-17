# distutils: language = c++, define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False

import numpy as np
import cython
cimport numpy as np
from libc.math cimport sqrt, cbrt
from cython cimport floating

ctypedef np.float64_t DTYPE_t

cdef double _cubic_root(double a, double b, double c, double d):
    cdef double p, q, delta, x_new, stmp
    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (9.0 * a * b * c - 27.0 * a * a * d - 2.0 * b * b * b) / (27.0 * a * a * a)

    if c > b * b / (3.0 * a):  # three real roots
        delta = sqrt(q * q / 4.0 + p * p * p / 27.0)
        x_new = cbrt(q / 2.0 - delta) + cbrt(q / 2.0 + delta)
    else:  # one real root
        stmp = b * b * b / (27.0 * a * a * a) - d / a
        x_new = cbrt(stmp)

    if x_new < 0.0:
        x_new = 0.0

    return x_new


def update_w(np.ndarray[DTYPE_t, ndim=2] m,
                np.ndarray[DTYPE_t, ndim=2] x0,
                int max_iter=100,
                double tol=1e-6,
                bint verbose=False):
    cdef int n = x0.shape[0]
    cdef int r = x0.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] x = x0.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] xtx = np.dot(x.T, x)
    cdef np.ndarray[DTYPE_t, ndim=1] diag = np.einsum("ij,ij->i", x, x)

    cdef double a = 4.0
    cdef int it, i, j
    cdef double max_delta, old, b, c, d, new, delta
    cdef np.ndarray[DTYPE_t, ndim=1] update_row

    for it in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            for j in range(r):
                old = x[i, j]
                b = 12.0 * old
                c = 4.0 * ((diag[i] - m[i, i]) + xtx[j, j] + old * old)
                d = 4.0 * np.dot(x[i], xtx[:, j]) - 4.0 * np.dot(m[i], x[:, j])

                new = _cubic_root(a, b, c, d)
                
                delta = new - old
                if abs(delta) > max_delta:
                    max_delta = abs(delta)

                diag[i] += new * new - old * old
                update_row = delta * x[i]
                xtx[j, :] += update_row
                xtx[:, j] += update_row
                xtx[j, j] += delta * delta
                x[i, j] = new

        if verbose:
            evar = 1 - (np.linalg.norm(m - np.dot(x, x.T), "fro") / np.linalg.norm(m, "fro"))
            print(f"it {it:3d}  evar {evar:.6f}")

        if max_delta < tol and tol > 0.0:
            break

    return x