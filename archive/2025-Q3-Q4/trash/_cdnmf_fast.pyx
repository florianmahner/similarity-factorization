"""
This code is developed based on the following:
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_cdnmf_fast.pyx
by authors: Mathieu Blondel, Tom Dupre la Tour
"""

import cython
from cython cimport floating
from libc.math cimport fabs


def _update_cdnmf_fast(floating[:, ::1] W, floating[:, :] HHt,
                       floating[:, :] XHt, float upperbd, Py_ssize_t[::1] permutation):
    cdef:
        floating violation = 0
        Py_ssize_t n_components = W.shape[1]
        Py_ssize_t n_samples = W.shape[0]  # n_features for H update
        floating grad, pg, hess
        Py_ssize_t i, r, s, t

    with nogil:
        with cython.boundscheck(False):
            for s in range(n_components):
                t = permutation[s]

                for i in range(n_samples):
                    # gradient = GW[t, i] where GW = np.dot(W, HHt) - XHt
                    grad = -XHt[i, t]

                    for r in range(n_components):
                        grad += HHt[t, r] * W[i, r]

                    # projected gradient
                    pg = min(0., grad) if W[i, t] == 0 else grad
                    violation += fabs(pg)

                    # Hessian
                    hess = HHt[t, t]

                    if hess != 0:
                        if upperbd > 0:
                            W[i, t] = min(max(W[i, t] - grad / hess, 0.), upperbd)
                        else:
                            W[i, t] = max(W[i, t] - grad / hess, 0.)

    return violation



def _update_cdnmf_fast_per_entry_bounds(
        floating[:, ::1] W,            
        floating[:, :]   HHt,          
        floating[:, :]   XHt,          
        floating[:, ::1] upperbd,      
        Py_ssize_t[::1]  permutation   
    ) -> floating:
    cdef Py_ssize_t n_samples    = W.shape[0]
    cdef Py_ssize_t n_components = W.shape[1]

    cdef floating violation = 0.0
    cdef floating grad, pg, hess, ub, w_new
    cdef Py_ssize_t i, r, s, t

    for s in range(n_components):
        t = permutation[s]

        for i in range(n_samples):
            # -------- gradient --------
            grad = -XHt[i, t]
            for r in range(n_components):
                grad += HHt[t, r] * W[i, r]

            # -------- projected-gradient (monitoring only) --------
            if W[i, t] == 0.0:
                pg = min(0.0, grad)
            else:
                pg = grad
            violation += fabs(pg)

            # -------- Newton update with clamp --------
            hess = HHt[t, t]
            if hess != 0.0:
                ub = upperbd[i, t]
                w_new = W[i, t] - grad / hess
                if   w_new < 0.0: W[i, t] = 0.0
                elif w_new > ub:  W[i, t] = ub
                else:             W[i, t] = w_new

    return violation