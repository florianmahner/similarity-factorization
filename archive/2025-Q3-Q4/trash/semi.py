import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Lasso


import numpy as np


def separate_positive(m):
    return (np.abs(m) + m) / 2.0


def separate_negative(m):
    return (np.abs(m) - m) / 2.0


def update_w(data, H):
    W1 = np.dot(data, H.T)
    W2 = np.dot(H, H.T)
    W = np.dot(W1, np.linalg.inv(W2))
    return W


def update_h(data, W, H):
    XW = np.dot(data.T, W)
    WW = np.dot(W.T, W)
    WW_pos = separate_positive(WW)
    WW_neg = separate_negative(WW)

    XW_pos = separate_positive(XW)
    H1 = (XW_pos + np.dot(H.T, WW_neg)).T

    XW_neg = separate_negative(XW)
    H2 = (XW_neg + np.dot(H.T, WW_pos)).T + 10**-9

    H = H * np.sqrt(H1 / H2)
    return H


def semi_nmf(data, num_bases=4, niter=10, verbose=False):
    # Initialize W and H
    data_dimension, num_samples = data.shape
    W = np.random.rand(data_dimension, num_bases)
    H = np.random.rand(num_bases, num_samples)

    # Factorization loop
    for i in range(niter):
        W = update_w(data, H)
        H = update_h(data, W, H)

        frob = np.linalg.norm(data - W @ H, "fro")
        if verbose:
            print(f"Iteration {i+1} of {niter}, Frobenius norm: {frob:.4f}")

    return W, H


# ----------------------------------------
# simulate symmetric positive-definite matrix
np.random.seed(0)
n, k = 50, 5
x_true = np.random.randn(n, k)
s = x_true @ x_true.T

# SVD of s
u, sigma, _ = np.linalg.svd(s)
u_k = u[:, :k]
sigma_k = sigma[:k]

# apply semi-NMF with sparsity on Q
niter = 10
W, H = semi_nmf(u_k, num_bases=k, niter=niter, verbose=True)

# reconstruct A and S
Q = H.T
A = Q @ np.diag(sigma_k) @ Q.T
S_recon = W @ A @ W.T
err_s = np.linalg.norm(s - S_recon, "fro") / np.linalg.norm(s, "fro")


# print(f"semi-NMF converged in {iters} iterations, U-approx rel error: {err_u:.4f}")
print(f"Final reconstruction error on S: {err_s:.4f}")
