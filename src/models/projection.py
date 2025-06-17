"""Projection NMF models - simple and direct."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .utils import init_factor, frobenius_norm

class ProjectionNMF(BaseEstimator, TransformerMixin):
    """Projection NMF for data X, finding W such that X ≈ W @ W.T @ X."""
    
    def __init__(self, rank=10, max_iter=1000, tol=1e-5, random_state=None,
                 init="random", verbose=False, eps=np.finfo(float).eps):
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.verbose = verbose
        self.eps = eps
    
    def fit(self, x, y=None):
        """Exact same algorithm from ProjectionNMF class."""
        x = self._validate_data(x, ensure_2d=True)
        
        w = init_factor(x, self.rank, self.init, self.random_state, self.eps)
        error_init = frobenius_norm(x, w @ w.T @ x)
        previous_error = error_init
        xx = x @ x.T

        for iteration in range(1, self.max_iter + 1):
            # Exact same multiplicative update
            a = 2 * xx @ w  # numerator
            b = w @ (w.T @ a) + xx @ (w @ (w.T @ w))  # denominator
            np.maximum(b, self.eps, out=b)
            w *= a / b  # multiplicative update
            
            # Exact same normalization
            wxxw = w.T @ xx @ w
            norm_factor = np.sqrt(np.trace(wxxw) / np.trace(wxxw @ (w.T @ w)))
            w *= norm_factor

            error = frobenius_norm(x, w @ w.T @ x)
            diff = (previous_error - error) / error_init

            if diff < self.tol:
                break
            previous_error = error

        # Sort by sum (exact same logic)
        order = np.argsort(-np.sum(w, axis=0))
        w = w[:, order]
        
        # Store results
        self.w_ = w
        self.components_ = w
        self.x_hat_ = w @ w.T @ x
        self.n_iter_ = iteration
        self.reconstruction_err_ = error
        
        return self
    
    def transform(self, x):
        check_is_fitted(self)
        x = self._validate_data(x, reset=False)
        return self.w_ @ self.w_.T @ x
    
    def fit_transform(self, x, y=None):
        """Returns w."""
        self.fit(x, y)
        return self.w_


class ProjectionKernelNMF(BaseEstimator, TransformerMixin):
    """Projection NMF for kernel matrix K, finding W such that K ≈ W @ W.T @ K."""
    
    def __init__(self, rank=10, max_iter=1000, tol=1e-5, random_state=None,
                 init="random", verbose=False, eps=np.finfo(float).eps):
        self.rank = rank
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.verbose = verbose
        self.eps = eps
    
    def _separate_pos_neg(self, x):
        """Exact same from ProjectionKernelNMF."""
        pos = (abs(x) + x) / 2
        neg = (abs(x) - x) / 2
        return pos, neg
    
    def fit(self, k, y=None):
        """Exact same algorithm from ProjectionKernelNMF class."""
        k = self._validate_data(k, ensure_2d=True)
        
        # Initialize w 
        rng = np.random.RandomState(self.random_state)
        w = rng.rand(k.shape[0], self.rank)
        
        error_init = frobenius_norm(k, w @ w.T @ k)
        previous_error = error_init

        for iteration in range(1, self.max_iter + 1):
            w_prev = w.copy()

            # Handle negative values in kernel matrix
            if k.min() < 0:
                pos, neg = self._separate_pos_neg(k)
                ww = w @ w.T
                a = pos @ w + ww @ neg @ w
                b = neg @ w + ww @ pos @ w
            else:
                a = k @ w
                b = w @ (w.T @ a)
            
            np.maximum(b, 1e-12, out=b)
            
            # Multiplicative update
            w *= a
            w /= b

            # Exact same normalization (equation 54 from paper)
            wxw = w.T @ k @ w
            norm_factor = np.sqrt(np.trace(wxw) / np.trace(wxw @ (w.T @ w)))
            w *= norm_factor

            error = frobenius_norm(k, w @ w.T @ k)
            diff = frobenius_norm(w_prev, w) / np.linalg.norm(w_prev, ord="fro")

            if diff < self.tol and self.tol > 0:
                break

        # Sort by sum (exact same logic)
        order = np.argsort(-np.sum(w, axis=0))
        w = w[:, order]
        
        # Store results
        self.w_ = w
        self.components_ = w
        self.k_hat_ = w @ w.T @ k
        self.n_iter_ = iteration
        self.reconstruction_err_ = error
        
        return self
    
    def transform(self, k):
        check_is_fitted(self)
        k = self._validate_data(k, reset=False)
        return self.w_ @ self.w_.T @ k
    
    def fit_transform(self, k, y=None):
        """Returns w."""
        self.fit(k, y)
        return self.w_
