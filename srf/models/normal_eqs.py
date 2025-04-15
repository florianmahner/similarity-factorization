import numpy as np
from .nnls_block import nnlsm_blockpivot

Array = np.ndarray


def regularized_normal_eqs(
    s: Array, fixed: Array, alpha: float, update: Array
) -> Array:
    """
    We stack the matrices and rearrange the normal equations.
    see eq. 16 and 17 in Kuang et al. (2014)
    """
    sqrt_alpha = np.sqrt(alpha)
    rank = fixed.shape[1]
    # Build the stacked matrices:
    # a_stack has A^T on top and sqrt(alpha)*fixed.T on the bottom.
    a_stack = np.vstack([s.T, sqrt_alpha * fixed.T])
    # c_stack has fixed on top and sqrt(alpha)*I on the bottom.
    c_stack = np.vstack([fixed, sqrt_alpha * np.eye(rank)])

    # solve for x (which approximates the updated factor's transpose) via NNLS.
    # this basically solves the normal equations for the stacked matrices
    left = c_stack.T @ c_stack
    right = c_stack.T @ a_stack
    x_t = nnlsm_blockpivot(left, right, True, update.T)[0]
    # transpose back to get the updated factor.
    return x_t.T
