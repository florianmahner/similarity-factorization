import math
import numpy as np


def _real_cubic_root(s_qi: float, norm_b: float) -> float:
    """
    Solve  t³ + s_qi·t - norm_b = 0   for t ≥ 0   (unique real root because s_qi ≥ 0).
    """
    if norm_b == 0.0:
        return 0.0
    delta = math.sqrt(norm_b**2 / 4.0 + (s_qi**3) / 27.0)
    c1 = math.copysign(abs(norm_b / 2.0 + delta) ** (1.0 / 3.0), norm_b / 2.0 + delta)
    c2 = math.copysign(abs(norm_b / 2.0 - delta) ** (1.0 / 3.0), norm_b / 2.0 - delta)
    return c1 + c2


def update_vsum(
    z: np.ndarray,
    w0: np.ndarray,
    max_outer_iter: int = 50,
    inner_iter: int = 1,
) -> np.ndarray:
    n, r = w0.shape
    w = w0.copy()
    xtx = w.T @ w  # (r, r)

    eye_r = np.eye(r)

    for iter in range(max_outer_iter):
        for i in range(n):
            x_i = w[i, :].copy()  # current row (view gets overwritten)
            pi = xtx - np.outer(x_i, x_i)  # P_i  (Step 3)
            qi = w.T @ z[:, i] - z[i, i] * x_i  # q_i  (Step 4)
            qi_mat = pi - z[i, i] * eye_r  # Q_i  (needed for Lipschitz)

            # Lipschitz constant  s_Qi  — use spectral radius (r is small, exact eig is cheap)
            s_qi = max(0.0, np.linalg.eigvalsh(qi_mat).max())

            for _ in range(inner_iter):  # Steps 5–8
                b_i = qi + (s_qi + z[i, i]) * x_i - pi @ x_i
                b_pos = np.maximum(b_i, 0.0)
                if not b_pos.any():
                    x_new = np.zeros_like(x_i)
                else:
                    t = _real_cubic_root(s_qi, np.linalg.norm(b_pos))
                    x_new = (t / np.linalg.norm(b_pos)) * b_pos

                # rank-1 update of  XᵀX  (Step 9)
                xtx += np.outer(x_new, x_new) - np.outer(x_i, x_i)
                x_i[:] = x_new  # write row back

                # refresh fast variables for the next inner loop
                pi = xtx - np.outer(x_i, x_i)
                qi = w.T @ z[:, i] - z[i, i] * x_i

        print("iter", iter, "obj", np.linalg.norm(w @ w.T - z), end="\r")

    return w
