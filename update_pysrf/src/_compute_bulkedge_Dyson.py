from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.linalg as la


def _symmetrize_with_nan(S):
    """
    Symmetrize a matrix while preserving missing-value semantics.

    For each pair (i, j):
    1. If both S_ij and S_ji are finite, average them.
    2. If exactly one side is finite, copy the finite value to both sides.
    3. If both are NaN, keep NaN.

    Diagonal NaNs are replaced by 0.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square 2D array.")

    St = S.T
    fa = np.isfinite(S)
    fb = np.isfinite(St)

    out = np.full_like(S, np.nan, dtype=float)

    both = fa & fb
    out[both] = 0.5 * (S[both] + St[both])

    only_a = fa & (~fb)
    out[only_a] = S[only_a]

    only_b = (~fa) & fb
    out[only_b] = St[only_b]

    d = np.diag(out).copy()
    d[~np.isfinite(d)] = 0.0
    np.fill_diagonal(out, d)
    return out


def _spectral_norm_power(A, n_iter=40, tol=1e-6, random_state=0):
    """
    Estimate ||A||_2 by power iteration for symmetric real A.

    The update is x <- A x / ||A x|| and the estimate is the Rayleigh quotient
    x^T A x. Complexity is O(n^2 * n_iter) without eigendecomposition.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    rng = np.random.default_rng(random_state)
    x = rng.standard_normal(n)
    x /= (la.norm(x) + 1e-16)

    last = 0.0
    for _ in range(int(n_iter)):
        y = A @ x
        ny = la.norm(y)
        if ny < 1e-16:
            return 0.0
        x = y / ny
        val = float(np.dot(x, A @ x))
        if abs(val - last) <= tol * max(1.0, abs(val)):
            return abs(val)
        last = val
    return abs(last)


def _build_s2_missing_aware(S_sym, missing_policy="zero", eps=1e-12):
    """
    Build S2 used by the Dyson operator V = alpha * S2.

    Let W_ij = 1 if S_ij is observed, 0 otherwise, after symmetric harmonization.
    Let S0 be S with NaNs replaced by 0.

    Policies:
    - 'zero':      S2_ij = W_ij * S0_ij^2
    - 'mcar_unbiased':
                   off-diagonal S2_ij = W_ij * S0_ij^2 / q
                   where q is the observed off-diagonal rate
                   q = mean_{i<j} W_ij.
                   Diagonal is not rescaled.
    """
    if missing_policy not in ("zero", "mcar_unbiased"):
        raise ValueError("missing_policy must be one of {'zero', 'mcar_unbiased'}.")

    S_sym = np.asarray(S_sym, dtype=float)
    n = S_sym.shape[0]
    W = np.isfinite(S_sym).astype(float)
    W = ((W + W.T) > 0.0).astype(float)
    np.fill_diagonal(W, 1.0)

    S0 = np.nan_to_num(S_sym, nan=0.0)
    S2 = W * (S0 * S0)

    if n <= 1:
        q = 1.0
    else:
        iu = np.triu_indices(n, k=1)
        off = W[iu]
        q = float(off.mean()) if off.size else 1.0
    q = float(np.clip(q, eps, 1.0))

    if missing_policy == "mcar_unbiased" and n > 1:
        iu = np.triu_indices(n, k=1)
        S2[iu] /= q
        S2[(iu[1], iu[0])] /= q

    S2 = 0.5 * (S2 + S2.T)
    return S0, S2, W, q


class VDEPrecomp:
    """
    Precompute reusable objects for fast vector Dyson solves.

    Model:
    1. Let alpha = p(1-p), w = z + i*eta, z > 0, eta > 0.
    2. The self-consistent equation for m(w) in C^n is
       m_i(w) = -1 / (w + alpha * sum_j S2_ij m_j(w)).
    3. In vector form:
       m(w) = - (w * 1 + alpha * S2 m(w))^{-1}
       where inverse is element-wise.

    S2 construction:
    - If S has no NaNs, S2 = S^2 element-wise.
    - If S has NaN off-diagonal entries, those entries are treated as missing.
      missing_policy='zero': missing entries contribute 0 to S2.
      missing_policy='mcar_unbiased': off-diagonal S2 is divided by observation
      rate q so E[S2_est] matches the fully observed quantity under MCAR.
    """

    __slots__ = (
        "S",
        "S0",
        "S2",
        "W",
        "n",
        "q_offdiag_obs",
        "has_missing_offdiag",
        "missing_policy",
        "normS_est",
        "normS2_est",
    )

    def __init__(
        self,
        S,
        norm_iters=40,
        random_state=0,
        missing_policy="zero",
    ):
        S_sym = _symmetrize_with_nan(S)
        S0, S2, W, q = _build_s2_missing_aware(S_sym, missing_policy=missing_policy)

        self.S = S_sym
        self.S0 = S0
        self.S2 = S2
        self.W = W
        self.n = S_sym.shape[0]
        self.q_offdiag_obs = float(q)
        self.has_missing_offdiag = bool(q < (1.0 - 1e-15))
        self.missing_policy = str(missing_policy)
        self.normS_est = _spectral_norm_power(S0, n_iter=norm_iters, random_state=random_state)
        self.normS2_est = _spectral_norm_power(S2, n_iter=norm_iters, random_state=random_state)


def _as_precomp(precomp_or_S, missing_policy="zero", norm_iters=40, random_state=0):
    """
    Accept either a VDEPrecomp or a raw matrix S and return VDEPrecomp.

    This keeps public APIs backward-compatible for callers that pass S directly.
    """
    if isinstance(precomp_or_S, VDEPrecomp):
        return precomp_or_S
    return VDEPrecomp(
        precomp_or_S,
        norm_iters=norm_iters,
        random_state=random_state,
        missing_policy=missing_policy,
    )


def _solve_vde_bulk_fast(
    precomp: VDEPrecomp,
    p: float,
    z: float,
    eta=1e-3,
    max_iter=2000,
    tol=1e-7,
    omega=0.8,
    warm=None,
):
    """
    Solve the vector Dyson fixed-point equation at one real location z.

    Equation:
      m = F(m), F_i(m) = -1 / (w + alpha * (S2 m)_i)
    where alpha = p(1-p), w = z + i*eta.

    Iteration:
      m^{t+1} = omega * F(m^t) + (1-omega) * m^t.
    """
    p = float(p)
    alpha = p * (1.0 - p)
    w = complex(float(z), float(eta))

    if warm is None:
        m = -np.ones(precomp.n, dtype=complex) / w
    else:
        m = np.asarray(warm, dtype=complex).copy()

    for _ in range(int(max_iter)):
        denom = w + alpha * (precomp.S2 @ m)
        tiny = np.abs(denom) < 1e-16
        denom[tiny] = 1e-16 + 0.0j
        m_new = -1.0 / denom

        diff = m_new - m
        m = omega * m_new + (1.0 - omega) * m
        if np.max(np.abs(diff)) < tol:
            break
    return m


def _solve_vde_bulk_batched_fast(
    precomp: VDEPrecomp,
    p: float,
    zs,
    eta=1e-3,
    max_iter=2000,
    tol=1e-7,
    omega=0.8,
):
    """
    Solve the Dyson fixed point for all z in one batched iteration.

    This exposes parallelism across the z-grid because all columns of m are
    updated together:
      M = -1 / (W + alpha * S2 M)
    where M in C^{n x G}, W_{ig} = z_g + i*eta.
    """
    p = float(p)
    alpha = p * (1.0 - p)
    zs = np.asarray(zs, dtype=float)
    G = zs.size

    w_row = zs.reshape(1, G) + 1j * float(eta)
    m = -np.ones((precomp.n, G), dtype=complex) / w_row

    for _ in range(int(max_iter)):
        denom = w_row + alpha * (precomp.S2 @ m)
        tiny = np.abs(denom) < 1e-16
        denom[tiny] = 1e-16 + 0.0j
        m_new = -1.0 / denom

        diff = m_new - m
        m = omega * m_new + (1.0 - omega) * m
        if np.max(np.abs(diff)) < tol:
            break
    return m


def _solve_vde_mprime_fast(
    precomp: VDEPrecomp,
    p: float,
    m: np.ndarray,
    max_iter=5000,
    tol=1e-10,
    omega=0.8,
):
    """
    Solve for m' = dm/dw at fixed w using a fixed-point method.

    Differentiating m = -1/(w + alpha S2 m) gives:
      (I - diag(m^2) alpha S2) m' = m^2.

    Equivalent fixed-point form:
      m' = m^2 + m^2 * (alpha * S2 m'),
    where products with m^2 are element-wise.
    """
    p = float(p)
    alpha = p * (1.0 - p)
    m2 = m * m
    mp = m2.copy()

    for _ in range(int(max_iter)):
        mp_new = m2 + m2 * (alpha * (precomp.S2 @ mp))
        diff = mp_new - mp
        mp = omega * mp_new + (1.0 - omega) * mp
        if np.max(np.abs(diff)) < tol:
            break
    return mp


def compute_im_mavg_grid_fast(
    precomp: VDEPrecomp,
    p: float,
    eta=1e-3,
    ngrid=120,
    omega=0.8,
    z_min=1e-8,
    solver="auto",
    batched_max_entries=2_000_000,
    missing_policy="zero",
    norm_iters=40,
    random_state=0,
):
    """
    Compute Im(mean_i m_i(z + i*eta)) on a descending z-grid.

    Edge heuristic:
    - Build z_g from z_max down to z_min where
      z_max = ||S||_2 + 8 * sqrt(p(1-p) * ||S2||_2).
    - Evaluate m(w_g) at each grid point.
    - Return (z_g, Im average m).

    Solver modes:
    - solver='warm': sequential z-loop with warm starts.
    - solver='batched': solve all z points simultaneously (matrix-batched).
    - solver='auto': choose batched when n*ngrid is moderate; else warm.
    """
    precomp = _as_precomp(
        precomp,
        missing_policy=missing_policy,
        norm_iters=norm_iters,
        random_state=random_state,
    )

    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return np.array([0.0]), np.array([0.0])

    s2_max_est = max(precomp.normS2_est, 1e-16)
    z_max = precomp.normS_est + 8.0 * np.sqrt(p * (1.0 - p) * s2_max_est)
    zs = np.linspace(z_max, float(z_min), int(ngrid))

    if solver == "auto":
        use_batched = (precomp.n * int(ngrid)) <= int(batched_max_entries) and int(ngrid) >= 64
        solver_eff = "batched" if use_batched else "warm"
    else:
        solver_eff = str(solver)

    if solver_eff == "batched":
        M = _solve_vde_bulk_batched_fast(
            precomp=precomp,
            p=p,
            zs=zs,
            eta=eta,
            omega=omega,
        )
        im_mavg = np.imag(np.mean(M, axis=0)).astype(float, copy=False)
        return zs, im_mavg

    if solver_eff != "warm":
        raise ValueError("solver must be one of {'auto', 'warm', 'batched'}.")

    warm = None
    im_mavg = np.empty_like(zs, dtype=float)
    for i, z in enumerate(zs):
        m = _solve_vde_bulk_fast(precomp, p, float(z), eta=eta, warm=warm, omega=omega)
        warm = m
        im_mavg[i] = float(np.imag(np.mean(m)))
    return zs, im_mavg


def detect_bulk_edge_from_im(zs, im_mavg, jump_frac=0.1):
    """
    Detect right bulk edge from the Stieltjes-imaginary profile.

    Given y_g = Im(mean m(z_g + i*eta)), detect first g such that y_g exceeds a
    relative threshold:
      y_g > jump_frac * max_h |y_{h+1} - y_h|.
    """
    im_mavg = np.asarray(im_mavg, dtype=float)
    zs = np.asarray(zs, dtype=float)

    if len(zs) < 2:
        return float(zs[-1]), len(zs) - 1

    jumps = np.diff(im_mavg)
    max_jump = np.max(np.abs(jumps))
    if max_jump < 1e-12:
        return float(zs[-1]), len(zs) - 1

    threshold = float(jump_frac) * max_jump
    mask = im_mavg > threshold
    idx = int(np.argmax(mask)) if np.any(mask) else len(zs) - 1
    return float(zs[idx]), idx


def _propose_eta_fast(precomp: VDEPrecomp, p: float):
    """Heuristic eta scale from p(1-p)||S2||_2."""
    s2_max_est = max(precomp.normS2_est, 1e-16)
    scale = np.sqrt(float(p) * (1.0 - float(p)) * s2_max_est)
    eta0 = 1e-3 * max(1.0, float(scale))
    return float(np.clip(eta0, 1e-4, 1e-2))


def _propose_ngrid(n):
    """Heuristic grid size proportional to matrix dimension."""
    return int(np.clip(2 * int(n), 80, 200))


def _evaluate_dyson_cfg(
    precomp,
    p,
    eta,
    ngrid,
    omega,
    jump_frac_candidates,
    solver,
):
    zs, im_mavg = compute_im_mavg_grid_fast(
        precomp,
        p,
        eta=eta,
        ngrid=ngrid,
        omega=omega,
        solver=solver,
    )

    edges = []
    idxs = []
    for jf in jump_frac_candidates:
        edge, idx = detect_bulk_edge_from_im(zs, im_mavg, jump_frac=jf)
        edges.append(edge)
        idxs.append(idx)

    edges = np.asarray(edges, dtype=float)
    idxs = np.asarray(idxs, dtype=int)
    spread = float(np.max(edges) - np.min(edges))
    edge_at_extreme = np.any((idxs == 0) | (idxs == len(zs) - 1))
    extreme_penalty = 1e3 if edge_at_extreme else 0.0
    score = spread + extreme_penalty

    median_edge = float(np.median(edges))
    best_j_idx = int(np.argmin(np.abs(edges - median_edge)))
    best_jump = float(jump_frac_candidates[best_j_idx])
    return score, float(eta), int(ngrid), best_jump


def tune_dyson_hyperparams_fast(
    precomp: VDEPrecomp,
    p: float,
    omega=0.8,
    jump_frac_candidates=(0.05, 0.1, 0.2),
    solver="auto",
    n_jobs=1,
    missing_policy="zero",
    norm_iters=40,
    random_state=0,
):
    """
    Tune (eta, ngrid, jump_frac) by minimizing edge-instability score.

    Candidate score:
      score = spread(edges over jump_frac) + 1e3 * 1{edge at boundary}.

    Parallelism:
    - n_jobs > 1 evaluates (eta, ngrid) candidates concurrently.
    """
    precomp = _as_precomp(
        precomp,
        missing_policy=missing_policy,
        norm_iters=norm_iters,
        random_state=random_state,
    )

    base_eta = _propose_eta_fast(precomp, p)
    base_ngrid = _propose_ngrid(precomp.n)

    eta_candidates = [base_eta / 2.0, base_eta, base_eta * 2.0]
    eta_candidates = [float(np.clip(e, 1e-4, 5e-2)) for e in eta_candidates]
    ngrid_candidates = [max(60, int(1.5 * precomp.n)), base_ngrid, min(300, int(3 * precomp.n))]

    cfgs = [(eta, ngrid) for eta in eta_candidates for ngrid in ngrid_candidates]
    jump_frac_candidates = tuple(float(x) for x in jump_frac_candidates)

    if int(n_jobs) > 1 and len(cfgs) > 1:
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            results = list(
                ex.map(
                    lambda c: _evaluate_dyson_cfg(
                        precomp,
                        p,
                        c[0],
                        c[1],
                        omega,
                        jump_frac_candidates,
                        solver,
                    ),
                    cfgs,
                )
            )
    else:
        results = [
            _evaluate_dyson_cfg(precomp, p, eta, ngrid, omega, jump_frac_candidates, solver)
            for (eta, ngrid) in cfgs
        ]

    if not results:
        return float(base_eta), int(base_ngrid), 0.1

    best = min(results, key=lambda x: x[0])
    return best[1], best[2], best[3]


def lambda_bulk_dyson_raw_fast(
    precomp: VDEPrecomp,
    p: float,
    eta=1e-3,
    ngrid=120,
    jump_frac=0.1,
    omega=0.8,
    solver="auto",
    missing_policy="zero",
    norm_iters=40,
    random_state=0,
):
    """Compute right bulk edge with user-specified Dyson settings."""
    zs, im_mavg = compute_im_mavg_grid_fast(
        precomp,
        p,
        eta=eta,
        ngrid=ngrid,
        omega=omega,
        solver=solver,
        missing_policy=missing_policy,
        norm_iters=norm_iters,
        random_state=random_state,
    )
    edge, _ = detect_bulk_edge_from_im(zs, im_mavg, jump_frac=jump_frac)
    return float(edge)


def lambda_bulk_dyson_auto_fast(
    precomp: VDEPrecomp,
    p: float,
    omega=0.8,
    solver="auto",
    n_jobs=1,
    missing_policy="zero",
    norm_iters=40,
    random_state=0,
):
    """Auto-tune and compute the right bulk edge from the Dyson profile."""
    precomp = _as_precomp(
        precomp,
        missing_policy=missing_policy,
        norm_iters=norm_iters,
        random_state=random_state,
    )
    eta, ngrid, jump_frac = tune_dyson_hyperparams_fast(
        precomp,
        p,
        omega=omega,
        solver=solver,
        n_jobs=n_jobs,
        missing_policy=missing_policy,
        norm_iters=norm_iters,
        random_state=random_state,
    )
    zs, im_mavg = compute_im_mavg_grid_fast(
        precomp,
        p,
        eta=eta,
        ngrid=ngrid,
        omega=omega,
        solver=solver,
    )
    edge, idx = detect_bulk_edge_from_im(zs, im_mavg, jump_frac=jump_frac)

    info = {
        "eta": float(eta),
        "ngrid": int(ngrid),
        "jump_frac": float(jump_frac),
        "solver": str(solver),
        "q_offdiag_obs": float(precomp.q_offdiag_obs),
        "missing_policy": str(precomp.missing_policy),
        "has_missing_offdiag": bool(precomp.has_missing_offdiag),
        "zs": zs,
        "im_mavg": im_mavg,
        "edge_idx": int(idx),
    }
    return float(edge), info
