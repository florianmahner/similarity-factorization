import numpy as np
from scipy.optimize import root_scalar
from numpy.linalg import norm
from tqdm import tqdm
from numpy.linalg import eigvalsh
from kneed import KneeLocator

from sklearn.decomposition import KernelPCA
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer
from pyclustering.cluster.gmeans import gmeans

import matplotlib.pyplot as plt

def estimate_p_bound(
    S,
    gamma=1.05, # pmin
    eta=0.05, # pmin
    rho=0.95, # pmin
    method='dyson', # pmax
    omega=0.8, # pmax
    eta_pmax=1e-3, # pmax
    jump_frac=0.1, # pmax
    tol=1e-4, # pmax
    gap=0.05,
    verbose=False, 
    random_state=31213,
):

    pmin, _, _, _, _ = pmin_bound(
        S, 
        gamma=gamma,
        eta=eta,
        rho=rho, 
        random_state=random_state, verbose=verbose)

    eff_dim = np.ceil((np.linalg.norm(S, 'fro') / np.linalg.norm(S, 2))**2).astype(int)
    
    pmax = p_upper_only_k(
        S, 
        k=eff_dim, 
        method=method,
        tol=tol, 
        omega=omega, 
        eta=eta_pmax, 
        jump_frac=jump_frac,
        verbose=verbose, 
        seed=random_state)

    S_noise = S

    if pmin > pmax-gap:
        print('Noise regime triggered')
        print(f"pmin = {pmin}, pmax = {pmax}")
        
        epsilon = np.linalg.norm(S, 2)/np.sqrt(S.shape[0])

        t_range = np.linspace(0.0, epsilon, 10) 
        eff_dim_list = []
        pmin_list = []
        pmax_list = []

        # future updates ( n realization of A instead of 1)
        A = np.random.rand(S.shape[0], S.shape[1])

        t_threshold = 0

        t_iter = iter(t_range)
        t = next(t_iter)
        
        while True:
            S_noise = S + t * (A + A.T)
            
            pmin, _, _, _, _ = pmin_bound(
                S_noise,
                gamma=gamma,
                eta=eta,
                rho=rho, 
                random_state=random_state, verbose=verbose)
            
            eff_dim = np.ceil((np.linalg.norm(S_noise, 'fro') / np.linalg.norm(S_noise, 2))**2).astype(int)
            pmax = p_upper_only_k(
                S_noise, 
                k=eff_dim, 
                method=method,
                tol=tol, 
                omega=omega, 
                eta=eta_pmax, 
                jump_frac=jump_frac,
                verbose=verbose, 
                seed=random_state)
            if verbose:
                print(t, pmin, eff_dim, pmax)
                
            eff_dim_list.append(eff_dim)    
            pmin_list.append(pmin)
            pmax_list.append(pmax)

            if pmin < pmax - gap:
                t_threshold = t
                break  # stop once condition is met
        
            try:
                t = next(t_iter)
            except StopIteration:
                break  # exit if no more t values

        S_noise = S + t_threshold * (A + A.T)

    return pmin, pmax, S_noise








def pmin_bound(S, gamma=1.05, eta=0.05, rho=0.95, n_realizations=500, random_state=None, verbose=False, monte_carlo=False):

    np.random.seed(random_state)
    n = S.shape[0]
    # assert np.allclose(S, S.T), "Matrix S must be symmetric"
    _is_symmetric = np.allclose(S, S.T)

    # compute Sigma(Δ)^2 = || \sum_{Δ} || = p(1 - p)*L_max
    # where L_max := (\max_i \sum_{j!=i} S_{ij}^2)
    L_max = np.max( (S ** 2).sum(axis=1) - np.diag(S) ** 2 )

    # empirical estimation of L_max
    # replace max_i[\sum_{j != i} S_{ij}}^2] with mean[\sum_{j != i} S_{ij}}^2]
    _row_sq = (S ** 2).sum(axis=1) - np.diag(S) ** 2
    # empirical_L_max = _row_sq.sum() / n
    empirical_L_max = np.quantile(_row_sq, rho)
    
    # compute || S || = \max_{ ||x||_2 = 1 } || Sx ||_2 = \lambda_{\max}(S)
    S_norm = np.linalg.norm(S, 2)

    # compute || S ||_{\infty}
    L_infty = 2*np.max(np.abs(S))
    empirical_L_infty = 2*np.quantile(np.abs(S), rho)

    # compute effect dimension
    effective_dimension = ( np.linalg.norm(S, 'fro') / S_norm ) ** 2
    if verbose:
        print('effective dimension : ', effective_dimension)
    
    # monte-carlo estimation of E[|| M * S ||]
    MC_expected_MS_norms = np.zeros(n_realizations)
    if monte_carlo:
        for i in range(n_realizations):
            _p = np.random.rand()
            _mask = np.random.binomial(1, _p, size=S.shape)
            if _is_symmetric:
                _mask = np.triu(_mask, 1)
                _mask += _mask.T
            MC_expected_MS_norms[i] = np.linalg.norm( _mask * S, 2 )

    N_bernstein = (gamma * L_infty * S_norm) / (3 * L_max) + 1
    N_empirical = (gamma * empirical_L_infty * S_norm) / (3 * empirical_L_max) + 1
    N_empirical_alternative = (gamma * L_infty * S_norm) / (3 * empirical_L_max) + 1
    N_theory_upperbound = (gamma * S_norm) / 3 + 1

    # D_bernstein = (gamma * S_norm)**2 / (L_max * np.log( 2 * n / eta )) + 2
    # D_empirical = (gamma * S_norm)**2 / (empirical_L_max * np.log( 2 * effective_dimension / eta )) + 2
    # D_theory_lowerbound = (gamma**2 * S_norm) / np.log( 2 * n / eta ) + 2

    D_bernstein = ((gamma * S_norm)**2 / (2 * L_max) + 1) / np.log( 2 * n / eta )
    D_empirical = ((gamma * S_norm)**2 / (2 * empirical_L_max) + 1) / np.log( 2 * effective_dimension / eta )
    D_theory_lowerbound = ((gamma**2 * S_norm) / (2 * empirical_L_max) + 1) / np.log( 2 * n / eta )
    

    if verbose:
        print(N_bernstein, D_bernstein)
        print(N_empirical, D_empirical)
        print(N_empirical_alternative, D_empirical)
        print(N_theory_upperbound, D_theory_lowerbound)
    
    p_min = N_bernstein / D_bernstein
    p_min_empirical = N_empirical / D_empirical
    p_min_empirical_alternative = N_empirical_alternative / D_empirical
    p_min_lowerbound = N_theory_upperbound / D_theory_lowerbound

    if verbose:
        print( 'p_min', p_min )
        print( 'empirical_p_min', p_min_empirical )
        print( 'empirical_p_min_alternative', p_min_empirical_alternative )
        print( 'theory_p_min', p_min_lowerbound )
    
    return p_min_empirical, p_min, p_min_lowerbound, p_min_empirical_alternative, MC_expected_MS_norms


# ---------- RAW bulk edges ----------
def _solve_vde(S, p, z, eta=1e-3, max_iter=2000, tol=1e-7, omega=0.8, warm=None):
    V = p*(1-p)*(S**2)
    w = z + 1j*eta
    m = -np.ones(S.shape[0], dtype=complex)/w if warm is None else warm
    for _ in range(max_iter):
        denom = w + V.dot(m)
        denom[np.abs(denom) < 1e-16] = 1e-16
        m_new = -1.0/denom
        diff = m_new - m
        m = omega*m_new + (1-omega)*m
        if np.max(np.abs(diff)) < tol:
            break
    return m

from matplotlib import pyplot

# def lambda_bulk_dyson_raw(S, p, omega=0.8, eta=1e-3, ngrid=120, dyson_tol=1e-4):
#     if p <= 0 or p >= 1:  # bulk collapses at boundaries
#         return 0.0
#     s2_max = np.max(eigvalsh(S**2))
#     z_max = np.linalg.norm(S,2) + 8.0*np.sqrt(p*(1-p)*s2_max)
#     z_min = 1e-8
#     zs = np.linspace(z_max, z_min, ngrid)
#     warm = None
#     for z in zs:
#         m = _solve_vde(S, p, z, eta=eta, warm=warm, omega=omega); warm = m
#         if np.imag(np.mean(m)) > dyson_tol:
#             return float(z)

#     return float(zs[-1])

def lambda_bulk_dyson_raw(S, p, omega=0.8, eta=1e-3, ngrid=100, jump_frac=0.1):

    if p <= 0 or p >= 1:  # bulk collapses at boundaries
        return 0.0

    n = S.shape[0]
    s2_max = np.max(eigvalsh(S**2))
    z_max = np.linalg.norm(S, 2) + 8.0 * np.sqrt(p * (1 - p) * s2_max)
    z_min = 1e-8
    zs = np.linspace(z_max, z_min, ngrid)

    # solve VDE at all grid points
    warm = None
    im_mavg = []
    for z in zs:
        m = _solve_vde(S, p, z, eta=eta, warm=warm, omega=omega)
        warm = m
        im_mavg.append(np.imag(np.mean(m)))

    im_mavg = np.array(im_mavg)
    # compute jumps between consecutive points
    jumps = np.diff(im_mavg)
    max_jump = np.max(jumps)
    threshold = jump_frac * max_jump

    # find first x where Im(m_avg) exceeds threshold
    idx = np.argmax(im_mavg > threshold)

    return float(zs[idx])

def monte_carlo_bulk_edge_raw(S, p, n_trials=400, quantile=0.9, seed=None):
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    max_eigs = []
    for _ in range(n_trials):
        # symmetric centered Bernoulli mask
        eps = rng.binomial(1, p, size=(n,n)).astype(float) - p
        eps = np.triu(eps, 1); eps = eps + eps.T
        np.fill_diagonal(eps, rng.binomial(1, p, size=n).astype(float) - p)
        Delta = eps * S  # RAW scale
        w = eigvalsh(Delta)
        max_eigs.append(w[-1])
    return float(np.quantile(max_eigs, quantile))

# ---------- main ----------
def p_upper_only_k(S, k=1, method='dyson', mc_trials=600, mc_quantile=0.9,
                   tol=1e-4, verbose=False, seed=None, omega=0.8, eta=1e-3, jump_frac=0.1):
    lam = np.sort(eigvalsh(S))[::-1]
    n = len(lam)
    if not (1 <= k <= n):
        raise ValueError("k must be between 1 and n")
    lam_k  = lam[k-1]
    lam_k1 = lam[k] if k < n else None

    if lam_k <= 0:
        if verbose: print("λ_k ≤ 0 → no positive spike to separate.")
        return 0.0
    if (lam_k1 is None) or (lam_k1 <= 0):
        # If there is no (k+1)-th positive spike, then for large p the bulk --> 0
        # and only the first k remain out. Upper bound is 1.0.
        if verbose: print("λ_{k+1} ≤ 0 → only first k can be out for all large p; return 1.0.")
        return 1.0

    edge = (lambda p: lambda_bulk_dyson_raw(S, p, omega=omega, eta=eta, jump_frac=jump_frac)) if method=='dyson' else \
           (lambda p: monte_carlo_bulk_edge_raw(S, p, n_trials=mc_trials,
                                                quantile=mc_quantile, seed=seed))

    # Count how many spikes are out at p
    def count_out(p):
        e = edge(p)
        return int(np.sum(p*lam > e)), e

    # Quick sanity: at p close to 1, usually more spikes pop out
    c_hi, e_hi = count_out(0.99)
    if verbose:
        print(f"[sanity] p=0.99: bulk={e_hi:.4g}, count_out={c_hi}, λ1={lam[0]:.4g}, λ2={lam[1] if n>1 else np.nan:.4g}")

    # If even at p≈1 we have < k spikes out, the spikes are too weak → 1
    if c_hi < k:
        if verbose: print(f"Even at p≈1, only {c_hi} spikes out (< k). Returning 1.0.")
        return 1.0

    # If at p \approx 1 we already have >= k+1 out, we can try to push boundary leftward
    # We want the largest p where exactly k are out
    # First find any p with exactly k out
    grid = np.linspace(0.02, 0.99, 80)
    feas = [p for p in grid if count_out(p)[0] == k]
    if not feas:
        # If no exact-k point on the grid, it usually means:
        #   - always >= k+1 (then take the leftmost boundary)
        #   - or always < k (already handled above)

        
        # bisect on g(p) = p*λ_{k+1} - edge(p) to find where (k+1) emerges.
        def g(p): return p*lam_k1 - edge(p)
        a, b = 1e-3, 0.99
        ga, gb = g(a), g(b)
        
        # If (k+1) is already out at small p, return small boundary; else if never out, return 1
        if ga >= 0 and gb >= 0:
            # (k+1) out everywhere on [a,b] → largest p with exactly k does not exist; return smallest feasible
            if verbose: print("(k+1) spike is out for all p; returning smallest p where count==k (none found) → 0.")
            return 0.0

        if ga < 0 and gb <= 0:
            # (k+1) never out up to 0.99 --> at p≈1 exactly k out --> 1.0
            if verbose: print("(k+1) never emerges up to 0.99; returning 1.0.")
            return 1.0

            
        # Bisection to the root g(p)=0 from the right (emergence point)
        lo, hi = a, b
        for _ in range(60):
            mid = 0.5*(lo+hi)
            if g(mid) >= 0: hi = mid
            else:           lo = mid
            if (hi-lo) < tol: break
                
        p_star = max(0.0, min(1.0, lo - 2*tol))
        
        return p_star

    # We have feasible regions --> take the rightmost p with exactly k and refine boundary upward
    p_lo = max(feas)
    # Grow to where k+1 pops out
    def cond_ge_kplus1(p): return count_out(p)[0] >= (k+1)
    p_hi = min(0.99, p_lo + 0.05)
    # expand upward until k+1 is out or hit 0.99
    while (p_hi < 0.99) and (not cond_ge_kplus1(p_hi)):
        p_hi = min(0.99, p_hi + 0.05)
    if not cond_ge_kplus1(p_hi):
        # never reaches k+1 --> 1.0 works
        return 1.0
    # bisection
    lo, hi = p_lo, p_hi
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if cond_ge_kplus1(mid): hi = mid
        else:                   lo = mid
        if (hi-lo) < tol: break
    p_star = max(0.0, min(1.0, lo))
    if verbose:
        c_star, e_star = count_out(p_star)
        print(f"p*={p_star:.4f}, bulk={e_star:.6g}, count_out(p*)={c_star}")
    return p_star



import numpy as np
from numpy.linalg import eigvalsh, norm
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import eigh

def rmt_effective_dimension(S, verbose=False):
    """
    Estimate effective dimension of S using random matrix theory (bulk edge).
    
    Parameters
    ----------
    S : (n,n) symmetric ndarray
        The similarity / covariance matrix.
    verbose : bool
        If True, prints intermediate info.
    
    Returns
    -------
    d_eff : int
        Effective dimension (number of spikes above bulk)
    spikes : ndarray
        Eigenvalues that exceed bulk edge
    """
    S = np.asarray(S, dtype=float)
    n = S.shape[0]
    sigma = eigvalsh(S)[::-1]  # descending order
    
    # Step 1: estimate bulk variance from smallest n/2 eigenvalues
    # crude heuristic: assume half are noise
    # k = max(1, n // 4)
    k, _ = aic_dimension(S)
    # k = int(np.ceil ( (np.linalg.norm(S, 'fro') / np.linalg.norm(S, 2))**2 ) )
    
    noise_eigs = sigma[k:]
    sigma_bulk = np.sqrt(np.mean(noise_eigs**2))
    
    # Marchenko-Pastur bulk edge for square matrix (γ=1)
    lambda_plus = sigma_bulk * (1 + np.sqrt(1.0))**2  # ≈ 4 * sigma_bulk
    
    # Step 2: count spikes above bulk edge
    spikes = sigma[sigma > lambda_plus]
    d_eff = len(spikes)
    
    if verbose:
        print(f"Bulk edge λ+ ≈ {lambda_plus:.4f}")
        print(f"Effective dimension = {d_eff}")
        print("Spikes above bulk:", spikes)
    
    return d_eff, spikes

def rmt_effective_dimension_auto(S, gamma=1.0, model="MP", tol=1e-6, max_iter=50, verbose=False):
    """
    Estimate effective dimension of S using self-consistent RMT bulk edge.
    
    Parameters
    ----------
    S : (n,n) symmetric ndarray
        Similarity or covariance matrix.
    gamma : float
        Aspect ratio n/m (MP model), default 1 for square.
    model : str
        "MP" for Marchenko–Pastur bulk, "Wigner" for symmetric noise.
    tol : float
        Convergence tolerance for iterative bulk edge fitting.
    max_iter : int
        Maximum iterations for self-consistent fitting.
    verbose : bool
        Print diagnostics.
    
    Returns
    -------
    d_eff : int
        Effective dimension (number of spikes above bulk).
    spikes : ndarray
        Eigenvalues above the bulk edge.
    lambda_plus : float
        Estimated bulk edge.
    """
    S = np.asarray(S, dtype=float)
    n = S.shape[0]
    evals = eigvalsh(S)
    evals.sort()  # ascending
    
    # Start with all but the largest eigenvalue as "noise"

    k, _ = bic_dimension(S)
    
    noise_eigs = evals[:-(k+1)]
    
    for it in range(max_iter):
        sigma_bulk = np.sqrt(np.mean(noise_eigs**2))
        
        if model == "MP":
            lambda_plus = sigma_bulk * (1 + np.sqrt(gamma))**2
        elif model == "Wigner":
            lambda_plus = 2 * sigma_bulk
        else:
            raise ValueError("model must be 'MP' or 'Wigner'")
        
        new_noise_eigs = evals[evals <= lambda_plus + tol]
        
        if len(new_noise_eigs) == len(noise_eigs):
            break
        noise_eigs = new_noise_eigs
    
    spikes = evals[evals > lambda_plus]
    d_eff = len(spikes)
    
    if verbose:
        print(f"Bulk edge λ+ ≈ {lambda_plus:.6f}")
        print(f"Effective dimension = {d_eff}")
        print(f"Spikes above bulk edge: {spikes[::-1]}")
    
    return d_eff, spikes[::-1], lambda_plus

def bic_dimension(S, r_max=None):
    """
    Estimate the intrinsic dimension of a symmetric similarity matrix using BIC.
    
    Parameters
    ----------
    S : (n, n) ndarray
        Symmetric similarity matrix.
    r_max : int, optional
        Maximum rank to consider. Default: min(20, n-1)
        
    Returns
    -------
    r_bic : int
        Rank minimizing BIC.
    bic_values : ndarray
        BIC values for each rank 1..r_max
    """
    n = S.shape[0]
    if r_max is None:
        r_max = min(20, n-1)
    
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    bic_values = []
    
    # Number of independent entries in symmetric matrix
    m = n * (n + 1) / 2
    
    for r in range(1, r_max + 1):
        Ur = eigvecs[:, :r]
        Lr = np.diag(eigvals[:r])
        S_hat = Ur @ Lr @ Ur.T
        
        residual = S - S_hat
        sigma2 = np.sum(residual**2) / m
        
        k_r = n * r - r*(r-1)//2
        bic = m * np.log(sigma2) + k_r * np.log(m)
        bic_values.append(bic)
    
    bic_values = np.array(bic_values)
    r_bic = np.argmin(bic_values) + 1
    
    return r_bic, bic_values

    
def aic_dimension(S, r_max=None):
    """
    Estimate the intrinsic dimension of a symmetric similarity matrix using AIC.
    
    Parameters
    ----------
    S : (n, n) ndarray
        Symmetric similarity matrix.
    r_max : int, optional
        Maximum rank to consider. Default: min(20, n-1)
        
    Returns
    -------
    r_aic : int
        Rank minimizing AIC.
    aic_values : ndarray
        AIC values for each rank 1..r_max
    """
    n = S.shape[0]
    if r_max is None:
        r_max = min(20, n-1)
    
    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(S)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    aic_values = []
    
    # Number of independent entries in symmetric matrix
    m = n * (n + 1) / 2
    
    for r in range(1, r_max + 1):
        # Low-rank approximation
        Ur = eigvecs[:, :r]
        Lr = np.diag(eigvals[:r])
        S_hat = Ur @ Lr @ Ur.T
        
        # Residual variance
        residual = S - S_hat
        sigma2 = np.sum(residual**2) / m
        
        # Number of parameters in rank-r symmetric factorization
        k_r = n * r - r*(r-1)//2
        
        # AIC
        aic = m * np.log(sigma2) + 2 * k_r
        aic_values.append(aic)
    
    aic_values = np.array(aic_values)
    r_aic = np.argmin(aic_values) + 1  # ranks start at 1
    
    return r_aic, aic_values



def scree_plot_with_knee(similarity_matrix):
    """
    Create a scree plot of eigenvalues from a symmetric similarity matrix
    and estimate the knee (elbow) point.

    Parameters:
        similarity_matrix (numpy.ndarray): Symmetric similarity matrix (n x n)

    Returns:
        knee (int): Index of the knee point in eigenvalues
        eigenvalues (numpy.ndarray): Sorted eigenvalues
    """
    # Ensure matrix is symmetric
    if not np.allclose(similarity_matrix, similarity_matrix.T):
        raise ValueError("Matrix must be symmetric")

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(similarity_matrix)
    eigenvalues = np.flip(np.sort(eigenvalues))  # sort descending

    # Find the knee point
    x = np.arange(1, len(eigenvalues) + 1)
    kneedle = KneeLocator(x, eigenvalues, curve="convex", direction="decreasing")
    knee = kneedle.knee

    # Plot scree plot
    plt.figure(figsize=(6, 4))
    plt.plot(x, eigenvalues, 'bo-', label="Eigenvalues")
    if knee is not None:
        plt.axvline(x=knee, color='r', linestyle='--', label=f"Knee at {knee}")
        plt.scatter(knee, eigenvalues[knee-1], color='red', s=80, zorder=5)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title("Scree Plot of Eigenvalues")
    plt.legend()
    plt.grid(True)
    plt.show()

    return knee, eigenvalues


def ncluster_Gmean(S, n_components=10, repeat=5):
    # need raw points, so use PCA embedding
    X_pca = PCA(n_components=n_components).fit_transform(S)
    
    # G-means
    gm = gmeans(X_pca, repeat=repeat)
    gm.process()
    clusters_gmeans = gm.get_clusters()
    
    print("\n[pyclustering results]")
    # print(f"X-means estimated clusters = {len(clusters_xmeans)}")
    print(f"G-means estimated clusters = {len(clusters_gmeans)}")
    return len(clusters_gmeans)



def make_psd(S, tol=1e-8):
    """Project a symmetric matrix onto the PSD cone."""
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals[eigvals < tol] = tol
    S_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return S_psd

def ncluster_kernelPCA(S, variance_explain=0.9):
    
    kpca_full = KernelPCA(n_components=None, kernel="precomputed", fit_inverse_transform=False, random_state=31213)
    kpca_full.fit(make_psd(S))
    
    eigenvalues = kpca_full.eigenvalues_
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Step 2: choose embedding dimension by variance threshold
    n_dims = np.searchsorted(cumulative_variance, variance_explain) + 1
    
    # Step 3: Embed into chosen dimension
    kpca = KernelPCA(n_components=n_dims, kernel="precomputed", random_state=31213)
    X_embedded = kpca.fit_transform(make_psd(S))
    
    
    # DP-GMM on embedded space
    dpgmm = BayesianGaussianMixture(
        n_components=50,
        covariance_type='full',
        weight_concentration_prior_type='dirichlet_process',
        random_state=31213
    )
    labels = dpgmm.fit_predict(X_embedded)
    
    print("kernelPCA : Estimated clusters:", len(set(labels)))
    return len(set(labels))
