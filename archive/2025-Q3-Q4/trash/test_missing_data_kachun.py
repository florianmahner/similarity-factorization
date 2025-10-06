"""
Detailed step-by-step comparison of symNMF and ADMM algorithms.
This script tracks each function call individually to identify exactly where divergence occurs.
"""

import numpy as np
from models.admm_kachun import symNMF
from models.admm import ADMM, update_v, update_lambda, _get_update_w_function
from models.utils import init_factor


def compare_algorithms_detailed(
    n=20,
    rank=4,
    rho=3.0,
    iterations=10,
    missing_ratio=0.25,
    seed=42,
    tol=1e-12,
):
    """Compare algorithms step-by-step with detailed tracking."""

    print(f"=== Detailed Algorithm Comparison ===")
    print(f"Matrix size: {n}x{n}, Rank: {rank}, rho: {rho}")
    print(f"Missing ratio: {missing_ratio}, Seed: {seed}")
    print(f"Tolerance: {tol}")
    print("=" * 50)

    rng = np.random.default_rng(seed)

    # Generate synthetic POSITIVE data for NMF
    # Create a true factorization W @ W.T where W is positive
    w_true = rng.random((n, rank)) * 2.0  # Scale up for better conditioning
    s = w_true @ w_true.T  # This ensures positive semi-definite matrix
    # Add small diagonal for numerical stability
    s += np.eye(n) * 0.01

    # Ensure symmetry (it should already be, but just to be safe)
    s = 0.5 * (s + s.T)

    mask = rng.random((n, n)) >= missing_ratio
    mask = np.triu(mask) + np.tril(mask.T, -1)
    np.fill_diagonal(mask, True)

    # Create data for ADMM (with NaNs for missing values)
    data_admm = s.copy()
    data_admm[~mask.astype(bool)] = np.nan

    # Create data for symNMF (zeros for missing values)
    data_snmf = np.where(mask, s, 0.0)
    nan_mask = mask.astype(float)

    # Initialize both algorithms with EXACTLY the same W
    initial_w = init_factor(s, rank, "random", random_state=seed)
    print(f"Initial W shape: {initial_w.shape}")
    print(f"Initial W range: [{initial_w.min():.6f}, {initial_w.max():.6f}]")
    print(f"Matrix s mean: {s.mean():.6f}, s range: [{s.min():.6f}, {s.max():.6f}]")
    print(f"Matrix is positive: {np.all(s >= 0)}")

    # === symNMF initialization ===
    observed_values = data_admm[~np.isnan(data_admm)]
    bound_min = observed_values.min()
    bound_max = observed_values.max()
    snmf = symNMF(
        n_components=rank,
        rho=rho,
        min_iter=0,
        max_iter=10**9,
        M_lowerbd=(True, bound_min),  # Use data-driven bounds!
        M_upperbd=(True, bound_max),  # Use data-driven bounds!
    )
    snmf.initialize(data_snmf, nan_mask, W_initial=initial_w.copy())

    # === ADMM initialization ===
    admm = ADMM(
        rank=rank,
        rho=rho,
        max_inner=snmf.bsum_iter,
        bounds=(bound_min, bound_max),
        random_state=seed,
    )
    admm._validate_input_arguments()

    # Store the original data with NaNs
    admm.x = data_admm.copy()
    admm._missing_mask = np.isnan(admm.x)
    admm._observation_mask = ~admm._missing_mask

    # Replace NaNs with zeros (this is what ADMM does internally)
    admm.x[admm._missing_mask] = 0.0

    # Set bounds based on observed data only
    observed_values = data_admm[admm._observation_mask]
    bound_min = observed_values.min()
    bound_max = observed_values.max()
    admm.bound_min, admm.bound_max = bound_min, bound_max

    # Force identical initialization
    admm.w = initial_w.copy()
    admm.lam = np.zeros_like(admm.x)

    # Initialize v properly - start with the data (zeros for missing)
    admm.v = admm.x.copy()

    # Get update function
    update_w_func = _get_update_w_function()

    print(f"\nInitial state comparison:")
    print(f"symNMF W range: [{snmf.W.min():.6f}, {snmf.W.max():.6f}]")
    print(f"ADMM w range: [{admm.w.min():.6f}, {admm.w.max():.6f}]")
    print(f"Initial W difference: {np.max(np.abs(snmf.W - admm.w)):.2e}")

    print(f"symNMF Z range: [{snmf.Z.min():.6f}, {snmf.Z.max():.6f}]")
    print(f"ADMM v range: [{admm.v.min():.6f}, {admm.v.max():.6f}]")
    print(f"Initial Z/v difference: {np.max(np.abs(snmf.Z - admm.v)):.2e}")

    print(f"symNMF aZ range: [{snmf.aZ.min():.6f}, {snmf.aZ.max():.6f}]")
    print(f"ADMM lam range: [{admm.lam.min():.6f}, {admm.lam.max():.6f}]")
    print(f"Initial aZ/lam difference: {np.max(np.abs(snmf.aZ - admm.lam)):.2e}")

    print(f"\nData setup verification:")
    print(f"Missing ratio: {np.mean(~mask):.3f}")
    print(f"symNMF data range: [{data_snmf.min():.6f}, {data_snmf.max():.6f}]")
    print(
        f"ADMM data range (after NaN replacement): [{admm.x.min():.6f}, {admm.x.max():.6f}]"
    )
    print(f"Data difference: {np.max(np.abs(data_snmf - admm.x)):.2e}")

    # Main comparison loop
    for iteration in range(1, iterations + 1):
        print(f"\n{'='*20} ITERATION {iteration} {'='*20}")

        # Store states before updates
        snmf_w_before = snmf.W.copy()
        snmf_z_before = snmf.Z.copy()
        snmf_az_before = snmf.aZ.copy()

        admm_w_before = admm.w.copy()
        admm_v_before = admm.v.copy()
        admm_lam_before = admm.lam.copy()

        # === symNMF STEPS ===
        print("\n--- symNMF Updates ---")

        # Step 1: Update W
        target_matrix = snmf.Z + snmf.aZ / snmf.rho
        print(f"1. Computing target matrix Z + aZ/rho")
        print(
            f"   Target matrix range: [{target_matrix.min():.6f}, {target_matrix.max():.6f}]"
        )

        snmf._update_W(target_matrix, snmf.W)
        snmf_w_after_update = snmf.W.copy()
        print(f"   W after update range: [{snmf.W.min():.6f}, {snmf.W.max():.6f}]")
        print(f"   W change magnitude: {np.max(np.abs(snmf.W - snmf_w_before)):.2e}")

        # Step 2: Compute R = W @ W.T
        R = snmf.W @ snmf.W.T
        print(f"2. Computing R = W @ W.T")
        print(f"   R range: [{R.min():.6f}, {R.max():.6f}]")

        # Step 3: Update Z - WITH DETAILED DEBUGGING
        z_target = R - snmf.aZ / snmf.rho
        print(f"3. Computing Z target: R - aZ/rho")
        print(f"   Z target range: [{z_target.min():.6f}, {z_target.max():.6f}]")

        # MANUAL Z UPDATE TO DEBUG
        print(f"   DEBUG: Manual Z calculation")
        print(f"   snmf.M range: [{snmf.M.min():.6f}, {snmf.M.max():.6f}]")
        print(
            f"   snmf.nan_mask range: [{snmf.nan_mask.min():.6f}, {snmf.nan_mask.max():.6f}]"
        )
        print(f"   z_target range: [{z_target.min():.6f}, {z_target.max():.6f}]")

        # Calculate Z manually step by step
        numerator = snmf.M * snmf.nan_mask + snmf.rho * z_target
        denominator = snmf.rho + snmf.nan_mask * 1.0
        z_before_bounds = numerator / denominator

        print(f"   Numerator range: [{numerator.min():.6f}, {numerator.max():.6f}]")
        print(
            f"   Denominator range: [{denominator.min():.6f}, {denominator.max():.6f}]"
        )
        print(
            f"   Z before bounds range: [{z_before_bounds.min():.6f}, {z_before_bounds.max():.6f}]"
        )
        print(f"   symNMF bounds: [{snmf.Mmin:.6f}, {snmf.Mmax:.6f}]")

        # Apply bounds manually
        z_after_bounds = z_before_bounds.copy()
        if snmf.M_lowerbd[0]:
            clipped_lower = np.sum(z_after_bounds < snmf.Mmin)
            z_after_bounds[z_after_bounds < snmf.Mmin] = snmf.Mmin
            print(f"   Clipped {clipped_lower} values to lower bound {snmf.Mmin}")

        if snmf.M_upperbd[0]:
            clipped_upper = np.sum(z_after_bounds > snmf.Mmax)
            z_after_bounds[z_after_bounds > snmf.Mmax] = snmf.Mmax
            print(f"   Clipped {clipped_upper} values to upper bound {snmf.Mmax}")

        print(
            f"   Z after manual bounds: [{z_after_bounds.min():.6f}, {z_after_bounds.max():.6f}]"
        )

        # Now call the actual update and compare
        snmf._update_Z(z_target)
        snmf_z_after_update = snmf.Z.copy()
        print(f"   Z after actual update: [{snmf.Z.min():.6f}, {snmf.Z.max():.6f}]")
        print(
            f"   Manual vs actual difference: {np.max(np.abs(z_after_bounds - snmf.Z)):.2e}"
        )

        # Step 4: Symmetrize Z
        snmf.Z = 0.5 * (snmf.Z + snmf.Z.T)
        print(f"4. Symmetrizing Z")
        print(
            f"   Z after symmetrization range: [{snmf.Z.min():.6f}, {snmf.Z.max():.6f}]"
        )
        print(
            f"   Symmetrization change: {np.max(np.abs(snmf.Z - snmf_z_after_update)):.2e}"
        )

        # Step 5: Update aZ
        snmf.aZ += snmf.rho * (snmf.Z - R)
        print(f"5. Updating aZ += rho * (Z - R)")
        print(f"   aZ after update range: [{snmf.aZ.min():.6f}, {snmf.aZ.max():.6f}]")
        print(f"   aZ change magnitude: {np.max(np.abs(snmf.aZ - snmf_az_before)):.2e}")

        # === ADMM STEPS ===
        print("\n--- ADMM Updates ---")

        # Step 1: Compute t = v + lam/rho
        t = admm.v + admm.lam / admm.rho
        print(f"1. Computing t = v + lam/rho")
        print(f"   t range: [{t.min():.6f}, {t.max():.6f}]")

        # Step 2: Update w
        admm.w = update_w_func(t, admm.w, max_iter=admm.max_inner, tol=admm.tol)
        admm_w_after_update = admm.w.copy()
        print(f"2. Updating w via BSUM")
        print(f"   w after update range: [{admm.w.min():.6f}, {admm.w.max():.6f}]")
        print(f"   w change magnitude: {np.max(np.abs(admm.w - admm_w_before)):.2e}")

        # Step 3: Update v
        admm.v = update_v(
            admm._observation_mask,
            admm.x,
            admm.w,
            admm.lam,
            admm.rho,
            admm.bound_min,
            admm.bound_max,
        )
        admm_v_after_update = admm.v.copy()
        print(f"3. Updating v")
        print(f"   v after update range: [{admm.v.min():.6f}, {admm.v.max():.6f}]")
        print(f"   v change magnitude: {np.max(np.abs(admm.v - admm_v_before)):.2e}")

        # Step 4: Symmetrize v
        admm.v = 0.5 * (admm.v + admm.v.T)
        print(f"4. Symmetrizing v")
        print(
            f"   v after symmetrization range: [{admm.v.min():.6f}, {admm.v.max():.6f}]"
        )
        print(
            f"   Symmetrization change: {np.max(np.abs(admm.v - admm_v_after_update)):.2e}"
        )

        # Step 5: Update lam
        admm.lam = update_lambda(admm.lam, admm.v, admm.w, admm.rho)
        print(f"5. Updating lam")
        print(
            f"   lam after update range: [{admm.lam.min():.6f}, {admm.lam.max():.6f}]"
        )
        print(
            f"   lam change magnitude: {np.max(np.abs(admm.lam - admm_lam_before)):.2e}"
        )

        # === COMPARISON ===
        print(f"\n--- End-of-Iteration Comparison ---")

        w_diff = np.max(np.abs(snmf.W - admm.w))
        z_v_diff = np.max(np.abs(snmf.Z - admm.v))
        az_lam_diff = np.max(np.abs(snmf.aZ - admm.lam))
        rec_diff = np.max(np.abs(snmf.W @ snmf.W.T - admm.w @ admm.w.T))

        print(f"W differences: {w_diff:.2e}")
        print(f"Z/v differences: {z_v_diff:.2e}")
        print(f"aZ/lam differences: {az_lam_diff:.2e}")
        print(f"Reconstruction differences: {rec_diff:.2e}")

        # Check for divergence
        max_diff = max(w_diff, z_v_diff, az_lam_diff, rec_diff)
        if max_diff > tol:
            print(f"\n*** DIVERGENCE DETECTED at iteration {iteration} ***")
            print(f"Maximum difference: {max_diff:.2e} > tolerance {tol:.2e}")

            # Additional diagnostic info
            print(f"\n--- Diagnostic Information ---")
            print(f"symNMF target matrix (Z + aZ/rho) vs ADMM target (v + lam/rho):")
            target_diff = np.max(np.abs(target_matrix - t))
            print(f"   Target matrix difference: {target_diff:.2e}")

            if target_diff > 1e-15:
                print("   -> Input to W update differs between algorithms!")
                print(f"   symNMF Z range: [{snmf.Z.min():.6f}, {snmf.Z.max():.6f}]")
                print(f"   ADMM v range: [{admm.v.min():.6f}, {admm.v.max():.6f}]")
                print(f"   symNMF aZ range: [{snmf.aZ.min():.6f}, {snmf.aZ.max():.6f}]")
                print(
                    f"   ADMM lam range: [{admm.lam.min():.6f}, {admm.lam.max():.6f}]"
                )

            print(f"W update comparison:")
            print(f"   W inputs identical: {target_diff < 1e-15}")
            print(
                f"   W outputs differ by: {np.max(np.abs(snmf_w_after_update - admm_w_after_update)):.2e}"
            )

            # Check the Z/v update differences
            print(f"Z/v update comparison:")
            print(
                f"   symNMF Z target (R - aZ/rho): [{z_target.min():.6f}, {z_target.max():.6f}]"
            )
            ww_admm = admm.w @ admm.w.T
            print(
                f"   ADMM reconstruction (w @ w.T): [{ww_admm.min():.6f}, {ww_admm.max():.6f}]"
            )
            print(f"   Reconstruction difference: {np.max(np.abs(R - ww_admm)):.2e}")

            return False
    else:
        print(f"Algorithms still synchronized (max diff: {max_diff:.2e})")

    print(f"\n*** ALGORITHMS REMAINED SYNCHRONIZED for {iterations} iterations ***")
    print(f"Final maximum difference: {max_diff:.2e}")
    return True


if __name__ == "__main__":
    # Run with different parameters to test
    success = compare_algorithms_detailed(
        n=200,
        rank=10,
        rho=3.0,
        iterations=3,
        missing_ratio=0.2,
        seed=123,
        tol=1e-10,
    )

    if not success:
        print("\nTrying with different parameters...")
        compare_algorithms_detailed(
            n=200,
            rank=10,
            rho=3.0,
            iterations=3,
            missing_ratio=0.2,
            seed=123,
            tol=1e-10,
        )
