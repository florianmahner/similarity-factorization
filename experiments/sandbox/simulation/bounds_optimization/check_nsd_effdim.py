"""Check NSD effective dimension and eigenvalue structure."""

import numpy as np
from pysrf.bounds import p_upper_only_k_fast


def main():
    print("Loading saved NSD RSM...")
    S = np.load("/tmp/nsd_subj1_rsm.npy")
    print(f"Shape: {S.shape}")

    # Compute norms
    fro_norm = np.linalg.norm(S, "fro")
    spec_norm = np.linalg.norm(S, 2)

    eff_dim_raw = (fro_norm / spec_norm) ** 2
    eff_dim = int(np.ceil(eff_dim_raw))

    print(f"\nFrobenius norm: {fro_norm:.4f}")
    print(f"Spectral norm: {spec_norm:.4f}")
    print(f"Effective dimension (raw): {eff_dim_raw:.4f}")
    print(f"Effective dimension (ceil): {eff_dim}")

    # Eigenvalue structure (top few only for speed)
    print("\nComputing top eigenvalues...")
    from scipy.sparse.linalg import eigsh
    eigvals, _ = eigsh(S, k=min(20, S.shape[0]-1), which='LM')
    eigvals = np.sort(eigvals)[::-1]

    print(f"\nTop 10 eigenvalues:")
    for i, ev in enumerate(eigvals[:10]):
        print(f"  λ_{i+1} = {ev:.4f}")

    print(f"\nλ_1 / λ_2 ratio: {eigvals[0] / eigvals[1]:.2f}")

    # Test pmax with different k values
    print("\n--- pmax with different k values ---")
    for k in [1, eff_dim]:
        print(f"Testing k={k}...")
        pmax = p_upper_only_k_fast(S, k=k, verbose=True, seed=0, n_jobs=1)
        print(f"  k={k}: pmax = {pmax:.4f}\n")


if __name__ == "__main__":
    main()
