import numpy as np
from models.admm_kachun import symNMF
from models.admm import ADMM
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pathlib import Path


def _fit_symnmf(rank, m, nan_mask, seed, **kwargs):
    """Helper function for parallel symNMF fitting."""
    clf = symNMF(
        M_upperbd=(True, np.max(m)),
        M_lowerbd=(True, np.min(m)),
        bsum_iter=50,
        n_components=rank,
        rho=3.0,
        tol=1e-4,
        max_iter=50,
        random_state=seed,
        verbose=False,
    )
    w, loss, bsum_loss = clf.fit_transform(m * nan_mask, nan_mask)
    # Add final W matrix to params for comparison
    clf.params["final_W"] = w
    return np.mean((1 - nan_mask) * (m - (w @ w.T)) ** 2), clf.params


def _fit_admm(rank, m, nan_mask, seed):
    """Helper function for parallel ADMM fitting."""
    training_data = np.where(nan_mask.astype(bool), m, np.nan)
    model = ADMM(
        rank=rank,
        max_outer=50,
        max_inner=50,
        rho=3.0,
        bounds=(np.min(m), np.max(m)),
        tol=1e-4,
        init="eigenspectrum",
        verbose=False,
        random_state=seed,
    )
    try:
        w = model.fit_transform(training_data)
        # Add final W matrix to params for comparison
        model.params["final_W"] = w
        return np.mean((1 - nan_mask) * (m - (w @ w.T)) ** 2), model.params
    except Exception:
        return np.inf, {}


def compare_parameter_evolution(symnmf_params, admm_params, rank):
    """Compare parameter evolution between symNMF and ADMM for a specific rank."""
    if not symnmf_params or not admm_params:
        print(f"Missing parameter data for rank {rank}")
        return

    print(f"\n=== Parameter Evolution for Rank {rank} ===")
    print(
        "Iteration | symNMF W_min | ADMM W_min | symNMF W_max | ADMM W_max | symNMF V_min | ADMM V_min | symNMF lam_min | ADMM lam_min"
    )
    print("-" * 120)

    max_iters = min(len(symnmf_params.get("W", [])), len(admm_params.get("W", [])))

    for i in range(max_iters):
        symnmf_w = (
            symnmf_params["W"][i] if i < len(symnmf_params.get("W", [])) else "N/A"
        )
        admm_w = admm_params["W"][i] if i < len(admm_params.get("W", [])) else "N/A"
        symnmf_w_max = (
            symnmf_params["W_max"][i]
            if i < len(symnmf_params.get("W_max", []))
            else "N/A"
        )
        admm_w_max = (
            admm_params["W_max"][i] if i < len(admm_params.get("W_max", [])) else "N/A"
        )
        symnmf_v = (
            symnmf_params["V"][i] if i < len(symnmf_params.get("V", [])) else "N/A"
        )
        admm_v = admm_params["V"][i] if i < len(admm_params.get("V", [])) else "N/A"
        symnmf_lam = (
            symnmf_params["lam"][i] if i < len(symnmf_params.get("lam", [])) else "N/A"
        )
        admm_lam = (
            admm_params["lam"][i] if i < len(admm_params.get("lam", [])) else "N/A"
        )

        print(
            f"{i:9d} | {symnmf_w:11.6f} | {admm_w:10.6f} | "
            f"{symnmf_w_max:11.6f} | {admm_w_max:10.6f} | "
            f"{symnmf_v:11.6f} | {admm_v:10.6f} | "
            f"{symnmf_lam:13.6f} | {admm_lam:12.6f}"
        )


def simple_cross_validation(
    matrix,
    ranks_to_test=range(5, 15),
    observed_ratio=0.7,
    algorithm="both",
    seed=42,
    n_jobs=1,
    track_parameters=False,
):
    """
    Cross-validation with parallel execution for each model.

    Parameters:
    -----------
    matrix : np.ndarray
        The complete matrix to factorize
    ranks_to_test : range or list
        Ranks to test
    observed_ratio : float
        Fraction of entries to observe (rest are "missing")
    algorithm : str
        "symnmf", "admm", or "both"
    seed : int
        Random seed for reproducible masking
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    track_parameters : bool
        Whether to track and compare parameter evolution
    **kwargs : dict
        Additional parameters passed to the algorithms
    """
    rng = np.random.default_rng(seed)
    m = matrix.copy()

    # EXACT Kachun masking - don't change this!
    nan_mask = rng.random(m.shape) > (1 - observed_ratio)
    nan_mask = nan_mask + nan_mask.T
    nan_mask = nan_mask > 0

    results = {}

    if algorithm in ["symnmf", "both"]:
        if track_parameters and n_jobs == 1:
            # Sequential execution to track parameters
            loss_list_symnmf = []
            params_list_symnmf = []
            for rank in ranks_to_test:
                loss, params = _fit_symnmf(rank, m, nan_mask, seed)
                loss_list_symnmf.append(loss)
                params_list_symnmf.append(params)
        else:
            # Parallel execution
            results_symnmf = Parallel(n_jobs=n_jobs)(
                delayed(_fit_symnmf)(rank, m, nan_mask, seed) for rank in ranks_to_test
            )
            loss_list_symnmf = [r[0] for r in results_symnmf]
            params_list_symnmf = (
                [r[1] for r in results_symnmf] if track_parameters else None
            )

        results["symnmf"] = {
            "losses": loss_list_symnmf,
            "selected_rank": ranks_to_test[np.argmin(loss_list_symnmf)],
            "min_loss": min(loss_list_symnmf),
            "params": params_list_symnmf if track_parameters else None,
        }

    if algorithm in ["admm", "both"]:
        if track_parameters and n_jobs == 1:
            # Sequential execution to track parameters
            loss_list_admm = []
            params_list_admm = []
            for rank in ranks_to_test:
                loss, params = _fit_admm(rank, m, nan_mask, seed)
                loss_list_admm.append(loss)
                params_list_admm.append(params)
        else:
            # Parallel execution
            results_admm = Parallel(n_jobs=n_jobs)(
                delayed(_fit_admm)(rank, m, nan_mask, seed) for rank in ranks_to_test
            )
            loss_list_admm = [r[0] for r in results_admm]
            params_list_admm = (
                [r[1] for r in results_admm] if track_parameters else None
            )

        results["admm"] = {
            "losses": loss_list_admm,
            "selected_rank": ranks_to_test[np.argmin(loss_list_admm)],
            "min_loss": min(loss_list_admm),
            "params": params_list_admm if track_parameters else None,
        }

    return results, nan_mask


def plot_cv_results(results, ranks_to_test, true_rank=None):
    """Plot cross-validation results."""
    n_plots = len(results)
    plt.figure(figsize=(6 * n_plots, 4))

    for i, (name, result) in enumerate(results.items(), 1):
        plt.subplot(1, n_plots, i)
        color = "blue" if name == "symnmf" else "orange"
        plt.plot(ranks_to_test, result["losses"], "o-", label=name.upper(), color=color)

        if true_rank:
            plt.axvline(
                x=true_rank,
                color="red",
                linestyle="--",
                label=f"True rank ({true_rank})",
            )
        plt.axvline(
            x=result["selected_rank"],
            color="green",
            linestyle=":",
            label=f"Selected ({result['selected_rank']})",
        )

        plt.xlabel("Rank")
        plt.ylabel("Loss (SSE on missing entries)")
        plt.title(f"{name.upper()} Cross-validation")
        plt.legend()
        plt.yscale("log")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from pathlib import Path

    n, true_rank = 300, 5
    ranks = range(2, 10)
    seed = 42

    rng = np.random.default_rng(seed)
    xtrue = rng.random((n, true_rank))
    m = np.dot(xtrue, xtrue.T)

    # First run without parameter tracking for speed
    results, mask = simple_cross_validation(
        matrix=m,
        ranks_to_test=ranks,
        observed_ratio=0.3,
        algorithm="both",
        seed=seed,
        n_jobs=-1,
        track_parameters=False,
    )

    if "symnmf" in results:
        print(f"symNMF selected rank: {results['symnmf']['selected_rank']}")
    if "admm" in results:
        print(f"ADMM selected rank: {results['admm']['selected_rank']}")

    plot_cv_results(results, ranks, true_rank)

    # Now run with parameter tracking for selected ranks
    selected_ranks = []
    if "symnmf" in results:
        selected_ranks.append(results["symnmf"]["selected_rank"])
    if "admm" in results:
        selected_ranks.append(results["admm"]["selected_rank"])

    # Also include true rank for comparison
    comparison_ranks = list(set(selected_ranks + [true_rank]))

    print(f"\nRunning parameter tracking for ranks: {comparison_ranks}")

    results_detailed, _ = simple_cross_validation(
        matrix=m,
        ranks_to_test=comparison_ranks,
        observed_ratio=0.3,
        algorithm="both",
        seed=seed,
        n_jobs=1,  # Sequential for parameter tracking
        track_parameters=True,
    )

    output_dir = Path("./")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare parameter evolution for each rank
    for i, rank in enumerate(comparison_ranks):
        if "symnmf" in results_detailed and "admm" in results_detailed:
            symnmf_params = results_detailed["symnmf"]["params"][i]
            admm_params = results_detailed["admm"]["params"][i]

            compare_parameter_evolution(symnmf_params, admm_params, rank)

    # Original comparison plot
    if len(results) == 2:
        plt.figure(figsize=(4, 3), dpi=200)
        plt.plot(ranks, results["symnmf"]["losses"], "o-", label="symNMF")
        plt.plot(ranks, results["admm"]["losses"], "o-", label="ADMM", color="orange")
        plt.axvline(
            x=true_rank, color="red", linestyle="--", label=f"True rank ({true_rank})"
        )
        plt.axvline(
            x=results["symnmf"]["selected_rank"],
            color="blue",
            linestyle=":",
            label=f"symNMF selected ({results['symnmf']['selected_rank']})",
        )
        plt.axvline(
            x=results["admm"]["selected_rank"],
            color="green",
            linestyle=":",
            label=f"ADMM selected ({results['admm']['selected_rank']})",
        )
        plt.xlabel("Rank", fontsize=7)
        plt.ylabel("Loss (MSE on missing entries)", fontsize=7)
        plt.title("Cross-validation loss", fontsize=7)
        plt.legend(fontsize=7)
        plt.yscale("log")
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()

        plt.savefig(output_dir / "admm_vs_symnmf_cv_loss.png", dpi=200)
        plt.close()

    for i, rank in enumerate(comparison_ranks):
        if "symnmf" in results_detailed and "admm" in results_detailed:
            symnmf_w = results_detailed["symnmf"]["params"][i]["final_W"]
            admm_w = results_detailed["admm"]["params"][i]["final_W"]

            print(f"\nRank {rank} Final W matrices:")
            print(f"Are W matrices identical? {np.allclose(symnmf_w, admm_w)}")
            print(f"Max difference: {np.max(np.abs(symnmf_w - admm_w))}")
            print(f"symNMF loss: {results_detailed['symnmf']['losses'][i]}")
            print(f"ADMM loss: {results_detailed['admm']['losses'][i]}")
