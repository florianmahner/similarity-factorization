import numpy as np
from models.admm_kachun import symNMF
from models.admm import ADMM
import matplotlib.pyplot as plt
from pathlib import Path


def compare_optimization_histories():
    n, true_rank = 100, 5
    seed = 42

    rng = np.random.default_rng(seed)
    xtrue = rng.random((n, true_rank))
    m = np.dot(xtrue, xtrue.T)

    observed_ratio = 0.7
    nan_mask = rng.random(m.shape) > (1 - observed_ratio)
    nan_mask = nan_mask + nan_mask.T
    nan_mask = nan_mask > 0

    training_data = np.where(nan_mask.astype(bool), m, np.nan)

    # symNMF
    model_symnmf = symNMF(
        n_components=true_rank,
        rho=3.0,
        max_iter=50,
        bsum_iter=50,
        verbose=True,
        random_state=seed,
        M_lowerbd=(True, m.min()),
        M_upperbd=(True, m.max()),
    )
    w_symnmf, _, _ = model_symnmf.fit_transform(m * nan_mask, nan_mask)

    # ADMM
    model_admm = ADMM(
        rank=true_rank,
        rho=3.0,
        max_outer=50,
        max_inner=50,
        verbose=True,
        random_state=seed,
        init="eigenspectrum",
        bounds=(m.min(), m.max()),
    )
    model_admm.fit(training_data)

    # Print parameter histories
    print("\n=== Parameter Histories (Sum) ===")
    print(
        "Iteration | symNMF W_sum     | ADMM W_sum    | symNMF V_sum     | ADMM V_sum    | symNMF lam_sum   | ADMM lam_sum  "
    )
    print("-" * 90)

    max_iters = min(len(model_symnmf.params["W_sum"]), len(model_admm.params["W_sum"]))

    for i in range(max_iters):
        print(
            f"{i:9d} | {model_symnmf.params['W_sum'][i]:14.6f} | {model_admm.params['W_sum'][i]:13.6f} | "
            f"{model_symnmf.params['V_sum'][i]:14.6f} | {model_admm.params['V_sum'][i]:13.6f} | "
            f"{model_symnmf.params['lam_sum'][i]:15.6f} | {model_admm.params['lam_sum'][i]:13.6f}"
        )
    plt.figure(figsize=(12, 4), dpi=200)

    plt.subplot(1, 3, 1)
    plt.plot(model_symnmf.params["W"], "o-", label="symNMF", alpha=0.7)
    plt.plot(model_admm.params["W"], "s-", label="ADMM", alpha=0.7)
    plt.xlabel("Iteration", fontsize=7)
    plt.ylabel("W minimum", fontsize=7)
    plt.title("W parameter evolution", fontsize=7)
    plt.legend(fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.subplot(1, 3, 2)
    plt.plot(model_symnmf.params["V"], "o-", label="symNMF", alpha=0.7)
    plt.plot(model_admm.params["V"], "s-", label="ADMM", alpha=0.7)
    plt.xlabel("Iteration", fontsize=7)
    plt.ylabel("V minimum", fontsize=7)
    plt.title("V parameter evolution", fontsize=7)
    plt.legend(fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.subplot(1, 3, 3)
    plt.plot(model_symnmf.params["lam"], "o-", label="symNMF", alpha=0.7)
    plt.plot(model_admm.params["lam"], "s-", label="ADMM", alpha=0.7)
    plt.xlabel("Iteration", fontsize=7)
    plt.ylabel("Lambda minimum", fontsize=7)
    plt.title("Lambda parameter evolution", fontsize=7)
    plt.legend(fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.tight_layout()

    output_dir = Path("./")
    plt.savefig(output_dir / "parameter_evolution_comparison.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    compare_optimization_histories()
