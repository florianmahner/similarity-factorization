import numpy as np


def simulate_similarity_matrix(
    n_objects, true_rank, observed_fraction=1.0, snr=np.inf, seed=42
):
    """Generate similarity matrix with optional noise and masking."""
    rng = np.random.default_rng(seed)

    w_true = rng.random((n_objects, true_rank))
    s_matrix = w_true @ w_true.T

    if snr < np.inf:
        noise = rng.normal(0, 1, s_matrix.shape)
        noise = (noise + noise.T) / 2
        np.fill_diagonal(noise, 0)
        signal_power = np.var(s_matrix[np.triu_indices_from(s_matrix, k=1)])
        noise_power = signal_power / snr
        s_matrix += noise * np.sqrt(noise_power)

    s_matrix = np.clip(s_matrix, 0, None)
    np.fill_diagonal(s_matrix, np.diag(w_true @ w_true.T))

    if observed_fraction < 1.0:
        mask = rng.random(s_matrix.shape) < observed_fraction
        mask = np.triu(mask) + np.triu(mask, 1).T
        np.fill_diagonal(mask, True)
        s_matrix = np.where(mask, s_matrix, np.nan)

    return s_matrix


def evaluate_condition(
    n_objects,
    true_rank,
    observed_fraction,
    snr,
    rho,
    max_outer,
    max_inner,
    trial_id,
    seed,
):
    """Evaluate rank detection for single parameter combination."""
    s_matrix = simulate_similarity_matrix(
        n_objects, true_rank, observed_fraction, snr, seed
    )

    rank_range = list(range(max(1, true_rank - 2), true_rank + 4))
    param_grid = {
        "rank": rank_range,
        "max_outer": [max_outer],
        "max_inner": [max_inner],
        "rho": [rho],
        "tol": [0.0],
    }

    cv_results = cross_val_score(
        s_matrix,
        param_grid=param_grid,
        n_repeats=1,
        observed_fraction=0.8,
        n_jobs=1,
        random_state=seed,
        verbose=0,
    )

    scores = cv_results.cv_results_.groupby("rank")["score"].mean()
    best_rank = scores.idxmin()

    return {
        "n_objects": n_objects,
        "true_rank": true_rank,
        "observed_fraction": observed_fraction,
        "snr": snr,
        "rho": rho,
        "max_outer": max_outer,
        "max_inner": max_inner,
        "trial_id": trial_id,
        "best_rank": best_rank,
        "best_score": scores.min(),
        "rank_correct": best_rank == true_rank,
        "rank_error": abs(best_rank - true_rank),
        "seed": seed,
    }


def run_rank_experiment(
    n_objects_list=[200, 500],
    true_ranks=[10, 20],
    observed_fractions=[1.0],
    snr_values=[np.inf],
    rho_values=[1.0],
    max_outer_values=[20],
    max_inner_values=[50],
    n_trials=5,
    n_jobs=-1,
    output_dir="results",
):
    """Main experiment orchestrator."""
    conditions = []
    seed_counter = 0

    for n_objects in n_objects_list:
        for true_rank in true_ranks:
            for observed_fraction in observed_fractions:
                for snr in snr_values:
                    for rho in rho_values:
                        for max_outer in max_outer_values:
                            for max_inner in max_inner_values:
                                for trial_id in range(n_trials):
                                    conditions.append(
                                        (
                                            n_objects,
                                            true_rank,
                                            observed_fraction,
                                            snr,
                                            rho,
                                            max_outer,
                                            max_inner,
                                            trial_id,
                                            seed_counter,
                                        )
                                    )
                                    seed_counter += 1

    print(f"Running {len(conditions)} conditions")
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(evaluate_condition)(*condition) for condition in conditions
    )

    results_df = pd.DataFrame(results)

    output_path = Path(output_dir) / "rank_experiment.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    return results_df
