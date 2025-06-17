#! /usr/bin/env python3


"""This is for thw simulation to check howe well we can clustering in high and low noise regimes!"""

import pandas as pd
import numpy as np
import itertools
from sklearn.decomposition import NMF
from sklearn.base import clone
from utils.helpers import map_labels_with_hungarian, compute_metrics
from models.admm import ADMM
from utils.metrics import compute_similarity
from utils.simulation import generate_simulation_data
from joblib import Parallel, delayed

from .registry import register_benchmark


def _prepare_nmf_model(x, simulation_params, seed):
    """Prepares the NMF model configuration."""
    X_shifted = x - x.min() if x.min() < 0 else x
    nmf_model = NMF(
        n_components=simulation_params.k,
        init="random",
        solver="mu",
        random_state=seed,
        max_iter=1000,
        tol=0.0,
    )
    return {"data": X_shifted, "model": nmf_model}


def _prepare_symnmf_model(x, simulation_params, similarity_measure, seed):
    """Prepares the SymNMF model configuration using the provided instance."""
    S = compute_similarity(x, x, similarity_measure)
    model = ADMM(
        rank=simulation_params.k,
        random_state=seed,
        max_outer=10,
        max_inner=100,
        tol=1e-4,
        verbose=False,
        init="random_sqrt",
    )
    return {"data": S, "model": model}


def _evaluate_model(true_labels, model_info):
    """Fits a model, predicts labels, computes, and returns metrics."""
    W = model_info["model"].fit_transform(model_info["data"])
    labels = np.argmax(W, axis=1)
    mapped_labels = map_labels_with_hungarian(true_labels, labels)
    return compute_metrics(true_labels, mapped_labels)  # ARI, NMI, Purity, Entropy


def _process_single_combination(
    noise,
    seed,
    simulation_params,
    similarity_measure,
    metric_names,
):
    """Processes a single combination of noise and seed."""
    # Generate data
    current_sim_params = simulation_params.copy()
    current_sim_params.snr = noise
    current_sim_params.rng_state = seed
    X, M = generate_simulation_data(current_sim_params)
    true_labels = np.argmax(M, axis=1)

    local_results = []

    nmf = _prepare_nmf_model(X, current_sim_params, seed)

    # --- Evaluate NMF X ---
    nmf_metrics = _evaluate_model(true_labels, nmf)
    for name, value in zip(metric_names, nmf_metrics):
        local_results.append(
            {
                "SNR": noise,
                "Seed": seed,
                "Model": "NMF X",
                "Metric": name,
                "Value": value,
            }
        )

    symnmf_config = _prepare_symnmf_model(
        X, current_sim_params, similarity_measure, seed
    )
    symnmf_metrics = _evaluate_model(true_labels, symnmf_config)
    for name, value in zip(metric_names, symnmf_metrics):
        local_results.append(
            {
                "SNR": noise,
                "Seed": seed,
                "Model": "SymNMF",
                "Metric": name,
                "Value": value,
            }
        )

    return local_results


@register_benchmark("denoising")
def run_model_comparison_simulation(
    simulation_params,
    seeds=range(10),
    noise_levels=np.linspace(0.0, 1.0, 10),
    similarity_measure="cosine",
):
    """Runs model comparison simulation across noise levels and seeds."""
    metric_names = ["ARI", "NMI", "Accuracy", "Entropy"]

    # Generate all combinations of noise and seed
    combinations = list(itertools.product(noise_levels, seeds))

    # Run combinations in parallel
    # Pass fixed arguments needed by _process_single_combination
    results_list = Parallel(n_jobs=-1)(
        delayed(_process_single_combination)(
            noise, seed, simulation_params, similarity_measure, metric_names
        )
        for noise, seed in combinations
    )

    # Flatten the list of results and convert to DataFrame
    flat_results = [item for sublist in results_list for item in sublist]
    return pd.DataFrame(flat_results)
