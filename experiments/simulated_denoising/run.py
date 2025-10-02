#!/usr/bin/env python3
"""This is for thw simulation to check howe well we can clustering in high and low noise regimes!"""

import pandas as pd
import numpy as np
import itertools
from sklearn.decomposition import NMF
import argparse
from pathlib import Path
from utils.helpers import map_labels_with_hungarian, compute_metrics, accuracy_score
from pysrf import SRF
from tools.rsa import compute_similarity
from utils.simulation import generate_simulation_data, SimulationParams
from joblib import Parallel, delayed


def process_combination(
    n_observations, n_features, sparsity, rank, snr, seed, similarity_measure
):
    """Process a single SNR/seed combination."""
    # Generate simulation data
    sim_params = SimulationParams(
        n=n_observations,
        p=n_features,
        k=rank,
        snr=snr,
        rng_state=seed,
        sparsity=sparsity,
        primary_concentration=5.0,
        base_concentration=0.1,
    )
    data, membership, _, _ = generate_simulation_data(sim_params)
    true_labels = np.argmax(membership, axis=1)

    results = []

    # NMF clustering
    w_nmf = NMF(
        n_components=rank,
        init="random",
        solver="cd",
        random_state=seed,
        max_iter=1000,
        tol=0.0,
    ).fit_transform(data)

    labels_nmf = np.argmax(w_nmf, axis=1)
    mapped_nmf = map_labels_with_hungarian(true_labels, labels_nmf)
    ari, nmi, purity, entropy = compute_metrics(true_labels, mapped_nmf)

    results.append(
        {
            "SNR": snr,
            "Seed": seed,
            "Model": "NMF",
            "ARI": ari,
            "NMI": nmi,
            "Purity": purity,
            "Entropy": entropy,
            "Accuracy": accuracy_score(true_labels, mapped_nmf),
        }
    )

    # SNMF clustering
    similarity_matrix = compute_similarity(data, data, similarity_measure)
    # similarity_matrix = data @ data.T
    # np.fill_diagonal(similarity_matrix, 0.0)
    w_snmf = SRF(
        rank=rank,
        random_state=seed,
        max_outer=1000,
        max_inner=1,
        tol=0.0,
        verbose=False,
        init="random_sqrt",
    ).fit_transform(similarity_matrix)

    labels_snmf = np.argmax(w_snmf, axis=1)
    mapped_snmf = map_labels_with_hungarian(true_labels, labels_snmf)
    ari, nmi, purity, entropy = compute_metrics(true_labels, mapped_snmf)

    results.append(
        {
            "SNR": snr,
            "Seed": seed,
            "Model": "SNMF",
            "ARI": ari,
            "NMI": nmi,
            "Purity": purity,
            "Entropy": entropy,
            "Accuracy": accuracy_score(true_labels, mapped_snmf),
        }
    )

    return results


def run_simulation(
    n_observations,
    n_features,
    sparsity,
    rank,
    snr_levels=np.arange(0.0, 1.0, 0.1),
    seeds=range(10),
    similarity_measure="linear",
    n_jobs=-1,
):
    """Run clustering simulation across SNR levels and seeds."""
    combinations = list(itertools.product(snr_levels, seeds))

    results_list = Parallel(n_jobs=n_jobs)(
        delayed(process_combination)(
            n_observations, n_features, sparsity, rank, snr, seed, similarity_measure
        )
        for snr, seed in combinations
    )

    flat_results = [item for sublist in results_list for item in sublist]
    return pd.DataFrame(flat_results)


def main():
    parser = argparse.ArgumentParser(description="Run clustering simulation")
    parser.add_argument("--n-observations", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument(
        "--output", type=str, default="results/simulated_clustering.csv"
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--similarity", type=str, default="linear")

    args = parser.parse_args()

    df = run_simulation(
        n_observations=args.n_observations,
        n_features=args.n_features,
        rank=args.rank,
        sparsity=args.sparsity,
        seeds=list(range(args.n_seeds)),
        n_jobs=args.n_jobs,
        similarity_measure=args.similarity,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
