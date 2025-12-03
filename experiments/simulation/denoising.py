from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.decomposition import NMF
from pysrf import SRF
from tools.rsa import compute_similarity
from utils.helpers import accuracy_score, compute_metrics, map_labels_with_hungarian
from utils.simulation import SimulationParams, generate_simulation_data


def _score_labels(true_labels: np.ndarray, predicted: np.ndarray, model: str) -> dict:
    mapped = map_labels_with_hungarian(true_labels, predicted)
    ari, nmi, purity, entropy = compute_metrics(true_labels, mapped)
    return {
        "Model": model,
        "ARI": ari,
        "NMI": nmi,
        "Purity": purity,
        "Entropy": entropy,
        "Accuracy": accuracy_score(true_labels, mapped),
    }


def run(cfg: DictConfig) -> None:
    sim_params = SimulationParams(
        n=cfg.n_observations,
        p=cfg.n_features,
        k=cfg.rank,
        snr=cfg.snr,
        rng_state=cfg.seed,
        sparsity=cfg.sparsity,
        primary_concentration=5.0,
        base_concentration=0.1,
    )
    data, membership, _, _ = generate_simulation_data(sim_params)
    true_labels = np.argmax(membership, axis=1)

    results = []

    # NMF baseline
    w_nmf = NMF(
        n_components=cfg.rank,
        init="random",
        solver="cd",
        random_state=cfg.seed,
        max_iter=1000,
        tol=0.0,
    ).fit_transform(data)
    labels_nmf = np.argmax(w_nmf, axis=1)
    nmf_scores = _score_labels(true_labels, labels_nmf, "NMF")
    nmf_scores["snr"] = cfg.snr
    nmf_scores["seed"] = cfg.seed
    results.append(nmf_scores)

    # SRF
    similarity_matrix = compute_similarity(data, data, cfg.similarity)
    w_srf = SRF(
        rank=cfg.rank,
        random_state=cfg.seed,
        max_outer=1000,
        max_inner=1,
        tol=0.0,
        verbose=False,
        init="random_sqrt",
    ).fit_transform(similarity_matrix)
    labels_srf = np.argmax(w_srf, axis=1)
    srf_scores = _score_labels(true_labels, labels_srf, "SRF")
    srf_scores["snr"] = cfg.snr
    srf_scores["seed"] = cfg.seed
    results.append(srf_scores)

    df = pd.DataFrame(results)

    # Save to JSON in the current working directory (handled by Hydra)
    # The callback will merge these and clean them up.
    output_file = Path.cwd() / "results.json"
    df.to_json(output_file, orient="records")
