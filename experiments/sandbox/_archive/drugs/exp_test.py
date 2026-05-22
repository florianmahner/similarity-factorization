from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cli import ExperimentContext, register_experiment

from lib.graph import (
    adjacency_with_nan,
    build_graph,
    load_chembl_data,
    load_sider_side_effects,
    save_processed_data,
)
from lib.enrichment import (
    compute_atc_enrichment,
    compute_side_effect_enrichment,
    compute_target_enrichment,
    run_srf,
    save_enrichment_results,
)


@dataclass(slots=True)
class Params:
    task: str = "build_graph"
    base_dir: Path = Path("experiments/drugs/outputs")
    sider_dir: Path = Path("data/drugs/sider")
    processed_dir: Path = Path("experiments/drugs/outputs/processed")
    enrichment_dir: Path = Path("experiments/drugs/outputs/final")
    rank: int = 20
    rho: float = 3.0


def add_arguments(parser) -> None:
    parser.add_argument(
        "--task", choices=["build_graph", "enrichment"], default="build_graph"
    )
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--rho", type=float, default=3.0)


def _build_graph_task(context: ExperimentContext, params: Params) -> None:
    base = params.base_dir
    if not base.exists():
        raise FileNotFoundError(f"Base directory not found: {base}")
    drug_targets, atc_codes, drug_info = load_chembl_data(base)
    side_effects = load_sider_side_effects(params.sider_dir)
    g = build_graph(drug_info, drug_targets, min_shared=1)
    adjacency, ordered_info = adjacency_with_nan(g, drug_info)
    save_processed_data(
        adjacency,
        ordered_info,
        atc_codes,
        drug_targets,
        side_effects,
        params.processed_dir,
    )
    context.logger.info("Processed graph saved to %s", params.processed_dir)


def _enrichment_task(context: ExperimentContext, params: Params) -> None:
    proc = params.processed_dir
    adjacency = np.load(proc / "drug_drug_adjacency.npy")
    drug_info = pd.read_csv(proc / "drug_info.csv")
    atc_codes = pd.read_csv(proc / "atc_codes.csv")
    drug_targets = pd.read_csv(proc / "drug_targets.csv")
    side_effects_path = proc / "side_effects.csv"
    side_effects = (
        pd.read_csv(side_effects_path)
        if side_effects_path.exists()
        else pd.DataFrame(columns=["drug_id", "side_effect"])
    )

    w = run_srf(adjacency, rank=params.rank, rho=params.rho)
    atc_enrich = compute_atc_enrichment(w, drug_info, atc_codes)
    target_enrich = compute_target_enrichment(w, drug_info, drug_targets)
    se_enrich = compute_side_effect_enrichment(w, drug_info, side_effects)
    save_enrichment_results(
        w, drug_info, atc_enrich, target_enrich, se_enrich, params.enrichment_dir
    )
    context.logger.info("Enrichment results stored in %s", params.enrichment_dir)


@register_experiment(
    name="drugs",
    params_type=Params,
    add_arguments=add_arguments,
    description="Drug graph processing and enrichment analyses",
)
def run(context: ExperimentContext, params: Params) -> None:
    if params.task == "build_graph":
        _build_graph_task(context, params)
    else:
        _enrichment_task(context, params)
