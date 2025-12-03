from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def load_chembl_data(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drug_targets = pd.read_csv(base_dir / "drug_targets.csv")
    atc_codes = pd.read_csv(base_dir / "atc_codes.csv")
    drug_info = pd.read_csv(base_dir / "drug_info.csv")
    return drug_targets, atc_codes, drug_info


def load_sider_side_effects(sider_path: Path) -> pd.DataFrame:
    se_file = sider_path / "meddra_all_se.tsv"
    if not se_file.exists():
        return pd.DataFrame(columns=["drug_id", "side_effect"])
    side_effects = pd.read_csv(
        se_file,
        sep="\t",
        names=["drug_id", "se_concept_id", "side_effect"],
        dtype=str,
    )
    return side_effects


def build_drug_target_dict(drug_targets: pd.DataFrame) -> dict[str, set[str]]:
    drug_targets_unique = drug_targets[["drug_id", "target_id"]].drop_duplicates()
    dt_dict: dict[str, set[str]] = {}
    for _, row in drug_targets_unique.iterrows():
        dt_dict.setdefault(row["drug_id"], set()).add(row["target_id"])
    return dt_dict


def build_graph(drug_info: pd.DataFrame, drug_targets: pd.DataFrame, min_shared: int) -> nx.Graph:
    dt_dict = build_drug_target_dict(drug_targets)
    g = nx.Graph()
    drugs = drug_info["drug_id"].unique()
    g.add_nodes_from(drugs)

    for i, drug1 in enumerate(drugs):
        if drug1 not in dt_dict:
            continue
        targets1 = dt_dict[drug1]
        for drug2 in drugs[i + 1 :]:
            if drug2 not in dt_dict:
                continue
            shared = len(targets1 & dt_dict[drug2])
            if shared >= min_shared:
                g.add_edge(drug1, drug2, weight=shared)
    return g


def adjacency_with_nan(g: nx.Graph, drug_info: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    vocabulary = sorted(g.nodes())
    n = len(vocabulary)
    idx = {drug: i for i, drug in enumerate(vocabulary)}
    adjacency = np.full((n, n), np.nan, dtype=np.float32)
    for d1, d2, data in g.edges(data=True):
        i, j = idx[d1], idx[d2]
        adjacency[i, j] = adjacency[j, i] = data["weight"]
    max_val = np.nanmax(adjacency)
    if max_val > 0:
        adjacency = adjacency / max_val
    ordered_info = drug_info.set_index("drug_id").loc[vocabulary].reset_index()
    return adjacency, ordered_info


def save_processed_data(
    adjacency: np.ndarray,
    drug_info: pd.DataFrame,
    atc_codes: pd.DataFrame,
    drug_targets: pd.DataFrame,
    side_effects: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "drug_drug_adjacency.npy", adjacency)
    drug_info.to_csv(output_dir / "drug_info.csv", index=False)
    atc_codes.to_csv(output_dir / "atc_codes.csv", index=False)
    drug_targets.to_csv(output_dir / "drug_targets.csv", index=False)
    if len(side_effects) > 0:
        side_effects.to_csv(output_dir / "side_effects.csv", index=False)

    n_obs = int(np.sum(~np.isnan(adjacency)) / 2)
    metadata = {
        "n_drugs": int(adjacency.shape[0]),
        "n_observed_edges": n_obs,
        "density": float(
            n_obs / (adjacency.shape[0] * (adjacency.shape[0] - 1) / 2)
        ),
        "pct_missing": float(np.sum(np.isnan(adjacency)) / adjacency.size * 100),
    }
    (output_dir / "graph_metadata.json").write_text(json.dumps(metadata, indent=2))

