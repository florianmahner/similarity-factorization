"""
Node Classification Benchmark - Single Configuration Runner.

Each run processes ONE (dataset, method, size, rank) configuration.
Use Hydra multirun (-m) for parameter sweeps, SLURM for cluster execution.

Usage:
    # Debug single config
    poetry run python node_classification_benchmark.py \
        dataset=wikipedia method=deepwalk size=500 rank=64

    # Local sweep
    poetry run python node_classification_benchmark.py -m \
        dataset=wikipedia method=deepwalk,srf size=500 rank=32,64

    # SLURM sweep
    poetry run python node_classification_benchmark.py -m \
        dataset=wikipedia,blogcatalog,ppi \
        method=srf,deepwalk,node2vec,line,spectral \
        size=500,1000,2000 rank=32,64,128 \
        hydra/launcher=slurm
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import hydra
import mygene
import networkx as nx
import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ppi.lib.embeddings import (
    embeddings_to_matrix,
    fit_openne,
    fit_spectral,
    fit_srf_aa,
)

log = logging.getLogger(__name__)

OPENNE_DATA = PROJECT_ROOT / "third_party/OpenNE/data"
PPI_DATA = PROJECT_ROOT / "data/ppi"


def load_wikipedia(data_dir: Path) -> tuple[nx.Graph, dict, bool]:
    """Load Wikipedia POS tag classification dataset."""
    g = nx.Graph()
    with open(data_dir / "wiki/Wiki_edgelist.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                g.add_edge(parts[0], parts[1])

    labels = {}
    with open(data_dir / "wiki/wiki_labels.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels[parts[0]] = [int(parts[1])]

    return g, labels, False


def load_blogcatalog(data_dir: Path) -> tuple[nx.Graph, dict, bool]:
    """Load BlogCatalog user interest classification dataset."""
    g = nx.Graph()
    with open(data_dir / "blogCatalog/bc_edgelist.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                g.add_edge(parts[0], parts[1])

    labels = {}
    with open(data_dir / "blogCatalog/bc_labels.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels[parts[0]] = [int(x) for x in parts[1:]]

    return g, labels, True


def load_ppi(data_dir: Path) -> tuple[nx.Graph, dict, bool]:
    """Load STRING PPI with GO annotations."""
    import pandas as pd

    df = pd.read_csv(data_dir / "STRING_human_min900_v12.csv")
    g = nx.Graph()
    for _, row in df.iterrows():
        g.add_edge(row["source"], row["target"])

    largest_cc = max(nx.connected_components(g), key=len)
    g = g.subgraph(largest_cc).copy()

    nodes = sorted(g.nodes())
    proteins = [p.split(".")[-1] if "." in p else p for p in nodes]

    cache = data_dir / "go_annotations_human.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            annotations = pickle.load(f)
    else:
        mg = mygene.MyGeneInfo()
        res = mg.querymany(proteins, scopes="symbol,ensembl.protein",
                          fields="go", species="human", verbose=False)
        annotations = {}
        for entry in res:
            if "go" not in entry:
                continue
            terms = []
            for ont in ["MF", "BP"]:
                if ont in entry["go"]:
                    t = entry["go"][ont]
                    if isinstance(t, list):
                        terms.extend([x["id"] for x in t if "id" in x])
                    elif isinstance(t, dict) and "id" in t:
                        terms.append(t["id"])
            if terms:
                annotations[entry["query"]] = list(set(terms))
        with open(cache, "wb") as f:
            pickle.dump(annotations, f)

    labels = {}
    for node, prot in zip(nodes, proteins):
        if prot in annotations:
            labels[node] = annotations[prot]

    return g, labels, True


def get_subgraph(g: nx.Graph, size: int) -> nx.Graph:
    """Extract top-n nodes by degree."""
    if size >= len(g):
        return g
    top = sorted(g.degree, key=lambda x: x[1], reverse=True)[:size]
    sub = g.subgraph([n for n, _ in top]).copy()
    cc = max(nx.connected_components(sub), key=len)
    return sub.subgraph(cc).copy()


def filter_go_terms(labels: dict, nodes: list, min_c: int, max_c: int) -> dict:
    """Filter GO terms by frequency."""
    counts = {}
    for n in nodes:
        if n in labels:
            for t in labels[n]:
                counts[t] = counts.get(t, 0) + 1
    valid = {t for t, c in counts.items() if min_c <= c <= max_c}
    return {n: [t for t in labels[n] if t in valid]
            for n in nodes if n in labels and any(t in valid for t in labels[n])}


def evaluate(W: np.ndarray, nodes: list, labels: dict,
             train_ratio: float, seed: int, multi_label: bool) -> dict:
    """Evaluate node classification."""
    idx = {n: i for i, n in enumerate(nodes)}
    labeled = [n for n in nodes if n in labels]

    X = np.array([W[idx[n]] for n in labeled])
    y = [labels[n] for n in labeled]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=train_ratio, random_state=seed)

    if multi_label:
        mlb = MultiLabelBinarizer()
        y_tr_bin = mlb.fit_transform(y_tr)
        y_te_bin = mlb.transform(y_te)
    else:
        lb = LabelBinarizer()
        y_tr_bin = lb.fit_transform([l[0] for l in y_tr])
        y_te_bin = lb.transform([l[0] for l in y_te])

    clf = OneVsRestClassifier(LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed))
    clf.fit(X_tr, y_tr_bin)
    y_pred = clf.predict(X_te)

    return {
        "accuracy": float(accuracy_score(y_te_bin, y_pred)),
        "micro_f1": float(f1_score(y_te_bin, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_te_bin, y_pred, average="macro", zero_division=0)),
    }


def run(cfg: DictConfig) -> None:
    """Run single (dataset, method, size, rank) configuration."""
    dataset = cfg.dataset
    method = cfg.method
    size = cfg.get("size")
    rank = cfg.rank
    seed = cfg.seed
    output_dir = Path.cwd()

    log.info(f"Config: dataset={dataset}, method={method}, size={size}, rank={rank}")

    # Load dataset
    if dataset == "wikipedia":
        g_full, labels, multi_label = load_wikipedia(OPENNE_DATA)
        train_ratios = [0.1, 0.5, 0.9]
        eval_keys = ["all"]
    elif dataset == "blogcatalog":
        g_full, labels, multi_label = load_blogcatalog(OPENNE_DATA)
        train_ratios = [0.1, 0.5, 0.9]
        eval_keys = ["all"]
    elif dataset == "ppi":
        g_full, labels, multi_label = load_ppi(PPI_DATA)
        train_ratios = [0.8]
        eval_keys = ["rare", "medium", "frequent"]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    g = get_subgraph(g_full, size) if size else g_full
    nodes = sorted(g.nodes())
    labels = {n: labels[n] for n in nodes if n in labels}

    log.info(f"Graph: {len(nodes)} nodes, {g.number_of_edges()} edges, {len(labels)} labeled")

    # Generate embeddings
    start = time.time()
    line_params = dict(cfg.line) if "line" in cfg else {}
    srf_params = dict(cfg.srf) if "srf" in cfg else {}
    walks_params = dict(cfg.walks) if "walks" in cfg else {}

    if method == "srf":
        emb = fit_srf_aa(nodes, list(g.edges()), rank, seed, **srf_params)
    elif method == "spectral":
        emb = fit_spectral(nodes, list(g.edges()), rank, seed)
    elif method in ["deepwalk", "node2vec", "line"]:
        emb = fit_openne(method, g, rank, seed, line_params, walks_params)
    else:
        raise ValueError(f"Unknown method: {method}")

    runtime = time.time() - start

    if not emb:
        log.error(f"No embeddings returned for {method}")
        return

    W = embeddings_to_matrix(emb, nodes)
    log.info(f"Embeddings: shape={W.shape}, runtime={runtime:.1f}s")

    # Evaluate
    results = []
    go_bins = {"rare": (11, 30), "medium": (31, 100), "frequent": (101, 300)}

    for eval_key in eval_keys:
        if dataset == "ppi":
            min_c, max_c = go_bins[eval_key]
            eval_labels = filter_go_terms(labels, nodes, min_c, max_c)
            if not eval_labels:
                log.warning(f"No labels for {eval_key}")
                continue
        else:
            eval_labels = labels

        for tr in train_ratios:
            metrics = evaluate(W, nodes, eval_labels, tr, seed, multi_label)
            results.append({
                "dataset": dataset,
                "method": method,
                "size": len(nodes),
                "rank": rank,
                "seed": seed,
                "eval_key": eval_key,
                "train_ratio": tr,
                "n_nodes": len(nodes),
                "n_labeled": len(eval_labels),
                "runtime": runtime,
                **metrics,
            })

    # Save single JSON per job
    out_file = output_dir / "result.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved {out_file}")

    for r in results:
        log.info(f"{r['eval_key']} tr={r['train_ratio']}: micro_f1={r['micro_f1']:.3f}")


@hydra.main(config_path=".", config_name="benchmark_config", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
