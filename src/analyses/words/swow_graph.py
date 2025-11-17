"""Reusable functions for building and processing SWOW word association graphs."""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

from pysrf import SRF


def load_swow_data(data_dir: Path, use_all_responses: bool) -> pd.DataFrame:
    """Load SWOW data with raw counts."""
    suffix = "R123" if use_all_responses else "R1"
    df = pd.read_csv(data_dir / f"strength.SWOW-EN.{suffix}.20180827.csv", sep="\t")
    return df[["cue", "response", suffix]].rename(columns={suffix: "count"})


def filter_by_frequency(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Filter to top N words by total frequency."""
    word_counts = pd.concat([df["cue"], df["response"]]).value_counts()
    top_words = set(word_counts.head(top_n).index)
    return df[df["cue"].isin(top_words) & df["response"].isin(top_words)]


def filter_by_degree(g: nx.DiGraph, top_n: int) -> nx.DiGraph:
    """Filter to top N words by degree."""
    degrees = dict(g.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_n]
    return g.subgraph(top_nodes).copy()


def build_count_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build directed graph from SWOW data with raw counts."""
    df = df[df["response"].notna()]
    g = nx.DiGraph()
    for _, row in df.iterrows():
        g.add_edge(row["cue"], row["response"], count=row["count"])
    g.remove_nodes_from([n for n in g.nodes() if g.out_degree(n) == 0])
    g.remove_edges_from(nx.selfloop_edges(g))
    largest_scc = max(nx.strongly_connected_components(g), key=len)
    return g.subgraph(largest_scc).copy()


def symmetrize_counts_then_ppmi(
    g: nx.DiGraph, method: str = "geometric_mean", handle_unidirectional: str = "copy"
) -> tuple[np.ndarray, list[str], dict]:
    """Symmetrize raw counts first, then calculate PPMI (correct approach)."""
    vocabulary = sorted(g.nodes())
    n = len(vocabulary)
    idx = {word: i for i, word in enumerate(vocabulary)}

    counts = np.zeros((n, n), dtype=np.float32)
    for u, v, data in g.edges(data=True):
        counts[idx[u], idx[v]] = data["count"]

    counts_sym = np.full((n, n), np.nan, dtype=np.float32)
    both, one, neither = 0, 0, 0

    for i in range(n):
        for j in range(i + 1, n):
            c_ij, c_ji = counts[i, j], counts[j, i]
            has_ij, has_ji = c_ij > 0, c_ji > 0

            if has_ij and has_ji:
                c = np.sqrt(c_ij * c_ji) if method == "geometric_mean" else (c_ij + c_ji) / 2
                counts_sym[i, j] = counts_sym[j, i] = c
                both += 1
            elif has_ij or has_ji:
                if handle_unidirectional == "discard":
                    neither += 1
                else:
                    c = (c_ij if has_ij else c_ji) * (0.5 if handle_unidirectional == "weight_half" else 1.0)
                    counts_sym[i, j] = counts_sym[j, i] = c
                    one += 1
            else:
                neither += 1

    observed = ~np.isnan(counts_sym)
    counts_clean = np.where(observed, counts_sym, 0)
    total = counts_clean.sum()

    ppmi = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if observed[i, j]:
                p_ij = counts_clean[i, j] / total
                p_i = counts_clean[i, :].sum() / total
                p_j = counts_clean[j, :].sum() / total
                if p_i > 0 and p_j > 0:
                    val = max(0, np.log2(p_ij / (p_i * p_j + 1e-10) + 1e-10))
                    ppmi[i, j] = ppmi[j, i] = val

    np.fill_diagonal(ppmi, np.nan)

    meta = {
        "n_words": n,
        "both_observed": both,
        "one_observed": one,
        "neither_observed": neither,
        "pct_missing": np.sum(np.isnan(ppmi)) / (n * n) * 100,
        "symmetrization_method": method,
        "ppmi_timing": "after_symmetrization",
    }
    return ppmi, vocabulary, meta


def calculate_ppmi_then_symmetrize(
    g: nx.DiGraph, df: pd.DataFrame, method: str = "geometric_mean"
) -> tuple[np.ndarray, list[str], dict]:
    """Calculate PPMI on directed graph, then symmetrize (old approach for comparison)."""
    vocabulary = sorted(g.nodes())
    n = len(vocabulary)
    idx = {word: i for i, word in enumerate(vocabulary)}

    df_filtered = df[df["cue"].isin(vocabulary) & df["response"].isin(vocabulary)].copy()
    total = df_filtered["count"].sum()
    df_filtered["p_cr"] = df_filtered["count"] / total
    df_filtered["p_cue"] = df_filtered.groupby("cue")["count"].transform("sum") / total
    df_filtered["p_resp"] = df_filtered.groupby("response")["count"].transform("sum") / total

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(df_filtered["p_cr"] / (df_filtered["p_cue"] * df_filtered["p_resp"]))
    df_filtered["ppmi"] = np.maximum(0, pmi).replace([np.inf, -np.inf], 0).fillna(0)

    ppmi_dir = np.full((n, n), np.nan, dtype=np.float32)
    for _, row in df_filtered.iterrows():
        ppmi_dir[idx[row["cue"]], idx[row["response"]]] = row["ppmi"]

    ppmi_sym = np.full((n, n), np.nan, dtype=np.float32)
    both, one = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            has_ij, has_ji = ~np.isnan(ppmi_dir[i, j]), ~np.isnan(ppmi_dir[j, i])
            if has_ij and has_ji:
                val = np.sqrt(ppmi_dir[i, j] * ppmi_dir[j, i]) if method == "geometric_mean" else (ppmi_dir[i, j] + ppmi_dir[j, i]) / 2
                ppmi_sym[i, j] = ppmi_sym[j, i] = val
                both += 1
            elif has_ij or has_ji:
                ppmi_sym[i, j] = ppmi_sym[j, i] = ppmi_dir[i, j] if has_ij else ppmi_dir[j, i]
                one += 1

    np.fill_diagonal(ppmi_sym, np.nan)

    meta = {
        "n_words": n,
        "both_observed": both,
        "one_observed": one,
        "neither_observed": 0,
        "pct_missing": np.sum(np.isnan(ppmi_sym)) / (n * n) * 100,
        "symmetrization_method": method,
        "ppmi_timing": "before_symmetrization",
    }
    return ppmi_sym, vocabulary, meta


def fit_srf(network: np.ndarray, rank: int, max_iter: int = 500, verbose: int = 0) -> np.ndarray:
    """Fit SRF model to network."""
    srf = SRF(
        rank=rank,
        rho=3.0,
        max_outer=max_iter,
        max_inner=30,
        random_state=42,
        init="random_sqrt",
        tol=1e-4,
        verbose=verbose,
    )
    srf.fit(network)
    return srf.w_
