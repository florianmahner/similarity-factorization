import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import networkx as nx
import warnings

warnings.filterwarnings("ignore")

from pysrf import SRF


def filter_by_node_degree_postgraph(
    g_directed: nx.DiGraph, top_n_words: int = 3000
) -> nx.DiGraph:
    node_degrees = dict(g_directed.degree())
    top_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)[:top_n_words]
    return g_directed.subgraph(top_nodes).copy()


def load_swow_data_paper_format(
    data_dir: Path | str, use_all_responses: bool = True
) -> pd.DataFrame:
    data_dir = Path(data_dir)

    if use_all_responses:
        strength_file = data_dir / "strength.SWOW-EN.R123.20180827.csv"
    else:
        strength_file = data_dir / "strength.SWOW-EN.R1.20180827.csv"

    df = pd.read_csv(strength_file, sep="\t")

    if "R123.Strength" in df.columns:
        df["strength"] = df["R123.Strength"]
        df["count"] = df["R123"]
    elif "R1.Strength" in df.columns:
        df["strength"] = df["R1.Strength"]
        df["count"] = df["R1"]
    else:
        raise ValueError("Cannot find strength column in SWOW data")

    return df


def calculate_ppmi(df_long: pd.DataFrame) -> pd.DataFrame:
    n_cues = df_long["cue"].nunique()

    p_r_sum = df_long.groupby("response")["strength"].sum().reset_index()
    p_r_sum = p_r_sum.rename(columns={"strength": "p_r_sum"})

    df_with_pr = pd.merge(df_long, p_r_sum, on="response", how="left")
    df_with_pr["p_r_sum"] = df_with_pr["p_r_sum"].fillna(0)

    numerator = df_with_pr["strength"] * n_cues
    denominator = df_with_pr["p_r_sum"]

    df_with_pr["ppmi_strength"] = 0.0

    positive_denom_mask = denominator > 1e-10
    ratio = np.full(df_with_pr.shape[0], 0.0)
    ratio[positive_denom_mask] = (
        numerator[positive_denom_mask] / denominator[positive_denom_mask]
    )

    positive_ratio_mask = ratio > 1e-10
    log_ratio = np.full(df_with_pr.shape[0], -np.inf)
    log_ratio[positive_ratio_mask] = np.log2(ratio[positive_ratio_mask])

    df_with_pr["ppmi_strength"] = np.maximum(0, log_ratio)

    return df_with_pr


def build_directed_graph(
    df_long: pd.DataFrame, weight_col: str = "strength"
) -> tuple[nx.DiGraph, set]:
    df_filtered = df_long[df_long["response"].notna()].copy()

    g = nx.DiGraph()
    g.add_weighted_edges_from(
        df_filtered[["cue", "response", weight_col]].itertuples(index=False, name=None)
    )

    nodes_to_remove = [node for node in g.nodes() if g.out_degree(node) == 0]
    g.remove_nodes_from(nodes_to_remove)

    self_loops = list(nx.selfloop_edges(g))
    g.remove_edges_from(self_loops)

    strongly_connected = list(nx.strongly_connected_components(g))
    largest_scc = max(strongly_connected, key=len)

    g_directed = g.subgraph(largest_scc).copy()
    removed_nodes = set(g.nodes()) - largest_scc

    return g_directed, removed_nodes


def symmetrize_with_nan_preservation(
    g_directed: nx.DiGraph,
) -> tuple[np.ndarray, list[str], dict[str, int | float]]:
    vocabulary = sorted(g_directed.nodes())
    n = len(vocabulary)
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

    a_directed = np.full((n, n), np.nan, dtype=np.float32)

    for source, target, data in g_directed.edges(data=True):
        i = word_to_idx[source]
        j = word_to_idx[target]
        a_directed[i, j] = data["weight"]

    a_symmetric = np.full((n, n), np.nan, dtype=np.float32)

    both_observed = 0
    one_observed = 0
    neither_observed = 0

    for i in range(n):
        for j in range(i, n):
            if i == j:
                continue

            has_ij = ~np.isnan(a_directed[i, j])
            has_ji = ~np.isnan(a_directed[j, i])

            if has_ij and has_ji:
                a_symmetric[i, j] = (a_directed[i, j] + a_directed[j, i]) / 2
                a_symmetric[j, i] = a_symmetric[i, j]
                both_observed += 1
            elif has_ij:
                a_symmetric[i, j] = a_directed[i, j]
                a_symmetric[j, i] = a_directed[i, j]
                one_observed += 1
            elif has_ji:
                a_symmetric[i, j] = a_directed[j, i]
                a_symmetric[j, i] = a_directed[j, i]
                one_observed += 1
            else:
                neither_observed += 1

    n_nan = np.sum(np.isnan(a_symmetric))

    asymmetry_values = []
    for i in range(n):
        for j in range(i + 1, n):
            if ~np.isnan(a_directed[i, j]) and ~np.isnan(a_directed[j, i]):
                asymm = abs(a_directed[i, j] - a_directed[j, i]) / (
                    a_directed[i, j] + a_directed[j, i] + 1e-10
                )
                asymmetry_values.append(asymm)

    metadata = {
        "n_words": n,
        "both_observed": both_observed,
        "one_observed": one_observed,
        "neither_observed": neither_observed,
        "pct_missing": n_nan / n**2 * 100,
        "mean_asymmetry": np.mean(asymmetry_values) if asymmetry_values else 0,
    }

    return a_symmetric, vocabulary, metadata


def fit_srf(network: np.ndarray, rank: int = 20) -> np.ndarray:
    srf = SRF(
        rank=rank,
        rho=3.0,
        max_outer=600,
        max_inner=50,
        random_state=42,
        init="random_sqrt",
        tol=1e-4,
        verbose=1,
    )

    srf.fit(network)

    return srf.w_


def compute_dimension_coherence(
    w: np.ndarray, network: np.ndarray, top_n: int = 50
) -> np.ndarray:
    n_dims = w.shape[1]
    coherence_scores = np.zeros(n_dims)

    network_clean = np.nan_to_num(network, nan=0)

    for dim in range(n_dims):
        loadings = w[:, dim]
        top_idx = np.argsort(loadings)[-top_n:]

        similarities = []
        for i in range(len(top_idx)):
            for j in range(i + 1, len(top_idx)):
                sim = network_clean[top_idx[i], top_idx[j]]
                if sim > 0:
                    similarities.append(sim)

        if similarities:
            coherence_scores[dim] = np.mean(similarities)
        else:
            coherence_scores[dim] = 0

    return coherence_scores


def compute_sparsity(w: np.ndarray) -> pd.DataFrame:
    n_dims = w.shape[1]
    metrics = {
        "dimension": [],
        "gini": [],
        "entropy": [],
        "pct_active": [],
    }

    for dim in range(n_dims):
        loadings = w[:, dim]

        sorted_loadings = np.sort(loadings)
        n = len(loadings)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_loadings)) / (n * np.sum(sorted_loadings)) - (
            n + 1
        ) / n

        threshold = 0.1 * loadings.max()
        pct_active = 100 * np.sum(loadings > threshold) / n

        probs = loadings / (loadings.sum() + 1e-10)
        entropy = -np.sum(probs * np.log(probs + 1e-10)) / np.log(n)

        metrics["dimension"].append(dim)
        metrics["gini"].append(gini)
        metrics["entropy"].append(entropy)
        metrics["pct_active"].append(pct_active)

    return pd.DataFrame(metrics)


def get_top_words_per_dimension(
    w: np.ndarray, vocabulary: list[str], top_n: int = 20
) -> pd.DataFrame:
    n_dims = w.shape[1]
    top_words_data = []

    for dim in range(n_dims):
        loadings = w[:, dim]
        top_idx = np.argsort(loadings)[-top_n:][::-1]

        for rank, idx in enumerate(top_idx):
            top_words_data.append(
                {
                    "dimension": dim,
                    "rank": rank,
                    "word": vocabulary[idx],
                    "loading": loadings[idx],
                }
            )

    return pd.DataFrame(top_words_data)


def main(
    output_dir: Path | str | None = None,
    rank: int = 20,
    use_all_responses: bool = True,
    top_n_words: int | None = None,
) -> None:
    data_dir = Path("data/small-world-of-words")

    if output_dir is None:
        timestamp = datetime.now().strftime("%y%m%d/%H%M%S")
        output_dir = (
            Path("experiments/development/word_association/outputs") / timestamp
        )
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    df_long = load_swow_data_paper_format(data_dir, use_all_responses=use_all_responses)

    df_long = calculate_ppmi(df_long)

    g_directed, removed_nodes = build_directed_graph(
        df_long, weight_col="ppmi_strength"
    )

    if top_n_words is not None:
        g_directed = filter_by_node_degree_postgraph(g_directed, top_n_words)

    similarity, vocabulary, metadata = symmetrize_with_nan_preservation(g_directed)

    observed_mask = ~np.isnan(similarity)

    w_srf = fit_srf(similarity, rank=rank)

    coherence_scores = compute_dimension_coherence(w_srf, similarity, top_n=50)
    sparsity_metrics = compute_sparsity(w_srf)
    top_words_df = get_top_words_per_dimension(w_srf, vocabulary, top_n=20)

    np.save(output_dir / "srf_factors.npy", w_srf)
    np.save(output_dir / "similarity_matrix.npy", similarity)

    with open(output_dir / "vocabulary.txt", "w") as f:
        f.write("\n".join(str(word) for word in vocabulary if isinstance(word, str)))

    sparsity_metrics.to_csv(output_dir / "sparsity_metrics.csv", index=False)

    coherence_df = pd.DataFrame(
        {"dimension": range(len(coherence_scores)), "coherence": coherence_scores}
    )
    coherence_df.to_csv(output_dir / "coherence_scores.csv", index=False)

    top_words_df.to_csv(output_dir / "top_words_per_dimension.csv", index=False)

    metadata_df = pd.DataFrame([metadata])
    metadata_df.to_csv(output_dir / "graph_metadata.csv", index=False)

    reconstruction = w_srf @ w_srf.T

    sim_actual = similarity[observed_mask]
    sim_pred = reconstruction[observed_mask]
    corr = np.corrcoef(sim_actual, sim_pred)[0, 1]
    mse = np.mean((sim_actual - sim_pred) ** 2)

    metrics_summary = pd.DataFrame(
        [
            {
                "n_words": metadata["n_words"],
                "rank": rank,
                "pct_missing": metadata["pct_missing"],
                "mean_asymmetry": metadata["mean_asymmetry"],
                "mean_gini": sparsity_metrics["gini"].mean(),
                "mean_coherence": coherence_scores.mean(),
                "reconstruction_corr": corr,
                "reconstruction_mse": mse,
            }
        ]
    )
    metrics_summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    print(
        f"{len(vocabulary)} words | {rank} dims | "
        f"missing={metadata['pct_missing']:.1f}% | "
        f"gini={sparsity_metrics['gini'].mean():.3f} | "
        f"coherence={coherence_scores.mean():.4f} | "
        f"r={corr:.3f} → {output_dir}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--use-all-responses", action="store_true", default=True)
    parser.add_argument("--r1-only", action="store_true")
    parser.add_argument("--top-n-words", type=int, default=None)

    args = parser.parse_args()

    use_all_responses = not args.r1_only

    main(
        output_dir=args.output_dir,
        rank=args.rank,
        use_all_responses=use_all_responses,
        top_n_words=args.top_n_words,
    )
