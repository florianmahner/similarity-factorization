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
    suffix = "R123" if use_all_responses else "R1"

    df = pd.read_csv(data_dir / f"strength.SWOW-EN.{suffix}.20180827.csv", sep="\t")
    df["strength"] = df[f"{suffix}.Strength"]
    df["count"] = df[suffix]

    return df


def calculate_ppmi(df):
    df = df.copy()
    total = df["count"].sum()
    if total <= 0:
        raise ValueError("total count must be > 0")

    df["p_cr"] = df["count"] / total
    df["p_cue"] = df.groupby("cue")["count"].transform("sum") / total
    df["p_resp"] = df.groupby("response")["count"].transform("sum") / total

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(df["p_cr"] / (df["p_cue"] * df["p_resp"]))

    df["ppmi"] = np.maximum(0, pmi).replace([np.inf, -np.inf], 0).fillna(0)

    return df[["cue", "response", "count", "ppmi"]]


def build_directed_graph(
    df_long: pd.DataFrame, weight_col: str = "strength"
) -> tuple[nx.DiGraph, set]:
    df_filtered = df_long[df_long["response"].notna()]

    g = nx.DiGraph()
    g.add_weighted_edges_from(
        df_filtered[["cue", "response", weight_col]].itertuples(index=False, name=None)
    )
    g.remove_nodes_from([n for n in g.nodes() if g.out_degree(n) == 0])
    g.remove_edges_from(nx.selfloop_edges(g))

    largest_scc = max(nx.strongly_connected_components(g), key=len)
    return g.subgraph(largest_scc).copy(), set(g.nodes()) - largest_scc


def symmetrize_with_nan_preservation(
    g_directed: nx.DiGraph, method: str = "geometric_mean"
) -> tuple[np.ndarray, list[str], dict[str, int | float]]:
    vocabulary = sorted(g_directed.nodes())
    n = len(vocabulary)
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

    a_directed = np.full((n, n), np.nan, dtype=np.float32)
    for source, target, data in g_directed.edges(data=True):
        a_directed[word_to_idx[source], word_to_idx[target]] = data["weight"]

    a_symmetric = np.full((n, n), np.nan, dtype=np.float32)
    both_observed = one_observed = neither_observed = 0
    asymmetry_values = []

    for i in range(n):
        for j in range(i, n):
            if i == j:
                continue

            has_ij, has_ji = ~np.isnan(a_directed[i, j]), ~np.isnan(a_directed[j, i])

            if has_ij and has_ji:
                if method == "geometric_mean":
                    val = np.sqrt(a_directed[i, j] * a_directed[j, i])
                elif method == "arithmetic_mean":
                    val = (a_directed[i, j] + a_directed[j, i]) / 2
                elif method == "max":
                    val = max(a_directed[i, j], a_directed[j, i])
                else:
                    raise ValueError(f"Unknown symmetrization method: {method}")
                a_symmetric[i, j] = a_symmetric[j, i] = val
                both_observed += 1
                asymm = abs(a_directed[i, j] - a_directed[j, i]) / (
                    a_directed[i, j] + a_directed[j, i] + 1e-10
                )
                asymmetry_values.append(asymm)
            elif has_ij or has_ji:
                val = a_directed[i, j] if has_ij else a_directed[j, i]
                a_symmetric[i, j] = a_symmetric[j, i] = val
                one_observed += 1
            else:
                neither_observed += 1

    metadata = {
        "n_words": n,
        "both_observed": both_observed,
        "one_observed": one_observed,
        "neither_observed": neither_observed,
        "pct_missing": np.sum(np.isnan(a_symmetric)) / n**2 * 100,
        "mean_asymmetry": np.mean(asymmetry_values) if asymmetry_values else 0,
        "symmetrization_method": method,
    }

    return a_symmetric, vocabulary, metadata


def fit_srf(network: np.ndarray, rank: int = 20) -> np.ndarray:
    srf = SRF(
        rank=rank,
        rho=3.0,
        max_outer=4000,
        max_inner=30,
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
    network_clean = np.nan_to_num(network, nan=0)
    coherence = np.zeros(w.shape[1])

    for dim in range(w.shape[1]):
        top_idx = np.argsort(w[:, dim])[-top_n:]
        sims = [
            network_clean[i, j]
            for i in top_idx
            for j in top_idx
            if i < j and network_clean[i, j] > 0
        ]
        coherence[dim] = np.mean(sims) if sims else 0

    return coherence


def compute_sparsity(w: np.ndarray) -> pd.DataFrame:
    n, n_dims = w.shape
    data = []

    for dim in range(n_dims):
        loadings = w[:, dim]
        gini = (2 * np.sum(np.arange(1, n + 1) * np.sort(loadings))) / (
            n * loadings.sum()
        ) - (n + 1) / n
        pct_active = 100 * (loadings > 0.1 * loadings.max()).sum() / n
        probs = loadings / (loadings.sum() + 1e-10)
        entropy = -np.sum(probs * np.log(probs + 1e-10)) / np.log(n)
        data.append((dim, gini, entropy, pct_active))

    return pd.DataFrame(data, columns=["dimension", "gini", "entropy", "pct_active"])


def get_top_words_per_dimension(
    w: np.ndarray, vocabulary: list[str], top_n: int = 20
) -> pd.DataFrame:
    data = [
        (dim, rank, vocabulary[idx], w[idx, dim])
        for dim in range(w.shape[1])
        for rank, idx in enumerate(np.argsort(w[:, dim])[-top_n:][::-1])
    ]
    return pd.DataFrame(data, columns=["dimension", "rank", "word", "loading"])


def main(
    output_dir: Path | str | None = None,
    rank: int = 20,
    use_all_responses: bool = True,
    top_n_words: int | None = None,
    symmetrization_method: str = "geometric_mean",
) -> None:
    data_dir = Path("data/small-world-of-words")

    if output_dir is None:
        timestamp = datetime.now().strftime("%y%m%d/%H%M%S")
        output_dir = Path("experiments/word_association/outputs") / timestamp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_long = load_swow_data_paper_format(data_dir, use_all_responses)
    df_long = calculate_ppmi(df_long)

    g_directed, _ = build_directed_graph(df_long, weight_col="ppmi")

    if top_n_words:
        g_directed = filter_by_node_degree_postgraph(g_directed, top_n_words)

    similarity, vocabulary, metadata = symmetrize_with_nan_preservation(
        g_directed, method=symmetrization_method
    )
    observed_mask = ~np.isnan(similarity)

    w_srf = fit_srf(similarity, rank)
    coherence = compute_dimension_coherence(w_srf, similarity, top_n=50)
    sparsity = compute_sparsity(w_srf)
    top_words = get_top_words_per_dimension(w_srf, vocabulary, top_n=20)

    np.save(output_dir / "srf_factors.npy", w_srf)
    np.save(output_dir / "similarity_matrix.npy", similarity)
    Path(output_dir / "vocabulary.txt").write_text("\n".join(vocabulary))

    dimension_metrics = sparsity.copy()
    dimension_metrics["coherence"] = coherence
    dimension_metrics.to_csv(output_dir / "dimension_metrics.csv", index=False)

    top_words.to_csv(output_dir / "top_words.csv", index=False)

    reconstruction = w_srf @ w_srf.T
    sim_actual, sim_pred = similarity[observed_mask], reconstruction[observed_mask]
    corr = np.corrcoef(sim_actual, sim_pred)[0, 1]
    mse = np.mean((sim_actual - sim_pred) ** 2)

    summary = pd.DataFrame(
        [
            {
                **metadata,
                "rank": rank,
                "mean_gini": sparsity["gini"].mean(),
                "mean_entropy": sparsity["entropy"].mean(),
                "mean_pct_active": sparsity["pct_active"].mean(),
                "mean_coherence": coherence.mean(),
                "reconstruction_corr": corr,
                "reconstruction_mse": mse,
            }
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    print(
        f"{len(vocabulary)} words | {rank} dims | "
        f"missing={metadata['pct_missing']:.1f}% | "
        f"gini={sparsity['gini'].mean():.3f} | "
        f"coherence={coherence.mean():.4f} | "
        f"r={corr:.3f} → {output_dir}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute SRF embeddings from SWOW word association data using PPMI."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--rank", type=int, default=20, help="SRF rank (number of dimensions)"
    )
    parser.add_argument(
        "--use-all-responses",
        action="store_true",
        default=True,
        help="Use R123 (all responses)",
    )
    parser.add_argument(
        "--r1-only", action="store_true", help="Use R1 (first response only)"
    )
    parser.add_argument(
        "--top-n-words",
        type=int,
        default=None,
        help="Limit vocabulary to top N words by degree",
    )
    parser.add_argument(
        "--symmetrization-method",
        type=str,
        default="geometric_mean",
        choices=["geometric_mean", "arithmetic_mean", "max"],
        help="Method for symmetrizing directed associations (default: geometric_mean)",
    )

    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        rank=args.rank,
        use_all_responses=not args.r1_only,
        top_n_words=args.top_n_words,
        symmetrization_method=args.symmetrization_method,
    )
