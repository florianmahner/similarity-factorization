import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from tools.rsa import compute_rsm
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.cluster import SpectralClustering

from datasets import load_dataset
from pysrf import SRF
from utils.helpers import (
    map_labels_with_hungarian,
    purity_score,
    compute_entropy,
    compute_sparseness,
    compute_orthogonality,
)

from pathlib import Path


# Model and Dataset Registries
MODEL_REGISTRY = {}
DATASET_REGISTRY = {}


# Data Processing Functions
def flatten_dataset(x):
    return x.reshape(x.shape[0], -1)


def positive_shift_and_scale(x):
    """Normalize data to range [0,1]."""
    return (x - x.min(axis=0)) / (x.max(axis=0) + 1e-10)


def standardize_data(x):
    """Z-score standardization."""
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-10)


def pearson_rsm(x):
    """Compute RSM using Pearson correlation."""
    return compute_rsm(x, metric="pearson")


def cosine_rsm(x):
    """Compute RSM using cosine similarity."""
    return compute_rsm(x, metric="cosine")


def rbf_rsm(x):
    """Compute RSM using RBF kernel."""
    return compute_rsm(x, metric="gaussian_kernel")


def construct_similarity_graph(x, nn=7):
    """Construct similarity graph using self-tuning kernel normalization."""
    n = x.shape[0]
    k = int(np.floor(np.log2(n))) + 1
    dist = pairwise_distances(x, metric="euclidean", squared=True)

    nn = min(nn, n - 1)
    sorted_cols = np.sort(dist, axis=0)
    local_scales = np.sqrt(sorted_cols[nn, :])

    denom = np.outer(local_scales, local_scales)
    graph = np.exp(-dist / denom)
    np.fill_diagonal(graph, 0.0)

    row_sorted_idx = np.argsort(dist, axis=1)
    mask = np.zeros_like(graph, dtype=bool)
    for i in range(n):
        nbrs_i = row_sorted_idx[i, 1 : k + 1]
        mask[i, nbrs_i] = mask[nbrs_i, i] = True
    graph[~mask] = 0.0

    dd = np.sqrt(1.0 / (graph.sum(axis=0) + 1e-12))
    graph *= dd
    graph = graph.T
    graph *= dd
    return 0.5 * (graph + graph.T)


# Model and Dataset Registries
def register_model(name, model, preprocessors=None):
    MODEL_REGISTRY[name] = {"model": model, "preprocessors": preprocessors or {}}


def register_dataset(name, x, y):
    """Register datasets along with a default preprocessing function if needed."""
    DATASET_REGISTRY[name] = {"x": x, "y": y}


def load_all_datasets():
    # Small classic tabular datasets
    iris = load_dataset("iris")
    register_dataset("iris", iris.data, iris.targets)

    wine = load_dataset("wine")
    register_dataset("wine", wine.data, wine.targets)

    breast_cancer = load_dataset("breast_cancer")
    register_dataset("breast_cancer", breast_cancer.data, breast_cancer.targets)

    # Image datasets
    digits = load_dataset("digits")
    register_dataset("digits", digits.data, digits.targets)

    mnist = load_dataset("mnist")
    MAX_MNIST_SAMPLES = 1000
    register_dataset(
        "mnist",
        mnist.train_data[:MAX_MNIST_SAMPLES],
        mnist.train_targets[:MAX_MNIST_SAMPLES],
    )

    orl = load_dataset("orl")
    register_dataset("orl", orl.data, orl.targets)

    # Text datasets (excellent for NMF)
    # newsgroups = load_dataset("20newsgroups")  # 4 categories
    # register_dataset("20newsgroups", newsgroups.data, newsgroups.targets)

    # Optional: Full newsgroups (commented out for speed)
    # newsgroups_full = load_dataset("20newsgroups_full")  # All 20 categories
    # register_dataset("20newsgroups_full", newsgroups_full.data, newsgroups_full.targets)


def create_models(rank, seed, max_outer, max_inner, verbose):
    """Register clustering models with dataset-specific preprocessing."""

    # For similarity matrices (already computed) - used by SyNMF ADMM
    kernel_preprocessors = {
        "mnist": lambda x: construct_similarity_graph(flatten_dataset(x)),
        "orl": lambda x: construct_similarity_graph(flatten_dataset(x)),
        "iris": lambda x: construct_similarity_graph(positive_shift_and_scale(x)),
        "wine": lambda x: construct_similarity_graph(positive_shift_and_scale(x)),
        "breast_cancer": lambda x: construct_similarity_graph(
            positive_shift_and_scale(x)
        ),
        "digits": lambda x: construct_similarity_graph(flatten_dataset(x)),
        "20newsgroups": lambda x: construct_similarity_graph(x),  # Already flattened
        "20newsgroups_full": lambda x: construct_similarity_graph(x),
        "peterson-various": lambda x: x,  # Already a similarity matrix
    }

    # For raw data preprocessing - used by NMF (MUST be non-negative!)
    data_preprocessors = {
        "mnist": lambda x: positive_shift_and_scale(flatten_dataset(x)),
        "orl": lambda x: positive_shift_and_scale(flatten_dataset(x)),
        "iris": lambda x: positive_shift_and_scale(x),  # FIXED: NMF needs non-negative
        "wine": lambda x: positive_shift_and_scale(x),  # FIXED: NMF needs non-negative
        "breast_cancer": lambda x: positive_shift_and_scale(
            x
        ),  # FIXED: NMF needs non-negative
        "digits": lambda x: positive_shift_and_scale(flatten_dataset(x)),
        "20newsgroups": lambda x: x,  # TF-IDF already non-negative and normalized
        "20newsgroups_full": lambda x: x,
        "peterson-various": lambda x: x,  # Similarity matrix
    }

    register_model(
        "SNMF",
        SRF(
            init="random_sqrt",
            rank=rank,
            max_outer=max_outer,
            max_inner=max_inner,
            tol=0.0,
            verbose=verbose,
        ),
        preprocessors=kernel_preprocessors,
    )

    register_model(
        "KMeans X",
        KMeans(
            n_clusters=rank,
            init="random",
            random_state=seed,
            max_iter=max_outer * max_inner,
            n_init=1,
        ),
        preprocessors=data_preprocessors,
    )

    register_model(
        "NMF X",
        NMF(
            n_components=rank,
            random_state=seed,
            init="random",
            max_iter=max_outer * max_inner,
            solver="cd",
            tol=0.0,
        ),
        preprocessors=data_preprocessors,
    )


def evaluate_model_on_dataset(model, dataset_name, model_name, seed):
    """Evaluates a model on a dataset with the correct preprocessing applied."""

    model_preprocessors = MODEL_REGISTRY[model_name]["preprocessors"]
    x, y = DATASET_REGISTRY[dataset_name]["x"], DATASET_REGISTRY[dataset_name]["y"]

    # Apply dataset-specific preprocessing if model-specific exists, else default dataset preprocessing
    if dataset_name in model_preprocessors:
        x = model_preprocessors[dataset_name](x)

    if isinstance(model, (SpectralClustering, KMeans)):
        w = np.eye(x.shape[0])
        y_hat = model.fit_predict(x)
    else:
        w = model.fit_transform(x)
        y_hat = np.argmax(w, axis=1)

    y_hat = map_labels_with_hungarian(y, y_hat)

    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "Seed": seed,
        "Accuracy": accuracy_score(y, y_hat),
        "Purity": purity_score(y, y_hat),
        "Entropy": compute_entropy(y, y_hat),
        "Sparseness": compute_sparseness(w),
        "Orthogonality": compute_orthogonality(w),
    }


def run_clustering_benchmark(
    seeds=None,
    datasets=None,
    max_outer=100,
    max_inner=100,
    verbose=False,
    n_jobs=-1,
    output_dir="./results/clustering",
):
    if seeds is None:
        seeds = range(100)

    # Set module-level constants
    global MAX_OUTER, MAX_INNER, VERBOSE
    MAX_OUTER = max_outer
    MAX_INNER = max_inner
    VERBOSE = verbose

    load_all_datasets()
    tasks = []

    for seed in seeds:
        for dname, ds_info in DATASET_REGISTRY.items():
            if datasets is None or dname in datasets:
                rank = len(np.unique(ds_info["y"]))
                create_models(rank, seed, max_outer, max_inner, verbose)

                tasks.extend(
                    (
                        model_info["model"],
                        dname,
                        model_name,
                        seed,
                    )
                    for model_name, model_info in MODEL_REGISTRY.items()
                )

    results = joblib.Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        joblib.delayed(evaluate_model_on_dataset)(*task) for task in tasks
    )

    results_df = pd.DataFrame(results)

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / "graph_clustering.csv"
    results_df.to_csv(fname, index=False)

    if verbose:
        print(f"Clustering benchmark complete. Results saved to '{fname}'.")

    return results_df


# if __name__ == "__main__":
#     run_clustering_benchmark()
