import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

Array = np.ndarray


def norm_columns(v: Array) -> Array:
    col_norms = np.linalg.norm(v, axis=0, keepdims=True)
    col_norms[col_norms == 0] = 1
    return v / col_norms


from joblib import Parallel, delayed


def submit_parallel_jobs(
    func, args, joblib_kwargs: dict = {"n_jobs": -1, "verbose": 10}
):
    """
    Submit parallel jobs to a function.
    """
    parallel = Parallel(**joblib_kwargs)
    return parallel(delayed(func)(*arg) for arg in args)


def clustering_single(x: Array, n: int, algorithm: type, cluster_kwargs: dict) -> dict:
    clustering = algorithm(n_clusters=n, **cluster_kwargs)
    clustering.fit(x)
    cluster_labels = clustering.labels_

    return {
        "n_clusters": n,
        "silhouette_score": silhouette_score(x, cluster_labels),
        "davies_bouldin_score": davies_bouldin_score(x, cluster_labels),
        "calinski_harabasz_score": calinski_harabasz_score(x, cluster_labels),
    }


def clustering_parallel(
    x: Array,
    algorithm: type = KMeans,
    min_clusters: int = 2,
    max_clusters: int = 90,
    step: int = 5,
    cluster_kwargs: dict = {},
) -> pd.DataFrame:
    cluster_range = range(min_clusters, max_clusters + 1, step)
    func = run_clustering_single
    args = [(x, n, algorithm, cluster_kwargs) for n in cluster_range]
    cluster_results = submit_parallel_jobs(func, args)

    return cluster_results


def cluster_stacked_embeddings(
    embeddings: dict[str, Array],
    algorithm: type = KMeans,
    cluster_kwargs: dict = {"random_state": None, "init": "k-means++", "n_init": 30},
    metric: str = "silhouette_score",
    min_clusters: int = 50,
    max_clusters: int = 60,
    step: int = 5,
    norm_cols: bool = True,
) -> dict[str, Array]:
    """Run clustering for a dictionary of embeddings, where each key is a model name
    and each value is a stacked embedding matrix with columns being dimensions of the embeddings.
    across nmf initializations.
    """

    results = []
    clustered_embeddings = {}

    def process_model(model_data):
        model_name, stacked_embeddings = model_data
        if norm_cols:
            stacked_embeddings = norm_columns(stacked_embeddings)

        # Run clustering in parallel for this model
        cluster_results = clustering_parallel(
            stacked_embeddings.T,
            algorithm=algorithm,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            step=step,
            cluster_kwargs=cluster_kwargs,
        )
        # Find optimal number of clusters based on metric
        best_k = max(cluster_results, key=lambda x: x[metric])["n_clusters"]

        # add the model name and the best k to the cluster results
        cluster_results = [
            {"model": model_name, **d, "best_k": best_k} for d in cluster_results
        ]

        # Fit final clustering and merge clusters
        kmeans = algorithm(n_clusters=best_k, **cluster_kwargs)
        labels = kmeans.fit(stacked_embeddings.T).labels_
        clustered = merge_clusters_by_median(
            stacked_embeddings, labels, best_k, sort_by_sum=True
        )

        return model_name, cluster_results, clustered

    # Process all models in parallel
    args = [(item,) for item in embeddings.items()]
    parallel_results = submit_parallel_jobs(process_model, args)

    # Combine results
    for model_name, model_results, model_clustered in parallel_results:
        results.extend(model_results)
        clustered_embeddings[model_name] = model_clustered

    results = [{k: v for k, v in d.items() if k != "Unnamed: 0"} for d in results]

    return pd.DataFrame(results), clustered_embeddings


def find_best_number_of_clusters(
    df: pd.DataFrame, key: str = "silhouette_score"
) -> int:
    return int(df.iloc[np.argmax(df[key])]["n_clusters"])


def merge_clusters_by_median(
    x: Array, clusters: Array, num_clusters: int, sort_by_sum: bool = True
) -> Array:
    """Merge clusters by taking the median of the data (ie embeddings) in each cluster."""
    cluster_medians = np.array(
        [np.median(x[:, clusters == i], axis=1) for i in range(num_clusters)]
    ).T

    if sort_by_sum:
        cluster_sums = np.array(
            [np.sum(x[:, clusters == i]) for i in range(num_clusters)]
        )
        sort_indices = np.argsort(-cluster_sums)
        cluster_medians = cluster_medians[:, sort_indices]

    return cluster_medians
