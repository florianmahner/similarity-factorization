import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import gzip
from tqdm import tqdm


def load_gtex_median_tpm(gtex_path: Path) -> pd.DataFrame:
    with gzip.open(gtex_path, 'rt') as f:
        lines = f.readlines()

    header_line = None
    for i, line in enumerate(lines):
        if line.startswith('Name\tDescription'):
            header_line = i
            break

    df = pd.read_csv(gtex_path, compression='gzip', sep='\t', skiprows=header_line)

    df.index = df['Description']
    df = df.drop(columns=['Name', 'Description'])

    df = df[df.index != '']
    df = df.groupby(df.index).mean()

    return df


def sample_protein_pairs(proteins: np.ndarray, n_pairs: int = 50000, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    n_proteins = len(proteins)

    pairs = []
    while len(pairs) < n_pairs:
        i, j = np.random.choice(n_proteins, size=2, replace=False)
        if i != j:
            pairs.append((min(i, j), max(i, j)))

    pairs = np.array(sorted(set(pairs)))

    return pairs[:n_pairs]


def compute_embedding_distances(embedding: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    distances = np.zeros(len(pairs))

    for idx, (i, j) in enumerate(pairs):
        sim = np.dot(embedding[i], embedding[j]) / (np.linalg.norm(embedding[i]) * np.linalg.norm(embedding[j]))
        distances[idx] = 1 - sim

    return distances


def compute_expression_correlations(expression_df: pd.DataFrame, proteins: np.ndarray, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    correlations = []
    valid_mask = []

    for i, j in pairs:
        gene_i = proteins[i]
        gene_j = proteins[j]

        if gene_i in expression_df.index and gene_j in expression_df.index:
            expr_i = expression_df.loc[gene_i].values
            expr_j = expression_df.loc[gene_j].values

            if np.std(expr_i) > 0 and np.std(expr_j) > 0:
                corr, _ = pearsonr(expr_i, expr_j)
                correlations.append(corr)
                valid_mask.append(True)
            else:
                correlations.append(0)
                valid_mask.append(False)
        else:
            correlations.append(0)
            valid_mask.append(False)

    return np.array(correlations), np.array(valid_mask)


def analyze_tissue_specificity(
    embedding: np.ndarray,
    expression_df: pd.DataFrame,
    proteins: np.ndarray,
    top_factors: list[int],
    n_top_proteins: int = 50
) -> pd.DataFrame:
    results = []

    for factor_idx in top_factors:
        loadings = embedding[:, factor_idx]
        top_indices = np.argsort(loadings)[-n_top_proteins:]

        top_genes = []
        for idx in top_indices:
            if proteins[idx] in expression_df.index:
                top_genes.append(proteins[idx])

        if len(top_genes) < 10:
            continue

        factor_expression = expression_df.loc[top_genes]
        mean_expression = factor_expression.mean(axis=0)
        max_tissue = mean_expression.idxmax()
        max_expr = mean_expression.max()
        median_expr = mean_expression.median()

        specificity_score = max_expr / (median_expr + 1e-6)

        results.append({
            'factor': factor_idx,
            'n_genes': len(top_genes),
            'max_tissue': max_tissue,
            'max_expression': max_expr,
            'median_expression': median_expr,
            'specificity_score': specificity_score
        })

    return pd.DataFrame(results)


def plot_distance_correlation(distances: np.ndarray, correlations: np.ndarray, valid_mask: np.ndarray, method: str, output_dir: Path):
    valid_distances = distances[valid_mask]
    valid_correlations = correlations[valid_mask]

    spearman_corr, spearman_pval = spearmanr(valid_distances, valid_correlations)

    fig, ax = plt.subplots(figsize=(8, 6))

    sample_indices = np.random.choice(len(valid_distances), size=min(5000, len(valid_distances)), replace=False)
    ax.scatter(
        valid_distances[sample_indices],
        valid_correlations[sample_indices],
        alpha=0.3,
        s=10,
        color='#2E86AB'
    )

    ax.set_xlabel('Embedding distance (1 - cosine similarity)')
    ax.set_ylabel('Expression correlation (Pearson)')
    ax.set_title(f'{method}: embedding distance vs co-expression\nSpearman = {spearman_corr:.3f} (p < {spearman_pval:.2e})')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{method.lower()}_distance_correlation.pdf', dpi=300, bbox_inches='tight')
    plt.close()

    return spearman_corr, spearman_pval


def plot_tissue_specificity(tissue_df: pd.DataFrame, output_dir: Path):
    if len(tissue_df) == 0:
        print("No tissue specificity data to plot")
        return

    top_factors = tissue_df.nlargest(min(10, len(tissue_df)), 'specificity_score')

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(top_factors))
    ax.barh(y_pos, top_factors['specificity_score'], alpha=0.7, color='#2E86AB')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Factor {f} ({t[:20]})" for f, t in zip(top_factors['factor'], top_factors['max_tissue'])], fontsize=9)
    ax.set_xlabel('Tissue specificity score (max/median expression)')
    ax.set_title('Factors with highest tissue specificity')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'tissue_specificity.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    base_dir = Path("/LOCAL/fmahner/similarity-factorization")
    data_dir = base_dir / "data"
    output_dir = base_dir / "experiments" / "ppi" / "outputs" / "corum_validation"
    validation_output_dir = base_dir / "experiments" / "ppi" / "validation" / "outputs" / "coexpression"
    validation_output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GTEx median TPM data...")
    gtex_path = data_dir / "gtex" / "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
    gtex_df = load_gtex_median_tpm(gtex_path)
    print(f"GTEx shape (genes x tissues): {gtex_df.shape}")
    print(f"Tissues: {gtex_df.shape[1]}")

    print("\nLoading rank-50 SRF embedding...")
    embedding = np.load(output_dir / "embedding.npy")
    print(f"Embedding shape: {embedding.shape}")

    with open(output_dir / "proteins.txt", "r") as f:
        proteins = np.array([line.strip() for line in f])
    print(f"Number of proteins: {len(proteins)}")

    overlap_proteins = set(proteins) & set(gtex_df.index)
    print(f"\nOverlapping proteins (in both embedding and GTEx): {len(overlap_proteins)}")

    print("\nSampling protein pairs...")
    pairs = sample_protein_pairs(proteins, n_pairs=50000, seed=42)
    print(f"Sampled {len(pairs)} protein pairs")

    print("\nComputing embedding distances...")
    embedding_distances = compute_embedding_distances(embedding, pairs)

    print("Computing expression correlations...")
    expression_correlations, valid_mask = compute_expression_correlations(gtex_df, proteins, pairs)
    print(f"Valid pairs (both proteins in GTEx): {valid_mask.sum()}/{len(pairs)}")

    print("\nAnalyzing correlation between embedding distance and co-expression...")
    spearman_corr, spearman_pval = plot_distance_correlation(
        embedding_distances,
        expression_correlations,
        valid_mask,
        "SRF",
        validation_output_dir
    )

    results = {
        'method': 'SRF',
        'n_pairs': len(pairs),
        'n_valid_pairs': int(valid_mask.sum()),
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_pval
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(validation_output_dir / "results.csv", index=False)

    print("\nAnalyzing tissue specificity of factors...")
    top_factors = list(range(min(20, embedding.shape[1])))
    tissue_specificity = analyze_tissue_specificity(
        embedding,
        gtex_df,
        proteins,
        top_factors,
        n_top_proteins=50
    )
    tissue_specificity.to_csv(validation_output_dir / "tissue_specificity.csv", index=False)

    print("\nPlotting tissue specificity...")
    plot_tissue_specificity(tissue_specificity, validation_output_dir)

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nCorrelation Analysis:")
    print(f"  Spearman correlation: {spearman_corr:.4f}")
    print(f"  P-value: {spearman_pval:.2e}")
    print(f"  Valid pairs: {valid_mask.sum()}/{len(pairs)}")

    print(f"\nTissue Specificity:")
    print(tissue_specificity.nlargest(5, 'specificity_score').to_string(index=False))

    print(f"\nResults saved to: {validation_output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
