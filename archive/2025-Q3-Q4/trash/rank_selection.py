# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tools.rsa import compute_similarity
from utils.helpers import median_matrix_split
from datasets import load_dataset
from experiments.cross_validation import find_best_rank


def plot_rank_selection(df_full, df_pos, df_neg):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))
    # Plot full dataframes with variance across repeats
    sns.lineplot(data=df_full, x="rank", y="rmse", ax=ax1, errorbar="sd")
    ax1.set_title("Full RSM")
    ax1.axvline(
        x=df_full.groupby("rank")["rmse"].mean().idxmin(),
        color="r",
        linestyle="--",
        alpha=0.5,
    )

    sns.lineplot(data=df_pos, x="rank", y="rmse", ax=ax2, errorbar="sd")
    ax2.set_title("Positive")
    ax2.axvline(
        x=df_pos.groupby("rank")["rmse"].mean().idxmin(),
        color="r",
        linestyle="--",
        alpha=0.5,
    )

    sns.lineplot(data=df_neg, x="rank", y="rmse", ax=ax3, errorbar="sd")
    ax3.set_title("Negative")
    ax3.axvline(
        x=df_neg.groupby("rank")["rmse"].mean().idxmin(),
        color="r",
        linestyle="--",
        alpha=0.5,
    )

    plt.tight_layout()
    plt.show()


# %%

x = np.loadtxt("/LOCAL/fmahner/srf/data/misc/spose_embedding_66d.txt")
rsm = compute_similarity(x, x, "pearson")
n = rsm.shape[0]

s_plus, s_minus, thresh, mask = median_matrix_split(rsm)


plt.figure(figsize=(4, 4))
sns.clustermap(s_minus, cmap="viridis")
plt.title("Negative RSM")
plt.figure(figsize=(4, 4))
sns.clustermap(s_plus, cmap="viridis")
plt.title("Positive RSM")
plt.tight_layout()
plt.show()


# %%

dataset = load_dataset("peterson-animals")

images = dataset.images
rsm = dataset.rsm
n = rsm.shape[0]
repeats = 50
s_plus, s_minus, thresh, mask = median_matrix_split(rsm)

ratio = 0.2
df_full = find_best_rank(
    rsm,
    range(1, 12),
    train_ratio=ratio,
    similarity_measure="linear",
    n_repeats=repeats,
)
df_pos = find_best_rank(
    s_plus,
    range(1, 12),
    train_ratio=ratio,
    similarity_measure="linear",
    n_repeats=repeats,
)
df_neg = find_best_rank(
    s_minus * -1 + thresh,
    range(1, 12),
    train_ratio=ratio,
    similarity_measure="linear",
    n_repeats=repeats,
)

plot_rank_selection(df_full, df_pos, df_neg)
