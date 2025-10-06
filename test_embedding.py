# %%
from datasets import load_dataset
from pysrf import cross_val_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ds = load_dataset("peterson-animals")


cv = cross_val_score(
    ds.rsm,
    estimate_sampling_fraction=True,
    # sampling_fraction=0.9,
    param_grid={"rank": [1, 2, 3, 4, 5, 6, 7, 8]},
    n_jobs=4,
    n_repeats=10,
    fit_final_estimator=True,
)

sns.lineplot(x="rank", y="score", data=cv.cv_results_)


from sklearn import pipeline
from pysrf.consensus import EnsembleEmbedding, ClusterEmbedding
from pysrf import SRF

pipe = pipeline.Pipeline(
    [
        ("ensemble", EnsembleEmbedding(SRF(**cv.best_params_), n_runs=50)),
        ("cluster", ClusterEmbedding(min_clusters=2, max_clusters=6, step=1)),
    ]
)

output = pipe.fit(ds.rsm)
sns.lineplot(
    x="n_clusters", y="silhouette_score", data=output["cluster"].cluster_results_
)

final_embeddings = pipe.transform(ds.rsm)

# %%
# plot the embedding

images = ds.metadata["images"]

# please make on large image grid with the topk images per dimension
topk = 12

emb = final_embeddings
n_dims = emb.shape[1]
fig, axs = plt.subplots(n_dims, topk, figsize=(topk, n_dims), dpi=200)
for i in range(n_dims):
    idx = np.argsort(emb[:, i])[::-1][:topk]
    for j in range(topk):
        import PIL.Image as Image

        img = Image.open(images[idx[j]])
        axs[i, j].imshow(img)
        axs[i, j].axis("off")
plt.tight_layout()
plt.show()

# %%
