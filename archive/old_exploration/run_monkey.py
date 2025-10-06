# %%
# from experiments.sparse_spose_recon import compute_similarity_matrix
from pysrf import SRF
from pathlib import Path
import numpy as np
from utils.plotting import plot_images_from_embedding
from utils.io import load_things_image_data
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tools.rsa import correlate_rsms, compute_similarity
from utils.helpers import rbf_entropy_heuristic_subsampled
from sklearn.preprocessing import StandardScaler

monkey = load_dataset("things-monkey-22k", min_reliab=0.2)
rsm = monkey.rsm

it_data = monkey.it

# it_zscored = StandardScaler().fit_transform(it_data)


sigma_values = np.linspace(0.01, 30, 100)
sigma, entropy = rbf_entropy_heuristic_subsampled(
    it_data, sigma_values, return_entropy=True, max_samples=7_000
)

rsm = compute_similarity(it_data, it_data, "gaussian_kernel", sigma=sigma)

# rsm = compute_similarity(it_data, it_data, "gaussian_kernel")


rank = 30


print(f"Fitting model with rank {rank}")
model = SRF(
    rank=rank,
    max_outer=20,
    max_inner=5,
    tol=0.0,
    verbose=True,
    rho=1.0,
)

w = model.fit_transform(rsm)


np.save("monkey_w_gaussian_entropy.npy", w)
