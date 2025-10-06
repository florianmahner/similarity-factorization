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

from pysrf import SRF
from utils.plotting import plot_images_from_embedding
from datasets import load_dataset
from tools.rsa import compute_similarity
from tools.metrics import median_sigma_heuristic
from utils.helpers import rbf_entropy_heuristic
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import time
import json

# Start timing
start_time = time.time()


x = np.load(
    "/SSD/projects/deepsim/raw/features/dataset/laion2b_s32b_b82k/ViT-L-14/visual/features.npy"
)
rsm = compute_similarity(x, x, "gaussian_kernel")

rank = 30

print(f"Fitting model with rank {rank}")
model_start = time.time()
model = SRF(
    rank=rank,
    max_outer=100,
    max_inner=5,
    tol=0.0,
    verbose=True,
    rho=1.0,
)

w = model.fit_transform(rsm)


timing_results = {
    "time_minutes": (time.time() - start_time) / 60,
    "rank": rank,
    "data_shape": x.shape,
    "rsm_shape": rsm.shape,
}

# Save timing to JSON
with open("./embedings/vit_timing_results.json", "w") as f:
    json.dump(timing_results, f, indent=2)


# Save results
np.save("./embedings/vit_laion_27k.npy", w)

# Prepare timing results
