# %%
%load_ext autoreload
%autoreload 2

from srf.mixed.admm import run_cv_experiment, find_best_rank, admm_symnmf_masked, ADMM
from srf.helpers import evar
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tools.rsa import compute_similarity
from srf.mixed.admm import admm_symnmf_masked, train_val_split
from srf.datasets import load_dataset
from srf.helpers import median_matrix_split, best_pairwise_match, zero_matrix_split
from srf.io import load_things_image_data
from srf.plotting import plot_images_from_embedding

# %%

# Here check quickly the properties of the matrix
w_true = np.loadtxt("/LOCAL/fmahner/srf/data/misc/spose_embedding_66d.txt")
w_true = np.maximum(w_true, 0.0)
s = compute_similarity(w_true, w_true, "pearson")
n, r = w_true.shape
s_plus, s_minus, median, mask_plus = median_matrix_split(s)

# s_plus, s_minus, median, mask_plus = zero_matrix_split(s)



#%%

# think of this again later.

results_pos = find_best_rank(s_plus, range(5, 95, 5), train_ratio=0.1, similarity_measure="linear")
results_minus = find_best_rank(s_minus, range(5, 95, 10), train_ratio=0.1, similarity_measure="linear")

#%%

model = ADMM(rank=20, max_outer=10, w_inner=40, tol=0.0, verbose=True)

w_plus = model.fit_transform(s_plus, mask_plus, bounds=(s_plus.min(), s_plus.max()))
w_minus = model.fit_transform(s_minus, ~mask_plus, bounds=(s_minus.min(), s_minus.max()))



#%%

w = np.concatenate([w_plus, w_minus], axis=1)
# give me a pearson correlation matrix of w columns 
corr_matrix = np.corrcoef(w, rowvar=False)

# plot the correlation matrix
plt.figure(figsize=(10, 8))
np.fill_diagonal(corr_matrix, 0.0)
plt.imshow(corr_matrix, cmap="coolwarm")

plt.colorbar()
plt.xlabel(f'Features (Pos: {w_plus.shape[1]}, Neg: {w_minus.shape[1]})')
# Add vertical line to show separation
plt.axvline(x=w_plus.shape[1]-0.5, color='white', linestyle='--', linewidth=2)
plt.axhline(y=w_plus.shape[1]-0.5, color='white', linestyle='--', linewidth=2)
plt.show()

# TODO find the ones that highly correlate between pos and neg later






#%%

s_plus_hat = mask_plus * (w_plus @ w_plus.T)
s_minus_hat = ~mask_plus * (w_minus @ w_minus.T)
print("Evar(s_plus, s_plus_hat):", evar(s_plus, s_plus_hat))
print("Evar(s_minus, s_minus_hat):", evar(s_minus, s_minus_hat))

s_hat_additive = s_plus_hat - s_minus_hat + median
print("Evar(s, s_hat):", evar(s, s_hat))
print("Evar(s, s_hat_additive):", evar(s, s_hat_additive))


#%%
plt.hist(w_plus.flatten(), bins=30)
plt.show()
plt.hist(w_minus.flatten(), bins=30)
plt.show()


# %%


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
corrs = best_pairwise_match(w_true, w_plus)
plt.plot(corrs)
plt.title("Pairwise corrs of w_plus with x_spose")
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
corrs = best_pairwise_match(w_true, w_minus)
plt.plot(corrs)
plt.title("Pairwise corrs of w_minus with x_spose")
plt.show()



#%%


images = load_things_image_data("/SSD/datasets/things", filter_behavior=True)

fig = plot_images_from_embedding(w_plus, images, top_k=12)

fig = plot_images_from_embedding(w_minus, images, top_k=12)



#%%

n=300
r=10
w = np.random.rand(n, r)
s = w @ w.T

model = ADMM(rank=r, max_outer=10, w_inner=40, tol=0.0, verbose=True)
w_hat = model.fit_transform(s, bounds=(s.min(), s.max()))

objective = model.history_["total_objective"]
plt.plot(objective)

# check if decreasing monotonically
print("\nIs objective decreasing monotonically?", np.all(np.diff(objective) < 0))






#%%






# %%

from srf.mixed.cd_updates import update_w

def pos_neg_residual_loop(
    s: np.ndarray,
    rank: int,
    max_outer: int = 50,
    w_inner: int = 40,
    tol: float = 1e-5,
    seed: int | None = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Two-stage residual scheme using a *single* non-negative factor w.

    Step 1 (positive pass) :  z_pos = max(s + r_neg, 0)
                              w  ← update_w(z_pos, w, ...)
    Step 2 (residual pass) :  r_neg = max(-(s - w wᵀ), 0)

    Returns
    -------
    w        The learned factor (n × rank, non-negative)
    history  Relative Frobenius residual ‖S−w wᵀ‖/‖S‖ per outer iter
    """
    rng = np.random.default_rng(seed)
    n = s.shape[0]
    w = 0.01 * rng.random((n, rank))         # non-negative init
    r_neg = np.zeros_like(s)                 # carry-over negative residual
    history: list[float] = []

    for outer in range(max_outer):
        # ---- positive stage -------------------------------------------------
        z_pos = np.maximum(s + r_neg, 0.0)   # non-neg target for update_w
        w = update_w(z_pos, w, max_iter=w_inner, tol=1e-8)

        # ---- residual bookkeeping ------------------------------------------
        residual = s - w @ w.T
        r_neg = np.maximum(-residual, 0.0)   # everything S can't yet explain
        rel_err = np.linalg.norm(residual, "fro") / np.linalg.norm(s, "fro")
        history.append(rel_err)
        if rel_err < tol:                    # convergence criterion
            break

    return w, history

w = np.random.rand(n, r)
s = w @ w.T
w, history = pos_neg_residual_loop(s, r, max_outer=20, w_inner=20, tol=1e-5)

#%%



# %%
