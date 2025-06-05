# %%
%load_ext autoreload
%autoreload 2
# 
from srf.mixed.admm import run_cv_experiment, find_best_rank, admm_symnmf_masked, ADMM
from srf.helpers import evar
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tools.rsa import compute_similarity
from srf.mixed.admm import admm_symnmf_masked, train_val_split, ADMM
from srf.datasets import load_dataset
from srf.helpers import median_matrix_split, best_pairwise_match, zero_matrix_split
from srf.plotting import plot_images_from_embedding

# %%
# TODO Can check this later if it works and how we can do this with the separate positive and negative masks!
# Important is that we use the linear kernel for the rank selection experiments, definitely for behavior. for the rest i will need to check!!


dataset = load_dataset("things-monkey-2k")

images = dataset.images
rsm = dataset.rsm
# rsm = dataset.group_rsm
n = rsm.shape[0]
s_plus, s_minus, thresh, mask = median_matrix_split(rsm)

ratio = 0.5

results_all = find_best_rank(rsm, range(5, 55, 10), train_ratio=ratio, similarity_measure="linear")



results_all = find_best_rank(rsm, range(1, 30), train_ratio=ratio, similarity_measure="linear", bounds=(rsm.min(), rsm.max()))

#%%



dataset = load_dataset("cichy118")

images = dataset.images
rsm = dataset.rsm
# rsm = dataset.group_rsm
n = rsm.shape[0]
s_plus, s_minus, thresh, mask = median_matrix_split(rsm)

ratio = 0.1

results_all = find_best_rank(rsm, range(1, 30), train_ratio=ratio, similarity_measure="linear", bounds=(rsm.min(), rsm.max()))





# results_pos = find_best_rank(
#     s_plus, range(1, 30), train_ratio=ratio, mask=mask, similarity_measure="linear"
# )
# results_neg = find_best_rank(
#     s_minus, range(1, 30), train_ratio=ratio, mask=~mask, similarity_measure="linear"
# )

# results_pos = find_best_rank(
#     s_plus, range(1, 30), train_ratio=ratio, similarity_measure="linear", bounds=(s_plus.min(), s_plus.max())
# )
# results_neg = find_best_rank(
#     s_minus, range(1, 30), train_ratio=ratio, similarity_measure="linear", bounds=(s_minus.min(), s_minus.max())
# )




#%%
model = ADMM(rank=2, max_outer=10, w_inner=40, tol=0.0, verbose=True)
w_minus = model.fit_transform(s_minus, bounds=(s_minus.min(), s_minus.max()))
model = ADMM(rank=6, max_outer=10, w_inner=40, tol=0.0, verbose=True)
w_plus = model.fit_transform(s_plus, bounds=(s_plus.min(), s_plus.max()))

plot_images_from_embedding(w_plus, images, top_k=10)
plot_images_from_embedding(w_minus, images, top_k=10)


#%%

s_minus_hat = ~mask * (w_minus @ w_minus.T)
s_plus_hat = mask * (w_plus @ w_plus.T)
s_hat = s_minus_hat - s_plus_hat + thresh
print(evar(rsm, s_minus_hat))




#%%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

ax1.plot(results_all.keys(), results_all.values(), label="all")
ax1.set_title("All")
ax1.legend()

ax2.plot(results_pos.keys(), results_pos.values(), label="pos")
ax2.set_title("Positive")
ax2.legend()

ax3.plot(results_neg.keys(), results_neg.values(), label="neg")
ax3.set_title("Negative")
ax3.legend()

plt.tight_layout()
plt.show()


rank_pos = min(results_pos, key=results_pos.get)
rank_neg = min(results_neg, key=results_neg.get)
rank_all = min(results_all, key=results_all.get)

print(f"best rank for positive matrix: {rank_pos}")
print(f"best rank for negative matrix: {rank_neg}")
print(f"best rank for all matrix: {rank_all}")
# %%


# THIs is for spose to see if we can recover the rank swiftly
w_true = np.loadtxt("/LOCAL/fmahner/srf/data/misc/spose_embedding_66d.txt")
w_true = np.maximum(w_true, 0.0)
s = compute_similarity(w_true, w_true, "pearson")
n, r = w_true.shape
s_plus, s_minus, median, mask_plus = median_matrix_split(s)

results_pos = find_best_rank(s_plus, range(5, 95, 5), train_ratio=0.1, similarity_measure="linear")
results_minus = find_best_rank(s_minus, range(5, 95, 10), train_ratio=0.1, similarity_measure="linear")



# %%


#NOTE wehn rho is too high the rank seleciton crashes somehow! This the simulation. make this in greater detail to understand first
results = run_cv_experiment(
    n=200,
    r_true=10,
    train_ratio=0.3,
    n_jobs=50,
    candidate_ranks=range(2, 15),
    similarity_measure="linear",
)
plt.plot(results.keys(), results.values())

print("\nValidation error by rank:", results)

# look for the best (minimum) validation error
best_rank = min(results, key=results.get)
print(f"\nBest rank selected by CV: {best_rank}")

# %%
# Lets run a large parameter sweep over the ADMM parameters


from srf.mixed.hparam_analysis import run_rank_recovery_sweep, plot_success_heatmaps


df_res = run_rank_recovery_sweep(
    n=100,
    true_rank=5,
    candidate_ranks=range(2, 9),
    mask_fractions=np.linspace(0.1, 0.9, 20),
    rho_values=np.linspace(0.1, 10.0, 20),
    noise_levels=[0.0, 0.02],
    repetitions=10,
    n_jobs=-1,
)
print(df_res.head())

#%%
plot_success_heatmaps(df_res)