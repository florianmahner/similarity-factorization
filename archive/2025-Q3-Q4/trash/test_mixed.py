# %%
import pyximport

import numpy as np

pyximport.install(language_level=3, setup_args={"include_dirs": np.get_include()})


from srf.mixed.bsum_cython import update_bsum


# %%

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from srf.models.mixed import SymmetricMixedResFree
from srf.helpers import evar, best_pairwise_match
from sklearn.metrics.pairwise import cosine_similarity


def make_positive_and_negative_matrices(n, r, random_state=42):
    rng = np.random.RandomState(random_state)
    w_p = 0.5 * rng.rand(n, r)
    w_n = 0.5 * rng.rand(n, r)
    h_p = w_p.copy()
    h_n = w_n.copy()
    # corrected a: positive block and negative block on diagonal
    a = np.zeros((2 * r, 2 * r))
    a[:r, :r] = np.eye(r)
    a[r:, r:] = -np.eye(r)
    w_stacked = np.hstack([w_p, w_n])
    h_stacked = np.hstack([h_p, h_n])
    s = w_stacked @ a @ h_stacked.T
    return s, w_stacked, a, h_stacked


n, r = 300, 10
s, w_orig, a, h_orig = make_positive_and_negative_matrices(n=n, r=r, random_state=42)

plt.imshow(s)
plt.colorbar()
plt.show()


# %%

model = SymmetricMixedResFree(
    rank=r,
    alpha=1.0,
    init="random",
    verbose=False,
    max_iter=1000,
    tol=0.0,
)
model.fit(s)

wp = model.w_pos_
wn = model.w_neg_
hp = model.h_pos_
hn = model.h_neg_
s_plus = model.s_plus_
s_minus = model.s_minus_

s_plus_hat = wp @ hp.T
s_minus_hat = wn @ hn.T

s_hat = model.s_hat_

print("Evar(s_minus, s_minus_hat):", evar(s_minus, s_minus_hat))
print("Evar(s_plus, s_plus_hat):", evar(s_plus, s_plus_hat))


print("Evar(s, s_hat):", evar(s, s_hat))


# %%


his = model.history

plt.figure(figsize=(10, 5))
plt.plot(his["s_plus_evar"])
plt.plot(his["s_minus_evar"])
plt.show()


# %%
plt.figure(figsize=(10, 5))
# plt.plot(his["rec_error"])
plt.plot(his["neg_mass"])
plt.plot(his["pos_mass"])
plt.show()

# plt s and s_hat side by side using seaborn
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(s, cmap="viridis", ax=axs[0])
sns.heatmap(s_hat, cmap="viridis", ax=axs[1])
plt.show()


# %%

w_spose = np.loadtxt("/LOCAL/fmahner/srf/data/misc/spose_embedding_66d.txt")
n, r = w_spose.shape
s = w_spose @ w_spose.T
# s = s / np.max(s)
s = cosine_similarity(w_spose)


# %%
model.rank = r
model.alpha = 1.0
model.init = "random"
model.verbose = True
model.max_iter = 500
model.fit(s)


# %%
wp = model.w_pos_
wn = model.w_neg_
hp = model.h_pos_
hn = model.h_neg_
s_plus = model.s_plus_
s_minus = model.s_minus_

s_plus_hat = wp @ hp.T
s_minus_hat = wn @ hn.T

s_hat = model.s_hat_

print("Evar(s_minus, s_minus_hat):", evar(s_minus, s_minus_hat))
print("Evar(s_plus, s_plus_hat):", evar(s_plus, s_plus_hat))


print("Evar(s, s_hat):", evar(s, s_hat))

w_pred = wp - wn
corrs = best_pairwise_match(w_spose, w_pred)
plt.plot(corrs)

# %%
sns.heatmap(model.s_minus_)
