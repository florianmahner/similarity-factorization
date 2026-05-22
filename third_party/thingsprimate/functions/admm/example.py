# Sorry for not compiling it in a notebook, it might be best to compile it in cython for the 8.4k objects,
# since this makes it roughly 130x faster.

import numpy as np
from publish.functions.admm.cross_validation import cross_val_score


w = np.random.rand(100, 10)
s = w @ w.T

scorer = cross_val_score(
    s,
    param_grid={"rank": [5, 10, 15, 20]},
    n_repeats=5,
    observed_fraction=0.8,
    random_state=0,
    verbose=1,
    n_jobs=-1,
    fit_final_estimator=True,
)

# returns a scorer object with the following attributes:
# - cv_results_
# - best_estimator_
# - best_params_
# - best_score_
# - rank_
# - rho_
# - max_outer_
# - max_inner_

cv_results = (
    scorer.cv_results_
)  # this is a dataframe where you can also plot it for instance with seaborn
best_estimator = scorer.best_estimator_


best_embedding = best_estimator.transform(s)
