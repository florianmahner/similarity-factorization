"""Recipe-K + SBM cluster-count estimation (Experiment_SBM).

This sub-package is *additive* to the existing RECIPE_K source tree. It is the
exploratory pairing of Recipe K (spectral calibration of an operating sampling
probability) with an SBM-style clustering family scored by held-out predictive
loss, as described in ``RECIPE_K/docs/plan_SBM_clustering.md``.

Nothing in this directory modifies any existing Recipe K function. The
spectral pass, ``recipe_K`` itself, and the ADMM-based ``nested_cv`` in
``_common.py`` are used as-is via ordinary imports.
"""

from .datasets import build_dataset, DATASET_BUILDERS
from .losses import score_sbm_predictions, LOSS_FAMILIES
from .fitters import fit_sbm, fit_spectral_block_mean
from .cv import nested_cv_sbm, single_mask_cv_sbm, plain_cv_sbm
from .selection import argmin_rank, one_se_rank

__all__ = [
    "build_dataset",
    "DATASET_BUILDERS",
    "score_sbm_predictions",
    "LOSS_FAMILIES",
    "fit_sbm",
    "fit_spectral_block_mean",
    "nested_cv_sbm",
    "single_mask_cv_sbm",
    "plain_cv_sbm",
    "argmin_rank",
    "one_se_rank",
]
