"""Bootstrap-based rank estimation for symmetric similarity matrices.

The :class:`RankEstimator` estimates the number of signal dimensions
in a symmetric similarity matrix by measuring how stable each
eigenspace dimension is under random entry masking, and returns the
minimum sampling fraction needed to recover that rank's eigenspace
within a chosen spectral tolerance.

Suitable as the ``sampling_fraction`` argument of
:func:`pysrf.cross_val_score` (repeated-holdout CV) or for
:class:`EntryKFold` (k-fold partition CV, after inflation via
``cv_sampling_fraction(n_folds)``).
"""

from .estimator import RankEstimator

__all__ = ["RankEstimator"]
