Peterson Animals Similarity Matrix
===================================

File: peterson_animals_similarity.npy
Format: NumPy array, float64, shape (120, 120)

Behavioral similarity ratings for 120 animal concepts from Peterson et al.
Participants rated pairwise similarity on a 0-10 scale.

Properties:
  - 120 x 120 (one row/column per animal)
  - Symmetric, no NaN
  - Values in [0.0, 10.0]
  - Diagonal = 10.0 (max similarity = self)

Known issue with coherence analysis:
  - Kappa changepoint gives k*=9 (stable across k_max)
  - But signal-noise liftoff returns p*=None because the boundary
    dimension (k=9) never clearly separates from the noise reference
  - The noise reference curve sits ABOVE the signal curve for all p
  - This suggests the liftoff logic needs improvement for matrices
    with weak boundary dimensions

Loading:
  import numpy as np
  s = np.load("peterson_animals_similarity.npy")
