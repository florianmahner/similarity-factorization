THINGS Behavioral Similarity Matrix
====================================

File: things_behavior_similarity.npy
Format: NumPy array, float64, shape (1854, 1854)

This is the behavioral similarity matrix for the THINGS object concepts,
derived from 4.7 million odd-one-out triplet judgments (Hebart et al., 2020).

Properties:
  - 1854 x 1854 (one row/column per object concept)
  - Symmetric, with diagonal = 1.0
  - Values in [0.067, 1.0]
  - 8 NaN entries (4 symmetric pairs where no triplet covered that object pair)
  - NOT a correlation matrix -- values represent triplet-derived similarity

Construction:
  For each pair (i, j), similarity = fraction of triplets containing both i and j
  where a human judged them as more similar than the third object. Pairs never
  co-occurring in any triplet are NaN.

Loading:
  import numpy as np
  s = np.load("things_behavior_similarity.npy")
