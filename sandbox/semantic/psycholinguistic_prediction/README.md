# Psycholinguistic Prediction from SWOW Embeddings

## Why Cosine Similarity, Not Dot Product?

SRF factorizes the PPMI matrix via W @ W.T (dot product reconstruction). However, for downstream semantic tasks, **cosine similarity outperforms dot product**:

| Metric | SimLex-999 ρ | WordSim-353 ρ |
|--------|--------------|---------------|
| Dot product | 0.38 | 0.69 |
| Cosine similarity | 0.55 | 0.72 |

**Reason**: PPMI encodes both semantic relatedness AND word frequency. High-frequency words accumulate larger embedding norms (21.8x range in our embeddings). Dot product is dominated by these magnitude differences, while cosine normalizes them out, isolating semantic direction.

```
Lowest norm:  "South Pole", "quid", "magma" (rare words)
Highest norm: "love", "ocean", "church" (common words)
```

**Conclusion**: For PPMI-based embeddings, always use cosine similarity for semantic tasks.

## Is PPMI the Best Similarity Measure?

PPMI is well-established for word association data, but alternatives exist:

| Measure | Pros | Cons |
|---------|------|------|
| **PPMI** | Removes frequency bias, emphasizes rare co-occurrences | Can overweight rare events |
| Raw counts | Simple | Dominated by frequent words |
| Log counts | Reduces frequency skew | Less interpretable |
| NPMI (normalized) | Bounded [-1, 1] | May lose signal for rare pairs |

For SWOW specifically, PPMI is a reasonable default because:
1. Word associations are sparse (most pairs never co-occur)
2. We want to emphasize meaningful associations over frequency
3. It's the standard for SVD/NMF-based word embeddings (e.g., GloVe uses related weighting)

**Possible improvements to explore**:
- Context distribution smoothing (α=0.75 as in word2vec)
- Shifted PPMI (subtract log(k) to reduce dimensionality bias)
- Different symmetrization methods (currently geometric mean)
