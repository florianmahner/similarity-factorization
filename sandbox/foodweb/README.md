# Food Web Analysis with Symmetric NMF

Analysis of the Grand Caricaie Marsh food web using Symmetric Non-negative Matrix Factorization (SRF).

## Quick Start

```bash
# Run SRF analysis with degree normalization
poetry run python experiments/development/foodweb/run.py

# Validate structure with community detection
poetry run python experiments/development/foodweb/validate_structure.py

# Compare methods
poetry run python experiments/development/foodweb/compare_methods.py
```

## Dataset

**Grand Caricaie Marsh Food Web** (Switzerland)
- 163 species (96% invertebrates, 4% vertebrates)
- 2,120 directed interactions (2,085 undirected)
- 15.8% network density
- Body mass: 0.00003g - 5,000g (9 orders of magnitude)
- Complete metadata: body mass, metabolic type, movement type

## Results

### SRF Performance (k=15, degree-normalized)
- **Link prediction**: AUC = 0.929 ± 0.004, AP = 0.920 ± 0.002
- **Factor sparsity**: 50-110 species per factor
- **Body mass correlations**: 6/15 factors with |ρ| > 0.2
- **Convergence**: 423 iterations, R² = 0.987

### Community Detection (Louvain)
- **Communities**: 3 (Q = 0.343)
- Community 0 (29): Mixed, high body mass variability
- Community 1 (57): Flying species + vertebrates
- Community 2 (77): Small walkers, low body mass

## Key Findings

1. **Degree normalization is critical** for hub-dominated networks
   - Reed bunting bird connects to 66% of species
   - Normalization reduces hub bias, improves factor interpretability

2. **Higher rank captures heterogeneity**
   - k=15 provides better sparsity than k=5
   - Invertebrate-dominated webs need finer structure

3. **Factors correlate with ecological traits**
   - Factor 0: Small detritivores (ρ = -0.376 with body mass)
   - Factor 4: Larger mobile predators (ρ = +0.230)
   - Factors separate by movement type and trophic position

4. **Trophic structure**
   - 47% basal species (TL=1.0)
   - 31% top predators (TL ≥ 2.5)
   - Max trophic level: 3.26

## Files

```
experiments/development/foodweb/
├── run.py                      # Main SRF analysis
├── validate_structure.py       # Community detection validation
├── compare_methods.py          # Method comparison plots
├── ANALYSIS_SUMMARY.md         # Detailed results
├── README.md                   # This file
└── outputs/
    ├── [timestamp]/
    │   ├── embedding.npy       # W matrix (163 × 15)
    │   ├── reconstruction.npy  # Reconstructed adjacency
    │   ├── species_with_factors.csv
    │   ├── statistics.csv
    │   └── link_prediction_results.pkl
    ├── structure_validation.csv
    ├── structure_validation.png
    └── method_comparison.png
```

## Implementation Details

### Preprocessing
1. Load symmetric adjacency matrix (binary, undirected)
2. Apply degree normalization: D^(-1/2) A D^(-1/2)
3. Compute Lindeman trophic levels from directed adjacency

### SRF Configuration
- Rank: k=15
- Max iterations: 2000
- Random state: 42
- Degree normalization: True

### Link Prediction
- 5-fold cross-validation
- 80% sampling fraction
- Metrics: AUC-ROC, Average Precision

## Recommendations

1. **Always use degree normalization** for food webs with hubs
2. **Use k=10-15** for invertebrate-dominated webs
3. **Validate with community detection** to confirm structure
4. **Body mass provides independent validation** (not circular)

## References

- Gateway database: https://www.idiv.de/en/gateway.html
- Lindeman trophic levels: TL = 1 + mean(prey TL)
- Symmetric NMF: Non-negative factorization with W=H^T

