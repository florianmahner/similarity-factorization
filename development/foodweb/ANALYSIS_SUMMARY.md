# Grand Caricaie Food Web Analysis - SRF Results

## Dataset
- **Name**: Grand Caricaie Marsh Food Web (Switzerland)
- **Species**: 163 invertebrates + 7 vertebrates
- **Interactions**: 2,120 directed / 2,085 undirected edges
- **Density**: 15.8%
- **Body mass range**: 9 orders of magnitude (0.00003g - 5,000g)

## Methods Comparison

### 1. SRF (Symmetric Non-negative Matrix Factorization)
**Configuration**: k=15, degree-normalized adjacency, 423 iterations

**Results**:
- Link prediction: AUC = 0.929 ± 0.004, AP = 0.920 ± 0.002
- Factor sparsity: 50-110 species per factor (vs 163 in dense version)
- Body mass correlations: 6/15 factors show |ρ| > 0.2 with log body mass

**Key factors**:
- Factor 0: Small-bodied species (ρ = -0.376)
- Factor 4: Larger-bodied species (ρ = +0.230)
- Factor 8: Flying/mobile species (ρ = +0.215)

### 2. Louvain Community Detection
**Configuration**: Standard modularity optimization

**Results**:
- Communities: 3 (Q = 0.343)
- Community 0 (29 species): Mixed, high body mass variability
- Community 1 (57 species): Flying species + vertebrates
- Community 2 (77 species): Small walkers, low body mass

## Key Findings

### Network Structure Validation
✅ **Network has clear modular structure** (Q = 0.343)
✅ **SRF discovers finer-grained patterns** (15 vs 3 communities)
✅ **Communities align with movement type and body size**
✅ **Degree normalization critical** for reducing hub dominance

### Trophic Organization
- **Basal species**: 76/163 (47%) at TL=1.0
- **Intermediate consumers**: 37/163 (23%)
- **Top predators**: 50/163 (31%) at TL ≥ 2.5
- **Maximum trophic level**: 3.26

### Hub Species
**Top predator (by degree centrality)**:
- Emberiza schoeniclus (Reed Bunting bird): 67% connectivity, TL=2.85

**Top prey (by PageRank)**:
- Aphidoidea (aphids), Springtails, Mites: Basal species with high prey value

## Interpretation

### Why SRF Works Well Here
1. **Symmetric factorization appropriate**: Co-occurrence patterns matter more than directed predation
2. **Degree normalization reduces hub bias**: Reed bunting connects to 108/163 species
3. **Higher rank captures heterogeneity**: Invertebrate-dominated food web needs finer structure
4. **Excellent link prediction**: 93% AUC indicates strong latent structure

### Ecological Meaning
- SRF factors capture **microhabitat + foraging guilds**
- Factor 0: Small detritivores
- Factor 4: Larger mobile predators
- Factors 8-14: Mixed guilds by movement type

### Limitations
- Data is 96% invertebrates → limited trophic diversity
- High connectivity (16% density) → less sparse than typical food webs
- Symmetric adjacency loses predator/prey directionality

## Recommendations
1. **Use k=10-15 for this dataset** (not k=5)
2. **Always apply degree normalization** for hub-dominated networks
3. **Compare to community detection** to validate structure
4. **Body mass provides independent validation** (not circular like trophic levels)

## Files Generated
- `experiments/development/foodweb/run.py` - Main SRF analysis
- `experiments/development/foodweb/validate_structure.py` - Community detection validation
- `experiments/development/foodweb/outputs/[timestamp]/` - SRF results
  - `embedding.npy` - W matrix (163 × 15)
  - `reconstruction.npy` - Reconstructed adjacency
  - `species_with_factors.csv` - Species + factor loadings + trophic levels
  - `statistics.csv` - Performance metrics
  - `link_prediction_results.pkl` - CV results
- `experiments/development/foodweb/outputs/structure_validation.csv` - Community assignments + centrality
- `experiments/development/foodweb/outputs/structure_validation.png` - Validation plots

