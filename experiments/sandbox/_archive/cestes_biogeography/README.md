# CESTES Biogeography: Species Co-occurrence Analysis

## Goal
Discover ecological dimensions from species co-occurrence patterns using SRF.

## Dataset
- **Source**: Dune meadow vegetation data from R's `vegan` package (Jongman et al. 1987)
- **20 sites** across dune meadows with varying moisture and management
- **33 plant species**: Grasses, herbs, rushes typical of European dune ecosystems
- Binary presence/absence matrix

This is a classic ecological dataset used for ordination tutorials.

## RSM Construction
Jaccard similarity from co-occurrence (proper kernel for binary data):
```
similarity[i,j] = |sites_both_present| / |sites_either_present|
```
Species that co-occur frequently share habitat preferences.

## Results
SRF rank=8 (CV-selected), RMSE=0.085

**Ecological dimensions discovered**:
- F0: Common grasses (Poa annua, Lolium perenne) — ubiquitous species
- F1: Wet meadow (Juncus bufonius, Ranunculus flammula) — moisture-loving
- F2: Dry grassland herbs (Plantago lanceolata, Achillea millefolium)
- F5: Pioneer species (Hypochaeris radicata)

## Why This Works
- Jaccard similarity from presence/absence is a proper PSD kernel
- SRF non-negativity means species can load on multiple niches (realistic for generalists)
- Factors align with known moisture/disturbance gradients in dune systems
- Unlike PCA, factors are directly interpretable as ecological niches

## Potential Extensions
1. Correlate factors with environmental variables (moisture, management intensity)
2. Compare to NMDS ordination (standard ecological approach)
3. Predict species occurrence from site × factor loadings
