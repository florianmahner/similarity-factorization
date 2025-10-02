# RSA Comparison

Unified experiment for RSA hypothesis testing and comparison.

## Description

Compares RSA (Representational Similarity Analysis), NMF, and Latent space methods for detecting structure in similarity data. Tests statistical power across different noise levels and experimental designs.

## Modes

### SPOSE Mode
Tests with varying object counts and dimensions using SPOSE embeddings.

```bash
./run rsa_comparison --mode spose --object-counts 50 100 200
```

**Outputs:** `outputs/spose.csv`

### Factorial Mode  
Tests using factorial design with specific factors (animacy, size, curvature, color).

```bash
./run rsa_comparison --mode factorial
```

**Outputs:** `outputs/factorial.csv`

## Arguments

```
--mode {spose,factorial}      Experiment mode (default: spose)
--n-repeats INT               Number of repetitions (default: 100)
--n-permutations INT          Permutation test iterations (default: 1000)
--max-jobs INT                Parallel jobs (default: 140)
--snrs FLOAT [FLOAT ...]      Signal-to-noise ratios to test

# SPOSE mode only:
--object-counts INT [INT ...]  Object counts (default: 50 100 200)
--num-dims INT                 Number of dimensions (default: 10)

# Factorial mode only:
--similarity-metric {linear,cosine,euclidean}  (default: linear)
```

## Outputs

All results saved to `outputs/`:
- `spose.csv` - SPOSE-based RSA comparison
- `factorial.csv` - Factorial design results
- `rsa_power_nobjects_*.pdf` - Power analysis plots
- `block_rsms.pdf` - Factorial RSM visualizations

## Notebook

Analysis and visualization: `notebooks/rsa_comparison.ipynb`

## Common Code

RSA testing functions extracted to `src/analysis/rsa_testing.py`:
- `mantel_test()` - Matrix correlation with permutation testing
- `permutation_test()` - Vector correlation with permutation testing

## Previous Experiments

This combines three previously separate experiments:
- `rsa_comparison` (simple mode)
- `factorial_analysis` (factorial mode)
- `hypothesis_testing` (plots only, now in outputs/)

