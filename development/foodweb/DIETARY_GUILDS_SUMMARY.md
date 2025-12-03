# Dietary Guild Analysis with Symmetric NMF

## The Key Insight

**Problem**: Symmetric NMF on raw food webs doesn't work because food webs are inherently directed (predator → prey).

**Solution**: Transform the directed adjacency matrix into a **similarity matrix** that captures meaningful ecological relationships.

## Three Approaches Tested

### 1. Dietary Guilds (AA^T) ✅ **BEST**
- **Question**: "Who eats like whom?"
- **Method**: Cosine similarity between diet vectors (rows of A)
- **Result**: 13.3% non-zero entries (sparse, interpretable)
- **Interpretation**: Species cluster by dietary overlap

### 2. Prey Vulnerability Groups (A^T A)
- **Question**: "Who gets eaten like whom?"
- **Method**: Cosine similarity between predator vectors (columns of A)
- **Result**: 61.4% non-zero (too dense)
- **Interpretation**: Species sharing predators

### 3. Structural Compartments (A + A^T)
- **Question**: "Who interacts with whom?"
- **Method**: Symmetrized adjacency
- **Result**: 15.7% non-zero (binary, less informative)

## Results: Dietary Guilds (k=10)

### Clear Structure Discovered!

**Guild 0**: Generalist predators (20-47 prey species)
- Spiders: Tetragnatha, Larinioides, Araneidae
- Damselflies: Coenagrion, Zygoptera
- Body mass: 0.0001g - 0.3g

**Guild 3**: Specialist predators (3-6 prey species)
- Tiny beetles: Edaphus, Anotylus (0.00007g)
- Mites: Acari pred (0.00003g)
- **Highly selective diets**

**Guild 9**: Top predators (19-62 prey species)
- Large spiders: Pisaura (0.07g, 62 prey!)
- Tetragnatha extensa (0.037g, 19 prey)
- **ρ = +0.328 with body mass (p < 0.00001)**

### Validation

**Body Mass Correlations** (independent validation):
- Guild 0: ρ = +0.203 (p < 0.01)
- Guild 4: ρ = +0.172 (p < 0.05)
- Guild 9: ρ = +0.328 (p < 0.00001) ⭐

**3/10 guilds significantly correlate with body size!**

## Why This Works

1. **Biologically meaningful**: Dietary overlap is a real ecological concept
2. **Independent validation**: Body mass not used in algorithm
3. **Interpretable factors**: Each guild has clear ecological meaning
4. **Sparse representation**: 13.3% non-zero vs 16% for raw network

## Key Takeaway

**Don't use symmetric NMF on raw directed data!**

Instead, transform directed relationships into symmetric similarities:
- Food webs → Dietary similarity (AA^T)
- Social networks → Co-occurrence (AA^T)
- Citation networks → Co-citation (A^T A)

This is analogous to how PPI networks use protein-protein co-complex membership for validation!
