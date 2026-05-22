# RSA vs SRF Statistical Testing: Step-by-Step Explanation

## 1. The Setup

We have a **factorial design** with n=48 objects characterized by 4 factors:
- Animacy: 2 levels (animate, inanimate)
- Size: 2 levels (big, small)
- Curvature: 3 levels (curved, straight, mixed)
- Color: 4 levels (red, blue, green, yellow)

This gives us a **ground truth design matrix X** of shape (48 × 11):
```
X = [animacy_animate, animacy_inanimate, size_big, size_small, ...]
```
Each column is a binary indicator (1 if object has that property, 0 otherwise).

The **true similarity** is S_true = X @ X.T (objects are similar if they share properties).

We add noise to get **measured similarity** S_noisy.

---

## 2. The Goal

**Question:** Can we detect which factors are represented in the similarity structure?

**RSA approach:** Build a hypothesis RSM from factor indicators, correlate with measured RSM.

**SRF approach:** Decompose S ≈ W @ W.T, then test if W columns correspond to X columns.

---

## 3. The Problem with SRF Testing

### Step 1: Fit SRF
```python
model = SRF(rank=11)
model.fit(S_noisy)
W = model.w_  # Shape: (48, 11)
```

### Step 2: Match W columns to X columns
We don't know which W column corresponds to which X column (e.g., which W column encodes "red").

**Naive approach:** For each X column, find the W column with highest correlation.

**Problem:** This is **selection bias**. If we search for the best match and then test that match, we inflate false positives.

---

## 4. Why Selection Bias is a Problem

Imagine S_noisy is pure noise (SNR=0). SRF fits W to this noise.

Now we ask: "Is the 'red' indicator represented in W?"

**Naive test:**
1. Compute correlation between X[:,red] and each W column
2. Pick the W column with highest correlation
3. Test if this correlation is significant

**The problem:** Even with random W, one of the 11 columns will have SOME correlation with X[:,red] by chance. By picking the best one, we're cherry-picking.

**Result:** False positive rate >> 5% (we saw 90%+ in initial tests!)

---

## 5. The Solution: Leave-One-Out Cross-Validation

### The Idea
Don't use the same data for selection AND testing.

### Step-by-Step LOO Procedure

For testing whether "red" is represented:

```
For each object i = 1, ..., 48:
    1. Leave object i out
    2. Using remaining 47 objects, find which W column best correlates with X[:,red]
    3. Record which column was selected (might differ across folds!)
    4. For object i, use the selected W column's value as the "prediction"

Result: A vector of 48 LOO predictions
```

**Key insight:** Object i was never used to select the W column that predicts it.

### The Test Statistic
Correlation between X[:,red] and the LOO prediction vector.

---

## 6. The Permutation Test (Critical!)

### Naive Permutation (WRONG)
```
observed_corr = corr(X[:,red], LOO_predictions)
for p in 1..1000:
    X_perm = permute(X[:,red])
    null[p] = corr(X_perm, LOO_predictions)  # WRONG!
p_value = fraction of null >= observed_corr
```

**Why wrong?** The LOO_predictions were computed using the original X[:,red]. The null doesn't account for the selection process.

### Correct Permutation
```
observed_corr = corr(X[:,red], LOO_predictions)
for p in 1..1000:
    X_perm = permute(X[:,red])
    LOO_predictions_perm = do_full_LOO(X_perm, W)  # Re-do LOO!
    null[p] = corr(X_perm, LOO_predictions_perm)
p_value = fraction of null >= observed_corr
```

**Why correct?** The null does exactly the same selection process as the observed. This is the fair comparison.

**Cost:** Much slower (n_perm × n × k operations instead of n_perm × 1).

---

## 7. From Level P-values to Factor P-values

### The Problem
We now have 11 p-values (one per level):
- p_animate, p_inanimate (animacy factor)
- p_big, p_small (size factor)
- p_curved, p_straight, p_mixed (curvature factor)
- p_red, p_blue, p_green, p_yellow (color factor)

**Question:** How do we get a single p-value for "is color represented?"

### Solution: Fisher's Method

For the color factor with 4 levels:
```
χ² = -2 × (log(p_red) + log(p_blue) + log(p_green) + log(p_yellow))
```

Under H0 (no color effect), χ² follows a χ²(8) distribution (df = 2 × number of p-values).

**Interpretation:** Fisher's method tests "is at least one color level represented?"

### Why Fisher?
We tested three methods empirically:

| Method | SNR=0 (FPR) | SNR=0.3 Power |
|--------|-------------|---------------|
| Fisher | **6%** ✓ | 74% (all factors) |
| Simes | 0% ✗ | 78% |
| Bonferroni | 0% ✗ | 76% |

- Fisher is well-calibrated (~5% false positive rate)
- Simes and Bonferroni are too conservative (0% FPR = underpowered)

---

## 8. The Complete SRF Testing Pipeline

### Per-Level Testing
```
For each level l in {animate, inanimate, big, small, ...}:
    1. Compute LOO predictions for X[:,l] vs W
    2. observed_corr = corr(X[:,l], LOO_predictions)
    3. For p in 1..n_perm:
        X_perm = permute(X[:,l])
        LOO_perm = do_full_LOO(X_perm, W)
        null[p] = corr(X_perm, LOO_perm)
    4. p_l = (count(null >= observed_corr) + 1) / (n_perm + 1)
```

### Per-Factor Testing (using Fisher)
```
For each factor f in {animacy, size, curvature, color}:
    levels_of_f = get_levels(f)  # e.g., [red, blue, green, yellow] for color
    p_values = [p_l for l in levels_of_f]
    χ² = -2 × sum(log(p_values))
    p_f = 1 - chi2_cdf(χ², df=2*len(levels_of_f))
```

### Multiple Comparison Correction
```
# Factor-level (for comparison with RSA)
p_factors = [p_animacy, p_size, p_curvature, p_color]
p_factors_corrected = FDR(p_factors)

# Level-level (for dissociation - SRF only)
p_levels = [p_animate, p_inanimate, ..., p_yellow]  # 11 values
p_levels_corrected = FDR(p_levels)
```

---

## 9. Comparison with RSA

### RSA Testing Pipeline
```
For each factor f in {animacy, size, curvature, color}:
    1. Build hypothesis RSM: H_f = X_f @ X_f.T
       where X_f contains only columns for factor f
    2. Mantel test: correlate H_f with S_noisy
    3. Permutation null (respecting factorial structure)
    4. Get p_f

p_factors = [p_animacy, p_size, p_curvature, p_color]
p_factors_corrected = FDR(p_factors)
```

### Key Difference

| Aspect | RSA | SRF |
|--------|-----|-----|
| Tests | 4 (one per factor) | 11 (one per level) → combined to 4 |
| Can dissociate levels? | **No** | **Yes** |
| What it tests | "Does color affect similarity?" | "Is red/blue/green/yellow represented?" |

**RSA limitation:** If RSA says "color is significant," you don't know if it's because of red, blue, green, yellow, or all of them.

**SRF advantage:** SRF can say "red and blue are significant, but green and yellow are not."

---

## 10. Summary

### The Statistical Framework

1. **Avoid selection bias:** Use LOO cross-validation
2. **Fair null distribution:** Re-do LOO for each permutation
3. **Combine p-values:** Use Fisher's method (well-calibrated)
4. **Multiple comparisons:** FDR correction

### What We Report

1. **Factor-level power** (comparable to RSA):
   - P(factor significant after FDR)
   - Both RSA and SRF have 4 tests

2. **Level-level power** (SRF only):
   - P(level significant after FDR)
   - Shows dissociation ability

### The Story

- SRF is at least as powerful as RSA for detecting factors
- SRF additionally provides dissociation (which levels drive the effect)
- This is a qualitative advantage of embedding-based analysis

---

## 11. Empirical Validation

### Setup
- 48 objects, 11 dimensions, 4 factors
- 50 repeats, 100 permutations
- Tested at SNR=0 (pure noise) and SNR=0.3

### Results at SNR=0 (Calibration Check)
- Fisher: **6% false positive rate** (expected: 5%) ✓
- Without proper LOO null: 90%+ false positive rate ✗

### Results at SNR=0.3 (Power)
- Fisher: 100% power to detect any factor, 74% to detect all factors
- Per-factor power: 88-96%

### Conclusion
The framework is well-calibrated and powerful when implemented correctly.
