# Double-Centering, Eigendecomposition, and Rank Estimation

**Date:** 2026-03-24
**Context:** Rigorous mathematical analysis of what double-centering does to a similarity matrix and its eigendecomposition, with specific implications for rank estimation via eigenspace coherence (kappa changepoint and PCT). Complements the perturbation-theoretic analysis in `pct_centering_analysis.md` with an algebraic perspective.

---

## 1. Double-Centering: Definition and Basic Properties

Let $S \in \mathbb{R}^{n \times n}$ be a symmetric similarity matrix. Define the **centering matrix**

$$H = I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^\top,$$

where $\mathbf{1} = (1, \ldots, 1)^\top \in \mathbb{R}^n$. The **double-centered matrix** is

$$S_c = H S H.$$

**Algebraic properties of $H$.**

1. **Idempotent:** $H^2 = H$, since $\left(I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top\right)^2 = I - \frac{2}{n}\mathbf{1}\mathbf{1}^\top + \frac{1}{n^2}\mathbf{1}\mathbf{1}^\top\mathbf{1}\mathbf{1}^\top = I - \frac{2}{n}\mathbf{1}\mathbf{1}^\top + \frac{1}{n}\mathbf{1}\mathbf{1}^\top = H$.

2. **Symmetric:** $H^\top = H$.

3. **Rank:** $\mathrm{rank}(H) = n - 1$. The null space of $H$ is $\mathrm{span}(\mathbf{1})$.

4. **Projection:** $H$ is the orthogonal projection onto $\mathbf{1}^\perp = \{x \in \mathbb{R}^n : \mathbf{1}^\top x = 0\}$.

**Entry-wise formula.** The $(i,j)$ entry of $S_c$ is

$$(S_c)_{ij} = S_{ij} - \bar{S}_{i \cdot} - \bar{S}_{\cdot j} + \bar{S}_{\cdot\cdot},$$

where $\bar{S}_{i\cdot} = \frac{1}{n}\sum_j S_{ij}$ is the $i$-th row mean, $\bar{S}_{\cdot j} = \frac{1}{n}\sum_i S_{ij}$ is the $j$-th column mean, and $\bar{S}_{\cdot\cdot} = \frac{1}{n^2}\sum_{i,j} S_{ij}$ is the grand mean.

---

## 2. Effect on the Eigendecomposition

### 2.1. General transformation

Let $S$ have eigendecomposition

$$S = \sum_{k=1}^{n} \lambda_k \, v_k v_k^\top, \quad \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n,$$

where $\{v_1, \ldots, v_n\}$ is an orthonormal eigenbasis. The double-centered matrix is

$$S_c = HSH = \sum_{k=1}^{n} \lambda_k \, (Hv_k)(Hv_k)^\top.$$

The key operation is the projection of each eigenvector:

$$Hv_k = v_k - \frac{1}{n}(\mathbf{1}^\top v_k)\,\mathbf{1} = v_k - \bar{v}_k \, \mathbf{1},$$

where $\bar{v}_k = \frac{1}{n}\sum_{i=1}^n (v_k)_i$ is the mean of the entries of $v_k$. This subtracts the component of $v_k$ along the all-ones direction, projecting it onto $\mathbf{1}^\perp$.

**Three cases arise:**

**(a) $v_k \propto \mathbf{1}$.** Then $Hv_k = 0$, and the entire contribution $\lambda_k v_k v_k^\top$ is annihilated. This eigenvalue vanishes from $S_c$.

**(b) $v_k \perp \mathbf{1}$.** Then $\mathbf{1}^\top v_k = 0$, so $Hv_k = v_k$. The eigenvector and eigenvalue pass through unchanged: $\lambda_k(Hv_k)(Hv_k)^\top = \lambda_k v_k v_k^\top$.

**(c) General $v_k$ (neither parallel nor orthogonal to $\mathbf{1}$).** Then $Hv_k$ is a nonzero vector shorter than $v_k$, with $\|Hv_k\|^2 = 1 - n\bar{v}_k^2$. The rank-one term $\lambda_k(Hv_k)(Hv_k)^\top$ contributes a rescaled, projected version. Importantly, $\{Hv_k\}$ are no longer orthonormal in general, so the individual terms do not directly give the eigendecomposition of $S_c$.

### 2.2. The clean case: first eigenvector approximately constant

In the practically important case where $v_1 \approx \frac{1}{\sqrt{n}}\mathbf{1}$ and all other eigenvectors are approximately orthogonal to $\mathbf{1}$, the structure simplifies dramatically.

**Proposition 1.** Suppose $v_1 = \frac{1}{\sqrt{n}}\mathbf{1}$ exactly, and $v_k \perp \mathbf{1}$ for all $k \geq 2$. Then:

$$S_c = HSH = \sum_{k=2}^{n} \lambda_k \, v_k v_k^\top.$$

The eigendecomposition of $S_c$ is obtained from that of $S$ by deleting the first eigenvalue-eigenvector pair and keeping the rest intact.

*Proof.* Since $Hv_1 = H \cdot \frac{1}{\sqrt{n}}\mathbf{1} = 0$ and $Hv_k = v_k$ for $k \geq 2$, we have $S_c = \sum_{k=2}^n \lambda_k v_k v_k^\top$. The vectors $v_2, \ldots, v_n$ are orthonormal and are eigenvectors of $S_c$ with eigenvalues $\lambda_2, \ldots, \lambda_n$. The eigenvalue $\lambda_1$ is replaced by $0$ (associated with $v_1 = \frac{1}{\sqrt{n}}\mathbf{1}$, which lies in the null space of $H$). $\square$

**Corollary.** Under the conditions of Proposition 1, the rank of $S_c$ equals $\mathrm{rank}(S) - 1$ (provided $\lambda_1 > 0$), and the eigenvalue gaps of $S_c$ between consecutive signal dimensions are identical to those of $S$:

$$\lambda_k(S_c) - \lambda_{k+1}(S_c) = \lambda_{k+1}(S) - \lambda_{k+2}(S), \quad k = 1, \ldots, n-2.$$

### 2.3. Perturbation from exact constancy

When $v_1$ is close to but not exactly $\frac{1}{\sqrt{n}}\mathbf{1}$, write

$$v_1 = \cos\phi \cdot \frac{\mathbf{1}}{\sqrt{n}} + \sin\phi \cdot w,$$

where $w \perp \mathbf{1}$, $\|w\| = 1$, and $\phi$ is the angle between $v_1$ and $\frac{1}{\sqrt{n}}\mathbf{1}$. Then

$$Hv_1 = \sin\phi \cdot w,$$

and the residual contribution of the first eigenvalue to $S_c$ is

$$\lambda_1 (Hv_1)(Hv_1)^\top = \lambda_1 \sin^2\phi \cdot ww^\top.$$

This is a rank-1 perturbation with magnitude $\lambda_1 \sin^2\phi$. When $v_1$ is nearly constant ($\phi \approx 0$), this perturbation is negligible. Quantitatively, if $\phi < \epsilon$, the spectral norm of this residual is at most $\lambda_1 \epsilon^2$, which is small when $\epsilon \ll 1$ even though $\lambda_1$ is large.

Similarly, each $v_k$ for $k \geq 2$ has a small component along $\mathbf{1}$:

$$v_k = \cos\phi_k \cdot w_k + \sin\phi_k \cdot \frac{\mathbf{1}}{\sqrt{n}}, \quad w_k \perp \mathbf{1},$$

where $\phi_k$ is the angle of $v_k$ from $\mathbf{1}^\perp$. Then $Hv_k = \cos\phi_k \cdot w_k$, and the eigenvector is shortened but its direction within $\mathbf{1}^\perp$ is preserved. The resulting eigenvalue shifts are of order $\lambda_k \sin^2\phi_k$, which are small when $v_k$ is nearly orthogonal to $\mathbf{1}$.

---

## 3. When Centering Changes Rank Estimation

### 3.1. The additive constant model

Consider the decomposition

$$S = \mu \, \mathbf{1}\mathbf{1}^\top + L,$$

where $\mu > 0$ is a constant baseline and $L$ is the low-rank signal matrix with $\mathrm{rank}(L) = r$.

**Eigenstructure of $S$ (uncentered).** The matrix $\mu \mathbf{1}\mathbf{1}^\top$ has eigenvalue $n\mu$ with eigenvector $\frac{1}{\sqrt{n}}\mathbf{1}$, and eigenvalue $0$ with multiplicity $n-1$. If $L$ has eigenvalues $\ell_1 \geq \cdots \geq \ell_r > 0$ with eigenvectors $\{u_1, \ldots, u_r\}$, and if these eigenvectors are orthogonal to $\mathbf{1}$ (which holds when $L$ has zero row sums, or approximately when $L$ is a centered signal), then by the orthogonality of the two components:

$$\lambda_1(S) = n\mu, \quad v_1(S) = \frac{1}{\sqrt{n}}\mathbf{1},$$
$$\lambda_{k+1}(S) = \ell_k, \quad v_{k+1}(S) = u_k, \quad k = 1, \ldots, r.$$

The first eigenvalue is $n\mu$, which can be orders of magnitude larger than $\ell_1$ when $\mu$ is large. The eigenvalue ratio is

$$\frac{\lambda_1}{\lambda_2} = \frac{n\mu}{\ell_1}.$$

**Eigenstructure of $S_c$ (centered).** Since $H(\mu\mathbf{1}\mathbf{1}^\top)H = 0$ and $HLH = L$ (assuming $L$ has zero row sums):

$$S_c = L,$$

with eigenvalues $\ell_1, \ldots, \ell_r, 0, \ldots, 0$ and eigenvectors $u_1, \ldots, u_r$.

**Impact on rank estimation.** Any rank estimation method that examines the eigenvalue spectrum of $S$ directly (without centering) sees:

- $\lambda_1 = n\mu \gg \lambda_2 = \ell_1$: a massive first eigenvalue, followed by a sharp drop.
- $\lambda_2, \ldots, \lambda_{r+1}$: the true signal eigenvalues, potentially with gradual decay.
- $\lambda_{r+2}, \ldots, \lambda_n \approx 0$: noise floor.

The dominant gap is at $k = 1$ (between the constant and the signal), not within the signal. A gap-based method like the kappa changepoint applied to $S$ would place the changepoint at $k = 1$, concluding that there is only one "dimension" (the constant). The true $r$-dimensional structure of $L$ is invisible because it is dwarfed by the baseline.

After centering, the spectrum of $S_c = L$ shows only the signal eigenvalues $\ell_1 \geq \cdots \geq \ell_r$, with gaps that reflect the true structure. Rank estimation now operates on the correct object.

### 3.2. What "rank" means before and after centering

This distinction is conceptually important:

- **Rank of $S$:** includes the constant baseline as a "dimension." For $S = \mu\mathbf{1}\mathbf{1}^\top + L$, the rank is $r + 1$ (generically).
- **Rank of $S_c = HSH$:** the rank of the centered signal. This is $r$, the number of structurally informative dimensions.

When we ask "how many latent dimensions underlie the similarity structure?", we are asking about $r$, not $r + 1$. The constant baseline $\mu$ tells us that all pairs have some nonzero average similarity, but this is a trivial, uninformative dimension. Centering removes it, and rank estimation on $S_c$ targets the correct quantity.

---

## 4. The Triplet Similarity Case

### 4.1. Structure of triplet-derived similarity matrices

In the THINGS dataset, each entry $S_{ij}$ represents the empirical probability that objects $i$ and $j$ are judged as most similar in an odd-one-out triplet task:

$$S_{ij} = \frac{\text{number of times } (i,j) \text{ chosen as the similar pair}}{\text{number of triplets in which both } i \text{ and } j \text{ appeared}}.$$

(With Laplace smoothing: $S_{ij} = (\text{count} + \alpha) / (\text{shown} + 2\alpha)$.)

Under **random responding** (no perceptual similarity structure), each of the three pairs in a triplet $(i, j, k)$ is equally likely to be chosen. The random-choice probability is

$$S_{ij}^{\text{null}} = \frac{1}{3} \quad \text{for all } i \neq j.$$

Therefore, the similarity matrix decomposes as

$$S = \frac{1}{3}\mathbf{1}\mathbf{1}^\top + \Delta + D,$$

where $\Delta$ is the deviation matrix encoding actual similarity structure (with entries $\Delta_{ij} = S_{ij} - 1/3$ for $i \neq j$), and $D$ is a diagonal correction (since the diagonal is set to 1, not $1/3$).

### 4.2. Eigenvalue structure

The constant component $\frac{1}{3}\mathbf{1}\mathbf{1}^\top$ contributes a first eigenvalue of

$$\lambda_1^{\text{const}} = \frac{n}{3}.$$

For THINGS with $n = 1854$, this gives $\lambda_1^{\text{const}} \approx 618$. The signal eigenvalues from $\Delta$ are much smaller. Empirically, $\lambda_2(S) \approx 40$ for the full THINGS matrix, giving a ratio

$$\frac{\lambda_1}{\lambda_2} \approx \frac{618}{40} \approx 15.$$

The first eigenvalue dominates by more than an order of magnitude, and its eigenvector is approximately $\frac{1}{\sqrt{n}}\mathbf{1}$ (since all entries are close to $1/3$, the constant structure is nearly uniform across objects).

### 4.3. Effect of centering on THINGS

After double-centering:

$$S_c = HSH \approx H\Delta H + HDH.$$

The constant $\frac{1}{3}\mathbf{1}\mathbf{1}^\top$ is annihilated. The diagonal correction $D$ contributes a centered diagonal matrix, which is a low-rank perturbation (rank $n-1$ but with small spectral norm when diagonal entries are similar).

The eigenvalues of $S_c$ are the signal eigenvalues of $\Delta$, revealing the true dimensionality of the similarity structure without the $1/3$ baseline.

### 4.4. Why centering is appropriate for triplet proportions

The $1/3$ baseline is a consequence of the experimental design (three-alternative forced choice), not of the underlying similarity structure. It is analogous to the intercept in a regression model: always present, uninformative about the relationships between predictors. Removing it via centering is the spectral analog of subtracting the intercept.

**Remark.** The existing codebase sets `center=False` for triplet data in the PCT, with the comment that the "entry mean is informative." This is technically true in the sense that the $1/3$ baseline is a known constant, not noise. However, for rank estimation, the $1/3$ baseline inflates $\lambda_1$ and creates a dominant gap at $k = 1$ that can obscure the signal structure in the kappa changepoint analysis. Centering removes this nuisance component and allows rank estimation to focus on $\Delta$.

---

## 5. When Centering Is Appropriate vs. Inappropriate

The decision to center depends on whether the matrix has a dominant constant component that is uninformative for the rank estimation question.

### 5.1. Centering is appropriate (or necessary)

**Kernel matrices (RBF, cosine).** For a kernel $k(x_i, x_j)$, the diagonal entries $k(x_i, x_i) = c > 0$ (typically $c = 1$), and off-diagonal entries have a constant offset depending on the kernel bandwidth. For Gaussian kernels, $S_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2) \in (0, 1]$. With moderate bandwidth, the matrix is near-constant ($m \approx 0.6$, $\sigma_{\text{entry}} \approx 0.03$). Centering is mandatory: without it, the masking perturbation is proportional to $m$ while signal gaps are proportional to $\sigma_{\text{entry}}$, and the test has no power (see `pct_centering_analysis.md` for the full perturbation analysis).

Centering kernel matrices is standard practice in kernel PCA (Scholkopf et al., 1998) and multidimensional scaling (MDS).

**Triplet proportions.** The $1/3$ baseline creates a dominant constant eigenvalue. Centering removes it and reveals the signal rank. The baseline is a design constant, not structure.

**DNN feature similarities.** Cosine or dot-product similarities between neural network features often have a large positive mean. The mean reflects the average activation level, not the discriminative structure.

### 5.2. Centering is unnecessary (but harmless)

**Correlation matrices.** A correlation matrix $R$ satisfies $R_{ii} = 1$ and $-1 \leq R_{ij} \leq 1$. If the variables are diverse, the mean off-diagonal correlation $\bar{R}$ is close to zero, and the first eigenvector is not approximately constant. Centering removes a small component and barely changes the spectrum. It is harmless but unnecessary.

**Low-mean behavioral RDMs.** Some behavioral similarity matrices (e.g., Mur et al. 92-object, Peterson et al.) have entries already centered around a small value, with the first eigenvector reflecting genuine structure (a "global similarity" dimension) rather than a constant offset. Centering is harmless because $\mu$ is small relative to the signal.

### 5.3. Centering requires care

**PPMI matrices (word association).** Positive Pointwise Mutual Information matrices are sparse and non-negative, with many exact zeros. The entries do not have a constant baseline; zeros represent genuinely unrelated word pairs, not a baseline to be removed. Double-centering creates negative values and fills in the zeros, fundamentally altering the matrix structure. The centering matrix $H$ spreads mass from observed entries into the zero positions, destroying the sparsity that is itself informative.

**Raw count matrices.** Similar to PPMI: the zeros carry meaning (no co-occurrence), and centering creates artificial negative entries.

**Partially observed matrices with many missing entries.** When the observation rate $q \ll 1$, zero-filled entries dominate the row/column means. Double-centering via $HS_0H$ centers the zeros together with the observed entries, introducing a structured bias (see Section 3.3 of `pct_centering_analysis.md`). For sparse observation, observed-entry centering is preferable.

### 5.4. Summary table

| Matrix type | Has constant offset? | $v_1 \approx c\mathbf{1}$? | Center? | Reason |
|:---|:---|:---|:---|:---|
| RBF/cosine kernel | Yes ($k(x,x)$ terms) | Yes | Yes (required) | Offset dominates masking noise |
| Triplet proportions | Yes ($1/3$ baseline) | Approximately | Yes (recommended) | Baseline inflates $\lambda_1$ |
| DNN feature similarity | Often | Often | Yes | Mean activation offset |
| Correlation matrix | No (mean $\approx 0$) | No | Unnecessary | Already centered by construction |
| PPMI | No (sparse, non-negative) | No | No | Destroys sparsity structure |
| Raw counts | No | No | No | Zeros carry meaning |

---

## 6. A Principled Diagnostic

Rather than relying on domain knowledge to decide whether to center, we propose two quantitative diagnostics.

### 6.1. Eigenvalue ratio test

Compute the ratio of the first two eigenvalues:

$$\rho = \frac{\lambda_1}{\lambda_2}.$$

**Decision rule:** If $\rho > \tau_\rho$ (e.g., $\tau_\rho = 5$), the matrix has a dominant first eigenvalue that likely corresponds to a constant offset. Centering will help rank estimation by removing this component and revealing the gap structure among the remaining eigenvalues.

If $\rho < 3$, the first eigenvalue is not dominant, and centering is unnecessary (though still harmless if $v_1$ is approximately constant).

**Calibration.** For $S = \mu\mathbf{1}\mathbf{1}^\top + L$ with $\mathrm{rank}(L) = r$ and eigenvalues of $L$ on the order of $\ell$, the ratio is $\rho = n\mu / \ell$. This exceeds 5 whenever $\mu > 5\ell / n$. For THINGS ($n = 1854$, $\ell_1 \approx 40$), the threshold corresponds to $\mu > 0.11$, well below the actual $\mu = 1/3$.

### 6.2. Eigenvector constancy test

Compute the first eigenvector $v_1$ and measure its deviation from constancy. The normalized all-ones vector has all entries equal to $1/\sqrt{n}$. Define

$$\kappa_1 = \frac{\max_i |(v_1)_i|}{\min_i |(v_1)_i|}.$$

**Decision rule:** If $\kappa_1 \approx 1$ (say $\kappa_1 < 1.5$), then $v_1$ is approximately constant and centering will effectively remove the first eigenvalue. If $\kappa_1 \gg 1$, the first eigenvector encodes non-trivial structure (e.g., a block structure or a gradient), and removing it may discard signal.

**A more robust alternative** is the squared projection onto $\mathbf{1}$:

$$\cos^2\alpha = \left(\frac{\mathbf{1}^\top v_1}{\|\mathbf{1}\| \cdot \|v_1\|}\right)^2 = n \bar{v}_1^2.$$

If $\cos^2\alpha > 0.9$, then $v_1$ is at least 90% aligned with the constant vector, and centering removes at least 90% of the first eigenvalue's contribution. If $\cos^2\alpha < 0.5$, centering would remove genuine structure.

### 6.3. Combined diagnostic

Apply both tests. Centering is recommended when:

1. $\rho = \lambda_1 / \lambda_2 > 5$, **and**
2. $\cos^2\alpha > 0.9$ (first eigenvector is nearly constant).

When condition 1 holds but condition 2 does not, the dominant eigenvalue reflects heterogeneous structure (e.g., a block structure), and centering may be inappropriate. When both conditions fail, centering is unnecessary.

---

## 7. Connection to SRF Factorization

SRF computes the non-negative factorization $S \approx WW^\top$ with $W \geq 0$.

### 7.1. How SRF absorbs the constant baseline

For $S = \mu\mathbf{1}\mathbf{1}^\top + L$, the SRF factorization operates on the raw (uncentered) matrix. The non-negativity constraint $W \geq 0$ means that every column of $W$ must have non-negative entries. The constant component $\mu\mathbf{1}\mathbf{1}^\top$ can be absorbed by adding a positive offset to every column:

$$WW^\top = \mu\mathbf{1}\mathbf{1}^\top + L \quad \Longleftrightarrow \quad W = W_0 + \sqrt{\frac{\mu}{r+1}}\mathbf{1}e^\top + \cdots$$

where $W_0$ captures the signal and the offset ensures every column has a positive floor. In practice, the constant baseline manifests as a positive shift in all columns of $W$: every dimension has nonzero loading for every object.

This is not a problem for the factorization itself. The ADMM algorithm finds the best non-negative approximation regardless of whether a constant baseline is present. However, it does mean that without centering, every column of $W$ is "contaminated" by the baseline, reducing sparsity and interpretability.

### 7.2. Centering for rank estimation, not for factorization

The critical insight is that **centering is needed for rank estimation, not for the factorization itself.** The workflow is:

1. **Rank estimation** (kappa changepoint or PCT): operate on $S_c = HSH$ to determine the signal rank $r$. The centering removes the constant baseline so that the eigenvalue gaps and eigenspace coherence reflect the true signal structure.

2. **SRF factorization**: operate on the raw $S$ (uncentered) with $\mathrm{rank} = r$ (the rank estimated from step 1). The factorization handles the constant component implicitly through the non-negativity constraint.

This separation is natural because the two steps serve different purposes:
- Rank estimation asks: "How many structurally distinct dimensions underlie the similarity?" This is a question about $L$, not about $\mu\mathbf{1}\mathbf{1}^\top + L$.
- Factorization asks: "Given rank $r$, what is the best non-negative decomposition of $S$?" This should use $S$ as given, including the baseline.

### 7.3. Why not center before factorization?

Centering before SRF would be incorrect for two reasons:

1. **Non-negativity violation.** $S_c = HSH$ has both positive and negative entries (centering shifts the mean to zero). The non-negativity constraint $W \geq 0$ and the model $S_c \approx WW^\top$ are incompatible because $WW^\top$ is always PSD with non-negative diagonal, but $S_c$ has negative off-diagonal entries.

2. **Loss of the probabilistic interpretation.** For triplet data, $S_{ij}$ is a probability in $[0, 1]$. The factorization $S \approx WW^\top$ with $W \geq 0$ has a natural interpretation: each column of $W$ is a non-negative "feature" and the similarity between objects is the inner product of their feature vectors. Centering destroys this interpretation.

### 7.4. What centering does to the factorization rank

If $S$ has true signal rank $r$ (i.e., $L$ has rank $r$), then:
- $\mathrm{rank}(S) = r + 1$ (adding the constant component).
- $\mathrm{rank}(S_c) = r$ (constant removed).

Running SRF on $S$ with rank $r$ forces the factorization to approximate a rank-$(r+1)$ matrix with a rank-$r$ factorization. The constant component is partially absorbed into the $r$ columns of $W$ (as a distributed positive offset), and the residual is minimized. This is actually beneficial: the constant is a "nuisance" dimension, and forcing $r$ dimensions means the factorization must prioritize the $r$ structurally informative dimensions over the constant.

Running SRF on $S$ with rank $r + 1$ would allow one column of $W$ to be approximately constant ($w_j \approx c\mathbf{1}$ for some $j$), capturing the baseline explicitly. The remaining $r$ columns capture the signal. This is also valid but wastes one dimension on the uninformative baseline.

---

## 8. Formal Statement: Centering and Rank Estimation

We collect the main results into a single theorem.

**Theorem (Double-centering and signal rank).** Let $S = \mu\mathbf{1}\mathbf{1}^\top + L + E$ where:
- $\mu > 0$ is a constant baseline,
- $L \in \mathbb{R}^{n \times n}$ is symmetric PSD with $\mathrm{rank}(L) = r$ and $L\mathbf{1} = 0$ (zero row sums),
- $E$ is a symmetric perturbation (noise) with $\|E\|_2 \leq \epsilon$.

Let $\delta_k^{(L)} = \lambda_k(L) - \lambda_{k+1}(L)$ denote the eigenvalue gaps of $L$ (with $\lambda_{r+1}(L) = 0$).

**(i) Eigenvalues of $S$.** The eigenvalues of $S$ satisfy
$$\lambda_1(S) = n\mu + O(\epsilon), \quad \lambda_{k+1}(S) = \lambda_k(L) + O(\epsilon), \quad k = 1, \ldots, n-1.$$

**(ii) Eigenvalues of $S_c$.** The eigenvalues of $S_c = HSH$ satisfy
$$\lambda_k(S_c) = \lambda_k(L) + O(\epsilon), \quad k = 1, \ldots, n-1.$$

**(iii) Gap structure.** The largest eigenvalue gap of $S$ is $\lambda_1(S) - \lambda_2(S) = n\mu - \lambda_1(L) + O(\epsilon)$, located at $k = 1$. The gaps within the signal are $\lambda_{k+1}(S) - \lambda_{k+2}(S) = \delta_k^{(L)} + O(\epsilon)$ for $k = 1, \ldots, n-2$. In contrast, the eigenvalue gaps of $S_c$ directly reflect the signal: $\lambda_k(S_c) - \lambda_{k+1}(S_c) = \delta_k^{(L)} + O(\epsilon)$.

**(iv) Kappa changepoint.** Applied to $S$, the changepoint detects the gap at $k = 1$ (constant vs. signal) rather than the gap at the signal-noise boundary $k = r$. Applied to $S_c$, the changepoint detects the gap at $k = r$ (signal vs. noise), which is the target.

**(v) PCT.** Applied to $S$ without centering, the PCT tests whether the first eigenvector (the constant $\frac{1}{\sqrt{n}}\mathbf{1}$) is more coherent than the permutation null. This always passes (the constant is trivially stable), consuming the "first significant dimension" on a nuisance component. Applied to $S_c$, the PCT tests the coherence of signal eigenvectors directly.

*Proof.* Parts (i)-(iii) follow from Weyl's inequality applied to $S = (\mu\mathbf{1}\mathbf{1}^\top + L) + E$ and the orthogonality $L\mathbf{1} = 0$. Part (iv) follows from (iii): the kappa changepoint finds $\arg\max_k(\kappa_{k+1} - \kappa_k)$, which corresponds to the largest spectral gap. Part (v) follows from the Davis-Kahan bound: the constant eigenvector has gap $\delta_1 = n\mu - \ell_1 = O(n\mu)$, giving $I_1(p) \approx 1$ for any masking rate $p$. $\square$

---

## References

- Davis, C. and Kahan, W. M. (1970). The rotation of eigenvectors by a perturbation. III. *SIAM J. Numer. Anal.*, 7(1):1-46.
- Scholkopf, B., Smola, A., and Muller, K.-R. (1998). Nonlinear component analysis as a kernel eigenvalue problem. *Neural Computation*, 10(5):1299-1319.
- Tropp, J. A. (2012). User-friendly tail bounds for sums of random matrices. *Foundations of Computational Mathematics*, 12(4):389-434.
