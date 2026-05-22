# Why the Permutation Coherence Test Needs Double-Centering

**Date:** 2026-03-19
**Context:** The Permutation Coherence Test (PCT) fails on near-constant kernel matrices (e.g., Gaussian kernel with entries ~0.6, std ~0.03). Double-centering fixes this empirically. This document provides the theoretical analysis of why.

---

## Step 1: Why Iproj Fails for Near-Constant Matrices

### 1.1. Setup

Let $S \in \mathbb{R}^{n \times n}$ be a symmetric PSD similarity matrix. Decompose it as

$$S = m J + \sigma X,$$

where $J = \mathbf{1}\mathbf{1}^\top$ is the rank-1 all-ones matrix, $m > 0$ is the mean off-diagonal entry, and $X$ is a zero-mean signal matrix with entries of order $O(1)$ (so that $\sigma$ controls the signal scale). For a near-constant kernel, $m \gg \sigma$.

The eigenstructure of $S$ has a dominant eigenvector $u_1 \propto \mathbf{1}$ with eigenvalue $\lambda_1 \approx mn + O(\sigma n)$, and signal eigenvectors $u_2, \ldots, u_{r+1}$ with eigenvalues $\lambda_k = O(\sigma n)$ for $k \geq 2$. The eigenvalue gaps between consecutive signal dimensions are $\delta_k = \lambda_k - \lambda_{k+1} = O(\sigma n)$ for $k \geq 2$, while the gap between the mean component and the first signal component is $\delta_1 = \lambda_1 - \lambda_2 = O(mn)$.

### 1.2. Masking Perturbation

Under Bernoulli($p$) masking with $1/p$ rescaling, the masked matrix is

$$\widetilde{S}(p) = D_S + \frac{1}{p}(S - D_S) \circ M(p),$$

where $M(p)$ is a symmetric Bernoulli mask on off-diagonal entries and $D_S$ is the diagonal of $S$. The perturbation is

$$E = \widetilde{S}(p) - S = (S - D_S) \circ \left(\frac{M(p)}{p} - \mathbf{1}\mathbf{1}^\top + I\right).$$

For off-diagonal entries $(i,j)$ with $i \neq j$:

$$E_{ij} = S_{ij}\left(\frac{M_{ij}}{p} - 1\right),$$

and $E_{ii} = 0$ (diagonal is preserved).

Each off-diagonal $E_{ij}$ is a mean-zero random variable with

$$\mathrm{Var}(E_{ij}) = S_{ij}^2 \cdot \frac{1-p}{p}.$$

### 1.3. Spectral Norm of the Perturbation via Matrix Bernstein

Write $E = \sum_{i < j} E^{(ij)}$ where $E^{(ij)}$ is the symmetric matrix with $E_{ij}$ in positions $(i,j)$ and $(j,i)$ and zero elsewhere. These are independent, mean-zero, and symmetric.

**Almost-sure bound.** Each summand satisfies $\|E^{(ij)}\|_2 = |E_{ij}| \leq |S_{ij}| \cdot \max(1/p - 1, 1) \leq S_{\max}/p$, where $S_{\max} = \max_{i \neq j} |S_{ij}|$. For near-constant matrices, $S_{\max} \approx m$, so the per-summand bound is $L = m/p$.

**Variance parameter.** The matrix variance statistic is

$$\sigma_B^2 = \left\|\sum_{i < j} \mathbb{E}[(E^{(ij)})^2]\right\|_2.$$

For $E^{(ij)}$ supported on positions $(i,j)$ and $(j,i)$:

$$(E^{(ij)})^2 = E_{ij}^2 (e_i e_i^\top + e_j e_j^\top),$$

so

$$\sum_{i < j} \mathbb{E}[(E^{(ij)})^2] = \frac{1-p}{p} \sum_{i < j} S_{ij}^2 (e_i e_i^\top + e_j e_j^\top) = \frac{1-p}{p} \cdot \mathrm{diag}\left(\sum_{j \neq i} S_{ij}^2\right).$$

The spectral norm of this diagonal matrix is

$$\sigma_B^2 = \frac{1-p}{p} \cdot \max_i \sum_{j \neq i} S_{ij}^2.$$

For the near-constant matrix $S \approx mJ$, we have $S_{ij}^2 \approx m^2$ for all $i \neq j$, giving

$$\sigma_B^2 \approx \frac{(1-p)}{p} \cdot (n-1) m^2.$$

**Matrix Bernstein inequality** (Tropp, 2012). For independent, mean-zero, symmetric random matrices $X_1, \ldots, X_N$ with $\|X_i\|_2 \leq L$ almost surely:

$$\Pr\left(\left\|\sum_i X_i\right\|_2 \geq t\right) \leq 2n \exp\left(\frac{-t^2/2}{\sigma_B^2 + Lt/3}\right).$$

Setting the right side to a small probability $\delta$ and solving for $t$, the spectral norm concentrates around

$$\|E\|_2 \lesssim \sigma_B + L \log(n/\delta) \approx m\sqrt{\frac{n(1-p)}{p}} + \frac{m}{p}\log n.$$

For typical $p$ bounded away from 0 (say $p \geq 0.1$), the dominant term is

$$\boxed{\|E\|_2 = O\!\left(m\sqrt{\frac{n}{p}}\right).}$$

### 1.4. Applying Davis-Kahan

The Davis-Kahan $\sin\Theta$ theorem states: for symmetric $A$ and $\hat{A} = A + E$, the angle $\theta_k$ between the $k$-th eigenvector of $A$ and the top-$k$ eigensubspace of $\hat{A}$ satisfies

$$\sin(\theta_k) \leq \frac{\|E\|_2}{\delta_k},$$

where $\delta_k$ is the eigenvalue gap separating the top-$k$ eigenvalues from the rest. For the projection coherence $I_k(p) = \|P_k(\widetilde{S}(p)) u_k(S)\|^2$, which measures how well the $k$-th reference eigenvector is captured by the top-$k$ eigensubspace of $\widetilde{S}(p)$, the relevant gap is $\delta_k = \lambda_k - \lambda_{k+1}$ (separating what is in the subspace from what is not).

The angle $\theta_k$ between $u_k(S)$ and the top-$k$ subspace of $\widetilde{S}(p)$ satisfies $I_k(p) = \cos^2(\theta_k)$. By Davis-Kahan, the largest canonical angle $\theta_{\max}$ between the top-$k$ subspaces of $S$ and $\widetilde{S}(p)$ satisfies $\sin(\theta_{\max}) \leq \|E\|_2 / \delta_k$. Since $u_k(S)$ lies in the top-$k$ subspace of $S$, the angle from $u_k$ to the top-$k$ subspace of $\widetilde{S}(p)$ is at most $\theta_{\max}$, giving

$$I_k(p) \geq \cos^2(\theta_{\max}) \geq 1 - \frac{\|E\|_2^2}{\delta_k^2}.$$

**For the mean eigenvector ($k=1$):** The gap is $\delta_1 = \lambda_1 - \lambda_2 = O(mn)$, so

$$\sin(\theta_1) \leq \frac{O(m\sqrt{n/p})}{O(mn)} = O\!\left(\frac{1}{\sqrt{np}}\right).$$

This is small for $n$ large: the mean eigenvector is always well-recovered. Hence $I_1(p) \approx 1$.

**For signal eigenvectors ($k \geq 2$):** The gap is $\delta_k = O(\sigma n)$ (between signal eigenvalues), so

$$\sin(\theta_k) \leq \frac{O(m\sqrt{n/p})}{O(\sigma n)} = O\!\left(\frac{m}{\sigma\sqrt{np}}\right).$$

For this to be small (i.e., for $I_k(p)$ to be close to 1), we need

$$\frac{m}{\sigma} \ll \sqrt{np}.$$

**The failure condition:** When $m/\sigma$ is large (near-constant kernel), this bound is uninformative unless $n$ is extremely large. For example, with $m = 0.6$, $\sigma = 0.03$, we have $m/\sigma = 20$. With $n = 200$ and $p = 0.5$, $\sqrt{np} = 10$, and the bound gives $\sin(\theta_k) \leq 2$, which is vacuous. The perturbation from masking the large mean component destroys all information about the signal eigensubspace.

**Intuition:** Masking introduces noise proportional to entry magnitudes. For near-constant matrices, entry magnitudes are dominated by $m$, but the signal is encoded in deviations of order $\sigma \ll m$. The masking noise swamps the signal gaps, causing $I_k(p) \approx 0$ for $k \geq 2$.

### 1.5. Why the Permutation Null Also Has Low Iproj

Under the null (permuted entries), the permuted matrix $S_\pi$ has the same entry magnitudes (hence the same $m$) and the same masking noise. Its eigenvectors are random, but they are also destroyed by masking at the same rate. Both observed and null Iproj values are near zero for $k \geq 2$. The test has no power because $I_k^{\mathrm{obs}}(p) \approx I_k^{\mathrm{null}}(p) \approx 0$.

---

## Step 2: What Double-Centering Does Mathematically

### 2.1. Definition and Effect on the Decomposition

Double-centering applies the centering matrix $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ on both sides:

$$S_c = H S H.$$

Since $H$ is an orthogonal projection onto the subspace orthogonal to $\mathbf{1}$ (i.e., $H\mathbf{1} = 0$ and $H^2 = H$), double-centering removes the mean component:

For $S = mJ + \sigma X$:

- $HJH = H \mathbf{1}\mathbf{1}^\top H = 0$, since $H\mathbf{1} = 0$.
- $H X H = X_c$, the double-centered signal matrix.

Therefore:

$$S_c = \sigma X_c.$$

The mean component $mJ$ is completely annihilated. The entries of $S_c$ are of order $O(\sigma)$, not $O(m)$.

### 2.2. Eigenstructure After Centering

The eigenvalues of $S_c$ are the signal eigenvalues of $S$ (those corresponding to eigenvectors orthogonal to $\mathbf{1}$), with $\lambda_1(S_c) = O(\sigma n)$. The mean eigenvalue $mn$ is removed.

The eigenvectors of $S_c$ are the signal eigenvectors of $S$ (projected and re-indexed). The rank-$r$ signal structure is preserved: if $S$ had $r$ significant signal dimensions beyond the mean, $S_c$ has $r$ significant dimensions.

### 2.3. Masking Perturbation After Centering

After centering, the entries are $O(\sigma)$, so the masking perturbation becomes:

$$E_c = \widetilde{S}_c(p) - S_c,$$

where $\widetilde{S}_c$ is the centered version of the masked matrix. The off-diagonal entries satisfy

$$\mathrm{Var}((E_c)_{ij}) = (S_c)_{ij}^2 \cdot \frac{1-p}{p} = O\!\left(\sigma^2 \cdot \frac{1-p}{p}\right).$$

By the same matrix Bernstein analysis as in Section 1.3:

$$\boxed{\|E_c\|_2 = O\!\left(\sigma\sqrt{\frac{n}{p}}\right).}$$

### 2.4. Davis-Kahan After Centering

The signal eigenvalue gaps remain $\delta_k = O(\sigma n)$ (centering shifts eigenvalues by removing the mean, but does not change the gaps between signal eigenvalues). Therefore:

$$\sin(\theta_k) \leq \frac{\|E_c\|_2}{\delta_k} = \frac{O(\sigma\sqrt{n/p})}{O(\sigma n)} = O\!\left(\frac{1}{\sqrt{np}}\right).$$

**The ratio $m/\sigma$ has canceled out.** The perturbation bound is now independent of the mean level $m$. For $n = 200$, $p = 0.5$: $\sin(\theta_k) = O(1/10)$, giving $I_k(p) \approx 0.99$. This is why centering restores power.

### 2.5. Summary of the Mechanism

| Quantity | Before centering | After centering |
|----------|-----------------|-----------------|
| Entry magnitudes | $O(m)$ | $O(\sigma)$ |
| $\|E\|_2$ | $O(m\sqrt{n/p})$ | $O(\sigma\sqrt{n/p})$ |
| Signal gap $\delta_k$ | $O(\sigma n)$ | $O(\sigma n)$ |
| $\sin(\theta_k)$ bound | $O(m/(\sigma\sqrt{np}))$ | $O(1/\sqrt{np})$ |
| Iproj informative? | Only when $m/\sigma \ll \sqrt{np}$ | Always (for $np$ large) |

Centering ensures that the masking perturbation scales with the signal variance, not the total variance.

---

## Step 3: Is Centering Always Correct?

### 3.1. Case (a): Fully Observed PSD Matrix with High Mean

Examples: DINOv2 kernel similarities, Gaussian kernel matrices.

Double-centering is the standard preprocessing for kernel PCA (Scholkopf et al., 1998). It removes the constant offset that contributes a rank-1 component to the kernel matrix but carries no discriminative information. For PCT, centering is not only safe but necessary: without it, the dominant mean component creates masking noise that overwhelms signal gaps (as shown in Step 1).

**Conclusion:** centering is required and well-understood.

### 3.2. Case (b): Fully Observed PSD Matrix with Low Mean

Examples: Mur et al. (2013) 92-object RDM, Peterson et al. (2018) behavioral similarity.

When $m$ is already small relative to $\sigma$ (i.e., the matrix is already "centered" or nearly so), centering has minimal effect:

- $HJH = 0$ removes a small component.
- $HXH \approx X$ (centering barely changes entries when the mean is small).
- The masking perturbation is $O(\sigma\sqrt{n/p})$ with or without centering.

**Conclusion:** centering is harmless; the bound improves slightly but was already informative.

### 3.3. Case (c): Partially Observed Matrix (NaN Filled with Zero)

This is the subtle case. Let $\Omega$ be the set of observed off-diagonal pairs and $q = |\Omega|/\binom{n}{2}$ be the observation rate. The zero-filled matrix is

$$S_0[i,j] = \begin{cases} S_{ij} & (i,j) \in \Omega, \\ 0 & (i,j) \notin \Omega. \end{cases}$$

After double-centering:

$$(S_c)_{ij} = S_0[i,j] - \bar{S}_{0,i\cdot} - \bar{S}_{0,\cdot j} + \bar{S}_{0,\cdot\cdot},$$

where $\bar{S}_{0,i\cdot} = \frac{1}{n}\sum_j S_0[i,j]$ is the $i$-th row mean of $S_0$ (including the zeros from missing entries).

**Key concern:** The zeros from missing entries are also centered. A zero entry $(i,j) \notin \Omega$ becomes $-\bar{S}_{0,i\cdot} - \bar{S}_{0,\cdot j} + \bar{S}_{0,\cdot\cdot}$, which is generally nonzero (and negative when the matrix has positive mean). Does this bias the test?

**Analysis of the PCT comparison.** In the PCT, the permutation null permutes observed entries among observed positions. Both $S_0$ and $S_\pi$ have:

- Identical sparsity pattern (same $\Omega$).
- Identical zeros at the same positions.
- Same set of observed values (just reassigned).

After centering, both $S_{0,c}$ and $S_{\pi,c}$ have:

- Different centered values at observed positions (because permutation changes row/column means).
- Different values at missing positions (because the fill-in $-\bar{S}_{0,i\cdot} - \bar{S}_{0,\cdot j} + \bar{S}_{0,\cdot\cdot}$ depends on the specific value assignment).

The comparison is still valid in the following sense: the test asks "does the eigenstructure of $S_{0,c}$ produce more coherence under masking than the eigenstructure of $S_{\pi,c}$?" Both matrices undergo the same centering operator applied to the same sparsity pattern. The null distribution correctly reflects the behavior of structure-free matrices under this specific centering.

**However, centering the zeros does distort the eigenvalue structure.** The row means $\bar{S}_{0,i\cdot}$ are biased downward by the zeros: if object $i$ has many missing entries, its row mean is artificially low, and centering introduces heterogeneous shifts across rows. This does not break the validity of the test (Type I error is still controlled, because the null respects the same distortion), but it can reduce power:

1. **Heterogeneous centering shifts** add noise to the eigenstructure beyond what would be present if centering used only observed entries.
2. **The centered zeros create a structured perturbation** (not random) that can interact with the signal eigenspace.

**Quantitative assessment.** Let $q$ be the observation rate. The row mean is

$$\bar{S}_{0,i\cdot} = \frac{1}{n}\sum_{j:(i,j) \in \Omega} S_{ij} \approx q \cdot \bar{S}_{i\cdot}^{\mathrm{full}},$$

where $\bar{S}_{i\cdot}^{\mathrm{full}}$ is the row mean of the fully observed matrix. After centering, the missing-entry fill-in values are of order $q \cdot m$ (since $\bar{S}_{0,i\cdot} \approx qm$ and $\bar{S}_{0,\cdot\cdot} \approx qm$). The centered zeros become approximately $-qm - qm + qm = -qm$. This is a rank-1-like perturbation of magnitude $O(qm)$ supported on the missing positions.

For the test to remain powerful, this perturbation must be small relative to the signal. The fraction of missing positions is $1 - q$, so the Frobenius norm of this perturbation is $O(qm \cdot n\sqrt{1-q})$, and its spectral norm is at most $O(qm \cdot n(1-q))$ (since the missing positions form a roughly uniform pattern). This is problematic when both $m$ is large AND $q$ is moderate (say $q = 0.5$), because the centered-zero perturbation has spectral norm $O(mn)$, comparable to the signal eigenvalues $O(\sigma n)$.

**Mitigation.** For partially observed matrices with high mean, one should ideally:

1. Compute row/column means using only observed entries: $\bar{S}_{i\cdot}^{\mathrm{obs}} = \frac{1}{|\{j : (i,j) \in \Omega\}|}\sum_{j:(i,j)\in\Omega} S_{ij}$.
2. Center observed entries using these means.
3. Keep missing entries as zero (or, equivalently, set them to the centered value of zero, which is now correct since the mean has been removed from observed entries only).

This "observed-entry centering" avoids centering the zeros and is the correct analog of double-centering for incomplete matrices.

**However, in the current PCT implementation**, the centering is applied after zero-filling, which corresponds to the simpler but slightly biased approach. The bias is small when either (i) $m$ is not too large relative to $\sigma$, or (ii) the observation rate $q$ is close to 1. For the primary use case (THINGS behavioral similarity with $q > 0.99$), the bias is negligible.

**Conclusion for case (c):** Centering after zero-filling is approximately correct when $q$ is near 1. For sparse observation ($q \ll 1$), observed-entry centering is preferable.

---

## Step 4: Formal Theoretical Framework

### 4.1. Notation

- $S \in \mathbb{R}^{n \times n}$: symmetric PSD similarity matrix.
- $\Omega$: observed off-diagonal pairs, $q = |\Omega|/\binom{n}{2}$.
- $S_0$: zero-filled matrix ($S_{ij}$ for $(i,j) \in \Omega$, zero otherwise).
- $S_c = H S_0 H$: double-centered zero-filled matrix.
- $p \in (0, 1]$: masking fraction.
- $\widetilde{S}_c(p)$: Bernoulli($p$)-masked, $1/p$-rescaled version of $S_c$.
- $E_c(p) = \widetilde{S}_c(p) - S_c$: masking perturbation.
- $\lambda_1 \geq \cdots \geq \lambda_n$: eigenvalues of $S_c$.
- $\delta_k = \lambda_k - \lambda_{k+1}$: eigenvalue gap separating the top-$k$ subspace from the rest.
- $I_k(p) = \|P_k(\widetilde{S}_c(p))\, u_k(S_c)\|^2$: projection coherence.
- $\sigma_{\mathrm{entry}}^2 = \max_i \sum_{j \neq i} (S_c)_{ij}^2$: maximum row sum of squared centered entries.

### 4.2. Perturbation Bound

**Lemma (Masking perturbation for centered matrix).** Let $S_c$ be double-centered with $\|(S_c)_{ij}\|_\infty \leq s_{\max}$. Under Bernoulli($p$) masking with $1/p$ rescaling (diagonal preserved), the perturbation $E_c(p) = \widetilde{S}_c(p) - S_c$ satisfies, with probability at least $1 - \eta$:

$$\|E_c(p)\|_2 \leq \sqrt{\frac{2(1-p)}{p} \sigma_{\mathrm{entry}}^2 \log\frac{2n}{\eta}} + \frac{2 s_{\max}}{3p} \log\frac{2n}{\eta}.$$

*Proof sketch.* Write $E_c = \sum_{i < j} E_c^{(ij)}$ where $E_c^{(ij)}$ is a symmetric rank-2 matrix supported on positions $(i,j)$ and $(j,i)$. These are independent, mean-zero, with $\|E_c^{(ij)}\|_2 \leq |(S_c)_{ij}|/p \leq s_{\max}/p$. The matrix variance statistic is $\|\sum_{i<j} \mathbb{E}[(E_c^{(ij)})^2]\|_2 = \frac{1-p}{p} \sigma_{\mathrm{entry}}^2$. Apply the matrix Bernstein inequality (Tropp, 2012). $\square$

**Corollary.** For $S_c$ with entries of order $O(\sigma)$ and $n$ rows each contributing $O(n)$ terms:

$$\sigma_{\mathrm{entry}}^2 = O(n\sigma^2), \quad s_{\max} = O(\sigma).$$

Therefore $\|E_c(p)\|_2 = O(\sigma\sqrt{n \log n / p})$ with high probability.

### 4.3. Power Condition

**Theorem (informal).** Consider the PCT with double-centering applied to a similarity matrix $S$ with observation rate $q$ close to 1. Suppose the centered matrix $S_c = HSH$ (or $HS_0H$ for partially observed) has eigenvalues $\lambda_1 \geq \cdots \geq \lambda_n$ and eigenvalue gap $\delta_k$ at dimension $k$. The PCT at significance level $\alpha$ with $J$ null replicates:

**(Type I error control)** Under the null hypothesis (no structure beyond what a random reassignment of entries would produce), $\Pr(p_k < \alpha) \leq \alpha$ for each dimension $k$. This holds regardless of centering, because the permutation null respects the centering operation.

**(Non-trivial power)** The test has power approaching 1 for dimension $k$ whenever

$$\frac{\delta_k}{\sigma_{\mathrm{entry}} \sqrt{1/p}} \gg \sqrt{\log n},$$

or equivalently,

$$\delta_k \gg \sigma_{\mathrm{entry}} \cdot \sqrt{\frac{\log n}{p}}.$$

*Interpretation.* The left side is the eigenvalue gap of the centered matrix (measuring signal separation). The right side is the spectral norm of the masking perturbation. When the gap dominates the perturbation, $I_k(p) \approx 1$ while the null (which destroys gaps by permutation) has $I_k^{\mathrm{null}}(p)$ well below 1.

### 4.4. Two Regimes

**High signal-to-noise (detectable).** When

$$\frac{\delta_k}{\sigma_{\mathrm{entry}}} \gg \sqrt{\frac{\log n}{p}},$$

the centered masking perturbation is small relative to the eigenvalue gap. The Iproj for dimension $k$ is close to 1, well above the null. The PCT detects this dimension with high probability.

After centering, $\sigma_{\mathrm{entry}}^2 \approx n\sigma^2$ and $\delta_k = O(\sigma n)$, so the condition becomes $\sqrt{n} \gg \sqrt{\log n / p}$, which is satisfied for any fixed $p > 0$ and large $n$.

Without centering, $\sigma_{\mathrm{entry}}^2 \approx nm^2$ while $\delta_k$ remains $O(\sigma n)$, so the condition becomes $\sigma\sqrt{n} \gg m\sqrt{\log n / p}$, i.e., $m/\sigma \ll \sqrt{np / \log n}$. This fails when $m \gg \sigma$.

**Low signal-to-noise (undetectable).** When $\delta_k / \sigma_{\mathrm{entry}} \lesssim \sqrt{\log n / p}$, the masking perturbation is comparable to or larger than the gap. The signal eigenvector is not stably recovered under masking, so $I_k(p)$ is low and indistinguishable from the null. The test correctly fails to reject (low power, not inflated error).

---

## Step 5: Is There a Better Approach Than Ad-Hoc Centering?

### 5.1. Alternative (a): Normalize the Masking Perturbation by Entry Variance

Instead of centering $S$, one could define a normalized Iproj:

$$\hat{I}_k(p) = \frac{I_k(p)}{\mathbb{E}[I_k(p) \mid H_0]},$$

or standardize: $Z_k(p) = (I_k(p) - \mu_{\mathrm{null}}) / \sigma_{\mathrm{null}}$.

**Problem:** This is essentially what the permutation test already does (comparing observed Iproj to the null distribution). The issue is not with the comparison but with the signal itself: when $I_k^{\mathrm{obs}}(p) \approx 0$, no normalization of the null can create separation. The signal eigenvectors are literally destroyed by masking noise, so there is no signal to detect.

**Verdict:** Does not address the root cause.

### 5.2. Alternative (b): Shift-Invariant Iproj Definition

Define Iproj using centered eigenvectors: project onto the subspace orthogonal to $\mathbf{1}$ before computing angles. Formally, replace $u_k(S)$ with $Hu_k(S)$ (or equivalently, work with the eigendecomposition of $HSH$).

This is mathematically equivalent to double-centering the matrix, because $Hu_k(S) = u_k(HSH)$ for eigenvectors orthogonal to $\mathbf{1}$ (which are all signal eigenvectors for $k \geq 2$). The mean eigenvector $u_1 \propto \mathbf{1}$ is projected to zero, effectively removing it.

**Verdict:** Equivalent to double-centering.

### 5.3. Alternative (c): Center Within the Coherence Computation

Apply centering to both the reference matrix and the masked matrix inside the Iproj computation, rather than as a preprocessing step:

$$I_k^c(p) = \|P_k(H\widetilde{S}(p)H)\, u_k(HSH)\|^2.$$

This has a conceptual advantage: the centering is part of the test definition, not a preprocessing step. It makes explicit that we are testing the structure of the centered matrix.

However, there is a subtlety: $H\widetilde{S}(p)H \neq \widetilde{(HSH)}(p)$ in general, because centering and masking do not commute. Specifically:

$$H\widetilde{S}(p)H = H\left[D_S + \frac{1}{p}(S - D_S) \circ M\right]H$$

while

$$\widetilde{(HSH)}(p) = D_{HSH} + \frac{1}{p}(HSH - D_{HSH}) \circ M.$$

The first centers after masking; the second masks the already-centered matrix. They differ because centering the masked matrix accounts for the pattern of revealed entries (which changes per bootstrap), while masking the centered matrix uses a fixed centering.

**In the current implementation,** the PCT centers $S_0$ once and then masks the centered matrix $S_c$. This corresponds to $\widetilde{(HSH)}(p)$, which is the correct choice: the reference eigenspace is computed from $S_c$, and each masked replicate is a masking of $S_c$. The centering is applied once and consistently to all replicates.

**Verdict:** The current approach (center once, then mask) is the most principled because it ensures the reference eigenspace and all masked replicates share the same centering. Centering inside the loop (center each masked matrix separately) would introduce variability in the centering step itself.

### 5.4. The Most Principled Approach

The right framework is to view centering as part of the test definition, not as an ad-hoc fix:

**Definition (Centered Permutation Coherence Test).** The PCT operates on the double-centered matrix $S_c = HSH$ (or $HS_0H$ for partially observed data). The test statistic $I_k(p)$ measures the projection coherence of the $k$-th eigenvector of $S_c$ onto the top-$k$ eigensubspace of the Bernoulli-masked version of $S_c$.

This is natural because:

1. The mean component $m\mathbf{1}\mathbf{1}^\top$ carries no structural information (it is the same for all pairs).
2. The rank estimation question is about the signal $X$, not the signal-plus-mean $mJ + \sigma X$.
3. Centering is the standard spectral preprocessing in kernel PCA, MDS, and related methods.
4. The masking perturbation after centering scales with signal variance, making the test sensitive to actual structure rather than dominated by the constant background.

### 5.5. Observed-Entry Centering for Sparse Data

For partially observed matrices with $q$ significantly below 1, the principled generalization replaces naive double-centering with observed-entry centering:

$$\bar{S}_{i\cdot}^{\mathrm{obs}} = \frac{\sum_{j: (i,j) \in \Omega} S_{ij}}{|\{j : (i,j) \in \Omega\}|}, \quad \bar{S}^{\mathrm{obs}} = \frac{\sum_{(i,j) \in \Omega} S_{ij}}{|\Omega|},$$

$$(S_c^{\mathrm{obs}})_{ij} = \begin{cases} S_{ij} - \bar{S}_{i\cdot}^{\mathrm{obs}} - \bar{S}_{\cdot j}^{\mathrm{obs}} + \bar{S}^{\mathrm{obs}} & (i,j) \in \Omega, \\ 0 & (i,j) \notin \Omega. \end{cases}$$

This avoids centering the imputed zeros, which is the source of bias in the naive approach. The permutation null would similarly permute observed entries and then apply observed-entry centering to the permuted matrix.

---

## Step 6: Conclusions

### 6.1. Is Double-Centering the Right Fix?

**Yes.** Double-centering is the correct and necessary preprocessing for the PCT. It eliminates the dominant mean component that creates masking noise disproportionate to the signal. Without it, the test is powerless for any matrix where the mean entry level $m$ is large relative to the signal standard deviation $\sigma$.

### 6.2. Is It Always Safe?

**Nearly.** For fully observed matrices and nearly-fully-observed matrices ($q > 0.9$), double-centering via $HSH$ (or $HS_0H$) is safe and beneficial. The three cases:

- **High mean, fully observed:** centering is essential (without it, the test fails).
- **Low mean, fully observed:** centering is harmless (removes a negligible component).
- **Partially observed with zero-filling:** centering is approximately correct when $q$ is close to 1. For very sparse observation ($q \ll 1$), observed-entry centering (Section 5.5) is more principled, because naive centering distorts the zero-filled entries.

### 6.3. Theoretical Justification (One Paragraph)

The PCT measures $I_k(p) = \cos^2(\theta_k)$, the squared cosine of the angle between a reference eigenvector and the eigensubspace of a Bernoulli-masked matrix. By the Davis-Kahan $\sin\Theta$ theorem, this angle is bounded by $\|E\|_2 / \delta_k$, where $\|E\|_2$ is the spectral norm of the masking perturbation and $\delta_k$ is the eigenvalue gap. The masking perturbation scales with entry magnitudes: $\|E\|_2 = O(s_{\max}\sqrt{n/p})$ where $s_{\max}$ is the magnitude of the largest entry. For an uncentered near-constant matrix, $s_{\max} \approx m$ while the signal gaps are $\delta_k = O(\sigma n)$, giving $\sin(\theta_k) = O(m/(\sigma\sqrt{np}))$, which is uninformative when $m \gg \sigma$. Double-centering removes the mean component, reducing $s_{\max}$ from $O(m)$ to $O(\sigma)$, so the bound becomes $\sin(\theta_k) = O(1/\sqrt{np})$, independent of $m/\sigma$. This ensures that the masking perturbation is commensurate with the signal, making the test sensitive to actual eigenstructure rather than dominated by the constant background.

### 6.4. Presentation: Preprocessing Step or Part of the Test Definition?

**Part of the test definition.** Double-centering should not be presented as an optional preprocessing step or an empirical fix. It should be built into the definition of the PCT:

> *"The Centered Permutation Coherence Test operates on $S_c = HSH$..."*

This framing is justified because (i) the mean component is uninformative for rank estimation, (ii) centering is standard in spectral methods on kernel matrices, and (iii) without centering, the test has a provable failure mode for an important class of inputs.

### 6.5. Remaining Limitations

1. **Sparse partial observation.** When $q \ll 1$, naive centering (center after zero-filling) introduces bias from centering the imputed zeros. For such data, observed-entry centering (Section 5.5) should be used instead. The current implementation uses naive centering, which is adequate for the primary use case ($q > 0.99$) but should be generalized for sparse regimes.

2. **Non-constant mean structure.** The analysis assumes a roughly constant mean ($S \approx mJ + \sigma X$). If the mean structure is heterogeneous (e.g., block-diagonal with different mean levels per block), centering removes only the grand mean, leaving within-block mean components that can still dominate masking noise. For such matrices, a more refined preprocessing (e.g., block-wise centering or standardization) might be needed.

3. **Centering can remove signal.** If the signal itself has a component proportional to $\mathbf{1}$ (i.e., one of the signal dimensions represents a "size" or "overall similarity" factor), centering removes it. This is analogous to centering in PCA removing the mean: it is usually the right thing to do, but it means the PCT estimates the rank of the centered signal, not the raw signal. The rank of $HSH$ is at most $\mathrm{rank}(S) - 1$ (the mean eigenvector is removed).

4. **The perturbation bound is worst-case.** The matrix Bernstein bound is not tight for structured matrices (it treats all off-diagonal entries equally). For matrices with decaying off-diagonal structure, the actual perturbation may be much smaller than the bound suggests, and the test may have more power than the theory predicts.

5. **Interaction of centering with the permutation null.** After centering, the permuted matrix $S_{\pi,c}$ is *not* a centering of the permuted uncentered matrix, because centering is applied to the (already-centered) $S_c$ and then entries of $S_c$ (not $S_0$) are permuted. This means the null reflects the structure of permuted centered entries, which is the correct null for the centered test. But it means the test is testing the structure of the centered matrix, not the raw matrix. This is a feature, not a bug, but it should be stated clearly.

---

## Appendix: Detailed Matrix Bernstein Calculation

For completeness, we derive the spectral norm bound for the masking perturbation of a general symmetric matrix $A$ with zero diagonal (after centering, the diagonal is not masked).

Let $A \in \mathbb{R}^{n \times n}$ be symmetric with $A_{ii} = 0$. Under Bernoulli($p$) masking:

$$E = \sum_{i < j} A_{ij}\left(\frac{B_{ij}}{p} - 1\right)(e_i e_j^\top + e_j e_i^\top),$$

where $B_{ij} \sim \mathrm{Bernoulli}(p)$ are independent.

**Summand bound:**

$$\left\|A_{ij}\left(\frac{B_{ij}}{p} - 1\right)(e_i e_j^\top + e_j e_i^\top)\right\|_2 = |A_{ij}| \cdot \left|\frac{B_{ij}}{p} - 1\right| \leq \frac{|A_{ij}|}{p} \leq \frac{a_{\max}}{p} =: L,$$

where $a_{\max} = \max_{i \neq j} |A_{ij}|$.

**Variance parameter:**

$$V = \sum_{i < j} A_{ij}^2 \cdot \frac{1-p}{p} \cdot (e_i e_i^\top + e_j e_j^\top) = \frac{1-p}{p} \cdot \mathrm{diag}\left(\sum_{j \neq i} A_{ij}^2\right)_{i=1}^n.$$

$$\sigma_B^2 = \|V\|_2 = \frac{1-p}{p} \cdot \max_i \sum_{j \neq i} A_{ij}^2 =: \frac{1-p}{p} \cdot R_{\max}^2.$$

**Matrix Bernstein inequality:**

$$\Pr(\|E\|_2 \geq t) \leq 2n \exp\left(\frac{-t^2/2}{\sigma_B^2 + Lt/3}\right).$$

Setting $\eta = 2n \exp(\cdots)$ and solving:

$$\|E\|_2 \leq \sqrt{2\sigma_B^2 \log(2n/\eta)} + \frac{2L}{3}\log(2n/\eta)$$

with probability at least $1 - \eta$.

**For the centered matrix** with $a_{\max} = O(\sigma)$ and $R_{\max}^2 = O(n\sigma^2)$:

$$\|E_c\|_2 \leq \sigma\sqrt{\frac{2n(1-p)}{p}\log\frac{2n}{\eta}} + \frac{2\sigma}{3p}\log\frac{2n}{\eta} = O\!\left(\sigma\sqrt{\frac{n\log n}{p}}\right).$$

**For the uncentered near-constant matrix** with $a_{\max} = O(m)$ and $R_{\max}^2 = O(nm^2)$:

$$\|E\|_2 \leq m\sqrt{\frac{2n(1-p)}{p}\log\frac{2n}{\eta}} + \frac{2m}{3p}\log\frac{2n}{\eta} = O\!\left(m\sqrt{\frac{n\log n}{p}}\right).$$

The ratio between the two is $m/\sigma$, which is the signal-to-mean ratio. For Gaussian kernel matrices with $m/\sigma \approx 20$, the uncentered perturbation is 20x larger, while the signal gaps are the same. This is the quantitative explanation for the failure.
