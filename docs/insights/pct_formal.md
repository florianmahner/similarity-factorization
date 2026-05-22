# The Permutation Coherence Test for Rank Estimation of Symmetric Similarity Matrices

## Abstract

We present the Permutation Coherence Test (PCT), a nonparametric procedure for estimating the rank of a symmetric positive semi-definite similarity matrix under partial observation. Classical parallel analysis compares eigenvalue magnitudes against a permutation null, but eigenvalue magnitudes are confounded by observation rate, producing non-monotone rank estimates when data is incomplete. PCT replaces the eigenvalue comparison with *projection coherence*, a scale-free quantity measuring the angular stability of eigensubspaces under random masking. By testing whether the observed coherence exceeds that of entry-permuted null matrices (each evaluated against its own reference eigenspace), PCT achieves monotone rank estimates that converge to the true rank as the observation rate increases. We complement PCT with a gap-based diagnostic (the kappa changepoint) grounded in the Davis-Kahan $\sin\Theta$ theorem, and show that the two methods together bracket the effective dimensionality. Experiments on synthetic data and the THINGS behavioral similarity dataset demonstrate that PCT-estimated ranks yield 2-3x fewer dimensions than existing approaches while matching or exceeding predictive accuracy.

---

## 1. Setting and Notation

Let $S \in \mathbb{R}^{n \times n}$ be a symmetric positive semi-definite (PSD) similarity matrix with eigendecomposition

$$S = U \Lambda U^\top, \quad \Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n), \quad \lambda_1 \geq \cdots \geq \lambda_n \geq 0,$$

where $U = [u_1 \mid \cdots \mid u_n]$ is orthogonal. The *effective rank* of $S$ is the number of eigenvalues that reflect latent structure rather than noise.

**Partial observation.** Let $\Omega \subseteq \{(i,j) : 1 \leq i < j \leq n\}$ denote the set of observed off-diagonal pairs, with observation rate

$$q = \frac{|\Omega|}{\binom{n}{2}}.$$

Define $S_0$ as the matrix obtained from $S$ by setting unobserved off-diagonal entries to zero while preserving the diagonal:

$$(S_0)_{ij} = \begin{cases} S_{ij} & \text{if } (i,j) \in \Omega \text{ or } i = j, \\ 0 & \text{otherwise.} \end{cases}$$

**Bernoulli masking model.** For a sampling fraction $p \in (0, 1]$, define a symmetric Bernoulli mask $M(p) \in \{0, 1\}^{n \times n}$ where $M_{ij}(p) \sim \operatorname{Bernoulli}(p)$ independently for $i < j$, with $M_{ji}(p) = M_{ij}(p)$ and $M_{ii}(p) = 1$. The masked estimator is

$$\widetilde{S}(p) = D_S + \frac{1}{p}(S_0 - D_S) \circ M(p),$$

where $D_S = \operatorname{diag}(S_{11}, \ldots, S_{nn})$ and $\circ$ denotes the Hadamard product. The $1/p$ rescaling ensures $\mathbb{E}[\widetilde{S}(p)] = S_0$, so $\widetilde{S}(p)$ is an unbiased estimator. We write $E(p) = \widetilde{S}(p) - S_0$ for the perturbation induced by masking.

---

## 2. Projection Coherence

Let $U_k(A) \in \mathbb{R}^{n \times k}$ denote the matrix whose columns are the top-$k$ eigenvectors of a symmetric matrix $A$, and let

$$P_k(A) = U_k(A)\, U_k(A)^\top$$

be the orthogonal projection onto the corresponding eigensubspace. The *incremental projection coherence* of the $k$-th dimension under masking at rate $p$ is

$$I_k^{\mathrm{proj}}(p) = \left\| P_k\!\left(\widetilde{S}(p)\right) u_k(S_0) \right\|^2 = \sum_{j=1}^{k} \left( u_j\!\left(\widetilde{S}(p)\right)^\top u_k(S_0) \right)^2.$$

This quantity equals $\cos^2(\theta)$, where $\theta$ is the angle between the $k$-th reference eigenvector $u_k(S_0)$ and the top-$k$ eigensubspace of $\widetilde{S}(p)$. It satisfies $I_k^{\mathrm{proj}}(p) \in [0, 1]$, with values near 1 indicating that the $k$-th dimension is stably recoverable at masking rate $p$.

**Connection to Davis-Kahan.** The $\sin\Theta$ theorem (Davis and Kahan, 1970) bounds the angular perturbation of eigensubspaces:

$$\|\sin\Theta(\hat{U}_k, U_k)\|_2 \leq \frac{\|E(p)\|_2}{\delta_k}, \quad \text{where } \delta_k = \lambda_k - \lambda_{k+1}.$$

Since $I_k^{\mathrm{proj}}(p) = \cos^2\theta_k$ and $\sin^2\theta_k \leq \|E(p)\|_2^2 / \delta_k^2$, we obtain the lower bound

$$I_k^{\mathrm{proj}}(p) \geq 1 - \frac{\|E(p)\|_2^2}{\delta_k^2}.$$

Dimensions with large spectral gaps $\delta_k$ relative to the perturbation norm have coherence close to 1; dimensions in the noise tail, where $\delta_k \approx 0$, have no such guarantee.

---

## 3. Three Approaches to Rank Estimation from Coherence

We describe three methods in order of development. The first two motivate the design of the third.

### 3.1. Activation Counting

**Null model.** Compare each $I_k^{\mathrm{proj}}(p)$ against the coherence expected for a random direction projected onto the reference eigenspace. For a uniformly random unit vector $v \in \mathbb{R}^n$, we have $\mathbb{E}[\|P_k v\|^2] = k/n$, yielding a threshold $\tau_k(p) \sim k/n$.

**Decision rule.** Dimension $k$ is classified as signal if the lower confidence bound of $I_k^{\mathrm{proj}}(p)$ exceeds $\tau_k(p)$ at any masking rate $p$. The rank estimate is

$$k^* = \max\left\{k : \inf_{\text{CI}} I_k^{\mathrm{proj}}(p) > \tau_k(p) \text{ for some } p\right\}.$$

**Failure mode.** For large $n$, the threshold $\tau_k \sim k/n$ is extremely small. Even noise eigenvectors, which are not truly random but are structured by the observation pattern, can exceed this threshold. On THINGS ($n = 1854$), activation counting yields $k^* = 97$, far above any reasonable rank. The root cause is that the null model (uniform random vectors) is inappropriate: eigenvectors of a noisy incomplete matrix are not isotropically distributed. They concentrate in directions compatible with the sparsity pattern, producing systematically higher coherence than a uniform null predicts.

### 3.2. Kappa Changepoint (Gap-Based, B1)

**Statistic.** For each dimension $k$, define the *scaled leakage rate*

$$\kappa_k = \operatorname{median}_{p \in \mathcal{P}_{\mathrm{high}}} \left[\frac{(1 - I_k^{\mathrm{proj}}(p)) \cdot p}{1 - p}\right],$$

where $\mathcal{P}_{\mathrm{high}}$ denotes masking rates in the upper portion of the schedule. The quantity $\kappa_k$ measures how rapidly coherence degrades under masking. By the Davis-Kahan bound, $\kappa_k$ is small when $\delta_k$ is large (strong signal) and large when $\delta_k$ is small (noise).

**Decision rule.** The rank estimate $k^*_{B1}$ is the position of the largest jump in the sequence $\kappa_1, \kappa_2, \ldots$:

$$k^*_{B1} = \arg\max_k (\kappa_{k+1} - \kappa_k).$$

**Limitation.** The method assumes a single changepoint separating signal from noise. For matrices with gradually decaying spectra, the largest jump may not shift monotonically as data accumulates. On THINGS, kappa gives $k^* = 11$ for both 10% and 20% data, flat where one expects an increase. The single-changepoint assumption is too rigid for continuous spectral decay.

### 3.3. Permutation Coherence Test (PCT, B2)

We now develop the main method, which overcomes the limitations of both approaches above. Full details are given in Sections 5-7.

---

## 4. Why Eigenvalue-Based Parallel Analysis Fails for Partial Data

Before presenting PCT, we explain why the classical eigenvalue-based parallel analysis (Horn, 1965) is unsuitable for partially observed matrices, as this failure motivates the switch from eigenvalue magnitudes to projection coherence.

**Eigenvalue scaling under partial observation.** When a fraction $q$ of off-diagonal entries are observed, the observed matrix $S_0$ has a systematic relationship to the true matrix $S$. In expectation:

- Signal eigenvalues scale as $\lambda_k(S_0) \approx q \cdot \lambda_k(S)$.
- Null eigenvalues of a permuted matrix $S_\pi$ scale as $\lambda_k(S_\pi) \propto \sqrt{q}$ (consistent with random matrix theory for sparse Wigner-type ensembles).

The signal-to-null eigenvalue ratio therefore scales as $\sqrt{q}$, which improves with observation rate. However, the *absolute* comparison $\lambda_k(S_0) > \lambda_k(S_\pi)$ is confounded: at low $q$, both quantities are small and close together, and finite-sample fluctuations in the null threshold can dominate the comparison.

**Empirical demonstration.** For a rank-10 synthetic matrix ($n = 200$, Dirichlet-generated factors with $\alpha = 0.5$), the eigenvalue-based permutation test across observation rates $q \in \{0.1, 0.2, 0.3, 0.5, 0.7, 1.0\}$ gives:

$$k^*_{\mathrm{eig}} \in \{20, 8, 11, 14, 10, 10\}.$$

The estimate is 20 at $q = 0.1$ (severe overestimation), drops to 8 at $q = 0.2$, and oscillates before converging at full observation. This non-monotonicity is unacceptable for a rank estimation procedure: adding data should never reduce the estimated rank.

**Root cause.** The eigenvalue comparison conflates two effects: (1) the strength of latent structure and (2) the density of the observation pattern. Both the observed and null matrices share the same sparsity-induced eigenvalue scaling, so any test based on eigenvalue magnitudes is confounded by observation rate. The correct approach is to test a quantity that depends only on eigenvector *directions*, not eigenvalue *magnitudes*.

---

## 5. PCT: Null Model

**Null hypothesis $H_0$.** The eigenvector structure of $S_0$ produces no more coherence under masking than a matrix with the same observed entries randomly reassigned among observed positions.

**Construction of $S_\pi$.** Let $\Omega$ be the set of observed off-diagonal positions in $S_0$. A null replicate $S_\pi$ is constructed as follows:

1. Extract the multiset of observed off-diagonal values: $\mathcal{V} = \{S_0[i,j] : (i,j) \in \Omega\}$.
2. Draw a uniformly random permutation $\sigma$ of $\mathcal{V}$.
3. Assign the permuted values to the positions in $\Omega$: for each $(i,j) \in \Omega$, set $S_\pi[i,j] = \sigma((i,j))$.
4. For $(i,j) \notin \Omega$, set $S_\pi[i,j] = 0$ (preserving the sparsity pattern).
5. Symmetrize: $S_\pi[j,i] = S_\pi[i,j]$ for all $i < j$.
6. Preserve the diagonal: $S_\pi[i,i] = S_0[i,i]$ for all $i$.

**Critical property.** The matrices $S_0$ and $S_\pi$ have *identical sparsity patterns*. They share the same diagonal, the same set of nonzero off-diagonal positions, and the same marginal distribution of off-diagonal values (all moments are preserved). The only difference is the assignment of values to positions.

**What is preserved:** marginal distribution of observed entries, diagonal, sparsity pattern $\Omega$, symmetry.

**What is destroyed:** all pairwise correlational structure, transitivity relations, latent factor structure.

---

## 6. PCT: Test Statistic and Procedure

### 6.1. Observed coherence (bootstrap-aggregated)

Let $\mathcal{P} = \{p_1, \ldots, p_P\}$ be a schedule of masking rates and let $K$ be the maximum rank to evaluate.

1. Compute the reference eigenspace: the top-$K$ eigenvectors $U_K(S_0)$ of $S_0$.
2. For each bootstrap replicate $b = 1, \ldots, B$ and each masking rate $p \in \mathcal{P}$:
   - Draw an independent Bernoulli($p$) mask $M^{(b)}(p)$.
   - Compute the masked estimator $\widetilde{S}^{(b)}(p)$.
   - Compute the top-$K$ eigenvectors of $\widetilde{S}^{(b)}(p)$.
   - For each $k = 1, \ldots, K$, compute $I_k^{(b)}(p)$ using the projection coherence formula (Section 2).
3. Aggregate across bootstrap replicates:

$$I_k^{\mathrm{obs}}(p) = \operatorname{median}_{b=1}^{B}\, I_k^{(b)}(p).$$

### 6.2. Null coherence (permutation replicates)

For each null replicate $j = 1, \ldots, J$:

1. Generate $S_\pi^{(j)}$ by entry permutation within $\Omega$ (Section 5).
2. Compute the reference eigenspace of $S_\pi^{(j)}$: the top-$K$ eigenvectors of $S_\pi^{(j)}$ itself.
3. For each masking rate $p \in \mathcal{P}$:
   - Draw a single Bernoulli($p$) mask, apply to $S_\pi^{(j)}$, and compute the top-$K$ eigenvectors of the masked version.
   - Compute $I_k^{\mathrm{null},j}(p)$ by measuring the projection coherence of the masked eigenspace of $S_\pi^{(j)}$ against the reference eigenspace of $S_\pi^{(j)}$.

**Self-referencing property.** Each null replicate's coherence is measured against its *own* reference eigenspace, not against the reference eigenspace of $S_0$. This is essential. If the null coherence were measured against $S_0$'s eigenspace, the null would test a different hypothesis (alignment of random structure with the observed signal) rather than the intended hypothesis (intrinsic coherence of random structure under masking). Self-referencing ensures that the null captures the baseline level of eigensubspace stability that arises from random value-to-position assignments.

### 6.3. Threshold

At each pair $(k, p)$, the null distribution is

$$\mathcal{F}_k^{\mathrm{null}}(p) = \left\{I_k^{\mathrm{null},j}(p) : j = 1, \ldots, J\right\}.$$

The critical value at significance level $\alpha$ is

$$\tau_k(p, \alpha) = Q_{1-\alpha}\!\left(\mathcal{F}_k^{\mathrm{null}}(p)\right),$$

where $Q_{1-\alpha}$ denotes the $(1-\alpha)$-quantile.

**Pointwise test.** Reject $H_0$ for dimension $k$ at masking rate $p$ if $I_k^{\mathrm{obs}}(p) > \tau_k(p, \alpha)$.

---

## 7. Multiple Testing and Decision Rule

### 7.1. Intersection test over masking rates

Testing at a single masking rate $p$ may be unreliable: at very low $p$, noise dominates; at very high $p$, signal and noise are both well-estimated. We require significance across a range of rates to ensure robustness.

Define $p_0 = \operatorname{median}(\mathcal{P})$. Dimension $k$ is classified as signal if the pointwise test rejects at every $p \geq p_0$:

$$R_k = \bigcap_{p \in \mathcal{P},\; p \geq p_0} \left\{I_k^{\mathrm{obs}}(p) > \tau_k(p, \alpha)\right\}.$$

The restriction to $p \geq p_0$ focuses on masking rates where the signal-to-noise ratio is favorable while retaining enough perturbation to discriminate signal from noise.

### 7.2. Per-dimension p-value

The intersection is implemented via a conservative p-value:

$$\hat{p}_k = \max_{p \in \mathcal{P},\; p \geq p_0} \left\{\frac{1 + \sum_{j=1}^{J} \mathbf{1}\!\left[I_k^{\mathrm{null},j}(p) \geq I_k^{\mathrm{obs}}(p)\right]}{1 + J}\right\}.$$

The maximum over $p$ implements the intersection: a dimension must be significant at every tested masking rate. The $+1$ correction in numerator and denominator (Phipson and Smyth, 2010) ensures the p-value is never exactly zero and guarantees finite-sample validity.

### 7.3. Rank estimate

$$k^*_{\mathrm{PCT}} = \max\{k : \hat{p}_k < \alpha\},$$

with $k^*_{\mathrm{PCT}} = 0$ if no dimension rejects.

**Proposition 1 (Type-I error control).** Under $H_0$, for each $k$, $\Pr(\hat{p}_k < \alpha) \leq \alpha$.

*Proof.* The proof has three steps: (i) establish that the self-coherence functional has the same distribution for the observed and null matrices under $H_0$, (ii) show that median aggregation ($B > 1$) preserves validity, and (iii) apply the intersection.

**Step 1 (Distributional equivalence).** Define the *self-coherence functional*: for a symmetric matrix $A$ with the same diagonal, sparsity pattern, and value multiset as $S_0$, let

$$C_k(A, p, \omega) = \|P_k(\widetilde{A}(p, \omega))\, u_k(A)\|^2,$$

where $\widetilde{A}(p, \omega)$ is the Bernoulli-masked version of $A$ using mask seed $\omega$, and $u_k(A)$ is the $k$-th eigenvector of $A$ (self-referenced). Under $H_0$, the assignment of values to positions in $\Omega$ is uniformly random. Writing $S_0 = \phi(\sigma_0)$ and $S_\pi^{(j)} = \phi(\sigma_j)$ where $\sigma_0, \sigma_1, \ldots, \sigma_J$ are i.i.d. uniform permutations of the observed value multiset, the random variables $C_k(\phi(\sigma_i), p, \omega_i)$ are i.i.d. for independent mask seeds $\omega_i$. Let $F$ denote their common distribution.

**Step 2 (Median aggregation is conservative).** The observed coherence is $I_k^{\mathrm{obs}}(p) = \mathrm{median}_{b=1}^{B}\, C_k(S_0, p, \omega_b)$, while each null coherence $I_k^{\mathrm{null},j}(p) = C_k(S_\pi^{(j)}, p, \omega_j)$ is a single draw from $F$. The observed and null statistics are therefore computed differently when $B > 1$: the observed is a sample median, the null is a single draw.

For $B = 1$, the standard permutation test applies: all $J + 1$ statistics are i.i.d. draws from $F$, and $\Pr(\hat{p}_k^{(p)} < \alpha) \leq \alpha$ by the classical result (Lehmann and Romano, 2005, Theorem 15.2.1).

For $B > 1$, the sample median $\mathrm{Median}_B(F)$ is more concentrated than a single draw. The p-value $\hat{p}_k^{(p)} \leq \alpha$ requires $I_k^{\mathrm{obs}}(p)$ to exceed the $(1 - \alpha)$-quantile of the null draws. This occurs only if $\mathrm{Median}_B(F) > F^{-1}(1 - \alpha)$, which requires at least $\lceil B/2 \rceil$ of $B$ i.i.d. draws from $F$ to exceed $F^{-1}(1 - \alpha)$. Each draw exceeds this quantile with probability $\alpha$, so

$$\Pr\!\left(\mathrm{Median}_B(F) > F^{-1}(1 - \alpha)\right) = \Pr\!\left(\mathrm{Bin}(B, \alpha) \geq \lceil B/2 \rceil\right) \leq \alpha$$

for all $\alpha \in (0, 1/2]$ and $B \geq 1$, with strict inequality when $B \geq 2$.

**Step 3 (Intersection).** Since $\hat{p}_k = \max_{p \geq p_0} \hat{p}_k^{(p)} \geq \hat{p}_k^{(p')}$ for any particular $p'$,

$$\Pr(\hat{p}_k < \alpha) \leq \Pr(\hat{p}_k^{(p')} < \alpha) \leq \alpha. \qquad \square$$

**Remark 1.** There are two independent sources of conservatism: (a) the median aggregation ($B > 1$), which concentrates the observed statistic near the center of $F$ rather than its tails, and (b) the intersection-union structure, which requires significance at every tested masking rate. For typical parameters ($B = 30$, $\alpha = 0.05$), the Step 2 bound gives $\Pr(\mathrm{Bin}(30, 0.05) \geq 15) < 10^{-15}$, so the effective type-I error is far below the nominal $\alpha$. This conservatism is deliberate: overestimating rank is typically more harmful than underestimation in downstream applications such as non-negative matrix factorization.

**Remark 2.** Setting $B = 1$ recovers exact permutation test validity via exchangeability, at the cost of higher variance in the observed statistic. The choice of $B > 1$ trades exact level-$\alpha$ control for a more stable (less noisy) observed coherence estimate, with validity preserved by the stochastic dominance argument in Step 2.

---

## 8. Why PCT is Scale-Free and Monotone

### 8.1. Scale-freeness

The projection coherence $I_k^{\mathrm{proj}}(p) = \cos^2(\theta)$ measures the angle between eigensubspaces, not the magnitude of eigenvalues. For any scalar $c > 0$, the matrix $cS$ has the same eigenvectors as $S$ (assuming distinct eigenvalues), so

$$I_k^{\mathrm{proj}}(p; cS) = I_k^{\mathrm{proj}}(p; S).$$

More generally, changing the observation rate $q$ rescales eigenvalues of $S_0$ but does not systematically change eigenvector directions. The null matrix $S_\pi$ shares the same observation rate, so any residual scaling effect cancels in the comparison. This is the fundamental advantage over eigenvalue-based tests.

### 8.2. Monotonicity

**Claim.** For nested observations $\Omega_1 \subset \Omega_2$ with $q_1 < q_2$, the expected number of detected dimensions is non-decreasing: $\mathbb{E}[k^*_{\mathrm{PCT}}(q_1)] \leq \mathbb{E}[k^*_{\mathrm{PCT}}(q_2)]$.

**Argument (informal).** Consider the behavior of signal and null coherences separately as $q$ increases:

*Signal dimensions.* More observed entries better constrain the reference eigenspace of $S_0$, yielding eigenvectors closer to those of the true $S$. Under masking, the masked eigenvectors also improve. Both effects increase $I_k^{\mathrm{obs}}(p)$ for signal dimensions.

*Null dimensions.* The permuted matrix $S_\pi$ has the same observation rate $q$. Its eigenvectors encode no latent structure (the structure was destroyed by permutation), so additional data does not systematically improve their stability. The null threshold $\tau_k(p, \alpha)$ remains approximately constant as $q$ increases.

*Net effect.* The gap between $I_k^{\mathrm{obs}}$ and $\tau_k$ widens with $q$ for signal dimensions, causing more dimensions to cross the significance threshold.

**Empirical validation.** For a rank-10 synthetic matrix ($n = 200$, Dirichlet-generated factors with $\alpha = 0.5$, averaged over 5 random seeds), the PCT gives:

| Observation rate $q$ | 10% | 20% | 30% | 50% | 70% | 100% |
|:---|:---|:---|:---|:---|:---|:---|
| Mean $k^*_{\mathrm{PCT}}$ | 0.8 | 2.4 | 3.0 | 7.6 | 9.8 | 10.0 |

The sequence is monotone and converges to the true rank of 10. By contrast, the eigenvalue-based permutation test on the same data gives $k^*_{\mathrm{eig}} \in \{20, 8, 11, 14, 10, 10\}$ on a single seed (non-monotone, overestimates at low $q$).

---

## 9. Connection to Davis-Kahan and Combined Inference

### 9.1. The Davis-Kahan $\sin\Theta$ theorem

Let $\widetilde{S} = S + E$ be a perturbed symmetric matrix. The $\sin\Theta$ theorem (Davis and Kahan, 1970) states that the principal angle $\theta_k$ between the $k$-th eigensubspaces of $S$ and $\widetilde{S}$ satisfies

$$\sin\theta_k \leq \frac{\|E\|_2}{\delta_k}, \quad \delta_k = \lambda_k(S) - \lambda_{k+1}(S),$$

provided $\delta_k > \|E\|_2$. Equivalently, $I_k^{\mathrm{proj}} = \cos^2\theta_k \geq 1 - \|E\|_2^2 / \delta_k^2$.

### 9.2. Gap-based rank estimation (B1, kappa changepoint)

The kappa changepoint operationalizes the Davis-Kahan condition. Define the scaled leakage

$$\kappa_k = \operatorname{median}_{p \in \mathcal{P}_{\mathrm{high}}} \left[\frac{(1 - I_k^{\mathrm{proj}}(p)) \cdot p}{1 - p}\right].$$

By the Davis-Kahan bound, $1 - I_k^{\mathrm{proj}}(p) \lesssim \|E(p)\|_2^2 / \delta_k^2$. Since $\|E(p)\|_2^2 \propto (1 - p)/p$ for Bernoulli masking, we obtain $\kappa_k \propto 1/\delta_k^2$. The kappa sequence is small for dimensions with large gaps (signal) and large for dimensions with vanishing gaps (noise). The rank estimate is

$$k^*_{B1} = \arg\max_k\, (\kappa_{k+1} - \kappa_k),$$

the position of the largest jump, which identifies the transition from signal to noise in the spectral gap sequence.

### 9.3. Complementarity of B1 and B2

| Property | B1 (kappa changepoint) | B2 (PCT) |
|:---|:---|:---|
| Tests | Subspace stability under perturbation | Eigenvector structure vs. random assignment |
| Requires | Large consecutive gap $\delta_k$ | Coherence above permutation null |
| Scale-free | Yes (uses $I_k^{\mathrm{proj}}$) | Yes (uses $I_k^{\mathrm{proj}}$) |
| Monotone in data | Yes (more data clarifies gaps) | Yes (more data stabilizes eigenvectors) |
| Conservative when | Gradual spectral decay | Strong marginal entry distribution |

### 9.4. Combined interval

Define $k_{B1} = k^*_{B1}$ (kappa changepoint) and $k_{B2} = k^*_{\mathrm{PCT}}$ (permutation coherence test). The interval $[k_{B1}, k_{B2}]$ brackets the effective dimensionality:

- $k_{B1}$: number of dimensions with *robust spectral gaps*, recoverable regardless of the specific perturbation.
- $k_{B2}$: number of dimensions with *eigenvector structure above chance*, as determined by comparison to the permutation null.

For well-separated spectra, $k_{B1} \approx k_{B2}$. For gradual spectral decay, $k_{B1} < k_{B2}$, and the interval width quantifies the inherent ambiguity in rank selection for that data regime.

---

## 10. Empirical Results on THINGS Behavioral Data

We evaluate PCT and kappa on the THINGS behavioral similarity dataset (Hebart et al., 2020), which contains $n = 1854$ object concepts with pairwise similarity judgments collected via odd-one-out triplet tasks. We subsample the triplet data at various percentages and compare rank estimates against VICE (Variational Interpretable Concept Embeddings; Muttenthaler et al., 2022), a variational method that jointly learns embeddings and selects rank via pruning. Predictive accuracy is measured on held-out triplets.

| Triplet % | PCT $k^*$ | Kappa $k^*$ | VICE dims | SRF@PCT acc | SRF@kappa acc | SRF@VICE acc | VICE acc |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 5% | 4 | 6 | 10 | 58.90 | 59.69 | 59.89 | 56.63 |
| 10% | 10 | 11 | 19 | 61.57 | 61.62 | 61.20 | 59.79 |
| 20% | 14 | 11 | 31 | 62.77 | 62.41 | 62.59 | 61.83 |
| 50% | 20 | 21 | 52 | 63.84 | 63.88 | 64.43 | 63.62 |
| 100% | 25 | 26 | -- | 64.37 | 64.41 | 64.09 | 64.22 |

**Key findings.**

1. *PCT rank is monotone.* The PCT sequence $(4, 10, 14, 20, 25)$ is strictly increasing with data. The kappa sequence $(6, 11, 11, 21, 26)$ is non-monotone between 10% and 20%, consistent with the single-changepoint limitation discussed in Section 3.2.

2. *Coherence-estimated ranks are parsimonious.* Both PCT and kappa select 2-3x fewer dimensions than VICE at every data level. Despite using far fewer dimensions, SRF at coherence-estimated ranks matches or exceeds VICE accuracy at all data levels.

3. *SRF outperforms VICE at low data.* At 5% data, SRF@kappa ($k = 6$) achieves 59.69% accuracy versus VICE's 56.63%, a gap of over 3 percentage points. The advantage diminishes at higher data levels, consistent with both methods converging to the true structure.

4. *VICE rank overfits at full data.* At 100% data, VICE selects approximately 66 dimensions (not shown in table; no VICE dims column because VICE was not re-run). SRF at the VICE rank ($k = 66$) achieves 64.09%, while SRF at the coherence rank ($k = 25\text{-}26$) achieves 64.37-64.41%. The additional 40+ dimensions do not improve prediction and may slightly hurt generalization.

---

## 11. Computational Considerations

### 11.1. Cost analysis

The dominant cost is eigendecomposition. Each call to `eigh` on an $n \times n$ matrix costs $O(n^2 K)$ when only the top $K$ eigenpairs are needed (via iterative methods) or $O(n^3)$ for full decomposition.

The total number of eigendecompositions is:

- **Observed coherence:** $B \times P$ (one per bootstrap replicate per masking rate), plus 1 for the reference eigenspace of $S_0$.
- **Null coherence:** $J \times (1 + P)$ (one reference eigenspace per null replicate, plus one masked eigenspace per masking rate).

Total: $(J + B) \cdot P + J + 1$ eigendecompositions.

### 11.2. Practical scaling

For $n = 1854$, $K = 120$, $J = 100$, $B = 30$, $P = 15$:

- Null: $100 \times (1 + 15) = 1600$ eigendecompositions.
- Observed: $30 \times 15 + 1 = 451$ eigendecompositions.
- Total: 2051 calls to `eigh` on $1854 \times 1854$ matrices.
- Each `eigh` call takes approximately 2 seconds.
- Serial runtime: approximately 68 minutes.
- Parallel runtime (100+ cores): approximately 5 minutes.

### 11.3. Parallelization

The computation is embarrassingly parallel over (replicate, masking rate) pairs. Each worker receives the upper-triangle values and diagonal, reconstructs or permutes the matrix, applies the mask, computes eigenvectors, and returns $I_k(p)$ for $k = 1, \ldots, K$. Memory per worker is one $n \times n$ matrix (approximately 27 MB for $n = 1854$ in double precision).

### 11.4. Reducing cost

Several strategies reduce computation without substantial loss of power:

- Reduce the number of masking rates to $P = 10$, since the intersection test only uses rates $p \geq p_0$.
- Use fewer null replicates ($J = 50$) for exploratory analysis, increasing to $J = 100$ for final results.
- The reference eigenspace of $S_0$ is computed once and reused across all $B$ bootstrap masks.
- For very large $n$, randomized eigensolvers (Halko et al., 2011) reduce per-call cost from $O(n^3)$ to $O(n^2 K)$.

---

## References

- Davis, C. and Kahan, W. M. (1970). The rotation of eigenvectors by a perturbation. III. *SIAM Journal on Numerical Analysis*, 7(1):1-46.
- Halko, N., Martinsson, P. G., and Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2):217-288.
- Hebart, M. N., Zheng, C. Y., Pereira, F., and Baker, C. I. (2020). Revealing the multidimensional mental representations of natural objects underlying human similarity judgements. *Nature Human Behaviour*, 4(11):1173-1185.
- Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika*, 30(2):179-185.
- Lehmann, E. L. and Romano, J. P. (2005). *Testing Statistical Hypotheses*. Springer, 3rd edition.
- Muttenthaler, L., Zheng, C. Y., McClure, P., and Hebart, M. N. (2022). VICE: Variational interpretable concept embeddings. *Advances in Neural Information Processing Systems*, 35.
- Phipson, B. and Smyth, G. K. (2010). Permutation p-values should never be zero: Calculating exact p-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1):Article 39.
