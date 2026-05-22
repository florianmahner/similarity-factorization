# Simulation Experiment Notes

## Factor Recovery Chance Baseline

Factor recovery measures how well SRF recovers ground truth factors $W$ from similarity $S = WW^\top$. The metric is the mean absolute correlation between aligned factor columns.

### Metric Definition

Given true factors $W$ and learned factors $\hat{W}$ (both $n \times k$):
1. Align $\hat{W}$ to $W$ via Hungarian algorithm on normalized similarity $W_{\text{norm}}^\top \hat{W}_{\text{norm}}$
2. Compute $r_i = |\text{corr}(W_{:,i}, \hat{W}_{:,\pi(i)})|$ for optimal permutation $\pi$
3. Return $\frac{1}{k}\sum_i r_i$

### Part 1: Expected Absolute Correlation

For two independent random vectors $x, y \in \mathbb{R}^n$, the sample correlation coefficient $r$ satisfies:

$$\mathbb{E}[r] = 0, \quad \text{Var}[r] \approx \frac{1}{n-1}$$

The absolute value $|r|$ follows a folded normal distribution. For large $n$:

$$\mathbb{E}[|r|] = \sqrt{\frac{2}{\pi}} \cdot \text{SD}[r] = \sqrt{\frac{2}{\pi(n-1)}}$$

$$\text{Var}[|r|] = \text{Var}[r] - \mathbb{E}[|r|]^2 = \frac{1}{n-1} - \frac{2}{\pi(n-1)} = \frac{\pi - 2}{\pi(n-1)}$$

For $n = 300$:

$$\mu \equiv \mathbb{E}[|r|] \approx \sqrt{\frac{2}{\pi \cdot 299}} \approx 0.046$$

$$\sigma \equiv \text{SD}[|r|] \approx \sqrt{\frac{\pi - 2}{\pi \cdot 299}} \approx 0.035$$

### Part 2: Hungarian Algorithm Selection Bias

Consider the $k \times k$ correlation matrix $M$ where $M_{ij} = |\text{corr}(W_{:,i}, \hat{W}_{:,j})|$. When $W$ and $\hat{W}$ are independent, entries are approximately i.i.d. with mean $\mu$ and standard deviation $\sigma$.

The Hungarian algorithm solves:

$$\max_{\pi \in S_k} \sum_{i=1}^{k} M_{i,\pi(i)}$$

For i.i.d. entries, the expected optimal assignment sum exceeds $k\mu$ due to selection bias. From random assignment theory, the expected boost scales as:

$$\mathbb{E}\left[\frac{1}{k}\sum_{i} M_{i,\pi^*(i)}\right] \approx \mu + \frac{\sigma}{\sqrt{k}} \cdot \gamma_k$$

where $\gamma_k$ captures the selection effect. For moderate $k$, this can be approximated using extreme value theory. The optimal assignment selects values biased toward the upper tail of each row/column, with effective boost:

$$\gamma_k \approx \sqrt{2\log k}$$

### Part 3: Combined Estimate

The chance-level factor recovery is:

$$\text{chance} \approx \mu + \frac{\sigma \sqrt{2\log k}}{\sqrt{k}} = \sqrt{\frac{2}{\pi(n-1)}} \left(1 + \sqrt{\frac{\pi-2}{2}} \cdot \frac{\sqrt{2\log k}}{\sqrt{k}}\right)$$

For $n = 300$, $k = 5$:

$$\text{chance} \approx 0.046 + \frac{0.035 \cdot \sqrt{2 \log 5}}{\sqrt{5}} \approx 0.046 + \frac{0.035 \cdot 1.79}{2.24} \approx 0.046 + 0.028 \approx 0.074$$

### Empirical Verification

Monte Carlo simulation ($n = 300$, $k = 5$, 100 trials of random Dirichlet(1) factors):

$$\textbf{Chance baseline} = 0.068 \pm 0.015$$

This matches the theoretical prediction of $\approx 0.07$.

### Summary

| Component | Formula | Value ($n$=300, $k$=5) |
|-----------|---------|------------------------|
| Base $\mathbb{E}[\|r\|]$ | $\sqrt{2/\pi(n-1)}$ | 0.046 |
| Selection boost | $\sigma\sqrt{2\log k}/\sqrt{k}$ | 0.028 |
| **Total** | | **0.074** |

The chance level depends on $n$ and $k$, not on $1/k$ (which would apply to classification accuracy, not correlation).
