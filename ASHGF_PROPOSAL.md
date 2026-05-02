# ASHGF Improvement Proposal

> **Status**: preliminary analysis based on low-dimensional tests (dim ≤ 100).
> **⚠️  To be re-evaluated** once results for dim ∈ {500, 1000, 2000} become available.

---

## 1. Current behaviour (dim = 50, max_iter = 500)

| Function | ASHGF best | ASGF best | GD best | Notes |
|----------|-----------|-----------|---------|-------|
| sphere | **1.05e-15** (iter 20) | 4.26e-16 | 4.18e+01 | Perfect convergence, no oscillations |
| ackley | **4.53e-09** (iter 35) | 6.59e-09 | 5.28e+00 | Fast convergence |
| rastrigin | **1.59e+02** (iter 143) | 2.63e+02 | 4.88e+01 | Oscillates after reaching plateau |
| levy | **3.58e-01** (iter 499) | 8.95e-01 | 3.27e+01 | Very slow progress, persistent oscillations |
| rosenbrock | **2.27e+01** (iter 500) | 2.79e+01 | 3.72e+01 | Slow but steady progress |

**Key observations:**

- ASHGF **converges fast** on well-behaved functions (sphere, ackley): reaches machine precision in 20–35 iterations.
- On multi-modal or ill-conditioned functions (rastrigin, levy, rosenbrock) it **reaches a plateau then oscillates** for hundreds of iterations.
- On rastrigin: 6 basis resets, sigma oscillates 8.2% of iterations, alpha stuck at 0.9 for 62% of the time.
- On levy: sigma oscillates between 8e-3 and 6e-1, step size spikes to 4.0, causing large jumps away from the minimum.

---

## 2. Root causes identified

### 2.1 Sigma oscillation (threshold-based adaptation)

The current rule multiplies/divides sigma by `gamma_sigma_plus` / `gamma_sigma_minus` when the ratio of directional derivatives to Lipschitz constants crosses thresholds A or B. This binary logic creates a **bang-bang controller**:

```
sigma too large → ratio < A → sigma *= 0.9
sigma too small → ratio > B → sigma *= 1.111...
sigma too large → ratio < A → sigma *= 0.9
...
```

With no hysteresis or damping, sigma oscillates indefinitely.

### 2.2 Alpha stuck at extremes

Alpha (probability of random directions) reaches `k1 = 0.9` and stays there, meaning **90% of directions are random** even though gradient history is available. The update rule compares mean min-evaluations between gradient-subspace and random directions; when the function is flat (plateau), the comparison is noisy and alpha drifts to extremes.

### 2.3 Step size spikes

The step size `sigma / L_nabla` can spike when sigma grows faster than L_nabla adapts. On levy we observed step sizes jumping from 0.2 to 4.0 in a single iteration, causing the algorithm to leave a good region.

### 2.4 Basis resets are disruptive

When sigma drops below `ro * sigma_zero`, the entire orthonormal basis is replaced with a random one and sigma is restored to its initial value. This is too aggressive: it discards all gradient-history information and restarts exploration from scratch. On rastrigin, 6 resets occurred in 343 iterations.

### 2.5 No gradient momentum

The update `x_{k+1} = x_k - step_size * grad_k` uses only the current noisy gradient estimate. With no momentum (Polyak heavy-ball or Nesterov), the algorithm cannot smooth out gradient noise and is more prone to oscillation.

---

## 3. Proposed improvements

### 3.1 Adaptive sigma with exponential smoothing (HIGH priority)

**Replace** the multiplicative bang-bang rule with an exponential-moving-average (EMA) adaptation:

```
target_sigma = sigma * gamma_sigma^(sign)   # same as now
sigma_new = (1 - beta_sigma) * target_sigma + beta_sigma * sigma_old
```

with `beta_sigma ∈ [0.7, 0.95]`. This damps oscillations while preserving the ability to adapt. Additionally, enforce `sigma_min` and `sigma_max` bounds relative to the initial sigma:

```
sigma_min = ro * sigma_zero        # keep current lower bound
sigma_max = 10 * sigma_zero        # prevent blow-up
```

### 3.2 Smoothed alpha update (HIGH priority)

**Replace** the instant alpha update with an EMA:

```
alpha_target = clip(current_update, k2, k1)
alpha_new = (1 - beta_alpha) * alpha_target + beta_alpha * alpha_old
```

with `beta_alpha ∈ [0.8, 0.95]`. This prevents alpha from oscillating between k2 and k1 every iteration.

Additionally, add a **threshold** on the difference `|r - r_hat|`: only update alpha when the difference is statistically significant (e.g., `|r - r_hat| > 0.01 * |r|`).

### 3.3 Step size clipping (HIGH priority)

Add a maximum step size relative to the problem scale:

```
step_size = sigma / L_nabla
step_size = clip(step_size, 0, max_step)
max_step = kappa * ||x_0||   # or kappa * sigma_zero
```

with `kappa ∈ [1.0, 10.0]`. This prevents the algorithm from taking arbitrarily large steps when L_nabla is small.

### 3.4 Gradient momentum (MEDIUM priority)

Add Polyak heavy-ball momentum:

```
v_{k+1} = mu * v_k + step_size * grad_k
x_{k+1} = x_k - v_{k+1}
```

with `mu ∈ [0.5, 0.9]`. This smooths the noisy gradient estimates and helps push through flat regions.

### 3.5 Soft basis reset (MEDIUM priority)

**Replace** the hard reset with a **partial rotation**: instead of generating a completely new random basis, rotate the current basis by a random orthogonal matrix with a small angle:

```
Q = special_ortho_group.rvs(dim)
basis_new = (1 - theta) * basis_old + theta * Q @ basis_old
basis_new = orth(basis_new)
```

with `theta ∈ [0.1, 0.3]`. This preserves gradient-history information while injecting fresh exploration directions.

### 3.6 Adaptive patience for alpha (LOW priority)

When alpha is at k1 for many consecutive iterations, it means the gradient subspace is not helping. In this case, gradually increase sigma to escape the plateau (simulated annealing-like behaviour):

```
if alpha == k1 for N_consecutive iterations:
    sigma *= 1.05   # gentle increase
```

### 3.7 Restart from best point (LOW priority)

When stagnation is detected, instead of just stopping, restart from the best point found with a reduced sigma:

```
if stalled for patience iterations:
    x = x_best
    sigma = sigma_best / 2
    reset stall counter
```

This gives the algorithm a second chance with finer granularity.

---

## 4. Implementation plan

| # | Change | Files affected | Risk | Priority |
|---|--------|---------------|------|----------|
| 1 | Smoothed sigma adaptation | `ashgf.py` | Low | 🔴 HIGH |
| 2 | Smoothed alpha update | `ashgf.py` | Low | 🔴 HIGH |
| 3 | Step size clipping | `ashgf.py` (in `_get_step_size`) | Low | 🔴 HIGH |
| 4 | Gradient momentum | `base.py` + `ashgf.py` | Medium | 🟡 MEDIUM |
| 5 | Soft basis reset | `ashgf.py` | Medium | 🟡 MEDIUM |
| 6 | Adaptive patience for alpha | `ashgf.py` | Low | 🟢 LOW |
| 7 | Restart from best point | `base.py` | Medium | 🟢 LOW |

Each change will be implemented as a separate commit with its own unit test.

---

## 5. Evaluation criteria

After implementing each change, evaluate on:

- **Convergence speed**: iterations to reach f(x) < tol
- **Final accuracy**: best f(x) achieved
- **Stability**: variance of f(x) after convergence
- **Oscillation count**: number of sigma/alpha sign changes
- **Scalability**: behaviour at dim ∈ {100, 500, 1000, 2000}

The baseline is the current ASHGF on the full benchmark suite (80 functions × 6 dimensions).

---

## 6. Open questions (to revisit with high-dim data)

1. Does the oscillation problem get **worse** or **better** at high dimensions?
2. Is the `dim * (m-1)` function evaluations per iteration still affordable at dim=2000?
3. Should `m` (Gauss-Hermite nodes) scale with dimension?
4. Does the gradient-history buffer size `t` need to scale with dimension?
5. Is the cost of `scipy.linalg.orth` (O(d³)) a bottleneck at dim ≥ 1000?
