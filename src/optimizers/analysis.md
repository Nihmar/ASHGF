# Analysis: Optimizer Implementations vs. Thesis

## 1. ES (`gd.py`) — Algorithm 2

### Correctness
The ES implementation is **mostly correct**. The gradient estimator follows eq. 2.4:
`∇Fσ(x) ≈ 1/(2σn) Σ ξᵢ(F(x+σξᵢ) - F(x-σξᵢ))` with `n = dim`.

The update rule `x ← x - lr * grad` and the division by `2 * sigma * dim` both match the thesis.

**No bugs found.**

### Performance
The main bottleneck is the Python-level `for` loop over `dim` directions inside `grad_estimator`. Each iteration does element-wise operations that NumPy handles efficiently, but the loop overhead is unnecessary. The entire gradient can be assembled as a single matrix–vector product: `grad = diffs @ directions / (2σn)`, where `diffs` is a vector of `(F⁺ - F⁻)` values and `directions` is the `(n, dim)` matrix.

---

## 2. SGES (`sges.py`) — Algorithm 5

### Correctness — CRITICAL BUG

There is a **critical bug** in `grad_estimator` that makes the SGES-specific direction sampling completely inoperative:

```python
if sges:
    directions, M = self.compute_directions_sges(dim, G, alpha)
directions = self.compute_directions(dim)  # ← OVERWRITES the SGES directions!
```

The line `directions = self.compute_directions(dim)` runs unconditionally, so the SGES directions computed on the previous line are immediately thrown away. The fix is to put it in an `else` branch:

```python
if sges:
    directions, M = self.compute_directions_sges(dim, G, alpha)
else:
    directions = self.compute_directions(dim)
```

**Impact**: With this bug, SGES degenerates into plain ES after the warmup — the gradient-dependent direction sampling never actually happens.

### Second bug: seed reset inside `grad_estimator`

```python
def grad_estimator(self, x, f, G=None, sges=False, alpha=0):
    np.random.seed(self.seed)  # ← resets seed every call!
```

Resetting the seed every iteration makes all random directions identical across iterations, defeating Monte Carlo sampling. This line should be removed.

### Performance
Same vectorization opportunity as ES — replace the direction loop with a matrix product.

---

## 3. ASEBO (`asebo.py`) — Algorithm 3

### Correctness — GRADIENT NORMALIZATION BUG

The gradient is divided by `2 * sigma` but **not** by `n_samples`. The thesis (Algorithm 3, line 11) specifies:

> `∇ = 1/(2nₜσ) Σⱼ (F(xₜ+gⱼ) - F(xₜ-gⱼ)) gⱼ`

The code:
```python
grad /= (2 * self.sigma)  # missing / n_samples
```

Should be:
```python
grad /= (2 * self.sigma * n_samples)
```

**Impact**: The gradient magnitude scales linearly with `n_samples`, which means the effective learning rate changes depending on the number of samples. This can make the optimizer diverge or converge much too slowly depending on the phase.

### Performance
The Cholesky sampling `l.dot(standard_normal(len(x)))` can be batched: generate the full `(n_samples, dim)` matrix of standard normals and multiply by `L.T` in one shot.

---

## 4. ASGF (`asgf.py`) — Algorithm 6 + 7

### Correctness — Lipschitz estimation too weak

The thesis defines (eq. 3.1):

> Lⱼ = max_{(i,k) ∈ I} |[F(x+σpᵢξⱼ) - F(x+σpₖξⱼ)] / [σ(pᵢ-pₖ)]|

where I = {(i,k) : |i−⌊m/2⌋−1| ≠ |k−⌊m/2⌋−1|}.

The code only uses **adjacent pairs** `(k, k+1)`:
```python
for k in range(len(points[i]) - 1):
    denom = sigma * (points[i][k + 1] - points[i][k])
    value = abs((evaluations[i][k + 1] - evaluations[i][k]) / denom)
```

This is a **subset** of the full pair set I, meaning the Lipschitz constant is systematically underestimated. This makes the adaptive learning rate `lr = σ/L∇` too large, risking instability.

### Performance
The quadrature evaluations can be restructured: precompute all perturbation points `x + σ·p_k·basis[j]` as a 2D array and evaluate them in a batch. The Lipschitz pair set I should be precomputed once rather than recomputed every call.

---

## 5. ASHGF (`ashgf.py`) — Algorithms 8 + 9

### Correctness — Multiple Issues

**Bug 1: Pair index set construction**

The `buffer` construction has a subtle Python bug:

```python
if [i, j] or [j, i] not in buffer:
```

In Python, `[i, j]` is a non-empty list, which is always truthy. So `[i, j] or [j, i]` evaluates to `[i, j]` (short-circuit), and then `[i, j] not in buffer` is checked. The `[j, i]` duplicate check never runs. The correct form would be:

```python
if [i, j] not in buffer and [j, i] not in buffer:
```

In practice this doesn't cause wrong results (it just adds some duplicate pairs to `buffer`), but it means the code doesn't match the author's intent.

**Bug 2: L∇ computation fragility**

The L∇ update has a try/except that silently falls back:
```python
try:
    if M > 0 and len(lipschitz_coefficients) > 0:
        L_nabla = (1 - gamma_L) * np.max(lipschitz_coefficients[:M]) + gamma_L * L_nabla
    else:
        L_nabla = max(lipschitz_coefficients) ...
except:
    L_nabla = max(lipschitz_coefficients) ...
```

The fallback uses `max(lipschitz_coefficients)` (all directions) instead of `max(lipschitz_coefficients[:M])` (only gradient-subspace directions), which contradicts the thesis (eq. 3.2 specifies `L_G` is the max over the M gradient-subspace directions).

**Bug 3: Basis shape confusion in subroutine**

The `orth()` call returns a `(dim, rank)` matrix (columns are orthonormal), but the rest of the code assumes rows are basis vectors. The complementing logic `basis = np.concatenate((basis.T, v.T))` transposes inconsistently depending on the path.

### Performance
Same Gauss-Hermite vectorization applies. The `compute_directions_sges` function re-creates the covariance matrix every call — this is unavoidable since G changes, but the binomial sampling for M can replace the explicit loop:

```python
# Original: loop over dim, calling np.random.choice each time
M = np.random.binomial(dim, 1 - alpha)  # single call, same distribution
```

---

## Summary of Bugs

| File | Severity | Description |
|------|----------|-------------|
| sges.py | **CRITICAL** | SGES directions overwritten — algorithm degenerates to plain ES |
| sges.py | HIGH | Seed reset every grad call — identical directions each iteration |
| asebo.py | HIGH | Missing `/n_samples` — gradient magnitude scales with sample count |
| asgf.py | MEDIUM | Lipschitz estimation uses only adjacent pairs, not full set I |
| ashgf.py | MEDIUM | Pair index set bug (`or` short-circuit) — adds duplicate pairs |
| ashgf.py | MEDIUM | L∇ fallback uses all directions instead of only gradient-subspace |
| ashgf.py | LOW | Basis shape/transpose inconsistency in subroutine |

## Performance Improvements Applied

1. **Vectorized gradient assembly**: replaced `for i in range(dim): grad += diff * d` loops with `diffs @ directions` matrix products (ES, SGES, ASEBO).
2. **Batched Cholesky sampling**: replaced per-direction `l.dot(z)` with `z @ L.T` (ASEBO).
3. **Precomputed constants**: Gauss-Hermite nodes/weights and pair index set I computed once, not every iteration (ASGF, ASHGF).
4. **Replaced explicit loops with `np.random.binomial`** for counting gradient-subspace directions (SGES, ASHGF).
5. **Eliminated unnecessary dict storage** of all steps — only track current and best values.
6. **Removed redundant `np.random.seed`** calls inside methods.
