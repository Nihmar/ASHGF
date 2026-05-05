# ASHGF: Adaptive Stochastic Historical Gradient-Free Optimization

Repository for the master's thesis:
[Link](https://thesis.unipd.it/handle/20.500.12608/21569)

**Dual implementation** — a Python package (`ashgf/`) and a Rust library (`src/`)
for **derivative-free optimization** using Gaussian smoothing and directional
derivative estimation.  The Python reference is the canonical implementation
(all 30+ algorithm variants).  The Rust port mirrors a subset of the thesis
algorithms (GD, SGES, ASGF, ASHGF) for high-performance throughput.

The deprecated original code `src_old/` is kept for archival purposes only.
The thesis sources live in `thesis/`.

---

## 🦀 Rust Implementation

The Rust library (`src/`) mirrors the thesis algorithms (GD, SGES, ASGF, ASHGF)
with full mathematical fidelity.  It uses:

- [`ndarray`](https://crates.io/crates/ndarray) + OpenBLAS for linear algebra
- [`rayon`](https://crates.io/crates/rayon) for parallel function evaluation
- [`gauss-quad`](https://crates.io/crates/gauss-quad) for Gauss-Hermite quadrature
- [`ndarray-linalg`](https://crates.io/crates/ndarray-linalg) for QR/Cholesky

### Prerequisites

- **Rust** 1.75+ (`rustup` recommended)
- **OpenBLAS** dev headers (`libopenblas-dev` on Debian/Ubuntu)

### Build

```bash
# Release build (with BLAS)
cargo build --release

# Debug build
cargo build
```

### Tests

```bash
# All unit tests
cargo test

# Release mode for realistic timings
cargo test --release

# Golden-file regression tests (requires --release)
cargo test --release -- --ignored

# Micro-benchmarks via Criterion
cargo bench
```

### CLI

```bash
cargo run --release -- benchmark --dims 10,100 --iter 500 --patience 50 --output results
cargo run --release -- stats --function sphere --algos gd,asgf --dim 100 --runs 30
```

See the [Rust section in AGENTS.md](AGENTS.md#rust) for the full reference.

---

## 🐍 Python Implementation (`ashgf/`)

The Python package is the **canonical implementation**.  It includes the thesis
algorithms plus the extensive family of 2-step variants developed and benchmarked
during this project.

### Installation

Requires **Python 3.10+**.

```bash
pip install -e ".[dev]"
```

Or with `uv`:
```bash
uv sync --group dev
```

### Quick Start

```python
from ashgf.algorithms import ASGF2SLV
from ashgf.functions import get_function

f = get_function("sphere")
algo = ASGF2SLV(seed=42)
best, all_vals = algo.optimize(f, dim=100, max_iter=1000)
print(f"Best: {best[-1][1]:.6e}")
```

### CLI

```bash
# List available test functions
python -m ashgf list

# Run a single algorithm
python -m ashgf run --algo asgf-2slv --function sphere --dim 100 --iter 1000

# Full benchmark across all functions and dimensions
python -m ashgf benchmark --dims "10,100" --iter 500 --patience 50 --seed 2003 --jobs 12

# Compare multiple algorithms on a single function
python -m ashgf compare --algos asgf-2s asgf-2sl asgf-2slv --function rastrigin --dim 50
```

### CLI options (`benchmark`)

| Option | Default | Description |
|--------|---------|-------------|
| `--algos` | all | Algorithm names (space-separated, hyphenated) |
| `--dims` | — | Comma-separated dimensions, e.g. `"10,100"` |
| `--iter` | 1000 | Max iterations per run |
| `--patience` | — | Stop after N no-improvement iterations |
| `--seed` | 2003 | Random seed |
| `--output` | `results` | Output directory for CSVs and plots |
| `--jobs` | 1 | Number of parallel workers |
| `--quiet` | — | Suppress progress output |

### Available CLI algorithm names

```
gd sges asebo asgf ashgf                      # thesis (original 5)
asgf-2f asgf-2s asgf-2sa asgf-2sm ...         # 2-step variants
asgf-2sl asgf-2slr asgf-2sls ...              # Lipschitz-weighted variants
asgf-2slv asgf-2slv2 asgf-2slvp ...           # vote variants
```

---

## Algorithm Catalog

This section lists every algorithm implemented in the codebase, grouped by family,
with a mathematical description of its defining mechanism.

---

### Thesis algorithms (implemented in both Python and Rust)

#### GD — Gradient Descent with Gaussian Smoothing

Vanilla steepest descent on a gradient estimate obtained via central Gaussian
smoothing with $M$ random directions $\{d_j\}_{j=1}^M \sim \mathcal{N}(0, I_d)$:

$$\hat{\nabla}_\sigma f(x) = \frac{1}{2\sigma M} \sum_{j=1}^M \bigl[f(x+\sigma d_j) - f(x-\sigma d_j)\bigr] d_j$$

$$x_{t+1} = x_t - \alpha \, \hat{\nabla}_\sigma f(x_t)$$

#### SGES — Self-Guided Evolution Strategies

Mixes a gradient-history direction with random directions.  At iteration $t$,
a fraction $\alpha_t$ of the $M$ directions are drawn along the gradient history
$h_t$; the rest are random:

$$d_j \sim \begin{cases}
\frac{h_t}{\|h_t\|} + \mathcal{N}(0, \eta^2 I), & j < \alpha_t M \\[4pt]
\mathcal{N}(0, I), & \text{otherwise}
\end{cases}$$

The mixing coefficient $\alpha_t$ adapts based on the relative improvement
rate of history-guided vs. random directions.

#### ASGF — Adaptive Stochastic Gradient-Free

The core adaptive algorithm.  Uses Gauss-Hermite quadrature for exact
directional derivatives along an orthonormal basis $\{b_i\}_{i=1}^d$:

$$\hat{\nabla}_i f(x) = \frac{2}{\sigma\sqrt{\pi}} \sum_{k=1}^m w_k p_k \,
f\bigl(x + \sigma p_k b_i\bigr)$$

where $(p_k, w_k)$ are the $m$-point Gauss-Hermite nodes and weights.

The gradient is reconstructed as $\hat{g} = B^\mathsf{T} D$ where
$D_i = \hat{\nabla}_i f(x)$.

Three adaptation mechanisms operate per iteration:

1. **Per-direction Lipschitz estimation**:
   $$L_{i} = \max_{p,q} \frac{|f(x+\sigma p_i b_i) - f(x+\sigma p_q b_i)|}{\sigma |p_i - p_q|}$$

2. **Global Lipschitz EMA**: $L_{\nabla,t} = (1-\gamma_L) \max_i L_{i,t} + \gamma_L L_{\nabla,t-1}$

3. **Sigma adaptation via ratio** $\rho_i = |D_i| / L_i$:
   - $\max\rho < A$: $\sigma \gets \gamma_\sigma \sigma$ (shrink, gradient reliable)
   - $\max\rho > B$: $\sigma \gets \sigma / \gamma_\sigma$ (grow, gradient noisy)
   - Otherwise: widen $[A, B]$ to stabilise

4. **Basis rotation** via Householder reflection: align first basis vector
   with the estimated gradient, $\mathcal{O}(d^2)$.

5. **Reset**: when $\sigma < r_o \sigma_0$ and resets remain, draw a fresh
   random basis and restore $\sigma \gets \sigma_0$.

Learning rate: $\alpha_t = \sigma_t / L_{\nabla,t}$.

#### ASHGF — Adaptive Historical Gradient-Free

Extends ASGF with a gradient-history buffer $H \in \mathbb{R}^{k \times d}$
($k$ = history length).  The SGES direction-sampling mechanism is overlaid:
directions are drawn from a mixture of $H$ (principal components) and random
vectors.  Designed for RL environments where gradient history captures
temporal correlations.

#### ASEBO — Active Subspaces with Evolution Strategies

Performs PCA on the $M \times d$ direction matrix to identify the active
subspace, then projects gradients into the top-$k$ principal components.
Uses a fixed learning rate and fixed smoothing bandwidth $\sigma$.

---

### 2-Step family (ASGF-2* variants)

All **2S** variants extend ASGF with the *frequency-gated step boost*: after
computing the standard base step, they check whether the improvement
"streak" $s_t$ warrants an additional boost factor $k \in [1, 2]$:

Base step:
$$x_{t+1}^{\text{base}} = x_t - \alpha_t \hat{g}_t$$

Streak update (standard):
$$s_t = \begin{cases}
s_{t-1} + 1, & f(x_{t+1}^{\text{base}}) < f(x_t^{\text{base}}) \\
\max(0, s_{t-1} - 1), & \text{otherwise}
\end{cases}$$

Confidence and boost factor:
$$c_t = \min\left(\frac{s_t}{w}, 1.0\right), \qquad
k_t = 1 + c_t \cdot 1.0$$

Big step candidate:
$$x_{t+1}^{\text{big}} = x_t - k_t \alpha_t \hat{g}_t$$

Safety gate (accept big step only when it beats both base and current):
$$f(x_{t+1}^{\text{big}}) < f(x_{t+1}^{\text{base}})
\quad\wedge\quad f(x_{t+1}^{\text{big}}) < f(x_t)$$

The following table lists every 2S variant, its defining modification,
and the mathematical change:

| Algorithm | File | Key Modification | Formula Change |
|-----------|------|------------------|----------------|
| **ASGF-2F** | `asgf_2f.py` | Original 2F (predecessor) | No safety gate: accept if $f(2x) < f(x_t)$ |
| **ASGF-2S** | `asgf_2s.py` | 2F + safety gate + smooth blend | Gate: $f(x^{\text{big}}) < f(x^{\text{base}}) \wedge f(x^{\text{big}}) < f(x_t)$ |
| ASGF-2SA | `asgf_2sa.py` | Adam-style per-coordinate scaling | $\Delta x_i = \alpha \hat{g}_i / \sqrt{v_i + \epsilon}$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)\hat{g}_t^2$ |
| ASGF-2SM | `asgf_2sm.py` | Heavy-ball momentum | $v_t = \beta v_{t-1} + (1-\beta)\hat{g}_t$, $x^{\text{base}} = x_t - \alpha v_t$ |
| ASGF-2SW | `asgf_2sw.py` | Magnitude-weighted streak | $\Delta s = 1 + \min(\frac{|\Delta f|}{|f|}, 2)$ on improvement, decay proportional on loss |
| ASGF-2SR | `asgf_2sr.py` | Restart-on-stall | Perturb $x_{\text{best}}$ by $\sigma \mathcal{N}(0,I)$ when patience exhausted |
| ASGF-2SMA | `asgf_2sma.py` | Adaptive momentum via cosine similarity | $\beta_t = \beta_{\min} + (\beta_{\max}-\beta_{\min})(1 - \text{EMA}(\cos(\hat{g}_t, \hat{g}_{t-1})))$ |
| ASGF-2SMI | `asgf_2smi.py` | Momentum + weighted streak | Combines 2SM and 2SW |
| ASGF-2SMC | `asgf_2smc.py` | Momentum + step-norm clipping | $\|\Delta x\| \gets \max(\|\Delta x\|, C)$ |
| ASGF-2SN | `asgf_2sn.py` | Sigmoid confidence | $c_t = \sigma((s_t/w - 0.5) \cdot \beta)$, $\sigma(z) = 1/(1+e^{-z})$ |
| ASGF-2SQ | `asgf_2sq.py$ | Quadratic interpolation | Fit parabola through $(0, f_t), (1, f^{\text{base}}), (k, f^{\text{big}})$, evaluate at argmin |
| ASGF-2SG | `asgf_2sg.py$ | Relaxed safety gate | At $c > 0.8$: accept if $f^{\text{big}} < f(x_t)$ and $f^{\text{big}} < f^{\text{base}}(1+\epsilon)$ |
| ASGF-2SAW | `asgf_2saw.py$ | Adaptive warmup | $w \gets w \pm 1$ every $N$ iters based on 2x success rate |
| ASGF-2SD | `asgf_2sd.py$ | Decay streak | $s_t \gets 0.7 \cdot s_t$ on regression instead of $s_t^{-1}$ |

---

### 2SL family (Lipschitz-weighted)

Replace the isotropic boost factor $k$ with a **per-direction** boost that
is inversely proportional to the per-direction Lipschitz estimate $L_i$.

$$r_i = \frac{L_i}{\bar{L}}, \qquad
k_i = 1 + c_t \cdot \frac{\bar{L}}{L_i} \quad \text{(full)}$$

where $\bar{L} = \frac{1}{d}\sum_i L_i$ and $L_i$ comes from ASGF's
per-direction estimation (zero extra cost).

Vector form:
$$x_{t+1}^{\text{big}} = x_t - \text{diag}(k) \cdot \alpha_t \hat{g}_t$$

| Algorithm | File | Modification | Weight formula |
|-----------|------|--------------|----------------|
| **ASGF-2SL** | `asgf_2sl.py` | Baseline per-direction | $k_i = 1 + c_t / r_i$ |
| ASGF-2SLS | `asgf_2sls.py$ | EMA-smoothed Lipschitz | $\tilde{L}_i = \gamma \tilde{L}_i + (1-\gamma) L_i$, then standard $k_i$ |
| ASGF-2SLR | `asgf_2slr.py$ | Square-root compression | $k_i = 1 + c_t / \sqrt{r_i}$ (range $[0.45, 2.24]$ vs $[0.2, 5.0]$) |
| ASGF-2SLA | `asgf_2sla.py$ | Adaptive clip | $c_{\text{lip}}$ adjusted via 2x success-rate tracking |
| ASGF-2SLB | `asgf_2slb.py$ | Blend isotropic/aniso | $k = (1-\beta)k^{\text{uni}} + \beta k^{\text{ani}}$, $\beta$ adapts via gradient consistency |
| ASGF-2SLP | `asgf_2slp.py$ | Spread-gated blend | $\beta = \text{clip}((\bar{L}_{\max}/\bar{L} - 1)/(c_{\text{ref}} - 1), 0, 1)$ |
| ASGF-2SLT | `asgf_2slt.py$ | Two-stage fallback | Try uniform first; if fails and spread $>$ threshold, retry anisotropic |
| ASGF-2SLD | `asgf_2sld.py$ | Decay + Lipschitz | Combines 2SD decay streak + 2SL per-direction weighting |

---

### 2SLV family — Vote (the winner)

The key insight from extensive benchmarking: rather than *predicting* which
step direction (isotropic vs. anisotropic) is better, **evaluate both** and
pick the best that passes the safety gate.

**ASGF-2SLV** — "try both, pick best":

$$\text{Candidates} = \begin{cases}
x^{\text{uni}} = x_t - k \alpha_t \hat{g}_t, & k = 1 + c_t \\[4pt]
x^{\text{ani}} = x_t - \text{diag}(k^{\text{ani}}) \alpha_t \hat{g}_t, &
k^{\text{ani}}_i = 1 + c_t \cdot \frac{\bar{L}}{L_i}
\end{cases}$$

Each candidate is accepted only if it passes the safety gate:
$$f(\cdot) < f(x_t^{\text{base}}) \quad\wedge\quad f(\cdot) < f(x_t)$$

The selected step is the valid candidate with the lowest $f$:
$$\text{selected} = \arg\min \{f(\text{candidate}) \mid \text{safety}(\text{candidate})\}$$

If no candidate passes, the base step $x_t^{\text{base}}$ is returned.

Cost: 2 extra function evaluations per confident iteration (one for uniform,
one for anisotropic) vs. 1 for plain 2S.

**Benchmark result**: 95 / 156 wins (60.9%) across 78 test functions
at dimensions 10 and 100, using seed 2003, 500 iterations, patience 50.
The second-best algorithm (ASGF-2SL) achieves 59 wins (37.8%).

| Variant | File | Modification | Cost | Wins |
|---------|------|--------------|------|------|
| **2SLV** | `asgf_2slv.py` | Baseline vote: uniform + aniso | +2 eval | **95 (60.9%)** |
| 2SLV2 | `asgf_2slv2.py` | Triple candidate (+ sqrt) | +3 eval | 69 (44.2%) |
| 2SLVP | `asgf_2slvp.py` | Persistence bias | +2 eval | 77 (49.4%) |
| 2SLVS | `asgf_2slvs.py` | Spread-conditioned bonus | +2 eval | 79 (50.6%) |
| 2SLV2P | `asgf_2slv2p.py` | Triple + persistence | +3 eval | 66 (42.3%) |
| 2SLV2S | `asgf_2slv2s.py` | Triple + spread | +3 eval | 69 (44.2%) |
| 2SLVPS | `asgf_2slvps.py` | Persistence + spread | +2 eval | 77 (49.4%) |
| 2SLV2PS | `asgf_2slv2ps.py$ | Triple + persistence + spread | +3 eval | 66 (42.3%) |

---

### ASHGF-2SLV family — gradient-history meets 2SLV

Transfers the 2SLV vote mechanism to the ASHGF gradient-history framework.
Several variants were tested to overcome ASHGF's structural weaknesses
(warm-up phase, QR-based basis construction, noisy gradient mixing):

| Variant | File | Modification | d=10 | d=100 | Total |
|---------|------|--------------|------|-------|-------|
| **ASHGF-2SLV** | `ashgf_2slv.py` | Baseline ASHGF + 2SLV vote | 20 | 13 | 33 |
| ASHGF-2SLV0 | `ashgf_2slv0.py` | Householder O(d²), no warm-up | 28 | 22 | 50 |
| ASHGF-2SLVA | `ashgf_2slva.py` | alpha=0 (gradient-subspace only) | 23 | 22 | 45 |
| ASHGF-2SLV2G | `ashgf_2slv2g.py$ | Dual gradient (ASGF + ASHGF) | 26 | 27 | 53 |
| ASHGF-2SLV2GD | `ashgf_2slv2gd.py$ | Dual gradient + history decay | 32 | 22 | 54 |
| ASHGF-2SLV2GA | `ashgf_2slv2ga.py$ | Alternate gradients (ASGF/ASHGF per iter) | 22 | 15 | 37 |

None of the ASHGF variants reach ASGF-2SLV's performance (95 wins).
The gradient-history framework consistently underperforms ASGF's
adaptive Householder rotation, regardless of the step-vote mechanism.

---

### Other algorithms

| Algorithm | File | Description |
|-----------|------|-------------|
| ASGF-2A | `asgf_2a.py` | ASGF with $A$ threshold adaptation |
| ASGF-2G | `asgf_2g.py` | ASGF with centred quadrature gradient |
| ASGF-2H | `asgf_2h.py` | ASGF-H hybrid |
| ASGF-2I | `asgf_2i.py$ | ASGF with IGPE-based step |
| ASGF-2J | `asgf_2j.py$ | ASGF with joint-variance step |
| ASGF-2P | `asgf_2p.py$ | ASGF with projected step perturbation |
| ASGF-2T | `asgf_2t.py$ | ASGF with trust-region step |
| ASGF-2X | `asgf_2x.py$ | ASGF-2 variant (ablation) |
| ASGF-CD | `asgf_cd.py$ | Coordinate descent |
| ASGF-LS* | `asgf_ls*.py$ | Line search variants (5 versions) |
| ASGF-M | `asgf_m.py$ | Momentum |
| ASGF-RS | `asgf_rs.py$ | Restart-based |
| ASGF-SS | `asgf_ss.py$ | Step-size adaptation |
| ASGF-AQ | `asgf_aq.py$ | Adaptive quadrature |
| ASGF-BW | `asgf_bw.py$ | Bandwidth adaptation |
| ASGF-C | `asgf_c.py$ | Constrained variant |
| ASGF-HX | `asgf_hx.py$ | History-based |
| ASHGF-2F | `ashgf_2f.py$ | ASHGF + 2-step |
| ASHGF-2FD | `ashgf_2fd.py$ | ASHGF + 2-step with damping |
| ASHGF-2SMA | `ashgf_2sma.py$ | ASHGF + adaptive momentum |
| ASHGF-2SLV | `ashgf_2slv.py$ | ASHGF + 2SLV vote |
| ASHGF-2SLV0 | `ashgf_2slv0.py$ | ASHGF-2SLV with Householder O(d²) |
| ASHGF-2SLVA | `ashgf_2slva.py$ | ASHGF-2SLV with alpha=0 |
| ASHGF-2SLV2G | `ashgf_2slv2g.py$ | ASHGF-2SLV with dual-gradient vote |
| ASHGF-2SLV2GD | `ashgf_2slv2gd.py$ | Dual-gradient + history decay |
| ASHGF-2SLV2GA | `ashgf_2slv2ga.py$ | Alternating ASGF/ASHGF gradients |
| ASHGF-2X | `ashgf_2x.py$ | ASHGF-2 ablation |
| ASHGF-D | `ashgf_d.py$ | ASHGF with damping |
| ASHGF-NG | `ashgf_ng.py$ | ASHGF without gradient history |
| ASHGF-S | `ashgf_s.py$ | ASHGF simplified |

---

## Benchmark results (final)

78 test functions × 2 dimensions (10, 100) = 156 combos.

Seed 2003, 500 iterations, patience 50, `--jobs 12`.

```
Algorithm         Wins   Win%   vs ASGF
ASGF-2SLV          95   60.9%   +3.7x
ASGF-2SL           59   37.8%   +2.3x
ASGF-2S            56   35.9%   +2.2x
ASGF               26   16.7%   baseline
GD                 10    6.4%
ASEBO/SGES          0    0.0%
```

---

## Project Structure

```
ashgf/
├── __init__.py
├── __main__.py
├── benchmark.py              # Benchmark orchestration + plotting
├── algorithms/
│   ├── __init__.py
│   ├── base.py               # BaseOptimizer (template method pattern)
│   ├── gd.py / sges.py / ... # Thesis algorithms
│   ├── asgf.py               # Core adaptive gradient-free
│   ├── asgf_2s.py            # 2S: frequency-gated step boost
│   ├── asgf_2s*.py           # 20+ 2S/2SL/2SLV variants
│   ├── asgf_2slv.py          # 2SLV: try both, pick best (winner)
│   └── ...
├── functions/
│   ├── __init__.py
│   ├── classic.py / extended.py / benchmark.py / rl.py
├── gradient/
│   ├── estimators.py         # Gauss-Hermite, Gaussian Smoothing
│   └── sampling.py           # Direction sampling
├── utils/                    # Logging
└── cli/run.py                # CLI entry point (argparse)
src/                          # Rust implementation (thesis subset)
src_old/                      # ⚠️ Deprecated — archival only
thesis/                       # LaTeX sources + PDF
```

## License

See [LICENSE](LICENSE).

## Thesis

All LaTeX sources are in `thesis/`. The compiled PDF is at
`thesis/dissertation.pdf`.
