# ASHGF-NG: Implementation Progress

> **Tracking**: stato dell'implementazione di ASHGF-NG come da `ASHGF_PROPOSAL.md`.
> Ordine: prima Rust (`src/`), poi Python (`ashgf/`).

---

## Stato globale

| Componente | Rust | Python | Test |
|-----------|------|--------|------|
| 1. PID σ controller | ✅ | ✅ | ⬜ |
| 2. Bayesian α filter | ✅ | ✅ | ⬜ |
| 3. Soft basis evolution (Cayley/QR) | ✅ | ✅ | ⬜ |
| 4. Nesterov momentum (μ adattivo) | ✅ | ✅ | ⬜ |
| 5. Trust-region step + backtracking | ✅ | ✅ | ⬜ |
| 6. Adaptive quadrature (m ∈ {3,5,7}) | ✅ | ✅ | ⬜ |
| 7. Smart restart | ✅ | ✅ | ⬜ |
| CLI integration | ✅ | ✅ | — |
| Benchmark integration | ✅ | ✅ | — |

Legenda: ⬜ = da fare, 🔲 = in corso, ✅ = completato

> **⚠️ Bug critici scoperti e risolti (2024):** vedi sezione [Bug Fixes](#bug-fixes) in fondo.

---

## 1. PID σ Controller — `sigma_controller`

**File**: `src/algorithms/ashgf_ng.rs` (Rust), `ashgf/algorithms/ashgf_ng.py` (Python)

**Stato**: ⬜ da implementare

**Note**:
- Sostituisce la moltiplicazione binaria `σ *= 0.9` / `σ *= 1.11`
- Formula: `log σ_{k+1} = log σ_k - (Kp·e_k + Ki·E_k + Kd·(e_k - e_{k-1}))`
- Anti-windup: `E_k = clip(E_{k-1} + e_k, -E_max, E_max)`
- Bound: `σ ∈ [ρ·σ₀, 10·σ₀]`

---

## 2. Bayesian α Filter — `alpha_filter`

**File**: `src/algorithms/ashgf_ng.rs` (Rust), `ashgf/algorithms/ashgf_ng.py` (Python)

**Stato**: ⬜ da implementare

**Note**:
- Sostituisce `α /= 1.1` / `α *= 1.1` con barriere
- Beta-Bernoulli posterior: `θ₁ ← γ·θ₁ + (1-γ)·s_k`, `θ₂ ← γ·θ₂ + (1-γ)·(1-s_k)`
- Segnale fuzzy: `s_k = sigmoid((r - r_hat) / τ)`
- `α_k = θ₁ / (θ₁ + θ₂)`

---

## 3. Soft Basis Evolution — `basis_evolution`

**File**: `src/algorithms/ashgf_ng.rs` (Rust), `ashgf/algorithms/ashgf_ng.py` (Python)

**Stato**: ⬜ da implementare

**Note**:
- Sostituisce il reset hard della base
- QR-based blending: `B_new = QR((1-η)·B_old + η·B_qr)`
- `B_qr` costruita da gradient subspace + random directions
- Quando `σ < ρ·σ₀`: `η = η_reset` (più esplorazione), `σ` recuperato parzialmente

---

## 4. Nesterov Momentum — `nesterov_momentum`

**File**: `src/algorithms/ashgf_ng.rs` (Rust), `ashgf/algorithms/ashgf_ng.py` (Python)

**Stato**: ⬜ da implementare

**Note**:
- `v_{k+1} = μ_k·v_k + step·∇f(x_k + μ_k·v_k)`
- `μ_k = μ_min + (μ_max - μ_min)·exp(-|Δf_k| / δ)`
- `x_{k+1} = x_k - v_{k+1}`

---

## 5. Trust-Region Step — `trust_region_step`

**File**: `src/algorithms/ashgf_ng.rs` (Rust), `src/algorithms/base.rs` (Rust modifica), idem Python

**Stato**: ⬜ da implementare

**Note**:
- `ρ_k = (f(x_k) - f(x_{k+1})) / max(step·‖∇f‖², ε)`
- Se `ρ_k > η_accept`: accetta, eventualmente aumenta step
- Se `ρ_k ≤ η_accept`: rifiuta, backtracking (`step /= 2`)
- `step_max = κ·‖x₀‖`, `max_backtracks = 3`

---

## 6. Adaptive Quadrature — `adaptive_quadrature`

**File**: `src/gradient/estimators.rs` (Rust modifica), `ashgf/gradient/estimators.py` (Python)

**Stato**: ⬜ da implementare

**Note**:
- Cross-validation tra nodi pari e dispari
- `Δ = |D_i^{(even)} - D_i^{(odd)}| / max(|D_i|, ε)`
- Se `Δ > tol_m` → aumenta `m` (5 o 7), se `Δ < tol_m/2` → riduci `m` (min 3)

---

## 7. Smart Restart — `smart_restart`

**File**: `src/algorithms/ashgf_ng.rs` (Rust), `ashgf/algorithms/ashgf_ng.py` (Python)

**Stato**: ⬜ da implementare

**Note**:
- Se `stall_count >= patience`: `x ← x_best`, `σ ← σ_best / 2`, reset contatore
- Complementare al meccanismo esistente di early stopping

---

## 8. CLI Integration

**File**: `src/main.rs`, `src/cli/args.rs` (Rust), `ashgf/cli/run.py` (Python)

**Stato**: ⬜ da implementare

**Note**:
- Aggiungere `AshgfNg` come variante in `AlgoName`
- Aggiungere handler in tutti i comandi (`run`, `compare`, `benchmark`, `stats`)

---

## 9. Benchmark Integration

**File**: `src/benchmark/runner.rs` (Rust), `ashgf/benchmark.py` (Python)

**Stato**: ⬜ da implementare

**Note**:
- Aggiungere ASHGF-NG alla suite di default (o renderlo opzionale via flag)


---


---

## Bug Fixes (post-implementazione)

4 bug critici scoperti durante i primi test su dim=10 e risolti:

| # | Bug | Impatto | Fix |
|---|-----|---------|-----|
| 1 | `post_iteration` riceveva `Array1::zeros(dim)` invece del vero gradiente | Base ortonormale corrotta ad ogni iterazione (blend con rumore casuale) | Passare `grad_accepted` dal loop di `optimize` |
| 2 | `try_smart_restart(&mut x.clone())` operava su un clone | Smart restart mai effettivo | Spostare la logica inline in `optimize` |
| 3 | `grad_estimator` usava `current_alpha` come `mu` di Nesterov | Look-ahead al punto sbagliato | Rimuovere look-ahead da `grad_estimator`; farlo in `optimize` |
| 4 | Nessun campo `self.mu` per il coefficiente di momento | `grad_estimator` non poteva accedere a `mu` corretto | Aggiungere `self.mu`, aggiornarlo a inizio iterazione |

**Lezione**: il `grad_estimator` deve stimare grad f **esattamente al punto che riceve**.
Il look-ahead di Nesterov (y_k = x_k + mu * v_k) va costruito dal chiamante (`optimize`)
prima di invocare `grad_estimator(&x_look, ...)`.
