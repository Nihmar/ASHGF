# PORTING_RUST.md — Piano di conversione ASHGF da Python a Rust

## 1. Panoramica

Il porting converte l'implementazione Python di riferimento (`ashgf/`) a Rust stabile
(edizione 2021).  L'architettura ricalca la struttura modulare di `ashgf/`,
sfruttando i tratti (`trait`) al posto dell'ereditarietà e
`ndarray` come sostituto di NumPy.  Il codice originale deprecato `src_old/`
non è stato usato come riferimento per il porting.

**Obiettivi**:
- Fedeltà matematica completa (inclusa correzione del fattore `√2` nella
  quadratura di Gauss-Hermite, cfr. § 7)
- Performance almeno 2× superiori su carichi CPU-bound
- CLI equivalente a `ashgf` con sottocomandi `run`, `compare`, `benchmark`,
  `stats`, `list`
- Riproducibilità deterministica (seed fissi, nessuna dipendenza da stato
  globale)

---

## 2. Struttura del progetto

```
ASHGF/
├── Cargo.toml
├── PORTING_RUST.md              ← questo file
├── src/
│   ├── main.rs                  # entry-point CLI
│   ├── lib.rs                   # re-export pubblici
│   │
│   ├── algorithms/
│   │   ├── mod.rs
│   │   ├── base.rs              # trait Optimizer + OptimizeOptions/Result
│   │   ├── gd.rs                # GD
│   │   ├── sges.rs              # SGES
│   │   ├── asebo.rs             # ASEBO
│   │   ├── asgf.rs              # ASGF
│   │   └── ashgf.rs             # ASHGF
│   │
│   ├── gradient/
│   │   ├── mod.rs
│   │   ├── estimators.rs        # gaussian_smoothing, gauss_hermite_derivative
│   │   └── sampling.rs          # compute_directions*, _random_orthogonal
│   │
│   ├── functions/
│   │   ├── mod.rs               # registry + get_function / list_functions
│   │   ├── classic.rs           # sphere, rastrigin, ackley, ...
│   │   ├── extended.rs          # rosenbrock, white_holst, ...
│   │   ├── benchmark.rs         # perturbed_quadratic, raydan, ...
│   │   └── rl.rs                # Pendulum, CartPole (dietro feature flag)
│   │
│   ├── cli/
│   │   ├── mod.rs
│   │   └── args.rs              # clap derive structs
│   │
│   ├── benchmark/
│   │   ├── mod.rs
│   │   ├── runner.rs            # benchmark / statistics loop
│   │   └── plot.rs              # salvataggio CSV/JSON per plotting
│   │
│   └── utils/
│       ├── mod.rs
│       ├── parallel.rs          # rayon helper
│       └── rng.rs               # SeededRng wrapper
│
├── tests/
│   ├── integration/
│   │   ├── algorithms.rs
│   │   ├── gradient.rs
│   │   ├── functions.rs
│   │   └── sampling.rs
│   └── regression/              # golden-file tests per riproducibilità
│
└── benches/
    └── benchmark.rs             # criterion benchmarks
```

---

## 3. Dipendenze (Cargo.toml)

| Ruolo                        | Crate                       | Versione   | Note                                      |
|------------------------------|-----------------------------|------------|-------------------------------------------|
| Array n-dimensionali         | `ndarray`                   | `0.16`     | Array + algebra lineare                   |
| Algebra lineare              | `ndarray-linalg`            | `0.16`     | QR, Cholesky, SVD, eigendecomposition     |
| Statistiche su array         | `ndarray-stats`             | `0.6`      | `mean`, `std`, `min` su assi              |
| Random numbers               | `rand`                      | `0.8`      | RNG core                                  |
| Distribuzioni                | `rand_distr`                | `0.4`      | `Normal`, `Binomial`, `StandardNormal`    |
| Quadratura Gauss-Hermite     | `gauss-quad`                | `0.1`      | Nodi e pesi (impl. Golub-Welsch)          |
| PCA incrementale             | `smartcore`                 | `0.3`      | `IncrementalPCA` (dietro feature `pca`)   |
| Parallelismo                 | `rayon`                     | `1.10`     | `par_iter`, `par_bridge`                  |
| CLI                          | `clap`                      | `4.5`      | Derive macros                             |
| Logging                      | `tracing` + `tracing-subscriber` | `0.1` | Log strutturato                        |
| Serializzazione              | `serde` + `serde_json`      | `1`        | Salvataggio risultati                     |
| CSV                          | `csv`                       | `1.3`      | Output benchmark                          |
| RL environments              | `gymnasium` binding custom  | –          | Feature `rl` (opzionale)                  |
| Plotting                     | _delegato a script Python_  | –          | Salva CSV/JSON; plotting esterno          |
| Test fixtures                | `rstest`                    | `0.21`     | Fixture parametrici                       |
| Benchmarking                 | `criterion`                 | `0.5`      | Micro-benchmark                           |
| Error handling               | `thiserror` + `anyhow`      | `1` / `1`  | Errori typed + contestuali                |
| Matematica                   | `approx`                    | `0.5`      | `assert_abs_diff_eq!` nei test            |

**Feature flags**:
- `default = ["pca"]`
- `pca` → abilita `ASEBO` (richiede `smartcore`)
- `rl` → abilita ambienti RL (richiede binding `gymnasium`)
- `plot` → (riservato) eventuale plotting nativo con `plotters`

---

## 4. Architettura — Dal Template Method ai Tratti

### 4.1 Tratto `Optimizer`

```rust
use ndarray::Array1;

/// Opzioni condivise da tutti gli ottimizzatori.
#[derive(Debug, Clone)]
pub struct OptimizeOptions {
    pub max_iter: usize,
    pub maximize: bool,
    pub patience: Option<usize>,
    pub ftol: Option<f64>,
    pub log_interval: usize,
}

/// Risultato di una ottimizzazione.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    pub best_values: Vec<(Array1<f64>, f64)>,
    pub all_values: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Tratto principale.  Fornisce un default per `optimize` che chiama
/// i metodi astratti / hooks secondo il pattern Template Method.
pub trait Optimizer {
    /// Stima il gradiente ∇f(x).
    fn grad_estimator(&mut self, x: &Array1<f64>, f: &dyn Fn(&Array1<f64>) -> f64)
        -> Array1<f64>;

    /// Restituisce la step-size corrente.
    fn step_size(&self) -> f64;

    /// Hook chiamato prima del loop principale.
    fn setup(&mut self, _f: &dyn Fn(&Array1<f64>) -> f64, _dim: usize, _x: &Array1<f64>) {}

    /// Hook chiamato dopo ogni iterazione.
    fn post_iteration(
        &mut self,
        _iteration: usize,
        _x: &Array1<f64>,
        _grad: &Array1<f64>,
        _f_val: f64,
    ) {
    }

    /// Metodo template — raramente sovrascritto.
    fn optimize(
        &mut self,
        f: &dyn Fn(&Array1<f64>) -> f64,
        dim: usize,
        x_init: Option<&Array1<f64>>,
        options: &OptimizeOptions,
        rng: &mut impl Rng,
    ) -> OptimizeResult {
        // ... implementazione analoga a BaseOptimizer.optimize ...
    }
}
```

### 4.2 Vantaggi rispetto a Python

- **Nessuna GIL**: `rayon` parallelizza le valutazioni di `f` su più thread
- **Monomorfizzazione**: le chiamate a `f` possono essere inline-ate se `f` è
  un closure concreto (zero-cost abstraction)
- **Nessuna allocazione implicita**: `Array1` è un tipo concreto con layout
  compatto in memoria
- **Errori a compile-time**: dimenticare di chiamare `setup` non è possibile
  — il compilatore forza l'inizializzazione tramite builder o `new`

---

## 5. Piano modulare — Dettaglio per file

### 5.1 `algorithms/base.rs` — Tratto `Optimizer` e `OptimizeResult`

**Da portare**: `BaseOptimizer.optimize()`

**Note**:
- La logica di early-stopping (`patience`, `ftol`) va riprodotta fedelmente.
- Il controllo `NaN`/`Inf` sul gradiente e su `x` usa `Array1::mapv` con
  `f64::is_finite`.
- Il seed è incapsulato in un wrapper `SeededRng` (cfr. `utils/rng.rs`) che
  il chiamante passa esplicitamente.

### 5.2 `algorithms/gd.rs` — GD

**Stato**: banale.  `step_size()` restituisce `lr` fissato.  
`grad_estimator` chiama `gaussian_smoothing`.

### 5.3 `algorithms/sges.rs` — SGES

**Da riprodurre**:
- Buffer circolare `G_buffer: Array2<f64>` di dimensione `t × dim`
- `compute_directions_sges()` → genera `dim` direzioni, `M` dal sottospazio
  gradiente, resto isotropiche
- Adattamento di `alpha` (con semantica *probabilità di direzione casuale*,
  invertita rispetto alla tesi ma equivalente — documentare nel docstring)
- `gaussian_smoothing` con direzioni a norma unitaria (come da tesi SGES)

### 5.4 `algorithms/asebo.rs` — ASEBO (feature `pca`)

**Da riprodurre**:
- `IncrementalPCA` da `smartcore` (o `linfa-decomposition`)
- Matrice di covarianza *blended*: `Σ = σ·[(α/d)·I + ((1-α)/r)·P_active] + λ·I`
- Campionamento da `N(0, Σ)` via Cholesky (`ndarray-linalg::cholesky`)
- **Nessuna normalizzazione** delle direzioni (preservare distribuzione χ)
- Buffer gradiente con decadimento esponenziale (fattore `0.99`)
- Adattamento `alpha = clip(‖P_ort·g‖ / ‖P_active·g‖, 0, 1)`

### 5.5 `algorithms/asgf.rs` — ASGF

**Da riprodurre**:
- `gauss_hermite_derivative()` con `m` nodi dispari
- Stima costanti di Lipschitz: formula `L_j = max_{i,k∈I} |ΔF|/(σ|p_i-p_k|)`
  con `I` che esclude coppie simmetriche (`mid = m / 2`, corretto).
- Learning rate adattivo: `λ = σ / L_∇`
- Adattamento `sigma` via soglie `A, B` e fattori `A_±, B_±`
- Reset quando `σ < ρ·σ₀`
- Rotazione base: `M[0] = grad/‖grad‖`, resto `N(0,I)`, QR

### 5.6 `algorithms/ashgf.rs` — ASHGF

**Da riprodurre** (combina SGES + ASGF):
- Buffer `G_buffer` di dimensione `t`
- Dopo warm-up (`t` iterazioni): `compute_directions_ashgf` (delega a
  `compute_directions_sges`) + QR per ortonormalizzare
- Durante warm-up: base ortonormale casuale
- `_update_alpha` con logica invertita (documentata)
- Stima GH + Lipschitz + adattamento `sigma` (da ASGF)
- `L_∇` calcolato solo sulle prime `M` direzioni (sottospazio gradiente)

### 5.7 `gradient/estimators.rs`

**Funzioni**:
- `gaussian_smoothing(x, f, sigma, directions, n_jobs) -> Array1<f64>`
  - Formula: `∇ ≈ (1/(2σ·M)) Σ_j (f(x+σd_j) - f(x-σd_j)) · d_j`
- `gauss_hermite_derivative(x, f, sigma, basis, m, value_at_x) -> (Array1<f64>, Array2<f64>, Array1<f64>, Array1<f64>)`
  - **Correzione matematica**: la formula corrente usa `2/(σ√π)` e
    perturbazione `σ·p_k`.  La formula corretta con nodi "fisicisti"
    (peso `e^{-x²}`) è `√2/(σ√π)` con perturbazione `√2·σ·p_k`.
    Alternativamente, si possono usare i nodi "probabilistici"
    (`HermiteE`) che eliminano i `√2`:
    `D_i = (1/σ) · (1/√(2π)) Σ w_k^{(prob)} v_k · F(x + σ·v_k·b_i)`.
    → **Decisione**: usare `gauss-quad` crate con convenzione
    probabilistica, verificando con test su `f(x)=‖x‖²` dove
    `∇f(x) = 2x` è noto analiticamente.
- `estimate_lipschitz_constants(evaluations, points, sigma) -> Array1<f64>`
- `parallel_eval(f, points, n_jobs) -> Vec<f64>` (via `rayon`)

### 5.8 `gradient/sampling.rs`

**Funzioni**:
- `random_orthogonal(dim, rng) -> Array2<f64>`
  - `M = standard_normal((dim, dim))`; QR → Q
- `compute_directions(dim) -> Array2<f64>`
  - `standard_normal((dim, dim))` (righe = direzioni)
- `compute_directions_sges(dim, G, alpha) -> (Array2<f64>, usize)`
  - Covarianza empirica `cov(G.T)`, regolarizzazione `1e-6·I`
  - `choices ~ Binomial(dim, 1-alpha)`
  - `choices` direzioni da `N(0, cov)`, resto da `N(0, I)`
  - Normalizzazione a norma unitaria
- `compute_directions_ashgf(dim, G, alpha, M) -> (Array2<f64>, usize)`
  - Delega a `compute_directions_sges` (compatibilità)

### 5.9 `functions/*.rs` — Funzioni test

Tutte le funzioni Python vanno convertite 1:1.  Pattern:

```rust
/// Registro globale: HashMap<&'static str, fn(&Array1<f64>) -> f64>
static REGISTRY: Lazy<HashMap<&'static str, fn(&Array1<f64>) -> f64>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("sphere", sphere as fn(_) -> _);
    m.insert("rastrigin", rastrigin as fn(_) -> _);
    // ...
    m
});

pub fn get_function(name: &str) -> Option<fn(&Array1<f64>) -> f64> {
    REGISTRY.get(name).copied()
}

pub fn list_functions() -> Vec<&'static str> {
    let mut v: Vec<_> = REGISTRY.keys().copied().collect();
    v.sort();
    v
}
```

**File**:
- `classic.rs`: 13 funzioni (sphere, rastrigin, ackley, griewank, levy,
  schwefel, sum_of_different_powers, trid, zakharov, cosine, sine, sincos,
  relu, softmax)
- `extended.rs`: 19 funzioni
- `benchmark.rs`: ~48 funzioni (da CUTEst / Andrei)
- `rl.rs`: `Pendulum`, `CartPole` (dietro feature `rl`)

### 5.10 `cli/args.rs` — CLI con `clap`

Sottocomandi (identici a Python):

```
ashgf run     --algo gd --function sphere --dim 100 --iter 1000
ashgf compare --algos gd sges --function rastrigin --dim 100
ashgf list
ashgf benchmark --dims 10,100,1000 --iter 1000 --output results/
ashgf stats    --function ackley --dim 100 --runs 30
```

**Struct** (`clap` derive):

```rust
#[derive(Parser)]
#[command(name = "ashgf", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Run { /* ... */ },
    Compare { /* ... */ },
    List,
    Benchmark { /* ... */ },
    Stats { /* ... */ },
}
```

### 5.11 `benchmark/` — Runner e plotting

- `runner.rs`: funzioni `benchmark()` e `statistics()` che iterano su
  algoritmi × funzioni × dimensioni, salvando CSV
- `plot.rs`: salvataggio dati in formato JSON/CSV per plotting esterno
  (Python/matplotlib o `plotters` in futuro)

### 5.12 `utils/`

- `rng.rs`: wrapper `SeededRng` con `new(seed: u64)` che incapsula
  `rand::rngs::StdRng` e permette la riproducibilità.
  ```rust
  pub struct SeededRng {
      pub rng: StdRng,
      pub seed: u64,
  }
  ```
- `parallel.rs`: helper `parallel_eval(f, points, n_jobs) -> Vec<f64>` che
  usa `rayon::par_iter` se `n_jobs > 1`, altrimenti iteratore seriale.

---

## 6. Dettaglio delle strutture dati chiave

### 6.1 Buffer circolare (SGES / ASHGF)

```rust
pub struct CircularBuffer {
    data: Array2<f64>,      // (capacity, dim)
    cursor: usize,
    count: usize,
    capacity: usize,
}

impl CircularBuffer {
    pub fn new(capacity: usize, dim: usize) -> Self { ... }
    pub fn push(&mut self, gradient: &Array1<f64>) { ... }
    pub fn as_slice(&self) -> ArrayView2<f64> { ... }  // prime `count` righe
}
```

### 6.2 Stato adattivo ASGF / ASHGF

```rust
pub struct AdaptiveState {
    pub sigma: f64,
    pub sigma_zero: f64,
    pub a: f64,             // soglia A
    pub b: f64,             // soglia B
    pub r: usize,           // reset rimanenti
    pub l_nabla: f64,       // Lipschitz globale smoothed
    pub lipschitz: Array1<f64>,  // per-direzione
    pub basis: Array2<f64>,      // base ortonormale (righe)
}
```

---

## 7. Correzioni matematiche nel porting

### 7.1 Fattore `√2` nella quadratura di Gauss-Hermite

**Problema**: il codice Python usa `2/(σ√π)` come fattore di scala e `σ·p_k`
come perturbazione, dove `p_k` sono i nodi di Hermite *fisicista* (peso
`e^{-x²}`).  La derivazione corretta dà `√2/(σ√π)` e `√2·σ·p_k`, oppure
equivalentemente si possono usare i nodi *probabilistici* (peso `e^{-x²/2}`)
che eliminano i `√2`.

**Soluzione proposta**:
- Usare il crate `gauss-quad` con la convenzione probabilistica
  (`StandardNormal` weight), ottenendo nodi `v_k` e pesi `w_k`.
- Formula pulita: `D_i ≈ (1/σ)·(1/√(2π)) Σ_k w_k·v_k·F(x + σ·v_k·b_i)`.
- Il fattore `2` nel codice Python è probabilmente un artefatto
  dell'implementazione originale; la nuova implementazione userà la formula
  derivata analiticamente.
- **Validazione**: test su `f(x) = ‖x‖²` (gradiente esatto: `2x`),
  `f(x) = Σ x_i⁴` (gradiente esatto: `4x_i³`), verificando convergenza
  dell'errore in `O(m!)`.

### 7.2 Warm-up di ASHGF

Il codice Python usa una base ortonormale casuale durante il warm-up. La
tesi descrive l'uso di direzioni `N(0, I)` non ortonormalizzate (come SGES).
→ Allineare alla tesi: warm-up con `compute_directions(dim)` (non
ortonormalizzate) e `gaussian_smoothing`, poi passare a GH+ortonormale dopo
`t` iterazioni.

### 7.3 Formula di ASEBO

Mantenere l'approccio *blended covariance* (documentato), ma aggiungere
anche l'implementazione della *miscela probabilistica* originale della tesi
dietro un flag `--asebo-sampling=mixture|blended` per permettere confronti.

---

## 8. Strategia di test

### 8.1 Test di unità

| Modulo          | Cosa testare                                                |
|-----------------|-------------------------------------------------------------|
| `classic`       | `f(x*)` ≈ minimo globale noto (es. `sphere(&[0,0,0]) ≈ 0`) |
| `extended`      | Idem, dove il minimo è noto                                 |
| `estimators`    | `gaussian_smoothing` su `f(x)=‖x‖²` → grad ≈ 2x            |
| `estimators`    | `gauss_hermite_derivative` su `f(x)=‖x‖²` → errore < 1e-6  |
| `estimators`    | `estimate_lipschitz_constants` su `f(x)=‖x‖²` → L ≈ 2      |
| `sampling`      | `random_orthogonal` → `QᵀQ ≈ I`, `det(Q) ≈ ±1`             |
| `sampling`      | `compute_directions_sges` → output shape, norme             |
| `algorithms`    | `GD` su `sphere` in dim=10 → converge a 0 in < 500 iter     |
| `algorithms`    | `ASHGF` su `sphere` in dim=10 → converge a 0                |
| `circular_buffer`| push, slice, wrap-around                                   |

### 8.2 Test di regressione (golden file)

Salvare il percorso di ottimizzazione (sequenza `f(x_i)`) per ogni algoritmo
su `sphere` e `rastrigin` con seed fissi.  Confrontare con i valori Python
(tolleranza `1e-10`).  File in `tests/regression/`.

### 8.3 Property-based testing (con `proptest`)

- Per ogni funzione test, verificare che `f(x)` sia finito per input casuali
- Per `gaussian_smoothing`, verificare che il gradiente stimato punti nella
  direzione di massima discesa locale

---

## 9. Performance e parallelismo

### 9.1 Parallel evaluation

In Python, `_parallel_eval` usa `ThreadPoolExecutor`.  In Rust, `rayon`
offre parallelismo data-parallel a zero overhead:

```rust
use rayon::prelude::*;

pub fn parallel_eval<F>(f: &F, points: &[Array1<f64>], n_jobs: usize) -> Vec<f64>
where
    F: Fn(&Array1<f64>) -> f64 + Sync,
{
    if n_jobs <= 1 || points.len() < 4 {
        points.iter().map(|p| f(p)).collect()
    } else {
        points.par_iter().map(|p| f(p)).collect()
    }
}
```

### 9.2 Ottimizzazioni numeriche

- **Cache GH**: `Lazy<HashMap<usize, (Array1<f64>, Array1<f64>)>>` per nodi
  e pesi (come in Python `_GH_CACHE`)
- **Cache array range**: come `_ARR_CACHE` in Python, usare `Lazy` o
  `once_cell::sync::Lazy`
- **SIMD**: `ndarray` con feature `blas` delega a OpenBLAS; le operazioni
  `ArrayBase::dot` e `matmul` beneficiano automaticamente di vettorizzazione
- **Allocazione**: pre-allocare array in `_setup()` ed evitarne di nuovi nel
  loop caldo

---

## 10. Milestone

| # | Obiettivo                                  | File/Moduli                              | Stima |
|---|--------------------------------------------|------------------------------------------|-------|
| 1 | Scheletro progetto + CLI                   | `Cargo.toml`, `main.rs`, `cli/args.rs`   | 1g    |
| 2 | `functions/*` + test                       | `classic.rs`, `extended.rs`, `benchmark.rs` | 2g |
| 3 | `gradient/sampling.rs` + test              | `sampling.rs`                            | 1g    |
| 4 | `gradient/estimators.rs` + test            | `estimators.rs`                          | 2g    |
| 5 | `algorithms/base.rs` + `gd.rs` + test      | `base.rs`, `gd.rs`                       | 1g    |
| 6 | `algorithms/sges.rs` + test                | `sges.rs`                                | 2g    |
| 7 | `algorithms/asgf.rs` + test                | `asgf.rs`                                | 2g    |
| 8 | `algorithms/ashgf.rs` + test               | `ashgf.rs`                               | 2g    |
| 9 | `algorithms/asebo.rs` + test (feature pca) | `asebo.rs`                               | 2g    |
|10 | `benchmark/runner.rs`                      | `runner.rs`, salvataggio CSV             | 1g    |
|11 | Regression tests (golden file)             | `tests/regression/`                      | 1g    |
|12 | `functions/rl.rs` (feature rl)             | `rl.rs`                                  | 1g    |
|13 | Benchmark + profiling                      | `benches/`, flamegraph                   | 1g    |
|14 | Documentazione (`cargo doc`)               | tutti i moduli                           | 1g    |

**Totale stimato**: ~20 giorni-lavoro per un singolo sviluppatore.

---

## 11. Note finali

1. **Nessuna dipendenza da Python a runtime**: il plotting rimane delegato
   a script Python esterni (i dati sono salvati in CSV/JSON).  In futuro si
   potrà usare `plotters` per grafici nativi.

2. **Feature gate per PCA e RL**: chi non necessita di ASEBO o ambienti RL
   compila senza `smartcore` e senza `gymnasium`, riducendo i tempi di build
   e la superficie di dipendenze.

3. **Riproducibilità**: il seed è sempre esplicito.  Nessun `thread_rng()`
   usato implicitamente.  I test di regressione garantiscono che i risultati
   siano identici (a meno di floating-point) alla reference Python.

4. **Matematica**: il porting è l'occasione per allineare completamente
   l'implementazione alla tesi, correggendo le discrepanze identificate
   nell'analisi di coerenza (fattore GH, warm-up ASHGF, normalizzazione
   ASEBO).
