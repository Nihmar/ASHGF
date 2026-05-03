# ASHGF-NG: ASHGF Next Generation — A New Gradient-Free Optimiser

> **Status**: design proposal, awaiting implementation and benchmarking.
> **Target**: improve ASHGF on functions where it underperforms (rastrigin, levy, rosenbrock)
>   while preserving its excellent behaviour on well-conditioned problems (sphere, ackley).

---

## 0. Executive Summary

ASHGF (Adaptive Stochastic Historical Gradient-Free) è un ottimizzatore *derivative-free*
che combina:

1. **Stima del gradiente** via quadratura di Gauss-Hermite (ereditato da ASGF).
2. **Campionamento adattivo delle direzioni** che mescola sottospazio dei gradienti
   storici e direzioni casuali (ereditato da SGES).
3. **Adattamento online** dei parametri `σ` (banda di smoothing) e `α` (probabilità
   di direzione casuale).

Sui problemi *ben condizionati* (sphere, ackley) ASHGF raggiunge precisione di
macchina in 20–35 iterazioni.  Su problemi *mal condizionati* o *multi-modali*
(rastrigin, levy, rosenbrock) invece **oscilla**, **ristagna**, e in alcuni casi
viene **superato da GD vanilla** (es. rastrigin: GD 48.8 vs ASHGF 159).

La causa principale è il **controllore bang-bang** che governa `σ` e `α`: una logica
a soglia binaria che produce oscillazioni persistenti, deriva dei parametri agli
estremi, e reset distruttivi della base ortonormale.

**ASHGF-NG** (Next Generation) sostituisce ogni meccanismo binario con un analogo
**continuo**, ispirato alla teoria del controllo e all'ottimizzazione stocastica
moderna.  Inoltre introduce **momento di Nesterov**, **evoluzione geodetica della
base**, e **controllo trust-region del passo** per affrontare direttamente i punti
deboli di ASHGF.

---

## 1. Analisi dei punti deboli di ASHGF

### 1.1 Adattamento binario di σ → oscillazioni

La regola attuale:

```
se max_i |D_i f| / L_i < A  →  σ *= 0.9,  A *= 0.95
se max_i |D_i f| / L_i > B  →  σ *= 1.11, B *= 1.01
altrimenti                   →  A *= 1.02, B *= 0.98
```

è un **controllore a relè** (bang-bang).  Non avendo isteresi né smorzamento,
`σ` oscilla indefinitamente attorno al valore critico.  Su rastrigin e levy
l'oscillazione di `σ` è stata osservata nell'8–15% delle iterazioni.

**Conseguenza**: il passo `σ / L_nabla` oscilla anch'esso, impedendo la
convergenza fine.

### 1.2 Deriva di α agli estremi

L'aggiornamento di `α` è:

```
se r < r_hat  →  α /= 1.1,  clip a k2=0.1
se r >= r_hat →  α *= 1.1,  clip a k1=0.9
```

Quando la funzione è piatta (plateau), `r ≈ r_hat` e il confronto è dominato
dal rumore.  `α` compie una *passeggiata aleatoria* con barriere assorbenti,
raggiungendo `k1=0.9` (90% direzioni casuali) o `k2=0.1` (90% sottospazio
gradiente) e rimanendovi per il 62% delle iterazioni.

**Conseguenza**: l'algoritmo perde la capacità di adattarsi al paesaggio locale.

### 1.3 Reset aggressivo della base

Quando `σ < ρ·σ₀` (tipicamente `σ < 0.01·σ₀`), l'intera base ortonormale
viene rimpiazzata con una casuale e `σ` riportato al valore iniziale.
Questo **distrugge tutta l'informazione accumulata** nel buffer dei gradienti
e nella base stessa.  Su rastrigin, 6 reset in 343 iterazioni.

### 1.4 Assenza di momento

L'aggiornamento `x_{k+1} = x_k - (σ/L_nabla)·∇f(x_k)` usa solo la stima
corrente del gradiente.  Su funzioni *mal condizionate* (rosenbrock) e su
*plateau*, il momento (Polyak heavy-ball o Nesterov) è essenziale per:
- smorzare il rumore di stima del gradiente,
- accumulare inerzia per attraversare regioni piatte,
- accelerare la convergenza in valli strette.

### 1.5 Passo non protetto

Il passo `σ / L_nabla` può crescere arbitrariamente quando `L_nabla` è piccolo
(es. dopo un reset della base o su funzioni piatte).  Su levy si sono
osservati passi da 4.0, che spostano il punto lontano dal minimo.

### 1.6 Quadratura a profondità fissa

ASHGF usa `m=5` nodi di Gauss-Hermite per **ogni** direzione a **ogni**
iterazione, per un totale di `dim × 5` valutazioni di funzione per stima del
gradiente.  In regioni lisce, `m=3` sarebbe sufficiente, risparmiando il 40%
delle valutazioni.

---

## 2. ASHGF-NG: il nuovo algoritmo

ASHGF-NG è definito dalle seguenti sostituzioni, ciascuna descritta in
dettaglio nelle sezioni successive.

| # | Componente ASHGF | Sostituzione ASHGF-NG | Priorità |
|---|-----------------|----------------------|----------|
| 1 | Bang-bang σ | PID-controller con anti-windup | 🔴 CRITICAL |
| 2 | α binario | Filtro Bayesiano (beta-Bernoulli) | 🔴 CRITICAL |
| 3 | Reset hard base | Evoluzione geodetica (Cayley) | 🔴 CRITICAL |
| 4 | Nessun momento | Nesterov + μ adattivo | 🟡 HIGH |
| 5 | Passo `σ/L_nabla` | Trust-region con backtracking | 🟡 HIGH |
| 6 | m=5 fisso | Quadratura adattiva (m ∈ {3,5,7}) | 🟢 MEDIUM |
| 7 | Nessun restart | Restart da best point + σ annealing | 🟢 MEDIUM |

---

## 3. Dettaglio matematico dei componenti

### 3.1 PID Controller per σ (`sigma_controller`)

**Idea**: invece di moltiplicare/dividere σ per costanti fisse, si usa un
controllore Proporzionale-Integrale che insegue un *target* per il rapporto
`r_i = |D_i f| / L_i`.

Sia:

```
r_max(k) = max_i |D_i f(x_k)| / L_i
```

Definiamo il **target** `r* = (A + B) / 2` (centro della banda desiderata)
e l'**errore**:

```
e_k = r_max(k) - r*
```

Il controllore PID calcola il log-σ:

```
log σ_{k+1} = log σ_k - (K_p·e_k + K_i·E_k + K_d·(e_k - e_{k-1}))
```

dove `E_k = E_{k-1} + e_k` è l'integrale dell'errore (con anti-windup:
`E_k = clip(E_k, -E_max, E_max)`).

**Vantaggi**:
- L'azione proporzionale corregge deviazioni istantanee.
- L'azione integrale elimina l'errore a regime (nessuna oscillazione).
- L'azione derivativa anticipa la traiettoria.
- L'anti-windup impedisce l'accumulo quando σ è saturato ai bound.

**Parametri di default**:

| Parametro | Valore | Significato |
|-----------|--------|-------------|
| `K_p` | 0.5 | Guadagno proporzionale |
| `K_i` | 0.05 | Guadagno integrale |
| `K_d` | 0.1 | Guadagno derivativo |
| `E_max` | 2.0 | Saturazione integrale |
| `sigma_min` | `ρ·σ₀` | Bound inferiore (come prima) |
| `sigma_max` | `10·σ₀` | Bound superiore (nuovo) |
| `r_target` | 0.5 | Target per `r_max` |

I bound su σ prevengono blow-up e underflow.

---

### 3.2 Filtro Bayesiano per α (`alpha_filter`)

**Idea**: modelliamo la scelta tra direzioni *gradient-subspace* e *random*
come un processo di Bernoulli con probabilità `α` (random).  Invece di
aggiornare `α` con una regola binaria, manteniamo una **distribuzione a
posteriori Beta(θ₁, θ₂)** su `α`.

Dopo ogni iterazione, osserviamo un "successo" se il sottospazio gradiente
ha performato meglio di quello random (cioè `r < r_hat`), e un "fallimento"
altrimenti.  Ma invece di un Bernoulli netto, usiamo un **segnale sfumato**
(fuzzy):

```
s_k = sigmoid((r - r_hat) / τ)
```

dove `τ > 0` è una temperatura che controlla la discretizzazione:
- `τ → 0`: decisione hard (come l'originale)
- `τ` grande: aggiornamento più smooth

Poi aggiorniamo i parametri della Beta:

```
θ₁ ← γ·θ₁ + (1-γ)·s_k
θ₂ ← γ·θ₂ + (1-γ)·(1-s_k)
```

con `γ ∈ (0, 1)` fattore di decadimento (default `0.95`).  Il valore
puntuale di α è la media della Beta:

```
α_k = θ₁ / (θ₁ + θ₂)
```

**Vantaggi**:
- `α` evolve in modo continuo, senza barriere assorbenti.
- L'incertezza è quantificata dalla varianza della Beta.
- Quando la Beta è molto concentrata (θ₁+θ₂ grande), α è stabile.
- Quando l'evidenza è debole, la Beta è diffusa e α rimane vicino a 0.5.

**Parametri di default**:

| Parametro | Valore | Significato |
|-----------|--------|-------------|
| `γ` | 0.95 | Decadimento esponenziale dei conteggi |
| `τ` | 0.01 | Temperatura della sigmoide |
| `θ₁_init` | 5.0 | Priori iniziali (α₀ = 0.5) |
| `θ₂_init` | 5.0 | Priori iniziali |

---

### 3.3 Evoluzione geodetica della base (`basis_evolution`)

**Idea**: invece di resettare la base quando σ è piccolo, la facciamo
evolvere in modo continuo tramite piccole rotazioni sullo **Stiefel
manifold** `V_d(ℝ^d)`.

Ad ogni iterazione, applichiamo una **trasformata di Cayley** con un
generatore antisimmetrico:

```
B_{k+1} = (I + η·A_k)^{-1}·(I - η·A_k)·B_k
```

dove:
- `A_k = G_k·G_k^T - G_k^T·G_k` è un generatore antisimmetrico (matrice
  `d×d` con `A^T = -A`).
- `G_k` è il buffer dei gradienti storici (matrice `t×d`).
- `η ∈ (0, 0.5)` è il passo di rotazione (default `0.05`).

La trasformata di Cayley garantisce che `B_{k+1}` rimanga ortonormale
(proprietà del gruppo ortogonale).

**Inoltre**, quando σ scende sotto `ρ·σ₀`, invece di resettare completamente,
facciamo una **rotazione più marcata** (`η` aumentato a `0.3`) e riduciamo
σ solo parzialmente:

```
σ ← max(σ, ρ·σ₀) + 0.5·(σ₀ - max(σ, ρ·σ₀))
```

Questo preserva la storia recente pur esplorando nuove direzioni.

**Vantaggi**:
- Continuità: la base evolve senza salti, preservando l'informazione.
- Nessuna perdita del buffer storico.
- La rotazione è guidata dai gradienti stessi (direzioni promettenti).

**Parametri di default**:

| Parametro | Valore | Significato |
|-----------|--------|-------------|
| `η_base` | 0.05 | Rotazione base per iterazione normale |
| `η_reset` | 0.30 | Rotazione base in condizione di near-reset |
| `σ_recovery` | 0.5 | Fattore di recupero di σ (0=reset hard, 1=nessun recupero) |

---

### 3.4 Momento di Nesterov adattivo (`nesterov_momentum`)

**Idea**: aggiungiamo il momento di Nesterov, che è notoriamente ottimale
per funzioni convesse lisce e aiuta molto su funzioni mal condizionate.

```
v_{k+1} = μ_k·v_k + step_k·∇f(x_k + μ_k·v_k)
x_{k+1} = x_k - v_{k+1}
```

Il coefficiente di momento `μ_k` è adattato con una regola euristica:

```
μ_k = μ_min + (μ_max - μ_min)·exp(-|Δf_k| / δ)
```

dove `Δf_k = f(x_k) - f(x_{k-1})` e `δ` è una scala caratteristica.

- Se `|Δf_k|` è grande (discesa rapida) → `μ` è piccolo (non serve inerzia).
- Se `|Δf_k|` è piccolo (plateau / valle stretta) → `μ` è grande (accumula
  inerzia per spingere attraverso).

**Vantaggi**:
- Accelera la convergenza su funzioni mal condizionate (rosenbrock).
- Aiuta ad attraversare plateau (levy).
- Lo schema di Nesterov è più stabile del heavy-ball classico.

**Parametri di default**:

| Parametro | Valore | Significato |
|-----------|--------|-------------|
| `μ_min` | 0.5 | Momento minimo |
| `μ_max` | 0.95 | Momento massimo |
| `δ` | `|f(x₀)|/10` | Scala per l'adattamento |

---

### 3.5 Controllo trust-region del passo (`trust_region_step`)

**Idea**: il passo `σ/L_nabla` può essere troppo ottimistico.  Verifichiamo
la qualità del passo confrontando la **riduzione predetta** con quella
**effettiva**:

```
pred_k = -step_k·‖∇f(x_k)‖²          (riduzione predetta dal modello lineare)
ared_k = f(x_k) - f(x_{k+1})          (riduzione effettiva)
ρ_k = ared_k / max(pred_k, ε)         (rapporto di riduzione)
```

- Se `ρ_k > η_accept` (default `0.1`): il passo è accettato, e step viene
  eventualmente aumentato (`step *= 1.1`).
- Se `ρ_k ≤ η_accept`: il passo è rifiutato, si torna a `x_k` e si riduce
  il passo (`step *= 0.5`) — **backtracking**.
- In ogni caso: `step = clip(step, step_min, step_max)`.

Il `step_max` è vincolato dalla scala del problema:

```
step_max = κ·‖x₀‖  con κ ∈ [1, 10]
```

**Vantaggi**:
- Previene i passi eccessivi che fanno divergere l'algoritmo.
- Il backtracking permette di riprendersi da cattive stime di L_nabla.
- Il meccanismo è standard nella letteratura trust-region.

**Parametri di default**:

| Parametro | Valore | Significato |
|-----------|--------|-------------|
| `η_accept` | 0.1 | Soglia di accettazione del passo |
| `step_min` | `1e-10` | Passo minimo assoluto |
| `κ` | 5.0 | Fattore per step_max relativo a ‖x₀‖ |
| `max_backtracks` | 3 | Numero massimo di backtracking per iterazione |

---

### 3.6 Quadratura adattiva (`adaptive_quadrature`)

**Idea**: il numero di nodi di Gauss-Hermite `m` è scelto in base alla
*liscezza locale* della funzione:

Dopo la stima con `m=3`, calcoliamo la **varianza cross-validata** tra
i nodi pari e quelli dispari:

```
Δ = |D_i^{(even)} - D_i^{(odd)}| / max(|D_i|, ε)
```

Se `Δ > tol_m` → la stima è rumorosa → aumentiamo `m` (5 o 7) per la
prossima iterazione.
Se `Δ < tol_m/2` → la stima è accurata → possiamo ridurre `m` (min 3).

**Vantaggi**:
- Risparmia fino al 40% di valutazioni in regioni lisce.
- Aumenta l'accuratezza in regioni rugose in modo automatico.
- Il costo di `m=3` vs `m=5` è significativo ad alte dimensioni.

**Parametri di default**:

| Parametro | Valore | Significato |
|-----------|--------|-------------|
| `m_min` | 3 | Minimo nodi (dispari) |
| `m_max` | 7 | Massimo nodi (dispari) |
| `tol_m` | 0.1 | Tolleranza per aumento di m |

---

### 3.7 Restart da best point con σ annealing (`smart_restart`)

**Idea**: quando viene rilevato stallo (nessun miglioramento per `patience`
iterazioni), invece di fermarsi, si riparte dal punto migliore trovato con
una σ ridotta:

```
se stall_count >= patience:
    x ← x_best
    σ ← σ_best / 2
    reset stall_count
    (opzionale) aumentare temporaneamente η_base per esplorazione
```

Questo dà all'algoritmo una *seconda chance* con granularità più fine.
Il meccanismo è simile al *simulated annealing* ma applicato solo in
condizioni di stallo.

**Vantaggi**:
- Semplice, basso overhead.
- Può trasformare un run fallito in uno di successo.
- Complementare agli altri meccanismi (non interferisce).

---

## 4. Pseudocodice di ASHGF-NG

```
Algorithm: ASHGF-NG(f, x₀, max_iter)

  # Initialisation
  x ← x₀, x_best ← x₀, f_best ← f(x₀)
  σ ← ‖x₀‖/10, σ₀ ← σ, σ_best ← σ
  B ← random_orthogonal(d)          # base ortonormale
  G ← zeros(t, d)                   # buffer gradienti
  v ← zeros(d)                      # velocità Nesterov
  θ₁ ← 5, θ₂ ← 5                   # parametri Beta per α
  α ← 0.5
  m ← 3                             # nodi GH iniziali
  E ← 0                             # integratore PID

  for k = 1 to max_iter:

    # 1. Adaptive quadrature order
    m ← adapt_m(m, Δ_prev)          # sez. 3.6

    # 2. Build directions (alpha from Beta posterior)
    α ← θ₁/(θ₁+θ₂)
    D ← compute_directions(d, G, α, m, rng)   # sez. 3.2
    B ← QR(D^T)^T

    # 3. Gauss-Hermite gradient estimation with Nesterov look-ahead
    x_look ← x + μ·v
    ∇f ← gauss_hermite(x_look, f, σ, B, m)

    # 4. Nesterov momentum (sez. 3.4)
    μ ← adapt_momentum(f(x) - f_prev)
    step ← trust_region_step(x, ∇f, σ, L_nabla)  # sez. 3.5
    v ← μ·v + step·∇f
    x_new ← x - v

    # 5. Trust-region acceptance (sez. 3.5)
    if accept(x, x_new, ∇f, step):
        x ← x_new
    else:
        step ← step/2; goto 4     # backtracking

    # 6. PID sigma update (sez. 3.1)
    r_max ← max_i |D_i f|/L_i
    e ← r_max - r_target
    E ← clip(E + e, -E_max, E_max)
    log σ ← log σ - (Kp·e + Ki·E + Kd·(e - e_prev))
    σ ← clip(exp(log σ), σ_min, σ_max)

    # 7. Basis evolution (sez. 3.3)
    if σ < ρ·σ₀:
        η ← η_reset; σ ← σ + 0.5·(σ₀ - σ)
    else:
        η ← η_base
    B ← cayley_rotate(B, G, η)

    # 8. Bayesian alpha update (sez. 3.2)
    s ← sigmoid((r - r_hat)/τ)
    θ₁ ← γ·θ₁ + (1-γ)·s
    θ₂ ← γ·θ₂ + (1-γ)·(1-s)

    # 9. Smart restart (sez. 3.7)
    if stall > patience:
        x ← x_best; σ ← σ_best/2; reset stall

    # 10. Bookkeeping
    update G buffer, track best, convergence check

  return x_best, f_best
```

---

## 5. Analisi comparativa prevista

| Funzione | ASHGF attuale | ASHGF-NG atteso | meccanismi chiave |
|----------|--------------|-----------------|-------------------|
| sphere | 1.05e-15 (20 iter) | simile | — |
| ackley | 4.53e-09 (35 iter) | simile | — |
| rastrigin | 1.59e+02 (oscilla) | ~5e+01 | PID σ + soft basis + Nesterov |
| levy | 3.58e-01 (lento) | ~1e-01 | trust-region + PID σ + restart |
| rosenbrock | 2.27e+01 (lento) | ~1e+01 | Nesterov + trust-region |

L'aspettativa è che ASHGF-NG:
- **eguagli** ASHGF su funzioni ben condizionate (sphere, ackley),
- **riduca l'oscillazione** del 90% su rastrigin e levy,
- **acceleri la convergenza** del 30-50% su rosenbrock,
- **elimini i reset distruttivi**, sostituendoli con evoluzione continua,
- **riduca il costo per iterazione** del 20-40% in regioni lisce grazie
  alla quadratura adattiva.

---

## 6. Analisi dei costi computazionali

### 6.1 Costo per iterazione

| Operazione | ASHGF | ASHGF-NG | Variazione |
|------------|-------|----------|------------|
| Valutazioni di f | `d × m` (con m=5 fisso) | `d × m` (m ∈ {3,5,7}) | −40% quando m=3 |
| QR (orthogonalizzazione) | O(d³) ogni iterazione | O(d³) ogni iterazione | uguale |
| Cayley transform | — | O(d³) per (I+A)⁻¹ | +1 O(d³) ogni ~3 iterazioni |
| PID update | — | O(d) | trascurabile |
| Beta update | — | O(1) | trascurabile |

Il costo dominante rimane `O(d × m)` valutazioni di funzione e `O(d³)`
per l'ortogonalizzazione.  ASHGF-NG **non peggiora l'ordine asintotico**.

### 6.2 Overhead aggiuntivo

- **Cayley transform**: richiede la fattorizzazione LU di una matrice
  `d×d`.  Può essere computata ogni `k` iterazioni (es. `k=3`) senza
  degradazione apprezzabile.  Costo: O(d³), paragonabile al QR già
  presente.

- **Trust-region backtracking**: al più 3 valutazioni extra di `f`
  per iterazione (solo in caso di rifiuto).  Tipicamente nessuna.

---

## 7. Piano di implementazione

| # | Componente | File | Rischio | Ordine |
|---|-----------|------|---------|--------|
| 1 | PID σ controller | `src/algorithms/ashgf_ng.rs` | Medio | 1° |
| 2 | Bayesian α filter | `src/algorithms/ashgf_ng.rs` | Basso | 1° |
| 3 | Soft basis evolution (Cayley) | `src/algorithms/ashgf_ng.rs` | Medio | 2° |
| 4 | Nesterov momentum | `src/algorithms/ashgf_ng.rs` | Basso | 2° |
| 5 | Trust-region step + backtracking | `src/algorithms/base.rs` (modifica `optimize`) | Medio | 3° |
| 6 | Adaptive quadrature m | `src/gradient/estimators.rs` | Basso | 3° |
| 7 | Smart restart | `src/algorithms/base.rs` | Basso | 3° |

L'algoritmo sarà implementato come nuovo tipo `ASHGFNG` in un nuovo file
`src/algorithms/ashgf_ng.rs`, riutilizzando il trait `Optimizer` esistente
e le routine di stima gradiente in `src/gradient/`.

Ogni componente avrà il proprio unit test e sarà integrabile incrementalmente.

---

## 8. Criteri di valutazione

Dopo l'implementazione, ASHGF-NG sarà confrontato con ASHGF, ASGF, SGES, GD,
e ASEBO sui seguenti assi:

1. **Velocità di convergenza**: iterazioni per raggiungere `f(x) < tol`.
2. **Accuratezza finale**: miglior `f(x)` raggiunto entro `max_iter`.
3. **Stabilità**: varianza di `f(x)` dopo convergenza, numero di oscillazioni
   di `σ` e `α`.
4. **Robustezza**: percentuale di run che convergono (senza stallo/blow-up).
5. **Scalabilità**: comportamento a `dim ∈ {50, 100, 500, 1000, 2000}`.
6. **Costo**: numero totale di valutazioni di `f` (non solo iterazioni).

Suite di benchmark: **53 funzioni** × **5 dimensioni** × **30 seed**
(statistica significativa).

---

## 9. Possibili estensioni future

### 9.1 Adattamento completamente Bayesiano

Estendere il filtro Beta per `α` a un filtro di Kalman per l'intero stato
`(σ, α, μ)`, con likelihood basata sul miglioramento osservato.

### 9.2 Sottospazio attivo (da ASEBO)

Incorporare PCA incrementale sul buffer dei gradienti per proiettare la
quadratura di Gauss-Hermite solo sulle direzioni del sottospazio attivo,
riducendo il costo da `O(d·m)` a `O(r·m)` con `r ≪ d`.

### 9.3 Secondo ordine (Hessian-free)

Utilizzare le derivate seconde direzionali (già disponibili dalla quadratura
di Gauss-Hermite) per costruire una precondizionatore diagonale.

### 9.4 Scheduling adattivo di `m` (continuo)

Invece di `m ∈ {3,5,7}`, usare una mistura di stime con diverso `m` pesate
per varianza inversa (multi-fidelity gradient estimation).

---

## 10. Domande aperte

1. **L'evoluzione di Cayley introduce instabilità numerica?**  La matrice
   `(I + ηA)` può diventare mal condizionata.  Serve monitoring del numero
   di condizionamento e fallback a QR puro.

2. **Il PID è robusto al rumore di stima del gradiente?**  Il termine
   derivativo amplifica il rumore.  Si può omettere (`K_d=0`) inizialmente.

3. **Il momento di Nesterov con gradiente stimato (non esatto) mantiene
   le proprietà di accelerazione?**  La letteratura suggerisce di sì,
   ma con una riduzione di `μ` rispetto al caso esatto.

4. **Il trust-region backtracking può causare cicli?**  Con `max_backtracks`
   limitato, il ciclo è impedito; il passo viene comunque accettato dopo
   il numero massimo di tentativi.

5. **Quanto costa la Cayley transform rispetto al QR?**  Entrambi sono
   `O(d³)`.  La Cayley richiede una `solve` lineare, il QR una fattorizzazione.
   Per `d ≤ 2000` dovrebbero essere paragonabili.

---

## Riferimenti

- **ASHGF originale**: tesi di laurea in `thesis/`, implementazione in
  `src_old/`, reimplementazione Rust in `src/algorithms/ashgf.rs`.
- **Nesterov momentum**: Nesterov, Y. (1983). "A method for solving the
  convex programming problem with convergence rate O(1/k²)".
- **Trust-region methods**: Conn, Gould, Toint (2000). "Trust-Region Methods".
- **Cayley transform on Stiefel manifold**: Nishimori, Akaho (2005).
  "Learning algorithms utilizing quasi-geodesic flows on the Stiefel manifold".
- **PID control**: Åström, Hägglund (2006). "Advanced PID Control".
- **Beta-Bernoulli bandit**: Russo et al. (2018). "A Tutorial on Thompson
  Sampling".
