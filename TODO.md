# TODO — Ottimizzazioni per velocizzare i benchmark

Ordinate per rapporto impatto/sforzo.

---

## 🔴 Priorità 1 — Cambio minimo, impatto massimo

### [x] #1 Sostituire `special_ortho_group.rvs` con `qr(randn)` in ASGF e ASHGF ✅
- **File**: `ashgf/algorithms/asgf.py`, `ashgf/algorithms/ashgf.py`, `ashgf/gradient/sampling.py`
- **Impatto**: 3-10x su ASGF/ASHGF (evita O(d³) SVD ogni iterazione)
- **Fix**: Creato helper `_random_orthogonal(dim, rng)` in `sampling.py` che usa
  `np.linalg.qr(rng.standard_normal((dim, dim)))`. Sostituite tutte le occorrenze
  di `special_ortho_group.rvs` in ASGF e ASHGF. In ASGF, la rotazione della base
  ora usa direttamente QR invece di `special_ortho_group + orth`.

### [x] #2 Sostituire `orth()` (SVD) con `qr()` in ASHGF ✅
- **File**: `ashgf/algorithms/ashgf.py` → `grad_estimator`
- **Impatto**: 2-3x sul `grad_estimator` di ASHGF
- **Fix**: `Q, _ = np.linalg.qr(directions.T); basis = Q.T`

### [x] #3 Cache degli array di indici nelle funzioni benchmark ✅
- **File**: `ashgf/functions/benchmark.py`, `ashgf/functions/classic.py`,
  `ashgf/functions/extended.py`
- **Impatto**: 20-30% sul tempo di valutazione delle funzioni
- **Fix**: Aggiunto `_ARR_CACHE` e helper `_cached_arange(n)` in ciascun modulo.
  Sostituite tutte le occorrenze di `np.arange(1, n+1)` e varianti.

---

## 🟡 Priorità 2 — Impatto alto, sforzo medio

### [x] #4 Parallelizzare `benchmark()` e `statistics()` con `ProcessPoolExecutor` ✅
- **File**: `ashgf/benchmark.py`, `ashgf/cli/run.py`
- **Impatto**: Fino a Nx con N core (le run sono indipendenti)
- **Fix**: Aggiunte `_run_benchmark_task` e `_run_stats_task` picklabili.
  `benchmark()` e `statistics()` ora accettano `n_jobs` (default=1 sequenziale).
  CLI espone `--jobs N`. `benchmark_multi` propaga `n_jobs`.

### [x] #5 Vectorizzare la generazione dei punti nei gradient estimator ✅
- **File**: `ashgf/gradient/estimators.py`
- **Impatto**: 2-5x sulla costruzione dei punti in `gaussian_smoothing` e
  `gauss_hermite_derivative`
- **Descrizione**: I punti perturbati sono generati con loop Python annidati.
  Si può usare broadcasting NumPy: `x[None,:] + sigma_dirs` per generare
  tutte le perturbazioni in una sola operazione.
- **Fix**: Riscrivere `gaussian_smoothing` e `gauss_hermite_derivative` per
  costruire i punti via broadcasting. Adattare `_parallel_eval` per accettare
  array 2D.

### [x] #6 Pre-allocare array nel loop base invece del dict `steps` ✅
- **File**: `ashgf/algorithms/base.py`
- **Impatto**: 20-40% sull'overhead del loop principale
- **Descrizione**: `steps: dict[int, tuple[ndarray, float]]` crea overhead
  di hashing e allocazione. Pre-allocare `np.empty((max_iter+1,))` per i valori.
- **Fix**: Sostituire `steps` dict con array pre-allocati.

### [x] #7 PCA incrementale per ASEBO ✅
- **File**: `ashgf/algorithms/asebo.py`
- **Impatto**: 2-5x su ASEBO
- **Descrizione**: ASEBO chiama `PCA().fit()` sull'intero buffer storico ogni
  iterazione. Usare `IncrementalPCA.partial_fit()` per aggiornare solo con
  l'ultimo gradiente.
- **Fix**: Sostituire `PCA` con `IncrementalPCA`, chiamare `partial_fit`
  incrementalmente.

---

## 🟢 Priorità 3 — Miglioramenti minori

### [x] #8 Evitare `np.exp(x)` ripetuti in `diagonal_5` e simili ✅
- **File**: `ashgf/functions/benchmark.py`
- **Impatto**: 5-10% su funzioni specifiche
- **Descrizione**: `diagonal_5` chiama `np.exp(x) + np.exp(-x)` (due exp).
  Calcolare `e_x = np.exp(x)` e usare `e_x + 1/e_x`.

### [x] #9 Ridurre la frequenza del check di convergenza ✅
- **File**: `ashgf/algorithms/base.py`
- **Impatto**: 5% sull'overhead del loop
- **Descrizione**: `la.norm(x - x_prev)` è O(d). Controllare solo ogni
  k iterazioni o usare `np.max(np.abs(x - x_prev))` (norma infinito).

### [ ] #10 Covarianza incrementale in `compute_directions_sges`
- **File**: `ashgf/gradient/sampling.py`
- **Impatto**: 1.5-2x su SGES/ASHGF
- **Descrizione**: `np.cov(G_arr.T)` ricalcola tutto da zero. Mantenere
  stima running con aggiornamento di Welford.

### [ ] #11 CSV writing asincrono in `benchmark()`
- **File**: `ashgf/benchmark.py`
- **Impatto**: Basso (evita I/O bloccante)
- **Descrizione**: Accumulare i dati in memoria e scrivere alla fine in bulk
  o in un thread separato.
