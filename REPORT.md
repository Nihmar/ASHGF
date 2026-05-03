# REPORT: Analisi del Progetto ASHGF

## 📌 Panoramica

> ⚠️ **Nota**: questo report analizza il codice **originale e deprecato** nella cartella `src_old/`.
> L'implementazione Python di riferimento corrente è nella cartella `ashgf/`.
> Il report è conservato come documentazione storica dei problemi riscontrati e risolti.

Questo report analizza il codice originale nella cartella `src_old/` (deprecata) e la tesi
nella cartella `thesis/`, identificando:
1. **Problemi di programmazione** (struttura, qualità, bottleneck) — tutti risolti in `ashgf/`
2. **Problemi matematici** (traduzione teoria → codice) — tutti corretti in `ashgf/` e `src/`
3. **Test non più validi**
4. **Piano di miglioramento e suite di test** — implementato in `ashgf/` e `src/`

---

## 1. PROBLEMI DI PROGRAMMAZIONE

### 1.1 Classe `Function` monolitica (`src_old/functions.py`)

**Problema**: La classe `Function` contiene oltre 70 metodi (uno per ogni funzione test) più metodi ausiliari (`relu`, `softmax`, `evaluate`). È un "God object" di ~700 righe.

**Impatto**: 
- Difficile da mantenere e testare
- Ogni nuova funzione richiede la modifica della classe
- Il dizionario `functions_list` mappa stringhe a metodi, rendendo fragile il lookup

**Soluzione proposta**:
- Creare un package `functions/` con un file per ogni funzione (o gruppo di funzioni)
- Usare un pattern **Registry** (decoratore `@register_function("nome")`) per popolare automaticamente il catalogo
- Separare le funzioni RL in un modulo dedicato

### 1.2 Stato globale mutabile (`ASHGF.data`, `ASGF.data`)

**Problema**: Dizionari a livello di classe (`ASHGF.data`, `ASGF.data`) vengono mutati durante `optimize()`. Due istanze che eseguono in parallelo condividono lo stato.

```python
# src_old/ashgf.py#L15-28
class ASHGF:
    data = {
        'm': 5,
        'A': 0.1,
        ...
    }
```

E in `optimize()`:
```python
ASHGF.data['sigma_zero'] = norm / 10  # muta lo stato condiviso!
```

**Soluzione proposta**:
- Spostare `data` in attributi d'istanza (`self.data = {...}`) e copiare in `__init__`
- Passare i parametri come argomenti espliciti invece di modificarli via riferimento globale

### 1.3 Import con wildcard (`from functions import *`)

**Problema**: In tutti i file (`asgf.py`, `ashgf.py`, `sges.py`, `gd.py`, `asebo.py`, `profiles.py`, `stat_plots.py`, `testing_stuffs.py`) si usa `from functions import *`.

**Impatto**: Inquina il namespace, impedisce l'analisi statica, rende ambigue le dipendenze.

**Soluzione proposta**:
- Usare `from ashgf.functions import Function` (import esplicito)
- Configurare `__all__` se proprio necessario

### 1.4 Duplicazione massiva di codice

#### 1.4.1 Loop principale `optimize()` duplicato 5 volte

`gd.py`, `sges.py`, `asebo.py`, `asgf.py`, `ashgf.py` condividono la stessa struttura del loop di ottimizzazione:

```
inizializzazione → for i in range(it) → stima gradiente → aggiorna x → 
controllo convergenza → aggiorna parametri → salva best values
```

**Soluzione proposta**:
- Estrarre una classe base astratta `BaseOptimizer` con il template method `optimize()`
- Le sottoclassi implementano solo `grad_estimator()` e `update_parameters()`

#### 1.4.2 `compute_directions_sges()` duplicato in `sges.py` e `ashgf.py`

Codice **identico**. 

**Soluzione**: estrarlo in un modulo condiviso `sampling.py` o nella classe base.

#### 1.4.3 `grad_estimator()` quasi identico in `asgf.py` e `ashgf.py`

Differiscono solo per:
- L'uso di `evaluations` per calcolare `r`/`r_hat` in ASHGF
- Il calcolo di `L_nabla` (max su tutte le direzioni in ASGF, max sulle prime M in ASHGF)

**Soluzione**: unico metodo con parametro opzionale.

#### 1.4.4 File duplicato `sges old.py`

È una versione obsoleta di `sges.py`. **Da eliminare.**

#### 1.4.5 Metodo `liarwhd` definito due volte in `functions.py`

Alle linee ~368 e ~469 con implementazioni *diverse*:
- Prima definizione: `term_1 = 4 * np.sum(-x[0] + x**2)`
- Seconda definizione: `term_1 = 4 * np.sum((x**2 - x[0])**2)`

Una delle due è errata. **Da verificare con la letteratura e rimuovere quella sbagliata.**

### 1.5 Bug critici di programmazione

#### 1.5.1 SGES: `grad_estimator()` sovrascrive le direzioni

```python
# src_old/sges.py#L167-169
if sges:
    directions, M = self.compute_directions_sges(dim, G, alpha)
directions = self.compute_directions(dim)  # ← sovrascrive sempre!
```

Le direzioni speciali SGES non vengono **mai** usate. Bug introdotto probabilmente durante un refactoring.

#### 1.5.2 ASHGF: usa `f(x_{i-1})` invece di `f(x_i)` nel gradient estimator

```python
# src_old/ashgf.py#L120
grad, lipschitz_coefficients, lr, derivatives, L_nabla, evaluations = self.grad_estimator(
    x, ASHGF.data['m'], sigma, len(x), lipschitz_coefficients,
    basis, f, L_nabla, M, steps[i - 1][1])  # ← f(x_{i-1}), non f(x_i)!
```

Il valore `steps[i-1][1]` è `f(x_{i-1})`, ma l'algoritmo ha già aggiornato `x` e dovrebbe usare `f(x_i)` per il nodo centrale della quadratura.

#### 1.5.3 SGES: `np.random.seed(self.seed)` dentro `grad_estimator()`

```python
# src_old/sges.py#L162
def grad_estimator(self, x, f, G=None, sges=False, alpha=0):
    np.random.seed(self.seed)  # ← resetta il seed globale ad ogni chiamata!
```

Questo rende le direzioni **identiche** ad ogni iterazione, vanificando la stocasticità dell'algoritmo.

### 1.6 Gestione errori inadeguata

```python
try:
    r_ = (1 / M) * np.sum([min(evaluations[j]) for j in range(M)])
except:
    r_ = False  # cattura TUTTO, incluso TypeError, KeyError...
```

- `except:` nudo (senza specificare l'eccezione) è anti-pattern
- Non viene loggato nulla
- `r_ = False` come sentinella è fragile (confonde `False` con `0`)

### 1.7 Naming inconsistente

- `grad_estimator` (snake_case) vs `compute_directions_sges` (snake_case) — ok
- `compute_directions` vs `compute_directions_sges` — ok
- Ma: `A_minus`, `A_plus`, `B_minus`, `B_plus`, `gamma_L`, `gamma_sigma` (naming non descrittivo)
- `thresh` vs `threshold` (abbreviazioni inconsistenti)
- `norm` (variabile) vs `np.linalg.norm` (funzione) — shadowing

### 1.8 Assenza di type hints

Nessun file usa annotazioni di tipo. Esempio di miglioramento:

```python
def grad_estimator(
    self, 
    x: np.ndarray, 
    m: int, 
    sigma: float, 
    dim: int, 
    lipschitz_coefficients: np.ndarray, 
    basis: np.ndarray, 
    f: Function, 
    L_nabla: float, 
    value: float
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
```

### 1.9 Path hardcoded e parsing fragile

```python
# src_old/profiles.py
main_folder = path.join('code', 'results', 'profiles')  # path assoluto presupposto
```

```python
# src_old/profiles.py, RL_problems.py
def get_functions():
    lines = []
    with open(path.join('code', 'functions.txt')) as f:  # path fragile
        ...
    for line in lines:
        if '*' in line:  # logica di parsing oscura
            fs.append(line.replace('*', '')[:-1])
```

**Soluzione**: 
- Usare `pathlib.Path` per i path
- Rimuovere `functions.txt` e affidarsi al Registry di funzioni
- Rendere i path configurabili (es. tramite CLI o variabili d'ambiente)

### 1.10 Debug via `print()` invece di `logging`

```python
if debug:
    print(i, 'th iteration - value:', ...)
```

**Soluzione**: usare il modulo `logging` con livelli (`DEBUG`, `INFO`, `WARNING`).

### 1.11 Memory: `steps` dict cresce indefinitamente

```python
steps = {}
steps[0] = [x, f.evaluate(x)]
...
steps[i] = [x, f.evaluate(x)]  # conserva TUTTI gli iterati
```

Per 10000 iterazioni in dimensione 1000, questo occupa ~80 MB. **Soluzione**: opzione per salvare solo ogni k-esimo passo, o restituire solo i best values.

### 1.12 Assenza di struttura a package

```
src_old/
├── functions.py
├── gd.py
├── sges.py
├── sges old.py      ← da eliminare
├── asebo.py
├── asgf.py
├── ashgf.py
├── profiles.py
├── stat_plots.py
├── RL_problems.py
├── testing_stuffs.py
├── functions.txt
├── *.ipynb           ← da eliminare
```

**Struttura proposta**:
```
ashgf/
├── __init__.py
├── algorithms/
│   ├── __init__.py
│   ├── base.py          # BaseOptimizer
│   ├── gd.py
│   ├── sges.py
│   ├── asebo.py
│   ├── asgf.py
│   └── ashgf.py
├── functions/
│   ├── __init__.py      # Registry
│   ├── classic.py       # sphere, rastrigin, ackley, ...
│   ├── extended.py      # extended_rosenbrock, ...
│   ├── benchmark.py     # diagonal_*, perturbed_*, ...
│   └── rl.py            # RLenvironment*
├── gradient/
│   ├── __init__.py
│   ├── estimators.py    # Gauss-Hermite, Gaussian Smoothing
│   └── sampling.py      # compute_directions, compute_directions_sges
├── utils/
│   ├── __init__.py
│   ├── profiling.py
│   └── plotting.py
├── cli/
│   ├── __init__.py
│   └── run.py           # entry point CLI
└── tests/
    ├── __init__.py
    ├── test_functions.py
    ├── test_gradient.py
    ├── test_algorithms.py
    └── conftest.py
```

### 1.13 Notebook Jupyter (`.ipynb`)

I file `notebook_profiles.ipynb`, `rl_analysis.ipynb`, `stats.ipynb`:
- Non sono versionabili in modo efficace
- Non sono testabili
- Contengono probabilmente codice ad-hoc

**Decisione**: rimuovere i notebook, convertire la logica in script Python o moduli testabili.

### 1.14 `testing_stuffs.py` — test ad-hoc

Contiene codice di test non strutturato (istanzia GD e ASGF, li esegue, plotta). Non è una suite di test.

**Da rimuovere** e sostituire con test pytest.

### 1.15 `requirements.txt` incompleto

Manca il versionamento esatto delle dipendenze. Manca `gym` nel formato corretto (ora si chiama `gymnasium`?). Manca `setuptools`/`wheel` per il packaging.

---

## 2. PROBLEMI MATEMATICI

### 2.1 Insieme degli indici `I` per le costanti di Lipschitz (off-by-one)

**Tesi** (eq. subito dopo `Lipschitz constants`):
```
I = { {i,k} ∈ {1,…,m}² : |i - ⌊m/2⌋ - 1| ≠ |k - ⌊m/2⌋ - 1| }
```

Per `m=5`: `⌊5/2⌋ = 2`, quindi `|i-3| ≠ |k-3|`.

**Codice** (`ashgf.py#L176-178`):
```python
if np.abs(i - int(m / 2)) != np.abs(j - int(m / 2)):
    buffer.append([i, j])
```

Per `m=5`: `int(5/2) = 2`, quindi `|i-2| ≠ |j-2|`.

**Discrepanza**: il codice esclude le coppie simmetriche rispetto al centro `2` (0-indice), mentre la tesi esclude quelle simmetriche rispetto a `3` (1-indice? o 0-indice?). Dato che gli indici in Python sono 0-based, se la tesi usa indici 1-based, allora `|i-3| ≠ |k-3|` in 1-based corrisponde a `|i-2| ≠ |j-2|` in 0-based. **Il codice sembra corretto**, ma va documentato esplicitamente il passaggio da 1-based a 0-based.

### 2.2 Logica di aggiornamento di `α` (probabilità di campionamento) invertita

**Tesi**: `α` è la probabilità di campionare dal sottospazio dei gradienti storici. Se le direzioni guidate dai gradienti danno valori di funzione migliori (r < r_hat per minimizzazione), allora `α` dovrebbe **aumentare** per sfruttare meglio quella direzione.

**Codice** (`ashgf.py#L140-146`):
```python
if not r or r < r_hat:
    alpha = min([self.delta * alpha, self.k1])      # aumenta alpha
elif not r_hat or r >= r_hat:
    alpha = max([(1 / self.delta) * alpha, self.k2]) # diminuisce alpha
```

**Docstring di `compute_directions_sges`**:
> `alpha: probability of sampling from gradients`

Ma in `compute_directions_sges`:
```python
choices += int(np.random.choice([0, 1], size=1, p=[alpha, 1 - alpha]))
```
Dove `choices` conta il numero di direzioni **dal sottospazio gradienti** e la probabilità di ottenere `1` (cioè di campionare dal sottospazio) è `1 - alpha`. Quindi `alpha` è in realtà la probabilità di campionare **dallo spazio ortogonale/complementare**, non dal sottospazio gradienti.

**Conseguenza**: quando `r < r_hat` (le direzioni gradient-guided performano **meglio**), il codice **diminuisce** `1-alpha` (aumenta `alpha`), quindi usa **meno** direzioni gradient-guided. Questo è l'**opposto** di quanto desiderato.

**Raccomandazione**: verificare la semantica corretta di `α` nella tesi e allineare il codice. Probabilmente:
- O `alpha` deve essere ribattezzato `beta = 1 - alpha` (probabilità di esplorazione)
- O la logica di update va invertita

### 2.3 Formula della learning rate e `L_nabla`

**Tesi** (eq. `learning rate`):
```
L_∇ ← (1-γ_L) L_G + γ_L L_∇,   λ = σ / L_∇
```
dove `L_G` è il massimo delle costanti di Lipschitz associate alle direzioni campionate da `L_G`.

**Codice ASHGF** (`ashgf.py#L192-193`):
```python
L_nabla = (1 - ASHGF.data['gamma_L']) * np.max(lipschitz_coefficients[:M]) + ASHGF.data['gamma_L'] * L_nabla
```
Usa `np.max(lipschitz_coefficients[:M])` per `L_G`. OK.

**Codice ASGF** (`asgf.py#L163`):
```python
L_nabla = (1 - ASGF.data['gamma_L']) * lipschitz_coefficients[0] + ASGF.data['gamma_L'] * L_nabla
```
Usa solo `lipschitz_coefficients[0]` (la prima direzione, che è quella del gradiente precedente). Questo sembra un'approssimazione più forte. La tesi non distingue tra ASGF e ASHGF su questo punto. **Verificare se intenzionale.**

### 2.4 Normalizzazione del gradiente in GD/SGES/ASEBO vs ASGF/ASHGF

**GD/SGES/ASEBO**: `grad = grad / (2 * self.sigma * dim)` — divide per `dim`
**ASGF/ASHGF**: il gradiente è ricostruito come `Σ derivatives[i] * basis[i]` e ogni derivata direzionale è già stimata con `2/(σ√π) * Σ w_i p_i f(...)`. Non c'è divisione per `dim`.

La differenza è data dal fatto che GD/SGES/ASEBO usano Gaussian Smoothing standard (d-dimensionale) dove il numero di direzioni `M = dim`, mentre ASGF/ASHGF usano Directional Gaussian Smoothing dove ogni componente del gradiente è stimata separatamente. **Matematicamente corretto**, ma va documentato.

### 2.5 Formula della derivata direzionale nella quadratura di Gauss-Hermite

**Codice** (`ashgf.py#L169`):
```python
new_estimate = 2 / (sigma * np.sqrt(math.pi)) * np.sum(p_w_5 * np.array(temp))
```
dove `p_w_5 = p_5 * w_5` (nodi × pesi).

**Formula di riferimento**: La derivata direzionale della Gaussian smoothing è:
```
D_u F_σ(x) = 2/(σ√π) ∫_{-∞}^{∞} t · F(x + σ t u) · e^{-t²} dt
```

La quadratura di Gauss-Hermite approssima `∫ g(t) e^{-t²} dt ≈ Σ w_i g(p_i)`.
Quindi con `g(t) = t · F(x + σ t u)`: `Σ w_i · p_i · F(x + σ p_i u)`.

Il codice calcola `Σ (w_i · p_i) · F(x + σ p_i u)` = `Σ p_w_5[i] · temp[i]`. **Corretto.**

### 2.6 Azzeramento del termine centrale nella quadratura

```python
if int(m / 2) == k:
    evaluation = value
```

Il punto centrale (k=2 per m=5) corrisponde a `p_5[2] = 0` (nodo a 0). La formula usa `w_i · p_i · F(x + σ p_i u)`. Quando `p_i = 0`, il termine è `w_i · 0 · F(x) = 0`, quindi il valore di F(x) non contribuisce. Tuttavia il codice lo calcola comunque (passando `value`). Questo è corretto matematicamente (il termine si annulla per via del nodo nullo) ma è computazionalmente superfluo. Non è un bug, ma l'ottimizzazione di passare `value = f(x_{i-1})` introduce il bug 1.5.2.

### 2.7 Stima delle costanti di Lipschitz direzionali

Il codice stima:
```
L_j = max_{(i,k) ∈ I} | (f(x+σ p_i ξ_j) - f(x+σ p_k ξ_j)) / (σ(p_i - p_k)) |
```

Questo stima la costante di Lipschitz della derivata direzionale `g(t) = f(x + σ t ξ_j)` usando differenze finite tra punti di quadratura. È una stima ragionevole, ma non ha garanzie teoriche di essere un upper bound.

### 2.8 Problemi di dominio per le funzioni test

Molte funzioni test (es. `raydan_1`, `diagonal_1`, `bdexp`) contengono `np.exp(x)` che overflowa per `x > ~709`. I punti iniziali `np.random.randn(dim)` possono produrre valori elevati. **Nessuna validazione** viene fatta.

### 2.9 Funzioni RL: reward massimizzato ma codice di ottimizzazione generico

Le funzioni RL restituiscono una **ricompensa totale** (da massimizzare). Il codice gestisce questo con:
```python
if not 'RLenvironment' in function:
    x = x - lr * grad
else:
    x = x + lr * grad  # gradient ascent per RL
```
Funziona ma è fragile (basta un nome funzione che contiene "RLenvironment" per sbaglio).

### 2.10 M (dimensione del sottospazio) inizializzato a `dim/2`

```python
M = dim  # in ASHGF.__init__? No... 
# In optimize():
M = dim  # alla prima iterazione
# In subroutine (non-historical):
M = int(len(grad) / 2)
```

`M` parte da `dim` (tutte le direzioni), poi nel ramo non-historical diventa `dim/2`. Nella subroutine con `historical=True`, `M` viene impostato da `compute_directions_sges` (numero di direzioni gradient-guided). C'è un salto dimensionale alla prima iterazione dopo il warmup: `M` passa da `dim` a `choices` (~`(1-alpha)*dim`).

---

## 3. TEST NON PIÙ VALIDI

### 3.1 `sges old.py`
Vecchia versione di SGES. **Da eliminare.** Non è un test, ma crea confusione.

### 3.2 `testing_stuffs.py`
Script di test ad-hoc che:
- Usa parametri diversi da quelli della tesi (`lr=0.5, sigma=1` per GD)
- Plotta risultati senza asserzioni
- Non è automatizzato

**Da eliminare** e sostituire con test pytest.

### 3.3 Notebook Jupyter
- `notebook_profiles.ipynb`
- `rl_analysis.ipynb`  
- `stats.ipynb`

**Decisione**: rimuovere. Se contengono logica utile, convertirla in script Python testabili.

### 3.4 `functions.txt`
Non è un test, ma è un file di configurazione obsoleto. Il parsing basato su `*` non è documentato. **Da rimuovere** in favore del Registry.

---

## 4. PIANO DI MIGLIORAMENTO

### Fase 1: Riorganizzazione strutturale (refactoring non invasivo)

1. **Creare la struttura a package** come proposto in §1.12
2. **Estrarre `BaseOptimizer`** con il template method `optimize()`
3. **Separare `Function`** in moduli con Registry pattern
4. **Estrarre `grad_estimator` e `compute_directions`** in moduli condivisi
5. **Eliminare `sges old.py`, i notebook, `testing_stuffs.py`, `functions.txt`**
6. **Sostituire `print` con `logging`**
7. **Aggiungere type hints**
8. **Sostituire wildcard imports con import espliciti**

### Fase 2: Correzione bug

1. **SGES**: fix sovrascrittura direzioni in `grad_estimator()` (bug 1.5.1)
2. **ASHGF**: usare `f(x_i)` non `f(x_{i-1})` nel gradient estimator (bug 1.5.2)
3. **SGES**: rimuovere `np.random.seed()` da `grad_estimator()` (bug 1.5.3)
4. **Duplicato `liarwhd`**: verificare la formula corretta e rimuovere quella errata
5. **Alpha update**: verificare e correggere la logica (issue 2.2)
6. **Fix stato globale**: spostare `data` a livello di istanza

### Fase 3: Miglioramenti matematici

1. **Validazione input**: aggiungere controlli di dominio per le funzioni con `exp`
2. **Costanti di Lipschitz**: documentare l'approssimazione e considerare alternative più robuste
3. **Normalizzazione**: unificare e documentare la normalizzazione del gradiente tra algoritmi
4. **Early stopping**: usare `np.linalg.norm(x - x_prev) < eps` come già fatto, ma renderlo configurabile

### Fase 4: Suite di test

Utilizzare **pytest** con la seguente organizzazione:

```
tests/
├── conftest.py              # fixture condivise
├── test_functions.py        # test per ogni funzione del catalogo
├── test_gradient_estimator.py  # test per gli stimatori di gradiente
├── test_algorithms.py       # test di integrazione per ogni algoritmo
├── test_sampling.py         # test per compute_directions*
├── test_optimizer_base.py   # test per BaseOptimizer
└── regression/
    ├── reference_values.json  # valori di riferimento per regression test
    └── test_regression.py
```

#### 4.1 `test_functions.py`

```python
class TestSphere:
    def test_dim_10(self): ...
    def test_known_minimum(self): ...      # f(0) = 0
    def test_gradient_shape(self): ...     # output shape

class TestRastrigin:
    def test_known_minimum(self): ...      # f(0) = 0
    def test_symmetry(self): ...           # f(x) = f(-x)
```

#### 4.2 `test_gradient_estimator.py`

```python
class TestGaussianSmoothing:
    def test_sphere_gradient(self):
        # Per f(x) = ||x||², il gradiente esatto è 2x
        # Verificare che l'errore ||estimator - 2x|| < tolleranza
        
    def test_linear_function(self):
        # Per f(x) = aᵀx, il gradiente esatto è a
        # Lo stimatore dovrebbe recuperarlo esattamente (a meno di errori numerici)

class TestDirectionalGaussianSmoothing:
    def test_orthonormal_basis(self): ...
    def test_consistency_with_gaussian_smoothing(self): ...
```

#### 4.3 `test_algorithms.py`

```python
class TestOptimizers:
    @pytest.mark.parametrize("algo", [GD, SGES, ASGF, ASHGF, ASEBO])
    @pytest.mark.parametrize("function", ["sphere", "quadratic"])
    def test_convergence_on_convex(self, algo, function):
        """Verifica che tutti gli algoritmi convergano su funzioni convesse semplici"""
        
    def test_deterministic_with_seed(self):
        """Due esecuzioni con stesso seed devono dare stesso risultato"""
        
    def test_best_value_monotonic(self):
        """I best values devono essere non crescenti (per minimizzazione)"""
```

#### 4.4 `test_regression.py`

Salvare valori di riferimento (es. dopo 100 iterazioni su sphere) e verificare che modifiche future non degradino le prestazioni.

### Fase 5: CLI e automazione

#### 5.1 Comando per eseguire i test

```bash
# Esecuzione completa
pytest tests/ -v

# Solo test veloci (senza integrazione)
pytest tests/ -v -m "not slow"

# Con coverage
pytest tests/ --cov=ashgf --cov-report=html

# Regression test
pytest tests/regression/ -v
```

#### 5.2 Entry point CLI

```bash
# Eseguire un algoritmo su una funzione
python -m ashgf run --algo ashgf --function sphere --dim 100 --iter 10000

# Confrontare più algoritmi
python -m ashgf compare --algos gd,sges,asgf,ashgf --function rastrigin --dim 100
```

#### 5.3 CI/CD

Aggiungere `.github/workflows/test.yml`:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --cov=ashgf
```

### Fase 6: Documentazione

1. `README.md` con:
   - Descrizione del progetto
   - Istruzioni di installazione
   - Esempi di utilizzo
   - Riferimenti alla tesi
2. Docstring in formato NumPy/Google per tutte le funzioni pubbliche
3. `CONTRIBUTING.md` per chi volesse contribuire

---

## 5. RIEPILOGO BUG CRITICI (da fixare subito)

| # | Gravità | File | Descrizione |
|---|---------|------|-------------|
| 1 | 🔴 CRITICO | `sges.py:167-169` | Direzioni SGES sovrascritte, algoritmo equivalente a GD |
| 2 | 🔴 CRITICO | `ashgf.py:120` | `value = f(x_{i-1})` invece di `f(x_i)`, errore nella quadratura |
| 3 | 🔴 CRITICO | `sges.py:162` | `np.random.seed()` dentro `grad_estimator` rende le direzioni deterministiche |
| 4 | 🟠 ALTO | `ashgf.py:140-146` | Logica alpha update probabilmente invertita |
| 5 | 🟠 ALTO | `functions.py` | `liarwhd` definito due volte con implementazioni diverse |
| 6 | 🟠 ALTO | `asebo.py` | Formula gradiente: manca divisione per `n_samples` (`grad / (2*sigma)` anziché `grad / (2*sigma*n_samples)`) |
| 7 | 🟠 ALTO | `asebo.py` | Campionamento direzioni non conforme alla tesi (blending unico invece di probabilistic mixture) |
| 8 | 🟡 MEDIO | `ashgf.py`, `asgf.py` | Stato globale mutabile `ASHGF.data`/`ASGF.data` |
| 9 | 🟡 MEDIO | `ashgf.py:178` | Possibile off-by-one nell'insieme I per le costanti di Lipschitz |
| 10 | 🟡 MEDIO | `asebo.py` | Normalizzazione direzioni: norma unitaria invece di norma $\sim \chi(d)$ |

---

## 6. RISPOSTE DEL COMMITTENTE

1. **Versione di Python target**: ✅ Python 3.10+ confermato. Si aggiungeranno type hints completi (incluso `X | Y` per Union, `list[float]` per liste tipizzate, ecc.).
2. **Gym → Gymnasium**: ✅ Si migrerà, ma in una fase successiva. Prioritariamente concentrarsi sui test puramente matematici (funzioni analitiche, non RL). Le funzioni RL verranno temporaneamente disabilitate dai test.
3. **Performance profiles**: ✅ Basta renderli funzionanti. Se il codice è di bassa qualità o non valido, va aggiornato.
4. **Parametri di default**: ✅ La tesi ha la priorit√† assoluta. I parametri nel codice vanno verificati e, se discordanti, allineati alla tesi.
5. **ASEBO**: ✅ Verificare che il codice sia una traduzione corretta della teoria. Deve prevalere la correttezza matematica.

---

## 7. VERIFICA INCROCIATA TESI ↔ CODICE

> **Nota**: la verifica qui sotto è stata condotta sul codice originale `src_old/` (deprecato).
> Tutti i problemi identificati sono stati corretti nell'implementazione di riferimento `ashgf/`
> e nella reimplementazione Rust `src/`.

### 7.1 Parametri di ASHGF

**Tabella della tesi** (Chapter 3, Table of parameters):

| parametro | $m$ | $A$ | $B$ | $A_-$ | $A_+$ | $B_-$ | $B_+$ | $\gamma_L$ | $\gamma_\sigma$ | $r$ | $\rho$ | $k$ | $maxiter$ | $\varepsilon$ |
|-----------|-----|-----|-----|-------|-------|-------|-------|-------------|-------------------|-----|--------|-----|-----------|----------------|
| valore    | 5   | 0.1 | 0.9 | 0.95  | 1.02  | 0.98  | 1.01  | 0.9         | 0.9               | 10  | 0.01   | 50  | 10000     | $10^{-8}$      |

**Codice** (`ashgf.py`):

| parametro tesi | nel codice | valore codice | match |
|----------------|------------|---------------|-------|
| $m$ | `data['m']` | 5 | ✅ |
| $A$ | `data['A']` | 0.1 | ✅ |
| $B$ | `data['B']` | 0.9 | ✅ |
| $A_-$ | `data['A_minus']` | 0.95 | ✅ |
| $A_+$ | `data['A_plus']` | 1.02 | ✅ |
| $B_-$ | `data['B_minus']` | 0.98 | ✅ |
| $B_+$ | `data['B_plus']` | 1.01 | ✅ |
| $\gamma_L$ | `data['gamma_L']` | 0.9 | ✅ |
| $\gamma_\sigma$ | `data['gamma_sigma']` | 0.9 | ✅ |
| $r$ | `data['r']` | 10 | ✅ |
| $\rho$ | `data['ro']` | 0.01 | ✅ |
| $k$ | `self.t` (in `__init__`) | 50 | ✅ |
| $\varepsilon$ | `self.eps` (in `__init__`) | $10^{-8}$ | ✅ |
| — | `data['gamma_sigma_plus']` | 1/0.9 ≈ 1.111 | ✅ (eq. a $1/\gamma_\sigma$) |
| — | `data['gamma_sigma_minus']` | 0.9 | ✅ (eq. a $\gamma_\sigma$) |
| — | `data['threshold']` | $10^{-6}$ | ⚠️ **non usato nella tesi** |
| — | `data['sigma_zero']` | 0.01 | ✅ (sovrascritto in `optimize()`) |

**Verifica**: I parametri corrispondono. `threshold` ($10^{-6}$) non appare nella tabella della tesi: potrebbe essere un parametro inutilizzato o legacy. **Da rimuovere** se non serve.

### 7.2 Verifica della subroutine di aggiornamento parametri

**Tesi** (Algorithm `ASHGFSubroutine`):
- Se $r>0$ e $\sigma < \rho\sigma_0$: reset (nuova base ortonormale casuale, $\sigma=\sigma_0$, $A,B$ ai valori iniziali, $r=r-1$)
- Altrimenti:
  - Se `historical`: genera $d$ direzioni come da historical sampling, ortonormalizza, complementa se necessario
  - Altrimenti: base ortonormale casuale
  - Se $\max_j |\widetilde{\mathcal{D}}^M[...] / L_j| < A$: $\sigma = \sigma \gamma_\sigma$, $A = A \cdot A_-$
  - Se $> B$: $\sigma = \sigma / \gamma_\sigma$, $B = B \cdot B_+$
  - Altrimenti: $A = A \cdot A_+$, $B = B \cdot B_-$

**Codice** (`ashgf.py#L200-240`):
- ✅ Reset block: `if r > 0 and sigma < ro * sigma_zero` → `basis = random`, `sigma = sigma_zero`, `A = data['A']`, `B = data['B']`, `r = r - 1`, `M = int(len(grad)/2)`, ritorna anticipato
- ✅ Non-reset, historical: `basis, M = compute_directions_sges(...)` → `basis = orth(basis)`
- ✅ Non-reset, non-historical: `M = int(len(grad)/2)`, `basis = random`
- ✅ `while basis.shape != (len(grad), len(grad))`: complementa con vettori casuali e ortonormalizza
- ✅ `value < A`: `sigma *= gamma_sigma_minus` (= $\sigma \cdot 0.9$), `A *= A_minus`
- ✅ `value > B`: `sigma *= gamma_sigma_plus` (= $\sigma \cdot 1/0.9 = \sigma / 0.9$), `B *= B_plus`
- ✅ else: `A *= A_plus`, `B *= B_minus`

**Verdetto**: La subroutine è implementata correttamente. C'è un'unica differenza: la tesi parla di "Update $\Xi = \mathcal{D}$" dopo aver stabilito la base, mentre il codice restituisce la base e la riassegna nel chiamante. Equivalente.

### 7.3 Verifica del gradient estimator (Directional Gaussian Smoothing)

**Tesi** (eq. per $\widetilde{\mathcal{D}}^M[G_\sigma(0|x,\xi_j)]$):

$$\widetilde{\mathcal{D}}^M[G_\sigma(0|x,\xi)] = \frac{2}{\sigma\sqrt{\pi}} \sum_{i=1}^M w_i p_i F(x + \sigma p_i \xi)$$

dove $(p_i, w_i)$ sono nodi e pesi della quadratura di Gauss-Hermite con $M=5$.

**Codice** (`ashgf.py#L157-169`):
```python
p_5, w_5 = np.polynomial.hermite.hermgauss(m)  # m=5
p_w_5 = p_5 * w_5
sigma_p_5 = sigma * p_5
...
new_estimate = 2 / (sigma * np.sqrt(math.pi)) * np.sum(p_w_5 * np.array(temp))
```

**Verdetto**: ✅ La formula è implementata correttamente. `hermgauss(m)` restituisce nodi e pesi per $\int f(t) e^{-t^2} dt$. Il codice calcola $\frac{2}{\sigma\sqrt{\pi}} \sum_k w_k p_k F(x + \sigma p_k \xi)$.

### 7.4 Verifica ASEBO contro la teoria

L'algoritmo ASEBO nella tesi (Algorithm 4) e il codice presentano alcune **differenze sostanziali**:

#### 7.4.1 Campionamento delle direzioni

**Tesi**: Si campionano $n_t$ vettori come segue:
- con probabilità $p^t$ da $\mathcal{N}(0, \mathbf{U}^{\text{act}}(\mathbf{U}^{\text{act}})^\top)$
- con probabilità $1-p^t$ da $\mathcal{N}(0, \mathbf{U}^{\perp}(\mathbf{U}^{\perp})^\top)$

I vettori vengono poi **rinormalizzati** in modo che $\|\mathbf{g}_i\|_2 \sim \chi(d)$.

**Codice** (`asebo.py#L130-143`):
```python
cov = (alpha / len(x)) * np.eye(len(x)) + ((1 - alpha) / n_samples) * UUT
cov *= self.sigma
A = np.zeros((n_samples, len(x)))
try:
    l = cholesky(cov, ...)
    for j in range(n_samples):
        A[j] = np.zeros(len(x)) + l.dot(standard_normal(len(x)))
except LinAlgError:
    for j in range(n_samples):
        A[j] = np.random.randn(len(x))
A /= np.linalg.norm(A, axis=-1)[:, np.newaxis]
```

**Differenze**:
1. ❌ Il codice usa una matrice di covarianza **unica** che fonde componente attiva e ortogonale: `cov = α/d · I + (1-α)/n · UUT`. La tesi descrive un **campionamento probabilistico** (o si campiona dallo spazio attivo, o da quello ortogonale).
2. ❌ Il codice **non rinominalizza** per avere norma $\sim \chi(d)$, ma semplicemente **normalizza a norma unitaria** (`A /= norm`). Questo cambia la distribuzione delle direzioni.
3. ❌ Il codice usa `alpha` come peso di blending, mentre la tesi usa $p^t$ come probabilità di selezione. Sono due meccanismi diversi.

#### 7.4.2 Aggiornamento della covarianza

**Tesi**: $\text{Cov}_{t+1} = \lambda \text{Cov}_t + (1-\lambda) \Gamma$, dove $\Gamma = \widehat{\nabla} \widehat{\nabla}^\top$.

**Codice** (`asebo.py#L109-110`):
```python
if i == 1:
    G = np.array(grad)
else:
    G *= 0.99
    G = np.vstack([G, grad])
```

**Differenze**:
4. ❌ La tesi mantiene una matrice di covarianza esplicita e la aggiorna con media mobile esponenziale. Il codice mantiene un **archivio di gradienti** (`G`) e applica un decadimento `0.99` a tutti i gradienti passati prima di aggiungerne uno nuovo. Poi usa `PCA.fit(G)` per estrarre le componenti. Sono approcci equivalenti solo se il PCA sull'archivio decorrelato produce autovettori simili alla matrice di covarianza esponenziale. È un'approssimazione.

#### 7.4.3 Calcolo di $\alpha$ (o $p^t$)

**Tesi**: $p^{t+1}$ è calcolato dall'Algoritmo 2 (subroutine) che esegue un'ottimizzazione interna su un orizzonte $C$ per trovare il valore ottimale di $p$.

**Codice** (`asebo.py#L148-149`):
```python
alpha = np.linalg.norm(np.dot(grad, UUT_ort)) / np.linalg.norm(np.dot(grad, UUT))
```

**Differenze**:
5. ❌ Il codice usa una formula chiusa basata sul rapporto tra la norma del gradiente proiettato sullo spazio ortogonale e quella sullo spazio attivo. La tesi descrive un algoritmo di ottimizzazione interno (Algoritmo 2) con orizzonte $C$, learning rate, regularizer, ecc. Completamente diverso.

#### 7.4.4 Prima fase (full random sampling)

**Tesi**: Per $t < l$ (prime $l$ iterazioni), si campionano $n_t = d$ direzioni da $\mathcal{N}(0, I_d)$.

**Codice** (`asebo.py#L100-106`):
```python
if i >= self.k:
    pca = PCA()
    pca_fit = pca.fit(G)
    ...
    if i == self.k:
        n_samples = 100
else:
    UUT = np.zeros([len(x), len(x)])
    alpha = 1
    n_samples = 100
```

**Differenze**:
6. ⚠️ Il codice per $i < k$ usa `UUT = 0`, `alpha = 1`, e `n_samples = 100` (non $d$, ma 100 fissato). La tesi usa $n_t = d$ (tutte le direzioni). Inoltre il codice non ha un parametro $l$ separato: usa lo stesso `k` sia per la capacità dell'archivio che per il warmup.

#### Giudizio complessivo su ASEBO

| aspetto | tesi | codice | giudizio |
|---------|------|--------|----------|
| Warmup | $l$ iterazioni con $n_t=d$ | $k$ iterazioni con $n=100$, `alpha=1` | ⚠️ Diverso |
| Sampling | Probabilistico (attivo vs ortogonale) | Fusione in un'unica matrice di covarianza | ❌ Diverso |
| Normalizzazione | $\|g_i\| \sim \chi(d)$ | $\|g_i\| = 1$ (norma unitaria) | ❌ Diverso |
| Update covarianza | Media mobile su matrice di covarianza | Archivio gradienti con decadimento + PCA | ⚠️ Approssimazione |
| Update $p^t$ | Algoritmo 2 (ottimizzazione interna) | Formula chiusa (rapporto di norme) | ❌ Diverso |
| Gradiente | $\frac{1}{2 n_t \sigma} \sum (F_+ - F_-) g_i$ | `grad / (2 * self.sigma)` (senza $1/n_t$) | ❌ Manca divisione per $n_t$ |

**Conclusione**: Il codice ASEBO NON è una traduzione fedele dell'algoritmo descritto nella tesi. È una variante che mescola idee dall'implementazione originale dell'autore (Choromanski) con approssimazioni numeriche. **Va riscritto** per aderire alla teoria, oppure va documentato esplicitamente come variante e giustificate le differenze.

**Bug aggiuntivo ASEBO**: La formula del gradiente è `grad / (2 * self.sigma)` ma dovrebbe essere `grad / (2 * self.sigma * n_samples)` per coerenza con la definizione della tesi. Questo causa un fattore di scala errato nel learning rate effettivo.
