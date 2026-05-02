# TODO — Verifica e correzione bug Tesi ↔ Codice

Bug trovati durante la verifica incrociata tra la tesi (`thesis/chapters/`) e
le implementazioni in `ashgf/algorithms/`.

---

## 🔴 Critici

### [x] #1 SGES — Alpha update invertito ✅ RISOLTO
- **File**: `ashgf/algorithms/sges.py`
- **Descrizione**: Quando lo spazio gradiente produce valori funzione migliori
  ($r < \hat{r}$), il codice **aumentava** `alpha` (probabilità di direzioni
  casuali), mentre la tesi prescrive di **diminuire** $\alpha$ per favorire lo
  sfruttamento. La stessa correzione già applicata in ASHGF è stata portata in
  SGES.
- **Fix**: Invertiti i rami dell'`if r < r_hat`: ora se il gradiente è migliore
  `alpha /= delta` (diminuisce), altrimenti `alpha *= delta` (aumenta).
- **Riferimento tesi**: Eq. `adapt alpha`, Sezione SGES (Cap. 2, Sez. 4)

---

## 🟡 Moderati

### [x] #2 — Stima costanti di Lipschitz: solo nodi consecutivi ✅ RISOLTO
- **File**: `ashgf/gradient/estimators.py` → `estimate_lipschitz_constants()`
- **Descrizione**: La tesi (Eq. `Lipschitz constants`) definisce l'insieme $I$
  che include **tutte** le coppie $\{i,k\}$ tranne quelle simmetriche rispetto
  al centro. Il codice usava `np.diff` che calcolava solo differenze tra nodi
  **consecutivi** $(p_k, p_{k+1})$, sottostimando la costante di Lipschitz in
  presenza di variazioni brusche tra nodi non adiacenti.
- **Fix**: Riscritta la funzione per generare tutte le coppie valide via
  `np.triu_indices`, filtrare quelle simmetriche, e calcolare il max su tutte
  le coppie rimanenti (vectorizzato).

### ~ #3 ASEBO — Campionamento blended covariance vs probabilistic mixture ⚠️ DOCUMENTATO
- **File**: `ashgf/algorithms/asebo.py`
- **Descrizione**: La tesi descrive un campionamento **probabilistico**: ogni
  direzione viene campionata o dallo spazio attivo o da quello ortogonale. Il
  codice usa una matrice di covarianza "blended" che fonde i due contributi.
  La scelta è già documentata nel docstring come "design choice" e non impatta
  la correttezza matematica (entrambi i metodi sono validi).
- **Stato**: Scelta progettuale consapevole. Non modificata.

### ~ #4 ASEBO — Aggiornamento $p^t$/$\alpha$ con formula chiusa ⚠️ DOCUMENTATO
- **File**: `ashgf/algorithms/asebo.py`
- **Descrizione**: La tesi descrive un Algoritmo 2 interno di ottimizzazione su
  orizzonte $C$ per determinare $p^t$. Il codice usa una semplice formula chiusa
  basata sul rapporto delle norme proiettate:
  `alpha = norm_ort / norm_active`. È una semplificazione drastica.
- **Stato**: Differenza sostanziale. Richiederebbe una riscrittura significativa
  per implementare l'Algoritmo 2. Rimandato.

### ~ #5 ASEBO — Aggiornamento covarianza via PCA su buffer ⚠️ DOCUMENTATO
- **File**: `ashgf/algorithms/asebo.py`
- **Descrizione**: La tesi usa media mobile esponenziale su matrice di
  covarianza: $\text{Cov}_{t+1} = \lambda \text{Cov}_t + (1-\lambda)\Gamma$.
  Il codice mantiene un buffer di gradienti con decadimento 0.99 e applica PCA.
  Non sono equivalenti in generale.
- **Stato**: Approssimazione ragionevole. Modifica rinviata.

---

## 🟢 Minori

### [x] #6 ASEBO — Warm-up: $n_t=100$ fisso invece di $n_t=d$ ✅ RISOLTO
- **File**: `ashgf/algorithms/asebo.py`
- **Descrizione**: La tesi prescrive $n_t = d$ direzioni durante il warm-up,
  il codice usava `M = 100` fisso. Ora usa `M = dim`.
- **Fix**: Sostituito `M = 100` con `M = dim` nel ramo warm-up e nel primo
  passo PCA.

---

## ✅ Già risolti (verificati)

- **BUG 1.5.1** — SGES `grad_estimator()` sovrascriveva le direzioni → ✅
- **BUG 1.5.2** — ASHGF usava `f(x_{i-1})` invece di `f(x_i)` → ✅
- **BUG 1.5.3** — SGES chiamava `np.random.seed()` dentro `grad_estimator()` → ✅
- **Dati globali** — `ASHGF.data` / `ASGF.data` spostati ad attributi di istanza → ✅
- **Formula gradiente ASEBO** — divisione per `n_samples` aggiunta → ✅
- **Normalizzazione ASEBO** — normalizzazione a norma 1 rimossa (preserva $\chi(d)$) → ✅
