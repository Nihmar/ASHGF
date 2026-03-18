use crate::optimizers::base::{Optimizer, OptimizerError, OptimizerPoint, OptimizerResult};
use nalgebra::DMatrix;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Binomial, StandardNormal};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

pub struct SGES {
    pub lr: f64,
    pub sigma: f64,
    pub k: usize,
    pub k1: f64,
    pub k2: f64,
    pub alpha: f64,
    pub delta: f64,
    pub t: usize,
    pub eps: f64,
    rng: StdRng,
}

// ---------------------------------------------------------------------------
// Helper: valutazione di `2 * dim` punti perturbati
//
// Questa funzione è l'unico posto in cui si chiama `f`. È separata da
// `grad_estimator` e `grad_estimator_sges` così la logica parallela/sequenziale
// è scritta una volta sola.
//
// # Layout di `directions`
// Vec<f64> piatto row-major di lunghezza `dim * dim`.
// La riga `i` (direzione i-esima) è lo slice `directions[i*dim..(i+1)*dim]`.
//
// # Restituzione
// `(evals_plus, evals_minus)` entrambi di lunghezza `dim`, con
// `evals_plus[i]  = f(x + sigma * directions[i])`
// `evals_minus[i] = f(x - sigma * directions[i])`
// ---------------------------------------------------------------------------

/// Versione parallela: `F: Sync` richiesto perché `f` è condivisa tra thread.
/// Ogni thread alloca i propri buffer locali `point_plus`/`point_minus`.
#[cfg(feature = "parallel")]
fn eval_directions<F>(
    directions: &[f64],
    x: &[f64],
    dim: usize,
    sigma: f64,
    f: &F,
) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let results: Vec<(f64, f64)> = (0..dim)
        .into_par_iter()
        .map(|i| {
            let row = &directions[i * dim..(i + 1) * dim];
            // Ogni worker alloca i propri buffer: nessuna condivisione mutabile.
            let pp: Vec<f64> = x.iter().zip(row).map(|(xi, di)| xi + sigma * di).collect();
            let pm: Vec<f64> = x.iter().zip(row).map(|(xi, di)| xi - sigma * di).collect();
            (f(&pp), f(&pm))
        })
        .collect();

    results.into_iter().unzip()
}

/// Versione sequenziale: riusa i buffer `point_plus`/`point_minus` ad ogni iterazione.
/// `F: Sync` è richiesto per uniformità di firma con la versione parallela,
/// ma non è strettamente necessario qui (tutte le funzioni normali lo soddisfano).
#[cfg(not(feature = "parallel"))]
fn eval_directions<F>(
    directions: &[f64],
    x: &[f64],
    dim: usize,
    sigma: f64,
    f: &F,
) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let mut evals_plus = Vec::with_capacity(dim);
    let mut evals_minus = Vec::with_capacity(dim);
    let mut pp = x.to_vec();
    let mut pm = x.to_vec();

    for i in 0..dim {
        let row = &directions[i * dim..(i + 1) * dim];
        pp.iter_mut()
            .zip(x.iter().zip(row))
            .for_each(|(p, (xi, di))| *p = xi + sigma * di);
        pm.iter_mut()
            .zip(x.iter().zip(row))
            .for_each(|(p, (xi, di))| *p = xi - sigma * di);
        evals_plus.push(f(&pp));
        evals_minus.push(f(&pm));
    }

    (evals_plus, evals_minus)
}

// ---------------------------------------------------------------------------
// Impl SGES
// ---------------------------------------------------------------------------

impl SGES {
    pub fn new() -> Self {
        Self::with_params(1e-4, 1e-4, 50, 0.9, 0.1, 0.5, 1.1, 50, 2003, 1e-8).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        lr: f64,
        sigma: f64,
        k: usize,
        k1: f64,
        k2: f64,
        alpha: f64,
        delta: f64,
        t: usize,
        seed: u64,
        eps: f64,
    ) -> Result<Self, OptimizerError> {
        if lr < 0.0 {
            return Err(OptimizerError::InvalidParameter(
                "learning rate must be >= 0".into(),
            ));
        }
        if sigma < 0.0 {
            return Err(OptimizerError::InvalidParameter(
                "sigma must be >= 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(OptimizerError::InvalidParameter(
                "alpha must be in [0, 1]".into(),
            ));
        }
        Ok(Self {
            lr,
            sigma,
            k,
            k1,
            k2,
            alpha,
            delta,
            t,
            eps,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    // -----------------------------------------------------------------------
    // Stima del gradiente: fase di warmup (smoothing gaussiano centrale)
    // -----------------------------------------------------------------------

    /// Genera `dim` direzioni ~ N(0, I), valuta `f` nei punti perturbati
    /// tramite `eval_directions` (sequenziale o parallelo a seconda della feature),
    /// e restituisce la stima del gradiente e il vettore di valutazioni interleaved.
    fn grad_estimator<F>(&mut self, x: &[f64], dim: usize, f: &F) -> (Vec<f64>, Vec<f64>)
    where
        F: Fn(&[f64]) -> f64 + Sync,
    {
        // Genera direzioni ~ N(0, I): dim × dim, row-major. Usa self.rng (non Send),
        // quindi questa parte rimane sempre sequenziale.
        let mut directions = vec![0.0f64; dim * dim];
        for d in directions.iter_mut() {
            *d = StandardNormal.sample(&mut self.rng);
        }

        let (evals_plus, evals_minus) = eval_directions(&directions, x, dim, self.sigma, f);

        // grad = Σ_i (f⁺_i − f⁻_i) * directions[i] / (2σ·dim)
        // Accumulo sequenziale: operazione leggera, non vale parallelizzare.
        let mut grad = vec![0.0f64; dim];
        let factor = 1.0 / (2.0 * self.sigma * dim as f64);
        for i in 0..dim {
            let d = (evals_plus[i] - evals_minus[i]) * factor;
            let row = &directions[i * dim..(i + 1) * dim];
            grad.iter_mut()
                .zip(row.iter())
                .for_each(|(g, di)| *g += d * di);
        }

        // Interleave: [f⁺_0, f⁻_0, f⁺_1, f⁻_1, ...]
        let mut evaluations = vec![0.0f64; 2 * dim];
        for i in 0..dim {
            evaluations[2 * i] = evals_plus[i];
            evaluations[2 * i + 1] = evals_minus[i];
        }

        (grad, evaluations)
    }

    // -----------------------------------------------------------------------
    // Stima del gradiente: fase SGES (post-warmup)
    // -----------------------------------------------------------------------

    fn grad_estimator_sges<F>(
        &mut self,
        x: &[f64],
        dim: usize,
        g_history: &[Vec<f64>],
        alpha: f64,
        f: &F,
    ) -> (Vec<f64>, Vec<f64>, usize)
    where
        F: Fn(&[f64]) -> f64 + Sync,
    {
        // Direzioni: Vec<f64> piatto row-major (generazione sequenziale, usa self.rng).
        let (directions, m_dirs) = self.compute_directions_sges(dim, g_history, alpha);

        let (evals_plus, evals_minus) = eval_directions(&directions, x, dim, self.sigma, f);

        let mut grad = vec![0.0f64; dim];
        let factor = 1.0 / (2.0 * self.sigma * dim as f64);
        for i in 0..dim {
            let d = (evals_plus[i] - evals_minus[i]) * factor;
            let row = &directions[i * dim..(i + 1) * dim];
            grad.iter_mut()
                .zip(row.iter())
                .for_each(|(g, di)| *g += d * di);
        }

        let mut evaluations = vec![0.0f64; 2 * dim];
        for i in 0..dim {
            evaluations[2 * i] = evals_plus[i];
            evaluations[2 * i + 1] = evals_minus[i];
        }

        (grad, evaluations, m_dirs)
    }

    // -----------------------------------------------------------------------
    // Costruzione delle direzioni SGES con SVD (fedele alla versione Python)
    // -----------------------------------------------------------------------

    /// Costruisce `dim` direzioni di ricerca:
    /// - M ~ Binomiale(dim, alpha) dal sottospazio dei gradienti recenti (SVD)
    /// - (dim − M) da N(0, I)                                            (esplorazione)
    ///
    /// Ritorna un `Vec<f64>` piatto row-major di lunghezza `dim * dim`:
    /// la riga `i` si accede come `directions[i*dim..(i+1)*dim]`.
    ///
    /// # Nota
    /// Le direzioni NON vengono normalizzate (il commento Python è esplicito:
    /// la normalizzazione altererebbe la distribuzione corretta).
    fn compute_directions_sges(
        &mut self,
        dim: usize,
        g_history: &[Vec<f64>],
        alpha: f64,
    ) -> (Vec<f64>, usize) {
        // Campiona M ~ Binomiale(dim, alpha)
        let binom = Binomial::new(dim as u64, alpha).expect("alpha deve essere in [0,1]");
        let m_dirs = (binom.sample(&mut self.rng) as usize).clamp(0, dim);

        // Buffer piatto: dim righe × dim colonne, row-major.
        // Inizializzato a zero; le sezioni vengono riempite sotto.
        let mut directions = vec![0.0f64; dim * dim];

        // Filtra gradienti con valori non finiti (equivalente a G_clean in Python)
        let g_clean: Vec<&Vec<f64>> = g_history
            .iter()
            .filter(|g| g.iter().all(|v| v.is_finite()))
            .collect();

        if m_dirs > 0 {
            if g_clean.len() >= 2 {
                // ----------------------------------------------------------------
                // Base ortonormale del sottospazio via SVD (identico a Python):
                //   U, s, _ = np.linalg.svd(G_clean.T, full_matrices=False)
                //   rank = np.sum(s > 1e-10)
                //   U_sub = U[:, :rank]           # (dim, rank)
                //   z = np.random.randn(rank, M)   # (rank, M)
                //   dirs_G = (U_sub @ z).T         # (M, dim)  — NON normalizzate
                // ----------------------------------------------------------------
                let n_grads = g_clean.len();
                let flat: Vec<f64> = g_clean.iter().flat_map(|g| g.iter().copied()).collect();
                let g_mat = DMatrix::from_row_slice(n_grads, dim, &flat);

                let gt = g_mat.transpose(); // dim × n_grads
                let svd = gt.svd(true, false); // thin SVD, calcola solo U
                let u = svd.u.expect("SVD: U non disponibile"); // dim × min(dim, n_grads)
                let rank = svd.singular_values.iter().filter(|&&sv| sv > 1e-10).count();

                if rank > 0 {
                    let u_sub = u.columns(0, rank); // dim × rank

                    // z ~ N(0, I): rank × M (col-major, come DMatrix)
                    let z_data: Vec<f64> = (0..rank * m_dirs)
                        .map(|_| StandardNormal.sample(&mut self.rng))
                        .collect();
                    let z = DMatrix::from_column_slice(rank, m_dirs, &z_data);

                    // dirs_G_mat = U_sub @ z → dim × M (col-major)
                    // Trasponi nel buffer row-major: riga col = direzione col
                    let dirs_g_mat = u_sub * z;
                    for col in 0..m_dirs {
                        let base = col * dim;
                        for row in 0..dim {
                            directions[base + row] = dirs_g_mat[(row, col)];
                        }
                    }
                } else {
                    // Rango zero (improbabile) → fallback N(0, I)
                    for v in directions[..m_dirs * dim].iter_mut() {
                        *v = StandardNormal.sample(&mut self.rng);
                    }
                }
            } else {
                // Non ci sono abbastanza gradienti validi → N(0, I) per le prime m_dirs righe
                for v in directions[..m_dirs * dim].iter_mut() {
                    *v = StandardNormal.sample(&mut self.rng);
                }
            }
        }

        // Direzioni esplorative: righe [m_dirs..dim] ~ N(0, I)
        for v in directions[m_dirs * dim..].iter_mut() {
            *v = StandardNormal.sample(&mut self.rng);
        }

        debug_assert_eq!(directions.len(), dim * dim);

        (directions, m_dirs)
    }
}

impl Default for SGES {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Trait Optimizer
// ---------------------------------------------------------------------------

impl Optimizer for SGES {
    fn name(&self) -> &'static str {
        "Self-Guided Evolution Strategies"
    }

    fn optimize<F>(
        &mut self,
        function: F,
        dim: usize,
        it: usize,
        x_init: Option<&[f64]>,
        debug: bool,
        itprint: usize,
    ) -> Result<OptimizerResult, OptimizerError>
    where
        F: Fn(&[f64]) -> f64 + Copy,
    {
        // Inizializza il punto di partenza
        let mut x: Vec<f64> = match x_init {
            Some(init) => {
                if init.len() != dim {
                    return Err(OptimizerError::DimensionMismatch {
                        expected: dim,
                        got: init.len(),
                    });
                }
                init.to_vec()
            }
            None => (0..dim)
                .map(|_| StandardNormal.sample(&mut self.rng))
                .collect(),
        };

        let mut current_val = function(&x);
        let mut best_value = current_val;
        let mut best_points = vec![OptimizerPoint {
            x: x.clone(),
            value: best_value,
        }];
        let mut all_values = vec![current_val];

        // Storia dei gradienti: sliding window di dimensione k.
        // Durante il warmup (i < t) la finestra cresce liberamente.
        let mut g_history: Vec<Vec<f64>> = Vec::new();
        let mut alpha = self.alpha;

        if debug {
            println!(
                "algorithm: sges  dimension: {}  initial value: {}",
                dim, current_val
            );
        }

        let mut x_prev = vec![0.0f64; dim];

        for i in 1..=it {
            if debug && i % itprint == 0 {
                println!(
                    "{}th iteration - value: {}  last best value: {}",
                    i, current_val, best_value
                );
            }

            // ----------------------------------------------------------------
            // Stima del gradiente (warmup vs SGES).
            // `function` è passato per riferimento a eval_directions, che
            // richiede `F: Sync`. `F: Copy` implica `F: Sync` per tutti i
            // tipi concreti ragionevoli (fn pointer, closure senza catture !Sync).
            // ----------------------------------------------------------------
            let (grad, evaluations, m_dirs) = if i < self.t {
                let (g, evals) = self.grad_estimator(&x, dim, &function);
                (g, evals, dim) // m_dirs non usato in fase warmup
            } else {
                let (g, evals, m) = self.grad_estimator_sges(&x, dim, &g_history, alpha, &function);
                (g, evals, m)
            };

            // ----------------------------------------------------------------
            // Aggiorna la sliding window: libera durante warmup, cap a k dopo.
            // ----------------------------------------------------------------
            g_history.push(grad.clone());
            if i >= self.t && g_history.len() > self.k {
                g_history.remove(0);
            }

            // ----------------------------------------------------------------
            // Passo di discesa del gradiente: x ← x - lr * grad
            // ----------------------------------------------------------------
            x_prev.copy_from_slice(&x);
            x.iter_mut()
                .zip(grad.iter())
                .for_each(|(xi, gi)| *xi -= self.lr * gi);

            current_val = function(&x);
            all_values.push(current_val);

            if current_val < best_value {
                best_value = current_val;
                best_points.push(OptimizerPoint {
                    x: x.clone(),
                    value: best_value,
                });
            }

            // ----------------------------------------------------------------
            // Criterio di convergenza: ‖x − x_prev‖₂ < eps
            // ----------------------------------------------------------------
            let norm_diff: f64 = x
                .iter()
                .zip(x_prev.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if norm_diff < self.eps {
                if debug {
                    println!(
                        "Converged at iteration {} (norm_diff = {:.2e})",
                        i, norm_diff
                    );
                }
                break;
            }

            // ----------------------------------------------------------------
            // Adattamento di alpha (eq. 2.14 della tesi) — solo post-warmup
            //
            // Python:
            //   if r_G is None or (r_G_ort is not None and r_G < r_G_ort):
            //       alpha = min(delta * alpha, k1)
            //   elif r_G_ort is None or r_G >= r_G_ort:
            //       alpha = max(alpha / delta, k2)
            // ----------------------------------------------------------------
            if i >= self.t {
                let vals_g: Vec<f64> = (0..m_dirs)
                    .map(|j| evaluations[2 * j].min(evaluations[2 * j + 1]))
                    .collect();
                let vals_ort: Vec<f64> = (m_dirs..dim)
                    .map(|j| evaluations[2 * j].min(evaluations[2 * j + 1]))
                    .collect();

                let r_g = if vals_g.is_empty() {
                    None
                } else {
                    Some(vals_g.iter().sum::<f64>() / vals_g.len() as f64)
                };
                let r_g_ort = if vals_ort.is_empty() {
                    None
                } else {
                    Some(vals_ort.iter().sum::<f64>() / vals_ort.len() as f64)
                };

                match (r_g, r_g_ort) {
                    // r_G è None: incrementa alpha (sfruttamento del sottospazio)
                    (None, _) => {
                        alpha = (self.delta * alpha).min(self.k1);
                    }
                    // r_G < r_G_ort: sottospazio più promettente → incrementa alpha
                    (Some(rg), Some(rg_ort)) if rg < rg_ort => {
                        alpha = (self.delta * alpha).min(self.k1);
                    }
                    // r_G_ort è None oppure r_G >= r_G_ort: riduce alpha (esplorazione)
                    (Some(_), None) | (Some(_), Some(_)) => {
                        alpha = (alpha / self.delta).max(self.k2);
                    }
                }
            }
        }

        if debug {
            println!(
                "\nlast evaluation: {}  last_iterate: {}  best evaluation: {}\n",
                all_values.last().unwrap(),
                all_values.len() - 1,
                best_value
            );
        }

        Ok(OptimizerResult {
            best_points,
            all_values,
        })
    }
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&v| v * v).sum()
    }

    #[test]
    fn test_sges_convergence() {
        let mut sges = SGES::new();
        let result = sges
            .optimize(sphere, 10, 200, None, false, 25)
            .expect("optimization should succeed");

        let initial = result.all_values[0];
        let final_best = result.best_value();
        assert!(
            final_best < initial,
            "SGES should improve from initial value (initial={initial}, best={final_best})"
        );
    }

    #[test]
    fn test_reproducibility() {
        let mut sges1 =
            SGES::with_params(1e-4, 1e-4, 50, 0.9, 0.1, 0.5, 1.1, 50, 42, 1e-8).unwrap();
        let mut sges2 =
            SGES::with_params(1e-4, 1e-4, 50, 0.9, 0.1, 0.5, 1.1, 50, 42, 1e-8).unwrap();

        let res1 = sges1.optimize(sphere, 5, 50, None, false, 25).unwrap();
        let res2 = sges2.optimize(sphere, 5, 50, None, false, 25).unwrap();

        assert_eq!(res1.all_values, res2.all_values);
        assert_eq!(res1.best_value(), res2.best_value());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut sges = SGES::new();
        let result = sges.optimize(sphere, 10, 10, Some(&[1.0, 2.0, 3.0]), false, 1);
        assert!(matches!(
            result,
            Err(OptimizerError::DimensionMismatch {
                expected: 10,
                got: 3
            })
        ));
    }

    #[test]
    fn test_sges_phase_activates() {
        // t=5 → dal sesto step usa grad_estimator_sges
        let mut sges = SGES::with_params(1e-4, 1e-4, 10, 0.9, 0.1, 0.5, 1.1, 5, 99, 1e-12).unwrap();
        let result = sges
            .optimize(sphere, 8, 30, None, false, 1)
            .expect("should run without panic");
        assert!(!result.all_values.is_empty());
    }

    /// Verifica che la modalità parallela (se abilitata) non produca panic.
    /// Il risultato numerico può differire leggermente dalla modalità sequenziale
    /// per via dell'ordine di scheduling dei thread, ma la convergenza deve reggere.
    #[test]
    fn test_parallel_does_not_panic() {
        let mut sges = SGES::with_params(1e-4, 1e-4, 10, 0.9, 0.1, 0.5, 1.1, 5, 7, 1e-12).unwrap();
        let result = sges
            .optimize(sphere, 12, 60, None, false, 1)
            .expect("should not panic in parallel mode");
        assert!(result.best_value() < result.all_values[0]);
    }
}
