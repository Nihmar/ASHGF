use crate::optimizers::base::{Optimizer, OptimizerError, OptimizerPoint, OptimizerResult};
use nalgebra::{DMatrix, DVector};
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

pub struct ASEBO {
    pub lr: f64,
    pub sigma: f64,
    pub k: usize,
    pub thresh: f64,
    pub eps: f64,
    seed: u64,
}

// ---------------------------------------------------------------------------
// Helper: valutazione di `2 * n_samples` punti perturbati
//
// # Layout di `directions`
// Vec<f64> piatto row-major di lunghezza `n_samples * dim`.
// La riga `i` si accede come `directions[i*dim..(i+1)*dim]`.
// ---------------------------------------------------------------------------

#[cfg(feature = "parallel")]
fn eval_directions<F>(
    directions: &[f64],
    x: &[f64],
    n_samples: usize,
    dim: usize,
    sigma: f64,
    f: &F,
) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let results: Vec<(f64, f64)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let row = &directions[i * dim..(i + 1) * dim];
            let pp: Vec<f64> = x.iter().zip(row).map(|(xi, di)| xi + sigma * di).collect();
            let pm: Vec<f64> = x.iter().zip(row).map(|(xi, di)| xi - sigma * di).collect();
            (f(&pp), f(&pm))
        })
        .collect();
    results.into_iter().unzip()
}

#[cfg(not(feature = "parallel"))]
fn eval_directions<F>(
    directions: &[f64],
    x: &[f64],
    n_samples: usize,
    dim: usize,
    sigma: f64,
    f: &F,
) -> (Vec<f64>, Vec<f64>)
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let mut evals_plus = Vec::with_capacity(n_samples);
    let mut evals_minus = Vec::with_capacity(n_samples);
    let mut pp = x.to_vec();
    let mut pm = x.to_vec();

    for i in 0..n_samples {
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
// Impl ASEBO
// ---------------------------------------------------------------------------

impl ASEBO {
    pub fn new() -> Self {
        Self {
            lr: 1e-4,
            sigma: 1e-4,
            k: 50,
            thresh: 0.995,
            eps: 1e-8,
            seed: 2003,
        }
    }

    pub fn with_params(
        lr: f64,
        sigma: f64,
        k: usize,
        thresh: f64,
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
        if !(0.0..=1.0).contains(&thresh) {
            return Err(OptimizerError::InvalidParameter(
                "thresh must be in [0, 1]".into(),
            ));
        }
        Ok(Self {
            lr,
            sigma,
            k,
            thresh,
            eps,
            seed,
        })
    }

    // -----------------------------------------------------------------------
    // Campionamento delle direzioni
    //
    // FIX 1: `rng` viene passato per riferimento mutabile (condiviso con
    //         l'ottimizzatore) invece di essere ricreato ad ogni chiamata.
    // FIX 2: le componenti gaussiane usano `StandardNormal.sample` (non uniform).
    //         Il branch attivo/ortogonale usa `rng.gen::<f64>()` per U(0,1).
    // FIX 3: χ(dim) è la norma di un vettore di `dim` gaussiane indipendenti,
    //         equivalente a sqrt(χ²(dim)).
    // -----------------------------------------------------------------------

    /// Campiona `n_samples` direzioni di perturbazione ∈ ℝ^dim.
    ///
    /// Se `u_act` è `None`, il campionamento è isotropo.
    /// Se `u_act` è `Some(U)` (dim × r con colonne ortonormali), il campionamento
    /// è un mix:
    ///   - con probabilità `p`: proiezione sul sottospazio attivo (span di U)
    ///   - con probabilità `1-p`: complemento ortogonale
    ///
    /// In entrambi i casi la direzione viene normalizzata a norma unitaria
    /// e poi scalata da un campione di χ(dim).
    ///
    /// Ritorna un `Vec<f64>` piatto row-major di lunghezza `n_samples * dim`.
    fn sample_directions(
        &self,
        n_samples: usize,
        dim: usize,
        u_act: Option<&DMatrix<f64>>,
        p: f64,
        rng: &mut StdRng,
    ) -> Vec<f64> {
        let mut directions = vec![0.0f64; n_samples * dim];

        for i in 0..n_samples {
            // v ~ N(0, I_dim)  [FIX 2]
            let v: Vec<f64> = (0..dim).map(|_| StandardNormal.sample(rng)).collect();

            // Seleziona la componente (attiva o ortogonale)
            let raw: Vec<f64> = if let Some(u) = u_act {
                let v_vec = DVector::from_column_slice(&v);
                // proj = U * (U^T * v)
                let proj_vec = u * (u.transpose() * &v_vec);

                if rng.gen::<f64>() < p {
                    // componente nel sottospazio attivo
                    proj_vec.as_slice().to_vec()
                } else {
                    // complemento ortogonale: v - proj
                    v.iter()
                        .zip(proj_vec.as_slice())
                        .map(|(vi, pi)| vi - pi)
                        .collect()
                }
            } else {
                v
            };

            // Normalizza a vettore unitario
            let norm_raw: f64 = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
            let unit: Vec<f64> = if norm_raw < 1e-12 {
                // Fallback (estremamente improbabile)
                let fb: Vec<f64> = (0..dim).map(|_| StandardNormal.sample(rng)).collect();
                let norm_fb: f64 = fb.iter().map(|x| x * x).sum::<f64>().sqrt();
                fb.iter().map(|x| x / norm_fb.max(1e-12)).collect()
            } else {
                raw.iter().map(|x| x / norm_raw).collect()
            };

            // Scala per χ(dim): norma di un vettore gaussiano dim-dimensionale  [FIX 3]
            let chi: f64 = (0..dim)
                .map(|_| {
                    let g: f64 = StandardNormal.sample(rng);
                    g * g
                })
                .sum::<f64>()
                .sqrt();

            let base = i * dim;
            for (j, uj) in unit.iter().enumerate() {
                directions[base + j] = chi * uj;
            }
        }

        directions
    }

    // -----------------------------------------------------------------------
    // PCA sul buffer dei gradienti
    //
    // FIX 4: gli autovalori (e i corrispondenti autovettori) vengono ordinati
    //         per varianza decrescente prima di selezionare le prime r componenti.
    //         `symmetric_eigen` di nalgebra non garantisce alcun ordinamento.
    // -----------------------------------------------------------------------

    /// Esegue la PCA sul buffer dei gradienti e restituisce la matrice
    /// U_act (dim × r) delle prime r componenti principali (per varianza
    /// decrescente) che spiegano almeno `self.thresh` della varianza totale,
    /// e il rango `r`.
    ///
    /// Restituisce `None` se il buffer contiene meno di 2 campioni.
    fn compute_pca(&self, grad_buffer: &[Vec<f64>]) -> Option<(DMatrix<f64>, usize)> {
        if grad_buffer.len() < 2 {
            return None;
        }

        let n = grad_buffer.len();
        let dim = grad_buffer[0].len();

        // Centra i dati (media per colonna)
        let means: Vec<f64> = (0..dim)
            .map(|j| grad_buffer.iter().map(|g| g[j]).sum::<f64>() / n as f64)
            .collect();

        let flat: Vec<f64> = grad_buffer
            .iter()
            .flat_map(|g| g.iter().zip(&means).map(|(gj, mj)| gj - mj))
            .collect();
        let data = DMatrix::from_row_slice(n, dim, &flat);

        // Matrice di covarianza: dim × dim (non-biased, come sklearn)
        let cov = (data.transpose() * data) / (n - 1) as f64;

        // Decomposizione spettrale (nalgebra non ordina gli autovalori)
        let eigen = cov.symmetric_eigen();

        // FIX 4: ordina per autovalore decrescente
        let mut pairs: Vec<(f64, usize)> = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &ev)| (ev, i))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_var: f64 = pairs.iter().map(|(ev, _)| ev.max(0.0)).sum();

        // Minimo r tale che la varianza cumulativa >= thresh
        let mut cum_var = 0.0;
        let mut r = 0;
        for &(ev, _) in &pairs {
            cum_var += ev.max(0.0) / total_var.max(f64::EPSILON);
            r += 1;
            if cum_var >= self.thresh {
                break;
            }
        }
        r = r.max(1); // almeno 1 componente

        // Assembla U_act: dim × r, colonne = autovettori ordinati per varianza
        let mut u_act = DMatrix::zeros(dim, r);
        for (col_idx, &(_, orig_idx)) in pairs[..r].iter().enumerate() {
            let ev_col = eigen.eigenvectors.column(orig_idx);
            for row_idx in 0..dim {
                u_act[(row_idx, col_idx)] = ev_col[row_idx];
            }
        }

        Some((u_act, r))
    }
}

impl Default for ASEBO {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Trait Optimizer
// ---------------------------------------------------------------------------

impl Optimizer for ASEBO {
    fn name(&self) -> &'static str {
        "Adaptive ES-Active Subspaces"
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
        // FIX 1: unico RNG condiviso da tutti i metodi che generano casualità
        let mut rng = StdRng::seed_from_u64(self.seed);

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
            None => (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect(),
        };

        let mut current_val = function(&x);
        let mut best_value = current_val;
        let mut best_points = vec![OptimizerPoint {
            x: x.clone(),
            value: best_value,
        }];
        let mut all_values = vec![current_val];

        let mut grad_buffer: Vec<Vec<f64>> = Vec::new();
        let mut p = 0.5_f64;

        if debug {
            println!(
                "algorithm: asebo  dimension: {}  initial value: {}",
                dim, current_val
            );
        }

        let mut x_prev = x.clone(); // [FIX 5] per il criterio di arresto

        for i in 1..=it {
            if debug && i % itprint == 0 {
                println!(
                    "{}th iteration - value: {}  last best value: {}",
                    i, current_val, best_value
                );
            }

            // ----------------------------------------------------------------
            // Determina n_samples e U_act (warmup vs fase ASEBO)
            // ----------------------------------------------------------------
            let (n_samples, u_act) = if i <= self.k {
                (dim, None)
            } else {
                match self.compute_pca(&grad_buffer) {
                    Some((u, r)) => (r, Some(u)),
                    None => (dim, None),
                }
            };

            // ----------------------------------------------------------------
            // Campiona le direzioni e valuta f
            // FIX 1: passa &mut rng
            // ----------------------------------------------------------------
            let directions = self.sample_directions(n_samples, dim, u_act.as_ref(), p, &mut rng);

            let (evals_plus, evals_minus) =
                eval_directions(&directions, &x, n_samples, dim, self.sigma, &function);

            // ----------------------------------------------------------------
            // Stima del gradiente: grad = diffs @ directions / (2σ·n_samples)
            // ----------------------------------------------------------------
            let mut grad = vec![0.0f64; dim];
            let factor = 1.0 / (2.0 * self.sigma * n_samples as f64);
            for idx in 0..n_samples {
                let d = (evals_plus[idx] - evals_minus[idx]) * factor;
                let row = &directions[idx * dim..(idx + 1) * dim];
                grad.iter_mut()
                    .zip(row.iter())
                    .for_each(|(g, di)| *g += d * di);
            }

            // ----------------------------------------------------------------
            // Aggiorna il buffer dei gradienti (FIFO, max k)
            // ----------------------------------------------------------------
            grad_buffer.push(grad.clone());
            if grad_buffer.len() > self.k {
                grad_buffer.remove(0);
            }

            // ----------------------------------------------------------------
            // Aggiorna p (solo nella fase post-warmup con U_act disponibile)
            // ----------------------------------------------------------------
            if let Some(ref u) = u_act {
                let r = u.ncols();
                let grad_vec = DVector::from_column_slice(&grad);
                let grad_act_vec = u * (u.transpose() * &grad_vec);

                let s_act: f64 = grad_act_vec.iter().map(|x| x * x).sum();
                let s_perp: f64 = grad_vec
                    .iter()
                    .zip(grad_act_vec.iter())
                    .map(|(g, ga)| (g - ga).powi(2))
                    .sum();

                let d_act = r;
                let d_perp = dim - r;

                let sqrt_act = if s_act > 0.0 {
                    (s_act * (d_act + 2) as f64).sqrt()
                } else {
                    0.0
                };
                let sqrt_perp = if s_perp > 0.0 {
                    (s_perp * (d_perp + 2) as f64).sqrt()
                } else {
                    0.0
                };

                p = if sqrt_act + sqrt_perp > 0.0 {
                    (sqrt_act / (sqrt_act + sqrt_perp)).clamp(0.01, 0.99)
                } else {
                    0.5
                };
            }

            // ----------------------------------------------------------------
            // Passo di discesa: x ← x - lr * grad
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
            // Criterio di arresto: ‖x − x_prev‖₂ < eps  [FIX 5]
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
    fn test_asebo_convergence() {
        let mut asebo = ASEBO::new();
        let result = asebo
            .optimize(sphere, 10, 200, None, false, 25)
            .expect("optimization should succeed");

        let initial = result.all_values[0];
        let final_best = result.best_value();
        assert!(
            final_best < initial,
            "ASEBO should improve from initial value (initial={initial}, best={final_best})"
        );
    }

    #[test]
    fn test_reproducibility() {
        let mut a1 = ASEBO::with_params(1e-4, 1e-4, 10, 0.995, 42, 1e-8).unwrap();
        let mut a2 = ASEBO::with_params(1e-4, 1e-4, 10, 0.995, 42, 1e-8).unwrap();

        let r1 = a1.optimize(sphere, 5, 30, None, false, 25).unwrap();
        let r2 = a2.optimize(sphere, 5, 30, None, false, 25).unwrap();

        assert_eq!(r1.all_values, r2.all_values);
        assert_eq!(r1.best_value(), r2.best_value());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut asebo = ASEBO::new();
        let result = asebo.optimize(sphere, 10, 10, Some(&[1.0, 2.0, 3.0]), false, 1);
        assert!(matches!(
            result,
            Err(OptimizerError::DimensionMismatch {
                expected: 10,
                got: 3
            })
        ));
    }

    #[test]
    fn test_pca_phase_activates() {
        // k=3 → dal quarto step usa la PCA
        let mut asebo = ASEBO::with_params(1e-4, 1e-4, 3, 0.995, 7, 1e-12).unwrap();
        let result = asebo
            .optimize(sphere, 6, 20, None, false, 1)
            .expect("should not panic");
        assert!(!result.all_values.is_empty());
    }
}
