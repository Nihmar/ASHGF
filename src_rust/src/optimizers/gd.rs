use crate::optimizers::base::{
    validate_x_init, Optimizer, OptimizerError, OptimizerPoint, OptimizerResult,
};
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

pub struct GD {
    pub lr: f64,
    pub sigma: f64,
    pub eps: f64,
    rng: StdRng,
}

impl GD {
    /// Costruttore con parametri di default.
    pub fn new() -> Self {
        // SAFETY: i valori di default sono sempre validi, unwrap() non può fallire.
        Self::with_params(1e-4, 1e-4, 2003, 1e-8).unwrap()
    }

    /// Costruttore con parametri espliciti.
    ///
    /// # Errori
    /// Restituisce `Err` se `lr < 0` oppure `sigma < 0`.
    pub fn with_params(lr: f64, sigma: f64, seed: u64, eps: f64) -> Result<Self, OptimizerError> {
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
        Ok(Self {
            lr,
            sigma,
            eps,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    // -----------------------------------------------------------------------
    // Gradient estimator
    // -----------------------------------------------------------------------

    /// Stima del gradiente via Central Gaussian Smoothing (vettorizzato).
    ///
    /// Formula:
    ///   ∇Fσ(x) ≈ 1/(2σn) Σᵢ (F(x+σξᵢ) - F(x-σξᵢ)) ξᵢ
    ///
    /// Ottimizzazioni rispetto alla versione originale:
    /// - `directions` è un buffer flat `dim×dim` (cache-friendly, no Vec<Vec>).
    /// - `point_plus` / `point_minus` sono pre-allocati e riscritti in-place
    ///   ad ogni direzione (evita dim*2 allocazioni heap per iterazione).
    fn grad_estimator_vectorized<F>(&mut self, x: &[f64], f: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let dim = x.len();

        // --- Genera le direzioni come matrice flat (dim * dim) ---------------
        // directions[i * dim + j] corrisponde a directions[i][j] in Python.
        // Layout contiguo in memoria → accessi sequenziali nella somma finale.
        let directions: Vec<f64> = (0..dim * dim)
            .map(|_| StandardNormal.sample(&mut self.rng))
            .collect();

        // --- Buffers riusabili per i punti perturbati ------------------------
        // Alloco una sola volta fuori dal loop e copio in-place ogni volta.
        let mut point_plus = x.to_vec();
        let mut point_minus = x.to_vec();

        let mut diffs = Vec::with_capacity(dim);

        for i in 0..dim {
            let dir = &directions[i * dim..(i + 1) * dim]; // slice della i-esima riga

            // Reset in-place e perturbazione
            for j in 0..dim {
                point_plus[j] = x[j] + self.sigma * dir[j];
                point_minus[j] = x[j] - self.sigma * dir[j];
            }

            diffs.push(f(&point_plus) - f(&point_minus));
        }

        // --- Gradiente: Σ diffs[i] * directions[i]  -------------------------
        // Equivalente al prodotto matrice-vettore diffs @ directions in Python.
        let mut grad = vec![0.0f64; dim];
        for i in 0..dim {
            let dir = &directions[i * dim..(i + 1) * dim];
            let d = diffs[i];
            for j in 0..dim {
                grad[j] += d * dir[j];
            }
        }

        // Normalizzazione: 1 / (2σ * dim)
        let factor = 1.0 / (2.0 * self.sigma * dim as f64);
        grad.iter_mut().for_each(|g| *g *= factor);

        grad
    }
}

// ---------------------------------------------------------------------------
// Default
// ---------------------------------------------------------------------------

impl Default for GD {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Trait Optimizer
// ---------------------------------------------------------------------------

impl Optimizer for GD {
    fn name(&self) -> &'static str {
        "Vanilla Gradient Descent"
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
        // --- Punto iniziale --------------------------------------------------
        // validate_x_init propaga DimensionMismatch se x_init ha dimensione errata.
        let mut x = validate_x_init(x_init, dim, &mut self.rng)?;

        let mut current_val = function(&x);
        let mut best_value = current_val;

        let mut best_points = vec![OptimizerPoint {
            x: x.clone(),
            value: best_value,
        }];
        let mut all_values = vec![current_val];

        if debug {
            println!(
                "algorithm: {}  dimension: {}  initial value: {}",
                self.name(),
                dim,
                current_val
            );
        }

        // --- Loop principale -------------------------------------------------
        let mut x_prev = vec![0.0f64; dim]; // buffer riusabile per la norma

        for i in 1..=it {
            if debug && i % itprint == 0 {
                println!("{i}th iteration - value: {current_val}  last best value: {best_value}");
            }

            let grad = self.grad_estimator_vectorized(&x, function);

            // Salva x corrente nel buffer (per il criterio di arresto)
            x_prev.copy_from_slice(&x);

            // Aggiornamento: x ← x - lr * grad
            for j in 0..dim {
                x[j] -= self.lr * grad[j];
            }

            current_val = function(&x);
            all_values.push(current_val);

            if current_val < best_value {
                best_value = current_val;
                best_points.push(OptimizerPoint {
                    x: x.clone(),
                    value: best_value,
                });
            }

            // Criterio di arresto: ‖x_new - x_prev‖₂ < eps
            let norm_diff: f64 = x
                .iter()
                .zip(x_prev.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if norm_diff < self.eps {
                if debug {
                    println!("Converged at iteration {i} (norm_diff = {norm_diff:.2e})");
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
    // use crate::functions::sphere; // unused

    fn sphere_local(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }

    #[test]
    fn test_default_params_are_valid() {
        // GD::new() non deve andare in panic
        let _ = GD::new();
    }

    #[test]
    fn test_invalid_lr_returns_err() {
        assert!(GD::with_params(-1.0, 1e-4, 0, 1e-8).is_err());
    }

    #[test]
    fn test_invalid_sigma_returns_err() {
        assert!(GD::with_params(1e-4, -1.0, 0, 1e-8).is_err());
    }

    #[test]
    fn test_dimension_mismatch_returns_err() {
        let mut gd = GD::new();
        let wrong_init = vec![0.0f64; 5]; // dim atteso = 10
        let result = gd.optimize(sphere_local, 10, 100, Some(&wrong_init), false, 25);
        assert!(matches!(
            result,
            Err(OptimizerError::DimensionMismatch {
                expected: 10,
                got: 5
            })
        ));
    }

    #[test]
    fn test_gd_convergence() {
        let mut gd = GD::new();
        let result = gd
            .optimize(sphere_local, 10, 500, None, false, 25)
            .expect("optimization should succeed");

        let initial = result.all_values[0];
        let best = result.best_value();
        assert!(best < initial, "GD should improve from initial value");
    }

    #[test]
    fn test_result_structure() {
        let mut gd = GD::new();
        let result = gd.optimize(sphere_local, 5, 100, None, false, 25).unwrap();

        // best_points deve contenere almeno il punto iniziale
        assert!(!result.best_points.is_empty());
        // ogni OptimizerPoint deve avere x di lunghezza dim
        for p in &result.best_points {
            assert_eq!(p.x.len(), 5);
        }
        // all_values deve contenere it+1 elementi (o meno se converge prima)
        assert!(result.all_values.len() <= 101);
    }
}
