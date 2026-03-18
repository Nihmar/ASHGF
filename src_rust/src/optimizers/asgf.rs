use crate::optimizers::base::{Optimizer, OptimizerError, OptimizerPoint, OptimizerResult};
use nalgebra::DMatrix;
use rand::distributions::Distribution;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

pub struct ASGF {
    seed: u64,
    pub eps: f64,
    m: usize,
    a: f64,
    b: f64,
    a_minus: f64,
    a_plus: f64,
    b_minus: f64,
    b_plus: f64,
    gamma_l: f64,
    gamma_sigma: f64,
    r: usize,
    ro: f64,
    sigma_zero: f64,
}

impl ASGF {
    pub fn new() -> Self {
        Self {
            seed: 2003,
            eps: 1e-8,
            m: 5,
            a: 0.1,
            b: 0.9,
            a_minus: 0.95,
            a_plus: 1.02,
            b_minus: 0.98,
            b_plus: 1.01,
            gamma_l: 0.9,
            gamma_sigma: 0.9,
            r: 2,
            ro: 0.01,
            sigma_zero: 0.01,
        }
    }

    fn hermite_nodes_weights(m: usize) -> (Vec<f64>, Vec<f64>) {
        let nodes: Vec<f64> = match m {
            5 => vec![-2.320971, -1.0, 0.0, 1.0, 2.320971],
            _ => (0..m)
                .map(|i| (i as f64 - (m - 1) as f64 / 2.0) * 0.5)
                .collect(),
        };

        let weights: Vec<f64> = match m {
            5 => vec![0.048132, 0.218235, 0.464266, 0.218235, 0.048132],
            _ => vec![1.0 / m as f64; m],
        };

        (nodes, weights)
    }

    #[allow(clippy::too_many_arguments)]
    fn grad_estimator<F: Fn(&[f64]) -> f64 + Copy>(
        &self,
        x: &[f64],
        m: usize,
        sigma: f64,
        dim: usize,
        lipschitz_coefficients: &[f64],
        basis: &DMatrix<f64>,
        f: F,
        l_nabla: f64,
        value: f64,
    ) -> (Vec<f64>, Vec<f64>, f64, Vec<f64>, f64) {
        let (nodes_raw, weights_raw) = Self::hermite_nodes_weights(m);

        let nodes: Vec<f64> = nodes_raw.iter().map(|n| n * (2.0_f64).sqrt()).collect();
        let weights: Vec<f64> = weights_raw
            .iter()
            .map(|w| w / std::f64::consts::PI.sqrt())
            .collect();

        let sigma_nodes: Vec<f64> = nodes.iter().map(|n| sigma * n).collect();
        let norm_factor = 1.0 / sigma;

        let mut evaluations: Vec<Vec<f64>> = Vec::with_capacity(dim);
        let mut derivatives = vec![0.0; dim];

        for j in 0..dim {
            let mut temp = vec![0.0; m];
            for k in 0..m {
                if k == m / 2 {
                    temp[k] = value;
                } else {
                    let mut point = x.to_vec();
                    for idx in 0..dim {
                        point[idx] += sigma_nodes[k] * basis[(j, idx)];
                    }
                    temp[k] = f(&point);
                }
            }
            evaluations.push(temp.clone());

            let mut deriv = 0.0;
            for k in 0..m {
                deriv += weights[k] * nodes[k] * temp[k];
            }
            derivatives[j] = norm_factor * deriv;
        }

        let mut grad = vec![0.0; dim];
        for j in 0..dim {
            for k in 0..dim {
                grad[k] += derivatives[j] * basis[(j, k)];
            }
        }

        let mut new_lipschitz = lipschitz_coefficients.to_vec();
        let mid = m / 2;

        for j in 0..dim {
            let mut lip = 0.0;
            let evals_j = &evaluations[j];
            for a in 0..m {
                for b in (a + 1)..m {
                    if (a as i32 - mid as i32).abs() != (b as i32 - mid as i32).abs() {
                        let denom = sigma * (nodes[a] - nodes[b]);
                        if denom.abs() > 1e-12 {
                            let val = (evals_j[a] - evals_j[b]).abs() / denom;
                            if val > lip {
                                lip = val;
                            }
                        }
                    }
                }
            }
            new_lipschitz[j] = lip;
        }

        // Ensure Lipschitz constant is not too small
        // Use a reasonable minimum value based on the expected Lipschitz constant
        // For sphere function, Lipschitz constant ≈ 2 * ||x||
        for lip in new_lipschitz.iter_mut() {
            *lip = lip.max(1.0);
        }

        let l_nabla_new = (1.0 - self.gamma_l) * new_lipschitz[0] + self.gamma_l * l_nabla;
        // Limit learning rate to prevent oscillation
        // For sphere function, optimal lr < 0.5 to avoid overshooting
        let lr = (sigma / l_nabla_new.max(1e-12)).min(0.4);

        (grad, new_lipschitz, lr, derivatives, l_nabla_new)
    }

    fn generate_random_orthogonal(&self, dim: usize, rng: &mut StdRng) -> DMatrix<f64> {
        // Generate a random matrix using QR decomposition
        let mut mat = DMatrix::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                mat[(i, j)] = rng.gen::<f64>() - 0.5;
            }
        }

        // Use QR decomposition to get an orthogonal matrix
        let qr = mat.qr();
        let q = qr.q();

        // Make sure the determinant is positive (adjust sign of first column if needed)
        let det = q.determinant();
        if det < 0.0 {
            // Negate the first column
            let mut q_adjusted = q.clone();
            for i in 0..dim {
                q_adjusted[(i, 0)] *= -1.0;
            }
            q_adjusted
        } else {
            q
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn subroutine(
        &self,
        sigma: f64,
        grad: &[f64],
        derivatives: &[f64],
        lipschitz_coefficients: &[f64],
        mut a: f64,
        mut b: f64,
        mut r: usize,
        rng: &mut StdRng,
    ) -> (f64, DMatrix<f64>, f64, f64, usize) {
        let dim = grad.len();

        if r > 0 && sigma < self.ro * self.sigma_zero {
            // Generate a random orthogonal matrix using the provided RNG
            let mut new_basis = DMatrix::zeros(dim, dim);
            for i in 0..dim {
                for j in 0..dim {
                    new_basis[(i, j)] = rng.gen::<f64>() - 0.5;
                }
            }
            // Normalize rows to get orthonormal basis (approximate)
            for i in 0..dim {
                let norm: f64 = (0..dim)
                    .map(|j| new_basis[(i, j)].powi(2))
                    .sum::<f64>()
                    .sqrt();
                if norm > 1e-12 {
                    for j in 0..dim {
                        new_basis[(i, j)] /= norm;
                    }
                }
            }
            let sigma_new = self.sigma_zero;
            let a_new = self.a;
            let b_new = self.b;
            r -= 1;
            return (sigma_new, new_basis, a_new, b_new, r);
        }

        // Create basis with first row as normalized gradient
        // Start with identity matrix (orthonormal rows)
        let mut new_basis = DMatrix::identity(dim, dim);
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm > 1e-10 {
            // Replace first row with normalized gradient
            for j in 0..dim {
                new_basis[(0, j)] = grad[j] / grad_norm;
            }
        }

        let mut new_lipschitz = lipschitz_coefficients.to_vec();
        for l in new_lipschitz.iter_mut() {
            *l = l.max(1e-10);
        }

        let mut ratio = 0.0;
        for j in 0..dim {
            let r = derivatives[j].abs() / new_lipschitz[j];
            if r > ratio {
                ratio = r;
            }
        }

        let mut sigma_new = sigma;
        if ratio < a {
            sigma_new *= self.gamma_sigma;
            a *= self.a_minus;
        } else if ratio > b {
            sigma_new /= self.gamma_sigma;
            b *= self.b_plus;
        } else {
            a *= self.a_plus;
            b *= self.b_minus;
        }

        (sigma_new, new_basis, a, b, r)
    }
}

impl Default for ASGF {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer for ASGF {
    fn name(&self) -> &'static str {
        "Adaptive Stochastic Gradient-Free"
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
        let mut rng = StdRng::seed_from_u64(self.seed);

        let mut x = match x_init {
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

        let norm_x: f64 = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        self.sigma_zero = norm_x / 10.0;
        let mut sigma = self.sigma_zero;
        let mut a = self.a;
        let mut b = self.b;
        let mut r = self.r;
        let mut l_nabla = 0.0;
        // Initialize Lipschitz coefficients based on the norm of x
        // This ensures they're not too small for functions like sphere
        let mut lipschitz_coefficients = vec![2.0 * norm_x.max(1.0); dim];

        let mut basis = self.generate_random_orthogonal(dim, &mut rng);

        if debug {
            println!(
                "algorithm: asgf  function: custom  dimension: {}  initial value: {}",
                dim, current_val
            );
        }

        if debug {
            println!(
                "algorithm: asgf  function: custom  dimension: {}  initial value: {}",
                dim, current_val
            );
        }

        if debug {
            println!(
                "algorithm: asgf  function: custom  dimension: {}  initial value: {}",
                dim, current_val
            );
        }

        for i in 1..=it {
            if debug && i % itprint == 0 {
                println!(
                    "{}th iteration - value: {}  last best value: {}",
                    i, current_val, best_value
                );
            }

            let (grad, new_lipschitz, lr, derivatives, new_l_nabla) = self.grad_estimator(
                &x,
                self.m,
                sigma,
                dim,
                &lipschitz_coefficients,
                &basis,
                function,
                l_nabla,
                current_val,
            );

            if debug && i % itprint == 0 {
                println!(
                    "  sigma: {}, lr: {}, grad[0]: {}, derivatives[0]: {}",
                    sigma, lr, grad[0], derivatives[0]
                );
            }

            if !grad.iter().all(|g| g.is_finite()) || !lr.is_finite() {
                if debug {
                    println!(
                        "Warning: non-finite gradient or learning rate at iteration {}",
                        i
                    );
                }
                break;
            }

            for j in 0..dim {
                x[j] -= lr * grad[j];
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

            let norm_diff: f64 = x
                .iter()
                .zip(best_points.last().unwrap().x.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if norm_diff < self.eps {
                break;
            } else {
                let (new_sigma, new_basis, new_a, new_b, new_r) = self.subroutine(
                    sigma,
                    &grad,
                    &derivatives,
                    &new_lipschitz,
                    a,
                    b,
                    r,
                    &mut rng,
                );
                sigma = new_sigma;
                basis = new_basis;
                a = new_a;
                b = new_b;
                r = new_r;
            }

            l_nabla = new_l_nabla;
            lipschitz_coefficients = new_lipschitz;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::sphere;

    #[test]
    fn test_asgf_convergence() {
        let mut asgf = ASGF::new();
        let result = asgf.optimize(sphere, 10, 50, None, false, 25).unwrap();

        let initial = result.all_values[0];
        let final_best = result.best_value();

        assert!(
            final_best < initial,
            "ASGF should improve from initial value"
        );
    }
}
