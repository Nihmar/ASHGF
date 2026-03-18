use crate::optimizers::base::{Optimizer, OptimizerResult};
use nalgebra::DMatrix;
use rand::prelude::*;
use rand::rngs::StdRng;

pub struct ASHGF {
    pub k1: f64,
    pub k2: f64,
    pub alpha: f64,
    pub delta: f64,
    pub t: usize,
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
    gamma_sigma_plus: f64,
    gamma_sigma_minus: f64,
    r: usize,
    ro: f64,
    sigma_zero: f64,
}

impl ASHGF {
    pub fn new() -> Self {
        Self {
            k1: 0.9,
            k2: 0.1,
            alpha: 0.5,
            delta: 1.1,
            t: 50,
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
            gamma_sigma_plus: 1.0 / 0.9,
            gamma_sigma_minus: 0.9,
            r: 10,
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
        m_dirs: usize,
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

        let m_eff = m_dirs.max(1).min(dim);
        let l_g = new_lipschitz[..m_eff]
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        let l_nabla_new = (1.0 - self.gamma_l) * l_g + self.gamma_l * l_nabla;
        let lr = sigma / l_nabla_new.max(1e-12);

        (grad, new_lipschitz, lr, derivatives, l_nabla_new)
    }

    fn compute_directions_sges(
        &self,
        dim: usize,
        _g_history: &[Vec<f64>],
        alpha: f64,
    ) -> (DMatrix<f64>, usize) {
        let mut rng = StdRng::seed_from_u64(self.seed);

        let m_dirs = ((rng.gen::<f64>() * dim as f64 * alpha) as usize)
            .max(0)
            .min(dim);

        let mut dirs = Vec::new();

        if m_dirs > 0 {
            for _ in 0..m_dirs {
                let mut dir = Vec::new();
                for _ in 0..dim {
                    dir.push(rng.gen::<f64>() - 0.5);
                }
                let norm: f64 = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    for d in dir.iter_mut() {
                        *d /= norm;
                    }
                }
                dirs.push(dir);
            }
        }

        let remaining = dim - m_dirs;
        for _ in 0..remaining {
            let mut dir = Vec::new();
            for _ in 0..dim {
                dir.push(rng.gen::<f64>() - 0.5);
            }
            let norm: f64 = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for d in dir.iter_mut() {
                    *d /= norm;
                }
            }
            dirs.push(dir);
        }

        let mut basis = DMatrix::zeros(dim, dim);
        for (i, dir) in dirs.iter().enumerate() {
            for j in 0..dim {
                basis[(i, j)] = dir[j];
            }
        }

        (basis, m_dirs)
    }
}

impl Default for ASHGF {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer for ASHGF {
    fn name(&self) -> &'static str {
        "Adaptive Stochastic Historical Gradient-Free"
    }

    fn optimize<F>(
        &mut self,
        function: F,
        dim: usize,
        it: usize,
        x_init: Option<&[f64]>,
        debug: bool,
        itprint: usize,
    ) -> OptimizerResult
    where
        F: Fn(&[f64]) -> f64 + Copy,
    {
        let mut rng = StdRng::seed_from_u64(self.seed);

        let mut x: Vec<f64> = match x_init {
            Some(init) => init.to_vec(),
            None => (0..dim).map(|_| rng.gen::<f64>()).collect(),
        };

        let mut current_val = function(&x);
        let mut best_value = current_val;
        let mut best_values = vec![x.clone()];
        best_values[0].push(best_value);
        let mut all_values = vec![current_val];

        let norm_x: f64 = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        self.sigma_zero = norm_x / 10.0;
        let mut sigma = self.sigma_zero;
        let mut a = self.a;
        let mut b = self.b;
        let mut r = self.r;
        let mut l_nabla = 0.0;
        let mut m_dirs = dim;
        let mut lipschitz_coefficients = vec![1.0; dim];

        let mut basis = DMatrix::identity(dim, dim);

        let mut g_history: Vec<Vec<f64>> = Vec::new();

        if debug {
            println!(
                "algorithm: ashgf  function: custom  dimension: {}  initial value: {}",
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
                m_dirs,
                current_val,
            );

            if !grad.iter().all(|g| g.is_finite()) || !lr.is_finite() {
                if debug {
                    println!(
                        "Warning: non-finite gradient or learning rate at iteration {}",
                        i
                    );
                }
                break;
            }

            g_history.push(grad.clone());
            if g_history.len() > self.t {
                g_history.remove(0);
            }

            for j in 0..dim {
                x[j] -= lr * grad[j];
            }

            current_val = function(&x);
            all_values.push(current_val);

            if current_val < best_value {
                best_value = current_val;
                let mut new_best = x.clone();
                new_best.push(best_value);
                best_values.push(new_best);
            }

            let norm_diff: f64 = x
                .iter()
                .zip(best_values.last().unwrap().iter().take(dim))
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if norm_diff < self.eps {
                break;
            }

            if i < self.t {
                m_dirs = dim;
            } else {
                let m_eff = m_dirs.max(1).min(dim);
                let vals_g: Vec<f64> = (0..m_eff).map(|j| derivatives[j]).collect();

                if !vals_g.is_empty() {
                    let r_g_val =
                        vals_g.iter().fold(0.0_f64, |sum, v| sum + v) / vals_g.len() as f64;
                    if r_g_val < 0.0 {
                        self.alpha = (self.delta * self.alpha).min(self.k1);
                    } else {
                        self.alpha = (self.alpha / self.delta).max(self.k2);
                    }
                }

                m_dirs = ((self.alpha * dim as f64) as usize).max(1).min(dim);
            }

            if r > 0 && sigma < self.ro * self.sigma_zero {
                basis = DMatrix::identity(dim, dim);
                sigma = self.sigma_zero;
                a = self.a;
                b = self.b;
                r -= 1;
                m_dirs = dim;
            } else if i >= self.t {
                let (new_basis, new_m) = self.compute_directions_sges(dim, &g_history, self.alpha);
                basis = new_basis;
                m_dirs = new_m;
            } else {
                basis = DMatrix::identity(dim, dim);
                m_dirs = dim;
            }

            let mut ratio = 0.0;
            for j in 0..dim {
                let denom = new_lipschitz[j].max(1e-10);
                let r = derivatives[j].abs() / denom;
                if r > ratio {
                    ratio = r;
                }
            }

            if ratio < a {
                sigma *= self.gamma_sigma_minus;
                a *= self.a_minus;
            } else if ratio > b {
                sigma *= self.gamma_sigma_plus;
                b *= self.b_plus;
            } else {
                a *= self.a_plus;
                b *= self.b_minus;
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

        (best_values, all_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::sphere;

    #[test]
    fn test_ashgf_convergence() {
        let mut ashgf = ASHGF::new();
        let (best, all) = ashgf.optimize(sphere, 10, 100, None, false, 25);

        let initial = all[0];
        let final_best = best.last().unwrap()[1];

        assert!(
            final_best < initial,
            "ASHGF should improve from initial value"
        );
    }
}
