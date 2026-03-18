use crate::optimizers::base::{Optimizer, OptimizerError, OptimizerPoint, OptimizerResult};
use nalgebra::{DMatrix, DVector};
use rand::distributions::Distribution;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

pub struct ASEBO {
    pub lr: f64,
    pub sigma: f64,
    pub k: usize,
    pub thresh: f64,
    seed: u64,
    pub eps: f64,
}

impl ASEBO {
    pub fn new() -> Self {
        Self {
            lr: 1e-4,
            sigma: 1e-4,
            k: 50,
            thresh: 0.995,
            seed: 2003,
            eps: 1e-8,
        }
    }

    fn sample_directions(
        &self,
        n: usize,
        dim: usize,
        u_act: Option<&DMatrix<f64>>,
        p: f64,
    ) -> Vec<Vec<f64>> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut directions = Vec::with_capacity(n);

        for _ in 0..n {
            let mut v: Vec<f64> = (0..dim).map(|_| rng.gen::<f64>()).collect();

            let raw = if let Some(u) = u_act {
                let u_t = u.transpose();
                let proj = u * (u_t * DVector::from_vec(v.clone()));
                if rng.gen::<f64>() < p {
                    proj.as_slice().to_vec()
                } else {
                    for i in 0..dim {
                        v[i] -= proj[i];
                    }
                    v
                }
            } else {
                v
            };

            let norm_raw: f64 = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
            let mut unit = if norm_raw < 1e-12 {
                (0..dim).map(|_| rng.gen::<f64>()).collect::<Vec<f64>>()
            } else {
                raw.iter().map(|x| x / norm_raw).collect()
            };

            let chi: f64 = (0..dim)
                .map(|_| rng.gen::<f64>().powi(2))
                .sum::<f64>()
                .sqrt();
            for vi in unit.iter_mut() {
                *vi *= chi;
            }

            directions.push(unit);
        }

        directions
    }

    fn compute_pca(&self, grad_buffer: &[Vec<f64>]) -> Option<(DMatrix<f64>, usize)> {
        if grad_buffer.len() < 2 {
            return None;
        }

        let dim = grad_buffer[0].len();
        let n = grad_buffer.len();

        let means: Vec<f64> = (0..dim)
            .map(|j| grad_buffer.iter().map(|g| g[j]).sum::<f64>() / n as f64)
            .collect();

        let mut data_matrix = DMatrix::zeros(n, dim);
        for (i, g) in grad_buffer.iter().enumerate() {
            for j in 0..dim {
                data_matrix[(i, j)] = g[j] - means[j];
            }
        }

        let cov = (data_matrix.transpose() * data_matrix) / (n - 1) as f64;

        let eigen = cov.symmetric_eigen();
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        let total_var: f64 = eigenvalues.iter().sum();
        let mut cum_var = 0.0;
        let mut r = 0;
        for (i, &ev) in eigenvalues.iter().enumerate() {
            cum_var += ev / total_var;
            r = i + 1;
            if cum_var >= self.thresh {
                break;
            }
        }

        let mut u_act_vec: Vec<Vec<f64>> = Vec::with_capacity(r);
        for i in 0..r {
            let mut col = Vec::with_capacity(dim);
            for j in 0..dim {
                col.push(eigenvectors[(j, i)]);
            }
            u_act_vec.push(col);
        }

        let mut u_act = DMatrix::zeros(dim, r);
        for i in 0..r {
            for j in 0..dim {
                u_act[(j, i)] = u_act_vec[i][j];
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

        let mut grad_buffer: Vec<Vec<f64>> = Vec::new();
        let mut p = 0.5;

        if debug {
            println!(
                "algorithm: asebo  function: custom  dimension: {}  initial value: {}",
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

            let (n_samples, u_act) = if i <= self.k {
                (dim, None)
            } else {
                match self.compute_pca(&grad_buffer) {
                    Some((u, r)) => (r, Some(u)),
                    None => (dim, None),
                }
            };

            let directions = self.sample_directions(n_samples, dim, u_act.as_ref(), p);

            let mut evals_plus = Vec::with_capacity(n_samples);
            let mut evals_minus = Vec::with_capacity(n_samples);

            for dir in &directions {
                let mut point_plus = x.clone();
                for j in 0..dim {
                    point_plus[j] += self.sigma * dir[j];
                }
                evals_plus.push(function(&point_plus));

                let mut point_minus = x.clone();
                for j in 0..dim {
                    point_minus[j] -= self.sigma * dir[j];
                }
                evals_minus.push(function(&point_minus));
            }

            let diffs: Vec<f64> = evals_plus
                .iter()
                .zip(evals_minus.iter())
                .map(|(a, b)| a - b)
                .collect();

            let mut grad = vec![0.0; dim];
            for idx in 0..n_samples {
                for j in 0..dim {
                    grad[j] += diffs[idx] * directions[idx][j];
                }
            }

            for g in grad.iter_mut() {
                *g /= 2.0 * self.sigma * n_samples as f64;
            }

            grad_buffer.push(grad.clone());
            if grad_buffer.len() > self.k {
                grad_buffer.remove(0);
            }

            if let Some(ref u) = u_act {
                let u_t = u.transpose();
                let grad_vec = DVector::from_vec(grad.clone());
                let grad_act = u * (u_t * grad_vec);
                let mut grad_perp = grad.clone();
                for j in 0..dim {
                    grad_perp[j] -= grad_act[j];
                }

                let s_act: f64 = grad_act.as_slice().iter().map(|x| x * x).sum();
                let s_perp: f64 = grad_perp.iter().map(|x| x * x).sum();
                let d_act = u.ncols();
                let d_perp = dim - d_act;

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

                let p_new = if sqrt_act + sqrt_perp > 0.0 {
                    sqrt_act / (sqrt_act + sqrt_perp)
                } else {
                    0.5
                };

                p = p_new.clamp(0.01, 0.99);
            }

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

            let norm_diff: f64 = x
                .iter()
                .zip(best_points.last().unwrap().x.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if norm_diff < self.eps {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::sphere;

    #[test]
    fn test_asebo_convergence() {
        let mut asebo = ASEBO::new();
        let result = asebo.optimize(sphere, 10, 100, None, false, 25).unwrap();

        let initial = result.all_values[0];
        let final_best = result.best_value();

        assert!(
            final_best < initial,
            "ASEBO should improve from initial value"
        );
    }
}
