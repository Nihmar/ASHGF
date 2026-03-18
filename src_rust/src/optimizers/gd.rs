use crate::optimizers::base::{Optimizer, OptimizerResult};
use rand::prelude::*;
use rand::rngs::StdRng;

pub struct GD {
    pub lr: f64,
    pub sigma: f64,
    seed: u64,
    pub eps: f64,
}

impl GD {
    pub fn new() -> Self {
        Self {
            lr: 1e-4,
            sigma: 1e-4,
            seed: 2003,
            eps: 1e-8,
        }
    }

    pub fn with_params(lr: f64, sigma: f64, seed: u64, eps: f64) -> Self {
        Self {
            lr,
            sigma,
            seed,
            eps,
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn grad_estimator_vectorized<F: Fn(&[f64]) -> f64 + Copy>(&self, x: &[f64], f: F) -> Vec<f64> {
        let dim = x.len();
        let mut rng = StdRng::seed_from_u64(self.seed);

        let directions: Vec<Vec<f64>> = (0..dim)
            .map(|_| (0..dim).map(|_| rng.gen::<f64>()).collect())
            .collect();

        let mut evals_plus = Vec::with_capacity(dim);
        let mut evals_minus = Vec::with_capacity(dim);

        for i in 0..dim {
            let mut point_plus = x.to_vec();
            for j in 0..dim {
                point_plus[j] += self.sigma * directions[i][j];
            }
            evals_plus.push(f(&point_plus));

            let mut point_minus = x.to_vec();
            for j in 0..dim {
                point_minus[j] -= self.sigma * directions[i][j];
            }
            evals_minus.push(f(&point_minus));
        }

        let diffs: Vec<f64> = evals_plus
            .iter()
            .zip(evals_minus.iter())
            .map(|(a, b)| a - b)
            .collect();

        let mut grad = vec![0.0; dim];
        for i in 0..dim {
            for j in 0..dim {
                grad[j] += diffs[i] * directions[i][j];
            }
        }

        for g in grad.iter_mut() {
            *g /= 2.0 * self.sigma * dim as f64;
        }

        grad
    }
}

impl Default for GD {
    fn default() -> Self {
        Self::new()
    }
}

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

        if debug {
            println!(
                "algorithm: gd  function: custom  dimension: {}  initial value: {}",
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

            let grad = self.grad_estimator_vectorized(&x, function);

            for j in 0..dim {
                x[j] -= self.lr * grad[j];
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
    fn test_gd_convergence() {
        let mut gd = GD::new();
        let (best, all) = gd.optimize(sphere, 10, 100, None, false, 25);

        let initial = all[0];
        let final_best = best.last().unwrap()[1];

        assert!(final_best < initial, "GD should improve from initial value");
    }
}
