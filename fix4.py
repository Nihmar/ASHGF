p = "src/algorithms/ashgf_ng.rs"
with open(p) as f:
    c = f.read()

old = """            // ---- Safeguarded step with simple backtracking ----
            let mut step = self.compute_step_size();
            let v_old = self.velocity.clone().unwrap_or_else(|| Array1::zeros(dim));
            // Nesterov look-ahead: y_k = x_k + mu * v_k
            let x_look = &x + &(self.mu * &v_old);
            let grad = self.grad_estimator(&x_look, f, rng);

            if !grad.iter().all(|v| v.is_finite()) {
                tracing::warn!("iter={}: gradient contains NaN/inf - terminating", i);
                break;
            }

            let mut x_new;
            let mut current_val;
            let max_bt = 3;
            for bt in 0..=max_bt {
                // Nesterov velocity update: v_{k+1} = mu * v_k + step * grad
                let v_new = &(self.mu * &v_old) + &(step * &grad);
                x_new = if options.maximize {
                    &x + &v_new
                } else {
                    &x - &v_new
                };

                if !x_new.iter().all(|v| v.is_finite()) {
                    tracing::warn!("iter={}: x contains NaN/inf - terminating", i);
                    break;
                }

                current_val = f(&x_new);

                if !current_val.is_finite() {
                    if bt < max_bt {
                        step /= 2.0;
                        self.sigma /= 2.0;
                        tracing::debug!("iter={} bt={}: f(x) not finite, sigma halved to {:.4e}", i, bt, self.sigma);
                        continue;
                    }
                    tracing::warn!("iter={}: f(x) = {} - terminating", i, current_val);
                    break;
                }

                // Accept if function decreased, or if we exhausted backtracks
                if options.maximize || current_val <= f_prev || bt >= max_bt {
                    break;
                }
                // Otherwise backtrack
                step /= 2.0;
                tracing::debug!("iter={} bt={}: f increased {:.3e} -> {:.3e}, step halved", i, bt, f_prev, current_val);
            }

            let grad_accepted = Some(grad);"""

new = """            // ---- Safeguarded step: gradient re-estimated inside backtracking ----
            let v_old = self.velocity.clone().unwrap_or_else(|| Array1::zeros(dim));
            let mut x_new = x.clone();
            let mut current_val = f_prev;
            let mut grad_accepted: Option<Array1<f64>> = None;
            let max_bt = 3;

            for bt in 0..=max_bt {
                let step = self.compute_step_size();
                let x_look = &x + &(self.mu * &v_old);
                let grad = self.grad_estimator(&x_look, f, rng);

                if !grad.iter().all(|v| v.is_finite()) {
                    tracing::debug!("iter={} bt={}: gradient NaN/inf, halving sigma", i, bt);
                    self.sigma /= 2.0;
                    continue;
                }

                let v_new = &(self.mu * &v_old) + &(step * &grad);
                x_new = if options.maximize { &x + &v_new } else { &x - &v_new };

                if !x_new.iter().all(|v| v.is_finite()) {
                    self.sigma /= 2.0;
                    continue;
                }

                current_val = f(&x_new);

                if !current_val.is_finite() || (!options.maximize && current_val > f_prev && bt < max_bt) {
                    self.sigma /= 2.0;
                    tracing::debug!("iter={} bt={}: bad step, sigma halved to {:.4e}", i, bt, self.sigma);
                    continue;
                }

                grad_accepted = Some(grad);
                break;
            }"""

assert old in c, "old text not found"
c = c.replace(old, new)
with open(p, "w") as f:
    f.write(c)
print("OK")
