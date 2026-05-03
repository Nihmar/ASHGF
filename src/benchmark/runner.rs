//! Benchmark orchestration: run multiple algorithms on multiple functions
//! and dimensions, collecting results for analysis.

use std::collections::HashMap;

use crate::algorithms::base::{OptimizeOptions, Optimizer};
use crate::benchmark::plot::RunResultWithHistory;
use crate::functions::{get_function, list_functions, TestFunction};
use crate::utils::SeededRng;

/// Result of a single benchmark run.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub algorithm: String,
    pub function: String,
    pub dim: usize,
    pub final_value: f64,
    pub best_value: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Run benchmarks across algorithms, functions, and dimensions.
///
/// Returns a map `dim -> Vec<RunResult>`.
pub fn run_benchmarks(
    algorithms: &mut [(&str, &mut dyn Optimizer)],
    pattern: Option<&str>,
    dimensions: &[usize],
    max_iter: usize,
    seed: u64,
    patience: Option<usize>,
    ftol: Option<f64>,
) -> HashMap<usize, Vec<RunResult>> {
    let (results, _) = run_benchmarks_with_history(
        algorithms, pattern, dimensions, max_iter, seed, patience, ftol,
    );
    results
}

/// Run benchmarks with per-iteration history for plotting.
///
/// Returns `(dim -> Vec<RunResult>, Vec<RunResultWithHistory>)`.
pub fn run_benchmarks_with_history(
    algorithms: &mut [(&str, &mut dyn Optimizer)],
    pattern: Option<&str>,
    dimensions: &[usize],
    max_iter: usize,
    seed: u64,
    patience: Option<usize>,
    ftol: Option<f64>,
) -> (HashMap<usize, Vec<RunResult>>, Vec<RunResultWithHistory>) {
    let all_funcs = list_functions();
    let func_names: Vec<&str> = if let Some(pat) = pattern {
        let pat_lower = pat.to_lowercase();
        all_funcs
            .into_iter()
            .filter(|n| n.to_lowercase().contains(&pat_lower))
            .collect()
    } else {
        all_funcs.clone()
    };

    let mut results: HashMap<usize, Vec<RunResult>> = HashMap::new();
    let mut history: Vec<RunResultWithHistory> = Vec::new();

    for &dim in dimensions {
        let mut dim_results = Vec::new();

        for func_name in &func_names {
            let f: TestFunction = match get_function(func_name) {
                Some(f) => f,
                None => continue,
            };

            for (algo_name, algo) in algorithms.iter_mut() {
                let mut rng = SeededRng::new(seed);
                let options = OptimizeOptions {
                    max_iter,
                    patience,
                    ftol,
                    ..Default::default()
                };
                let result = algo.optimize(&f, dim, None, &options, &mut rng);

                let best = result
                    .best_values
                    .last()
                    .map(|(_, v)| *v)
                    .unwrap_or(f64::NAN);

                dim_results.push(RunResult {
                    algorithm: algo_name.to_string(),
                    function: func_name.to_string(),
                    dim,
                    final_value: *result.all_values.last().unwrap_or(&f64::NAN),
                    best_value: best,
                    iterations: result.iterations,
                    converged: result.converged,
                });

                history.push(RunResultWithHistory {
                    algorithm: algo_name.to_string(),
                    function: func_name.to_string(),
                    dim,
                    values: result.all_values.clone(),
                    best_value: best,
                    iterations: result.iterations,
                    converged: result.converged,
                });
            }
        }

        results.insert(dim, dim_results);
    }

    (results, history)
}
