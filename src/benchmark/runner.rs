//! Benchmark orchestration: run multiple algorithms on multiple functions
//! and dimensions, collecting results for analysis.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::algorithms::base::{OptimizeOptions, Optimizer, ProgressCallback};
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
    n_jobs: usize,
    quiet: bool,
) -> HashMap<usize, Vec<RunResult>> {
    let (results, _) = run_benchmarks_with_history(
        algorithms, pattern, dimensions, max_iter, seed, patience, ftol, n_jobs, quiet,
    );
    results
}

/// Run benchmarks with per-iteration history for plotting.
///
/// When `n_jobs > 1`, runs tasks in parallel using rayon.  In that case,
/// the `algorithms` slice is ignored and each task creates its own
/// algorithm instance via `Default::default()`.
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
    n_jobs: usize,
    quiet: bool,
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

    // ---- Sequential path (n_jobs <= 1) ----
    if n_jobs <= 1 {
        let mut results: HashMap<usize, Vec<RunResult>> = HashMap::new();
        let mut history: Vec<RunResultWithHistory> = Vec::new();

        let multi = if quiet {
            None
        } else {
            Some(MultiProgress::new())
        };

        let style = ProgressStyle::default_bar()
            .template("{prefix:.bold} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-");

        for &dim in dimensions {
            let mut dim_results = Vec::new();

            for func_name in &func_names {
                let f: TestFunction = match get_function(func_name) {
                    Some(f) => f,
                    None => continue,
                };

                for (algo_name, algo) in algorithms.iter_mut() {
                    let pb = multi.as_ref().map(|m| {
                        let bar = m.add(ProgressBar::new(max_iter as u64));
                        bar.set_style(style.clone());
                        bar.set_prefix(format!(
                            "dim={:<4} {:<25} {:<10}",
                            dim, func_name, algo_name
                        ));
                        bar
                    });

                    let progress_cb: Option<ProgressCallback> =
                        pb.as_ref().map(|bar| {
                            let bar = bar.clone();
                            Arc::new(move |iter: usize, fval: f64| {
                                bar.set_position(iter as u64);
                                bar.set_message(format!("f={:.4e}", fval));
                            }) as ProgressCallback
                        });

                    let mut rng = SeededRng::new(seed);
                    let options = OptimizeOptions {
                        max_iter,
                        patience,
                        ftol,
                        progress_cb,
                        ..Default::default()
                    };
                    let result = algo.optimize(&f, dim, None, &options, &mut rng);

                    if let Some(bar) = pb {
                        bar.finish_and_clear();
                    }

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

        return (results, history);
    }

    // ---- Parallel path (n_jobs > 1) ----
    let mut tasks: Vec<(String, String, usize)> = Vec::new();
    for &dim in dimensions {
        for func_name in &func_names {
            for (algo_name, _) in algorithms.iter() {
                tasks.push((algo_name.to_string(), func_name.to_string(), dim));
            }
        }
    }

    let total_tasks = tasks.len();
    let pool_size = n_jobs;
    if quiet {
        tracing::info!(
            "BENCH | {} tasks ({} algos x {} funcs x {} dims) | workers={}",
            total_tasks,
            algorithms.len(),
            func_names.len(),
            dimensions.len(),
            pool_size,
        );
    }

    // Thread-safe result collectors
    let results: Mutex<HashMap<usize, Vec<RunResult>>> = Mutex::new(HashMap::new());
    let history: Mutex<Vec<RunResultWithHistory>> = Mutex::new(Vec::new());

    // ---- TUI setup ----
    let multi = if quiet {
        None
    } else {
        Some(MultiProgress::new())
    };
    let completed = Arc::new(AtomicUsize::new(0));

    let overall_pb = multi.as_ref().map(|m| {
        let pb = m.add(ProgressBar::new(total_tasks as u64));
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{prefix:.bold} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("##-"),
        );
        pb.set_prefix("Benchmark");
        pb
    });

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_size)
        .build()
        .expect("Failed to build rayon thread pool");

    pool.install(|| {
        rayon::scope(|s| {
            for task in &tasks {
                let results = &results;
                let history = &history;
                let algo_name = task.0.clone();
                let func_name = task.1.clone();
                let dim = task.2;
                let pb = overall_pb.clone();
                let completed = Arc::clone(&completed);

                let label = format!(
                    "dim={:<4} {:<25} {:<10}",
                    dim, func_name, algo_name
                );

                s.spawn(move |_| {
                    let progress_cb: Option<ProgressCallback> =
                        pb.as_ref().map(|bar| {
                            let bar = bar.clone();
                            let label = label.clone();
                            Arc::new(move |iter: usize, fval: f64| {
                                bar.set_message(format!(
                                    "{} | iter {}/{} f={:.4e}",
                                    label, iter, max_iter, fval
                                ));
                            }) as ProgressCallback
                        });

                    let f: TestFunction = match get_function(&func_name) {
                        Some(f) => f,
                        None => return,
                    };

                    // Create algorithm instance from its name
                    let mut algo: Box<dyn Optimizer> = match algo_name.as_str() {
                        "GD" => Box::new(crate::algorithms::GD::default()),
                        "SGES" => Box::new(crate::algorithms::SGES::default()),
                        "ASGF" => Box::new(crate::algorithms::ASGF::default()),
                        "ASGF-2S" => Box::new(crate::algorithms::Asgf2s::default()),
                        "ASGF-2SW" => Box::new(crate::algorithms::Asgf2sw::default()),
                        "ASHGF" => Box::new(crate::algorithms::ASHGF::default()),
                        "ASHGF-NG" => Box::new(crate::algorithms::ASHGFNG::default()),
                        "ASHGF-S" => Box::new(crate::algorithms::ASHGFS::default()),
                        "ASEBO" => Box::new(crate::algorithms::ASEBO::default()),
                        _ => return,
                    };

                    let mut rng = SeededRng::new(seed);
                    let options = OptimizeOptions {
                        max_iter,
                        patience,
                        ftol,
                        progress_cb,
                        ..Default::default()
                    };
                    let result = algo.optimize(&f, dim, None, &options, &mut rng);

                    let best = result
                        .best_values
                        .last()
                        .map(|(_, v)| *v)
                        .unwrap_or(f64::NAN);

                    let run_result = RunResult {
                        algorithm: algo_name.clone(),
                        function: func_name.clone(),
                        dim,
                        final_value: *result.all_values.last().unwrap_or(&f64::NAN),
                        best_value: best,
                        iterations: result.iterations,
                        converged: result.converged,
                    };

                    let hist_entry = RunResultWithHistory {
                        algorithm: algo_name.clone(),
                        function: func_name.clone(),
                        dim,
                        values: result.all_values.clone(),
                        best_value: best,
                        iterations: result.iterations,
                        converged: result.converged,
                    };

                    let mut res = results.lock().unwrap();
                    res.entry(dim).or_default().push(run_result);
                    drop(res);

                    history.lock().unwrap().push(hist_entry);

                    // Update overall progress
                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    if let Some(ref bar) = pb {
                        bar.set_position(done as u64);
                    }
                });
            }
        });
    });

    let results = results.into_inner().unwrap();
    let history = history.into_inner().unwrap();

    (results, history)
}
