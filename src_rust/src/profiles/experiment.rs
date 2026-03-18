use crate::functions::Function;
use crate::optimizers::{Optimizer, ASEBO, ASGF, ASHGF, GD, SGES};
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use std::collections::HashMap;

use crate::profiles::io::{load_results, save_results, RunResult};

#[derive(Clone, Copy, Debug)]
pub enum Algorithm {
    GD,
    SGES,
    ASGF,
    ASHGF,
    ASEBO,
}

impl Algorithm {
    pub fn name(&self) -> &'static str {
        match self {
            Algorithm::GD => "GD",
            Algorithm::SGES => "SGES",
            Algorithm::ASGF => "ASGF",
            Algorithm::ASHGF => "ASHGF",
            Algorithm::ASEBO => "ASEBO",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "GD" => Some(Algorithm::GD),
            "SGES" => Some(Algorithm::SGES),
            "ASGF" => Some(Algorithm::ASGF),
            "ASHGF" => Some(Algorithm::ASHGF),
            "ASEBO" => Some(Algorithm::ASEBO),
            _ => None,
        }
    }
}

pub struct ExperimentResult {
    pub function: String,
    pub algorithm: String,
    pub run: i32,
    pub values: Vec<f64>,
    pub status: String,
    pub error_msg: Option<String>,
    pub warnings: Option<String>,
}

impl ExperimentResult {
    pub fn success(
        function: String,
        algorithm: String,
        run: i32,
        values: Vec<f64>,
        warnings: Option<String>,
    ) -> Self {
        Self {
            function,
            algorithm,
            run,
            values,
            status: "success".to_string(),
            error_msg: None,
            warnings,
        }
    }

    pub fn failed(function: String, algorithm: String, run: i32, error_msg: String) -> Self {
        Self {
            function,
            algorithm,
            run,
            values: Vec::new(),
            status: "failed".to_string(),
            error_msg: Some(error_msg),
            warnings: None,
        }
    }

    pub fn to_run_result(self) -> RunResult {
        RunResult {
            function: self.function,
            algorithm: self.algorithm,
            run: self.run,
            values: self.values,
            warnings: self.warnings,
        }
    }
}

pub struct Runner {
    dim: usize,
    iters: usize,
    n_runs: usize,
    seed: u64,
    batch_size: usize,
}

impl Runner {
    pub fn new(dim: usize, iters: usize, n_runs: usize, seed: u64) -> Self {
        Self {
            dim,
            iters,
            n_runs,
            seed,
            batch_size: 20,
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn run(
        &self,
        functions: &[Function],
        algorithms: &[Algorithm],
        overwrite: bool,
        verbose: bool,
    ) -> anyhow::Result<Vec<RunResult>> {
        let mut all_results = load_results(self.dim as u32).unwrap_or_default();

        let mut pending: Vec<RunResult> = Vec::new();
        let mut completed = 0;
        let mut skipped = 0;
        let mut failed = 0;

        let total_tasks = functions.len() * algorithms.len() * self.n_runs;

        if verbose {
            println!(
                "Running {} experiments for dim={}...",
                total_tasks, self.dim
            );
        }

        for func in functions {
            let func_name = func.name();

            for alg in algorithms {
                let existing_runs: Vec<i32> = all_results
                    .iter()
                    .filter(|r| r.function == func_name && r.algorithm == alg.name())
                    .map(|r| r.run)
                    .collect();

                for run_idx in 0..self.n_runs {
                    let run_id = run_idx as i32;
                    if !overwrite && existing_runs.contains(&run_id) {
                        skipped += 1;
                        continue;
                    }

                    let task_seed = self.seed.wrapping_add((func_name.len() + run_idx) as u64);

                    let result =
                        run_single_experiment(*func, *alg, self.dim, self.iters, run_id, task_seed);

                    match result.status.as_str() {
                        "success" => {
                            pending.push(result.to_run_result());
                            if verbose {
                                println!("[{}] {} run {}: OK", alg.name(), func_name, run_id);
                            }
                        }
                        _ => {
                            failed += 1;
                            if verbose {
                                println!(
                                    "[{}] {} run {}: FAILED - {}",
                                    alg.name(),
                                    func_name,
                                    run_id,
                                    result.error_msg.as_deref().unwrap_or("unknown")
                                );
                            }
                        }
                    }

                    completed += 1;

                    if pending.len() >= self.batch_size {
                        all_results.append(&mut pending);
                        save_results(&all_results, self.dim as u32)?;
                    }

                    if verbose && completed % 100 == 0 {
                        println!("Progress: {}/{}", completed, total_tasks);
                    }
                }
            }
        }

        all_results.append(&mut pending);
        save_results(&all_results, self.dim as u32)?;

        if verbose {
            println!(
                "Completed: {} success, {} failed, {} skipped",
                completed - failed - skipped,
                failed,
                skipped
            );
        }

        Ok(all_results)
    }
}

fn run_single_experiment(
    function: Function,
    algorithm: Algorithm,
    dim: usize,
    iters: usize,
    run_id: i32,
    seed: u64,
) -> ExperimentResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let x_init: Vec<f64> = (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect();

    let func_closure = |x: &[f64]| function.evaluate(x);

    match algorithm {
        Algorithm::GD => {
            let mut gd = GD::with_params(1e-4, 1e-4, seed, 1e-8).unwrap();
            match gd.optimize(func_closure, dim, iters, Some(&x_init), false, 100) {
                Ok(result) => ExperimentResult::success(
                    function.name().to_string(),
                    "GD".to_string(),
                    run_id,
                    result.all_values,
                    None,
                ),
                Err(e) => ExperimentResult::failed(
                    function.name().to_string(),
                    "GD".to_string(),
                    run_id,
                    e.to_string(),
                ),
            }
        }
        Algorithm::SGES => {
            let mut sges =
                SGES::with_params(1e-4, 1e-4, 50, 0.9, 0.1, 0.5, 1.1, 50, seed, 1e-8).unwrap();
            match sges.optimize(func_closure, dim, iters, Some(&x_init), false, 100) {
                Ok(result) => ExperimentResult::success(
                    function.name().to_string(),
                    "SGES".to_string(),
                    run_id,
                    result.all_values,
                    None,
                ),
                Err(e) => ExperimentResult::failed(
                    function.name().to_string(),
                    "SGES".to_string(),
                    run_id,
                    e.to_string(),
                ),
            }
        }
        Algorithm::ASGF => {
            let mut asgf = ASGF::new();
            match asgf.optimize(func_closure, dim, iters, Some(&x_init), false, 100) {
                Ok(result) => ExperimentResult::success(
                    function.name().to_string(),
                    "ASGF".to_string(),
                    run_id,
                    result.all_values,
                    None,
                ),
                Err(e) => ExperimentResult::failed(
                    function.name().to_string(),
                    "ASGF".to_string(),
                    run_id,
                    e.to_string(),
                ),
            }
        }
        Algorithm::ASHGF => {
            let mut ashgf = ASHGF::new();
            match ashgf.optimize(func_closure, dim, iters, Some(&x_init), false, 100) {
                Ok(result) => ExperimentResult::success(
                    function.name().to_string(),
                    "ASHGF".to_string(),
                    run_id,
                    result.all_values,
                    None,
                ),
                Err(e) => ExperimentResult::failed(
                    function.name().to_string(),
                    "ASHGF".to_string(),
                    run_id,
                    e.to_string(),
                ),
            }
        }
        Algorithm::ASEBO => {
            let mut asebo = ASEBO::new();
            match asebo.optimize(func_closure, dim, iters, Some(&x_init), false, 100) {
                Ok(result) => ExperimentResult::success(
                    function.name().to_string(),
                    "ASEBO".to_string(),
                    run_id,
                    result.all_values,
                    None,
                ),
                Err(e) => ExperimentResult::failed(
                    function.name().to_string(),
                    "ASEBO".to_string(),
                    run_id,
                    e.to_string(),
                ),
            }
        }
    }
}

pub fn get_default_functions() -> Vec<Function> {
    vec![
        Function::ExtendedFeudensteinAndRoth,
        Function::ExtendedTrigonometric,
        Function::ExtendedRosenbrock,
        Function::GeneralizedRosenbrock,
        Function::ExtendedWhiteAndHolst,
        Function::PerturbedQuadratic,
        Function::ExtendedTridiagonal1,
        Function::ExtendedHimmelblau,
        Function::GeneralizedWhiteAndHolst,
        Function::ExtendedPsc1,
        Function::PerturbedQuadraticDiagonal,
        Function::ExtendedHiebert,
        Function::ExtendedTridiagonal2,
        Function::AlmostPerturbedQuadratic,
        Function::Power,
        Function::Cube,
        Function::GeneralizedQuartic,
        Function::Ackley,
        Function::Griewank,
        Function::Levy,
        Function::Rastrigin,
        Function::Schwefel,
        Function::Sphere,
        Function::SumOfDifferentPowers,
        Function::Trid,
        Function::Zakharov,
    ]
}

pub fn analyze_results(dim: u32) -> anyhow::Result<()> {
    let results = load_results(dim)?;

    if results.is_empty() {
        println!("No results found for dim={}", dim);
        return Ok(());
    }

    let mut grouped: HashMap<(String, String), Vec<&Vec<f64>>> = HashMap::new();
    for r in &results {
        grouped
            .entry((r.function.clone(), r.algorithm.clone()))
            .or_default()
            .push(&r.values);
    }

    println!("\n{}", "=".repeat(60));
    println!("Summary for dim={}", dim);
    println!("{}", "=".repeat(60));
    println!(
        "{:30} {:15} {:>8} {:>15} {:>15}",
        "Function", "Algorithm", "Runs", "Mean Final", "Std Final"
    );
    println!("{}", "-".repeat(60));

    let mut keys: Vec<_> = grouped.keys().collect();
    keys.sort();

    for (func, alg) in keys {
        let values = &grouped[&(func.clone(), alg.clone())];
        let finals: Vec<f64> = values
            .iter()
            .filter(|v| !v.is_empty())
            .map(|v| *v.last().unwrap())
            .collect();

        if finals.is_empty() {
            continue;
        }

        let mean = finals.iter().sum::<f64>() / finals.len() as f64;
        let std = if finals.len() > 1 {
            let variance =
                finals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / finals.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        println!(
            "{:30} {:15} {:>8} {:>15.6e} {:>15.6e}",
            func,
            alg,
            finals.len(),
            mean,
            std
        );
    }

    Ok(())
}
