//! CLI entry point for ASHGF.

use std::process;

use std::fs;
use std::path::Path;

use ashgf::algorithms::{
    Asgf2s, Asgf2sw, OptimizeOptions, Optimizer, ASEBO, ASGF, ASHGF, ASHGFNG, ASHGFS, GD, SGES,
};
use ashgf::benchmark::plot::{plot_comparison_bars, plot_convergence_grid, plot_per_function};
use ashgf::benchmark::report::generate_report;
use ashgf::benchmark::runner::run_benchmarks_with_history;
use ashgf::cli::args::{AlgoName, Cli, Command};
use ashgf::functions::{get_function, list_functions};
use ashgf::utils::SeededRng;
use clap::Parser;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let exit_code = run(cli);
    process::exit(exit_code);
}

fn run(cli: Cli) -> i32 {
    match cli.command {
        Command::List => {
            let funcs = list_functions();
            println!("Available test functions ({}):", funcs.len());
            for name in &funcs {
                println!("  {name}");
            }
            0
        }

        Command::Run(args) => {
            let f = match get_function(&args.function) {
                Some(f) => f,
                None => {
                    eprintln!("Unknown function: '{}'", args.function);
                    eprintln!("Use 'ashgf list' to see available functions.");
                    return 1;
                }
            };

            let options = OptimizeOptions {
                max_iter: args.iter,
                patience: args.patience,
                ftol: args.ftol,
                ..Default::default()
            };

            let mut rng = SeededRng::new(args.seed);

            let result = match args.algo {
                AlgoName::Gd => {
                    let mut algo = GD::new(args.lr, args.sigma, 1e-8);
                    algo.optimize(&f, args.dim, None, &options, &mut rng)
                }
                AlgoName::Sges => {
                    let mut algo = SGES::new(args.lr, args.sigma, 0.9, 0.1, 0.5, 1.1, 50, 1e-8);
                    algo.optimize(&f, args.dim, None, &options, &mut rng)
                }
                AlgoName::Asgf => {
                    let mut algo = ASGF::default();
                    algo.optimize(&f, args.dim, None, &options, &mut rng)
                }
                AlgoName::Asgf2s => {
                    let mut algo = Asgf2s::default();
                    algo.optimize(&f, args.dim, None, &options, &mut rng)
                }
                AlgoName::Asgf2sw => {
                    let mut algo = Asgf2sw::default();
                    algo.optimize(&f, args.dim, None, &options, &mut rng)
                }
                AlgoName::Ashgf => {
                    let mut algo = ASHGF::default();
                    algo.optimize(&f, args.dim, None, &options, &mut rng)
                }
                AlgoName::AshgfS => {
                    let mut algo = ASHGFS::default();
                    algo.optimize(&f, args.dim, None, &options, &mut rng)
                }
                AlgoName::AshgfNg => {
                    let mut algo = ASHGFNG::default();
                    algo.optimize(&f, args.dim, None, &options, &mut rng)
                }
                AlgoName::Asebo => {
                    let mut algo = ASEBO::default();
                    algo.optimize(&f, args.dim, None, &options, &mut rng)
                }
            };

            println!(
                "Best value: {:.6e}",
                result
                    .best_values
                    .last()
                    .map(|(_, v)| *v)
                    .unwrap_or(f64::NAN)
            );
            println!("Iterations: {}", result.iterations);
            println!("Converged: {}", result.converged);
            0
        }

        Command::Compare(args) => {
            let f = match get_function(&args.function) {
                Some(f) => f,
                None => {
                    eprintln!("Unknown function: '{}'", args.function);
                    return 1;
                }
            };

            let options = OptimizeOptions {
                max_iter: args.iter,
                patience: args.patience,
                ftol: args.ftol,
                ..Default::default()
            };

            for algo_name in &args.algos {
                let mut rng = SeededRng::new(args.seed);

                let result = match algo_name {
                    AlgoName::Gd => {
                        let mut algo = GD::new(1e-4, 1e-4, 1e-8);
                        algo.optimize(&f, args.dim, None, &options, &mut rng)
                    }
                    AlgoName::Sges => {
                        let mut algo = SGES::default();
                        algo.optimize(&f, args.dim, None, &options, &mut rng)
                    }
                    AlgoName::Asgf => {
                        let mut algo = ASGF::default();
                        algo.optimize(&f, args.dim, None, &options, &mut rng)
                    }
                    AlgoName::Asgf2s => {
                        let mut algo = Asgf2s::default();
                        algo.optimize(&f, args.dim, None, &options, &mut rng)
                    }
                    AlgoName::Asgf2sw => {
                        let mut algo = Asgf2sw::default();
                        algo.optimize(&f, args.dim, None, &options, &mut rng)
                    }
                    AlgoName::Ashgf => {
                        let mut algo = ASHGF::default();
                        algo.optimize(&f, args.dim, None, &options, &mut rng)
                    }
                    AlgoName::AshgfS => {
                        let mut algo = ASHGFS::default();
                        algo.optimize(&f, args.dim, None, &options, &mut rng)
                    }
                    AlgoName::AshgfNg => {
                        let mut algo = ASHGFNG::default();
                        algo.optimize(&f, args.dim, None, &options, &mut rng)
                    }
                    AlgoName::Asebo => {
                        let mut algo = ASEBO::default();
                        algo.optimize(&f, args.dim, None, &options, &mut rng)
                    }
                };

                let best = result
                    .best_values
                    .last()
                    .map(|(_, v)| *v)
                    .unwrap_or(f64::NAN);
                let final_val = result.all_values.last().copied().unwrap_or(f64::NAN);
                let min_val = result
                    .all_values
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b));
                println!(
                    "{:>6}: final={:.6e}  best={:.6e}  min={:.6e}  iter={}",
                    format!("{:?}", algo_name).to_lowercase(),
                    final_val,
                    best,
                    min_val,
                    result.iterations,
                );
            }
            0
        }

        Command::Benchmark(args) => {
            let dims: Vec<usize> = if let Some(ref d) = args.dims {
                d.split(',').filter_map(|s| s.trim().parse().ok()).collect()
            } else if let Some(d) = args.dim {
                vec![d]
            } else {
                vec![100]
            };

            let algos_to_run: Vec<AlgoName> = args.algos.unwrap_or_else(|| {
                vec![
                    AlgoName::Gd,
                    AlgoName::Sges,
                    AlgoName::Asgf,
                    AlgoName::Asgf2s,
                    AlgoName::Asgf2sw,
                    AlgoName::Ashgf,
                    AlgoName::AshgfNg,
                    AlgoName::AshgfS,
                    AlgoName::Asebo,
                ]
            });

            let mut algorithms: Vec<(&str, &mut dyn Optimizer)> = Vec::new();
            let mut gd = GD::new(args.lr, args.sigma, 1e-8);
            let mut sges = SGES::new(args.lr, args.sigma, 0.9, 0.1, 0.5, 1.1, 50, 1e-8);
            let mut asgf = ASGF::default();
            let mut asgf_2s = Asgf2s::default();
            let mut asgf_2sw = Asgf2sw::default();
            let mut ashgf = ASHGF::default();
            let mut ashgf_ng = ASHGFNG::default();
            let mut ashgf_s = ASHGFS::default();
            let mut asebo = ASEBO::default();

            // Set parallelism from CLI (0 = auto-detect via rayon)
            gd.n_jobs = args.jobs;
            sges.n_jobs = args.jobs;
            asgf.n_jobs = args.jobs;
            asgf_2s.inner.n_jobs = args.jobs;
            asgf_2sw.inner.n_jobs = args.jobs;
            ashgf.n_jobs = args.jobs;
            ashgf_ng.n_jobs = args.jobs;
            ashgf_s.n_jobs = args.jobs;
            asebo.n_jobs = args.jobs;

            // Push each algorithm only once; borrow checker needs separate statements
            if algos_to_run.contains(&AlgoName::Gd) {
                algorithms.push(("GD", &mut gd));
            }
            if algos_to_run.contains(&AlgoName::Sges) {
                algorithms.push(("SGES", &mut sges));
            }
            if algos_to_run.contains(&AlgoName::Asgf) {
                algorithms.push(("ASGF", &mut asgf));
            }
            if algos_to_run.contains(&AlgoName::Asgf2s) {
                algorithms.push(("ASGF-2S", &mut asgf_2s));
            }
            if algos_to_run.contains(&AlgoName::Asgf2sw) {
                algorithms.push(("ASGF-2SW", &mut asgf_2sw));
            }
            if algos_to_run.contains(&AlgoName::Ashgf) {
                algorithms.push(("ASHGF", &mut ashgf));
            }
            if algos_to_run.contains(&AlgoName::AshgfS) {
                algorithms.push(("ASHGF-S", &mut ashgf_s));
            }
            if algos_to_run.contains(&AlgoName::AshgfNg) {
                algorithms.push(("ASHGF-NG", &mut ashgf_ng));
            }
            if algos_to_run.contains(&AlgoName::Asebo) {
                algorithms.push(("ASEBO", &mut asebo));
            }

            if args.quiet {
                tracing::info!(
                    "Benchmark: {} algos × {} dims",
                    algorithms.len(),
                    dims.len()
                );
            }

            let (results, history) = run_benchmarks_with_history(
                &mut algorithms,
                args.pattern.as_deref(),
                &dims,
                args.iter,
                args.seed,
                args.patience,
                args.ftol,
                args.jobs,
                args.quiet,
            );

            // Organise output: one subdirectory per dimension
            let output_dir = Path::new(&args.output);
            fs::create_dir_all(output_dir).unwrap_or_else(|e| {
                tracing::error!("Cannot create output dir '{}': {}", args.output, e);
            });

            for &dim in &dims {
                let dim_dir = output_dir.join(format!("dim_{dim}"));
                fs::create_dir_all(&dim_dir).ok();

                let dim_results: Vec<_> = results.get(&dim).map(|v| v.clone()).unwrap_or_default();
                let dim_history: Vec<_> =
                    history.iter().filter(|r| r.dim == dim).cloned().collect();

                if dim_results.is_empty() {
                    continue;
                }

                // ---- CSV ----
                let csv_path = dim_dir.join("benchmark_results.csv");
                let mut wtr = csv::Writer::from_path(&csv_path).unwrap_or_else(|e| {
                    panic!("Cannot open CSV file '{}': {}", csv_path.display(), e);
                });
                wtr.write_record(&[
                    "algorithm",
                    "function",
                    "dim",
                    "final_value",
                    "best_value",
                    "iterations",
                    "converged",
                ])
                .ok();

                let mut sorted: Vec<_> = dim_results.clone();
                sorted.sort_by(|a, b| {
                    a.function
                        .cmp(&b.function)
                        .then(a.algorithm.cmp(&b.algorithm))
                });
                for r in &sorted {
                    wtr.write_record(&[
                        &r.algorithm,
                        &r.function,
                        &r.dim.to_string(),
                        &format!("{:.6e}", r.final_value),
                        &format!("{:.6e}", r.best_value),
                        &r.iterations.to_string(),
                        &r.converged.to_string(),
                    ])
                    .ok();
                }
                wtr.flush().ok();

                // ---- Plots ----
                let bar_path = dim_dir.join("comparison_bars.png");
                plot_comparison_bars(&dim_results, &bar_path);

                if !dim_history.is_empty() {
                    let grid_path = dim_dir.join("convergence_grid.png");
                    plot_convergence_grid(&dim_history, &grid_path);

                    let per_func_dir = dim_dir.join("per_function");
                    let saved = plot_per_function(&dim_history, &per_func_dir);
                    println!("  dim={:<4}  {} per-function plot(s)", dim, saved.len());
                }

                // ---- Summary ----
                println!(
                    "\ndim={} — {} results saved to {}",
                    dim,
                    sorted.len(),
                    dim_dir.display()
                );
                println!(
                    "{:<8} {:<35} {:>6} {:>14} {:>14} {:>6}",
                    "ALGO", "FUNCTION", "DIM", "FINAL", "BEST", "ITER"
                );
                println!("{}", "─".repeat(90));
                for r in &sorted {
                    println!(
                        "{:<8} {:<35} {:>6} {:>14.6e} {:>14.6e} {:>6}",
                        r.algorithm, r.function, r.dim, r.final_value, r.best_value, r.iterations
                    );
                }
            }

            // Generate REPORT.md
            let all_results: Vec<_> = results.values().flatten().cloned().collect();
            generate_report(&all_results, &dims, args.seed, args.iter, output_dir);

            println!("\nAll results saved under {}/dim_*/", output_dir.display());

            0
        }

        Command::Stats(args) => {
            let f = match get_function(&args.function) {
                Some(f) => f,
                None => {
                    eprintln!("Unknown function: '{}'", args.function);
                    return 1;
                }
            };

            let algos_to_run: Vec<AlgoName> = args.algos.unwrap_or_else(|| {
                vec![
                    AlgoName::Gd,
                    AlgoName::Sges,
                    AlgoName::Asgf,
                    AlgoName::Asgf2s,
                    AlgoName::Asgf2sw,
                    AlgoName::Ashgf,
                    AlgoName::AshgfNg,
                    AlgoName::AshgfS,
                    AlgoName::Asebo,
                ]
            });

            let options = OptimizeOptions {
                max_iter: args.iter,
                ..Default::default()
            };

            println!(
                "Stats: {} runs × {} algos on {}(d={})",
                args.runs,
                algos_to_run.len(),
                args.function,
                args.dim
            );

            for algo_name in &algos_to_run {
                let mut all_best: Vec<f64> = Vec::with_capacity(args.runs);
                let mut all_final: Vec<f64> = Vec::with_capacity(args.runs);

                for run in 0..args.runs {
                    let mut rng = SeededRng::new(args.seed + run as u64);

                    let result = match algo_name {
                        AlgoName::Gd => {
                            let mut algo = GD::new(args.lr, args.sigma, 1e-8);
                            algo.optimize(&f, args.dim, None, &options, &mut rng)
                        }
                        AlgoName::Sges => {
                            let mut algo =
                                SGES::new(args.lr, args.sigma, 0.9, 0.1, 0.5, 1.1, 50, 1e-8);
                            algo.optimize(&f, args.dim, None, &options, &mut rng)
                        }
                        AlgoName::Asgf => {
                            let mut algo = ASGF::default();
                            algo.optimize(&f, args.dim, None, &options, &mut rng)
                        }
                        AlgoName::Asgf2s => {
                            let mut algo = Asgf2s::default();
                            algo.optimize(&f, args.dim, None, &options, &mut rng)
                        }
                        AlgoName::Asgf2sw => {
                            let mut algo = Asgf2sw::default();
                            algo.optimize(&f, args.dim, None, &options, &mut rng)
                        }
                        AlgoName::Ashgf => {
                            let mut algo = ASHGF::default();
                            algo.optimize(&f, args.dim, None, &options, &mut rng)
                        }
                        AlgoName::AshgfS => {
                            let mut algo = ASHGFS::default();
                            algo.optimize(&f, args.dim, None, &options, &mut rng)
                        }
                        AlgoName::AshgfNg => {
                            let mut algo = ASHGFNG::default();
                            algo.optimize(&f, args.dim, None, &options, &mut rng)
                        }
                        AlgoName::Asebo => {
                            let mut algo = ASEBO::default();
                            algo.optimize(&f, args.dim, None, &options, &mut rng)
                        }
                    };

                    let best = result
                        .best_values
                        .last()
                        .map(|(_, v)| *v)
                        .unwrap_or(f64::NAN);
                    let final_val = result.all_values.last().copied().unwrap_or(f64::NAN);
                    all_best.push(best);
                    all_final.push(final_val);
                }

                let mean_best = all_best.iter().sum::<f64>() / args.runs as f64;
                let std_best = (all_best
                    .iter()
                    .map(|v| (v - mean_best).powi(2))
                    .sum::<f64>()
                    / args.runs as f64)
                    .sqrt();
                let min_best = all_best.iter().fold(f64::INFINITY, |a, &b| a.min(b));

                let mean_final = all_final.iter().sum::<f64>() / args.runs as f64;
                let std_final = (all_final
                    .iter()
                    .map(|v| (v - mean_final).powi(2))
                    .sum::<f64>()
                    / args.runs as f64)
                    .sqrt();

                println!(
                    "{:>6}: best={:.6e} ± {:.6e}  min={:.6e}  final={:.6e} ± {:.6e}",
                    format!("{:?}", algo_name).to_lowercase(),
                    mean_best,
                    std_best,
                    min_best,
                    mean_final,
                    std_final,
                );
            }
            0
        }
    }
}
