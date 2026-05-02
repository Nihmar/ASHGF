//! CLI entry point for ASHGF.

use std::process;

use ashgf::algorithms::{OptimizeOptions, Optimizer, ASGF, ASHGF, GD, SGES};
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
                AlgoName::Ashgf => {
                    let mut algo = ASHGF::default();
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
                    AlgoName::Ashgf => {
                        let mut algo = ASHGF::default();
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

        Command::Benchmark(_args) => {
            tracing::info!("Benchmark mode — results are saved to CSV/JSON.");
            tracing::warn!("Full benchmark runner not yet implemented in Rust.");
            // TODO: full benchmark runner with CSV output
            0
        }

        Command::Stats(_args) => {
            tracing::info!("Stats mode — multiple trials with convergence stats.");
            tracing::warn!("Stats runner not yet implemented in Rust.");
            // TODO: multi-trial statistics
            0
        }
    }
}
