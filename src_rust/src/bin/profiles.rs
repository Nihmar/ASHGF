use anyhow::Result;
use clap::Parser;

use ashgf::functions::Function;
use ashgf::profiles::{analyze_results, get_default_functions, Algorithm, Runner};

#[derive(Parser, Debug)]
#[command(name = "ashgf-profiles")]
#[command(about = "Run performance profiles for optimization algorithms")]
struct Args {
    #[arg(long, default_value_t = 100)]
    dim: usize,

    #[arg(long, default_value_t = 10000)]
    iters: usize,

    #[arg(long, default_value_t = 10)]
    n_runs: usize,

    #[arg(long, default_value_t = 0)]
    seed: u64,

    #[arg(long)]
    overwrite: bool,

    #[arg(long)]
    functions: Vec<String>,

    #[arg(long, num_args = 1..)]
    algorithms: Option<Vec<String>>,

    #[arg(long)]
    analyze: bool,

    #[arg(long)]
    batch_size: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.analyze {
        analyze_results(args.dim as u32)?;
        return Ok(());
    }

    let algos: Vec<Algorithm> = if let Some(algs) = args.algorithms {
        algs.iter().filter_map(|a| Algorithm::from_str(a)).collect()
    } else {
        vec![
            Algorithm::GD,
            Algorithm::SGES,
            Algorithm::ASGF,
            Algorithm::ASHGF,
            Algorithm::ASEBO,
        ]
    };

    let funcs: Vec<Function> = if args.functions.is_empty() {
        get_default_functions()
    } else {
        args.functions
            .iter()
            .filter_map(|name| Function::from_name(name))
            .collect()
    };

    if funcs.is_empty() {
        anyhow::bail!("No valid functions specified");
    }

    if algos.is_empty() {
        anyhow::bail!("No valid algorithms specified");
    }

    println!(
        "Functions: {:?}",
        funcs.iter().map(|f| f.name()).collect::<Vec<_>>()
    );
    println!(
        "Algorithms: {:?}",
        algos.iter().map(|a| a.name()).collect::<Vec<_>>()
    );
    println!(
        "Dim: {}, Iters: {}, Runs: {}, Seed: {}",
        args.dim, args.iters, args.n_runs, args.seed
    );

    let mut runner = Runner::new(args.dim, args.iters, args.n_runs, args.seed);
    if let Some(batch) = args.batch_size {
        runner = runner.with_batch_size(batch);
    }

    runner.run(&funcs, &algos, args.overwrite, true)?;

    println!(
        "\nResults saved to results/profiles/dim={}/results.parquet",
        args.dim
    );

    Ok(())
}
