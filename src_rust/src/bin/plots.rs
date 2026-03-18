use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use ashgf::plots::io::{get_results_path, ResultsData};
use ashgf::plots::render::{
    generate_summary_csv, plot_all_algorithms, plot_convergence_with_stats,
};

#[derive(Parser, Debug)]
#[command(name = "ashgf-plots")]
#[command(about = "Generate convergence plots from JSON results")]
struct Args {
    #[arg(long, default_value_t = 100)]
    dim: u32,

    #[arg(long)]
    functions: Vec<String>,

    #[arg(long, num_args = 1..)]
    algorithms: Option<Vec<String>>,

    #[arg(long)]
    plot_comparison: bool,

    #[arg(long)]
    summary: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let results_path = get_results_path(args.dim);

    if !results_path.exists() {
        anyhow::bail!(
            "Results not found: {:?}\nRun profiles first to generate results.",
            results_path
        );
    }

    println!("Loading results from {:?}...", results_path);
    let data = ResultsData::from_json(&results_path)?;
    println!("Loaded {} records for dim={}", data.results.len(), args.dim);

    let all_functions = data.get_functions();
    let functions_to_plot = if args.functions.is_empty() {
        println!(
            "No functions specified, using all {} functions from data",
            all_functions.len()
        );
        all_functions
    } else {
        args.functions
            .into_iter()
            .filter(|f| all_functions.contains(f))
            .collect()
    };

    let all_algorithms = data.get_algorithms();
    let default_algorithms = vec![
        "GD".to_string(),
        "SGES".to_string(),
        "ASGF".to_string(),
        "ASHGF".to_string(),
        "ASEBO".to_string(),
    ];
    let algorithms_input = args.algorithms.unwrap_or(default_algorithms);
    let algorithms_to_plot: Vec<String> = algorithms_input
        .into_iter()
        .filter(|a| all_algorithms.contains(a))
        .collect();

    println!("Functions: {:?}", functions_to_plot);
    println!("Algorithms: {:?}", algorithms_to_plot);

    let filtered = data.filter(Some(&functions_to_plot), Some(&algorithms_to_plot));

    if filtered.results.is_empty() {
        println!("No data found matching criteria.");
        return Ok(());
    }

    let binding = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = binding.parent().unwrap_or(std::path::Path::new("."));
    let base_output_dir = project_root
        .join("results")
        .join("plots")
        .join(format!("dim={}", args.dim));

    for function in &functions_to_plot {
        let func_dir = base_output_dir.join(function);
        std::fs::create_dir_all(&func_dir)?;

        println!(
            "\nProcessing: {} ({} records)",
            function,
            filtered
                .results
                .iter()
                .filter(|r| r.function == *function)
                .count()
        );

        for algorithm in &algorithms_to_plot {
            let alg_count = filtered
                .results
                .iter()
                .filter(|r| r.function == *function && r.algorithm == *algorithm)
                .count();

            if alg_count == 0 {
                println!("  No data for {}", algorithm);
                continue;
            }

            let conv_path = func_dir.join(format!("{}_convergence.png", algorithm));
            plot_convergence_with_stats(&filtered, function, algorithm, &conv_path)?;
        }

        if args.plot_comparison {
            plot_all_algorithms(&filtered, function, &func_dir)?;
            println!("  Saved: comparison.png");
        }
    }

    if args.summary {
        let summary_path = base_output_dir.join("summary.csv");
        generate_summary_csv(&filtered, &summary_path)?;
    }

    println!("\nAll plots saved to: {:?}", base_output_dir);
    println!("Done!");
    Ok(())
}
