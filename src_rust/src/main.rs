use ashgf::functions::sphere;
use ashgf::{Optimizer, ASHGF, GD};

fn main() {
    println!("ASHGF - Rust Optimization Library");
    println!("===================================\n");

    let dim = 10;
    let iters = 100;

    // Test with sphere function
    let mut gd = GD::new();
    let (best_values, all_values) = gd.optimize(sphere, dim, iters, None, true, 25);

    println!("GD Results:");
    println!(
        "  Best value: {:.6e}",
        best_values.last().map(|v| v[1]).unwrap_or(0.0)
    );
    println!("  Iterations: {}", all_values.len());

    // Test with ASHGF
    let mut ashgf = ASHGF::new();
    let (best_values, all_values) = ashgf.optimize(sphere, dim, iters, None, true, 25);

    println!("\nASHGF Results:");
    println!(
        "  Best value: {:.6e}",
        best_values.last().map(|v| v[1]).unwrap_or(0.0)
    );
    println!("  Iterations: {}", all_values.len());
}
