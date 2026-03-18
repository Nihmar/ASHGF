use ashgf::functions::sphere;
use ashgf::{Optimizer, ASHGF, GD};

fn main() {
    println!("ASHGF - Rust Optimization Library");
    println!("===================================\n");

    let dim = 10;
    let iters = 100;

    // Test with sphere function
    let mut gd = GD::new();
    let result = gd.optimize(sphere, dim, iters, None, true, 25).unwrap();

    println!("GD Results:");
    println!("  Best value: {:.6e}", result.best_value());
    println!("  Iterations: {}", result.all_values.len());

    // Test with ASHGF
    let mut ashgf = ASHGF::new();
    let result = ashgf.optimize(sphere, dim, iters, None, true, 25).unwrap();

    println!("\nASHGF Results:");
    println!("  Best value: {:.6e}", result.best_value());
    println!("  Iterations: {}", result.all_values.len());
}
