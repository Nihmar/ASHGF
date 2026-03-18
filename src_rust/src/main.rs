use ashgf::functions::sphere;
use ashgf::optimizers::{ASEBO, ASGF, SGES};
use ashgf::{Optimizer, ASHGF, GD};

fn main() {
    println!("ASHGF - Rust Optimization Library");
    println!("===================================\n");

    let dim = 10;
    let iters = 100;

    // GD
    {
        let mut optimizer = GD::new();
        let result = optimizer
            .optimize(sphere, dim, iters, None, true, 25)
            .unwrap();
        println!("GD Results:");
        println!("  Best value: {:.6e}", result.best_value());
        println!("  Iterations: {}\n", result.all_values.len());
    }

    // SGES
    {
        let mut optimizer = SGES::new();
        let result = optimizer
            .optimize(sphere, dim, iters, None, true, 25)
            .unwrap();
        println!("SGES Results:");
        println!("  Best value: {:.6e}", result.best_value());
        println!("  Iterations: {}\n", result.all_values.len());
    }

    // ASGF
    {
        let mut optimizer = ASGF::new();
        let result = optimizer
            .optimize(sphere, dim, iters, None, true, 25)
            .unwrap();
        println!("ASGF Results:");
        println!("  Best value: {:.6e}", result.best_value());
        println!("  Iterations: {}\n", result.all_values.len());
    }

    // ASHGF
    {
        let mut optimizer = ASHGF::new();
        let result = optimizer
            .optimize(sphere, dim, iters, None, true, 25)
            .unwrap();
        println!("ASHGF Results:");
        println!("  Best value: {:.6e}", result.best_value());
        println!("  Iterations: {}\n", result.all_values.len());
    }

    // ASEBO
    {
        let mut optimizer = ASEBO::new();
        let result = optimizer
            .optimize(sphere, dim, iters, None, true, 25)
            .unwrap();
        println!("ASEBO Results:");
        println!("  Best value: {:.6e}", result.best_value());
        println!("  Iterations: {}\n", result.all_values.len());
    }
}
