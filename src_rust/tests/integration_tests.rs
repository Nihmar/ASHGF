use ashgf::functions::{
    ackley, extended_rosenbrock, generalized_rosenbrock, griewank, levy, power, rastrigin,
    schwefel, sphere,
};

#[test]
fn test_sphere_at_origin() {
    let x = vec![0.0; 10];
    let result = sphere(&x);
    approx::assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
}

#[test]
fn test_sphere_values() {
    let x = vec![1.0, 2.0, 3.0];
    let result = sphere(&x);
    approx::assert_abs_diff_eq!(result, 14.0, epsilon = 1e-10);
}

#[test]
fn test_rastrigin_at_origin() {
    let x = vec![0.0; 10];
    let result = rastrigin(&x);
    approx::assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
}

#[test]
fn test_ackley_at_origin() {
    let x = vec![0.0; 10];
    let result = ackley(&x);
    approx::assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
}

#[test]
fn test_rosenbrock_at_minimum() {
    let x = vec![1.0; 10];
    let result = generalized_rosenbrock(&x);
    approx::assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
}

#[test]
fn test_extended_rosenbrock() {
    let x = vec![1.0; 20];
    let result = extended_rosenbrock(&x);
    approx::assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
}

#[test]
fn test_power() {
    let x = vec![1.0, 1.0, 1.0];
    let result = power(&x);
    let expected = 1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 1.0;
    approx::assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
}

#[test]
fn test_levy() {
    let x = vec![1.0; 10];
    let result = levy(&x);
    approx::assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
}

#[test]
fn test_schwefel() {
    let x = vec![420.968746; 10];
    let result = schwefel(&x);
    approx::assert_abs_diff_eq!(result, 0.0, epsilon = 1e-3);
}

#[test]
fn test_griewank_at_origin() {
    let x = vec![0.0; 10];
    let result = griewank(&x);
    approx::assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);
}

mod optimizers {
    use super::*;
    use ashgf::{Optimizer, ASHGF, GD};

    #[test]
    fn test_gd_improves_on_sphere() {
        let mut gd = GD::new();
        let result = gd.optimize(sphere, 10, 100, None, false, 25).unwrap();

        let initial = result.all_values[0];
        let final_best = result.best_value();

        assert!(
            final_best < initial,
            "GD should improve from initial value {} to {}",
            initial,
            final_best
        );
    }

    #[test]
    fn test_gd_improves_on_rastrigin() {
        let mut gd = GD::new();
        let x_init = vec![3.0; 10];
        let result = gd
            .optimize(rastrigin, 10, 100, Some(&x_init), false, 25)
            .unwrap();

        let initial = result.all_values[0];
        let final_best = result.best_value();

        assert!(final_best < initial, "GD should improve on rastrigin");
    }

    #[test]
    fn test_ashgf_improves_on_sphere() {
        let mut ashgf = ASHGF::new();
        let result = ashgf.optimize(sphere, 10, 100, None, false, 25).unwrap();

        let initial = result.all_values[0];
        let final_best = result.best_value();

        assert!(
            final_best < initial,
            "ASHGF should improve from initial value"
        );
    }

    #[test]
    fn test_ashgf_improves_on_rastrigin() {
        let mut ashgf = ASHGF::new();
        let x_init = vec![3.0; 10];
        let result = ashgf
            .optimize(rastrigin, 10, 100, Some(&x_init), false, 25)
            .unwrap();

        let initial = result.all_values[0];
        let final_best = result.best_value();

        assert!(final_best < initial, "ASHGF should improve on rastrigin");
    }

    #[test]
    fn test_ashgf_respects_epsilon_termination() {
        let mut ashgf = ASHGF::new();
        ashgf.eps = 1e-15;

        let result = ashgf.optimize(sphere, 5, 10000, None, false, 1000).unwrap();

        assert!(
            result.all_values.len() < 10000,
            "ASHGF should terminate early due to epsilon"
        );
    }
}
