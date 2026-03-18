use ashgf::optimizers::asgf::ASGF;

fn main() {
    let asgf = ASGF::new();
    let dim = 10;
    let mut rng = rand::rngs::StdRng::seed_from_u64(2003);
    let basis = asgf.generate_random_orthogonal(dim, &mut rng);
    
    println!("Testing orthonormality of basis matrix:");
    
    // Check if rows are unit vectors
    for i in 0..dim {
        let norm: f64 = (0..dim).map(|j| basis[(i, j)].powi(2)).sum::<f64>().sqrt();
        println!("Row {} norm: {}", i, norm);
    }
    
    // Check if rows are orthogonal
    for i in 0..dim {
        for j in (i+1)..dim {
            let dot: f64 = (0..dim).map(|k| basis[(i, k)] * basis[(j, k)]).sum();
            println!("Dot product of rows {} and {}: {}", i, j, dot);
        }
    }
}
