use ashgf::functions::{ackley, rastrigin, sphere};
use ashgf::{Optimizer, ASHGF, GD};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("functions");

    for dim in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("sphere", dim), dim, |b, &dim| {
            let x: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.1).collect();
            b.iter(|| sphere(&x));
        });

        group.bench_with_input(BenchmarkId::new("rastrigin", dim), dim, |b, &dim| {
            let x: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.1).collect();
            b.iter(|| rastrigin(&x));
        });

        group.bench_with_input(BenchmarkId::new("ackley", dim), dim, |b, &dim| {
            let x: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.1).collect();
            b.iter(|| ackley(&x));
        });
    }

    group.finish();
}

fn bench_optimizers(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizers");

    for dim in [10, 50].iter() {
        group.bench_with_input(BenchmarkId::new("GD", dim), dim, |b, &dim| {
            let mut gd = GD::new();
            b.iter(|| gd.optimize(sphere, dim, 50, None, false, 25));
        });

        group.bench_with_input(BenchmarkId::new("ASHGF", dim), dim, |b, &dim| {
            let mut ashgf = ASHGF::new();
            b.iter(|| ashgf.optimize(sphere, dim, 50, None, false, 25));
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_sphere, bench_optimizers
}
criterion_main!(benches);
