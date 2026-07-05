//! Criterion benchmarks for the core engine: variography, kriging and
//! sequential Gaussian simulation on synthetic data.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use geostat_core::{
    Grid2D, Kriging, KrigingConfig, ModelKind, PointSet, Rng, SgsConfig, Structure,
    VariogramConfig, VariogramModel, experimental_variogram, sequential_gaussian_simulation,
};

fn synthetic(n: usize, seed: u64) -> PointSet {
    let mut rng = Rng::new(seed);
    let mut coords = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.uniform() * 1000.0;
        let y = rng.uniform() * 1000.0;
        coords.push([x, y]);
        values.push((x / 150.0).sin() + (y / 200.0).cos() + 0.1 * rng.normal());
    }
    PointSet::new(coords, values).unwrap()
}

fn model() -> VariogramModel {
    VariogramModel::new(0.02, vec![Structure::new(ModelKind::Spherical, 1.0, 300.0)]).unwrap()
}

fn bench_variogram(c: &mut Criterion) {
    let data = synthetic(2000, 1);
    let cfg = VariogramConfig {
        n_lags: 15,
        max_dist: 500.0,
        direction: None,
    };
    c.bench_function("variogram_2k_points", |b| {
        b.iter(|| experimental_variogram(black_box(&data), black_box(&cfg)).unwrap())
    });
}

fn bench_kriging(c: &mut Criterion) {
    let data = synthetic(1000, 2);
    let m = model();
    let grid = Grid2D::from_bbox([0.0, 0.0], [1000.0, 1000.0], 50, 50).unwrap();
    // `KrigingConfig` is `#[non_exhaustive]`: build from `Default::default()`
    // and assign fields.
    let mut cfg = KrigingConfig::default();
    cfg.max_neighbors = Some(32);
    c.bench_function("ok_2500_cells_1k_data_k32", |b| {
        b.iter(|| {
            let k = Kriging::new(black_box(&data), &m, cfg.clone()).unwrap();
            black_box(k.predict_grid(&grid))
        })
    });
}

fn bench_sgs(c: &mut Criterion) {
    let data = synthetic(300, 3);
    let m = model();
    let grid = Grid2D::from_bbox([0.0, 0.0], [1000.0, 1000.0], 50, 50).unwrap();
    // `SgsConfig` is `#[non_exhaustive]`: build from `Default::default()`
    // and assign fields.
    let mut cfg = SgsConfig::default();
    cfg.n_realizations = 1;
    cfg.seed = 42;
    cfg.max_neighbors = 16;
    cfg.search_radius = None;
    let mut group = c.benchmark_group("sgs");
    group.sample_size(10);
    group.bench_function("sgs_2500_cells_1_realization_k16", |b| {
        b.iter(|| {
            black_box(sequential_gaussian_simulation(black_box(&data), &m, &grid, &cfg).unwrap())
        })
    });
    group.finish();
}

criterion_group!(benches, bench_variogram, bench_kriging, bench_sgs);
criterion_main!(benches);
