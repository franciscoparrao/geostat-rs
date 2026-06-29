//! Demonstrates the Vecchia log-likelihood's `O(n m^3)` scaling against the
//! exact `O(n^3)` Gaussian log-likelihood.
//!
//!     cargo run --release --example vecchia_scaling -p geostat-core
//!
//! The exact likelihood needs the full n x n covariance factorization, so it
//! becomes impractical well before the Vecchia approximation does; the latter
//! stays cheap and tracks the exact value to a small relative error.

use std::time::Instant;

use geostat_core::linalg::lu_factor;
use geostat_core::rng::Rng;
use geostat_core::variogram::{ModelKind, Structure, VariogramModel};
use geostat_core::vecchia::vecchia_loglik;
use geostat_core::PointSet;

fn exact_loglik(data: &PointSet, model: &VariogramModel) -> f64 {
    let n = data.len();
    let c = data.coords();
    let mean = data.mean();
    let z: Vec<f64> = data.values().iter().map(|v| v - mean).collect();
    let mut cov = vec![0.0; n * n];
    for a in 0..n {
        for b in 0..n {
            let h = [c[a][0] - c[b][0], c[a][1] - c[b][1]];
            cov[a * n + b] = model.covariance_dh(h);
        }
    }
    let arr = ndarray::Array2::from_shape_vec((n, n), cov).unwrap();
    let lu = lu_factor(arr).unwrap();
    let x = lu.solve(z.clone());
    let quad: f64 = z.iter().zip(&x).map(|(&zi, &xi)| zi * xi).sum();
    -0.5 * (n as f64 * (2.0 * std::f64::consts::PI).ln() + lu.ln_det_abs() + quad)
}

fn field(n: usize) -> PointSet {
    let mut rng = Rng::new(42);
    let mut coords = Vec::new();
    let mut values = Vec::new();
    for _ in 0..n {
        let x = rng.uniform() * 1000.0;
        let y = rng.uniform() * 1000.0;
        coords.push([x, y]);
        values.push((x / 200.0).sin() + (y / 250.0).cos() + 0.1 * rng.normal());
    }
    PointSet::new(coords, values).unwrap()
}

fn main() {
    let model =
        VariogramModel::new(0.05, vec![Structure::new(ModelKind::Exponential, 1.0, 200.0)]).unwrap();
    let m = 20;
    println!("Vecchia (m={m}) vs exact Gaussian log-likelihood\n");
    println!("{:>7}  {:>12}  {:>12}  {:>10}", "n", "vecchia (ms)", "exact (ms)", "rel.err");
    for &n in &[500usize, 1000, 2000, 4000, 8000] {
        let data = field(n);

        let t = Instant::now();
        let v = vecchia_loglik(&data, &model, m, None).unwrap();
        let vt = t.elapsed().as_secs_f64() * 1e3;

        // The exact likelihood is only attempted while it stays practical.
        if n <= 2000 {
            let t = Instant::now();
            let e = exact_loglik(&data, &model);
            let et = t.elapsed().as_secs_f64() * 1e3;
            let rel = (v - e).abs() / e.abs();
            println!("{n:>7}  {vt:>12.1}  {et:>12.1}  {rel:>10.2e}");
        } else {
            println!("{n:>7}  {vt:>12.1}  {:>12}  {:>10}", "(skipped)", "-");
        }
    }
}
