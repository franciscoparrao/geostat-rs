//! Measures the Guinness (2018) grouping speedup on the Vecchia
//! log-likelihood and MLE fit: blocks of `group_size` consecutive points in
//! the max-min ordering share one Cholesky factorization instead of paying
//! O(m^3) per point.
//!
//!     cargo run --release --example vecchia_grouping_bench -p geostat-core

use std::time::Instant;

use geostat_core::PointSet;
use geostat_core::rng::Rng;
use geostat_core::variogram::{ModelKind, Structure, VariogramModel};
use geostat_core::vecchia::{vecchia_loglik, vecchia_loglik_grouped, vecchia_mle, vecchia_mle_grouped};

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
    let model = VariogramModel::new(
        0.05,
        vec![Structure::new(ModelKind::Exponential, 1.0, 200.0)],
    )
    .unwrap();
    let m = 30;
    let n = 4000;
    let data = field(n);
    let reps = 5;

    println!("Vecchia log-likelihood, n={n}, m={m}, {reps} evaluations\n");
    let t = Instant::now();
    let mut last = 0.0;
    for _ in 0..reps {
        last = vecchia_loglik(&data, &model, m, None).unwrap();
    }
    let base_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!("group_size=1 (ungrouped): {base_ms:>8.2} ms/eval  (loglik {last:.1})");

    println!(
        "{:>10}  {:>12}  {:>8}  {:>14}",
        "group_size", "ms/eval", "speedup", "loglik"
    );
    for &g in &[2usize, 4, 8, 16] {
        let t = Instant::now();
        let mut ll = 0.0;
        for _ in 0..reps {
            ll = vecchia_loglik_grouped(&data, &model, m, None, g).unwrap();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("{g:>10}  {ms:>12.2}  {:>7.2}x  {ll:>14.1}", base_ms / ms);
    }

    // A larger, denser field: once the max-min traversal has filled in the
    // domain (the common case for realistic n), consecutive selections are
    // more often local refinements with real neighbour overlap.
    let n2 = 20_000;
    let data2 = field(n2);
    println!("\nVecchia log-likelihood, n={n2}, m={m}, 3 evaluations\n");
    let t = Instant::now();
    let mut last2 = 0.0;
    for _ in 0..3 {
        last2 = vecchia_loglik(&data2, &model, m, None).unwrap();
    }
    let base2_ms = t.elapsed().as_secs_f64() * 1e3 / 3.0;
    println!("group_size=1 (ungrouped): {base2_ms:>8.2} ms/eval  (loglik {last2:.1})");
    for &g in &[4usize, 8, 16, 32] {
        let t = Instant::now();
        let mut ll = 0.0;
        for _ in 0..3 {
            ll = vecchia_loglik_grouped(&data2, &model, m, None, g).unwrap();
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / 3.0;
        println!("{g:>10}  {ms:>12.2}  {:>7.2}x  {ll:>14.1}", base2_ms / ms);
    }

    // Small-n MLE fit: illustrative only (Nelder-Mead multi-start calls the
    // likelihood thousands of times, so the per-eval speedup above is what
    // actually matters at scale).
    let n_mle = 600;
    let m_mle = 15;
    let data_mle = field(n_mle);
    println!("\nVecchia MLE fit, n={n_mle}, m={m_mle}\n");
    let t = Instant::now();
    let fit = vecchia_mle(&data_mle, ModelKind::Exponential, m_mle, None).unwrap();
    let base_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("group_size=1 (ungrouped): {base_ms:>8.0} ms  (loglik {:.1})", fit.loglik);

    for &g in &[2usize, 4] {
        let t = Instant::now();
        let fit = vecchia_mle_grouped(&data_mle, ModelKind::Exponential, m_mle, None, g).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "group_size={g:<9}: {ms:>8.0} ms  ({:.2}x)  (loglik {:.1})",
            base_ms / ms,
            fit.loglik
        );
    }
}
