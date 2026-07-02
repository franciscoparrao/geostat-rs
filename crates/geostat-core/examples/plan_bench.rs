//! Quick scaling check of vecchia_plan after the O(n log n) rewrite.
use geostat_core::rng::Rng;
use geostat_core::vecchia::vecchia_plan;
use std::time::Instant;

fn main() {
    for &n in &[10_000usize, 50_000, 100_000, 200_000] {
        let mut rng = Rng::new(42);
        let coords: Vec<[f64; 2]> = (0..n)
            .map(|_| [rng.uniform() * 1000.0, rng.uniform() * 1000.0])
            .collect();
        let t = Instant::now();
        let plan = vecchia_plan(&coords, 20, None).unwrap();
        println!(
            "n = {n:>7}: plan built in {:>8.2?} (order len {})",
            t.elapsed(),
            plan.order.len()
        );
    }
}
