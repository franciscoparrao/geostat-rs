//! Sequential Gaussian simulation (SGS).
//!
//! Pipeline per realization:
//! 1. Normal-score transform of the conditioning data.
//! 2. Random path over the grid nodes (deterministic, seeded).
//! 3. At each node, simple kriging (mean 0) in Gaussian space from the
//!    nearest conditioning points (data + previously simulated nodes).
//! 4. Draw from the conditional Gaussian and add the node to the
//!    conditioning set.
//! 5. Back-transform the realization to data units.
//!
//! Realizations run in parallel; each derives its own RNG stream from the
//! base seed, so results are reproducible regardless of thread scheduling.

use ndarray::Array2;

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::grid::Grid2D;
use crate::linalg::solve;
use crate::rng::{Rng, splitmix64};
use crate::search::BucketGrid;
use crate::transform::NormalScore;
use crate::variogram::VariogramModel;

/// Configuration for sequential Gaussian simulation.
#[derive(Debug, Clone)]
pub struct SgsConfig {
    /// Number of realizations to generate.
    pub n_realizations: usize,
    /// Base seed; realization `r` uses a stream derived from `(seed, r)`.
    pub seed: u64,
    /// Maximum number of conditioning points per node.
    pub max_neighbors: usize,
    /// Optional search radius for conditioning points.
    pub search_radius: Option<f64>,
}

impl Default for SgsConfig {
    fn default() -> Self {
        Self {
            n_realizations: 1,
            seed: 42,
            max_neighbors: 16,
            search_radius: None,
        }
    }
}

/// Result of an SGS run: realizations in grid storage order, in data units.
#[derive(Debug, Clone)]
pub struct SgsResult {
    /// The simulation grid.
    pub grid: Grid2D,
    /// One vector of `grid.n_cells()` values per realization.
    pub realizations: Vec<Vec<f64>>,
}

/// Runs conditional sequential Gaussian simulation on a grid.
///
/// `model_ns` must be a variogram model fitted to the *normal scores* of the
/// data (its total sill should therefore be close to 1).
pub fn sequential_gaussian_simulation(
    data: &PointSet,
    model_ns: &VariogramModel,
    grid: &Grid2D,
    cfg: &SgsConfig,
) -> Result<SgsResult> {
    if cfg.n_realizations == 0 {
        return Err(GeostatError::InvalidParameter(
            "n_realizations must be at least 1".into(),
        ));
    }
    if cfg.max_neighbors == 0 {
        return Err(GeostatError::InvalidParameter(
            "max_neighbors must be at least 1".into(),
        ));
    }
    if let Some(r) = cfg.search_radius
        && !(r > 0.0)
    {
        return Err(GeostatError::InvalidParameter(format!(
            "search radius must be positive, got {r}"
        )));
    }

    let ns = NormalScore::fit(data.values())?;
    let data_scores: Vec<f64> = data.values().iter().map(|&v| ns.transform(v)).collect();

    let realizations: Vec<Vec<f64>> = crate::parallel::par_try_map(cfg.n_realizations, |r| {
        let mut seed_state = cfg.seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let seed_r = splitmix64(&mut seed_state);
        simulate_one(data, &data_scores, &ns, model_ns, grid, cfg, seed_r)
    })?;

    Ok(SgsResult {
        grid: grid.clone(),
        realizations,
    })
}

fn simulate_one(
    data: &PointSet,
    data_scores: &[f64],
    ns: &NormalScore,
    model: &VariogramModel,
    grid: &Grid2D,
    cfg: &SgsConfig,
    seed: u64,
) -> Result<Vec<f64>> {
    let mut rng = Rng::new(seed);
    let centers = grid.centers();
    let n_cells = centers.len();
    let c0 = model.covariance_dh([0.0, 0.0]);
    // Tiny diagonal stabilizer: previously simulated nodes can sit arbitrarily
    // close to data points, which makes exact systems near-singular.
    let stabilizer = c0 * 1e-9;

    let mut path: Vec<usize> = (0..n_cells).collect();
    rng.shuffle(&mut path);

    // Conditioning store: bucket grid covering data and simulation extents.
    let (dmin, dmax) = data.bbox();
    let min = [dmin[0].min(grid.x0), dmin[1].min(grid.y0)];
    let max = [
        dmax[0].max(grid.x0 + grid.dx * grid.nx as f64),
        dmax[1].max(grid.y0 + grid.dy * grid.ny as f64),
    ];
    let mut search = BucketGrid::new(min, max, data.len() + n_cells);
    let mut cond_coords: Vec<[f64; 2]> = data.coords().to_vec();
    let mut cond_vals: Vec<f64> = data_scores.to_vec();
    for &p in data.coords() {
        search.insert(p);
    }
    let mut sim_ns = vec![0.0_f64; n_cells];

    for &cell in &path {
        let target = centers[cell];
        let nb = search.k_nearest(target, cfg.max_neighbors, cfg.search_radius);

        let (mean, var) = if nb.is_empty() {
            (0.0, c0)
        } else {
            simple_kriging_ns(&cond_coords, &cond_vals, model, target, &nb, c0, stabilizer)?
        };

        let z = mean + var.max(0.0).sqrt() * rng.normal();
        sim_ns[cell] = z;
        search.insert(target);
        cond_coords.push(target);
        cond_vals.push(z);
    }

    Ok(sim_ns.iter().map(|&s| ns.back_transform(s)).collect())
}

/// Simple kriging with mean 0 in Gaussian space; returns (mean, variance).
fn simple_kriging_ns(
    coords: &[[f64; 2]],
    vals: &[f64],
    model: &VariogramModel,
    target: [f64; 2],
    nb: &[usize],
    c0: f64,
    stabilizer: f64,
) -> Result<(f64, f64)> {
    let n = nb.len();
    let mut a = Array2::<f64>::zeros((n, n));
    let mut b = vec![0.0; n];
    for (ii, &i) in nb.iter().enumerate() {
        let pi = coords[i];
        a[[ii, ii]] = c0 + stabilizer;
        for (jj, &j) in nb.iter().enumerate().skip(ii + 1) {
            let pj = coords[j];
            let c = model.covariance_dh([pi[0] - pj[0], pi[1] - pj[1]]);
            a[[ii, jj]] = c;
            a[[jj, ii]] = c;
        }
        b[ii] = model.covariance_dh([pi[0] - target[0], pi[1] - target[1]]);
    }
    let b0 = b.clone();
    let w = solve(a, b)?;
    let mut mean = 0.0;
    let mut reduction = 0.0;
    for ii in 0..n {
        mean += w[ii] * vals[nb[ii]];
        reduction += w[ii] * b0[ii];
    }
    Ok((mean, (c0 - reduction).max(0.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::{ModelKind, Structure};

    fn setup() -> (PointSet, VariogramModel, Grid2D) {
        let data = PointSet::new(
            vec![
                [10.0, 10.0],
                [90.0, 10.0],
                [10.0, 90.0],
                [90.0, 90.0],
                [50.0, 50.0],
                [30.0, 70.0],
                [70.0, 30.0],
            ],
            vec![1.0, 5.0, 2.0, 8.0, 4.0, 3.0, 6.0],
        )
        .unwrap();
        // Model for normal scores: sill ~ 1.
        let model = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 0.95, 30.0)],
        )
        .unwrap();
        let grid = Grid2D::from_bbox([0.0, 0.0], [100.0, 100.0], 10, 10).unwrap();
        (data, model, grid)
    }

    #[test]
    fn reproducible_with_same_seed() {
        let (data, model, grid) = setup();
        let cfg = SgsConfig {
            n_realizations: 3,
            seed: 123,
            ..Default::default()
        };
        let a = sequential_gaussian_simulation(&data, &model, &grid, &cfg).unwrap();
        let b = sequential_gaussian_simulation(&data, &model, &grid, &cfg).unwrap();
        assert_eq!(a.realizations, b.realizations);
        // Realizations differ from each other.
        assert_ne!(a.realizations[0], a.realizations[1]);
        // Different seed, different result.
        let cfg2 = SgsConfig { seed: 124, ..cfg };
        let c = sequential_gaussian_simulation(&data, &model, &grid, &cfg2).unwrap();
        assert_ne!(a.realizations[0], c.realizations[0]);
    }

    #[test]
    fn values_within_data_range() {
        // Back-transform clamps to the data range in this MVP.
        let (data, model, grid) = setup();
        let cfg = SgsConfig {
            n_realizations: 2,
            seed: 7,
            ..Default::default()
        };
        let res = sequential_gaussian_simulation(&data, &model, &grid, &cfg).unwrap();
        for real in &res.realizations {
            assert_eq!(real.len(), grid.n_cells());
            for &v in real {
                assert!((1.0..=8.0).contains(&v), "value {v} outside data range");
            }
        }
    }

    #[test]
    fn rejects_bad_config() {
        let (data, model, grid) = setup();
        let cfg = SgsConfig {
            n_realizations: 0,
            ..Default::default()
        };
        assert!(sequential_gaussian_simulation(&data, &model, &grid, &cfg).is_err());
        let cfg = SgsConfig {
            max_neighbors: 0,
            ..Default::default()
        };
        assert!(sequential_gaussian_simulation(&data, &model, &grid, &cfg).is_err());
    }
}
