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

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::grid::Grid2D;
use crate::linalg::cholesky_solve_in_place;
use crate::rng::{Rng, splitmix64};
use crate::search::BucketGrid;
use crate::transform::{NormalScore, Tails};
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
    /// GSLIB-style tail extrapolation for the back-transform. The default
    /// clamps realizations to the data range; set tail models and bounds to
    /// let extremes exceed the observed extremes.
    pub tails: Tails,
    /// Optional declustering weights (one positive weight per data point,
    /// e.g. from [`crate::declustering::cell_declustering_weights`]): the
    /// normal-score reference distribution is fitted with them so
    /// preferential sampling does not bias the simulated histogram.
    pub decluster_weights: Option<Vec<f64>>,
}

impl Default for SgsConfig {
    fn default() -> Self {
        Self {
            n_realizations: 1,
            seed: 42,
            max_neighbors: 16,
            search_radius: None,
            tails: Tails::default(),
            decluster_weights: None,
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

/// Runs conditional sequential Gaussian simulation on a 2-D grid.
///
/// `model_ns` must be a variogram model fitted to the *normal scores* of the
/// data (its total sill should therefore be close to 1).
pub fn sequential_gaussian_simulation(
    data: &PointSet,
    model_ns: &VariogramModel,
    grid: &Grid2D,
    cfg: &SgsConfig,
) -> Result<SgsResult> {
    let realizations = sgs_at(data, model_ns, &grid.centers(), cfg)?;
    Ok(SgsResult {
        grid: grid.clone(),
        realizations,
    })
}

/// Runs conditional sequential Gaussian simulation on a 3-D grid, returning
/// the realizations in grid storage order.
pub fn sequential_gaussian_simulation_3d(
    data: &PointSet<3>,
    model_ns: &VariogramModel,
    grid: &crate::grid::Grid3D,
    cfg: &SgsConfig,
) -> Result<Vec<Vec<f64>>> {
    sgs_at(data, model_ns, &grid.centers(), cfg)
}

/// SGS at an arbitrary set of simulation nodes (sequential path over the
/// node list).
pub fn sgs_at<const D: usize>(
    data: &PointSet<D>,
    model_ns: &VariogramModel,
    nodes: &[[f64; D]],
    cfg: &SgsConfig,
) -> Result<Vec<Vec<f64>>> {
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

    if nodes.is_empty() {
        return Err(GeostatError::InvalidParameter(
            "no simulation nodes given".into(),
        ));
    }
    let ns = match &cfg.decluster_weights {
        Some(w) => {
            if w.len() != data.len() {
                return Err(GeostatError::DimensionMismatch(format!(
                    "{} declustering weights vs {} data points",
                    w.len(),
                    data.len()
                )));
            }
            NormalScore::fit_weighted_with_tails(data.values(), w, cfg.tails)?
        }
        None => NormalScore::fit_with_tails(data.values(), cfg.tails)?,
    };
    let data_scores: Vec<f64> = data.values().iter().map(|&v| ns.transform(v)).collect();

    crate::parallel::par_try_map(cfg.n_realizations, |r| {
        let mut seed_state = cfg.seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let seed_r = splitmix64(&mut seed_state);
        simulate_one(data, &data_scores, &ns, model_ns, nodes, cfg, seed_r)
    })
}

fn simulate_one<const D: usize>(
    data: &PointSet<D>,
    data_scores: &[f64],
    ns: &NormalScore,
    model: &VariogramModel,
    centers: &[[f64; D]],
    cfg: &SgsConfig,
    seed: u64,
) -> Result<Vec<f64>> {
    let mut rng = Rng::new(seed);
    let n_cells = centers.len();
    let c0 = model.covariance_dh([0.0; D]);
    // Tiny diagonal stabilizer: previously simulated nodes can sit arbitrarily
    // close to data points, which makes exact systems near-singular.
    let stabilizer = c0 * 1e-9;

    let mut path: Vec<usize> = (0..n_cells).collect();
    rng.shuffle(&mut path);

    // Conditioning store: bucket grid covering data and simulation extents.
    let (dmin, dmax) = data.bbox();
    let mut min = dmin;
    let mut max = dmax;
    for c in centers {
        for d in 0..D {
            min[d] = min[d].min(c[d]);
            max[d] = max[d].max(c[d]);
        }
    }
    let mut search = BucketGrid::new(min, max, data.len() + n_cells);
    let mut cond_coords: Vec<[f64; D]> = data.coords().to_vec();
    let mut cond_vals: Vec<f64> = data_scores.to_vec();
    for &p in data.coords() {
        search.insert(p);
    }
    let mut sim_ns = vec![0.0_f64; n_cells];
    // Workspaces reused across the whole realization: this is the engine's
    // hottest loop, and per-node allocation dominated it.
    let mut ws = SkWorkspace::default();

    for &cell in &path {
        let target = centers[cell];
        let nb = search.k_nearest(target, cfg.max_neighbors, cfg.search_radius);

        let (mean, var) = if nb.is_empty() {
            (0.0, c0)
        } else {
            simple_kriging_ns(
                &cond_coords,
                &cond_vals,
                model,
                target,
                &nb,
                c0,
                stabilizer,
                &mut ws,
            )?
        };

        let z = mean + var.max(0.0).sqrt() * rng.normal();
        sim_ns[cell] = z;
        search.insert(target);
        cond_coords.push(target);
        cond_vals.push(z);
    }

    Ok(sim_ns.iter().map(|&s| ns.back_transform(s)).collect())
}

/// Reusable buffers for the per-node simple-kriging systems.
#[derive(Default)]
struct SkWorkspace {
    a: Vec<f64>,
    b: Vec<f64>,
    w: Vec<f64>,
}

/// Simple kriging with mean 0 in Gaussian space; returns (mean, variance).
/// The covariance system is SPD (stabilized diagonal), so it is solved by
/// Cholesky in the caller-provided workspace — no allocation per node.
#[allow(clippy::too_many_arguments)]
fn simple_kriging_ns<const D: usize>(
    coords: &[[f64; D]],
    vals: &[f64],
    model: &VariogramModel,
    target: [f64; D],
    nb: &[usize],
    c0: f64,
    stabilizer: f64,
    ws: &mut SkWorkspace,
) -> Result<(f64, f64)> {
    let n = nb.len();
    ws.a.clear();
    ws.a.resize(n * n, 0.0);
    ws.b.clear();
    ws.b.resize(n, 0.0);
    let sep = |a: [f64; D], b: [f64; D]| {
        let mut dh = [0.0; D];
        for d in 0..D {
            dh[d] = a[d] - b[d];
        }
        dh
    };
    for (ii, &i) in nb.iter().enumerate() {
        let pi = coords[i];
        ws.a[ii * n + ii] = c0 + stabilizer;
        for (jj, &j) in nb.iter().enumerate().skip(ii + 1) {
            let c = model.covariance_dh(sep(pi, coords[j]));
            ws.a[ii * n + jj] = c;
            ws.a[jj * n + ii] = c;
        }
        ws.b[ii] = model.covariance_dh(sep(pi, target));
    }
    ws.w.clear();
    ws.w.extend_from_slice(&ws.b);
    cholesky_solve_in_place(&mut ws.a, n, &mut ws.w)?;
    let mut mean = 0.0;
    let mut reduction = 0.0;
    for ii in 0..n {
        mean += ws.w[ii] * vals[nb[ii]];
        reduction += ws.w[ii] * ws.b[ii];
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
    fn tails_let_realizations_exceed_the_data_range() {
        let (data, model, grid) = setup();
        let (lo, hi) = data
            .values()
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        let cfg = SgsConfig {
            n_realizations: 20,
            seed: 7,
            max_neighbors: 12,
            search_radius: None,
            tails: crate::transform::Tails {
                lower: crate::tails::TailModel::Linear,
                upper: crate::tails::TailModel::Linear,
                lower_bound: Some(lo - 2.0),
                upper_bound: Some(hi + 2.0),
            },
            ..Default::default()
        };
        let res = sequential_gaussian_simulation(&data, &model, &grid, &cfg).unwrap();
        let mut exceeds = false;
        for r in &res.realizations {
            for &v in r {
                assert!(v >= lo - 2.0 - 1e-9 && v <= hi + 2.0 + 1e-9);
                if v > hi || v < lo {
                    exceeds = true;
                }
            }
        }
        assert!(
            exceeds,
            "with tails enabled some values should leave the data range"
        );
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
