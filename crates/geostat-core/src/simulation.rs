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
///
/// `#[non_exhaustive]`: construct via `SgsConfig { n_realizations, seed, ..
/// Default::default() }` (AUDIT-2026-07-v2.md §6 Fase 5).
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// Separate quota for previously simulated nodes (GSLIB `nodmax`).
    /// When set, each neighborhood takes up to `max_neighbors` original
    /// data **plus** up to this many simulated nodes, so dense simulated
    /// nodes cannot crowd the hard data out of the conditioning set. The
    /// default `None` keeps a single shared pool of `max_neighbors`
    /// (data and nodes competing by distance).
    pub max_node_neighbors: Option<usize>,
    /// Multiple-grid simulation levels (GSLIB `nmult`; grid entry points
    /// only): nodes on the coarsest sub-grid (stride `2^multigrid`) are
    /// simulated first, then each refinement, random within a level. This
    /// improves long-range variogram reproduction on dense grids. `0`
    /// (default) keeps a fully random path.
    pub multigrid: u8,
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
            max_node_neighbors: None,
            multigrid: 0,
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
    let levels = (cfg.multigrid > 0).then(|| {
        (0..grid.n_cells())
            .map(|i| grid_level(&[i % grid.nx, i / grid.nx], cfg.multigrid))
            .collect::<Vec<u8>>()
    });
    let realizations = sgs_at_with_levels(data, model_ns, &grid.centers(), levels.as_deref(), cfg)?;
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
    let levels = (cfg.multigrid > 0).then(|| {
        (0..grid.n_cells())
            .map(|i| {
                let ix = i % grid.nx;
                let iy = (i / grid.nx) % grid.ny;
                let iz = i / (grid.nx * grid.ny);
                grid_level(&[ix, iy, iz], cfg.multigrid)
            })
            .collect::<Vec<u8>>()
    });
    sgs_at_with_levels(data, model_ns, &grid.centers(), levels.as_deref(), cfg)
}

/// Multigrid level of a grid cell: the largest `g <= max_level` such that
/// every index is a multiple of `2^g` (coarser sub-grids get higher levels).
/// `pub(crate)`: shared with [`crate::sis`]'s multigrid path.
pub(crate) fn grid_level(idx: &[usize], max_level: u8) -> u8 {
    (0..=max_level)
        .rev()
        .find(|&g| idx.iter().all(|&i| i % (1usize << g) == 0))
        .unwrap_or(0)
}

/// SGS at an arbitrary set of simulation nodes (sequential path over the
/// node list). `multigrid` is ignored here (it needs grid topology); use
/// the grid entry points for multiple-grid simulation.
pub fn sgs_at<const D: usize>(
    data: &PointSet<D>,
    model_ns: &VariogramModel,
    nodes: &[[f64; D]],
    cfg: &SgsConfig,
) -> Result<Vec<Vec<f64>>> {
    sgs_at_with_levels(data, model_ns, nodes, None, cfg)
}

fn sgs_at_with_levels<const D: usize>(
    data: &PointSet<D>,
    model_ns: &VariogramModel,
    nodes: &[[f64; D]],
    levels: Option<&[u8]>,
    cfg: &SgsConfig,
) -> Result<Vec<Vec<f64>>> {
    if model_ns.has_power() {
        return Err(GeostatError::InvalidParameter(
            "SGS needs a valid covariance function and cannot use the unbounded Power model".into(),
        ));
    }
    // AUDIT-2026-07-v3.md §1.6: the dimensional guard (e.g. `Circular` is not
    // a valid covariance in 3-D) landed in Kriging/SIS/IK/collocated but not
    // here -- SGS with such a model used to simulate silently from a non-PD
    // covariance instead of erroring.
    if let Some(kind) = model_ns.invalid_structure_for_dim(D) {
        return Err(GeostatError::InvalidParameter(format!(
            "{kind:?} is not a valid covariance in {D} dimensions; use Spherical instead for a \
             3-D-safe bounded structure"
        )));
    }
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

    // Extents covering data and nodes, shared by both search structures.
    let (dmin, dmax) = data.bbox();
    let mut min = dmin;
    let mut max = dmax;
    for c in nodes {
        for d in 0..D {
            min[d] = min[d].min(c[d]);
            max[d] = max[d].max(c[d]);
        }
    }
    // Static store of the original data, built once and shared across
    // realizations (only simulated nodes change per realization).
    let mut data_grid = BucketGrid::new(min, max, data.len());
    for &p in data.coords() {
        data_grid.insert(p);
    }

    crate::parallel::par_try_map(cfg.n_realizations, |r| {
        let mut seed_state = cfg.seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let seed_r = splitmix64(&mut seed_state);
        simulate_one(
            data,
            &data_scores,
            &data_grid,
            &ns,
            model_ns,
            nodes,
            levels,
            (min, max),
            cfg,
            seed_r,
        )
    })
}

/// Raw conditional Gaussian-field simulation for one realization: random
/// path (optionally multigrid-ordered via `levels`), moving-neighborhood
/// simple kriging in the caller's value units (already-Gaussian or
/// pseudo-Gaussian — no normal-score transform happens here), sequential
/// conditioning. Returns one value per `centers[i]`, in the same units as
/// `data_scores` (untransformed).
///
/// `pub(crate)`: shared by [`simulate_one`] (SGS, which wraps this with a
/// `NormalScore` fit/back-transform) and [`crate::tgs`] (which wraps this
/// with proportions-derived thresholds instead of a back-transform).
#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_gaussian_path<const D: usize>(
    data_coords: &[[f64; D]],
    data_scores: &[f64],
    data_grid: &BucketGrid<D>,
    model: &VariogramModel,
    centers: &[[f64; D]],
    levels: Option<&[u8]>,
    extents: ([f64; D], [f64; D]),
    max_neighbors: usize,
    search_radius: Option<f64>,
    max_node_neighbors: Option<usize>,
    seed: u64,
) -> Result<Vec<f64>> {
    let mut rng = Rng::new(seed);
    let n_cells = centers.len();
    let n_data = data_coords.len();
    let c0 = model.covariance_dh([0.0; D]);
    // Tiny diagonal stabilizer: previously simulated nodes can sit arbitrarily
    // close to data points, which makes exact systems near-singular.
    let stabilizer = c0 * 1e-9;

    // Random path; with multigrid levels, coarse levels first (stable sort
    // keeps the shuffle within each level).
    let mut path: Vec<usize> = (0..n_cells).collect();
    rng.shuffle(&mut path);
    if let Some(levels) = levels {
        path.sort_by_key(|&i| std::cmp::Reverse(levels[i]));
    }

    // Simulated nodes get their own store; the data store is shared.
    let (min, max) = extents;
    let mut node_grid = BucketGrid::new(min, max, n_cells);
    let mut cond_coords: Vec<[f64; D]> = data_coords.to_vec();
    let mut cond_vals: Vec<f64> = data_scores.to_vec();

    // Quotas: `max_neighbors` original data plus `nodmax` simulated nodes
    // (GSLIB ndmax/nodmax). Without an explicit nodmax the two candidate
    // lists are merged by distance and truncated to `max_neighbors`,
    // reproducing the previous single-pool behavior.
    let nodmax = max_node_neighbors.unwrap_or(max_neighbors);
    let single_pool = max_node_neighbors.is_none();

    let mut sim_ns = vec![0.0_f64; n_cells];
    // Workspaces reused across the whole realization: this is the engine's
    // hottest loop, and per-node allocation dominated it.
    let mut ws = SkWorkspace::default();
    let mut nb: Vec<usize> = Vec::new();

    let d2 = |a: [f64; D], b: [f64; D]| -> f64 {
        let mut s = 0.0;
        for d in 0..D {
            let dd = a[d] - b[d];
            s += dd * dd;
        }
        s
    };

    for &cell in &path {
        let target = centers[cell];
        let nd = data_grid.k_nearest(target, max_neighbors, search_radius);
        let nn = node_grid.k_nearest(target, nodmax, search_radius);

        // Merge the two distance-ascending lists (node indices offset by
        // n_data into the conditioning arrays).
        nb.clear();
        let (mut a, mut b) = (0, 0);
        while a < nd.len() || b < nn.len() {
            let take_data = match (nd.get(a), nn.get(b)) {
                (Some(&i), Some(&j)) => {
                    d2(target, cond_coords[i]) <= d2(target, cond_coords[n_data + j])
                }
                (Some(_), None) => true,
                _ => false,
            };
            if take_data {
                nb.push(nd[a]);
                a += 1;
            } else {
                nb.push(n_data + nn[b]);
                b += 1;
            }
        }
        if single_pool {
            nb.truncate(max_neighbors);
        }

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
        node_grid.insert(target);
        cond_coords.push(target);
        cond_vals.push(z);
    }

    Ok(sim_ns)
}

#[allow(clippy::too_many_arguments)]
fn simulate_one<const D: usize>(
    data: &PointSet<D>,
    data_scores: &[f64],
    data_grid: &BucketGrid<D>,
    ns: &NormalScore,
    model: &VariogramModel,
    centers: &[[f64; D]],
    levels: Option<&[u8]>,
    extents: ([f64; D], [f64; D]),
    cfg: &SgsConfig,
    seed: u64,
) -> Result<Vec<f64>> {
    let sim_ns = simulate_gaussian_path(
        data.coords(),
        data_scores,
        data_grid,
        model,
        centers,
        levels,
        extents,
        cfg.max_neighbors,
        cfg.search_radius,
        cfg.max_node_neighbors,
        seed,
    )?;
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
            let c = c0 - model.gamma_dh(sep(pi, coords[j]));
            ws.a[ii * n + jj] = c;
            ws.a[jj * n + ii] = c;
        }
        ws.b[ii] = c0 - model.gamma_dh(sep(pi, target));
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
    fn grid_level_assigns_coarse_subgrids() {
        assert_eq!(grid_level(&[0, 0], 3), 3);
        assert_eq!(grid_level(&[8, 8], 3), 3);
        assert_eq!(grid_level(&[4, 0], 3), 2);
        assert_eq!(grid_level(&[2, 2], 3), 1);
        assert_eq!(grid_level(&[3, 1], 3), 0);
        assert_eq!(grid_level(&[6, 4], 3), 1);
    }

    #[test]
    fn multigrid_and_node_quota_run_reproducibly() {
        let (data, model, grid) = setup();
        let base = SgsConfig {
            n_realizations: 3,
            seed: 11,
            max_neighbors: 12,
            search_radius: None,
            ..Default::default()
        };
        let plain = sequential_gaussian_simulation(&data, &model, &grid, &base).unwrap();
        // Multigrid path: reproducible, in bounds, and a different (still
        // valid) realization than the fully random path.
        let mg_cfg = SgsConfig {
            multigrid: 2,
            ..base.clone()
        };
        let mg1 = sequential_gaussian_simulation(&data, &model, &grid, &mg_cfg).unwrap();
        let mg2 = sequential_gaussian_simulation(&data, &model, &grid, &mg_cfg).unwrap();
        assert_eq!(mg1.realizations, mg2.realizations);
        assert_ne!(mg1.realizations, plain.realizations);
        // Separate node quota (GSLIB nodmax): reproducible and distinct.
        let quota_cfg = SgsConfig {
            max_node_neighbors: Some(8),
            ..base
        };
        let q1 = sequential_gaussian_simulation(&data, &model, &grid, &quota_cfg).unwrap();
        let q2 = sequential_gaussian_simulation(&data, &model, &grid, &quota_cfg).unwrap();
        assert_eq!(q1.realizations, q2.realizations);
        assert_ne!(q1.realizations, plain.realizations);
        let (lo, hi) = data
            .values()
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        for r in mg1.realizations.iter().chain(&q1.realizations) {
            for &v in r {
                assert!(v >= lo - 1e-9 && v <= hi + 1e-9);
            }
        }
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

    #[test]
    fn rejects_power_model() {
        let (data, _model, grid) = setup();
        let power_model =
            VariogramModel::new(0.0, vec![Structure::new(ModelKind::Power(1.0), 1.0, 1.0)])
                .unwrap();
        let cfg = SgsConfig::default();
        assert!(sequential_gaussian_simulation(&data, &power_model, &grid, &cfg).is_err());
    }

    #[test]
    fn sgs_refactor_matches_pre_refactor_snapshot() {
        // Golden values captured from `sequential_gaussian_simulation` before
        // `simulate_one` was split into a thin NormalScore wrapper around the
        // new transform-agnostic `simulate_gaussian_path` (shared with
        // `crate::tgs`) -- proves the refactor is a pure code move, not a
        // behavior change, the same "snapshot before refactor" discipline
        // this project used for the median/full-IK split.
        let (data, model, grid) = setup();
        let cfg = SgsConfig {
            n_realizations: 2,
            seed: 123,
            ..Default::default()
        };
        let res = sequential_gaussian_simulation(&data, &model, &grid, &cfg).unwrap();
        let expected: Vec<f64> = vec![
            1.609420096732717,
            1.0,
            1.0515346005203996,
            2.778411439103282,
            4.282795501942834,
            2.986262247251089,
            3.5585944410421164,
            5.811724913756676,
            7.119203135431646,
            8.0,
            2.5925959550776896,
            1.0,
            1.0,
            1.6591486366170949,
            4.372894004001342,
            2.4371800456403356,
            1.3158665392812854,
            3.524203973462011,
            3.824105715519657,
            6.127456848991475,
            4.1849385816072555,
            2.044573638149189,
            2.958049790302484,
            2.1122222064394576,
            6.083656710257774,
            5.708714375082879,
            3.398078964972221,
            5.301482740803119,
            5.982805284959382,
            6.521200258044789,
            3.501632661837138,
            2.2660851445575223,
            5.084109560065783,
            3.8678474756306365,
            5.205436846611172,
            5.424285390110229,
            3.0688701373988354,
            4.946915315783958,
            5.13451519955948,
            5.8628058448608,
            1.986017360997654,
            2.2818732657988643,
            2.471338999348178,
            5.1859137009052265,
            6.576441633841625,
            4.687728078492773,
            1.5334374469590868,
            6.504324595308734,
            5.350529158393245,
            5.321600138398312,
            3.181025193259936,
            2.8949132472461723,
            5.857362598104042,
            7.822404869323578,
            5.666883982110092,
            4.657268856373125,
            4.439022623375225,
            5.8090284460950565,
            5.195894098813175,
            5.459835190507572,
            1.705803819054489,
            3.424635324501761,
            4.4074358046316435,
            2.2417134662697946,
            5.82994325167567,
            6.4627681414523845,
            6.273030754638173,
            5.289524111052564,
            8.0,
            7.024141677273456,
            2.4846333217546444,
            3.182374701416424,
            3.324547601895643,
            1.8933211231369014,
            5.111392384469608,
            7.268378917765277,
            7.12778324860589,
            8.0,
            7.755186799502751,
            8.0,
            2.2596148878177535,
            2.741045637627748,
            2.1452266429270956,
            1.8973160441753687,
            5.426669141193214,
            5.45992168176052,
            5.855586250305649,
            8.0,
            7.325284504780275,
            8.0,
            2.3865470360086274,
            2.4009628888743553,
            5.761686742216664,
            6.427873276616176,
            6.689714935440602,
            4.749073730225176,
            5.31632515753265,
            6.612751973575912,
            5.700912341666603,
            7.139726497966095,
        ];
        assert_eq!(res.realizations[0].len(), expected.len());
        for (a, b) in res.realizations[0].iter().zip(&expected) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }
}
