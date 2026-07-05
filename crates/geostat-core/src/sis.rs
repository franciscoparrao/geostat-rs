//! Sequential indicator simulation (SIS).
//!
//! GSLIB-style `sisim` for continuous variables: the local conditional
//! distribution is estimated by simple indicator kriging at each cutoff
//! (with the global proportion as the known mean), order-relation
//! corrections are applied (average of upward/downward passes), and the
//! simulated value is drawn from the corrected ccdf with linear
//! interpolation within classes and linear tails to `[tail_min, tail_max]`.

use ndarray::Array2;

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::grid::Grid2D;
use crate::linalg::{cholesky_solve_in_place, solve};
use crate::rng::{Rng, splitmix64};
use crate::search::BucketGrid;
use crate::simulation::SgsResult;
use crate::tails::{self, TailModel};
use crate::variogram::VariogramModel;

/// Reusable buffers for [`indicator_weights`]' system, avoiding a fresh
/// allocation per node/target -- mirrors `SkWorkspace` in `simulation.rs`
/// (AUDIT-2026-07-v2.md §7 Fase 6 item #15: SIS was "a generation behind"
/// SGS in its own hot loop). Shared by [`crate::sis::simulate_one`]
/// (reused across every simulated node in a realization) and
/// [`crate::ik::indicator_kriging`] (reused across a target's cutoffs).
#[derive(Default)]
pub(crate) struct IndicatorWorkspace {
    a: Vec<f64>,
    b: Vec<f64>,
    sol: Vec<f64>,
}

/// Configuration for sequential indicator simulation.
///
/// `#[non_exhaustive]`: construct via `SisConfig { cutoffs, models, ..
/// Default::default() }` (AUDIT-2026-07-v2.md §6 Fase 5).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SisConfig {
    /// Indicator cutoffs, strictly ascending, inside the data value range.
    pub cutoffs: Vec<f64>,
    /// Indicator variogram model(s): either one per cutoff (full IK), or a
    /// single shared model for all cutoffs (**median IK**, GSLIB `mik=1`) —
    /// the same spatial structure fit to the median cutoff's indicator and
    /// reused everywhere, which amortizes one factorization across every
    /// cutoff at each node instead of paying for `nc` (an ~nc× saving in
    /// the hot loop; see `indicator_ccdf`). Sill ≈ p(1-p) either way.
    pub models: Vec<VariogramModel>,
    /// Ordinary indicator kriging (`Σw=1`, no assumed known mean) instead of
    /// the default simple IK (global proportion as the known mean).
    pub ordinary: bool,
    /// Number of realizations.
    pub n_realizations: usize,
    /// Base seed.
    pub seed: u64,
    /// Maximum conditioning points per node.
    pub max_neighbors: usize,
    /// Optional search radius.
    pub search_radius: Option<f64>,
    /// Lower tail bound (default: data minimum).
    pub tail_min: Option<f64>,
    /// Upper tail bound (default: data maximum).
    pub tail_max: Option<f64>,
    /// Lower-tail interpolation between `tail_min` and the first cutoff
    /// (GSLIB `ltail`; `Linear` is the GSLIB and pre-v0.7 default).
    pub lower_tail: TailModel,
    /// Upper-tail interpolation between the last cutoff and `tail_max`
    /// (GSLIB `utail`; hyperbolic tails are capped at `tail_max`).
    pub upper_tail: TailModel,
    /// Optional declustering weights (one positive weight per data point,
    /// e.g. from [`crate::declustering::cell_declustering_weights`]): the
    /// global cutoff proportions (the simple-IK means) are computed as
    /// weighted means instead of plain counts, so preferential sampling
    /// does not bias the marginal ccdf the same way it would bias an
    /// unweighted normal-score transform (AUDIT-2026-07-v2.md §7 Fase 6
    /// item #15 -- SIS previously ignored declustering entirely).
    pub decluster_weights: Option<Vec<f64>>,
    /// Separate quota for previously simulated nodes (GSLIB `nodmax`),
    /// exactly like [`crate::simulation::SgsConfig::max_node_neighbors`]:
    /// when set, each neighbourhood takes up to `max_neighbors` original
    /// data **plus** up to this many simulated nodes, instead of one pool
    /// where dense simulated nodes can crowd the hard data out.
    pub max_node_neighbors: Option<usize>,
    /// Multiple-grid simulation levels (GSLIB `nmult`; grid entry points
    /// only), exactly like [`crate::simulation::SgsConfig::multigrid`]: `0`
    /// (default) keeps a fully random path.
    pub multigrid: u8,
}

impl Default for SisConfig {
    fn default() -> Self {
        Self {
            cutoffs: Vec::new(),
            models: Vec::new(),
            ordinary: false,
            n_realizations: 1,
            seed: 42,
            max_neighbors: 16,
            search_radius: None,
            tail_min: None,
            tail_max: None,
            lower_tail: TailModel::Linear,
            upper_tail: TailModel::Linear,
            decluster_weights: None,
            max_node_neighbors: None,
            multigrid: 0,
        }
    }
}

/// Runs conditional sequential indicator simulation on a 2-D grid.
pub fn sequential_indicator_simulation(
    data: &PointSet,
    grid: &Grid2D,
    cfg: &SisConfig,
) -> Result<SgsResult> {
    let levels = (cfg.multigrid > 0).then(|| {
        (0..grid.n_cells())
            .map(|i| crate::simulation::grid_level(&[i % grid.nx, i / grid.nx], cfg.multigrid))
            .collect::<Vec<u8>>()
    });
    let realizations = sis_at_with_levels(data, &grid.centers(), levels.as_deref(), cfg)?;
    Ok(SgsResult {
        grid: grid.clone(),
        realizations,
    })
}

/// Runs conditional sequential indicator simulation on a 3-D grid, returning
/// the realizations in grid storage order (mirrors
/// [`crate::simulation::sequential_gaussian_simulation_3d`]).
pub fn sequential_indicator_simulation_3d(
    data: &PointSet<3>,
    grid: &crate::grid::Grid3D,
    cfg: &SisConfig,
) -> Result<Vec<Vec<f64>>> {
    let levels = (cfg.multigrid > 0).then(|| {
        (0..grid.n_cells())
            .map(|i| {
                let ix = i % grid.nx;
                let iy = (i / grid.nx) % grid.ny;
                let iz = i / (grid.nx * grid.ny);
                crate::simulation::grid_level(&[ix, iy, iz], cfg.multigrid)
            })
            .collect::<Vec<u8>>()
    });
    sis_at_with_levels(data, &grid.centers(), levels.as_deref(), cfg)
}

/// SIS at an arbitrary set of simulation nodes. `multigrid` is ignored here
/// (it needs grid topology); use the grid entry points for multiple-grid
/// simulation.
pub fn sis_at<const D: usize>(
    data: &PointSet<D>,
    nodes: &[[f64; D]],
    cfg: &SisConfig,
) -> Result<Vec<Vec<f64>>> {
    sis_at_with_levels(data, nodes, None, cfg)
}

fn sis_at_with_levels<const D: usize>(
    data: &PointSet<D>,
    nodes: &[[f64; D]],
    levels: Option<&[u8]>,
    cfg: &SisConfig,
) -> Result<Vec<Vec<f64>>> {
    let nc = cfg.cutoffs.len();
    if nc == 0 {
        return Err(GeostatError::InvalidParameter(
            "at least one cutoff required".into(),
        ));
    }
    if cfg.cutoffs.windows(2).any(|w| !(w[0] < w[1])) {
        return Err(GeostatError::InvalidParameter(
            "cutoffs must be strictly ascending".into(),
        ));
    }
    if cfg.models.len() != nc && cfg.models.len() != 1 {
        return Err(GeostatError::DimensionMismatch(format!(
            "{} models for {nc} cutoffs (expected {nc}, or 1 for median IK)",
            cfg.models.len()
        )));
    }
    if cfg.models.iter().any(VariogramModel::has_power) {
        return Err(GeostatError::InvalidParameter(
            "SIS needs a valid indicator covariance function and cannot use the unbounded \
             Power model"
                .into(),
        ));
    }
    if let Some(kind) = cfg
        .models
        .iter()
        .find_map(|m| m.invalid_structure_for_dim(D))
    {
        return Err(GeostatError::InvalidParameter(format!(
            "{kind:?} is not a valid covariance in {D} dimensions; use Spherical instead for a \
             3-D-safe bounded structure"
        )));
    }
    if cfg.n_realizations == 0 || cfg.max_neighbors == 0 {
        return Err(GeostatError::InvalidParameter(
            "n_realizations and max_neighbors must be at least 1".into(),
        ));
    }
    if let Some(r) = cfg.search_radius
        && !(r > 0.0)
    {
        return Err(GeostatError::InvalidParameter(format!(
            "search radius must be positive, got {r}"
        )));
    }

    // Global proportions: the SK means of the indicators, weighted by
    // `decluster_weights` when given (unweighted counts otherwise) --
    // AUDIT-2026-07-v2.md §7 Fase 6 item #15.
    let props: Vec<f64> = match &cfg.decluster_weights {
        Some(w) => {
            if w.len() != data.len() {
                return Err(GeostatError::DimensionMismatch(format!(
                    "{} declustering weights vs {} data points",
                    w.len(),
                    data.len()
                )));
            }
            let wsum: f64 = w.iter().sum();
            if !(wsum > 0.0) {
                return Err(GeostatError::InvalidParameter(
                    "declustering weights must sum to a positive value".into(),
                ));
            }
            cfg.cutoffs
                .iter()
                .map(|&c| {
                    data.values()
                        .iter()
                        .zip(w)
                        .filter(|&(&v, _)| v <= c)
                        .map(|(_, &wi)| wi)
                        .sum::<f64>()
                        / wsum
                })
                .collect()
        }
        None => {
            let n = data.len() as f64;
            cfg.cutoffs
                .iter()
                .map(|&c| data.values().iter().filter(|&&v| v <= c).count() as f64 / n)
                .collect()
        }
    };
    for (k, &p) in props.iter().enumerate() {
        if !(p > 0.0 && p < 1.0) {
            return Err(GeostatError::InvalidParameter(format!(
                "cutoff {} (= {}) leaves no data on one side (proportion {p})",
                k, cfg.cutoffs[k]
            )));
        }
    }
    let (dmin, dmax) = data
        .values()
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    let tail_min = cfg.tail_min.unwrap_or(dmin);
    let tail_max = cfg.tail_max.unwrap_or(dmax);
    if !(tail_min <= cfg.cutoffs[0]) || !(tail_max >= cfg.cutoffs[nc - 1]) {
        return Err(GeostatError::InvalidParameter(
            "tail bounds must bracket the cutoffs".into(),
        ));
    }
    crate::ik::validate_ccdf_tails(cfg.lower_tail, cfg.upper_tail, cfg.cutoffs[nc - 1])?;

    if nodes.is_empty() {
        return Err(GeostatError::InvalidParameter(
            "no simulation nodes given".into(),
        ));
    }

    // Extents covering data and nodes, shared by both search structures.
    let (dbmin, dbmax) = data.bbox();
    let mut min = dbmin;
    let mut max = dbmax;
    for c in nodes {
        for d in 0..D {
            min[d] = min[d].min(c[d]);
            max[d] = max[d].max(c[d]);
        }
    }
    // Static store of the original data, built once and shared across
    // realizations (only simulated nodes change per realization) --
    // mirrors `sgs_at_with_levels` in `simulation.rs`.
    let mut data_grid = BucketGrid::new(min, max, data.len());
    for &p in data.coords() {
        data_grid.insert(p);
    }

    crate::parallel::par_try_map(cfg.n_realizations, |r| {
        let mut seed_state = cfg.seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let seed_r = splitmix64(&mut seed_state);
        simulate_one(
            data,
            &data_grid,
            nodes,
            levels,
            (min, max),
            cfg,
            &props,
            tail_min,
            tail_max,
            seed_r,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn simulate_one<const D: usize>(
    data: &PointSet<D>,
    data_grid: &BucketGrid<D>,
    centers: &[[f64; D]],
    levels: Option<&[u8]>,
    extents: ([f64; D], [f64; D]),
    cfg: &SisConfig,
    props: &[f64],
    tail_min: f64,
    tail_max: f64,
    seed: u64,
) -> Result<Vec<f64>> {
    let nc = cfg.cutoffs.len();
    let mut rng = Rng::new(seed);
    let n_cells = centers.len();
    let n_data = data.len();

    // Random path; with multigrid levels, coarse levels first (stable sort
    // keeps the shuffle within each level) -- mirrors `simulate_one` in
    // `simulation.rs`.
    let mut path: Vec<usize> = (0..n_cells).collect();
    rng.shuffle(&mut path);
    if let Some(levels) = levels {
        path.sort_by_key(|&i| std::cmp::Reverse(levels[i]));
    }

    // Simulated nodes get their own store; the data store is shared.
    let (min, max) = extents;
    let mut node_grid = BucketGrid::new(min, max, n_cells);
    let mut cond_coords: Vec<[f64; D]> = data.coords().to_vec();
    let mut cond_vals: Vec<f64> = data.values().to_vec();

    // Quotas: `max_neighbors` original data plus `nodmax` simulated nodes
    // (GSLIB ndmax/nodmax), exactly like SGS.
    let nodmax = cfg.max_node_neighbors.unwrap_or(cfg.max_neighbors);
    let single_pool = cfg.max_node_neighbors.is_none();

    let mut sim = vec![0.0_f64; n_cells];
    let mut ccdf = vec![0.0_f64; nc];
    let mut ws = IndicatorWorkspace::default();
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
        let nd = data_grid.k_nearest(target, cfg.max_neighbors, cfg.search_radius);
        let nn = node_grid.k_nearest(target, nodmax, cfg.search_radius);

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
            nb.truncate(cfg.max_neighbors);
        }

        if nb.is_empty() {
            ccdf.copy_from_slice(props);
        } else {
            indicator_ccdf(
                &cond_coords,
                &cond_vals,
                &nb,
                &cfg.cutoffs,
                props,
                &cfg.models,
                target,
                cfg.ordinary,
                &mut ccdf,
                &mut ws,
            )?;
            order_corrections(&mut ccdf);
        }

        let z = sample_ccdf(
            &ccdf,
            &cfg.cutoffs,
            tail_min,
            tail_max,
            cfg.lower_tail,
            cfg.upper_tail,
            rng.uniform(),
        );
        sim[cell] = z;
        node_grid.insert(target);
        cond_coords.push(target);
        cond_vals.push(z);
    }
    Ok(sim)
}

/// Indicator-kriging weights for one target's neighbourhood under a single
/// covariance model — the expensive part (build + factorize + solve an
/// `n×n`, or `(n+1)×(n+1)` for ordinary, system) — written into `ws.sol`
/// (`nb.len()` entries for simple, one more for ordinary's Lagrange
/// multiplier) instead of returning a freshly allocated `Vec`, and reusing
/// `ws`'s buffers instead of allocating a new `Array2` every call
/// (AUDIT-2026-07-v2.md §7 Fase 6 item #15, mirroring `SkWorkspace` in
/// `simulation.rs`). Shared across every cutoff when the *same* model is
/// used for all of them (median IK, GSLIB `mik=1`): the weights depend only
/// on the spatial covariance structure, not on which cutoff's indicator
/// data they are dotted with, so one factorization serves all `nc` cutoffs
/// — an ~nc× saving in the hot loop this function is called from (see
/// [`indicator_ccdf`]).
///
/// Simple IK's system is SPD (solved by in-place Cholesky, no allocation);
/// `ordinary` adds the `Σw=1` (Lagrange) row/column, making it indefinite,
/// so that case still goes through the general (allocating) LU `solve`.
fn indicator_weights<const D: usize>(
    coords: &[[f64; D]],
    nb: &[usize],
    model: &VariogramModel,
    target: [f64; D],
    ordinary: bool,
    ws: &mut IndicatorWorkspace,
) -> Result<()> {
    let n = nb.len();
    let c0 = model.covariance_dh([0.0; D]);
    let stabilizer = c0 * 1e-9;
    let dim = if ordinary { n + 1 } else { n };
    ws.a.clear();
    ws.a.resize(dim * dim, 0.0);
    ws.b.clear();
    ws.b.resize(dim, 0.0);
    let sep = |a: [f64; D], b: [f64; D]| {
        let mut dh = [0.0; D];
        for d in 0..D {
            dh[d] = a[d] - b[d];
        }
        dh
    };
    for (ii, &i) in nb.iter().enumerate() {
        let pi = coords[i];
        ws.a[ii * dim + ii] = c0 + stabilizer;
        for (jj, &j) in nb.iter().enumerate().skip(ii + 1) {
            let c = c0 - model.gamma_dh(sep(pi, coords[j]));
            ws.a[ii * dim + jj] = c;
            ws.a[jj * dim + ii] = c;
        }
        ws.b[ii] = c0 - model.gamma_dh(sep(pi, target));
        if ordinary {
            ws.a[ii * dim + n] = 1.0;
            ws.a[n * dim + ii] = 1.0;
        }
    }
    if ordinary {
        ws.b[n] = 1.0;
    }
    if ordinary {
        // Indefinite (Lagrange row): needs a pivoted solve, not Cholesky.
        let mut a = Array2::<f64>::zeros((dim, dim));
        for r in 0..dim {
            for c in 0..dim {
                a[[r, c]] = ws.a[r * dim + c];
            }
        }
        ws.sol = solve(a, ws.b.clone())?;
    } else {
        ws.sol.clear();
        ws.sol.extend_from_slice(&ws.b);
        cholesky_solve_in_place(&mut ws.a, dim, &mut ws.sol)?;
    }
    Ok(())
}

/// Indicator/ccdf estimate at one cutoff from precomputed weights (see
/// [`indicator_weights`]): `Σ w_i·i_i` for ordinary (no known mean needed —
/// that is the point of the unbiasedness constraint), `p + Σ w_i·(i_i - p)`
/// for simple.
fn indicator_estimate(
    w: &[f64],
    nb: &[usize],
    vals: &[f64],
    cutoff: f64,
    p: f64,
    ordinary: bool,
) -> f64 {
    if ordinary {
        nb.iter()
            .enumerate()
            .map(|(ii, &i)| {
                let ind = if vals[i] <= cutoff { 1.0 } else { 0.0 };
                w[ii] * ind
            })
            .sum()
    } else {
        let mut est = p;
        for (ii, &i) in nb.iter().enumerate() {
            let ind = if vals[i] <= cutoff { 1.0 } else { 0.0 };
            est += w[ii] * (ind - p);
        }
        est
    }
}

/// ccdf at every cutoff for one target: shares one factorization across all
/// cutoffs when `models.len() == 1` (median IK), or uses one model per
/// cutoff (`models.len() == cutoffs.len()`, full IK) otherwise. `ordinary`
/// selects ordinary vs simple indicator kriging (see [`indicator_weights`]).
/// This is the single entry point [`crate::sis::simulate_one`] and
/// [`crate::ik::indicator_kriging`] both call. `ws` is caller-provided so
/// its buffers can be reused across many targets/nodes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn indicator_ccdf<const D: usize>(
    coords: &[[f64; D]],
    vals: &[f64],
    nb: &[usize],
    cutoffs: &[f64],
    props: &[f64],
    models: &[VariogramModel],
    target: [f64; D],
    ordinary: bool,
    ccdf: &mut [f64],
    ws: &mut IndicatorWorkspace,
) -> Result<()> {
    if models.len() == 1 {
        indicator_weights(coords, nb, &models[0], target, ordinary, ws)?;
        for k in 0..cutoffs.len() {
            ccdf[k] = indicator_estimate(&ws.sol, nb, vals, cutoffs[k], props[k], ordinary);
        }
    } else {
        for k in 0..cutoffs.len() {
            indicator_weights(coords, nb, &models[k], target, ordinary, ws)?;
            ccdf[k] = indicator_estimate(&ws.sol, nb, vals, cutoffs[k], props[k], ordinary);
        }
    }
    Ok(())
}

/// Markov-Bayes calibration for one cutoff (Zhu & Journel 1993): the
/// correlation `rho` between the hard indicator and a collocated soft datum,
/// and the soft datum's marginal standard deviation `sigma_soft`, both
/// estimated from calibration pairs (locations where both a hard
/// measurement and a collocated secondary reading are available) — see
/// [`calibrate_markov_bayes`]. This is the same Markov screening hypothesis
/// as [`crate::collocated::MarkovModel::Mm1`] (`crate::collocated`),
/// applied here per cutoff to a soft *probability* channel instead of once
/// to a continuous secondary variable.
#[derive(Debug, Clone, Copy)]
pub struct MarkovBayesCalibration {
    /// Correlation between the hard indicator and the soft datum at
    /// collocated calibration pairs, in `[-1, 1]`.
    pub rho: f64,
    /// Marginal standard deviation of the soft datum.
    pub sigma_soft: f64,
    /// Marginal mean of the soft datum at the calibration pairs (`E[Y]`).
    /// In the simple-kriging system each datum enters as a deviation from
    /// *its own* mean; the soft channel's mean is generally **not** the
    /// hard indicator's global proportion `p` unless the soft probabilities
    /// happen to be perfectly calibrated on average (AUDIT-2026-07-v2.md
    /// §1.7 — using `p` here silently imported the soft channel's
    /// calibration bias, if any, into every prediction with weight `w[n]`).
    pub mean_soft: f64,
}

/// Estimates one [`MarkovBayesCalibration`] per cutoff from collocated
/// hard/soft calibration pairs: `hard[i][k]` is the 0/1 indicator
/// (`Z(x_i) <= cutoffs[k]`) and `soft[i][k]` the corresponding soft
/// probability at the *same* location `i`, for every cutoff `k` (e.g. wells
/// with both a hard measurement and a co-located secondary reading).
pub fn calibrate_markov_bayes(
    hard: &[Vec<f64>],
    soft: &[Vec<f64>],
) -> Result<Vec<MarkovBayesCalibration>> {
    if hard.len() != soft.len() {
        return Err(GeostatError::DimensionMismatch(format!(
            "{} hard calibration rows vs {} soft rows",
            hard.len(),
            soft.len()
        )));
    }
    if hard.is_empty() {
        return Err(GeostatError::InsufficientData(
            "at least one calibration pair required".into(),
        ));
    }
    let nc = hard[0].len();
    if nc == 0 || hard.iter().any(|r| r.len() != nc) || soft.iter().any(|r| r.len() != nc) {
        return Err(GeostatError::DimensionMismatch(
            "all calibration rows must share the same (nonzero) number of cutoffs".into(),
        ));
    }
    (0..nc)
        .map(|k| {
            let h: Vec<f64> = hard.iter().map(|r| r[k]).collect();
            let s: Vec<f64> = soft.iter().map(|r| r[k]).collect();
            let (rho, _sigma_hard, sigma_soft) =
                crate::collocated::estimate_collocated_stats(&h, &s)?;
            let mean_soft = s.iter().sum::<f64>() / s.len() as f64;
            Ok(MarkovBayesCalibration {
                rho,
                sigma_soft,
                mean_soft,
            })
        })
        .collect()
}

/// Indicator-kriging weights with one collocated soft datum (Markov-Bayes;
/// simple-kriging form only — see [`MarkovBayesCalibration`]): extends
/// [`indicator_weights`]'s hard system by one row/column for the soft term,
/// whose cross-covariance to each hard neighbour follows the hard
/// indicator's own spatial shape (the same MM1 hypothesis
/// [`crate::collocated`] uses for continuous secondaries).
fn indicator_weights_soft<const D: usize>(
    coords: &[[f64; D]],
    nb: &[usize],
    model: &VariogramModel,
    target: [f64; D],
    calib: MarkovBayesCalibration,
) -> Result<Vec<f64>> {
    let n = nb.len();
    let c0 = model.covariance_dh([0.0; D]);
    let sigma_i = c0.max(1e-300).sqrt();
    let stabilizer = c0 * 1e-9;
    let dim = n + 1;
    let mut a = Array2::<f64>::zeros((dim, dim));
    let mut b = vec![0.0; dim];
    let sep = |a: [f64; D], b: [f64; D]| {
        let mut dh = [0.0; D];
        for d in 0..D {
            dh[d] = a[d] - b[d];
        }
        dh
    };
    let cross = |h: [f64; D]| -> f64 {
        calib.rho * (calib.sigma_soft / sigma_i) * (c0 - model.gamma_dh(h))
    };
    for (ii, &i) in nb.iter().enumerate() {
        let pi = coords[i];
        a[[ii, ii]] = c0 + stabilizer;
        for (jj, &j) in nb.iter().enumerate().skip(ii + 1) {
            let c = c0 - model.gamma_dh(sep(pi, coords[j]));
            a[[ii, jj]] = c;
            a[[jj, ii]] = c;
        }
        b[ii] = c0 - model.gamma_dh(sep(pi, target));
        let c_is = cross(sep(pi, target));
        a[[ii, n]] = c_is;
        a[[n, ii]] = c_is;
    }
    a[[n, n]] = calib.sigma_soft * calib.sigma_soft + stabilizer;
    b[n] = calib.rho * calib.sigma_soft * sigma_i;
    solve(a, b)
}

fn indicator_estimate_soft(
    w: &[f64],
    nb: &[usize],
    vals: &[f64],
    cutoff: f64,
    p: f64,
    soft_val: f64,
    mean_soft: f64,
) -> f64 {
    let n = nb.len();
    let mut est = p;
    for (ii, &i) in nb.iter().enumerate() {
        let ind = if vals[i] <= cutoff { 1.0 } else { 0.0 };
        est += w[ii] * (ind - p);
    }
    // Each simple-kriging datum enters as a deviation from its own mean:
    // the soft channel's mean is `mean_soft` (from calibration), not the
    // hard indicator's global proportion `p` (AUDIT-2026-07-v2.md §1.7).
    est += w[n] * (soft_val - mean_soft);
    est
}

/// ccdf at every cutoff for one target, incorporating one collocated soft
/// datum per cutoff via Markov-Bayes (simple IK only; see
/// [`indicator_weights_soft`]). `soft[k]`/`calib[k]` line up with
/// `cutoffs[k]`. Unlike [`indicator_ccdf`], median IK's shared-factorization
/// trick does not extend here: the calibration (and hence the soft
/// cross-covariance) is cutoff-specific even when the hard model is shared,
/// so weights are still solved once per cutoff.
#[allow(clippy::too_many_arguments)]
pub(crate) fn indicator_ccdf_soft<const D: usize>(
    coords: &[[f64; D]],
    vals: &[f64],
    nb: &[usize],
    cutoffs: &[f64],
    props: &[f64],
    models: &[VariogramModel],
    target: [f64; D],
    soft: &[f64],
    calib: &[MarkovBayesCalibration],
    ccdf: &mut [f64],
) -> Result<()> {
    for k in 0..cutoffs.len() {
        let model = if models.len() == 1 {
            &models[0]
        } else {
            &models[k]
        };
        let w = indicator_weights_soft(coords, nb, model, target, calib[k])?;
        ccdf[k] = indicator_estimate_soft(
            &w,
            nb,
            vals,
            cutoffs[k],
            props[k],
            soft[k],
            calib[k].mean_soft,
        );
    }
    Ok(())
}

/// GSLIB-style order-relation corrections: clamp to [0, 1], then average an
/// upward (running max) and a downward (running min) pass.
pub(crate) fn order_corrections(ccdf: &mut [f64]) {
    let nc = ccdf.len();
    for f in ccdf.iter_mut() {
        *f = f.clamp(0.0, 1.0);
    }
    let mut up = ccdf.to_vec();
    for k in 1..nc {
        up[k] = up[k].max(up[k - 1]);
    }
    let mut down = ccdf.to_vec();
    for k in (0..nc.saturating_sub(1)).rev() {
        down[k] = down[k].min(down[k + 1]);
    }
    for k in 0..nc {
        ccdf[k] = 0.5 * (up[k] + down[k]);
    }
}

/// Draws a value from the corrected ccdf with intra-class linear
/// interpolation and the configured tail models. `pub(crate)`: also used as
/// a deterministic ccdf quantile lookup by
/// [`crate::validation::accuracy_plot_ccdf`] (passing a fixed probability
/// instead of a random draw for `u` is exactly the inverse-ccdf transform).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_ccdf(
    ccdf: &[f64],
    cutoffs: &[f64],
    tail_min: f64,
    tail_max: f64,
    lower_tail: TailModel,
    upper_tail: TailModel,
    u: f64,
) -> f64 {
    let nc = ccdf.len();
    let mut f_lo = 0.0;
    let mut z_lo = tail_min;
    for k in 0..nc {
        if u <= ccdf[k] {
            let span = ccdf[k] - f_lo;
            let t = if span > 1e-12 { (u - f_lo) / span } else { 0.5 };
            if k == 0 {
                return tails::draw_lower(lower_tail, tail_min, cutoffs[0], t);
            }
            return z_lo + t * (cutoffs[k] - z_lo);
        }
        f_lo = ccdf[k];
        z_lo = cutoffs[k];
    }
    // Upper tail.
    let span = 1.0 - f_lo;
    let t = if span > 1e-12 { (u - f_lo) / span } else { 0.5 };
    tails::draw_upper(upper_tail, z_lo, tail_max, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::{ModelKind, Structure};

    fn setup() -> (PointSet, SisConfig, Grid2D) {
        let mut rng = Rng::new(21);
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for _ in 0..60 {
            let x = rng.uniform() * 100.0;
            let y = rng.uniform() * 100.0;
            coords.push([x, y]);
            values.push((x / 30.0).sin() * 2.0 + (y / 25.0).cos() + 0.3 * rng.normal());
        }
        let data = PointSet::new(coords, values).unwrap();
        let mut sorted = data.values().to_vec();
        sorted.sort_by(f64::total_cmp);
        let q = |p: f64| sorted[(p * sorted.len() as f64) as usize];
        let cutoffs = vec![q(0.25), q(0.5), q(0.75)];
        let models = cutoffs
            .iter()
            .map(|_| {
                VariogramModel::new(
                    0.02,
                    vec![Structure::new(ModelKind::Exponential, 0.2, 30.0)],
                )
                .unwrap()
            })
            .collect();
        let cfg = SisConfig {
            cutoffs,
            models,
            n_realizations: 4,
            seed: 11,
            ..Default::default()
        };
        let grid = Grid2D::from_bbox([0.0, 0.0], [100.0, 100.0], 12, 12).unwrap();
        (data, cfg, grid)
    }

    #[test]
    fn reproducible_and_bounded() {
        let (data, cfg, grid) = setup();
        let a = sequential_indicator_simulation(&data, &grid, &cfg).unwrap();
        let b = sequential_indicator_simulation(&data, &grid, &cfg).unwrap();
        assert_eq!(a.realizations, b.realizations);
        assert_ne!(a.realizations[0], a.realizations[1]);
        let (dmin, dmax) = data
            .values()
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        for real in &a.realizations {
            assert_eq!(real.len(), grid.n_cells());
            for &v in real {
                assert!(
                    v >= dmin - 1e-9 && v <= dmax + 1e-9,
                    "value {v} out of range"
                );
            }
        }
    }

    #[test]
    fn median_ik_matches_full_ik_when_models_coincide() {
        // `setup()` uses the identical model for every cutoff, so full IK
        // and median IK (`models.len()==1`, one shared factorization per
        // node instead of one per cutoff) must produce byte-identical
        // realizations for the same seed.
        let (data, full_cfg, grid) = setup();
        let mut median_cfg = full_cfg.clone();
        median_cfg.models = vec![full_cfg.models[0].clone()];

        let full = sequential_indicator_simulation(&data, &grid, &full_cfg).unwrap();
        let median = sequential_indicator_simulation(&data, &grid, &median_cfg).unwrap();
        assert_eq!(full.realizations, median.realizations);
    }

    #[test]
    fn ordinary_sis_is_reproducible_and_bounded() {
        let (data, mut cfg, grid) = setup();
        cfg.ordinary = true;
        let a = sequential_indicator_simulation(&data, &grid, &cfg).unwrap();
        let b = sequential_indicator_simulation(&data, &grid, &cfg).unwrap();
        assert_eq!(a.realizations, b.realizations);
        let (dmin, dmax) = data
            .values()
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        for real in &a.realizations {
            for &v in real {
                assert!(
                    v >= dmin - 1e-9 && v <= dmax + 1e-9,
                    "value {v} out of range"
                );
            }
        }
    }

    #[test]
    fn multigrid_and_node_quota_run_reproducibly() {
        // Mirrors `multigrid_and_node_quota_run_reproducibly` in
        // `simulation.rs` (AUDIT-2026-07-v2.md §7 Fase 6 item #15: SIS
        // lacked both features entirely before this).
        let (data, base, grid) = setup();
        let plain = sequential_indicator_simulation(&data, &grid, &base).unwrap();

        let mut mg_cfg = base.clone();
        mg_cfg.multigrid = 2;
        let mg1 = sequential_indicator_simulation(&data, &grid, &mg_cfg).unwrap();
        let mg2 = sequential_indicator_simulation(&data, &grid, &mg_cfg).unwrap();
        assert_eq!(mg1.realizations, mg2.realizations);
        assert_ne!(mg1.realizations, plain.realizations);

        let mut quota_cfg = base.clone();
        quota_cfg.max_node_neighbors = Some(6);
        let q1 = sequential_indicator_simulation(&data, &grid, &quota_cfg).unwrap();
        let q2 = sequential_indicator_simulation(&data, &grid, &quota_cfg).unwrap();
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
    fn decluster_weights_shift_the_global_proportions() {
        // AUDIT-2026-07-v2.md §7 Fase 6 item #15: SIS previously computed
        // its global cutoff proportions (the simple-IK known mean) as plain
        // counts, ignoring `decluster_weights` entirely. A tight cluster of
        // 9 zero-value points plus one isolated value-10 point: unweighted,
        // 9/10 of the mass sits at 0 (proportion 0.9 for cutoff 5.0);
        // weighting the cluster down to a combined weight of 1 (matching
        // the lone point) should pull the proportion to 0.5.
        //
        // The target sits far outside `search_radius`, so `sis_at` falls
        // back to `ccdf[0] = props[0]` exactly (the "no neighbours" branch)
        // and `sample_ccdf` draws from the lower/upper tail with
        // probability `props[0]`/`1 - props[0]` -- so the *ensemble*
        // fraction of realizations landing at or below the cutoff
        // converges to `props[0]` (the standard inverse-CDF sampling
        // argument), letting a black-box simulation test the internal,
        // otherwise-private proportion computation.
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..9 {
            coords.push([(i % 3) as f64 * 0.1, (i / 3) as f64 * 0.1]);
            values.push(0.0);
        }
        coords.push([50.0, 50.0]);
        values.push(10.0);
        let data = PointSet::new(coords, values).unwrap();
        let model = VariogramModel::new(
            0.02,
            vec![Structure::new(ModelKind::Exponential, 0.2, 30.0)],
        )
        .unwrap();
        let far_target = [1.0e6, 1.0e6];
        let base_cfg = SisConfig {
            cutoffs: vec![5.0],
            models: vec![model],
            n_realizations: 400,
            search_radius: Some(1.0),
            ..Default::default()
        };

        let unweighted = sis_at(&data, &[far_target], &base_cfg).unwrap();
        let below_unweighted =
            unweighted.iter().filter(|r| r[0] <= 5.0).count() as f64 / unweighted.len() as f64;
        assert!(
            (below_unweighted - 0.9).abs() < 0.08,
            "unweighted ensemble proportion {below_unweighted}"
        );

        let mut weighted_cfg = base_cfg;
        weighted_cfg.decluster_weights = Some(
            std::iter::repeat_n(1.0 / 9.0, 9)
                .chain(std::iter::once(1.0))
                .collect(),
        );
        let weighted = sis_at(&data, &[far_target], &weighted_cfg).unwrap();
        let below_weighted =
            weighted.iter().filter(|r| r[0] <= 5.0).count() as f64 / weighted.len() as f64;
        assert!(
            (below_weighted - 0.5).abs() < 0.08,
            "weighted ensemble proportion {below_weighted}"
        );
    }

    #[test]
    fn decluster_weights_validate_length_and_positivity() {
        let (data, mut cfg, _grid) = setup();
        cfg.decluster_weights = Some(vec![1.0; data.len() - 1]); // wrong length
        assert!(sis_at(&data, &[[50.0, 50.0]], &cfg).is_err());
        cfg.decluster_weights = Some(vec![0.0; data.len()]); // sums to zero
        assert!(sis_at(&data, &[[50.0, 50.0]], &cfg).is_err());
    }

    #[test]
    fn rejects_bad_model_count() {
        let (data, cfg, grid) = setup();
        let mut bad = cfg.clone();
        bad.models = vec![cfg.models[0].clone(), cfg.models[0].clone()]; // 2 for 3 cutoffs
        assert!(sequential_indicator_simulation(&data, &grid, &bad).is_err());
    }

    #[test]
    fn rejects_power_model() {
        let (data, mut cfg, grid) = setup();
        cfg.models = vec![
            VariogramModel::new(0.0, vec![Structure::new(ModelKind::Power(1.0), 1.0, 1.0)])
                .unwrap(),
        ];
        assert!(sequential_indicator_simulation(&data, &grid, &cfg).is_err());
    }

    #[test]
    fn ensemble_proportions_track_global() {
        let (data, mut cfg, grid) = setup();
        cfg.n_realizations = 20;
        let res = sequential_indicator_simulation(&data, &grid, &cfg).unwrap();
        let n_data = data.len() as f64;
        for (k, &c) in cfg.cutoffs.iter().enumerate() {
            let global = data.values().iter().filter(|&&v| v <= c).count() as f64 / n_data;
            let pooled: usize = res
                .realizations
                .iter()
                .flat_map(|r| r.iter())
                .filter(|&&v| v <= c)
                .count();
            let sim_prop = pooled as f64 / (20.0 * grid.n_cells() as f64);
            assert!(
                (sim_prop - global).abs() < 0.12,
                "cutoff {k}: simulated proportion {sim_prop:.3} vs global {global:.3}"
            );
        }
    }

    #[test]
    fn order_corrections_are_monotone_and_bounded() {
        let mut ccdf = vec![0.4, 0.2, 1.3, 0.9];
        order_corrections(&mut ccdf);
        assert!(ccdf.windows(2).all(|w| w[0] <= w[1] + 1e-12));
        assert!(ccdf.iter().all(|&f| (0.0..=1.0).contains(&f)));
    }

    #[test]
    fn rejects_bad_config() {
        let (data, cfg, grid) = setup();
        let mut bad = cfg.clone();
        bad.cutoffs = vec![1.0, 0.5];
        bad.models.truncate(2);
        assert!(sequential_indicator_simulation(&data, &grid, &bad).is_err());
        let mut bad = cfg.clone();
        bad.cutoffs[0] = 1e9; // outside the data range
        assert!(sequential_indicator_simulation(&data, &grid, &bad).is_err());
        let mut bad = cfg;
        bad.models.pop();
        assert!(sequential_indicator_simulation(&data, &grid, &bad).is_err());
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn order_corrections_are_always_monotone_and_bounded(
                mut raw in prop::collection::vec(-5.0f64..5.0, 1..15),
            ) {
                order_corrections(&mut raw);
                prop_assert!(raw.windows(2).all(|w| w[0] <= w[1] + 1e-12), "{raw:?}");
                prop_assert!(raw.iter().all(|&f| (0.0..=1.0).contains(&f)), "{raw:?}");
            }
        }
    }

    /// AUDIT-2026-07-v2.md §1.7: `indicator_estimate_soft` must use the soft
    /// datum's own calibrated mean, not the hard indicator's global
    /// proportion `p`, as the deviation point for the soft term -- a
    /// mismatch (soft channel calibrated to a mean far from `p`) must not
    /// silently leak into every prediction.
    #[test]
    fn indicator_estimate_soft_deviates_from_its_own_mean_not_from_p() {
        let w = [0.1, 0.2, 0.15]; // 2 hard neighbours + 1 soft weight
        let nb = [0usize, 1usize];
        let vals = [10.0, 20.0];
        let cutoff = 15.0;
        let p = 0.4; // hard indicator's global proportion, deliberately != mean_soft

        // Soft value fed to the target sits exactly at its own calibrated
        // mean: with a correct implementation the soft term's contribution
        // is exactly `w[2] * (soft_val - mean_soft) = 0`, regardless of how
        // far `mean_soft` is from `p`.
        let mean_soft = 0.9;
        let soft_val = mean_soft;
        let est = indicator_estimate_soft(&w, &nb, &vals, cutoff, p, soft_val, mean_soft);
        let hard_only: f64 = p + w[..2]
            .iter()
            .zip(&nb)
            .map(|(&wi, &i)| wi * (if vals[i] <= cutoff { 1.0 } else { 0.0 } - p))
            .sum::<f64>();
        assert!(
            (est - hard_only).abs() < 1e-12,
            "soft term should vanish when soft_val == mean_soft: {est} vs {hard_only}"
        );

        // Changing `mean_soft` while holding `soft_val` fixed must shift the
        // estimate by exactly `-w[2] * delta` (linear in mean_soft) -- the
        // old code (deviating from `p` instead) would not respond to
        // `mean_soft` at all.
        let est_shifted = indicator_estimate_soft(&w, &nb, &vals, cutoff, p, soft_val, 0.2);
        let expected_delta = -w[2] * (0.2 - mean_soft);
        assert!(
            (est_shifted - est - expected_delta).abs() < 1e-12,
            "{est_shifted} vs {} (est {est} + delta {expected_delta})",
            est + expected_delta
        );
    }
}
