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
use crate::linalg::solve;
use crate::rng::{Rng, splitmix64};
use crate::search::BucketGrid;
use crate::simulation::SgsResult;
use crate::tails::{self, TailModel};
use crate::variogram::VariogramModel;

/// Configuration for sequential indicator simulation.
#[derive(Debug, Clone)]
pub struct SisConfig {
    /// Indicator cutoffs, strictly ascending, inside the data value range.
    pub cutoffs: Vec<f64>,
    /// Indicator variogram model(s): either one per cutoff (full IK), or a
    /// single shared model for all cutoffs (**median IK**, GSLIB `mik=1`) —
    /// the same spatial structure fit to the median cutoff's indicator and
    /// reused everywhere, which amortizes one factorization across every
    /// cutoff at each node instead of paying for `nc` (an ~nc× saving in
    /// the hot loop; see [`indicator_ccdf`]). Sill ≈ p(1-p) either way.
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
        }
    }
}

/// Runs conditional sequential indicator simulation on a 2-D grid.
pub fn sequential_indicator_simulation(
    data: &PointSet,
    grid: &Grid2D,
    cfg: &SisConfig,
) -> Result<SgsResult> {
    let realizations = sis_at(data, &grid.centers(), cfg)?;
    Ok(SgsResult {
        grid: grid.clone(),
        realizations,
    })
}

/// SIS at an arbitrary set of simulation nodes.
pub fn sis_at<const D: usize>(
    data: &PointSet<D>,
    nodes: &[[f64; D]],
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

    // Global proportions: the SK means of the indicators.
    let n = data.len() as f64;
    let props: Vec<f64> = cfg
        .cutoffs
        .iter()
        .map(|&c| data.values().iter().filter(|&&v| v <= c).count() as f64 / n)
        .collect();
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
    crate::parallel::par_try_map(cfg.n_realizations, |r| {
        let mut seed_state = cfg.seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let seed_r = splitmix64(&mut seed_state);
        simulate_one(data, nodes, cfg, &props, tail_min, tail_max, seed_r)
    })
}

#[allow(clippy::too_many_arguments)]
fn simulate_one<const D: usize>(
    data: &PointSet<D>,
    centers: &[[f64; D]],
    cfg: &SisConfig,
    props: &[f64],
    tail_min: f64,
    tail_max: f64,
    seed: u64,
) -> Result<Vec<f64>> {
    let nc = cfg.cutoffs.len();
    let mut rng = Rng::new(seed);
    let n_cells = centers.len();

    let mut path: Vec<usize> = (0..n_cells).collect();
    rng.shuffle(&mut path);

    let (dbmin, dbmax) = data.bbox();
    let mut min = dbmin;
    let mut max = dbmax;
    for c in centers {
        for d in 0..D {
            min[d] = min[d].min(c[d]);
            max[d] = max[d].max(c[d]);
        }
    }
    let mut search = BucketGrid::new(min, max, data.len() + n_cells);
    let mut cond_coords: Vec<[f64; D]> = data.coords().to_vec();
    let mut cond_vals: Vec<f64> = data.values().to_vec();
    for &p in data.coords() {
        search.insert(p);
    }

    let mut sim = vec![0.0_f64; n_cells];
    let mut ccdf = vec![0.0_f64; nc];

    for &cell in &path {
        let target = centers[cell];
        let nb = search.k_nearest(target, cfg.max_neighbors, cfg.search_radius);

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
        search.insert(target);
        cond_coords.push(target);
        cond_vals.push(z);
    }
    Ok(sim)
}

/// Indicator-kriging weights for one target's neighbourhood under a single
/// covariance model — the expensive part (build + factorize + solve an
/// `n×n`, or `(n+1)×(n+1)` for ordinary, system). Shared across every
/// cutoff when the *same* model is used for all of them (median IK, GSLIB
/// `mik=1`): the weights depend only on the spatial covariance structure,
/// not on which cutoff's indicator data they are dotted with, so one
/// factorization serves all `nc` cutoffs — an ~nc× saving in the hot loop
/// this function is called from (see [`indicator_ccdf`]).
///
/// `ordinary` adds the `Σw=1` (Lagrange) row/column — ordinary indicator
/// kriging instead of simple IK with the global proportion as the known
/// mean. The returned vector has `nb.len()` entries for simple, one more
/// (the Lagrange multiplier) for ordinary.
fn indicator_weights<const D: usize>(
    coords: &[[f64; D]],
    nb: &[usize],
    model: &VariogramModel,
    target: [f64; D],
    ordinary: bool,
) -> Result<Vec<f64>> {
    let n = nb.len();
    let c0 = model.covariance_dh([0.0; D]);
    let stabilizer = c0 * 1e-9;
    let dim = if ordinary { n + 1 } else { n };
    let mut a = Array2::<f64>::zeros((dim, dim));
    let mut b = vec![0.0; dim];
    let sep = |a: [f64; D], b: [f64; D]| {
        let mut dh = [0.0; D];
        for d in 0..D {
            dh[d] = a[d] - b[d];
        }
        dh
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
        if ordinary {
            a[[ii, n]] = 1.0;
            a[[n, ii]] = 1.0;
        }
    }
    if ordinary {
        b[n] = 1.0;
    }
    solve(a, b)
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
/// [`crate::ik::indicator_kriging`] both call.
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
) -> Result<()> {
    if models.len() == 1 {
        let w = indicator_weights(coords, nb, &models[0], target, ordinary)?;
        for k in 0..cutoffs.len() {
            ccdf[k] = indicator_estimate(&w, nb, vals, cutoffs[k], props[k], ordinary);
        }
    } else {
        for k in 0..cutoffs.len() {
            let w = indicator_weights(coords, nb, &models[k], target, ordinary)?;
            ccdf[k] = indicator_estimate(&w, nb, vals, cutoffs[k], props[k], ordinary);
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
            let (rho, _sigma_hard, sigma_soft) = crate::collocated::estimate_collocated_stats(&h, &s)?;
            Ok(MarkovBayesCalibration { rho, sigma_soft })
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
) -> f64 {
    let n = nb.len();
    let mut est = p;
    for (ii, &i) in nb.iter().enumerate() {
        let ind = if vals[i] <= cutoff { 1.0 } else { 0.0 };
        est += w[ii] * (ind - p);
    }
    est += w[n] * (soft_val - p);
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
        let model = if models.len() == 1 { &models[0] } else { &models[k] };
        let w = indicator_weights_soft(coords, nb, model, target, calib[k])?;
        ccdf[k] = indicator_estimate_soft(&w, nb, vals, cutoffs[k], props[k], soft[k]);
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
/// interpolation and the configured tail models.
#[allow(clippy::too_many_arguments)]
fn sample_ccdf(
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
}
