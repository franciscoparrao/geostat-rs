//! Sequential indicator simulation (SIS).
//!
//! GSLIB-style `sisim` for continuous variables: the local conditional
//! distribution is estimated by simple indicator kriging at each cutoff
//! (with the global proportion as the known mean), order-relation
//! corrections are applied (average of upward/downward passes), and the
//! simulated value is drawn from the corrected ccdf with linear
//! interpolation within classes and linear tails to `[tail_min, tail_max]`.

use ndarray::Array2;
use rayon::prelude::*;

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::grid::Grid2D;
use crate::linalg::solve;
use crate::rng::{Rng, splitmix64};
use crate::search::BucketGrid;
use crate::simulation::SgsResult;
use crate::variogram::VariogramModel;

/// Configuration for sequential indicator simulation.
#[derive(Debug, Clone)]
pub struct SisConfig {
    /// Indicator cutoffs, strictly ascending, inside the data value range.
    pub cutoffs: Vec<f64>,
    /// Indicator variogram model per cutoff (sill ≈ p(1-p)).
    pub models: Vec<VariogramModel>,
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
}

impl Default for SisConfig {
    fn default() -> Self {
        Self {
            cutoffs: Vec::new(),
            models: Vec::new(),
            n_realizations: 1,
            seed: 42,
            max_neighbors: 16,
            search_radius: None,
            tail_min: None,
            tail_max: None,
        }
    }
}

/// Runs conditional sequential indicator simulation on a grid.
pub fn sequential_indicator_simulation(
    data: &PointSet,
    grid: &Grid2D,
    cfg: &SisConfig,
) -> Result<SgsResult> {
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
    if cfg.models.len() != nc {
        return Err(GeostatError::DimensionMismatch(format!(
            "{} models for {nc} cutoffs",
            cfg.models.len()
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

    let realizations: Vec<Vec<f64>> = (0..cfg.n_realizations)
        .into_par_iter()
        .map(|r| {
            let mut seed_state = cfg.seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let seed_r = splitmix64(&mut seed_state);
            simulate_one(data, grid, cfg, &props, tail_min, tail_max, seed_r)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(SgsResult {
        grid: grid.clone(),
        realizations,
    })
}

#[allow(clippy::too_many_arguments)]
fn simulate_one(
    data: &PointSet,
    grid: &Grid2D,
    cfg: &SisConfig,
    props: &[f64],
    tail_min: f64,
    tail_max: f64,
    seed: u64,
) -> Result<Vec<f64>> {
    let nc = cfg.cutoffs.len();
    let mut rng = Rng::new(seed);
    let centers = grid.centers();
    let n_cells = centers.len();

    let mut path: Vec<usize> = (0..n_cells).collect();
    rng.shuffle(&mut path);

    let (dbmin, dbmax) = data.bbox();
    let min = [dbmin[0].min(grid.x0), dbmin[1].min(grid.y0)];
    let max = [
        dbmax[0].max(grid.x0 + grid.dx * grid.nx as f64),
        dbmax[1].max(grid.y0 + grid.dy * grid.ny as f64),
    ];
    let mut search = BucketGrid::new(min, max, data.len() + n_cells);
    let mut cond_coords: Vec<[f64; 2]> = data.coords().to_vec();
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
            for k in 0..nc {
                ccdf[k] = indicator_sk(
                    &cond_coords,
                    &cond_vals,
                    &nb,
                    cfg.cutoffs[k],
                    props[k],
                    &cfg.models[k],
                    target,
                )?;
            }
            order_corrections(&mut ccdf);
        }

        let z = sample_ccdf(&ccdf, &cfg.cutoffs, tail_min, tail_max, rng.uniform());
        sim[cell] = z;
        search.insert(target);
        cond_coords.push(target);
        cond_vals.push(z);
    }
    Ok(sim)
}

/// Simple indicator kriging at one cutoff: `F = p + sum(w_i (i_i - p))`.
fn indicator_sk(
    coords: &[[f64; 2]],
    vals: &[f64],
    nb: &[usize],
    cutoff: f64,
    p: f64,
    model: &VariogramModel,
    target: [f64; 2],
) -> Result<f64> {
    let n = nb.len();
    let c0 = model.covariance_dh([0.0, 0.0]);
    let stabilizer = c0 * 1e-9;
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
    let w = solve(a, b)?;
    let mut est = p;
    for (ii, &i) in nb.iter().enumerate() {
        let ind = if vals[i] <= cutoff { 1.0 } else { 0.0 };
        est += w[ii] * (ind - p);
    }
    Ok(est)
}

/// GSLIB-style order-relation corrections: clamp to [0, 1], then average an
/// upward (running max) and a downward (running min) pass.
fn order_corrections(ccdf: &mut [f64]) {
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
/// interpolation and linear tails.
fn sample_ccdf(ccdf: &[f64], cutoffs: &[f64], tail_min: f64, tail_max: f64, u: f64) -> f64 {
    let nc = ccdf.len();
    let mut f_lo = 0.0;
    let mut z_lo = tail_min;
    for k in 0..nc {
        if u <= ccdf[k] {
            let span = ccdf[k] - f_lo;
            let t = if span > 1e-12 { (u - f_lo) / span } else { 0.5 };
            return z_lo + t * (cutoffs[k] - z_lo);
        }
        f_lo = ccdf[k];
        z_lo = cutoffs[k];
    }
    // Upper tail.
    let span = 1.0 - f_lo;
    let t = if span > 1e-12 { (u - f_lo) / span } else { 0.5 };
    z_lo + t * (tail_max - z_lo)
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
