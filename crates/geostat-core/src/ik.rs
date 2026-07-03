//! Standalone indicator kriging: local ccdf estimation at a set of cutoffs.
//!
//! At each target, simple indicator kriging (with the global proportion as
//! the known mean) estimates `F(cutoff_k)`; GSLIB-style order-relation
//! corrections are applied, and an E-type estimate plus conditional
//! variance are derived from the corrected ccdf assuming intra-class
//! uniformity (consistent with the SIS sampler).

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::search::KdTree;
use crate::sis::{indicator_ccdf, order_corrections};
use crate::tails::TailModel;
use crate::variogram::VariogramModel;

/// Configuration for indicator kriging.
#[derive(Debug, Clone)]
pub struct IkConfig {
    /// Indicator cutoffs, strictly ascending, inside the data value range.
    pub cutoffs: Vec<f64>,
    /// Indicator variogram model(s): either one per cutoff (full IK), or a
    /// single shared model for all cutoffs (**median IK**, GSLIB `mik=1`) —
    /// see [`crate::sis::SisConfig::models`] for why this amortizes the
    /// factorization cost across cutoffs.
    pub models: Vec<VariogramModel>,
    /// Ordinary indicator kriging (`Σw=1`, no assumed known mean) instead of
    /// the default simple IK (global proportion as the known mean).
    pub ordinary: bool,
    /// Maximum nearest conditioning points per target (all when unset).
    pub max_neighbors: Option<usize>,
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

/// Local ccdf estimate at one target.
#[derive(Debug, Clone)]
pub struct CcdfEstimate {
    /// Order-corrected `F(cutoff_k)` values.
    pub ccdf: Vec<f64>,
    /// E-type estimate (mean of the local distribution).
    pub e_type: f64,
    /// Conditional variance of the local distribution.
    pub cond_var: f64,
}

/// Indicator kriging at arbitrary targets. Returns one ccdf estimate per
/// target, in order.
pub fn indicator_kriging<const D: usize>(
    data: &PointSet<D>,
    targets: &[[f64; D]],
    cfg: &IkConfig,
) -> Result<Vec<CcdfEstimate>> {
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
    if cfg.max_neighbors == Some(0) {
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
    validate_ccdf_tails(cfg.lower_tail, cfg.upper_tail, cfg.cutoffs[nc - 1])?;

    let local = cfg.max_neighbors.is_some() || cfg.search_radius.is_some();
    let tree = local.then(|| KdTree::build(data.coords()));
    let all: Vec<usize> = (0..data.len()).collect();

    crate::parallel::par_try_map(targets.len(), |t| {
        let target = targets[t];
        let nb = match &tree {
            Some(tree) => tree.k_nearest(
                target,
                cfg.max_neighbors.unwrap_or(data.len()),
                cfg.search_radius,
            ),
            None => all.clone(),
        };
        let mut ccdf = vec![0.0; nc];
        if nb.is_empty() {
            ccdf.copy_from_slice(&props);
        } else {
            indicator_ccdf(
                data.coords(),
                data.values(),
                &nb,
                &cfg.cutoffs,
                &props,
                &cfg.models,
                target,
                cfg.ordinary,
                &mut ccdf,
            )?;
            order_corrections(&mut ccdf);
        }
        let (e_type, cond_var) = ccdf_moments(
            &ccdf,
            &cfg.cutoffs,
            tail_min,
            tail_max,
            cfg.lower_tail,
            cfg.upper_tail,
        );
        Ok(CcdfEstimate {
            ccdf,
            e_type,
            cond_var,
        })
    })
}

/// Validates ccdf tail models: `None` is not a distribution here, and a
/// hyperbolic upper tail needs a positive last cutoff (Pareto support).
pub(crate) fn validate_ccdf_tails(
    lower: TailModel,
    upper: TailModel,
    last_cutoff: f64,
) -> Result<()> {
    lower.validate_lower()?;
    upper.validate_upper()?;
    if lower == TailModel::None || upper == TailModel::None {
        return Err(GeostatError::InvalidParameter(
            "ccdf tails need an interpolation model (linear, power or hyperbolic)".into(),
        ));
    }
    if matches!(upper, TailModel::Hyperbolic(_)) && !(last_cutoff > 0.0) {
        return Err(GeostatError::InvalidParameter(
            "hyperbolic upper tail requires a positive last cutoff".into(),
        ));
    }
    Ok(())
}

/// Mean and variance of the ccdf assuming intra-class uniformity, with the
/// configured tail models on `[tail_min, c_1]` and `[c_K, tail_max]`.
pub(crate) fn ccdf_moments(
    ccdf: &[f64],
    cutoffs: &[f64],
    tail_min: f64,
    tail_max: f64,
    lower_tail: TailModel,
    upper_tail: TailModel,
) -> (f64, f64) {
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut f_lo = 0.0;
    let mut z_lo = tail_min;
    let nc = ccdf.len();
    for k in 0..=nc {
        let f_hi = if k < nc { ccdf[k] } else { 1.0 };
        let p = (f_hi - f_lo).max(0.0);
        if p > 0.0 {
            let (ez, ez2) = if k == 0 {
                crate::tails::lower_moments(lower_tail, tail_min, cutoffs[0])
            } else if k == nc {
                crate::tails::upper_moments(upper_tail, cutoffs[nc - 1], tail_max)
            } else {
                // Uniform within the class [z_lo, z_hi].
                let z_hi = cutoffs[k];
                (
                    0.5 * (z_lo + z_hi),
                    (z_lo * z_lo + z_lo * z_hi + z_hi * z_hi) / 3.0,
                )
            };
            mean += p * ez;
            m2 += p * ez2;
        }
        f_lo = f_hi;
        if k < nc {
            z_lo = cutoffs[k];
        }
    }
    (mean, (m2 - mean * mean).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::variogram::{ModelKind, Structure};

    fn setup() -> (PointSet, IkConfig) {
        let mut rng = Rng::new(17);
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for _ in 0..80 {
            let x = rng.uniform() * 100.0;
            let y = rng.uniform() * 100.0;
            coords.push([x, y]);
            values.push((x / 25.0).sin() * 2.0 + (y / 30.0).cos() + 0.2 * rng.normal());
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
        let cfg = IkConfig {
            cutoffs,
            models,
            ordinary: false,
            max_neighbors: None,
            search_radius: None,
            tail_min: None,
            tail_max: None,
            lower_tail: TailModel::Linear,
            upper_tail: TailModel::Linear,
        };
        (data, cfg)
    }

    #[test]
    fn ccdf_is_monotone_bounded_and_exactish_at_data() {
        let (data, cfg) = setup();
        let targets: Vec<[f64; 2]> = (0..10).map(|i| data.coord(i * 7)).collect();
        let est = indicator_kriging(&data, &targets, &cfg).unwrap();
        for (e, t) in est.iter().zip(&targets) {
            assert!(e.ccdf.windows(2).all(|w| w[0] <= w[1] + 1e-12));
            assert!(e.ccdf.iter().all(|&f| (0.0..=1.0).contains(&f)));
            assert!(e.cond_var >= 0.0);
            // At a datum the ccdf should be decisive about its class:
            // the observed value's class has most of the probability.
            let v = data.value(data.coords().iter().position(|c| c == t).unwrap());
            let mut f_lo = 0.0;
            let mut best = 0.0;
            for (k, &f) in e.ccdf.iter().enumerate() {
                let p = f - f_lo;
                let in_class = if k == 0 {
                    v <= cfg.cutoffs[0]
                } else {
                    v > cfg.cutoffs[k - 1] && v <= cfg.cutoffs[k]
                };
                if in_class {
                    best = p;
                }
                f_lo = f;
            }
            if v > *cfg.cutoffs.last().unwrap() {
                best = 1.0 - f_lo;
            }
            assert!(best > 0.5, "class probability {best} at datum");
        }
    }

    #[test]
    fn median_ik_matches_full_ik_when_models_coincide() {
        // `setup()` already uses the identical model for every cutoff, so
        // full IK (one system per cutoff) and median IK (`models.len()==1`,
        // one shared factorization) must agree exactly -- the whole point
        // of the optimization is that it changes *how* the same weights are
        // computed, never the result.
        let (data, full_cfg) = setup();
        let mut median_cfg = full_cfg.clone();
        median_cfg.models = vec![full_cfg.models[0].clone()];

        let targets: Vec<[f64; 2]> = (0..15).map(|i| [i as f64 * 6.0, 40.0]).collect();
        let full = indicator_kriging(&data, &targets, &full_cfg).unwrap();
        let median = indicator_kriging(&data, &targets, &median_cfg).unwrap();
        for (f, m) in full.iter().zip(&median) {
            for (fc, mc) in f.ccdf.iter().zip(&m.ccdf) {
                assert!((fc - mc).abs() < 1e-10, "{fc} vs {mc}");
            }
            assert!((f.e_type - m.e_type).abs() < 1e-10);
            assert!((f.cond_var - m.cond_var).abs() < 1e-10);
        }
    }

    #[test]
    fn ordinary_ik_is_bounded_monotone_and_exact_at_data() {
        let (data, mut cfg) = setup();
        cfg.ordinary = true;
        let targets: Vec<[f64; 2]> = (0..10).map(|i| data.coord(i * 7)).collect();
        let est = indicator_kriging(&data, &targets, &cfg).unwrap();
        for (e, t) in est.iter().zip(&targets) {
            assert!(e.ccdf.windows(2).all(|w| w[0] <= w[1] + 1e-12));
            assert!(e.ccdf.iter().all(|&f| (0.0..=1.0).contains(&f)));
            assert!(e.cond_var >= 0.0);
            let v = data.value(data.coords().iter().position(|c| c == t).unwrap());
            let mut f_lo = 0.0;
            let mut best = 0.0;
            for (k, &f) in e.ccdf.iter().enumerate() {
                let p = f - f_lo;
                let in_class = if k == 0 {
                    v <= cfg.cutoffs[0]
                } else {
                    v > cfg.cutoffs[k - 1] && v <= cfg.cutoffs[k]
                };
                if in_class {
                    best = p;
                }
                f_lo = f;
            }
            if v > *cfg.cutoffs.last().unwrap() {
                best = 1.0 - f_lo;
            }
            assert!(best > 0.5, "class probability {best} at datum");
        }
    }

    #[test]
    fn rejects_bad_model_count() {
        let (data, cfg) = setup();
        let targets: Vec<[f64; 2]> = vec![[50.0, 50.0]];
        let mut bad = cfg.clone();
        bad.models = vec![cfg.models[0].clone(), cfg.models[0].clone()]; // 2 for 3 cutoffs
        assert!(indicator_kriging(&data, &targets, &bad).is_err());
    }

    #[test]
    fn far_field_returns_global_distribution() {
        let (data, cfg) = setup();
        let est = indicator_kriging(&data, &[[1e6, 1e6]], &cfg).unwrap();
        let n = data.len() as f64;
        for (k, &c) in cfg.cutoffs.iter().enumerate() {
            let p = data.values().iter().filter(|&&v| v <= c).count() as f64 / n;
            assert!(
                (est[0].ccdf[k] - p).abs() < 1e-6,
                "cutoff {k}: {} vs global {p}",
                est[0].ccdf[k]
            );
        }
    }

    #[test]
    fn e_type_tracks_kriging_roughly() {
        // The E-type estimate should correlate strongly with the data field.
        let (data, cfg) = setup();
        let targets: Vec<[f64; 2]> = data.coords().to_vec();
        let est = indicator_kriging(&data, &targets, &cfg).unwrap();
        let n = data.len() as f64;
        let mo = data.mean();
        let me = est.iter().map(|e| e.e_type).sum::<f64>() / n;
        let mut cov = 0.0;
        let mut vo = 0.0;
        let mut ve = 0.0;
        for (e, &o) in est.iter().zip(data.values()) {
            cov += (e.e_type - me) * (o - mo);
            vo += (o - mo) * (o - mo);
            ve += (e.e_type - me) * (e.e_type - me);
        }
        let corr = cov / (vo.sqrt() * ve.sqrt());
        assert!(corr > 0.85, "E-type vs data correlation {corr}");
    }

    #[test]
    fn tail_models_shift_the_ccdf_moments() {
        let ccdf = [0.3, 0.6, 0.85];
        let cutoffs = [1.0, 2.0, 3.0];
        // Power(1) tails reproduce the linear (pre-v0.7) moments exactly.
        let lin = ccdf_moments(
            &ccdf,
            &cutoffs,
            0.0,
            5.0,
            TailModel::Linear,
            TailModel::Linear,
        );
        let pow1 = ccdf_moments(
            &ccdf,
            &cutoffs,
            0.0,
            5.0,
            TailModel::Power(1.0),
            TailModel::Power(1.0),
        );
        assert!((lin.0 - pow1.0).abs() < 1e-12 && (lin.1 - pow1.1).abs() < 1e-12);
        // A hyperbolic upper tail (capped well above) pushes the E-type and
        // the conditional variance up relative to the linear tail.
        let hyp = ccdf_moments(
            &ccdf,
            &cutoffs,
            0.0,
            50.0,
            TailModel::Linear,
            TailModel::Hyperbolic(1.5),
        );
        assert!(hyp.0 > lin.0, "e-type {} vs {}", hyp.0, lin.0);
        assert!(hyp.1 > lin.1, "cond var {} vs {}", hyp.1, lin.1);
        // Tail models must be actual distributions for IK.
        assert!(validate_ccdf_tails(TailModel::None, TailModel::Linear, 3.0).is_err());
        assert!(validate_ccdf_tails(TailModel::Linear, TailModel::Hyperbolic(1.5), -1.0).is_err());
        assert!(validate_ccdf_tails(TailModel::Linear, TailModel::Hyperbolic(1.5), 3.0).is_ok());
    }

    #[test]
    fn rejects_bad_config() {
        let (data, cfg) = setup();
        let mut bad = cfg.clone();
        bad.cutoffs = vec![2.0, 1.0, 3.0];
        assert!(indicator_kriging(&data, &[[0.0, 0.0]], &bad).is_err());
        let mut bad = cfg.clone();
        bad.models.pop();
        assert!(indicator_kriging(&data, &[[0.0, 0.0]], &bad).is_err());
        let mut bad = cfg;
        bad.cutoffs[2] = 1e9;
        assert!(indicator_kriging(&data, &[[0.0, 0.0]], &bad).is_err());
    }
}
