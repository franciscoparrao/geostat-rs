//! Normal score transform and inverse normal CDF.

use crate::error::{GeostatError, Result};
use crate::tails::{self, TailModel};

/// Standard normal CDF (Abramowitz & Stegun 26.2.19, the same approximation
/// as GSLIB's `gcum`; absolute error < 7.5e-8).
pub fn norm_cdf(z: f64) -> f64 {
    const B: [f64; 5] = [
        0.319_381_530,
        -0.356_563_782,
        1.781_477_937,
        -1.821_255_978,
        1.330_274_429,
    ];
    const P: f64 = 0.231_641_9;
    let x = z.abs();
    let t = 1.0 / (1.0 + P * x);
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let poly = t * (B[0] + t * (B[1] + t * (B[2] + t * (B[3] + t * B[4]))));
    let upper = 1.0 - pdf * poly;
    if z >= 0.0 { upper } else { 1.0 - upper }
}

/// Tail options for the normal-score back-transform (GSLIB `ltail`/`utail`
/// with `zmin`/`zmax`). The default extrapolates nothing (clamps at the data
/// extremes, the pre-v0.7 behavior).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Tails {
    /// Lower-tail model (`Hyperbolic` is invalid here).
    pub lower: TailModel,
    /// Upper-tail model.
    pub upper: TailModel,
    /// Lower bound `zmin` (required unless `lower` is `None`).
    pub lower_bound: Option<f64>,
    /// Upper bound `zmax` (required unless `upper` is `None`; caps the
    /// hyperbolic tail).
    pub upper_bound: Option<f64>,
}

/// Inverse standard normal CDF (Acklam's rational approximation,
/// absolute error < 1.15e-9). Input is clamped to the open unit interval.
pub fn inv_norm_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.02425;

    let p = p.clamp(1e-300, 1.0 - 1e-16);

    let tail = |p: f64| -> f64 {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    if p < P_LOW {
        tail(p)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -tail(1.0 - p)
    }
}

/// Rank-based normal score transform with linear interpolation between
/// quantile knots. Ties are assigned the mean score of their rank group.
///
/// By default back-transformed values are clamped to the observed data
/// range; [`NormalScore::fit_with_tails`] enables GSLIB-style tail
/// extrapolation beyond the extremes.
#[derive(Debug, Clone)]
pub struct NormalScore {
    knots_v: Vec<f64>,
    knots_s: Vec<f64>,
    tails: Tails,
}

impl NormalScore {
    /// Fits the transform to a sample. Requires at least two distinct values.
    /// Back-transformed values are clamped to the data range (no tails).
    pub fn fit(values: &[f64]) -> Result<Self> {
        Self::fit_with_tails(values, Tails::default())
    }

    /// Fits the transform with declustering weights: the reference cdf uses
    /// weighted plotting positions (GSLIB `nscore` with a weight column), so
    /// preferentially clustered samples do not bias the target distribution.
    /// Weights must be positive; with equal weights this is exactly
    /// [`NormalScore::fit`].
    pub fn fit_weighted(values: &[f64], weights: &[f64]) -> Result<Self> {
        Self::fit_weighted_with_tails(values, weights, Tails::default())
    }

    /// Fits the transform with GSLIB-style tail extrapolation for the
    /// back-transform. `tails.lower_bound`/`upper_bound` must bracket the
    /// data range when the corresponding tail model is not `None`.
    pub fn fit_with_tails(values: &[f64], tails: Tails) -> Result<Self> {
        Self::fit_weighted_with_tails(values, &vec![1.0; values.len()], tails)
    }

    /// Weighted fit with tail extrapolation; see [`NormalScore::fit_weighted`]
    /// and [`NormalScore::fit_with_tails`].
    pub fn fit_weighted_with_tails(values: &[f64], weights: &[f64], tails: Tails) -> Result<Self> {
        if values.len() < 2 {
            return Err(GeostatError::InsufficientData(
                "normal score transform requires at least 2 values".into(),
            ));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(GeostatError::InvalidParameter("non-finite value".into()));
        }
        if weights.len() != values.len() {
            return Err(GeostatError::DimensionMismatch(format!(
                "{} weights vs {} values",
                weights.len(),
                values.len()
            )));
        }
        if weights.iter().any(|w| !(w.is_finite() && *w > 0.0)) {
            return Err(GeostatError::InvalidParameter(
                "declustering weights must be finite and positive".into(),
            ));
        }
        let n = values.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let total: f64 = weights.iter().sum();

        // Weighted plotting positions p_i = (cum_{i-1} + w_i / 2) / W; with
        // equal weights this reduces to the (i + 0.5) / n convention.
        let mut cum = 0.0;
        let scores: Vec<f64> = order
            .iter()
            .map(|&i| {
                let p = (cum + 0.5 * weights[i]) / total;
                cum += weights[i];
                inv_norm_cdf(p)
            })
            .collect();

        // Collapse ties into single knots with the weighted mean score of
        // the group.
        let mut knots_v = Vec::new();
        let mut knots_s = Vec::new();
        let mut i = 0;
        while i < n {
            let vi = values[order[i]];
            let mut j = i;
            let mut sum = 0.0;
            let mut wsum = 0.0;
            while j < n && values[order[j]] == vi {
                sum += weights[order[j]] * scores[j];
                wsum += weights[order[j]];
                j += 1;
            }
            knots_v.push(vi);
            knots_s.push(sum / wsum);
            i = j;
        }
        if knots_v.len() < 2 {
            return Err(GeostatError::InvalidParameter(
                "all values are identical; cannot build a normal score transform".into(),
            ));
        }

        tails.lower.validate_lower()?;
        tails.upper.validate_upper()?;
        if tails.lower != TailModel::None {
            let zmin = tails.lower_bound.ok_or_else(|| {
                GeostatError::InvalidParameter("lower tail requires a lower bound (zmin)".into())
            })?;
            if !(zmin <= knots_v[0]) {
                return Err(GeostatError::InvalidParameter(format!(
                    "zmin ({zmin}) must not exceed the data minimum ({})",
                    knots_v[0]
                )));
            }
        }
        if tails.upper != TailModel::None {
            let zmax = tails.upper_bound.ok_or_else(|| {
                GeostatError::InvalidParameter("upper tail requires an upper bound (zmax)".into())
            })?;
            let vmax = *knots_v.last().expect("at least two knots");
            if !(zmax >= vmax) {
                return Err(GeostatError::InvalidParameter(format!(
                    "zmax ({zmax}) must not be below the data maximum ({vmax})"
                )));
            }
            if matches!(tails.upper, TailModel::Hyperbolic(_)) && !(vmax > 0.0) {
                return Err(GeostatError::InvalidParameter(
                    "hyperbolic upper tail requires a positive data maximum".into(),
                ));
            }
        }
        Ok(Self {
            knots_v,
            knots_s,
            tails,
        })
    }

    /// Maps a data value to its normal score.
    pub fn transform(&self, v: f64) -> f64 {
        interp(&self.knots_v, &self.knots_s, v)
    }

    /// Maps a normal score back to data units. Scores beyond the table are
    /// extrapolated with the configured tail models (clamped to the data
    /// range by default).
    pub fn back_transform(&self, s: f64) -> f64 {
        let last = self.knots_s.len() - 1;
        if s < self.knots_s[0] && self.tails.lower != TailModel::None {
            // Relative cdf position within the lower-tail class [0, F1].
            let f1 = norm_cdf(self.knots_s[0]);
            let t = if f1 > 0.0 { norm_cdf(s) / f1 } else { 0.0 };
            let zmin = self.tails.lower_bound.expect("validated at fit");
            return tails::draw_lower(self.tails.lower, zmin, self.knots_v[0], t);
        }
        if s > self.knots_s[last] && self.tails.upper != TailModel::None {
            // Relative cdf position within the upper-tail class [Fk, 1].
            let fk = norm_cdf(self.knots_s[last]);
            let span = 1.0 - fk;
            let t = if span > 0.0 {
                (norm_cdf(s) - fk) / span
            } else {
                1.0
            };
            let zmax = self.tails.upper_bound.expect("validated at fit");
            return tails::draw_upper(self.tails.upper, self.knots_v[last], zmax, t);
        }
        interp(&self.knots_s, &self.knots_v, s)
    }
}

/// Piecewise-linear interpolation over strictly increasing `xs`, clamped at
/// the extremes.
fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let last = xs.len() - 1;
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[last] {
        return ys[last];
    }
    let hi = xs.partition_point(|&v| v < x);
    let lo = hi - 1;
    let t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    ys[lo] + t * (ys[hi] - ys[lo])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_norm_cdf_known_values() {
        assert!(inv_norm_cdf(0.5).abs() < 1e-9);
        assert!((inv_norm_cdf(0.975) - 1.959_964).abs() < 1e-5);
        assert!((inv_norm_cdf(0.025) + 1.959_964).abs() < 1e-5);
        assert!((inv_norm_cdf(0.001) + 3.090_232).abs() < 1e-5);
    }

    #[test]
    fn round_trip_identity() {
        let values = vec![3.1, 0.2, 7.5, 1.1, 4.4, 2.2, 9.9, 0.5];
        let ns = NormalScore::fit(&values).unwrap();
        for &v in &values {
            let back = ns.back_transform(ns.transform(v));
            assert!((back - v).abs() < 1e-9, "{v} -> {back}");
        }
    }

    #[test]
    fn scores_are_centered() {
        let values: Vec<f64> = (0..101).map(|i| i as f64).collect();
        let ns = NormalScore::fit(&values).unwrap();
        // Median value maps to ~0.
        assert!(ns.transform(50.0).abs() < 1e-9);
        // Monotone.
        assert!(ns.transform(10.0) < ns.transform(90.0));
    }

    #[test]
    fn weighted_fit_with_equal_weights_matches_unweighted() {
        let values = vec![3.1, 0.2, 7.5, 1.1, 4.4, 2.2, 9.9, 0.5, 1.1];
        let a = NormalScore::fit(&values).unwrap();
        // Unit weights reproduce the plotting positions bit for bit.
        let b = NormalScore::fit_weighted(&values, &vec![1.0; values.len()]).unwrap();
        for &v in &values {
            assert_eq!(a.transform(v), b.transform(v), "score differs at {v}");
        }
        // Any other constant weight is the same distribution (up to
        // floating-point accumulation).
        let c = NormalScore::fit_weighted(&values, &vec![2.5; values.len()]).unwrap();
        for &v in &values {
            assert!((a.transform(v) - c.transform(v)).abs() < 1e-12);
        }
    }

    #[test]
    fn weights_shift_the_reference_distribution() {
        // Upweighting the isolated low value gives it more cdf mass: its
        // own score moves toward the center and every value above it is
        // pushed further up the cdf.
        let values = vec![1.0, 5.0, 5.5, 6.0, 6.5, 7.0];
        let equal = NormalScore::fit(&values).unwrap();
        let mut weights = vec![0.2; values.len()];
        weights[0] = 2.0;
        let weighted = NormalScore::fit_weighted(&values, &weights).unwrap();
        assert!(weighted.transform(1.0) > equal.transform(1.0));
        assert!(weighted.transform(5.0) > equal.transform(5.0));
        assert!(weighted.transform(7.0) > equal.transform(7.0));
        // Invalid weights rejected.
        assert!(NormalScore::fit_weighted(&values, &[1.0; 3]).is_err());
        let mut bad = vec![1.0; values.len()];
        bad[2] = 0.0;
        assert!(NormalScore::fit_weighted(&values, &bad).is_err());
    }

    #[test]
    fn norm_cdf_matches_inverse() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((norm_cdf(1.959_964) - 0.975).abs() < 1e-7);
        assert!((norm_cdf(-1.959_964) - 0.025).abs() < 1e-7);
        for p in [0.01, 0.2, 0.5, 0.8, 0.99] {
            assert!((norm_cdf(inv_norm_cdf(p)) - p).abs() < 1e-6, "p = {p}");
        }
    }

    #[test]
    fn tail_extrapolation_extends_beyond_data_range() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tails = Tails {
            lower: TailModel::Linear,
            upper: TailModel::Power(2.0),
            lower_bound: Some(0.0),
            upper_bound: Some(20.0),
        };
        let ns = NormalScore::fit_with_tails(&values, tails).unwrap();
        // Beyond the last knot the back-transform now exceeds the data max
        // (the clamped default would return 8.0) but respects zmax.
        let hi = ns.back_transform(3.5);
        assert!(hi > 8.0 && hi <= 20.0, "upper tail value {hi}");
        let lo = ns.back_transform(-3.5);
        assert!((0.0..1.0).contains(&lo), "lower tail value {lo}");
        // Monotone through the tail regions.
        assert!(ns.back_transform(-4.0) <= ns.back_transform(-3.0));
        assert!(ns.back_transform(3.0) <= ns.back_transform(4.0));
        // Continuous at the table edge (power tails have a vertical tangent
        // there, so approach is sqrt-slow in the score offset).
        let s_last = ns.transform(8.0);
        assert!((ns.back_transform(s_last + 1e-9) - 8.0).abs() < 1e-2);
        assert!((ns.back_transform(s_last + 1e-13) - 8.0).abs() < 1e-4);
        // Interior behavior unchanged.
        for &v in &values {
            assert!((ns.back_transform(ns.transform(v)) - v).abs() < 1e-9);
        }
    }

    #[test]
    fn hyperbolic_upper_tail_caps_at_zmax() {
        let values = vec![1.0, 2.0, 3.0, 5.0, 9.0];
        let tails = Tails {
            lower: TailModel::None,
            upper: TailModel::Hyperbolic(1.5),
            lower_bound: None,
            upper_bound: Some(100.0),
        };
        let ns = NormalScore::fit_with_tails(&values, tails).unwrap();
        let z = ns.back_transform(5.0);
        assert!(z > 9.0 && z <= 100.0, "hyperbolic tail value {z}");
    }

    #[test]
    fn tail_validation_rejects_bad_configs() {
        let values = vec![1.0, 2.0, 3.0];
        // Missing bounds.
        let t = Tails {
            lower: TailModel::Linear,
            ..Default::default()
        };
        assert!(NormalScore::fit_with_tails(&values, t).is_err());
        // Bound inside the data range.
        let t = Tails {
            upper: TailModel::Linear,
            upper_bound: Some(2.5),
            ..Default::default()
        };
        assert!(NormalScore::fit_with_tails(&values, t).is_err());
        // Hyperbolic lower tail.
        let t = Tails {
            lower: TailModel::Hyperbolic(1.5),
            lower_bound: Some(0.0),
            ..Default::default()
        };
        assert!(NormalScore::fit_with_tails(&values, t).is_err());
        // Hyperbolic with non-positive data maximum.
        let t = Tails {
            upper: TailModel::Hyperbolic(1.5),
            upper_bound: Some(1.0),
            ..Default::default()
        };
        assert!(NormalScore::fit_with_tails(&[-3.0, -2.0, -1.0], t).is_err());
    }

    #[test]
    fn handles_ties_and_clamps_tails() {
        let values = vec![1.0, 1.0, 1.0, 2.0, 3.0];
        let ns = NormalScore::fit(&values).unwrap();
        assert!(ns.transform(1.0) < 0.0);
        // Outside data range: clamped.
        assert_eq!(ns.back_transform(10.0), 3.0);
        assert_eq!(ns.back_transform(-10.0), 1.0);
        // Degenerate sample rejected.
        assert!(NormalScore::fit(&[5.0, 5.0, 5.0]).is_err());
    }
}
