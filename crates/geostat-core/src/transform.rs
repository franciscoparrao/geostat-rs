//! Normal score transform and inverse normal CDF.

use crate::error::{GeostatError, Result};

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
/// Back-transformed values are clamped to the observed data range
/// (no tail extrapolation in this MVP).
#[derive(Debug, Clone)]
pub struct NormalScore {
    knots_v: Vec<f64>,
    knots_s: Vec<f64>,
}

impl NormalScore {
    /// Fits the transform to a sample. Requires at least two distinct values.
    pub fn fit(values: &[f64]) -> Result<Self> {
        if values.len() < 2 {
            return Err(GeostatError::InsufficientData(
                "normal score transform requires at least 2 values".into(),
            ));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(GeostatError::InvalidParameter("non-finite value".into()));
        }
        let n = values.len();
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let scores: Vec<f64> = (0..n)
            .map(|i| inv_norm_cdf((i as f64 + 0.5) / n as f64))
            .collect();

        // Collapse ties into single knots with the mean score of the group.
        let mut knots_v = Vec::new();
        let mut knots_s = Vec::new();
        let mut i = 0;
        while i < n {
            let mut j = i;
            let mut sum = 0.0;
            while j < n && sorted[j] == sorted[i] {
                sum += scores[j];
                j += 1;
            }
            knots_v.push(sorted[i]);
            knots_s.push(sum / (j - i) as f64);
            i = j;
        }
        if knots_v.len() < 2 {
            return Err(GeostatError::InvalidParameter(
                "all values are identical; cannot build a normal score transform".into(),
            ));
        }
        Ok(Self { knots_v, knots_s })
    }

    /// Maps a data value to its normal score.
    pub fn transform(&self, v: f64) -> f64 {
        interp(&self.knots_v, &self.knots_s, v)
    }

    /// Maps a normal score back to data units (clamped to the data range).
    pub fn back_transform(&self, s: f64) -> f64 {
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
