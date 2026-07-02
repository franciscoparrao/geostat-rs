//! GSLIB-style distribution-tail extrapolation (`ltail`/`utail`).
//!
//! Sequential simulation and indicator kriging must extrapolate beyond the
//! last table knot or cutoff; truncating at the data extremes systematically
//! shrinks the simulated variance and the extreme quantiles. These are the
//! GSLIB interpolation options (Deutsch & Journel 1998, §V.1.6): linear or
//! power in the cdf between the extreme and a bound, plus a hyperbolic
//! (Pareto) model for the upper tail.

use crate::error::{GeostatError, Result};

/// Tail interpolation model (GSLIB `ltail`/`utail`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TailModel {
    /// No extrapolation: clamp at the extreme table value. Only meaningful
    /// for the normal-score back-transform (the pre-v0.7 behavior).
    #[default]
    None,
    /// Linear interpolation in the cdf to the bound (GSLIB option 1).
    Linear,
    /// Power interpolation with exponent `omega > 0` (GSLIB option 2):
    /// `z = a + (b - a) t^(1/omega)`. `Power(1.0)` equals `Linear`.
    Power(f64),
    /// Hyperbolic (Pareto) upper tail with exponent `omega > 0`
    /// (GSLIB option 4): `z = z_k ((1-F_k)/(1-F))^(1/omega)`, capped at the
    /// upper bound. Upper tail only; requires a positive last cutoff/knot.
    Hyperbolic(f64),
}

impl TailModel {
    /// Validates the model for use as a lower tail.
    pub(crate) fn validate_lower(self) -> Result<()> {
        match self {
            TailModel::Hyperbolic(_) => Err(GeostatError::InvalidParameter(
                "hyperbolic tails apply to the upper tail only".into(),
            )),
            TailModel::Power(w) if !(w > 0.0) => Err(GeostatError::InvalidParameter(format!(
                "power tail exponent must be positive, got {w}"
            ))),
            _ => Ok(()),
        }
    }

    /// Validates the model for use as an upper tail.
    pub(crate) fn validate_upper(self) -> Result<()> {
        match self {
            TailModel::Power(w) | TailModel::Hyperbolic(w) if !(w > 0.0) => Err(
                GeostatError::InvalidParameter(format!("tail exponent must be positive, got {w}")),
            ),
            _ => Ok(()),
        }
    }
}

/// Parses a tail spec: `none`, `linear`, `power:<omega>` or `hyper:<omega>`.
impl std::str::FromStr for TailModel {
    type Err = GeostatError;

    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();
        let (name, par) = match s.split_once(':') {
            Some((n, p)) => (n.trim(), Some(p.trim())),
            None => (s, None),
        };
        let omega = |def: f64| -> Result<f64> {
            match par {
                None => Ok(def),
                Some(p) => p.parse::<f64>().map_err(|_| {
                    GeostatError::InvalidParameter(format!("invalid tail exponent '{p}'"))
                }),
            }
        };
        match name.to_ascii_lowercase().as_str() {
            "none" => Ok(TailModel::None),
            "linear" => Ok(TailModel::Linear),
            "power" => Ok(TailModel::Power(omega(2.0)?)),
            "hyper" | "hyperbolic" => Ok(TailModel::Hyperbolic(omega(1.5)?)),
            other => Err(GeostatError::InvalidParameter(format!(
                "unknown tail model '{other}' (expected none, linear, power[:w] or hyper[:w])"
            ))),
        }
    }
}

/// Power-interval draw `a + (b - a) t^(1/omega)` for `t` in `[0, 1]`
/// (Linear is `omega = 1`).
fn power_draw(a: f64, b: f64, omega: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0).powf(1.0 / omega)
}

/// `(E[Z], E[Z^2])` of the power-interval distribution on `[a, b]`.
fn power_moments(a: f64, b: f64, omega: f64) -> (f64, f64) {
    let d = b - a;
    let m1 = omega / (omega + 1.0);
    let m2 = omega / (omega + 2.0);
    (a + d * m1, a * a + 2.0 * a * d * m1 + d * d * m2)
}

/// Draws from the lower-tail class between `zmin` and the first knot `z1`,
/// where `t` in `[0, 1]` is the relative cdf position within the class.
pub(crate) fn draw_lower(model: TailModel, zmin: f64, z1: f64, t: f64) -> f64 {
    match model {
        TailModel::None => z1,
        TailModel::Linear => power_draw(zmin, z1, 1.0, t),
        TailModel::Power(w) => power_draw(zmin, z1, w, t),
        TailModel::Hyperbolic(_) => unreachable!("rejected by validate_lower"),
    }
}

/// Draws from the upper-tail class between the last knot `zk` and `zmax`,
/// where `t` in `[0, 1]` is the relative cdf position within the class.
pub(crate) fn draw_upper(model: TailModel, zk: f64, zmax: f64, t: f64) -> f64 {
    match model {
        TailModel::None => zk,
        TailModel::Linear => power_draw(zk, zmax, 1.0, t),
        TailModel::Power(w) => power_draw(zk, zmax, w, t),
        TailModel::Hyperbolic(w) => {
            // t is the cdf position within the class; u = 1 - t is the
            // Pareto survival fraction.
            let u = (1.0 - t).max(f64::MIN_POSITIVE);
            (zk * u.powf(-1.0 / w)).min(zmax)
        }
    }
}

/// `(E[Z], E[Z^2])` of the lower-tail class distribution on `[zmin, z1]`.
pub(crate) fn lower_moments(model: TailModel, zmin: f64, z1: f64) -> (f64, f64) {
    match model {
        TailModel::None => (z1, z1 * z1),
        TailModel::Linear => power_moments(zmin, z1, 1.0),
        TailModel::Power(w) => power_moments(zmin, z1, w),
        TailModel::Hyperbolic(_) => unreachable!("rejected by validate_lower"),
    }
}

/// `(E[Z], E[Z^2])` of the upper-tail class distribution between `zk` and
/// `zmax` (hyperbolic capped at `zmax`).
pub(crate) fn upper_moments(model: TailModel, zk: f64, zmax: f64) -> (f64, f64) {
    match model {
        TailModel::None => (zk, zk * zk),
        TailModel::Linear => power_moments(zk, zmax, 1.0),
        TailModel::Power(w) => power_moments(zk, zmax, w),
        TailModel::Hyperbolic(w) => {
            // z(u) = zk u^(-1/w) for the survival fraction u ~ U(0, 1],
            // capped at zmax; the cap is active for u < u0.
            let u0 = (zk / zmax).powf(w).clamp(0.0, 1.0);
            let mean = zmax * u0
                + if (w - 1.0).abs() < 1e-12 {
                    zk * (1.0 / u0).ln()
                } else {
                    zk * (w / (w - 1.0)) * (1.0 - u0.powf((w - 1.0) / w))
                };
            let m2 = zmax * zmax * u0
                + if (w - 2.0).abs() < 1e-12 {
                    zk * zk * (1.0 / u0).ln()
                } else {
                    zk * zk * (w / (w - 2.0)) * (1.0 - u0.powf((w - 2.0) / w))
                };
            (mean, m2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_specs() {
        assert_eq!("linear".parse::<TailModel>().unwrap(), TailModel::Linear);
        assert_eq!("none".parse::<TailModel>().unwrap(), TailModel::None);
        assert_eq!(
            "power:1.5".parse::<TailModel>().unwrap(),
            TailModel::Power(1.5)
        );
        assert_eq!(
            "hyper:2".parse::<TailModel>().unwrap(),
            TailModel::Hyperbolic(2.0)
        );
        assert_eq!(
            "hyper".parse::<TailModel>().unwrap(),
            TailModel::Hyperbolic(1.5)
        );
        assert!("pareto".parse::<TailModel>().is_err());
        assert!("power:x".parse::<TailModel>().is_err());
    }

    #[test]
    fn power_one_is_linear() {
        for t in [0.0, 0.3, 0.7, 1.0] {
            assert!(
                (draw_lower(TailModel::Power(1.0), 2.0, 5.0, t)
                    - draw_lower(TailModel::Linear, 2.0, 5.0, t))
                .abs()
                    < 1e-14
            );
        }
        let (m1a, m2a) = upper_moments(TailModel::Power(1.0), 3.0, 9.0);
        let (m1b, m2b) = upper_moments(TailModel::Linear, 3.0, 9.0);
        assert!((m1a - m1b).abs() < 1e-12 && (m2a - m2b).abs() < 1e-12);
        // Linear moments equal the uniform on [a, b].
        assert!((m1b - 6.0).abs() < 1e-12);
        assert!((m2b - (9.0 + 27.0 + 81.0) / 3.0).abs() < 1e-12);
    }

    #[test]
    fn hyperbolic_upper_decays_and_caps() {
        let m = TailModel::Hyperbolic(1.5);
        // Monotone in t, capped at zmax.
        let z1 = draw_upper(m, 4.0, 100.0, 0.5);
        let z2 = draw_upper(m, 4.0, 100.0, 0.99);
        assert!(z1 > 4.0 && z2 > z1 && z2 <= 100.0);
        assert!(draw_upper(m, 4.0, 100.0, 1.0 - 1e-16) <= 100.0);
        // Moments: numerical integral cross-check.
        let (mean, m2) = upper_moments(m, 4.0, 100.0);
        let n = 200_000;
        let (mut s1, mut s2) = (0.0, 0.0);
        for i in 0..n {
            let t = (i as f64 + 0.5) / n as f64;
            let z = draw_upper(m, 4.0, 100.0, t);
            s1 += z;
            s2 += z * z;
        }
        s1 /= n as f64;
        s2 /= n as f64;
        assert!((mean - s1).abs() < 1e-3 * s1, "{mean} vs {s1}");
        assert!((m2 - s2).abs() < 5e-3 * s2, "{m2} vs {s2}");
    }

    #[test]
    fn moments_match_numerical_integrals_for_power() {
        for w in [0.5, 1.0, 2.5] {
            let (mean, m2) = lower_moments(TailModel::Power(w), 1.0, 3.0);
            let n = 100_000;
            let (mut s1, mut s2) = (0.0, 0.0);
            for i in 0..n {
                let t = (i as f64 + 0.5) / n as f64;
                let z = draw_lower(TailModel::Power(w), 1.0, 3.0, t);
                s1 += z;
                s2 += z * z;
            }
            s1 /= n as f64;
            s2 /= n as f64;
            assert!((mean - s1).abs() < 1e-4, "w={w}: {mean} vs {s1}");
            assert!((m2 - s2).abs() < 1e-3, "w={w}: {m2} vs {s2}");
        }
    }
}
