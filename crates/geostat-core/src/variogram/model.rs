//! Theoretical variogram models.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{GeostatError, Result};

/// Supported variogram model families.
///
/// The Matérn variants use the Rasmussen & Williams parameterization
/// (correlation `(1 + √3 h/a) exp(-√3 h/a)` for ν = 3/2, etc.), so `range`
/// is comparable across families. Note that Matérn with ν = 1/2 is exactly
/// the exponential model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    /// Spherical model; reaches the sill exactly at `range`.
    Spherical,
    /// Exponential model; practical range ≈ 3 × `range`.
    Exponential,
    /// Gaussian model; parabolic near the origin, very smooth fields.
    Gaussian,
    /// Matérn with smoothness ν = 3/2.
    Matern15,
    /// Matérn with smoothness ν = 5/2.
    Matern25,
}

impl ModelKind {
    /// All supported kinds, for "fit the best model" workflows.
    pub const ALL: [ModelKind; 5] = [
        ModelKind::Spherical,
        ModelKind::Exponential,
        ModelKind::Gaussian,
        ModelKind::Matern15,
        ModelKind::Matern25,
    ];

    /// Normalized variogram (unit sill) at lag `h` for range parameter `a`.
    fn g(self, h: f64, a: f64) -> f64 {
        let d = h / a;
        match self {
            ModelKind::Spherical => {
                if d >= 1.0 {
                    1.0
                } else {
                    1.5 * d - 0.5 * d * d * d
                }
            }
            ModelKind::Exponential => 1.0 - (-d).exp(),
            ModelKind::Gaussian => 1.0 - (-(d * d)).exp(),
            ModelKind::Matern15 => {
                let s = 3.0_f64.sqrt() * d;
                1.0 - (1.0 + s) * (-s).exp()
            }
            ModelKind::Matern25 => {
                let s = 5.0_f64.sqrt() * d;
                1.0 - (1.0 + s + s * s / 3.0) * (-s).exp()
            }
        }
    }

    /// Short GSLIB-style abbreviation.
    pub fn abbrev(self) -> &'static str {
        match self {
            ModelKind::Spherical => "Sph",
            ModelKind::Exponential => "Exp",
            ModelKind::Gaussian => "Gau",
            ModelKind::Matern15 => "Mat1.5",
            ModelKind::Matern25 => "Mat2.5",
        }
    }
}

impl std::str::FromStr for ModelKind {
    type Err = GeostatError;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "sph" | "spherical" => Ok(ModelKind::Spherical),
            "exp" | "exponential" => Ok(ModelKind::Exponential),
            "gau" | "gaussian" => Ok(ModelKind::Gaussian),
            "mat15" | "matern15" => Ok(ModelKind::Matern15),
            "mat25" | "matern25" => Ok(ModelKind::Matern25),
            other => Err(GeostatError::InvalidParameter(format!(
                "unknown model kind '{other}' (expected spherical, exponential, gaussian, matern15 or matern25)"
            ))),
        }
    }
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.abbrev())
    }
}

/// One nested structure: a model family with its partial sill and range.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Structure {
    /// Model family.
    pub kind: ModelKind,
    /// Partial sill (variance contribution of this structure).
    pub sill: f64,
    /// Range parameter, in coordinate units.
    pub range: f64,
}

impl Structure {
    /// Convenience constructor.
    pub fn new(kind: ModelKind, sill: f64, range: f64) -> Self {
        Self { kind, sill, range }
    }
}

/// A nested variogram model: nugget plus one or more structures.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariogramModel {
    /// Nugget effect (discontinuity at the origin).
    pub nugget: f64,
    /// Nested structures, each with partial sill and range.
    pub structures: Vec<Structure>,
}

impl VariogramModel {
    /// Builds and validates a model: nugget and sills must be non-negative,
    /// ranges positive, and the total sill positive.
    pub fn new(nugget: f64, structures: Vec<Structure>) -> Result<Self> {
        if !(nugget >= 0.0) || !nugget.is_finite() {
            return Err(GeostatError::InvalidParameter(format!(
                "nugget must be finite and >= 0, got {nugget}"
            )));
        }
        for s in &structures {
            if !(s.sill >= 0.0) || !s.sill.is_finite() {
                return Err(GeostatError::InvalidParameter(format!(
                    "partial sill must be finite and >= 0, got {}",
                    s.sill
                )));
            }
            if !(s.range > 0.0) || !s.range.is_finite() {
                return Err(GeostatError::InvalidParameter(format!(
                    "range must be finite and > 0, got {}",
                    s.range
                )));
            }
        }
        let model = Self { nugget, structures };
        if !(model.total_sill() > 0.0) {
            return Err(GeostatError::InvalidParameter(
                "total sill must be positive".into(),
            ));
        }
        Ok(model)
    }

    /// Semivariance at lag `h` (`gamma(0) = 0` by convention).
    pub fn gamma(&self, h: f64) -> f64 {
        if h <= 0.0 {
            return 0.0;
        }
        self.nugget
            + self
                .structures
                .iter()
                .map(|s| s.sill * s.kind.g(h, s.range))
                .sum::<f64>()
    }

    /// Total sill: nugget plus all partial sills.
    pub fn total_sill(&self) -> f64 {
        self.nugget + self.structures.iter().map(|s| s.sill).sum::<f64>()
    }

    /// Covariance at lag `h` under second-order stationarity:
    /// `C(h) = total_sill - gamma(h)`, with `C(0) = total_sill`.
    pub fn covariance(&self, h: f64) -> f64 {
        self.total_sill() - self.gamma(h)
    }
}

impl fmt::Display for VariogramModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4} Nug", self.nugget)?;
        for s in &self.structures {
            write!(f, " + {:.4} {}({:.1})", s.sill, s.kind, s.range)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sph() -> VariogramModel {
        VariogramModel::new(0.1, vec![Structure::new(ModelKind::Spherical, 0.9, 100.0)]).unwrap()
    }

    #[test]
    fn gamma_boundary_behavior() {
        let m = sph();
        assert_eq!(m.gamma(0.0), 0.0);
        // Just past the origin: nugget shows up.
        assert!(m.gamma(1e-9) >= 0.1);
        // At and beyond the range: total sill.
        assert!((m.gamma(100.0) - 1.0).abs() < 1e-12);
        assert!((m.gamma(500.0) - 1.0).abs() < 1e-12);
        assert!((m.total_sill() - 1.0).abs() < 1e-12);
        // Covariance complements.
        assert!((m.covariance(0.0) - 1.0).abs() < 1e-12);
        assert!(m.covariance(500.0).abs() < 1e-12);
    }

    #[test]
    fn all_models_bounded_and_monotone_to_sill() {
        for kind in ModelKind::ALL {
            let m = VariogramModel::new(0.0, vec![Structure::new(kind, 2.0, 50.0)]).unwrap();
            let mut prev = 0.0;
            for i in 1..=100 {
                let g = m.gamma(i as f64 * 5.0);
                assert!(g >= prev - 1e-12, "{kind}: non-monotone at {i}");
                assert!(g <= 2.0 + 1e-12, "{kind}: exceeds sill");
                prev = g;
            }
            // Far field reaches (practically) the sill.
            assert!(m.gamma(5000.0) > 1.99, "{kind}");
        }
    }

    #[test]
    fn validation_rejects_bad_params() {
        assert!(VariogramModel::new(-0.1, vec![]).is_err());
        assert!(VariogramModel::new(0.0, vec![]).is_err()); // zero total sill
        assert!(
            VariogramModel::new(0.0, vec![Structure::new(ModelKind::Spherical, 1.0, 0.0)]).is_err()
        );
        assert!(
            VariogramModel::new(0.0, vec![Structure::new(ModelKind::Spherical, -1.0, 10.0)])
                .is_err()
        );
        // Pure nugget is valid.
        assert!(VariogramModel::new(1.0, vec![]).is_ok());
    }

    #[test]
    fn parses_kind_names() {
        assert_eq!("sph".parse::<ModelKind>().unwrap(), ModelKind::Spherical);
        assert_eq!(
            "Matern15".parse::<ModelKind>().unwrap(),
            ModelKind::Matern15
        );
        assert!("foo".parse::<ModelKind>().is_err());
    }
}
