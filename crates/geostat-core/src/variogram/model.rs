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

    /// Parses a comma-separated list of model names, or the shortcuts
    /// `"best"`/`"all"` for [`ModelKind::ALL`] (the spec every front-end
    /// accepts for auto-fitting).
    pub fn parse_list(spec: &str) -> Result<Vec<ModelKind>> {
        let spec = spec.trim();
        if spec.eq_ignore_ascii_case("best") || spec.eq_ignore_ascii_case("all") {
            return Ok(Self::ALL.to_vec());
        }
        spec.split(',').map(|s| s.parse()).collect()
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
fn one() -> f64 {
    1.0
}

fn is_one(v: &f64) -> bool {
    *v == 1.0
}

/// Geometric (range) anisotropy, gstat/GSLIB convention: `azimuth_deg` is
/// the direction of the *major* axis in degrees clockwise from north, and
/// `ratio = minor_range / major_range` in `(0, 1]`. The structure's `range`
/// is the major-axis range.
///
/// In 3-D, `ratio_z` is the vertical/major range ratio (gstat's second
/// anisotropy ratio with zero dip/rake rotations); it is ignored in 2-D.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Anisotropy {
    /// Major-axis direction, degrees clockwise from north.
    pub azimuth_deg: f64,
    /// Minor/major range ratio in `(0, 1]` (horizontal in 3-D).
    pub ratio: f64,
    /// Vertical/major range ratio in `(0, 1]` (3-D only; default 1).
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub ratio_z: f64,
}

/// One nested structure: a model family with its partial sill, range and
/// optional geometric anisotropy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Structure {
    /// Model family.
    pub kind: ModelKind,
    /// Partial sill (variance contribution of this structure).
    pub sill: f64,
    /// Range parameter, in coordinate units (major-axis range if anisotropic).
    pub range: f64,
    /// Optional geometric anisotropy (isotropic when `None`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anis: Option<Anisotropy>,
}

impl Structure {
    /// Isotropic structure.
    pub fn new(kind: ModelKind, sill: f64, range: f64) -> Self {
        Self {
            kind,
            sill,
            range,
            anis: None,
        }
    }

    /// Structure with geometric anisotropy.
    pub fn with_anisotropy(
        kind: ModelKind,
        sill: f64,
        range: f64,
        azimuth_deg: f64,
        ratio: f64,
    ) -> Self {
        Self {
            kind,
            sill,
            range,
            anis: Some(Anisotropy {
                azimuth_deg,
                ratio,
                ratio_z: 1.0,
            }),
        }
    }

    /// Effective isotropic-equivalent lag distance for a separation vector:
    /// the horizontal components are rotated into the (major, minor) frame
    /// and the minor (and, in 3-D, vertical) components stretched by the
    /// inverse ratios.
    fn effective_h<const D: usize>(&self, dh: [f64; D]) -> f64 {
        match self.anis {
            None => {
                let mut s = 0.0;
                for &v in &dh {
                    s += v * v;
                }
                s.sqrt()
            }
            Some(a) => {
                let (s, c) = a.azimuth_deg.to_radians().sin_cos();
                let h_major = dh[0] * s + dh[1] * c;
                let h_minor = (dh[0] * c - dh[1] * s) / a.ratio;
                let mut sum = h_major * h_major + h_minor * h_minor;
                if D == 3 {
                    let h_z = dh[2] / a.ratio_z;
                    sum += h_z * h_z;
                }
                sum.sqrt()
            }
        }
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
            if let Some(a) = s.anis {
                if !(a.ratio_z > 0.0) || a.ratio_z > 1.0 {
                    return Err(GeostatError::InvalidParameter(format!(
                        "anisotropy ratio_z must be in (0, 1], got {}",
                        a.ratio_z
                    )));
                }
                if !(a.ratio > 0.0) || a.ratio > 1.0 {
                    return Err(GeostatError::InvalidParameter(format!(
                        "anisotropy ratio must be in (0, 1], got {}",
                        a.ratio
                    )));
                }
                if !a.azimuth_deg.is_finite() {
                    return Err(GeostatError::InvalidParameter(
                        "anisotropy azimuth must be finite".into(),
                    ));
                }
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

    /// Semivariance at scalar lag `h` (`gamma(0) = 0` by convention).
    /// For anisotropic structures, `h` is interpreted as a distance along
    /// the major axis; use [`VariogramModel::gamma_dh`] for full vectors.
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

    /// Semivariance for a separation vector `dh`, honoring per-structure
    /// geometric anisotropy.
    pub fn gamma_dh<const D: usize>(&self, dh: [f64; D]) -> f64 {
        if dh.iter().all(|&v| v == 0.0) {
            return 0.0;
        }
        self.nugget
            + self
                .structures
                .iter()
                .map(|s| s.sill * s.kind.g(s.effective_h(dh), s.range))
                .sum::<f64>()
    }

    /// Total sill: nugget plus all partial sills.
    pub fn total_sill(&self) -> f64 {
        self.nugget + self.structures.iter().map(|s| s.sill).sum::<f64>()
    }

    /// Covariance at scalar lag `h` under second-order stationarity:
    /// `C(h) = total_sill - gamma(h)`, with `C(0) = total_sill`.
    pub fn covariance(&self, h: f64) -> f64 {
        self.total_sill() - self.gamma(h)
    }

    /// Covariance for a separation vector `dh`, honoring anisotropy.
    pub fn covariance_dh<const D: usize>(&self, dh: [f64; D]) -> f64 {
        self.total_sill() - self.gamma_dh(dh)
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
    fn anisotropy_stretches_minor_axis() {
        // Major axis N-S (azimuth 0), ratio 0.5: range 100 along N-S,
        // 50 along E-W.
        let m = VariogramModel::new(
            0.0,
            vec![Structure::with_anisotropy(
                ModelKind::Spherical,
                1.0,
                100.0,
                0.0,
                0.5,
            )],
        )
        .unwrap();
        // Along the major axis: same as isotropic with range 100.
        assert!((m.gamma_dh([0.0, 100.0]) - 1.0).abs() < 1e-12);
        assert!(m.gamma_dh([0.0, 50.0]) < 1.0 - 1e-6);
        // Along the minor axis the sill is reached at 50.
        assert!((m.gamma_dh([50.0, 0.0]) - 1.0).abs() < 1e-12);
        // Same lag distance hits harder across the minor axis.
        assert!(m.gamma_dh([30.0, 0.0]) > m.gamma_dh([0.0, 30.0]));
        // Isotropic structures: gamma_dh agrees with scalar gamma.
        let iso = VariogramModel::new(0.1, vec![Structure::new(ModelKind::Exponential, 0.9, 40.0)])
            .unwrap();
        let h = (3.0_f64 * 3.0 + 4.0 * 4.0).sqrt();
        assert!((iso.gamma_dh([3.0, 4.0]) - iso.gamma(h)).abs() < 1e-12);
        // Invalid ratio rejected.
        assert!(
            VariogramModel::new(
                0.0,
                vec![Structure::with_anisotropy(
                    ModelKind::Spherical,
                    1.0,
                    100.0,
                    0.0,
                    1.5
                )]
            )
            .is_err()
        );
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
