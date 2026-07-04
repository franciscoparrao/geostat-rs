//! Theoretical variogram models.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::bessel;
use crate::error::{GeostatError, Result};

/// Supported variogram model families.
///
/// The Matérn variants use the Rasmussen & Williams parameterization
/// (correlation `(1 + √3 h/a) exp(-√3 h/a)` for ν = 3/2, etc.), so `range`
/// is comparable across families. Note that Matérn with ν = 1/2 is exactly
/// the exponential model.
///
/// [`ModelKind::Matern15`] and [`ModelKind::Matern25`] keep their closed-form
/// evaluation (cheaper, no quadrature); [`ModelKind::Matern`] covers any
/// other real `ν > 0` via the general correlation `(2^{1-ν}/Γ(ν)) (√(2ν) d)^ν
/// K_ν(√(2ν) d)` (`K_ν` the modified Bessel function of the second kind, see
/// [`super::bessel`]) and agrees with the closed forms at ν = 1.5/2.5 to
/// within quadrature error (~1e-9, see
/// `tests::matern_matches_closed_form_special_cases`).
///
/// **gstat interop**: gstat's `"Ste"` model (M. Stein's parameterization)
/// scales the Bessel argument by `2√ν` instead of `√(2ν)`, a *different but
/// equally valid* convention for what "range" means. The two `range`s are
/// related by a constant independent of `ν`: `range_here = range_gstat_Ste /
/// √2` (derived and cross-checked against `fit.variogram(..., vgm(..., "Ste",
/// ..., kappa=ν), fit.kappa=FALSE)` on Meuse in
/// `validation/matern_gstat.R`/`compare_matern.py`, parity ~5e-7). gstat's
/// older `"Mat"` model uses yet another convention (`h/range` with no
/// `ν`-dependent scaling at all) and is not the one to compare against.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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
    /// Matérn with an arbitrary real smoothness `ν > 0`.
    Matern(f64),
    /// Circular model (2-D disc indicator covariance); reaches the sill
    /// exactly at `range`, steeper near the origin than spherical.
    Circular,
    /// Stable (power-exponential) model with shape `α ∈ (0, 2]`:
    /// `γ(d) = 1 - exp(-d^α)`, `d = h/range`. Generalizes exponential
    /// (`α = 1`) and Gaussian (`α = 2`); matches gstat's `"Exc"` model
    /// exactly (no range-convention subtlety, unlike Matérn/Ste).
    Stable(f64),
    /// Hole-effect (cardinal-sine) model: `γ(d) = 1 - sin(d)/d`,
    /// `d = h/range`. Oscillates around the sill (bounded overshoot),
    /// modeling periodic/pseudo-periodic phenomena — **not** monotone, and
    /// deliberately excluded from [`ModelKind::ALL`]. Matches gstat's
    /// `"Hol"` model exactly.
    Hole,
    /// Wave (cardinal-sine) model: `γ(d) = 1 - sin(πd)/(πd)`,
    /// `d = h/range` — like [`ModelKind::Hole`] but scaled so the
    /// covariance's first zero-crossing sits exactly at `h = range`.
    /// Matches gstat's `"Wav"` model exactly.
    Wave,
    /// Power (intrinsic random function of order 0), unbounded:
    /// `γ(h) = h^θ`, `θ ∈ (0, 2)`; the structure's `sill` is the slope
    /// coefficient `c` (`γ(h) = c·h^θ`) and `range` is **ignored** (Power
    /// has no plateau, so no length-scale is needed — matches gstat's
    /// `"Pow"` exactly, which has the same `psill·h^range` convention with
    /// `range` doubling as the exponent). No covariance function exists for
    /// this model (infinite variance): [`VariogramModel::covariance_dh`]/
    /// [`VariogramModel::total_sill`] are meaningless on a model containing
    /// it. Only [`crate::kriging::Kriging`] with `Ordinary`/`Universal`/
    /// `ExternalDrift` supports it (kriged directly in variogram form, the
    /// classical IRF-0 generalization); every covariance-based path
    /// (`Simple` kriging, Vecchia, SIS, SGS, co-kriging) rejects it — see
    /// [`VariogramModel::has_power`].
    Power(f64),
}

impl ModelKind {
    /// All supported *bounded, monotone* kinds, for "fit the best model"
    /// workflows (auto-fit assumes a single well-defined sill/range guess,
    /// which does not make sense for the oscillating hole-effect models).
    pub const ALL: [ModelKind; 6] = [
        ModelKind::Spherical,
        ModelKind::Exponential,
        ModelKind::Gaussian,
        ModelKind::Matern15,
        ModelKind::Matern25,
        ModelKind::Circular,
    ];

    /// Normalized variogram (unit sill) at lag `h` for range parameter `a`.
    pub(crate) fn g(self, h: f64, a: f64) -> f64 {
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
            ModelKind::Matern(nu) => {
                if d <= 0.0 {
                    0.0
                } else {
                    let s = (2.0 * nu).sqrt() * d;
                    let corr =
                        2f64.powf(1.0 - nu) / bessel::gamma(nu) * s.powf(nu) * bessel::bessel_k(nu, s);
                    1.0 - corr
                }
            }
            ModelKind::Circular => {
                let d = d.min(1.0);
                1.0 - (2.0 / std::f64::consts::PI) * (d.acos() - d * (1.0 - d * d).sqrt())
            }
            ModelKind::Stable(alpha) => 1.0 - (-d.powf(alpha)).exp(),
            ModelKind::Hole => {
                if d <= 0.0 {
                    0.0
                } else {
                    1.0 - d.sin() / d
                }
            }
            ModelKind::Power(theta) => h.abs().powf(theta),
            ModelKind::Wave => {
                if d <= 0.0 {
                    0.0
                } else {
                    let pd = std::f64::consts::PI * d;
                    1.0 - pd.sin() / pd
                }
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
    pub fn abbrev(self) -> String {
        match self {
            ModelKind::Spherical => "Sph".to_string(),
            ModelKind::Exponential => "Exp".to_string(),
            ModelKind::Gaussian => "Gau".to_string(),
            ModelKind::Matern15 => "Mat1.5".to_string(),
            ModelKind::Matern25 => "Mat2.5".to_string(),
            ModelKind::Matern(nu) => format!("Mat(ν={nu:.3})"),
            ModelKind::Circular => "Cir".to_string(),
            ModelKind::Stable(alpha) => format!("Stable(α={alpha:.3})"),
            ModelKind::Hole => "Hol".to_string(),
            ModelKind::Wave => "Wav".to_string(),
            ModelKind::Power(theta) => format!("Pow(θ={theta:.3})"),
        }
    }
}

impl std::str::FromStr for ModelKind {
    type Err = GeostatError;

    fn from_str(s: &str) -> Result<Self> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "sph" | "spherical" => return Ok(ModelKind::Spherical),
            "exp" | "exponential" => return Ok(ModelKind::Exponential),
            "gau" | "gaussian" => return Ok(ModelKind::Gaussian),
            "mat15" | "matern15" => return Ok(ModelKind::Matern15),
            "mat25" | "matern25" => return Ok(ModelKind::Matern25),
            "cir" | "circular" => return Ok(ModelKind::Circular),
            "hol" | "hole" => return Ok(ModelKind::Hole),
            "wav" | "wave" => return Ok(ModelKind::Wave),
            _ => {}
        }
        // General Matern: "matern:<nu>" or "mat:<nu>" (e.g. "matern:1.2").
        if let Some(nu_str) = lower
            .strip_prefix("matern:")
            .or_else(|| lower.strip_prefix("mat:"))
        {
            let nu: f64 = nu_str.parse().map_err(|_| {
                GeostatError::InvalidParameter(format!("invalid Matern nu '{nu_str}'"))
            })?;
            if !(nu > 0.0) || !nu.is_finite() {
                return Err(GeostatError::InvalidParameter(format!(
                    "Matern nu must be finite and > 0, got {nu}"
                )));
            }
            return Ok(ModelKind::Matern(nu));
        }
        // Stable: "stable:<alpha>" (e.g. "stable:1.2"), alpha in (0, 2].
        if let Some(alpha_str) = lower.strip_prefix("stable:") {
            let alpha: f64 = alpha_str.parse().map_err(|_| {
                GeostatError::InvalidParameter(format!("invalid Stable alpha '{alpha_str}'"))
            })?;
            if !(alpha > 0.0) || alpha > 2.0 {
                return Err(GeostatError::InvalidParameter(format!(
                    "Stable alpha must be in (0, 2], got {alpha}"
                )));
            }
            return Ok(ModelKind::Stable(alpha));
        }
        // Power: "power:<theta>" (e.g. "power:1.2"), theta in (0, 2). Range
        // on the enclosing Structure is ignored (see ModelKind::Power docs).
        if let Some(theta_str) = lower.strip_prefix("power:") {
            let theta: f64 = theta_str.parse().map_err(|_| {
                GeostatError::InvalidParameter(format!("invalid Power theta '{theta_str}'"))
            })?;
            if !(theta > 0.0) || theta >= 2.0 {
                return Err(GeostatError::InvalidParameter(format!(
                    "Power theta must be in (0, 2), got {theta}"
                )));
            }
            return Ok(ModelKind::Power(theta));
        }
        Err(GeostatError::InvalidParameter(format!(
            "unknown model kind '{s}' (expected spherical, exponential, gaussian, matern15, \
             matern25, matern:<nu>, circular, hole, wave, stable:<alpha> or power:<theta>)"
        )))
    }
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.abbrev())
    }
}

/// One nested structure: a model family with its partial sill and range.
fn one() -> f64 {
    1.0
}

fn is_one(v: &f64) -> bool {
    *v == 1.0
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

/// Geometric (range) anisotropy, gstat/GSLIB convention: `azimuth_deg` is a
/// reference direction in degrees clockwise from north, `range` (on the
/// enclosing [`Structure`]) is that direction's range, and `ratio` scales
/// the orthogonal (`azimuth_deg + 90°`) direction's range relative to it.
///
/// `ratio ∈ (0, 1]` is the common case: `azimuth_deg` names the *major* axis
/// and `ratio = minor/major`. **Zonal anisotropy** — a *longer* range
/// orthogonal to `azimuth_deg` (e.g. much greater lateral than vertical
/// continuity) — is `ratio > 1`; no axis relabeling ("major" becomes
/// whichever axis ends up longer) is needed, `effective_h`'s rotate-then-
/// scale math is valid for any `ratio > 0` unchanged.
///
/// In 3-D, `ratio_z` plays the same role for the vertical axis; `dip_deg`
/// and `rake_deg` (both default 0, ignored in 2-D) tilt the whole
/// major/minor/vertical frame off the horizontal — full GSLIB `ang1`
/// (`azimuth_deg`) / `ang2` (`dip_deg`) / `ang3` (`rake_deg`) rotation
/// (Deutsch & Journel 1998), needed to match an experimental variogram
/// whose direction of maximum continuity plunges rather than staying
/// horizontal (the experimental variogram already accepts `dip_deg` via
/// [`super::DirectionConfig`]; before this, the fitted *model* could not
/// express it — see AUDIT-2026-07.md §3). Cross-checked against gstat's
/// `anis = c(ang1, ang2, ang3, anis1, anis2)` to machine precision for
/// azimuth-only, dip-only, rake-only and combined rotations (gstat's own
/// `vgm()` warns its third-angle code carries a known GSLIB quirk; this
/// matches gstat's actual behavior exactly, quirk included, since gstat
/// interop is the point — see `tests::rotation_3d_matches_gstat_setrot`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Anisotropy {
    /// Reference direction (GSLIB `ang1`), degrees clockwise from north.
    pub azimuth_deg: f64,
    /// `(azimuth_deg + 90°)` / `azimuth_deg` range ratio, `> 0` (horizontal
    /// in 3-D, GSLIB `anis1`). `≤ 1` for the common "major axis" case; `> 1`
    /// for zonal anisotropy (see the struct docs).
    pub ratio: f64,
    /// Vertical / `azimuth_deg` range ratio, `> 0` (3-D only, GSLIB
    /// `anis2`; default 1).
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub ratio_z: f64,
    /// Dip (GSLIB `ang2`): tilts the major axis off the horizontal, degrees
    /// (3-D only, ignored in 2-D; default 0).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub dip_deg: f64,
    /// Rake/plunge (GSLIB `ang3`): rotates the minor/vertical axes about
    /// the (already dipped) major axis, degrees (3-D only, ignored in 2-D;
    /// default 0).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub rake_deg: f64,
}

impl Anisotropy {
    /// GSLIB `setrot`-equivalent rotation+scaling matrix: applying it to a
    /// separation vector and taking the Euclidean norm of the result gives
    /// the isotropic-equivalent effective distance (before dividing by the
    /// structure's `range`). Only meaningful in 3-D; see the struct docs
    /// for the gstat cross-check.
    fn rotation_matrix_3d(&self) -> [[f64; 3]; 3] {
        let alpha = if (0.0..270.0).contains(&self.azimuth_deg) {
            (90.0 - self.azimuth_deg).to_radians()
        } else {
            (450.0 - self.azimuth_deg).to_radians()
        };
        let beta = (-self.dip_deg).to_radians();
        let theta = self.rake_deg.to_radians();
        let (sina, cosa) = alpha.sin_cos();
        let (sinb, cosb) = beta.sin_cos();
        let (sint, cost) = theta.sin_cos();
        let afac1 = 1.0 / self.ratio;
        let afac2 = 1.0 / self.ratio_z;
        [
            [cosb * cosa, cosb * sina, -sinb],
            [
                afac1 * (-cost * sina + sint * sinb * cosa),
                afac1 * (cost * cosa + sint * sinb * sina),
                afac1 * (sint * cosb),
            ],
            [
                afac2 * (sint * sina + cost * sinb * cosa),
                afac2 * (-sint * cosa + cost * sinb * sina),
                afac2 * (cost * cosb),
            ],
        ]
    }
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

    /// Structure with geometric anisotropy (azimuth + ratio only; no
    /// dip/rake — see [`Structure::with_rotation`] for full 3-D tilt).
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
                dip_deg: 0.0,
                rake_deg: 0.0,
            }),
        }
    }

    /// Structure with full 3-D geometric anisotropy (GSLIB `ang1`/`ang2`/
    /// `ang3` = `azimuth_deg`/`dip_deg`/`rake_deg`); see [`Anisotropy`].
    /// `dip_deg = rake_deg = 0.0` is exactly [`Structure::with_anisotropy`].
    #[allow(clippy::too_many_arguments)]
    pub fn with_rotation(
        kind: ModelKind,
        sill: f64,
        range: f64,
        azimuth_deg: f64,
        dip_deg: f64,
        rake_deg: f64,
        ratio: f64,
        ratio_z: f64,
    ) -> Self {
        Self {
            kind,
            sill,
            range,
            anis: Some(Anisotropy {
                azimuth_deg,
                ratio,
                ratio_z,
                dip_deg,
                rake_deg,
            }),
        }
    }

    /// Effective isotropic-equivalent lag distance for a separation vector.
    /// In 2-D, the components are rotated into the (major, minor) frame and
    /// the minor component stretched by the inverse ratio. In 3-D, the full
    /// GSLIB rotation+scaling matrix ([`Anisotropy::rotation_matrix_3d`]) is
    /// applied; with `dip_deg = rake_deg = 0` this reduces *exactly* to the
    /// 2-D case's rotate-then-scale math extended with `dh[2]/ratio_z`
    /// (verified analytically: the matrix's rows collapse to the same
    /// trigonometric expressions).
    pub(crate) fn effective_h<const D: usize>(&self, dh: [f64; D]) -> f64 {
        match self.anis {
            None => {
                let mut s = 0.0;
                for &v in &dh {
                    s += v * v;
                }
                s.sqrt()
            }
            Some(a) => {
                if D == 3 {
                    let (h0, h1, h2) = (dh[0], dh[1], dh[2]);
                    let rot = a.rotation_matrix_3d();
                    let e0 = rot[0][0] * h0 + rot[0][1] * h1 + rot[0][2] * h2;
                    let e1 = rot[1][0] * h0 + rot[1][1] * h1 + rot[1][2] * h2;
                    let e2 = rot[2][0] * h0 + rot[2][1] * h1 + rot[2][2] * h2;
                    (e0 * e0 + e1 * e1 + e2 * e2).sqrt()
                } else {
                    let (s, c) = a.azimuth_deg.to_radians().sin_cos();
                    let h_major = dh[0] * s + dh[1] * c;
                    let h_minor = (dh[0] * c - dh[1] * s) / a.ratio;
                    (h_major * h_major + h_minor * h_minor).sqrt()
                }
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
            if let ModelKind::Matern(nu) = s.kind
                && (!(nu > 0.0) || !nu.is_finite()) {
                    return Err(GeostatError::InvalidParameter(format!(
                        "Matern nu must be finite and > 0, got {nu}"
                    )));
                }
            if let ModelKind::Stable(alpha) = s.kind
                && (!(alpha > 0.0) || alpha > 2.0) {
                    return Err(GeostatError::InvalidParameter(format!(
                        "Stable alpha must be in (0, 2], got {alpha}"
                    )));
                }
            if let ModelKind::Power(theta) = s.kind
                && (!(theta > 0.0) || theta >= 2.0) {
                    return Err(GeostatError::InvalidParameter(format!(
                        "Power theta must be in (0, 2), got {theta}"
                    )));
                }
            if let Some(a) = s.anis {
                // ratio > 1 is valid (zonal anisotropy: the orthogonal axis
                // is longer than the labeled one; see `Anisotropy` docs) --
                // only non-finite/non-positive values are rejected.
                if !(a.ratio_z > 0.0) || !a.ratio_z.is_finite() {
                    return Err(GeostatError::InvalidParameter(format!(
                        "anisotropy ratio_z must be finite and > 0, got {}",
                        a.ratio_z
                    )));
                }
                if !(a.ratio > 0.0) || !a.ratio.is_finite() {
                    return Err(GeostatError::InvalidParameter(format!(
                        "anisotropy ratio must be finite and > 0, got {}",
                        a.ratio
                    )));
                }
                if !a.azimuth_deg.is_finite() {
                    return Err(GeostatError::InvalidParameter(
                        "anisotropy azimuth must be finite".into(),
                    ));
                }
                if !a.dip_deg.is_finite() || !a.rake_deg.is_finite() {
                    return Err(GeostatError::InvalidParameter(
                        "anisotropy dip/rake must be finite".into(),
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
    ///
    /// **Meaningless if [`VariogramModel::has_power`]**: a `Power`
    /// structure has no plateau (infinite variance), so this sums a
    /// slope coefficient as if it were a true sill. Every covariance-based
    /// code path (this method, [`VariogramModel::covariance`]/
    /// [`VariogramModel::covariance_dh`], and everything built on them —
    /// simple kriging, Vecchia, SIS, SGS, co-kriging) rejects models
    /// containing `Power` at their public entry points instead of silently
    /// computing a wrong number here.
    pub fn total_sill(&self) -> f64 {
        self.nugget + self.structures.iter().map(|s| s.sill).sum::<f64>()
    }

    /// `true` if any structure is [`ModelKind::Power`] (no covariance
    /// function exists for this model — see [`VariogramModel::total_sill`]).
    pub fn has_power(&self) -> bool {
        self.structures
            .iter()
            .any(|s| matches!(s.kind, ModelKind::Power(_)))
    }

    /// Covariance at scalar lag `h` under second-order stationarity:
    /// `C(h) = total_sill - gamma(h)`, with `C(0) = total_sill`. See the
    /// [`VariogramModel::has_power`] caveat.
    pub fn covariance(&self, h: f64) -> f64 {
        self.total_sill() - self.gamma(h)
    }

    /// Covariance for a separation vector `dh`, honoring anisotropy. See the
    /// [`VariogramModel::has_power`] caveat.
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
        // Invalid ratio rejected (non-positive/non-finite; ratio > 1 is now
        // valid -- zonal anisotropy, see `zonal_anisotropy_ratio_above_one`).
        for bad_ratio in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                VariogramModel::new(
                    0.0,
                    vec![Structure::with_anisotropy(
                        ModelKind::Spherical,
                        1.0,
                        100.0,
                        0.0,
                        bad_ratio
                    )]
                )
                .is_err(),
                "ratio {bad_ratio} should be rejected"
            );
        }
    }

    #[test]
    fn zonal_anisotropy_ratio_above_one() {
        // azimuth 0 (N-S) labeled range 50; ratio 2 means the orthogonal
        // (E-W) axis has *twice* the range (100) -- no axis relabeling
        // "trick" needed, unlike the old `ratio <= 1` restriction.
        let m = VariogramModel::new(
            0.0,
            vec![Structure::with_anisotropy(
                ModelKind::Spherical,
                1.0,
                50.0,
                0.0,
                2.0,
            )],
        )
        .unwrap();
        // Along the labeled (N-S) axis: sill reached at 50.
        assert!((m.gamma_dh([0.0, 50.0]) - 1.0).abs() < 1e-12);
        // Along the orthogonal (E-W) axis: sill reached at 100 (2x).
        assert!((m.gamma_dh([100.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!(m.gamma_dh([50.0, 0.0]) < 1.0 - 1e-6);
        // Same lag distance hits *less* hard across the now-longer axis.
        assert!(m.gamma_dh([30.0, 0.0]) < m.gamma_dh([0.0, 30.0]));
        // This is exactly the mirror image of ratio=0.5 with a 90-degree
        // azimuth shift (major/minor swapped).
        let mirrored = VariogramModel::new(
            0.0,
            vec![Structure::with_anisotropy(
                ModelKind::Spherical,
                1.0,
                100.0,
                90.0,
                0.5,
            )],
        )
        .unwrap();
        for &(x, y) in &[(0.0, 50.0), (100.0, 0.0), (30.0, 0.0), (0.0, 30.0), (17.0, -42.0)] {
            assert!(
                (m.gamma_dh([x, y]) - mirrored.gamma_dh([x, y])).abs() < 1e-12,
                "({x},{y}): {} vs {}",
                m.gamma_dh([x, y]),
                mirrored.gamma_dh([x, y])
            );
        }
    }

    /// Ground truth from gstat's `variogramLine` with `anis = c(30, 20, 40,
    /// 0.5, 0.3)` (azimuth 30, dip 20, rake 40, ratio 0.5, ratio_z 0.3),
    /// `h = 20`, range 100, spherical -- the azimuth-only/dip-only/rake-only
    /// special cases were also cross-checked (residual ~1e-16) while
    /// deriving `rotation_matrix_3d`; see the session notes.
    #[test]
    fn rotation_3d_matches_gstat_setrot() {
        let m = VariogramModel::new(
            0.0,
            vec![Structure::with_rotation(
                ModelKind::Spherical,
                1.0,
                100.0,
                30.0,
                20.0,
                40.0,
                0.5,
                0.3,
            )],
        )
        .unwrap();
        let cases: [([f64; 3], f64); 6] = [
            ([1.0, 0.0, 0.0], 0.605458280770632),
            ([0.0, 1.0, 0.0], 0.577390658041395),
            ([0.0, 0.0, 1.0], 0.732990706603906),
            (
                [0.577350269190, 0.577350269190, 0.577350269190],
                0.432164835884142,
            ),
            (
                [0.666666666667, -0.666666666667, 0.333333333333],
                0.810043758779196,
            ),
            (
                [-0.534522483825, 0.267261241912, 0.801783725737],
                0.575462768274161,
            ),
        ];
        for (dir, expected) in cases {
            let dh = [dir[0] * 20.0, dir[1] * 20.0, dir[2] * 20.0];
            let got = m.gamma_dh(dh);
            assert!(
                (got - expected).abs() < 1e-9,
                "dir={dir:?}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn rotation_3d_reduces_to_2d_style_when_dip_and_rake_are_zero() {
        // dip=rake=0 must reproduce `with_anisotropy` (2-D-style formula
        // extended with dh[2]/ratio_z) to within floating-point roundoff --
        // proven analytically (see `effective_h` docs), checked numerically.
        let full = VariogramModel::new(
            0.05,
            vec![Structure::with_rotation(
                ModelKind::Exponential,
                0.9,
                80.0,
                55.0,
                0.0,
                0.0,
                0.4,
                0.6,
            )],
        )
        .unwrap();
        // Direct cross-check against the OLD (pre-rotation) closed-form
        // formula, replicated here so a regression in `rotation_matrix_3d`
        // would be caught even though `with_anisotropy` no longer builds a
        // 3-D structure directly.
        let az = 55.0_f64.to_radians();
        let (s, c) = az.sin_cos();
        for &(x, y, z) in &[(10.0, 20.0, 5.0), (-30.0, 15.0, -8.0), (40.0, -25.0, 0.0)] {
            let h_major = x * s + y * c;
            let h_minor = (x * c - y * s) / 0.4;
            let h_z = z / 0.6;
            let old_eff = (h_major * h_major + h_minor * h_minor + h_z * h_z).sqrt();
            let old_gamma = 0.05 + 0.9 * (1.0 - (-old_eff / 80.0).exp());
            let new_gamma = full.gamma_dh([x, y, z]);
            assert!(
                (old_gamma - new_gamma).abs() < 1e-12,
                "({x},{y},{z}): old {old_gamma} vs new {new_gamma}"
            );
        }
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

    #[test]
    fn parses_matern_nu_spec() {
        assert_eq!(
            "matern:1.2".parse::<ModelKind>().unwrap(),
            ModelKind::Matern(1.2)
        );
        assert_eq!(
            "MAT:0.75".parse::<ModelKind>().unwrap(),
            ModelKind::Matern(0.75)
        );
        assert!("matern:0".parse::<ModelKind>().is_err());
        assert!("matern:-1".parse::<ModelKind>().is_err());
        assert!("matern:nope".parse::<ModelKind>().is_err());
    }

    #[test]
    fn parses_new_family_names() {
        assert_eq!("cir".parse::<ModelKind>().unwrap(), ModelKind::Circular);
        assert_eq!(
            "circular".parse::<ModelKind>().unwrap(),
            ModelKind::Circular
        );
        assert_eq!("hol".parse::<ModelKind>().unwrap(), ModelKind::Hole);
        assert_eq!("wave".parse::<ModelKind>().unwrap(), ModelKind::Wave);
        assert_eq!(
            "stable:1.7".parse::<ModelKind>().unwrap(),
            ModelKind::Stable(1.7)
        );
        assert!("stable:0".parse::<ModelKind>().is_err());
        assert!("stable:2.1".parse::<ModelKind>().is_err());
    }

    /// The continuous-ν Matern (evaluated via the Bessel-K quadrature) must
    /// agree with the hardcoded closed forms at ν = 1.5, 2.5, and with the
    /// exponential model at ν = 0.5 (all classical special cases of Matern).
    #[test]
    fn matern_matches_closed_form_special_cases() {
        // At very small d, gamma itself is tiny (~1e-6 at d=0.001), so a
        // pure relative-error check is dominated by the quadrature's
        // absolute error floor; accept either a tight relative match or a
        // tiny absolute one.
        let close_enough = |a: f64, b: f64| -> bool { (a - b).abs() < 1e-9 || (a - b).abs() / b.max(1e-12) < 1e-6 };
        let d_values = [0.001, 0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0];
        for &d in &d_values {
            let general15 = ModelKind::Matern(1.5).g(d, 1.0);
            let closed15 = ModelKind::Matern15.g(d, 1.0);
            assert!(
                close_enough(general15, closed15),
                "nu=1.5, d={d}: {general15} vs {closed15}"
            );

            let general25 = ModelKind::Matern(2.5).g(d, 1.0);
            let closed25 = ModelKind::Matern25.g(d, 1.0);
            assert!(
                close_enough(general25, closed25),
                "nu=2.5, d={d}: {general25} vs {closed25}"
            );

            let general05 = ModelKind::Matern(0.5).g(d, 1.0);
            let exp = ModelKind::Exponential.g(d, 1.0);
            assert!(
                close_enough(general05, exp),
                "nu=0.5, d={d}: {general05} vs {exp}"
            );
        }
    }

    #[test]
    fn matern_continuous_nu_bounded_and_monotone() {
        for &nu in &[0.15, 0.3, 0.75, 1.0, 1.8, 3.0, 4.5] {
            let m = VariogramModel::new(0.0, vec![Structure::new(ModelKind::Matern(nu), 2.0, 50.0)])
                .unwrap();
            let mut prev = 0.0;
            for i in 1..=100 {
                let g = m.gamma(i as f64 * 5.0);
                assert!(g >= prev - 1e-9, "nu={nu}: non-monotone at {i}");
                assert!(g <= 2.0 + 1e-9, "nu={nu}: exceeds sill");
                prev = g;
            }
            assert!(m.gamma(5000.0) > 1.99, "nu={nu}");
            assert_eq!(m.gamma(0.0), 0.0, "nu={nu}");
        }
    }

    /// Ground truth from gstat's `variogramLine` at `range = 100`, sill = 1
    /// (`Cir`, `Hol`, `Wav`, `Exc` with `kappa = 1.5`/`0.8`/`2.0`); see the
    /// session notes deriving these exact closed forms by matching gstat's
    /// output numerically (no ambiguity/range-convention issue here, unlike
    /// Matérn/Ste).
    #[test]
    fn new_families_match_gstat_reference_values() {
        let hs = [1.0, 10.0, 30.0, 50.0, 80.0, 100.0, 150.0, 200.0, 300.0, 500.0];

        let cir_gstat = [
            0.01273218, 0.12711143, 0.37616234, 0.60899778, 0.89591196, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        for (&h, &expected) in hs.iter().zip(&cir_gstat) {
            let got = ModelKind::Circular.g(h, 100.0);
            assert!((got - expected).abs() < 1e-7, "Cir h={h}: {got} vs {expected}");
        }

        let hol_gstat = [
            1.666658e-05,
            1.665834e-03,
            1.493264e-02,
            4.114892e-02,
            1.033049e-01,
            1.585290e-01,
            3.350033e-01,
            5.453513e-01,
            9.529600e-01,
            1.191785e+00,
        ];
        for (&h, &expected) in hs.iter().zip(&hol_gstat) {
            let got = ModelKind::Hole.g(h, 100.0);
            assert!((got - expected).abs() < 1e-6, "Hol h={h}: {got} vs {expected}");
        }

        let wav_gstat = [
            0.0001644853,
            0.0163683569,
            0.1416063087,
            0.3633802276,
            0.7661276791,
            1.0,
            1.2122065908,
            1.0,
            1.0,
            1.0,
        ];
        for (&h, &expected) in hs.iter().zip(&wav_gstat) {
            let got = ModelKind::Wave.g(h, 100.0);
            assert!((got - expected).abs() < 1e-6, "Wav h={h}: {got} vs {expected}");
        }

        let exc15_gstat = [
            0.0009995002,
            0.0311280057,
            0.1515267892,
            0.2978114987,
            0.5110728376,
            0.6321205588,
            0.8407240915,
            0.9408942534,
            0.9944621693,
            0.9999860543,
        ];
        for (&h, &expected) in hs.iter().zip(&exc15_gstat) {
            let got = ModelKind::Stable(1.5).g(h, 100.0);
            assert!((got - expected).abs() < 1e-7, "Exc(1.5) h={h}: {got} vs {expected}");
        }
    }

    #[test]
    fn circular_and_stable_bounded_and_monotone() {
        for kind in [
            ModelKind::Circular,
            ModelKind::Stable(0.5),
            ModelKind::Stable(1.0),
            ModelKind::Stable(2.0),
        ] {
            let m = VariogramModel::new(0.0, vec![Structure::new(kind, 2.0, 50.0)]).unwrap();
            let mut prev = 0.0;
            for i in 1..=100 {
                let g = m.gamma(i as f64 * 5.0);
                assert!(g >= prev - 1e-12, "{kind}: non-monotone at {i}");
                assert!(g <= 2.0 + 1e-12, "{kind}: exceeds sill");
                prev = g;
            }
            assert!(m.gamma(5000.0) > 1.99, "{kind}");
        }
    }

    #[test]
    fn hole_and_wave_oscillate_and_are_valid_covariances() {
        // Both are valid (positive-definite) covariance models but are NOT
        // monotone: they overshoot the sill (negative covariance) once past
        // their first zero-crossing.
        let hol = VariogramModel::new(0.0, vec![Structure::new(ModelKind::Hole, 1.0, 100.0)])
            .unwrap();
        let wav = VariogramModel::new(0.0, vec![Structure::new(ModelKind::Wave, 1.0, 100.0)])
            .unwrap();
        assert_eq!(hol.gamma(0.0), 0.0);
        assert_eq!(wav.gamma(0.0), 0.0);
        // Wave's covariance is exactly zero at h = range by construction.
        assert!(wav.covariance(100.0).abs() < 1e-9, "{}", wav.covariance(100.0));
        // Hole overshoots the sill (negative covariance) well past its range.
        assert!(hol.gamma(500.0) > 1.0, "Hol should overshoot by h=500");
        assert!(hol.covariance(500.0) < 0.0);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// The bounded, monotone kinds (matches [`ModelKind::ALL`] plus the
        /// closed-form Materns) -- the ones expected to plateau at the sill
        /// and never exceed it, unlike Power (unbounded) or Hole/Wave
        /// (deliberately oscillating).
        fn bounded_kind() -> impl Strategy<Value = ModelKind> {
            prop_oneof![
                Just(ModelKind::Spherical),
                Just(ModelKind::Exponential),
                Just(ModelKind::Gaussian),
                Just(ModelKind::Matern15),
                Just(ModelKind::Matern25),
                Just(ModelKind::Circular),
            ]
        }

        proptest! {
            #[test]
            fn gamma_is_zero_at_the_origin_and_nonnegative(
                kind in bounded_kind(),
                nugget in 0.0f64..5.0,
                sill in 1e-3f64..5.0,
                range in 1e-3f64..500.0,
                h in 0.0f64..2000.0,
            ) {
                let m = VariogramModel::new(nugget, vec![Structure::new(kind, sill, range)]).unwrap();
                prop_assert_eq!(m.gamma(0.0), 0.0);
                prop_assert!(m.gamma(h) >= 0.0, "{kind}: gamma({h}) = {}", m.gamma(h));
            }

            #[test]
            fn covariance_equals_total_sill_minus_gamma(
                kind in bounded_kind(),
                nugget in 0.0f64..5.0,
                sill in 1e-3f64..5.0,
                range in 1e-3f64..500.0,
                h in 0.0f64..2000.0,
            ) {
                let m = VariogramModel::new(nugget, vec![Structure::new(kind, sill, range)]).unwrap();
                let lhs = m.covariance(h);
                let rhs = m.total_sill() - m.gamma(h);
                prop_assert!((lhs - rhs).abs() < 1e-9, "{lhs} vs {rhs}");
            }

            #[test]
            fn bounded_kind_never_exceeds_its_sill(
                kind in bounded_kind(),
                sill in 1e-3f64..5.0,
                range in 1e-3f64..500.0,
                h in 0.0f64..5000.0,
            ) {
                let m = VariogramModel::new(0.0, vec![Structure::new(kind, sill, range)]).unwrap();
                prop_assert!(
                    m.gamma(h) <= sill * (1.0 + 1e-9),
                    "{kind}: gamma({h}) = {} > sill {sill}", m.gamma(h)
                );
            }
        }
    }
}
