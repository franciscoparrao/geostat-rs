//! Weighted least-squares fitting of variogram models via Nelder–Mead.
//!
//! Weights follow gstat's default (`fit.method = 7`): `N_j / h_j^2`, which
//! emphasizes short lags and well-populated bins.

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::optim::nelder_mead;
use crate::variogram::experimental::{
    DirectionConfig, ExperimentalVariogram, VariogramConfig, experimental_variogram,
};
use crate::variogram::model::{Anisotropy, ModelKind, Structure, VariogramModel};

/// Result of fitting a model to an experimental variogram.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// Fitted model (nugget + one structure).
    pub model: VariogramModel,
    /// Weighted sum of squared errors at the optimum.
    pub wsse: f64,
}

/// Fits a single-structure model of the given kind (plus nugget) to an
/// experimental variogram by weighted least squares.
pub fn fit_model(exp_v: &ExperimentalVariogram, kind: ModelKind) -> Result<FitResult> {
    let pts: Vec<(f64, f64, f64)> = exp_v
        .bins
        .iter()
        .filter(|b| b.n_pairs > 0 && b.h > 0.0 && b.gamma.is_finite())
        .map(|b| (b.h, b.gamma, b.n_pairs as f64 / (b.h * b.h)))
        .collect();
    if pts.len() < 4 {
        return Err(GeostatError::InsufficientData(format!(
            "model fitting requires at least 4 non-empty lag bins, got {}",
            pts.len()
        )));
    }

    // Initial guesses from the empirical curve.
    let n_tail = (pts.len() / 3).max(1);
    let sill0 =
        (pts[pts.len() - n_tail..].iter().map(|p| p.1).sum::<f64>() / n_tail as f64).max(1e-12);
    let max_h = pts.iter().fold(0.0_f64, |m, p| m.max(p.0));
    let range0 = pts
        .iter()
        .find(|p| p.1 >= 0.95 * sill0)
        .map(|p| p.0)
        .unwrap_or(0.5 * max_h)
        .max(1e-12 * max_h.max(1.0));
    let nugget0 = pts[0].1.min(0.5 * sill0).max(0.0);

    // Parameters are optimized as multipliers of (sill0, sill0, range0).
    let objective = |x: &[f64]| -> f64 {
        let nugget = x[0] * sill0;
        let psill = x[1] * sill0;
        let range = x[2] * range0;
        let mut pen = 0.0;
        if nugget < 0.0 {
            pen += nugget * nugget;
        }
        if psill < 0.0 {
            pen += psill * psill;
        }
        if range <= 0.0 {
            pen += 1.0 + range * range;
        }
        if pen > 0.0 {
            return 1e12 * (1.0 + pen);
        }
        let model = VariogramModel {
            nugget,
            structures: vec![Structure::new(kind, psill, range)],
        };
        pts.iter()
            .map(|&(h, g, w)| {
                let e = g - model.gamma(h);
                w * e * e
            })
            .sum()
    };

    let x0 = [nugget0 / sill0, ((sill0 - nugget0).max(1e-9)) / sill0, 1.0];
    let (xb, wsse) = nelder_mead(objective, &x0, 0.25, 1000);

    let model = VariogramModel::new(
        (xb[0] * sill0).max(0.0),
        vec![Structure::new(
            kind,
            (xb[1] * sill0).max(0.0),
            (xb[2] * range0).max(1e-12),
        )],
    )?;
    Ok(FitResult { model, wsse })
}

/// Fits each candidate kind and returns the one with the lowest weighted SSE.
pub fn fit_best(exp_v: &ExperimentalVariogram, kinds: &[ModelKind]) -> Result<FitResult> {
    if kinds.is_empty() {
        return Err(GeostatError::InvalidParameter(
            "no candidate model kinds given".into(),
        ));
    }
    let mut best: Option<FitResult> = None;
    let mut last_err = None;
    for &kind in kinds {
        match fit_model(exp_v, kind) {
            Ok(r) => {
                if best.as_ref().is_none_or(|b| r.wsse < b.wsse) {
                    best = Some(r);
                }
            }
            Err(e) => last_err = Some(e),
        }
    }
    best.ok_or_else(|| {
        last_err.unwrap_or_else(|| GeostatError::InvalidParameter("fit failed".into()))
    })
}

/// Fits one indicator variogram model per cutoff: the data are coded as
/// `1.0` where `value <= cutoff` (else `0.0`), an experimental variogram is
/// computed under `cfg`, and the best of `kinds` is fitted. This is the
/// auto-fit path shared by sequential indicator simulation and indicator
/// kriging in every front-end.
pub fn fit_indicator_models<const D: usize>(
    data: &PointSet<D>,
    cutoffs: &[f64],
    kinds: &[ModelKind],
    cfg: &VariogramConfig,
) -> Result<Vec<VariogramModel>> {
    cutoffs
        .iter()
        .map(|&c| {
            let indicators: Vec<f64> = data
                .values()
                .iter()
                .map(|&v| if v <= c { 1.0 } else { 0.0 })
                .collect();
            let ind = PointSet::new(data.coords().to_vec(), indicators)?;
            let ev = experimental_variogram(&ind, cfg)?;
            Ok(fit_best(&ev, kinds)?.model)
        })
        .collect()
}

/// One directional bin used by the anisotropy fit: the lag's unit direction
/// `(sin φ, cos φ)`, its mean distance, the semivariance and the fit weight.
struct DirSample {
    sin: f64,
    cos: f64,
    h: f64,
    gamma: f64,
    weight: f64,
}

/// Computes experimental variograms in `n_dirs` evenly spaced azimuths over
/// `[0, 180)` (cones tiling the half-circle) and flattens their non-empty bins
/// into directional samples for joint fitting.
fn directional_samples(
    data: &PointSet<2>,
    n_dirs: usize,
    n_lags: usize,
    max_dist: f64,
) -> Result<Vec<DirSample>> {
    let tol = 90.0 / n_dirs as f64;
    let mut samples = Vec::new();
    for k in 0..n_dirs {
        let az = 180.0 * k as f64 / n_dirs as f64;
        let cfg = VariogramConfig {
            n_lags,
            max_dist,
            direction: Some(DirectionConfig::horizontal(az, tol)),
        };
        let ev = experimental_variogram(data, &cfg)?;
        let (sin, cos) = az.to_radians().sin_cos();
        for b in &ev.bins {
            if b.n_pairs > 0 && b.h > 0.0 && b.gamma.is_finite() {
                samples.push(DirSample {
                    sin,
                    cos,
                    h: b.h,
                    gamma: b.gamma,
                    weight: b.n_pairs as f64 / (b.h * b.h),
                });
            }
        }
    }
    Ok(samples)
}

/// Fits a single-structure model with geometric anisotropy of the given kind by
/// jointly weighted-least-squares fitting directional variograms. Returns the
/// model (with `azimuth_deg` of the major axis and `ratio = minor/major`) and
/// the weighted SSE. Azimuth is multi-started to avoid local minima.
fn fit_anisotropic_kind(
    samples: &[DirSample],
    kind: ModelKind,
    sill0: f64,
    range0: f64,
) -> Result<FitResult> {
    let objective = |x: &[f64]| -> f64 {
        let nugget = x[0] * sill0;
        let psill = x[1] * sill0;
        let range = x[2] * range0;
        // ratio in (0, 1) via a smooth transform: no boundary for Nelder-Mead.
        let ratio = 0.5 * (1.0 + x[3].tanh());
        let az = x[4];
        let mut pen = 0.0;
        if nugget < 0.0 {
            pen += nugget * nugget;
        }
        if psill < 0.0 {
            pen += psill * psill;
        }
        if range <= 0.0 {
            pen += 1.0 + range * range;
        }
        if pen > 0.0 {
            return 1e12 * (1.0 + pen);
        }
        let model = VariogramModel {
            nugget,
            structures: vec![Structure {
                kind,
                sill: psill,
                range,
                anis: Some(Anisotropy {
                    azimuth_deg: az,
                    ratio,
                    ratio_z: 1.0,
                }),
            }],
        };
        samples
            .iter()
            .map(|s| {
                let e = s.gamma - model.gamma_dh([s.h * s.sin, s.h * s.cos]);
                s.weight * e * e
            })
            .sum()
    };

    // Azimuth is the multimodal parameter; start the simplex from four seeds.
    let mut best: Option<(Vec<f64>, f64)> = None;
    for az0 in [0.0, 45.0, 90.0, 135.0] {
        let x0 = [
            0.25,
            ((sill0 - 0.25 * sill0).max(1e-9)) / sill0,
            1.0,
            0.0, // ratio = 0.5
            az0,
        ];
        let (xb, wsse) = nelder_mead(objective, &x0, 0.3, 2000);
        if best.as_ref().is_none_or(|(_, b)| wsse < *b) {
            best = Some((xb, wsse));
        }
    }
    let (xb, wsse) = best.expect("at least one start");

    let ratio = 0.5 * (1.0 + xb[3].tanh());
    // Normalize azimuth to [0, 180): the major axis is sign- and π-agnostic.
    let az = xb[4].rem_euclid(180.0);
    let model = VariogramModel::new(
        (xb[0] * sill0).max(0.0),
        vec![Structure::with_anisotropy(
            kind,
            (xb[1] * sill0).max(0.0),
            (xb[2] * range0).max(1e-12),
            az,
            ratio.clamp(1e-6, 1.0),
        )],
    )?;
    Ok(FitResult { model, wsse })
}

/// Fits a single-structure model with geometric anisotropy, choosing the best
/// of `kinds` by weighted SSE. Directional variograms are computed in `n_dirs`
/// azimuths over `[0, 180)` and fitted jointly, so the major-axis direction and
/// the anisotropy ratio are estimated from data rather than set by hand.
///
/// For (near-)isotropic data the fitted `ratio` approaches 1 and the azimuth is
/// not meaningful.
pub fn fit_anisotropic(
    data: &PointSet<2>,
    kinds: &[ModelKind],
    n_dirs: usize,
    n_lags: usize,
    max_dist: f64,
) -> Result<FitResult> {
    if kinds.is_empty() {
        return Err(GeostatError::InvalidParameter(
            "no candidate model kinds given".into(),
        ));
    }
    if n_dirs < 2 {
        return Err(GeostatError::InvalidParameter(
            "anisotropy fitting needs at least 2 directions".into(),
        ));
    }
    let samples = directional_samples(data, n_dirs, n_lags, max_dist)?;
    let n_dirs_with_data = {
        let mut dirs = samples.iter().map(|s| (s.sin, s.cos)).collect::<Vec<_>>();
        dirs.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9);
        dirs.len()
    };
    if samples.len() < 6 || n_dirs_with_data < 2 {
        return Err(GeostatError::InsufficientData(format!(
            "anisotropy fitting needs at least 6 directional bins across 2+ directions, \
             got {} bins in {n_dirs_with_data} directions",
            samples.len()
        )));
    }

    // Initial sill/range from the omnidirectional variogram.
    let omni = experimental_variogram(
        data,
        &VariogramConfig {
            n_lags,
            max_dist,
            direction: None,
        },
    )?;
    let pts: Vec<(f64, f64)> = omni
        .bins
        .iter()
        .filter(|b| b.n_pairs > 0 && b.h > 0.0 && b.gamma.is_finite())
        .map(|b| (b.h, b.gamma))
        .collect();
    let n_tail = (pts.len() / 3).max(1);
    let sill0 =
        (pts[pts.len() - n_tail..].iter().map(|p| p.1).sum::<f64>() / n_tail as f64).max(1e-12);
    let max_h = pts.iter().fold(0.0_f64, |m, p| m.max(p.0));
    let range0 = pts
        .iter()
        .find(|p| p.1 >= 0.95 * sill0)
        .map(|p| p.0)
        .unwrap_or(0.5 * max_h)
        .max(1e-12 * max_h.max(1.0));

    let mut best: Option<FitResult> = None;
    let mut last_err = None;
    for &kind in kinds {
        match fit_anisotropic_kind(&samples, kind, sill0, range0) {
            Ok(r) => {
                if best.as_ref().is_none_or(|b| r.wsse < b.wsse) {
                    best = Some(r);
                }
            }
            Err(e) => last_err = Some(e),
        }
    }
    best.ok_or_else(|| {
        last_err.unwrap_or_else(|| GeostatError::InvalidParameter("anisotropy fit failed".into()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::experimental::LagBin;

    /// Builds synthetic "experimental" bins exactly on a known model.
    fn synthetic_bins(model: &VariogramModel, max_dist: f64, n: usize) -> ExperimentalVariogram {
        let width = max_dist / n as f64;
        let bins = (0..n)
            .map(|i| {
                let h = (i as f64 + 0.5) * width;
                LagBin {
                    h,
                    gamma: model.gamma(h),
                    n_pairs: 100,
                }
            })
            .collect();
        ExperimentalVariogram { bins, max_dist }
    }

    #[test]
    fn recovers_spherical_parameters() {
        let truth =
            VariogramModel::new(0.1, vec![Structure::new(ModelKind::Spherical, 0.9, 300.0)])
                .unwrap();
        let ev = synthetic_bins(&truth, 450.0, 15);
        let fit = fit_model(&ev, ModelKind::Spherical).unwrap();
        let s = fit.model.structures[0];
        assert!(
            (fit.model.nugget - 0.1).abs() < 0.05,
            "nugget {}",
            fit.model.nugget
        );
        assert!((s.sill - 0.9).abs() < 0.1, "sill {}", s.sill);
        assert!((s.range - 300.0).abs() < 30.0, "range {}", s.range);
    }

    #[test]
    fn best_model_selects_generating_family() {
        let truth =
            VariogramModel::new(0.05, vec![Structure::new(ModelKind::Gaussian, 1.0, 200.0)])
                .unwrap();
        let ev = synthetic_bins(&truth, 600.0, 20);
        let fit = fit_best(&ev, &ModelKind::ALL).unwrap();
        assert_eq!(fit.model.structures[0].kind, ModelKind::Gaussian);
        assert!(fit.wsse < 1e-6, "wsse {}", fit.wsse);
    }

    #[test]
    fn requires_enough_bins() {
        let truth = VariogramModel::new(0.0, vec![Structure::new(ModelKind::Spherical, 1.0, 10.0)])
            .unwrap();
        let ev = synthetic_bins(&truth, 10.0, 3);
        assert!(fit_model(&ev, ModelKind::Spherical).is_err());
    }

    /// Circular azimuth difference in degrees, modulo 180.
    fn az_diff(a: f64, b: f64) -> f64 {
        ((a - b + 90.0).rem_euclid(180.0) - 90.0).abs()
    }

    /// Directional samples sitting exactly on a known anisotropic model.
    fn aniso_samples(
        truth: &VariogramModel,
        dirs: &[f64],
        n_lags: usize,
        max_dist: f64,
    ) -> Vec<DirSample> {
        let width = max_dist / n_lags as f64;
        let mut s = Vec::new();
        for &az in dirs {
            let (sin, cos) = az.to_radians().sin_cos();
            for i in 0..n_lags {
                let h = (i as f64 + 0.5) * width;
                s.push(DirSample {
                    sin,
                    cos,
                    h,
                    gamma: truth.gamma_dh([h * sin, h * cos]),
                    weight: 1.0,
                });
            }
        }
        s
    }

    #[test]
    fn recovers_anisotropy_from_synthetic_bins() {
        let truth = VariogramModel::new(
            0.1,
            vec![Structure::with_anisotropy(
                ModelKind::Spherical,
                0.9,
                120.0,
                40.0,
                0.4,
            )],
        )
        .unwrap();
        let dirs = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0];
        let samples = aniso_samples(&truth, &dirs, 12, 150.0);
        let fit = fit_anisotropic_kind(&samples, ModelKind::Spherical, 1.0, 120.0).unwrap();
        let s = fit.model.structures[0];
        let a = s.anis.unwrap();
        assert!(
            (fit.model.nugget - 0.1).abs() < 0.05,
            "nugget {}",
            fit.model.nugget
        );
        assert!((s.sill - 0.9).abs() < 0.1, "sill {}", s.sill);
        assert!((s.range - 120.0).abs() < 20.0, "range {}", s.range);
        assert!((a.ratio - 0.4).abs() < 0.1, "ratio {}", a.ratio);
        assert!(
            az_diff(a.azimuth_deg, 40.0) < 8.0,
            "azimuth {}",
            a.azimuth_deg
        );
    }

    #[test]
    fn fits_anisotropic_field_from_points() {
        use crate::rng::Rng;
        // Random field with long continuity along x (azimuth 90) and short along
        // y, built from Fourier modes whose y-frequencies dominate.
        let mut rng = Rng::new(7);
        let modes: Vec<(f64, f64, f64)> = (0..40)
            .map(|_| {
                let ox = (rng.uniform() - 0.5) * 0.02; // small -> long x range
                let oy = (rng.uniform() - 0.5) * 0.12; // large -> short y range
                let ph = rng.uniform() * std::f64::consts::TAU;
                (ox, oy, ph)
            })
            .collect();
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for _ in 0..400 {
            let x = rng.uniform() * 200.0;
            let y = rng.uniform() * 200.0;
            let v: f64 = modes
                .iter()
                .map(|&(ox, oy, ph)| (ox * x + oy * y + ph).cos())
                .sum();
            coords.push([x, y]);
            values.push(v);
        }
        let data = PointSet::new(coords, values).unwrap();
        let fit = fit_anisotropic(
            &data,
            &[ModelKind::Spherical, ModelKind::Exponential],
            4,
            12,
            120.0,
        )
        .unwrap();
        let a = fit.model.structures[0].anis.unwrap();
        // Major axis along x = azimuth 90; clearly anisotropic.
        assert!(a.ratio < 0.85, "ratio {}", a.ratio);
        assert!(
            az_diff(a.azimuth_deg, 90.0) < 25.0,
            "azimuth {}",
            a.azimuth_deg
        );
    }

    #[test]
    fn anisotropy_fit_rejects_too_few_directions() {
        let truth = VariogramModel::new(0.0, vec![Structure::new(ModelKind::Spherical, 1.0, 10.0)])
            .unwrap();
        let _ = truth;
        let data = PointSet::new(
            vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            vec![0.0, 1.0, 2.0],
        )
        .unwrap();
        assert!(fit_anisotropic(&data, &[ModelKind::Spherical], 1, 5, 3.0).is_err());
    }
}
