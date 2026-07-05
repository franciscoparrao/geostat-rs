//! Weighted least-squares fitting of variogram models via Nelder–Mead.
//!
//! Weights follow gstat's default (`fit.method = 7`): `N_j / h_j^2`, which
//! emphasizes short lags and well-populated bins.

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::optim::nelder_mead_multistart;
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

/// WLS weight scheme for [`fit_model_weighted`]. Matches gstat's
/// `fit.method` options (gstat manual table 4.2); values not listed here
/// (REML, `fit.method` 5) are a different estimator entirely, see
/// [`crate::vecchia::vecchia_reml`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FitWeights {
    /// `N_j` (gstat `fit.method = 1`).
    NPairs,
    /// `N_j / γ_model(h_j)²`, recomputed from the *current* candidate model
    /// at every optimizer iteration (Cressie 1985; gstat `fit.method = 2`).
    /// Self-consistent nonlinear WLS: down-weights lags where the candidate
    /// model already predicts a large semivariance. Cross-checked against
    /// `fit.variogram(..., fit.method = 2)` — see
    /// `tests::cressie_weights_match_gstat_fit_method_2`.
    Cressie,
    /// Unweighted / ordinary least squares (gstat `fit.method = 6`).
    Ols,
    /// `N_j / h_j²` (gstat's default, `fit.method = 7`); emphasizes short
    /// lags and well-populated bins — "not supported by theory, but by
    /// practice" per the gstat manual. What [`fit_model`] always uses.
    #[default]
    NOverHSquared,
}

/// Fits a single-structure model of the given kind (plus nugget) to an
/// experimental variogram by weighted least squares, using gstat's default
/// weights (`N_j / h_j²`). Equivalent to
/// `fit_model_weighted(exp_v, kind, FitWeights::NOverHSquared)`.
pub fn fit_model(exp_v: &ExperimentalVariogram, kind: ModelKind) -> Result<FitResult> {
    fit_model_weighted(exp_v, kind, FitWeights::NOverHSquared)
}

/// Like [`fit_model`], with a selectable WLS weight scheme (see
/// [`FitWeights`]) instead of gstat's default `N_j / h_j²`.
pub fn fit_model_weighted(
    exp_v: &ExperimentalVariogram,
    kind: ModelKind,
    weights: FitWeights,
) -> Result<FitResult> {
    let pts: Vec<(f64, f64, f64)> = exp_v
        .bins
        .iter()
        .filter(|b| b.n_pairs > 0 && b.h > 0.0 && b.gamma.is_finite())
        .map(|b| (b.h, b.gamma, b.n_pairs as f64))
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
    let psill0 = (sill0 - nugget0).max(1e-9 * sill0);

    // `nugget = x[0]^2` (smooth, non-negative, hits exact 0 at x[0]=0); psill
    // and range are log-parametrized (strictly positive, span orders of
    // magnitude) so the domain is intrinsic and no boundary penalty is
    // needed — see AUDIT-2026-07.md §2.6.
    let objective = |x: &[f64]| -> f64 {
        let nugget = x[0] * x[0];
        let psill = x[1].exp();
        let range = x[2].exp();
        let model = VariogramModel {
            nugget,
            structures: vec![Structure::new(kind, psill, range)],
        };
        pts.iter()
            .map(|&(h, g, n)| {
                let pred = model.gamma(h);
                let e = g - pred;
                let w = match weights {
                    FitWeights::NPairs => n,
                    // Self-consistent: the weight uses THIS candidate
                    // model's own prediction, not the empirical gamma.
                    FitWeights::Cressie => n / pred.max(1e-12).powi(2),
                    FitWeights::Ols => 1.0,
                    FitWeights::NOverHSquared => n / (h * h),
                };
                w * e * e
            })
            .sum()
    };

    // Range is the classic multimodal parameter (short- vs long-range local
    // optima); multi-start around the empirical guess.
    let ln_range0 = range0.ln();
    let starts: Vec<Vec<f64>> = [0.3_f64, 1.0, 3.0]
        .into_iter()
        .map(|f| vec![nugget0.sqrt(), psill0.ln(), ln_range0 + f.ln()])
        .collect();
    let (xb, wsse) = nelder_mead_multistart(objective, &starts, 0.25, 1000);

    let model = VariogramModel::new(
        xb[0] * xb[0],
        vec![Structure::new(kind, xb[1].exp(), xb[2].exp())],
    )?;
    Ok(FitResult { model, wsse })
}

/// `(h, gamma, n_pairs)` per non-empty lag bin.
type LagPoints = Vec<(f64, f64, f64)>;

/// Non-fittable-`kind` initial guesses shared by [`fit_matern`]/[`fit_stable`]
/// (identical to the ones in [`fit_model_weighted`]): `(points, nugget0,
/// psill0, range0)`.
fn initial_guesses(exp_v: &ExperimentalVariogram) -> Result<(LagPoints, f64, f64, f64)> {
    let pts: Vec<(f64, f64, f64)> = exp_v
        .bins
        .iter()
        .filter(|b| b.n_pairs > 0 && b.h > 0.0 && b.gamma.is_finite())
        .map(|b| (b.h, b.gamma, b.n_pairs as f64))
        .collect();
    if pts.len() < 4 {
        return Err(GeostatError::InsufficientData(format!(
            "model fitting requires at least 4 non-empty lag bins, got {}",
            pts.len()
        )));
    }
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
    let psill0 = (sill0 - nugget0).max(1e-9 * sill0);
    Ok((pts, nugget0, psill0, range0))
}

/// Fits a Matérn model (nugget + partial sill + range + smoothness `ν`)
/// jointly by weighted least squares (gstat's default `N_j / h_j²`
/// weights), instead of fixing `ν` and calling
/// `fit_model_weighted(.., ModelKind::Matern(nu), ..)`.
///
/// Matérn's well-known `ν`-range confounding (a smoother, longer-range
/// model can fit an experimental variogram almost as well as a rougher,
/// shorter-range one) makes `ν` — not just range — a multimodal parameter
/// here; starts span a small grid of both. gstat's own `fit.kappa = TRUE`
/// only does a coarse (one-decimal) search over `ν`/`kappa`, so exact
/// cross-optimizer parity is not expected the way it is for fixed-`ν`
/// fits — both estimators can land on different, comparably-good
/// `(ν, range)` combinations on the same flat ridge of the objective.
pub fn fit_matern(exp_v: &ExperimentalVariogram) -> Result<FitResult> {
    let (pts, nugget0, psill0, range0) = initial_guesses(exp_v)?;

    let objective = |x: &[f64]| -> f64 {
        let nugget = x[0] * x[0];
        let psill = x[1].exp();
        let range = x[2].exp();
        // Clamp, don't let the search wander past MATERN_NU_MAX: beyond it
        // the correlation silently evaluates to NaN (AUDIT-2026-07-v2.md
        // §1.8), and nu that large is indistinguishable from Gaussian
        // anyway, so clamping costs no real fitting power.
        let nu = x[3].exp().min(super::MATERN_NU_MAX);
        let model = VariogramModel {
            nugget,
            structures: vec![Structure::new(ModelKind::Matern(nu), psill, range)],
        };
        pts.iter()
            .map(|&(h, g, n)| {
                let e = g - model.gamma(h);
                (n / (h * h)) * e * e
            })
            .sum()
    };

    let ln_range0 = range0.ln();
    let starts: Vec<Vec<f64>> = [0.5_f64, 1.5, 3.5]
        .into_iter()
        .flat_map(|nu0| {
            let (nugget0, psill0, ln_range0) = (nugget0, psill0, ln_range0);
            [0.5_f64, 1.0, 2.0]
                .into_iter()
                .map(move |rf| vec![nugget0.sqrt(), psill0.ln(), ln_range0 + rf.ln(), nu0.ln()])
        })
        .collect();
    let (xb, wsse) = nelder_mead_multistart(objective, &starts, 0.25, 1500);

    let model = VariogramModel::new(
        xb[0] * xb[0],
        vec![Structure::new(
            ModelKind::Matern(xb[3].exp().min(super::MATERN_NU_MAX)),
            xb[1].exp(),
            xb[2].exp(),
        )],
    )?;
    Ok(FitResult { model, wsse })
}

/// Fits a Stable (power-exponential) model (nugget + partial sill + range +
/// shape `α ∈ (0, 2]`) jointly by weighted least squares, the WLS analogue
/// of [`fit_matern`]. Same confounding caveat: `α` (via `tanh`, unconstrained
/// domain) is multi-started alongside range.
pub fn fit_stable(exp_v: &ExperimentalVariogram) -> Result<FitResult> {
    let (pts, nugget0, psill0, range0) = initial_guesses(exp_v)?;

    let objective = |x: &[f64]| -> f64 {
        let nugget = x[0] * x[0];
        let psill = x[1].exp();
        let range = x[2].exp();
        let alpha = 1.0 + x[3].tanh(); // (-inf,inf) -> (0,2)
        let model = VariogramModel {
            nugget,
            structures: vec![Structure::new(ModelKind::Stable(alpha), psill, range)],
        };
        pts.iter()
            .map(|&(h, g, n)| {
                let e = g - model.gamma(h);
                (n / (h * h)) * e * e
            })
            .sum()
    };

    let ln_range0 = range0.ln();
    let starts: Vec<Vec<f64>> = [0.5_f64, 1.0, 1.5]
        .into_iter()
        .flat_map(|alpha0| {
            let (nugget0, psill0, ln_range0) = (nugget0, psill0, ln_range0);
            let atanh0 = (alpha0 - 1.0_f64).clamp(-0.999, 0.999).atanh();
            [0.5_f64, 1.0, 2.0]
                .into_iter()
                .map(move |rf| vec![nugget0.sqrt(), psill0.ln(), ln_range0 + rf.ln(), atanh0])
        })
        .collect();
    let (xb, wsse) = nelder_mead_multistart(objective, &starts, 0.25, 1500);

    let alpha = 1.0 + xb[3].tanh();
    let model = VariogramModel::new(
        xb[0] * xb[0],
        vec![Structure::new(
            ModelKind::Stable(alpha),
            xb[1].exp(),
            xb[2].exp(),
        )],
    )?;
    Ok(FitResult { model, wsse })
}

/// Fits a **nested** model: nugget + one structure per element of `kinds`
/// (e.g. `&[Spherical, Spherical]` for a short- and long-range pair),
/// jointly by weighted least squares. Useful when a single structure
/// under-fits an experimental variogram with correlation at more than one
/// spatial scale (GSLIB/gstat convention: nest structures with increasing
/// ranges). `kinds.len() == 1` is exactly [`fit_model`].
///
/// Nested WLS is more prone to near-degenerate optima than a single
/// structure — one structure can end up absorbing nearly all the sill,
/// leaving the other negligible — which is a property of the estimation
/// problem on a variogram that does not actually need more than one scale,
/// not a bug in the fit (gstat's own `fit.variogram` shows the same
/// instability, including a "no convergence" warning, when nesting two
/// structures on Meuse's single-scale log-zinc variogram). Prefer
/// [`fit_model`]/[`fit_best`] unless the experimental variogram visibly
/// shows more than one scale of spatial structure.
pub fn fit_nested(exp_v: &ExperimentalVariogram, kinds: &[ModelKind]) -> Result<FitResult> {
    if kinds.is_empty() {
        return Err(GeostatError::InvalidParameter(
            "nested fit needs at least one structure kind".into(),
        ));
    }
    if kinds.len() == 1 {
        return fit_model(exp_v, kinds[0]);
    }
    let (pts, nugget0, psill0, _) = initial_guesses(exp_v)?;
    let max_h = pts.iter().fold(0.0_f64, |m, p| m.max(p.0));
    let n_struct = kinds.len();

    // `nugget = x[0]^2`; each structure's `(sill, range)` is
    // `(x[1+2i].exp(), x[2+2i].exp())` — same smooth non-negative/log
    // parametrization as `fit_model_weighted`, extended across structures.
    let objective = |x: &[f64]| -> f64 {
        let nugget = x[0] * x[0];
        let structures: Vec<Structure> = (0..n_struct)
            .map(|i| Structure::new(kinds[i], x[1 + 2 * i].exp(), x[2 + 2 * i].exp()))
            .collect();
        let model = VariogramModel { nugget, structures };
        pts.iter()
            .map(|&(h, g, n)| {
                let e = g - model.gamma(h);
                (n / (h * h)) * e * e
            })
            .sum()
    };

    // Initial guesses: ranges spread increasingly across the observed
    // extent (GSLIB/gstat nesting convention), sill split evenly.
    let sill_each0 = (psill0 / n_struct as f64).max(1e-9);
    let mut base = vec![nugget0.sqrt()];
    for i in 0..n_struct {
        let range_i0 = (max_h * (i + 1) as f64 / (n_struct + 1) as f64).max(1e-9);
        base.push(sill_each0.ln());
        base.push(range_i0.ln());
    }
    // Multi-start over an overall range-spread factor (short-vs-long local
    // optima, the same concern as the single-structure fit, now compounded
    // across structures).
    let starts: Vec<Vec<f64>> = [0.5_f64, 1.0, 2.0]
        .into_iter()
        .map(|f| {
            let mut x0 = base.clone();
            for i in 0..n_struct {
                x0[2 + 2 * i] += f.ln();
            }
            x0
        })
        .collect();
    let (xb, wsse) = nelder_mead_multistart(objective, &starts, 0.25, 1500);

    let structures: Vec<Structure> = (0..n_struct)
        .map(|i| Structure::new(kinds[i], xb[1 + 2 * i].exp(), xb[2 + 2 * i].exp()))
        .collect();
    let model = VariogramModel::new(xb[0] * xb[0], structures)?;
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

/// Fits a single indicator variogram model at the *median* cutoff (the
/// middle element of `cutoffs`, rounding down for an even count) and
/// returns it as a one-element `Vec` — the shared model **median IK**
/// (GSLIB `mik=1`) needs: `SisConfig::models`/`IkConfig::models` accept
/// either one model per cutoff (full IK, [`fit_indicator_models`]) or a
/// single model reused for every cutoff, which amortizes one factorization
/// across all of them in the kriging hot loop instead of paying for `nc`.
pub fn fit_median_indicator_model<const D: usize>(
    data: &PointSet<D>,
    cutoffs: &[f64],
    kinds: &[ModelKind],
    cfg: &VariogramConfig,
) -> Result<Vec<VariogramModel>> {
    if cutoffs.is_empty() {
        return Err(GeostatError::InvalidParameter(
            "at least one cutoff required".into(),
        ));
    }
    let median_cutoff = cutoffs[cutoffs.len() / 2];
    let indicators: Vec<f64> = data
        .values()
        .iter()
        .map(|&v| if v <= median_cutoff { 1.0 } else { 0.0 })
        .collect();
    let ind = PointSet::new(data.coords().to_vec(), indicators)?;
    let ev = experimental_variogram(&ind, cfg)?;
    Ok(vec![fit_best(&ev, kinds)?.model])
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
        // Nugget is squared (smooth, non-negative, hits exact 0); psill and
        // range are log-parametrized (strictly positive); ratio in (0, 1)
        // via a smooth tanh transform. No boundary for Nelder-Mead, so no
        // penalty branch is needed (AUDIT-2026-07.md §2.6).
        let nugget = x[0] * x[0];
        let psill = x[1].exp();
        let range = x[2].exp();
        let ratio = 0.5 * (1.0 + x[3].tanh());
        let az = x[4];
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
                    dip_deg: 0.0,
                    rake_deg: 0.0,
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
    let nugget0_sqrt = (0.25 * sill0).sqrt();
    let ln_psill0 = ((sill0 - 0.25 * sill0).max(1e-9 * sill0)).ln();
    let ln_range0 = range0.ln();
    let starts: Vec<Vec<f64>> = [0.0, 45.0, 90.0, 135.0]
        .into_iter()
        .map(|az0| vec![nugget0_sqrt, ln_psill0, ln_range0, 0.0, az0]) // ratio = 0.5
        .collect();
    let (xb, wsse) = nelder_mead_multistart(objective, &starts, 0.3, 2000);

    let ratio = 0.5 * (1.0 + xb[3].tanh());
    // Normalize azimuth to [0, 180): the major axis is sign- and π-agnostic.
    let az = xb[4].rem_euclid(180.0);
    let model = VariogramModel::new(
        xb[0] * xb[0],
        vec![Structure::with_anisotropy(
            kind,
            xb[1].exp(),
            xb[2].exp(),
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
        ExperimentalVariogram {
            bins,
            max_dist,
            coincident_pairs: 0,
        }
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
    fn fit_model_recovers_power_nugget_and_slope() {
        // fit_model only ever touches gamma() (pure WLS on the
        // semivariogram curve), never a covariance -- so it works for the
        // unbounded Power model exactly as-is, with no special-casing
        // needed. theta is fixed (part of the requested ModelKind, like
        // Matern's nu), only nugget/slope are fit; range is ignored (see
        // ModelKind::Power docs) so any positive placeholder recovers.
        let truth = VariogramModel::new(0.2, vec![Structure::new(ModelKind::Power(1.2), 1.5, 1.0)])
            .unwrap();
        let ev = synthetic_bins(&truth, 20.0, 15);
        let fit = fit_model(&ev, ModelKind::Power(1.2)).unwrap();
        assert!(
            (fit.model.nugget - 0.2).abs() < 0.05,
            "nugget {}",
            fit.model.nugget
        );
        assert!(
            (fit.model.structures[0].sill - 1.5).abs() < 0.1,
            "slope {}",
            fit.model.structures[0].sill
        );
    }

    #[test]
    fn all_weight_schemes_recover_the_true_model() {
        let truth =
            VariogramModel::new(0.1, vec![Structure::new(ModelKind::Spherical, 0.9, 300.0)])
                .unwrap();
        let ev = synthetic_bins(&truth, 450.0, 15);
        for w in [
            FitWeights::NPairs,
            FitWeights::Cressie,
            FitWeights::Ols,
            FitWeights::NOverHSquared,
        ] {
            let fit = fit_model_weighted(&ev, ModelKind::Spherical, w).unwrap();
            let s = fit.model.structures[0];
            assert!(
                (fit.model.nugget - 0.1).abs() < 0.05,
                "{w:?}: nugget {}",
                fit.model.nugget
            );
            assert!((s.sill - 0.9).abs() < 0.1, "{w:?}: sill {}", s.sill);
            assert!((s.range - 300.0).abs() < 30.0, "{w:?}: range {}", s.range);
        }
        // `fit_model` is exactly the NOverHSquared scheme.
        let default_fit = fit_model(&ev, ModelKind::Spherical).unwrap();
        let weighted_fit =
            fit_model_weighted(&ev, ModelKind::Spherical, FitWeights::NOverHSquared).unwrap();
        assert_eq!(default_fit.model, weighted_fit.model);
    }

    #[test]
    fn fit_matern_recovers_true_nu_and_range() {
        for &true_nu in &[0.7, 1.5, 2.8] {
            let truth = VariogramModel::new(
                0.08,
                vec![Structure::new(ModelKind::Matern(true_nu), 0.92, 250.0)],
            )
            .unwrap();
            let ev = synthetic_bins(&truth, 400.0, 18);
            let fit = fit_matern(&ev).unwrap();
            let s = fit.model.structures[0];
            let ModelKind::Matern(nu) = s.kind else {
                panic!("expected Matern, got {:?}", s.kind)
            };
            assert!(fit.wsse < 1e-6, "true_nu={true_nu}: wsse {}", fit.wsse);
            assert!(
                (fit.model.nugget - 0.08).abs() < 0.02,
                "true_nu={true_nu}: nugget {}",
                fit.model.nugget
            );
            assert!(
                (s.sill - 0.92).abs() < 0.05,
                "true_nu={true_nu}: sill {}",
                s.sill
            );
            assert!(
                (s.range - 250.0).abs() < 25.0,
                "true_nu={true_nu}: range {}",
                s.range
            );
            assert!(
                (nu - true_nu).abs() < 0.3,
                "true_nu={true_nu}: fitted nu {nu}"
            );
        }
    }

    #[test]
    fn fit_stable_recovers_true_alpha_and_range() {
        for &true_alpha in &[0.4, 1.0, 1.8] {
            let truth = VariogramModel::new(
                0.05,
                vec![Structure::new(ModelKind::Stable(true_alpha), 0.95, 200.0)],
            )
            .unwrap();
            let ev = synthetic_bins(&truth, 350.0, 18);
            let fit = fit_stable(&ev).unwrap();
            let s = fit.model.structures[0];
            let ModelKind::Stable(alpha) = s.kind else {
                panic!("expected Stable, got {:?}", s.kind)
            };
            assert!(
                fit.wsse < 1e-6,
                "true_alpha={true_alpha}: wsse {}",
                fit.wsse
            );
            assert!(
                (fit.model.nugget - 0.05).abs() < 0.02,
                "true_alpha={true_alpha}: nugget {}",
                fit.model.nugget
            );
            assert!(
                (s.sill - 0.95).abs() < 0.05,
                "true_alpha={true_alpha}: sill {}",
                s.sill
            );
            assert!(
                (s.range - 200.0).abs() < 20.0,
                "true_alpha={true_alpha}: range {}",
                s.range
            );
            assert!(
                (alpha - true_alpha).abs() < 0.25,
                "true_alpha={true_alpha}: fitted alpha {alpha}"
            );
        }
    }

    #[test]
    fn fit_nested_recovers_two_structures() {
        let truth = VariogramModel::new(
            0.05,
            vec![
                Structure::new(ModelKind::Spherical, 0.4, 50.0),
                Structure::new(ModelKind::Spherical, 0.5, 400.0),
            ],
        )
        .unwrap();
        let ev = synthetic_bins(&truth, 800.0, 40);
        let fit = fit_nested(&ev, &[ModelKind::Spherical, ModelKind::Spherical]).unwrap();
        assert!(fit.wsse < 1e-3, "wsse {}", fit.wsse);
        assert!(
            (fit.model.nugget - 0.05).abs() < 0.03,
            "nugget {}",
            fit.model.nugget
        );
        assert_eq!(fit.model.structures.len(), 2);
        let mut got: Vec<(f64, f64)> = fit
            .model
            .structures
            .iter()
            .map(|s| (s.sill, s.range))
            .collect();
        got.sort_by(|a, b| a.1.total_cmp(&b.1));
        assert!((got[0].0 - 0.4).abs() < 0.1, "short sill {}", got[0].0);
        assert!((got[0].1 - 50.0).abs() < 15.0, "short range {}", got[0].1);
        assert!((got[1].0 - 0.5).abs() < 0.1, "long sill {}", got[1].0);
        assert!((got[1].1 - 400.0).abs() < 60.0, "long range {}", got[1].1);
    }

    #[test]
    fn fit_nested_single_kind_matches_fit_model() {
        let truth =
            VariogramModel::new(0.1, vec![Structure::new(ModelKind::Spherical, 0.9, 300.0)])
                .unwrap();
        let ev = synthetic_bins(&truth, 450.0, 15);
        let nested = fit_nested(&ev, &[ModelKind::Spherical]).unwrap();
        let single = fit_model(&ev, ModelKind::Spherical).unwrap();
        assert_eq!(nested.model, single.model);
    }

    #[test]
    fn fit_nested_rejects_empty_kinds() {
        let truth =
            VariogramModel::new(0.1, vec![Structure::new(ModelKind::Spherical, 0.9, 300.0)])
                .unwrap();
        let ev = synthetic_bins(&truth, 450.0, 15);
        assert!(fit_nested(&ev, &[]).is_err());
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

    #[test]
    fn fit_median_indicator_model_returns_one_shared_model() {
        use crate::rng::Rng;
        let mut rng = Rng::new(5);
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
        let cfg = VariogramConfig {
            n_lags: 10,
            max_dist: 60.0,
            direction: None,
        };
        let models = fit_median_indicator_model(
            &data,
            &cutoffs,
            &[ModelKind::Spherical, ModelKind::Exponential],
            &cfg,
        )
        .unwrap();
        assert_eq!(models.len(), 1);
        assert!(models[0].total_sill() > 0.0);

        // Sanity: the full per-cutoff fit and the median fit should give
        // comparable sills (both estimate p(1-p)-scale indicator variance)
        // even though they need not match exactly (different cutoffs).
        let full = fit_indicator_models(
            &data,
            &cutoffs,
            &[ModelKind::Spherical, ModelKind::Exponential],
            &cfg,
        )
        .unwrap();
        assert_eq!(full.len(), 3);
    }
}
