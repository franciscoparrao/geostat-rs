//! Weighted least-squares fitting of variogram models via Nelder–Mead.
//!
//! Weights follow gstat's default (`fit.method = 7`): `N_j / h_j^2`, which
//! emphasizes short lags and well-populated bins.

use crate::error::{GeostatError, Result};
use crate::variogram::experimental::ExperimentalVariogram;
use crate::variogram::model::{ModelKind, Structure, VariogramModel};

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
            structures: vec![Structure {
                kind,
                sill: psill,
                range,
            }],
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

/// Standard Nelder–Mead simplex minimization.
fn nelder_mead<F>(f: F, x0: &[f64], step: f64, max_iter: usize) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    const ALPHA: f64 = 1.0; // reflection
    const GAMMA: f64 = 2.0; // expansion
    const RHO: f64 = 0.5; // contraction
    const SIGMA: f64 = 0.5; // shrink

    let n = x0.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut p = x0.to_vec();
        p[i] += step;
        simplex.push(p);
    }
    let mut fv: Vec<f64> = simplex.iter().map(|p| f(p)).collect();

    for _ in 0..max_iter {
        // Order simplex by objective value.
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| fv[a].total_cmp(&fv[b]));
        simplex = idx.iter().map(|&i| simplex[i].clone()).collect();
        fv = idx.iter().map(|&i| fv[i]).collect();

        if (fv[n] - fv[0]).abs() <= 1e-12 * fv[0].abs().max(1e-12) {
            break;
        }

        let mut centroid = vec![0.0; n];
        for p in &simplex[..n] {
            for j in 0..n {
                centroid[j] += p[j] / n as f64;
            }
        }

        let xr: Vec<f64> = (0..n)
            .map(|j| centroid[j] + ALPHA * (centroid[j] - simplex[n][j]))
            .collect();
        let fr = f(&xr);

        if fr < fv[0] {
            let xe: Vec<f64> = (0..n)
                .map(|j| centroid[j] + GAMMA * (centroid[j] - simplex[n][j]))
                .collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fv[n] = fe;
            } else {
                simplex[n] = xr;
                fv[n] = fr;
            }
        } else if fr < fv[n - 1] {
            simplex[n] = xr;
            fv[n] = fr;
        } else {
            let (xc, fc) = if fr < fv[n] {
                let xc: Vec<f64> = (0..n)
                    .map(|j| centroid[j] + RHO * (xr[j] - centroid[j]))
                    .collect();
                let fc = f(&xc);
                (xc, fc)
            } else {
                let xc: Vec<f64> = (0..n)
                    .map(|j| centroid[j] - RHO * (centroid[j] - simplex[n][j]))
                    .collect();
                let fc = f(&xc);
                (xc, fc)
            };
            if fc < fr.min(fv[n]) {
                simplex[n] = xc;
                fv[n] = fc;
            } else {
                let best = simplex[0].clone();
                for i in 1..=n {
                    for (pj, &bj) in simplex[i].iter_mut().zip(&best) {
                        *pj = bj + SIGMA * (*pj - bj);
                    }
                    fv[i] = f(&simplex[i]);
                }
            }
        }
    }

    let mut best = 0;
    for i in 1..=n {
        if fv[i] < fv[best] {
            best = i;
        }
    }
    (simplex[best].clone(), fv[best])
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
}
