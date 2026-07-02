//! Regression kriging: a trend model plus kriging of its residuals.
//!
//! Regression kriging (RK) predicts `z*(x0) = m*(x0) + r*(x0)`, where `m*` is
//! a trend (drift) fitted in a first, *separate* step and `r*` is the kriged
//! residual `z - m`. Unlike kriging with an external drift ([`crate::Kriging::
//! with_external_drift`]), which solves the trend and the spatial structure in
//! a single system, RK decouples them — so the trend can come from *any*
//! regression, not just a linear drift: ordinary least squares, a generalized
//! linear model, or a machine-learning model (random forest, gradient
//! boosting). This is the two-step form promoted by Hengl et al. and Li (2021)
//! and the natural bridge to an ML trend engine.
//!
//! The core type, [`RegressionKriging`], takes the trend already evaluated at
//! the data locations (and, at prediction time, at the targets): a
//! "bring-your-own-trend" contract that works with an external model. For
//! convenience, [`OlsTrend`] fits a plain linear trend on covariates so the
//! whole pipeline is self-contained when no external model is supplied.

use ndarray::Array2;

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::kriging::{Kriging, KrigingConfig, KrigingEstimate};
use crate::linalg;
use crate::variogram::VariogramModel;

/// A fitted linear trend `m(c) = b0 + b1 c1 + ... + bp cp`, by ordinary least
/// squares on covariates `c` (an intercept is added automatically).
#[derive(Debug, Clone)]
pub struct OlsTrend {
    /// Coefficients, intercept first: `[b0, b1, ..., bp]`.
    coefficients: Vec<f64>,
}

impl OlsTrend {
    /// Fits the trend by OLS. `covariates[i]` holds the `p` covariate values
    /// at observation `i`; all rows must share the same length, and there must
    /// be more observations than coefficients (`n > p + 1`).
    pub fn fit(covariates: &[Vec<f64>], values: &[f64]) -> Result<Self> {
        let n = covariates.len();
        if n != values.len() {
            return Err(GeostatError::DimensionMismatch(format!(
                "{n} covariate rows vs {} values",
                values.len()
            )));
        }
        let p = covariates.first().map_or(0, Vec::len);
        if covariates.iter().any(|row| row.len() != p) {
            return Err(GeostatError::DimensionMismatch(
                "covariate rows differ in length".into(),
            ));
        }
        let k = p + 1; // intercept + slopes
        if n <= k {
            return Err(GeostatError::InsufficientData(format!(
                "OLS trend needs more than {k} observations, got {n}"
            )));
        }

        // Design matrix X (n x k) with a leading column of ones.
        let mut x = Array2::<f64>::zeros((n, k));
        for (i, row) in covariates.iter().enumerate() {
            x[[i, 0]] = 1.0;
            for (j, &c) in row.iter().enumerate() {
                x[[i, j + 1]] = c;
            }
        }

        // Normal equations (X^T X) b = X^T y.
        let mut xtx = Array2::<f64>::zeros((k, k));
        let mut xty = vec![0.0; k];
        for a in 0..k {
            for b in a..k {
                let mut s = 0.0;
                for i in 0..n {
                    s += x[[i, a]] * x[[i, b]];
                }
                xtx[[a, b]] = s;
                xtx[[b, a]] = s;
            }
            let mut s = 0.0;
            for i in 0..n {
                s += x[[i, a]] * values[i];
            }
            xty[a] = s;
        }

        let coefficients = linalg::solve(xtx, xty)?;
        Ok(Self { coefficients })
    }

    /// Predicted trend at a covariate row of length `p`.
    pub fn predict(&self, covariates: &[f64]) -> f64 {
        let mut m = self.coefficients[0];
        for (b, &c) in self.coefficients[1..].iter().zip(covariates) {
            m += b * c;
        }
        m
    }

    /// Fitted coefficients, intercept first.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

/// OLS residuals of a polynomial trend in the coordinates (degree 1 or 2),
/// as a point set sharing the data's coordinates.
///
/// This is the standard preparation for *universal-kriging variography*
/// (gstat's `variogram(z ~ x + y, ...)`): fitting a variogram to raw data
/// that carry a trend inflates range and sill, so the variogram must be
/// estimated on trend residuals. Coordinates are centered and scaled
/// internally for numerical conditioning; the residuals are unaffected
/// (an affine change of variables spans the same polynomial space).
pub fn detrend_polynomial<const D: usize>(
    data: &PointSet<D>,
    degree: u8,
) -> Result<(PointSet<D>, OlsTrend)> {
    if !(1..=2).contains(&degree) {
        return Err(GeostatError::InvalidParameter(format!(
            "detrend degree must be 1 or 2, got {degree}"
        )));
    }
    let (min, max) = data.bbox();
    let mut center = [0.0; D];
    let mut spread = 0.0_f64;
    for d in 0..D {
        center[d] = 0.5 * (min[d] + max[d]);
        spread = spread.max(max[d] - min[d]);
    }
    let scale = if spread > 0.0 { spread } else { 1.0 };
    let covariates: Vec<Vec<f64>> = data
        .coords()
        .iter()
        .map(|c| {
            let u: Vec<f64> = (0..D).map(|d| (c[d] - center[d]) / scale).collect();
            let mut row = u.clone();
            if degree == 2 {
                for a in 0..D {
                    for b in a..D {
                        row.push(u[a] * u[b]);
                    }
                }
            }
            row
        })
        .collect();
    detrend_external(data, &covariates)
}

/// OLS residuals of a linear trend in external covariates (the
/// kriging-with-external-drift analogue of [`detrend_polynomial`]):
/// `covariates[i]` holds the covariate values at data point `i`.
pub fn detrend_external<const D: usize>(
    data: &PointSet<D>,
    covariates: &[Vec<f64>],
) -> Result<(PointSet<D>, OlsTrend)> {
    let trend = OlsTrend::fit(covariates, data.values())?;
    let resid: Vec<f64> = data
        .values()
        .iter()
        .zip(covariates)
        .map(|(&z, row)| z - trend.predict(row))
        .collect();
    Ok((PointSet::new(data.coords().to_vec(), resid)?, trend))
}

/// Regression-kriging predictor: kriges the residuals of an externally
/// supplied trend and adds the trend back at the target.
#[derive(Debug)]
pub struct RegressionKriging<const D: usize = 2> {
    /// Same coordinates as the data, with values `z - m` (residuals).
    residuals: PointSet<D>,
}

impl<const D: usize> RegressionKriging<D> {
    /// Builds the predictor from the data and the trend evaluated at every
    /// data location. The residual `z_i - trend_at_data[i]` is what gets
    /// kriged; fit the residual variogram on [`Self::residuals`].
    pub fn new(data: &PointSet<D>, trend_at_data: &[f64]) -> Result<Self> {
        if trend_at_data.len() != data.len() {
            return Err(GeostatError::DimensionMismatch(format!(
                "{} trend values vs {} data points",
                trend_at_data.len(),
                data.len()
            )));
        }
        let resid: Vec<f64> = data
            .values()
            .iter()
            .zip(trend_at_data)
            .map(|(&z, &m)| z - m)
            .collect();
        let residuals = PointSet::new(data.coords().to_vec(), resid)?;
        Ok(Self { residuals })
    }

    /// The residual point set (`z - m`), for fitting the residual variogram.
    pub fn residuals(&self) -> &PointSet<D> {
        &self.residuals
    }

    /// Predicts at `targets`, adding the supplied trend at each target back to
    /// the kriged residual. The returned `value` is `m*(x0) + r*(x0)`; the
    /// `variance` is the residual kriging variance (the trend's own prediction
    /// uncertainty is model-specific and not included — document accordingly
    /// when a calibrated total variance is required).
    pub fn predict(
        &self,
        targets: &[[f64; D]],
        trend_at_targets: &[f64],
        residual_model: &VariogramModel,
        config: &KrigingConfig,
    ) -> Result<Vec<KrigingEstimate>> {
        if trend_at_targets.len() != targets.len() {
            return Err(GeostatError::DimensionMismatch(format!(
                "{} trend values vs {} targets",
                trend_at_targets.len(),
                targets.len()
            )));
        }
        let kriging = Kriging::new(&self.residuals, residual_model, config.clone())?;
        let resid_est = kriging.predict_many(targets);
        Ok(resid_est
            .into_iter()
            .zip(trend_at_targets)
            .map(|(e, &m)| KrigingEstimate {
                value: m + e.value,
                variance: e.variance,
                lagrange: e.lagrange,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kriging::KrigingMethod;
    use crate::rng::Rng;
    use crate::variogram::{ModelKind, Structure};

    #[test]
    fn ols_recovers_linear_coefficients() {
        // y = 2 + 3*c1 - 1*c2 exactly.
        let covariates: Vec<Vec<f64>> = (0..30)
            .map(|i| {
                let c1 = (i % 7) as f64;
                let c2 = (i % 5) as f64;
                vec![c1, c2]
            })
            .collect();
        let values: Vec<f64> = covariates.iter().map(|c| 2.0 + 3.0 * c[0] - c[1]).collect();
        let trend = OlsTrend::fit(&covariates, &values).unwrap();
        let b = trend.coefficients();
        assert!((b[0] - 2.0).abs() < 1e-9, "intercept {}", b[0]);
        assert!((b[1] - 3.0).abs() < 1e-9, "slope1 {}", b[1]);
        assert!((b[2] + 1.0).abs() < 1e-9, "slope2 {}", b[2]);
        assert!((trend.predict(&[4.0, 2.0]) - (2.0 + 12.0 - 2.0)).abs() < 1e-9);
    }

    fn sample_field(n: usize, seed: u64) -> (PointSet, Vec<Vec<f64>>) {
        let mut rng = Rng::new(seed);
        let mut coords = Vec::new();
        let mut values = Vec::new();
        let mut covs = Vec::new();
        for _ in 0..n {
            let x = rng.uniform() * 100.0;
            let y = rng.uniform() * 100.0;
            // Trend in a covariate (here: x itself) + smooth correlated noise.
            let resid = (x / 25.0).sin() + (y / 30.0).cos() + 0.2 * rng.normal();
            let v = 5.0 + 0.4 * x + resid;
            coords.push([x, y]);
            values.push(v);
            covs.push(vec![x]);
        }
        (PointSet::new(coords, values).unwrap(), covs)
    }

    #[test]
    fn detrend_removes_exact_polynomial() {
        // Values exactly linear in the coordinates: residuals must vanish.
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                let p = [180_000.0 + i as f64 * 100.0, 330_000.0 + j as f64 * 100.0];
                coords.push(p);
                values.push(3.0 + 0.002 * p[0] - 0.001 * p[1]);
            }
        }
        let data = PointSet::new(coords, values).unwrap();
        let (resid, _) = detrend_polynomial(&data, 1).unwrap();
        for &r in resid.values() {
            assert!(r.abs() < 1e-6, "residual {r}");
        }
        // Degree 2 handles a quadratic surface (large UTM-like coordinates
        // exercise the internal normalization).
        let vals2: Vec<f64> = data
            .coords()
            .iter()
            .map(|c| 1.0 + 1e-7 * c[0] * c[1] - 2e-8 * c[0] * c[0])
            .collect();
        let data2 = PointSet::new(data.coords().to_vec(), vals2).unwrap();
        let (resid2, _) = detrend_polynomial(&data2, 2).unwrap();
        for &r in resid2.values() {
            assert!(r.abs() < 1e-5, "quadratic residual {r}");
        }
    }

    #[test]
    fn residual_variogram_deflates_trended_sill() {
        use crate::variogram::{VariogramConfig, experimental_variogram};
        // Strong linear trend over weak noise: the raw variogram grows
        // without bound while the residual variogram stays near the noise.
        let mut rng = Rng::new(17);
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for _ in 0..200 {
            let x = rng.uniform() * 100.0;
            let y = rng.uniform() * 100.0;
            coords.push([x, y]);
            values.push(0.5 * x + 0.1 * rng.normal());
        }
        let data = PointSet::new(coords, values).unwrap();
        let cfg = VariogramConfig {
            n_lags: 10,
            max_dist: 60.0,
            direction: None,
        };
        let raw = experimental_variogram(&data, &cfg).unwrap();
        let (resid, trend) = detrend_polynomial(&data, 1).unwrap();
        let res = experimental_variogram(&resid, &cfg).unwrap();
        let last_raw = raw.bins.last().unwrap().gamma;
        let last_res = res.bins.last().unwrap().gamma;
        assert!(
            last_res < 0.05 * last_raw,
            "residual gamma {last_res} vs raw {last_raw}"
        );
        // The de-trended slope is recoverable from the normalized coefficients.
        assert!(trend.coefficients().len() == 3);
    }

    #[test]
    fn rk_is_exact_at_data_points() {
        let (data, covs) = sample_field(80, 3);
        let trend = OlsTrend::fit(&covs, data.values()).unwrap();
        let trend_at_data: Vec<f64> = covs.iter().map(|c| trend.predict(c)).collect();
        let rk = RegressionKriging::new(&data, &trend_at_data).unwrap();
        let model =
            VariogramModel::new(0.001, vec![Structure::new(ModelKind::Spherical, 0.5, 40.0)])
                .unwrap();
        // Predict at a handful of data locations: residual kriging is an exact
        // interpolator, so z* = m + r returns the observed value.
        let cfg = KrigingConfig::default();
        for i in (0..data.len()).step_by(20) {
            let est = rk
                .predict(&[data.coord(i)], &[trend_at_data[i]], &model, &cfg)
                .unwrap();
            assert!(
                (est[0].value - data.value(i)).abs() < 1e-6,
                "{} vs {}",
                est[0].value,
                data.value(i)
            );
        }
    }

    #[test]
    fn rk_with_constant_trend_equals_ordinary_kriging() {
        // A constant trend makes residuals = z - c; ordinary kriging is
        // invariant to an additive constant, so RK must match OK on z.
        let (data, _) = sample_field(60, 7);
        let c = 10.0;
        let trend_at_data = vec![c; data.len()];
        let rk = RegressionKriging::new(&data, &trend_at_data).unwrap();
        let model = VariogramModel::new(
            0.01,
            vec![Structure::new(ModelKind::Exponential, 0.6, 35.0)],
        )
        .unwrap();
        let cfg = KrigingConfig {
            method: KrigingMethod::Ordinary,
            max_neighbors: None,
            search_radius: None,
            ..Default::default()
        };
        let targets = [[40.0, 55.0], [70.0, 20.0], [15.0, 80.0]];

        let rk_est = rk.predict(&targets, &[c; 3], &model, &cfg).unwrap();
        let ok = Kriging::new(&data, &model, cfg).unwrap();
        for (t, e) in targets.iter().zip(&rk_est) {
            let ok_e = ok.predict(*t).unwrap();
            assert!(
                (e.value - ok_e.value).abs() < 1e-9,
                "{} vs {}",
                e.value,
                ok_e.value
            );
            assert!((e.variance - ok_e.variance).abs() < 1e-9);
        }
    }
}
