//! Lognormal (trans-Gaussian) kriging.
//!
//! The data are log-transformed, kriged in log space with the supplied
//! variogram of the logs, and back-transformed with the unbiased lognormal
//! correction:
//!
//! - simple kriging:  `z = exp(y + sigma2/2)`
//! - ordinary kriging: `z = exp(y + sigma2/2 - mu)`
//!
//! where `y` and `sigma2` are the kriging mean and variance in log space and
//! `mu` is the ordinary-kriging Lagrange multiplier in covariance form (the
//! `KrigingEstimate::lagrange` value), following Journel & Huijbregts (1978)
//! / Chilès & Delfiner. The simple-kriging back-transform is validated
//! against gstat at machine precision; the log-space kriging matches gstat's
//! OK to 1e-13. Note `gstat::krigeTg` uses a *different*, GLS-based
//! trans-Gaussian correction, so it is not a bit-for-bit oracle for the OK
//! case here.

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::kriging::{Kriging, KrigingConfig, KrigingMethod};
use crate::variogram::VariogramModel;

/// A lognormal-kriging estimate.
#[derive(Debug, Clone, Copy)]
pub struct LognormalEstimate {
    /// Back-transformed prediction, in original (data) units.
    pub value: f64,
    /// Kriging mean in log space.
    pub log_mean: f64,
    /// Kriging variance in log space.
    pub log_variance: f64,
}

/// Ordinary/simple lognormal kriging at arbitrary targets.
///
/// `data` holds the **original** (strictly positive) values; `log_model` is
/// the variogram fitted to `ln(value)`. Universal and external-drift methods
/// are rejected (the back-transform correction is only defined for simple
/// and ordinary kriging here).
pub fn lognormal_kriging<const D: usize>(
    data: &PointSet<D>,
    targets: &[[f64; D]],
    log_model: &VariogramModel,
    config: &KrigingConfig,
) -> Result<Vec<LognormalEstimate>> {
    let simple = match config.method {
        KrigingMethod::Simple { .. } => true,
        KrigingMethod::Ordinary => false,
        _ => {
            return Err(GeostatError::InvalidParameter(
                "lognormal kriging supports simple and ordinary kriging only".into(),
            ));
        }
    };
    if data.values().iter().any(|&v| !(v > 0.0)) {
        return Err(GeostatError::InvalidParameter(
            "lognormal kriging requires strictly positive data values".into(),
        ));
    }

    // For simple kriging the mean must be given in log units; if the user
    // passed the original-scale mean, that is their responsibility. We krige
    // the logs directly with the supplied config.
    let logs: Vec<f64> = data.values().iter().map(|&v| v.ln()).collect();
    let log_data = PointSet::new(data.coords().to_vec(), logs)?;
    let kriging = Kriging::new(&log_data, log_model, config.clone())?;

    Ok(crate::parallel::par_map(targets.len(), |t| {
        match kriging.predict(targets[t]) {
            Ok(est) => {
                let mu = est.lagrange.unwrap_or(0.0);
                let correction = if simple {
                    0.5 * est.variance
                } else {
                    0.5 * est.variance - mu
                };
                LognormalEstimate {
                    value: (est.value + correction).exp(),
                    log_mean: est.value,
                    log_variance: est.variance,
                }
            }
            Err(_) => LognormalEstimate {
                value: f64::NAN,
                log_mean: f64::NAN,
                log_variance: f64::NAN,
            },
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::variogram::{ModelKind, Structure};

    fn lognormal_field(n: usize, seed: u64) -> PointSet {
        let mut rng = Rng::new(seed);
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for _ in 0..n {
            let x = rng.uniform() * 100.0;
            let y = rng.uniform() * 100.0;
            // Smooth log field -> lognormal values.
            let logv = (x / 30.0).sin() + (y / 25.0).cos() + 0.2 * rng.normal();
            coords.push([x, y]);
            values.push(logv.exp());
        }
        PointSet::new(coords, values).unwrap()
    }

    #[test]
    fn exact_at_data_points() {
        let data = lognormal_field(120, 3);
        let model =
            VariogramModel::new(0.01, vec![Structure::new(ModelKind::Spherical, 0.5, 40.0)])
                .unwrap();
        // At a datum, log variance ~0 and lagrange ~0, so back-transform
        // returns the original value.
        let targets: Vec<[f64; 2]> = (0..5).map(|i| data.coord(i * 13)).collect();
        let est = lognormal_kriging(&data, &targets, &model, &KrigingConfig::default()).unwrap();
        for (e, i) in est.iter().zip((0..5).map(|i| i * 13)) {
            assert!(
                (e.value - data.value(i)).abs() < 1e-4,
                "{} vs {}",
                e.value,
                data.value(i)
            );
        }
    }

    #[test]
    fn back_transform_inflates_above_naive_exp() {
        // Away from data, the bias correction makes z > exp(log_mean).
        let data = lognormal_field(80, 9);
        let model = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 0.6, 30.0)],
        )
        .unwrap();
        let est =
            lognormal_kriging(&data, &[[50.0, 50.0]], &model, &KrigingConfig::default()).unwrap();
        let e = est[0];
        assert!(e.log_variance > 0.0);
        assert!(e.value > e.log_mean.exp(), "correction must inflate");
    }

    #[test]
    fn rejects_nonpositive_and_drift_methods() {
        let bad = PointSet::new(
            vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
            vec![1.0, -1.0, 2.0],
        )
        .unwrap();
        let model = VariogramModel::new(0.0, vec![Structure::new(ModelKind::Spherical, 1.0, 10.0)])
            .unwrap();
        assert!(lognormal_kriging(&bad, &[[0.5, 0.5]], &model, &KrigingConfig::default()).is_err());
        let good = lognormal_field(20, 1);
        let cfg = KrigingConfig {
            method: KrigingMethod::Universal { degree: 1 },
            ..Default::default()
        };
        assert!(lognormal_kriging(&good, &[[1.0, 1.0]], &model, &cfg).is_err());
    }
}
