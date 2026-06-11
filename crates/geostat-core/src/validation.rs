//! Leave-one-out cross-validation for kriging models.

use crate::parallel::par_try_map;

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::kriging::{Kriging, KrigingConfig};
use crate::variogram::VariogramModel;

/// Leave-one-out cross-validation result.
#[derive(Debug, Clone)]
pub struct CvResult {
    /// Observed values, in dataset order.
    pub observed: Vec<f64>,
    /// Cross-validation predictions.
    pub predicted: Vec<f64>,
    /// Kriging variances at the held-out locations.
    pub variance: Vec<f64>,
}

impl CvResult {
    /// Mean error (bias); ideally ~0.
    pub fn mean_error(&self) -> f64 {
        self.residuals().sum::<f64>() / self.observed.len() as f64
    }

    /// Mean absolute error.
    pub fn mae(&self) -> f64 {
        self.residuals().map(f64::abs).sum::<f64>() / self.observed.len() as f64
    }

    /// Root mean squared error.
    pub fn rmse(&self) -> f64 {
        (self.residuals().map(|e| e * e).sum::<f64>() / self.observed.len() as f64).sqrt()
    }

    /// Mean squared deviation ratio `mean(e^2 / sigma^2)`; ideally ~1.
    /// Bins with (numerically) zero kriging variance are skipped.
    pub fn msdr(&self) -> f64 {
        let mut sum = 0.0;
        let mut n = 0usize;
        for ((&p, &o), &v) in self
            .predicted
            .iter()
            .zip(&self.observed)
            .zip(&self.variance)
        {
            if v > 1e-12 {
                let e = p - o;
                sum += e * e / v;
                n += 1;
            }
        }
        if n == 0 { f64::NAN } else { sum / n as f64 }
    }

    fn residuals(&self) -> impl Iterator<Item = f64> {
        self.predicted
            .iter()
            .zip(&self.observed)
            .map(|(&p, &o)| p - o)
    }
}

/// Leave-one-out cross-validation: each point is predicted from all others
/// using the given model and kriging configuration.
pub fn leave_one_out(
    data: &PointSet,
    model: &VariogramModel,
    config: &KrigingConfig,
) -> Result<CvResult> {
    if data.len() < 3 {
        return Err(GeostatError::InsufficientData(
            "leave-one-out cross-validation requires at least 3 points".into(),
        ));
    }
    let estimates: Vec<(f64, f64)> = par_try_map(data.len(), |i| {
        let sub = data.excluding(i);
        let kriging = Kriging::new(&sub, model, config.clone())?;
        let est = kriging.predict(data.coord(i))?;
        Ok((est.value, est.variance))
    })?;

    let (predicted, variance) = estimates.into_iter().unzip();
    Ok(CvResult {
        observed: data.values().to_vec(),
        predicted,
        variance,
    })
}

/// Leave-one-out cross-validation for external-drift kriging. `drift_data[i]`
/// holds the covariates at data point `i`.
pub fn leave_one_out_with_drift(
    data: &PointSet,
    drift_data: &[Vec<f64>],
    model: &VariogramModel,
    config: &KrigingConfig,
) -> Result<CvResult> {
    if data.len() < 3 {
        return Err(GeostatError::InsufficientData(
            "leave-one-out cross-validation requires at least 3 points".into(),
        ));
    }
    if drift_data.len() != data.len() {
        return Err(GeostatError::DimensionMismatch(format!(
            "{} drift rows vs {} data points",
            drift_data.len(),
            data.len()
        )));
    }
    let estimates: Vec<(f64, f64)> = par_try_map(data.len(), |i| {
        let sub = data.excluding(i);
        let mut sub_drift = drift_data.to_vec();
        sub_drift.remove(i);
        let kriging = Kriging::with_external_drift(&sub, model, config.clone(), sub_drift)?;
        let est = kriging.predict_with_drift(data.coord(i), &drift_data[i])?;
        Ok((est.value, est.variance))
    })?;

    let (predicted, variance) = estimates.into_iter().unzip();
    Ok(CvResult {
        observed: data.values().to_vec(),
        predicted,
        variance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::variogram::{ModelKind, Structure};

    #[test]
    fn cv_beats_mean_predictor_on_smooth_field() {
        // Smooth deterministic field sampled at pseudo-random locations.
        let mut rng = Rng::new(11);
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for _ in 0..120 {
            let x = rng.uniform() * 100.0;
            let y = rng.uniform() * 100.0;
            coords.push([x, y]);
            values.push((x / 20.0).sin() + (y / 20.0).cos());
        }
        let data = PointSet::new(coords, values).unwrap();
        let model =
            VariogramModel::new(0.01, vec![Structure::new(ModelKind::Spherical, 1.0, 50.0)])
                .unwrap();
        let cv = leave_one_out(&data, &model, &KrigingConfig::default()).unwrap();

        let mean = data.mean();
        let std = (data
            .values()
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / data.len() as f64)
            .sqrt();

        assert!(cv.rmse() < 0.5 * std, "rmse {} vs std {std}", cv.rmse());
        assert!(cv.mean_error().abs() < 0.1);
        assert!(cv.mae() <= cv.rmse());
        assert!(cv.msdr().is_finite());
    }

    #[test]
    fn requires_minimum_points() {
        let data = PointSet::new(vec![[0.0, 0.0], [1.0, 1.0]], vec![1.0, 2.0]).unwrap();
        let model =
            VariogramModel::new(0.0, vec![Structure::new(ModelKind::Exponential, 1.0, 1.0)])
                .unwrap();
        assert!(leave_one_out(&data, &model, &KrigingConfig::default()).is_err());
    }
}
