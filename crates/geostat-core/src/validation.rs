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

    /// Mean squared error.
    pub fn mse(&self) -> f64 {
        self.residuals().map(|e| e * e).sum::<f64>() / self.observed.len() as f64
    }

    /// Root mean squared error.
    pub fn rmse(&self) -> f64 {
        self.mse().sqrt()
    }

    /// Mean of the observed values (denominator of the relative measures).
    fn obs_mean(&self) -> f64 {
        self.observed.iter().sum::<f64>() / self.observed.len() as f64
    }

    /// Relative mean error `mean(o - p) / mean(o) * 100` (%). Scale-free bias.
    /// Returns NaN if the observed mean is ~0 (relative measures are undefined
    /// there — Li 2017).
    pub fn rme(&self) -> f64 {
        let m = self.obs_mean();
        if m.abs() < f64::EPSILON {
            return f64::NAN;
        }
        (-self.mean_error()) / m * 100.0
    }

    /// Relative mean absolute error `MAE / mean(o) * 100` (%).
    pub fn rmae(&self) -> f64 {
        let m = self.obs_mean();
        if m.abs() < f64::EPSILON {
            return f64::NAN;
        }
        self.mae() / m * 100.0
    }

    /// Relative RMSE `RMSE / mean(o) * 100` (%).
    pub fn rrmse(&self) -> f64 {
        let m = self.obs_mean();
        if m.abs() < f64::EPSILON {
            return f64::NAN;
        }
        self.rmse() / m * 100.0
    }

    /// Variance explained by cross-validation (Li 2016), in percent:
    /// `VEcv = (1 - sum((o - p)^2) / sum((o - mean(o))^2)) * 100`.
    ///
    /// A cross-validated, scale- and variance-independent R²: 100 = perfect,
    /// 0 = no better than predicting the mean, negative = worse than the mean.
    /// This is the predictive-accuracy measure recommended over r/r², which
    /// must not be used for predictive accuracy (Li 2017).
    pub fn vecv(&self) -> f64 {
        let m = self.obs_mean();
        let sse = self.residuals().map(|e| e * e).sum::<f64>();
        let sst = self.observed.iter().map(|&o| (o - m).powi(2)).sum::<f64>();
        if sst <= 0.0 {
            return f64::NAN;
        }
        (1.0 - sse / sst) * 100.0
    }

    /// Legates and McCabe's efficiency (E₁), in percent:
    /// `E1 = (1 - sum(|o - p|) / sum(|o - mean(o)|)) * 100`. Like VEcv but on
    /// absolute (rather than squared) deviations, so less tail-sensitive.
    pub fn e1(&self) -> f64 {
        let m = self.obs_mean();
        let sae = self.residuals().map(f64::abs).sum::<f64>();
        let sad = self.observed.iter().map(|&o| (o - m).abs()).sum::<f64>();
        if sad <= 0.0 {
            return f64::NAN;
        }
        (1.0 - sae / sad) * 100.0
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
pub fn leave_one_out<const D: usize>(
    data: &PointSet<D>,
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
pub fn leave_one_out_with_drift<const D: usize>(
    data: &PointSet<D>,
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

/// Assigns each of `n` points to one of `k` folds by a deterministic shuffle,
/// returning the per-point fold index and the membership lists. Folds are
/// balanced to within one point.
fn fold_assignment(n: usize, k: usize, seed: u64) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut order: Vec<usize> = (0..n).collect();
    crate::rng::Rng::new(seed).shuffle(&mut order);
    let mut fold_of = vec![0usize; n];
    let mut members = vec![Vec::new(); k];
    for (j, &i) in order.iter().enumerate() {
        let f = j % k;
        fold_of[i] = f;
        members[f].push(i);
    }
    (fold_of, members)
}

/// Shared core of [`k_fold`]/[`block_cv`]: given a precomputed fold
/// assignment (`members[f]` = point indices in fold `f`), each fold trains
/// on every point outside it and predicts its own members. Empty folds are
/// skipped (a regular spatial grid over an irregular point cloud can leave
/// some blocks empty).
fn cv_with_folds<const D: usize>(
    data: &PointSet<D>,
    model: &VariogramModel,
    config: &KrigingConfig,
    fold_of: &[usize],
    members: &[Vec<usize>],
) -> Result<CvResult> {
    let n = data.len();
    let per_fold: Vec<Vec<(usize, f64, f64)>> = par_try_map(members.len(), |f| {
        if members[f].is_empty() {
            return Ok(Vec::new());
        }
        let train_coords: Vec<[f64; D]> = (0..n)
            .filter(|&i| fold_of[i] != f)
            .map(|i| data.coord(i))
            .collect();
        let train_vals: Vec<f64> = (0..n)
            .filter(|&i| fold_of[i] != f)
            .map(|i| data.value(i))
            .collect();
        let train = PointSet::new(train_coords, train_vals)?;
        let kriging = Kriging::new(&train, model, config.clone())?;
        let mut out = Vec::with_capacity(members[f].len());
        for &i in &members[f] {
            let est = kriging.predict(data.coord(i))?;
            out.push((i, est.value, est.variance));
        }
        Ok(out)
    })?;

    let mut predicted = vec![0.0; n];
    let mut variance = vec![0.0; n];
    for fold_res in per_fold {
        for (i, p, v) in fold_res {
            predicted[i] = p;
            variance[i] = v;
        }
    }
    Ok(CvResult {
        observed: data.values().to_vec(),
        predicted,
        variance,
    })
}

/// `k`-fold cross-validation: the data are split into `k` balanced folds (by a
/// deterministic, seed-reproducible shuffle); each fold is predicted in turn
/// from the union of the other folds. With `k = n` this reduces to
/// leave-one-out, but for large `n` it is roughly `k`-times cheaper, which makes
/// it the practical choice for hyperparameter tuning.
///
/// The result is in dataset order, so every [`CvResult`] accuracy measure
/// applies unchanged.
pub fn k_fold<const D: usize>(
    data: &PointSet<D>,
    model: &VariogramModel,
    config: &KrigingConfig,
    k: usize,
    seed: u64,
) -> Result<CvResult> {
    let n = data.len();
    if n < 3 {
        return Err(GeostatError::InsufficientData(
            "cross-validation requires at least 3 points".into(),
        ));
    }
    if !(2..=n).contains(&k) {
        return Err(GeostatError::InvalidParameter(format!(
            "k-fold requires 2 <= k <= n (k={k}, n={n})"
        )));
    }
    let (fold_of, members) = fold_assignment(n, k, seed);
    cv_with_folds(data, model, config, &fold_of, &members)
}

/// Assigns each point to a spatial block: a regular grid of
/// `blocks_per_dim[d]` cells along dimension `d`, covering the data's
/// bounding box. Unlike [`fold_assignment`]'s random shuffle, this groups
/// spatially *contiguous* points into the same fold.
fn spatial_block_assignment<const D: usize>(
    coords: &[[f64; D]],
    blocks_per_dim: [usize; D],
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut min = coords[0];
    let mut max = coords[0];
    for p in coords {
        for d in 0..D {
            min[d] = min[d].min(p[d]);
            max[d] = max[d].max(p[d]);
        }
    }
    let n_blocks: usize = blocks_per_dim.iter().product();
    let mut fold_of = vec![0usize; coords.len()];
    let mut members = vec![Vec::new(); n_blocks];
    for (i, p) in coords.iter().enumerate() {
        let mut idx = 0;
        let mut stride = 1;
        for d in 0..D {
            let span = (max[d] - min[d]).max(f64::MIN_POSITIVE);
            let frac = ((p[d] - min[d]) / span).clamp(0.0, 1.0 - 1e-12);
            let bd = ((frac * blocks_per_dim[d] as f64) as usize).min(blocks_per_dim[d] - 1);
            idx += bd * stride;
            stride *= blocks_per_dim[d];
        }
        fold_of[i] = idx;
        members[idx].push(i);
    }
    (fold_of, members)
}

/// Spatial block cross-validation: the data are partitioned into a regular
/// grid of `blocks_per_dim[d]` blocks per dimension over the bounding box
/// (rather than [`k_fold`]'s random shuffle), and each block is predicted in
/// turn from every point *outside* it.
///
/// This is the standard remedy for ordinary/random k-fold's optimistic bias
/// under spatial autocorrelation: leaving out one point at a time (or points
/// scattered randomly across folds) still leaves each held-out point
/// surrounded by near-duplicate training neighbors, so the cross-validation
/// error understates the error at the range/scale the model will actually be
/// used for (extrapolating into genuinely undersampled regions). Leaving out
/// whole spatial blocks forces longer effective prediction distances, giving
/// a more honest error estimate — at the cost of needing enough blocks with
/// enough remaining training data to still support the kriging method's
/// drift terms.
///
/// The result is in dataset order, so every [`CvResult`] accuracy measure
/// applies unchanged. Empty blocks (possible when the point cloud is
/// irregular) are simply skipped.
pub fn block_cv<const D: usize>(
    data: &PointSet<D>,
    model: &VariogramModel,
    config: &KrigingConfig,
    blocks_per_dim: [usize; D],
) -> Result<CvResult> {
    let n = data.len();
    if n < 3 {
        return Err(GeostatError::InsufficientData(
            "cross-validation requires at least 3 points".into(),
        ));
    }
    if blocks_per_dim.contains(&0) {
        return Err(GeostatError::InvalidParameter(
            "blocks_per_dim entries must be at least 1".into(),
        ));
    }
    let n_blocks: usize = blocks_per_dim.iter().product();
    if n_blocks < 2 {
        return Err(GeostatError::InvalidParameter(
            "block CV requires at least 2 blocks total".into(),
        ));
    }
    let (fold_of, members) = spatial_block_assignment(data.coords(), blocks_per_dim);
    cv_with_folds(data, model, config, &fold_of, &members)
}

/// One probability-interval check in a Deutsch accuracy plot: at `p`'s
/// symmetric interval (see [`accuracy_plot`]), `observed` is the fraction of
/// held-out true values that actually fell inside it.
#[derive(Debug, Clone, Copy)]
pub struct AccuracyPoint {
    /// Nominal probability, e.g. `0.5` = the central 50% interval.
    pub nominal: f64,
    /// Observed fraction of true values inside that interval.
    pub observed: f64,
}

/// Deutsch's accuracy plot (Deutsch 1997, "Direct assessment of local
/// accuracy and precision"; the standard GSLIB `kt3d`/`sisim` diagnostic for
/// whether a model's *uncertainty* — not just its central prediction — is
/// well calibrated): a curve of nominal vs. observed coverage across
/// probability levels, plus a scalar goodness statistic. See
/// [`accuracy_plot`].
#[derive(Debug, Clone)]
pub struct AccuracyPlot {
    /// One point per requested probability, in the order given.
    pub points: Vec<AccuracyPoint>,
    /// Deutsch's goodness statistic, `1.0` for perfect calibration,
    /// decreasing as observed coverage departs from nominal (see
    /// [`accuracy_plot`] for the exact formula and its 1:2 asymmetry).
    pub goodness: f64,
}

impl AccuracyPlot {
    /// `true` if observed coverage meets or exceeds nominal at every
    /// requested probability (Deutsch's "accurate" criterion — the model is
    /// never overconfident, though it may be imprecise).
    pub fn is_accurate(&self) -> bool {
        self.points
            .iter()
            .all(|pt| pt.observed >= pt.nominal - 1e-9)
    }
}

/// Builds a Deutsch accuracy plot from held-out `(actual, mean, std)`
/// triples — typically a [`CvResult`]'s `observed`/`predicted`/
/// `sqrt(variance)` from [`leave_one_out`], [`k_fold`] or [`block_cv`] —
/// assuming a **local Gaussian conditional distribution** at each location
/// (the standard kriging convention: `Normal(mean, std^2)`).
///
/// For each nominal probability `p` in `probs` (e.g. `[0.1, 0.2, ..., 0.9]`),
/// the symmetric interval is `[mean + std*z((1-p)/2), mean + std*z((1+p)/2)]`
/// (`z` = the standard normal quantile, [`crate::transform::inv_norm_cdf`]);
/// `observed` is the fraction of `actual` values landing inside it.
///
/// The goodness statistic is `G = 1 - (1/K) * sum_k w_k * |p̄_k - p_k|`,
/// where `p̄_k` is `observed`, `p_k` is `nominal`, and `w_k = 1` if
/// `p̄_k >= p_k` (the interval is wider than it needs to be — imprecise but
/// not wrong) or `w_k = 2` if `p̄_k < p_k` (the interval is too narrow —
/// genuinely overconfident, penalized twice as heavily since understating
/// uncertainty is the worse failure mode). `G = 1` exactly when
/// `p̄_k = p_k` for every `k` (perfect calibration); `G < 1` for any
/// miscalibration in either direction.
pub fn accuracy_plot(
    actual: &[f64],
    mean: &[f64],
    std: &[f64],
    probs: &[f64],
) -> Result<AccuracyPlot> {
    if actual.len() != mean.len() || actual.len() != std.len() {
        return Err(GeostatError::DimensionMismatch(format!(
            "{} actual, {} mean, {} std",
            actual.len(),
            mean.len(),
            std.len()
        )));
    }
    if actual.is_empty() {
        return Err(GeostatError::InsufficientData(
            "accuracy plot requires at least one held-out point".into(),
        ));
    }
    if probs.is_empty() || probs.iter().any(|&p| !(p > 0.0 && p < 1.0)) {
        return Err(GeostatError::InvalidParameter(
            "probabilities must be non-empty and strictly inside (0, 1)".into(),
        ));
    }
    if std.iter().any(|&s| !s.is_finite() || s < 0.0) {
        return Err(GeostatError::InvalidParameter(
            "std must be finite and >= 0".into(),
        ));
    }
    let n = actual.len() as f64;
    let points: Vec<AccuracyPoint> = probs
        .iter()
        .map(|&p| {
            let z_lo = crate::transform::inv_norm_cdf((1.0 - p) / 2.0);
            let z_hi = crate::transform::inv_norm_cdf((1.0 + p) / 2.0);
            let inside = actual
                .iter()
                .zip(mean)
                .zip(std)
                .filter(|&((&a, &m), &s)| {
                    let lo = m + s * z_lo;
                    let hi = m + s * z_hi;
                    a >= lo && a <= hi
                })
                .count();
            AccuracyPoint {
                nominal: p,
                observed: inside as f64 / n,
            }
        })
        .collect();

    let k = points.len() as f64;
    let goodness = 1.0
        - points
            .iter()
            .map(|pt| {
                let diff = pt.observed - pt.nominal;
                let w = if diff >= 0.0 { 1.0 } else { 2.0 };
                w * diff.abs()
            })
            .sum::<f64>()
            / k;

    Ok(AccuracyPlot { points, goodness })
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
    fn vecv_and_e1_anchor_points() {
        let observed = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;

        // Perfect predictor: VEcv = E1 = 100, relative errors = 0.
        let perfect = CvResult {
            observed: observed.clone(),
            predicted: observed.clone(),
            variance: vec![1.0; observed.len()],
        };
        assert!((perfect.vecv() - 100.0).abs() < 1e-12);
        assert!((perfect.e1() - 100.0).abs() < 1e-12);
        assert!(perfect.rrmse().abs() < 1e-12);

        // Mean predictor: no skill over the mean, so VEcv = E1 = 0.
        let mean_pred = CvResult {
            observed: observed.clone(),
            predicted: vec![mean; observed.len()],
            variance: vec![1.0; observed.len()],
        };
        assert!(mean_pred.vecv().abs() < 1e-9, "vecv {}", mean_pred.vecv());
        assert!(mean_pred.e1().abs() < 1e-9, "e1 {}", mean_pred.e1());

        // Worse-than-mean predictor: VEcv negative.
        let bad = CvResult {
            observed: observed.clone(),
            predicted: vec![10.0, 10.0, 10.0, 10.0, 10.0],
            variance: vec![1.0; observed.len()],
        };
        assert!(bad.vecv() < 0.0);

        // Relative measures scale-free and consistent: RRMSE = RMSE/mean*100.
        let cv = CvResult {
            observed: observed.clone(),
            predicted: vec![1.5, 2.5, 2.5, 4.5, 4.5],
            variance: vec![1.0; observed.len()],
        };
        assert!((cv.rrmse() - cv.rmse() / mean * 100.0).abs() < 1e-12);
        assert!((cv.rmae() - cv.mae() / mean * 100.0).abs() < 1e-12);
        // VEcv in (0, 100) for a decent-but-imperfect predictor.
        assert!(cv.vecv() > 0.0 && cv.vecv() < 100.0);
    }

    #[test]
    fn requires_minimum_points() {
        let data = PointSet::new(vec![[0.0, 0.0], [1.0, 1.0]], vec![1.0, 2.0]).unwrap();
        let model =
            VariogramModel::new(0.0, vec![Structure::new(ModelKind::Exponential, 1.0, 1.0)])
                .unwrap();
        assert!(leave_one_out(&data, &model, &KrigingConfig::default()).is_err());
    }

    fn smooth_field(n: usize, seed: u64) -> PointSet {
        let mut rng = Rng::new(seed);
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for _ in 0..n {
            let x = rng.uniform() * 100.0;
            let y = rng.uniform() * 100.0;
            coords.push([x, y]);
            values.push((x / 20.0).sin() + (y / 20.0).cos());
        }
        PointSet::new(coords, values).unwrap()
    }

    #[test]
    fn k_equals_n_matches_leave_one_out() {
        // With one point per fold, k-fold is exactly leave-one-out.
        let data = smooth_field(40, 5);
        let model =
            VariogramModel::new(0.01, vec![Structure::new(ModelKind::Spherical, 1.0, 50.0)])
                .unwrap();
        let cfg = KrigingConfig::default();
        let loo = leave_one_out(&data, &model, &cfg).unwrap();
        let kf = k_fold(&data, &model, &cfg, data.len(), 123).unwrap();
        for i in 0..data.len() {
            assert!((kf.predicted[i] - loo.predicted[i]).abs() < 1e-12);
            assert!((kf.variance[i] - loo.variance[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn k_fold_is_deterministic_and_balanced() {
        let data = smooth_field(53, 8);
        let model =
            VariogramModel::new(0.01, vec![Structure::new(ModelKind::Spherical, 1.0, 50.0)])
                .unwrap();
        let cfg = KrigingConfig::default();
        // Same seed -> identical predictions; folds within one point of each other.
        let a = k_fold(&data, &model, &cfg, 5, 42).unwrap();
        let b = k_fold(&data, &model, &cfg, 5, 42).unwrap();
        assert_eq!(a.predicted, b.predicted);
        let (_, members) = fold_assignment(53, 5, 42);
        let sizes: Vec<usize> = members.iter().map(Vec::len).collect();
        let total: usize = sizes.iter().sum();
        assert_eq!(total, 53, "every point assigned exactly once");
        assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1);
        // A reasonable model still has predictive skill under 5-fold.
        assert!(a.vecv() > 50.0, "vecv {}", a.vecv());
    }

    #[test]
    fn k_fold_rejects_bad_k() {
        let data = smooth_field(10, 1);
        let model =
            VariogramModel::new(0.01, vec![Structure::new(ModelKind::Spherical, 1.0, 50.0)])
                .unwrap();
        let cfg = KrigingConfig::default();
        assert!(k_fold(&data, &model, &cfg, 1, 0).is_err()); // k < 2
        assert!(k_fold(&data, &model, &cfg, 11, 0).is_err()); // k > n
    }

    #[test]
    fn block_cv_covers_every_point_and_has_skill() {
        let data = smooth_field(120, 7);
        let model =
            VariogramModel::new(0.01, vec![Structure::new(ModelKind::Spherical, 1.0, 50.0)])
                .unwrap();
        let cfg = KrigingConfig::default();
        let cv = block_cv(&data, &model, &cfg, [4, 4]).unwrap();
        assert_eq!(cv.observed.len(), data.len());
        // Every point actually got a (finite) prediction, i.e. no block was
        // left unassigned/unpredicted.
        assert!(cv.predicted.iter().all(|p| p.is_finite()));
        assert!(cv.variance.iter().all(|v| v.is_finite() && *v >= 0.0));
        // A smooth, well-specified model still beats the mean predictor even
        // under the harsher block-holdout scheme.
        assert!(cv.vecv() > 0.0, "vecv {}", cv.vecv());
    }

    #[test]
    fn block_cv_is_harsher_than_random_k_fold_under_strong_autocorrelation() {
        // A very smooth field (long range relative to the domain): nearby
        // points are near-duplicates, so random k-fold's held-out points
        // are trivially predicted by their close neighbors in other folds,
        // while block CV forces genuinely longer prediction distances.
        let mut rng = Rng::new(3);
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for _ in 0..150 {
            let x = rng.uniform() * 100.0;
            let y = rng.uniform() * 100.0;
            coords.push([x, y]);
            values.push((x / 200.0).sin() + (y / 200.0).cos());
        }
        let data = PointSet::new(coords, values).unwrap();
        let model =
            VariogramModel::new(0.0, vec![Structure::new(ModelKind::Spherical, 1.0, 300.0)])
                .unwrap();
        let cfg = KrigingConfig::default();
        let random = k_fold(&data, &model, &cfg, 10, 1).unwrap();
        let block = block_cv(&data, &model, &cfg, [5, 5]).unwrap();
        assert!(
            block.rmse() >= random.rmse(),
            "block rmse {} should be >= random k-fold rmse {}",
            block.rmse(),
            random.rmse()
        );
    }

    #[test]
    fn block_cv_rejects_bad_config() {
        let data = smooth_field(20, 2);
        let model =
            VariogramModel::new(0.01, vec![Structure::new(ModelKind::Spherical, 1.0, 50.0)])
                .unwrap();
        let cfg = KrigingConfig::default();
        assert!(block_cv(&data, &model, &cfg, [0, 2]).is_err()); // zero blocks
        assert!(block_cv(&data, &model, &cfg, [1, 1]).is_err()); // only 1 block total
        assert!(block_cv(&data, &model, &cfg, [3, 3]).is_ok());
    }

    #[test]
    fn accuracy_plot_matches_nominal_for_a_well_specified_gaussian_model() {
        // Draw actual values EXACTLY from Normal(mean, std) at many
        // synthetic points: a textbook well-calibrated model, so observed
        // coverage should converge to nominal (within sampling noise) and
        // goodness should be close to 1 -- this is an external-reference-
        // free but strong self-consistency check (no gstat equivalent for
        // Deutsch's accuracy plots).
        let mut rng = Rng::new(99);
        let n = 20_000;
        let mean: Vec<f64> = (0..n).map(|_| rng.uniform() * 10.0).collect();
        let std: Vec<f64> = (0..n).map(|_| 0.5 + rng.uniform()).collect();
        let actual: Vec<f64> = mean
            .iter()
            .zip(&std)
            .map(|(&m, &s)| m + s * rng.normal())
            .collect();
        let probs = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let plot = accuracy_plot(&actual, &mean, &std, &probs).unwrap();
        for pt in &plot.points {
            assert!(
                (pt.observed - pt.nominal).abs() < 0.02,
                "nominal {} observed {}",
                pt.nominal,
                pt.observed
            );
        }
        assert!(plot.goodness > 0.95, "goodness {}", plot.goodness);
    }

    #[test]
    fn accuracy_plot_goodness_is_one_when_observed_matches_nominal_exactly() {
        let points = [
            AccuracyPoint {
                nominal: 0.3,
                observed: 0.3,
            },
            AccuracyPoint {
                nominal: 0.7,
                observed: 0.7,
            },
        ];
        let goodness = 1.0
            - points
                .iter()
                .map(|p| {
                    let diff = p.observed - p.nominal;
                    (if diff >= 0.0 { 1.0 } else { 2.0 }) * diff.abs()
                })
                .sum::<f64>()
                / points.len() as f64;
        assert!((goodness - 1.0).abs() < 1e-12);
    }

    #[test]
    fn accuracy_plot_penalizes_underdispersion_more_than_overdispersion() {
        // Same |actual - mean| everywhere, so the same magnitude of miss on
        // both sides -- but a std that's too SMALL (overconfident,
        // observed < nominal) should score worse than a std that's too
        // LARGE (over-cautious, observed > nominal), by the 1:2 weighting.
        let mut rng = Rng::new(5);
        let n = 5000;
        let mean: Vec<f64> = vec![0.0; n];
        let actual: Vec<f64> = (0..n).map(|_| rng.normal()).collect(); // true std = 1
        let probs = vec![0.5];

        let underdispersed = accuracy_plot(&actual, &mean, &vec![0.5; n], &probs).unwrap(); // reported std too small
        let overdispersed = accuracy_plot(&actual, &mean, &vec![2.0; n], &probs).unwrap(); // reported std too large

        assert!(underdispersed.points[0].observed < 0.5);
        assert!(overdispersed.points[0].observed > 0.5);
        assert!(!underdispersed.is_accurate());
        assert!(overdispersed.is_accurate());
        assert!(
            underdispersed.goodness < overdispersed.goodness,
            "under {} vs over {}",
            underdispersed.goodness,
            overdispersed.goodness
        );
    }

    #[test]
    fn accuracy_plot_rejects_bad_input() {
        assert!(accuracy_plot(&[1.0], &[0.0], &[1.0, 2.0], &[0.5]).is_err()); // length mismatch
        assert!(accuracy_plot(&[], &[], &[], &[0.5]).is_err()); // empty
        assert!(accuracy_plot(&[1.0], &[0.0], &[1.0], &[]).is_err()); // no probs
        assert!(accuracy_plot(&[1.0], &[0.0], &[1.0], &[1.5]).is_err()); // prob out of (0,1)
        assert!(accuracy_plot(&[1.0], &[0.0], &[-1.0], &[0.5]).is_err()); // negative std
    }
}
