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

    let goodness = goodness_from_points(&points);
    Ok(AccuracyPlot { points, goodness })
}

/// Deutsch's goodness statistic (see [`accuracy_plot`]'s docs for the exact
/// 1:2-asymmetric formula), shared between the Gaussian-interval
/// [`accuracy_plot`] and the ccdf-based [`accuracy_plot_ccdf`] -- the
/// statistic only depends on the nominal/observed coverage pairs, not on
/// how the interval was constructed.
fn goodness_from_points(points: &[AccuracyPoint]) -> f64 {
    let k = points.len() as f64;
    1.0 - points
        .iter()
        .map(|pt| {
            let diff = pt.observed - pt.nominal;
            let w = if diff >= 0.0 { 1.0 } else { 2.0 };
            w * diff.abs()
        })
        .sum::<f64>()
        / k
}

/// Deutsch's accuracy plot using each held-out location's full ccdf
/// directly, instead of [`accuracy_plot`]'s Gaussian-interval approximation
/// -- Deutsch (1997)'s original formulation is over ccdfs, and indicator
/// kriging / SIS already produce exactly that (AUDIT-2026-07-v2.md §4/§7
/// Fase 6 item #18). For each nominal probability `p`, the symmetric
/// interval `[quantile((1-p)/2), quantile((1+p)/2)]` is read directly off
/// the (order-corrected, tail-extrapolated) ccdf via the same
/// interpolation `sis`'s sequential simulation uses to *draw* from a ccdf
/// -- passing a fixed probability instead of a random one is exactly the
/// inverse-ccdf transform, so no new interpolation logic is needed.
///
/// `ccdfs[i]` must correspond to `actual[i]` and have `cutoffs.len()`
/// entries each; `tail_min`/`tail_max`/`lower_tail`/`upper_tail` are the
/// same tail-extrapolation parameters the ccdf was estimated with (see
/// [`crate::ik::IkConfig`]/[`crate::sis::SisConfig`]).
#[allow(clippy::too_many_arguments)]
pub fn accuracy_plot_ccdf(
    actual: &[f64],
    ccdfs: &[Vec<f64>],
    cutoffs: &[f64],
    tail_min: f64,
    tail_max: f64,
    lower_tail: crate::tails::TailModel,
    upper_tail: crate::tails::TailModel,
    probs: &[f64],
) -> Result<AccuracyPlot> {
    if actual.len() != ccdfs.len() {
        return Err(GeostatError::DimensionMismatch(format!(
            "{} actual values vs {} ccdfs",
            actual.len(),
            ccdfs.len()
        )));
    }
    if actual.is_empty() {
        return Err(GeostatError::InsufficientData(
            "accuracy plot requires at least one held-out point".into(),
        ));
    }
    if ccdfs.iter().any(|c| c.len() != cutoffs.len()) {
        return Err(GeostatError::DimensionMismatch(
            "every ccdf must have one entry per cutoff".into(),
        ));
    }
    if probs.is_empty() || probs.iter().any(|&p| !(p > 0.0 && p < 1.0)) {
        return Err(GeostatError::InvalidParameter(
            "probabilities must be non-empty and strictly inside (0, 1)".into(),
        ));
    }
    let n = actual.len() as f64;
    let points: Vec<AccuracyPoint> = probs
        .iter()
        .map(|&p| {
            let lo_p = (1.0 - p) / 2.0;
            let hi_p = (1.0 + p) / 2.0;
            let inside = actual
                .iter()
                .zip(ccdfs)
                .filter(|&(a, ccdf)| {
                    let lo = crate::sis::sample_ccdf(
                        ccdf, cutoffs, tail_min, tail_max, lower_tail, upper_tail, lo_p,
                    );
                    let hi = crate::sis::sample_ccdf(
                        ccdf, cutoffs, tail_min, tail_max, lower_tail, upper_tail, hi_p,
                    );
                    *a >= lo && *a <= hi
                })
                .count();
            AccuracyPoint {
                nominal: p,
                observed: inside as f64 / n,
            }
        })
        .collect();
    let goodness = goodness_from_points(&points);
    Ok(AccuracyPlot { points, goodness })
}

/// Leave-one-out cross-validation for ordinary co-kriging (Fase 6 item
/// #18): each primary observation is predicted from every other primary
/// observation plus the (unchanged) secondaries. Heterotopic co-kriging's
/// secondaries are exogenous data, not held-out targets, so only the
/// primary is ever excluded -- the same convention GSLIB/gstat use for
/// co-kriging CV.
pub fn leave_one_out_cokriging<const D: usize>(
    datasets: &[PointSet<D>],
    lmc: &crate::cokriging::Lmc,
    config: &crate::cokriging::CoKrigingConfig,
) -> Result<CvResult> {
    if datasets.len() < 2 {
        return Err(GeostatError::InvalidParameter(
            "co-kriging needs a primary and at least one secondary variable".into(),
        ));
    }
    let primary = &datasets[0];
    if primary.len() < 3 {
        return Err(GeostatError::InsufficientData(
            "leave-one-out cross-validation requires at least 3 primary points".into(),
        ));
    }
    let estimates: Vec<(f64, f64)> = par_try_map(primary.len(), |i| {
        let sub_primary = primary.excluding(i);
        let mut subs = vec![sub_primary];
        subs.extend(datasets[1..].iter().cloned());
        let refs: Vec<&PointSet<D>> = subs.iter().collect();
        let ck = crate::cokriging::CoKriging::new(refs, lmc, config.clone())?;
        let est = ck.predict(primary.coord(i))?;
        Ok((est.value, est.variance))
    })?;
    let (predicted, variance) = estimates.into_iter().unzip();
    Ok(CvResult {
        observed: primary.values().to_vec(),
        predicted,
        variance,
    })
}

/// Leave-one-out cross-validation for collocated co-kriging (Fase 6 item
/// #18): each primary observation is predicted from the others plus its
/// own collocated secondary value (`secondary_at_primary[i]`) -- exactly
/// mirroring how [`crate::collocated::CollocatedCokriging::predict`] is
/// used operationally (one secondary value per target).
#[allow(clippy::too_many_arguments)]
pub fn leave_one_out_collocated<const D: usize>(
    primary: &PointSet<D>,
    secondary_at_primary: &[f64],
    model1: &VariogramModel,
    mean1: f64,
    mean2: f64,
    rho12: f64,
    sigma1: f64,
    sigma2: f64,
    markov: &crate::collocated::MarkovModel,
    config: &crate::collocated::CollocatedConfig,
) -> Result<CvResult> {
    if primary.len() < 3 {
        return Err(GeostatError::InsufficientData(
            "leave-one-out cross-validation requires at least 3 points".into(),
        ));
    }
    if secondary_at_primary.len() != primary.len() {
        return Err(GeostatError::DimensionMismatch(format!(
            "{} collocated secondary values vs {} primary points",
            secondary_at_primary.len(),
            primary.len()
        )));
    }
    let estimates: Vec<(f64, f64)> = par_try_map(primary.len(), |i| {
        let sub = primary.excluding(i);
        let cck = crate::collocated::CollocatedCokriging::new(
            &sub,
            model1,
            mean1,
            mean2,
            rho12,
            sigma1,
            sigma2,
            markov.clone(),
            *config,
        )?;
        let est = cck.predict(primary.coord(i), secondary_at_primary[i])?;
        Ok((est.value, est.variance))
    })?;
    let (predicted, variance) = estimates.into_iter().unzip();
    Ok(CvResult {
        observed: primary.values().to_vec(),
        predicted,
        variance,
    })
}

/// Leave-one-out cross-validation for lognormal kriging (Fase 6 item #18).
/// `variance` is the back-transformed **predictive** variance of the
/// lognormal distribution implied by the log-space kriging mean/variance
/// (`(exp(log_var) - 1) * exp(2*log_mean + log_var)`, the standard lognormal
/// variance formula -- not just `log_variance` re-used verbatim, which
/// would be in the wrong units for [`CvResult::msdr`] and friends to
/// compare against real-unit residuals).
pub fn leave_one_out_lognormal<const D: usize>(
    data: &PointSet<D>,
    log_model: &VariogramModel,
    config: &KrigingConfig,
) -> Result<CvResult> {
    if data.len() < 3 {
        return Err(GeostatError::InsufficientData(
            "leave-one-out cross-validation requires at least 3 points".into(),
        ));
    }
    if data.values().iter().any(|&v| !(v > 0.0)) {
        return Err(GeostatError::InvalidParameter(
            "lognormal kriging requires strictly positive data values".into(),
        ));
    }
    let estimates: Vec<(f64, f64)> = par_try_map(data.len(), |i| {
        let sub = data.excluding(i);
        let est = crate::trans::lognormal_kriging(&sub, &[data.coord(i)], log_model, config)?
            .into_iter()
            .next()
            .expect("exactly one target");
        let var = (est.log_variance.exp() - 1.0) * (2.0 * est.log_mean + est.log_variance).exp();
        Ok((est.value, var))
    })?;
    let (predicted, variance) = estimates.into_iter().unzip();
    Ok(CvResult {
        observed: data.values().to_vec(),
        predicted,
        variance,
    })
}

/// Leave-one-out cross-validation result for indicator kriging: one ccdf
/// per held-out point, since IK's whole point is the *distribution*, not a
/// single predicted value that [`CvResult`]'s RMSE-family measures assume.
#[derive(Debug, Clone)]
pub struct IkCvResult {
    /// Observed values, in dataset order.
    pub observed: Vec<f64>,
    /// Held-out ccdf estimate per point, `cutoffs.len()` entries each.
    pub ccdf: Vec<Vec<f64>>,
    /// The cutoffs every ccdf was evaluated at.
    pub cutoffs: Vec<f64>,
}

impl IkCvResult {
    /// Mean Ranked Probability Score (Epstein 1969): for each held-out
    /// point, `sum_k (F(cutoff_k) - 1{actual <= cutoff_k})^2` -- the
    /// standard proper scoring rule for a forecast ccdf against a single
    /// realized value at a discrete set of thresholds (0 = perfect, larger
    /// = worse; a ccdf that is either badly located or badly shaped is
    /// penalized, unlike a bias/coverage check alone).
    pub fn rps(&self) -> f64 {
        let n = self.observed.len() as f64;
        self.observed
            .iter()
            .zip(&self.ccdf)
            .map(|(&obs, ccdf)| {
                self.cutoffs
                    .iter()
                    .zip(ccdf)
                    .map(|(&c, &f)| {
                        let ind = if obs <= c { 1.0 } else { 0.0 };
                        (f - ind).powi(2)
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
            / n
    }
}

/// Leave-one-out cross-validation for indicator kriging (Fase 6 item #18):
/// each point's ccdf is estimated from every other point, then compared
/// against its own held-out value via [`IkCvResult::rps`].
pub fn leave_one_out_indicator<const D: usize>(
    data: &PointSet<D>,
    cfg: &crate::ik::IkConfig,
) -> Result<IkCvResult> {
    if data.len() < 3 {
        return Err(GeostatError::InsufficientData(
            "leave-one-out cross-validation requires at least 3 points".into(),
        ));
    }
    let ccdf: Vec<Vec<f64>> = par_try_map(data.len(), |i| {
        let sub = data.excluding(i);
        let est = crate::ik::indicator_kriging(&sub, &[data.coord(i)], cfg)?
            .into_iter()
            .next()
            .expect("exactly one target");
        Ok(est.ccdf)
    })?;
    Ok(IkCvResult {
        observed: data.values().to_vec(),
        ccdf,
        cutoffs: cfg.cutoffs.clone(),
    })
}

/// One lag's variogram-reproduction check across an ensemble of
/// realizations (SGS/SIS output): the target model's `gamma` at `h`, next
/// to the ensemble mean and standard deviation of the *experimental*
/// `gamma` computed independently on each realization.
#[derive(Debug, Clone, Copy)]
pub struct VariogramQcPoint {
    /// Mean pair distance in this bin (from the target model's own binning).
    pub h: f64,
    /// The target model's semivariance at `h`.
    pub target_gamma: f64,
    /// Ensemble mean of the realizations' experimental semivariance at this
    /// bin. `NaN` if no realization had any pairs in this bin.
    pub mean_gamma: f64,
    /// Ensemble standard deviation across realizations at this bin. `NaN`
    /// with fewer than 2 contributing realizations.
    pub std_gamma: f64,
}

/// Quality-control check of how well an ensemble of realizations
/// reproduces a target variogram model (Fase 6 item #18): every
/// geostatistical simulation workflow re-runs the experimental variogram on
/// its own output informally to sanity-check this; this promotes it to a
/// library API instead of a validation-script one-off.
#[derive(Debug, Clone)]
pub struct VariogramQc {
    /// One point per lag bin, ordered by distance.
    pub points: Vec<VariogramQcPoint>,
}

impl VariogramQc {
    /// Largest relative deviation `|mean_gamma - target_gamma| /
    /// target_gamma` across bins with a well-defined target sill fraction
    /// (skips bins where `target_gamma` is ~0, i.e. right at the origin).
    /// A well-reproduced ensemble should keep this small (a few percent to
    /// a few tens of percent depending on the number of realizations and
    /// how close to the nugget/sill the bin sits); there is no universal
    /// pass/fail threshold, so this is reported as a diagnostic, not
    /// asserted against internally.
    pub fn max_relative_deviation(&self) -> f64 {
        self.points
            .iter()
            .filter(|p| p.target_gamma.abs() > 1e-9 && p.mean_gamma.is_finite())
            .map(|p| (p.mean_gamma - p.target_gamma).abs() / p.target_gamma.abs())
            .fold(0.0, f64::max)
    }
}

/// Computes [`VariogramQc`] for a set of realizations sharing the same
/// `coords` (typically grid cell centers from
/// [`crate::simulation::SgsResult`]/[`crate::sis::sequential_indicator_simulation`]'s
/// output, alongside the `target_model`/`cfg` the simulation was
/// conditioned on): the experimental variogram is recomputed independently
/// on each realization, and its per-bin `gamma` values are aggregated
/// (mean, std) and compared against the target model's own `gamma` at that
/// bin's distance.
pub fn realization_variogram_qc<const D: usize>(
    coords: &[[f64; D]],
    realizations: &[Vec<f64>],
    target_model: &VariogramModel,
    cfg: &crate::variogram::VariogramConfig,
) -> Result<VariogramQc> {
    if realizations.is_empty() {
        return Err(GeostatError::InsufficientData(
            "variogram QC requires at least one realization".into(),
        ));
    }
    if realizations.iter().any(|r| r.len() != coords.len()) {
        return Err(GeostatError::DimensionMismatch(
            "every realization must have one value per coordinate".into(),
        ));
    }
    let n_lags = cfg.n_lags;
    let per_realization: Vec<crate::variogram::ExperimentalVariogram> = realizations
        .iter()
        .map(|r| {
            let ps = PointSet::new(coords.to_vec(), r.clone())?;
            crate::variogram::experimental_variogram(&ps, cfg)
        })
        .collect::<Result<_>>()?;

    let mut h = vec![0.0; n_lags];
    let mut gathered: Vec<Vec<f64>> = vec![Vec::new(); n_lags];
    for ev in &per_realization {
        for (k, b) in ev.bins.iter().enumerate() {
            h[k] = b.h;
            if b.n_pairs > 0 {
                gathered[k].push(b.gamma);
            }
        }
    }
    let points = (0..n_lags)
        .map(|k| {
            let vals = &gathered[k];
            let mean_gamma = if vals.is_empty() {
                f64::NAN
            } else {
                vals.iter().sum::<f64>() / vals.len() as f64
            };
            let std_gamma = if vals.len() < 2 {
                f64::NAN
            } else {
                (vals.iter().map(|&g| (g - mean_gamma).powi(2)).sum::<f64>()
                    / (vals.len() - 1) as f64)
                    .sqrt()
            };
            VariogramQcPoint {
                h: h[k],
                target_gamma: target_model.gamma(h[k]),
                mean_gamma,
                std_gamma,
            }
        })
        .collect();
    Ok(VariogramQc { points })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::tails::TailModel;
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

    fn field_std(data: &PointSet) -> f64 {
        let mean = data.mean();
        (data
            .values()
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / data.len() as f64)
            .sqrt()
    }

    #[test]
    fn leave_one_out_cokriging_beats_the_mean_with_an_informative_secondary() {
        let primary = smooth_field(80, 21);
        // Secondary correlated with the primary at the same locations
        // (collocated, autocorrelated) plus noise -- an informative
        // covariate for co-kriging.
        let mut rng = Rng::new(22);
        let secondary_vals: Vec<f64> = primary
            .values()
            .iter()
            .map(|&v| 2.0 * v + 1.0 + 0.3 * rng.normal())
            .collect();
        let secondary = PointSet::new(primary.coords().to_vec(), secondary_vals).unwrap();

        let lmc = crate::cokriging::Lmc::new(
            vec![vec![0.02, 0.0], vec![0.0, 0.1]],
            vec![crate::cokriging::LmcStructure {
                kind: ModelKind::Spherical,
                range: 25.0,
                anis: None,
                sills: vec![vec![1.0, 1.8], vec![1.8, 4.0]],
            }],
        )
        .unwrap();
        let datasets = [primary.clone(), secondary];
        let cv = leave_one_out_cokriging(
            &datasets,
            &lmc,
            &crate::cokriging::CoKrigingConfig::default(),
        )
        .unwrap();
        assert_eq!(cv.observed.len(), primary.len());
        assert!(cv.rmse().is_finite());
        assert!(
            cv.rmse() < field_std(&primary),
            "cokriging rmse {} should beat naive std {}",
            cv.rmse(),
            field_std(&primary)
        );
    }

    #[test]
    fn leave_one_out_cokriging_rejects_bad_input() {
        let primary = smooth_field(80, 1);
        let lmc = crate::cokriging::Lmc::new(
            vec![vec![0.1]],
            vec![crate::cokriging::LmcStructure {
                kind: ModelKind::Spherical,
                range: 25.0,
                anis: None,
                sills: vec![vec![1.0]],
            }],
        )
        .unwrap();
        // Only one dataset (no secondary).
        assert!(
            leave_one_out_cokriging(
                std::slice::from_ref(&primary),
                &lmc,
                &crate::cokriging::CoKrigingConfig::default()
            )
            .is_err()
        );
    }

    #[test]
    fn leave_one_out_collocated_beats_the_mean_with_an_informative_secondary() {
        let primary = smooth_field(80, 31);
        let mut rng = Rng::new(32);
        let secondary_at_primary: Vec<f64> = primary
            .values()
            .iter()
            .map(|&v| 1.5 * v - 0.5 + 0.2 * rng.normal())
            .collect();
        let (rho12, sigma1, sigma2) =
            crate::collocated::estimate_collocated_stats(primary.values(), &secondary_at_primary)
                .unwrap();
        let model1 =
            VariogramModel::new(0.02, vec![Structure::new(ModelKind::Spherical, 1.0, 25.0)])
                .unwrap();
        let cv = leave_one_out_collocated(
            &primary,
            &secondary_at_primary,
            &model1,
            primary.mean(),
            secondary_at_primary.iter().sum::<f64>() / secondary_at_primary.len() as f64,
            rho12,
            sigma1,
            sigma2,
            &crate::collocated::MarkovModel::Mm1,
            &crate::collocated::CollocatedConfig::default(),
        )
        .unwrap();
        assert!(cv.rmse().is_finite());
        assert!(
            cv.rmse() < field_std(&primary),
            "collocated cokriging rmse {} should beat naive std {}",
            cv.rmse(),
            field_std(&primary)
        );
    }

    #[test]
    fn leave_one_out_collocated_rejects_length_mismatch() {
        let primary = smooth_field(20, 1);
        let model1 =
            VariogramModel::new(0.02, vec![Structure::new(ModelKind::Spherical, 1.0, 25.0)])
                .unwrap();
        assert!(
            leave_one_out_collocated(
                &primary,
                &[1.0, 2.0], // wrong length
                &model1,
                0.0,
                0.0,
                0.5,
                1.0,
                1.0,
                &crate::collocated::MarkovModel::Mm1,
                &crate::collocated::CollocatedConfig::default(),
            )
            .is_err()
        );
    }

    #[test]
    fn leave_one_out_lognormal_beats_the_mean_on_a_smooth_positive_field() {
        let mut rng = Rng::new(41);
        let mut coords = Vec::new();
        let mut logs = Vec::new();
        for _ in 0..80 {
            let x = rng.uniform() * 100.0;
            let y = rng.uniform() * 100.0;
            coords.push([x, y]);
            logs.push((x / 20.0).sin() + (y / 20.0).cos());
        }
        let values: Vec<f64> = logs.iter().map(|&l| l.exp()).collect();
        let data = PointSet::new(coords, values).unwrap();
        let log_model =
            VariogramModel::new(0.02, vec![Structure::new(ModelKind::Spherical, 1.0, 25.0)])
                .unwrap();
        let cfg = KrigingConfig::default();
        let cv = leave_one_out_lognormal(&data, &log_model, &cfg).unwrap();
        assert!(cv.predicted.iter().all(|&p| p.is_finite() && p > 0.0));
        assert!(cv.variance.iter().all(|&v| v.is_finite() && v >= 0.0));
        assert!(
            cv.rmse() < field_std(&data),
            "lognormal cv rmse {} should beat naive std {}",
            cv.rmse(),
            field_std(&data)
        );
    }

    #[test]
    fn leave_one_out_lognormal_rejects_nonpositive_values() {
        let data = PointSet::new(
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            vec![1.0, -2.0, 3.0],
        )
        .unwrap();
        let log_model =
            VariogramModel::new(0.02, vec![Structure::new(ModelKind::Spherical, 1.0, 5.0)])
                .unwrap();
        assert!(leave_one_out_lognormal(&data, &log_model, &KrigingConfig::default()).is_err());
    }

    #[test]
    fn leave_one_out_indicator_rps_is_bounded_and_low_for_a_good_model() {
        let data = smooth_field(80, 51);
        let mut sorted = data.values().to_vec();
        sorted.sort_by(f64::total_cmp);
        let cutoffs = vec![sorted[20], sorted[40], sorted[60]];
        let model =
            VariogramModel::new(0.02, vec![Structure::new(ModelKind::Spherical, 0.2, 25.0)])
                .unwrap();
        let cfg = crate::ik::IkConfig {
            cutoffs: cutoffs.clone(),
            models: vec![model],
            ..Default::default()
        };
        let cv = leave_one_out_indicator(&data, &cfg).unwrap();
        assert_eq!(cv.observed.len(), data.len());
        assert_eq!(cv.ccdf.len(), data.len());
        let rps = cv.rps();
        // Each per-cutoff term is in [0, 1] (a squared difference of two
        // values in [0, 1]), so RPS is bounded by the number of cutoffs.
        assert!(rps.is_finite() && rps >= 0.0 && rps <= cutoffs.len() as f64);
        // A model that tracks the data at all should clearly beat the
        // "always predict the global proportion" baseline (RPS close to
        // its worst-case ceiling would indicate the model learned nothing).
        assert!(rps < 0.5, "rps {rps} too high for an informative model");
    }

    #[test]
    fn leave_one_out_indicator_rejects_too_few_points() {
        let data = PointSet::new(vec![[0.0, 0.0], [1.0, 0.0]], vec![1.0, 2.0]).unwrap();
        let cfg = crate::ik::IkConfig {
            cutoffs: vec![1.5],
            models: vec![
                VariogramModel::new(0.0, vec![Structure::new(ModelKind::Spherical, 0.25, 5.0)])
                    .unwrap(),
            ],
            ..Default::default()
        };
        assert!(leave_one_out_indicator(&data, &cfg).is_err());
    }

    #[test]
    fn accuracy_plot_ccdf_matches_gaussian_accuracy_plot_for_gaussian_ccdfs() {
        // Build ccdfs that are exactly the Gaussian CDF at each cutoff for a
        // known (mean, std): `accuracy_plot_ccdf` should then agree closely
        // with `accuracy_plot` run on the same (mean, std) pairs, since both
        // describe the same underlying distribution -- just through
        // different representations (closed-form Gaussian vs. a
        // cutoff-sampled ccdf with linear interpolation).
        let mut rng = Rng::new(61);
        let n = 3000;
        let mean = 0.0;
        let std = 1.0;
        let actual: Vec<f64> = (0..n).map(|_| mean + std * rng.normal()).collect();
        let cutoffs: Vec<f64> = (-40..=40).map(|k| k as f64 * 0.1).collect();
        let ccdfs: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                cutoffs
                    .iter()
                    .map(|&c| crate::transform::norm_cdf((c - mean) / std))
                    .collect()
            })
            .collect();
        let probs = vec![0.5, 0.8];
        let tail_min = -8.0;
        let tail_max = 8.0;
        let plot_ccdf = accuracy_plot_ccdf(
            &actual,
            &ccdfs,
            &cutoffs,
            tail_min,
            tail_max,
            TailModel::Linear,
            TailModel::Linear,
            &probs,
        )
        .unwrap();
        let means = vec![mean; n];
        let stds = vec![std; n];
        let plot_gaussian = accuracy_plot(&actual, &means, &stds, &probs).unwrap();
        for (a, b) in plot_ccdf.points.iter().zip(&plot_gaussian.points) {
            assert!(
                (a.observed - b.observed).abs() < 0.03,
                "ccdf {} vs gaussian {} at p={}",
                a.observed,
                b.observed,
                a.nominal
            );
        }
    }

    #[test]
    fn accuracy_plot_ccdf_rejects_bad_input() {
        assert!(
            accuracy_plot_ccdf(
                &[1.0],
                &[vec![0.5], vec![0.5]], // length mismatch vs actual
                &[1.0],
                0.0,
                2.0,
                TailModel::Linear,
                TailModel::Linear,
                &[0.5],
            )
            .is_err()
        );
        assert!(
            accuracy_plot_ccdf(
                &[1.0],
                &[vec![0.5, 0.6]], // wrong number of cutoffs
                &[1.0],
                0.0,
                2.0,
                TailModel::Linear,
                TailModel::Linear,
                &[0.5],
            )
            .is_err()
        );
    }

    #[test]
    fn realization_variogram_qc_tracks_the_conditioning_model() {
        // A real SGS ensemble should, on average, reproduce the model it
        // was conditioned on -- the whole point of the QC check.
        let data = smooth_field(60, 71);
        let model = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 0.95, 25.0)],
        )
        .unwrap();
        let grid = crate::grid::Grid2D::from_bbox([0.0, 0.0], [100.0, 100.0], 20, 20).unwrap();
        let cfg = crate::simulation::SgsConfig {
            n_realizations: 30,
            seed: 7,
            max_neighbors: 16,
            ..Default::default()
        };
        let res =
            crate::simulation::sequential_gaussian_simulation(&data, &model, &grid, &cfg).unwrap();

        let vcfg = crate::variogram::VariogramConfig {
            n_lags: 8,
            max_dist: 50.0,
            direction: None,
        };
        let qc =
            realization_variogram_qc(&grid.centers(), &res.realizations, &model, &vcfg).unwrap();
        assert_eq!(qc.points.len(), 8);
        // Short lags (well within the range, plenty of pairs) should track
        // the model reasonably closely on average across 30 realizations.
        let short_lag = &qc.points[1];
        assert!(short_lag.mean_gamma.is_finite());
        assert!(
            (short_lag.mean_gamma - short_lag.target_gamma).abs() < 0.3,
            "mean {} vs target {} at h={}",
            short_lag.mean_gamma,
            short_lag.target_gamma,
            short_lag.h
        );
        assert!(qc.max_relative_deviation().is_finite());
    }

    #[test]
    fn realization_variogram_qc_rejects_bad_input() {
        let model = VariogramModel::new(0.0, vec![Structure::new(ModelKind::Spherical, 1.0, 10.0)])
            .unwrap();
        let cfg = crate::variogram::VariogramConfig {
            n_lags: 4,
            max_dist: 10.0,
            direction: None,
        };
        let coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        assert!(realization_variogram_qc(&coords, &[], &model, &cfg).is_err()); // no realizations
        assert!(
            realization_variogram_qc(&coords, &[vec![1.0, 2.0]], &model, &cfg).is_err() // wrong length
        );
    }
}
