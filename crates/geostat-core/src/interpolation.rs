//! Mathematical (non-geostatistical) interpolators: inverse-distance
//! weighting and k-nearest-neighbor averaging.
//!
//! These are deliberately simple baselines. The central message of Li (2021)
//! is that no spatial predictor dominates — kriging does not always win — so a
//! fair benchmark needs cheap, assumption-light references to compare against.
//! Both run over `PointSet<D>` (2-D or 3-D) and reuse the kd-tree moving
//! neighborhood, and both expose a leave-one-out cross-validation that returns
//! a [`CvResult`], so they score on the same VEcv/RMSE footing as kriging.

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::parallel::par_map;
use crate::search::KdTree;
use crate::validation::CvResult;

fn distance<const D: usize>(a: [f64; D], b: [f64; D]) -> f64 {
    (0..D).map(|d| (a[d] - b[d]).powi(2)).sum::<f64>().sqrt()
}

/// Inverse-distance weighting: `z*(x0) = Σ w_i z_i / Σ w_i`, with
/// `w_i = 1 / d_i^power`. Exact at the data (a coincident point returns its
/// own value). `power` controls locality (larger = more local; 2 is typical).
#[derive(Debug)]
pub struct Idw<'a, const D: usize = 2> {
    data: &'a PointSet<D>,
    tree: KdTree<D>,
    power: f64,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
}

impl<'a, const D: usize> Idw<'a, D> {
    /// Builds an IDW predictor. With `max_neighbors`/`radius` unset, every
    /// datum contributes to each estimate.
    pub fn new(
        data: &'a PointSet<D>,
        power: f64,
        max_neighbors: Option<usize>,
        radius: Option<f64>,
    ) -> Result<Self> {
        if !(power > 0.0) {
            return Err(GeostatError::InvalidParameter(
                "IDW power must be positive".into(),
            ));
        }
        if data.is_empty() {
            return Err(GeostatError::InsufficientData("IDW needs ≥ 1 point".into()));
        }
        Ok(Self {
            tree: KdTree::build(data.coords()),
            data,
            power,
            max_neighbors,
            radius,
        })
    }

    /// Predicts at a target. Returns NaN if no point lies within the search
    /// neighborhood.
    pub fn predict(&self, target: [f64; D]) -> f64 {
        let k = self.max_neighbors.unwrap_or_else(|| self.data.len());
        let idx = self.tree.k_nearest(target, k, self.radius);
        if idx.is_empty() {
            return f64::NAN;
        }
        let mut num = 0.0;
        let mut den = 0.0;
        for i in idx {
            let d = distance(target, self.data.coord(i));
            if d <= 1e-12 {
                return self.data.value(i); // exact interpolation at a datum
            }
            let w = d.powf(-self.power);
            num += w * self.data.value(i);
            den += w;
        }
        num / den
    }

    /// Predicts at many targets (in parallel when the `parallel` feature is on).
    pub fn predict_many(&self, targets: &[[f64; D]]) -> Vec<f64> {
        par_map(targets.len(), |i| self.predict(targets[i]))
    }
}

/// k-nearest-neighbor averaging: the mean of the `k` nearest observed values
/// (`k = 1` is nearest-neighbor / Voronoi / Thiessen prediction).
#[derive(Debug)]
pub struct Knn<'a, const D: usize = 2> {
    data: &'a PointSet<D>,
    tree: KdTree<D>,
    k: usize,
    radius: Option<f64>,
}

impl<'a, const D: usize> Knn<'a, D> {
    /// Builds a k-NN predictor (`k ≥ 1`).
    pub fn new(data: &'a PointSet<D>, k: usize, radius: Option<f64>) -> Result<Self> {
        if k == 0 {
            return Err(GeostatError::InvalidParameter("k must be ≥ 1".into()));
        }
        if data.is_empty() {
            return Err(GeostatError::InsufficientData(
                "k-NN needs ≥ 1 point".into(),
            ));
        }
        Ok(Self {
            tree: KdTree::build(data.coords()),
            data,
            k,
            radius,
        })
    }

    /// Predicts at a target as the mean of the `k` nearest values. Returns NaN
    /// if no point lies within the search neighborhood.
    pub fn predict(&self, target: [f64; D]) -> f64 {
        let idx = self.tree.k_nearest(target, self.k, self.radius);
        if idx.is_empty() {
            return f64::NAN;
        }
        let sum: f64 = idx.iter().map(|&i| self.data.value(i)).sum();
        sum / idx.len() as f64
    }

    /// Predicts at many targets (in parallel when the `parallel` feature is on).
    pub fn predict_many(&self, targets: &[[f64; D]]) -> Vec<f64> {
        par_map(targets.len(), |i| self.predict(targets[i]))
    }
}

/// Leave-one-out cross-validation for IDW: each point is predicted from all
/// others. The returned [`CvResult`] has NaN variances (IDW has no estimation
/// variance), so MSDR is undefined while VEcv/RMSE/etc. are not.
pub fn idw_cross_validate<const D: usize>(
    data: &PointSet<D>,
    power: f64,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
) -> Result<CvResult> {
    if data.len() < 3 {
        return Err(GeostatError::InsufficientData(
            "cross-validation requires at least 3 points".into(),
        ));
    }
    // Validate parameters once up front.
    Idw::new(data, power, max_neighbors, radius)?;
    let predicted = par_map(data.len(), |i| {
        let sub = data.excluding(i);
        match Idw::new(&sub, power, max_neighbors, radius) {
            Ok(idw) => idw.predict(data.coord(i)),
            Err(_) => f64::NAN,
        }
    });
    Ok(CvResult {
        observed: data.values().to_vec(),
        predicted,
        variance: vec![f64::NAN; data.len()],
    })
}

/// Leave-one-out cross-validation for k-NN averaging (see
/// [`idw_cross_validate`] for the variance caveat).
pub fn knn_cross_validate<const D: usize>(
    data: &PointSet<D>,
    k: usize,
    radius: Option<f64>,
) -> Result<CvResult> {
    if data.len() < 3 {
        return Err(GeostatError::InsufficientData(
            "cross-validation requires at least 3 points".into(),
        ));
    }
    Knn::new(data, k, radius)?;
    let predicted = par_map(data.len(), |i| {
        let sub = data.excluding(i);
        match Knn::new(&sub, k, radius) {
            Ok(knn) => knn.predict(data.coord(i)),
            Err(_) => f64::NAN,
        }
    });
    Ok(CvResult {
        observed: data.values().to_vec(),
        predicted,
        variance: vec![f64::NAN; data.len()],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn smooth_field(n: usize, seed: u64) -> PointSet {
        let mut rng = Rng::new(seed);
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for _ in 0..n {
            let x = rng.uniform() * 100.0;
            let y = rng.uniform() * 100.0;
            coords.push([x, y]);
            values.push((x / 20.0).sin() + (y / 25.0).cos());
        }
        PointSet::new(coords, values).unwrap()
    }

    #[test]
    fn idw_is_exact_at_data_points() {
        let data = smooth_field(60, 1);
        let idw = Idw::new(&data, 2.0, None, None).unwrap();
        for i in (0..data.len()).step_by(10) {
            let p = idw.predict(data.coord(i));
            assert!((p - data.value(i)).abs() < 1e-9, "{p} vs {}", data.value(i));
        }
    }

    #[test]
    fn knn_k1_returns_nearest_value() {
        let data = smooth_field(50, 2);
        let nn = Knn::new(&data, 1, None).unwrap();
        // A target nudged slightly off a datum returns that datum's value.
        for i in (0..data.len()).step_by(10) {
            let c = data.coord(i);
            let near = [c[0] + 0.01, c[1] - 0.01];
            assert!((nn.predict(near) - data.value(i)).abs() < 1e-12);
        }
    }

    #[test]
    fn idw_and_knn_beat_the_mean_on_smooth_field() {
        let data = smooth_field(150, 5);
        let idw_cv = idw_cross_validate(&data, 2.0, Some(12), None).unwrap();
        let knn_cv = knn_cross_validate(&data, 6, None).unwrap();
        // VEcv > 0 means better than predicting the global mean.
        assert!(idw_cv.vecv() > 0.0, "idw vecv {}", idw_cv.vecv());
        assert!(knn_cv.vecv() > 0.0, "knn vecv {}", knn_cv.vecv());
        // MSDR is undefined (no estimation variance).
        assert!(idw_cv.msdr().is_nan());
    }

    #[test]
    fn idw_power_must_be_positive() {
        let data = smooth_field(10, 3);
        assert!(Idw::new(&data, 0.0, None, None).is_err());
        assert!(Knn::new(&data, 0, None).is_err());
    }
}
