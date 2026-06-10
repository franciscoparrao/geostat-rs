//! Simple, ordinary and universal kriging.

use ndarray::Array2;
use rayon::prelude::*;

use crate::data::{PointSet, dist, k_nearest};
use crate::error::{GeostatError, Result};
use crate::grid::Grid2D;
use crate::linalg::solve;
use crate::variogram::VariogramModel;

/// Kriging flavor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KrigingMethod {
    /// Simple kriging with a known stationary mean.
    Simple {
        /// Known mean of the field.
        mean: f64,
    },
    /// Ordinary kriging (unknown constant mean).
    Ordinary,
    /// Universal kriging with a polynomial drift in the coordinates.
    Universal {
        /// Drift polynomial degree (1 = linear, 2 = quadratic).
        degree: u8,
    },
}

impl KrigingMethod {
    /// Number of drift basis functions (0 for simple kriging).
    fn n_drift(self) -> usize {
        match self {
            KrigingMethod::Simple { .. } => 0,
            KrigingMethod::Ordinary => 1,
            KrigingMethod::Universal { degree: 1 } => 3,
            KrigingMethod::Universal { degree: 2 } => 6,
            KrigingMethod::Universal { .. } => 0,
        }
    }
}

/// Kriging configuration: method plus search neighborhood.
///
/// With both `max_neighbors` and `search_radius` unset, a global
/// neighborhood (all data points) is used.
#[derive(Debug, Clone, PartialEq)]
pub struct KrigingConfig {
    /// Kriging flavor.
    pub method: KrigingMethod,
    /// Maximum number of nearest conditioning points per estimate.
    pub max_neighbors: Option<usize>,
    /// Maximum search distance for conditioning points.
    pub search_radius: Option<f64>,
}

impl Default for KrigingConfig {
    fn default() -> Self {
        Self {
            method: KrigingMethod::Ordinary,
            max_neighbors: None,
            search_radius: None,
        }
    }
}

/// A kriging estimate: predicted value and kriging variance.
#[derive(Debug, Clone, Copy)]
pub struct KrigingEstimate {
    /// Predicted value.
    pub value: f64,
    /// Kriging (estimation) variance, clamped at zero.
    pub variance: f64,
}

/// Kriging predictor bound to a dataset and variogram model.
#[derive(Debug)]
pub struct Kriging<'a> {
    data: &'a PointSet,
    model: &'a VariogramModel,
    config: KrigingConfig,
    // Coordinate normalization for numerically stable drift terms.
    cx: f64,
    cy: f64,
    scale: f64,
}

impl<'a> Kriging<'a> {
    /// Builds a predictor, validating the configuration.
    pub fn new(
        data: &'a PointSet,
        model: &'a VariogramModel,
        config: KrigingConfig,
    ) -> Result<Self> {
        if let KrigingMethod::Universal { degree } = config.method
            && !(1..=2).contains(&degree)
        {
            return Err(GeostatError::InvalidParameter(format!(
                "universal kriging drift degree must be 1 or 2, got {degree}"
            )));
        }
        if let Some(r) = config.search_radius
            && !(r > 0.0)
        {
            return Err(GeostatError::InvalidParameter(format!(
                "search radius must be positive, got {r}"
            )));
        }
        if config.max_neighbors == Some(0) {
            return Err(GeostatError::InvalidParameter(
                "max_neighbors must be at least 1".into(),
            ));
        }
        let (min, max) = data.bbox();
        let cx = 0.5 * (min[0] + max[0]);
        let cy = 0.5 * (min[1] + max[1]);
        let scale = 0.5 * ((max[0] - min[0]).max(max[1] - min[1])).max(f64::MIN_POSITIVE);
        Ok(Self {
            data,
            model,
            config,
            cx,
            cy,
            scale,
        })
    }

    fn fill_basis(&self, p: [f64; 2], out: &mut [f64]) {
        match self.config.method {
            KrigingMethod::Simple { .. } => {}
            KrigingMethod::Ordinary => out[0] = 1.0,
            KrigingMethod::Universal { degree } => {
                let x = (p[0] - self.cx) / self.scale;
                let y = (p[1] - self.cy) / self.scale;
                out[0] = 1.0;
                out[1] = x;
                out[2] = y;
                if degree == 2 {
                    out[3] = x * x;
                    out[4] = x * y;
                    out[5] = y * y;
                }
            }
        }
    }

    fn neighbors(&self, target: [f64; 2]) -> Vec<usize> {
        match (self.config.max_neighbors, self.config.search_radius) {
            (None, None) => (0..self.data.len()).collect(),
            (k, r) => k_nearest(self.data.coords(), target, k.unwrap_or(self.data.len()), r),
        }
    }

    /// Kriging estimate at a single target location.
    pub fn predict(&self, target: [f64; 2]) -> Result<KrigingEstimate> {
        let nb = self.neighbors(target);
        if nb.is_empty() {
            return Err(GeostatError::NoNeighbors);
        }
        let m = self.config.method.n_drift();
        let n = nb.len();
        if n < m {
            return Err(GeostatError::InsufficientData(format!(
                "{n} neighbors cannot support {m} drift terms"
            )));
        }
        let dim = n + m;
        let c0 = self.model.covariance(0.0);

        let mut a = Array2::<f64>::zeros((dim, dim));
        let mut b = vec![0.0; dim];
        let mut f = vec![0.0; m];
        for (ii, &i) in nb.iter().enumerate() {
            let pi = self.data.coord(i);
            a[[ii, ii]] = c0;
            for (jj, &j) in nb.iter().enumerate().skip(ii + 1) {
                let c = self.model.covariance(dist(pi, self.data.coord(j)));
                a[[ii, jj]] = c;
                a[[jj, ii]] = c;
            }
            self.fill_basis(pi, &mut f);
            for (k, &fk) in f.iter().enumerate() {
                a[[ii, n + k]] = fk;
                a[[n + k, ii]] = fk;
            }
            b[ii] = self.model.covariance(dist(pi, target));
        }
        self.fill_basis(target, &mut f);
        for (k, &fk) in f.iter().enumerate() {
            b[n + k] = fk;
        }

        let b0 = b.clone();
        let w = solve(a, b)?;

        let (value, variance) = match self.config.method {
            KrigingMethod::Simple { mean } => {
                let mut v = mean;
                let mut reduction = 0.0;
                for ii in 0..n {
                    v += w[ii] * (self.data.value(nb[ii]) - mean);
                    reduction += w[ii] * b0[ii];
                }
                (v, c0 - reduction)
            }
            _ => {
                let v: f64 = (0..n).map(|ii| w[ii] * self.data.value(nb[ii])).sum();
                // Includes the Lagrange-multiplier terms via b0[n..].
                let reduction: f64 = (0..dim).map(|i| w[i] * b0[i]).sum();
                (v, c0 - reduction)
            }
        };

        Ok(KrigingEstimate {
            value,
            variance: variance.max(0.0),
        })
    }

    /// Kriging estimates at many targets, in parallel. Targets whose system
    /// fails (e.g. no neighbors) yield NaN estimates.
    pub fn predict_many(&self, targets: &[[f64; 2]]) -> Vec<KrigingEstimate> {
        targets
            .par_iter()
            .map(|&t| {
                self.predict(t).unwrap_or(KrigingEstimate {
                    value: f64::NAN,
                    variance: f64::NAN,
                })
            })
            .collect()
    }

    /// Kriging over all cell centers of a grid. Returns `(values, variances)`
    /// in grid storage order.
    pub fn predict_grid(&self, grid: &Grid2D) -> (Vec<f64>, Vec<f64>) {
        let ests = self.predict_many(&grid.centers());
        ests.into_iter().map(|e| (e.value, e.variance)).unzip()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::{ModelKind, Structure};

    fn sample_data() -> PointSet {
        PointSet::new(
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.6]],
            vec![1.0, 2.0, 1.5, 2.5, 1.7],
        )
        .unwrap()
    }

    fn model() -> VariogramModel {
        VariogramModel::new(0.05, vec![Structure::new(ModelKind::Spherical, 0.95, 2.0)]).unwrap()
    }

    #[test]
    fn ordinary_kriging_is_exact_at_data() {
        let data = sample_data();
        let m = model();
        let k = Kriging::new(&data, &m, KrigingConfig::default()).unwrap();
        for i in 0..data.len() {
            let est = k.predict(data.coord(i)).unwrap();
            assert!(
                (est.value - data.value(i)).abs() < 1e-8,
                "point {i}: {} vs {}",
                est.value,
                data.value(i)
            );
            assert!(est.variance < 1e-8, "point {i}: var {}", est.variance);
        }
    }

    #[test]
    fn ordinary_kriging_far_field_variance() {
        let data = sample_data();
        let m = model();
        let k = Kriging::new(&data, &m, KrigingConfig::default()).unwrap();
        let est = k.predict([1e6, 1e6]).unwrap();
        // Beyond the range, OK variance >= total sill.
        assert!(est.variance >= 0.99 * m.total_sill());
        assert!(est.value.is_finite());
    }

    #[test]
    fn simple_kriging_far_field_returns_mean() {
        let data = sample_data();
        let m = model();
        let cfg = KrigingConfig {
            method: KrigingMethod::Simple { mean: 1.74 },
            ..Default::default()
        };
        let k = Kriging::new(&data, &m, cfg).unwrap();
        let est = k.predict([1e6, 1e6]).unwrap();
        assert!((est.value - 1.74).abs() < 1e-9);
        assert!((est.variance - m.total_sill()).abs() < 1e-9);
    }

    #[test]
    fn universal_kriging_reproduces_exact_drift() {
        // Data lying exactly on the plane z = 2 + 3x - y.
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let x = i as f64;
                let y = j as f64;
                coords.push([x, y]);
                values.push(2.0 + 3.0 * x - y);
            }
        }
        let data = PointSet::new(coords, values).unwrap();
        let m = model();
        let cfg = KrigingConfig {
            method: KrigingMethod::Universal { degree: 1 },
            ..Default::default()
        };
        let k = Kriging::new(&data, &m, cfg).unwrap();
        // Inside and outside the convex hull.
        for target in [[1.5, 1.5], [5.0, 2.0]] {
            let expected = 2.0 + 3.0 * target[0] - target[1];
            let est = k.predict(target).unwrap();
            assert!(
                (est.value - expected).abs() < 1e-7,
                "{target:?}: {} vs {expected}",
                est.value
            );
        }
    }

    #[test]
    fn neighborhood_limits_apply() {
        let data = sample_data();
        let m = model();
        let cfg = KrigingConfig {
            method: KrigingMethod::Ordinary,
            max_neighbors: Some(3),
            search_radius: Some(0.001),
        };
        let k = Kriging::new(&data, &m, cfg).unwrap();
        // No data within 1mm of this target.
        assert!(matches!(
            k.predict([0.5, 0.2]),
            Err(GeostatError::NoNeighbors)
        ));
        // At a datum, its own location qualifies.
        let est = k.predict(data.coord(0)).unwrap();
        assert!((est.value - data.value(0)).abs() < 1e-8);
    }

    #[test]
    fn grid_prediction_shapes_and_nonnegative_variance() {
        let data = sample_data();
        let m = model();
        let k = Kriging::new(&data, &m, KrigingConfig::default()).unwrap();
        let grid = Grid2D::from_bbox([0.0, 0.0], [1.0, 1.0], 5, 4).unwrap();
        let (vals, vars) = k.predict_grid(&grid);
        assert_eq!(vals.len(), 20);
        assert_eq!(vars.len(), 20);
        assert!(vals.iter().all(|v| v.is_finite()));
        assert!(vars.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn rejects_invalid_config() {
        let data = sample_data();
        let m = model();
        assert!(
            Kriging::new(
                &data,
                &m,
                KrigingConfig {
                    method: KrigingMethod::Universal { degree: 3 },
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            Kriging::new(
                &data,
                &m,
                KrigingConfig {
                    max_neighbors: Some(0),
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}
