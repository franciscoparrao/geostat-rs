//! Simple, ordinary, universal and external-drift kriging.

use ndarray::Array2;

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::grid::Grid2D;
use crate::linalg::solve;
use crate::search::KdTree;
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
    /// Kriging with external drift: the mean is a linear function of
    /// `n_vars` known covariates supplied for both data and targets.
    ExternalDrift {
        /// Number of external drift variables.
        n_vars: usize,
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
            KrigingMethod::ExternalDrift { n_vars } => 1 + n_vars,
        }
    }
}

/// Kriging configuration: method plus search neighborhood.
///
/// With both `max_neighbors` and `search_radius` unset, a global
/// neighborhood (all data points) is used. Neighbor distances are
/// Euclidean even for anisotropic models (gstat/GSLIB behavior).
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

const NAN_ESTIMATE: KrigingEstimate = KrigingEstimate {
    value: f64::NAN,
    variance: f64::NAN,
};

/// Kriging predictor bound to a dataset and variogram model.
#[derive(Debug)]
pub struct Kriging<'a> {
    data: &'a PointSet,
    model: &'a VariogramModel,
    config: KrigingConfig,
    tree: Option<KdTree>,
    // Coordinate normalization for numerically stable polynomial drift.
    cx: f64,
    cy: f64,
    scale: f64,
    // External drift values per data point, plus per-variable normalization.
    drift_data: Vec<Vec<f64>>,
    drift_mean: Vec<f64>,
    drift_scale: Vec<f64>,
}

impl<'a> Kriging<'a> {
    /// Builds a predictor for simple/ordinary/universal kriging.
    pub fn new(
        data: &'a PointSet,
        model: &'a VariogramModel,
        config: KrigingConfig,
    ) -> Result<Self> {
        if matches!(config.method, KrigingMethod::ExternalDrift { .. }) {
            return Err(GeostatError::InvalidParameter(
                "external drift kriging requires Kriging::with_external_drift".into(),
            ));
        }
        Self::build(data, model, config, Vec::new())
    }

    /// Builds an external-drift kriging predictor. `drift_data[i]` holds the
    /// covariate values at data point `i` and must have `n_vars` entries
    /// matching `KrigingMethod::ExternalDrift`.
    pub fn with_external_drift(
        data: &'a PointSet,
        model: &'a VariogramModel,
        config: KrigingConfig,
        drift_data: Vec<Vec<f64>>,
    ) -> Result<Self> {
        let KrigingMethod::ExternalDrift { n_vars } = config.method else {
            return Err(GeostatError::InvalidParameter(
                "with_external_drift requires KrigingMethod::ExternalDrift".into(),
            ));
        };
        if n_vars == 0 {
            return Err(GeostatError::InvalidParameter(
                "external drift requires at least one covariate".into(),
            ));
        }
        if drift_data.len() != data.len() {
            return Err(GeostatError::DimensionMismatch(format!(
                "{} drift rows vs {} data points",
                drift_data.len(),
                data.len()
            )));
        }
        for (i, row) in drift_data.iter().enumerate() {
            if row.len() != n_vars {
                return Err(GeostatError::DimensionMismatch(format!(
                    "drift row {i} has {} values, expected {n_vars}",
                    row.len()
                )));
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(GeostatError::InvalidParameter(format!(
                    "non-finite drift value at data point {i}"
                )));
            }
        }
        Self::build(data, model, config, drift_data)
    }

    fn build(
        data: &'a PointSet,
        model: &'a VariogramModel,
        config: KrigingConfig,
        drift_data: Vec<Vec<f64>>,
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

        // Per-covariate normalization for a well-conditioned drift block.
        let n_vars = drift_data.first().map_or(0, Vec::len);
        let mut drift_mean = vec![0.0; n_vars];
        let mut drift_scale = vec![1.0; n_vars];
        if n_vars > 0 {
            let n = drift_data.len() as f64;
            for k in 0..n_vars {
                let mean = drift_data.iter().map(|r| r[k]).sum::<f64>() / n;
                let var = drift_data
                    .iter()
                    .map(|r| (r[k] - mean).powi(2))
                    .sum::<f64>()
                    / n;
                drift_mean[k] = mean;
                drift_scale[k] = var.sqrt().max(f64::MIN_POSITIVE);
            }
        }

        let tree = (config.max_neighbors.is_some() || config.search_radius.is_some())
            .then(|| KdTree::build(data.coords()));

        Ok(Self {
            data,
            model,
            config,
            tree,
            cx,
            cy,
            scale,
            drift_data,
            drift_mean,
            drift_scale,
        })
    }

    fn fill_basis(&self, p: [f64; 2], drift: Option<&[f64]>, out: &mut [f64]) {
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
            KrigingMethod::ExternalDrift { n_vars } => {
                let drift = drift.expect("external drift values required");
                out[0] = 1.0;
                for k in 0..n_vars {
                    out[1 + k] = (drift[k] - self.drift_mean[k]) / self.drift_scale[k];
                }
            }
        }
    }

    fn neighbors(&self, target: [f64; 2]) -> Vec<usize> {
        match &self.tree {
            None => (0..self.data.len()).collect(),
            Some(tree) => tree.k_nearest(
                target,
                self.config.max_neighbors.unwrap_or(self.data.len()),
                self.config.search_radius,
            ),
        }
    }

    /// Kriging estimate at a single target location (simple/ordinary/
    /// universal kriging; external drift needs [`Kriging::predict_with_drift`]).
    pub fn predict(&self, target: [f64; 2]) -> Result<KrigingEstimate> {
        if matches!(self.config.method, KrigingMethod::ExternalDrift { .. }) {
            return Err(GeostatError::InvalidParameter(
                "external drift kriging requires predict_with_drift".into(),
            ));
        }
        self.predict_inner(target, None)
    }

    /// External-drift kriging estimate: `target_drift` holds the covariate
    /// values at the target location.
    pub fn predict_with_drift(
        &self,
        target: [f64; 2],
        target_drift: &[f64],
    ) -> Result<KrigingEstimate> {
        let KrigingMethod::ExternalDrift { n_vars } = self.config.method else {
            return Err(GeostatError::InvalidParameter(
                "predict_with_drift requires KrigingMethod::ExternalDrift".into(),
            ));
        };
        if target_drift.len() != n_vars {
            return Err(GeostatError::DimensionMismatch(format!(
                "target drift has {} values, expected {n_vars}",
                target_drift.len()
            )));
        }
        self.predict_inner(target, Some(target_drift))
    }

    fn predict_inner(
        &self,
        target: [f64; 2],
        target_drift: Option<&[f64]>,
    ) -> Result<KrigingEstimate> {
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
        let c0 = self.model.covariance_dh([0.0, 0.0]);

        let mut a = Array2::<f64>::zeros((dim, dim));
        let mut b = vec![0.0; dim];
        let mut f = vec![0.0; m];
        for (ii, &i) in nb.iter().enumerate() {
            let pi = self.data.coord(i);
            a[[ii, ii]] = c0;
            for (jj, &j) in nb.iter().enumerate().skip(ii + 1) {
                let pj = self.data.coord(j);
                let c = self.model.covariance_dh([pi[0] - pj[0], pi[1] - pj[1]]);
                a[[ii, jj]] = c;
                a[[jj, ii]] = c;
            }
            self.fill_basis(pi, self.drift_data.get(i).map(Vec::as_slice), &mut f);
            for (k, &fk) in f.iter().enumerate() {
                a[[ii, n + k]] = fk;
                a[[n + k, ii]] = fk;
            }
            b[ii] = self
                .model
                .covariance_dh([pi[0] - target[0], pi[1] - target[1]]);
        }
        self.fill_basis(target, target_drift, &mut f);
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
        crate::parallel::par_map(targets.len(), |i| {
            self.predict(targets[i]).unwrap_or(NAN_ESTIMATE)
        })
    }

    /// External-drift estimates at many targets, in parallel. `drifts[i]`
    /// holds the covariates of `targets[i]`.
    pub fn predict_many_with_drift(
        &self,
        targets: &[[f64; 2]],
        drifts: &[Vec<f64>],
    ) -> Result<Vec<KrigingEstimate>> {
        if targets.len() != drifts.len() {
            return Err(GeostatError::DimensionMismatch(format!(
                "{} targets vs {} drift rows",
                targets.len(),
                drifts.len()
            )));
        }
        Ok(crate::parallel::par_map(targets.len(), |i| {
            self.predict_with_drift(targets[i], &drifts[i])
                .unwrap_or(NAN_ESTIMATE)
        }))
    }

    /// Kriging over all cell centers of a grid. Returns `(values, variances)`
    /// in grid storage order.
    pub fn predict_grid(&self, grid: &Grid2D) -> (Vec<f64>, Vec<f64>) {
        let ests = self.predict_many(&grid.centers());
        ests.into_iter().map(|e| (e.value, e.variance)).unzip()
    }

    /// Block kriging estimate: predicts the average of a block centered at
    /// `center`, discretized at `center + offsets[u]`. Supported for simple
    /// and ordinary kriging (polynomial or external drift would require
    /// averaging the drift over the block).
    pub fn predict_block(&self, center: [f64; 2], offsets: &[[f64; 2]]) -> Result<KrigingEstimate> {
        match self.config.method {
            KrigingMethod::Simple { .. } | KrigingMethod::Ordinary => {}
            _ => {
                return Err(GeostatError::InvalidParameter(
                    "block kriging supports simple and ordinary kriging only".into(),
                ));
            }
        }
        if offsets.is_empty() {
            return Err(GeostatError::InvalidParameter(
                "block discretization needs at least one point".into(),
            ));
        }
        let nb = self.neighbors(center);
        if nb.is_empty() {
            return Err(GeostatError::NoNeighbors);
        }
        let m = self.config.method.n_drift();
        let n = nb.len();
        let dim = n + m;
        let c0 = self.model.covariance_dh([0.0, 0.0]);
        let nu = offsets.len() as f64;

        let mut a = Array2::<f64>::zeros((dim, dim));
        let mut b = vec![0.0; dim];
        for (ii, &i) in nb.iter().enumerate() {
            let pi = self.data.coord(i);
            a[[ii, ii]] = c0;
            for (jj, &j) in nb.iter().enumerate().skip(ii + 1) {
                let pj = self.data.coord(j);
                let c = self.model.covariance_dh([pi[0] - pj[0], pi[1] - pj[1]]);
                a[[ii, jj]] = c;
                a[[jj, ii]] = c;
            }
            if m == 1 {
                a[[ii, n]] = 1.0;
                a[[n, ii]] = 1.0;
            }
            // Point-to-block covariance: average over discretization points.
            b[ii] = offsets
                .iter()
                .map(|off| {
                    self.model
                        .covariance_dh([pi[0] - (center[0] + off[0]), pi[1] - (center[1] + off[1])])
                })
                .sum::<f64>()
                / nu;
        }
        if m == 1 {
            b[n] = 1.0;
        }

        // Within-block covariance C̄(B,B). For coincident discretization
        // points (u = v) the nugget is excluded: it is a measure-zero
        // discontinuity in the block integral (gstat/GSLIB convention).
        let c0_continuous = c0 - self.model.nugget;
        let mut cbb = 0.0;
        for (ui, u) in offsets.iter().enumerate() {
            for (vi, v) in offsets.iter().enumerate() {
                cbb += if ui == vi {
                    c0_continuous
                } else {
                    self.model.covariance_dh([u[0] - v[0], u[1] - v[1]])
                };
            }
        }
        cbb /= nu * nu;

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
                (v, cbb - reduction)
            }
            _ => {
                let v: f64 = (0..n).map(|ii| w[ii] * self.data.value(nb[ii])).sum();
                let reduction: f64 = (0..dim).map(|i| w[i] * b0[i]).sum();
                (v, cbb - reduction)
            }
        };
        Ok(KrigingEstimate {
            value,
            variance: variance.max(0.0),
        })
    }

    /// Block kriging over all grid cells, with blocks of `block_size`
    /// discretized as a regular `discr[0]` x `discr[1]` point grid.
    pub fn predict_block_grid(
        &self,
        grid: &Grid2D,
        block_size: [f64; 2],
        discr: [usize; 2],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let offsets = block_offsets(block_size, discr)?;
        let centers = grid.centers();
        let ests = crate::parallel::par_map(centers.len(), |i| {
            self.predict_block(centers[i], &offsets)
                .unwrap_or(NAN_ESTIMATE)
        });
        Ok(ests.into_iter().map(|e| (e.value, e.variance)).unzip())
    }
}

/// Regular block-discretization offsets: an `n[0]` x `n[1]` grid of cell
/// centers covering a block of `size`, centered on the origin.
pub fn block_offsets(size: [f64; 2], n: [usize; 2]) -> Result<Vec<[f64; 2]>> {
    if n[0] == 0 || n[1] == 0 {
        return Err(GeostatError::InvalidParameter(
            "block discretization needs at least 1 point per axis".into(),
        ));
    }
    if !(size[0] > 0.0) || !(size[1] > 0.0) {
        return Err(GeostatError::InvalidParameter(format!(
            "block size must be positive, got {size:?}"
        )));
    }
    let mut offsets = Vec::with_capacity(n[0] * n[1]);
    for iy in 0..n[1] {
        for ix in 0..n[0] {
            offsets.push([
                ((ix as f64 + 0.5) / n[0] as f64 - 0.5) * size[0],
                ((iy as f64 + 0.5) / n[1] as f64 - 0.5) * size[1],
            ]);
        }
    }
    Ok(offsets)
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
    fn external_drift_reproduces_linear_relation() {
        // z is exactly linear in a known covariate d (plus nothing else):
        // KED with that covariate must reproduce z = 2 + 3 d everywhere.
        let mut coords = Vec::new();
        let mut values = Vec::new();
        let mut drift = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                let x = i as f64 * 10.0;
                let y = j as f64 * 10.0;
                let d = (x * 0.13 + y * 0.07).sin();
                coords.push([x, y]);
                drift.push(vec![d]);
                values.push(2.0 + 3.0 * d);
            }
        }
        let data = PointSet::new(coords, values).unwrap();
        let m = model();
        let cfg = KrigingConfig {
            method: KrigingMethod::ExternalDrift { n_vars: 1 },
            ..Default::default()
        };
        let k = Kriging::with_external_drift(&data, &m, cfg, drift).unwrap();
        for (target, d) in [([12.0, 33.0], 0.4_f64), ([45.0, 5.0], -0.7)] {
            let est = k.predict_with_drift(target, &[d]).unwrap();
            let expected = 2.0 + 3.0 * d;
            assert!(
                (est.value - expected).abs() < 1e-7,
                "{target:?}: {} vs {expected}",
                est.value
            );
        }
        // Mismatched usage is rejected.
        assert!(k.predict([0.0, 0.0]).is_err());
        assert!(k.predict_with_drift([0.0, 0.0], &[1.0, 2.0]).is_err());
        assert!(
            Kriging::new(
                &data,
                &m,
                KrigingConfig {
                    method: KrigingMethod::ExternalDrift { n_vars: 1 },
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn anisotropic_model_weights_by_direction() {
        // Two data points equidistant from the target, one along the major
        // axis (N-S), one along the minor: the major-axis point is better
        // correlated and must get the larger weight, pulling the estimate
        // toward its value.
        let data = PointSet::new(
            vec![[0.0, 30.0], [30.0, 0.0], [-25.0, -25.0]],
            vec![10.0, 0.0, 5.0],
        )
        .unwrap();
        let m = VariogramModel::new(
            0.0,
            vec![Structure::with_anisotropy(
                ModelKind::Spherical,
                1.0,
                100.0,
                0.0,
                0.25,
            )],
        )
        .unwrap();
        let k = Kriging::new(&data, &m, KrigingConfig::default()).unwrap();
        let est = k.predict([0.0, 0.0]).unwrap();
        // Closer (in correlation) to the value at [0, 30] = 10 than to 0.
        assert!(est.value > 5.0, "estimate {} not pulled north", est.value);
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
        assert!(matches!(
            k.predict([0.5, 0.2]),
            Err(GeostatError::NoNeighbors)
        ));
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
    fn block_kriging_averages_point_predictions() {
        // With a global neighborhood, the BLUP of the block average equals
        // the average of point BLUPs (linearity); the block variance must be
        // smaller than the central point variance.
        let data = sample_data();
        let m = model();
        let k = Kriging::new(&data, &m, KrigingConfig::default()).unwrap();
        let center = [0.6, 0.4];
        let offsets = block_offsets([0.4, 0.4], [4, 4]).unwrap();
        let block = k.predict_block(center, &offsets).unwrap();
        let mean_pts = offsets
            .iter()
            .map(|o| {
                k.predict([center[0] + o[0], center[1] + o[1]])
                    .unwrap()
                    .value
            })
            .sum::<f64>()
            / offsets.len() as f64;
        assert!(
            (block.value - mean_pts).abs() < 1e-10,
            "block {} vs mean of points {mean_pts}",
            block.value
        );
        let point_var = k.predict(center).unwrap().variance;
        assert!(
            block.variance < point_var,
            "{} vs {point_var}",
            block.variance
        );
        // Drift methods are rejected.
        let uk = Kriging::new(
            &data,
            &m,
            KrigingConfig {
                method: KrigingMethod::Universal { degree: 1 },
                ..Default::default()
            },
        )
        .unwrap();
        assert!(uk.predict_block(center, &offsets).is_err());
        assert!(block_offsets([0.0, 1.0], [4, 4]).is_err());
        assert!(block_offsets([1.0, 1.0], [0, 4]).is_err());
    }

    #[test]
    fn kdtree_neighborhood_matches_global_at_full_k() {
        // With k = n, the moving-neighborhood path must agree with the
        // global path exactly.
        let data = sample_data();
        let m = model();
        let global = Kriging::new(&data, &m, KrigingConfig::default()).unwrap();
        let local = Kriging::new(
            &data,
            &m,
            KrigingConfig {
                max_neighbors: Some(data.len()),
                ..Default::default()
            },
        )
        .unwrap();
        for target in [[0.3, 0.3], [0.9, 0.1], [2.0, 2.0]] {
            let a = global.predict(target).unwrap();
            let b = local.predict(target).unwrap();
            assert!((a.value - b.value).abs() < 1e-12);
            assert!((a.variance - b.variance).abs() < 1e-12);
        }
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
