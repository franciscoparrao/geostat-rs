//! Simple, ordinary, universal and external-drift kriging.

use ndarray::Array2;

use crate::covariance::Covariance;
use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::grid::Grid2D;
use crate::linalg::{Lu, lu_factor, solve};
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
    /// Number of drift basis functions in `dim` dimensions (0 for simple
    /// kriging). Degree-1 drift: `1 + dim` terms; degree-2: quadratic
    /// monomials added.
    fn n_drift(self, dim: usize) -> usize {
        match self {
            KrigingMethod::Simple { .. } => 0,
            KrigingMethod::Ordinary => 1,
            KrigingMethod::Universal { degree: 1 } => 1 + dim,
            KrigingMethod::Universal { degree: 2 } => 1 + dim + dim * (dim + 1) / 2,
            KrigingMethod::Universal { .. } => 0,
            KrigingMethod::ExternalDrift { n_vars } => 1 + n_vars,
        }
    }
}

/// Kriging configuration: method plus search neighborhood.
///
/// With `max_neighbors`, `search_radius` and `max_per_octant` unset, a
/// global neighborhood (all data points) is used. Neighbor distances are
/// Euclidean even for anisotropic models (gstat/GSLIB behavior).
#[derive(Debug, Clone, PartialEq)]
pub struct KrigingConfig {
    /// Kriging flavor.
    pub method: KrigingMethod,
    /// Maximum number of nearest conditioning points per estimate.
    pub max_neighbors: Option<usize>,
    /// Maximum search distance for conditioning points.
    pub search_radius: Option<f64>,
    /// Minimum conditioning points per estimate (GSLIB `ndmin`): targets
    /// whose neighborhood is smaller fail with `NoNeighbors` instead of
    /// estimating from too little data.
    pub min_neighbors: Option<usize>,
    /// Maximum conditioning points taken per octant around the target
    /// (GSLIB `noct`; quadrants in 2-D). Balances the neighborhood when the
    /// data are clustered: nearest-only search would take every point from
    /// one side and screen the rest of the domain.
    pub max_per_octant: Option<usize>,
}

impl Default for KrigingConfig {
    fn default() -> Self {
        Self {
            method: KrigingMethod::Ordinary,
            max_neighbors: None,
            search_radius: None,
            min_neighbors: None,
            max_per_octant: None,
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
    /// Lagrange multiplier on the unbiasedness (constant) constraint, when
    /// present (ordinary/universal/external-drift kriging); `None` for
    /// simple kriging. Needed by the ordinary lognormal back-transform.
    pub lagrange: Option<f64>,
}

const NAN_ESTIMATE: KrigingEstimate = KrigingEstimate {
    value: f64::NAN,
    variance: f64::NAN,
    lagrange: None,
};

/// Kriging predictor bound to a dataset and covariance/variogram model
/// (2-D by default; `Kriging<'_, 3>` for 3-D data). The model defaults to
/// [`VariogramModel`], but any type implementing [`Covariance<D>`] works
/// (`Kriging<'_, 2, MyCovariance>`) — see [`Covariance`] for the seam this
/// opens for custom, non-catalog covariance functions.
#[derive(Debug)]
pub struct Kriging<'a, const D: usize = 2, M: Covariance<D> = VariogramModel> {
    data: &'a PointSet<D>,
    model: &'a M,
    config: KrigingConfig,
    tree: Option<KdTree<D>>,
    // Coordinate normalization for numerically stable polynomial drift.
    center: [f64; D],
    scale: f64,
    // External drift values per data point, plus per-variable normalization.
    drift_data: Vec<Vec<f64>>,
    drift_mean: Vec<f64>,
    drift_scale: Vec<f64>,
    // Per-datum measurement-error variance added to the data-data diagonal
    // (empty = error-free data). See `with_measurement_error`.
    measurement_error: Vec<f64>,
    // For a global neighbourhood the kriging-system matrix is the same for
    // every target, so it is factored once here and only back-substituted per
    // target. `None` for moving neighbourhoods (a different matrix per target)
    // or if the global system is singular (predict then errors per target).
    global_lu: Option<Lu>,
}

impl<'a, const D: usize, M: Covariance<D>> Kriging<'a, D, M> {
    /// Builds a predictor for simple/ordinary/universal kriging.
    pub fn new(data: &'a PointSet<D>, model: &'a M, config: KrigingConfig) -> Result<Self> {
        if matches!(config.method, KrigingMethod::ExternalDrift { .. }) {
            return Err(GeostatError::InvalidParameter(
                "external drift kriging requires Kriging::with_external_drift".into(),
            ));
        }
        Self::build(data, model, config, Vec::new(), Vec::new())
    }

    /// Builds a predictor for data observed with measurement error
    /// (gstat's `Err` component): `errors[i]` is the error *variance* of
    /// datum `i`, added to the data-data diagonal only. Kriging then
    /// predicts the underlying signal — it is no longer an exact
    /// interpolator where the error is positive, and the observations are
    /// smoothed instead of honored.
    pub fn with_measurement_error(
        data: &'a PointSet<D>,
        model: &'a M,
        config: KrigingConfig,
        errors: Vec<f64>,
    ) -> Result<Self> {
        if matches!(config.method, KrigingMethod::ExternalDrift { .. }) {
            return Err(GeostatError::InvalidParameter(
                "measurement error with external drift is not supported yet".into(),
            ));
        }
        if errors.len() != data.len() {
            return Err(GeostatError::DimensionMismatch(format!(
                "{} error variances vs {} data points",
                errors.len(),
                data.len()
            )));
        }
        if errors.iter().any(|e| !(e.is_finite() && *e >= 0.0)) {
            return Err(GeostatError::InvalidParameter(
                "measurement-error variances must be finite and non-negative".into(),
            ));
        }
        Self::build(data, model, config, Vec::new(), errors)
    }

    /// Builds an external-drift kriging predictor. `drift_data[i]` holds the
    /// covariate values at data point `i` and must have `n_vars` entries
    /// matching `KrigingMethod::ExternalDrift`.
    pub fn with_external_drift(
        data: &'a PointSet<D>,
        model: &'a M,
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
        Self::build(data, model, config, drift_data, Vec::new())
    }

    fn build(
        data: &'a PointSet<D>,
        model: &'a M,
        config: KrigingConfig,
        drift_data: Vec<Vec<f64>>,
        measurement_error: Vec<f64>,
    ) -> Result<Self> {
        if let KrigingMethod::Universal { degree } = config.method
            && !(1..=2).contains(&degree)
        {
            return Err(GeostatError::InvalidParameter(format!(
                "universal kriging drift degree must be 1 or 2, got {degree}"
            )));
        }
        if model.has_power() && matches!(config.method, KrigingMethod::Simple { .. }) {
            return Err(GeostatError::InvalidParameter(
                "Power (unbounded) models have no covariance function, so simple kriging \
                 cannot use them; use Ordinary, Universal or ExternalDrift instead (kriged \
                 directly in semivariogram form, the classical IRF-0 generalization)"
                    .into(),
            ));
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
        if let Some((i, j)) = data.duplicate_pair() {
            return Err(GeostatError::DuplicatePoints(i, j));
        }
        if config.max_per_octant == Some(0) {
            return Err(GeostatError::InvalidParameter(
                "max_per_octant must be at least 1".into(),
            ));
        }
        if let Some(min_nb) = config.min_neighbors {
            if min_nb == 0 {
                return Err(GeostatError::InvalidParameter(
                    "min_neighbors must be at least 1".into(),
                ));
            }
            if min_nb > data.len() {
                return Err(GeostatError::InsufficientData(format!(
                    "min_neighbors ({min_nb}) exceeds the number of data points ({})",
                    data.len()
                )));
            }
            if let Some(max_nb) = config.max_neighbors
                && min_nb > max_nb
            {
                return Err(GeostatError::InvalidParameter(format!(
                    "min_neighbors ({min_nb}) exceeds max_neighbors ({max_nb})"
                )));
            }
        }
        let (min, max) = data.bbox();
        let mut center = [0.0; D];
        let mut spread = 0.0_f64;
        for d in 0..D {
            center[d] = 0.5 * (min[d] + max[d]);
            spread = spread.max(max[d] - min[d]);
        }
        let scale = (0.5 * spread).max(f64::MIN_POSITIVE);

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

        let tree = (config.max_neighbors.is_some()
            || config.search_radius.is_some()
            || config.max_per_octant.is_some())
        .then(|| KdTree::build(data.coords()));

        let mut kriging = Self {
            data,
            model,
            config,
            tree,
            center,
            scale,
            drift_data,
            drift_mean,
            drift_scale,
            measurement_error,
            global_lu: None,
        };
        // Global neighbourhood: factor the (target-independent) system once.
        // A singular factorization is left as None so predict errors per target
        // exactly as before.
        if kriging.tree.is_none() {
            let nb: Vec<usize> = (0..kriging.data.len()).collect();
            let m = kriging.config.method.n_drift(D);
            if kriging.data.len() >= m {
                let a = kriging.build_lhs(&nb);
                kriging.global_lu = lu_factor(a).ok();
            }
        }
        Ok(kriging)
    }

    fn fill_basis(&self, p: [f64; D], drift: Option<&[f64]>, out: &mut [f64]) {
        match self.config.method {
            KrigingMethod::Simple { .. } => {}
            KrigingMethod::Ordinary => out[0] = 1.0,
            KrigingMethod::Universal { degree } => {
                let mut xn = [0.0; D];
                for d in 0..D {
                    xn[d] = (p[d] - self.center[d]) / self.scale;
                }
                out[0] = 1.0;
                out[1..=D].copy_from_slice(&xn);
                if degree == 2 {
                    let mut idx = 1 + D;
                    for a in 0..D {
                        for b in a..D {
                            out[idx] = xn[a] * xn[b];
                            idx += 1;
                        }
                    }
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

    fn neighbors(&self, target: [f64; D]) -> Vec<usize> {
        let Some(tree) = &self.tree else {
            return (0..self.data.len()).collect();
        };
        let Some(per_octant) = self.config.max_per_octant else {
            return tree.k_nearest(
                target,
                self.config.max_neighbors.unwrap_or(self.data.len()),
                self.config.search_radius,
            );
        };
        // Octant search (GSLIB noct): walk the radius-bounded candidates in
        // distance order, taking at most `per_octant` per 2^D sector around
        // the target, then cap the total at max_neighbors (GSLIB ndmax).
        let mut cand = tree.k_nearest(target, self.data.len(), self.config.search_radius);
        let d2 = |i: usize| -> f64 {
            let c = self.data.coord(i);
            (0..D).map(|d| (c[d] - target[d]).powi(2)).sum()
        };
        cand.sort_by(|&a, &b| d2(a).total_cmp(&d2(b)));
        let mut counts = vec![0_usize; 1 << D];
        let mut selected = Vec::new();
        for &i in &cand {
            let c = self.data.coord(i);
            let mut oct = 0;
            for d in 0..D {
                if c[d] >= target[d] {
                    oct |= 1 << d;
                }
            }
            if counts[oct] < per_octant {
                counts[oct] += 1;
                selected.push(i);
            }
        }
        if let Some(k) = self.config.max_neighbors {
            selected.truncate(k); // distance-ordered, so this keeps the nearest
        }
        selected
    }

    /// Kriging estimate at a single target location (simple/ordinary/
    /// universal kriging; external drift needs [`Kriging::predict_with_drift`]).
    pub fn predict(&self, target: [f64; D]) -> Result<KrigingEstimate> {
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
        target: [f64; D],
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

    /// Builds the (target-independent) kriging-system matrix for the
    /// conditioning points `nb`: the data--data covariance block plus the drift
    /// basis rows/columns and the Lagrange constraints.
    fn build_lhs(&self, nb: &[usize]) -> Array2<f64> {
        let m = self.config.method.n_drift(D);
        let n = nb.len();
        let dim = n + m;
        // Power models have no covariance (unbounded variance): the system
        // is built directly in semivariogram form instead, `entry(h) =
        // gamma(h)` rather than `c0 - gamma(h)` (`c0` unused, `entry(0) =
        // 0`) -- see `VariogramModel::has_power` and `predict_inner` for the
        // matching change to the right-hand side/variance formula. This is
        // the standard IRF-0 generalization of ordinary/universal kriging
        // (Cressie 1993 §3.4.5, GSLIB `kt3d`'s linear/power option): valid
        // because OK/UK's derivation only needs the *unbiasedness*
        // constraint `sum(w)=1`, never a finite C(0), unlike simple kriging.
        let power = self.model.has_power();
        let c0 = self.model.covariance_dh([0.0; D]);
        let entry = |h: [f64; D]| {
            if power {
                self.model.gamma_dh(h)
            } else {
                c0 - self.model.gamma_dh(h)
            }
        };
        let mut a = Array2::<f64>::zeros((dim, dim));
        let mut f = vec![0.0; m];
        for (ii, &i) in nb.iter().enumerate() {
            let pi = self.data.coord(i);
            a[[ii, ii]] = entry([0.0; D]) + self.measurement_error.get(i).copied().unwrap_or(0.0);
            for (jj, &j) in nb.iter().enumerate().skip(ii + 1) {
                let pj = self.data.coord(j);
                let c = entry(sep(pi, pj));
                a[[ii, jj]] = c;
                a[[jj, ii]] = c;
            }
            self.fill_basis(pi, self.drift_data.get(i).map(Vec::as_slice), &mut f);
            for (k, &fk) in f.iter().enumerate() {
                a[[ii, n + k]] = fk;
                a[[n + k, ii]] = fk;
            }
        }
        a
    }

    fn predict_inner(
        &self,
        target: [f64; D],
        target_drift: Option<&[f64]>,
    ) -> Result<KrigingEstimate> {
        let nb = self.neighbors(target);
        if nb.len() < self.config.min_neighbors.unwrap_or(1) {
            return Err(GeostatError::NoNeighbors);
        }
        let m = self.config.method.n_drift(D);
        let n = nb.len();
        if n < m {
            return Err(GeostatError::InsufficientData(format!(
                "{n} neighbors cannot support {m} drift terms"
            )));
        }
        let dim = n + m;
        let power = self.model.has_power();
        let c0 = self.model.covariance_dh([0.0; D]);

        // Right-hand side: covariances to the target, then its drift basis
        // (semivariances for a Power model — see `build_lhs`).
        let mut b = vec![0.0; dim];
        for (ii, &i) in nb.iter().enumerate() {
            let h = sep(self.data.coord(i), target);
            b[ii] = if power {
                self.model.gamma_dh(h)
            } else {
                c0 - self.model.gamma_dh(h)
            };
        }
        let mut f = vec![0.0; m];
        self.fill_basis(target, target_drift, &mut f);
        for (k, &fk) in f.iter().enumerate() {
            b[n + k] = fk;
        }

        let b0 = b.clone();
        // Reuse the factored global system when available; otherwise build and
        // solve the per-target (moving-neighbourhood) system.
        let w = match &self.global_lu {
            Some(lu) => lu.solve(b),
            None => solve(self.build_lhs(&nb), b)?,
        };

        let (value, variance, lagrange) = match self.config.method {
            KrigingMethod::Simple { mean } => {
                let mut v = mean;
                let mut reduction = 0.0;
                for ii in 0..n {
                    v += w[ii] * (self.data.value(nb[ii]) - mean);
                    reduction += w[ii] * b0[ii];
                }
                (v, c0 - reduction, None)
            }
            _ => {
                let v: f64 = (0..n).map(|ii| w[ii] * self.data.value(nb[ii])).sum();
                // Includes the Lagrange-multiplier terms via b0[n..].
                let reduction: f64 = (0..dim).map(|i| w[i] * b0[i]).sum();
                // w[n] is the multiplier on the constant basis function (the
                // first drift term), i.e. the OK unbiasedness constraint.
                // In semivariogram form the textbook variance is exactly
                // `sum(w_i*gamma(i,0)) + mu` (Isaaks & Srivastava eq.
                // 12.19), i.e. `reduction` with no `c0 -` -- `c0` isn't a
                // meaningful quantity for Power in the first place.
                let variance = if power { reduction } else { c0 - reduction };
                (v, variance, Some(w[n]))
            }
        };

        Ok(KrigingEstimate {
            value,
            variance: variance.max(0.0),
            lagrange,
        })
    }

    /// Kriging estimates at many targets, in parallel. Targets whose system
    /// fails (e.g. no neighbors) yield NaN estimates.
    pub fn predict_many(&self, targets: &[[f64; D]]) -> Vec<KrigingEstimate> {
        crate::parallel::par_map(targets.len(), |i| {
            self.predict(targets[i]).unwrap_or(NAN_ESTIMATE)
        })
    }

    /// External-drift estimates at many targets, in parallel. `drifts[i]`
    /// holds the covariates of `targets[i]`.
    pub fn predict_many_with_drift(
        &self,
        targets: &[[f64; D]],
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

    /// Block kriging estimate: predicts the average of a block centered at
    /// `center`, discretized at `center + offsets[u]`. Supported for simple
    /// and ordinary kriging (polynomial or external drift would require
    /// averaging the drift over the block).
    pub fn predict_block(&self, center: [f64; D], offsets: &[[f64; D]]) -> Result<KrigingEstimate> {
        match self.config.method {
            KrigingMethod::Simple { .. } | KrigingMethod::Ordinary => {}
            _ => {
                return Err(GeostatError::InvalidParameter(
                    "block kriging supports simple and ordinary kriging only".into(),
                ));
            }
        }
        if self.model.has_power() {
            return Err(GeostatError::InvalidParameter(
                "block kriging with a Power model is not supported yet (needs a block-averaged \
                 semivariogram gamma-bar(B,B), not implemented) -- use point kriging instead"
                    .into(),
            ));
        }
        if offsets.is_empty() {
            return Err(GeostatError::InvalidParameter(
                "block discretization needs at least one point".into(),
            ));
        }
        let nb = self.neighbors(center);
        if nb.len() < self.config.min_neighbors.unwrap_or(1) {
            return Err(GeostatError::NoNeighbors);
        }
        let m = self.config.method.n_drift(D);
        let n = nb.len();
        let dim = n + m;
        let c0 = self.model.covariance_dh([0.0; D]);
        let nu = offsets.len() as f64;

        let mut a = Array2::<f64>::zeros((dim, dim));
        let mut b = vec![0.0; dim];
        for (ii, &i) in nb.iter().enumerate() {
            let pi = self.data.coord(i);
            a[[ii, ii]] = c0 + self.measurement_error.get(i).copied().unwrap_or(0.0);
            for (jj, &j) in nb.iter().enumerate().skip(ii + 1) {
                let pj = self.data.coord(j);
                let c = c0 - self.model.gamma_dh(sep(pi, pj));
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
                    let dh: [f64; D] = std::array::from_fn(|k| pi[k] - (center[k] + off[k]));
                    c0 - self.model.gamma_dh(dh)
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
        let c0_continuous = c0 - self.model.nugget();
        let mut cbb = 0.0;
        for (ui, u) in offsets.iter().enumerate() {
            for (vi, v) in offsets.iter().enumerate() {
                cbb += if ui == vi {
                    c0_continuous
                } else {
                    c0 - self.model.gamma_dh(sep(*u, *v))
                };
            }
        }
        cbb /= nu * nu;

        let b0 = b.clone();
        let w = solve(a, b)?;
        let (value, variance, lagrange) = match self.config.method {
            KrigingMethod::Simple { mean } => {
                let mut v = mean;
                let mut reduction = 0.0;
                for ii in 0..n {
                    v += w[ii] * (self.data.value(nb[ii]) - mean);
                    reduction += w[ii] * b0[ii];
                }
                (v, cbb - reduction, None)
            }
            _ => {
                let v: f64 = (0..n).map(|ii| w[ii] * self.data.value(nb[ii])).sum();
                let reduction: f64 = (0..dim).map(|i| w[i] * b0[i]).sum();
                (v, cbb - reduction, Some(w[n]))
            }
        };
        Ok(KrigingEstimate {
            value,
            variance: variance.max(0.0),
            lagrange,
        })
    }
}

impl<M: Covariance<2>> Kriging<'_, 2, M> {
    /// Kriging over all cell centers of a grid. Returns `(values, variances)`
    /// in grid storage order.
    pub fn predict_grid(&self, grid: &Grid2D) -> (Vec<f64>, Vec<f64>) {
        let ests = self.predict_many(&grid.centers());
        ests.into_iter().map(|e| (e.value, e.variance)).unzip()
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

/// Separation vector between two points.
fn sep<const D: usize>(a: [f64; D], b: [f64; D]) -> [f64; D] {
    let mut dh = [0.0; D];
    for d in 0..D {
        dh[d] = a[d] - b[d];
    }
    dh
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
    fn measurement_error_smooths_instead_of_honoring() {
        let data = sample_data();
        let m = model();
        let exact = Kriging::new(&data, &m, KrigingConfig::default()).unwrap();
        let noisy = Kriging::with_measurement_error(
            &data,
            &m,
            KrigingConfig::default(),
            vec![0.5; data.len()],
        )
        .unwrap();
        for i in 0..data.len() {
            let e = exact.predict(data.coord(i)).unwrap();
            let n = noisy.predict(data.coord(i)).unwrap();
            // Exact kriging honors the datum; with error the estimate is
            // smoothed toward the neighbors and the variance is positive.
            assert!((e.value - data.value(i)).abs() < 1e-8);
            assert!(
                (n.value - data.value(i)).abs() > 1e-3,
                "datum {i} still honored: {} vs {}",
                n.value,
                data.value(i)
            );
            assert!(n.variance > 1e-3, "variance {}", n.variance);
        }
        // Zero error variances reproduce the exact predictor.
        let zero = Kriging::with_measurement_error(
            &data,
            &m,
            KrigingConfig::default(),
            vec![0.0; data.len()],
        )
        .unwrap();
        let t = [0.3, 0.7];
        let a = exact.predict(t).unwrap();
        let b = zero.predict(t).unwrap();
        assert!((a.value - b.value).abs() < 1e-14);
        assert!((a.variance - b.variance).abs() < 1e-14);
        // Validation.
        assert!(
            Kriging::with_measurement_error(&data, &m, KrigingConfig::default(), vec![0.1; 2])
                .is_err()
        );
        assert!(
            Kriging::with_measurement_error(
                &data,
                &m,
                KrigingConfig::default(),
                vec![-0.1; data.len()]
            )
            .is_err()
        );
    }

    #[test]
    fn min_neighbors_fails_sparse_targets() {
        let data = sample_data();
        let m = model();
        let k = Kriging::new(
            &data,
            &m,
            KrigingConfig {
                search_radius: Some(0.75),
                min_neighbors: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        // Near the cloud: enough neighbors.
        assert!(k.predict([0.5, 0.5]).is_ok());
        // Isolated corner: only one point within 0.75 -> refused.
        match k.predict([1.5, 1.5]) {
            Err(GeostatError::NoNeighbors) => {}
            other => panic!("expected NoNeighbors, got {other:?}"),
        }
        // Invalid combinations rejected at build.
        assert!(
            Kriging::new(
                &data,
                &m,
                KrigingConfig {
                    min_neighbors: Some(0),
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
                    max_neighbors: Some(2),
                    min_neighbors: Some(3),
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn octant_search_balances_clustered_neighbors() {
        // Dense cluster of high values just east of the target; a few low
        // values west, slightly farther. Nearest-only search sees only the
        // cluster; a 1-per-quadrant search must include the west points.
        let mut coords = vec![
            [1.0, 0.1],
            [1.1, -0.1],
            [1.2, 0.2],
            [1.3, -0.2],
            [1.4, 0.05],
        ];
        let mut values = vec![10.0; coords.len()];
        coords.extend_from_slice(&[[-2.0, 0.5], [-2.0, -0.5], [0.5, -2.0], [0.5, 2.0]]);
        values.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        let data = PointSet::new(coords, values).unwrap();
        let m = model();
        let target = [0.0, 0.0];

        let nearest = Kriging::new(
            &data,
            &m,
            KrigingConfig {
                max_neighbors: Some(4),
                ..Default::default()
            },
        )
        .unwrap()
        .predict(target)
        .unwrap();
        let octant = Kriging::new(
            &data,
            &m,
            KrigingConfig {
                max_neighbors: Some(4),
                max_per_octant: Some(1),
                ..Default::default()
            },
        )
        .unwrap()
        .predict(target)
        .unwrap();
        // Pure k-nearest conditions only on the cluster (estimate ~10);
        // octant search mixes in the low values from the other quadrants.
        assert!(nearest.value > 9.0, "nearest-only {}", nearest.value);
        assert!(
            octant.value < nearest.value - 2.0,
            "octant {} vs nearest {}",
            octant.value,
            nearest.value
        );
    }

    #[test]
    fn duplicate_points_rejected_up_front() {
        let data = PointSet::new(
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            vec![1.0, 2.0, 2.1, 1.5],
        )
        .unwrap();
        let m = model();
        match Kriging::new(&data, &m, KrigingConfig::default()) {
            Err(GeostatError::DuplicatePoints(1, 2)) => {}
            other => panic!("expected DuplicatePoints(1, 2), got {other:?}"),
        }
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
    fn power_model_matches_gstat_ordinary_kriging() {
        // gstat reference (R): vgm(2.0, "Pow", 1.2) is gamma(h) = 2.0*h^1.2
        // (gstat's Pow psill/range double as the slope/exponent directly,
        // no length-scale). d/m/target coords match validation/power_gstat.R
        // exactly.
        //   d <- data.frame(x=c(0,10,0,10,5), y=c(0,0,10,10,5),
        //                   z=c(1.0,2.0,1.5,2.5,1.8))
        //   krige(z~1, d, target, model=vgm(2.0,"Pow",1.2))
        let data = PointSet::new(
            vec![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [5.0, 5.0]],
            vec![1.0, 2.0, 1.5, 2.5, 1.8],
        )
        .unwrap();
        let m = VariogramModel::new(0.0, vec![Structure::new(ModelKind::Power(1.2), 2.0, 1.0)])
            .unwrap();
        let k = Kriging::new(&data, &m, KrigingConfig::default()).unwrap();

        let cases = [
            ([3.0, 4.0], 1.5252052569, 6.4437897810),
            ([7.0, 8.0], 2.1366210000, 7.7780140000),
        ];
        for (target, exp_value, exp_var) in cases {
            let est = k.predict(target).unwrap();
            assert!(
                (est.value - exp_value).abs() < 1e-6,
                "value {} vs gstat {}",
                est.value,
                exp_value
            );
            assert!(
                (est.variance - exp_var).abs() < 1e-5,
                "variance {} vs gstat {}",
                est.variance,
                exp_var
            );
        }
        // Exact at a datum (5,5) -> z=1.8, ~zero variance.
        let at_datum = k.predict([5.0, 5.0]).unwrap();
        assert!((at_datum.value - 1.8).abs() < 1e-6);
        assert!(at_datum.variance < 1e-6);
    }

    #[test]
    fn power_model_rejects_simple_kriging() {
        let data = sample_data();
        let m = VariogramModel::new(0.0, vec![Structure::new(ModelKind::Power(1.0), 1.0, 1.0)])
            .unwrap();
        let cfg = KrigingConfig {
            method: KrigingMethod::Simple { mean: 1.5 },
            ..Default::default()
        };
        assert!(Kriging::new(&data, &m, cfg).is_err());
    }

    #[test]
    fn power_model_rejects_block_kriging() {
        let data = sample_data();
        let m = VariogramModel::new(0.0, vec![Structure::new(ModelKind::Power(1.0), 1.0, 1.0)])
            .unwrap();
        let k = Kriging::new(&data, &m, KrigingConfig::default()).unwrap();
        let offsets = vec![[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]];
        assert!(k.predict_block([0.5, 0.5], &offsets).is_err());
    }

    #[test]
    fn power_model_universal_kriging_recovers_linear_trend() {
        // With a genuinely linear field, universal kriging with a Power
        // (IRF-0) model plus a linear drift should reproduce the trend
        // almost exactly away from data, same invariant as the bounded-model
        // universal kriging test above.
        let coords = vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 2.0],
            [2.0, 8.0],
            [8.0, 3.0],
        ];
        let values: Vec<f64> = coords.iter().map(|p| 2.0 + 0.3 * p[0] - 0.1 * p[1]).collect();
        let data = PointSet::new(coords, values).unwrap();
        let m = VariogramModel::new(0.01, vec![Structure::new(ModelKind::Power(1.0), 0.5, 1.0)])
            .unwrap();
        let cfg = KrigingConfig {
            method: KrigingMethod::Universal { degree: 1 },
            ..Default::default()
        };
        let k = Kriging::new(&data, &m, cfg).unwrap();
        let est = k.predict([6.0, 4.0]).unwrap();
        let expected = 2.0 + 0.3 * 6.0 - 0.1 * 4.0;
        assert!(
            (est.value - expected).abs() < 0.05,
            "{} vs {expected}",
            est.value
        );
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
            ..Default::default()
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

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// 3-8 points in [0,100]^2 with every pair at least 2 units apart
        /// (duplicate/near-duplicate coordinates make the kriging system
        /// singular for reasons unrelated to the property under test), each
        /// carrying an arbitrary value.
        fn well_spaced_points() -> impl Strategy<Value = Vec<(f64, f64, f64)>> {
            prop::collection::vec((0.0f64..100.0, 0.0f64..100.0, -10.0f64..10.0), 3..8).prop_filter(
                "points must be pairwise well separated",
                |pts| {
                    for i in 0..pts.len() {
                        for j in (i + 1)..pts.len() {
                            let dx = pts[i].0 - pts[j].0;
                            let dy = pts[i].1 - pts[j].1;
                            if (dx * dx + dy * dy).sqrt() < 2.0 {
                                return false;
                            }
                        }
                    }
                    true
                },
            )
        }

        proptest! {
            #[test]
            fn ordinary_kriging_is_exact_at_data_points_for_any_well_spaced_field(
                pts in well_spaced_points(),
                sill in 0.1f64..5.0,
                range in 5.0f64..200.0,
            ) {
                let coords: Vec<[f64; 2]> = pts.iter().map(|p| [p.0, p.1]).collect();
                let values: Vec<f64> = pts.iter().map(|p| p.2).collect();
                let data = PointSet::new(coords, values).unwrap();
                // Nugget-free: with a nugget OK smooths instead of
                // interpolating exactly, which would break this property
                // by design (see `measurement_error_smooths_instead_of_honoring`).
                let model = VariogramModel::new(
                    0.0,
                    vec![Structure::new(ModelKind::Exponential, sill, range)],
                )
                .unwrap();
                let k = Kriging::new(&data, &model, KrigingConfig::default()).unwrap();
                for i in 0..data.len() {
                    let est = k.predict(data.coord(i)).unwrap();
                    prop_assert!(
                        (est.value - data.value(i)).abs() < 1e-6,
                        "point {i}: {} vs {}", est.value, data.value(i)
                    );
                    prop_assert!(est.variance < 1e-6, "point {i}: var {}", est.variance);
                }
            }
        }
    }
}
