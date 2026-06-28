//! Python bindings for geostat-core (module name: `geostat_rs`).
//!
//! Inputs are plain sequences of floats (lists, tuples or 1-D numpy
//! arrays), so the module has no numpy dependency. Build with maturin
//! (`maturin develop`) or copy the cdylib as `geostat_rs.so`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use geostat_core as core;
use geostat_core::{
    DirectionConfig, Grid2D, Kriging, KrigingConfig, KrigingMethod, PointSet, SgsConfig, SisConfig,
    VariogramConfig,
};

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn point_set(x: Vec<f64>, y: Vec<f64>, values: Vec<f64>) -> PyResult<PointSet> {
    PointSet::from_xyz(&x, &y, &values).map_err(err)
}

fn point_set_3d(x: Vec<f64>, y: Vec<f64>, z: Vec<f64>, values: Vec<f64>) -> PyResult<PointSet<3>> {
    PointSet::<3>::from_xyzv(&x, &y, &z, &values).map_err(err)
}

fn vario_config<const D: usize>(
    data: &PointSet<D>,
    n_lags: usize,
    max_dist: Option<f64>,
    azimuth: Option<f64>,
    dip: f64,
    tolerance: f64,
) -> VariogramConfig {
    let (min, max) = data.bbox();
    let diag = (0..D)
        .map(|d| (max[d] - min[d]).powi(2))
        .sum::<f64>()
        .sqrt();
    VariogramConfig {
        n_lags,
        max_dist: max_dist.unwrap_or(diag / 3.0),
        direction: azimuth.map(|az| DirectionConfig {
            azimuth_deg: az,
            dip_deg: dip,
            tolerance_deg: tolerance,
        }),
    }
}

fn parse_kinds(spec: &str) -> PyResult<Vec<core::ModelKind>> {
    if spec.eq_ignore_ascii_case("best") || spec.eq_ignore_ascii_case("all") {
        return Ok(core::ModelKind::ALL.to_vec());
    }
    spec.split(',')
        .map(|s| s.parse::<core::ModelKind>().map_err(err))
        .collect()
}

fn build_method(
    method: &str,
    mean: Option<f64>,
    degree: u8,
    data: &PointSet,
) -> PyResult<KrigingMethod> {
    match method {
        "ordinary" => Ok(KrigingMethod::Ordinary),
        "simple" => Ok(KrigingMethod::Simple {
            mean: mean.unwrap_or_else(|| data.mean()),
        }),
        "universal" => Ok(KrigingMethod::Universal { degree }),
        other => Err(PyValueError::new_err(format!(
            "unknown method '{other}' (expected ordinary, simple or universal)"
        ))),
    }
}

/// A fitted variogram model (JSON-compatible with the geostat CLI).
#[pyclass(module = "geostat_rs")]
#[derive(Clone)]
struct VariogramModel {
    inner: core::VariogramModel,
}

#[pymethods]
impl VariogramModel {
    /// Parses a model from its JSON representation.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let m: core::VariogramModel = serde_json::from_str(json).map_err(err)?;
        Ok(Self {
            inner: core::VariogramModel::new(m.nugget, m.structures).map_err(err)?,
        })
    }

    /// JSON representation (usable with the geostat CLI).
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(err)
    }

    /// Semivariance at scalar lag `h`.
    fn gamma(&self, h: f64) -> f64 {
        self.inner.gamma(h)
    }

    /// Total sill (nugget + partial sills).
    fn total_sill(&self) -> f64 {
        self.inner.total_sill()
    }

    /// Geometric anisotropy of the first anisotropic structure as
    /// `(major_azimuth_deg, minor_over_major_ratio)`, or `None` if isotropic.
    fn anisotropy(&self) -> Option<(f64, f64)> {
        self.inner
            .structures
            .iter()
            .find_map(|s| s.anis.map(|a| (a.azimuth_deg, a.ratio)))
    }

    fn __repr__(&self) -> String {
        format!("VariogramModel({})", self.inner)
    }
}

/// Experimental semivariogram. Returns `(h, gamma, n_pairs)` lists; empty
/// bins carry NaN gamma.
#[pyfunction]
#[pyo3(signature = (x, y, values, n_lags = 15, max_dist = None, azimuth = None, tolerance = 22.5))]
fn experimental_variogram(
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    n_lags: usize,
    max_dist: Option<f64>,
    azimuth: Option<f64>,
    tolerance: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let data = point_set(x, y, values)?;
    let cfg = vario_config(&data, n_lags, max_dist, azimuth, 0.0, tolerance);
    let ev = core::experimental_variogram(&data, &cfg).map_err(err)?;
    let mut h = Vec::new();
    let mut gamma = Vec::new();
    let mut np = Vec::new();
    for b in &ev.bins {
        h.push(b.h);
        gamma.push(b.gamma);
        np.push(b.n_pairs);
    }
    Ok((h, gamma, np))
}

/// 2-D variogram map (lag-space semivariance surface) for spotting anisotropy.
/// Returns a dict with `size` (grid side = 2*n_lags+1), `lag_width`, and flat
/// row-major (`iy*size+ix`) lists `hx`, `hy`, `gamma` (NaN where empty) and
/// `n_pairs`.
#[pyfunction]
#[pyo3(signature = (x, y, values, n_lags = 15, lag_width = 1.0))]
fn variogram_map(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    n_lags: usize,
    lag_width: f64,
) -> PyResult<Py<PyDict>> {
    let data = point_set(x, y, values)?;
    let m = core::variogram_map(&data, n_lags, lag_width).map_err(err)?;
    let (mut hx, mut hy) = (Vec::new(), Vec::new());
    for iy in 0..m.size {
        for ix in 0..m.size {
            let (lx, ly) = m.lag(ix, iy);
            hx.push(lx);
            hy.push(ly);
        }
    }
    let out = PyDict::new(py);
    out.set_item("size", m.size)?;
    out.set_item("lag_width", m.lag_width)?;
    out.set_item("hx", hx)?;
    out.set_item("hy", hy)?;
    out.set_item("gamma", &m.gamma)?;
    out.set_item("n_pairs", &m.n_pairs)?;
    Ok(out.into())
}

/// Fits a variogram model to the data by weighted least squares.
/// `kinds` is "best" or a comma-separated list (spherical, exponential,
/// gaussian, matern15, matern25).
#[pyfunction]
#[pyo3(signature = (x, y, values, n_lags = 15, max_dist = None, kinds = "best"))]
fn fit_variogram(
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    n_lags: usize,
    max_dist: Option<f64>,
    kinds: &str,
) -> PyResult<VariogramModel> {
    let data = point_set(x, y, values)?;
    let cfg = vario_config(&data, n_lags, max_dist, None, 0.0, 22.5);
    let ev = core::experimental_variogram(&data, &cfg).map_err(err)?;
    let fit = core::fit_best(&ev, &parse_kinds(kinds)?).map_err(err)?;
    Ok(VariogramModel { inner: fit.model })
}

/// Fits a geometrically anisotropic variogram model: estimates the major-axis
/// azimuth and the minor/major range ratio from `n_dirs` directional
/// variograms. `kinds` is "best" or a comma-separated family list. The returned
/// model carries the anisotropy; `model.anisotropy()` exposes (azimuth, ratio).
#[pyfunction]
#[pyo3(signature = (x, y, values, n_dirs = 4, n_lags = 15, max_dist = None, kinds = "best"))]
fn fit_anisotropic(
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    n_dirs: usize,
    n_lags: usize,
    max_dist: Option<f64>,
    kinds: &str,
) -> PyResult<VariogramModel> {
    let data = point_set(x, y, values)?;
    let cfg = vario_config(&data, n_lags, max_dist, None, 0.0, 22.5);
    let fit = core::fit_anisotropic(&data, &parse_kinds(kinds)?, n_dirs, n_lags, cfg.max_dist)
        .map_err(err)?;
    Ok(VariogramModel { inner: fit.model })
}

/// Kriging at arbitrary target locations. Returns `(predictions, variances)`;
/// failed targets yield NaN.
#[pyfunction]
#[pyo3(signature = (x, y, values, model, target_x, target_y, method = "ordinary",
    mean = None, degree = 1, max_neighbors = None, radius = None))]
#[allow(clippy::too_many_arguments)]
fn krige(
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    model: &VariogramModel,
    target_x: Vec<f64>,
    target_y: Vec<f64>,
    method: &str,
    mean: Option<f64>,
    degree: u8,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if target_x.len() != target_y.len() {
        return Err(PyValueError::new_err(
            "target_x and target_y differ in length",
        ));
    }
    let data = point_set(x, y, values)?;
    let config = KrigingConfig {
        method: build_method(method, mean, degree, &data)?,
        max_neighbors,
        search_radius: radius,
    };
    let kriging = Kriging::new(&data, &model.inner, config).map_err(err)?;
    let targets: Vec<[f64; 2]> = target_x
        .into_iter()
        .zip(target_y)
        .map(|(tx, ty)| [tx, ty])
        .collect();
    let ests = kriging.predict_many(&targets);
    Ok(ests.into_iter().map(|e| (e.value, e.variance)).unzip())
}

/// Ordinary/simple lognormal kriging. `values` are the original (positive)
/// data; `log_model` is the variogram of `ln(value)`. Returns
/// `(predictions, log_variances)` — predictions are back-transformed to
/// original units. For simple kriging pass `method="simple"` and `mean` in
/// log units.
#[pyfunction]
#[pyo3(signature = (x, y, values, log_model, target_x, target_y,
    method = "ordinary", mean = None, max_neighbors = None, radius = None))]
#[allow(clippy::too_many_arguments)]
fn lognormal_kriging(
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    log_model: &VariogramModel,
    target_x: Vec<f64>,
    target_y: Vec<f64>,
    method: &str,
    mean: Option<f64>,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if target_x.len() != target_y.len() {
        return Err(PyValueError::new_err(
            "target_x and target_y differ in length",
        ));
    }
    let data = point_set(x, y, values)?;
    let kmethod = match method {
        "ordinary" => KrigingMethod::Ordinary,
        "simple" => KrigingMethod::Simple {
            mean: mean.ok_or_else(|| {
                PyValueError::new_err("simple lognormal kriging needs `mean` (in log units)")
            })?,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "lognormal kriging supports ordinary or simple, got '{other}'"
            )));
        }
    };
    let config = KrigingConfig {
        method: kmethod,
        max_neighbors,
        search_radius: radius,
    };
    let targets: Vec<[f64; 2]> = target_x
        .into_iter()
        .zip(target_y)
        .map(|(tx, ty)| [tx, ty])
        .collect();
    let ests = core::lognormal_kriging(&data, &targets, &log_model.inner, &config).map_err(err)?;
    Ok(ests.into_iter().map(|e| (e.value, e.log_variance)).unzip())
}

/// Kriging over a regular grid (`bbox = (xmin, ymin, xmax, ymax)`), row-major
/// cell centers with y increasing. Returns `(predictions, variances)`.
#[pyfunction]
#[pyo3(signature = (x, y, values, model, bbox, nx, ny, method = "ordinary",
    mean = None, degree = 1, max_neighbors = None, radius = None, block = None, block_discr = (4, 4)))]
#[allow(clippy::too_many_arguments)]
fn krige_grid(
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    model: &VariogramModel,
    bbox: (f64, f64, f64, f64),
    nx: usize,
    ny: usize,
    method: &str,
    mean: Option<f64>,
    degree: u8,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
    block: Option<(f64, f64)>,
    block_discr: (usize, usize),
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let data = point_set(x, y, values)?;
    let grid = Grid2D::from_bbox([bbox.0, bbox.1], [bbox.2, bbox.3], nx, ny).map_err(err)?;
    let config = KrigingConfig {
        method: build_method(method, mean, degree, &data)?,
        max_neighbors,
        search_radius: radius,
    };
    let kriging = Kriging::new(&data, &model.inner, config).map_err(err)?;
    match block {
        Some((bw, bh)) => kriging
            .predict_block_grid(&grid, [bw, bh], [block_discr.0, block_discr.1])
            .map_err(err),
        None => Ok(kriging.predict_grid(&grid)),
    }
}

/// Cross-validation. Leave-one-out by default; pass `folds=k` for `k`-fold
/// (faster on large datasets, reproducible via `seed`). Returns a dict with
/// `me`, `mae`, `rmse`, `msdr`, `vecv`, `e1`, `predicted` and `variance`.
#[pyfunction]
#[pyo3(signature = (x, y, values, model, method = "ordinary", mean = None,
    degree = 1, max_neighbors = None, radius = None, folds = None, seed = 0))]
#[allow(clippy::too_many_arguments)]
fn loo_cv(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    model: &VariogramModel,
    method: &str,
    mean: Option<f64>,
    degree: u8,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
    folds: Option<usize>,
    seed: u64,
) -> PyResult<Py<PyDict>> {
    let data = point_set(x, y, values)?;
    let config = KrigingConfig {
        method: build_method(method, mean, degree, &data)?,
        max_neighbors,
        search_radius: radius,
    };
    let cv = match folds {
        Some(k) => core::k_fold(&data, &model.inner, &config, k, seed).map_err(err)?,
        None => core::leave_one_out(&data, &model.inner, &config).map_err(err)?,
    };
    let out = PyDict::new(py);
    out.set_item("me", cv.mean_error())?;
    out.set_item("mae", cv.mae())?;
    out.set_item("mse", cv.mse())?;
    out.set_item("rmse", cv.rmse())?;
    out.set_item("msdr", cv.msdr())?;
    out.set_item("rme", cv.rme())?;
    out.set_item("rmae", cv.rmae())?;
    out.set_item("rrmse", cv.rrmse())?;
    out.set_item("vecv", cv.vecv())?;
    out.set_item("e1", cv.e1())?;
    out.set_item("predicted", &cv.predicted)?;
    out.set_item("variance", &cv.variance)?;
    Ok(out.into())
}

/// Regression kriging: fits an OLS trend on `covariates` (a list of rows, one
/// per data point), fits the residual variogram automatically, and kriges the
/// residuals at the targets, adding the trend back. `target_covariates` is a
/// list of rows, one per target, with the same columns as `covariates`.
///
/// To use a trend from an external model (e.g. a machine-learning regressor),
/// supply `trend_at_data` and `trend_at_targets` directly; then `covariates`
/// and `target_covariates` are ignored for the trend (the residual variogram
/// is still fitted internally).
///
/// Returns a dict with `prediction`, `variance` (residual kriging variance),
/// and, when the built-in OLS trend is used, `trend_coef` (intercept first).
#[pyfunction]
#[pyo3(signature = (x, y, values, covariates, target_x, target_y, target_covariates,
    trend_at_data = None, trend_at_targets = None,
    n_lags = 15, max_dist = None, max_neighbors = None, radius = None))]
#[allow(clippy::too_many_arguments)]
fn regression_kriging(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    target_x: Vec<f64>,
    target_y: Vec<f64>,
    target_covariates: Vec<Vec<f64>>,
    trend_at_data: Option<Vec<f64>>,
    trend_at_targets: Option<Vec<f64>>,
    n_lags: usize,
    max_dist: Option<f64>,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
) -> PyResult<Py<PyDict>> {
    use core::{OlsTrend, RegressionKriging};

    if target_x.len() != target_y.len() {
        return Err(PyValueError::new_err(
            "target_x and target_y differ in length",
        ));
    }
    let data = point_set(x, y, values)?;
    let targets: Vec<[f64; 2]> = target_x
        .iter()
        .zip(&target_y)
        .map(|(&tx, &ty)| [tx, ty])
        .collect();

    // Trend: either supplied externally or fitted by built-in OLS.
    let out = PyDict::new(py);
    let (trend_data, trend_targets) = match (trend_at_data, trend_at_targets) {
        (Some(td), Some(tt)) => (td, tt),
        (None, None) => {
            let trend = OlsTrend::fit(&covariates, data.values()).map_err(err)?;
            let td: Vec<f64> = covariates.iter().map(|c| trend.predict(c)).collect();
            let tt: Vec<f64> = target_covariates.iter().map(|c| trend.predict(c)).collect();
            out.set_item("trend_coef", trend.coefficients())?;
            (td, tt)
        }
        _ => {
            return Err(PyValueError::new_err(
                "supply both trend_at_data and trend_at_targets, or neither",
            ));
        }
    };

    let rk = RegressionKriging::new(&data, &trend_data).map_err(err)?;
    let cfg = vario_config(rk.residuals(), n_lags, max_dist, None, 0.0, 22.5);
    let ev = core::experimental_variogram(rk.residuals(), &cfg).map_err(err)?;
    let resid_model = core::fit_best(&ev, &core::ModelKind::ALL)
        .map_err(err)?
        .model;
    let config = KrigingConfig {
        method: KrigingMethod::Ordinary,
        max_neighbors,
        search_radius: radius,
    };
    let ests = rk
        .predict(&targets, &trend_targets, &resid_model, &config)
        .map_err(err)?;
    let prediction: Vec<f64> = ests.iter().map(|e| e.value).collect();
    let variance: Vec<f64> = ests.iter().map(|e| e.variance).collect();
    out.set_item("prediction", prediction)?;
    out.set_item("variance", variance)?;
    Ok(out.into())
}

/// Ordinary co-kriging of a primary variable using a correlated secondary,
/// under a linear model of coregionalization fitted automatically. The two
/// variables may have different supports (heterotopic): the direct variograms
/// use each variable's own points and the cross-variogram is fitted on the
/// collocated subset (points shared by both, matched on exact coordinates).
/// `ridge` (default 1e-2) inflates the co-kriging matrix diagonal to keep the
/// notoriously ill-conditioned system stable; set 0.0 for the exact system.
/// Returns `(predictions, variances)` of the primary at the targets.
#[pyfunction]
#[pyo3(signature = (px, py, pv, sx, sy, sv, target_x, target_y,
    n_lags = 15, max_dist = None, max_neighbors = None, radius = None, ridge = 1e-2))]
#[allow(clippy::too_many_arguments)]
fn co_kriging(
    px: Vec<f64>,
    py: Vec<f64>,
    pv: Vec<f64>,
    sx: Vec<f64>,
    sy: Vec<f64>,
    sv: Vec<f64>,
    target_x: Vec<f64>,
    target_y: Vec<f64>,
    n_lags: usize,
    max_dist: Option<f64>,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
    ridge: f64,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    use std::collections::HashMap;
    if target_x.len() != target_y.len() {
        return Err(PyValueError::new_err(
            "target_x and target_y differ in length",
        ));
    }
    let primary = point_set(px, py, pv)?;
    let secondary = point_set(sx, sy, sv)?;
    let cfg = vario_config(&primary, n_lags, max_dist, None, 0.0, 22.5);
    let ea = core::experimental_variogram(&primary, &cfg).map_err(err)?;
    let eb = core::experimental_variogram(&secondary, &cfg).map_err(err)?;

    // Collocated subset (exact coordinate match) for the cross-variogram.
    let key = |c: &[f64; 2]| (c[0].to_bits(), c[1].to_bits());
    let sec_lookup: HashMap<(u64, u64), f64> = secondary
        .coords()
        .iter()
        .zip(secondary.values())
        .map(|(c, &v)| (key(c), v))
        .collect();
    let (mut co_coords, mut co_pv, mut co_sv) = (Vec::new(), Vec::new(), Vec::new());
    for (c, &v) in primary.coords().iter().zip(primary.values()) {
        if let Some(&s) = sec_lookup.get(&key(c)) {
            co_coords.push(*c);
            co_pv.push(v);
            co_sv.push(s);
        }
    }
    if co_coords.len() < n_lags {
        return Err(PyValueError::new_err(
            "too few collocated primary/secondary points to fit the cross-variogram",
        ));
    }
    let prim_co = PointSet::new(co_coords.clone(), co_pv).map_err(err)?;
    let sec_co = PointSet::new(co_coords, co_sv).map_err(err)?;
    let eab = core::experimental_cross_variogram(&prim_co, &sec_co, &cfg).map_err(err)?;

    let template = core::fit_best(&ea, &core::ModelKind::ALL).map_err(err)?;
    let lmc = core::fit_lmc(&ea, &eb, &eab, &template.model).map_err(err)?;
    let config = core::CoKrigingConfig {
        max_neighbors,
        search_radius: radius,
        ridge,
    };
    let ck = core::CoKriging::new(vec![&primary, &secondary], &lmc, config).map_err(err)?;
    let targets: Vec<[f64; 2]> = target_x
        .iter()
        .zip(&target_y)
        .map(|(&a, &b)| [a, b])
        .collect();
    let ests = ck.predict_many(&targets);
    Ok(ests.into_iter().map(|e| (e.value, e.variance)).unzip())
}

/// Inverse-distance weighting at the targets. `power` controls locality
/// (2 is typical); exact at the data. Returns a list of predictions.
#[pyfunction]
#[pyo3(signature = (x, y, values, target_x, target_y, power = 2.0,
    max_neighbors = None, radius = None))]
#[allow(clippy::too_many_arguments)]
fn idw(
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    target_x: Vec<f64>,
    target_y: Vec<f64>,
    power: f64,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
) -> PyResult<Vec<f64>> {
    if target_x.len() != target_y.len() {
        return Err(PyValueError::new_err(
            "target_x and target_y differ in length",
        ));
    }
    let data = point_set(x, y, values)?;
    let targets: Vec<[f64; 2]> = target_x
        .iter()
        .zip(&target_y)
        .map(|(&a, &b)| [a, b])
        .collect();
    let pred = core::Idw::new(&data, power, max_neighbors, radius).map_err(err)?;
    Ok(pred.predict_many(&targets))
}

/// k-nearest-neighbor averaging at the targets (`k = 1` is nearest-neighbor).
/// Returns a list of predictions.
#[pyfunction]
#[pyo3(signature = (x, y, values, target_x, target_y, k = 8, radius = None))]
fn knn(
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    target_x: Vec<f64>,
    target_y: Vec<f64>,
    k: usize,
    radius: Option<f64>,
) -> PyResult<Vec<f64>> {
    if target_x.len() != target_y.len() {
        return Err(PyValueError::new_err(
            "target_x and target_y differ in length",
        ));
    }
    let data = point_set(x, y, values)?;
    let targets: Vec<[f64; 2]> = target_x
        .iter()
        .zip(&target_y)
        .map(|(&a, &b)| [a, b])
        .collect();
    let pred = core::Knn::new(&data, k, radius).map_err(err)?;
    Ok(pred.predict_many(&targets))
}

/// Compares interpolation methods by leave-one-out cross-validation, ranked by
/// VEcv. Fits the variogram automatically for ordinary kriging. Returns a dict
/// `method -> {rmse, mae, vecv, e1}` for ordinary kriging, IDW, k-NN and NN.
#[pyfunction]
#[pyo3(signature = (x, y, values, n_lags = 15, max_dist = None,
    max_neighbors = None, radius = None, idw_power = 2.0, knn_k = 8))]
#[allow(clippy::too_many_arguments)]
fn compare_methods(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    n_lags: usize,
    max_dist: Option<f64>,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
    idw_power: f64,
    knn_k: usize,
) -> PyResult<Py<PyDict>> {
    let data = point_set(x, y, values)?;
    let cfg = vario_config(&data, n_lags, max_dist, None, 0.0, 22.5);
    let ev = core::experimental_variogram(&data, &cfg).map_err(err)?;
    let model = core::fit_best(&ev, &core::ModelKind::ALL)
        .map_err(err)?
        .model;
    let ok_config = KrigingConfig {
        method: KrigingMethod::Ordinary,
        max_neighbors,
        search_radius: radius,
    };

    let entries = [
        (
            "ordinary_kriging".to_string(),
            core::leave_one_out(&data, &model, &ok_config).map_err(err)?,
        ),
        (
            "idw".to_string(),
            core::idw_cross_validate(&data, idw_power, max_neighbors, radius).map_err(err)?,
        ),
        (
            "knn".to_string(),
            core::knn_cross_validate(&data, knn_k, radius).map_err(err)?,
        ),
        (
            "nearest_neighbor".to_string(),
            core::knn_cross_validate(&data, 1, radius).map_err(err)?,
        ),
    ];

    let out = PyDict::new(py);
    for (name, cv) in &entries {
        let m = PyDict::new(py);
        m.set_item("rmse", cv.rmse())?;
        m.set_item("mae", cv.mae())?;
        m.set_item("vecv", cv.vecv())?;
        m.set_item("e1", cv.e1())?;
        out.set_item(name, m)?;
    }
    Ok(out.into())
}

/// Tunes the IDW `power` by leave-one-out VEcv. Returns a dict with `best`,
/// `best_vecv` and `trace` (a list of `(power, vecv)`).
#[pyfunction]
#[pyo3(signature = (x, y, values, powers = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    max_neighbors = None, radius = None))]
fn tune_idw_power(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    powers: Vec<f64>,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let data = point_set(x, y, values)?;
    let res = core::tune_idw_power(&data, &powers, max_neighbors, radius).map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("best", res.best)?;
    out.set_item("best_vecv", res.best_vecv)?;
    out.set_item("trace", res.trace)?;
    Ok(out.into())
}

/// Tunes the k-NN `k` by leave-one-out VEcv. Returns a dict with `best`,
/// `best_vecv` and `trace` (a list of `(k, vecv)`).
#[pyfunction]
#[pyo3(signature = (x, y, values, ks = vec![1, 2, 3, 4, 6, 8, 12, 16, 24], radius = None))]
fn tune_knn_k(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    ks: Vec<usize>,
    radius: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let data = point_set(x, y, values)?;
    let res = core::tune_knn_k(&data, &ks, radius).map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("best", res.best)?;
    out.set_item("best_vecv", res.best_vecv)?;
    out.set_item("trace", res.trace)?;
    Ok(out.into())
}

/// Tunes the ordinary-kriging search-neighborhood size by leave-one-out VEcv,
/// fitting the variogram automatically. Returns a dict with `best`,
/// `best_vecv` and `trace` (a list of `(n_neighbors, vecv)`).
#[pyfunction]
#[pyo3(signature = (x, y, values, candidates = vec![4, 8, 12, 16, 24, 32, 48],
    n_lags = 15, max_dist = None, radius = None))]
#[allow(clippy::too_many_arguments)]
fn tune_kriging_neighbors(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    candidates: Vec<usize>,
    n_lags: usize,
    max_dist: Option<f64>,
    radius: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let data = point_set(x, y, values)?;
    let cfg = vario_config(&data, n_lags, max_dist, None, 0.0, 22.5);
    let ev = core::experimental_variogram(&data, &cfg).map_err(err)?;
    let model = core::fit_best(&ev, &core::ModelKind::ALL)
        .map_err(err)?
        .model;
    let res =
        core::tune_kriging_neighbors(&data, &model, KrigingMethod::Ordinary, &candidates, radius)
            .map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("best", res.best)?;
    out.set_item("best_vecv", res.best_vecv)?;
    out.set_item("trace", res.trace)?;
    Ok(out.into())
}

/// Conditional sequential Gaussian simulation. `model_ns` is a model fitted
/// to the normal scores (fit one with `fit_variogram` on transformed data,
/// or let the CLI auto-fit). Returns one list per realization, in grid
/// storage order.
#[pyfunction]
#[pyo3(signature = (x, y, values, model_ns, bbox, nx, ny, n_realizations = 10,
    seed = 42, max_neighbors = 16, radius = None))]
#[allow(clippy::too_many_arguments)]
fn sgs(
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    model_ns: &VariogramModel,
    bbox: (f64, f64, f64, f64),
    nx: usize,
    ny: usize,
    n_realizations: usize,
    seed: u64,
    max_neighbors: usize,
    radius: Option<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    let data = point_set(x, y, values)?;
    let grid = Grid2D::from_bbox([bbox.0, bbox.1], [bbox.2, bbox.3], nx, ny).map_err(err)?;
    let cfg = SgsConfig {
        n_realizations,
        seed,
        max_neighbors,
        search_radius: radius,
    };
    let res =
        core::sequential_gaussian_simulation(&data, &model_ns.inner, &grid, &cfg).map_err(err)?;
    Ok(res.realizations)
}

/// Conditional sequential indicator simulation. Indicator variogram models
/// are fitted automatically at each cutoff (spherical/exponential).
#[pyfunction]
#[pyo3(signature = (x, y, values, cutoffs, bbox, nx, ny, n_realizations = 10,
    seed = 42, max_neighbors = 16, radius = None, n_lags = 15, max_dist = None))]
#[allow(clippy::too_many_arguments)]
fn sis(
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    cutoffs: Vec<f64>,
    bbox: (f64, f64, f64, f64),
    nx: usize,
    ny: usize,
    n_realizations: usize,
    seed: u64,
    max_neighbors: usize,
    radius: Option<f64>,
    n_lags: usize,
    max_dist: Option<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    let data = point_set(x, y, values)?;
    let grid = Grid2D::from_bbox([bbox.0, bbox.1], [bbox.2, bbox.3], nx, ny).map_err(err)?;
    let cfg_v = vario_config(&data, n_lags, max_dist, None, 0.0, 22.5);
    let kinds = [core::ModelKind::Spherical, core::ModelKind::Exponential];
    let mut models = Vec::with_capacity(cutoffs.len());
    for &c in &cutoffs {
        let indicators: Vec<f64> = data
            .values()
            .iter()
            .map(|&v| if v <= c { 1.0 } else { 0.0 })
            .collect();
        let ind = PointSet::new(data.coords().to_vec(), indicators).map_err(err)?;
        let ev = core::experimental_variogram(&ind, &cfg_v).map_err(err)?;
        models.push(core::fit_best(&ev, &kinds).map_err(err)?.model);
    }
    let cfg = SisConfig {
        cutoffs,
        models,
        n_realizations,
        seed,
        max_neighbors,
        search_radius: radius,
        tail_min: None,
        tail_max: None,
    };
    let res = core::sequential_indicator_simulation(&data, &grid, &cfg).map_err(err)?;
    Ok(res.realizations)
}

/// 3-D experimental semivariogram. Returns `(h, gamma, n_pairs)` lists.
/// `dip` (degrees, positive downward) and `azimuth` enable a directional
/// cone; omit both for an omnidirectional variogram.
#[pyfunction]
#[pyo3(signature = (x, y, z, values, n_lags = 15, max_dist = None,
    azimuth = None, dip = 0.0, tolerance = 22.5))]
#[allow(clippy::too_many_arguments)]
fn experimental_variogram_3d(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    values: Vec<f64>,
    n_lags: usize,
    max_dist: Option<f64>,
    azimuth: Option<f64>,
    dip: f64,
    tolerance: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let data = point_set_3d(x, y, z, values)?;
    let cfg = vario_config(&data, n_lags, max_dist, azimuth, dip, tolerance);
    let ev = core::experimental_variogram(&data, &cfg).map_err(err)?;
    let mut h = Vec::new();
    let mut gamma = Vec::new();
    let mut np = Vec::new();
    for b in &ev.bins {
        h.push(b.h);
        gamma.push(b.gamma);
        np.push(b.n_pairs);
    }
    Ok((h, gamma, np))
}

/// Fits a variogram model to 3-D data by weighted least squares.
#[pyfunction]
#[pyo3(signature = (x, y, z, values, n_lags = 15, max_dist = None, kinds = "best"))]
fn fit_variogram_3d(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    values: Vec<f64>,
    n_lags: usize,
    max_dist: Option<f64>,
    kinds: &str,
) -> PyResult<VariogramModel> {
    let data = point_set_3d(x, y, z, values)?;
    let cfg = vario_config(&data, n_lags, max_dist, None, 0.0, 22.5);
    let ev = core::experimental_variogram(&data, &cfg).map_err(err)?;
    let fit = core::fit_best(&ev, &parse_kinds(kinds)?).map_err(err)?;
    Ok(VariogramModel { inner: fit.model })
}

/// 3-D kriging at arbitrary target locations. Returns
/// `(predictions, variances)`; failed targets yield NaN. `method` is
/// "ordinary", "simple" or "universal".
#[pyfunction]
#[pyo3(signature = (x, y, z, values, model, target_x, target_y, target_z,
    method = "ordinary", mean = None, degree = 1, max_neighbors = None, radius = None))]
#[allow(clippy::too_many_arguments)]
fn krige_3d(
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    values: Vec<f64>,
    model: &VariogramModel,
    target_x: Vec<f64>,
    target_y: Vec<f64>,
    target_z: Vec<f64>,
    method: &str,
    mean: Option<f64>,
    degree: u8,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if target_x.len() != target_y.len() || target_x.len() != target_z.len() {
        return Err(PyValueError::new_err(
            "target_x, target_y and target_z differ in length",
        ));
    }
    let data = point_set_3d(x, y, z, values)?;
    let kmethod = match method {
        "ordinary" => KrigingMethod::Ordinary,
        "simple" => KrigingMethod::Simple {
            mean: mean.unwrap_or_else(|| data.mean()),
        },
        "universal" => KrigingMethod::Universal { degree },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown method '{other}' (expected ordinary, simple or universal)"
            )));
        }
    };
    let config = KrigingConfig {
        method: kmethod,
        max_neighbors,
        search_radius: radius,
    };
    let kriging: Kriging<'_, 3> = Kriging::new(&data, &model.inner, config).map_err(err)?;
    let targets: Vec<[f64; 3]> = target_x
        .into_iter()
        .zip(target_y)
        .zip(target_z)
        .map(|((tx, ty), tz)| [tx, ty, tz])
        .collect();
    let ests = kriging.predict_many(&targets);
    Ok(ests.into_iter().map(|e| (e.value, e.variance)).unzip())
}

/// Indicator kriging at arbitrary target locations. Returns a dict with
/// `ccdf` (list of per-target ccdf lists), `e_type` and `cond_var`.
/// Indicator variogram models are fitted automatically per cutoff.
#[pyfunction]
#[pyo3(signature = (x, y, values, cutoffs, target_x, target_y,
    max_neighbors = None, radius = None, n_lags = 15, max_dist = None))]
#[allow(clippy::too_many_arguments)]
fn indicator_kriging(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    cutoffs: Vec<f64>,
    target_x: Vec<f64>,
    target_y: Vec<f64>,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
    n_lags: usize,
    max_dist: Option<f64>,
) -> PyResult<Py<PyDict>> {
    if target_x.len() != target_y.len() {
        return Err(PyValueError::new_err(
            "target_x and target_y differ in length",
        ));
    }
    let data = point_set(x, y, values)?;
    let cfg_v = vario_config(&data, n_lags, max_dist, None, 0.0, 22.5);
    let kinds = [core::ModelKind::Spherical, core::ModelKind::Exponential];
    let mut models = Vec::with_capacity(cutoffs.len());
    for &c in &cutoffs {
        let indicators: Vec<f64> = data
            .values()
            .iter()
            .map(|&v| if v <= c { 1.0 } else { 0.0 })
            .collect();
        let ind = PointSet::new(data.coords().to_vec(), indicators).map_err(err)?;
        let ev = core::experimental_variogram(&ind, &cfg_v).map_err(err)?;
        models.push(core::fit_best(&ev, &kinds).map_err(err)?.model);
    }
    let cfg = core::IkConfig {
        cutoffs,
        models,
        max_neighbors,
        search_radius: radius,
        tail_min: None,
        tail_max: None,
    };
    let targets: Vec<[f64; 2]> = target_x
        .into_iter()
        .zip(target_y)
        .map(|(tx, ty)| [tx, ty])
        .collect();
    let ests = core::indicator_kriging(&data, &targets, &cfg).map_err(err)?;
    let out = PyDict::new(py);
    let ccdf: Vec<Vec<f64>> = ests.iter().map(|e| e.ccdf.clone()).collect();
    let e_type: Vec<f64> = ests.iter().map(|e| e.e_type).collect();
    let cond_var: Vec<f64> = ests.iter().map(|e| e.cond_var).collect();
    out.set_item("ccdf", ccdf)?;
    out.set_item("e_type", e_type)?;
    out.set_item("cond_var", cond_var)?;
    Ok(out.into())
}
#[pymodule]
fn geostat_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VariogramModel>()?;
    m.add_function(wrap_pyfunction!(experimental_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(variogram_map, m)?)?;
    m.add_function(wrap_pyfunction!(fit_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(fit_anisotropic, m)?)?;
    m.add_function(wrap_pyfunction!(krige, m)?)?;
    m.add_function(wrap_pyfunction!(krige_grid, m)?)?;
    m.add_function(wrap_pyfunction!(loo_cv, m)?)?;
    m.add_function(wrap_pyfunction!(regression_kriging, m)?)?;
    m.add_function(wrap_pyfunction!(idw, m)?)?;
    m.add_function(wrap_pyfunction!(knn, m)?)?;
    m.add_function(wrap_pyfunction!(co_kriging, m)?)?;
    m.add_function(wrap_pyfunction!(compare_methods, m)?)?;
    m.add_function(wrap_pyfunction!(tune_idw_power, m)?)?;
    m.add_function(wrap_pyfunction!(tune_knn_k, m)?)?;
    m.add_function(wrap_pyfunction!(tune_kriging_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(sgs, m)?)?;
    m.add_function(wrap_pyfunction!(sis, m)?)?;
    m.add_function(wrap_pyfunction!(experimental_variogram_3d, m)?)?;
    m.add_function(wrap_pyfunction!(fit_variogram_3d, m)?)?;
    m.add_function(wrap_pyfunction!(krige_3d, m)?)?;
    m.add_function(wrap_pyfunction!(indicator_kriging, m)?)?;
    m.add_function(wrap_pyfunction!(lognormal_kriging, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
