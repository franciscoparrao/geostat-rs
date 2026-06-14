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

/// Leave-one-out cross-validation. Returns a dict with `me`, `mae`, `rmse`,
/// `msdr`, `predicted` and `variance`.
#[pyfunction]
#[pyo3(signature = (x, y, values, model, method = "ordinary", mean = None,
    degree = 1, max_neighbors = None, radius = None))]
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
) -> PyResult<Py<PyDict>> {
    let data = point_set(x, y, values)?;
    let config = KrigingConfig {
        method: build_method(method, mean, degree, &data)?,
        max_neighbors,
        search_radius: radius,
    };
    let cv = core::leave_one_out(&data, &model.inner, &config).map_err(err)?;
    let out = PyDict::new(py);
    out.set_item("me", cv.mean_error())?;
    out.set_item("mae", cv.mae())?;
    out.set_item("rmse", cv.rmse())?;
    out.set_item("msdr", cv.msdr())?;
    out.set_item("predicted", &cv.predicted)?;
    out.set_item("variance", &cv.variance)?;
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
    m.add_function(wrap_pyfunction!(fit_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(krige, m)?)?;
    m.add_function(wrap_pyfunction!(krige_grid, m)?)?;
    m.add_function(wrap_pyfunction!(loo_cv, m)?)?;
    m.add_function(wrap_pyfunction!(sgs, m)?)?;
    m.add_function(wrap_pyfunction!(sis, m)?)?;
    m.add_function(wrap_pyfunction!(experimental_variogram_3d, m)?)?;
    m.add_function(wrap_pyfunction!(fit_variogram_3d, m)?)?;
    m.add_function(wrap_pyfunction!(krige_3d, m)?)?;
    m.add_function(wrap_pyfunction!(indicator_kriging, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
