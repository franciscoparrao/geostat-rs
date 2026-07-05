//! Python bindings for geostat-core (module name: `geostat_rs`).
//!
//! Inputs are plain sequences of floats (lists, tuples or 1-D numpy
//! arrays). Array-shaped outputs (grid predictions, simulation
//! realizations, cross-validation arrays) are returned as owned numpy
//! arrays (via the `numpy` crate) rather than Python lists. Every function
//! doing non-trivial computation releases the GIL for that part
//! (`Python::allow_threads`), so other Python threads keep running while a
//! krige/vecchia/SGS/SIS call is in flight. Build with maturin (`maturin
//! develop`) or copy the cdylib as `geostat_rs.so`.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use geostat_core as core;
use geostat_core::{
    Anisotropy, DirectionConfig, Grid2D, Kriging, KrigingConfig, KrigingMethod, PointSet,
    SgsConfig, SisConfig, VariogramConfig,
};

/// Builds `KrigingConfig::anisotropic_search` from the CLI-mirroring
/// `search_azimuth`/`search_ratio`/`search_ratio_z`/`search_dip`/
/// `search_rake` parameters shared by `krige`/`krige_grid`: `None` unless
/// `search_azimuth` is given.
#[allow(clippy::too_many_arguments)]
fn anisotropic_search(
    search_azimuth: Option<f64>,
    search_ratio: f64,
    search_ratio_z: f64,
    search_dip: f64,
    search_rake: f64,
) -> Option<Anisotropy> {
    search_azimuth.map(|azimuth_deg| Anisotropy {
        azimuth_deg,
        ratio: search_ratio,
        ratio_z: search_ratio_z,
        dip_deg: search_dip,
        rake_deg: search_rake,
    })
}

/// A pair of same-length 1-D numpy arrays (predictions, variances).
type ArrayPair = (Py<PyArray1<f64>>, Py<PyArray1<f64>>);

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Moves a `Vec<f64>` into an owned 1-D numpy array with no extra copy.
fn arr1(py: Python<'_>, v: Vec<f64>) -> Py<PyArray1<f64>> {
    v.into_pyarray(py).unbind()
}

/// Stacks `rows` (all the same length) into an owned `rows.len() x
/// rows[0].len()` numpy array (e.g. simulation realizations, one row each).
fn arr2(py: Python<'_>, rows: Vec<Vec<f64>>) -> PyResult<Py<PyArray2<f64>>> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    if flat.len() != nrows * ncols {
        return Err(err("all rows must have the same length"));
    }
    Ok(flat
        .into_pyarray(py)
        .reshape([nrows, ncols])
        .map_err(err)?
        .unbind())
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
    VariogramConfig::for_data(
        data,
        n_lags,
        max_dist,
        azimuth.map(|az| DirectionConfig {
            azimuth_deg: az,
            dip_deg: dip,
            tolerance_deg: tolerance,
        }),
    )
}

fn parse_kinds(spec: &str) -> PyResult<Vec<core::ModelKind>> {
    core::ModelKind::parse_list(spec).map_err(err)
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

/// Detrends `data` before variography: `detrend` (1 or 2) removes an OLS
/// polynomial trend in the coordinates; `detrend_drift` (one row of covariate
/// values per point) removes an OLS linear trend in external covariates.
fn maybe_detrend(
    data: PointSet,
    detrend: Option<u8>,
    detrend_drift: Option<Vec<Vec<f64>>>,
) -> PyResult<PointSet> {
    if detrend.is_some() && detrend_drift.is_some() {
        return Err(PyValueError::new_err(
            "detrend and detrend_drift are mutually exclusive",
        ));
    }
    if let Some(rows) = detrend_drift {
        return Ok(core::detrend_external(&data, &rows).map_err(err)?.0);
    }
    if let Some(deg) = detrend {
        return Ok(core::detrend_polynomial(&data, deg).map_err(err)?.0);
    }
    Ok(data)
}

fn parse_estimator(s: &str) -> PyResult<core::EstimatorKind> {
    match s.trim().to_lowercase().as_str() {
        "matheron" => Ok(core::EstimatorKind::Matheron),
        "cressie-hawkins" | "cressie_hawkins" | "ch" => Ok(core::EstimatorKind::CressieHawkins),
        "dowd" => Ok(core::EstimatorKind::Dowd),
        "madogram" => Ok(core::EstimatorKind::Madogram),
        other => Err(PyValueError::new_err(format!(
            "unknown estimator '{other}' (expected matheron, cressie-hawkins, dowd or madogram)"
        ))),
    }
}

/// Experimental semivariogram. Returns `(h, gamma, n_pairs)` lists; empty
/// bins carry NaN gamma. `detrend` (degree 1 or 2) computes the variogram on
/// OLS residuals of a polynomial trend — the correct variography for
/// universal kriging; `detrend_drift` (one row of covariates per point) does
/// the same for an external drift (KED). `estimator` selects the point-pair
/// estimator: "matheron" (default, mean-squared-difference),
/// "cressie-hawkins"/"ch", "dowd" or "madogram" -- the latter three trade
/// some efficiency under Gaussian differences for resistance to a few
/// outlier pairs (GSLIB `gamv`/gstat estimator family).
#[pyfunction]
#[pyo3(signature = (x, y, values, n_lags = 15, max_dist = None, azimuth = None, tolerance = 22.5,
    detrend = None, detrend_drift = None, estimator = "matheron"))]
#[allow(clippy::too_many_arguments)]
fn experimental_variogram(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    n_lags: usize,
    max_dist: Option<f64>,
    azimuth: Option<f64>,
    tolerance: f64,
    detrend: Option<u8>,
    detrend_drift: Option<Vec<Vec<f64>>>,
    estimator: &str,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<usize>)> {
    let data = maybe_detrend(point_set(x, y, values)?, detrend, detrend_drift)?;
    let estimator = parse_estimator(estimator)?;
    py.allow_threads(|| {
        let cfg = vario_config(&data, n_lags, max_dist, azimuth, 0.0, tolerance);
        let ev = core::experimental_variogram_robust(&data, &cfg, estimator).map_err(err)?;
        let mut h = Vec::new();
        let mut gamma = Vec::new();
        let mut np = Vec::new();
        for b in &ev.bins {
            h.push(b.h);
            gamma.push(b.gamma);
            np.push(b.n_pairs);
        }
        Ok((h, gamma, np))
    })
}

/// 2-D variogram map (lag-space semivariance surface) for spotting anisotropy.
/// Returns a dict with `size` (grid side = 2*n_lags+1), `lag_width`, and flat
/// row-major (`iy*size+ix`) lists `hx`, `hy`, `gamma` (NaN where empty) and
/// `n_pairs`. `lag_width` defaults to a fifteenth of the data's
/// bounding-box half-diagonal when omitted (same convention as the CLI's
/// `vmap --lag-width`), rather than an arbitrary fixed distance unit.
#[pyfunction]
#[pyo3(signature = (x, y, values, n_lags = 15, lag_width = None))]
fn variogram_map(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    n_lags: usize,
    lag_width: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let data = point_set(x, y, values)?;
    let (m, hx, hy) = py.allow_threads(|| {
        let lag_width = lag_width.unwrap_or_else(|| core::default_lag_width(&data, n_lags));
        let m = core::variogram_map(&data, n_lags, lag_width).map_err(err)?;
        let (mut hx, mut hy) = (Vec::new(), Vec::new());
        for iy in 0..m.size {
            for ix in 0..m.size {
                let (lx, ly) = m.lag(ix, iy);
                hx.push(lx);
                hy.push(ly);
            }
        }
        Ok::<_, PyErr>((m, hx, hy))
    })?;
    let out = PyDict::new(py);
    out.set_item("size", m.size)?;
    out.set_item("lag_width", m.lag_width)?;
    out.set_item("hx", hx)?;
    out.set_item("hy", hy)?;
    out.set_item("gamma", &m.gamma)?;
    out.set_item("n_pairs", &m.n_pairs)?;
    Ok(out.into())
}

/// Fits a variogram model to the data by weighted least squares. `kinds` is
/// "best" (tries the 6 bounded families: spherical, exponential, gaussian,
/// matern15, matern25, circular) or an explicit comma-separated list, which
/// may also include hole, wave, matern:<nu>, stable:<alpha> and
/// power:<theta>. `detrend`/`detrend_drift` fit the model on OLS trend
/// residuals (see `experimental_variogram`) — use them when the model feeds
/// universal or external-drift kriging.
#[pyfunction]
#[pyo3(signature = (x, y, values, n_lags = 15, max_dist = None, kinds = "best",
    detrend = None, detrend_drift = None))]
#[allow(clippy::too_many_arguments)]
fn fit_variogram(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    n_lags: usize,
    max_dist: Option<f64>,
    kinds: &str,
    detrend: Option<u8>,
    detrend_drift: Option<Vec<Vec<f64>>>,
) -> PyResult<VariogramModel> {
    let data = maybe_detrend(point_set(x, y, values)?, detrend, detrend_drift)?;
    let kinds = parse_kinds(kinds)?;
    let model = py.allow_threads(|| {
        let cfg = vario_config(&data, n_lags, max_dist, None, 0.0, 22.5);
        let ev = core::experimental_variogram(&data, &cfg).map_err(err)?;
        core::fit_best(&ev, &kinds).map_err(err)
    })?;
    Ok(VariogramModel { inner: model.model })
}

/// Fits a geometrically anisotropic variogram model: estimates the major-axis
/// azimuth and the minor/major range ratio from `n_dirs` directional
/// variograms. `kinds` is "best" or a comma-separated family list. The returned
/// model carries the anisotropy; `model.anisotropy()` exposes (azimuth, ratio).
#[pyfunction]
#[pyo3(signature = (x, y, values, n_dirs = 4, n_lags = 15, max_dist = None, kinds = "best"))]
#[allow(clippy::too_many_arguments)]
fn fit_anisotropic(
    py: Python<'_>,
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
    let kinds = parse_kinds(kinds)?;
    let fit = py.allow_threads(|| {
        core::fit_anisotropic(&data, &kinds, n_dirs, n_lags, cfg.max_dist).map_err(err)
    })?;
    Ok(VariogramModel { inner: fit.model })
}

/// Fits a single-structure model by Vecchia maximum likelihood: maximizes the
/// Vecchia-approximated Gaussian log-likelihood (conditioning size `m`) instead
/// of variogram weighted least squares. Scales as O(n m^3), so it fits the
/// covariance to the data likelihood on large `n`. `kind` is one family:
/// spherical, exponential, gaussian, matern15, matern25, circular, hole,
/// wave, matern:<nu> or stable:<alpha> (Power has no covariance function,
/// so it is not supported here — use `fit_variogram` + `krige` instead).
#[pyfunction]
#[pyo3(signature = (x, y, values, kind = "exponential", m = 20))]
fn vecchia_mle(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    kind: &str,
    m: usize,
) -> PyResult<VariogramModel> {
    let data = point_set(x, y, values)?;
    let k = *parse_kinds(kind)?
        .first()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no model kind given"))?;
    let fit = py.allow_threads(|| core::vecchia_mle(&data, k, m, None).map_err(err))?;
    Ok(VariogramModel { inner: fit.model })
}

/// Fits a single-structure model by Vecchia restricted/trend maximum
/// likelihood: the mean is a polynomial trend of `drift_degree` (0 = constant,
/// 1 = linear, 2 = quadratic) estimated by GLS, and the covariance is fit to the
/// error contrasts. Use when the field has a spatial trend (unlike
/// `vecchia_mle`, a trend does not inflate the fitted range).
#[pyfunction]
#[pyo3(signature = (x, y, values, kind = "exponential", m = 20, drift_degree = 1))]
fn vecchia_reml(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    kind: &str,
    m: usize,
    drift_degree: u8,
) -> PyResult<VariogramModel> {
    let data = point_set(x, y, values)?;
    let k = *parse_kinds(kind)?
        .first()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no model kind given"))?;
    let fit =
        py.allow_threads(|| core::vecchia_reml(&data, k, m, drift_degree, None).map_err(err))?;
    Ok(VariogramModel { inner: fit.model })
}

/// Fits a single-structure model by Vecchia external-drift REML: the mean is
/// `beta_0 + sum_j beta_j x_j` over the `covariates` columns (a list of rows,
/// one per point), estimated by GLS while the covariance is fit to the error
/// contrasts. This is kriging-with-external-drift by maximum likelihood --- use
/// when a measured covariate (e.g. distance to a feature) drives the trend.
#[pyfunction]
#[pyo3(signature = (x, y, values, covariates, kind = "exponential", m = 20))]
fn vecchia_reml_drift(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    kind: &str,
    m: usize,
) -> PyResult<VariogramModel> {
    let data = point_set(x, y, values)?;
    let k = *parse_kinds(kind)?
        .first()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no model kind given"))?;
    let fit =
        py.allow_threads(|| core::vecchia_reml_drift(&data, k, m, &covariates, None).map_err(err))?;
    Ok(VariogramModel { inner: fit.model })
}

/// Asymptotic standard errors `(se_nugget, se_sill, se_range)` of a
/// single-structure model's covariance parameters, from the observed Fisher
/// information of the constant-mean Vecchia log-likelihood. A boundary parameter
/// (e.g. zero nugget) or a non-positive-definite information yields NaN.
#[pyfunction]
#[pyo3(signature = (x, y, values, model, m = 20))]
fn vecchia_param_se(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    model: &VariogramModel,
    m: usize,
) -> PyResult<(f64, f64, f64)> {
    let data = point_set(x, y, values)?;
    let se =
        py.allow_threads(|| core::vecchia_param_se(&data, &model.inner, m, None).map_err(err))?;
    Ok((se[0], se[1], se[2]))
}

/// Vecchia prediction at arbitrary targets (Katzfuss-Guinness): targets in
/// max-min order condition on their `m` nearest previous points, observed
/// data and already-processed targets alike, which keeps the joint
/// predictive consistent at small `m`. Simple-kriging mean (the data mean).
/// Scalable to very large data/target sets; equals exact global simple
/// kriging when `m` covers everything. Returns `(predictions, variances)`.
#[pyfunction]
#[pyo3(signature = (x, y, values, model, target_x, target_y, m = 30))]
#[allow(clippy::too_many_arguments)]
fn vecchia_krige(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    model: &VariogramModel,
    target_x: Vec<f64>,
    target_y: Vec<f64>,
    m: usize,
) -> PyResult<ArrayPair> {
    if target_x.len() != target_y.len() {
        return Err(PyValueError::new_err(
            "target_x and target_y differ in length",
        ));
    }
    let data = point_set(x, y, values)?;
    let targets: Vec<[f64; 2]> = target_x
        .into_iter()
        .zip(target_y)
        .map(|(tx, ty)| [tx, ty])
        .collect();
    let (values, variances) = py.allow_threads(|| {
        let ests = core::vecchia_predict(&data, &model.inner, &targets, m).map_err(err)?;
        Ok::<_, PyErr>(ests.into_iter().map(|e| (e.value, e.variance)).unzip())
    })?;
    Ok((arr1(py, values), arr1(py, variances)))
}

/// Vecchia-approximated Gaussian log-likelihood of the data under `model`, with
/// conditioning size `m` (exact when `m >= n-1`).
#[pyfunction]
#[pyo3(signature = (x, y, values, model, m = 20))]
fn vecchia_loglik(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    model: &VariogramModel,
    m: usize,
) -> PyResult<f64> {
    let data = point_set(x, y, values)?;
    py.allow_threads(|| core::vecchia_loglik(&data, &model.inner, m, None).map_err(err))
}

/// Kriging at arbitrary target locations. Returns `(predictions, variances)`;
/// failed targets yield NaN. `min_neighbors` (GSLIB ndmin) fails estimates
/// with too few conditioning points; `octant` (GSLIB noct) caps the
/// neighbors taken per quadrant to balance clustered data. With
/// `search_azimuth` set, the search neighborhood is a rotated ellipsoid
/// (GSLIB kt3d sang1/sanis1) instead of a Euclidean one: `radius` becomes
/// the major-axis radius, and `search_ratio`/`search_dip`/`search_rake`
/// shape it the same way an anisotropic variogram's `ratio`/`dip`/`rake` do.
#[pyfunction]
#[pyo3(signature = (x, y, values, model, target_x, target_y, method = "ordinary",
    mean = None, degree = 1, max_neighbors = None, radius = None,
    min_neighbors = None, octant = None, measurement_error = None,
    search_azimuth = None, search_ratio = 1.0, search_ratio_z = 1.0,
    search_dip = 0.0, search_rake = 0.0))]
#[allow(clippy::too_many_arguments)]
fn krige(
    py: Python<'_>,
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
    min_neighbors: Option<usize>,
    octant: Option<usize>,
    measurement_error: Option<Vec<f64>>,
    search_azimuth: Option<f64>,
    search_ratio: f64,
    search_ratio_z: f64,
    search_dip: f64,
    search_rake: f64,
) -> PyResult<ArrayPair> {
    if target_x.len() != target_y.len() {
        return Err(PyValueError::new_err(
            "target_x and target_y differ in length",
        ));
    }
    let data = point_set(x, y, values)?;
    // `KrigingConfig` is `#[non_exhaustive]`: build from `Default::default()`
    // and assign fields.
    let mut config = KrigingConfig::default();
    config.method = build_method(method, mean, degree, &data)?;
    config.max_neighbors = max_neighbors;
    config.search_radius = radius;
    config.min_neighbors = min_neighbors;
    config.max_per_octant = octant;
    config.anisotropic_search = anisotropic_search(
        search_azimuth,
        search_ratio,
        search_ratio_z,
        search_dip,
        search_rake,
    );
    let targets: Vec<[f64; 2]> = target_x
        .into_iter()
        .zip(target_y)
        .map(|(tx, ty)| [tx, ty])
        .collect();
    let (values, variances) = py.allow_threads(|| {
        let kriging = match measurement_error {
            Some(errors) => {
                Kriging::with_measurement_error(&data, &model.inner, config, errors).map_err(err)?
            }
            None => Kriging::new(&data, &model.inner, config).map_err(err)?,
        };
        let ests = kriging.predict_many(&targets);
        Ok::<_, PyErr>(ests.into_iter().map(|e| (e.value, e.variance)).unzip())
    })?;
    Ok((arr1(py, values), arr1(py, variances)))
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
    py: Python<'_>,
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
) -> PyResult<ArrayPair> {
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
    // `KrigingConfig` is `#[non_exhaustive]`: build from `Default::default()`
    // and assign fields.
    let mut config = KrigingConfig::default();
    config.method = kmethod;
    config.max_neighbors = max_neighbors;
    config.search_radius = radius;
    let targets: Vec<[f64; 2]> = target_x
        .into_iter()
        .zip(target_y)
        .map(|(tx, ty)| [tx, ty])
        .collect();
    let (values, variances) = py.allow_threads(|| {
        let ests =
            core::lognormal_kriging(&data, &targets, &log_model.inner, &config).map_err(err)?;
        Ok::<_, PyErr>(ests.into_iter().map(|e| (e.value, e.log_variance)).unzip())
    })?;
    Ok((arr1(py, values), arr1(py, variances)))
}

/// Kriging over a regular grid (`bbox = (xmin, ymin, xmax, ymax)`), row-major
/// cell centers with y increasing. Returns `(predictions, variances)`. With
/// `search_azimuth` set, the search neighborhood is a rotated ellipsoid
/// instead of a Euclidean one -- see `krige`.
#[pyfunction]
#[pyo3(signature = (x, y, values, model, bbox, nx, ny, method = "ordinary",
    mean = None, degree = 1, max_neighbors = None, radius = None, block = None, block_discr = (4, 4),
    min_neighbors = None, octant = None, search_azimuth = None, search_ratio = 1.0,
    search_ratio_z = 1.0, search_dip = 0.0, search_rake = 0.0))]
#[allow(clippy::too_many_arguments)]
fn krige_grid(
    py: Python<'_>,
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
    min_neighbors: Option<usize>,
    octant: Option<usize>,
    search_azimuth: Option<f64>,
    search_ratio: f64,
    search_ratio_z: f64,
    search_dip: f64,
    search_rake: f64,
) -> PyResult<ArrayPair> {
    let data = point_set(x, y, values)?;
    let grid = Grid2D::from_bbox([bbox.0, bbox.1], [bbox.2, bbox.3], nx, ny).map_err(err)?;
    // `KrigingConfig` is `#[non_exhaustive]`: build from `Default::default()`
    // and assign fields.
    let mut config = KrigingConfig::default();
    config.method = build_method(method, mean, degree, &data)?;
    config.max_neighbors = max_neighbors;
    config.search_radius = radius;
    config.min_neighbors = min_neighbors;
    config.max_per_octant = octant;
    config.anisotropic_search = anisotropic_search(
        search_azimuth,
        search_ratio,
        search_ratio_z,
        search_dip,
        search_rake,
    );
    let (values, variances) = py.allow_threads(|| {
        let kriging = Kriging::new(&data, &model.inner, config).map_err(err)?;
        match block {
            Some((bw, bh)) => kriging
                .predict_block_grid(&grid, [bw, bh], [block_discr.0, block_discr.1])
                .map_err(err),
            None => Ok(kriging.predict_grid(&grid)),
        }
    })?;
    Ok((arr1(py, values), arr1(py, variances)))
}

/// Cross-validation. Leave-one-out by default; pass `folds=k` for `k`-fold
/// (faster on large datasets, reproducible via `seed`), or `blocks=(nx,ny)`
/// for spatial block CV (partitions the domain into an `nx`-by-`ny` grid and
/// holds out whole blocks at a time — a more honest error estimate than
/// random k-fold/LOO under spatial autocorrelation; `blocks` and `folds` are
/// mutually exclusive). Returns a dict with `me`, `mae`, `rmse`, `msdr`,
/// `vecv`, `e1`, `observed`, `predicted` and `variance`.
#[pyfunction]
#[pyo3(signature = (x, y, values, model, method = "ordinary", mean = None,
    degree = 1, max_neighbors = None, radius = None, folds = None, seed = 0,
    min_neighbors = None, octant = None, blocks = None))]
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
    min_neighbors: Option<usize>,
    octant: Option<usize>,
    blocks: Option<(usize, usize)>,
) -> PyResult<Py<PyDict>> {
    let data = point_set(x, y, values)?;
    // `KrigingConfig` is `#[non_exhaustive]`: build from `Default::default()`
    // and assign fields.
    let mut config = KrigingConfig::default();
    config.method = build_method(method, mean, degree, &data)?;
    config.max_neighbors = max_neighbors;
    config.search_radius = radius;
    config.min_neighbors = min_neighbors;
    config.max_per_octant = octant;
    if blocks.is_some() && folds.is_some() {
        return Err(PyValueError::new_err(
            "blocks and folds are mutually exclusive",
        ));
    }
    let cv = py.allow_threads(|| match (blocks, folds) {
        (Some((nx, ny)), _) => core::block_cv(&data, &model.inner, &config, [nx, ny]).map_err(err),
        (None, Some(k)) => core::k_fold(&data, &model.inner, &config, k, seed).map_err(err),
        (None, None) => core::leave_one_out(&data, &model.inner, &config).map_err(err),
    })?;
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
    out.set_item("observed", arr1(py, cv.observed))?;
    out.set_item("predicted", arr1(py, cv.predicted))?;
    out.set_item("variance", arr1(py, cv.variance))?;
    Ok(out.into())
}

/// Deutsch (1997) accuracy plot: checks whether a model's *uncertainty* (not
/// just its central prediction) is well calibrated. `actual`/`mean`/`std`
/// are typically a cross-validation's `observed`/`predicted`/
/// `sqrt(variance)` (see `loo_cv`); `probs` are the nominal probabilities to
/// check (default: 0.1..0.9 step 0.1). Returns a dict with `nominal`,
/// `observed` (lists, one entry per probability) and `goodness` (the
/// calibration statistic, 1.0 = perfect).
#[pyfunction]
#[pyo3(signature = (actual, mean, std, probs = None))]
fn accuracy_plot(
    py: Python<'_>,
    actual: Vec<f64>,
    mean: Vec<f64>,
    std: Vec<f64>,
    probs: Option<Vec<f64>>,
) -> PyResult<Py<PyDict>> {
    let probs = probs.unwrap_or_else(|| (1..=9).map(|i| i as f64 * 0.1).collect());
    let plot = core::accuracy_plot(&actual, &mean, &std, &probs).map_err(err)?;
    let out = PyDict::new(py);
    out.set_item(
        "nominal",
        plot.points.iter().map(|p| p.nominal).collect::<Vec<_>>(),
    )?;
    out.set_item(
        "observed",
        plot.points.iter().map(|p| p.observed).collect::<Vec<_>>(),
    )?;
    out.set_item("goodness", plot.goodness)?;
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

    // `KrigingConfig` is `#[non_exhaustive]`: build from `Default::default()`
    // and assign fields.
    let mut config = KrigingConfig::default();
    config.method = KrigingMethod::Ordinary;
    config.max_neighbors = max_neighbors;
    config.search_radius = radius;
    let (prediction, variance) = py.allow_threads(|| {
        let rk = RegressionKriging::new(&data, &trend_data).map_err(err)?;
        let cfg = vario_config(rk.residuals(), n_lags, max_dist, None, 0.0, 22.5);
        let ev = core::experimental_variogram(rk.residuals(), &cfg).map_err(err)?;
        let resid_model = core::fit_best(&ev, &core::ModelKind::ALL)
            .map_err(err)?
            .model;
        let ests = rk
            .predict(&targets, &trend_targets, &resid_model, &config)
            .map_err(err)?;
        let prediction: Vec<f64> = ests.iter().map(|e| e.value).collect();
        let variance: Vec<f64> = ests.iter().map(|e| e.variance).collect();
        Ok::<_, PyErr>((prediction, variance))
    })?;
    out.set_item("prediction", arr1(py, prediction))?;
    out.set_item("variance", arr1(py, variance))?;
    Ok(out.into())
}

/// Ordinary co-kriging of a primary variable using a correlated secondary,
/// under a linear model of coregionalization fitted automatically. The two
/// variables may have different supports (heterotopic): the direct variograms
/// use each variable's own points and the cross-variogram is fitted on the
/// collocated subset (points shared by both, matched on exact coordinates).
/// `ridge` (default 0.0: the exact system, matching the CLI and the gstat
/// validation) inflates the co-kriging matrix diagonal; set e.g. 1e-2 to
/// stabilize notoriously ill-conditioned heterotopic systems.
/// Returns `(predictions, variances)` of the primary at the targets.
#[pyfunction]
#[pyo3(signature = (px, py, pv, sx, sy, sv, target_x, target_y,
    n_lags = 15, max_dist = None, max_neighbors = None, radius = None, ridge = 0.0))]
#[allow(clippy::too_many_arguments)]
fn co_kriging(
    gil: Python<'_>,
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
) -> PyResult<ArrayPair> {
    if target_x.len() != target_y.len() {
        return Err(PyValueError::new_err(
            "target_x and target_y differ in length",
        ));
    }
    let primary = point_set(px, py, pv)?;
    let secondary = point_set(sx, sy, sv)?;
    let targets: Vec<[f64; 2]> = target_x
        .iter()
        .zip(&target_y)
        .map(|(&a, &b)| [a, b])
        .collect();
    let (values, variances) = gil.allow_threads(|| {
        let cfg = vario_config(&primary, n_lags, max_dist, None, 0.0, 22.5);
        let lmc = core::fit_lmc_collocated(&primary, &secondary, &cfg, &core::ModelKind::ALL)
            .map_err(err)?;
        // `CoKrigingConfig` is `#[non_exhaustive]`: build from
        // `Default::default()` and assign fields.
        let mut config = core::CoKrigingConfig::default();
        config.max_neighbors = max_neighbors;
        config.search_radius = radius;
        config.ridge = ridge;
        let ck = core::CoKriging::new(vec![&primary, &secondary], &lmc, config).map_err(err)?;
        let ests = ck.predict_many(&targets);
        Ok::<_, PyErr>(ests.into_iter().map(|e| (e.value, e.variance)).unzip())
    })?;
    Ok((arr1(gil, values), arr1(gil, variances)))
}

/// Inverse-distance weighting at the targets. `power` controls locality
/// (2 is typical); exact at the data. Returns a list of predictions.
#[pyfunction]
#[pyo3(signature = (x, y, values, target_x, target_y, power = 2.0,
    max_neighbors = None, radius = None))]
#[allow(clippy::too_many_arguments)]
fn idw(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    target_x: Vec<f64>,
    target_y: Vec<f64>,
    power: f64,
    max_neighbors: Option<usize>,
    radius: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
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
    let pred = py.allow_threads(|| {
        let pred = core::Idw::new(&data, power, max_neighbors, radius).map_err(err)?;
        Ok::<_, PyErr>(pred.predict_many(&targets))
    })?;
    Ok(arr1(py, pred))
}

/// k-nearest-neighbor averaging at the targets (`k = 1` is nearest-neighbor).
/// Returns a list of predictions.
#[pyfunction]
#[pyo3(signature = (x, y, values, target_x, target_y, k = 8, radius = None))]
#[allow(clippy::too_many_arguments)]
fn knn(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    target_x: Vec<f64>,
    target_y: Vec<f64>,
    k: usize,
    radius: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
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
    let pred = py.allow_threads(|| {
        let pred = core::Knn::new(&data, k, radius).map_err(err)?;
        Ok::<_, PyErr>(pred.predict_many(&targets))
    })?;
    Ok(arr1(py, pred))
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
    // `KrigingConfig` is `#[non_exhaustive]`: build from `Default::default()`
    // and assign fields.
    let mut ok_config = KrigingConfig::default();
    ok_config.method = KrigingMethod::Ordinary;
    ok_config.max_neighbors = max_neighbors;
    ok_config.search_radius = radius;

    let entries = py.allow_threads(|| {
        let cfg = vario_config(&data, n_lags, max_dist, None, 0.0, 22.5);
        let ev = core::experimental_variogram(&data, &cfg).map_err(err)?;
        let model = core::fit_best(&ev, &core::ModelKind::ALL)
            .map_err(err)?
            .model;
        Ok::<_, PyErr>([
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
        ])
    })?;

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
    let res = py
        .allow_threads(|| core::tune_idw_power(&data, &powers, max_neighbors, radius))
        .map_err(err)?;
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
    let res = py
        .allow_threads(|| core::tune_knn_k(&data, &ks, radius))
        .map_err(err)?;
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
    let res = py.allow_threads(|| {
        let cfg = vario_config(&data, n_lags, max_dist, None, 0.0, 22.5);
        let ev = core::experimental_variogram(&data, &cfg).map_err(err)?;
        let model = core::fit_best(&ev, &core::ModelKind::ALL)
            .map_err(err)?
            .model;
        core::tune_kriging_neighbors(&data, &model, KrigingMethod::Ordinary, &candidates, radius)
            .map_err(err)
    })?;
    let out = PyDict::new(py);
    out.set_item("best", res.best)?;
    out.set_item("best_vecv", res.best_vecv)?;
    out.set_item("trace", res.trace)?;
    Ok(out.into())
}

/// Parses a GSLIB-style tail spec ("none", "linear", "power:<w>",
/// "hyper:<w>") into a core TailModel.
fn parse_tail(spec: &str) -> PyResult<core::TailModel> {
    spec.parse::<core::TailModel>().map_err(err)
}

/// Cell-declustering weights (GSLIB declus) for preferentially sampled
/// data. With `cell_size` computes the weights directly; otherwise scans
/// `n_sizes` sizes between `min_size`/`max_size` (default: bbox diagonal /
/// 50 to / 5) and keeps the size that minimizes the declustered mean (set
/// `minimize=False` for data clustered in low values). Returns a dict with
/// `weights` (sum to n), `cell_size`, `declustered_mean` and the scan
/// `trace` as a list of `(size, mean)` pairs.
#[pyfunction]
#[pyo3(signature = (x, y, values, cell_size = None, min_size = None, max_size = None,
    n_sizes = 20, n_offsets = 4, minimize = true))]
#[allow(clippy::too_many_arguments)]
fn decluster_weights(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    values: Vec<f64>,
    cell_size: Option<f64>,
    min_size: Option<f64>,
    max_size: Option<f64>,
    n_sizes: usize,
    n_offsets: usize,
    minimize: bool,
) -> PyResult<Py<PyDict>> {
    let data = point_set(x, y, values)?;
    enum Res {
        Cell {
            weights: Vec<f64>,
            size: f64,
            mean: f64,
        },
        Scan {
            weights: Vec<f64>,
            size: f64,
            mean: f64,
            trace: Vec<(f64, f64)>,
        },
    }
    let res = py.allow_threads(|| -> PyResult<Res> {
        match cell_size {
            Some(size) => {
                let w = core::cell_declustering_weights(&data, size, n_offsets).map_err(err)?;
                let mean = w
                    .iter()
                    .zip(data.values())
                    .map(|(&wi, &v)| wi * v)
                    .sum::<f64>()
                    / data.len() as f64;
                Ok(Res::Cell {
                    weights: w,
                    size,
                    mean,
                })
            }
            None => {
                let (min, max) = data.bbox();
                let diag = ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2)).sqrt();
                let lo = min_size.unwrap_or(diag / 50.0);
                let hi = max_size.unwrap_or(diag / 5.0);
                let scan = core::decluster_scan(&data, lo, hi, n_sizes, n_offsets, minimize)
                    .map_err(err)?;
                Ok(Res::Scan {
                    weights: scan.weights,
                    size: scan.best_size,
                    mean: scan.best_mean,
                    trace: scan.trace,
                })
            }
        }
    })?;
    let out = PyDict::new(py);
    match res {
        Res::Cell {
            weights,
            size,
            mean,
        } => {
            out.set_item("weights", arr1(py, weights))?;
            out.set_item("cell_size", size)?;
            out.set_item("declustered_mean", mean)?;
            out.set_item("trace", Vec::<(f64, f64)>::new())?;
        }
        Res::Scan {
            weights,
            size,
            mean,
            trace,
        } => {
            out.set_item("weights", arr1(py, weights))?;
            out.set_item("cell_size", size)?;
            out.set_item("declustered_mean", mean)?;
            out.set_item("trace", trace)?;
        }
    }
    Ok(out.into())
}

/// Conditional sequential Gaussian simulation. `model_ns` is a model fitted
/// to the normal scores (fit one with `fit_variogram` on transformed data,
/// or let the CLI auto-fit). Returns one list per realization, in grid
/// storage order.
///
/// By default realizations are clamped to the data range; `ltail`/`utail`
/// ("linear", "power:<w>", "hyper:<w>", GSLIB ltail/utail — same names as
/// `sis`/`indicator_kriging`) with `zmin`/`zmax` extrapolate the
/// back-transform beyond the data extremes — without them the simulated
/// variance and extreme quantiles are systematically truncated.
#[pyfunction]
#[pyo3(signature = (x, y, values, model_ns, bbox, nx, ny, n_realizations = 10,
    seed = 42, max_neighbors = 16, radius = None,
    ltail = "none", utail = "none", zmin = None, zmax = None,
    decluster_cell = None, max_node_neighbors = None, multigrid = 0))]
#[allow(clippy::too_many_arguments)]
fn sgs(
    py: Python<'_>,
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
    ltail: &str,
    utail: &str,
    zmin: Option<f64>,
    zmax: Option<f64>,
    decluster_cell: Option<f64>,
    max_node_neighbors: Option<usize>,
    multigrid: u8,
) -> PyResult<Py<PyArray2<f64>>> {
    let data = point_set(x, y, values)?;
    let grid = Grid2D::from_bbox([bbox.0, bbox.1], [bbox.2, bbox.3], nx, ny).map_err(err)?;
    let decluster_weights = match decluster_cell {
        Some(size) => Some(core::cell_declustering_weights(&data, size, 4).map_err(err)?),
        None => None,
    };
    let tails = core::Tails {
        lower: parse_tail(ltail)?,
        upper: parse_tail(utail)?,
        lower_bound: zmin,
        upper_bound: zmax,
    };
    // `SgsConfig` is `#[non_exhaustive]`: build from `Default::default()`
    // and assign fields.
    let mut cfg = SgsConfig::default();
    cfg.n_realizations = n_realizations;
    cfg.seed = seed;
    cfg.max_neighbors = max_neighbors;
    cfg.search_radius = radius;
    cfg.tails = tails;
    cfg.decluster_weights = decluster_weights;
    cfg.max_node_neighbors = max_node_neighbors;
    cfg.multigrid = multigrid;
    let realizations = py.allow_threads(|| {
        core::sequential_gaussian_simulation(&data, &model_ns.inner, &grid, &cfg)
            .map_err(err)
            .map(|res| res.realizations)
    })?;
    arr2(py, realizations)
}

/// Conditional sequential indicator simulation. Indicator variogram models
/// are fitted automatically at each cutoff, or a single shared model at the
/// median cutoff when `mik=True` (GSLIB `mik=1` median IK — amortizes one
/// factorization across every cutoff). `fit` is a comma-separated list of
/// candidate families for that auto-fit (same spec as the CLI's `--fit`:
/// "spherical", "exponential", "gaussian", "matern15", "matern25",
/// "circular", "hole", "wave", "matern:<nu>", "stable:<alpha>",
/// "power:<theta>" — the best-fitting one is picked per cutoff).
/// `ltail`/`utail` ("linear", "power:<w>", "hyper:<w>") set the GSLIB tail
/// interpolation between `tail_min`/`tail_max` (default: data extremes) and
/// the extreme cutoffs; hyperbolic upper tails are capped at `tail_max`.
/// `ordinary=True` uses ordinary (Σw=1) instead of simple indicator kriging.
/// `decluster_cell`, `max_node_neighbors` and `multigrid` mirror `sgs`'s
/// same-named parameters (GSLIB declus/nodmax/nmult).
#[pyfunction]
#[pyo3(signature = (x, y, values, cutoffs, bbox, nx, ny, n_realizations = 10,
    seed = 42, max_neighbors = 16, radius = None, n_lags = 15, max_dist = None,
    fit = "spherical,exponential", ltail = "linear", utail = "linear",
    tail_min = None, tail_max = None, mik = false, ordinary = false,
    decluster_cell = None, max_node_neighbors = None, multigrid = 0))]
#[allow(clippy::too_many_arguments)]
fn sis(
    py: Python<'_>,
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
    fit: &str,
    ltail: &str,
    utail: &str,
    tail_min: Option<f64>,
    tail_max: Option<f64>,
    mik: bool,
    ordinary: bool,
    decluster_cell: Option<f64>,
    max_node_neighbors: Option<usize>,
    multigrid: u8,
) -> PyResult<Py<PyArray2<f64>>> {
    let data = point_set(x, y, values)?;
    let grid = Grid2D::from_bbox([bbox.0, bbox.1], [bbox.2, bbox.3], nx, ny).map_err(err)?;
    let cfg_v = vario_config(&data, n_lags, max_dist, None, 0.0, 22.5);
    let kinds = core::ModelKind::parse_list(fit).map_err(err)?;
    let models = if mik {
        core::fit_median_indicator_model(&data, &cutoffs, &kinds, &cfg_v).map_err(err)?
    } else {
        core::fit_indicator_models(&data, &cutoffs, &kinds, &cfg_v).map_err(err)?
    };
    let lower_tail = parse_tail(ltail)?;
    let upper_tail = parse_tail(utail)?;
    let decluster_weights = match decluster_cell {
        Some(size) => Some(core::cell_declustering_weights(&data, size, 4).map_err(err)?),
        None => None,
    };
    // `SisConfig` is `#[non_exhaustive]`: build from `Default::default()`
    // and assign fields.
    let mut cfg = SisConfig::default();
    cfg.cutoffs = cutoffs;
    cfg.models = models;
    cfg.ordinary = ordinary;
    cfg.n_realizations = n_realizations;
    cfg.seed = seed;
    cfg.max_neighbors = max_neighbors;
    cfg.search_radius = radius;
    cfg.tail_min = tail_min;
    cfg.tail_max = tail_max;
    cfg.lower_tail = lower_tail;
    cfg.upper_tail = upper_tail;
    cfg.decluster_weights = decluster_weights;
    cfg.max_node_neighbors = max_node_neighbors;
    cfg.multigrid = multigrid;
    let realizations = py.allow_threads(|| {
        core::sequential_indicator_simulation(&data, &grid, &cfg)
            .map_err(err)
            .map(|res| res.realizations)
    })?;
    arr2(py, realizations)
}

/// 3-D experimental semivariogram. Returns `(h, gamma, n_pairs)` lists.
/// `dip` (degrees; same sign convention as a fitted model's rotation --
/// GSLIB ang2 / gstat) and `azimuth` enable a directional cone; omit both
/// for an omnidirectional variogram.
#[pyfunction]
#[pyo3(signature = (x, y, z, values, n_lags = 15, max_dist = None,
    azimuth = None, dip = 0.0, tolerance = 22.5))]
#[allow(clippy::too_many_arguments)]
fn experimental_variogram_3d(
    py: Python<'_>,
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
    py.allow_threads(|| {
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
    })
}

/// Fits a variogram model to 3-D data by weighted least squares.
#[pyfunction]
#[pyo3(signature = (x, y, z, values, n_lags = 15, max_dist = None, kinds = "best"))]
#[allow(clippy::too_many_arguments)]
fn fit_variogram_3d(
    py: Python<'_>,
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    values: Vec<f64>,
    n_lags: usize,
    max_dist: Option<f64>,
    kinds: &str,
) -> PyResult<VariogramModel> {
    let data = point_set_3d(x, y, z, values)?;
    let kinds = parse_kinds(kinds)?;
    let fit = py.allow_threads(|| {
        let cfg = vario_config(&data, n_lags, max_dist, None, 0.0, 22.5);
        let ev = core::experimental_variogram(&data, &cfg).map_err(err)?;
        core::fit_best(&ev, &kinds).map_err(err)
    })?;
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
    py: Python<'_>,
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
) -> PyResult<ArrayPair> {
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
    // `KrigingConfig` is `#[non_exhaustive]`: build from `Default::default()`
    // and assign fields.
    let mut config = KrigingConfig::default();
    config.method = kmethod;
    config.max_neighbors = max_neighbors;
    config.search_radius = radius;
    let targets: Vec<[f64; 3]> = target_x
        .into_iter()
        .zip(target_y)
        .zip(target_z)
        .map(|((tx, ty), tz)| [tx, ty, tz])
        .collect();
    let (values, variances) = py.allow_threads(|| {
        let kriging: Kriging<'_, 3> = Kriging::new(&data, &model.inner, config).map_err(err)?;
        let ests = kriging.predict_many(&targets);
        Ok::<_, PyErr>(ests.into_iter().map(|e| (e.value, e.variance)).unzip())
    })?;
    Ok((arr1(py, values), arr1(py, variances)))
}

/// Indicator kriging at arbitrary target locations. Returns a dict with
/// `ccdf` (list of per-target ccdf lists), `e_type` and `cond_var`.
/// Indicator variogram models are fitted automatically per cutoff, or a
/// single shared model at the median cutoff when `mik=True` (GSLIB `mik=1`
/// median IK). `fit` is a comma-separated list of candidate families for
/// that auto-fit (same spec as the CLI's `--fit`: "spherical", "exponential",
/// "gaussian", "matern15", "matern25", "circular", "hole", "wave",
/// "matern:<nu>", "stable:<alpha>", "power:<theta>"). `ltail`/`utail`
/// ("linear", "power:<w>", "hyper:<w>") set the GSLIB tail interpolation
/// used for the E-type and conditional-variance integrals. `ordinary=True`
/// uses ordinary (Σw=1) instead of simple IK.
#[pyfunction]
#[pyo3(signature = (x, y, values, cutoffs, target_x, target_y,
    max_neighbors = None, radius = None, n_lags = 15, max_dist = None,
    fit = "spherical,exponential", ltail = "linear", utail = "linear",
    tail_min = None, tail_max = None, mik = false, ordinary = false))]
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
    fit: &str,
    ltail: &str,
    utail: &str,
    tail_min: Option<f64>,
    tail_max: Option<f64>,
    mik: bool,
    ordinary: bool,
) -> PyResult<Py<PyDict>> {
    if target_x.len() != target_y.len() {
        return Err(PyValueError::new_err(
            "target_x and target_y differ in length",
        ));
    }
    let data = point_set(x, y, values)?;
    let lower_tail = parse_tail(ltail)?;
    let upper_tail = parse_tail(utail)?;
    let targets: Vec<[f64; 2]> = target_x
        .into_iter()
        .zip(target_y)
        .map(|(tx, ty)| [tx, ty])
        .collect();
    let (ccdf, e_type, cond_var) = py.allow_threads(|| {
        let cfg_v = vario_config(&data, n_lags, max_dist, None, 0.0, 22.5);
        let kinds = core::ModelKind::parse_list(fit).map_err(err)?;
        let models = if mik {
            core::fit_median_indicator_model(&data, &cutoffs, &kinds, &cfg_v).map_err(err)?
        } else {
            core::fit_indicator_models(&data, &cutoffs, &kinds, &cfg_v).map_err(err)?
        };
        // `IkConfig` is `#[non_exhaustive]`: build from `Default::default()`
        // and assign fields.
        let mut cfg = core::IkConfig::default();
        cfg.cutoffs = cutoffs;
        cfg.models = models;
        cfg.ordinary = ordinary;
        cfg.max_neighbors = max_neighbors;
        cfg.search_radius = radius;
        cfg.tail_min = tail_min;
        cfg.tail_max = tail_max;
        cfg.lower_tail = lower_tail;
        cfg.upper_tail = upper_tail;
        let ests = core::indicator_kriging(&data, &targets, &cfg).map_err(err)?;
        let ccdf: Vec<Vec<f64>> = ests.iter().map(|e| e.ccdf.clone()).collect();
        let e_type: Vec<f64> = ests.iter().map(|e| e.e_type).collect();
        let cond_var: Vec<f64> = ests.iter().map(|e| e.cond_var).collect();
        Ok::<_, PyErr>((ccdf, e_type, cond_var))
    })?;
    let out = PyDict::new(py);
    out.set_item("ccdf", arr2(py, ccdf)?)?;
    out.set_item("e_type", arr1(py, e_type))?;
    out.set_item("cond_var", arr1(py, cond_var))?;
    Ok(out.into())
}
#[pymodule]
fn geostat_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VariogramModel>()?;
    m.add_function(wrap_pyfunction!(experimental_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(variogram_map, m)?)?;
    m.add_function(wrap_pyfunction!(fit_variogram, m)?)?;
    m.add_function(wrap_pyfunction!(fit_anisotropic, m)?)?;
    m.add_function(wrap_pyfunction!(vecchia_mle, m)?)?;
    m.add_function(wrap_pyfunction!(vecchia_reml, m)?)?;
    m.add_function(wrap_pyfunction!(vecchia_reml_drift, m)?)?;
    m.add_function(wrap_pyfunction!(vecchia_param_se, m)?)?;
    m.add_function(wrap_pyfunction!(vecchia_loglik, m)?)?;
    m.add_function(wrap_pyfunction!(vecchia_krige, m)?)?;
    m.add_function(wrap_pyfunction!(krige, m)?)?;
    m.add_function(wrap_pyfunction!(krige_grid, m)?)?;
    m.add_function(wrap_pyfunction!(loo_cv, m)?)?;
    m.add_function(wrap_pyfunction!(accuracy_plot, m)?)?;
    m.add_function(wrap_pyfunction!(regression_kriging, m)?)?;
    m.add_function(wrap_pyfunction!(idw, m)?)?;
    m.add_function(wrap_pyfunction!(knn, m)?)?;
    m.add_function(wrap_pyfunction!(co_kriging, m)?)?;
    m.add_function(wrap_pyfunction!(compare_methods, m)?)?;
    m.add_function(wrap_pyfunction!(tune_idw_power, m)?)?;
    m.add_function(wrap_pyfunction!(tune_knn_k, m)?)?;
    m.add_function(wrap_pyfunction!(tune_kriging_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(sgs, m)?)?;
    m.add_function(wrap_pyfunction!(decluster_weights, m)?)?;
    m.add_function(wrap_pyfunction!(sis, m)?)?;
    m.add_function(wrap_pyfunction!(experimental_variogram_3d, m)?)?;
    m.add_function(wrap_pyfunction!(fit_variogram_3d, m)?)?;
    m.add_function(wrap_pyfunction!(krige_3d, m)?)?;
    m.add_function(wrap_pyfunction!(indicator_kriging, m)?)?;
    m.add_function(wrap_pyfunction!(lognormal_kriging, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
