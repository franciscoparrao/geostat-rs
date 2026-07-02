//! WebAssembly bindings for geostat-core (single-threaded build).
//!
//! Build with wasm-pack: `wasm-pack build crates/geostat-wasm --target web`.
//! The API mirrors the CLI conventions: models travel as JSON strings,
//! coordinates and values as `Float64Array`s, and grid results come back
//! flattened (`[predictions..., variances...]` or concatenated
//! realizations) in row-major grid order.

use wasm_bindgen::prelude::*;

use geostat_core::{
    DirectionConfig, Grid2D, Kriging, KrigingConfig, PointSet, SgsConfig, VariogramConfig,
    VariogramModel, experimental_variogram as core_variogram, fit_best,
    sequential_gaussian_simulation,
};

fn err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn point_set(x: &[f64], y: &[f64], values: &[f64]) -> Result<PointSet, JsValue> {
    PointSet::from_xyz(x, y, values).map_err(err)
}

fn vario_config(data: &PointSet, n_lags: usize, max_dist: f64) -> VariogramConfig {
    let max_dist = (max_dist > 0.0).then_some(max_dist);
    VariogramConfig::for_data(data, n_lags, max_dist, None::<DirectionConfig>)
}

/// Experimental variogram as a JSON string:
/// `{"h": [...], "gamma": [...], "n_pairs": [...]}`.
/// Pass `max_dist <= 0` for the default (bbox diagonal / 3).
#[wasm_bindgen]
pub fn variogram_json(
    x: &[f64],
    y: &[f64],
    values: &[f64],
    n_lags: usize,
    max_dist: f64,
) -> Result<String, JsValue> {
    let data = point_set(x, y, values)?;
    let cfg = vario_config(&data, n_lags, max_dist);
    let ev = core_variogram(&data, &cfg).map_err(err)?;
    let h: Vec<f64> = ev.bins.iter().map(|b| b.h).collect();
    let gamma: Vec<f64> = ev.bins.iter().map(|b| b.gamma).collect();
    let n_pairs: Vec<usize> = ev.bins.iter().map(|b| b.n_pairs).collect();
    serde_json::to_string(&serde_json::json!({
        "h": h, "gamma": gamma, "n_pairs": n_pairs
    }))
    .map_err(err)
}

/// Fits the best variogram model; returns the model as JSON (same format
/// as the geostat CLI).
#[wasm_bindgen]
pub fn fit_variogram_json(
    x: &[f64],
    y: &[f64],
    values: &[f64],
    n_lags: usize,
    max_dist: f64,
) -> Result<String, JsValue> {
    let data = point_set(x, y, values)?;
    let cfg = vario_config(&data, n_lags, max_dist);
    let ev = core_variogram(&data, &cfg).map_err(err)?;
    let fit = fit_best(&ev, &geostat_core::ModelKind::ALL).map_err(err)?;
    serde_json::to_string(&fit.model).map_err(err)
}

/// Ordinary kriging onto a regular grid. Returns a flat array of length
/// `2 * nx * ny`: predictions first, then variances, row-major with y
/// increasing. `max_neighbors = 0` means a global neighborhood.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn krige_grid(
    x: &[f64],
    y: &[f64],
    values: &[f64],
    model_json: &str,
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
    nx: usize,
    ny: usize,
    max_neighbors: usize,
) -> Result<Vec<f64>, JsValue> {
    let data = point_set(x, y, values)?;
    let parsed: VariogramModel = serde_json::from_str(model_json).map_err(err)?;
    let model = VariogramModel::new(parsed.nugget, parsed.structures).map_err(err)?;
    let grid = Grid2D::from_bbox([xmin, ymin], [xmax, ymax], nx, ny).map_err(err)?;
    let config = KrigingConfig {
        max_neighbors: (max_neighbors > 0).then_some(max_neighbors),
        ..Default::default()
    };
    let kriging = Kriging::new(&data, &model, config).map_err(err)?;
    let (mut preds, vars) = kriging.predict_grid(&grid);
    preds.extend(vars);
    Ok(preds)
}

/// Conditional SGS onto a regular grid. Returns the realizations
/// concatenated (`n_realizations * nx * ny` values, row-major per
/// realization). `model_ns_json` must describe the normal-score variogram.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn sgs_grid(
    x: &[f64],
    y: &[f64],
    values: &[f64],
    model_ns_json: &str,
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
    nx: usize,
    ny: usize,
    n_realizations: usize,
    seed: u64,
    max_neighbors: usize,
) -> Result<Vec<f64>, JsValue> {
    let data = point_set(x, y, values)?;
    let parsed: VariogramModel = serde_json::from_str(model_ns_json).map_err(err)?;
    let model = VariogramModel::new(parsed.nugget, parsed.structures).map_err(err)?;
    let grid = Grid2D::from_bbox([xmin, ymin], [xmax, ymax], nx, ny).map_err(err)?;
    let cfg = SgsConfig {
        n_realizations,
        seed,
        max_neighbors: max_neighbors.max(1),
        search_radius: None,
    };
    let res = sequential_gaussian_simulation(&data, &model, &grid, &cfg).map_err(err)?;
    Ok(res.realizations.into_iter().flatten().collect())
}
