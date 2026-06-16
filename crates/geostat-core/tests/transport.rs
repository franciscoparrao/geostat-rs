//! Cross-checks anchoring warped (transport) kriging to known results:
//!
//! 1. With an exact log marginal, the Monte Carlo E-type estimate converges
//!    to the analytic lognormal back-transform — itself validated against
//!    gstat (see validation/). This ties the transport machinery to a
//!    gstat-anchored value.
//! 2. With an identity marginal, transport kriging reduces to ordinary
//!    kriging in data space.

use geostat_core::{
    BoxCox, FittedMarginal, Identity, Kriging, KrigingConfig, KrigingMethod, ModelKind, PointSet,
    Rng, Structure, TransportKriging, VariogramConfig, VariogramModel, experimental_variogram,
    fit_best, lognormal_kriging,
};

fn lognormal_field(n: usize, seed: u64) -> PointSet {
    let mut rng = Rng::new(seed);
    let mut coords = Vec::new();
    let mut values = Vec::new();
    for _ in 0..n {
        let x = rng.uniform() * 100.0;
        let y = rng.uniform() * 100.0;
        let logv = (x / 30.0).sin() + (y / 25.0).cos() + 0.3 * rng.normal();
        coords.push([x, y]);
        values.push(logv.exp());
    }
    PointSet::new(coords, values).unwrap()
}

#[test]
fn transport_etype_matches_analytic_lognormal() {
    let data = lognormal_field(220, 13);

    // Latent variogram on log(z) (the model both paths share).
    let logs: Vec<f64> = data.values().iter().map(|&z| z.ln()).collect();
    let log_pts = PointSet::new(data.coords().to_vec(), logs).unwrap();
    let cfg = VariogramConfig {
        n_lags: 12,
        max_dist: 50.0,
        direction: None,
    };
    let ev = experimental_variogram(&log_pts, &cfg).unwrap();
    let log_model = fit_best(&ev, &ModelKind::ALL).unwrap().model;

    // Exact log marginal: to_latent(z) = ln(z), to_data(y) = exp(y).
    let log_marginal = FittedMarginal::new(
        BoxCox {
            lambda: 0.0,
            shift: 0.0,
        },
        0.0,
        1.0,
    )
    .unwrap();
    let tk =
        TransportKriging::new(&data, log_marginal, &log_model, KrigingConfig::default()).unwrap();

    // Analytic ordinary lognormal kriging (E-type) on the same data + model.
    let targets = [[40.0, 40.0], [60.0, 25.0], [20.0, 70.0]];
    let analytic =
        lognormal_kriging(&data, &targets, &log_model, &KrigingConfig::default()).unwrap();

    for (t, a) in targets.iter().zip(&analytic) {
        let e = tk.predict(*t, &[], 200_000, 99).unwrap();
        // Monte Carlo E-type converges to the analytic lognormal mean.
        let rel = (e.mean - a.value).abs() / a.value.abs().max(1.0);
        assert!(
            rel < 0.01,
            "transport E-type {} vs analytic lognormal {} (rel {rel:.4})",
            e.mean,
            a.value
        );
    }
}

#[test]
fn identity_marginal_reduces_to_ordinary_kriging() {
    let data = lognormal_field(180, 21);
    let cfg = VariogramConfig {
        n_lags: 12,
        max_dist: 50.0,
        direction: None,
    };
    // Fit the variogram in data space; standardize so the latent is a
    // rescaled copy of the data (identity shape).
    let mean = data.values().iter().sum::<f64>() / data.len() as f64;
    let std = (data
        .values()
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / data.len() as f64)
        .sqrt();
    let id_marginal = FittedMarginal::new(Identity, mean, std).unwrap();

    let dev = experimental_variogram(&data, &cfg).unwrap();
    let dmodel = fit_best(&dev, &ModelKind::ALL).unwrap().model;

    // The latent process is (z - mean)/std, so its variogram is the data
    // variogram scaled by 1/std^2. Rescale the model accordingly.
    let latent_model = VariogramModel::new(
        dmodel.nugget / (std * std),
        dmodel
            .structures
            .iter()
            .map(|s| Structure {
                kind: s.kind,
                sill: s.sill / (std * std),
                range: s.range,
                anis: s.anis,
            })
            .collect(),
    )
    .unwrap();

    let tk =
        TransportKriging::new(&data, id_marginal, &latent_model, KrigingConfig::default()).unwrap();
    let ok = Kriging::new(
        &data,
        &dmodel,
        KrigingConfig {
            method: KrigingMethod::Ordinary,
            ..Default::default()
        },
    )
    .unwrap();

    for t in [[40.0, 40.0], [60.0, 25.0]] {
        let e = tk.predict(t, &[], 100_000, 3).unwrap();
        let o = ok.predict(t).unwrap();
        // Identity (affine) marginal: E-type mean converges to the OK mean.
        let rel = (e.mean - o.value).abs() / o.value.abs().max(1.0);
        assert!(
            rel < 0.02,
            "transport {} vs OK {} (rel {rel:.4})",
            e.mean,
            o.value
        );
    }
}
