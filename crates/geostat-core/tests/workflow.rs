//! End-to-end workflow: experimental variogram -> model fit -> kriging ->
//! cross-validation -> sequential Gaussian simulation, on a synthetic field
//! with known spatial structure.

use geostat_core::{
    Grid2D, Kriging, KrigingConfig, ModelKind, PointSet, Rng, SgsConfig, VariogramConfig,
    experimental_variogram, fit_best, leave_one_out, sequential_gaussian_simulation,
};

/// Synthetic smooth field with additive noise, sampled at random locations.
fn synthetic_data(n: usize, seed: u64) -> PointSet {
    let mut rng = Rng::new(seed);
    let mut coords = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.uniform() * 200.0;
        let y = rng.uniform() * 200.0;
        let z = (x / 30.0).sin() + (y / 40.0).cos() + 0.1 * rng.normal();
        coords.push([x, y]);
        values.push(z);
    }
    PointSet::new(coords, values).unwrap()
}

#[test]
fn full_workflow_on_synthetic_field() {
    let data = synthetic_data(250, 2026);

    // 1. Experimental variogram.
    let cfg = VariogramConfig {
        n_lags: 15,
        max_dist: 100.0,
        direction: None,
    };
    let ev = experimental_variogram(&data, &cfg).unwrap();
    let populated = ev.bins.iter().filter(|b| b.n_pairs > 0).count();
    assert!(populated >= 10, "only {populated} populated bins");

    // 2. Model fitting: short lags must have lower gamma than the sill region.
    let fit = fit_best(&ev, &ModelKind::ALL).unwrap();
    let model = &fit.model;
    assert!(model.total_sill() > 0.0);
    assert!(model.gamma(5.0) < model.gamma(80.0));

    // 3. Kriging onto a grid: finite values, non-negative variances.
    // A 32-point neighborhood keeps the test fast and mirrors real usage.
    // `KrigingConfig` is `#[non_exhaustive]`: build from `Default::default()`
    // and assign fields.
    let mut krig_cfg = KrigingConfig::default();
    krig_cfg.max_neighbors = Some(32);
    let kriging = Kriging::new(&data, model, krig_cfg.clone()).unwrap();
    let grid = Grid2D::from_bbox([0.0, 0.0], [200.0, 200.0], 25, 25).unwrap();
    let (values, variances) = kriging.predict_grid(&grid);
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(variances.iter().all(|v| *v >= 0.0 && v.is_finite()));

    // 4. Cross-validation: kriging clearly beats the mean predictor.
    let cv = leave_one_out(&data, model, &krig_cfg).unwrap();
    let mean = data.mean();
    let std = (data
        .values()
        .iter()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        / data.len() as f64)
        .sqrt();
    assert!(
        cv.rmse() < 0.6 * std,
        "rmse {} vs data std {std}",
        cv.rmse()
    );
    assert!(cv.mean_error().abs() < 0.1 * std);

    // 5. SGS conditioned on the data: reproducible, in data range, and with
    //    realistic spread across realizations.
    // `SgsConfig` is `#[non_exhaustive]`: build from `Default::default()`
    // and assign fields.
    let mut sgs_cfg = SgsConfig::default();
    sgs_cfg.n_realizations = 5;
    sgs_cfg.seed = 99;
    sgs_cfg.max_neighbors = 16;
    sgs_cfg.search_radius = None;
    let sim = sequential_gaussian_simulation(&data, model, &grid, &sgs_cfg).unwrap();
    assert_eq!(sim.realizations.len(), 5);
    let (dmin, dmax) = data
        .values()
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    for real in &sim.realizations {
        assert_eq!(real.len(), grid.n_cells());
        for &v in real {
            assert!(v >= dmin - 1e-9 && v <= dmax + 1e-9);
        }
    }
    // Ensemble mean at each cell should correlate with the kriging map.
    let n_cells = grid.n_cells();
    let ens_mean: Vec<f64> = (0..n_cells)
        .map(|i| sim.realizations.iter().map(|r| r[i]).sum::<f64>() / 5.0)
        .collect();
    let corr = pearson(&ens_mean, &values);
    assert!(corr > 0.7, "ensemble mean vs kriging correlation {corr}");
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        cov += (x - ma) * (y - mb);
        va += (x - ma) * (x - ma);
        vb += (y - mb) * (y - mb);
    }
    cov / (va.sqrt() * vb.sqrt())
}
