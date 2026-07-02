//! 3-D engine tests: variography, kriging and sequential simulation with
//! `PointSet<3>` — same code paths as 2-D via const generics.

use geostat_core::{
    Grid3D, Kriging, KrigingConfig, KrigingMethod, ModelKind, PointSet, Rng, SgsConfig, Structure,
    VariogramConfig, VariogramModel, experimental_variogram, fit_best, leave_one_out, sgs_at,
};

fn synthetic_3d(n: usize, seed: u64) -> PointSet<3> {
    let mut rng = Rng::new(seed);
    let mut coords = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.uniform() * 100.0;
        let y = rng.uniform() * 100.0;
        let z = rng.uniform() * 40.0;
        coords.push([x, y, z]);
        values.push((x / 15.0).sin() + (y / 20.0).cos() + (z / 8.0).sin() + 0.05 * rng.normal());
    }
    PointSet::new(coords, values).unwrap()
}

#[test]
fn variogram_3d_hand_computed() {
    // Two points separated only in z: 3-D distance must see them.
    let data = PointSet::new(
        vec![[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 4.0]],
        vec![0.0, 1.0, 2.0],
    )
    .unwrap();
    let cfg = VariogramConfig {
        n_lags: 2,
        max_dist: 4.0,
        direction: None,
    };
    let ev = experimental_variogram(&data, &cfg).unwrap();
    // d=2 pairs (x2, gamma 0.5) in bin 0; d=4 pair (gamma 2.0) in bin 1.
    assert_eq!(ev.bins[0].n_pairs, 2);
    assert!((ev.bins[0].gamma - 0.5).abs() < 1e-12);
    assert_eq!(ev.bins[1].n_pairs, 1);
    assert!((ev.bins[1].gamma - 2.0).abs() < 1e-12);
}

#[test]
fn kriging_3d_exact_and_consistent() {
    let data = synthetic_3d(150, 5);
    let model =
        VariogramModel::new(0.01, vec![Structure::new(ModelKind::Spherical, 1.0, 40.0)]).unwrap();
    let k: Kriging<'_, 3> = Kriging::new(&data, &model, KrigingConfig::default()).unwrap();
    // Exact at data points.
    for i in (0..data.len()).step_by(30) {
        let est = k.predict(data.coord(i)).unwrap();
        assert!((est.value - data.value(i)).abs() < 1e-7);
        assert!(est.variance < 1e-7);
    }
    // Far field: variance >= total sill.
    let far = k.predict([1e5, 1e5, 1e5]).unwrap();
    assert!(far.variance >= 0.99 * model.total_sill());
    // kd-tree neighborhood at k = n matches global.
    let local: Kriging<'_, 3> = Kriging::new(
        &data,
        &model,
        KrigingConfig {
            max_neighbors: Some(data.len()),
            ..Default::default()
        },
    )
    .unwrap();
    for t in [[50.0, 50.0, 20.0], [10.0, 80.0, 5.0]] {
        let a = k.predict(t).unwrap();
        let b = local.predict(t).unwrap();
        assert!((a.value - b.value).abs() < 1e-10);
        assert!((a.variance - b.variance).abs() < 1e-10);
    }
}

#[test]
fn universal_kriging_3d_reproduces_drift() {
    // z-value exactly linear in all three coordinates.
    let mut coords = Vec::new();
    let mut values = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            for l in 0..3 {
                let p = [i as f64 * 10.0, j as f64 * 10.0, l as f64 * 5.0];
                coords.push(p);
                values.push(1.0 + 0.2 * p[0] - 0.1 * p[1] + 0.5 * p[2]);
            }
        }
    }
    let data = PointSet::new(coords, values).unwrap();
    let model = VariogramModel::new(
        0.05,
        vec![Structure::new(ModelKind::Exponential, 1.0, 30.0)],
    )
    .unwrap();
    let k: Kriging<'_, 3> = Kriging::new(
        &data,
        &model,
        KrigingConfig {
            method: KrigingMethod::Universal { degree: 1 },
            ..Default::default()
        },
    )
    .unwrap();
    let t = [15.0, 25.0, 7.5];
    let expected = 1.0 + 0.2 * t[0] - 0.1 * t[1] + 0.5 * t[2];
    let est = k.predict(t).unwrap();
    assert!(
        (est.value - expected).abs() < 1e-7,
        "{} vs {expected}",
        est.value
    );
}

#[test]
fn vertical_anisotropy_shortens_z_range() {
    // ratio_z = 0.25: the sill is reached 4x faster vertically.
    let m = VariogramModel::new(
        0.0,
        vec![Structure {
            kind: ModelKind::Spherical,
            sill: 1.0,
            range: 100.0,
            anis: Some(geostat_core::Anisotropy {
                azimuth_deg: 0.0,
                ratio: 1.0,
                ratio_z: 0.25,
            }),
        }],
    )
    .unwrap();
    assert!((m.gamma_dh([0.0, 100.0, 0.0]) - 1.0).abs() < 1e-12);
    assert!((m.gamma_dh([0.0, 0.0, 25.0]) - 1.0).abs() < 1e-12);
    assert!(m.gamma_dh([0.0, 0.0, 10.0]) > m.gamma_dh([0.0, 10.0, 0.0]));
}

#[test]
fn sgs_3d_reproducible_and_bounded() {
    let data = synthetic_3d(80, 7);
    let model = VariogramModel::new(
        0.05,
        vec![Structure::new(ModelKind::Exponential, 0.95, 30.0)],
    )
    .unwrap();
    let grid = Grid3D::from_bbox([0.0, 0.0, 0.0], [100.0, 100.0, 40.0], 8, 8, 4).unwrap();
    let cfg = SgsConfig {
        n_realizations: 2,
        seed: 31,
        max_neighbors: 12,
        search_radius: None,
    };
    let a = sgs_at(&data, &model, &grid.centers(), &cfg).unwrap();
    let b = sgs_at(&data, &model, &grid.centers(), &cfg).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 2);
    assert_eq!(a[0].len(), grid.n_cells());
    let (lo, hi) = data
        .values()
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
            (l.min(v), h.max(v))
        });
    for r in &a {
        for &v in r {
            assert!(v >= lo - 1e-9 && v <= hi + 1e-9);
        }
    }
}

#[test]
fn block_kriging_3d_uses_z_separation() {
    let data = synthetic_3d(100, 13);
    let model =
        VariogramModel::new(0.05, vec![Structure::new(ModelKind::Spherical, 1.0, 30.0)]).unwrap();
    let k: Kriging<'_, 3> = Kriging::new(&data, &model, KrigingConfig::default()).unwrap();
    let center = [42.0, 37.0, 18.0];

    // A one-point block solves the same system as point kriging: identical
    // value, and the variance only drops the nugget from C̄(B,B).
    let block = k.predict_block(center, &[[0.0, 0.0, 0.0]]).unwrap();
    let point = k.predict(center).unwrap();
    assert!(
        (block.value - point.value).abs() < 1e-10,
        "{} vs {}",
        block.value,
        point.value
    );
    assert!((block.variance - (point.variance - model.nugget)).abs() < 1e-10);

    // A vertical discretization must feed z into the point-to-block
    // covariances: spreading the block along z changes the estimate.
    let degenerate = k.predict_block(center, &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).unwrap();
    let vertical = k.predict_block(center, &[[0.0, 0.0, -5.0], [0.0, 0.0, 5.0]]).unwrap();
    assert!(
        (vertical.value - degenerate.value).abs() > 1e-6,
        "z offsets ignored: {} == {}",
        vertical.value,
        degenerate.value
    );
    assert!(vertical.variance >= 0.0);
}

#[test]
fn cv_3d_beats_mean_predictor() {
    let data = synthetic_3d(120, 11);
    let cfg = VariogramConfig {
        n_lags: 12,
        max_dist: 50.0,
        direction: None,
    };
    let ev = experimental_variogram(&data, &cfg).unwrap();
    let fit = fit_best(&ev, &ModelKind::ALL).unwrap();
    let cv = leave_one_out(&data, &fit.model, &KrigingConfig::default()).unwrap();
    let mean = data.mean();
    let std = (data
        .values()
        .iter()
        .map(|v| (v - mean) * (v - mean))
        .sum::<f64>()
        / data.len() as f64)
        .sqrt();
    assert!(cv.rmse() < 0.6 * std, "rmse {} vs std {std}", cv.rmse());
}
