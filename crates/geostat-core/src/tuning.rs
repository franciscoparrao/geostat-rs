//! Hyperparameter tuning by predictive accuracy.
//!
//! Li (2021) optimizes a method's parameters by *predictive accuracy* —
//! leave-one-out cross-validation — rather than by a model fit (e.g. variogram
//! weighted least squares). Each candidate value is scored by VEcv (variance
//! explained by cross-validation), and the value maximizing it is chosen. This
//! tunes things WLS cannot: the IDW power, the k-NN `k`, and the kriging search
//! neighborhood size, all on the same predictive footing used to compare
//! methods ([`crate::interpolation`], [`crate::validation`]).

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::interpolation::{idw_cross_validate, knn_cross_validate};
use crate::kriging::{KrigingConfig, KrigingMethod};
use crate::validation::leave_one_out;
use crate::variogram::VariogramModel;

/// Result of a parameter search: the best value, its VEcv, and the full
/// `(candidate, VEcv)` trace (in the order the candidates were supplied).
#[derive(Debug, Clone)]
pub struct TuneResult<P> {
    /// The candidate that maximized VEcv.
    pub best: P,
    /// VEcv (%) at the best candidate.
    pub best_vecv: f64,
    /// `(candidate, VEcv)` for every candidate; non-finite VEcv means that
    /// candidate's cross-validation failed and it was skipped in the pick.
    pub trace: Vec<(P, f64)>,
}

fn pick_best<P: Copy>(trace: Vec<(P, f64)>) -> Result<TuneResult<P>> {
    let best = trace
        .iter()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .copied();
    match best {
        Some((best, best_vecv)) => Ok(TuneResult {
            best,
            best_vecv,
            trace,
        }),
        None => Err(GeostatError::InvalidParameter(
            "no candidate produced a finite VEcv".into(),
        )),
    }
}

/// Tunes the IDW `power` over `powers` by leave-one-out VEcv.
pub fn tune_idw_power<const D: usize>(
    data: &PointSet<D>,
    powers: &[f64],
    max_neighbors: Option<usize>,
    radius: Option<f64>,
) -> Result<TuneResult<f64>> {
    if powers.is_empty() {
        return Err(GeostatError::InvalidParameter(
            "no IDW powers to try".into(),
        ));
    }
    let trace = powers
        .iter()
        .map(|&p| {
            let v = idw_cross_validate(data, p, max_neighbors, radius)
                .map(|cv| cv.vecv())
                .unwrap_or(f64::NAN);
            (p, v)
        })
        .collect();
    pick_best(trace)
}

/// Tunes the k-NN `k` over `ks` by leave-one-out VEcv.
pub fn tune_knn_k<const D: usize>(
    data: &PointSet<D>,
    ks: &[usize],
    radius: Option<f64>,
) -> Result<TuneResult<usize>> {
    if ks.is_empty() {
        return Err(GeostatError::InvalidParameter("no k values to try".into()));
    }
    let trace = ks
        .iter()
        .map(|&k| {
            let v = knn_cross_validate(data, k, radius)
                .map(|cv| cv.vecv())
                .unwrap_or(f64::NAN);
            (k, v)
        })
        .collect();
    pick_best(trace)
}

/// Tunes the kriging search-neighborhood size (`max_neighbors`) over
/// `candidates` by leave-one-out VEcv, holding the variogram model and method
/// fixed.
pub fn tune_kriging_neighbors<const D: usize>(
    data: &PointSet<D>,
    model: &VariogramModel,
    method: KrigingMethod,
    candidates: &[usize],
    radius: Option<f64>,
) -> Result<TuneResult<usize>> {
    if candidates.is_empty() {
        return Err(GeostatError::InvalidParameter(
            "no neighborhood sizes to try".into(),
        ));
    }
    let trace = candidates
        .iter()
        .map(|&n| {
            let cfg = KrigingConfig {
                method,
                max_neighbors: Some(n),
                search_radius: radius,
                ..Default::default()
            };
            let v = leave_one_out(data, model, &cfg)
                .map(|cv| cv.vecv())
                .unwrap_or(f64::NAN);
            (n, v)
        })
        .collect();
    pick_best(trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::variogram::{ModelKind, Structure};

    fn field(n: usize, seed: u64) -> PointSet {
        let mut rng = Rng::new(seed);
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for _ in 0..n {
            let x = rng.uniform() * 100.0;
            let y = rng.uniform() * 100.0;
            coords.push([x, y]);
            values.push((x / 20.0).sin() + (y / 25.0).cos() + 0.1 * rng.normal());
        }
        PointSet::new(coords, values).unwrap()
    }

    #[test]
    fn tunes_idw_power_to_a_finite_optimum() {
        let data = field(120, 4);
        let powers = [0.5, 1.0, 2.0, 3.0, 5.0];
        let res = tune_idw_power(&data, &powers, Some(16), None).unwrap();
        assert!(powers.contains(&res.best));
        assert!(res.best_vecv.is_finite());
        // The chosen power has the highest VEcv in the trace.
        let max = res
            .trace
            .iter()
            .map(|&(_, v)| v)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((res.best_vecv - max).abs() < 1e-12);
        assert_eq!(res.trace.len(), powers.len());
    }

    #[test]
    fn tunes_knn_k() {
        let data = field(120, 8);
        let res = tune_knn_k(&data, &[1, 2, 4, 8, 16], None).unwrap();
        assert!([1, 2, 4, 8, 16].contains(&res.best));
        assert!(res.best_vecv.is_finite());
    }

    #[test]
    fn tunes_kriging_neighbors() {
        let data = field(120, 9);
        let model =
            VariogramModel::new(0.02, vec![Structure::new(ModelKind::Spherical, 1.0, 40.0)])
                .unwrap();
        let res = tune_kriging_neighbors(
            &data,
            &model,
            KrigingMethod::Ordinary,
            &[4, 8, 16, 32],
            None,
        )
        .unwrap();
        assert!([4, 8, 16, 32].contains(&res.best));
        assert!(res.best_vecv.is_finite());
    }

    #[test]
    fn empty_candidates_error() {
        let data = field(10, 1);
        assert!(tune_idw_power(&data, &[], None, None).is_err());
        assert!(tune_knn_k(&data, &[], None).is_err());
    }
}
