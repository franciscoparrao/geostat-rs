//! Vecchia approximation of the Gaussian log-likelihood.
//!
//! The Vecchia approximation \[Vecchia 1988; Katzfuss & Guinness 2021\] factors
//! the joint Gaussian density of `n` observations into an ordered product of
//! univariate conditionals, where each point conditions on only its `m` nearest
//! *predecessors* in a chosen ordering. This turns an `O(n^3)` log-likelihood
//! (which needs the full `n x n` covariance factorization) into `O(n m^3)`,
//! making maximum-likelihood covariance estimation tractable for large `n` ---
//! the scalability frontier that dense solvers (and hence the exact kriging
//! likelihood) cannot reach.
//!
//! With full conditioning (`m >= n-1`) the approximation is *exact*: it equals
//! the multivariate-normal log-likelihood regardless of ordering. Accuracy at
//! small `m` depends on the ordering; a max-min ordering ([`maxmin_order`]) is
//! the standard choice and is far better than the natural input order.
//!
//! This complements the engine's moving-neighbourhood *prediction* (which is
//! already a nearest-neighbour approximate kriging): here the contribution is a
//! scalable *likelihood* for fitting covariance parameters, which the
//! variogram-WLS path does not provide.

use std::f64::consts::PI;

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::linalg::solve;
use crate::variogram::VariogramModel;

/// Squared Euclidean distance between two points.
fn dist2<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut s = 0.0;
    for d in 0..D {
        let v = a[d] - b[d];
        s += v * v;
    }
    s
}

/// Separation vector `a - b`.
fn sep<const D: usize>(a: &[f64; D], b: &[f64; D]) -> [f64; D] {
    let mut h = [0.0; D];
    for d in 0..D {
        h[d] = a[d] - b[d];
    }
    h
}

/// Max-min (farthest-point) ordering: start near the centroid, then repeatedly
/// append the point whose minimum distance to the already-ordered set is
/// largest. This spreads early points across the domain, which makes the
/// Vecchia approximation accurate at small conditioning sizes.
pub fn maxmin_order<const D: usize>(coords: &[[f64; D]]) -> Vec<usize> {
    let n = coords.len();
    if n == 0 {
        return Vec::new();
    }
    // Centroid, and the first point as the one closest to it.
    let mut c = [0.0; D];
    for p in coords {
        for d in 0..D {
            c[d] += p[d] / n as f64;
        }
    }
    let first = (0..n)
        .min_by(|&i, &j| dist2(&coords[i], &c).total_cmp(&dist2(&coords[j], &c)))
        .unwrap();

    let mut order = Vec::with_capacity(n);
    let mut chosen = vec![false; n];
    let mut min_d2 = vec![f64::INFINITY; n];
    order.push(first);
    chosen[first] = true;
    for i in 0..n {
        min_d2[i] = dist2(&coords[i], &coords[first]);
    }
    for _ in 1..n {
        let next = (0..n)
            .filter(|&i| !chosen[i])
            .max_by(|&i, &j| min_d2[i].total_cmp(&min_d2[j]))
            .unwrap();
        order.push(next);
        chosen[next] = true;
        for i in 0..n {
            if !chosen[i] {
                min_d2[i] = min_d2[i].min(dist2(&coords[i], &coords[next]));
            }
        }
    }
    order
}

/// The `m` nearest predecessors of `coords[target]` among `prev` (indices of
/// already-ordered points), by Euclidean distance.
fn nearest_predecessors<const D: usize>(
    coords: &[[f64; D]],
    target: usize,
    prev: &[usize],
    m: usize,
) -> Vec<usize> {
    if prev.len() <= m {
        return prev.to_vec();
    }
    let mut scored: Vec<(f64, usize)> = prev
        .iter()
        .map(|&j| (dist2(&coords[target], &coords[j]), j))
        .collect();
    scored.select_nth_unstable_by(m - 1, |a, b| a.0.total_cmp(&b.0));
    scored.truncate(m);
    scored.into_iter().map(|(_, j)| j).collect()
}

/// Vecchia-approximated Gaussian log-likelihood of `data` under `model`, with
/// each point conditioning on its `m` nearest predecessors in `order` (defaults
/// to [`maxmin_order`] when `None`). Values are centred by their mean (a known
/// constant-mean Gaussian field).
///
/// With `m >= n-1` this returns the exact multivariate-normal log-likelihood.
pub fn vecchia_loglik<const D: usize>(
    data: &PointSet<D>,
    model: &VariogramModel,
    m: usize,
    order: Option<&[usize]>,
) -> Result<f64> {
    let n = data.len();
    if n < 2 {
        return Err(GeostatError::InsufficientData(
            "Vecchia log-likelihood requires at least 2 points".into(),
        ));
    }
    if m == 0 {
        return Err(GeostatError::InvalidParameter(
            "conditioning size m must be at least 1".into(),
        ));
    }
    let coords = data.coords();
    let mean = data.mean();
    let z: Vec<f64> = data.values().iter().map(|v| v - mean).collect();
    let sill = model.total_sill(); // C(0)

    let owned;
    let order = match order {
        Some(o) => {
            if o.len() != n {
                return Err(GeostatError::DimensionMismatch(format!(
                    "order has {} entries for {n} points",
                    o.len()
                )));
            }
            o
        }
        None => {
            owned = maxmin_order(coords);
            &owned
        }
    };

    let mut loglik = 0.0;
    for (k, &i) in order.iter().enumerate() {
        let nb = nearest_predecessors(coords, i, &order[..k], m);
        let (mu, var) = if nb.is_empty() {
            (0.0, sill) // first point: marginal
        } else {
            let s = nb.len();
            // K_SS (neighbour covariance) and k_i (target-neighbour covariance).
            let mut kss = vec![0.0; s * s];
            let mut ki = vec![0.0; s];
            for a in 0..s {
                ki[a] = model.covariance_dh(sep(&coords[i], &coords[nb[a]]));
                for b in 0..s {
                    kss[a * s + b] =
                        model.covariance_dh(sep(&coords[nb[a]], &coords[nb[b]]));
                }
            }
            let arr = ndarray::Array2::from_shape_vec((s, s), kss)
                .expect("square K_SS");
            let w = solve(arr, ki.clone())?; // w = K_SS^{-1} k_i
            let mu: f64 = w.iter().zip(&nb).map(|(&wj, &j)| wj * z[j]).sum();
            let reduction: f64 = w.iter().zip(&ki).map(|(&wj, &kj)| wj * kj).sum();
            (mu, (sill - reduction).max(1e-12))
        };
        let r = z[i] - mu;
        loglik += -0.5 * ((2.0 * PI).ln() + var.ln() + r * r / var);
    }
    Ok(loglik)
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
            values.push((x / 25.0).sin() + (y / 30.0).cos() + 0.1 * rng.normal());
        }
        PointSet::new(coords, values).unwrap()
    }

    /// Exact multivariate-normal log-likelihood (zero-mean, centred), the
    /// oracle for the full-conditioning Vecchia case.
    fn exact_loglik(data: &PointSet, model: &VariogramModel) -> f64 {
        let n = data.len();
        let coords = data.coords();
        let mean = data.mean();
        let z: Vec<f64> = data.values().iter().map(|v| v - mean).collect();
        let mut cov = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                cov[a * n + b] = model.covariance_dh(sep(&coords[a], &coords[b]));
            }
        }
        let arr = ndarray::Array2::from_shape_vec((n, n), cov).unwrap();
        let lu = crate::linalg::lu_factor(arr).unwrap();
        let x = lu.solve(z.clone());
        let quad: f64 = z.iter().zip(&x).map(|(&zi, &xi)| zi * xi).sum();
        -0.5 * (n as f64 * (2.0 * PI).ln() + lu.ln_det_abs() + quad)
    }

    #[test]
    fn full_conditioning_equals_exact() {
        let data = field(14, 3);
        let model =
            VariogramModel::new(0.05, vec![Structure::new(ModelKind::Exponential, 1.0, 40.0)])
                .unwrap();
        let exact = exact_loglik(&data, &model);
        // m >= n-1 -> every point conditions on all predecessors -> exact.
        let v = vecchia_loglik(&data, &model, data.len() - 1, None).unwrap();
        assert!((v - exact).abs() < 1e-9, "vecchia {v} vs exact {exact}");
    }

    #[test]
    fn approximation_is_close_with_maxmin() {
        let data = field(90, 11);
        let model =
            VariogramModel::new(0.05, vec![Structure::new(ModelKind::Exponential, 1.0, 30.0)])
                .unwrap();
        let exact = exact_loglik(&data, &model);
        let approx = vecchia_loglik(&data, &model, 12, None).unwrap();
        // A modest conditioning size already tracks the exact log-likelihood.
        let rel = (approx - exact).abs() / exact.abs();
        assert!(rel < 0.02, "rel err {rel}: approx {approx} vs exact {exact}");
    }

    #[test]
    fn maxmin_order_is_a_permutation() {
        let data = field(50, 7);
        let ord = maxmin_order(data.coords());
        let mut seen = ord.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 50, "ordering must be a permutation");
    }

    #[test]
    fn rejects_bad_args() {
        let data = field(10, 1);
        let model =
            VariogramModel::new(0.1, vec![Structure::new(ModelKind::Spherical, 1.0, 20.0)]).unwrap();
        assert!(vecchia_loglik(&data, &model, 0, None).is_err());
    }
}
