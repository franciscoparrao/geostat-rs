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

use ndarray::Array2;

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::linalg::{cholesky_solve_in_place, lu_factor};
use crate::optim::nelder_mead;
use crate::search::BucketGrid;
use crate::variogram::{ModelKind, Structure, VariogramModel};

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

/// Heap key for the lazy farthest-point traversal: max by squared distance,
/// ties broken toward the larger index (matching the previous `max_by`
/// implementation, which returned the last maximal element).
#[derive(PartialEq)]
struct MaxminKey {
    d2: f64,
    idx: usize,
}

impl Eq for MaxminKey {}

impl Ord for MaxminKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.d2.total_cmp(&other.d2).then(self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for MaxminKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Max-min (farthest-point) ordering: start near the centroid, then repeatedly
/// append the point whose minimum distance to the already-ordered set is
/// largest. This spreads early points across the domain, which makes the
/// Vecchia approximation accurate at small conditioning sizes.
///
/// Exact farthest-point traversal in `O(n log n)`-like time: each point's
/// key (squared distance to the selected set) can only *decrease* as points
/// are selected, so a max-heap with lazy re-evaluation is exact — pop the
/// stale top, refresh its key against the current selected set (bucket-grid
/// nearest-neighbor query), and select it if it still dominates.
pub fn maxmin_order<const D: usize>(coords: &[[f64; D]]) -> Vec<usize> {
    use std::collections::BinaryHeap;
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
        .expect("n > 0");

    let mut min = coords[0];
    let mut max = coords[0];
    for p in coords {
        for d in 0..D {
            min[d] = min[d].min(p[d]);
            max[d] = max[d].max(p[d]);
        }
    }
    let mut selected_grid = BucketGrid::new(min, max, n);
    let mut selected_coords: Vec<[f64; D]> = Vec::with_capacity(n);
    let mut order = Vec::with_capacity(n);
    order.push(first);
    selected_grid.insert(coords[first]);
    selected_coords.push(coords[first]);

    let mut heap: BinaryHeap<MaxminKey> = (0..n)
        .filter(|&i| i != first)
        .map(|i| MaxminKey {
            d2: dist2(&coords[i], &coords[first]),
            idx: i,
        })
        .collect();

    // While the selected set is small, a linear scan is cheaper (and avoids
    // long shell walks in the bucket grid when the nearest point is far).
    const BRUTE_LIMIT: usize = 48;
    while let Some(top) = heap.pop() {
        let i = top.idx;
        // Refresh the stale key against the current selected set.
        let current = if selected_coords.len() <= BRUTE_LIMIT {
            selected_coords
                .iter()
                .map(|s| dist2(&coords[i], s))
                .fold(f64::INFINITY, f64::min)
        } else {
            let nb = selected_grid.k_nearest(coords[i], 1, None);
            dist2(&coords[i], &selected_coords[nb[0]])
        };
        let refreshed = MaxminKey {
            d2: current,
            idx: i,
        };
        // Keys only decrease, so the other entries' stale keys are upper
        // bounds: if the refreshed key still dominates the heap top, `i` is
        // the true farthest point.
        if heap.peek().is_none_or(|next| refreshed >= *next) {
            order.push(i);
            selected_grid.insert(coords[i]);
            selected_coords.push(coords[i]);
        } else {
            heap.push(refreshed);
        }
    }
    order
}

/// A reusable Vecchia conditioning structure: the point ordering and, for each
/// ordered point, the indices of its nearest predecessors. It depends only on
/// the geometry (coordinates, `m`, ordering), not on the covariance model, so it
/// is built once and reused across every likelihood evaluation --- which is what
/// makes maximum-likelihood fitting ([`vecchia_mle`]) efficient.
#[derive(Debug, Clone)]
pub struct VecchiaPlan {
    /// The point ordering (a permutation of `0..n`).
    pub order: Vec<usize>,
    /// `neighbours[k]` = conditioning indices for `order[k]`.
    neighbours: Vec<Vec<usize>>,
}

/// Builds a [`VecchiaPlan`] for `coords` with conditioning size `m`, using the
/// supplied ordering or [`maxmin_order`] when `None`.
pub fn vecchia_plan<const D: usize>(
    coords: &[[f64; D]],
    m: usize,
    order: Option<&[usize]>,
) -> Result<VecchiaPlan> {
    let n = coords.len();
    if n < 2 {
        return Err(GeostatError::InsufficientData(
            "Vecchia requires at least 2 points".into(),
        ));
    }
    if m == 0 {
        return Err(GeostatError::InvalidParameter(
            "conditioning size m must be at least 1".into(),
        ));
    }
    if let Some((i, j)) = crate::data::duplicate_coord_pair(coords) {
        return Err(GeostatError::DuplicatePoints(i, j));
    }
    let order = match order {
        Some(o) => {
            if o.len() != n {
                return Err(GeostatError::DimensionMismatch(format!(
                    "order has {} entries for {n} points",
                    o.len()
                )));
            }
            o.to_vec()
        }
        None => maxmin_order(coords),
    };
    // Nearest predecessors by incremental insertion: after inserting the
    // first k ordered points, a k-nearest query sees exactly the
    // predecessors of point k (bucket-grid indices are insertion order).
    let mut min = coords[0];
    let mut max = coords[0];
    for p in coords {
        for d in 0..D {
            min[d] = min[d].min(p[d]);
            max[d] = max[d].max(p[d]);
        }
    }
    let mut grid = BucketGrid::new(min, max, n);
    let mut neighbours = Vec::with_capacity(n);
    for &i in &order {
        let nb = grid.k_nearest(coords[i], m, None);
        neighbours.push(nb.into_iter().map(|pos| order[pos]).collect());
        grid.insert(coords[i]);
    }
    Ok(VecchiaPlan { order, neighbours })
}

/// Vecchia log-likelihood for a precomputed [`VecchiaPlan`] (the hot path).
fn loglik_with_plan<const D: usize>(
    data: &PointSet<D>,
    model: &VariogramModel,
    plan: &VecchiaPlan,
) -> Result<f64> {
    let coords = data.coords();
    let mean = data.mean();
    let z: Vec<f64> = data.values().iter().map(|v| v - mean).collect();
    let sill = model.total_sill(); // C(0)

    // Workspaces reused across every point (the optimizer calls this
    // thousands of times; per-point allocation dominated the profile).
    let mut kss: Vec<f64> = Vec::new();
    let mut ki: Vec<f64> = Vec::new();
    let mut w: Vec<f64> = Vec::new();

    let mut loglik = 0.0;
    for (k, &i) in plan.order.iter().enumerate() {
        let nb = &plan.neighbours[k];
        let (mu, var) = if nb.is_empty() {
            (0.0, sill) // first point: marginal
        } else {
            let s = nb.len();
            // K_SS (neighbour covariance) and k_i (target-neighbour covariance).
            kss.clear();
            kss.resize(s * s, 0.0);
            ki.clear();
            ki.resize(s, 0.0);
            for a in 0..s {
                ki[a] = model.covariance_dh(sep(&coords[i], &coords[nb[a]]));
                for b in a..s {
                    let c = model.covariance_dh(sep(&coords[nb[a]], &coords[nb[b]]));
                    kss[a * s + b] = c;
                    kss[b * s + a] = c;
                }
            }
            // w = K_SS^{-1} k_i via Cholesky (K_SS is SPD).
            w.clear();
            w.extend_from_slice(&ki);
            cholesky_solve_in_place(&mut kss, s, &mut w)?;
            let mu: f64 = w.iter().zip(nb).map(|(&wj, &j)| wj * z[j]).sum();
            let reduction: f64 = w.iter().zip(&ki).map(|(&wj, &kj)| wj * kj).sum();
            (mu, (sill - reduction).max(1e-12))
        };
        let r = z[i] - mu;
        loglik += -0.5 * ((2.0 * PI).ln() + var.ln() + r * r / var);
    }
    Ok(loglik)
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
    let plan = vecchia_plan(data.coords(), m, order)?;
    loglik_with_plan(data, model, &plan)
}

/// Result of a Vecchia maximum-likelihood fit.
#[derive(Debug, Clone)]
pub struct VecchiaFit {
    /// Fitted single-structure model (nugget + one structure).
    pub model: VariogramModel,
    /// Maximized Vecchia log-likelihood.
    pub loglik: f64,
}

/// Fits a single-structure model of the given `kind` (nugget + partial sill +
/// range) to `data` by **maximum likelihood**, maximizing the Vecchia
/// approximation with conditioning size `m`. The plan is built once and reused
/// across optimizer evaluations.
///
/// This is the scalable counterpart to variogram weighted-least-squares fitting:
/// it fits the covariance directly to the data likelihood, and with the Vecchia
/// approximation it stays `O(n m^3)` per evaluation rather than `O(n^3)`.
pub fn vecchia_mle<const D: usize>(
    data: &PointSet<D>,
    kind: ModelKind,
    m: usize,
    order: Option<&[usize]>,
) -> Result<VecchiaFit> {
    let plan = vecchia_plan(data.coords(), m, order)?;

    // Initial scales: sample variance for the sill, a fraction of the domain
    // extent for the range.
    let mean = data.mean();
    let var0 = (data
        .values()
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / data.len() as f64)
        .max(1e-12);
    let coords = data.coords();
    let mut lo = [f64::INFINITY; D];
    let mut hi = [f64::NEG_INFINITY; D];
    for p in coords {
        for d in 0..D {
            lo[d] = lo[d].min(p[d]);
            hi[d] = hi[d].max(p[d]);
        }
    }
    let extent = (0..D).map(|d| (hi[d] - lo[d]).powi(2)).sum::<f64>().sqrt();
    let range0 = (extent / 3.0).max(1e-9);

    // Parameters are multipliers of (var0, var0, range0); penalize invalid ones.
    let objective = |x: &[f64]| -> f64 {
        let nugget = x[0] * var0;
        let psill = x[1] * var0;
        let range = x[2] * range0;
        let mut pen = 0.0;
        if nugget < 0.0 {
            pen += nugget * nugget;
        }
        if psill <= 0.0 {
            pen += 1.0 + psill * psill;
        }
        if range <= 0.0 {
            pen += 1.0 + range * range;
        }
        if pen > 0.0 {
            return 1e12 * (1.0 + pen);
        }
        // A range far beyond the domain is unidentifiable (indistinguishable
        // from a trend); regularize it back rather than letting it run away.
        let range_max = 10.0 * extent;
        let reg = if range > range_max {
            ((range - range_max) / range_max).powi(2)
        } else {
            0.0
        };
        let model = VariogramModel {
            nugget,
            structures: vec![Structure::new(kind, psill, range)],
        };
        // Minimize the negative log-likelihood; a singular system is rejected.
        match loglik_with_plan(data, &model, &plan) {
            Ok(ll) if ll.is_finite() => -ll + 1e6 * reg,
            _ => 1e12,
        }
    };

    let x0 = [0.1, 0.9, 1.0];
    let (xb, neg_ll) = nelder_mead(objective, &x0, 0.3, 2000);
    let model = VariogramModel::new(
        (xb[0] * var0).max(0.0),
        vec![Structure::new(
            kind,
            (xb[1] * var0).max(1e-12),
            (xb[2] * range0).max(1e-12),
        )],
    )?;
    Ok(VecchiaFit {
        model,
        loglik: -neg_ll,
    })
}

/// All monomial exponent tuples over `D` dimensions with total degree
/// `<= degree` (includes the constant term).
fn monomials<const D: usize>(degree: u8) -> Vec<[u8; D]> {
    fn rec<const D: usize>(d: usize, rem: u8, cur: &mut [u8; D], out: &mut Vec<[u8; D]>) {
        if d == D {
            out.push(*cur);
            return;
        }
        for e in 0..=rem {
            cur[d] = e;
            rec(d + 1, rem - e, cur, out);
        }
        cur[d] = 0;
    }
    let mut out = Vec::new();
    let mut cur = [0u8; D];
    rec(0, degree, &mut cur, &mut out);
    out
}

/// Polynomial trend basis `F` (one row per point, one column per monomial up to
/// `degree`). Coordinates are centred and scaled per dimension so the design
/// matrix stays well-conditioned at higher degrees.
fn poly_basis<const D: usize>(coords: &[[f64; D]], degree: u8) -> Vec<Vec<f64>> {
    let n = coords.len();
    let mut mean = [0.0; D];
    for p in coords {
        for d in 0..D {
            mean[d] += p[d] / n as f64;
        }
    }
    let mut scale = [1.0; D];
    for d in 0..D {
        let var = coords.iter().map(|p| (p[d] - mean[d]).powi(2)).sum::<f64>() / n as f64;
        scale[d] = var.sqrt().max(1e-12);
    }
    let exps = monomials::<D>(degree);
    coords
        .iter()
        .map(|p| {
            let cs: [f64; D] = std::array::from_fn(|d| (p[d] - mean[d]) / scale[d]);
            exps.iter()
                .map(|e| {
                    let mut v = 1.0;
                    for d in 0..D {
                        for _ in 0..e[d] {
                            v *= cs[d];
                        }
                    }
                    v
                })
                .collect()
        })
        .collect()
}

/// Restricted (and trend-) maximum-likelihood Vecchia log-likelihood: the mean
/// is `F beta` (estimated by generalized least squares), and the covariance is
/// fit to the resulting error contrasts. Whitening `z` and the columns of `F`
/// with the Vecchia factor turns the GLS solve into a small `p x p` system.
fn reml_loglik_with_plan<const D: usize>(
    data: &PointSet<D>,
    model: &VariogramModel,
    plan: &VecchiaPlan,
    basis: &[Vec<f64>],
) -> Result<f64> {
    let coords = data.coords();
    let z = data.values();
    let p = basis[0].len();
    let n = coords.len();
    let sill = model.total_sill();

    let mut logdet = 0.0;
    let mut uz_uz = 0.0;
    let mut a = vec![0.0; p * p]; // F^T Sigma^{-1} F
    let mut c = vec![0.0; p]; // F^T Sigma^{-1} z
    // Workspaces reused across every point (hot path of the optimizer).
    let mut kss: Vec<f64> = Vec::new();
    let mut ki: Vec<f64> = Vec::new();
    let mut w: Vec<f64> = Vec::new();
    for (k, &i) in plan.order.iter().enumerate() {
        let nb = &plan.neighbours[k];
        // Whitened z and trend-row at this ordered point.
        let (uz, uf, d) = if nb.is_empty() {
            let sd = sill.sqrt();
            let uf: Vec<f64> = basis[i].iter().map(|&f| f / sd).collect();
            (z[i] / sd, uf, sill)
        } else {
            let s = nb.len();
            kss.clear();
            kss.resize(s * s, 0.0);
            ki.clear();
            ki.resize(s, 0.0);
            for aa in 0..s {
                ki[aa] = model.covariance_dh(sep(&coords[i], &coords[nb[aa]]));
                for bb in aa..s {
                    let cv = model.covariance_dh(sep(&coords[nb[aa]], &coords[nb[bb]]));
                    kss[aa * s + bb] = cv;
                    kss[bb * s + aa] = cv;
                }
            }
            // w = K_SS^{-1} k_i via Cholesky (K_SS is SPD).
            w.clear();
            w.extend_from_slice(&ki);
            cholesky_solve_in_place(&mut kss, s, &mut w)?;
            let d = (sill - w.iter().zip(&ki).map(|(&wj, &kj)| wj * kj).sum::<f64>()).max(1e-12);
            let sd = d.sqrt();
            let uz = (z[i] - w.iter().zip(nb).map(|(&wj, &j)| wj * z[j]).sum::<f64>()) / sd;
            let uf: Vec<f64> = (0..p)
                .map(|col| {
                    (basis[i][col]
                        - w.iter()
                            .zip(nb)
                            .map(|(&wj, &j)| wj * basis[j][col])
                            .sum::<f64>())
                        / sd
                })
                .collect();
            (uz, uf, d)
        };
        logdet += d.ln();
        uz_uz += uz * uz;
        for aa in 0..p {
            c[aa] += uf[aa] * uz;
            for bb in 0..p {
                a[aa * p + bb] += uf[aa] * uf[bb];
            }
        }
    }

    // GLS: beta = A^{-1} c; residual quadratic form = uz_uz - c^T beta.
    let a_arr = Array2::from_shape_vec((p, p), a).expect("square A");
    let lu = lu_factor(a_arr)?;
    let beta = lu.solve(c.clone());
    let ln_det_a = lu.ln_det_abs();
    let resid = uz_uz - c.iter().zip(&beta).map(|(&ci, &bi)| ci * bi).sum::<f64>();

    let nm = (n - p) as f64;
    Ok(-0.5 * (nm * (2.0 * PI).ln() + logdet + ln_det_a + resid))
}

/// Fits a single-structure model by **restricted/trend maximum likelihood**:
/// the mean is a polynomial trend of `drift_degree` (0 = constant, 1 = linear,
/// 2 = quadratic) estimated by GLS, and the covariance is fit to the error
/// contrasts via the Vecchia REML likelihood. Unlike [`vecchia_mle`] (constant
/// plug-in mean), this does not let a spatial trend inflate the range --- it is
/// the estimator to use when the field is non-stationary in the mean.
pub fn vecchia_reml<const D: usize>(
    data: &PointSet<D>,
    kind: ModelKind,
    m: usize,
    drift_degree: u8,
    order: Option<&[usize]>,
) -> Result<VecchiaFit> {
    let plan = vecchia_plan(data.coords(), m, order)?;
    let basis = poly_basis(data.coords(), drift_degree);
    reml_fit_with_basis(data, kind, &plan, &basis)
}

/// Restricted/trend ML with **external covariates**: the mean is
/// `beta_0 + sum_j beta_j x_j` over the supplied covariate columns
/// (`drift[i]` holds the covariates at point `i`), fit by GLS while the
/// covariance is fit to the error contrasts. This is kriging-with-external-drift
/// estimated by maximum likelihood --- the remedy when a non-stationary mean is
/// driven by a measured covariate (e.g.\ distance to a feature) rather than a
/// smooth polynomial.
pub fn vecchia_reml_drift<const D: usize>(
    data: &PointSet<D>,
    kind: ModelKind,
    m: usize,
    drift: &[Vec<f64>],
    order: Option<&[usize]>,
) -> Result<VecchiaFit> {
    let n = data.len();
    if drift.len() != n {
        return Err(GeostatError::DimensionMismatch(format!(
            "{} drift rows vs {n} data points",
            drift.len()
        )));
    }
    let ncov = drift.first().map(Vec::len).unwrap_or(0);
    if ncov == 0 {
        return Err(GeostatError::InvalidParameter(
            "external-drift REML needs at least one covariate column".into(),
        ));
    }
    if drift.iter().any(|r| r.len() != ncov) {
        return Err(GeostatError::DimensionMismatch(
            "all drift rows must have the same number of covariates".into(),
        ));
    }
    // Basis = intercept + covariate columns.
    let basis: Vec<Vec<f64>> = drift
        .iter()
        .map(|row| {
            let mut b = Vec::with_capacity(ncov + 1);
            b.push(1.0);
            b.extend_from_slice(row);
            b
        })
        .collect();
    let plan = vecchia_plan(data.coords(), m, order)?;
    reml_fit_with_basis(data, kind, &plan, &basis)
}

/// Shared REML optimizer: fits (nugget, sill, range) for `kind` by maximizing
/// the Vecchia REML likelihood under an arbitrary trend basis.
fn reml_fit_with_basis<const D: usize>(
    data: &PointSet<D>,
    kind: ModelKind,
    plan: &VecchiaPlan,
    basis: &[Vec<f64>],
) -> Result<VecchiaFit> {
    let p = basis[0].len();
    if data.len() <= p + 1 {
        return Err(GeostatError::InsufficientData(format!(
            "REML with {p} trend terms needs more than {} points",
            p + 1
        )));
    }

    // Initial scales from the GLS-residual variance about the trend.
    let mean = data.mean();
    let var0 = (data
        .values()
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f64>()
        / data.len() as f64)
        .max(1e-12);
    let coords = data.coords();
    let (mut lo, mut hi) = ([f64::INFINITY; D], [f64::NEG_INFINITY; D]);
    for pt in coords {
        for d in 0..D {
            lo[d] = lo[d].min(pt[d]);
            hi[d] = hi[d].max(pt[d]);
        }
    }
    let extent = (0..D).map(|d| (hi[d] - lo[d]).powi(2)).sum::<f64>().sqrt();
    let range0 = (extent / 3.0).max(1e-9);

    let objective = |x: &[f64]| -> f64 {
        let nugget = x[0] * var0;
        let psill = x[1] * var0;
        let range = x[2] * range0;
        let mut pen = 0.0;
        if nugget < 0.0 {
            pen += nugget * nugget;
        }
        if psill <= 0.0 {
            pen += 1.0 + psill * psill;
        }
        if range <= 0.0 {
            pen += 1.0 + range * range;
        }
        if pen > 0.0 {
            return 1e12 * (1.0 + pen);
        }
        // A range far beyond the domain is unidentifiable (indistinguishable
        // from a trend); regularize it back rather than letting it run away.
        let range_max = 10.0 * extent;
        let reg = if range > range_max {
            ((range - range_max) / range_max).powi(2)
        } else {
            0.0
        };
        let model = VariogramModel {
            nugget,
            structures: vec![Structure::new(kind, psill, range)],
        };
        match reml_loglik_with_plan(data, &model, plan, basis) {
            Ok(ll) if ll.is_finite() => -ll + 1e6 * reg,
            _ => 1e12,
        }
    };

    let x0 = [0.1, 0.9, 1.0];
    let (xb, neg_ll) = nelder_mead(objective, &x0, 0.3, 2000);
    let model = VariogramModel::new(
        (xb[0] * var0).max(0.0),
        vec![Structure::new(
            kind,
            (xb[1] * var0).max(1e-12),
            (xb[2] * range0).max(1e-12),
        )],
    )?;
    Ok(VecchiaFit {
        model,
        loglik: -neg_ll,
    })
}

/// Asymptotic standard errors of a single-structure model's covariance
/// parameters `(nugget, partial sill, range)`, from the observed Fisher
/// information of the constant-mean Vecchia log-likelihood at `model`.
///
/// The information matrix is the numerical Hessian of the negative
/// log-likelihood; the standard errors are the square roots of the diagonal of
/// its inverse. A parameter sitting on a boundary (e.g.\ a zero nugget) or a
/// non-positive-definite Hessian yields `NaN` for that entry --- the asymptotic
/// theory does not apply there. This is inference the variogram-WLS path cannot
/// provide.
pub fn vecchia_param_se<const D: usize>(
    data: &PointSet<D>,
    model: &VariogramModel,
    m: usize,
    order: Option<&[usize]>,
) -> Result<[f64; 3]> {
    if model.structures.len() != 1 {
        return Err(GeostatError::InvalidParameter(
            "standard errors are defined for a single-structure model".into(),
        ));
    }
    let plan = vecchia_plan(data.coords(), m, order)?;
    let kind = model.structures[0].kind;
    let theta = [
        model.nugget,
        model.structures[0].sill,
        model.structures[0].range,
    ];

    // Negative log-likelihood at a parameter vector (reusing the plan).
    let nll = |p: &[f64; 3]| -> f64 {
        if p[1] <= 0.0 || p[2] <= 0.0 || p[0] < 0.0 {
            return f64::INFINITY;
        }
        let model = VariogramModel {
            nugget: p[0],
            structures: vec![Structure::new(kind, p[1], p[2])],
        };
        match loglik_with_plan(data, &model, &plan) {
            Ok(ll) if ll.is_finite() => -ll,
            _ => f64::INFINITY,
        }
    };

    // Relative finite-difference steps.
    let h: [f64; 3] = std::array::from_fn(|i| (theta[i].abs() * 1e-3).max(1e-5));
    let f0 = nll(&theta);
    let mut hess = [[0.0; 3]; 3];
    for i in 0..3 {
        let mut tp = theta;
        let mut tm = theta;
        tp[i] += h[i];
        tm[i] -= h[i];
        hess[i][i] = (nll(&tp) - 2.0 * f0 + nll(&tm)) / (h[i] * h[i]);
        for j in (i + 1)..3 {
            let mut tpp = theta;
            let mut tpm = theta;
            let mut tmp = theta;
            let mut tmm = theta;
            tpp[i] += h[i];
            tpp[j] += h[j];
            tpm[i] += h[i];
            tpm[j] -= h[j];
            tmp[i] -= h[i];
            tmp[j] += h[j];
            tmm[i] -= h[i];
            tmm[j] -= h[j];
            let v = (nll(&tpp) - nll(&tpm) - nll(&tmp) + nll(&tmm)) / (4.0 * h[i] * h[j]);
            hess[i][j] = v;
            hess[j][i] = v;
        }
    }

    // Invert the 3x3 information matrix; SE_i = sqrt((H^{-1})_ii).
    let flat: Vec<f64> = hess.iter().flatten().copied().collect();
    let arr = Array2::from_shape_vec((3, 3), flat).expect("3x3");
    let lu = match lu_factor(arr) {
        Ok(lu) => lu,
        Err(_) => return Ok([f64::NAN; 3]), // singular information
    };
    let mut se = [f64::NAN; 3];
    for i in 0..3 {
        let mut e = vec![0.0; 3];
        e[i] = 1.0;
        let col = lu.solve(e);
        se[i] = if col[i] > 0.0 {
            col[i].sqrt()
        } else {
            f64::NAN
        };
    }
    Ok(se)
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
        let model = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 1.0, 40.0)],
        )
        .unwrap();
        let exact = exact_loglik(&data, &model);
        // m >= n-1 -> every point conditions on all predecessors -> exact.
        let v = vecchia_loglik(&data, &model, data.len() - 1, None).unwrap();
        assert!((v - exact).abs() < 1e-9, "vecchia {v} vs exact {exact}");
    }

    #[test]
    fn approximation_is_close_with_maxmin() {
        let data = field(90, 11);
        let model = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 1.0, 30.0)],
        )
        .unwrap();
        let exact = exact_loglik(&data, &model);
        let approx = vecchia_loglik(&data, &model, 12, None).unwrap();
        // A modest conditioning size already tracks the exact log-likelihood.
        let rel = (approx - exact).abs() / exact.abs();
        assert!(
            rel < 0.02,
            "rel err {rel}: approx {approx} vs exact {exact}"
        );
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

    /// Reference O(n^2) farthest-point traversal (the pre-v0.7 code) used to
    /// verify the lazy-heap implementation is exact.
    fn maxmin_order_brute<const D: usize>(coords: &[[f64; D]]) -> Vec<usize> {
        let n = coords.len();
        let mut c = [0.0; D];
        for p in coords {
            for d in 0..D {
                c[d] += p[d] / n as f64;
            }
        }
        let first = (0..n)
            .min_by(|&i, &j| dist2(&coords[i], &c).total_cmp(&dist2(&coords[j], &c)))
            .unwrap();
        let mut order = vec![first];
        let mut chosen = vec![false; n];
        chosen[first] = true;
        let mut min_d2: Vec<f64> = (0..n).map(|i| dist2(&coords[i], &coords[first])).collect();
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

    #[test]
    fn lazy_maxmin_matches_the_exact_traversal() {
        // Continuous random coordinates (tie-free): the lazy-heap ordering
        // must reproduce the brute-force farthest-point traversal exactly.
        for seed in [3, 11, 29] {
            let data = field(400, seed);
            assert_eq!(
                maxmin_order(data.coords()),
                maxmin_order_brute(data.coords()),
                "seed {seed}"
            );
        }
    }

    #[test]
    fn incremental_predecessors_match_brute_force() {
        let data = field(300, 13);
        let coords = data.coords();
        let m = 8;
        let plan = vecchia_plan(coords, m, None).unwrap();
        for (k, nb) in plan.neighbours.iter().enumerate() {
            let target = plan.order[k];
            // Brute force: m nearest among the k predecessors.
            let mut prev: Vec<(f64, usize)> = plan.order[..k]
                .iter()
                .map(|&j| (dist2(&coords[target], &coords[j]), j))
                .collect();
            prev.sort_by(|a, b| a.0.total_cmp(&b.0));
            prev.truncate(m);
            let mut expected: Vec<usize> = prev.into_iter().map(|(_, j)| j).collect();
            let mut got = nb.clone();
            expected.sort_unstable();
            got.sort_unstable();
            assert_eq!(got, expected, "predecessors differ at position {k}");
        }
    }

    #[test]
    fn rejects_bad_args() {
        let data = field(10, 1);
        let model = VariogramModel::new(0.1, vec![Structure::new(ModelKind::Spherical, 1.0, 20.0)])
            .unwrap();
        assert!(vecchia_loglik(&data, &model, 0, None).is_err());
    }

    #[test]
    fn mle_full_conditioning_is_exact_optimum() {
        // With full conditioning the Vecchia likelihood is exact, so the MLE
        // optimum must be a local optimum of the exact log-likelihood: no small
        // parameter perturbation may raise it.
        let data = field(28, 5);
        let full = data.len() - 1;
        let fit = vecchia_mle(&data, ModelKind::Exponential, full, None).unwrap();
        let s = fit.model.structures[0];
        assert!(s.sill > 0.0 && s.range > 0.0 && fit.model.nugget >= 0.0);

        // fit.loglik is the exact log-likelihood at the optimum (full cond.).
        for (fn_, fp, fr) in [
            (1.0, 0.7, 1.0),
            (1.0, 1.4, 1.0),
            (1.0, 1.0, 0.7),
            (1.0, 1.0, 1.4),
            (3.0, 1.0, 1.0),
        ] {
            let pert = VariogramModel::new(
                (fit.model.nugget * fn_).max(1e-9),
                vec![Structure::new(s.kind, s.sill * fp, s.range * fr)],
            )
            .unwrap();
            let ll = vecchia_loglik(&data, &pert, full, None).unwrap();
            assert!(
                fit.loglik >= ll - 1e-6,
                "perturbation ({fn_},{fp},{fr}) beats the optimum: {ll} > {}",
                fit.loglik
            );
        }
    }

    #[test]
    fn mle_beats_a_wrong_model() {
        let data = field(60, 9);
        let fit = vecchia_mle(&data, ModelKind::Exponential, 12, None).unwrap();
        // The fitted likelihood must exceed that of a badly mis-scaled model.
        let wrong =
            VariogramModel::new(0.5, vec![Structure::new(ModelKind::Exponential, 0.1, 5.0)])
                .unwrap();
        let ll_wrong = vecchia_loglik(&data, &wrong, 15, None).unwrap();
        assert!(fit.loglik > ll_wrong, "{} vs wrong {ll_wrong}", fit.loglik);
    }

    #[test]
    fn plan_is_reusable() {
        let data = field(30, 2);
        let plan = vecchia_plan(data.coords(), 8, None).unwrap();
        assert_eq!(plan.order.len(), 30);
        let m1 = VariogramModel::new(0.1, vec![Structure::new(ModelKind::Spherical, 1.0, 20.0)])
            .unwrap();
        let m2 = VariogramModel::new(0.2, vec![Structure::new(ModelKind::Spherical, 2.0, 40.0)])
            .unwrap();
        // Same plan, different models -> two valid, distinct likelihoods.
        let a = super::loglik_with_plan(&data, &m1, &plan).unwrap();
        let b = super::loglik_with_plan(&data, &m2, &plan).unwrap();
        assert!(a.is_finite() && b.is_finite() && (a - b).abs() > 0.0);
    }

    /// Exact restricted log-likelihood with a polynomial trend, the oracle for
    /// the full-conditioning Vecchia REML.
    fn exact_reml(data: &PointSet, model: &VariogramModel, degree: u8) -> f64 {
        let n = data.len();
        let coords = data.coords();
        let z = data.values();
        let f = super::poly_basis(coords, degree);
        let p = f[0].len();
        // Sigma and its factorization.
        let mut cov = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                cov[a * n + b] = model.covariance_dh(sep(&coords[a], &coords[b]));
            }
        }
        let lu_s = crate::linalg::lu_factor(Array2::from_shape_vec((n, n), cov).unwrap()).unwrap();
        // Sigma^{-1} z and Sigma^{-1} F.
        let siz = lu_s.solve(z.to_vec());
        let sif: Vec<Vec<f64>> = (0..p)
            .map(|c| lu_s.solve((0..n).map(|r| f[r][c]).collect()))
            .collect();
        let mut amat = vec![0.0; p * p];
        let mut cvec = vec![0.0; p];
        for a in 0..p {
            cvec[a] = (0..n).map(|r| f[r][a] * siz[r]).sum();
            for b in 0..p {
                amat[a * p + b] = (0..n).map(|r| f[r][a] * sif[b][r]).sum();
            }
        }
        let lu_a = crate::linalg::lu_factor(Array2::from_shape_vec((p, p), amat).unwrap()).unwrap();
        let beta = lu_a.solve(cvec.clone());
        let ztsiz: f64 = (0..n).map(|r| z[r] * siz[r]).sum();
        let resid = ztsiz - cvec.iter().zip(&beta).map(|(&c, &b)| c * b).sum::<f64>();
        -0.5 * ((n - p) as f64 * (2.0 * PI).ln() + lu_s.ln_det_abs() + lu_a.ln_det_abs() + resid)
    }

    #[test]
    fn reml_full_conditioning_equals_exact() {
        let data = field(16, 4);
        let model = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 1.0, 35.0)],
        )
        .unwrap();
        let plan = vecchia_plan(data.coords(), data.len() - 1, None).unwrap();
        let basis = super::poly_basis(data.coords(), 1);
        let v = super::reml_loglik_with_plan(&data, &model, &plan, &basis).unwrap();
        let exact = exact_reml(&data, &model, 1);
        assert!(
            (v - exact).abs() < 1e-7,
            "vecchia REML {v} vs exact {exact}"
        );
    }

    /// Lower Cholesky factor of an SPD matrix (row-major), for drawing a GP.
    fn cholesky(n: usize, a: &[f64]) -> Vec<f64> {
        let mut l = vec![0.0; n * n];
        for j in 0..n {
            let mut d = a[j * n + j];
            for k in 0..j {
                d -= l[j * n + k] * l[j * n + k];
            }
            l[j * n + j] = d.max(1e-12).sqrt();
            for i in (j + 1)..n {
                let mut s = a[i * n + j];
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = s / l[j * n + j];
            }
        }
        l
    }

    #[test]
    fn reml_tames_the_range_under_a_trend() {
        // Draw a genuine short-range Gaussian field (range 10) from its exact
        // covariance, add a strong linear trend. Constant-mean ML absorbs the
        // trend as long-range correlation; REML(degree 1) removes it and
        // recovers a range near the truth, well below the ML range.
        let mut rng = Rng::new(20);
        let n = 90;
        let coords: Vec<[f64; 2]> = (0..n)
            .map(|_| [rng.uniform() * 60.0, rng.uniform() * 60.0])
            .collect();
        let truth = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 1.0, 10.0)],
        )
        .unwrap();
        let mut cov = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                cov[a * n + b] = truth.covariance_dh(sep(&coords[a], &coords[b]));
            }
        }
        let l = cholesky(n, &cov);
        let eps: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mut values = vec![0.0; n];
        for i in 0..n {
            let gp: f64 = (0..=i).map(|k| l[i * n + k] * eps[k]).sum();
            values[i] = 0.3 * coords[i][0] - 0.2 * coords[i][1] + gp; // trend + GP
        }
        let data = PointSet::new(coords, values).unwrap();
        let ml = vecchia_mle(&data, ModelKind::Exponential, 8, None).unwrap();
        let reml = vecchia_reml(&data, ModelKind::Exponential, 8, 1, None).unwrap();
        let ml_r = ml.model.structures[0].range;
        let reml_r = reml.model.structures[0].range;
        assert!(
            reml_r < ml_r,
            "REML range {reml_r} should be below constant-mean ML range {ml_r}"
        );
        assert!(
            reml_r < 40.0,
            "REML range {reml_r} should be near the truth (10)"
        );
    }

    #[test]
    fn reml_drift_recovers_range_with_a_covariate() {
        // Mean driven by a NON-polynomial covariate (a Gaussian bump in space)
        // plus a genuine short-range GP. A polynomial trend cannot absorb the
        // covariate, but external-drift REML can -- and recovers the range.
        let mut rng = Rng::new(21);
        let n = 90;
        let coords: Vec<[f64; 2]> = (0..n)
            .map(|_| [rng.uniform() * 60.0, rng.uniform() * 60.0])
            .collect();
        // Covariate: distance-to-centre style feature (non-polynomial effect).
        let cov: Vec<f64> = coords
            .iter()
            .map(|p| (-((p[0] - 30.0).powi(2) + (p[1] - 30.0).powi(2)) / 200.0).exp())
            .collect();
        let truth = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 1.0, 10.0)],
        )
        .unwrap();
        let mut sig = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                sig[a * n + b] = truth.covariance_dh(sep(&coords[a], &coords[b]));
            }
        }
        let l = cholesky(n, &sig);
        let eps: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let values: Vec<f64> = (0..n)
            .map(|i| 8.0 * cov[i] + (0..=i).map(|k| l[i * n + k] * eps[k]).sum::<f64>())
            .collect();
        let data = PointSet::new(coords, values).unwrap();

        let drift: Vec<Vec<f64>> = cov.iter().map(|&c| vec![c]).collect();
        let fit = vecchia_reml_drift(&data, ModelKind::Exponential, 8, &drift, None).unwrap();
        let r = fit.model.structures[0].range;
        assert!(
            r < 40.0,
            "covariate-REML range {r} should be near the truth (10)"
        );
    }

    #[test]
    fn reml_drift_rejects_bad_covariates() {
        let data = field(20, 1);
        // Wrong number of drift rows.
        assert!(vecchia_reml_drift(&data, ModelKind::Exponential, 5, &[vec![1.0]], None).is_err());
        // No covariate columns.
        let empty: Vec<Vec<f64>> = (0..20).map(|_| Vec::new()).collect();
        assert!(vecchia_reml_drift(&data, ModelKind::Exponential, 5, &empty, None).is_err());
    }

    #[test]
    fn param_se_are_finite_and_informative() {
        // Draw a GP from a known model; the fitted parameters' standard errors
        // should be finite, positive, and smaller than the estimates (i.e. the
        // parameters are identified).
        let mut rng = Rng::new(31);
        let n = 140;
        let coords: Vec<[f64; 2]> = (0..n)
            .map(|_| [rng.uniform() * 60.0, rng.uniform() * 60.0])
            .collect();
        let truth =
            VariogramModel::new(0.2, vec![Structure::new(ModelKind::Exponential, 1.0, 12.0)])
                .unwrap();
        let mut sig = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                sig[a * n + b] = truth.covariance_dh(sep(&coords[a], &coords[b]));
            }
        }
        let l = cholesky(n, &sig);
        let eps: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let values: Vec<f64> = (0..n)
            .map(|i| (0..=i).map(|k| l[i * n + k] * eps[k]).sum())
            .collect();
        let data = PointSet::new(coords, values).unwrap();

        let fit = vecchia_mle(&data, ModelKind::Exponential, 12, None).unwrap();
        let se = vecchia_param_se(&data, &fit.model, 12, None).unwrap();
        assert!(se[1].is_finite() && se[1] > 0.0, "sill SE {}", se[1]);
        assert!(se[2].is_finite() && se[2] > 0.0, "range SE {}", se[2]);
        assert!(
            se[2] < fit.model.structures[0].range,
            "range SE {} should be below the range estimate",
            se[2]
        );
    }

    #[test]
    fn param_se_rejects_multi_structure() {
        let data = field(20, 1);
        let model = VariogramModel::new(
            0.1,
            vec![
                Structure::new(ModelKind::Spherical, 0.5, 10.0),
                Structure::new(ModelKind::Exponential, 0.5, 30.0),
            ],
        )
        .unwrap();
        assert!(vecchia_param_se(&data, &model, 5, None).is_err());
    }

    #[test]
    fn reml_rejects_too_few_points() {
        let data = field(4, 1);
        // Degree 2 in 2-D needs 6 basis columns -> >7 points required.
        assert!(vecchia_reml(&data, ModelKind::Exponential, 3, 2, None).is_err());
    }
}
