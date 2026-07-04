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
use crate::linalg::{
    cholesky_factor_in_place, cholesky_forward_solve, cholesky_solve_in_place, lu_factor,
};
use crate::optim::nelder_mead_multistart;
use crate::search::BucketGrid;
use crate::variogram::{ModelKind, Structure, VariogramModel};

/// Vecchia needs a genuine covariance function to Cholesky-factor each
/// conditioning set; the unbounded [`ModelKind::Power`] has none (infinite
/// variance), unlike ordinary/universal kriging's semivariogram-form system
/// (see `crate::kriging::Kriging`, which does support it).
const POWER_UNSUPPORTED: &str = "Vecchia needs a valid covariance function and cannot use the \
     unbounded Power model (use crate::kriging::Kriging with Ordinary/Universal/ExternalDrift \
     instead, which krige directly in semivariogram form)";

fn reject_power_model(model: &VariogramModel) -> Result<()> {
    if model.has_power() {
        return Err(GeostatError::InvalidParameter(POWER_UNSUPPORTED.into()));
    }
    Ok(())
}

fn reject_power_kind(kind: ModelKind) -> Result<()> {
    if matches!(kind, ModelKind::Power(_)) {
        return Err(GeostatError::InvalidParameter(POWER_UNSUPPORTED.into()));
    }
    Ok(())
}

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
                // `sill - gamma_dh(..)` instead of `covariance_dh(..)`: the
                // latter recomputes `total_sill()` (a sum over structures)
                // on every pair; the caller already has it cached.
                ki[a] = sill - model.gamma_dh(sep(&coords[i], &coords[nb[a]]));
                for b in a..s {
                    let c = sill - model.gamma_dh(sep(&coords[nb[a]], &coords[nb[b]]));
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

/// Fraction of a candidate point's own `m` neighbours that may be new to a
/// growing Guinness block before the block is closed. Max-min ordering
/// spreads early selections across the whole domain (near-zero neighbour
/// overlap between consecutive points there), but late in the ordering,
/// once the domain has filled in, consecutive selections are often local
/// refinements whose neighbour sets do overlap heavily. This threshold lets
/// blocks form only where that overlap is real, so grouping never costs
/// more than the ungrouped path: a block that cannot grow past size 1
/// degenerates to exactly one factorization per point, same as unrouped.
const GUINNESS_MERGE_NEW_FRAC: f64 = 0.34;

/// Greedily partitions `plan.order` into Guinness blocks: geometry-only (no
/// covariance model needed), so it can be computed once per `(plan,
/// group_size)` and reused across every likelihood evaluation in an
/// optimizer's hot loop. Each block starts at the next unassigned position
/// and grows while the next candidate's own neighbour set is mostly already
/// covered by the block's accumulated predecessor set (see
/// [`GUINNESS_MERGE_NEW_FRAC`]), capped at `group_size`.
fn guinness_blocks(plan: &VecchiaPlan, group_size: usize) -> Vec<(usize, usize)> {
    let n = plan.order.len();
    let mut blocks = Vec::new();
    let mut v_ids: Vec<usize> = Vec::new();
    let mut k0 = 0;
    while k0 < n {
        v_ids.clear();
        v_ids.extend_from_slice(&plan.neighbours[k0]);
        let mut k1 = k0 + 1;
        while k1 < n && (k1 - k0) < group_size {
            let nb = &plan.neighbours[k1];
            let m = nb.len().max(1);
            let in_block = |j: &usize| plan.order[k0..k1].contains(j);
            let new_count = nb.iter().filter(|j| !v_ids.contains(j) && !in_block(j)).count();
            if new_count as f64 > GUINNESS_MERGE_NEW_FRAC * m as f64 {
                break;
            }
            for &j in nb {
                if !v_ids.contains(&j) && !in_block(&j) {
                    v_ids.push(j);
                }
            }
            k1 += 1;
        }
        blocks.push((k0, k1));
        k0 = k1;
    }
    blocks
}

/// Vecchia log-likelihood for a precomputed [`VecchiaPlan`], grouping
/// points into blocks (see [`guinness_blocks`]) that share **one** joint
/// Cholesky factorization (Guinness 2018).
///
/// A block's combined conditioning set `V = S ∪ block` (`S` = the union of
/// the block members' own nearest-predecessor sets, `block` = the block
/// members themselves) is, by construction of [`guinness_blocks`], close to
/// one member's own `m` neighbours, so factoring it once amortizes the
/// O(|V|^3) cost across the whole block instead of paying O(m^3) per point.
/// Per-point conditional means/variances then come from the *same*
/// whitening identity the unrouped path uses (`w = L^{-1}(z - mean)`;
/// `Var[z_t|z_{<t}] = L_tt^2`, `E[z_t|z_{<t}] = z_t - L_tt w_t`), which holds
/// for *any* valid ordering of `V`'s predecessors — so grouping several
/// members behind one factorization does not change the per-point
/// conditional mean/variance formula, only how it is computed. Each block
/// member's effective conditioning set becomes `S ∪ {earlier block
/// members}`, a superset of its own `m` nearest neighbours (by construction,
/// since `S` is their union) — a strictly richer approximation, never a
/// degraded one.
///
/// `group_size <= 1` is exactly [`loglik_with_plan`] (delegated to directly,
/// so the ungrouped path is untouched bit-for-bit). With full conditioning
/// (`m >= n-1`) grouping does not change the mathematical result: every
/// block member's true neighbour set is already `{0,...,k-1}` in order
/// position, which any block covering it reproduces exactly regardless of
/// `group_size`.
fn loglik_with_plan_grouped<const D: usize>(
    data: &PointSet<D>,
    model: &VariogramModel,
    plan: &VecchiaPlan,
    group_size: usize,
) -> Result<f64> {
    if group_size <= 1 {
        return loglik_with_plan(data, model, plan);
    }
    loglik_with_blocks(data, model, plan, &guinness_blocks(plan, group_size))
}

/// Core of [`loglik_with_plan_grouped`], taking a precomputed block
/// partition. `guinness_blocks` depends only on plan geometry (not the
/// covariance model), so an optimizer's hot loop computes it **once** and
/// reuses it across every likelihood evaluation instead of recomputing it
/// per iteration.
fn loglik_with_blocks<const D: usize>(
    data: &PointSet<D>,
    model: &VariogramModel,
    plan: &VecchiaPlan,
    blocks: &[(usize, usize)],
) -> Result<f64> {
    let coords = data.coords();
    let mean = data.mean();
    let z: Vec<f64> = data.values().iter().map(|v| v - mean).collect();
    let sill = model.total_sill();

    // Reused across every block.
    let mut v_ids: Vec<usize> = Vec::new();
    let mut kvv: Vec<f64> = Vec::new();
    let mut w: Vec<f64> = Vec::new();

    let mut loglik = 0.0;
    for &(k0, k1) in blocks {
        let block_ids = &plan.order[k0..k1];

        // S = union of the block members' own predecessor sets, excluding
        // members of this same block (those are appended below, in order,
        // and picked up by the sequential whitening recursion instead).
        v_ids.clear();
        for k in k0..k1 {
            for &j in &plan.neighbours[k] {
                if !block_ids.contains(&j) && !v_ids.contains(&j) {
                    v_ids.push(j);
                }
            }
        }
        let s_len = v_ids.len();
        v_ids.extend_from_slice(block_ids);
        let vn = v_ids.len();

        kvv.clear();
        kvv.resize(vn * vn, 0.0);
        for a in 0..vn {
            kvv[a * vn + a] = sill; // covariance_dh(0) == total_sill for any stationary model
            for b in (a + 1)..vn {
                let c = sill - model.gamma_dh(sep(&coords[v_ids[a]], &coords[v_ids[b]]));
                kvv[a * vn + b] = c;
                kvv[b * vn + a] = c;
            }
        }
        cholesky_factor_in_place(&mut kvv, vn)?;
        w.clear();
        w.extend(v_ids.iter().map(|&id| z[id]));
        cholesky_forward_solve(&kvv, vn, &mut w);

        for t in s_len..vn {
            let l_tt = kvv[t * vn + t];
            let var = l_tt * l_tt;
            let wt = w[t];
            loglik += -0.5 * ((2.0 * PI).ln() + var.ln() + wt * wt);
        }
    }
    Ok(loglik)
}

/// One Vecchia prediction: posterior mean and variance under the Vecchia
/// joint approximation.
#[derive(Debug, Clone, Copy)]
pub struct VecchiaEstimate {
    /// Predictive mean.
    pub value: f64,
    /// Predictive variance.
    pub variance: f64,
}

/// Vecchia prediction at `targets` (Katzfuss & Guinness 2021, §5): the
/// targets are appended *after* the data in max-min order, and each target
/// conditions on its `m` nearest previous points — observed data **and**
/// already-processed targets. Unlike plain moving-neighborhood kriging
/// (which conditions each target on data only), the target-on-target
/// conditioning makes the joint predictive distribution consistent across
/// targets, which is what lets a small `m` track the exact answer.
///
/// The mean of the field is treated as known and equal to the data mean
/// (simple-kriging convention, matching [`vecchia_loglik`]). Predictive
/// means propagate exactly through the ordered conditionals; predictive
/// variances propagate the per-target innovation coefficients (sparse rows
/// of the implied Cholesky factor), truncating relative coefficients below
/// `1e-8` — numerically irrelevant, but it bounds the fill-in.
///
/// With `m >= n + targets` the result equals exact global simple kriging.
pub fn vecchia_predict<const D: usize>(
    data: &PointSet<D>,
    model: &VariogramModel,
    targets: &[[f64; D]],
    m: usize,
) -> Result<Vec<VecchiaEstimate>> {
    reject_power_model(model)?;
    if m == 0 {
        return Err(GeostatError::InvalidParameter(
            "conditioning size m must be at least 1".into(),
        ));
    }
    if targets.is_empty() {
        return Ok(Vec::new());
    }
    if let Some((i, j)) = crate::data::duplicate_coord_pair(data.coords()) {
        return Err(GeostatError::DuplicatePoints(i, j));
    }
    let n = data.len();
    let mean = data.mean();
    let z: Vec<f64> = data.values().iter().map(|v| v - mean).collect();
    let c0 = model.covariance_dh([0.0; D]);

    // Conditioning store over data + processed targets.
    let mut min = data.coords()[0];
    let mut max = min;
    for p in data.coords().iter().chain(targets) {
        for d in 0..D {
            min[d] = min[d].min(p[d]);
            max[d] = max[d].max(p[d]);
        }
    }
    let mut grid = BucketGrid::new(min, max, n + targets.len());
    // store_coords[j]: coordinates of store entry j (0..n = data, then
    // targets in processing order), with their centred conditional means.
    let mut store_coords: Vec<[f64; D]> = data.coords().to_vec();
    let mut store_mean: Vec<f64> = z.clone();
    for &p in data.coords() {
        grid.insert(p);
    }

    // Innovation-coefficient rows (sparse) for processed targets, indexed by
    // innovation id; observed data carry no innovations.
    let mut rows: Vec<Vec<(u32, f64)>> = Vec::with_capacity(targets.len());
    let order = maxmin_order(targets);
    let mut estimates = vec![
        VecchiaEstimate {
            value: f64::NAN,
            variance: f64::NAN,
        };
        targets.len()
    ];

    let mut kss: Vec<f64> = Vec::new();
    let mut ki: Vec<f64> = Vec::new();
    let mut w: Vec<f64> = Vec::new();
    let mut acc: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    let coef_tol = 1e-8 * c0.sqrt();

    for &t in &order {
        let target = targets[t];
        let nb = grid.k_nearest(target, m, None);
        let (mu, var, row) = if nb.is_empty() {
            (0.0, c0, vec![(rows.len() as u32, c0.max(0.0).sqrt())])
        } else {
            let s = nb.len();
            kss.clear();
            kss.resize(s * s, 0.0);
            ki.clear();
            ki.resize(s, 0.0);
            for a in 0..s {
                ki[a] = c0 - model.gamma_dh(sep(&target, &store_coords[nb[a]]));
                for b in a..s {
                    let c = c0 - model.gamma_dh(sep(&store_coords[nb[a]], &store_coords[nb[b]]));
                    kss[a * s + b] = c;
                    kss[b * s + a] = c;
                }
            }
            w.clear();
            w.extend_from_slice(&ki);
            cholesky_solve_in_place(&mut kss, s, &mut w)?;
            let mu: f64 = w.iter().zip(&nb).map(|(&wj, &j)| wj * store_mean[j]).sum();
            let d = (c0 - w.iter().zip(&ki).map(|(&wj, &kj)| wj * kj).sum::<f64>()).max(0.0);
            // Propagate innovation coefficients: this target's deviation is
            // sum_j w_j (deviation of conditioning target j) + sqrt(d) eps_t.
            acc.clear();
            for (&wj, &j) in w.iter().zip(&nb) {
                if j >= n {
                    for &(inno, coef) in &rows[j - n] {
                        *acc.entry(inno).or_insert(0.0) += wj * coef;
                    }
                }
            }
            let mut row: Vec<(u32, f64)> = acc
                .iter()
                .filter(|&(_, &c)| c.abs() > coef_tol)
                .map(|(&i, &c)| (i, c))
                .collect();
            row.push((rows.len() as u32, d.sqrt()));
            let var: f64 = row.iter().map(|&(_, c)| c * c).sum();
            (mu, var, row)
        };
        estimates[t] = VecchiaEstimate {
            value: mean + mu,
            variance: var,
        };
        grid.insert(target);
        store_coords.push(target);
        store_mean.push(mu);
        rows.push(row);
    }
    Ok(estimates)
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
    reject_power_model(model)?;
    let plan = vecchia_plan(data.coords(), m, order)?;
    loglik_with_plan(data, model, &plan)
}

/// Like [`vecchia_loglik`], but amortizes Cholesky factorizations across
/// blocks of `group_size` consecutive points in the ordering (Guinness 2018
/// grouping; see [`loglik_with_plan_grouped`]). `group_size <= 1` reproduces
/// `vecchia_loglik` exactly; larger values trade a richer (never worse)
/// per-point conditioning set for fewer, larger factorizations.
pub fn vecchia_loglik_grouped<const D: usize>(
    data: &PointSet<D>,
    model: &VariogramModel,
    m: usize,
    order: Option<&[usize]>,
    group_size: usize,
) -> Result<f64> {
    reject_power_model(model)?;
    let plan = vecchia_plan(data.coords(), m, order)?;
    loglik_with_plan_grouped(data, model, &plan, group_size)
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
    reject_power_kind(kind)?;
    let plan = vecchia_plan(data.coords(), m, order)?;
    mle_fit_with_plan(data, kind, &plan, 1)
}

/// Like [`vecchia_mle`], but every likelihood evaluation in the optimizer's
/// hot loop uses [`loglik_with_plan_grouped`] with the given `group_size`
/// (Guinness 2018 grouping). This is where grouping earns its keep: the
/// optimizer calls the likelihood thousands of times, so amortizing
/// factorizations across blocks compounds directly into faster fits.
/// `group_size <= 1` reproduces `vecchia_mle` exactly.
pub fn vecchia_mle_grouped<const D: usize>(
    data: &PointSet<D>,
    kind: ModelKind,
    m: usize,
    order: Option<&[usize]>,
    group_size: usize,
) -> Result<VecchiaFit> {
    reject_power_kind(kind)?;
    let plan = vecchia_plan(data.coords(), m, order)?;
    mle_fit_with_plan(data, kind, &plan, group_size)
}

fn mle_fit_with_plan<const D: usize>(
    data: &PointSet<D>,
    kind: ModelKind,
    plan: &VecchiaPlan,
    group_size: usize,
) -> Result<VecchiaFit> {
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

    // Block geometry depends only on the plan, not the covariance model:
    // compute it once and reuse it across every objective evaluation
    // instead of paying for it on each of the optimizer's thousands of calls.
    let blocks = (group_size > 1).then(|| guinness_blocks(plan, group_size));

    // `nugget = x[0]^2` (smooth, non-negative, hits exact 0); psill and range
    // are log-parametrized (strictly positive, span orders of magnitude), so
    // the domain is intrinsic and no boundary penalty is needed
    // (AUDIT-2026-07.md §2.6).
    let objective = |x: &[f64]| -> f64 {
        let nugget = x[0] * x[0];
        let psill = x[1].exp();
        let range = x[2].exp();
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
        let ll = match &blocks {
            Some(b) => loglik_with_blocks(data, &model, plan, b),
            None => loglik_with_plan(data, &model, plan),
        };
        match ll {
            Ok(ll) if ll.is_finite() => -ll + 1e6 * reg,
            _ => 1e12,
        }
    };

    // Range is the classic multimodal parameter for covariance MLE
    // (short-vs-long-range local optima); multi-start around the initial guess.
    let ln_range0 = range0.ln();
    let starts: Vec<Vec<f64>> = [0.3_f64, 1.0, 3.0]
        .into_iter()
        .map(|f| vec![(0.1 * var0).sqrt(), (0.9 * var0).ln(), ln_range0 + f.ln()])
        .collect();
    let (xb, neg_ll) = nelder_mead_multistart(objective, &starts, 0.3, 2000);
    let model = VariogramModel::new(xb[0] * xb[0], vec![Structure::new(kind, xb[1].exp(), xb[2].exp())])?;
    Ok(VecchiaFit {
        model,
        loglik: -neg_ll,
    })
}

/// Fits a Matérn model (nugget + partial sill + range + smoothness `ν`)
/// jointly by maximum likelihood on the Vecchia approximation, instead of
/// fixing `ν` and calling `vecchia_mle(data, ModelKind::Matern(nu), m,
/// order)`. Same `ν`-range confounding caveat as [`fit_matern`] (its WLS
/// counterpart): `ν` is multi-started alongside range rather than treated
/// as a fixed input.
///
/// [`fit_matern`]: crate::variogram::fit_matern
pub fn vecchia_mle_matern<const D: usize>(
    data: &PointSet<D>,
    m: usize,
    order: Option<&[usize]>,
) -> Result<VecchiaFit> {
    let plan = vecchia_plan(data.coords(), m, order)?;

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

    let objective = |x: &[f64]| -> f64 {
        let nugget = x[0] * x[0];
        let psill = x[1].exp();
        let range = x[2].exp();
        let nu = x[3].exp();
        let range_max = 10.0 * extent;
        let reg = if range > range_max {
            ((range - range_max) / range_max).powi(2)
        } else {
            0.0
        };
        let model = VariogramModel {
            nugget,
            structures: vec![Structure::new(ModelKind::Matern(nu), psill, range)],
        };
        match loglik_with_plan(data, &model, &plan) {
            Ok(ll) if ll.is_finite() => -ll + 1e6 * reg,
            _ => 1e12,
        }
    };

    // Each start here re-evaluates the Vecchia likelihood (an O(n m^3) pass
    // over every conditioning pair, each needing a K_nu Bessel quadrature)
    // up to `max_iter` times, unlike a fixed-kind fit -- keep the (nu0,
    // range0) grid modest so a joint fit stays practical.
    let ln_range0 = range0.ln();
    let starts: Vec<Vec<f64>> = [0.7_f64, 2.5]
        .into_iter()
        .flat_map(|nu0| {
            [0.5_f64, 2.0].into_iter().map(move |f| {
                vec![
                    (0.1 * var0).sqrt(),
                    (0.9 * var0).ln(),
                    ln_range0 + f.ln(),
                    nu0.ln(),
                ]
            })
        })
        .collect();
    let (xb, neg_ll) = nelder_mead_multistart(objective, &starts, 0.3, 1200);
    let model = VariogramModel::new(
        xb[0] * xb[0],
        vec![Structure::new(ModelKind::Matern(xb[3].exp()), xb[1].exp(), xb[2].exp())],
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
                ki[aa] = sill - model.gamma_dh(sep(&coords[i], &coords[nb[aa]]));
                for bb in aa..s {
                    let cv = sill - model.gamma_dh(sep(&coords[nb[aa]], &coords[nb[bb]]));
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

/// Grouped counterpart of [`reml_loglik_with_plan`] (Guinness 2018 blocks,
/// see [`loglik_with_plan_grouped`] for the algorithm): each block's shared
/// Cholesky factor whitens `z` *and* every basis column with one extra
/// forward-solve per column, since `uz`/`uf` are both just the factor applied
/// to a data vector. Takes a precomputed block partition (see
/// [`loglik_with_blocks`] for why an optimizer's hot loop computes blocks
/// once and reuses them across every evaluation).
fn reml_loglik_with_blocks<const D: usize>(
    data: &PointSet<D>,
    model: &VariogramModel,
    plan: &VecchiaPlan,
    basis: &[Vec<f64>],
    blocks: &[(usize, usize)],
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

    let mut v_ids: Vec<usize> = Vec::new();
    let mut kvv: Vec<f64> = Vec::new();
    let mut wz: Vec<f64> = Vec::new();
    let mut wf: Vec<Vec<f64>> = vec![Vec::new(); p];

    for &(k0, k1) in blocks {
        let block_ids = &plan.order[k0..k1];

        v_ids.clear();
        for k in k0..k1 {
            for &j in &plan.neighbours[k] {
                if !block_ids.contains(&j) && !v_ids.contains(&j) {
                    v_ids.push(j);
                }
            }
        }
        let s_len = v_ids.len();
        v_ids.extend_from_slice(block_ids);
        let vn = v_ids.len();

        kvv.clear();
        kvv.resize(vn * vn, 0.0);
        for aa in 0..vn {
            kvv[aa * vn + aa] = sill;
            for bb in (aa + 1)..vn {
                let cv = sill - model.gamma_dh(sep(&coords[v_ids[aa]], &coords[v_ids[bb]]));
                kvv[aa * vn + bb] = cv;
                kvv[bb * vn + aa] = cv;
            }
        }
        cholesky_factor_in_place(&mut kvv, vn)?;

        // Each right-hand side (z, then every basis column) is whitened
        // against the *same* factor: uz/uf share the identical telescoping
        // identity, differing only in which data vector is forward-solved.
        wz.clear();
        wz.extend(v_ids.iter().map(|&id| z[id]));
        cholesky_forward_solve(&kvv, vn, &mut wz);
        for (col, wcol) in wf.iter_mut().enumerate() {
            wcol.clear();
            wcol.extend(v_ids.iter().map(|&id| basis[id][col]));
            cholesky_forward_solve(&kvv, vn, wcol);
        }

        for t in s_len..vn {
            let l_tt = kvv[t * vn + t];
            let d = l_tt * l_tt;
            let uz = wz[t];
            logdet += d.ln();
            uz_uz += uz * uz;
            for aa in 0..p {
                let ufa = wf[aa][t];
                c[aa] += ufa * uz;
                for bb in 0..p {
                    a[aa * p + bb] += ufa * wf[bb][t];
                }
            }
        }
    }

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
    reject_power_kind(kind)?;
    let plan = vecchia_plan(data.coords(), m, order)?;
    let basis = poly_basis(data.coords(), drift_degree);
    reml_fit_with_basis(data, kind, &plan, &basis, 1)
}

/// Like [`vecchia_reml`], grouping the REML likelihood's Cholesky
/// factorizations by `group_size` (Guinness 2018; see
/// [`reml_loglik_with_blocks`]). `group_size <= 1` reproduces
/// `vecchia_reml` exactly.
pub fn vecchia_reml_grouped<const D: usize>(
    data: &PointSet<D>,
    kind: ModelKind,
    m: usize,
    drift_degree: u8,
    order: Option<&[usize]>,
    group_size: usize,
) -> Result<VecchiaFit> {
    reject_power_kind(kind)?;
    let plan = vecchia_plan(data.coords(), m, order)?;
    let basis = poly_basis(data.coords(), drift_degree);
    reml_fit_with_basis(data, kind, &plan, &basis, group_size)
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
    reject_power_kind(kind)?;
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
    reml_fit_with_basis(data, kind, &plan, &basis, 1)
}

/// Grouped counterpart of [`vecchia_reml_drift`] (see
/// [`reml_loglik_with_blocks`]). `group_size <= 1` reproduces
/// `vecchia_reml_drift` exactly.
#[allow(clippy::too_many_arguments)]
pub fn vecchia_reml_drift_grouped<const D: usize>(
    data: &PointSet<D>,
    kind: ModelKind,
    m: usize,
    drift: &[Vec<f64>],
    order: Option<&[usize]>,
    group_size: usize,
) -> Result<VecchiaFit> {
    reject_power_kind(kind)?;
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
    reml_fit_with_basis(data, kind, &plan, &basis, group_size)
}

/// Shared REML optimizer: fits (nugget, sill, range) for `kind` by maximizing
/// the Vecchia REML likelihood under an arbitrary trend basis.
fn reml_fit_with_basis<const D: usize>(
    data: &PointSet<D>,
    kind: ModelKind,
    plan: &VecchiaPlan,
    basis: &[Vec<f64>],
    group_size: usize,
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

    // See `mle_fit_with_plan`: block geometry is model-independent, so it is
    // computed once and reused across every objective evaluation.
    let blocks = (group_size > 1).then(|| guinness_blocks(plan, group_size));

    // Same smooth non-negative/log parametrization as `vecchia_mle` (see the
    // comment there); no boundary penalty needed.
    let objective = |x: &[f64]| -> f64 {
        let nugget = x[0] * x[0];
        let psill = x[1].exp();
        let range = x[2].exp();
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
        let ll = match &blocks {
            Some(b) => reml_loglik_with_blocks(data, &model, plan, basis, b),
            None => reml_loglik_with_plan(data, &model, plan, basis),
        };
        match ll {
            Ok(ll) if ll.is_finite() => -ll + 1e6 * reg,
            _ => 1e12,
        }
    };

    let ln_range0 = range0.ln();
    let starts: Vec<Vec<f64>> = [0.3_f64, 1.0, 3.0]
        .into_iter()
        .map(|f| vec![(0.1 * var0).sqrt(), (0.9 * var0).ln(), ln_range0 + f.ln()])
        .collect();
    let (xb, neg_ll) = nelder_mead_multistart(objective, &starts, 0.3, 2000);
    let model = VariogramModel::new(xb[0] * xb[0], vec![Structure::new(kind, xb[1].exp(), xb[2].exp())])?;
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
    reject_power_model(model)?;
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
    fn power_model_is_rejected_everywhere() {
        let data = field(20, 1);
        let power_model =
            VariogramModel::new(0.0, vec![Structure::new(ModelKind::Power(1.0), 1.0, 1.0)])
                .unwrap();
        assert!(vecchia_predict(&data, &power_model, &[[1.0, 1.0]], 5).is_err());
        assert!(vecchia_loglik(&data, &power_model, 5, None).is_err());
        assert!(vecchia_loglik_grouped(&data, &power_model, 5, None, 2).is_err());
        assert!(vecchia_param_se(&data, &power_model, 5, None).is_err());
        assert!(vecchia_mle(&data, ModelKind::Power(1.0), 5, None).is_err());
        assert!(vecchia_mle_grouped(&data, ModelKind::Power(1.0), 5, None, 2).is_err());
        assert!(vecchia_reml(&data, ModelKind::Power(1.0), 5, 1, None).is_err());
        assert!(vecchia_reml_grouped(&data, ModelKind::Power(1.0), 5, 1, None, 2).is_err());
        let drift = vec![vec![1.0]; data.len()];
        assert!(vecchia_reml_drift(&data, ModelKind::Power(1.0), 5, &drift, None).is_err());
        assert!(
            vecchia_reml_drift_grouped(&data, ModelKind::Power(1.0), 5, &drift, None, 2).is_err()
        );
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
    fn guinness_blocks_partition_the_order_within_group_size() {
        let data = field(4000, 42);
        let m = 30;
        let plan = vecchia_plan(data.coords(), m, None).unwrap();
        for &g in &[2usize, 4, 8, 16] {
            let blocks = guinness_blocks(&plan, g);
            let mut covered = 0;
            for &(k0, k1) in &blocks {
                assert_eq!(k0, covered, "blocks must tile 0..n with no gaps/overlap");
                assert!(k1 > k0 && k1 - k0 <= g, "block ({k0},{k1}) exceeds group_size {g}");
                covered = k1;
            }
            assert_eq!(covered, plan.order.len());
        }
    }

    #[test]
    fn grouped_matches_ungrouped_when_group_size_is_one() {
        let data = field(90, 11);
        let model = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 1.0, 30.0)],
        )
        .unwrap();
        let ungrouped = vecchia_loglik(&data, &model, 12, None).unwrap();
        let grouped = vecchia_loglik_grouped(&data, &model, 12, None, 1).unwrap();
        assert!(
            (ungrouped - grouped).abs() < 1e-12,
            "ungrouped {ungrouped} vs group_size=1 {grouped}"
        );
    }

    #[test]
    fn grouped_full_conditioning_equals_exact() {
        let data = field(20, 4);
        let model = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 1.0, 40.0)],
        )
        .unwrap();
        let exact = exact_loglik(&data, &model);
        // m >= n-1 -> every block's combined conditioning set is still
        // exactly {0,...,k-1} regardless of group_size (see the doc comment
        // on `loglik_with_plan_grouped`).
        for group_size in [2, 3, 5, 8] {
            let v = vecchia_loglik_grouped(&data, &model, data.len() - 1, None, group_size)
                .unwrap();
            assert!(
                (v - exact).abs() < 1e-8,
                "group_size {group_size}: vecchia {v} vs exact {exact}"
            );
        }
    }

    #[test]
    fn grouped_approximation_is_at_least_as_close_as_ungrouped() {
        let data = field(90, 11);
        let model = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 1.0, 30.0)],
        )
        .unwrap();
        let exact = exact_loglik(&data, &model);
        let ungrouped_err = (vecchia_loglik(&data, &model, 12, None).unwrap() - exact).abs();
        for group_size in [2, 4, 6] {
            let grouped = vecchia_loglik_grouped(&data, &model, 12, None, group_size).unwrap();
            let grouped_err = (grouped - exact).abs();
            // Grouping conditions each point on a superset of its own m
            // nearest neighbours, so the approximation should not get worse.
            assert!(
                grouped_err <= ungrouped_err + 1e-9,
                "group_size {group_size}: grouped err {grouped_err} vs ungrouped err {ungrouped_err}"
            );
        }
    }

    #[test]
    fn vecchia_mle_grouped_matches_ungrouped_under_full_conditioning() {
        let data = field(16, 21);
        let m = data.len() - 1;
        let ungrouped = vecchia_mle(&data, ModelKind::Exponential, m, None).unwrap();
        let grouped = vecchia_mle_grouped(&data, ModelKind::Exponential, m, None, 4).unwrap();
        assert!(
            (ungrouped.loglik - grouped.loglik).abs() < 1e-6,
            "ungrouped ll {} vs grouped ll {}",
            ungrouped.loglik,
            grouped.loglik
        );
        assert!((ungrouped.model.nugget - grouped.model.nugget).abs() < 1e-4);
        assert!((ungrouped.model.structures[0].sill - grouped.model.structures[0].sill).abs() < 1e-3);
        assert!(
            (ungrouped.model.structures[0].range - grouped.model.structures[0].range).abs() < 1e-2
        );
    }

    #[test]
    fn predict_full_conditioning_equals_exact_simple_kriging() {
        use crate::kriging::{Kriging, KrigingConfig, KrigingMethod};
        let data = field(60, 5);
        let model =
            VariogramModel::new(0.1, vec![Structure::new(ModelKind::Exponential, 0.9, 30.0)])
                .unwrap();
        let targets: Vec<[f64; 2]> = vec![[12.0, 33.0], [55.0, 71.0], [80.0, 20.0], [40.0, 40.0]];
        // Full conditioning: every target sees all data and all previous
        // targets, so the Vecchia joint is exact and the marginals must
        // match global simple kriging with the same (data-mean) mean.
        let est = vecchia_predict(&data, &model, &targets, 1000).unwrap();
        let sk = Kriging::new(
            &data,
            &model,
            KrigingConfig {
                method: KrigingMethod::Simple { mean: data.mean() },
                ..Default::default()
            },
        )
        .unwrap();
        for (t, e) in targets.iter().zip(&est) {
            let exact = sk.predict(*t).unwrap();
            assert!(
                (e.value - exact.value).abs() < 1e-8,
                "mean {} vs exact {}",
                e.value,
                exact.value
            );
            assert!(
                (e.variance - exact.variance).abs() < 1e-8,
                "var {} vs exact {}",
                e.variance,
                exact.variance
            );
        }
    }

    #[test]
    fn predict_small_m_tracks_exact_kriging() {
        use crate::kriging::{Kriging, KrigingConfig, KrigingMethod};
        let data = field(150, 9);
        let model = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 0.95, 25.0)],
        )
        .unwrap();
        let targets: Vec<[f64; 2]> = (0..40)
            .map(|i| [2.5 * i as f64 + 1.0, 97.0 - 2.3 * i as f64])
            .collect();
        let est = vecchia_predict(&data, &model, &targets, 25).unwrap();
        let sk = Kriging::new(
            &data,
            &model,
            KrigingConfig {
                method: KrigingMethod::Simple { mean: data.mean() },
                ..Default::default()
            },
        )
        .unwrap();
        let mut max_dv = 0.0_f64;
        let mut max_dm = 0.0_f64;
        for (t, e) in targets.iter().zip(&est) {
            let exact = sk.predict(*t).unwrap();
            max_dm = max_dm.max((e.value - exact.value).abs());
            max_dv = max_dv.max((e.variance - exact.variance).abs());
            assert!(e.variance >= 0.0 && e.variance <= model.total_sill() + 1e-9);
        }
        assert!(max_dm < 0.02, "mean deviation {max_dm}");
        assert!(max_dv < 0.02, "variance deviation {max_dv}");
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
    fn vecchia_mle_matern_matches_or_beats_a_fixed_nu_fit() {
        let data = field(30, 9);
        let m = 8;
        // nu=1.5 is directly representable in the joint search (one of the
        // multi-start seeds), so the joint optimum's likelihood must be at
        // least as good as fixing nu at that same value.
        let fixed = vecchia_mle(&data, ModelKind::Matern15, m, None).unwrap();
        let joint = vecchia_mle_matern(&data, m, None).unwrap();
        assert!(
            joint.loglik >= fixed.loglik - 1e-6,
            "joint {} vs fixed-nu=1.5 {}",
            joint.loglik,
            fixed.loglik
        );
        let s = joint.model.structures[0];
        let ModelKind::Matern(nu) = s.kind else {
            panic!("expected Matern, got {:?}", s.kind)
        };
        assert!(nu > 0.0 && nu.is_finite(), "nu = {nu}");
        assert!(s.sill > 0.0 && s.range > 0.0 && joint.model.nugget >= 0.0);
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
