//! Ordinary co-kriging under a linear model of coregionalization (LMC).
//!
//! The LMC shares structure shapes (kind, range, anisotropy) across all
//! variables; each structure carries a symmetric positive semi-definite
//! matrix of (co)sills. Unbiasedness follows the traditional convention:
//! primary weights sum to 1, each secondary variable's weights sum to 0.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::grid::Grid2D;
use crate::kriging::KrigingEstimate;
use crate::linalg::solve;
use crate::search::KdTree;
use crate::variogram::{Anisotropy, ExperimentalVariogram, ModelKind, VariogramModel};

/// One shared structure of the LMC with its matrix of (co)sills.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LmcStructure {
    /// Model family (shared by all variable pairs).
    pub kind: ModelKind,
    /// Range parameter (shared).
    pub range: f64,
    /// Optional geometric anisotropy (shared).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anis: Option<Anisotropy>,
    /// Symmetric PSD matrix of partial (co)sills, `sills[u][v]`.
    pub sills: Vec<Vec<f64>>,
}

/// Linear model of coregionalization: a nugget matrix plus shared
/// structures with per-structure sill matrices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lmc {
    /// Symmetric PSD nugget matrix, `nugget[u][v]`.
    pub nugget: Vec<Vec<f64>>,
    /// Shared structures.
    pub structures: Vec<LmcStructure>,
}

/// Checks that `m` is square of size `n`, symmetric and PSD (via Cholesky
/// with a relative tolerance).
#[allow(clippy::needless_range_loop)]
fn check_psd(m: &[Vec<f64>], n: usize, what: &str) -> Result<()> {
    if m.len() != n || m.iter().any(|r| r.len() != n) {
        return Err(GeostatError::DimensionMismatch(format!(
            "{what} matrix must be {n}x{n}"
        )));
    }
    let scale = m
        .iter()
        .flatten()
        .fold(0.0_f64, |a, v| a.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    for i in 0..n {
        for j in 0..i {
            if (m[i][j] - m[j][i]).abs() > 1e-9 * scale {
                return Err(GeostatError::InvalidParameter(format!(
                    "{what} matrix is not symmetric at ({i},{j})"
                )));
            }
        }
        if !m[i][i].is_finite() || m[i][i] < -1e-12 * scale {
            return Err(GeostatError::InvalidParameter(format!(
                "{what} matrix has negative diagonal at {i}"
            )));
        }
    }
    // Cholesky with tolerance: fails on a meaningfully negative pivot.
    let mut a: Vec<Vec<f64>> = m.to_vec();
    for k in 0..n {
        for j in 0..k {
            a[k][k] -= a[k][j] * a[k][j];
        }
        if a[k][k] < -1e-9 * scale {
            return Err(GeostatError::InvalidParameter(format!(
                "{what} matrix is not positive semi-definite (pivot {k})"
            )));
        }
        let d = a[k][k].max(0.0).sqrt();
        for i in (k + 1)..n {
            for j in 0..k {
                a[i][k] -= a[i][j] * a[k][j];
            }
            a[i][k] = if d > 1e-12 * scale.sqrt() {
                a[i][k] / d
            } else {
                0.0
            };
        }
    }
    Ok(())
}

impl Lmc {
    /// Builds and validates an LMC: consistent dimensions, symmetric PSD
    /// matrices, valid structure parameters.
    pub fn new(nugget: Vec<Vec<f64>>, structures: Vec<LmcStructure>) -> Result<Self> {
        let n = nugget.len();
        if n == 0 {
            return Err(GeostatError::InvalidParameter(
                "LMC needs at least one variable".into(),
            ));
        }
        check_psd(&nugget, n, "nugget")?;
        for (s, st) in structures.iter().enumerate() {
            if !(st.range > 0.0) || !st.range.is_finite() {
                return Err(GeostatError::InvalidParameter(format!(
                    "structure {s}: range must be finite and > 0, got {}",
                    st.range
                )));
            }
            if let Some(a) = st.anis
                && (!(a.ratio > 0.0) || a.ratio > 1.0 || !a.azimuth_deg.is_finite())
            {
                return Err(GeostatError::InvalidParameter(format!(
                    "structure {s}: invalid anisotropy"
                )));
            }
            check_psd(&st.sills, n, &format!("structure {s} sill"))?;
        }
        let lmc = Self { nugget, structures };
        for v in 0..n {
            if !(lmc.total_sill(v, v) > 0.0) {
                return Err(GeostatError::InvalidParameter(format!(
                    "variable {v} has non-positive total sill"
                )));
            }
        }
        Ok(lmc)
    }

    /// Number of variables.
    pub fn n_vars(&self) -> usize {
        self.nugget.len()
    }

    /// Total (co)sill for a variable pair.
    pub fn total_sill(&self, u: usize, v: usize) -> f64 {
        self.nugget[u][v] + self.structures.iter().map(|s| s.sills[u][v]).sum::<f64>()
    }

    /// Cross-semivariance for a separation vector.
    pub fn gamma_dh<const D: usize>(&self, u: usize, v: usize, dh: [f64; D]) -> f64 {
        if dh.iter().all(|&x| x == 0.0) {
            return 0.0;
        }
        let mut g = self.nugget[u][v];
        for st in &self.structures {
            // Reuse the single-variable structure math for the shape,
            // without allocating a throwaway `VariogramModel`/`Vec` per pair
            // (this runs inside the O(k^2) cokriging system loop).
            let shape = crate::variogram::Structure {
                kind: st.kind,
                sill: 1.0,
                range: st.range,
                anis: st.anis,
            };
            g += st.sills[u][v] * st.kind.g(shape.effective_h(dh), st.range);
        }
        g
    }

    /// Cross-covariance for a separation vector:
    /// `C_uv(dh) = total_sill_uv - gamma_uv(dh)`.
    pub fn covariance_dh<const D: usize>(&self, u: usize, v: usize, dh: [f64; D]) -> f64 {
        self.total_sill(u, v) - self.gamma_dh(u, v, dh)
    }

    /// The direct (single-variable) model of variable `v`.
    pub fn direct_model(&self, v: usize) -> Result<VariogramModel> {
        VariogramModel::new(
            self.nugget[v][v],
            self.structures
                .iter()
                .map(|s| crate::variogram::Structure {
                    kind: s.kind,
                    sill: s.sills[v][v],
                    range: s.range,
                    anis: s.anis,
                })
                .collect(),
        )
    }
}

/// Co-kriging search configuration (applied per variable).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CoKrigingConfig {
    /// Maximum nearest neighbors per variable (all points when unset).
    pub max_neighbors: Option<usize>,
    /// Search radius per variable.
    pub search_radius: Option<f64>,
    /// Relative ridge added to the data-data diagonal of the co-kriging system
    /// (each diagonal entry is multiplied by `1 + ridge`). Co-kriging matrices
    /// are notoriously ill-conditioned — a strongly correlated secondary or a
    /// cross-variogram fitted on few collocated points can drive the system
    /// near-singular and blow the weights up. A tiny ridge (e.g. `1e-6`)
    /// stabilizes it. Default `0.0` keeps the exact, gstat-validated system.
    pub ridge: f64,
}

/// Ordinary co-kriging predictor. `datasets[0]` is the primary variable
/// (the one being predicted); the rest are secondaries. Datasets need not
/// be collocated (heterotopic co-kriging is supported).
#[derive(Debug)]
pub struct CoKriging<'a, const D: usize = 2> {
    datasets: Vec<&'a PointSet<D>>,
    lmc: &'a Lmc,
    config: CoKrigingConfig,
    trees: Vec<Option<KdTree<D>>>,
}

impl<'a, const D: usize> CoKriging<'a, D> {
    /// Builds the predictor, validating dimensions.
    pub fn new(
        datasets: Vec<&'a PointSet<D>>,
        lmc: &'a Lmc,
        config: CoKrigingConfig,
    ) -> Result<Self> {
        if datasets.len() < 2 {
            return Err(GeostatError::InvalidParameter(
                "co-kriging needs a primary and at least one secondary variable".into(),
            ));
        }
        if datasets.len() != lmc.n_vars() {
            return Err(GeostatError::DimensionMismatch(format!(
                "{} datasets vs {} LMC variables",
                datasets.len(),
                lmc.n_vars()
            )));
        }
        if config.max_neighbors == Some(0) {
            return Err(GeostatError::InvalidParameter(
                "max_neighbors must be at least 1".into(),
            ));
        }
        if let Some(r) = config.search_radius
            && !(r > 0.0)
        {
            return Err(GeostatError::InvalidParameter(format!(
                "search radius must be positive, got {r}"
            )));
        }
        let local = config.max_neighbors.is_some() || config.search_radius.is_some();
        let trees = datasets
            .iter()
            .map(|d| local.then(|| KdTree::build(d.coords())))
            .collect();
        Ok(Self {
            datasets,
            lmc,
            config,
            trees,
        })
    }

    fn neighbors(&self, v: usize, target: [f64; D]) -> Vec<usize> {
        match &self.trees[v] {
            None => (0..self.datasets[v].len()).collect(),
            Some(tree) => tree.k_nearest(
                target,
                self.config.max_neighbors.unwrap_or(self.datasets[v].len()),
                self.config.search_radius,
            ),
        }
    }

    /// Co-kriging estimate of the primary variable at a target location.
    pub fn predict(&self, target: [f64; D]) -> Result<KrigingEstimate> {
        self.predict_inner(target, None)
    }

    /// Block co-kriging: predicts the average of the primary variable over a
    /// block centered at `center`, discretized at `center + offsets[u]`.
    pub fn predict_block(&self, center: [f64; D], offsets: &[[f64; D]]) -> Result<KrigingEstimate> {
        if offsets.is_empty() {
            return Err(GeostatError::InvalidParameter(
                "block discretization needs at least one point".into(),
            ));
        }
        self.predict_inner(center, Some(offsets))
    }

    fn predict_inner(
        &self,
        center: [f64; D],
        offsets: Option<&[[f64; D]]>,
    ) -> Result<KrigingEstimate> {
        let n_vars = self.datasets.len();
        let nbs: Vec<Vec<usize>> = (0..n_vars).map(|v| self.neighbors(v, center)).collect();
        if nbs[0].is_empty() {
            return Err(GeostatError::NoNeighbors);
        }
        let counts: Vec<usize> = nbs.iter().map(Vec::len).collect();
        let n_total: usize = counts.iter().sum();
        // Block offsets per variable.
        let mut offset = vec![0usize; n_vars];
        for v in 1..n_vars {
            offset[v] = offset[v - 1] + counts[v - 1];
        }
        let dim = n_total + n_vars;

        // Point-to-(point or block) cross-covariance of variable `v` against
        // the primary (variable 0) at `pi`.
        let rhs_cov = |v: usize, pi: [f64; D]| -> f64 {
            match offsets {
                None => {
                    let mut dh = [0.0; D];
                    for dd in 0..D {
                        dh[dd] = pi[dd] - center[dd];
                    }
                    self.lmc.covariance_dh(v, 0, dh)
                }
                Some(offs) => {
                    let mut acc = 0.0;
                    for off in offs {
                        let mut dh = [0.0; D];
                        for dd in 0..D {
                            dh[dd] = pi[dd] - (center[dd] + off[dd]);
                        }
                        acc += self.lmc.covariance_dh(v, 0, dh);
                    }
                    acc / offs.len() as f64
                }
            }
        };

        let mut a = Array2::<f64>::zeros((dim, dim));
        let mut b = vec![0.0; dim];
        for v in 0..n_vars {
            for (ii, &i) in nbs[v].iter().enumerate() {
                let pi = self.datasets[v].coord(i);
                let row = offset[v] + ii;
                for w in v..n_vars {
                    for (jj, &j) in nbs[w].iter().enumerate() {
                        let col = offset[w] + jj;
                        if col < row {
                            continue;
                        }
                        let pj = self.datasets[w].coord(j);
                        let mut dh = [0.0; D];
                        for dd in 0..D {
                            dh[dd] = pi[dd] - pj[dd];
                        }
                        let c = self.lmc.covariance_dh(v, w, dh);
                        a[[row, col]] = c;
                        a[[col, row]] = c;
                    }
                }
                // Unbiasedness: one Lagrange multiplier per variable.
                a[[row, n_total + v]] = 1.0;
                a[[n_total + v, row]] = 1.0;
                b[row] = rhs_cov(v, pi);
            }
        }
        // Primary weights sum to 1, secondary weights to 0.
        b[n_total] = 1.0;

        // Optional ridge on the data-data diagonal to stabilize ill-conditioned
        // systems (leaves the Lagrange rows untouched).
        if self.config.ridge > 0.0 {
            for k in 0..n_total {
                a[[k, k]] *= 1.0 + self.config.ridge;
            }
        }

        let b0 = b.clone();
        let w = solve(a, b)?;

        let mut value = 0.0;
        for v in 0..n_vars {
            for (ii, &i) in nbs[v].iter().enumerate() {
                value += w[offset[v] + ii] * self.datasets[v].value(i);
            }
        }
        let reduction: f64 = (0..dim).map(|i| w[i] * b0[i]).sum();
        // Target variance: C(0) of the primary at a point, or the within-block
        // average covariance C̄(B,B) (nugget excluded for coincident points,
        // matching block kriging / gstat).
        let c_target = match offsets {
            None => self.lmc.total_sill(0, 0),
            Some(offs) => {
                let c0_cont = self.lmc.total_sill(0, 0) - self.lmc.nugget[0][0];
                let mut cbb = 0.0;
                for (ui, u) in offs.iter().enumerate() {
                    for (vi, vv) in offs.iter().enumerate() {
                        cbb += if ui == vi {
                            c0_cont
                        } else {
                            let mut dh = [0.0; D];
                            for dd in 0..D {
                                dh[dd] = u[dd] - vv[dd];
                            }
                            self.lmc.covariance_dh(0, 0, dh)
                        };
                    }
                }
                cbb / (offs.len() * offs.len()) as f64
            }
        };
        let variance = (c_target - reduction).max(0.0);
        // w[n_total] is the multiplier on the primary unbiasedness constraint.
        Ok(KrigingEstimate {
            value,
            variance,
            lagrange: Some(w[n_total]),
        })
    }

    /// Estimates at many targets, in parallel (NaN on failed systems).
    pub fn predict_many(&self, targets: &[[f64; D]]) -> Vec<KrigingEstimate> {
        crate::parallel::par_map(targets.len(), |i| {
            self.predict(targets[i]).unwrap_or(KrigingEstimate {
                value: f64::NAN,
                variance: f64::NAN,
                lagrange: None,
            })
        })
    }
}

impl CoKriging<'_, 2> {
    /// Co-kriging over all grid cell centers.
    pub fn predict_grid(&self, grid: &Grid2D) -> (Vec<f64>, Vec<f64>) {
        let ests = self.predict_many(&grid.centers());
        ests.into_iter().map(|e| (e.value, e.variance)).unzip()
    }

    /// Block co-kriging over all grid cells, with blocks of `block_size`
    /// discretized as a regular `discr[0]` x `discr[1]` point grid.
    pub fn predict_block_grid(
        &self,
        grid: &Grid2D,
        block_size: [f64; 2],
        discr: [usize; 2],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let offsets = crate::kriging::block_offsets(block_size, discr)?;
        let centers = grid.centers();
        let ests = crate::parallel::par_map(centers.len(), |i| {
            self.predict_block(centers[i], &offsets)
                .unwrap_or(KrigingEstimate {
                    value: f64::NAN,
                    variance: f64::NAN,
                    lagrange: None,
                })
        });
        Ok(ests.into_iter().map(|e| (e.value, e.variance)).unzip())
    }
}

/// Auto-fits a 2-variable LMC from the raw point sets: direct variograms on
/// each variable's own points, cross-variogram on the *collocated subset*
/// (points shared by both, matched on exact coordinates), template chosen by
/// [`fit_best`](crate::variogram::fit_best) over `kinds` on the primary.
///
/// This is the shared auto-fit path of the CLI and Python front-ends. For
/// isotopic data (all points shared, same order) it reduces exactly to
/// fitting the direct cross-variogram.
pub fn fit_lmc_collocated<const D: usize>(
    primary: &PointSet<D>,
    secondary: &PointSet<D>,
    cfg: &crate::variogram::VariogramConfig,
    kinds: &[ModelKind],
) -> Result<Lmc> {
    use std::collections::HashMap;
    let ea = crate::variogram::experimental_variogram(primary, cfg)?;
    let eb = crate::variogram::experimental_variogram(secondary, cfg)?;

    // Collocated subset (exact coordinate match) for the cross-variogram.
    let key = |c: &[f64; D]| -> [u64; D] { std::array::from_fn(|d| (c[d] + 0.0).to_bits()) };
    let sec_lookup: HashMap<[u64; D], f64> = secondary
        .coords()
        .iter()
        .zip(secondary.values())
        .map(|(c, &v)| (key(c), v))
        .collect();
    let (mut co_coords, mut co_pv, mut co_sv) = (Vec::new(), Vec::new(), Vec::new());
    for (c, &v) in primary.coords().iter().zip(primary.values()) {
        if let Some(&s) = sec_lookup.get(&key(c)) {
            co_coords.push(*c);
            co_pv.push(v);
            co_sv.push(s);
        }
    }
    if co_coords.len() < cfg.n_lags {
        return Err(GeostatError::InsufficientData(format!(
            "{} collocated primary/secondary points for {} lags: too few to \
             fit the cross-variogram",
            co_coords.len(),
            cfg.n_lags
        )));
    }
    let prim_co = PointSet::new(co_coords.clone(), co_pv)?;
    let sec_co = PointSet::new(co_coords, co_sv)?;
    let eab = crate::variogram::experimental_cross_variogram(&prim_co, &sec_co, cfg)?;

    let template = crate::variogram::fit_best(&ea, kinds)?;
    fit_lmc(&ea, &eb, &eab, &template.model)
}

/// One experimental curve prepared for the LMC fit: semivariances, WLS
/// weights and the basis functions evaluated at each retained bin.
struct LmcCurve {
    gamma: Vec<f64>,
    w: Vec<f64>,
    /// `basis[bin][k]`: nugget (k = 0) and unit-sill structures.
    basis: Vec<Vec<f64>>,
}

/// Fits a 2-variable LMC by the iterative Goulard & Voltz (1992) algorithm.
///
/// The structure shapes (nugget presence, kinds, ranges) are taken from
/// `template` (typically a fit of the primary variable). Starting from the
/// per-pair WLS solution (weights `N/h²`, gstat's default), the algorithm
/// cycles over structures: each per-structure sill matrix is re-solved by
/// WLS given the others and projected onto the PSD cone (eigenvalue
/// clipping), until the weighted SSE stops improving. When no PSD
/// constraint is active this reduces exactly to the separable per-pair WLS
/// optimum.
///
/// The template must be isotropic: the experimental inputs are
/// omnidirectional curves, so an anisotropic template would fit sills
/// against a direction-averaged curve and then apply them directionally —
/// silently inconsistent. Fit anisotropy separately or supply the LMC
/// explicitly.
pub fn fit_lmc(
    direct_a: &ExperimentalVariogram,
    direct_b: &ExperimentalVariogram,
    cross_ab: &ExperimentalVariogram,
    template: &VariogramModel,
) -> Result<Lmc> {
    if template.structures.iter().any(|s| s.anis.is_some()) {
        return Err(GeostatError::InvalidParameter(
            "fit_lmc requires an isotropic template: sills are fitted to \
             omnidirectional curves and cannot be reused anisotropically \
             (fit directional models separately or supply the LMC explicitly)"
                .into(),
        ));
    }
    let n_str = template.structures.len();
    let n_par = 1 + n_str; // nugget + one sill per structure

    let prep = |ev: &ExperimentalVariogram| -> Result<LmcCurve> {
        let mut curve = LmcCurve {
            gamma: Vec::new(),
            w: Vec::new(),
            basis: Vec::new(),
        };
        for b in &ev.bins {
            if b.n_pairs == 0 || !(b.h > 0.0) || !b.gamma.is_finite() {
                continue;
            }
            let mut row = vec![1.0; n_par];
            for (s, st) in template.structures.iter().enumerate() {
                row[1 + s] = st.kind.g(b.h, st.range);
            }
            curve.gamma.push(b.gamma);
            curve.w.push(b.n_pairs as f64 / (b.h * b.h));
            curve.basis.push(row);
        }
        if curve.gamma.len() < n_par + 1 {
            return Err(GeostatError::InsufficientData(format!(
                "LMC fit needs more non-empty lag bins than parameters ({n_par})"
            )));
        }
        Ok(curve)
    };
    let curves = [prep(direct_a)?, prep(direct_b)?, prep(cross_ab)?];

    // Initial estimate: independent WLS per pair (normal equations with a
    // tiny ridge against collinear bases), then a one-shot PSD projection.
    let wls = |curve: &LmcCurve| -> Result<Vec<f64>> {
        let mut ata = Array2::<f64>::zeros((n_par, n_par));
        let mut atb = vec![0.0; n_par];
        for ((row, &g), &wgt) in curve.basis.iter().zip(&curve.gamma).zip(&curve.w) {
            for r in 0..n_par {
                for c in 0..n_par {
                    ata[[r, c]] += wgt * row[r] * row[c];
                }
                atb[r] += wgt * row[r] * g;
            }
        }
        let trace_scale = (0..n_par).map(|r| ata[[r, r]]).fold(0.0_f64, f64::max);
        for r in 0..n_par {
            ata[[r, r]] += 1e-9 * trace_scale.max(f64::MIN_POSITIVE);
        }
        solve(ata, atb)
    };
    let sa = wls(&curves[0])?;
    let sb = wls(&curves[1])?;
    let sab = wls(&curves[2])?;

    // sills[k] = (aa, bb, ab) of parameter k, kept PSD at all times.
    let mut sills: Vec<(f64, f64, f64)> = (0..n_par)
        .map(|k| {
            let m = project_psd_2x2(sa[k].max(0.0), sab[k], sb[k].max(0.0));
            (m[0][0], m[1][1], m[0][1])
        })
        .collect();

    // Weighted SSE of a candidate sill set over the three curves, in the
    // Goulard-Voltz (Frobenius) convention: the cross entry appears twice in
    // the symmetric coregionalization matrix, so its curve counts double.
    // Under this norm the eigenvalue clip is the exact solution of each
    // per-structure subproblem when the curves share bins and weights.
    let entry = |s: &(f64, f64, f64), pair: usize| match pair {
        0 => s.0,
        1 => s.1,
        _ => s.2,
    };
    let wss = |sills: &[(f64, f64, f64)]| -> f64 {
        let mut total = 0.0;
        for (pair, curve) in curves.iter().enumerate() {
            let mult = if pair == 2 { 2.0 } else { 1.0 };
            for ((row, &g), &wgt) in curve.basis.iter().zip(&curve.gamma).zip(&curve.w) {
                let fit: f64 = (0..n_par).map(|k| entry(&sills[k], pair) * row[k]).sum();
                total += mult * wgt * (g - fit) * (g - fit);
            }
        }
        total
    };

    // Goulard–Voltz coordinate descent: re-solve one structure's sill matrix
    // given the others, project onto PSD, keep while the WSS improves.
    let mut best = wss(&sills);
    for _ in 0..100 {
        let previous = sills.clone();
        for k in 0..n_par {
            let mut beta = [0.0_f64; 3];
            for (pair, curve) in curves.iter().enumerate() {
                let (mut num, mut den) = (0.0, 0.0);
                for ((row, &g), &wgt) in curve.basis.iter().zip(&curve.gamma).zip(&curve.w) {
                    let others: f64 = (0..n_par)
                        .filter(|&j| j != k)
                        .map(|j| entry(&sills[j], pair) * row[j])
                        .sum();
                    num += wgt * row[k] * (g - others);
                    den += wgt * row[k] * row[k];
                }
                beta[pair] = if den > 0.0 { num / den } else { 0.0 };
            }
            let m = project_psd_2x2(beta[0].max(0.0), beta[2], beta[1].max(0.0));
            sills[k] = (m[0][0], m[1][1], m[0][1]);
        }
        let current = wss(&sills);
        if current > best {
            // The projection is not exactly the weighted subproblem optimum
            // when the three curves carry different weights; never accept a
            // WSS increase.
            sills = previous;
            break;
        }
        let converged = best - current <= 1e-12 * best.max(1e-300);
        best = current;
        if converged {
            break;
        }
    }

    let to_matrix = |s: &(f64, f64, f64)| vec![vec![s.0, s.2], vec![s.2, s.1]];
    let nugget = to_matrix(&sills[0]);
    let structures = template
        .structures
        .iter()
        .enumerate()
        .map(|(s, st)| LmcStructure {
            kind: st.kind,
            range: st.range,
            anis: None,
            sills: to_matrix(&sills[1 + s]),
        })
        .collect();
    Lmc::new(nugget, structures)
}

/// Projects a symmetric 2x2 matrix `[[a, b], [b, c]]` onto the PSD cone by
/// clipping negative eigenvalues.
fn project_psd_2x2(a: f64, b: f64, c: f64) -> Vec<Vec<f64>> {
    let tr = a + c;
    let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let l1 = 0.5 * (tr + disc);
    let l2 = 0.5 * (tr - disc);
    if l2 >= 0.0 {
        return vec![vec![a, b], vec![b, c]];
    }
    // Eigenvector for l1.
    let (vx, vy) = if b.abs() > 1e-300 {
        (l1 - c, b)
    } else if a >= c {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let norm = (vx * vx + vy * vy).sqrt().max(f64::MIN_POSITIVE);
    let (vx, vy) = (vx / norm, vy / norm);
    let l1 = l1.max(0.0);
    vec![
        vec![l1 * vx * vx, l1 * vx * vy],
        vec![l1 * vx * vy, l1 * vy * vy],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kriging::{Kriging, KrigingConfig};
    use crate::variogram::{
        Structure, VariogramConfig, experimental_cross_variogram, experimental_variogram,
    };

    fn primary() -> PointSet {
        PointSet::new(
            vec![
                [0.0, 0.0],
                [10.0, 0.0],
                [0.0, 10.0],
                [10.0, 10.0],
                [4.0, 6.0],
            ],
            vec![1.0, 2.0, 1.5, 2.5, 1.7],
        )
        .unwrap()
    }

    fn secondary() -> PointSet {
        // Collocated, correlated variable plus an extra location.
        PointSet::new(
            vec![
                [0.0, 0.0],
                [10.0, 0.0],
                [0.0, 10.0],
                [10.0, 10.0],
                [4.0, 6.0],
                [6.0, 3.0],
            ],
            vec![2.1, 4.2, 3.0, 5.1, 3.5, 3.9],
        )
        .unwrap()
    }

    fn lmc(cross: f64) -> Lmc {
        Lmc::new(
            vec![vec![0.05, 0.0], vec![0.0, 0.1]],
            vec![LmcStructure {
                kind: ModelKind::Spherical,
                range: 15.0,
                anis: None,
                sills: vec![vec![1.0, cross], vec![cross, 2.0]],
            }],
        )
        .unwrap()
    }

    #[test]
    fn lmc_validation() {
        // Non-PSD sill matrix rejected (cross > sqrt(1*2)).
        assert!(
            Lmc::new(
                vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                vec![LmcStructure {
                    kind: ModelKind::Spherical,
                    range: 10.0,
                    anis: None,
                    sills: vec![vec![1.0, 1.6], vec![1.6, 2.0]],
                }]
            )
            .is_err()
        );
        // Asymmetric rejected.
        assert!(Lmc::new(vec![vec![1.0, 0.5], vec![0.2, 1.0]], vec![]).is_err());
        // Valid model passes and exposes direct models.
        let m = lmc(0.8);
        let d0 = m.direct_model(0).unwrap();
        assert!((d0.total_sill() - 1.05).abs() < 1e-12);
    }

    #[test]
    fn cokriging_is_exact_at_primary_data() {
        let p = primary();
        let s = secondary();
        let m = lmc(0.8);
        let ck = CoKriging::new(vec![&p, &s], &m, CoKrigingConfig::default()).unwrap();
        for i in 0..p.len() {
            let est = ck.predict(p.coord(i)).unwrap();
            assert!(
                (est.value - p.value(i)).abs() < 1e-7,
                "point {i}: {} vs {}",
                est.value,
                p.value(i)
            );
            assert!(est.variance < 1e-7);
        }
    }

    #[test]
    fn block_cokriging_averages_point_cokriging() {
        // With a global neighborhood, the block co-kriging estimate equals
        // the average of point co-kriging estimates over the discretization
        // (linearity), and the block variance is below the central point.
        let p = primary();
        let s = secondary();
        let m = lmc(0.8);
        let ck = CoKriging::new(vec![&p, &s], &m, CoKrigingConfig::default()).unwrap();
        let center = [5.0, 5.0];
        let offsets = crate::kriging::block_offsets([4.0, 4.0], [3, 3]).unwrap();
        let block = ck.predict_block(center, &offsets).unwrap();
        let mean_pts = offsets
            .iter()
            .map(|o| {
                ck.predict([center[0] + o[0], center[1] + o[1]])
                    .unwrap()
                    .value
            })
            .sum::<f64>()
            / offsets.len() as f64;
        assert!(
            (block.value - mean_pts).abs() < 1e-9,
            "block {} vs mean of points {mean_pts}",
            block.value
        );
        let point_var = ck.predict(center).unwrap().variance;
        assert!(
            block.variance < point_var,
            "block var {} vs point var {point_var}",
            block.variance
        );
        assert!(ck.predict_block(center, &[]).is_err());
    }

    #[test]
    fn zero_cross_correlation_reduces_to_ordinary_kriging() {
        let p = primary();
        let s = secondary();
        let m = lmc(0.0);
        let ck = CoKriging::new(vec![&p, &s], &m, CoKrigingConfig::default()).unwrap();
        let direct = m.direct_model(0).unwrap();
        let ok = Kriging::new(&p, &direct, KrigingConfig::default()).unwrap();
        for target in [[5.0, 5.0], [2.0, 8.0], [12.0, 4.0]] {
            let a = ck.predict(target).unwrap();
            let b = ok.predict(target).unwrap();
            assert!(
                (a.value - b.value).abs() < 1e-9,
                "{target:?}: {} vs {}",
                a.value,
                b.value
            );
            assert!((a.variance - b.variance).abs() < 1e-9);
        }
    }

    #[test]
    fn secondary_reduces_variance() {
        let p = primary();
        let s = secondary();
        let with_cross = lmc(1.2);
        let without = lmc(0.0);
        let target = [6.0, 3.0]; // a secondary-only location
        let ck1 = CoKriging::new(vec![&p, &s], &with_cross, CoKrigingConfig::default()).unwrap();
        let ck0 = CoKriging::new(vec![&p, &s], &without, CoKrigingConfig::default()).unwrap();
        let v1 = ck1.predict(target).unwrap().variance;
        let v0 = ck0.predict(target).unwrap().variance;
        assert!(
            v1 < v0,
            "correlated secondary must reduce variance: {v1} vs {v0}"
        );
    }

    #[test]
    fn fit_lmc_recovers_synthetic_sills() {
        // Build synthetic experimental curves exactly on a known LMC.
        let truth = lmc(0.9);
        let cfg = VariogramConfig {
            n_lags: 12,
            max_dist: 24.0,
            direction: None,
        };
        let mk = |u: usize, v: usize| -> ExperimentalVariogram {
            let width = cfg.max_dist / cfg.n_lags as f64;
            let bins = (0..cfg.n_lags)
                .map(|i| {
                    let h = (i as f64 + 0.5) * width;
                    crate::variogram::LagBin {
                        h,
                        gamma: truth.gamma_dh(u, v, [h, 0.0]),
                        n_pairs: 50,
                    }
                })
                .collect();
            ExperimentalVariogram {
                bins,
                max_dist: cfg.max_dist,
            }
        };
        let template =
            VariogramModel::new(0.05, vec![Structure::new(ModelKind::Spherical, 1.0, 15.0)])
                .unwrap();
        let fitted = fit_lmc(&mk(0, 0), &mk(1, 1), &mk(0, 1), &template).unwrap();
        assert!((fitted.structures[0].sills[0][0] - 1.0).abs() < 0.05);
        assert!((fitted.structures[0].sills[1][1] - 2.0).abs() < 0.05);
        assert!((fitted.structures[0].sills[0][1] - 0.9).abs() < 0.05);
        assert!((fitted.nugget[0][0] - 0.05).abs() < 0.02);
    }

    #[test]
    fn fit_lmc_projects_to_psd() {
        // Cross sills exceeding the Cauchy-Schwarz bound get projected.
        let cfg_width = 2.0;
        let mk_gamma = |sill: f64, nug: f64| -> ExperimentalVariogram {
            let m =
                VariogramModel::new(nug, vec![Structure::new(ModelKind::Spherical, sill, 15.0)])
                    .unwrap();
            let bins = (0..12)
                .map(|i| {
                    let h = (i as f64 + 0.5) * cfg_width;
                    crate::variogram::LagBin {
                        h,
                        gamma: m.gamma(h),
                        n_pairs: 50,
                    }
                })
                .collect();
            ExperimentalVariogram {
                bins,
                max_dist: 24.0,
            }
        };
        let template =
            VariogramModel::new(0.0, vec![Structure::new(ModelKind::Spherical, 1.0, 15.0)])
                .unwrap();
        // Direct sills 1 and 1, "cross" sill 1.8 (impossible): must clip.
        let fitted = fit_lmc(
            &mk_gamma(1.0, 0.0),
            &mk_gamma(1.0, 0.0),
            &mk_gamma(1.8, 0.0),
            &template,
        )
        .unwrap();
        let s = &fitted.structures[0].sills;
        assert!(s[0][1] * s[0][1] <= s[0][0] * s[1][1] + 1e-9);
    }

    #[test]
    fn fit_lmc_rejects_anisotropic_template() {
        let cfg_width = 2.0;
        let mk = || -> ExperimentalVariogram {
            let m = VariogramModel::new(0.0, vec![Structure::new(ModelKind::Spherical, 1.0, 15.0)])
                .unwrap();
            let bins = (0..12)
                .map(|i| {
                    let h = (i as f64 + 0.5) * cfg_width;
                    crate::variogram::LagBin {
                        h,
                        gamma: m.gamma(h),
                        n_pairs: 50,
                    }
                })
                .collect();
            ExperimentalVariogram {
                bins,
                max_dist: 24.0,
            }
        };
        let template = VariogramModel::new(
            0.0,
            vec![Structure {
                kind: ModelKind::Spherical,
                sill: 1.0,
                range: 15.0,
                anis: Some(crate::variogram::Anisotropy {
                    azimuth_deg: 30.0,
                    ratio: 0.5,
                    ratio_z: 1.0,
                }),
            }],
        )
        .unwrap();
        assert!(fit_lmc(&mk(), &mk(), &mk(), &template).is_err());
    }

    #[test]
    fn goulard_voltz_beats_one_shot_clipping() {
        // Direct curves carry a real nugget (0.3) and structure sill 1; the
        // cross curve has no nugget but an impossible structure sill of 1.8.
        // The one-shot clip of [[1, 1.8], [1.8, 1]] leaves a large residual
        // on the cross curve that a re-fitted cross nugget (bounded by the
        // direct nuggets) can absorb - exactly the coupling the iterated
        // Goulard-Voltz pass exploits.
        let cfg_width = 2.0;
        let mk_gamma = |sill: f64, nug: f64| -> ExperimentalVariogram {
            let m =
                VariogramModel::new(nug, vec![Structure::new(ModelKind::Spherical, sill, 15.0)])
                    .unwrap();
            let bins = (0..12)
                .map(|i| {
                    let h = (i as f64 + 0.5) * cfg_width;
                    crate::variogram::LagBin {
                        h,
                        gamma: m.gamma(h),
                        n_pairs: 50,
                    }
                })
                .collect();
            ExperimentalVariogram {
                bins,
                max_dist: 24.0,
            }
        };
        let template =
            VariogramModel::new(0.1, vec![Structure::new(ModelKind::Spherical, 1.0, 15.0)])
                .unwrap();
        let curves = [mk_gamma(1.0, 0.3), mk_gamma(1.0, 0.3), mk_gamma(1.8, 0.0)];
        let fitted = fit_lmc(&curves[0], &curves[1], &curves[2], &template).unwrap();

        // Frobenius-convention WSS (cross curve counted twice).
        let wss_of = |nug: &[Vec<f64>], sills: &[Vec<f64>]| -> f64 {
            let base =
                VariogramModel::new(0.0, vec![Structure::new(ModelKind::Spherical, 1.0, 15.0)])
                    .unwrap();
            let mut total = 0.0;
            for (pair, (i, j)) in [(0usize, 0usize), (1, 1), (0, 1)].iter().enumerate() {
                let mult = if pair == 2 { 2.0 } else { 1.0 };
                for b in &curves[pair].bins {
                    let fit = nug[*i][*j] + sills[*i][*j] * base.gamma(b.h);
                    let w = b.n_pairs as f64 / (b.h * b.h);
                    total += mult * w * (b.gamma - fit) * (b.gamma - fit);
                }
            }
            total
        };
        // One-shot reference: independent WLS recovers the true per-pair
        // coefficients exactly (the curves are noiseless), so the clip acts
        // on nugget [[0.3, 0], [0, 0.3]] (already PSD) and structure
        // [[1, 1.8], [1.8, 1]] -> [[1.4, 1.4], [1.4, 1.4]].
        let clip_nug = vec![vec![0.3, 0.0], vec![0.0, 0.3]];
        let clip_str = vec![vec![1.4, 1.4], vec![1.4, 1.4]];
        let wss_gv = wss_of(&fitted.nugget, &fitted.structures[0].sills);
        let wss_clip = wss_of(&clip_nug, &clip_str);
        assert!(
            wss_gv <= wss_clip + 1e-9,
            "GV wss {wss_gv} vs one-shot {wss_clip}"
        );
        assert!(wss_gv < 0.98 * wss_clip, "expected strict improvement");
        // The improvement comes through a positive cross nugget within the
        // PSD bound of the direct nuggets.
        let cn = fitted.nugget[0][1];
        assert!(
            cn > 0.01 && cn * cn <= fitted.nugget[0][0] * fitted.nugget[1][1] + 1e-9,
            "cross nugget {cn}"
        );
    }

    #[test]
    fn cross_variogram_feeds_lmc_fit() {
        // End-to-end smoke: experimental direct + cross variograms from
        // correlated synthetic data fit into a valid LMC.
        let mut rng = crate::rng::Rng::new(3);
        let mut coords = Vec::new();
        let mut va = Vec::new();
        let mut vb = Vec::new();
        for _ in 0..150 {
            let x = rng.uniform() * 100.0;
            let y = rng.uniform() * 100.0;
            let base = (x / 25.0).sin() + (y / 25.0).cos();
            coords.push([x, y]);
            va.push(base + 0.1 * rng.normal());
            vb.push(2.0 * base + 0.2 * rng.normal());
        }
        let a = PointSet::new(coords.clone(), va).unwrap();
        let b = PointSet::new(coords, vb).unwrap();
        let cfg = VariogramConfig {
            n_lags: 12,
            max_dist: 50.0,
            direction: None,
        };
        let ea = experimental_variogram(&a, &cfg).unwrap();
        let eb = experimental_variogram(&b, &cfg).unwrap();
        let eab = experimental_cross_variogram(&a, &b, &cfg).unwrap();
        let template = crate::variogram::fit_best(&ea, &[ModelKind::Spherical]).unwrap();
        let lmc = fit_lmc(&ea, &eb, &eab, &template.model).unwrap();
        assert_eq!(lmc.n_vars(), 2);
        // Positive cross-correlation captured.
        assert!(lmc.total_sill(0, 1) > 0.0);
    }
}
