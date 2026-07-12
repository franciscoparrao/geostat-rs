//! Collocated cokriging under the Markov screening hypothesis (MM1/MM2).
//!
//! Full cokriging (see [`crate::cokriging`]) needs the secondary variable at
//! (or near) every primary datum, plus a fitted cross-variogram; that is
//! impractical when the secondary is exhaustively sampled (seismic, remote
//! sensing, a raster) and the primary is sparse — the dominant real-world
//! case cited against this engine before this module (AUDIT-2026-07.md
//! §3/§6 item #17). Collocated cokriging instead conditions each prediction
//! on the primary's own moving neighbourhood plus a *single* secondary
//! value collocated with the target, and derives the cross-covariance from
//! a Markov screening hypothesis (Journel 1999; Xu et al. 1992) instead of
//! fitting a cross-variogram directly:
//!
//! - **MM1**: `C12(h) = ρ12 (σ2/σ1) C1(h)` — the cross-covariance follows
//!   the *primary*'s spatial shape (needs only the primary's own variogram).
//! - **MM2**: `C12(h) = ρ12 (σ1/σ2) C2(h)` — follows the *secondary*'s shape
//!   (needs the secondary's own variogram).
//!
//! Both agree at `h = 0`: `C12(0) = ρ12 σ1 σ2` (the same correlation
//! coefficient and marginal standard deviations parametrize either model).
//!
//! This implements the **simple-kriging** form (known means): the ordinary
//! (unknown-mean) collocated system is well known to produce unstable/
//! negative weights (the single secondary equation makes the unbiasedness
//! constraint ill-posed relative to the primary neighbourhood), so GSLIB and
//! SGeMS both default to simple kriging for collocated cokriging, and this
//! engine does the same.

use ndarray::Array2;

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::kriging::KrigingEstimate;
use crate::linalg::solve;
use crate::search::KdTree;
use crate::variogram::VariogramModel;

/// How the cross-covariance `C12` is derived from the Markov screening
/// hypothesis (Journel 1999; see the module docs).
#[derive(Debug, Clone)]
pub enum MarkovModel {
    /// `C12(h) = ρ12 (σ2/σ1) C1(h)`: needs only the primary's variogram.
    Mm1,
    /// `C12(h) = ρ12 (σ1/σ2) C2(h)`: needs the secondary's own variogram.
    Mm2 {
        /// The secondary variable's variogram model.
        secondary_model: VariogramModel,
    },
}

/// Search-neighbourhood configuration for [`CollocatedCokriging`] (primary
/// data only; the secondary is always exactly the one value collocated with
/// the target).
///
/// `#[non_exhaustive]`: construct via `CollocatedConfig { ridge, ..
/// Default::default() }` (AUDIT-2026-07-v2.md §6 Fase 5).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[non_exhaustive]
pub struct CollocatedConfig {
    /// Maximum number of nearest primary conditioning points (`None` = all).
    pub max_neighbors: Option<usize>,
    /// Maximum search distance for primary conditioning points.
    pub search_radius: Option<f64>,
    /// Ridge added to the system diagonal (stabilizes near-singular
    /// systems; same role as [`crate::cokriging::CoKrigingConfig::ridge`]).
    pub ridge: f64,
}

/// Collocated cokriging predictor: a primary [`PointSet`] with its moving
/// neighbourhood, plus one secondary value collocated with each target (2-D
/// by default; `CollocatedCokriging<'_, 3>` for 3-D data).
#[derive(Debug)]
pub struct CollocatedCokriging<'a, const D: usize = 2> {
    primary: &'a PointSet<D>,
    model1: &'a VariogramModel,
    mean1: f64,
    mean2: f64,
    rho12: f64,
    sigma1: f64,
    sigma2: f64,
    markov: MarkovModel,
    config: CollocatedConfig,
    tree: Option<KdTree<D>>,
}

impl<'a, const D: usize> CollocatedCokriging<'a, D> {
    /// Builds a collocated cokriging predictor.
    ///
    /// `mean1`/`mean2` are the (known/population) means of the primary and
    /// secondary variables — for an exhaustive secondary (e.g. a raster)
    /// this is typically its areal mean, not just its value at the primary
    /// locations. `rho12` is the correlation coefficient between the two
    /// variables at zero separation (collocated pairs), and `sigma1`/
    /// `sigma2` their marginal standard deviations; see
    /// [`estimate_collocated_stats`] to compute all three from collocated
    /// sample pairs.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        primary: &'a PointSet<D>,
        model1: &'a VariogramModel,
        mean1: f64,
        mean2: f64,
        rho12: f64,
        sigma1: f64,
        sigma2: f64,
        markov: MarkovModel,
        config: CollocatedConfig,
    ) -> Result<Self> {
        if model1.has_power()
            || matches!(&markov, MarkovModel::Mm2 { secondary_model } if secondary_model.has_power())
        {
            return Err(GeostatError::InvalidParameter(
                "collocated co-kriging needs a valid covariance function and cannot use the \
                 unbounded Power model"
                    .into(),
            ));
        }
        let dim_offender = model1
            .invalid_structure_for_dim(D)
            .or_else(|| match &markov {
                MarkovModel::Mm2 { secondary_model } => {
                    secondary_model.invalid_structure_for_dim(D)
                }
                MarkovModel::Mm1 => None,
            });
        if let Some(kind) = dim_offender {
            return Err(GeostatError::InvalidParameter(format!(
                "{kind:?} is not a valid covariance in {D} dimensions; use Spherical instead for \
                 a 3-D-safe bounded structure"
            )));
        }
        if !rho12.is_finite() || !(-1.0..=1.0).contains(&rho12) {
            return Err(GeostatError::InvalidParameter(format!(
                "rho12 must be finite and in [-1, 1], got {rho12}"
            )));
        }
        if !(sigma1 > 0.0) || !(sigma2 > 0.0) {
            return Err(GeostatError::InvalidParameter(
                "sigma1/sigma2 must be finite and > 0".into(),
            ));
        }
        if !config.ridge.is_finite() || !(config.ridge >= 0.0) {
            return Err(GeostatError::InvalidParameter(
                "ridge must be finite and >= 0".into(),
            ));
        }
        if let Some((i, j)) = primary.duplicate_pair() {
            return Err(GeostatError::DuplicatePoints(i, j));
        }
        let tree = if config.max_neighbors.is_some() || config.search_radius.is_some() {
            Some(KdTree::build(primary.coords()))
        } else {
            None
        };
        Ok(Self {
            primary,
            model1,
            mean1,
            mean2,
            rho12,
            sigma1,
            sigma2,
            markov,
            config,
            tree,
        })
    }

    /// Cross-covariance `C12(dh)` under the configured Markov model:
    /// `rho12 * sigma1 * sigma2 * rho_k(dh)`, where `rho_k = Ck(dh)/Ck(0)`
    /// is the *correlogram* of whichever model (`k = 1` for MM1, `k = 2` for
    /// MM2) the Markov screening hypothesis follows.
    ///
    /// Using the correlogram (not the raw covariance) is what makes this
    /// consistent with the `C12(0) = rho12 * sigma1 * sigma2` identity the
    /// module docs state: `sigma1`/`sigma2` are caller-supplied marginal
    /// standard deviations (typically sample estimates from
    /// [`estimate_collocated_stats`]), which need not exactly equal
    /// `sqrt(model1.covariance_dh(0))` / `sqrt(secondary_model.covariance_dh(0))`
    /// -- fitting a variogram and estimating a sample variance are two
    /// different estimators of the same quantity. The raw-covariance formula
    /// used before this fix silently assumed they matched exactly
    /// (AUDIT-2026-07-v2.md §1.6): the predict system's data-secondary
    /// cross terms and its RHS `C12(0)` entry were built from two formulas
    /// that agreed only in that special case.
    fn c12(&self, dh: [f64; D]) -> f64 {
        match &self.markov {
            MarkovModel::Mm1 => {
                let c11_0 = self.model1.covariance_dh([0.0; D]);
                self.rho12 * self.sigma1 * self.sigma2 * (self.model1.covariance_dh(dh) / c11_0)
            }
            MarkovModel::Mm2 { secondary_model } => {
                let c22_0 = secondary_model.covariance_dh([0.0; D]);
                self.rho12 * self.sigma1 * self.sigma2 * (secondary_model.covariance_dh(dh) / c22_0)
            }
        }
    }

    fn neighbors(&self, target: [f64; D]) -> Vec<usize> {
        let Some(tree) = &self.tree else {
            return (0..self.primary.len()).collect();
        };
        tree.k_nearest(
            target,
            self.config.max_neighbors.unwrap_or(self.primary.len()),
            self.config.search_radius,
        )
    }

    /// Predicts the primary variable at `target`, given the secondary
    /// value collocated with it. With zero primary neighbours (e.g. an
    /// empty search radius) this degenerates gracefully to simple linear
    /// regression on the secondary alone.
    ///
    /// The primary block is built from `sigma1^2 * rho1(h)` (`rho1` being
    /// `model1`'s own correlogram), not `model1`'s raw covariance
    /// (AUDIT-2026-07-v3.md §1.4): `sigma1` is a caller-supplied marginal
    /// standard deviation (typically a sample estimate) that need not equal
    /// `sqrt(model1.covariance_dh(0))` — mixing the two scales (as the raw
    /// covariance did) makes the joint primary+secondary system PSD only by
    /// accident (`rho12^2 * sigma1^2 <= model1.covariance_dh(0)`), and
    /// violating it produces negative "variance" silently clamped to zero
    /// (false certainty from what is really a plain regression on the
    /// secondary). Standardizing throughout to sigma1^2/sigma2^2 matches
    /// the Markov-model construction in Journel (1999): the combined
    /// covariance is then PSD for any `|rho12| <= 1`, by the same argument
    /// that guarantees `c12`'s cross term is a valid covariance.
    pub fn predict(&self, target: [f64; D], secondary_at_target: f64) -> Result<KrigingEstimate> {
        let nb = self.neighbors(target);
        let n = nb.len();
        let dim = n + 1;
        let c11_0 = self.model1.covariance_dh([0.0; D]);
        let sigma1_sq = self.sigma1 * self.sigma1;
        let c22_0 = self.sigma2 * self.sigma2;
        let c12_0 = self.rho12 * self.sigma1 * self.sigma2;

        let coords = self.primary.coords();
        let mut a = Array2::<f64>::zeros((dim, dim));
        let mut b = vec![0.0; dim];
        for (ii, &i) in nb.iter().enumerate() {
            let pi = coords[i];
            a[[ii, ii]] = sigma1_sq + self.config.ridge;
            for (jj, &j) in nb.iter().enumerate().skip(ii + 1) {
                let c = sigma1_sq * (self.model1.covariance_dh(sep(pi, coords[j])) / c11_0);
                a[[ii, jj]] = c;
                a[[jj, ii]] = c;
            }
            let c1s = self.c12(sep(pi, target));
            a[[ii, n]] = c1s;
            a[[n, ii]] = c1s;
            b[ii] = sigma1_sq * (self.model1.covariance_dh(sep(pi, target)) / c11_0);
        }
        a[[n, n]] = c22_0 + self.config.ridge;
        b[n] = c12_0;

        let w = solve(a, b.clone())?;
        let values = self.primary.values();
        let mut value = self.mean1 + w[n] * (secondary_at_target - self.mean2);
        let mut reduction = w[n] * c12_0;
        for (ii, &i) in nb.iter().enumerate() {
            value += w[ii] * (values[i] - self.mean1);
            reduction += w[ii] * b[ii];
        }
        let variance = (sigma1_sq - reduction).max(0.0);
        Ok(KrigingEstimate {
            value,
            variance,
            lagrange: None,
        })
    }

    /// Predictions at many targets, in parallel. `secondary[i]` is the
    /// secondary value collocated with `targets[i]`. A target that fails
    /// (e.g. a near-singular local system) gets a NaN estimate rather than
    /// aborting the whole batch, matching [`crate::kriging::Kriging::predict_many`].
    pub fn predict_many(
        &self,
        targets: &[[f64; D]],
        secondary: &[f64],
    ) -> Result<Vec<KrigingEstimate>> {
        if targets.len() != secondary.len() {
            return Err(GeostatError::DimensionMismatch(format!(
                "{} targets vs {} secondary values",
                targets.len(),
                secondary.len()
            )));
        }
        Ok(crate::parallel::par_map(targets.len(), |i| {
            self.predict(targets[i], secondary[i])
                .unwrap_or(NAN_ESTIMATE)
        }))
    }
}

const NAN_ESTIMATE: KrigingEstimate = KrigingEstimate {
    value: f64::NAN,
    variance: f64::NAN,
    lagrange: None,
};

fn sep<const D: usize>(a: [f64; D], b: [f64; D]) -> [f64; D] {
    let mut dh = [0.0; D];
    for d in 0..D {
        dh[d] = a[d] - b[d];
    }
    dh
}

/// Estimates `(rho12, sigma1, sigma2)` from collocated sample pairs
/// `(primary[i], secondary[i])` — Pearson correlation and sample standard
/// deviations, the usual inputs to [`CollocatedCokriging::new`] when no
/// external estimate of the population statistics is available.
pub fn estimate_collocated_stats(primary: &[f64], secondary: &[f64]) -> Result<(f64, f64, f64)> {
    if primary.len() != secondary.len() {
        return Err(GeostatError::DimensionMismatch(format!(
            "{} primary values vs {} secondary values",
            primary.len(),
            secondary.len()
        )));
    }
    let n = primary.len();
    if n < 2 {
        return Err(GeostatError::InsufficientData(
            "at least 2 collocated pairs are needed to estimate correlation".into(),
        ));
    }
    let n_f = n as f64;
    let m1 = primary.iter().sum::<f64>() / n_f;
    let m2 = secondary.iter().sum::<f64>() / n_f;
    let (mut s11, mut s22, mut s12) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let d1 = primary[i] - m1;
        let d2 = secondary[i] - m2;
        s11 += d1 * d1;
        s22 += d2 * d2;
        s12 += d1 * d2;
    }
    let sigma1 = (s11 / n_f).sqrt();
    let sigma2 = (s22 / n_f).sqrt();
    if !(sigma1 > 0.0) || !(sigma2 > 0.0) {
        return Err(GeostatError::InvalidParameter(
            "primary/secondary values have zero variance".into(),
        ));
    }
    let rho12 = ((s12 / n_f) / (sigma1 * sigma2)).clamp(-1.0, 1.0);
    Ok((rho12, sigma1, sigma2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::variogram::{ModelKind, Structure};

    fn primary() -> PointSet {
        PointSet::new(
            vec![
                [0.0, 0.0],
                [10.0, 0.0],
                [0.0, 10.0],
                [10.0, 10.0],
                [5.0, 5.0],
            ],
            vec![1.0, 2.0, 1.5, 2.5, 1.8],
        )
        .unwrap()
    }

    fn model1() -> VariogramModel {
        VariogramModel::new(0.0, vec![Structure::new(ModelKind::Spherical, 1.0, 20.0)]).unwrap()
    }

    #[test]
    fn zero_correlation_reduces_to_simple_kriging() {
        use crate::kriging::{Kriging, KrigingConfig, KrigingMethod};
        let data = primary();
        let m = model1();
        let sk = Kriging::new(
            &data,
            &m,
            KrigingConfig {
                method: KrigingMethod::Simple { mean: data.mean() },
                ..Default::default()
            },
        )
        .unwrap();

        let cck = CollocatedCokriging::new(
            &data,
            &m,
            data.mean(),
            0.0,
            0.0, // rho12 = 0: secondary must contribute nothing
            1.0,
            3.0,
            MarkovModel::Mm1,
            CollocatedConfig::default(),
        )
        .unwrap();

        for &(target, sec) in &[([3.0, 4.0], 99.0), ([7.0, 2.0], -50.0), ([5.0, 5.0], 12.3)] {
            let exact = sk.predict(target).unwrap();
            let est = cck.predict(target, sec).unwrap();
            assert!(
                (est.value - exact.value).abs() < 1e-10,
                "value {} vs {}",
                est.value,
                exact.value
            );
            assert!(
                (est.variance - exact.variance).abs() < 1e-10,
                "variance {} vs {}",
                est.variance,
                exact.variance
            );
        }
    }

    #[test]
    fn predict_many_matches_looped_predict() {
        let data = primary();
        let m = model1();
        let cck = CollocatedCokriging::new(
            &data,
            &m,
            data.mean(),
            1.0,
            0.6,
            1.0,
            2.0,
            MarkovModel::Mm1,
            CollocatedConfig::default(),
        )
        .unwrap();

        let targets = [[3.0, 4.0], [7.0, 2.0], [5.0, 5.0], [-1.0, -1.0]];
        let secondary = [10.0, -3.0, 4.2, 0.0];
        let batch = cck.predict_many(&targets, &secondary).unwrap();
        assert_eq!(batch.len(), targets.len());
        for (i, &t) in targets.iter().enumerate() {
            let single = cck.predict(t, secondary[i]).unwrap();
            assert!((batch[i].value - single.value).abs() < 1e-12);
            assert!((batch[i].variance - single.variance).abs() < 1e-12);
        }

        let err = cck.predict_many(&targets, &secondary[..2]).unwrap_err();
        assert!(matches!(err, GeostatError::DimensionMismatch(_)));
    }

    #[test]
    fn predict_is_exact_at_a_primary_datum() {
        // A primary datum in its own neighbourhood (separation 0) pins the
        // prediction exactly, the usual kriging-exactness property --
        // checked here with a non-degenerate rho12/sigma pair (see
        // `mm1_perfectly_collinear_secondary_can_be_singular` for why
        // rho12=1 with sigma1==sigma2 is a special case that breaks this).
        let data = primary();
        let m = model1();
        let cck = CollocatedCokriging::new(
            &data,
            &m,
            data.mean(),
            data.mean(),
            0.6,
            1.0,
            3.0,
            MarkovModel::Mm1,
            CollocatedConfig::default(),
        )
        .unwrap();
        let est = cck.predict([0.0, 0.0], 42.0).unwrap();
        assert!((est.value - 1.0).abs() < 1e-6, "value {}", est.value);
        assert!(est.variance < 1e-6, "variance {}", est.variance);
    }

    #[test]
    fn mm1_perfectly_collinear_secondary_errors_instead_of_guessing() {
        // Documented edge case: rho12=1 with sigma1==sigma2==sqrt(C1(0))
        // makes MM1's C12(h) *identical* to C1(h) -- the secondary becomes
        // an exact informational duplicate of the primary, so the row for a
        // primary neighbour at zero separation and the (linearly dependent)
        // row for the collocated secondary make the system exactly
        // singular. Before AUDIT-2026-07-v2.md §1.6's fix, `c12` used the
        // raw covariance (`rho12 * (sigma2/sigma1) * C1(h)`), which reached
        // this exact-duplicate condition for *any* sigma1==sigma2 --
        // including sigma values that did not match the model's own
        // C1(0), an internal inconsistency described in the audit. That
        // bug happened to make partial-pivoting LU land on a near-zero (but
        // nonzero) pivot, returning an "unexpected but not erroring"
        // solution. The corrected, internally-consistent correlogram
        // formula (`rho12 * sigma1 * sigma2 * (C1(h)/C1(0))`) reaches the
        // true exact-duplicate condition only when sigma matches the
        // model's C1(0), and there the pivot is exactly zero: the fixed
        // code reports `SingularSystem` explicitly instead of silently
        // returning an arbitrary answer -- an improvement over the old
        // "unexpected but not erroring" behavior, not a regression.
        // `rho12` is a *sample* correlation in practice and is essentially
        // never exactly `1.0`, so this is not expected to bite real usage.
        let data = primary();
        let m = model1();
        let sigma = m.covariance_dh([0.0, 0.0]).sqrt();
        let cck = CollocatedCokriging::new(
            &data,
            &m,
            data.mean(),
            data.mean(),
            1.0,
            sigma,
            sigma,
            MarkovModel::Mm1,
            CollocatedConfig::default(),
        )
        .unwrap();
        let err = cck.predict([0.0, 0.0], 42.0).unwrap_err();
        assert!(
            matches!(err, GeostatError::SingularSystem(_)),
            "expected a clean singular-system error, got {err:?}"
        );
    }

    #[test]
    fn c12_matches_rho_sigma1_sigma2_at_zero_lag_even_when_sigma_mismatches_the_model_sill() {
        // AUDIT-2026-07-v2.md §1.6: the module docs (and `predict`'s use of
        // `c12_0 = rho12 * sigma1 * sigma2` as the system's RHS) require
        // `C12(0) == rho12 * sigma1 * sigma2` exactly. Before the fix this
        // only held when sigma1^2 happened to equal `model1.covariance_dh(0)`
        // (MM1) / `secondary_model.covariance_dh(0)` (MM2); this test
        // deliberately picks a sigma that does *not* match either model's
        // sill, which would have broken the identity under the old
        // raw-covariance formula.
        let data = primary();
        let m1 = model1(); // sill = 1.0
        let rho12 = 0.4;
        let (sigma1, sigma2) = (3.0, 7.0); // neither is sqrt(1.0)
        let expected = rho12 * sigma1 * sigma2;

        let mm1 = CollocatedCokriging::new(
            &data,
            &m1,
            0.0,
            0.0,
            rho12,
            sigma1,
            sigma2,
            MarkovModel::Mm1,
            CollocatedConfig::default(),
        )
        .unwrap();
        assert!((mm1.c12([0.0, 0.0]) - expected).abs() < 1e-10);

        let secondary_model =
            VariogramModel::new(0.0, vec![Structure::new(ModelKind::Exponential, 5.0, 15.0)])
                .unwrap(); // sill = 5.0, also mismatched
        let mm2 = CollocatedCokriging::new(
            &data,
            &m1,
            0.0,
            0.0,
            rho12,
            sigma1,
            sigma2,
            MarkovModel::Mm2 { secondary_model },
            CollocatedConfig::default(),
        )
        .unwrap();
        assert!((mm2.c12([0.0, 0.0]) - expected).abs() < 1e-10);
    }

    #[test]
    fn mm1_and_mm2_agree_when_secondary_covariance_is_proportional_to_primary() {
        // If C2(h) = k * C1(h) for some k > 0, MM1 and MM2 are mathematically
        // the same model when sigma2^2 = k * sigma1^2 (both reduce to the
        // identical cross-covariance function): a strong internal-consistency
        // check on the two code paths.
        let data = primary();
        let m1 = model1();
        let k = 4.0; // sigma2 = 2 * sigma1
        let m2 =
            VariogramModel::new(0.0, vec![Structure::new(ModelKind::Spherical, k, 20.0)]).unwrap();
        let sigma1 = 1.0;
        let sigma2 = (k * sigma1 * sigma1).sqrt();
        let rho12 = 0.7;

        let cck_mm1 = CollocatedCokriging::new(
            &data,
            &m1,
            data.mean(),
            5.0,
            rho12,
            sigma1,
            sigma2,
            MarkovModel::Mm1,
            CollocatedConfig::default(),
        )
        .unwrap();
        let cck_mm2 = CollocatedCokriging::new(
            &data,
            &m1,
            data.mean(),
            5.0,
            rho12,
            sigma1,
            sigma2,
            MarkovModel::Mm2 {
                secondary_model: m2,
            },
            CollocatedConfig::default(),
        )
        .unwrap();

        for &(target, sec) in &[([3.0, 4.0], 8.0), ([7.0, 2.0], 2.0), ([-2.0, 6.0], 5.5)] {
            let e1 = cck_mm1.predict(target, sec).unwrap();
            let e2 = cck_mm2.predict(target, sec).unwrap();
            assert!(
                (e1.value - e2.value).abs() < 1e-9,
                "value MM1 {} vs MM2 {}",
                e1.value,
                e2.value
            );
            assert!(
                (e1.variance - e2.variance).abs() < 1e-9,
                "variance MM1 {} vs MM2 {}",
                e1.variance,
                e2.variance
            );
        }
    }

    #[test]
    fn stronger_correlation_reduces_variance_more() {
        let data = primary();
        let m = model1();
        let target = [3.0, 7.0];
        let mut prev_var = f64::INFINITY;
        for &rho in &[0.0, 0.3, 0.6, 0.9] {
            let cck = CollocatedCokriging::new(
                &data,
                &m,
                data.mean(),
                0.0,
                rho,
                1.0,
                1.0,
                MarkovModel::Mm1,
                CollocatedConfig::default(),
            )
            .unwrap();
            let est = cck.predict(target, 1.0).unwrap();
            assert!(
                est.variance <= prev_var + 1e-9,
                "rho={rho}: variance {} should not exceed {prev_var}",
                est.variance
            );
            prev_var = est.variance;
        }
    }

    #[test]
    fn estimate_collocated_stats_recovers_known_correlation() {
        let mut rng = Rng::new(11);
        let n = 500;
        let mut primary = Vec::with_capacity(n);
        let mut secondary = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.normal();
            let noise = rng.normal();
            primary.push(2.0 + 3.0 * x);
            // rho(primary, secondary) = 0.8 by construction (mixing weights
            // 0.8/0.6 on standardized components).
            secondary.push(10.0 - 1.5 * (0.8 * x + 0.6 * noise));
        }
        let (rho12, sigma1, sigma2) = estimate_collocated_stats(&primary, &secondary).unwrap();
        assert!((rho12 - (-0.8)).abs() < 0.05, "rho12 {rho12}");
        assert!((sigma1 - 3.0).abs() < 0.3, "sigma1 {sigma1}");
        assert!((sigma2 - 1.5).abs() < 0.15, "sigma2 {sigma2}");
    }

    #[test]
    fn predict_variance_matches_conditional_gaussian_formula_with_no_primary_neighbours() {
        // AUDIT-2026-07-v3.md §1.4: with the primary neighbourhood empty,
        // `predict` degenerates to plain regression on the secondary, whose
        // exact conditional variance is `sigma1^2 * (1 - rho12^2)`
        // (bivariate normal conditioning) -- always non-negative for
        // `|rho12| <= 1`. Before standardizing the primary block to
        // `sigma1^2` (instead of `model1`'s own, generally different,
        // `C1(0)`), this could be deeply negative (the audit found -6.29
        // with these exact rho12=0.9/sigma1=3/sigma2=7 values, sill=1.0)
        // and got clamped to a false-certainty 0.
        let data = primary();
        let m = model1(); // sill = 1.0
        for &(rho12, sigma1, sigma2) in &[(0.9, 3.0, 7.0), (0.4, 3.0, 7.0), (-0.7, 5.0, 2.0)] {
            let cck = CollocatedCokriging::new(
                &data,
                &m,
                data.mean(),
                0.0,
                rho12,
                sigma1,
                sigma2,
                MarkovModel::Mm1,
                CollocatedConfig {
                    max_neighbors: Some(0),
                    ..Default::default()
                },
            )
            .unwrap();
            let est = cck.predict([3.0, 4.0], 1.0).unwrap();
            let expected = sigma1 * sigma1 * (1.0 - rho12 * rho12);
            assert!(
                (est.variance - expected).abs() < 1e-9,
                "rho12={rho12} sigma1={sigma1}: variance {} vs expected {expected}",
                est.variance
            );
        }
    }

    #[test]
    fn predict_variance_stays_bounded_with_mismatched_sigma_and_primary_neighbours() {
        // Same mismatch (sigma1 != sqrt(model1's C1(0))) but with a real
        // primary neighbourhood: the standardized system must still keep
        // 0 <= variance <= sigma1^2 for any valid rho12, not just the
        // empty-neighbourhood closed-form case above.
        let data = primary();
        let m = model1();
        for &(rho12, sigma1, sigma2) in &[(0.9, 3.0, 7.0), (-0.6, 4.0, 0.5)] {
            let cck = CollocatedCokriging::new(
                &data,
                &m,
                data.mean(),
                0.0,
                rho12,
                sigma1,
                sigma2,
                MarkovModel::Mm1,
                CollocatedConfig::default(),
            )
            .unwrap();
            for &target in &[[3.0, 4.0], [7.0, 2.0], [-1.0, -1.0]] {
                let est = cck.predict(target, 1.0).unwrap();
                assert!(
                    (0.0..=sigma1 * sigma1 + 1e-9).contains(&est.variance),
                    "rho12={rho12} sigma1={sigma1} target={target:?}: variance {} out of [0, {}]",
                    est.variance,
                    sigma1 * sigma1
                );
            }
        }
    }

    #[test]
    fn rejects_bad_params() {
        let data = primary();
        let m = model1();
        assert!(
            CollocatedCokriging::new(
                &data,
                &m,
                0.0,
                0.0,
                1.5,
                1.0,
                1.0,
                MarkovModel::Mm1,
                CollocatedConfig::default()
            )
            .is_err(),
            "rho12 > 1 must be rejected"
        );
        assert!(
            CollocatedCokriging::new(
                &data,
                &m,
                0.0,
                0.0,
                0.5,
                0.0,
                1.0,
                MarkovModel::Mm1,
                CollocatedConfig::default()
            )
            .is_err(),
            "sigma1 <= 0 must be rejected"
        );
    }

    #[test]
    fn rejects_power_model() {
        let data = primary();
        let power_model =
            VariogramModel::new(0.0, vec![Structure::new(ModelKind::Power(1.0), 1.0, 1.0)])
                .unwrap();
        assert!(
            CollocatedCokriging::new(
                &data,
                &power_model,
                0.0,
                0.0,
                0.5,
                1.0,
                1.0,
                MarkovModel::Mm1,
                CollocatedConfig::default()
            )
            .is_err()
        );
        let bounded = model1();
        assert!(
            CollocatedCokriging::new(
                &data,
                &bounded,
                0.0,
                0.0,
                0.5,
                1.0,
                1.0,
                MarkovModel::Mm2 {
                    secondary_model: power_model
                },
                CollocatedConfig::default()
            )
            .is_err()
        );
    }
}
