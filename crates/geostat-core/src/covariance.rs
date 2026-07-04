//! Public abstraction over "a covariance/semivariogram model" — the seam
//! for plugging a custom, non-catalog covariance function into
//! [`crate::kriging::Kriging`] without going through [`VariogramModel`]'s
//! fixed set of [`crate::variogram::ModelKind`]s.

use crate::variogram::VariogramModel;

/// A covariance/semivariogram model usable by [`crate::kriging::Kriging`].
///
/// Implement this for any type that can answer "how are two points at
/// separation `dh` spatially related", and the built-in kriging machinery
/// (ordinary/universal/external-drift; simple kriging too, as long as
/// [`Covariance::covariance_dh`] is a genuine covariance function) works
/// with it exactly as it does with [`VariogramModel`] — `Kriging<'_, D>`
/// defaults its model type parameter to `VariogramModel`, so existing code
/// is unaffected; this trait only matters when you want to krige against
/// something else.
///
/// The default-method surface intentionally mirrors what a *smooth*
/// covariance looks like (no nugget discontinuity, a genuine finite
/// covariance): [`Covariance::nugget`] and [`Covariance::has_power`] only
/// need overriding by implementations that actually have a nugget or are
/// unbounded (like [`VariogramModel`], which overrides both).
/// `Send + Sync` supertraits: kriging parallelizes prediction across
/// targets internally, so any covariance model needs to be safely shared
/// across threads.
pub trait Covariance<const D: usize = 2>: Send + Sync {
    /// Semivariance for separation vector `dh`. By convention
    /// `gamma_dh([0.0; D]) == 0.0` (kriging call sites rely on this and skip
    /// evaluating it at zero separation).
    fn gamma_dh(&self, dh: [f64; D]) -> f64;

    /// Covariance for separation vector `dh`, if one exists — see
    /// [`Covariance::has_power`] for models that have none. Should be
    /// consistent with [`Covariance::gamma_dh`] under second-order
    /// stationarity (`covariance_dh(dh) = covariance_dh([0.0; D]) -
    /// gamma_dh(dh)` is the usual relationship, as in [`VariogramModel`]).
    fn covariance_dh(&self, dh: [f64; D]) -> f64;

    /// Nugget effect: the discontinuity in [`Covariance::covariance_dh`] at
    /// exactly zero separation, i.e. `covariance_dh([0;D]) -
    /// lim_{dh -> 0, dh != 0} covariance_dh(dh)`. Used only by block
    /// kriging's within-block covariance (a nugget is a measure-zero
    /// discontinuity in the block-averaging integral, so it must be
    /// excluded from the continuous part). Defaults to `0.0` — correct for
    /// any genuinely continuous covariance, i.e. most custom
    /// implementations; [`VariogramModel`] overrides this with its actual
    /// nugget.
    fn nugget(&self) -> f64 {
        0.0
    }

    /// `true` if this model has no covariance function at all (infinite
    /// variance — e.g. [`crate::variogram::ModelKind::Power`]), meaning
    /// [`Covariance::covariance_dh`] is meaningless and only
    /// ordinary/universal/external-drift kriging (never simple kriging,
    /// block kriging, or anything else that needs a real covariance) can
    /// use it. Defaults to `false`; [`VariogramModel`] overrides this via
    /// [`VariogramModel::has_power`].
    fn has_power(&self) -> bool {
        false
    }

    /// `true` if this covariance is mathematically valid in `D` dimensions
    /// (e.g. positive-definite). Defaults to `true` — correct for most
    /// hand-rolled covariances, which typically aren't tied to a dimension;
    /// [`VariogramModel`] overrides this for kinds like
    /// [`crate::variogram::ModelKind::Circular`] that are only valid in
    /// `D <= 2` (see [`VariogramModel::invalid_structure_for_dim`]).
    fn is_valid_dim(&self) -> bool {
        true
    }
}

impl<const D: usize> Covariance<D> for VariogramModel {
    fn gamma_dh(&self, dh: [f64; D]) -> f64 {
        VariogramModel::gamma_dh(self, dh)
    }

    fn covariance_dh(&self, dh: [f64; D]) -> f64 {
        VariogramModel::covariance_dh(self, dh)
    }

    fn nugget(&self) -> f64 {
        self.nugget
    }

    fn has_power(&self) -> bool {
        VariogramModel::has_power(self)
    }

    fn is_valid_dim(&self) -> bool {
        self.invalid_structure_for_dim(D).is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::PointSet;
    use crate::kriging::{Kriging, KrigingConfig};
    use crate::variogram::{ModelKind, Structure};

    /// A hand-rolled exponential covariance, entirely independent of
    /// [`VariogramModel`]/[`ModelKind`] — the point of this test is that
    /// [`Kriging`] works with *any* [`Covariance`] implementation, not just
    /// the built-in catalog.
    struct HandRolledExponential {
        nugget: f64,
        sill: f64,
        range: f64,
    }

    impl<const D: usize> Covariance<D> for HandRolledExponential {
        fn gamma_dh(&self, dh: [f64; D]) -> f64 {
            let h: f64 = dh.iter().map(|d| d * d).sum::<f64>().sqrt();
            if h == 0.0 {
                0.0
            } else {
                self.nugget + self.sill * (1.0 - (-h / self.range).exp())
            }
        }

        fn covariance_dh(&self, dh: [f64; D]) -> f64 {
            (self.nugget + self.sill) - Covariance::<D>::gamma_dh(self, dh)
        }

        fn nugget(&self) -> f64 {
            self.nugget
        }
    }

    fn sample_data() -> PointSet {
        PointSet::new(
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.4, 0.6]],
            vec![1.0, 2.0, 1.5, 2.5, 1.7],
        )
        .unwrap()
    }

    #[test]
    fn custom_covariance_matches_the_equivalent_variogram_model_exactly() {
        let data = sample_data();
        let custom = HandRolledExponential {
            nugget: 0.05,
            sill: 0.95,
            range: 2.0,
        };
        let equivalent = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 0.95, 2.0)],
        )
        .unwrap();

        let cfg = KrigingConfig::default();
        let via_trait = Kriging::new(&data, &custom, cfg.clone()).unwrap();
        let via_concrete = Kriging::new(&data, &equivalent, cfg).unwrap();

        let targets = [[0.3, 0.3], [0.9, 0.1], [2.0, 2.0]];
        for t in targets {
            let a = via_trait.predict(t).unwrap();
            let b = via_concrete.predict(t).unwrap();
            assert!((a.value - b.value).abs() < 1e-12, "{} vs {}", a.value, b.value);
            assert!(
                (a.variance - b.variance).abs() < 1e-12,
                "{} vs {}",
                a.variance,
                b.variance
            );
        }
    }

    #[test]
    fn custom_covariance_is_exact_at_data_and_has_positive_variance_elsewhere() {
        let data = sample_data();
        let custom = HandRolledExponential {
            nugget: 0.0,
            sill: 1.0,
            range: 1.5,
        };
        let k = Kriging::new(&data, &custom, KrigingConfig::default()).unwrap();
        for i in 0..data.len() {
            let est = k.predict(data.coord(i)).unwrap();
            assert!((est.value - data.value(i)).abs() < 1e-8);
            assert!(est.variance < 1e-8);
        }
        let far = k.predict([50.0, 50.0]).unwrap();
        assert!(far.variance > 0.0);
    }

    #[test]
    fn variogram_model_covariance_impl_matches_its_inherent_methods() {
        let m = VariogramModel::new(0.1, vec![Structure::new(ModelKind::Spherical, 0.9, 10.0)])
            .unwrap();
        for h in [0.0, 1.0, 5.0, 10.0, 20.0] {
            let dh = [h, 0.0];
            assert_eq!(Covariance::<2>::gamma_dh(&m, dh), m.gamma_dh(dh));
            assert_eq!(Covariance::<2>::covariance_dh(&m, dh), m.covariance_dh(dh));
        }
        assert_eq!(Covariance::<2>::nugget(&m), m.nugget);
        assert_eq!(Covariance::<2>::has_power(&m), m.has_power());
    }
}
