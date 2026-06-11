//! Experimental cross-variograms between collocated variables.

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::variogram::experimental::{ExperimentalVariogram, VariogramConfig, pair_bins};

/// Computes the experimental cross-semivariogram
/// `gamma_ab(h) = mean(0.5 * (a_i - a_j)(b_i - b_j))` between two variables
/// sampled at exactly the same locations (full collocation required).
pub fn experimental_cross_variogram(
    a: &PointSet,
    b: &PointSet,
    cfg: &VariogramConfig,
) -> Result<ExperimentalVariogram> {
    if a.len() != b.len() {
        return Err(GeostatError::DimensionMismatch(format!(
            "{} vs {} points",
            a.len(),
            b.len()
        )));
    }
    if a.coords().iter().zip(b.coords()).any(|(ca, cb)| ca != cb) {
        return Err(GeostatError::InvalidParameter(
            "cross-variogram requires fully collocated datasets (same coordinates, same order)"
                .into(),
        ));
    }
    pair_bins(a.coords(), a.values(), b.values(), cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_of_variable_with_itself_is_direct() {
        let coords = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let a = PointSet::new(coords.clone(), vec![0.0, 1.0, 4.0, 2.0]).unwrap();
        let cfg = VariogramConfig {
            n_lags: 3,
            max_dist: 3.0,
            direction: None,
        };
        let direct = crate::variogram::experimental_variogram(&a, &cfg).unwrap();
        let cross = experimental_cross_variogram(&a, &a, &cfg).unwrap();
        for (d, c) in direct.bins.iter().zip(&cross.bins) {
            assert_eq!(d.n_pairs, c.n_pairs);
            if d.n_pairs > 0 {
                assert!((d.gamma - c.gamma).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn anticorrelated_variables_give_negative_gamma() {
        let coords = vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let a = PointSet::new(coords.clone(), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let b = PointSet::new(coords, vec![3.0, 2.0, 1.0, 0.0]).unwrap();
        let cfg = VariogramConfig {
            n_lags: 1,
            max_dist: 1.0,
            direction: None,
        };
        let cross = experimental_cross_variogram(&a, &b, &cfg).unwrap();
        assert!(cross.bins[0].gamma < 0.0);
    }

    #[test]
    fn rejects_non_collocated() {
        let a = PointSet::new(vec![[0.0, 0.0], [1.0, 0.0]], vec![1.0, 2.0]).unwrap();
        let b = PointSet::new(vec![[0.0, 0.0], [1.0, 0.1]], vec![1.0, 2.0]).unwrap();
        let cfg = VariogramConfig {
            n_lags: 2,
            max_dist: 2.0,
            direction: None,
        };
        assert!(experimental_cross_variogram(&a, &b, &cfg).is_err());
    }
}
