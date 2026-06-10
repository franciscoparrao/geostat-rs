//! Experimental (empirical) variogram estimation.

use std::f64::consts::{FRAC_PI_2, PI};

use rayon::prelude::*;

use crate::data::PointSet;
use crate::error::{GeostatError, Result};

/// Directional tolerance for anisotropic variograms.
///
/// Azimuth follows the gstat/GSLIB convention: degrees clockwise from north
/// (0 = N, 90 = E). Pairs are accepted when the angle between the pair vector
/// and the azimuth is within `tolerance_deg` (modulo 180°). A tolerance of
/// 90° is equivalent to an omnidirectional variogram.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionConfig {
    /// Azimuth in degrees clockwise from north.
    pub azimuth_deg: f64,
    /// Half-aperture tolerance in degrees.
    pub tolerance_deg: f64,
}

/// Configuration for the experimental variogram.
#[derive(Debug, Clone, PartialEq)]
pub struct VariogramConfig {
    /// Number of lag bins.
    pub n_lags: usize,
    /// Maximum pair distance considered; lag width is `max_dist / n_lags`.
    pub max_dist: f64,
    /// Optional directional restriction (anisotropic variogram).
    pub direction: Option<DirectionConfig>,
}

/// One lag bin of the experimental variogram.
#[derive(Debug, Clone, Copy)]
pub struct LagBin {
    /// Mean pair distance within the bin (bin midpoint if empty).
    pub h: f64,
    /// Semivariance estimate (NaN if the bin is empty).
    pub gamma: f64,
    /// Number of point pairs in the bin.
    pub n_pairs: usize,
}

/// Experimental variogram: a sequence of lag bins.
#[derive(Debug, Clone)]
pub struct ExperimentalVariogram {
    /// Lag bins, ordered by distance.
    pub bins: Vec<LagBin>,
    /// Maximum distance used to build the bins.
    pub max_dist: f64,
}

#[derive(Clone)]
struct Acc {
    sq: Vec<f64>,
    h: Vec<f64>,
    n: Vec<usize>,
}

impl Acc {
    fn new(n_lags: usize) -> Self {
        Self {
            sq: vec![0.0; n_lags],
            h: vec![0.0; n_lags],
            n: vec![0; n_lags],
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for b in 0..self.sq.len() {
            self.sq[b] += other.sq[b];
            self.h[b] += other.h[b];
            self.n[b] += other.n[b];
        }
        self
    }
}

/// Computes the experimental semivariogram
/// `gamma(h) = mean(0.5 * (z_i - z_j)^2)` over distance-binned point pairs.
pub fn experimental_variogram(
    data: &PointSet,
    cfg: &VariogramConfig,
) -> Result<ExperimentalVariogram> {
    if data.len() < 2 {
        return Err(GeostatError::InsufficientData(
            "experimental variogram requires at least 2 points".into(),
        ));
    }
    if cfg.n_lags == 0 {
        return Err(GeostatError::InvalidParameter(
            "n_lags must be at least 1".into(),
        ));
    }
    if !(cfg.max_dist > 0.0) || !cfg.max_dist.is_finite() {
        return Err(GeostatError::InvalidParameter(format!(
            "max_dist must be finite and positive, got {}",
            cfg.max_dist
        )));
    }
    if let Some(d) = &cfg.direction
        && (!(d.tolerance_deg > 0.0) || d.tolerance_deg > 90.0)
    {
        return Err(GeostatError::InvalidParameter(format!(
            "direction tolerance must be in (0, 90] degrees, got {}",
            d.tolerance_deg
        )));
    }

    let n = data.len();
    let n_lags = cfg.n_lags;
    let width = cfg.max_dist / n_lags as f64;
    let dir = cfg
        .direction
        .as_ref()
        .map(|d| (d.azimuth_deg.to_radians(), d.tolerance_deg.to_radians()));
    let coords = data.coords();
    let values = data.values();

    let acc = (0..n)
        .into_par_iter()
        .fold(
            || Acc::new(n_lags),
            |mut acc, i| {
                for j in (i + 1)..n {
                    let dx = coords[j][0] - coords[i][0];
                    let dy = coords[j][1] - coords[i][1];
                    let d = (dx * dx + dy * dy).sqrt();
                    if d <= 0.0 || d > cfg.max_dist {
                        continue;
                    }
                    if let Some((az, tol)) = dir {
                        // Azimuth of the pair vector, clockwise from north.
                        let ang = dx.atan2(dy);
                        let mut diff = (ang - az).rem_euclid(PI);
                        if diff > FRAC_PI_2 {
                            diff -= PI;
                        }
                        if diff.abs() > tol {
                            continue;
                        }
                    }
                    let bin = ((d / width) as usize).min(n_lags - 1);
                    let dz = values[i] - values[j];
                    acc.sq[bin] += 0.5 * dz * dz;
                    acc.h[bin] += d;
                    acc.n[bin] += 1;
                }
                acc
            },
        )
        .reduce(|| Acc::new(n_lags), Acc::merge);

    let bins = (0..n_lags)
        .map(|b| {
            if acc.n[b] > 0 {
                let np = acc.n[b] as f64;
                LagBin {
                    h: acc.h[b] / np,
                    gamma: acc.sq[b] / np,
                    n_pairs: acc.n[b],
                }
            } else {
                LagBin {
                    h: (b as f64 + 0.5) * width,
                    gamma: f64::NAN,
                    n_pairs: 0,
                }
            }
        })
        .collect();

    Ok(ExperimentalVariogram {
        bins,
        max_dist: cfg.max_dist,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_data() -> PointSet {
        // Three collinear points with values 0, 1, 2.
        PointSet::new(
            vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            vec![0.0, 1.0, 2.0],
        )
        .unwrap()
    }

    #[test]
    fn hand_computed_bins() {
        let cfg = VariogramConfig {
            n_lags: 4,
            max_dist: 2.0,
            direction: None,
        };
        let ev = experimental_variogram(&line_data(), &cfg).unwrap();
        // Pairs: d=1 (x2, gamma 0.5 each) -> bin 2; d=2 (gamma 2.0) -> bin 3.
        assert_eq!(ev.bins[0].n_pairs, 0);
        assert_eq!(ev.bins[1].n_pairs, 0);
        assert_eq!(ev.bins[2].n_pairs, 2);
        assert!((ev.bins[2].gamma - 0.5).abs() < 1e-12);
        assert!((ev.bins[2].h - 1.0).abs() < 1e-12);
        assert_eq!(ev.bins[3].n_pairs, 1);
        assert!((ev.bins[3].gamma - 2.0).abs() < 1e-12);
    }

    #[test]
    fn directional_filtering() {
        // Square: pairs along E-W, N-S and diagonals.
        let data = PointSet::new(
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            vec![0.0, 1.0, 2.0, 3.0],
        )
        .unwrap();
        // East-West direction (azimuth 90), tight tolerance: only the two
        // horizontal pairs qualify.
        let cfg = VariogramConfig {
            n_lags: 1,
            max_dist: 1.01,
            direction: Some(DirectionConfig {
                azimuth_deg: 90.0,
                tolerance_deg: 10.0,
            }),
        };
        let ev = experimental_variogram(&data, &cfg).unwrap();
        assert_eq!(ev.bins[0].n_pairs, 2);
        // gamma = mean(0.5*1, 0.5*1) = 0.5
        assert!((ev.bins[0].gamma - 0.5).abs() < 1e-12);

        // North-South: pairs (0,2) and (1,3), dz = 2 both.
        let cfg = VariogramConfig {
            n_lags: 1,
            max_dist: 1.01,
            direction: Some(DirectionConfig {
                azimuth_deg: 0.0,
                tolerance_deg: 10.0,
            }),
        };
        let ev = experimental_variogram(&data, &cfg).unwrap();
        assert_eq!(ev.bins[0].n_pairs, 2);
        assert!((ev.bins[0].gamma - 2.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_config() {
        let d = line_data();
        assert!(
            experimental_variogram(
                &d,
                &VariogramConfig {
                    n_lags: 0,
                    max_dist: 1.0,
                    direction: None
                }
            )
            .is_err()
        );
        assert!(
            experimental_variogram(
                &d,
                &VariogramConfig {
                    n_lags: 5,
                    max_dist: -1.0,
                    direction: None
                }
            )
            .is_err()
        );
    }
}
