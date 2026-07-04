//! Experimental (empirical) variogram estimation.

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::parallel::{n_chunks, par_map};

/// Directional tolerance for anisotropic variograms.
///
/// Azimuth follows the gstat/GSLIB convention: degrees clockwise from north
/// (0 = N, 90 = E). In 3-D, `dip_deg` tilts the direction vector using the
/// same sign convention as [`super::Anisotropy::dip_deg`] (GSLIB `ang2` /
/// gstat, verified against `rotation_matrix_3d` in
/// `direction_matches_model_major_axis` below — a positive `dip_deg` here
/// picks out the same 3-D direction as a fitted model with the same
/// `azimuth_deg`/`dip_deg`, so a variogram computed along `DirectionConfig`
/// and fit to a model built from the same angles are mutually consistent).
/// Pairs are accepted within a cone of half-aperture `tolerance_deg` around
/// the (sign-agnostic) direction. A tolerance of 90° is equivalent to an
/// omnidirectional variogram.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionConfig {
    /// Azimuth in degrees clockwise from north.
    pub azimuth_deg: f64,
    /// Dip in degrees (3-D only; 0 in 2-D). Same sign convention as
    /// [`super::Anisotropy::dip_deg`] — see the struct docs.
    pub dip_deg: f64,
    /// Half-aperture tolerance in degrees.
    pub tolerance_deg: f64,
}

impl DirectionConfig {
    /// Horizontal direction (zero dip).
    pub fn horizontal(azimuth_deg: f64, tolerance_deg: f64) -> Self {
        Self {
            azimuth_deg,
            dip_deg: 0.0,
            tolerance_deg,
        }
    }
}

/// Unit direction vector for an azimuth/dip pair, in the same sign
/// convention as [`super::Anisotropy::rotation_matrix_3d`]'s major-axis row
/// (see `direction_matches_model_major_axis`). `D = 2` ignores `dip_deg`.
pub(crate) fn direction_unit_vector<const D: usize>(azimuth_deg: f64, dip_deg: f64) -> [f64; D] {
    let az = azimuth_deg.to_radians();
    let dip = dip_deg.to_radians();
    let mut u = [0.0; D];
    u[0] = dip.cos() * az.sin();
    u[1] = dip.cos() * az.cos();
    if D == 3 {
        u[2] = dip.sin();
    }
    u
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

impl VariogramConfig {
    /// Builds a config with the default-distance convention shared by every
    /// front-end (CLI, Python, WASM): when `max_dist` is `None`, a third of
    /// the bounding-box diagonal of `data` is used.
    pub fn for_data<const D: usize>(
        data: &PointSet<D>,
        n_lags: usize,
        max_dist: Option<f64>,
        direction: Option<DirectionConfig>,
    ) -> Self {
        let (min, max) = data.bbox();
        let diag = (0..D)
            .map(|d| (max[d] - min[d]).powi(2))
            .sum::<f64>()
            .sqrt();
        Self {
            n_lags,
            max_dist: max_dist.unwrap_or(diag / 3.0),
            direction,
        }
    }
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
pub fn experimental_variogram<const D: usize>(
    data: &PointSet<D>,
    cfg: &VariogramConfig,
) -> Result<ExperimentalVariogram> {
    pair_bins(data.coords(), data.values(), data.values(), cfg)
}

/// Shared pair-accumulation kernel: bins
/// `0.5 * (a_i - a_j) * (b_i - b_j)` by pair distance. With `a == b` this is
/// the direct semivariogram; with two collocated variables, the cross
/// semivariogram.
pub(crate) fn pair_bins<const D: usize>(
    coords: &[[f64; D]],
    values_a: &[f64],
    values_b: &[f64],
    cfg: &VariogramConfig,
) -> Result<ExperimentalVariogram> {
    if coords.len() < 2 {
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

    let n = coords.len();
    let n_lags = cfg.n_lags;
    let width = cfg.max_dist / n_lags as f64;
    // Sign-agnostic cone test: |dot(pair, u)| >= |pair| * cos(tol).
    let dir = cfg.direction.as_ref().map(|d| {
        let u: [f64; D] = direction_unit_vector(d.azimuth_deg, d.dip_deg);
        (u, d.tolerance_deg.to_radians().cos())
    });

    // Pair accumulation, split into row chunks for parallelism.
    let chunks = n_chunks();
    let rows_per_chunk = n.div_ceil(chunks);
    let acc = par_map(chunks, |c| {
        let mut acc = Acc::new(n_lags);
        let lo = c * rows_per_chunk;
        let hi = ((c + 1) * rows_per_chunk).min(n);
        for i in lo..hi {
            for j in (i + 1)..n {
                let mut dh = [0.0; D];
                let mut d2 = 0.0;
                for (dd, dhd) in dh.iter_mut().enumerate() {
                    *dhd = coords[j][dd] - coords[i][dd];
                    d2 += *dhd * *dhd;
                }
                let d = d2.sqrt();
                if d <= 0.0 || d > cfg.max_dist {
                    continue;
                }
                if let Some((u, cos_tol)) = dir {
                    let mut dot = 0.0;
                    for dd in 0..D {
                        dot += dh[dd] * u[dd];
                    }
                    if dot.abs() < d * cos_tol {
                        continue;
                    }
                }
                // Right-closed lag intervals ((b-1)w, bw], matching the
                // gstat/GSLIB convention for pairs exactly on a boundary.
                let bin = ((d / width).ceil() as usize - 1).min(n_lags - 1);
                acc.sq[bin] += 0.5 * (values_a[i] - values_a[j]) * (values_b[i] - values_b[j]);
                acc.h[bin] += d;
                acc.n[bin] += 1;
            }
        }
        acc
    })
    .into_iter()
    .fold(Acc::new(n_lags), |a, b| a.merge(b));

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
        // Right-closed bins of width 0.5: d=1 (x2, gamma 0.5 each) falls in
        // (0.5, 1.0] -> bin 1; d=2 (gamma 2.0) in (1.5, 2.0] -> bin 3.
        assert_eq!(ev.bins[0].n_pairs, 0);
        assert_eq!(ev.bins[1].n_pairs, 2);
        assert!((ev.bins[1].gamma - 0.5).abs() < 1e-12);
        assert!((ev.bins[1].h - 1.0).abs() < 1e-12);
        assert_eq!(ev.bins[2].n_pairs, 0);
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
            direction: Some(DirectionConfig::horizontal(90.0, 10.0)),
        };
        let ev = experimental_variogram(&data, &cfg).unwrap();
        assert_eq!(ev.bins[0].n_pairs, 2);
        // gamma = mean(0.5*1, 0.5*1) = 0.5
        assert!((ev.bins[0].gamma - 0.5).abs() < 1e-12);

        // North-South: pairs (0,2) and (1,3), dz = 2 both.
        let cfg = VariogramConfig {
            n_lags: 1,
            max_dist: 1.01,
            direction: Some(DirectionConfig::horizontal(0.0, 10.0)),
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
