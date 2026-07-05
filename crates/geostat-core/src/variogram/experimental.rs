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
    /// Number of point pairs at distance `0` (duplicate locations, or a
    /// genuine zero-distance measurement pair) that were silently dropped
    /// from every bin -- worth reporting because a large count usually
    /// means duplicated data rather than a real spatial feature
    /// (AUDIT-2026-07-v2.md §4 "reporte de pares coincidentes", previously
    /// a mute `continue`).
    pub coincident_pairs: usize,
}

#[derive(Clone)]
struct Acc {
    sq: Vec<f64>,
    h: Vec<f64>,
    n: Vec<usize>,
    coincident: usize,
}

impl Acc {
    fn new(n_lags: usize) -> Self {
        Self {
            sq: vec![0.0; n_lags],
            h: vec![0.0; n_lags],
            n: vec![0; n_lags],
            coincident: 0,
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for b in 0..self.sq.len() {
            self.sq[b] += other.sq[b];
            self.h[b] += other.h[b];
            self.n[b] += other.n[b];
        }
        self.coincident += other.coincident;
        self
    }
}

/// A direction-cone test: `|dot(pair, unit_vector)| >= |pair| * cos_tol`.
type DirCone<const D: usize> = ([f64; D], f64);

/// Shared setup for every pair-loop estimator: validates `cfg`, and returns
/// the lag width and the sign-agnostic direction-cone test, if any.
fn validate_and_setup<const D: usize>(
    n_points: usize,
    cfg: &VariogramConfig,
) -> Result<(f64, Option<DirCone<D>>)> {
    if n_points < 2 {
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
    let width = cfg.max_dist / cfg.n_lags as f64;
    let dir = cfg.direction.as_ref().map(|d| {
        let u: [f64; D] = direction_unit_vector(d.azimuth_deg, d.dip_deg);
        (u, d.tolerance_deg.to_radians().cos())
    });
    Ok((width, dir))
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
    let (width, dir) = validate_and_setup::<D>(coords.len(), cfg)?;
    let n = coords.len();
    let n_lags = cfg.n_lags;

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
                if d <= 0.0 {
                    acc.coincident += 1;
                    continue;
                }
                if d > cfg.max_dist {
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

    let coincident_pairs = acc.coincident;
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
        coincident_pairs,
    })
}

/// Point-pair semivariogram estimators (gstat/GSLIB `gamv` estimator
/// family; AUDIT-2026-07-v2.md §4). `Matheron` is the classical
/// mean-squared-difference estimator ([`experimental_variogram`]'s only
/// estimator); the others trade some statistical efficiency under Gaussian
/// differences for resistance to a few outlier pairs -- "lo primero que un
/// usuario gstat/reviewer de Mathematical Geosciences nota" if missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimatorKind {
    /// `gamma(h) = mean(0.5*(z_i-z_j)^2)`. The BLUE estimator under Gaussian
    /// differences, but a single extreme pair can dominate a bin.
    Matheron,
    /// Cressie & Hawkins (1980): `2*gamma(h) = mean(|z_i-z_j|^0.5)^4 /
    /// (0.457 + 0.494/N(h))`. The `^0.5` power-transform compresses
    /// outliers before averaging, then the quartic + bias-correction
    /// constants (Cressie 1993, eq. 2.4.12) undo the transform in
    /// expectation under Gaussian differences.
    CressieHawkins,
    /// Dowd (1984): `gamma(h) = median(|z_i-z_j|)^2 / (2*0.4529)`. The most
    /// outlier-resistant of the three -- only the rank-median pair matters,
    /// so a single wild value cannot move the estimate at all.
    Dowd,
    /// Madogram (first-order variogram): `gamma(h) = 0.5*mean(|z_i-z_j|)`.
    /// Bounded growth rate under heavy-tailed/fractal fields where the
    /// squared-difference variogram's second moment may not even exist.
    Madogram,
}

/// Experimental variogram under a robust/alternative estimator (see
/// [`EstimatorKind`]). `Matheron` delegates to [`experimental_variogram`]
/// unchanged; the others run their own pair loop since they need the full
/// per-pair distribution within a bin (a quartic mean or a median), not
/// just a running sum.
pub fn experimental_variogram_robust<const D: usize>(
    data: &PointSet<D>,
    cfg: &VariogramConfig,
    estimator: EstimatorKind,
) -> Result<ExperimentalVariogram> {
    if estimator == EstimatorKind::Matheron {
        return experimental_variogram(data, cfg);
    }
    robust_bins(data.coords(), data.values(), cfg, estimator)
}

fn robust_bins<const D: usize>(
    coords: &[[f64; D]],
    values: &[f64],
    cfg: &VariogramConfig,
    estimator: EstimatorKind,
) -> Result<ExperimentalVariogram> {
    let (width, dir) = validate_and_setup::<D>(coords.len(), cfg)?;
    let n = coords.len();
    let n_lags = cfg.n_lags;

    #[derive(Clone)]
    struct RobustAcc {
        // Per-bin: `|diff|^0.5` for Cressie-Hawkins, `|diff|` for Dowd and
        // Madogram (Dowd needs the full distribution for its median; the
        // others just need it summed, but collecting is cheap next to the
        // O(n^2) pair loop itself and keeps one shared accumulator/loop).
        s: Vec<Vec<f64>>,
        h: Vec<f64>,
        n: Vec<usize>,
        coincident: usize,
    }
    impl RobustAcc {
        fn new(n_lags: usize) -> Self {
            Self {
                s: vec![Vec::new(); n_lags],
                h: vec![0.0; n_lags],
                n: vec![0; n_lags],
                coincident: 0,
            }
        }
        fn merge(mut self, mut other: Self) -> Self {
            for b in 0..self.s.len() {
                self.s[b].append(&mut other.s[b]);
                self.h[b] += other.h[b];
                self.n[b] += other.n[b];
            }
            self.coincident += other.coincident;
            self
        }
    }

    let chunks = n_chunks();
    let rows_per_chunk = n.div_ceil(chunks);
    let acc = par_map(chunks, |c| {
        let mut acc = RobustAcc::new(n_lags);
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
                if d <= 0.0 {
                    acc.coincident += 1;
                    continue;
                }
                if d > cfg.max_dist {
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
                let bin = ((d / width).ceil() as usize - 1).min(n_lags - 1);
                let diff = (values[i] - values[j]).abs();
                let s = match estimator {
                    EstimatorKind::CressieHawkins => diff.sqrt(),
                    EstimatorKind::Dowd | EstimatorKind::Madogram => diff,
                    EstimatorKind::Matheron => unreachable!("handled by experimental_variogram"),
                };
                acc.s[bin].push(s);
                acc.h[bin] += d;
                acc.n[bin] += 1;
            }
        }
        acc
    })
    .into_iter()
    .fold(RobustAcc::new(n_lags), |a, b| a.merge(b));

    let coincident_pairs = acc.coincident;
    let bins = (0..n_lags)
        .map(|b| {
            let np = acc.n[b];
            if np > 0 {
                let n_f = np as f64;
                let gamma = match estimator {
                    EstimatorKind::CressieHawkins => {
                        let mean_sqrt = acc.s[b].iter().sum::<f64>() / n_f;
                        mean_sqrt.powi(4) / (2.0 * (0.457 + 0.494 / n_f))
                    }
                    EstimatorKind::Dowd => {
                        let mut v = acc.s[b].clone();
                        v.sort_by(f64::total_cmp);
                        let med = median_of_sorted(&v);
                        med * med / (2.0 * 0.4529)
                    }
                    EstimatorKind::Madogram => 0.5 * acc.s[b].iter().sum::<f64>() / n_f,
                    EstimatorKind::Matheron => unreachable!("handled by experimental_variogram"),
                };
                LagBin {
                    h: acc.h[b] / n_f,
                    gamma,
                    n_pairs: np,
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
        coincident_pairs,
    })
}

fn median_of_sorted(v: &[f64]) -> f64 {
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// One point-pair in the variogram cloud (gstat
/// `plot(variogram(..., cloud = TRUE))` / GSLIB `gamv` pairwise output):
/// every individual pair within `max_dist` (and the direction cone, if
/// any), unbinned -- lets a user trace an outlier bin back to the specific
/// locations driving it, instead of only seeing the aggregated `gamma(h)`.
#[derive(Debug, Clone, Copy)]
pub struct CloudPair {
    /// Index of the first point (into the original data).
    pub i: usize,
    /// Index of the second point (into the original data).
    pub j: usize,
    /// Separation distance.
    pub h: f64,
    /// This pair's contribution `0.5*(z_i-z_j)^2` (Matheron's per-pair
    /// term; the bin average of these is `experimental_variogram`'s
    /// `gamma(h)`).
    pub gamma: f64,
}

/// Builds the variogram cloud: one [`CloudPair`] per point pair within
/// `cfg.max_dist` (and direction cone, if set). `O(n^2)` in both time and
/// output size, same as the pair loop behind every other estimator here --
/// meant for exploratory diagnosis on the kind of dataset variography is
/// normally run on (hundreds to a few thousand points), not for grid-sized
/// point counts.
pub fn variogram_cloud<const D: usize>(
    data: &PointSet<D>,
    cfg: &VariogramConfig,
) -> Result<Vec<CloudPair>> {
    let coords = data.coords();
    let values = data.values();
    let (_, dir) = validate_and_setup::<D>(coords.len(), cfg)?;
    let n = coords.len();

    let chunks = n_chunks();
    let rows_per_chunk = n.div_ceil(chunks);
    let pairs = par_map(chunks, |c| {
        let mut out = Vec::new();
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
                out.push(CloudPair {
                    i,
                    j,
                    h: d,
                    gamma: 0.5 * (values[i] - values[j]) * (values[i] - values[j]),
                });
            }
        }
        out
    })
    .into_iter()
    .flatten()
    .collect();
    Ok(pairs)
}

/// Ergodic correlogram `rho(h) = 1 - gamma(h)/variance` derived from an
/// already-computed experimental variogram, using the intrinsic-stationarity
/// identity `gamma(h) = C(0) - C(h)` with a single global `variance`
/// standing in for `C(0)` (typically the sample variance of the data). This
/// is the simple, ergodic correlogram; it is *not* GSLIB `gamv`'s
/// non-ergodic correlogram, which instead standardizes by the head/tail
/// means and variances computed separately *within each lag* -- a
/// materially different (and more expensive) estimator, left for a future
/// session (AUDIT-2026-07-v2.md §4 groups them together, but they answer
/// different questions: this one assumes stationarity across the whole
/// domain, the non-ergodic one tolerates local drift).
pub fn correlogram(ev: &ExperimentalVariogram, variance: f64) -> Vec<f64> {
    ev.bins.iter().map(|b| 1.0 - b.gamma / variance).collect()
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

    #[test]
    fn robust_estimators_match_hand_computation() {
        // Same setup as `hand_computed_bins`: bin 3 (h in (1.5, 2.0]) holds
        // exactly one pair, d=2, |diff|=2.
        let cfg = VariogramConfig {
            n_lags: 4,
            max_dist: 2.0,
            direction: None,
        };
        let madogram =
            experimental_variogram_robust(&line_data(), &cfg, EstimatorKind::Madogram).unwrap();
        assert!((madogram.bins[3].gamma - 1.0).abs() < 1e-12); // 0.5*2
        assert_eq!(madogram.bins[3].n_pairs, 1);

        let ch = experimental_variogram_robust(&line_data(), &cfg, EstimatorKind::CressieHawkins)
            .unwrap();
        // (sqrt(2))^4 / (2*(0.457+0.494/1)) = 4 / 1.902
        assert!((ch.bins[3].gamma - 4.0 / 1.902).abs() < 1e-9);

        let dowd = experimental_variogram_robust(&line_data(), &cfg, EstimatorKind::Dowd).unwrap();
        // median(|diff|)=2 (only one pair) -> 2^2/(2*0.4529)
        assert!((dowd.bins[3].gamma - 4.0 / 0.9058).abs() < 1e-9);

        // `Matheron` must delegate to `experimental_variogram` exactly.
        let matheron =
            experimental_variogram_robust(&line_data(), &cfg, EstimatorKind::Matheron).unwrap();
        let plain = experimental_variogram(&line_data(), &cfg).unwrap();
        for (a, b) in matheron.bins.iter().zip(&plain.bins) {
            assert_eq!(a.n_pairs, b.n_pairs);
            assert!(a.gamma.is_nan() && b.gamma.is_nan() || (a.gamma - b.gamma).abs() < 1e-15);
        }
    }

    #[test]
    fn robust_estimators_resist_a_single_outlier_pair() {
        // 5 collinear points; adjacent pairs (d=1) all differ by 1 except
        // the last, which differs by 97 -- one wild pair among four.
        // AUDIT-2026-07-v2.md §4: Cressie-Hawkins/Dowd/madogram exist
        // precisely so a single bad pair cannot dominate a bin the way it
        // dominates Matheron's mean-of-squares.
        let data = PointSet::new(
            vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            vec![0.0, 1.0, 2.0, 3.0, 100.0],
        )
        .unwrap();
        let cfg = VariogramConfig {
            n_lags: 1,
            max_dist: 1.0,
            direction: None,
        };
        let matheron = experimental_variogram(&data, &cfg).unwrap().bins[0].gamma;
        let madogram = experimental_variogram_robust(&data, &cfg, EstimatorKind::Madogram)
            .unwrap()
            .bins[0]
            .gamma;
        let ch = experimental_variogram_robust(&data, &cfg, EstimatorKind::CressieHawkins)
            .unwrap()
            .bins[0]
            .gamma;
        let dowd = experimental_variogram_robust(&data, &cfg, EstimatorKind::Dowd)
            .unwrap()
            .bins[0]
            .gamma;

        // Hand-computed: diffs [1, 1, 1, 97].
        assert!((matheron - 1176.5).abs() < 1e-9);
        assert!((madogram - 12.5).abs() < 1e-9);
        assert!((dowd - 1.0 / 0.9058).abs() < 1e-9);
        // Cressie-Hawkins: mean(sqrt(diff)) = (1+1+1+sqrt(97))/4.
        let mean_sqrt = (3.0 + 97.0_f64.sqrt()) / 4.0;
        let expected_ch = mean_sqrt.powi(4) / (2.0 * (0.457 + 0.494 / 4.0));
        assert!((ch - expected_ch).abs() < 1e-9);

        // The whole point: increasing resistance to the outlier pair.
        assert!(dowd < madogram, "{dowd} vs {madogram}");
        assert!(madogram < ch, "{madogram} vs {ch}");
        assert!(ch < matheron, "{ch} vs {matheron}");
    }

    #[test]
    fn coincident_pairs_are_reported() {
        // Two points share a location; the rest are spread out.
        let data = PointSet::new(
            vec![[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            vec![0.0, 5.0, 1.0, 2.0],
        )
        .unwrap();
        let cfg = VariogramConfig {
            n_lags: 4,
            max_dist: 2.0,
            direction: None,
        };
        let ev = experimental_variogram(&data, &cfg).unwrap();
        assert_eq!(ev.coincident_pairs, 1);
        let total_pairs: usize = ev.bins.iter().map(|b| b.n_pairs).sum();
        // C(4,2) = 6 pairs total; 1 coincident, 5 binned.
        assert_eq!(total_pairs, 5);

        let robust = experimental_variogram_robust(&data, &cfg, EstimatorKind::Dowd).unwrap();
        assert_eq!(robust.coincident_pairs, 1);
    }

    #[test]
    fn variogram_cloud_lists_every_pair() {
        let cfg = VariogramConfig {
            n_lags: 4,
            max_dist: 2.0,
            direction: None,
        };
        let cloud = variogram_cloud(&line_data(), &cfg).unwrap();
        // 3 points -> C(3,2) = 3 pairs, all within max_dist=2.0.
        assert_eq!(cloud.len(), 3);
        let ev = experimental_variogram(&line_data(), &cfg).unwrap();
        let total_pairs: usize = ev.bins.iter().map(|b| b.n_pairs).sum();
        assert_eq!(cloud.len(), total_pairs);
        // Every cloud pair's gamma matches the direct hand computation.
        for p in &cloud {
            let expected = 0.5
                * (line_data().value(p.i) - line_data().value(p.j))
                * (line_data().value(p.i) - line_data().value(p.j));
            assert!((p.gamma - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn correlogram_is_one_minus_gamma_over_variance() {
        let cfg = VariogramConfig {
            n_lags: 4,
            max_dist: 2.0,
            direction: None,
        };
        let ev = experimental_variogram(&line_data(), &cfg).unwrap();
        let variance = 1.0; // arbitrary, just checking the formula
        let rho = correlogram(&ev, variance);
        assert_eq!(rho.len(), ev.bins.len());
        for (r, b) in rho.iter().zip(&ev.bins) {
            if b.gamma.is_nan() {
                assert!(r.is_nan());
            } else {
                assert!((r - (1.0 - b.gamma / variance)).abs() < 1e-15);
            }
        }
    }
}
