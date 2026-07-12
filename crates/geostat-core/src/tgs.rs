//! Truncated Gaussian simulation (TGS) for ordered categorical/facies data.
//!
//! Simulates ONE underlying continuous Gaussian random field via the same
//! sequential conditioning engine as [`crate::simulation`] (shared through
//! [`crate::simulation::simulate_gaussian_path`]), then truncates it into
//! `K` ordered categories at `K - 1` thresholds derived from global
//! category proportions via the inverse standard-normal CDF
//! ([`crate::transform::inv_norm_cdf`]). Hard categorical data is converted
//! to a "pseudo-Gaussian" conditioning value — the inverse-normal-CDF of
//! its category's cumulative-probability-interval midpoint (Emery 2004;
//! Armstrong et al. 2011 §3) — so the field can be conditioned on it like
//! any other Gaussian datum.
//!
//! This is the classical, single-field method (GSLIB `tgsim`): categories
//! must have a fixed spatial order (e.g. a depositional/grading sequence),
//! since a 1-D threshold ladder cannot represent arbitrary contact
//! relationships between facies.
//!
//! **Two things this module deliberately does not do:**
//! - `model` (the underlying Gaussian field's variogram) is always
//!   caller-supplied, never auto-fitted. TGS conditions on only a handful
//!   of discrete pseudo-Gaussian levels (one per category) — an
//!   experimental variogram fit directly to those would be statistically
//!   degenerate. GSLIB practice calibrates this model by iteratively
//!   matching the *facies indicator* variograms it reproduces against
//!   target ones, an external, judgment-driven step not automated here.
//! - **Plurigaussian simulation** (2+ correlated Gaussian fields plus a
//!   flexible 2-D truncation rule, which allows arbitrary — not just
//!   ordered — facies contact relationships) is not implemented. That
//!   would need a second cross-correlated field (joint simulation, not
//!   just independent conditioning), a user- or auto-derived 2-D
//!   partition instead of this module's 1-D threshold ladder, and
//!   category proportions computed as bivariate-normal areas instead of a
//!   univariate CDF split — a real architectural extension, not a small
//!   addition here.
//!
//! No gstat/GSLIB reference implementation of this exact scheme was
//! available to cross-validate against; this module is validated by
//! self-consistency and known theoretical properties instead, following
//! the same precedent as collocated cokriging/Markov-Bayes.

use crate::error::{GeostatError, Result};
use crate::grid::{Grid2D, Grid3D};
use crate::rng::splitmix64;
use crate::search::BucketGrid;
use crate::simulation::{grid_level, simulate_gaussian_path};
use crate::transform::inv_norm_cdf;
use crate::variogram::VariogramModel;

/// Hard categorical (facies) observations for truncated Gaussian
/// simulation. Distinct from [`crate::PointSet`]: `categories` are 0-based
/// ordered category indices, not a continuous attribute.
#[derive(Debug, Clone)]
pub struct CategoricalData<const D: usize = 2> {
    coords: Vec<[f64; D]>,
    categories: Vec<usize>,
    n_categories: usize,
}

impl<const D: usize> CategoricalData<D> {
    /// Builds a categorical data set. `n_categories`: pass `None` to infer
    /// it as `max(categories) + 1`.
    pub fn new(
        coords: Vec<[f64; D]>,
        categories: Vec<usize>,
        n_categories: Option<usize>,
    ) -> Result<Self> {
        if coords.is_empty() {
            return Err(GeostatError::InsufficientData("no points provided".into()));
        }
        if coords.len() != categories.len() {
            return Err(GeostatError::DimensionMismatch(format!(
                "{} coordinates vs {} categories",
                coords.len(),
                categories.len()
            )));
        }
        if coords.iter().flatten().any(|v| !v.is_finite()) {
            return Err(GeostatError::InvalidParameter(
                "non-finite coordinate".into(),
            ));
        }
        let n_categories =
            n_categories.unwrap_or_else(|| categories.iter().copied().max().map_or(0, |m| m + 1));
        if let Some(&bad) = categories.iter().find(|&&c| c >= n_categories) {
            return Err(GeostatError::InvalidParameter(format!(
                "category {bad} out of range (0..{n_categories})"
            )));
        }
        if n_categories < 2 {
            return Err(GeostatError::InvalidParameter(
                "at least 2 categories required".into(),
            ));
        }
        Ok(Self {
            coords,
            categories,
            n_categories,
        })
    }

    /// All coordinates.
    pub fn coords(&self) -> &[[f64; D]] {
        &self.coords
    }

    /// All category labels (0-based, ordered).
    pub fn categories(&self) -> &[usize] {
        &self.categories
    }

    /// Number of ordered categories.
    pub fn n_categories(&self) -> usize {
        self.n_categories
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Whether the set is empty (never true for a constructed value).
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Axis-aligned bounding box as `(min, max)` corners.
    pub fn bbox(&self) -> ([f64; D], [f64; D]) {
        let mut min = [f64::INFINITY; D];
        let mut max = [f64::NEG_INFINITY; D];
        for c in &self.coords {
            for d in 0..D {
                min[d] = min[d].min(c[d]);
                max[d] = max[d].max(c[d]);
            }
        }
        (min, max)
    }
}

/// Configuration for truncated Gaussian simulation.
///
/// `#[non_exhaustive]`: construct via `TgsConfig { n_realizations, seed, ..
/// Default::default() }`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TgsConfig {
    /// Number of realizations to generate.
    pub n_realizations: usize,
    /// Base seed; realization `r` uses a stream derived from `(seed, r)`.
    pub seed: u64,
    /// Maximum number of conditioning points per node.
    pub max_neighbors: usize,
    /// Optional search radius for conditioning points.
    pub search_radius: Option<f64>,
    /// Global category proportions (ascending category order, one entry
    /// per [`CategoricalData::n_categories`], summing to 1). Empty
    /// (default) auto-estimates from hard-data frequency via
    /// [`category_proportions`].
    pub proportions: Vec<f64>,
    /// Optional declustering weights for the auto-estimated proportions
    /// (mirrors `SisConfig::decluster_weights`); ignored when `proportions`
    /// is given explicitly.
    pub decluster_weights: Option<Vec<f64>>,
    /// Separate quota for previously simulated nodes (GSLIB `nodmax`).
    pub max_node_neighbors: Option<usize>,
    /// Multiple-grid simulation levels (GSLIB `nmult`; grid entry points
    /// only). `0` (default) keeps a fully random path.
    pub multigrid: u8,
}

impl Default for TgsConfig {
    fn default() -> Self {
        Self {
            n_realizations: 1,
            seed: 42,
            max_neighbors: 16,
            search_radius: None,
            proportions: Vec::new(),
            decluster_weights: None,
            max_node_neighbors: None,
            multigrid: 0,
        }
    }
}

/// Result of a 2-D TGS run: realizations in grid storage order, one
/// category id per cell.
#[derive(Debug, Clone)]
pub struct TgsResult {
    /// The simulation grid.
    pub grid: Grid2D,
    /// One vector of `grid.n_cells()` category ids per realization.
    pub realizations: Vec<Vec<usize>>,
}

/// `K - 1` ascending thresholds truncating a standard-normal field into `K`
/// ordered categories with the given global proportions: category `0` is
/// `Y <= thresholds[0]`, category `k` (`0 < k < K - 1`) is
/// `thresholds[k-1] < Y <= thresholds[k]`, and category `K - 1` is
/// `Y > thresholds[K-2]`. Requires at least 2 proportions, all positive,
/// summing to 1 (tolerance `1e-6`).
pub fn tgs_thresholds(proportions: &[f64]) -> Result<Vec<f64>> {
    if proportions.len() < 2 {
        return Err(GeostatError::InvalidParameter(
            "at least 2 categories required".into(),
        ));
    }
    if proportions.iter().any(|&p| !(p > 0.0)) {
        return Err(GeostatError::InvalidParameter(
            "category proportions must be positive".into(),
        ));
    }
    let sum: f64 = proportions.iter().sum();
    if (sum - 1.0).abs() > 1e-6 {
        return Err(GeostatError::InvalidParameter(format!(
            "proportions must sum to 1 (got {sum})"
        )));
    }
    let mut cum = 0.0;
    Ok(proportions[..proportions.len() - 1]
        .iter()
        .map(|&p| {
            cum += p;
            inv_norm_cdf(cum)
        })
        .collect())
}

/// Category index (0-based) for a standard-normal value `y`, given
/// ascending `thresholds` from [`tgs_thresholds`]. Ties fall in the lower
/// category (matches how `tgs_thresholds` builds each category's upper
/// bound as a closed interval).
pub fn tgs_classify(y: f64, thresholds: &[f64]) -> usize {
    thresholds.partition_point(|&t| t < y)
}

/// Converts hard categorical labels into pseudo-Gaussian conditioning
/// values: each category's cumulative-probability-interval midpoint,
/// mapped through the inverse normal CDF (Emery 2004; Armstrong et al.
/// 2011 §3 — the standard TGS device for conditioning the single
/// underlying field on hard facies data).
pub fn category_to_pseudo_gaussian(categories: &[usize], proportions: &[f64]) -> Result<Vec<f64>> {
    let k = proportions.len();
    let mut cum = vec![0.0; k + 1];
    for i in 0..k {
        cum[i + 1] = cum[i] + proportions[i];
    }
    categories
        .iter()
        .map(|&c| {
            if c >= k {
                return Err(GeostatError::InvalidParameter(format!(
                    "category {c} out of range (0..{k})"
                )));
            }
            Ok(inv_norm_cdf(0.5 * (cum[c] + cum[c + 1])))
        })
        .collect()
}

/// Global category proportions from hard-data frequency, optionally
/// declustering-weighted (mirrors `SisConfig`'s cutoff-proportion
/// auto-estimate). Every category in `0..n_categories` should occur at
/// least once with positive weight; a category with zero estimated
/// proportion makes [`tgs_thresholds`] reject the result -- supply
/// `TgsConfig::proportions` explicitly instead in that case.
pub fn category_proportions(
    categories: &[usize],
    n_categories: usize,
    weights: Option<&[f64]>,
) -> Result<Vec<f64>> {
    if let Some(w) = weights
        && w.len() != categories.len()
    {
        return Err(GeostatError::DimensionMismatch(format!(
            "{} weights vs {} categories",
            w.len(),
            categories.len()
        )));
    }
    let mut props = vec![0.0; n_categories];
    match weights {
        Some(w) => {
            let wsum: f64 = w.iter().sum();
            if !(wsum > 0.0) {
                return Err(GeostatError::InvalidParameter(
                    "declustering weights must sum to a positive value".into(),
                ));
            }
            for (&c, &wi) in categories.iter().zip(w) {
                props[c] += wi;
            }
            for p in &mut props {
                *p /= wsum;
            }
        }
        None => {
            for &c in categories {
                props[c] += 1.0;
            }
            let n = categories.len() as f64;
            for p in &mut props {
                *p /= n;
            }
        }
    }
    Ok(props)
}

/// TGS at an arbitrary set of simulation nodes (sequential path over the
/// node list). `multigrid` is ignored here (it needs grid topology); use
/// the grid entry points for multiple-grid simulation.
pub fn tgs_at<const D: usize>(
    data: &CategoricalData<D>,
    model: &VariogramModel,
    nodes: &[[f64; D]],
    cfg: &TgsConfig,
) -> Result<Vec<Vec<usize>>> {
    tgs_at_with_levels(data, model, nodes, None, cfg)
}

/// Runs truncated Gaussian simulation on a 2-D grid.
///
/// `model` must be a variogram of the underlying standard-Gaussian field
/// (its total sill should therefore be close to 1) -- see the module docs
/// for why this is not auto-fitted.
pub fn truncated_gaussian_simulation(
    data: &CategoricalData<2>,
    model: &VariogramModel,
    grid: &Grid2D,
    cfg: &TgsConfig,
) -> Result<TgsResult> {
    let levels = (cfg.multigrid > 0).then(|| {
        (0..grid.n_cells())
            .map(|i| grid_level(&[i % grid.nx, i / grid.nx], cfg.multigrid))
            .collect::<Vec<u8>>()
    });
    let realizations = tgs_at_with_levels(data, model, &grid.centers(), levels.as_deref(), cfg)?;
    Ok(TgsResult {
        grid: grid.clone(),
        realizations,
    })
}

/// Runs truncated Gaussian simulation on a 3-D grid, returning the
/// realizations in grid storage order.
pub fn truncated_gaussian_simulation_3d(
    data: &CategoricalData<3>,
    model: &VariogramModel,
    grid: &Grid3D,
    cfg: &TgsConfig,
) -> Result<Vec<Vec<usize>>> {
    let levels = (cfg.multigrid > 0).then(|| {
        (0..grid.n_cells())
            .map(|i| {
                let ix = i % grid.nx;
                let iy = (i / grid.nx) % grid.ny;
                let iz = i / (grid.nx * grid.ny);
                grid_level(&[ix, iy, iz], cfg.multigrid)
            })
            .collect::<Vec<u8>>()
    });
    tgs_at_with_levels(data, model, &grid.centers(), levels.as_deref(), cfg)
}

fn tgs_at_with_levels<const D: usize>(
    data: &CategoricalData<D>,
    model: &VariogramModel,
    nodes: &[[f64; D]],
    levels: Option<&[u8]>,
    cfg: &TgsConfig,
) -> Result<Vec<Vec<usize>>> {
    if model.has_power() {
        return Err(GeostatError::InvalidParameter(
            "TGS needs a valid covariance function and cannot use the unbounded Power model".into(),
        ));
    }
    if let Some(kind) = model.invalid_structure_for_dim(D) {
        return Err(GeostatError::InvalidParameter(format!(
            "{kind:?} is not a valid covariance in {D} dimensions; use Spherical instead for a \
             3-D-safe bounded structure"
        )));
    }
    if cfg.n_realizations == 0 {
        return Err(GeostatError::InvalidParameter(
            "n_realizations must be at least 1".into(),
        ));
    }
    if cfg.max_neighbors == 0 {
        return Err(GeostatError::InvalidParameter(
            "max_neighbors must be at least 1".into(),
        ));
    }
    if let Some(r) = cfg.search_radius
        && !(r > 0.0)
    {
        return Err(GeostatError::InvalidParameter(format!(
            "search radius must be positive, got {r}"
        )));
    }
    if nodes.is_empty() {
        return Err(GeostatError::InvalidParameter(
            "no simulation nodes given".into(),
        ));
    }

    let proportions = if cfg.proportions.is_empty() {
        category_proportions(
            data.categories(),
            data.n_categories(),
            cfg.decluster_weights.as_deref(),
        )?
    } else {
        if cfg.proportions.len() != data.n_categories() {
            return Err(GeostatError::DimensionMismatch(format!(
                "{} proportions vs {} categories",
                cfg.proportions.len(),
                data.n_categories()
            )));
        }
        cfg.proportions.clone()
    };
    let thresholds = tgs_thresholds(&proportions)?;
    let pseudo = category_to_pseudo_gaussian(data.categories(), &proportions)?;

    // Extents covering data and nodes, shared by both search structures --
    // mirrors `sgs_at_with_levels`/`sis_at_with_levels`.
    let (dmin, dmax) = data.bbox();
    let mut min = dmin;
    let mut max = dmax;
    for c in nodes {
        for d in 0..D {
            min[d] = min[d].min(c[d]);
            max[d] = max[d].max(c[d]);
        }
    }
    let mut data_grid = BucketGrid::new(min, max, data.len());
    for &p in data.coords() {
        data_grid.insert(p);
    }

    crate::parallel::par_try_map(cfg.n_realizations, |r| {
        let mut seed_state = cfg.seed ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let seed_r = splitmix64(&mut seed_state);
        let sim_y = simulate_gaussian_path(
            data.coords(),
            &pseudo,
            &data_grid,
            model,
            nodes,
            levels,
            (min, max),
            cfg.max_neighbors,
            cfg.search_radius,
            cfg.max_node_neighbors,
            seed_r,
        )?;
        Ok(sim_y
            .iter()
            .map(|&y| tgs_classify(y, &thresholds))
            .collect())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variogram::{ModelKind, Structure};

    fn setup() -> (CategoricalData, VariogramModel, Grid2D) {
        let data = CategoricalData::new(
            vec![
                [10.0, 10.0],
                [90.0, 10.0],
                [10.0, 90.0],
                [90.0, 90.0],
                [50.0, 50.0],
                [30.0, 70.0],
                [70.0, 30.0],
            ],
            vec![0, 2, 0, 2, 1, 1, 1],
            Some(3),
        )
        .unwrap();
        // Standard-normal-field model: sill ~ 1.
        let model = VariogramModel::new(
            0.05,
            vec![Structure::new(ModelKind::Exponential, 0.95, 30.0)],
        )
        .unwrap();
        let grid = Grid2D::from_bbox([0.0, 0.0], [100.0, 100.0], 20, 20).unwrap();
        (data, model, grid)
    }

    #[test]
    fn thresholds_match_inv_norm_cdf_by_hand() {
        let t = tgs_thresholds(&[0.2, 0.3, 0.5]).unwrap();
        assert_eq!(t.len(), 2);
        assert!((t[0] - inv_norm_cdf(0.2)).abs() < 1e-15);
        assert!((t[1] - inv_norm_cdf(0.5)).abs() < 1e-15);
    }

    #[test]
    fn two_category_median_threshold_is_zero() {
        let t = tgs_thresholds(&[0.5, 0.5]).unwrap();
        assert_eq!(t.len(), 1);
        assert!(t[0].abs() < 1e-9, "expected ~0.0, got {}", t[0]);
    }

    #[test]
    fn category_proportions_matches_empirical_frequency() {
        let categories = [0, 0, 0, 1, 1, 2, 2, 2, 2];
        let props = category_proportions(&categories, 3, None).unwrap();
        assert!((props[0] - 3.0 / 9.0).abs() < 1e-12);
        assert!((props[1] - 2.0 / 9.0).abs() < 1e-12);
        assert!((props[2] - 4.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn ensemble_proportions_track_input_proportions() {
        let (data, model, grid) = setup();
        let proportions = vec![0.2, 0.5, 0.3];
        let cfg = TgsConfig {
            n_realizations: 40,
            seed: 7,
            proportions: proportions.clone(),
            ..Default::default()
        };
        let res = truncated_gaussian_simulation(&data, &model, &grid, &cfg).unwrap();
        let mut counts = [0usize; 3];
        let mut total = 0usize;
        for r in &res.realizations {
            for &c in r {
                counts[c] += 1;
                total += 1;
            }
        }
        for (k, &p) in proportions.iter().enumerate() {
            let empirical = counts[k] as f64 / total as f64;
            assert!(
                (empirical - p).abs() < 0.05,
                "category {k}: empirical {empirical} vs input {p}"
            );
        }
    }

    #[test]
    fn hard_data_category_honored_at_conditioning_locations() {
        let (data, model, _grid) = setup();
        let cfg = TgsConfig {
            n_realizations: 5,
            seed: 11,
            ..Default::default()
        };
        let realizations = tgs_at(&data, &model, data.coords(), &cfg).unwrap();
        for r in &realizations {
            assert_eq!(r, data.categories());
        }
    }

    #[test]
    fn reproducible_with_same_seed() {
        let (data, model, grid) = setup();
        let cfg = TgsConfig {
            n_realizations: 3,
            seed: 123,
            ..Default::default()
        };
        let a = truncated_gaussian_simulation(&data, &model, &grid, &cfg).unwrap();
        let b = truncated_gaussian_simulation(&data, &model, &grid, &cfg).unwrap();
        assert_eq!(a.realizations, b.realizations);
        let cfg2 = TgsConfig { seed: 124, ..cfg };
        let c = truncated_gaussian_simulation(&data, &model, &grid, &cfg2).unwrap();
        assert_ne!(a.realizations[0], c.realizations[0]);
    }

    #[test]
    fn rejects_bad_config() {
        let (data, model, grid) = setup();

        let cfg = TgsConfig {
            n_realizations: 0,
            ..Default::default()
        };
        assert!(truncated_gaussian_simulation(&data, &model, &grid, &cfg).is_err());

        let cfg = TgsConfig {
            max_neighbors: 0,
            ..Default::default()
        };
        assert!(truncated_gaussian_simulation(&data, &model, &grid, &cfg).is_err());

        let cfg = TgsConfig {
            proportions: vec![0.5, 0.6],
            ..Default::default()
        };
        assert!(truncated_gaussian_simulation(&data, &model, &grid, &cfg).is_err());

        let cfg = TgsConfig {
            proportions: vec![0.3, 0.7],
            ..Default::default()
        };
        assert!(truncated_gaussian_simulation(&data, &model, &grid, &cfg).is_err());

        let power_model =
            VariogramModel::new(0.0, vec![Structure::new(ModelKind::Power(1.0), 1.0, 1.0)])
                .unwrap();
        assert!(
            truncated_gaussian_simulation(&data, &power_model, &grid, &TgsConfig::default())
                .is_err()
        );

        assert!(CategoricalData::<2>::new(vec![[0.0, 0.0]], vec![5], Some(3)).is_err());
        assert!(CategoricalData::<2>::new(vec![[0.0, 0.0]], vec![], None).is_err());
    }
}
