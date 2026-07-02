//! Cell declustering (GSLIB `declus`).
//!
//! Preferential sampling — drillholes clustered in high-grade zones is the
//! canonical case — biases every statistic computed with equal weights,
//! including the reference distribution of the normal-score transform. Cell
//! declustering overlays a grid of a given cell size and weights each sample
//! inversely to the number of samples sharing its cell, so clustered points
//! share their influence. Weights are averaged over several grid-origin
//! offsets to reduce placement artifacts and normalized to sum to `n`.

use std::collections::HashMap;

use crate::data::PointSet;
use crate::error::{GeostatError, Result};

/// Cell-declustering weights for a given cell size (same size in every
/// dimension), averaged over `n_offsets` systematic grid-origin offsets.
/// The returned weights are positive and sum to `data.len()` (mean 1).
pub fn cell_declustering_weights<const D: usize>(
    data: &PointSet<D>,
    cell_size: f64,
    n_offsets: usize,
) -> Result<Vec<f64>> {
    if !(cell_size > 0.0) || !cell_size.is_finite() {
        return Err(GeostatError::InvalidParameter(format!(
            "cell size must be finite and > 0, got {cell_size}"
        )));
    }
    if n_offsets == 0 {
        return Err(GeostatError::InvalidParameter(
            "n_offsets must be at least 1".into(),
        ));
    }
    let n = data.len();
    let (min, _) = data.bbox();
    let mut weights = vec![0.0_f64; n];
    let mut cells: HashMap<[i64; D], usize> = HashMap::new();
    let mut cell_of = vec![[0_i64; D]; n];

    for o in 0..n_offsets {
        // Systematic origin shift: a fraction of the cell size per offset.
        let shift = cell_size * (o as f64 / n_offsets as f64);
        cells.clear();
        for (i, c) in data.coords().iter().enumerate() {
            let idx: [i64; D] =
                std::array::from_fn(|d| ((c[d] - min[d] + shift) / cell_size).floor() as i64);
            *cells.entry(idx).or_insert(0) += 1;
            cell_of[i] = idx;
        }
        let n_occupied = cells.len() as f64;
        for i in 0..n {
            weights[i] += 1.0 / (cells[&cell_of[i]] as f64 * n_occupied);
        }
    }
    // Each offset's weights sum to 1; normalize the average to sum to n.
    let scale = n as f64 / n_offsets as f64;
    for w in &mut weights {
        *w *= scale;
    }
    Ok(weights)
}

/// Result of a declustering cell-size scan.
#[derive(Debug, Clone)]
pub struct DeclusterScan {
    /// Cell size whose declustered mean is the extreme requested.
    pub best_size: f64,
    /// Declustered mean at `best_size`.
    pub best_mean: f64,
    /// Weights at `best_size` (sum to `n`).
    pub weights: Vec<f64>,
    /// `(cell_size, declustered_mean)` for every size scanned.
    pub trace: Vec<(f64, f64)>,
}

/// Scans `n_sizes` cell sizes between `min_size` and `max_size` (GSLIB
/// `declus` practice) and keeps the one that minimizes (`minimize = true`,
/// for data preferentially clustered in high values) or maximizes the
/// declustered mean.
pub fn decluster_scan<const D: usize>(
    data: &PointSet<D>,
    min_size: f64,
    max_size: f64,
    n_sizes: usize,
    n_offsets: usize,
    minimize: bool,
) -> Result<DeclusterScan> {
    if !(min_size > 0.0) || !(max_size >= min_size) {
        return Err(GeostatError::InvalidParameter(format!(
            "invalid scan range [{min_size}, {max_size}]"
        )));
    }
    if n_sizes == 0 {
        return Err(GeostatError::InvalidParameter(
            "n_sizes must be at least 1".into(),
        ));
    }
    let n = data.len() as f64;
    let mut best: Option<DeclusterScan> = None;
    let mut trace = Vec::with_capacity(n_sizes);
    for k in 0..n_sizes {
        let size = if n_sizes == 1 {
            min_size
        } else {
            min_size + (max_size - min_size) * k as f64 / (n_sizes - 1) as f64
        };
        let weights = cell_declustering_weights(data, size, n_offsets)?;
        let mean = weights
            .iter()
            .zip(data.values())
            .map(|(&w, &v)| w * v)
            .sum::<f64>()
            / n;
        trace.push((size, mean));
        let better = match &best {
            None => true,
            Some(b) => {
                if minimize {
                    mean < b.best_mean
                } else {
                    mean > b.best_mean
                }
            }
        };
        if better {
            best = Some(DeclusterScan {
                best_size: size,
                best_mean: mean,
                weights,
                trace: Vec::new(),
            });
        }
    }
    let mut best = best.expect("n_sizes >= 1");
    best.trace = trace;
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense cluster of high values in one corner plus a sparse regular
    /// background of low values.
    fn clustered() -> PointSet {
        let mut coords = Vec::new();
        let mut values = Vec::new();
        // Sparse background: 5x5 regular grid, value 1.
        for i in 0..5 {
            for j in 0..5 {
                coords.push([i as f64 * 20.0 + 5.0, j as f64 * 20.0 + 5.0]);
                values.push(1.0);
            }
        }
        // Dense cluster near the origin: 25 points in a 4x4 box, value 10.
        for i in 0..5 {
            for j in 0..5 {
                coords.push([i as f64 + 0.5, j as f64 + 0.5]);
                values.push(10.0);
            }
        }
        PointSet::new(coords, values).unwrap()
    }

    #[test]
    fn uniform_grid_gets_uniform_weights() {
        let mut coords = Vec::new();
        for i in 0..6 {
            for j in 0..6 {
                coords.push([i as f64 * 10.0, j as f64 * 10.0]);
            }
        }
        let n = coords.len();
        let data = PointSet::new(coords, vec![1.0; n]).unwrap();
        let w = cell_declustering_weights(&data, 10.0, 1).unwrap();
        assert!((w.iter().sum::<f64>() - n as f64).abs() < 1e-9);
        for &wi in &w {
            assert!((wi - 1.0).abs() < 1e-9, "weight {wi}");
        }
    }

    #[test]
    fn clustered_points_get_downweighted() {
        let data = clustered();
        let w = cell_declustering_weights(&data, 20.0, 4).unwrap();
        assert!((w.iter().sum::<f64>() - data.len() as f64).abs() < 1e-9);
        // Cluster points (last 25) must weigh less than background points.
        let bg_mean = w[..25].iter().sum::<f64>() / 25.0;
        let cl_mean = w[25..].iter().sum::<f64>() / 25.0;
        assert!(
            cl_mean < 0.3 * bg_mean,
            "cluster {cl_mean} vs background {bg_mean}"
        );
        // Declustered mean must drop toward the background value.
        let naive = data.mean();
        let declustered = w
            .iter()
            .zip(data.values())
            .map(|(&wi, &v)| wi * v)
            .sum::<f64>()
            / data.len() as f64;
        assert!(
            declustered < naive - 1.0,
            "declustered {declustered} vs naive {naive}"
        );
    }

    #[test]
    fn scan_finds_a_mean_reducing_size() {
        let data = clustered();
        let scan = decluster_scan(&data, 2.0, 50.0, 20, 4, true).unwrap();
        assert_eq!(scan.trace.len(), 20);
        assert!(scan.best_mean < data.mean() - 1.0);
        assert!((scan.weights.iter().sum::<f64>() - data.len() as f64).abs() < 1e-9);
        // The reported best is the minimum of the trace.
        let trace_min = scan.trace.iter().map(|t| t.1).fold(f64::INFINITY, f64::min);
        assert!((scan.best_mean - trace_min).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_parameters() {
        let data = clustered();
        assert!(cell_declustering_weights(&data, 0.0, 4).is_err());
        assert!(cell_declustering_weights(&data, 10.0, 0).is_err());
        assert!(decluster_scan(&data, 10.0, 5.0, 5, 4, true).is_err());
        assert!(decluster_scan(&data, 5.0, 10.0, 0, 4, true).is_err());
    }
}
