//! Two-dimensional variogram map (variogram surface).
//!
//! Instead of collapsing point pairs onto a single distance axis, the variogram
//! map bins each pair by its full lag *vector* `(h_x, h_y)` on a regular grid
//! centred at the origin. The semivariance surface this produces makes
//! geometric anisotropy visible directly: the low-semivariance region elongates
//! along the direction of greatest spatial continuity, and its orientation and
//! axis ratio read off the map without scanning directional variograms by hand.

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::parallel::{n_chunks, par_map};

/// A 2-D variogram map: a `size x size` grid of semivariance in lag space,
/// where `size = 2*n_lags + 1` and the centre cell is the zero lag.
///
/// Cells are stored row-major, `iy*size + ix`, with `ix`/`iy` increasing with
/// `h_x`/`h_y`. The map is symmetric by construction (`gamma(h) = gamma(-h)`).
#[derive(Debug, Clone)]
pub struct VariogramMap {
    /// Number of lag cells on each side of the origin.
    pub n_lags: usize,
    /// Cell size in distance units (the lag step on each axis).
    pub lag_width: f64,
    /// Grid side length, `2*n_lags + 1`.
    pub size: usize,
    /// Semivariance per cell (`NaN` where no pairs fell in the cell).
    pub gamma: Vec<f64>,
    /// Pair count per cell.
    pub n_pairs: Vec<usize>,
}

impl VariogramMap {
    /// Lag vector `(h_x, h_y)` at the centre of cell `(ix, iy)`.
    pub fn lag(&self, ix: usize, iy: usize) -> (f64, f64) {
        let c = self.n_lags as f64;
        (
            (ix as f64 - c) * self.lag_width,
            (iy as f64 - c) * self.lag_width,
        )
    }

    /// Semivariance at cell `(ix, iy)`.
    pub fn gamma_at(&self, ix: usize, iy: usize) -> f64 {
        self.gamma[iy * self.size + ix]
    }
}

#[derive(Clone)]
struct Acc {
    sq: Vec<f64>,
    n: Vec<usize>,
}

impl Acc {
    fn new(cells: usize) -> Self {
        Self {
            sq: vec![0.0; cells],
            n: vec![0; cells],
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for c in 0..self.sq.len() {
            self.sq[c] += other.sq[c];
            self.n[c] += other.n[c];
        }
        self
    }
}

/// Computes a 2-D variogram map with `n_lags` cells on each side of the origin
/// and a cell size of `lag_width` distance units. Each unordered pair is binned
/// at both `(h_x, h_y)` and `(-h_x, -h_y)`, so the surface is symmetric.
pub fn variogram_map(data: &PointSet<2>, n_lags: usize, lag_width: f64) -> Result<VariogramMap> {
    if data.len() < 2 {
        return Err(GeostatError::InsufficientData(
            "variogram map requires at least 2 points".into(),
        ));
    }
    if n_lags == 0 {
        return Err(GeostatError::InvalidParameter(
            "n_lags must be at least 1".into(),
        ));
    }
    if !(lag_width > 0.0) || !lag_width.is_finite() {
        return Err(GeostatError::InvalidParameter(format!(
            "lag_width must be finite and positive, got {lag_width}"
        )));
    }

    let coords = data.coords();
    let values = data.values();
    let n = coords.len();
    let size = 2 * n_lags + 1;
    let cells = size * size;
    let c = n_lags as i64;

    // Bins a lag vector; returns the flat cell index, or None if off the grid.
    let cell = |dx: f64, dy: f64| -> Option<usize> {
        let ix = (dx / lag_width).round() as i64 + c;
        let iy = (dy / lag_width).round() as i64 + c;
        if ix < 0 || iy < 0 || ix >= size as i64 || iy >= size as i64 {
            return None;
        }
        Some(iy as usize * size + ix as usize)
    };

    let chunks = n_chunks();
    let rows_per_chunk = n.div_ceil(chunks);
    let acc = par_map(chunks, |ch| {
        let mut acc = Acc::new(cells);
        let lo = ch * rows_per_chunk;
        let hi = ((ch + 1) * rows_per_chunk).min(n);
        for i in lo..hi {
            for j in (i + 1)..n {
                let dx = coords[j][0] - coords[i][0];
                let dy = coords[j][1] - coords[i][1];
                let semi = 0.5 * (values[i] - values[j]).powi(2);
                // Symmetric: bin both the lag and its negation.
                for (sx, sy) in [(dx, dy), (-dx, -dy)] {
                    if let Some(k) = cell(sx, sy) {
                        acc.sq[k] += semi;
                        acc.n[k] += 1;
                    }
                }
            }
        }
        acc
    })
    .into_iter()
    .fold(Acc::new(cells), |a, b| a.merge(b));

    let gamma = (0..cells)
        .map(|k| {
            if acc.n[k] > 0 {
                acc.sq[k] / acc.n[k] as f64
            } else {
                f64::NAN
            }
        })
        .collect();

    Ok(VariogramMap {
        n_lags,
        lag_width,
        size,
        gamma,
        n_pairs: acc.n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_config() {
        let d = PointSet::new(vec![[0.0, 0.0], [1.0, 0.0]], vec![0.0, 1.0]).unwrap();
        assert!(variogram_map(&d, 0, 1.0).is_err());
        assert!(variogram_map(&d, 3, -1.0).is_err());
        assert!(variogram_map(&d, 3, f64::NAN).is_err());
    }

    #[test]
    fn symmetric_and_centred() {
        // A few points; the map must be symmetric about the centre cell.
        let d = PointSet::new(
            vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            vec![0.0, 1.0, 2.0, 1.0, 2.0],
        )
        .unwrap();
        let m = variogram_map(&d, 2, 1.0).unwrap();
        assert_eq!(m.size, 5);
        let mid = m.n_lags;
        // Centre cell is the zero lag: no pair contributes there.
        assert_eq!(m.n_pairs[mid * m.size + mid], 0);
        // Symmetry gamma(h) == gamma(-h) and matching pair counts.
        for iy in 0..m.size {
            for ix in 0..m.size {
                let g = m.gamma_at(ix, iy);
                let g_opp = m.gamma_at(m.size - 1 - ix, m.size - 1 - iy);
                if g.is_finite() || g_opp.is_finite() {
                    assert!((g - g_opp).abs() < 1e-12, "asymmetry at ({ix},{iy})");
                }
                assert_eq!(
                    m.n_pairs[iy * m.size + ix],
                    m.n_pairs[(m.size - 1 - iy) * m.size + (m.size - 1 - ix)]
                );
            }
        }
    }

    #[test]
    fn matches_known_lag() {
        // Collinear points along x with values 0,1,2: lag (1,0) has pairs
        // (0,1) and (1,2) with semivariance 0.5 each; lag (2,0) has pair (0,2)
        // with semivariance 2.0.
        let d = PointSet::new(
            vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            vec![0.0, 1.0, 2.0],
        )
        .unwrap();
        let m = variogram_map(&d, 2, 1.0).unwrap();
        let mid = m.n_lags;
        // cell (ix=mid+1, iy=mid) is lag (+1, 0).
        assert_eq!(m.n_pairs[mid * m.size + (mid + 1)], 2);
        assert!((m.gamma_at(mid + 1, mid) - 0.5).abs() < 1e-12);
        // cell (ix=mid+2, iy=mid) is lag (+2, 0).
        assert_eq!(m.n_pairs[mid * m.size + (mid + 2)], 1);
        assert!((m.gamma_at(mid + 2, mid) - 2.0).abs() < 1e-12);
        let (hx, hy) = m.lag(mid + 1, mid);
        assert_eq!((hx, hy), (1.0, 0.0));
    }
}
