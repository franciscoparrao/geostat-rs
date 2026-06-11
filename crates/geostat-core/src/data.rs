//! Spatial point data containers and neighbor search.

use crate::error::{GeostatError, Result};

/// A set of 2-D sample points with one attribute value per point.
///
/// Coordinates are planar (projected CRS assumed); distances are Euclidean.
#[derive(Debug, Clone)]
pub struct PointSet {
    coords: Vec<[f64; 2]>,
    values: Vec<f64>,
}

impl PointSet {
    /// Builds a point set from coordinates and values of equal length.
    ///
    /// All coordinates and values must be finite.
    pub fn new(coords: Vec<[f64; 2]>, values: Vec<f64>) -> Result<Self> {
        if coords.is_empty() {
            return Err(GeostatError::InsufficientData("no points provided".into()));
        }
        if coords.len() != values.len() {
            return Err(GeostatError::DimensionMismatch(format!(
                "{} coordinates vs {} values",
                coords.len(),
                values.len()
            )));
        }
        if coords.iter().flatten().any(|v| !v.is_finite()) {
            return Err(GeostatError::InvalidParameter(
                "non-finite coordinate".into(),
            ));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(GeostatError::InvalidParameter("non-finite value".into()));
        }
        Ok(Self { coords, values })
    }

    /// Builds a point set from separate x, y and value slices.
    pub fn from_xyz(x: &[f64], y: &[f64], z: &[f64]) -> Result<Self> {
        if x.len() != y.len() || x.len() != z.len() {
            return Err(GeostatError::DimensionMismatch(format!(
                "x: {}, y: {}, z: {}",
                x.len(),
                y.len(),
                z.len()
            )));
        }
        let coords = x.iter().zip(y).map(|(&xi, &yi)| [xi, yi]).collect();
        Self::new(coords, z.to_vec())
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Whether the set is empty (never true for a constructed `PointSet`).
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// All coordinates.
    pub fn coords(&self) -> &[[f64; 2]] {
        &self.coords
    }

    /// All attribute values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Coordinate of point `i`.
    pub fn coord(&self, i: usize) -> [f64; 2] {
        self.coords[i]
    }

    /// Value of point `i`.
    pub fn value(&self, i: usize) -> f64 {
        self.values[i]
    }

    /// Arithmetic mean of the attribute values.
    pub fn mean(&self) -> f64 {
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Axis-aligned bounding box as `(min, max)` corners.
    pub fn bbox(&self) -> ([f64; 2], [f64; 2]) {
        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        for c in &self.coords {
            for d in 0..2 {
                min[d] = min[d].min(c[d]);
                max[d] = max[d].max(c[d]);
            }
        }
        (min, max)
    }

    /// A copy of this point set with point `i` removed (for cross-validation).
    pub fn excluding(&self, i: usize) -> Self {
        let mut coords = self.coords.clone();
        let mut values = self.values.clone();
        coords.remove(i);
        values.remove(i);
        Self { coords, values }
    }
}

/// Euclidean distance between two points.
pub fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointset_validates_input() {
        assert!(PointSet::new(vec![], vec![]).is_err());
        assert!(PointSet::new(vec![[0.0, 0.0]], vec![1.0, 2.0]).is_err());
        assert!(PointSet::new(vec![[f64::NAN, 0.0]], vec![1.0]).is_err());
        assert!(PointSet::new(vec![[0.0, 0.0]], vec![f64::INFINITY]).is_err());
        let ps = PointSet::new(vec![[0.0, 0.0], [1.0, 1.0]], vec![1.0, 2.0]).unwrap();
        assert_eq!(ps.len(), 2);
        assert!((ps.mean() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn bbox_and_excluding() {
        let ps = PointSet::new(
            vec![[0.0, 5.0], [2.0, 1.0], [1.0, 3.0]],
            vec![1.0, 2.0, 3.0],
        )
        .unwrap();
        let (min, max) = ps.bbox();
        assert_eq!(min, [0.0, 1.0]);
        assert_eq!(max, [2.0, 5.0]);
        let sub = ps.excluding(1);
        assert_eq!(sub.len(), 2);
        assert_eq!(sub.value(1), 3.0);
    }
}
