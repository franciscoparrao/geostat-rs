//! Regular 2-D grid definition for prediction and simulation targets.

use serde::{Deserialize, Serialize};

use crate::error::{GeostatError, Result};

/// A regular 2-D grid. `(x0, y0)` is the lower-left *edge* of the grid;
/// predictions are made at cell centers. Cells are stored row-major with
/// index `iy * nx + ix`, y increasing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Grid2D {
    /// X coordinate of the left edge.
    pub x0: f64,
    /// Y coordinate of the bottom edge.
    pub y0: f64,
    /// Cell width.
    pub dx: f64,
    /// Cell height.
    pub dy: f64,
    /// Number of columns.
    pub nx: usize,
    /// Number of rows.
    pub ny: usize,
}

impl Grid2D {
    /// Grid covering `[min, max]` with `nx` x `ny` cells.
    pub fn from_bbox(min: [f64; 2], max: [f64; 2], nx: usize, ny: usize) -> Result<Self> {
        if nx == 0 || ny == 0 {
            return Err(GeostatError::InvalidParameter(
                "grid must have at least one cell per axis".into(),
            ));
        }
        if !(max[0] > min[0]) || !(max[1] > min[1]) {
            return Err(GeostatError::InvalidParameter(format!(
                "degenerate bbox: min {min:?}, max {max:?}"
            )));
        }
        Ok(Self {
            x0: min[0],
            y0: min[1],
            dx: (max[0] - min[0]) / nx as f64,
            dy: (max[1] - min[1]) / ny as f64,
            nx,
            ny,
        })
    }

    /// Grid covering `[min, max]` with square cells of side `res`
    /// (the extent is expanded to a whole number of cells).
    pub fn with_resolution(min: [f64; 2], max: [f64; 2], res: f64) -> Result<Self> {
        if !(res > 0.0) {
            return Err(GeostatError::InvalidParameter(format!(
                "resolution must be positive, got {res}"
            )));
        }
        if !(max[0] > min[0]) || !(max[1] > min[1]) {
            return Err(GeostatError::InvalidParameter(format!(
                "degenerate bbox: min {min:?}, max {max:?}"
            )));
        }
        let nx = ((max[0] - min[0]) / res).ceil().max(1.0) as usize;
        let ny = ((max[1] - min[1]) / res).ceil().max(1.0) as usize;
        Ok(Self {
            x0: min[0],
            y0: min[1],
            dx: res,
            dy: res,
            nx,
            ny,
        })
    }

    /// Total number of cells.
    pub fn n_cells(&self) -> usize {
        self.nx * self.ny
    }

    /// Center of cell `(ix, iy)`.
    pub fn center(&self, ix: usize, iy: usize) -> [f64; 2] {
        [
            self.x0 + (ix as f64 + 0.5) * self.dx,
            self.y0 + (iy as f64 + 0.5) * self.dy,
        ]
    }

    /// All cell centers in storage order (`iy * nx + ix`).
    pub fn centers(&self) -> Vec<[f64; 2]> {
        let mut out = Vec::with_capacity(self.n_cells());
        for iy in 0..self.ny {
            for ix in 0..self.nx {
                out.push(self.center(ix, iy));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centers_order_and_count() {
        let g = Grid2D::from_bbox([0.0, 0.0], [4.0, 2.0], 4, 2).unwrap();
        assert_eq!(g.n_cells(), 8);
        let c = g.centers();
        assert_eq!(c.len(), 8);
        assert_eq!(c[0], [0.5, 0.5]);
        assert_eq!(c[1], [1.5, 0.5]);
        assert_eq!(c[4], [0.5, 1.5]);
        assert_eq!(c[7], [3.5, 1.5]);
    }

    #[test]
    fn resolution_grid() {
        let g = Grid2D::with_resolution([0.0, 0.0], [10.0, 5.0], 2.5).unwrap();
        assert_eq!((g.nx, g.ny), (4, 2));
        assert!(Grid2D::with_resolution([0.0, 0.0], [1.0, 1.0], 0.0).is_err());
        assert!(Grid2D::from_bbox([0.0, 0.0], [0.0, 1.0], 2, 2).is_err());
    }
}

/// A regular 3-D grid. `(x0, y0, z0)` is the lower corner *edge*; cells are
/// stored with index `(iz * ny + iy) * nx + ix` (x fastest, then y, then z).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Grid3D {
    /// X coordinate of the lower edge.
    pub x0: f64,
    /// Y coordinate of the lower edge.
    pub y0: f64,
    /// Z coordinate of the lower edge.
    pub z0: f64,
    /// Cell width.
    pub dx: f64,
    /// Cell depth.
    pub dy: f64,
    /// Cell height.
    pub dz: f64,
    /// Number of cells in X.
    pub nx: usize,
    /// Number of cells in Y.
    pub ny: usize,
    /// Number of cells in Z.
    pub nz: usize,
}

impl Grid3D {
    /// Grid covering `[min, max]` with `nx` x `ny` x `nz` cells.
    pub fn from_bbox(
        min: [f64; 3],
        max: [f64; 3],
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<Self> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(GeostatError::InvalidParameter(
                "grid must have at least one cell per axis".into(),
            ));
        }
        for d in 0..3 {
            if !(max[d] > min[d]) {
                return Err(GeostatError::InvalidParameter(format!(
                    "degenerate bbox: min {min:?}, max {max:?}"
                )));
            }
        }
        Ok(Self {
            x0: min[0],
            y0: min[1],
            z0: min[2],
            dx: (max[0] - min[0]) / nx as f64,
            dy: (max[1] - min[1]) / ny as f64,
            dz: (max[2] - min[2]) / nz as f64,
            nx,
            ny,
            nz,
        })
    }

    /// Total number of cells.
    pub fn n_cells(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// All cell centers in storage order.
    pub fn centers(&self) -> Vec<[f64; 3]> {
        let mut out = Vec::with_capacity(self.n_cells());
        for iz in 0..self.nz {
            for iy in 0..self.ny {
                for ix in 0..self.nx {
                    out.push([
                        self.x0 + (ix as f64 + 0.5) * self.dx,
                        self.y0 + (iy as f64 + 0.5) * self.dy,
                        self.z0 + (iz as f64 + 0.5) * self.dz,
                    ]);
                }
            }
        }
        out
    }
}
