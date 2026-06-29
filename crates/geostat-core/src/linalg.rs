//! Minimal dense linear algebra: LU solve with partial pivoting.
//!
//! Kriging systems with Lagrange multipliers are symmetric but indefinite,
//! so Cholesky is not applicable; LU with partial pivoting is robust and
//! avoids pulling in a LAPACK dependency.

use ndarray::Array2;

use crate::error::{GeostatError, Result};

/// An LU factorization with partial pivoting, reusable across many
/// right-hand sides (the same system matrix, different `b`). This is what makes
/// global-neighbourhood kriging fast: the system matrix is identical for every
/// target, so it is factored once and only back-substituted per target.
#[derive(Debug, Clone)]
pub struct Lu {
    /// Combined factors: the strict lower triangle holds the L multipliers
    /// (unit diagonal implied), the upper triangle (incl. diagonal) holds U.
    lu: Array2<f64>,
    /// Row swapped with row `k` at elimination step `k`.
    piv: Vec<usize>,
    n: usize,
}

/// Factorizes `A = P L U` in place via LU with partial pivoting.
#[allow(clippy::needless_range_loop)]
pub fn lu_factor(mut a: Array2<f64>) -> Result<Lu> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(GeostatError::DimensionMismatch(format!(
            "A is {}x{}, must be square",
            a.nrows(),
            a.ncols()
        )));
    }
    let scale = a
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    let mut piv = vec![0usize; n];

    for k in 0..n {
        // Partial pivoting.
        let mut p = k;
        let mut max = a[[k, k]].abs();
        for i in (k + 1)..n {
            let v = a[[i, k]].abs();
            if v > max {
                max = v;
                p = i;
            }
        }
        if max < 1e-12 * scale {
            return Err(GeostatError::SingularSystem(format!(
                "pivot {max:.3e} at column {k} (matrix scale {scale:.3e})"
            )));
        }
        piv[k] = p;
        if p != k {
            for j in 0..n {
                a.swap([k, j], [p, j]);
            }
        }
        let pivot = a[[k, k]];
        for i in (k + 1)..n {
            let f = a[[i, k]] / pivot;
            a[[i, k]] = f; // store the multiplier in the L part
            if f != 0.0 {
                for j in (k + 1)..n {
                    a[[i, j]] -= f * a[[k, j]];
                }
            }
        }
    }
    Ok(Lu { lu: a, piv, n })
}

impl Lu {
    /// Log of the absolute value of the determinant, `ln|det A| = Σ ln|U_ii|`.
    /// For a symmetric positive-definite matrix this is `ln(det A)`.
    pub fn ln_det_abs(&self) -> f64 {
        (0..self.n).map(|i| self.lu[[i, i]].abs().ln()).sum()
    }

    /// Solves `A x = b` by reusing the factorization. `b` must have length `n`.
    #[allow(clippy::needless_range_loop)]
    pub fn solve(&self, mut b: Vec<f64>) -> Vec<f64> {
        let n = self.n;
        debug_assert_eq!(b.len(), n);
        // Apply the row swaps recorded during factorization.
        for k in 0..n {
            b.swap(k, self.piv[k]);
        }
        // Forward substitution (L, unit diagonal).
        for i in 0..n {
            let mut s = b[i];
            for j in 0..i {
                s -= self.lu[[i, j]] * b[j];
            }
            b[i] = s;
        }
        // Back substitution (U).
        for i in (0..n).rev() {
            let mut s = b[i];
            for j in (i + 1)..n {
                s -= self.lu[[i, j]] * b[j];
            }
            b[i] = s / self.lu[[i, i]];
        }
        b
    }
}

/// Solves the dense system `A x = b` via LU with partial pivoting.
pub fn solve(a: Array2<f64>, b: Vec<f64>) -> Result<Vec<f64>> {
    let n = a.nrows();
    if b.len() != n {
        return Err(GeostatError::DimensionMismatch(format!(
            "A is {}x{}, b has length {}",
            a.nrows(),
            a.ncols(),
            b.len()
        )));
    }
    Ok(lu_factor(a)?.solve(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn solves_known_system() {
        let a = array![[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]];
        let b = vec![8.0, -11.0, -3.0];
        let x = solve(a, b).unwrap();
        let expected = [2.0, 3.0, -1.0];
        for (xi, ei) in x.iter().zip(expected) {
            assert!((xi - ei).abs() < 1e-12, "{xi} vs {ei}");
        }
    }

    #[test]
    fn detects_singular() {
        let a = array![[1.0, 2.0], [2.0, 4.0]];
        let b = vec![1.0, 2.0];
        assert!(matches!(solve(a, b), Err(GeostatError::SingularSystem(_))));
    }

    #[test]
    fn solves_indefinite_kriging_like_system() {
        // Symmetric indefinite (ordinary-kriging-like with Lagrange row).
        let a = array![[1.0, 0.3, 1.0], [0.3, 1.0, 1.0], [1.0, 1.0, 0.0]];
        let b = vec![0.7, 0.5, 1.0];
        let x = solve(a.clone(), b.clone()).unwrap();
        // Check residual.
        for i in 0..3 {
            let r: f64 = (0..3).map(|j| a[[i, j]] * x[j]).sum::<f64>() - b[i];
            assert!(r.abs() < 1e-12);
        }
        // Weights sum to 1 (unbiasedness row).
        assert!((x[0] + x[1] - 1.0).abs() < 1e-12);
    }
}
