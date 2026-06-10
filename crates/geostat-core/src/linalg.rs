//! Minimal dense linear algebra: LU solve with partial pivoting.
//!
//! Kriging systems with Lagrange multipliers are symmetric but indefinite,
//! so Cholesky is not applicable; LU with partial pivoting is robust and
//! avoids pulling in a LAPACK dependency.

use ndarray::Array2;

use crate::error::{GeostatError, Result};

/// Solves the dense system `A x = b` in place via LU with partial pivoting.
#[allow(clippy::needless_range_loop)]
pub fn solve(mut a: Array2<f64>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(GeostatError::DimensionMismatch(format!(
            "A is {}x{}, b has length {}",
            a.nrows(),
            a.ncols(),
            b.len()
        )));
    }
    let scale = a
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);

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
        if p != k {
            for j in 0..n {
                a.swap([k, j], [p, j]);
            }
            b.swap(k, p);
        }
        let pivot = a[[k, k]];
        for i in (k + 1)..n {
            let f = a[[i, k]] / pivot;
            if f != 0.0 {
                for j in (k + 1)..n {
                    a[[i, j]] -= f * a[[k, j]];
                }
                b[i] -= f * b[k];
            }
            a[[i, k]] = 0.0;
        }
    }

    // Back substitution.
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[[i, j]] * x[j];
        }
        x[i] = s / a[[i, i]];
    }
    Ok(x)
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
