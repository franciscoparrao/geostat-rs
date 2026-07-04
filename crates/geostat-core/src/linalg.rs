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

/// Cholesky-factorizes the symmetric positive-definite `n x n` matrix `a`
/// (row-major, only the lower triangle is read) in place: on success, `a`'s
/// lower triangle (including the diagonal) holds `L` such that `A = L Lᵀ`.
///
/// Split out from [`cholesky_solve_in_place`] so callers that need the factor
/// itself (e.g.\ grouped Vecchia likelihoods, which read off `L`'s diagonal
/// and forward-solve several right-hand sides against one factorization) do
/// not pay for a back-substitution they do not need. Fails with
/// `SingularSystem` when a pivot is not meaningfully positive.
pub fn cholesky_factor_in_place(a: &mut [f64], n: usize) -> Result<()> {
    if a.len() != n * n {
        return Err(GeostatError::DimensionMismatch(format!(
            "A has {} entries, expected {}",
            a.len(),
            n * n
        )));
    }
    let scale = (0..n)
        .map(|i| a[i * n + i])
        .fold(0.0_f64, f64::max)
        .max(f64::MIN_POSITIVE);
    for k in 0..n {
        let mut d = a[k * n + k];
        for j in 0..k {
            d -= a[k * n + j] * a[k * n + j];
        }
        if !(d > 1e-12 * scale) {
            return Err(GeostatError::SingularSystem(format!(
                "non-positive Cholesky pivot {d:.3e} at column {k} (scale {scale:.3e})"
            )));
        }
        let d = d.sqrt();
        a[k * n + k] = d;
        for i in (k + 1)..n {
            let mut s = a[i * n + k];
            for j in 0..k {
                s -= a[i * n + j] * a[k * n + j];
            }
            a[i * n + k] = s / d;
        }
    }
    Ok(())
}

/// Forward substitution `L y = b` against a factor from
/// [`cholesky_factor_in_place`]; overwrites `b` with `y`.
pub fn cholesky_forward_solve(a: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= a[i * n + j] * b[j];
        }
        b[i] = s / a[i * n + i];
    }
}

/// Back substitution `Lᵀ x = y` against a factor from
/// [`cholesky_factor_in_place`]; overwrites `b` (holding `y`) with `x`.
pub fn cholesky_back_solve(a: &[f64], n: usize, b: &mut [f64]) {
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[j * n + i] * b[j];
        }
        b[i] = s / a[i * n + i];
    }
}

/// Solves the symmetric positive-definite system `A x = b` in place:
/// `a` holds the row-major `n x n` matrix (only the lower triangle is read;
/// it is overwritten with the Cholesky factor) and `b` is overwritten with
/// the solution.
///
/// Built for hot loops: both buffers are caller-owned and reusable across
/// calls, and the unpivoted Cholesky factorization costs half the flops of
/// the LU path (covariance blocks in Vecchia and sequential simulation are
/// SPD, so no pivoting is needed). Fails with `SingularSystem` when a pivot
/// is not meaningfully positive.
pub fn cholesky_solve_in_place(a: &mut [f64], n: usize, b: &mut [f64]) -> Result<()> {
    if b.len() != n {
        return Err(GeostatError::DimensionMismatch(format!(
            "b has {} entries, expected {n}",
            b.len()
        )));
    }
    cholesky_factor_in_place(a, n)?;
    cholesky_forward_solve(a, n, b);
    cholesky_back_solve(a, n, b);
    Ok(())
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
    fn cholesky_matches_lu_on_spd_and_rejects_non_pd() {
        // SPD covariance-like matrix.
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let b = [1.0, 2.0, 3.0];
        let mut aw = a;
        let mut bw = b;
        cholesky_solve_in_place(&mut aw, 3, &mut bw).unwrap();
        let a_nd = Array2::from_shape_vec((3, 3), a.to_vec()).unwrap();
        let x_lu = solve(a_nd, b.to_vec()).unwrap();
        for (c, l) in bw.iter().zip(&x_lu) {
            assert!((c - l).abs() < 1e-12, "{c} vs {l}");
        }
        // Semi-definite (rank 1) rejected.
        let mut sd = [1.0, 2.0, 2.0, 4.0];
        let mut b2 = [1.0, 2.0];
        assert!(matches!(
            cholesky_solve_in_place(&mut sd, 2, &mut b2),
            Err(GeostatError::SingularSystem(_))
        ));
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

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        /// A random well-conditioned SPD system `(A, b)`: `A = MᵀM +
        /// 0.5·I` is SPD for any `M` (Gram matrix plus a positive diagonal
        /// shift keeps it away from singular), so `solve` should always
        /// succeed and reproduce `b`.
        fn spd_system() -> impl Strategy<Value = (Array2<f64>, Vec<f64>)> {
            (2usize..6).prop_flat_map(|n| {
                (
                    prop::collection::vec(-2.0f64..2.0, n * n),
                    prop::collection::vec(-5.0f64..5.0, n),
                    Just(n),
                )
                    .prop_map(|(raw, b, n)| {
                        let m = Array2::from_shape_vec((n, n), raw).unwrap();
                        let a = m.t().dot(&m) + Array2::<f64>::eye(n) * 0.5;
                        (a, b)
                    })
            })
        }

        proptest! {
            #[test]
            fn solve_reproduces_the_right_hand_side((a, b) in spd_system()) {
                let n = b.len();
                let x = solve(a.clone(), b.clone()).unwrap();
                for i in 0..n {
                    let r: f64 = (0..n).map(|j| a[[i, j]] * x[j]).sum::<f64>() - b[i];
                    prop_assert!(r.abs() < 1e-6, "residual {r} at row {i}");
                }
            }
        }
    }
}
