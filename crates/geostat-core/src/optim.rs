//! Small derivative-free optimizers shared across the crate (variogram
//! fitting) and by downstream crates (e.g. marginal-transport fitting).

/// Standard Nelder–Mead simplex minimization. Returns the best parameter
/// vector found and its objective value.
pub fn nelder_mead<F>(f: F, x0: &[f64], step: f64, max_iter: usize) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    const ALPHA: f64 = 1.0; // reflection
    const GAMMA: f64 = 2.0; // expansion
    const RHO: f64 = 0.5; // contraction
    const SIGMA: f64 = 0.5; // shrink

    let n = x0.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut p = x0.to_vec();
        p[i] += step;
        simplex.push(p);
    }
    let mut fv: Vec<f64> = simplex.iter().map(|p| f(p)).collect();

    for _ in 0..max_iter {
        // Order simplex by objective value.
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| fv[a].total_cmp(&fv[b]));
        simplex = idx.iter().map(|&i| simplex[i].clone()).collect();
        fv = idx.iter().map(|&i| fv[i]).collect();

        if (fv[n] - fv[0]).abs() <= 1e-12 * fv[0].abs().max(1e-12) {
            break;
        }

        let mut centroid = vec![0.0; n];
        for p in &simplex[..n] {
            for j in 0..n {
                centroid[j] += p[j] / n as f64;
            }
        }

        let xr: Vec<f64> = (0..n)
            .map(|j| centroid[j] + ALPHA * (centroid[j] - simplex[n][j]))
            .collect();
        let fr = f(&xr);

        if fr < fv[0] {
            let xe: Vec<f64> = (0..n)
                .map(|j| centroid[j] + GAMMA * (centroid[j] - simplex[n][j]))
                .collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fv[n] = fe;
            } else {
                simplex[n] = xr;
                fv[n] = fr;
            }
        } else if fr < fv[n - 1] {
            simplex[n] = xr;
            fv[n] = fr;
        } else {
            let (xc, fc) = if fr < fv[n] {
                let xc: Vec<f64> = (0..n)
                    .map(|j| centroid[j] + RHO * (xr[j] - centroid[j]))
                    .collect();
                let fc = f(&xc);
                (xc, fc)
            } else {
                let xc: Vec<f64> = (0..n)
                    .map(|j| centroid[j] - RHO * (centroid[j] - simplex[n][j]))
                    .collect();
                let fc = f(&xc);
                (xc, fc)
            };
            if fc < fr.min(fv[n]) {
                simplex[n] = xc;
                fv[n] = fc;
            } else {
                let best = simplex[0].clone();
                for i in 1..=n {
                    for (pj, &bj) in simplex[i].iter_mut().zip(&best) {
                        *pj = bj + SIGMA * (*pj - bj);
                    }
                    fv[i] = f(&simplex[i]);
                }
            }
        }
    }

    let mut best = 0;
    for i in 1..=n {
        if fv[i] < fv[best] {
            best = i;
        }
    }
    (simplex[best].clone(), fv[best])
}

/// Runs [`nelder_mead`] from each of `starts` and keeps the best optimum
/// found. Covariance log-likelihoods and WLS variogram objectives can be
/// multimodal (range/sill trade-offs, periodic azimuth); a single starting
/// simplex can converge to a local optimum that a different start avoids.
/// `starts` must be non-empty.
pub fn nelder_mead_multistart<F>(
    f: F,
    starts: &[Vec<f64>],
    step: f64,
    max_iter: usize,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let mut best: Option<(Vec<f64>, f64)> = None;
    for x0 in starts {
        let (x, fx) = nelder_mead(&f, x0, step, max_iter);
        if best.as_ref().is_none_or(|(_, bf)| fx < *bf) {
            best = Some((x, fx));
        }
    }
    best.expect("nelder_mead_multistart: `starts` must be non-empty")
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn identity_flat(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|idx| if idx % (n + 1) == 0 { 1.0 } else { 0.0 })
        .collect()
}

/// `-H * v` for a flat row-major `n x n` matrix `h`.
fn neg_mat_vec(h: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| -(0..n).map(|j| h[i * n + j] * v[j]).sum::<f64>())
        .collect()
}

/// Outer product `u * vᵀ`, flat row-major `n x n`.
fn outer(u: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = u[i] * v[j];
        }
    }
    m
}

fn mat_mat(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// BFGS inverse-Hessian update (Nocedal & Wright, eq. 6.17):
/// `H' = (I - rho*s*yᵀ) H (I - rho*y*sᵀ) + rho*s*sᵀ`, `rho = 1/(sᵀy)`.
fn bfgs_update(h: &[f64], s: &[f64], y: &[f64], sy: f64, n: usize) -> Vec<f64> {
    let rho = 1.0 / sy;
    let mut left = identity_flat(n);
    let syt = outer(s, y, n);
    for i in 0..n * n {
        left[i] -= rho * syt[i];
    }
    let mut right = identity_flat(n);
    let yst = outer(y, s, n);
    for i in 0..n * n {
        right[i] -= rho * yst[i];
    }
    let mut h_new = mat_mat(&mat_mat(&left, h, n), &right, n);
    let sst = outer(s, s, n);
    for i in 0..n * n {
        h_new[i] += rho * sst[i];
    }
    h_new
}

/// BFGS quasi-Newton minimization with a backtracking (Armijo) line search,
/// using caller-supplied analytical gradients. Converges superlinearly on
/// smooth objectives -- typically a few dozen evaluations for a handful of
/// parameters, instead of the thousands [`nelder_mead`] needs -- the
/// "cheaper alternative" AUDIT-2026-07-v2.md §5.1 calls out for Vecchia
/// MLE/REML once analytical gradients are available. Falls back to plain
/// steepest descent for one step whenever the current Hessian approximation
/// stops being a descent direction (can happen after an ill-conditioned
/// curvature update), rather than failing outright.
///
/// Stops on **either** a small gradient norm **or** negligible progress
/// (relative function-value change and step size both tiny): a purely
/// gradient-norm stopping test can plateau just above its threshold near a
/// flat optimum (observed empirically: a Vecchia log-likelihood fit whose
/// gradient settled at ~4e-7, just above a naive `1e-10` cutoff, burning
/// through the entire `max_iter` budget re-doing a full backtracking line
/// search every iteration for zero further improvement).
pub fn bfgs<F, G>(f: F, grad: G, x0: &[f64], max_iter: usize) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut h = identity_flat(n);
    let mut g = grad(&x);
    let mut fx = f(&x);

    for _ in 0..max_iter {
        let gnorm = dot(&g, &g).sqrt();
        if gnorm < 1e-10 {
            break;
        }
        let mut p = neg_mat_vec(&h, &g, n);
        let mut slope = dot(&g, &p);
        if slope >= 0.0 {
            h = identity_flat(n);
            p = g.iter().map(|&v| -v).collect();
            slope = dot(&g, &p);
        }
        let c1 = 1e-4;
        let mut alpha = 1.0_f64;
        let (x_new, f_new) = loop {
            let cand: Vec<f64> = x.iter().zip(&p).map(|(&xi, &pi)| xi + alpha * pi).collect();
            let f_cand = f(&cand);
            if f_cand <= fx + c1 * alpha * slope || alpha < 1e-12 {
                break (cand, f_cand);
            }
            alpha *= 0.5;
        };
        let g_new = grad(&x_new);
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(&a, &b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(&a, &b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-10 {
            h = bfgs_update(&h, &s, &y, sy, n);
        }
        let stalled = (fx - f_new).abs() <= 1e-13 * (1.0 + fx.abs())
            && dot(&s, &s).sqrt() <= 1e-10 * (1.0 + dot(&x, &x).sqrt());
        x = x_new;
        g = g_new;
        fx = f_new;
        if stalled {
            break;
        }
    }
    (x, fx)
}

/// Runs [`bfgs`] from each of `starts` and keeps the best optimum found
/// (same rationale as [`nelder_mead_multistart`]: covariance
/// log-likelihoods are multimodal in the range parameter). `starts` must be
/// non-empty.
pub fn bfgs_multistart<F, G>(f: F, grad: G, starts: &[Vec<f64>], max_iter: usize) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let mut best: Option<(Vec<f64>, f64)> = None;
    for x0 in starts {
        let (x, fx) = bfgs(&f, &grad, x0, max_iter);
        if best.as_ref().is_none_or(|(_, bf)| fx < *bf) {
            best = Some((x, fx));
        }
    }
    best.expect("bfgs_multistart: `starts` must be non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimizes_rosenbrock() {
        // Rosenbrock has its minimum at (1, 1) with value 0.
        let f = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let (x, fx) = nelder_mead(f, &[-1.2, 1.0], 0.1, 5000);
        assert!((x[0] - 1.0).abs() < 1e-3, "x0 = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-3, "x1 = {}", x[1]);
        assert!(fx < 1e-6);
    }

    #[test]
    fn multistart_escapes_a_local_optimum() {
        // Two wells of different depth; a start near the shallow one alone
        // would converge to it (single-start nelder_mead does).
        let f = |x: &[f64]| {
            let a = x[0] + 3.0;
            let b = x[0] - 3.0;
            -2.0 * (-0.5 * a * a).exp() - 3.0 * (-0.5 * b * b).exp()
        };
        let (_, single) = nelder_mead(f, &[-3.0], 0.1, 500);
        assert!(single > -2.5, "expected the shallow well, got {single}");

        let starts = vec![vec![-3.0], vec![3.0], vec![0.0]];
        let (x, fx) = nelder_mead_multistart(f, &starts, 0.1, 500);
        assert!((x[0] - 3.0).abs() < 1e-2, "x0 = {}", x[0]);
        assert!(fx < -2.9, "fx = {fx}");
    }

    #[test]
    fn bfgs_minimizes_rosenbrock_in_far_fewer_evaluations_than_nelder_mead() {
        let f = |x: &[f64]| {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            a * a + 100.0 * b * b
        };
        let grad = |x: &[f64]| -> Vec<f64> {
            let a = 1.0 - x[0];
            let b = x[1] - x[0] * x[0];
            vec![-2.0 * a - 400.0 * x[0] * b, 200.0 * b]
        };
        let (x, fx) = bfgs(f, grad, &[-1.2, 1.0], 200);
        assert!((x[0] - 1.0).abs() < 1e-4, "x0 = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x1 = {}", x[1]);
        assert!(fx < 1e-8, "fx = {fx}");

        // The whole point: BFGS should reach a tighter optimum than
        // Nelder-Mead within the same small iteration budget.
        let (nm_x, _) = nelder_mead(f, &[-1.2, 1.0], 0.1, 30);
        let (bfgs_x, _) = bfgs(f, grad, &[-1.2, 1.0], 30);
        let nm_err = ((nm_x[0] - 1.0).powi(2) + (nm_x[1] - 1.0).powi(2)).sqrt();
        let bfgs_err = ((bfgs_x[0] - 1.0).powi(2) + (bfgs_x[1] - 1.0).powi(2)).sqrt();
        assert!(
            bfgs_err < nm_err,
            "expected BFGS to converge faster within 30 iterations: bfgs_err={bfgs_err} \
             nm_err={nm_err}"
        );
    }

    #[test]
    fn bfgs_multistart_escapes_a_local_optimum() {
        // Same two-well objective as `multistart_escapes_a_local_optimum`.
        let f = |x: &[f64]| {
            let a = x[0] + 3.0;
            let b = x[0] - 3.0;
            -2.0 * (-0.5 * a * a).exp() - 3.0 * (-0.5 * b * b).exp()
        };
        let grad = |x: &[f64]| -> Vec<f64> {
            let a = x[0] + 3.0;
            let b = x[0] - 3.0;
            vec![2.0 * a * (-0.5 * a * a).exp() + 3.0 * b * (-0.5 * b * b).exp()]
        };
        let (single, single_fx) = bfgs(f, grad, &[-3.0], 100);
        assert!(
            single_fx > -2.5,
            "expected the shallow well, got {single_fx} at {single:?}"
        );

        let starts = vec![vec![-3.0], vec![3.0], vec![0.0]];
        let (x, fx) = bfgs_multistart(f, grad, &starts, 100);
        assert!((x[0] - 3.0).abs() < 1e-3, "x0 = {}", x[0]);
        assert!(fx < -2.9, "fx = {fx}");
    }

    #[test]
    fn bfgs_matches_finite_difference_gradient_on_a_quadratic() {
        // Sanity check for the BFGS machinery itself against a well-behaved
        // convex quadratic with a known minimum.
        let center = [2.0, -1.0, 0.5];
        let f = |x: &[f64]| (0..3).map(|i| (x[i] - center[i]).powi(2)).sum::<f64>();
        let grad = |x: &[f64]| -> Vec<f64> { (0..3).map(|i| 2.0 * (x[i] - center[i])).collect() };
        let (x, fx) = bfgs(f, grad, &[0.0, 0.0, 0.0], 100);
        for i in 0..3 {
            assert!((x[i] - center[i]).abs() < 1e-6, "x[{i}] = {}", x[i]);
        }
        assert!(fx < 1e-10);
    }
}
