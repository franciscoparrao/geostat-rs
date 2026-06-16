//! Small derivative-free optimizers shared across the crate (variogram
//! fitting, marginal-transport fitting).

/// Standard Nelder–Mead simplex minimization. Returns the best parameter
/// vector found and its objective value.
pub(crate) fn nelder_mead<F>(f: F, x0: &[f64], step: f64, max_iter: usize) -> (Vec<f64>, f64)
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
}
