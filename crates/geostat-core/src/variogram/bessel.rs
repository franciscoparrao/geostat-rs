//! Modified Bessel function of the second kind (`K_ν`) and the Gamma
//! function, needed for the continuous-smoothness Matérn covariance
//! (`ModelKind::Matern`). No external special-function crate covers `K_ν`
//! for real, non-half-integer `ν` without pulling in a heavier dependency
//! (the project's WASM target rules out `libc`-backed alternatives), so both
//! are implemented in-house, matching how the crate already implements its
//! own RNG and optimizer.
//!
//! `K_ν(x)` is evaluated via its integral representation
//! `K_ν(x) = ∫₀^∞ exp(-x cosh t) cosh(ν t) dt` (valid for any real `ν`,
//! `x > 0`, no special-casing for integer/half-integer order), using
//! composite Gauss–Legendre quadrature on `[0, 1] ∪ [1, T]` (60 + 40 nodes —
//! sized down from an initial 120 + 40 after profiling showed the joint
//! (nugget, sill, range, ν) Vecchia MLE fit, which evaluates `K_ν` inside
//! every neighbourhood pair of every likelihood call, was impractically slow
//! at the higher node count; 100 nodes keeps <=~2e-8 relative error, far
//! below what fitting itself needs). The integrand is smooth and strictly
//! positive (no cancellation), and `T` is chosen so the tail past it is
//! numerically negligible. Validated against R's `besselK` across `ν ∈
//! [0.05, 15]`, `x ∈ [1e-6, 150]` to a relative error ≤ ~2e-8 (≤ ~1e-9 over
//! the practically relevant fitting range); see
//! `variogram::model::tests::matern_matches_closed_form_special_cases`.

use std::sync::OnceLock;

/// Gauss–Legendre nodes/weights on `[-1, 1]` via Newton iteration on
/// Legendre polynomials (textbook algorithm, e.g. Numerical Recipes
/// `gauleg`).
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let m = n.div_ceil(2);
    for i in 0..m {
        let mut z = ((std::f64::consts::PI * (i as f64 + 0.75)) / (n as f64 + 0.5)).cos();
        loop {
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j as f64 + 1.0) * z * p2 - j as f64 * p3) / (j as f64 + 1.0);
            }
            let pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z -= p1 / pp;
            if (z - z1).abs() <= 1e-15 {
                w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
                x[i] = -z;
                w[n - 1 - i] = w[i];
                x[n - 1 - i] = z;
                break;
            }
        }
    }
    (x, w)
}

fn gl_near() -> &'static (Vec<f64>, Vec<f64>) {
    static CELL: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    CELL.get_or_init(|| gauss_legendre(60))
}

fn gl_far() -> &'static (Vec<f64>, Vec<f64>) {
    static CELL: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    CELL.get_or_init(|| gauss_legendre(40))
}

/// `K_ν(x)`, the modified Bessel function of the second kind, for `x > 0`
/// and any real `ν` (`K_{-ν} = K_ν`). See the module docs for the method.
pub(crate) fn bessel_k(nu: f64, x: f64) -> f64 {
    let nu = nu.abs();
    let f = |t: f64| (-x * t.cosh()).exp() * (nu * t).cosh();

    // Panel [0, 1]: covers the integrand's peak for any x in the range this
    // crate evaluates (h/a scaled by sqrt(2*nu), practically well under 1e2).
    let (gx1, gw1) = gl_near();
    let mut integral = 0.0;
    for i in 0..gx1.len() {
        let t = 0.5 * (gx1[i] + 1.0);
        integral += gw1[i] * f(t);
    }
    integral *= 0.5;

    // Panel [1, T]: T chosen so exp(-x cosh T) is negligible (~1e-26).
    let t_max = (60.0 / x.max(1e-12)).acosh().max(4.0) + 1.0;
    let (gx2, gw2) = gl_far();
    let mut integral2 = 0.0;
    for i in 0..gx2.len() {
        let t = 1.0 + 0.5 * (t_max - 1.0) * (gx2[i] + 1.0);
        integral2 += gw2[i] * f(t);
    }
    integral2 *= 0.5 * (t_max - 1.0);

    integral + integral2
}

/// The Gamma function via the Lanczos approximation (`g=7`, `n=9`), accurate
/// to machine precision for `x` away from the negative real axis. Uses the
/// reflection formula for `x < 0.5`.
pub(crate) fn gamma(x: f64) -> f64 {
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x))
    } else {
        let xm1 = x - 1.0;
        let mut a = COEF[0];
        let t = xm1 + 7.5;
        for (i, &c) in COEF.iter().enumerate().skip(1) {
            a += c / (xm1 + i as f64);
        }
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(xm1 + 0.5) * (-t).exp() * a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_matches_known_values() {
        let cases = [
            (0.5, std::f64::consts::PI.sqrt()),
            (1.0, 1.0),
            (2.0, 1.0),
            (3.0, 2.0),
            (4.5, 11.631_728_396_567_4),
            (0.1, 9.513_507_698_668_73),
        ];
        for (x, expected) in cases {
            let got = gamma(x);
            assert!(
                ((got - expected) / expected).abs() < 1e-9,
                "gamma({x}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn bessel_k_matches_r_besselk_reference() {
        // Spot-checked against R's besselK(x, nu) (base R, no package).
        let cases = [
            (0.5_f64, 1.0_f64, 0.461_068_504_447_895),
            (1.5, 1.0, 0.922_137_008_895_789),
            (2.5, 2.0, 0.389_797_758_896_2),
            (0.3, 5.0, 0.003_721_669_328_873_42),
            (4.0, 10.0, 3.786_143_716_089_2e-5),
        ];
        for (nu, x, expected) in cases {
            let got = bessel_k(nu, x);
            let rel = ((got - expected) / expected).abs();
            assert!(
                rel < 1e-8,
                "K_{nu}({x}) = {got}, expected {expected} (rel {rel})"
            );
        }
    }
}
