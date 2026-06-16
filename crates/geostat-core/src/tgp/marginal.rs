//! Marginal transport maps: learnable monotone warpings of the data axis.
//!
//! These are the marginal component of a Transport Gaussian Process (Rios &
//! Tobar, 2019): a monotone map `T` such that the warped data `T(z)` is
//! approximately a standard Gaussian. They generalize the fixed log /
//! normal-score warpings already in the crate to parametric, learnable
//! families fitted by maximum likelihood (Gaussian anamorphosis with the
//! Jacobian term).
//!
//! Each transform exposes `forward` (z -> latent), `inverse` (latent -> z)
//! and `log_grad` (`ln|dT/dz|`), the last entering the likelihood so the fit
//! is not degenerate.

use crate::error::{GeostatError, Result};
use crate::optim::nelder_mead;

/// Standard normal density constant `ln(1/sqrt(2 pi))`.
const LN_NORM_CONST: f64 = -0.918_938_533_204_672_7;

/// A monotone marginal transport map.
pub trait MarginalTransport {
    /// Maps a data value to the latent (Gaussian) axis.
    fn forward(&self, z: f64) -> f64;
    /// Maps a latent value back to data units.
    fn inverse(&self, y: f64) -> f64;
    /// `ln|dT/dz|` at `z` (the log Jacobian of the forward map).
    fn log_grad(&self, z: f64) -> f64;
}

/// Identity transform (the plain-Gaussian / no-warp baseline).
#[derive(Debug, Clone, Copy)]
pub struct Identity;

impl MarginalTransport for Identity {
    fn forward(&self, z: f64) -> f64 {
        z
    }
    fn inverse(&self, y: f64) -> f64 {
        y
    }
    fn log_grad(&self, _z: f64) -> f64 {
        0.0
    }
}

/// Box–Cox power transform with a shift, `T(z) = ((z+s)^lambda - 1)/lambda`
/// (`ln(z+s)` at `lambda = 0`). Requires `z + s > 0`; `lambda = 0` recovers
/// the log transform (lognormal kriging as a special case).
#[derive(Debug, Clone, Copy)]
pub struct BoxCox {
    /// Power parameter.
    pub lambda: f64,
    /// Shift applied before the power (keeps `z + shift > 0`).
    pub shift: f64,
}

impl MarginalTransport for BoxCox {
    fn forward(&self, z: f64) -> f64 {
        let u = z + self.shift;
        if self.lambda.abs() < 1e-9 {
            u.ln()
        } else {
            (u.powf(self.lambda) - 1.0) / self.lambda
        }
    }
    fn inverse(&self, y: f64) -> f64 {
        if self.lambda.abs() < 1e-9 {
            y.exp() - self.shift
        } else {
            (self.lambda * y + 1.0).max(0.0).powf(1.0 / self.lambda) - self.shift
        }
    }
    fn log_grad(&self, z: f64) -> f64 {
        // dT/dz = (z+s)^(lambda-1).
        (self.lambda - 1.0) * (z + self.shift).ln()
    }
}

/// Yeo–Johnson transform: a Box–Cox variant defined on the whole real line
/// (no shift needed, handles non-positive data).
#[derive(Debug, Clone, Copy)]
pub struct YeoJohnson {
    /// Power parameter.
    pub lambda: f64,
}

impl MarginalTransport for YeoJohnson {
    fn forward(&self, z: f64) -> f64 {
        let l = self.lambda;
        if z >= 0.0 {
            if l.abs() < 1e-9 {
                (z + 1.0).ln()
            } else {
                ((z + 1.0).powf(l) - 1.0) / l
            }
        } else if (l - 2.0).abs() < 1e-9 {
            -(-z + 1.0).ln()
        } else {
            -((-z + 1.0).powf(2.0 - l) - 1.0) / (2.0 - l)
        }
    }
    fn inverse(&self, y: f64) -> f64 {
        let l = self.lambda;
        if y >= 0.0 {
            if l.abs() < 1e-9 {
                y.exp() - 1.0
            } else {
                (l * y + 1.0).max(0.0).powf(1.0 / l) - 1.0
            }
        } else if (l - 2.0).abs() < 1e-9 {
            1.0 - (-y).exp()
        } else {
            1.0 - ((2.0 - l) * (-y) + 1.0).max(0.0).powf(1.0 / (2.0 - l))
        }
    }
    fn log_grad(&self, z: f64) -> f64 {
        // d/dz: (z+1)^(lambda-1) for z>=0, (-z+1)^(1-lambda) for z<0.
        if z >= 0.0 {
            (self.lambda - 1.0) * (z + 1.0).ln()
        } else {
            (1.0 - self.lambda) * (-z + 1.0).ln()
        }
    }
}

/// Sinh–arcsinh transform `T(z) = sinh(delta * asinh(z) - epsilon)`,
/// applied to standardized data. `epsilon` controls skewness, `delta > 0`
/// controls tail weight (kurtosis). Operates on `(z - loc) / scale`.
#[derive(Debug, Clone, Copy)]
pub struct SinhArcsinh {
    /// Skewness parameter.
    pub epsilon: f64,
    /// Tail-weight parameter (> 0).
    pub delta: f64,
    /// Location (subtracted before standardizing).
    pub loc: f64,
    /// Scale (> 0).
    pub scale: f64,
}

impl MarginalTransport for SinhArcsinh {
    fn forward(&self, z: f64) -> f64 {
        let u = (z - self.loc) / self.scale;
        (self.delta * u.asinh() - self.epsilon).sinh()
    }
    fn inverse(&self, y: f64) -> f64 {
        let u = ((y.asinh() + self.epsilon) / self.delta).sinh();
        u * self.scale + self.loc
    }
    fn log_grad(&self, z: f64) -> f64 {
        // T(z) = sinh(delta*asinh(u) - epsilon), u = (z-loc)/scale.
        // dT/dz = cosh(delta*asinh(u)-eps) * delta / sqrt(1+u^2) / scale.
        let u = (z - self.loc) / self.scale;
        let inner = self.delta * u.asinh() - self.epsilon;
        inner.cosh().ln() + self.delta.ln() - 0.5 * (1.0 + u * u).ln() - self.scale.ln()
    }
}

/// A fitted marginal transform plus the latent mean/std used to standardize
/// the warped data to a standard Gaussian.
#[derive(Debug, Clone)]
pub struct FittedMarginal<T: MarginalTransport> {
    transform: T,
    latent_mean: f64,
    latent_std: f64,
}

impl<T: MarginalTransport> FittedMarginal<T> {
    /// Builds a fitted marginal from a transform and an explicit latent
    /// standardizer. Use the `fit_*` functions to learn these from data;
    /// this constructor is for fixed/known warpings (e.g. an exact log map
    /// `BoxCox { lambda: 0, shift: 0 }` with `latent_mean = 0`,
    /// `latent_std = 1`).
    pub fn new(transform: T, latent_mean: f64, latent_std: f64) -> Result<Self> {
        if !(latent_std > 0.0) || !latent_mean.is_finite() {
            return Err(GeostatError::InvalidParameter(
                "latent_std must be positive and latent_mean finite".into(),
            ));
        }
        Ok(Self {
            transform,
            latent_mean,
            latent_std,
        })
    }

    /// Standardized latent value `(T(z) - mean) / std` (a standard Gaussian
    /// score under a good fit).
    pub fn to_latent(&self, z: f64) -> f64 {
        (self.transform.forward(z) - self.latent_mean) / self.latent_std
    }
    /// Back-transform a standardized latent value to data units.
    pub fn to_data(&self, y: f64) -> f64 {
        self.transform
            .inverse(y * self.latent_std + self.latent_mean)
    }
    /// The underlying transform.
    pub fn transform(&self) -> &T {
        &self.transform
    }
    /// Latent mean of `T(z)`.
    pub fn latent_mean(&self) -> f64 {
        self.latent_mean
    }
    /// Latent standard deviation of `T(z)`.
    pub fn latent_std(&self) -> f64 {
        self.latent_std
    }
}

/// Negative log-likelihood of the data under `T`:
/// `-sum( ln N(T(z); m, s^2) + ln|T'(z)| )`, with `m, s` the empirical
/// mean/std of `T(z)`. Lower means a more Gaussian warped sample.
fn neg_log_likelihood<T: MarginalTransport>(transform: &T, data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let warped: Vec<f64> = data.iter().map(|&z| transform.forward(z)).collect();
    if warped.iter().any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    let mean = warped.iter().sum::<f64>() / n;
    let var = warped.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / n;
    if !(var > 0.0) {
        return f64::INFINITY;
    }
    let std = var.sqrt();
    let mut nll = 0.0;
    for (&z, &w) in data.iter().zip(&warped) {
        let s = (w - mean) / std;
        let log_density = LN_NORM_CONST - std.ln() - 0.5 * s * s;
        let lg = transform.log_grad(z);
        if !lg.is_finite() {
            return f64::INFINITY;
        }
        nll -= log_density + lg;
    }
    nll
}

fn standardizer<T: MarginalTransport>(transform: &T, data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    let warped: Vec<f64> = data.iter().map(|&z| transform.forward(z)).collect();
    let mean = warped.iter().sum::<f64>() / n;
    let var = warped.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt().max(f64::MIN_POSITIVE))
}

/// Fits a Box–Cox transform by maximum likelihood. The shift is fixed so
/// that `min(z) + shift` is a small positive fraction of the data range
/// (keeping the power well-defined); only `lambda` is optimized.
pub fn fit_box_cox(data: &[f64]) -> Result<FittedMarginal<BoxCox>> {
    if data.len() < 3 {
        return Err(GeostatError::InsufficientData(
            "marginal fitting needs at least 3 values".into(),
        ));
    }
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(f64::MIN_POSITIVE);
    let shift = if min > 0.0 { 0.0 } else { 1e-3 * range - min };
    let obj = |p: &[f64]| {
        let t = BoxCox {
            lambda: p[0],
            shift,
        };
        neg_log_likelihood(&t, data)
    };
    let (best, _) = nelder_mead(obj, &[0.5], 0.3, 500);
    let transform = BoxCox {
        lambda: best[0],
        shift,
    };
    let (m, s) = standardizer(&transform, data);
    Ok(FittedMarginal {
        transform,
        latent_mean: m,
        latent_std: s,
    })
}

/// Fits a Yeo–Johnson transform by maximum likelihood (handles any-sign
/// data; `lambda` is the only parameter).
pub fn fit_yeo_johnson(data: &[f64]) -> Result<FittedMarginal<YeoJohnson>> {
    if data.len() < 3 {
        return Err(GeostatError::InsufficientData(
            "marginal fitting needs at least 3 values".into(),
        ));
    }
    let obj = |p: &[f64]| neg_log_likelihood(&YeoJohnson { lambda: p[0] }, data);
    let (best, _) = nelder_mead(obj, &[1.0], 0.3, 500);
    let transform = YeoJohnson { lambda: best[0] };
    let (m, s) = standardizer(&transform, data);
    Ok(FittedMarginal {
        transform,
        latent_mean: m,
        latent_std: s,
    })
}

/// Fits a sinh–arcsinh transform by maximum likelihood. `loc` and `scale`
/// are fixed to the data mean/std; `epsilon` (skew) and `delta` (tails) are
/// optimized.
pub fn fit_sinh_arcsinh(data: &[f64]) -> Result<FittedMarginal<SinhArcsinh>> {
    if data.len() < 4 {
        return Err(GeostatError::InsufficientData(
            "sinh-arcsinh fitting needs at least 4 values".into(),
        ));
    }
    let n = data.len() as f64;
    let loc = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|z| (z - loc).powi(2)).sum::<f64>() / n;
    let scale = var.sqrt().max(f64::MIN_POSITIVE);
    let obj = |p: &[f64]| {
        let delta = p[1].abs().max(1e-3);
        neg_log_likelihood(
            &SinhArcsinh {
                epsilon: p[0],
                delta,
                loc,
                scale,
            },
            data,
        )
    };
    let (best, _) = nelder_mead(obj, &[0.0, 1.0], 0.3, 800);
    let transform = SinhArcsinh {
        epsilon: best[0],
        delta: best[1].abs().max(1e-3),
        loc,
        scale,
    };
    let (m, s) = standardizer(&transform, data);
    Ok(FittedMarginal {
        transform,
        latent_mean: m,
        latent_std: s,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn round_trip<T: MarginalTransport>(t: &T, zs: &[f64]) {
        for &z in zs {
            let back = t.inverse(t.forward(z));
            assert!((back - z).abs() < 1e-7, "{z} -> {back}");
        }
    }

    #[test]
    fn transforms_round_trip() {
        round_trip(
            &BoxCox {
                lambda: 0.0,
                shift: 1.0,
            },
            &[0.5, 1.0, 5.0, 20.0],
        );
        round_trip(
            &BoxCox {
                lambda: 0.3,
                shift: 1.0,
            },
            &[0.5, 1.0, 5.0, 20.0],
        );
        round_trip(&YeoJohnson { lambda: 0.5 }, &[-3.0, -0.5, 0.0, 2.0, 7.0]);
        round_trip(&YeoJohnson { lambda: 1.7 }, &[-3.0, -0.5, 0.0, 2.0, 7.0]);
        round_trip(
            &SinhArcsinh {
                epsilon: 0.4,
                delta: 1.3,
                loc: 2.0,
                scale: 1.5,
            },
            &[-1.0, 0.0, 2.0, 5.0],
        );
    }

    #[test]
    fn box_cox_lambda_zero_is_log() {
        let t = BoxCox {
            lambda: 0.0,
            shift: 0.0,
        };
        for &z in &[0.5, 1.0, 3.0, 10.0] {
            assert!((t.forward(z) - z.ln()).abs() < 1e-12);
            // log_grad of ln(z) is -ln(z).
            assert!((t.log_grad(z) - (-z.ln())).abs() < 1e-12);
        }
    }

    #[test]
    fn log_grad_matches_numerical_derivative() {
        let cases: Vec<Box<dyn MarginalTransport>> = vec![
            Box::new(BoxCox {
                lambda: 0.4,
                shift: 2.0,
            }),
            Box::new(YeoJohnson { lambda: 0.6 }),
            Box::new(YeoJohnson { lambda: 1.5 }),
            Box::new(SinhArcsinh {
                epsilon: 0.3,
                delta: 1.2,
                loc: 1.0,
                scale: 2.0,
            }),
        ];
        let h = 1e-6;
        for t in &cases {
            for &z in &[0.5, 1.5, 3.0] {
                let num = (t.forward(z + h) - t.forward(z - h)) / (2.0 * h);
                let ana = t.log_grad(z).exp();
                assert!(
                    (num - ana).abs() < 1e-4 * (1.0 + ana.abs()),
                    "num {num} vs ana {ana} at z={z}"
                );
            }
        }
    }

    #[test]
    fn fit_recovers_lognormal_warp() {
        // Lognormal data: log(z) is Gaussian, so Box-Cox should fit lambda ~ 0.
        let mut rng = Rng::new(7);
        let data: Vec<f64> = (0..400).map(|_| (1.0 + 0.6 * rng.normal()).exp()).collect();
        let fit = fit_box_cox(&data).unwrap();
        assert!(
            fit.transform().lambda.abs() < 0.15,
            "lambda = {}",
            fit.transform().lambda
        );
        // Latent scores are ~standard normal.
        let scores: Vec<f64> = data.iter().map(|&z| fit.to_latent(z)).collect();
        let m = scores.iter().sum::<f64>() / scores.len() as f64;
        assert!(m.abs() < 1e-6);
    }

    #[test]
    fn fit_handles_skew_with_sinh_arcsinh() {
        // Skewed data; the fit should produce a finite, invertible transform.
        let mut rng = Rng::new(11);
        let data: Vec<f64> = (0..300)
            .map(|_| {
                let g = rng.normal();
                g + 0.5 * g * g.abs() // right-skew
            })
            .collect();
        let fit = fit_sinh_arcsinh(&data).unwrap();
        let scores: Vec<f64> = data.iter().map(|&z| fit.to_latent(z)).collect();
        let m = scores.iter().sum::<f64>() / scores.len() as f64;
        let v = scores.iter().map(|s| (s - m).powi(2)).sum::<f64>() / scores.len() as f64;
        assert!((v - 1.0).abs() < 0.05, "latent var {v}");
        // Round-trip through the fitted standardized map.
        for &z in data.iter().take(20) {
            let back = fit.to_data(fit.to_latent(z));
            assert!((back - z).abs() < 1e-6, "{z} -> {back}");
        }
    }
}
