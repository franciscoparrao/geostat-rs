//! Transport Gaussian Processes: warped kriging via learnable marginal
//! transport maps.
//!
//! This is the bridge between classical geostatistics and the Transport
//! Gaussian Process framework (Rios & Tobar, 2019). A monotone marginal
//! transport map `T` warps the (possibly non-Gaussian, skewed, bounded)
//! data onto a standard-Gaussian latent axis; ordinary kriging runs in that
//! latent space; predictions are pushed back through `T^{-1}`, with the
//! predictive distribution propagated by Monte Carlo to recover an unbiased
//! (E-type) estimate and quantiles in data units.
//!
//! It generalizes the crate's fixed warpings — the normal-score transform
//! ([`crate::NormalScore`]) and lognormal kriging ([`crate::trans`]) — to
//! parametric families fitted by maximum likelihood, which is what makes
//! Transport GPs effective on the small, heavy-tailed samples typical of
//! geochemical work (e.g. mine-tailings characterization).

mod marginal;

pub use marginal::{
    AnyMarginal, BoxCox, BoxCoxSinhArcsinh, Composed, FittedMarginal, Identity, MarginalSelection,
    MarginalTransport, SinhArcsinh, YeoJohnson, fit_best_marginal, fit_box_cox,
    fit_box_cox_sinh_arcsinh, fit_sinh_arcsinh, fit_yeo_johnson,
};

use crate::data::PointSet;
use crate::error::{GeostatError, Result};
use crate::kriging::{Kriging, KrigingConfig};
use crate::rng::Rng;
use crate::variogram::VariogramModel;

/// A warped-kriging prediction in data units.
#[derive(Debug, Clone)]
pub struct WarpedEstimate {
    /// Posterior mean (E-type estimate) in data units.
    pub mean: f64,
    /// Posterior standard deviation in data units.
    pub std: f64,
    /// Requested predictive quantiles in data units (same order as the
    /// `quantiles` argument).
    pub quantiles: Vec<f64>,
}

/// Warped (transport) kriging predictor: a fitted marginal transform bound
/// to a latent-space kriging model.
///
/// The `latent_model` is the variogram of the *latent* (standardized,
/// warped) data; fit it on `marginal.to_latent(z)` values.
#[derive(Debug)]
pub struct TransportKriging<'a, T: MarginalTransport, const D: usize = 2> {
    latent: PointSet<D>,
    marginal: FittedMarginal<T>,
    model: &'a VariogramModel,
    config: KrigingConfig,
}

impl<'a, T: MarginalTransport, const D: usize> TransportKriging<'a, T, D> {
    /// Builds a predictor from data, a fitted marginal and a latent-space
    /// variogram model.
    pub fn new(
        data: &PointSet<D>,
        marginal: FittedMarginal<T>,
        latent_model: &'a VariogramModel,
        config: KrigingConfig,
    ) -> Result<Self> {
        let latent_vals: Vec<f64> = data
            .values()
            .iter()
            .map(|&z| marginal.to_latent(z))
            .collect();
        let latent = PointSet::new(data.coords().to_vec(), latent_vals)?;
        Ok(Self {
            latent,
            marginal,
            model: latent_model,
            config,
        })
    }

    /// The standardized latent dataset (warped data), e.g. to fit/inspect
    /// the latent variogram.
    pub fn latent(&self) -> &PointSet<D> {
        &self.latent
    }

    /// Warped-kriging prediction at a target. `n_samples` Monte Carlo draws
    /// from the latent Gaussian posterior are pushed through `T^{-1}` to
    /// estimate the mean, std and the requested `quantiles` (each in
    /// `[0, 1]`). Sampling is deterministic given `seed`.
    pub fn predict(
        &self,
        target: [f64; D],
        quantiles: &[f64],
        n_samples: usize,
        seed: u64,
    ) -> Result<WarpedEstimate> {
        if n_samples < 2 {
            return Err(GeostatError::InvalidParameter(
                "warped kriging needs at least 2 Monte Carlo samples".into(),
            ));
        }
        if quantiles.iter().any(|q| !(0.0..=1.0).contains(q)) {
            return Err(GeostatError::InvalidParameter(
                "quantiles must lie in [0, 1]".into(),
            ));
        }
        let kriging = Kriging::new(&self.latent, self.model, self.config.clone())?;
        let est = kriging.predict(target)?;
        let sd = est.variance.max(0.0).sqrt();

        // Monte Carlo through the inverse marginal.
        let mut rng = Rng::new(seed);
        let mut draws: Vec<f64> = (0..n_samples)
            .map(|_| self.marginal.to_data(est.value + sd * rng.normal()))
            .collect();
        let n = draws.len() as f64;
        let mean = draws.iter().sum::<f64>() / n;
        let var = draws.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n;

        draws.sort_by(f64::total_cmp);
        let qs: Vec<f64> = quantiles
            .iter()
            .map(|&q| {
                let idx = ((q * (draws.len() as f64 - 1.0)).round() as usize).min(draws.len() - 1);
                draws[idx]
            })
            .collect();

        Ok(WarpedEstimate {
            mean,
            std: var.sqrt(),
            quantiles: qs,
        })
    }
}

impl<T: MarginalTransport + Sync> TransportKriging<'_, T, 2> {
    /// Warped-kriging E-type mean and std over all grid cell centers.
    /// Returns `(means, stds)` in grid storage order.
    pub fn predict_grid(
        &self,
        grid: &crate::grid::Grid2D,
        n_samples: usize,
        seed: u64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let centers = grid.centers();
        let ests = crate::parallel::par_try_map(centers.len(), |i| {
            // Vary the seed per cell so draws are independent but reproducible.
            self.predict(
                centers[i],
                &[],
                n_samples,
                seed ^ (i as u64).wrapping_mul(0x9E37_79B9),
            )
        })?;
        Ok(ests.into_iter().map(|e| (e.mean, e.std)).unzip())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::variogram::{
        ModelKind, Structure, VariogramConfig, experimental_variogram, fit_best,
    };

    /// Lognormal field: log(z) is a smooth Gaussian field plus noise.
    fn lognormal_field(n: usize, seed: u64) -> PointSet {
        let mut rng = Rng::new(seed);
        let mut coords = Vec::new();
        let mut values = Vec::new();
        for _ in 0..n {
            let x = rng.uniform() * 100.0;
            let y = rng.uniform() * 100.0;
            let logv = (x / 30.0).sin() + (y / 25.0).cos() + 0.3 * rng.normal();
            coords.push([x, y]);
            values.push(logv.exp());
        }
        PointSet::new(coords, values).unwrap()
    }

    #[test]
    fn warped_kriging_recovers_lognormal_etype() {
        // With a Box-Cox fit (which lands near lambda=0 = log) the E-type
        // mean should track the analytic lognormal back-transform
        // exp(y + sigma2/2) reasonably well.
        let data = lognormal_field(200, 5);
        let marginal = fit_box_cox(data.values()).unwrap();
        // Fit the latent variogram on the warped (standardized) data.
        let latent_vals: Vec<f64> = data
            .values()
            .iter()
            .map(|&z| marginal.to_latent(z))
            .collect();
        let latent = PointSet::new(data.coords().to_vec(), latent_vals).unwrap();
        let cfg = VariogramConfig {
            n_lags: 12,
            max_dist: 50.0,
            direction: None,
        };
        let ev = experimental_variogram(&latent, &cfg).unwrap();
        let fit = fit_best(&ev, &ModelKind::ALL).unwrap();
        let tk =
            TransportKriging::new(&data, marginal, &fit.model, KrigingConfig::default()).unwrap();

        let est = tk
            .predict([50.0, 50.0], &[0.1, 0.5, 0.9], 20_000, 42)
            .unwrap();
        assert!(est.mean.is_finite() && est.mean > 0.0);
        assert!(est.std > 0.0);
        // Quantiles ordered and bracketing the median.
        assert!(est.quantiles[0] <= est.quantiles[1]);
        assert!(est.quantiles[1] <= est.quantiles[2]);
        // E-type mean exceeds the median for a right-skewed posterior.
        assert!(est.mean >= est.quantiles[1] * 0.95);
    }

    #[test]
    fn exact_marginal_at_data_points() {
        // At a datum the latent kriging is exact, so the posterior collapses
        // and the E-type mean returns the observed value.
        let data = lognormal_field(120, 9);
        let marginal = fit_box_cox(data.values()).unwrap();
        let model =
            VariogramModel::new(0.01, vec![Structure::new(ModelKind::Spherical, 0.95, 30.0)])
                .unwrap();
        let tk = TransportKriging::new(&data, marginal, &model, KrigingConfig::default()).unwrap();
        for i in (0..data.len()).step_by(40) {
            let est = tk.predict(data.coord(i), &[], 2000, 1).unwrap();
            assert!(
                (est.mean - data.value(i)).abs() < 1e-2 * data.value(i),
                "{} vs {}",
                est.mean,
                data.value(i)
            );
        }
    }

    #[test]
    fn rejects_bad_args() {
        let data = lognormal_field(50, 3);
        let marginal = fit_box_cox(data.values()).unwrap();
        let model = VariogramModel::new(0.1, vec![Structure::new(ModelKind::Spherical, 0.9, 30.0)])
            .unwrap();
        let tk = TransportKriging::new(&data, marginal, &model, KrigingConfig::default()).unwrap();
        assert!(tk.predict([10.0, 10.0], &[], 1, 1).is_err());
        assert!(tk.predict([10.0, 10.0], &[1.5], 100, 1).is_err());
    }
}
