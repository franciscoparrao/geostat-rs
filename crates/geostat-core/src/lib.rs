//! # geostat-core
//!
//! Geostatistics engine in pure Rust: variography, kriging and sequential
//! Gaussian simulation. No I/O, no heavy dependencies — a modern take on
//! the GSLIB/gstat feature set, designed for native (Rayon), Python (PyO3)
//! and WASM targets.
//!
//! ## Feature overview
//!
//! - Experimental variograms, omnidirectional and directional.
//! - Theoretical models: spherical, exponential, Gaussian, Matérn (ν = 3/2,
//!   5/2), nested structures plus nugget.
//! - Weighted least-squares model fitting (Nelder–Mead, gstat-style weights).
//! - Simple, ordinary and universal kriging with optional search
//!   neighborhoods, in parallel over prediction targets.
//! - Leave-one-out cross-validation (ME, MAE, RMSE, MSDR).
//! - Conditional sequential Gaussian simulation with deterministic seeding.
//!
//! ## Example
//!
//! ```
//! use geostat_core::{
//!     Kriging, KrigingConfig, ModelKind, PointSet, Structure, VariogramModel,
//! };
//!
//! let data = PointSet::new(
//!     vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.2]],
//!     vec![1.0, 2.0, 1.5, 2.5, 1.2],
//! )
//! .unwrap();
//! let model = VariogramModel::new(
//!     0.0,
//!     vec![Structure::new(ModelKind::Spherical, 1.0, 2.0)],
//! )
//! .unwrap();
//! let kriging = Kriging::new(&data, &model, KrigingConfig::default()).unwrap();
//! let est = kriging.predict([0.5, 0.5]).unwrap();
//! assert!(est.value.is_finite() && est.variance >= 0.0);
//! ```

#![warn(missing_docs)]
// `!(x > 0.0)` is used deliberately throughout parameter validation: unlike
// `x <= 0.0`, it also rejects NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod cokriging;
pub mod data;
pub mod error;
pub mod grid;
pub mod ik;
pub mod interpolation;
pub mod kriging;
pub mod linalg;
pub mod optim;
mod parallel;
pub mod regression;
pub mod rng;
mod search;
pub mod simulation;
pub mod sis;
pub mod trans;
pub mod transform;
pub mod tuning;
pub mod validation;
pub mod variogram;
pub mod vecchia;

pub use cokriging::{CoKriging, CoKrigingConfig, Lmc, LmcStructure, fit_lmc, fit_lmc_collocated};
pub use data::PointSet;
pub use error::{GeostatError, Result};
pub use grid::{Grid2D, Grid3D};
pub use ik::{CcdfEstimate, IkConfig, indicator_kriging};
pub use interpolation::{Idw, Knn, idw_cross_validate, knn_cross_validate};
pub use kriging::{Kriging, KrigingConfig, KrigingEstimate, KrigingMethod, block_offsets};
pub use regression::{OlsTrend, RegressionKriging, detrend_external, detrend_polynomial};
pub use rng::Rng;
pub use simulation::{
    SgsConfig, SgsResult, sequential_gaussian_simulation, sequential_gaussian_simulation_3d, sgs_at,
};
pub use sis::{SisConfig, sequential_indicator_simulation, sis_at};
pub use trans::{LognormalEstimate, lognormal_kriging};
pub use transform::NormalScore;
pub use tuning::{TuneResult, tune_idw_power, tune_knn_k, tune_kriging_neighbors};
pub use validation::{CvResult, k_fold, leave_one_out, leave_one_out_with_drift};
pub use variogram::{
    Anisotropy, DirectionConfig, ExperimentalVariogram, FitResult, LagBin, ModelKind, Structure,
    VariogramConfig, VariogramMap, VariogramModel, experimental_cross_variogram,
    experimental_variogram, fit_anisotropic, fit_best, fit_indicator_models, fit_model,
    variogram_map,
};
pub use vecchia::{
    VecchiaFit, VecchiaPlan, maxmin_order, vecchia_loglik, vecchia_mle, vecchia_param_se,
    vecchia_plan, vecchia_reml, vecchia_reml_drift,
};
