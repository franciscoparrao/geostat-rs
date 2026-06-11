//! Variography: experimental variograms, theoretical models and fitting.

mod cross;
mod experimental;
mod fit;
mod model;

pub use cross::experimental_cross_variogram;
pub use experimental::{
    DirectionConfig, ExperimentalVariogram, LagBin, VariogramConfig, experimental_variogram,
};
pub use fit::{FitResult, fit_best, fit_model};
pub use model::{Anisotropy, ModelKind, Structure, VariogramModel};
