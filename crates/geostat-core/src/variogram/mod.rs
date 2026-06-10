//! Variography: experimental variograms, theoretical models and fitting.

mod experimental;
mod fit;
mod model;

pub use experimental::{
    DirectionConfig, ExperimentalVariogram, LagBin, VariogramConfig, experimental_variogram,
};
pub use fit::{FitResult, fit_best, fit_model};
pub use model::{ModelKind, Structure, VariogramModel};
