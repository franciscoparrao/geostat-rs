//! Error types for the geostat-core crate.

use thiserror::Error;

/// All errors produced by this crate.
#[derive(Debug, Error)]
pub enum GeostatError {
    /// The dataset is empty or too small for the requested operation.
    #[error("empty or insufficient data: {0}")]
    InsufficientData(String),

    /// Mismatched lengths or shapes between inputs.
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// A kriging system could not be solved.
    #[error("singular kriging system: {0}")]
    SingularSystem(String),

    /// An invalid parameter value was supplied.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// No conditioning points were found inside the search neighborhood.
    #[error("no neighbors found within the search neighborhood")]
    NoNeighbors,

    /// Two data points share exactly the same coordinates, which makes the
    /// kriging system singular. Collapse or jitter duplicates first.
    #[error("duplicate data points: indices {0} and {1} share the same coordinates")]
    DuplicatePoints(usize, usize),
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, GeostatError>;
