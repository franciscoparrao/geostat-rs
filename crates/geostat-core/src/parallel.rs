//! Execution helpers behind the `parallel` feature: with it (the default),
//! work fans out via rayon; without it (e.g. wasm32 targets), the same
//! helpers run sequentially with identical semantics and bounds.

use crate::error::Result;

/// Maps `0..n` to a vector, in parallel when enabled.
#[cfg(feature = "parallel")]
pub(crate) fn par_map<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    use rayon::prelude::*;
    (0..n).into_par_iter().map(f).collect()
}

/// Sequential fallback of [`par_map`].
#[cfg(not(feature = "parallel"))]
pub(crate) fn par_map<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    (0..n).map(f).collect()
}

/// Fallible variant of [`par_map`]: short-circuits on the first error.
#[cfg(feature = "parallel")]
pub(crate) fn par_try_map<T, F>(n: usize, f: F) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(usize) -> Result<T> + Send + Sync,
{
    use rayon::prelude::*;
    (0..n).into_par_iter().map(f).collect()
}

/// Sequential fallback of [`par_try_map`].
#[cfg(not(feature = "parallel"))]
pub(crate) fn par_try_map<T, F>(n: usize, f: F) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(usize) -> Result<T> + Send + Sync,
{
    (0..n).map(f).collect()
}

/// Number of work chunks to split pair-accumulation loops into.
#[cfg(feature = "parallel")]
pub(crate) fn n_chunks() -> usize {
    rayon::current_num_threads().max(1) * 4
}

/// Sequential fallback: a single chunk.
#[cfg(not(feature = "parallel"))]
pub(crate) fn n_chunks() -> usize {
    1
}
