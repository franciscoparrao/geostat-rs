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
///
/// Fixed, independent of the runtime thread count. Chunk boundaries fix the
/// order in which partial sums are merged, and floating-point addition is
/// not associative — deriving this from `rayon::current_num_threads()` made
/// the experimental variogram (and anything that auto-fits from it) differ
/// at the ulp level between machines with different core counts, breaking
/// the bit-for-bit reproducibility claim. Rayon still spreads these chunks
/// across however many threads are actually available; only the partition
/// itself needs to be stable. 256 comfortably covers real-world core
/// counts; excess chunks on small inputs are empty ranges, not overhead.
#[cfg(feature = "parallel")]
pub(crate) fn n_chunks() -> usize {
    256
}

/// Sequential fallback: a single chunk.
#[cfg(not(feature = "parallel"))]
pub(crate) fn n_chunks() -> usize {
    1
}
