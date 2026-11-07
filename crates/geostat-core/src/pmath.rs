//! Portable transcendental functions.
//!
//! `f64::ln`, `exp`, `sin`, ... call the platform's libm (glibc on Linux, the
//! system library on macOS/Windows, a Rust port of musl's libm on wasm32), and
//! IEEE 754 does not require those to round identically. Native and WebAssembly
//! builds of the same simulation therefore differed in the last bit of ~1% of
//! simulated values (validation/bitident). Routing every transcendental call in
//! the engine through the pure-Rust `libm` crate gives the same implementation,
//! hence the same bits, on every target. `sqrt` and arithmetic are correctly
//! rounded by IEEE 754 and need no replacement; `powi` lowers to repeated
//! multiplication.

/// Extension methods mirroring the `f64` inherent ones, computed by `libm`.
pub trait PortableF64 {
    fn ln_p(self) -> f64;
    fn exp_p(self) -> f64;
    fn sin_p(self) -> f64;
    fn cos_p(self) -> f64;
    fn sin_cos_p(self) -> (f64, f64);
    fn acos_p(self) -> f64;
    fn powf_p(self, e: f64) -> f64;
    fn cosh_p(self) -> f64;
    fn tanh_p(self) -> f64;
    fn atanh_p(self) -> f64;
}

impl PortableF64 for f64 {
    #[inline]
    fn ln_p(self) -> f64 {
        libm::log(self)
    }
    #[inline]
    fn exp_p(self) -> f64 {
        libm::exp(self)
    }
    #[inline]
    fn sin_p(self) -> f64 {
        libm::sin(self)
    }
    #[inline]
    fn cos_p(self) -> f64 {
        libm::cos(self)
    }
    #[inline]
    fn sin_cos_p(self) -> (f64, f64) {
        libm::sincos(self)
    }
    #[inline]
    fn acos_p(self) -> f64 {
        libm::acos(self)
    }
    #[inline]
    fn powf_p(self, e: f64) -> f64 {
        libm::pow(self, e)
    }
    #[inline]
    fn cosh_p(self) -> f64 {
        libm::cosh(self)
    }
    #[inline]
    fn tanh_p(self) -> f64 {
        libm::tanh(self)
    }
    #[inline]
    fn atanh_p(self) -> f64 {
        libm::atanh(self)
    }
}
