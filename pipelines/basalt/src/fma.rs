//! `mul_add` that stays fast in the browser.
//!
//! The port evaluates Eigen's FMA packets with `f32::mul_add`. Natively that
//! is one instruction (or libm's hardware-backed `fmaf`). WebAssembly has no
//! scalar fused multiply-add: each call becomes an out-of-line, bit-exact
//! software `fmaf` that the optimizer cannot inline, which took 40% of a
//! frame in the browser build (`bindings/wasm`).
//!
//! [`FusedMulAdd::fused_mul_add`] is `mul_add` itself except on wasm, where
//! the `f32` case is the product and sum in `f64`: the product of two `f32`s
//! is exact there, so the result is the fused one except when the `f64` sum
//! lands exactly halfway between two `f32`s (double rounding), a case that
//! does not arise in practice. `f64` keeps `mul_add` everywhere.

pub(crate) trait FusedMulAdd: Copy {
    /// `self * a + b`, fused.
    fn fused_mul_add(self, a: Self, b: Self) -> Self;
}

impl FusedMulAdd for f32 {
    #[inline(always)]
    fn fused_mul_add(self, a: f32, b: f32) -> f32 {
        #[cfg(not(target_family = "wasm"))]
        return self.mul_add(a, b);
        #[cfg(target_family = "wasm")]
        return (f64::from(self) * f64::from(a) + f64::from(b)) as f32;
    }
}

impl FusedMulAdd for f64 {
    #[inline(always)]
    fn fused_mul_add(self, a: f64, b: f64) -> f64 {
        self.mul_add(a, b)
    }
}
