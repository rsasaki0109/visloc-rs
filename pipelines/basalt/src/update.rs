//! Numeric SE(2) inverse-compositional update primitives.
//!
//! Basalt stores an optical-flow warp as Eigen's `AffineCompact2f` and applies
//! `transform *= SE2::exp(inc).matrix()`. The types here make that right
//! composition explicit and keep the float32 arithmetic used by the upstream
//! optical-flow factory.

use crate::fma::FusedMulAdd;
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use thiserror::Error;

const SMALL_ANGLE: f32 = 1e-5;
const MAX_INCREMENT_INFINITY_NORM: f32 = 1e6;

/// A compact 2D affine warp `[linear | translation]` with float32 entries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineCompact2f {
    linear: Matrix2<f32>,
    translation: Vector2<f32>,
}

impl AffineCompact2f {
    pub fn identity() -> Self {
        Self {
            linear: Matrix2::identity(),
            translation: Vector2::zeros(),
        }
    }

    pub const fn new(linear: Matrix2<f32>, translation: Vector2<f32>) -> Self {
        Self {
            linear,
            translation,
        }
    }

    pub const fn linear(&self) -> &Matrix2<f32> {
        &self.linear
    }

    pub const fn translation(&self) -> &Vector2<f32> {
        &self.translation
    }

    pub fn transform_point(&self, point: Vector2<f32>) -> Vector2<f32> {
        // Eigen's fixed-size 2x2-by-2xN product starts each row with the
        // first column product and folds the second column with FMA before
        // adding the affine translation.  Keep that contraction explicit;
        // nalgebra's generic matrix product can choose a different scalar
        // association and move the warped sample by one f32 bit.
        let x = self.linear[(0, 0)] * point.x;
        let x = self.linear[(0, 1)].fused_mul_add(point.y, x) + self.translation.x;
        let y = self.linear[(1, 0)] * point.x;
        let y = self.linear[(1, 1)].fused_mul_add(point.y, y) + self.translation.y;
        Vector2::new(x, y)
    }

    pub fn homogeneous_matrix(&self) -> Matrix3<f32> {
        Matrix3::new(
            self.linear[(0, 0)],
            self.linear[(0, 1)],
            self.translation.x,
            self.linear[(1, 0)],
            self.linear[(1, 1)],
            self.translation.y,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Applies the Basalt IC update by right-composing `SE2::exp(increment)`.
    pub fn right_compose_se2(&mut self, increment: Vector3<f32>) {
        let update = Se2::exp(increment);
        let old_linear = self.linear;
        let old_translation = self.translation;

        // `AffineCompact2f *= SE2::exp(inc).matrix()` dispatches Eigen's
        // fixed 2x3-by-3x3 product.  Its packet kernel seeds each output with
        // the second inner coefficient, folds the homogeneous third
        // coefficient, and then folds the first coefficient.  Spelling that
        // order avoids nalgebra's generic 2x2/2x1 products changing the last
        // bit of a later IC iteration.
        let a00 = old_linear[(0, 0)];
        let a01 = old_linear[(0, 1)];
        let a10 = old_linear[(1, 0)];
        let a11 = old_linear[(1, 1)];
        let r00 = update.rotation[(0, 0)];
        let r01 = update.rotation[(0, 1)];
        let r10 = update.rotation[(1, 0)];
        let r11 = update.rotation[(1, 1)];

        let n00 = a01 * r10;
        let n00 = old_translation.x.fused_mul_add(0.0, n00);
        let n00 = a00.fused_mul_add(r00, n00);
        let n10 = a11 * r10;
        let n10 = old_translation.y.fused_mul_add(0.0, n10);
        let n10 = a10.fused_mul_add(r00, n10);
        let n01 = a01 * r11;
        let n01 = old_translation.x.fused_mul_add(0.0, n01);
        let n01 = a00.fused_mul_add(r01, n01);
        let n11 = a11 * r11;
        let n11 = old_translation.y.fused_mul_add(0.0, n11);
        let n11 = a10.fused_mul_add(r01, n11);

        let tx = a01 * update.translation.y;
        let tx = old_translation.x.fused_mul_add(1.0, tx);
        let tx = a00.fused_mul_add(update.translation.x, tx);
        let ty = a11 * update.translation.y;
        let ty = old_translation.y.fused_mul_add(1.0, ty);
        let ty = a10.fused_mul_add(update.translation.x, ty);

        self.linear = Matrix2::new(n00, n01, n10, n11);
        self.translation = Vector2::new(tx, ty);
    }

    /// Applies the update with the same finite/large-increment guards used by
    /// Basalt's optical-flow loop.
    pub fn try_right_compose_se2(&mut self, increment: Vector3<f32>) -> Result<(), Se2UpdateError> {
        if !increment.iter().all(|value| value.is_finite()) {
            return Err(Se2UpdateError::NonFiniteIncrement);
        }
        if increment.amax() >= MAX_INCREMENT_INFINITY_NORM {
            return Err(Se2UpdateError::IncrementTooLarge {
                infinity_norm: increment.amax(),
            });
        }
        self.right_compose_se2(increment);
        Ok(())
    }
}

/// A planar rigid transform produced by the Sophus-compatible exponential map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se2 {
    pub rotation: Matrix2<f32>,
    pub translation: Vector2<f32>,
}

/// Float32 spelling used by the Eigen/Sophus upstream code path.
pub type Se2f = Se2;

impl Se2 {
    /// Exponential map for a tangent ordered `[tx, ty, theta]`.
    pub fn exp(tangent: Vector3<f32>) -> Self {
        let theta = tangent.z;
        // Use the Arm-derived portable sine/cosine only for finite angles.
        // The non-finite fallback preserves the standard library's NaN/Inf
        // behavior at this public API boundary; all valid Basalt increments
        // take the deterministic path.
        let (sin_theta, cos_theta) = if theta.is_finite() {
            (portable_cosf::sinf(theta), portable_cosf::cosf(theta))
        } else {
            (theta.sin(), theta.cos())
        };
        // Sophus SO2 construction normalizes the sincos pair with hypotf.
        // For this bounded pair, f64 products retain every f32 product bit;
        // round the norm to f32 before the two scalar divisions, as upstream.
        let norm = ((sin_theta as f64) * (sin_theta as f64)
            + (cos_theta as f64) * (cos_theta as f64))
            .sqrt() as f32;
        let sin_theta = sin_theta / norm;
        let cos_theta = cos_theta / norm;
        let (sin_over_theta, one_minus_cos_over_theta) = if theta.abs() < SMALL_ANGLE {
            let theta_sq = theta * theta;
            (
                1.0 - (1.0 / 6.0) * theta_sq,
                0.5 * theta - (1.0 / 24.0) * theta * theta_sq,
            )
        } else {
            (sin_theta / theta, (1.0 - cos_theta) / theta)
        };

        let rotation = Matrix2::new(cos_theta, -sin_theta, sin_theta, cos_theta);
        // Sophus::SE2::exp spells these two products directly rather than
        // forming V and dispatching Eigen's generic 2x2-by-2x1 product.
        let translation = Vector2::new(
            // Eigen contracts the x numerator as the product with tx minus
            // the already-rounded ty product.  Keep the y numerator's fixed
            // FMA shape as well; both spellings are part of the pinned
            // Sophus/Eigen f32 operation order.
            sin_over_theta.fused_mul_add(tangent.x, -(one_minus_cos_over_theta * tangent.y)),
            one_minus_cos_over_theta.fused_mul_add(tangent.x, sin_over_theta * tangent.y),
        );
        Self {
            rotation,
            translation,
        }
    }
}

/// Portable scalar trigonometric helpers translated from the pinned Arm
/// Optimized Routines sources. The production callsite uses both helpers only
/// for finite angles in `Se2::exp`; they are not replacements for the
/// process-wide standard-library API.
///
/// Upstream: ARM-software/optimized-routines v26.01,
/// commit 649ccc8411e7c4862965710f25c570d854a1483f.
/// Sources: math/cosf.c, math/sinf.c, math/sincosf.h, math/sincosf_data.c.
/// Upstream attribution: Copyright (c) 2018-2024, Arm Limited for cosf.c,
/// sinf.c, and sincosf.h; Copyright (c) 2018-2019, Arm Limited for
/// sincosf_data.c.
/// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception.
/// Exact-tag source hashes are recorded in
/// `work/m11_aor_cosf_immutable_source_evidence_20260907.json` and
/// `work/m11_aor_sinf_immutable_source_evidence_20260907.json`.
/// No LGPL glibc source is used.
pub(crate) mod portable_cosf {
    #[derive(Clone, Copy)]
    struct Coefficients {
        sign: [f64; 4],
        c0: f64,
        c1: f64,
        c2: f64,
        c3: f64,
        c4: f64,
        s1: f64,
        s2: f64,
        s3: f64,
    }

    const P0: Coefficients = Coefficients {
        sign: [1.0, -1.0, -1.0, 1.0],
        c0: f64::from_bits(0x3ff0000000000000),
        c1: f64::from_bits(0xbfdffffffd0c621c),
        c2: f64::from_bits(0x3fa55553e1068f19),
        c3: f64::from_bits(0xbf56c087e89a359d),
        c4: f64::from_bits(0x3ef99343027bf8c3),
        s1: f64::from_bits(0xbfc555545995a603),
        s2: f64::from_bits(0x3f81107605230bc4),
        s3: f64::from_bits(0xbf2994eb3774cf24),
    };

    const P1: Coefficients = Coefficients {
        sign: [1.0, -1.0, -1.0, 1.0],
        c0: f64::from_bits(0xbff0000000000000),
        c1: f64::from_bits(0x3fdffffffd0c621c),
        c2: f64::from_bits(0xbfa55553e1068f19),
        c3: f64::from_bits(0x3f56c087e89a359d),
        c4: f64::from_bits(0xbef99343027bf8c3),
        s1: f64::from_bits(0xbfc555545995a603),
        s2: f64::from_bits(0x3f81107605230bc4),
        s3: f64::from_bits(0xbf2994eb3774cf24),
    };

    const HPI_INV: f64 = f64::from_bits(0x3fe45f306dc9c883);
    const HPI: f64 = f64::from_bits(0x3ff921fb54442d18);
    const PI63: f64 = f64::from_bits(0x3c1921fb54442d18);
    const PIO4_TOP: u32 = 0x3f4;
    const SMALL_TOP: u32 = 0x398;
    const LARGE_TOP: u32 = 0x42f;
    const INFINITY_TOP: u32 = 0x7f8;

    const INV_PIO4: [u32; 24] = [
        0xa2, 0xa2f9, 0xa2f983, 0xa2f9836e, 0xf9836e4e, 0x836e4e44, 0x6e4e4415, 0x4e441529,
        0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1, 0x2757d1f5, 0x57d1f534, 0xd1f534dd,
        0xf534ddc0, 0x34ddc0db, 0xddc0db62, 0xc0db6295, 0xdb629599, 0x6295993c, 0x95993c43,
        0x993c4390, 0x3c439041,
    ];

    const fn abstop12(value: f32) -> u32 {
        (value.to_bits() >> 20) & 0x7ff
    }

    fn sinf_poly(x: f64, x2: f64, p: &Coefficients, n: i32) -> f32 {
        if (n & 1) == 0 {
            let x3 = x * x2;
            let s1 = p.s2 + x2 * p.s3;
            let x7 = x3 * x2;
            let s = x + x3 * p.s1;
            (s + x7 * s1) as f32
        } else {
            let x4 = x2 * x2;
            let c2 = p.c3 + x2 * p.c4;
            let c1 = p.c0 + x2 * p.c1;
            let x6 = x4 * x2;
            let c = c1 + x4 * p.c2;
            (c + x6 * c2) as f32
        }
    }

    fn reduce_large(bits: u32) -> (f64, u64) {
        // The table index must use the original exponent/mantissa word,
        // before the mantissa is normalized below.
        let index = ((bits >> 26) & 15) as usize;
        let mut xi = (bits & 0x00ff_ffff) | 0x0080_0000;
        let shift = (bits >> 23) & 7;
        xi <<= shift;
        let mut res0 = (xi as u64) * (INV_PIO4[index] as u64);
        let res1 = (xi as u64) * (INV_PIO4[index + 4] as u64);
        let res2 = (xi as u64) * (INV_PIO4[index + 8] as u64);
        res0 = (res2 >> 32) | (res0 << 32);
        res0 = res0.wrapping_add(res1);
        let n = (res0.wrapping_add(1_u64 << 61)) >> 62;
        res0 = res0.wrapping_sub(n << 62);
        ((res0 as i64) as f64 * PI63, n)
    }

    pub(crate) fn cosf(y: f32) -> f32 {
        let top = abstop12(y);
        if top < PIO4_TOP {
            let x = y as f64;
            let x2 = x * x;
            if top < SMALL_TOP {
                return 1.0;
            }
            return sinf_poly(x, x2, &P0, 1);
        }
        if top < LARGE_TOP {
            let x = y as f64;
            // This is the AOR TOINT_INTRINSICS-style unscaled reduction:
            // round(x * 2/pi), then subtract the double pi/2.  It is retained
            // because it matches the pinned Basalt native runtime bit-for-bit
            // on the recorded and broad validation corpora.
            let r = x * HPI_INV;
            let n = r.round() as i32;
            let reduced = x - (n as f64) * HPI;
            let p = if (n & 2) != 0 { &P1 } else { &P0 };
            return sinf_poly(
                reduced * p.sign[(n as usize) & 3],
                reduced * reduced,
                p,
                n ^ 1,
            );
        }
        if top < INFINITY_TOP {
            let bits = y.to_bits();
            let sign = bits >> 31;
            let (reduced, n) = reduce_large(bits);
            let quadrant = n.wrapping_add(sign as u64);
            let p = if (quadrant & 2) != 0 { &P1 } else { &P0 };
            return sinf_poly(
                reduced * p.sign[(quadrant & 3) as usize],
                reduced * reduced,
                p,
                (n as i32) ^ 1,
            );
        }
        f32::NAN
    }

    /// AOR `sinf` wrapper used by the finite-angle Se2 path.
    pub(crate) fn sinf(y: f32) -> f32 {
        let top = abstop12(y);
        if top < PIO4_TOP {
            let x = y as f64;
            let x2 = x * x;
            if top < SMALL_TOP {
                // Returning the input preserves the result-bit behavior for
                // tiny values and signed zero. Rust does not reproduce the
                // upstream force-evaluation fenv/errno side effect here.
                return y;
            }
            return sinf_poly(x, x2, &P0, 0);
        }
        if top < LARGE_TOP {
            let x = y as f64;
            let r = x * HPI_INV;
            let n = r.round() as i32;
            let reduced = x - (n as f64) * HPI;
            let p = if (n & 2) != 0 { &P1 } else { &P0 };
            return sinf_poly(reduced * p.sign[(n as usize) & 3], reduced * reduced, p, n);
        }
        if top < INFINITY_TOP {
            let bits = y.to_bits();
            let sign = bits >> 31;
            let (reduced, n) = reduce_large(bits);
            let quadrant = n.wrapping_add(sign as u64);
            let p = if (quadrant & 2) != 0 { &P1 } else { &P0 };
            return sinf_poly(
                reduced * p.sign[(quadrant & 3) as usize],
                reduced * reduced,
                p,
                n as i32,
            );
        }
        // The translated public contract is finite-only; retain the existing
        // helper's canonical NaN fallback for non-finite diagnostic inputs.
        f32::NAN
    }
}

/// Errors raised by the guarded IC update.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum Se2UpdateError {
    #[error("SE2 increment contains a non-finite value")]
    NonFiniteIncrement,
    #[error("SE2 increment infinity norm {infinity_norm} is too large")]
    IncrementTooLarge { infinity_norm: f32 },
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix2, Vector2, Vector3};

    use super::*;

    #[test]
    fn native_se2_normalized_sincos_affine_update() {
        // Pinned native frame19/cam1/track30, level2 iteration2.
        // hypotf(cos(theta),sin(theta)) rounds to 0x3f7fffff, not 1.
        let v = f32::from_bits;
        let mut affine = AffineCompact2f::new(
            Matrix2::new(v(0x3f7c112a), v(0x3e32ce30), v(0xbe32ce2f), v(0x3f7c112b)),
            Vector2::new(v(0x423bbf9c), v(0x41d70c3c)),
        );
        affine.right_compose_se2(Vector3::new(v(0x3def4594), v(0xbd724cd1), v(0x3e3eca86)));
        let actual: Vec<u32> = affine
            .linear()
            .iter()
            .chain(affine.translation().iter())
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(
            actual,
            [0x3f7ffc2e, 0x3c3106e0, 0xbc3106c2, 0x3f7ffc2f, 0x423c31b1, 0x41d68004]
        );
    }

    #[test]
    fn small_angle_exp_is_finite() {
        let transform = Se2::exp(Vector3::new(0.2, -0.3, 1e-7));
        assert!(transform.translation.iter().all(|value| value.is_finite()));
        assert!((transform.rotation - Matrix2::identity()).norm() < 1e-6);
    }

    #[test]
    fn guarded_update_rejects_non_finite_and_large_increments() {
        let mut transform = AffineCompact2f::identity();
        assert!(matches!(
            transform.try_right_compose_se2(Vector3::new(f32::NAN, 0.0, 0.0)),
            Err(Se2UpdateError::NonFiniteIncrement)
        ));
        assert!(matches!(
            transform.try_right_compose_se2(Vector3::new(1e6, 0.0, 0.0)),
            Err(Se2UpdateError::IncrementTooLarge { .. })
        ));
    }

    #[test]
    fn right_composition_preserves_affine_homogeneous_structure() {
        let mut transform =
            AffineCompact2f::new(Matrix2::new(1.1, 0.2, -0.3, 0.9), Vector2::new(10.0, -4.0));
        transform.right_compose_se2(Vector3::new(0.25, -0.15, 0.2));
        let matrix = transform.homogeneous_matrix();
        assert_eq!(matrix[(2, 0)], 0.0);
        assert_eq!(matrix[(2, 1)], 0.0);
        assert_eq!(matrix[(2, 2)], 1.0);
    }

    #[test]
    fn m11_se2_exp_portable_cosf_witness_matches_native_fixture() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            tangent_bits: [String; 3],
            expected_rotation_bits_column_major: [String; 4],
            expected_translation_bits: [String; 2],
        }

        fn bits(value: &str) -> u32 {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("m11 cosine witness bit pattern")
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../tests/fixtures/m11_se2_exp_portable_cosf_witness.json"
        ))
        .expect("m11 cosine witness fixture must parse");
        let tangent = Vector3::from(std::array::from_fn(|index| {
            f32::from_bits(bits(&fixture.tangent_bits[index]))
        }));
        let update = Se2::exp(tangent);
        let actual_rotation = [
            update.rotation[(0, 0)],
            update.rotation[(1, 0)],
            update.rotation[(0, 1)],
            update.rotation[(1, 1)],
        ];
        for (actual, expected) in actual_rotation
            .iter()
            .zip(fixture.expected_rotation_bits_column_major.iter())
        {
            assert_eq!(actual.to_bits(), bits(expected));
        }
        let actual_translation = [update.translation.x, update.translation.y];
        for (actual, expected) in actual_translation
            .iter()
            .zip(fixture.expected_translation_bits.iter())
        {
            assert_eq!(actual.to_bits(), bits(expected));
        }
    }

    #[test]
    fn m11_se2_exp_portable_sinf_witness_matches_pinned_linux_runtime() {
        // Frame-14 KLT level-2 iteration-1 increment. Linux glibc 2.35
        // sinf/sincosf return the exact sine bit pattern below; the
        // pre-candidate MSVC standard-library path returned one ULP lower.
        // Nonzero translation makes this a full Se2::exp regression rather
        // than a rotation-only witness.
        let tangent = Vector3::new(
            f32::from_bits(0x3ea40336),
            f32::from_bits(0x3db21f9f),
            f32::from_bits(0x3e1f1054),
        );
        let update = Se2::exp(tangent);
        assert_eq!(update.rotation[(0, 0)].to_bits(), 0x3f7ceaec);
        assert_eq!(update.rotation[(1, 0)].to_bits(), 0x3e1e6cc5);
        assert_eq!(update.rotation[(0, 1)].to_bits(), 0xbe1e6cc5);
        assert_eq!(update.rotation[(1, 1)].to_bits(), 0x3f7ceaec);
        assert_eq!(update.translation.x.to_bits(), 0x3e9fe6ef);
        assert_eq!(update.translation.y.to_bits(), 0x3de44282);
    }

    #[test]
    fn portable_sinf_preserves_signed_zero_tiny_and_nonfinite_contract() {
        assert_eq!(portable_cosf::sinf(0.0).to_bits(), 0x0000_0000);
        assert_eq!(portable_cosf::sinf(-0.0).to_bits(), 0x8000_0000);
        assert_eq!(
            portable_cosf::sinf(f32::from_bits(0x0000_0001)).to_bits(),
            0x0000_0001
        );
        assert_eq!(
            portable_cosf::sinf(f32::from_bits(0x8000_0001)).to_bits(),
            0x8000_0001
        );
        assert!(portable_cosf::sinf(f32::INFINITY).is_nan());
        assert!(portable_cosf::sinf(f32::NEG_INFINITY).is_nan());
        assert!(portable_cosf::sinf(f32::NAN).is_nan());
    }

    #[test]
    fn right_composition_matches_native_l3_i1_boundary() {
        let mut transform = AffineCompact2f::new(
            Matrix2::new(
                f32::from_bits(0x3f73999b),
                f32::from_bits(0xbe9d6ac3),
                f32::from_bits(0x3e9d6ac3),
                f32::from_bits(0x3f73999b),
            ),
            Vector2::new(f32::from_bits(0x40d8ad14), f32::from_bits(0x41db5ebe)),
        );
        transform.right_compose_se2(Vector3::new(
            f32::from_bits(0xbf47a90b),
            f32::from_bits(0xbd615534),
            f32::from_bits(0xbe91b7c1),
        ));

        let expected = [
            0x3f7fe679, 0xbce4a105, 0x40c07598, 0x3ce4a103, 0x3f7fe67a, 0x41d9e26b,
        ];
        let matrix = transform.homogeneous_matrix();
        let actual = [
            matrix[(0, 0)],
            matrix[(0, 1)],
            matrix[(0, 2)],
            matrix[(1, 0)],
            matrix[(1, 1)],
            matrix[(1, 2)],
        ];
        for (value, expected_bits) in actual.into_iter().zip(expected) {
            assert_eq!(value.to_bits(), expected_bits);
        }
    }

    #[test]
    fn m7cq_frame9_cam1_stereo_update_frontier_is_reproduced() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            pre_transform_bits: [String; 6],
            increment_bits: [String; 3],
            exp_bits: [String; 6],
            native_post_bits: [String; 6],
            rust_post_bits: [String; 6],
        }

        fn bits(value: &str) -> u32 {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("m7cq fixture bit pattern")
        }

        fn values<const N: usize>(entries: &[String; N]) -> [f32; N] {
            std::array::from_fn(|index| f32::from_bits(bits(&entries[index])))
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../tests/fixtures/m7cq_frame9_cam1_stereo_update_frontier.json"
        ))
        .expect("m7cq frontier fixture must parse");
        let pre = values(&fixture.pre_transform_bits);
        let increment = Vector3::from(values(&fixture.increment_bits));
        let expected_exp = values(&fixture.exp_bits);
        let update = Se2::exp(increment);
        let observed_exp = [
            update.rotation[(0, 0)],
            update.rotation[(0, 1)],
            update.rotation[(1, 0)],
            update.rotation[(1, 1)],
            update.translation.x,
            update.translation.y,
        ];
        for (value, expected) in observed_exp.iter().zip(expected_exp) {
            assert_eq!(value.to_bits(), expected.to_bits());
        }

        let mut transform = AffineCompact2f::new(
            Matrix2::new(pre[0], pre[1], pre[2], pre[3]),
            Vector2::new(pre[4], pre[5]),
        );
        transform.right_compose_se2(increment);
        let observed = [
            transform.linear()[(0, 0)],
            transform.linear()[(0, 1)],
            transform.linear()[(1, 0)],
            transform.linear()[(1, 1)],
            transform.translation().x,
            transform.translation().y,
        ];
        let expected_rust = values(&fixture.rust_post_bits);
        for (value, expected) in observed.iter().zip(expected_rust) {
            assert_eq!(value.to_bits(), expected.to_bits());
        }

        let expected_native = values(&fixture.native_post_bits);
        let mismatches: Vec<usize> = expected_native
            .iter()
            .zip(expected_rust.iter())
            .enumerate()
            .filter_map(|(index, (native, rust))| {
                (native.to_bits() != rust.to_bits()).then_some(index)
            })
            .collect();
        assert_eq!(mismatches, vec![0, 1, 2]);
    }

    #[test]
    #[ignore = "diagnostic-only runtime sin/cos witness; requires explicit theta bits and trace path"]
    fn m11_se2_exp_runtime_cos_witness() {
        let theta_bits = std::env::var("VISLOC_BASALT_SE2_WITNESS_THETA_BITS")
            .expect("VISLOC_BASALT_SE2_WITNESS_THETA_BITS")
            .trim_start_matches("0x")
            .to_owned();
        let theta = f32::from_bits(
            u32::from_str_radix(&theta_bits, 16)
                .expect("VISLOC_BASALT_SE2_WITNESS_THETA_BITS must be hexadecimal"),
        );
        let trace_path = std::path::PathBuf::from(
            std::env::var_os("VISLOC_BASALT_SE2_WITNESS_TRACE")
                .expect("VISLOC_BASALT_SE2_WITNESS_TRACE"),
        );
        let tangent = Vector3::new(0.0_f32, 0.0_f32, theta);
        let direct_sin = theta.sin();
        let direct_cos = theta.cos();
        let update = Se2::exp(tangent);
        let record = serde_json::json!({
            "schema": "visloc.basalt.m11.se2_exp_runtime_cos_witness.v1",
            "source": "rust",
            "capture_status": "TRACE_CAPTURED",
            "profile": std::env::var("VISLOC_BASALT_SE2_WITNESS_PROFILE")
                .unwrap_or_else(|_| "unspecified".to_owned()),
            "theta_f32": theta,
            "theta_f32_bits": format!("{:08x}", theta.to_bits()),
            "direct_sin_f32": direct_sin,
            "direct_sin_f32_bits": format!("{:08x}", direct_sin.to_bits()),
            "direct_cos_f32": direct_cos,
            "direct_cos_f32_bits": format!("{:08x}", direct_cos.to_bits()),
            "se2_exp_rotation_f32": [
                [update.rotation[(0, 0)], update.rotation[(0, 1)]],
                [update.rotation[(1, 0)], update.rotation[(1, 1)]],
            ],
            "se2_exp_rotation_f32_bits_column_major": [
                format!("{:08x}", update.rotation[(0, 0)].to_bits()),
                format!("{:08x}", update.rotation[(1, 0)].to_bits()),
                format!("{:08x}", update.rotation[(0, 1)].to_bits()),
                format!("{:08x}", update.rotation[(1, 1)].to_bits()),
            ],
            "se2_exp_translation_f32": [update.translation.x, update.translation.y],
            "se2_exp_translation_f32_bits": [
                format!("{:08x}", update.translation.x.to_bits()),
                format!("{:08x}", update.translation.y.to_bits()),
            ],
            "note": "Direct runtime f32 sin/cos and the production Se2::exp rotation are reported; this witness makes no parity assertion.",
        });
        if let Some(parent) = trace_path.parent() {
            std::fs::create_dir_all(parent).expect("SE2 witness trace parent");
        }
        std::fs::write(
            &trace_path,
            serde_json::to_vec(&record).expect("SE2 witness trace JSON"),
        )
        .expect("SE2 witness trace output");
        println!(
            "{}",
            serde_json::to_string(&record).expect("SE2 witness trace line")
        );
    }
}
