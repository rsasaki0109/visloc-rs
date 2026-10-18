//! Mean-normalized Basalt `Pattern51` patches and inverse-compositional
//! Jacobians.

use crate::fma::FusedMulAdd;
use nalgebra::{SMatrix, SVector, Vector2, Vector3};

use crate::{pattern::Pattern51, pyramid::RawU16Image, update::AffineCompact2f};

pub type PatchData51 = SVector<f32, 52>;
pub type PatchJacobian51 = SMatrix<f32, 52, 3>;
pub type PatchInverseJacobian51 = SMatrix<f32, 3, 52>;

const EPS: f32 = f32::EPSILON;

/// A normalized 52-sample patch and its SE(2) inverse-compositional data.
#[derive(Debug, Clone, PartialEq)]
pub struct MeanNormalizedPatch51 {
    pub position: Vector2<f32>,
    /// Mean of valid, unnormalized samples, matching Basalt's `mean` field.
    pub mean: f32,
    /// Normalized samples; `-1` marks samples outside the gradient margin.
    pub data: PatchData51,
    /// Jacobian of normalized samples with respect to an SE(2) warp at zero.
    pub jacobian_se2: PatchJacobian51,
    /// `(JᵀJ)⁻¹Jᵀ`, used by the IC update primitive.
    pub h_se2_inv_j_se2_t: PatchInverseJacobian51,
    pub valid_samples: usize,
    pub valid: bool,
}

impl MeanNormalizedPatch51 {
    /// Samples `image` at `position + Pattern51::offsets()` and computes the
    /// same normalized data/Jacobian sequence as Basalt's
    /// `OpticalFlowPatch::setDataJacSe2`.
    pub fn from_image(image: &RawU16Image, position: Vector2<f32>) -> Self {
        let mut data = PatchData51::from_element(-1.0);
        let mut jacobian = PatchJacobian51::zeros();
        let mut sum = 0.0_f32;
        let mut valid_samples = 0_usize;

        for (index, [offset_x, offset_y]) in Pattern51::offsets().iter().enumerate() {
            let offset = Vector2::new(*offset_x, *offset_y);
            let sample_position = position + offset;
            if image.in_bounds(sample_position, 2.0) {
                if let Some([value, gradient_x, gradient_y]) = image.interp_grad(sample_position) {
                    data[index] = value;
                    sum += value;
                    jacobian[(index, 0)] = gradient_x;
                    jacobian[(index, 1)] = gradient_y;
                    // Eigen's fixed 1x2-by-2x3 product emits a plain first
                    // product followed by an FMA for the SE(2) rotation
                    // column.  Keep that contraction order explicit.
                    jacobian[(index, 2)] =
                        gradient_y.fused_mul_add(*offset_x, gradient_x * (-*offset_y));
                    valid_samples += 1;
                }
            }
        }

        let mut patch = Self {
            position,
            mean: if valid_samples > 0 {
                sum / valid_samples as f32
            } else {
                0.0
            },
            data,
            jacobian_se2: jacobian,
            h_se2_inv_j_se2_t: PatchInverseJacobian51::zeros(),
            valid_samples,
            valid: false,
        };

        if valid_samples == 0 || sum <= EPS || !sum.is_finite() {
            return patch;
        }

        let mean_inv = valid_samples as f32 / sum;
        let mut grad_sum = Vector3::zeros();
        for index in 0..Pattern51::SAMPLE_COUNT {
            grad_sum += patch.jacobian_se2.row(index).transpose();
        }
        for index in 0..Pattern51::SAMPLE_COUNT {
            if patch.data[index] >= 0.0 {
                // Keep Eigen's source order: multiply the accumulated
                // gradient by the raw sample first, then divide by the raw
                // sum.  `grad_sum * (data / sum)` rounds differently.
                let correction = grad_sum * patch.data[index] / sum;
                let current_row = patch.jacobian_se2.row(index).into_owned();
                patch
                    .jacobian_se2
                    .row_mut(index)
                    .copy_from(&(current_row - correction.transpose()));
                patch.data[index] *= mean_inv;
            } else {
                patch.jacobian_se2.row_mut(index).fill(0.0);
            }
        }
        patch.jacobian_se2 *= mean_inv;

        // Eigen's fixed-size `J.transpose() * J` uses a fused multiply-add
        // accumulator for each dot product.  Nalgebra's generic matrix
        // product takes a different packet/reduction path, which changes a
        // few f32 Hessian entries by one ulp.  Spell out the three dot
        // products with the upstream contraction order.
        let hessian = hessian_fma(&patch.jacobian_se2);
        if let Some(hessian_inverse) = ldlt_inverse(hessian) {
            // Eigen's fixed-size 3x3-by-3x52 product starts each packet
            // accumulator with the second inner coefficient, then folds the
            // third and first coefficients with FMA.  The mathematically
            // equivalent 0,1,2 spelling changes several f32 bits.
            patch.h_se2_inv_j_se2_t =
                hessian_inverse_jacobian_product(hessian_inverse, &patch.jacobian_se2);
            patch.valid = patch.mean > EPS
                && patch.data.iter().all(|value| value.is_finite())
                && patch.jacobian_se2.iter().all(|value| value.is_finite())
                && patch
                    .h_se2_inv_j_se2_t
                    .iter()
                    .all(|value| value.is_finite());
        }

        patch
    }

    pub const fn sample_count(&self) -> usize {
        Pattern51::SAMPLE_COUNT
    }

    /// Computes Basalt's inverse-compositional increment
    /// `inc = -H⁻¹Jᵀ residual`.
    pub fn ic_increment(&self, residual: &PatchData51) -> Vector3<f32> {
        -gemv_packet8(&self.h_se2_inv_j_se2_t, residual)
    }

    /// Returns the mean-normalized residual at an affine warp. This mirrors
    /// Basalt's `OpticalFlowPatch::residual`: invalid source samples and
    /// target samples are zeroed, and more than half of the 52 samples must
    /// overlap for a usable residual.
    pub fn residual(
        &self,
        image: &RawU16Image,
        transform: &AffineCompact2f,
    ) -> Result<PatchData51, PatchResidualError> {
        let transformed = self.transformed_pattern(transform);
        let mut residual = PatchData51::from_element(-1.0);
        let mut sum = 0.0_f32;
        let mut valid_target_samples = 0_usize;

        for (index, point) in transformed.iter().enumerate() {
            if image.in_bounds(*point, 2.0) {
                if let Some(value) = image.interp(*point) {
                    residual[index] = value;
                    sum += value;
                    valid_target_samples += 1;
                }
            }
        }

        if valid_target_samples == 0 || sum <= EPS || !sum.is_finite() {
            return Err(PatchResidualError::NoValidTargetSamples);
        }

        let mut valid_residuals = 0_usize;
        for index in 0..Pattern51::SAMPLE_COUNT {
            if residual[index] >= 0.0 && self.data[index] >= 0.0 {
                let value = residual[index];
                residual[index] = valid_target_samples as f32 * value / sum - self.data[index];
                valid_residuals += 1;
            } else {
                residual[index] = 0.0;
            }
        }

        if valid_residuals <= Pattern51::SAMPLE_COUNT / 2 {
            return Err(PatchResidualError::InsufficientOverlap {
                valid_samples: valid_residuals,
            });
        }
        Ok(residual)
    }

    pub fn transformed_pattern(&self, transform: &AffineCompact2f) -> [Vector2<f32>; 52] {
        std::array::from_fn(|index| {
            let [x, y] = Pattern51::OFFSETS[index];
            transform.transform_point(Vector2::new(x, y))
        })
    }
}

#[inline(never)]
fn hessian_fma(jacobian: &PatchJacobian51) -> SMatrix<f32, 3, 3> {
    let mut hessian = SMatrix::<f32, 3, 3>::zeros();
    for row in 0..3 {
        for column in 0..3 {
            let mut value = 0.0_f32;
            for sample in 0..Pattern51::SAMPLE_COUNT {
                value = jacobian[(sample, row)].fused_mul_add(jacobian[(sample, column)], value);
            }
            hessian[(row, column)] = value;
        }
    }
    hessian
}

#[inline(never)]
fn hessian_inverse_jacobian_product(
    inverse: SMatrix<f32, 3, 3>,
    jacobian: &PatchJacobian51,
) -> PatchInverseJacobian51 {
    let mut product = PatchInverseJacobian51::zeros();
    for row in 0..3 {
        for sample in 0..Pattern51::SAMPLE_COUNT {
            let value = inverse[(row, 1)] * jacobian[(sample, 1)];
            let value = inverse[(row, 2)].fused_mul_add(jacobian[(sample, 2)], value);
            let value = inverse[(row, 0)].fused_mul_add(jacobian[(sample, 0)], value);
            product[(row, sample)] = value;
        }
    }
    product
}

/// Computes the fixed-size `3x52 * 52x1` GEMV in the order emitted by the
/// pinned Eigen 5.0.1 AVX packet-8 path.
///
/// Eigen's packet kernel is unrolled for this exact shape.  With three output
/// rows, GCC lowers the packet reduction to eight independent chunks per row;
/// each chunk is a short FMA chain and the chunks are combined with a fixed
/// binary tree.  A normal scalar loop, or a balanced reduction over all 52
/// products, therefore rounds differently even though it is mathematically
/// equivalent.  Keep the packet lanes and horizontal reduction explicit here
/// so this remains general for every 3x52 matrix and residual, rather than a
/// fixture-specific correction.
#[inline(never)]
fn gemv_packet8(matrix: &PatchInverseJacobian51, residual: &PatchData51) -> Vector3<f32> {
    Vector3::new(
        gemv_packet8_row(matrix, residual, 0),
        gemv_packet8_row(matrix, residual, 1),
        gemv_packet8_row(matrix, residual, 2),
    )
}

#[inline(never)]
fn gemv_packet8_row(matrix: &PatchInverseJacobian51, residual: &PatchData51, row: usize) -> f32 {
    let coefficients: [f32; Pattern51::SAMPLE_COUNT] =
        std::array::from_fn(|sample| matrix[(row, sample)]);
    let samples = residual.as_slice();

    // Each lane starts with a plain multiply, then folds the remaining terms
    // with f32::mul_add.  This is the exact vmulss/vfmadd231ss sequence from
    // Eigen's generated fixed-size kernel.
    let lane0 = {
        let value = coefficients[4] * samples[4];
        let value = coefficients[5].fused_mul_add(samples[5], value);
        coefficients[3].fused_mul_add(samples[3], value)
    };
    let lane1 = {
        let value = coefficients[1] * samples[1];
        let value = coefficients[2].fused_mul_add(samples[2], value);
        coefficients[0].fused_mul_add(samples[0], value)
    };
    let lane2 = {
        let value = coefficients[11] * samples[11];
        coefficients[12].fused_mul_add(samples[12], value)
    };
    let lane3 = {
        let value = coefficients[9] * samples[9];
        coefficients[10].fused_mul_add(samples[10], value)
    };
    let lane4 = {
        let value = coefficients[7] * samples[7];
        let value = coefficients[8].fused_mul_add(samples[8], value);
        coefficients[6].fused_mul_add(samples[6], value)
    };
    let lane5 = {
        let value = coefficients[17] * samples[17];
        let value = coefficients[18].fused_mul_add(samples[18], value);
        coefficients[16].fused_mul_add(samples[16], value)
    };
    let lane6 = {
        let value = coefficients[14] * samples[14];
        let value = coefficients[15].fused_mul_add(samples[15], value);
        coefficients[13].fused_mul_add(samples[13], value)
    };
    let lane7 = {
        let value = coefficients[24] * samples[24];
        coefficients[25].fused_mul_add(samples[25], value)
    };
    let lane8 = {
        let value = coefficients[22] * samples[22];
        coefficients[23].fused_mul_add(samples[23], value)
    };
    let lane9 = {
        let value = coefficients[20] * samples[20];
        let value = coefficients[21].fused_mul_add(samples[21], value);
        coefficients[19].fused_mul_add(samples[19], value)
    };
    let lane10 = {
        let value = coefficients[30] * samples[30];
        let value = coefficients[31].fused_mul_add(samples[31], value);
        coefficients[29].fused_mul_add(samples[29], value)
    };
    let lane11 = {
        let value = coefficients[27] * samples[27];
        let value = coefficients[28].fused_mul_add(samples[28], value);
        coefficients[26].fused_mul_add(samples[26], value)
    };
    let lane12 = {
        let value = coefficients[35] * samples[35];
        coefficients[36].fused_mul_add(samples[36], value)
    };
    let lane13 = {
        let value = coefficients[37] * samples[37];
        coefficients[38].fused_mul_add(samples[38], value)
    };
    let lane14 = {
        let value = coefficients[33] * samples[33];
        let value = coefficients[34].fused_mul_add(samples[34], value);
        coefficients[32].fused_mul_add(samples[32], value)
    };
    let lane15 = {
        let value = coefficients[43] * samples[43];
        let value = coefficients[44].fused_mul_add(samples[44], value);
        coefficients[42].fused_mul_add(samples[42], value)
    };
    let lane16 = {
        let value = coefficients[40] * samples[40];
        let value = coefficients[41].fused_mul_add(samples[41], value);
        coefficients[39].fused_mul_add(samples[39], value)
    };
    let lane17 = {
        let value = coefficients[48] * samples[48];
        coefficients[49].fused_mul_add(samples[49], value)
    };
    let lane18 = {
        let value = coefficients[50] * samples[50];
        coefficients[51].fused_mul_add(samples[51], value)
    };
    let lane19 = {
        let value = coefficients[46] * samples[46];
        let value = coefficients[47].fused_mul_add(samples[47], value);
        coefficients[45].fused_mul_add(samples[45], value)
    };

    // Horizontal reduction, including every intermediate parenthesis, from
    // the generated assembly.  Do not replace these adds with a loop.
    let left0 = lane0 + lane1;
    let left1 = (lane2 + lane3) + lane4;
    let left2 = left0 + left1;
    let left3 = lane5 + lane6;
    let left4 = (lane7 + lane8) + lane9;
    let left5 = left3 + left4;
    let left = left2 + left5;

    let right0 = lane10 + lane11;
    let right1 = (lane12 + lane13) + lane14;
    let right2 = right0 + right1;
    let right3 = lane15 + lane16;
    let right4 = (lane17 + lane18) + lane19;
    let right5 = right3 + right4;
    let right = right2 + right5;

    left + right
}

/// Small fixed-size equivalent of Eigen's pivoted `LDLT::solveInPlace`.
///
/// Basalt factors the 3x3 Hessian with Eigen's diagonal-pivoting LDLT and
/// solves it against the identity before multiplying by `J.transpose()`.
/// Nalgebra has no pivoted LDLT decomposition, while `try_inverse` uses a
/// different elimination path.  Keeping the factor and solve scalar makes
/// the f32 ordering explicit and preserves the pinned permutation contract.
fn ldlt_inverse(input: SMatrix<f32, 3, 3>) -> Option<SMatrix<f32, 3, 3>> {
    let mut matrix = input;
    let mut transpositions = [0_usize; 3];

    for k in 0..3 {
        let mut pivot = k;
        for index in (k + 1)..3 {
            if matrix[(index, index)].abs() > matrix[(pivot, pivot)].abs() {
                pivot = index;
            }
        }
        transpositions[k] = pivot;

        if pivot != k {
            for column in 0..k {
                let value = matrix[(k, column)];
                matrix[(k, column)] = matrix[(pivot, column)];
                matrix[(pivot, column)] = value;
            }
            let tail = 3 - pivot - 1;
            for offset in 0..tail {
                let row = pivot + 1 + offset;
                let value = matrix[(row, k)];
                matrix[(row, k)] = matrix[(row, pivot)];
                matrix[(row, pivot)] = value;
            }
            let value = matrix[(k, k)];
            matrix[(k, k)] = matrix[(pivot, pivot)];
            matrix[(pivot, pivot)] = value;
            for row in (k + 1)..pivot {
                let value = matrix[(row, k)];
                matrix[(row, k)] = matrix[(pivot, row)];
                matrix[(pivot, row)] = value;
            }
        }

        if k > 0 {
            let mut temp = [0.0_f32; 3];
            for column in 0..k {
                temp[column] = matrix[(column, column)] * matrix[(k, column)];
            }
            let mut diagonal_update = 0.0_f32;
            for column in 0..k {
                diagonal_update = matrix[(k, column)].fused_mul_add(temp[column], diagonal_update);
            }
            matrix[(k, k)] -= diagonal_update;
            for row in (k + 1)..3 {
                let mut update = 0.0_f32;
                for column in 0..k {
                    update = matrix[(row, column)].fused_mul_add(temp[column], update);
                }
                matrix[(row, k)] -= update;
            }
        }

        let diagonal = matrix[(k, k)];
        if !diagonal.is_finite() || diagonal == 0.0 {
            return None;
        }
        for row in (k + 1)..3 {
            matrix[(row, k)] /= diagonal;
        }
    }

    let mut result = SMatrix::<f32, 3, 3>::identity();
    for k in 0..3 {
        let pivot = transpositions[k];
        if pivot != k {
            result.swap_rows(k, pivot);
        }
    }

    // Unit-lower solve. Eigen's matrix solver updates the rows below the
    // current row in place, so retain that order rather than forming a dot
    // product for each row.
    for row in 0..3 {
        let row_value = result.row(row).into_owned();
        for below in (row + 1)..3 {
            let coefficient = matrix[(below, row)];
            for column in 0..3 {
                result[(below, column)] =
                    (-coefficient).fused_mul_add(row_value[column], result[(below, column)]);
            }
        }
    }

    for row in 0..3 {
        for column in 0..3 {
            result[(row, column)] /= matrix[(row, row)];
        }
    }

    // Unit-upper solve (the transpose of the stored lower factor).  Eigen's
    // matrix RHS path is row-major here: it forms the dot product of all
    // already solved rows, then performs one subtraction.  This differs from
    // updating each coefficient in place and is observable in f32.
    for row in (0..3).rev() {
        let known = 2 - row;
        for column in 0..3 {
            let mut dot = 0.0_f32;
            for offset in 0..known {
                let coefficient = matrix[(row + 1 + offset, row)];
                let value = result[(row + 1 + offset, column)];
                dot = coefficient.fused_mul_add(value, dot);
            }
            result[(row, column)] -= dot;
        }
    }

    for k in (0..3).rev() {
        let pivot = transpositions[k];
        if pivot != k {
            result.swap_rows(k, pivot);
        }
    }
    Some(result)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchResidualError {
    NoValidTargetSamples,
    InsufficientOverlap { valid_samples: usize },
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector2;

    use super::*;

    #[test]
    fn all_black_patch_is_invalid_without_non_finite_outputs() {
        let image = RawU16Image::new(32, 32, vec![0; 32 * 32]).unwrap();
        let patch = MeanNormalizedPatch51::from_image(&image, Vector2::new(16.0, 16.0));
        assert!(!patch.valid);
        assert_eq!(patch.valid_samples, 52);
        assert!(patch.data.iter().all(|value| *value == 0.0));
        assert!(patch.jacobian_se2.iter().all(|value| *value == 0.0));
        assert!(patch.h_se2_inv_j_se2_t.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn hessian_fma_has_exact_fixed_fixture() {
        let mut jacobian = PatchJacobian51::zeros();
        jacobian
            .row_mut(0)
            .copy_from(&SVector::<f32, 3>::from_row_slice(&[1.0, 2.0, 3.0]).transpose());
        jacobian
            .row_mut(1)
            .copy_from(&SVector::<f32, 3>::from_row_slice(&[4.0, 5.0, 6.0]).transpose());
        jacobian
            .row_mut(2)
            .copy_from(&SVector::<f32, 3>::from_row_slice(&[-1.0, 0.5, -2.0]).transpose());

        let hessian = hessian_fma(&jacobian);
        let expected = SMatrix::<f32, 3, 3>::from_row_slice(&[
            18.0, 21.5, 29.0, 21.5, 29.25, 35.0, 29.0, 35.0, 49.0,
        ]);
        for row in 0..3 {
            for column in 0..3 {
                assert_eq!(
                    hessian[(row, column)].to_bits(),
                    expected[(row, column)].to_bits()
                );
            }
        }
    }

    #[test]
    fn ldlt_inverse_matches_native_eigen_fixture() {
        let hessian = SMatrix::<f32, 3, 3>::from_row_slice(&[
            f32::from_bits(0x3e9802f4),
            f32::from_bits(0x3c717f13),
            f32::from_bits(0xbf754e2c),
            f32::from_bits(0x3c717f13),
            f32::from_bits(0x3f0a99cc),
            f32::from_bits(0x3f191dcc),
            f32::from_bits(0xbf754e2c),
            f32::from_bits(0x3f191dcc),
            f32::from_bits(0x40f9661a),
        ]);
        let inverse = ldlt_inverse(hessian).expect("native fixture is nonsingular");
        let expected = [
            0x40c3e34e, 0xbf8b97a3, 0x3f56191e, 0xbf8b97a3, 0x400d983b, 0xbe9b94da, 0x3f56191e,
            0xbe9b94db, 0x3e82479c,
        ];
        for row in 0..3 {
            for column in 0..3 {
                assert_eq!(inverse[(row, column)].to_bits(), expected[row * 3 + column]);
            }
        }
    }

    #[test]
    fn m7ci_frame4_cam0_ldlt_boundary_is_recorded() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            hessian_bits: Vec<String>,
            native_inverse_bits: Vec<String>,
            rust_inverse_bits: Vec<String>,
            native_l3_i0_increment_bits: Vec<String>,
            rust_l3_i0_increment_bits: Vec<String>,
        }

        fn bits(value: &str) -> u32 {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("fixture bit pattern")
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../tests/fixtures/m7ci_frame4_cam0_ldlt_boundary.json"
        ))
        .expect("m7ci LDLT boundary fixture must parse");
        assert_eq!(fixture.hessian_bits.len(), 9);
        assert_eq!(fixture.native_inverse_bits.len(), 9);
        assert_eq!(fixture.rust_inverse_bits.len(), 9);
        assert_eq!(fixture.native_l3_i0_increment_bits.len(), 3);
        assert_eq!(fixture.rust_l3_i0_increment_bits.len(), 3);

        let hessian = SMatrix::<f32, 3, 3>::from_row_slice(
            &fixture
                .hessian_bits
                .iter()
                .map(|value| f32::from_bits(bits(value)))
                .collect::<Vec<_>>(),
        );
        let inverse = ldlt_inverse(hessian).expect("m7ci Hessian is nonsingular");
        let observed: Vec<u32> = (0..3)
            .flat_map(|row| (0..3).map(move |column| inverse[(row, column)].to_bits()))
            .collect();
        let expected_rust: Vec<u32> = fixture
            .rust_inverse_bits
            .iter()
            .map(|value| bits(value))
            .collect();
        let expected_native: Vec<u32> = fixture
            .native_inverse_bits
            .iter()
            .map(|value| bits(value))
            .collect();
        // The fixture's Rust inverse is the pre-M7cm baseline.  Preserve its
        // two-lane difference as provenance while requiring the retained
        // general solve to reproduce the native inverse on all nine lanes.
        let baseline_mismatches: Vec<usize> = expected_rust
            .iter()
            .zip(expected_native.iter())
            .enumerate()
            .filter_map(|(index, (rust, native))| (rust != native).then_some(index))
            .collect();
        assert_eq!(baseline_mismatches, vec![2, 8]);
        assert_eq!(observed, expected_native);

        let native_increment: Vec<u32> = fixture
            .native_l3_i0_increment_bits
            .iter()
            .map(|value| bits(value))
            .collect();
        let rust_increment: Vec<u32> = fixture
            .rust_l3_i0_increment_bits
            .iter()
            .map(|value| bits(value))
            .collect();
        assert_eq!(native_increment[1], rust_increment[1]);
        assert_ne!(native_increment[0], rust_increment[0]);
        assert_ne!(native_increment[2], rust_increment[2]);
    }

    #[test]
    fn ic_increment_applies_stored_inverse_jacobian() {
        let mut patch = MeanNormalizedPatch51 {
            position: Vector2::zeros(),
            mean: 1.0,
            data: PatchData51::zeros(),
            jacobian_se2: PatchJacobian51::zeros(),
            h_se2_inv_j_se2_t: PatchInverseJacobian51::zeros(),
            valid_samples: Pattern51::SAMPLE_COUNT,
            valid: true,
        };
        patch.h_se2_inv_j_se2_t[(0, 0)] = 1.0;
        patch.h_se2_inv_j_se2_t[(1, 1)] = 1.0;
        patch.h_se2_inv_j_se2_t[(2, 2)] = 1.0;
        let mut residual = PatchData51::zeros();
        residual[0] = 1.0;
        residual[1] = -2.0;
        residual[2] = 3.0;
        assert_eq!(patch.ic_increment(&residual), Vector3::new(-1.0, 2.0, -3.0));
    }

    #[test]
    fn packet8_gemv_matches_native_l3_i0_and_i1_all_lanes() {
        // L3 is the new-stereo fixed 3x52 fixture captured from Eigen 5.0.1.
        // Store bits so the test checks the contraction, not decimal parsing.
        let matrix_bits: [[u32; 52]; 3] = [
            [
                0x3f59213a, 0x3f410d7f, 0x3eaf49ef, 0x3dd03e71, 0x3f006704, 0x3f1126f8, 0x3eab6645,
                0x3d8ffa42, 0x3e0f6660, 0x3dc8a001, 0x00000000, 0xbcbcbb1e, 0x3e466735, 0x3e04468b,
                0x3a6dcb42, 0x3e1b0fbd, 0x3d665d19, 0xbe83e108, 0x00000000, 0xbee13c8a, 0x3e81ca36,
                0x3e778a0a, 0xbc145354, 0x3ccdf4cc, 0xbe69173a, 0xbe735234, 0x00000000, 0xbde4e2fa,
                0x3efa0153, 0x3d9f812f, 0xbe7580e7, 0xbe38b9d2, 0xbee49c5e, 0xbe4ecbc0, 0x00000000,
                0x3e260869, 0x3e755892, 0xbdd318e3, 0xbe76fde5, 0xbea948b9, 0xbf0ac07e, 0xbd76abcb,
                0x3ebbb768, 0x3c93d920, 0xbde93028, 0xbdd32bb1, 0xbee8825d, 0xbf1ea438, 0x3ebcb2e7,
                0xbcbc3b80, 0xbf004175, 0xbf882dc7,
            ],
            [
                0xbe7427ca, 0xbe5fa99d, 0xbe006b56, 0xbe13597d, 0x3d367290, 0xb99efcd3, 0x3d0357ad,
                0x3daa4b36, 0xbd0f8222, 0xbe127837, 0x00000000, 0x3eacc78f, 0x3ea3b21b, 0x3e57a358,
                0x3e355d6d, 0x3d2aa022, 0x3d1c6742, 0x3da6008f, 0x00000000, 0x3ec217c1, 0x3e7da7e2,
                0x3e249496, 0x3e4c1153, 0x3e0a101b, 0x3e471931, 0x3e3e0389, 0x00000000, 0x3d380c6b,
                0xbd514668, 0x3dbe95dd, 0x3e28338f, 0x3e1a7cbc, 0x3e526075, 0x3df810b2, 0x00000000,
                0xbe6f1b0c, 0xbe13456e, 0x3d89981d, 0x3d43d168, 0x3d917dde, 0x3e0735f4, 0xbc25fe83,
                0xbeec3189, 0xbeacfa84, 0xbe5e811e, 0xbe4c6615, 0xbd28d2e7, 0x3d8d1813, 0xbf1c1871,
                0xbf04d75a, 0xbe9a99f5, 0xbb3a5a6d,
            ],
            [
                0x3d86c988, 0x3db18801, 0x3de7303e, 0x3e155862, 0xbcd599e9, 0x3c581b60, 0x3cfeca1c,
                0x3d5542d7, 0x3dbc50b3, 0x3dcfe007, 0x00000000, 0xbdf0ff83, 0xbd879a03, 0xbc7b47a6,
                0x3c69a669, 0x3d812b27, 0x3da95bae, 0x3da050d0, 0x00000000, 0xbe1c91f4, 0xbd1c4cd8,
                0xba8a20f0, 0xbb744310, 0x3cf68e05, 0x3ce8a37b, 0x3d5294d2, 0x00000000, 0xbd4f985a,
                0x3d315861, 0xbc5a6299, 0xbd2e30d1, 0xbc7191f5, 0xbd0e9c3e, 0x3c2768f5, 0x00000000,
                0x3cfcfd98, 0x3ce52451, 0xbd1b3424, 0xbd41e77f, 0xbd5c8a5e, 0xbda39c03, 0xbad0a008,
                0x3dbaaad5, 0x3c8868b4, 0xbca702ad, 0xbcc05e6f, 0xbdb74dc9, 0xbdebddb5, 0x3dbdafe5,
                0x3bbdf5dd, 0xbdd2a5df, 0xbe5fe4ca,
            ],
        ];
        let matrix =
            PatchInverseJacobian51::from_fn(|row, sample| f32::from_bits(matrix_bits[row][sample]));
        let patch = MeanNormalizedPatch51 {
            position: Vector2::zeros(),
            mean: 1.0,
            data: PatchData51::zeros(),
            jacobian_se2: PatchJacobian51::zeros(),
            h_se2_inv_j_se2_t: matrix,
            valid_samples: Pattern51::SAMPLE_COUNT,
            valid: true,
        };
        let residual_bits: [[u32; 52]; 2] = [
            [
                0xbe77c550, 0xbe87d734, 0xbe48fe40, 0xbdc51d60, 0xbe83b514, 0xbeccd728, 0xbed7e194,
                0xbe79ceb0, 0xbe27fd40, 0xbde60700, 0x00000000, 0xbe526480, 0xbebdc230, 0xbec6bd1c,
                0xbe563ad8, 0xbe606c28, 0xbe3a16a0, 0xbe0cdc08, 0x00000000, 0xbcc4cc80, 0xbdec47b8,
                0xbe4ea61c, 0xbdce7f90, 0xbe204e00, 0xbde5f6c0, 0xbd848ad0, 0x00000000, 0x3dd44d58,
                0x3d3df570, 0xbd9c0900, 0xbce18480, 0xbd1942f0, 0x3cd98220, 0x3d4d55a0, 0x00000000,
                0x3e536efc, 0x3e6a6cd8, 0x3e3ccfbc, 0x3e41cf48, 0x3e2973a0, 0x3e318354, 0x3df451dc,
                0x3e7fe6e8, 0x3eb87c86, 0x3ee1564c, 0x3ee53f74, 0x3ec1738e, 0x3e9cdeeb, 0x3e4f6cf0,
                0x3eaef570, 0x3ecde4f8, 0x3ecf64ec,
            ],
            [
                0x3d6bad40, 0x3db375f0, 0x3e2e0220, 0x3e634900, 0xbd6b2c20, 0xbb0b0000, 0x3cc16500,
                0x3d971650, 0x3e101f68, 0x3e099258, 0x00000000, 0xbe6d8100, 0xbd11f420, 0xbcd9b040,
                0x3d832c50, 0x3df885d0, 0x3e07f700, 0x3dbe9ec0, 0x00000000, 0xbd09bf00, 0x3da001a0,
                0xbd330420, 0x3d707240, 0x3d97f938, 0x3dca9d30, 0x3de93898, 0x00000000, 0x3d80aba0,
                0x3d1a6410, 0xbdaee2c8, 0x3b223200, 0xbc9455c0, 0xbc333a80, 0x3d669960, 0x00000000,
                0x3b5acb00, 0xbcba85a0, 0xbd69e8c0, 0xbd618090, 0xbde51978, 0xbd56bc30, 0x3d582988,
                0x3d9aa5a0, 0xbbe9e280, 0xbcd6cf00, 0xbdd37098, 0xbe3a2fb6, 0xbd36a8c0, 0xbcd7d540,
                0xbe33b25c, 0xbe8dd83a, 0xbe60f126,
            ],
        ];
        let expected = [
            [0x400fc2da, 0x3f9be212, 0x3ea0024b],
            [0xbf47a90b, 0xbd615534, 0xbe91b7c1],
        ];

        for fixture in 0..2 {
            let residual =
                PatchData51::from_fn(|sample, _| f32::from_bits(residual_bits[fixture][sample]));
            let increment = patch.ic_increment(&residual);
            for lane in 0..3 {
                assert_eq!(
                    increment[lane].to_bits(),
                    expected[fixture][lane],
                    "fixture {fixture}, lane {lane}"
                );
            }
        }
    }

    #[test]
    fn packet8_gemv_matches_native_existing_stereo_l3_i0_all_lanes() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            matrix_bits: Vec<Vec<u32>>,
            residual_bits: Vec<u32>,
            expected_increment_bits: Vec<u32>,
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../tests/fixtures/m7cb_existing_stereo_gemv.json"
        ))
        .expect("existing-stereo packet8 fixture must parse");
        assert_eq!(fixture.matrix_bits.len(), 3);
        assert!(fixture.matrix_bits.iter().all(|row| row.len() == 52));
        assert_eq!(fixture.residual_bits.len(), 52);
        assert_eq!(fixture.expected_increment_bits.len(), 3);
        let matrix = PatchInverseJacobian51::from_fn(|row, sample| {
            f32::from_bits(fixture.matrix_bits[row][sample])
        });
        let patch = MeanNormalizedPatch51 {
            position: Vector2::zeros(),
            mean: 1.0,
            data: PatchData51::zeros(),
            jacobian_se2: PatchJacobian51::zeros(),
            h_se2_inv_j_se2_t: matrix,
            valid_samples: Pattern51::SAMPLE_COUNT,
            valid: true,
        };
        let residual =
            PatchData51::from_fn(|sample, _| f32::from_bits(fixture.residual_bits[sample]));
        let increment = patch.ic_increment(&residual);
        for lane in 0..3 {
            assert_eq!(
                increment[lane].to_bits(),
                fixture.expected_increment_bits[lane],
                "existing-stereo L3/I0 lane {lane}"
            );
        }
    }
}
