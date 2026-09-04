// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// This module is the Rust implementation boundary for PoseLib's generated
// generalized-relative-pose six-point solver:
// https://github.com/PoseLib/PoseLib/blob/master/PoseLib/solvers/gen_relpose_6pt.cc

use nalgebra::{DMatrix, Matrix3, Matrix6, Quaternion, SMatrix, SVector, UnitQuaternion, Vector3};

use super::gr6p_data::{C0_IND, C1_IND, COEFFS0_IND, COEFFS1_IND, PT_INDEX};

/// A single bearing observation in one generalized-camera rig.
///
/// `origin` is the camera center in the rig frame and `bearing` is a ray in
/// that same frame. Constructors normalize non-zero finite bearings so callers
/// do not have to reproduce the unit-vector convention used by GR6P.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneralizedCameraObservation {
    pub origin: Vector3<f64>,
    pub bearing: Vector3<f64>,
}

impl GeneralizedCameraObservation {
    /// Construct an observation and normalize its bearing.
    pub fn new(
        origin: Vector3<f64>,
        bearing: Vector3<f64>,
    ) -> Result<Self, GeneralizedRelativePoseError> {
        if !origin.iter().all(|value| value.is_finite())
            || !bearing.iter().all(|value| value.is_finite())
        {
            return Err(GeneralizedRelativePoseError::NonFiniteObservation);
        }
        let norm = bearing.norm();
        if !norm.is_finite() || norm <= f64::EPSILON {
            return Err(GeneralizedRelativePoseError::ZeroLengthBearing);
        }
        Ok(Self {
            origin,
            bearing: bearing / norm,
        })
    }
}

/// A matched pair of generalized-camera observations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneralizedRelativeCorrespondence {
    pub rig1: GeneralizedCameraObservation,
    pub rig2: GeneralizedCameraObservation,
}

/// Configuration shared by GR6P validation and the polynomial solver.
/// The tolerances are deliberately explicit so mapper integration can freeze
/// them when the solver is enabled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneralizedRelativePoseConfig {
    /// Minimum acceptable norm for a bearing before normalization.
    pub bearing_epsilon: f64,
    /// Minimum spread of the camera origins in each rig.
    pub origin_spread_epsilon: f64,
}

impl Default for GeneralizedRelativePoseConfig {
    fn default() -> Self {
        Self {
            bearing_epsilon: 1.0e-12,
            origin_spread_epsilon: 1.0e-12,
        }
    }
}

/// A metric transform mapping rig 1 coordinates to rig 2 coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneralizedRelativePose {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
}

/// Errors detected before the polynomial solver is entered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneralizedRelativePoseError {
    /// GR6P is a six-correspondence minimal solver.
    WrongCorrespondenceCount { expected: usize, actual: usize },
    /// An origin or bearing contains NaN or infinity.
    NonFiniteObservation,
    /// A bearing cannot be normalized.
    ZeroLengthBearing,
    /// A supplied tolerance is not finite and positive.
    InvalidConfiguration,
}

/// Public result type for a generalized-relative-pose estimate.
pub type GeneralizedRelativePoseResult =
    Result<Vec<GeneralizedRelativePose>, GeneralizedRelativePoseError>;

/// Validate and normalize the six observations required by GR6P.
pub fn normalize_gr6p_correspondences(
    correspondences: &[GeneralizedRelativeCorrespondence],
    config: &GeneralizedRelativePoseConfig,
) -> Result<[GeneralizedRelativeCorrespondence; 6], GeneralizedRelativePoseError> {
    if correspondences.len() != 6 {
        return Err(GeneralizedRelativePoseError::WrongCorrespondenceCount {
            expected: 6,
            actual: correspondences.len(),
        });
    }
    if !config.bearing_epsilon.is_finite()
        || config.bearing_epsilon <= 0.0
        || !config.origin_spread_epsilon.is_finite()
        || config.origin_spread_epsilon <= 0.0
    {
        return Err(GeneralizedRelativePoseError::InvalidConfiguration);
    }

    let mut normalized = [GeneralizedRelativeCorrespondence {
        rig1: GeneralizedCameraObservation {
            origin: Vector3::zeros(),
            bearing: Vector3::z(),
        },
        rig2: GeneralizedCameraObservation {
            origin: Vector3::zeros(),
            bearing: Vector3::z(),
        },
    }; 6];
    for (index, correspondence) in correspondences.iter().enumerate() {
        for observation in [correspondence.rig1, correspondence.rig2] {
            if !observation.origin.iter().all(|value| value.is_finite())
                || !observation.bearing.iter().all(|value| value.is_finite())
            {
                return Err(GeneralizedRelativePoseError::NonFiniteObservation);
            }
        }
        let norm1 = correspondence.rig1.bearing.norm();
        let norm2 = correspondence.rig2.bearing.norm();
        if !norm1.is_finite()
            || !norm2.is_finite()
            || norm1 <= config.bearing_epsilon
            || norm2 <= config.bearing_epsilon
        {
            return Err(GeneralizedRelativePoseError::ZeroLengthBearing);
        }
        normalized[index] = GeneralizedRelativeCorrespondence {
            rig1: GeneralizedCameraObservation {
                origin: correspondence.rig1.origin,
                bearing: correspondence.rig1.bearing / norm1,
            },
            rig2: GeneralizedCameraObservation {
                origin: correspondence.rig2.origin,
                bearing: correspondence.rig2.bearing / norm2,
            },
        };
    }
    Ok(normalized)
}

/// Estimate a metric rig-to-rig pose from exactly six generalized bearings.
///
/// The generated PoseLib polynomial core returns all finite, cheiral solutions.
pub fn estimate_gr6p(
    correspondences: &[GeneralizedRelativeCorrespondence],
) -> GeneralizedRelativePoseResult {
    estimate_gr6p_with_config(correspondences, &GeneralizedRelativePoseConfig::default())
}

/// Configurable form of estimate_gr6p.
pub fn estimate_gr6p_with_config(
    correspondences: &[GeneralizedRelativeCorrespondence],
    config: &GeneralizedRelativePoseConfig,
) -> GeneralizedRelativePoseResult {
    let normalized = normalize_gr6p_correspondences(correspondences, config)?;
    if origin_spread(&normalized, true) <= config.origin_spread_epsilon
        && origin_spread(&normalized, false) <= config.origin_spread_epsilon
    {
        return Ok(Vec::new());
    }
    solve_gr6p_polynomial(&normalized)
}

fn origin_spread(correspondences: &[GeneralizedRelativeCorrespondence; 6], first_rig: bool) -> f64 {
    let reference = if first_rig {
        correspondences[0].rig1.origin
    } else {
        correspondences[0].rig2.origin
    };
    correspondences
        .iter()
        .map(|correspondence| {
            let origin = if first_rig {
                correspondence.rig1.origin
            } else {
                correspondence.rig2.origin
            };
            (origin - reference).norm()
        })
        .fold(0.0, f64::max)
}

fn mul2_2(a: &[f64], b: &[f64], c: &mut [f64]) {
    c[0] = a[0] * b[0];

    c[1] = a[0] * b[1] + a[1] * b[0];

    c[2] = a[0] * b[3] + a[3] * b[0];

    c[3] = a[0] * b[6] + a[6] * b[0];

    c[4] = a[0] * b[2] + a[1] * b[1] + a[2] * b[0];

    c[5] = a[0] * b[4] + a[1] * b[3] + a[3] * b[1] + a[4] * b[0];

    c[6] = a[0] * b[7] + a[1] * b[6] + a[6] * b[1] + a[7] * b[0];

    c[7] = a[0] * b[5] + a[5] * b[0] + a[3] * b[3];

    c[8] = a[0] * b[8] + a[8] * b[0] + a[3] * b[6] + a[6] * b[3];

    c[9] = a[0] * b[9] + a[9] * b[0] + a[6] * b[6];

    c[10] = a[1] * b[2] + a[2] * b[1];

    c[11] = a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1];

    c[12] = a[1] * b[7] + a[2] * b[6] + a[6] * b[2] + a[7] * b[1];

    c[13] = a[1] * b[5] + a[5] * b[1] + a[3] * b[4] + a[4] * b[3];

    c[14] = a[1] * b[8] + a[8] * b[1] + a[3] * b[7] + a[4] * b[6] + a[6] * b[4] + a[7] * b[3];

    c[15] = a[1] * b[9] + a[9] * b[1] + a[6] * b[7] + a[7] * b[6];

    c[16] = a[3] * b[5] + a[5] * b[3];

    c[17] = a[3] * b[8] + a[5] * b[6] + a[6] * b[5] + a[8] * b[3];

    c[18] = a[3] * b[9] + a[9] * b[3] + a[6] * b[8] + a[8] * b[6];

    c[19] = a[6] * b[9] + a[9] * b[6];

    c[20] = a[2] * b[2];

    c[21] = a[2] * b[4] + a[4] * b[2];

    c[22] = a[2] * b[7] + a[7] * b[2];

    c[23] = a[2] * b[5] + a[5] * b[2] + a[4] * b[4];

    c[24] = a[2] * b[8] + a[8] * b[2] + a[4] * b[7] + a[7] * b[4];

    c[25] = a[2] * b[9] + a[9] * b[2] + a[7] * b[7];

    c[26] = a[4] * b[5] + a[5] * b[4];

    c[27] = a[4] * b[8] + a[5] * b[7] + a[7] * b[5] + a[8] * b[4];

    c[28] = a[4] * b[9] + a[9] * b[4] + a[7] * b[8] + a[8] * b[7];

    c[29] = a[7] * b[9] + a[9] * b[7];

    c[30] = a[5] * b[5];

    c[31] = a[5] * b[8] + a[8] * b[5];

    c[32] = a[5] * b[9] + a[9] * b[5] + a[8] * b[8];

    c[33] = a[8] * b[9] + a[9] * b[8];

    c[34] = a[9] * b[9];
}

fn mul2_2m(a: &[f64], b: &[f64], c: &mut [f64]) {
    c[0] -= a[0] * b[0];

    c[1] -= a[0] * b[1] + a[1] * b[0];

    c[2] -= a[0] * b[3] + a[3] * b[0];

    c[3] -= a[0] * b[6] + a[6] * b[0];

    c[4] -= a[0] * b[2] + a[1] * b[1] + a[2] * b[0];

    c[5] -= a[0] * b[4] + a[1] * b[3] + a[3] * b[1] + a[4] * b[0];

    c[6] -= a[0] * b[7] + a[1] * b[6] + a[6] * b[1] + a[7] * b[0];

    c[7] -= a[0] * b[5] + a[5] * b[0] + a[3] * b[3];

    c[8] -= a[0] * b[8] + a[8] * b[0] + a[3] * b[6] + a[6] * b[3];

    c[9] -= a[0] * b[9] + a[9] * b[0] + a[6] * b[6];

    c[10] -= a[1] * b[2] + a[2] * b[1];

    c[11] -= a[1] * b[4] + a[2] * b[3] + a[3] * b[2] + a[4] * b[1];

    c[12] -= a[1] * b[7] + a[2] * b[6] + a[6] * b[2] + a[7] * b[1];

    c[13] -= a[1] * b[5] + a[5] * b[1] + a[3] * b[4] + a[4] * b[3];

    c[14] -= a[1] * b[8] + a[8] * b[1] + a[3] * b[7] + a[4] * b[6] + a[6] * b[4] + a[7] * b[3];

    c[15] -= a[1] * b[9] + a[9] * b[1] + a[6] * b[7] + a[7] * b[6];

    c[16] -= a[3] * b[5] + a[5] * b[3];

    c[17] -= a[3] * b[8] + a[5] * b[6] + a[6] * b[5] + a[8] * b[3];

    c[18] -= a[3] * b[9] + a[9] * b[3] + a[6] * b[8] + a[8] * b[6];

    c[19] -= a[6] * b[9] + a[9] * b[6];

    c[20] -= a[2] * b[2];

    c[21] -= a[2] * b[4] + a[4] * b[2];

    c[22] -= a[2] * b[7] + a[7] * b[2];

    c[23] -= a[2] * b[5] + a[5] * b[2] + a[4] * b[4];

    c[24] -= a[2] * b[8] + a[8] * b[2] + a[4] * b[7] + a[7] * b[4];

    c[25] -= a[2] * b[9] + a[9] * b[2] + a[7] * b[7];

    c[26] -= a[4] * b[5] + a[5] * b[4];

    c[27] -= a[4] * b[8] + a[5] * b[7] + a[7] * b[5] + a[8] * b[4];

    c[28] -= a[4] * b[9] + a[9] * b[4] + a[7] * b[8] + a[8] * b[7];

    c[29] -= a[7] * b[9] + a[9] * b[7];

    c[30] -= a[5] * b[5];

    c[31] -= a[5] * b[8] + a[8] * b[5];

    c[32] -= a[5] * b[9] + a[9] * b[5] + a[8] * b[8];

    c[33] -= a[8] * b[9] + a[9] * b[8];

    c[34] -= a[9] * b[9];
}

fn mul2_4p(a: &[f64], b: &[f64], c: &mut [f64]) {
    c[0] += a[0] * b[0];

    c[1] += a[0] * b[1] + a[1] * b[0];

    c[2] += a[1] * b[1] + a[2] * b[0] + a[0] * b[4];

    c[3] += a[2] * b[1] + a[1] * b[4] + a[0] * b[10];

    c[4] += a[2] * b[4] + a[1] * b[10] + a[0] * b[20];

    c[5] += a[2] * b[10] + a[1] * b[20];

    c[6] += a[2] * b[20];

    c[7] += a[0] * b[2] + a[3] * b[0];

    c[8] += a[1] * b[2] + a[3] * b[1] + a[4] * b[0] + a[0] * b[5];

    c[9] += a[2] * b[2] + a[4] * b[1] + a[1] * b[5] + a[3] * b[4] + a[0] * b[11];

    c[10] += a[2] * b[5] + a[4] * b[4] + a[1] * b[11] + a[3] * b[10] + a[0] * b[21];

    c[11] += a[2] * b[11] + a[4] * b[10] + a[1] * b[21] + a[3] * b[20];

    c[12] += a[2] * b[21] + a[4] * b[20];

    c[13] += a[3] * b[2] + a[5] * b[0] + a[0] * b[7];

    c[14] += a[4] * b[2] + a[5] * b[1] + a[1] * b[7] + a[3] * b[5] + a[0] * b[13];

    c[15] += a[2] * b[7] + a[4] * b[5] + a[5] * b[4] + a[1] * b[13] + a[3] * b[11] + a[0] * b[23];

    c[16] += a[2] * b[13] + a[4] * b[11] + a[5] * b[10] + a[1] * b[23] + a[3] * b[21];

    c[17] += a[2] * b[23] + a[4] * b[21] + a[5] * b[20];

    c[18] += a[5] * b[2] + a[3] * b[7] + a[0] * b[16];

    c[19] += a[5] * b[5] + a[4] * b[7] + a[3] * b[13] + a[1] * b[16] + a[0] * b[26];

    c[20] += a[5] * b[11] + a[4] * b[13] + a[2] * b[16] + a[3] * b[23] + a[1] * b[26];

    c[21] += a[5] * b[21] + a[4] * b[23] + a[2] * b[26];

    c[22] += a[5] * b[7] + a[3] * b[16] + a[0] * b[30];

    c[23] += a[5] * b[13] + a[4] * b[16] + a[3] * b[26] + a[1] * b[30];

    c[24] += a[5] * b[23] + a[4] * b[26] + a[2] * b[30];

    c[25] += a[5] * b[16] + a[3] * b[30];

    c[26] += a[5] * b[26] + a[4] * b[30];

    c[27] += a[5] * b[30];

    c[28] += a[0] * b[3] + a[6] * b[0];

    c[29] += a[1] * b[3] + a[0] * b[6] + a[6] * b[1] + a[7] * b[0];

    c[30] += a[2] * b[3] + a[1] * b[6] + a[7] * b[1] + a[6] * b[4] + a[0] * b[12];

    c[31] += a[2] * b[6] + a[7] * b[4] + a[1] * b[12] + a[6] * b[10] + a[0] * b[22];

    c[32] += a[2] * b[12] + a[7] * b[10] + a[1] * b[22] + a[6] * b[20];

    c[33] += a[2] * b[22] + a[7] * b[20];

    c[34] += a[3] * b[3] + a[0] * b[8] + a[6] * b[2] + a[8] * b[0];

    c[35] += a[4] * b[3]
        + a[1] * b[8]
        + a[3] * b[6]
        + a[7] * b[2]
        + a[8] * b[1]
        + a[6] * b[5]
        + a[0] * b[14];

    c[36] += a[2] * b[8]
        + a[4] * b[6]
        + a[7] * b[5]
        + a[8] * b[4]
        + a[1] * b[14]
        + a[3] * b[12]
        + a[6] * b[11]
        + a[0] * b[24];

    c[37] += a[2] * b[14]
        + a[4] * b[12]
        + a[7] * b[11]
        + a[8] * b[10]
        + a[1] * b[24]
        + a[3] * b[22]
        + a[6] * b[21];

    c[38] += a[2] * b[24] + a[4] * b[22] + a[7] * b[21] + a[8] * b[20];

    c[39] += a[5] * b[3] + a[8] * b[2] + a[3] * b[8] + a[6] * b[7] + a[0] * b[17];

    c[40] += a[5] * b[6]
        + a[4] * b[8]
        + a[8] * b[5]
        + a[7] * b[7]
        + a[3] * b[14]
        + a[1] * b[17]
        + a[6] * b[13]
        + a[0] * b[27];

    c[41] += a[5] * b[12]
        + a[4] * b[14]
        + a[2] * b[17]
        + a[8] * b[11]
        + a[7] * b[13]
        + a[3] * b[24]
        + a[1] * b[27]
        + a[6] * b[23];

    c[42] += a[5] * b[22] + a[4] * b[24] + a[2] * b[27] + a[8] * b[21] + a[7] * b[23];

    c[43] += a[5] * b[8] + a[8] * b[7] + a[3] * b[17] + a[6] * b[16] + a[0] * b[31];

    c[44] += a[5] * b[14]
        + a[4] * b[17]
        + a[8] * b[13]
        + a[7] * b[16]
        + a[3] * b[27]
        + a[1] * b[31]
        + a[6] * b[26];

    c[45] += a[5] * b[24] + a[4] * b[27] + a[8] * b[23] + a[2] * b[31] + a[7] * b[26];

    c[46] += a[5] * b[17] + a[8] * b[16] + a[3] * b[31] + a[6] * b[30];

    c[47] += a[5] * b[27] + a[8] * b[26] + a[4] * b[31] + a[7] * b[30];

    c[48] += a[5] * b[31] + a[8] * b[30];

    c[49] += a[0] * b[9] + a[6] * b[3] + a[9] * b[0];

    c[50] += a[1] * b[9] + a[7] * b[3] + a[9] * b[1] + a[6] * b[6] + a[0] * b[15];

    c[51] += a[2] * b[9] + a[7] * b[6] + a[9] * b[4] + a[1] * b[15] + a[6] * b[12] + a[0] * b[25];

    c[52] += a[2] * b[15] + a[7] * b[12] + a[9] * b[10] + a[1] * b[25] + a[6] * b[22];

    c[53] += a[2] * b[25] + a[7] * b[22] + a[9] * b[20];

    c[54] += a[8] * b[3] + a[9] * b[2] + a[3] * b[9] + a[6] * b[8] + a[0] * b[18];

    c[55] += a[4] * b[9]
        + a[8] * b[6]
        + a[9] * b[5]
        + a[7] * b[8]
        + a[3] * b[15]
        + a[1] * b[18]
        + a[6] * b[14]
        + a[0] * b[28];

    c[56] += a[4] * b[15]
        + a[2] * b[18]
        + a[8] * b[12]
        + a[9] * b[11]
        + a[7] * b[14]
        + a[3] * b[25]
        + a[1] * b[28]
        + a[6] * b[24];

    c[57] += a[4] * b[25] + a[2] * b[28] + a[8] * b[22] + a[9] * b[21] + a[7] * b[24];

    c[58] += a[5] * b[9] + a[8] * b[8] + a[9] * b[7] + a[3] * b[18] + a[6] * b[17] + a[0] * b[32];

    c[59] += a[5] * b[15]
        + a[4] * b[18]
        + a[8] * b[14]
        + a[9] * b[13]
        + a[7] * b[17]
        + a[3] * b[28]
        + a[1] * b[32]
        + a[6] * b[27];

    c[60] +=
        a[5] * b[25] + a[4] * b[28] + a[8] * b[24] + a[9] * b[23] + a[2] * b[32] + a[7] * b[27];

    c[61] += a[5] * b[18] + a[8] * b[17] + a[9] * b[16] + a[3] * b[32] + a[6] * b[31];

    c[62] += a[5] * b[28] + a[8] * b[27] + a[9] * b[26] + a[4] * b[32] + a[7] * b[31];

    c[63] += a[5] * b[32] + a[8] * b[31] + a[9] * b[30];

    c[64] += a[9] * b[3] + a[6] * b[9] + a[0] * b[19];

    c[65] += a[9] * b[6] + a[7] * b[9] + a[1] * b[19] + a[6] * b[15] + a[0] * b[29];

    c[66] += a[2] * b[19] + a[9] * b[12] + a[7] * b[15] + a[1] * b[29] + a[6] * b[25];

    c[67] += a[2] * b[29] + a[9] * b[22] + a[7] * b[25];

    c[68] += a[8] * b[9] + a[9] * b[8] + a[3] * b[19] + a[6] * b[18] + a[0] * b[33];

    c[69] += a[4] * b[19]
        + a[8] * b[15]
        + a[9] * b[14]
        + a[7] * b[18]
        + a[3] * b[29]
        + a[1] * b[33]
        + a[6] * b[28];

    c[70] += a[4] * b[29] + a[8] * b[25] + a[9] * b[24] + a[2] * b[33] + a[7] * b[28];

    c[71] += a[5] * b[19] + a[8] * b[18] + a[9] * b[17] + a[3] * b[33] + a[6] * b[32];

    c[72] += a[5] * b[29] + a[8] * b[28] + a[9] * b[27] + a[4] * b[33] + a[7] * b[32];

    c[73] += a[5] * b[33] + a[8] * b[32] + a[9] * b[31];

    c[74] += a[9] * b[9] + a[6] * b[19] + a[0] * b[34];

    c[75] += a[9] * b[15] + a[7] * b[19] + a[1] * b[34] + a[6] * b[29];

    c[76] += a[9] * b[25] + a[2] * b[34] + a[7] * b[29];

    c[77] += a[8] * b[19] + a[9] * b[18] + a[3] * b[34] + a[6] * b[33];

    c[78] += a[8] * b[29] + a[9] * b[28] + a[4] * b[34] + a[7] * b[33];

    c[79] += a[5] * b[34] + a[8] * b[33] + a[9] * b[32];

    c[80] += a[9] * b[19] + a[6] * b[34];

    c[81] += a[9] * b[29] + a[7] * b[34];

    c[82] += a[8] * b[34] + a[9] * b[33];

    c[83] += a[9] * b[34];
}
fn setup_coeff_matrix(
    pp1: &[Vector3<f64>],
    xx1: &[Vector3<f64>],
    pp2: &[Vector3<f64>],
    xx2: &[Vector3<f64>],
) -> SMatrix<f64, 84, 15> {
    let mut f1 = [0.0_f64; 30];
    let mut f2 = [0.0_f64; 30];
    let mut f3 = [0.0_f64; 30];
    let mut matrix = SMatrix::<f64, 84, 15>::zeros();
    let mut qq1 = [Vector3::<f64>::zeros(); 6];
    let mut qq2 = [Vector3::<f64>::zeros(); 6];
    for k in 0..6 {
        qq1[k] = xx1[k].cross(&pp1[k]);
        qq2[k] = xx2[k].cross(&pp2[k]);
    }
    for equation in 0..15 {
        let index0 = PT_INDEX[4 * equation];
        let x1 = xx1[index0];
        let p1 = pp1[index0];
        let x2 = xx2[index0];
        let p2 = pp2[index0];
        for i in 0..3 {
            let index1 = PT_INDEX[4 * equation + i + 1];
            let xp1 = xx1[index1];
            let qp1 = qq1[index1];
            let xp2 = xx2[index1];
            let qp2 = qq2[index1];
            f1[10 * i] = qp1[0] * xp2[0] + qp2[0] * xp1[0]
                - qp1[1] * xp2[1]
                - qp2[1] * xp1[1]
                - qp1[2] * xp2[2]
                - qp2[2] * xp1[2]
                + xp1[0] * (xp2[2] * (p1[1] + p2[1]) - xp2[1] * (p1[2] + p2[2]))
                + xp1[2] * (xp2[0] * (p1[1] + p2[1]) + xp2[1] * (p1[0] - p2[0]))
                - xp1[1] * (xp2[0] * (p1[2] + p2[2]) + xp2[2] * (p1[0] - p2[0]));
            f1[1 + 10 * i] = 2.0 * qp1[0] * xp2[1]
                + 2.0 * qp1[1] * xp2[0]
                + 2.0 * qp2[0] * xp1[1]
                + 2.0 * qp2[1] * xp1[0]
                - xp1[0] * (2.0 * p2[0] * xp2[2] - xp2[0] * (2.0 * p1[2] + 2.0 * p2[2]))
                + xp1[1] * (2.0 * p2[1] * xp2[2] - xp2[1] * (2.0 * p1[2] + 2.0 * p2[2]))
                - xp1[2] * (2.0 * p1[0] * xp2[0] - 2.0 * p1[1] * xp2[1]);
            f1[2 + 10 * i] = qp1[1] * xp2[1] - qp2[0] * xp1[0] - qp1[0] * xp2[0] + qp2[1] * xp1[1]
                - qp1[2] * xp2[2]
                - qp2[2] * xp1[2]
                - xp1[1] * (xp2[2] * (p1[0] + p2[0]) - xp2[0] * (p1[2] + p2[2]))
                - xp1[2] * (xp2[1] * (p1[0] + p2[0]) + xp2[0] * (p1[1] - p2[1]))
                + xp1[0] * (xp2[1] * (p1[2] + p2[2]) + xp2[2] * (p1[1] - p2[1]));
            f1[3 + 10 * i] = 2.0 * qp1[0] * xp2[2]
                + 2.0 * qp1[2] * xp2[0]
                + 2.0 * qp2[0] * xp1[2]
                + 2.0 * qp2[2] * xp1[0]
                + xp1[0] * (2.0 * p2[0] * xp2[1] - xp2[0] * (2.0 * p1[1] + 2.0 * p2[1]))
                - xp1[2] * (2.0 * p2[2] * xp2[1] - xp2[2] * (2.0 * p1[1] + 2.0 * p2[1]))
                + xp1[1] * (2.0 * p1[0] * xp2[0] - 2.0 * p1[2] * xp2[2]);
            f1[4 + 10 * i] = 2.0 * qp1[1] * xp2[2]
                + 2.0 * qp1[2] * xp2[1]
                + 2.0 * qp2[1] * xp1[2]
                + 2.0 * qp2[2] * xp1[1]
                - xp1[1] * (2.0 * p2[1] * xp2[0] - xp2[1] * (2.0 * p1[0] + 2.0 * p2[0]))
                + xp1[2] * (2.0 * p2[2] * xp2[0] - xp2[2] * (2.0 * p1[0] + 2.0 * p2[0]))
                - xp1[0] * (2.0 * p1[1] * xp2[1] - 2.0 * p1[2] * xp2[2]);
            f1[5 + 10 * i] = qp1[2] * xp2[2]
                - qp2[0] * xp1[0]
                - qp1[1] * xp2[1]
                - qp2[1] * xp1[1]
                - qp1[0] * xp2[0]
                + qp2[2] * xp1[2]
                + xp1[2] * (xp2[1] * (p1[0] + p2[0]) - xp2[0] * (p1[1] + p2[1]))
                + xp1[1] * (xp2[2] * (p1[0] + p2[0]) + xp2[0] * (p1[2] - p2[2]))
                - xp1[0] * (xp2[2] * (p1[1] + p2[1]) + xp2[1] * (p1[2] - p2[2]));
            f1[6 + 10 * i] = 2.0 * qp1[1] * xp2[2] - 2.0 * qp1[2] * xp2[1] - 2.0 * qp2[1] * xp1[2]
                + 2.0 * qp2[2] * xp1[1]
                - xp1[1] * (2.0 * p2[1] * xp2[0] + xp2[1] * (2.0 * p1[0] - 2.0 * p2[0]))
                - xp1[2] * (2.0 * p2[2] * xp2[0] + xp2[2] * (2.0 * p1[0] - 2.0 * p2[0]))
                + xp1[0] * (2.0 * p1[1] * xp2[1] + 2.0 * p1[2] * xp2[2]);
            f1[7 + 10 * i] = 2.0 * qp1[2] * xp2[0] - 2.0 * qp1[0] * xp2[2] + 2.0 * qp2[0] * xp1[2]
                - 2.0 * qp2[2] * xp1[0]
                - xp1[0] * (2.0 * p2[0] * xp2[1] + xp2[0] * (2.0 * p1[1] - 2.0 * p2[1]))
                - xp1[2] * (2.0 * p2[2] * xp2[1] + xp2[2] * (2.0 * p1[1] - 2.0 * p2[1]))
                + xp1[1] * (2.0 * p1[0] * xp2[0] + 2.0 * p1[2] * xp2[2]);
            f1[8 + 10 * i] = 2.0 * qp1[0] * xp2[1] - 2.0 * qp1[1] * xp2[0] - 2.0 * qp2[0] * xp1[1]
                + 2.0 * qp2[1] * xp1[0]
                - xp1[0] * (2.0 * p2[0] * xp2[2] + xp2[0] * (2.0 * p1[2] - 2.0 * p2[2]))
                - xp1[1] * (2.0 * p2[1] * xp2[2] + xp2[1] * (2.0 * p1[2] - 2.0 * p2[2]))
                + xp1[2] * (2.0 * p1[0] * xp2[0] + 2.0 * p1[1] * xp2[1]);
            f1[9 + 10 * i] = xp1[1] * (xp2[2] * (p1[0] - p2[0]) - xp2[0] * (p1[2] - p2[2]))
                - xp1[2] * (xp2[1] * (p1[0] - p2[0]) - xp2[0] * (p1[1] - p2[1]))
                - xp1[0] * (xp2[2] * (p1[1] - p2[1]) - xp2[1] * (p1[2] - p2[2]))
                + qp1[0] * xp2[0]
                + qp2[0] * xp1[0]
                + qp1[1] * xp2[1]
                + qp2[1] * xp1[1]
                + qp1[2] * xp2[2]
                + qp2[2] * xp1[2];
            f2[10 * i] = xp1[2] * (x1[0] * xp2[1] + x1[1] * xp2[0])
                - xp1[1] * (x1[0] * xp2[2] + x1[2] * xp2[0])
                + xp1[0] * (x1[1] * xp2[2] - x1[2] * xp2[1]);
            f2[1 + 10 * i] = 2.0 * x1[2] * xp1[0] * xp2[0]
                - xp1[2] * (2.0 * x1[0] * xp2[0] - 2.0 * x1[1] * xp2[1])
                - 2.0 * x1[2] * xp1[1] * xp2[1];
            f2[2 + 10 * i] = xp1[0] * (x1[1] * xp2[2] + x1[2] * xp2[1])
                - xp1[2] * (x1[0] * xp2[1] + x1[1] * xp2[0])
                - xp1[1] * (x1[0] * xp2[2] - x1[2] * xp2[0]);
            f2[3 + 10 * i] = xp1[1] * (2.0 * x1[0] * xp2[0] - 2.0 * x1[2] * xp2[2])
                - 2.0 * x1[1] * xp1[0] * xp2[0]
                + 2.0 * x1[1] * xp1[2] * xp2[2];
            f2[4 + 10 * i] = 2.0 * x1[0] * xp1[1] * xp2[1]
                - xp1[0] * (2.0 * x1[1] * xp2[1] - 2.0 * x1[2] * xp2[2])
                - 2.0 * x1[0] * xp1[2] * xp2[2];
            f2[5 + 10 * i] = xp1[1] * (x1[0] * xp2[2] + x1[2] * xp2[0])
                + xp1[2] * (x1[0] * xp2[1] - x1[1] * xp2[0])
                - xp1[0] * (x1[1] * xp2[2] + x1[2] * xp2[1]);
            f2[6 + 10 * i] = xp1[0] * (2.0 * x1[1] * xp2[1] + 2.0 * x1[2] * xp2[2])
                - 2.0 * x1[0] * xp1[1] * xp2[1]
                - 2.0 * x1[0] * xp1[2] * xp2[2];
            f2[7 + 10 * i] = xp1[1] * (2.0 * x1[0] * xp2[0] + 2.0 * x1[2] * xp2[2])
                - 2.0 * x1[1] * xp1[0] * xp2[0]
                - 2.0 * x1[1] * xp1[2] * xp2[2];
            f2[8 + 10 * i] = xp1[2] * (2.0 * x1[0] * xp2[0] + 2.0 * x1[1] * xp2[1])
                - 2.0 * x1[2] * xp1[0] * xp2[0]
                - 2.0 * x1[2] * xp1[1] * xp2[1];
            f2[9 + 10 * i] = xp1[1] * (x1[0] * xp2[2] - x1[2] * xp2[0])
                - xp1[2] * (x1[0] * xp2[1] - x1[1] * xp2[0])
                - xp1[0] * (x1[1] * xp2[2] - x1[2] * xp2[1]);
            f3[10 * i] = xp1[1] * (x2[0] * xp2[2] - x2[2] * xp2[0])
                - xp1[2] * (x2[0] * xp2[1] - x2[1] * xp2[0])
                + xp1[0] * (x2[1] * xp2[2] - x2[2] * xp2[1]);
            f3[1 + 10 * i] = xp1[1] * (2.0 * x2[1] * xp2[2] - 2.0 * x2[2] * xp2[1])
                - xp1[0] * (2.0 * x2[0] * xp2[2] - 2.0 * x2[2] * xp2[0]);
            f3[2 + 10 * i] = -xp1[2] * (x2[0] * xp2[1] - x2[1] * xp2[0])
                - xp1[1] * (x2[0] * xp2[2] - x2[2] * xp2[0])
                - xp1[0] * (x2[1] * xp2[2] - x2[2] * xp2[1]);
            f3[3 + 10 * i] = xp1[0] * (2.0 * x2[0] * xp2[1] - 2.0 * x2[1] * xp2[0])
                + xp1[2] * (2.0 * x2[1] * xp2[2] - 2.0 * x2[2] * xp2[1]);
            f3[4 + 10 * i] = xp1[1] * (2.0 * x2[0] * xp2[1] - 2.0 * x2[1] * xp2[0])
                - xp1[2] * (2.0 * x2[0] * xp2[2] - 2.0 * x2[2] * xp2[0]);
            f3[5 + 10 * i] = xp1[2] * (x2[0] * xp2[1] - x2[1] * xp2[0])
                + xp1[1] * (x2[0] * xp2[2] - x2[2] * xp2[0])
                - xp1[0] * (x2[1] * xp2[2] - x2[2] * xp2[1]);
            f3[6 + 10 * i] = xp1[1] * (2.0 * x2[0] * xp2[1] - 2.0 * x2[1] * xp2[0])
                + xp1[2] * (2.0 * x2[0] * xp2[2] - 2.0 * x2[2] * xp2[0]);
            f3[7 + 10 * i] = xp1[2] * (2.0 * x2[1] * xp2[2] - 2.0 * x2[2] * xp2[1])
                - xp1[0] * (2.0 * x2[0] * xp2[1] - 2.0 * x2[1] * xp2[0]);
            f3[8 + 10 * i] = -xp1[0] * (2.0 * x2[0] * xp2[2] - 2.0 * x2[2] * xp2[0])
                - xp1[1] * (2.0 * x2[1] * xp2[2] - 2.0 * x2[2] * xp2[1]);
            f3[9 + 10 * i] = xp1[2] * (x2[0] * xp2[1] - x2[1] * xp2[0])
                - xp1[1] * (x2[0] * xp2[2] - x2[2] * xp2[0])
                + xp1[0] * (x2[1] * xp2[2] - x2[2] * xp2[1]);
        }
        let mut p4 = [0.0_f64; 35];
        let mut coeffs = [0.0_f64; 84];
        mul2_2(&f2[10..20], &f3[20..30], &mut p4);
        mul2_2m(&f2[20..30], &f3[10..20], &mut p4);
        mul2_4p(&f1[0..10], &p4, &mut coeffs);
        mul2_2(&f2[20..30], &f3[0..10], &mut p4);
        mul2_2m(&f2[0..10], &f3[20..30], &mut p4);
        mul2_4p(&f1[10..20], &p4, &mut coeffs);
        mul2_2(&f2[0..10], &f3[10..20], &mut p4);
        mul2_2m(&f2[10..20], &f3[0..10], &mut p4);
        mul2_4p(&f1[20..30], &p4, &mut coeffs);
        for monomial in 0..84 {
            matrix[(monomial, equation)] = coeffs[monomial];
        }
    }
    matrix
}
fn solve_gr6p_polynomial(
    correspondences: &[GeneralizedRelativeCorrespondence; 6],
) -> GeneralizedRelativePoseResult {
    let p1: Vec<Vector3<f64>> = correspondences.iter().map(|c| c.rig1.origin).collect();
    let x1: Vec<Vector3<f64>> = correspondences.iter().map(|c| c.rig1.bearing).collect();
    let p2: Vec<Vector3<f64>> = correspondences.iter().map(|c| c.rig2.origin).collect();
    let x2: Vec<Vector3<f64>> = correspondences.iter().map(|c| c.rig2.bearing).collect();
    let matrix = setup_coeff_matrix(&p1, &x1, &p2, &x2);
    let mut c0 = DMatrix::<f64>::zeros(99, 99);
    let mut c1 = DMatrix::<f64>::zeros(99, 64);
    for (i, &index) in COEFFS0_IND.iter().enumerate() {
        let source = matrix[(index % 84, index / 84)];
        let target = C0_IND[i];
        c0[(target % 99, target / 99)] = source;
    }
    for (i, &index) in COEFFS1_IND.iter().enumerate() {
        let source = matrix[(index % 84, index / 84)];
        let target = C1_IND[i];
        c1[(target % 99, target / 99)] = source;
    }
    let Some(c12) = c0.lu().solve(&c1) else {
        return Ok(Vec::new());
    };

    let mut action = SMatrix::<f64, 64, 64>::zeros();
    action[(0, 57)] = 1.0;
    action[(1, 34)] = 1.0;
    action[(2, 19)] = 1.0;
    action[(3, 11)] = 1.0;
    action[(4, 7)] = 1.0;
    set_negative_row(&mut action, 5, &c12, 78);
    set_negative_row(&mut action, 6, &c12, 79);
    set_negative_row(&mut action, 7, &c12, 80);
    action[(8, 10)] = 1.0;
    set_negative_row(&mut action, 9, &c12, 81);
    set_negative_row(&mut action, 10, &c12, 82);
    action[(11, 12)] = 1.0;
    set_negative_row(&mut action, 12, &c12, 83);
    action[(13, 17)] = 1.0;
    action[(14, 16)] = 1.0;
    set_negative_row(&mut action, 15, &c12, 84);
    set_negative_row(&mut action, 16, &c12, 85);
    action[(17, 18)] = 1.0;
    set_negative_row(&mut action, 18, &c12, 86);
    action[(19, 20)] = 1.0;
    action[(20, 21)] = 1.0;
    action[(21, 22)] = 1.0;
    set_negative_row(&mut action, 22, &c12, 87);
    action[(23, 30)] = 1.0;
    action[(24, 28)] = 1.0;
    action[(25, 27)] = 1.0;
    set_negative_row(&mut action, 26, &c12, 88);
    set_negative_row(&mut action, 27, &c12, 89);
    action[(28, 29)] = 1.0;
    set_negative_row(&mut action, 29, &c12, 90);
    action[(30, 31)] = 1.0;
    action[(31, 32)] = 1.0;
    action[(32, 33)] = 1.0;
    set_negative_row(&mut action, 33, &c12, 91);
    action[(34, 35)] = 1.0;
    action[(35, 36)] = 1.0;
    action[(36, 37)] = 1.0;
    action[(37, 38)] = 1.0;
    set_negative_row(&mut action, 38, &c12, 92);
    action[(39, 52)] = 1.0;
    action[(40, 48)] = 1.0;
    action[(41, 45)] = 1.0;
    action[(42, 44)] = 1.0;
    set_negative_row(&mut action, 43, &c12, 93);
    set_negative_row(&mut action, 44, &c12, 94);
    action[(45, 46)] = 1.0;
    action[(46, 47)] = 1.0;
    set_negative_row(&mut action, 47, &c12, 95);
    action[(48, 49)] = 1.0;
    action[(49, 50)] = 1.0;
    action[(50, 51)] = 1.0;
    set_negative_row(&mut action, 51, &c12, 96);
    action[(52, 53)] = 1.0;
    action[(53, 54)] = 1.0;
    action[(54, 55)] = 1.0;
    action[(55, 56)] = 1.0;
    set_negative_row(&mut action, 56, &c12, 97);
    action[(57, 58)] = 1.0;
    action[(58, 59)] = 1.0;
    action[(59, 60)] = 1.0;
    action[(60, 61)] = 1.0;
    action[(61, 62)] = 1.0;
    action[(62, 63)] = 1.0;
    set_negative_row(&mut action, 63, &c12, 98);

    let eigenvalues = action.complex_eigenvalues();
    let real_eigenvalues: Vec<f64> = eigenvalues
        .iter()
        .filter_map(|value| (value.im.abs() < 1.0e-6).then_some(value.re))
        .collect();
    let mut solutions = SMatrix::<f64, 3, 64>::zeros();
    fast_eigenvector_solver(&real_eigenvalues, &action, &mut solutions);
    let mut output = Vec::with_capacity(real_eigenvalues.len());
    for index in 0..real_eigenvalues.len() {
        let w = Vector3::new(
            solutions[(0, index)],
            solutions[(1, index)],
            solutions[(2, index)],
        );
        let quaternion = Quaternion::new(1.0, w[0], w[1], w[2]);
        if !quaternion.coords.iter().all(|value| value.is_finite()) {
            continue;
        }
        let rotation = UnitQuaternion::from_quaternion(quaternion.normalize());
        let mut a = Matrix3::<f64>::zeros();
        let mut b = Vector3::<f64>::zeros();
        for point in 0..6 {
            let u = (rotation * x1[point]).cross(&x2[point]);
            let v = p2[point] - rotation * p1[point];
            a += u * u.transpose();
            b += u * u.dot(&v);
        }
        let Some(translation) = a.lu().solve(&b) else {
            continue;
        };
        let pose = GeneralizedRelativePose {
            rotation,
            translation,
        };
        if p1
            .iter()
            .zip(x1.iter())
            .zip(p2.iter().zip(x2.iter()))
            .all(|((p1, x1), (p2, x2))| check_cheirality(&pose, p1, x1, p2, x2))
        {
            output.push(pose);
        }
    }
    root_refinement(&p1, &x1, &p2, &x2, &mut output);
    output.retain(|pose| {
        pose.translation.iter().all(|value| value.is_finite())
            && pose
                .rotation
                .quaternion()
                .coords
                .iter()
                .all(|value| value.is_finite())
    });
    Ok(output)
}

fn set_negative_row(
    action: &mut SMatrix<f64, 64, 64>,
    row: usize,
    c12: &DMatrix<f64>,
    source_row: usize,
) {
    for column in 0..64 {
        action[(row, column)] = -c12[(source_row, column)];
    }
}

fn copy_action_column(
    destination: &mut SMatrix<f64, 21, 21>,
    source: &SMatrix<f64, 21, 64>,
    destination_column: usize,
    source_column: usize,
) {
    for row in 0..21 {
        destination[(row, destination_column)] = source[(row, source_column)];
    }
}

fn add_action_column(
    destination: &mut SMatrix<f64, 21, 21>,
    source: &SMatrix<f64, 21, 64>,
    destination_column: usize,
    source_column: usize,
    scale: f64,
) {
    for row in 0..21 {
        destination[(row, destination_column)] += scale * source[(row, source_column)];
    }
}

fn fast_eigenvector_solver(
    eigenvalues: &[f64],
    action: &SMatrix<f64, 64, 64>,
    solutions: &mut SMatrix<f64, 3, 64>,
) {
    const IND: [usize; 21] = [
        5, 6, 7, 9, 10, 12, 15, 16, 18, 22, 26, 27, 29, 33, 38, 43, 44, 47, 51, 56, 63,
    ];
    let mut action_small = SMatrix::<f64, 21, 64>::zeros();
    for (row, &source_row) in IND.iter().enumerate() {
        for column in 0..64 {
            action_small[(row, column)] = action[(source_row, column)];
        }
    }
    for (solution_index, &eigenvalue) in eigenvalues.iter().enumerate() {
        let mut z = [0.0_f64; 8];
        z[0] = eigenvalue;
        for power in 1..8 {
            z[power] = z[power - 1] * eigenvalue;
        }
        let mut aa = SMatrix::<f64, 21, 21>::zeros();
        copy_action_column(&mut aa, &action_small, 0, 5);
        copy_action_column(&mut aa, &action_small, 1, 6);
        copy_action_column(&mut aa, &action_small, 2, 4);
        add_action_column(&mut aa, &action_small, 2, 7, z[0]);
        copy_action_column(&mut aa, &action_small, 3, 9);
        copy_action_column(&mut aa, &action_small, 4, 8);
        add_action_column(&mut aa, &action_small, 4, 10, z[0]);
        copy_action_column(&mut aa, &action_small, 5, 3);
        add_action_column(&mut aa, &action_small, 5, 11, z[0]);
        add_action_column(&mut aa, &action_small, 5, 12, z[1]);
        copy_action_column(&mut aa, &action_small, 6, 15);
        copy_action_column(&mut aa, &action_small, 7, 14);
        add_action_column(&mut aa, &action_small, 7, 16, z[0]);
        copy_action_column(&mut aa, &action_small, 8, 13);
        add_action_column(&mut aa, &action_small, 8, 17, z[0]);
        add_action_column(&mut aa, &action_small, 8, 18, z[1]);
        copy_action_column(&mut aa, &action_small, 9, 2);
        add_action_column(&mut aa, &action_small, 9, 19, z[0]);
        add_action_column(&mut aa, &action_small, 9, 20, z[1]);
        add_action_column(&mut aa, &action_small, 9, 21, z[2]);
        add_action_column(&mut aa, &action_small, 9, 22, z[3]);
        copy_action_column(&mut aa, &action_small, 10, 26);
        copy_action_column(&mut aa, &action_small, 11, 25);
        add_action_column(&mut aa, &action_small, 11, 27, z[0]);
        copy_action_column(&mut aa, &action_small, 12, 24);
        add_action_column(&mut aa, &action_small, 12, 28, z[0]);
        add_action_column(&mut aa, &action_small, 12, 29, z[1]);
        copy_action_column(&mut aa, &action_small, 13, 23);
        add_action_column(&mut aa, &action_small, 13, 30, z[0]);
        add_action_column(&mut aa, &action_small, 13, 31, z[1]);
        add_action_column(&mut aa, &action_small, 13, 32, z[2]);
        add_action_column(&mut aa, &action_small, 13, 33, z[3]);
        copy_action_column(&mut aa, &action_small, 14, 1);
        add_action_column(&mut aa, &action_small, 14, 34, z[0]);
        add_action_column(&mut aa, &action_small, 14, 35, z[1]);
        add_action_column(&mut aa, &action_small, 14, 36, z[2]);
        add_action_column(&mut aa, &action_small, 14, 37, z[3]);
        add_action_column(&mut aa, &action_small, 14, 38, z[4]);
        copy_action_column(&mut aa, &action_small, 15, 43);
        copy_action_column(&mut aa, &action_small, 16, 42);
        add_action_column(&mut aa, &action_small, 16, 44, z[0]);
        copy_action_column(&mut aa, &action_small, 17, 41);
        add_action_column(&mut aa, &action_small, 17, 45, z[0]);
        add_action_column(&mut aa, &action_small, 17, 46, z[1]);
        add_action_column(&mut aa, &action_small, 17, 47, z[2]);
        copy_action_column(&mut aa, &action_small, 18, 40);
        add_action_column(&mut aa, &action_small, 18, 48, z[0]);
        add_action_column(&mut aa, &action_small, 18, 49, z[1]);
        add_action_column(&mut aa, &action_small, 18, 50, z[2]);
        add_action_column(&mut aa, &action_small, 18, 51, z[3]);
        copy_action_column(&mut aa, &action_small, 19, 39);
        add_action_column(&mut aa, &action_small, 19, 52, z[0]);
        add_action_column(&mut aa, &action_small, 19, 53, z[1]);
        add_action_column(&mut aa, &action_small, 19, 54, z[2]);
        add_action_column(&mut aa, &action_small, 19, 55, z[3]);
        add_action_column(&mut aa, &action_small, 19, 56, z[4]);
        copy_action_column(&mut aa, &action_small, 20, 0);
        add_action_column(&mut aa, &action_small, 20, 57, z[0]);
        add_action_column(&mut aa, &action_small, 20, 58, z[1]);
        add_action_column(&mut aa, &action_small, 20, 59, z[2]);
        add_action_column(&mut aa, &action_small, 20, 60, z[3]);
        add_action_column(&mut aa, &action_small, 20, 61, z[4]);
        add_action_column(&mut aa, &action_small, 20, 62, z[5]);
        add_action_column(&mut aa, &action_small, 20, 63, z[6]);
        for &(row, power) in &[
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 0),
            (4, 1),
            (5, 2),
            (6, 0),
            (7, 1),
            (8, 2),
            (9, 4),
            (10, 0),
            (11, 1),
            (12, 2),
            (13, 4),
            (14, 5),
            (15, 0),
            (16, 1),
            (17, 3),
            (18, 4),
            (19, 5),
            (20, 7),
        ] {
            aa[(row, row)] -= z[power];
        }
        let lhs = aa.fixed_view::<21, 20>(0, 0).into_owned();
        let rhs = -aa.column(20).into_owned();
        let qr = lhs.qr();
        let mut q_transpose_rhs = rhs;
        qr.q_tr_mul(&mut q_transpose_rhs);
        let upper = qr.r();
        let rhs_upper = q_transpose_rhs.fixed_rows::<20>(0).into_owned();
        let Some(solution) = upper.lu().solve(&rhs_upper) else {
            continue;
        };
        solutions[(0, solution_index)] = solution[14];
        solutions[(1, solution_index)] = solution[19];
        solutions[(2, solution_index)] = z[0];
    }
}
fn check_cheirality(
    pose: &GeneralizedRelativePose,
    p1: &Vector3<f64>,
    x1: &Vector3<f64>,
    p2: &Vector3<f64>,
    x2: &Vector3<f64>,
) -> bool {
    let rx1 = pose.rotation * *x1;
    let rhs = pose.translation + pose.rotation * *p1 - *p2;
    let a = -rx1.dot(x2);
    let b1 = -rx1.dot(&rhs);
    let b2 = x2.dot(&rhs);
    let lambda1 = b1 - a * b2;
    let lambda2 = -a * b1 + b2;
    lambda1.is_finite() && lambda2.is_finite() && lambda1 > 0.0 && lambda2 > 0.0
}

fn root_refinement(
    p1: &[Vector3<f64>],
    x1: &[Vector3<f64>],
    p2: &[Vector3<f64>],
    x2: &[Vector3<f64>],
    poses: &mut [GeneralizedRelativePose],
) {
    for pose in poses {
        for _ in 0..5 {
            let mut jacobian = Matrix6::<f64>::zeros();
            let mut residual = SVector::<f64, 6>::zeros();
            for point in 0..6 {
                let x2t = x2[point].cross(&pose.translation);
                let rx1 = pose.rotation * x1[point];
                let qq1 = x1[point].cross(&p1[point]);
                let qq2 = x2[point].cross(&p2[point]);
                let rqq1 = pose.rotation * qq1;
                residual[point] = (x2t - qq2).dot(&rx1) - x2[point].dot(&rqq1);
                let rotational = -x2t.cross(&rx1) + qq2.cross(&rx1) + x2[point].cross(&rqq1);
                let translational = -x2[point].cross(&rx1);
                for component in 0..3 {
                    jacobian[(point, component)] = rotational[component];
                    jacobian[(point, component + 3)] = translational[component];
                }
            }
            if residual.norm() < 1.0e-12 {
                break;
            }
            let Some(delta) = jacobian.lu().solve(&residual) else {
                break;
            };
            if !delta.iter().all(|value| value.is_finite()) {
                break;
            }
            let w = Vector3::new(-delta[0], -delta[1], -delta[2]);
            pose.rotation = quat_step_pre(pose.rotation, w);
            pose.translation -= Vector3::new(delta[3], delta[4], delta[5]);
        }
    }
}

fn quat_exp(w: Vector3<f64>) -> UnitQuaternion<f64> {
    let theta2 = w.dot(&w);
    let theta = theta2.sqrt();
    let theta_half = 0.5 * theta;
    let (re, im) = if theta > 1.0e-6 {
        (theta_half.cos(), theta_half.sin() / theta)
    } else {
        let theta4 = theta2 * theta2;
        let mut re = 1.0 - 0.125 * theta2 + theta4 / 384.0;
        let mut im = 0.5 - theta2 / 48.0 + theta4 / 3840.0;
        let scale = (re * re + im * im * theta2).sqrt();
        re /= scale;
        im /= scale;
        (re, im)
    };
    UnitQuaternion::from_quaternion(Quaternion::new(re, im * w[0], im * w[1], im * w[2]))
}

fn quat_step_pre(q: UnitQuaternion<f64>, w_delta: Vector3<f64>) -> UnitQuaternion<f64> {
    quat_exp(w_delta) * q
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compact copy of the frozen oracle input from
    // `benchmarks/electro/fixtures/colmap-gr6p-v1.json` (schema
    // `colmap_gr6p_fixture_v1`, SHA-256
    // `6f7d7aaf85ea5762ffcfd672064d282d7b709d77f523c6593dcc9eaa26984da2`).
    // Keeping the constants here makes packaged-crate tests independent of
    // repository-root benchmark files while preserving the oracle provenance.
    #[allow(clippy::excessive_precision)]
    const COLMAP_GR6P_TRUTH_ROTATION_WXYZ: [f64; 4] = [
        0.98801153077423987,
        0.080538613900793504,
        -0.12210693075281595,
        0.049362376261776662,
    ];
    #[allow(clippy::excessive_precision)]
    const COLMAP_GR6P_TRUTH_TRANSLATION: [f64; 3] = [
        0.71999999999999997,
        -0.20999999999999999,
        0.28000000000000003,
    ];
    #[allow(clippy::excessive_precision, clippy::type_complexity)]
    const COLMAP_GR6P_CORRESPONDENCES: [([f64; 3], [f64; 3], [f64; 3], [f64; 3]); 6] = [
        (
            [0.0, 0.0, 0.0],
            [
                0.48955441993989979,
                0.023880703411702432,
                0.8716456745271387,
            ],
            [0.46, 0.08, 0.03],
            [0.2858099465445772, -0.11749905187823649, 0.9510555437322793],
        ),
        (
            [0.46, 0.08, 0.03],
            [
                0.29342727011119996,
                -0.13962338503432564,
                0.94573027206844007,
            ],
            [-0.22, 0.39, 0.11],
            [
                0.21296570288703207,
                -0.30780983156611624,
                0.92730723979977181,
            ],
        ),
        (
            [-0.22, 0.39, 0.11],
            [0.4967503404611251, 0.10754388814106831, 0.86120462805030118],
            [0.0, 0.0, 0.0],
            [
                0.29092172452294407,
                0.0082214741377753756,
                0.95671153309845602,
            ],
        ),
        (
            [0.0, 0.0, 0.0],
            [
                0.28691543455612101,
                0.30693279045538524,
                0.90745346743331301,
            ],
            [-0.22, 0.39, 0.11],
            [
                0.15112281044908871,
                0.086335199195388221,
                0.9847375942563894,
            ],
        ),
        (
            [0.46, 0.08, 0.03],
            [
                0.52651043361942218,
                -0.14570662254663128,
                0.83758960322818099,
            ],
            [0.0, 0.0, 0.0],
            [
                0.40238580735343288,
                -0.23680087193048691,
                0.88431386345204788,
            ],
        ),
        (
            [-0.22, 0.39, 0.11],
            [
                0.22329317261043111,
                -0.29135579741681922,
                0.93018920568730401,
            ],
            [0.46, 0.08, 0.03],
            [
                0.028782348815617344,
                -0.40661370323818369,
                0.91314668741423266,
            ],
        ),
    ];

    fn correspondence() -> GeneralizedRelativeCorrespondence {
        GeneralizedRelativeCorrespondence {
            rig1: GeneralizedCameraObservation {
                origin: Vector3::new(0.1, 0.0, 0.0),
                bearing: Vector3::new(0.0, 0.0, 2.0),
            },
            rig2: GeneralizedCameraObservation {
                origin: Vector3::new(-0.1, 0.0, 0.0),
                bearing: Vector3::new(0.0, 1.0, 1.0),
            },
        }
    }

    #[test]
    fn validates_six_and_normalizes_bearings() {
        let input = [correspondence(); 6];
        let normalized =
            normalize_gr6p_correspondences(&input, &GeneralizedRelativePoseConfig::default())
                .unwrap();
        assert!((normalized[0].rig1.bearing.norm() - 1.0).abs() < 1.0e-12);
        assert!((normalized[0].rig2.bearing.norm() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn rejects_wrong_count_before_solver_gate() {
        let input = [correspondence(); 5];
        assert_eq!(
            estimate_gr6p(&input),
            Err(GeneralizedRelativePoseError::WrongCorrespondenceCount {
                expected: 6,
                actual: 5,
            })
        );
    }

    #[test]
    fn rejects_nonfinite_and_zero_bearings() {
        let mut input = [correspondence(); 6];
        input[2].rig1.origin[1] = f64::NAN;
        assert_eq!(
            normalize_gr6p_correspondences(&input, &Default::default()),
            Err(GeneralizedRelativePoseError::NonFiniteObservation)
        );

        let mut input = [correspondence(); 6];
        input[4].rig2.bearing = Vector3::zeros();
        assert_eq!(
            normalize_gr6p_correspondences(&input, &Default::default()),
            Err(GeneralizedRelativePoseError::ZeroLengthBearing)
        );
    }

    fn fixture_input() -> (
        [GeneralizedRelativeCorrespondence; 6],
        UnitQuaternion<f64>,
        Vector3<f64>,
    ) {
        let truth_rotation = UnitQuaternion::from_quaternion(Quaternion::new(
            COLMAP_GR6P_TRUTH_ROTATION_WXYZ[0],
            COLMAP_GR6P_TRUTH_ROTATION_WXYZ[1],
            COLMAP_GR6P_TRUTH_ROTATION_WXYZ[2],
            COLMAP_GR6P_TRUTH_ROTATION_WXYZ[3],
        ));
        let truth_translation = Vector3::new(
            COLMAP_GR6P_TRUTH_TRANSLATION[0],
            COLMAP_GR6P_TRUTH_TRANSLATION[1],
            COLMAP_GR6P_TRUTH_TRANSLATION[2],
        );

        let mut output = [correspondence(); 6];
        for (item, (origin1, bearing1, origin2, bearing2)) in
            output.iter_mut().zip(COLMAP_GR6P_CORRESPONDENCES)
        {
            item.rig1.origin = Vector3::from(origin1);
            item.rig1.bearing = Vector3::from(bearing1);
            item.rig2.origin = Vector3::from(origin2);
            item.rig2.bearing = Vector3::from(bearing2);
        }
        (output, truth_rotation, truth_translation)
    }

    fn pose_error(
        pose: &GeneralizedRelativePose,
        truth_rotation: UnitQuaternion<f64>,
        truth_translation: Vector3<f64>,
    ) -> f64 {
        let rotation_error = (truth_rotation.inverse() * pose.rotation).angle();
        rotation_error.max((pose.translation - truth_translation).norm())
    }

    #[test]
    fn matches_colmap_gr6p_oracle_fixture() {
        let (input, truth_rotation, truth_translation) = fixture_input();
        let solutions = estimate_gr6p(&input).unwrap();
        assert_eq!(solutions.len(), 8);
        assert!(solutions.iter().all(|pose| {
            pose.translation.iter().all(|value| value.is_finite())
                && pose
                    .rotation
                    .quaternion()
                    .coords
                    .iter()
                    .all(|value| value.is_finite())
        }));
        let best_error = solutions
            .iter()
            .map(|pose| pose_error(pose, truth_rotation, truth_translation))
            .fold(f64::INFINITY, f64::min);
        assert!(best_error <= 1.0e-8, "best GR6P error: {best_error:e}");
    }

    #[test]
    fn gr6p_fixture_is_deterministic_and_swapping_inverts_pose() {
        let (input, truth_rotation, truth_translation) = fixture_input();
        let first = estimate_gr6p(&input).unwrap();
        let second = estimate_gr6p(&input).unwrap();
        assert_eq!(first, second);

        let swapped: Vec<_> = input
            .iter()
            .map(|correspondence| GeneralizedRelativeCorrespondence {
                rig1: correspondence.rig2,
                rig2: correspondence.rig1,
            })
            .collect();
        let swapped_solutions = estimate_gr6p(&swapped).unwrap();
        let inverse_rotation = truth_rotation.inverse();
        let inverse_translation = -(inverse_rotation * truth_translation);
        let best_error = swapped_solutions
            .iter()
            .map(|pose| pose_error(pose, inverse_rotation, inverse_translation))
            .fold(f64::INFINITY, f64::min);
        assert!(
            best_error <= 1.0e-8,
            "best swapped GR6P error: {best_error:e}"
        );
    }

    #[test]
    fn rejects_central_and_degenerate_rigs_before_polynomial_solver() {
        let (mut input, _, _) = fixture_input();
        for correspondence in &mut input {
            correspondence.rig1.origin = Vector3::zeros();
            correspondence.rig2.origin = Vector3::zeros();
        }
        assert!(estimate_gr6p(&input).unwrap().is_empty());

        let mut degenerate = input;
        for (index, correspondence) in degenerate.iter_mut().enumerate() {
            correspondence.rig1.origin = Vector3::new(index as f64, 0.0, 0.0);
            correspondence.rig2.origin = Vector3::new(index as f64, 0.0, 0.0);
            correspondence.rig1.bearing = Vector3::z();
            correspondence.rig2.bearing = Vector3::z();
        }
        assert!(estimate_gr6p(&degenerate).unwrap().is_empty());
    }

    #[test]
    fn accepts_one_sided_central_rig_with_generalized_target() {
        let truth_rotation = UnitQuaternion::from_euler_angles(0.17, -0.11, 0.09);
        let truth_translation = Vector3::new(0.8, -0.3, 0.4);
        let points = [
            Vector3::new(1.2, -0.4, 4.5),
            Vector3::new(-0.8, 0.7, 5.4),
            Vector3::new(1.9, 1.1, 6.2),
            Vector3::new(-1.4, -0.9, 4.1),
            Vector3::new(0.3, 1.8, 7.0),
            Vector3::new(-2.0, 0.2, 6.5),
        ];
        let target_origins = [
            Vector3::zeros(),
            Vector3::new(0.45, 0.05, -0.02),
            Vector3::new(-0.2, 0.35, 0.12),
        ];
        let input: Vec<_> = points
            .iter()
            .enumerate()
            .map(|(index, point_in_rig1)| {
                let origin_in_rig2 = target_origins[index % target_origins.len()];
                let point_in_rig2 = truth_rotation * *point_in_rig1 + truth_translation;
                GeneralizedRelativeCorrespondence {
                    rig1: GeneralizedCameraObservation {
                        origin: Vector3::zeros(),
                        bearing: point_in_rig1.normalize(),
                    },
                    rig2: GeneralizedCameraObservation {
                        origin: origin_in_rig2,
                        bearing: (point_in_rig2 - origin_in_rig2).normalize(),
                    },
                }
            })
            .collect();
        let solutions = estimate_gr6p(&input).unwrap();
        let best_error = solutions
            .iter()
            .map(|pose| pose_error(pose, truth_rotation, truth_translation))
            .fold(f64::INFINITY, f64::min);
        assert!(
            best_error <= 1.0e-8,
            "one-sided central GR6P error: {best_error:e}, candidates: {}",
            solutions.len()
        );
    }
}
