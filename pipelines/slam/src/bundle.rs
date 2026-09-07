//! Bundle adjustment with Schur-complement landmark elimination.
//!
//! Optimizes camera poses jointly with landmark positions to minimize the
//! sum of squared 2D reprojection residuals. Pinhole intrinsics are held
//! fixed; the variables are pose `T_world_to_camera` (6 DoF, right
//! perturbation `T ← T · Exp(ξ)` with `ξ = [ρ; ω]`) per non-fixed pose
//! and `X_w` (3 DoF) per non-fixed landmark.
//!
//! The Schur complement of the block-diagonal landmark Hessian `H_LL`
//! reduces the linear system to one of pose-only size `(6P) × (6P)` per
//! iteration, regardless of how many landmarks the scene has, then
//! back-substitutes for the landmark updates. Each iteration is a
//! Levenberg-Marquardt step with optional cost-rejection.
//!
//! Gauge fixing is the caller's responsibility: monocular BA has 7 DoF
//! gauge freedom (6 SE(3) + 1 scale). At minimum fix the first pose
//! (anchor) and one of the following to remove scale: a second pose, a
//! second landmark, or a known-distance pair. Rectified-stereo BA (any
//! [`BaStereoObservation`] present) has only 6 DoF gauge freedom — the
//! baseline anchors metric scale — so a single fixed pose is enough.
//!
//! # Parallelism
//!
//! [`BaConfig::parallel`] (default `false`, so [`BundleAdjustment::optimize`]
//! and friends are unchanged unless a caller opts in) parallelizes the three
//! per-item hot loops of [`BundleAdjustment::optimize_weighted`]'s
//! Levenberg-Marquardt iteration with `rayon`:
//!
//! - **Assembly** (`build_normal_equations`'s monocular observation loop):
//!   each observation's residual/Jacobian is a pure function of the current
//!   pose and landmark estimate — it touches no shared state — so it is
//!   computed on the rayon pool in fixed-size chunks
//!   ([`PARALLEL_OBSERVATION_CHUNK`]), collected into a plain per-chunk
//!   `Vec`; the actual `+=` scatter into `h_pp` / `b_p` / the per-landmark
//!   blocks stays a single serial pass over each chunk's precomputed
//!   contributions, *in the original per-observation order*.
//! - **Schur reduction** (`solve_step`'s per-landmark `S -= H_PL H_LL⁻¹
//!   H_PLᵀ` loop): each landmark's `3×3` factorization is independent and
//!   computed directly in parallel (disjoint output slots, no merge needed);
//!   the pose-pair contributions it produces are computed the same chunked
//!   way as assembly, collecting each landmark's `(Vec<(p,q,block)>,
//!   Vec<(p,upd)>)` pair per chunk and then flattening/merging into the
//!   shared reduced system `s` / `b_reduced` by a serial pass over each
//!   chunk, in landmark-ascending order. Pure-visual sparse BA instead keeps
//!   this stage serial so parallel observation assembly does not force a
//!   dense camera Hessian or change the block-sparse O(nnz) memory bound.
//! - **Back-substitution** (`solve_step`'s per-landmark `δ_L` loop): each
//!   landmark writes only its own 3 rows of `delta_l`, so this is
//!   embarrassingly parallel with no merge step at all.
//!
//! Unlike [`crate::block_cholesky`]'s intra-column path — which reassociates
//! a floating-point sum across contributors and is therefore only
//! deterministic *to rounding* — every merge here reproduces the exact
//! summation order the serial code would have used, so the parallel path is
//! bit-identical to the serial one at any thread count or chunk size; the
//! chunk constants below exist only to cap peak memory (a full-sequence BA
//! can carry tens of millions of observations, so materializing one
//! contribution per observation up front is not an option) and to amortize
//! the per-dispatch rayon overhead, never to change the result. Each path is
//! also work-gated ([`PARALLEL_MIN_OBSERVATIONS`], [`PARALLEL_MIN_LANDMARKS`])
//! so small problems stay on the plain serial loop even with the flag on,
//! matching `block_cholesky`'s `PARALLEL_MIN_BLOCKS` precedent.
//!
//! Not parallelized: [`BundleAdjustment::optimize_joint_intrinsics`]'s own
//! Schur reduction (a separate, less-used code path — self-calibration BA is
//! opt-in and typically run on far smaller problems than a full-sequence
//! pose/structure solve) and the cost-evaluation passes (`robust_cost_weighted`
//! / `reprojection_squared_residuals`, shared by many callers beyond the LM
//! loop, so gating them on `BaConfig` would require threading the flag
//! through call sites that have nothing to do with this optimizer).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::ops::{Index, IndexMut};

use nalgebra::{
    DMatrix, DVector, Matrix2x3, Matrix2x4, Matrix2x6, Matrix3, Matrix3x4, Matrix3x6, Matrix4x3,
    Matrix4x6, Matrix6, Matrix6x3, Point2, Point3, Vector2, Vector3, Vector4, Vector6,
};

use visloc_core::geometry::{Pose, SE3, SO3};
use visloc_core::types::{Camera, CameraModel, VisualMap};
use visloc_mapping::{
    LocalMapWindow, LocalRefinementReason, LocalRefinementResult, LocalRefiner, StagedMapUpdate,
};

use crate::gnc::{GncConfig, GncState};
use crate::imu_preintegration::ImuPreintegrationFactor;
use crate::process_memory::log as log_process_memory;
use crate::{solve_normal_equations, LinearSolver, PoseGraphError, RobustKernel};

/// Keep solver-step diagnostics opt-in even when a caller already enables a
/// higher-level SFM trace.  This flag is intentionally read here rather than
/// threaded through [`BaConfig`], so the public/default optimizer state stays
/// byte-identical and a diagnostic cannot accidentally become a production
/// behavior switch.
fn ba_step_debug_enabled() -> bool {
    std::env::var_os("VISLOC_SFM_DEBUG_BA").is_some()
        && std::env::var_os("VISLOC_SFM_DEBUG_BA_STEPS").is_some()
}

/// Enable the scalar LM-step quality diagnostic only when all existing BA
/// debug gates and the dedicated quality gate are present.  The diagnostic is
/// deliberately not threaded through `BaConfig`: when this returns false the
/// matrix-free solver does not allocate its residual/backward-error scratch
/// or perform an additional normal-equation scan.
fn ba_lm_step_quality_debug_enabled() -> bool {
    ba_step_debug_enabled() && std::env::var_os("VISLOC_SFM_DEBUG_BA_LM_QUALITY").is_some()
}

/// Context for the opt-in local Schur-block diagnostic.  This is deliberately
/// private and borrowed: the normal solver does not retain a pose/landmark
/// history or any diagnostic records.
#[derive(Debug, Clone, Copy)]
struct SchurBlockDebugContext<'a> {
    iteration: usize,
    pose_slot: usize,
    frame_id: Option<u64>,
    scaled_coordinates: bool,
    landmark_index: &'a BTreeMap<u64, usize>,
}

#[derive(Debug, Clone, Copy, Default)]
struct SchurBlockDebugCounts {
    local_landmarks: usize,
    valid_hll: usize,
    singular_hll: usize,
    cross_entries: usize,
    same_pose_groups: usize,
    same_pose_extra_cross_entries: usize,
    max_elimination_norm: Option<f64>,
    max_elimination_landmark: Option<usize>,
    max_hll_inverse_residual: Option<f64>,
    max_elimination_cross: Option<Matrix6x3<f64>>,
    max_hll: Option<Matrix3<f64>>,
    max_hll_inverse: Option<Matrix3<f64>>,
    nonfinite_elimination: bool,
    nonfinite_hll_inverse_residual: bool,
}

/// The slot is intentionally selected through an explicit diagnostic
/// environment variable.  The existing two debug flags remain a required
/// gate, so setting only the slot cannot perturb normal BA.
fn ba_schur_debug_slot() -> Option<usize> {
    if !ba_step_debug_enabled() {
        return None;
    }
    std::env::var("VISLOC_SFM_DEBUG_BA_SCHUR_SLOT")
        .ok()?
        .parse::<usize>()
        .ok()
}

fn matrix_free_schur_debug_context<'a>(
    pose_index: &'a BTreeMap<u64, usize>,
    landmark_index: &'a BTreeMap<u64, usize>,
    iteration: usize,
    scaled_coordinates: bool,
) -> Option<SchurBlockDebugContext<'a>> {
    let pose_slot = ba_schur_debug_slot()?;
    Some(schur_debug_context_for_slot_with_coordinates(
        pose_index,
        landmark_index,
        iteration,
        pose_slot,
        scaled_coordinates,
    ))
}

#[cfg(test)]
fn schur_debug_context_for_slot<'a>(
    pose_index: &'a BTreeMap<u64, usize>,
    landmark_index: &'a BTreeMap<u64, usize>,
    iteration: usize,
    pose_slot: usize,
) -> SchurBlockDebugContext<'a> {
    schur_debug_context_for_slot_with_coordinates(
        pose_index,
        landmark_index,
        iteration,
        pose_slot,
        false,
    )
}

fn schur_debug_context_for_slot_with_coordinates<'a>(
    pose_index: &'a BTreeMap<u64, usize>,
    landmark_index: &'a BTreeMap<u64, usize>,
    iteration: usize,
    pose_slot: usize,
    scaled_coordinates: bool,
) -> SchurBlockDebugContext<'a> {
    let frame_id = pose_index
        .iter()
        .find_map(|(id, slot)| (*slot == pose_slot).then_some(*id));
    SchurBlockDebugContext {
        iteration,
        pose_slot,
        frame_id,
        scaled_coordinates,
        landmark_index,
    }
}

fn finite_matrix6(matrix: &Matrix6<f64>) -> bool {
    matrix.iter().all(|value| value.is_finite())
}

fn symmetric_from_lower(matrix: &Matrix6<f64>) -> Matrix6<f64> {
    let mut symmetric = *matrix;
    for row in 0..6 {
        for column in (row + 1)..6 {
            symmetric[(row, column)] = matrix[(column, row)];
        }
    }
    symmetric
}

fn symmetric_from_upper(matrix: &Matrix6<f64>) -> Matrix6<f64> {
    let mut symmetric = *matrix;
    for row in 0..6 {
        for column in (row + 1)..6 {
            symmetric[(column, row)] = matrix[(row, column)];
        }
    }
    symmetric
}

fn matrix6_asymmetry(matrix: &Matrix6<f64>) -> f64 {
    let mut maximum = 0.0_f64;
    for row in 0..6 {
        for column in (row + 1)..6 {
            maximum = maximum.max((matrix[(row, column)] - matrix[(column, row)]).abs());
        }
    }
    maximum
}

fn matrix6_min_eigenvalue(matrix: &Matrix6<f64>) -> Option<f64> {
    if !finite_matrix6(matrix) {
        return None;
    }
    // Never let a diagnostic eigensolve run indefinitely on a pathological
    // block.  The production Cholesky result remains the acceptance decision.
    let eigenvalues =
        nalgebra::linalg::SymmetricEigen::try_new(*matrix, f64::EPSILON, 1024)?.eigenvalues;
    let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    minimum.is_finite().then_some(minimum)
}

/// Return the smallest diagonal radicand (before its square root) of a
/// lower-triangle Cholesky factorization. This is a diagnostic-only scalar
/// 6x6 routine; the production factorization remains `Matrix6::cholesky` and
/// is not replaced or symmetrized by this audit.
fn matrix6_min_lower_cholesky_pivot(matrix: &Matrix6<f64>) -> Option<f64> {
    if !finite_matrix6(matrix) {
        return None;
    }
    let mut lower = Matrix6::zeros();
    let mut minimum = f64::INFINITY;
    for row in 0..6 {
        let mut pivot = matrix[(row, row)];
        for column in 0..row {
            pivot -= lower[(row, column)] * lower[(row, column)];
        }
        if !pivot.is_finite() || pivot <= 0.0 {
            return None;
        }
        minimum = minimum.min(pivot);
        lower[(row, row)] = pivot.sqrt();
        for next_row in (row + 1)..6 {
            let mut value = matrix[(next_row, row)];
            for column in 0..row {
                value -= lower[(next_row, column)] * lower[(row, column)];
            }
            lower[(next_row, row)] = value / lower[(row, row)];
        }
    }
    minimum.is_finite().then_some(minimum)
}

fn format_matrix6(matrix: &Matrix6<f64>) -> String {
    let mut output = String::with_capacity(6 * 6 * 20 + 1);
    output.push('[');
    for row in 0..6 {
        if row != 0 {
            output.push(';');
        }
        for column in 0..6 {
            if column != 0 {
                output.push(',');
            }
            write!(&mut output, "{:.17e}", matrix[(row, column)])
                .expect("writing a String cannot fail");
        }
    }
    output.push(']');
    output
}

fn format_matrix6x3(matrix: &Matrix6x3<f64>) -> String {
    let mut output = String::with_capacity(6 * 3 * 20 + 1);
    output.push('[');
    for row in 0..6 {
        if row != 0 {
            output.push(';');
        }
        for column in 0..3 {
            if column != 0 {
                output.push(',');
            }
            write!(&mut output, "{:.17e}", matrix[(row, column)])
                .expect("writing a String cannot fail");
        }
    }
    output.push(']');
    output
}

fn format_matrix3(matrix: &Matrix3<f64>) -> String {
    let mut output = String::with_capacity(3 * 3 * 20 + 1);
    output.push('[');
    for row in 0..3 {
        if row != 0 {
            output.push(';');
        }
        for column in 0..3 {
            if column != 0 {
                output.push(',');
            }
            write!(&mut output, "{:.17e}", matrix[(row, column)])
                .expect("writing a String cannot fail");
        }
    }
    output.push(']');
    output
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SchurBlockDebugMetrics {
    finite: bool,
    asymmetry: f64,
    lower_min_eigenvalue: Option<f64>,
    upper_min_eigenvalue: Option<f64>,
    lower_min_cholesky_radicand: Option<f64>,
    upper_min_cholesky_radicand: Option<f64>,
}

fn inspect_schur_block(matrix: &Matrix6<f64>) -> SchurBlockDebugMetrics {
    let finite = finite_matrix6(matrix);
    let lower = symmetric_from_lower(matrix);
    let upper = symmetric_from_upper(matrix);
    SchurBlockDebugMetrics {
        finite,
        asymmetry: matrix6_asymmetry(matrix),
        lower_min_eigenvalue: matrix6_min_eigenvalue(&lower),
        upper_min_eigenvalue: matrix6_min_eigenvalue(&upper),
        lower_min_cholesky_radicand: matrix6_min_lower_cholesky_pivot(&lower),
        upper_min_cholesky_radicand: matrix6_min_lower_cholesky_pivot(&upper),
    }
}

fn collect_schur_block_debug_counts(
    system: &NormalEquationsBa,
    h_ll_inverse: &[Option<Matrix3<f64>>],
    lambda: f64,
    pose_slot: usize,
) -> SchurBlockDebugCounts {
    let mut counts = SchurBlockDebugCounts::default();
    for (landmark_index, (landmark, inverse)) in
        system.landmarks.iter().zip(h_ll_inverse).enumerate()
    {
        let mut same_pose_cross = Matrix6x3::zeros();
        let mut cross_entries = 0_usize;
        for (pose, cross) in &landmark.cross {
            if *pose == pose_slot {
                cross_entries += 1;
                same_pose_cross += cross;
            }
        }
        if cross_entries == 0 {
            continue;
        }
        counts.local_landmarks += 1;
        counts.cross_entries += cross_entries;
        if cross_entries > 1 {
            counts.same_pose_groups += 1;
            counts.same_pose_extra_cross_entries += cross_entries - 1;
        }
        let Some(inverse) = inverse else {
            counts.singular_hll += 1;
            continue;
        };
        counts.valid_hll += 1;
        let elimination = same_pose_cross * inverse * same_pose_cross.transpose();
        let elimination_norm = elimination.norm();
        if !elimination_norm.is_finite() {
            counts.nonfinite_elimination = true;
            continue;
        }
        let mut h_ll = landmark.h_ll;
        for component in 0..3 {
            h_ll[(component, component)] += lambda;
        }
        let inverse_residual = (h_ll * inverse - Matrix3::identity()).norm();
        if !inverse_residual.is_finite() {
            counts.nonfinite_hll_inverse_residual = true;
        }
        if counts
            .max_elimination_norm
            .is_none_or(|maximum| elimination_norm > maximum)
        {
            counts.max_elimination_norm = Some(elimination_norm);
            counts.max_elimination_landmark = Some(landmark_index);
            counts.max_hll_inverse_residual = Some(inverse_residual);
            counts.max_elimination_cross = Some(same_pose_cross);
            counts.max_hll = Some(h_ll);
            counts.max_hll_inverse = Some(*inverse);
        }
    }
    counts
}

fn emit_schur_block_debug(
    context: SchurBlockDebugContext<'_>,
    lambda: f64,
    h_pp_lambda: &Matrix6<f64>,
    schur_block: &Matrix6<f64>,
    counts: SchurBlockDebugCounts,
) {
    let h_pp_metrics = inspect_schur_block(h_pp_lambda);
    let schur_metrics = inspect_schur_block(schur_block);
    let elimination = *h_pp_lambda - *schur_block;
    let elimination_norm = elimination.norm();
    let finite = h_pp_metrics.finite
        && schur_metrics.finite
        && elimination_norm.is_finite()
        && !counts.nonfinite_elimination
        && !counts.nonfinite_hll_inverse_residual;
    let landmark_id = counts.max_elimination_landmark.and_then(|index| {
        context
            .landmark_index
            .iter()
            .find_map(|(id, mapped)| (*mapped == index).then_some(*id))
    });
    let frame = context
        .frame_id
        .map_or_else(|| "none".to_owned(), |id| id.to_string());
    let max_landmark = counts
        .max_elimination_landmark
        .map_or_else(|| "none".to_owned(), |index| index.to_string());
    let max_landmark_id = landmark_id.map_or_else(|| "none".to_owned(), |id| id.to_string());
    let max_cross = counts
        .max_elimination_cross
        .map_or_else(|| "none".to_owned(), |cross| format_matrix6x3(&cross));
    let max_hll = counts
        .max_hll
        .map_or_else(|| "none".to_owned(), |hll| format_matrix3(&hll));
    let max_hll_inverse = counts
        .max_hll_inverse
        .map_or_else(|| "none".to_owned(), |inverse| format_matrix3(&inverse));
    // Keep the legacy diagnostic byte-for-byte stable.  The coordinate and
    // damping labels are meaningful only for the opt-in scaled system; an
    // empty annotation leaves the old `finite=... solver_...` token sequence
    // unchanged.
    let coordinate_annotation = if context.scaled_coordinates {
        " coordinate_system=scaled damping_metric=identity"
    } else {
        ""
    };
    eprintln!(
        concat!(
            "sfm-debug-ba-schur-block iteration={} slot={} frame_id={} lambda={:.17e} ",
            "finite={}{} solver_symmetric_interpretation=lower_triangle ",
            "hpp_lambda={} schur_block={} elimination_norm={:.17e} ",
            "schur_asymmetry={:.17e} lower_min_eigen={:?} upper_min_eigen={:?} ",
            "lower_min_cholesky_radicand={:?} upper_min_cholesky_radicand={:?} ",
            "local_landmarks={} valid_hll={} singular_hll={} cross_entries={} ",
            "same_pose_groups={} same_pose_extra_cross_entries={} ",
            "max_elimination_norm={:?} max_elimination_landmark={} ",
            "max_elimination_point_id={} max_hll_inverse_residual={:?} ",
            "max_elimination_cross={} max_hll={} max_hll_inverse={}"
        ),
        context.iteration,
        context.pose_slot,
        frame,
        lambda,
        finite,
        coordinate_annotation,
        format_matrix6(h_pp_lambda),
        format_matrix6(schur_block),
        elimination_norm,
        schur_metrics.asymmetry,
        schur_metrics.lower_min_eigenvalue,
        schur_metrics.upper_min_eigenvalue,
        schur_metrics.lower_min_cholesky_radicand,
        schur_metrics.upper_min_cholesky_radicand,
        counts.local_landmarks,
        counts.valid_hll,
        counts.singular_hll,
        counts.cross_entries,
        counts.same_pose_groups,
        counts.same_pose_extra_cross_entries,
        counts.max_elimination_norm,
        max_landmark,
        max_landmark_id,
        counts.max_hll_inverse_residual,
        max_cross,
        max_hll,
        max_hll_inverse,
    );
}

#[cfg(test)]
mod schur_block_debug_tests {
    use super::*;

    #[test]
    fn local_block_metrics_distinguish_spd_non_spd_nonfinite_and_asymmetry() {
        let spd = Matrix6::from_diagonal(&Vector6::from_element(2.0));
        let spd_metrics = inspect_schur_block(&spd);
        assert!(spd_metrics.finite);
        assert_eq!(spd_metrics.asymmetry, 0.0);
        assert_eq!(spd_metrics.lower_min_eigenvalue, Some(2.0));
        assert_eq!(spd_metrics.upper_min_eigenvalue, Some(2.0));
        assert_eq!(spd_metrics.lower_min_cholesky_radicand, Some(2.0));
        assert_eq!(spd_metrics.upper_min_cholesky_radicand, Some(2.0));

        let mut non_spd = spd;
        non_spd[(0, 0)] = -1.0;
        let non_spd_metrics = inspect_schur_block(&non_spd);
        assert!(non_spd_metrics.finite);
        assert!(non_spd_metrics.lower_min_eigenvalue.unwrap() < 0.0);
        assert_eq!(non_spd_metrics.lower_min_cholesky_radicand, None);

        let mut nonfinite = spd;
        nonfinite[(0, 0)] = f64::NAN;
        let nonfinite_metrics = inspect_schur_block(&nonfinite);
        assert!(!nonfinite_metrics.finite);
        assert_eq!(nonfinite_metrics.lower_min_eigenvalue, None);
        assert_eq!(nonfinite_metrics.upper_min_cholesky_radicand, None);

        let mut asymmetric = spd;
        asymmetric[(0, 1)] = 3.0;
        asymmetric[(1, 0)] = 2.0;
        let asymmetric_metrics = inspect_schur_block(&asymmetric);
        assert_eq!(asymmetric_metrics.asymmetry, 1.0);
        assert!(asymmetric_metrics.lower_min_eigenvalue.is_some());
        assert!(asymmetric_metrics.upper_min_eigenvalue.is_some());
        assert_ne!(
            asymmetric_metrics.lower_min_eigenvalue,
            asymmetric_metrics.upper_min_eigenvalue
        );
    }

    #[test]
    fn debug_context_maps_variable_slot_after_fixed_pose_and_counts_rig_crosses() {
        let pose_index = BTreeMap::from([(10_u64, 0_usize), (20_u64, 1_usize)]);
        let landmark_index = BTreeMap::from([(42_u64, 0_usize)]);
        // Use the pure mapping helper.  The environment-gated wrapper is not
        // exercised here so tests remain independent under the parallel test
        // runner.
        let context = schur_debug_context_for_slot(&pose_index, &landmark_index, 7, 1);
        assert_eq!(context.iteration, 7);
        assert_eq!(context.pose_slot, 1);
        assert_eq!(context.frame_id, Some(20));

        let cross = Matrix6x3::from_fn(|row, column| if row == column { 1.0 } else { 0.0 });
        let system = NormalEquationsBa {
            h_pp: CameraHessian::PoseDiagonal(vec![Matrix6::identity(); 2]),
            b_p: DVector::zeros(12),
            landmarks: vec![LandmarkBlock {
                h_ll: Matrix3::identity(),
                b_l: Vector3::zeros(),
                cross: vec![(1, cross), (1, cross)],
            }],
        };
        let h_ll_inverse = (system.landmarks[0].h_ll + 0.25 * Matrix3::<f64>::identity())
            .try_inverse()
            .unwrap();
        let counts = collect_schur_block_debug_counts(&system, &[Some(h_ll_inverse)], 0.25, 1);
        assert_eq!(counts.local_landmarks, 1);
        assert_eq!(counts.valid_hll, 1);
        assert_eq!(counts.singular_hll, 0);
        assert_eq!(counts.cross_entries, 2);
        assert_eq!(counts.same_pose_groups, 1);
        assert_eq!(counts.same_pose_extra_cross_entries, 1);
        assert_eq!(counts.max_elimination_landmark, Some(0));
        assert!(counts.max_hll_inverse_residual.unwrap() < 1.0e-12);
        assert_eq!(context.landmark_index.get(&42), Some(&0));
    }
}

#[cfg(test)]
mod generalized_rig_factor_tests {
    use nalgebra::{Point3, UnitQuaternion, Vector3};

    use super::*;

    #[test]
    fn arbitrary_sensor_factors_refine_one_shared_body_pose() {
        let camera_left = Camera::pinhole(1, 848, 800, 285.0, 286.0, 425.5, 398.5);
        let camera_right = Camera::pinhole(2, 848, 800, 284.8, 286.1, 428.0, 397.5);
        let sensors = [
            (camera_left.clone(), SE3::identity()),
            (
                camera_right,
                SE3::new(UnitQuaternion::identity(), Vector3::new(-0.20, 0.0, 0.0)),
            ),
        ];
        let truth = [
            Pose::identity(),
            Pose::from_world_to_camera(
                UnitQuaternion::from_euler_angles(0.01, -0.04, 0.02),
                Vector3::new(-0.35, 0.03, 0.01),
            ),
        ];
        let mut problem = BundleAdjustment::new(camera_left);
        problem.add_pose(0, truth[0].clone());
        problem.add_pose(
            1,
            Pose::from_world_to_camera(
                UnitQuaternion::from_euler_angles(0.03, -0.01, -0.01),
                Vector3::new(-0.27, -0.02, 0.04),
            ),
        );
        problem.fix_pose(0);
        for landmark in 0..24u64 {
            let point = Point3::new(
                (landmark % 6) as f64 * 0.25 - 0.6,
                (landmark / 6) as f64 * 0.22 - 0.3,
                4.0 + (landmark % 5) as f64 * 0.15,
            );
            problem.add_landmark(
                landmark,
                Point3::from(point.coords + Vector3::new(0.02, -0.01, 0.03)),
            );
            for (frame, frame_pose) in truth.iter().enumerate() {
                for (camera, sensor_from_rig) in &sensors {
                    let point_rig = frame_pose.transform_world_point(&point);
                    let pixel = camera
                        .project(&sensor_from_rig.transform_point(&point_rig))
                        .unwrap();
                    problem.add_rig_observation(BaRigObservation {
                        keyframe_id: frame as u64,
                        landmark_id: landmark,
                        xy: pixel,
                        camera: camera.clone(),
                        sensor_from_rig: sensor_from_rig.clone(),
                    });
                }
            }
        }
        let initial_cost = problem.cost();
        let initial_center_error =
            (problem.poses[&1].camera_center_world() - truth[1].camera_center_world()).norm();
        let result = problem
            .optimize(&BaConfig {
                max_iterations: 30,
                ..BaConfig::default()
            })
            .unwrap();
        let final_center_error =
            (problem.poses[&1].camera_center_world() - truth[1].camera_center_world()).norm();
        assert!(result.final_cost < initial_cost * 1.0e-6);
        assert!(final_center_error < initial_center_error * 1.0e-3);
    }
}

/// Convert optional normalized matcher confidences into relative BA
/// information weights without changing the visual factor group's mean scale.
///
/// Learned match probabilities are not calibrated inverse variances. Feeding
/// them directly into tight VI-BA would weaken the entire visual block against
/// the physically whitened IMU block. Explicit scores are therefore divided
/// by their own finite mean; observations without a score stay at `1`.
/// Returns `None` when no valid confidence signal is present.
pub(crate) fn relative_observation_confidence_weights(
    confidences: impl IntoIterator<Item = Option<f32>>,
) -> Option<Vec<f64>> {
    let confidences = confidences
        .into_iter()
        .map(|confidence| {
            confidence.filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
        })
        .collect::<Vec<_>>();
    let explicit = confidences.iter().flatten().copied().collect::<Vec<_>>();
    if explicit.is_empty() {
        return None;
    }
    let mean = explicit.iter().map(|value| *value as f64).sum::<f64>() / explicit.len() as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return None;
    }
    Some(
        confidences
            .into_iter()
            .map(|confidence| confidence.map_or(1.0, |value| value as f64 / mean))
            .collect(),
    )
}

/// FEJ-style dense Gaussian prior over one or more navigation states.
///
/// Each keyframe contributes `[pose(6), velocity(3), bias(6)]` in that order.
/// `information`, `gradient`, and `constant_cost` describe the quadratic at
/// `reference`: `c + 2 g^T dx + dx^T H dx`. Pose deltas use the same right
/// perturbation as BA, `T = T_ref Exp(dx)`. Keeping the reference fixed avoids
/// silently changing the linearisation point as a fixed-lag window slides.
#[derive(Debug, Clone, PartialEq)]
pub struct NavigationStatePrior {
    pub keyframe_ids: Vec<u64>,
    pub reference_poses: BTreeMap<u64, Pose>,
    pub reference_velocities: BTreeMap<u64, Vector3<f64>>,
    pub reference_biases: BTreeMap<u64, Vector6<f64>>,
    pub information: DMatrix<f64>,
    pub gradient: DVector<f64>,
    pub constant_cost: f64,
}

impl NavigationStatePrior {
    pub fn is_well_formed(&self) -> bool {
        let dim = self.keyframe_ids.len() * 15;
        self.information.nrows() == dim
            && self.information.ncols() == dim
            && self.gradient.len() == dim
            && self.constant_cost.is_finite()
            && self.information.iter().all(|value| value.is_finite())
            && self.gradient.iter().all(|value| value.is_finite())
            && self.keyframe_ids.iter().all(|id| {
                self.reference_poses.contains_key(id)
                    && self.reference_velocities.contains_key(id)
                    && self.reference_biases.contains_key(id)
            })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NavigationLinearization {
    pub pose_ids: Vec<u64>,
    pub velocity_ids: Vec<u64>,
    pub bias_ids: Vec<u64>,
    pub information: DMatrix<f64>,
    pub gradient: DVector<f64>,
}

/// One 2D image-point measurement linking a keyframe to a landmark.
#[derive(Debug, Clone, PartialEq)]
pub struct BaObservation {
    pub keyframe_id: u64,
    pub landmark_id: u64,
    /// Pixel coordinates `(u, v)` in the keyframe's image.
    pub xy: Point2<f64>,
}

/// One rectified-stereo measurement linking a keyframe to a landmark. The
/// keyframe's pose is the LEFT camera's `T_world_to_camera`. The right camera
/// is assumed rectified: shared intrinsics, optical axes parallel, and image
/// rows aligned, so the right pixel only needs its horizontal coordinate
/// (`v_r = v_l`). The shared baseline lives on [`BundleAdjustment`].
///
/// Compared with two independent [`BaObservation`]s for the left and right
/// pixel, a single [`BaStereoObservation`] (i) avoids carrying a separate
/// right-camera pose (it is implicitly the left's translated by `b·x̂`) and
/// (ii) couples the two residuals through the same landmark variable, which
/// is the standard rectified-stereo BA formulation.
#[derive(Debug, Clone, PartialEq)]
pub struct BaStereoObservation {
    pub keyframe_id: u64,
    pub landmark_id: u64,
    /// Left-image pixel coordinates `(u_l, v_l)`.
    pub xy: Point2<f64>,
    /// Right-image horizontal pixel coordinate `u_r`. The vertical coordinate
    /// `v_r` is taken to equal `xy.y` (rectified-stereo assumption).
    pub u_right: f64,
}

/// One calibrated, non-rectified stereo observation. The keyframe pose is the
/// left camera's `T_left<-world`; `left_to_right` is the fixed rig transform
/// `T_right<-left`. Unlike [`BaStereoObservation`], both right-image
/// coordinates and the right camera intrinsics are retained, so rigs with a
/// rotational cam0/cam1 extrinsic (including EuRoC) contribute their true
/// four-dimensional reprojection residual.
#[derive(Debug, Clone, PartialEq)]
pub struct BaGeneralStereoObservation {
    pub keyframe_id: u64,
    pub landmark_id: u64,
    pub xy_left: Point2<f64>,
    pub xy_right: Point2<f64>,
    pub right_camera: Camera,
    pub left_to_right: SE3,
}

/// One observation from an arbitrary sensor rigidly attached to a shared rig
/// frame. `poses[keyframe_id]` stores `T_rig<-world`; the fixed extrinsic is
/// `T_sensor<-rig`. This is the non-central counterpart of
/// [`BaObservation`] and allows every sensor pixel to constrain one body pose
/// without requiring a same-landmark observation in a designated left camera.
#[derive(Debug, Clone, PartialEq)]
pub struct BaRigObservation {
    pub keyframe_id: u64,
    pub landmark_id: u64,
    pub xy: Point2<f64>,
    pub camera: Camera,
    pub sensor_from_rig: SE3,
}

/// Rotation-alignment gravity prior on every non-fixed pose.
///
/// Adds a 3-vector residual `r = R_wc · g_world − g_camera_observed` per
/// pose, where `R_wc` is the pose's world-to-camera rotation and the two
/// gravity vectors are caller-supplied. The most common use is a level
/// prior: set both vectors to the same down-direction (e.g.
/// `(0, 9.81, 0)` for a KITTI-style y-down camera that starts level)
/// and the optimiser will resist pitch / roll drift that re-projection
/// residuals cannot disambiguate on coplanar-feature scenes.
///
/// This prior constrains ROTATION only. Pure-translation drift (such as
/// the structural vertical bias on KITTI sequence 08, where the camera
/// rotation already matches ground truth) is NOT corrected by this
/// prior — that would require a translation/altitude prior fed from
/// IMU velocity or GNSS, which lives outside [`BundleAdjustment`] in
/// its current form.
#[derive(Debug, Clone, PartialEq)]
pub struct GravityPrior {
    /// Gravity direction in world frame. Magnitude defines the
    /// residual's natural scale; using the physical 9.81 m/s² keeps the
    /// per-pose residual in the same order of magnitude as a pixel
    /// reprojection residual, so a default Huber `delta ≈ 3` does not
    /// over- or under-weight the prior.
    pub g_world: Vector3<f64>,
    /// Gravity direction observed (or assumed) in camera frame for
    /// every pose. For a level prior this matches the camera-frame
    /// direction of `g_world` at the anchor pose, e.g. `(0, 9.81, 0)`.
    pub g_camera_observed: Vector3<f64>,
    /// Scalar weight applied to the gravity contribution. The cost
    /// added per pose is `weight · ‖r‖²` and the normal-equations
    /// contribution is `weight · Jᵀ J` / `weight · Jᵀ r`. A weight of
    /// `1.0` makes a 9.81 m/s² gravity residual count comparably to a
    /// single 9.81 px reprojection residual; lower this for a softer
    /// prior, raise it for a stiffer one.
    pub weight: f64,
}

/// Per-keyframe observation of the gravity direction in camera
/// coordinates. Each entry constrains
/// `R_wc · g_world ≈ g_camera_observed` at the named keyframe; the
/// residual and Jacobian shape are identical to [`GravityPrior`]'s
/// global pose-independent variant, except the observation is sourced
/// per-keyframe rather than shared across all poses.
///
/// The intended source of `g_camera_observed` is an accelerometer
/// sample (or a low-pass-filtered window of samples) at the keyframe
/// timestamp, rotated into the camera frame via the body→camera
/// extrinsic. Unlike [`PositionPrior`], which can leak ground-truth
/// poses when fed from GNSS/INS-fused trajectories, a properly-
/// generated per-keyframe gravity prior is a true online sensor
/// observation — the same signal a deployed VIO would consume.
#[derive(Debug, Clone, PartialEq)]
pub struct PerPoseGravityObservation {
    /// Keyframe whose pose rotation is being constrained. The pose
    /// must already be added to [`BundleAdjustment`]. Fixed poses
    /// still contribute to the cost report but generate no Jacobian
    /// rows because they have no Hessian slot.
    pub keyframe_id: u64,
    /// Observed gravity direction in camera frame at this keyframe.
    /// Magnitude should match [`PerPoseGravityPrior::g_world`] (e.g.
    /// `9.81 m/s²` for a physical accelerometer-derived observation),
    /// so the per-pose residual stays in the same order of magnitude
    /// as a pixel reprojection residual.
    pub g_camera_observed: Vector3<f64>,
    /// Per-observation stiffness multiplier applied on top of the
    /// global [`PerPoseGravityPrior::weight`]. `1.0` is neutral;
    /// raise to up-weight a high-confidence sample, lower to soften a
    /// motion-contaminated one. Setting to `0.0` mutes the
    /// observation entirely (useful for keeping all keyframe slots
    /// while gating obviously bad samples).
    pub weight: f64,
}

impl PerPoseGravityObservation {
    /// Build an observation with the default neutral per-obs weight
    /// (`1.0`). Use the public `weight` field directly when emitting
    /// per-sample stiffness from a sensor model.
    pub fn new(keyframe_id: u64, g_camera_observed: Vector3<f64>) -> Self {
        Self {
            keyframe_id,
            g_camera_observed,
            weight: 1.0,
        }
    }
}

/// Per-keyframe gravity-alignment prior. Each
/// [`PerPoseGravityObservation`] adds a rotation-domain residual at
/// its keyframe; the prior as a whole shares a single world-frame
/// gravity vector and stiffness.
///
/// This is the online-friendly companion to [`GravityPrior`] (single
/// observation shared across all poses) — it accepts per-keyframe
/// observations rather than baking in a single "level-world" assumption.
/// Use it when the body's pitch/roll varies meaningfully along the
/// trajectory (climbing/descending on a slope, banking on a curve,
/// etc.) so the gravity-in-camera-frame direction is no longer
/// constant.
///
/// Like [`GravityPrior`], the prior constrains ROTATION only. Pure-
/// translation drift (such as the structural vertical bias on KITTI
/// sequence 08, where the camera rotation already matches ground
/// truth) is NOT corrected by this prior.
#[derive(Debug, Clone, PartialEq)]
pub struct PerPoseGravityPrior {
    /// Per-keyframe observations. May contain at most one entry per
    /// `keyframe_id`; duplicates are accepted but each contributes
    /// independently (the optimiser does not deduplicate).
    pub observations: Vec<PerPoseGravityObservation>,
    /// Gravity direction in world frame, shared across all
    /// observations. Magnitude defines the residual's natural scale.
    pub g_world: Vector3<f64>,
    /// Global scalar weight applied to every observation's
    /// contribution, multiplied with each observation's
    /// [`PerPoseGravityObservation::weight`]. `weight = 1.0` plus
    /// per-obs `1.0` makes a 9.81 m/s² gravity residual count
    /// comparably to a single 9.81 px reprojection residual; lower
    /// the global scale for a softer prior, raise for a stiffer one.
    /// The per-observation field stays neutral unless the upstream
    /// sensor model emits inverse-variance weights.
    pub weight: f64,
}

impl PerPoseGravityPrior {
    pub fn new(g_world: Vector3<f64>, weight: f64) -> Self {
        Self {
            observations: Vec::new(),
            g_world,
            weight,
        }
    }

    pub fn push(&mut self, observation: PerPoseGravityObservation) {
        self.observations.push(observation);
    }
}

/// One absolute position measurement for a single keyframe. The
/// expected world-frame camera centre is compared against the BA's
/// current estimate of `−Rᵀ · t`. Designed for translation-domain
/// priors fed from GNSS, an external altimeter, or — in evaluation
/// scenarios — ground-truth poses; the prior constrains TRANSLATION
/// only, complementing [`GravityPrior`] which constrains ROTATION
/// only.
///
/// `axis_weights` enables per-axis stiffness: a per-pose altitude
/// constraint sets `axis_weights = (0, w, 0)` so only the vertical
/// component is anchored (the most common shape for fixing seq08-style
/// vertical drift without claiming horizontal GNSS accuracy).
#[derive(Debug, Clone, PartialEq)]
pub struct PositionPriorObservation {
    /// Keyframe whose world camera centre is being constrained. The
    /// pose must already be added to [`BundleAdjustment`]. Fixed poses
    /// still contribute to the cost (for diagnostics) but generate no
    /// Jacobian rows because they have no Hessian slot.
    pub keyframe_id: u64,
    /// Expected world-frame camera centre. For a level KITTI-style
    /// y-down camera this is the same coordinate frame as
    /// `pose.camera_center_world()`.
    pub camera_center_world: Point3<f64>,
    /// Per-axis weights in the cost `Σ wᵢ · (Cᵢ − targetᵢ)²` and the
    /// normal-equations contribution. A zero entry removes that axis
    /// from the prior entirely; mixed positive entries pin a subset of
    /// axes with different stiffnesses (`(0, w, 0)` for altitude-only).
    pub axis_weights: Vector3<f64>,
}

/// A relative-pose constraint between two BA keyframes, e.g. an IMU
/// pre-integration delta, a wheel-odometry tick, or an external
/// pose-graph edge being lifted into BA.
///
/// At convergence the measurement equals the BA-implied relative pose
/// `T_j · T_iⁱ` (`world_to_camera_j` of the "to" keyframe composed with
/// the inverse of the "from" keyframe). The residual is the SE(3) log
/// of the disagreement:
///
/// ```text
/// r = log(measurement⁻¹ · T_j · T_iⁱ)  ∈ ℝ⁶
/// ```
///
/// Jacobians under right-perturbation `T ← T · exp(δ)`:
///
/// - `∂r / ∂δ_j =  Ad(T_i)`
/// - `∂r / ∂δ_i = −Ad(T_i)`
///
/// This is the same Jacobian shape used by
/// `PoseGraph::optimize_se3_iterative`; the factor lifts those edges
/// into [`BundleAdjustment`] so visual residuals and external-sensor
/// pose deltas can be jointly optimised in a single LM solve. Full
/// IMU pre-integration with velocity/bias states is a future
/// extension; this v1 factor assumes the pre-integrator has already
/// produced a single `(Δp, ΔR)` pair plus a scalar weight.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwisePoseFactor {
    /// "From" keyframe id (the one Ad(T_from) is computed about).
    pub keyframe_id_from: u64,
    /// "To" keyframe id.
    pub keyframe_id_to: u64,
    /// Measured relative pose `T_meas` such that, at convergence,
    /// `T_meas = T_j · T_iⁱ` where `T_i` and `T_j` are the BA poses
    /// for `keyframe_id_from` and `keyframe_id_to` respectively.
    pub measurement: Pose,
    /// Scalar weight (sqrt-information squared). The cost added is
    /// `weight · ‖r‖²` so `weight = 1 / σ²` for an isotropic
    /// measurement with standard deviation `σ` (per-axis). Anisotropic
    /// 6×6 sqrt-information matrices are deferred to a future
    /// extension.
    pub weight: f64,
}

/// Bias random-walk factor between two keyframes' 6-vector IMU
/// biases. Adds the residual `r = b_j − b_i` with independent gyro and
/// accelerometer weights. Use this to keep neighbouring
/// keyframes' biases close to each other when the IMU factor's data-
/// driven Jacobian leaves some bias DoFs unobservable in isolation
/// (e.g., gyro biases on a straight-line trajectory).
///
/// Both endpoint biases must be registered via
/// [`BundleAdjustment::add_bias`] for the factor to contribute. If
/// either side has a non-fixed bias slot, the factor adds its 6×6
/// Jacobian (`J_i = −I`, `J_j = I`) to the normal equations; fully-
/// fixed endpoints still contribute to the cost report but no
/// Jacobian rows.
#[derive(Debug, Clone, PartialEq)]
pub struct BiasRandomWalkFactor {
    /// "From" keyframe id (the bias on the `−I` side of the Jacobian).
    pub keyframe_id_from: u64,
    /// "To" keyframe id (the bias on the `+I` side).
    pub keyframe_id_to: u64,
    /// Gyroscope-bias sqrt-information squared. A typical value is
    /// `1 / (σ_bg² · Δt_ij)` for continuous random-walk density `σ_bg`.
    pub weight_gyro: f64,
    /// Accelerometer-bias sqrt-information squared. A typical value is
    /// `1 / (σ_ba² · Δt_ij)` for continuous random-walk density `σ_ba`.
    pub weight_accel: f64,
}

/// A bundle of per-keyframe absolute position constraints.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionPrior {
    pub observations: Vec<PositionPriorObservation>,
    /// When true, the camera-centre residual uses the full Jacobian
    /// `[-I | [C_w]_x]`, so pose rotation updates can also move the
    /// constrained centre. When false, the residual uses `[-I | 0]`
    /// and acts as a translation-only centre prior. Keep this true for
    /// the historical BA semantics; turn it off for sensor height/grade
    /// priors that should not pull rotation away from visual evidence.
    pub couple_rotation: bool,
}

impl PositionPrior {
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            couple_rotation: true,
        }
    }

    pub fn with_rotation_coupling(mut self, couple_rotation: bool) -> Self {
        self.couple_rotation = couple_rotation;
        self
    }

    pub fn push(&mut self, observation: PositionPriorObservation) {
        self.observations.push(observation);
    }
}

impl Default for PositionPrior {
    fn default() -> Self {
        Self::new()
    }
}

/// Bundle-adjustment problem: poses, landmarks, observations, plus a single
/// shared pinhole camera (multi-camera support is left as a future extension).
#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustment {
    pub poses: BTreeMap<u64, Pose>,
    pub landmarks: BTreeMap<u64, Point3<f64>>,
    pub observations: Vec<BaObservation>,
    /// Rectified-stereo observations sharing [`Self::stereo_baseline`]. They
    /// reference the same `poses` / `landmarks` collections as
    /// [`Self::observations`], so a single landmark can have both monocular
    /// and stereo evidence.
    pub stereo_observations: Vec<BaStereoObservation>,
    /// Calibrated non-rectified stereo observations. These use [`Self::camera`]
    /// as the left camera and carry their right camera/extrinsic explicitly.
    pub general_stereo_observations: Vec<BaGeneralStereoObservation>,
    /// Arbitrary calibrated rig-sensor observations sharing the frame poses.
    pub rig_observations: Vec<BaRigObservation>,
    pub camera: Camera,
    /// Pose ids whose `Pose` is held constant during optimization.
    pub fixed_poses: BTreeSet<u64>,
    /// Pose ids whose rotation is held constant while their translation may
    /// still be optimized.  The six-dimensional pose slot is retained for
    /// the Schur system, but the rotation rows/columns are constrained to
    /// zero by [`Self::optimize`].  An empty set preserves the historical
    /// pose/structure solve exactly.
    pub fixed_pose_rotations: BTreeSet<u64>,
    /// Landmark ids whose `Point3` is held constant during optimization.
    pub fixed_landmarks: BTreeSet<u64>,
    /// Rectified-stereo baseline in metric units. The right camera is at
    /// `+stereo_baseline · x̂` of the left in the left-camera frame. Required
    /// (positive, finite) when [`Self::stereo_observations`] is non-empty;
    /// ignored otherwise. `None` means "monocular BA".
    pub stereo_baseline: Option<f64>,
    /// Optional rotation-alignment gravity prior. When `Some`, every
    /// non-fixed pose contributes a 3-vector gravity-alignment residual
    /// (see [`GravityPrior`]). Fixed poses are still included in the
    /// cost report but do not generate Jacobian rows.
    pub gravity_prior: Option<GravityPrior>,
    /// Optional per-keyframe gravity-alignment prior. Like
    /// [`Self::gravity_prior`] but the `g_camera_observed` varies per
    /// observation, so the prior accepts e.g. accelerometer-derived
    /// per-keyframe observations rather than a single shared level-
    /// world assumption. See [`PerPoseGravityPrior`].
    pub per_pose_gravity_prior: Option<PerPoseGravityPrior>,
    /// Optional per-keyframe absolute position prior. Each observation
    /// adds an axis-weighted residual `(C_w − target)` with Jacobian
    /// `[−I | [C_w]_×]` (right perturbation, xi-order `[ρ; ω]`). See
    /// [`PositionPrior`].
    pub position_prior: Option<PositionPrior>,
    /// Pairwise relative-pose factors. Each factor lifts an external
    /// relative-pose measurement (IMU pre-integration, wheel odometry,
    /// loop-closure verification, etc.) into the BA solve. See
    /// [`PairwisePoseFactor`].
    pub pairwise_pose_factors: Vec<PairwisePoseFactor>,
    /// Per-keyframe world-frame velocity state. Populated for the
    /// keyframes that participate in any [`ImuPreintegrationFactor`]; the
    /// optimiser jointly refines pose + velocity. Keyframes without an
    /// IMU factor referencing them can leave their velocity slot empty
    /// (the reprojection / pairwise pose / prior factors don't read it).
    pub velocities: BTreeMap<u64, Vector3<f64>>,
    /// Velocity ids held constant during optimisation (mirrors
    /// [`Self::fixed_poses`] / [`Self::fixed_landmarks`]).
    pub fixed_velocities: BTreeSet<u64>,
    /// On-manifold IMU pre-integration factors. Each factor carries a
    /// gravity-compensated `(ΔR, Δv, Δp)` produced by
    /// [`crate::imu_preintegration::ImuPreintegrator`] and binds two
    /// keyframes' `(pose, velocity)` states with a 9-vector residual
    /// `[r_R; r_v; r_p]` (Forster 2017 eq. 45-47). The optimiser
    /// linearises the rotation residual via the SO(3) right-Jacobian
    /// inverse.
    pub imu_factors: Vec<ImuPreintegrationFactor>,
    /// Rigid transform from the tracked camera/sensor frame into the IMU
    /// body frame (`T_b<-c`, EuRoC `T_BS`). Visual residuals continue to use
    /// the stored camera poses; IMU residuals compose this extrinsic to obtain
    /// body poses. Identity preserves the historical co-located rig behavior.
    pub imu_body_to_camera: SE3,
    /// Per-keyframe IMU bias state, packing `(bias_gyro, bias_acc)` as a
    /// 6-vector. Populated for the keyframes whose
    /// [`ImuPreintegrationFactor`] should be bias-corrected (the
    /// integration window from `i` to `j` uses `bias[i]` for its
    /// first-order correction). Keyframes without an IMU factor
    /// referencing them, or whose bias has not been registered, fall
    /// back to using the integrator's linearisation bias (no
    /// correction).
    pub biases: BTreeMap<u64, Vector6<f64>>,
    /// Bias ids held constant during optimisation (mirrors
    /// [`Self::fixed_poses`] / [`Self::fixed_velocities`]).
    pub fixed_biases: BTreeSet<u64>,
    /// Bias random-walk priors between consecutive keyframes. See
    /// [`BiasRandomWalkFactor`]; the cost contribution is
    /// `weight · ‖b_j − b_i‖²` and the Jacobian places `±I` against
    /// each non-fixed bias slot.
    pub bias_random_walk_factors: Vec<BiasRandomWalkFactor>,
    /// Dense fixed-lag prior carried from the preceding VI window.
    pub navigation_state_prior: Option<NavigationStatePrior>,
}

impl BundleAdjustment {
    pub fn new(camera: Camera) -> Self {
        Self {
            poses: BTreeMap::new(),
            landmarks: BTreeMap::new(),
            observations: Vec::new(),
            stereo_observations: Vec::new(),
            general_stereo_observations: Vec::new(),
            rig_observations: Vec::new(),
            camera,
            fixed_poses: BTreeSet::new(),
            fixed_pose_rotations: BTreeSet::new(),
            fixed_landmarks: BTreeSet::new(),
            stereo_baseline: None,
            gravity_prior: None,
            per_pose_gravity_prior: None,
            position_prior: None,
            pairwise_pose_factors: Vec::new(),
            velocities: BTreeMap::new(),
            fixed_velocities: BTreeSet::new(),
            imu_factors: Vec::new(),
            imu_body_to_camera: SE3::identity(),
            biases: BTreeMap::new(),
            fixed_biases: BTreeSet::new(),
            bias_random_walk_factors: Vec::new(),
            navigation_state_prior: None,
        }
    }

    pub fn set_navigation_state_prior(&mut self, prior: NavigationStatePrior) {
        self.navigation_state_prior = Some(prior);
    }

    /// Linearize the navigation-only portion at the current estimate.
    /// Used by the fixed-lag VI stage to Schur-marginalize a state leaving
    /// the window. Landmark-bearing problems are deliberately rejected here:
    /// the boundary prior owns only inertial-chain information, preventing
    /// retained visual observations from being counted both in the prior and
    /// again in the next window.
    pub(crate) fn linearized_navigation_system(&self) -> Option<NavigationLinearization> {
        if !self.landmarks.is_empty()
            || !self.observations.is_empty()
            || !self.stereo_observations.is_empty()
            || !self.general_stereo_observations.is_empty()
            || !self.rig_observations.is_empty()
        {
            return None;
        }
        let intrinsics = self.intrinsics()?;
        let mut pose_index = BTreeMap::new();
        for &id in self.poses.keys() {
            if !self.fixed_poses.contains(&id) {
                let next = pose_index.len();
                pose_index.insert(id, next);
            }
        }
        let mut velocity_index = BTreeMap::new();
        for &id in self.velocities.keys() {
            if !self.fixed_velocities.contains(&id) {
                let next = velocity_index.len();
                velocity_index.insert(id, next);
            }
        }
        let mut bias_index = BTreeMap::new();
        for &id in self.biases.keys() {
            if !self.fixed_biases.contains(&id) {
                let next = bias_index.len();
                bias_index.insert(id, next);
            }
        }
        let system = build_normal_equations(
            self,
            &intrinsics,
            &pose_index,
            &BTreeMap::new(),
            &velocity_index,
            &bias_index,
            &RobustKernel::None,
            None,
            false,
            false,
        );
        let ordered_ids = |index: &BTreeMap<u64, usize>| {
            let mut ids: Vec<(usize, u64)> = index.iter().map(|(id, slot)| (*slot, *id)).collect();
            ids.sort_unstable();
            ids.into_iter().map(|(_, id)| id).collect()
        };
        Some(NavigationLinearization {
            pose_ids: ordered_ids(&pose_index),
            velocity_ids: ordered_ids(&velocity_index),
            bias_ids: ordered_ids(&bias_index),
            information: system.h_pp.into_dense(),
            gradient: system.b_p,
        })
    }

    /// Append a relative-pose factor between two keyframes. See
    /// [`PairwisePoseFactor`] for semantics.
    pub fn add_pairwise_pose_factor(&mut self, factor: PairwisePoseFactor) {
        self.pairwise_pose_factors.push(factor);
    }

    /// Register an initial world-frame velocity for the given keyframe.
    /// Required for any keyframe referenced by an
    /// [`ImuPreintegrationFactor`] — the velocity becomes a BA variable
    /// (unless also passed to [`Self::fix_velocity`]).
    pub fn add_velocity(&mut self, id: u64, velocity: Vector3<f64>) {
        self.velocities.insert(id, velocity);
    }

    /// Pin the velocity of `id` so it does not change during
    /// [`Self::optimize`]. Useful for anchoring the initial keyframe's
    /// velocity to a measured value when the IMU factor would otherwise
    /// leave it under-constrained.
    pub fn fix_velocity(&mut self, id: u64) {
        self.fixed_velocities.insert(id);
    }

    /// Append an on-manifold IMU pre-integration factor between two
    /// keyframes. Both keyframes must have a [`Self::add_velocity`]
    /// entry; otherwise the factor is silently skipped during build
    /// (the rest of the BA still runs).
    pub fn add_imu_factor(&mut self, factor: ImuPreintegrationFactor) {
        self.imu_factors.push(factor);
    }

    /// Set the calibrated camera/sensor-to-body transform used only by IMU
    /// residuals. Reprojection residuals always retain the camera pose state.
    pub fn set_imu_body_to_camera(&mut self, body_to_camera: SE3) {
        self.imu_body_to_camera = body_to_camera;
    }

    /// Register an initial IMU bias state for the given keyframe,
    /// packing the gyro bias in the first 3 components and the
    /// accelerometer bias in the last 3. The bias becomes a BA
    /// variable (unless also passed to [`Self::fix_bias`]). Required
    /// for any keyframe that should provide the bias-correction term
    /// for its outgoing [`ImuPreintegrationFactor`]; keyframes without
    /// a registered bias use the factor's linearisation bias
    /// (no correction) and do not contribute a bias Jacobian column.
    pub fn add_bias(&mut self, id: u64, bias: Vector6<f64>) {
        self.biases.insert(id, bias);
    }

    /// Pin the bias of `id` so it does not change during
    /// [`Self::optimize`]. The bias is still used for the residual's
    /// first-order correction (so the integration's linearisation
    /// point and the BA-side bias estimate stay decoupled), but no
    /// bias Jacobian column is added to the normal equations.
    pub fn fix_bias(&mut self, id: u64) {
        self.fixed_biases.insert(id);
    }

    /// Append a bias random-walk prior. Both endpoints should have
    /// been registered via [`Self::add_bias`]; the factor pulls
    /// `bias[keyframe_id_to]` toward `bias[keyframe_id_from]` with
    /// a weight of `weight`. See [`BiasRandomWalkFactor`].
    pub fn add_bias_random_walk_factor(&mut self, factor: BiasRandomWalkFactor) {
        self.bias_random_walk_factors.push(factor);
    }

    /// Install (or replace) the gravity prior used by [`Self::optimize`]
    /// and [`Self::robust_cost`]. See [`GravityPrior`] for semantics.
    pub fn set_gravity_prior(&mut self, prior: GravityPrior) {
        self.gravity_prior = Some(prior);
    }

    /// Install (or replace) the per-keyframe gravity prior. See
    /// [`PerPoseGravityPrior`] for semantics; this is the online-
    /// friendly companion to [`Self::set_gravity_prior`] that accepts
    /// per-keyframe `g_camera_observed` observations.
    pub fn set_per_pose_gravity_prior(&mut self, prior: PerPoseGravityPrior) {
        self.per_pose_gravity_prior = Some(prior);
    }

    /// Install (or replace) the absolute position prior. See
    /// [`PositionPrior`] for semantics.
    pub fn set_position_prior(&mut self, prior: PositionPrior) {
        self.position_prior = Some(prior);
    }

    pub fn add_pose(&mut self, id: u64, pose: Pose) {
        self.poses.insert(id, pose);
    }

    pub fn fix_pose(&mut self, id: u64) {
        self.fixed_poses.insert(id);
    }

    /// Pin only the rotation of `id` during bundle adjustment.  Translation
    /// remains a variable, which is useful for diagnostic decompositions that
    /// separate rotational from translational error.  Calling this for a
    /// fully fixed pose is harmless.
    pub fn fix_pose_rotation(&mut self, id: u64) {
        self.fixed_pose_rotations.insert(id);
    }

    pub fn add_landmark(&mut self, id: u64, xyz: Point3<f64>) {
        self.landmarks.insert(id, xyz);
    }

    pub fn fix_landmark(&mut self, id: u64) {
        self.fixed_landmarks.insert(id);
    }

    pub fn add_observation(&mut self, obs: BaObservation) {
        self.observations.push(obs);
    }

    /// Append a rectified-stereo observation. Caller must call
    /// [`Self::set_stereo_baseline`] (with the same baseline used to
    /// triangulate `landmark_id`) before [`Self::optimize`], or the optimizer
    /// returns [`BaError::MissingStereoBaseline`].
    pub fn add_stereo_observation(&mut self, obs: BaStereoObservation) {
        self.stereo_observations.push(obs);
    }

    pub fn add_general_stereo_observation(&mut self, obs: BaGeneralStereoObservation) {
        self.general_stereo_observations.push(obs);
    }

    pub fn add_rig_observation(&mut self, observation: BaRigObservation) {
        self.rig_observations.push(observation);
    }

    /// Set the rectified-stereo baseline (positive, metric, in the units of
    /// the landmark coordinates). Required when any
    /// [`Self::stereo_observations`] are present.
    pub fn set_stereo_baseline(&mut self, baseline: f64) {
        self.stereo_baseline = Some(baseline);
    }

    /// Sum of squared reprojection residuals `Σ ||π(K · T · X_w) − u||²`.
    /// Observations whose camera-frame point falls behind the camera are
    /// skipped (cost is reported as if those observations are absent).
    /// Equivalent to [`Self::robust_cost`] called with [`RobustKernel::None`].
    pub fn cost(&self) -> f64 {
        self.robust_cost(&RobustKernel::None)
    }

    /// Number of visual observations that cannot currently be projected.
    ///
    /// Reprojection cost and Jacobian assembly intentionally skip points on
    /// or behind the camera plane.  Without a separate feasibility check an
    /// LM step can therefore lower its reported cost merely by moving hard
    /// observations behind a camera.  Candidate steps are only accepted when
    /// this count does not increase.
    fn nonprojectable_observation_count(&self) -> usize {
        let mono = self.observations.iter().filter(|obs| {
            let (Some(pose), Some(point)) = (
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
            ) else {
                return true;
            };
            let xc = pose.transform_world_point(point);
            xc.z <= 0.0 || self.camera.project(&xc).is_none()
        });

        let intrinsics = self.intrinsics();
        let baseline_valid = self
            .stereo_baseline
            .is_some_and(|baseline| baseline.is_finite() && baseline > 0.0);
        let stereo = self.stereo_observations.iter().filter(|obs| {
            let (Some(intrinsics), true, Some(pose), Some(point)) = (
                intrinsics,
                baseline_valid,
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
            ) else {
                return true;
            };
            let xc = pose.transform_world_point(point);
            xc.z <= 0.0 || project_pinhole(&intrinsics, &xc).is_none()
        });

        let general_stereo = self.general_stereo_observations.iter().filter(|obs| {
            let (Some(pose), Some(point), Some(left_intrinsics)) = (
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
                self.intrinsics(),
            ) else {
                return true;
            };
            general_stereo_residual_jacobians(&left_intrinsics, obs, pose, point).is_none()
        });

        let rig = self.rig_observations.iter().filter(|observation| {
            let (Some(pose), Some(point)) = (
                self.poses.get(&observation.keyframe_id),
                self.landmarks.get(&observation.landmark_id),
            ) else {
                return true;
            };
            rig_residual_jacobians(observation, pose, point).is_none()
        });

        mono.count() + stereo.count() + general_stereo.count() + rig.count()
    }

    /// Robust reprojection cost: `Σ ρ(||r||²)` where `ρ` is the supplied
    /// [`RobustKernel`]. With [`RobustKernel::None`] this matches
    /// [`Self::cost`]. Stereo observations contribute a 3-vector residual
    /// `(u_l_pred − u_l_meas, v_l_pred − v_l_meas, u_r_pred − u_r_meas)`
    /// where `u_r_pred = u_l_pred − fx · b / Z` (rectified-stereo assumption,
    /// see [`BaStereoObservation`]).
    pub fn robust_cost(&self, kernel: &RobustKernel) -> f64 {
        self.robust_cost_weighted(kernel, None)
    }

    /// Additive cost decomposition at the current linearisation point.
    ///
    /// `imu_normalized_squared_residual_per_dof` is the mean whitened
    /// squared IMU residual (NIS / 9 DoF per preintegration factor). It is
    /// useful for detecting a visual/inertial consistency failure that a
    /// single aggregate BA cost would otherwise hide.
    pub fn cost_breakdown(&self, kernel: &RobustKernel) -> BaCostBreakdown {
        self.cost_breakdown_weighted(kernel, None)
    }

    /// Confidence-weighted counterpart of [`Self::cost_breakdown`]. Visual
    /// terms use the same flattened weight vector as
    /// [`Self::optimize_with_observation_weights`]; IMU, navigation and other
    /// structural terms retain their physical information matrices.
    pub fn cost_breakdown_with_observation_weights(
        &self,
        kernel: &RobustKernel,
        observation_weights: &[f64],
    ) -> Result<BaCostBreakdown, BaError> {
        self.validate_observation_weights(observation_weights)?;
        Ok(self.cost_breakdown_weighted(kernel, Some(observation_weights)))
    }

    fn cost_breakdown_weighted(
        &self,
        kernel: &RobustKernel,
        observation_weights: Option<&[f64]>,
    ) -> BaCostBreakdown {
        let total = self.robust_cost_weighted(kernel, observation_weights);

        let mut visual_problem = self.clone();
        visual_problem.gravity_prior = None;
        visual_problem.per_pose_gravity_prior = None;
        visual_problem.position_prior = None;
        visual_problem.pairwise_pose_factors.clear();
        visual_problem.bias_random_walk_factors.clear();
        visual_problem.imu_factors.clear();
        visual_problem.navigation_state_prior = None;
        let visual = visual_problem.robust_cost_weighted(kernel, observation_weights);

        let mut imu_problem = self.clone();
        clear_visual_and_structural_costs(&mut imu_problem);
        imu_problem.bias_random_walk_factors.clear();
        imu_problem.navigation_state_prior = None;
        let imu = imu_problem.robust_cost(kernel);

        let mut bias_problem = self.clone();
        clear_visual_and_structural_costs(&mut bias_problem);
        bias_problem.imu_factors.clear();
        bias_problem.navigation_state_prior = None;
        let bias_random_walk = bias_problem.robust_cost(kernel);

        let mut navigation_problem = self.clone();
        clear_visual_and_structural_costs(&mut navigation_problem);
        navigation_problem.imu_factors.clear();
        navigation_problem.bias_random_walk_factors.clear();
        let navigation_prior = navigation_problem.robust_cost(kernel);

        let other_structural = total - visual - imu - bias_random_walk - navigation_prior;
        let imu_normalized_squared_residual_per_dof =
            (!self.imu_factors.is_empty()).then_some(imu / (9.0 * self.imu_factors.len() as f64));
        let (
            imu_rotation_residual_rms_rad,
            imu_velocity_residual_rms_mps,
            imu_position_residual_rms_meters,
        ) = self
            .imu_raw_residual_rms()
            .map_or((None, None, None), |(rotation, velocity, position)| {
                (Some(rotation), Some(velocity), Some(position))
            });
        BaCostBreakdown {
            total,
            visual,
            imu,
            bias_random_walk,
            navigation_prior,
            other_structural,
            imu_normalized_squared_residual_per_dof,
            imu_rotation_residual_rms_rad,
            imu_velocity_residual_rms_mps,
            imu_position_residual_rms_meters,
        }
    }

    /// Unwhitened per-axis RMS of the rotation, velocity, and position IMU
    /// residual blocks at the current state, in physical units.
    pub fn imu_raw_residual_rms(&self) -> Option<(f64, f64, f64)> {
        let mut rotation_squared = 0.0;
        let mut velocity_squared = 0.0;
        let mut position_squared = 0.0;
        let mut evaluated = 0usize;
        for factor in &self.imu_factors {
            let (Some(pose_i), Some(pose_j), Some(v_i), Some(v_j)) = (
                self.poses.get(&factor.keyframe_id_from),
                self.poses.get(&factor.keyframe_id_to),
                self.velocities.get(&factor.keyframe_id_from),
                self.velocities.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let body_i = self.imu_body_to_camera.compose(&pose_i.world_to_camera);
            let body_j = self.imu_body_to_camera.compose(&pose_j.world_to_camera);
            let r_i = SO3::from_quaternion(body_i.rotation.inverse());
            let r_j = SO3::from_quaternion(body_j.rotation.inverse());
            let p_i: Vector3<f64> = body_i.inverse().translation;
            let p_j: Vector3<f64> = body_j.inverse().translation;
            let [r_rotation, r_velocity, r_position] =
                if let Some(bias) = self.biases.get(&factor.keyframe_id_from) {
                    let bias_gyro: Vector3<f64> = bias.fixed_rows::<3>(0).into_owned();
                    let bias_accel: Vector3<f64> = bias.fixed_rows::<3>(3).into_owned();
                    factor.residual_with_bias_correction(
                        &r_i,
                        &p_i,
                        v_i,
                        &r_j,
                        &p_j,
                        v_j,
                        &bias_gyro,
                        &bias_accel,
                    )
                } else {
                    factor.residual(&r_i, &p_i, v_i, &r_j, &p_j, v_j)
                };
            rotation_squared += r_rotation.norm_squared();
            velocity_squared += r_velocity.norm_squared();
            position_squared += r_position.norm_squared();
            evaluated += 1;
        }
        (evaluated > 0).then(|| {
            let denominator = 3.0 * evaluated as f64;
            (
                (rotation_squared / denominator).sqrt(),
                (velocity_squared / denominator).sqrt(),
                (position_squared / denominator).sqrt(),
            )
        })
    }

    /// Like [`Self::robust_cost`] but multiplies each reprojection
    /// observation's contribution by an external per-observation weight
    /// (the Graduated Non-Convexity Black-Rangarajan weight `w ∈ [0,1]`).
    /// `gnc_weights` is indexed monocular-observations-first
    /// (`0 .. observations.len()`), rectified stereo, then general stereo
    /// (`observations.len() .. + stereo_observations.len()`). `None`
    /// reproduces [`Self::robust_cost`] exactly. Structural and inertial
    /// terms (gravity / position priors, pairwise pose, bias random-walk,
    /// IMU) are never reweighted — only outlier-prone feature
    /// reprojections are, so a wrong correspondence is the only thing GNC
    /// can switch off.
    fn robust_cost_weighted(&self, kernel: &RobustKernel, gnc_weights: Option<&[f64]>) -> f64 {
        let intrinsics = match self.intrinsics() {
            Some(k) => k,
            None => return 0.0,
        };
        let mut total = 0.0;
        for (obs_idx, obs) in self.observations.iter().enumerate() {
            let (Some(pose), Some(point)) = (
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
            ) else {
                continue;
            };
            let xc = pose.transform_world_point(point);
            if xc.z <= 0.0 {
                continue;
            }
            // Distortion-aware projection (identical to `project_pinhole` when the
            // camera carries no distortion, so all existing callers are unchanged).
            if let Some(predicted) = self.camera.project(&xc) {
                let r = predicted - obs.xy;
                let s = r.x * r.x + r.y * r.y;
                let w = gnc_weights.map_or(1.0, |gw| gw[obs_idx]);
                total += w * kernel.cost(s);
            }
        }
        if let Some(baseline) = self.stereo_baseline {
            if baseline.is_finite() && baseline > 0.0 {
                let (fx, _fy, _cx, _cy) = intrinsics;
                let stereo_offset = self.observations.len();
                for (st_idx, obs) in self.stereo_observations.iter().enumerate() {
                    let (Some(pose), Some(point)) = (
                        self.poses.get(&obs.keyframe_id),
                        self.landmarks.get(&obs.landmark_id),
                    ) else {
                        continue;
                    };
                    let xc = pose.transform_world_point(point);
                    if xc.z <= 0.0 {
                        continue;
                    }
                    if let Some(predicted) = project_pinhole(&intrinsics, &xc) {
                        let u_r_pred = predicted.x - fx * baseline / xc.z;
                        let dx = predicted.x - obs.xy.x;
                        let dy = predicted.y - obs.xy.y;
                        let dr = u_r_pred - obs.u_right;
                        let s = dx * dx + dy * dy + dr * dr;
                        let w = gnc_weights.map_or(1.0, |gw| gw[stereo_offset + st_idx]);
                        total += w * kernel.cost(s);
                    }
                }
            }
        }
        let general_offset = self.observations.len() + self.stereo_observations.len();
        for (index, obs) in self.general_stereo_observations.iter().enumerate() {
            let (Some(pose), Some(point)) = (
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
            ) else {
                continue;
            };
            let Some((residual, _, _)) =
                general_stereo_residual_jacobians(&intrinsics, obs, pose, point)
            else {
                continue;
            };
            let w = gnc_weights.map_or(1.0, |weights| weights[general_offset + index]);
            total += w * kernel.cost(residual.norm_squared());
        }
        let rig_offset = general_offset + self.general_stereo_observations.len();
        for (index, observation) in self.rig_observations.iter().enumerate() {
            let (Some(pose), Some(point)) = (
                self.poses.get(&observation.keyframe_id),
                self.landmarks.get(&observation.landmark_id),
            ) else {
                continue;
            };
            let Some((residual, _, _)) = rig_residual_jacobians(observation, pose, point) else {
                continue;
            };
            let weight = gnc_weights.map_or(1.0, |weights| weights[rig_offset + index]);
            total += weight * kernel.cost(residual.norm_squared());
        }
        if let Some(prior) = &self.gravity_prior {
            for pose in self.poses.values() {
                let r_mat = pose
                    .world_to_camera
                    .rotation
                    .to_rotation_matrix()
                    .into_inner();
                let r_vec: Vector3<f64> = r_mat * prior.g_world - prior.g_camera_observed;
                let s = r_vec.norm_squared();
                // Gravity prior uses an L2 (non-robust) contribution: the
                // measurement is global per pose, not a per-feature
                // outlier-prone observation, so a robust kernel here would
                // hide rather than down-weight prior–data conflicts.
                total += prior.weight * s;
            }
        }
        if let Some(prior) = &self.per_pose_gravity_prior {
            for obs in &prior.observations {
                let Some(pose) = self.poses.get(&obs.keyframe_id) else {
                    continue;
                };
                let r_mat = pose
                    .world_to_camera
                    .rotation
                    .to_rotation_matrix()
                    .into_inner();
                let r_vec: Vector3<f64> = r_mat * prior.g_world - obs.g_camera_observed;
                let s = r_vec.norm_squared();
                // Same L2 (non-robust) reasoning as [`Self::gravity_prior`].
                total += prior.weight * obs.weight * s;
            }
        }
        if let Some(prior) = &self.position_prior {
            for obs in &prior.observations {
                let Some(pose) = self.poses.get(&obs.keyframe_id) else {
                    continue;
                };
                let c_world = pose.camera_center_world();
                let r_vec = c_world - obs.camera_center_world;
                // Axis-weighted L2 cost: Σ wᵢ · rᵢ². Zero weight axes
                // contribute nothing, so an altitude-only prior with
                // `axis_weights = (0, w, 0)` is exact.
                let s = obs.axis_weights.x * r_vec.x * r_vec.x
                    + obs.axis_weights.y * r_vec.y * r_vec.y
                    + obs.axis_weights.z * r_vec.z * r_vec.z;
                total += s;
            }
        }
        for factor in &self.pairwise_pose_factors {
            let (Some(from), Some(to)) = (
                self.poses.get(&factor.keyframe_id_from),
                self.poses.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let predicted = to.world_to_camera.compose(&from.world_to_camera.inverse());
            let r = factor
                .measurement
                .world_to_camera
                .inverse()
                .compose(&predicted)
                .log();
            total += factor.weight * r.norm_squared();
        }
        for factor in &self.bias_random_walk_factors {
            let (Some(b_i), Some(b_j)) = (
                self.biases.get(&factor.keyframe_id_from),
                self.biases.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let r: Vector6<f64> = b_j - b_i;
            total += factor.weight_gyro.max(0.0) * r.fixed_rows::<3>(0).norm_squared()
                + factor.weight_accel.max(0.0) * r.fixed_rows::<3>(3).norm_squared();
        }
        // IMU pre-integration factors: 9-vector residual [r_R; r_v; r_p]
        // weighted axis-wise. The factor's `residual` helper takes the
        // body-to-world rotation and world-frame body centre obtained by
        // composing the camera pose with the calibrated T_b<-c extrinsic.
        // When `self.biases` carries a bias for the factor's "from"
        // keyframe, the bias-corrected residual is used (Forster eq. 44).
        for factor in &self.imu_factors {
            let (Some(pose_i), Some(pose_j)) = (
                self.poses.get(&factor.keyframe_id_from),
                self.poses.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let (Some(v_i), Some(v_j)) = (
                self.velocities.get(&factor.keyframe_id_from),
                self.velocities.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let body_i = self.imu_body_to_camera.compose(&pose_i.world_to_camera);
            let body_j = self.imu_body_to_camera.compose(&pose_j.world_to_camera);
            let r_i = SO3::from_quaternion(body_i.rotation.inverse());
            let r_j = SO3::from_quaternion(body_j.rotation.inverse());
            let p_i: Vector3<f64> = body_i.inverse().translation;
            let p_j: Vector3<f64> = body_j.inverse().translation;
            let [r_rot, r_vel, r_pos] =
                if let Some(bias) = self.biases.get(&factor.keyframe_id_from) {
                    let bg: Vector3<f64> = bias.fixed_rows::<3>(0).into_owned();
                    let ba: Vector3<f64> = bias.fixed_rows::<3>(3).into_owned();
                    factor.residual_with_bias_correction(&r_i, &p_i, v_i, &r_j, &p_j, v_j, &bg, &ba)
                } else {
                    factor.residual(&r_i, &p_i, v_i, &r_j, &p_j, v_j)
                };
            if let Some(whitener) = factor.covariance_sqrt_information() {
                let mut residual = nalgebra::SVector::<f64, 9>::zeros();
                residual.fixed_rows_mut::<3>(0).copy_from(&r_rot);
                residual.fixed_rows_mut::<3>(3).copy_from(&r_vel);
                residual.fixed_rows_mut::<3>(6).copy_from(&r_pos);
                total += (whitener * residual).norm_squared();
            } else {
                total += factor.weight_rotation * r_rot.norm_squared()
                    + factor.weight_velocity * r_vel.norm_squared()
                    + factor.weight_position * r_pos.norm_squared();
            }
        }
        if let Some(prior) = &self.navigation_state_prior {
            if let Some(delta) = navigation_prior_delta(self, prior) {
                let quadratic = delta.dot(&(&prior.information * &delta));
                total += prior.constant_cost + 2.0 * prior.gradient.dot(&delta) + quadratic;
            }
        }
        total
    }

    fn intrinsics(&self) -> Option<(f64, f64, f64, f64)> {
        match self.camera.model {
            CameraModel::Pinhole | CameraModel::SimplePinhole => self.camera.intrinsics(),
            _ => None,
        }
    }

    /// Per-observation squared reprojection residual `s = ‖r‖²` (pixel²),
    /// evaluated at the current state and aligned to the GNC weight layout
    /// used everywhere in this file: monocular observations first
    /// (`0 .. observations.len()`), then rectified stereo, then general
    /// stereo. An observation that cannot
    /// be evaluated now (missing pose / landmark, behind the camera, or
    /// non-projectable, or — for stereo — no usable baseline) is reported
    /// as `f64::NAN`, so it neither sets the GNC inlier scale nor is
    /// classified as an inlier or outlier.
    fn reprojection_squared_residuals(&self) -> Vec<f64> {
        let n = self.observations.len()
            + self.stereo_observations.len()
            + self.general_stereo_observations.len()
            + self.rig_observations.len();
        let mut out = Vec::with_capacity(n);
        let Some(intrinsics) = self.intrinsics() else {
            out.resize(n, f64::NAN);
            return out;
        };
        for obs in &self.observations {
            let s = (|| {
                let pose = self.poses.get(&obs.keyframe_id)?;
                let point = self.landmarks.get(&obs.landmark_id)?;
                let xc = pose.transform_world_point(point);
                if xc.z <= 0.0 {
                    return None;
                }
                let predicted = project_pinhole(&intrinsics, &xc)?;
                let r = predicted - obs.xy;
                Some(r.x * r.x + r.y * r.y)
            })();
            out.push(s.unwrap_or(f64::NAN));
        }
        let baseline = match self.stereo_baseline {
            Some(b) if b.is_finite() && b > 0.0 => Some(b),
            _ => None,
        };
        let (fx, _, _, _) = intrinsics;
        for obs in &self.stereo_observations {
            let s = baseline.and_then(|baseline| {
                let pose = self.poses.get(&obs.keyframe_id)?;
                let point = self.landmarks.get(&obs.landmark_id)?;
                let xc = pose.transform_world_point(point);
                if xc.z <= 0.0 {
                    return None;
                }
                let predicted = project_pinhole(&intrinsics, &xc)?;
                let u_r_pred = predicted.x - fx * baseline / xc.z;
                let dx = predicted.x - obs.xy.x;
                let dy = predicted.y - obs.xy.y;
                let dr = u_r_pred - obs.u_right;
                Some(dx * dx + dy * dy + dr * dr)
            });
            out.push(s.unwrap_or(f64::NAN));
        }
        for obs in &self.general_stereo_observations {
            let s = (|| {
                let pose = self.poses.get(&obs.keyframe_id)?;
                let point = self.landmarks.get(&obs.landmark_id)?;
                let (residual, _, _) =
                    general_stereo_residual_jacobians(&intrinsics, obs, pose, point)?;
                Some(residual.norm_squared())
            })();
            out.push(s.unwrap_or(f64::NAN));
        }
        for observation in &self.rig_observations {
            let squared = (|| {
                let pose = self.poses.get(&observation.keyframe_id)?;
                let point = self.landmarks.get(&observation.landmark_id)?;
                let (residual, _, _) = rig_residual_jacobians(observation, pose, point)?;
                Some(residual.norm_squared())
            })();
            out.push(squared.unwrap_or(f64::NAN));
        }
        out
    }

    /// Run Levenberg-Marquardt bundle adjustment with Schur-complement
    /// landmark elimination. Returns iteration trace and final cost.
    pub fn optimize(&mut self, config: &BaConfig) -> Result<BaResult, BaError> {
        if !config.refine_intrinsics
            || !self.general_stereo_observations.is_empty()
            || !self.rig_observations.is_empty()
        {
            return self.optimize_weighted(config, None);
        }
        // Joint pose + structure + intrinsics refinement (the COLMAP self-
        // calibration formulation). Falls back to the pose/structure-only solve
        // for non-pinhole cameras, which carry no refinable 4-parameter intrinsics.
        if self.camera.model != CameraModel::Pinhole || self.camera.intrinsics().is_none() {
            return self.optimize_weighted(config, None);
        }
        self.optimize_joint_intrinsics(config)
    }

    /// Run pose/structure bundle adjustment with one external confidence
    /// weight per visual observation.
    ///
    /// The flattened order is monocular observations first, followed by
    /// rectified stereo and then calibrated general stereo. A weight of `1`
    /// preserves the ordinary BA contribution and `0` disables that visual
    /// residual. Intermediate values provide the confidence-weighted least
    /// squares used by learned correspondence update operators such as
    /// DROID-SLAM. The external weight multiplies the configured robust-kernel
    /// weight; inertial and structural priors are never reweighted.
    ///
    /// Joint intrinsics refinement is deliberately rejected because its
    /// separate normal-equation builder does not yet consume these weights.
    pub fn optimize_with_observation_weights(
        &mut self,
        config: &BaConfig,
        observation_weights: &[f64],
    ) -> Result<BaResult, BaError> {
        self.validate_observation_weights(observation_weights)?;
        if config.refine_intrinsics {
            return Err(BaError::ObservationWeightsWithIntrinsicsRefinement);
        }
        self.optimize_weighted(config, Some(observation_weights))
    }

    /// Run the opt-in matrix-free pure-visual bundle adjustment backend.
    ///
    /// The method deliberately leaves [`BaConfig`] and [`LinearSolver`]
    /// unchanged.  It assembles the same pure-visual normal equations as the
    /// ordinary optimizer, requires a `PoseDiagonal` camera Hessian, and
    /// solves its landmark-eliminated system with bounded preconditioned CG.
    /// `config.linear_solver` is ignored for this method; there is no dense
    /// fallback.  External observation weights and GNC remain on the legacy
    /// weighted entry points.
    /// Existing LM acceptance, rollback, damping, and non-projectable gates
    /// are shared with the ordinary weighted optimizer through a private
    /// backend dispatch.  A failed reduced solve is recorded and rejected by
    /// the same bounded LM retry path before any pose or landmark is modified.
    ///
    /// Gauge completeness is the caller's responsibility.  This entry point
    /// requires at least one actual fixed pose as a minimal anchor, but does
    /// not attempt to prove that every connected component (or monocular
    /// scale) is fully anchored.  Positive damping and PCG convergence are
    /// numerical facts, not a gauge proof.
    pub fn optimize_matrix_free(
        &mut self,
        config: &BaConfig,
        options: MatrixFreeBaOptions,
    ) -> Result<MatrixFreeBaResult, MatrixFreeBaError> {
        self.validate_matrix_free_entry(config, options)?;
        let (result, runtime) =
            self.run_matrix_free_backend(config, MatrixFreeRuntime::new(options))?;
        Ok(MatrixFreeBaResult {
            initial_cost: result.initial_cost,
            final_cost: result.final_cost,
            iterations: result.iterations,
            matrix_free_iterations: runtime.iterations,
            converged: result.converged,
        })
    }

    /// Run matrix-free BA with explicit column equilibration and scaled LM
    /// damping.  This is a separate opt-in policy: the legacy matrix-free
    /// entry point keeps scalar `lambda * I` damping and its exact arithmetic.
    /// The returned PCG residuals are measured in the scaled coordinates.
    pub fn optimize_matrix_free_column_scaled(
        &mut self,
        config: &BaConfig,
        options: MatrixFreeBaColumnScalingOptions,
    ) -> Result<MatrixFreeBaColumnScalingResult, MatrixFreeBaError> {
        self.validate_matrix_free_entry(config, options.pcg)?;
        let (result, runtime) = self.run_matrix_free_column_scaled_backend(
            config,
            MatrixFreeRuntime::with_column_scaling(options.pcg),
        )?;
        Ok(MatrixFreeBaColumnScalingResult {
            ba: MatrixFreeBaResult {
                initial_cost: result.initial_cost,
                final_cost: result.final_cost,
                iterations: result.iterations,
                matrix_free_iterations: runtime.iterations,
                converged: result.converged,
            },
            scaling_iterations: runtime.column_scaling_iterations.unwrap_or_default(),
        })
    }

    /// Run column-scaled matrix-free BA with the opt-in adaptive LM damping
    /// policy.  The existing column-scaled entry point remains unchanged;
    /// this method changes only the accepted-step lambda update and records
    /// its scalar decision trace.  It requires zero initially non-projectable
    /// observations, a finite positive same-observation quadratic prediction,
    /// and the existing cost/feasibility gates for candidate acceptance.
    /// Linear failures and rejected candidates retain the configured lambda
    /// increase factor; only accepted candidates use the fixed rho rule
    /// `max(1/3, 1 - (2*rho - 1)^3)`.  PCG options and the normal-equation
    /// solve are otherwise identical.
    pub fn optimize_matrix_free_column_scaled_adaptive(
        &mut self,
        config: &BaConfig,
        options: MatrixFreeBaColumnScalingOptions,
    ) -> Result<MatrixFreeBaAdaptiveDampingResult, MatrixFreeBaError> {
        self.validate_matrix_free_entry(config, options.pcg)?;
        if self.nonprojectable_observation_count() != 0 {
            return Err(MatrixFreeBaError::Ineligible(
                "adaptive damping requires zero initial non-projectable observations",
            ));
        }
        let (result, runtime) = self.run_matrix_free_column_scaled_backend(
            config,
            MatrixFreeRuntime::with_column_scaling_adaptive(options.pcg),
        )?;
        Ok(MatrixFreeBaAdaptiveDampingResult {
            ba: MatrixFreeBaResult {
                initial_cost: result.initial_cost,
                final_cost: result.final_cost,
                iterations: result.iterations,
                matrix_free_iterations: runtime.iterations,
                converged: result.converged,
            },
            scaling_iterations: runtime.column_scaling_iterations.unwrap_or_default(),
            adaptive_iterations: runtime.adaptive_iterations.unwrap_or_default(),
        })
    }

    /// Run the matrix-free backend with a bounded true-residual restart policy.
    ///
    /// This is an additive diagnostic entry point.  The existing
    /// [`Self::optimize_matrix_free`] path remains the default and does not
    /// allocate restart diagnostics.  A restart is allowed at most once per
    /// reduced linear solve and never resets the total PCG iteration count.
    pub fn optimize_matrix_free_with_restart(
        &mut self,
        config: &BaConfig,
        options: MatrixFreeBaOptions,
        restart: MatrixFreeBaRestartOptions,
    ) -> Result<MatrixFreeBaRestartResult, MatrixFreeBaError> {
        self.validate_matrix_free_entry(config, options)?;
        if restart.max_restarts_per_solve > 1 {
            return Err(MatrixFreeBaError::InvalidConfiguration(
                "max_restarts_per_solve must be 0 or 1",
            ));
        }
        let (result, runtime) = self.run_matrix_free_backend(
            config,
            MatrixFreeRuntime::with_restart(options, restart.max_restarts_per_solve),
        )?;
        Ok(MatrixFreeBaRestartResult {
            ba: MatrixFreeBaResult {
                initial_cost: result.initial_cost,
                final_cost: result.final_cost,
                iterations: result.iterations,
                matrix_free_iterations: runtime.iterations,
                converged: result.converged,
            },
            restart_iterations: runtime.restart_iterations.unwrap_or_default(),
        })
    }

    fn run_matrix_free_backend(
        &mut self,
        config: &BaConfig,
        runtime: MatrixFreeRuntime,
    ) -> Result<(BaResult, MatrixFreeRuntime), MatrixFreeBaError> {
        self.run_matrix_free_backend_variant(config, BaSolveBackend::MatrixFree(runtime))
    }

    fn run_matrix_free_column_scaled_backend(
        &mut self,
        config: &BaConfig,
        runtime: MatrixFreeRuntime,
    ) -> Result<(BaResult, MatrixFreeRuntime), MatrixFreeBaError> {
        self.run_matrix_free_backend_variant(
            config,
            BaSolveBackend::MatrixFreeColumnScaled(runtime),
        )
    }

    fn run_matrix_free_backend_variant(
        &mut self,
        config: &BaConfig,
        mut backend: BaSolveBackend,
    ) -> Result<(BaResult, MatrixFreeRuntime), MatrixFreeBaError> {
        let result = self
            .optimize_weighted_backend(config, None, &mut backend)
            .map_err(|error| {
                let runtime = match &backend {
                    BaSolveBackend::MatrixFree(runtime)
                    | BaSolveBackend::MatrixFreeColumnScaled(runtime) => runtime,
                    BaSolveBackend::Legacy => return MatrixFreeBaError::Ba(error),
                };
                if let Some(failure) = &runtime.failure {
                    return failure.clone();
                }
                MatrixFreeBaError::Ba(error)
            })?;
        let runtime = match backend {
            BaSolveBackend::MatrixFree(runtime)
            | BaSolveBackend::MatrixFreeColumnScaled(runtime) => runtime,
            BaSolveBackend::Legacy => {
                unreachable!("matrix-free entry installs a matrix-free backend")
            }
        };
        Ok((result, runtime))
    }

    fn validate_matrix_free_entry(
        &self,
        config: &BaConfig,
        options: MatrixFreeBaOptions,
    ) -> Result<(), MatrixFreeBaError> {
        if config.refine_intrinsics || config.refine_distortion {
            return Err(MatrixFreeBaError::Ineligible(
                "intrinsics/distortion refinement is not supported",
            ));
        }
        if config.max_iterations == 0 {
            return Err(MatrixFreeBaError::InvalidConfiguration(
                "max_iterations must be positive",
            ));
        }
        let initial_lambda = match config.initial_lambda {
            Some(value) if value.is_finite() && value > 0.0 => value,
            Some(_) => {
                return Err(MatrixFreeBaError::InvalidConfiguration(
                    "initial_lambda must be finite and positive",
                ))
            }
            None => {
                return Err(MatrixFreeBaError::InvalidConfiguration(
                    "matrix-free LM requires an explicit positive initial_lambda",
                ))
            }
        };
        if !config.lambda_increase_factor.is_finite()
            || config.lambda_increase_factor <= 1.0
            || !config.lambda_decrease_factor.is_finite()
            || config.lambda_decrease_factor <= 0.0
            || config.lambda_decrease_factor >= 1.0
            || !config.max_lambda.is_finite()
            || config.max_lambda < initial_lambda
            || !config.min_lambda.is_finite()
            || config.min_lambda <= 0.0
            || config.min_lambda > initial_lambda
            || !config.step_tolerance.is_finite()
            || config.step_tolerance < 0.0
            || !config.cost_tolerance.is_finite()
            || config.cost_tolerance < 0.0
            || config
                .relative_cost_tolerance
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(MatrixFreeBaError::InvalidConfiguration(
                "LM damping/tolerance values are not finite or ordered",
            ));
        }
        if options.max_pcg_iterations == 0
            || !options.pcg_relative_tolerance.is_finite()
            || options.pcg_relative_tolerance < 0.0
            || !options.pcg_absolute_tolerance.is_finite()
            || options.pcg_absolute_tolerance <= 0.0
        {
            return Err(MatrixFreeBaError::InvalidConfiguration(
                "PCG iteration and tolerance values are invalid",
            ));
        }
        match config.robust_kernel {
            RobustKernel::None => {}
            RobustKernel::Huber { delta } if delta.is_finite() && delta > 0.0 => {}
            RobustKernel::Cauchy { c } if c.is_finite() && c > 0.0 => {}
            _ => {
                return Err(MatrixFreeBaError::InvalidConfiguration(
                    "robust-kernel scale must be finite and positive",
                ))
            }
        }

        let calibration_is_finite = |camera: &Camera| {
            let Some((fx, fy, cx, cy)) = camera.intrinsics() else {
                return false;
            };
            let no_nonzero_distortion = camera
                .radial_distortion()
                .is_none_or(|(k1, k2)| k1 == 0.0 && k2 == 0.0);
            matches!(
                camera.model,
                CameraModel::Pinhole | CameraModel::SimplePinhole
            ) && no_nonzero_distortion
                && fx.is_finite()
                && fy.is_finite()
                && cx.is_finite()
                && cy.is_finite()
                && fx > 0.0
                && fy > 0.0
                && camera.params.iter().all(|value| value.is_finite())
        };
        if !calibration_is_finite(&self.camera) {
            return Err(MatrixFreeBaError::Ineligible(
                "camera calibration is unsupported or non-finite",
            ));
        }
        if !self.poses.keys().any(|id| self.fixed_poses.contains(id)) {
            return Err(MatrixFreeBaError::Ineligible(
                "at least one existing pose must be fixed as a gauge anchor",
            ));
        }
        if !self.velocities.is_empty()
            || !self.biases.is_empty()
            || !self.imu_factors.is_empty()
            || !self.fixed_velocities.is_empty()
            || !self.fixed_biases.is_empty()
            || !self.bias_random_walk_factors.is_empty()
            || self.navigation_state_prior.is_some()
            || self.gravity_prior.is_some()
            || self.per_pose_gravity_prior.is_some()
            || self.position_prior.is_some()
            || !self.pairwise_pose_factors.is_empty()
        {
            return Err(MatrixFreeBaError::Ineligible(
                "non-visual states or priors are not supported",
            ));
        }
        if self.poses.is_empty() {
            return Err(MatrixFreeBaError::Ba(BaError::NoPoses));
        }
        let has_visual_observations = !self.observations.is_empty()
            || !self.stereo_observations.is_empty()
            || !self.general_stereo_observations.is_empty()
            || !self.rig_observations.is_empty();
        if !has_visual_observations {
            return Err(MatrixFreeBaError::Ba(BaError::NoObservations));
        }
        if self.landmarks.is_empty() {
            return Err(MatrixFreeBaError::Ba(BaError::NoLandmarks));
        }
        if !self.poses.keys().any(|id| !self.fixed_poses.contains(id)) {
            return Err(MatrixFreeBaError::Ineligible(
                "matrix-free backend requires a variable pose",
            ));
        }
        let pose_and_landmark_are_finite = self.poses.values().all(|pose| {
            pose.world_to_camera
                .matrix()
                .iter()
                .all(|value| value.is_finite())
        }) && self
            .landmarks
            .values()
            .all(|point| point.coords.iter().all(|value| value.is_finite()));
        if !pose_and_landmark_are_finite {
            return Err(MatrixFreeBaError::Ineligible(
                "pose or landmark state is non-finite",
            ));
        }
        if !self
            .robust_cost_weighted(&config.robust_kernel, None)
            .is_finite()
        {
            return Err(MatrixFreeBaError::Ineligible(
                "initial visual cost is non-finite",
            ));
        }
        if !self.observations.iter().all(|observation| {
            observation.xy.coords.iter().all(|value| value.is_finite())
                && self.poses.contains_key(&observation.keyframe_id)
                && self.landmarks.contains_key(&observation.landmark_id)
        }) {
            return Err(MatrixFreeBaError::Ineligible(
                "monocular observation has an invalid pose, landmark, or pixel",
            ));
        }
        if !self.stereo_observations.iter().all(|observation| {
            observation.xy.coords.iter().all(|value| value.is_finite())
                && observation.u_right.is_finite()
                && self.poses.contains_key(&observation.keyframe_id)
                && self.landmarks.contains_key(&observation.landmark_id)
        }) {
            return Err(MatrixFreeBaError::Ineligible(
                "stereo observation has invalid ids or pixels",
            ));
        }
        if !self.stereo_observations.is_empty()
            && !matches!(
                self.stereo_baseline,
                Some(value) if value.is_finite() && value > 0.0
            )
        {
            return Err(MatrixFreeBaError::Ba(BaError::MissingStereoBaseline));
        }
        if !self.general_stereo_observations.iter().all(|observation| {
            observation
                .xy_left
                .coords
                .iter()
                .all(|value| value.is_finite())
                && observation
                    .xy_right
                    .coords
                    .iter()
                    .all(|value| value.is_finite())
                && calibration_is_finite(&observation.right_camera)
                && self.poses.contains_key(&observation.keyframe_id)
                && self.landmarks.contains_key(&observation.landmark_id)
                && observation
                    .left_to_right
                    .translation
                    .iter()
                    .all(|value| value.is_finite())
                && observation
                    .left_to_right
                    .rotation
                    .coords
                    .iter()
                    .all(|value| value.is_finite())
        }) {
            return Err(MatrixFreeBaError::Ineligible(
                "general stereo calibration or observation is invalid",
            ));
        }
        if !self.rig_observations.iter().all(|observation| {
            observation.xy.coords.iter().all(|value| value.is_finite())
                && calibration_is_finite(&observation.camera)
                && self.poses.contains_key(&observation.keyframe_id)
                && self.landmarks.contains_key(&observation.landmark_id)
                && observation
                    .sensor_from_rig
                    .translation
                    .iter()
                    .all(|value| value.is_finite())
                && observation
                    .sensor_from_rig
                    .rotation
                    .coords
                    .iter()
                    .all(|value| value.is_finite())
        }) {
            return Err(MatrixFreeBaError::Ineligible(
                "rig calibration or observation is invalid",
            ));
        }
        Ok(())
    }

    fn validate_observation_weights(&self, observation_weights: &[f64]) -> Result<(), BaError> {
        let expected = self.observations.len()
            + self.stereo_observations.len()
            + self.general_stereo_observations.len()
            + self.rig_observations.len();
        if observation_weights.len() != expected {
            return Err(BaError::ObservationWeightCount {
                expected,
                actual: observation_weights.len(),
            });
        }
        if let Some((index, _)) = observation_weights
            .iter()
            .enumerate()
            .find(|(_, weight)| !weight.is_finite() || **weight < 0.0)
        {
            return Err(BaError::InvalidObservationWeight(index));
        }
        Ok(())
    }

    /// Bundle adjustment that carries the shared pinhole intrinsics
    /// `(fx, fy, cx, cy)` as four extra unknowns **inside** the Schur-complement
    /// camera system, jointly with the poses and (eliminated) landmarks.
    ///
    /// This is the difference that matters versus an *alternating* refinement
    /// (update the intrinsics by Gauss-Newton against a *converged* structure,
    /// then re-solve): there the structure-fixed gradient `∂cost/∂K` is ≈ 0 (the
    /// structure has already absorbed any focal error), so it cannot move a wrong
    /// focal. The joint solve uses the **coupled** gradient — the
    /// reduced-camera gradient *after* landmark elimination — which is non-zero,
    /// so it pulls the intrinsics and poses together toward the true calibration.
    ///
    /// SfM-only: handles monocular + rectified-stereo reprojection observations
    /// and ignores IMU / velocity / bias / gravity / position-prior factors (which
    /// SfM intrinsics refinement does not use). The intrinsics are always a free
    /// block; the caller fixes poses (anchor + farthest, or ≥2 stereo observers) to
    /// pin the remaining gauge. Writes refined poses, landmarks, and intrinsics
    /// into `self`.
    fn optimize_joint_intrinsics(&mut self, config: &BaConfig) -> Result<BaResult, BaError> {
        let kernel = config.robust_kernel;

        // Variable layout: non-fixed poses occupy `6·p .. 6·p+6`; the 4 shared
        // intrinsics occupy the final block `k_off .. k_off+4`. Fixed poses and
        // fixed landmarks contribute residuals but get no variable slot.
        let mut pose_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if self.fixed_poses.contains(&id) {
                continue;
            }
            let next = pose_index.len();
            pose_index.insert(id, next);
        }
        let mut landmark_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.landmarks.keys() {
            if self.fixed_landmarks.contains(&id) {
                continue;
            }
            let next = landmark_index.len();
            landmark_index.insert(id, next);
        }
        let p_count = pose_index.len();
        let k_off = p_count * 6;
        // Also self-calibrate radial distortion (k1, k2) when asked — but only on
        // a monocular reconstruction (rectified stereo is already undistorted, and
        // its baseline term does not carry a distortion model). The two coefficients
        // get appended to the camera block, so `k_dim` is 6 instead of 4.
        let refine_dist = config.refine_distortion
            && self.camera.model == CameraModel::Pinhole
            && self.stereo_observations.is_empty();
        if refine_dist {
            // Ensure the camera carries the two distortion slots (start at 0).
            while self.camera.params.len() < 6 {
                self.camera.params.push(0.0);
            }
        }
        let k_dim = if refine_dist { 6 } else { 4 };
        let cam_dim = k_off + k_dim;

        let initial_cost = self.robust_cost_weighted(&kernel, None);
        let mut iterations: Vec<BaIterationStats> = Vec::with_capacity(config.max_iterations);
        let mut current_cost = initial_cost;
        let mut current_nonprojectable = self.nonprojectable_observation_count();
        let mut lambda = config.initial_lambda.unwrap_or(0.0);
        let mut converged = false;

        for iteration in 0..config.max_iterations {
            // Current distortion (reflects the running k1, k2 estimate) drives the
            // distortion-aware projection / Jacobians inside the build.
            let dist = self.camera.radial_distortion();
            let (cam_dim_n, h_cc, b_c, lm_blocks) = self.build_joint_intrinsics_system(
                &pose_index,
                &landmark_index,
                &kernel,
                k_dim,
                dist,
            );
            debug_assert_eq!(cam_dim_n, cam_dim);

            // Damped Schur reduction (Levenberg I·λ on both the camera and the
            // landmark diagonals, exactly as `solve_step`).
            let mut s = h_cc.clone();
            if lambda > 0.0 {
                for d in 0..cam_dim {
                    s[(d, d)] += lambda;
                }
            }
            let mut b_reduced = -&b_c;
            let mut h_ll_inv_cache: Vec<Option<Matrix3<f64>>> = Vec::with_capacity(lm_blocks.len());
            for lm in &lm_blocks {
                let mut h_ll = lm.h_ll;
                if lambda > 0.0 {
                    h_ll[(0, 0)] += lambda;
                    h_ll[(1, 1)] += lambda;
                    h_ll[(2, 2)] += lambda;
                }
                let inv = h_ll.try_inverse();
                h_ll_inv_cache.push(inv);
                let Some(inv) = inv else { continue };
                // S -= Σ cross_a^T · H_ll^{-1} · cross_b ; b += cross_a^T H_ll^{-1} b_l.
                for (cs_a, a) in &lm.cross {
                    let ah = a * inv; // (rows_a × 3)
                    for (cs_b, b) in &lm.cross {
                        let block = &ah * b.transpose(); // (rows_a × rows_b)
                        for r in 0..a.nrows() {
                            for c in 0..b.nrows() {
                                s[(cs_a + r, cs_b + c)] -= block[(r, c)];
                            }
                        }
                    }
                    let upd = &ah * lm.b_l; // (rows_a)
                    for r in 0..a.nrows() {
                        b_reduced[cs_a + r] += upd[r];
                    }
                }
            }

            let delta_cam = match solve_normal_equations(&s, &b_reduced) {
                Ok(d) => d,
                Err(_) => {
                    lambda = (lambda * config.lambda_increase_factor).min(config.max_lambda);
                    iterations.push(BaIterationStats {
                        iteration,
                        cost_before: current_cost,
                        cost_after: current_cost,
                        max_pose_step: 0.0,
                        max_landmark_step: 0.0,
                        lambda,
                        step_accepted: false,
                    });
                    if lambda >= config.max_lambda {
                        break;
                    }
                    continue;
                }
            };

            // Back-substitute landmark updates: δ_L = H_ll^{-1}(−b_l − Σ crossᵀ δ_cam).
            let mut delta_lm: BTreeMap<u64, Vector3<f64>> = BTreeMap::new();
            for (lm, inv) in lm_blocks.iter().zip(&h_ll_inv_cache) {
                let Some(inv) = inv else { continue };
                let mut acc = -lm.b_l;
                for (cs, a) in &lm.cross {
                    let mut dcam = DVector::<f64>::zeros(a.nrows());
                    for r in 0..a.nrows() {
                        dcam[r] = delta_cam[cs + r];
                    }
                    acc -= a.transpose() * dcam;
                }
                delta_lm.insert(lm.id, inv * acc);
            }

            // Tentative update (save → apply → cost → accept/reject).
            let saved_poses = self.poses.clone();
            let saved_landmarks = self.landmarks.clone();
            let saved_params = self.camera.params.clone();
            let cost_before = current_cost;

            let mut max_pose_step = 0.0f64;
            for (&id, &p) in &pose_index {
                let xi: Vector6<f64> = delta_cam.fixed_rows::<6>(p * 6).into_owned();
                max_pose_step = max_pose_step.max(xi.norm());
                let pose = self.poses.get_mut(&id).expect("pose exists");
                pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&xi));
            }
            let mut max_landmark_step = 0.0f64;
            for (&id, dl) in &delta_lm {
                max_landmark_step = max_landmark_step.max(dl.norm());
                let pt = self.landmarks.get_mut(&id).expect("landmark exists");
                *pt = Point3::from(pt.coords + dl);
            }
            // Intrinsics (and, when k_dim == 6, distortion) block.
            for j in 0..k_dim {
                self.camera.params[j] += delta_cam[k_off + j];
            }

            let cost_after = self.robust_cost_weighted(&kernel, None);
            let nonprojectable_after = self.nonprojectable_observation_count();
            let cost_accepted = match config.initial_lambda {
                None => true,
                Some(_) => cost_after < cost_before,
            };
            let step_accepted = cost_accepted && nonprojectable_after <= current_nonprojectable;
            if !step_accepted {
                self.poses = saved_poses;
                self.landmarks = saved_landmarks;
                self.camera.params = saved_params;
                lambda = (lambda * config.lambda_increase_factor).min(config.max_lambda);
                iterations.push(BaIterationStats {
                    iteration,
                    cost_before,
                    cost_after,
                    max_pose_step,
                    max_landmark_step,
                    lambda,
                    step_accepted: false,
                });
                if config.initial_lambda.is_none() {
                    break;
                }
                if lambda >= config.max_lambda {
                    break;
                }
                continue;
            }

            iterations.push(BaIterationStats {
                iteration,
                cost_before,
                cost_after,
                max_pose_step,
                max_landmark_step,
                lambda,
                step_accepted: true,
            });
            current_cost = cost_after;
            current_nonprojectable = nonprojectable_after;
            if config.initial_lambda.is_some() {
                lambda = (lambda * config.lambda_decrease_factor).max(config.min_lambda);
            }
            if max_pose_step < config.step_tolerance && max_landmark_step < config.step_tolerance {
                converged = true;
                break;
            }
            if (cost_before - cost_after).abs() < config.cost_tolerance {
                converged = true;
                break;
            }
            if config.relative_cost_tolerance.is_some_and(|tolerance| {
                tolerance.is_finite()
                    && tolerance >= 0.0
                    && (cost_before - cost_after) / cost_before.abs().max(f64::EPSILON) < tolerance
            }) {
                converged = true;
                break;
            }
        }

        Ok(BaResult {
            initial_cost,
            final_cost: current_cost,
            iterations,
            converged,
        })
    }

    /// Assemble the raw (un-damped) joint normal equations for
    /// [`Self::optimize_joint_intrinsics`]: the camera-block Hessian `H_cc`
    /// (poses then the 4 intrinsics) and gradient `b_c`, plus per-landmark
    /// `{H_ll, b_l, cross}` blocks where `cross` maps each touching camera-block
    /// column-start to `Jᵀ_cam · J_lm`. Mirrors `build_normal_equations`'
    /// reprojection Jacobians, extended with the intrinsics columns
    /// `J_K = ∂(predicted)/∂(fx, fy, cx, cy)`.
    fn build_joint_intrinsics_system(
        &self,
        pose_index: &BTreeMap<u64, usize>,
        landmark_index: &BTreeMap<u64, usize>,
        kernel: &RobustKernel,
        k_dim: usize,
        dist: Option<(f64, f64)>,
    ) -> (usize, DMatrix<f64>, DVector<f64>, Vec<JointLandmarkBlock>) {
        let intrinsics = self.intrinsics().expect("pinhole checked by caller");
        let (fx, fy, cx, cy) = intrinsics;
        let p_count = pose_index.len();
        let k_off = p_count * 6;
        let cam_dim = k_off + k_dim;
        let mut h_cc = DMatrix::<f64>::zeros(cam_dim, cam_dim);
        let mut b_c = DVector::<f64>::zeros(cam_dim);
        let mut lm_blocks: Vec<JointLandmarkBlock> = landmark_index
            .iter()
            .map(|(&id, _)| JointLandmarkBlock {
                id,
                h_ll: Matrix3::zeros(),
                b_l: Vector3::zeros(),
                cross: BTreeMap::new(),
            })
            .collect();

        // Accumulate a camera×camera block (rows_a × cols_b) at (row_start, col_start).
        let mut add_cc = |rs: usize, cs: usize, blk: &DMatrix<f64>| {
            for r in 0..blk.nrows() {
                for c in 0..blk.ncols() {
                    h_cc[(rs + r, cs + c)] += blk[(r, c)];
                }
            }
        };

        // Monocular observations.
        for obs in &self.observations {
            let pose = &self.poses[&obs.keyframe_id];
            let point = &self.landmarks[&obs.landmark_id];
            let xc = pose.transform_world_point(point);
            if xc.z <= 0.0 {
                continue;
            }
            let z_inv = 1.0 / xc.z;
            let x = xc.x * z_inv;
            let y = xc.y * z_inv;
            let r2 = x * x + y * y;
            // Radial distortion factor d = 1 + k1·r² + k2·r⁴ and its radial
            // derivative helper g = k1 + 2·k2·r² (d=1, g=0 when distortion-free).
            let (k1, k2) = dist.unwrap_or((0.0, 0.0));
            let d = 1.0 + k1 * r2 + k2 * r2 * r2;
            let g = k1 + 2.0 * k2 * r2;
            let (xd, yd) = (x * d, y * d);
            let predicted = Point2::new(fx * xd + cx, fy * yd + cy);
            let residual = Vector2::new(predicted.x - obs.xy.x, predicted.y - obs.xy.y);
            let r_mat = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            // J_π = diag(fx, fy) · D · ∂(x, y)/∂X_c, where the distortion Jacobian
            //   D = [[d + 2x²g, 2xyg], [2xyg, d + 2y²g]]  (= I when distortion-free)
            // and ∂(x, y)/∂X_c = (1/Z)·[[1, 0, -x], [0, 1, -y]].
            let d11 = d + 2.0 * x * x * g;
            let d12 = 2.0 * x * y * g;
            let d22 = d + 2.0 * y * y * g;
            let mut j_pi = Matrix2x3::<f64>::zeros();
            j_pi[(0, 0)] = fx * d11 * z_inv;
            j_pi[(0, 1)] = fx * d12 * z_inv;
            j_pi[(0, 2)] = -fx * (d11 * x + d12 * y) * z_inv;
            j_pi[(1, 0)] = fy * d12 * z_inv;
            j_pi[(1, 1)] = fy * d22 * z_inv;
            j_pi[(1, 2)] = -fy * (d12 * x + d22 * y) * z_inv;
            let mut dx_dxi = Matrix3x6::<f64>::zeros();
            dx_dxi.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
            dx_dxi
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-r_mat * skew(&point.coords)));
            let j_pose: Matrix2x6<f64> = j_pi * dx_dxi;
            let j_lm: Matrix2x3<f64> = j_pi * r_mat;
            // ∂(predicted)/∂K with K = (fx, fy, cx, cy[, k1, k2]) (2×k_dim).
            let mut j_k = DMatrix::<f64>::zeros(2, k_dim);
            j_k[(0, 0)] = xd;
            j_k[(0, 2)] = 1.0;
            j_k[(1, 1)] = yd;
            j_k[(1, 3)] = 1.0;
            if k_dim == 6 {
                j_k[(0, 4)] = fx * x * r2;
                j_k[(0, 5)] = fx * x * r2 * r2;
                j_k[(1, 4)] = fy * y * r2;
                j_k[(1, 5)] = fy * y * r2 * r2;
            }

            let s = residual.x * residual.x + residual.y * residual.y;
            let w = kernel.weight(s);
            let i_pose = pose_index.get(&obs.keyframe_id).copied();
            let i_lm = landmark_index.get(&obs.landmark_id).copied();

            // Dynamic-sized residual / pose for the K-coupled products.
            let res2 = DVector::from_column_slice(&[residual.x, residual.y]);
            let jkt = j_k.transpose(); // k_dim×2

            // K-K and K gradient (intrinsics are always variable).
            add_cc(k_off, k_off, &(w * (&jkt * &j_k)));
            let bk = w * (&jkt * &res2);
            for j in 0..k_dim {
                b_c[k_off + j] += bk[j];
            }
            if let Some(p) = i_pose {
                let hpp = w * (j_pose.transpose() * j_pose);
                add_cc(p * 6, p * 6, &DMatrix::from_fn(6, 6, |r, c| hpp[(r, c)]));
                let bp = w * (j_pose.transpose() * residual);
                for r in 0..6 {
                    b_c[p * 6 + r] += bp[r];
                }
                // pose-K coupling (and its transpose).
                let jp_dyn = DMatrix::from_iterator(2, 6, j_pose.iter().copied());
                let hpk = w * (jp_dyn.transpose() * &j_k); // 6×k_dim
                add_cc(p * 6, k_off, &hpk);
                add_cc(k_off, p * 6, &hpk.transpose());
            }
            if let Some(l) = i_lm {
                lm_blocks[l].h_ll += w * (j_lm.transpose() * j_lm);
                lm_blocks[l].b_l += w * (j_lm.transpose() * residual);
                if let Some(p) = i_pose {
                    let cr = w * (j_pose.transpose() * j_lm); // 6×3
                    add_cross(
                        &mut lm_blocks[l].cross,
                        p * 6,
                        6,
                        &DMatrix::from_fn(6, 3, |r, c| cr[(r, c)]),
                    );
                }
                let jl_dyn = DMatrix::from_iterator(2, 3, j_lm.iter().copied());
                let crk = w * (&jkt * &jl_dyn); // k_dim×3
                add_cross(&mut lm_blocks[l].cross, k_off, k_dim, &crk);
            }
        }

        // Rectified-stereo observations (3D residual u_l, v_l, u_r).
        if !self.stereo_observations.is_empty() {
            if let Some(baseline) = self.stereo_baseline {
                if baseline.is_finite() && baseline > 0.0 {
                    for obs in &self.stereo_observations {
                        let pose = &self.poses[&obs.keyframe_id];
                        let point = &self.landmarks[&obs.landmark_id];
                        let xc = pose.transform_world_point(point);
                        if xc.z <= 0.0 {
                            continue;
                        }
                        let Some(predicted) = project_pinhole(&intrinsics, &xc) else {
                            continue;
                        };
                        let z_inv = 1.0 / xc.z;
                        let z_inv2 = z_inv * z_inv;
                        let u_r_pred = predicted.x - fx * baseline * z_inv;
                        let residual = Vector3::new(
                            predicted.x - obs.xy.x,
                            predicted.y - obs.xy.y,
                            u_r_pred - obs.u_right,
                        );
                        let r_mat = pose
                            .world_to_camera
                            .rotation
                            .to_rotation_matrix()
                            .into_inner();
                        let mut j_pi = Matrix3::<f64>::zeros();
                        j_pi[(0, 0)] = fx * z_inv;
                        j_pi[(0, 2)] = -fx * xc.x * z_inv2;
                        j_pi[(1, 1)] = fy * z_inv;
                        j_pi[(1, 2)] = -fy * xc.y * z_inv2;
                        j_pi[(2, 0)] = fx * z_inv;
                        j_pi[(2, 2)] = -fx * (xc.x - baseline) * z_inv2;
                        let mut dx_dxi = Matrix3x6::<f64>::zeros();
                        dx_dxi.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
                        dx_dxi
                            .fixed_view_mut::<3, 3>(0, 3)
                            .copy_from(&(-r_mat * skew(&point.coords)));
                        let j_pose: Matrix3x6<f64> = j_pi * dx_dxi;
                        let j_lm: Matrix3<f64> = j_pi * r_mat;
                        // u_r = fx·(X−b)/Z + cx, so ∂u_r/∂fx = (X−b)/Z, ∂u_r/∂cx = 1.
                        let mut j_k = Matrix3x4::<f64>::zeros();
                        j_k[(0, 0)] = xc.x * z_inv;
                        j_k[(0, 2)] = 1.0;
                        j_k[(1, 1)] = xc.y * z_inv;
                        j_k[(1, 3)] = 1.0;
                        j_k[(2, 0)] = (xc.x - baseline) * z_inv;
                        j_k[(2, 2)] = 1.0;

                        let s = residual.norm_squared();
                        let w = kernel.weight(s);
                        let i_pose = pose_index.get(&obs.keyframe_id).copied();
                        let i_lm = landmark_index.get(&obs.landmark_id).copied();

                        add_cc(
                            k_off,
                            k_off,
                            &DMatrix::from_fn(4, 4, |r, c| w * (j_k.transpose() * j_k)[(r, c)]),
                        );
                        let bk = w * (j_k.transpose() * residual);
                        for j in 0..4 {
                            b_c[k_off + j] += bk[j];
                        }
                        if let Some(p) = i_pose {
                            let hpp = w * (j_pose.transpose() * j_pose);
                            add_cc(p * 6, p * 6, &DMatrix::from_fn(6, 6, |r, c| hpp[(r, c)]));
                            let bp = w * (j_pose.transpose() * residual);
                            for r in 0..6 {
                                b_c[p * 6 + r] += bp[r];
                            }
                            let hpk = w * (j_pose.transpose() * j_k);
                            add_cc(p * 6, k_off, &DMatrix::from_fn(6, 4, |r, c| hpk[(r, c)]));
                            add_cc(k_off, p * 6, &DMatrix::from_fn(4, 6, |r, c| hpk[(c, r)]));
                        }
                        if let Some(l) = i_lm {
                            lm_blocks[l].h_ll += w * (j_lm.transpose() * j_lm);
                            lm_blocks[l].b_l += w * (j_lm.transpose() * residual);
                            if let Some(p) = i_pose {
                                let cr = w * (j_pose.transpose() * j_lm);
                                add_cross(
                                    &mut lm_blocks[l].cross,
                                    p * 6,
                                    6,
                                    &DMatrix::from_fn(6, 3, |r, c| cr[(r, c)]),
                                );
                            }
                            let crk = w * (j_k.transpose() * j_lm);
                            add_cross(
                                &mut lm_blocks[l].cross,
                                k_off,
                                4,
                                &DMatrix::from_fn(4, 3, |r, c| crk[(r, c)]),
                            );
                        }
                    }
                }
            }
        }

        (cam_dim, h_cc, b_c, lm_blocks)
    }

    /// Outlier-robust bundle adjustment via Graduated Non-Convexity (GNC).
    ///
    /// A local M-estimator (`RobustKernel::Huber` / `Cauchy`) only
    /// down-weights gross reprojection errors *near* the current estimate,
    /// so a cluster of wrong correspondences that the initialisation
    /// already believes can capture the solution in a bad basin. GNC
    /// instead anneals a control parameter `μ` from a convex surrogate
    /// (every observation trusted — ordinary least squares) toward the true
    /// non-convex robust cost, recomputing the per-observation
    /// Black-Rangarajan weight `w ∈ [0,1]` at each level. Each level is a
    /// bounded weighted-LS solve reusing the same Schur-complement assembly
    /// as [`Self::optimize`] with `RobustKernel::None` (GNC supersedes the
    /// M-estimator). See [`crate::gnc`] for the surrogate math.
    ///
    /// `config` drives the inner LM solve (linear solver, λ schedule);
    /// `config.robust_kernel` is ignored — GNC sets the weights. `gnc.c` is
    /// the inlier reprojection scale **in pixels** (so `c²` is the squared-
    /// residual band): pick it from the expected inlier reprojection error,
    /// e.g. `c ≈ 3` for ~1 px noise. The returned
    /// [`BaGncResult::observation_weights`] gives the final per-observation
    /// weight (monocular, rectified stereo, then general stereo; `NaN` for un-evaluable
    /// observations); near-zero entries are the rejected outliers.
    pub fn optimize_gnc(
        &mut self,
        config: &BaConfig,
        gnc: &GncConfig,
    ) -> Result<BaGncResult, BaError> {
        let kernel_none = RobustKernel::None;
        let initial_cost = self.robust_cost(&kernel_none);
        let n = self.observations.len()
            + self.stereo_observations.len()
            + self.general_stereo_observations.len()
            + self.rig_observations.len();

        // GNC inlier scale: largest residual seeds the convex μ₀; the same
        // residuals optionally drive the MAD auto-estimate of `c` (with the
        // configured `c` as a floor) so the pixel threshold tracks the actual
        // reprojection noise instead of a hand-set value.
        let squared_residuals = self.reprojection_squared_residuals();
        let s_max = squared_residuals
            .iter()
            .copied()
            .filter(|s| s.is_finite())
            .fold(0.0_f64, f64::max);
        let effective_gnc = match gnc.auto_scale {
            Some(k) => {
                let c = crate::gnc::estimate_scale_mad(&squared_residuals, k)
                    .map_or(gnc.c, |est| est.max(gnc.c));
                GncConfig { c, ..*gnc }
            }
            None => *gnc,
        };
        let mut inlier_scale = effective_gnc.c;
        let mut state = GncState::new(&effective_gnc, s_max);

        // Inner solve: a short weighted LM with no M-estimator (the GNC
        // weights are the only robustification) restarted at each μ level.
        let mut inner = *config;
        inner.robust_kernel = RobustKernel::None;
        inner.max_iterations = gnc.inner_iterations.max(1);

        let mut weights = vec![1.0_f64; n];
        let mut converged = false;
        let mut outer_iterations = 0usize;
        for _ in 0..gnc.max_outer.max(1) {
            outer_iterations += 1;
            // The terminal level (μ at its recovered extreme) reproduces the
            // true robust cost; we run it, then stop.
            let terminal_level = state.is_terminal();
            let residuals = self.reprojection_squared_residuals();
            // Adaptive inlier scale: re-derive `c` from the current residuals
            // each level (configured `c` as a floor). Level 0 reproduces the
            // one-shot estimate; later levels tighten as the surrogate
            // suppresses outliers and inlier residuals shrink.
            if gnc.auto_scale_readapt {
                if let Some(k) = gnc.auto_scale {
                    if let Some(est) = crate::gnc::estimate_scale_mad(&residuals, k) {
                        let c = est.max(gnc.c);
                        state.set_inlier_scale(c);
                        inlier_scale = c;
                    }
                }
            }
            for (w, &s) in weights.iter_mut().zip(residuals.iter()) {
                *w = if s.is_finite() { state.weight(s) } else { 1.0 };
            }
            self.optimize_weighted(&inner, Some(&weights))?;
            if terminal_level {
                converged = true;
                break;
            }
            state.anneal();
        }

        // Final per-observation weights at the recovered estimate (NaN for
        // observations that cannot be evaluated, matching the result
        // contract and skipped by the weighted cost anyway).
        let residuals = self.reprojection_squared_residuals();
        for (w, &s) in weights.iter_mut().zip(residuals.iter()) {
            *w = if s.is_finite() {
                state.weight(s)
            } else {
                f64::NAN
            };
        }
        let final_cost = self.robust_cost_weighted(&kernel_none, Some(&weights));

        // Inlier-only cost: hard 0/1 mask at the classification threshold,
        // so the reported cost reflects what survives outlier rejection.
        const INLIER_THRESHOLD: f64 = 0.5;
        let inlier_mask: Vec<f64> = weights
            .iter()
            .map(|&w| {
                if w.is_finite() {
                    if w >= INLIER_THRESHOLD {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    f64::NAN
                }
            })
            .collect();
        let inlier_cost = self.robust_cost_weighted(&kernel_none, Some(&inlier_mask));

        Ok(BaGncResult {
            initial_cost,
            final_cost,
            inlier_cost,
            inlier_scale,
            observation_count: n,
            outer_iterations,
            converged,
            observation_weights: weights,
        })
    }

    /// Levenberg-Marquardt bundle adjustment with optional per-observation
    /// GNC weights folded into every reprojection contribution. `None`
    /// runs standard (optionally `RobustKernel`-IRLS) BA, bit-identical to
    /// the public [`Self::optimize`]; `Some(weights)` is the inner solve of
    /// [`Self::optimize_gnc`], where `weights` are the current
    /// Graduated-Non-Convexity surrogate weights and the cost used for the
    /// LM accept / reject test is correspondingly reweighted. `weights` is
    /// indexed monocular, rectified stereo, then general stereo (see
    /// [`Self::robust_cost_weighted`]).
    fn optimize_weighted(
        &mut self,
        config: &BaConfig,
        gnc_weights: Option<&[f64]>,
    ) -> Result<BaResult, BaError> {
        let mut backend = BaSolveBackend::Legacy;
        self.optimize_weighted_backend(config, gnc_weights, &mut backend)
    }

    fn optimize_weighted_backend(
        &mut self,
        config: &BaConfig,
        gnc_weights: Option<&[f64]>,
        mut backend: &mut BaSolveBackend,
    ) -> Result<BaResult, BaError> {
        let intrinsics = self.intrinsics().ok_or(BaError::UnsupportedCameraModel)?;
        if self.poses.is_empty() {
            return Err(BaError::NoPoses);
        }
        let has_visual_observations = !self.observations.is_empty()
            || !self.stereo_observations.is_empty()
            || !self.general_stereo_observations.is_empty()
            || !self.rig_observations.is_empty();
        let has_imu_factors = !self.imu_factors.is_empty();
        if !has_visual_observations && !has_imu_factors {
            return Err(BaError::NoObservations);
        }
        // Visual residuals require landmarks; inertial-only solves (used by
        // the motion-based VI initialiser's VIBA1 stage) do not.
        if has_visual_observations && self.landmarks.is_empty() {
            return Err(BaError::NoLandmarks);
        }
        for obs in &self.observations {
            if !self.poses.contains_key(&obs.keyframe_id) {
                return Err(BaError::MissingPose(obs.keyframe_id));
            }
            if !self.landmarks.contains_key(&obs.landmark_id) {
                return Err(BaError::MissingLandmark(obs.landmark_id));
            }
        }
        if !self.stereo_observations.is_empty() {
            match self.stereo_baseline {
                Some(b) if b.is_finite() && b > 0.0 => {}
                _ => return Err(BaError::MissingStereoBaseline),
            }
            for obs in &self.stereo_observations {
                if !self.poses.contains_key(&obs.keyframe_id) {
                    return Err(BaError::MissingPose(obs.keyframe_id));
                }
                if !self.landmarks.contains_key(&obs.landmark_id) {
                    return Err(BaError::MissingLandmark(obs.landmark_id));
                }
            }
        }
        for obs in &self.general_stereo_observations {
            if !self.poses.contains_key(&obs.keyframe_id) {
                return Err(BaError::MissingPose(obs.keyframe_id));
            }
            if !self.landmarks.contains_key(&obs.landmark_id) {
                return Err(BaError::MissingLandmark(obs.landmark_id));
            }
            if obs.right_camera.intrinsics().is_none() {
                return Err(BaError::UnsupportedCameraModel);
            }
        }
        for observation in &self.rig_observations {
            if !self.poses.contains_key(&observation.keyframe_id) {
                return Err(BaError::MissingPose(observation.keyframe_id));
            }
            if !self.landmarks.contains_key(&observation.landmark_id) {
                return Err(BaError::MissingLandmark(observation.landmark_id));
            }
            if observation.camera.intrinsics().is_none() {
                return Err(BaError::UnsupportedCameraModel);
            }
        }

        // Variable layout: only non-fixed entries get a slot in the linear
        // system. Fixed poses / landmarks contribute residuals but no Hessian
        // or gradient block.
        let mut pose_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if self.fixed_poses.contains(&id) {
                continue;
            }
            let next = pose_index.len();
            pose_index.insert(id, next);
        }
        let mut landmark_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.landmarks.keys() {
            if self.fixed_landmarks.contains(&id) {
                continue;
            }
            let next = landmark_index.len();
            landmark_index.insert(id, next);
        }
        // Velocity slots: non-fixed velocities that appear on at least
        // one IMU factor. We DO NOT add a slot for every registered
        // velocity — only the ones the factor touches — so a stray
        // `add_velocity` without a corresponding `add_imu_factor` does
        // not introduce an unconstrained DoF that would singularise the
        // system.
        let mut velocity_index: BTreeMap<u64, usize> = BTreeMap::new();
        for factor in &self.imu_factors {
            for kf_id in [factor.keyframe_id_from, factor.keyframe_id_to] {
                if !self.velocities.contains_key(&kf_id) {
                    continue;
                }
                if self.fixed_velocities.contains(&kf_id) {
                    continue;
                }
                if velocity_index.contains_key(&kf_id) {
                    continue;
                }
                let next = velocity_index.len();
                velocity_index.insert(kf_id, next);
            }
        }
        if let Some(prior) = &self.navigation_state_prior {
            for &kf_id in &prior.keyframe_ids {
                if self.velocities.contains_key(&kf_id)
                    && !self.fixed_velocities.contains(&kf_id)
                    && !velocity_index.contains_key(&kf_id)
                {
                    let next = velocity_index.len();
                    velocity_index.insert(kf_id, next);
                }
            }
        }
        // Bias slots: non-fixed biases registered on the "from" side of
        // an IMU factor OR on either side of a bias random-walk factor.
        // Same singularity guard as `velocity_index` — a stray
        // `add_bias` without a matching factor does not introduce an
        // unconstrained DoF.
        let mut bias_index: BTreeMap<u64, usize> = BTreeMap::new();
        let register_bias_slot = |kf_id: u64, idx: &mut BTreeMap<u64, usize>| {
            if !self.biases.contains_key(&kf_id) {
                return;
            }
            if self.fixed_biases.contains(&kf_id) {
                return;
            }
            if idx.contains_key(&kf_id) {
                return;
            }
            let next = idx.len();
            idx.insert(kf_id, next);
        };
        for factor in &self.imu_factors {
            register_bias_slot(factor.keyframe_id_from, &mut bias_index);
        }
        for factor in &self.bias_random_walk_factors {
            register_bias_slot(factor.keyframe_id_from, &mut bias_index);
            register_bias_slot(factor.keyframe_id_to, &mut bias_index);
        }
        if let Some(prior) = &self.navigation_state_prior {
            for &kf_id in &prior.keyframe_ids {
                register_bias_slot(kf_id, &mut bias_index);
            }
        }
        if pose_index.is_empty()
            && landmark_index.is_empty()
            && velocity_index.is_empty()
            && bias_index.is_empty()
        {
            return Err(BaError::AllPosesFixed);
        }

        let kernel = config.robust_kernel;
        let initial_cost = self.robust_cost_weighted(&kernel, gnc_weights);
        let mut iterations: Vec<BaIterationStats> = Vec::with_capacity(config.max_iterations);
        let mut current_cost = initial_cost;
        let mut current_nonprojectable = self.nonprojectable_observation_count();
        let mut lambda = config.initial_lambda.unwrap_or(0.0);
        let mut converged = false;
        let mut block_symbolic_cache = None;

        for iteration in 0..config.max_iterations {
            let adaptive_damping = match &backend {
                BaSolveBackend::MatrixFreeColumnScaled(runtime) => runtime.adaptive_damping,
                _ => false,
            };
            let prefer_pose_blocks = matches!(
                backend,
                BaSolveBackend::MatrixFree(_) | BaSolveBackend::MatrixFreeColumnScaled(_)
            ) || config.linear_solver == LinearSolver::Sparse;
            let mut system = build_normal_equations(
                self,
                &intrinsics,
                &pose_index,
                &landmark_index,
                &velocity_index,
                &bias_index,
                &kernel,
                gnc_weights,
                config.parallel,
                prefer_pose_blocks,
            );
            constrain_fixed_pose_rotations(&self.fixed_pose_rotations, &pose_index, &mut system);
            log_process_memory("ba-after-normal-equations");

            // Build the reduced (Schur-complement) camera system. λ is added
            // to both the pose and landmark diagonals before reduction so the
            // augmented system stays SPD when the un-damped one is rank-
            // deficient (as monocular BA generally is).
            let saved_poses = self.poses.clone();
            let saved_landmarks = self.landmarks.clone();
            let saved_velocities = self.velocities.clone();
            let saved_biases = self.biases.clone();
            let cost_before = current_cost;
            // Keep the lambda actually supplied to this linear solve.  The
            // public BaIterationStats lambda intentionally retains its
            // historical post-rejection semantics.
            let solve_lambda = lambda;
            log_process_memory("ba-before-solve-step");

            let solve_result = match backend {
                BaSolveBackend::Legacy => solve_step(
                    &mut system,
                    pose_index.len(),
                    landmark_index.len(),
                    velocity_index.len(),
                    bias_index.len(),
                    lambda,
                    config.linear_solver,
                    config.parallel,
                    &mut block_symbolic_cache,
                )
                .map(|(delta_poses, delta_landmarks)| (delta_poses, delta_landmarks, None, None)),
                BaSolveBackend::MatrixFree(runtime)
                | BaSolveBackend::MatrixFreeColumnScaled(runtime) => {
                    let column_scaled = runtime.column_scaling_iterations.is_some();
                    let scaling_result = if column_scaled {
                        match column_equilibrate_normal_system(&mut system) {
                            Ok(mut state) => {
                                state.stats.iteration = iteration;
                                runtime
                                    .column_scaling_iterations
                                    .as_mut()
                                    .expect("column scaling diagnostics are enabled")
                                    .push(state.stats);
                                Ok(Some(state))
                            }
                            Err(error) => Err(MatrixFreeStepError {
                                diagnostic: format!("column scaling failed: {error}"),
                                pcg_iterations: None,
                                pcg_residual_norm: None,
                                pcg_target: None,
                                restart_diagnostics: None,
                            }),
                        }
                    } else {
                        Ok(None)
                    };
                    let schur_debug_context = matrix_free_schur_debug_context(
                        &pose_index,
                        &landmark_index,
                        iteration,
                        column_scaled,
                    );
                    let solve_result = match scaling_result {
                        Err(error) => Err(error),
                        Ok(scaling_state) => solve_matrix_free_step(
                            &system,
                            lambda,
                            runtime.options,
                            runtime.restart_limit,
                            runtime.restart_iterations.is_some(),
                            scaling_state.as_ref(),
                            schur_debug_context,
                            adaptive_damping,
                        ),
                    };
                    match solve_result {
                        Ok(mut outcome) => {
                            outcome.diagnostics.iteration = iteration;
                            runtime.iterations.push(outcome.diagnostics);
                            if let Some(mut restart_diagnostics) = outcome.restart_diagnostics {
                                restart_diagnostics.iteration = iteration;
                                runtime
                                    .restart_iterations
                                    .as_mut()
                                    .expect("restart diagnostics are enabled")
                                    .push(restart_diagnostics);
                            }
                            Ok((
                                outcome.delta_poses,
                                outcome.delta_landmarks,
                                outcome.adaptive_prediction,
                                outcome.quality,
                            ))
                        }
                        Err(error) => {
                            runtime.iterations.push(MatrixFreeBaIterationStats {
                                iteration,
                                pcg_iterations: error.pcg_iterations,
                                pcg_residual_norm: error.pcg_residual_norm,
                                pcg_target: error.pcg_target,
                                pcg_failure: Some(error.diagnostic.clone()),
                            });
                            if let Some(mut restart_diagnostics) = error.restart_diagnostics {
                                restart_diagnostics.iteration = iteration;
                                runtime
                                    .restart_iterations
                                    .as_mut()
                                    .expect("restart diagnostics are enabled")
                                    .push(restart_diagnostics);
                            }
                            if ba_lm_step_quality_debug_enabled() {
                                emit_matrix_free_step_quality_failure(
                                    iteration,
                                    solve_lambda,
                                    &error.diagnostic,
                                );
                            }
                            runtime.failure = Some(MatrixFreeBaError::LinearSolve {
                                iteration,
                                diagnostic: error.diagnostic,
                            });
                            Err(BaError::SingularSystem)
                        }
                    }
                }
            };
            let (delta_poses, delta_landmarks, adaptive_prediction, quality) = match solve_result {
                Ok(d) => d,
                Err(BaError::SingularSystem) => {
                    // Treat singular system the same as a rejected LM step:
                    // bump λ and retry.
                    let next_lambda = if adaptive_damping {
                        bounded_adaptive_lambda(
                            lambda,
                            config.lambda_increase_factor,
                            config.min_lambda,
                            config.max_lambda,
                        )
                    } else {
                        (lambda * config.lambda_increase_factor).min(config.max_lambda)
                    };
                    if adaptive_damping {
                        if let BaSolveBackend::MatrixFreeColumnScaled(runtime) = &mut backend {
                            runtime
                                .adaptive_iterations
                                .as_mut()
                                .expect("adaptive damping diagnostics are enabled")
                                .push(MatrixFreeBaAdaptiveDampingIterationStats {
                                    iteration,
                                    solve_lambda,
                                    next_lambda,
                                    predicted_undamped_squared_decrease: None,
                                    actual_cost_decrease: None,
                                    rho: None,
                                    cost_gate: None,
                                    feasibility_gate: None,
                                    nonprojectable_before: current_nonprojectable,
                                    nonprojectable_after: None,
                                    accepted: false,
                                    reason: "linear_failure".to_owned(),
                                });
                        }
                    }
                    lambda = next_lambda;
                    iterations.push(BaIterationStats {
                        iteration,
                        cost_before,
                        cost_after: cost_before,
                        max_pose_step: 0.0,
                        max_landmark_step: 0.0,
                        lambda,
                        step_accepted: false,
                    });
                    if lambda >= config.max_lambda {
                        break;
                    }
                    continue;
                }
                Err(e) => return Err(e),
            };
            log_process_memory("ba-after-solve-step");

            // Apply tentative update. `delta_poses` packs pose slots
            // first (`i * 6 .. i * 6 + 6`), then velocity slots
            // (`6P + v * 3 .. 6P + v * 3 + 3`), then bias slots
            // (`6P + 3V + b * 6 .. 6P + 3V + b * 6 + 6`).
            let mut max_pose_step: f64 = 0.0;
            let vel_offset_in_delta = pose_index.len() * 6;
            let bias_offset_in_delta = vel_offset_in_delta + velocity_index.len() * 3;
            for (&id, &i) in &pose_index {
                let mut xi = delta_poses.fixed_rows::<6>(i * 6).into_owned();
                if self.fixed_pose_rotations.contains(&id) {
                    xi[3] = 0.0;
                    xi[4] = 0.0;
                    xi[5] = 0.0;
                }
                let xi_vec: Vector6<f64> = xi;
                let step = xi_vec.norm();
                if step > max_pose_step {
                    max_pose_step = step;
                }
                let pose = self.poses.get_mut(&id).expect("pose exists");
                pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&xi_vec));
            }
            for (&id, &v) in &velocity_index {
                let dv = delta_poses
                    .fixed_rows::<3>(vel_offset_in_delta + v * 3)
                    .into_owned();
                let dv_vec: Vector3<f64> = dv;
                let step = dv_vec.norm();
                if step > max_pose_step {
                    max_pose_step = step;
                }
                let v_ref = self.velocities.get_mut(&id).expect("velocity exists");
                *v_ref += dv_vec;
            }
            for (&id, &b) in &bias_index {
                let db = delta_poses
                    .fixed_rows::<6>(bias_offset_in_delta + b * 6)
                    .into_owned();
                let db_vec: Vector6<f64> = db;
                let step = db_vec.norm();
                if step > max_pose_step {
                    max_pose_step = step;
                }
                let b_ref = self.biases.get_mut(&id).expect("bias exists");
                *b_ref += db_vec;
            }
            let mut max_landmark_step: f64 = 0.0;
            for (&id, &i) in &landmark_index {
                let dx = delta_landmarks.fixed_rows::<3>(i * 3).into_owned();
                let v: Vector3<f64> = dx;
                let step = v.norm();
                if step > max_landmark_step {
                    max_landmark_step = step;
                }
                let pt = self.landmarks.get_mut(&id).expect("landmark exists");
                *pt = Point3::from(pt.coords + v);
            }

            let nonprojectable_before = current_nonprojectable;
            let cost_after = self.robust_cost_weighted(&kernel, gnc_weights);
            let nonprojectable_after = self.nonprojectable_observation_count();
            let cost_accepted = match config.initial_lambda {
                None => true, // Pure GN: accept unconditionally.
                Some(_) => cost_after < cost_before,
            };
            let feasibility_gate = nonprojectable_after <= current_nonprojectable;
            let mut step_accepted = cost_accepted && feasibility_gate;
            let adaptive_decision = if adaptive_damping {
                let decision = adaptive_step_decision(
                    adaptive_prediction,
                    cost_before,
                    cost_after,
                    nonprojectable_before,
                    nonprojectable_after,
                    cost_accepted,
                    nonprojectable_before == nonprojectable_after && nonprojectable_after == 0,
                );
                step_accepted &= decision.accepted;
                Some(decision)
            } else {
                None
            };

            if let Some(quality) = quality {
                emit_matrix_free_step_quality(
                    iteration,
                    solve_lambda,
                    quality,
                    cost_before,
                    cost_after,
                    nonprojectable_before,
                    nonprojectable_after,
                    cost_accepted,
                    nonprojectable_after <= nonprojectable_before,
                    step_accepted,
                );
            }

            // In particular, expose the two independent acceptance gates for
            // camera-fixed (landmark-only) solves.  A high robust cost can be
            // dominated by observations that are already down-weighted; a
            // candidate can also lower that cost while making more points
            // non-projectable, in which case the feasibility gate correctly
            // rejects it.  This line is diagnostic-only and is never emitted
            // unless both explicit BA step environment flags are set.
            if ba_step_debug_enabled() && (pose_index.is_empty() || !step_accepted) {
                eprintln!(
                    concat!(
                        "sfm-debug-ba-step-detail: poses={} landmarks={} iteration={} ",
                        "accepted={} cost_gate={} feasibility_gate={} ",
                        "nonprojectable={}->{} cost={:.9e}->{:.9e} lambda={:.3e}"
                    ),
                    pose_index.len(),
                    landmark_index.len(),
                    iteration,
                    step_accepted,
                    cost_accepted,
                    nonprojectable_after <= nonprojectable_before,
                    nonprojectable_before,
                    nonprojectable_after,
                    cost_before,
                    cost_after,
                    lambda,
                );
            }

            if !step_accepted {
                self.poses = saved_poses;
                self.landmarks = saved_landmarks;
                self.velocities = saved_velocities;
                self.biases = saved_biases;
                let next_lambda = if adaptive_damping {
                    bounded_adaptive_lambda(
                        lambda,
                        config.lambda_increase_factor,
                        config.min_lambda,
                        config.max_lambda,
                    )
                } else {
                    (lambda * config.lambda_increase_factor).min(config.max_lambda)
                };
                if let Some(decision) = adaptive_decision {
                    if let BaSolveBackend::MatrixFreeColumnScaled(runtime) = &mut backend {
                        runtime
                            .adaptive_iterations
                            .as_mut()
                            .expect("adaptive damping diagnostics are enabled")
                            .push(MatrixFreeBaAdaptiveDampingIterationStats {
                                iteration,
                                solve_lambda,
                                next_lambda,
                                predicted_undamped_squared_decrease: decision.prediction,
                                actual_cost_decrease: decision.actual_cost_decrease,
                                rho: decision.rho,
                                cost_gate: Some(decision.cost_gate),
                                feasibility_gate: Some(decision.feasibility_gate),
                                nonprojectable_before,
                                nonprojectable_after: Some(nonprojectable_after),
                                accepted: false,
                                reason: decision.reason,
                            });
                    }
                }
                lambda = next_lambda;
                iterations.push(BaIterationStats {
                    iteration,
                    cost_before,
                    cost_after,
                    max_pose_step,
                    max_landmark_step,
                    lambda,
                    step_accepted: false,
                });
                if config.initial_lambda.is_none() {
                    break;
                }
                if lambda >= config.max_lambda {
                    break;
                }
                continue;
            }

            iterations.push(BaIterationStats {
                iteration,
                cost_before,
                cost_after,
                max_pose_step,
                max_landmark_step,
                lambda,
                step_accepted: true,
            });
            current_cost = cost_after;
            current_nonprojectable = nonprojectable_after;
            if let Some(decision) = adaptive_decision {
                let next_lambda = adaptive_accepted_lambda(
                    lambda,
                    decision.rho.expect("accepted adaptive step has rho"),
                    config.min_lambda,
                    config.max_lambda,
                )
                .expect("accepted adaptive step has finite positive rho");
                if let BaSolveBackend::MatrixFreeColumnScaled(runtime) = &mut backend {
                    runtime
                        .adaptive_iterations
                        .as_mut()
                        .expect("adaptive damping diagnostics are enabled")
                        .push(MatrixFreeBaAdaptiveDampingIterationStats {
                            iteration,
                            solve_lambda,
                            next_lambda,
                            predicted_undamped_squared_decrease: decision.prediction,
                            actual_cost_decrease: decision.actual_cost_decrease,
                            rho: decision.rho,
                            cost_gate: Some(decision.cost_gate),
                            feasibility_gate: Some(decision.feasibility_gate),
                            nonprojectable_before,
                            nonprojectable_after: Some(nonprojectable_after),
                            accepted: true,
                            reason: decision.reason,
                        });
                }
                lambda = next_lambda;
            } else if config.initial_lambda.is_some() {
                lambda = (lambda * config.lambda_decrease_factor).max(config.min_lambda);
            }

            if max_pose_step < config.step_tolerance && max_landmark_step < config.step_tolerance {
                converged = true;
                break;
            }
            if (cost_before - cost_after).abs() < config.cost_tolerance {
                converged = true;
                break;
            }
            if config.relative_cost_tolerance.is_some_and(|tolerance| {
                tolerance.is_finite()
                    && tolerance >= 0.0
                    && (cost_before - cost_after) / cost_before.abs().max(f64::EPSILON) < tolerance
            }) {
                converged = true;
                break;
            }
        }

        Ok(BaResult {
            initial_cost,
            final_cost: current_cost,
            iterations,
            converged,
        })
    }
}

fn clear_visual_and_structural_costs(ba: &mut BundleAdjustment) {
    ba.observations.clear();
    ba.stereo_observations.clear();
    ba.general_stereo_observations.clear();
    ba.rig_observations.clear();
    ba.gravity_prior = None;
    ba.per_pose_gravity_prior = None;
    ba.position_prior = None;
    ba.pairwise_pose_factors.clear();
}

/// Configuration for [`BundleAdjustment::optimize`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaConfig {
    pub max_iterations: usize,
    pub initial_lambda: Option<f64>,
    pub lambda_increase_factor: f64,
    pub lambda_decrease_factor: f64,
    pub max_lambda: f64,
    pub min_lambda: f64,
    pub step_tolerance: f64,
    pub cost_tolerance: f64,
    /// Optional relative accepted-cost-decrease stopping threshold.
    ///
    /// An accepted iteration converges when
    /// `(cost_before - cost_after) / max(abs(cost_before), epsilon)` is below
    /// this value. `None` preserves the historical absolute-cost and step-only
    /// stopping rules.
    pub relative_cost_tolerance: Option<f64>,
    /// Linear-solver backend for the Schur-reduced pose system. The
    /// landmark elimination is always done analytically via per-landmark
    /// `3×3` block inversion (since `H_LL` is block-diagonal).
    pub linear_solver: LinearSolver,
    /// Robust IRLS kernel applied per-observation to its squared
    /// reprojection residual. [`RobustKernel::None`] runs standard
    /// non-robust BA; `Huber` / `Cauchy` down-weight outliers so a small
    /// number of bad correspondences cannot pull the solution away from
    /// the inlier consensus.
    pub robust_kernel: RobustKernel,
    /// Also refine the shared pinhole intrinsics `(fx, fy, cx, cy)` **jointly**:
    /// when set, [`BundleAdjustment::optimize`] carries the 4 intrinsics as extra
    /// unknowns inside the Schur-complement camera system, co-estimated with the
    /// poses and (eliminated) landmarks — the COLMAP self-calibration formulation.
    /// This is the lever for unknown / inaccurate calibration: a wrong fixed focal
    /// forces a residual onto the poses, and the joint solve lets the camera absorb
    /// it. (The coupled, landmark-eliminated gradient is what makes this work; an
    /// alternating refinement against converged structure cannot move a wrong focal,
    /// because the structure-fixed gradient is ~0.) Only the 4-parameter
    /// [`CameraModel::Pinhole`] is refined; any other model falls back to the
    /// pose/structure-only solve. **`false` by default** (the public
    /// [`BundleAdjustment::optimize`] is then bit-identical to before).
    pub refine_intrinsics: bool,
    /// Additionally self-calibrate the two radial-distortion coefficients
    /// `(k1, k2)` jointly with the intrinsics (the camera block grows from 4 to 6).
    /// Requires `refine_intrinsics`; only applies to a **monocular** pinhole
    /// reconstruction (rectified stereo is already undistorted). The coefficients
    /// are appended to `Camera::params` as `[fx, fy, cx, cy, k1, k2]`. **`false`
    /// by default.**
    pub refine_distortion: bool,
    /// Run the per-observation assembly, per-landmark Schur reduction, and
    /// back-substitution loops of [`BundleAdjustment::optimize_weighted`]'s
    /// Levenberg-Marquardt iteration on the `rayon` pool (see the module's
    /// "Parallelism" section). The result is bit-identical to the serial
    /// path at any thread count — this only changes *how* the normal
    /// equations are computed, never the summation order — so it is safe to
    /// flip independently of everything else in this config. Small problems
    /// stay serial even when this is set (see `PARALLEL_MIN_OBSERVATIONS` /
    /// `PARALLEL_MIN_LANDMARKS`). Only consumed by `optimize_weighted`
    /// (the plain pose/structure/IMU solve); `optimize_joint_intrinsics`
    /// ignores it. **`false` by default** (the public
    /// [`BundleAdjustment::optimize`] is then bit-identical to before).
    pub parallel: bool,
}

impl Default for BaConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            initial_lambda: Some(1e-4),
            lambda_increase_factor: 10.0,
            lambda_decrease_factor: 0.1,
            max_lambda: 1e12,
            min_lambda: 1e-9,
            step_tolerance: 1e-7,
            cost_tolerance: 1e-9,
            relative_cost_tolerance: None,
            linear_solver: LinearSolver::Dense,
            robust_kernel: RobustKernel::None,
            refine_intrinsics: false,
            refine_distortion: false,
            parallel: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BaIterationStats {
    pub iteration: usize,
    pub cost_before: f64,
    pub cost_after: f64,
    pub max_pose_step: f64,
    pub max_landmark_step: f64,
    pub lambda: f64,
    pub step_accepted: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BaResult {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: Vec<BaIterationStats>,
    pub converged: bool,
}

/// Options for the opt-in matrix-free pure-visual BA entry point.
///
/// This is deliberately separate from [`BaConfig`]: adding a solver variant
/// there would change the default path and would make existing callers
/// accidentally opt into a new numerical backend.  The matrix-free entry
/// point requires a finite, positive LM start value and uses PCG for each
/// reduced pose solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixFreeBaOptions {
    /// Maximum PCG iterations for one LM linear solve.
    pub max_pcg_iterations: usize,
    /// Relative PCG residual tolerance.
    pub pcg_relative_tolerance: f64,
    /// Absolute PCG residual tolerance.
    pub pcg_absolute_tolerance: f64,
}

impl Default for MatrixFreeBaOptions {
    fn default() -> Self {
        Self {
            max_pcg_iterations: 128,
            pcg_relative_tolerance: 1.0e-12,
            pcg_absolute_tolerance: 1.0e-12,
        }
    }
}

/// Options for the opt-in column-equilibrated matrix-free entry point.
///
/// The diagonal bounds are deliberately private fixed policy constants.  The
/// nested PCG options reuse the existing matrix-free option shape without
/// changing its defaults or adding a scaling switch to the legacy API.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MatrixFreeBaColumnScalingOptions {
    pub pcg: MatrixFreeBaOptions,
}

/// Per-LM-iteration diagnostics returned by [`BundleAdjustment::optimize_matrix_free`].
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaIterationStats {
    pub iteration: usize,
    pub pcg_iterations: Option<usize>,
    pub pcg_residual_norm: Option<f64>,
    pub pcg_target: Option<f64>,
    pub pcg_failure: Option<String>,
}

/// Result of the opt-in matrix-free BA run.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaResult {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: Vec<BaIterationStats>,
    pub matrix_free_iterations: Vec<MatrixFreeBaIterationStats>,
    pub converged: bool,
}

/// Scalar accounting for one normal-system equilibration.  PCG residuals in
/// the nested [`MatrixFreeBaResult`] are in scaled coordinates; physical pose
/// and landmark step norms remain in its ordinary LM iteration statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixFreeBaColumnScalingIterationStats {
    pub iteration: usize,
    /// Minimum of the clamped normal-equation diagonal values `d_j`, not of
    /// the transforms `1/sqrt(d_j)`.
    pub minimum_diagonal: f64,
    /// Maximum of the clamped normal-equation diagonal values `d_j`, not of
    /// the transforms `1/sqrt(d_j)`.
    pub maximum_diagonal: f64,
    pub clamped_to_minimum: usize,
    pub clamped_to_maximum: usize,
}

/// Result of the opt-in column-equilibrated matrix-free BA run.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaColumnScalingResult {
    pub ba: MatrixFreeBaResult,
    pub scaling_iterations: Vec<MatrixFreeBaColumnScalingIterationStats>,
}

/// Scalar per-LM-iteration accounting for adaptive column-scaled damping.
/// Prediction is evaluated in the current scaled coordinates; it is not a
/// full backward-error diagnostic and does not retain normal-equation state.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaAdaptiveDampingIterationStats {
    pub iteration: usize,
    pub solve_lambda: f64,
    pub next_lambda: f64,
    pub predicted_undamped_squared_decrease: Option<f64>,
    pub actual_cost_decrease: Option<f64>,
    pub rho: Option<f64>,
    pub cost_gate: Option<bool>,
    pub feasibility_gate: Option<bool>,
    pub nonprojectable_before: usize,
    pub nonprojectable_after: Option<usize>,
    pub accepted: bool,
    pub reason: String,
}

/// Result of the opt-in adaptive column-scaled matrix-free BA run.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaAdaptiveDampingResult {
    pub ba: MatrixFreeBaResult,
    pub scaling_iterations: Vec<MatrixFreeBaColumnScalingIterationStats>,
    pub adaptive_iterations: Vec<MatrixFreeBaAdaptiveDampingIterationStats>,
}

/// Additive options for the bounded true-residual restart diagnostic.
///
/// `max_restarts_per_solve` is intentionally limited to zero or one.  The
/// PCG iteration budget in [`MatrixFreeBaOptions`] is global to each reduced
/// solve and is never reset by a restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatrixFreeBaRestartOptions {
    pub max_restarts_per_solve: usize,
}

/// Per-LM-iteration diagnostics for the bounded restart entry point.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaRestartIterationStats {
    pub iteration: usize,
    /// Total alpha iterations used by this PCG solve.  This value is never
    /// reset when a residual restart occurs.
    pub pcg_iterations: Option<usize>,
    /// Number of explicit true-residual checks performed.
    pub true_residual_rechecks: usize,
    /// Number of those checks whose norm exceeded the configured target.  This
    /// includes checks at the iteration cap; it is not a count of
    /// restart-eligible checks.
    pub failed_true_residual_rechecks: usize,
    pub restarts: usize,
    pub terminal_failure: Option<String>,
}

/// Result of [`BundleAdjustment::optimize_matrix_free_with_restart`].
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaRestartResult {
    pub ba: MatrixFreeBaResult,
    pub restart_iterations: Vec<MatrixFreeBaRestartIterationStats>,
}

/// Failure from the opt-in matrix-free pure-visual BA entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixFreeBaError {
    /// The problem contains a factor/state that the pure-visual operator does
    /// not assemble (for example IMU, navigation, or a non-visual prior).
    Ineligible(&'static str),
    /// The matrix-free LM/PCG options are not finite or cannot make progress.
    InvalidConfiguration(&'static str),
    /// The existing BA input validation rejected the problem.
    Ba(BaError),
    /// The reduced operator or PCG solve failed.  The textual payload is a
    /// deterministic diagnostic representation of the private numerical error.
    LinearSolve {
        iteration: usize,
        diagnostic: String,
    },
}

impl std::fmt::Display for MatrixFreeBaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ineligible(reason) => write!(f, "matrix-free BA is ineligible: {reason}"),
            Self::InvalidConfiguration(reason) => {
                write!(f, "invalid matrix-free BA configuration: {reason}")
            }
            Self::Ba(error) => write!(f, "matrix-free BA input error: {error}"),
            Self::LinearSolve {
                iteration,
                diagnostic,
            } => write!(
                f,
                "matrix-free reduced solve failed at iteration {iteration}: {diagnostic}"
            ),
        }
    }
}

impl std::error::Error for MatrixFreeBaError {}

const COLUMN_SCALING_MIN_DIAGONAL: f64 = 1.0e-6;
const COLUMN_SCALING_MAX_DIAGONAL: f64 = 1.0e32;

#[derive(Debug)]
struct ColumnEquilibrationState {
    pose_transforms: Vec<Vector6<f64>>,
    landmark_transforms: Vec<Vector3<f64>>,
    stats: MatrixFreeBaColumnScalingIterationStats,
}

impl ColumnEquilibrationState {
    fn unscale_deltas(
        &self,
        delta_poses: &mut DVector<f64>,
        delta_landmarks: &mut DVector<f64>,
    ) -> Result<(), &'static str> {
        let expected_pose = self.pose_transforms.len() * 6;
        let expected_landmark = self.landmark_transforms.len() * 3;
        if delta_poses.len() != expected_pose || delta_landmarks.len() != expected_landmark {
            return Err("scaled delta dimensions do not match the normal system");
        }
        for (pose, transform) in self.pose_transforms.iter().enumerate() {
            for component in 0..6 {
                let index = pose * 6 + component;
                let value = delta_poses[index] * transform[component];
                if !value.is_finite() {
                    return Err("unscaled pose delta is non-finite");
                }
                delta_poses[index] = value;
            }
        }
        for (landmark, transform) in self.landmark_transforms.iter().enumerate() {
            for component in 0..3 {
                let index = landmark * 3 + component;
                let value = delta_landmarks[index] * transform[component];
                if !value.is_finite() {
                    return Err("unscaled landmark delta is non-finite");
                }
                delta_landmarks[index] = value;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MatrixFreeCoordinateQuality {
    /// The undamped squared-cost quadratic prediction in this coordinate
    /// system: `-2 b·δ - δᵀHδ`.  For a scaled solve this is computed from the
    /// rounded scaled normal system and is not an independent reconstruction
    /// of the pre-scaling normal system.
    predicted_undamped_squared_decrease: f64,
    /// Normwise backward error in the coordinate system represented by this
    /// report.  The denominator is `||A||F ||δ||2 + ||b||2` for the damped
    /// full pose+landmark system.
    normwise_backward_error: f64,
    /// Componentwise backward error `max_i |r_i| / (|A||δ|+|b|)_i`.
    componentwise_backward_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MatrixFreeStepQuality {
    scaled_coordinates: bool,
    coordinate: Option<MatrixFreeCoordinateQuality>,
    physical_equivalent: Option<MatrixFreeCoordinateQuality>,
    coordinate_failure: Option<&'static str>,
    physical_equivalent_failure: Option<&'static str>,
}

#[derive(Clone, Copy)]
enum MatrixFreeQualityCoordinate<'a> {
    /// The coordinates currently stored in `system` and supplied to the
    /// matrix-free solve.  This is physical for legacy MF and scaled for the
    /// column-equilibrated path.
    Current,
    /// The physical-equivalent rounded system obtained from a scaled system
    /// by applying `T⁻¹` to rows and columns.  This is intentionally not
    /// called the original normal system: the in-place scaling arithmetic has
    /// already rounded its coefficients.
    PhysicalEquivalent(&'a ColumnEquilibrationState),
}

fn quality_add(total: &mut f64, value: f64) -> Result<(), &'static str> {
    if !value.is_finite() {
        return Err("quality diagnostic encountered a non-finite term");
    }
    *total += value;
    if total.is_finite() {
        Ok(())
    } else {
        Err("quality diagnostic accumulator overflowed")
    }
}

fn quality_add_square(total: &mut f64, value: f64) -> Result<(), &'static str> {
    if !value.is_finite() {
        return Err("quality diagnostic encountered a non-finite value");
    }
    let square = value * value;
    if !square.is_finite() {
        return Err("quality diagnostic square overflowed");
    }
    quality_add(total, square)
}

fn quality_fold_residual_row(
    residual_norm_squared: &mut f64,
    componentwise_backward_error: &mut f64,
    residual: f64,
    denominator: f64,
) -> Result<(), &'static str> {
    if !residual.is_finite() || !denominator.is_finite() {
        return Err("quality diagnostic residual or denominator is non-finite");
    }
    quality_add_square(residual_norm_squared, residual)?;
    if denominator == 0.0 {
        if residual != 0.0 {
            return Err("quality diagnostic has nonzero residual over zero denominator");
        }
        // Zero-over-zero rows are exact zero rows (commonly fixed rotations)
        // and contribute zero to componentwise eta.
    } else {
        let ratio = residual.abs() / denominator;
        if !ratio.is_finite() {
            return Err("quality diagnostic componentwise ratio is non-finite");
        }
        *componentwise_backward_error = (*componentwise_backward_error).max(ratio);
    }
    Ok(())
}

fn quality_pose_factors(
    coordinate: MatrixFreeQualityCoordinate<'_>,
    pose: usize,
    component: usize,
) -> Result<(f64, f64), &'static str> {
    match coordinate {
        MatrixFreeQualityCoordinate::Current => Ok((1.0, 1.0)),
        MatrixFreeQualityCoordinate::PhysicalEquivalent(state) => {
            if component >= 6 {
                return Err("quality diagnostic pose transform component is invalid");
            }
            let transform = state
                .pose_transforms
                .get(pose)
                .ok_or("quality diagnostic pose transform index is invalid")?[component];
            let inverse = transform.recip();
            if !transform.is_finite() || !inverse.is_finite() {
                return Err("quality diagnostic pose transform is non-finite");
            }
            // First factor scales rows/columns of the current system.  The
            // second factor maps the current delta into the target system.
            Ok((inverse, transform))
        }
    }
}

fn quality_landmark_factors(
    coordinate: MatrixFreeQualityCoordinate<'_>,
    landmark: usize,
    component: usize,
) -> Result<(f64, f64), &'static str> {
    match coordinate {
        MatrixFreeQualityCoordinate::Current => Ok((1.0, 1.0)),
        MatrixFreeQualityCoordinate::PhysicalEquivalent(state) => {
            if component >= 3 {
                return Err("quality diagnostic landmark transform component is invalid");
            }
            let transform = state
                .landmark_transforms
                .get(landmark)
                .ok_or("quality diagnostic landmark transform index is invalid")?[component];
            let inverse = transform.recip();
            if !transform.is_finite() || !inverse.is_finite() {
                return Err("quality diagnostic landmark transform is non-finite");
            }
            Ok((inverse, transform))
        }
    }
}

fn quality_coordinate_metrics(
    system: &NormalEquationsBa,
    lambda: f64,
    delta_poses: &DVector<f64>,
    delta_landmarks: &DVector<f64>,
    coordinate: MatrixFreeQualityCoordinate<'_>,
) -> Result<MatrixFreeCoordinateQuality, &'static str> {
    if !lambda.is_finite() {
        return Err("quality diagnostic lambda is non-finite");
    }
    let CameraHessian::PoseDiagonal(pose_blocks) = &system.h_pp else {
        return Err("quality diagnostic requires pose-diagonal normal equations");
    };
    let pose_count = pose_blocks.len();
    let landmark_count = system.landmarks.len();
    if system.b_p.len() != pose_count * 6
        || delta_poses.len() != pose_count * 6
        || delta_landmarks.len() != landmark_count * 3
    {
        return Err("quality diagnostic delta dimensions mismatch");
    }

    // These are O(P) scratch vectors; landmark scratch is streamed and cross
    // grouping is bounded by the maximum track length. Cross terms enter
    // these vectors after same-pose entries are coalesced per landmark, so the
    // componentwise denominator represents the actual matrix rather than a
    // sum of absolute values of duplicate stored entries.
    let mut pose_residual = vec![0.0; pose_count * 6];
    let mut pose_denominator = vec![0.0; pose_count * 6];
    let mut matrix_norm_squared = 0.0;
    let mut delta_norm_squared = 0.0;
    let mut rhs_norm_squared = 0.0;
    let mut gradient_dot = 0.0;
    let mut hessian_quadratic = 0.0;
    let mut residual_norm_squared = 0.0;
    let mut componentwise_backward_error: f64 = 0.0;

    for (pose, block) in pose_blocks.iter().enumerate() {
        for row in 0..6 {
            let (row_scale, row_delta_factor) = quality_pose_factors(coordinate, pose, row)?;
            let row_index = pose * 6 + row;
            let delta_row = delta_poses[row_index] * row_delta_factor;
            let rhs_row = system.b_p[row_index] * row_scale;
            if !delta_row.is_finite() || !rhs_row.is_finite() {
                return Err("quality diagnostic pose delta or rhs is non-finite");
            }
            pose_residual[row_index] = rhs_row;
            pose_denominator[row_index] = rhs_row.abs();
            quality_add_square(&mut delta_norm_squared, delta_row)?;
            quality_add_square(&mut rhs_norm_squared, rhs_row)?;
            quality_add(&mut gradient_dot, rhs_row * delta_row)?;

            for column in 0..6 {
                let (column_scale, column_delta_factor) =
                    quality_pose_factors(coordinate, pose, column)?;
                let column_index = pose * 6 + column;
                let delta_column = delta_poses[column_index] * column_delta_factor;
                let h = block[(row, column)] * row_scale * column_scale;
                let a = h + if row == column {
                    lambda * row_scale * column_scale
                } else {
                    0.0
                };
                if !delta_column.is_finite() || !h.is_finite() || !a.is_finite() {
                    return Err("quality diagnostic pose system is non-finite");
                }
                quality_add_square(&mut matrix_norm_squared, a)?;
                quality_add(&mut pose_residual[row_index], a * delta_column)?;
                quality_add(
                    &mut pose_denominator[row_index],
                    a.abs() * delta_column.abs(),
                )?;
                quality_add(&mut hessian_quadratic, delta_row * h * delta_column)?;
            }
        }
    }

    for (landmark_index, landmark) in system.landmarks.iter().enumerate() {
        // Landmark rows have no coupling to other landmarks, so keep their
        // residual/denominator as a single-landmark scratch array and fold
        // them into the scalar metrics before moving to the next landmark.
        let mut landmark_residual = [0.0; 3];
        let mut landmark_denominator = [0.0; 3];
        for row in 0..3 {
            let (row_scale, row_delta_factor) =
                quality_landmark_factors(coordinate, landmark_index, row)?;
            let row_index = landmark_index * 3 + row;
            let delta_row = delta_landmarks[row_index] * row_delta_factor;
            let rhs_row = landmark.b_l[row] * row_scale;
            if !delta_row.is_finite() || !rhs_row.is_finite() {
                return Err("quality diagnostic landmark delta or rhs is non-finite");
            }
            landmark_residual[row] = rhs_row;
            landmark_denominator[row] = rhs_row.abs();
            quality_add_square(&mut delta_norm_squared, delta_row)?;
            quality_add_square(&mut rhs_norm_squared, rhs_row)?;
            quality_add(&mut gradient_dot, rhs_row * delta_row)?;

            for column in 0..3 {
                let (column_scale, column_delta_factor) =
                    quality_landmark_factors(coordinate, landmark_index, column)?;
                let column_index = landmark_index * 3 + column;
                let delta_column = delta_landmarks[column_index] * column_delta_factor;
                let h = landmark.h_ll[(row, column)] * row_scale * column_scale;
                let a = h + if row == column {
                    lambda * row_scale * column_scale
                } else {
                    0.0
                };
                if !delta_column.is_finite() || !h.is_finite() || !a.is_finite() {
                    return Err("quality diagnostic landmark system is non-finite");
                }
                quality_add_square(&mut matrix_norm_squared, a)?;
                quality_add(&mut landmark_residual[row], a * delta_column)?;
                quality_add(&mut landmark_denominator[row], a.abs() * delta_column.abs())?;
                quality_add(&mut hessian_quadratic, delta_row * h * delta_column)?;
            }
        }

        // The storage can contain multiple sensor contributions for the same
        // pose.  Coalesce each pose before taking absolute values or norms.
        let mut grouped_cross: BTreeMap<usize, Matrix6x3<f64>> = BTreeMap::new();
        for (pose, cross) in &landmark.cross {
            if *pose >= pose_count {
                return Err("quality diagnostic cross pose index is invalid");
            }
            grouped_cross
                .entry(*pose)
                .and_modify(|accumulated| *accumulated += *cross)
                .or_insert(*cross);
        }
        for (pose, cross) in grouped_cross {
            for row in 0..6 {
                let (pose_scale, pose_delta_factor) = quality_pose_factors(coordinate, pose, row)?;
                let pose_index = pose * 6 + row;
                let pose_delta = delta_poses[pose_index] * pose_delta_factor;
                for column in 0..3 {
                    let (landmark_scale, landmark_delta_factor) =
                        quality_landmark_factors(coordinate, landmark_index, column)?;
                    let landmark_index_flat = landmark_index * 3 + column;
                    let landmark_delta =
                        delta_landmarks[landmark_index_flat] * landmark_delta_factor;
                    let g = cross[(row, column)] * pose_scale * landmark_scale;
                    if !pose_delta.is_finite() || !landmark_delta.is_finite() || !g.is_finite() {
                        return Err("quality diagnostic cross system is non-finite");
                    }
                    quality_add_square(&mut matrix_norm_squared, g)?;
                    quality_add_square(&mut matrix_norm_squared, g)?;
                    quality_add(&mut pose_residual[pose_index], g * landmark_delta)?;
                    quality_add(&mut landmark_residual[column], g * pose_delta)?;
                    quality_add(
                        &mut pose_denominator[pose_index],
                        g.abs() * landmark_delta.abs(),
                    )?;
                    quality_add(
                        &mut landmark_denominator[column],
                        g.abs() * pose_delta.abs(),
                    )?;
                    // The symmetric cross block appears twice in δᵀHδ.
                    quality_add(
                        &mut hessian_quadratic,
                        2.0 * pose_delta * g * landmark_delta,
                    )?;
                }
            }
        }

        for row in 0..3 {
            quality_fold_residual_row(
                &mut residual_norm_squared,
                &mut componentwise_backward_error,
                landmark_residual[row],
                landmark_denominator[row],
            )?;
        }
    }

    for (residual, denominator) in pose_residual.iter().zip(pose_denominator.iter()) {
        quality_fold_residual_row(
            &mut residual_norm_squared,
            &mut componentwise_backward_error,
            *residual,
            *denominator,
        )?;
    }
    let residual_norm = residual_norm_squared.sqrt();
    let matrix_norm = matrix_norm_squared.sqrt();
    let delta_norm = delta_norm_squared.sqrt();
    let rhs_norm = rhs_norm_squared.sqrt();
    let normwise_denominator = matrix_norm * delta_norm + rhs_norm;
    if !residual_norm.is_finite()
        || !matrix_norm.is_finite()
        || !delta_norm.is_finite()
        || !rhs_norm.is_finite()
        || !normwise_denominator.is_finite()
    {
        return Err("quality diagnostic normwise quantity is non-finite");
    }
    let normwise_backward_error = if normwise_denominator == 0.0 {
        return Err("quality diagnostic normwise denominator is zero");
    } else {
        residual_norm / normwise_denominator
    };
    let predicted_undamped_squared_decrease = -2.0 * gradient_dot - hessian_quadratic;
    if !predicted_undamped_squared_decrease.is_finite()
        || !normwise_backward_error.is_finite()
        || !componentwise_backward_error.is_finite()
    {
        return Err("quality diagnostic result is non-finite");
    }
    Ok(MatrixFreeCoordinateQuality {
        predicted_undamped_squared_decrease,
        normwise_backward_error,
        componentwise_backward_error,
    })
}

/// Compute only the scalar undamped quadratic prediction needed by the
/// adaptive damping policy.  This intentionally does not allocate the
/// residual/denominator scratch used by the opt-in backward-error diagnostic.
/// Same-pose cross blocks are coalesced in insertion order per landmark so
/// duplicate rig-sensor contributions have one bounded local representation.
fn matrix_free_undamped_prediction(
    system: &NormalEquationsBa,
    delta_poses: &DVector<f64>,
    delta_landmarks: &DVector<f64>,
) -> Result<f64, &'static str> {
    let CameraHessian::PoseDiagonal(pose_blocks) = &system.h_pp else {
        return Err("adaptive prediction requires pose-diagonal normal equations");
    };
    if system.b_p.len() != pose_blocks.len() * 6
        || delta_poses.len() != pose_blocks.len() * 6
        || delta_landmarks.len() != system.landmarks.len() * 3
    {
        return Err("adaptive prediction delta dimensions mismatch");
    }

    let mut gradient_dot = 0.0;
    let mut hessian_quadratic = 0.0;
    for (pose, block) in pose_blocks.iter().enumerate() {
        for row in 0..6 {
            let row_index = pose * 6 + row;
            let delta_row = delta_poses[row_index];
            let rhs_row = system.b_p[row_index];
            quality_add(&mut gradient_dot, rhs_row * delta_row)?;
            for column in 0..6 {
                let delta_column = delta_poses[pose * 6 + column];
                quality_add(
                    &mut hessian_quadratic,
                    delta_row * block[(row, column)] * delta_column,
                )?;
            }
        }
    }

    for (landmark_index, landmark) in system.landmarks.iter().enumerate() {
        for row in 0..3 {
            let row_index = landmark_index * 3 + row;
            let delta_row = delta_landmarks[row_index];
            let rhs_row = landmark.b_l[row];
            quality_add(&mut gradient_dot, rhs_row * delta_row)?;
            for column in 0..3 {
                let delta_column = delta_landmarks[landmark_index * 3 + column];
                quality_add(
                    &mut hessian_quadratic,
                    delta_row * landmark.h_ll[(row, column)] * delta_column,
                )?;
            }
        }

        let mut grouped_cross: BTreeMap<usize, Matrix6x3<f64>> = BTreeMap::new();
        for (pose, cross) in &landmark.cross {
            if *pose >= pose_blocks.len() {
                return Err("adaptive prediction cross pose index is invalid");
            }
            if let Some(accumulated) = grouped_cross.get_mut(pose) {
                *accumulated += *cross;
                if !accumulated.iter().all(|value| value.is_finite()) {
                    return Err("adaptive prediction cross accumulation is non-finite");
                }
            } else {
                grouped_cross.insert(*pose, *cross);
            }
        }
        for (pose, cross) in grouped_cross {
            for row in 0..6 {
                let pose_delta = delta_poses[pose * 6 + row];
                for column in 0..3 {
                    let landmark_delta = delta_landmarks[landmark_index * 3 + column];
                    // The symmetric pose/landmark block occurs twice in
                    // delta^T H delta.
                    quality_add(
                        &mut hessian_quadratic,
                        2.0 * pose_delta * cross[(row, column)] * landmark_delta,
                    )?;
                }
            }
        }
    }

    let prediction = -2.0 * gradient_dot - hessian_quadratic;
    if prediction.is_finite() {
        Ok(prediction)
    } else {
        Err("adaptive prediction is non-finite")
    }
}

fn matrix_free_step_quality(
    system: &NormalEquationsBa,
    lambda: f64,
    delta_poses: &DVector<f64>,
    delta_landmarks: &DVector<f64>,
    scaling_state: Option<&ColumnEquilibrationState>,
) -> MatrixFreeStepQuality {
    let scaled_coordinates = scaling_state.is_some();
    let coordinate_result = quality_coordinate_metrics(
        system,
        lambda,
        delta_poses,
        delta_landmarks,
        MatrixFreeQualityCoordinate::Current,
    );
    let coordinate = coordinate_result.as_ref().ok().copied();
    let coordinate_failure = coordinate_result.as_ref().err().copied();
    let physical_result = scaling_state.map(|state| {
        quality_coordinate_metrics(
            system,
            lambda,
            delta_poses,
            delta_landmarks,
            MatrixFreeQualityCoordinate::PhysicalEquivalent(state),
        )
    });
    let physical_equivalent = match physical_result.as_ref() {
        Some(result) => result.as_ref().ok().copied(),
        None => coordinate,
    };
    let physical_equivalent_failure = physical_result
        .as_ref()
        .and_then(|result| result.as_ref().err().copied())
        .or_else(|| {
            if scaling_state.is_none() {
                coordinate_failure
            } else {
                None
            }
        });
    MatrixFreeStepQuality {
        scaled_coordinates,
        coordinate,
        physical_equivalent,
        coordinate_failure,
        physical_equivalent_failure,
    }
}

fn matrix_free_quality_actual_and_rho(
    cost_before: f64,
    cost_after: f64,
    coordinate_prediction: Option<f64>,
    nonprojectable_before: usize,
    nonprojectable_after: usize,
) -> (Option<f64>, Option<f64>, &'static str) {
    let actual_cost_decrease = if cost_before.is_finite() && cost_after.is_finite() {
        let decrease = cost_before - cost_after;
        decrease.is_finite().then_some(decrease)
    } else {
        None
    };
    let (rho, rho_reason) = if nonprojectable_before != 0 || nonprojectable_after != 0 {
        (None, "nonprojectable_count_nonzero")
    } else if actual_cost_decrease.is_none() {
        (None, "actual_cost_decrease_nonfinite")
    } else {
        match coordinate_prediction {
            Some(prediction) if prediction.is_finite() && prediction > 0.0 => {
                let rho = actual_cost_decrease.expect("finite actual decrease") / prediction;
                if rho.is_finite() {
                    (Some(rho), "defined")
                } else {
                    (None, "rho_nonfinite")
                }
            }
            Some(_) => (None, "prediction_nonpositive_or_nonfinite"),
            None => (None, "prediction_unavailable"),
        }
    };
    (actual_cost_decrease, rho, rho_reason)
}

#[allow(clippy::too_many_arguments)]
fn emit_matrix_free_step_quality(
    iteration: usize,
    solve_lambda: f64,
    quality: MatrixFreeStepQuality,
    cost_before: f64,
    cost_after: f64,
    nonprojectable_before: usize,
    nonprojectable_after: usize,
    cost_gate: bool,
    feasibility_gate: bool,
    step_accepted: bool,
) {
    let coordinate_prediction = quality
        .coordinate
        .map(|metrics| metrics.predicted_undamped_squared_decrease);
    let physical_prediction = quality
        .physical_equivalent
        .map(|metrics| metrics.predicted_undamped_squared_decrease);
    let (actual_cost_decrease, rho, rho_reason) = matrix_free_quality_actual_and_rho(
        cost_before,
        cost_after,
        coordinate_prediction,
        nonprojectable_before,
        nonprojectable_after,
    );
    let coordinate_name = if quality.scaled_coordinates {
        "scaled"
    } else {
        "physical"
    };
    let physical_equivalent_evaluation = if quality.scaled_coordinates {
        "recomputed_from_rounded_scaled_coefficients"
    } else {
        "current_physical_system"
    };
    let coordinate_normwise = quality
        .coordinate
        .map(|metrics| metrics.normwise_backward_error);
    let coordinate_componentwise = quality
        .coordinate
        .map(|metrics| metrics.componentwise_backward_error);
    let physical_normwise = quality
        .physical_equivalent
        .map(|metrics| metrics.normwise_backward_error);
    let physical_componentwise = quality
        .physical_equivalent
        .map(|metrics| metrics.componentwise_backward_error);
    eprintln!(
        concat!(
            "sfm-debug-ba-lm-quality: iteration={} solve_lambda={:.17e} ",
            "coordinates={} prediction_coordinates={} ",
            "physical_equivalent_evaluation={} ",
            "predicted_undamped_squared_decrease={:?} ",
            "physical_equivalent_predicted_undamped_squared_decrease={:?} ",
            "actual_cost_decrease={:?} rho={:?} rho_reason={} ",
            "rho_prediction_coordinates={} actual_cost_scope=existing_ba_cost ",
            "rho_cost_comparable={} nonprojectable_before={} nonprojectable_after={} ",
            "cost_gate={} feasibility_gate={} accepted={} ",
            "coordinate_full_normwise_backward_error={:?} ",
            "coordinate_full_componentwise_eta={:?} ",
            "physical_equivalent_full_normwise_backward_error={:?} ",
            "physical_equivalent_full_componentwise_eta={:?} ",
            "coordinate_quality_failure={:?} physical_equivalent_quality_failure={:?}"
        ),
        iteration,
        solve_lambda,
        coordinate_name,
        coordinate_name,
        physical_equivalent_evaluation,
        coordinate_prediction,
        physical_prediction,
        actual_cost_decrease,
        rho,
        rho_reason,
        coordinate_name,
        nonprojectable_before == 0 && nonprojectable_after == 0,
        nonprojectable_before,
        nonprojectable_after,
        cost_gate,
        feasibility_gate,
        step_accepted,
        coordinate_normwise,
        coordinate_componentwise,
        physical_normwise,
        physical_componentwise,
        quality.coordinate_failure,
        quality.physical_equivalent_failure,
    );
}

fn emit_matrix_free_step_quality_failure(iteration: usize, solve_lambda: f64, diagnostic: &str) {
    eprintln!(
        "sfm-debug-ba-lm-quality: iteration={} solve_lambda={:.17e} linear_failure={}",
        iteration, solve_lambda, diagnostic
    );
}

fn column_scaling_transform(
    diagonal: f64,
    minimum: &mut f64,
    maximum: &mut f64,
    clamped_to_minimum: &mut usize,
    clamped_to_maximum: &mut usize,
) -> Result<f64, &'static str> {
    if !diagonal.is_finite() {
        return Err("normal diagonal is non-finite");
    }
    if diagonal < 0.0 {
        return Err("normal diagonal is negative");
    }
    let clamped = diagonal.clamp(COLUMN_SCALING_MIN_DIAGONAL, COLUMN_SCALING_MAX_DIAGONAL);
    if clamped == COLUMN_SCALING_MIN_DIAGONAL && diagonal < COLUMN_SCALING_MIN_DIAGONAL {
        *clamped_to_minimum += 1;
    }
    if clamped == COLUMN_SCALING_MAX_DIAGONAL && diagonal > COLUMN_SCALING_MAX_DIAGONAL {
        *clamped_to_maximum += 1;
    }
    *minimum = minimum.min(clamped);
    *maximum = maximum.max(clamped);
    let transform = clamped.sqrt().recip();
    if !transform.is_finite() {
        return Err("column scaling transform is non-finite");
    }
    Ok(transform)
}

fn column_equilibrate_normal_system(
    system: &mut NormalEquationsBa,
) -> Result<ColumnEquilibrationState, &'static str> {
    let pose_transforms = match &system.h_pp {
        CameraHessian::PoseDiagonal(blocks) => {
            if blocks.is_empty() || system.b_p.len() != blocks.len() * 6 {
                return Err("pose diagonal dimensions are invalid");
            }
            let mut transforms = Vec::with_capacity(blocks.len());
            for block in blocks {
                let mut transform = Vector6::zeros();
                for component in 0..6 {
                    // The actual aggregate statistics are computed below in
                    // one pass over both pose and landmark columns.
                    transform[component] = block[(component, component)];
                }
                transforms.push(transform);
            }
            transforms
        }
        CameraHessian::Dense(_) => {
            return Err("column scaling requires pose-diagonal normal equations");
        }
    };

    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0;
    let mut clamped_to_minimum = 0;
    let mut clamped_to_maximum = 0;
    let pose_transforms = pose_transforms
        .into_iter()
        .map(|diagonal| {
            let mut transform = Vector6::zeros();
            for component in 0..6 {
                transform[component] = column_scaling_transform(
                    diagonal[component],
                    &mut minimum,
                    &mut maximum,
                    &mut clamped_to_minimum,
                    &mut clamped_to_maximum,
                )?;
            }
            Ok(transform)
        })
        .collect::<Result<Vec<_>, &'static str>>()?;

    let mut landmark_transforms = Vec::with_capacity(system.landmarks.len());
    for landmark in &system.landmarks {
        let mut transform = Vector3::zeros();
        for component in 0..3 {
            transform[component] = column_scaling_transform(
                landmark.h_ll[(component, component)],
                &mut minimum,
                &mut maximum,
                &mut clamped_to_minimum,
                &mut clamped_to_maximum,
            )?;
        }
        landmark_transforms.push(transform);
    }

    let scale_block6 = |block: &mut Matrix6<f64>, transform: &Vector6<f64>| {
        for row in 0..6 {
            for column in 0..6 {
                let value = block[(row, column)] * transform[row] * transform[column];
                if !value.is_finite() {
                    return Err("scaled pose Hessian is non-finite");
                }
                block[(row, column)] = value;
            }
        }
        Ok(())
    };
    let CameraHessian::PoseDiagonal(blocks) = &mut system.h_pp else {
        unreachable!("pose diagonal was checked above");
    };
    for (pose, block) in blocks.iter_mut().enumerate() {
        scale_block6(block, &pose_transforms[pose])?;
        for (component, transform) in pose_transforms[pose].iter().enumerate() {
            let index = pose * 6 + component;
            let value = system.b_p[index] * transform;
            if !value.is_finite() {
                return Err("scaled pose gradient is non-finite");
            }
            system.b_p[index] = value;
        }
    }

    for (landmark_index, landmark) in system.landmarks.iter_mut().enumerate() {
        let transform = landmark_transforms[landmark_index];
        for row in 0..3 {
            for column in 0..3 {
                let value = landmark.h_ll[(row, column)] * transform[row] * transform[column];
                if !value.is_finite() {
                    return Err("scaled landmark Hessian is non-finite");
                }
                landmark.h_ll[(row, column)] = value;
            }
            let value = landmark.b_l[row] * transform[row];
            if !value.is_finite() {
                return Err("scaled landmark gradient is non-finite");
            }
            landmark.b_l[row] = value;
        }
        for (pose, cross) in &mut landmark.cross {
            let Some(pose_transform) = pose_transforms.get(*pose) else {
                return Err("landmark cross pose index is invalid");
            };
            for row in 0..6 {
                for column in 0..3 {
                    let value = cross[(row, column)] * pose_transform[row] * transform[column];
                    if !value.is_finite() {
                        return Err("scaled landmark cross is non-finite");
                    }
                    cross[(row, column)] = value;
                }
            }
        }
    }

    Ok(ColumnEquilibrationState {
        pose_transforms,
        landmark_transforms,
        stats: MatrixFreeBaColumnScalingIterationStats {
            iteration: 0,
            minimum_diagonal: minimum,
            maximum_diagonal: maximum,
            clamped_to_minimum,
            clamped_to_maximum,
        },
    })
}

#[cfg(test)]
mod column_scaling_tests {
    use super::*;

    fn synthetic_system() -> NormalEquationsBa {
        let h_pp = Matrix6::from_diagonal(&Vector6::from_row_slice(&[
            4.0, 9.0, 16.0, 25.0, 36.0, 49.0,
        ]));
        let h_ll = Matrix3::from_diagonal(&Vector3::from_row_slice(&[4.0, 9.0, 16.0]));
        let cross_a = Matrix6x3::from_fn(|row, column| {
            if row == column {
                0.25
            } else if row == column + 3 {
                -0.125
            } else {
                0.0
            }
        });
        let cross_b = Matrix6x3::from_fn(|row, column| if row == column { -0.0625 } else { 0.0 });
        NormalEquationsBa {
            h_pp: CameraHessian::PoseDiagonal(vec![h_pp]),
            b_p: DVector::from_row_slice(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]),
            landmarks: vec![LandmarkBlock {
                h_ll,
                b_l: Vector3::new(0.5, -0.75, 1.25),
                // Two entries for the same pose model two sensors observing
                // one landmark.  The transform must touch both entries.
                cross: vec![(0, cross_a), (0, cross_b)],
            }],
        }
    }

    fn full_normal(system: &NormalEquationsBa) -> (DMatrix<f64>, DVector<f64>) {
        let CameraHessian::PoseDiagonal(pose_blocks) = &system.h_pp else {
            panic!("synthetic system must use pose blocks");
        };
        let pose_count = pose_blocks.len();
        let landmark_count = system.landmarks.len();
        let dimension = pose_count * 6 + landmark_count * 3;
        let mut h = DMatrix::zeros(dimension, dimension);
        let mut b = DVector::zeros(dimension);
        for (pose, block) in pose_blocks.iter().enumerate() {
            for row in 0..6 {
                b[pose * 6 + row] = system.b_p[pose * 6 + row];
                for column in 0..6 {
                    h[(pose * 6 + row, pose * 6 + column)] = block[(row, column)];
                }
            }
        }
        for (landmark_index, landmark) in system.landmarks.iter().enumerate() {
            let offset = pose_count * 6 + landmark_index * 3;
            for row in 0..3 {
                b[offset + row] = landmark.b_l[row];
                for column in 0..3 {
                    h[(offset + row, offset + column)] = landmark.h_ll[(row, column)];
                }
            }
            for (pose, cross) in &landmark.cross {
                for row in 0..6 {
                    for column in 0..3 {
                        h[(pose * 6 + row, offset + column)] += cross[(row, column)];
                        h[(offset + column, pose * 6 + row)] += cross[(row, column)];
                    }
                }
            }
        }
        (h, b)
    }

    fn diagonal_damping(system: &NormalEquationsBa) -> DVector<f64> {
        let (h, _) = full_normal(system);
        DVector::from_iterator(
            h.nrows(),
            h.diagonal()
                .iter()
                .map(|value| value.clamp(COLUMN_SCALING_MIN_DIAGONAL, COLUMN_SCALING_MAX_DIAGONAL)),
        )
    }

    #[test]
    fn scaled_system_matches_h_plus_lambda_diagonal_damping() {
        let mut original = synthetic_system();
        let (full_h, full_b) = full_normal(&original);
        let damping_diagonal = diagonal_damping(&original);
        let lambda = 0.25;

        let state = column_equilibrate_normal_system(&mut original).unwrap();
        assert_eq!(state.stats.minimum_diagonal, 4.0);
        assert_eq!(state.stats.maximum_diagonal, 49.0);
        assert_eq!(state.stats.clamped_to_minimum, 0);
        assert_eq!(state.stats.clamped_to_maximum, 0);

        let (scaled_h, scaled_b) = full_normal(&original);
        let scaled_solution: DVector<f64> = (scaled_h + lambda * DMatrix::<f64>::identity(9, 9))
            .lu()
            .solve(&(-scaled_b))
            .expect("scaled synthetic system should solve");
        let mut scaled_pose = DVector::from_iterator(6, scaled_solution.rows(0, 6).iter().copied());
        let mut scaled_landmarks =
            DVector::from_iterator(3, scaled_solution.rows(6, 3).iter().copied());
        state
            .unscale_deltas(&mut scaled_pose, &mut scaled_landmarks)
            .unwrap();
        let mut physical_solution = DVector::zeros(9);
        physical_solution.rows_mut(0, 6).copy_from(&scaled_pose);
        physical_solution
            .rows_mut(6, 3)
            .copy_from(&scaled_landmarks);

        let mut damped_h = full_h;
        for index in 0..9 {
            damped_h[(index, index)] += lambda * damping_diagonal[index];
        }
        let expected = damped_h
            .lu()
            .solve(&(-full_b))
            .expect("physical synthetic system should solve");
        assert!((physical_solution - expected).norm() < 1.0e-12);
    }

    #[test]
    fn matrix_free_scaled_step_matches_the_diagonally_damped_full_system() {
        let mut scaled = synthetic_system();
        let (full_h, full_b) = full_normal(&scaled);
        let damping_diagonal = diagonal_damping(&scaled);
        let state = column_equilibrate_normal_system(&mut scaled).unwrap();
        let lambda = 0.25;

        // Exercise the same Schur/PCG/back-substitution path used by the
        // production opt-in entry point.  The assertion below is against the
        // independently assembled full system, rather than against another
        // call to the scaled operator.
        let outcome = match solve_matrix_free_step(
            &scaled,
            lambda,
            MatrixFreeBaOptions {
                max_pcg_iterations: 128,
                pcg_relative_tolerance: 1.0e-10,
                pcg_absolute_tolerance: 1.0e-12,
            },
            0,
            false,
            Some(&state),
            None,
            false,
        ) {
            Ok(outcome) => outcome,
            Err(_) => panic!("scaled synthetic matrix-free solve should succeed"),
        };
        assert!(outcome.diagnostics.pcg_failure.is_none());

        let mut damped_h = full_h;
        for index in 0..damped_h.nrows() {
            damped_h[(index, index)] += lambda * damping_diagonal[index];
        }
        let expected = damped_h
            .lu()
            .solve(&(-full_b))
            .expect("physical synthetic system should solve");
        let mut actual = DVector::zeros(9);
        actual.rows_mut(0, 6).copy_from(&outcome.delta_poses);
        actual.rows_mut(6, 3).copy_from(&outcome.delta_landmarks);
        assert!((actual - expected).norm() < 1.0e-8);
    }

    #[test]
    fn scaling_preserves_fixed_rotation_identity_and_rejects_bad_diagonals() {
        let mut fixed = synthetic_system();
        let pose_index = BTreeMap::from([(7_u64, 0_usize)]);
        let fixed_rotations = BTreeSet::from([7_u64]);
        constrain_fixed_pose_rotations(&fixed_rotations, &pose_index, &mut fixed);
        let state = column_equilibrate_normal_system(&mut fixed).unwrap();
        assert_eq!(state.pose_transforms[0][3], 1.0);
        assert_eq!(state.pose_transforms[0][4], 1.0);
        assert_eq!(state.pose_transforms[0][5], 1.0);
        let CameraHessian::PoseDiagonal(blocks) = fixed.h_pp else {
            panic!("fixed synthetic system must use pose blocks");
        };
        for component in 3..6 {
            assert_eq!(blocks[0][(component, component)], 1.0);
        }

        let mut clamped = synthetic_system();
        if let CameraHessian::PoseDiagonal(blocks) = &mut clamped.h_pp {
            blocks[0][(0, 0)] = 0.0;
            blocks[0][(1, 1)] = 1.0e40;
        }
        let clamped_state = column_equilibrate_normal_system(&mut clamped).unwrap();
        assert_eq!(clamped_state.stats.minimum_diagonal, 1.0e-6);
        assert_eq!(clamped_state.stats.maximum_diagonal, 1.0e32);
        assert!(clamped_state.stats.clamped_to_minimum >= 1);
        assert!(clamped_state.stats.clamped_to_maximum >= 1);

        let mut negative = synthetic_system();
        if let CameraHessian::PoseDiagonal(blocks) = &mut negative.h_pp {
            blocks[0][(0, 0)] = -1.0;
        }
        assert!(column_equilibrate_normal_system(&mut negative)
            .unwrap_err()
            .contains("negative"));

        let mut nonfinite = synthetic_system();
        nonfinite.landmarks[0].h_ll[(1, 1)] = f64::NAN;
        assert!(column_equilibrate_normal_system(&mut nonfinite)
            .unwrap_err()
            .contains("non-finite"));

        let mut nonfinite_offdiag = synthetic_system();
        if let CameraHessian::PoseDiagonal(blocks) = &mut nonfinite_offdiag.h_pp {
            blocks[0][(0, 1)] = f64::NAN;
        }
        assert!(column_equilibrate_normal_system(&mut nonfinite_offdiag)
            .unwrap_err()
            .contains("pose Hessian"));

        let mut nonfinite_pose_gradient = synthetic_system();
        nonfinite_pose_gradient.b_p[0] = f64::NAN;
        assert!(
            column_equilibrate_normal_system(&mut nonfinite_pose_gradient)
                .unwrap_err()
                .contains("pose gradient")
        );

        let mut nonfinite_cross = synthetic_system();
        nonfinite_cross.landmarks[0].cross[0].1[(0, 0)] = f64::NAN;
        assert!(column_equilibrate_normal_system(&mut nonfinite_cross)
            .unwrap_err()
            .contains("landmark cross"));

        let state = column_equilibrate_normal_system(&mut synthetic_system()).unwrap();
        let mut wrong_pose = DVector::zeros(1);
        let mut valid_landmarks = DVector::zeros(3);
        assert!(state
            .unscale_deltas(&mut wrong_pose, &mut valid_landmarks)
            .unwrap_err()
            .contains("dimensions"));
        let mut nonfinite_pose = DVector::from_element(6, f64::NAN);
        let mut valid_landmarks = DVector::zeros(3);
        assert!(state
            .unscale_deltas(&mut nonfinite_pose, &mut valid_landmarks)
            .unwrap_err()
            .contains("non-finite"));
    }
}

#[cfg(test)]
mod lm_step_quality_tests {
    use super::*;

    fn synthetic_system() -> NormalEquationsBa {
        let h_pp = Matrix6::from_diagonal(&Vector6::from_row_slice(&[
            4.0, 9.0, 16.0, 25.0, 36.0, 49.0,
        ]));
        let h_ll = Matrix3::from_diagonal(&Vector3::new(4.0, 9.0, 16.0));
        let cross_a = Matrix6x3::from_fn(|row, column| {
            if row == column {
                0.25
            } else if row == column + 3 {
                -0.125
            } else {
                0.0
            }
        });
        let cross_b = Matrix6x3::from_fn(|row, column| if row == column { -0.0625 } else { 0.0 });
        NormalEquationsBa {
            h_pp: CameraHessian::PoseDiagonal(vec![h_pp]),
            b_p: DVector::from_row_slice(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]),
            landmarks: vec![LandmarkBlock {
                h_ll,
                b_l: Vector3::new(0.5, -0.75, 1.25),
                // Deliberate same-pose duplicate: the quality denominator
                // must use |cross_a + cross_b|, not |cross_a|+|cross_b|.
                cross: vec![(0, cross_a), (0, cross_b)],
            }],
        }
    }

    fn full_normal(system: &NormalEquationsBa) -> (DMatrix<f64>, DVector<f64>) {
        let CameraHessian::PoseDiagonal(pose_blocks) = &system.h_pp else {
            panic!("quality test requires pose blocks");
        };
        let pose_count = pose_blocks.len();
        let landmark_count = system.landmarks.len();
        let dimension = pose_count * 6 + landmark_count * 3;
        let mut h = DMatrix::zeros(dimension, dimension);
        let mut b = DVector::zeros(dimension);
        for (pose, block) in pose_blocks.iter().enumerate() {
            for row in 0..6 {
                b[pose * 6 + row] = system.b_p[pose * 6 + row];
                for column in 0..6 {
                    h[(pose * 6 + row, pose * 6 + column)] = block[(row, column)];
                }
            }
        }
        for (landmark_index, landmark) in system.landmarks.iter().enumerate() {
            let offset = pose_count * 6 + landmark_index * 3;
            for row in 0..3 {
                b[offset + row] = landmark.b_l[row];
                for column in 0..3 {
                    h[(offset + row, offset + column)] = landmark.h_ll[(row, column)];
                }
            }
            for (pose, cross) in &landmark.cross {
                for row in 0..6 {
                    for column in 0..3 {
                        h[(pose * 6 + row, offset + column)] += cross[(row, column)];
                        h[(offset + column, pose * 6 + row)] += cross[(row, column)];
                    }
                }
            }
        }
        (h, b)
    }

    fn assert_close(actual: f64, expected: f64) {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!((actual - expected).abs() <= 1.0e-11 * scale);
    }

    fn dense_quality_metrics(
        undamped: &DMatrix<f64>,
        damped: &DMatrix<f64>,
        rhs: &DVector<f64>,
        delta: &DVector<f64>,
    ) -> (f64, f64, f64) {
        let residual = damped * delta + rhs;
        let prediction = -2.0 * rhs.dot(delta) - delta.dot(&(undamped * delta));
        let denominator = damped.norm() * delta.norm() + rhs.norm();
        let mut componentwise: f64 = 0.0;
        for row in 0..damped.nrows() {
            let row_denominator = (0..damped.ncols())
                .map(|column| damped[(row, column)].abs() * delta[column].abs())
                .sum::<f64>()
                + rhs[row].abs();
            let ratio = if row_denominator == 0.0 {
                assert_eq!(residual[row], 0.0);
                0.0
            } else {
                residual[row].abs() / row_denominator
            };
            componentwise = componentwise.max(ratio);
        }
        (prediction, residual.norm() / denominator, componentwise)
    }

    #[test]
    fn quality_matches_explicit_full_normal_prediction_and_eta() {
        let system = synthetic_system();
        let before = full_normal(&system);
        let delta_pose = DVector::from_row_slice(&[0.2, -0.3, 0.4, -0.5, 0.6, -0.7]);
        let delta_landmark = DVector::from_row_slice(&[0.8, -0.9, 1.0]);
        let lambda = 0.5;
        let quality = quality_coordinate_metrics(
            &system,
            lambda,
            &delta_pose,
            &delta_landmark,
            MatrixFreeQualityCoordinate::Current,
        )
        .unwrap();

        let (h, b) = full_normal(&system);
        let mut damped = h.clone();
        for index in 0..damped.nrows() {
            damped[(index, index)] += lambda;
        }
        let mut delta = DVector::zeros(9);
        delta.rows_mut(0, 6).copy_from(&delta_pose);
        delta.rows_mut(6, 3).copy_from(&delta_landmark);
        let (expected_prediction, expected_normwise, expected_componentwise) =
            dense_quality_metrics(&h, &damped, &b, &delta);

        assert_close(
            quality.predicted_undamped_squared_decrease,
            expected_prediction,
        );
        assert_close(quality.normwise_backward_error, expected_normwise);
        assert_close(quality.componentwise_backward_error, expected_componentwise);
        assert_eq!(before, full_normal(&system));
    }

    #[test]
    fn quality_streams_multiple_landmarks_against_full_normal_oracle() {
        let mut system = synthetic_system();
        system.landmarks.push(LandmarkBlock {
            h_ll: Matrix3::from_diagonal(&Vector3::new(7.0, 8.0, 9.0)),
            b_l: Vector3::new(-0.2, 0.4, -0.6),
            cross: vec![(
                0,
                Matrix6x3::from_fn(|row, column| if row == column + 1 { 0.15 } else { 0.0 }),
            )],
        });
        let delta_pose = DVector::from_row_slice(&[0.2, -0.3, 0.4, -0.5, 0.6, -0.7]);
        let delta_landmarks = DVector::from_row_slice(&[0.8, -0.9, 1.0, -1.1, 1.2, -1.3]);
        let lambda = 0.5;
        let quality = quality_coordinate_metrics(
            &system,
            lambda,
            &delta_pose,
            &delta_landmarks,
            MatrixFreeQualityCoordinate::Current,
        )
        .unwrap();
        let (h, b) = full_normal(&system);
        let mut delta = DVector::zeros(12);
        delta.rows_mut(0, 6).copy_from(&delta_pose);
        delta.rows_mut(6, 6).copy_from(&delta_landmarks);
        let mut damped = h.clone();
        for index in 0..damped.nrows() {
            damped[(index, index)] += lambda;
        }
        let (expected_prediction, expected_normwise, expected_componentwise) =
            dense_quality_metrics(&h, &damped, &b, &delta);
        assert_close(
            quality.predicted_undamped_squared_decrease,
            expected_prediction,
        );
        assert_close(quality.normwise_backward_error, expected_normwise);
        assert_close(quality.componentwise_backward_error, expected_componentwise);
    }

    #[test]
    fn adaptive_prediction_matches_full_normal_with_rig_crosses_and_fixed_rotation() {
        let mut system = NormalEquationsBa {
            h_pp: CameraHessian::PoseDiagonal(vec![
                Matrix6::from_diagonal(&Vector6::from_row_slice(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
                Matrix6::from_diagonal(&Vector6::from_row_slice(&[
                    10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                ])),
            ]),
            b_p: DVector::from_row_slice(&[
                1.0, -2.0, 3.0, -4.0, 5.0, -6.0, -1.5, 2.5, -3.5, 0.0, 0.0, 0.0,
            ]),
            landmarks: vec![
                LandmarkBlock {
                    h_ll: Matrix3::from_diagonal(&Vector3::new(3.0, 4.0, 5.0)),
                    b_l: Vector3::new(0.25, -0.5, 0.75),
                    cross: vec![
                        (
                            0,
                            Matrix6x3::from_fn(|row, column| {
                                if row == column {
                                    0.2
                                } else if row == column + 3 {
                                    -0.1
                                } else {
                                    0.0
                                }
                            }),
                        ),
                        (
                            0,
                            Matrix6x3::from_fn(
                                |row, column| {
                                    if row == column {
                                        -0.05
                                    } else {
                                        0.0
                                    }
                                },
                            ),
                        ),
                        (
                            1,
                            Matrix6x3::from_fn(
                                |row, column| {
                                    if row == column + 3 {
                                        0.08
                                    } else {
                                        0.0
                                    }
                                },
                            ),
                        ),
                    ],
                },
                LandmarkBlock {
                    h_ll: Matrix3::from_diagonal(&Vector3::new(6.0, 7.0, 8.0)),
                    b_l: Vector3::new(-0.4, 0.6, -0.8),
                    cross: vec![
                        (
                            1,
                            Matrix6x3::from_fn(
                                |row, column| {
                                    if row == column {
                                        -0.12
                                    } else {
                                        0.0
                                    }
                                },
                            ),
                        ),
                        (
                            0,
                            Matrix6x3::from_fn(
                                |row, column| {
                                    if row == column + 3 {
                                        0.07
                                    } else {
                                        0.0
                                    }
                                },
                            ),
                        ),
                    ],
                },
            ],
        };
        constrain_fixed_pose_rotations(
            &BTreeSet::from([20_u64]),
            &BTreeMap::from([(10_u64, 0_usize), (20_u64, 1_usize)]),
            &mut system,
        );
        let before = full_normal(&system);
        let (full_h, full_b) = full_normal(&system);
        let lambda = 0.5;
        let damped = &full_h + lambda * DMatrix::<f64>::identity(18, 18);
        let solved = damped
            .lu()
            .solve(&(-full_b.clone()))
            .expect("multi-pose prediction fixture should solve");
        let perturbation = DVector::from_row_slice(&[
            0.003, -0.002, 0.001, -0.0015, 0.0025, -0.001, 0.0012, -0.0018, 0.0009, 0.0, 0.0, 0.0,
            0.0017, -0.0011, 0.0008, -0.0014, 0.0006, -0.0009,
        ]);
        let delta = solved + perturbation;
        let delta_poses = delta.rows(0, 12).into_owned();
        let delta_landmarks = delta.rows(12, 6).into_owned();
        let expected_prediction = -2.0 * full_b.dot(&delta) - delta.dot(&(&full_h * &delta));
        let prediction = matrix_free_undamped_prediction(&system, &delta_poses, &delta_landmarks)
            .expect("adaptive prediction fixture should be finite");
        let quality = quality_coordinate_metrics(
            &system,
            lambda,
            &delta_poses,
            &delta_landmarks,
            MatrixFreeQualityCoordinate::Current,
        )
        .expect("quality fixture should be finite");
        assert_close(prediction, expected_prediction);
        assert_close(
            quality.predicted_undamped_squared_decrease,
            expected_prediction,
        );
        assert_eq!(before, full_normal(&system));
    }

    #[test]
    fn componentwise_eta_is_invariant_under_positive_column_scaling() {
        let original = synthetic_system();
        let mut scaled = synthetic_system();
        let state = column_equilibrate_normal_system(&mut scaled).unwrap();
        let lambda = 0.5;
        let (scaled_h, scaled_b) = full_normal(&scaled);
        let mut scaled_damped = scaled_h.clone();
        for index in 0..scaled_damped.nrows() {
            scaled_damped[(index, index)] += lambda;
        }
        let scaled_solution = scaled_damped
            .clone()
            .lu()
            .solve(&(-scaled_b.clone()))
            .expect("scaled synthetic system should solve");
        // Use a solved step plus a small, non-collinear perturbation.  This
        // keeps the residual nonzero without making eta a saturated 1.0
        // artifact of an arbitrary hand-written delta.
        let scaled_delta = scaled_solution
            + DVector::from_row_slice(&[
                3.0e-3, -2.0e-3, 1.5e-3, -1.0e-3, 2.5e-3, -1.25e-3, 1.75e-3, -2.25e-3, 0.875e-3,
            ]);
        let scaled_pose = scaled_delta.rows(0, 6).into_owned();
        let scaled_landmark = scaled_delta.rows(6, 3).into_owned();

        let mut physical_transform = DVector::zeros(9);
        for component in 0..6 {
            physical_transform[component] = state.pose_transforms[0][component];
        }
        for component in 0..3 {
            physical_transform[6 + component] = state.landmark_transforms[0][component];
        }
        let mut inverse_transform = DVector::zeros(9);
        let mut physical_delta = DVector::zeros(9);
        for index in 0..9 {
            inverse_transform[index] = physical_transform[index].recip();
            physical_delta[index] = physical_transform[index] * scaled_delta[index];
        }
        let mut physical_h = DMatrix::zeros(9, 9);
        let mut physical_damped = DMatrix::zeros(9, 9);
        let mut physical_b = DVector::zeros(9);
        for row in 0..9 {
            physical_b[row] = inverse_transform[row] * scaled_b[row];
            for column in 0..9 {
                physical_h[(row, column)] =
                    inverse_transform[row] * scaled_h[(row, column)] * inverse_transform[column];
                physical_damped[(row, column)] = inverse_transform[row]
                    * scaled_damped[(row, column)]
                    * inverse_transform[column];
            }
        }

        let (scaled_expected_prediction, scaled_expected_normwise, scaled_expected_eta) =
            dense_quality_metrics(&scaled_h, &scaled_damped, &scaled_b, &scaled_delta);
        let (physical_expected_prediction, physical_expected_normwise, physical_expected_eta) =
            dense_quality_metrics(&physical_h, &physical_damped, &physical_b, &physical_delta);
        let scaled_quality = quality_coordinate_metrics(
            &scaled,
            lambda,
            &scaled_pose,
            &scaled_landmark,
            MatrixFreeQualityCoordinate::Current,
        )
        .unwrap();
        let physical_equivalent = quality_coordinate_metrics(
            &scaled,
            lambda,
            &scaled_pose,
            &scaled_landmark,
            MatrixFreeQualityCoordinate::PhysicalEquivalent(&state),
        )
        .unwrap();

        assert_close(
            scaled_quality.predicted_undamped_squared_decrease,
            scaled_expected_prediction,
        );
        assert_close(
            scaled_quality.normwise_backward_error,
            scaled_expected_normwise,
        );
        assert_close(
            scaled_quality.componentwise_backward_error,
            scaled_expected_eta,
        );
        assert_close(
            physical_equivalent.predicted_undamped_squared_decrease,
            physical_expected_prediction,
        );
        assert_close(
            physical_equivalent.normwise_backward_error,
            physical_expected_normwise,
        );
        assert_close(
            physical_equivalent.componentwise_backward_error,
            physical_expected_eta,
        );
        assert_close(scaled_expected_eta, physical_expected_eta);
        assert!(scaled_expected_eta > 0.0 && scaled_expected_eta < 1.0);
        assert!(physical_expected_eta > 0.0 && physical_expected_eta < 1.0);

        // Undamped prediction is invariant under the positive diagonal
        // coordinate transform, even though lambda I is represented as
        // lambda*diag(d) after returning to physical coordinates.
        let (original_h, original_b) = full_normal(&original);
        let original_quality = quality_coordinate_metrics(
            &original,
            lambda,
            &physical_delta.rows(0, 6).into_owned(),
            &physical_delta.rows(6, 3).into_owned(),
            MatrixFreeQualityCoordinate::Current,
        )
        .unwrap();
        let original_expected_prediction = -2.0 * original_b.dot(&physical_delta)
            - physical_delta.dot(&(&original_h * &physical_delta));
        assert_close(
            original_quality.predicted_undamped_squared_decrease,
            original_expected_prediction,
        );
        assert_close(
            original_quality.predicted_undamped_squared_decrease,
            physical_expected_prediction,
        );
    }

    #[test]
    fn quality_allows_fixed_zero_rows_and_reports_invalid_inputs() {
        let mut fixed = synthetic_system();
        constrain_fixed_pose_rotations(
            &BTreeSet::from([7_u64]),
            &BTreeMap::from([(7_u64, 0_usize)]),
            &mut fixed,
        );
        let zero_rotation = DVector::from_row_slice(&[0.2, -0.3, 0.4, 0.0, 0.0, 0.0]);
        let landmark = DVector::from_row_slice(&[0.8, -0.9, 1.0]);
        let fixed_quality = quality_coordinate_metrics(
            &fixed,
            0.5,
            &zero_rotation,
            &landmark,
            MatrixFreeQualityCoordinate::Current,
        )
        .unwrap();
        assert!(fixed_quality.componentwise_backward_error.is_finite());

        let mut nonfinite = synthetic_system();
        nonfinite.b_p[0] = f64::NAN;
        assert!(quality_coordinate_metrics(
            &nonfinite,
            0.5,
            &zero_rotation,
            &landmark,
            MatrixFreeQualityCoordinate::Current,
        )
        .unwrap_err()
        .contains("non-finite"));

        let zero_system = NormalEquationsBa {
            h_pp: CameraHessian::PoseDiagonal(vec![Matrix6::zeros()]),
            b_p: DVector::zeros(6),
            landmarks: Vec::new(),
        };
        let zero_delta = DVector::zeros(6);
        assert!(quality_coordinate_metrics(
            &zero_system,
            0.0,
            &zero_delta,
            &DVector::zeros(0),
            MatrixFreeQualityCoordinate::Current,
        )
        .unwrap_err()
        .contains("denominator is zero"));
    }

    #[test]
    fn rho_is_undefined_when_nonprojectable_set_changes() {
        let (actual, rho, reason) =
            matrix_free_quality_actual_and_rho(100.0, 90.0, Some(20.0), 0, 1);
        assert_eq!(actual, Some(10.0));
        assert_eq!(rho, None);
        assert_eq!(reason, "nonprojectable_count_nonzero");

        let (actual, rho, reason) =
            matrix_free_quality_actual_and_rho(100.0, 90.0, Some(20.0), 0, 0);
        assert_eq!(actual, Some(10.0));
        assert_eq!(rho, Some(0.5));
        assert_eq!(reason, "defined");

        let (actual, rho, reason) =
            matrix_free_quality_actual_and_rho(f64::MAX, -f64::MAX, Some(1.0), 0, 0);
        assert_eq!(actual, None);
        assert_eq!(rho, None);
        assert_eq!(reason, "actual_cost_decrease_nonfinite");

        let (actual, rho, reason) =
            matrix_free_quality_actual_and_rho(1.0, 0.0, Some(f64::from_bits(1)), 0, 0);
        assert_eq!(actual, Some(1.0));
        assert_eq!(rho, None);
        assert_eq!(reason, "rho_nonfinite");
    }
}

/// Internal dispatch state for the shared LM loop.  The legacy variant is
/// deliberately the default path; the matrix-free variant is only installed
/// by [`BundleAdjustment::optimize_matrix_free`].
enum BaSolveBackend {
    Legacy,
    MatrixFree(MatrixFreeRuntime),
    MatrixFreeColumnScaled(MatrixFreeRuntime),
}

struct MatrixFreeRuntime {
    options: MatrixFreeBaOptions,
    iterations: Vec<MatrixFreeBaIterationStats>,
    restart_limit: usize,
    restart_iterations: Option<Vec<MatrixFreeBaRestartIterationStats>>,
    column_scaling_iterations: Option<Vec<MatrixFreeBaColumnScalingIterationStats>>,
    adaptive_damping: bool,
    adaptive_iterations: Option<Vec<MatrixFreeBaAdaptiveDampingIterationStats>>,
    failure: Option<MatrixFreeBaError>,
}

impl MatrixFreeRuntime {
    fn new(options: MatrixFreeBaOptions) -> Self {
        Self {
            options,
            iterations: Vec::new(),
            restart_limit: 0,
            restart_iterations: None,
            column_scaling_iterations: None,
            adaptive_damping: false,
            adaptive_iterations: None,
            failure: None,
        }
    }

    fn with_restart(options: MatrixFreeBaOptions, restart_limit: usize) -> Self {
        Self {
            options,
            iterations: Vec::new(),
            restart_limit,
            restart_iterations: Some(Vec::new()),
            column_scaling_iterations: None,
            adaptive_damping: false,
            adaptive_iterations: None,
            failure: None,
        }
    }

    fn with_column_scaling(options: MatrixFreeBaOptions) -> Self {
        Self {
            options,
            iterations: Vec::new(),
            restart_limit: 0,
            restart_iterations: None,
            column_scaling_iterations: Some(Vec::new()),
            adaptive_damping: false,
            adaptive_iterations: None,
            failure: None,
        }
    }

    fn with_column_scaling_adaptive(options: MatrixFreeBaOptions) -> Self {
        Self {
            options,
            iterations: Vec::new(),
            restart_limit: 0,
            restart_iterations: None,
            column_scaling_iterations: Some(Vec::new()),
            adaptive_damping: true,
            adaptive_iterations: Some(Vec::new()),
            failure: None,
        }
    }
}

const ADAPTIVE_ACCEPT_MIN_LAMBDA_FACTOR: f64 = 1.0 / 3.0;

fn bounded_adaptive_lambda(lambda: f64, multiplier: f64, min_lambda: f64, max_lambda: f64) -> f64 {
    let product = lambda * multiplier;
    if !product.is_finite() {
        max_lambda
    } else {
        product.clamp(min_lambda, max_lambda)
    }
}

fn adaptive_accepted_lambda(
    lambda: f64,
    rho: f64,
    min_lambda: f64,
    max_lambda: f64,
) -> Result<f64, &'static str> {
    if !rho.is_finite() || rho <= 0.0 {
        return Err("adaptive rho is non-finite or non-positive");
    }
    // Clamp before cubing so a very large finite rho cannot overflow the
    // policy expression.  This is a fixed policy, not a user-tunable sweep.
    let centered = if rho >= 1.0 {
        1.0
    } else {
        (2.0 * rho - 1.0).max(-1.0)
    };
    let multiplier = (1.0 - centered * centered * centered).max(ADAPTIVE_ACCEPT_MIN_LAMBDA_FACTOR);
    if !multiplier.is_finite() {
        return Err("adaptive lambda multiplier is non-finite");
    }
    Ok(bounded_adaptive_lambda(
        lambda, multiplier, min_lambda, max_lambda,
    ))
}

#[derive(Debug)]
struct AdaptiveStepDecision {
    prediction: Option<f64>,
    actual_cost_decrease: Option<f64>,
    rho: Option<f64>,
    cost_gate: bool,
    feasibility_gate: bool,
    accepted: bool,
    reason: String,
}

fn adaptive_step_decision(
    prediction: Option<Result<f64, &'static str>>,
    cost_before: f64,
    cost_after: f64,
    nonprojectable_before: usize,
    nonprojectable_after: usize,
    cost_gate: bool,
    feasibility_gate: bool,
) -> AdaptiveStepDecision {
    let prediction_value = prediction.as_ref().and_then(|value| match value {
        Ok(value) if value.is_finite() => Some(*value),
        _ => None,
    });
    let prediction_error = match prediction.as_ref() {
        Some(Err(error)) => Some(*error),
        Some(Ok(value)) if !value.is_finite() => Some("adaptive prediction is non-finite"),
        _ => None,
    };
    let (actual_cost_decrease, rho, rho_reason) = matrix_free_quality_actual_and_rho(
        cost_before,
        cost_after,
        prediction_value,
        nonprojectable_before,
        nonprojectable_after,
    );
    let rho_positive = rho.is_some_and(|value| value > 0.0);
    let accepted = cost_gate && feasibility_gate && rho_positive;
    let reason = if accepted {
        "accepted".to_owned()
    } else if !cost_gate && !feasibility_gate {
        "candidate_rejected_cost_and_feasibility_gate".to_owned()
    } else if !cost_gate {
        "candidate_rejected_cost_gate".to_owned()
    } else if !feasibility_gate {
        "candidate_rejected_feasibility_gate".to_owned()
    } else if let Some(error) = prediction_error {
        format!("candidate_rejected_prediction:{error}")
    } else if rho.is_some() && !rho_positive {
        "candidate_rejected_rho_nonpositive".to_owned()
    } else {
        format!("candidate_rejected_rho:{rho_reason}")
    };
    AdaptiveStepDecision {
        prediction: prediction_value,
        actual_cost_decrease,
        rho,
        cost_gate,
        feasibility_gate,
        accepted,
        reason,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaCostBreakdown {
    pub total: f64,
    pub visual: f64,
    pub imu: f64,
    pub bias_random_walk: f64,
    pub navigation_prior: f64,
    pub other_structural: f64,
    pub imu_normalized_squared_residual_per_dof: Option<f64>,
    pub imu_rotation_residual_rms_rad: Option<f64>,
    pub imu_velocity_residual_rms_mps: Option<f64>,
    pub imu_position_residual_rms_meters: Option<f64>,
}

/// Result of [`BundleAdjustment::optimize_gnc`].
#[derive(Debug, Clone, PartialEq)]
pub struct BaGncResult {
    /// Non-robust reprojection cost at the input estimate.
    pub initial_cost: f64,
    /// GNC-weighted reprojection cost at the recovered estimate (every
    /// observation scaled by its final `w`).
    pub final_cost: f64,
    /// Reprojection cost over the classified inliers only (outliers
    /// contribute nothing), using the `0.5` weight threshold.
    pub inlier_cost: f64,
    /// The inlier scale `c` (pixels) the solve actually used: the configured
    /// [`GncConfig::c`] verbatim, or — under [`GncConfig::auto_scale`] — the
    /// MAD estimate (floored at the configured `c`).
    pub inlier_scale: f64,
    /// Number of reprojection observations (monocular + stereo) the weight
    /// vector covers.
    pub observation_count: usize,
    /// GNC outer (μ) levels actually executed.
    pub outer_iterations: usize,
    /// Whether the μ schedule reached its terminal level.
    pub converged: bool,
    /// Final per-observation Black-Rangarajan weight `w ∈ [0,1]`, indexed
    /// monocular, rectified stereo, then general stereo. `NaN` marks an observation that could
    /// not be evaluated at the recovered estimate. Near-zero finite entries
    /// are the rejected outliers.
    pub observation_weights: Vec<f64>,
}

impl BaGncResult {
    /// Count of observations classified as inliers (`w ≥ threshold`).
    /// `NaN` (un-evaluable) observations are excluded from both counts.
    pub fn inlier_count(&self, threshold: f64) -> usize {
        self.observation_weights
            .iter()
            .filter(|w| w.is_finite() && **w >= threshold)
            .count()
    }

    /// Count of observations classified as outliers (`w < threshold`).
    pub fn outlier_count(&self, threshold: f64) -> usize {
        self.observation_weights
            .iter()
            .filter(|w| w.is_finite() && **w < threshold)
            .count()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaError {
    NoPoses,
    NoLandmarks,
    NoObservations,
    /// Every pose AND every landmark is fixed, so there is nothing to
    /// optimize. (Pose-only or landmark-only BA is allowed.)
    AllPosesFixed,
    MissingPose(u64),
    MissingLandmark(u64),
    /// Camera model is not pinhole (multi-model BA is a future extension).
    UnsupportedCameraModel,
    /// One or more [`BaStereoObservation`]s were added but
    /// [`BundleAdjustment::stereo_baseline`] is missing or non-positive.
    MissingStereoBaseline,
    /// The external visual-weight vector does not match the flattened visual
    /// observation count (mono, rectified stereo, general stereo).
    ObservationWeightCount {
        expected: usize,
        actual: usize,
    },
    /// An external visual weight is negative, NaN, or infinite.
    InvalidObservationWeight(usize),
    /// A fixed-rotation diagnostic supplied a vector that is not aligned with
    /// the pose vector being optimized.
    InvalidFixedRotationCount {
        expected: usize,
        actual: usize,
    },
    /// Confidence-weighted joint intrinsics refinement is not implemented.
    ObservationWeightsWithIntrinsicsRefinement,
    /// Reduced camera system was singular even after λ damping. Usually
    /// means the gauge is under-fixed (e.g., monocular without enough
    /// fixed poses or landmarks to remove scale).
    SingularSystem,
}

impl std::fmt::Display for BaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BaError::NoPoses => write!(f, "bundle adjustment has no poses"),
            BaError::NoLandmarks => write!(f, "bundle adjustment has no landmarks"),
            BaError::NoObservations => write!(f, "bundle adjustment has no observations"),
            BaError::AllPosesFixed => write!(f, "every pose is fixed; nothing to optimize"),
            BaError::MissingPose(id) => write!(f, "observation references unknown pose {id}"),
            BaError::MissingLandmark(id) => {
                write!(f, "observation references unknown landmark {id}")
            }
            BaError::UnsupportedCameraModel => {
                write!(f, "only pinhole camera models are supported")
            }
            BaError::MissingStereoBaseline => {
                write!(f, "stereo observations require a positive stereo_baseline")
            }
            BaError::ObservationWeightCount { expected, actual } => write!(
                f,
                "observation weight count mismatch: expected {expected}, got {actual}"
            ),
            BaError::InvalidObservationWeight(index) => write!(
                f,
                "observation weight at index {index} must be finite and non-negative"
            ),
            BaError::InvalidFixedRotationCount { expected, actual } => write!(
                f,
                "fixed-rotation pose count mismatch: expected {expected}, got {actual}"
            ),
            BaError::ObservationWeightsWithIntrinsicsRefinement => write!(
                f,
                "observation weights are not supported with joint intrinsics refinement"
            ),
            BaError::SingularSystem => write!(f, "reduced camera system is singular"),
        }
    }
}

impl std::error::Error for BaError {}

/// Per-landmark contribution: the `H_ll` block, the `b_l` gradient, and
/// the `H_pl` cross blocks (one per pose that observed this landmark, in
/// arbitrary order). This is the only place the cross blocks are stored —
/// there is no full `H_PL` matrix.
#[derive(Clone)]
struct LandmarkBlock {
    /// `3×3` Hessian summed over observations. Includes any λ damping.
    h_ll: Matrix3<f64>,
    /// `3-vec` gradient summed over observations.
    b_l: Vector3<f64>,
    /// `(pose_idx, J_pose^T · J_lm)` per observation that touches a
    /// non-fixed pose. Shape `6×3`.
    cross: Vec<(usize, Matrix6x3<f64>)>,
}

/// Per-landmark block for the joint pose+structure+intrinsics solve
/// ([`BundleAdjustment::optimize_joint_intrinsics`]). Unlike [`LandmarkBlock`]
/// the cross blocks are keyed by the touching camera-block column-start (a pose
/// block of width 6, or the shared 4-wide intrinsics block) so a single map
/// holds both the pose and intrinsics couplings, and observations sharing a
/// camera block (e.g. the intrinsics, seen by every observation) accumulate.
struct JointLandmarkBlock {
    id: u64,
    h_ll: Matrix3<f64>,
    b_l: Vector3<f64>,
    /// `column_start → Σ_obs Jᵀ_cam · J_lm` (`rows × 3`, `rows ∈ {6, 4}`).
    cross: BTreeMap<usize, DMatrix<f64>>,
}

/// Accumulate a `rows × 3` cross block into the per-landmark map at `col_start`.
fn navigation_prior_delta(
    ba: &BundleAdjustment,
    prior: &NavigationStatePrior,
) -> Option<DVector<f64>> {
    if !prior.is_well_formed() {
        return None;
    }
    let mut delta = DVector::zeros(prior.keyframe_ids.len() * 15);
    for (slot, id) in prior.keyframe_ids.iter().enumerate() {
        let pose = ba.poses.get(id)?;
        let velocity = ba.velocities.get(id)?;
        let bias = ba.biases.get(id)?;
        let reference_pose = prior.reference_poses.get(id)?;
        let reference_velocity = prior.reference_velocities.get(id)?;
        let reference_bias = prior.reference_biases.get(id)?;
        let pose_delta = reference_pose
            .world_to_camera
            .inverse()
            .compose(&pose.world_to_camera)
            .log();
        delta.fixed_rows_mut::<6>(slot * 15).copy_from(&pose_delta);
        delta
            .fixed_rows_mut::<3>(slot * 15 + 6)
            .copy_from(&(velocity - reference_velocity));
        delta
            .fixed_rows_mut::<6>(slot * 15 + 9)
            .copy_from(&(bias - reference_bias));
    }
    Some(delta)
}

fn add_cross(
    cross: &mut BTreeMap<usize, DMatrix<f64>>,
    col_start: usize,
    rows: usize,
    blk: &DMatrix<f64>,
) {
    *cross
        .entry(col_start)
        .or_insert_with(|| DMatrix::zeros(rows, 3)) += blk;
}

/// Output of the per-iteration normal-equations build.
struct NormalEquationsBa {
    /// Camera-state Hessian. Pure-visual sparse BA keeps only the diagonal
    /// 6×6 pose blocks assembled before Schur reduction; the general visual-
    /// inertial / prior-bearing path retains the dense representation.
    h_pp: CameraHessian,
    /// Pose gradient, dense `6P`.
    b_p: DVector<f64>,
    /// Landmark blocks indexed by landmark variable index.
    landmarks: Vec<LandmarkBlock>,
}

/// Pre-Schur camera Hessian representation.
///
/// For pure visual BA every observation contributes only to one diagonal
/// pose block. Off-diagonal pose coupling appears later, during landmark
/// elimination, so allocating a dense `(6P)²` matrix here is unnecessary.
enum CameraHessian {
    Dense(DMatrix<f64>),
    PoseDiagonal(Vec<Matrix6<f64>>),
}

impl CameraHessian {
    fn dense(dim: usize) -> Self {
        Self::Dense(DMatrix::zeros(dim, dim))
    }

    fn pose_diagonal(pose_count: usize) -> Self {
        Self::PoseDiagonal(vec![Matrix6::zeros(); pose_count])
    }

    fn into_dense(self) -> DMatrix<f64> {
        match self {
            Self::Dense(matrix) => matrix,
            Self::PoseDiagonal(blocks) => {
                let mut matrix = DMatrix::zeros(blocks.len() * 6, blocks.len() * 6);
                for (pose, block) in blocks.into_iter().enumerate() {
                    matrix
                        .fixed_view_mut::<6, 6>(pose * 6, pose * 6)
                        .copy_from(&block);
                }
                matrix
            }
        }
    }
}

impl Index<(usize, usize)> for CameraHessian {
    type Output = f64;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        match self {
            Self::Dense(matrix) => &matrix[(row, col)],
            Self::PoseDiagonal(blocks) => {
                assert_eq!(row / 6, col / 6, "off-diagonal pose Hessian access");
                &blocks[row / 6][(row % 6, col % 6)]
            }
        }
    }
}

impl IndexMut<(usize, usize)> for CameraHessian {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        match self {
            Self::Dense(matrix) => &mut matrix[(row, col)],
            Self::PoseDiagonal(blocks) => {
                assert_eq!(row / 6, col / 6, "off-diagonal pose Hessian access");
                &mut blocks[row / 6][(row % 6, col % 6)]
            }
        }
    }
}

// --- Parallel assembly / Schur-reduction support (see the module's
// "Parallelism" section). ---

/// Below this many observations, [`assemble_mono_observations_parallel`] is
/// not dispatched even when [`BaConfig::parallel`] is set: a full-sequence
/// BA's per-observation cost dwarfs the rayon per-chunk overhead, but small
/// problems (a handful of keyframes) do not, so they stay on the plain
/// serial loop. Mirrors `block_cholesky::PARALLEL_MIN_BLOCKS`.
const PARALLEL_MIN_OBSERVATIONS: usize = 4_096;
/// Observations computed per rayon chunk in the parallel assembly path.
/// Bounds the transient `Vec<Option<MonoObsContribution>>` buffer to a few
/// megabytes regardless of the total observation count — a full-sequence BA
/// can carry tens of millions of observations, so materializing one
/// contribution per observation up front (rather than chunk by chunk) would
/// multiply the assembly's peak memory several-fold. The value does not
/// affect the result at all (see the merge comment on
/// [`assemble_mono_observations_parallel`]), so it is chosen purely for that
/// memory/dispatch-overhead trade-off.
const PARALLEL_OBSERVATION_CHUNK: usize = 65_536;

/// Below this many landmarks, the parallel Schur-reduction and back-
/// substitution paths in [`solve_step`] are not dispatched. Mirrors
/// [`PARALLEL_MIN_OBSERVATIONS`].
const PARALLEL_MIN_LANDMARKS: usize = 2_048;
/// Landmarks processed per rayon chunk in the parallel Schur reduction.
/// Bounds the transient per-chunk `(pose, pose, block)` triplet buffer the
/// same way [`PARALLEL_OBSERVATION_CHUNK`] bounds the assembly path's; it
/// does not affect the result (same reasoning as that constant).
const PARALLEL_LANDMARK_CHUNK: usize = 16_384;

/// Precomputed contribution of one monocular observation to the normal
/// equations: the weighted `H_pp` / `b_p` block for the observing pose (if
/// non-fixed), the weighted `H_ll` / `b_l` block for the observed landmark
/// (if non-fixed), and their cross term (if both are). `None` when the
/// observation is skipped exactly as the serial loop in
/// [`build_normal_equations`] skips it — behind the camera or not
/// projectable. This is the unit of work
/// [`assemble_mono_observations_parallel`] farms out to the rayon pool: it
/// is a pure function of the current pose/landmark estimate, so many can be
/// computed concurrently with no shared mutable state.
struct MonoObsContribution {
    pose: Option<(usize, Matrix6<f64>, Vector6<f64>)>,
    landmark: Option<(usize, Matrix3<f64>, Vector3<f64>)>,
    cross: Option<(usize, usize, Matrix6x3<f64>)>,
}

/// Compute one observation's [`MonoObsContribution`]. Deliberately kept
/// byte-for-byte in step with the inline loop body in
/// [`build_normal_equations`] (same operations in the same order, so the
/// weighted blocks are bitwise identical to what that loop would compute) —
/// if the residual/Jacobian model there ever changes, this must change with
/// it.
#[allow(clippy::too_many_arguments)]
fn compute_mono_contribution(
    ba: &BundleAdjustment,
    intrinsics: &(f64, f64, f64, f64),
    kernel: &RobustKernel,
    gnc_weights: Option<&[f64]>,
    pose_index: &BTreeMap<u64, usize>,
    landmark_index: &BTreeMap<u64, usize>,
    obs_idx: usize,
    obs: &BaObservation,
) -> Option<MonoObsContribution> {
    let pose = &ba.poses[&obs.keyframe_id];
    let point = &ba.landmarks[&obs.landmark_id];
    let r_mat = pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let xc = pose.transform_world_point(point);
    if xc.z <= 0.0 {
        return None;
    }
    let predicted = project_pinhole(intrinsics, &xc)?;
    let residual = Vector2::new(predicted.x - obs.xy.x, predicted.y - obs.xy.y);

    let (fx, fy, _, _) = *intrinsics;
    let z_inv = 1.0 / xc.z;
    let mut j_pi = Matrix2x3::<f64>::zeros();
    j_pi[(0, 0)] = fx * z_inv;
    j_pi[(0, 1)] = 0.0;
    j_pi[(0, 2)] = -fx * xc.x * z_inv * z_inv;
    j_pi[(1, 0)] = 0.0;
    j_pi[(1, 1)] = fy * z_inv;
    j_pi[(1, 2)] = -fy * xc.y * z_inv * z_inv;

    let xw_skew = skew(&point.coords);
    let mut dx_dxi = nalgebra::Matrix3x6::<f64>::zeros();
    dx_dxi.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
    dx_dxi
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-r_mat * xw_skew));
    let j_pose: Matrix2x6<f64> = j_pi * dx_dxi;
    let j_lm: Matrix2x3<f64> = j_pi * r_mat;

    let s = residual.x * residual.x + residual.y * residual.y;
    let w = kernel.weight(s) * gnc_weights.map_or(1.0, |gw| gw[obs_idx]);

    let i_pose = pose_index.get(&obs.keyframe_id).copied();
    let i_lm = landmark_index.get(&obs.landmark_id).copied();

    let pose_update = i_pose.map(|p| {
        let h_pp_block: Matrix6<f64> = j_pose.transpose() * j_pose;
        let b_p_block: Vector6<f64> = j_pose.transpose() * residual;
        (p, w * h_pp_block, w * b_p_block)
    });
    let landmark_update = i_lm.map(|l| {
        let h_ll_block: Matrix3<f64> = j_lm.transpose() * j_lm;
        let b_l_block: Vector3<f64> = j_lm.transpose() * residual;
        (l, w * h_ll_block, w * b_l_block)
    });
    let cross_update = match (i_pose, i_lm) {
        (Some(p), Some(l)) => {
            let cross: Matrix6x3<f64> = j_pose.transpose() * j_lm;
            Some((p, l, w * cross))
        }
        _ => None,
    };

    Some(MonoObsContribution {
        pose: pose_update,
        landmark: landmark_update,
        cross: cross_update,
    })
}

/// Scatter one [`MonoObsContribution`] into the shared accumulators, in
/// exactly the pose-then-landmark-then-cross order the serial loop in
/// [`build_normal_equations`] uses. Never called concurrently on the same
/// `h_pp` / `b_p` / `landmarks` — see the serial merge loop in
/// [`assemble_mono_observations_parallel`].
fn apply_mono_contribution(
    contribution: MonoObsContribution,
    h_pp: &mut CameraHessian,
    b_p: &mut DVector<f64>,
    landmarks: &mut [LandmarkBlock],
) {
    if let Some((p, h_pp_block, b_p_block)) = contribution.pose {
        for r in 0..6 {
            for c in 0..6 {
                h_pp[(p * 6 + r, p * 6 + c)] += h_pp_block[(r, c)];
            }
            b_p[p * 6 + r] += b_p_block[r];
        }
    }
    if let Some((l, h_ll_block, b_l_block)) = contribution.landmark {
        landmarks[l].h_ll += h_ll_block;
        landmarks[l].b_l += b_l_block;
    }
    if let Some((p, l, cross)) = contribution.cross {
        landmarks[l].cross.push((p, cross));
    }
}

#[allow(clippy::too_many_arguments)]
/// Parallel counterpart of the monocular loop in [`build_normal_equations`].
///
/// Processes observations in fixed-size chunks
/// ([`PARALLEL_OBSERVATION_CHUNK`]): within a chunk, every observation's
/// [`MonoObsContribution`] is computed concurrently on the rayon pool (pure
/// function, no shared state) and collected into a `Vec` that preserves the
/// original index order exactly like the serial loop would produce it; the
/// chunk's contributions are then scattered into `h_pp` / `b_p` /
/// `landmarks` by a single serial pass, *in ascending observation-index
/// order*, before the next chunk starts. Because the scatter order is
/// therefore always identical to what the plain serial loop would produce —
/// only the (order-independent) computation of each contribution moved to
/// the pool — the result is bit-identical to the serial path at any thread
/// count or chunk size, unlike a reassociating parallel reduction.
fn assemble_mono_observations_parallel(
    ba: &BundleAdjustment,
    intrinsics: &(f64, f64, f64, f64),
    pose_index: &BTreeMap<u64, usize>,
    landmark_index: &BTreeMap<u64, usize>,
    kernel: &RobustKernel,
    gnc_weights: Option<&[f64]>,
    h_pp: &mut CameraHessian,
    b_p: &mut DVector<f64>,
    landmarks: &mut [LandmarkBlock],
) {
    use rayon::prelude::*;

    let mut start = 0;
    while start < ba.observations.len() {
        let end = (start + PARALLEL_OBSERVATION_CHUNK).min(ba.observations.len());
        let chunk = &ba.observations[start..end];
        let contributions: Vec<Option<MonoObsContribution>> = chunk
            .par_iter()
            .enumerate()
            .map(|(offset, obs)| {
                compute_mono_contribution(
                    ba,
                    intrinsics,
                    kernel,
                    gnc_weights,
                    pose_index,
                    landmark_index,
                    start + offset,
                    obs,
                )
            })
            .collect();
        for contribution in contributions.into_iter().flatten() {
            apply_mono_contribution(contribution, h_pp, b_p, landmarks);
        }
        start = end;
    }
}

#[allow(clippy::too_many_arguments)]
fn build_normal_equations(
    ba: &BundleAdjustment,
    intrinsics: &(f64, f64, f64, f64),
    pose_index: &BTreeMap<u64, usize>,
    landmark_index: &BTreeMap<u64, usize>,
    velocity_index: &BTreeMap<u64, usize>,
    bias_index: &BTreeMap<u64, usize>,
    kernel: &RobustKernel,
    gnc_weights: Option<&[f64]>,
    parallel: bool,
    prefer_pose_blocks: bool,
) -> NormalEquationsBa {
    let p_count = pose_index.len();
    let l_count = landmark_index.len();
    let v_count = velocity_index.len();
    let b_count = bias_index.len();
    // Joint pose+velocity+bias Hessian. Pose slots occupy rows/cols
    // `0 .. 6P`; velocity slots occupy `6P .. 6P + 3V`; bias slots
    // occupy `6P + 3V .. 6P + 3V + 6B`. When IMU factors are absent
    // (`V = 0, B = 0`) the matrix layout is identical to the legacy
    // pose-only system, so the existing call sites that pass empty
    // velocity / bias indices see no change.
    let pose_dim = p_count * 6;
    let vel_offset = pose_dim;
    let bias_offset = pose_dim + v_count * 3;
    let total_dim = pose_dim + v_count * 3 + b_count * 6;
    let pure_visual_pose_blocks = prefer_pose_blocks
        && v_count == 0
        && b_count == 0
        && ba.position_prior.is_none()
        && ba.pairwise_pose_factors.is_empty()
        && ba.gravity_prior.is_none()
        && ba.per_pose_gravity_prior.is_none()
        && ba.imu_factors.is_empty()
        && ba.bias_random_walk_factors.is_empty()
        && ba.navigation_state_prior.is_none();
    let mut h_pp = if pure_visual_pose_blocks {
        CameraHessian::pose_diagonal(p_count)
    } else {
        CameraHessian::dense(total_dim)
    };
    let mut b_p = DVector::<f64>::zeros(total_dim);
    let mut landmarks: Vec<LandmarkBlock> = (0..l_count)
        .map(|_| LandmarkBlock {
            h_ll: Matrix3::zeros(),
            b_l: Vector3::zeros(),
            cross: Vec::new(),
        })
        .collect();

    // Flag-gated, work-gated parallel assembly (see the module's
    // "Parallelism" section): bit-identical to the plain loop below at any
    // thread count, so the branch only changes how the contributions are
    // computed, never the result.
    if parallel && ba.observations.len() >= PARALLEL_MIN_OBSERVATIONS {
        assemble_mono_observations_parallel(
            ba,
            intrinsics,
            pose_index,
            landmark_index,
            kernel,
            gnc_weights,
            &mut h_pp,
            &mut b_p,
            &mut landmarks,
        );
    } else {
        for (obs_idx, obs) in ba.observations.iter().enumerate() {
            let pose = &ba.poses[&obs.keyframe_id];
            let point = &ba.landmarks[&obs.landmark_id];
            let r_mat = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let xc = pose.transform_world_point(point);
            if xc.z <= 0.0 {
                continue;
            }
            // Predicted pixel - measured pixel.
            let predicted = match project_pinhole(intrinsics, &xc) {
                Some(p) => p,
                None => continue,
            };
            let residual = Vector2::new(predicted.x - obs.xy.x, predicted.y - obs.xy.y);

            // Projection Jacobian J_π (2×3) at X_c = (X, Y, Z):
            //   J_π = (1/Z) [[fx, 0, -fx X/Z], [0, fy, -fy Y/Z]]
            let (fx, fy, _, _) = *intrinsics;
            let z_inv = 1.0 / xc.z;
            let mut j_pi = Matrix2x3::<f64>::zeros();
            j_pi[(0, 0)] = fx * z_inv;
            j_pi[(0, 1)] = 0.0;
            j_pi[(0, 2)] = -fx * xc.x * z_inv * z_inv;
            j_pi[(1, 0)] = 0.0;
            j_pi[(1, 1)] = fy * z_inv;
            j_pi[(1, 2)] = -fy * xc.y * z_inv * z_inv;

            // Right perturbation pose Jacobian:
            //   ∂X_c / ∂[ρ; ω] = [R, -R · [X_w]_×]   (3×6)
            let xw_skew = skew(&point.coords);
            let mut dx_dxi = nalgebra::Matrix3x6::<f64>::zeros();
            dx_dxi.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
            dx_dxi
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-r_mat * xw_skew));
            let j_pose: Matrix2x6<f64> = j_pi * dx_dxi;
            // Landmark Jacobian: ∂X_c / ∂X_w = R, so J_lm = J_π · R (2×3).
            let j_lm: Matrix2x3<f64> = j_pi * r_mat;

            // IRLS weight applied per-observation. With `RobustKernel::None`
            // this is `1.0` and the build matches plain Gauss-Newton; with
            // Huber / Cauchy the weight shrinks for large residuals so a
            // single bad observation cannot dominate the normal equations.
            let s = residual.x * residual.x + residual.y * residual.y;
            let w = kernel.weight(s) * gnc_weights.map_or(1.0, |gw| gw[obs_idx]);

            let i_pose = pose_index.get(&obs.keyframe_id).copied();
            let i_lm = landmark_index.get(&obs.landmark_id).copied();

            if let Some(p) = i_pose {
                let h_pp_block: Matrix6<f64> = j_pose.transpose() * j_pose;
                let b_p_block: Vector6<f64> = j_pose.transpose() * residual;
                for r in 0..6 {
                    for c in 0..6 {
                        h_pp[(p * 6 + r, p * 6 + c)] += w * h_pp_block[(r, c)];
                    }
                    b_p[p * 6 + r] += w * b_p_block[r];
                }
            }
            if let Some(l) = i_lm {
                let h_ll_block: Matrix3<f64> = j_lm.transpose() * j_lm;
                let b_l_block: Vector3<f64> = j_lm.transpose() * residual;
                landmarks[l].h_ll += w * h_ll_block;
                landmarks[l].b_l += w * b_l_block;
            }
            if let (Some(p), Some(l)) = (i_pose, i_lm) {
                let cross: Matrix6x3<f64> = j_pose.transpose() * j_lm;
                landmarks[l].cross.push((p, w * cross));
            }
        }
    }

    // Stereo observations: 3D residual `(u_l, v_l, u_r)` with
    // `u_r_pred = u_l_pred − fx · b / Z`. Jacobian of the residual w.r.t.
    // `X_c = (X, Y, Z)` is the 3×3 matrix
    //   J_π_st = [[fx/Z, 0,    -fx·X/Z²       ],
    //             [0,    fy/Z, -fy·Y/Z²       ],
    //             [fx/Z, 0,    -fx·(X-b)/Z²   ]].
    // The pose / landmark Jacobians have shape 3×6 / 3×3, but the
    // accumulated `H_pp = J^T J` and `H_pl = J^T J_lm` blocks have the
    // same 6×6 / 6×3 / 3×3 shapes as the monocular path so the rest of
    // the pipeline (Schur complement, back-substitution) is unchanged.
    if !ba.stereo_observations.is_empty() {
        if let Some(baseline) = ba.stereo_baseline {
            if baseline.is_finite() && baseline > 0.0 {
                let (fx, fy, _, _) = *intrinsics;
                let stereo_offset = ba.observations.len();
                for (st_idx, obs) in ba.stereo_observations.iter().enumerate() {
                    let pose = &ba.poses[&obs.keyframe_id];
                    let point = &ba.landmarks[&obs.landmark_id];
                    let r_mat = pose
                        .world_to_camera
                        .rotation
                        .to_rotation_matrix()
                        .into_inner();
                    let xc = pose.transform_world_point(point);
                    if xc.z <= 0.0 {
                        continue;
                    }
                    let predicted = match project_pinhole(intrinsics, &xc) {
                        Some(p) => p,
                        None => continue,
                    };
                    let u_r_pred = predicted.x - fx * baseline / xc.z;
                    let residual = Vector3::new(
                        predicted.x - obs.xy.x,
                        predicted.y - obs.xy.y,
                        u_r_pred - obs.u_right,
                    );

                    let z_inv = 1.0 / xc.z;
                    let z_inv2 = z_inv * z_inv;
                    let mut j_pi = Matrix3::<f64>::zeros();
                    j_pi[(0, 0)] = fx * z_inv;
                    j_pi[(0, 2)] = -fx * xc.x * z_inv2;
                    j_pi[(1, 1)] = fy * z_inv;
                    j_pi[(1, 2)] = -fy * xc.y * z_inv2;
                    j_pi[(2, 0)] = fx * z_inv;
                    j_pi[(2, 2)] = -fx * (xc.x - baseline) * z_inv2;

                    let xw_skew = skew(&point.coords);
                    let mut dx_dxi = Matrix3x6::<f64>::zeros();
                    dx_dxi.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
                    dx_dxi
                        .fixed_view_mut::<3, 3>(0, 3)
                        .copy_from(&(-r_mat * xw_skew));
                    let j_pose: Matrix3x6<f64> = j_pi * dx_dxi;
                    let j_lm: Matrix3<f64> = j_pi * r_mat;

                    let s = residual.norm_squared();
                    let w =
                        kernel.weight(s) * gnc_weights.map_or(1.0, |gw| gw[stereo_offset + st_idx]);

                    let i_pose = pose_index.get(&obs.keyframe_id).copied();
                    let i_lm = landmark_index.get(&obs.landmark_id).copied();

                    if let Some(p) = i_pose {
                        let h_pp_block: Matrix6<f64> = j_pose.transpose() * j_pose;
                        let b_p_block: Vector6<f64> = j_pose.transpose() * residual;
                        for r in 0..6 {
                            for c in 0..6 {
                                h_pp[(p * 6 + r, p * 6 + c)] += w * h_pp_block[(r, c)];
                            }
                            b_p[p * 6 + r] += w * b_p_block[r];
                        }
                    }
                    if let Some(l) = i_lm {
                        let h_ll_block: Matrix3<f64> = j_lm.transpose() * j_lm;
                        let b_l_block: Vector3<f64> = j_lm.transpose() * residual;
                        landmarks[l].h_ll += w * h_ll_block;
                        landmarks[l].b_l += w * b_l_block;
                    }
                    if let (Some(p), Some(l)) = (i_pose, i_lm) {
                        let cross: Matrix6x3<f64> = j_pose.transpose() * j_lm;
                        landmarks[l].cross.push((p, w * cross));
                    }
                }
            }
        }
    }

    // Calibrated non-rectified stereo: four residual rows `(u_l, v_l, u_r,
    // v_r)`. The helper composes `T_right<-left` before right projection and
    // stacks both cameras' Jacobians into the same pose/landmark blocks, so
    // the Schur structure remains unchanged.
    let general_offset = ba.observations.len() + ba.stereo_observations.len();
    for (index, obs) in ba.general_stereo_observations.iter().enumerate() {
        let pose = &ba.poses[&obs.keyframe_id];
        let point = &ba.landmarks[&obs.landmark_id];
        let Some((residual, j_pose, j_lm)) =
            general_stereo_residual_jacobians(intrinsics, obs, pose, point)
        else {
            continue;
        };
        let s = residual.norm_squared();
        let w =
            kernel.weight(s) * gnc_weights.map_or(1.0, |weights| weights[general_offset + index]);
        let i_pose = pose_index.get(&obs.keyframe_id).copied();
        let i_lm = landmark_index.get(&obs.landmark_id).copied();
        if let Some(p) = i_pose {
            let h_pp_block: Matrix6<f64> = j_pose.transpose() * j_pose;
            let b_p_block: Vector6<f64> = j_pose.transpose() * residual;
            for r in 0..6 {
                for c in 0..6 {
                    h_pp[(p * 6 + r, p * 6 + c)] += w * h_pp_block[(r, c)];
                }
                b_p[p * 6 + r] += w * b_p_block[r];
            }
        }
        if let Some(l) = i_lm {
            let h_ll_block: Matrix3<f64> = j_lm.transpose() * j_lm;
            let b_l_block: Vector3<f64> = j_lm.transpose() * residual;
            landmarks[l].h_ll += w * h_ll_block;
            landmarks[l].b_l += w * b_l_block;
        }
        if let (Some(p), Some(l)) = (i_pose, i_lm) {
            let cross: Matrix6x3<f64> = j_pose.transpose() * j_lm;
            landmarks[l].cross.push((p, w * cross));
        }
    }

    // Arbitrary calibrated rig sensor: two residual rows tied directly to
    // the shared body pose through the fixed `sensor <- rig` extrinsic.
    let rig_offset = general_offset + ba.general_stereo_observations.len();
    for (index, observation) in ba.rig_observations.iter().enumerate() {
        let pose = &ba.poses[&observation.keyframe_id];
        let point = &ba.landmarks[&observation.landmark_id];
        let Some((residual, j_pose, j_lm)) = rig_residual_jacobians(observation, pose, point)
        else {
            continue;
        };
        let squared_residual = residual.norm_squared();
        let weight = kernel.weight(squared_residual)
            * gnc_weights.map_or(1.0, |weights| weights[rig_offset + index]);
        let pose_slot = pose_index.get(&observation.keyframe_id).copied();
        let landmark_slot = landmark_index.get(&observation.landmark_id).copied();
        if let Some(slot) = pose_slot {
            let hessian: Matrix6<f64> = j_pose.transpose() * j_pose;
            let gradient: Vector6<f64> = j_pose.transpose() * residual;
            for row in 0..6 {
                for column in 0..6 {
                    h_pp[(slot * 6 + row, slot * 6 + column)] += weight * hessian[(row, column)];
                }
                b_p[slot * 6 + row] += weight * gradient[row];
            }
        }
        if let Some(slot) = landmark_slot {
            let hessian: Matrix3<f64> = j_lm.transpose() * j_lm;
            let gradient: Vector3<f64> = j_lm.transpose() * residual;
            landmarks[slot].h_ll += weight * hessian;
            landmarks[slot].b_l += weight * gradient;
        }
        if let (Some(pose_slot), Some(landmark_slot)) = (pose_slot, landmark_slot) {
            let cross: Matrix6x3<f64> = j_pose.transpose() * j_lm;
            landmarks[landmark_slot]
                .cross
                .push((pose_slot, weight * cross));
        }
    }

    // Optional per-pose absolute-position prior. Residual per
    // observation: r = C_w − target, where C_w = −Rᵀt is the world
    // camera centre. From the gravity-prior derivation:
    //   d C_w / d ρ = −I,   d C_w / d ω = [C_w]_×.
    // Per-axis stiffness is applied as a 3×3 diagonal scaling so a
    // zero entry collapses that row to zero — i.e. drops it from the
    // normal equations entirely.
    if let Some(prior) = &ba.position_prior {
        for obs in &prior.observations {
            let Some(&p) = pose_index.get(&obs.keyframe_id) else {
                continue;
            };
            let Some(pose) = ba.poses.get(&obs.keyframe_id) else {
                continue;
            };
            let c_world = pose.camera_center_world();
            let r_vec: Vector3<f64> = c_world - obs.camera_center_world;
            let c_skew = skew(&c_world.coords);
            let mut j_pose = nalgebra::Matrix3x6::<f64>::zeros();
            j_pose
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&-Matrix3::identity());
            if prior.couple_rotation {
                j_pose.fixed_view_mut::<3, 3>(0, 3).copy_from(&c_skew);
            }
            // Apply √w on each residual row so JᵀJ and Jᵀr come out
            // axis-weighted exactly as in the cost.
            let sqrt_w_x = obs.axis_weights.x.max(0.0).sqrt();
            let sqrt_w_y = obs.axis_weights.y.max(0.0).sqrt();
            let sqrt_w_z = obs.axis_weights.z.max(0.0).sqrt();
            let mut weighted_j = j_pose;
            weighted_j.row_mut(0).scale_mut(sqrt_w_x);
            weighted_j.row_mut(1).scale_mut(sqrt_w_y);
            weighted_j.row_mut(2).scale_mut(sqrt_w_z);
            let mut weighted_r = r_vec;
            weighted_r.x *= sqrt_w_x;
            weighted_r.y *= sqrt_w_y;
            weighted_r.z *= sqrt_w_z;
            let h_pp_block: Matrix6<f64> = weighted_j.transpose() * weighted_j;
            let b_p_block: Vector6<f64> = weighted_j.transpose() * weighted_r;
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(p * 6 + rr, p * 6 + cc)] += h_pp_block[(rr, cc)];
                }
                b_p[p * 6 + rr] += b_p_block[rr];
            }
        }
    }

    // Pairwise relative-pose factors. For each factor:
    //   r = log(meas⁻¹ · T_j · T_iⁱ)
    //   ∂r/∂δ_j =  Ad(T_i)   (6×6)
    //   ∂r/∂δ_i = -Ad(T_i)
    // Hessian/gradient contributions per factor (with scalar weight w):
    //   H[j,j] += w · AdᵀAd     b[j] += w · Adᵀ · r
    //   H[i,i] += w · AdᵀAd     b[i] -= w · Adᵀ · r
    //   H[j,i] -= w · AdᵀAd     H[i,j] = (H[j,i])ᵀ
    // Diagonal block AdᵀAd is shared (same magnitude on both sides);
    // the off-diagonal block carries a minus sign because the from
    // Jacobian is the negative of the to Jacobian.
    for factor in &ba.pairwise_pose_factors {
        let (Some(t_from_pose), Some(t_to_pose)) = (
            ba.poses.get(&factor.keyframe_id_from),
            ba.poses.get(&factor.keyframe_id_to),
        ) else {
            continue;
        };
        let t_from = &t_from_pose.world_to_camera;
        let t_to = &t_to_pose.world_to_camera;
        let predicted = t_to.compose(&t_from.inverse());
        let r = factor
            .measurement
            .world_to_camera
            .inverse()
            .compose(&predicted)
            .log();
        let ad_from = t_from.adjoint();
        let ata: Matrix6<f64> = ad_from.transpose() * ad_from;
        let atr: Vector6<f64> = ad_from.transpose() * r;
        let w = factor.weight;
        let i_to = pose_index.get(&factor.keyframe_id_to).copied();
        let i_from = pose_index.get(&factor.keyframe_id_from).copied();
        if let Some(j) = i_to {
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(j * 6 + rr, j * 6 + cc)] += w * ata[(rr, cc)];
                }
                b_p[j * 6 + rr] += w * atr[rr];
            }
        }
        if let Some(i) = i_from {
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(i * 6 + rr, i * 6 + cc)] += w * ata[(rr, cc)];
                }
                b_p[i * 6 + rr] -= w * atr[rr];
            }
        }
        if let (Some(j), Some(i)) = (i_to, i_from) {
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(j * 6 + rr, i * 6 + cc)] -= w * ata[(rr, cc)];
                    h_pp[(i * 6 + rr, j * 6 + cc)] -= w * ata[(cc, rr)];
                }
            }
        }
    }

    // Optional gravity-alignment prior on every non-fixed pose.
    //
    // Residual per pose: r = R_wc · g_world − g_camera_observed (3-vec).
    // Under right perturbation T_new = T_old · exp([ρ; ω]):
    //     R_new = R_old · R(ω) ≈ R_old · (I + [ω]_×)
    //     R_new · g_w ≈ R_old · g_w − R_old · [g_w]_× · ω
    // so the Jacobian w.r.t. xi = [ρ; ω] is J = [0_3×3 | −R_old · [g_w]_×].
    // Translation does not appear, which leaves the gauge of horizontal
    // translation completely unconstrained — this prior fixes rotation
    // ambiguity only, not translation drift.
    if let Some(prior) = &ba.gravity_prior {
        for (&pose_id, pose) in &ba.poses {
            let Some(&p) = pose_index.get(&pose_id) else {
                continue;
            };
            let r_mat = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let r_vec: Vector3<f64> = r_mat * prior.g_world - prior.g_camera_observed;
            let g_skew = skew(&prior.g_world);
            let mut j_pose = nalgebra::Matrix3x6::<f64>::zeros();
            // ∂r/∂ρ = 0, ∂r/∂ω = −R · [g_w]_×.
            j_pose
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-(r_mat * g_skew)));
            let h_pp_block: Matrix6<f64> = j_pose.transpose() * j_pose;
            let b_p_block: Vector6<f64> = j_pose.transpose() * r_vec;
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(p * 6 + rr, p * 6 + cc)] += prior.weight * h_pp_block[(rr, cc)];
                }
                b_p[p * 6 + rr] += prior.weight * b_p_block[rr];
            }
        }
    }

    // Per-keyframe gravity-alignment prior. Same residual / Jacobian
    // shape as the global gravity prior above, but the observation is
    // sourced per-keyframe (and pose entries not in the observations
    // list contribute nothing).
    if let Some(prior) = &ba.per_pose_gravity_prior {
        let g_skew = skew(&prior.g_world);
        for obs in &prior.observations {
            let Some(pose) = ba.poses.get(&obs.keyframe_id) else {
                continue;
            };
            let Some(&p) = pose_index.get(&obs.keyframe_id) else {
                continue;
            };
            // Effective stiffness combines the prior's global scale and the
            // per-observation multiplier; zero per-obs weight mutes the
            // observation without removing its slot.
            let stiffness = prior.weight * obs.weight;
            if stiffness == 0.0 {
                continue;
            }
            let r_mat = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let r_vec: Vector3<f64> = r_mat * prior.g_world - obs.g_camera_observed;
            let mut j_pose = nalgebra::Matrix3x6::<f64>::zeros();
            j_pose
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-(r_mat * g_skew)));
            let h_pp_block: Matrix6<f64> = j_pose.transpose() * j_pose;
            let b_p_block: Vector6<f64> = j_pose.transpose() * r_vec;
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(p * 6 + rr, p * 6 + cc)] += stiffness * h_pp_block[(rr, cc)];
                }
                b_p[p * 6 + rr] += stiffness * b_p_block[rr];
            }
        }
    }

    // Forster 2017 IMU pre-integration factors.
    //
    // The factor binds (pose, velocity) at keyframe i and j via the
    // gravity-compensated relative measurement (ΔR, Δv, Δp). Residual
    // and Jacobians follow Forster eq. 45-47 with the BA right-perturbation
    // convention (T_new = T_old · exp([ρ; ω])):
    //
    //   r_R = log(ΔR.T · R_iᵀ · R_j)            (Forster's R_i = R_wbᵢ)
    //   r_v = R_bwᵢ · (v_j − v_i − g·Δt) − Δv
    //   r_p = R_bwᵢ · (B_j − B_i − v_i·Δt − ½ g Δt²) − Δp
    //
    // Right-perturbation Jacobians (ρ = translation perturbation, ω =
    // rotation perturbation, both 3-vec; world body centre
    // B = −Rᵀt so ∂B/∂ρ = −I, ∂B/∂ω = [B]×). Since
    // T_bw = T_bc T_cw, a right perturbation of T_cw is the same right
    // perturbation of T_bw, so no additional adjoint is required:
    //
    //   ∂r_R/∂ω_i =  Jr_inv(r_R) · R_bwⱼ
    //   ∂r_R/∂ω_j = −Jr_inv(r_R) · R_bwⱼ
    //   ∂r_v/∂ω_i = −R_bwᵢ · [v_j − v_i − g·Δt]×
    //   ∂r_v/∂v_i = −R_bwᵢ
    //   ∂r_v/∂v_j =  R_bwᵢ
    //   ∂r_p/∂ρ_i =  R_bwᵢ            ∂r_p/∂ρ_j = −R_bwᵢ
    //   ∂r_p/∂ω_i = −R_bwᵢ · [B_j − v_i·Δt − ½ g Δt²]×
    //   ∂r_p/∂ω_j =  R_bwᵢ · [B_j]×
    //   ∂r_p/∂v_i = −Δt · R_bwᵢ
    //
    // The 9-vector residual is stacked [r_R; r_v; r_p] with axis-wise
    // weights `sqrt(weight_rotation, weight_velocity, weight_position)`
    // applied as a per-block √w scaling so JᵀJ / Jᵀr come out
    // axis-weighted.
    for factor in &ba.imu_factors {
        let (Some(pose_i), Some(pose_j)) = (
            ba.poses.get(&factor.keyframe_id_from),
            ba.poses.get(&factor.keyframe_id_to),
        ) else {
            continue;
        };
        let (Some(v_i_w), Some(v_j_w)) = (
            ba.velocities.get(&factor.keyframe_id_from),
            ba.velocities.get(&factor.keyframe_id_to),
        ) else {
            continue;
        };
        let body_i = ba.imu_body_to_camera.compose(&pose_i.world_to_camera);
        let body_j = ba.imu_body_to_camera.compose(&pose_j.world_to_camera);
        let r_wc_i = body_i.rotation.to_rotation_matrix().into_inner();
        let r_wc_j = body_j.rotation.to_rotation_matrix().into_inner();
        let c_i: Vector3<f64> = body_i.inverse().translation;
        let c_j: Vector3<f64> = body_j.inverse().translation;
        let dt = factor.delta.delta_time;
        let g = factor.gravity_world;

        // Residual (use the same formulation as the factor's residual()
        // helper so the cost evaluation and Jacobian linearise at the
        // same point). When the "from" keyframe has a registered bias,
        // apply the first-order bias correction; otherwise fall back
        // to the un-corrected residual (the linearisation bias is
        // implicit in the integrated delta).
        let r_i_so3 = SO3::from_quaternion(body_i.rotation.inverse());
        let r_j_so3 = SO3::from_quaternion(body_j.rotation.inverse());
        let bias_for_factor = ba.biases.get(&factor.keyframe_id_from);
        let [r_rot, r_vel, r_pos] = if let Some(bias) = bias_for_factor {
            let bg: Vector3<f64> = bias.fixed_rows::<3>(0).into_owned();
            let ba_acc: Vector3<f64> = bias.fixed_rows::<3>(3).into_owned();
            factor.residual_with_bias_correction(
                &r_i_so3, &c_i, v_i_w, &r_j_so3, &c_j, v_j_w, &bg, &ba_acc,
            )
        } else {
            factor.residual(&r_i_so3, &c_i, v_i_w, &r_j_so3, &c_j, v_j_w)
        };

        // Full preintegration-covariance whitening when available; legacy
        // hand-tuned block weights remain the exact zero-covariance fallback.
        let whitener = factor.covariance_sqrt_information().unwrap_or_else(|| {
            let mut diagonal = nalgebra::SVector::<f64, 9>::zeros();
            diagonal
                .fixed_rows_mut::<3>(0)
                .fill(factor.weight_rotation.max(0.0).sqrt());
            diagonal
                .fixed_rows_mut::<3>(3)
                .fill(factor.weight_velocity.max(0.0).sqrt());
            diagonal
                .fixed_rows_mut::<3>(6)
                .fill(factor.weight_position.max(0.0).sqrt());
            crate::imu_preintegration::Matrix9::from_diagonal(&diagonal)
        });

        let q_diff = v_j_w - v_i_w - g * dt;
        let q_pos_i = c_j - v_i_w * dt - 0.5 * g * dt * dt;
        let jr_inv = right_jacobian_inverse_so3(&r_rot);
        let jr_inv_rwc_j = jr_inv * r_wc_j;

        // Build the 9×N Jacobian blocks per side, with √w scaling
        // baked into each row block so the JᵀJ / Jᵀr accumulation does
        // the right axis-weighting automatically.
        // J_pose_i: 9×6, columns [ρ_i | ω_i].
        let mut j_pose_i = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U6, _>::zeros();
        // r_R block: ω_i column = Jr_inv · R_wc_j.
        j_pose_i
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&jr_inv_rwc_j);
        // r_v block: ω_i column = −R_wcᵢ · [q_diff]×.
        j_pose_i
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(-r_wc_i * skew(&q_diff)));
        // r_p block: ρ_i = R_wcᵢ, ω_i = −R_wcᵢ · [q_pos_i]×.
        j_pose_i.fixed_view_mut::<3, 3>(6, 0).copy_from(&r_wc_i);
        j_pose_i
            .fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(-r_wc_i * skew(&q_pos_i)));

        // J_pose_j: 9×6, columns [ρ_j | ω_j].
        let mut j_pose_j = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U6, _>::zeros();
        j_pose_j
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-jr_inv_rwc_j));
        // r_v has no R_j / p_j dependence (block stays zero).
        // r_p block: ρ_j = −R_wcᵢ, ω_j = R_wcᵢ · [C_j]×.
        j_pose_j.fixed_view_mut::<3, 3>(6, 0).copy_from(&(-r_wc_i));
        j_pose_j
            .fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(r_wc_i * skew(&c_j)));

        // J_vel_i / J_vel_j: 9×3 each.
        let mut j_vel_i = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U3, _>::zeros();
        j_vel_i.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-r_wc_i));
        j_vel_i
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(-dt * r_wc_i));
        let mut j_vel_j = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U3, _>::zeros();
        j_vel_j.fixed_view_mut::<3, 3>(3, 0).copy_from(&r_wc_i);

        // Residual before applying the common full-row whitening transform.
        let mut r_stack = nalgebra::SVector::<f64, 9>::zeros();
        r_stack.fixed_rows_mut::<3>(0).copy_from(&r_rot);
        r_stack.fixed_rows_mut::<3>(3).copy_from(&r_vel);
        r_stack.fixed_rows_mut::<3>(6).copy_from(&r_pos);

        // Bias Jacobian (9×6, columns [δb_g | δb_a]) at keyframe i:
        //
        //   ∂r_R/∂δb_g = −Jr⁻¹(r_R) · Exp(−r_R) · J_R_bg
        //   ∂r_R/∂δb_a = 0
        //   ∂r_v/∂δb_g = −J_v_bg                 ∂r_v/∂δb_a = −J_v_ba
        //   ∂r_p/∂δb_g = −J_p_bg                 ∂r_p/∂δb_a = −J_p_ba
        //
        // Forster eq. 159 (simplified by dropping the `Jr(J_R · δb)` factor,
        // which equals identity at the linearisation point and is ~I for
        // the small `|J_R · δb|` regime where biases live in practice).
        // The `J_*_b*` matrices are the bias Jacobians stored on the
        // pre-integrated delta (Forster eq. 35-39); see
        // [`crate::imu_preintegration`].
        let i_bias = bias_index.get(&factor.keyframe_id_from).copied();
        let j_bias_block = if i_bias.is_some() {
            let neg_r_rot_mat: Matrix3<f64> = nalgebra::Rotation3::from_scaled_axis(-r_rot).into();
            let lhs_rot = -jr_inv * neg_r_rot_mat;
            let mut j_bias = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U6, _>::zeros();
            j_bias
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(lhs_rot * factor.delta.j_rotation_bg));
            j_bias
                .fixed_view_mut::<3, 3>(3, 0)
                .copy_from(&(-factor.delta.j_velocity_bg));
            j_bias
                .fixed_view_mut::<3, 3>(3, 3)
                .copy_from(&(-factor.delta.j_velocity_ba));
            j_bias
                .fixed_view_mut::<3, 3>(6, 0)
                .copy_from(&(-factor.delta.j_position_bg));
            j_bias
                .fixed_view_mut::<3, 3>(6, 3)
                .copy_from(&(-factor.delta.j_position_ba));
            Some(j_bias)
        } else {
            None
        };

        let j_pose_i = whitener * j_pose_i;
        let j_pose_j = whitener * j_pose_j;
        let j_vel_i = whitener * j_vel_i;
        let j_vel_j = whitener * j_vel_j;
        let j_bias_block = j_bias_block.map(|jacobian| whitener * jacobian);
        let r_stack = whitener * r_stack;

        let i_pose = pose_index.get(&factor.keyframe_id_from).copied();
        let j_pose = pose_index.get(&factor.keyframe_id_to).copied();
        let i_vel = velocity_index.get(&factor.keyframe_id_from).copied();
        let j_vel = velocity_index.get(&factor.keyframe_id_to).copied();

        // Helper: accumulate block A^T B into h_pp at (row_block, col_block).
        // Compute the (6 or 3)-row / (6 or 3)-col block and accumulate.
        // (Done inline below per block pair.)

        if let Some(p) = i_pose {
            let blk: Matrix6<f64> = j_pose_i.transpose() * j_pose_i;
            let bblk: Vector6<f64> = j_pose_i.transpose() * r_stack;
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(p * 6 + rr, p * 6 + cc)] += blk[(rr, cc)];
                }
                b_p[p * 6 + rr] += bblk[rr];
            }
        }
        if let Some(p) = j_pose {
            let blk: Matrix6<f64> = j_pose_j.transpose() * j_pose_j;
            let bblk: Vector6<f64> = j_pose_j.transpose() * r_stack;
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(p * 6 + rr, p * 6 + cc)] += blk[(rr, cc)];
                }
                b_p[p * 6 + rr] += bblk[rr];
            }
        }
        if let (Some(pi), Some(pj)) = (i_pose, j_pose) {
            let blk: Matrix6<f64> = j_pose_i.transpose() * j_pose_j;
            for rr in 0..6 {
                for cc in 0..6 {
                    let v = blk[(rr, cc)];
                    h_pp[(pi * 6 + rr, pj * 6 + cc)] += v;
                    h_pp[(pj * 6 + cc, pi * 6 + rr)] += v;
                }
            }
        }
        if let Some(v) = i_vel {
            let blk: Matrix3<f64> = j_vel_i.transpose() * j_vel_i;
            let bblk: Vector3<f64> = j_vel_i.transpose() * r_stack;
            for rr in 0..3 {
                for cc in 0..3 {
                    h_pp[(vel_offset + v * 3 + rr, vel_offset + v * 3 + cc)] += blk[(rr, cc)];
                }
                b_p[vel_offset + v * 3 + rr] += bblk[rr];
            }
        }
        if let Some(v) = j_vel {
            let blk: Matrix3<f64> = j_vel_j.transpose() * j_vel_j;
            let bblk: Vector3<f64> = j_vel_j.transpose() * r_stack;
            for rr in 0..3 {
                for cc in 0..3 {
                    h_pp[(vel_offset + v * 3 + rr, vel_offset + v * 3 + cc)] += blk[(rr, cc)];
                }
                b_p[vel_offset + v * 3 + rr] += bblk[rr];
            }
        }
        if let (Some(vi), Some(vj)) = (i_vel, j_vel) {
            let blk: Matrix3<f64> = j_vel_i.transpose() * j_vel_j;
            for rr in 0..3 {
                for cc in 0..3 {
                    let v = blk[(rr, cc)];
                    h_pp[(vel_offset + vi * 3 + rr, vel_offset + vj * 3 + cc)] += v;
                    h_pp[(vel_offset + vj * 3 + cc, vel_offset + vi * 3 + rr)] += v;
                }
            }
        }
        if let (Some(pi), Some(vi)) = (i_pose, i_vel) {
            let blk: nalgebra::Matrix6x3<f64> = j_pose_i.transpose() * j_vel_i;
            for rr in 0..6 {
                for cc in 0..3 {
                    let v = blk[(rr, cc)];
                    h_pp[(pi * 6 + rr, vel_offset + vi * 3 + cc)] += v;
                    h_pp[(vel_offset + vi * 3 + cc, pi * 6 + rr)] += v;
                }
            }
        }
        if let (Some(pi), Some(vj)) = (i_pose, j_vel) {
            let blk: nalgebra::Matrix6x3<f64> = j_pose_i.transpose() * j_vel_j;
            for rr in 0..6 {
                for cc in 0..3 {
                    let v = blk[(rr, cc)];
                    h_pp[(pi * 6 + rr, vel_offset + vj * 3 + cc)] += v;
                    h_pp[(vel_offset + vj * 3 + cc, pi * 6 + rr)] += v;
                }
            }
        }
        if let (Some(pj), Some(vi)) = (j_pose, i_vel) {
            let blk: nalgebra::Matrix6x3<f64> = j_pose_j.transpose() * j_vel_i;
            for rr in 0..6 {
                for cc in 0..3 {
                    let v = blk[(rr, cc)];
                    h_pp[(pj * 6 + rr, vel_offset + vi * 3 + cc)] += v;
                    h_pp[(vel_offset + vi * 3 + cc, pj * 6 + rr)] += v;
                }
            }
        }
        if let (Some(pj), Some(vj)) = (j_pose, j_vel) {
            let blk: nalgebra::Matrix6x3<f64> = j_pose_j.transpose() * j_vel_j;
            for rr in 0..6 {
                for cc in 0..3 {
                    let v = blk[(rr, cc)];
                    h_pp[(pj * 6 + rr, vel_offset + vj * 3 + cc)] += v;
                    h_pp[(vel_offset + vj * 3 + cc, pj * 6 + rr)] += v;
                }
            }
        }

        // Bias contributions: J_bias_i (9×6) accumulates against itself
        // and against every other side (pose_i, pose_j, vel_i, vel_j).
        if let (Some(b), Some(jb)) = (i_bias, j_bias_block) {
            let blk: Matrix6<f64> = jb.transpose() * jb;
            let bblk: Vector6<f64> = jb.transpose() * r_stack;
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(bias_offset + b * 6 + rr, bias_offset + b * 6 + cc)] += blk[(rr, cc)];
                }
                b_p[bias_offset + b * 6 + rr] += bblk[rr];
            }
            if let Some(p) = i_pose {
                let cross: Matrix6<f64> = j_pose_i.transpose() * jb;
                for rr in 0..6 {
                    for cc in 0..6 {
                        let v = cross[(rr, cc)];
                        h_pp[(p * 6 + rr, bias_offset + b * 6 + cc)] += v;
                        h_pp[(bias_offset + b * 6 + cc, p * 6 + rr)] += v;
                    }
                }
            }
            if let Some(p) = j_pose {
                let cross: Matrix6<f64> = j_pose_j.transpose() * jb;
                for rr in 0..6 {
                    for cc in 0..6 {
                        let v = cross[(rr, cc)];
                        h_pp[(p * 6 + rr, bias_offset + b * 6 + cc)] += v;
                        h_pp[(bias_offset + b * 6 + cc, p * 6 + rr)] += v;
                    }
                }
            }
            if let Some(v) = i_vel {
                let cross: nalgebra::Matrix3x6<f64> = j_vel_i.transpose() * jb;
                for rr in 0..3 {
                    for cc in 0..6 {
                        let val = cross[(rr, cc)];
                        h_pp[(vel_offset + v * 3 + rr, bias_offset + b * 6 + cc)] += val;
                        h_pp[(bias_offset + b * 6 + cc, vel_offset + v * 3 + rr)] += val;
                    }
                }
            }
            if let Some(v) = j_vel {
                let cross: nalgebra::Matrix3x6<f64> = j_vel_j.transpose() * jb;
                for rr in 0..3 {
                    for cc in 0..6 {
                        let val = cross[(rr, cc)];
                        h_pp[(vel_offset + v * 3 + rr, bias_offset + b * 6 + cc)] += val;
                        h_pp[(bias_offset + b * 6 + cc, vel_offset + v * 3 + rr)] += val;
                    }
                }
            }
        }
    }

    // Bias random-walk factors: residual `r = b_j − b_i`, Jacobian
    // `J_i = −I, J_j = I`, with separate gyro/accel weights. The
    // factor only touches the bias slots (no pose / velocity coupling),
    // so `Jᵀ J` lands as `+w · I` on each diagonal bias block, `−w · I`
    // on the off-diagonal cross block, and `Jᵀ r` distributes
    // `−w · r` to `b_i` and `+w · r` to `b_j`.
    for factor in &ba.bias_random_walk_factors {
        let (Some(b_i), Some(b_j)) = (
            ba.biases.get(&factor.keyframe_id_from),
            ba.biases.get(&factor.keyframe_id_to),
        ) else {
            continue;
        };
        let r: Vector6<f64> = b_j - b_i;
        let weights = [
            factor.weight_gyro.max(0.0),
            factor.weight_gyro.max(0.0),
            factor.weight_gyro.max(0.0),
            factor.weight_accel.max(0.0),
            factor.weight_accel.max(0.0),
            factor.weight_accel.max(0.0),
        ];
        if weights.iter().all(|weight| *weight <= 0.0) {
            continue;
        }
        let i_slot = bias_index.get(&factor.keyframe_id_from).copied();
        let j_slot = bias_index.get(&factor.keyframe_id_to).copied();
        if let Some(i) = i_slot {
            for k in 0..6 {
                let w = weights[k];
                h_pp[(bias_offset + i * 6 + k, bias_offset + i * 6 + k)] += w;
                b_p[bias_offset + i * 6 + k] += -w * r[k];
            }
        }
        if let Some(j) = j_slot {
            for k in 0..6 {
                let w = weights[k];
                h_pp[(bias_offset + j * 6 + k, bias_offset + j * 6 + k)] += w;
                b_p[bias_offset + j * 6 + k] += w * r[k];
            }
        }
        if let (Some(i), Some(j)) = (i_slot, j_slot) {
            for k in 0..6 {
                let w = weights[k];
                h_pp[(bias_offset + i * 6 + k, bias_offset + j * 6 + k)] += -w;
                h_pp[(bias_offset + j * 6 + k, bias_offset + i * 6 + k)] += -w;
            }
        }
    }

    // Dense FEJ navigation prior. Its Jacobian is frozen to identity in the
    // stored right-perturbation coordinates; only the residual displacement
    // from the fixed reference changes. Fixed variables still affect the
    // current gradient through H*delta, but receive no solver rows.
    if let Some(prior) = &ba.navigation_state_prior {
        if let Some(delta) = navigation_prior_delta(ba, prior) {
            let current_gradient = &prior.information * delta + &prior.gradient;
            let mut global_indices: Vec<Option<usize>> =
                Vec::with_capacity(prior.keyframe_ids.len() * 15);
            for id in &prior.keyframe_ids {
                let pose_slot = pose_index.get(id).copied();
                for component in 0..6 {
                    global_indices.push(pose_slot.map(|slot| slot * 6 + component));
                }
                let velocity_slot = velocity_index.get(id).copied();
                for component in 0..3 {
                    global_indices
                        .push(velocity_slot.map(|slot| vel_offset + slot * 3 + component));
                }
                let bias_slot = bias_index.get(id).copied();
                for component in 0..6 {
                    global_indices.push(bias_slot.map(|slot| bias_offset + slot * 6 + component));
                }
            }
            for (prior_row, global_row) in global_indices.iter().enumerate() {
                let Some(global_row) = *global_row else {
                    continue;
                };
                b_p[global_row] += current_gradient[prior_row];
                for (prior_col, global_col) in global_indices.iter().enumerate() {
                    let Some(global_col) = *global_col else {
                        continue;
                    };
                    h_pp[(global_row, global_col)] += prior.information[(prior_row, prior_col)];
                }
            }
        }
    }

    NormalEquationsBa {
        h_pp,
        b_p,
        landmarks,
    }
}

/// Project a normal-equation system onto the subspace in which selected pose
/// rotations are fixed.  Pose blocks intentionally remain six-dimensional so
/// the existing Schur and linear-solver layouts are unchanged.  The
/// constrained rotation rows/columns are made identity rows with zero right
/// hand side; this is equivalent to removing those variables and is also
/// well-defined for an undamped Gauss--Newton solve.  Landmark cross blocks
/// are cleared for the same rows so the translation/landmark solve cannot
/// use a discarded rotation update.
fn constrain_fixed_pose_rotations(
    fixed_rotations: &BTreeSet<u64>,
    pose_index: &BTreeMap<u64, usize>,
    system: &mut NormalEquationsBa,
) {
    for &image_id in fixed_rotations {
        let Some(&pose_slot) = pose_index.get(&image_id) else {
            continue;
        };
        for component in 3..6 {
            let index = pose_slot * 6 + component;
            match &mut system.h_pp {
                CameraHessian::Dense(matrix) => {
                    for column in 0..matrix.ncols() {
                        matrix[(index, column)] = 0.0;
                    }
                    for row in 0..matrix.nrows() {
                        matrix[(row, index)] = 0.0;
                    }
                    matrix[(index, index)] = 1.0;
                }
                CameraHessian::PoseDiagonal(blocks) => {
                    let local = index % 6;
                    for column in 0..6 {
                        blocks[pose_slot][(local, column)] = 0.0;
                    }
                    for row in 0..6 {
                        blocks[pose_slot][(row, local)] = 0.0;
                    }
                    blocks[pose_slot][(local, local)] = 1.0;
                }
            }
            system.b_p[index] = 0.0;
            for landmark in &mut system.landmarks {
                for (landmark_pose_slot, cross) in &mut landmark.cross {
                    if *landmark_pose_slot == pose_slot {
                        for column in 0..3 {
                            cross[(component, column)] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

struct MatrixFreeStepOutcome {
    delta_poses: DVector<f64>,
    delta_landmarks: DVector<f64>,
    diagnostics: MatrixFreeBaIterationStats,
    restart_diagnostics: Option<MatrixFreeBaRestartIterationStats>,
    adaptive_prediction: Option<Result<f64, &'static str>>,
    quality: Option<MatrixFreeStepQuality>,
}

struct MatrixFreeStepError {
    diagnostic: String,
    pcg_iterations: Option<usize>,
    pcg_residual_norm: Option<f64>,
    pcg_target: Option<f64>,
    restart_diagnostics: Option<MatrixFreeBaRestartIterationStats>,
}

// This private error carries the optional per-solve restart diagnostic along
// with the existing textual PCG failure.  Keeping it inline avoids an
// allocation on the successful/default path; the large-error lint is not an
// API concern here.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn solve_matrix_free_step(
    system: &NormalEquationsBa,
    lambda: f64,
    options: MatrixFreeBaOptions,
    restart_limit: usize,
    collect_restart_diagnostics: bool,
    scaling_state: Option<&ColumnEquilibrationState>,
    schur_debug_context: Option<SchurBlockDebugContext<'_>>,
    collect_adaptive_prediction: bool,
) -> Result<MatrixFreeStepOutcome, MatrixFreeStepError> {
    let to_step_error = |error: implicit_schur::ImplicitSchurError| {
        let (pcg_iterations, pcg_residual_norm, pcg_target) = error.diagnostics();
        MatrixFreeStepError {
            diagnostic: format!("{error:?}"),
            pcg_iterations,
            pcg_residual_norm,
            pcg_target,
            restart_diagnostics: collect_restart_diagnostics.then(|| {
                MatrixFreeBaRestartIterationStats {
                    iteration: 0,
                    pcg_iterations,
                    true_residual_rechecks: 0,
                    failed_true_residual_rechecks: 0,
                    restarts: 0,
                    terminal_failure: Some(format!("{error:?}")),
                }
            }),
        }
    };
    let operator = match schur_debug_context {
        Some(context) => {
            implicit_schur::ImplicitSchurOperator::new_with_debug(system, lambda, Some(context))
        }
        None => implicit_schur::ImplicitSchurOperator::new(system, lambda),
    }
    .map_err(to_step_error)?;
    let pcg_options = implicit_schur::PcgOptions {
        max_iterations: options.max_pcg_iterations,
        relative_tolerance: options.pcg_relative_tolerance,
        absolute_tolerance: options.pcg_absolute_tolerance,
    };
    let (pcg, restart_diagnostics) = if restart_limit == 0 && !collect_restart_diagnostics {
        (
            operator
                .solve_pcg(operator.rhs(), pcg_options)
                .map_err(to_step_error)?,
            None,
        )
    } else {
        let run = operator
            .solve_pcg_with_restart(operator.rhs(), pcg_options, restart_limit)
            .map_err(|failure| {
                let implicit_schur::PcgSolveFailure { error, diagnostics } = failure;
                let (pcg_iterations, pcg_residual_norm, pcg_target) = error.diagnostics();
                MatrixFreeStepError {
                    diagnostic: format!("{error:?}"),
                    pcg_iterations,
                    pcg_residual_norm,
                    pcg_target,
                    restart_diagnostics: Some(MatrixFreeBaRestartIterationStats {
                        iteration: 0,
                        pcg_iterations: diagnostics.pcg_iterations,
                        true_residual_rechecks: diagnostics.true_residual_rechecks,
                        failed_true_residual_rechecks: diagnostics.failed_true_residual_rechecks,
                        restarts: diagnostics.restarts,
                        terminal_failure: Some(format!("{error:?}")),
                    }),
                }
            })?;
        (run.result, Some(run.diagnostics))
    };
    let mut delta_poses = pcg.solution;
    let mut delta_landmarks = operator.complete_delta(&delta_poses).map_err(|error| {
        if let Some(diagnostics) = restart_diagnostics {
            let mut restart_diagnostics: MatrixFreeBaRestartIterationStats = diagnostics.into();
            restart_diagnostics.terminal_failure = Some(format!("{error:?}"));
            MatrixFreeStepError {
                diagnostic: format!("{error:?}"),
                pcg_iterations: Some(pcg.iterations),
                pcg_residual_norm: Some(pcg.residual_norm),
                pcg_target: Some(pcg.target),
                restart_diagnostics: Some(restart_diagnostics),
            }
        } else {
            to_step_error(error)
        }
    })?;
    let adaptive_prediction = collect_adaptive_prediction
        .then(|| matrix_free_undamped_prediction(system, &delta_poses, &delta_landmarks));
    let quality = ba_lm_step_quality_debug_enabled().then(|| {
        matrix_free_step_quality(
            system,
            lambda,
            &delta_poses,
            &delta_landmarks,
            scaling_state,
        )
    });
    if let Some(scaling_state) = scaling_state {
        if let Err(error) = scaling_state.unscale_deltas(&mut delta_poses, &mut delta_landmarks) {
            let restart_diagnostics = restart_diagnostics.map(|diagnostics| {
                let mut stats: MatrixFreeBaRestartIterationStats = diagnostics.into();
                stats.terminal_failure = Some(format!("column scaling unscale failed: {error}"));
                stats
            });
            return Err(MatrixFreeStepError {
                diagnostic: format!("column scaling unscale failed: {error}"),
                pcg_iterations: Some(pcg.iterations),
                pcg_residual_norm: Some(pcg.residual_norm),
                pcg_target: Some(pcg.target),
                restart_diagnostics,
            });
        }
    }
    Ok(MatrixFreeStepOutcome {
        delta_poses,
        delta_landmarks,
        diagnostics: MatrixFreeBaIterationStats {
            iteration: 0,
            pcg_iterations: Some(pcg.iterations),
            pcg_residual_norm: Some(pcg.residual_norm),
            pcg_target: Some(pcg.target),
            pcg_failure: None,
        },
        restart_diagnostics: restart_diagnostics.map(Into::into),
        adaptive_prediction,
        quality,
    })
}

#[allow(clippy::too_many_arguments)]
fn solve_step(
    system: &mut NormalEquationsBa,
    p_count: usize,
    l_count: usize,
    v_count: usize,
    b_count: usize,
    lambda: f64,
    linear_solver: LinearSolver,
    parallel: bool,
    block_symbolic_cache: &mut Option<crate::block_cholesky::BlockSymbolic>,
) -> Result<(DVector<f64>, DVector<f64>), BaError> {
    // Landmark-only BA: H_LL is block-diagonal so each landmark gets an
    // independent 3×3 solve. No Schur complement needed. Every landmark's
    // solve is independent and writes only its own 3 rows of `delta_l`, so
    // the parallel path is a direct embarrassingly-parallel dispatch with no
    // merge step: bit-identical to the serial loop below at any thread
    // count.
    if p_count == 0 && v_count == 0 && b_count == 0 {
        let mut delta_l = DVector::<f64>::zeros(l_count * 3);
        if parallel && l_count >= PARALLEL_MIN_LANDMARKS {
            use rayon::prelude::*;
            let solved: Result<Vec<Vector3<f64>>, BaError> = system
                .landmarks
                .par_iter()
                .map(|landmark| {
                    let mut h_ll = landmark.h_ll;
                    if lambda > 0.0 {
                        h_ll[(0, 0)] += lambda;
                        h_ll[(1, 1)] += lambda;
                        h_ll[(2, 2)] += lambda;
                    }
                    let h_ll_inv = h_ll.try_inverse().ok_or(BaError::SingularSystem)?;
                    Ok(-(h_ll_inv * landmark.b_l))
                })
                .collect();
            for (l, dl) in solved?.into_iter().enumerate() {
                for k in 0..3 {
                    delta_l[l * 3 + k] = dl[k];
                }
            }
        } else {
            for (l, landmark) in system.landmarks.iter().enumerate() {
                let mut h_ll = landmark.h_ll;
                if lambda > 0.0 {
                    h_ll[(0, 0)] += lambda;
                    h_ll[(1, 1)] += lambda;
                    h_ll[(2, 2)] += lambda;
                }
                let h_ll_inv = h_ll.try_inverse().ok_or(BaError::SingularSystem)?;
                let dl: Vector3<f64> = -(h_ll_inv * landmark.b_l);
                for k in 0..3 {
                    delta_l[l * 3 + k] = dl[k];
                }
            }
        }
        return Ok((DVector::<f64>::zeros(0), delta_l));
    }

    if matches!(system.h_pp, CameraHessian::PoseDiagonal(_)) {
        debug_assert_eq!(v_count, 0);
        debug_assert_eq!(b_count, 0);
        debug_assert_eq!(linear_solver, LinearSolver::Sparse);
        let CameraHessian::PoseDiagonal(diagonal) =
            std::mem::replace(&mut system.h_pp, CameraHessian::pose_diagonal(0))
        else {
            unreachable!();
        };
        return solve_step_pose_blocks(
            system,
            diagonal,
            p_count,
            l_count,
            lambda,
            block_symbolic_cache,
        );
    }

    // λ damping on the joint pose+velocity+bias diagonal. The first 6P
    // rows are pose perturbations; the next 3V are velocity
    // perturbations; the final 6B are bias perturbations.
    let pose_dim = p_count * 6;
    let total_dim = pose_dim + v_count * 3 + b_count * 6;
    // `h_pp` is not needed after this solve step. Move it into the reduced
    // system instead of retaining the original dense matrix alongside an
    // equally-sized Schur copy. The empty replacement preserves the
    // `NormalEquationsBa` layout for the rest of this function, which only
    // reads landmarks and b_p after this point.
    let CameraHessian::Dense(mut s) =
        std::mem::replace(&mut system.h_pp, CameraHessian::Dense(DMatrix::zeros(0, 0)))
    else {
        unreachable!();
    };
    log_process_memory("ba-after-reduced-system-take");
    if lambda > 0.0 {
        for k in 0..total_dim {
            s[(k, k)] += lambda;
        }
    }
    let mut b_reduced = -&system.b_p;

    // Per-landmark Schur reduction. Each landmark contributes:
    //   S -= H_PL_l · H_LL_l^{-1} · H_PL_l^T
    //   b_S = -b_P + H_PL_l · H_LL_l^{-1} · b_l
    // (the RHS starts at -b_P, so each landmark contributes with a plus
    // sign). Both updates only touch the rows/cols of S corresponding to poses
    // that observed this landmark, so we never materialize the full H_PL.
    // Flag-gated, work-gated parallel reduction (see the module's
    // "Parallelism" section): bit-identical to the plain loop below at any
    // thread count / chunk size.
    let (h_ll_inv_cache, b_l_cache): (Vec<Option<Matrix3<f64>>>, Vec<Vector3<f64>>) =
        if parallel && system.landmarks.len() >= PARALLEL_MIN_LANDMARKS {
            schur_reduce_parallel(system, lambda, &mut s, &mut b_reduced)
        } else {
            let mut h_ll_inv_cache: Vec<Option<Matrix3<f64>>> =
                Vec::with_capacity(system.landmarks.len());
            let mut b_l_cache: Vec<Vector3<f64>> = Vec::with_capacity(system.landmarks.len());
            for landmark in &system.landmarks {
                let mut h_ll = landmark.h_ll;
                if lambda > 0.0 {
                    h_ll[(0, 0)] += lambda;
                    h_ll[(1, 1)] += lambda;
                    h_ll[(2, 2)] += lambda;
                }
                let h_ll_inv = h_ll.try_inverse();
                h_ll_inv_cache.push(h_ll_inv);
                b_l_cache.push(landmark.b_l);
                let h_ll_inv = match h_ll_inv {
                    Some(h) => h,
                    None => continue,
                };
                // Update S:
                //   for (p, A) in cross, for (q, B) in cross:
                //     S[p, q] -= A · h_ll_inv · B^T
                for (p, a) in &landmark.cross {
                    // Precompute A · h_ll_inv (6×3) once per outer pose.
                    let a_h: Matrix6x3<f64> = a * h_ll_inv;
                    for (q, b) in &landmark.cross {
                        let block: Matrix6<f64> = a_h * b.transpose();
                        for r in 0..6 {
                            for c in 0..6 {
                                s[(p * 6 + r, q * 6 + c)] -= block[(r, c)];
                            }
                        }
                    }
                    // Update reduced rhs. Schur derivation:
                    //   S · δ_p = -g_p + H_PL · H_LL^{-1} · g_l,
                    // so each landmark `l` contributes `+ A · h_ll_inv · b_l`
                    // to `b_reduced` (which already starts at `-g_p = -b_p`).
                    let upd: Vector6<f64> = a_h * landmark.b_l;
                    for k in 0..6 {
                        b_reduced[p * 6 + k] += upd[k];
                    }
                }
            }
            (h_ll_inv_cache, b_l_cache)
        };

    log_process_memory("ba-after-schur-reduction");

    // Solve the reduced pose system. The dense path uses Cholesky/LU; the
    // sparse path goes through CscCholesky on S as a CSC matrix.
    let delta_p = match linear_solver {
        LinearSolver::Dense => match solve_normal_equations(&s, &b_reduced) {
            Ok(d) => d,
            Err(PoseGraphError::SingularSystem) => return Err(BaError::SingularSystem),
            Err(_) => return Err(BaError::SingularSystem),
        },
        LinearSolver::Sparse => {
            let dim = total_dim;
            // Collect the structural nonzeros of the reduced system once. Both
            // back-ends consume the same triplet list; its sparsity pattern is
            // dictated by which pose pairs share a landmark observation.
            let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
            for c in 0..dim {
                for r in 0..dim {
                    let v = s[(r, c)];
                    if v != 0.0 {
                        triplets.push((r, c, v));
                    }
                }
            }
            // Sparse back-ends consume the triplets, not the dense reduced
            // matrix. Release S before allocating/factoring the sparse
            // representation so those large temporaries do not overlap.
            drop(std::mem::take(&mut s));
            log_process_memory("ba-sparse-before-factor");
            let rhs = DMatrix::from_column_slice(dim, 1, b_reduced.as_slice());

            // Pose and IMU-bias variables are 6×6 diagonal blocks, so when the
            // system carries no 3-DOF velocity blocks (the common pure-visual
            // BA case) the reduced matrix tiles cleanly into 6×6 blocks and the
            // block Cholesky back-end — the same one the pose graph uses —
            // factors it without the scalar gather/scatter bookkeeping. λ is
            // already folded into `s`, so we pass `lambda = 0`. Visual-inertial
            // systems interleave 3-DOF velocities, breaking the uniform tiling,
            // so they fall back to the scalar `CscCholesky` factorization.
            let sol = if v_count == 0 {
                crate::block_cholesky::solve_spd_block(&triplets, dim, 6, &rhs, 0.0)
                    .map_err(|_| BaError::SingularSystem)?
            } else {
                use nalgebra_sparse::{factorization::CscCholesky, CooMatrix, CscMatrix};
                let mut coo = CooMatrix::<f64>::new(dim, dim);
                for &(r, c, v) in &triplets {
                    coo.push(r, c, v);
                }
                let csc = CscMatrix::from(&coo);
                let chol = CscCholesky::factor(&csc).map_err(|_| BaError::SingularSystem)?;
                chol.solve(&rhs)
            };
            let delta = DVector::from_column_slice(sol.as_slice());
            drop(sol);
            drop(triplets);
            delta
        }
    };
    // The dense path still needs S until the solve returns; release it before
    // allocating/back-substituting the landmark updates. In the sparse path S
    // was already replaced with an empty matrix before factorization.
    drop(s);
    log_process_memory("ba-after-linear-solve");

    // Back-substitute landmark updates:
    //   δ_L[l] = h_ll_inv · (-b_l - Σ_p H_pl^T · δ_p)
    // Each landmark writes only its own 3 rows of `delta_l` and only reads
    // the (already solved, read-only from here on) `delta_p` and its own
    // cache entries, so — like the landmark-only branch above — the
    // parallel path is a direct embarrassingly-parallel dispatch with no
    // merge step: bit-identical to the serial loop below at any thread
    // count.
    let mut delta_l = DVector::<f64>::zeros(system.landmarks.len() * 3);
    if parallel && system.landmarks.len() >= PARALLEL_MIN_LANDMARKS {
        use rayon::prelude::*;
        delta_l
            .as_mut_slice()
            .par_chunks_mut(3)
            .zip(system.landmarks.par_iter())
            .zip(h_ll_inv_cache.par_iter())
            .zip(b_l_cache.par_iter())
            .for_each(|(((out, landmark), h_ll_inv), b_l)| {
                let Some(h_ll_inv) = h_ll_inv else {
                    return;
                };
                let mut acc = -*b_l;
                for (p, a) in &landmark.cross {
                    let dp = delta_p.fixed_rows::<6>(p * 6);
                    let dp_vec: Vector6<f64> = dp.into_owned();
                    let sub: Vector3<f64> = a.transpose() * dp_vec;
                    acc -= sub;
                }
                let dl = *h_ll_inv * acc;
                out.copy_from_slice(dl.as_slice());
            });
    } else {
        for (l, landmark) in system.landmarks.iter().enumerate() {
            let h_ll_inv = match h_ll_inv_cache[l] {
                Some(h) => h,
                None => continue,
            };
            let mut acc = -b_l_cache[l];
            for (p, a) in &landmark.cross {
                let dp = delta_p.fixed_rows::<6>(p * 6);
                let dp_vec: Vector6<f64> = dp.into_owned();
                let sub: Vector3<f64> = a.transpose() * dp_vec;
                acc -= sub;
            }
            let dl = h_ll_inv * acc;
            for k in 0..3 {
                delta_l[l * 3 + k] = dl[k];
            }
        }
    }

    Ok((delta_p, delta_l))
}

/// Pure-visual sparse Schur solve that never materializes the dense camera
/// Hessian. Blocks are stored by block-column and row-sorted within each
/// column, which makes the emitted scalar triplets match the legacy dense
/// column-major scan exactly while visiting structural blocks only.
fn solve_step_pose_blocks(
    system: &NormalEquationsBa,
    diagonal: Vec<Matrix6<f64>>,
    p_count: usize,
    l_count: usize,
    lambda: f64,
    block_symbolic_cache: &mut Option<crate::block_cholesky::BlockSymbolic>,
) -> Result<(DVector<f64>, DVector<f64>), BaError> {
    let dim = p_count * 6;
    let mut columns: Vec<BTreeMap<usize, Matrix6<f64>>> =
        (0..p_count).map(|_| BTreeMap::new()).collect();
    for (pose, mut block) in diagonal.into_iter().enumerate() {
        if lambda > 0.0 {
            for k in 0..6 {
                block[(k, k)] += lambda;
            }
        }
        columns[pose].insert(pose, block);
    }
    log_process_memory("ba-after-reduced-system-take");

    let mut b_reduced = -&system.b_p;
    let mut h_ll_inv_cache: Vec<Option<Matrix3<f64>>> = Vec::with_capacity(system.landmarks.len());
    let mut b_l_cache: Vec<Vector3<f64>> = Vec::with_capacity(system.landmarks.len());
    for landmark in &system.landmarks {
        let mut h_ll = landmark.h_ll;
        if lambda > 0.0 {
            h_ll[(0, 0)] += lambda;
            h_ll[(1, 1)] += lambda;
            h_ll[(2, 2)] += lambda;
        }
        let h_ll_inv = h_ll.try_inverse();
        h_ll_inv_cache.push(h_ll_inv);
        b_l_cache.push(landmark.b_l);
        let Some(h_ll_inv) = h_ll_inv else {
            continue;
        };
        for (p, a) in &landmark.cross {
            let a_h: Matrix6x3<f64> = a * h_ll_inv;
            for (q, b) in &landmark.cross {
                // Block Cholesky consumes the lower triangle only. The upper
                // block is its transpose and was discarded by the legacy
                // scalar-triplet assembly, so never store it here.
                if p < q {
                    continue;
                }
                let contribution: Matrix6<f64> = a_h * b.transpose();
                let target = columns[*q].entry(*p).or_insert_with(Matrix6::zeros);
                // Preserve the dense path's per-scalar subtraction order.
                for r in 0..6 {
                    for c in 0..6 {
                        target[(r, c)] -= contribution[(r, c)];
                    }
                }
            }
            let update: Vector6<f64> = a_h * landmark.b_l;
            for k in 0..6 {
                b_reduced[p * 6 + k] += update[k];
            }
        }
    }
    log_process_memory("ba-after-schur-reduction");

    log_process_memory("ba-sparse-before-factor");
    let rhs = DMatrix::from_column_slice(dim, 1, b_reduced.as_slice());
    let solution =
        crate::block_cholesky::solve_spd_blocks6_cached(block_symbolic_cache, columns, &rhs)
            .map_err(|_| BaError::SingularSystem)?;
    let delta_p = DVector::from_column_slice(solution.as_slice());
    drop(solution);
    log_process_memory("ba-after-linear-solve");

    let mut delta_l = DVector::<f64>::zeros(l_count * 3);
    for (l, landmark) in system.landmarks.iter().enumerate() {
        let Some(h_ll_inv) = h_ll_inv_cache[l] else {
            continue;
        };
        let mut acc = -b_l_cache[l];
        for (p, a) in &landmark.cross {
            let dp: Vector6<f64> = delta_p.fixed_rows::<6>(p * 6).into_owned();
            acc -= a.transpose() * dp;
        }
        let dl = h_ll_inv * acc;
        for k in 0..3 {
            delta_l[l * 3 + k] = dl[k];
        }
    }

    Ok((delta_p, delta_l))
}

/// Parallel counterpart of the per-landmark Schur reduction in
/// [`solve_step`] (the `p_count > 0` branch's main loop). Fills
/// `h_ll_inv_cache` / `b_l_cache` directly in parallel — each landmark's
/// `3×3` factorization only touches its own output slot, no merge needed —
/// then updates `s` / `b_reduced` from fixed-size landmark chunks
/// ([`PARALLEL_LANDMARK_CHUNK`]) the same way
/// [`assemble_mono_observations_parallel`] updates the assembly
/// accumulators: within a chunk, every landmark's pose-pair contributions
/// (its `S[p, q] -= …` blocks and its `b_reduced[p] += …` update) are
/// computed concurrently — a pure function of that landmark's cached
/// inverse, `b_l`, and cross blocks, collected into a `(Vec<(p, q, block)>,
/// Vec<(p, upd)>)` pair per landmark — and then folded into `s` /
/// `b_reduced` by a single serial pass, landmark by landmark in ascending
/// index order, before the next chunk starts. `s` and `b_reduced` are
/// disjoint arrays, so grouping a landmark's `S` updates before its
/// `b_reduced` update (rather than interleaving them per pose as the serial
/// loop does) does not change either accumulator's own summation order —
/// only operations that target the *same* memory location are
/// order-sensitive for floating point, and each one's order here is exactly
/// the serial loop's. The result is therefore bit-identical to the serial
/// path at any thread count or chunk size.
fn schur_reduce_parallel(
    system: &NormalEquationsBa,
    lambda: f64,
    s: &mut DMatrix<f64>,
    b_reduced: &mut DVector<f64>,
) -> (Vec<Option<Matrix3<f64>>>, Vec<Vector3<f64>>) {
    use rayon::prelude::*;

    let (h_ll_inv_cache, b_l_cache): (Vec<Option<Matrix3<f64>>>, Vec<Vector3<f64>>) = system
        .landmarks
        .par_iter()
        .map(|landmark| {
            let mut h_ll = landmark.h_ll;
            if lambda > 0.0 {
                h_ll[(0, 0)] += lambda;
                h_ll[(1, 1)] += lambda;
                h_ll[(2, 2)] += lambda;
            }
            (h_ll.try_inverse(), landmark.b_l)
        })
        .unzip();

    let mut start = 0;
    while start < system.landmarks.len() {
        let end = (start + PARALLEL_LANDMARK_CHUNK).min(system.landmarks.len());

        #[allow(clippy::type_complexity)]
        let (s_updates, b_updates): (
            Vec<Vec<(usize, usize, Matrix6<f64>)>>,
            Vec<Vec<(usize, Vector6<f64>)>>,
        ) = system.landmarks[start..end]
            .par_iter()
            .zip(h_ll_inv_cache[start..end].par_iter())
            .map(|(landmark, h_ll_inv)| {
                let mut s_acc = Vec::new();
                let mut b_acc = Vec::new();
                let Some(h_ll_inv) = h_ll_inv else {
                    return (s_acc, b_acc);
                };
                for (p, a) in &landmark.cross {
                    let a_h: Matrix6x3<f64> = a * h_ll_inv;
                    for (q, b) in &landmark.cross {
                        let block: Matrix6<f64> = a_h * b.transpose();
                        s_acc.push((*p, *q, block));
                    }
                    let upd: Vector6<f64> = a_h * landmark.b_l;
                    b_acc.push((*p, upd));
                }
                (s_acc, b_acc)
            })
            .unzip();
        let s_updates: Vec<(usize, usize, Matrix6<f64>)> =
            s_updates.into_iter().flatten().collect();
        let b_updates: Vec<(usize, Vector6<f64>)> = b_updates.into_iter().flatten().collect();

        for (p, q, block) in s_updates {
            for r in 0..6 {
                for c in 0..6 {
                    s[(p * 6 + r, q * 6 + c)] -= block[(r, c)];
                }
            }
        }
        for (p, upd) in b_updates {
            for k in 0..6 {
                b_reduced[p * 6 + k] += upd[k];
            }
        }

        start = end;
    }

    (h_ll_inv_cache, b_l_cache)
}

fn project_pinhole(intrinsics: &(f64, f64, f64, f64), xc: &Point3<f64>) -> Option<Point2<f64>> {
    if xc.z <= 0.0 {
        return None;
    }
    let (fx, fy, cx, cy) = *intrinsics;
    Some(Point2::new(fx * xc.x / xc.z + cx, fy * xc.y / xc.z + cy))
}

/// Maximum absolute and relative (symmetric, Frobenius-normalized) error
/// between two small Jacobians.  This is intentionally a diagnostic helper,
/// not part of the optimizer: it reports both an absolute error (important
/// when a derivative is close to zero) and a scale-free error (important when
/// comparing translation, rotation, point, and intrinsics columns).
fn jacobian_error<I>(pairs: I) -> (f64, f64)
where
    I: IntoIterator<Item = (f64, f64)>,
{
    let mut max_abs = 0.0_f64;
    let mut diff_squared = 0.0_f64;
    let mut scale_squared = 0.0_f64;
    for (analytic, numerical) in pairs {
        if !analytic.is_finite() || !numerical.is_finite() {
            return (f64::INFINITY, f64::INFINITY);
        }
        let diff = analytic - numerical;
        max_abs = max_abs.max(diff.abs());
        diff_squared += diff * diff;
        let scale = analytic.abs().max(numerical.abs());
        scale_squared += scale * scale;
    }
    let scale = scale_squared.sqrt().max(1.0e-15);
    (max_abs, diff_squared.sqrt() / scale)
}

/// Finite-difference audit of the visual pinhole residual Jacobians for one
/// observation.  The production assembly has two deliberately duplicated
/// fast paths (serial and rayon), so this helper mirrors their formulas while
/// evaluating the residual through the public `Camera::project` API.  It is
/// only called by the explicit `VISLOC_SFM_DEBUG_BA_JACOBIANS` diagnostic and
/// by focused unit tests; it never participates in normal BA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BaVisualJacobianCase {
    pub residual_norm: f64,
    pub depth: f64,
    pub pose_max_abs: f64,
    pub pose_relative: f64,
    pub pose_translation_max_abs: f64,
    pub pose_translation_relative: f64,
    pub pose_rotation_max_abs: f64,
    pub pose_rotation_relative: f64,
    pub landmark_max_abs: f64,
    pub landmark_relative: f64,
    pub intrinsics_max_abs: f64,
    pub intrinsics_relative: f64,
}

/// Compare the analytic right-pose, world-landmark, and pinhole-intrinsics
/// Jacobians to central differences at one state.  Intrinsics are reported for
/// the four-parameter pinhole/OpenCV layout; radial distortion is deliberately
/// rejected because the ordinary BA path uses a separate distortion-aware
/// Jacobian in its joint-intrinsics solver.
pub(crate) fn audit_visual_jacobian_case(
    camera: &Camera,
    pose: &Pose,
    point: &Point3<f64>,
    measured: &Point2<f64>,
    epsilon: f64,
) -> Option<BaVisualJacobianCase> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return None;
    }
    if !matches!(camera.model, CameraModel::Pinhole | CameraModel::OpenCv)
        || camera.params.len() < 4
        || camera
            .radial_distortion()
            .is_some_and(|(k1, k2)| k1 != 0.0 || k2 != 0.0)
    {
        return None;
    }
    let intrinsics = camera.intrinsics()?;
    let point_camera = pose.transform_world_point(point);
    let predicted = camera.project(&point_camera)?;
    let residual = predicted - *measured;
    let j_projection = pinhole_projection_jacobian(&intrinsics, &point_camera)?;
    let rotation = pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let mut dpoint_dpose = Matrix3x6::<f64>::zeros();
    dpoint_dpose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation);
    dpoint_dpose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-rotation * skew(&point.coords)));
    let analytic_pose = j_projection * dpoint_dpose;
    let analytic_landmark = j_projection * rotation;
    let x = point_camera.x / point_camera.z;
    let y = point_camera.y / point_camera.z;
    let analytic_intrinsics = Matrix2x4::new(x, 0.0, 1.0, 0.0, 0.0, y, 0.0, 1.0);

    let mut numerical_pose = Matrix2x6::<f64>::zeros();
    for axis in 0..6 {
        let mut plus = pose.clone();
        let mut minus = pose.clone();
        let mut delta = Vector6::<f64>::zeros();
        delta[axis] = epsilon;
        plus.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&delta));
        delta[axis] = -epsilon;
        minus.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&delta));
        let plus = camera.project(&plus.transform_world_point(point))?;
        let minus = camera.project(&minus.transform_world_point(point))?;
        let derivative = (plus - minus) / (2.0 * epsilon);
        numerical_pose[(0, axis)] = derivative.x;
        numerical_pose[(1, axis)] = derivative.y;
    }

    let mut numerical_landmark = Matrix2x3::<f64>::zeros();
    for axis in 0..3 {
        let mut plus = point.coords;
        let mut minus = point.coords;
        plus[axis] += epsilon;
        minus[axis] -= epsilon;
        let plus = camera.project(&pose.transform_world_point(&Point3::from(plus)))?;
        let minus = camera.project(&pose.transform_world_point(&Point3::from(minus)))?;
        let derivative = (plus - minus) / (2.0 * epsilon);
        numerical_landmark[(0, axis)] = derivative.x;
        numerical_landmark[(1, axis)] = derivative.y;
    }

    let mut numerical_intrinsics = Matrix2x4::<f64>::zeros();
    for axis in 0..4 {
        let parameter_epsilon = epsilon * camera.params[axis].abs().max(1.0);
        let mut plus = camera.clone();
        let mut minus = camera.clone();
        plus.params[axis] += parameter_epsilon;
        minus.params[axis] -= parameter_epsilon;
        let plus = plus.project(&point_camera)?;
        let minus = minus.project(&point_camera)?;
        let derivative = (plus - minus) / (2.0 * parameter_epsilon);
        numerical_intrinsics[(0, axis)] = derivative.x;
        numerical_intrinsics[(1, axis)] = derivative.y;
    }

    let (pose_max_abs, pose_relative) = jacobian_error(
        analytic_pose
            .iter()
            .copied()
            .zip(numerical_pose.iter().copied()),
    );
    let pose_error = |columns: std::ops::Range<usize>| {
        jacobian_error((0..2).flat_map(|row| {
            columns
                .clone()
                .map(move |column| (analytic_pose[(row, column)], numerical_pose[(row, column)]))
        }))
    };
    let (pose_translation_max_abs, pose_translation_relative) = pose_error(0..3);
    let (pose_rotation_max_abs, pose_rotation_relative) = pose_error(3..6);
    let (landmark_max_abs, landmark_relative) = jacobian_error(
        analytic_landmark
            .iter()
            .copied()
            .zip(numerical_landmark.iter().copied()),
    );
    let (intrinsics_max_abs, intrinsics_relative) = jacobian_error(
        analytic_intrinsics
            .iter()
            .copied()
            .zip(numerical_intrinsics.iter().copied()),
    );
    Some(BaVisualJacobianCase {
        residual_norm: residual.norm(),
        depth: point_camera.z,
        pose_max_abs,
        pose_relative,
        pose_translation_max_abs,
        pose_translation_relative,
        pose_rotation_max_abs,
        pose_rotation_relative,
        landmark_max_abs,
        landmark_relative,
        intrinsics_max_abs,
        intrinsics_relative,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct BaVisualJacobianBucket {
    pub samples: usize,
    pub pose_max_abs: f64,
    pub pose_relative_max: f64,
    pub pose_translation_max_abs: f64,
    pub pose_translation_relative_max: f64,
    pub pose_rotation_max_abs: f64,
    pub pose_rotation_relative_max: f64,
    pub landmark_max_abs: f64,
    pub landmark_relative_max: f64,
    pub intrinsics_max_abs: f64,
    pub intrinsics_relative_max: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct BaVisualJacobianAudit {
    pub observations_seen: usize,
    pub samples_audited: usize,
    pub invalid_samples: usize,
    pub normal: BaVisualJacobianBucket,
    pub far_depth: BaVisualJacobianBucket,
    pub low_parallax: BaVisualJacobianBucket,
    pub high_residual: BaVisualJacobianBucket,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BaJacobianBucketKind {
    Normal,
    FarDepth,
    LowParallax,
    HighResidual,
}

fn update_jacobian_bucket(bucket: &mut BaVisualJacobianBucket, case: &BaVisualJacobianCase) {
    bucket.samples += 1;
    bucket.pose_max_abs = bucket.pose_max_abs.max(case.pose_max_abs);
    bucket.pose_relative_max = bucket.pose_relative_max.max(case.pose_relative);
    bucket.pose_translation_max_abs = bucket
        .pose_translation_max_abs
        .max(case.pose_translation_max_abs);
    bucket.pose_translation_relative_max = bucket
        .pose_translation_relative_max
        .max(case.pose_translation_relative);
    bucket.pose_rotation_max_abs = bucket.pose_rotation_max_abs.max(case.pose_rotation_max_abs);
    bucket.pose_rotation_relative_max = bucket
        .pose_rotation_relative_max
        .max(case.pose_rotation_relative);
    bucket.landmark_max_abs = bucket.landmark_max_abs.max(case.landmark_max_abs);
    bucket.landmark_relative_max = bucket.landmark_relative_max.max(case.landmark_relative);
    bucket.intrinsics_max_abs = bucket.intrinsics_max_abs.max(case.intrinsics_max_abs);
    bucket.intrinsics_relative_max = bucket.intrinsics_relative_max.max(case.intrinsics_relative);
}

fn observation_jacobian_bucket(
    ba: &BundleAdjustment,
    obs_idx: usize,
    point_camera: &Point3<f64>,
    residual_norm: f64,
) -> BaJacobianBucketKind {
    // A large residual is the most useful disjoint bucket for checking the
    // robust-weighting path.  For geometric conditioning, estimate the widest
    // ray angle to another observation of the same landmark.  This is a
    // deterministic, diagnostic-only proxy; no optimizer decision uses it.
    if residual_norm > 10.0 {
        return BaJacobianBucketKind::HighResidual;
    }
    let observation = &ba.observations[obs_idx];
    let Some(anchor_pose) = ba.poses.get(&observation.keyframe_id) else {
        return BaJacobianBucketKind::Normal;
    };
    let point = &ba.landmarks[&observation.landmark_id];
    let anchor_ray = point.coords - anchor_pose.camera_center_world().coords;
    let mut max_angle = None;
    for (other_idx, other) in ba.observations.iter().enumerate() {
        if other_idx == obs_idx || other.landmark_id != observation.landmark_id {
            continue;
        }
        let Some(other_pose) = ba.poses.get(&other.keyframe_id) else {
            continue;
        };
        let other_ray = point.coords - other_pose.camera_center_world().coords;
        let (Some(a), Some(b)) = (
            anchor_ray.try_normalize(1.0e-15),
            other_ray.try_normalize(1.0e-15),
        ) else {
            continue;
        };
        let cosine = a.dot(&b).clamp(-1.0, 1.0);
        let angle = cosine.acos();
        if angle.is_finite() {
            max_angle = Some(max_angle.map_or(angle, |current: f64| current.max(angle)));
        }
    }
    if max_angle.is_some_and(|angle| angle.to_degrees() < 1.0) {
        BaJacobianBucketKind::LowParallax
    } else if point_camera.z > 100.0 {
        BaJacobianBucketKind::FarDepth
    } else {
        BaJacobianBucketKind::Normal
    }
}

/// Audit a deterministic, small sample from a live BA state.  The function is
/// intentionally `pub(crate)` so the incremental SFM diagnostic can invoke it
/// without exposing a new public solver API.  It returns no result used by the
/// optimizer and is never called unless the explicit debug environment flag is
/// enabled by the caller.
pub(crate) fn audit_bundle_visual_jacobians(
    ba: &BundleAdjustment,
    max_samples: usize,
) -> BaVisualJacobianAudit {
    let mut report = BaVisualJacobianAudit {
        observations_seen: ba.observations.len(),
        ..BaVisualJacobianAudit::default()
    };
    if max_samples == 0 {
        return report;
    }

    // Reserve an equal deterministic quota for each conditioning bucket, so a
    // long track ordered entirely by one region cannot hide the other cases.
    let quota = max_samples.div_ceil(4);
    let mut candidates: [Vec<usize>; 4] = std::array::from_fn(|_| Vec::new());
    for (obs_idx, observation) in ba.observations.iter().enumerate() {
        let (Some(pose), Some(point)) = (
            ba.poses.get(&observation.keyframe_id),
            ba.landmarks.get(&observation.landmark_id),
        ) else {
            continue;
        };
        let point_camera = pose.transform_world_point(point);
        let Some(predicted) = ba.camera.project(&point_camera) else {
            continue;
        };
        let residual_norm = (predicted - observation.xy).norm();
        let kind = observation_jacobian_bucket(ba, obs_idx, &point_camera, residual_norm);
        let slot = match kind {
            BaJacobianBucketKind::Normal => 0,
            BaJacobianBucketKind::FarDepth => 1,
            BaJacobianBucketKind::LowParallax => 2,
            BaJacobianBucketKind::HighResidual => 3,
        };
        if candidates[slot].len() < quota {
            candidates[slot].push(obs_idx);
        }
    }

    for indices in candidates {
        for obs_idx in indices {
            if report.samples_audited >= max_samples {
                break;
            }
            let observation = &ba.observations[obs_idx];
            let (Some(pose), Some(point)) = (
                ba.poses.get(&observation.keyframe_id),
                ba.landmarks.get(&observation.landmark_id),
            ) else {
                report.invalid_samples += 1;
                continue;
            };
            let Some(case) =
                audit_visual_jacobian_case(&ba.camera, pose, point, &observation.xy, 1.0e-6)
            else {
                report.invalid_samples += 1;
                continue;
            };
            let point_camera = pose.transform_world_point(point);
            let kind = observation_jacobian_bucket(ba, obs_idx, &point_camera, case.residual_norm);
            match kind {
                BaJacobianBucketKind::Normal => update_jacobian_bucket(&mut report.normal, &case),
                BaJacobianBucketKind::FarDepth => {
                    update_jacobian_bucket(&mut report.far_depth, &case)
                }
                BaJacobianBucketKind::LowParallax => {
                    update_jacobian_bucket(&mut report.low_parallax, &case)
                }
                BaJacobianBucketKind::HighResidual => {
                    update_jacobian_bucket(&mut report.high_residual, &case)
                }
            }
            report.samples_audited += 1;
        }
    }
    report
}

fn pinhole_projection_jacobian(
    intrinsics: &(f64, f64, f64, f64),
    point: &Point3<f64>,
) -> Option<Matrix2x3<f64>> {
    if point.z <= 0.0 {
        return None;
    }
    let (fx, fy, _, _) = *intrinsics;
    let z_inv = point.z.recip();
    let z_inv2 = z_inv * z_inv;
    Some(Matrix2x3::new(
        fx * z_inv,
        0.0,
        -fx * point.x * z_inv2,
        0.0,
        fy * z_inv,
        -fy * point.y * z_inv2,
    ))
}

fn general_stereo_residual_jacobians(
    left_intrinsics: &(f64, f64, f64, f64),
    observation: &BaGeneralStereoObservation,
    pose: &Pose,
    point_world: &Point3<f64>,
) -> Option<(Vector4<f64>, Matrix4x6<f64>, Matrix4x3<f64>)> {
    let right_intrinsics = observation.right_camera.intrinsics()?;
    let point_left = pose.transform_world_point(point_world);
    let point_right = observation.left_to_right.transform_point(&point_left);
    let predicted_left = project_pinhole(left_intrinsics, &point_left)?;
    let predicted_right = project_pinhole(&right_intrinsics, &point_right)?;
    let residual = Vector4::new(
        predicted_left.x - observation.xy_left.x,
        predicted_left.y - observation.xy_left.y,
        predicted_right.x - observation.xy_right.x,
        predicted_right.y - observation.xy_right.y,
    );

    let j_left_projection = pinhole_projection_jacobian(left_intrinsics, &point_left)?;
    let j_right_projection = pinhole_projection_jacobian(&right_intrinsics, &point_right)?;
    let rotation_world_to_left = pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let rotation_left_to_right = observation
        .left_to_right
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let mut d_left_d_pose = Matrix3x6::<f64>::zeros();
    d_left_d_pose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation_world_to_left);
    d_left_d_pose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-rotation_world_to_left * skew(&point_world.coords)));

    let j_left_pose = j_left_projection * d_left_d_pose;
    let j_right_pose = j_right_projection * rotation_left_to_right * d_left_d_pose;
    let j_left_landmark = j_left_projection * rotation_world_to_left;
    let j_right_landmark = j_right_projection * rotation_left_to_right * rotation_world_to_left;
    let mut j_pose = Matrix4x6::<f64>::zeros();
    j_pose.fixed_rows_mut::<2>(0).copy_from(&j_left_pose);
    j_pose.fixed_rows_mut::<2>(2).copy_from(&j_right_pose);
    let mut j_landmark = Matrix4x3::<f64>::zeros();
    j_landmark
        .fixed_rows_mut::<2>(0)
        .copy_from(&j_left_landmark);
    j_landmark
        .fixed_rows_mut::<2>(2)
        .copy_from(&j_right_landmark);
    Some((residual, j_pose, j_landmark))
}

fn rig_residual_jacobians(
    observation: &BaRigObservation,
    pose: &Pose,
    point_world: &Point3<f64>,
) -> Option<(Vector2<f64>, Matrix2x6<f64>, Matrix2x3<f64>)> {
    let intrinsics = observation.camera.intrinsics()?;
    let point_rig = pose.transform_world_point(point_world);
    let point_sensor = observation.sensor_from_rig.transform_point(&point_rig);
    let predicted = project_pinhole(&intrinsics, &point_sensor)?;
    let residual = predicted - observation.xy;
    let projection = pinhole_projection_jacobian(&intrinsics, &point_sensor)?;
    let rotation_world_to_rig = pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let rotation_rig_to_sensor = observation
        .sensor_from_rig
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let mut d_rig_d_pose = Matrix3x6::<f64>::zeros();
    d_rig_d_pose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation_world_to_rig);
    d_rig_d_pose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-rotation_world_to_rig * skew(&point_world.coords)));
    let j_pose = projection * rotation_rig_to_sensor * d_rig_d_pose;
    let j_landmark = projection * rotation_rig_to_sensor * rotation_world_to_rig;
    Some((residual, j_pose, j_landmark))
}

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Right-Jacobian inverse on SO(3): Jr⁻¹(φ) = I + ½[φ]× + c·[φ]×²,
/// with `c = (1/θ²) · (1 − (θ/2)·cot(θ/2))` for `θ = ‖φ‖`. At `θ → 0`
/// falls back to the leading Taylor expansion `I + ½[φ]× + (1/12)·[φ]×²`.
/// Used by [`build_normal_equations`] to linearise the SO(3) log
/// residual of the Forster IMU factor.
///
/// Identity: `Jr_inv(φ) = Jl_inv(−φ)`, which means relative to the
/// `so3_left_jacobian_inverse` formula in [`visloc_core::geometry`]
/// only the sign of the linear `[φ]×` term flips (the quadratic
/// `[φ]×²` coefficient stays the same).
fn right_jacobian_inverse_so3(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta_sq = phi.norm_squared();
    let phi_skew = skew(phi);
    if theta_sq < 1e-10 {
        Matrix3::identity() + 0.5 * phi_skew + (1.0 / 12.0) * phi_skew * phi_skew
    } else {
        let theta = theta_sq.sqrt();
        let half_theta = 0.5 * theta;
        let c = (1.0 - half_theta * half_theta.cos() / half_theta.sin()) / theta_sq;
        Matrix3::identity() + 0.5 * phi_skew + c * phi_skew * phi_skew
    }
}

/// `LocalRefiner` implementation that runs windowed bundle adjustment on
/// a staged map update. Existing keyframes and landmarks in the window
/// (already in `VisualMap`) are added as fixed gauge; the newly-staged
/// keyframe poses and landmark positions are the BA variables. Observations
/// from both the existing window and the staged update feed the residual.
///
/// Refined poses / landmarks are written back into the staged update so
/// subsequent `apply_to(&mut map)` lands the BA-corrected values. The
/// existing map is never mutated by this refiner.
#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustmentRefiner {
    pub config: BaConfig,
}

impl BundleAdjustmentRefiner {
    pub fn new(config: BaConfig) -> Self {
        Self { config }
    }
}

impl Default for BundleAdjustmentRefiner {
    fn default() -> Self {
        Self::new(BaConfig::default())
    }
}

impl LocalRefiner for BundleAdjustmentRefiner {
    fn refine(
        &self,
        map: &VisualMap,
        local_window: &LocalMapWindow,
        staged_update: &mut StagedMapUpdate,
    ) -> LocalRefinementResult {
        // Pick a camera. Prefer the camera attached to the first staged
        // keyframe; fall back to any camera in the existing window.
        let camera = staged_update
            .keyframes
            .iter()
            .find_map(|kf| map.cameras.get(&kf.frame.camera_id).cloned())
            .or_else(|| {
                local_window
                    .keyframe_ids
                    .iter()
                    .find_map(|id| map.keyframes.get(id))
                    .and_then(|kf| map.cameras.get(&kf.frame.camera_id).cloned())
            });
        let Some(camera) = camera else {
            return LocalRefinementResult::skipped(LocalRefinementReason::Noop);
        };

        let mut ba = BundleAdjustment::new(camera);

        // Treat staged keyframes / landmarks as variable. Everything else
        // in the window is already in `map` and serves as a fixed gauge.
        // The local window typically already includes the newly-staged
        // keyframe (the local-mapping pipeline inserts it into a working
        // map before computing the window), so we must skip those when
        // adding fixed poses — otherwise the BA variable would become a
        // fixed gauge and never move.
        let staged_kf_ids: std::collections::BTreeSet<u64> = staged_update
            .keyframes
            .iter()
            .map(|kf| kf.frame.id)
            .collect();
        let staged_lm_ids: std::collections::BTreeSet<u64> =
            staged_update.landmarks.iter().map(|lm| lm.id).collect();

        // Fixed gauge: window keyframes that are NOT in the staged update.
        for &kf_id in &local_window.keyframe_ids {
            if staged_kf_ids.contains(&kf_id) {
                continue;
            }
            let Some(kf) = map.keyframes.get(&kf_id) else {
                continue;
            };
            let Some(pose) = kf.frame.pose.clone() else {
                continue;
            };
            ba.add_pose(kf_id, pose);
            ba.fix_pose(kf_id);
        }
        // Variable: newly-staged keyframe poses.
        for kf in &staged_update.keyframes {
            let id = kf.frame.id;
            let Some(pose) = kf.frame.pose.clone() else {
                continue;
            };
            ba.add_pose(id, pose);
        }

        // Fixed gauge: window landmarks that are NOT in the staged update.
        for &lm_id in &local_window.landmark_ids {
            if staged_lm_ids.contains(&lm_id) {
                continue;
            }
            let Some(lm) = map.landmarks.get(&lm_id) else {
                continue;
            };
            ba.add_landmark(lm_id, lm.position);
            ba.fix_landmark(lm_id);
        }
        // Variable: newly-staged landmarks.
        for lm in &staged_update.landmarks {
            ba.add_landmark(lm.id, lm.position);
        }

        // Observations: existing keyframes' observations of fixed landmarks
        // (anchor the gauge), plus new staged observations.
        for &kf_id in &local_window.keyframe_ids {
            let Some(kf) = map.keyframes.get(&kf_id) else {
                continue;
            };
            for obs in &kf.observations {
                if ba.poses.contains_key(&obs.frame_id)
                    && ba.landmarks.contains_key(&obs.landmark_id)
                {
                    ba.add_observation(BaObservation {
                        keyframe_id: obs.frame_id,
                        landmark_id: obs.landmark_id,
                        xy: obs.xy,
                    });
                }
            }
        }
        for obs in &staged_update.observations {
            if ba.poses.contains_key(&obs.frame_id) && ba.landmarks.contains_key(&obs.landmark_id) {
                ba.add_observation(BaObservation {
                    keyframe_id: obs.frame_id,
                    landmark_id: obs.landmark_id,
                    xy: obs.xy,
                });
            }
        }

        // No observations to optimize against → nothing to refine.
        if ba.observations.is_empty() {
            return LocalRefinementResult::skipped(LocalRefinementReason::Noop);
        }
        // Every variable pose is fixed (or no variable poses exist) AND
        // every variable landmark is fixed → nothing to optimize. The BA
        // would still report `AllPosesFixed`; just skip cleanly.
        let has_variable_pose = ba.poses.keys().any(|id| !ba.fixed_poses.contains(id));
        let has_variable_landmark = ba
            .landmarks
            .keys()
            .any(|id| !ba.fixed_landmarks.contains(id));
        if !has_variable_pose && !has_variable_landmark {
            return LocalRefinementResult::skipped(LocalRefinementReason::Noop);
        }

        if ba.optimize(&self.config).is_err() {
            return LocalRefinementResult::skipped(LocalRefinementReason::Noop);
        }

        // Write back refined values into staged_update. Fixed entries are
        // never modified.
        let mut keyframe_count = 0usize;
        for kf in staged_update.keyframes.iter_mut() {
            let id = kf.frame.id;
            if ba.fixed_poses.contains(&id) {
                continue;
            }
            if let Some(refined) = ba.poses.get(&id).cloned() {
                kf.frame.pose = Some(refined);
                keyframe_count += 1;
            }
        }
        let mut landmark_count = 0usize;
        for lm in staged_update.landmarks.iter_mut() {
            if ba.fixed_landmarks.contains(&lm.id) {
                continue;
            }
            if let Some(refined) = ba.landmarks.get(&lm.id).copied() {
                lm.position = refined;
                landmark_count += 1;
            }
        }

        LocalRefinementResult {
            refined: keyframe_count > 0 || landmark_count > 0,
            reason: LocalRefinementReason::Refined,
            keyframe_count,
            landmark_count,
        }
    }
}

/// Private matrix-free Schur backend.
///
/// The backend consumes an assembled pure-visual `PoseDiagonal` system and
/// never constructs pose-pair blocks, triplets, or a Cholesky factor.  Its
/// public entry point remains on [`BundleAdjustment`]; keeping the numerical
/// implementation here avoids a second operator implementation in tests.
mod implicit_schur {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub(super) enum ImplicitSchurError {
        DenseCameraHessian,
        EmptySystem,
        DimensionMismatch {
            expected: usize,
            actual: usize,
        },
        NonFinite(&'static str),
        InvalidLambda,
        InvalidTolerance,
        NonSpdPreconditioner(usize),
        NonPositiveCurvature,
        ResidualCheckFailed {
            iterations: usize,
            recursive_norm: f64,
            true_norm: f64,
            target: f64,
        },
        MaxIterations {
            iterations: usize,
            recursive_norm: f64,
            residual_norm: f64,
            target: f64,
        },
    }

    impl ImplicitSchurError {
        pub(super) fn diagnostics(&self) -> (Option<usize>, Option<f64>, Option<f64>) {
            match *self {
                Self::ResidualCheckFailed {
                    iterations,
                    true_norm,
                    target,
                    ..
                }
                | Self::MaxIterations {
                    iterations,
                    residual_norm: true_norm,
                    target,
                    ..
                } => (Some(iterations), Some(true_norm), Some(target)),
                _ => (None, None, None),
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub(super) struct PcgOptions {
        pub(super) max_iterations: usize,
        pub(super) relative_tolerance: f64,
        pub(super) absolute_tolerance: f64,
    }

    impl Default for PcgOptions {
        fn default() -> Self {
            Self {
                max_iterations: 128,
                relative_tolerance: 1.0e-12,
                absolute_tolerance: 1.0e-12,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub(super) struct PcgResult {
        pub(super) solution: DVector<f64>,
        pub(super) iterations: usize,
        pub(super) residual_norm: f64,
        pub(super) target: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub(super) struct PcgRunDiagnostics {
        pub(super) pcg_iterations: Option<usize>,
        pub(super) true_residual_rechecks: usize,
        pub(super) failed_true_residual_rechecks: usize,
        pub(super) restarts: usize,
    }

    impl From<PcgRunDiagnostics> for MatrixFreeBaRestartIterationStats {
        fn from(value: PcgRunDiagnostics) -> Self {
            Self {
                iteration: 0,
                pcg_iterations: value.pcg_iterations,
                true_residual_rechecks: value.true_residual_rechecks,
                failed_true_residual_rechecks: value.failed_true_residual_rechecks,
                restarts: value.restarts,
                terminal_failure: None,
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub(super) struct PcgSolveFailure {
        pub(super) error: ImplicitSchurError,
        pub(super) diagnostics: PcgRunDiagnostics,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub(super) struct PcgRun {
        pub(super) result: PcgResult,
        pub(super) diagnostics: PcgRunDiagnostics,
    }

    /// A pure-visual Schur operator over the six-dimensional pose blocks.
    ///
    /// `landmarks` and their cross blocks are borrowed from
    /// `NormalEquationsBa`; only the per-landmark 3×3 inverse cache and the
    /// pose-block-Jacobi inverse are retained.  A repeated pose slot in one
    /// landmark is intentional: the constructor groups those blocks only for
    /// the preconditioner, while `apply` evaluates every original cross block
    /// so same-frame multi-sensor terms are not lost.
    pub(super) struct ImplicitSchurOperator<'a> {
        diagonal: Vec<Matrix6<f64>>,
        landmarks: &'a [LandmarkBlock],
        h_ll_inverse: Vec<Option<Matrix3<f64>>>,
        preconditioner_inverse: Vec<Matrix6<f64>>,
        rhs: DVector<f64>,
    }

    impl<'a> ImplicitSchurOperator<'a> {
        pub(super) fn new(
            system: &'a NormalEquationsBa,
            lambda: f64,
        ) -> Result<Self, ImplicitSchurError> {
            Self::new_with_debug(system, lambda, None)
        }

        pub(super) fn new_with_debug(
            system: &'a NormalEquationsBa,
            lambda: f64,
            debug: Option<SchurBlockDebugContext<'_>>,
        ) -> Result<Self, ImplicitSchurError> {
            if !lambda.is_finite() || lambda < 0.0 {
                return Err(ImplicitSchurError::InvalidLambda);
            }
            let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
                return Err(ImplicitSchurError::DenseCameraHessian);
            };
            if diagonal.is_empty() {
                return Err(ImplicitSchurError::EmptySystem);
            }
            let dimension = diagonal.len() * 6;
            if system.b_p.len() != dimension {
                return Err(ImplicitSchurError::DimensionMismatch {
                    expected: dimension,
                    actual: system.b_p.len(),
                });
            }
            if !system.b_p.iter().all(|value| value.is_finite()) {
                return Err(ImplicitSchurError::NonFinite("pose gradient"));
            }
            if !diagonal
                .iter()
                .flat_map(|block| block.iter())
                .all(|value| value.is_finite())
            {
                return Err(ImplicitSchurError::NonFinite("pose Hessian"));
            }

            let mut damped_diagonal = diagonal.clone();
            if lambda > 0.0 {
                for block in &mut damped_diagonal {
                    for component in 0..6 {
                        block[(component, component)] += lambda;
                    }
                }
            }
            if !damped_diagonal
                .iter()
                .flat_map(|block| block.iter())
                .all(|value| value.is_finite())
            {
                return Err(ImplicitSchurError::NonFinite("damped pose Hessian"));
            }

            let mut h_ll_inverse = Vec::with_capacity(system.landmarks.len());
            let mut rhs = -&system.b_p;
            for landmark in &system.landmarks {
                if !landmark
                    .h_ll
                    .iter()
                    .chain(landmark.b_l.iter())
                    .all(|value| value.is_finite())
                {
                    return Err(ImplicitSchurError::NonFinite("landmark block"));
                }
                for (pose, cross) in &landmark.cross {
                    if *pose >= diagonal.len() {
                        return Err(ImplicitSchurError::DimensionMismatch {
                            expected: diagonal.len(),
                            actual: (*pose).saturating_add(1),
                        });
                    }
                    if !cross.iter().all(|value| value.is_finite()) {
                        return Err(ImplicitSchurError::NonFinite("landmark cross block"));
                    }
                }

                let mut h_ll = landmark.h_ll;
                if lambda > 0.0 {
                    for component in 0..3 {
                        h_ll[(component, component)] += lambda;
                    }
                }
                if !h_ll.iter().all(|value| value.is_finite()) {
                    return Err(ImplicitSchurError::NonFinite("damped landmark Hessian"));
                }
                let inverse = h_ll.try_inverse();
                if let Some(inverse) = inverse {
                    if !inverse.iter().all(|value| value.is_finite()) {
                        return Err(ImplicitSchurError::NonFinite("landmark inverse"));
                    }
                }
                h_ll_inverse.push(inverse);
                let Some(inverse) = inverse else {
                    // Match solve_step: an un-invertible landmark contributes
                    // neither to the Schur system nor to back-substitution.
                    continue;
                };
                for (pose, cross) in &landmark.cross {
                    let update: Vector6<f64> = cross * inverse * landmark.b_l;
                    for component in 0..6 {
                        rhs[pose * 6 + component] += update[component];
                    }
                }
            }
            if !rhs.iter().all(|value| value.is_finite()) {
                return Err(ImplicitSchurError::NonFinite("reduced right hand side"));
            }

            // Block-Jacobi preconditioner: aggregate all cross blocks that
            // touch the same pose before forming the diagonal Schur term.
            // This is essential when two sensors observe the same landmark
            // from one rig frame: the cross-sensor terms belong in the same
            // pose block, not in two independent diagonal blocks.
            let mut preconditioner = damped_diagonal.clone();
            for (landmark, inverse) in system.landmarks.iter().zip(&h_ll_inverse) {
                let Some(inverse) = inverse else {
                    continue;
                };
                let mut grouped: BTreeMap<usize, Matrix6x3<f64>> = BTreeMap::new();
                for (pose, cross) in &landmark.cross {
                    *grouped.entry(*pose).or_insert_with(Matrix6x3::zeros) += cross;
                }
                for (pose, cross) in grouped {
                    preconditioner[pose] -= cross * inverse * cross.transpose();
                }
            }

            if let Some(context) = debug {
                if context.pose_slot < preconditioner.len() {
                    let counts = collect_schur_block_debug_counts(
                        system,
                        &h_ll_inverse,
                        lambda,
                        context.pose_slot,
                    );
                    let h_pp_lambda = &damped_diagonal[context.pose_slot];
                    let schur_block = &preconditioner[context.pose_slot];
                    emit_schur_block_debug(context, lambda, h_pp_lambda, schur_block, counts);
                }
            }

            if !preconditioner
                .iter()
                .flat_map(|block| block.iter())
                .all(|value| value.is_finite())
            {
                return Err(ImplicitSchurError::NonFinite("Schur preconditioner"));
            }

            let mut preconditioner_inverse = Vec::with_capacity(preconditioner.len());
            for (pose, block) in preconditioner.iter().enumerate() {
                let Some(cholesky) = block.cholesky() else {
                    return Err(ImplicitSchurError::NonSpdPreconditioner(pose));
                };
                let inverse = cholesky.inverse();
                if !inverse.iter().all(|value| value.is_finite()) {
                    return Err(ImplicitSchurError::NonFinite("preconditioner inverse"));
                }
                preconditioner_inverse.push(inverse);
            }

            Ok(Self {
                diagonal: damped_diagonal,
                landmarks: &system.landmarks,
                h_ll_inverse,
                preconditioner_inverse,
                rhs,
            })
        }

        pub(super) fn dimension(&self) -> usize {
            self.diagonal.len() * 6
        }

        pub(super) fn rhs(&self) -> &DVector<f64> {
            &self.rhs
        }

        #[cfg(test)]
        pub(super) fn h_ll_inverse(&self, index: usize) -> Option<Matrix3<f64>> {
            self.h_ll_inverse[index]
        }

        pub(super) fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, ImplicitSchurError> {
            if x.len() != self.dimension() {
                return Err(ImplicitSchurError::DimensionMismatch {
                    expected: self.dimension(),
                    actual: x.len(),
                });
            }
            if !x.iter().all(|value| value.is_finite()) {
                return Err(ImplicitSchurError::NonFinite("operator input"));
            }
            let mut out = DVector::zeros(self.dimension());
            for (pose, block) in self.diagonal.iter().enumerate() {
                let x_pose: Vector6<f64> = x.fixed_rows::<6>(pose * 6).into_owned();
                let value = block * x_pose;
                for component in 0..6 {
                    out[pose * 6 + component] = value[component];
                }
            }
            for (landmark, inverse) in self.landmarks.iter().zip(&self.h_ll_inverse) {
                let Some(inverse) = inverse else {
                    continue;
                };
                let mut projected = Vector3::zeros();
                // Keep the original cross vector order.  Do not deduplicate
                // here: duplicate pose slots encode separate sensor rows.
                for (pose, cross) in &landmark.cross {
                    let x_pose: Vector6<f64> = x.fixed_rows::<6>(pose * 6).into_owned();
                    projected += cross.transpose() * x_pose;
                }
                let reduced = *inverse * projected;
                for (pose, cross) in &landmark.cross {
                    let value: Vector6<f64> = cross * reduced;
                    for component in 0..6 {
                        out[pose * 6 + component] -= value[component];
                    }
                }
            }
            if !out.iter().all(|value| value.is_finite()) {
                return Err(ImplicitSchurError::NonFinite("operator output"));
            }
            Ok(out)
        }

        pub(super) fn apply_preconditioner(
            &self,
            residual: &DVector<f64>,
        ) -> Result<DVector<f64>, ImplicitSchurError> {
            if residual.len() != self.dimension() {
                return Err(ImplicitSchurError::DimensionMismatch {
                    expected: self.dimension(),
                    actual: residual.len(),
                });
            }
            if !residual.iter().all(|value| value.is_finite()) {
                return Err(ImplicitSchurError::NonFinite("preconditioner input"));
            }
            let mut out = DVector::zeros(self.dimension());
            for (pose, inverse) in self.preconditioner_inverse.iter().enumerate() {
                let residual_pose: Vector6<f64> = residual.fixed_rows::<6>(pose * 6).into_owned();
                let value = inverse * residual_pose;
                for component in 0..6 {
                    out[pose * 6 + component] = value[component];
                }
            }
            if !out.iter().all(|value| value.is_finite()) {
                return Err(ImplicitSchurError::NonFinite("preconditioner output"));
            }
            Ok(out)
        }

        pub(super) fn solve_pcg(
            &self,
            rhs: &DVector<f64>,
            options: PcgOptions,
        ) -> Result<PcgResult, ImplicitSchurError> {
            self.solve_pcg_with_restart(rhs, options, 0)
                .map(|run| run.result)
                .map_err(|failure| failure.error)
        }

        pub(super) fn solve_pcg_with_restart(
            &self,
            rhs: &DVector<f64>,
            options: PcgOptions,
            max_restarts: usize,
        ) -> Result<PcgRun, PcgSolveFailure> {
            self.solve_pcg_with_restart_hook(rhs, options, max_restarts, |_, residual, norm| {
                Ok((residual, norm))
            })
        }

        #[cfg(test)]
        pub(super) fn solve_pcg_with_injected_recursive_residual_for_test(
            &self,
            rhs: &DVector<f64>,
            options: PcgOptions,
            max_restarts: usize,
        ) -> Result<PcgRun, PcgSolveFailure> {
            let mut inject_once = true;
            self.solve_pcg_with_restart_hook(
                rhs,
                options,
                max_restarts,
                move |check, residual, norm| {
                    if check == 1 && inject_once {
                        inject_once = false;
                        // This test-only hook emulates recursive residual
                        // cancellation after the first alpha update.  The subsequent true
                        // residual check still computes b-Ax from the
                        // operator, so no true residual or norm is forged.
                        return Ok((DVector::zeros(residual.len()), 0.0));
                    }
                    Ok((residual, norm))
                },
            )
        }

        fn solve_pcg_with_restart_hook<F>(
            &self,
            rhs: &DVector<f64>,
            options: PcgOptions,
            max_restarts: usize,
            mut recursive_residual_adjustment: F,
        ) -> Result<PcgRun, PcgSolveFailure>
        where
            F: FnMut(usize, DVector<f64>, f64) -> Result<(DVector<f64>, f64), ImplicitSchurError>,
        {
            let mut diagnostics = PcgRunDiagnostics::default();
            let fail = |error, diagnostics| PcgSolveFailure { error, diagnostics };
            if rhs.len() != self.dimension() {
                return Err(fail(
                    ImplicitSchurError::DimensionMismatch {
                        expected: self.dimension(),
                        actual: rhs.len(),
                    },
                    diagnostics,
                ));
            }
            if !rhs.iter().all(|value| value.is_finite()) {
                return Err(fail(
                    ImplicitSchurError::NonFinite("PCG right hand side"),
                    diagnostics,
                ));
            }
            if !options.relative_tolerance.is_finite()
                || options.relative_tolerance < 0.0
                || !options.absolute_tolerance.is_finite()
                || options.absolute_tolerance < 0.0
            {
                return Err(fail(ImplicitSchurError::InvalidTolerance, diagnostics));
            }
            // A converged zero-RHS or positively damped system only proves a
            // numerical linear solve.  It does not prove that the model's
            // monocular/rig gauge has been anchored; that remains a caller-
            // level invariant outside this prototype.
            let rhs_norm = rhs.norm();
            let target = options
                .absolute_tolerance
                .max(options.relative_tolerance * rhs_norm);
            if !target.is_finite() {
                return Err(fail(
                    ImplicitSchurError::NonFinite("PCG target"),
                    diagnostics,
                ));
            }
            let mut solution = DVector::zeros(self.dimension());
            let mut residual = rhs.clone();
            let mut residual_norm = residual.norm();
            if !residual_norm.is_finite() {
                return Err(fail(
                    ImplicitSchurError::NonFinite("initial residual"),
                    diagnostics,
                ));
            }
            // The solve has passed input/target validation.  Recording zero
            // explicitly makes zero-RHS and zero-budget outcomes distinct
            // from failures that occurred before a PCG iteration began.
            diagnostics.pcg_iterations = Some(0);
            if residual_norm <= target {
                diagnostics.true_residual_rechecks += 1;
                let (true_residual, true_norm) = self
                    .true_residual(rhs, &solution)
                    .map_err(|error| fail(error, diagnostics))?;
                if true_norm <= target {
                    return Ok(PcgRun {
                        result: PcgResult {
                            solution,
                            iterations: 0,
                            residual_norm: true_norm,
                            target,
                        },
                        diagnostics,
                    });
                }
                diagnostics.failed_true_residual_rechecks += 1;
                if max_restarts == 0 || options.max_iterations == 0 {
                    return Err(fail(
                        ImplicitSchurError::ResidualCheckFailed {
                            iterations: 0,
                            recursive_norm: residual_norm,
                            true_norm,
                            target,
                        },
                        diagnostics,
                    ));
                }
                residual = true_residual;
                residual_norm = true_norm;
                diagnostics.restarts = 1;
            }
            if options.max_iterations == 0 {
                return Err(fail(
                    ImplicitSchurError::MaxIterations {
                        iterations: 0,
                        recursive_norm: residual_norm,
                        residual_norm,
                        target,
                    },
                    diagnostics,
                ));
            }

            let (mut direction, mut rho) = self
                .restart_direction(&residual)
                .map_err(|error| fail(error, diagnostics))?;

            for iteration in 1..=options.max_iterations {
                let applied = self
                    .apply(&direction)
                    .map_err(|error| fail(error, diagnostics))?;
                let curvature = direction.dot(&applied);
                if !curvature.is_finite() || curvature <= 0.0 {
                    return Err(fail(ImplicitSchurError::NonPositiveCurvature, diagnostics));
                }
                let alpha = rho / curvature;
                if !alpha.is_finite() {
                    return Err(fail(ImplicitSchurError::NonFinite("PCG step"), diagnostics));
                }
                solution += alpha * &direction;
                residual -= alpha * applied;
                residual_norm = residual.norm();
                (residual, residual_norm) =
                    recursive_residual_adjustment(iteration, residual, residual_norm)
                        .map_err(|error| fail(error, diagnostics))?;
                diagnostics.pcg_iterations = Some(iteration);
                if !solution.iter().all(|value| value.is_finite()) || !residual_norm.is_finite() {
                    return Err(fail(
                        ImplicitSchurError::NonFinite("PCG iterate"),
                        diagnostics,
                    ));
                }
                if residual_norm <= target {
                    diagnostics.true_residual_rechecks += 1;
                    let (true_residual, true_norm) = self
                        .true_residual(rhs, &solution)
                        .map_err(|error| fail(error, diagnostics))?;
                    if true_norm <= target {
                        return Ok(PcgRun {
                            result: PcgResult {
                                solution,
                                iterations: iteration,
                                residual_norm: true_norm,
                                target,
                            },
                            diagnostics,
                        });
                    }
                    diagnostics.failed_true_residual_rechecks += 1;
                    if diagnostics.restarts < max_restarts && iteration < options.max_iterations {
                        residual = true_residual;
                        (direction, rho) = self
                            .restart_direction(&residual)
                            .map_err(|error| fail(error, diagnostics))?;
                        diagnostics.restarts += 1;
                        continue;
                    }
                    return Err(fail(
                        ImplicitSchurError::ResidualCheckFailed {
                            iterations: iteration,
                            recursive_norm: residual_norm,
                            true_norm,
                            target,
                        },
                        diagnostics,
                    ));
                }
                if iteration == options.max_iterations {
                    diagnostics.true_residual_rechecks += 1;
                    let (_, true_norm) = self
                        .true_residual(rhs, &solution)
                        .map_err(|error| fail(error, diagnostics))?;
                    if true_norm <= target {
                        return Ok(PcgRun {
                            result: PcgResult {
                                solution,
                                iterations: iteration,
                                residual_norm: true_norm,
                                target,
                            },
                            diagnostics,
                        });
                    }
                    diagnostics.failed_true_residual_rechecks += 1;
                    return Err(fail(
                        ImplicitSchurError::MaxIterations {
                            iterations: iteration,
                            recursive_norm: residual_norm,
                            residual_norm: true_norm,
                            target,
                        },
                        diagnostics,
                    ));
                }
                let preconditioned = self
                    .apply_preconditioner(&residual)
                    .map_err(|error| fail(error, diagnostics))?;
                let next_rho = residual.dot(&preconditioned);
                if !next_rho.is_finite() || next_rho <= 0.0 {
                    return Err(fail(ImplicitSchurError::NonPositiveCurvature, diagnostics));
                }
                let beta = next_rho / rho;
                if !beta.is_finite() {
                    return Err(fail(
                        ImplicitSchurError::NonFinite("PCG direction"),
                        diagnostics,
                    ));
                }
                direction = &preconditioned + beta * direction;
                rho = next_rho;
            }
            unreachable!("the max-iteration branch returns above");
        }

        fn restart_direction(
            &self,
            residual: &DVector<f64>,
        ) -> Result<(DVector<f64>, f64), ImplicitSchurError> {
            let preconditioned = self.apply_preconditioner(residual)?;
            let rho = residual.dot(&preconditioned);
            if !rho.is_finite() || rho <= 0.0 {
                return Err(ImplicitSchurError::NonPositiveCurvature);
            }
            Ok((preconditioned, rho))
        }

        fn true_residual(
            &self,
            rhs: &DVector<f64>,
            solution: &DVector<f64>,
        ) -> Result<(DVector<f64>, f64), ImplicitSchurError> {
            let true_residual = rhs - &self.apply(solution)?;
            let true_norm = true_residual.norm();
            if !true_norm.is_finite() {
                return Err(ImplicitSchurError::NonFinite("true PCG residual"));
            }
            Ok((true_residual, true_norm))
        }

        #[cfg(test)]
        fn checked_result(
            &self,
            rhs: &DVector<f64>,
            solution: DVector<f64>,
            iterations: usize,
            recursive_norm: f64,
            target: f64,
        ) -> Result<PcgResult, ImplicitSchurError> {
            let (_, true_norm) = self.true_residual(rhs, &solution)?;
            if true_norm > target {
                return Err(ImplicitSchurError::ResidualCheckFailed {
                    iterations,
                    recursive_norm,
                    true_norm,
                    target,
                });
            }
            Ok(PcgResult {
                solution,
                iterations,
                residual_norm: true_norm,
                target,
            })
        }

        pub(super) fn complete_delta(
            &self,
            delta_pose: &DVector<f64>,
        ) -> Result<DVector<f64>, ImplicitSchurError> {
            if delta_pose.len() != self.dimension() {
                return Err(ImplicitSchurError::DimensionMismatch {
                    expected: self.dimension(),
                    actual: delta_pose.len(),
                });
            }
            if !delta_pose.iter().all(|value| value.is_finite()) {
                return Err(ImplicitSchurError::NonFinite("pose delta"));
            }
            let mut delta_landmarks = DVector::zeros(self.landmarks.len() * 3);
            for (index, (landmark, inverse)) in
                self.landmarks.iter().zip(&self.h_ll_inverse).enumerate()
            {
                let Some(inverse) = inverse else {
                    continue;
                };
                let mut accumulated = -landmark.b_l;
                for (pose, cross) in &landmark.cross {
                    let delta: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
                    accumulated -= cross.transpose() * delta;
                }
                let value = *inverse * accumulated;
                for component in 0..3 {
                    delta_landmarks[index * 3 + component] = value[component];
                }
            }
            if !delta_landmarks.iter().all(|value| value.is_finite()) {
                return Err(ImplicitSchurError::NonFinite("landmark delta"));
            }
            Ok(delta_landmarks)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use nalgebra::UnitQuaternion;

        fn explicit_schur(system: &NormalEquationsBa, lambda: f64) -> DMatrix<f64> {
            let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
                panic!("test oracle requires pose blocks");
            };
            let dimension = diagonal.len() * 6;
            let mut schur = DMatrix::zeros(dimension, dimension);
            for (pose, block) in diagonal.iter().enumerate() {
                let mut damped = *block;
                for component in 0..6 {
                    damped[(component, component)] += lambda;
                }
                schur
                    .fixed_view_mut::<6, 6>(pose * 6, pose * 6)
                    .copy_from(&damped);
            }
            for landmark in &system.landmarks {
                let mut h_ll = landmark.h_ll;
                for component in 0..3 {
                    h_ll[(component, component)] += lambda;
                }
                let Some(inverse) = h_ll.try_inverse() else {
                    continue;
                };
                for (pose, a) in &landmark.cross {
                    for (other_pose, b) in &landmark.cross {
                        let block = a * inverse * b.transpose();
                        for row in 0..6 {
                            for column in 0..6 {
                                schur[(pose * 6 + row, other_pose * 6 + column)] -=
                                    block[(row, column)];
                            }
                        }
                    }
                }
            }
            schur
        }

        fn synthetic_system() -> NormalEquationsBa {
            let diagonal = vec![
                Matrix6::from_diagonal(&Vector6::from_element(20.0)),
                Matrix6::from_diagonal(&Vector6::from_element(22.0)),
            ];
            let cross0 =
                Matrix6x3::from_fn(|row, column| 0.02 * (row as f64 + 1.0) * (column as f64 + 1.0));
            let cross1 = Matrix6x3::from_fn(|row, column| {
                0.015 * (row as f64 + 2.0) * (column as f64 + 1.0)
            });
            let cross2 =
                Matrix6x3::from_fn(|row, column| 0.01 * (row as f64 + 3.0) * (column as f64 + 2.0));
            NormalEquationsBa {
                h_pp: CameraHessian::PoseDiagonal(diagonal),
                b_p: DVector::from_iterator(12, (0..12).map(|index| 0.03 * (index + 1) as f64)),
                landmarks: vec![LandmarkBlock {
                    h_ll: Matrix3::from_diagonal(&Vector3::new(7.0, 8.0, 9.0)),
                    b_l: Vector3::new(0.2, -0.1, 0.3),
                    // The first two entries deliberately share pose 0, as two
                    // sensors on one rig frame would.
                    cross: vec![(0, cross0), (0, cross1), (1, cross2)],
                }],
            }
        }

        fn hand_check_system() -> NormalEquationsBa {
            let mut a = Matrix6x3::zeros();
            for component in 0..3 {
                a[(component, component)] = 1.0;
            }
            NormalEquationsBa {
                h_pp: CameraHessian::PoseDiagonal(vec![
                    Matrix6::from_diagonal(&Vector6::from_element(10.0)),
                    Matrix6::from_diagonal(&Vector6::from_element(10.0)),
                ]),
                b_p: DVector::from_iterator(12, (1..=12).map(|value| value as f64)),
                landmarks: vec![LandmarkBlock {
                    h_ll: Matrix3::from_diagonal(&Vector3::from_element(4.0)),
                    b_l: Vector3::new(1.0, 2.0, 3.0),
                    // B = 10 I, H_ll = 4 I, A = [I; 0], with two sensor
                    // observations sharing pose 0 and one observation at pose 1.
                    cross: vec![(0, a), (0, 2.0 * a), (1, -a)],
                }],
            }
        }

        fn full_normal_system(
            system: &NormalEquationsBa,
            lambda: f64,
        ) -> (DMatrix<f64>, DVector<f64>) {
            let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
                panic!("test oracle requires pose blocks");
            };
            let mut normal = DMatrix::zeros(15, 15);
            for (pose, block) in diagonal.iter().enumerate() {
                let mut damped = *block;
                for component in 0..6 {
                    damped[(component, component)] += lambda;
                }
                normal
                    .fixed_view_mut::<6, 6>(pose * 6, pose * 6)
                    .copy_from(&damped);
            }
            let mut h_ll = system.landmarks[0].h_ll;
            for component in 0..3 {
                h_ll[(component, component)] += lambda;
            }
            normal.fixed_view_mut::<3, 3>(12, 12).copy_from(&h_ll);
            for (pose, cross) in &system.landmarks[0].cross {
                for row in 0..6 {
                    for column in 0..3 {
                        normal[(pose * 6 + row, 12 + column)] += cross[(row, column)];
                        normal[(12 + column, pose * 6 + row)] += cross[(row, column)];
                    }
                }
            }
            let mut rhs = DVector::zeros(15);
            for (index, value) in system.b_p.iter().enumerate() {
                rhs[index] = -*value;
            }
            for (index, value) in system.landmarks[0].b_l.iter().enumerate() {
                rhs[12 + index] = -*value;
            }
            (normal, rhs)
        }

        #[test]
        fn implicit_operator_matches_explicit_schur_rhs_preconditioner_and_step() {
            let lambda = 0.25;
            let system = synthetic_system();
            let operator = ImplicitSchurOperator::new(&system, lambda).unwrap();
            let explicit = explicit_schur(&system, lambda);
            let x = DVector::from_iterator(12, (0..12).map(|index| 0.04 * (index + 1) as f64));
            let explicit_product = &explicit * &x;
            let implicit_product = operator.apply(&x).unwrap();
            assert!((&explicit_product - implicit_product).norm() < 1.0e-12);

            let mut explicit_rhs = -&system.b_p;
            let h_ll_inverse = operator.h_ll_inverse(0).unwrap();
            for (pose, cross) in &system.landmarks[0].cross {
                let update: Vector6<f64> = cross * h_ll_inverse * system.landmarks[0].b_l;
                for component in 0..6 {
                    explicit_rhs[pose * 6 + component] += update[component];
                }
            }
            assert_eq!(operator.rhs.as_slice(), explicit_rhs.as_slice());

            let mut direct_system = synthetic_system();
            let direct = solve_step(
                &mut direct_system,
                2,
                1,
                0,
                0,
                lambda,
                LinearSolver::Sparse,
                false,
                &mut None,
            )
            .unwrap();
            let result = operator
                .solve_pcg(operator.rhs(), PcgOptions::default())
                .unwrap();
            assert!(result.residual_norm <= result.target);
            assert!((&result.solution - &direct.0).norm() < 1.0e-9);
            let implicit_landmark_delta = operator.complete_delta(&result.solution).unwrap();
            assert!((&implicit_landmark_delta - direct.1).norm() < 1.0e-9);

            // The pose-0 preconditioner block must include the cross term between
            // the two repeated pose-0 sensor blocks, not two independent terms.
            let mut pose0_cross = system.landmarks[0].cross[0].1;
            pose0_cross += system.landmarks[0].cross[1].1;
            let mut expected = Matrix6::from_diagonal(&Vector6::from_element(20.0));
            for component in 0..6 {
                expected[(component, component)] += lambda;
            }
            expected -= pose0_cross * h_ll_inverse * pose0_cross.transpose();
            let probe = DVector::from_iterator(12, (0..12).map(|index| (index + 1) as f64));
            let applied = operator.apply_preconditioner(&probe).unwrap();
            let expected_inverse = expected.cholesky().unwrap().inverse();
            let expected_pose0 =
                expected_inverse * Vector6::from_iterator((0..6).map(|i| (i + 1) as f64));
            assert!((applied.fixed_rows::<6>(0).into_owned() - expected_pose0).norm() < 1.0e-12);
        }

        #[test]
        fn hand_check_fixture_matches_schur_and_full_normal_for_both_damping_values() {
            for lambda in [0.0, 0.5] {
                let system = hand_check_system();
                let operator = ImplicitSchurOperator::new(&system, lambda).unwrap();
                let schur = explicit_schur(&system, lambda);
                let expected_pose0 = if lambda == 0.0 { 7.75 } else { 8.5 };
                let expected_pose1 = if lambda == 0.0 {
                    9.75
                } else {
                    10.277777777777779
                };
                let expected_cross = if lambda == 0.0 {
                    0.75
                } else {
                    0.6666666666666666
                };
                for component in 0..3 {
                    assert!((schur[(component, component)] - expected_pose0).abs() < 1.0e-12);
                    assert!(
                        (schur[(6 + component, 6 + component)] - expected_pose1).abs() < 1.0e-12
                    );
                    assert!((schur[(component, 6 + component)] - expected_cross).abs() < 1.0e-12);
                    assert!((schur[(6 + component, component)] - expected_cross).abs() < 1.0e-12);
                }
                for component in 3..6 {
                    assert!((schur[(component, component)] - (10.0 + lambda)).abs() < 1.0e-12);
                    assert!(
                        (schur[(6 + component, 6 + component)] - (10.0 + lambda)).abs() < 1.0e-12
                    );
                    assert_eq!(schur[(component, 6 + component)], 0.0);
                }

                let expected_rhs0 = if lambda == 0.0 {
                    [-0.25, -0.5, -0.75]
                } else {
                    [-1.0 / 3.0, -2.0 / 3.0, -1.0]
                };
                let expected_rhs1 = if lambda == 0.0 {
                    [-7.25, -8.5, -9.75]
                } else {
                    [-65.0 / 9.0, -76.0 / 9.0, -29.0 / 3.0]
                };
                for component in 0..3 {
                    assert!((operator.rhs[component] - expected_rhs0[component]).abs() < 1.0e-12);
                    assert!(
                        (operator.rhs[6 + component] - expected_rhs1[component]).abs() < 1.0e-12
                    );
                }

                let (normal, full_rhs) = full_normal_system(&system, lambda);
                let full_delta = solve_normal_equations(&normal, &full_rhs).unwrap();
                let mut direct_system = hand_check_system();
                let direct = solve_step(
                    &mut direct_system,
                    2,
                    1,
                    0,
                    0,
                    lambda,
                    LinearSolver::Sparse,
                    false,
                    &mut None,
                )
                .unwrap();
                assert!((direct.0.clone() - full_delta.rows(0, 12)).norm() < 1.0e-10);
                assert!((direct.1.clone() - full_delta.rows(12, 3)).norm() < 1.0e-10);
                let result = operator
                    .solve_pcg(operator.rhs(), PcgOptions::default())
                    .unwrap();
                assert!((result.solution.clone() - full_delta.rows(0, 12)).norm() < 1.0e-9);
                assert!(
                    (operator.complete_delta(&result.solution).unwrap() - full_delta.rows(12, 3))
                        .norm()
                        < 1.0e-9
                );
            }
        }

        #[test]
        fn implicit_operator_matches_explicit_schur_and_direct_step_without_damping() {
            let system = synthetic_system();
            let operator = ImplicitSchurOperator::new(&system, 0.0).unwrap();
            let explicit = explicit_schur(&system, 0.0);
            let x = DVector::from_iterator(12, (0..12).map(|index| 0.02 * (index + 2) as f64));
            assert!((operator.apply(&x).unwrap() - explicit * &x).norm() < 1.0e-10);

            let mut direct_system = synthetic_system();
            let direct = solve_step(
                &mut direct_system,
                2,
                1,
                0,
                0,
                0.0,
                LinearSolver::Sparse,
                false,
                &mut None,
            )
            .unwrap();
            let result = operator
                .solve_pcg(operator.rhs(), PcgOptions::default())
                .unwrap();
            assert!((&result.solution - &direct.0).norm() < 1.0e-9);
            assert!(
                (operator.complete_delta(&result.solution).unwrap() - direct.1).norm() < 1.0e-9
            );
        }

        #[test]
        fn rig_sensor_observations_share_one_pose_slot_and_keep_cross_terms() {
            let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
            let mut ba = BundleAdjustment::new(camera.clone());
            ba.add_pose(0, Pose::identity());
            ba.add_landmark(0, Point3::new(0.3, -0.2, 4.0));
            ba.add_rig_observation(BaRigObservation {
                keyframe_id: 0,
                landmark_id: 0,
                xy: Point2::new(357.5, 215.0),
                camera: camera.clone(),
                sensor_from_rig: SE3::identity(),
            });
            ba.add_rig_observation(BaRigObservation {
                keyframe_id: 0,
                landmark_id: 0,
                xy: Point2::new(382.0, 215.0),
                camera,
                sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(0.2, 0.0, 0.0)),
            });
            let pose_index = BTreeMap::from([(0_u64, 0_usize)]);
            let landmark_index = BTreeMap::from([(0_u64, 0_usize)]);
            let system = build_normal_equations(
                &ba,
                &ba.camera.intrinsics().unwrap(),
                &pose_index,
                &landmark_index,
                &BTreeMap::new(),
                &BTreeMap::new(),
                &RobustKernel::None,
                None,
                false,
                true,
            );
            assert_eq!(system.landmarks[0].cross.len(), 2);
            assert_eq!(
                system.landmarks[0]
                    .cross
                    .iter()
                    .map(|(pose, _)| *pose)
                    .collect::<Vec<_>>(),
                vec![0, 0]
            );
            let operator = ImplicitSchurOperator::new(&system, 0.1).unwrap();
            let x = DVector::from_element(6, 0.2);
            let expected = explicit_schur(&system, 0.1) * &x;
            let actual = operator.apply(&x).unwrap();
            // Explicit and implicit forms reassociate cross-sensor products.  The
            // rounding scale is the sum of the unreduced Bx term and the
            // eliminated E C⁻¹ Eᵀx term, since their subtraction can cancel to a
            // much smaller Schur result.  This is scale-aware rather than a
            // pixel-constant absolute threshold and follows camera/rig scaling.
            let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
                panic!("rig matvec test requires pose blocks");
            };
            let mut base = DVector::zeros(x.len());
            for (pose, block) in diagonal.iter().enumerate() {
                let mut damped = *block;
                for component in 0..6 {
                    damped[(component, component)] += 0.1;
                }
                let value = damped * x.fixed_rows::<6>(pose * 6).into_owned();
                for component in 0..6 {
                    base[pose * 6 + component] = value[component];
                }
            }
            // Sum the magnitudes of the individual E C⁻¹ Eᵀx pair products, not
            // only their potentially-cancelled aggregate.  This captures the
            // arithmetic scale of the two evaluation orders.
            let inverse = (system.landmarks[0].h_ll + 0.1 * Matrix3::<f64>::identity())
                .try_inverse()
                .unwrap();
            let mut eliminated_term_scale = 0.0;
            for (_pose, a) in &system.landmarks[0].cross {
                for (other_pose, b) in &system.landmarks[0].cross {
                    let x_other: Vector6<f64> = x.fixed_rows::<6>(other_pose * 6).into_owned();
                    let term: Vector6<f64> = a * inverse * b.transpose() * x_other;
                    eliminated_term_scale += term.norm();
                }
            }
            let scale =
                (base.norm() + eliminated_term_scale + actual.norm() + expected.norm()).max(1.0);
            let tolerance = 64.0 * f64::EPSILON * scale;
            let difference = (&actual - &expected).norm();
            assert!(
                difference <= tolerance,
                "rig matvec diff={} scale={} tolerance={}",
                difference,
                scale,
                tolerance
            );
        }

        #[test]
        fn fixed_rotation_is_preserved_by_operator_and_back_substitution() {
            let mut system = synthetic_system();
            let pose_index = BTreeMap::from([(10_u64, 0_usize), (20_u64, 1_usize)]);
            let fixed = BTreeSet::from([10_u64]);
            constrain_fixed_pose_rotations(&fixed, &pose_index, &mut system);
            let operator = ImplicitSchurOperator::new(&system, 0.5).unwrap();
            assert_eq!(operator.rhs[3], 0.0);
            assert_eq!(operator.rhs[4], 0.0);
            assert_eq!(operator.rhs[5], 0.0);
            let mut x = DVector::zeros(12);
            x[3] = 1.0;
            x[4] = -2.0;
            x[5] = 3.0;
            let applied = operator.apply(&x).unwrap();
            assert_eq!(applied[3], 1.5);
            assert_eq!(applied[4], -3.0);
            assert_eq!(applied[5], 4.5);
            let result = operator
                .solve_pcg(operator.rhs(), PcgOptions::default())
                .unwrap();
            assert_eq!(result.solution[3], 0.0);
            assert_eq!(result.solution[4], 0.0);
            assert_eq!(result.solution[5], 0.0);
        }

        #[test]
        fn operator_rejects_dense_dimension_and_nonfinite_inputs() {
            let mut system = synthetic_system();
            let dimension = system.b_p.len();
            system.h_pp = CameraHessian::Dense(DMatrix::identity(dimension, dimension));
            assert!(matches!(
                ImplicitSchurOperator::new(&system, 0.0),
                Err(ImplicitSchurError::DenseCameraHessian)
            ));

            let mut system = synthetic_system();
            system.b_p = DVector::zeros(1);
            assert!(matches!(
                ImplicitSchurOperator::new(&system, 0.0),
                Err(ImplicitSchurError::DimensionMismatch {
                    expected: 12,
                    actual: 1,
                })
            ));

            let mut system = synthetic_system();
            system.landmarks[0].cross[0].1[(0, 0)] = f64::NAN;
            assert!(matches!(
                ImplicitSchurOperator::new(&system, 0.0),
                Err(ImplicitSchurError::NonFinite("landmark cross block"))
            ));

            let mut system = synthetic_system();
            system.landmarks[0].cross[0].0 = usize::MAX;
            assert!(matches!(
                ImplicitSchurOperator::new(&system, 0.0),
                Err(ImplicitSchurError::DimensionMismatch {
                    expected: 2,
                    actual: usize::MAX,
                })
            ));

            let system = synthetic_system();
            let operator = ImplicitSchurOperator::new(&system, 0.0).unwrap();
            let mut nonfinite = DVector::zeros(operator.dimension());
            nonfinite[0] = f64::NAN;
            assert_eq!(
                operator.apply(&nonfinite),
                Err(ImplicitSchurError::NonFinite("operator input"))
            );

            let system = synthetic_system();
            assert!(matches!(
                ImplicitSchurOperator::new(&system, f64::NAN),
                Err(ImplicitSchurError::InvalidLambda)
            ));
            let operator = ImplicitSchurOperator::new(&system, 0.0).unwrap();
            assert!(matches!(
                operator.solve_pcg(
                    operator.rhs(),
                    PcgOptions {
                        relative_tolerance: -1.0,
                        ..PcgOptions::default()
                    }
                ),
                Err(ImplicitSchurError::InvalidTolerance)
            ));
            let mut huge_rhs = operator.rhs().clone();
            huge_rhs[0] = f64::MAX;
            assert!(matches!(
                operator.solve_pcg(
                    &huge_rhs,
                    PcgOptions {
                        relative_tolerance: 2.0,
                        ..PcgOptions::default()
                    }
                ),
                Err(ImplicitSchurError::NonFinite("PCG target"))
            ));
            let invalid_restart = operator
                .solve_pcg_with_restart(
                    operator.rhs(),
                    PcgOptions {
                        relative_tolerance: f64::NAN,
                        ..PcgOptions::default()
                    },
                    1,
                )
                .unwrap_err();
            assert!(matches!(
                invalid_restart.error,
                ImplicitSchurError::InvalidTolerance
            ));
            assert_eq!(invalid_restart.diagnostics.pcg_iterations, None);
        }

        #[test]
        fn empty_pose_operator_rejects_pose_reduced_system() {
            let empty = NormalEquationsBa {
                h_pp: CameraHessian::PoseDiagonal(Vec::new()),
                b_p: DVector::zeros(0),
                landmarks: Vec::new(),
            };
            assert!(matches!(
                ImplicitSchurOperator::new(&empty, 0.0),
                Err(ImplicitSchurError::EmptySystem)
            ));
            // This only checks the prototype's empty pose-reduced input.  It is
            // not a production all-fixed-pose solver test: solve_step has a
            // separate p_count == 0 landmark-only branch.
        }

        #[test]
        fn singular_landmark_inverse_is_skipped_like_direct_solver() {
            let mut system = synthetic_system();
            system.landmarks[0].h_ll = Matrix3::zeros();
            let operator = ImplicitSchurOperator::new(&system, 0.0).unwrap();
            assert!(operator.h_ll_inverse(0).is_none());
            let expected = DVector::from_iterator(12, system.b_p.iter().map(|value| -value));
            assert_eq!(operator.rhs.as_slice(), expected.as_slice());
            let delta_landmarks = operator.complete_delta(&DVector::zeros(12)).unwrap();
            assert!(delta_landmarks.iter().all(|value| *value == 0.0));
        }

        #[test]
        fn pcg_reports_limits_curvature_and_zero_rhs_without_gauge_claim() {
            let system = synthetic_system();
            let operator = ImplicitSchurOperator::new(&system, 0.25).unwrap();
            let zero = DVector::zeros(operator.dimension());
            let zero_result = operator
                .solve_pcg(
                    &zero,
                    PcgOptions {
                        max_iterations: 0,
                        ..PcgOptions::default()
                    },
                )
                .unwrap();
            assert_eq!(zero_result.iterations, 0);
            assert_eq!(zero_result.residual_norm, 0.0);
            let zero_restart = operator
                .solve_pcg_with_restart(
                    &zero,
                    PcgOptions {
                        max_iterations: 0,
                        ..PcgOptions::default()
                    },
                    1,
                )
                .unwrap();
            assert_eq!(zero_restart.diagnostics.pcg_iterations, Some(0));
            assert_eq!(zero_restart.diagnostics.restarts, 0);

            let limited = operator.solve_pcg(
                operator.rhs(),
                PcgOptions {
                    max_iterations: 0,
                    ..PcgOptions::default()
                },
            );
            assert!(matches!(
                limited,
                Err(ImplicitSchurError::MaxIterations { .. })
            ));
            let limited_restart = operator
                .solve_pcg_with_restart(
                    operator.rhs(),
                    PcgOptions {
                        max_iterations: 0,
                        ..PcgOptions::default()
                    },
                    1,
                )
                .unwrap_err();
            assert!(matches!(
                limited_restart.error,
                ImplicitSchurError::MaxIterations { .. }
            ));
            assert_eq!(limited_restart.diagnostics.pcg_iterations, Some(0));

            let bad_solution = DVector::zeros(operator.dimension());
            assert!(matches!(
                operator.checked_result(operator.rhs(), bad_solution, 0, 0.0, 0.0),
                Err(ImplicitSchurError::ResidualCheckFailed { .. })
            ));

            let mut indefinite = synthetic_system();
            if let CameraHessian::PoseDiagonal(diagonal) = &mut indefinite.h_pp {
                for block in diagonal {
                    *block = Matrix6::from_diagonal(&Vector6::from_element(5.0));
                }
            }
            indefinite.landmarks[0].h_ll = Matrix3::identity();
            indefinite.landmarks[0].cross = vec![
                (
                    0,
                    Matrix6x3::from_fn(
                        |row, column| {
                            if row == 0 && column == 0 {
                                2.0
                            } else {
                                0.0
                            }
                        },
                    ),
                ),
                (
                    1,
                    Matrix6x3::from_fn(
                        |row, column| {
                            if row == 0 && column == 0 {
                                2.0
                            } else {
                                0.0
                            }
                        },
                    ),
                ),
            ];
            let indefinite_operator = ImplicitSchurOperator::new(&indefinite, 0.0).unwrap();
            let mut negative_eigenvector = DVector::zeros(indefinite_operator.dimension());
            negative_eigenvector[0] = 1.0;
            negative_eigenvector[6] = 1.0;
            assert!(matches!(
                indefinite_operator.solve_pcg(&negative_eigenvector, PcgOptions::default()),
                Err(ImplicitSchurError::NonPositiveCurvature)
            ));
            let curvature_restart = indefinite_operator
                .solve_pcg_with_restart(&negative_eigenvector, PcgOptions::default(), 1)
                .unwrap_err();
            assert!(matches!(
                curvature_restart.error,
                ImplicitSchurError::NonPositiveCurvature
            ));
            assert_eq!(curvature_restart.diagnostics.pcg_iterations, Some(0));
            assert_eq!(curvature_restart.diagnostics.restarts, 0);
        }

        #[test]
        fn injected_recursive_residual_gap_restarts_once_without_resetting_budget() {
            let system = synthetic_system();
            let operator = ImplicitSchurOperator::new(&system, 0.25).unwrap();
            let options = PcgOptions {
                max_iterations: 32,
                // The test-only hook reports a deterministic zero
                // recursive residual after the first alpha update.  The
                // following honest b-Ax check forces restart.
                relative_tolerance: 1.0e-12,
                absolute_tolerance: 1.0e-12,
            };
            let no_restart = operator
                .solve_pcg_with_injected_recursive_residual_for_test(operator.rhs(), options, 0)
                .unwrap_err();
            assert!(matches!(
                no_restart.error,
                ImplicitSchurError::ResidualCheckFailed { iterations: 1, .. }
            ));
            assert_eq!(no_restart.diagnostics.pcg_iterations, Some(1));
            assert_eq!(no_restart.diagnostics.restarts, 0);

            let mut capped_options = options;
            capped_options.max_iterations = 1;
            let capped = operator
                .solve_pcg_with_injected_recursive_residual_for_test(
                    operator.rhs(),
                    capped_options,
                    1,
                )
                .unwrap_err();
            assert!(matches!(
                capped.error,
                ImplicitSchurError::ResidualCheckFailed { iterations: 1, .. }
            ));
            assert_eq!(capped.diagnostics.pcg_iterations, Some(1));
            assert_eq!(capped.diagnostics.restarts, 0);

            let run = operator
                .solve_pcg_with_injected_recursive_residual_for_test(operator.rhs(), options, 1)
                .expect("one injected recursive residual gap should be recoverable");
            assert_eq!(run.diagnostics.restarts, 1);
            assert_eq!(run.diagnostics.failed_true_residual_rechecks, 1);
            assert!(run.diagnostics.true_residual_rechecks >= 2);
            assert!(run.diagnostics.pcg_iterations.unwrap() <= 32);
            assert!(run.result.residual_norm <= run.result.target);

            let repeat = operator
                .solve_pcg_with_injected_recursive_residual_for_test(operator.rhs(), options, 1)
                .expect("the injected restart must be deterministic");
            assert_eq!(run, repeat);
        }

        #[test]
        fn repeated_pcg_runs_are_deterministic() {
            let system = synthetic_system();
            let operator = ImplicitSchurOperator::new(&system, 0.25).unwrap();
            let first = operator
                .solve_pcg(operator.rhs(), PcgOptions::default())
                .unwrap();
            let second = operator
                .solve_pcg(operator.rhs(), PcgOptions::default())
                .unwrap();
            assert_eq!(first, second);
        }
    }
}

#[cfg(test)]
mod matrix_free_ba_api_tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    fn make_problem() -> BundleAdjustment {
        let camera = Camera::pinhole(7, 640, 480, 420.0, 418.0, 320.0, 240.0);
        let mut problem = BundleAdjustment::new(camera.clone());
        let truth_pose0 = Pose::identity();
        let truth_pose1 = Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.01, -0.02, 0.015),
            Vector3::new(-0.18, 0.015, 0.02),
        );
        let truth_pose2 = Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(-0.015, 0.025, -0.01),
            Vector3::new(-0.34, -0.01, 0.04),
        );
        problem.poses.insert(0, truth_pose0.clone());
        problem.poses.insert(
            1,
            Pose::from_world_to_camera(
                UnitQuaternion::from_euler_angles(0.012, -0.018, 0.016),
                Vector3::new(-0.205, 0.022, 0.026),
            ),
        );
        problem.poses.insert(
            2,
            Pose::from_world_to_camera(
                UnitQuaternion::from_euler_angles(-0.013, 0.023, -0.012),
                Vector3::new(-0.365, -0.004, 0.045),
            ),
        );
        problem.fixed_poses.insert(0);
        let points = [
            Point3::new(-0.8, -0.45, 3.8),
            Point3::new(-0.35, 0.55, 4.2),
            Point3::new(0.05, -0.25, 4.6),
            Point3::new(0.45, 0.35, 5.0),
            Point3::new(0.85, -0.5, 5.4),
            Point3::new(-0.65, 0.25, 5.8),
            Point3::new(0.25, 0.7, 6.2),
            Point3::new(0.7, 0.15, 6.8),
        ];
        for (landmark_id, truth_point) in points.into_iter().enumerate() {
            problem.landmarks.insert(
                landmark_id as u64,
                Point3::from(truth_point.coords + Vector3::new(0.006, -0.004, 0.008)),
            );
            for (keyframe_id, pose) in [(0, &truth_pose0), (1, &truth_pose1), (2, &truth_pose2)] {
                let xy = camera
                    .project(&pose.transform_world_point(&truth_point))
                    .expect("synthetic point must project");
                problem.observations.push(BaObservation {
                    keyframe_id,
                    landmark_id: landmark_id as u64,
                    xy,
                });
            }
        }
        problem.fixed_landmarks.insert(0);
        problem
    }

    fn matrix_free_config() -> BaConfig {
        BaConfig {
            max_iterations: 4,
            linear_solver: LinearSolver::Sparse,
            ..BaConfig::default()
        }
    }

    #[test]
    fn matrix_free_mono_matches_sparse_cost_and_keeps_anchor_fixed() {
        let problem = make_problem();
        let mut direct = problem.clone();
        let mut matrix_free = problem.clone();
        let config = matrix_free_config();
        let direct_result = direct.optimize(&config).unwrap();
        let matrix_result = matrix_free
            .optimize_matrix_free(&config, MatrixFreeBaOptions::default())
            .unwrap();
        assert!(matrix_result.final_cost <= matrix_result.initial_cost);
        assert!(direct_result.final_cost <= direct_result.initial_cost);
        assert!((direct_result.final_cost - matrix_result.final_cost).abs() < 1.0e-6);
        assert_eq!(matrix_free.poses[&0], problem.poses[&0]);
        for id in 1..=2 {
            assert!(matrix_free.poses[&id] != problem.poses[&id]);
            assert!(
                (direct.poses[&id].world_to_camera.matrix()
                    - matrix_free.poses[&id].world_to_camera.matrix())
                .norm()
                    < 1.0e-5
            );
        }
        assert_eq!(matrix_free.landmarks[&0], problem.landmarks[&0]);
        for id in 1..8 {
            assert!(
                (direct.landmarks[&id].coords - matrix_free.landmarks[&id].coords).norm() < 1.0e-5
            );
        }
        assert!(matrix_result
            .matrix_free_iterations
            .iter()
            .all(|iteration| iteration.pcg_failure.is_none()));

        let mut dense_config = config;
        dense_config.linear_solver = LinearSolver::Dense;
        let mut dense_dispatch = problem.clone();
        let dense_dispatch_result = dense_dispatch
            .optimize_matrix_free(&dense_config, MatrixFreeBaOptions::default())
            .unwrap();
        assert_eq!(matrix_result, dense_dispatch_result);

        let mut repeat = problem.clone();
        let repeat_result = repeat
            .optimize_matrix_free(&config, MatrixFreeBaOptions::default())
            .unwrap();
        assert_eq!(matrix_result, repeat_result);
        assert_eq!(matrix_free, repeat);
    }

    #[test]
    fn matrix_free_supports_rectified_stereo_and_rig_visual_factors() {
        let mut stereo = make_problem();
        stereo.fixed_landmarks.clear();
        let camera = stereo.camera.clone();
        let baseline = 0.12;
        stereo.stereo_baseline = Some(baseline);
        stereo.stereo_observations = stereo
            .observations
            .drain(..)
            .map(|observation| {
                let pose = &stereo.poses[&observation.keyframe_id];
                let point = &stereo.landmarks[&observation.landmark_id];
                let xc = pose.transform_world_point(point);
                BaStereoObservation {
                    keyframe_id: observation.keyframe_id,
                    landmark_id: observation.landmark_id,
                    xy: observation.xy,
                    u_right: observation.xy.x - camera.params[0] * baseline / xc.z,
                }
            })
            .collect();
        let mut stereo_direct = stereo.clone();
        let stereo_direct_result = stereo_direct.optimize(&matrix_free_config()).unwrap();
        let stereo_result = stereo
            .optimize_matrix_free(&matrix_free_config(), MatrixFreeBaOptions::default())
            .unwrap();
        assert!(stereo_result.final_cost.is_finite());
        assert!(stereo_result.final_cost < stereo_result.initial_cost);
        assert!((stereo_direct_result.final_cost - stereo_result.final_cost).abs() < 1.0e-6);

        let mut rig = make_problem();
        rig.fixed_landmarks.clear();
        let camera = rig.camera.clone();
        let sensor1 = SE3::new(
            UnitQuaternion::from_euler_angles(0.02, -0.01, 0.03),
            Vector3::new(0.22, -0.015, 0.01),
        );
        let observations = std::mem::take(&mut rig.observations);
        for observation in observations {
            let pose = &rig.poses[&observation.keyframe_id].world_to_camera;
            let point = &rig.landmarks[&observation.landmark_id];
            for sensor_from_rig in [SE3::identity(), sensor1.clone()] {
                let sensor_pose = sensor_from_rig.compose(pose);
                let xy = camera
                    .project(&sensor_pose.transform_point(point))
                    .expect("rig synthetic point must project");
                rig.rig_observations.push(BaRigObservation {
                    keyframe_id: observation.keyframe_id,
                    landmark_id: observation.landmark_id,
                    xy,
                    camera: camera.clone(),
                    sensor_from_rig,
                });
            }
        }
        let rig_before_extrinsics: Vec<SE3> = rig
            .rig_observations
            .iter()
            .map(|observation| observation.sensor_from_rig.clone())
            .collect();
        rig.poses.get_mut(&1).unwrap().world_to_camera.translation.x += 0.03;
        rig.poses.get_mut(&2).unwrap().world_to_camera.translation.y -= 0.02;
        for id in 1..8 {
            rig.landmarks.get_mut(&id).unwrap().coords.z += 0.015;
        }
        let mut rig_scaled = rig.clone();
        rig_scaled.fixed_landmarks.insert(0);
        // Keep the gauge anchor pose fixed through `fixed_poses`, while
        // constraining the rotation of a genuinely variable pose so the
        // scaled path exercises the identity rotation rows.
        rig_scaled.fixed_pose_rotations.insert(1);
        let anchor_rig_pose = rig_scaled.poses[&0].clone();
        let fixed_rig_rotation = rig_scaled.poses[&1].world_to_camera.rotation;
        let fixed_rig_landmark = rig_scaled.landmarks[&0];
        let scaled_extrinsics: Vec<SE3> = rig_scaled
            .rig_observations
            .iter()
            .map(|observation| observation.sensor_from_rig.clone())
            .collect();
        let mut rig_direct = rig.clone();
        let rig_direct_result = rig_direct.optimize(&matrix_free_config()).unwrap();
        let rig_result = rig
            .optimize_matrix_free(&matrix_free_config(), MatrixFreeBaOptions::default())
            .unwrap();
        assert!(rig_result.final_cost.is_finite());
        assert!(rig_result.final_cost < rig_result.initial_cost);
        assert!((rig_direct_result.final_cost - rig_result.final_cost).abs() < 1.0e-6);
        for id in 1..=2 {
            assert!(
                (rig_direct.poses[&id].world_to_camera.matrix()
                    - rig.poses[&id].world_to_camera.matrix())
                .norm()
                    < 1.0e-5
            );
        }
        for id in 0..8 {
            assert!((rig_direct.landmarks[&id].coords - rig.landmarks[&id].coords).norm() < 1.0e-5);
        }
        assert_eq!(
            rig.rig_observations
                .iter()
                .map(|observation| observation.sensor_from_rig.clone())
                .collect::<Vec<_>>(),
            rig_before_extrinsics
        );

        let scaled_result = rig_scaled
            .optimize_matrix_free_column_scaled(
                &matrix_free_config(),
                MatrixFreeBaColumnScalingOptions::default(),
            )
            .unwrap();
        assert!(scaled_result.ba.final_cost.is_finite());
        assert!(scaled_result.ba.final_cost <= scaled_result.ba.initial_cost);
        assert!(scaled_result
            .ba
            .iterations
            .iter()
            .any(|iteration| iteration.step_accepted));
        assert_eq!(rig_scaled.poses[&0], anchor_rig_pose);
        assert_eq!(
            rig_scaled.poses[&1].world_to_camera.rotation,
            fixed_rig_rotation
        );
        assert_eq!(rig_scaled.landmarks[&0], fixed_rig_landmark);
        assert_eq!(
            rig_scaled
                .rig_observations
                .iter()
                .map(|observation| observation.sensor_from_rig.clone())
                .collect::<Vec<_>>(),
            scaled_extrinsics
        );
        assert!(!scaled_result.scaling_iterations.is_empty());

        let mut rig_adaptive = rig.clone();
        rig_adaptive.fixed_landmarks.insert(0);
        rig_adaptive.fixed_pose_rotations.insert(1);
        let adaptive_anchor_pose = rig_adaptive.poses[&0].clone();
        let adaptive_fixed_rotation = rig_adaptive.poses[&1].world_to_camera.rotation;
        let adaptive_fixed_landmark = rig_adaptive.landmarks[&0];
        let adaptive_result = rig_adaptive
            .optimize_matrix_free_column_scaled_adaptive(
                &matrix_free_config(),
                MatrixFreeBaColumnScalingOptions::default(),
            )
            .unwrap();
        assert!(adaptive_result.ba.final_cost.is_finite());
        assert!(adaptive_result
            .ba
            .iterations
            .iter()
            .any(|iteration| iteration.step_accepted));
        assert_eq!(rig_adaptive.poses[&0], adaptive_anchor_pose);
        assert_eq!(
            rig_adaptive.poses[&1].world_to_camera.rotation,
            adaptive_fixed_rotation
        );
        assert_eq!(rig_adaptive.landmarks[&0], adaptive_fixed_landmark);
    }

    #[test]
    fn matrix_free_rejects_unsupported_input_without_mutation() {
        let mut no_anchor = make_problem();
        no_anchor.fixed_poses.clear();
        let before = no_anchor.clone();
        assert!(matches!(
            no_anchor.optimize_matrix_free(&matrix_free_config(), MatrixFreeBaOptions::default()),
            Err(MatrixFreeBaError::Ineligible(_))
        ));
        assert_eq!(no_anchor, before);

        let mut with_prior = make_problem();
        with_prior.position_prior = Some(PositionPrior::default());
        assert!(matches!(
            with_prior.optimize_matrix_free(&matrix_free_config(), MatrixFreeBaOptions::default()),
            Err(MatrixFreeBaError::Ineligible(_))
        ));

        let mut bad_config = make_problem();
        let mut config = matrix_free_config();
        config.initial_lambda = None;
        assert!(matches!(
            bad_config.optimize_matrix_free(&config, MatrixFreeBaOptions::default()),
            Err(MatrixFreeBaError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn matrix_free_preflight_rejects_nonfinite_and_unsupported_cases_deterministically() {
        let mut cases: Vec<(&str, BundleAdjustment, BaConfig)> = Vec::new();
        for (label, initial_lambda) in [
            ("none-lambda", None),
            ("zero-lambda", Some(0.0)),
            ("nan-lambda", Some(f64::NAN)),
        ] {
            let mut config = matrix_free_config();
            config.initial_lambda = initial_lambda;
            cases.push((label, make_problem(), config));
        }
        let mut reversed = matrix_free_config();
        reversed.min_lambda = 2.0;
        reversed.max_lambda = 1.0;
        cases.push(("reversed-lambda", make_problem(), reversed));

        let mut invalid_kernel = matrix_free_config();
        invalid_kernel.robust_kernel = RobustKernel::Huber { delta: f64::NAN };
        cases.push(("invalid-kernel", make_problem(), invalid_kernel));

        let mut nonfinite_pose = make_problem();
        nonfinite_pose
            .poses
            .get_mut(&1)
            .unwrap()
            .world_to_camera
            .translation
            .x = f64::NAN;
        cases.push(("nonfinite-pose", nonfinite_pose, matrix_free_config()));

        let mut nonfinite_landmark = make_problem();
        nonfinite_landmark.landmarks.get_mut(&1).unwrap().coords.y = f64::NAN;
        cases.push((
            "nonfinite-landmark",
            nonfinite_landmark,
            matrix_free_config(),
        ));

        let mut distorted = make_problem();
        distorted.camera =
            Camera::pinhole_radial(7, 640, 480, 420.0, 418.0, 320.0, 240.0, 0.01, 0.0);
        cases.push(("nonzero-distortion", distorted, matrix_free_config()));

        let mut fake_anchor = make_problem();
        fake_anchor.fixed_poses.insert(99);
        fake_anchor.fixed_poses.remove(&0);
        cases.push(("unknown-anchor", fake_anchor, matrix_free_config()));

        let mut all_fixed = make_problem();
        all_fixed.fixed_poses.insert(1);
        all_fixed.fixed_poses.insert(2);
        cases.push(("all-fixed", all_fixed, matrix_free_config()));

        let mut unsupported_state = make_problem();
        unsupported_state.velocities.insert(1, Vector3::zeros());
        cases.push(("velocity-state", unsupported_state, matrix_free_config()));

        for (label, mut problem, config) in cases {
            let pose_snapshot = format!("{:?}", problem.poses);
            let landmark_snapshot = format!("{:?}", problem.landmarks);
            let result = problem.optimize_matrix_free(&config, MatrixFreeBaOptions::default());
            assert!(
                matches!(
                    result,
                    Err(MatrixFreeBaError::InvalidConfiguration(_))
                        | Err(MatrixFreeBaError::Ineligible(_))
                ),
                "{label}: unexpected result {result:?}"
            );
            assert_eq!(
                format!("{:?}", problem.poses),
                pose_snapshot,
                "{label} poses changed"
            );
            assert_eq!(
                format!("{:?}", problem.landmarks),
                landmark_snapshot,
                "{label} landmarks changed"
            );
        }
    }

    #[test]
    fn matrix_free_failed_pcg_rolls_back_and_reports_attempt() {
        let mut problem = make_problem();
        let before = problem.clone();
        let mut config = matrix_free_config();
        config.max_iterations = 3;
        let options = MatrixFreeBaOptions {
            max_pcg_iterations: 1,
            pcg_relative_tolerance: 0.0,
            pcg_absolute_tolerance: 1.0e-30,
        };
        let result = problem.optimize_matrix_free(&config, options).unwrap();
        assert_eq!(problem, before);
        assert_eq!(result.matrix_free_iterations.len(), 3);
        assert!(result.matrix_free_iterations.iter().all(|iteration| {
            iteration.pcg_failure.is_some()
                && iteration.pcg_iterations == Some(1)
                && iteration.pcg_residual_norm.is_some()
        }));
        assert_eq!(result.iterations.len(), 3);
        assert!(result.iterations[0].lambda < result.iterations[1].lambda);
        assert!(result.iterations[1].lambda < result.iterations[2].lambda);
    }

    #[test]
    fn matrix_free_column_scaled_is_deterministic_and_rolls_back_failed_steps() {
        let problem = make_problem();
        let config = matrix_free_config();
        let mut first = problem.clone();
        let mut second = problem.clone();
        let first_result = first
            .optimize_matrix_free_column_scaled(
                &config,
                MatrixFreeBaColumnScalingOptions::default(),
            )
            .unwrap();
        let second_result = second
            .optimize_matrix_free_column_scaled(
                &config,
                MatrixFreeBaColumnScalingOptions::default(),
            )
            .unwrap();
        assert_eq!(first_result, second_result);
        assert_eq!(first, second);
        assert!(!first_result.scaling_iterations.is_empty());

        let mut failed = make_problem();
        let before = failed.clone();
        let mut failure_config = matrix_free_config();
        failure_config.max_iterations = 3;
        let failure_result = failed
            .optimize_matrix_free_column_scaled(
                &failure_config,
                MatrixFreeBaColumnScalingOptions {
                    pcg: MatrixFreeBaOptions {
                        max_pcg_iterations: 1,
                        pcg_relative_tolerance: 0.0,
                        pcg_absolute_tolerance: 1.0e-30,
                    },
                },
            )
            .unwrap();
        assert_eq!(failed, before);
        assert_eq!(failure_result.ba.iterations.len(), 3);
        assert!(failure_result
            .ba
            .matrix_free_iterations
            .iter()
            .all(|iteration| iteration.pcg_failure.is_some()));
    }

    #[test]
    fn adaptive_damping_is_deterministic_and_records_candidate_gates() {
        let problem = make_problem();
        let config = matrix_free_config();
        let mut first = problem.clone();
        let mut second = problem;
        let first_result = first
            .optimize_matrix_free_column_scaled_adaptive(
                &config,
                MatrixFreeBaColumnScalingOptions::default(),
            )
            .unwrap();
        let second_result = second
            .optimize_matrix_free_column_scaled_adaptive(
                &config,
                MatrixFreeBaColumnScalingOptions::default(),
            )
            .unwrap();
        assert_eq!(first_result, second_result);
        assert_eq!(first, second);
        assert!(!first_result.adaptive_iterations.is_empty());
        assert!(first_result
            .adaptive_iterations
            .iter()
            .all(|stats| stats.next_lambda.is_finite()
                && stats.next_lambda >= config.min_lambda
                && stats.next_lambda <= config.max_lambda
                && stats.cost_gate.is_some()
                && stats.feasibility_gate.is_some()
                && stats.nonprojectable_after.is_some()));
        assert!(first_result
            .adaptive_iterations
            .iter()
            .any(|stats| stats.accepted));
    }

    #[test]
    fn adaptive_damping_linear_failure_rolls_back_and_records_none_candidate_metrics() {
        let mut problem = make_problem();
        let before = problem.clone();
        let mut config = matrix_free_config();
        config.max_iterations = 3;
        let result = problem
            .optimize_matrix_free_column_scaled_adaptive(
                &config,
                MatrixFreeBaColumnScalingOptions {
                    pcg: MatrixFreeBaOptions {
                        max_pcg_iterations: 1,
                        pcg_relative_tolerance: 0.0,
                        pcg_absolute_tolerance: 1.0e-30,
                    },
                },
            )
            .unwrap();
        assert_eq!(problem, before);
        assert_eq!(result.adaptive_iterations.len(), config.max_iterations);
        assert!(result.adaptive_iterations.iter().all(|stats| stats
            .predicted_undamped_squared_decrease
            .is_none()
            && stats.actual_cost_decrease.is_none()
            && stats.rho.is_none()
            && stats.cost_gate.is_none()
            && stats.feasibility_gate.is_none()
            && stats.nonprojectable_after.is_none()
            && !stats.accepted
            && stats.reason == "linear_failure"));
        assert!(result
            .adaptive_iterations
            .windows(2)
            .all(|pair| pair[1].solve_lambda == pair[0].next_lambda));
        assert!(result
            .ba
            .matrix_free_iterations
            .iter()
            .all(|stats| stats.pcg_failure.is_some()));
    }

    #[test]
    fn adaptive_damping_rejects_nonpositive_rho_without_panic() {
        let decision =
            adaptive_step_decision(Some(Ok(f64::MAX)), f64::from_bits(1), 0.0, 0, 0, true, true);
        assert_eq!(decision.actual_cost_decrease, Some(f64::from_bits(1)));
        assert_eq!(decision.rho, Some(0.0));
        assert!(!decision.accepted);
        assert_eq!(decision.reason, "candidate_rejected_rho_nonpositive");
        assert_eq!(
            adaptive_accepted_lambda(1.0, 0.0, 1.0e-6, 1.0e6),
            Err("adaptive rho is non-finite or non-positive")
        );
    }

    #[test]
    fn adaptive_damping_policy_clamps_and_distinguishes_prediction_failure() {
        let high_rho = adaptive_accepted_lambda(1.0, 2.0, 1.0e-6, 1.0e6).unwrap();
        assert_eq!(high_rho, 1.0 / 3.0);
        let moderate_rho = adaptive_accepted_lambda(1.0, 0.25, 1.0e-6, 1.0e6).unwrap();
        assert!((moderate_rho - 1.125).abs() < 1.0e-15);
        let bounded = adaptive_accepted_lambda(1.0e6, 2.0, 1.0e-6, 10.0).unwrap();
        assert_eq!(bounded, 10.0);
        assert_eq!(
            adaptive_accepted_lambda(1.0e-12, 1.0e-300, 1.0e-6, 1.0e6).unwrap(),
            1.0e-6
        );
        assert_eq!(
            adaptive_accepted_lambda(1.0, f64::MAX, 1.0e-6, 1.0e6).unwrap(),
            1.0 / 3.0
        );

        let cost_rejected = adaptive_step_decision(Some(Ok(1.0)), 2.0, 3.0, 0, 0, false, true);
        assert!(!cost_rejected.accepted);
        assert_eq!(cost_rejected.reason, "candidate_rejected_cost_gate");
        let feasibility_rejected =
            adaptive_step_decision(Some(Ok(1.0)), 2.0, 1.0, 0, 1, true, false);
        assert!(!feasibility_rejected.accepted);
        assert_eq!(
            feasibility_rejected.reason,
            "candidate_rejected_feasibility_gate"
        );
        let nonpositive_prediction =
            adaptive_step_decision(Some(Ok(0.0)), 2.0, 1.0, 0, 0, true, true);
        assert!(!nonpositive_prediction.accepted);
        assert_eq!(nonpositive_prediction.rho, None);
        assert_eq!(
            nonpositive_prediction.reason,
            "candidate_rejected_rho:prediction_nonpositive_or_nonfinite"
        );
        let infinite_prediction =
            adaptive_step_decision(Some(Ok(f64::INFINITY)), 2.0, 1.0, 0, 0, true, true);
        assert!(!infinite_prediction.accepted);
        assert_eq!(
            infinite_prediction.reason,
            "candidate_rejected_prediction:adaptive prediction is non-finite"
        );
        let overflowed_cost_difference =
            adaptive_step_decision(Some(Ok(1.0)), f64::MAX, -f64::MAX, 0, 0, true, true);
        assert!(!overflowed_cost_difference.accepted);
        assert_eq!(
            overflowed_cost_difference.reason,
            "candidate_rejected_rho:actual_cost_decrease_nonfinite"
        );

        let rejected = adaptive_step_decision(
            Some(Err("adaptive prediction is non-finite")),
            2.0,
            1.0,
            0,
            0,
            true,
            true,
        );
        assert!(!rejected.accepted);
        assert_eq!(rejected.prediction, None);
        assert_eq!(rejected.rho, None);
        assert_eq!(
            rejected.reason,
            "candidate_rejected_prediction:adaptive prediction is non-finite"
        );

        let invalid = adaptive_step_decision(Some(Ok(f64::NAN)), 2.0, 1.0, 0, 0, true, true);
        assert!(!invalid.accepted);
        assert_eq!(invalid.prediction, None);
        assert_eq!(invalid.rho, None);
    }

    #[test]
    fn adaptive_damping_rejects_initial_nonprojectable_state_without_mutation() {
        let mut problem = make_problem();
        problem.landmarks.get_mut(&1).unwrap().coords.z = -1.0;
        let before = problem.clone();
        let result = problem.optimize_matrix_free_column_scaled_adaptive(
            &matrix_free_config(),
            MatrixFreeBaColumnScalingOptions::default(),
        );
        assert!(matches!(
            result,
            Err(MatrixFreeBaError::Ineligible(
                "adaptive damping requires zero initial non-projectable observations"
            ))
        ));
        assert_eq!(problem, before);
    }

    #[test]
    fn matrix_free_restart_zero_matches_default_result_and_state() {
        let problem = make_problem();
        let config = matrix_free_config();
        let mut default_path = problem.clone();
        let mut restart_path = problem;
        let default_result = default_path
            .optimize_matrix_free(&config, MatrixFreeBaOptions::default())
            .unwrap();
        let restart_result = restart_path
            .optimize_matrix_free_with_restart(
                &config,
                MatrixFreeBaOptions::default(),
                MatrixFreeBaRestartOptions::default(),
            )
            .unwrap();
        assert_eq!(default_result, restart_result.ba);
        assert_eq!(default_path, restart_path);
        assert_eq!(
            restart_result.restart_iterations.len(),
            restart_result.ba.matrix_free_iterations.len()
        );
        assert!(restart_result
            .restart_iterations
            .iter()
            .all(|stats| stats.restarts <= 1 && stats.terminal_failure.is_none()));
    }

    #[test]
    fn matrix_free_restart_limit_rejects_without_mutation() {
        let mut problem = make_problem();
        let before = problem.clone();
        let result = problem.optimize_matrix_free_with_restart(
            &matrix_free_config(),
            MatrixFreeBaOptions::default(),
            MatrixFreeBaRestartOptions {
                max_restarts_per_solve: 2,
            },
        );
        assert!(matches!(
            result,
            Err(MatrixFreeBaError::InvalidConfiguration(_))
        ));
        assert_eq!(problem, before);
    }

    #[test]
    fn matrix_free_restart_failure_records_bounded_stats_and_rolls_back() {
        let mut problem = make_problem();
        let before = problem.clone();
        let mut config = matrix_free_config();
        config.max_iterations = 3;
        let result = problem
            .optimize_matrix_free_with_restart(
                &config,
                MatrixFreeBaOptions {
                    max_pcg_iterations: 1,
                    pcg_relative_tolerance: 0.0,
                    pcg_absolute_tolerance: 1.0e-30,
                },
                MatrixFreeBaRestartOptions {
                    max_restarts_per_solve: 1,
                },
            )
            .unwrap();
        assert_eq!(problem, before);
        assert_eq!(result.restart_iterations.len(), config.max_iterations);
        assert!(result.restart_iterations.iter().all(|stats| {
            stats.pcg_iterations == Some(1)
                && stats.true_residual_rechecks >= 1
                && stats.failed_true_residual_rechecks >= 1
                && stats.restarts == 0
                && stats.terminal_failure.is_some()
        }));
    }
}

#[cfg(test)]
mod matrix_free_real_oracle_tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::env;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::path::Path;

    const MAX_VARIABLE_POSES: usize = 512;
    const MAX_LANDMARKS: usize = 8_192;
    const MAX_OBSERVATIONS: usize = 262_144;
    const MAX_SCHUR_SCALARS: usize = 3_072;
    const MAX_CROSS_PAIR_WORK: usize = 64_000_000;
    const MAX_CAMERAS: usize = 64;
    const ORACLE_PC_TOLERANCE: f64 = 1.0e-12;

    #[derive(Debug, Clone, PartialEq)]
    struct OracleFixture {
        source_hashes: BTreeMap<String, String>,
        initial_cost: f64,
        initial_cost_bits: u64,
        ba: BundleAdjustment,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct PredictedDecrease {
        half_damped: f64,
        squared_damped: f64,
        squared_undamped: f64,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Feasibility {
        finite: bool,
        pose_true_residual: f64,
        implicit_pose_true_residual: Option<f64>,
        max_landmark_backsub_residual: f64,
        geometry_observation_count: usize,
        geometry_invalid_observations: usize,
        geometry_nonpositive_depth: usize,
        geometry_cost: f64,
        geometry_rms_error: f64,
        geometry_max_error: f64,
        geometry_feasible: bool,
        feasible: bool,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct OracleCaseReport {
        lambda: f64,
        pcg_max_iterations: usize,
        pose_blocks: usize,
        landmark_count: usize,
        observation_count: usize,
        schur_dimension: usize,
        singular_landmarks: usize,
        raw_schur_asymmetry: Option<f64>,
        schur_base_action_norm: Option<f64>,
        schur_eliminated_action_norm: Option<f64>,
        schur_arithmetic_scale: Option<f64>,
        operator_raw_action_error: Option<f64>,
        operator_raw_action_relative_error: Option<f64>,
        operator_lower_action_error: Option<f64>,
        operator_lower_action_relative_error: Option<f64>,
        operator_rhs_error: Option<f64>,
        direct_explicit_pose_error: Option<f64>,
        direct_explicit_landmark_error: Option<f64>,
        direct_feasibility: Option<Feasibility>,
        explicit_feasibility: Option<Feasibility>,
        direct_prediction: Option<PredictedDecrease>,
        explicit_prediction: Option<PredictedDecrease>,
        pcg_status: String,
        pcg_iterations: Option<usize>,
        pcg_true_residual: Option<f64>,
        pcg_target: Option<f64>,
        matrix_free_pose_error: Option<f64>,
        matrix_free_landmark_error: Option<f64>,
        matrix_free_feasibility: Option<Feasibility>,
        matrix_free_prediction: Option<PredictedDecrease>,
    }

    /// Result of the test-only PCG recurrence when the explicitly materialized
    /// lower-mirrored Schur matrix is used as the action.  A failed solve does
    /// not retain its last iterate: in particular, no failed trial is passed
    /// to the geometry or prediction checks below.
    #[derive(Debug, Clone, PartialEq)]
    struct ExplicitPcgIsolationReport {
        lambda: f64,
        pcg_max_iterations: usize,
        status: String,
        iterations: Option<usize>,
        recursive_residual: Option<f64>,
        lower_true_residual: Option<f64>,
        implicit_true_residual: Option<f64>,
        target: Option<f64>,
        dense_lower_true_residual: Option<f64>,
        dense_implicit_true_residual: Option<f64>,
        pose_error_vs_dense: Option<f64>,
        landmark_error_vs_dense: Option<f64>,
        implicit_pcg_status: String,
        implicit_pcg_iterations: Option<usize>,
        implicit_pcg_true_residual: Option<f64>,
        implicit_pcg_target: Option<f64>,
        explicit_vs_implicit_pose_error: Option<f64>,
        explicit_vs_implicit_landmark_error: Option<f64>,
        feasibility: Option<Feasibility>,
        prediction: Option<PredictedDecrease>,
    }

    /// Diagnostics collected before choosing the landmark-block factorization.
    /// The asymmetry and scale are deliberately reported separately: a large
    /// block scale can make a small absolute difference look alarming, while
    /// a small-looking difference can still matter after Schur cancellation.
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct LandmarkFactorMetrics {
        h_ll_max_asymmetry: f64,
        h_ll_max_scale: f64,
        inverse_max_asymmetry: Option<f64>,
        inverse_max_scale: Option<f64>,
        h_ll_inverse_identity_residual: Option<f64>,
    }

    impl Default for LandmarkFactorMetrics {
        fn default() -> Self {
            Self {
                h_ll_max_asymmetry: 0.0,
                h_ll_max_scale: 0.0,
                inverse_max_asymmetry: None,
                inverse_max_scale: None,
                h_ll_inverse_identity_residual: None,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct CholeskyPcgArmReport {
        status: String,
        iterations: Option<usize>,
        recursive_residual: Option<f64>,
        action_true_residual: Option<f64>,
        target: Option<f64>,
        lower_true_residual: Option<f64>,
        dense_reference_lower_true_residual: Option<f64>,
        dense_reference_action_true_residual: Option<f64>,
        dense_reference_original_pose_equation_residual: Option<f64>,
        original_pose_equation_residual: Option<f64>,
        pose_error_vs_dense: Option<f64>,
        landmark_error_vs_dense: Option<f64>,
        feasibility: Option<Feasibility>,
        prediction: Option<PredictedDecrease>,
    }

    #[derive(Debug, Clone, PartialEq)]
    struct CholeskyPcgIsolationReport {
        lambda: f64,
        pcg_max_iterations: usize,
        general_metrics: LandmarkFactorMetrics,
        cholesky_metrics: LandmarkFactorMetrics,
        general: CholeskyPcgArmReport,
        cholesky: CholeskyPcgArmReport,
        general_vs_cholesky_pose_error: Option<f64>,
        general_vs_cholesky_landmark_error: Option<f64>,
        rhs_error: Option<f64>,
        probe_action_error: Option<f64>,
        probe_preconditioner_error: Option<f64>,
    }

    fn parse_f64(token: &str, context: &str) -> Result<f64, String> {
        let value = token
            .parse::<f64>()
            .map_err(|error| format!("{context}: {error}"))?;
        if !value.is_finite() {
            return Err(format!("{context}: non-finite value"));
        }
        Ok(value)
    }

    fn parse_usize(token: &str, context: &str) -> Result<usize, String> {
        token
            .parse::<usize>()
            .map_err(|error| format!("{context}: {error}"))
    }

    fn parse_u64(token: &str, context: &str) -> Result<u64, String> {
        token
            .parse::<u64>()
            .map_err(|error| format!("{context}: {error}"))
    }

    fn parse_se3(fields: &[&str], start: usize, context: &str) -> Result<SE3, String> {
        if fields.len() < start + 7 {
            return Err(format!("{context}: expected quaternion and translation"));
        }
        let values = (0..7)
            .map(|index| parse_f64(fields[start + index], context))
            .collect::<Result<Vec<_>, _>>()?;
        let quaternion = nalgebra::Quaternion::new(values[0], values[1], values[2], values[3]);
        let norm = quaternion.norm();
        if !norm.is_finite() || norm <= 1.0e-12 || (norm - 1.0).abs() > 1.0e-6 {
            return Err(format!("{context}: quaternion norm is not one"));
        }
        Ok(SE3::new(
            // The exporter already validated and emitted a unit quaternion.
            // Preserve its f64 components exactly; normalizing here would
            // change the initial normal system by a rounding-dependent amount.
            // `from_quaternion` normalizes.  The fixture has already checked
            // the norm, so use the unchecked constructor to preserve every
            // serialized f64 bit and keep the normal system identical.
            nalgebra::UnitQuaternion::new_unchecked(quaternion),
            Vector3::new(values[4], values[5], values[6]),
        ))
    }

    fn parse_fixture(path: &Path) -> Result<OracleFixture, String> {
        let file = File::open(path).map_err(|error| format!("open fixture: {error}"))?;
        let reader = BufReader::new(file);
        let mut source_hashes = BTreeMap::new();
        let mut expected_initial_cost = None;
        let mut expected_initial_cost_bits = None;
        let mut expected_counts = BTreeMap::<String, usize>::new();
        let mut cameras = BTreeMap::<u64, Camera>::new();
        let mut poses = BTreeMap::<u64, Pose>::new();
        let mut landmarks = BTreeMap::<u64, Point3<f64>>::new();
        let mut fixed_poses = BTreeSet::new();
        let mut rig_observations = Vec::new();
        let mut saw_header = false;
        let mut saw_end = false;

        for (line_index, line_result) in reader.lines().enumerate() {
            let line_number = line_index + 1;
            let line =
                line_result.map_err(|error| format!("fixture line {line_number}: {error}"))?;
            let fields = line.split_whitespace().collect::<Vec<_>>();
            if fields.is_empty() {
                continue;
            }
            if saw_end && fields[0] != "END" {
                return Err(format!(
                    "fixture line {line_number}: records after END are not allowed"
                ));
            }
            if !saw_header && fields[0] != "VISLOC_BA_ORACLE_FIXTURE" {
                return Err(format!(
                    "fixture line {line_number}: header must be the first record"
                ));
            }
            match fields[0] {
                "VISLOC_BA_ORACLE_FIXTURE" => {
                    if fields != ["VISLOC_BA_ORACLE_FIXTURE", "1"] {
                        return Err(format!("fixture line {line_number}: unsupported version"));
                    }
                    if saw_header {
                        return Err(format!("fixture line {line_number}: duplicate header"));
                    }
                    saw_header = true;
                }
                "SOURCE_SHA256"
                | "SOURCE_SHA256_CAMERAS"
                | "SOURCE_SHA256_IMAGES"
                | "SOURCE_SHA256_POINTS"
                | "SOURCE_SHA256_MANIFEST" => {
                    if fields.len() != 2
                        || fields[1].len() != 64
                        || !fields[1].bytes().all(|byte| byte.is_ascii_hexdigit())
                    {
                        return Err(format!("fixture line {line_number}: invalid source hash"));
                    }
                    if source_hashes
                        .insert(fields[0].to_owned(), fields[1].to_owned())
                        .is_some()
                    {
                        return Err(format!("fixture line {line_number}: duplicate source hash"));
                    }
                }
                "INITIAL_COST" => {
                    if fields.len() != 2 || expected_initial_cost.is_some() {
                        return Err(format!("fixture line {line_number}: invalid INITIAL_COST"));
                    }
                    expected_initial_cost = Some(parse_f64(fields[1], "initial cost")?);
                }
                "INITIAL_COST_BITS" => {
                    if fields.len() != 2 || expected_initial_cost_bits.is_some() {
                        return Err(format!(
                            "fixture line {line_number}: invalid INITIAL_COST_BITS"
                        ));
                    }
                    expected_initial_cost_bits = Some(parse_u64(fields[1], "initial cost bits")?);
                }
                "CAMERA_COUNT" | "POSE_COUNT" | "LANDMARK_COUNT" | "OBSERVATION_COUNT" => {
                    if fields.len() != 2 {
                        return Err(format!("fixture line {line_number}: invalid count"));
                    }
                    let count = parse_usize(fields[1], "fixture count")?;
                    let cap = match fields[0] {
                        "CAMERA_COUNT" => MAX_CAMERAS,
                        "POSE_COUNT" => MAX_VARIABLE_POSES + 1,
                        "LANDMARK_COUNT" => MAX_LANDMARKS,
                        "OBSERVATION_COUNT" => MAX_OBSERVATIONS,
                        _ => unreachable!(),
                    };
                    if count > cap {
                        return Err(format!(
                            "fixture line {line_number}: {} exceeds cap {cap}",
                            fields[0]
                        ));
                    }
                    if expected_counts
                        .insert(fields[0].to_owned(), count)
                        .is_some()
                    {
                        return Err(format!("fixture line {line_number}: duplicate count"));
                    }
                }
                "CAMERA" => {
                    if fields.len() < 6 {
                        return Err(format!("fixture line {line_number}: short camera"));
                    }
                    if cameras.len() >= MAX_CAMERAS {
                        return Err(format!("fixture line {line_number}: camera cap exceeded"));
                    }
                    let id = parse_u64(fields[1], "camera id")?;
                    let model = CameraModel::from_colmap_name(fields[2]);
                    if matches!(model, CameraModel::Unknown(_)) {
                        return Err(format!("fixture line {line_number}: unknown camera model"));
                    }
                    let width = fields[3]
                        .parse::<u32>()
                        .map_err(|error| format!("fixture line {line_number}: width: {error}"))?;
                    let height = fields[4]
                        .parse::<u32>()
                        .map_err(|error| format!("fixture line {line_number}: height: {error}"))?;
                    let parameter_count = parse_usize(fields[5], "camera parameter count")?;
                    if parameter_count > 16 {
                        return Err(format!(
                            "fixture line {line_number}: camera parameter cap exceeded"
                        ));
                    }
                    if fields.len() != 6 + parameter_count {
                        return Err(format!(
                            "fixture line {line_number}: camera parameter count mismatch"
                        ));
                    }
                    let params = fields[6..]
                        .iter()
                        .map(|token| parse_f64(token, "camera parameter"))
                        .collect::<Result<Vec<_>, _>>()?;
                    if cameras
                        .insert(
                            id,
                            Camera {
                                id,
                                model,
                                width,
                                height,
                                params,
                            },
                        )
                        .is_some()
                    {
                        return Err(format!("fixture line {line_number}: duplicate camera"));
                    }
                }
                "POSE" => {
                    if fields.len() != 9 {
                        return Err(format!("fixture line {line_number}: invalid pose"));
                    }
                    if poses.len() > MAX_VARIABLE_POSES {
                        return Err(format!("fixture line {line_number}: pose cap exceeded"));
                    }
                    let id = parse_u64(fields[1], "pose id")?;
                    let pose = Pose {
                        world_to_camera: parse_se3(&fields, 2, "pose")?,
                    };
                    if poses.insert(id, pose).is_some() {
                        return Err(format!("fixture line {line_number}: duplicate pose"));
                    }
                }
                "FIXED_POSE" => {
                    if fields.len() != 2 {
                        return Err(format!("fixture line {line_number}: invalid fixed pose"));
                    }
                    if !fixed_poses.insert(parse_u64(fields[1], "fixed pose id")?) {
                        return Err(format!("fixture line {line_number}: duplicate fixed pose"));
                    }
                }
                "LANDMARK" => {
                    if fields.len() != 5 {
                        return Err(format!("fixture line {line_number}: invalid landmark"));
                    }
                    if landmarks.len() >= MAX_LANDMARKS {
                        return Err(format!("fixture line {line_number}: landmark cap exceeded"));
                    }
                    let id = parse_u64(fields[1], "landmark id")?;
                    let point = Point3::new(
                        parse_f64(fields[2], "landmark x")?,
                        parse_f64(fields[3], "landmark y")?,
                        parse_f64(fields[4], "landmark z")?,
                    );
                    if landmarks.insert(id, point).is_some() {
                        return Err(format!("fixture line {line_number}: duplicate landmark"));
                    }
                }
                "RIG_OBSERVATION" => {
                    if fields.len() != 13 {
                        return Err(format!(
                            "fixture line {line_number}: invalid rig observation"
                        ));
                    }
                    if rig_observations.len() >= MAX_OBSERVATIONS {
                        return Err(format!(
                            "fixture line {line_number}: observation cap exceeded"
                        ));
                    }
                    let frame_id = parse_u64(fields[1], "observation frame id")?;
                    let landmark_id = parse_u64(fields[2], "observation landmark id")?;
                    let xy = Point2::new(
                        parse_f64(fields[3], "observation x")?,
                        parse_f64(fields[4], "observation y")?,
                    );
                    let camera_id = parse_u64(fields[5], "observation camera id")?;
                    let sensor_from_rig = parse_se3(&fields, 6, "sensor extrinsic")?;
                    rig_observations.push((frame_id, landmark_id, xy, camera_id, sensor_from_rig));
                }
                "END" => {
                    if fields.len() != 1 || saw_end {
                        return Err(format!("fixture line {line_number}: invalid END"));
                    }
                    saw_end = true;
                }
                other => {
                    return Err(format!(
                        "fixture line {line_number}: unknown record {other:?}"
                    ));
                }
            }
        }
        if !saw_header || !saw_end {
            return Err("fixture is missing header or END".to_owned());
        }
        let expected_initial_cost =
            expected_initial_cost.ok_or_else(|| "fixture is missing INITIAL_COST".to_owned())?;
        let expected_initial_cost_bits = expected_initial_cost_bits
            .ok_or_else(|| "fixture is missing INITIAL_COST_BITS".to_owned())?;
        for key in [
            "SOURCE_SHA256",
            "SOURCE_SHA256_CAMERAS",
            "SOURCE_SHA256_IMAGES",
            "SOURCE_SHA256_POINTS",
            "SOURCE_SHA256_MANIFEST",
        ] {
            if !source_hashes.contains_key(key) {
                return Err(format!("fixture is missing {key}"));
            }
        }
        if cameras.is_empty()
            || poses.is_empty()
            || landmarks.is_empty()
            || rig_observations.is_empty()
        {
            return Err("fixture has no BA records".to_owned());
        }
        if fixed_poses.len() != 1 || !fixed_poses.iter().all(|id| poses.contains_key(id)) {
            return Err("fixture fixed pose set is invalid".to_owned());
        }
        let first_camera = cameras
            .values()
            .next()
            .cloned()
            .ok_or_else(|| "fixture has no camera".to_owned())?;
        let mut ba = BundleAdjustment::new(first_camera);
        for (id, pose) in poses {
            ba.add_pose(id, pose);
        }
        for id in fixed_poses {
            ba.fix_pose(id);
        }
        for (id, point) in landmarks {
            ba.add_landmark(id, point);
        }
        for (frame_id, landmark_id, xy, camera_id, sensor_from_rig) in rig_observations {
            let camera = cameras
                .get(&camera_id)
                .cloned()
                .ok_or_else(|| format!("observation references unknown camera {camera_id}"))?;
            if !ba.poses.contains_key(&frame_id) {
                return Err(format!("observation references unknown pose {frame_id}"));
            }
            if !ba.landmarks.contains_key(&landmark_id) {
                return Err(format!(
                    "observation references unknown landmark {landmark_id}"
                ));
            }
            ba.add_rig_observation(BaRigObservation {
                keyframe_id: frame_id,
                landmark_id,
                xy,
                camera,
                sensor_from_rig,
            });
        }
        let expected = [
            ("CAMERA_COUNT", cameras.len()),
            ("POSE_COUNT", ba.poses.len()),
            ("LANDMARK_COUNT", ba.landmarks.len()),
            ("OBSERVATION_COUNT", ba.rig_observations.len()),
        ];
        for (key, actual) in expected {
            if expected_counts.get(key).copied() != Some(actual) {
                return Err(format!("fixture {key} does not match records"));
            }
        }
        let actual_initial_cost = ba.cost();
        if !actual_initial_cost.is_finite()
            || actual_initial_cost.to_bits() != expected_initial_cost_bits
            || actual_initial_cost.to_bits() != expected_initial_cost.to_bits()
        {
            return Err(format!(
                "fixture initial cost bit mismatch: expected {} ({expected_initial_cost:.17e}), actual {} ({actual_initial_cost:.17e})",
                expected_initial_cost_bits,
                actual_initial_cost.to_bits(),
            ));
        }
        Ok(OracleFixture {
            source_hashes,
            initial_cost: expected_initial_cost,
            initial_cost_bits: expected_initial_cost_bits,
            ba,
        })
    }

    fn variable_indices(ba: &BundleAdjustment) -> (BTreeMap<u64, usize>, BTreeMap<u64, usize>) {
        let pose_index = ba
            .poses
            .keys()
            .copied()
            .filter(|id| !ba.fixed_poses.contains(id))
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect();
        let landmark_index = ba
            .landmarks
            .keys()
            .copied()
            .filter(|id| !ba.fixed_landmarks.contains(id))
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect();
        (pose_index, landmark_index)
    }

    fn observation_count(ba: &BundleAdjustment) -> usize {
        ba.observations.len()
            + ba.stereo_observations.len()
            + ba.general_stereo_observations.len()
            + ba.rig_observations.len()
    }

    fn cross_pair_work(system: &NormalEquationsBa) -> Result<usize, String> {
        system
            .landmarks
            .iter()
            .try_fold(0_usize, |total, landmark| {
                let pairs = landmark
                    .cross
                    .len()
                    .checked_mul(landmark.cross.len())
                    .ok_or_else(|| "cross-pair work overflows".to_owned())?;
                total
                    .checked_add(pairs)
                    .ok_or_else(|| "cross-pair work overflows".to_owned())
            })
    }

    fn check_caps(
        pose_blocks: usize,
        landmarks: usize,
        observations: usize,
        cross_pairs: usize,
    ) -> Result<usize, String> {
        if pose_blocks > MAX_VARIABLE_POSES {
            return Err(format!("oracle variable-pose cap exceeded: {pose_blocks}"));
        }
        if landmarks > MAX_LANDMARKS {
            return Err(format!("oracle landmark cap exceeded: {landmarks}"));
        }
        if observations > MAX_OBSERVATIONS {
            return Err(format!("oracle observation cap exceeded: {observations}"));
        }
        if cross_pairs > MAX_CROSS_PAIR_WORK {
            return Err(format!("oracle cross-pair cap exceeded: {cross_pairs}"));
        }
        let dimension = pose_blocks
            .checked_mul(6)
            .ok_or_else(|| "oracle Schur dimension overflows".to_owned())?;
        if dimension > MAX_SCHUR_SCALARS {
            return Err(format!("oracle Schur dimension cap exceeded: {dimension}"));
        }
        Ok(dimension)
    }

    fn build_oracle_system(
        ba: &BundleAdjustment,
    ) -> Result<(NormalEquationsBa, usize, usize, usize), String> {
        if ba.observations.is_empty()
            && ba.stereo_observations.is_empty()
            && ba.general_stereo_observations.is_empty()
            && ba.rig_observations.is_empty()
        {
            return Err("oracle requires visual observations".to_owned());
        }
        if ba.velocities.is_empty()
            && ba.biases.is_empty()
            && ba.imu_factors.is_empty()
            && ba.bias_random_walk_factors.is_empty()
            && ba.gravity_prior.is_none()
            && ba.per_pose_gravity_prior.is_none()
            && ba.position_prior.is_none()
            && ba.pairwise_pose_factors.is_empty()
            && ba.navigation_state_prior.is_none()
        {
            // Pure-visual eligibility is intentionally explicit.  The empty
            // branch is only a guard; all actual work follows below.
        } else {
            return Err("oracle only accepts pure visual input".to_owned());
        }
        let se3_is_finite = |transform: &SE3| {
            transform
                .rotation
                .quaternion()
                .coords
                .iter()
                .all(|value| value.is_finite())
                && transform.translation.iter().all(|value| value.is_finite())
        };
        if !ba
            .poses
            .values()
            .all(|pose| se3_is_finite(&pose.world_to_camera))
            || !ba
                .landmarks
                .values()
                .all(|point| point.coords.iter().all(|v| v.is_finite()))
        {
            return Err("oracle input pose or landmark is non-finite".to_owned());
        }
        for observation in &ba.rig_observations {
            if !matches!(
                observation.camera.model,
                CameraModel::Pinhole | CameraModel::SimplePinhole
            ) {
                return Err("oracle requires distortion-free pinhole cameras".to_owned());
            }
            if observation
                .camera
                .radial_distortion()
                .is_some_and(|(k1, k2)| k1 != 0.0 || k2 != 0.0)
            {
                return Err("oracle rejects nonzero camera distortion".to_owned());
            }
            let Some(intrinsics) = observation.camera.intrinsics() else {
                return Err("oracle camera has no pinhole intrinsics".to_owned());
            };
            if ![intrinsics.0, intrinsics.1, intrinsics.2, intrinsics.3]
                .iter()
                .all(|value| value.is_finite())
            {
                return Err("oracle camera intrinsics are non-finite".to_owned());
            }
            if !se3_is_finite(&observation.sensor_from_rig) {
                return Err("oracle sensor extrinsic is non-finite".to_owned());
            }
        }
        let intrinsics = ba
            .camera
            .intrinsics()
            .ok_or_else(|| "oracle camera has no pinhole intrinsics".to_owned())?;
        let (pose_index, landmark_index) = variable_indices(ba);
        if pose_index.is_empty() {
            return Err("oracle requires at least one variable pose".to_owned());
        }
        let mut system = build_normal_equations(
            ba,
            &intrinsics,
            &pose_index,
            &landmark_index,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &RobustKernel::None,
            None,
            false,
            true,
        );
        constrain_fixed_pose_rotations(&ba.fixed_pose_rotations, &pose_index, &mut system);
        let observations = observation_count(ba);
        let pairs = cross_pair_work(&system)?;
        check_caps(pose_index.len(), landmark_index.len(), observations, pairs)?;
        Ok((system, pose_index.len(), landmark_index.len(), observations))
    }

    fn explicit_schur_rhs(
        system: &NormalEquationsBa,
        lambda: f64,
    ) -> Result<(DMatrix<f64>, DVector<f64>, usize), String> {
        let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
            return Err("oracle requires pose-diagonal system".to_owned());
        };
        let dimension = diagonal.len() * 6;
        let mut schur = DMatrix::zeros(dimension, dimension);
        for (pose, block) in diagonal.iter().enumerate() {
            let mut damped = *block;
            for component in 0..6 {
                damped[(component, component)] += lambda;
            }
            schur
                .fixed_view_mut::<6, 6>(pose * 6, pose * 6)
                .copy_from(&damped);
        }
        let mut rhs = -&system.b_p;
        let mut singular_landmarks = 0;
        for landmark in &system.landmarks {
            let mut h_ll = landmark.h_ll;
            for component in 0..3 {
                h_ll[(component, component)] += lambda;
            }
            let Some(inverse) = h_ll.try_inverse() else {
                singular_landmarks += 1;
                continue;
            };
            for (pose, cross) in &landmark.cross {
                let update: Vector6<f64> = cross * inverse * landmark.b_l;
                for component in 0..6 {
                    rhs[pose * 6 + component] += update[component];
                }
                for (other_pose, other_cross) in &landmark.cross {
                    let block: Matrix6<f64> = cross * inverse * other_cross.transpose();
                    for row in 0..6 {
                        for column in 0..6 {
                            schur[(pose * 6 + row, other_pose * 6 + column)] -=
                                block[(row, column)];
                        }
                    }
                }
            }
        }
        if !schur.iter().all(|value| value.is_finite())
            || !rhs.iter().all(|value| value.is_finite())
        {
            return Err("oracle explicit Schur is non-finite".to_owned());
        }
        Ok((schur, rhs, singular_landmarks))
    }

    type LandmarkCholesky = nalgebra::Cholesky<f64, nalgebra::Const<3>>;

    const CHOLESKY_HLL_SYMMETRY_RELATIVE_TOLERANCE: f64 = 1.0e-12;

    fn damped_landmark_hessian(
        landmark: &LandmarkBlock,
        lambda: f64,
    ) -> Result<Matrix3<f64>, String> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(format!("invalid landmark damping {lambda}"));
        }
        let mut h_ll = landmark.h_ll;
        for component in 0..3 {
            h_ll[(component, component)] += lambda;
        }
        if !h_ll.iter().all(|value| value.is_finite()) {
            return Err("damped landmark Hessian is non-finite".to_owned());
        }
        Ok(h_ll)
    }

    fn matrix3_max_abs(matrix: &Matrix3<f64>) -> f64 {
        matrix
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f64, f64::max)
    }

    fn matrix3_max_asymmetry(matrix: &Matrix3<f64>) -> f64 {
        let mut max_difference = 0.0_f64;
        for row in 0..3 {
            for column in (row + 1)..3 {
                max_difference =
                    max_difference.max((matrix[(row, column)] - matrix[(column, row)]).abs());
            }
        }
        max_difference
    }

    fn update_landmark_factor_metrics(
        metrics: &mut LandmarkFactorMetrics,
        h_ll: &Matrix3<f64>,
        inverse: Option<&Matrix3<f64>>,
    ) -> Result<(), String> {
        let h_ll_asymmetry = matrix3_max_asymmetry(h_ll);
        let h_ll_scale = matrix3_max_abs(h_ll);
        if !h_ll_asymmetry.is_finite() || !h_ll_scale.is_finite() {
            return Err("landmark Hessian metrics are non-finite".to_owned());
        }
        metrics.h_ll_max_asymmetry = metrics.h_ll_max_asymmetry.max(h_ll_asymmetry);
        metrics.h_ll_max_scale = metrics.h_ll_max_scale.max(h_ll_scale);
        let Some(inverse) = inverse else {
            return Ok(());
        };
        let inverse_asymmetry = matrix3_max_asymmetry(inverse);
        let inverse_scale = matrix3_max_abs(inverse);
        let identity_residual = (h_ll * inverse - Matrix3::identity()).norm();
        if !inverse_asymmetry.is_finite()
            || !inverse_scale.is_finite()
            || !identity_residual.is_finite()
        {
            return Err("landmark inverse metrics are non-finite".to_owned());
        }
        metrics.inverse_max_asymmetry = Some(
            metrics
                .inverse_max_asymmetry
                .unwrap_or(0.0)
                .max(inverse_asymmetry),
        );
        metrics.inverse_max_scale =
            Some(metrics.inverse_max_scale.unwrap_or(0.0).max(inverse_scale));
        metrics.h_ll_inverse_identity_residual = Some(
            metrics
                .h_ll_inverse_identity_residual
                .unwrap_or(0.0)
                .max(identity_residual),
        );
        Ok(())
    }

    fn general_landmark_factor_metrics(
        system: &NormalEquationsBa,
        lambda: f64,
    ) -> Result<LandmarkFactorMetrics, String> {
        let mut metrics = LandmarkFactorMetrics::default();
        for landmark in &system.landmarks {
            let h_ll = damped_landmark_hessian(landmark, lambda)?;
            let inverse = h_ll.try_inverse();
            if let Some(inverse) = inverse {
                if !inverse.iter().all(|value| value.is_finite()) {
                    return Err("general landmark inverse is non-finite".to_owned());
                }
                update_landmark_factor_metrics(&mut metrics, &h_ll, Some(&inverse))?;
            } else {
                update_landmark_factor_metrics(&mut metrics, &h_ll, None)?;
            }
        }
        Ok(metrics)
    }

    /// Cholesky consumes the lower triangle, but only after this numerical
    /// symmetry audit.  The upper triangle is then replaced from the lower
    /// triangle so every Cholesky operation (RHS, Schur action, preconditioner
    /// and back-substitution) sees the same explicitly audited matrix.
    fn lower_symmetric_landmark_hessian(h_ll: &Matrix3<f64>) -> Result<Matrix3<f64>, String> {
        let asymmetry = matrix3_max_asymmetry(h_ll);
        let scale = matrix3_max_abs(h_ll).max(1.0);
        if asymmetry > CHOLESKY_HLL_SYMMETRY_RELATIVE_TOLERANCE * scale {
            return Err(format!(
                "landmark Hessian is not symmetric: asymmetry={asymmetry:.17e} scale={scale:.17e}"
            ));
        }
        let mut symmetric = *h_ll;
        for row in 0..3 {
            for column in (row + 1)..3 {
                symmetric[(row, column)] = symmetric[(column, row)];
            }
        }
        Ok(symmetric)
    }

    #[derive(Debug, Clone)]
    struct CholeskyBuildFailure {
        metrics: LandmarkFactorMetrics,
        reason: String,
    }

    /// Test-only Schur operator whose landmark elimination uses a validated
    /// lower-triangle Cholesky factor and triangular solves.  The production
    /// `ImplicitSchurOperator` remains untouched; this operator exists only to
    /// compare the complete elimination path under the same PCG recurrence.
    struct CholeskySchurOperator<'a> {
        diagonal: Vec<Matrix6<f64>>,
        landmarks: &'a [LandmarkBlock],
        factors: Vec<LandmarkCholesky>,
        preconditioner_inverse: Vec<Matrix6<f64>>,
        rhs: DVector<f64>,
    }

    impl<'a> CholeskySchurOperator<'a> {
        fn new(
            system: &'a NormalEquationsBa,
            lambda: f64,
        ) -> Result<(Self, LandmarkFactorMetrics), CholeskyBuildFailure> {
            let mut metrics = LandmarkFactorMetrics::default();
            let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: "oracle requires pose-diagonal system".to_owned(),
                });
            };
            if diagonal.is_empty() {
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: "oracle has no variable pose blocks".to_owned(),
                });
            }
            let dimension = diagonal.len() * 6;
            if system.b_p.len() != dimension {
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: format!(
                        "pose RHS dimension mismatch: expected {dimension}, actual {}",
                        system.b_p.len()
                    ),
                });
            }
            if !system.b_p.iter().all(|value| value.is_finite()) {
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: "pose RHS is non-finite".to_owned(),
                });
            }

            let mut damped_diagonal = diagonal.clone();
            for block in &mut damped_diagonal {
                for component in 0..6 {
                    block[(component, component)] += lambda;
                }
            }
            if !damped_diagonal
                .iter()
                .flat_map(|block| block.iter())
                .all(|value| value.is_finite())
            {
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: "damped pose Hessian is non-finite".to_owned(),
                });
            }

            let mut factors = Vec::with_capacity(system.landmarks.len());
            for (landmark_index, landmark) in system.landmarks.iter().enumerate() {
                let h_ll = match damped_landmark_hessian(landmark, lambda) {
                    Ok(h_ll) => h_ll,
                    Err(reason) => {
                        return Err(CholeskyBuildFailure { metrics, reason });
                    }
                };
                for (pose, cross) in &landmark.cross {
                    if *pose >= diagonal.len() {
                        return Err(CholeskyBuildFailure {
                            metrics,
                            reason: format!(
                                "landmark {landmark_index} references pose {pose} outside {} blocks",
                                diagonal.len()
                            ),
                        });
                    }
                    if !cross.iter().all(|value| value.is_finite()) {
                        return Err(CholeskyBuildFailure {
                            metrics,
                            reason: format!("landmark {landmark_index} cross is non-finite"),
                        });
                    }
                }
                if !landmark.b_l.iter().all(|value| value.is_finite()) {
                    return Err(CholeskyBuildFailure {
                        metrics,
                        reason: format!("landmark {landmark_index} RHS is non-finite"),
                    });
                }
                let symmetric_h_ll = match lower_symmetric_landmark_hessian(&h_ll) {
                    Ok(h_ll) => h_ll,
                    Err(reason) => {
                        if let Err(metric_reason) =
                            update_landmark_factor_metrics(&mut metrics, &h_ll, None)
                        {
                            return Err(CholeskyBuildFailure {
                                metrics,
                                reason: metric_reason,
                            });
                        }
                        return Err(CholeskyBuildFailure { metrics, reason });
                    }
                };
                let Some(cholesky) = symmetric_h_ll.cholesky() else {
                    if let Err(metric_reason) =
                        update_landmark_factor_metrics(&mut metrics, &h_ll, None)
                    {
                        return Err(CholeskyBuildFailure {
                            metrics,
                            reason: metric_reason,
                        });
                    }
                    return Err(CholeskyBuildFailure {
                        metrics,
                        reason: format!("landmark {landmark_index} damped Hessian is not SPD"),
                    });
                };
                let inverse = cholesky.inverse();
                if !inverse.iter().all(|value| value.is_finite()) {
                    if let Err(metric_reason) =
                        update_landmark_factor_metrics(&mut metrics, &h_ll, None)
                    {
                        return Err(CholeskyBuildFailure {
                            metrics,
                            reason: metric_reason,
                        });
                    }
                    return Err(CholeskyBuildFailure {
                        metrics,
                        reason: format!("landmark {landmark_index} Cholesky inverse is non-finite"),
                    });
                }
                if let Err(metric_reason) =
                    update_landmark_factor_metrics(&mut metrics, &h_ll, Some(&inverse))
                {
                    return Err(CholeskyBuildFailure {
                        metrics,
                        reason: metric_reason,
                    });
                }
                factors.push(cholesky);
            }

            let mut rhs = -&system.b_p;
            for (landmark, factor) in system.landmarks.iter().zip(&factors) {
                let solved = factor.solve(&landmark.b_l);
                if !solved.iter().all(|value| value.is_finite()) {
                    return Err(CholeskyBuildFailure {
                        metrics,
                        reason: "Cholesky RHS solve is non-finite".to_owned(),
                    });
                }
                for (pose, cross) in &landmark.cross {
                    let update: Vector6<f64> = cross * solved;
                    for component in 0..6 {
                        rhs[pose * 6 + component] += update[component];
                    }
                }
            }
            if !rhs.iter().all(|value| value.is_finite()) {
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: "Cholesky reduced RHS is non-finite".to_owned(),
                });
            }

            let mut preconditioner = damped_diagonal.clone();
            for (landmark, factor) in system.landmarks.iter().zip(&factors) {
                let mut grouped: BTreeMap<usize, Matrix6x3<f64>> = BTreeMap::new();
                for (pose, cross) in &landmark.cross {
                    *grouped.entry(*pose).or_insert_with(Matrix6x3::zeros) += cross;
                }
                for (pose, cross) in grouped {
                    let cross_transpose = cross.transpose();
                    let solved = factor.solve(&cross_transpose);
                    if !solved.iter().all(|value| value.is_finite()) {
                        return Err(CholeskyBuildFailure {
                            metrics,
                            reason: "Cholesky preconditioner solve is non-finite".to_owned(),
                        });
                    }
                    preconditioner[pose] -= cross * solved;
                }
            }
            if !preconditioner
                .iter()
                .flat_map(|block| block.iter())
                .all(|value| value.is_finite())
            {
                return Err(CholeskyBuildFailure {
                    metrics,
                    reason: "Cholesky Schur preconditioner is non-finite".to_owned(),
                });
            }
            let mut preconditioner_inverse = Vec::with_capacity(preconditioner.len());
            for (pose, block) in preconditioner.iter().enumerate() {
                let Some(cholesky) = block.cholesky() else {
                    return Err(CholeskyBuildFailure {
                        metrics,
                        reason: format!("Cholesky Schur preconditioner block {pose} is not SPD"),
                    });
                };
                let inverse = cholesky.inverse();
                if !inverse.iter().all(|value| value.is_finite()) {
                    return Err(CholeskyBuildFailure {
                        metrics,
                        reason: format!(
                            "Cholesky Schur preconditioner inverse {pose} is non-finite"
                        ),
                    });
                }
                preconditioner_inverse.push(inverse);
            }

            Ok((
                Self {
                    diagonal: damped_diagonal,
                    landmarks: &system.landmarks,
                    factors,
                    preconditioner_inverse,
                    rhs,
                },
                metrics,
            ))
        }

        fn dimension(&self) -> usize {
            self.diagonal.len() * 6
        }

        fn rhs(&self) -> &DVector<f64> {
            &self.rhs
        }

        fn apply(
            &self,
            x: &DVector<f64>,
        ) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError> {
            if x.len() != self.dimension() {
                return Err(implicit_schur::ImplicitSchurError::DimensionMismatch {
                    expected: self.dimension(),
                    actual: x.len(),
                });
            }
            if !x.iter().all(|value| value.is_finite()) {
                return Err(implicit_schur::ImplicitSchurError::NonFinite(
                    "Cholesky operator input",
                ));
            }
            let mut out = DVector::zeros(self.dimension());
            for (pose, block) in self.diagonal.iter().enumerate() {
                let x_pose: Vector6<f64> = x.fixed_rows::<6>(pose * 6).into_owned();
                let value = block * x_pose;
                for component in 0..6 {
                    out[pose * 6 + component] = value[component];
                }
            }
            for (landmark, factor) in self.landmarks.iter().zip(&self.factors) {
                let mut projected = Vector3::zeros();
                for (pose, cross) in &landmark.cross {
                    let x_pose: Vector6<f64> = x.fixed_rows::<6>(pose * 6).into_owned();
                    projected += cross.transpose() * x_pose;
                }
                let reduced = factor.solve(&projected);
                if !reduced.iter().all(|value| value.is_finite()) {
                    return Err(implicit_schur::ImplicitSchurError::NonFinite(
                        "Cholesky operator landmark solve",
                    ));
                }
                for (pose, cross) in &landmark.cross {
                    let value: Vector6<f64> = cross * reduced;
                    for component in 0..6 {
                        out[pose * 6 + component] -= value[component];
                    }
                }
            }
            if !out.iter().all(|value| value.is_finite()) {
                return Err(implicit_schur::ImplicitSchurError::NonFinite(
                    "Cholesky operator output",
                ));
            }
            Ok(out)
        }

        fn apply_preconditioner(
            &self,
            residual: &DVector<f64>,
        ) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError> {
            if residual.len() != self.dimension() {
                return Err(implicit_schur::ImplicitSchurError::DimensionMismatch {
                    expected: self.dimension(),
                    actual: residual.len(),
                });
            }
            if !residual.iter().all(|value| value.is_finite()) {
                return Err(implicit_schur::ImplicitSchurError::NonFinite(
                    "Cholesky preconditioner input",
                ));
            }
            let mut out = DVector::zeros(self.dimension());
            for (pose, inverse) in self.preconditioner_inverse.iter().enumerate() {
                let residual_pose: Vector6<f64> = residual.fixed_rows::<6>(pose * 6).into_owned();
                let value = inverse * residual_pose;
                for component in 0..6 {
                    out[pose * 6 + component] = value[component];
                }
            }
            if !out.iter().all(|value| value.is_finite()) {
                return Err(implicit_schur::ImplicitSchurError::NonFinite(
                    "Cholesky preconditioner output",
                ));
            }
            Ok(out)
        }

        fn complete_delta(
            &self,
            delta_pose: &DVector<f64>,
        ) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError> {
            if delta_pose.len() != self.dimension() {
                return Err(implicit_schur::ImplicitSchurError::DimensionMismatch {
                    expected: self.dimension(),
                    actual: delta_pose.len(),
                });
            }
            if !delta_pose.iter().all(|value| value.is_finite()) {
                return Err(implicit_schur::ImplicitSchurError::NonFinite(
                    "Cholesky pose delta",
                ));
            }
            let mut delta_landmarks = DVector::zeros(self.landmarks.len() * 3);
            for (index, (landmark, factor)) in self.landmarks.iter().zip(&self.factors).enumerate()
            {
                let mut accumulated = -landmark.b_l;
                for (pose, cross) in &landmark.cross {
                    let delta: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
                    accumulated -= cross.transpose() * delta;
                }
                let value = factor.solve(&accumulated);
                if !value.iter().all(|entry| entry.is_finite()) {
                    return Err(implicit_schur::ImplicitSchurError::NonFinite(
                        "Cholesky landmark delta",
                    ));
                }
                for component in 0..3 {
                    delta_landmarks[index * 3 + component] = value[component];
                }
            }
            Ok(delta_landmarks)
        }
    }

    fn explicit_cholesky_schur_rhs(
        system: &NormalEquationsBa,
        operator: &CholeskySchurOperator<'_>,
    ) -> Result<(DMatrix<f64>, DVector<f64>), String> {
        let dimension = operator.dimension();
        let mut schur = DMatrix::zeros(dimension, dimension);
        for (pose, block) in operator.diagonal.iter().enumerate() {
            schur
                .fixed_view_mut::<6, 6>(pose * 6, pose * 6)
                .copy_from(block);
        }
        let rhs = operator.rhs.clone();
        for (landmark, factor) in system.landmarks.iter().zip(&operator.factors) {
            // Solve each 3x6 cross block once per landmark.  The nested
            // pose-pair accumulation below is O(B^2), but repeated triangular
            // solves would add an unnecessary O(B^2) factorization cost.
            let mut solved_crosses = Vec::with_capacity(landmark.cross.len());
            for (other_pose, other_cross) in &landmark.cross {
                let solved = factor.solve(&other_cross.transpose());
                if !solved.iter().all(|value| value.is_finite()) {
                    return Err("Cholesky explicit Schur block is non-finite".to_owned());
                }
                solved_crosses.push((*other_pose, solved));
            }
            for (pose, cross) in &landmark.cross {
                for (other_pose, solved) in &solved_crosses {
                    let block: Matrix6<f64> = cross * solved;
                    for row in 0..6 {
                        for column in 0..6 {
                            schur[(pose * 6 + row, other_pose * 6 + column)] -=
                                block[(row, column)];
                        }
                    }
                }
            }
        }
        if !schur.iter().all(|value| value.is_finite())
            || !rhs.iter().all(|value| value.is_finite())
        {
            return Err("Cholesky explicit Schur or RHS is non-finite".to_owned());
        }
        Ok((schur, rhs))
    }

    fn explicit_lower_action(
        schur: &DMatrix<f64>,
        x: &DVector<f64>,
    ) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError> {
        if schur.nrows() != schur.ncols() {
            return Err(implicit_schur::ImplicitSchurError::DimensionMismatch {
                expected: schur.nrows(),
                actual: schur.ncols(),
            });
        }
        if x.len() != schur.nrows() {
            return Err(implicit_schur::ImplicitSchurError::DimensionMismatch {
                expected: schur.nrows(),
                actual: x.len(),
            });
        }
        if !x.iter().all(|value| value.is_finite()) {
            return Err(implicit_schur::ImplicitSchurError::NonFinite(
                "operator input",
            ));
        }
        let output = schur * x;
        if !output.iter().all(|value| value.is_finite()) {
            return Err(implicit_schur::ImplicitSchurError::NonFinite(
                "operator output",
            ));
        }
        Ok(output)
    }

    fn checked_test_pcg_result<Apply>(
        rhs: &DVector<f64>,
        solution: DVector<f64>,
        iterations: usize,
        recursive_norm: f64,
        target: f64,
        apply: &mut Apply,
    ) -> Result<implicit_schur::PcgResult, implicit_schur::ImplicitSchurError>
    where
        Apply: FnMut(&DVector<f64>) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError>,
    {
        let applied = apply(&solution)?;
        let true_residual = rhs - &applied;
        let true_norm = true_residual.norm();
        if !true_norm.is_finite() {
            return Err(implicit_schur::ImplicitSchurError::NonFinite(
                "true PCG residual",
            ));
        }
        if true_norm > target {
            return Err(implicit_schur::ImplicitSchurError::ResidualCheckFailed {
                iterations,
                recursive_norm,
                true_norm,
                target,
            });
        }
        Ok(implicit_schur::PcgResult {
            solution,
            iterations,
            residual_norm: true_norm,
            target,
        })
    }

    /// Test-only generic PCG recurrence copied from the production implicit
    /// operator.  The callbacks make it possible to run the exact recurrence
    /// against either the implicit action or a borrowed explicit lower-Schur
    /// matrix, while keeping the production solver/API untouched.
    fn solve_test_pcg<Apply, Preconditioner>(
        rhs: &DVector<f64>,
        dimension: usize,
        options: implicit_schur::PcgOptions,
        mut apply: Apply,
        mut apply_preconditioner: Preconditioner,
    ) -> Result<implicit_schur::PcgResult, implicit_schur::ImplicitSchurError>
    where
        Apply: FnMut(&DVector<f64>) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError>,
        Preconditioner:
            FnMut(&DVector<f64>) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError>,
    {
        if rhs.len() != dimension {
            return Err(implicit_schur::ImplicitSchurError::DimensionMismatch {
                expected: dimension,
                actual: rhs.len(),
            });
        }
        if !rhs.iter().all(|value| value.is_finite()) {
            return Err(implicit_schur::ImplicitSchurError::NonFinite(
                "PCG right hand side",
            ));
        }
        if !options.relative_tolerance.is_finite()
            || options.relative_tolerance < 0.0
            || !options.absolute_tolerance.is_finite()
            || options.absolute_tolerance < 0.0
        {
            return Err(implicit_schur::ImplicitSchurError::InvalidTolerance);
        }
        let rhs_norm = rhs.norm();
        let target = options
            .absolute_tolerance
            .max(options.relative_tolerance * rhs_norm);
        if !target.is_finite() {
            return Err(implicit_schur::ImplicitSchurError::NonFinite("PCG target"));
        }
        let mut solution = DVector::zeros(dimension);
        let mut residual = rhs.clone();
        let mut residual_norm = residual.norm();
        if !residual_norm.is_finite() {
            return Err(implicit_schur::ImplicitSchurError::NonFinite(
                "initial residual",
            ));
        }
        if residual_norm <= target {
            return checked_test_pcg_result(rhs, solution, 0, residual_norm, target, &mut apply);
        }
        if options.max_iterations == 0 {
            return Err(implicit_schur::ImplicitSchurError::MaxIterations {
                iterations: 0,
                recursive_norm: residual_norm,
                residual_norm,
                target,
            });
        }

        let mut preconditioned = apply_preconditioner(&residual)?;
        let mut direction = preconditioned.clone();
        let mut rho = residual.dot(&preconditioned);
        if !rho.is_finite() || rho <= 0.0 {
            return Err(implicit_schur::ImplicitSchurError::NonPositiveCurvature);
        }

        for iteration in 1..=options.max_iterations {
            let applied = apply(&direction)?;
            let curvature = direction.dot(&applied);
            if !curvature.is_finite() || curvature <= 0.0 {
                return Err(implicit_schur::ImplicitSchurError::NonPositiveCurvature);
            }
            let alpha = rho / curvature;
            if !alpha.is_finite() {
                return Err(implicit_schur::ImplicitSchurError::NonFinite("PCG step"));
            }
            solution += alpha * &direction;
            residual -= alpha * applied;
            residual_norm = residual.norm();
            if !solution.iter().all(|value| value.is_finite()) || !residual_norm.is_finite() {
                return Err(implicit_schur::ImplicitSchurError::NonFinite("PCG iterate"));
            }
            if residual_norm <= target {
                return checked_test_pcg_result(
                    rhs,
                    solution,
                    iteration,
                    residual_norm,
                    target,
                    &mut apply,
                );
            }
            if iteration == options.max_iterations {
                let true_residual = rhs - &apply(&solution)?;
                let true_norm = true_residual.norm();
                if !true_norm.is_finite() {
                    return Err(implicit_schur::ImplicitSchurError::NonFinite(
                        "true PCG residual",
                    ));
                }
                if true_norm <= target {
                    return Ok(implicit_schur::PcgResult {
                        solution,
                        iterations: iteration,
                        residual_norm: true_norm,
                        target,
                    });
                }
                return Err(implicit_schur::ImplicitSchurError::MaxIterations {
                    iterations: iteration,
                    recursive_norm: residual_norm,
                    residual_norm: true_norm,
                    target,
                });
            }
            preconditioned = apply_preconditioner(&residual)?;
            let next_rho = residual.dot(&preconditioned);
            if !next_rho.is_finite() || next_rho <= 0.0 {
                return Err(implicit_schur::ImplicitSchurError::NonPositiveCurvature);
            }
            let beta = next_rho / rho;
            if !beta.is_finite() {
                return Err(implicit_schur::ImplicitSchurError::NonFinite(
                    "PCG direction",
                ));
            }
            direction = &preconditioned + beta * direction;
            rho = next_rho;
        }
        unreachable!("the max-iteration branch returns above");
    }

    struct CholeskyPcgArmOutcome {
        report: CholeskyPcgArmReport,
        solution: Option<DVector<f64>>,
        landmark_delta: Option<DVector<f64>>,
    }

    #[allow(clippy::too_many_arguments)]
    fn run_cholesky_pcg_arm<Apply, Preconditioner, Complete>(
        rhs: &DVector<f64>,
        dimension: usize,
        options: implicit_schur::PcgOptions,
        lower_schur: &DMatrix<f64>,
        dense_solution: Option<&DVector<f64>>,
        dense_landmark_delta: Option<&DVector<f64>>,
        dense_reference_action_true_residual: Option<f64>,
        dense_reference_original_pose_equation_residual: Option<f64>,
        apply: Apply,
        apply_preconditioner: Preconditioner,
        mut complete_delta: Complete,
        success_label: &str,
    ) -> CholeskyPcgArmOutcome
    where
        Apply: FnMut(&DVector<f64>) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError>,
        Preconditioner:
            FnMut(&DVector<f64>) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError>,
        Complete: FnMut(&DVector<f64>) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError>,
    {
        match solve_test_pcg(rhs, dimension, options, apply, apply_preconditioner) {
            Ok(result) => {
                let lower_true_residual = explicit_lower_action(lower_schur, &result.solution)
                    .ok()
                    .map(|applied| (rhs - applied).norm());
                let landmark_delta = complete_delta(&result.solution).ok();
                let dense_reference_lower_true_residual = dense_solution.and_then(|solution| {
                    explicit_lower_action(lower_schur, solution)
                        .ok()
                        .map(|applied| (rhs - applied).norm())
                });
                let pose_error_vs_dense =
                    dense_solution.map(|solution| (&result.solution - solution).norm());
                let landmark_error_vs_dense = match (landmark_delta.as_ref(), dense_landmark_delta)
                {
                    (Some(actual), Some(expected)) => Some((actual - expected).norm()),
                    _ => None,
                };
                CholeskyPcgArmOutcome {
                    report: CholeskyPcgArmReport {
                        status: success_label.to_owned(),
                        iterations: Some(result.iterations),
                        recursive_residual: None,
                        action_true_residual: Some(result.residual_norm),
                        target: Some(result.target),
                        lower_true_residual,
                        dense_reference_lower_true_residual,
                        dense_reference_action_true_residual,
                        dense_reference_original_pose_equation_residual,
                        original_pose_equation_residual: None,
                        pose_error_vs_dense,
                        landmark_error_vs_dense,
                        feasibility: None,
                        prediction: None,
                    },
                    solution: Some(result.solution),
                    landmark_delta,
                }
            }
            Err(error) => {
                let (iterations, recursive_residual, action_true_residual, target) =
                    pcg_error_details(&error);
                CholeskyPcgArmOutcome {
                    report: CholeskyPcgArmReport {
                        status: format!("failure:{error:?}"),
                        iterations,
                        recursive_residual,
                        action_true_residual,
                        target,
                        lower_true_residual: None,
                        dense_reference_lower_true_residual: dense_solution.and_then(|solution| {
                            explicit_lower_action(lower_schur, solution)
                                .ok()
                                .map(|applied| (rhs - applied).norm())
                        }),
                        dense_reference_action_true_residual,
                        dense_reference_original_pose_equation_residual,
                        original_pose_equation_residual: None,
                        pose_error_vs_dense: None,
                        landmark_error_vs_dense: None,
                        feasibility: None,
                        prediction: None,
                    },
                    solution: None,
                    landmark_delta: None,
                }
            }
        }
    }

    fn failed_cholesky_pcg_arm(
        status: String,
        rhs: &DVector<f64>,
        lower_schur: Option<&DMatrix<f64>>,
        dense_solution: Option<&DVector<f64>>,
        dense_reference_action_true_residual: Option<f64>,
        dense_reference_original_pose_equation_residual: Option<f64>,
    ) -> CholeskyPcgArmOutcome {
        let dense_reference_lower_true_residual = dense_solution.and_then(|solution| {
            lower_schur.and_then(|lower| {
                explicit_lower_action(lower, solution)
                    .ok()
                    .map(|applied| (rhs - applied).norm())
            })
        });
        CholeskyPcgArmOutcome {
            report: CholeskyPcgArmReport {
                status,
                iterations: None,
                recursive_residual: None,
                action_true_residual: None,
                target: None,
                lower_true_residual: None,
                dense_reference_lower_true_residual,
                dense_reference_action_true_residual,
                dense_reference_original_pose_equation_residual,
                original_pose_equation_residual: None,
                pose_error_vs_dense: None,
                landmark_error_vs_dense: None,
                feasibility: None,
                prediction: None,
            },
            solution: None,
            landmark_delta: None,
        }
    }

    fn populate_cholesky_arm_diagnostics(
        outcome: &mut CholeskyPcgArmOutcome,
        ba: &BundleAdjustment,
        system: &NormalEquationsBa,
        lambda: f64,
        lower_schur: &DMatrix<f64>,
        rhs: &DVector<f64>,
        factors: Option<&[LandmarkCholesky]>,
    ) {
        let (Some(solution), Some(landmark_delta)) =
            (outcome.solution.as_ref(), outcome.landmark_delta.as_ref())
        else {
            return;
        };
        outcome.report.original_pose_equation_residual =
            original_pose_equation_residual(system, lambda, solution, landmark_delta);
        outcome.report.prediction =
            predicted_decrease(system, lambda, solution, landmark_delta).ok();
        outcome.report.feasibility = Some(match factors {
            Some(factors) => feasibility_with_cholesky(
                ba,
                system,
                lambda,
                lower_schur,
                rhs,
                solution,
                landmark_delta,
                outcome.report.action_true_residual,
                factors,
            ),
            None => feasibility(
                ba,
                system,
                lambda,
                lower_schur,
                rhs,
                solution,
                landmark_delta,
                outcome.report.action_true_residual,
            ),
        });
    }

    fn pcg_error_details(
        error: &implicit_schur::ImplicitSchurError,
    ) -> (Option<usize>, Option<f64>, Option<f64>, Option<f64>) {
        match *error {
            implicit_schur::ImplicitSchurError::ResidualCheckFailed {
                iterations,
                recursive_norm,
                true_norm,
                target,
            }
            | implicit_schur::ImplicitSchurError::MaxIterations {
                iterations,
                recursive_norm,
                residual_norm: true_norm,
                target,
            } => (
                Some(iterations),
                Some(recursive_norm),
                Some(true_norm),
                Some(target),
            ),
            _ => (None, None, None, None),
        }
    }

    fn failed_explicit_pcg_reports(lambda: f64, status: String) -> Vec<ExplicitPcgIsolationReport> {
        [128_usize, 512_usize]
            .into_iter()
            .map(|pcg_max_iterations| ExplicitPcgIsolationReport {
                lambda,
                pcg_max_iterations,
                status: status.clone(),
                iterations: None,
                recursive_residual: None,
                lower_true_residual: None,
                implicit_true_residual: None,
                target: None,
                dense_lower_true_residual: None,
                dense_implicit_true_residual: None,
                pose_error_vs_dense: None,
                landmark_error_vs_dense: None,
                implicit_pcg_status: "not_run".to_owned(),
                implicit_pcg_iterations: None,
                implicit_pcg_true_residual: None,
                implicit_pcg_target: None,
                explicit_vs_implicit_pose_error: None,
                explicit_vs_implicit_landmark_error: None,
                feasibility: None,
                prediction: None,
            })
            .collect()
    }

    fn run_explicit_pcg_isolation(
        ba: &BundleAdjustment,
    ) -> Result<Vec<ExplicitPcgIsolationReport>, String> {
        let (system, pose_blocks, _landmark_count, _observation_count) = build_oracle_system(ba)?;
        let CameraHessian::PoseDiagonal(_diagonal) = &system.h_pp else {
            return Err("explicit PCG oracle requires pose blocks".to_owned());
        };
        let dimension = pose_blocks * 6;
        let mut reports = Vec::new();

        // Keep the two PR #80 damping cases and add the measured practical
        // acceptance-region case without changing the legacy four reports.
        for lambda in [1.0e-4, 1.0e5, 1.0e10] {
            let (raw_schur, rhs, _singular_landmarks) = match explicit_schur_rhs(&system, lambda) {
                Ok(value) => value,
                Err(error) => {
                    reports.extend(failed_explicit_pcg_reports(
                        lambda,
                        format!("failure:explicit_schur:{error}"),
                    ));
                    continue;
                }
            };
            let mut lower_schur = raw_schur;
            mirror_lower_triangle(&mut lower_schur);
            let operator = match implicit_schur::ImplicitSchurOperator::new(&system, lambda) {
                Ok(operator) => operator,
                Err(error) => {
                    reports.extend(failed_explicit_pcg_reports(
                        lambda,
                        format!("failure:implicit_operator:{error:?}"),
                    ));
                    continue;
                }
            };

            let explicit_pose = solve_normal_equations(&lower_schur, &rhs).ok();
            let explicit_landmarks = explicit_pose
                .as_ref()
                .and_then(|pose| operator.complete_delta(pose).ok());
            let dense_lower_true_residual = explicit_pose.as_ref().and_then(|pose| {
                explicit_lower_action(&lower_schur, pose)
                    .ok()
                    .map(|applied| (&rhs - applied).norm())
            });
            let dense_implicit_true_residual = explicit_pose.as_ref().and_then(|pose| {
                operator
                    .apply(pose)
                    .ok()
                    .map(|applied| (&rhs - applied).norm())
            });

            for pcg_max_iterations in [128_usize, 512_usize] {
                let options = implicit_schur::PcgOptions {
                    max_iterations: pcg_max_iterations,
                    relative_tolerance: ORACLE_PC_TOLERANCE,
                    absolute_tolerance: ORACLE_PC_TOLERANCE,
                };

                // Both arms consume this same rhs and preconditioner from one
                // assembled normal system.  The production arm is retained as
                // the recurrence/diagnostic comparison for the generic test
                // recurrence used by the explicit action.
                let implicit_result = operator.solve_pcg(&rhs, options);
                let (
                    implicit_pcg_status,
                    implicit_pcg_iterations,
                    implicit_pcg_true_residual,
                    implicit_pcg_target,
                    implicit_solution,
                ) = match implicit_result {
                    Ok(result) => (
                        "success".to_owned(),
                        Some(result.iterations),
                        Some(result.residual_norm),
                        Some(result.target),
                        Some(result.solution),
                    ),
                    Err(error) => {
                        let (iterations, _recursive, residual, target) = pcg_error_details(&error);
                        (
                            format!("failure:{error:?}"),
                            iterations,
                            residual,
                            target,
                            None,
                        )
                    }
                };

                let explicit_result = solve_test_pcg(
                    &rhs,
                    dimension,
                    options,
                    |x| explicit_lower_action(&lower_schur, x),
                    |residual| operator.apply_preconditioner(residual),
                );
                let (
                    explicit_status,
                    explicit_iterations,
                    explicit_recursive_residual,
                    explicit_lower_true_residual,
                    explicit_implicit_true_residual,
                    explicit_target,
                    explicit_solution,
                ) = match explicit_result {
                    Ok(result) => {
                        let lower_true_residual =
                            explicit_lower_action(&lower_schur, &result.solution)
                                .ok()
                                .map(|applied| (&rhs - applied).norm());
                        let implicit_true_residual = operator
                            .apply(&result.solution)
                            .ok()
                            .map(|applied| (&rhs - applied).norm());
                        (
                            "success_lower_pcg".to_owned(),
                            Some(result.iterations),
                            None,
                            lower_true_residual,
                            implicit_true_residual,
                            Some(result.target),
                            Some(result.solution),
                        )
                    }
                    Err(error) => {
                        let (iterations, recursive_residual, lower_true_residual, target) =
                            pcg_error_details(&error);
                        (
                            format!("failure:{error:?}"),
                            iterations,
                            recursive_residual,
                            lower_true_residual,
                            None,
                            target,
                            None,
                        )
                    }
                };

                let delta_landmarks = explicit_solution
                    .as_ref()
                    .and_then(|pose| operator.complete_delta(pose).ok());
                let feasibility = explicit_solution.as_ref().and_then(|pose| {
                    delta_landmarks.as_ref().map(|landmarks| {
                        feasibility(
                            ba,
                            &system,
                            lambda,
                            &lower_schur,
                            &rhs,
                            pose,
                            landmarks,
                            explicit_implicit_true_residual,
                        )
                    })
                });
                let prediction = explicit_solution.as_ref().and_then(|pose| {
                    delta_landmarks.as_ref().and_then(|landmarks| {
                        predicted_decrease(&system, lambda, pose, landmarks).ok()
                    })
                });
                let pose_error_vs_dense = explicit_solution
                    .as_ref()
                    .and_then(|pose| explicit_pose.as_ref().map(|dense| (pose - dense).norm()));
                let landmark_error_vs_dense = delta_landmarks.as_ref().and_then(|landmarks| {
                    explicit_landmarks
                        .as_ref()
                        .map(|dense| (landmarks - dense).norm())
                });
                let explicit_vs_implicit_pose_error =
                    explicit_solution.as_ref().and_then(|explicit| {
                        implicit_solution
                            .as_ref()
                            .map(|implicit| (explicit - implicit).norm())
                    });
                let implicit_landmarks = implicit_solution
                    .as_ref()
                    .and_then(|pose| operator.complete_delta(pose).ok());
                let explicit_vs_implicit_landmark_error =
                    delta_landmarks.as_ref().and_then(|explicit| {
                        implicit_landmarks
                            .as_ref()
                            .map(|implicit| (explicit - implicit).norm())
                    });
                reports.push(ExplicitPcgIsolationReport {
                    lambda,
                    pcg_max_iterations,
                    status: explicit_status,
                    iterations: explicit_iterations,
                    recursive_residual: explicit_recursive_residual,
                    lower_true_residual: explicit_lower_true_residual,
                    implicit_true_residual: explicit_implicit_true_residual,
                    target: explicit_target,
                    dense_lower_true_residual,
                    dense_implicit_true_residual,
                    pose_error_vs_dense,
                    landmark_error_vs_dense,
                    implicit_pcg_status,
                    implicit_pcg_iterations,
                    implicit_pcg_true_residual,
                    implicit_pcg_target,
                    explicit_vs_implicit_pose_error,
                    explicit_vs_implicit_landmark_error,
                    feasibility,
                    prediction,
                });
            }
        }
        Ok(reports)
    }

    fn run_cholesky_pcg_isolation(
        ba: &BundleAdjustment,
    ) -> Result<Vec<CholeskyPcgIsolationReport>, String> {
        let (system, pose_blocks, _landmark_count, _observation_count) = build_oracle_system(ba)?;
        let CameraHessian::PoseDiagonal(_diagonal) = &system.h_pp else {
            return Err("Cholesky PCG oracle requires pose blocks".to_owned());
        };
        let dimension = pose_blocks * 6;
        let mut reports = Vec::with_capacity(6);

        for lambda in [1.0e-4, 1.0e5, 1.0e10] {
            let general_metrics = general_landmark_factor_metrics(&system, lambda)?;
            let (general_raw_schur, general_rhs, _singular_landmarks) =
                explicit_schur_rhs(&system, lambda)?;
            let mut general_lower_schur = general_raw_schur;
            mirror_lower_triangle(&mut general_lower_schur);
            let general_operator = implicit_schur::ImplicitSchurOperator::new(&system, lambda)
                .map_err(|error| format!("general operator: {error:?}"))?;
            let general_dense_solution =
                solve_normal_equations(&general_lower_schur, &general_rhs).ok();
            let general_dense_landmark_delta = general_dense_solution
                .as_ref()
                .and_then(|solution| general_operator.complete_delta(solution).ok());
            let general_dense_action_true_residual =
                general_dense_solution.as_ref().and_then(|solution| {
                    general_operator
                        .apply(solution)
                        .ok()
                        .map(|applied| (general_operator.rhs() - applied).norm())
                });
            let general_dense_original_pose_equation_residual =
                match (&general_dense_solution, &general_dense_landmark_delta) {
                    (Some(solution), Some(landmarks)) => {
                        original_pose_equation_residual(&system, lambda, solution, landmarks)
                    }
                    _ => None,
                };

            let cholesky_build = CholeskySchurOperator::new(&system, lambda);
            let (cholesky_operator, cholesky_metrics) = match cholesky_build {
                Ok(value) => value,
                Err(failure) => {
                    for pcg_max_iterations in [128_usize, 512_usize] {
                        let options = implicit_schur::PcgOptions {
                            max_iterations: pcg_max_iterations,
                            relative_tolerance: ORACLE_PC_TOLERANCE,
                            absolute_tolerance: ORACLE_PC_TOLERANCE,
                        };
                        let mut general = run_cholesky_pcg_arm(
                            &general_rhs,
                            dimension,
                            options,
                            &general_lower_schur,
                            general_dense_solution.as_ref(),
                            general_dense_landmark_delta.as_ref(),
                            general_dense_action_true_residual,
                            general_dense_original_pose_equation_residual,
                            |x| general_operator.apply(x),
                            |residual| general_operator.apply_preconditioner(residual),
                            |pose| general_operator.complete_delta(pose),
                            "general_pcg",
                        );
                        populate_cholesky_arm_diagnostics(
                            &mut general,
                            ba,
                            &system,
                            lambda,
                            &general_lower_schur,
                            &general_rhs,
                            None,
                        );
                        let cholesky = failed_cholesky_pcg_arm(
                            format!("failure:cholesky_build:{}", failure.reason),
                            &general_rhs,
                            None,
                            None,
                            None,
                            None,
                        );
                        reports.push(CholeskyPcgIsolationReport {
                            lambda,
                            pcg_max_iterations,
                            general_metrics,
                            cholesky_metrics: failure.metrics,
                            general: general.report,
                            cholesky: cholesky.report,
                            general_vs_cholesky_pose_error: None,
                            general_vs_cholesky_landmark_error: None,
                            rhs_error: None,
                            probe_action_error: None,
                            probe_preconditioner_error: None,
                        });
                    }
                    continue;
                }
            };

            let (cholesky_raw_schur, cholesky_rhs) =
                match explicit_cholesky_schur_rhs(&system, &cholesky_operator) {
                    Ok(value) => value,
                    Err(error) => {
                        for pcg_max_iterations in [128_usize, 512_usize] {
                            let options = implicit_schur::PcgOptions {
                                max_iterations: pcg_max_iterations,
                                relative_tolerance: ORACLE_PC_TOLERANCE,
                                absolute_tolerance: ORACLE_PC_TOLERANCE,
                            };
                            let mut general = run_cholesky_pcg_arm(
                                &general_rhs,
                                dimension,
                                options,
                                &general_lower_schur,
                                general_dense_solution.as_ref(),
                                general_dense_landmark_delta.as_ref(),
                                general_dense_action_true_residual,
                                general_dense_original_pose_equation_residual,
                                |x| general_operator.apply(x),
                                |residual| general_operator.apply_preconditioner(residual),
                                |pose| general_operator.complete_delta(pose),
                                "general_pcg",
                            );
                            populate_cholesky_arm_diagnostics(
                                &mut general,
                                ba,
                                &system,
                                lambda,
                                &general_lower_schur,
                                &general_rhs,
                                None,
                            );
                            let cholesky = failed_cholesky_pcg_arm(
                                format!("failure:cholesky_explicit_schur:{error}"),
                                &general_rhs,
                                None,
                                None,
                                None,
                                None,
                            );
                            reports.push(CholeskyPcgIsolationReport {
                                lambda,
                                pcg_max_iterations,
                                general_metrics,
                                cholesky_metrics,
                                general: general.report,
                                cholesky: cholesky.report,
                                general_vs_cholesky_pose_error: None,
                                general_vs_cholesky_landmark_error: None,
                                rhs_error: None,
                                probe_action_error: None,
                                probe_preconditioner_error: None,
                            });
                        }
                        continue;
                    }
                };
            let mut cholesky_lower_schur = cholesky_raw_schur;
            mirror_lower_triangle(&mut cholesky_lower_schur);
            let cholesky_dense_solution =
                solve_normal_equations(&cholesky_lower_schur, &cholesky_rhs).ok();
            let cholesky_dense_landmark_delta = cholesky_dense_solution
                .as_ref()
                .and_then(|solution| cholesky_operator.complete_delta(solution).ok());
            let cholesky_dense_action_true_residual =
                cholesky_dense_solution.as_ref().and_then(|solution| {
                    cholesky_operator
                        .apply(solution)
                        .ok()
                        .map(|applied| (cholesky_operator.rhs() - applied).norm())
                });
            let cholesky_dense_original_pose_equation_residual =
                match (&cholesky_dense_solution, &cholesky_dense_landmark_delta) {
                    (Some(solution), Some(landmarks)) => {
                        original_pose_equation_residual(&system, lambda, solution, landmarks)
                    }
                    _ => None,
                };

            let probe = DVector::from_iterator(
                dimension,
                (0..dimension)
                    .map(|index| 0.001 * ((index % 17) as f64 - 8.0) + 0.00001 * index as f64),
            );
            let rhs_error = (&general_rhs - cholesky_operator.rhs()).norm();
            let probe_action_error = match (
                general_operator.apply(&probe),
                cholesky_operator.apply(&probe),
            ) {
                (Ok(general), Ok(cholesky)) => Some((general - cholesky).norm()),
                _ => None,
            };
            let probe_preconditioner_error = match (
                general_operator.apply_preconditioner(&probe),
                cholesky_operator.apply_preconditioner(&probe),
            ) {
                (Ok(general), Ok(cholesky)) => Some((general - cholesky).norm()),
                _ => None,
            };

            for pcg_max_iterations in [128_usize, 512_usize] {
                let options = implicit_schur::PcgOptions {
                    max_iterations: pcg_max_iterations,
                    relative_tolerance: ORACLE_PC_TOLERANCE,
                    absolute_tolerance: ORACLE_PC_TOLERANCE,
                };
                let mut general = run_cholesky_pcg_arm(
                    &general_rhs,
                    dimension,
                    options,
                    &general_lower_schur,
                    general_dense_solution.as_ref(),
                    general_dense_landmark_delta.as_ref(),
                    general_dense_action_true_residual,
                    general_dense_original_pose_equation_residual,
                    |x| general_operator.apply(x),
                    |residual| general_operator.apply_preconditioner(residual),
                    |pose| general_operator.complete_delta(pose),
                    "general_pcg",
                );
                populate_cholesky_arm_diagnostics(
                    &mut general,
                    ba,
                    &system,
                    lambda,
                    &general_lower_schur,
                    &general_rhs,
                    None,
                );
                let mut cholesky = run_cholesky_pcg_arm(
                    cholesky_operator.rhs(),
                    dimension,
                    options,
                    &cholesky_lower_schur,
                    cholesky_dense_solution.as_ref(),
                    cholesky_dense_landmark_delta.as_ref(),
                    cholesky_dense_action_true_residual,
                    cholesky_dense_original_pose_equation_residual,
                    |x| cholesky_operator.apply(x),
                    |residual| cholesky_operator.apply_preconditioner(residual),
                    |pose| cholesky_operator.complete_delta(pose),
                    "cholesky_pcg",
                );
                populate_cholesky_arm_diagnostics(
                    &mut cholesky,
                    ba,
                    &system,
                    lambda,
                    &cholesky_lower_schur,
                    cholesky_operator.rhs(),
                    Some(&cholesky_operator.factors),
                );
                let general_vs_cholesky_pose_error =
                    match (general.solution.as_ref(), cholesky.solution.as_ref()) {
                        (Some(general), Some(cholesky)) => Some((general - cholesky).norm()),
                        _ => None,
                    };
                let general_vs_cholesky_landmark_error = match (
                    general.landmark_delta.as_ref(),
                    cholesky.landmark_delta.as_ref(),
                ) {
                    (Some(general), Some(cholesky)) => Some((general - cholesky).norm()),
                    _ => None,
                };
                reports.push(CholeskyPcgIsolationReport {
                    lambda,
                    pcg_max_iterations,
                    general_metrics,
                    cholesky_metrics,
                    general: general.report,
                    cholesky: cholesky.report,
                    general_vs_cholesky_pose_error,
                    general_vs_cholesky_landmark_error,
                    rhs_error: Some(rhs_error),
                    probe_action_error,
                    probe_preconditioner_error,
                });
            }
        }
        Ok(reports)
    }

    fn schur_asymmetry(matrix: &DMatrix<f64>) -> f64 {
        let mut max_difference: f64 = 0.0;
        for row in 0..matrix.nrows() {
            for column in (row + 1)..matrix.ncols() {
                max_difference =
                    max_difference.max((matrix[(row, column)] - matrix[(column, row)]).abs());
            }
        }
        max_difference
    }

    fn mirror_lower_triangle(matrix: &mut DMatrix<f64>) {
        for row in 0..matrix.nrows() {
            for column in (row + 1)..matrix.ncols() {
                matrix[(row, column)] = matrix[(column, row)];
            }
        }
    }

    fn predicted_decrease(
        system: &NormalEquationsBa,
        lambda: f64,
        delta_pose: &DVector<f64>,
        delta_landmarks: &DVector<f64>,
    ) -> Result<PredictedDecrease, String> {
        let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
            return Err("predicted-decrease oracle requires pose blocks".to_owned());
        };
        if delta_pose.len() != diagonal.len() * 6
            || delta_landmarks.len() != system.landmarks.len() * 3
        {
            return Err("predicted-decrease delta dimensions mismatch".to_owned());
        }
        let mut gradient_dot = system.b_p.dot(delta_pose);
        let mut hessian_quadratic = 0.0;
        let mut delta_squared = delta_pose.norm_squared();
        for (pose, block) in diagonal.iter().enumerate() {
            let delta: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
            hessian_quadratic += delta.dot(&(block * delta));
        }
        for (landmark_index, landmark) in system.landmarks.iter().enumerate() {
            let delta: Vector3<f64> = delta_landmarks
                .fixed_rows::<3>(landmark_index * 3)
                .into_owned();
            gradient_dot += landmark.b_l.dot(&delta);
            hessian_quadratic += delta.dot(&(landmark.h_ll * delta));
            delta_squared += delta.norm_squared();
            for (pose, cross) in &landmark.cross {
                let pose_delta: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
                hessian_quadratic += 2.0 * pose_delta.dot(&(cross * delta));
            }
        }
        let damped_quadratic = hessian_quadratic + lambda * delta_squared;
        let half_damped = -gradient_dot - 0.5 * damped_quadratic;
        let squared_damped = -2.0 * gradient_dot - damped_quadratic;
        let squared_undamped = -2.0 * gradient_dot - hessian_quadratic;
        let values = [half_damped, squared_damped, squared_undamped];
        if !values.iter().all(|value| value.is_finite()) {
            return Err("predicted decrease is non-finite".to_owned());
        }
        Ok(PredictedDecrease {
            half_damped,
            squared_damped,
            squared_undamped,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn feasibility(
        ba: &BundleAdjustment,
        system: &NormalEquationsBa,
        lambda: f64,
        schur: &DMatrix<f64>,
        rhs: &DVector<f64>,
        delta_pose: &DVector<f64>,
        delta_landmarks: &DVector<f64>,
        implicit_pose_true_residual: Option<f64>,
    ) -> Feasibility {
        let finite = delta_pose.iter().all(|value| value.is_finite())
            && delta_landmarks.iter().all(|value| value.is_finite());
        let pose_residual = rhs - schur * delta_pose;
        let pose_true_residual = pose_residual.norm();
        let mut max_landmark = 0.0_f64;
        for (index, landmark) in system.landmarks.iter().enumerate() {
            let delta_l: Vector3<f64> = delta_landmarks.fixed_rows::<3>(index * 3).into_owned();
            let mut h_ll = landmark.h_ll;
            for component in 0..3 {
                h_ll[(component, component)] += lambda;
            }
            if h_ll.try_inverse().is_none() {
                continue;
            }
            let mut residual = h_ll * delta_l + landmark.b_l;
            for (pose, cross) in &landmark.cross {
                let delta_p: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
                residual += cross.transpose() * delta_p;
            }
            max_landmark = max_landmark.max(residual.norm());
        }
        let (pose_index, landmark_index) = variable_indices(ba);
        let mut geometry_cost = 0.0_f64;
        let mut geometry_max_error = 0.0_f64;
        let mut geometry_observation_count = 0_usize;
        let mut geometry_invalid_observations = 0_usize;
        let mut geometry_nonpositive_depth = 0_usize;
        for observation in &ba.rig_observations {
            let Some(pose) = ba.poses.get(&observation.keyframe_id) else {
                geometry_invalid_observations += 1;
                continue;
            };
            let Some(point) = ba.landmarks.get(&observation.landmark_id) else {
                geometry_invalid_observations += 1;
                continue;
            };
            let mut updated_pose = pose.world_to_camera.clone();
            if let Some(&index) = pose_index.get(&observation.keyframe_id) {
                let xi: Vector6<f64> = delta_pose.fixed_rows::<6>(index * 6).into_owned();
                if xi.iter().all(|value| value.is_finite()) {
                    updated_pose = updated_pose.compose(&SE3::exp(&xi));
                } else {
                    geometry_invalid_observations += 1;
                    continue;
                }
            }
            let mut updated_point = *point;
            if let Some(&index) = landmark_index.get(&observation.landmark_id) {
                let delta: Vector3<f64> = delta_landmarks.fixed_rows::<3>(index * 3).into_owned();
                if delta.iter().all(|value| value.is_finite()) {
                    updated_point = Point3::from(point.coords + delta);
                } else {
                    geometry_invalid_observations += 1;
                    continue;
                }
            }
            let point_camera = observation
                .sensor_from_rig
                .compose(&updated_pose)
                .transform_point(&updated_point);
            if !point_camera.z.is_finite() || point_camera.z <= 0.0 {
                geometry_nonpositive_depth += 1;
                geometry_invalid_observations += 1;
                continue;
            }
            let Some(projected) = observation.camera.project(&point_camera) else {
                geometry_invalid_observations += 1;
                continue;
            };
            let residual = projected - observation.xy;
            let squared = residual.norm_squared();
            if !point_camera.coords.iter().all(|value| value.is_finite())
                || !projected.coords.iter().all(|value| value.is_finite())
                || !squared.is_finite()
            {
                geometry_invalid_observations += 1;
                continue;
            }
            let error = squared.sqrt();
            if !error.is_finite() {
                geometry_invalid_observations += 1;
                continue;
            }
            geometry_observation_count += 1;
            geometry_cost += squared;
            geometry_max_error = geometry_max_error.max(error);
        }
        let geometry_rms_error = if geometry_observation_count == 0 {
            f64::NAN
        } else {
            (geometry_cost / geometry_observation_count as f64).sqrt()
        };
        let geometry_feasible = geometry_observation_count > 0
            && geometry_invalid_observations == 0
            && geometry_cost.is_finite()
            && geometry_rms_error.is_finite()
            && geometry_max_error.is_finite();
        Feasibility {
            finite,
            pose_true_residual,
            implicit_pose_true_residual,
            max_landmark_backsub_residual: max_landmark,
            geometry_observation_count,
            geometry_invalid_observations,
            geometry_nonpositive_depth,
            geometry_cost,
            geometry_rms_error,
            geometry_max_error,
            geometry_feasible,
            feasible: finite
                && pose_true_residual.is_finite()
                && max_landmark.is_finite()
                && geometry_feasible,
        }
    }

    fn cholesky_backsub_residual(
        system: &NormalEquationsBa,
        lambda: f64,
        _factors: &[LandmarkCholesky],
        delta_pose: &DVector<f64>,
        delta_landmarks: &DVector<f64>,
    ) -> f64 {
        let mut max_residual = 0.0_f64;
        for (index, landmark) in system.landmarks.iter().enumerate() {
            let delta_l: Vector3<f64> = delta_landmarks.fixed_rows::<3>(index * 3).into_owned();
            let mut h_ll = landmark.h_ll;
            for component in 0..3 {
                h_ll[(component, component)] += lambda;
            }
            let mut residual = h_ll * delta_l + landmark.b_l;
            for (pose, cross) in &landmark.cross {
                let delta_p: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
                residual += cross.transpose() * delta_p;
            }
            let norm = residual.norm();
            if !norm.is_finite() {
                return f64::NAN;
            }
            max_residual = max_residual.max(norm);
        }
        max_residual
    }

    #[allow(clippy::too_many_arguments)]
    fn feasibility_with_cholesky(
        ba: &BundleAdjustment,
        system: &NormalEquationsBa,
        lambda: f64,
        schur: &DMatrix<f64>,
        rhs: &DVector<f64>,
        delta_pose: &DVector<f64>,
        delta_landmarks: &DVector<f64>,
        implicit_pose_true_residual: Option<f64>,
        factors: &[LandmarkCholesky],
    ) -> Feasibility {
        let mut result = feasibility(
            ba,
            system,
            lambda,
            schur,
            rhs,
            delta_pose,
            delta_landmarks,
            implicit_pose_true_residual,
        );
        result.max_landmark_backsub_residual =
            cholesky_backsub_residual(system, lambda, factors, delta_pose, delta_landmarks);
        result.feasible = result.finite
            && result.pose_true_residual.is_finite()
            && result.max_landmark_backsub_residual.is_finite()
            && result.geometry_feasible;
        result
    }

    fn original_pose_equation_residual(
        system: &NormalEquationsBa,
        lambda: f64,
        delta_pose: &DVector<f64>,
        delta_landmarks: &DVector<f64>,
    ) -> Option<f64> {
        let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
            return None;
        };
        if delta_pose.len() != diagonal.len() * 6
            || delta_landmarks.len() != system.landmarks.len() * 3
        {
            return None;
        }
        let mut residual = system.b_p.clone();
        for (pose, block) in diagonal.iter().enumerate() {
            let mut damped = *block;
            for component in 0..6 {
                damped[(component, component)] += lambda;
            }
            let delta: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
            let value = damped * delta;
            for component in 0..6 {
                residual[pose * 6 + component] += value[component];
            }
        }
        for (landmark_index, landmark) in system.landmarks.iter().enumerate() {
            let delta: Vector3<f64> = delta_landmarks
                .fixed_rows::<3>(landmark_index * 3)
                .into_owned();
            for (pose, cross) in &landmark.cross {
                let value: Vector6<f64> = cross * delta;
                for component in 0..6 {
                    residual[pose * 6 + component] += value[component];
                }
            }
        }
        let norm = residual.norm();
        norm.is_finite().then_some(norm)
    }

    fn schur_action_scales(
        system: &NormalEquationsBa,
        lambda: f64,
        x: &DVector<f64>,
    ) -> Result<(f64, f64, f64), String> {
        let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
            return Err("Schur scale oracle requires pose blocks".to_owned());
        };
        if x.len() != diagonal.len() * 6 {
            return Err("Schur scale probe has the wrong dimension".to_owned());
        }
        let mut base = DVector::<f64>::zeros(x.len());
        for (pose, block) in diagonal.iter().enumerate() {
            let mut damped = *block;
            for component in 0..6 {
                damped[(component, component)] += lambda;
            }
            let value: Vector6<f64> = damped * x.fixed_rows::<6>(pose * 6).into_owned();
            base.fixed_rows_mut::<6>(pose * 6).copy_from(&value);
        }
        let mut eliminated = DVector::<f64>::zeros(x.len());
        for landmark in &system.landmarks {
            let mut h_ll = landmark.h_ll;
            for component in 0..3 {
                h_ll[(component, component)] += lambda;
            }
            let Some(inverse) = h_ll.try_inverse() else {
                continue;
            };
            let mut projected = Vector3::zeros();
            for (pose, cross) in &landmark.cross {
                let x_pose: Vector6<f64> = x.fixed_rows::<6>(pose * 6).into_owned();
                projected += cross.transpose() * x_pose;
            }
            let reduced = inverse * projected;
            for (pose, cross) in &landmark.cross {
                let value: Vector6<f64> = cross * reduced;
                for component in 0..6 {
                    eliminated[pose * 6 + component] += value[component];
                }
            }
        }
        let base_norm = base.norm();
        let eliminated_norm = eliminated.norm();
        let arithmetic_scale = (base_norm + eliminated_norm).max(1.0);
        if ![base_norm, eliminated_norm, arithmetic_scale]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err("Schur arithmetic scale is non-finite".to_owned());
        }
        Ok((base_norm, eliminated_norm, arithmetic_scale))
    }

    #[allow(clippy::too_many_arguments)]
    fn oracle_failure_reports(
        lambda: f64,
        pose_blocks: usize,
        landmark_count: usize,
        observation_count: usize,
        dimension: usize,
        singular_landmarks: usize,
        raw_schur_asymmetry: Option<f64>,
        schur_base_action_norm: Option<f64>,
        schur_eliminated_action_norm: Option<f64>,
        schur_arithmetic_scale: Option<f64>,
        status: String,
    ) -> Vec<OracleCaseReport> {
        [128_usize, 512_usize]
            .into_iter()
            .map(|pcg_max_iterations| OracleCaseReport {
                lambda,
                pcg_max_iterations,
                pose_blocks,
                landmark_count,
                observation_count,
                schur_dimension: dimension,
                singular_landmarks,
                raw_schur_asymmetry,
                schur_base_action_norm,
                schur_eliminated_action_norm,
                schur_arithmetic_scale,
                operator_raw_action_error: None,
                operator_raw_action_relative_error: None,
                operator_lower_action_error: None,
                operator_lower_action_relative_error: None,
                operator_rhs_error: None,
                direct_explicit_pose_error: None,
                direct_explicit_landmark_error: None,
                direct_feasibility: None,
                explicit_feasibility: None,
                direct_prediction: None,
                explicit_prediction: None,
                pcg_status: status.clone(),
                pcg_iterations: None,
                pcg_true_residual: None,
                pcg_target: None,
                matrix_free_pose_error: None,
                matrix_free_landmark_error: None,
                matrix_free_feasibility: None,
                matrix_free_prediction: None,
            })
            .collect()
    }

    fn run_oracle(ba: &BundleAdjustment) -> Result<Vec<OracleCaseReport>, String> {
        let (system, pose_blocks, landmark_count, observation_count) = build_oracle_system(ba)?;
        let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
            return Err("oracle system did not retain pose blocks".to_owned());
        };
        let mut reports = Vec::new();
        for lambda in [1.0e-4, 1.0e10] {
            let (raw_schur, explicit_rhs, singular_landmarks) =
                match explicit_schur_rhs(&system, lambda) {
                    Ok(value) => value,
                    Err(error) => {
                        reports.extend(oracle_failure_reports(
                            lambda,
                            pose_blocks,
                            landmark_count,
                            observation_count,
                            pose_blocks * 6,
                            0,
                            None,
                            None,
                            None,
                            None,
                            format!("oracle_failure:explicit_schur:{error}"),
                        ));
                        continue;
                    }
                };
            let raw_asymmetry = schur_asymmetry(&raw_schur);
            let dimension = raw_schur.nrows();
            let probe = DVector::from_iterator(
                dimension,
                (0..dimension)
                    .map(|index| 0.001 * ((index % 17) as f64 - 8.0) + 0.00001 * index as f64),
            );
            let mut lower_schur = raw_schur.clone();
            mirror_lower_triangle(&mut lower_schur);
            let lower_action = &lower_schur * &probe;
            let raw_action = &raw_schur * &probe;
            let scale_metrics = schur_action_scales(&system, lambda, &probe).ok();
            let schur_base_action_norm = scale_metrics.map(|metrics| metrics.0);
            let schur_eliminated_action_norm = scale_metrics.map(|metrics| metrics.1);
            let schur_arithmetic_scale = scale_metrics.map(|metrics| metrics.2);
            let mut solve_failures = Vec::new();
            let explicit_pose = match solve_normal_equations(&lower_schur, &explicit_rhs) {
                Ok(solution) => Some(solution),
                Err(error) => {
                    solve_failures.push(format!("explicit_schur_factor:{error:?}"));
                    None
                }
            };
            let explicit_operator =
                match implicit_schur::ImplicitSchurOperator::new(&system, lambda) {
                    Ok(operator) => operator,
                    Err(error) => {
                        reports.extend(oracle_failure_reports(
                            lambda,
                            pose_blocks,
                            landmark_count,
                            observation_count,
                            dimension,
                            singular_landmarks,
                            Some(raw_asymmetry),
                            schur_base_action_norm,
                            schur_eliminated_action_norm,
                            schur_arithmetic_scale,
                            format!("oracle_failure:implicit_operator:{error:?}"),
                        ));
                        continue;
                    }
                };
            let operator_action = match explicit_operator.apply(&probe) {
                Ok(action) => action,
                Err(error) => {
                    reports.extend(oracle_failure_reports(
                        lambda,
                        pose_blocks,
                        landmark_count,
                        observation_count,
                        dimension,
                        singular_landmarks,
                        Some(raw_asymmetry),
                        schur_base_action_norm,
                        schur_eliminated_action_norm,
                        schur_arithmetic_scale,
                        format!("oracle_failure:implicit_apply:{error:?}"),
                    ));
                    continue;
                }
            };
            let operator_rhs_error = (explicit_operator.rhs() - &explicit_rhs).norm();
            let operator_raw_action_error = (&operator_action - &raw_action).norm();
            let operator_lower_action_error = (&operator_action - &lower_action).norm();
            let operator_raw_action_relative_error =
                schur_arithmetic_scale.map(|scale| operator_raw_action_error / scale);
            let operator_lower_action_relative_error =
                schur_arithmetic_scale.map(|scale| operator_lower_action_error / scale);
            let mut direct_cache = None;
            let direct_solution = match solve_step_pose_blocks(
                &system,
                diagonal.clone(),
                pose_blocks,
                landmark_count,
                lambda,
                &mut direct_cache,
            ) {
                Ok(solution) => Some(solution),
                Err(error) => {
                    solve_failures.push(format!("direct_schur_factor:{error:?}"));
                    None
                }
            };
            let direct_pose = direct_solution.as_ref().map(|solution| &solution.0);
            let direct_landmarks = direct_solution.as_ref().map(|solution| &solution.1);
            let explicit_landmarks = explicit_pose
                .as_ref()
                .and_then(|pose| explicit_operator.complete_delta(pose).ok());
            let direct_implicit_true_residual = direct_pose.and_then(|pose| {
                explicit_operator
                    .apply(pose)
                    .ok()
                    .map(|applied| (explicit_operator.rhs() - applied).norm())
            });
            let explicit_implicit_true_residual = explicit_pose.as_ref().and_then(|pose| {
                explicit_operator
                    .apply(pose)
                    .ok()
                    .map(|applied| (explicit_operator.rhs() - applied).norm())
            });
            let direct_feasibility = match (direct_pose, direct_landmarks) {
                (Some(pose), Some(landmarks)) => Some(feasibility(
                    ba,
                    &system,
                    lambda,
                    &lower_schur,
                    &explicit_rhs,
                    pose,
                    landmarks,
                    direct_implicit_true_residual,
                )),
                _ => None,
            };
            let explicit_feasibility = match (explicit_pose.as_ref(), explicit_landmarks.as_ref()) {
                (Some(pose), Some(landmarks)) => Some(feasibility(
                    ba,
                    &system,
                    lambda,
                    &lower_schur,
                    &explicit_rhs,
                    pose,
                    landmarks,
                    explicit_implicit_true_residual,
                )),
                _ => None,
            };
            let direct_prediction = match (direct_pose, direct_landmarks) {
                (Some(pose), Some(landmarks)) => {
                    predicted_decrease(&system, lambda, pose, landmarks).ok()
                }
                _ => None,
            };
            let explicit_prediction = match (explicit_pose.as_ref(), explicit_landmarks.as_ref()) {
                (Some(pose), Some(landmarks)) => {
                    predicted_decrease(&system, lambda, pose, landmarks).ok()
                }
                _ => None,
            };
            let direct_explicit_pose_error = match (direct_pose, explicit_pose.as_ref()) {
                (Some(direct), Some(explicit)) => Some((direct - explicit).norm()),
                _ => None,
            };
            let direct_explicit_landmark_error =
                match (direct_landmarks, explicit_landmarks.as_ref()) {
                    (Some(direct), Some(explicit)) => Some((direct - explicit).norm()),
                    _ => None,
                };
            for pcg_max_iterations in [128_usize, 512_usize] {
                let pcg_result = explicit_operator.solve_pcg(
                    explicit_operator.rhs(),
                    implicit_schur::PcgOptions {
                        max_iterations: pcg_max_iterations,
                        relative_tolerance: ORACLE_PC_TOLERANCE,
                        absolute_tolerance: ORACLE_PC_TOLERANCE,
                    },
                );
                let (
                    pcg_status,
                    pcg_iterations,
                    pcg_true_residual,
                    pcg_target,
                    matrix_free_pose_error,
                    matrix_free_landmark_error,
                    matrix_free_feasibility,
                    matrix_free_prediction,
                ) = match pcg_result {
                    Ok(result) => match explicit_operator.apply(&result.solution) {
                        Err(error) => (
                            format!("failure:recheck_true_residual:{error:?}"),
                            Some(result.iterations),
                            None,
                            Some(result.target),
                            None,
                            None,
                            None,
                            None,
                        ),
                        Ok(applied) => {
                            let true_residual = explicit_operator.rhs() - applied;
                            match explicit_operator.complete_delta(&result.solution) {
                                Err(error) => (
                                    format!("failure:landmark_backsub:{error:?}"),
                                    Some(result.iterations),
                                    Some(true_residual.norm()),
                                    Some(result.target),
                                    None,
                                    None,
                                    None,
                                    None,
                                ),
                                Ok(matrix_free_landmarks) => {
                                    let feasibility = feasibility(
                                        ba,
                                        &system,
                                        lambda,
                                        &lower_schur,
                                        &explicit_rhs,
                                        &result.solution,
                                        &matrix_free_landmarks,
                                        Some(true_residual.norm()),
                                    );
                                    match predicted_decrease(
                                        &system,
                                        lambda,
                                        &result.solution,
                                        &matrix_free_landmarks,
                                    ) {
                                        Err(error) => (
                                            format!("failure:prediction:{error}"),
                                            Some(result.iterations),
                                            Some(true_residual.norm()),
                                            Some(result.target),
                                            explicit_pose
                                                .as_ref()
                                                .map(|pose| (&result.solution - pose).norm()),
                                            explicit_landmarks.as_ref().map(|landmarks| {
                                                (&matrix_free_landmarks - landmarks).norm()
                                            }),
                                            Some(feasibility),
                                            None,
                                        ),
                                        Ok(prediction) => (
                                            "success".to_owned(),
                                            Some(result.iterations),
                                            Some(true_residual.norm()),
                                            Some(result.target),
                                            explicit_pose
                                                .as_ref()
                                                .map(|pose| (&result.solution - pose).norm()),
                                            explicit_landmarks.as_ref().map(|landmarks| {
                                                (&matrix_free_landmarks - landmarks).norm()
                                            }),
                                            Some(feasibility),
                                            Some(prediction),
                                        ),
                                    }
                                }
                            }
                        }
                    },
                    Err(error) => {
                        let (iterations, residual, target) = error.diagnostics();
                        (
                            format!("failure:{error:?}"),
                            iterations,
                            residual,
                            target,
                            None,
                            None,
                            None,
                            None,
                        )
                    }
                };
                let pcg_status = if solve_failures.is_empty() {
                    pcg_status
                } else {
                    format!("{};pcg={}", solve_failures.join(","), pcg_status)
                };
                reports.push(OracleCaseReport {
                    lambda,
                    pcg_max_iterations,
                    pose_blocks,
                    landmark_count,
                    observation_count,
                    schur_dimension: dimension,
                    singular_landmarks,
                    raw_schur_asymmetry: Some(raw_asymmetry),
                    schur_base_action_norm,
                    schur_eliminated_action_norm,
                    schur_arithmetic_scale,
                    operator_raw_action_error: Some(operator_raw_action_error),
                    operator_raw_action_relative_error,
                    operator_lower_action_error: Some(operator_lower_action_error),
                    operator_lower_action_relative_error,
                    operator_rhs_error: Some(operator_rhs_error),
                    direct_explicit_pose_error,
                    direct_explicit_landmark_error,
                    direct_feasibility: direct_feasibility.clone(),
                    explicit_feasibility: explicit_feasibility.clone(),
                    direct_prediction: direct_prediction.clone(),
                    explicit_prediction: explicit_prediction.clone(),
                    pcg_status,
                    pcg_iterations,
                    pcg_true_residual,
                    pcg_target,
                    matrix_free_pose_error,
                    matrix_free_landmark_error,
                    matrix_free_feasibility,
                    matrix_free_prediction,
                });
            }
        }
        Ok(reports)
    }

    fn synthetic_rig_problem() -> BundleAdjustment {
        let camera = Camera::pinhole(1, 640, 480, 420.0, 418.0, 320.0, 240.0);
        let sensor_one = SE3::new(
            nalgebra::UnitQuaternion::from_euler_angles(0.01, -0.02, 0.03),
            Vector3::new(0.2, -0.01, 0.02),
        );
        let truth_poses = [
            Pose::identity(),
            Pose::from_world_to_camera(
                nalgebra::UnitQuaternion::from_euler_angles(0.01, -0.015, 0.02),
                Vector3::new(-0.18, 0.01, 0.03),
            ),
        ];
        let mut ba = BundleAdjustment::new(camera.clone());
        ba.add_pose(0, truth_poses[0].clone());
        ba.add_pose(
            1,
            Pose::from_world_to_camera(
                nalgebra::UnitQuaternion::from_euler_angles(0.013, -0.012, 0.018),
                Vector3::new(-0.20, 0.02, 0.04),
            ),
        );
        ba.fix_pose(0);
        for id in 0..8_u64 {
            let truth = Point3::new(
                -0.8 + 0.23 * id as f64,
                -0.4 + 0.11 * (id % 4) as f64,
                4.0 + 0.3 * id as f64,
            );
            ba.add_landmark(
                id,
                Point3::from(truth.coords + Vector3::new(0.004, -0.003, 0.006)),
            );
            for (frame_id, pose) in truth_poses.iter().enumerate() {
                for sensor_from_rig in [SE3::identity(), sensor_one.clone()] {
                    let sensor_pose = sensor_from_rig.compose(&pose.world_to_camera);
                    let xy = camera
                        .project(&sensor_pose.transform_point(&truth))
                        .expect("synthetic rig point must project");
                    ba.add_rig_observation(BaRigObservation {
                        keyframe_id: frame_id as u64,
                        landmark_id: id,
                        xy,
                        camera: camera.clone(),
                        sensor_from_rig,
                    });
                }
            }
        }
        ba
    }

    fn non_diagonal_landmark_system() -> NormalEquationsBa {
        let h_pp = Matrix6::identity() * 20.0;
        let cross = Matrix6x3::from_row_slice(&[
            0.8, -0.2, 0.1, 0.0, 0.4, -0.3, 0.2, 0.1, 0.5, -0.1, 0.3, 0.2, 0.6, -0.4, 0.2, 0.1,
            0.2, 0.7,
        ]);
        NormalEquationsBa {
            h_pp: CameraHessian::PoseDiagonal(vec![h_pp]),
            b_p: DVector::from_iterator(6, (0..6).map(|index| 0.2 + index as f64 * 0.03)),
            landmarks: vec![LandmarkBlock {
                h_ll: Matrix3::new(8.0, 1.0, 0.5, 1.0, 6.0, -0.3, 0.5, -0.3, 7.0),
                b_l: Vector3::new(0.4, -0.7, 0.2),
                cross: vec![(0, cross)],
            }],
        }
    }

    #[test]
    fn cholesky_landmark_arm_matches_general_inverse_for_nondiagonal_block() {
        let system = non_diagonal_landmark_system();
        for lambda in [0.0, 0.5, 1.0e5] {
            let general = implicit_schur::ImplicitSchurOperator::new(&system, lambda).unwrap();
            let (cholesky, metrics) = CholeskySchurOperator::new(&system, lambda).unwrap();
            assert!(metrics.h_ll_max_asymmetry < 1.0e-12);
            assert!(metrics.inverse_max_asymmetry.unwrap() < 1.0e-12);
            assert!(metrics.h_ll_inverse_identity_residual.unwrap().is_finite());
            assert!((general.rhs() - cholesky.rhs()).norm() < 1.0e-10);
            let probe = DVector::from_iterator(6, (0..6).map(|index| 0.1 + index as f64 * 0.07));
            assert!(
                (general.apply(&probe).unwrap() - cholesky.apply(&probe).unwrap()).norm() < 1.0e-10
            );
            assert!(
                (general.apply_preconditioner(&probe).unwrap()
                    - cholesky.apply_preconditioner(&probe).unwrap())
                .norm()
                    < 1.0e-10
            );
            assert!(
                (general.complete_delta(&probe).unwrap()
                    - cholesky.complete_delta(&probe).unwrap())
                .norm()
                    < 1.0e-10
            );

            let (general_raw, general_rhs, _) = explicit_schur_rhs(&system, lambda).unwrap();
            let (cholesky_raw, cholesky_rhs) =
                explicit_cholesky_schur_rhs(&system, &cholesky).unwrap();
            let mut general_lower = general_raw;
            let mut cholesky_lower = cholesky_raw;
            mirror_lower_triangle(&mut general_lower);
            mirror_lower_triangle(&mut cholesky_lower);
            assert!((general_rhs - cholesky_rhs).norm() < 1.0e-10);
            assert!((general_lower - cholesky_lower).norm() < 1.0e-10);
        }
    }

    #[test]
    fn cholesky_landmark_arm_rejects_asymmetric_input_without_fallback() {
        let mut system = non_diagonal_landmark_system();
        if let Some(landmark) = system.landmarks.first_mut() {
            landmark.h_ll[(0, 1)] += 1.0e-6;
        }
        let error = match CholeskySchurOperator::new(&system, 0.5) {
            Ok(_) => panic!("asymmetric landmark block must not use a Cholesky fallback"),
            Err(error) => error,
        };
        assert!(error.reason.contains("not symmetric"));
    }

    #[test]
    fn cholesky_landmark_arm_rejects_symmetric_non_spd_input_without_general_fallback() {
        let mut system = non_diagonal_landmark_system();
        if let Some(landmark) = system.landmarks.first_mut() {
            landmark.h_ll = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
            landmark.cross = vec![(0, Matrix6x3::zeros())];
        }
        assert!(implicit_schur::ImplicitSchurOperator::new(&system, 0.1).is_ok());
        let error = match CholeskySchurOperator::new(&system, 0.1) {
            Ok(_) => panic!("symmetric non-SPD block must not use a general fallback"),
            Err(error) => error,
        };
        assert!(error.reason.contains("not SPD"));
    }

    #[test]
    fn synthetic_oracle_compares_both_damping_values_and_coefficients() {
        let reports = run_oracle(&synthetic_rig_problem()).unwrap();
        assert_eq!(reports.len(), 4);
        for report in reports {
            assert!(report.raw_schur_asymmetry.unwrap().is_finite());
            assert!(report.operator_rhs_error.unwrap() < 1.0e-8);
            assert!(report.operator_raw_action_relative_error.unwrap() < 1.0e-8);
            assert!(
                report.operator_lower_action_error.unwrap().is_finite(),
                "lower-mirrored operator comparison must remain finite: {report:?}"
            );
            assert!(report.direct_explicit_pose_error.unwrap() < 1.0e-7);
            assert!(report.direct_explicit_landmark_error.unwrap() < 1.0e-7);
            let direct_feasibility = report
                .direct_feasibility
                .as_ref()
                .expect("synthetic direct arm should succeed");
            let explicit_feasibility = report
                .explicit_feasibility
                .as_ref()
                .expect("synthetic explicit arm should succeed");
            assert!(direct_feasibility.feasible);
            assert!(explicit_feasibility.feasible);
            assert!(direct_feasibility.geometry_feasible);
            assert!(explicit_feasibility.geometry_feasible);
            assert_eq!(direct_feasibility.geometry_invalid_observations, 0);
            assert_eq!(explicit_feasibility.geometry_invalid_observations, 0);
            assert!(direct_feasibility
                .implicit_pose_true_residual
                .is_some_and(f64::is_finite));
            assert!(explicit_feasibility
                .implicit_pose_true_residual
                .is_some_and(f64::is_finite));
            let direct_prediction = report
                .direct_prediction
                .as_ref()
                .expect("synthetic direct prediction should succeed");
            let explicit_prediction = report
                .explicit_prediction
                .as_ref()
                .expect("synthetic explicit prediction should succeed");
            assert!(
                (direct_prediction.squared_damped - 2.0 * direct_prediction.half_damped).abs()
                    < 1.0e-9
            );
            assert!(
                (explicit_prediction.squared_damped - 2.0 * explicit_prediction.half_damped).abs()
                    < 1.0e-9
            );
            if let Some(prediction) = report.matrix_free_prediction {
                assert!((prediction.squared_damped - 2.0 * prediction.half_damped).abs() < 1.0e-9);
            }
        }

        let isolation_reports = run_explicit_pcg_isolation(&synthetic_rig_problem()).unwrap();
        assert_eq!(isolation_reports.len(), 6);
        let mut successful_explicit_arms = 0;
        for report in isolation_reports {
            assert!(report.target.is_some_and(f64::is_finite));
            assert!(report.implicit_pcg_target.is_some_and(f64::is_finite));
            if report.status == "success_lower_pcg" {
                successful_explicit_arms += 1;
                assert!(report.lower_true_residual.is_some_and(f64::is_finite));
                assert!(report.implicit_true_residual.is_some_and(f64::is_finite));
                assert!(report.feasibility.is_some());
                assert!(report.prediction.is_some());
            }
            if report.implicit_pcg_status == "success" {
                assert!(report
                    .implicit_pcg_true_residual
                    .is_some_and(f64::is_finite));
            }
            if report.lambda == 1.0e10 {
                assert_eq!(report.status, "success_lower_pcg");
                assert_eq!(report.implicit_pcg_status, "success");
                let explicit_target = report.target.expect("high-lambda explicit target");
                let explicit_lower_residual = report
                    .lower_true_residual
                    .expect("high-lambda explicit lower residual");
                assert!(explicit_lower_residual <= explicit_target);
                assert!(report
                    .implicit_true_residual
                    .expect("high-lambda explicit implicit residual")
                    .is_finite());
                assert!(report
                    .dense_lower_true_residual
                    .expect("high-lambda dense lower residual")
                    .is_finite());
                assert!(report
                    .dense_implicit_true_residual
                    .expect("high-lambda dense implicit residual")
                    .is_finite());
                let implicit_target = report
                    .implicit_pcg_target
                    .expect("high-lambda implicit target");
                let implicit_residual = report
                    .implicit_pcg_true_residual
                    .expect("high-lambda implicit residual");
                assert!(implicit_residual <= implicit_target);
                assert!(
                    report
                        .pose_error_vs_dense
                        .expect("high-lambda explicit dense pose delta")
                        < 1.0e-7
                );
                assert!(
                    report
                        .landmark_error_vs_dense
                        .expect("high-lambda explicit dense landmark delta")
                        < 1.0e-7
                );
                assert!(
                    report
                        .explicit_vs_implicit_pose_error
                        .expect("high-lambda pose arm comparison")
                        < 1.0e-7
                );
                assert!(
                    report
                        .explicit_vs_implicit_landmark_error
                        .expect("high-lambda landmark arm comparison")
                        < 1.0e-7
                );
                assert!(
                    report
                        .feasibility
                        .as_ref()
                        .expect("high-lambda explicit feasibility")
                        .feasible
                );
                assert!(report
                    .prediction
                    .as_ref()
                    .expect("high-lambda explicit prediction")
                    .squared_damped
                    .is_finite());
            }
        }
        assert!(successful_explicit_arms > 0);

        let cholesky_reports = run_cholesky_pcg_isolation(&synthetic_rig_problem()).unwrap();
        assert_eq!(cholesky_reports.len(), 6);
        for report in cholesky_reports {
            assert!(report.general_metrics.h_ll_max_scale.is_finite());
            assert!(report.cholesky_metrics.h_ll_max_scale.is_finite());
            assert!(report.cholesky_metrics.h_ll_max_asymmetry < 1.0e-12);
            assert!(report.rhs_error.is_some_and(|error| error < 1.0e-8));
            assert!(report
                .probe_action_error
                .is_some_and(|error| error < 1.0e-8));
            assert!(report
                .probe_preconditioner_error
                .is_some_and(|error| error < 1.0e-8));
            if report.lambda == 1.0e10 {
                assert_eq!(report.general.status, "general_pcg");
                assert_eq!(report.cholesky.status, "cholesky_pcg");
                assert!(report
                    .general
                    .pose_error_vs_dense
                    .is_some_and(|error| error < 1.0e-7));
                assert!(report
                    .cholesky
                    .pose_error_vs_dense
                    .is_some_and(|error| error < 1.0e-7));
                assert!(report
                    .general
                    .landmark_error_vs_dense
                    .is_some_and(|error| error < 1.0e-7));
                assert!(report
                    .cholesky
                    .landmark_error_vs_dense
                    .is_some_and(|error| error < 1.0e-7));
                assert!(report
                    .general_vs_cholesky_pose_error
                    .is_some_and(|error| error < 1.0e-7));
                assert!(report
                    .general_vs_cholesky_landmark_error
                    .is_some_and(|error| error < 1.0e-7));
                assert!(report
                    .general
                    .original_pose_equation_residual
                    .is_some_and(f64::is_finite));
                assert!(report
                    .cholesky
                    .original_pose_equation_residual
                    .is_some_and(f64::is_finite));
                assert!(report
                    .general
                    .dense_reference_action_true_residual
                    .is_some_and(f64::is_finite));
                assert!(report
                    .cholesky
                    .dense_reference_action_true_residual
                    .is_some_and(f64::is_finite));
                assert!(report
                    .general
                    .dense_reference_original_pose_equation_residual
                    .is_some_and(f64::is_finite));
                assert!(report
                    .cholesky
                    .dense_reference_original_pose_equation_residual
                    .is_some_and(f64::is_finite));
                assert!(report
                    .general
                    .feasibility
                    .as_ref()
                    .is_some_and(|value| value.feasible));
                assert!(report
                    .cholesky
                    .feasibility
                    .as_ref()
                    .is_some_and(|value| value.feasible));
                assert!(report.general.prediction.is_some());
                assert!(report.cholesky.prediction.is_some());
            }
        }
    }

    #[test]
    fn predicted_decrease_matches_independent_squared_cost_fixture() {
        let mut cross = Matrix6x3::zeros();
        for index in 0..3 {
            cross[(index, index)] = 0.5;
        }
        let mut h_pp = Matrix6::identity();
        h_pp *= 2.0;
        let mut h_ll = Matrix3::identity();
        h_ll *= 3.0;
        let system = NormalEquationsBa {
            h_pp: CameraHessian::PoseDiagonal(vec![h_pp]),
            b_p: DVector::from_element(6, 1.0),
            landmarks: vec![LandmarkBlock {
                h_ll,
                b_l: Vector3::from_element(2.0),
                cross: vec![(0, cross)],
            }],
        };
        let delta_pose = DVector::from_element(6, 0.1);
        let delta_landmark = DVector::from_element(3, -0.2);
        let prediction = predicted_decrease(&system, 0.5, &delta_pose, &delta_landmark).unwrap();
        assert!((prediction.squared_undamped - 0.78).abs() < 1.0e-12);
        assert!((prediction.half_damped - 0.345).abs() < 1.0e-12);
        assert!((prediction.squared_damped - 0.69).abs() < 1.0e-12);
    }

    #[test]
    fn fixture_quaternion_reader_preserves_serialized_bits() {
        let values = [
            0.9238795325112867_f64,
            0.0_f64,
            0.3826834323650898_f64,
            0.0_f64,
            0.125_f64,
            -0.25_f64,
            0.5_f64,
        ];
        let serialized = values
            .iter()
            .map(|value| format!("{value:.17e}"))
            .collect::<Vec<_>>();
        let fields = serialized.iter().map(String::as_str).collect::<Vec<_>>();
        let parsed = parse_se3(&fields, 0, "bit-roundtrip").unwrap();
        let quaternion = parsed.rotation.quaternion();
        assert_eq!(quaternion.w.to_bits(), values[0].to_bits());
        assert_eq!(quaternion.i.to_bits(), values[1].to_bits());
        assert_eq!(quaternion.j.to_bits(), values[2].to_bits());
        assert_eq!(quaternion.k.to_bits(), values[3].to_bits());
        assert_eq!(parsed.translation.x.to_bits(), values[4].to_bits());
        assert_eq!(parsed.translation.y.to_bits(), values[5].to_bits());
        assert_eq!(parsed.translation.z.to_bits(), values[6].to_bits());

        let near_unit = [
            1.0 + f64::EPSILON,
            -0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
            0.0_f64,
        ];
        let serialized = near_unit
            .iter()
            .map(|value| format!("{value:.17e}"))
            .collect::<Vec<_>>();
        let fields = serialized.iter().map(String::as_str).collect::<Vec<_>>();
        let parsed = parse_se3(&fields, 0, "near-unit-bit-roundtrip").unwrap();
        let quaternion = parsed.rotation.quaternion();
        assert_eq!(quaternion.w.to_bits(), near_unit[0].to_bits());
        assert_eq!(quaternion.i.to_bits(), near_unit[1].to_bits());
    }

    #[test]
    fn fixture_reader_rejects_bad_order_and_oversized_header() {
        let base =
            std::env::temp_dir().join(format!("visloc-ba-oracle-parser-{}", std::process::id()));
        let _ = std::fs::remove_file(&base);
        std::fs::write(&base, "END\n").unwrap();
        let error = parse_fixture(&base).unwrap_err();
        assert!(error.contains("header must be the first record"));
        std::fs::write(&base, "VISLOC_BA_ORACLE_FIXTURE 1\nPOSE_COUNT 514\nEND\n").unwrap();
        let error = parse_fixture(&base).unwrap_err();
        assert!(error.contains("exceeds cap"));
        std::fs::write(&base, "VISLOC_BA_ORACLE_FIXTURE 1\nEND\nPOSE_COUNT 1\n").unwrap();
        let error = parse_fixture(&base).unwrap_err();
        assert!(error.contains("records after END"));
        let _ = std::fs::remove_file(base);
    }

    #[test]
    fn fixture_reader_roundtrips_initial_cost_bits() {
        let path = std::env::temp_dir().join(format!(
            "visloc-ba-oracle-valid-parser-{}",
            std::process::id()
        ));
        let hash = "0".repeat(64);
        let fixture = format!(
            "VISLOC_BA_ORACLE_FIXTURE 1\nSOURCE_SHA256 {hash}\nSOURCE_SHA256_CAMERAS {hash}\nSOURCE_SHA256_IMAGES {hash}\nSOURCE_SHA256_POINTS {hash}\nSOURCE_SHA256_MANIFEST {hash}\nINITIAL_COST 0.00000000000000000e+00\nINITIAL_COST_BITS 0\nCAMERA_COUNT 1\nPOSE_COUNT 1\nLANDMARK_COUNT 1\nOBSERVATION_COUNT 1\nCAMERA 1 PINHOLE 10 10 4 2 2 0 0\nPOSE 0 1 0 0 0 0 0 0\nFIXED_POSE 0\nLANDMARK 0 0 0 2\nRIG_OBSERVATION 0 0 0 0 1 1 0 0 0 0 0 0\nEND\n"
        );
        std::fs::write(&path, fixture).unwrap();
        let parsed = parse_fixture(&path).unwrap();
        assert_eq!(parsed.initial_cost.to_bits(), 0);
        assert_eq!(parsed.initial_cost_bits, 0);
        assert_eq!(parsed.ba.cost().to_bits(), 0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn oracle_caps_reject_before_dense_dimension_allocation() {
        assert!(check_caps(512, 8_192, 262_144, MAX_CROSS_PAIR_WORK).is_ok());
        assert!(check_caps(513, 1, 1, 1).is_err());
        assert!(check_caps(1, 8_193, 1, 1).is_err());
        assert!(check_caps(1, 1, 262_145, 1).is_err());
        assert!(check_caps(1, 1, 1, MAX_CROSS_PAIR_WORK + 1).is_err());
        assert!(check_caps(513, 1, 1, 1).is_err());
    }

    #[test]
    fn generic_test_pcg_matches_implicit_recurrence_and_failure_diagnostics() {
        let diagonal = vec![
            Matrix6::from_diagonal(&Vector6::from_element(20.0)),
            Matrix6::from_diagonal(&Vector6::from_element(22.0)),
        ];
        let mut cross = Matrix6x3::zeros();
        for component in 0..3 {
            cross[(component, component)] = 0.25;
        }
        let system = NormalEquationsBa {
            h_pp: CameraHessian::PoseDiagonal(diagonal),
            b_p: DVector::from_iterator(12, (0..12).map(|index| 0.03 * (index + 1) as f64)),
            landmarks: vec![LandmarkBlock {
                h_ll: Matrix3::from_diagonal(&Vector3::new(7.0, 8.0, 9.0)),
                b_l: Vector3::new(0.2, -0.1, 0.3),
                cross: vec![(0, cross), (1, -cross)],
            }],
        };
        let operator = implicit_schur::ImplicitSchurOperator::new(&system, 0.25).unwrap();
        let options = [
            implicit_schur::PcgOptions::default(),
            implicit_schur::PcgOptions {
                max_iterations: 1,
                relative_tolerance: 0.0,
                absolute_tolerance: 1.0e-30,
            },
        ];
        let mut saw_success = false;
        let mut saw_failure = false;
        for options in options {
            let expected = operator.solve_pcg(operator.rhs(), options);
            let actual = solve_test_pcg(
                operator.rhs(),
                operator.dimension(),
                options,
                |x| operator.apply(x),
                |residual| operator.apply_preconditioner(residual),
            );
            assert_eq!(actual, expected);
            match expected {
                Ok(_) => saw_success = true,
                Err(implicit_schur::ImplicitSchurError::MaxIterations { .. })
                | Err(implicit_schur::ImplicitSchurError::ResidualCheckFailed { .. }) => {
                    saw_failure = true;
                }
                Err(error) => panic!("unexpected PCG diagnostic: {error:?}"),
            }
        }
        assert!(saw_success);
        assert!(saw_failure);
    }

    #[test]
    #[ignore = "requires an explicitly exported frozen 1k fixture and source hash"]
    fn ignored_real_fixture_runs_bounded_damping_oracle() {
        let path = env::var_os("VISLOC_MATRIX_FREE_ORACLE_FIXTURE")
            .expect("VISLOC_MATRIX_FREE_ORACLE_FIXTURE is required; fixture absence is failure");
        let fixture = parse_fixture(Path::new(&path)).unwrap();
        let expected_hash = env::var("VISLOC_MATRIX_FREE_ORACLE_EXPECTED_SOURCE_SHA256")
            .expect("VISLOC_MATRIX_FREE_ORACLE_EXPECTED_SOURCE_SHA256 is required");
        assert_eq!(
            fixture.source_hashes["SOURCE_SHA256"], expected_hash,
            "fixture source hash does not match the requested frozen input"
        );
        let reports = run_oracle(&fixture.ba).unwrap();
        assert_eq!(reports.len(), 4);
        println!(
            "matrix_free_oracle_input_cost={:.17e} bits={}",
            fixture.initial_cost, fixture.initial_cost_bits
        );
        for report in reports {
            println!("matrix_free_oracle {report:?}");
        }
        let isolation_reports = run_explicit_pcg_isolation(&fixture.ba).unwrap();
        assert_eq!(isolation_reports.len(), 6);
        for report in isolation_reports {
            println!("explicit_pcg_isolation {report:?}");
        }
        let cholesky_reports = run_cholesky_pcg_isolation(&fixture.ba).unwrap();
        assert_eq!(cholesky_reports.len(), 6);
        for report in cholesky_reports {
            println!("cholesky_pcg_isolation {report:?}");
        }
    }
}

#[cfg(test)]
mod visual_jacobian_audit_tests {
    use super::*;
    use nalgebra::UnitQuaternion;

    fn audit_camera() -> Camera {
        Camera::pinhole(1, 1600, 1066, 879.4, 879.4, 803.4, 532.6)
    }

    fn audit_pose() -> Pose {
        Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.08, -0.11, 0.17),
            Vector3::new(0.24, -0.13, 0.31),
        )
    }

    fn assert_case_is_accurate(label: &str, case: BaVisualJacobianCase) {
        eprintln!(
            "ba-jacobian-test: case={label} residual={:.3e} depth={:.3e} pose=(abs {:.3e},rel {:.3e}; trans {:.3e}/{:.3e}; rot {:.3e}/{:.3e}) landmark=(abs {:.3e},rel {:.3e}) intrinsics=(abs {:.3e},rel {:.3e})",
            case.residual_norm,
            case.depth,
            case.pose_max_abs,
            case.pose_relative,
            case.pose_translation_max_abs,
            case.pose_translation_relative,
            case.pose_rotation_max_abs,
            case.pose_rotation_relative,
            case.landmark_max_abs,
            case.landmark_relative,
            case.intrinsics_max_abs,
            case.intrinsics_relative,
        );
        assert!(
            case.pose_max_abs < 1.0e-5 && case.pose_relative < 1.0e-6,
            "{label} pose Jacobian mismatch: {:?}",
            case
        );
        assert!(
            case.landmark_max_abs < 1.0e-5 && case.landmark_relative < 1.0e-6,
            "{label} landmark Jacobian mismatch: {:?}",
            case
        );
        assert!(
            case.intrinsics_max_abs < 1.0e-6 && case.intrinsics_relative < 1.0e-8,
            "{label} intrinsics Jacobian mismatch: {:?}",
            case
        );
    }

    #[test]
    fn analytic_visual_jacobians_match_finite_differences_across_regimes() {
        let camera = audit_camera();
        let pose = audit_pose();
        let normal_point = Point3::new(0.45, -0.35, 4.8);
        let normal_measurement = camera
            .project(&pose.transform_world_point(&normal_point))
            .unwrap();
        assert_case_is_accurate(
            "normal",
            audit_visual_jacobian_case(&camera, &pose, &normal_point, &normal_measurement, 1.0e-6)
                .unwrap(),
        );

        let far_point = Point3::new(15.0, -8.0, 10_000.0);
        let far_measurement = camera
            .project(&pose.transform_world_point(&far_point))
            .unwrap();
        assert_case_is_accurate(
            "far-depth",
            audit_visual_jacobian_case(&camera, &pose, &far_point, &far_measurement, 1.0e-6)
                .unwrap(),
        );

        // A very small camera baseline relative to depth is the low-parallax
        // regime that made the captured 27-camera point block ill-conditioned.
        // The per-observation Jacobian itself remains well-defined, so this
        // case checks that no special-case branch changes its numerical value.
        let low_parallax_point = Point3::new(-0.15, 0.12, 100.0);
        let low_parallax_measurement = camera
            .project(&pose.transform_world_point(&low_parallax_point))
            .unwrap();
        assert_case_is_accurate(
            "low-parallax",
            audit_visual_jacobian_case(
                &camera,
                &pose,
                &low_parallax_point,
                &low_parallax_measurement,
                1.0e-6,
            )
            .unwrap(),
        );

        let high_residual_measurement = normal_measurement + Vector2::new(80.0, -55.0);
        assert_case_is_accurate(
            "high-residual",
            audit_visual_jacobian_case(
                &camera,
                &pose,
                &normal_point,
                &high_residual_measurement,
                1.0e-6,
            )
            .unwrap(),
        );
    }

    #[test]
    fn bundle_audit_reports_low_parallax_and_high_residual_buckets() {
        let camera = audit_camera();
        let pose0 = Pose::identity();
        let pose1 =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.01, 0.0, 0.0));
        let point = Point3::new(0.2, -0.1, 100.0);
        let mut ba = BundleAdjustment::new(camera.clone());
        ba.add_pose(0, pose0.clone());
        ba.add_pose(1, pose1);
        ba.add_landmark(0, point);
        let exact0 = camera
            .project(&pose0.transform_world_point(&point))
            .unwrap();
        let exact1 = camera
            .project(&ba.poses[&1].transform_world_point(&point))
            .unwrap();
        ba.add_observation(BaObservation {
            keyframe_id: 0,
            landmark_id: 0,
            xy: exact0,
        });
        ba.add_observation(BaObservation {
            keyframe_id: 1,
            landmark_id: 0,
            xy: exact1 + Vector2::new(25.0, 0.0),
        });
        let report = audit_bundle_visual_jacobians(&ba, 16);
        assert_eq!(report.observations_seen, 2);
        assert_eq!(report.samples_audited, 2);
        assert_eq!(report.low_parallax.samples, 1);
        assert_eq!(report.high_residual.samples, 1);
        assert_eq!(report.invalid_samples, 0);
    }

    #[test]
    fn huber_weight_is_the_derivative_of_the_squared_residual_cost() {
        let kernel = RobustKernel::Huber { delta: 3.0 };
        for squared_residual in [1.0, 4.0, 16.0, 100.0] {
            let epsilon = 1.0e-6 * squared_residual;
            let numerical = (kernel.cost(squared_residual + epsilon)
                - kernel.cost(squared_residual - epsilon))
                / (2.0 * epsilon);
            let analytic = kernel.weight(squared_residual);
            assert!(
                (analytic - numerical).abs() < 1.0e-8,
                "s={squared_residual}: rho'={analytic}, finite difference={numerical}"
            );
        }
    }

    #[test]
    fn schur_rhs_and_back_substitution_match_the_full_normal_system() {
        let mut h_pp = DMatrix::<f64>::zeros(6, 6);
        for i in 0..6 {
            h_pp[(i, i)] = 10.0 + i as f64;
        }
        let mut h_ll = Matrix3::<f64>::zeros();
        h_ll[(0, 0)] = 4.0;
        h_ll[(1, 1)] = 5.0;
        h_ll[(2, 2)] = 6.0;
        let cross = Matrix6x3::<f64>::from_fn(|r, c| 0.03 * (r as f64 + 1.0) * (c as f64 + 2.0));
        let b_p = DVector::from_iterator(6, (0..6).map(|i| 0.2 * (i as f64 + 1.0)));
        let b_l = Vector3::new(-0.4, 0.3, 0.2);
        let mut system = NormalEquationsBa {
            h_pp: CameraHessian::Dense(h_pp.clone()),
            b_p: b_p.clone(),
            landmarks: vec![LandmarkBlock {
                h_ll,
                b_l,
                cross: vec![(0, cross)],
            }],
        };

        let (delta_p, delta_l) = solve_step(
            &mut system,
            1,
            1,
            0,
            0,
            0.0,
            LinearSolver::Dense,
            false,
            &mut None,
        )
        .unwrap();

        let mut sparse_system = NormalEquationsBa {
            h_pp: CameraHessian::Dense(h_pp.clone()),
            b_p: b_p.clone(),
            landmarks: vec![LandmarkBlock {
                h_ll,
                b_l,
                cross: vec![(0, cross)],
            }],
        };
        let (sparse_delta_p, sparse_delta_l) = solve_step(
            &mut sparse_system,
            1,
            1,
            0,
            0,
            0.0,
            LinearSolver::Sparse,
            false,
            &mut None,
        )
        .unwrap();

        let mut block_system = NormalEquationsBa {
            h_pp: CameraHessian::PoseDiagonal(vec![h_pp.fixed_view::<6, 6>(0, 0).into_owned()]),
            b_p: b_p.clone(),
            landmarks: vec![LandmarkBlock {
                h_ll,
                b_l,
                cross: vec![(0, cross)],
            }],
        };
        let (block_delta_p, block_delta_l) = solve_step(
            &mut block_system,
            1,
            1,
            0,
            0,
            0.0,
            LinearSolver::Sparse,
            false,
            &mut None,
        )
        .unwrap();

        let mut full_h = DMatrix::<f64>::zeros(9, 9);
        full_h.view_mut((0, 0), (6, 6)).copy_from(&h_pp);
        for r in 0..6 {
            for c in 0..3 {
                full_h[(r, 6 + c)] = cross[(r, c)];
                full_h[(6 + c, r)] = cross[(r, c)];
            }
        }
        full_h.view_mut((6, 6), (3, 3)).copy_from(&h_ll);
        let mut full_rhs = DVector::<f64>::zeros(9);
        for i in 0..6 {
            full_rhs[i] = -b_p[i];
        }
        for i in 0..3 {
            full_rhs[6 + i] = -b_l[i];
        }
        let full_delta = solve_normal_equations(&full_h, &full_rhs).unwrap();
        assert!((delta_p - full_delta.rows(0, 6)).norm() < 1.0e-10);
        assert!((delta_l - full_delta.rows(6, 3)).norm() < 1.0e-10);
        assert_eq!(block_delta_p.as_slice(), sparse_delta_p.as_slice());
        assert_eq!(block_delta_l.as_slice(), sparse_delta_l.as_slice());
        assert!((&sparse_delta_p - full_delta.rows(0, 6)).norm() < 1.0e-10);
        assert!((&sparse_delta_l - full_delta.rows(6, 3)).norm() < 1.0e-10);
    }

    #[test]
    fn pose_block_sparse_path_matches_dense_sparse_path_bitwise() {
        let diagonal = vec![
            Matrix6::from_diagonal(&Vector6::new(20.0, 21.0, 22.0, 23.0, 24.0, 25.0)),
            Matrix6::from_diagonal(&Vector6::new(26.0, 27.0, 28.0, 29.0, 30.0, 31.0)),
        ];
        let mut dense = DMatrix::zeros(12, 12);
        for (pose, block) in diagonal.iter().enumerate() {
            dense
                .fixed_view_mut::<6, 6>(pose * 6, pose * 6)
                .copy_from(block);
        }
        let cross0 = Matrix6x3::from_fn(|r, c| 0.01 * (r + c + 1) as f64);
        let cross1 = Matrix6x3::from_fn(|r, c| 0.015 * (2 * r + c + 1) as f64);
        let landmark = LandmarkBlock {
            h_ll: Matrix3::from_diagonal(&Vector3::new(8.0, 9.0, 10.0)),
            b_l: Vector3::new(0.2, -0.1, 0.3),
            cross: vec![(0, cross0), (1, cross1)],
        };
        let gradient = DVector::from_iterator(12, (0..12).map(|i| 0.03 * (i + 1) as f64));
        let mut dense_system = NormalEquationsBa {
            h_pp: CameraHessian::Dense(dense),
            b_p: gradient.clone(),
            landmarks: vec![landmark.clone()],
        };
        let mut block_system = NormalEquationsBa {
            h_pp: CameraHessian::PoseDiagonal(diagonal),
            b_p: gradient,
            landmarks: vec![landmark],
        };

        let dense_delta = solve_step(
            &mut dense_system,
            2,
            1,
            0,
            0,
            0.25,
            LinearSolver::Sparse,
            false,
            &mut None,
        )
        .unwrap();
        let mut symbolic_cache = None;
        let block_delta = solve_step(
            &mut block_system,
            2,
            1,
            0,
            0,
            0.25,
            LinearSolver::Sparse,
            false,
            &mut symbolic_cache,
        )
        .unwrap();

        assert_eq!(block_delta.0.as_slice(), dense_delta.0.as_slice());
        assert_eq!(block_delta.1.as_slice(), dense_delta.1.as_slice());
        assert!(symbolic_cache.is_some());
    }
}

#[cfg(test)]
mod imu_gradient_tests {
    use super::*;
    use crate::imu_preintegration::ImuPreintegrator;
    use nalgebra::UnitQuaternion;

    fn make_problem() -> BundleAdjustment {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut ba = BundleAdjustment::new(camera);
        let pose_i = Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.12, -0.08, 0.2),
            Vector3::new(0.3, -0.2, 0.1),
        );
        let pose_j = Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.17, -0.03, 0.27),
            Vector3::new(0.42, -0.15, 0.18),
        );
        ba.add_pose(10, pose_i);
        ba.add_pose(20, pose_j);
        ba.add_velocity(10, Vector3::new(0.4, -0.1, 0.2));
        ba.add_velocity(20, Vector3::new(0.35, 0.05, 0.1));
        ba.add_bias(10, Vector6::new(0.002, -0.001, 0.003, 0.02, -0.01, 0.03));
        ba.add_bias(20, Vector6::zeros());
        ba.set_imu_body_to_camera(SE3::new(
            UnitQuaternion::from_euler_angles(-0.04, 0.03, -0.02),
            Vector3::new(-0.02, -0.06, 0.01),
        ));
        let mut preintegrator = ImuPreintegrator::new();
        for _ in 0..20 {
            preintegrator.integrate_sample(
                Vector3::new(0.03, -0.02, 0.04),
                Vector3::new(0.2, -0.1, 9.7),
                0.01,
            );
        }
        ba.add_imu_factor(ImuPreintegrationFactor {
            keyframe_id_from: 10,
            keyframe_id_to: 20,
            delta: preintegrator.delta(),
            gravity_world: Vector3::new(0.0, 0.0, -9.81),
            weight_position: 1.3,
            weight_velocity: 0.8,
            weight_rotation: 1.1,
        });
        ba
    }

    fn perturb(problem: &mut BundleAdjustment, coordinate: usize, step: f64) {
        match coordinate {
            0..=11 => {
                let pose_slot = coordinate / 6;
                let component = coordinate % 6;
                let id = [10_u64, 20][pose_slot];
                let mut xi = Vector6::zeros();
                xi[component] = step;
                let pose = problem.poses.get_mut(&id).unwrap();
                pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&xi));
            }
            12..=17 => {
                let velocity_slot = (coordinate - 12) / 3;
                let component = (coordinate - 12) % 3;
                let id = [10_u64, 20][velocity_slot];
                problem.velocities.get_mut(&id).unwrap()[component] += step;
            }
            18..=29 => {
                let bias_slot = (coordinate - 18) / 6;
                let component = (coordinate - 18) % 6;
                let id = [10_u64, 20][bias_slot];
                problem.biases.get_mut(&id).unwrap()[component] += step;
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn analytic_imu_gradient_matches_central_difference_with_extrinsic() {
        let problem = make_problem();
        let linearized = problem.linearized_navigation_system().unwrap();
        assert_eq!(linearized.information.shape(), (30, 30));
        let epsilon = 1.0e-6;
        for coordinate in 0..30 {
            let mut plus = problem.clone();
            let mut minus = problem.clone();
            perturb(&mut plus, coordinate, epsilon);
            perturb(&mut minus, coordinate, -epsilon);
            let numerical = (plus.robust_cost(&RobustKernel::None)
                - minus.robust_cost(&RobustKernel::None))
                / (4.0 * epsilon);
            let analytic = linearized.gradient[coordinate];
            let tolerance = 2.0e-4 * analytic.abs().max(numerical.abs()).max(1.0);
            assert!(
                (analytic - numerical).abs() <= tolerance,
                "coordinate {coordinate}: analytic={analytic} numerical={numerical} tolerance={tolerance}"
            );
        }
    }
}

/// Tests for [`BaConfig::parallel`] (see the module's "Parallelism"
/// section): the serial and parallel assembly / Schur-reduction /
/// back-substitution paths must agree, and the parallel path must be
/// deterministic. The synthetic problem is sized past
/// `PARALLEL_MIN_OBSERVATIONS` / `PARALLEL_MIN_LANDMARKS` so these tests
/// actually exercise the parallel dispatch rather than falling through the
/// work gate to the serial loops.
#[cfg(test)]
mod parallel_ba_tests {
    use super::*;
    use nalgebra::UnitQuaternion;

    /// Deterministic `[0, 1)` pseudo-random value (GLSL-style sine hash) —
    /// avoids pulling in a `rand` dependency just to scatter synthetic
    /// points/poses reproducibly.
    fn pseudo_rand(seed: u64) -> f64 {
        let x = (seed as f64 + 1.0) * 12.9898;
        let y = x.sin() * 43758.5453;
        y - y.floor()
    }

    /// Build a synthetic multi-camera, multi-landmark monocular BA problem:
    /// `num_cameras` cameras translated along a horizontal baseline (plus a
    /// small yaw each) all observe every one of `num_landmarks` landmarks
    /// scattered in front of them, so every camera/landmark pair is a valid,
    /// positive-depth observation. The first two poses are fixed (anchor +
    /// scale, per the module doc's gauge-fixing rule); every other pose and
    /// every landmark is then nudged away from the ground truth that
    /// generated the observations by a small deterministic offset, so LM has
    /// a real (if easy, well-conditioned) problem to converge on rather than
    /// starting already at the optimum.
    fn build_synthetic_ba(num_cameras: usize, num_landmarks: usize) -> BundleAdjustment {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut ba = BundleAdjustment::new(camera.clone());

        let center = (num_cameras as f64 - 1.0) / 2.0;
        let gt_poses: Vec<Pose> = (0..num_cameras)
            .map(|i| {
                let baseline = (i as f64 - center) * 0.4;
                let yaw = (i as f64 - center) * 0.02;
                let rotation = UnitQuaternion::from_euler_angles(0.0, yaw, 0.0);
                let translation = Vector3::new(-baseline, 0.0, 0.0);
                Pose::from_world_to_camera(rotation, translation)
            })
            .collect();
        let gt_points: Vec<Point3<f64>> = (0..num_landmarks)
            .map(|j| {
                let x = (pseudo_rand(j as u64 * 3) - 0.5) * 6.0;
                let y = (pseudo_rand(j as u64 * 3 + 1) - 0.5) * 6.0;
                let z = 8.0 + pseudo_rand(j as u64 * 3 + 2) * 4.0;
                Point3::new(x, y, z)
            })
            .collect();

        for (i, pose) in gt_poses.iter().enumerate() {
            ba.add_pose(i as u64, pose.clone());
        }
        for (j, point) in gt_points.iter().enumerate() {
            ba.add_landmark(1_000_000 + j as u64, *point);
        }
        for (i, pose) in gt_poses.iter().enumerate() {
            for (j, point) in gt_points.iter().enumerate() {
                let xc = pose.transform_world_point(point);
                let xy = camera
                    .project(&xc)
                    .expect("synthetic landmarks stay in front of every camera");
                ba.add_observation(BaObservation {
                    keyframe_id: i as u64,
                    landmark_id: 1_000_000 + j as u64,
                    xy,
                });
            }
        }

        ba.fix_pose(0);
        ba.fix_pose(1);

        // Nudge every non-fixed pose off ground truth.
        for i in 2..num_cameras {
            let dxi = Vector6::new(
                (pseudo_rand(i as u64 * 7) - 0.5) * 0.02,
                (pseudo_rand(i as u64 * 7 + 1) - 0.5) * 0.02,
                (pseudo_rand(i as u64 * 7 + 2) - 0.5) * 0.02,
                (pseudo_rand(i as u64 * 7 + 3) - 0.5) * 0.01,
                (pseudo_rand(i as u64 * 7 + 4) - 0.5) * 0.01,
                (pseudo_rand(i as u64 * 7 + 5) - 0.5) * 0.01,
            );
            let pose = ba.poses.get_mut(&(i as u64)).expect("pose was just added");
            pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&dxi));
        }
        // Nudge every landmark off ground truth.
        for j in 0..num_landmarks {
            let d = Vector3::new(
                (pseudo_rand(j as u64 * 11) - 0.5) * 0.05,
                (pseudo_rand(j as u64 * 11 + 1) - 0.5) * 0.05,
                (pseudo_rand(j as u64 * 11 + 2) - 0.5) * 0.05,
            );
            let point = ba
                .landmarks
                .get_mut(&(1_000_000 + j as u64))
                .expect("landmark was just added");
            *point = Point3::from(point.coords + d);
        }

        ba
    }

    /// `num_landmarks` clears `PARALLEL_MIN_LANDMARKS` and `num_cameras *
    /// num_landmarks` clears `PARALLEL_MIN_OBSERVATIONS`, so every parallel
    /// path in the module (assembly, Schur reduction, back-substitution) is
    /// actually dispatched by these tests instead of falling through the
    /// work gate.
    const TEST_CAMERAS: usize = 6;
    const TEST_LANDMARKS: usize = 2_500;

    #[test]
    fn parallel_config_defaults_to_off() {
        assert!(!BaConfig::default().parallel);
    }

    #[test]
    fn serial_and_parallel_converge_to_the_same_result() {
        let mut ba_serial = build_synthetic_ba(TEST_CAMERAS, TEST_LANDMARKS);
        let mut ba_parallel = ba_serial.clone();

        let serial_config = BaConfig {
            max_iterations: 8,
            parallel: false,
            ..BaConfig::default()
        };
        let parallel_config = BaConfig {
            parallel: true,
            ..serial_config
        };

        let result_serial = ba_serial
            .optimize(&serial_config)
            .expect("serial BA should solve the synthetic problem");
        let result_parallel = ba_parallel
            .optimize(&parallel_config)
            .expect("parallel BA should solve the synthetic problem");

        assert!(result_serial.converged, "serial run should converge");
        assert!(result_parallel.converged, "parallel run should converge");

        // The parallel assembly / Schur-reduction / back-substitution paths
        // change only *how* the normal equations are computed, never the
        // summation order (see the module's "Parallelism" section), so the
        // two runs must land on bit-identical states -- a far tighter check
        // than the "~1e-9 relative" bar a reassociating design would need.
        assert_eq!(
            result_serial.final_cost, result_parallel.final_cost,
            "final cost must match exactly"
        );
        assert_eq!(
            result_serial.iterations.len(),
            result_parallel.iterations.len(),
            "iteration count must match exactly"
        );
        assert_eq!(
            ba_serial.poses, ba_parallel.poses,
            "poses must match exactly"
        );
        assert_eq!(
            ba_serial.landmarks, ba_parallel.landmarks,
            "landmarks must match exactly"
        );
    }

    #[test]
    fn parallel_sparse_ba_keeps_pose_block_system_and_matches_serial() {
        let mut ba_serial = build_synthetic_ba(TEST_CAMERAS, TEST_LANDMARKS);
        let mut ba_parallel = ba_serial.clone();
        let serial_config = BaConfig {
            max_iterations: 8,
            linear_solver: LinearSolver::Sparse,
            parallel: false,
            ..BaConfig::default()
        };
        let parallel_config = BaConfig {
            parallel: true,
            ..serial_config
        };

        let result_serial = ba_serial
            .optimize(&serial_config)
            .expect("serial sparse BA should solve the synthetic problem");
        let result_parallel = ba_parallel
            .optimize(&parallel_config)
            .expect("parallel sparse BA should solve the synthetic problem");

        assert_eq!(result_serial.final_cost, result_parallel.final_cost);
        assert_eq!(result_serial.iterations, result_parallel.iterations);
        assert_eq!(ba_serial.poses, ba_parallel.poses);
        assert_eq!(ba_serial.landmarks, ba_parallel.landmarks);
    }

    #[test]
    fn parallel_path_is_deterministic_across_runs() {
        let ba = build_synthetic_ba(TEST_CAMERAS, TEST_LANDMARKS);
        let mut ba_run_a = ba.clone();
        let mut ba_run_b = ba.clone();

        let config = BaConfig {
            max_iterations: 8,
            parallel: true,
            ..BaConfig::default()
        };

        let result_a = ba_run_a
            .optimize(&config)
            .expect("parallel BA should solve the synthetic problem");
        let result_b = ba_run_b
            .optimize(&config)
            .expect("parallel BA should solve the synthetic problem");

        assert_eq!(
            result_a.final_cost, result_b.final_cost,
            "repeated parallel runs must produce bitwise-identical cost"
        );
        assert_eq!(
            ba_run_a.poses, ba_run_b.poses,
            "repeated parallel runs must produce bitwise-identical poses"
        );
        assert_eq!(
            ba_run_a.landmarks, ba_run_b.landmarks,
            "repeated parallel runs must produce bitwise-identical landmarks"
        );
    }
}
