//! Schur-complement normal-equation assembly and the dense / sparse step solves.

use super::*;

/// Per-landmark contribution: the `H_ll` block, the `b_l` gradient, and
/// the `H_pl` cross blocks (one per pose that observed this landmark, in
/// arbitrary order). This is the only place the cross blocks are stored —
/// there is no full `H_PL` matrix.
#[derive(Clone)]
pub(super) struct LandmarkBlock {
    /// `3×3` Hessian summed over observations. Includes any λ damping.
    pub(super) h_ll: Matrix3<f64>,
    /// `3-vec` gradient summed over observations.
    pub(super) b_l: Vector3<f64>,
    /// `(pose_idx, J_pose^T · J_lm)` per observation that touches a
    /// non-fixed pose. Shape `6×3`.
    pub(super) cross: Vec<(usize, Matrix6x3<f64>)>,
}

/// Per-landmark block for the joint pose+structure+intrinsics solve
/// ([`BundleAdjustment::optimize_joint_intrinsics`]). Unlike [`LandmarkBlock`]
/// the cross blocks are keyed by the touching camera-block column-start (a pose
/// block of width 6, or the shared 4-wide intrinsics block) so a single map
/// holds both the pose and intrinsics couplings, and observations sharing a
/// camera block (e.g. the intrinsics, seen by every observation) accumulate.
pub(super) struct JointLandmarkBlock {
    pub(super) id: u64,
    pub(super) h_ll: Matrix3<f64>,
    pub(super) b_l: Vector3<f64>,
    /// `column_start → Σ_obs Jᵀ_cam · J_lm` (`rows × 3`, `rows ∈ {6, 4}`).
    pub(super) cross: BTreeMap<usize, DMatrix<f64>>,
}

/// Accumulate a `rows × 3` cross block into the per-landmark map at `col_start`.
pub(super) fn navigation_prior_delta(
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

pub(super) fn add_cross(
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
pub(super) struct NormalEquationsBa {
    /// Camera-state Hessian. Pure-visual sparse BA keeps only the diagonal
    /// 6×6 pose blocks assembled before Schur reduction; the general visual-
    /// inertial / prior-bearing path retains the dense representation.
    pub(super) h_pp: CameraHessian,
    /// Pose gradient, dense `6P`.
    pub(super) b_p: DVector<f64>,
    /// Landmark blocks indexed by landmark variable index.
    pub(super) landmarks: Vec<LandmarkBlock>,
}

/// Pre-Schur camera Hessian representation.
///
/// For pure visual BA every observation contributes only to one diagonal
/// pose block. Off-diagonal pose coupling appears later, during landmark
/// elimination, so allocating a dense `(6P)²` matrix here is unnecessary.
pub(super) enum CameraHessian {
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

    pub(super) fn into_dense(self) -> DMatrix<f64> {
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
pub(super) const PARALLEL_MIN_OBSERVATIONS: usize = 4_096;
/// Observations computed per rayon chunk in the parallel assembly path.
/// Bounds the transient `Vec<Option<MonoObsContribution>>` buffer to a few
/// megabytes regardless of the total observation count — a full-sequence BA
/// can carry tens of millions of observations, so materializing one
/// contribution per observation up front (rather than chunk by chunk) would
/// multiply the assembly's peak memory several-fold. The value does not
/// affect the result at all (see the merge comment on
/// [`assemble_mono_observations_parallel`]), so it is chosen purely for that
/// memory/dispatch-overhead trade-off.
pub(super) const PARALLEL_OBSERVATION_CHUNK: usize = 65_536;

/// Below this many landmarks, the parallel Schur-reduction and back-
/// substitution paths in [`solve_step`] are not dispatched. Mirrors
/// [`PARALLEL_MIN_OBSERVATIONS`].
pub(super) const PARALLEL_MIN_LANDMARKS: usize = 2_048;
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
    let projection = MonoProjection::for_camera(&ba.camera);
    let (predicted, j_pi) = mono_project_with_jacobian(projection, intrinsics, &xc)?;
    let residual = Vector2::new(predicted.x - obs.xy.x, predicted.y - obs.xy.y);

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
pub(super) fn build_normal_equations(
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
        let projection = MonoProjection::for_camera(&ba.camera);
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
            // Predicted pixel - measured pixel, and the projection Jacobian
            // J_π (2×3) at X_c (pinhole: (1/Z)[[fx, 0, -fx X/Z], [0, fy, -fy Y/Z]];
            // distortion-aware otherwise, see `MonoProjection`).
            let Some((predicted, j_pi)) = mono_project_with_jacobian(projection, intrinsics, &xc)
            else {
                continue;
            };
            let residual = Vector2::new(predicted.x - obs.xy.x, predicted.y - obs.xy.y);

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
    // Right-perturbation Jacobians (ﾏ・= translation perturbation, ﾏ・=
    // rotation perturbation, both 3-vec; world body centre
    // B = 竏坦盞t so 竏・/竏ぱ・= 竏棚, 竏・/竏ぱ・= [B]ﾃ・. Since
    // T_bw = T_bc T_cw, a right perturbation of T_cw is the same right
    // perturbation of T_bw, so no additional adjoint is required:
    //
    //   竏Ｓ_R/竏ぱ雲i =  Jr_inv(r_R) ﾂｷ R_bw箜ｼ
    //   竏Ｓ_R/竏ぱ雲j = 竏谷r_inv(r_R) ﾂｷ R_bw箜ｼ
    //   竏Ｓ_v/竏ぱ雲i = 竏坦_bw盞｢ ﾂｷ [v_j 竏・v_i 竏・gﾂｷﾎ杯]ﾃ・
    //   竏Ｓ_v/竏Ｗ_i = 竏坦_bw盞｢
    //   竏Ｓ_v/竏Ｗ_j =  R_bw盞｢
    //   竏Ｓ_p/竏ぱ＼i =  R_bw盞｢            竏Ｓ_p/竏ぱ＼j = 竏坦_bw盞｢
    //   竏Ｓ_p/竏ぱ雲i = 竏坦_bw盞｢ ﾂｷ [B_j 竏・v_iﾂｷﾎ杯 竏・ﾂｽ g ﾎ杯ﾂｲ]ﾃ・
    //   竏Ｓ_p/竏ぱ雲j =  R_bw盞｢ ﾂｷ [B_j]ﾃ・
    //   竏Ｓ_p/竏Ｗ_i = 竏槻杯 ﾂｷ R_bw盞｢
    //
    // The 9-vector residual is stacked [r_R; r_v; r_p] with axis-wise
    // weights `sqrt(weight_rotation, weight_velocity, weight_position)`
    // applied as a per-block 竏嗹 scaling so J盞J / J盞r come out
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

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn solve_step(
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
    solve_step_with_debug(
        system,
        p_count,
        l_count,
        v_count,
        b_count,
        lambda,
        linear_solver,
        parallel,
        block_symbolic_cache,
        None,
    )
}

// Keep the solver dimensions and policy explicit, as in the test wrapper above.
#[allow(clippy::too_many_arguments)]
pub(super) fn solve_step_with_debug(
    system: &mut NormalEquationsBa,
    p_count: usize,
    l_count: usize,
    v_count: usize,
    b_count: usize,
    lambda: f64,
    linear_solver: LinearSolver,
    parallel: bool,
    block_symbolic_cache: &mut Option<crate::block_cholesky::BlockSymbolic>,
    debug_context: Option<SchurBlockDebugContext<'_>>,
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
        return solve_step_pose_blocks_with_debug(
            system,
            diagonal,
            p_count,
            l_count,
            lambda,
            block_symbolic_cache,
            debug_context,
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
                // The reduced Hessian is already dense. Feed its lower blocks
                // directly to block Cholesky so a large temporary
                // Vec<(usize, usize, f64)> is never materialized.
                let streamed = crate::block_cholesky::DenseBlockTriplets::new(&s, 6);
                let sol = crate::block_cholesky::solve_spd_block(&streamed, dim, 6, &rhs, 0.0)
                    .map_err(|_| BaError::SingularSystem)?;
                drop(std::mem::take(&mut s));
                log_process_memory("ba-sparse-before-factor");
                sol
            } else {
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
#[cfg(test)]
pub(super) fn solve_step_pose_blocks(
    system: &NormalEquationsBa,
    diagonal: Vec<Matrix6<f64>>,
    p_count: usize,
    l_count: usize,
    lambda: f64,
    block_symbolic_cache: &mut Option<crate::block_cholesky::BlockSymbolic>,
) -> Result<(DVector<f64>, DVector<f64>), BaError> {
    solve_step_pose_blocks_with_debug(
        system,
        diagonal,
        p_count,
        l_count,
        lambda,
        block_symbolic_cache,
        None,
    )
}

pub(super) fn solve_step_pose_blocks_with_debug(
    system: &NormalEquationsBa,
    diagonal: Vec<Matrix6<f64>>,
    p_count: usize,
    l_count: usize,
    lambda: f64,
    block_symbolic_cache: &mut Option<crate::block_cholesky::BlockSymbolic>,
    debug_context: Option<SchurBlockDebugContext<'_>>,
) -> Result<(DVector<f64>, DVector<f64>), BaError> {
    let debug_diagonal = debug_context.as_ref().and_then(|context| {
        diagonal.get(context.pose_slot).map(|block| {
            let mut damped = *block;
            for k in 0..6 {
                damped[(k, k)] += lambda.max(0.0);
            }
            damped
        })
    });
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

    if let (Some(context), Some(diagonal)) = (debug_context, debug_diagonal) {
        if let Some(reduced) = columns
            .get(context.pose_slot)
            .and_then(|c| c.get(&context.pose_slot))
        {
            let counts = collect_schur_block_debug_counts(
                system,
                &h_ll_inv_cache,
                lambda,
                context.pose_slot,
            );
            emit_schur_block_debug(context, lambda, &diagonal, reduced, counts);
        }
    }
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
