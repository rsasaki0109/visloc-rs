//! `BundleAdjustment` joint pose / landmark / intrinsics refinement.

use super::*;

impl BundleAdjustment {
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
    pub(super) fn optimize_joint_intrinsics(
        &mut self,
        config: &BaConfig,
    ) -> Result<BaResult, BaError> {
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
            && matches!(
                self.camera.model,
                CameraModel::Pinhole | CameraModel::OpenCv
            )
            && self.stereo_observations.is_empty();
        if refine_dist {
            // Ensure the camera carries the two distortion slots (start at 0).
            while self.camera.params.len() < 6 {
                self.camera.params.push(0.0);
            }
        }
        // Opt-in tangential (p1, p2): the camera becomes an `OpenCv`
        // `[fx, fy, cx, cy, k1, k2, p1, p2]` (the only layout that carries them;
        // with p1 = p2 = 0 it projects exactly like the radial pinhole) and the
        // camera block grows to 8.
        let refine_tangential = refine_dist && config.refine_tangential_distortion;
        if refine_tangential {
            self.camera.model = CameraModel::OpenCv;
            while self.camera.params.len() < 8 {
                self.camera.params.push(0.0);
            }
        }
        let k_dim = if refine_tangential {
            8
        } else if refine_dist {
            6
        } else {
            4
        };
        let cam_dim = k_off + k_dim;
        if config.shared_focal {
            let f = 0.5 * (self.camera.params[0] + self.camera.params[1]);
            self.camera.params[0] = f;
            self.camera.params[1] = f;
        }

        let initial_cost = self.robust_cost_weighted(&kernel, None);
        let mut iterations: Vec<BaIterationStats> = Vec::with_capacity(config.max_iterations);
        let mut current_cost = initial_cost;
        let mut current_nonprojectable = self.nonprojectable_observation_count();
        let mut lambda = config.initial_lambda.unwrap_or(0.0);
        let mut converged = false;
        // `LinearSolver::Sparse`: keep the reduced camera system as a
        // block-sparse pose part plus a dense intrinsics border and solve it
        // with the block Cholesky (see `solve_joint_sparse`); `Dense` keeps
        // the historical dense `(6 P + k)^2` solve byte-for-byte.
        let sparse = config.linear_solver == LinearSolver::Sparse;
        let mut block_cache: Option<crate::block_cholesky::BlockSymbolic> = None;

        for iteration in 0..config.max_iterations {
            // Current distortion (reflects the running k1, k2 estimate) drives the
            // distortion-aware projection / Jacobians inside the build.
            let dist = self.camera.radial_distortion();
            // Tangential terms take the full Brown-Conrady branch whenever they
            // are being refined or the camera already carries non-zero (p1, p2).
            let tangential = if refine_tangential {
                Some(self.camera.tangential_distortion().unwrap_or((0.0, 0.0)))
            } else {
                self.camera.tangential_distortion()
            };
            let t_build = std::time::Instant::now();
            let (cam_dim_n, h_cc, b_c, lm_blocks) = self.build_joint_intrinsics_system(
                &pose_index,
                &landmark_index,
                &kernel,
                k_dim,
                dist,
                tangential,
                sparse,
            );
            debug_assert_eq!(cam_dim_n, cam_dim);
            let build_s = t_build.elapsed().as_secs_f64();
            let t_solve = std::time::Instant::now();
            let (solved, h_ll_inv_cache) = match h_cc {
                JointCameraHessian::Sparse {
                    pose_diag,
                    pose_k,
                    kk,
                } => solve_joint_sparse(
                    pose_diag,
                    pose_k,
                    kk,
                    &b_c,
                    &lm_blocks,
                    lambda,
                    config.shared_focal,
                    &mut block_cache,
                ),
                JointCameraHessian::Dense(h_cc) => {
                    // Damped Schur reduction (Levenberg Iﾂｷﾎｻ on both the camera and the
                    // landmark diagonals, exactly as `solve_step`).
                    let mut s = h_cc.clone();
                    if lambda > 0.0 {
                        for d in 0..cam_dim {
                            s[(d, d)] += lambda;
                        }
                    }
                    let mut b_reduced = -&b_c;
                    let mut h_ll_inv_cache: Vec<Option<Matrix3<f64>>> =
                        Vec::with_capacity(lm_blocks.len());
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
                        // S -= ﾎ｣ cross_a^T ﾂｷ H_ll^{-1} ﾂｷ cross_b ; b += cross_a^T H_ll^{-1} b_l.
                        for (cs_a, a) in &lm.cross {
                            let ah = a * inv; // (rows_a ﾃ・3)
                            for (cs_b, b) in &lm.cross {
                                let block = &ah * b.transpose(); // (rows_a ﾃ・rows_b)
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

                    let solved = if config.shared_focal {
                        // Reduce with T: full -> shared, fy (index k_off + 1) folded into
                        // fx (k_off): S' = T^T S T, b' = T^T b, delta = T delta'.
                        let fy = k_off + 1;
                        let map = |i: usize| {
                            if i < fy {
                                i
                            } else if i == fy {
                                k_off
                            } else {
                                i - 1
                            }
                        };
                        let mut s_red = DMatrix::<f64>::zeros(cam_dim - 1, cam_dim - 1);
                        let mut b_red = DVector::<f64>::zeros(cam_dim - 1);
                        for i in 0..cam_dim {
                            b_red[map(i)] += b_reduced[i];
                            for j in 0..cam_dim {
                                s_red[(map(i), map(j))] += s[(i, j)];
                            }
                        }
                        solve_normal_equations(&s_red, &b_red)
                            .map(|d| DVector::<f64>::from_fn(cam_dim, |i, _| d[map(i)]))
                    } else {
                        solve_normal_equations(&s, &b_reduced)
                    };
                    (solved.map_err(|_| ()), h_ll_inv_cache)
                }
            };
            let solve_s = t_solve.elapsed().as_secs_f64();
            let t_rest = std::time::Instant::now();
            let delta_cam = match solved {
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

            // Back-substitute landmark updates: ﾎｴ_L = H_ll^{-1}(竏鍛_l 竏・ﾎ｣ cross盞 ﾎｴ_cam).
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

            // Tentative update (save 竊・apply 竊・cost 竊・accept/reject).
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

            let t_cost = std::time::Instant::now();
            let cost_after = self.robust_cost_weighted(&kernel, None);
            let nonprojectable_after = self.nonprojectable_observation_count();
            if std::env::var_os("VISLOC_JOINT_BA_TIMING").is_some() {
                eprintln!(
                    "joint-ba iter {iteration}: build {build_s:.3}s solve {solve_s:.3}s backsub+apply {:.3}s cost {:.3}s",
                    (t_cost - t_rest).as_secs_f64(),
                    t_cost.elapsed().as_secs_f64()
                );
            }
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

    /// Residual and Jacobians (pose, landmark, camera block) of one monocular
    /// observation for the joint intrinsics system; `None` when the point is
    /// behind the camera (or not projectable through the tangential model).
    #[allow(clippy::type_complexity)]
    fn joint_mono_terms(
        &self,
        obs: &BaObservation,
        (fx, fy, cx, cy): (f64, f64, f64, f64),
        k_dim: usize,
        dist: Option<(f64, f64)>,
        tangential: Option<(f64, f64)>,
    ) -> Option<(Vector2<f64>, Matrix2x6<f64>, Matrix2x3<f64>, DMatrix<f64>)> {
        let pose = &self.poses[&obs.keyframe_id];
        let point = &self.landmarks[&obs.landmark_id];
        let xc = pose.transform_world_point(point);
        if xc.z <= 0.0 {
            return None;
        }
        let z_inv = 1.0 / xc.z;
        let x = xc.x * z_inv;
        let y = xc.y * z_inv;
        let r2 = x * x + y * y;
        // Radial distortion factor d = 1 + k1ﾂｷrﾂｲ + k2ﾂｷr竅ｴ and its radial
        // derivative helper g = k1 + 2ﾂｷk2ﾂｷrﾂｲ (d=1, g=0 when distortion-free).
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
        // J_ﾏ = diag(fx, fy) ﾂｷ D ﾂｷ 竏・x, y)/竏９_c, where the distortion Jacobian
        //   D = [[d + 2xﾂｲg, 2xyg], [2xyg, d + 2yﾂｲg]]  (= I when distortion-free)
        // and 竏・x, y)/竏９_c = (1/Z)ﾂｷ[[1, 0, -x], [0, 1, -y]].
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
        // 竏・predicted)/竏・ with K = (fx, fy, cx, cy[, k1, k2]) (2ﾃ楊_dim).
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
        // Tangential `(p1, p2)` (an `OpenCv` camera that carries them, or is
        // refining them): the radial-only terms above are replaced by the
        // full Brown-Conrady model. The residual is `Camera::project`'s and
        // the point Jacobian its analytic derivative; the camera columns
        // extend to `[.., k1, k2, p1, p2]` when `k_dim` is 8.
        let (residual, j_pose, j_lm, j_k) = match tangential {
            None => (residual, j_pose, j_lm, j_k),
            Some((p1, p2)) => {
                let Some((predicted, j_pi)) = self.camera.project_with_point_jacobian(&xc) else {
                    return None;
                };
                let residual = predicted - obs.xy;
                let j_pose: Matrix2x6<f64> = j_pi * dx_dxi;
                let j_lm: Matrix2x3<f64> = j_pi * r_mat;
                let xy2 = 2.0 * x * y;
                let xd = x * d + p1 * xy2 + p2 * (r2 + 2.0 * x * x);
                let yd = y * d + p1 * (r2 + 2.0 * y * y) + p2 * xy2;
                let mut j_k = DMatrix::<f64>::zeros(2, k_dim);
                j_k[(0, 0)] = xd;
                j_k[(0, 2)] = 1.0;
                j_k[(1, 1)] = yd;
                j_k[(1, 3)] = 1.0;
                if k_dim >= 6 {
                    j_k[(0, 4)] = fx * x * r2;
                    j_k[(0, 5)] = fx * x * r2 * r2;
                    j_k[(1, 4)] = fy * y * r2;
                    j_k[(1, 5)] = fy * y * r2 * r2;
                }
                if k_dim == 8 {
                    j_k[(0, 6)] = fx * xy2;
                    j_k[(0, 7)] = fx * (r2 + 2.0 * x * x);
                    j_k[(1, 6)] = fy * (r2 + 2.0 * y * y);
                    j_k[(1, 7)] = fy * xy2;
                }
                (residual, j_pose, j_lm, j_k)
            }
        };
        Some((residual, j_pose, j_lm, j_k))
    }

    /// Parallel counterpart of the monocular half of
    /// [`Self::build_joint_intrinsics_system`] for the sparse solve.
    ///
    /// Landmarks are processed in fixed-size waves: inside a wave each
    /// landmark's observations are linearized on a rayon worker (its
    /// `H_ll`, `b_l` and cross blocks are private to it), and the pose /
    /// intrinsics contributions are then added serially in observation order,
    /// so the result does not depend on the thread count. Observations of
    /// fixed landmarks only feed the pose / intrinsics blocks.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn build_joint_sparse_parallel(
        &self,
        pose_index: &BTreeMap<u64, usize>,
        landmark_index: &BTreeMap<u64, usize>,
        kernel: &RobustKernel,
        intr: (f64, f64, f64, f64),
        k_dim: usize,
        dist: Option<(f64, f64)>,
        tangential: Option<(f64, f64)>,
    ) -> (JointCameraHessian, DVector<f64>, Vec<JointLandmarkBlock>) {
        use rayon::prelude::*;
        let p_count = pose_index.len();
        let k_off = p_count * 6;
        // Observations grouped by free-landmark slot (fixed landmarks last).
        let mut by_lm: Vec<Vec<usize>> = vec![Vec::new(); landmark_index.len() + 1];
        for (i, obs) in self.observations.iter().enumerate() {
            let slot = landmark_index
                .get(&obs.landmark_id)
                .copied()
                .unwrap_or(landmark_index.len());
            by_lm[slot].push(i);
        }
        let ids: Vec<u64> = landmark_index.keys().copied().collect();
        let mut pose_diag = vec![Matrix6::<f64>::zeros(); p_count];
        let mut pose_k = vec![DMatrix::<f64>::zeros(6, k_dim); p_count];
        let mut kk = DMatrix::<f64>::zeros(k_dim, k_dim);
        let mut b_c = DVector::<f64>::zeros(k_off + k_dim);
        let mut lm_blocks: Vec<JointLandmarkBlock> = Vec::with_capacity(ids.len());

        // One observation's pose / intrinsics share, scattered serially.
        struct PoseShare {
            pose: Option<usize>,
            hpp: Matrix6<f64>,
            bp: Vector6<f64>,
            hpk: DMatrix<f64>,
            hkk: DMatrix<f64>,
            bk: DVector<f64>,
        }
        let linearize = |slot: usize| -> (Option<JointLandmarkBlock>, Vec<PoseShare>) {
            let free = slot < ids.len();
            let mut block = free.then(|| JointLandmarkBlock {
                id: ids[slot],
                h_ll: Matrix3::zeros(),
                b_l: Vector3::zeros(),
                cross: BTreeMap::new(),
            });
            let mut shares = Vec::with_capacity(by_lm[slot].len());
            for &i in &by_lm[slot] {
                let obs = &self.observations[i];
                let Some((residual, j_pose, j_lm, j_k)) =
                    self.joint_mono_terms(obs, intr, k_dim, dist, tangential)
                else {
                    continue;
                };
                let w = kernel.weight(residual.norm_squared());
                let pose = pose_index.get(&obs.keyframe_id).copied();
                let res2 = DVector::from_column_slice(&[residual.x, residual.y]);
                let jkt = j_k.transpose();
                let jp_dyn = DMatrix::from_iterator(2, 6, j_pose.iter().copied());
                shares.push(PoseShare {
                    pose,
                    hpp: w * (j_pose.transpose() * j_pose),
                    bp: w * (j_pose.transpose() * residual),
                    hpk: w * (jp_dyn.transpose() * &j_k),
                    hkk: w * (&jkt * &j_k),
                    bk: w * (&jkt * &res2),
                });
                if let Some(block) = block.as_mut() {
                    block.h_ll += w * (j_lm.transpose() * j_lm);
                    block.b_l += w * (j_lm.transpose() * residual);
                    if let Some(p) = pose {
                        let cr = w * (j_pose.transpose() * j_lm);
                        add_cross(
                            &mut block.cross,
                            p * 6,
                            6,
                            &DMatrix::from_fn(6, 3, |r, c| cr[(r, c)]),
                        );
                    }
                    let jl_dyn = DMatrix::from_iterator(2, 3, j_lm.iter().copied());
                    add_cross(&mut block.cross, k_off, k_dim, &(w * (&jkt * &jl_dyn)));
                }
            }
            (block, shares)
        };
        let slots: Vec<usize> = (0..by_lm.len()).collect();
        for wave in slots.chunks(JOINT_WAVE_LANDMARKS) {
            let done: Vec<_> = wave.par_iter().map(|&slot| linearize(slot)).collect();
            for (block, shares) in done {
                for sh in shares {
                    kk += &sh.hkk;
                    for j in 0..k_dim {
                        b_c[k_off + j] += sh.bk[j];
                    }
                    if let Some(p) = sh.pose {
                        pose_diag[p] += sh.hpp;
                        for r in 0..6 {
                            b_c[p * 6 + r] += sh.bp[r];
                        }
                        pose_k[p] += &sh.hpk;
                    }
                }
                if let Some(block) = block {
                    lm_blocks.push(block);
                }
            }
        }
        (
            JointCameraHessian::Sparse {
                pose_diag,
                pose_k,
                kk,
            },
            b_c,
            lm_blocks,
        )
    }

    /// Assemble the raw (un-damped) joint normal equations for
    /// [`Self::optimize_joint_intrinsics`]: the camera-block Hessian `H_cc`
    /// (poses then the 4 intrinsics) and gradient `b_c`, plus per-landmark
    /// `{H_ll, b_l, cross}` blocks where `cross` maps each touching camera-block
    /// column-start to `J盞_cam ﾂｷ J_lm`. Mirrors `build_normal_equations`'
    /// reprojection Jacobians, extended with the intrinsics columns
    /// `J_K = 竏・predicted)/竏・fx, fy, cx, cy)`.
    #[allow(clippy::too_many_arguments)]
    fn build_joint_intrinsics_system(
        &self,
        pose_index: &BTreeMap<u64, usize>,
        landmark_index: &BTreeMap<u64, usize>,
        kernel: &RobustKernel,
        k_dim: usize,
        dist: Option<(f64, f64)>,
        tangential: Option<(f64, f64)>,
        sparse: bool,
    ) -> (
        usize,
        JointCameraHessian,
        DVector<f64>,
        Vec<JointLandmarkBlock>,
    ) {
        let intrinsics = self.intrinsics().expect("pinhole checked by caller");
        let (fx, fy, cx, cy) = intrinsics;
        let p_count = pose_index.len();
        let k_off = p_count * 6;
        let cam_dim = k_off + k_dim;
        if sparse && self.stereo_observations.is_empty() {
            let (h, b, lms) = self.build_joint_sparse_parallel(
                pose_index,
                landmark_index,
                kernel,
                (fx, fy, cx, cy),
                k_dim,
                dist,
                tangential,
            );
            return (cam_dim, h, b, lms);
        }
        let mut h_cc = if sparse {
            JointCameraHessian::Sparse {
                pose_diag: vec![Matrix6::zeros(); p_count],
                pose_k: vec![DMatrix::zeros(6, k_dim); p_count],
                kk: DMatrix::zeros(k_dim, k_dim),
            }
        } else {
            JointCameraHessian::Dense(DMatrix::<f64>::zeros(cam_dim, cam_dim))
        };
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

        // Accumulate a cameraﾃ幼amera block (rows_a ﾃ・cols_b) at (row_start, col_start).
        let mut add_cc = |rs: usize, cs: usize, blk: &DMatrix<f64>| h_cc.add(rs, cs, k_off, blk);

        // Monocular observations.
        for obs in &self.observations {
            let Some((residual, j_pose, j_lm, j_k)) =
                self.joint_mono_terms(obs, (fx, fy, cx, cy), k_dim, dist, tangential)
            else {
                continue;
            };

            let s = residual.x * residual.x + residual.y * residual.y;
            let w = kernel.weight(s);
            let i_pose = pose_index.get(&obs.keyframe_id).copied();
            let i_lm = landmark_index.get(&obs.landmark_id).copied();

            // Dynamic-sized residual / pose for the K-coupled products.
            let res2 = DVector::from_column_slice(&[residual.x, residual.y]);
            let jkt = j_k.transpose(); // k_dimﾃ・

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
                let hpk = w * (jp_dyn.transpose() * &j_k); // 6ﾃ楊_dim
                add_cc(p * 6, k_off, &hpk);
                add_cc(k_off, p * 6, &hpk.transpose());
            }
            if let Some(l) = i_lm {
                lm_blocks[l].h_ll += w * (j_lm.transpose() * j_lm);
                lm_blocks[l].b_l += w * (j_lm.transpose() * residual);
                if let Some(p) = i_pose {
                    let cr = w * (j_pose.transpose() * j_lm); // 6ﾃ・
                    add_cross(
                        &mut lm_blocks[l].cross,
                        p * 6,
                        6,
                        &DMatrix::from_fn(6, 3, |r, c| cr[(r, c)]),
                    );
                }
                let jl_dyn = DMatrix::from_iterator(2, 3, j_lm.iter().copied());
                let crk = w * (&jkt * &jl_dyn); // k_dimﾃ・
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
                        // u_r = fxﾂｷ(X竏鍛)/Z + cx, so 竏Ｖ_r/竏Ｇx = (X竏鍛)/Z, 竏Ｖ_r/竏Ｄx = 1.
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
}

/// Landmarks per parallel wave of the sparse joint system build / elimination
/// (bounds the per-wave buffers; the result does not depend on it).
const JOINT_WAVE_LANDMARKS: usize = 8192;

/// The reduced camera system of [`BundleAdjustment::optimize_joint_intrinsics`]
/// before landmark elimination: dense, or (for `LinearSolver::Sparse`) as its
/// structure — 6×6 pose diagonal blocks, a `6 × k` pose–intrinsics coupling per
/// pose, and the `k × k` intrinsics block (raw observations never couple two
/// poses directly; that only happens through landmark elimination).
pub(super) enum JointCameraHessian {
    Dense(DMatrix<f64>),
    Sparse {
        pose_diag: Vec<Matrix6<f64>>,
        pose_k: Vec<DMatrix<f64>>,
        kk: DMatrix<f64>,
    },
}

impl JointCameraHessian {
    /// Add `blk` at `(rs, cs)` of the full `(6 P + k)` camera matrix. The
    /// sparse form keeps only the lower arrow: the intrinsics-row copy of a
    /// pose–intrinsics block (`rs == k_off`, `cs < k_off`) is implied.
    fn add(&mut self, rs: usize, cs: usize, k_off: usize, blk: &DMatrix<f64>) {
        match self {
            Self::Dense(h) => {
                for r in 0..blk.nrows() {
                    for c in 0..blk.ncols() {
                        h[(rs + r, cs + c)] += blk[(r, c)];
                    }
                }
            }
            Self::Sparse {
                pose_diag,
                pose_k,
                kk,
            } => match (rs < k_off, cs < k_off) {
                (true, true) => {
                    debug_assert_eq!(rs, cs, "raw joint system has no pose-pose blocks");
                    let d = &mut pose_diag[rs / 6];
                    for r in 0..6 {
                        for c in 0..6 {
                            d[(r, c)] += blk[(r, c)];
                        }
                    }
                }
                (true, false) => pose_k[rs / 6] += blk,
                (false, true) => {}
                (false, false) => *kk += blk,
            },
        }
    }
}

/// One damped Levenberg–Marquardt step of the joint pose + intrinsics bundle
/// adjustment, solved without ever forming the dense camera matrix.
///
/// After eliminating the landmarks, the camera system is an arrowhead
/// `[[A, B], [Bᵀ, C]]`: `A` is block-sparse over the poses (a 6×6 block per
/// pair of poses that share a landmark), `B` is the dense `6 P × k`
/// pose–intrinsics coupling and `C` the `k × k` intrinsics block. `A` is
/// factored once with the block Cholesky against the `k + 1` right-hand sides
/// `[b_p | B]`, giving `y = A⁻¹ b_p` and `Z = A⁻¹ B`; the intrinsics step solves
/// the small Schur complement `(C − Bᵀ Z) δk = b_k − Bᵀ y` (with `shared_focal`,
/// `fy` folded into `fx` there, exactly as the dense path folds it), and the
/// poses follow as `δp = y − Z δk`. Memory and time scale with the covisibility
/// pattern instead of `(6 P)²` / `(6 P)³`.
///
/// Returns the full `(6 P + k)` camera step (or `Err` when a block is not
/// positive-definite or the intrinsics block is singular, which the caller treats like the dense solve's failure)
/// and the per-landmark damped `H_ll⁻¹` used for back-substitution.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn solve_joint_sparse(
    pose_diag: Vec<Matrix6<f64>>,
    pose_k: Vec<DMatrix<f64>>,
    mut kk: DMatrix<f64>,
    b_c: &DVector<f64>,
    lm_blocks: &[JointLandmarkBlock],
    lambda: f64,
    shared_focal: bool,
    cache: &mut Option<crate::block_cholesky::BlockSymbolic>,
) -> (Result<DVector<f64>, ()>, Vec<Option<Matrix3<f64>>>) {
    let p_count = pose_diag.len();
    let k_off = p_count * 6;
    let k_dim = kk.nrows();
    // Lower block columns of A (row >= column), starting from the damped
    // diagonal; B and C as dense blocks.
    let mut columns: Vec<BTreeMap<usize, Matrix6<f64>>> = pose_diag
        .into_iter()
        .enumerate()
        .map(|(p, mut d)| {
            for i in 0..6 {
                d[(i, i)] += lambda;
            }
            BTreeMap::from([(p, d)])
        })
        .collect();
    let mut pose_k = pose_k;
    for i in 0..k_dim {
        kk[(i, i)] += lambda;
    }
    // Landmark elimination, in waves: each landmark's Schur products are
    // computed on a rayon worker, then the 6×6 updates are bucketed by pose
    // column and applied one column per worker (pose–intrinsics, intrinsics
    // and gradient updates serially), always in landmark order, so the sums
    // do not depend on the thread count.
    use rayon::prelude::*;
    struct LandmarkUpdate {
        inv: Option<Matrix3<f64>>,
        rhs: Vec<(usize, DVector<f64>)>,
        pose_pose: Vec<(usize, usize, Matrix6<f64>)>,
        pose_k: Vec<(usize, DMatrix<f64>)>,
        kk: Option<DMatrix<f64>>,
    }
    let eliminate = |lm: &JointLandmarkBlock| -> LandmarkUpdate {
        let mut h_ll = lm.h_ll;
        if lambda > 0.0 {
            h_ll[(0, 0)] += lambda;
            h_ll[(1, 1)] += lambda;
            h_ll[(2, 2)] += lambda;
        }
        let inv = h_ll.try_inverse();
        let mut up = LandmarkUpdate {
            inv,
            rhs: Vec::new(),
            pose_pose: Vec::new(),
            pose_k: Vec::new(),
            kk: None,
        };
        let Some(inv) = inv else { return up };
        for (&cs_a, a) in &lm.cross {
            let ah = a * inv;
            up.rhs.push((cs_a, &ah * lm.b_l));
            for (&cs_b, b) in &lm.cross {
                match (cs_a < k_off, cs_b < k_off) {
                    (true, true) if cs_a >= cs_b => {
                        let block = &ah * b.transpose();
                        up.pose_pose.push((
                            cs_b / 6,
                            cs_a / 6,
                            Matrix6::from_iterator(block.iter().copied()),
                        ));
                    }
                    (true, false) => up.pose_k.push((cs_a / 6, &ah * b.transpose())),
                    (false, false) => up.kk = Some(&ah * b.transpose()),
                    _ => {}
                }
            }
        }
        up
    };
    let mut b_reduced = -b_c;
    let mut h_ll_inv_cache: Vec<Option<Matrix3<f64>>> = Vec::with_capacity(lm_blocks.len());
    for wave in lm_blocks.chunks(JOINT_WAVE_LANDMARKS) {
        let updates: Vec<LandmarkUpdate> = wave.par_iter().map(eliminate).collect();
        let mut buckets: Vec<Vec<(usize, Matrix6<f64>)>> = vec![Vec::new(); p_count];
        for up in updates {
            h_ll_inv_cache.push(up.inv);
            for (cs, v) in up.rhs {
                for r in 0..v.nrows() {
                    b_reduced[cs + r] += v[r];
                }
            }
            for (col, row, m) in up.pose_pose {
                buckets[col].push((row, m));
            }
            for (p, m) in up.pose_k {
                pose_k[p] -= m;
            }
            if let Some(m) = up.kk {
                kk -= m;
            }
        }
        columns
            .par_iter_mut()
            .zip(buckets)
            .for_each(|(column, bucket)| {
                for (row, m) in bucket {
                    *column.entry(row).or_insert_with(Matrix6::zeros) -= m;
                }
            });
    }

    // [b_p | B] -> [y | Z] = A⁻¹ [b_p | B].
    let (y, z) = if p_count > 0 {
        let mut rhs = DMatrix::<f64>::zeros(k_off, 1 + k_dim);
        for i in 0..k_off {
            rhs[(i, 0)] = b_reduced[i];
        }
        for (p, blk) in pose_k.iter().enumerate() {
            for r in 0..6 {
                for c in 0..k_dim {
                    rhs[(p * 6 + r, 1 + c)] = blk[(r, c)];
                }
            }
        }
        match crate::block_cholesky::solve_spd_blocks6_cached(cache, columns, &rhs) {
            Ok(x) => (x.column(0).into_owned(), x.columns(1, k_dim).into_owned()),
            Err(()) => return (Err(()), h_ll_inv_cache),
        }
    } else {
        (DVector::zeros(0), DMatrix::zeros(0, k_dim))
    };
    // Bᵀ Z and Bᵀ y without materializing the dense B.
    let mut bt_z = DMatrix::<f64>::zeros(k_dim, k_dim);
    let mut bt_y = DVector::<f64>::zeros(k_dim);
    for (p, blk) in pose_k.iter().enumerate() {
        let zp = z.rows(p * 6, 6);
        bt_z += blk.transpose() * zp;
        bt_y += blk.transpose() * y.rows(p * 6, 6);
    }
    let s_k = &kk - bt_z;
    let r_k = b_reduced.rows(k_off, k_dim) - bt_y;
    let delta_k = if shared_focal {
        // T folds fy (index 1) into fx (index 0): S' = Tᵀ S T, r' = Tᵀ r.
        let map = |i: usize| if i == 0 || i == 1 { 0 } else { i - 1 };
        let mut s_red = DMatrix::<f64>::zeros(k_dim - 1, k_dim - 1);
        let mut r_red = DVector::<f64>::zeros(k_dim - 1);
        for i in 0..k_dim {
            r_red[map(i)] += r_k[i];
            for j in 0..k_dim {
                s_red[(map(i), map(j))] += s_k[(i, j)];
            }
        }
        solve_normal_equations(&s_red, &r_red)
            .map(|d| DVector::<f64>::from_fn(k_dim, |i, _| d[map(i)]))
    } else {
        solve_normal_equations(&s_k, &r_k)
    };
    let delta_k = match delta_k {
        Ok(d) => d,
        Err(_) => return (Err(()), h_ll_inv_cache),
    };
    let delta_p = &y - &z * &delta_k;
    let mut delta = DVector::<f64>::zeros(k_off + k_dim);
    delta.rows_mut(0, k_off).copy_from(&delta_p);
    delta.rows_mut(k_off, k_dim).copy_from(&delta_k);
    (Ok(delta), h_ll_inv_cache)
}
