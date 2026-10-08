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
            let (cam_dim_n, h_cc, b_c, lm_blocks) = self.build_joint_intrinsics_system(
                &pose_index,
                &landmark_index,
                &kernel,
                k_dim,
                dist,
                tangential,
            );
            debug_assert_eq!(cam_dim_n, cam_dim);

            // Damped Schur reduction (Levenberg Iﾂｷﾎｻ on both the camera and the
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
    /// column-start to `J盞_cam ﾂｷ J_lm`. Mirrors `build_normal_equations`'
    /// reprojection Jacobians, extended with the intrinsics columns
    /// `J_K = 竏・predicted)/竏・fx, fy, cx, cy)`.
    fn build_joint_intrinsics_system(
        &self,
        pose_index: &BTreeMap<u64, usize>,
        landmark_index: &BTreeMap<u64, usize>,
        kernel: &RobustKernel,
        k_dim: usize,
        dist: Option<(f64, f64)>,
        tangential: Option<(f64, f64)>,
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

        // Accumulate a cameraﾃ幼amera block (rows_a ﾃ・cols_b) at (row_start, col_start).
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
                    let Some((predicted, j_pi)) = self.camera.project_with_point_jacobian(&xc)
                    else {
                        continue;
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
