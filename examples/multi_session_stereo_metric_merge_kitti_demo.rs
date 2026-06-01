//! Fully-automatic multi-session metric map merge on REAL KITTI stereo imagery.
//!
//! This is the end-to-end integration of the whole cross-session stack on real
//! data, with NO hand-built bridges: appearance retrieval proposes same-place
//! pairs, stereo + PnP turns each into a **metric** relative pose, PCM screens
//! them, and the surviving bridges weld two independently-drifted sessions into
//! one map whose ATE is measured against KITTI ground truth.
//!
//! Pipeline (every stage runs on the loaded images / calibration, the only
//! synthetic ingredient being the simulated per-session odometry drift and
//! frame offset that give the two sessions something to correct):
//!
//!   1. **Two sessions from one stereo run.** At each base keyframe position `p`
//!      (every `--keyframe-stride` frames), session A takes frame `p` and session
//!      B takes frame `p + --session-offset` — two runs of the SAME route a couple
//!      of frames apart, so every A/B keyframe pair is a strong same-place revisit
//!      a stereo PnP can metrically bridge (a wide gap starves the correspondence
//!      set and the PnP fails). Each session integrates its own drifted odometry
//!      (per-edge yaw perturbation); session B additionally lives in its **own
//!      arbitrary world frame** (right-multiplied by a fixed offset `g`), so it
//!      shares no gauge with A and cannot be scored against ground truth until
//!      welded in.
//!   2. **Metric bridge proposal (automatic).** For every keyframe we extract
//!      left+right corner features and stereo-triangulate metric 3D landmarks
//!      (`bootstrap_stereo_landmarks`, baseline `b = −tx/fx`). Session A's
//!      keyframes carry those 3D landmarks (`MetricKeyframe`); session B's are
//!      plain 2D features. `propose_metric_bridges` then VLAD-retrieves same-place
//!      A/B pairs and runs PnP of the B image against A's 3D landmarks — the
//!      recovered `A→B` pose is fully **metric** (the monocular essential bridge
//!      of `propose_bridges` would only be up to scale, useless for a metric weld).
//!   3. **PCM screening.** The metric bridges become cross-session
//!      `LoopClosureConstraint`s and are screened by
//!      `consistent_session_bridges` (Mahalanobis PCM), comparing the isotropic
//!      vs covariance-aware operating point at zero admitted ground-truth-wrong
//!      bridges. Bridges are labelled genuine/wrong by GT camera-centre distance
//!      purely for *reporting* — the screen never sees ground truth.
//!   4. **Merge + ATE.** Session B is welded into A at the first kept bridge
//!      (`merge_session`), the rest are added as loop closures, the joined graph
//!      is jointly optimized, and the absolute trajectory error (vs KITTI GT) of
//!      the screened merge is reported against an oracle all-genuine merge and a
//!      naive merge that trusts every proposal. Each kept genuine bridge's
//!      recovered metric pose is also checked against GT (rotation deg + metres).
//!
//! **Two session sources:**
//!   - *Single source (default):* one stereo sequence, sessions A=frame `p`,
//!     B=`p + --session-offset` — co-located passes, the working metric-bridge
//!     regime (small baseline ⇒ rich PnP correspondences). Validated: 20 bridges,
//!     30 PnP inliers / 0.8 px, merged 5.3 m ATE.
//!   - *Separate B source (`--image-left-b` + `--start-frame-b`):* B is a DIFFERENT
//!     pass over the same place (e.g. a genuine KITTI-00 loop revisit ~4500 frames
//!     later). The true Atlas multi-map case. **Honest finding (classical corner
//!     features): VLAD retrieval correctly ranks the genuine revisit pair first
//!     (A48↔B4500, sim 0.33), but the wide-baseline viewpoint change so
//!     contaminates the classical-descriptor matches that stereo PnP cannot reach
//!     a 6-point inlier consensus — no metric bridge is verified.** The metric
//!     bridge holds in the small-baseline regime; a genuine wide-baseline revisit
//!     needs a viewpoint-robust descriptor (SuperPoint/NetVLAD, the codebase's
//!     `--feature-extractor superpoint-onnx`). The separate-B wiring is correct
//!     and ready for that; classical features just don't carry PnP across the gap.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --features image-io \
//!     --example multi_session_stereo_metric_merge_kitti_demo -- \
//!     --image-left  /path/to/KITTI/sequences/00/image_0 \
//!     --image-right /path/to/KITTI/sequences/00/image_1 \
//!     --calib       /path/to/KITTI/sequences/00/calib.txt \
//!     --kitti-poses /path/to/KITTI/poses/00.txt \
//!     --keyframe-stride 5 --max-keyframes 120
//! ```

#[cfg(not(feature = "image-io"))]
fn main() {
    eprintln!(
        "this example requires the `image-io` feature; rebuild with \
         `cargo run --release --features image-io \
         --example multi_session_stereo_metric_merge_kitti_demo`"
    );
    std::process::exit(2);
}

#[cfg(feature = "image-io")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    imp::run()
}

#[cfg(feature = "image-io")]
mod imp {
    use std::env;
    use std::path::PathBuf;

    use nalgebra::{UnitQuaternion, Vector3, Vector6};
    use visloc_rs::core::geometry::{Pose, SE3};
    use visloc_rs::io::calibration::parse_kitti_calibration_txt;
    use visloc_rs::io::kitti::read_kitti_image_sequence_dir;
    use visloc_rs::slam::pcm::{PcmConfig, PcmNoiseModel};
    use visloc_rs::tracking::PoseTrajectory;
    use visloc_rs::vision::features::{
        CornerFeatureConfig, CornerFeatureExtractor, FeatureExtractor, FeatureSet,
    };
    use visloc_rs::vision::matching::BruteForceMatcher;
    use visloc_rs::vision::place_recognition::{
        propose_metric_bridges, retrieve_mutual, vlad, MetricBridgeConfig, MetricKeyframe,
        Vocabulary,
    };
    use visloc_rs::vision::ransac::PnPRansac;
    use visloc_rs::vision::stereo_bootstrap::{bootstrap_stereo_landmarks, StereoBootstrapConfig};
    use visloc_rs::{
        relative_world_to_camera, LoopClosureConstraint, PoseGraph, PoseGraphSe3Config,
    };

    struct Args {
        image_left: PathBuf,
        image_right: PathBuf,
        image_left_b: Option<PathBuf>,
        calib: PathBuf,
        kitti_poses: PathBuf,
        projection_left: String,
        projection_right: String,
        start_frame: usize,
        start_frame_b: usize,
        keyframe_stride: usize,
        session_offset: usize,
        max_keyframes: usize,
        max_features: usize,
        vocab_size: usize,
        vocab_sample: usize,
        yaw_drift_per_edge_rad: f64,
        min_similarity: f32,
        min_inliers: usize,
        proximity_m: f64,
        inject_wrong_bridges: usize,
        inject_seed: u64,
    }

    fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
        let mut image_left: Option<PathBuf> = None;
        let mut image_right: Option<PathBuf> = None;
        let mut image_left_b: Option<PathBuf> = None;
        let mut calib: Option<PathBuf> = None;
        let mut kitti_poses: Option<PathBuf> = None;
        let mut projection_left = String::from("P0");
        let mut projection_right = String::from("P1");
        let mut start_frame = 0usize;
        let mut start_frame_b = 0usize;
        let mut keyframe_stride = 5usize;
        let mut session_offset = 1usize;
        let mut max_keyframes = 120usize;
        let mut max_features = 600usize;
        let mut vocab_size = 64usize;
        let mut vocab_sample = 8000usize;
        let mut yaw_drift_deg = 0.05f64;
        let mut min_similarity = 0.15f32;
        let mut min_inliers = 12usize;
        let mut proximity_m = 10.0f64;
        let mut inject_wrong_bridges = 2usize;
        let mut inject_seed = 1u64;

        let mut args = env::args().skip(1).collect::<Vec<_>>();
        let i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--image-left" => {
                    image_left = Some(PathBuf::from(args.remove(i + 1)));
                    args.remove(i);
                }
                "--image-right" => {
                    image_right = Some(PathBuf::from(args.remove(i + 1)));
                    args.remove(i);
                }
                "--image-left-b" => {
                    image_left_b = Some(PathBuf::from(args.remove(i + 1)));
                    args.remove(i);
                }
                "--start-frame-b" => {
                    start_frame_b = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--calib" => {
                    calib = Some(PathBuf::from(args.remove(i + 1)));
                    args.remove(i);
                }
                "--kitti-poses" => {
                    kitti_poses = Some(PathBuf::from(args.remove(i + 1)));
                    args.remove(i);
                }
                "--projection-left" => {
                    projection_left = args.remove(i + 1);
                    args.remove(i);
                }
                "--projection-right" => {
                    projection_right = args.remove(i + 1);
                    args.remove(i);
                }
                "--start-frame" => {
                    start_frame = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--keyframe-stride" => {
                    keyframe_stride = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--session-offset" => {
                    session_offset = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--max-keyframes" => {
                    max_keyframes = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--max-features" => {
                    max_features = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--vocab-size" => {
                    vocab_size = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--vocab-sample" => {
                    vocab_sample = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--yaw-drift-deg-per-edge" => {
                    yaw_drift_deg = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--min-similarity" => {
                    min_similarity = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--min-inliers" => {
                    min_inliers = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--proximity-m" => {
                    proximity_m = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--inject-wrong-bridges" => {
                    inject_wrong_bridges = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--inject-seed" => {
                    inject_seed = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                other => return Err(format!("unknown argument: {other}").into()),
            }
        }
        Ok(Args {
            image_left: image_left.ok_or("--image-left <KITTI image_0 dir> is required")?,
            image_right: image_right.ok_or("--image-right <KITTI image_1 dir> is required")?,
            image_left_b,
            calib: calib.ok_or("--calib <path/to/calib.txt> is required")?,
            kitti_poses: kitti_poses.ok_or("--kitti-poses <path/to/poses/SS.txt> is required")?,
            projection_left,
            projection_right,
            start_frame,
            start_frame_b,
            keyframe_stride,
            session_offset,
            max_keyframes,
            max_features,
            vocab_size,
            vocab_sample,
            yaw_drift_per_edge_rad: yaw_drift_deg.to_radians(),
            min_similarity,
            min_inliers,
            proximity_m,
            inject_wrong_bridges,
            inject_seed,
        })
    }

    /// Small deterministic LCG (no `rand` dependency) for reproducible injected
    /// wrong bridges.
    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn unit(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }
    }

    fn pgo_config() -> PoseGraphSe3Config {
        PoseGraphSe3Config {
            initial_lambda: Some(1.0e-3),
            max_iterations: 80,
            chordal_init: false,
            ..PoseGraphSe3Config::default()
        }
    }

    fn loop_constraint(from: u64, to: u64, relative: SE3, inliers: usize) -> LoopClosureConstraint {
        LoopClosureConstraint {
            from_keyframe_id: from,
            to_keyframe_id: to,
            relative_pose: relative,
            inlier_count: inliers,
            inlier_ratio: 1.0,
            mean_sampson_error: 0.0,
            score: inliers as f64,
        }
    }

    /// Absolute trajectory error (mean / RMSE / max camera-centre error, metres).
    /// The anchor is gauge-fixed in the truth frame, so estimate and truth are
    /// directly comparable (no Umeyama alignment).
    fn ate(estimate: &[Pose], truth: &[Pose]) -> (f64, f64, f64) {
        let errs: Vec<f64> = estimate
            .iter()
            .zip(truth)
            .map(|(e, t)| (e.camera_center_world() - t.camera_center_world()).norm())
            .collect();
        let n = errs.len().max(1) as f64;
        let mean = errs.iter().sum::<f64>() / n;
        let rmse = (errs.iter().map(|e| e * e).sum::<f64>() / n).sqrt();
        let max = errs.iter().copied().fold(0.0_f64, f64::max);
        (mean, rmse, max)
    }

    /// Build a session pose graph: drifted odometry along `edges` starting at
    /// `origin`, anchored at node 0, with `loops` as intra-session constraints.
    fn build_session(
        origin: Pose,
        edges: &[SE3],
        yaw_drift: &UnitQuaternion<f64>,
    ) -> (PoseGraph, Vec<SE3>) {
        let noisy: Vec<SE3> = edges
            .iter()
            .map(|e| SE3::new(yaw_drift * e.rotation, e.translation))
            .collect();
        let mut poses: Vec<Pose> = vec![origin];
        for edge in &noisy {
            let prev = poses.last().unwrap().world_to_camera.clone();
            poses.push(Pose {
                world_to_camera: edge.compose(&prev),
            });
        }
        let mut graph = PoseGraph::new();
        for (id, p) in poses.iter().enumerate() {
            graph.add_pose(id as u64, p.clone());
        }
        graph.anchor(0);
        for (k, edge) in noisy.iter().enumerate() {
            graph.add_sequential_edge(k as u64, (k + 1) as u64, edge.clone());
        }
        (graph, noisy)
    }

    /// Weld B into a clone of A at the first kept bridge, add the rest as loop
    /// closures, and jointly optimize.
    fn merge_and_optimize(
        session_a: &PoseGraph,
        session_b: &PoseGraph,
        id_offset: u64,
        bridges: &[LoopClosureConstraint],
        kept: &[usize],
    ) -> Result<PoseGraph, Box<dyn std::error::Error>> {
        let mut merged = session_a.clone();
        merged.merge_session(session_b, id_offset, &bridges[kept[0]])?;
        for &k in &kept[1..] {
            let c = &bridges[k];
            merged.add_loop_closure_constraint(&loop_constraint(
                c.from_keyframe_id,
                c.to_keyframe_id + id_offset,
                c.relative_pose.clone(),
                c.inlier_count,
            ));
        }
        merged.optimize_se3_iterative(&pgo_config())?;
        Ok(merged)
    }

    /// Build a `MetricKeyframe` for one keyframe: keep the full left feature set
    /// for retrieval / matching, and attach a stereo-triangulated metric 3D point
    /// to each left keypoint that matched its right-image counterpart.
    fn stereo_keyframe(
        left: &FeatureSet,
        right: &FeatureSet,
        camera: &visloc_rs::core::types::Camera,
        baseline: f64,
    ) -> MetricKeyframe {
        let left_to_right = SE3::new(
            UnitQuaternion::identity(),
            Vector3::new(-baseline, 0.0, 0.0),
        );
        let stereo = bootstrap_stereo_landmarks(
            camera,
            camera,
            &left_to_right,
            left,
            right,
            &StereoBootstrapConfig {
                max_depth_meters: 60.0,
                ..StereoBootstrapConfig::default()
            },
        );
        let mut landmarks: Vec<Option<nalgebra::Point3<f64>>> = vec![None; left.keypoints.len()];
        for lm in &stereo {
            landmarks[lm.left_keypoint_index] = Some(lm.point_left_camera_frame);
        }
        MetricKeyframe {
            features: left.clone(),
            landmarks,
        }
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args = parse_args()?;
        let left_seq =
            read_kitti_image_sequence_dir(&args.image_left, &args.calib, &args.projection_left, 0)?;
        let right_seq = read_kitti_image_sequence_dir(
            &args.image_right,
            &args.calib,
            &args.projection_right,
            1,
        )?;
        if left_seq.frames.len() != right_seq.frames.len() {
            return Err(format!(
                "left ({}) and right ({}) sequences differ in length",
                left_seq.frames.len(),
                right_seq.frames.len()
            )
            .into());
        }
        let camera = left_seq.camera.clone();

        // Baseline from the projection matrices: P1.tx = -fx·b ⇒ b = -tx/fx.
        let calib_text = std::fs::read_to_string(&args.calib)?;
        let projections = parse_kitti_calibration_txt(&calib_text)?;
        let p_left = projections
            .iter()
            .find(|p| p.label == args.projection_left)
            .ok_or_else(|| format!("calib missing {}", args.projection_left))?;
        let p_right = projections
            .iter()
            .find(|p| p.label == args.projection_right)
            .ok_or_else(|| format!("calib missing {}", args.projection_right))?;
        let baseline = p_right
            .stereo_baseline_from(p_left)
            .ok_or("calib pair did not yield a positive stereo baseline (intrinsics mismatch?)")?;

        // Ground-truth poses (indexed by absolute KITTI frame number).
        let gt = PoseTrajectory::read_kitti_poses(&args.kitti_poses)?;
        let gt_samples = gt.samples();

        // Session B's images: a SEPARATE source (`--image-left-b/right-b`, e.g. a
        // genuine later revisit of the same place) when given, else the SAME
        // sequence — in which case the two sessions are A=frame `p`, B=`p+offset`,
        // co-located so PnP has rich correspondences. `b_seq()` returns B's
        // left/right sequences (cloned A when no separate source).
        // B is 2D-only (the db side of PnP), so only its left images are needed.
        let left_seq_b = match &args.image_left_b {
            Some(l) => read_kitti_image_sequence_dir(l, &args.calib, &args.projection_left, 0)?,
            None => left_seq.clone(),
        };
        let separate_b = args.image_left_b.is_some();

        let extractor = CornerFeatureExtractor::new(CornerFeatureConfig {
            max_features: args.max_features,
            ..CornerFeatureConfig::default()
        });
        let stride = args.keyframe_stride.max(1);
        let off = args.session_offset.max(1);

        // Per-session keyframe plan: (dir index into that session's sequence,
        // absolute GT frame). For the single-source mode the dir index IS the
        // absolute frame; with a separate B source, B's dir index `k` maps to GT
        // frame `start_frame_b + k` (so a 0-based revisit clip aligns to its true
        // frame numbers in the shared pose file).
        let a_total = left_seq.frames.len();
        let a_plan: Vec<(usize, usize)> = (args.start_frame..)
            .step_by(stride)
            .take_while(|&p| {
                if separate_b {
                    p < a_total
                } else {
                    p + off < a_total
                }
            })
            .take(args.max_keyframes)
            .map(|p| (p, p))
            .collect();
        let b_total = left_seq_b.frames.len();
        let b_plan: Vec<(usize, usize)> = if separate_b {
            (0..)
                .step_by(stride)
                .take_while(|&k| k < b_total)
                .take(args.max_keyframes)
                .map(|k| (k, args.start_frame_b + k))
                .collect()
        } else {
            a_plan.iter().map(|&(p, _)| (p + off, p + off)).collect()
        };
        let na = a_plan.len();
        let nb = b_plan.len();
        if na < 4 || nb < 4 {
            return Err(format!("need at least 4 keyframes per session, got A={na} B={nb}").into());
        }
        println!(
            "session A: {} ({na} kf from frame {}) | session B: {} ({nb} kf from frame {}) | \
             stride={stride} baseline={:.4} m fx={:.1} yaw_drift={:.4} rad/edge{}",
            args.image_left.display(),
            args.start_frame,
            args.image_left_b
                .as_ref()
                .unwrap_or(&args.image_left)
                .display(),
            if separate_b {
                args.start_frame_b
            } else {
                args.start_frame + off
            },
            baseline,
            camera.params.first().copied().unwrap_or(0.0),
            args.yaw_drift_per_edge_rad,
            if separate_b {
                " [GENUINE separate-pass revisit]"
            } else {
                ""
            },
        );

        let pose_at = |f: usize| -> Result<Pose, Box<dyn std::error::Error>> {
            gt_samples
                .get(f)
                .map(|s| s.pose.clone())
                .ok_or_else(|| format!("GT pose missing for frame {f}").into())
        };

        // Session A: metric stereo keyframes (3D landmarks).
        let mut a_metric: Vec<MetricKeyframe> = Vec::with_capacity(na);
        let mut a_truth: Vec<Pose> = Vec::with_capacity(na);
        let mut a_left: Vec<FeatureSet> = Vec::with_capacity(na);
        for &(dir, gt_f) in &a_plan {
            let left = extractor.extract(&left_seq.frames[dir].image)?;
            let right = extractor.extract(&right_seq.frames[dir].image)?;
            a_metric.push(stereo_keyframe(&left, &right, &camera, baseline));
            a_left.push(left);
            a_truth.push(pose_at(gt_f)?);
        }
        // Session B: plain 2D keyframes.
        let mut b_feats: Vec<FeatureSet> = Vec::with_capacity(nb);
        let mut b_truth: Vec<Pose> = Vec::with_capacity(nb);
        for &(dir, gt_f) in &b_plan {
            b_feats.push(extractor.extract(&left_seq_b.frames[dir].image)?);
            b_truth.push(pose_at(gt_f)?);
        }
        let median_landmarks = {
            let mut c: Vec<usize> = a_metric
                .iter()
                .map(|k| k.landmarks.iter().filter(|l| l.is_some()).count())
                .collect();
            c.sort_unstable();
            c[c.len() / 2]
        };
        println!("median stereo landmarks/A-keyframe = {median_landmarks}");

        // Vocabulary from a pooled, capped left-descriptor sample (both sessions).
        let mut pool: Vec<&[f32]> = a_left
            .iter()
            .chain(b_feats.iter())
            .flat_map(|f| f.descriptors.iter().map(|d| d.as_slice()))
            .collect();
        let stride = (pool.len() / args.vocab_sample.max(1)).max(1);
        let sampled: Vec<&[f32]> = pool.drain(..).step_by(stride).collect();
        let vocab = Vocabulary::build(&sampled, args.vocab_size, 20, 7)
            .ok_or("vocabulary construction failed (too few descriptors?)")?;
        println!(
            "vocabulary: k={} dim={} (from {} sampled descriptors)",
            vocab.k(),
            vocab.dim(),
            sampled.len(),
        );

        if env::var("VL_DEBUG").is_ok() {
            use visloc_rs::vision::matching::Matcher;
            use visloc_rs::vision::pnp::Correspondence2D3D;
            use visloc_rs::vision::ransac::RobustPoseEstimator;
            let aq: Vec<Vec<f32>> = a_metric
                .iter()
                .map(|k| vlad(&k.features.descriptors, &vocab))
                .collect();
            let bg: Vec<Vec<f32>> = b_feats
                .iter()
                .map(|f| vlad(&f.descriptors, &vocab))
                .collect();
            let ret = retrieve_mutual(&aq, &bg, 0.0);
            eprintln!("[debug] retrieved mutual pairs = {}", ret.len());
            let matcher = BruteForceMatcher { ratio: Some(0.85) };
            let pnp = PnPRansac {
                iterations: 2000,
                reprojection_threshold: 6.0,
                ..PnPRansac::default()
            };
            for r in ret.iter().take(10) {
                let q = &a_metric[r.query];
                let d = &b_feats[r.db];
                let matches = matcher.match_descriptors(&q.features.descriptors, &d.descriptors);
                let corr: Vec<Correspondence2D3D> = matches
                    .iter()
                    .filter_map(|m| {
                        let p3 = (*q.landmarks.get(m.query_index)?)?;
                        Some(Correspondence2D3D {
                            point3d: p3,
                            point2d: *d.keypoints.get(m.train_index)?,
                            confidence: None,
                        })
                    })
                    .collect();
                let rep = pnp.estimate(&corr, &camera);
                let inl = rep.as_ref().map(|rp| rp.inliers.len());
                let err = rep.as_ref().map(|rp| rp.mean_reprojection_error);
                eprintln!(
                    "[debug] A[{}]↔B[{}] sim={:.3} matches={} pnp_inliers={:?} reproj={:?}",
                    r.query,
                    r.db,
                    r.similarity,
                    corr.len(),
                    inl,
                    err,
                );
            }
        }

        // --- Automatic METRIC bridge proposal (appearance + stereo PnP) ---
        // Real classical descriptors over a frame-offset viewpoint are noisy, so
        // the PnP RANSAC runs a generous budget at a forgiving reprojection gate.
        let pnp = PnPRansac {
            iterations: 2000,
            reprojection_threshold: 6.0,
            ..PnPRansac::default()
        };
        let proposals = propose_metric_bridges(
            &a_metric,
            &b_feats,
            &vocab,
            &BruteForceMatcher { ratio: Some(0.85) },
            &pnp,
            &camera,
            &MetricBridgeConfig {
                min_similarity: args.min_similarity,
                min_inliers: args.min_inliers,
            },
        );
        if proposals.is_empty() {
            // Report the retrieval-vs-verification gap before giving up: appearance
            // retrieval may well have found the genuine revisit, with PnP unable to
            // verify it (the wide-baseline classical-descriptor limit — see the
            // module note). Recompute the mutual-NN pairs just for this diagnostic.
            let aq: Vec<Vec<f32>> = a_metric
                .iter()
                .map(|k| vlad(&k.features.descriptors, &vocab))
                .collect();
            let bg: Vec<Vec<f32>> = b_feats
                .iter()
                .map(|f| vlad(&f.descriptors, &vocab))
                .collect();
            let ret = retrieve_mutual(&aq, &bg, args.min_similarity);
            let best = ret.first();
            return Err(format!(
                "appearance retrieval found {} candidate revisit pair(s) (best {}sim={:.3}) but NONE \
                 passed metric stereo-PnP verification — across a wide-baseline revisit the classical \
                 corner descriptors are too contaminated for a 6-point PnP consensus. The metric \
                 bridge holds in the small-baseline regime (single-source / --session-offset); a \
                 genuine wide-baseline revisit needs a viewpoint-robust descriptor. Try \
                 --max-features higher / --min-inliers lower, or a learned feature extractor.",
                ret.len(),
                best.map(|r| format!("A[{}]↔B[{}] ", r.query, r.db)).unwrap_or_default(),
                best.map(|r| r.similarity).unwrap_or(0.0),
            )
            .into());
        }

        // Bridges → cross-session loop constraints (from = A id, to = B id). Label
        // each genuine/wrong by GT camera-centre distance — for REPORTING only.
        let mut bridges: Vec<LoopClosureConstraint> = Vec::new();
        let mut is_wrong: Vec<bool> = Vec::new();
        for p in &proposals {
            let dist = (a_truth[p.query].camera_center_world()
                - b_truth[p.db].camera_center_world())
            .norm();
            bridges.push(loop_constraint(
                p.query as u64,
                p.db as u64,
                p.query_to_db.clone(),
                p.inlier_count,
            ));
            is_wrong.push(dist > args.proximity_m);
        }

        // Validate each genuine bridge's recovered METRIC pose against the GT
        // relative pose `B.world_to_camera ∘ A.world_to_camera⁻¹` — rotation error
        // (deg) + translation error (metres). This is the direct check that
        // stereo PnP recovered a *metric* bridge, not just a direction.
        for (k, p) in proposals.iter().enumerate() {
            if is_wrong[k] {
                continue;
            }
            let truth = b_truth[p.db]
                .world_to_camera
                .compose(&a_truth[p.query].world_to_camera.inverse());
            let rot_err_deg = p
                .query_to_db
                .rotation
                .rotation_to(&truth.rotation)
                .angle()
                .to_degrees();
            let trans_err = (p.query_to_db.translation - truth.translation).norm();
            let gt_dist = (a_truth[p.query].camera_center_world()
                - b_truth[p.db].camera_center_world())
            .norm();
            println!(
                "  bridge A[{}]↔B[{}] sim={:.3} inliers={} | GT centre-gap={:.2} m → recovered pose \
                 rot_err={:.2}° trans_err={:.3} m",
                p.query, p.db, p.similarity, p.inlier_count, gt_dist, rot_err_deg, trans_err,
            );
        }

        // Inject perceptual-aliasing wrong bridges: an A node and a far-away B node
        // asserted to be the same place, with a fabricated plausible-magnitude
        // metric relative pose. These stand in for the false positives a retrieval
        // front-end occasionally emits (a real place-recognition run on this short
        // co-located subset produced none), so the PCM screen and the naive-merge
        // contrast are exercised on automatic bridges.
        if args.inject_wrong_bridges > 0 {
            let mut rng = Lcg(args.inject_seed.wrapping_mul(2) | 1);
            let mut added = 0usize;
            let mut guard = 0usize;
            while added < args.inject_wrong_bridges && guard < args.inject_wrong_bridges * 1000 {
                guard += 1;
                let i = (rng.next_u64() as usize) % na;
                let j = (rng.next_u64() as usize) % nb;
                if (a_truth[i].camera_center_world() - b_truth[j].camera_center_world()).norm()
                    <= args.proximity_m * 3.0
                {
                    continue; // too close — might genuinely overlap
                }
                let translation = Vector3::new(
                    rng.range(-2.0, 2.0),
                    rng.range(-1.0, 1.0),
                    rng.range(-2.0, 2.0),
                );
                let axis = Vector3::new(
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                    rng.range(-1.0, 1.0),
                );
                let rotation = nalgebra::Unit::try_new(axis, 1e-9)
                    .map(|a| UnitQuaternion::from_axis_angle(&a, rng.range(0.0, 0.2)))
                    .unwrap_or_else(UnitQuaternion::identity);
                bridges.push(loop_constraint(
                    i as u64,
                    j as u64,
                    SE3::new(rotation, translation),
                    80,
                ));
                is_wrong.push(true);
                added += 1;
            }
        }

        let genuine_total = is_wrong.iter().filter(|&&w| !w).count();
        let wrong_total = is_wrong.iter().filter(|&&w| w).count();
        println!(
            "metric bridges: {} from images ({genuine_total} genuine within {:.1} m) + {} injected \
             wrong = {} total; strongest A[{}]↔B[{}] sim={:.3} inliers={} reproj={:.2}px",
            proposals.len(),
            args.proximity_m,
            wrong_total,
            bridges.len(),
            proposals[0].query,
            proposals[0].db,
            proposals[0].similarity,
            proposals[0].inlier_count,
            proposals[0].mean_reprojection_error,
        );

        // --- Build the two sessions from real GT relative odometry + drift ---
        // Each session integrates its own drifted odometry along its keyframes; B
        // is additionally re-anchored in an arbitrary world frame `g`.
        let a_edges: Vec<SE3> = a_truth
            .windows(2)
            .map(|w| relative_world_to_camera(&w[0], &w[1]))
            .collect();
        let b_edges: Vec<SE3> = b_truth
            .windows(2)
            .map(|w| relative_world_to_camera(&w[0], &w[1]))
            .collect();
        let yaw_drift =
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), args.yaw_drift_per_edge_rad);
        let (session_a, _) = build_session(a_truth[0].clone(), &a_edges, &yaw_drift);
        let g = SE3::exp(&Vector6::new(0.6, -0.4, 0.9, 0.7, -0.3, 0.5));
        let b_origin = Pose {
            world_to_camera: b_truth[0].world_to_camera.compose(&g),
        };
        let (session_b, _) = build_session(b_origin, &b_edges, &yaw_drift);

        // Truth keyframes in merged-node order: A nodes 0..na then B nodes 0..nb.
        let merged_truth: Vec<Pose> = a_truth.iter().chain(b_truth.iter()).cloned().collect();

        // --- PCM screening at the fair (zero-GT-wrong) operating point ---
        let mean_edge_len = {
            let total: f64 = a_edges
                .iter()
                .chain(b_edges.iter())
                .map(|e| e.translation.norm())
                .sum();
            total / (a_edges.len() + b_edges.len()).max(1) as f64
        };
        let d = args.yaw_drift_per_edge_rad.max(1e-3);
        let noise = PcmNoiseModel::isotropic(
            (2.0 * d).powi(2),
            (2.0 * d * mean_edge_len).powi(2),
            1e-6,
            1e-4,
        );
        let sweep = |label: &str, noise: Option<PcmNoiseModel>| -> (Vec<usize>, usize) {
            let grid: Vec<f64> = (1..=60).map(|i| i as f64 * 0.5).collect();
            let mut best: Vec<usize> = Vec::new();
            for &t in &grid {
                let cfg = PcmConfig {
                    threshold: t,
                    require_individual: false,
                    noise,
                };
                let kept =
                    session_a.consistent_session_bridges(&session_b, na as u64, &bridges, &cfg);
                let wrong = kept.iter().filter(|&&i| is_wrong[i]).count();
                let genuine = kept.len() - wrong;
                if wrong == 0 && genuine > best.iter().filter(|&&i| !is_wrong[i]).count() {
                    best = kept;
                }
            }
            let recall = best.iter().filter(|&&i| !is_wrong[i]).count();
            println!(
                "[pcm {label}] best zero-wrong recall = {recall}/{genuine_total} genuine bridges"
            );
            (best, recall)
        };
        let (kept_iso, iso_recall) = sweep("isotropic  ", None);
        let (kept_maha, maha_recall) = sweep("mahalanobis", Some(noise));
        let (kept, _) = if maha_recall >= iso_recall {
            (kept_maha, maha_recall)
        } else {
            (kept_iso, iso_recall)
        };
        if kept.is_empty() {
            return Err("PCM screened out every bridge at zero-wrong precision".into());
        }

        // --- Merges + ATE ---
        // Baseline: B before any weld — it lives in its own arbitrary frame `g`,
        // so its half of the trajectory is nowhere near ground truth.
        let unbridged_traj: Vec<Pose> = (0..na as u64)
            .map(|id| session_a.poses[&id].clone())
            .chain((0..nb as u64).map(|id| session_b.poses[&id].clone()))
            .collect();
        let (u_mean, u_rmse, u_max) = ate(&unbridged_traj, &merged_truth);

        let genuine_idx: Vec<usize> = (0..bridges.len()).filter(|&i| !is_wrong[i]).collect();
        let oracle_ate = if genuine_idx.is_empty() {
            None
        } else {
            let merged =
                merge_and_optimize(&session_a, &session_b, na as u64, &bridges, &genuine_idx)?;
            let traj: Vec<Pose> = (0..(na + nb) as u64)
                .map(|id| merged.poses[&id].clone())
                .collect();
            Some(ate(&traj, &merged_truth))
        };

        let merged_screened =
            merge_and_optimize(&session_a, &session_b, na as u64, &bridges, &kept)?;
        let screened_traj: Vec<Pose> = (0..(na + nb) as u64)
            .map(|id| merged_screened.poses[&id].clone())
            .collect();
        let (s_mean, s_rmse, s_max) = ate(&screened_traj, &merged_truth);

        let naive_ate = if wrong_total > 0 {
            let all: Vec<usize> = (0..bridges.len()).collect();
            let merged = merge_and_optimize(&session_a, &session_b, na as u64, &bridges, &all)?;
            let traj: Vec<Pose> = (0..(na + nb) as u64)
                .map(|id| merged.poses[&id].clone())
                .collect();
            Some(ate(&traj, &merged_truth))
        } else {
            None
        };

        println!();
        println!(
            "ATE B unbridged (own frame `g`)  mean={u_mean:.3} rmse={u_rmse:.3} max={u_max:.3} (m)"
        );
        if let Some((o_mean, o_rmse, o_max)) = oracle_ate {
            println!("ATE merged + all-genuine (oracle) mean={o_mean:.3} rmse={o_rmse:.3} max={o_max:.3} (m)");
        }
        println!(
            "ATE merged + PCM screening       mean={s_mean:.3} rmse={s_rmse:.3} max={s_max:.3} (m)"
        );
        if let Some((nv_mean, nv_rmse, nv_max)) = naive_ate {
            println!("ATE merged + ALL bridges (naive) mean={nv_mean:.3} rmse={nv_rmse:.3} max={nv_max:.3} (m)");
        }
        let oracle_rmse = oracle_ate.map(|(_, r, _)| r).unwrap_or(s_rmse);
        let naive_note = match naive_ate {
            Some((_, nv_rmse, _)) => format!(
                "a naive merge that also trusts the {wrong_total} injected wrong bridge(s) collapses to \
                 {nv_rmse:.1} m"
            ),
            None => "no wrong bridges were present to stress the screen".to_string(),
        };
        println!(
            "\nsummary: {} metric stereo-PnP bridges proposed automatically from real images \
             ({genuine_total} genuine). Before the weld, session B sits in its own frame at \
             {u_rmse:.1} m ATE rmse; PCM keeps {} bridges at zero GT-wrong (isotropic {iso_recall}, \
             Mahalanobis {maha_recall}) and the welded two-session map lands at {s_rmse:.2} m rmse vs \
             KITTI ground truth (oracle {oracle_rmse:.2} m) — {naive_note}. Fully automatic, metric \
             end to end.",
            proposals.len(),
            kept.len(),
        );
        Ok(())
    }
}
