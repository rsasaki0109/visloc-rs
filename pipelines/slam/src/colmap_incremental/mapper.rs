//! Faithful (C2-scoped) port of `sfm/incremental_mapper.{h,cc}`'s
//! [`IncrementalMapper`].
//!
//! Ported from (commit `64805cb870b574a569dccc34918d95a2db2b2fee`, pinned by
//! `docs/colmap_rig_mapper_port_plan.md` §0):
//! `src/colmap/sfm/incremental_mapper.h/.cc` — `Options` (`.h:70-173`,
//! control defaults per plan §1.2/§0.1), `BeginReconstruction`/
//! `EndReconstruction` (`.cc:109-152`), `FindInitialImagePair`/
//! `RegisterInitialImagePair` (`.cc:154-231`), `RegisterNextImage`
//! (`.cc:233-490`, Path A), `RegisterNextGeneralFrame` (`.cc:492-669`,
//! Path B), `TriangulateImage`/`Retriangulate`/`CompleteTracks`/
//! `MergeTracks`/`CompleteAndMergeTracks` (`.cc:951-988`, thin delegates to
//! [`super::incremental_triangulator::IncrementalTriangulator`]),
//! `AdjustLocalBundle` (`.cc:990-1116`), `AdjustGlobalBundle`
//! (`.cc:1118-1246`), `IterativeLocalRefinement`/`IterativeGlobalRefinement`
//! (`.cc:1248-1317`), `FilterFrames`/`FilterPoints` (`.cc:1319-1361`),
//! `RegisterFrameEvent`/`DeRegisterFrameEvent` (`.cc:1423-1471`).
//!
//! ## Deviations (see also per-function doc comments)
//!
//! - **Path B (`register_next_general_frame`) is the path essentially
//!   always taken by this control configuration** — re-reading
//!   `incremental_mapper.cc:250-271`'s branch condition carefully: Path A
//!   only fires when a camera lacks `has_prior_focal_length` *and* has
//!   never been registered before, or has bogus parameters. This port's
//!   inputs are pre-calibrated (`has_prior_focal_length` modeled as always
//!   `true`, matching the control's `ba_refine_focal_length=0` /
//!   "no intrinsic refinement"), so `all_cameras_have_good_focal_length`
//!   is `true` from the very first registration attempt, for every
//!   registration of every 2-camera-rig frame — Path A is a structurally
//!   complete but, for this control run, **effectively dead** code path
//!   (ported per the task brief's explicit ask, exercised in unit tests,
//!   but not expected to fire on the real OpenLORIS runs in §8). This
//!   reverses an earlier (incorrect) reading in an intermediate draft of
//!   this port's planning notes.
//! - **`register_next_general_frame` now uses COLMAP's true 3-point
//!   polynomial GP3P by default** (`GeneralizedPnPRansac` with
//!   `minimal_solver: MinimalSolver::Gp3p`,
//!   `visloc_vision::pnp::gp3p::gp3p_solve`, ported from PoseLib's
//!   `gp3p`/`re3q3` — see that module's doc for the full algorithm and its
//!   own documented deviations), matching COLMAP's
//!   `EstimateGeneralizedAbsolutePose` -> `GP3PEstimator`
//!   (`estimators/generalized_pose.cc:131-190`,
//!   `estimators/solvers/generalized_absolute_pose.cc`) per-attempt
//!   correspondence floor of 3, not 6. This closes the former **C3 GP3P
//!   gap**: the pre-C3 6-point linear-DLT minimal solver
//!   (`GeneralizedDltPoseEstimator`) is kept selectable via
//!   `Options::pose_solver = PoseSolverBackend::Dlt6pt` for A/B parity
//!   checks only, per the task brief's "keep the existing 6-point DLT path
//!   selectable (option) for A/B" instruction.
//! - **`camera.has_prior_focal_length`** has no field in this port's
//!   `Camera` (`crates/core/src/types/camera.rs`) — modeled as always
//!   `true` (see above), a direct, documented consequence of this port's
//!   inputs always being pre-calibrated pinhole cameras from the rig
//!   manifest, never cameras with unknown intrinsics being estimated
//!   on-the-fly (out of scope: `ba_refine_focal_length=0`).

use std::collections::{BTreeMap, BTreeSet};

use visloc_core::geometry::SE3;
use visloc_vision::pnp::{
    Correspondence2D3D, GaussNewtonPoseRefiner, GeneralizedCameraRig,
    GeneralizedCorrespondence2D3D, GeneralizedPnPRansac, MinimalSolver, P3PGrunert, RigSensor,
};
use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};
use visloc_vision::two_view::CorrespondenceGraph;

use super::bundle_adjustment::{self, BundleAdjustmentConfig, BundleAdjustmentOptions, Gauge};
use super::incremental_triangulator::IncrementalTriangulator;
pub use super::incremental_triangulator::Options as TriangulatorOptions;
use super::mapper_impl::{self, InitGateOptions};
use super::observation_manager::{camera_has_bogus_params, ObservationManager};
use super::reconstruction::{Reconstruction, TrackElement};
use super::types::{CameraT, FrameT, ImageT, Point3DT, RigT, SensorT};

/// Which minimal solver [`IncrementalMapper::register_next_general_frame`]
/// (Path B) uses inside its `GeneralizedPnPRansac`. See module doc's C3
/// deviation note. `Gp3p` is the faithful COLMAP-matching default; `Dlt6pt`
/// is this port's pre-C3 6-point linear DLT path, kept selectable for A/B
/// parity checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PoseSolverBackend {
    #[default]
    Gp3p,
    Dlt6pt,
}

/// Port of `IncrementalMapper::Options` (`.h:70-173`), control-relevant
/// subset, defaulted to the pinned control's exact values (per this
/// module's scope — see module doc and `docs/colmap_rig_mapper_port_plan.md`
/// §0.1/§1.2) rather than COLMAP's own upstream defaults where they differ.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// C3: which minimal solver Path B's RANSAC uses. Not part of COLMAP's
    /// own `Options` surface (COLMAP always uses GP3P) — added purely as
    /// this port's A/B switch, see [`PoseSolverBackend`].
    pub pose_solver: PoseSolverBackend,
    pub init_min_num_inliers: usize,
    pub init_max_error: f64,
    pub init_max_forward_motion: f64,
    pub init_min_tri_angle_deg: f64,
    pub init_max_reg_trials: usize,
    pub abs_pose_max_error: f64,
    pub abs_pose_min_num_inliers: usize,
    pub abs_pose_min_inlier_ratio: f64,
    pub ba_local_num_images: usize,
    pub ba_local_min_tri_angle_deg: f64,
    pub filter_max_reproj_error: f64,
    pub filter_min_tri_angle_deg: f64,
    pub max_reg_trials: usize,
    pub min_focal_length_ratio: f64,
    pub max_focal_length_ratio: f64,
    pub max_extra_param: f64,
    pub random_seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            pose_solver: PoseSolverBackend::default(),
            init_min_num_inliers: 100,
            init_max_error: 4.0,
            init_max_forward_motion: 0.95,
            init_min_tri_angle_deg: 16.0,
            init_max_reg_trials: 2,
            abs_pose_max_error: 12.0,
            abs_pose_min_num_inliers: 8, // control override (§0.1)
            abs_pose_min_inlier_ratio: 0.25,
            ba_local_num_images: 6,
            ba_local_min_tri_angle_deg: 6.0,
            filter_max_reproj_error: 4.0,
            filter_min_tri_angle_deg: 1.5,
            max_reg_trials: 3,
            min_focal_length_ratio: 0.1,
            max_focal_length_ratio: 10.0,
            max_extra_param: 1.0,
            random_seed: 0, // control override (§0.1)
        }
    }
}

impl Options {
    fn init_gate(&self) -> InitGateOptions {
        InitGateOptions {
            init_min_num_inliers: self.init_min_num_inliers,
            init_max_error: self.init_max_error,
            init_max_forward_motion: self.init_max_forward_motion,
            init_min_tri_angle_deg: self.init_min_tri_angle_deg,
            init_max_reg_trials: self.init_max_reg_trials,
            random_seed: self.random_seed,
        }
    }
}

/// Port of `LocalBundleAdjustmentReport` (`.h:175-180`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LocalBundleAdjustmentReport {
    pub num_merged_observations: usize,
    pub num_completed_observations: usize,
    pub num_filtered_observations: usize,
    pub num_adjusted_observations: usize,
}

/// Port of `IncrementalMapper` (`.h:67-390`). See `observation_manager.rs`'s
/// module doc for the "no cached refs" convention this continues: every
/// method takes `recon`/`graph` explicitly rather than caching them.
#[derive(Debug, Clone, Default)]
pub struct IncrementalMapper {
    pub obs: ObservationManager,
    pub triangulator: IncrementalTriangulator,
    /// Human-readable run-log lines (registration path/inliers/BA events),
    /// not part of COLMAP's own API surface — added for this port's CLI
    /// (`examples/colmap_incremental_mapper.rs`) per the C2 task brief's
    /// "run log with per-registration lines" requirement.
    pub log: Vec<String>,
    /// Per-path registration attempt/outcome counters, added for the C2
    /// report's requirement to quantify how often each path fires and how
    /// often Path B is rejected purely for having fewer than
    /// `GeneralizedDltPoseEstimator::MINIMUM_CORRESPONDENCES` (6)
    /// correspondences — the C3 GP3P-minimal-solver gap this port
    /// documents (see `mapper.rs`'s module doc).
    pub path_a_attempts: usize,
    pub path_a_registered: usize,
    pub path_b_attempts: usize,
    pub path_b_registered: usize,
    pub path_b_rejected_lt6_corrs: usize,

    num_total_reg_images: usize,
    num_shared_reg_images: usize,
    init_num_reg_trials: BTreeMap<ImageT, usize>,
    init_image_pairs: BTreeSet<(ImageT, ImageT)>,
    num_reg_frames_per_rig: BTreeMap<RigT, usize>,
    num_reg_images_per_camera: BTreeMap<CameraT, usize>,
    num_registrations: BTreeMap<ImageT, usize>,
    num_reg_trials: BTreeMap<ImageT, usize>,
    filtered_frames: BTreeSet<FrameT>,
    existing_frame_ids: BTreeSet<FrameT>,
}

impl IncrementalMapper {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `msg` to `self.log` and immediately `eprintln!`s it too, so a
    /// long real-data run (see the C2 report's tier-1000/2500 timings) is
    /// observable live in the CLI's stderr instead of only after `run`
    /// returns — added after a real-data run went silent for many minutes
    /// with no visible progress before this port's development diagnosed
    /// it (a separate memory issue tracked in the C2 report, not a hang).
    fn log_event(&mut self, msg: String) {
        eprintln!("{msg}");
        self.log.push(msg);
    }

    pub fn num_total_reg_images(&self) -> usize {
        self.num_total_reg_images
    }
    pub fn num_shared_reg_images(&self) -> usize {
        self.num_shared_reg_images
    }
    pub fn filtered_frames(&self) -> &BTreeSet<FrameT> {
        &self.filtered_frames
    }
    pub fn reset_initialization_stats(&mut self) {
        self.init_image_pairs.clear();
        self.init_num_reg_trials.clear();
    }

    /// Port of `BeginReconstruction` (`.cc:109-133`).
    pub fn begin_reconstruction(&mut self, recon: &Reconstruction, graph: &CorrespondenceGraph) {
        self.obs = ObservationManager::new(recon, graph);
        self.obs.size_pyramids(recon);
        self.triangulator = IncrementalTriangulator::new();

        self.num_shared_reg_images = 0;
        self.num_reg_frames_per_rig.clear();
        self.num_reg_images_per_camera.clear();
        for &frame_id in recon.reg_frame_ids() {
            self.register_frame_event(recon, frame_id);
        }
        self.existing_frame_ids = recon.reg_frame_ids().iter().copied().collect();
        self.filtered_frames.clear();
        self.num_reg_trials.clear();
    }

    /// Port of `EndReconstruction` (`.cc:135-152`).
    pub fn end_reconstruction(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        discard: bool,
    ) {
        if discard {
            let ids: Vec<FrameT> = recon.reg_frame_ids().to_vec();
            for frame_id in ids {
                self.obs.deregister_frame(recon, graph, frame_id);
                self.deregister_frame_event(recon, frame_id);
            }
        }
    }

    fn register_frame_event(&mut self, recon: &Reconstruction, frame_id: FrameT) {
        let frame = recon.frame(frame_id);
        *self
            .num_reg_frames_per_rig
            .entry(frame.rig_id())
            .or_insert(0) += 1;
        let image_ids: Vec<ImageT> = frame.image_ids().collect();
        for image_id in image_ids {
            let camera_id = recon.image(image_id).camera_id;
            *self.num_reg_images_per_camera.entry(camera_id).or_insert(0) += 1;
            let count = self.num_registrations.entry(image_id).or_insert(0);
            *count += 1;
            if *count == 1 {
                self.num_total_reg_images += 1;
            } else {
                self.num_shared_reg_images += 1;
            }
        }
    }

    fn deregister_frame_event(&mut self, recon: &Reconstruction, frame_id: FrameT) {
        let frame = recon.frame(frame_id);
        if let Some(count) = self.num_reg_frames_per_rig.get_mut(&frame.rig_id()) {
            *count = count.saturating_sub(1);
        }
        let image_ids: Vec<ImageT> = frame.image_ids().collect();
        for image_id in image_ids {
            let camera_id = recon.image(image_id).camera_id;
            if let Some(count) = self.num_reg_images_per_camera.get_mut(&camera_id) {
                *count = count.saturating_sub(1);
            }
            let count = self.num_registrations.entry(image_id).or_insert(0);
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.num_total_reg_images = self.num_total_reg_images.saturating_sub(1);
            } else {
                self.num_shared_reg_images = self.num_shared_reg_images.saturating_sub(1);
            }
        }
    }

    /// Port of `FindInitialImagePair` (`.cc:154-180`) composed with
    /// `IncrementalMapperImpl::FindInitialImagePair` (`impl.cc:189-292`),
    /// sequential (see `mapper_impl.rs` module doc).
    pub fn find_initial_image_pair(
        &mut self,
        options: &Options,
        recon: &Reconstruction,
        graph: &CorrespondenceGraph,
    ) -> Option<(ImageT, ImageT, SE3)> {
        let gate = options.init_gate();
        let image_ids1 = mapper_impl::find_first_initial_image(
            &gate,
            recon,
            graph,
            &self.init_num_reg_trials,
            &self.num_registrations,
        );
        for image_id1 in image_ids1 {
            let image_ids2 = mapper_impl::find_second_initial_image(
                &gate,
                image_id1,
                recon,
                graph,
                &self.num_registrations,
            );
            for image_id2 in image_ids2 {
                let key = (image_id1.min(image_id2), image_id1.max(image_id2));
                if !self.init_image_pairs.insert(key) {
                    continue;
                }
                let estimate = mapper_impl::estimate_initial_two_view_geometry(
                    &gate, recon, graph, image_id1, image_id2,
                );
                if let Some(cam2_from_cam1) = estimate {
                    return Some((image_id1, image_id2, cam2_from_cam1));
                }
            }
        }
        None
    }

    /// Port of `EstimateInitialTwoViewGeometry` (`.cc:1473-1491`): evaluate
    /// one caller-provided pair (not used by this port's CLI, which always
    /// auto-searches, but kept for API completeness / unit tests).
    pub fn estimate_initial_two_view_geometry(
        &self,
        options: &Options,
        recon: &Reconstruction,
        graph: &CorrespondenceGraph,
        image_id1: ImageT,
        image_id2: ImageT,
    ) -> Option<SE3> {
        mapper_impl::estimate_initial_two_view_geometry(
            &options.init_gate(),
            recon,
            graph,
            image_id1,
            image_id2,
        )
    }

    /// Port of `RegisterInitialImagePair` (`.cc:194-231`).
    pub fn register_initial_image_pair(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        image_id1: ImageT,
        image_id2: ImageT,
        cam2_from_cam1: SE3,
    ) {
        assert_eq!(recon.num_reg_frames(), 0);

        *self.init_num_reg_trials.entry(image_id1).or_insert(0) += 1;
        *self.init_num_reg_trials.entry(image_id2).or_insert(0) += 1;
        *self.num_reg_trials.entry(image_id1).or_insert(0) += 1;
        *self.num_reg_trials.entry(image_id2).or_insert(0) += 1;
        self.init_image_pairs
            .insert((image_id1.min(image_id2), image_id1.max(image_id2)));

        let frame_id1 = recon.image(image_id1).frame_id;
        let camera_id1 = recon.image(image_id1).camera_id;
        let rig1 = recon.rig(recon.frame(frame_id1).rig_id()).clone();
        recon
            .frame_mut(frame_id1)
            .set_cam_from_world(&rig1, camera_id1, SE3::identity());

        let frame_id2 = recon.image(image_id2).frame_id;
        let camera_id2 = recon.image(image_id2).camera_id;
        let rig2 = recon.rig(recon.frame(frame_id2).rig_id()).clone();
        recon
            .frame_mut(frame_id2)
            .set_cam_from_world(&rig2, camera_id2, cam2_from_cam1);

        self.obs.register_frame(recon, graph, frame_id1);
        self.register_frame_event(recon, frame_id1);
        self.obs.register_frame(recon, graph, frame_id2);
        self.register_frame_event(recon, frame_id2);
        self.log_event(format!(
            "INIT_PAIR image1={} image2={} frame1={} frame2={}",
            image_id1, image_id2, frame_id1, frame_id2
        ));
    }

    /// Port of `RegisterNextImage`'s dispatcher (`.cc:233-271`). See module
    /// doc: Path B fires for essentially every call in this control config.
    pub fn register_next_image(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        image_id: ImageT,
    ) -> bool {
        let frame_id = recon.image(image_id).frame_id;
        let rig = recon.rig(recon.frame(frame_id).rig_id()).clone();

        if rig.num_sensors() > 1 {
            let frame_images: Vec<ImageT> = recon.frame(frame_id).image_ids().collect();
            let all_good = frame_images.iter().all(|&iid| {
                let camera = recon.camera(recon.image(iid).camera_id);
                !camera_has_bogus_params(
                    camera,
                    options.min_focal_length_ratio,
                    options.max_focal_length_ratio,
                    options.max_extra_param,
                )
            });
            if all_good {
                return self.register_next_general_frame(options, recon, graph, frame_id);
            }
        }

        self.register_next_image_path_a(options, recon, graph, image_id)
    }

    /// Port of the ordinary single-camera path (`.cc:273-489`).
    fn register_next_image_path_a(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        image_id: ImageT,
    ) -> bool {
        *self.num_reg_trials.entry(image_id).or_insert(0) += 1;
        self.path_a_attempts += 1;

        if self.obs.num_visible_points3d(image_id) < options.abs_pose_min_num_inliers {
            return false;
        }

        let mut tri_corrs: Vec<(usize, Point3DT)> = Vec::new();
        let mut correspondences: Vec<Correspondence2D3D> = Vec::new();
        let num_points2d = recon.image(image_id).num_points2d();
        for idx in 0..num_points2d {
            let mut seen: BTreeSet<Point3DT> = BTreeSet::new();
            for corr in graph.find_correspondences(image_id as usize, idx) {
                let corr_image_id = corr.image_id as ImageT;
                if !recon.is_image_registered(corr_image_id) {
                    continue;
                }
                let corr_image = recon.image(corr_image_id);
                let Some(point3d_id) = corr_image.points2d[corr.point2d_idx].point3d_id else {
                    continue;
                };
                if !seen.insert(point3d_id) {
                    continue;
                }
                let corr_camera = recon.camera(corr_image.camera_id);
                if camera_has_bogus_params(
                    corr_camera,
                    options.min_focal_length_ratio,
                    options.max_focal_length_ratio,
                    options.max_extra_param,
                ) {
                    continue;
                }
                tri_corrs.push((idx, point3d_id));
                correspondences.push(Correspondence2D3D {
                    point2d: recon.image(image_id).points2d[idx].xy,
                    point3d: recon.point3d(point3d_id).xyz,
                    confidence: None,
                });
            }
        }

        if correspondences.len() < options.abs_pose_min_num_inliers {
            return false;
        }

        let camera = recon.camera(recon.image(image_id).camera_id).clone();
        let ransac = PnPRansac::<P3PGrunert, GaussNewtonPoseRefiner> {
            pose_estimator: P3PGrunert,
            pose_refiner: Some(GaussNewtonPoseRefiner::default()),
            iterations: 128,
            reprojection_threshold: options.abs_pose_max_error,
            seed: options.random_seed,
            early_stop_min_iterations: 0,
            early_stop_inlier_ratio: None,
            confidence: None,
        };
        let Some(report) = ransac.estimate(&correspondences, &camera) else {
            return false;
        };
        if report.inliers.len() < options.abs_pose_min_num_inliers {
            return false;
        }
        if (report.inliers.len() as f64) / (correspondences.len() as f64)
            < options.abs_pose_min_inlier_ratio
        {
            return false;
        }

        self.path_a_registered += 1;
        let cam_from_world = report.pose.world_to_camera;
        let frame_id = recon.image(image_id).frame_id;
        let camera_id = recon.image(image_id).camera_id;
        let rig = recon.rig(recon.frame(frame_id).rig_id()).clone();
        recon
            .frame_mut(frame_id)
            .set_cam_from_world(&rig, camera_id, cam_from_world);

        self.obs.register_frame(recon, graph, frame_id);
        self.register_frame_event(recon, frame_id);
        self.log_event(format!(
            "REGISTER path=A frame={} image={} inliers={} correspondences={} num_reg_frames={} num_points3d={}",
            frame_id,
            image_id,
            report.inliers.len(),
            correspondences.len(),
            recon.num_reg_frames(),
            recon.num_points3d()
        ));

        for &inlier_idx in &report.inliers {
            let (point2d_idx, point3d_id) = tri_corrs[inlier_idx];
            if !recon.image(image_id).points2d[point2d_idx].has_point3d() {
                self.obs.add_observation(
                    recon,
                    graph,
                    point3d_id,
                    TrackElement {
                        image_id,
                        point2d_idx,
                    },
                );
                self.triangulator.add_modified_point3d(point3d_id);
            }
        }
        true
    }

    /// Port of `RegisterNextGeneralFrame` (`.cc:492-669`). See module doc:
    /// this is the path that actually fires for this control config, and
    /// its minimal solver is the C3 GP3P gap (documented deviation).
    fn register_next_general_frame(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        frame_id: FrameT,
    ) -> bool {
        self.path_b_attempts += 1;
        let rig = recon.rig(recon.frame(frame_id).rig_id()).clone();
        let frame_images: Vec<ImageT> = recon.frame(frame_id).image_ids().collect();

        let mut sensors: Vec<RigSensor> = Vec::with_capacity(frame_images.len());
        let mut sensor_index_of: BTreeMap<ImageT, usize> = BTreeMap::new();
        for &iid in &frame_images {
            let camera_id = recon.image(iid).camera_id;
            let camera = recon.camera(camera_id).clone();
            let sensor_id = SensorT::camera(camera_id);
            let sensor_from_rig = if rig.is_ref_sensor(sensor_id) {
                SE3::identity()
            } else {
                rig.sensor_from_rig(sensor_id)
            };
            sensor_index_of.insert(iid, sensors.len());
            sensors.push(RigSensor {
                camera,
                sensor_from_rig,
            });
            *self.num_reg_trials.entry(iid).or_insert(0) += 1;
        }
        let Some(gcam_rig) = GeneralizedCameraRig::new(sensors) else {
            return false;
        };

        let mut tri_corrs: Vec<(ImageT, usize, Point3DT)> = Vec::new();
        let mut gcorrs: Vec<GeneralizedCorrespondence2D3D> = Vec::new();
        for &iid in &frame_images {
            let sensor_idx = sensor_index_of[&iid];
            let num_points2d = recon.image(iid).num_points2d();
            for idx in 0..num_points2d {
                let mut seen: BTreeSet<Point3DT> = BTreeSet::new();
                for corr in graph.find_correspondences(iid as usize, idx) {
                    let corr_image_id = corr.image_id as ImageT;
                    if !recon.is_image_registered(corr_image_id) {
                        continue;
                    }
                    let corr_image = recon.image(corr_image_id);
                    let Some(point3d_id) = corr_image.points2d[corr.point2d_idx].point3d_id else {
                        continue;
                    };
                    if !seen.insert(point3d_id) {
                        continue;
                    }
                    let corr_camera = recon.camera(corr_image.camera_id);
                    if camera_has_bogus_params(
                        corr_camera,
                        options.min_focal_length_ratio,
                        options.max_focal_length_ratio,
                        options.max_extra_param,
                    ) {
                        continue;
                    }
                    tri_corrs.push((iid, idx, point3d_id));
                    gcorrs.push(GeneralizedCorrespondence2D3D {
                        sensor_index: sensor_idx,
                        point2d: recon.image(iid).points2d[idx].xy,
                        point3d: recon.point3d(point3d_id).xyz,
                        confidence: None,
                    });
                }
            }
        }

        // `GeneralizedDltPoseEstimator::MINIMUM_CORRESPONDENCES == 6` — kept
        // as a diagnostic counter for the A/B `Dlt6pt` path (see module doc
        // C3 deviation note); with the default `Gp3p` solver COLMAP's own
        // 3-correspondence floor applies instead (checked next, via
        // `abs_pose_min_num_inliers`, which the control already sets >= 3).
        if gcorrs.len() < 6 {
            self.path_b_rejected_lt6_corrs += 1;
        }
        if gcorrs.len() < options.abs_pose_min_num_inliers {
            return false;
        }

        let minimal_solver = match options.pose_solver {
            PoseSolverBackend::Gp3p => MinimalSolver::Gp3p,
            PoseSolverBackend::Dlt6pt => MinimalSolver::Dlt6pt,
        };
        let ransac = GeneralizedPnPRansac {
            minimal_solver,
            reprojection_threshold: options.abs_pose_max_error,
            seed: options.random_seed,
            ..GeneralizedPnPRansac::default()
        };
        let Some(report) = ransac.estimate(&gcam_rig, &gcorrs) else {
            return false;
        };
        if report.inliers.len() < options.abs_pose_min_num_inliers {
            return false;
        }
        if (report.inliers.len() as f64) / (gcorrs.len() as f64) < options.abs_pose_min_inlier_ratio
        {
            return false;
        }

        self.path_b_registered += 1;
        let rig_from_world = report.pose.world_to_camera;
        recon.frame_mut(frame_id).set_rig_from_world(rig_from_world);
        self.obs.register_frame(recon, graph, frame_id);
        self.register_frame_event(recon, frame_id);
        self.log_event(format!(
            "REGISTER path=B frame={} inliers={} correspondences={} num_reg_frames={} num_points3d={}",
            frame_id,
            report.inliers.len(),
            gcorrs.len(),
            recon.num_reg_frames(),
            recon.num_points3d()
        ));

        for &inlier_idx in &report.inliers {
            let (image_id, point2d_idx, point3d_id) = tri_corrs[inlier_idx];
            if !recon.image(image_id).points2d[point2d_idx].has_point3d() {
                self.obs.add_observation(
                    recon,
                    graph,
                    point3d_id,
                    TrackElement {
                        image_id,
                        point2d_idx,
                    },
                );
                self.triangulator.add_modified_point3d(point3d_id);
            }
        }
        true
    }

    // ---- Triangulation delegates (`.cc:951-988`) ---------------------

    pub fn triangulate_image(
        &mut self,
        tri_options: &TriangulatorOptions,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        image_id: ImageT,
    ) -> usize {
        self.triangulator
            .triangulate_image(tri_options, recon, graph, &mut self.obs, image_id)
    }

    pub fn retriangulate(
        &mut self,
        tri_options: &TriangulatorOptions,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
    ) -> usize {
        self.triangulator
            .retriangulate(tri_options, recon, graph, &mut self.obs)
    }

    pub fn complete_tracks(
        &mut self,
        tri_options: &TriangulatorOptions,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
    ) -> usize {
        self.triangulator
            .complete_all_tracks(tri_options, recon, graph, &mut self.obs)
    }

    pub fn merge_tracks(
        &mut self,
        tri_options: &TriangulatorOptions,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
    ) -> usize {
        self.triangulator
            .merge_all_tracks(tri_options, recon, graph, &mut self.obs)
    }

    /// Port of `CompleteAndMergeTracks` (`.cc:981-988`).
    pub fn complete_and_merge_tracks(
        &mut self,
        tri_options: &TriangulatorOptions,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
    ) -> usize {
        self.complete_tracks(tri_options, recon, graph)
            + self.merge_tracks(tri_options, recon, graph)
    }

    /// Port of `FindLocalBundle` (`.cc:1417-1421`, delegating to
    /// `mapper_impl.rs`).
    pub fn find_local_bundle(
        &self,
        options: &Options,
        recon: &Reconstruction,
        image_id: ImageT,
    ) -> Vec<ImageT> {
        mapper_impl::find_local_bundle(
            options.ba_local_num_images,
            options.ba_local_min_tri_angle_deg,
            image_id,
            recon,
        )
    }

    /// Port of `AdjustLocalBundle` (`.cc:990-1116`).
    #[allow(clippy::too_many_arguments)] // mirrors the COLMAP signature
    pub fn adjust_local_bundle(
        &mut self,
        options: &Options,
        ba_options: &BundleAdjustmentOptions,
        tri_options: &TriangulatorOptions,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        image_id: ImageT,
        point3d_ids: &BTreeSet<Point3DT>,
    ) -> LocalBundleAdjustmentReport {
        let mut report = LocalBundleAdjustmentReport::default();
        let local_bundle = self.find_local_bundle(options, recon, image_id);

        let mut config = BundleAdjustmentConfig::new();
        let mut config_image_ids: Vec<ImageT> = Vec::new();
        if !local_bundle.is_empty() {
            config.fix_gauge(Gauge::ThreePoints);

            let frame_id = recon.image(image_id).frame_id;
            config.add_frame(recon, frame_id);
            for &local_image_id in &local_bundle {
                let lf = recon.image(local_image_id).frame_id;
                config.add_frame(recon, lf);
            }
            config_image_ids = config.images().iter().copied().collect();

            // Deviation: COLMAP additionally fixes rig/camera parameter
            // blocks not fully represented in the local window
            // (`incremental_mapper.cc:1036-1065`) — moot here since
            // `sensor_from_rig` and camera intrinsics are never Ceres
            // parameter blocks in this port at all (control:
            // `ba_refine_sensor_from_rig=0`, no intrinsics refinement; see
            // `bundle_adjustment.rs` module doc). Nothing to fix.

            let mut variable_ids: Vec<Point3DT> = Vec::new();
            for &pid in point3d_ids {
                if !recon.exists_point3d(pid) {
                    continue;
                }
                let point3d = recon.point3d(pid);
                let has_error = point3d.error >= 0.0;
                const MAX_TRACK_LENGTH: usize = 15;
                if !has_error || point3d.track.len() <= MAX_TRACK_LENGTH {
                    config.add_variable_point(pid);
                    variable_ids.push(pid);
                }
            }

            let ba_ok = bundle_adjustment::solve(ba_options, &config, recon);
            report.num_adjusted_observations = config_image_ids
                .iter()
                .map(|&iid| recon.image(iid).num_points3d())
                .sum();
            self.log_event(format!(
                "BA local image={} images_in_window={} points={} ok={}",
                image_id,
                config_image_ids.len(),
                variable_ids.len(),
                ba_ok
            ));

            report.num_merged_observations = self.triangulator.merge_tracks(
                tri_options,
                recon,
                graph,
                &mut self.obs,
                &variable_ids,
            );
            report.num_completed_observations = self.triangulator.complete_tracks(
                tri_options,
                recon,
                graph,
                &mut self.obs,
                &variable_ids,
            );
            report.num_completed_observations += self.triangulator.complete_image(
                tri_options,
                recon,
                graph,
                &mut self.obs,
                image_id,
            );
        }

        report.num_filtered_observations = self.obs.filter_points3d_in_images(
            recon,
            graph,
            options.filter_max_reproj_error,
            options.filter_min_tri_angle_deg,
            &config_image_ids,
        );
        let point3d_ids_vec: Vec<Point3DT> = point3d_ids.iter().copied().collect();
        report.num_filtered_observations += self.obs.filter_points3d(
            recon,
            graph,
            options.filter_max_reproj_error,
            options.filter_min_tri_angle_deg,
            &point3d_ids_vec,
        );

        report
    }

    /// Port of `AdjustGlobalBundle` (`.cc:1118-1246`), pose-prior/redundant-
    /// point pruning paths omitted (`use_prior_position=false`,
    /// `ba_global_ignore_redundant_points3D=false` in this control).
    pub fn adjust_global_bundle(
        &mut self,
        ba_options: &BundleAdjustmentOptions,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
    ) -> bool {
        self.obs
            .filter_observations_with_negative_depth(recon, graph);

        let mut config = BundleAdjustmentConfig::new();
        for &frame_id in recon.reg_frame_ids() {
            config.add_frame(recon, frame_id);
        }
        if config.num_images() < 2 {
            return false;
        }
        config.fix_gauge(Gauge::TwoFramesFromWorld);
        let num_images = config.num_images();
        let num_points = recon.num_points3d();
        let ok = bundle_adjustment::solve(ba_options, &config, recon);
        self.log_event(format!(
            "BA global images={} points={} ok={}",
            num_images, num_points, ok
        ));
        ok
    }

    /// Port of `IterativeLocalRefinement` (`.cc:1248-1284`). Deviation: no
    /// robust-loss-then-trivial-loss switch between iterations (COLMAP
    /// alternates `SOFT_L1` then `TRIVIAL` — `bundle_adjustment.rs` already
    /// defaults to `RobustKernel::None`/trivial throughout, matching the
    /// control's steady-state loss exactly per §1.6, so there is nothing to
    /// switch away from).
    #[allow(clippy::too_many_arguments)] // mirrors the COLMAP signature
    /// Port of `IterativeLocalRefinement` (`.cc:1248-1284`). C2.7: mirrors
    /// `custom_ba_options.ceres->loss_function_type =
    /// CeresBundleAdjustmentOptions::LossFunctionType::TRIVIAL;`
    /// (`sfm/incremental_mapper.cc:1277-1281`, "Only use robust cost
    /// function for first iteration") — `ba_options` (the caller-supplied
    /// `BundleAdjustmentOptions::local()`, `LossFunction::SoftL1(1.0)` by
    /// default) is used as-is for the first iteration; every subsequent
    /// iteration of this same call downgrades to `LossFunction::Trivial`,
    /// exactly like COLMAP's per-call-site `custom_ba_options` copy.
    pub fn iterative_local_refinement(
        &mut self,
        max_num_refinements: usize,
        max_refinement_change: f64,
        options: &Options,
        ba_options: &BundleAdjustmentOptions,
        tri_options: &TriangulatorOptions,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        image_id: ImageT,
    ) {
        let mut custom_ba_options = *ba_options;
        for _ in 0..max_num_refinements {
            let modified: BTreeSet<Point3DT> =
                self.triangulator.modified_point3d_ids(recon).clone();
            let report = self.adjust_local_bundle(
                options,
                &custom_ba_options,
                tri_options,
                recon,
                graph,
                image_id,
                &modified,
            );
            let changed = if report.num_adjusted_observations == 0 {
                0.0
            } else {
                (report.num_merged_observations
                    + report.num_completed_observations
                    + report.num_filtered_observations) as f64
                    / report.num_adjusted_observations as f64
            };
            if changed < max_refinement_change {
                break;
            }
            // Only use robust cost function for first iteration
            // (`incremental_mapper.cc:1277-1281`).
            custom_ba_options.loss_function = super::bundle_adjustment::LossFunction::Trivial;
        }
        self.triangulator.clear_modified_point3d_ids();
    }

    /// Port of `IterativeGlobalRefinement` (`.cc:1286-1317`). `Normalize()`
    /// is not called — see `docs/colmap_rig_mapper_port_plan.md`'s Lead
    /// review discussion of `AlignReconstructionToOrigRigScales`: with
    /// `ba_refine_sensor_from_rig=0` (this control), that post-hoc rescale
    /// always recovers scale factor `1.0` exactly (the calibrated baseline
    /// never changes), so `Normalize()` (numerical-conditioning only, never
    /// metric-relevant) followed by that rescale is a no-op round trip for
    /// this control configuration; omitting both changes nothing about the
    /// final reconstruction.
    #[allow(clippy::too_many_arguments)] // mirrors the COLMAP signature
    pub fn iterative_global_refinement(
        &mut self,
        max_num_refinements: usize,
        max_refinement_change: f64,
        options: &Options,
        ba_options: &BundleAdjustmentOptions,
        tri_options: &TriangulatorOptions,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
    ) {
        self.complete_and_merge_tracks(tri_options, recon, graph);
        self.retriangulate(tri_options, recon, graph);
        for _ in 0..max_num_refinements {
            let num_observations: usize = recon.points3d().values().map(|p| p.track.len()).sum();
            self.adjust_global_bundle(ba_options, recon, graph);
            let mut num_changed = self.complete_and_merge_tracks(tri_options, recon, graph);
            num_changed += self.filter_points(options, recon, graph);
            let changed = if num_observations == 0 {
                0.0
            } else {
                num_changed as f64 / num_observations as f64
            };
            if changed < max_refinement_change {
                break;
            }
        }
        self.triangulator.clear_modified_point3d_ids();
    }

    /// Port of `FilterFrames` (`.cc:1319-1352`).
    pub fn filter_frames(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
    ) -> usize {
        const MIN_NUM_FRAMES: usize = 20;
        if recon.num_reg_frames() < MIN_NUM_FRAMES {
            return 0;
        }
        let filter_frame_ids = self.obs.find_frames_to_filter(
            recon,
            options.min_focal_length_ratio,
            options.max_focal_length_ratio,
            options.max_extra_param,
            1,
        );
        let mut num_filtered = 0;
        for frame_id in filter_frame_ids {
            self.obs.deregister_frame(recon, graph, frame_id);
            self.deregister_frame_event(recon, frame_id);
            self.filtered_frames.insert(frame_id);
            num_filtered += 1;
        }
        num_filtered
    }

    /// Port of `FilterPoints` (`.cc:1354-1361`).
    pub fn filter_points(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
    ) -> usize {
        self.obs.filter_all_points3d(
            recon,
            graph,
            options.filter_max_reproj_error,
            options.filter_min_tri_angle_deg,
        )
    }

    /// Port of `FindNextImages` (`.cc:182-192`, delegating to
    /// `mapper_impl.rs`); `structure_less` mode is not ported (see
    /// `pipeline.rs` module doc's deviation on
    /// `structure_less_registration_fallback`).
    pub fn find_next_images(&self, options: &Options, recon: &Reconstruction) -> Vec<ImageT> {
        mapper_impl::find_next_images(
            options.abs_pose_min_num_inliers,
            options.max_reg_trials,
            recon,
            &self.obs,
            &self.filtered_frames,
            &self.num_reg_trials,
        )
    }
}
