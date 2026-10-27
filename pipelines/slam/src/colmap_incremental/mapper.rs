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

use nalgebra::{Point2, Point3, Vector3};
use visloc_core::geometry::{reproject, SE3};
use visloc_vision::pnp::{
    estimate_gr6p_ransac_with_config, Correspondence2D3D, GaussNewtonPoseRefiner,
    GeneralizedCameraObservation, GeneralizedCameraRig, GeneralizedCorrespondence2D3D,
    GeneralizedPnPRansac, GeneralizedRelativeCorrespondence, GeneralizedRelativePoseRansacConfig,
    MinimalSolver, P3PGrunert, RigSensor,
};
use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};
use visloc_vision::two_view::CorrespondenceGraph;

use super::bundle_adjustment::{
    self, BundleAdjustmentConfig, BundleAdjustmentOptions, Gauge, LocalBaPointPolicy, LossFunction,
};
pub use super::incremental_triangulator::Options as TriangulatorOptions;
use super::incremental_triangulator::{estimate_triangulation_of_views, IncrementalTriangulator};
pub use super::mapper_impl::CandidateRound;
use super::mapper_impl::{self, InitGateOptions};
use super::observation_manager::{
    camera_has_bogus_params, image_cam_from_world, ObservationManager,
};
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
    /// COLMAP 4's `IncrementalPipelineOptions::structure_less_registration_fallback`:
    /// when no image registers from 2D-3D correspondences, try the
    /// candidates again from 2D-2D correspondences to registered images
    /// ([`IncrementalMapper::register_next_structure_less_image`]). Off by
    /// default here (the port's controls predate it); COLMAP 4 enables it.
    pub structure_less_registration_fallback: bool,
    /// Not in COLMAP: a structure-less registration's inlier matches that
    /// continue triangulated tracks must number at least this many, and
    /// the new pose must reproject at least [`STRUCTURE_LESS_MIN_VERIFIED_RATIO`]
    /// of their 3D points within `abs_pose_max_error`. 0 (the default)
    /// leaves the check out. Matches that nearly all go to one registered
    /// image fix the pose poorly, and RANSAC can settle on a rival pose
    /// almost as well supported. On an L4, ETH3D kicker's DSC_6487 came
    /// out 66° off that way, its 175 matches all but 17 to one image and 3
    /// continuing tracks.
    ///
    /// With the check on, a structure-less pose also needs
    /// [`STRUCTURE_LESS_MIN_PARALLAX_RATIO`] of its inliers to see their
    /// point with parallax, and a pose that fails the track check gets a
    /// second try with its distance from the matched cameras taken from
    /// the tracks (see `rescue_structure_less_pose`): views panned from one
    /// spot fix the direction to a new image but hardly its distance.
    pub structure_less_min_verified: usize,
}

/// See [`Options::structure_less_min_verified`].
pub const STRUCTURE_LESS_MIN_VERIFIED_RATIO: f64 = 0.7;

/// See [`Options::structure_less_min_verified`]: the share of a
/// structure-less registration's inliers that must see their point at
/// [`STRUCTURE_LESS_MIN_TRI_ANGLE_DEG`] or more.
pub const STRUCTURE_LESS_MIN_PARALLAX_RATIO: f64 = 0.5;

/// See [`STRUCTURE_LESS_MIN_PARALLAX_RATIO`] (COLMAP's default
/// `min_tri_angle`).
pub const STRUCTURE_LESS_MIN_TRI_ANGLE_DEG: f64 = 1.5;

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
            structure_less_registration_fallback: false,
            structure_less_min_verified: 0,
        }
    }
}

impl Options {
    /// Apply one `key=value` override (the numeric thresholds only), e.g.
    /// `abs_pose_min_num_inliers=30`. Errors name the offending key.
    pub fn apply_override(&mut self, kv: &str) -> Result<(), String> {
        let (key, value) = kv
            .split_once('=')
            .ok_or_else(|| format!("{kv}: expected key=value"))?;
        let num = || -> Result<f64, String> {
            value
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("{key}: {e}"))
        };
        match key.trim() {
            "init_min_num_inliers" => self.init_min_num_inliers = num()? as usize,
            "init_max_error" => self.init_max_error = num()?,
            "init_max_forward_motion" => self.init_max_forward_motion = num()?,
            "init_min_tri_angle_deg" => self.init_min_tri_angle_deg = num()?,
            "init_max_reg_trials" => self.init_max_reg_trials = num()? as usize,
            "abs_pose_max_error" => self.abs_pose_max_error = num()?,
            "abs_pose_min_num_inliers" => self.abs_pose_min_num_inliers = num()? as usize,
            "abs_pose_min_inlier_ratio" => self.abs_pose_min_inlier_ratio = num()?,
            "ba_local_num_images" => self.ba_local_num_images = num()? as usize,
            "ba_local_min_tri_angle_deg" => self.ba_local_min_tri_angle_deg = num()?,
            "filter_max_reproj_error" => self.filter_max_reproj_error = num()?,
            "filter_min_tri_angle_deg" => self.filter_min_tri_angle_deg = num()?,
            "max_reg_trials" => self.max_reg_trials = num()? as usize,
            "structure_less_registration_fallback" => {
                self.structure_less_registration_fallback = num()? != 0.0;
            }
            "structure_less_min_verified" => {
                self.structure_less_min_verified = num()? as usize;
            }
            _ => return Err(format!("unknown key {key}")),
        }
        Ok(())
    }

    const fn init_gate(&self) -> InitGateOptions {
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
    num_structure_less_reg_trials: BTreeMap<ImageT, usize>,
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

    pub const fn num_total_reg_images(&self) -> usize {
        self.num_total_reg_images
    }
    pub const fn num_shared_reg_images(&self) -> usize {
        self.num_shared_reg_images
    }
    pub const fn filtered_frames(&self) -> &BTreeSet<FrameT> {
        &self.filtered_frames
    }
    pub fn reset_initialization_stats(&mut self) {
        self.init_image_pairs.clear();
        self.init_num_reg_trials.clear();
    }

    /// Port of `BeginReconstruction` (`.cc:109-133`).
    pub fn begin_reconstruction(&mut self, recon: &Reconstruction, graph: &CorrespondenceGraph) {
        self.obs = ObservationManager::new(recon, graph);
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
        self.num_structure_less_reg_trials.clear();
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
            // Same visiting order and `init_image_pairs` bookkeeping as the
            // sequential loop, but the (pure, seeded) two-view estimates of
            // the next few untried pairs are evaluated in parallel; the first
            // success in order wins, so the chosen pair is unchanged.
            let untried: Vec<ImageT> = image_ids2
                .into_iter()
                .filter(|&image_id2| {
                    !self
                        .init_image_pairs
                        .contains(&(image_id1.min(image_id2), image_id1.max(image_id2)))
                })
                .collect();
            for chunk in untried.chunks(32) {
                use rayon::prelude::*;
                let estimates: Vec<Option<SE3>> = chunk
                    .par_iter()
                    .map(|&image_id2| {
                        mapper_impl::estimate_initial_two_view_geometry(
                            &gate, recon, graph, image_id1, image_id2,
                        )
                    })
                    .collect();
                for (&image_id2, estimate) in chunk.iter().zip(estimates) {
                    self.init_image_pairs
                        .insert((image_id1.min(image_id2), image_id1.max(image_id2)));
                    if let Some(cam2_from_cam1) = estimate {
                        return Some((image_id1, image_id2, cam2_from_cam1));
                    }
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

    /// Port of COLMAP 4's `RegisterNextStructureLessImage`: register
    /// `image_id` from its 2D-2D correspondences to registered images when
    /// it sees too few 3D points for [`Self::register_next_image`].
    ///
    /// The registered images form one generalized camera in world
    /// coordinates (each correspondence a ray from its camera centre), the
    /// new image a central camera; GR6P RANSAC between the two gives
    /// `cam_from_world` with metric scale. Inlier correspondences then
    /// continue an existing track, or triangulate a new one, and a
    /// Cauchy-loss BA of the image and its new points refines the pose.
    ///
    /// Deviations from COLMAP 4: GR6P only (no 5+1 solver, grouped sampler
    /// or LO step), a bounded trial count, and the angular epipolar
    /// residual with the pixel threshold divided by the focal length
    /// instead of the tangent Sampson error. Single-camera frames only.
    pub fn register_next_structure_less_image(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        image_id: ImageT,
    ) -> bool {
        if recon.num_reg_images() < 2 {
            return false;
        }
        *self
            .num_structure_less_reg_trials
            .entry(image_id)
            .or_insert(0) += 1;
        let frame_id = recon.image(image_id).frame_id;
        if recon.frame(frame_id).image_ids().count() != 1 {
            return false;
        }

        // A 2D-2D correspondence constrains one degree of freedom, a 2D-3D
        // one two: twice the inliers.
        let min_num_inliers = 2 * options.abs_pose_min_num_inliers;
        if self.obs.num_visible_correspondences(image_id) < min_num_inliers {
            return false;
        }

        let camera = recon.camera(recon.image(image_id).camera_id).clone();
        // (point2d_idx, world image, world point2d_idx) and the GR6P rays.
        let mut corrs: Vec<(usize, ImageT, usize)> = Vec::new();
        let mut gr_corrs: Vec<GeneralizedRelativeCorrespondence> = Vec::new();
        let mut world_poses: BTreeMap<ImageT, Option<(SE3, Vector3<f64>)>> = BTreeMap::new();
        let num_points2d = recon.image(image_id).num_points2d();
        for idx in 0..num_points2d {
            let xy = recon.image(image_id).points2d[idx].xy;
            let Some(query_ray) = pixel_ray(&camera, &xy) else {
                continue;
            };
            for corr in graph.find_correspondences(image_id as usize, idx) {
                let world_id = corr.image_id as ImageT;
                if !recon.is_image_registered(world_id) {
                    continue;
                }
                let pose = world_poses.entry(world_id).or_insert_with(|| {
                    let world_camera = recon.camera(recon.image(world_id).camera_id);
                    if camera_has_bogus_params(
                        world_camera,
                        options.min_focal_length_ratio,
                        options.max_focal_length_ratio,
                        options.max_extra_param,
                    ) {
                        return None;
                    }
                    let cam_from_world = image_cam_from_world(recon, world_id);
                    let centre = -(cam_from_world.rotation.inverse() * cam_from_world.translation);
                    Some((cam_from_world, centre))
                });
                let Some((cam_from_world, centre)) = pose else {
                    continue;
                };
                let world_image = recon.image(world_id);
                let world_camera = recon.camera(world_image.camera_id);
                let Some(world_ray_cam) = pixel_ray(
                    world_camera,
                    &world_image.points2d[corr.point2d_idx as usize].xy,
                ) else {
                    continue;
                };
                let world_ray = cam_from_world.rotation.inverse() * world_ray_cam;
                let (Ok(world_obs), Ok(query_obs)) = (
                    GeneralizedCameraObservation::new(*centre, world_ray),
                    GeneralizedCameraObservation::new(Vector3::zeros(), query_ray),
                ) else {
                    continue;
                };
                gr_corrs.push(GeneralizedRelativeCorrespondence {
                    rig1: world_obs,
                    rig2: query_obs,
                });
                corrs.push((idx, world_id, corr.point2d_idx as usize));
            }
        }
        if corrs.len() < min_num_inliers {
            return false;
        }
        // All rays from one centre (`IsPanoramicCameraSelection`): no scale.
        let centres: Vec<Vector3<f64>> = world_poses.values().flatten().map(|(_, c)| *c).collect();
        if centres.iter().all(|c| (c - centres[0]).norm() <= 1e-9) {
            return false;
        }

        let pixels_per_radian = camera.pixels_per_radian().unwrap_or(1.0);
        let cfg = GeneralizedRelativePoseRansacConfig {
            max_iterations: 2000,
            min_iterations: 100,
            confidence: 0.9999,
            // Epipolar errors gate inliers more tightly (`0.5 *`).
            inlier_angular_threshold: 0.5 * options.abs_pose_max_error / pixels_per_radian,
            seed: options.random_seed,
            min_inliers: 6,
        };
        let Some(report) = estimate_gr6p_ransac_with_config(&gr_corrs, &cfg)
            .ok()
            .flatten()
        else {
            return false;
        };
        if report.inliers.len() < min_num_inliers {
            return false;
        }
        // Inliers per registered image, most first, and those that continue
        // a triangulated track (diagnostics for the log line below).
        let mut per_world: BTreeMap<ImageT, usize> = BTreeMap::new();
        let mut continuing = 0;
        for &i in &report.inliers {
            let (_, world_id, world_idx) = corrs[i];
            *per_world.entry(world_id).or_insert(0) += 1;
            if recon.image(world_id).points2d[world_idx]
                .point3d_id
                .is_some()
            {
                continuing += 1;
            }
        }
        let mut per_world: Vec<usize> = per_world.into_values().collect();
        per_world.sort_unstable_by(|a, b| b.cmp(a));

        let mut cam_from_world = SE3::new(report.pose.rotation, report.pose.translation);
        // Inlier 2D points whose match continues a triangulated track, one
        // 3D point each: the pose must reproject them within
        // `abs_pose_max_error` (an independent check of the pose).
        let mut checked: BTreeSet<usize> = BTreeSet::new();
        let mut tracks: Vec<(Point2<f64>, Vector3<f64>)> = Vec::new();
        for &i in &report.inliers {
            let (idx, world_id, world_idx) = corrs[i];
            let Some(point3d_id) = recon.image(world_id).points2d[world_idx].point3d_id else {
                continue;
            };
            if !checked.insert(idx) {
                continue;
            }
            tracks.push((
                recon.image(image_id).points2d[idx].xy,
                recon.point3d(point3d_id).xyz.coords,
            ));
        }
        let passes = |verified: usize| {
            options.structure_less_min_verified == 0
                || (verified >= options.structure_less_min_verified
                    && (verified as f64) >= STRUCTURE_LESS_MIN_VERIFIED_RATIO * tracks.len() as f64)
        };
        let mut verified = count_reprojected(
            &camera,
            &cam_from_world,
            &tracks,
            options.abs_pose_max_error,
        );
        let epipolar: Vec<(Vector3<f64>, Vector3<f64>, Vector3<f64>)> = report
            .inliers
            .iter()
            .map(|&i| {
                (
                    gr_corrs[i].rig2.bearing,
                    gr_corrs[i].rig1.origin,
                    gr_corrs[i].rig1.bearing,
                )
            })
            .collect();
        // Matches to a view taken from the same spot hold wherever the camera
        // is, so most inliers must also see their point with parallax.
        let parallax = |pose: &SE3| {
            epipolar_support(
                pose,
                &epipolar,
                options.abs_pose_max_error,
                pixels_per_radian,
            )
            .1
        };
        let has_parallax = |count: usize| {
            options.structure_less_min_verified == 0
                || (count as f64) >= STRUCTURE_LESS_MIN_PARALLAX_RATIO * epipolar.len() as f64
        };
        let mut with_parallax = parallax(&cam_from_world);
        if !passes(verified) || !has_parallax(with_parallax) {
            let rescued = if passes(verified) {
                None
            } else {
                rescue_structure_less_pose(
                    &camera,
                    &cam_from_world,
                    &epipolar,
                    &tracks,
                    options.abs_pose_max_error,
                    pixels_per_radian,
                )
                .map(|pose| {
                    let count =
                        count_reprojected(&camera, &pose, &tracks, options.abs_pose_max_error);
                    let with_parallax = parallax(&pose);
                    (pose, count, with_parallax)
                })
                .filter(|&(_, count, with_parallax)| passes(count) && has_parallax(with_parallax))
            };
            let Some((pose, count, rescued_parallax)) = rescued else {
                self.log_event(format!(
                    "REJECT path=SL image={} inliers={} inliers_per_image={:?} verified={}/{} parallax={}",
                    image_id,
                    report.inliers.len(),
                    &per_world[..per_world.len().min(4)],
                    verified,
                    tracks.len(),
                    with_parallax
                ));
                return false;
            };
            self.log_event(format!(
                "RESCUE path=SL image={} verified={}/{} -> {}/{} parallax={} -> {}",
                image_id,
                verified,
                tracks.len(),
                count,
                tracks.len(),
                with_parallax,
                rescued_parallax
            ));
            cam_from_world = pose;
            verified = count;
            with_parallax = rescued_parallax;
        }
        let camera_id = recon.image(image_id).camera_id;
        let rig = recon.rig(recon.frame(frame_id).rig_id()).clone();
        recon
            .frame_mut(frame_id)
            .set_cam_from_world(&rig, camera_id, cam_from_world);
        self.obs.register_frame(recon, graph, frame_id);
        self.register_frame_event(recon, frame_id);
        self.log_event(format!(
            "REGISTER path=SL frame={} image={} inliers={} correspondences={} num_reg_frames={} num_points3d={} inliers_per_image={:?} continuing={} verified={}/{} parallax={}",
            frame_id,
            image_id,
            report.inliers.len(),
            corrs.len(),
            recon.num_reg_frames(),
            recon.num_points3d(),
            &per_world[..per_world.len().min(4)],
            continuing,
            verified,
            tracks.len(),
            with_parallax
        ));

        // Inlier correspondences per 2D point, in COLMAP's order.
        let mut inlier_corrs: BTreeMap<usize, Vec<(ImageT, usize)>> = BTreeMap::new();
        for &i in &report.inliers {
            let (idx, world_id, world_idx) = corrs[i];
            inlier_corrs
                .entry(idx)
                .or_default()
                .push((world_id, world_idx));
        }
        let mut config = BundleAdjustmentConfig::new();
        config.add_image(image_id);
        for (idx, views) in inlier_corrs {
            // Continue the first triangulated track among the matches.
            let continued = views.iter().find_map(|&(world_id, world_idx)| {
                recon.image(world_id).points2d[world_idx].point3d_id
            });
            if let Some(point3d_id) = continued {
                self.obs.add_observation(
                    recon,
                    graph,
                    point3d_id,
                    TrackElement {
                        image_id,
                        point2d_idx: idx,
                    },
                );
                self.triangulator.add_modified_point3d(point3d_id);
                continue;
            }
            // Otherwise triangulate a new track with the default (angular)
            // residual. COLMAP bounds it by `abs_pose_max_error` (pixels) as
            // if it were radians, which accepts every ray; convert, as the
            // GR6P threshold above does.
            let mut tri_views = views.clone();
            tri_views.push((image_id, idx));
            let Some((xyz, inliers)) = estimate_triangulation_of_views(
                recon,
                &tri_views,
                options.abs_pose_max_error / pixels_per_radian,
                options.filter_min_tri_angle_deg.to_radians(),
            ) else {
                continue;
            };
            if !inliers.last().copied().unwrap_or(false) {
                continue;
            }
            let track: Vec<TrackElement> = tri_views
                .iter()
                .zip(&inliers)
                .filter(|(_, &inlier)| inlier)
                .map(|(&(image_id, point2d_idx), _)| TrackElement {
                    image_id,
                    point2d_idx,
                })
                .collect();
            let point3d_id = self.obs.add_point3d(recon, graph, xyz, track);
            self.triangulator.add_modified_point3d(point3d_id);
            config.add_variable_point(point3d_id);
        }

        // Refine the pose with the triangulated structure.
        let mut ba_options = BundleAdjustmentOptions::local();
        ba_options.loss_function = LossFunction::Cauchy(1.0);
        ba_options.local_ba_point_policy = LocalBaPointPolicy::Colmap;
        bundle_adjustment::solve(&ba_options, &config, recon)
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
                let Some(point3d_id) = corr_image.points2d[corr.point2d_idx as usize].point3d_id
                else {
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
            if std::env::var_os("VISLOC_DEBUG_REG_FAIL").is_some() {
                eprintln!(
                    "REG_FAIL frame={} image={} correspondences={} reason=no_pose",
                    recon.image(image_id).frame_id,
                    image_id,
                    correspondences.len()
                );
            }
            return false;
        };
        if report.inliers.len() < options.abs_pose_min_num_inliers {
            if std::env::var_os("VISLOC_DEBUG_REG_FAIL").is_some() {
                eprintln!(
                    "REG_FAIL frame={} image={} correspondences={} inliers={} reason=too_few_inliers",
                    recon.image(image_id).frame_id,
                    image_id,
                    correspondences.len(),
                    report.inliers.len()
                );
            }
            if std::env::var("VISLOC_DEBUG_REG_FAIL_DUMP")
                .ok()
                .and_then(|v| v.parse::<ImageT>().ok())
                == Some(image_id)
            {
                let inlier_set: BTreeSet<usize> = report.inliers.iter().copied().collect();
                for (i, (point2d_idx, point3d_id)) in tri_corrs.iter().enumerate() {
                    let corr = &correspondences[i];
                    let is_inlier = inlier_set.contains(&i);
                    let track_len = recon.point3d(*point3d_id).track.len();
                    let xyz = recon.point3d(*point3d_id).xyz;
                    let residual = reproject(&camera, &report.pose, &corr.point3d)
                        .map(|p| (p - corr.point2d).norm());
                    eprintln!(
                        "REG_FAIL_DUMP frame={} image={} i={} point2d_idx={} point3d_id={} track_len={} xyz=({:.6},{:.6},{:.6}) px2d=({:.3},{:.3}) residual_px={} inlier={}",
                        recon.image(image_id).frame_id,
                        image_id,
                        i,
                        point2d_idx,
                        point3d_id,
                        track_len,
                        xyz.x,
                        xyz.y,
                        xyz.z,
                        corr.point2d.x,
                        corr.point2d.y,
                        residual
                            .map(|r| format!("{r:.3}"))
                            .unwrap_or_else(|| "nan".to_string()),
                        is_inlier
                    );
                }
            }
            return false;
        }
        if (report.inliers.len() as f64) / (correspondences.len() as f64)
            < options.abs_pose_min_inlier_ratio
        {
            if std::env::var_os("VISLOC_DEBUG_REG_FAIL").is_some() {
                eprintln!(
                    "REG_FAIL frame={} image={} correspondences={} inliers={} reason=low_ratio",
                    recon.image(image_id).frame_id,
                    image_id,
                    correspondences.len(),
                    report.inliers.len()
                );
            }
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
                    let Some(point3d_id) =
                        corr_image.points2d[corr.point2d_idx as usize].point3d_id
                    else {
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

        // Env-gated registration diagnostic (10k boundary investigation):
        // `VISLOC_DEBUG_REGISTER_FRAMES=4497,...` logs the correspondence and
        // inlier counts (raw and unique-3D-point) for those frames.
        if let Some(spec) = std::env::var_os("VISLOC_DEBUG_REGISTER_FRAMES") {
            let wanted: BTreeSet<FrameT> = spec
                .to_string_lossy()
                .split(',')
                .filter_map(|s| s.trim().parse::<FrameT>().ok())
                .collect();
            if wanted.contains(&frame_id) {
                let mut unique = BTreeSet::new();
                for &idx in &report.inliers {
                    unique.insert(tri_corrs[idx].2);
                }
                eprintln!(
                    "REGISTER_DBG frame={} gcorrs={} raw_inliers={} unique_point3d={} reproj={:?} refinement_applied={}",
                    frame_id,
                    gcorrs.len(),
                    report.inliers.len(),
                    unique.len(),
                    report.max_reprojection_error,
                    report.refinement_applied,
                );
            }
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
        let t = std::time::Instant::now();
        let completed = self.complete_tracks(tri_options, recon, graph);
        let complete_ms = t.elapsed().as_millis();
        let t = std::time::Instant::now();
        let merged = self.merge_tracks(tri_options, recon, graph);
        eprintln!(
            "TIMING complete_and_merge complete_ms={complete_ms} merge_ms={} completed={completed} merged={merged}",
            t.elapsed().as_millis()
        );
        completed + merged
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
        let t_find = std::time::Instant::now();
        let local_bundle = self.find_local_bundle(options, recon, image_id);
        let find_ms = t_find.elapsed().as_millis();

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

            // COLMAP fixes the camera (and rig) parameter blocks not fully
            // represented in the local window (`incremental_mapper.cc:
            // 1036-1065`): a camera is refined only while every registered
            // image of it is in the window, a rig's `sensor_from_rig` poses
            // only while every registered frame of it is.
            if ba_options.refine_focal_length || ba_options.refine_principal_point {
                let mut in_window: BTreeMap<CameraT, usize> = BTreeMap::new();
                for &iid in &config_image_ids {
                    *in_window.entry(recon.image(iid).camera_id).or_default() += 1;
                }
                let registered = &self.num_reg_images_per_camera;
                if in_window
                    .iter()
                    .any(|(id, &n)| n < registered.get(id).copied().unwrap_or(0))
                {
                    config.set_constant_intrinsics();
                }
            }
            if ba_options.refine_sensor_from_rig {
                let frame_ids: BTreeSet<FrameT> = config_image_ids
                    .iter()
                    .map(|&iid| recon.image(iid).frame_id)
                    .collect();
                let mut in_window: BTreeMap<RigT, usize> = BTreeMap::new();
                for &fid in &frame_ids {
                    *in_window.entry(recon.frame(fid).rig_id()).or_default() += 1;
                }
                let mut registered: BTreeMap<RigT, usize> = BTreeMap::new();
                for &fid in recon.reg_frame_ids() {
                    *registered.entry(recon.frame(fid).rig_id()).or_default() += 1;
                }
                if in_window.iter().any(|(id, &n)| n < registered[id]) {
                    config.set_constant_sensors_from_rig();
                }
            }

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

            let t_lba = std::time::Instant::now();
            let ba_ok = bundle_adjustment::solve(ba_options, &config, recon);
            let lba_ms = t_lba.elapsed().as_millis();
            let t_lmc = std::time::Instant::now();
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
            let (completed_tracks, completed_image) = self.triangulator.complete_tracks_and_image(
                tri_options,
                recon,
                graph,
                &mut self.obs,
                &variable_ids,
                image_id,
            );
            report.num_completed_observations = completed_tracks + completed_image;
            eprintln!(
                "TIMING local_parts find_ms={find_ms} ba_ms={lba_ms} merge_complete_ms={} points={}",
                t_lmc.elapsed().as_millis(),
                variable_ids.len()
            );
        }
        let t_lf = std::time::Instant::now();

        let (num_filtered, unchanged) = self.obs.filter_points3d_in_images_unchanged(
            recon,
            graph,
            options.filter_max_reproj_error,
            options.filter_min_tri_angle_deg,
            &config_image_ids,
        );
        report.num_filtered_observations = num_filtered;
        // The window filter's untouched points would be untouched again
        // (see `filter_points3d_in_images_unchanged`); both lists ascend.
        let mut unchanged = unchanged.into_iter().peekable();
        let point3d_ids_vec: Vec<Point3DT> = point3d_ids
            .iter()
            .copied()
            .filter(|&id| {
                while unchanged.next_if(|&u| u < id).is_some() {}
                unchanged.peek() != Some(&id)
            })
            .collect();
        report.num_filtered_observations += self.obs.filter_points3d(
            recon,
            graph,
            options.filter_max_reproj_error,
            options.filter_min_tri_angle_deg,
            &point3d_ids_vec,
        );
        eprintln!(
            "TIMING local_filter elapsed_ms={}",
            t_lf.elapsed().as_millis()
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
        let t = std::time::Instant::now();
        self.obs
            .filter_observations_with_negative_depth(recon, graph);
        let filter_ms = t.elapsed().as_millis();

        let t = std::time::Instant::now();
        let mut config = BundleAdjustmentConfig::new();
        for &frame_id in recon.reg_frame_ids() {
            config.add_frame(recon, frame_id);
        }
        if config.num_images() < 2 {
            return false;
        }
        // COLMAP's `AddImageToProblem` parameterizes every observed point
        // (source-verified against 64805cb, see
        // `docs/colmap_rig_mapper_port_plan.md` §8); `solve` then keeps each
        // variable iff `refine_points3d`. Without this the global pass only
        // optimizes poses against frozen structure.
        if ba_options.refine_points3d {
            for point3d_id in bundle_adjustment::points_observed_by(recon, config.images()) {
                config.add_variable_point(point3d_id);
            }
        }
        config.fix_gauge(Gauge::OneFrameFromWorld);
        let num_images = config.num_images();
        let num_points = recon.num_points3d();
        let config_ms = t.elapsed().as_millis();
        let t = std::time::Instant::now();
        let ok = bundle_adjustment::solve(ba_options, &config, recon);
        eprintln!(
            "TIMING global_ba_parts filter_ms={filter_ms} config_ms={config_ms} solve_ms={}",
            t.elapsed().as_millis()
        );
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
        let mut ms = [0u128; 4];
        let t = std::time::Instant::now();
        self.complete_and_merge_tracks(tri_options, recon, graph);
        ms[0] += t.elapsed().as_millis();
        let t = std::time::Instant::now();
        self.retriangulate(tri_options, recon, graph);
        ms[1] += t.elapsed().as_millis();
        for _ in 0..max_num_refinements {
            let num_observations: usize = recon.points3d().values().map(|p| p.track.len()).sum();
            let t = std::time::Instant::now();
            self.adjust_global_bundle(ba_options, recon, graph);
            ms[2] += t.elapsed().as_millis();
            let t = std::time::Instant::now();
            let mut num_changed = self.complete_and_merge_tracks(tri_options, recon, graph);
            ms[0] += t.elapsed().as_millis();
            let t = std::time::Instant::now();
            num_changed += self.filter_points(options, recon, graph);
            ms[3] += t.elapsed().as_millis();
            eprintln!(
                "TIMING global_refinement_parts complete_merge_ms={} retriangulate_ms={} \
                 global_ba_ms={} filter_ms={}",
                ms[0], ms[1], ms[2], ms[3]
            );
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
    /// `mapper_impl.rs`). The structure-less rounds rank by 2D-2D
    /// correspondences to registered images and count structure-less
    /// trials separately.
    pub fn find_next_images(
        &self,
        options: &Options,
        recon: &Reconstruction,
        round: CandidateRound,
    ) -> Vec<ImageT> {
        let ranked = mapper_impl::find_next_images(
            options.abs_pose_min_num_inliers,
            options.max_reg_trials,
            recon,
            &self.obs,
            &self.filtered_frames,
            if round == CandidateRound::Structure {
                &self.num_reg_trials
            } else {
                &self.num_structure_less_reg_trials
            },
            round,
        );

        // Env-gated boundary diagnostic for the 10k model-splitting
        // investigation: dump the visibility/trial state of specific frames
        // as the frontier approaches them, and the chosen next image.
        // `VISLOC_DEBUG_BOUNDARY_FRAMES=4493,4494,...`,
        // `VISLOC_DEBUG_BOUNDARY_MIN_REG=4480`.
        if let Some(spec) = std::env::var_os("VISLOC_DEBUG_BOUNDARY_FRAMES") {
            let min_reg: usize = std::env::var("VISLOC_DEBUG_BOUNDARY_MIN_REG")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let n_reg = recon.num_reg_frames();
            if n_reg >= min_reg && n_reg <= min_reg + 40 {
                let frames: BTreeSet<FrameT> = spec
                    .to_string_lossy()
                    .split(',')
                    .filter_map(|s| s.trim().parse::<FrameT>().ok())
                    .collect();
                let chosen = ranked.first().map(|&iid| (iid, recon.image(iid).frame_id));
                eprintln!(
                    "BOUNDARY n_reg={} n_candidates={} chosen={:?}",
                    n_reg,
                    ranked.len(),
                    chosen
                );
                for (&image_id, image) in recon.images() {
                    if !frames.contains(&image.frame_id) {
                        continue;
                    }
                    eprintln!(
                        "BOUNDARY frame={} image={} registered={} visible={} trials={} in_candidates={}",
                        image.frame_id,
                        image_id,
                        recon.is_image_registered(image_id),
                        self.obs.num_visible_points3d(image_id),
                        *self.num_reg_trials.get(&image_id).unwrap_or(&0),
                        ranked.contains(&image_id),
                    );
                }
            }
        }
        ranked
    }
}

/// Of the GR6P inliers `epipolar` (query ray, world centre, world ray):
/// those within `max_error / 2` pixels of their epipolar planes under
/// `cam_from_world`, and those of them whose two rays meet at
/// [`STRUCTURE_LESS_MIN_TRI_ANGLE_DEG`] or more.
fn epipolar_support(
    cam_from_world: &SE3,
    epipolar: &[(Vector3<f64>, Vector3<f64>, Vector3<f64>)],
    max_error: f64,
    pixels_per_radian: f64,
) -> (usize, usize) {
    let world_from_cam = cam_from_world.rotation.inverse();
    let centre = -(world_from_cam * cam_from_world.translation);
    let min_cos = STRUCTURE_LESS_MIN_TRI_ANGLE_DEG.to_radians().cos();
    let (mut kept, mut parallax) = (0, 0);
    for (query, origin, world_ray) in epipolar {
        let normal = (centre - origin).cross(world_ray);
        let norm = normal.norm();
        let query = (world_from_cam * query).normalize();
        if norm <= 1e-12 || pixels_per_radian * (normal / norm).dot(&query).abs() > 0.5 * max_error
        {
            continue;
        }
        kept += 1;
        if query.dot(&world_ray.normalize()) <= min_cos {
            parallax += 1;
        }
    }
    (kept, parallax)
}

/// How many of `tracks` (pixel, 3D point) `cam_from_world` reprojects in
/// front of the camera within `max_error` pixels.
fn count_reprojected(
    camera: &visloc_core::types::Camera,
    cam_from_world: &SE3,
    tracks: &[(Point2<f64>, Vector3<f64>)],
    max_error: f64,
) -> usize {
    tracks
        .iter()
        .filter(|(xy, xyz)| {
            let in_cam = cam_from_world.rotation * xyz + cam_from_world.translation;
            in_cam.z > 0.0
                && camera
                    .project(&Point3::from(in_cam))
                    .is_some_and(|uv| (uv - xy).norm() <= max_error)
        })
        .count()
}

/// Not in COLMAP: another pose for a structure-less registration whose
/// GR6P pose fails the track check, when the registered cameras it matches
/// sit at nearly one centre (views panned from one spot). Their 2D-2D
/// matches fix the rotation and the direction from that centre but hardly
/// the distance: on ETH3D electro, DSC_9289's matches went to three views
/// 5 cm apart, and GR6P put it 15 cm off with 0 of 14 tracks reprojecting.
///
/// The tracks set the distance only, one degree of freedom, so the caller's
/// check on them stays nearly independent:
/// - along GR6P's direction from the inliers' mean world centre, the
///   distance that reprojects the most `tracks` within `max_error`;
/// - the pose refined from there on the inliers' epipolar errors alone, in
///   pixels (Levenberg-Marquardt, Huber at `max_error / 6`);
/// - the distance chosen again along the refined direction.
///
/// Returns `None` unless a track agrees and 90% of the inliers stay within
/// `max_error / 2` of their epipolar planes. `epipolar` holds, per GR6P
/// inlier, the query camera's ray and the world camera's centre and ray.
/// The caller checks the result as it checks GR6P's pose.
fn rescue_structure_less_pose(
    camera: &visloc_core::types::Camera,
    pose: &SE3,
    epipolar: &[(Vector3<f64>, Vector3<f64>, Vector3<f64>)],
    tracks: &[(Point2<f64>, Vector3<f64>)],
    max_error: f64,
    pixels_per_radian: f64,
) -> Option<SE3> {
    use nalgebra::{Matrix6, UnitQuaternion, Vector6};
    if epipolar.is_empty() || tracks.is_empty() {
        return None;
    }
    let cluster = epipolar
        .iter()
        .fold(Vector3::zeros(), |sum, (_, origin, _)| sum + origin)
        / epipolar.len() as f64;
    let at = |rotation: UnitQuaternion<f64>, centre: Vector3<f64>| {
        SE3::new(rotation, -(rotation * centre))
    };
    // The centre along `centre - cluster` from which the most tracks
    // reproject, and how many: each track's ray passes closest to its 3D
    // point from one distance.
    let place = |rotation: UnitQuaternion<f64>, centre: Vector3<f64>| {
        let distance = (centre - cluster).norm();
        if distance.is_nan() || distance <= 1e-12 {
            return None;
        }
        let direction = (centre - cluster) / distance;
        let world_from_cam = rotation.inverse();
        let mut best: Option<(usize, Vector3<f64>)> = None;
        for (xy, xyz) in tracks {
            let Some(ray) = pixel_ray(camera, xy) else {
                continue;
            };
            let ray = (world_from_cam * ray).normalize();
            let b = direction.dot(&ray);
            let det = 1.0 - b * b;
            if det < 1e-9 {
                continue;
            }
            let offset = xyz - cluster;
            let (along, depth) = (direction.dot(&offset), ray.dot(&offset));
            let lambda = (along - b * depth) / det;
            if lambda <= 0.0 || (depth - b * along) / det <= 0.0 {
                continue;
            }
            let candidate = cluster + lambda * direction;
            let count = count_reprojected(camera, &at(rotation, candidate), tracks, max_error);
            if best.is_none_or(|(n, _)| count > n) {
                best = Some((count, candidate));
            }
        }
        best
    };
    let epipolar_errors = |rotation: UnitQuaternion<f64>, centre: Vector3<f64>| -> Vec<f64> {
        let cam_from_world = rotation.inverse();
        epipolar
            .iter()
            .map(|(query, origin, world_ray)| {
                let normal = (centre - origin).cross(world_ray);
                let norm = normal.norm();
                if norm > 1e-12 {
                    pixels_per_radian * (normal / norm).dot(&(cam_from_world * query).normalize())
                } else {
                    0.0
                }
            })
            .collect()
    };

    let gr6p_centre = -(pose.rotation.inverse() * pose.translation);
    let start = place(pose.rotation, gr6p_centre)
        .filter(|&(count, _)| count > 0)
        .map_or(gr6p_centre, |(_, centre)| centre);
    let scale = (start - cluster).norm().max(1e-12);

    // Levenberg-Marquardt on the epipolar errors: a left-multiplied rotation
    // (scaled axis) and a centre update.
    let unpack = |p: &Vector6<f64>| {
        (
            UnitQuaternion::from_scaled_axis(Vector3::new(p[0], p[1], p[2])) * pose.rotation,
            start + Vector3::new(p[3], p[4], p[5]),
        )
    };
    let huber = max_error / 6.0;
    let cost = |r: &[f64]| -> f64 {
        r.iter()
            .map(|&e| {
                if e.abs() <= huber {
                    0.5 * e * e
                } else {
                    huber * (e.abs() - 0.5 * huber)
                }
            })
            .sum()
    };
    let steps = [1e-7, 1e-7, 1e-7, 1e-7 * scale, 1e-7 * scale, 1e-7 * scale];
    let mut p = Vector6::zeros();
    let (r0, c0) = unpack(&p);
    let mut residuals = epipolar_errors(r0, c0);
    let mut current = cost(&residuals);
    let mut damping = 1e-4;
    for _ in 0..30 {
        let mut jtj = Matrix6::<f64>::zeros();
        let mut jtr = Vector6::<f64>::zeros();
        let mut jac = vec![Vector6::<f64>::zeros(); residuals.len()];
        for (k, h) in steps.iter().enumerate() {
            let (mut plus, mut minus) = (p, p);
            plus[k] += h;
            minus[k] -= h;
            let (rp, cp) = unpack(&plus);
            let (rm, cm) = unpack(&minus);
            for (row, (a, b)) in jac
                .iter_mut()
                .zip(epipolar_errors(rp, cp).iter().zip(&epipolar_errors(rm, cm)))
            {
                row[k] = (a - b) / (2.0 * h);
            }
        }
        for (row, &e) in jac.iter().zip(&residuals) {
            let w = if e.abs() <= huber {
                1.0
            } else {
                huber / e.abs()
            };
            jtj += w * row * row.transpose();
            jtr += w * e * row;
        }
        let mut improved = false;
        while damping < 1e8 {
            let lhs = jtj + Matrix6::from_diagonal(&(jtj.diagonal() * damping));
            let Some(step) = lhs.cholesky().map(|c| c.solve(&(-jtr))) else {
                damping *= 10.0;
                continue;
            };
            let trial = p + step;
            let (rt, ct) = unpack(&trial);
            let trial_residuals = epipolar_errors(rt, ct);
            let trial_cost = cost(&trial_residuals);
            if trial_cost < current {
                improved = current - trial_cost > 1e-12 * current;
                (p, residuals, current) = (trial, trial_residuals, trial_cost);
                damping = (damping * 0.1).max(1e-9);
                break;
            }
            damping *= 10.0;
        }
        if !improved {
            break;
        }
    }
    let (rotation, centre) = unpack(&p);
    let (count, centre) = place(rotation, centre)?;
    let rescued = at(rotation, centre);
    let (kept, _) = epipolar_support(&rescued, epipolar, max_error, pixels_per_radian);
    if count == 0 || (kept as f64) < 0.9 * epipolar.len() as f64 {
        return None;
    }
    Some(rescued)
}

/// The camera-frame ray of pixel `xy`: `(x, y, 1)` on the normalized image
/// plane, or a spherical camera's unit ray, which may point behind it.
fn pixel_ray(camera: &visloc_core::types::Camera, xy: &Point2<f64>) -> Option<Vector3<f64>> {
    if camera.model.is_spherical() {
        return camera.unit_ray_from_pixel(xy);
    }
    let n = camera.normalize_pixel(xy)?;
    Some(Vector3::new(n.x, n.y, 1.0))
}

#[cfg(test)]
mod tests {
    use super::{CandidateRound, IncrementalMapper, Options};

    use crate::colmap_incremental::pipeline::reconstruction_from_cache;
    use crate::colmap_incremental::test_support::build_three_view_scene;
    use crate::colmap_incremental::types::ImageT;

    /// Structure-less registration places an image that sees no 3D point
    /// (two registered images, no structure) at its true pose from its
    /// 2D-2D matches alone, with metric scale, and triangulates its matches.
    #[test]
    fn structure_less_registration_recovers_the_pose_without_structure() {
        let (db, poses, num_points) = build_three_view_scene();
        let mut recon = reconstruction_from_cache(&db);
        for (k, pose) in poses.iter().take(2).enumerate() {
            recon.frame_mut(k as u64).set_rig_from_world(pose.clone());
            recon.register_frame(k as u64);
        }

        let mut mapper = IncrementalMapper::new();
        mapper.begin_reconstruction(&recon, db.correspondence_graph());
        let options = Options {
            abs_pose_min_num_inliers: 20,
            ..Options::default()
        };
        assert_eq!(
            mapper.find_next_images(&options, &recon, CandidateRound::Structure),
            Vec::<ImageT>::new(),
            "no 3D point is visible, so structure-based registration has no candidate"
        );
        assert_eq!(
            mapper.find_next_images(&options, &recon, CandidateRound::StructureLess),
            Vec::<ImageT>::new(),
            "COLMAP's structure-less round also asks for visible 3D points"
        );
        assert_eq!(
            mapper.find_next_images(
                &options,
                &recon,
                CandidateRound::StructureLessByCorrespondences
            ),
            vec![2],
            "120 correspondences to registered images make image 2 a candidate"
        );
        assert!(mapper.register_next_structure_less_image(
            &options,
            &mut recon,
            db.correspondence_graph(),
            2
        ));
        let got = recon.frame(2).rig_from_world();
        let rotation_error = (got.rotation.inverse() * poses[2].rotation).angle();
        let centre_error = (got.inverse().translation - poses[2].inverse().translation).norm();
        assert!(
            rotation_error < 1e-6,
            "rotation off by {rotation_error} rad"
        );
        assert!(centre_error < 1e-6, "centre off by {centre_error}");
        assert_eq!(recon.num_points3d(), num_points);
    }

    /// With `structure_less_min_verified`, a structure-less registration
    /// whose matches continue too few triangulated tracks is refused: the
    /// same scene, with no 3D point at all, leaves image 2 unregistered.
    #[test]
    fn structure_less_registration_needs_verified_tracks_when_asked() {
        let (db, poses, _) = build_three_view_scene();
        let mut recon = reconstruction_from_cache(&db);
        for (k, pose) in poses.iter().take(2).enumerate() {
            recon.frame_mut(k as u64).set_rig_from_world(pose.clone());
            recon.register_frame(k as u64);
        }
        let mut mapper = IncrementalMapper::new();
        mapper.begin_reconstruction(&recon, db.correspondence_graph());
        let options = Options {
            abs_pose_min_num_inliers: 20,
            structure_less_min_verified: 10,
            ..Options::default()
        };
        assert!(!mapper.register_next_structure_less_image(
            &options,
            &mut recon,
            db.correspondence_graph(),
            2
        ));
        assert!(!recon.is_image_registered(2));
        assert_eq!(recon.num_points3d(), 0);
    }

    /// A camera whose 2D-2D matches all go to three views 2 cm apart, 1 m
    /// away: started 30% short of that distance and 0.6° off, the rescue
    /// puts it back from 15 tracks, with parallax on most inliers. At 1 cm
    /// from them, whatever it returns lacks that parallax, and the caller
    /// refuses it.
    #[test]
    fn rescue_sets_the_distance_from_a_panned_cluster() {
        use super::{
            count_reprojected, epipolar_support, rescue_structure_less_pose,
            STRUCTURE_LESS_MIN_PARALLAX_RATIO,
        };
        use nalgebra::{Point2, UnitQuaternion, Vector3};
        use visloc_core::geometry::SE3;
        use visloc_core::types::Camera;

        let camera = Camera::pinhole(1, 1280, 960, 1000.0, 1000.0, 640.0, 480.0);
        let cluster: Vec<Vector3<f64>> = (0..3)
            .map(|j| Vector3::new(0.01 * j as f64, 0.0, 0.0))
            .collect();
        let mut seed = 7u64;
        let mut uniform = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let points: Vec<Vector3<f64>> = (0..300)
            .map(|_| {
                Vector3::new(
                    4.0 * uniform() - 1.5,
                    3.0 * uniform() - 1.5,
                    3.0 + 4.0 * uniform(),
                )
            })
            .collect();
        let scene = |centre: Vector3<f64>, rotation: UnitQuaternion<f64>| {
            let pose = SE3::new(rotation, -(rotation * centre));
            let mut epipolar = Vec::new();
            let mut tracks = Vec::new();
            for x in &points {
                let in_cam = rotation * (x - centre);
                let Some(uv) = camera.project(&nalgebra::Point3::from(in_cam)) else {
                    continue;
                };
                if in_cam.z <= 0.0 || uv.x < 0.0 || uv.y < 0.0 || uv.x > 1280.0 || uv.y > 960.0 {
                    continue;
                }
                for c in &cluster {
                    epipolar.push((in_cam.normalize(), *c, (x - c).normalize()));
                }
                if tracks.len() < 15 {
                    tracks.push((Point2::new(uv.x, uv.y), *x));
                }
            }
            (pose, epipolar, tracks)
        };
        let rotation = UnitQuaternion::from_scaled_axis(Vector3::new(0.0, -3f64.to_radians(), 0.0));
        let tilt = UnitQuaternion::from_scaled_axis(
            Vector3::new(0.006, 0.004, -0.008).normalize() * 0.6f64.to_radians(),
        );
        let mean = (cluster[0] + cluster[1] + cluster[2]) / 3.0;

        let centre = Vector3::new(1.0, 0.0, 0.2);
        let (truth, epipolar, tracks) = scene(centre, rotation);
        let short = mean + 0.7 * (centre - mean);
        let start = SE3::new(tilt * rotation, -((tilt * rotation) * short));
        assert_eq!(count_reprojected(&camera, &start, &tracks, 12.0), 0);
        let rescued = rescue_structure_less_pose(&camera, &start, &epipolar, &tracks, 12.0, 1000.0)
            .expect("rescued pose");
        let rescued_centre = -(rescued.rotation.inverse() * rescued.translation);
        assert!(
            (rescued_centre - centre).norm() < 1e-3,
            "centre {rescued_centre:?}"
        );
        assert!(rescued.rotation.angle_to(&truth.rotation).to_degrees() < 0.01);
        assert_eq!(
            count_reprojected(&camera, &rescued, &tracks, 12.0),
            tracks.len()
        );
        let share = |pose: &SE3, epipolar: &[(Vector3<f64>, Vector3<f64>, Vector3<f64>)]| {
            epipolar_support(pose, epipolar, 12.0, 1000.0).1 as f64 / epipolar.len() as f64
        };
        assert!(share(&rescued, &epipolar) >= STRUCTURE_LESS_MIN_PARALLAX_RATIO);

        let centre = mean + Vector3::new(0.01, 0.0, 0.0);
        let (_, epipolar, tracks) = scene(centre, rotation);
        let far = mean + Vector3::new(1.0, 0.0, 0.0);
        let start = SE3::new(rotation, -(rotation * far));
        if let Some(pose) =
            rescue_structure_less_pose(&camera, &start, &epipolar, &tracks, 12.0, 1000.0)
        {
            assert!(share(&pose, &epipolar) < STRUCTURE_LESS_MIN_PARALLAX_RATIO);
        }
    }

    #[test]
    fn options_apply_overrides_by_key() {
        let mut options = Options::default();
        options
            .apply_override("abs_pose_min_num_inliers=30")
            .unwrap();
        options
            .apply_override("filter_max_reproj_error = 2.5")
            .unwrap();
        assert_eq!(options.abs_pose_min_num_inliers, 30);
        assert_eq!(options.filter_max_reproj_error, 2.5);
        options
            .apply_override("structure_less_min_verified=10")
            .unwrap();
        assert_eq!(options.structure_less_min_verified, 10);
        assert!(options.apply_override("no_such_key=1").is_err());
        assert!(options.apply_override("max_reg_trials").is_err());
        assert!(options.apply_override("max_reg_trials=x").is_err());
    }
}
