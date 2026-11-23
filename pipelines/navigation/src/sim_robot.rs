//! The simulated robot localized in a visual map, as a navigation stack
//! sees it: wheel odometry in an `odom` frame, and a `map -> odom`
//! transform from map-based localization.
//!
//! [`SimRobot`] drives a [`DiffDrive`] in a [`World`] on velocity commands,
//! integrates [`WheelOdometry`], and, at the camera rate, localizes the
//! camera's detections in the map with visloc's
//! [`LocalizationPipeline`] (descriptor matching and PnP-RANSAC), feeding
//! the poses to an [`OdomCorrection`] that keeps `map -> odom`. This is the
//! REP-105 contract Nav2 expects from a localization system (as AMCL
//! provides): `map -> odom -> base_link`.
//!
//! [`LocalizationMode`] selects where `map -> odom` comes from, so the
//! same drive can be compared: the ground truth (perfect localization), the
//! start pose alone (odometry only, drifting), or visloc.

use std::time::Instant;

use nalgebra::Point3;
use visloc_core::geometry::SE3;
use visloc_core::types::{LandmarkDescriptorStore, VisualMap};
use visloc_localization::{
    AllLandmarksSelector, FixOutcome, LocalizationConfig, LocalizationPipeline, OdomCorrection,
    OdomCorrectionConfig, RadiusLandmarkSelector,
};
use visloc_vision::matching::BruteForceMatcher;
use visloc_vision::pnp::{GaussNewtonPoseRefiner, P3PSolver, P3P};
use visloc_vision::ransac::PnPRansac;

/// The localization pipeline: descriptor matching and P3P-RANSAC, as
/// `basalt_euroc_localize_demo` localizes a robot in a saved map (a
/// minimal solver that, unlike DLT, handles a view of a single wall).
type Pipeline = LocalizationPipeline<BruteForceMatcher, AllLandmarksSelector, PnPRansac<P3P>>;

use crate::sim::{
    CameraNoise, CameraRig, DiffDrive, DriveConfig, OdometryConfig, Pose2, ScanConfig, SimRng,
    WheelOdometry, World,
};

/// Where the robot's `map -> odom` comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalizationMode {
    /// The true pose: perfect localization, the upper bound.
    GroundTruth,
    /// The start pose, given once (as an initial pose set in RViz), then
    /// wheel odometry alone.
    Odometry,
    /// visloc map-based localization corrects the wheel odometry; the start
    /// pose is not given.
    Visloc,
}

impl std::str::FromStr for LocalizationMode {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "ground_truth" => Ok(Self::GroundTruth),
            "odometry" => Ok(Self::Odometry),
            "visloc" => Ok(Self::Visloc),
            other => Err(format!(
                "unknown localization mode {other} (ground_truth, odometry or visloc)"
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimRobotConfig {
    pub mode: LocalizationMode,
    pub drive: DriveConfig,
    pub odometry: OdometryConfig,
    pub camera_noise: CameraNoise,
    pub scan: ScanConfig,
    /// Camera images per second (localization attempts).
    pub camera_rate_hz: f64,
    /// A localization with this many PnP inliers is a fix ...
    pub fix_inliers: usize,
    /// ... and one with this many a weak fix (it can only help to set the
    /// first `map -> odom`); fewer is a failure.
    pub weak_fix_inliers: usize,
    /// Once localized, only landmarks this close to the predicted camera
    /// are matched.
    pub search_radius_m: f64,
    /// The P3P solver inside RANSAC (Lambda Twist is the more accurate one
    /// on views of walls; Grunert is the historical default).
    pub p3p_solver: P3PSolver,
    /// `PnPRansac::inlier_refits`.
    pub inlier_refits: usize,
    pub correction: OdomCorrectionConfig,
    /// In [`LocalizationMode::Visloc`], turn in place (ignoring commands)
    /// until the first `map -> odom`: a robot that wakes up facing a bare
    /// wall looks around for the map.
    pub spin_until_localized: bool,
    pub seed: u64,
}

impl Default for SimRobotConfig {
    fn default() -> Self {
        Self {
            mode: LocalizationMode::Visloc,
            drive: DriveConfig::default(),
            odometry: OdometryConfig::default(),
            camera_noise: CameraNoise::default(),
            scan: ScanConfig::default(),
            camera_rate_hz: 5.0,
            fix_inliers: 15,
            weak_fix_inliers: 8,
            search_radius_m: 8.0,
            p3p_solver: P3PSolver::Grunert,
            inlier_refits: 0,
            correction: OdomCorrectionConfig {
                // Wheel odometry drifts faster in heading than a VIO.
                gate_yaw_deg: 5.0,
                gate_yaw_per_m_deg: 3.0,
                window_s: 2.0,
                ..OdomCorrectionConfig::default()
            },
            spin_until_localized: true,
            seed: 11,
        }
    }
}

/// One localization attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalizationEvent {
    pub time_s: f64,
    pub inliers: usize,
    pub keypoints: usize,
    /// The localized base pose, if the pipeline returned one.
    pub map_pose: Option<Pose2>,
    /// What the correction did with it (`None`: too few inliers).
    pub outcome: Option<FixOutcome>,
    pub latency_s: f64,
}

/// What one [`SimRobot::step`] produced.
#[derive(Debug, Clone, PartialEq)]
pub struct StepOutput {
    pub time_s: f64,
    /// True base pose in the map (world) frame.
    pub ground_truth: Pose2,
    /// Wheel odometry pose and velocities.
    pub odom: Pose2,
    pub linear: f64,
    pub angular: f64,
    /// `map -> odom`, once known.
    pub map_from_odom: Option<Pose2>,
    pub localization: Option<LocalizationEvent>,
}

impl StepOutput {
    /// The base pose in the map frame the navigation stack believes.
    pub fn estimate(&self) -> Option<Pose2> {
        self.map_from_odom.map(|m| m.compose(&self.odom))
    }
}

/// Running counts of a [`SimRobot`]'s localization.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LocalizationCounts {
    pub attempts: usize,
    pub fixes: usize,
    pub weak_fixes: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub resets: usize,
    pub latency_sum_s: f64,
    pub latency_max_s: f64,
}

/// The simulated robot with its odometry and localization.
pub struct SimRobot {
    pub config: SimRobotConfig,
    pub world: World,
    pub rig: CameraRig,
    pub robot: DiffDrive,
    pub odometry: WheelOdometry,
    map: VisualMap,
    descriptors: LandmarkDescriptorStore,
    pipeline: Pipeline,
    correction: OdomCorrection,
    /// `map -> odom` of the odometry mode (the start pose).
    start: Pose2,
    rng: SimRng,
    time_s: f64,
    next_camera_s: f64,
    pub counts: LocalizationCounts,
}

impl SimRobot {
    /// A robot at `start` (true pose) in `world`, localizing in `map`.
    pub fn new(
        world: World,
        rig: CameraRig,
        map: VisualMap,
        descriptors: LandmarkDescriptorStore,
        start: Pose2,
        config: SimRobotConfig,
    ) -> Self {
        let localization = LocalizationConfig {
            min_inliers: config.weak_fix_inliers,
            ..LocalizationConfig::default()
        };
        let ransac = PnPRansac {
            pose_estimator: P3P::new(config.p3p_solver),
            pose_refiner: Some(GaussNewtonPoseRefiner::default()),
            iterations: 200,
            reprojection_threshold: localization.reprojection_threshold,
            seed: config.seed,
            early_stop_min_iterations: 0,
            early_stop_inlier_ratio: None,
            confidence: Some(0.999),
            inlier_refits: config.inlier_refits,
        };
        let pipeline = LocalizationPipeline::with_pose_estimator(
            BruteForceMatcher {
                ratio: localization.ratio,
            },
            AllLandmarksSelector,
            ransac,
            localization,
        );
        Self {
            world,
            rig,
            robot: DiffDrive::new(config.drive, start),
            odometry: WheelOdometry::new(config.odometry),
            map,
            descriptors,
            pipeline,
            correction: OdomCorrection::new(config.correction.clone()),
            start,
            rng: SimRng::new(config.seed),
            time_s: 0.0,
            next_camera_s: 0.0,
            counts: LocalizationCounts::default(),
            config,
        }
    }

    pub const fn time_s(&self) -> f64 {
        self.time_s
    }

    /// `map -> odom` now, by the configured mode.
    pub fn map_from_odom(&self) -> Option<Pose2> {
        match self.config.mode {
            LocalizationMode::GroundTruth => {
                Some(self.robot.pose.compose(&self.odometry.pose.inverse()))
            }
            LocalizationMode::Odometry => Some(self.start),
            LocalizationMode::Visloc => self
                .correction
                .map_from_odom()
                .map(|transform| Pose2::from_se3(&transform)),
        }
    }

    /// A planar range scan from the robot's true pose.
    pub fn scan(&mut self) -> Vec<Option<f64>> {
        let pose = self.robot.pose;
        self.world.scan(&pose, &self.config.scan, &mut self.rng)
    }

    /// Advances `dt` seconds under the velocity command, localizing when a
    /// camera image is due.
    pub fn step(&mut self, linear_cmd: f64, angular_cmd: f64, dt: f64) -> StepOutput {
        let (mut linear_cmd, mut angular_cmd) = (linear_cmd, angular_cmd);
        if self.config.mode == LocalizationMode::Visloc
            && self.config.spin_until_localized
            && !self.correction.is_set()
        {
            linear_cmd = 0.0;
            angular_cmd = 0.4;
        }
        let (v, w) = self
            .robot
            .step(&self.world, linear_cmd, angular_cmd, dt, &mut self.rng);
        self.odometry.step(v, w, dt, &mut self.rng);
        self.time_s += dt;
        let stamp_ns = (self.time_s * 1e9).round() as i64;
        let odom_se3 = self.odometry.pose.to_se3();
        self.correction.observe_odometry(stamp_ns, &odom_se3);

        let localization = (self.config.mode == LocalizationMode::Visloc
            && self.time_s + 1e-9 >= self.next_camera_s)
            .then(|| {
                self.next_camera_s = self.time_s + 1.0 / self.config.camera_rate_hz;
                self.localize(stamp_ns, &odom_se3)
            });
        StepOutput {
            time_s: self.time_s,
            ground_truth: self.robot.pose,
            odom: self.odometry.pose,
            linear: self.odometry.linear,
            angular: self.odometry.angular,
            map_from_odom: self.map_from_odom(),
            localization,
        }
    }

    fn localize(&mut self, stamp_ns: i64, odom: &SE3) -> LocalizationEvent {
        let truth = self.robot.pose;
        let (query, _) = self.rig.observe(
            &self.world,
            &truth,
            &self.config.camera_noise,
            &mut self.rng,
        );
        let predicted = self.correction.map_from_body(odom);
        let started = Instant::now();
        let result = match &predicted {
            Some(map_from_base) => {
                let prior = self.rig.camera_pose(&Pose2::from_se3(map_from_base));
                let center = prior.camera_center_world();
                self.pipeline
                    .localize_with_candidate_selector_and_descriptor_store_and_pose_prior(
                        &query,
                        &self.map,
                        &self.descriptors,
                        RadiusLandmarkSelector::new(
                            Point3::from(center.coords),
                            self.config.search_radius_m,
                        ),
                        Some(&prior),
                    )
            }
            None => self
                .pipeline
                .localize_with_candidate_selector_and_descriptor_store_and_pose_prior(
                    &query,
                    &self.map,
                    &self.descriptors,
                    AllLandmarksSelector,
                    None,
                ),
        };
        let latency_s = started.elapsed().as_secs_f64();
        let counts = &mut self.counts;
        counts.attempts += 1;
        counts.latency_sum_s += latency_s;
        counts.latency_max_s = counts.latency_max_s.max(latency_s);
        let map_from_base = result
            .pose
            .as_ref()
            .filter(|_| result.success)
            .map(|pose| self.rig.base_from_camera_pose(pose));
        let outcome = map_from_base.as_ref().and_then(|map_from_base| {
            // The robot is on the floor: keep the localized pose planar.
            let planar = Pose2::from_se3(map_from_base).to_se3();
            if result.inlier_count >= self.config.fix_inliers {
                counts.fixes += 1;
                Some(self.correction.add_fix(stamp_ns, &planar, odom))
            } else if result.inlier_count >= self.config.weak_fix_inliers {
                counts.weak_fixes += 1;
                Some(self.correction.add_weak_fix(stamp_ns, &planar, odom))
            } else {
                None
            }
        });
        match outcome {
            Some(FixOutcome::Accepted | FixOutcome::Initialized) => counts.accepted += 1,
            Some(FixOutcome::Rejected) => counts.rejected += 1,
            Some(FixOutcome::Reset) => counts.resets += 1,
            _ => {}
        }
        LocalizationEvent {
            time_s: self.time_s,
            inliers: result.inlier_count,
            keypoints: query.keypoints.len(),
            map_pose: map_from_base.map(|pose| Pose2::from_se3(&pose)),
            outcome,
            latency_s,
        }
    }
}
