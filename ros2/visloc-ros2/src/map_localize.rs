//! Transport-agnostic core of `visloc_map_localize_node`: robot-style map
//! localization against a map saved by
//! `basalt_euroc_online_slam_demo --save-map`, correcting VIO odometry into
//! the map frame.
//!
//! Same path as `basalt_euroc_localize_demo --vio`
//! (`docs/basalt_map_localization.md`): each localized image goes through the
//! Basalt mapper's own ORB frontend, is matched to the map's landmark
//! descriptors by Hamming distance and solved with P3P RANSAC. Once the
//! odometry has a map pose, the correction and the odometry predict the
//! pose, and matching only compares each
//! landmark with the keypoints near its predicted projection (falling back
//! to the map-wide search when that fails), which keeps localization cheap
//! enough to run next to the VIO in real time; an accepted
//! pose (a *fix*) is paired with the odometry pose at the same stamp and fed
//! to [`OdomCorrection`], which keeps the `map -> odom` transform (REP-105).
//! Every odometry pose is then reported in the map frame, with the time and
//! distance since the correction's newest fix.
//!
//! Fixes and odometry arrive on different threads and in either order (the
//! VIO may publish a frame's odometry before or after that frame is
//! localized), so [`OdomCorrector`] keeps a short odometry history and holds a
//! fix until the odometry for its stamp has arrived.

use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::time::Instant;

use nalgebra::{Quaternion as NaQuaternion, UnitQuaternion, Vector3};
use visloc_basalt::mapper::{
    compute_angles, compute_descriptors, detect_keypoints_mapping, localization_camera_id,
    orb_descriptor_to_f32,
};
use visloc_basalt::RawU16Image;
use visloc_core::geometry::{Pose as CameraPose, SE3};
use visloc_core::types::{Camera, LandmarkDescriptorStore, VisualMap};
use visloc_io::colmap::read_colmap_text_model;
use visloc_io::descriptors::read_landmark_descriptors_txt;
use visloc_localization::{
    AllLandmarksSelector, BinaryLandmarkIndex, BinaryMatchGates, FixOutcome, LocalizationConfig,
    LocalizationPipeline, OdomCorrection, OdomCorrectionConfig, PackedDescriptors,
};
use visloc_vision::matching::HammingMatcher;
use visloc_vision::pnp::{GaussNewtonPoseRefiner, P3PGrunert};
use visloc_vision::ransac::PnPRansac;

use crate::msgs::{
    Covariance6, DiagnosticArray, DiagnosticStatus, Header, KeyValue, Pose, PoseWithCovariance,
    PoseWithCovarianceStamped, TfMessage, Time, Transform, TransformStamped, Vector3 as RosVector3,
};
use crate::vio::ros_pose;

/// Schema written by `examples/basalt_map/map_files.rs`.
pub const MAP_SCHEMA: &str = "visloc.basalt_localization_map.v1";

/// Tunables of the map localization node. The defaults are the ones measured
/// in `docs/basalt_map_localization.md`.
#[derive(Clone, Debug)]
pub struct MapLocalizeOptions {
    /// Detector budget; `None` uses the map's own (`map.json`).
    pub max_points: Option<usize>,
    pub max_hamming: u32,
    pub ratio: Option<f32>,
    /// Below 30 inliers wrong poses appear on EuRoC.
    pub min_inliers: usize,
    pub ransac_iterations: usize,
    pub reprojection_px: f64,
    /// Once the odometry has a map pose: search radius around each landmark's predicted
    /// projection; `None` always matches map-wide.
    pub projection_radius_px: Option<f64>,
    /// The prediction is a prior only this long and this far (odometry
    /// travel) after the correction's newest fix: after a long gap, a drifted
    /// prediction can steer the projection search to a wrong pose that
    /// agrees with it.
    pub projection_fresh_s: f64,
    pub projection_fresh_m: f64,
    /// Threads of the map-wide search.
    pub match_threads: usize,
    /// Map-wide search: landmarks closer than this to the best one count as
    /// the same point in the ratio test
    /// (`BinaryLandmarkIndex::with_duplicate_radius`); `None` is the plain
    /// ratio test.
    pub duplicate_radius_m: Option<f64>,
    pub correction: OdomCorrectionConfig,
    /// Odometry poses kept for pairing with fixes.
    pub odom_history_s: f64,
    /// Largest gap between two odometry poses a fix may be interpolated in.
    pub max_odom_gap_s: f64,
    /// Published 1-sigma position: `position_stddev_m +
    /// position_stddev_per_m * metres_since_fix` (fitted to the measured
    /// growth of the error with VIO travel since the last fix).
    pub position_stddev_m: f64,
    pub position_stddev_per_m: f64,
    pub orientation_stddev_rad: f64,
}

impl Default for MapLocalizeOptions {
    fn default() -> Self {
        Self {
            max_points: None,
            max_hamming: 70,
            ratio: Some(0.8),
            min_inliers: 30,
            ransac_iterations: 1000,
            reprojection_px: 3.0,
            projection_radius_px: Some(20.0),
            projection_fresh_s: 5.0,
            projection_fresh_m: 2.0,
            match_threads: 1,
            duplicate_radius_m: None,
            correction: OdomCorrectionConfig::default(),
            odom_history_s: 10.0,
            max_odom_gap_s: 0.2,
            position_stddev_m: 0.03,
            position_stddev_per_m: 0.025,
            orientation_stddev_rad: 0.02,
        }
    }
}

/// A map saved by `--save-map`, loaded for localization.
pub struct SavedBasaltMap {
    pub map: VisualMap,
    pub descriptors: LandmarkDescriptorStore,
    /// `T_imu_cam` per Basalt camera index.
    pub imu_from_camera: Vec<SE3>,
    /// Detector budget the map's keyframes were built with.
    pub max_points: usize,
}

fn se3_from_json(value: &serde_json::Value) -> Option<SE3> {
    let numbers = |key: &str| -> Option<Vec<f64>> {
        value
            .get(key)?
            .as_array()?
            .iter()
            .map(serde_json::Value::as_f64)
            .collect()
    };
    let [tx, ty, tz]: [f64; 3] = numbers("translation")?.try_into().ok()?;
    let [qw, qx, qy, qz]: [f64; 4] = numbers("quaternion_wxyz")?.try_into().ok()?;
    Some(SE3::new(
        UnitQuaternion::from_quaternion(NaQuaternion::new(qw, qx, qy, qz)),
        Vector3::new(tx, ty, tz),
    ))
}

/// Loads `<map_dir>/{map.json, sparse/, landmark_descriptors.txt}`.
pub fn load_basalt_map(map_dir: &Path) -> Result<SavedBasaltMap, String> {
    let summary_path = map_dir.join("map.json");
    let summary: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&summary_path)
            .map_err(|error| format!("{}: {error}", summary_path.display()))?,
    )
    .map_err(|error| format!("{}: {error}", summary_path.display()))?;
    if summary.get("schema").and_then(serde_json::Value::as_str) != Some(MAP_SCHEMA) {
        return Err(format!("{}: not a {MAP_SCHEMA} map", map_dir.display()));
    }
    let imu_from_camera = summary
        .get("cameras")
        .and_then(serde_json::Value::as_array)
        .ok_or("map.json: missing cameras")?
        .iter()
        .map(|camera| {
            camera
                .get("t_imu_cam")
                .and_then(se3_from_json)
                .ok_or_else(|| "map.json: camera without t_imu_cam".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let max_points = summary
        .get("feature_max_points")
        .and_then(serde_json::Value::as_u64)
        .ok_or("map.json: missing feature_max_points")? as usize;
    let sparse = map_dir.join("sparse");
    let map = read_colmap_text_model(&sparse)
        .map_err(|error| format!("{}: {error}", sparse.display()))?;
    let descriptors_path = map_dir.join("landmark_descriptors.txt");
    let descriptors = read_landmark_descriptors_txt(&descriptors_path)
        .map_err(|error| format!("{}: {error}", descriptors_path.display()))?;
    Ok(SavedBasaltMap {
        map,
        descriptors,
        imu_from_camera,
        max_points,
    })
}

type Pipeline = LocalizationPipeline<HammingMatcher, AllLandmarksSelector, PnPRansac<P3PGrunert>>;

/// How the 2D-3D matches of a localization attempt were found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matching {
    /// Every keypoint against every map landmark.
    Global,
    /// Each landmark against the keypoints near its projection from the
    /// pose predicted by the correction and the odometry.
    Projection,
}

impl Matching {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Projection => "projection",
        }
    }
}

/// One localization attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct Fix {
    /// The body (IMU) pose in the map frame, when accepted.
    pub map_from_body: Option<SE3>,
    /// The search of the final attempt; `None` when matching did not run.
    pub matching: Option<Matching>,
    pub keypoints: usize,
    pub matches: usize,
    pub inliers: usize,
    pub elapsed_ms: f64,
    pub failure: Option<String>,
}

impl Fix {
    /// An attempt that failed before matching.
    pub const fn failed(failure: String) -> Self {
        Self {
            map_from_body: None,
            matching: None,
            keypoints: 0,
            matches: 0,
            inliers: 0,
            elapsed_ms: 0.0,
            failure: Some(failure),
        }
    }
}

/// ORB features of one image, ready for [`MapLocalizer::localize_features`].
pub struct ImageFeatures {
    pub keypoints: Vec<nalgebra::Point2<f64>>,
    descriptors: PackedDescriptors,
    pub extract_ms: f64,
}

/// Localizes one camera's images against a [`SavedBasaltMap`].
pub struct MapLocalizer {
    map: SavedBasaltMap,
    index: BinaryLandmarkIndex,
    camera: Camera,
    camera_from_imu: SE3,
    max_points: usize,
    gates: BinaryMatchGates,
    projection_radius_px: Option<f64>,
    match_threads: usize,
    pipeline: Pipeline,
}

impl MapLocalizer {
    pub fn new(
        map: SavedBasaltMap,
        camera_index: u16,
        options: &MapLocalizeOptions,
    ) -> Result<Self, String> {
        let camera = map
            .map
            .cameras
            .get(&localization_camera_id(camera_index))
            .cloned()
            .ok_or_else(|| format!("camera {camera_index} is not in the map"))?;
        let camera_from_imu = map
            .imu_from_camera
            .get(usize::from(camera_index))
            .ok_or_else(|| format!("map.json has no extrinsics for camera {camera_index}"))?
            .inverse();
        let config = LocalizationConfig {
            ratio: options.ratio,
            ransac_iterations: options.ransac_iterations,
            reprojection_threshold: options.reprojection_px,
            min_inliers: options.min_inliers,
            ..LocalizationConfig::default()
        };
        let ransac = PnPRansac {
            pose_estimator: P3PGrunert,
            pose_refiner: Some(GaussNewtonPoseRefiner::default()),
            iterations: options.ransac_iterations,
            reprojection_threshold: options.reprojection_px,
            seed: 7,
            early_stop_min_iterations: 0,
            early_stop_inlier_ratio: None,
            confidence: Some(0.999),
        };
        // The pipeline only estimates and gates poses; matching goes through
        // the packed index with the same gates.
        let matcher = HammingMatcher {
            max_distance: options.max_hamming,
            ratio: options.ratio,
        };
        Ok(Self {
            index: BinaryLandmarkIndex::new(&map.map, &map.descriptors)
                .with_duplicate_radius(options.duplicate_radius_m.unwrap_or(0.0)),
            max_points: options.max_points.unwrap_or(map.max_points),
            map,
            camera,
            camera_from_imu,
            gates: BinaryMatchGates {
                max_distance: options.max_hamming,
                ratio: options.ratio,
            },
            projection_radius_px: options.projection_radius_px,
            match_threads: options.match_threads.max(1),
            pipeline: LocalizationPipeline::with_pose_estimator(
                matcher,
                AllLandmarksSelector,
                ransac,
                config,
            ),
        })
    }

    pub const fn map(&self) -> &SavedBasaltMap {
        &self.map
    }

    pub const fn camera(&self) -> &Camera {
        &self.camera
    }

    /// Localizes one raw image; `prior` is the body pose in the map
    /// predicted for it (see [`OdomCorrector::prior`]).
    pub fn localize(&self, image: &RawU16Image, prior: Option<&SE3>) -> Fix {
        let started = Instant::now();
        let mut fix = match self.extract(image) {
            Ok(features) => self.localize_features(&features, prior),
            Err(failure) => Fix::failed(failure),
        };
        fix.elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        fix
    }

    /// Runs the mapper's ORB frontend on one raw image; the error is the
    /// failure reason.
    pub fn extract(&self, image: &RawU16Image) -> Result<ImageFeatures, String> {
        let started = Instant::now();
        if (image.width(), image.height())
            != (self.camera.width as usize, self.camera.height as usize)
        {
            return Err(format!(
                "image is {}x{}, the map camera {}x{}",
                image.width(),
                image.height(),
                self.camera.width,
                self.camera.height
            ));
        }
        let corners = detect_keypoints_mapping(image, self.max_points);
        let angles = compute_angles(image, &corners, true);
        let descriptors = compute_descriptors(image, &corners, &angles)
            .map_err(|error| format!("ORB descriptors: {error}"))?;
        let descriptors = descriptors
            .iter()
            .map(orb_descriptor_to_f32)
            .collect::<Vec<_>>();
        Ok(ImageFeatures {
            descriptors: self.index.pack_query(&descriptors),
            keypoints: corners,
            extract_ms: started.elapsed().as_secs_f64() * 1e3,
        })
    }

    /// Features from keypoints and unpacked binary descriptors (one `0.0` /
    /// `1.0` per bit) extracted elsewhere.
    pub fn features(
        &self,
        keypoints: Vec<nalgebra::Point2<f64>>,
        descriptors: &[Vec<f32>],
    ) -> ImageFeatures {
        ImageFeatures {
            descriptors: self.index.pack_query(descriptors),
            keypoints,
            extract_ms: 0.0,
        }
    }

    /// `T_camera_imu` from the map's calibration.
    pub const fn camera_from_imu(&self) -> &SE3 {
        &self.camera_from_imu
    }

    /// Matches and solves one image's features, for a body that is the
    /// map's IMU. With a `prior` (and a projection radius) the projection
    /// search runs first and the map-wide one only if it fails.
    pub fn localize_features(&self, features: &ImageFeatures, prior: Option<&SE3>) -> Fix {
        self.localize_features_with(features, prior, &self.camera_from_imu)
    }

    /// [`Self::localize_features`] for a body `camera_from_body` away from
    /// the camera (e.g. a robot's `base_link`, from `/tf_static`); `prior`
    /// and the fix are that body's poses.
    pub fn localize_features_with(
        &self,
        features: &ImageFeatures,
        prior: Option<&SE3>,
        camera_from_body: &SE3,
    ) -> Fix {
        let started = Instant::now();
        let projection = prior.zip(self.projection_radius_px).map(|(prior, radius)| {
            let pose = CameraPose {
                world_to_camera: camera_from_body.compose(&prior.inverse()),
            };
            self.solve(
                Matching::Projection,
                self.index.match_projected(
                    &self.camera,
                    &pose,
                    &features.keypoints,
                    &features.descriptors,
                    radius,
                    self.gates,
                ),
                features,
                camera_from_body,
            )
        });
        let mut fix = match projection {
            Some(fix) if fix.map_from_body.is_some() => fix,
            _ => self.solve(
                Matching::Global,
                self.index.match_all(
                    &features.keypoints,
                    &features.descriptors,
                    self.gates,
                    self.match_threads,
                ),
                features,
                camera_from_body,
            ),
        };
        fix.elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        fix
    }

    fn solve(
        &self,
        matching: Matching,
        correspondences: visloc_localization::CorrespondenceSet,
        features: &ImageFeatures,
        camera_from_body: &SE3,
    ) -> Fix {
        let result = self
            .pipeline
            .localize_correspondences(&self.camera, correspondences);
        let map_from_body = (result.success)
            .then_some(result.pose.as_ref())
            .flatten()
            .map(|pose| pose.camera_to_world().compose(camera_from_body));
        Fix {
            failure: map_from_body.is_none().then(|| {
                result.failure_reason.as_ref().map_or_else(
                    || format!("{} inliers", result.inlier_count),
                    |reason| format!("{reason:?}"),
                )
            }),
            map_from_body,
            matching: Some(matching),
            keypoints: features.keypoints.len(),
            matches: result.match_count,
            inliers: result.inlier_count,
            elapsed_ms: 0.0,
        }
    }
}

/// Converts a ROS pose to an SE3 (`T_frame_child`).
pub fn se3_from_ros(pose: &Pose) -> SE3 {
    SE3::new(
        UnitQuaternion::from_quaternion(NaQuaternion::new(
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        )),
        Vector3::new(pose.position.x, pose.position.y, pose.position.z),
    )
}

/// What the odometry history knows about one stamp.
#[derive(Clone, Debug, PartialEq)]
pub enum OdomLookup {
    Pose(SE3),
    /// Newer than the newest odometry: may still arrive.
    NotYet,
    /// Older than the history, or inside a gap wider than allowed.
    Unavailable,
}

/// Recent odometry poses by stamp, for pairing with fixes.
#[derive(Clone, Debug)]
pub struct OdomBuffer {
    poses: VecDeque<(i64, SE3)>,
    history_ns: i64,
}

impl OdomBuffer {
    pub fn new(history_s: f64) -> Self {
        Self {
            poses: VecDeque::new(),
            history_ns: (history_s * 1e9) as i64,
        }
    }

    /// Adds a pose; out-of-order or repeated stamps are ignored.
    pub fn push(&mut self, stamp_ns: i64, pose: SE3) {
        if self.poses.back().is_some_and(|(last, _)| stamp_ns <= *last) {
            return;
        }
        self.poses.push_back((stamp_ns, pose));
        while self
            .poses
            .front()
            .is_some_and(|(first, _)| stamp_ns - first > self.history_ns)
        {
            self.poses.pop_front();
        }
    }

    /// The pose at `stamp_ns`: exact, or interpolated between neighbours at
    /// most `max_gap_ns` apart.
    pub fn lookup(&self, stamp_ns: i64, max_gap_ns: i64) -> OdomLookup {
        let Some((newest, _)) = self.poses.back() else {
            return OdomLookup::NotYet;
        };
        if stamp_ns > *newest {
            return OdomLookup::NotYet;
        }
        let index = self.poses.partition_point(|(stamp, _)| *stamp < stamp_ns);
        let (after_ns, after) = &self.poses[index];
        if *after_ns == stamp_ns {
            return OdomLookup::Pose(after.clone());
        }
        let Some((before_ns, before)) = index.checked_sub(1).map(|i| &self.poses[i]) else {
            return OdomLookup::Unavailable;
        };
        if after_ns - before_ns > max_gap_ns {
            return OdomLookup::Unavailable;
        }
        let alpha = (stamp_ns - before_ns) as f64 / (after_ns - before_ns) as f64;
        OdomLookup::Pose(SE3::new(
            before.rotation.slerp(&after.rotation, alpha),
            before.translation.lerp(&after.translation, alpha),
        ))
    }
}

/// An odometry pose expressed in the map frame.
#[derive(Clone, Debug, PartialEq)]
pub struct MapPose {
    pub stamp_ns: i64,
    pub map_from_odom: SE3,
    pub map_from_body: SE3,
    pub seconds_since_fix: f64,
    pub metres_since_fix: f64,
}

/// What happened to one fix in [`OdomCorrector`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixEvent {
    Applied {
        stamp_ns: i64,
        outcome: FixOutcome,
    },
    /// No odometry for its stamp (too old, or inside a gap).
    Dropped {
        stamp_ns: i64,
    },
}

/// Pairs fixes with odometry by stamp and keeps the [`OdomCorrection`].
#[derive(Clone, Debug)]
pub struct OdomCorrector {
    correction: OdomCorrection,
    odom: OdomBuffer,
    pending: VecDeque<(i64, SE3)>,
    max_gap_ns: i64,
    latest_odom: Option<(i64, SE3)>,
    latest_fix_ns: Option<i64>,
    fresh_ns: i64,
    fresh_m: f64,
}

impl OdomCorrector {
    pub fn new(options: &MapLocalizeOptions) -> Self {
        Self {
            correction: OdomCorrection::new(options.correction.clone()),
            odom: OdomBuffer::new(options.odom_history_s),
            pending: VecDeque::new(),
            max_gap_ns: (options.max_odom_gap_s * 1e9) as i64,
            latest_odom: None,
            latest_fix_ns: None,
            fresh_ns: (options.projection_fresh_s * 1e9) as i64,
            fresh_m: options.projection_fresh_m,
        }
    }

    /// Whether odometry poses have a map pose.
    pub const fn is_set(&self) -> bool {
        self.correction.is_set()
    }

    pub fn pending_fixes(&self) -> usize {
        self.pending.len()
    }

    /// Stamp of the newest localization added, applied or not.
    pub const fn latest_fix_ns(&self) -> Option<i64> {
        self.latest_fix_ns
    }

    /// Queues an accepted localization and applies every queued fix whose
    /// odometry is known.
    pub fn add_fix(&mut self, stamp_ns: i64, map_from_body: SE3) -> Vec<FixEvent> {
        self.latest_fix_ns = Some(self.latest_fix_ns.map_or(stamp_ns, |t| t.max(stamp_ns)));
        self.pending.push_back((stamp_ns, map_from_body));
        self.flush()
    }

    /// Adds an odometry pose, applies the fixes it unblocks, and returns the
    /// pose in the map frame once the correction is set.
    pub fn add_odom(
        &mut self,
        stamp_ns: i64,
        odom_from_body: SE3,
    ) -> (Vec<FixEvent>, Option<MapPose>) {
        self.odom.push(stamp_ns, odom_from_body.clone());
        if self
            .latest_odom
            .as_ref()
            .is_none_or(|(last, _)| stamp_ns > *last)
        {
            self.latest_odom = Some((stamp_ns, odom_from_body));
        }
        let events = self.flush();
        (events, self.latest_map_pose())
    }

    /// The body pose in the map at `stamp_ns` that the correction and the
    /// odometry predict, as a localization prior; [`OdomLookup::Unavailable`]
    /// before the correction is set, and once its newest fix is more than
    /// `projection_fresh_s` / `projection_fresh_m` behind.
    pub fn prior(&self, stamp_ns: i64) -> OdomLookup {
        let (Some(map_from_odom), Some((fix_ns, fix_position))) =
            (self.correction.map_from_odom(), self.correction.last_fix())
        else {
            return OdomLookup::Unavailable;
        };
        if stamp_ns - fix_ns > self.fresh_ns {
            return OdomLookup::Unavailable;
        }
        match self.odom.lookup(stamp_ns, self.max_gap_ns) {
            OdomLookup::Pose(odom_from_body)
                if (odom_from_body.translation - fix_position).norm() > self.fresh_m =>
            {
                OdomLookup::Unavailable
            }
            OdomLookup::Pose(odom_from_body) => {
                OdomLookup::Pose(map_from_odom.compose(&odom_from_body))
            }
            other => other,
        }
    }

    /// The newest odometry pose in the map frame, once the correction is set.
    pub fn latest_map_pose(&self) -> Option<MapPose> {
        let (stamp_ns, odom_from_body) = self.latest_odom.as_ref()?;
        let map_from_odom = self.correction.map_from_odom()?;
        let (fix_ns, fix_position) = self.correction.last_fix()?;
        Some(MapPose {
            stamp_ns: *stamp_ns,
            map_from_body: map_from_odom.compose(odom_from_body),
            map_from_odom,
            seconds_since_fix: (*stamp_ns - fix_ns) as f64 * 1e-9,
            metres_since_fix: (odom_from_body.translation - fix_position).norm(),
        })
    }

    fn flush(&mut self) -> Vec<FixEvent> {
        let mut events = Vec::new();
        while let Some((stamp_ns, _)) = self.pending.front() {
            let stamp_ns = *stamp_ns;
            match self.odom.lookup(stamp_ns, self.max_gap_ns) {
                OdomLookup::NotYet => break,
                OdomLookup::Unavailable => {
                    self.pending.pop_front();
                    events.push(FixEvent::Dropped { stamp_ns });
                }
                OdomLookup::Pose(odom_from_body) => {
                    let (_, map_from_body) = self.pending.pop_front().expect("front exists");
                    let outcome =
                        self.correction
                            .add_fix(stamp_ns, &map_from_body, &odom_from_body);
                    events.push(FixEvent::Applied { stamp_ns, outcome });
                }
            }
        }
        events
    }
}

/// Which received images to localize: every `every`-th once the odometry
/// has a map pose, every `initial_every`-th before. No frame has a map pose
/// until two agreeing fixes set the correction, so the first fixes are
/// worth every image. That holds within `initial_window_s` of the first
/// image or of the latest localization; after a longer stretch without one
/// the map likely does not cover the place, and every `every`-th image is
/// tried until one localizes again.
#[derive(Clone, Debug)]
pub struct ImageCadence {
    every: u64,
    initial_every: u64,
    initial_window_ns: i64,
    first_ns: Option<i64>,
    received: u64,
}

impl ImageCadence {
    pub fn new(every: u64, initial_every: u64, initial_window_s: f64) -> Self {
        Self {
            every: every.max(1),
            initial_every: initial_every.max(1),
            initial_window_ns: (initial_window_s.max(0.0) * 1e9) as i64,
            first_ns: None,
            received: 0,
        }
    }

    /// Counts a received image stamped `stamp_ns`; whether to localize it,
    /// given whether the odometry has a map pose and the stamp of the
    /// latest localization ([`OdomCorrector::latest_fix_ns`]).
    pub fn take(&mut self, stamp_ns: i64, has_map_pose: bool, latest_fix_ns: Option<i64>) -> bool {
        let first_ns = *self.first_ns.get_or_insert(stamp_ns);
        let since = stamp_ns - latest_fix_ns.unwrap_or(first_ns);
        let every = if has_map_pose || since > self.initial_window_ns {
            self.every
        } else {
            self.initial_every
        };
        let take = self.received.is_multiple_of(every);
        self.received += 1;
        take
    }
}

/// `/tf` with `map_frame -> odom_frame` from a map pose.
pub fn map_to_odom_tf(pose: &MapPose, map_frame: &str, odom_frame: &str) -> TfMessage {
    let ros = ros_pose(&pose.map_from_odom);
    TfMessage {
        transforms: vec![TransformStamped {
            header: Header::new(Time::from_nanos(pose.stamp_ns), map_frame),
            child_frame_id: odom_frame.into(),
            transform: Transform {
                translation: RosVector3 {
                    x: ros.position.x,
                    y: ros.position.y,
                    z: ros.position.z,
                },
                rotation: ros.orientation,
            },
        }],
    }
}

/// The body pose in the map frame, with a position covariance that grows
/// with the VIO travel since the last fix.
pub fn map_pose_msg(
    pose: &MapPose,
    map_frame: &str,
    options: &MapLocalizeOptions,
) -> PoseWithCovarianceStamped {
    let position_stddev =
        options.position_stddev_m + options.position_stddev_per_m * pose.metres_since_fix;
    PoseWithCovarianceStamped {
        header: Header::new(Time::from_nanos(pose.stamp_ns), map_frame),
        pose: PoseWithCovariance {
            pose: ros_pose(&pose.map_from_body),
            covariance: Covariance6::diagonal(
                position_stddev.powi(2),
                options.orientation_stddev_rad.powi(2),
            ),
        },
    }
}

pub fn fix_outcome_name(outcome: FixOutcome) -> &'static str {
    match outcome {
        FixOutcome::Pending => "pending",
        FixOutcome::Initialized => "initialized",
        FixOutcome::Accepted => "accepted",
        FixOutcome::Rejected => "rejected",
        FixOutcome::Reset => "reset",
    }
}

/// `diagnostic_msgs/DiagnosticArray` for one localization attempt.
pub fn map_diagnostics_msg(
    fix: &Fix,
    stamp: Time,
    map_pose: Option<&MapPose>,
    pending_fixes: usize,
    node_name: &str,
) -> DiagnosticArray {
    let value = |key: &str, value: String| KeyValue {
        key: key.into(),
        value,
    };
    let (level, message) = match (map_pose, &fix.map_from_body) {
        (Some(_), Some(_)) => (
            DiagnosticStatus::OK,
            format!("localized with {} inliers", fix.inliers),
        ),
        (Some(_), None) => (
            DiagnosticStatus::OK,
            "map pose from the odometry; this image did not localize".to_owned(),
        ),
        (None, _) => (
            DiagnosticStatus::WARN,
            "no map pose yet: waiting for two agreeing fixes".to_owned(),
        ),
    };
    DiagnosticArray {
        header: Header::new(stamp, ""),
        status: vec![DiagnosticStatus {
            level,
            name: format!("{node_name}: map localization"),
            message,
            hardware_id: String::new(),
            values: vec![
                value("localized", fix.map_from_body.is_some().to_string()),
                value("inliers", fix.inliers.to_string()),
                value("matches", fix.matches.to_string()),
                value("keypoints", fix.keypoints.to_string()),
                value(
                    "matching",
                    fix.matching.map_or("", Matching::name).to_owned(),
                ),
                value("processing_ms", format!("{:.2}", fix.elapsed_ms)),
                value("failure", fix.failure.clone().unwrap_or_default()),
                value("map_pose", map_pose.is_some().to_string()),
                value(
                    "seconds_since_fix",
                    map_pose.map_or_else(String::new, |p| format!("{:.3}", p.seconds_since_fix)),
                ),
                value(
                    "metres_since_fix",
                    map_pose.map_or_else(String::new, |p| format!("{:.3}", p.metres_since_fix)),
                ),
                value("pending_fixes", pending_fixes.to_string()),
            ],
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;

    fn pose(x: f64, yaw: f64) -> SE3 {
        SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, yaw),
            Vector3::new(x, 0.5 * x, 0.1),
        )
    }

    #[test]
    fn odom_lookup_is_exact_interpolated_pending_or_unavailable() {
        let mut odom = OdomBuffer::new(1.0);
        odom.push(100 * MS, pose(0.0, 0.0));
        odom.push(150 * MS, pose(1.0, 0.2));
        odom.push(140 * MS, pose(9.0, 0.0)); // out of order: ignored
        odom.push(600 * MS, pose(2.0, 0.2));

        assert_eq!(
            odom.lookup(150 * MS, 100 * MS),
            OdomLookup::Pose(pose(1.0, 0.2))
        );
        let OdomLookup::Pose(mid) = odom.lookup(125 * MS, 100 * MS) else {
            panic!("interpolates");
        };
        assert!((mid.translation - pose(0.5, 0.1).translation).norm() < 1e-12);
        assert!(mid.rotation.angle_to(&pose(0.5, 0.1).rotation) < 1e-12);
        assert_eq!(odom.lookup(601 * MS, 100 * MS), OdomLookup::NotYet);
        assert_eq!(odom.lookup(300 * MS, 100 * MS), OdomLookup::Unavailable); // 450 ms gap
        assert_eq!(odom.lookup(50 * MS, 100 * MS), OdomLookup::Unavailable); // before history

        // History trimming: 1 s after 600 ms drops the 100 ms entry.
        odom.push(1_120 * MS, pose(3.0, 0.2));
        assert_eq!(odom.lookup(100 * MS, 100 * MS), OdomLookup::Unavailable);
    }

    /// The map frame is the odometry frame moved by `offset`.
    fn offset() -> SE3 {
        SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.7),
            Vector3::new(2.0, -1.0, 0.3),
        )
    }

    #[test]
    fn fixes_wait_for_their_odometry_and_correct_the_trajectory() {
        let options = MapLocalizeOptions::default();
        let mut core = OdomCorrector::new(&options);
        let stamps = (0..6).map(|i| i64::from(i) * 50 * MS).collect::<Vec<_>>();
        let odom = |i: usize| pose(i as f64 * 0.1, i as f64 * 0.05);

        // Odometry for frame 0 first, then a fix for frame 0: applied at once.
        let (events, map_pose) = core.add_odom(stamps[0], odom(0));
        assert!(events.is_empty() && map_pose.is_none());
        let events = core.add_fix(stamps[0], offset().compose(&odom(0)));
        assert_eq!(
            events,
            vec![FixEvent::Applied {
                stamp_ns: stamps[0],
                outcome: FixOutcome::Pending
            }]
        );

        // A fix for frame 1 before its odometry: held until the odometry comes.
        assert!(core
            .add_fix(stamps[1], offset().compose(&odom(1)))
            .is_empty());
        assert_eq!(core.pending_fixes(), 1);
        let (events, map_pose) = core.add_odom(stamps[1], odom(1));
        assert_eq!(
            events,
            vec![FixEvent::Applied {
                stamp_ns: stamps[1],
                outcome: FixOutcome::Initialized
            }]
        );
        let map_pose = map_pose.expect("a map pose after two agreeing fixes");
        assert!((map_pose.map_from_odom.translation - offset().translation).norm() < 1e-9);
        assert_eq!(map_pose.seconds_since_fix, 0.0);

        // Later odometry without fixes is reported in the map frame.
        for (i, stamp) in stamps.iter().enumerate().skip(2) {
            let (_, map_pose) = core.add_odom(*stamp, odom(i));
            let map_pose = map_pose.expect("still a map pose");
            let expected = offset().compose(&odom(i));
            assert!((map_pose.map_from_body.translation - expected.translation).norm() < 1e-9);
            assert!((map_pose.seconds_since_fix - (i - 1) as f64 * 0.05).abs() < 1e-9);
            assert!(map_pose.metres_since_fix > 0.0);
        }

        // The prediction is a prior while the newest fix is recent and near.
        let prior = |core: &OdomCorrector, stamp: i64| match core.prior(stamp) {
            OdomLookup::Pose(pose) => Some(pose),
            _ => None,
        };
        let expected = offset().compose(&odom(5));
        let near = prior(&core, stamps[5]).expect("fresh prior");
        assert!((near.translation - expected.translation).norm() < 1e-9);
        let mut late = OdomCorrector::new(&MapLocalizeOptions {
            projection_fresh_s: 0.1,
            ..MapLocalizeOptions::default()
        });
        let mut far = OdomCorrector::new(&MapLocalizeOptions {
            projection_fresh_m: 0.3,
            ..MapLocalizeOptions::default()
        });
        for core in [&mut late, &mut far] {
            for (i, stamp) in stamps.iter().enumerate() {
                core.add_odom(*stamp, odom(i));
            }
            core.add_fix(stamps[0], offset().compose(&odom(0)));
            core.add_fix(stamps[1], offset().compose(&odom(1)));
            assert!(core.is_set());
            assert!(prior(core, stamps[2]).is_some());
        }
        // 0.15 s after the newest fix; 0.22 m and 0.45 m of travel from it.
        assert!(prior(&late, stamps[4]).is_none());
        assert!(prior(&far, stamps[3]).is_some());
        assert!(prior(&far, stamps[5]).is_none());

        // A fix older than the odometry history is dropped, not applied.
        let mut stale = OdomCorrector::new(&MapLocalizeOptions {
            odom_history_s: 0.1,
            ..MapLocalizeOptions::default()
        });
        for (i, stamp) in stamps.iter().enumerate() {
            stale.add_odom(*stamp, odom(i));
        }
        assert_eq!(
            stale.add_fix(stamps[0], offset().compose(&odom(0))),
            vec![FixEvent::Dropped {
                stamp_ns: stamps[0]
            }]
        );
    }

    #[test]
    fn every_image_is_localized_until_the_first_map_pose() {
        const S: i64 = 1_000_000_000;
        // Images at 10 Hz from `start_s`; how many of them are taken.
        let taken = |cadence: &mut ImageCadence,
                     start_s: f64,
                     images: usize,
                     has_map_pose: bool,
                     latest_fix_ns: Option<i64>| {
            (0..images)
                .filter(|&i| {
                    let stamp_ns = ((start_s + i as f64 * 0.1) * S as f64) as i64;
                    cadence.take(stamp_ns, has_map_pose, latest_fix_ns)
                })
                .count()
        };
        let mut cadence = ImageCadence::new(5, 1, 30.0);
        assert_eq!(taken(&mut cadence, 0.0, 7, false, None), 7);
        // A map pose from the 8th image on: the 11th and 16th of 8..=20.
        assert_eq!(taken(&mut cadence, 0.7, 13, true, None), 2);
        // 30 s without a localization: every 5th; after one, every image.
        let mut cadence = ImageCadence::new(5, 1, 30.0);
        assert_eq!(taken(&mut cadence, 10.0, 301, false, None), 301);
        assert_eq!(taken(&mut cadence, 40.1, 10, false, None), 2);
        assert_eq!(taken(&mut cadence, 41.1, 10, false, Some(41 * S)), 10);
        assert_eq!(
            taken(&mut ImageCadence::new(5, 5, 30.0), 0.0, 10, false, None),
            2
        );
        assert_eq!(
            taken(&mut ImageCadence::new(0, 0, 30.0), 0.0, 3, true, None),
            3
        );
    }

    /// A saved map in the `--save-map` layout: one fisheye camera, a grid of
    /// landmarks seen by one keyframe, each with its own descriptor.
    fn write_saved_map(dir: &Path, imu_from_camera: &SE3) -> (Camera, Vec<nalgebra::Point3<f64>>) {
        use visloc_core::geometry::Pose as CameraPose;
        use visloc_core::types::{CameraModel, Frame, Keyframe, Landmark, Observation};
        let camera = Camera {
            id: 1,
            model: CameraModel::OpenCvFisheye,
            width: 752,
            height: 480,
            params: vec![461.2, 459.9, 367.2, 248.4, 0.0047, -0.005, -0.0009, 0.00006],
        };
        let points = (0..120)
            .map(|i| {
                nalgebra::Point3::new(
                    -2.0 + 0.35 * (i % 12) as f64,
                    -1.2 + 0.3 * (i / 12) as f64,
                    3.0 + 0.5 * ((i * 7) % 5) as f64,
                )
            })
            .collect::<Vec<_>>();
        let mut map = VisualMap::new();
        map.cameras.insert(1, camera.clone());
        let mut frame = Frame::new(1, 1);
        frame.pose = Some(CameraPose {
            world_to_camera: SE3::identity(),
        });
        let mut keyframe = Keyframe {
            frame,
            observations: Vec::new(),
        };
        let mut descriptors = LandmarkDescriptorStore::new();
        for (index, point) in points.iter().enumerate() {
            let xy = camera.project(point).expect("visible");
            keyframe.frame.keypoints.push(xy);
            let observation = Observation {
                frame_id: 1,
                landmark_id: index as u64,
                keypoint_index: index,
                xy,
            };
            keyframe.observations.push(observation.clone());
            let mut landmark = Landmark::new(index as u64, *point);
            landmark.observations.push(observation);
            map.landmarks.insert(index as u64, landmark);
            descriptors.insert(index as u64, descriptor(index));
        }
        map.keyframes.insert(1, keyframe);
        visloc_io::colmap::write_colmap_text_model(&map, dir.join("sparse")).expect("model");
        visloc_io::descriptors::write_landmark_descriptors_txt(
            &descriptors,
            dir.join("landmark_descriptors.txt"),
        )
        .expect("descriptors");
        let q = imu_from_camera.rotation.quaternion();
        let t = imu_from_camera.translation;
        let summary = serde_json::json!({
            "schema": MAP_SCHEMA,
            "feature_max_points": 800,
            "cameras": [{"camera_id": 1, "t_imu_cam": {
                "translation": [t.x, t.y, t.z], "quaternion_wxyz": [q.w, q.i, q.j, q.k]}}],
        });
        fs::write(dir.join("map.json"), summary.to_string()).expect("map.json");
        (camera, points)
    }

    fn descriptor(seed: usize) -> Vec<f32> {
        let mut state = (seed as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        (0..256)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 1) as f32
            })
            .collect()
    }

    #[test]
    fn a_saved_map_localizes_a_new_viewpoint_as_a_body_pose() {
        let imu_from_camera = SE3::new(
            UnitQuaternion::from_euler_angles(-1.5, 0.02, 1.55),
            Vector3::new(-0.02, -0.06, 0.01),
        );
        let dir = std::env::temp_dir().join(format!("visloc_ros2_map_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("dir");
        let (camera, points) = write_saved_map(&dir, &imu_from_camera);
        let map = load_basalt_map(&dir).expect("loads");
        fs::remove_dir_all(&dir).expect("cleanup");
        assert_eq!(map.max_points, 800);
        assert_eq!(map.map.landmarks.len(), points.len());
        assert!(load_basalt_map(Path::new("/nonexistent/map")).is_err());

        let localizer =
            MapLocalizer::new(map, 0, &MapLocalizeOptions::default()).expect("camera 0");
        assert!(MapLocalizer::new(
            load_basalt_map_for_error_case(),
            1,
            &MapLocalizeOptions::default()
        )
        .is_err());

        // Body pose in the map; the query camera sits at body * imu_from_camera.
        let map_from_body = SE3::new(
            UnitQuaternion::from_euler_angles(0.05, -0.03, 0.06),
            Vector3::new(0.15, 0.2, -0.1),
        )
        .compose(&imu_from_camera.inverse());
        let camera_from_map = map_from_body.compose(&imu_from_camera).inverse();
        let (mut keypoints, mut descriptors) = (Vec::new(), Vec::new());
        for (index, point) in points.iter().enumerate() {
            let xy = camera
                .project(&camera_from_map.transform_point(point))
                .expect("visible");
            if index % 5 == 0 {
                keypoints.push(nalgebra::Point2::new(xy.x + 37.0, xy.y - 21.0));
                descriptors.push(descriptor(1000 + index));
            } else {
                keypoints.push(xy);
                descriptors.push(descriptor(index));
            }
        }
        let features = localizer.features(keypoints, &descriptors);
        let fix = localizer.localize_features(&features, None);
        assert_eq!(fix.matching, Some(Matching::Global));
        assert!(fix.inliers >= 90, "{fix:?}");
        let estimate = fix.map_from_body.expect("localized");
        assert!((estimate.translation - map_from_body.translation).norm() < 1e-3);
        assert!(estimate.rotation.angle_to(&map_from_body.rotation) < 1e-3);

        // With a prior a few centimetres and a degree off, the projection
        // search finds the same pose.
        let prior = map_from_body.compose(&SE3::new(
            UnitQuaternion::from_euler_angles(0.01, -0.01, 0.005),
            Vector3::new(0.03, -0.02, 0.01),
        ));
        let fix = localizer.localize_features(&features, Some(&prior));
        assert_eq!(fix.matching, Some(Matching::Projection));
        assert!(fix.inliers >= 90, "{fix:?}");
        let estimate = fix.map_from_body.expect("localized");
        assert!((estimate.translation - map_from_body.translation).norm() < 1e-3);

        // A prior far off finds nothing nearby; the map-wide search recovers.
        let wrong = map_from_body.compose(&SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.6),
            Vector3::zeros(),
        ));
        let fix = localizer.localize_features(&features, Some(&wrong));
        assert_eq!(fix.matching, Some(Matching::Global));
        assert!(fix.map_from_body.is_some(), "{fix:?}");

        // A wrong-sized image is reported, not localized.
        let image = RawU16Image::new(64, 48, vec![0; 64 * 48]).expect("image");
        let fix = localizer.localize(&image, None);
        assert!(fix.map_from_body.is_none());
        assert!(fix.failure.expect("reason").contains("64x48"));
    }

    /// A map whose only camera has index 0, to check a missing camera index.
    fn load_basalt_map_for_error_case() -> SavedBasaltMap {
        SavedBasaltMap {
            map: VisualMap::new(),
            descriptors: LandmarkDescriptorStore::new(),
            imu_from_camera: vec![SE3::identity()],
            max_points: 800,
        }
    }

    #[test]
    fn published_covariance_grows_with_travel_since_the_fix() {
        let options = MapLocalizeOptions::default();
        let mut pose = MapPose {
            stamp_ns: 5 * MS,
            map_from_odom: offset(),
            map_from_body: offset(),
            seconds_since_fix: 0.0,
            metres_since_fix: 0.0,
        };
        let near = map_pose_msg(&pose, "map", &options).pose.covariance.0[0];
        pose.metres_since_fix = 10.0;
        let far = map_pose_msg(&pose, "map", &options).pose.covariance.0[0];
        assert!((near - 0.03_f64.powi(2)).abs() < 1e-12);
        assert!((far - 0.28_f64.powi(2)).abs() < 1e-12);

        let tf = map_to_odom_tf(&pose, "map", "odom");
        assert_eq!(tf.transforms[0].header.frame_id, "map");
        assert_eq!(tf.transforms[0].child_frame_id, "odom");
        assert_eq!(tf.transforms[0].header.stamp, Time::from_nanos(5 * MS));
    }
}
