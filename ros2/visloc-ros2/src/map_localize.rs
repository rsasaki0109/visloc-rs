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
use visloc_basalt::contrast::LocalContrast;
use visloc_basalt::mapper::features::{detect_keypoints_mapping_with, MAPPING_QUALITY_LEVEL};
use visloc_basalt::mapper::{
    compute_angles, compute_descriptors, localization_camera_id, orb_descriptor_to_f32,
};
use visloc_basalt::smoothing::GaussianSmoothing;
use visloc_basalt::RawU16Image;
use visloc_core::geometry::{Pose as CameraPose, SE3};
use visloc_core::types::{Camera, LandmarkDescriptorStore, LocalizationFailureReason, VisualMap};
use visloc_io::colmap::read_colmap_text_model;
use visloc_io::descriptors::{read_landmark_descriptor_rows, read_landmark_descriptors_txt};
use visloc_localization::{
    apply_inlier_spread_gate, fixes_agree_under_a_source, AllLandmarksSelector,
    BinaryLandmarkIndex, BinaryMatchGates, FixOutcome, ImageGrid, LocalizationConfig,
    LocalizationPipeline, MapFusion, MapFusionConfig, OdomCorrection, OdomCorrectionConfig,
    OdometryNoise, PackedDescriptors,
};
use visloc_vision::matching::HammingMatcher;
use visloc_vision::pnp::{GaussNewtonPoseRefiner, P3PSolver, P3P};
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
    /// A map-wide localization with this many inliers, but fewer than
    /// `min_inliers`, is a weak fix (`OdomCorrection::add_weak_fix`): it
    /// helps only to set the first correction, which then takes three
    /// agreeing fixes. At or above `min_inliers`: off.
    pub weak_min_inliers: usize,
    pub ransac_iterations: usize,
    /// The P3P solver inside RANSAC. Grunert (the default) is what every
    /// measurement so far used; Lambda Twist is the more accurate one when
    /// most landmarks in view lie on walls (see `P3PSolver`).
    pub p3p_solver: P3PSolver,
    /// Rounds of re-collecting the inliers of the refined pose and refining
    /// again (`PnPRansac::inlier_refits`); 0 refines once.
    pub inlier_refits: usize,
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
    /// After an odometry reset (`OdomCorrector::reset`), the body's last
    /// map pose is the projection prior for the new odometry frame, for
    /// this long after that pose (and within `projection_fresh_m` of travel
    /// in the new frame): a VIO reset by its lost-tracking monitor stops the
    /// robot, so it is still about there. Fixes found with it pass the same
    /// gates as any; 0 turns it off (map-wide matching only until the
    /// correction is set again).
    pub reset_prior_s: f64,
    /// After this many full fixes in a row are rejected (without agreeing
    /// among themselves, which resets the correction instead), the odometry
    /// has left the map: [`OdomCorrector::recovery_requests`] counts one
    /// more, for the VIO to reset. 0 (the default) turns it off.
    pub recover_after: usize,
    /// Threads of the map-wide search (and of the radius search).
    pub match_threads: usize,
    /// With a prior whose projection search fails: match only the landmarks
    /// within this many metres of the prior's camera centre before going
    /// map-wide (`None`: straight to map-wide). For dense maps, where the
    /// map-wide search is slow.
    pub prior_radius_m: Option<f64>,
    /// With a prior, never match map-wide: the robot is near the prediction,
    /// and a match anywhere else would be wrong. Fixes come from the
    /// projection and radius searches only until the prior lapses.
    pub no_global_with_prior: bool,
    /// Map-wide search: landmarks closer than this to the best one count as
    /// the same point in the ratio test
    /// (`BinaryLandmarkIndex::with_duplicate_radius`); `None` is the plain
    /// ratio test.
    pub duplicate_radius_m: Option<f64>,
    /// Extra descriptors per landmark matched at most (the map's
    /// `landmark_descriptors_extra.txt`, from the views that differ most
    /// from the main descriptor); 0, the default, matches the main one
    /// only.
    pub extra_descriptors: usize,
    /// A localization whose inliers occupy fewer than this many cells of a
    /// 4 x 4 image grid is turned down as if it had too few inliers (it can
    /// still be weak): inliers bunched on one patch, the rest of the view
    /// hidden by people, give poorly conditioned poses
    /// (`docs/dynamic_obstacle_localization.md`). 0, the default: off.
    pub min_inlier_cells: u32,
    pub correction: OdomCorrectionConfig,
    /// With a second odometry (e.g. wheels beside a VIO): fuse both with
    /// the fixes in `MapFusion` and publish its map pose. `None`, the
    /// default: the correction of the one odometry, as before.
    pub fusion: Option<FusionOptions>,
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
            weak_min_inliers: 15,
            ransac_iterations: 1000,
            p3p_solver: P3PSolver::Grunert,
            inlier_refits: 0,
            reprojection_px: 3.0,
            projection_radius_px: Some(20.0),
            projection_fresh_s: 5.0,
            projection_fresh_m: 2.0,
            reset_prior_s: 30.0,
            recover_after: 0,
            match_threads: 1,
            prior_radius_m: None,
            no_global_with_prior: false,
            duplicate_radius_m: None,
            extra_descriptors: 0,
            min_inlier_cells: 0,
            correction: OdomCorrectionConfig::default(),
            fusion: None,
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
    /// `landmark_descriptors_extra.txt`: more descriptors per landmark (ids
    /// repeat), empty when the file is missing.
    pub extra_descriptors: Vec<(u64, Vec<f32>)>,
    /// `T_imu_cam` per Basalt camera index.
    pub imu_from_camera: Vec<SE3>,
    /// Detector budget the map's keyframes were built with.
    pub max_points: usize,
    /// Local contrast equalization the map's keyframes were built with
    /// (`map.json`'s `feature_local_contrast`), applied to every image
    /// before extraction.
    pub local_contrast: Option<LocalContrast>,
    /// The blur the map's keyframes were built with (`map.json`'s
    /// `feature_smoothing_sigma`), applied after the equalization.
    pub feature_smoothing: Option<GaussianSmoothing>,
    /// The ORB detector's quality level the map's keyframes were built with
    /// (`map.json`'s `feature_quality_level`; upstream's 0.01 when absent).
    pub feature_quality_level: Option<f32>,
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
    let local_contrast = match summary
        .get("feature_local_contrast")
        .filter(|value| !value.is_null())
    {
        None => None,
        Some(value) => {
            let tiles = value.get("tiles").and_then(serde_json::Value::as_u64);
            let clip_limit = value.get("clip_limit").and_then(serde_json::Value::as_f64);
            match (tiles, clip_limit) {
                (Some(tiles), Some(clip_limit)) if tiles > 0 && clip_limit > 0.0 => {
                    Some(LocalContrast {
                        tiles: tiles as usize,
                        clip_limit,
                    })
                }
                _ => return Err(format!("map.json: bad feature_local_contrast {value}")),
            }
        }
    };
    let feature_smoothing = match summary
        .get("feature_smoothing_sigma")
        .filter(|value| !value.is_null())
    {
        None => None,
        Some(value) => match value.as_f64() {
            Some(sigma) if sigma.is_finite() && sigma > 0.0 => Some(GaussianSmoothing { sigma }),
            _ => return Err(format!("map.json: bad feature_smoothing_sigma {value}")),
        },
    };
    let feature_quality_level = match summary
        .get("feature_quality_level")
        .filter(|value| !value.is_null())
    {
        None => None,
        Some(value) => match value.as_f64() {
            Some(level) if level.is_finite() && level > 0.0 && level <= 1.0 => Some(level as f32),
            _ => return Err(format!("map.json: bad feature_quality_level {value}")),
        },
    };
    let sparse = map_dir.join("sparse");
    let map = read_colmap_text_model(&sparse)
        .map_err(|error| format!("{}: {error}", sparse.display()))?;
    let descriptors_path = map_dir.join("landmark_descriptors.txt");
    let descriptors = read_landmark_descriptors_txt(&descriptors_path)
        .map_err(|error| format!("{}: {error}", descriptors_path.display()))?;
    let extra_path = map_dir.join("landmark_descriptors_extra.txt");
    let extra_descriptors = if extra_path.exists() {
        read_landmark_descriptor_rows(&extra_path)
            .map_err(|error| format!("{}: {error}", extra_path.display()))?
    } else {
        Vec::new()
    };
    Ok(SavedBasaltMap {
        map,
        descriptors,
        extra_descriptors,
        imu_from_camera,
        max_points,
        local_contrast,
        feature_smoothing,
        feature_quality_level,
    })
}

type Pipeline = LocalizationPipeline<HammingMatcher, AllLandmarksSelector, PnPRansac<P3P>>;

/// How the 2D-3D matches of a localization attempt were found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matching {
    /// Every keypoint against every map landmark.
    Global,
    /// Every keypoint against the landmarks near the prior's camera.
    Radius,
    /// Each landmark against the keypoints near its projection from the
    /// pose predicted by the correction and the odometry.
    Projection,
}

impl Matching {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Radius => "radius",
            Self::Projection => "projection",
        }
    }
}

/// One localization attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct Fix {
    /// The body (IMU) pose in the map frame, when accepted.
    pub map_from_body: Option<SE3>,
    /// The body pose of a weak localization: map-wide (or, with a weak gate
    /// set, by projection), with `weak_min_inliers` inliers or more but too
    /// few to be accepted.
    pub weak_map_from_body: Option<SE3>,
    /// The search of the final attempt; `None` when matching did not run.
    pub matching: Option<Matching>,
    pub keypoints: usize,
    pub matches: usize,
    pub inliers: usize,
    /// Each PnP inlier of an accepted localization: the image keypoint
    /// (pixels) and the map landmark it matched. Empty otherwise.
    pub inlier_matches: Vec<(nalgebra::Point2<f64>, u64)>,
    pub elapsed_ms: f64,
    pub failure: Option<String>,
}

impl Fix {
    /// An attempt that failed before matching.
    pub const fn failed(failure: String) -> Self {
        Self {
            map_from_body: None,
            weak_map_from_body: None,
            matching: None,
            keypoints: 0,
            matches: 0,
            inliers: 0,
            inlier_matches: Vec::new(),
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
    weak_min_inliers: Option<usize>,
    /// Whether a projection-matched localization can be weak: only when the
    /// correction uses weak fixes after it is set (`weak_gate_m`).
    weak_projected: bool,
    match_threads: usize,
    prior_radius_m: Option<f64>,
    no_global_with_prior: bool,
    /// `MapLocalizeOptions::min_inlier_cells` and its grid.
    min_inlier_cells: u32,
    grid: ImageGrid,
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
            pose_estimator: P3P::new(options.p3p_solver),
            pose_refiner: Some(GaussNewtonPoseRefiner::default()),
            iterations: options.ransac_iterations,
            reprojection_threshold: options.reprojection_px,
            seed: 7,
            early_stop_min_iterations: 0,
            early_stop_inlier_ratio: None,
            confidence: Some(0.999),
            inlier_refits: options.inlier_refits,
        };
        // The pipeline only estimates and gates poses; matching goes through
        // the packed index with the same gates.
        let matcher = HammingMatcher {
            max_distance: options.max_hamming,
            ratio: options.ratio,
        };
        Ok(Self {
            index: BinaryLandmarkIndex::new(&map.map, &map.descriptors)
                .with_extra_descriptors(&map.extra_descriptors, options.extra_descriptors)
                .with_duplicate_radius(options.duplicate_radius_m.unwrap_or(0.0)),
            max_points: options.max_points.unwrap_or(map.max_points),
            map,
            min_inlier_cells: options.min_inlier_cells,
            grid: ImageGrid::new(&camera, 4),
            camera,
            camera_from_imu,
            gates: BinaryMatchGates {
                max_distance: options.max_hamming,
                ratio: options.ratio,
            },
            projection_radius_px: options.projection_radius_px,
            weak_min_inliers: Some(options.weak_min_inliers)
                .filter(|&weak| weak < options.min_inliers),
            weak_projected: options.correction.weak_gate_m > 0.0,
            match_threads: options.match_threads.max(1),
            prior_radius_m: options.prior_radius_m,
            no_global_with_prior: options.no_global_with_prior,
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
    /// Extra descriptors matched besides each landmark's main one.
    pub fn extra_descriptor_count(&self) -> usize {
        self.index.extra_descriptor_count()
    }

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
        let equalized = self
            .map
            .local_contrast
            .map(|local_contrast| local_contrast.apply(image));
        let image = equalized.as_ref().unwrap_or(image);
        let smoothed = self
            .map
            .feature_smoothing
            .map(|feature_smoothing| feature_smoothing.apply(image));
        let image = smoothed.as_ref().unwrap_or(image);
        let corners = detect_keypoints_mapping_with(
            image,
            self.max_points,
            self.map
                .feature_quality_level
                .unwrap_or(MAPPING_QUALITY_LEVEL),
        );
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
        // Then, near the prior, the landmarks within a radius of its camera.
        let projection = match projection {
            Some(fix) if fix.map_from_body.is_some() => Some(fix),
            projection => match prior.zip(self.prior_radius_m) {
                Some((prior, radius)) => {
                    let centre = prior.compose(&camera_from_body.inverse()).translation;
                    let landmarks = self
                        .index
                        .landmarks_within(&nalgebra::Point3::from(centre), radius);
                    let fix = self.solve(
                        Matching::Radius,
                        self.index.match_among_threaded(
                            &features.keypoints,
                            &features.descriptors,
                            self.gates,
                            &landmarks,
                            self.match_threads,
                        ),
                        features,
                        camera_from_body,
                    );
                    if fix.map_from_body.is_some() || projection.is_none() {
                        Some(fix)
                    } else {
                        projection
                    }
                }
                None => projection,
            },
        };
        let mut fix = match projection {
            Some(fix) if fix.map_from_body.is_some() => fix,
            // Near a prior, the projection (and radius) searches have had their try.
            Some(fix) if self.no_global_with_prior && prior.is_some() => fix,
            projection => {
                let mut fix = self.solve(
                    Matching::Global,
                    self.index.match_all(
                        &features.keypoints,
                        &features.descriptors,
                        self.gates,
                        self.match_threads,
                    ),
                    features,
                    camera_from_body,
                );
                // A weak localization by projection counts first.
                if fix.map_from_body.is_none() {
                    if let Some(weak) = projection.and_then(|p| p.weak_map_from_body) {
                        fix.weak_map_from_body = Some(weak);
                    }
                }
                fix
            }
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
        let mut result = self
            .pipeline
            .localize_correspondences(&self.camera, correspondences);
        if self.min_inlier_cells > 0 {
            apply_inlier_spread_gate(
                &mut result,
                &self.grid,
                &features.keypoints,
                self.min_inlier_cells,
            );
        }
        let map_from_body = (result.success)
            .then_some(result.pose.as_ref())
            .flatten()
            .map(|pose| pose.camera_to_world().compose(camera_from_body));
        // Turned down only for too few inliers: a weak localization.
        let weak_map_from_body = ((matching == Matching::Global || self.weak_projected)
            && !result.success
            && result.failure_reason == Some(LocalizationFailureReason::QualityGateFailed)
            && self
                .weak_min_inliers
                .is_some_and(|weak| result.inlier_count >= weak))
        .then_some(result.pose.as_ref())
        .flatten()
        .map(|pose| pose.camera_to_world().compose(camera_from_body));
        let inlier_matches = if map_from_body.is_some() {
            result
                .inlier_query_indices
                .iter()
                .zip(&result.inlier_landmark_ids)
                .filter_map(|(&index, &landmark)| Some((*features.keypoints.get(index)?, landmark)))
                .collect()
        } else {
            Vec::new()
        };
        Fix {
            weak_map_from_body,
            inlier_matches,
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

/// With fusion: a fix within this distance and angle of the fusion's own
/// prediction is used (the same gate as the correction's at a fresh fix).
const FUSION_GATE_M: f64 = 0.15;
const FUSION_GATE_DEG: f64 = 3.0;
/// [`FusionOptions::seed_s`]: how closely an earlier fix must agree with the
/// one the correction starts from, under one odometry's motion between them
/// (and within `FUSION_GATE_DEG`).
const SEED_AGREEMENT_M: f64 = 0.10;
const SEED_AGREEMENT_PER_M: f64 = 0.10;

/// The second odometry fused with the first and the fixes
/// ([`MapLocalizeOptions::fusion`]).
#[derive(Clone, Debug, PartialEq)]
pub struct FusionOptions {
    /// Drift models of the first odometry (the `odom` topic) and the second.
    pub noise: OdometryNoise,
    pub second_noise: OdometryNoise,
    /// The second odometry's body in the first's body frame
    /// (`T_body_body2`, e.g. a VIO's IMU <- `base_link`).
    pub body_from_second_body: SE3,
    /// `MapFusionConfig::source_check_window_s`: leave out an odometry that
    /// disagrees with the fixes while it does (0: off).
    pub source_check_window_s: f64,
    /// When the correction initializes or resets, start the fusion from the
    /// full fixes of the last this many seconds that agree with that fix
    /// under one odometry's motion, not that fix alone (0: off).
    pub seed_s: f64,
}

impl Default for FusionOptions {
    fn default() -> Self {
        Self {
            noise: OdometryNoise::VIO,
            second_noise: OdometryNoise::WHEEL,
            body_from_second_body: SE3::identity(),
            source_check_window_s: 0.0,
            seed_s: 0.0,
        }
    }
}

/// An odometry drift model by name: `vio`, `wheel` or `legged`.
pub fn odometry_noise_by_name(name: &str) -> Result<OdometryNoise, String> {
    match name {
        "vio" => Ok(OdometryNoise::VIO),
        "wheel" => Ok(OdometryNoise::WHEEL),
        "legged" => Ok(OdometryNoise::LEGGED),
        other => Err(format!(
            "unknown odometry noise {other:?} (vio, wheel, legged)"
        )),
    }
}

/// A pose from `tx,ty,tz,qx,qy,qz,qw`.
pub fn se3_from_csv(text: &str) -> Result<SE3, String> {
    let numbers = text
        .split(',')
        .map(|field| {
            field
                .trim()
                .parse::<f64>()
                .map_err(|e| format!("{field:?}: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let [tx, ty, tz, qx, qy, qz, qw] = numbers[..] else {
        return Err(format!("expected tx,ty,tz,qx,qy,qz,qw, got {text:?}"));
    };
    let q = NaQuaternion::new(qw, qx, qy, qz);
    if !(q.norm() > 1e-9 && q.norm().is_finite()) {
        return Err(format!("not a rotation: {text:?}"));
    }
    Ok(SE3::new(
        UnitQuaternion::from_quaternion(q),
        Vector3::new(tx, ty, tz),
    ))
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
    options: MapLocalizeOptions,
    correction: OdomCorrection,
    odom: OdomBuffer,
    /// Fixes waiting for their odometry: stamp, map pose, weak.
    /// Stamp, map pose, weak, and strength (inliers) of each queued fix.
    pending: VecDeque<(i64, SE3, bool, f64)>,
    max_gap_ns: i64,
    latest_odom: Option<(i64, SE3)>,
    latest_fix_ns: Option<i64>,
    fresh_ns: i64,
    fresh_m: f64,
    /// The body's last map pose before an odometry reset, and its stamp.
    carried: Option<(i64, SE3)>,
    /// `carried` expressed for the new odometry frame: map from odom, and
    /// the odometry position it was taken at.
    reset_guess: Option<(SE3, Vector3<f64>)>,
    /// Fixes the correction used, as the body pose each implies in the
    /// odometry frame (through the correction right after it): stamp and
    /// pose. Drained by [`Self::take_odom_fixes`].
    odom_fixes: Vec<(i64, SE3)>,
    /// With [`MapLocalizeOptions::fusion`]: the fusion, fed the fixes the
    /// correction uses, and the second odometry, as poses of the first
    /// odometry's body in the second odometry's frame.
    fusion: Option<MapFusion>,
    second: OdomBuffer,
    /// Consecutive full fixes the running fusion turned away, with both
    /// odometries' poses at each: when enough agree under one odometry's
    /// motion, the fusion restarts from them.
    fusion_rejected: Vec<(i64, SE3, [Option<SE3>; 2])>,
    /// With [`FusionOptions::seed_s`]: the full fixes of that span.
    recent_fixes: VecDeque<(i64, SE3, [Option<SE3>; 2])>,
    /// Full fixes rejected in a row, and recoveries asked for so far
    /// ([`MapLocalizeOptions::recover_after`]); the count survives resets.
    rejected_in_a_row: usize,
    recovery_requests: u32,
}

impl OdomCorrector {
    pub fn new(options: &MapLocalizeOptions) -> Self {
        Self {
            options: options.clone(),
            correction: OdomCorrection::new(options.correction.clone()),
            odom: OdomBuffer::new(options.odom_history_s),
            pending: VecDeque::new(),
            max_gap_ns: (options.max_odom_gap_s * 1e9) as i64,
            latest_odom: None,
            latest_fix_ns: None,
            fresh_ns: (options.projection_fresh_s * 1e9) as i64,
            fresh_m: options.projection_fresh_m,
            carried: None,
            reset_guess: None,
            odom_fixes: Vec::new(),
            fusion: options.fusion.as_ref().map(|fusion| {
                MapFusion::new(MapFusionConfig {
                    sources: vec![fusion.noise, fusion.second_noise],
                    source_check_window_s: fusion.source_check_window_s,
                    ..MapFusionConfig::default()
                })
            }),
            second: OdomBuffer::new(options.odom_history_s),
            fusion_rejected: Vec::new(),
            recent_fixes: VecDeque::new(),
            rejected_in_a_row: 0,
            recovery_requests: 0,
        }
    }

    /// How many times the odometry was found to have left the map
    /// ([`MapLocalizeOptions::recover_after`] full fixes rejected in a row).
    /// The VIO node resets its estimator when this grows.
    pub const fn recovery_requests(&self) -> u32 {
        self.recovery_requests
    }

    /// Adds a pose of the second odometry's body in its frame (with
    /// [`MapLocalizeOptions::fusion`]; ignored otherwise).
    pub fn add_second_odom(&mut self, stamp_ns: i64, second_from_second_body: SE3) {
        let Some(fusion) = &self.options.fusion else {
            return;
        };
        self.second.push(
            stamp_ns,
            second_from_second_body.compose(&fusion.body_from_second_body.inverse()),
        );
    }

    /// Per odometry (the first, the second): the stretches the fusion's
    /// source check has left out; `None` without fusion.
    pub fn fusion_excluded_edges(&self) -> Option<Vec<u64>> {
        self.fusion
            .as_ref()
            .map(|fusion| fusion.excluded_edges().to_vec())
    }

    /// Both odometries' poses of the first one's body at `stamp_ns`, as the
    /// fusion takes them; the second is `None` where it has no pose yet.
    fn fusion_sources(&self, stamp_ns: i64, first: &SE3) -> [Option<SE3>; 2] {
        let second = match self.second.lookup(stamp_ns, self.max_gap_ns) {
            OdomLookup::Pose(pose) => Some(pose),
            _ => None,
        };
        [Some(first.clone()), second]
    }

    /// The fixes the correction used (initialized from, accepted, or reset
    /// to) since the last call, each as the body pose it implies in the
    /// odometry frame, oldest first. Fed back into a VIO as pose priors
    /// (`visloc_vio_node`'s `map_prior`), they pull the VIO itself toward
    /// the map.
    pub fn take_odom_fixes(&mut self) -> Vec<(i64, SE3)> {
        std::mem::take(&mut self.odom_fixes)
    }

    /// Starts over after the odometry restarted in a new frame (a VIO
    /// reset): the correction, the odometry and the queued fixes go; the map
    /// pose is unset until fixes set it again. The stamp of the latest
    /// localization stays, so that [`ImageCadence`] tries every image again.
    /// The body's last map pose is carried over as a prior for the new frame
    /// (`MapLocalizeOptions::reset_prior_s`), through further resets too.
    pub fn reset(&mut self) {
        let latest_fix_ns = self.latest_fix_ns;
        let recovery_requests = self.recovery_requests;
        let carried = self
            .latest_map_pose()
            .map(|pose| (pose.stamp_ns, pose.map_from_body))
            .or_else(|| self.carried.clone());
        *self = Self::new(&self.options);
        self.latest_fix_ns = latest_fix_ns;
        self.recovery_requests = recovery_requests;
        if self.options.reset_prior_s > 0.0 {
            self.carried = carried;
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
        self.queue_fix(stamp_ns, map_from_body, false, 0.0)
    }

    /// [`Self::add_fix`] with the localization's inlier count, which the
    /// correction's warm-up (`init_warmup_s`) ranks the first fixes by.
    pub fn add_scored_fix(
        &mut self,
        stamp_ns: i64,
        map_from_body: SE3,
        inliers: usize,
    ) -> Vec<FixEvent> {
        self.queue_fix(stamp_ns, map_from_body, false, inliers as f64)
    }

    /// [`Self::add_weak_fix`] with the localization's inlier count.
    pub fn add_scored_weak_fix(
        &mut self,
        stamp_ns: i64,
        map_from_body: SE3,
        inliers: usize,
    ) -> Vec<FixEvent> {
        self.queue_fix(stamp_ns, map_from_body, true, inliers as f64)
    }

    /// Queues a weak localization (`OdomCorrection::add_weak_fix`) like
    /// [`Self::add_fix`].
    pub fn add_weak_fix(&mut self, stamp_ns: i64, map_from_body: SE3) -> Vec<FixEvent> {
        self.queue_fix(stamp_ns, map_from_body, true, 0.0)
    }

    fn queue_fix(
        &mut self,
        stamp_ns: i64,
        map_from_body: SE3,
        weak: bool,
        score: f64,
    ) -> Vec<FixEvent> {
        self.latest_fix_ns = Some(self.latest_fix_ns.map_or(stamp_ns, |t| t.max(stamp_ns)));
        self.pending
            .push_back((stamp_ns, map_from_body, weak, score));
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
        self.correction.observe_odometry(stamp_ns, &odom_from_body);
        if self.fusion.is_some() {
            let sources = self.fusion_sources(stamp_ns, &odom_from_body);
            if let Some(fusion) = &mut self.fusion {
                fusion.observe(stamp_ns, &sources);
            }
        }
        if self.reset_guess.is_none() {
            if let Some((_, map_from_body)) = &self.carried {
                self.reset_guess = Some((
                    map_from_body.compose(&odom_from_body.inverse()),
                    odom_from_body.translation,
                ));
            }
        }
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
            return self.reset_prior(stamp_ns);
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

    /// Before the correction is set again after a reset: the carried map
    /// pose, moved by the new frame's odometry (see [`Self::reset`]).
    fn reset_prior(&self, stamp_ns: i64) -> OdomLookup {
        let (Some((carried_ns, _)), Some((map_from_odom, start))) =
            (&self.carried, &self.reset_guess)
        else {
            return OdomLookup::Unavailable;
        };
        if (stamp_ns - carried_ns) as f64 * 1e-9 > self.options.reset_prior_s {
            return OdomLookup::Unavailable;
        }
        match self.odom.lookup(stamp_ns, self.max_gap_ns) {
            OdomLookup::Pose(odom_from_body)
                if (odom_from_body.translation - start).norm() > self.fresh_m =>
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
        let mut map_from_odom = self.correction.map_from_odom()?;
        let (fix_ns, fix_position) = self.correction.last_fix()?;
        // With fusion, its pose; map -> odom follows from it, so the
        // published transform and pose agree.
        if let Some(fused) = self.fusion.as_ref().and_then(|fusion| {
            fusion.map_from_body(&self.fusion_sources(*stamp_ns, odom_from_body))
        }) {
            map_from_odom = fused.compose(&odom_from_body.inverse());
        }
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
        while let Some((stamp_ns, _, _, _)) = self.pending.front() {
            let stamp_ns = *stamp_ns;
            match self.odom.lookup(stamp_ns, self.max_gap_ns) {
                OdomLookup::NotYet => break,
                OdomLookup::Unavailable => {
                    self.pending.pop_front();
                    events.push(FixEvent::Dropped { stamp_ns });
                }
                OdomLookup::Pose(odom_from_body) => {
                    let (_, map_from_body, weak, score) =
                        self.pending.pop_front().expect("front exists");
                    let outcome = if weak {
                        self.correction.add_scored_weak_fix(
                            stamp_ns,
                            &map_from_body,
                            &odom_from_body,
                            score,
                        )
                    } else {
                        self.correction.add_scored_fix(
                            stamp_ns,
                            &map_from_body,
                            &odom_from_body,
                            score,
                        )
                    };
                    if !weak && self.fusion.is_some() {
                        let sources = self.fusion_sources(stamp_ns, &odom_from_body);
                        let seed_ns = self
                            .options
                            .fusion
                            .as_ref()
                            .map_or(0, |fusion| (fusion.seed_s * 1e9) as i64);
                        if seed_ns > 0 {
                            self.recent_fixes
                                .retain(|(stamp, _, _)| stamp_ns - stamp <= seed_ns);
                            self.recent_fixes.push_back((
                                stamp_ns,
                                map_from_body.clone(),
                                sources.clone(),
                            ));
                        }
                        if let Some(fusion) = &mut self.fusion {
                            // The fusion starts (and restarts) with the
                            // correction; once running it gates fixes
                            // against its own prediction, not the first
                            // odometry's correction, which a runaway first
                            // odometry would turn against good fixes.
                            if matches!(outcome, FixOutcome::Initialized | FixOutcome::Reset)
                                || (!fusion.is_set() && outcome == FixOutcome::Accepted)
                            {
                                fusion.reset();
                                // With seed_s: the earlier fixes that agree
                                // with this one under some odometry, so the
                                // source check has their history at once.
                                for (stamp, fix, fix_sources) in &self.recent_fixes {
                                    if *stamp < stamp_ns
                                        && fixes_agree_under_a_source(
                                            fix,
                                            fix_sources,
                                            &map_from_body,
                                            &sources,
                                            SEED_AGREEMENT_M,
                                            SEED_AGREEMENT_PER_M,
                                            FUSION_GATE_DEG,
                                        )
                                    {
                                        fusion.add_fix(*stamp, fix, fix_sources);
                                    }
                                }
                                fusion.add_fix(stamp_ns, &map_from_body, &sources);
                                self.fusion_rejected.clear();
                            } else if fusion.is_set() {
                                let close = fusion.map_from_body(&sources).is_some_and(|p| {
                                    (p.translation - map_from_body.translation).norm()
                                        <= FUSION_GATE_M
                                        && p.rotation.angle_to(&map_from_body.rotation)
                                            <= FUSION_GATE_DEG.to_radians()
                                });
                                if close {
                                    fusion.add_fix(stamp_ns, &map_from_body, &sources);
                                    self.fusion_rejected.clear();
                                } else {
                                    // Gating only on its own prediction, a
                                    // fusion that went wrong (both
                                    // odometries off, or a bad start) would
                                    // never take a fix again: restart it from
                                    // rejected fixes that agree, as the
                                    // correction resets.
                                    let needed = self.options.correction.reset_after.max(2);
                                    self.fusion_rejected.push((
                                        stamp_ns,
                                        map_from_body.clone(),
                                        sources.clone(),
                                    ));
                                    if self.fusion_rejected.len() > needed {
                                        self.fusion_rejected
                                            .drain(..self.fusion_rejected.len() - needed);
                                    }
                                    if self.fusion_rejected.len() == needed
                                        && fixes_agree_under_an_odometry(
                                            &self.fusion_rejected,
                                            self.options.correction.agreement_m,
                                        )
                                    {
                                        fusion.reset();
                                        for (stamp, fix, sources) in self.fusion_rejected.drain(..)
                                        {
                                            fusion.add_fix(stamp, &fix, &sources);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if matches!(
                        outcome,
                        FixOutcome::Initialized | FixOutcome::Accepted | FixOutcome::Reset
                    ) {
                        if let Some(map_from_odom) = self.correction.map_from_odom() {
                            self.odom_fixes
                                .push((stamp_ns, map_from_odom.inverse().compose(&map_from_body)));
                        }
                    }
                    // A weak reset (`weak_reset_after`) puts the odometry
                    // back on the map, like a full one.
                    if weak && outcome == FixOutcome::Reset {
                        self.rejected_in_a_row = 0;
                    }
                    if !weak && self.options.recover_after > 0 {
                        match outcome {
                            FixOutcome::Rejected => self.rejected_in_a_row += 1,
                            FixOutcome::Accepted
                            | FixOutcome::Initialized
                            | FixOutcome::Reset
                            | FixOutcome::Pending => self.rejected_in_a_row = 0,
                            FixOutcome::Unused => {}
                        }
                        if self.rejected_in_a_row >= self.options.recover_after {
                            self.rejected_in_a_row = 0;
                            self.recovery_requests += 1;
                        }
                    }
                    events.push(FixEvent::Applied { stamp_ns, outcome });
                }
            }
        }
        events
    }
}

/// Whether each pair of consecutive `fixes` moved as one odometry did
/// between them (the same odometry for all pairs): within `tolerance_m`
/// and 3 deg.
fn fixes_agree_under_an_odometry(fixes: &[(i64, SE3, [Option<SE3>; 2])], tolerance_m: f64) -> bool {
    (0..2).any(|source| {
        fixes.windows(2).all(|pair| {
            let (_, fix_a, sources_a) = &pair[0];
            let (_, fix_b, sources_b) = &pair[1];
            let (Some(odom_a), Some(odom_b)) = (&sources_a[source], &sources_b[source]) else {
                return false;
            };
            let fix_motion = fix_a.inverse().compose(fix_b);
            let odom_motion = odom_a.inverse().compose(odom_b);
            (fix_motion.translation - odom_motion.translation).norm() <= tolerance_m
                && fix_motion.rotation.angle_to(&odom_motion.rotation) <= 3f64.to_radians()
        })
    })
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

/// One localization against the map (`map_fix`): the body pose of the
/// image at `stamp_ns`, with the covariance of a fresh fix.
pub fn map_fix_msg(
    stamp_ns: i64,
    map_from_body: &SE3,
    map_frame: &str,
    options: &MapLocalizeOptions,
) -> PoseWithCovarianceStamped {
    PoseWithCovarianceStamped {
        header: Header::new(Time::from_nanos(stamp_ns), map_frame),
        pose: PoseWithCovariance {
            pose: ros_pose(map_from_body),
            covariance: Covariance6::diagonal(
                options.position_stddev_m.powi(2),
                options.orientation_stddev_rad.powi(2),
            ),
        },
    }
}

/// A fix the correction used, as the body pose it implies in the odometry
/// frame (`OdomCorrector::take_odom_fixes`), for `odom_fix`.
pub fn odom_fix_msg(
    stamp_ns: i64,
    odom_from_body: &SE3,
    odom_frame: &str,
    options: &MapLocalizeOptions,
) -> PoseWithCovarianceStamped {
    let mut msg = map_fix_msg(stamp_ns, odom_from_body, odom_frame, options);
    msg.header.frame_id = odom_frame.into();
    msg
}

pub fn fix_outcome_name(outcome: FixOutcome) -> &'static str {
    match outcome {
        FixOutcome::Pending => "pending",
        FixOutcome::Initialized => "initialized",
        FixOutcome::Accepted => "accepted",
        FixOutcome::Rejected => "rejected",
        FixOutcome::Reset => "reset",
        FixOutcome::Unused => "unused",
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
            "no map pose yet: waiting for agreeing fixes".to_owned(),
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
                value("weak", fix.weak_map_from_body.is_some().to_string()),
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

    /// A body driving a circle of 2 m radius at 0.5 rad/s, in the map.
    fn circle(t: f64) -> SE3 {
        SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.5 * t + std::f64::consts::FRAC_PI_2),
            Vector3::new(2.0 * (0.5 * t).cos(), 2.0 * (0.5 * t).sin(), 0.0),
        )
    }

    /// The first odometry (a VIO) runs off at 6 s, reporting three times the
    /// true motion; the second (wheels, mounted elsewhere on the body) and
    /// the fixes (10 Hz, none from 6 s until `fixes_back_s`) are exact. The
    /// worst map pose error from one second after the fixes are back.
    fn runaway_first_odometry(fusion: Option<FusionOptions>, fixes_back_s: f64) -> f64 {
        runaway(fusion, 6.0, fixes_back_s, 0.0, fixes_back_s.max(6.0) + 1.0)
    }

    /// As [`runaway_first_odometry`], with the VIO running off from
    /// `runaway_s`, the correction's warm-up `warmup_s`, and the worst error
    /// measured from `measure_s` on (or from the first map pose).
    fn runaway(
        fusion: Option<FusionOptions>,
        runaway_s: f64,
        fixes_back_s: f64,
        warmup_s: f64,
        measure_s: f64,
    ) -> f64 {
        let body_from_wheels = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.3),
            Vector3::new(0.2, -0.1, -0.4),
        );
        let fusion = fusion.map(|mut fusion| {
            fusion.body_from_second_body = body_from_wheels.clone();
            fusion
        });
        let mut options = MapLocalizeOptions {
            fusion,
            ..MapLocalizeOptions::default()
        };
        options.correction.init_warmup_s = warmup_s;
        let mut core = OdomCorrector::new(&options);
        let odom_from_map = offset().inverse();
        let wheels_from_map = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, -1.1),
            Vector3::new(-4.0, 3.0, 0.0),
        );
        let mut vio = odom_from_map.compose(&circle(0.0));
        let mut worst: f64 = 0.0;
        for k in 0..300_i64 {
            let t = k as f64 * 0.05;
            let stamp = k * 50 * MS;
            if k > 0 {
                let mut step = circle(t - 0.05).inverse().compose(&circle(t));
                if t >= runaway_s {
                    step.translation *= 3.0;
                }
                vio = vio.compose(&step);
            }
            core.add_second_odom(
                stamp,
                wheels_from_map
                    .compose(&circle(t))
                    .compose(&body_from_wheels),
            );
            core.add_odom(stamp, vio.clone());
            if k % 2 == 0 && !(runaway_s..fixes_back_s).contains(&t) {
                core.add_fix(stamp, circle(t));
            }
            if t >= measure_s {
                let Some(map_pose) = core.latest_map_pose() else {
                    continue;
                };
                worst =
                    worst.max((map_pose.map_from_body.translation - circle(t).translation).norm());
            }
        }
        worst
    }

    #[test]
    fn fusing_a_second_odometry_with_the_source_check_outlasts_a_runaway_first() {
        let alone = runaway_first_odometry(None, 0.0);
        let fused = runaway_first_odometry(
            Some(FusionOptions {
                source_check_window_s: 1.0,
                ..FusionOptions::default()
            }),
            0.0,
        );
        assert!(alone > 0.1, "the correction of the VIO alone: {alone}");
        assert!(fused < 0.05, "fused with the wheels, source check: {fused}");
    }

    #[test]
    fn a_fusion_carried_off_while_fixes_are_missing_restarts_from_agreeing_fixes() {
        // No fixes for 2 s (people in front of the camera) while the VIO
        // runs off: the fusion follows it well past its 15 cm gate, so it
        // must restart from the fixes that come back, which agree under the
        // wheels' motion.
        let fused = runaway_first_odometry(
            Some(FusionOptions {
                source_check_window_s: 1.0,
                ..FusionOptions::default()
            }),
            8.0,
        );
        assert!(fused < 0.05, "fused after the gap: {fused}");
    }

    #[test]
    fn seeding_the_fusion_with_the_warm_up_fixes_leaves_a_vio_bad_from_the_start_out() {
        // The VIO runs off from the start; the map pose starts after a 3 s
        // warm-up. Started from one fix, the fusion follows the VIO until
        // the source check has a second of fixes; seeded with the warm-up's
        // fixes, it leaves the VIO out at once.
        let fused = |seed_s| {
            runaway(
                Some(FusionOptions {
                    source_check_window_s: 1.0,
                    seed_s,
                    ..FusionOptions::default()
                }),
                0.0,
                0.0,
                3.0,
                0.0,
            )
        };
        let (one, seeded) = (fused(0.0), fused(3.0));
        assert!(seeded < 0.03, "seeded: {seeded}");
        assert!(one > 2.0 * seeded, "from one fix: {one}, seeded: {seeded}");
    }

    #[test]
    fn fusion_parameters_parse() {
        assert_eq!(odometry_noise_by_name("wheel"), Ok(OdometryNoise::WHEEL));
        assert!(odometry_noise_by_name("legs").is_err());
        let pose = se3_from_csv("1, 2, 3, 0, 0, 0.7071068, 0.7071068").expect("pose");
        assert!((pose.translation - Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-12);
        assert!((pose.rotation.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
        assert!(se3_from_csv("1,2,3").is_err());
        assert!(se3_from_csv("0,0,0,0,0,0,0").is_err());
    }

    #[test]
    fn without_fusion_the_second_odometry_is_ignored() {
        let mut core = OdomCorrector::new(&MapLocalizeOptions::default());
        core.add_second_odom(0, pose(1.0, 0.2));
        assert!(core.fusion_excluded_edges().is_none());
        assert_eq!(core.second.lookup(0, 1), OdomLookup::NotYet);
    }

    /// Used fixes come back as odometry-frame body poses, once each; pending
    /// and rejected ones do not.
    #[test]
    fn used_fixes_come_back_in_the_odometry_frame() {
        let mut core = OdomCorrector::new(&MapLocalizeOptions::default());
        let odom = |i: usize| pose(i as f64 * 0.1, i as f64 * 0.05);
        for i in 0..4 {
            core.add_odom(i as i64 * 50 * MS, odom(i));
            core.add_fix(i as i64 * 50 * MS, offset().compose(&odom(i)));
        }
        // An outlier 1 m off.
        core.add_odom(200 * MS, odom(4));
        let mut wrong = offset().compose(&odom(4));
        wrong.translation.x += 1.0;
        core.add_fix(200 * MS, wrong);
        let fixes = core.take_odom_fixes();
        // The first fix is pending; the next three initialize and are accepted.
        assert_eq!(
            fixes.iter().map(|(stamp, _)| *stamp).collect::<Vec<_>>(),
            [50 * MS, 100 * MS, 150 * MS]
        );
        for (stamp, odom_from_body) in &fixes {
            let expected = odom((*stamp / (50 * MS)) as usize);
            assert!((odom_from_body.translation - expected.translation).norm() < 1e-9);
            assert!(odom_from_body.rotation.angle_to(&expected.rotation) < 1e-9);
        }
        assert!(core.take_odom_fixes().is_empty());
    }

    #[test]
    fn a_warm_up_starts_from_the_fixes_with_most_inliers() {
        // Every third fix exact with 50 inliers, the others 20 cm off with
        // 25: a still body, fixes every 100 ms.
        let run = |warmup_s: f64| {
            let mut options = MapLocalizeOptions::default();
            options.correction.init_warmup_s = warmup_s;
            let mut core = OdomCorrector::new(&options);
            let still = pose(0.5, 0.2);
            for i in 0..20_i64 {
                core.add_odom(i * 100 * MS, still.clone());
                let mut map_from_body = offset().compose(&still);
                let inliers = if i % 3 == 0 {
                    50
                } else {
                    map_from_body.translation.x += 0.2;
                    25
                };
                core.add_scored_fix(i * 100 * MS, map_from_body, inliers);
            }
            let map_pose = core.latest_map_pose().expect("set");
            (map_pose.map_from_body.translation - offset().compose(&still).translation).norm()
        };
        // Default: the first two agreeing (biased) fixes set it.
        assert!((run(0.0) - 0.2).abs() < 1e-6);
        // A 1 s warm-up starts from the strong, exact ones.
        assert!(run(1.0) < 1e-9);
    }

    #[test]
    fn fixes_rejected_in_a_row_ask_for_a_recovery() {
        let options = MapLocalizeOptions {
            recover_after: 3,
            ..MapLocalizeOptions::default()
        };
        let mut core = OdomCorrector::new(&options);
        // Set the correction on a still odometry.
        let still = pose(0.5, 0.2);
        for i in 0..4_i64 {
            core.add_odom(i * 100 * MS, still.clone());
            core.add_fix(i * 100 * MS, offset().compose(&still));
        }
        assert!(core.is_set());
        assert_eq!(core.recovery_requests(), 0);
        // The odometry runs away: each fix falls further behind it, so the
        // rejected fixes do not agree with each other.
        for k in 1..=3_i64 {
            let i = 3 + k;
            let runaway = pose(0.5 + 0.5 * (k * k) as f64, 0.2);
            core.add_odom(i * 100 * MS, runaway);
            core.add_fix(i * 100 * MS, offset().compose(&still));
        }
        assert_eq!(core.recovery_requests(), 1);
        // A reset (the VIO's, on odom_reset) keeps the count.
        core.reset();
        assert_eq!(core.recovery_requests(), 1);
        // Off by default.
        let mut quiet = OdomCorrector::new(&MapLocalizeOptions::default());
        for i in 0..4_i64 {
            quiet.add_odom(i * 100 * MS, still.clone());
            quiet.add_fix(i * 100 * MS, offset().compose(&still));
        }
        for k in 1..=5_i64 {
            let i = 3 + k;
            quiet.add_odom(i * 100 * MS, pose(0.5 + 0.5 * (k * k) as f64, 0.2));
            quiet.add_fix(i * 100 * MS, offset().compose(&still));
        }
        assert_eq!(quiet.recovery_requests(), 0);
    }

    #[test]
    fn agreeing_weak_fixes_replace_a_drifted_correction_with_weak_reset_after() {
        let run = |weak_reset_after: usize| {
            let mut options = MapLocalizeOptions::default();
            options.correction.weak_reset_after = weak_reset_after;
            let mut core = OdomCorrector::new(&options);
            for i in 0..3_i64 {
                let odom = pose(0.2 * i as f64, 0.0);
                core.add_odom(i * 100 * MS, odom.clone());
                core.add_fix(i * 100 * MS, offset().compose(&odom));
            }
            assert!(core.is_set());
            // The odometry drifted 2 m while no full fix came; the weak
            // fixes see the robot where it is, 20 cm apart.
            let drift = SE3::new(UnitQuaternion::identity(), Vector3::new(2.0, 0.0, 0.0));
            let mut outcomes = Vec::new();
            for i in 3..6_i64 {
                let truth = pose(0.2 * i as f64, 0.0);
                core.add_odom(i * 100 * MS, drift.compose(&truth));
                for event in core.add_weak_fix(i * 100 * MS, offset().compose(&truth)) {
                    if let FixEvent::Applied { outcome, .. } = event {
                        outcomes.push(outcome);
                    }
                }
            }
            outcomes
        };
        assert_eq!(run(0), [FixOutcome::Unused; 3]);
        assert_eq!(
            run(3),
            [FixOutcome::Unused, FixOutcome::Unused, FixOutcome::Reset]
        );
    }

    #[test]
    fn weak_fixes_need_three_agreeing_and_wait_for_their_odometry_too() {
        let mut core = OdomCorrector::new(&MapLocalizeOptions::default());
        let stamps = (0..5).map(|i| i64::from(i) * 50 * MS).collect::<Vec<_>>();
        let odom = |i: usize| pose(i as f64 * 0.1, i as f64 * 0.05);
        let outcomes = |events: Vec<FixEvent>| {
            events
                .into_iter()
                .map(|event| match event {
                    FixEvent::Applied { outcome, .. } => outcome,
                    FixEvent::Dropped { .. } => panic!("dropped"),
                })
                .collect::<Vec<_>>()
        };
        // Weak fixes queued before their odometry, applied as it comes.
        for (i, stamp) in stamps.iter().enumerate().take(3) {
            assert!(core
                .add_weak_fix(*stamp, offset().compose(&odom(i)))
                .is_empty());
        }
        assert_eq!(core.latest_fix_ns(), Some(stamps[2]));
        let mut seen = Vec::new();
        for (i, stamp) in stamps.iter().enumerate().take(3) {
            seen.extend(outcomes(core.add_odom(*stamp, odom(i)).0));
        }
        assert_eq!(
            seen,
            [
                FixOutcome::Pending,
                FixOutcome::Pending,
                FixOutcome::Initialized
            ]
        );
        assert!(core.is_set());
        // Once set, a weak fix is not used.
        core.add_odom(stamps[3], odom(3));
        assert_eq!(
            outcomes(core.add_weak_fix(stamps[3], odom(3))),
            [FixOutcome::Unused]
        );
        assert_eq!(MapLocalizeOptions::default().weak_min_inliers, 15);
    }

    #[test]
    fn reset_drops_the_correction_for_a_new_odometry_frame() {
        let mut core = OdomCorrector::new(&MapLocalizeOptions::default());
        let odom = |i: i64| pose(i as f64 * 0.1, 0.0);
        for i in 0..3 {
            core.add_odom(i * 50 * MS, odom(i));
            core.add_fix(i * 50 * MS, offset().compose(&odom(i)));
        }
        assert!(core.is_set());
        let before = core.latest_map_pose().unwrap().map_from_body;
        core.reset();
        assert!(!core.is_set());
        assert!(core.latest_map_pose().is_none());
        assert_eq!(core.pending_fixes(), 0);
        assert_eq!(core.latest_fix_ns(), Some(100 * MS));
        assert!(matches!(core.prior(150 * MS), OdomLookup::Unavailable));
        // The VIO restarted at its own origin; fixes set a new correction
        // that puts the robot where it was.
        let restarted = |i: i64| pose(i as f64 * 0.001, 0.0);
        let map_from_new_odom = before.compose(&restarted(0).inverse());
        for i in 3..6 {
            core.add_odom(i * 50 * MS, restarted(i - 3));
            core.add_fix(i * 50 * MS, map_from_new_odom.compose(&restarted(i - 3)));
        }
        assert!(core.is_set());
        let after = core.latest_map_pose().unwrap().map_from_body;
        assert!((after.translation - before.translation).norm() < 0.01);
    }

    #[test]
    fn a_reset_carries_the_last_map_pose_as_a_prior() {
        let mut core = OdomCorrector::new(&MapLocalizeOptions::default());
        let odom = |i: i64| pose(i as f64 * 0.1, 0.0);
        for i in 0..3 {
            core.add_odom(i * 50 * MS, odom(i));
            core.add_fix(i * 50 * MS, offset().compose(&odom(i)));
        }
        let before = core.latest_map_pose().unwrap().map_from_body;
        core.reset();
        // Reset again before the new frame's first pose (a blackout resets
        // the VIO every half second): the pose is still carried.
        core.add_odom(200 * MS, pose(5.0, 0.0));
        core.reset();
        let restarted = |i: i64| pose(-3.0 + i as f64 * 0.01, 0.0);
        core.add_odom(300 * MS, restarted(0));
        core.add_odom(350 * MS, restarted(1));
        let OdomLookup::Pose(prior) = core.prior(350 * MS) else {
            panic!("no prior after the reset");
        };
        // Where the robot was, plus the 1 cm it moved in the new frame.
        let expected = before
            .compose(&restarted(0).inverse())
            .compose(&restarted(1));
        assert!((prior.translation - expected.translation).norm() < 1e-9);
        assert!((prior.translation - before.translation).norm() < 0.02);
        // Not after `reset_prior_s`, nor after `projection_fresh_m` of travel.
        core.add_odom(31 * 1000 * MS, restarted(2));
        assert!(matches!(
            core.prior(31 * 1000 * MS),
            OdomLookup::Unavailable
        ));
        let mut far = OdomCorrector::new(&MapLocalizeOptions::default());
        for i in 0..3 {
            far.add_odom(i * 50 * MS, odom(i));
            far.add_fix(i * 50 * MS, offset().compose(&odom(i)));
        }
        far.reset();
        far.add_odom(300 * MS, restarted(0));
        far.add_odom(350 * MS, pose(0.0, 0.0));
        assert!(matches!(far.prior(350 * MS), OdomLookup::Unavailable));
        // Off with reset_prior_s 0.
        let mut off = OdomCorrector::new(&MapLocalizeOptions {
            reset_prior_s: 0.0,
            ..MapLocalizeOptions::default()
        });
        for i in 0..3 {
            off.add_odom(i * 50 * MS, odom(i));
            off.add_fix(i * 50 * MS, offset().compose(&odom(i)));
        }
        off.reset();
        off.add_odom(300 * MS, restarted(0));
        assert!(matches!(off.prior(300 * MS), OdomLookup::Unavailable));
    }

    #[test]
    fn the_gate_widens_with_the_odometry_path_since_the_last_fix() {
        let mut core = OdomCorrector::new(&MapLocalizeOptions::default());
        let stamp = |i: usize| i as i64 * 50 * MS;
        // 5 m out along x and back: 10.8 m of path.
        let odom = |i: usize| pose(if i <= 50 { i } else { 100 - i } as f64 * 0.1, 0.0);
        for i in 0..2 {
            core.add_odom(stamp(i), odom(i));
            core.add_fix(stamp(i), offset().compose(&odom(i)));
        }
        assert!(core.is_set());
        for i in 2..100 {
            core.add_odom(stamp(i), odom(i));
        }
        // Frame 97 is 0.2 m from the last fix but 10.6 m along the path:
        // 0.3 m off is within the gate, though its localization arrives
        // after later odometry.
        let mut map_from_body = offset().compose(&odom(97));
        map_from_body.translation.z += 0.3;
        assert_eq!(
            core.add_fix(stamp(97), map_from_body),
            [FixEvent::Applied {
                stamp_ns: stamp(97),
                outcome: FixOutcome::Accepted
            }]
        );
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
        let same_map = load_basalt_map(&dir).expect("loads");
        let radius_map = load_basalt_map(&dir).expect("loads");
        let near_map = load_basalt_map(&dir).expect("loads");
        // A map built from equalized images says so; a bad entry is refused.
        let summary_path = dir.join("map.json");
        let raw_summary = fs::read_to_string(&summary_path).expect("map.json");
        let mut summary: serde_json::Value = serde_json::from_str(&raw_summary).expect("json");
        summary["feature_local_contrast"] = serde_json::json!({"tiles": 6, "clip_limit": 3.0});
        fs::write(&summary_path, summary.to_string()).expect("map.json");
        assert_eq!(
            load_basalt_map(&dir).expect("loads").local_contrast,
            Some(LocalContrast {
                tiles: 6,
                clip_limit: 3.0
            })
        );
        summary["feature_local_contrast"] = serde_json::json!({"tiles": 0, "clip_limit": 3.0});
        fs::write(&summary_path, summary.to_string()).expect("map.json");
        assert!(load_basalt_map(&dir).is_err());
        // So does one built from blurred images.
        summary["feature_local_contrast"] = serde_json::Value::Null;
        summary["feature_smoothing_sigma"] = serde_json::json!(1.5);
        fs::write(&summary_path, summary.to_string()).expect("map.json");
        assert_eq!(
            load_basalt_map(&dir).expect("loads").feature_smoothing,
            Some(GaussianSmoothing { sigma: 1.5 })
        );
        summary["feature_smoothing_sigma"] = serde_json::json!(-1.0);
        fs::write(&summary_path, summary.to_string()).expect("map.json");
        assert!(load_basalt_map(&dir).is_err());
        // And one built with another detector quality level.
        summary["feature_smoothing_sigma"] = serde_json::Value::Null;
        summary["feature_quality_level"] = serde_json::json!(0.002);
        fs::write(&summary_path, summary.to_string()).expect("map.json");
        assert_eq!(
            load_basalt_map(&dir).expect("loads").feature_quality_level,
            Some(0.002)
        );
        summary["feature_quality_level"] = serde_json::json!(0.0);
        fs::write(&summary_path, summary.to_string()).expect("map.json");
        assert!(load_basalt_map(&dir).is_err());
        fs::remove_dir_all(&dir).expect("cleanup");
        assert_eq!(map.local_contrast, None);
        assert_eq!(map.feature_smoothing, None);
        assert_eq!(map.feature_quality_level, None);
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

        // With a prior radius, the landmarks near that prior's camera
        // recover it before any map-wide search.
        let with_radius = MapLocalizer::new(
            radius_map,
            0,
            &MapLocalizeOptions {
                prior_radius_m: Some(50.0),
                match_threads: 2,
                ..MapLocalizeOptions::default()
            },
        )
        .expect("camera 0");
        let fix = with_radius.localize_features(&features, Some(&wrong));
        assert_eq!(fix.matching, Some(Matching::Radius));
        let estimate = fix.map_from_body.expect("localized");
        assert!((estimate.translation - map_from_body.translation).norm() < 1e-3);
        // A radius holding no landmark, and no map-wide search near a prior:
        // no fix, rather than one from anywhere in the map.
        let near_only = MapLocalizer::new(
            near_map,
            0,
            &MapLocalizeOptions {
                prior_radius_m: Some(1e-6),
                no_global_with_prior: true,
                ..MapLocalizeOptions::default()
            },
        )
        .expect("camera 0");
        let fix = near_only.localize_features(&features, Some(&wrong));
        assert_ne!(fix.matching, Some(Matching::Global));
        assert!(fix.map_from_body.is_none(), "{fix:?}");
        assert!(near_only
            .localize_features(&features, None)
            .map_from_body
            .is_some());

        // Above every image's inliers, a localization is only weak, by
        // projection too when a weak gate is set.
        let mut options = MapLocalizeOptions {
            min_inliers: 1000,
            ..MapLocalizeOptions::default()
        };
        options.correction.weak_gate_m = 0.05;
        let strict = MapLocalizer::new(same_map, 0, &options).expect("camera 0");
        assert!(strict.weak_projected && !localizer.weak_projected);
        let fix = strict.localize_features(&features, Some(&prior));
        assert!(fix.map_from_body.is_none(), "{fix:?}");
        let weak = fix.weak_map_from_body.expect("weak");
        assert!((weak.translation - map_from_body.translation).norm() < 1e-3);

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
            extra_descriptors: Vec::new(),
            imu_from_camera: vec![SE3::identity()],
            max_points: 800,
            local_contrast: None,
            feature_smoothing: None,
            feature_quality_level: None,
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
