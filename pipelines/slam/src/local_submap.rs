//! Independently reconstructed local submaps shared by sequential SfM and SLAM.
//!
//! A local submap is rebuilt from verified image correspondences. It never
//! consumes a live tracker pose or a live tracker depth, so its monocular gauge
//! is independent of the state it may later diagnose or correct. This is the
//! required boundary for sound scale-bearing loop measurements: two accepted
//! local submaps may later be aligned with a `Sim3`, while a low-parallax window
//! is rejected here before it can pretend to observe scale.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use nalgebra::{Point2, Point3};
use visloc_core::geometry::Pose;
use visloc_core::types::Camera;
use visloc_vision::features::FeatureSet;

use crate::{
    incremental_sfm, BaResult, IncrementalSfmConfig, IncrementalSfmError, PairwiseMatches,
    TrackBuildStats,
};

/// Quality gates applied after the independent reconstruction has completed.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalSubmapQualityConfig {
    pub min_registered_images: usize,
    pub min_registration_fraction: f64,
    pub min_landmarks: usize,
    pub min_observations: usize,
    pub min_median_track_length: f64,
    pub min_median_max_parallax_deg: f64,
    pub max_mean_reprojection_px: f64,
    /// Minimum fraction of held-out observations explained after rebuilding
    /// their landmark from the other views in the same track.
    pub min_leave_one_out_support_fraction: f64,
    /// Maximum median held-out reprojection error over successful rebuilds.
    pub max_median_leave_one_out_reprojection_px: f64,
}

impl Default for LocalSubmapQualityConfig {
    fn default() -> Self {
        Self {
            min_registered_images: 3,
            min_registration_fraction: 0.75,
            min_landmarks: 30,
            min_observations: 90,
            min_median_track_length: 3.0,
            min_median_max_parallax_deg: 2.0,
            max_mean_reprojection_px: 2.0,
            min_leave_one_out_support_fraction: 0.5,
            max_median_leave_one_out_reprojection_px: 2.0,
        }
    }
}

/// Construction and acceptance policy for [`LocalSubmapBuilder`].
#[derive(Debug, Clone, PartialEq)]
pub struct LocalSubmapConfig {
    pub sfm: IncrementalSfmConfig,
    pub quality: LocalSubmapQualityConfig,
}

impl Default for LocalSubmapConfig {
    fn default() -> Self {
        Self {
            sfm: IncrementalSfmConfig::default(),
            quality: LocalSubmapQualityConfig::default(),
        }
    }
}

/// A registered frame expressed in this submap's independent local gauge.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalSubmapFrame {
    pub local_frame_index: usize,
    pub source_frame_id: u64,
    pub pose: Pose,
}

/// One landmark observation, retaining both local and source frame identity.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalSubmapObservation {
    pub local_frame_index: usize,
    pub source_frame_id: u64,
    pub keypoint_index: usize,
    pub pixel: Point2<f64>,
}

/// A multi-view point reconstructed only from this submap's image evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalSubmapLandmark {
    pub local_landmark_id: u64,
    pub position: Point3<f64>,
    pub observations: Vec<LocalSubmapObservation>,
}

/// Auditable uncertainty/conditioning proxy for an independently built submap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalSubmapQuality {
    pub requested_images: usize,
    pub registered_images: usize,
    pub registration_fraction: f64,
    pub landmarks: usize,
    pub observations: usize,
    pub median_track_length: f64,
    /// Median, over landmarks, of the widest viewing-ray angle in that track.
    pub median_max_parallax_deg: f64,
    /// Largest distance between reconstructed camera centres in local gauge.
    pub camera_center_diameter: f64,
    pub mean_reprojection_px: f64,
    /// Number of observations in tracks long enough for leave-one-view-out.
    pub leave_one_out_attempts: usize,
    /// Attempts whose remaining views triangulate a point that explains the
    /// held-out pixel within the mapper's reprojection gate.
    pub leave_one_out_supported: usize,
    pub leave_one_out_support_fraction: f64,
    pub median_leave_one_out_reprojection_px: f64,
}

/// A reconstruction with no inherited pose/depth state from the live tracker.
#[derive(Debug, Clone)]
pub struct LocalSubmap {
    pub camera: Camera,
    /// Full input mapping, including frames that failed registration.
    pub source_frame_ids: Vec<u64>,
    pub frames: Vec<LocalSubmapFrame>,
    pub landmarks: Vec<LocalSubmapLandmark>,
    pub quality: LocalSubmapQuality,
    pub track_build_stats: TrackBuildStats,
    pub ba_result: Option<BaResult>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalSubmapRejectionReason {
    TooFewRegisteredImages,
    LowRegistrationFraction,
    TooFewLandmarks,
    TooFewObservations,
    ShortTracks,
    LowParallax,
    HighReprojectionError,
    InsufficientLeaveOneOutSupport,
    HighLeaveOneOutReprojection,
    NonFiniteGeometry,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LocalSubmapBuildError {
    SourceFrameCountMismatch {
        ids: usize,
        features: usize,
    },
    DuplicateSourceFrameId(u64),
    InvalidPairImageIndex {
        pair_index: usize,
        image_index: usize,
        image_count: usize,
    },
    SelfPair {
        pair_index: usize,
        image_index: usize,
    },
    InvalidMatchKeypointIndex {
        pair_index: usize,
        image_index: usize,
        keypoint_index: usize,
        keypoint_count: usize,
    },
    Reconstruction(IncrementalSfmError),
    QualityRejected {
        reason: LocalSubmapRejectionReason,
        quality: LocalSubmapQuality,
    },
}

impl fmt::Display for LocalSubmapBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceFrameCountMismatch { ids, features } => {
                write!(f, "source frame id count {ids} != feature count {features}")
            }
            Self::DuplicateSourceFrameId(id) => write!(f, "duplicate source frame id {id}"),
            Self::InvalidPairImageIndex {
                pair_index,
                image_index,
                image_count,
            } => write!(
                f,
                "pair {pair_index} references image {image_index}; image count is {image_count}"
            ),
            Self::SelfPair {
                pair_index,
                image_index,
            } => write!(f, "pair {pair_index} references image {image_index} twice"),
            Self::InvalidMatchKeypointIndex {
                pair_index,
                image_index,
                keypoint_index,
                keypoint_count,
            } => write!(
                f,
                "pair {pair_index} references keypoint {keypoint_index} in image {image_index}; keypoint count is {keypoint_count}"
            ),
            Self::Reconstruction(error) => write!(f, "local reconstruction failed: {error}"),
            Self::QualityRejected { reason, .. } => {
                write!(f, "local reconstruction rejected by {reason:?}")
            }
        }
    }
}

impl Error for LocalSubmapBuildError {}

impl From<IncrementalSfmError> for LocalSubmapBuildError {
    fn from(value: IncrementalSfmError) -> Self {
        Self::Reconstruction(value)
    }
}

/// Stateless builder; all gauge and geometric state belongs to each output.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalSubmapBuilder {
    pub config: LocalSubmapConfig,
}

impl Default for LocalSubmapBuilder {
    fn default() -> Self {
        Self {
            config: LocalSubmapConfig::default(),
        }
    }
}

impl LocalSubmapBuilder {
    pub fn new(config: LocalSubmapConfig) -> Self {
        Self { config }
    }

    /// Reconstruct a local map exclusively from `features` and pre-verified
    /// `pairwise` correspondences. `source_frame_ids` only preserve identity;
    /// their numeric values never influence geometry or selection.
    pub fn build(
        &self,
        camera: &Camera,
        source_frame_ids: &[u64],
        features: &[FeatureSet],
        pairwise: &[PairwiseMatches],
    ) -> Result<LocalSubmap, LocalSubmapBuildError> {
        validate_inputs(source_frame_ids, features, pairwise)?;
        let result = incremental_sfm(camera, features, pairwise, &self.config.sfm)?;
        let camera = result
            .refined_camera
            .clone()
            .unwrap_or_else(|| camera.clone());
        let quality = measure_quality(
            features.len(),
            &camera,
            &result.poses,
            &result.tracks,
            result.mean_reprojection_px,
            &self.config.sfm,
        );
        if let Some(reason) = rejection_reason(&quality, &self.config.quality) {
            return Err(LocalSubmapBuildError::QualityRejected { reason, quality });
        }

        let frames = result
            .poses
            .iter()
            .enumerate()
            .filter_map(|(local_frame_index, pose)| {
                pose.clone().map(|pose| LocalSubmapFrame {
                    local_frame_index,
                    source_frame_id: source_frame_ids[local_frame_index],
                    pose,
                })
            })
            .collect();
        let landmarks = result
            .tracks
            .iter()
            .enumerate()
            .map(|(local_landmark_id, track)| LocalSubmapLandmark {
                local_landmark_id: local_landmark_id as u64,
                position: track.position,
                observations: track
                    .observations
                    .iter()
                    .map(
                        |&(local_frame_index, keypoint_index, pixel)| LocalSubmapObservation {
                            local_frame_index,
                            source_frame_id: source_frame_ids[local_frame_index],
                            keypoint_index,
                            pixel,
                        },
                    )
                    .collect(),
            })
            .collect();

        Ok(LocalSubmap {
            camera,
            source_frame_ids: source_frame_ids.to_vec(),
            frames,
            landmarks,
            quality,
            track_build_stats: result.track_build_stats,
            ba_result: result.ba_result,
        })
    }
}

fn validate_inputs(
    source_frame_ids: &[u64],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
) -> Result<(), LocalSubmapBuildError> {
    if source_frame_ids.len() != features.len() {
        return Err(LocalSubmapBuildError::SourceFrameCountMismatch {
            ids: source_frame_ids.len(),
            features: features.len(),
        });
    }
    let mut unique_ids = HashSet::with_capacity(source_frame_ids.len());
    for &id in source_frame_ids {
        if !unique_ids.insert(id) {
            return Err(LocalSubmapBuildError::DuplicateSourceFrameId(id));
        }
    }
    for (pair_index, pair) in pairwise.iter().enumerate() {
        for image_index in [pair.image_i, pair.image_j] {
            if image_index >= features.len() {
                return Err(LocalSubmapBuildError::InvalidPairImageIndex {
                    pair_index,
                    image_index,
                    image_count: features.len(),
                });
            }
        }
        if pair.image_i == pair.image_j {
            return Err(LocalSubmapBuildError::SelfPair {
                pair_index,
                image_index: pair.image_i,
            });
        }
        for &(keypoint_i, keypoint_j) in &pair.matches {
            for (image_index, keypoint_index) in
                [(pair.image_i, keypoint_i), (pair.image_j, keypoint_j)]
            {
                let keypoint_count = features[image_index].keypoints.len();
                if keypoint_index >= keypoint_count {
                    return Err(LocalSubmapBuildError::InvalidMatchKeypointIndex {
                        pair_index,
                        image_index,
                        keypoint_index,
                        keypoint_count,
                    });
                }
            }
        }
    }
    Ok(())
}

fn measure_quality(
    requested_images: usize,
    camera: &Camera,
    poses: &[Option<Pose>],
    tracks: &[crate::SfmTrack],
    mean_reprojection_px: f64,
    sfm_config: &IncrementalSfmConfig,
) -> LocalSubmapQuality {
    let registered_images = poses.iter().filter(|pose| pose.is_some()).count();
    let registration_fraction = if requested_images == 0 {
        0.0
    } else {
        registered_images as f64 / requested_images as f64
    };
    let observations = tracks.iter().map(|track| track.observations.len()).sum();
    let median_track_length = median(
        tracks
            .iter()
            .map(|track| track.observations.len() as f64)
            .collect(),
    );
    let median_max_parallax_deg = median(
        tracks
            .iter()
            .filter_map(|track| track_max_parallax_deg(track, poses))
            .collect(),
    );
    let centres = poses
        .iter()
        .filter_map(|pose| pose.as_ref().map(Pose::camera_center_world))
        .collect::<Vec<_>>();
    let mut camera_center_diameter: f64 = 0.0;
    for i in 0..centres.len() {
        for j in (i + 1)..centres.len() {
            camera_center_diameter = camera_center_diameter.max((centres[i] - centres[j]).norm());
        }
    }
    let (leave_one_out_attempts, leave_one_out_supported, leave_one_out_errors) =
        leave_one_out_support(camera, poses, tracks, sfm_config);
    let leave_one_out_support_fraction = if leave_one_out_attempts == 0 {
        0.0
    } else {
        leave_one_out_supported as f64 / leave_one_out_attempts as f64
    };
    let median_leave_one_out_reprojection_px = median(leave_one_out_errors);
    LocalSubmapQuality {
        requested_images,
        registered_images,
        registration_fraction,
        landmarks: tracks.len(),
        observations,
        median_track_length,
        median_max_parallax_deg,
        camera_center_diameter,
        mean_reprojection_px,
        leave_one_out_attempts,
        leave_one_out_supported,
        leave_one_out_support_fraction,
        median_leave_one_out_reprojection_px,
    }
}

fn leave_one_out_support(
    camera: &Camera,
    poses: &[Option<Pose>],
    tracks: &[crate::SfmTrack],
    config: &IncrementalSfmConfig,
) -> (usize, usize, Vec<f64>) {
    let mut attempts = 0;
    let mut supported = 0;
    let mut errors = Vec::new();
    for track in tracks.iter().filter(|track| track.observations.len() >= 3) {
        for held_out in 0..track.observations.len() {
            attempts += 1;
            let remaining = track
                .observations
                .iter()
                .enumerate()
                .filter_map(|(index, &(image, _, pixel))| {
                    (index != held_out).then_some((image, pixel))
                })
                .collect::<Vec<_>>();
            let Some(point) = incremental_sfm::triangulate_track(camera, poses, &remaining, config)
            else {
                continue;
            };
            let (image, _, pixel) = track.observations[held_out];
            let Some(pose) = poses.get(image).and_then(Option::as_ref) else {
                continue;
            };
            let Some(error) = incremental_sfm::reprojection_error_px(camera, pose, &point, &pixel)
            else {
                continue;
            };
            errors.push(error);
            if error <= config.max_reprojection_error_px {
                supported += 1;
            }
        }
    }
    (attempts, supported, errors)
}

fn track_max_parallax_deg(track: &crate::SfmTrack, poses: &[Option<Pose>]) -> Option<f64> {
    let rays = track
        .observations
        .iter()
        .filter_map(|(image_index, _, _)| {
            let centre = poses.get(*image_index)?.as_ref()?.camera_center_world();
            (track.position - centre).try_normalize(1.0e-12)
        })
        .collect::<Vec<_>>();
    if rays.len() < 2 {
        return None;
    }
    let mut max_angle: f64 = 0.0;
    for i in 0..rays.len() {
        for j in (i + 1)..rays.len() {
            max_angle = max_angle.max(rays[i].dot(&rays[j]).clamp(-1.0, 1.0).acos());
        }
    }
    Some(max_angle.to_degrees())
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn rejection_reason(
    quality: &LocalSubmapQuality,
    config: &LocalSubmapQualityConfig,
) -> Option<LocalSubmapRejectionReason> {
    if !quality.registration_fraction.is_finite()
        || !quality.median_track_length.is_finite()
        || !quality.median_max_parallax_deg.is_finite()
        || !quality.camera_center_diameter.is_finite()
        || !quality.mean_reprojection_px.is_finite()
        || !quality.leave_one_out_support_fraction.is_finite()
        || !quality.median_leave_one_out_reprojection_px.is_finite()
    {
        return Some(LocalSubmapRejectionReason::NonFiniteGeometry);
    }
    if quality.registered_images < config.min_registered_images {
        return Some(LocalSubmapRejectionReason::TooFewRegisteredImages);
    }
    if quality.registration_fraction < config.min_registration_fraction {
        return Some(LocalSubmapRejectionReason::LowRegistrationFraction);
    }
    if quality.landmarks < config.min_landmarks {
        return Some(LocalSubmapRejectionReason::TooFewLandmarks);
    }
    if quality.observations < config.min_observations {
        return Some(LocalSubmapRejectionReason::TooFewObservations);
    }
    if quality.median_track_length < config.min_median_track_length {
        return Some(LocalSubmapRejectionReason::ShortTracks);
    }
    if quality.median_max_parallax_deg < config.min_median_max_parallax_deg {
        return Some(LocalSubmapRejectionReason::LowParallax);
    }
    if quality.mean_reprojection_px > config.max_mean_reprojection_px {
        return Some(LocalSubmapRejectionReason::HighReprojectionError);
    }
    if quality.leave_one_out_support_fraction < config.min_leave_one_out_support_fraction {
        return Some(LocalSubmapRejectionReason::InsufficientLeaveOneOutSupport);
    }
    if quality.median_leave_one_out_reprojection_px
        > config.max_median_leave_one_out_reprojection_px
    {
        return Some(LocalSubmapRejectionReason::HighLeaveOneOutReprojection);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use nalgebra::{Point3, UnitQuaternion, Vector3};

    struct Scene {
        camera: Camera,
        points: Vec<Point3<f64>>,
        poses: Vec<Pose>,
    }

    fn scene() -> Scene {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut points = Vec::new();
        for x in -2..=2 {
            for y in -2..=2 {
                for z in 0..=2 {
                    points.push(Point3::new(x as f64 * 0.3, y as f64 * 0.3, z as f64 * 0.3));
                }
            }
        }
        let mut poses = Vec::new();
        for index in 0..6 {
            let angle = -0.5 + index as f64 * 0.2;
            let centre = Point3::new(3.0 * angle.sin(), 0.0, -3.0 * angle.cos());
            let forward = (Point3::origin() - centre).normalize();
            let right = forward.cross(&Vector3::y()).normalize();
            let up = right.cross(&forward);
            let camera_to_world = nalgebra::Matrix3::from_columns(&[right, -up, forward]);
            let rotation = nalgebra::Rotation3::from_matrix_unchecked(camera_to_world);
            let world_to_camera = UnitQuaternion::from_rotation_matrix(&rotation).inverse();
            let translation = -(world_to_camera * centre.coords);
            poses.push(Pose::from_world_to_camera(world_to_camera, translation));
        }
        Scene {
            camera,
            points,
            poses,
        }
    }

    fn render(scene: &Scene) -> (Vec<FeatureSet>, Vec<PairwiseMatches>) {
        let mut features = Vec::new();
        let mut visible = Vec::new();
        for pose in &scene.poses {
            let mut keypoints = Vec::new();
            let mut descriptors = Vec::new();
            let mut point_to_keypoint = HashMap::new();
            for (point_index, point) in scene.points.iter().enumerate() {
                let camera_point = pose.transform_world_point(point);
                if camera_point.z <= 0.05 {
                    continue;
                }
                let Some(pixel) = scene.camera.project(&camera_point) else {
                    continue;
                };
                if pixel.x < 0.0
                    || pixel.x >= scene.camera.width as f64
                    || pixel.y < 0.0
                    || pixel.y >= scene.camera.height as f64
                {
                    continue;
                }
                point_to_keypoint.insert(point_index, keypoints.len());
                keypoints.push(pixel);
                descriptors.push(vec![point_index as f32, 1.0, 0.0, 0.0]);
            }
            features.push(FeatureSet::new(keypoints, descriptors).unwrap());
            visible.push(point_to_keypoint);
        }
        let mut pairwise = Vec::new();
        for image_i in 0..features.len() {
            for image_j in (image_i + 1)..features.len() {
                let matches = visible[image_i]
                    .iter()
                    .filter_map(|(point, keypoint_i)| {
                        visible[image_j]
                            .get(point)
                            .map(|keypoint_j| (*keypoint_i, *keypoint_j))
                    })
                    .collect::<Vec<_>>();
                if matches.len() >= 8 {
                    pairwise.push(PairwiseMatches {
                        image_i,
                        image_j,
                        matches,
                    });
                }
            }
        }
        (features, pairwise)
    }

    #[test]
    fn independently_builds_a_well_conditioned_multiview_submap() {
        let scene = scene();
        let (features, pairwise) = render(&scene);
        let frame_ids = (100..106).collect::<Vec<_>>();
        let config = LocalSubmapConfig {
            sfm: IncrementalSfmConfig {
                min_seed_matches: 8,
                ba_every: 0,
                ..IncrementalSfmConfig::default()
            },
            quality: LocalSubmapQualityConfig {
                min_registered_images: 6,
                min_registration_fraction: 1.0,
                min_landmarks: 30,
                min_observations: 180,
                min_median_track_length: 3.0,
                min_median_max_parallax_deg: 2.0,
                max_mean_reprojection_px: 0.1,
                min_leave_one_out_support_fraction: 0.5,
                max_median_leave_one_out_reprojection_px: 0.1,
            },
        };
        let submap = LocalSubmapBuilder::new(config)
            .build(&scene.camera, &frame_ids, &features, &pairwise)
            .expect("synthetic multiview scene should build an independent submap");

        assert_eq!(submap.frames.len(), 6);
        assert_eq!(submap.quality.registered_images, 6);
        assert!(submap.quality.landmarks >= 30);
        assert!(submap.quality.observations >= 180);
        assert!(submap.quality.median_max_parallax_deg >= 2.0);
        assert!(submap.quality.mean_reprojection_px < 0.1);
        assert!(submap.quality.leave_one_out_attempts >= 180);
        assert!(submap.quality.leave_one_out_support_fraction >= 0.5);
        assert!(submap.quality.median_leave_one_out_reprojection_px < 0.1);
        assert_eq!(submap.frames[0].source_frame_id, 100);
        assert!(submap
            .landmarks
            .iter()
            .flat_map(|landmark| &landmark.observations)
            .all(|observation| (100..106).contains(&observation.source_frame_id)));
    }

    #[test]
    fn source_ids_are_identity_only_but_must_be_unique() {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let features = vec![
            FeatureSet::new(Vec::new(), Vec::new()).unwrap(),
            FeatureSet::new(Vec::new(), Vec::new()).unwrap(),
        ];
        let error = LocalSubmapBuilder::default()
            .build(&camera, &[7, 7], &features, &[])
            .unwrap_err();
        assert_eq!(error, LocalSubmapBuildError::DuplicateSourceFrameId(7));
    }

    #[test]
    fn invalid_pair_indices_are_rejected_before_mapper_indexing() {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let features = vec![
            FeatureSet::new(Vec::new(), Vec::new()).unwrap(),
            FeatureSet::new(Vec::new(), Vec::new()).unwrap(),
        ];
        let pairwise = vec![PairwiseMatches {
            image_i: 0,
            image_j: 2,
            matches: Vec::new(),
        }];
        let error = LocalSubmapBuilder::default()
            .build(&camera, &[7, 8], &features, &pairwise)
            .unwrap_err();
        assert_eq!(
            error,
            LocalSubmapBuildError::InvalidPairImageIndex {
                pair_index: 0,
                image_index: 2,
                image_count: 2,
            }
        );
    }

    #[test]
    fn quality_gate_reports_one_deterministic_reason() {
        let quality = LocalSubmapQuality {
            requested_images: 4,
            registered_images: 2,
            registration_fraction: 0.5,
            landmarks: 0,
            observations: 0,
            median_track_length: 0.0,
            median_max_parallax_deg: 0.0,
            camera_center_diameter: 0.0,
            mean_reprojection_px: 0.0,
            leave_one_out_attempts: 0,
            leave_one_out_supported: 0,
            leave_one_out_support_fraction: 0.0,
            median_leave_one_out_reprojection_px: 0.0,
        };
        assert_eq!(
            rejection_reason(&quality, &LocalSubmapQualityConfig::default()),
            Some(LocalSubmapRejectionReason::TooFewRegisteredImages)
        );

        let otherwise_valid = LocalSubmapQuality {
            requested_images: 4,
            registered_images: 4,
            registration_fraction: 1.0,
            landmarks: 40,
            observations: 120,
            median_track_length: 3.0,
            median_max_parallax_deg: 3.0,
            camera_center_diameter: 1.0,
            mean_reprojection_px: 0.5,
            leave_one_out_attempts: 120,
            leave_one_out_supported: 48,
            leave_one_out_support_fraction: 0.4,
            median_leave_one_out_reprojection_px: 0.5,
        };
        assert_eq!(
            rejection_reason(&otherwise_valid, &LocalSubmapQualityConfig::default()),
            Some(LocalSubmapRejectionReason::InsufficientLeaveOneOutSupport)
        );
        assert_eq!(
            rejection_reason(
                &LocalSubmapQuality {
                    leave_one_out_supported: 120,
                    leave_one_out_support_fraction: 1.0,
                    median_leave_one_out_reprojection_px: 2.5,
                    ..otherwise_valid
                },
                &LocalSubmapQualityConfig::default(),
            ),
            Some(LocalSubmapRejectionReason::HighLeaveOneOutReprojection)
        );
    }

    #[test]
    fn leave_one_out_support_detects_an_observation_not_explained_by_other_views() {
        let scene = scene();
        let point = scene.points[scene.points.len() / 2];
        let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
        let observations = scene
            .poses
            .iter()
            .enumerate()
            .map(|(image, pose)| {
                let pixel = scene
                    .camera
                    .project(&pose.transform_world_point(&point))
                    .expect("scene point projects");
                (image, 0, pixel)
            })
            .collect::<Vec<_>>();
        let clean = crate::SfmTrack {
            position: point,
            observations: observations.clone(),
        };
        let config = IncrementalSfmConfig {
            min_triangulation_angle_deg: 0.1,
            max_reprojection_error_px: 2.0,
            ..IncrementalSfmConfig::default()
        };
        let (clean_attempts, clean_supported, clean_errors) =
            leave_one_out_support(&scene.camera, &poses, &[clean], &config);
        assert_eq!(clean_attempts, observations.len());
        assert_eq!(clean_supported, clean_attempts);
        assert!(median(clean_errors) < 1.0e-6);

        let mut corrupted_observations = observations;
        corrupted_observations[2].2.x += 100.0;
        let corrupted = crate::SfmTrack {
            position: point,
            observations: corrupted_observations,
        };
        let (corrupt_attempts, corrupt_supported, _) =
            leave_one_out_support(&scene.camera, &poses, &[corrupted], &config);
        assert_eq!(corrupt_attempts, clean_attempts);
        assert!(corrupt_supported < clean_supported);
    }
}
