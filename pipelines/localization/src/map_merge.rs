//! Merging maps of one place built in separate sessions.
//!
//! A map built by VIO-based SLAM lives in its session's gravity-aligned
//! world, so two sessions' maps of the same place differ by a yaw about the
//! vertical and a translation (four degrees of freedom), plus each session's
//! own drift. [`register_map`] finds that transform from the two maps alone:
//!
//! 1. Every keyframe of the other map becomes a query: its landmark
//!    observations are the keypoints and those landmarks' descriptors the
//!    query descriptors. It is localized in the base map like a camera image
//!    (Hamming matching against every base landmark, P3P RANSAC, at least
//!    `min_inliers` inliers), which places the keyframe in the base map and
//!    so implies one base-from-other transform.
//! 2. The transform that most localized keyframes agree with wins (camera
//!    centres within `agreement_m`, yaw within `agreement_yaw_deg`). It is
//!    refitted to the agreeing keyframes (circular mean yaw, then the mean
//!    translation of their camera centres) until that set settles. Fewer
//!    than `min_keyframes` agreeing keyframes fail the registration, so one
//!    wrong localization cannot place a map.
//!
//! A keyframe has fewer keypoints than a camera image (only those that
//! became landmarks), so `min_inliers` defaults to 15 rather than the
//! localizer's 30: the agreement of several keyframes, not each one's
//! inlier count, is what rules out wrong poses. The fit uses the camera
//! centres rather than the matched landmarks' 3D positions, which in the
//! other map are only as good as its triangulation.
//!
//! [`merge_maps`] then moves the other map into the base frame and adds it:
//! its keyframes and landmarks get new ids after the base map's, and each of
//! its landmarks that lies within `fuse_radius_m` of a base landmark whose
//! descriptor is within `fuse_max_hamming` bits is fused into that base
//! landmark (one to one, closest descriptors first): its observations move
//! over, and the base position and descriptor are kept. Without fusion, a
//! place both sessions saw would hold two landmarks with nearly the same
//! descriptor for each point, and the ratio test of matching would reject
//! most matches there.

use std::collections::HashMap;

use nalgebra::{Point2, Vector3};
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{
    Camera, CameraId, Keyframe, Landmark, LandmarkDescriptorStore, LandmarkId, Observation,
    VisualMap,
};
use visloc_vision::matching::HammingMatcher;
use visloc_vision::pnp::{GaussNewtonPoseRefiner, P3PGrunert};
use visloc_vision::ransac::PnPRansac;

use crate::map_anchor::{wrap_angle, yaw_of, Offset};
use crate::{
    AllLandmarksSelector, BinaryLandmarkIndex, BinaryMatchGates, LocalizationConfig,
    LocalizationPipeline,
};

/// Settings of [`register_map`] and [`merge_maps`].
#[derive(Debug, Clone, PartialEq)]
pub struct MapMergeConfig {
    /// Hamming gates of keyframe-to-map matching.
    pub gates: BinaryMatchGates,
    /// Inliers a keyframe needs to count as localized in the base map.
    pub min_inliers: usize,
    pub ransac_iterations: usize,
    pub reprojection_px: f64,
    /// Localized keyframes that must agree on the transform.
    pub min_keyframes: usize,
    /// Position tolerance, in metres at the keyframes' camera centres, for
    /// localized keyframes to agree with a transform.
    pub agreement_m: f64,
    /// Yaw tolerance, in degrees, for localized keyframes to agree with a
    /// transform.
    pub agreement_yaw_deg: f64,
    /// A landmark of the other map fuses into a base landmark within this
    /// distance (after the transform)...
    pub fuse_radius_m: f64,
    /// ...whose descriptor differs from its own by at most this many bits.
    pub fuse_max_hamming: u32,
    /// Threads localizing the other map's keyframes.
    pub threads: usize,
}

impl Default for MapMergeConfig {
    /// The match gates and PnP settings of `visloc_map_localize_node`; see
    /// the module documentation for `min_inliers`.
    fn default() -> Self {
        Self {
            gates: BinaryMatchGates {
                max_distance: 70,
                ratio: Some(0.8),
            },
            min_inliers: 15,
            ransac_iterations: 1000,
            reprojection_px: 3.0,
            min_keyframes: 3,
            agreement_m: 0.25,
            agreement_yaw_deg: 2.0,
            fuse_radius_m: 0.10,
            fuse_max_hamming: 50,
            threads: 1,
        }
    }
}

/// The transform found by [`register_map`] and how it was found.
#[derive(Debug, Clone, PartialEq)]
pub struct MapRegistration {
    /// `p_base = base_from_other * p_other`: a yaw about z and a translation.
    pub base_from_other: SE3,
    /// Keyframes of the other map with a pose.
    pub keyframes: usize,
    /// Of those, keyframes localized in the base map.
    pub localized_keyframes: usize,
    /// Of those, keyframes agreeing with the transform.
    pub agreeing_keyframes: usize,
    /// Root mean square distance between the agreeing keyframes' camera
    /// centres in the base map and where the transform puts them.
    pub keyframe_rms_m: f64,
}

/// Result of [`merge_maps`].
#[derive(Debug, Clone)]
pub struct MergedMap {
    pub map: VisualMap,
    pub descriptors: LandmarkDescriptorStore,
    /// Merged keyframe id of each keyframe of the other map.
    pub keyframe_ids: HashMap<u64, u64>,
    /// Merged landmark id of each landmark of the other map; a base
    /// landmark's id when it was fused.
    pub landmark_ids: HashMap<LandmarkId, LandmarkId>,
    /// Landmarks of the other map fused into base landmarks.
    pub fused_landmarks: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MapMergeError {
    /// The base map has no landmark with a descriptor.
    EmptyBaseMap,
    /// Fewer than `needed` localized keyframes agree on a transform.
    TooFewKeyframes {
        localized: usize,
        agreeing: usize,
        needed: usize,
    },
    /// A camera of the other map is missing from the base map or differs
    /// from it.
    CameraMismatch { camera_id: CameraId },
}

impl std::fmt::Display for MapMergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBaseMap => write!(f, "the base map has no landmark descriptors"),
            Self::TooFewKeyframes {
                localized,
                agreeing,
                needed,
            } => write!(
                f,
                "{localized} keyframes localized in the base map and at most {agreeing} agree \
                 on a transform; {needed} needed"
            ),
            Self::CameraMismatch { camera_id } => write!(
                f,
                "camera {camera_id} differs between the maps (or is missing from the base map)"
            ),
        }
    }
}

impl std::error::Error for MapMergeError {}

/// One keyframe of the other map localized in the base map.
#[derive(Debug, Clone)]
struct KeyframeFix {
    other_centre: Vector3<f64>,
    base_centre: Vector3<f64>,
    /// Yaw of the base-from-other rotation this keyframe implies.
    yaw: f64,
}

impl KeyframeFix {
    /// Distance from the base-map camera centre to where `offset` puts the
    /// other map's.
    fn residual(&self, offset: &Offset) -> f64 {
        (offset.rotation() * self.other_centre + offset.translation - self.base_centre).norm()
    }

    fn agrees(&self, offset: &Offset, config: &MapMergeConfig) -> bool {
        self.residual(offset) <= config.agreement_m
            && wrap_angle(self.yaw - offset.yaw).abs() <= config.agreement_yaw_deg.to_radians()
    }

    /// The transform this keyframe alone implies.
    fn offset(&self) -> Offset {
        fit_keyframes(&[self])
    }
}

/// Circular mean yaw, then the mean translation under that yaw.
fn fit_keyframes(fixes: &[&KeyframeFix]) -> Offset {
    let (sin, cos) = fixes.iter().fold((0.0, 0.0), |(sin, cos), fix| {
        (sin + fix.yaw.sin(), cos + fix.yaw.cos())
    });
    let mut offset = Offset {
        yaw: sin.atan2(cos),
        translation: Vector3::zeros(),
    };
    let rotation = offset.rotation();
    offset.translation = fixes
        .iter()
        .map(|fix| fix.base_centre - rotation * fix.other_centre)
        .sum::<Vector3<f64>>()
        / fixes.len() as f64;
    offset
}

type Pipeline = LocalizationPipeline<HammingMatcher, AllLandmarksSelector, PnPRansac<P3PGrunert>>;

fn pnp_pipeline(config: &MapMergeConfig) -> Pipeline {
    let ransac = PnPRansac {
        pose_estimator: P3PGrunert,
        pose_refiner: Some(GaussNewtonPoseRefiner::default()),
        iterations: config.ransac_iterations,
        reprojection_threshold: config.reprojection_px,
        seed: 7,
        early_stop_min_iterations: 0,
        early_stop_inlier_ratio: None,
        confidence: Some(0.999),
    };
    LocalizationPipeline::with_pose_estimator(
        HammingMatcher {
            max_distance: config.gates.max_distance,
            ratio: config.gates.ratio,
        },
        AllLandmarksSelector,
        ransac,
        LocalizationConfig {
            ratio: config.gates.ratio,
            ransac_iterations: config.ransac_iterations,
            reprojection_threshold: config.reprojection_px,
            min_inliers: config.min_inliers,
            ..LocalizationConfig::default()
        },
    )
}

/// Localizes one keyframe of `other` in the base map behind `index`.
fn localize_keyframe(
    keyframe: &Keyframe,
    other: &VisualMap,
    other_descriptors: &LandmarkDescriptorStore,
    index: &BinaryLandmarkIndex,
    pipeline: &Pipeline,
    config: &MapMergeConfig,
) -> Option<KeyframeFix> {
    let pose = keyframe.frame.pose.as_ref()?;
    let camera = other.cameras.get(&keyframe.frame.camera_id)?;
    let mut keypoints = Vec::<Point2<f64>>::new();
    let mut descriptors = Vec::<&[f32]>::new();
    for observation in &keyframe.observations {
        if let (true, Some(descriptor)) = (
            other.landmarks.contains_key(&observation.landmark_id),
            other_descriptors.get(observation.landmark_id),
        ) {
            keypoints.push(observation.xy);
            descriptors.push(descriptor);
        }
    }
    if keypoints.len() < config.min_inliers {
        return None;
    }
    let correspondences =
        index.match_all(&keypoints, &index.pack_query(&descriptors), config.gates, 1);
    let result = pipeline.localize_correspondences(camera, correspondences);
    let base_pose = result.pose.as_ref().filter(|_| result.success)?;
    let base_from_other = base_pose
        .camera_to_world()
        .compose(&pose.world_to_camera)
        .rotation;
    Some(KeyframeFix {
        other_centre: pose.camera_center_world().coords,
        base_centre: base_pose.camera_center_world().coords,
        yaw: yaw_of(&base_from_other),
    })
}

/// Finds the transform taking `other` into `base`; see the module
/// documentation.
pub fn register_map(
    base: &VisualMap,
    base_descriptors: &LandmarkDescriptorStore,
    other: &VisualMap,
    other_descriptors: &LandmarkDescriptorStore,
    config: &MapMergeConfig,
) -> Result<MapRegistration, MapMergeError> {
    let index = BinaryLandmarkIndex::new(base, base_descriptors);
    if index.is_empty() {
        return Err(MapMergeError::EmptyBaseMap);
    }
    let pipeline = pnp_pipeline(config);
    let mut keyframes = other
        .keyframes
        .values()
        .filter(|keyframe| keyframe.frame.pose.is_some())
        .collect::<Vec<_>>();
    keyframes.sort_by_key(|keyframe| keyframe.frame.id);

    let localize = |keyframes: &[&Keyframe]| {
        keyframes
            .iter()
            .map(|keyframe| {
                localize_keyframe(
                    keyframe,
                    other,
                    other_descriptors,
                    &index,
                    &pipeline,
                    config,
                )
            })
            .collect::<Vec<_>>()
    };
    let threads = config.threads.clamp(1, keyframes.len().max(1));
    let fixes = if threads == 1 {
        localize(&keyframes)
    } else {
        let chunk = keyframes.len().div_ceil(threads);
        std::thread::scope(|scope| {
            let handles = keyframes
                .chunks(chunk)
                .map(|chunk| {
                    let localize = &localize;
                    scope.spawn(move || localize(chunk))
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("registration thread panicked"))
                .collect()
        })
    };
    let fixes = fixes.into_iter().flatten().collect::<Vec<_>>();

    // The transform most keyframes agree with, refitted to them until the
    // agreeing set settles.
    let agreeing = |offset: &Offset| {
        fixes
            .iter()
            .filter(|fix| fix.agrees(offset, config))
            .collect::<Vec<_>>()
    };
    let mut best = Vec::new();
    for fix in &fixes {
        let candidate = agreeing(&fix.offset());
        if candidate.len() > best.len() {
            best = candidate;
        }
    }
    for _ in 0..5 {
        if best.is_empty() {
            break;
        }
        let refitted = agreeing(&fit_keyframes(&best));
        let settled = refitted.len() == best.len();
        best = refitted;
        if settled {
            break;
        }
    }
    if best.len() < config.min_keyframes.max(1) {
        return Err(MapMergeError::TooFewKeyframes {
            localized: fixes.len(),
            agreeing: best.len(),
            needed: config.min_keyframes.max(1),
        });
    }
    let offset = fit_keyframes(&best);
    let keyframe_rms_m = (best
        .iter()
        .map(|fix| fix.residual(&offset).powi(2))
        .sum::<f64>()
        / best.len() as f64)
        .sqrt();

    Ok(MapRegistration {
        base_from_other: offset.se3(),
        keyframes: keyframes.len(),
        localized_keyframes: fixes.len(),
        agreeing_keyframes: best.len(),
        keyframe_rms_m,
    })
}

fn same_camera(lhs: &Camera, rhs: &Camera) -> bool {
    lhs.model == rhs.model
        && (lhs.width, lhs.height) == (rhs.width, rhs.height)
        && lhs.params.len() == rhs.params.len()
        && lhs
            .params
            .iter()
            .zip(&rhs.params)
            .all(|(a, b)| (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0))
}

/// Number of differing bits of two unpacked binary descriptors (`>= 0.5` is
/// a set bit); `u32::MAX` when their lengths differ.
fn bit_distance(lhs: &[f32], rhs: &[f32]) -> u32 {
    if lhs.len() != rhs.len() {
        return u32::MAX;
    }
    lhs.iter()
        .zip(rhs)
        .filter(|(a, b)| (**a >= 0.5) != (**b >= 0.5))
        .count() as u32
}

/// For each landmark of `other` (positions already in the base frame), the
/// base landmark it fuses into: one to one, smallest descriptor distance
/// first, then smallest position distance.
fn fusion_targets(
    base: &VisualMap,
    base_descriptors: &LandmarkDescriptorStore,
    other_positions: &[(LandmarkId, Vector3<f64>, &[f32])],
    config: &MapMergeConfig,
) -> HashMap<LandmarkId, LandmarkId> {
    let radius = config.fuse_radius_m;
    if radius <= 0.0 {
        return HashMap::new();
    }
    let cell = |position: &Vector3<f64>| {
        (
            (position.x / radius).floor() as i64,
            (position.y / radius).floor() as i64,
            (position.z / radius).floor() as i64,
        )
    };
    let mut grid = HashMap::<(i64, i64, i64), Vec<(LandmarkId, Vector3<f64>, &[f32])>>::new();
    for landmark in base.landmarks.values() {
        if let Some(descriptor) = base_descriptors.get(landmark.id) {
            grid.entry(cell(&landmark.position.coords))
                .or_default()
                .push((landmark.id, landmark.position.coords, descriptor));
        }
    }
    let mut candidates = Vec::new();
    for &(other_id, position, descriptor) in other_positions {
        let (x, y, z) = cell(&position);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for (base_id, base_position, base_descriptor) in
                        grid.get(&(x + dx, y + dy, z + dz)).into_iter().flatten()
                    {
                        let distance = (base_position - position).norm();
                        let bits = bit_distance(descriptor, base_descriptor);
                        if distance <= radius && bits <= config.fuse_max_hamming {
                            candidates.push((bits, distance, other_id, *base_id));
                        }
                    }
                }
            }
        }
    }
    candidates.sort_by(|a, b| {
        (a.0, a.1, a.2, a.3)
            .partial_cmp(&(b.0, b.1, b.2, b.3))
            .expect("finite distances")
    });
    let mut targets = HashMap::new();
    let mut taken = std::collections::HashSet::new();
    for (_, _, other_id, base_id) in candidates {
        if !targets.contains_key(&other_id) && taken.insert(base_id) {
            targets.insert(other_id, base_id);
        }
    }
    targets
}

/// Adds `other`, moved by `base_from_other`, to `base`; see the module
/// documentation. Observation confidences, landmark covariances and stereo
/// observations of `other` are carried over.
pub fn merge_maps(
    base: &VisualMap,
    base_descriptors: &LandmarkDescriptorStore,
    other: &VisualMap,
    other_descriptors: &LandmarkDescriptorStore,
    base_from_other: &SE3,
    config: &MapMergeConfig,
) -> Result<MergedMap, MapMergeError> {
    let mut camera_ids = other.cameras.keys().copied().collect::<Vec<_>>();
    camera_ids.sort_unstable();
    for camera_id in camera_ids {
        if !base
            .cameras
            .get(&camera_id)
            .is_some_and(|camera| same_camera(camera, &other.cameras[&camera_id]))
        {
            return Err(MapMergeError::CameraMismatch { camera_id });
        }
    }

    let mut other_keyframe_ids = other.keyframes.keys().copied().collect::<Vec<_>>();
    other_keyframe_ids.sort_unstable();
    let first_keyframe = base.keyframes.keys().max().map_or(1, |id| id + 1);
    let keyframe_ids = other_keyframe_ids
        .iter()
        .enumerate()
        .map(|(index, &id)| (id, first_keyframe + index as u64))
        .collect::<HashMap<_, _>>();

    let mut other_landmarks = other.landmarks.values().collect::<Vec<_>>();
    other_landmarks.sort_by_key(|landmark| landmark.id);
    let moved = other_landmarks
        .iter()
        .filter_map(|landmark| {
            Some((
                landmark.id,
                base_from_other.transform_point(&landmark.position).coords,
                other_descriptors.get(landmark.id)?,
            ))
        })
        .collect::<Vec<_>>();
    let fused = fusion_targets(base, base_descriptors, &moved, config);
    let mut next_landmark = base.landmarks.keys().max().map_or(1, |id| id + 1);
    let mut landmark_ids = HashMap::with_capacity(other_landmarks.len());
    for landmark in &other_landmarks {
        let id = fused.get(&landmark.id).copied().unwrap_or_else(|| {
            next_landmark += 1;
            next_landmark - 1
        });
        landmark_ids.insert(landmark.id, id);
    }

    let remap = |observation: &Observation| {
        Some(Observation {
            frame_id: *keyframe_ids.get(&observation.frame_id)?,
            landmark_id: *landmark_ids.get(&observation.landmark_id)?,
            keypoint_index: observation.keypoint_index,
            xy: observation.xy,
        })
    };
    let rotation = base_from_other.rotation.to_rotation_matrix();
    let mut map = base.clone();
    let mut descriptors = base_descriptors.clone();
    for landmark in &other_landmarks {
        let id = landmark_ids[&landmark.id];
        let observations = landmark
            .observations
            .iter()
            .filter_map(remap)
            .collect::<Vec<_>>();
        if fused.contains_key(&landmark.id) {
            if let Some(target) = map.landmarks.get_mut(&id) {
                target.observations.extend(observations);
            }
            continue;
        }
        let mut moved = Landmark::new(id, base_from_other.transform_point(&landmark.position));
        moved.descriptor.clone_from(&landmark.descriptor);
        moved.observations = observations;
        map.landmarks.insert(id, moved);
        if let Some(descriptor) = other_descriptors.get(landmark.id) {
            descriptors.insert(id, descriptor.to_vec());
        }
        if let Some(covariance) = other.landmark_position_covariances.get(&landmark.id) {
            map.landmark_position_covariances.insert(
                id,
                rotation.matrix() * covariance * rotation.matrix().transpose(),
            );
        }
    }

    let other_from_base = base_from_other.inverse();
    for id in &other_keyframe_ids {
        let keyframe = &other.keyframes[id];
        let mut frame = keyframe.frame.clone();
        frame.id = keyframe_ids[id];
        frame.pose = frame.pose.map(|pose| Pose {
            world_to_camera: pose.world_to_camera.compose(&other_from_base),
        });
        let mut observations = Vec::with_capacity(keyframe.observations.len());
        for observation in &keyframe.observations {
            let Some(merged) = remap(observation) else {
                continue;
            };
            if let Some(confidence) = other.observation_confidence(observation) {
                map.set_observation_confidence(&merged, confidence);
            }
            observations.push(merged);
        }
        map.keyframes.insert(
            frame.id,
            Keyframe {
                frame,
                observations,
            },
        );
    }
    for stereo in &other.stereo_observations {
        if let (Some(&frame_id), Some(&landmark_id)) = (
            keyframe_ids.get(&stereo.frame_id),
            landmark_ids.get(&stereo.landmark_id),
        ) {
            let mut stereo = stereo.clone();
            stereo.frame_id = frame_id;
            stereo.landmark_id = landmark_id;
            map.stereo_observations.push(stereo);
        }
    }

    Ok(MergedMap {
        map,
        descriptors,
        keyframe_ids,
        landmark_ids,
        fused_landmarks: fused.len(),
    })
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use nalgebra::{Matrix3, Point3, Rotation3, UnitQuaternion};
    use visloc_core::types::Frame;

    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn unit(&mut self) -> f64 {
            (self.next() % 1_000_000) as f64 / 1_000_000.0
        }

        fn bits(&mut self) -> Vec<f32> {
            (0..256).map(|_| (self.next() & 1) as f32).collect()
        }
    }

    fn flip(descriptor: &[f32], count: usize, rng: &mut Rng) -> Vec<f32> {
        let mut flipped = descriptor.to_vec();
        for _ in 0..count {
            let bit = (rng.next() % 256) as usize;
            flipped[bit] = 1.0 - flipped[bit];
        }
        flipped
    }

    const CAMERA_ID: CameraId = 1;

    fn camera() -> Camera {
        Camera::pinhole(CAMERA_ID, 640, 480, 400.0, 400.0, 320.0, 240.0)
    }

    /// World from camera of a level camera at `position` looking along
    /// `heading` (yaw from +x).
    fn level_camera(position: Vector3<f64>, heading: f64) -> SE3 {
        let forward = Vector3::new(heading.cos(), heading.sin(), 0.0);
        let right = Vector3::new(heading.sin(), -heading.cos(), 0.0);
        let down = Vector3::new(0.0, 0.0, -1.0);
        let rotation =
            Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[right, down, forward]));
        SE3::new(UnitQuaternion::from_rotation_matrix(&rotation), position)
    }

    /// Points on the four walls of a 10 x 8 x 3 m room.
    fn room(rng: &mut Rng, count: usize) -> Vec<Point3<f64>> {
        (0..count)
            .map(|index| {
                let (u, v) = (rng.unit(), rng.unit());
                match index % 4 {
                    0 => Point3::new(u * 10.0, 0.0, v * 3.0),
                    1 => Point3::new(u * 10.0, 8.0, v * 3.0),
                    2 => Point3::new(0.0, u * 8.0, v * 3.0),
                    _ => Point3::new(10.0, u * 8.0, v * 3.0),
                }
            })
            .collect()
    }

    /// The map one session would build of `points` (world frame) from
    /// `cameras` (world from camera), in its own frame `session_from_world`.
    /// Point `i` is landmark `first_id + i`, jittered by up to `noise`
    /// metres per axis, with descriptor `descriptors[i]`.
    fn session_map(
        points: &[Point3<f64>],
        descriptors: &[Vec<f32>],
        cameras: &[SE3],
        session_from_world: &SE3,
        first_id: LandmarkId,
        noise: f64,
        rng: &mut Rng,
    ) -> (VisualMap, LandmarkDescriptorStore) {
        let camera = camera();
        let mut map = VisualMap::new();
        map.cameras.insert(CAMERA_ID, camera.clone());
        let mut store = LandmarkDescriptorStore::new();
        for (index, world_from_camera) in cameras.iter().enumerate() {
            let id = index as u64 + 1;
            let mut frame = Frame::new(id, CAMERA_ID);
            frame.pose = Some(Pose {
                world_to_camera: session_from_world.compose(world_from_camera).inverse(),
            });
            map.keyframes.insert(
                id,
                Keyframe {
                    frame,
                    observations: Vec::new(),
                },
            );
        }
        for (index, point) in points.iter().enumerate() {
            let landmark_id = first_id + index as u64;
            let mut observations = Vec::new();
            for (camera_index, world_from_camera) in cameras.iter().enumerate() {
                let in_camera = world_from_camera.inverse().transform_point(point);
                let Some(pixel) = camera.project(&in_camera).filter(|pixel| {
                    in_camera.z > 0.5
                        && pixel.x >= 0.0
                        && pixel.x < 640.0
                        && pixel.y >= 0.0
                        && pixel.y < 480.0
                }) else {
                    continue;
                };
                let frame_id = camera_index as u64 + 1;
                let keyframe = map.keyframes.get_mut(&frame_id).expect("keyframe");
                let observation = Observation {
                    frame_id,
                    landmark_id,
                    keypoint_index: keyframe.observations.len(),
                    xy: pixel,
                };
                keyframe.observations.push(observation.clone());
                observations.push(observation);
            }
            if observations.is_empty() {
                continue;
            }
            let jitter =
                Vector3::new(rng.unit() - 0.5, rng.unit() - 0.5, rng.unit() - 0.5) * (2.0 * noise);
            let mut landmark = Landmark::new(
                landmark_id,
                session_from_world.transform_point(&(point + jitter)),
            );
            landmark.observations = observations;
            map.landmarks.insert(landmark_id, landmark);
            store.insert(landmark_id, descriptors[index].clone());
        }
        (map, store)
    }

    struct Sessions {
        base: (VisualMap, LandmarkDescriptorStore),
        other: (VisualMap, LandmarkDescriptorStore),
        base_from_other: SE3,
    }

    /// Two sessions in one room, each in its own gravity-aligned frame. The
    /// base session maps points 0..600 from x = 3.5 m, the other all 800
    /// from x = 6.5 m with slightly different descriptors; points 600..800
    /// are new to the other session.
    fn sessions() -> Sessions {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let points = room(&mut rng, 800);
        let descriptors = (0..points.len()).map(|_| rng.bits()).collect::<Vec<_>>();
        let other_descriptors = descriptors
            .iter()
            .map(|descriptor| flip(descriptor, (rng.next() % 12) as usize, &mut rng))
            .collect::<Vec<_>>();
        let around = |x: f64, phase: f64| {
            (0..12)
                .map(|k| level_camera(Vector3::new(x, 4.0, 1.5), phase + k as f64 * TAU / 12.0))
                .collect::<Vec<_>>()
        };
        let base_from_other = SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.6),
            Vector3::new(1.0, -2.0, 0.3),
        );
        Sessions {
            base: session_map(
                &points[..600],
                &descriptors,
                &around(3.5, 0.0),
                &SE3::identity(),
                0,
                0.01,
                &mut rng,
            ),
            other: session_map(
                &points,
                &other_descriptors,
                &around(6.5, 0.1),
                &base_from_other.inverse(),
                100_000,
                0.01,
                &mut rng,
            ),
            base_from_other,
        }
    }

    fn assert_se3_close(actual: &SE3, expected: &SE3, metres: f64, degrees: f64) {
        let error = expected.inverse().compose(actual);
        assert!(
            error.translation.norm() < metres,
            "translation off by {} m",
            error.translation.norm()
        );
        assert!(
            error.rotation.angle().to_degrees() < degrees,
            "rotation off by {} deg",
            error.rotation.angle().to_degrees()
        );
    }

    #[test]
    fn registration_recovers_the_transform_between_sessions() {
        let s = sessions();
        let registration = register_map(
            &s.base.0,
            &s.base.1,
            &s.other.0,
            &s.other.1,
            &MapMergeConfig {
                threads: 3,
                ..MapMergeConfig::default()
            },
        )
        .expect("registered");
        assert_se3_close(
            &registration.base_from_other,
            &s.base_from_other,
            0.005,
            0.05,
        );
        assert_eq!(registration.keyframes, 12);
        assert_eq!(
            registration.agreeing_keyframes,
            registration.localized_keyframes
        );
        assert!(registration.agreeing_keyframes >= 10, "{registration:?}");
        assert!(registration.keyframe_rms_m < 0.02, "{registration:?}");

        // The thread count does not change the result.
        let serial = register_map(
            &s.base.0,
            &s.base.1,
            &s.other.0,
            &s.other.1,
            &MapMergeConfig::default(),
        )
        .expect("registered");
        assert_eq!(serial, registration);
    }

    #[test]
    fn merging_fuses_shared_landmarks_and_adds_new_ones() {
        let s = sessions();
        let config = MapMergeConfig::default();
        let merged = merge_maps(
            &s.base.0,
            &s.base.1,
            &s.other.0,
            &s.other.1,
            &s.base_from_other,
            &config,
        )
        .expect("merged");
        let shared = s
            .other
            .0
            .landmarks
            .keys()
            .filter(|id| s.base.0.landmarks.contains_key(&(*id - 100_000)))
            .count();
        assert!(shared > 500, "{shared}");
        assert_eq!(merged.fused_landmarks, shared);
        for (other_id, merged_id) in &merged.landmark_ids {
            if s.base.0.landmarks.contains_key(&(other_id - 100_000)) {
                assert_eq!(*merged_id, other_id - 100_000, "fused into its own point");
            } else {
                assert!(!s.base.0.landmarks.contains_key(merged_id));
            }
        }
        assert_eq!(
            merged.map.landmarks.len(),
            s.base.0.landmarks.len() + s.other.0.landmarks.len() - shared
        );
        assert_eq!(merged.descriptors.len(), merged.map.landmarks.len());
        assert_eq!(merged.map.keyframes.len(), 24);

        // A fused landmark keeps the base position and gains the other
        // session's observations, which point at moved keyframes.
        let fused = &merged.map.landmarks[&5];
        assert_eq!(fused.position, s.base.0.landmarks[&5].position);
        let other_observations = fused
            .observations
            .iter()
            .filter(|observation| observation.frame_id > 12)
            .collect::<Vec<_>>();
        assert_eq!(
            other_observations.len(),
            s.other.0.landmarks[&100_005].observations.len()
        );
        for observation in other_observations {
            let keyframe = &merged.map.keyframes[&observation.frame_id];
            assert!(keyframe.observations.contains(observation));
            let pose = keyframe.frame.pose.as_ref().expect("pose");
            let pixel = camera()
                .project(&pose.transform_world_point(&fused.position))
                .expect("in front");
            assert!((pixel - observation.xy).norm() < 5.0);
        }

        // Keyframes of the other session land on their true poses.
        let other_camera_one = level_camera(Vector3::new(6.5, 4.0, 1.5), 0.1);
        let moved = merged.map.keyframes[&merged.keyframe_ids[&1]]
            .frame
            .pose
            .as_ref()
            .expect("pose")
            .camera_to_world();
        assert_se3_close(&moved, &other_camera_one, 1e-9, 1e-9);
    }

    #[test]
    fn without_fusion_every_landmark_is_added() {
        let s = sessions();
        let merged = merge_maps(
            &s.base.0,
            &s.base.1,
            &s.other.0,
            &s.other.1,
            &s.base_from_other,
            &MapMergeConfig {
                fuse_radius_m: 0.0,
                ..MapMergeConfig::default()
            },
        )
        .expect("merged");
        assert_eq!(merged.fused_landmarks, 0);
        assert_eq!(
            merged.map.landmarks.len(),
            s.base.0.landmarks.len() + s.other.0.landmarks.len()
        );
    }

    #[test]
    fn a_map_of_another_place_does_not_register() {
        let s = sessions();
        let mut rng = Rng(42);
        let unrelated = s
            .other
            .0
            .landmarks
            .keys()
            .map(|&id| (id, rng.bits()))
            .collect::<Vec<_>>();
        let mut store = LandmarkDescriptorStore::new();
        for (id, descriptor) in unrelated {
            store.insert(id, descriptor);
        }
        let error = register_map(
            &s.base.0,
            &s.base.1,
            &s.other.0,
            &store,
            &MapMergeConfig::default(),
        )
        .expect_err("nothing to register");
        assert!(matches!(
            error,
            MapMergeError::TooFewKeyframes { agreeing: 0, .. }
        ));
    }

    #[test]
    fn maps_of_different_cameras_do_not_merge() {
        let s = sessions();
        let mut other = s.other.0.clone();
        other.cameras.get_mut(&CAMERA_ID).expect("camera").params[0] += 1.0;
        assert_eq!(
            merge_maps(
                &s.base.0,
                &s.base.1,
                &other,
                &s.other.1,
                &s.base_from_other,
                &MapMergeConfig::default(),
            )
            .err(),
            Some(MapMergeError::CameraMismatch {
                camera_id: CAMERA_ID
            })
        );
    }
}
