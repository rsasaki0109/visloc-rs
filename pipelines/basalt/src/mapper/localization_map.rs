//! Export of the mapper's map for later localization.
//!
//! The mapper keeps everything a map-based localizer needs: optimized
//! keyframe poses (`frame_poses`, `T_w_i`), every keyframe image's ORB
//! corners and 256-bit descriptors (`feature_corners`), and the triangulated
//! landmarks with their observations (`lmdb`). [`NfrMapper::localization_map`]
//! turns that state into the workspace's generic [`VisualMap`] plus a
//! [`LandmarkDescriptorStore`], which `visloc-io` saves as a COLMAP text model
//! and a `landmark_descriptors.txt` and `visloc-localization` loads again.
//!
//! Two conversions make the map readable by COLMAP-format tools:
//!
//! - **Camera model.** Basalt calibrates Double Sphere cameras, which COLMAP
//!   has no model for. Both Double Sphere and COLMAP's `OPENCV_FISHEYE`
//!   (Kannala-Brandt) map a ray at angle `θ` from the optical axis to a
//!   normalized radius that depends on `θ` alone, with the same
//!   `fx, fy, cx, cy`, so [`fit_opencv_fisheye`] keeps those four and fits
//!   `k1..k4` by linear least squares over the image's field of view. The
//!   pixels stored in the map are the original (distorted) detector corners.
//! - **Descriptors.** Each landmark keeps the medoid of its observations'
//!   ORB descriptors (the one with the smallest summed Hamming distance to
//!   the others), unpacked to 256 `0.0`/`1.0` values by
//!   [`orb_descriptor_to_f32`]. Squared L2 distance between two unpacked
//!   descriptors equals their Hamming distance, so float matchers work, and
//!   query images must be unpacked with the same function.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector, Point2, Point3};
use thiserror::Error;
use visloc_core::geometry::Pose;
use visloc_core::types::{
    Camera, CameraModel, Frame, Keyframe, Landmark, LandmarkDescriptorStore, Observation, VisualMap,
};

use super::{NfrMapper, TimeCamId};
use crate::DoubleSphereCamera;

/// Number of field-of-view samples used by [`fit_opencv_fisheye`].
const FIT_SAMPLES: usize = 512;

/// `OPENCV_FISHEYE` parameters fitted to one Double Sphere camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FisheyeFit {
    /// `[fx, fy, cx, cy, k1, k2, k3, k4]`, COLMAP's `OPENCV_FISHEYE` order.
    pub params: [f64; 8],
    /// Largest angle from the optical axis seen by the image, in radians.
    pub max_angle: f64,
    /// Largest pixel distance between the Double Sphere and fitted
    /// projections over `[0, max_angle]`.
    pub max_error_px: f64,
}

/// One exported keyframe image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalizationMapImage {
    /// COLMAP `IMAGE_ID` (dense, starting at 1, in `TimeCamId` order).
    pub image_id: u64,
    pub frame_id: u64,
    pub cam_id: u16,
    pub timestamp_ns: Option<i64>,
}

/// A mapper map in the workspace's localization map types.
#[derive(Debug, Clone)]
pub struct LocalizationMapExport {
    /// Cameras (`CAMERA_ID = cam_id + 1`), keyframe images with their
    /// world-to-camera poses and every detector corner as a keypoint, and
    /// landmarks in the mapper's world frame.
    pub map: VisualMap,
    /// One unpacked ORB descriptor per landmark.
    pub descriptors: LandmarkDescriptorStore,
    pub images: Vec<LocalizationMapImage>,
    /// Fitted camera per Basalt camera index.
    pub camera_fits: Vec<FisheyeFit>,
}

#[derive(Debug, Error, PartialEq)]
pub enum LocalizationMapError {
    #[error("the mapper has no calibration")]
    MissingCalibration,
    #[error(
        "camera {cam_id}: cannot fit an OPENCV_FISHEYE model to the Double Sphere calibration"
    )]
    CameraFit { cam_id: usize },
}

/// COLMAP camera id used for Basalt camera `cam_id`.
pub const fn localization_camera_id(cam_id: u16) -> u64 {
    cam_id as u64 + 1
}

/// Unpacks a 256-bit ORB descriptor to `0.0`/`1.0` values, bit `i` taken
/// from byte `i / 8`, bit `i % 8` (the packing of
/// [`compute_descriptors`](super::compute_descriptors)).
pub fn orb_descriptor_to_f32(descriptor: &[u8; 32]) -> Vec<f32> {
    (0..256)
        .map(|bit| f32::from((descriptor[bit / 8] >> (bit % 8)) & 1))
        .collect()
}

fn hamming(lhs: &[u8; 32], rhs: &[u8; 32]) -> u32 {
    lhs.iter().zip(rhs).map(|(a, b)| (a ^ b).count_ones()).sum()
}

/// Normalized radius of the Double Sphere projection of a unit ray at angle
/// `theta` from the optical axis.
fn double_sphere_radius(camera: &DoubleSphereCamera, theta: f64) -> Option<f64> {
    let (sin, cos) = theta.sin_cos();
    let zeta = camera.xi + cos;
    let d2 = (sin * sin + zeta * zeta).sqrt();
    let denominator = camera.alpha * d2 + (1.0 - camera.alpha) * zeta;
    (denominator > 1e-9).then(|| sin / denominator)
}

/// Fits COLMAP's `OPENCV_FISHEYE` model to a Double Sphere camera; see the
/// module documentation. Returns `None` when the image corners do not
/// unproject or the fit is degenerate.
pub fn fit_opencv_fisheye(camera: &DoubleSphereCamera) -> Option<FisheyeFit> {
    let (width, height) = (f64::from(camera.width), f64::from(camera.height));
    let mut max_angle = 0.0_f64;
    for (x, y) in [
        (0.0, 0.0),
        (width, 0.0),
        (0.0, height),
        (width, height),
        (width / 2.0, 0.0),
        (width / 2.0, height),
        (0.0, height / 2.0),
        (width, height / 2.0),
    ] {
        let ray = camera.unproject(&Point2::new(x, y))?;
        max_angle = max_angle.max(ray.z.clamp(-1.0, 1.0).acos());
    }
    if !max_angle.is_finite() || max_angle <= 0.0 {
        return None;
    }

    // radius_ds(θ) - θ = k1 θ³ + k2 θ⁵ + k3 θ⁷ + k4 θ⁹, sampled over (0, θmax].
    let mut design = DMatrix::zeros(FIT_SAMPLES, 4);
    let mut target = DVector::zeros(FIT_SAMPLES);
    let mut samples = Vec::with_capacity(FIT_SAMPLES);
    for row in 0..FIT_SAMPLES {
        let theta = max_angle * (row + 1) as f64 / FIT_SAMPLES as f64;
        let radius = double_sphere_radius(camera, theta)?;
        let theta2 = theta * theta;
        let mut power = theta * theta2;
        for column in 0..4 {
            design[(row, column)] = power;
            power *= theta2;
        }
        target[row] = radius - theta;
        samples.push((theta, radius));
    }
    let k = design.svd(true, true).solve(&target, 1e-12).ok()?;
    if !k.iter().all(|value| value.is_finite()) {
        return None;
    }

    let focal = camera.fx.max(camera.fy);
    let max_error_px = samples
        .iter()
        .map(|&(theta, radius)| {
            let theta2 = theta * theta;
            let fitted =
                theta * (1.0 + theta2 * (k[0] + theta2 * (k[1] + theta2 * (k[2] + theta2 * k[3]))));
            focal * (fitted - radius).abs()
        })
        .fold(0.0, f64::max);
    Some(FisheyeFit {
        params: [
            camera.fx, camera.fy, camera.cx, camera.cy, k[0], k[1], k[2], k[3],
        ],
        max_angle,
        max_error_px,
    })
}

impl NfrMapper {
    /// Exports the current map for localization; see the module
    /// documentation.
    ///
    /// Images are the keyframe images that have a pose and detector
    /// features. Landmarks are the initialized `lmdb` landmarks whose host
    /// pose is known, keeping the observations in exported images; a
    /// landmark left with fewer than two observations is dropped.
    pub fn localization_map(&self) -> Result<LocalizationMapExport, LocalizationMapError> {
        let calibration = self
            .calibration
            .as_ref()
            .ok_or(LocalizationMapError::MissingCalibration)?;

        let mut map = VisualMap::new();
        let mut camera_fits = Vec::with_capacity(calibration.cameras.len());
        for (cam_id, camera) in calibration.cameras.iter().enumerate() {
            let fit =
                fit_opencv_fisheye(camera).ok_or(LocalizationMapError::CameraFit { cam_id })?;
            let id = localization_camera_id(cam_id as u16);
            map.cameras.insert(
                id,
                Camera {
                    id,
                    model: CameraModel::OpenCvFisheye,
                    width: camera.width,
                    height: camera.height,
                    params: fit.params.to_vec(),
                },
            );
            camera_fits.push(fit);
        }

        let mut images = Vec::new();
        let mut image_ids = BTreeMap::<TimeCamId, u64>::new();
        for (&image, features) in &self.feature_corners {
            if features.is_empty() {
                continue;
            }
            let (Some(world_from_imu), Some(imu_from_camera)) = (
                self.frame_poses.get(&image.frame_id),
                calibration.camera_to_imu(image.cam_id),
            ) else {
                continue;
            };
            let image_id = images.len() as u64 + 1;
            let world_to_camera = world_from_imu.compose(imu_from_camera).inverse();
            let mut frame = Frame::new(image_id, localization_camera_id(image.cam_id));
            frame.keypoints = features.corners.clone();
            frame.pose = Some(Pose { world_to_camera });
            map.keyframes.insert(
                image_id,
                Keyframe {
                    frame,
                    observations: Vec::new(),
                },
            );
            image_ids.insert(image, image_id);
            images.push(LocalizationMapImage {
                image_id,
                frame_id: image.frame_id,
                cam_id: image.cam_id,
                timestamp_ns: self.frame_timestamps.get(&image.frame_id).copied(),
            });
        }

        let mut descriptors = LandmarkDescriptorStore::new();
        for (&track_id, landmark) in &self.lmdb.landmarks {
            let (Some(position_in_host), Some(world_from_imu), Some(imu_from_camera)) = (
                landmark.position_in_host(),
                self.frame_poses.get(&landmark.host.frame_id),
                calibration.camera_to_imu(landmark.host.cam_id),
            ) else {
                continue;
            };
            let position = world_from_imu
                .compose(imu_from_camera)
                .transform_point(&Point3::from(position_in_host));
            if !position.coords.iter().all(|value| value.is_finite()) {
                continue;
            }

            let mut observations = Vec::new();
            let mut observed_descriptors = Vec::new();
            for observation in &landmark.observations {
                let Some(&image_id) = image_ids.get(&observation.image) else {
                    continue;
                };
                let keypoint_index = observation.feature_id as usize;
                let Some(descriptor) = self
                    .feature_corners
                    .get(&observation.image)
                    .and_then(|features| features.descriptors.get(keypoint_index))
                else {
                    continue;
                };
                observations.push(Observation {
                    frame_id: image_id,
                    landmark_id: track_id,
                    keypoint_index,
                    xy: observation.pixel,
                });
                observed_descriptors.push(descriptor);
            }
            if observations.len() < 2 {
                continue;
            }

            let medoid = observed_descriptors
                .iter()
                .min_by_key(|candidate| {
                    observed_descriptors
                        .iter()
                        .map(|other| hamming(candidate, other))
                        .sum::<u32>()
                })
                .expect("at least two observations");
            descriptors.insert(track_id, orb_descriptor_to_f32(medoid));

            for observation in &observations {
                if let Some(keyframe) = map.keyframes.get_mut(&observation.frame_id) {
                    keyframe.observations.push(observation.clone());
                }
            }
            let mut exported = Landmark::new(track_id, position);
            exported.observations = observations;
            map.landmarks.insert(track_id, exported);
        }

        Ok(LocalizationMapExport {
            map,
            descriptors,
            images,
            camera_fits,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mapper::{
        MapperImageFeatures, MapperLandmark, MapperObservation, NfrMapperLandmarkDb,
    };
    use crate::vio::landmarks::StereographicDirection;
    use crate::BasaltCalibration;
    use nalgebra::{UnitQuaternion, Vector3};
    use visloc_core::geometry::SE3;

    /// EuRoC cam0 Double Sphere intrinsics from
    /// `configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json`.
    fn euroc_camera() -> DoubleSphereCamera {
        DoubleSphereCamera::new(
            461.2211371727118,
            459.87157074918844,
            367.22196700849764,
            248.4212673781908,
            5.486900730699193e-06,
            0.6571976377067552,
            752,
            480,
        )
        .expect("valid camera")
    }

    #[test]
    fn opencv_fisheye_fit_matches_double_sphere_projection() {
        let camera = euroc_camera();
        let fit = fit_opencv_fisheye(&camera).expect("fit");
        assert!(fit.max_error_px < 0.05, "fit error {}", fit.max_error_px);

        // Cross-check against the core camera's own OPENCV_FISHEYE projection.
        let fisheye = Camera {
            id: 1,
            model: CameraModel::OpenCvFisheye,
            width: camera.width,
            height: camera.height,
            params: fit.params.to_vec(),
        };
        for (x, y) in [(10.0, 10.0), (376.0, 240.0), (700.0, 60.0), (40.0, 470.0)] {
            let ray = camera.unproject(&Point2::new(x, y)).expect("ray");
            let projected = fisheye.project(&Point3::from(ray)).expect("projects");
            assert!(
                (projected - Point2::new(x, y)).norm() < 0.05,
                "({x}, {y}) -> {projected}"
            );
        }
    }

    #[test]
    fn orb_descriptor_unpacks_little_endian_bits() {
        let mut descriptor = [0_u8; 32];
        descriptor[0] = 0b0000_0101;
        descriptor[31] = 0b1000_0000;
        let unpacked = orb_descriptor_to_f32(&descriptor);
        assert_eq!(unpacked.len(), 256);
        let ones = unpacked
            .iter()
            .enumerate()
            .filter(|(_, value)| **value == 1.0)
            .map(|(bit, _)| bit)
            .collect::<Vec<_>>();
        assert_eq!(ones, vec![0, 2, 255]);
    }

    /// A two-keyframe stereo mapper whose landmarks project exactly onto
    /// their stored corners.
    fn synthetic_mapper() -> (NfrMapper, Vec<Point3<f64>>) {
        let camera = euroc_camera();
        let imu_from_cam0 = SE3::new(
            UnitQuaternion::from_euler_angles(-1.5, 0.02, 1.55),
            Vector3::new(-0.02, -0.06, 0.01),
        );
        let imu_from_cam1 = imu_from_cam0.compose(&SE3::new(
            UnitQuaternion::identity(),
            Vector3::new(0.11, 0.0, 0.0),
        ));
        let calibration = BasaltCalibration {
            t_imu_cam: vec![imu_from_cam0.clone(), imu_from_cam1.clone()],
            cameras: vec![camera, camera],
            resolutions: vec![(752, 480), (752, 480)],
            calib_accel_bias: vec![0.0; 3],
            calib_gyro_bias: vec![0.0; 3],
            imu_update_rate_hz: 200.0,
            accel_noise_std: Vector3::repeat(1.0),
            gyro_noise_std: Vector3::repeat(1.0),
            accel_bias_std: Vector3::repeat(1.0),
            gyro_bias_std: Vector3::repeat(1.0),
            t_mocap_world: SE3::identity(),
            t_imu_marker: SE3::identity(),
            mocap_time_offset_ns: 0,
            mocap_to_imu_offset_ns: 0,
            cam_time_offset_ns: 0,
        };

        let mut mapper = NfrMapper::default();
        mapper.calibration = Some(calibration);
        let poses = [
            SE3::new(UnitQuaternion::identity(), Vector3::zeros()),
            SE3::new(
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0.05),
                Vector3::new(0.3, 0.1, 0.0),
            ),
        ];
        for (frame_id, pose) in poses.iter().enumerate() {
            mapper.frame_poses.insert(frame_id as u64, pose.clone());
            mapper
                .frame_timestamps
                .insert(frame_id as u64, 1_000 + frame_id as i64);
        }

        // Points in front of cam0 at the first pose.
        let world_from_cam0 = poses[0].compose(&imu_from_cam0);
        let points = (0..30)
            .map(|index| {
                let x = -1.5 + 0.1 * index as f64;
                let y = -0.8 + 0.05 * (index % 7) as f64;
                let z = 3.0 + 0.2 * (index % 5) as f64;
                world_from_cam0.transform_point(&Point3::new(x, y, z))
            })
            .collect::<Vec<_>>();

        let mut corners = BTreeMap::<TimeCamId, MapperImageFeatures>::new();
        let mut landmarks = BTreeMap::new();
        for (track_id, point) in points.iter().enumerate() {
            let mut observations = Vec::new();
            for (frame_id, pose) in poses.iter().enumerate() {
                for (cam_id, imu_from_cam) in [&imu_from_cam0, &imu_from_cam1].iter().enumerate() {
                    let image = TimeCamId::new(frame_id as u64, cam_id as u16);
                    let point_camera = pose.compose(imu_from_cam).inverse().transform_point(point);
                    let pixel = camera.project(&point_camera).expect("visible");
                    let features = corners.entry(image).or_insert_with(|| MapperImageFeatures {
                        corners: Vec::new(),
                        corner_angles: Vec::new(),
                        descriptors: Vec::new(),
                        rays: Vec::new(),
                        hashes: Vec::new(),
                        bow_vector: Vec::new(),
                    });
                    let feature_id = features.corners.len() as u64;
                    features.corners.push(pixel);
                    let mut descriptor = [0_u8; 32];
                    descriptor[track_id % 32] = 1 << (track_id % 8);
                    // One observation per landmark carries a corrupted bit;
                    // the medoid must ignore it.
                    if frame_id == 1 && cam_id == 1 {
                        descriptor[(track_id + 7) % 32] ^= 0xff;
                    }
                    features.descriptors.push(descriptor);
                    observations.push(MapperObservation {
                        image,
                        feature_id,
                        pixel,
                    });
                }
            }
            let host = TimeCamId::new(0, 0);
            let host_point = world_from_cam0.inverse().transform_point(point);
            landmarks.insert(
                track_id as u64,
                MapperLandmark {
                    track_id: track_id as u64,
                    host,
                    second: TimeCamId::new(0, 1),
                    direction: StereographicDirection::from_bearing(host_point.coords)
                        .expect("direction"),
                    inverse_distance: 1.0 / host_point.coords.norm(),
                    observations,
                },
            );
        }
        mapper.feature_corners = corners;
        mapper.lmdb = NfrMapperLandmarkDb {
            landmarks,
            observations: BTreeMap::new(),
        };
        mapper.lmdb.rebuild_observation_index();
        (mapper, points)
    }

    #[test]
    fn export_reprojects_landmarks_onto_their_observations() {
        let (mapper, points) = synthetic_mapper();
        let export = mapper.localization_map().expect("export");

        assert_eq!(export.map.cameras.len(), 2);
        assert_eq!(export.images.len(), 4);
        assert_eq!(export.map.keyframes.len(), 4);
        assert_eq!(export.map.landmarks.len(), points.len());
        assert_eq!(export.descriptors.len(), points.len());
        assert_eq!(
            export.images[0],
            LocalizationMapImage {
                image_id: 1,
                frame_id: 0,
                cam_id: 0,
                timestamp_ns: Some(1_000),
            }
        );

        for (track_id, point) in points.iter().enumerate() {
            let landmark = &export.map.landmarks[&(track_id as u64)];
            assert!((landmark.position - point).norm() < 1e-9);
            assert_eq!(landmark.observations.len(), 4);
            for observation in &landmark.observations {
                let keyframe = &export.map.keyframes[&observation.frame_id];
                assert_eq!(
                    keyframe.frame.keypoints[observation.keypoint_index],
                    observation.xy
                );
                let camera = &export.map.cameras[&keyframe.frame.camera_id];
                let pose = keyframe.frame.pose.as_ref().expect("pose");
                let projected = camera
                    .project(&pose.transform_world_point(&landmark.position))
                    .expect("projects");
                assert!((projected - observation.xy).norm() < 0.05);
            }

            let mut expected = [0_u8; 32];
            expected[track_id % 32] = 1 << (track_id % 8);
            assert_eq!(
                export.descriptors.get(track_id as u64).expect("descriptor"),
                orb_descriptor_to_f32(&expected).as_slice()
            );
        }
        assert!(export.map.validate().is_valid());
    }

    #[test]
    fn landmarks_with_one_exported_observation_are_dropped() {
        let (mut mapper, _) = synthetic_mapper();
        // Without frame 1's pose only the two frame-0 observations remain.
        mapper.frame_poses.remove(&1);
        let export = mapper.localization_map().expect("export");
        assert_eq!(export.images.len(), 2);
        assert_eq!(export.map.landmarks.len(), 30);

        // Without camera 1 as well, every landmark has a single observation.
        mapper
            .calibration
            .as_mut()
            .expect("calibration")
            .t_imu_cam
            .truncate(1);
        let export = mapper.localization_map().expect("export");
        assert!(export.map.landmarks.is_empty());
        assert!(export.descriptors.is_empty());
    }

    #[test]
    fn export_requires_calibration() {
        let mapper = NfrMapper::default();
        assert_eq!(
            mapper.localization_map().unwrap_err(),
            LocalizationMapError::MissingCalibration
        );
    }
}
