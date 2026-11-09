//! Landmark reprojection metrics.

use super::*;

pub(super) fn evaluate_landmark_position(
    landmark: &LandmarkOutput,
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<(f64, f64), String> {
    let metrics =
        evaluate_landmark_metrics(landmark, observations, images, cameras, pose_overrides)?;
    Ok((metrics.mean_error, metrics.max_error))
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct LandmarkMetrics {
    pub(super) errors: Vec<f64>,
    pub(super) mean_error: f64,
    pub(super) rms_error: f64,
    pub(super) max_error: f64,
}

pub(super) fn evaluate_landmark_metrics(
    landmark: &LandmarkOutput,
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<LandmarkMetrics, String> {
    let mut errors = Vec::with_capacity(landmark.observations.len());
    for key in &landmark.observations {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "landmark references unknown image".to_owned())?;
        let camera = cameras
            .get(&image.atlas.camera_id)
            .ok_or_else(|| "landmark references unknown camera".to_owned())?;
        let observation = observations
            .get(key)
            .ok_or_else(|| "landmark references unknown observation".to_owned())?;
        let pose = pose_for_image(image, pose_overrides);
        let point_camera = pose.transform_world_point(&landmark.position);
        if !point_camera.coords.iter().all(|value| value.is_finite()) || point_camera.z <= 0.0 {
            return Err("existing landmark became invalid".to_owned());
        }
        let projected = camera
            .project(&point_camera)
            .ok_or_else(|| "existing landmark projection failed".to_owned())?;
        let error = (projected - observation.xy).norm();
        if !error.is_finite() {
            return Err("existing landmark reprojection became non-finite".to_owned());
        }
        errors.push(error);
    }
    if errors.is_empty() {
        return Err("landmark has no observations".to_owned());
    }
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let rms = (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    Ok(LandmarkMetrics {
        max_error: errors.iter().copied().fold(0.0, f64::max),
        errors,
        mean_error: mean,
        rms_error: rms,
    })
}

// The filter mode must inspect depth failures after the all-observation cost
// gate.  Camera::project intentionally rejects non-positive depth, so this
// raw-cost helper evaluates the same projection equations without applying
// the depth gate.  The depth gate remains mandatory during candidate
// filtering/retriangulation and in the final validator.
pub(super) fn project_allowing_nonpositive_depth(
    camera: &Camera,
    point_camera: &Point3<f64>,
) -> Option<Point2<f64>> {
    if !point_camera.coords.iter().all(|value| value.is_finite()) || point_camera.z == 0.0 {
        return None;
    }
    if point_camera.z > 0.0 {
        return camera.project(point_camera);
    }
    let (fx, fy, cx, cy) = camera.intrinsics()?;
    if ![fx, fy, cx, cy].iter().all(|value| value.is_finite()) {
        return None;
    }
    let mut x = point_camera.x / point_camera.z;
    let mut y = point_camera.y / point_camera.z;
    if let Some((k1, k2)) = camera.radial_distortion() {
        if !k1.is_finite() || !k2.is_finite() {
            return None;
        }
        let r2 = x * x + y * y;
        let d = 1.0 + k1 * r2 + k2 * r2 * r2;
        x *= d;
        y *= d;
    }
    let projected = Point2::new(fx * x + cx, fy * y + cy);
    projected
        .coords
        .iter()
        .all(|value| value.is_finite())
        .then_some(projected)
}

pub(super) fn evaluate_landmark_metrics_allowing_nonpositive_depth(
    landmark: &LandmarkOutput,
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<LandmarkMetrics, String> {
    let mut errors = Vec::with_capacity(landmark.observations.len());
    for key in &landmark.observations {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "landmark references unknown image".to_owned())?;
        let camera = cameras
            .get(&image.atlas.camera_id)
            .ok_or_else(|| "landmark references unknown camera".to_owned())?;
        let observation = observations
            .get(key)
            .ok_or_else(|| "landmark references unknown observation".to_owned())?;
        let pose = pose_for_image(image, pose_overrides);
        let point_camera = pose.transform_world_point(&landmark.position);
        let projected = project_allowing_nonpositive_depth(camera, &point_camera)
            .ok_or_else(|| "landmark projection became non-finite".to_owned())?;
        let error = (projected - observation.xy).norm();
        if !error.is_finite() {
            return Err("landmark reprojection became non-finite".to_owned());
        }
        errors.push(error);
    }
    if errors.is_empty() {
        return Err("landmark has no observations".to_owned());
    }
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let rms = (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    Ok(LandmarkMetrics {
        max_error: errors.iter().copied().fold(0.0, f64::max),
        errors,
        mean_error: mean,
        rms_error: rms,
    })
}

pub(super) fn pose_for_image<'a>(
    image: &'a GlobalImage,
    pose_overrides: &'a BTreeMap<u64, Pose>,
) -> &'a Pose {
    pose_overrides
        .get(&image.atlas.global_image_id)
        .unwrap_or(&image.atlas.pose)
}
