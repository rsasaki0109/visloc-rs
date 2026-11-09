//! Per-frame front-end helpers: the adaptive keypoint cap, keypoint
//! undistortion, stereo optical flow and frame descriptors.

use super::*;

/// Return a new [`FeatureSet`] whose keypoints have been mapped from
/// their raw distorted pixel positions to the "ideal pinhole" pixel
/// positions implied by cam0's calibration. Descriptors are copied
/// verbatim — they are extracted from raw image patches and the
/// undistortion only corrects the *position* at which each descriptor
/// was sampled. Returns the input unchanged when `distortion` is the
/// identity (zero-coefficient case) so the no-op path is allocation-free
/// Pace the SuperPoint ONNX extractor's per-frame keypoint cap to a rolling
/// frame-time budget.
///
/// The descriptor matcher dominates the frame and is `O(N^2)` in the keypoint
/// count, so the cap is scaled by `sqrt(budget / measured_mean)`: halving the
/// budget costs ~`1/sqrt(2)` the keypoints, which halves the quadratic term.
/// The result is clamped to the extractor's configured cap (never raised past
/// it) and to a small floor so tracking keeps enough structure. A single
/// request is smoothed toward the previous cap to avoid oscillation. The call
/// is a no-op for extractors without a runtime cap.
#[cfg(feature = "onnx-inference")]
pub(super) fn apply_adaptive_keypoint_cap(
    extractor: &DemoExtractor,
    budget_ms: f64,
    history: &[f64],
) {
    let DemoExtractor::SuperPointOnnx(onnx) = extractor else {
        return;
    };
    if !budget_ms.is_finite() || budget_ms <= 0.0 {
        return;
    }
    // Need a few frames of history to smooth over one-off decode stutters.
    const WINDOW: usize = 8;
    if history.len() < 4 {
        return;
    }
    let recent = &history[history.len().saturating_sub(WINDOW)..];
    let mean_ms = recent.iter().sum::<f64>() / recent.len() as f64;
    if !mean_ms.is_finite() || mean_ms <= 0.0 {
        return;
    }
    let configured_cap = onnx.config().max_keypoints;
    let previous_cap = onnx.effective_max_keypoints().min(configured_cap);
    let floor = (configured_cap / 4).max(256).min(configured_cap);
    // A single reciprocal correction overshoots: the matcher is quadratic, but
    // lowering the cap also removes structure the tracker needs, and the
    // frame-time history lags by the window. Apply a damped correction that
    // asymptotes to the budget instead of tracking it exactly.
    let ratio = budget_ms / mean_ms;
    let scale = if ratio >= 1.0 {
        // Under budget: recover toward the configured cap slowly.
        (1.0 + (ratio - 1.0) * 0.5).min(1.25)
    } else {
        // Over budget: reduce, but never by more than 10% per frame to keep
        // the keypoint count (and tracking) stable.
        (1.0 - (1.0 - ratio) * 0.5).clamp(0.9, 1.0)
    };
    let requested = ((previous_cap as f64) * scale).round() as usize;
    let target = requested.clamp(floor, configured_cap);
    if target != previous_cap {
        onnx.set_runtime_max_keypoints(target);
    }
}

/// No-op for builds without ONNX inference (the extractor type has no runtime
/// cap).
#[cfg(not(feature = "onnx-inference"))]
pub(super) fn apply_adaptive_keypoint_cap(
    _extractor: &DemoExtractor,
    _budget_ms: f64,
    _history: &[f64],
) {
}

/// modulo the clone.
///
/// A keypoint whose `undistort_pixel` returns `None` (camera intrinsics
/// missing, which should not happen for the EuRoC `Pinhole` camera) is
/// silently dropped along with its descriptor — keypoints / descriptors
/// must stay in lock-step so a half-failed undistortion can never poison
/// the matcher's index-based lookup.
#[cfg(feature = "image-io")]
pub(super) fn undistort_feature_keypoints(
    distortion: &RadialTangential,
    camera: &Camera,
    features: &FeatureSet,
) -> FeatureSet {
    if distortion.is_identity() {
        return features.clone();
    }
    let mut keypoints = Vec::with_capacity(features.keypoints.len());
    let mut descriptors = Vec::with_capacity(features.descriptors.len());
    for (kp, desc) in features.keypoints.iter().zip(features.descriptors.iter()) {
        let Some(undistorted) = distortion.undistort_pixel(camera, *kp) else {
            continue;
        };
        keypoints.push(undistorted);
        descriptors.push(desc.clone());
    }
    FeatureSet {
        keypoints,
        descriptors,
    }
}

/// Same-timestamp cam0→cam1 LK for OF without touching the temporal tracker.
///
/// Returns a cam1 [`FeatureSet`] whose descriptors are copies of the matched
/// cam0 track-id descriptors so replenish / rebootstrap matcher paths keep
/// working. Correspondences also pass Basalt's epipolar gate.
#[cfg(feature = "image-io")]
#[allow(clippy::too_many_arguments)]
pub(super) fn optical_flow_stereo_features_for_frame(
    of_extractor: &visloc_rs::vision::optical_flow::OpticalFlowFeatureExtractor,
    cam0_image: &GrayscaleImage,
    cam1_image: &GrayscaleImage,
    features_raw: &FeatureSet,
    features_undist: &FeatureSet,
    cam0_distortion: &RadialTangential,
    cam0_camera: &Camera,
    cam1_distortion: &RadialTangential,
    cam1_camera: &Camera,
    cam0_to_cam1: &SE3,
    epipolar_thresh: f64,
) -> Option<FeatureSet> {
    let left_pts: Vec<(f32, f32)> = features_raw
        .keypoints
        .iter()
        .map(|p| (p.x as f32, p.y as f32))
        .collect();
    let tracked = of_extractor.track_points_to(cam0_image, cam1_image, &left_pts);

    let mut right_undist_by_raw = vec![None; features_raw.keypoints.len()];
    for (raw_i, right) in tracked.into_iter().enumerate() {
        let Some((rx, ry)) = right else {
            continue;
        };
        right_undist_by_raw[raw_i] =
            cam1_distortion.undistort_pixel(cam1_camera, Point2::new(rx as f64, ry as f64));
    }

    let mut correspondences = Vec::new();
    let mut left_undist_idx = 0usize;
    for (raw_i, kp) in features_raw.keypoints.iter().enumerate() {
        let Some(undist_left) = cam0_distortion.undistort_pixel(cam0_camera, *kp) else {
            continue;
        };
        debug_assert_eq!(features_undist.keypoints[left_undist_idx], undist_left);
        if let Some(undist_right) = right_undist_by_raw[raw_i] {
            correspondences.push((left_undist_idx, undist_right));
        }
        left_undist_idx += 1;
    }
    let correspondences = visloc_rs::vision::stereo_bootstrap::filter_correspondences_by_epipolar(
        cam0_camera,
        cam1_camera,
        cam0_to_cam1,
        &features_undist.keypoints,
        &correspondences,
        epipolar_thresh,
    );
    if correspondences.is_empty() {
        return None;
    }

    let mut keypoints = Vec::with_capacity(correspondences.len());
    let mut descriptors = Vec::with_capacity(correspondences.len());
    for &(left_idx, right_px) in &correspondences {
        keypoints.push(right_px);
        descriptors.push(features_undist.descriptors[left_idx].clone());
    }
    FeatureSet::new(keypoints, descriptors).ok()
}

/// Adapt a [`FeatureSet`] into a [`Frame`] addressed to the cam0 camera id.
#[cfg(feature = "image-io")]
pub(super) fn frame_from_features(frame_id: u64, camera_id: u64, features: &FeatureSet) -> Frame {
    let mut frame = Frame::new(frame_id, camera_id);
    frame.keypoints = features.keypoints.clone();
    frame.descriptors = features.descriptors.clone();
    frame
}

#[cfg(feature = "image-io")]
pub(super) fn normalized_mean_descriptor(descriptors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = descriptors.first()?;
    if first.is_empty() {
        return None;
    }
    let dim = first.len();
    let mut mean = vec![0.0f32; dim];
    let mut count = 0usize;
    for descriptor in descriptors {
        if descriptor.len() != dim {
            return None;
        }
        for (acc, value) in mean.iter_mut().zip(descriptor) {
            *acc += *value;
        }
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let inv = 1.0 / count as f32;
    for value in &mut mean {
        *value *= inv;
    }
    let norm = mean.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut mean {
            *value /= norm;
        }
    }
    Some(mean)
}
