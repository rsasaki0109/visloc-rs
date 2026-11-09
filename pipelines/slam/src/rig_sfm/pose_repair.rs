//! Post-registration repair of isolated pose outliers and paired pose jumps.

use super::*;

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct IsolatedPoseRepairUpdate {
    pub(super) passes: usize,
    pub(super) repairs: usize,
}

/// Repair only a one-frame translation detour with registered temporal
/// neighbours. Candidates are collected from an immutable pass snapshot and
/// installed together, so ordering cannot affect the result. The rotation is
/// replaced by the midpoint SLERP as well, preserving a coherent SE(3) pose.
#[allow(clippy::too_many_arguments)]
pub(super) fn repair_isolated_pose_outliers(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    seed_frame_index: usize,
    max_midpoint_error_m: f64,
    min_detour_ratio: f64,
    max_passes: usize,
    frame_poses: &mut [Option<Pose>],
    image_poses: &mut [Option<Pose>],
) -> IsolatedPoseRepairUpdate {
    let mut update = IsolatedPoseRepairUpdate::default();
    for _ in 0..max_passes {
        let mut candidates = Vec::new();
        for frame in 1..frame_poses.len().saturating_sub(1) {
            if frame == seed_frame_index {
                continue;
            }
            let (Some(left), Some(current), Some(right)) = (
                frame_poses[frame - 1].as_ref(),
                frame_poses[frame].as_ref(),
                frame_poses[frame + 1].as_ref(),
            ) else {
                continue;
            };
            let left = left.camera_to_world();
            let current = current.camera_to_world();
            let right = right.camera_to_world();
            let midpoint = (left.translation + right.translation) * 0.5;
            let midpoint_error = (current.translation - midpoint).norm();
            let endpoint_chord = (right.translation - left.translation).norm();
            let detour = (current.translation - left.translation).norm()
                + (right.translation - current.translation).norm();
            // One millimetre avoids an unstable ratio for a stationary rig;
            // the independent midpoint-error gate still controls scale.
            if midpoint_error <= max_midpoint_error_m
                || detour <= min_detour_ratio * endpoint_chord.max(1.0e-3)
            {
                continue;
            }
            candidates.push((
                frame,
                Pose {
                    world_to_camera: SE3::new(left.rotation.slerp(&right.rotation, 0.5), midpoint)
                        .inverse(),
                },
            ));
        }
        if candidates.is_empty() {
            break;
        }
        update.passes += 1;
        update.repairs += candidates.len();
        for (frame, pose) in candidates {
            frame_poses[frame] = Some(pose);
            install_image_poses(
                rig,
                &frames[frame],
                frame_poses[frame]
                    .as_ref()
                    .expect("repaired pose was just installed"),
                image_poses,
            );
        }
    }
    update
}

#[derive(Debug, Clone, Copy)]
struct PairedPoseJumpCandidate {
    first_jump: usize,
    second_jump: usize,
    closure_ratio: f64,
    offset: Vector3<f64>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct PairedPoseJumpRepairUpdate {
    pub(super) repairs: usize,
    pub(super) repaired_frames: usize,
}

/// Repair short, mutually cancelling translation jumps in one registered
/// model.  If `v_i = C_(i+1) - C_i` and `v_j = C_(j+1) - C_j` are large but
/// nearly opposite, the observations are consistent with a world-frame
/// offset applied to the contiguous segment `i+1..=j`.  The symmetric offset
/// estimate `(v_i - v_j) / 2` is removed from every pose in that segment,
/// preserving its internal motion and each pose's rotation.
///
/// Candidate generation examines at most `max_frame_span` successors for each
/// frame.  Missing poses terminate a candidate interval, so model boundaries
/// are never crossed.  Candidate selection is deterministic and temporal: for
/// each first jump, the eligible second jump with the smallest closure ratio
/// (then the smallest frame index) wins, and overlapping intervals are
/// discarded.  Thus the
/// repair work and temporary state are bounded by the sparse `F * span`
/// envelope rather than an image/track Cartesian product.
#[allow(clippy::too_many_arguments)]
pub(super) fn repair_paired_pose_jumps(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    seed_frame_index: usize,
    absolute_step_m: f64,
    min_step_ratio: f64,
    max_frame_span: usize,
    max_closure_ratio: f64,
    frame_poses: &mut [Option<Pose>],
    image_poses: &mut [Option<Pose>],
) -> PairedPoseJumpRepairUpdate {
    let frame_count = frame_poses.len().min(frames.len());
    if frame_count < 3 || max_frame_span == 0 {
        return PairedPoseJumpRepairUpdate::default();
    }

    // Build the local-velocity vector once.  An absent pose leaves a hole,
    // which is also the model-boundary marker used by candidate generation.
    let mut velocities = vec![None; frame_count.saturating_sub(1)];
    let mut step_lengths = Vec::new();
    for frame in 0..frame_count.saturating_sub(1) {
        let (Some(left), Some(right)) =
            (frame_poses[frame].as_ref(), frame_poses[frame + 1].as_ref())
        else {
            continue;
        };
        let left = left.camera_center_world();
        let right = right.camera_center_world();
        let velocity = right - left;
        let length = velocity.norm();
        if !velocity.iter().all(|value| value.is_finite()) || !length.is_finite() {
            continue;
        }
        velocities[frame] = Some(velocity);
        step_lengths.push(length);
    }
    if step_lengths.is_empty() {
        return PairedPoseJumpRepairUpdate::default();
    }
    step_lengths.sort_unstable_by(f64::total_cmp);
    let midpoint = step_lengths.len() / 2;
    let global_median_step = if step_lengths.len() % 2 == 0 {
        (step_lengths[midpoint - 1] + step_lengths[midpoint]) * 0.5
    } else {
        step_lengths[midpoint]
    };
    let required_step = absolute_step_m.max(min_step_ratio * global_median_step);
    if !global_median_step.is_finite() || !required_step.is_finite() || required_step <= 0.0 {
        return PairedPoseJumpRepairUpdate::default();
    }

    // Prefix counts make the all-registered interval test O(1), keeping the
    // total detector cost linear in the configured span.
    let mut missing_prefix = vec![0usize; frame_count + 1];
    for frame in 0..frame_count {
        missing_prefix[frame + 1] =
            missing_prefix[frame] + usize::from(frame_poses[frame].is_none());
    }
    let interval_is_registered =
        |start: usize, end: usize| missing_prefix[end + 1] == missing_prefix[start];

    let mut candidates = Vec::new();
    for first_jump in 0..velocities.len() {
        let Some(first_velocity) = velocities[first_jump] else {
            continue;
        };
        let first_length = first_velocity.norm();
        if first_length < required_step {
            continue;
        }
        let last_jump = first_jump
            .saturating_add(max_frame_span)
            .min(velocities.len().saturating_sub(1));
        let mut best = None;
        for (second_jump, second_velocity) in velocities
            .iter()
            .enumerate()
            .take(last_jump + 1)
            .skip(first_jump + 1)
        {
            // A missing pose can only make later intervals invalid until the
            // right endpoint; avoid repeatedly scanning the same gap.
            if second_velocity.is_none() || !interval_is_registered(first_jump, second_jump + 1) {
                continue;
            }
            if first_jump < seed_frame_index && seed_frame_index <= second_jump {
                // The seed is a gauge anchor.  It may be an outer endpoint,
                // but it must not be shifted as part of the repaired segment.
                continue;
            }
            let second_velocity =
                second_velocity.expect("the candidate's second velocity was checked above");
            let second_length = second_velocity.norm();
            if second_length < required_step {
                continue;
            }
            let denominator = first_length.max(second_length);
            if denominator <= 0.0 || !denominator.is_finite() {
                continue;
            }
            let closure_ratio = (first_velocity + second_velocity).norm() / denominator;
            if !closure_ratio.is_finite() || closure_ratio > max_closure_ratio {
                continue;
            }
            let candidate = PairedPoseJumpCandidate {
                first_jump,
                second_jump,
                closure_ratio,
                offset: (first_velocity - second_velocity) * 0.5,
            };
            let replace = best.is_none_or(|current: PairedPoseJumpCandidate| {
                closure_ratio.total_cmp(&current.closure_ratio).is_lt()
                    || (closure_ratio.total_cmp(&current.closure_ratio).is_eq()
                        && second_jump < current.second_jump)
            });
            if replace {
                best = Some(candidate);
            }
        }
        if let Some(candidate) = best {
            candidates.push(candidate);
        }
    }

    // `candidates` has one entry per first jump and is therefore already in
    // temporal order.  Reserve the full candidate interval, including its
    // endpoints, so adjacent repairs cannot share an untrusted boundary.
    let mut occupied = vec![false; frame_count];
    let mut selected = Vec::new();
    for candidate in candidates {
        let interval_end = candidate.second_jump + 1;
        if occupied[candidate.first_jump..=interval_end]
            .iter()
            .any(|occupied| *occupied)
        {
            continue;
        }
        occupied[candidate.first_jump..=interval_end].fill(true);
        selected.push(candidate);
    }

    // Construct every replacement from the immutable pass snapshot before
    // writing any pose.  If an unexpected non-finite state appears, abort the
    // whole repair transaction rather than exposing a partially shifted map.
    let mut pending = Vec::new();
    for candidate in &selected {
        for (frame, current) in frame_poses
            .iter()
            .enumerate()
            .take(candidate.second_jump + 1)
            .skip(candidate.first_jump + 1)
        {
            let Some(current) = current.as_ref() else {
                return PairedPoseJumpRepairUpdate::default();
            };
            let current_center = current.camera_center_world();
            let repaired_center = current_center - candidate.offset;
            if !repaired_center.coords.iter().all(|value| value.is_finite()) {
                return PairedPoseJumpRepairUpdate::default();
            }
            let rotation = current.world_to_camera.rotation;
            let translation = -rotation.transform_vector(&repaired_center.coords);
            pending.push((frame, Pose::from_world_to_camera(rotation, translation)));
        }
    }
    let update = PairedPoseJumpRepairUpdate {
        repairs: selected.len(),
        repaired_frames: pending.len(),
    };
    for (frame, pose) in pending {
        frame_poses[frame] = Some(pose);
        install_image_poses(
            rig,
            &frames[frame],
            frame_poses[frame]
                .as_ref()
                .expect("repaired pose was just installed"),
            image_poses,
        );
    }
    update
}
