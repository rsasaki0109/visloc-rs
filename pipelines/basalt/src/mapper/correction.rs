//! Applying the mapper's keyframe corrections to a full VIO trajectory.
//!
//! The mapper optimizes keyframe poses only. These pure functions carry its
//! corrections over to every VIO frame, for the online demo
//! (`examples/basalt_euroc_online_slam_demo.rs`) and the ROS 2 VIO node.
//! They moved here from the demo unchanged, so the demo's outputs are the
//! same as before.
//!
//! Poses are body-to-world (`T_w_i`); a VIO trajectory maps each frame id
//! to its `(timestamp_ns, pose)`, and the mapper poses map keyframe ids to
//! their optimized pose.

use std::collections::BTreeMap;

use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use thiserror::Error;
use visloc_core::geometry::SE3;

use crate::mapper::NfrMapperResult;

/// Why the mapper's keyframe poses cannot be propagated.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PropagationError {
    #[error("mapper produced no keyframe poses")]
    NoKeyframes,
    #[error(
        "{count} mapper keyframe frame_ids are absent from the VIO trajectory (first few: {first:?})"
    )]
    MissingKeyframes { count: usize, first: Vec<u64> },
}

/// The correction that moves the VIO's estimate of a keyframe onto the
/// mapper's: `Delta_k = M_k * V_k^{-1}`.
///
/// It maps the VIO's world frame into the mapper's, so in ROS terms it is
/// `T_map_odom` when the VIO world frame is `odom`.
pub fn keyframe_correction(mapper_pose: &SE3, vio_pose: &SE3) -> SE3 {
    mapper_pose.compose(&vio_pose.inverse())
}

/// Rigid spanning-tree propagation of the mapper's keyframe corrections to
/// every VIO frame. A direct Rust port of
/// `scripts/propagate_basalt_mapper_corrections.py::propagate` (kept
/// numerically equivalent, not merely similar -- see that script's
/// module docstring for the exact convention):
///
/// ```text
/// V_f  = raw VIO body-to-world pose at frame f (rotation R_f, translation t_f)
/// M_k  = mapper-corrected body-to-world pose at the nearest preceding
///        keyframe k (frame_id <= f)
/// V_k  = raw VIO body-to-world pose at that same keyframe k
///
/// corrected(f) = Delta_k * V_f,  where Delta_k = M_k * V_k^{-1}
/// ```
///
/// i.e. each frame keeps its VIO-derived relative motion to the nearest
/// preceding keyframe; only the keyframe's mapper correction moves it. This
/// is exact at keyframes themselves (f == k, `Delta_k * V_k == M_k`) and,
/// like the Python original, uses the *first* keyframe's delta for any
/// frame before it (there is no earlier keyframe to interpolate from).
pub fn propagate_to_all_frames(
    vio_trajectory: &BTreeMap<u64, (i64, SE3)>,
    mapper_poses: &BTreeMap<u64, SE3>,
) -> Result<Vec<(i64, SE3)>, PropagationError> {
    let keyframe_ids: Vec<u64> = mapper_poses.keys().copied().collect();
    if keyframe_ids.is_empty() {
        return Err(PropagationError::NoKeyframes);
    }
    let missing: Vec<u64> = keyframe_ids
        .iter()
        .copied()
        .filter(|k| !vio_trajectory.contains_key(k))
        .collect();
    if !missing.is_empty() {
        return Err(PropagationError::MissingKeyframes {
            count: missing.len(),
            first: missing[..missing.len().min(5)].to_vec(),
        });
    }

    let delta_by_keyframe: BTreeMap<u64, SE3> = keyframe_ids
        .iter()
        .map(|&k| {
            let (_, v_k) = &vio_trajectory[&k];
            let m_k = &mapper_poses[&k];
            (k, keyframe_correction(m_k, v_k))
        })
        .collect();

    let mut output = Vec::with_capacity(vio_trajectory.len());
    let mut cursor = 0usize;
    for (&frame_id, (timestamp_ns, v_f)) in vio_trajectory {
        while cursor + 1 < keyframe_ids.len() && keyframe_ids[cursor + 1] <= frame_id {
            cursor += 1;
        }
        let delta = &delta_by_keyframe[&keyframe_ids[cursor]];
        output.push((*timestamp_ns, delta.compose(v_f)));
    }
    Ok(output)
}

/// Like [`propagate_to_all_frames`], but the correction applied to a frame
/// between keyframes `k0 < f < k1` is interpolated on SE(3) by timestamp:
///
/// ```text
/// Delta_f = Delta_k0 * Exp(alpha * Log(Delta_k0^{-1} * Delta_k1)),
/// alpha   = (t_f - t_k0) / (t_k1 - t_k0)
/// corrected(f) = Delta_f * V_f
/// ```
///
/// The piecewise-constant scheme applies `Delta_k0` all the way up to `k1`,
/// then jumps to `Delta_k1`. That discontinuity sits at every keyframe
/// boundary and shows up as relative-pose error. Keyframes themselves are
/// still exact, and frames before the first or after the last keyframe use
/// that keyframe's correction.
pub fn propagate_interpolated(
    vio_trajectory: &BTreeMap<u64, (i64, SE3)>,
    mapper_poses: &BTreeMap<u64, SE3>,
) -> Result<Vec<(i64, SE3)>, PropagationError> {
    // Reuse the validation of the piecewise-constant propagation.
    propagate_to_all_frames(vio_trajectory, mapper_poses)?;
    let keyframes: Vec<(u64, i64, SE3)> = mapper_poses
        .iter()
        .map(|(&k, m_k)| {
            let (t_k, v_k) = &vio_trajectory[&k];
            (k, *t_k, keyframe_correction(m_k, v_k))
        })
        .collect();
    let mut output = Vec::with_capacity(vio_trajectory.len());
    let mut cursor = 0usize;
    for (&frame_id, (timestamp_ns, v_f)) in vio_trajectory {
        while cursor + 1 < keyframes.len() && keyframes[cursor + 1].0 <= frame_id {
            cursor += 1;
        }
        let (k0, t0, delta0) = &keyframes[cursor];
        let delta = match keyframes.get(cursor + 1) {
            Some((_, t1, delta1)) if frame_id > *k0 && t1 > t0 => {
                let alpha = (*timestamp_ns - t0) as f64 / (t1 - t0) as f64;
                let step = delta0.inverse().compose(delta1).log() * alpha;
                delta0.compose(&SE3::exp(&step))
            }
            _ => delta0.clone(),
        };
        output.push((*timestamp_ns, delta.compose(v_f)));
    }
    Ok(output)
}

/// The keyframe poses of a mapper result, keyed by frame id.
pub fn result_keyframe_poses(result: &NfrMapperResult) -> BTreeMap<u64, SE3> {
    result
        .poses
        .iter()
        .map(|record| {
            let [qw, qx, qy, qz] = record.quaternion_wxyz;
            let pose = SE3::new(
                UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz)),
                Vector3::new(
                    record.translation[0],
                    record.translation[1],
                    record.translation[2],
                ),
            );
            (record.frame_id, pose)
        })
        .collect()
}

/// A TUM trajectory (`timestamp tx ty tz qx qy qz qw`, seconds), with the
/// header line and number format the online demo writes.
pub fn tum_string(poses: &[(i64, SE3)]) -> String {
    let mut buffer = String::from("# timestamp tx ty tz qx qy qz qw\n");
    for (timestamp_ns, pose) in poses {
        let q = pose.rotation.quaternion();
        buffer.push_str(&format!(
            "{:.18e} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e}\n",
            *timestamp_ns as f64 * 1e-9,
            pose.translation.x,
            pose.translation.y,
            pose.translation.z,
            q.i,
            q.j,
            q.k,
            q.w,
        ));
    }
    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    fn se3(tx: f64, ty: f64, tz: f64, yaw_deg: f64) -> SE3 {
        SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, yaw_deg.to_radians()),
            Vector3::new(tx, ty, tz),
        )
    }

    #[test]
    fn keyframe_correction_maps_the_vio_pose_onto_the_mapper_pose() {
        let vio = se3(1.0, 2.0, 0.5, 30.0);
        let mapper = se3(1.2, 1.9, 0.5, 33.0);
        let delta = keyframe_correction(&mapper, &vio);
        let moved = delta.compose(&vio);
        assert!((moved.translation - mapper.translation).norm() < 1e-12);
        assert!(moved.rotation.angle_to(&mapper.rotation) < 1e-12);
    }

    #[test]
    fn propagation_errors_name_the_problem() {
        let vio = BTreeMap::from([(0, (0, se3(0.0, 0.0, 0.0, 0.0)))]);
        assert_eq!(
            propagate_interpolated(&vio, &BTreeMap::new()),
            Err(PropagationError::NoKeyframes)
        );
        let mapper = BTreeMap::from([(7, se3(0.0, 0.0, 0.0, 0.0))]);
        let error = propagate_to_all_frames(&vio, &mapper).unwrap_err();
        assert_eq!(
            error.to_string(),
            "1 mapper keyframe frame_ids are absent from the VIO trajectory (first few: [7])"
        );
    }

    #[test]
    fn result_poses_round_trip_through_the_record_layout() {
        let pose = se3(1.0, -2.0, 3.0, 40.0);
        let q = pose.rotation.quaternion();
        let result = NfrMapperResult {
            poses: vec![crate::mapper::NfrMapperPoseRecord {
                frame_id: 5,
                timestamp_ns: 50,
                quaternion_wxyz: [q.w, q.i, q.j, q.k],
                translation: [1.0, -2.0, 3.0],
            }],
            landmarks: Vec::new(),
        };
        let poses = result_keyframe_poses(&result);
        assert_eq!(poses.len(), 1);
        assert!((poses[&5].translation - pose.translation).norm() < 1e-12);
        assert!(poses[&5].rotation.angle_to(&pose.rotation) < 1e-12);
        let tum = tum_string(&[(1_500_000_000, pose)]);
        assert!(tum.starts_with("# timestamp tx ty tz qx qy qz qw\n1.5"));
        assert_eq!(tum.lines().count(), 2);
    }
}
