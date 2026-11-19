use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
use serde_json::{json, Value};
use visloc_core::geometry::{Pose, SE3};
use visloc_rs::localization::{inspect_map_registration, MapMergeConfig, SessionConstraint};
use visloc_rs::slam::pose_graph::{
    relative_world_to_camera, LinearSolver, PoseGraph, PoseGraphEdge, PoseGraphEdgeKind,
    PoseGraphSe3Config, RobustKernel,
};

use super::map_files::SavedMap;

struct Frame {
    node: u64,
    camera_from_reference: SE3,
}

fn cycle_consistent(count: usize, edges: &[SessionConstraint], candidate: usize) -> bool {
    let target = &edges[candidate];
    let mut placements = vec![None; count];
    placements[target.base] = Some(SE3::identity());
    let mut queue = VecDeque::from([target.base]);
    while let Some(session) = queue.pop_front() {
        let pose = placements[session].clone().expect("visited session");
        for (index, edge) in edges.iter().enumerate() {
            if index == candidate {
                continue;
            }
            let (other, relative) = if edge.base == session {
                (edge.other, edge.base_from_other.clone())
            } else if edge.other == session {
                (edge.base, edge.base_from_other.inverse())
            } else {
                continue;
            };
            if placements[other].is_none() {
                placements[other] = Some(pose.compose(&relative));
                queue.push_back(other);
            }
        }
    }
    placements[target.other].as_ref().is_none_or(|independent| {
        let error = independent.inverse().compose(&target.base_from_other);
        error.translation.norm() <= 0.25 && error.rotation.angle().to_degrees() <= 5.0
    })
}

pub(super) fn refine(
    maps: &mut [SavedMap],
    placements: &[SE3],
    edges: &[SessionConstraint],
    directions: &[(usize, usize)],
    config: &MapMergeConfig,
    min_separation_m: f64,
    excluded_loops: &BTreeSet<[u64; 4]>,
) -> Result<Value, Box<dyn std::error::Error>> {
    let mut graph = PoseGraph::new();
    let mut frames = Vec::new();
    for (map, placement) in maps.iter().zip(placements) {
        let mut rig_frames = BTreeMap::<u64, Vec<_>>::new();
        for image in &map.images {
            if let Some(pose) = map
                .map
                .keyframes
                .get(&image.image_id)
                .and_then(|k| k.frame.pose.as_ref())
            {
                rig_frames.entry(image.frame_id).or_default().push((
                    image.cam_id,
                    image.image_id,
                    pose,
                ));
            }
        }
        if rig_frames.is_empty() {
            return Err(
                "keyframe refinement requires timestamped rig frame IDs in images.csv".into(),
            );
        }
        let mut by_image = HashMap::new();
        let mut previous = None;
        for views in rig_frames.values_mut() {
            views.sort_by_key(|view| (view.0, view.1));
            let reference = views[0].2;
            let node = graph.poses.len() as u64;
            let pose = Pose {
                world_to_camera: reference.world_to_camera.compose(&placement.inverse()),
            };
            graph.add_pose(node, pose);
            if let Some(from) = previous {
                let relative = relative_world_to_camera(&graph.poses[&from], &graph.poses[&node]);
                graph.add_sequential_edge(from, node, relative);
            } else if node == 0 {
                graph.anchor(0);
            } else {
                // Keep an initialization prior for sessions with sparse or absent loops.
                let measurement = relative_world_to_camera(&graph.poses[&0], &graph.poses[&node]);
                graph.edges.push(PoseGraphEdge {
                    from: 0,
                    to: node,
                    measurement,
                    kind: PoseGraphEdgeKind::Sequential,
                    weight: 1.0,
                    information: None,
                });
            }
            for (_, image, pose) in views {
                by_image.insert(
                    *image,
                    Frame {
                        node,
                        camera_from_reference: relative_world_to_camera(reference, pose),
                    },
                );
            }
            previous = Some(node);
        }
        frames.push(by_image);
    }

    let mut selected = 0;
    let mut matched_exclusions = BTreeSet::new();
    let mut pairs = Vec::new();
    let mut refined_sessions = vec![false; maps.len()];
    for (edge_index, &(base, query)) in directions.iter().enumerate() {
        if !cycle_consistent(maps.len(), edges, edge_index) {
            pairs.push(json!({"base": base, "query": query, "selected": 0, "reason": "cycle_inconsistent"}));
            continue;
        }
        let report = inspect_map_registration(
            &maps[base].map,
            &maps[base].descriptors,
            &maps[query].map,
            &maps[query].descriptors,
            config,
        );
        let query_frames = frames[query].len();
        let localized_queries = report
            .registration
            .as_ref()
            .ok()
            .map(|r| r.localized_keyframes);
        let mut candidates = report.candidates;
        let candidate_count = candidates.len();
        candidates.sort_by(|a, b| {
            b.base_keyframe_inliers
                .cmp(&a.base_keyframe_inliers)
                .then(a.other_keyframe_id.cmp(&b.other_keyframe_id))
                .then(a.base_keyframe_id.cmp(&b.base_keyframe_id))
        });
        let mut chosen: Vec<PoseGraphEdge> = Vec::new();
        let mut image_pairs = Vec::new();
        let mut measurements = Vec::new();
        let mut rejected = BTreeMap::<&str, usize>::new();
        let mut pose_consistent = 0;
        for candidate in &candidates {
            let reason = if !candidate.agrees_with_registration {
                Some("registration_disagreement")
            } else if candidate.base_keyframe_inliers < (config.min_inliers / 2).max(6) {
                Some("anchor_inliers")
            } else if !candidate
                .mean_reprojection_error_px
                .is_some_and(|error| error <= 2.0)
            {
                Some("reprojection")
            } else {
                None
            };
            if let Some(reason) = reason {
                *rejected.entry(reason).or_default() += 1;
                continue;
            }
            let (Some(from), Some(to)) = (
                frames[base].get(&candidate.base_keyframe_id),
                frames[query].get(&candidate.other_keyframe_id),
            ) else {
                *rejected.entry("missing_rig_frame").or_default() += 1;
                continue;
            };
            let measurement = to
                .camera_from_reference
                .inverse()
                .compose(&candidate.other_camera_from_base_camera)
                .compose(&from.camera_from_reference);
            let prediction =
                relative_world_to_camera(&graph.poses[&from.node], &graph.poses[&to.node]);
            let residual = prediction.inverse().compose(&measurement);
            if residual.translation.norm() > 0.12 {
                *rejected.entry("placement_translation").or_default() += 1;
                continue;
            }
            if residual.rotation.angle().to_degrees() > 4.0 {
                *rejected.entry("placement_rotation").or_default() += 1;
                continue;
            }
            pose_consistent += 1;
            let from_center = graph.poses[&from.node].camera_center_world();
            let to_center = graph.poses[&to.node].camera_center_world();
            if chosen.iter().any(|prior| {
                (from_center - graph.poses[&prior.from].camera_center_world()).norm()
                    < min_separation_m
                    || (to_center - graph.poses[&prior.to].camera_center_world()).norm()
                        < min_separation_m
            }) {
                *rejected.entry("spatial_separation").or_default() += 1;
                continue;
            }
            chosen.push(PoseGraphEdge {
                from: from.node,
                to: to.node,
                measurement,
                kind: PoseGraphEdgeKind::LoopClosure,
                weight: 0.1,
                information: None,
            });
            image_pairs.push([candidate.base_keyframe_id, candidate.other_keyframe_id]);
            measurements.push(json!({
                "base_image": candidate.base_keyframe_id,
                "query_image": candidate.other_keyframe_id,
                "query_camera_from_base_camera": super::map_files::se3_json(&candidate.other_camera_from_base_camera),
                "anchor_inliers": candidate.base_keyframe_inliers,
                "pnp_inliers": candidate.inlier_count,
                "mean_reprojection_error_px": candidate.mean_reprojection_error_px,
            }));
        }
        let spatially_separated = chosen.len();
        let mut retained = Vec::new();
        let mut retained_images = Vec::new();
        for (edge, [a, b]) in chosen.into_iter().zip(image_pairs) {
            let id = [base as u64, query as u64, a, b];
            if excluded_loops.contains(&id) {
                matched_exclusions.insert(id);
            } else {
                retained.push(edge);
                retained_images.push([a, b]);
            }
        }
        let count = if retained.len() >= 2 {
            let count = retained.len();
            graph.edges.extend(retained);
            refined_sessions[base] = true;
            refined_sessions[query] = true;
            count
        } else {
            retained_images.clear();
            0
        };
        selected += count;
        pairs.push(json!({
            "base": base, "query": query, "selected": count, "image_pairs": retained_images, "measurements": measurements,
            "query_frames": query_frames, "localized_queries": localized_queries,
            "anchored_candidates": candidate_count, "pose_consistent": pose_consistent,
            "spatially_separated": spatially_separated, "rejected": rejected,
        }));
    }
    if let Some(id) = excluded_loops.difference(&matched_exclusions).next() {
        return Err(
            format!("excluded keyframe loop was not present in the selected set: {id:?}").into(),
        );
    }
    let mut report = json!({"excluded_loops": matched_exclusions, "selected_loops": selected, "pairs": pairs, "retriangulated": 0,
                           "min_separation_m": min_separation_m, "applied": false});
    if selected == 0 {
        println!("keyframe refinement: no spatially separated loop measurements");
        return Ok(report);
    }
    let initial_poses = graph.poses.clone();
    let kernel = RobustKernel::Huber { delta: 0.05 };
    let result = graph.optimize_se3_iterative(&PoseGraphSe3Config {
        linear_solver: LinearSolver::Sparse,
        initial_lambda: Some(1e-4),
        robust_kernel: kernel,
        chordal_init: false,
        max_iterations: 15,
        ..Default::default()
    })?;
    // Retain the original tilt of each rig. Project world-frame rotation
    // corrections onto yaw about gravity while keeping solved camera centres.
    for (id, pose) in &mut graph.poses {
        if *id == 0 {
            continue;
        }
        let before = initial_poses[id].camera_to_world();
        let after = pose.camera_to_world();
        let rotation_delta = after.rotation * before.rotation.inverse();
        let matrix = rotation_delta.to_rotation_matrix();
        let yaw = matrix[(1, 0)].atan2(matrix[(0, 0)]);
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw) * before.rotation;
        pose.world_to_camera = SE3::new(rotation, after.translation).inverse();
    }
    let applied_cost = graph.robust_se3_cost(&kernel);
    report["initial_cost"] = json!(result.initial_cost);
    report["solver_cost"] = json!(result.final_cost);
    report["gravity_projected_cost"] = json!(applied_cost);
    if !applied_cost.is_finite() || applied_cost >= result.initial_cost {
        println!("keyframe refinement: gravity-preserving update did not lower the graph cost");
        return Ok(report);
    }
    let mut rebuilt = 0;
    let mut maximum: f64 = 0.0;
    for (((map, placement), image_frames), enabled) in maps
        .iter_mut()
        .zip(placements)
        .zip(&frames)
        .zip(refined_sessions)
    {
        if !enabled {
            continue;
        }
        let mut old = HashMap::new();
        for (image, frame) in image_frames {
            let keyframe = map.map.keyframes.get_mut(image).expect("known image");
            let refined = Pose {
                world_to_camera: frame
                    .camera_from_reference
                    .compose(&graph.poses[&frame.node].world_to_camera)
                    .compose(placement),
            };
            if let Some(previous) = keyframe.frame.pose.replace(refined) {
                maximum = maximum.max(
                    (previous.camera_center_world()
                        - keyframe
                            .frame
                            .pose
                            .as_ref()
                            .expect("refined pose")
                            .camera_center_world())
                    .norm(),
                );
                old.insert(*image, previous);
            }
        }
        map.map.landmark_position_covariances.clear();
        for landmark in map.map.landmarks.values_mut() {
            let mut normal = Matrix3::zeros();
            let mut rhs = Vector3::zeros();
            let mut views = Vec::new();
            let mut preserved = Vec::new();
            for obs in &landmark.observations {
                let Some(previous) = old.get(&obs.frame_id) else {
                    continue;
                };
                let keyframe = &map.map.keyframes[&obs.frame_id];
                let pose = keyframe.frame.pose.as_ref().expect("refined pose");
                let Some(camera) = map.map.cameras.get(&keyframe.frame.camera_id) else {
                    continue;
                };
                let Some(ray) = camera.unit_ray_from_pixel(&obs.xy) else {
                    continue;
                };
                let world = pose.camera_to_world();
                let direction = world.rotation * ray.normalize();
                let projector = Matrix3::identity() - direction * direction.transpose();
                normal += projector;
                rhs += projector * world.translation;
                views.push((camera, pose, &obs.xy));
                preserved.push(
                    world
                        .transform_point(
                            &previous.world_to_camera.transform_point(&landmark.position),
                        )
                        .coords,
                );
            }
            if preserved.is_empty() {
                continue;
            }
            let fallback = Point3::from(
                preserved.iter().copied().sum::<Vector3<f64>>() / preserved.len() as f64,
            );
            let reprojection_cost = |point: &Point3<f64>| -> f64 {
                views
                    .iter()
                    .map(|(cam, pose, pixel)| {
                        let p = pose.world_to_camera.transform_point(point);
                        cam.project(&p)
                            .filter(|_| cam.sees_in_front(&p, pixel))
                            .map_or(16.0, |uv| (uv - *pixel).norm_squared().min(16.0))
                    })
                    .sum()
            };
            let eigenvalues = normal.symmetric_eigen().eigenvalues;
            let solution = (views.len() >= 2 && eigenvalues.min() > 1e-4 * eigenvalues.max())
                .then(|| normal.lu().solve(&rhs))
                .flatten()
                .map(Point3::from)
                .filter(|point| {
                    (point - fallback).norm() <= 0.25
                        && views
                            .iter()
                            .filter(|(cam, pose, pixel)| {
                                let p = pose.world_to_camera.transform_point(point);
                                cam.sees_in_front(&p, pixel)
                                    && cam
                                        .project(&p)
                                        .is_some_and(|uv| (uv - **pixel).norm() <= 4.0)
                            })
                            .count()
                            >= 2
                        && reprojection_cost(point) <= reprojection_cost(&fallback)
                });
            rebuilt += usize::from(solution.is_some());
            landmark.position = solution.unwrap_or(fallback);
        }
    }
    report["applied"] = json!(true);
    report["retriangulated"] = json!(rebuilt);
    report["max_keyframe_correction_m"] = json!(maximum);
    println!(
        "keyframe refinement: {selected} loops, {rebuilt} retriangulated points, {:.6} -> {:.6}",
        result.initial_cost, applied_cost
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_path_rejects_an_inconsistent_triangle_measurement() {
        let edge = |base, other, x| SessionConstraint {
            base,
            other,
            base_from_other: SE3::new(UnitQuaternion::identity(), Vector3::new(x, 0.0, 0.0)),
        };
        let mut edges = vec![edge(0, 1, 1.0), edge(1, 2, 1.0), edge(2, 0, -2.0)];
        assert!(cycle_consistent(3, &edges, 2));
        edges[2] = edge(2, 0, -3.0);
        assert!(!cycle_consistent(3, &edges, 2));
        assert!(cycle_consistent(3, &edges[..2], 1));
    }
}
