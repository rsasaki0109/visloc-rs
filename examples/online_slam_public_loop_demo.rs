//! Loop-closure pose-graph demo that ingests a COLMAP-text-format
//! reconstruction from disk instead of building the map in code, exercising the
//! same I/O path that real public-data reconstructions (e.g., COLMAP South
//! Building or KITTI-derived sparse models) would take.
//!
//! Without flags the example synthesizes a 12-keyframe orbit fixture, writes
//! it as COLMAP text, reads it back, and runs the full tracking + verifier +
//! pose-graph SE(3) Gauss-Newton stack on the loaded data. With
//! `--colmap-path <dir>` it instead loads a user-supplied COLMAP sparse
//! reconstruction (`cameras.txt`, `images.txt`, `points3D.txt`); with
//! `--descriptors-path <file>` callers can pin landmark descriptors, otherwise
//! synthetic per-landmark descriptors are generated so the demo stays runnable
//! on any registered reconstruction.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::{Pose, SE3};
use visloc_rs::core::types::{
    Camera, Frame, Keyframe, Landmark, LandmarkDescriptorStore, Observation, VisualMap,
};
use visloc_rs::io::colmap::{read_colmap_text_model, write_colmap_text_model};
use visloc_rs::io::descriptors::read_landmark_descriptors_txt;
use visloc_rs::{
    loop_closure_constraints_from_candidates, relative_world_to_camera,
    verify_loop_closure_candidates, EssentialMatrixLoopClosureVerifier, LocalMappingPipeline,
    LocalizationPipeline, LoopClosureConfig, LoopClosureConstraint, LoopClosureVerifierConfig,
    OnlineSlamConfig, OnlineSlamPipeline, OnlineSlamResult, PoseGraph, PoseGraphSe3Config, Tracker,
    TrackingConfig,
};

const KEYFRAME_COUNT: u64 = 12;
const ORBIT_RADIUS: f64 = 4.0;
const LANDMARK_COUNT: u64 = 60;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let mut tmp_dir: Option<PathBuf> = None;

    let colmap_dir = if let Some(path) = args.colmap_path.clone() {
        path
    } else {
        let dir =
            std::env::temp_dir().join(format!("visloc_public_loop_demo_{}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        fs::create_dir_all(&dir)?;
        let fixture_map = build_synthetic_loop_fixture();
        write_colmap_text_model(&fixture_map, &dir)?;
        write_synthetic_descriptors(&fixture_map, &dir.join("landmark_descriptors.txt"))?;
        tmp_dir = Some(dir.clone());
        dir
    };
    println!("colmap_dir={}", colmap_dir.display());

    let mut map = read_colmap_text_model(&colmap_dir)?;
    let descriptor_path = args
        .descriptors_path
        .clone()
        .unwrap_or_else(|| colmap_dir.join("landmark_descriptors.txt"));
    let descriptors = if descriptor_path.exists() {
        read_landmark_descriptors_txt(&descriptor_path)?
    } else {
        synthesize_descriptors(&map)
    };
    for (id, descriptor) in descriptors.iter() {
        if let Some(landmark) = map.landmarks.get_mut(&id) {
            landmark.descriptor = Some(descriptor.to_vec());
        }
    }
    println!(
        "loaded cameras={} landmarks={} keyframes={}",
        map.cameras.len(),
        map.landmarks.len(),
        map.keyframes.len(),
    );

    // Snapshot the loaded keyframes for ground-truth comparison; the pipeline
    // re-localizes each frame against the cleared keyframe table below.
    let mut keyframe_ids: Vec<u64> = map.keyframes.keys().copied().collect();
    keyframe_ids.sort();
    let truth_poses: Vec<(u64, Pose)> = keyframe_ids
        .iter()
        .map(|id| {
            let kf = &map.keyframes[id];
            (*id, kf.frame.pose.clone().expect("colmap pose required"))
        })
        .collect();
    let frames: Vec<Frame> = keyframe_ids
        .iter()
        .map(|id| frame_from_keyframe(&map.keyframes[id], &descriptors))
        .collect();
    let camera = map
        .cameras
        .values()
        .next()
        .expect("colmap fixture must have at least one camera")
        .clone();

    // The pipeline's tracker fills in keyframes on its own, so start with the
    // landmark-only view of the map. We keep landmarks (with merged
    // descriptors) so localization has something to match against.
    let mut initial_map = VisualMap::new();
    initial_map.cameras = map.cameras.clone();
    initial_map.landmarks = map.landmarks.clone();

    // Frame ids in the synthetic fixture jump by 10 so the loop closure between
    // the first and last keyframe (gap = 110) is the only pair big enough to
    // pass min_frame_id_gap. Real COLMAP reconstructions may use any ids, so
    // derive the gap dynamically.
    let min_frame_id_gap = if keyframe_ids.len() >= 2 {
        let first = keyframe_ids[0];
        let last = *keyframe_ids.last().unwrap();
        let total_gap = last.saturating_sub(first);
        let mut max_consecutive: u64 = 0;
        for window in keyframe_ids.windows(2) {
            max_consecutive = max_consecutive.max(window[1] - window[0]);
        }
        total_gap.saturating_sub(max_consecutive / 2).max(1)
    } else {
        1
    };
    println!("loop_closure_min_frame_id_gap={min_frame_id_gap}");

    let translation_scale =
        if let (Some((_, first)), Some((_, last))) = (truth_poses.first(), truth_poses.last()) {
            let from_to_last = relative_world_to_camera(first, last);
            from_to_last.translation.norm().max(1e-6)
        } else {
            1.0
        };

    let mut slam = OnlineSlamPipeline::new(
        initial_map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig {
                min_frame_id_gap,
                min_shared_landmarks: 10,
                min_shared_landmark_ratio_percent: 50,
                ..LoopClosureConfig::default()
            },
        },
    );
    let verifier = EssentialMatrixLoopClosureVerifier {
        config: LoopClosureVerifierConfig {
            min_inliers: 8,
            min_inlier_ratio: 0.5,
            max_mean_sampson_error: 5.0e-3,
            default_translation_scale: translation_scale,
        },
        ..Default::default()
    };

    let mut results: Vec<OnlineSlamResult> = Vec::new();
    let mut all_constraints: Vec<LoopClosureConstraint> = Vec::new();
    for frame in &frames {
        let mut result = slam.process_frame(frame, []);
        if !result.loop_closure_candidates.is_empty() {
            verify_loop_closure_candidates(
                &mut result.loop_closure_candidates,
                frame,
                &result.tracking,
                slam.map(),
                &camera,
                &verifier,
            );
        }
        let frame_constraints =
            loop_closure_constraints_from_candidates(&result.loop_closure_candidates);
        println!(
            "frame={} tracking={} keyframes={} loop_candidates={} loop_constraints={}",
            result.tracking.frame_id,
            result.tracking_succeeded(),
            result.map_keyframe_count,
            result.loop_closure_candidates.len(),
            frame_constraints.len(),
        );
        all_constraints.extend(frame_constraints);
        results.push(result);
    }
    println!("total_loop_constraints={}", all_constraints.len());

    let mut graph = PoseGraph::new();
    for (frame_id, result) in keyframe_ids.iter().zip(results.iter()) {
        if let Some(pose) = result.tracking.localization.pose.clone() {
            graph.add_pose(*frame_id, pose);
        }
    }
    for window in keyframe_ids.windows(2) {
        let (Some(from_pose), Some(to_pose)) = (
            graph.poses.get(&window[0]).cloned(),
            graph.poses.get(&window[1]).cloned(),
        ) else {
            continue;
        };
        graph.add_sequential_edge(
            window[0],
            window[1],
            relative_world_to_camera(&from_pose, &to_pose),
        );
    }
    for constraint in &all_constraints {
        graph.add_loop_closure_constraint(constraint);
    }
    graph.anchor(keyframe_ids[0]);
    println!(
        "pose_graph nodes={} sequential_edges={} loop_edges={} translation_cost={:.6} se3_cost={:.6}",
        graph.poses.len(),
        graph.edges.len() - all_constraints.len(),
        all_constraints.len(),
        graph.translation_cost(),
        graph.se3_cost(),
    );

    // Inject a combined translation + rotation drift on the last keyframe
    // (frame closest to the loop-closing pair) and let SE(3) Gauss-Newton pull
    // it back along the verified loop.
    let drift_translation = Vector3::new(0.05, 0.0, -0.04);
    let drift_rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.18);
    if let Some(&last_id) = keyframe_ids.last() {
        if let Some(pose) = graph.poses.get_mut(&last_id) {
            let drifted_center = pose.camera_center_world().coords + drift_translation;
            let new_rotation = drift_rotation * pose.world_to_camera.rotation;
            let new_translation = -(new_rotation.transform_vector(&drifted_center));
            pose.world_to_camera.rotation = new_rotation;
            pose.world_to_camera.translation = new_translation;
        }
    }
    println!(
        "se3_drift_applied keyframe={} se3_cost_before={:.6}",
        keyframe_ids.last().copied().unwrap_or_default(),
        graph.se3_cost(),
    );

    match graph.optimize_se3_iterative(&PoseGraphSe3Config::default()) {
        Ok(result) => {
            println!(
                "pose_graph_se3 anchor={} edges={} variables={} initial_cost={:.6} final_cost={:.6} iterations={} converged={}",
                result.anchor_id,
                result.edge_count,
                result.variable_count,
                result.initial_cost,
                result.final_cost,
                result.iterations.len(),
                result.converged,
            );
            for stats in &result.iterations {
                println!(
                    "  iter={} cost_before={:.6} cost_after={:.6} max_step={:.6}",
                    stats.iteration, stats.cost_before, stats.cost_after, stats.max_step_norm,
                );
            }
            for (frame_id, truth) in truth_poses.iter() {
                if let Some(pose) = graph.poses.get(frame_id) {
                    let truth_center = truth.camera_center_world();
                    let center = pose.camera_center_world();
                    let translation_err = (center - truth_center).norm();
                    let rotation_err = pose
                        .world_to_camera
                        .rotation
                        .rotation_to(&truth.world_to_camera.rotation)
                        .angle();
                    println!(
                        "  post_se3 keyframe={} center=[{:.3}, {:.3}, {:.3}] truth=[{:.3}, {:.3}, {:.3}] t_err={:.4} rot_err={:.4}",
                        frame_id,
                        center.x,
                        center.y,
                        center.z,
                        truth_center.x,
                        truth_center.y,
                        truth_center.z,
                        translation_err,
                        rotation_err,
                    );
                }
            }
        }
        Err(error) => println!("pose_graph_se3 error={error}"),
    }

    if let Some(out_dir) = args.out_dir {
        fs::create_dir_all(&out_dir)?;
        if tmp_dir.is_some() {
            // Copy the generated COLMAP fixture into the output dir so users
            // can reuse it as a reference reconstruction.
            for entry in fs::read_dir(&colmap_dir)? {
                let entry = entry?;
                fs::copy(entry.path(), out_dir.join(entry.file_name()))?;
            }
            println!("wrote synthetic colmap fixture to {}", out_dir.display());
        }
    }

    if let Some(dir) = tmp_dir {
        if !args.keep_temp {
            fs::remove_dir_all(&dir).ok();
        }
    }

    Ok(())
}

#[derive(Debug, Default)]
struct CliArgs {
    colmap_path: Option<PathBuf>,
    descriptors_path: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    keep_temp: bool,
}

fn parse_args() -> CliArgs {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let mut out = CliArgs::default();
    let i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--colmap-path" => {
                if i + 1 >= args.len() {
                    eprintln!("--colmap-path requires a directory");
                    std::process::exit(2);
                }
                out.colmap_path = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--descriptors-path" => {
                if i + 1 >= args.len() {
                    eprintln!("--descriptors-path requires a file");
                    std::process::exit(2);
                }
                out.descriptors_path = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--out-dir" => {
                if i + 1 >= args.len() {
                    eprintln!("--out-dir requires a directory");
                    std::process::exit(2);
                }
                out.out_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--keep-temp" => {
                out.keep_temp = true;
                args.remove(i);
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }
    out
}

fn build_synthetic_loop_fixture() -> VisualMap {
    let mut map = VisualMap::new();
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    map.cameras.insert(camera.id, camera.clone());

    let landmarks = synthetic_landmark_grid();
    for (index, position) in landmarks.iter().enumerate() {
        let id = (index as u64) + 1;
        let descriptor = synthetic_descriptor(id);
        let mut landmark = Landmark::new(id, *position);
        landmark.descriptor = Some(descriptor);
        map.landmarks.insert(id, landmark);
    }

    for keyframe_index in 0..KEYFRAME_COUNT {
        let frame_id = (keyframe_index + 1) * 10;
        let theta = 2.0 * std::f64::consts::PI * (keyframe_index as f64) / (KEYFRAME_COUNT as f64);
        let pose = orbit_pose(theta);
        let mut frame = Frame::new(frame_id, camera.id);
        frame.pose = Some(pose.clone());
        let mut observations: Vec<Observation> = Vec::new();
        for (id, landmark) in map.landmarks.iter() {
            let camera_point = pose.transform_world_point(&landmark.position);
            if let Some(uv) = camera.project(&camera_point) {
                if uv.x < 0.0
                    || uv.x >= camera.width as f64
                    || uv.y < 0.0
                    || uv.y >= camera.height as f64
                {
                    continue;
                }
                let keypoint_index = frame.keypoints.len();
                frame.keypoints.push(uv);
                frame.descriptors.push(synthetic_descriptor(*id));
                observations.push(Observation {
                    frame_id,
                    landmark_id: *id,
                    keypoint_index,
                    xy: uv,
                });
            }
        }
        for obs in &observations {
            if let Some(landmark) = map.landmarks.get_mut(&obs.landmark_id) {
                landmark.observations.push(obs.clone());
            }
        }
        map.keyframes.insert(
            frame_id,
            Keyframe {
                frame,
                observations,
            },
        );
    }
    map
}

fn synthetic_landmark_grid() -> Vec<Point3<f64>> {
    let mut points = Vec::with_capacity(LANDMARK_COUNT as usize);
    let count = LANDMARK_COUNT as usize;
    for i in 0..count {
        // Distribute deterministically on a low-discrepancy lattice inside the
        // unit cube around the origin.
        let t = i as f64;
        let x = ((t * 0.61803398875).fract() - 0.5) * 1.6;
        let y = ((t * 0.41421356237).fract() - 0.5) * 1.0;
        let z = ((t * 0.31622776601).fract() - 0.5) * 1.6;
        points.push(Point3::new(x, y, z));
    }
    points
}

fn synthetic_descriptor(landmark_id: u64) -> Vec<f32> {
    vec![landmark_id as f32, 1.0]
}

fn orbit_pose(theta: f64) -> Pose {
    // Camera orbits the origin in the XZ plane, looking inward. With theta
    // measured from +Z toward +X, the camera center sits at
    // (R sin θ, 0, R cos θ) and the world-to-camera rotation is a yaw of
    // (π − θ) about the world Y axis so the camera's +Z axis points back to
    // the origin.
    let camera_center = Vector3::new(ORBIT_RADIUS * theta.sin(), 0.0, ORBIT_RADIUS * theta.cos());
    let yaw = std::f64::consts::PI - theta;
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw);
    let translation = -(rotation.transform_vector(&camera_center));
    Pose::from_world_to_camera(rotation, translation)
}

fn write_synthetic_descriptors(map: &VisualMap, path: &Path) -> std::io::Result<()> {
    let mut content = String::from("# LANDMARK_ID D0 D1\n");
    let mut ids: Vec<u64> = map.landmarks.keys().copied().collect();
    ids.sort();
    for id in ids {
        let descriptor = map
            .landmarks
            .get(&id)
            .and_then(|landmark| landmark.descriptor.as_ref())
            .cloned()
            .unwrap_or_else(|| synthetic_descriptor(id));
        let parts: Vec<String> = descriptor.iter().map(|v| format!("{v}")).collect();
        content.push_str(&format!("{} {}\n", id, parts.join(" ")));
    }
    fs::write(path, content)
}

fn synthesize_descriptors(map: &VisualMap) -> LandmarkDescriptorStore {
    let mut store = LandmarkDescriptorStore::new();
    for (id, _) in map.landmarks.iter() {
        store.insert(*id, synthetic_descriptor(*id));
    }
    store
}

fn frame_from_keyframe(keyframe: &Keyframe, descriptors: &LandmarkDescriptorStore) -> Frame {
    let mut frame = Frame::new(keyframe.frame.id, keyframe.frame.camera_id);
    let keypoints = keyframe.frame.keypoints.clone();
    let mut keypoint_descriptors: Vec<Vec<f32>> = vec![Vec::new(); keypoints.len()];
    for obs in &keyframe.observations {
        if let Some(d) = descriptors.get(obs.landmark_id) {
            if obs.keypoint_index < keypoint_descriptors.len() {
                keypoint_descriptors[obs.keypoint_index] = d.to_vec();
            }
        }
    }
    frame.keypoints = keypoints;
    frame.descriptors = keypoint_descriptors;
    frame
}

// Small visual-debug helper kept in the example so curious readers can inspect
// the generated trajectory geometry without grepping the codebase.
#[allow(dead_code)]
fn pose_summary(pose: &SE3) -> String {
    let r: Matrix3<f64> = pose.rotation.to_rotation_matrix().into_inner();
    format!(
        "t=[{:.3},{:.3},{:.3}] R col0=[{:.3},{:.3},{:.3}]",
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
        r[(0, 0)],
        r[(1, 0)],
        r[(2, 0)],
    )
}
