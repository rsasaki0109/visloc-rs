//! Synthetic rig-scene builder shared by this module's unit tests (C2 task
//! item 7: "triangulator on a synthetic scene; BA on a synthetic rig scene
//! converges to ground truth; mapper end-to-end on a small synthetic rig
//! sequence"). `#[cfg(test)]`-only (see `mod.rs`).

use std::collections::BTreeMap;

use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};

use visloc_core::geometry::SE3;
use visloc_vision::two_view::{ConfigurationType, CorrespondenceGraph};

use super::database_cache::DatabaseCache;
use super::reconstruction::{Camera, Image, Point2D};
use super::types::{DataT, Frame, FrameT, ImageT, Rig, SensorT};

pub(crate) struct SyntheticScene {
    pub db: DatabaseCache,
    pub ground_truth_rig_from_world: BTreeMap<FrameT, SE3>,
    pub ground_truth_points: Vec<Point3<f64>>,
    #[allow(dead_code)]
    pub cam2_from_rig: SE3,
    /// `(image_id_cam1, image_id_cam2)` per frame, in frame order.
    pub images_per_frame: Vec<(ImageT, ImageT)>,
}

/// Builds a `num_frames`-frame, 2-camera-rig synthetic sequence: a small
/// stereo rig (identity ref camera 1, `cam2_from_rig` a ~6cm/~1deg
/// baseline) translating+yawing along a path in front of a static point
/// cloud, all points visible from every frame (by construction, so the test
/// scene's registration success is not confounded by visibility gaps).
/// Every image pair within `window` frames of each other (including the
/// same-frame stereo pair) is ingested into the correspondence graph with
/// full point-index correspondences.
pub(crate) fn build_synthetic_rig_scene(num_frames: usize, window: usize) -> SyntheticScene {
    let mut rigs = BTreeMap::new();
    let mut rig = Rig::new();
    rig.set_rig_id(0);
    rig.add_ref_sensor(SensorT::camera(1));
    // A wider synthetic baseline than the real OpenLORIS ~6cm rig (and
    // closer points below) than a real short-baseline stereo pair would
    // use, chosen deliberately so every same-frame stereo pair alone
    // clears COLMAP's `min_angle`/`filter_min_tri_angle` (1.5deg) gates
    // without depending on cross-frame motion — keeps the small (2-frame)
    // triangulator unit test independent of the sequence's motion profile.
    let cam2_from_rig = SE3::new(
        UnitQuaternion::from_euler_angles(0.0, 0.02, 0.0),
        Vector3::new(-0.15, 0.0, 0.0),
    );
    rig.add_sensor(SensorT::camera(2), Some(cam2_from_rig.clone()));
    rigs.insert(0, rig);

    let mut cameras = BTreeMap::new();
    cameras.insert(1, Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0));
    cameras.insert(2, Camera::pinhole(2, 640, 480, 500.0, 500.0, 320.0, 240.0));

    let mut ground_truth_points = Vec::new();
    for gx in -2..=2i32 {
        for gy in -2..=2i32 {
            for gz in 0..2i32 {
                ground_truth_points.push(Point3::new(
                    gx as f64 * 0.35,
                    gy as f64 * 0.35,
                    2.4 + gz as f64 * 0.6,
                ));
            }
        }
    }

    let mut frames = BTreeMap::new();
    let mut images = BTreeMap::new();
    let mut ground_truth_rig_from_world = BTreeMap::new();
    let mut images_per_frame = Vec::with_capacity(num_frames);
    let mut next_image_id: ImageT = 0;

    for k in 0..num_frames {
        let frame_id = k as FrameT;
        let translation = Vector3::new(0.05 * k as f64, 0.01 * (k as f64).sin(), 0.0);
        let yaw = 0.004 * k as f64;
        let rig_from_world = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, yaw, 0.0),
            translation,
        );
        ground_truth_rig_from_world.insert(frame_id, rig_from_world.clone());

        let img1_id = next_image_id;
        next_image_id += 1;
        let img2_id = next_image_id;
        next_image_id += 1;

        let mut frame = Frame::new(frame_id, 0);
        frame.add_data_id(DataT::camera(1, img1_id));
        frame.add_data_id(DataT::camera(2, img2_id));
        frames.insert(frame_id, frame);

        let cam1_from_world = rig_from_world.clone();
        let cam2_from_world = cam2_from_rig.compose(&rig_from_world);
        let camera1 = cameras[&1].clone();
        let camera2 = cameras[&2].clone();

        let mut points1 = Vec::with_capacity(ground_truth_points.len());
        let mut points2 = Vec::with_capacity(ground_truth_points.len());
        for p in &ground_truth_points {
            let p1c = cam1_from_world.transform_point(p);
            let p2c = cam2_from_world.transform_point(p);
            let uv1 = camera1
                .project(&p1c)
                .expect("synthetic point must be visible in camera 1");
            let uv2 = camera2
                .project(&p2c)
                .expect("synthetic point must be visible in camera 2");
            points1.push(uv1);
            points2.push(uv2);
        }

        let mut image1 = Image::new(img1_id, 1, frame_id, format!("f{k:04}_c1.png"));
        image1.points2d = points1.into_iter().map(Point2D::new).collect();
        images.insert(img1_id, image1);

        let mut image2 = Image::new(img2_id, 2, frame_id, format!("f{k:04}_c2.png"));
        image2.points2d = points2.into_iter().map(Point2D::new).collect();
        images.insert(img2_id, image2);

        images_per_frame.push((img1_id, img2_id));
    }

    let mut graph = CorrespondenceGraph::new();
    for iid in 0..next_image_id {
        graph.add_image(iid as usize, ground_truth_points.len());
    }
    let full_matches: Vec<(usize, usize)> =
        (0..ground_truth_points.len()).map(|j| (j, j)).collect();
    for k in 0..num_frames {
        let (i1, i2) = images_per_frame[k];
        for dk in 0..=window {
            let k2 = k + dk;
            if k2 >= num_frames {
                continue;
            }
            let (j1, j2) = images_per_frame[k2];
            let pairs: Vec<(ImageT, ImageT)> = if dk == 0 {
                vec![(i1, i2)]
            } else {
                vec![(i1, j1), (i1, j2), (i2, j1), (i2, j2)]
            };
            for (a, b) in pairs {
                graph
                    .add_two_view_geometry(
                        a as usize,
                        b as usize,
                        &full_matches,
                        ConfigurationType::Calibrated,
                    )
                    .unwrap();
            }
        }
    }
    graph.finalize();

    let db = DatabaseCache::from_parts(rigs, cameras, frames, images, graph);

    SyntheticScene {
        db,
        ground_truth_rig_from_world,
        ground_truth_points,
        cam2_from_rig,
        images_per_frame,
    }
}

/// Squared-distance point cloud fit under the assumption both sets are
/// already in the same (world) frame (no further alignment needed since the
/// tests seed exact ground-truth poses and only check reconstructed 3D
/// point / pose error).
#[allow(dead_code)]
pub(crate) fn point2_dist(a: Point2<f64>, b: Point2<f64>) -> f64 {
    (a - b).norm()
}

/// C2.5 profiling harness: a `bundle::BundleAdjustment` (not a full
/// `Reconstruction`/`DatabaseCache` — this is deliberately built directly at
/// the BA-problem level, bypassing the triangulator/mapper, so a 2-3k-point,
/// 100k+-observation problem can be generated in milliseconds) sized to
/// resemble the real tier-1000 global-BA problem that exposed
/// `rig_ba_solver`'s O(track_len²)-per-point allocation/`BTreeMap` blowup
/// (`frames=333 obs=179623 landmarks=2541`, i.e. ~71 observations/point —
/// long tracks from corridor revisits, not the short 2-8-observation tracks
/// typical of a single pass). A 2-camera rig (matching the OpenLORIS-style
/// baseline) moves in a straight line; `num_points` landmarks are centered
/// on evenly-spread frames and visible for `track_span` consecutive frames
/// each (both cameras), so `avg observations/point ≈ 2 · (track_span + 1)`
/// and every point's *frame* track length is `track_span + 1` — reproducing
/// the O(k²) elimination-pair blowup a corridor-revisit scene causes,
/// without needing real data.
pub(crate) fn build_large_synthetic_ba_problem(
    num_frames: usize,
    num_points: usize,
    track_span: usize,
) -> crate::bundle::BundleAdjustment {
    use crate::bundle::{BaRigObservation, BundleAdjustment};
    use visloc_core::geometry::Pose;
    use visloc_core::types::Camera as CoreCamera;

    let camera1 = CoreCamera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let camera2 = CoreCamera::pinhole(2, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let cam2_from_rig = SE3::new(
        UnitQuaternion::from_euler_angles(0.0, 0.02, 0.0),
        Vector3::new(-0.15, 0.0, 0.0),
    );

    let mut ba = BundleAdjustment::new(camera1.clone());
    let mut gt_poses: Vec<SE3> = Vec::with_capacity(num_frames);
    for k in 0..num_frames {
        let translation = Vector3::new(0.05 * k as f64, 0.01 * (k as f64 * 0.3).sin(), 0.0);
        let yaw = 0.002 * k as f64;
        let pose = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, yaw, 0.0),
            translation,
        );
        gt_poses.push(pose.clone());
        ba.add_pose(
            k as u64,
            Pose {
                world_to_camera: pose,
            },
        );
    }
    ba.fix_pose(0);
    ba.fix_pose((num_frames - 1) as u64);

    let half_span = track_span / 2;
    for j in 0..num_points {
        let center = (j * num_frames) / num_points.max(1);
        let center_pose = &gt_poses[center];
        let lateral = ((j % 7) as f64 - 3.0) * 0.3;
        let vertical = ((j % 5) as f64 - 2.0) * 0.2;
        let depth = 2.5 + (j % 4) as f64 * 0.5;
        let point_cam = Point3::new(lateral, vertical, depth);
        let point_world = center_pose.inverse().transform_point(&point_cam);
        let landmark_id = j as u64;
        ba.add_landmark(landmark_id, point_world);

        let lo = center.saturating_sub(half_span);
        let hi = (center + half_span).min(num_frames.saturating_sub(1));
        for i in lo..=hi {
            let pose = &gt_poses[i];
            for (camera, sensor_from_rig) in [
                (&camera1, SE3::identity()),
                (&camera2, cam2_from_rig.clone()),
            ] {
                let p_rig = pose.transform_point(&point_world);
                let p_sensor = sensor_from_rig.transform_point(&p_rig);
                if p_sensor.z <= 0.2 {
                    continue;
                }
                let Some(xy) = camera.project(&p_sensor) else {
                    continue;
                };
                if xy.x < 0.0
                    || xy.x >= camera.width as f64
                    || xy.y < 0.0
                    || xy.y >= camera.height as f64
                {
                    continue;
                }
                ba.add_rig_observation(BaRigObservation {
                    keyframe_id: i as u64,
                    landmark_id,
                    xy,
                    camera: camera.clone(),
                    sensor_from_rig,
                });
            }
        }
    }
    ba
}
