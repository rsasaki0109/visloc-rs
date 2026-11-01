//! Faithful port of `sfm/incremental_mapper_impl.{h,cc}`'s free functions:
//! initial-pair search (`FindFirstInitialImage`/`FindSecondInitialImage`/
//! `FindInitialImagePair`/`EstimateInitialTwoViewGeometry` including the
//! rig-aware `EstimateInitialGeneralizedTwoViewGeometry`) and
//! `FindNextImages`/`FindLocalBundle`.
//!
//! Ported from (commit `64805cb870b574a569dccc34918d95a2db2b2fee`, pinned by
//! `docs/colmap_rig_mapper_port_plan.md` §0):
//! `src/colmap/sfm/incremental_mapper_impl.cc` — `FindFirstInitialImage`
//! (`.cc:104-145`), `FindSecondInitialImage` (`.cc:147-187`),
//! `FindInitialImagePair` (`.cc:189-292`, single-threaded equivalent —
//! see deviation below), `FindNextImages` (`.cc:294-364`), `FindLocalBundle`
//! (`.cc:366-537`), `EstimateInitialGeneralizedTwoViewGeometry`
//! (`.cc:541-650`), `EstimateInitialTwoViewGeometry` (`.cc:654-747`).
//!
//! ## Deviations
//!
//! - **`find_initial_image_pair` is single-threaded and exhaustive**, not
//!   COLMAP's thread-pool-parallel early-exit search (`.cc:227-291`).
//!   Functionally equivalent (COLMAP's own comment: "Iterate through the
//!   already computed results and return the first successful result. This
//!   is deterministic..." — i.e. the parallelism is a pure speed
//!   optimization over a deterministic sequential search); this port keeps
//!   the sequential form since `num_threads` control is out of scope.
//! - **Tie-breaking uses image id ascending** as a tertiary sort key
//!   (`find_first_initial_image`/`find_second_initial_image`/
//!   `find_next_images`) where COLMAP's `std::sort` leaves ties in an
//!   implementation-defined (but input-deterministic) order; this port's
//!   choice is different from libstdc++'s introsort tie order but equally
//!   deterministic and reproducible, documented per §3.5's determinism
//!   requirement.
//! - **Ordinary two-view geometry gating reuses
//!   `visloc_vision::two_view::RelativePoseEstimator`** (essential-matrix
//!   RANSAC + cheirality decomposition) rather than porting
//!   `estimators/two_view_geometry.cc`'s multi-model
//!   (E/F/H/degenerate/panoramic) classifier — `docs/colmap_port_plan.md`'s
//!   own verdict table already rates visloc's essential-matrix estimator at
//!   parity for the calibrated case, and this port's inputs are already the
//!   COLMAP-*verified* graph (§4.1 of the port plan — matches are
//!   pre-filtered by COLMAP's own two-view verification), so the
//!   multi-model classification COLMAP performs on raw candidate pairs is
//!   largely redundant here.
//! - **The rig-aware generalized relative pose step
//!   (`estimate_initial_generalized_two_view_geometry`) reuses
//!   `crates/vision/src/pnp/gr6p.rs`'s `estimate_gr6p_ransac_with_config`**
//!   (a genuine 6-point minimal generalized-relative-pose polynomial
//!   solver, matching COLMAP's `GR6PEstimator` in kind, per §3.3's explicit
//!   reuse instruction) — this is the metric-scale mechanism for the
//!   initial pair, faithfully mirroring `impl.cc:541-650`'s pooling of
//!   *all* cross-frame image-pair correspondences (not just the picked
//!   pair's own) and the same final recomposition
//!   `orig_cam2_from_rig2 * rig2_from_rig1 * inverse(orig_cam1_from_rig1)`.
//! - **Not in COLMAP: central rigs skip that step.** When every sensor of
//!   both rigs sits at its rig's origin (all `sensor_from_rig` translations
//!   zero, e.g. the cube faces of one panorama), every ray passes through
//!   the rig centre: GR6P is degenerate there and the pair's ordinary
//!   `cam2_from_cam1` (up to scale, like any single-camera init) is
//!   already the answer.
//! - **Not in COLMAP's port: spherical cameras.** A pair with an
//!   equirectangular camera initializes on unit rays
//!   (`estimate_initial_spherical_two_view_geometry`), without the
//!   forward-motion gate.

use nalgebra::{Point2, Vector3};
use std::collections::BTreeMap;

use visloc_core::geometry::SE3;
use visloc_vision::pnp::{
    estimate_gr6p_ransac_with_config, GeneralizedCameraObservation,
    GeneralizedRelativeCorrespondence, GeneralizedRelativePoseRansacConfig,
};
use visloc_vision::two_view::{CorrespondenceGraph, RelativePoseEstimator, TwoViewCorrespondence};

use super::incremental_triangulator::{matches_between_images, triangulate_dlt};
use super::observation_manager::calculate_triangulation_angle;
use super::reconstruction::{Image, Reconstruction};
use super::types::{Frame, ImageT, Rig, SensorT};

/// Debug-only diagnosis of why an initial pair was rejected, enabled with
/// `VISLOC_DEBUG_INIT=1`. Used to attribute a `forced_init_pair` rejection to
/// the direct two-view match gate vs the generalized GR6P gate.
fn debug_init(message: &str) {
    if std::env::var_os("VISLOC_DEBUG_INIT").is_some() {
        eprintln!("{message}");
    }
}

/// The control-relevant subset of `IncrementalMapper::Options` needed by
/// this module's free functions (the rest lives on `mapper::Options`; kept
/// separate to avoid a circular `use`).
pub struct InitGateOptions {
    pub init_min_num_inliers: usize,
    pub init_max_error: f64,
    pub init_max_forward_motion: f64,
    pub init_min_tri_angle_deg: f64,
    pub init_max_reg_trials: usize,
    pub random_seed: u64,
}

/// Port of `FindFirstInitialImage` (`.cc:104-145`). `has_prior_focal_length`
/// is always `true` for this port's calibrated inputs (see module doc), so
/// the sort key reduces to `num_correspondences` descending, id ascending.
pub fn find_first_initial_image(
    options: &InitGateOptions,
    recon: &Reconstruction,
    graph: &CorrespondenceGraph,
    init_num_reg_trials: &BTreeMap<ImageT, usize>,
    num_registrations: &BTreeMap<ImageT, usize>,
) -> Vec<ImageT> {
    let mut infos: Vec<(ImageT, usize)> = Vec::new();
    for &image_id in recon.images().keys() {
        if graph.num_correspondences_for_image(image_id as usize) == 0 {
            continue;
        }
        if *init_num_reg_trials.get(&image_id).unwrap_or(&0) >= options.init_max_reg_trials {
            continue;
        }
        if *num_registrations.get(&image_id).unwrap_or(&0) > 0 {
            continue;
        }
        infos.push((
            image_id,
            graph.num_correspondences_for_image(image_id as usize),
        ));
    }
    infos.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    infos.into_iter().map(|(id, _)| id).collect()
}

/// Port of `FindSecondInitialImage` (`.cc:147-187`).
pub fn find_second_initial_image(
    options: &InitGateOptions,
    image_id1: ImageT,
    recon: &Reconstruction,
    graph: &CorrespondenceGraph,
    num_registrations: &BTreeMap<ImageT, usize>,
) -> Vec<ImageT> {
    let mut counts: BTreeMap<ImageT, usize> = BTreeMap::new();
    let num_points2d = recon.image(image_id1).num_points2d();
    for idx in 0..num_points2d {
        for corr in graph.find_correspondences(image_id1 as usize, idx) {
            let cid = corr.image_id as ImageT;
            if *num_registrations.get(&cid).unwrap_or(&0) == 0 {
                *counts.entry(cid).or_insert(0) += 1;
            }
        }
    }
    let mut infos: Vec<(ImageT, usize)> = counts
        .into_iter()
        .filter(|&(_, c)| c >= options.init_min_num_inliers)
        .collect();
    infos.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    infos.into_iter().map(|(id, _)| id).collect()
}

/// The sensor-frame bearing of pixel `xy`: `(x, y, 1)` on the normalized
/// image plane, or the unit ray of a camera that sees behind itself (a
/// fisheye past 180°), which may point behind it.
fn sensor_bearing(
    camera: &super::reconstruction::Camera,
    xy: &Point2<f64>,
) -> Option<Vector3<f64>> {
    if camera.sees_behind() {
        return camera.unit_ray_from_pixel(xy);
    }
    let n = camera.normalize_pixel(xy)?;
    Some(Vector3::new(n.x, n.y, 1.0))
}

/// Port of `EstimateInitialGeneralizedTwoViewGeometry` (`.cc:541-650`). See
/// module doc for the GR6P reuse.
#[allow(clippy::too_many_arguments)] // mirrors the COLMAP signature
fn estimate_initial_generalized_two_view_geometry(
    options: &InitGateOptions,
    recon: &Reconstruction,
    graph: &CorrespondenceGraph,
    image1: &Image,
    image2: &Image,
    frame1: &Frame,
    frame2: &Frame,
    rig1: &Rig,
    rig2: &Rig,
) -> Option<SE3> {
    let frame1_images: Vec<ImageT> = frame1.image_ids().collect();
    let frame2_images: Vec<ImageT> = frame2.image_ids().collect();

    let sensor_from_rig_of = |rig: &Rig, image_id: ImageT| -> SE3 {
        let sensor_id = SensorT::camera(recon.image(image_id).camera_id);
        if rig.is_ref_sensor(sensor_id) {
            SE3::identity()
        } else {
            rig.sensor_from_rig(sensor_id)
        }
    };

    let mut gr_corrs: Vec<GeneralizedRelativeCorrespondence> = Vec::new();
    for &ia in &frame1_images {
        let camera_a = recon.camera(recon.image(ia).camera_id);
        let sensor_from_rig_a = sensor_from_rig_of(rig1, ia);
        let origin_a = sensor_from_rig_a.inverse().translation;
        for &ib in &frame2_images {
            let camera_b = recon.camera(recon.image(ib).camera_id);
            let sensor_from_rig_b = sensor_from_rig_of(rig2, ib);
            let origin_b = sensor_from_rig_b.inverse().translation;
            for (idx_a, idx_b) in matches_between_images(graph, recon, ia, ib) {
                let xy_a = recon.image(ia).points2d[idx_a].xy;
                let xy_b = recon.image(ib).points2d[idx_b].xy;
                let (Some(bearing_sensor_a), Some(bearing_sensor_b)) = (
                    sensor_bearing(camera_a, &xy_a),
                    sensor_bearing(camera_b, &xy_b),
                ) else {
                    continue;
                };
                let bearing_rig_a = sensor_from_rig_a.rotation.inverse() * bearing_sensor_a;
                let bearing_rig_b = sensor_from_rig_b.rotation.inverse() * bearing_sensor_b;
                let (Ok(obs_a), Ok(obs_b)) = (
                    GeneralizedCameraObservation::new(origin_a, bearing_rig_a),
                    GeneralizedCameraObservation::new(origin_b, bearing_rig_b),
                ) else {
                    continue;
                };
                gr_corrs.push(GeneralizedRelativeCorrespondence {
                    rig1: obs_a,
                    rig2: obs_b,
                });
            }
        }
    }

    if gr_corrs.len() < 6 {
        debug_init(&format!(
            "generalized init rejected: gr_corrs={} < 6 (images {}/{})",
            gr_corrs.len(),
            image1.image_id,
            image2.image_id
        ));
        return None;
    }

    let avg_focal = recon
        .camera(image1.camera_id)
        .intrinsics()
        .map_or(500.0, |(fx, fy, _, _)| 0.5 * (fx + fy));
    let cfg = GeneralizedRelativePoseRansacConfig {
        seed: options.random_seed,
        inlier_angular_threshold: (options.init_max_error / avg_focal).max(1.0e-4),
        min_inliers: 6,
        ..GeneralizedRelativePoseRansacConfig::default()
    };
    let Some(report) = estimate_gr6p_ransac_with_config(&gr_corrs, &cfg)
        .ok()
        .flatten()
    else {
        debug_init(&format!(
            "generalized init rejected: GR6P failed (images {}/{}, gr_corrs={})",
            image1.image_id,
            image2.image_id,
            gr_corrs.len()
        ));
        return None;
    };
    if report.inliers.len() < options.init_min_num_inliers {
        debug_init(&format!(
            "generalized init rejected: GR6P inliers={} < init_min_num_inliers={} (images {}/{}, gr_corrs={})",
            report.inliers.len(),
            options.init_min_num_inliers,
            image1.image_id,
            image2.image_id,
            gr_corrs.len()
        ));
        return None;
    }

    let rig2_from_rig1 = SE3::new(report.pose.rotation, report.pose.translation);
    let orig_cam1_from_rig1 = sensor_from_rig_of(rig1, image1.image_id);
    let orig_cam2_from_rig2 = sensor_from_rig_of(rig2, image2.image_id);
    Some(
        orig_cam2_from_rig2
            .compose(&rig2_from_rig1)
            .compose(&orig_cam1_from_rig1.inverse()),
    )
}

/// Port of `EstimateInitialTwoViewGeometry` (`.cc:654-747`): ordinary
/// two-view gating, then the rig-aware re-solve above when either frame is
/// non-trivial. Returns `cam2_from_cam1`.
pub fn estimate_initial_two_view_geometry(
    options: &InitGateOptions,
    recon: &Reconstruction,
    graph: &CorrespondenceGraph,
    image_id1: ImageT,
    image_id2: ImageT,
) -> Option<SE3> {
    let image1 = recon.image(image_id1);
    let image2 = recon.image(image_id2);
    let camera1 = recon.camera(image1.camera_id).clone();
    let camera2 = recon.camera(image2.camera_id).clone();

    let matches = matches_between_images(graph, recon, image_id1, image_id2);
    if matches.len() < options.init_min_num_inliers {
        debug_init(&format!(
            "init rejected: direct_matches={} < init_min_num_inliers={} (images {}/{})",
            matches.len(),
            options.init_min_num_inliers,
            image_id1,
            image_id2
        ));
        return None;
    }
    // A panorama always starts on unit rays; a fisheye past 180° too in a
    // single-camera frame, while a rig of them takes the ordinary path: its
    // pair gate on the rays ahead of the image plane, then the generalized
    // re-solve on every ray.
    let single_sensor = |image: &Image| {
        recon
            .rig(recon.frame(image.frame_id).rig_id())
            .num_sensors()
            <= 1
    };
    if camera1.model.is_spherical()
        || camera2.model.is_spherical()
        || ((camera1.sees_behind() || camera2.sees_behind())
            && single_sensor(image1)
            && single_sensor(image2))
    {
        return estimate_initial_spherical_two_view_geometry(
            options, recon, &matches, image1, image2, &camera1, &camera2,
        );
    }
    let correspondences: Vec<TwoViewCorrespondence> = matches
        .iter()
        .map(|&(i1, i2)| TwoViewCorrespondence::new(image1.points2d[i1].xy, image2.points2d[i2].xy))
        .collect();

    let mut estimator = RelativePoseEstimator::default();
    estimator.ransac.config.seed = options.random_seed;
    let avg_focal = camera1
        .intrinsics()
        .map_or(500.0, |(fx, fy, _, _)| 0.5 * (fx + fy));
    estimator.ransac.config.sampson_threshold = options.init_max_error / avg_focal;

    let Some(recovered) = estimator.estimate_with_cameras(&correspondences, &camera1, &camera2)
    else {
        debug_init(&format!(
            "init rejected: ordinary two-view estimate failed (images {image_id1}/{image_id2})"
        ));
        return None;
    };
    if recovered.inliers.len() < options.init_min_num_inliers {
        debug_init(&format!(
            "init rejected: ordinary inliers={} < init_min_num_inliers={} (images {image_id1}/{image_id2})",
            recovered.inliers.len(),
            options.init_min_num_inliers
        ));
        return None;
    }
    if recovered.previous_to_current.translation.z.abs() >= options.init_max_forward_motion {
        debug_init(&format!(
            "init rejected: forward motion |tz|={} >= init_max_forward_motion={} (images {image_id1}/{image_id2})",
            recovered.previous_to_current.translation.z.abs(),
            options.init_max_forward_motion
        ));
        return None;
    }

    // Triangulation angle (median over inliers, unit-scale cam2_from_cam1).
    let cam2_from_cam1 = recovered.previous_to_current.clone();
    let mut angles: Vec<f64> = Vec::with_capacity(recovered.inliers.len());
    let cam1_center = nalgebra::Point3::origin();
    let cam2_center = nalgebra::Point3::from(cam2_from_cam1.inverse().translation);
    for &idx in &recovered.inliers {
        let corr = &correspondences[idx];
        let Some(n1) = camera1.normalize_pixel(&corr.previous_xy) else {
            continue;
        };
        let Some(n2) = camera2.normalize_pixel(&corr.current_xy) else {
            continue;
        };
        let identity = SE3::identity();
        let Some(xyz) = triangulate_dlt(&[
            (&identity, Point2::new(n1.x, n1.y), &unit_camera()),
            (&cam2_from_cam1, Point2::new(n2.x, n2.y), &unit_camera()),
        ]) else {
            continue;
        };
        angles.push(calculate_triangulation_angle(cam1_center, cam2_center, xyz));
    }
    if angles.is_empty() {
        debug_init(&format!(
            "init rejected: no triangulable inliers for angle (images {image_id1}/{image_id2})"
        ));
        return None;
    }
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_angle = angles[angles.len() / 2];
    if median_angle <= options.init_min_tri_angle_deg.to_radians() {
        debug_init(&format!(
            "init rejected: median tri angle {:.2}deg <= {:.2}deg (images {image_id1}/{image_id2})",
            median_angle.to_degrees(),
            options.init_min_tri_angle_deg
        ));
        return None;
    }

    let frame1 = recon.frame(image1.frame_id);
    let frame2 = recon.frame(image2.frame_id);
    let rig1 = recon.rig(frame1.rig_id());
    let rig2 = recon.rig(frame2.rig_id());

    if (rig1.num_sensors() > 1 || rig2.num_sensors() > 1)
        && !(rig_is_central(rig1) && rig_is_central(rig2))
    {
        estimate_initial_generalized_two_view_geometry(
            options, recon, graph, image1, image2, frame1, frame2, rig1, rig2,
        )
    } else {
        Some(cam2_from_cam1)
    }
}

/// Not in COLMAP's port: [`estimate_initial_two_view_geometry`] for a pair
/// with a camera that sees behind itself (an equirectangular panorama, or a
/// fisheye past 180°: `Camera::sees_behind`), on unit rays: the essential
/// matrix from [`visloc_vision::two_view::spherical`] (its angular
/// threshold `init_max_error` over the pixels per radian), no forward-
/// motion gate (a panorama has no forward), and the triangulation angle
/// gated per view direction (below). Single-camera frames only.
fn estimate_initial_spherical_two_view_geometry(
    options: &InitGateOptions,
    recon: &Reconstruction,
    matches: &[(usize, usize)],
    image1: &Image,
    image2: &Image,
    camera1: &super::reconstruction::Camera,
    camera2: &super::reconstruction::Camera,
) -> Option<SE3> {
    use visloc_vision::two_view::spherical::{
        estimate_spherical_relative_pose, SphericalRansacConfig,
    };
    let (image_id1, image_id2) = (image1.image_id, image2.image_id);
    let frame1 = recon.frame(image1.frame_id);
    let frame2 = recon.frame(image2.frame_id);
    if recon.rig(frame1.rig_id()).num_sensors() > 1 || recon.rig(frame2.rig_id()).num_sensors() > 1
    {
        debug_init(&format!(
            "init rejected: spherical camera in a multi-sensor rig (images {image_id1}/{image_id2})"
        ));
        return None;
    }
    let mut pairs = Vec::with_capacity(matches.len());
    let (mut rays1, mut rays2) = (Vec::with_capacity(matches.len()), Vec::new());
    for &(i1, i2) in matches {
        let (xy1, xy2) = (image1.points2d[i1].xy, image2.points2d[i2].xy);
        if let (Some(r1), Some(r2)) = (
            camera1.unit_ray_from_pixel(&xy1),
            camera2.unit_ray_from_pixel(&xy2),
        ) {
            pairs.push((xy1, xy2));
            rays1.push(r1);
            rays2.push(r2);
        }
    }
    let pixels_per_radian = 0.5
        * (camera1.pixels_per_radian().unwrap_or(1.0) + camera2.pixels_per_radian().unwrap_or(1.0));
    let config = SphericalRansacConfig {
        max_angle: options.init_max_error / pixels_per_radian,
        seed: options.random_seed,
        ..SphericalRansacConfig::default()
    };
    let Some(pose) = estimate_spherical_relative_pose(&rays1, &rays2, &config) else {
        debug_init(&format!(
            "init rejected: spherical two-view estimate failed (images {image_id1}/{image_id2})"
        ));
        return None;
    };
    if pose.inliers.len() < options.init_min_num_inliers {
        debug_init(&format!(
            "init rejected: spherical inliers={} < init_min_num_inliers={} (images {image_id1}/{image_id2})",
            pose.inliers.len(),
            options.init_min_num_inliers
        ));
        return None;
    }
    // Exact for any angle (`from_matrix` iterates and can stall near 180°).
    let rotation = nalgebra::Rotation3::from_matrix_unchecked(pose.rotation);
    let cam2_from_cam1 = SE3::new(
        nalgebra::UnitQuaternion::from_rotation_matrix(&rotation),
        pose.translation,
    );
    let identity = SE3::identity();
    let cam1_center = nalgebra::Point3::origin();
    let cam2_center = nalgebra::Point3::from(cam2_from_cam1.inverse().translation);
    // The triangulation angle of each inlier, with the cube face its ray
    // points through in camera 1 (the largest of its six axis components).
    // A panorama always shows the epipoles, where any motion has no
    // parallax, and usually the far distance: the median over the whole
    // sphere sits far below what any one view direction sees. The gate is
    // the largest median of a face with at least half `init_min_num_inliers`
    // inliers: a pinhole camera looking that way would pass COLMAP's.
    let mut by_face: [Vec<f64>; 6] = Default::default();
    for &k in &pose.inliers {
        let (xy1, xy2) = pairs[k];
        let Some(xyz) =
            triangulate_dlt(&[(&identity, xy1, camera1), (&cam2_from_cam1, xy2, camera2)])
        else {
            continue;
        };
        let ray = rays1[k];
        let axis = (0..3)
            .max_by(|&a, &b| ray[a].abs().total_cmp(&ray[b].abs()))
            .expect("three axes");
        let face = 2 * axis + usize::from(ray[axis] < 0.0);
        by_face[face].push(calculate_triangulation_angle(cam1_center, cam2_center, xyz));
    }
    let min_face_inliers = (options.init_min_num_inliers / 2).max(1);
    let best_median = by_face
        .iter_mut()
        .filter(|angles| angles.len() >= min_face_inliers)
        .map(|angles| {
            angles.sort_by(|a, b| a.total_cmp(b));
            angles[angles.len() / 2]
        })
        .fold(f64::NEG_INFINITY, f64::max);
    if best_median <= options.init_min_tri_angle_deg.to_radians() {
        debug_init(&format!(
            "init rejected: best face median tri angle {:.2}deg <= {:.2}deg (images {image_id1}/{image_id2})",
            best_median.to_degrees(),
            options.init_min_tri_angle_deg
        ));
        return None;
    }
    Some(cam2_from_cam1)
}

/// Every sensor of `rig` at the rig's origin: each non-reference sensor
/// calibrated with a zero `sensor_from_rig` translation.
fn rig_is_central(rig: &Rig) -> bool {
    rig.non_ref_sensors()
        .values()
        .all(|s| s.as_ref().is_some_and(|s| s.translation.norm() < 1.0e-9))
}

fn unit_camera() -> super::reconstruction::Camera {
    super::reconstruction::Camera::pinhole(0, 1, 1, 1.0, 1.0, 0.0, 0.0)
}

/// The registration round [`find_next_images`] picks candidates for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateRound {
    /// From 2D-3D correspondences.
    Structure,
    /// COLMAP 4's structure-less fallback: the same candidates, ranked by
    /// their 2D-2D correspondences to registered images.
    StructureLess,
    /// Not in COLMAP: the structure-less fallback's candidates are the
    /// images with at least `2 * abs_pose_min_num_inliers` 2D-2D
    /// correspondences to registered images, the structure-less
    /// registration's own minimum (`Options::
    /// structure_less_candidates_from_correspondences`).
    StructureLessByCorrespondences,
}

/// Port of `FindNextImages` (`.cc:294-364`), `MIN_UNCERTAINTY` selection
/// method only (control default, `mapper.h:169-170`) — visibility-pyramid
/// score via `ObservationManager::point3d_visibility_score`.
pub fn find_next_images(
    abs_pose_min_num_inliers: usize,
    max_reg_trials: usize,
    recon: &Reconstruction,
    obs: &super::observation_manager::ObservationManager,
    filtered_frames: &std::collections::BTreeSet<super::types::FrameT>,
    num_reg_trials: &BTreeMap<ImageT, usize>,
    round: CandidateRound,
) -> Vec<ImageT> {
    let mut primary: Vec<(ImageT, u64)> = Vec::new();
    let mut secondary: Vec<(ImageT, u64)> = Vec::new();

    for (&image_id, image) in recon.images() {
        if recon.is_image_registered(image_id) {
            continue;
        }
        let visible = if round == CandidateRound::StructureLessByCorrespondences {
            obs.num_visible_correspondences(image_id) / 2
        } else {
            obs.num_visible_points3d(image_id)
        };
        if visible < abs_pose_min_num_inliers {
            continue;
        }
        let trials = *num_reg_trials.get(&image_id).unwrap_or(&0);
        if trials >= max_reg_trials {
            continue;
        }
        // Structure-less registration ranks by 2D-2D correspondences to
        // registered images (`.cc:276-281`).
        let rank = if round == CandidateRound::Structure {
            obs.point3d_visibility_score(image_id)
        } else {
            obs.num_visible_correspondences(image_id) as u64
        };
        if filtered_frames.contains(&image.frame_id) || trials != 0 {
            secondary.push((image_id, rank));
        } else {
            primary.push((image_id, rank));
        }
    }
    primary.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    secondary.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    primary
        .into_iter()
        .chain(secondary)
        .map(|(id, _)| id)
        .collect()
}

/// Port of `FindLocalBundle` (`.cc:366-537`), simplified to a single
/// relaxation tier instead of COLMAP's 8-step schedule
/// (`selection_thresholds`, `.cc:438-447`): rank images by shared-point
/// count, keep those clearing `ba_local_min_tri_angle` at the 75th
/// percentile (COLMAP's own `kTriangulationAnglePercentile`) up to
/// `ba_local_num_images - 1`, then fill any remainder with the next most-
/// overlapping images regardless of angle (mirrors COLMAP's final fallback,
/// `.cc:518-534`, without replaying every intermediate relaxation step —
/// documented deviation, does not change the *set* of images chosen in the
/// common case where the first tier already finds enough).
pub fn find_local_bundle(
    ba_local_num_images: usize,
    ba_local_min_tri_angle_deg: f64,
    image_id: ImageT,
    recon: &Reconstruction,
) -> Vec<ImageT> {
    let image = recon.image(image_id);
    // The image's 3D points, in its point2D order, and for every other
    // image its shared-point count and the positions (in that order) of the
    // points it also observes. Images are found through a table indexed by
    // id (ids are small and dense): a point's track is walked for every one
    // of the image's points, which on a city-scale model is tens of
    // thousands of lookups per call.
    let mut xyzs = Vec::new();
    let mut slot_of: Vec<u32> = Vec::new();
    let mut overlapping: Vec<(ImageT, usize, Vec<usize>)> = Vec::new();
    for p in &image.points2d {
        if let Some(pid) = p.point3d_id {
            let point3d = recon.point3d(pid);
            let k = xyzs.len();
            xyzs.push(point3d.xyz);
            for el in &point3d.track {
                if el.image_id == image_id {
                    continue;
                }
                let index = el.image_id as usize;
                if index >= slot_of.len() {
                    slot_of.resize(index + 1, u32::MAX);
                }
                if slot_of[index] == u32::MAX {
                    slot_of[index] = overlapping.len() as u32;
                    overlapping.push((el.image_id, 0, Vec::new()));
                }
                let (_, count, points) = &mut overlapping[slot_of[index] as usize];
                *count += 1;
                if points.last() != Some(&k) {
                    points.push(k);
                }
            }
        }
    }
    overlapping.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let num_eff = ba_local_num_images.saturating_sub(1).min(overlapping.len());
    if num_eff == 0 {
        return Vec::new();
    }

    let cam_center = |id: ImageT| super::observation_manager::image_projection_center(recon, id);
    let this_center = cam_center(image_id);
    let min_angle_rad = ba_local_min_tri_angle_deg.to_radians();

    let mut chosen: Vec<ImageT> = Vec::new();
    let mut used = vec![false; overlapping.len()];
    for (i, (other_id, _, points)) in overlapping.iter().enumerate() {
        if chosen.len() >= num_eff {
            break;
        }
        if points.is_empty() {
            continue;
        }
        let other_center = cam_center(*other_id);
        let mut angles: Vec<f64> = points
            .iter()
            .map(|&k| calculate_triangulation_angle(this_center, other_center, xyzs[k]))
            .collect();
        // The 75th percentile is the element a full sort would put there.
        let at = (angles.len() * 3 / 4).min(angles.len() - 1);
        let (_, &mut p75, _) = angles.select_nth_unstable_by(at, |a, b| a.partial_cmp(b).unwrap());
        if p75 >= min_angle_rad {
            chosen.push(*other_id);
            used[i] = true;
        }
    }
    if chosen.len() < num_eff {
        for (i, (other_id, _, _)) in overlapping.iter().enumerate() {
            if chosen.len() >= num_eff {
                break;
            }
            if !used[i] {
                used[i] = true;
                chosen.push(*other_id);
            }
        }
    }
    chosen
}
