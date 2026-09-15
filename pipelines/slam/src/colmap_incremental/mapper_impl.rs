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
                let (Some(na), Some(nb)) = (
                    camera_a.normalize_pixel(&xy_a),
                    camera_b.normalize_pixel(&xy_b),
                ) else {
                    continue;
                };
                let bearing_sensor_a = Vector3::new(na.x, na.y, 1.0);
                let bearing_sensor_b = Vector3::new(nb.x, nb.y, 1.0);
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
        return None;
    }

    let avg_focal = recon
        .camera(image1.camera_id)
        .intrinsics()
        .map(|(fx, fy, _, _)| 0.5 * (fx + fy))
        .unwrap_or(500.0);
    let cfg = GeneralizedRelativePoseRansacConfig {
        seed: options.random_seed,
        inlier_angular_threshold: (options.init_max_error / avg_focal).max(1.0e-4),
        min_inliers: 6,
        ..GeneralizedRelativePoseRansacConfig::default()
    };
    let report = estimate_gr6p_ransac_with_config(&gr_corrs, &cfg)
        .ok()
        .flatten()?;
    if report.inliers.len() < options.init_min_num_inliers {
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
        return None;
    }
    let correspondences: Vec<TwoViewCorrespondence> = matches
        .iter()
        .map(|&(i1, i2)| TwoViewCorrespondence::new(image1.points2d[i1].xy, image2.points2d[i2].xy))
        .collect();

    let mut estimator = RelativePoseEstimator::default();
    estimator.ransac.config.seed = options.random_seed;
    let avg_focal = camera1
        .intrinsics()
        .map(|(fx, fy, _, _)| 0.5 * (fx + fy))
        .unwrap_or(500.0);
    estimator.ransac.config.sampson_threshold = options.init_max_error / avg_focal;

    let recovered = estimator.estimate_with_cameras(&correspondences, &camera1, &camera2)?;
    if recovered.inliers.len() < options.init_min_num_inliers {
        return None;
    }
    if recovered.previous_to_current.translation.z.abs() >= options.init_max_forward_motion {
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
        return None;
    }
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_angle = angles[angles.len() / 2];
    if median_angle <= options.init_min_tri_angle_deg.to_radians() {
        return None;
    }

    let frame1 = recon.frame(image1.frame_id);
    let frame2 = recon.frame(image2.frame_id);
    let rig1 = recon.rig(frame1.rig_id());
    let rig2 = recon.rig(frame2.rig_id());

    if rig1.num_sensors() > 1 || rig2.num_sensors() > 1 {
        estimate_initial_generalized_two_view_geometry(
            options, recon, graph, image1, image2, frame1, frame2, rig1, rig2,
        )
    } else {
        Some(cam2_from_cam1)
    }
}

fn unit_camera() -> super::reconstruction::Camera {
    super::reconstruction::Camera::pinhole(0, 1, 1, 1.0, 1.0, 0.0, 0.0)
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
) -> Vec<ImageT> {
    let mut primary: Vec<(ImageT, u64)> = Vec::new();
    let mut secondary: Vec<(ImageT, u64)> = Vec::new();

    for (&image_id, image) in recon.images() {
        if recon.is_image_registered(image_id) {
            continue;
        }
        if obs.num_visible_points3d(image_id) < abs_pose_min_num_inliers {
            continue;
        }
        let trials = *num_reg_trials.get(&image_id).unwrap_or(&0);
        if trials >= max_reg_trials {
            continue;
        }
        let rank = obs.point3d_visibility_score(image_id);
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
    let mut shared: BTreeMap<ImageT, usize> = BTreeMap::new();
    let mut point3d_ids = Vec::new();
    for p in &image.points2d {
        if let Some(pid) = p.point3d_id {
            point3d_ids.push(pid);
            for el in &recon.point3d(pid).track {
                if el.image_id != image_id {
                    *shared.entry(el.image_id).or_insert(0) += 1;
                }
            }
        }
    }
    let mut overlapping: Vec<(ImageT, usize)> = shared.into_iter().collect();
    overlapping.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    let num_eff = ba_local_num_images.saturating_sub(1).min(overlapping.len());
    if num_eff == 0 {
        return Vec::new();
    }

    let cam_center = |id: ImageT| super::observation_manager::image_projection_center(recon, id);
    let this_center = cam_center(image_id);
    let min_angle_rad = ba_local_min_tri_angle_deg.to_radians();

    let mut chosen: Vec<ImageT> = Vec::new();
    let mut used: std::collections::BTreeSet<ImageT> = std::collections::BTreeSet::new();
    for &(other_id, _count) in &overlapping {
        if chosen.len() >= num_eff {
            break;
        }
        let other_center = cam_center(other_id);
        let mut angles: Vec<f64> = Vec::new();
        for &pid in &point3d_ids {
            if recon
                .point3d(pid)
                .track
                .iter()
                .any(|el| el.image_id == other_id)
            {
                angles.push(calculate_triangulation_angle(
                    this_center,
                    other_center,
                    recon.point3d(pid).xyz,
                ));
            }
        }
        if angles.is_empty() {
            continue;
        }
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p75 = angles[(angles.len() * 3 / 4).min(angles.len() - 1)];
        if p75 >= min_angle_rad {
            chosen.push(other_id);
            used.insert(other_id);
        }
    }
    if chosen.len() < num_eff {
        for &(other_id, _) in &overlapping {
            if chosen.len() >= num_eff {
                break;
            }
            if used.insert(other_id) {
                chosen.push(other_id);
            }
        }
    }
    chosen
}
