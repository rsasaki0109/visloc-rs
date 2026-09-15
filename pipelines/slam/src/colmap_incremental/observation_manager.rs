//! Faithful (C2-scoped) port of `sfm/observation_manager.{h,cc}`'s
//! [`ObservationManager`] and `scene/visibility_pyramid.{h,cc}`'s
//! [`VisibilityPyramid`].
//!
//! Ported from (commit `64805cb870b574a569dccc34918d95a2db2b2fee`, pinned by
//! `docs/colmap_rig_mapper_port_plan.md` §0):
//! - `src/colmap/scene/visibility_pyramid.h` / `.cc` — [`VisibilityPyramid`],
//!   ported near-verbatim (`SetPoint`/`ResetPoint`/`CellForPoint`/`Score`).
//! - `src/colmap/sfm/observation_manager.h` / `.cc` — [`ObservationManager`]:
//!   `NumObservations`/`NumCorrespondences`/`NumVisibleCorrespondences`/
//!   `NumVisiblePoints3D`/`Point3DVisibilityScore` (`.h:158-176`),
//!   `IncrementCorrespondenceHasPoint3D`/`DecrementCorrespondenceHasPoint3D`
//!   (`.cc:181-214`), `SetObservationAsTriangulated`/`ResetTriObservations`
//!   (`.cc:216-276`), `AddPoint3D`/`AddObservation`/`DeletePoint3D`/
//!   `DeleteObservation`/`MergePoints3D` (`.cc:278-351`),
//!   `FilterPoints3D`/`FilterPoints3DInImages`/`FilterAllPoints3D`/
//!   `FilterObservationsWithNegativeDepth`/
//!   `FilterPoints3DWithSmallTriangulationAngle`/
//!   `FilterPoints3DWithLargeReprojectionError` (`.cc:353-585`),
//!   `RegisterFrame`/`DeRegisterFrame`/`FindFramesToFilter`
//!   (`.cc:587-655`).
//!
//! ## Deviation: no cached `Reconstruction&`/`CorrespondenceGraph` refs
//!
//! COLMAP's `ObservationManager` stores a `Reconstruction&` and a
//! `shared_ptr<const CorrespondenceGraph>` as member fields (`.h:227-228`)
//! so its methods can be called with just e.g. an `image_id`. Rust's borrow
//! checker cannot express "this struct's methods take `&mut` of another
//! struct the caller also holds" without `Rc<RefCell<_>>` plumbing that
//! would ripple through every caller (`mapper.rs` needs simultaneous access
//! to `Reconstruction`, `DatabaseCache`, `ObservationManager`, and
//! `IncrementalTriangulator`). Continuing the precedent set by
//! `types.rs`'s `Frame` ("no `rig_ptr_`" — explicit `&Rig` parameters
//! instead of a cached pointer), every [`ObservationManager`] method that
//! needs the reconstruction or correspondence graph takes them as explicit
//! parameters instead. This is semantically identical (same objects, same
//! lookups), just passed explicitly rather than cached.

use std::collections::BTreeMap;

use nalgebra::{Point2, Point3};
use visloc_core::geometry::SE3;
use visloc_vision::two_view::CorrespondenceGraph;

use super::reconstruction::{Camera, Reconstruction, TrackElement};
use super::types::{FrameT, ImageT, Point2DT, Point3DT, SensorT};

// ---------------------------------------------------------------------
// VisibilityPyramid (`scene/visibility_pyramid.h/.cc`).
// ---------------------------------------------------------------------

/// Port of `VisibilityPyramid` (`scene/visibility_pyramid.h:51-81`,
/// `.cc:37-107`): captures the distribution of 2D points (e.g. visible 3D
/// points) in an image via a multi-resolution occupancy pyramid; a higher
/// [`Self::score`] means a more uniform spatial distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct VisibilityPyramid {
    width: usize,
    height: usize,
    score: u64,
    max_score: u64,
    /// One occupancy grid per level; level `i` has `dim = 2^(i+1)` cells per
    /// side (`.cc:39-49`).
    pyramid: Vec<Vec<u32>>,
    dims: Vec<usize>,
}

impl VisibilityPyramid {
    /// Port of `VisibilityPyramid()` (default, zero levels).
    pub fn empty() -> Self {
        Self::new(0, 0, 0)
    }

    /// Port of `VisibilityPyramid(num_levels, width, height)`
    /// (`.cc:39-50`).
    pub fn new(num_levels: usize, width: usize, height: usize) -> Self {
        let mut pyramid = Vec::with_capacity(num_levels);
        let mut dims = Vec::with_capacity(num_levels);
        let mut max_score: u64 = 0;
        for level in 0..num_levels {
            let dim = 1usize << (level + 1);
            pyramid.push(vec![0u32; dim * dim]);
            dims.push(dim);
            max_score += (dim as u64).pow(4);
        }
        Self {
            width,
            height,
            score: 0,
            max_score,
            pyramid,
            dims,
        }
    }

    pub fn score(&self) -> u64 {
        self.score
    }

    /// Port of `CellForPoint` (`.cc:96-105`).
    fn cell_for_point(&self, x: f64, y: f64) -> (usize, usize) {
        let max_dim = 1usize << self.pyramid.len();
        let cx =
            ((max_dim as f64) * x / (self.width as f64)).clamp(0.0, (max_dim - 1) as f64) as usize;
        let cy =
            ((max_dim as f64) * y / (self.height as f64)).clamp(0.0, (max_dim - 1) as f64) as usize;
        (cx, cy)
    }

    /// Port of `SetPoint` (`.cc:52-72`).
    pub fn set_point(&mut self, x: f64, y: f64) {
        assert!(!self.pyramid.is_empty());
        let (mut cx, mut cy) = self.cell_for_point(x, y);
        for level in (0..self.pyramid.len()).rev() {
            let dim = self.dims[level];
            let idx = cy * dim + cx;
            self.pyramid[level][idx] += 1;
            if self.pyramid[level][idx] == 1 {
                self.score += (dim * dim) as u64;
            }
            cx >>= 1;
            cy >>= 1;
        }
    }

    /// Port of `ResetPoint` (`.cc:74-94`).
    pub fn reset_point(&mut self, x: f64, y: f64) {
        assert!(!self.pyramid.is_empty());
        let (mut cx, mut cy) = self.cell_for_point(x, y);
        for level in (0..self.pyramid.len()).rev() {
            let dim = self.dims[level];
            let idx = cy * dim + cx;
            assert!(self.pyramid[level][idx] > 0);
            self.pyramid[level][idx] -= 1;
            if self.pyramid[level][idx] == 0 {
                self.score -= (dim * dim) as u64;
            }
            cx >>= 1;
            cy >>= 1;
        }
    }
}

/// `kNumPoint3DVisibilityPyramidLevels` (`observation_manager.h:53`).
pub const NUM_POINT3D_VISIBILITY_PYRAMID_LEVELS: usize = 6;

// ---------------------------------------------------------------------
// ObservationManager (`sfm/observation_manager.h/.cc`).
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImagePairStat {
    pub num_tri_corrs: usize,
    pub num_total_corrs: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageStat {
    pub num_observations: usize,
    pub num_correspondences: usize,
    pub num_visible_correspondences: usize,
    pub num_visible_points3d: usize,
    pub num_correspondences_have_point3d: Vec<usize>,
    pub point3d_visibility_pyramid: VisibilityPyramid,
}

/// Port of `class ObservationManager` (`sfm/observation_manager.h:50-234`).
/// See module doc for the "no cached refs" deviation: every method takes
/// `recon`/`graph` explicitly.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObservationManager {
    image_pair_stats: BTreeMap<(ImageT, ImageT), ImagePairStat>,
    image_stats: BTreeMap<ImageT, ImageStat>,
}

fn pair_key(a: ImageT, b: ImageT) -> (ImageT, ImageT) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// `Image::CamFromWorld()` equivalent: this image's absolute pose, derived
/// from its frame's `rig_from_world` via `Frame::SensorFromWorld` (§1.1).
pub fn image_cam_from_world(recon: &Reconstruction, image_id: ImageT) -> SE3 {
    let image = recon.image(image_id);
    let frame = recon.frame(image.frame_id);
    let rig = recon.rig(frame.rig_id());
    frame.sensor_from_world(rig, SensorT::camera(image.camera_id))
}

/// `Image::ProjectionCenter()`: camera center in world coordinates.
pub fn image_projection_center(recon: &Reconstruction, image_id: ImageT) -> Point3<f64> {
    let cam_from_world = image_cam_from_world(recon, image_id);
    Point3::from(cam_from_world.inverse().translation)
}

/// `Camera::HasBogusParams` equivalent. The control configuration
/// (`ba_refine_focal_length=0`, `ba_refine_principal_point=0`,
/// `ba_refine_extra_params=0`, plan §0.1/milestone bullet "no intrinsic
/// refinement") never changes camera intrinsics after they are seeded from
/// the calibrated manifest, so this always returns `false` in practice; kept
/// as a real (if simplified) check — focal-length-to-image-size ratio only,
/// no extra-param/distortion check since this port's cameras are
/// distortion-free pinhole (`Camera::pinhole`) — rather than hard-coding
/// `false`, in case a future caller feeds uncalibrated cameras.
pub fn camera_has_bogus_params(
    camera: &Camera,
    min_focal_length_ratio: f64,
    max_focal_length_ratio: f64,
    _max_extra_param: f64,
) -> bool {
    let Some((fx, fy, _, _)) = camera.intrinsics() else {
        return true;
    };
    let max_dim = camera.width.max(camera.height) as f64;
    if max_dim <= 0.0 {
        return true;
    }
    let min_focal = min_focal_length_ratio * max_dim;
    let max_focal = max_focal_length_ratio * max_dim;
    fx < min_focal || fx > max_focal || fy < min_focal || fy > max_focal
}

pub fn calculate_squared_reprojection_error(
    xy: Point2<f64>,
    point3d: Point3<f64>,
    cam_from_world: &SE3,
    camera: &Camera,
) -> f64 {
    let point_cam = cam_from_world.transform_point(&point3d);
    match camera.project(&point_cam) {
        Some(projected) => (projected - xy).norm_squared(),
        None => f64::INFINITY,
    }
}

pub fn has_point_positive_depth(cam_from_world: &SE3, xyz: Point3<f64>) -> bool {
    cam_from_world.transform_point(&xyz).z > 0.0
}

pub fn calculate_triangulation_angle(
    proj_center1: Point3<f64>,
    proj_center2: Point3<f64>,
    point3d: Point3<f64>,
) -> f64 {
    let ray1 = point3d - proj_center1;
    let ray2 = point3d - proj_center2;
    let n1 = ray1.norm();
    let n2 = ray2.norm();
    if n1 < 1e-12 || n2 < 1e-12 {
        return 0.0;
    }
    (ray1.dot(&ray2) / (n1 * n2)).clamp(-1.0, 1.0).acos()
}

impl ObservationManager {
    /// Port of the `ObservationManager` constructor (`.cc:57-98`).
    pub fn new(recon: &Reconstruction, graph: &CorrespondenceGraph) -> Self {
        let mut mgr = Self::default();

        for (&(i, j), &num_matches) in &graph.num_matches_between_all_images() {
            mgr.image_pair_stats.insert(
                pair_key(i as ImageT, j as ImageT),
                ImagePairStat {
                    num_tri_corrs: 0,
                    num_total_corrs: num_matches,
                },
            );
        }

        for (&image_id, image) in recon.images() {
            mgr.image_stats
                .insert(image_id, Self::init_image_stat(image_id, image, graph));
        }

        // Continuing from an existing (partially registered) reconstruction:
        // not exercised by this port's callers (always starts empty), ported
        // for completeness.
        let reg_frame_ids: Vec<FrameT> = recon.reg_frame_ids().to_vec();
        for frame_id in reg_frame_ids {
            let frame = recon.frame(frame_id);
            let image_ids: Vec<ImageT> = frame.image_ids().collect();
            for image_id in image_ids {
                let image = recon.image(image_id);
                let num_points2d = image.num_points2d();
                for point2d_idx in 0..num_points2d {
                    if image.points2d[point2d_idx].has_point3d() {
                        mgr.set_observation_as_triangulated(
                            recon,
                            graph,
                            image_id,
                            point2d_idx,
                            false,
                        );
                    }
                    for corr in graph.find_correspondences(image_id as usize, point2d_idx) {
                        if let Some(stats) = mgr.image_stats.get_mut(&(corr.image_id as ImageT)) {
                            stats.num_visible_correspondences += 1;
                        }
                    }
                }
            }
        }

        mgr
    }

    fn init_image_stat(
        image_id: ImageT,
        image: &super::reconstruction::Image,
        graph: &CorrespondenceGraph,
    ) -> ImageStat {
        let camera_dims = (image.num_points2d(), image.num_points2d());
        let _ = camera_dims;
        ImageStat {
            num_observations: graph.num_observations_for_image(image_id as usize),
            num_correspondences: graph.num_correspondences_for_image(image_id as usize),
            num_visible_correspondences: 0,
            num_visible_points3d: 0,
            num_correspondences_have_point3d: vec![0; image.num_points2d()],
            // Width/height are filled in by the caller (needs the Camera);
            // see `image_stat_with_camera`.
            point3d_visibility_pyramid: VisibilityPyramid::empty(),
        }
    }

    /// Split out of the constructor because `init_image_stat` above doesn't
    /// have the `Camera` (only `Reconstruction` does); called right after
    /// `new` populates `image_stats` to size the pyramid from the image's
    /// camera (`InitImageStat`, `.cc:163-179`).
    pub fn size_pyramids(&mut self, recon: &Reconstruction) {
        for (&image_id, stat) in self.image_stats.iter_mut() {
            let image = recon.image(image_id);
            let camera = recon.camera(image.camera_id);
            stat.point3d_visibility_pyramid = VisibilityPyramid::new(
                NUM_POINT3D_VISIBILITY_PYRAMID_LEVELS,
                camera.width as usize,
                camera.height as usize,
            );
        }
    }

    // ---- Simple getters (`.h:252-272`) -----------------------------

    pub fn num_observations(&self, image_id: ImageT) -> usize {
        self.image_stats[&image_id].num_observations
    }
    pub fn num_correspondences(&self, image_id: ImageT) -> usize {
        self.image_stats[&image_id].num_correspondences
    }
    pub fn num_visible_correspondences(&self, image_id: ImageT) -> usize {
        self.image_stats[&image_id].num_visible_correspondences
    }
    pub fn num_visible_points3d(&self, image_id: ImageT) -> usize {
        self.image_stats[&image_id].num_visible_points3d
    }
    pub fn point3d_visibility_score(&self, image_id: ImageT) -> u64 {
        self.image_stats[&image_id]
            .point3d_visibility_pyramid
            .score()
    }
    pub fn image_stats(&self) -> &BTreeMap<ImageT, ImageStat> {
        &self.image_stats
    }

    /// Port of `ImagePairs()` (`.h:69,247-250`), snapshotted as
    /// `(num_tri_corrs, num_total_corrs)` per canonical pair — used by
    /// `IncrementalTriangulator::Retriangulate` to find under-reconstructed
    /// pairs without holding a borrow of `self` across the caller's mutation.
    pub fn image_pair_stats_snapshot(&self) -> Vec<((ImageT, ImageT), (usize, usize))> {
        self.image_pair_stats
            .iter()
            .map(|(&k, v)| (k, (v.num_tri_corrs, v.num_total_corrs)))
            .collect()
    }

    // ---- Visibility bookkeeping (`.cc:181-214`) ---------------------

    pub fn increment_correspondence_has_point3d(
        &mut self,
        recon: &Reconstruction,
        image_id: ImageT,
        point2d_idx: Point2DT,
    ) {
        let xy = recon.image(image_id).points2d[point2d_idx].xy;
        let stats = self.image_stats.get_mut(&image_id).unwrap();
        stats.num_correspondences_have_point3d[point2d_idx] += 1;
        if stats.num_correspondences_have_point3d[point2d_idx] == 1 {
            stats.num_visible_points3d += 1;
        }
        stats.point3d_visibility_pyramid.set_point(xy.x, xy.y);
    }

    pub fn decrement_correspondence_has_point3d(
        &mut self,
        recon: &Reconstruction,
        image_id: ImageT,
        point2d_idx: Point2DT,
    ) {
        let xy = recon.image(image_id).points2d[point2d_idx].xy;
        let stats = self.image_stats.get_mut(&image_id).unwrap();
        assert!(stats.num_correspondences_have_point3d[point2d_idx] > 0);
        stats.num_correspondences_have_point3d[point2d_idx] -= 1;
        if stats.num_correspondences_have_point3d[point2d_idx] == 0 {
            stats.num_visible_points3d -= 1;
        }
        stats.point3d_visibility_pyramid.reset_point(xy.x, xy.y);
    }

    /// Port of `SetObservationAsTriangulated` (`.cc:216-247`).
    fn set_observation_as_triangulated(
        &mut self,
        recon: &Reconstruction,
        graph: &CorrespondenceGraph,
        image_id: ImageT,
        point2d_idx: Point2DT,
        is_continued_point3d: bool,
    ) {
        debug_assert!(recon.is_image_registered(image_id));
        let point3d_id = recon.image(image_id).points2d[point2d_idx].point3d_id;
        let corrs: Vec<(ImageT, Point2DT)> = graph
            .find_correspondences(image_id as usize, point2d_idx)
            .iter()
            .map(|c| (c.image_id as ImageT, c.point2d_idx))
            .collect();
        for (corr_image_id, corr_point2d_idx) in corrs {
            self.increment_correspondence_has_point3d(recon, corr_image_id, corr_point2d_idx);
            let corr_point3d_id = recon.image(corr_image_id).points2d[corr_point2d_idx].point3d_id;
            if point3d_id.is_some()
                && point3d_id == corr_point3d_id
                && (is_continued_point3d || image_id < corr_image_id)
            {
                let key = pair_key(image_id, corr_image_id);
                let stats = self.image_pair_stats.entry(key).or_default();
                stats.num_tri_corrs += 1;
            }
        }
    }

    /// Port of `ResetTriObservations` (`.cc:249-276`).
    fn reset_tri_observations(
        &mut self,
        recon: &Reconstruction,
        graph: &CorrespondenceGraph,
        image_id: ImageT,
        point2d_idx: Point2DT,
        is_deleted_point3d: bool,
    ) {
        let point3d_id = recon.image(image_id).points2d[point2d_idx].point3d_id;
        let corrs: Vec<(ImageT, Point2DT)> = graph
            .find_correspondences(image_id as usize, point2d_idx)
            .iter()
            .map(|c| (c.image_id as ImageT, c.point2d_idx))
            .collect();
        for (corr_image_id, corr_point2d_idx) in corrs {
            let corr_point3d_id = recon.image(corr_image_id).points2d[corr_point2d_idx].point3d_id;
            self.decrement_correspondence_has_point3d(recon, corr_image_id, corr_point2d_idx);
            if point3d_id.is_some()
                && point3d_id == corr_point3d_id
                && (!is_deleted_point3d || image_id < corr_image_id)
            {
                let key = pair_key(image_id, corr_image_id);
                let stats = self.image_pair_stats.entry(key).or_default();
                if stats.num_tri_corrs > 0 {
                    stats.num_tri_corrs -= 1;
                }
            }
        }
    }

    // ---- Point3D mutation (`.cc:278-351`) ---------------------------

    pub fn add_point3d(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        xyz: Point3<f64>,
        track: Vec<TrackElement>,
    ) -> Point3DT {
        let point3d_id = recon.add_point3d(xyz, track.clone());
        for el in track {
            self.set_observation_as_triangulated(recon, graph, el.image_id, el.point2d_idx, false);
        }
        point3d_id
    }

    pub fn add_observation(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        point3d_id: Point3DT,
        track_el: TrackElement,
    ) {
        recon.add_observation(point3d_id, track_el);
        self.set_observation_as_triangulated(
            recon,
            graph,
            track_el.image_id,
            track_el.point2d_idx,
            true,
        );
    }

    pub fn delete_point3d(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        point3d_id: Point3DT,
    ) {
        let track = recon.point3d(point3d_id).track.clone();
        for el in &track {
            self.reset_tri_observations(recon, graph, el.image_id, el.point2d_idx, true);
        }
        recon.delete_point3d(point3d_id);
    }

    pub fn delete_observation(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        image_id: ImageT,
        point2d_idx: Point2DT,
    ) {
        let point3d_id = recon.image(image_id).points2d[point2d_idx]
            .point3d_id
            .expect("delete_observation on a point2D without a point3D");
        if recon.point3d(point3d_id).track.len() <= 2 {
            self.delete_point3d(recon, graph, point3d_id);
            return;
        }
        self.reset_tri_observations(recon, graph, image_id, point2d_idx, false);
        // Port of `Reconstruction::DeleteObservation`: remove just this one
        // track element and clear its point2D back-reference.
        let point3d = recon.point3d_mut(point3d_id);
        point3d
            .track
            .retain(|el| !(el.image_id == image_id && el.point2d_idx == point2d_idx));
        recon.image_mut(image_id).points2d[point2d_idx].point3d_id = None;
    }

    pub fn merge_points3d(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        id1: Point3DT,
        id2: Point3DT,
    ) -> Point3DT {
        let track1 = recon.point3d(id1).track.clone();
        for el in &track1 {
            self.reset_tri_observations(recon, graph, el.image_id, el.point2d_idx, true);
        }
        let track2 = recon.point3d(id2).track.clone();
        for el in &track2 {
            self.reset_tri_observations(recon, graph, el.image_id, el.point2d_idx, true);
        }
        let merged_id = recon.merge_points3d(id1, id2);
        let merged_track = recon.point3d(merged_id).track.clone();
        for el in &merged_track {
            self.set_observation_as_triangulated(recon, graph, el.image_id, el.point2d_idx, false);
        }
        merged_id
    }

    // ---- Filtering (`.cc:353-585`) ----------------------------------

    pub fn filter_points3d(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        max_reproj_error: f64,
        min_tri_angle_deg: f64,
        point3d_ids: &[Point3DT],
    ) -> usize {
        let mut n = self.filter_points3d_with_large_reprojection_error(
            recon,
            graph,
            max_reproj_error,
            point3d_ids,
        );
        n += self.filter_points3d_with_small_triangulation_angle(
            recon,
            graph,
            min_tri_angle_deg,
            point3d_ids,
        );
        n
    }

    pub fn filter_points3d_in_images(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        max_reproj_error: f64,
        min_tri_angle_deg: f64,
        image_ids: &[ImageT],
    ) -> usize {
        let mut point3d_ids = Vec::new();
        for &image_id in image_ids {
            for p in &recon.image(image_id).points2d {
                if let Some(id) = p.point3d_id {
                    point3d_ids.push(id);
                }
            }
        }
        point3d_ids.sort_unstable();
        point3d_ids.dedup();
        self.filter_points3d(
            recon,
            graph,
            max_reproj_error,
            min_tri_angle_deg,
            &point3d_ids,
        )
    }

    pub fn filter_all_points3d(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        max_reproj_error: f64,
        min_tri_angle_deg: f64,
    ) -> usize {
        let ids = recon.point3d_ids();
        self.filter_points3d(recon, graph, max_reproj_error, min_tri_angle_deg, &ids)
    }

    pub fn filter_observations_with_negative_depth(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
    ) -> usize {
        let mut num_filtered = 0;
        let frame_ids = recon.reg_frame_ids().to_vec();
        for frame_id in frame_ids {
            let image_ids: Vec<ImageT> = recon.frame(frame_id).image_ids().collect();
            for image_id in image_ids {
                let cam_from_world = image_cam_from_world(recon, image_id);
                let num_points2d = recon.image(image_id).num_points2d();
                for point2d_idx in 0..num_points2d {
                    let point3d_id = recon.image(image_id).points2d[point2d_idx].point3d_id;
                    if let Some(point3d_id) = point3d_id {
                        let xyz = recon.point3d(point3d_id).xyz;
                        if !has_point_positive_depth(&cam_from_world, xyz) {
                            self.delete_observation(recon, graph, image_id, point2d_idx);
                            num_filtered += 1;
                        }
                    }
                }
            }
        }
        num_filtered
    }

    /// Port of `FilterPoints3DWithSmallTriangulationAngle` (`.cc:435-494`).
    pub fn filter_points3d_with_small_triangulation_angle(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        min_tri_angle_deg: f64,
        point3d_ids: &[Point3DT],
    ) -> usize {
        let min_tri_angle_rad = min_tri_angle_deg.to_radians();
        let mut num_filtered = 0;
        let mut proj_centers: BTreeMap<ImageT, Point3<f64>> = BTreeMap::new();
        for &point3d_id in point3d_ids {
            if !recon.exists_point3d(point3d_id) {
                continue;
            }
            let track = recon.point3d(point3d_id).track.clone();
            let mut keep = false;
            'outer: for i1 in 0..track.len() {
                let img1 = track[i1].image_id;
                let pc1 = *proj_centers
                    .entry(img1)
                    .or_insert_with(|| image_projection_center(recon, img1));
                for track_el2 in track.iter().take(i1) {
                    let img2 = track_el2.image_id;
                    let pc2 = *proj_centers
                        .entry(img2)
                        .or_insert_with(|| image_projection_center(recon, img2));
                    let angle =
                        calculate_triangulation_angle(pc1, pc2, recon.point3d(point3d_id).xyz);
                    if angle >= min_tri_angle_rad {
                        keep = true;
                        break 'outer;
                    }
                }
            }
            if !keep {
                num_filtered += track.len();
                self.delete_point3d(recon, graph, point3d_id);
            }
        }
        num_filtered
    }

    /// Port of `FilterPoints3DWithLargeReprojectionError` (`.cc:496-585`),
    /// PIXEL error type only (the only one this port's callers use).
    pub fn filter_points3d_with_large_reprojection_error(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        max_error: f64,
        point3d_ids: &[Point3DT],
    ) -> usize {
        let mut num_filtered = 0;
        for &point3d_id in point3d_ids {
            if !recon.exists_point3d(point3d_id) {
                continue;
            }
            let track = recon.point3d(point3d_id).track.clone();
            if track.len() < 2 {
                num_filtered += track.len();
                self.delete_point3d(recon, graph, point3d_id);
                continue;
            }
            let xyz = recon.point3d(point3d_id).xyz;
            let mut error_sum = 0.0;
            let mut to_delete = Vec::new();
            for el in &track {
                let image = recon.image(el.image_id);
                let camera = recon.camera(image.camera_id);
                let xy = image.points2d[el.point2d_idx].xy;
                let cam_from_world = image_cam_from_world(recon, el.image_id);
                let err =
                    calculate_squared_reprojection_error(xy, xyz, &cam_from_world, camera).sqrt();
                if err > max_error {
                    to_delete.push(*el);
                } else {
                    error_sum += err;
                }
            }
            if to_delete.len() + 1 >= track.len() {
                num_filtered += track.len();
                self.delete_point3d(recon, graph, point3d_id);
            } else {
                num_filtered += to_delete.len();
                for el in &to_delete {
                    self.delete_observation(recon, graph, el.image_id, el.point2d_idx);
                }
                recon.point3d_mut(point3d_id).error = error_sum / (track.len() as f64);
            }
        }
        num_filtered
    }

    // ---- Frame (de)registration (`.cc:587-655`) ---------------------

    pub fn register_frame(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        frame_id: FrameT,
    ) {
        let image_ids: Vec<ImageT> = recon.frame(frame_id).image_ids().collect();
        for image_id in image_ids {
            let num_points2d = recon.image(image_id).num_points2d();
            for point2d_idx in 0..num_points2d {
                for corr in graph.find_correspondences(image_id as usize, point2d_idx) {
                    if let Some(stats) = self.image_stats.get_mut(&(corr.image_id as ImageT)) {
                        stats.num_visible_correspondences += 1;
                    }
                }
            }
        }
        recon.register_frame(frame_id);
    }

    pub fn deregister_frame(
        &mut self,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        frame_id: FrameT,
    ) {
        let image_ids: Vec<ImageT> = recon.frame(frame_id).image_ids().collect();
        for image_id in image_ids {
            let num_points2d = recon.image(image_id).num_points2d();
            for point2d_idx in 0..num_points2d {
                for corr in graph.find_correspondences(image_id as usize, point2d_idx) {
                    if let Some(stats) = self.image_stats.get_mut(&(corr.image_id as ImageT)) {
                        if stats.num_visible_correspondences > 0 {
                            stats.num_visible_correspondences -= 1;
                        }
                    }
                }
                if recon.image(image_id).points2d[point2d_idx].has_point3d() {
                    self.delete_observation(recon, graph, image_id, point2d_idx);
                }
            }
        }
        recon.deregister_frame(frame_id);
    }

    /// Port of `FindFramesToFilter` (`.cc:630-655`).
    pub fn find_frames_to_filter(
        &self,
        recon: &Reconstruction,
        min_focal_length_ratio: f64,
        max_focal_length_ratio: f64,
        max_extra_param: f64,
        min_num_observations: usize,
    ) -> Vec<FrameT> {
        let mut out = Vec::new();
        for &frame_id in recon.reg_frame_ids() {
            let frame = recon.frame(frame_id);
            let mut bogus = false;
            let mut num_observations = 0usize;
            for image_id in frame.image_ids() {
                let image = recon.image(image_id);
                num_observations += image.num_points3d();
                let camera = recon.camera(image.camera_id);
                if camera_has_bogus_params(
                    camera,
                    min_focal_length_ratio,
                    max_focal_length_ratio,
                    max_extra_param,
                ) {
                    bogus = true;
                    break;
                }
            }
            if bogus || num_observations < min_num_observations {
                out.push(frame_id);
            }
        }
        out
    }
}
