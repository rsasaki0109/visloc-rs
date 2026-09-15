//! Faithful (C2-scoped) port of `sfm/incremental_triangulator.{h,cc}`'s
//! [`IncrementalTriangulator`].
//!
//! Ported from (commit `64805cb870b574a569dccc34918d95a2db2b2fee`, pinned by
//! `docs/colmap_rig_mapper_port_plan.md` §0):
//! - `src/colmap/sfm/incremental_triangulator.h` — [`Options`] (`.h:45-91`,
//!   defaults verbatim), [`CorrData`] (`.h:155-161`).
//! - `src/colmap/sfm/incremental_triangulator.cc` — `TriangulateImage`
//!   (`.cc:99-157`), `CompleteImage` (`.cc:159-247`), `CompleteTracks`/
//!   `CompleteAllTracks` (`.cc:249-276`), `MergeTracks`/`MergeAllTracks`
//!   (`.cc:278-305`), `Retriangulate` (`.cc:307-406`), `Find` (`.cc:440-479`),
//!   `Create` (`.cc:481-540`), `Continue` (`.cc:542-586`), `Merge`
//!   (`.cc:588-682`), `Complete` (`.cc:684-770`).
//!
//! ## Deviation: `estimate_triangulation` is a from-scratch robust
//! triangulator, not a port of `estimators/triangulation.cc`
//!
//! `estimators/triangulation.cc` (COLMAP's `EstimateTriangulation` /
//! `TriangulationEstimator`, a `RANSAC<TriangulationEstimator>` instance)
//! was **not** in this task's pinned reading list (only
//! `incremental_triangulator.{h,cc}`, `observation_manager.{h,cc}`,
//! `visibility_pyramid.{h,cc}` were asked for beyond the C1 set). This
//! module implements [`estimate_triangulation`] as a from-first-principles
//! robust multi-view triangulator with the same *contract* COLMAP's
//! `TriangulateTrack` helper (`.cc:39-66`) relies on — exhaustive pairwise
//! hypotheses (COLMAP forces `min_num_trials = C(n,2)` for track length
//! `<=15`, which every track in this port's usage is, so an exhaustive
//! search is a faithful, not merely convenient, substitute for RANSAC
//! here), a linear DLT triangulation from each pair, inlier scoring by
//! [`ResidualType`] (angular error for `Create`, reprojection error in
//! pixels for `CompleteImage`), and a final linear refit over the full
//! inlier set — but the DLT and scoring math themselves are new code, not
//! transcribed from `triangulation.cc`. Flagged here and in the C2 report
//! as the one estimator in this module that is a principled approximation
//! rather than a line-for-line port.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{DMatrix, Point2, Point3};
use visloc_vision::two_view::CorrespondenceGraph;

use super::observation_manager::{
    calculate_squared_reprojection_error, calculate_triangulation_angle, camera_has_bogus_params,
    image_cam_from_world, ObservationManager,
};
use super::reconstruction::{Camera, Reconstruction, TrackElement};
use super::types::{CameraT, ImageT, Point2DT, Point3DT};
use visloc_core::geometry::SE3;

/// Port of `IncrementalTriangulator::Options` (`.h:45-91`), COLMAP defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    pub max_transitivity: usize,
    pub create_max_angle_error: f64,
    pub continue_max_angle_error: f64,
    pub merge_max_reproj_error: f64,
    pub complete_max_reproj_error: f64,
    pub complete_max_transitivity: usize,
    pub re_max_angle_error: f64,
    pub re_min_ratio: f64,
    pub re_max_trials: usize,
    pub min_angle: f64,
    pub ignore_two_view_tracks: bool,
    pub min_focal_length_ratio: f64,
    pub max_focal_length_ratio: f64,
    pub max_extra_param: f64,
    pub random_seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_transitivity: 1,
            create_max_angle_error: 2.0,
            continue_max_angle_error: 2.0,
            merge_max_reproj_error: 4.0,
            complete_max_reproj_error: 4.0,
            complete_max_transitivity: 5,
            re_max_angle_error: 5.0,
            re_min_ratio: 0.2,
            re_max_trials: 1,
            min_angle: 1.5,
            ignore_two_view_tracks: true,
            min_focal_length_ratio: 0.1,
            max_focal_length_ratio: 10.0,
            max_extra_param: 1.0,
            random_seed: 0,
        }
    }
}

/// Port of `CorrData` (`.h:155-161`). Copies (rather than borrows) `camera`/
/// `xy`/`cam_from_world` to sidestep the aliasing `Reconstruction` would
/// otherwise require (see `observation_manager.rs`'s module doc on the same
/// tradeoff).
#[derive(Debug, Clone)]
struct CorrData {
    image_id: ImageT,
    point2d_idx: Point2DT,
    xy: Point2<f64>,
    has_point3d: bool,
    point3d_id: Option<Point3DT>,
    cam_from_world: SE3,
    camera: Camera,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidualType {
    Angular,
    Reprojection,
}

/// Port of `IncrementalTriangulator` (`.h:43-233`). See module doc on the
/// "no cached refs" convention shared with [`super::observation_manager`]:
/// every method takes `recon`/`graph`/`obs` explicitly.
#[derive(Debug, Clone, Default)]
pub struct IncrementalTriangulator {
    camera_has_bogus_params: BTreeMap<CameraT, bool>,
    merge_trials: BTreeSet<(Point3DT, Point3DT)>,
    re_num_trials: BTreeMap<(ImageT, ImageT), usize>,
    modified_point3d_ids: BTreeSet<Point3DT>,
}

impl IncrementalTriangulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn modified_point3d_ids(&mut self, recon: &Reconstruction) -> &BTreeSet<Point3DT> {
        self.modified_point3d_ids
            .retain(|id| recon.exists_point3d(*id));
        &self.modified_point3d_ids
    }

    pub fn clear_modified_point3d_ids(&mut self) {
        self.modified_point3d_ids.clear();
    }

    pub fn add_modified_point3d(&mut self, id: Point3DT) {
        self.modified_point3d_ids.insert(id);
    }

    fn clear_caches(&mut self) {
        self.camera_has_bogus_params.clear();
        self.merge_trials.clear();
    }

    fn has_camera_bogus_params(&mut self, options: &Options, camera: &Camera) -> bool {
        if let Some(&cached) = self.camera_has_bogus_params.get(&camera.id) {
            return cached;
        }
        let bogus = camera_has_bogus_params(
            camera,
            options.min_focal_length_ratio,
            options.max_focal_length_ratio,
            options.max_extra_param,
        );
        self.camera_has_bogus_params.insert(camera.id, bogus);
        bogus
    }

    fn corr_data(recon: &Reconstruction, image_id: ImageT, point2d_idx: Point2DT) -> CorrData {
        let image = recon.image(image_id);
        let point2d = &image.points2d[point2d_idx];
        let camera = recon.camera(image.camera_id).clone();
        let cam_from_world = image_cam_from_world(recon, image_id);
        CorrData {
            image_id,
            point2d_idx,
            xy: point2d.xy,
            has_point3d: point2d.has_point3d(),
            point3d_id: point2d.point3d_id,
            cam_from_world,
            camera,
        }
    }

    /// Port of `Find` (`.cc:440-479`).
    fn find(
        &mut self,
        options: &Options,
        recon: &Reconstruction,
        graph: &CorrespondenceGraph,
        image_id: ImageT,
        point2d_idx: Point2DT,
        transitivity: usize,
    ) -> (usize, Vec<CorrData>) {
        let found: Vec<(ImageT, Point2DT)> = if transitivity <= 1 {
            graph
                .find_correspondences(image_id as usize, point2d_idx)
                .iter()
                .map(|c| (c.image_id as ImageT, c.point2d_idx))
                .collect()
        } else {
            graph
                .extract_transitive_correspondences(image_id as usize, point2d_idx, transitivity)
                .into_iter()
                .map(|c| (c.image_id as ImageT, c.point2d_idx))
                .collect()
        };

        let mut corrs_data = Vec::with_capacity(found.len());
        let mut num_triangulated = 0;
        for (corr_image_id, corr_point2d_idx) in found {
            if !recon.is_image_registered(corr_image_id) {
                continue;
            }
            let corr_camera = recon.camera(recon.image(corr_image_id).camera_id);
            if self.has_camera_bogus_params(options, corr_camera) {
                continue;
            }
            let corr_data = Self::corr_data(recon, corr_image_id, corr_point2d_idx);
            if corr_data.has_point3d {
                num_triangulated += 1;
            }
            corrs_data.push(corr_data);
        }
        (num_triangulated, corrs_data)
    }

    /// Port of `TriangulateImage` (`.cc:99-157`).
    pub fn triangulate_image(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
        image_id: ImageT,
    ) -> usize {
        let mut num_tris = 0;
        self.clear_caches();

        if !recon.is_image_registered(image_id) {
            return 0;
        }
        let camera = recon.camera(recon.image(image_id).camera_id).clone();
        if self.has_camera_bogus_params(options, &camera) {
            return 0;
        }

        let num_points2d = recon.image(image_id).num_points2d();
        for point2d_idx in 0..num_points2d {
            let (num_triangulated, mut corrs_data) = self.find(
                options,
                recon,
                graph,
                image_id,
                point2d_idx,
                options.max_transitivity,
            );
            if corrs_data.is_empty() {
                continue;
            }

            let ref_corr = Self::corr_data(recon, image_id, point2d_idx);
            if num_triangulated == 0 {
                corrs_data.push(ref_corr);
                num_tris += self.create(options, recon, graph, obs, &corrs_data);
            } else {
                num_tris += self.r#continue(options, recon, graph, obs, &ref_corr, &corrs_data);
                // `Continue` may have just triangulated `ref_corr`'s own
                // point2D (via `obs.add_observation`); COLMAP's
                // `ref_corr_data.point2D` is a live pointer so `Create`'s
                // internal `!HasPoint3D()` filter sees this automatically
                // (`incremental_mapper.cc`-equivalent, `.cc:148-149`). This
                // port's `CorrData` is a by-value snapshot (see the module
                // doc's "no cached refs" rationale), so it must be
                // re-fetched here or `Create` would filter on a stale
                // `has_point3d=false` and try to re-triangulate an
                // already-triangulated point2D — observed as a
                // `delete_observation on a point2D without a point3D` panic
                // on real tier-1000 data during this port's development
                // (a duplicate track element from double-triangulating the
                // same point2D), fixed here.
                let ref_corr = Self::corr_data(recon, image_id, point2d_idx);
                corrs_data.push(ref_corr);
                num_tris += self.create(options, recon, graph, obs, &corrs_data);
            }
        }
        num_tris
    }

    /// Port of `CompleteImage` (`.cc:159-247`).
    pub fn complete_image(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
        image_id: ImageT,
    ) -> usize {
        let mut num_tris = 0;
        self.clear_caches();

        if !recon.is_image_registered(image_id) {
            return 0;
        }
        let camera = recon.camera(recon.image(image_id).camera_id).clone();
        if self.has_camera_bogus_params(options, &camera) {
            return 0;
        }

        let num_points2d = recon.image(image_id).num_points2d();
        for point2d_idx in 0..num_points2d {
            let point3d_id = recon.image(image_id).points2d[point2d_idx].point3d_id;
            if let Some(point3d_id) = point3d_id {
                num_tris += self.complete(options, recon, graph, obs, point3d_id);
                continue;
            }
            if options.ignore_two_view_tracks
                && graph.is_two_view_observation(image_id as usize, point2d_idx)
            {
                continue;
            }
            let (num_triangulated, mut corrs_data) = self.find(
                options,
                recon,
                graph,
                image_id,
                point2d_idx,
                options.max_transitivity,
            );
            if num_triangulated > 0 || corrs_data.is_empty() {
                continue;
            }
            let ref_corr = Self::corr_data(recon, image_id, point2d_idx);
            corrs_data.push(ref_corr);

            let max_error_px = options.complete_max_reproj_error;
            let min_tri_angle_rad = options.min_angle.to_radians();
            let Some((xyz, inliers)) = estimate_triangulation(
                &corrs_data,
                ResidualType::Reprojection,
                max_error_px,
                min_tri_angle_rad,
            ) else {
                continue;
            };
            let mut track = Vec::new();
            for (i, corr) in corrs_data.iter().enumerate() {
                if inliers[i] {
                    track.push(TrackElement {
                        image_id: corr.image_id,
                        point2d_idx: corr.point2d_idx,
                    });
                    num_tris += 1;
                }
            }
            let point3d_id = obs.add_point3d(recon, graph, xyz, track);
            self.modified_point3d_ids.insert(point3d_id);
        }
        num_tris
    }

    /// Port of `CompleteTracks` (`.cc:249-262`).
    pub fn complete_tracks(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
        point3d_ids: &[Point3DT],
    ) -> usize {
        self.clear_caches();
        let mut n = 0;
        for &id in point3d_ids {
            n += self.complete(options, recon, graph, obs, id);
        }
        n
    }

    /// Port of `CompleteAllTracks` (`.cc:264-276`).
    pub fn complete_all_tracks(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
    ) -> usize {
        let ids = recon.point3d_ids();
        self.complete_tracks(options, recon, graph, obs, &ids)
    }

    /// Port of `MergeTracks` (`.cc:278-291`).
    pub fn merge_tracks(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
        point3d_ids: &[Point3DT],
    ) -> usize {
        self.clear_caches();
        let mut n = 0;
        for &id in point3d_ids {
            n += self.merge(options, recon, graph, obs, id);
        }
        n
    }

    /// Port of `MergeAllTracks` (`.cc:293-305`).
    pub fn merge_all_tracks(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
    ) -> usize {
        let ids = recon.point3d_ids();
        self.merge_tracks(options, recon, graph, obs, &ids)
    }

    /// Port of `Retriangulate` (`.cc:307-406`).
    pub fn retriangulate(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
    ) -> usize {
        let mut num_tris = 0;
        self.clear_caches();

        let mut re_options = options.clone();
        re_options.continue_max_angle_error = options.re_max_angle_error;

        let pairs: Vec<((ImageT, ImageT), (usize, usize))> =
            obs.image_pair_stats_snapshot().into_iter().collect();

        for ((image_id1, image_id2), (num_tri_corrs, num_total_corrs)) in pairs {
            if num_total_corrs == 0 {
                continue;
            }
            let tri_ratio = num_tri_corrs as f64 / num_total_corrs as f64;
            if tri_ratio >= options.re_min_ratio {
                continue;
            }
            if !recon.is_image_registered(image_id1) || !recon.is_image_registered(image_id2) {
                continue;
            }
            let key = (image_id1.min(image_id2), image_id1.max(image_id2));
            let trials = self.re_num_trials.entry(key).or_insert(0);
            if *trials >= options.re_max_trials {
                continue;
            }
            *trials += 1;

            let camera1 = recon.camera(recon.image(image_id1).camera_id).clone();
            let camera2 = recon.camera(recon.image(image_id2).camera_id).clone();
            if self.has_camera_bogus_params(options, &camera1)
                || self.has_camera_bogus_params(options, &camera2)
            {
                continue;
            }

            let matches = matches_between_images(graph, recon, image_id1, image_id2);

            for (idx1, idx2) in matches {
                let p1_has = recon.image(image_id1).points2d[idx1].has_point3d();
                let p2_has = recon.image(image_id2).points2d[idx2].has_point3d();
                if p1_has && p2_has {
                    continue;
                }
                let corr1 = Self::corr_data(recon, image_id1, idx1);
                let corr2 = Self::corr_data(recon, image_id2, idx2);
                if p1_has && !p2_has {
                    num_tris += self.r#continue(&re_options, recon, graph, obs, &corr2, &[corr1]);
                } else if !p1_has && p2_has {
                    num_tris += self.r#continue(&re_options, recon, graph, obs, &corr1, &[corr2]);
                } else {
                    num_tris += self.create(options, recon, graph, obs, &[corr1, corr2]);
                }
            }
        }
        num_tris
    }

    /// Port of `Create` (`.cc:481-540`).
    /// Port of `Create` (`.cc:481-540`). Iterative, not recursive: COLMAP's
    /// `corr_data.point2D` is a live pointer into the `Reconstruction`, so
    /// its recursive re-entry re-filters `corrs_data` against
    /// *fresh* `HasPoint3D()` state. This port's [`CorrData`] instead
    /// snapshots `has_point3d` by value at construction time (see
    /// `observation_manager.rs`'s "no cached refs" rationale), so a naive
    /// recursive port that reused the same `CorrData` vector across levels
    /// would filter on a permanently-stale snapshot and recurse forever
    /// whenever a level didn't consume every correspondence (observed as
    /// an unbounded-recursion stack overflow on real 1k-tier data during
    /// this port's development — see the C2 report). This loop re-derives
    /// fresh `CorrData` (via [`Self::corr_data`], reading the just-mutated
    /// `recon`) before every subsequent round, exactly reproducing
    /// COLMAP's live-pointer re-filter semantics without recursion.
    fn create(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
        corrs_data: &[CorrData],
    ) -> usize {
        let mut current: Vec<CorrData> = corrs_data
            .iter()
            .filter(|c| !c.has_point3d)
            .cloned()
            .collect();
        let mut total = 0usize;
        // Safety cap on rounds: with the `estimate_triangulation` hypothesis
        // cap above bounding each round's own cost, this is a defense-in-
        // depth bound (not expected to bind in practice) against a
        // pathologically inconsistent correspondence pool re-splitting into
        // many small remainders round after round.
        const MAX_ROUNDS: usize = 200;
        let mut rounds = 0usize;

        loop {
            rounds += 1;
            if rounds > MAX_ROUNDS || current.len() < 2 {
                break;
            }
            if options.ignore_two_view_tracks && current.len() == 2 {
                let c0 = &current[0];
                if graph.is_two_view_observation(c0.image_id as usize, c0.point2d_idx) {
                    break;
                }
            }

            let max_angle_error_rad = options.create_max_angle_error.to_radians();
            let min_tri_angle_rad = options.min_angle.to_radians();
            let Some((xyz, inliers)) = estimate_triangulation(
                &current,
                ResidualType::Angular,
                max_angle_error_rad,
                min_tri_angle_rad,
            ) else {
                break;
            };

            let mut track = Vec::new();
            for (i, corr) in current.iter().enumerate() {
                if inliers[i] {
                    track.push(TrackElement {
                        image_id: corr.image_id,
                        point2d_idx: corr.point2d_idx,
                    });
                }
            }
            let track_length = track.len();
            let point3d_id = obs.add_point3d(recon, graph, xyz, track);
            self.modified_point3d_ids.insert(point3d_id);
            total += track_length;

            const MIN_RECURSIVE_TRACK_LENGTH: usize = 3;
            if current.len() - track_length >= MIN_RECURSIVE_TRACK_LENGTH {
                current = current
                    .iter()
                    .map(|c| Self::corr_data(recon, c.image_id, c.point2d_idx))
                    .filter(|c| !c.has_point3d)
                    .collect();
                continue;
            }
            break;
        }

        total
    }

    /// Port of `Continue` (`.cc:542-586`).
    fn r#continue(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
        ref_corr: &CorrData,
        corrs_data: &[CorrData],
    ) -> usize {
        if ref_corr.has_point3d {
            return 0;
        }
        let mut best_angle_error = f64::MAX;
        let mut best_point3d_id: Option<Point3DT> = None;
        for corr in corrs_data {
            let Some(point3d_id) = corr.point3d_id else {
                continue;
            };
            let xyz = recon.point3d(point3d_id).xyz;
            let angle_error = angular_reprojection_error(
                ref_corr.xy,
                xyz,
                &ref_corr.cam_from_world,
                &ref_corr.camera,
            );
            if angle_error < best_angle_error {
                best_angle_error = angle_error;
                best_point3d_id = Some(point3d_id);
            }
        }
        let max_angle_error_rad = options.continue_max_angle_error.to_radians();
        if let Some(point3d_id) = best_point3d_id {
            if best_angle_error <= max_angle_error_rad {
                obs.add_observation(
                    recon,
                    graph,
                    point3d_id,
                    TrackElement {
                        image_id: ref_corr.image_id,
                        point2d_idx: ref_corr.point2d_idx,
                    },
                );
                self.modified_point3d_ids.insert(point3d_id);
                return 1;
            }
        }
        0
    }

    /// Port of `Merge` (`.cc:588-682`).
    /// Port of `Merge` (`.cc:588-682`). Iterative, not recursive: COLMAP's
    /// `Merge` recurses into the freshly-merged point on every successful
    /// merge (`.cc:671`), and on real (dense, long-chain) graphs this
    /// recursion can run deep enough to overflow the stack (observed on
    /// the tier-1000 real-data run during this port's development — see
    /// the C2 report). This loop reproduces the exact same return-value
    /// contract (return the *last* successful merge's `num_merged`, or `0`
    /// if no merge ever succeeded) without unbounded call-stack growth.
    fn merge(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
        point3d_id: Point3DT,
    ) -> usize {
        let mut current_id = point3d_id;
        let mut last_num_merged = 0usize;
        while let Some((merged_id, num_merged)) =
            self.try_merge_once(options, recon, graph, obs, current_id)
        {
            last_num_merged = num_merged;
            current_id = merged_id;
        }
        last_num_merged
    }

    /// One non-recursive merge attempt for `merge`'s loop above: tries every
    /// untried candidate merge for `point3d_id`'s track and, on the first
    /// success, returns `(merged_point3d_id, num_merged)`.
    fn try_merge_once(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
        point3d_id: Point3DT,
    ) -> Option<(Point3DT, usize)> {
        if !recon.exists_point3d(point3d_id) {
            return None;
        }
        let max_sq_reproj_error = options.merge_max_reproj_error * options.merge_max_reproj_error;
        let track = recon.point3d(point3d_id).track.clone();

        for el in &track {
            let corrs: Vec<(ImageT, Point2DT)> = graph
                .find_correspondences(el.image_id as usize, el.point2d_idx)
                .iter()
                .map(|c| (c.image_id as ImageT, c.point2d_idx))
                .collect();
            for (corr_image_id, corr_point2d_idx) in corrs {
                if !recon.is_image_registered(corr_image_id) {
                    continue;
                }
                let corr_point3d_id =
                    recon.image(corr_image_id).points2d[corr_point2d_idx].point3d_id;
                let Some(corr_point3d_id) = corr_point3d_id else {
                    continue;
                };
                if corr_point3d_id == point3d_id {
                    continue;
                }
                let key = if point3d_id < corr_point3d_id {
                    (point3d_id, corr_point3d_id)
                } else {
                    (corr_point3d_id, point3d_id)
                };
                if !self.merge_trials.insert(key) {
                    continue;
                }

                let p3d = recon.point3d(point3d_id);
                let corr_p3d = recon.point3d(corr_point3d_id);
                let n1 = p3d.track.len() as f64;
                let n2 = corr_p3d.track.len() as f64;
                let merged_xyz =
                    Point3::from((p3d.xyz.coords * n1 + corr_p3d.xyz.coords * n2) / (n1 + n2));

                let mut merge_ok = true;
                'check: for check_track in [&p3d.track, &corr_p3d.track] {
                    for test_el in check_track {
                        let test_image = recon.image(test_el.image_id);
                        let test_camera = recon.camera(test_image.camera_id);
                        let test_xy = test_image.points2d[test_el.point2d_idx].xy;
                        let test_cam_from_world = image_cam_from_world(recon, test_el.image_id);
                        let err = calculate_squared_reprojection_error(
                            test_xy,
                            merged_xyz,
                            &test_cam_from_world,
                            test_camera,
                        );
                        if err > max_sq_reproj_error {
                            merge_ok = false;
                            break 'check;
                        }
                    }
                }

                if merge_ok {
                    let num_merged = p3d.track.len() + corr_p3d.track.len();
                    let merged_id = obs.merge_points3d(recon, graph, point3d_id, corr_point3d_id);
                    self.modified_point3d_ids.remove(&point3d_id);
                    self.modified_point3d_ids.remove(&corr_point3d_id);
                    self.modified_point3d_ids.insert(merged_id);
                    return Some((merged_id, num_merged));
                }
            }
        }
        None
    }

    /// Port of `Complete` (`.cc:684-770`). Simplified BFS (no reusable
    /// scratch-buffer members — Rust ownership makes the member-held-Vec
    /// swap-buffer optimization not worth the complexity here; functionally
    /// identical, just reallocates per call).
    fn complete(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
        point3d_id: Point3DT,
    ) -> usize {
        let mut num_completed = 0;
        if !recon.exists_point3d(point3d_id) {
            return 0;
        }
        let max_sq_reproj_error =
            options.complete_max_reproj_error * options.complete_max_reproj_error;
        let xyz = recon.point3d(point3d_id).xyz;

        let mut curr_queue: Vec<TrackElement> = recon.point3d(point3d_id).track.clone();
        let mut next_queue: Vec<TrackElement> = Vec::new();
        let mut visited: BTreeSet<(ImageT, Point2DT)> = curr_queue
            .iter()
            .map(|el| (el.image_id, el.point2d_idx))
            .collect();

        let max_transitivity = options.complete_max_transitivity;
        for transitivity in 1..=max_transitivity {
            while let Some(elem) = curr_queue.pop() {
                let corrs: Vec<(ImageT, Point2DT)> = graph
                    .find_correspondences(elem.image_id as usize, elem.point2d_idx)
                    .iter()
                    .map(|c| (c.image_id as ImageT, c.point2d_idx))
                    .collect();
                for (corr_image_id, corr_point2d_idx) in corrs {
                    if !visited.insert((corr_image_id, corr_point2d_idx)) {
                        continue;
                    }
                    if !recon.is_image_registered(corr_image_id) {
                        continue;
                    }
                    if recon.image(corr_image_id).points2d[corr_point2d_idx].has_point3d() {
                        continue;
                    }
                    let camera = recon.camera(recon.image(corr_image_id).camera_id).clone();
                    if self.has_camera_bogus_params(options, &camera) {
                        continue;
                    }
                    let xy = recon.image(corr_image_id).points2d[corr_point2d_idx].xy;
                    let cam_from_world = image_cam_from_world(recon, corr_image_id);
                    let err =
                        calculate_squared_reprojection_error(xy, xyz, &cam_from_world, &camera);
                    if err > max_sq_reproj_error {
                        continue;
                    }
                    obs.add_observation(
                        recon,
                        graph,
                        point3d_id,
                        TrackElement {
                            image_id: corr_image_id,
                            point2d_idx: corr_point2d_idx,
                        },
                    );
                    self.modified_point3d_ids.insert(point3d_id);
                    if transitivity < max_transitivity {
                        next_queue.push(TrackElement {
                            image_id: corr_image_id,
                            point2d_idx: corr_point2d_idx,
                        });
                    }
                    num_completed += 1;
                }
            }
            if next_queue.is_empty() {
                break;
            }
            std::mem::swap(&mut curr_queue, &mut next_queue);
        }
        num_completed
    }
}

fn angular_reprojection_error(
    xy: Point2<f64>,
    xyz: Point3<f64>,
    cam_from_world: &SE3,
    camera: &Camera,
) -> f64 {
    let point_cam = cam_from_world.transform_point(&xyz);
    let predicted_ray = point_cam.coords.normalize();
    let Some(observed_normalized) = camera.normalize_pixel(&xy) else {
        return f64::MAX;
    };
    let observed_ray =
        nalgebra::Vector3::new(observed_normalized.x, observed_normalized.y, 1.0).normalize();
    predicted_ray.dot(&observed_ray).clamp(-1.0, 1.0).acos()
}

/// See module doc: a from-scratch exhaustive-pairwise robust triangulator
/// standing in for COLMAP's `EstimateTriangulation`. `max_error` is in
/// radians for [`ResidualType::Angular`], pixels for
/// [`ResidualType::Reprojection`].
fn estimate_triangulation(
    corrs: &[CorrData],
    residual_type: ResidualType,
    max_error: f64,
    min_tri_angle_rad: f64,
) -> Option<(Point3<f64>, Vec<bool>)> {
    let n = corrs.len();
    if n < 2 {
        return None;
    }

    let residual = |xyz: Point3<f64>, corr: &CorrData| -> f64 {
        match residual_type {
            ResidualType::Angular => {
                angular_reprojection_error(corr.xy, xyz, &corr.cam_from_world, &corr.camera)
            }
            ResidualType::Reprojection => calculate_squared_reprojection_error(
                corr.xy,
                xyz,
                &corr.cam_from_world,
                &corr.camera,
            )
            .sqrt(),
        }
    };

    let mut best_score = 0usize;
    let mut best_pair: Option<(usize, usize)> = None;
    let mut best_xyz = Point3::origin();

    // Cap hypothesis-pair *generation* to a bounded subset of `corrs`
    // (scoring below still runs against the *full* `corrs` slice, so no
    // legitimate inlier is ever missed). Real correspondence graphs built
    // from retrieval-based candidate generation (not just sequential
    // neighbors) can give one point2D observation a correspondence-list
    // length in the hundreds (e.g. a frequently-revisited OpenLORIS
    // corridor location) — an exhaustive O(n^2) pair enumeration with O(n)
    // scoring per pair is then O(n^3), which was observed to exhaust
    // system memory/CPU on a real tier-1000 run during this port's
    // development (see the C2 report). COLMAP's own `EstimateTriangulation`
    // is itself a *bounded*-iteration RANSAC for any track length beyond
    // its `kExhaustiveSamplingThreshold=15` special case (`TriangulateTrack`,
    // `incremental_triangulator.cc:39-66`); this cap reproduces that bound
    // (a fixed hypothesis budget, not one growing with `n`) via a
    // deterministic evenly-strided subset rather than RANSAC's random
    // sampling, since determinism is otherwise free here (no threading of
    // `random_seed` into this free function would be needed).
    const MAX_HYPOTHESIS_CORRS: usize = 40;
    let hypothesis_indices: Vec<usize> = if n <= MAX_HYPOTHESIS_CORRS {
        (0..n).collect()
    } else {
        let stride = ((n as f64) / (MAX_HYPOTHESIS_CORRS as f64)).ceil() as usize;
        (0..n).step_by(stride.max(1)).collect()
    };

    for &i in &hypothesis_indices {
        for &j in hypothesis_indices.iter().filter(|&&j| j > i) {
            let Some(xyz) = triangulate_dlt(&[
                (&corrs[i].cam_from_world, corrs[i].xy, &corrs[i].camera),
                (&corrs[j].cam_from_world, corrs[j].xy, &corrs[j].camera),
            ]) else {
                continue;
            };
            if !positive_depth(&corrs[i].cam_from_world, xyz)
                || !positive_depth(&corrs[j].cam_from_world, xyz)
            {
                continue;
            }
            let angle = calculate_triangulation_angle(
                cam_center(&corrs[i].cam_from_world),
                cam_center(&corrs[j].cam_from_world),
                xyz,
            );
            if angle < min_tri_angle_rad {
                continue;
            }
            let score = corrs
                .iter()
                .filter(|c| residual(xyz, c) <= max_error)
                .count();
            if score > best_score {
                best_score = score;
                best_pair = Some((i, j));
                best_xyz = xyz;
            }
        }
    }

    let (i0, j0) = best_pair?;
    if best_score < 2 {
        return None;
    }

    // Final refit over the full inlier set (linear DLT), then recompute the
    // inlier mask against the refit point.
    let inlier_views: Vec<(&SE3, Point2<f64>, &Camera)> = corrs
        .iter()
        .filter(|c| residual(best_xyz, c) <= max_error)
        .map(|c| (&c.cam_from_world, c.xy, &c.camera))
        .collect();
    let refit_xyz = if inlier_views.len() >= 2 {
        triangulate_dlt(&inlier_views).unwrap_or(best_xyz)
    } else {
        best_xyz
    };

    let inlier_mask: Vec<bool> = corrs
        .iter()
        .map(|c| residual(refit_xyz, c) <= max_error)
        .collect();
    // The min_tri_angle check must still hold for at least the seed pair at
    // the refit point (mirrors the pre-refit gate above staying valid).
    let seed_angle = calculate_triangulation_angle(
        cam_center(&corrs[i0].cam_from_world),
        cam_center(&corrs[j0].cam_from_world),
        refit_xyz,
    );
    if seed_angle < min_tri_angle_rad * 0.5 {
        // Refit drifted the point far enough that the seed pair's angle
        // collapsed; fall back to the pre-refit estimate instead of failing
        // outright (COLMAP's own estimator similarly tolerates a final
        // linear-refit step without re-deriving the RANSAC-stage gate).
        let inlier_mask: Vec<bool> = corrs
            .iter()
            .map(|c| residual(best_xyz, c) <= max_error)
            .collect();
        return Some((best_xyz, inlier_mask));
    }

    Some((refit_xyz, inlier_mask))
}

/// No `ExtractMatchesBetweenImages` exists on this port's
/// `CorrespondenceGraph` (its API is per-`(image, point2D)`, not per-pair —
/// see `crates/vision/src/two_view/correspondence_graph.rs`); reconstructed
/// here by scanning image1's per-point2D correspondences for hits on
/// image2, the direct equivalent COLMAP's own `ExtractMatchesBetweenImages`
/// performs internally over its `flat_corrs`.
pub(super) fn matches_between_images(
    graph: &CorrespondenceGraph,
    recon: &Reconstruction,
    image_id1: ImageT,
    image_id2: ImageT,
) -> Vec<(Point2DT, Point2DT)> {
    let num_points2d = recon.image(image_id1).num_points2d();
    let mut out = Vec::new();
    for idx1 in 0..num_points2d {
        for corr in graph.find_correspondences(image_id1 as usize, idx1) {
            if corr.image_id as ImageT == image_id2 {
                out.push((idx1, corr.point2d_idx));
            }
        }
    }
    out
}

fn positive_depth(cam_from_world: &SE3, xyz: Point3<f64>) -> bool {
    cam_from_world.transform_point(&xyz).z > 0.0
}

fn cam_center(cam_from_world: &SE3) -> Point3<f64> {
    Point3::from(cam_from_world.inverse().translation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colmap_incremental::observation_manager::ObservationManager;
    use crate::colmap_incremental::pipeline::reconstruction_from_cache;
    use crate::colmap_incremental::test_support::build_synthetic_rig_scene;

    /// C2 task item 7: "triangulator on a synthetic scene". Two rig frames
    /// registered at their exact ground-truth poses (isolating the
    /// triangulator from registration); every synthetic point is visible
    /// from every image and directly matched via the correspondence graph,
    /// so `TriangulateImage` should recover all of them close to their true
    /// position.
    #[test]
    fn triangulate_image_recovers_synthetic_points() {
        let scene = build_synthetic_rig_scene(2, 1);
        let mut recon = reconstruction_from_cache(&scene.db);
        let graph = scene.db.correspondence_graph();

        for (&frame_id, gt) in &scene.ground_truth_rig_from_world {
            recon.frame_mut(frame_id).set_rig_from_world(gt.clone());
            recon.register_frame(frame_id);
        }
        let mut obs = ObservationManager::new(&recon, graph);
        obs.size_pyramids(&recon);
        let mut tri = IncrementalTriangulator::new();
        let options = Options::default();

        let mut total_tris = 0;
        for &(i1, i2) in &scene.images_per_frame {
            total_tris += tri.triangulate_image(&options, &mut recon, graph, &mut obs, i1);
            total_tris += tri.triangulate_image(&options, &mut recon, graph, &mut obs, i2);
        }
        assert!(total_tris > 0, "expected some points triangulated");
        assert_eq!(
            recon.num_points3d(),
            scene.ground_truth_points.len(),
            "every synthetic point should triangulate given full visibility+correspondences"
        );

        for point3d in recon.points3d().values() {
            let closest = scene
                .ground_truth_points
                .iter()
                .copied()
                .min_by(|a, b| {
                    (a - point3d.xyz)
                        .norm()
                        .partial_cmp(&(b - point3d.xyz).norm())
                        .unwrap()
                })
                .unwrap();
            let err = (closest - point3d.xyz).norm();
            assert!(
                err < 1e-3,
                "triangulated point {:?} far from nearest ground truth {:?} (err {err})",
                point3d.xyz,
                closest
            );
        }
    }

    /// Regression test for the infinite-recursion bug found running this
    /// port on real tier-1000 data (see `create`'s doc comment): builds a
    /// single point2D observation (`(image A, idx 0)`) whose correspondence
    /// pool is deliberately "poisoned" with observations of *two* distinct,
    /// spatially separated 3D points (a plausible real-world bad-match
    /// scenario, not just a synthetic edge case) — `A`+`B1..B3` truly
    /// observe `P1`; `C1..C3` truly observe a different `P2` but are
    /// (incorrectly) wired as correspondents of `(A, 0)` too. `Create`'s
    /// best-inlier-set search picks the larger `P1` group first
    /// (track length 4 of the original 7-correspondence pool), leaving
    /// exactly 3 remaining correspondences — `>= kMinRecursiveTrackLength`
    /// — which must trigger a **second** round that creates `P2` from the
    /// leftover `C1..C3`. Before the fix, the second round re-filtered a
    /// permanently-stale `has_point3d=false` snapshot of the *same* 7-item
    /// pool and recursed (looped) forever; this test bounds real wall time
    /// so a regression reliably fails instead of hanging.
    #[test]
    fn create_recursive_remainder_terminates_and_creates_two_points() {
        use crate::colmap_incremental::reconstruction::{Camera, Image, Point2D, Reconstruction};
        use crate::colmap_incremental::types::{DataT, Frame, Rig, SensorT};
        use nalgebra::Vector3;
        use visloc_core::geometry::SE3;
        use visloc_vision::two_view::ConfigurationType;

        let mut recon = Reconstruction::new();
        let mut rig = Rig::new();
        rig.set_rig_id(0);
        rig.add_ref_sensor(SensorT::camera(1));
        recon.add_rig(rig);
        recon.add_camera(Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0));

        let p1 = Point3::new(0.0, 0.0, 3.0);
        let p2 = Point3::new(2.0, 0.0, 3.0);

        // (frame_id, camera_x_position, true_world_point)
        let cams: Vec<(u64, f64, Point3<f64>)> = vec![
            (0, 0.0, p1),  // A
            (1, 0.3, p1),  // B1
            (2, -0.3, p1), // B2
            (3, 0.15, p1), // B3
            (4, 2.0, p2),  // C1
            (5, 2.3, p2),  // C2
            (6, 1.7, p2),  // C3
        ];

        let mut graph = CorrespondenceGraph::new();
        for &(frame_id, _, _) in &cams {
            graph.add_image(frame_id as usize, 1);
        }

        for &(frame_id, cam_x, true_point) in &cams {
            let mut frame = Frame::new(frame_id, 0);
            frame.add_data_id(DataT::camera(1, frame_id));
            recon.add_frame(frame);

            // Pure-translation pose: camera sits at (cam_x, 0, 0), no rotation.
            let cam_from_world = SE3::new(
                nalgebra::UnitQuaternion::identity(),
                Vector3::new(-cam_x, 0.0, 0.0),
            );
            let camera = recon.camera(1).clone();
            let xy = camera
                .project(&cam_from_world.transform_point(&true_point))
                .expect("synthetic point must be visible");

            let mut image = Image::new(frame_id, 1, frame_id, format!("f{frame_id}.png"));
            image.points2d = vec![Point2D::new(xy)];
            recon.add_image(image);

            recon.frame_mut(frame_id).set_rig_from_world(cam_from_world);
            recon.register_frame(frame_id);
        }

        // The "poisoned" correspondence pool: (A,0) <-> every other image's
        // point 0, regardless of which true 3D point they actually observe.
        for &(frame_id, _, _) in cams.iter().skip(1) {
            graph
                .add_two_view_geometry(
                    0,
                    frame_id as usize,
                    &[(0, 0)],
                    ConfigurationType::Calibrated,
                )
                .unwrap();
        }
        graph.finalize();

        let mut obs = ObservationManager::new(&recon, &graph);
        obs.size_pyramids(&recon);
        let mut tri = IncrementalTriangulator::new();
        let options = Options::default();

        let num_tris = tri.triangulate_image(&options, &mut recon, &graph, &mut obs, 0);

        assert_eq!(
            num_tris, 7,
            "expected all 7 poisoned correspondences to end up triangulated across two rounds"
        );
        assert_eq!(
            recon.num_points3d(),
            2,
            "expected exactly two distinct 3D points (P1 and P2), not one merged/bogus point"
        );

        let mut xyzs: Vec<Point3<f64>> = recon.points3d().values().map(|p| p.xyz).collect();
        xyzs.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
        assert!(
            (xyzs[0] - p1).norm() < 1e-6,
            "first point should match P1, got {:?}",
            xyzs[0]
        );
        assert!(
            (xyzs[1] - p2).norm() < 1e-6,
            "second point should match P2, got {:?}",
            xyzs[1]
        );

        let track_lens: Vec<usize> = {
            let mut v: Vec<usize> = recon.points3d().values().map(|p| p.track.len()).collect();
            v.sort_unstable();
            v
        };
        assert_eq!(
            track_lens,
            vec![3, 4],
            "expected track lengths 4 (P1: A,B1,B2,B3) and 3 (P2: C1,C2,C3)"
        );
    }

    /// Regression test for the O(n^3) `estimate_triangulation` blowup found
    /// running this port on real tier-1000 data (see `estimate_triangulation`'s
    /// doc comment): a single point2D observation with a correspondence pool
    /// of 200 images — plausible on a real retrieval-based (not purely
    /// sequential) correspondence graph, e.g. a frequently-revisited
    /// OpenLORIS corridor location — all genuinely observing the same true
    /// 3D point. Before the hypothesis-pair cap, `Create`'s exhaustive
    /// `O(n^2)` pair enumeration with `O(n)` scoring per pair
    /// (`O(n^3)` total) made a single `TriangulateImage` call on data this
    /// size expensive enough to exhaust system memory (observed directly on
    /// a real run); this test bounds wall time so a regression reliably
    /// fails fast instead of hanging/OOMing.
    #[test]
    fn create_scales_to_large_correspondence_pool() {
        use crate::colmap_incremental::reconstruction::{Camera, Image, Point2D, Reconstruction};
        use crate::colmap_incremental::types::{DataT, Frame, Rig, SensorT};
        use nalgebra::Vector3;
        use visloc_core::geometry::SE3;
        use visloc_vision::two_view::ConfigurationType;

        const NUM_CORRESPONDENTS: u64 = 200;

        let mut recon = Reconstruction::new();
        let mut rig = Rig::new();
        rig.set_rig_id(0);
        rig.add_ref_sensor(SensorT::camera(1));
        recon.add_rig(rig);
        recon.add_camera(Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0));

        let p1 = Point3::new(0.0, 0.0, 3.0);
        let mut graph = CorrespondenceGraph::new();
        for frame_id in 0..=NUM_CORRESPONDENTS {
            graph.add_image(frame_id as usize, 1);
        }

        for frame_id in 0..=NUM_CORRESPONDENTS {
            // Spread camera x-positions over a wide enough baseline that
            // every pair clears the default 1.5deg `min_angle` gate.
            let cam_x = -0.5 + (frame_id as f64) * (1.0 / NUM_CORRESPONDENTS as f64);
            let mut frame = Frame::new(frame_id, 0);
            frame.add_data_id(DataT::camera(1, frame_id));
            recon.add_frame(frame);

            let cam_from_world = SE3::new(
                nalgebra::UnitQuaternion::identity(),
                Vector3::new(-cam_x, 0.0, 0.0),
            );
            let camera = recon.camera(1).clone();
            let xy = camera
                .project(&cam_from_world.transform_point(&p1))
                .expect("synthetic point must be visible");

            let mut image = Image::new(frame_id, 1, frame_id, format!("f{frame_id}.png"));
            image.points2d = vec![Point2D::new(xy)];
            recon.add_image(image);

            recon.frame_mut(frame_id).set_rig_from_world(cam_from_world);
            recon.register_frame(frame_id);
        }

        for frame_id in 1..=NUM_CORRESPONDENTS {
            graph
                .add_two_view_geometry(
                    0,
                    frame_id as usize,
                    &[(0, 0)],
                    ConfigurationType::Calibrated,
                )
                .unwrap();
        }
        graph.finalize();

        let mut obs = ObservationManager::new(&recon, &graph);
        obs.size_pyramids(&recon);
        let mut tri = IncrementalTriangulator::new();
        let options = Options::default();

        let start = std::time::Instant::now();
        let num_tris = tri.triangulate_image(&options, &mut recon, &graph, &mut obs, 0);
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_secs() < 5,
            "triangulate_image on a 200-correspondence pool took {elapsed:?}, expected well under 5s"
        );
        assert!(
            num_tris >= 2,
            "expected at least the seed pair to triangulate"
        );
        assert!(
            recon.num_points3d() >= 1,
            "expected at least one 3D point created from the consistent pool"
        );
        for point3d in recon.points3d().values() {
            assert!(
                (point3d.xyz - p1).norm() < 1e-3,
                "triangulated point {:?} should match the single true point P1",
                point3d.xyz
            );
        }
    }

    /// Regression test for the `Continue`-then-`Create` staleness bug found
    /// running this port on real tier-1000 data (see `triangulate_image`'s
    /// doc comment): image A's ray (through one fixed pixel) is set up to
    /// pass exactly through two different true points, `p_near` (already
    /// triangulated, observed by `B`) and `p_far` (not yet triangulated,
    /// observed by `C`). `Find(A,0)` returns both `B` (triangulated) and
    /// `C` (not), so `TriangulateImage` takes the `Continue`-then-`Create`
    /// branch: `Continue` links `A`'s point2D to `p_near` via `B`; before
    /// the fix, `Create` was then handed a *stale* snapshot of `A`'s
    /// correspondence data (still reporting `has_point3d=false`) and could
    /// re-triangulate `A`+`C` into a *second* point (`p_far`), silently
    /// overwriting `A`'s `point2D::point3D_id` while leaving `A`'s track
    /// element dangling in `p_near`'s track — an invariant violation that
    /// surfaced downstream as a `delete_observation on a point2D without a
    /// point3D` panic. This test asserts the invariant directly: after
    /// triangulation, `A`'s point2D belongs to **at most one** point3D's
    /// track, and that track is consistent with `A`'s own
    /// `point2D::point3D_id`.
    #[test]
    fn triangulate_image_does_not_double_link_ref_point_after_continue() {
        use crate::colmap_incremental::observation_manager::ObservationManager;
        use crate::colmap_incremental::reconstruction::{
            Camera, Image, Point2D, Reconstruction, TrackElement,
        };
        use crate::colmap_incremental::types::{DataT, Frame, Rig, SensorT};
        use nalgebra::Vector3;
        use visloc_vision::two_view::ConfigurationType;

        let mut recon = Reconstruction::new();
        let mut rig = Rig::new();
        rig.set_rig_id(0);
        rig.add_ref_sensor(SensorT::camera(1));
        recon.add_rig(rig);
        recon.add_camera(Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0));
        let camera = recon.camera(1).clone();

        // A's ray direction; p_near and p_far both lie on it, so they
        // project to the *same* pixel in A.
        let bearing = Vector3::new(0.1, 0.03, 1.0);
        let p_near = Point3::from(bearing * 3.0);
        let p_far = Point3::from(bearing * 5.0);

        let add_frame = |recon: &mut Reconstruction, frame_id: u64, cam_pos: Vector3<f64>| -> SE3 {
            let mut frame = Frame::new(frame_id, 0);
            frame.add_data_id(DataT::camera(1, frame_id));
            recon.add_frame(frame);
            let cam_from_world = SE3::new(nalgebra::UnitQuaternion::identity(), -cam_pos);
            recon
                .frame_mut(frame_id)
                .set_rig_from_world(cam_from_world.clone());
            recon.register_frame(frame_id);
            cam_from_world
        };

        // A: at the origin.
        let cam_from_world_a = add_frame(&mut recon, 0, Vector3::zeros());
        let xy_a = camera
            .project(&cam_from_world_a.transform_point(&p_near))
            .unwrap();
        assert!(
            (xy_a
                - camera
                    .project(&cam_from_world_a.transform_point(&p_far))
                    .unwrap())
            .norm()
                < 1e-9,
            "p_near/p_far must project to the same pixel in A by construction"
        );
        let mut image_a = Image::new(0, 1, 0, "a.png".to_owned());
        image_a.points2d = vec![Point2D::new(xy_a)];
        recon.add_image(image_a);

        // B: genuinely observes p_near (real parallax).
        let cam_from_world_b = add_frame(&mut recon, 1, Vector3::new(0.3, 0.0, 0.0));
        let xy_b = camera
            .project(&cam_from_world_b.transform_point(&p_near))
            .unwrap();
        let mut image_b = Image::new(1, 1, 1, "b.png".to_owned());
        image_b.points2d = vec![Point2D::new(xy_b)];
        recon.add_image(image_b);

        // C: genuinely observes p_far (real parallax, not yet triangulated).
        let cam_from_world_c = add_frame(&mut recon, 2, Vector3::new(-0.4, 0.0, 0.0));
        let xy_c = camera
            .project(&cam_from_world_c.transform_point(&p_far))
            .unwrap();
        let mut image_c = Image::new(2, 1, 2, "c.png".to_owned());
        image_c.points2d = vec![Point2D::new(xy_c)];
        recon.add_image(image_c);

        let mut graph = CorrespondenceGraph::new();
        for image_id in 0..3usize {
            graph.add_image(image_id, 1);
        }
        graph
            .add_two_view_geometry(0, 1, &[(0, 0)], ConfigurationType::Calibrated)
            .unwrap();
        graph
            .add_two_view_geometry(0, 2, &[(0, 0)], ConfigurationType::Calibrated)
            .unwrap();
        graph.finalize();

        let mut obs = ObservationManager::new(&recon, &graph);
        obs.size_pyramids(&recon);
        // Seed B's point3D (p_near) as already-triangulated before A runs.
        obs.add_point3d(
            &mut recon,
            &graph,
            p_near,
            vec![TrackElement {
                image_id: 1,
                point2d_idx: 0,
            }],
        );

        let mut tri = IncrementalTriangulator::new();
        let options = Options::default();
        tri.triangulate_image(&options, &mut recon, &graph, &mut obs, 0);

        // Invariant: A's point2D belongs to at most one point3D's track,
        // consistent with its own `point2D::point3D_id`.
        let a_point3d_id = recon.image(0).points2d[0].point3d_id;
        for (&pid, point3d) in recon.points3d() {
            let contains_a = point3d
                .track
                .iter()
                .any(|el| el.image_id == 0 && el.point2d_idx == 0);
            if contains_a {
                assert_eq!(
                    Some(pid),
                    a_point3d_id,
                    "point3D {pid} contains A's track element but A's point2D::point3D_id is {a_point3d_id:?} (dangling/double-linked track)"
                );
            }
        }
    }
}

/// Linear (homogeneous DLT) multi-view triangulation from `>=2` views.
/// Standard algorithm (e.g. Hartley & Zisserman §12.2): stack two rows per
/// view (`x * P_row2 - P_row0`, `y * P_row2 - P_row1`) using the
/// *normalized* (undistorted, calibrated) ray so it applies uniformly
/// across cameras with different intrinsics, and take the right singular
/// vector of smallest singular value.
pub(super) fn triangulate_dlt(views: &[(&SE3, Point2<f64>, &Camera)]) -> Option<Point3<f64>> {
    let n = views.len();
    if n < 2 {
        return None;
    }
    let mut a = DMatrix::<f64>::zeros(2 * n, 4);
    for (row, (cam_from_world, xy, camera)) in views.iter().enumerate() {
        let normalized = camera.normalize_pixel(xy)?;
        let r = cam_from_world.rotation.to_rotation_matrix();
        let t = cam_from_world.translation;
        // Projection rows in the normalized (unit-focal, zero-principal-point)
        // camera frame: P = [R | t].
        let p0 = [r[(0, 0)], r[(0, 1)], r[(0, 2)], t.x];
        let p1 = [r[(1, 0)], r[(1, 1)], r[(1, 2)], t.y];
        let p2 = [r[(2, 0)], r[(2, 1)], r[(2, 2)], t.z];
        for c in 0..4 {
            a[(2 * row, c)] = normalized.x * p2[c] - p0[c];
            a[(2 * row + 1, c)] = normalized.y * p2[c] - p1[c];
        }
    }
    let svd = a.svd(true, true);
    let v_t = svd.v_t?;
    let last = v_t.row(v_t.nrows() - 1);
    let w = last[3];
    if w.abs() < 1e-12 {
        return None;
    }
    Some(Point3::new(last[0] / w, last[1] / w, last[2] / w))
}
