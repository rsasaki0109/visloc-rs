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
//! ## `TriangulateTrack` / `EstimateTriangulation`
//!
//! [`estimate_triangulation`] is a faithful port of `TriangulateTrack`
//! (`.cc:39-66`) and `estimators/triangulation.cc`'s `EstimateTriangulation`:
//! a `LORANSAC<TriangulationEstimator, TriangulationEstimator,
//! InlierSupportMeasurer, CombinationSampler>` with COLMAP's
//! `EstimateTriangulationOptions` defaults (`min_angle`, angular vs
//! reprojection residual, `max_error = 2°` / `complete_max_reproj_error`,
//! `confidence = 0.9999`, `min_inlier_ratio = 0.02`, `max_num_trials =
//! 10000`, `dyn_num_trials_multiplier = 3.0`). `TriangulateTrack` forces
//! `min_num_trials = C(n,2)` for tracks of at most
//! [`EXHAUSTIVE_SAMPLING_THRESHOLD`] views, which the size-2
//! `CombinationSampler` turns into an exhaustive, lexicographic enumeration
//! of every view pair. Local optimization refits the model on the current
//! inlier set (`TriangulationEstimator::Estimate` on the inlier views) and
//! keeps expanding while the inlier support strictly improves.

use std::collections::{BTreeMap, BTreeSet, HashSet};

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
    /// Points deleted / created by merges since the last drain (lets
    /// `merge_tracks` invalidate speculative plans without rescanning).
    merge_touched: Vec<Point3DT>,
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
    ///
    /// Speculatively parallel, with the serial loop's exact result: in waves,
    /// every point's completion is first planned against the current state
    /// on a rayon worker ([`plan_complete`] records each 2D point it looked
    /// at and the observations it would add), then the plans are applied
    /// serially in `point3d_ids` order. A point's BFS depends only on the
    /// state of the 2D points it examines (registration, poses, cameras and
    /// its own position are fixed during the pass), so a plan stays valid
    /// unless one of those 2D points was claimed earlier in the pass; such a
    /// point falls back to the serial [`Self::complete`].
    pub fn complete_tracks(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
        point3d_ids: &[Point3DT],
    ) -> usize {
        use rayon::prelude::*;
        self.clear_caches();
        let mut n = 0;
        let mut claimed: HashSet<(ImageT, Point2DT)> = HashSet::new();
        for wave in point3d_ids.chunks(PLAN_WAVE) {
            let recon_ref: &Reconstruction = recon;
            let plans: Vec<Option<CompletePlan>> = wave
                .par_iter()
                .map(|&id| plan_complete(options, recon_ref, graph, id))
                .collect();
            for (&id, plan) in wave.iter().zip(plans) {
                let Some(plan) = plan else { continue };
                if plan.examined.iter().any(|key| claimed.contains(key)) {
                    let before = recon.point3d(id).track.len();
                    n += self.complete(options, recon, graph, obs, id);
                    for el in &recon.point3d(id).track[before..] {
                        claimed.insert((el.image_id, el.point2d_idx));
                    }
                    continue;
                }
                for el in plan.adds {
                    obs.add_observation(recon, graph, id, el);
                    self.modified_point3d_ids.insert(id);
                    claimed.insert((el.image_id, el.point2d_idx));
                    n += 1;
                }
            }
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
    ///
    /// Speculatively parallel, with the serial loop's exact result: each
    /// point's first merge attempt is planned on a rayon worker against the
    /// current state ([`plan_merge`]: the candidate pairs it would test and
    /// whether any passes). Applying in `point3d_ids` order, a plan that found
    /// no merge is final as long as neither the point nor any candidate it
    /// tested was merged away earlier in the pass (the merge test of an
    /// unchanged pair is deterministic and symmetric); its tested pairs are
    /// recorded in `merge_trials` as the serial loop would. Every other point
    /// runs the serial [`Self::merge`].
    pub fn merge_tracks(
        &mut self,
        options: &Options,
        recon: &mut Reconstruction,
        graph: &CorrespondenceGraph,
        obs: &mut ObservationManager,
        point3d_ids: &[Point3DT],
    ) -> usize {
        use rayon::prelude::*;
        self.clear_caches();
        let mut n = 0;
        // Points created or deleted by merges in this pass.
        let mut touched: HashSet<Point3DT> = HashSet::new();
        for wave in point3d_ids.chunks(PLAN_WAVE) {
            let recon_ref: &Reconstruction = recon;
            let plans: Vec<Option<MergePlan>> = wave
                .par_iter()
                .map(|&id| plan_merge(options, recon_ref, graph, id))
                .collect();
            for (&id, plan) in wave.iter().zip(plans) {
                let stale = match &plan {
                    None => false, // point did not exist when planned (cannot reappear)
                    Some(p) => {
                        p.merges
                            || touched.contains(&id)
                            || p.candidates.iter().any(|c| touched.contains(c))
                    }
                };
                if !stale {
                    if let Some(p) = plan {
                        for &c in &p.candidates {
                            self.merge_trials
                                .insert(if id < c { (id, c) } else { (c, id) });
                        }
                    }
                    continue;
                }
                n += self.merge(options, recon, graph, obs, id);
                touched.extend(self.merge_touched.drain(..));
            }
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
                    self.merge_touched
                        .extend([point3d_id, corr_point3d_id, merged_id]);
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

/// Points per speculative planning wave of [`IncrementalTriangulator::complete_tracks`] /
/// [`IncrementalTriangulator::merge_tracks`] (bounds plan memory; the result
/// does not depend on it).
const PLAN_WAVE: usize = 16_384;

/// Read-only replay of [`IncrementalTriangulator::complete`]'s BFS.
struct CompletePlan {
    /// Every 2D point the BFS looked at beyond the point's own track.
    examined: Vec<(ImageT, Point2DT)>,
    /// Observations it would add, in the serial order.
    adds: Vec<TrackElement>,
}

fn plan_complete(
    options: &Options,
    recon: &Reconstruction,
    graph: &CorrespondenceGraph,
    point3d_id: Point3DT,
) -> Option<CompletePlan> {
    if !recon.exists_point3d(point3d_id) {
        return None;
    }
    let max_sq_reproj_error = options.complete_max_reproj_error * options.complete_max_reproj_error;
    let xyz = recon.point3d(point3d_id).xyz;
    let mut curr_queue: Vec<TrackElement> = recon.point3d(point3d_id).track.clone();
    let mut next_queue: Vec<TrackElement> = Vec::new();
    let mut visited: BTreeSet<(ImageT, Point2DT)> = curr_queue
        .iter()
        .map(|el| (el.image_id, el.point2d_idx))
        .collect();
    // 2D points this plan itself claims (the serial BFS sees them as taken
    // once added).
    let mut own: HashSet<(ImageT, Point2DT)> = HashSet::new();
    let mut plan = CompletePlan {
        examined: Vec::new(),
        adds: Vec::new(),
    };
    let max_transitivity = options.complete_max_transitivity;
    for transitivity in 1..=max_transitivity {
        while let Some(elem) = curr_queue.pop() {
            for corr in graph.find_correspondences(elem.image_id as usize, elem.point2d_idx) {
                let key = (corr.image_id as ImageT, corr.point2d_idx);
                if !visited.insert(key) {
                    continue;
                }
                plan.examined.push(key);
                let (corr_image_id, corr_point2d_idx) = key;
                if !recon.is_image_registered(corr_image_id) {
                    continue;
                }
                let image = recon.image(corr_image_id);
                if image.points2d[corr_point2d_idx].has_point3d() || own.contains(&key) {
                    continue;
                }
                let camera = recon.camera(image.camera_id);
                if camera_has_bogus_params(
                    camera,
                    options.min_focal_length_ratio,
                    options.max_focal_length_ratio,
                    options.max_extra_param,
                ) {
                    continue;
                }
                let xy = image.points2d[corr_point2d_idx].xy;
                let cam_from_world = image_cam_from_world(recon, corr_image_id);
                let err = calculate_squared_reprojection_error(xy, xyz, &cam_from_world, camera);
                if err > max_sq_reproj_error {
                    continue;
                }
                let el = TrackElement {
                    image_id: corr_image_id,
                    point2d_idx: corr_point2d_idx,
                };
                own.insert(key);
                plan.adds.push(el);
                if transitivity < max_transitivity {
                    next_queue.push(el);
                }
            }
        }
        if next_queue.is_empty() {
            break;
        }
        std::mem::swap(&mut curr_queue, &mut next_queue);
    }
    Some(plan)
}

/// Read-only replay of one [`IncrementalTriangulator::try_merge_once`] call
/// with an empty `merge_trials`.
struct MergePlan {
    /// Distinct candidate points it would test, in order.
    candidates: Vec<Point3DT>,
    /// Whether one of them passes the merge test (the serial path then runs).
    merges: bool,
}

fn plan_merge(
    options: &Options,
    recon: &Reconstruction,
    graph: &CorrespondenceGraph,
    point3d_id: Point3DT,
) -> Option<MergePlan> {
    if !recon.exists_point3d(point3d_id) {
        return None;
    }
    let max_sq_reproj_error = options.merge_max_reproj_error * options.merge_max_reproj_error;
    let p3d = recon.point3d(point3d_id);
    let mut seen: HashSet<Point3DT> = HashSet::new();
    let mut plan = MergePlan {
        candidates: Vec::new(),
        merges: false,
    };
    for el in &p3d.track {
        for corr in graph.find_correspondences(el.image_id as usize, el.point2d_idx) {
            let corr_image_id = corr.image_id as ImageT;
            if !recon.is_image_registered(corr_image_id) {
                continue;
            }
            let Some(corr_point3d_id) =
                recon.image(corr_image_id).points2d[corr.point2d_idx].point3d_id
            else {
                continue;
            };
            if corr_point3d_id == point3d_id || !seen.insert(corr_point3d_id) {
                continue;
            }
            plan.candidates.push(corr_point3d_id);
            let corr_p3d = recon.point3d(corr_point3d_id);
            let n1 = p3d.track.len() as f64;
            let n2 = corr_p3d.track.len() as f64;
            let merged_xyz =
                Point3::from((p3d.xyz.coords * n1 + corr_p3d.xyz.coords * n2) / (n1 + n2));
            let merge_ok = [&p3d.track, &corr_p3d.track]
                .into_iter()
                .all(|check_track| {
                    check_track.iter().all(|test_el| {
                        let test_image = recon.image(test_el.image_id);
                        let test_camera = recon.camera(test_image.camera_id);
                        let test_xy = test_image.points2d[test_el.point2d_idx].xy;
                        let test_cam_from_world = image_cam_from_world(recon, test_el.image_id);
                        calculate_squared_reprojection_error(
                            test_xy,
                            merged_xyz,
                            &test_cam_from_world,
                            test_camera,
                        ) <= max_sq_reproj_error
                    })
                });
            if merge_ok {
                plan.merges = true;
                return Some(plan);
            }
        }
    }
    Some(plan)
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

/// `TriangulationEstimator::kMinNumSamples` (`estimators/triangulation.h`).
const TRIANGULATION_MIN_NUM_SAMPLES: usize = 2;
/// `EstimateTriangulationOptions` defaults (`estimators/triangulation.h:100-110`).
const TRIANGULATION_CONFIDENCE: f64 = 0.9999;
const TRIANGULATION_MIN_INLIER_RATIO: f64 = 0.02;
const TRIANGULATION_MAX_NUM_TRIALS: usize = 10_000;
/// `RANSACOptions::dyn_num_trials_multiplier` default (`optim/ransac.h:65`).
const RANSAC_DYN_NUM_TRIALS_MULTIPLIER: f64 = 3.0;
/// `TriangulateTrack`'s `kExhaustiveSamplingThreshold`
/// (`sfm/incremental_triangulator.cc:59`).
const EXHAUSTIVE_SAMPLING_THRESHOLD: usize = 15;
/// `LORANSAC`'s `kMaxNumLocalTrials` (`optim/loransac.h:261`).
const MAX_NUM_LOCAL_TRIALS: usize = 10;

/// `InlierSupportMeasurer::Support` (`optim/support_measurement.h:42-48`).
#[derive(Clone, Copy)]
struct RansacSupport {
    num_inliers: usize,
    residual_sum: f64,
}

impl RansacSupport {
    const fn empty() -> Self {
        Self {
            num_inliers: 0,
            residual_sum: f64::MAX,
        }
    }

    /// `InlierSupportMeasurer::IsLeftBetter` (`support_measurement.cc`):
    /// more inliers wins; ties are broken by the lower residual sum.
    fn is_left_better(&self, other: &Self) -> bool {
        self.num_inliers > other.num_inliers
            || (self.num_inliers == other.num_inliers && self.residual_sum < other.residual_sum)
    }
}

fn support_from_residuals(residuals: &[f64], max_residual: f64) -> RansacSupport {
    let mut support = RansacSupport {
        num_inliers: 0,
        residual_sum: 0.0,
    };
    for &residual in residuals {
        if residual <= max_residual {
            support.num_inliers += 1;
            support.residual_sum += residual;
        }
    }
    support
}

/// `NChooseK(n, 2)` (`math/math.h`).
const fn n_choose_2(n: usize) -> usize {
    n * (n - 1) / 2
}

/// Port of `RANSAC::ComputeNumTrials` (`optim/ransac.h:178-210`) with
/// `kMinNumSamples = 2`: the trials needed for at least one outlier-free
/// minimal sample with the given confidence.
fn compute_num_trials(
    num_inliers: usize,
    num_samples: usize,
    confidence: f64,
    num_trials_multiplier: f64,
) -> usize {
    let prob_failure = 1.0 - confidence;
    if prob_failure <= 0.0 {
        return usize::MAX;
    }
    let mut prob_inlier = 1.0;
    for i in 0..TRIANGULATION_MIN_NUM_SAMPLES {
        if num_inliers < i || num_samples < i {
            return usize::MAX;
        }
        prob_inlier *= (num_inliers - i) as f64 / (num_samples - i) as f64;
    }
    let prob_outlier = 1.0 - prob_inlier;
    if prob_outlier <= 0.0 {
        return 1;
    }
    if prob_outlier >= 1.0 {
        return usize::MAX;
    }
    ((prob_failure.ln() / prob_outlier.ln()) * num_trials_multiplier).ceil() as usize
}

/// Squared residual of one observation against `xyz`, in the units COLMAP's
/// `TriangulationEstimator::Residuals` uses: squared angular error (radians)
/// for [`ResidualType::Angular`], squared reprojection error (pixels) for
/// [`ResidualType::Reprojection`].
fn triangulation_residual_sq(
    residual_type: ResidualType,
    xyz: Point3<f64>,
    corr: &CorrData,
) -> f64 {
    match residual_type {
        ResidualType::Angular => {
            let error =
                angular_reprojection_error(corr.xy, xyz, &corr.cam_from_world, &corr.camera);
            error * error
        }
        ResidualType::Reprojection => {
            calculate_squared_reprojection_error(corr.xy, xyz, &corr.cam_from_world, &corr.camera)
        }
    }
}

/// `TriangulationEstimator::Estimate` on a two-view sample
/// (`estimators/triangulation.cc`): two-view DLT, cheirality for both views,
/// and a seed-pair triangulation angle `>= min_tri_angle`.
fn estimate_minimal_triangulation(
    first: &CorrData,
    second: &CorrData,
    min_tri_angle_rad: f64,
) -> Option<Point3<f64>> {
    let xyz = triangulate_dlt(&[
        (&first.cam_from_world, first.xy, &first.camera),
        (&second.cam_from_world, second.xy, &second.camera),
    ])?;
    if !positive_depth(&first.cam_from_world, xyz) || !positive_depth(&second.cam_from_world, xyz) {
        return None;
    }
    let angle = calculate_triangulation_angle(
        cam_center(&first.cam_from_world),
        cam_center(&second.cam_from_world),
        xyz,
    );
    (angle >= min_tri_angle_rad).then_some(xyz)
}

/// `TriangulationEstimator::Estimate` on the local-optimization inlier set
/// (`estimators/triangulation.cc`): multi-view DLT, cheirality for every view,
/// and at least one view pair with angle `>= min_tri_angle`.
fn estimate_multiview_triangulation(
    corrs: &[CorrData],
    inlier_indices: &[usize],
    min_tri_angle_rad: f64,
) -> Option<Point3<f64>> {
    let views: Vec<(&SE3, Point2<f64>, &Camera)> = inlier_indices
        .iter()
        .map(|&i| (&corrs[i].cam_from_world, corrs[i].xy, &corrs[i].camera))
        .collect();
    let xyz = triangulate_dlt(&views)?;
    for &i in inlier_indices {
        if !positive_depth(&corrs[i].cam_from_world, xyz) {
            return None;
        }
    }
    for (position, &i) in inlier_indices.iter().enumerate() {
        for &j in &inlier_indices[..position] {
            let angle = calculate_triangulation_angle(
                cam_center(&corrs[i].cam_from_world),
                cam_center(&corrs[j].cam_from_world),
                xyz,
            );
            if angle >= min_tri_angle_rad {
                return Some(xyz);
            }
        }
    }
    None
}

/// Port of `LORANSAC`'s recursive local optimization (`optim/loransac.h:257-319`):
/// refit the model on the current inlier set, rescan the support, and keep
/// expanding while the support strictly improves (at most
/// [`MAX_NUM_LOCAL_TRIALS`] iterations). `COLMAP`'s local and global
/// estimators are the same class, so the winning model's inlier mask is
/// recomputed identically either way.
fn local_optimize_triangulation(
    corrs: &[CorrData],
    residual_type: ResidualType,
    sample_residuals: &[f64],
    sample_model: Point3<f64>,
    max_residual: f64,
    min_tri_angle_rad: f64,
) -> (RansacSupport, Point3<f64>) {
    let mut local_best_support = support_from_residuals(sample_residuals, max_residual);
    let mut local_best_model = sample_model;
    let mut current_residuals = sample_residuals.to_vec();
    if local_best_support.num_inliers > TRIANGULATION_MIN_NUM_SAMPLES {
        for _ in 0..MAX_NUM_LOCAL_TRIALS {
            let inlier_indices: Vec<usize> = current_residuals
                .iter()
                .enumerate()
                .filter(|(_, &residual)| residual <= max_residual)
                .map(|(index, _)| index)
                .collect();
            let Some(local_model) =
                estimate_multiview_triangulation(corrs, &inlier_indices, min_tri_angle_rad)
            else {
                break;
            };
            let local_residuals: Vec<f64> = corrs
                .iter()
                .map(|corr| triangulation_residual_sq(residual_type, local_model, corr))
                .collect();
            let local_support = support_from_residuals(&local_residuals, max_residual);
            if local_support.is_left_better(&local_best_support) {
                local_best_support = local_support;
                local_best_model = local_model;
                current_residuals = local_residuals;
            } else {
                break;
            }
        }
    }
    (local_best_support, local_best_model)
}

/// Faithful port of `TriangulateTrack` (`sfm/incremental_triangulator.cc:39-66`)
/// and `EstimateTriangulation` (`estimators/triangulation.cc`):
/// `LORANSAC<TriangulationEstimator, TriangulationEstimator,
/// InlierSupportMeasurer, CombinationSampler>`. `max_error` is the
/// *unsquared* threshold in radians for [`ResidualType::Angular`] and in
/// pixels for [`ResidualType::Reprojection`]; residuals are compared against
/// `max_error^2` (`optim/ransac.h:146`). Returns the best model and its
/// inlier mask, or `None` if fewer than two inliers support any model.
fn estimate_triangulation(
    corrs: &[CorrData],
    residual_type: ResidualType,
    max_error: f64,
    min_tri_angle_rad: f64,
) -> Option<(Point3<f64>, Vec<bool>)> {
    let n = corrs.len();
    if n < TRIANGULATION_MIN_NUM_SAMPLES {
        return None;
    }
    let max_residual = max_error * max_error;

    // `RANSAC`'s constructor clamps the requested trial budget by the count
    // implied by the a-priori `min_inlier_ratio` (`optim/ransac.h:167-176`);
    // `LORANSAC` then clamps it by the sampler's maximum number of samples
    // (`optim/loransac.h:150-151`), which for the size-2 `CombinationSampler`
    // is `C(n, 2)`.
    let ctor_max_num_trials = compute_num_trials(
        (TRIANGULATION_MIN_INLIER_RATIO * 100_000.0) as usize,
        100_000,
        TRIANGULATION_CONFIDENCE,
        RANSAC_DYN_NUM_TRIALS_MULTIPLIER,
    );
    let max_num_trials = TRIANGULATION_MAX_NUM_TRIALS
        .min(ctor_max_num_trials)
        .min(n_choose_2(n));
    // `TriangulateTrack` forces exhaustive sampling for short tracks.
    let min_num_trials = if n <= EXHAUSTIVE_SAMPLING_THRESHOLD {
        n_choose_2(n)
    } else {
        0
    };

    let mut best = RansacSupport::empty();
    let mut best_model: Option<Point3<f64>> = None;
    let mut dyn_max_num_trials = max_num_trials;

    // `CombinationSampler` enumerates the size-2 combinations of `0..n` in
    // lexicographic order starting at `(0, 1)` (`optim/combination_sampler.cc`).
    let mut sample = (0usize, 1usize);
    let mut curr = 0usize;
    while curr < max_num_trials {
        if let Some(model) =
            estimate_minimal_triangulation(&corrs[sample.0], &corrs[sample.1], min_tri_angle_rad)
        {
            let residuals: Vec<f64> = corrs
                .iter()
                .map(|corr| triangulation_residual_sq(residual_type, model, corr))
                .collect();
            let support = support_from_residuals(&residuals, max_residual);
            if support.is_left_better(&best) {
                let (local_support, local_model) = local_optimize_triangulation(
                    corrs,
                    residual_type,
                    &residuals,
                    model,
                    max_residual,
                    min_tri_angle_rad,
                );
                if local_support.is_left_better(&best) {
                    best = local_support;
                    best_model = Some(local_model);
                    dyn_max_num_trials = compute_num_trials(
                        best.num_inliers,
                        n,
                        TRIANGULATION_CONFIDENCE,
                        RANSAC_DYN_NUM_TRIALS_MULTIPLIER,
                    );
                }
            }
        }
        if curr >= dyn_max_num_trials && curr >= min_num_trials {
            break;
        }
        if sample.1 + 1 < n {
            sample.1 += 1;
        } else {
            sample.0 += 1;
            sample.1 = sample.0 + 1;
        }
        curr += 1;
    }

    let model = best_model?;
    if best.num_inliers < TRIANGULATION_MIN_NUM_SAMPLES {
        return None;
    }
    let inlier_mask = corrs
        .iter()
        .map(|corr| triangulation_residual_sq(residual_type, model, corr) <= max_residual)
        .collect();
    Some((model, inlier_mask))
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
