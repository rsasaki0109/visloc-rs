//! Transactional selective bundle adjustment across verified submap seams.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use visloc_core::geometry::{Pose, Sim3};

use crate::{
    BaConfig, BaError, BaObservation, BundleAdjustment, HierarchicalSubmapGraph, LinearSolver,
    RobustKernel,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HierarchicalSeamLandmarkLink {
    pub source_submap_id: u64,
    pub target_submap_id: u64,
    pub source_landmark_id: u64,
    pub target_landmark_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HierarchicalSeamBaConfig {
    pub ba: BaConfig,
    pub min_observations: usize,
    pub max_final_cost_ratio: f64,
    /// Maximum number of BA/filter rounds. Filtering is active only when this
    /// is greater than one, so a single round preserves the historical solve.
    pub max_rounds: usize,
    /// Remove observations whose post-BA reprojection error exceeds this
    /// threshold before the next round.
    pub max_reprojection_px: f64,
    /// Keep root-only prefix poses fixed but jointly refine root poses also
    /// reconstructed by another submap. At least two root poses remain fixed.
    pub optimize_shared_root_poses: bool,
    /// Hard iteration cap for the loop-welded second BA.
    pub loop_ba_max_iterations: usize,
    /// Stop the loop-welded second BA when one accepted iteration's relative
    /// cost decrease falls below this threshold.
    pub loop_ba_relative_cost_tolerance: f64,
    /// Reject the loop-welded second BA when its final mean robust cost per
    /// observation exceeds this multiple of the first seam BA's final mean.
    pub loop_ba_max_first_seam_mean_cost_ratio: f64,
}

impl Default for HierarchicalSeamBaConfig {
    fn default() -> Self {
        Self {
            ba: BaConfig {
                max_iterations: 5,
                linear_solver: LinearSolver::Sparse,
                robust_kernel: RobustKernel::Huber { delta: 3.0 },
                ..BaConfig::default()
            },
            min_observations: 100,
            max_final_cost_ratio: 1.0,
            max_rounds: 3,
            max_reprojection_px: 4.0,
            optimize_shared_root_poses: true,
            loop_ba_max_iterations: 120,
            loop_ba_relative_cost_tolerance: 1.0e-4,
            loop_ba_max_first_seam_mean_cost_ratio: 3.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalSeamBaResult {
    pub pose_count: usize,
    pub fixed_pose_count: usize,
    pub landmark_count: usize,
    pub observation_count: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    pub rounds: usize,
    pub filtered_observation_count: usize,
    pub dropped_landmark_count: usize,
    /// R2-verified local landmark identities merged into one BA variable.
    pub welded_landmark_groups: Vec<Vec<(u64, u64)>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HierarchicalSeamBaError {
    MissingAlignedTransform(u64),
    MissingSubmap(u64),
    TooFewObservations {
        required: usize,
        found: usize,
    },
    Optimize(BaError),
    CostWorsened {
        initial: f64,
        final_cost: f64,
    },
    LoopBaNonConverged {
        final_mean_cost: f64,
        first_seam_mean_cost: f64,
        max_ratio: f64,
    },
}

impl fmt::Display for HierarchicalSeamBaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingAlignedTransform(id) => write!(f, "submap {id} is not atlas-aligned"),
            Self::MissingSubmap(id) => write!(f, "seam link references missing submap {id}"),
            Self::TooFewObservations { required, found } => {
                write!(f, "seam BA has {found} observations; {required} required")
            }
            Self::Optimize(error) => write!(f, "seam BA failed: {error}"),
            Self::CostWorsened {
                initial,
                final_cost,
            } => write!(f, "seam BA cost worsened {initial} -> {final_cost}"),
            Self::LoopBaNonConverged {
                final_mean_cost,
                first_seam_mean_cost,
                max_ratio,
            } => write!(
                f,
                "loop-welded BA mean cost {final_mean_cost} exceeds {max_ratio}x \
                 first seam BA mean cost {first_seam_mean_cost}"
            ),
        }
    }
}

impl Error for HierarchicalSeamBaError {}

impl From<BaError> for HierarchicalSeamBaError {
    fn from(value: BaError) -> Self {
        Self::Optimize(value)
    }
}

pub(crate) fn refine_hierarchical_seams(
    hierarchy: &mut HierarchicalSubmapGraph,
    links: &[HierarchicalSeamLandmarkLink],
    config: &HierarchicalSeamBaConfig,
) -> Result<HierarchicalSeamBaResult, HierarchicalSeamBaError> {
    refine_hierarchical_seams_with_reference(hierarchy, links, config, false, None)
}

fn refine_hierarchical_seams_with_reference(
    hierarchy: &mut HierarchicalSubmapGraph,
    links: &[HierarchicalSeamLandmarkLink],
    config: &HierarchicalSeamBaConfig,
    retriangulate_welds: bool,
    first_seam_mean_cost: Option<f64>,
) -> Result<HierarchicalSeamBaResult, HierarchicalSeamBaError> {
    let nodes = hierarchy.nodes().collect::<Vec<_>>();
    let mut landmark_keys = Vec::<(u64, u64)>::new();
    let mut landmark_index = BTreeMap::<(u64, u64), usize>::new();
    for node in &nodes {
        if node.local_from_atlas.is_none() {
            return Err(HierarchicalSeamBaError::MissingAlignedTransform(node.id));
        }
        for landmark in &node.submap.landmarks {
            let key = (node.id, landmark.local_landmark_id);
            landmark_index.insert(key, landmark_keys.len());
            landmark_keys.push(key);
        }
    }
    let mut dsu = DisjointSet::new(landmark_keys.len());
    for link in links {
        let source = *landmark_index
            .get(&(link.source_submap_id, link.source_landmark_id))
            .ok_or(HierarchicalSeamBaError::MissingSubmap(
                link.source_submap_id,
            ))?;
        let target = *landmark_index
            .get(&(link.target_submap_id, link.target_landmark_id))
            .ok_or(HierarchicalSeamBaError::MissingSubmap(
                link.target_submap_id,
            ))?;
        dsu.union(source, target);
    }
    let mut global_id_by_root = BTreeMap::<usize, u64>::new();
    let mut global_landmark_id = BTreeMap::<(u64, u64), u64>::new();
    let mut members_by_root = BTreeMap::<usize, Vec<(u64, u64)>>::new();
    for (index, &key) in landmark_keys.iter().enumerate() {
        let root = dsu.find(index);
        members_by_root.entry(root).or_default().push(key);
        let next_id = global_id_by_root.len() as u64;
        let id = *global_id_by_root.entry(root).or_insert(next_id);
        global_landmark_id.insert(key, id);
    }
    let welded_landmark_groups = members_by_root
        .into_values()
        .filter(|members| members.len() > 1)
        .collect::<Vec<_>>();

    let mut pose_initial = BTreeMap::<u64, Pose>::new();
    let mut fixed_poses = BTreeSet::new();
    let mut frame_occurrences = BTreeMap::<u64, usize>::new();
    for node in &nodes {
        for frame in &node.submap.frames {
            *frame_occurrences.entry(frame.source_frame_id).or_default() += 1;
        }
    }
    let mut root_frame_ids = Vec::new();
    let mut point_initial = BTreeMap::<u64, nalgebra::Point3<f64>>::new();
    let mut observation_records = BTreeMap::<(u64, u64, usize), nalgebra::Point2<f64>>::new();
    for node in &nodes {
        let local_from_atlas = node.local_from_atlas.as_ref().unwrap();
        let atlas_from_local = local_from_atlas.inverse();
        for frame in &node.submap.frames {
            pose_initial
                .entry(frame.source_frame_id)
                .or_insert_with(|| local_pose_to_atlas(&frame.pose, local_from_atlas));
            if node.id == hierarchy.root_submap_id() {
                root_frame_ids.push(frame.source_frame_id);
                if !config.optimize_shared_root_poses
                    || frame_occurrences[&frame.source_frame_id] == 1
                {
                    fixed_poses.insert(frame.source_frame_id);
                }
            }
        }
        for landmark in &node.submap.landmarks {
            let global_id = global_landmark_id[&(node.id, landmark.local_landmark_id)];
            point_initial
                .entry(global_id)
                .or_insert_with(|| atlas_from_local.transform_point(&landmark.position));
            for observation in &landmark.observations {
                if pose_initial.contains_key(&observation.source_frame_id) {
                    observation_records
                        .entry((
                            observation.source_frame_id,
                            global_id,
                            observation.keypoint_index,
                        ))
                        .or_insert(observation.pixel);
                }
            }
        }
    }
    root_frame_ids.sort_unstable();
    root_frame_ids.dedup();
    if fixed_poses.len() < 2 {
        if let Some(&first) = root_frame_ids.first() {
            fixed_poses.insert(first);
        }
        if let Some(&last) = root_frame_ids.last() {
            fixed_poses.insert(last);
        }
    }
    // Landmarks seen only by fixed poses form independent structure-only
    // problems. Eliminating them cannot change any pose update, so omit them
    // from the selective seam solve exactly (not as an approximation).
    let active_landmarks = observation_records
        .keys()
        .filter_map(|(frame_id, landmark_id, _)| {
            (!fixed_poses.contains(frame_id)).then_some(*landmark_id)
        })
        .collect::<BTreeSet<_>>();
    point_initial.retain(|landmark_id, _| active_landmarks.contains(landmark_id));
    observation_records.retain(|(_, landmark_id, _), _| active_landmarks.contains(landmark_id));

    if observation_records.len() < config.min_observations {
        return Err(HierarchicalSeamBaError::TooFewObservations {
            required: config.min_observations,
            found: observation_records.len(),
        });
    }

    let camera = nodes[0].submap.camera.clone();
    let mut problem = BundleAdjustment::new(camera);
    for (id, pose) in pose_initial {
        problem.add_pose(id, pose);
        if fixed_poses.contains(&id) {
            problem.fix_pose(id);
        }
    }
    for (id, point) in point_initial {
        problem.add_landmark(id, point);
    }
    for ((keyframe_id, landmark_id, _), xy) in observation_records {
        problem.add_observation(BaObservation {
            keyframe_id,
            landmark_id,
            xy,
        });
    }
    if retriangulate_welds {
        retriangulate_welded_landmarks(
            &mut problem,
            welded_landmark_groups
                .iter()
                .filter_map(|members| members.first().map(|member| global_landmark_id[member])),
        );
    }
    let original_observation_count = problem.observations.len();
    let refinement = optimize_filter_rounds(&mut problem, config)?;
    let observation_set_changed = problem.observations.len() != original_observation_count;
    let (gate_initial_cost, gate_final_cost, gate_metric) = if observation_set_changed {
        (
            refinement.initial_cost / original_observation_count as f64,
            refinement.final_cost / problem.observations.len() as f64,
            "mean_per_observation",
        )
    } else {
        (refinement.initial_cost, refinement.final_cost, "sum")
    };
    let cost_worsened = !gate_final_cost.is_finite()
        || gate_final_cost > gate_initial_cost * config.max_final_cost_ratio;
    eprintln!(
        "hierarchical-seam-ba: initial_cost={:.6} final_cost={:.6} iterations={} \
         rounds={} cost_metric={} gate_initial={:.6} gate_final={:.6}{}",
        refinement.initial_cost,
        refinement.final_cost,
        refinement.iterations,
        refinement.rounds,
        gate_metric,
        gate_initial_cost,
        gate_final_cost,
        if cost_worsened {
            " status=CostWorsened"
        } else {
            ""
        }
    );
    if cost_worsened {
        return Err(HierarchicalSeamBaError::CostWorsened {
            initial: gate_initial_cost,
            final_cost: gate_final_cost,
        });
    }
    if let Some(first_seam_mean_cost) = first_seam_mean_cost {
        let final_mean_cost = refinement.final_cost / problem.observations.len().max(1) as f64;
        let max_ratio = config.loop_ba_max_first_seam_mean_cost_ratio;
        if !final_mean_cost.is_finite()
            || !first_seam_mean_cost.is_finite()
            || first_seam_mean_cost < 0.0
            || !max_ratio.is_finite()
            || max_ratio < 0.0
            || final_mean_cost > first_seam_mean_cost * max_ratio
        {
            return Err(HierarchicalSeamBaError::LoopBaNonConverged {
                final_mean_cost,
                first_seam_mean_cost,
                max_ratio,
            });
        }
    }

    drop(nodes);
    for node in hierarchy.nodes_mut() {
        let local_from_atlas = node
            .local_from_atlas
            .as_ref()
            .ok_or(HierarchicalSeamBaError::MissingAlignedTransform(node.id))?
            .clone();
        for frame in &mut node.submap.frames {
            if let Some(atlas_pose) = problem.poses.get(&frame.source_frame_id) {
                frame.pose = atlas_pose_to_local(atlas_pose, &local_from_atlas);
            }
        }
        for landmark in &mut node.submap.landmarks {
            let global_id = global_landmark_id[&(node.id, landmark.local_landmark_id)];
            if let Some(atlas_point) = problem.landmarks.get(&global_id) {
                landmark.position = local_from_atlas.transform_point(atlas_point);
            }
        }
    }
    Ok(HierarchicalSeamBaResult {
        pose_count: problem.poses.len(),
        fixed_pose_count: problem.fixed_poses.len(),
        landmark_count: problem.landmarks.len(),
        observation_count: problem.observations.len(),
        initial_cost: refinement.initial_cost,
        final_cost: refinement.final_cost,
        iterations: refinement.iterations,
        rounds: refinement.rounds,
        filtered_observation_count: refinement.filtered_observations,
        dropped_landmark_count: refinement.dropped_landmarks,
        welded_landmark_groups,
    })
}

/// Run the post-loop BA transaction. A cost-gate rejection is an expected
/// guarded outcome: `refine_hierarchical_seams` has not written anything back
/// yet, so the loop-PGO hierarchy remains intact.
pub(crate) fn refine_hierarchical_loop_welds(
    enabled: bool,
    hierarchy: &mut HierarchicalSubmapGraph,
    links: &[HierarchicalSeamLandmarkLink],
    config: &HierarchicalSeamBaConfig,
    first_seam_mean_cost: Option<f64>,
) -> Result<Option<HierarchicalSeamBaResult>, HierarchicalSeamBaError> {
    if !enabled {
        eprintln!("hierarchical-loop-ba: disabled; retaining loop-PGO result");
        return Ok(None);
    }
    let mut loop_config = *config;
    loop_config.ba.max_iterations = config.loop_ba_max_iterations;
    loop_config.ba.relative_cost_tolerance = Some(config.loop_ba_relative_cost_tolerance);
    match refine_hierarchical_seams_with_reference(
        hierarchy,
        links,
        &loop_config,
        true,
        first_seam_mean_cost,
    ) {
        Ok(result) => {
            eprintln!(
                "hierarchical-loop-ba: accepted poses={} landmarks={} observations={} \
                 welded_groups={} initial_cost={:.6} final_cost={:.6}",
                result.pose_count,
                result.landmark_count,
                result.observation_count,
                result.welded_landmark_groups.len(),
                result.initial_cost,
                result.final_cost,
            );
            Ok(Some(result))
        }
        Err(HierarchicalSeamBaError::CostWorsened {
            initial,
            final_cost,
        }) => {
            eprintln!(
                "hierarchical-loop-ba: CostWorsened initial={initial:.6} \
                 final_cost={final_cost:.6}; retaining loop-PGO result"
            );
            Ok(None)
        }
        Err(HierarchicalSeamBaError::LoopBaNonConverged {
            final_mean_cost,
            first_seam_mean_cost,
            max_ratio,
        }) => {
            eprintln!(
                "hierarchical-loop-ba: non-converged, falling back to PGO result \
                 final_mean_cost={final_mean_cost:.6} \
                 first_seam_mean_cost={first_seam_mean_cost:.6} max_ratio={max_ratio:.3}"
            );
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn retriangulate_welded_landmarks(
    problem: &mut BundleAdjustment,
    welded_landmark_ids: impl IntoIterator<Item = u64>,
) {
    let pose_ids = problem.poses.keys().copied().collect::<Vec<_>>();
    let pose_index = pose_ids
        .iter()
        .enumerate()
        .map(|(index, &id)| (id, index))
        .collect::<BTreeMap<_, _>>();
    let poses = pose_ids
        .iter()
        .map(|id| problem.poses.get(id).cloned())
        .collect::<Vec<_>>();
    let mut observations = BTreeMap::<u64, Vec<(usize, nalgebra::Point2<f64>)>>::new();
    for observation in &problem.observations {
        if let Some(&index) = pose_index.get(&observation.keyframe_id) {
            observations
                .entry(observation.landmark_id)
                .or_default()
                .push((index, observation.xy));
        }
    }
    let triangulation_config = crate::IncrementalSfmConfig::default();
    let mut retriangulated = 0usize;
    let mut failed = 0usize;
    for landmark_id in welded_landmark_ids {
        let candidate = observations.get(&landmark_id).and_then(|track| {
            crate::incremental_sfm::triangulate_track(
                &problem.camera,
                &poses,
                track,
                &triangulation_config,
            )
        });
        match candidate {
            Some(point) => {
                if let Some(current) = problem.landmarks.get_mut(&landmark_id) {
                    *current = point;
                    retriangulated += 1;
                }
            }
            None => failed += 1,
        }
    }
    eprintln!(
        "hierarchical-loop-ba: welded-landmark-retriangulation succeeded={} failed={} \
         failed_kept_old_position={}",
        retriangulated, failed, failed
    );
}

const MIN_LANDMARK_OBSERVATIONS: usize = 2;
const EARLY_STOP_FILTERED_FRACTION: f64 = 0.001;

#[derive(Debug, Clone, Copy)]
struct FilterRefinementResult {
    initial_cost: f64,
    final_cost: f64,
    iterations: usize,
    rounds: usize,
    filtered_observations: usize,
    dropped_landmarks: usize,
}

fn optimize_filter_rounds(
    problem: &mut BundleAdjustment,
    config: &HierarchicalSeamBaConfig,
) -> Result<FilterRefinementResult, BaError> {
    let max_rounds = config.max_rounds.max(1);
    let filtering_enabled = max_rounds > 1
        && config.max_reprojection_px.is_finite()
        && config.max_reprojection_px > 0.0;
    let mut initial_cost = None;
    let mut final_cost = f64::NAN;
    let mut iterations = 0;
    let mut rounds = 0;
    let mut filtered_observations = 0;
    let mut dropped_landmarks = 0;

    for round in 0..max_rounds {
        let result = problem.optimize(&config.ba)?;
        initial_cost.get_or_insert(result.initial_cost);
        iterations += result.iterations.len();
        rounds += 1;

        let (round_filtered, round_dropped) = if filtering_enabled {
            filter_observations(problem, config.max_reprojection_px)
        } else {
            (0, 0)
        };
        filtered_observations += round_filtered;
        dropped_landmarks += round_dropped;
        final_cost = problem.robust_cost(&config.ba.robust_kernel);
        eprintln!(
            "hierarchical-seam-ba-round: round={} filtered_obs={} dropped_landmarks={} cost={:.6}",
            round + 1,
            round_filtered,
            round_dropped,
            final_cost
        );

        if !filtering_enabled {
            break;
        }
        let filtered_fraction =
            round_filtered as f64 / (problem.observations.len() + round_filtered).max(1) as f64;
        if filtered_fraction < EARLY_STOP_FILTERED_FRACTION {
            break;
        }
    }

    Ok(FilterRefinementResult {
        initial_cost: initial_cost.unwrap_or(f64::NAN),
        final_cost,
        iterations,
        rounds,
        filtered_observations,
        dropped_landmarks,
    })
}

fn filter_observations(problem: &mut BundleAdjustment, max_reprojection_px: f64) -> (usize, usize) {
    let max_squared = max_reprojection_px * max_reprojection_px;
    let before = problem.observations.len();
    let camera = &problem.camera;
    let poses = &problem.poses;
    let landmarks = &problem.landmarks;
    problem.observations.retain(|observation| {
        let Some(pose) = poses.get(&observation.keyframe_id) else {
            return false;
        };
        let Some(point) = landmarks.get(&observation.landmark_id) else {
            return false;
        };
        let Some(projected) = camera.project(&pose.transform_world_point(point)) else {
            return false;
        };
        (projected - observation.xy).norm_squared() <= max_squared
    });
    let mut observation_counts = BTreeMap::<u64, usize>::new();
    for observation in &problem.observations {
        *observation_counts
            .entry(observation.landmark_id)
            .or_default() += 1;
    }
    let dropped_ids = problem
        .landmarks
        .keys()
        .filter(|id| observation_counts.get(id).copied().unwrap_or(0) < MIN_LANDMARK_OBSERVATIONS)
        .copied()
        .collect::<BTreeSet<_>>();
    problem.landmarks.retain(|id, _| !dropped_ids.contains(id));
    problem
        .fixed_landmarks
        .retain(|id| !dropped_ids.contains(id));
    problem
        .observations
        .retain(|observation| !dropped_ids.contains(&observation.landmark_id));

    (before - problem.observations.len(), dropped_ids.len())
}

fn local_pose_to_atlas(local: &Pose, local_from_atlas: &Sim3) -> Pose {
    let rotation = local.world_to_camera.rotation * local_from_atlas.rotation;
    let centre_atlas = local_from_atlas
        .inverse()
        .transform_point(&local.camera_center_world());
    Pose::from_world_to_camera(rotation, -(rotation * centre_atlas.coords))
}

fn atlas_pose_to_local(atlas: &Pose, local_from_atlas: &Sim3) -> Pose {
    let rotation = atlas.world_to_camera.rotation * local_from_atlas.rotation.inverse();
    let centre_local = local_from_atlas.transform_point(&atlas.camera_center_world());
    Pose::from_world_to_camera(rotation, -(rotation * centre_local.coords))
}

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, index: usize) -> usize {
        if self.parent[index] != index {
            self.parent[index] = self.find(self.parent[index]);
        }
        self.parent[index]
    }

    fn union(&mut self, left: usize, right: usize) {
        let left = self.find(left);
        let right = self.find(right);
        if left == right {
            return;
        }
        if self.rank[left] < self.rank[right] {
            self.parent[left] = right;
        } else {
            self.parent[right] = left;
            if self.rank[left] == self.rank[right] {
                self.rank[left] += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, UnitQuaternion, Vector2, Vector3};
    use visloc_core::types::Camera;

    use crate::{
        LocalSubmap, LocalSubmapFrame, LocalSubmapLandmark, LocalSubmapObservation,
        LocalSubmapQuality, TrackBuildStats,
    };

    fn filtering_problem(with_outlier: bool) -> BundleAdjustment {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let true_point = Point3::new(0.15, -0.08, 4.0);
        let mut problem = BundleAdjustment::new(camera.clone());
        for index in 0..10_u64 {
            let centre = Point3::new(index as f64 * 0.12 - 0.54, 0.0, 0.0);
            let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -centre.coords);
            let mut pixel = camera
                .project(&pose.transform_world_point(&true_point))
                .unwrap();
            if with_outlier && index == 9 {
                pixel += nalgebra::Vector2::new(80.0, -60.0);
            }
            problem.add_pose(index, pose);
            problem.fix_pose(index);
            problem.add_observation(BaObservation {
                keyframe_id: index,
                landmark_id: 0,
                xy: pixel,
            });
        }
        problem.add_landmark(0, Point3::new(0.35, -0.2, 4.5));
        problem
    }

    fn test_config(max_rounds: usize) -> HierarchicalSeamBaConfig {
        HierarchicalSeamBaConfig {
            ba: BaConfig {
                max_iterations: 30,
                linear_solver: LinearSolver::Dense,
                robust_kernel: RobustKernel::Huber { delta: 3.0 },
                ..BaConfig::default()
            },
            min_observations: 0,
            max_final_cost_ratio: 1.0,
            max_rounds,
            max_reprojection_px: 4.0,
            optimize_shared_root_poses: true,
            loop_ba_max_iterations: 120,
            loop_ba_relative_cost_tolerance: 1.0e-4,
            loop_ba_max_first_seam_mean_cost_ratio: 3.0,
        }
    }

    fn loop_ba_submap(id: u64, with_landmarks: bool, pixel_noise: bool) -> LocalSubmap {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let centre = Point3::new(id as f64 * 0.1, 0.0, 0.0);
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -centre.coords);
        let landmarks: Vec<LocalSubmapLandmark> = with_landmarks
            .then(|| {
                (0..8_u64)
                    .map(|landmark_id| {
                        let point = Point3::new(
                            (landmark_id % 4) as f64 * 0.2 - 0.3,
                            (landmark_id / 4) as f64 * 0.2 - 0.1,
                            4.0 + landmark_id as f64 * 0.03,
                        );
                        let mut pixel =
                            camera.project(&pose.transform_world_point(&point)).unwrap();
                        if pixel_noise {
                            pixel += Vector2::new(
                                landmark_id as f64 * 0.2 + 0.1,
                                -(landmark_id as f64) * 0.1,
                            );
                        }
                        LocalSubmapLandmark {
                            local_landmark_id: landmark_id,
                            position: point,
                            observations: vec![LocalSubmapObservation {
                                local_frame_index: 0,
                                source_frame_id: id,
                                keypoint_index: landmark_id as usize,
                                pixel,
                            }],
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        LocalSubmap {
            camera,
            source_frame_ids: vec![id],
            frames: vec![LocalSubmapFrame {
                local_frame_index: 0,
                source_frame_id: id,
                pose,
            }],
            quality: LocalSubmapQuality {
                requested_images: 1,
                registered_images: 1,
                registration_fraction: 1.0,
                landmarks: landmarks.len(),
                observations: landmarks.len(),
                median_track_length: 1.0,
                median_max_parallax_deg: 0.0,
                camera_center_diameter: 0.0,
                camera_center_step_median: 0.0,
                camera_center_step_max: 0.0,
                seed_pair_final_distance: 1.0,
                camera_center_window_drift_ratio: 0.0,
                mean_reprojection_px: 0.0,
                leave_one_out_attempts: 0,
                leave_one_out_supported: 0,
                leave_one_out_support_fraction: 0.0,
                median_leave_one_out_reprojection_px: 0.0,
            },
            landmarks,
            track_build_stats: TrackBuildStats::default(),
            ba_result: None,
            seed_source_frame_i: id,
            seed_source_frame_j: id,
            seed_match_count: 0,
        }
    }

    fn loop_ba_hierarchy(pixel_noise: bool) -> HierarchicalSubmapGraph {
        let mut hierarchy = HierarchicalSubmapGraph::new(0, loop_ba_submap(0, true, false));
        hierarchy
            .insert_independent(1, loop_ba_submap(1, false, false))
            .unwrap();
        hierarchy
            .insert_independent(2, loop_ba_submap(2, true, pixel_noise))
            .unwrap();
        for node in hierarchy.nodes_mut() {
            node.local_from_atlas = Some(Sim3::identity());
        }
        hierarchy
    }

    fn non_adjacent_loop_links() -> Vec<HierarchicalSeamLandmarkLink> {
        (0..8_u64)
            .map(|landmark_id| HierarchicalSeamLandmarkLink {
                source_submap_id: 0,
                target_submap_id: 2,
                source_landmark_id: landmark_id,
                target_landmark_id: landmark_id,
            })
            .collect()
    }

    fn hierarchy_state(
        hierarchy: &HierarchicalSubmapGraph,
    ) -> Vec<(
        u64,
        Option<Sim3>,
        Vec<LocalSubmapFrame>,
        Vec<LocalSubmapLandmark>,
    )> {
        hierarchy
            .nodes()
            .map(|node| {
                (
                    node.id,
                    node.local_from_atlas.clone(),
                    node.submap.frames.clone(),
                    node.submap.landmarks.clone(),
                )
            })
            .collect()
    }

    #[test]
    fn atlas_local_pose_conversion_round_trips_under_sim3_gauge() {
        let gauge = Sim3::new(
            UnitQuaternion::from_euler_angles(0.2, -0.1, 0.3),
            Vector3::new(2.0, -1.0, 0.5),
            2.7,
        );
        let local = Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(-0.15, 0.08, 0.11),
            Vector3::new(0.4, -0.2, 1.3),
        );
        let recovered = atlas_pose_to_local(&local_pose_to_atlas(&local, &gauge), &gauge);
        assert!(
            recovered
                .world_to_camera
                .rotation
                .rotation_to(&local.world_to_camera.rotation)
                .angle()
                < 1e-12
        );
        assert!(
            (recovered.world_to_camera.translation - local.world_to_camera.translation).norm()
                < 1e-12
        );
    }

    #[test]
    fn disjoint_set_merges_transitive_seam_landmarks() {
        let mut dsu = DisjointSet::new(4);
        dsu.union(0, 1);
        dsu.union(1, 3);
        assert_eq!(dsu.find(0), dsu.find(3));
        assert_ne!(dsu.find(0), dsu.find(2));
    }

    #[test]
    fn loop_welds_merge_landmark_identities_across_non_adjacent_submaps() {
        let mut hierarchy = loop_ba_hierarchy(false);
        let mut config = test_config(1);
        config.ba.max_iterations = 0;

        let result = refine_hierarchical_loop_welds(
            true,
            &mut hierarchy,
            &non_adjacent_loop_links(),
            &config,
            None,
        )
        .unwrap()
        .unwrap();

        assert_eq!(result.welded_landmark_groups.len(), 8);
        assert!(result.welded_landmark_groups.iter().all(|group| {
            group.len() == 2
                && group.iter().any(|&(submap_id, _)| submap_id == 0)
                && group.iter().any(|&(submap_id, _)| submap_id == 2)
        }));
    }

    #[test]
    fn disabled_loop_ba_preserves_the_loop_pgo_model() {
        let mut hierarchy = loop_ba_hierarchy(false);
        let before = hierarchy_state(&hierarchy);

        let result = refine_hierarchical_loop_welds(
            false,
            &mut hierarchy,
            &non_adjacent_loop_links(),
            &test_config(1),
            None,
        )
        .unwrap();

        assert!(result.is_none());
        assert_eq!(hierarchy_state(&hierarchy), before);
    }

    #[test]
    fn loop_ba_cost_worsened_fallback_preserves_the_loop_pgo_model() {
        let mut hierarchy = loop_ba_hierarchy(true);
        let before = hierarchy_state(&hierarchy);
        let mut config = test_config(1);
        config.ba.max_iterations = 0;
        config.max_final_cost_ratio = -1.0;

        let result = refine_hierarchical_loop_welds(
            true,
            &mut hierarchy,
            &non_adjacent_loop_links(),
            &config,
            Some(0.0),
        )
        .unwrap();

        assert!(result.is_none());
        assert_eq!(hierarchy_state(&hierarchy), before);
    }

    #[test]
    fn welded_landmark_retriangulation_improves_bad_initial_position() {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let truth = Point3::new(0.2, -0.1, 4.5);
        let mut problem = BundleAdjustment::new(camera.clone());
        for (id, x) in [(0_u64, -0.8), (1, 0.0), (2, 0.9)] {
            let centre = Point3::new(x, 0.0, 0.0);
            let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -centre.coords);
            let pixel = camera.project(&pose.transform_world_point(&truth)).unwrap();
            problem.add_pose(id, pose);
            problem.fix_pose(id);
            problem.add_observation(BaObservation {
                keyframe_id: id,
                landmark_id: 7,
                xy: pixel,
            });
        }
        problem.add_landmark(7, Point3::new(-2.0, 1.5, 9.0));
        let before = problem.robust_cost(&RobustKernel::None);

        retriangulate_welded_landmarks(&mut problem, [7]);

        let after = problem.robust_cost(&RobustKernel::None);
        assert!(
            after < before * 1.0e-8,
            "retriangulation cost {before} -> {after}"
        );
        assert!(
            (problem.landmarks[&7] - truth).norm() < 1.0e-8,
            "retriangulation should recover the planted welded point"
        );
    }

    #[test]
    fn loop_ba_sanity_fallback_preserves_pgo_on_non_converging_setup() {
        let mut hierarchy = loop_ba_hierarchy(false);
        let root_pose = hierarchy.node(0).unwrap().submap.frames[0].pose.clone();
        hierarchy
            .nodes_mut()
            .find(|node| node.id == 2)
            .unwrap()
            .submap
            .frames[0]
            .pose = root_pose;
        let before = hierarchy_state(&hierarchy);
        let mut config = test_config(1);
        config.loop_ba_max_iterations = 0;
        config.max_final_cost_ratio = 1.0;
        config.loop_ba_max_first_seam_mean_cost_ratio = 3.0;

        let result = refine_hierarchical_loop_welds(
            true,
            &mut hierarchy,
            &non_adjacent_loop_links(),
            &config,
            Some(1.0e-6),
        )
        .unwrap();

        assert!(
            result.is_none(),
            "sanity gate should reject the unfinished BA"
        );
        assert_eq!(hierarchy_state(&hierarchy), before);
    }

    #[test]
    fn iterative_filtering_removes_planted_outlier_and_improves_ba() {
        let mut one_round = filtering_problem(true);
        let one_round_result = one_round.optimize(&test_config(1).ba).unwrap();
        let one_round_mean = one_round_result.final_cost / one_round.observations.len() as f64;

        let mut iterative = filtering_problem(true);
        let result = optimize_filter_rounds(&mut iterative, &test_config(3)).unwrap();

        assert_eq!(result.filtered_observations, 1);
        assert_eq!(result.dropped_landmarks, 0);
        assert_eq!(iterative.observations.len(), 9);
        assert!(
            result.final_cost / (iterative.observations.len() as f64) < one_round_mean,
            "filtered BA mean cost {} should improve on one-round mean cost {}",
            result.final_cost / iterative.observations.len() as f64,
            one_round_mean
        );
        assert!(
            (iterative.landmarks[&0] - Point3::new(0.15, -0.08, 4.0)).norm() < 1.0e-4,
            "filtered BA should recover the planted landmark"
        );
    }

    #[test]
    fn filtering_early_stops_when_fraction_is_below_glomap_threshold() {
        let mut problem = filtering_problem(false);
        let result = optimize_filter_rounds(&mut problem, &test_config(3)).unwrap();

        assert_eq!(result.rounds, 1);
        assert_eq!(result.filtered_observations, 0);
        assert_eq!(result.dropped_landmarks, 0);
        assert_eq!(problem.observations.len(), 10);
    }
}
