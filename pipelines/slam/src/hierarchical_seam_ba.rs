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
    /// Keep root-only prefix poses fixed but jointly refine root poses also
    /// reconstructed by another submap. At least two root poses remain fixed.
    pub optimize_shared_root_poses: bool,
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
            optimize_shared_root_poses: true,
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
    /// R2-verified local landmark identities merged into one BA variable.
    pub welded_landmark_groups: Vec<Vec<(u64, u64)>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HierarchicalSeamBaError {
    MissingAlignedTransform(u64),
    MissingSubmap(u64),
    TooFewObservations { required: usize, found: usize },
    Optimize(BaError),
    CostWorsened { initial: f64, final_cost: f64 },
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
    let result = problem.optimize(&config.ba)?;
    if !result.final_cost.is_finite()
        || result.final_cost > result.initial_cost * config.max_final_cost_ratio
    {
        return Err(HierarchicalSeamBaError::CostWorsened {
            initial: result.initial_cost,
            final_cost: result.final_cost,
        });
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
        initial_cost: result.initial_cost,
        final_cost: result.final_cost,
        iterations: result.iterations.len(),
        welded_landmark_groups,
    })
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
    use nalgebra::{UnitQuaternion, Vector3};

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
}
