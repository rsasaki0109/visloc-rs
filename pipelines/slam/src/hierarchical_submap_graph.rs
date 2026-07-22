//! Typed hierarchy over independently reconstructed monocular submaps.
//!
//! Node state is `local_from_atlas`, rather than `atlas_from_local`. With this
//! convention an R2 measurement `target_from_source` composes directly as
//! `local_from_atlas[target] = target_from_source ∘ local_from_atlas[source]`,
//! exactly matching [`crate::Sim3PoseGraph`]'s edge convention.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

use visloc_core::geometry::Sim3;

use crate::{
    LocalSubmap, RotationOnlyConstraint, Sim3PoseGraph, Sim3PoseGraphConfig, Sim3PoseGraphResult,
    SubmapSim3Constraint, VerifiedSubmapConstraint,
};

pub type HierarchicalSubmapId = u64;

#[derive(Debug, Clone)]
pub struct HierarchicalSubmapNode {
    pub id: HierarchicalSubmapId,
    pub submap: LocalSubmap,
    /// Transform from the common atlas gauge into this node's local gauge.
    pub local_from_atlas: Option<Sim3>,
}

impl HierarchicalSubmapNode {
    pub fn atlas_from_local(&self) -> Option<Sim3> {
        self.local_from_atlas.as_ref().map(Sim3::inverse)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalSubmapOptimizationResult {
    pub root_submap_id: HierarchicalSubmapId,
    pub optimized_submap_count: usize,
    pub scale_constraint_count: usize,
    pub retained_rotation_constraint_count: usize,
    pub pose_graph: Sim3PoseGraphResult,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HierarchicalSubmapGraphError {
    DuplicateSubmap(HierarchicalSubmapId),
    MissingSubmap(HierarchicalSubmapId),
    SelfConstraint(HierarchicalSubmapId),
    NoScaleConstraints,
    DisconnectedSubmaps(Vec<HierarchicalSubmapId>),
    Optimization(String),
}

impl fmt::Display for HierarchicalSubmapGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSubmap(id) => write!(f, "duplicate hierarchical submap {id}"),
            Self::MissingSubmap(id) => write!(f, "missing hierarchical submap {id}"),
            Self::SelfConstraint(id) => write!(f, "submap {id} cannot constrain itself"),
            Self::NoScaleConstraints => write!(f, "hierarchy has no scale-bearing constraints"),
            Self::DisconnectedSubmaps(ids) => {
                write!(f, "submaps are disconnected from the root: {ids:?}")
            }
            Self::Optimization(message) => write!(f, "hierarchical Sim3 solve failed: {message}"),
        }
    }
}

impl Error for HierarchicalSubmapGraphError {}

#[derive(Debug, Clone)]
pub struct HierarchicalSubmapGraph {
    root_submap_id: HierarchicalSubmapId,
    nodes: BTreeMap<HierarchicalSubmapId, HierarchicalSubmapNode>,
    scale_constraints: Vec<SubmapSim3Constraint>,
    rotation_constraints: Vec<RotationOnlyConstraint>,
}

impl HierarchicalSubmapGraph {
    pub fn new(root_submap_id: HierarchicalSubmapId, root_submap: LocalSubmap) -> Self {
        let mut nodes = BTreeMap::new();
        nodes.insert(
            root_submap_id,
            HierarchicalSubmapNode {
                id: root_submap_id,
                submap: root_submap,
                local_from_atlas: Some(Sim3::identity()),
            },
        );
        Self {
            root_submap_id,
            nodes,
            scale_constraints: Vec::new(),
            rotation_constraints: Vec::new(),
        }
    }

    pub fn root_submap_id(&self) -> HierarchicalSubmapId {
        self.root_submap_id
    }

    pub fn nodes(&self) -> impl Iterator<Item = &HierarchicalSubmapNode> {
        self.nodes.values()
    }

    pub fn node(&self, id: HierarchicalSubmapId) -> Option<&HierarchicalSubmapNode> {
        self.nodes.get(&id)
    }

    pub(crate) fn nodes_mut(&mut self) -> impl Iterator<Item = &mut HierarchicalSubmapNode> {
        self.nodes.values_mut()
    }

    pub fn scale_constraints(&self) -> &[SubmapSim3Constraint] {
        &self.scale_constraints
    }

    pub fn rotation_constraints(&self) -> &[RotationOnlyConstraint] {
        &self.rotation_constraints
    }

    pub fn insert_independent(
        &mut self,
        id: HierarchicalSubmapId,
        submap: LocalSubmap,
    ) -> Result<(), HierarchicalSubmapGraphError> {
        if self.nodes.contains_key(&id) {
            return Err(HierarchicalSubmapGraphError::DuplicateSubmap(id));
        }
        self.nodes.insert(
            id,
            HierarchicalSubmapNode {
                id,
                submap,
                local_from_atlas: None,
            },
        );
        Ok(())
    }

    pub fn add_constraint(
        &mut self,
        constraint: VerifiedSubmapConstraint,
    ) -> Result<(), HierarchicalSubmapGraphError> {
        let (source, target) = match &constraint {
            VerifiedSubmapConstraint::RotationOnly(constraint) => {
                (constraint.source_submap_id, constraint.target_submap_id)
            }
            VerifiedSubmapConstraint::Sim3(constraint) => {
                (constraint.source_submap_id, constraint.target_submap_id)
            }
        };
        if source == target {
            return Err(HierarchicalSubmapGraphError::SelfConstraint(source));
        }
        for id in [source, target] {
            if !self.nodes.contains_key(&id) {
                return Err(HierarchicalSubmapGraphError::MissingSubmap(id));
            }
        }
        match constraint {
            VerifiedSubmapConstraint::RotationOnly(constraint) => {
                self.rotation_constraints.push(constraint)
            }
            VerifiedSubmapConstraint::Sim3(constraint) => self.scale_constraints.push(constraint),
        }
        Ok(())
    }

    pub fn optimize(
        &mut self,
        config: &Sim3PoseGraphConfig,
    ) -> Result<HierarchicalSubmapOptimizationResult, HierarchicalSubmapGraphError> {
        if self.scale_constraints.is_empty() {
            return Err(HierarchicalSubmapGraphError::NoScaleConstraints);
        }
        self.initialize_connected_gauges();
        let disconnected = self
            .nodes
            .iter()
            .filter_map(|(&id, node)| node.local_from_atlas.is_none().then_some(id))
            .collect::<Vec<_>>();
        if !disconnected.is_empty() {
            return Err(HierarchicalSubmapGraphError::DisconnectedSubmaps(
                disconnected,
            ));
        }

        let mut graph = Sim3PoseGraph::new();
        for (&id, node) in &self.nodes {
            graph.add_pose(
                id,
                node.local_from_atlas
                    .clone()
                    .expect("connected gauges checked above"),
            );
        }
        graph.anchor(self.root_submap_id);
        for constraint in &self.scale_constraints {
            graph.add_edge(
                constraint.source_submap_id,
                constraint.target_submap_id,
                constraint.target_from_source.clone(),
                constraint.inlier_match_indices.len() as f64,
            );
        }
        let pose_graph = graph
            .optimize(config)
            .map_err(|error| HierarchicalSubmapGraphError::Optimization(error.to_string()))?;
        for (id, transform) in graph.poses {
            self.nodes
                .get_mut(&id)
                .ok_or(HierarchicalSubmapGraphError::MissingSubmap(id))?
                .local_from_atlas = Some(transform);
        }
        Ok(HierarchicalSubmapOptimizationResult {
            root_submap_id: self.root_submap_id,
            optimized_submap_count: self.nodes.len(),
            scale_constraint_count: self.scale_constraints.len(),
            retained_rotation_constraint_count: self.rotation_constraints.len(),
            pose_graph,
        })
    }

    fn initialize_connected_gauges(&mut self) {
        for node in self.nodes.values_mut() {
            node.local_from_atlas = None;
        }
        self.nodes
            .get_mut(&self.root_submap_id)
            .expect("root node is retained for graph lifetime")
            .local_from_atlas = Some(Sim3::identity());

        let mut queued = BTreeSet::new();
        let mut queue = VecDeque::from([self.root_submap_id]);
        queued.insert(self.root_submap_id);
        while let Some(known_id) = queue.pop_front() {
            let known = self.nodes[&known_id]
                .local_from_atlas
                .clone()
                .expect("queued nodes are initialized");
            for constraint in &self.scale_constraints {
                let proposal = if constraint.source_submap_id == known_id {
                    Some((
                        constraint.target_submap_id,
                        constraint.target_from_source.compose(&known),
                    ))
                } else if constraint.target_submap_id == known_id {
                    Some((
                        constraint.source_submap_id,
                        constraint.target_from_source.inverse().compose(&known),
                    ))
                } else {
                    None
                };
                let Some((unknown_id, transform)) = proposal else {
                    continue;
                };
                let node = self
                    .nodes
                    .get_mut(&unknown_id)
                    .expect("constraint endpoints validated on insertion");
                if node.local_from_atlas.is_none() {
                    node.local_from_atlas = Some(transform);
                    if queued.insert(unknown_id) {
                        queue.push_back(unknown_id);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, UnitQuaternion, Vector3};

    fn empty_submap() -> LocalSubmap {
        LocalSubmap {
            camera: visloc_core::types::Camera::pinhole(0, 64, 48, 50.0, 50.0, 32.0, 24.0),
            source_frame_ids: Vec::new(),
            frames: Vec::new(),
            landmarks: Vec::new(),
            quality: crate::LocalSubmapQuality {
                requested_images: 0,
                registered_images: 0,
                registration_fraction: 0.0,
                landmarks: 0,
                observations: 0,
                median_track_length: 0.0,
                median_max_parallax_deg: 0.0,
                camera_center_diameter: 0.0,
                mean_reprojection_px: 0.0,
                leave_one_out_attempts: 0,
                leave_one_out_supported: 0,
                leave_one_out_support_fraction: 0.0,
                median_leave_one_out_reprojection_px: 0.0,
            },
            track_build_stats: crate::TrackBuildStats::default(),
            ba_result: None,
        }
    }

    fn constraint(source: u64, target: u64, transform: Sim3) -> VerifiedSubmapConstraint {
        VerifiedSubmapConstraint::Sim3(SubmapSim3Constraint {
            source_submap_id: source,
            target_submap_id: target,
            target_from_source: transform,
            correspondence_count: 20,
            inlier_match_indices: (0..20).collect(),
            inlier_ratio: 1.0,
            mean_residual_ratio: 0.0,
            rotation_disagreement_deg: 0.0,
            leave_one_out_log_scale_mad: 0.0,
            target_scene_scale: 1.0,
        })
    }

    #[test]
    fn constraint_direction_places_shared_points_in_one_atlas_gauge() {
        let target_from_source = Sim3::new(
            UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3),
            Vector3::new(1.0, -2.0, 0.5),
            2.5,
        );
        let mut hierarchy = HierarchicalSubmapGraph::new(0, empty_submap());
        hierarchy.insert_independent(1, empty_submap()).unwrap();
        hierarchy
            .add_constraint(constraint(0, 1, target_from_source.clone()))
            .unwrap();
        hierarchy.initialize_connected_gauges();

        let source_point = Point3::new(0.4, -0.7, 3.0);
        let target_point = target_from_source.transform_point(&source_point);
        let source_atlas = hierarchy.node(0).unwrap().atlas_from_local().unwrap();
        let target_atlas = hierarchy.node(1).unwrap().atlas_from_local().unwrap();
        assert!(
            (source_atlas.transform_point(&source_point)
                - target_atlas.transform_point(&target_point))
            .norm()
                < 1.0e-10
        );
    }

    #[test]
    fn optimizes_connected_scale_graph_and_retains_rotation_only_evidence() {
        let mut hierarchy = HierarchicalSubmapGraph::new(0, empty_submap());
        hierarchy.insert_independent(1, empty_submap()).unwrap();
        hierarchy.insert_independent(2, empty_submap()).unwrap();
        hierarchy
            .add_constraint(constraint(
                0,
                1,
                Sim3::new(UnitQuaternion::identity(), Vector3::new(1.0, 0.0, 0.0), 1.5),
            ))
            .unwrap();
        hierarchy
            .add_constraint(constraint(
                1,
                2,
                Sim3::new(UnitQuaternion::identity(), Vector3::new(0.0, 2.0, 0.0), 0.8),
            ))
            .unwrap();
        hierarchy
            .add_constraint(VerifiedSubmapConstraint::RotationOnly(
                RotationOnlyConstraint {
                    source_submap_id: 0,
                    target_submap_id: 2,
                    target_from_source_rotation: UnitQuaternion::identity(),
                    inlier_count: 50,
                    spatial_coverage: 0.8,
                    geometry: crate::RotationConstraintGeometry::Essential,
                },
            ))
            .unwrap();

        let result = hierarchy.optimize(&Sim3PoseGraphConfig::default()).unwrap();
        assert_eq!(result.optimized_submap_count, 3);
        assert_eq!(result.scale_constraint_count, 2);
        assert_eq!(result.retained_rotation_constraint_count, 1);
        assert!(result.pose_graph.final_cost <= result.pose_graph.initial_cost);
        assert_eq!(hierarchy.rotation_constraints().len(), 1);
    }

    #[test]
    fn refuses_disconnected_and_rotation_only_hierarchies() {
        let mut disconnected = HierarchicalSubmapGraph::new(0, empty_submap());
        disconnected.insert_independent(1, empty_submap()).unwrap();
        disconnected.insert_independent(2, empty_submap()).unwrap();
        disconnected
            .add_constraint(constraint(0, 1, Sim3::identity()))
            .unwrap();
        assert_eq!(
            disconnected.optimize(&Sim3PoseGraphConfig::default()),
            Err(HierarchicalSubmapGraphError::DisconnectedSubmaps(vec![2]))
        );

        let mut rotation_only = HierarchicalSubmapGraph::new(0, empty_submap());
        rotation_only.insert_independent(1, empty_submap()).unwrap();
        rotation_only
            .add_constraint(VerifiedSubmapConstraint::RotationOnly(
                RotationOnlyConstraint {
                    source_submap_id: 0,
                    target_submap_id: 1,
                    target_from_source_rotation: UnitQuaternion::identity(),
                    inlier_count: 40,
                    spatial_coverage: 0.7,
                    geometry: crate::RotationConstraintGeometry::PureRotation,
                },
            ))
            .unwrap();
        assert_eq!(
            rotation_only.optimize(&Sim3PoseGraphConfig::default()),
            Err(HierarchicalSubmapGraphError::NoScaleConstraints)
        );
    }
}
