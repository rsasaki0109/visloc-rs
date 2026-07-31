//! Descriptor-derived long-range constraints and post-seam-BA submap PGO.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use visloc_core::types::Camera;
use visloc_vision::features::FeatureSet;
use visloc_vision::matching::{BruteForceMatcher, CrossCheckMatcher, Matcher};
use visloc_vision::two_view::{RelativePoseEstimator, TwoViewCorrespondence};

use crate::{
    collect_submap_overlap_evidence, estimate_submap_sim3_constraint, HierarchicalSeamLandmarkLink,
    HierarchicalSubmapGraph, HierarchicalSubmapGraphError, PairRotationEvidence, Sim3PoseGraph,
    Sim3PoseGraphConfig, Sim3PoseGraphResult, SubmapOverlapConfig, SubmapPointMatch,
    SubmapSim3AlignmentConfig, SubmapSim3Constraint,
};

const REPRESENTATIVE_FRAME_STRIDE: usize = 4;
const VERIFICATION_FRAME_RADIUS_STEP: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalLoopClosureResult {
    /// Number of non-banded pairs scored by their cached global descriptors.
    pub candidates_ranked: usize,
    /// Number of ranked pairs admitted to expensive center-frame screening.
    pub candidates_screened: usize,
    pub pairs_matched: usize,
    pub edges_accepted: usize,
    pub edges_rejected: usize,
    /// `None` when screening/verification admitted no loop edge.
    pub pose_graph: Option<Sim3PoseGraphResult>,
}

pub(crate) struct HierarchicalLoopClosureOutput {
    pub result: HierarchicalLoopClosureResult,
    pub landmark_links: Vec<HierarchicalSeamLandmarkLink>,
}

pub(crate) fn maybe_close_hierarchical_loops(
    enabled: bool,
    hierarchy: &mut HierarchicalSubmapGraph,
    camera: &Camera,
    source_frame_ids: &[u64],
    features: &[FeatureSet],
    constraint_band: usize,
    top_k: usize,
    min_similarity: Option<f32>,
    match_ratio: f32,
    min_matches: usize,
    verification_frame_budget: usize,
    verification_correspondence_target: usize,
    verification_threads: usize,
    overlap_config: &SubmapOverlapConfig,
    alignment_config: &SubmapSim3AlignmentConfig,
    pose_graph_config: &Sim3PoseGraphConfig,
) -> Result<Option<HierarchicalLoopClosureOutput>, HierarchicalSubmapGraphError> {
    if !enabled {
        return Ok(None);
    }
    close_hierarchical_loops(
        hierarchy,
        camera,
        source_frame_ids,
        features,
        constraint_band,
        top_k,
        min_similarity,
        match_ratio,
        min_matches,
        verification_frame_budget,
        verification_correspondence_target,
        verification_threads,
        overlap_config,
        alignment_config,
        pose_graph_config,
    )
    .map(Some)
}

fn close_hierarchical_loops(
    hierarchy: &mut HierarchicalSubmapGraph,
    camera: &Camera,
    source_frame_ids: &[u64],
    features: &[FeatureSet],
    constraint_band: usize,
    top_k: usize,
    min_similarity: Option<f32>,
    match_ratio: f32,
    min_matches: usize,
    verification_frame_budget: usize,
    verification_correspondence_target: usize,
    verification_threads: usize,
    overlap_config: &SubmapOverlapConfig,
    alignment_config: &SubmapSim3AlignmentConfig,
    pose_graph_config: &Sim3PoseGraphConfig,
) -> Result<HierarchicalLoopClosureOutput, HierarchicalSubmapGraphError> {
    let loop_constraints = discover_loop_constraints(
        hierarchy,
        camera,
        source_frame_ids,
        features,
        constraint_band,
        top_k,
        min_similarity,
        match_ratio,
        min_matches,
        verification_frame_budget,
        verification_correspondence_target,
        verification_threads,
        overlap_config,
        alignment_config,
    );
    let mut result = HierarchicalLoopClosureResult {
        candidates_ranked: loop_constraints.candidates_ranked,
        candidates_screened: loop_constraints.candidates_screened,
        pairs_matched: loop_constraints.pairs_matched,
        edges_accepted: loop_constraints.constraints.len(),
        edges_rejected: loop_constraints.edges_rejected,
        pose_graph: None,
    };
    if loop_constraints.constraints.is_empty() {
        eprintln!(
            "hierarchical-loop: summary candidates_ranked={} candidates_screened={} pairs_matched={} \
             edges_accepted=0 edges_rejected={} pose_graph=skipped",
            result.candidates_ranked,
            result.candidates_screened,
            result.pairs_matched,
            result.edges_rejected,
        );
        return Ok(HierarchicalLoopClosureOutput {
            result,
            landmark_links: Vec::new(),
        });
    }

    let constraints = loop_constraints
        .constraints
        .iter()
        .map(|verified| verified.constraint.clone())
        .collect::<Vec<_>>();
    let pose_graph =
        optimize_post_ba_pose_graph(hierarchy, &constraints, constraint_band, pose_graph_config)?;
    eprintln!(
        "hierarchical-loop: pose-graph initial_cost={:.9} final_cost={:.9} \
         candidates_ranked={} candidates_screened={} pairs_matched={} edges_accepted={} edges_rejected={}",
        pose_graph.initial_cost,
        pose_graph.final_cost,
        result.candidates_ranked,
        result.candidates_screened,
        result.pairs_matched,
        result.edges_accepted,
        result.edges_rejected,
    );
    result.pose_graph = Some(pose_graph);
    Ok(HierarchicalLoopClosureOutput {
        result,
        landmark_links: loop_constraints
            .constraints
            .into_iter()
            .flat_map(|verified| verified.landmark_links)
            .collect(),
    })
}

struct LoopConstraintDiscovery {
    candidates_ranked: usize,
    candidates_screened: usize,
    pairs_matched: usize,
    edges_rejected: usize,
    constraints: Vec<VerifiedLoopConstraint>,
}

struct VerifiedLoopConstraint {
    constraint: SubmapSim3Constraint,
    landmark_links: Vec<HierarchicalSeamLandmarkLink>,
}

#[derive(Debug, Clone, Copy)]
struct ScreenedCandidate {
    source_index: usize,
    target_index: usize,
    similarity: f32,
}

struct VerificationOutcome {
    source_id: u64,
    target_id: u64,
    similarity: f32,
    elapsed: Duration,
    result: Result<VerifiedLoopConstraint, String>,
}

fn discover_loop_constraints(
    hierarchy: &HierarchicalSubmapGraph,
    camera: &Camera,
    source_frame_ids: &[u64],
    features: &[FeatureSet],
    constraint_band: usize,
    top_k: usize,
    min_similarity: Option<f32>,
    match_ratio: f32,
    min_matches: usize,
    verification_frame_budget: usize,
    verification_correspondence_target: usize,
    verification_threads: usize,
    overlap_config: &SubmapOverlapConfig,
    alignment_config: &SubmapSim3AlignmentConfig,
) -> LoopConstraintDiscovery {
    let feature_index = source_frame_ids
        .iter()
        .enumerate()
        .map(|(index, &frame_id)| (frame_id, index))
        .collect::<BTreeMap<_, _>>();
    let nodes = hierarchy.nodes().collect::<Vec<_>>();
    let global_descriptors = nodes
        .iter()
        .map(|node| submap_global_descriptor(&node.submap, &feature_index, features))
        .collect::<Vec<_>>();
    let ranked_candidates =
        rank_loop_candidates(&global_descriptors, constraint_band, top_k, min_similarity);
    let candidates_ranked = eligible_pair_count(nodes.len(), constraint_band);
    let matcher = CrossCheckMatcher::new(BruteForceMatcher {
        ratio: Some(match_ratio),
    });
    let required_matches = min_matches.max(1);
    let mut candidates_screened = 0;
    let mut screened_candidates = Vec::new();
    let mut edges_rejected = 0;

    eprintln!(
        "hierarchical-loop: prescreen candidates_ranked={} candidates_selected={} top_k={} min_similarity={:?}",
        candidates_ranked,
        ranked_candidates.len(),
        top_k,
        min_similarity,
    );
    for &(source_index, target_index, similarity) in &ranked_candidates {
        candidates_screened += 1;
        let source = nodes[source_index];
        let target = nodes[target_index];
        let Some(source_center) = source
            .submap
            .frames
            .get(source.submap.frames.len().saturating_sub(1) / 2)
        else {
            edges_rejected += 1;
            eprintln!(
                "hierarchical-loop: edge {}..{} rejected reason=no-registered-source-frame",
                source.id, target.id
            );
            continue;
        };
        let Some(target_center) = target
            .submap
            .frames
            .get(target.submap.frames.len().saturating_sub(1) / 2)
        else {
            edges_rejected += 1;
            eprintln!(
                "hierarchical-loop: edge {}..{} rejected reason=no-registered-target-frame",
                source.id, target.id
            );
            continue;
        };
        let Some(source_features) = feature_index
            .get(&source_center.source_frame_id)
            .and_then(|&index| features.get(index))
        else {
            edges_rejected += 1;
            eprintln!(
                "hierarchical-loop: edge {}..{} rejected reason=missing-source-features",
                source.id, target.id
            );
            continue;
        };
        let Some(target_features) = feature_index
            .get(&target_center.source_frame_id)
            .and_then(|&index| features.get(index))
        else {
            edges_rejected += 1;
            eprintln!(
                "hierarchical-loop: edge {}..{} rejected reason=missing-target-features",
                source.id, target.id
            );
            continue;
        };
        let screen_matches =
            matcher.match_descriptors(&source_features.descriptors, &target_features.descriptors);
        if screen_matches.len() < required_matches {
            edges_rejected += 1;
            eprintln!(
                "hierarchical-loop: edge {}..{} rejected reason=screen-matches \
                     similarity={:.6} raw_matches={} required={}",
                source.id,
                target.id,
                similarity,
                screen_matches.len(),
                required_matches,
            );
            continue;
        }

        screened_candidates.push(ScreenedCandidate {
            source_index,
            target_index,
            similarity,
        });
    }
    let pairs_matched = screened_candidates.len();
    let worker_count = verification_threads.max(1).min(pairs_matched.max(1));
    let verify = |candidate: &ScreenedCandidate| {
        verify_screened_candidate(
            *candidate,
            &nodes,
            camera,
            &feature_index,
            features,
            match_ratio,
            required_matches,
            verification_frame_budget,
            verification_correspondence_target,
            overlap_config,
            alignment_config,
        )
    };
    let outcomes = if worker_count == 1 {
        screened_candidates.iter().map(verify).collect::<Vec<_>>()
    } else {
        match rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
        {
            Ok(pool) => pool.install(|| screened_candidates.par_iter().map(verify).collect()),
            Err(error) => {
                eprintln!(
                    "hierarchical-loop: verification-pool threads={} error={}; falling back to serial",
                    worker_count, error
                );
                screened_candidates.iter().map(verify).collect()
            }
        }
    };
    let total_verification_seconds = outcomes
        .iter()
        .map(|outcome| outcome.elapsed.as_secs_f64())
        .sum::<f64>();
    eprintln!(
        "hierarchical-loop: verification-summary pairs_verified={} mean_seconds_per_pair={:.3}",
        outcomes.len(),
        if outcomes.is_empty() {
            0.0
        } else {
            total_verification_seconds / outcomes.len() as f64
        },
    );
    let mut constraints = Vec::new();
    for outcome in outcomes {
        match outcome.result {
            Ok(verified) => {
                eprintln!(
                    "hierarchical-loop: edge {}..{} accepted similarity={:.6} correspondences={} \
                     inliers={} inlier_ratio={:.6} mean_residual_ratio={:.9}",
                    outcome.source_id,
                    outcome.target_id,
                    outcome.similarity,
                    verified.constraint.correspondence_count,
                    verified.constraint.inlier_match_indices.len(),
                    verified.constraint.inlier_ratio,
                    verified.constraint.mean_residual_ratio,
                );
                constraints.push(verified);
            }
            Err(reason) => {
                edges_rejected += 1;
                eprintln!(
                    "hierarchical-loop: edge {}..{} rejected similarity={:.6} {}",
                    outcome.source_id, outcome.target_id, outcome.similarity, reason
                );
            }
        }
    }
    LoopConstraintDiscovery {
        candidates_ranked,
        candidates_screened,
        pairs_matched,
        edges_rejected,
        constraints,
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_screened_candidate(
    candidate: ScreenedCandidate,
    nodes: &[&crate::HierarchicalSubmapNode],
    camera: &Camera,
    feature_index: &BTreeMap<u64, usize>,
    features: &[FeatureSet],
    match_ratio: f32,
    required_matches: usize,
    verification_frame_budget: usize,
    verification_correspondence_target: usize,
    overlap_config: &SubmapOverlapConfig,
    alignment_config: &SubmapSim3AlignmentConfig,
) -> VerificationOutcome {
    let started = Instant::now();
    let source = nodes[candidate.source_index];
    let target = nodes[candidate.target_index];
    let matcher = CrossCheckMatcher::new(BruteForceMatcher {
        ratio: Some(match_ratio),
    });
    let source_observations = observed_landmarks(&source.submap);
    let target_observations = observed_landmarks(&target.submap);
    let source_frames = verification_frames(&source.submap, verification_frame_budget);
    let target_frames = verification_frames(&target.submap, verification_frame_budget);
    let correspondence_target =
        verification_correspondence_target.max(alignment_config.min_correspondences.max(3));
    let mut votes = BTreeMap::<(u64, u64), usize>::new();
    let mut rotation_evidence = Vec::new();
    let mut point_matches = Vec::new();

    'frame_pairs: for source_frame in &source_frames {
        for target_frame in &target_frames {
            let Some(source_features) = feature_index
                .get(&source_frame.source_frame_id)
                .and_then(|&index| features.get(index))
            else {
                continue;
            };
            let Some(target_features) = feature_index
                .get(&target_frame.source_frame_id)
                .and_then(|&index| features.get(index))
            else {
                continue;
            };
            let descriptor_matches = matcher
                .match_descriptors(&source_features.descriptors, &target_features.descriptors);
            if descriptor_matches.len() < required_matches {
                continue;
            }
            let correspondences = descriptor_matches
                .iter()
                .map(|descriptor_match| {
                    TwoViewCorrespondence::new(
                        source_features.keypoints[descriptor_match.query_index],
                        target_features.keypoints[descriptor_match.train_index],
                    )
                })
                .collect::<Vec<_>>();
            let Some(relative) =
                RelativePoseEstimator::default().estimate(&correspondences, camera)
            else {
                continue;
            };
            if relative.inliers.len() < required_matches {
                continue;
            }
            rotation_evidence.push(PairRotationEvidence {
                image_i: source_frame.source_frame_id,
                image_j: target_frame.source_frame_id,
                image_j_from_i: relative.previous_to_current.rotation,
                inlier_count: relative.inliers.len(),
            });
            for &inlier_index in &relative.inliers {
                let descriptor_match = &descriptor_matches[inlier_index];
                let source_key = (source_frame.source_frame_id, descriptor_match.query_index);
                let target_key = (target_frame.source_frame_id, descriptor_match.train_index);
                if let (Some(Some(source_id)), Some(Some(target_id))) = (
                    source_observations.get(&source_key),
                    target_observations.get(&target_key),
                ) {
                    *votes.entry((*source_id, *target_id)).or_default() += 1;
                }
            }
            point_matches = mutual_point_matches(&votes, &source.submap, &target.submap);
            if point_matches.len() >= correspondence_target {
                break 'frame_pairs;
            }
        }
    }

    let result = collect_submap_overlap_evidence(
        &source.submap,
        &target.submap,
        &rotation_evidence,
        overlap_config,
    )
    .map_err(|error| format!("reason=rotation-evidence error={error}"))
    .and_then(|evidence| {
        if point_matches.is_empty() && !votes.is_empty() {
            point_matches = mutual_point_matches(&votes, &source.submap, &target.submap);
        }
        estimate_submap_sim3_constraint(
            source.id,
            target.id,
            &point_matches,
            &evidence.target_from_source_rotation,
            alignment_config,
        )
        .map(|constraint| {
            let landmark_links = constraint
                .inlier_match_indices
                .iter()
                .map(|&index| {
                    let point_match = &point_matches[index];
                    HierarchicalSeamLandmarkLink {
                        source_submap_id: source.id,
                        target_submap_id: target.id,
                        source_landmark_id: point_match.source_landmark_id,
                        target_landmark_id: point_match.target_landmark_id,
                    }
                })
                .collect();
            VerifiedLoopConstraint {
                constraint,
                landmark_links,
            }
        })
        .map_err(|rejection| {
            format!(
                "reason={:?} correspondences={} inliers={} inlier_ratio={:.6} \
                 mean_residual_ratio={:?}",
                rejection.reason,
                rejection.correspondence_count,
                rejection.inlier_count,
                rejection.inlier_ratio,
                rejection.mean_residual_ratio,
            )
        })
    });
    VerificationOutcome {
        source_id: source.id,
        target_id: target.id,
        similarity: candidate.similarity,
        elapsed: started.elapsed(),
        result,
    }
}

/// Mean-pool every SuperPoint descriptor belonging to the submap's
/// representative frames, then L2-normalize once. This mirrors the A3
/// mean-pool retrieval scorer while avoiding any per-pair descriptor work.
fn submap_global_descriptor(
    submap: &crate::LocalSubmap,
    feature_index: &BTreeMap<u64, usize>,
    features: &[FeatureSet],
) -> Vec<f32> {
    let mut sum = Vec::<f32>::new();
    let mut count = 0usize;
    for frame in representative_frames(submap) {
        let Some(frame_features) = feature_index
            .get(&frame.source_frame_id)
            .and_then(|&index| features.get(index))
        else {
            continue;
        };
        for descriptor in &frame_features.descriptors {
            if descriptor.is_empty() {
                continue;
            }
            if sum.is_empty() {
                sum.resize(descriptor.len(), 0.0);
            }
            if descriptor.len() != sum.len() {
                continue;
            }
            for (accumulator, &value) in sum.iter_mut().zip(descriptor) {
                *accumulator += value;
            }
            count += 1;
        }
    }
    if count == 0 {
        return Vec::new();
    }
    let inverse_count = 1.0 / count as f32;
    for value in &mut sum {
        *value *= inverse_count;
    }
    let norm = sum.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 1.0e-12 {
        for value in &mut sum {
            *value /= norm;
        }
    }
    sum
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || left.len() != right.len() {
        return 0.0;
    }
    // Submap signatures are already L2-normalized, so cosine is one dot
    // product. Clamp tiny floating-point overshoots to cosine's valid range.
    left.iter()
        .zip(right)
        .map(|(&a, &b)| a * b)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

fn eligible_pair_count(submap_count: usize, constraint_band: usize) -> usize {
    (0..submap_count)
        .map(|source| submap_count.saturating_sub(source + constraint_band + 1))
        .sum()
}

/// Return the union of each submap's top-K non-banded partners, plus every
/// pair clearing the optional absolute similarity threshold.
fn rank_loop_candidates(
    global_descriptors: &[Vec<f32>],
    constraint_band: usize,
    top_k: usize,
    min_similarity: Option<f32>,
) -> Vec<(usize, usize, f32)> {
    let mut scored = Vec::with_capacity(eligible_pair_count(
        global_descriptors.len(),
        constraint_band,
    ));
    let mut incident = vec![Vec::<usize>::new(); global_descriptors.len()];
    for source in 0..global_descriptors.len() {
        for target in source + constraint_band + 1..global_descriptors.len() {
            let score = cosine_similarity(&global_descriptors[source], &global_descriptors[target]);
            let pair_index = scored.len();
            scored.push((source, target, score));
            incident[source].push(pair_index);
            incident[target].push(pair_index);
        }
    }

    let mut selected = BTreeSet::new();
    for pair_indices in &mut incident {
        pair_indices.sort_by(|&left, &right| {
            scored[right]
                .2
                .total_cmp(&scored[left].2)
                .then_with(|| scored[left].0.cmp(&scored[right].0))
                .then_with(|| scored[left].1.cmp(&scored[right].1))
        });
        selected.extend(pair_indices.iter().take(top_k.max(1)).copied());
    }
    if let Some(threshold) = min_similarity {
        selected.extend(
            scored
                .iter()
                .enumerate()
                .filter_map(|(index, &(_, _, score))| (score >= threshold).then_some(index)),
        );
    }

    selected.into_iter().map(|index| scored[index]).collect()
}

fn representative_frames(submap: &crate::LocalSubmap) -> Vec<&crate::LocalSubmapFrame> {
    let mut indices = (0..submap.frames.len())
        .step_by(REPRESENTATIVE_FRAME_STRIDE)
        .collect::<Vec<_>>();
    if !submap.frames.is_empty() {
        indices.push((submap.frames.len() - 1) / 2);
        indices.push(submap.frames.len() - 1);
    }
    indices.sort_unstable();
    indices.dedup();
    indices
        .into_iter()
        .map(|index| &submap.frames[index])
        .collect()
}

fn verification_frame_indices(frame_count: usize, budget: usize) -> Vec<usize> {
    if frame_count == 0 || budget == 0 {
        return Vec::new();
    }
    let center = (frame_count - 1) / 2;
    let mut indices = vec![center];
    for radius in 1.. {
        if indices.len() >= budget {
            break;
        }
        let offset = radius * VERIFICATION_FRAME_RADIUS_STEP;
        let left = center.saturating_sub(offset);
        if !indices.contains(&left) {
            indices.push(left);
        }
        if indices.len() >= budget {
            break;
        }
        let right = center.saturating_add(offset).min(frame_count - 1);
        if !indices.contains(&right) {
            indices.push(right);
        }
        if left == 0 && right == frame_count - 1 {
            break;
        }
    }
    indices
}

fn verification_frames(
    submap: &crate::LocalSubmap,
    budget: usize,
) -> Vec<&crate::LocalSubmapFrame> {
    verification_frame_indices(submap.frames.len(), budget)
        .into_iter()
        .map(|index| &submap.frames[index])
        .collect()
}

fn observed_landmarks(submap: &crate::LocalSubmap) -> BTreeMap<(u64, usize), Option<u64>> {
    let mut observations = BTreeMap::new();
    for landmark in &submap.landmarks {
        for observation in &landmark.observations {
            observations
                .entry((observation.source_frame_id, observation.keypoint_index))
                .and_modify(|entry| *entry = None)
                .or_insert(Some(landmark.local_landmark_id));
        }
    }
    observations
}

fn mutual_point_matches(
    votes: &BTreeMap<(u64, u64), usize>,
    source: &crate::LocalSubmap,
    target: &crate::LocalSubmap,
) -> Vec<SubmapPointMatch> {
    let mut best_source = BTreeMap::<u64, (u64, usize)>::new();
    let mut best_target = BTreeMap::<u64, (u64, usize)>::new();
    for (&(source_id, target_id), &count) in votes {
        update_best(&mut best_source, source_id, target_id, count);
        update_best(&mut best_target, target_id, source_id, count);
    }
    let source_points = source
        .landmarks
        .iter()
        .map(|landmark| (landmark.local_landmark_id, landmark.position))
        .collect::<BTreeMap<_, _>>();
    let target_points = target
        .landmarks
        .iter()
        .map(|landmark| (landmark.local_landmark_id, landmark.position))
        .collect::<BTreeMap<_, _>>();
    best_source
        .into_iter()
        .filter_map(|(source_id, (target_id, _))| {
            (best_target.get(&target_id)?.0 == source_id).then(|| SubmapPointMatch {
                source_landmark_id: source_id,
                target_landmark_id: target_id,
                source_point: source_points[&source_id],
                target_point: target_points[&target_id],
            })
        })
        .collect()
}

fn update_best(best: &mut BTreeMap<u64, (u64, usize)>, key: u64, candidate: u64, count: usize) {
    if best.get(&key).is_none_or(|&(current, current_count)| {
        count > current_count || (count == current_count && candidate < current)
    }) {
        best.insert(key, (candidate, count));
    }
}

pub(crate) fn optimize_post_ba_pose_graph(
    hierarchy: &mut HierarchicalSubmapGraph,
    loop_constraints: &[SubmapSim3Constraint],
    constraint_band: usize,
    config: &Sim3PoseGraphConfig,
) -> Result<Sim3PoseGraphResult, HierarchicalSubmapGraphError> {
    let mut graph = Sim3PoseGraph::new();
    for node in hierarchy.nodes() {
        let pose = node
            .local_from_atlas
            .clone()
            .ok_or(HierarchicalSubmapGraphError::MissingSubmap(node.id))?;
        graph.add_pose(node.id, pose);
    }
    graph.anchor(hierarchy.root_submap_id());

    // Re-measure all retained adjacent/banded edges from the current post-BA
    // gauges. They therefore encode the BA solution exactly; only independent
    // long-range loop measurements ask the graph to leave that attractor.
    for constraint in hierarchy.scale_constraints() {
        let separation = constraint
            .target_submap_id
            .abs_diff(constraint.source_submap_id) as usize;
        if separation == 0 || separation > constraint_band.max(1) {
            continue;
        }
        let source = graph.poses[&constraint.source_submap_id].clone();
        let target = graph.poses[&constraint.target_submap_id].clone();
        graph.add_edge(
            constraint.source_submap_id,
            constraint.target_submap_id,
            target.compose(&source.inverse()),
            constraint.inlier_match_indices.len() as f64,
        );
    }
    for constraint in loop_constraints {
        graph.add_edge(
            constraint.source_submap_id,
            constraint.target_submap_id,
            constraint.target_from_source.clone(),
            constraint.inlier_match_indices.len() as f64,
        );
    }
    let result = graph
        .optimize(config)
        .map_err(|error| HierarchicalSubmapGraphError::Optimization(error.to_string()))?;
    for node in hierarchy.nodes_mut() {
        node.local_from_atlas = graph.poses.get(&node.id).cloned();
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
    use visloc_core::geometry::{Pose, Sim3};

    use crate::{
        HierarchicalSfmConfig, LocalSubmap, LocalSubmapFrame, LocalSubmapLandmark,
        LocalSubmapObservation, LocalSubmapQuality, TrackBuildStats, VerifiedSubmapConstraint,
    };

    fn empty_submap(frame_id: u64) -> LocalSubmap {
        LocalSubmap {
            camera: Camera::pinhole(0, 64, 48, 50.0, 50.0, 32.0, 24.0),
            source_frame_ids: vec![frame_id],
            frames: vec![LocalSubmapFrame {
                local_frame_index: 0,
                source_frame_id: frame_id,
                pose: Pose::identity(),
            }],
            landmarks: Vec::new(),
            quality: LocalSubmapQuality {
                requested_images: 1,
                registered_images: 1,
                registration_fraction: 1.0,
                landmarks: 0,
                observations: 0,
                median_track_length: 0.0,
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
            track_build_stats: TrackBuildStats::default(),
            ba_result: None,
            seed_source_frame_i: frame_id,
            seed_source_frame_j: frame_id,
            seed_match_count: 0,
        }
    }

    fn constraint(from: u64, to: u64, measurement: Sim3, inliers: usize) -> SubmapSim3Constraint {
        SubmapSim3Constraint {
            source_submap_id: from,
            target_submap_id: to,
            target_from_source: measurement,
            correspondence_count: inliers,
            inlier_match_indices: (0..inliers).collect(),
            inlier_ratio: 1.0,
            mean_residual_ratio: 0.0,
            rotation_disagreement_deg: 0.0,
            leave_one_out_log_scale_mad: 0.0,
            target_scene_scale: 1.0,
        }
    }

    #[test]
    fn synthetic_two_submap_loop_edge_recovers_known_sim3() {
        let truth = Sim3::new(
            UnitQuaternion::from_euler_angles(0.12, -0.08, 0.2),
            Vector3::new(1.1, -0.4, 0.7),
            1.35,
        );
        let matches = (0..20)
            .map(|index| {
                let point = Point3::new(
                    (index % 5) as f64 * 0.4,
                    (index / 5) as f64 * 0.3,
                    ((index * 7) % 6) as f64 * 0.2,
                );
                SubmapPointMatch {
                    source_landmark_id: index,
                    target_landmark_id: index + 100,
                    source_point: point,
                    target_point: truth.transform_point(&point),
                }
            })
            .collect::<Vec<_>>();
        let recovered = estimate_submap_sim3_constraint(
            0,
            3,
            &matches,
            &truth.rotation,
            &SubmapSim3AlignmentConfig::default(),
        )
        .unwrap();
        assert!((recovered.target_from_source.scale - truth.scale).abs() < 1.0e-9);
        assert!(
            recovered
                .target_from_source
                .rotation
                .rotation_to(&truth.rotation)
                .angle()
                < 1.0e-9
        );
        assert!((recovered.target_from_source.translation - truth.translation).norm() < 1.0e-9);
    }

    #[test]
    fn post_ba_loop_pgo_improves_drifted_chain() {
        let mut hierarchy = HierarchicalSubmapGraph::new(0, empty_submap(0));
        for id in 1..4 {
            hierarchy.insert_independent(id, empty_submap(id)).unwrap();
        }
        let drifted_atlas_x = [0.0, 1.15, 2.45, 3.85];
        for node in hierarchy.nodes_mut() {
            node.local_from_atlas = Some(Sim3::new(
                UnitQuaternion::identity(),
                Vector3::new(-drifted_atlas_x[node.id as usize], 0.0, 0.0),
                1.0,
            ));
        }
        for id in 0..3 {
            hierarchy
                .add_constraint(VerifiedSubmapConstraint::Sim3(constraint(
                    id,
                    id + 1,
                    Sim3::identity(),
                    20,
                )))
                .unwrap();
        }
        let loop_edge = constraint(
            0,
            3,
            Sim3::new(
                UnitQuaternion::identity(),
                Vector3::new(-3.0, 0.0, 0.0),
                1.0,
            ),
            200,
        );
        let error = |hierarchy: &HierarchicalSubmapGraph| {
            hierarchy
                .nodes()
                .map(|node| {
                    let estimated = node
                        .atlas_from_local()
                        .unwrap()
                        .transform_point(&node.submap.frames[0].pose.camera_center_world())
                        .x;
                    (estimated - node.id as f64).powi(2)
                })
                .sum::<f64>()
                / 4.0
        };
        let before = error(&hierarchy);
        optimize_post_ba_pose_graph(
            &mut hierarchy,
            &[loop_edge],
            1,
            &Sim3PoseGraphConfig::default(),
        )
        .unwrap();
        let after = error(&hierarchy);
        assert!(after < before, "{before} -> {after}");
    }

    #[test]
    fn disabled_loop_closure_is_bit_identical() {
        let mut hierarchy = HierarchicalSubmapGraph::new(0, empty_submap(7));
        hierarchy.insert_independent(1, empty_submap(8)).unwrap();
        hierarchy.node(0).unwrap();
        let before = hierarchy
            .nodes()
            .map(|node| {
                (
                    node.id,
                    node.local_from_atlas.clone(),
                    node.submap.frames.clone(),
                    node.submap.landmarks.clone(),
                )
            })
            .collect::<Vec<_>>();
        // This is the exact caller-side off branch: no discovery, matching,
        // graph construction, or mutation is performed.
        let config = HierarchicalSfmConfig::default();
        let result = maybe_close_hierarchical_loops(
            config.submap_loop_closure,
            &mut hierarchy,
            &Camera::pinhole(0, 64, 48, 50.0, 50.0, 32.0, 24.0),
            &[],
            &[],
            config.submap_constraint_band,
            config.submap_loop_top_k,
            config.submap_loop_min_similarity,
            config.submap_loop_match_ratio,
            config.submap_loop_min_matches,
            config.submap_loop_verification_frame_budget,
            config.submap_loop_verification_correspondence_target,
            config.max_parallel_local_builds,
            &config.overlap,
            &config.alignment,
            &config.pose_graph,
        )
        .unwrap();
        assert!(result.is_none());
        let after = hierarchy
            .nodes()
            .map(|node| {
                (
                    node.id,
                    node.local_from_atlas.clone(),
                    node.submap.frames.clone(),
                    node.submap.landmarks.clone(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(before, after);
    }

    #[test]
    fn loop_verification_defaults_restore_rich_correspondence_sets() {
        let config = HierarchicalSfmConfig::default();
        assert_eq!(config.submap_loop_verification_frame_budget, 11);
        assert_eq!(config.submap_loop_verification_correspondence_target, 150);
        assert!(config.submap_loop_bundle_adjustment);
    }

    #[test]
    fn global_descriptor_ranking_keeps_top_k_per_submap_and_threshold_pairs() {
        // Eligible pairs use band=1. With top-K=1, each endpoint contributes
        // its strongest long-range partner. Pair 2..4 is deliberately below
        // both endpoint ranks but survives through the absolute threshold.
        let signatures = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.6, 0.8],
            vec![0.8, 0.6],
            vec![-1.0, 0.0],
        ];
        let ranked = rank_loop_candidates(&signatures, 1, 1, Some(-0.7));
        let pairs = ranked
            .iter()
            .map(|&(source, target, _)| (source, target))
            .collect::<BTreeSet<_>>();

        assert_eq!(eligible_pair_count(signatures.len(), 1), 6);
        assert!(pairs.contains(&(0, 2))); // endpoint 2's top partner
        assert!(pairs.contains(&(0, 3))); // endpoint 0's top partner
        assert!(pairs.contains(&(1, 3))); // endpoint 1's top partner
        assert!(pairs.contains(&(1, 4))); // endpoint 4's top partner
        assert!(pairs.contains(&(2, 4))); // threshold escape hatch (-0.6)
        assert!(!pairs.contains(&(0, 4))); // neither top-ranked nor thresholded
    }

    #[test]
    fn verification_frame_budget_selects_centered_eight_frame_spread() {
        assert_eq!(verification_frame_indices(65, 5), vec![32, 24, 40, 16, 48]);
        assert_eq!(verification_frame_indices(65, 1), vec![32]);
        assert_eq!(verification_frame_indices(65, 0), Vec::<usize>::new());
        assert_eq!(verification_frame_indices(0, 5), Vec::<usize>::new());
    }

    #[test]
    fn verification_frame_budget_clamps_at_registered_frame_boundaries() {
        assert_eq!(verification_frame_indices(9, 5), vec![4, 0, 8]);
        assert_eq!(verification_frame_indices(2, 5), vec![0, 1]);
    }

    #[allow(dead_code)]
    fn landmark(id: u64, point: Point3<f64>, frame_id: u64) -> LocalSubmapLandmark {
        LocalSubmapLandmark {
            local_landmark_id: id,
            position: point,
            observations: vec![LocalSubmapObservation {
                local_frame_index: 0,
                source_frame_id: frame_id,
                keypoint_index: id as usize,
                pixel: Point2::origin(),
            }],
        }
    }
}
