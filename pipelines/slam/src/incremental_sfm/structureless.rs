//! Structureless (two-view constraint) pose registration.

use super::*;

#[derive(Debug, Clone)]
pub(super) struct StructurelessConstraint {
    pub(super) neighbor: usize,
    pub(super) neighbor_center: Point3<f64>,
    pub(super) missing_rotation: UnitQuaternion<f64>,
    pub(super) center_direction: Vector3<f64>,
    pub(super) weight: f64,
}

#[derive(Debug, Clone)]
pub(super) struct StructurelessPoseProposal {
    pub(super) pose: Pose,
    pub(super) neighbor_spread: f64,
    pub(super) line_error_ratio: f64,
    pub(super) consensus_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum StructurelessRejection {
    TooFewNeighbors {
        found: usize,
        required: usize,
    },
    RotationDisagreement {
        max_deg: f64,
        allowed_deg: f64,
    },
    WeakCenterGeometry {
        max_angle_deg: f64,
        spread: f64,
    },
    NoCenterConsensus {
        rotation_consensus: usize,
    },
    SingularCenterFit,
    DirectionSign {
        neighbor: usize,
        along_ratio: f64,
        allowed: f64,
    },
    CenterLineResidual {
        ratio: f64,
        allowed: f64,
    },
}

/// Fit the missing camera centre to directed lines originating at registered
/// neighbour centres. Each line direction comes from an independently
/// recovered essential pose, while its origin carries the current model's
/// monocular scale. This is the scale-bearing part of structure-less recovery:
/// one line is deliberately under-constrained and is always rejected.
pub(super) fn solve_structureless_pose(
    constraints: &[StructurelessConstraint],
    config: &IncrementalSfmConfig,
) -> Result<StructurelessPoseProposal, StructurelessRejection> {
    let required_neighbors = config.structureless_min_neighbors.max(2);
    if constraints.len() < required_neighbors {
        return Err(StructurelessRejection::TooFewNeighbors {
            found: constraints.len(),
            required: required_neighbors,
        });
    }

    let max_rotation_rad = config
        .structureless_max_rotation_disagreement_deg
        .max(0.0)
        .to_radians();
    // A single bad essential edge must not veto an otherwise coherent set.
    // Enumerate every rotation as a deterministic consensus centre and keep
    // the largest, then highest-support, <=threshold subset.
    let mut consensus_indices = Vec::new();
    let mut consensus_weight = -1.0f64;
    let mut rotation_reference_index = None;
    for (reference_index, reference) in constraints.iter().enumerate() {
        let candidate: Vec<usize> = constraints
            .iter()
            .enumerate()
            .filter_map(|(index, constraint)| {
                ((reference.missing_rotation.inverse() * constraint.missing_rotation).angle()
                    <= max_rotation_rad)
                    .then_some(index)
            })
            .collect();
        let weight: f64 = candidate
            .iter()
            .map(|&index| constraints[index].weight)
            .sum();
        if candidate.len() > consensus_indices.len()
            || (candidate.len() == consensus_indices.len() && weight > consensus_weight)
            || (candidate.len() == consensus_indices.len()
                && weight.to_bits() == consensus_weight.to_bits()
                && candidate.first().copied().unwrap_or(reference_index)
                    < consensus_indices.first().copied().unwrap_or(usize::MAX))
        {
            consensus_indices = candidate;
            consensus_weight = weight;
            rotation_reference_index = Some(reference_index);
        }
    }
    if consensus_indices.len() < required_neighbors {
        let strongest = &constraints[0];
        let max_rotation_disagreement = constraints
            .iter()
            .map(|constraint| {
                (strongest.missing_rotation.inverse() * constraint.missing_rotation).angle()
            })
            .fold(0.0f64, f64::max);
        return Err(StructurelessRejection::RotationDisagreement {
            max_deg: max_rotation_disagreement.to_degrees(),
            allowed_deg: max_rotation_rad.to_degrees(),
        });
    }
    // Preserve the actual consensus centre. Choosing the strongest edge after
    // finding the set is not equivalent: two members can each lie within the
    // threshold of the centre yet be almost 2x the threshold apart.
    let reference_index = rotation_reference_index.expect("rotation consensus has a centre");
    let reference = &constraints[reference_index];

    let min_intersection_angle = config
        .structureless_min_intersection_angle_deg
        .max(0.0)
        .to_radians();
    let mut max_intersection_angle = 0.0f64;
    let mut rotation_consensus_spread = 0.0f64;
    for (position, &a_index) in consensus_indices.iter().enumerate() {
        let a = &constraints[a_index];
        for &b_index in consensus_indices.iter().skip(position + 1) {
            let b = &constraints[b_index];
            let cosine = a
                .center_direction
                .dot(&b.center_direction)
                .abs()
                .clamp(0.0, 1.0);
            max_intersection_angle = max_intersection_angle.max(cosine.acos());
            rotation_consensus_spread =
                rotation_consensus_spread.max((a.neighbor_center - b.neighbor_center).norm());
        }
    }
    if max_intersection_angle < min_intersection_angle || rotation_consensus_spread <= 1e-9 {
        return Err(StructurelessRejection::WeakCenterGeometry {
            max_angle_deg: max_intersection_angle.to_degrees(),
            spread: rotation_consensus_spread,
        });
    }

    let identity = Matrix3::identity();
    let fit_center = |indices: &[usize]| -> Option<Point3<f64>> {
        let mut normal = Matrix3::zeros();
        let mut rhs = Vector3::zeros();
        for &index in indices {
            let constraint = &constraints[index];
            let direction = constraint.center_direction.try_normalize(1e-12)?;
            let weight = constraint.weight.max(1.0);
            let projector = identity - direction * direction.transpose();
            normal += projector * weight;
            rhs += projector * constraint.neighbor_center.coords * weight;
        }
        Some(Point3::from(normal.try_inverse()? * rhs))
    };

    // Translation directions need their own robust consensus: agreeing
    // rotations do not imply that every essential decomposition has a reliable
    // baseline direction. Seed from every sufficiently non-parallel line pair,
    // score all rotation-consensus lines, then refit the largest 3+ set.
    let max_line_ratio = config.structureless_max_center_line_error_ratio.max(0.0);
    let mut center_consensus = Vec::new();
    let mut center_consensus_weight = -1.0f64;
    let mut center_consensus_error = f64::INFINITY;
    for (position, &a_index) in consensus_indices.iter().enumerate() {
        for &b_index in consensus_indices.iter().skip(position + 1) {
            let a = &constraints[a_index];
            let b = &constraints[b_index];
            let angle = a
                .center_direction
                .dot(&b.center_direction)
                .abs()
                .clamp(0.0, 1.0)
                .acos();
            if angle < min_intersection_angle {
                continue;
            }
            let Some(candidate_center) = fit_center(&[a_index, b_index]) else {
                continue;
            };
            let mut inliers = Vec::new();
            let mut squared_error = 0.0;
            let mut weight = 0.0;
            for &index in &consensus_indices {
                let constraint = &constraints[index];
                let displacement = candidate_center - constraint.neighbor_center;
                let along_ratio =
                    displacement.dot(&constraint.center_direction) / rotation_consensus_spread;
                let perpendicular = displacement
                    - constraint.center_direction * displacement.dot(&constraint.center_direction);
                let line_ratio = perpendicular.norm() / rotation_consensus_spread;
                if along_ratio >= config.structureless_min_forward_ratio
                    && line_ratio <= max_line_ratio
                {
                    inliers.push(index);
                    let edge_weight = constraint.weight.max(1.0);
                    squared_error += edge_weight * line_ratio * line_ratio;
                    weight += edge_weight;
                }
            }
            if inliers.len() < required_neighbors {
                continue;
            }
            let rms_error = (squared_error / weight.max(1.0)).sqrt();
            if inliers.len() > center_consensus.len()
                || (inliers.len() == center_consensus.len() && weight > center_consensus_weight)
                || (inliers.len() == center_consensus.len()
                    && weight.to_bits() == center_consensus_weight.to_bits()
                    && rms_error < center_consensus_error)
            {
                center_consensus = inliers;
                center_consensus_weight = weight;
                center_consensus_error = rms_error;
            }
        }
    }
    if center_consensus.len() < required_neighbors {
        return Err(StructurelessRejection::NoCenterConsensus {
            rotation_consensus: consensus_indices.len(),
        });
    }
    consensus_indices = center_consensus;
    // A weighted least-squares refit can move slightly outside the inlier set
    // that generated the winning two-line hypothesis. Reclassify after every
    // refit and discard only the inconsistent lines instead of allowing one
    // marginal edge to veto an otherwise valid 3+ neighbour consensus.
    // Removal is monotonic, so this converges in at most N iterations.
    let center = loop {
        let fitted =
            fit_center(&consensus_indices).ok_or(StructurelessRejection::SingularCenterFit)?;
        let retained: Vec<usize> = consensus_indices
            .iter()
            .copied()
            .filter(|&index| {
                let constraint = &constraints[index];
                let displacement = fitted - constraint.neighbor_center;
                let along_ratio =
                    displacement.dot(&constraint.center_direction) / rotation_consensus_spread;
                let perpendicular = displacement
                    - constraint.center_direction * displacement.dot(&constraint.center_direction);
                let line_ratio = perpendicular.norm() / rotation_consensus_spread;
                along_ratio >= config.structureless_min_forward_ratio
                    && line_ratio <= max_line_ratio
            })
            .collect();
        if retained.len() < required_neighbors {
            return Err(StructurelessRejection::NoCenterConsensus {
                rotation_consensus: consensus_indices.len(),
            });
        }
        if retained.len() == consensus_indices.len() {
            break fitted;
        }
        consensus_indices = retained;
    };
    let mut selected_neighbor_spread = 0.0f64;
    for (position, &a_index) in consensus_indices.iter().enumerate() {
        for &b_index in consensus_indices.iter().skip(position + 1) {
            selected_neighbor_spread = selected_neighbor_spread.max(
                (constraints[a_index].neighbor_center - constraints[b_index].neighbor_center)
                    .norm(),
            );
        }
    }
    if selected_neighbor_spread <= 1e-9 {
        return Err(StructurelessRejection::SingularCenterFit);
    }
    // Use the same rotation-consensus span used while scoring RANSAC centre
    // hypotheses. Switching to the smaller selected-subset span after refit
    // would make an inlier fail a stricter, inconsistent normalized gate.
    let neighbor_spread = rotation_consensus_spread;

    let mut weighted_squared_error = 0.0;
    let mut weight_sum = 0.0;
    for &index in &consensus_indices {
        let constraint = &constraints[index];
        let displacement = center - constraint.neighbor_center;
        // Essential decomposition resolves the sign through cheirality. A
        // negative line parameter means the multi-neighbour fit contradicts
        // that independent two-view geometry.
        let along = displacement.dot(&constraint.center_direction);
        let along_ratio = along / neighbor_spread;
        if along_ratio < config.structureless_min_forward_ratio {
            return Err(StructurelessRejection::DirectionSign {
                neighbor: constraint.neighbor,
                along_ratio,
                allowed: config.structureless_min_forward_ratio,
            });
        }
        let perpendicular = displacement
            - constraint.center_direction * displacement.dot(&constraint.center_direction);
        let weight = constraint.weight.max(1.0);
        weighted_squared_error += weight * perpendicular.norm_squared();
        weight_sum += weight;
    }
    let rms_line_error = (weighted_squared_error / weight_sum.max(1.0)).sqrt();
    let line_error_ratio = rms_line_error / neighbor_spread;
    if !line_error_ratio.is_finite()
        || line_error_ratio > config.structureless_max_center_line_error_ratio.max(0.0)
    {
        return Err(StructurelessRejection::CenterLineResidual {
            ratio: line_error_ratio,
            allowed: config.structureless_max_center_line_error_ratio.max(0.0),
        });
    }

    let rotation = reference.missing_rotation;
    let translation = -rotation.transform_vector(&center.coords);
    Ok(StructurelessPoseProposal {
        pose: Pose::from_world_to_camera(rotation, translation),
        neighbor_spread,
        line_error_ratio,
        consensus_indices,
    })
}

/// Return true when `new_pose` for `image` agrees with independent two-view
/// essentials against already-registered neighbours (same translation
/// hemisphere). With fewer than `min_neighbors` usable checks, accept.
#[allow(clippy::too_many_arguments)]
pub(super) fn pose_agrees_with_two_view_neighbors(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    poses: &[Option<Pose>],
    image: usize,
    new_pose: &Pose,
    min_neighbors: usize,
    min_agree_fraction: f64,
) -> bool {
    let estimator = RelativePoseEstimator::default();
    let new_c = new_pose.camera_center_world();
    let mut checked = 0usize;
    let mut agree = 0usize;
    for pair in pairwise {
        let (neighbor, matches_ij, new_is_i) = if pair.image_i == image {
            (pair.image_j, &pair.matches, true)
        } else if pair.image_j == image {
            (pair.image_i, &pair.matches, false)
        } else {
            continue;
        };
        let Some(neighbor_pose) = poses.get(neighbor).and_then(|p| p.as_ref()) else {
            continue;
        };
        if matches_ij.len() < 16 {
            continue;
        }
        let correspondences: Vec<TwoViewCorrespondence> = matches_ij
            .iter()
            .filter_map(|&(ki, kj)| {
                Some(TwoViewCorrespondence::new(
                    *features[pair.image_i].keypoints.get(ki)?,
                    *features[pair.image_j].keypoints.get(kj)?,
                ))
            })
            .collect();
        let Some(relative) = estimator.estimate(&correspondences, camera) else {
            continue;
        };
        if relative.inliers.len() < 16 {
            continue;
        }
        // Two-view: camera-j centre direction in camera-i frame.
        let r_ij = relative.previous_to_current.rotation;
        let t_ij = relative.previous_to_current.translation;
        let Some(dir_i_to_j) = (-r_ij.inverse().transform_vector(&t_ij)).try_normalize(1e-12)
        else {
            continue;
        };
        let neighbor_c = neighbor_pose.camera_center_world();
        let abs_agree = if new_is_i {
            // Absolute: neighbour in new (image_i) frame.
            let Some(abs_dir) = new_pose
                .world_to_camera
                .rotation
                .transform_vector(&(neighbor_c - new_c))
                .try_normalize(1e-12)
            else {
                continue;
            };
            // Two-view dir is i→j = new→neighbor.
            abs_dir.dot(&dir_i_to_j) > 0.0
        } else {
            // Absolute: new in neighbour (image_i) frame.
            let Some(abs_dir) = neighbor_pose
                .world_to_camera
                .rotation
                .transform_vector(&(new_c - neighbor_c))
                .try_normalize(1e-12)
            else {
                continue;
            };
            abs_dir.dot(&dir_i_to_j) > 0.0
        };
        checked += 1;
        if abs_agree {
            agree += 1;
        }
    }
    if checked < min_neighbors {
        return true;
    }
    (agree as f64 / checked as f64) >= min_agree_fraction
}

fn estimate_structureless_constraints(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    poses: &[Option<Pose>],
    missing: usize,
    config: &IncrementalSfmConfig,
) -> Vec<StructurelessConstraint> {
    let estimator = RelativePoseEstimator::default();
    let mut constraints = Vec::new();
    for pair in pairwise {
        let (neighbor, invert) = if pair.image_j == missing && poses[pair.image_i].is_some() {
            (pair.image_i, false)
        } else if pair.image_i == missing && poses[pair.image_j].is_some() {
            (pair.image_j, true)
        } else {
            continue;
        };
        let Some(neighbor_pose) = poses[neighbor].as_ref() else {
            continue;
        };
        let mut correspondences = Vec::with_capacity(pair.matches.len());
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let (Some(pixel_i), Some(pixel_j)) = (
                features[pair.image_i].keypoints.get(keypoint_i),
                features[pair.image_j].keypoints.get(keypoint_j),
            ) else {
                continue;
            };
            correspondences.push(TwoViewCorrespondence::new(*pixel_i, *pixel_j));
        }
        let Some(relative) = estimator.estimate(&correspondences, camera) else {
            continue;
        };
        if relative.inliers.len() < config.structureless_min_pair_inliers {
            continue;
        }
        let neighbor_to_missing = if invert {
            relative.previous_to_current.inverse()
        } else {
            relative.previous_to_current
        };
        let missing_rotation =
            neighbor_to_missing.rotation * neighbor_pose.world_to_camera.rotation;
        let Some(center_direction) = (-missing_rotation
            .inverse()
            .transform_vector(&neighbor_to_missing.translation))
        .try_normalize(1e-12) else {
            continue;
        };
        constraints.push(StructurelessConstraint {
            neighbor,
            neighbor_center: neighbor_pose.camera_center_world(),
            missing_rotation,
            center_direction,
            weight: relative.inliers.len() as f64,
        });
    }
    constraints.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.neighbor.cmp(&b.neighbor))
    });
    constraints
}

fn mean_reprojection_for_registered_mask(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
    registered_mask: &[bool],
    point_mask: &[bool],
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (track_id, track) in tracks.iter().enumerate() {
        if !point_mask.get(track_id).copied().unwrap_or(false) {
            continue;
        }
        let Some(point) = track_point.get(track_id).and_then(Option::as_ref) else {
            continue;
        };
        for &(image, keypoint) in track {
            if !registered_mask.get(image).copied().unwrap_or(false) {
                continue;
            }
            let (Some(pose), Some(pixel)) = (
                poses.get(image).and_then(Option::as_ref),
                features
                    .get(image)
                    .and_then(|set| set.keypoints.get(keypoint)),
            ) else {
                continue;
            };
            if let Some(error) = reprojection_error_px(camera, pose, point, pixel) {
                sum += error;
                count += 1;
            }
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn supported_tracks_for_image(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
    image: usize,
    max_error: f64,
) -> (usize, f64) {
    let Some(pose) = poses.get(image).and_then(Option::as_ref) else {
        return (0, f64::NAN);
    };
    let mut count = 0usize;
    let mut sum = 0.0;
    for (track_id, track) in tracks.iter().enumerate() {
        let Some(point) = track_point.get(track_id).and_then(Option::as_ref) else {
            continue;
        };
        let Some((_, keypoint)) = track.iter().find(|(track_image, _)| *track_image == image)
        else {
            continue;
        };
        let Some(pixel) = features[image].keypoints.get(*keypoint) else {
            continue;
        };
        let Some(error) = reprojection_error_px(camera, pose, point, pixel) else {
            continue;
        };
        if error <= max_error {
            count += 1;
            sum += error;
        }
    }
    if count == 0 {
        (0, f64::NAN)
    } else {
        (count, sum / count as f64)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct StructurelessPoseConsistency {
    pub(super) accepted: bool,
    pub(super) max_rotation_deg: f64,
    min_forward_ratio: f64,
    line_error_ratio: f64,
}

#[cfg(test)]
pub(super) fn interpolate_structureless_pose(from: &Pose, to: &Pose, alpha: f64) -> Pose {
    interpolate_structureless_pose_components(from, to, alpha, alpha)
}

fn interpolate_structureless_pose_components(
    from: &Pose,
    to: &Pose,
    rotation_alpha: f64,
    center_alpha: f64,
) -> Pose {
    let rotation_alpha = rotation_alpha.clamp(0.0, 1.0);
    let center_alpha = center_alpha.clamp(0.0, 1.0);
    if rotation_alpha <= 0.0 && center_alpha <= 0.0 {
        return from.clone();
    }
    if rotation_alpha >= 1.0 && center_alpha >= 1.0 {
        return to.clone();
    }
    let rotation = from
        .world_to_camera
        .rotation
        .slerp(&to.world_to_camera.rotation, rotation_alpha);
    let from_center = from.camera_center_world();
    let to_center = to.camera_center_world();
    let center =
        Point3::from(from_center.coords * (1.0 - center_alpha) + to_center.coords * center_alpha);
    let translation = -rotation.transform_vector(&center.coords);
    Pose::from_world_to_camera(rotation, translation)
}

pub(super) fn structureless_pose_consistency(
    pose: &Pose,
    constraints: &[StructurelessConstraint],
    proposal: &StructurelessPoseProposal,
    config: &IncrementalSfmConfig,
) -> StructurelessPoseConsistency {
    let center = pose.camera_center_world();
    let max_rotation = config
        .structureless_max_rotation_disagreement_deg
        .max(0.0)
        .to_radians();
    let mut weighted_squared_error = 0.0;
    let mut weight_sum = 0.0;
    let mut max_rotation_seen = 0.0f64;
    let mut min_forward_seen = f64::INFINITY;
    for &index in &proposal.consensus_indices {
        let constraint = &constraints[index];
        let rotation_error =
            (constraint.missing_rotation.inverse() * pose.world_to_camera.rotation).angle();
        max_rotation_seen = max_rotation_seen.max(rotation_error);
        let displacement = center - constraint.neighbor_center;
        let forward_ratio =
            displacement.dot(&constraint.center_direction) / proposal.neighbor_spread;
        min_forward_seen = min_forward_seen.min(forward_ratio);
        let perpendicular = displacement
            - constraint.center_direction * displacement.dot(&constraint.center_direction);
        let weight = constraint.weight.max(1.0);
        weighted_squared_error += weight * perpendicular.norm_squared();
        weight_sum += weight;
    }
    let ratio =
        (weighted_squared_error / weight_sum.max(1.0)).sqrt() / proposal.neighbor_spread.max(1e-12);
    StructurelessPoseConsistency {
        accepted: max_rotation_seen <= max_rotation
            && min_forward_seen >= config.structureless_min_forward_ratio
            && ratio.is_finite()
            && ratio <= config.structureless_max_center_line_error_ratio.max(0.0),
        max_rotation_deg: max_rotation_seen.to_degrees(),
        min_forward_ratio: min_forward_seen,
        line_error_ratio: ratio,
    }
}

/// Build an independent local submap from verified pairwise edges that were
/// not retained by the global union-find tracks. Observations already owned by
/// a global 3D track are never duplicated. Each new point must be seen by the
/// missing image and the configured number of registered consensus neighbours,
/// triangulate with sufficient parallax, and reproject within the initialization
/// gate in every contributing view.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn build_structureless_local_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &[Vec<(usize, usize)>],
    track_point: &[Option<Point3<f64>>],
    poses: &[Option<Pose>],
    missing: usize,
    constraints: &[StructurelessConstraint],
    proposal: &StructurelessPoseProposal,
    config: &IncrementalSfmConfig,
) -> Vec<(Vec<(usize, usize)>, Point3<f64>)> {
    let allowed_neighbors: HashSet<usize> = proposal
        .consensus_indices
        .iter()
        .map(|&index| constraints[index].neighbor)
        .collect();
    let occupied: HashSet<(usize, usize)> = tracks
        .iter()
        .enumerate()
        .filter(|(track_id, _)| track_point.get(*track_id).is_some_and(Option::is_some))
        .flat_map(|(_, track)| track.iter().copied())
        .collect();
    let mut by_missing_keypoint: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for pair in pairwise {
        let (neighbor, missing_first) = if pair.image_i == missing
            && allowed_neighbors.contains(&pair.image_j)
            && poses[pair.image_j].is_some()
        {
            (pair.image_j, true)
        } else if pair.image_j == missing
            && allowed_neighbors.contains(&pair.image_i)
            && poses[pair.image_i].is_some()
        {
            (pair.image_i, false)
        } else {
            continue;
        };
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let (missing_keypoint, neighbor_keypoint) = if missing_first {
                (keypoint_i, keypoint_j)
            } else {
                (keypoint_j, keypoint_i)
            };
            if features[missing].keypoints.get(missing_keypoint).is_none()
                || features[neighbor]
                    .keypoints
                    .get(neighbor_keypoint)
                    .is_none()
                || occupied.contains(&(missing, missing_keypoint))
                || occupied.contains(&(neighbor, neighbor_keypoint))
            {
                continue;
            }
            by_missing_keypoint
                .entry(missing_keypoint)
                .or_default()
                .push((neighbor, neighbor_keypoint));
        }
    }

    let mut missing_keypoints: Vec<usize> = by_missing_keypoint.keys().copied().collect();
    missing_keypoints.sort_unstable();
    let mut local_tracks = Vec::new();
    let mut claimed_observations = HashSet::new();
    for missing_keypoint in missing_keypoints {
        let mut neighbors = by_missing_keypoint.remove(&missing_keypoint).unwrap();
        neighbors.sort_unstable();
        neighbors.dedup_by_key(|observation| observation.0);
        let required_registered_views = config
            .structureless_min_local_track_views
            .max(2)
            .saturating_sub(1);
        if neighbors.len() < required_registered_views {
            continue;
        }
        let mut observations = vec![(missing, missing_keypoint)];
        observations.extend(neighbors);
        observations.sort_unstable();
        if observations
            .iter()
            .any(|observation| claimed_observations.contains(observation))
        {
            continue;
        }
        let pixels: Vec<(usize, Point2<f64>)> = observations
            .iter()
            .map(|&(image, keypoint)| (image, features[image].keypoints[keypoint]))
            .collect();
        let Some(point) = triangulate_track(camera, poses, &pixels, config) else {
            continue;
        };
        let mut valid = true;
        for &(image, keypoint) in &observations {
            let Some(error) = reprojection_error_px(
                camera,
                poses[image].as_ref().unwrap(),
                &point,
                &features[image].keypoints[keypoint],
            ) else {
                valid = false;
                break;
            };
            // This is an initialization gate only. The point is subsequently
            // refined in the fixed-pose local submap and must still clear the
            // stricter structure-less admission error below.
            if error > config.max_reprojection_error_px {
                valid = false;
                break;
            }
        }
        if valid {
            claimed_observations.extend(observations.iter().copied());
            local_tracks.push((observations, point));
            if local_tracks.len() >= config.structureless_max_local_tracks.max(1) {
                break;
            }
        }
    }
    local_tracks
}

/// Run [`structureless_registration_pass`] repeatedly until a round registers
/// nothing, the budget [`IncrementalSfmConfig::structureless_max_rounds`] is
/// spent, or every image is registered. Each round scans in the same fixed
/// ascending image order, so the loop is deterministic; images registered by
/// earlier rounds act as neighbours for later ones, which is what lets an
/// island chain inward through its bridge even when the bridge's index is
/// higher than the images it unlocks.
pub(super) fn structureless_registration_rounds(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &mut Vec<Vec<(usize, usize)>>,
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut Vec<Option<Point3<f64>>>,
) -> usize {
    let mut total = 0usize;
    let max_rounds = config.structureless_max_rounds.max(1);
    for round in 0..max_rounds {
        if !poses.iter().any(Option::is_none) {
            break;
        }
        let registered = structureless_registration_pass(
            camera,
            features,
            pairwise,
            tracks,
            config,
            poses,
            track_point,
        );
        if sfm_debug_enabled() && registered > 0 {
            eprintln!("sfm-debug: structure-less round {round} registered {registered} image(s)");
        }
        total += registered;
        if registered == 0 {
            break;
        }
    }
    total
}

/// One bounded multi-neighbour recovery sweep. Each missing image is attempted
/// at most once. Failed geometry, local BA, or admission gates restore the
/// complete pose/point state byte-for-byte before moving on.
fn structureless_registration_pass(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &mut Vec<Vec<(usize, usize)>>,
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut Vec<Option<Point3<f64>>>,
) -> usize {
    let missing_images: Vec<usize> = poses
        .iter()
        .enumerate()
        .filter_map(|(image, pose)| pose.is_none().then_some(image))
        .collect();
    let mut registered = 0usize;
    for image in missing_images {
        let constraints =
            estimate_structureless_constraints(camera, features, pairwise, poses, image, config);
        let proposal = match solve_structureless_pose(&constraints, config) {
            Ok(proposal) => proposal,
            Err(reason) => {
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: structure-less image {image} rejected before insertion \
                         ({} registered relative neighbours): {reason:?}",
                        constraints.len(),
                    );
                }
                continue;
            }
        };
        let tracks_before = tracks.clone();
        let tracks_before_len = tracks_before.len();
        let poses_before = poses.to_vec();
        let points_before = track_point.to_vec();
        let registered_mask: Vec<bool> = poses_before.iter().map(Option::is_some).collect();
        let clean_point_mask: Vec<bool> = points_before.iter().map(Option::is_some).collect();
        let clean_mean_before = mean_reprojection_for_registered_mask(
            camera,
            features,
            tracks,
            &poses_before,
            &points_before,
            &registered_mask,
            &clean_point_mask,
        );
        poses[image] = Some(proposal.pose.clone());
        let local_tracks = build_structureless_local_tracks(
            camera,
            features,
            pairwise,
            tracks,
            track_point,
            poses,
            image,
            &constraints,
            &proposal,
            config,
        );
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: structure-less image {image} synthesized {} independent local tracks",
                local_tracks.len()
            );
        }
        let local_observations: HashSet<(usize, usize)> = local_tracks
            .iter()
            .flat_map(|(track, _)| track.iter().copied())
            .collect();
        for (track_id, track) in tracks.iter_mut().enumerate().take(tracks_before_len) {
            if track_point[track_id].is_none() {
                track.retain(|observation| !local_observations.contains(observation));
            }
        }
        for (track, point) in local_tracks {
            tracks.push(track);
            track_point.push(Some(point));
        }
        triangulate_pending_with_config(camera, features, tracks, poses, config, track_point);
        let proposal_poses = poses.to_vec();
        let proposal_points = track_point.to_vec();
        let (proposal_support, proposal_image_mean) = supported_tracks_for_image(
            camera,
            features,
            tracks,
            &proposal_poses,
            &proposal_points,
            image,
            config.max_reprojection_error_px,
        );
        let proposal_clean_mean = mean_reprojection_for_registered_mask(
            camera,
            features,
            tracks,
            &proposal_poses,
            &proposal_points,
            &registered_mask,
            &clean_point_mask,
        );
        let proposal_consistency = proposal_poses[image]
            .as_ref()
            .map(|pose| structureless_pose_consistency(pose, &constraints, &proposal, config));
        // A structure-less proposal is already tied to the registered map by
        // several independently estimated relative poses.  Moving its
        // neighbours in the ordinary growth local-BA window can trade that
        // scale-bearing consensus for a lower pixel residual (MH_05 image 86
        // exposed exactly that failure).  Refine only the recovered pose and
        // its incident landmarks; every previously registered observer stays
        // fixed and therefore acts as the local-submap alignment boundary.
        let mut structureless_variable = HashSet::new();
        structureless_variable.insert(image);
        let local_result = bundle_adjust_local(
            camera,
            features,
            tracks,
            config,
            poses,
            track_point,
            &structureless_variable,
        );
        let refined_poses = poses.to_vec();
        let allowed_clean_mean = clean_mean_before
            * (1.0 + config.structureless_max_clean_error_increase_ratio.max(0.0));
        let (mut support, mut image_mean) = supported_tracks_for_image(
            camera,
            features,
            tracks,
            poses,
            track_point,
            image,
            config.max_reprojection_error_px,
        );
        let mut clean_mean_after = mean_reprojection_for_registered_mask(
            camera,
            features,
            tracks,
            poses,
            track_point,
            &registered_mask,
            &clean_point_mask,
        );
        let mut pose_consistency = poses[image]
            .as_ref()
            .map(|pose| structureless_pose_consistency(pose, &constraints, &proposal, config));
        let local_ok = local_result.is_ok();
        let mut support_ok = support >= config.structureless_min_support_tracks;
        let mut image_error_ok =
            image_mean.is_finite() && image_mean <= config.structureless_max_reprojection_error_px;
        let mut clean_ok = clean_mean_before.is_finite()
            && clean_mean_after.is_finite()
            && clean_mean_after <= allowed_clean_mean + 1e-12;
        let mut geometry_ok = pose_consistency.is_some_and(|diagnostic| diagnostic.accepted);
        let mut accepted = local_ok && support_ok && image_error_ok && clean_ok && geometry_ok;
        let mut trust_region_alpha = None;

        // The unconstrained local BA can cross the independently measured
        // relative-geometry boundary while greatly improving reprojection.
        // Search back along the camera part of that BA update. For each pose
        // inside the relative-geometry feasible region, re-solve only the new
        // landmarks against fixed cameras, then commit the largest step that
        // satisfies every admission gate. This is a bounded deterministic
        // local-submap projection, not a relaxed threshold.
        if local_ok && !accepted {
            let proposal_pose = proposal_poses[image].as_ref().unwrap();
            let refined_pose = refined_poses[image].as_ref().unwrap();
            let mut trust_candidates = Vec::with_capacity(400);
            for rotation_step in (0..20).rev() {
                for center_step in (0..20).rev() {
                    trust_candidates.push((rotation_step as f64 / 20.0, center_step as f64 / 20.0));
                }
            }
            let mut candidate_index = 0usize;
            let mut best_near_candidate: Option<(f64, f64, f64)> = None;
            let mut fine_candidates_enqueued = false;
            'trust_region: while candidate_index < trust_candidates.len() {
                let (rotation_alpha, center_alpha) = trust_candidates[candidate_index];
                candidate_index += 1;
                let mut candidate_poses = proposal_poses.clone();
                candidate_poses[image] = Some(interpolate_structureless_pose_components(
                    proposal_pose,
                    refined_pose,
                    rotation_alpha,
                    center_alpha,
                ));
                let candidate_consistency = structureless_pose_consistency(
                    candidate_poses[image].as_ref().unwrap(),
                    &constraints,
                    &proposal,
                    config,
                );
                // Local tracks triangulated at the unconstrained proposal may
                // be invalid at the projected pose (and vice versa). Rebuild
                // the bounded submap at each geometry-feasible trust-region
                // pose from the pre-insertion state. This keeps landmark
                // synthesis consistent with the camera pose being admitted.
                let mut candidate_tracks = tracks_before.clone();
                let mut candidate_points = points_before.clone();
                let candidate_local_tracks = if candidate_consistency.accepted {
                    build_structureless_local_tracks(
                        camera,
                        features,
                        pairwise,
                        &candidate_tracks,
                        &candidate_points,
                        &candidate_poses,
                        image,
                        &constraints,
                        &proposal,
                        config,
                    )
                } else {
                    Vec::new()
                };
                let candidate_local_observations: HashSet<(usize, usize)> = candidate_local_tracks
                    .iter()
                    .flat_map(|(track, _)| track.iter().copied())
                    .collect();
                for (track_id, track) in candidate_tracks.iter_mut().enumerate() {
                    if candidate_points[track_id].is_none() {
                        track.retain(|observation| {
                            !candidate_local_observations.contains(observation)
                        });
                    }
                }
                for (track, point) in candidate_local_tracks {
                    candidate_tracks.push(track);
                    candidate_points.push(Some(point));
                }
                if candidate_consistency.accepted {
                    triangulate_pending_with_config(
                        camera,
                        features,
                        &candidate_tracks,
                        &candidate_poses,
                        config,
                        &mut candidate_points,
                    );
                }
                let submap_ok = candidate_consistency.accepted
                    && refine_structureless_new_landmarks(
                        camera,
                        features,
                        &candidate_tracks,
                        config,
                        &candidate_poses,
                        &mut candidate_points,
                        image,
                        &clean_point_mask,
                    )
                    .is_ok();
                let (candidate_support, candidate_image_mean) = supported_tracks_for_image(
                    camera,
                    features,
                    &candidate_tracks,
                    &candidate_poses,
                    &candidate_points,
                    image,
                    config.max_reprojection_error_px,
                );
                let candidate_clean_mean = mean_reprojection_for_registered_mask(
                    camera,
                    features,
                    &candidate_tracks,
                    &candidate_poses,
                    &candidate_points,
                    &registered_mask,
                    &clean_point_mask,
                );
                let candidate_ok = submap_ok
                    && candidate_support >= config.structureless_min_support_tracks
                    && candidate_image_mean.is_finite()
                    && candidate_image_mean <= config.structureless_max_reprojection_error_px
                    && clean_mean_before.is_finite()
                    && candidate_clean_mean.is_finite()
                    && candidate_clean_mean <= allowed_clean_mean + 1e-12
                    && candidate_consistency.accepted;
                let near_candidate = submap_ok
                    && candidate_support >= config.structureless_min_support_tracks
                    && candidate_image_mean.is_finite()
                    && clean_mean_before.is_finite()
                    && candidate_clean_mean.is_finite()
                    && candidate_clean_mean <= allowed_clean_mean + 1e-12
                    && candidate_consistency.accepted;
                if near_candidate
                    && best_near_candidate
                        .is_none_or(|(_, _, best_mean)| candidate_image_mean < best_mean)
                {
                    best_near_candidate =
                        Some((rotation_alpha, center_alpha, candidate_image_mean));
                }
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: structure-less image {image} trust rotation-alpha={rotation_alpha:.2} \
                         center-alpha={center_alpha:.2} \
                         tracks={} support={candidate_support} mean={candidate_image_mean:.3}px \
                         clean={candidate_clean_mean:.6} rot={:.3}deg forward={:.4} \
                         line={:.4} submap-ok={submap_ok} accepted={candidate_ok}",
                        candidate_tracks.len().saturating_sub(tracks_before_len),
                        candidate_consistency.max_rotation_deg,
                        candidate_consistency.min_forward_ratio,
                        candidate_consistency.line_error_ratio,
                    );
                }
                if candidate_ok {
                    poses.clone_from_slice(&candidate_poses);
                    *tracks = candidate_tracks;
                    *track_point = candidate_points;
                    support = candidate_support;
                    image_mean = candidate_image_mean;
                    clean_mean_after = candidate_clean_mean;
                    pose_consistency = Some(candidate_consistency);
                    support_ok = true;
                    image_error_ok = true;
                    clean_ok = true;
                    geometry_ok = true;
                    accepted = true;
                    trust_region_alpha = Some((rotation_alpha, center_alpha));
                    break 'trust_region;
                }
                if candidate_index == trust_candidates.len() && !fine_candidates_enqueued {
                    fine_candidates_enqueued = true;
                    if let Some((best_rotation, best_center, _)) = best_near_candidate {
                        let rotation_percent = (best_rotation * 100.0).round() as i32;
                        let center_percent = (best_center * 100.0).round() as i32;
                        for fine_rotation in
                            ((rotation_percent - 5).max(0)..=(rotation_percent + 5).min(100)).rev()
                        {
                            for fine_center in
                                ((center_percent - 5).max(0)..=(center_percent + 5).min(100)).rev()
                            {
                                if fine_rotation % 5 != 0 || fine_center % 5 != 0 {
                                    trust_candidates.push((
                                        fine_rotation as f64 / 100.0,
                                        fine_center as f64 / 100.0,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        let proposal_accepted = proposal_support >= config.structureless_min_support_tracks
            && proposal_image_mean.is_finite()
            && proposal_image_mean <= config.structureless_max_reprojection_error_px
            && clean_mean_before.is_finite()
            && proposal_clean_mean.is_finite()
            && proposal_clean_mean <= allowed_clean_mean + 1e-12
            && proposal_consistency.is_some_and(|diagnostic| diagnostic.accepted);
        if accepted {
            registered += 1;
            if sfm_debug_enabled() {
                if let Some((rotation_alpha, center_alpha)) = trust_region_alpha {
                    eprintln!(
                        "sfm-debug: structure-less image {image} projected BA step \
                         to trust-region rotation-alpha={rotation_alpha:.2} \
                         center-alpha={center_alpha:.2}"
                    );
                }
                eprintln!(
                    "sfm-debug: structure-less registered image {image} \
                     (neighbors={} line-ratio={:.4} support={} mean={:.3}px \
                     clean={:.6}->{:.6})",
                    proposal.consensus_indices.len(),
                    proposal.line_error_ratio,
                    support,
                    image_mean,
                    clean_mean_before,
                    clean_mean_after,
                );
            }
        } else if proposal_accepted {
            // Local BA is optional for admission: if it leaves the independent
            // relative-pose consensus, retain the already-gated scale-bearing
            // proposal and its newly triangulated structure, not the BA drift.
            poses.clone_from_slice(&proposal_poses);
            track_point.clone_from_slice(&proposal_points);
            registered += 1;
            if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: structure-less registered image {image} pose-only \
                     (neighbors={} line-ratio={:.4} support={} mean={:.3}px \
                     clean={:.6}->{:.6}; local BA rejected)",
                    proposal.consensus_indices.len(),
                    proposal.line_error_ratio,
                    proposal_support,
                    proposal_image_mean,
                    clean_mean_before,
                    proposal_clean_mean,
                );
            }
        } else {
            *tracks = tracks_before;
            track_point.truncate(points_before.len());
            poses.clone_from_slice(&poses_before);
            track_point.clone_from_slice(&points_before);
            if sfm_debug_enabled() {
                let (pose_rotation_deg, pose_forward_ratio, pose_line_ratio) = pose_consistency
                    .map_or((f64::NAN, f64::NAN, f64::NAN), |diagnostic| {
                        (
                            diagnostic.max_rotation_deg,
                            diagnostic.min_forward_ratio,
                            diagnostic.line_error_ratio,
                        )
                    });
                eprintln!(
                    "sfm-debug: structure-less image {image} rolled back \
                     (neighbors={} line-ratio={:.4} support={} mean={:.3}px \
                     clean={:.6}->{:.6} allowed={:.6} local-ok={} support-ok={} \
                     image-ok={} clean-ok={} geometry-ok={} pose-rot={:.3}deg \
                     pose-forward={:.4} pose-line={:.4}; proposal-support={} \
                     proposal-mean={:.3}px proposal-clean={:.6} proposal-ok={})",
                    proposal.consensus_indices.len(),
                    proposal.line_error_ratio,
                    support,
                    image_mean,
                    clean_mean_before,
                    clean_mean_after,
                    allowed_clean_mean,
                    local_ok,
                    support_ok,
                    image_error_ok,
                    clean_ok,
                    geometry_ok,
                    pose_rotation_deg,
                    pose_forward_ratio,
                    pose_line_ratio,
                    proposal_support,
                    proposal_image_mean,
                    proposal_clean_mean,
                    proposal_accepted,
                );
            }
        }
    }
    registered
}
