use nalgebra::{Matrix3, Point2, Point3};
use std::collections::{HashMap, HashSet};

use super::{Camera, CameraId, FrameId, Keyframe, Observation};
use crate::geometry::SE3;

pub type LandmarkId = u64;

#[derive(Debug, Clone, PartialEq)]
pub struct Landmark {
    pub id: LandmarkId,
    pub position: Point3<f64>,
    pub descriptor: Option<Vec<f32>>,
    pub observations: Vec<Observation>,
}

impl Landmark {
    pub fn new(id: LandmarkId, position: Point3<f64>) -> Self {
        Self {
            id,
            position,
            descriptor: None,
            observations: Vec::new(),
        }
    }
}

/// Calibrated right-image measurement paired with the normal left-camera
/// [`Observation`] stored on a keyframe. This preserves the full stereo
/// measurement without assuming that the rig is rectified.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoObservation {
    pub frame_id: FrameId,
    pub landmark_id: LandmarkId,
    pub right_camera_id: CameraId,
    pub xy_right: Point2<f64>,
    /// Fixed rig transform `T_right<-left` for this measurement.
    pub left_to_right: SE3,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct VisualMap {
    pub cameras: HashMap<CameraId, Camera>,
    pub landmarks: HashMap<LandmarkId, Landmark>,
    /// Optional world-frame position covariance for landmarks whose source
    /// geometry provides one (for example a calibrated stereo seed).
    pub landmark_position_covariances: HashMap<LandmarkId, Matrix3<f64>>,
    /// Right-image measurements paired by `(frame_id, landmark_id)` with the
    /// ordinary left observations in `keyframes` / `landmarks`.
    pub stereo_observations: Vec<StereoObservation>,
    pub keyframes: HashMap<u64, Keyframe>,
    /// Optional learned/matcher confidence keyed by the canonical observation
    /// identity `(frame_id, landmark_id, keypoint_index)`. Classical
    /// observations are absent and therefore retain uniform weight. Kept
    /// separate from [`Observation`] so map geometry and feature-index formats
    /// remain backward compatible with existing importers.
    observation_confidences: HashMap<(FrameId, LandmarkId, usize), f32>,
}

impl VisualMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a normalized confidence to an observation. Returns false and
    /// leaves the map unchanged for NaN, infinity, or values outside `[0, 1]`.
    pub fn set_observation_confidence(
        &mut self,
        observation: &Observation,
        confidence: f32,
    ) -> bool {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return false;
        }
        self.observation_confidences.insert(
            (
                observation.frame_id,
                observation.landmark_id,
                observation.keypoint_index,
            ),
            confidence,
        );
        true
    }

    pub fn observation_confidence(&self, observation: &Observation) -> Option<f32> {
        self.observation_confidences
            .get(&(
                observation.frame_id,
                observation.landmark_id,
                observation.keypoint_index,
            ))
            .copied()
    }

    pub fn remove_observation_confidence(&mut self, observation: &Observation) -> Option<f32> {
        self.observation_confidences.remove(&(
            observation.frame_id,
            observation.landmark_id,
            observation.keypoint_index,
        ))
    }

    pub fn observation_confidence_count(&self) -> usize {
        self.observation_confidences.len()
    }

    /// `(min, mean, max)` over stored finite confidence values.
    pub fn observation_confidence_stats(&self) -> Option<(f32, f32, f32)> {
        let mut values = self
            .observation_confidences
            .values()
            .copied()
            .filter(|value| value.is_finite());
        let first = values.next()?;
        let mut min = first;
        let mut max = first;
        let mut sum = first as f64;
        let mut count = 1usize;
        for value in values {
            min = min.min(value);
            max = max.max(value);
            sum += value as f64;
            count += 1;
        }
        Some((min, (sum / count as f64) as f32, max))
    }

    pub fn validate(&self) -> VisualMapValidationReport {
        let mut report = VisualMapValidationReport::default();
        self.validate_structure_into(&mut report);
        report
    }

    pub fn validate_with_descriptors(
        &self,
        descriptor_store: Option<&LandmarkDescriptorStore>,
    ) -> VisualMapValidationReport {
        let mut report = self.validate();
        self.validate_descriptors_into(descriptor_store, &mut report);
        report
    }

    fn validate_structure_into(&self, report: &mut VisualMapValidationReport) {
        let mut keyframes = self.keyframes.iter().collect::<Vec<_>>();
        keyframes.sort_by_key(|(keyframe_id, _)| **keyframe_id);
        for (keyframe_id, keyframe) in keyframes {
            if *keyframe_id != keyframe.frame.id {
                report.push(VisualMapValidationIssue::KeyframeIdMismatch {
                    keyframe_id: *keyframe_id,
                    frame_id: keyframe.frame.id,
                });
            }

            if !self.cameras.contains_key(&keyframe.frame.camera_id) {
                report.push(VisualMapValidationIssue::MissingCameraForKeyframe {
                    frame_id: keyframe.frame.id,
                    camera_id: keyframe.frame.camera_id,
                });
            }

            for observation in &keyframe.observations {
                self.validate_keyframe_observation(keyframe, observation, report);
            }
        }

        let mut landmarks = self.landmarks.iter().collect::<Vec<_>>();
        landmarks.sort_by_key(|(landmark_id, _)| **landmark_id);
        for (landmark_id, landmark) in landmarks {
            for observation in &landmark.observations {
                self.validate_landmark_observation(*landmark_id, observation, report);
            }
        }

        let mut covariances = self
            .landmark_position_covariances
            .iter()
            .collect::<Vec<_>>();
        covariances.sort_by_key(|(landmark_id, _)| **landmark_id);
        for (landmark_id, covariance) in covariances {
            if !self.landmarks.contains_key(landmark_id) {
                report.push(VisualMapValidationIssue::CovarianceForMissingLandmark {
                    landmark_id: *landmark_id,
                });
                continue;
            }
            let symmetry_error = (covariance - covariance.transpose()).norm();
            let scale = 1.0 + covariance.norm();
            let finite = covariance.iter().all(|value| value.is_finite());
            let positive_semidefinite = finite
                && ((covariance + covariance.transpose()) * 0.5)
                    .symmetric_eigen()
                    .eigenvalues
                    .iter()
                    .all(|value| *value >= -1.0e-12);
            if !finite || symmetry_error > 1.0e-9 * scale || !positive_semidefinite {
                report.push(VisualMapValidationIssue::InvalidLandmarkCovariance {
                    landmark_id: *landmark_id,
                });
            }
        }

        let mut seen_stereo = HashSet::new();
        for stereo in &self.stereo_observations {
            let key = (stereo.frame_id, stereo.landmark_id);
            if !seen_stereo.insert(key) {
                report.push(VisualMapValidationIssue::DuplicateStereoObservation {
                    frame_id: stereo.frame_id,
                    landmark_id: stereo.landmark_id,
                });
            }
            if !self.keyframes.contains_key(&stereo.frame_id) {
                report.push(VisualMapValidationIssue::StereoObservationMissingKeyframe {
                    frame_id: stereo.frame_id,
                    landmark_id: stereo.landmark_id,
                });
            }
            if !self.landmarks.contains_key(&stereo.landmark_id) {
                report.push(VisualMapValidationIssue::StereoObservationMissingLandmark {
                    frame_id: stereo.frame_id,
                    landmark_id: stereo.landmark_id,
                });
            }
            if !self.cameras.contains_key(&stereo.right_camera_id) {
                report.push(
                    VisualMapValidationIssue::StereoObservationMissingRightCamera {
                        frame_id: stereo.frame_id,
                        landmark_id: stereo.landmark_id,
                        camera_id: stereo.right_camera_id,
                    },
                );
            }
            let has_left = self
                .keyframes
                .get(&stereo.frame_id)
                .is_some_and(|keyframe| {
                    keyframe
                        .observations
                        .iter()
                        .any(|obs| obs.landmark_id == stereo.landmark_id)
                });
            if !has_left {
                report.push(
                    VisualMapValidationIssue::StereoObservationMissingLeftObservation {
                        frame_id: stereo.frame_id,
                        landmark_id: stereo.landmark_id,
                    },
                );
            }
            let transform_finite = stereo
                .left_to_right
                .translation
                .iter()
                .chain(stereo.left_to_right.rotation.coords.iter())
                .all(|value| value.is_finite());
            if !stereo.xy_right.coords.iter().all(|value| value.is_finite()) || !transform_finite {
                report.push(VisualMapValidationIssue::InvalidStereoObservation {
                    frame_id: stereo.frame_id,
                    landmark_id: stereo.landmark_id,
                });
            }
        }
    }

    fn validate_keyframe_observation(
        &self,
        keyframe: &Keyframe,
        observation: &Observation,
        report: &mut VisualMapValidationReport,
    ) {
        if observation.frame_id != keyframe.frame.id {
            report.push(VisualMapValidationIssue::ObservationFrameMismatch {
                expected_frame_id: keyframe.frame.id,
                actual_frame_id: observation.frame_id,
                landmark_id: observation.landmark_id,
                keypoint_index: observation.keypoint_index,
            });
        }

        if !self.landmarks.contains_key(&observation.landmark_id) {
            report.push(VisualMapValidationIssue::ObservationMissingLandmark {
                frame_id: keyframe.frame.id,
                landmark_id: observation.landmark_id,
                keypoint_index: observation.keypoint_index,
            });
        }

        if observation.keypoint_index >= keyframe.frame.keypoints.len() {
            report.push(VisualMapValidationIssue::ObservationKeypointOutOfBounds {
                frame_id: keyframe.frame.id,
                landmark_id: observation.landmark_id,
                keypoint_index: observation.keypoint_index,
                keypoint_count: keyframe.frame.keypoints.len(),
            });
        }
    }

    fn validate_landmark_observation(
        &self,
        landmark_id: LandmarkId,
        observation: &Observation,
        report: &mut VisualMapValidationReport,
    ) {
        let Some(keyframe) = self.keyframes.get(&observation.frame_id) else {
            report.push(
                VisualMapValidationIssue::LandmarkObservationMissingKeyframe {
                    landmark_id,
                    frame_id: observation.frame_id,
                },
            );
            return;
        };

        if observation.landmark_id != landmark_id {
            report.push(
                VisualMapValidationIssue::LandmarkObservationLandmarkMismatch {
                    expected_landmark_id: landmark_id,
                    actual_landmark_id: observation.landmark_id,
                    frame_id: observation.frame_id,
                    keypoint_index: observation.keypoint_index,
                },
            );
        }

        if observation.keypoint_index >= keyframe.frame.keypoints.len() {
            report.push(VisualMapValidationIssue::ObservationKeypointOutOfBounds {
                frame_id: observation.frame_id,
                landmark_id,
                keypoint_index: observation.keypoint_index,
                keypoint_count: keyframe.frame.keypoints.len(),
            });
        }
    }

    fn validate_descriptors_into(
        &self,
        descriptor_store: Option<&LandmarkDescriptorStore>,
        report: &mut VisualMapValidationReport,
    ) {
        let mut landmark_ids = self.landmarks.keys().copied().collect::<Vec<_>>();
        landmark_ids.sort_unstable();

        if let Some(descriptor_store) = descriptor_store {
            for landmark_id in landmark_ids {
                if !descriptor_store.contains(landmark_id) {
                    report.push(VisualMapValidationIssue::MissingDescriptorForLandmark {
                        landmark_id,
                    });
                }
            }

            for (landmark_id, _) in descriptor_store.iter() {
                if !self.landmarks.contains_key(&landmark_id) {
                    report.push(VisualMapValidationIssue::DescriptorForMissingLandmark {
                        landmark_id,
                    });
                }
            }
        } else {
            for landmark_id in landmark_ids {
                if self
                    .landmarks
                    .get(&landmark_id)
                    .and_then(|landmark| landmark.descriptor.as_ref())
                    .is_none()
                {
                    report.push(VisualMapValidationIssue::MissingDescriptorForLandmark {
                        landmark_id,
                    });
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisualMapValidationIssue {
    KeyframeIdMismatch {
        keyframe_id: u64,
        frame_id: FrameId,
    },
    MissingCameraForKeyframe {
        frame_id: FrameId,
        camera_id: CameraId,
    },
    ObservationFrameMismatch {
        expected_frame_id: FrameId,
        actual_frame_id: FrameId,
        landmark_id: LandmarkId,
        keypoint_index: usize,
    },
    ObservationMissingLandmark {
        frame_id: FrameId,
        landmark_id: LandmarkId,
        keypoint_index: usize,
    },
    ObservationKeypointOutOfBounds {
        frame_id: FrameId,
        landmark_id: LandmarkId,
        keypoint_index: usize,
        keypoint_count: usize,
    },
    LandmarkObservationMissingKeyframe {
        landmark_id: LandmarkId,
        frame_id: FrameId,
    },
    LandmarkObservationLandmarkMismatch {
        expected_landmark_id: LandmarkId,
        actual_landmark_id: LandmarkId,
        frame_id: FrameId,
        keypoint_index: usize,
    },
    MissingDescriptorForLandmark {
        landmark_id: LandmarkId,
    },
    DescriptorForMissingLandmark {
        landmark_id: LandmarkId,
    },
    CovarianceForMissingLandmark {
        landmark_id: LandmarkId,
    },
    InvalidLandmarkCovariance {
        landmark_id: LandmarkId,
    },
    DuplicateStereoObservation {
        frame_id: FrameId,
        landmark_id: LandmarkId,
    },
    StereoObservationMissingKeyframe {
        frame_id: FrameId,
        landmark_id: LandmarkId,
    },
    StereoObservationMissingLandmark {
        frame_id: FrameId,
        landmark_id: LandmarkId,
    },
    StereoObservationMissingRightCamera {
        frame_id: FrameId,
        landmark_id: LandmarkId,
        camera_id: CameraId,
    },
    StereoObservationMissingLeftObservation {
        frame_id: FrameId,
        landmark_id: LandmarkId,
    },
    InvalidStereoObservation {
        frame_id: FrameId,
        landmark_id: LandmarkId,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VisualMapValidationReport {
    pub issues: Vec<VisualMapValidationIssue>,
}

impl VisualMapValidationReport {
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }

    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }

    pub fn push(&mut self, issue: VisualMapValidationIssue) {
        self.issues.push(issue);
    }

    pub fn into_result(self) -> Result<(), Self> {
        if self.is_valid() {
            Ok(())
        } else {
            Err(self)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LandmarkDescriptorStore {
    descriptors: HashMap<LandmarkId, Vec<f32>>,
}

impl LandmarkDescriptorStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_visual_map(map: &VisualMap) -> Self {
        let descriptors = map
            .landmarks
            .iter()
            .filter_map(|(landmark_id, landmark)| {
                landmark
                    .descriptor
                    .as_ref()
                    .map(|descriptor| (*landmark_id, descriptor.clone()))
            })
            .collect();
        Self { descriptors }
    }

    pub fn insert(&mut self, landmark_id: LandmarkId, descriptor: Vec<f32>) {
        self.descriptors.insert(landmark_id, descriptor);
    }

    pub fn get(&self, landmark_id: LandmarkId) -> Option<&[f32]> {
        self.descriptors.get(&landmark_id).map(Vec::as_slice)
    }

    pub fn contains(&self, landmark_id: LandmarkId) -> bool {
        self.descriptors.contains_key(&landmark_id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (LandmarkId, &[f32])> + '_ {
        let mut descriptors = self
            .descriptors
            .iter()
            .map(|(landmark_id, descriptor)| (*landmark_id, descriptor.as_slice()))
            .collect::<Vec<_>>();
        descriptors.sort_by_key(|(landmark_id, _)| *landmark_id);
        descriptors.into_iter()
    }

    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    pub fn ordered_landmark_descriptors<'a>(
        &'a self,
        map: &'a VisualMap,
    ) -> Vec<(&'a Landmark, &'a [f32])> {
        let mut paired = self
            .descriptors
            .iter()
            .filter_map(|(landmark_id, descriptor)| {
                map.landmarks
                    .get(landmark_id)
                    .map(|landmark| (landmark, descriptor.as_slice()))
            })
            .collect::<Vec<_>>();
        paired.sort_by_key(|(landmark, _)| landmark.id);
        paired
    }
}
