use nalgebra::Point3;
use std::collections::HashMap;

use super::{Camera, CameraId, FrameId, Keyframe, Observation};

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

#[derive(Debug, Clone, PartialEq, Default)]
pub struct VisualMap {
    pub cameras: HashMap<CameraId, Camera>,
    pub landmarks: HashMap<LandmarkId, Landmark>,
    pub keyframes: HashMap<u64, Keyframe>,
}

impl VisualMap {
    pub fn new() -> Self {
        Self::default()
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
