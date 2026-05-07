//! Visual localization foundation library built around reusable SfM / visual maps.
//!
//! `visloc-rs` starts with map-based localization: connect query image features
//! to 3D landmarks, estimate a camera pose with PnP + RANSAC, and return an
//! inspectable `LocalizationResult`. SLAM, mapping, and sensor fusion are kept
//! out of the initial core, but the type and trait boundaries leave room for
//! those pipelines to be added later.
//!
//! # Minimal Example
//!
//! ```
//! use nalgebra::{Point3, UnitQuaternion, Vector3};
//! use visloc_rs::core::geometry::Pose;
//! use visloc_rs::core::types::{Camera, Landmark, QueryImage, VisualMap};
//! use visloc_rs::localize;
//!
//! let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
//! let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
//! let point = Point3::new(0.0, 0.0, 5.0);
//!
//! let mut map = VisualMap::new();
//! let mut landmark = Landmark::new(1, point);
//! landmark.descriptor = Some(vec![1.0, 0.0]);
//! map.landmarks.insert(1, landmark);
//!
//! let query = QueryImage {
//!     camera: camera.clone(),
//!     keypoints: vec![camera.project(&pose.transform_world_point(&point)).unwrap()],
//!     descriptors: vec![vec![1.0, 0.0]],
//! };
//!
//! let result = localize(query, map);
//! assert_eq!(result.correspondence_count, 1);
//! ```
//!
//! Most applications should start with [`LocalizationPipeline`] or
//! [`ColmapMapProvider`](io::colmap::ColmapMapProvider) and then replace
//! feature extraction, matching, candidate selection, or pose estimation through
//! the exposed traits.

pub use visloc_core as core;
pub use visloc_fusion as fusion;
pub use visloc_io as io;
pub use visloc_localization as localization;
pub use visloc_mapping as mapping;
pub use visloc_slam as slam;
pub use visloc_tracking as tracking;
pub use visloc_vision as vision;

pub use visloc_core::geometry;
pub use visloc_core::types;
pub use visloc_core::types::{
    PoseEstimationFailureDiagnostics, PoseEstimationFailureReason, PoseEstimatorDiagnostics,
};
pub use visloc_fusion::{
    FrameTimestampIndex, GnssMeasurement, ImuMeasurement, LocalizationPriorProvider,
    MeasurementBuffer, PoseCovariance, PoseCovarianceMatrix, PosePriorMeasurement,
    PositionCovariance, PriorConfig, TimeDelta, Timed, TimedFrame, TimedMeasurement, TimedPose,
    Timestamp,
};
pub use visloc_localization::{
    localize, localize_frame, localize_frame_with_descriptor_store, localize_frames,
    localize_frames_with_descriptor_store, localize_with_descriptor_store, map_provider_stats,
    AllLandmarksSelector, AllMapSelector, CandidateSelector, CorrespondenceBuildError,
    CorrespondenceBuilder, CorrespondenceSet, DescriptorProvider, FixedLandmarkSelector,
    FixedLandmarkSubmapSelector, FrameLocalizationResult, ImageLocalizer, InMemoryMapProvider,
    IntersectCandidateSelector, LocalizationConfig, LocalizationPipeline, LocalizationPrior,
    MapProvider, MapProviderStats, PriorSubmapSelector, RadiusLandmarkSelector,
    RadiusSubmapSelector, SelectableMapProvider, SubmapSelector,
};
pub use visloc_mapping::{
    AppliedMapUpdate, KeyframeDecision, KeyframeDecisionReason, KeyframePolicy,
    KeyframePolicyConfig, LandmarkCandidate, LandmarkCandidateId, LandmarkCandidateMappingFailure,
    LandmarkCandidateMappingFailureReason, LandmarkCandidateObservation,
    LandmarkCandidateValidationConfig, LandmarkCandidateValidationIssue,
    LandmarkCandidateValidationReport, LinearTriangulator, LocalMapWindow, LocalMapWindowConfig,
    LocalMappingPipeline, LocalMappingResult, LocalRefinementReason, LocalRefinementResult,
    LocalRefiner, MapUpdateValidationIssue, MapUpdateValidationReport, NoopLocalRefiner,
    SimpleKeyframePolicy, StagedMapUpdate, TriangulatedLandmark, TriangulationConfig,
    TriangulationFailureReason, Triangulator,
};
pub use visloc_slam::{OnlineSlamConfig, OnlineSlamPipeline, OnlineSlamResult};
pub use visloc_tracking::{
    ConstantPoseMotionModel, ConstantVelocityMotionModel, FrameLocalizer, ImageTracker,
    MotionModel, Tracker, TrackingConfig, TrackingEvent, TrackingFailureReason, TrackingResult,
    TrackingState, TrackingStats,
};
pub use visloc_vision::features::{
    FeatureExtractor, FeatureSet, FeatureSetError, FnFeatureExtractor, ProvidedFeatureExtractor,
};
pub use visloc_vision::matching::{BruteForceMatcher, CrossCheckMatcher, DescriptorMatch, Matcher};
pub use visloc_vision::pnp::{
    Correspondence2D3D, DltPnP, GaussNewtonPoseRefiner, PoseEstimator, PoseRefiner,
};
pub use visloc_vision::ransac::RobustPoseEstimator;
