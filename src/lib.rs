#![forbid(unsafe_code)]
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
//! use visloc_rs::prelude::*;
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
//! [`ColmapMapProvider`] and then replace
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
pub use visloc_core::geometry::{reproject, Pose, SE3, SO3};
pub use visloc_core::types;
pub use visloc_core::types::{
    Camera, CameraId, CameraModel, Frame, FrameId, Keyframe, Landmark, LandmarkDescriptorStore,
    LandmarkId, LocalizationFailureReason, LocalizationResult, LocalizationSuccess, Observation,
    PoseEstimationFailureDiagnostics, PoseEstimationFailureReason, PoseEstimatorDiagnostics,
    QueryImage, VisualMap, VisualMapValidationIssue, VisualMapValidationReport,
};
pub use visloc_fusion::{
    FramePriorSource, FrameTimestampIndex, GnssMeasurement, ImuMeasurement,
    LocalizationPriorProvider, MeasurementBuffer, PoseCovariance, PoseCovarianceMatrix,
    PosePriorMeasurement, PositionCovariance, PriorConfig, TimeDelta, Timed, TimedFrame,
    TimedMeasurement, TimedPose, Timestamp,
};
pub use visloc_io::colmap::ColmapMapProvider;
#[cfg(feature = "image-io")]
pub use visloc_io::images::{
    common_image_sequence_summary, decode_common_image, parse_timestamp_nanoseconds_txt,
    read_common_image, read_common_image_sequence, read_common_image_sequence_dir,
    read_common_image_sequence_dir_with_timestamp_file,
    read_common_image_sequence_dir_with_timestamps, read_common_image_sequence_with_timestamps,
    read_timestamp_nanoseconds_txt, validate_common_image_sequence_dimensions,
    validate_common_image_sequence_timestamps, write_png_gray, CommonImageError,
    ImageSequenceError, ImageSequenceSummary, ImageSequenceValidationIssue, LoadedImageFrame,
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
    tracking_results_to_csv, tracking_results_to_html_report, write_tracking_results_csv,
    write_tracking_results_html_report, ConstantPoseMotionModel, ConstantVelocityMotionModel,
    FrameLocalizer, ImageTracker, KittiTrajectoryParseError, MotionModel, PoseTrajectory, Tracker,
    TrackingConfig, TrackingEvaluationConfig, TrackingEvaluationFailure, TrackingEvaluationResult,
    TrackingEvent, TrackingFailureReason, TrackingResult, TrackingState, TrackingStats,
    TrajectoryAlignment, TrajectoryErrorSummary, TrajectoryEvaluationConfig,
    TrajectoryEvaluationFailure, TrajectoryEvaluationResult, TrajectoryFileError, TrajectorySample,
    TrajectorySummary, TrajectoryTranslationError, TumTrajectoryParseError,
};
pub use visloc_vision::features::{
    CornerFeatureConfig, CornerFeatureError, CornerFeatureExtractor, FeatureExtractor, FeatureSet,
    FeatureSetError, FnFeatureExtractor, GrayscaleImage, GrayscaleImageError,
    ProvidedFeatureExtractor,
};
pub use visloc_vision::matching::{BruteForceMatcher, CrossCheckMatcher, DescriptorMatch, Matcher};
pub use visloc_vision::pnp::{
    Correspondence2D3D, DltPnP, GaussNewtonPoseRefiner, PoseEstimator, PoseRefiner,
};
pub use visloc_vision::ransac::{PnPRansac, RansacReport, RobustPoseEstimator};

/// Common imports for applications using the default visual-localization stack.
///
/// The prelude is intentionally focused on the stable, high-level surface:
/// core map/query/pose types, the default localization pipeline, provider
/// traits, feature/matching/pose-estimation extension traits, and lightweight
/// tracking/mapping/fusion handles. Module paths remain available for callers
/// that prefer explicit imports.
pub mod prelude {
    pub use crate::{
        localize, localize_frame, localize_frame_with_descriptor_store, localize_frames,
        localize_frames_with_descriptor_store, localize_with_descriptor_store, map_provider_stats,
        reproject, tracking_results_to_csv, tracking_results_to_html_report,
        write_tracking_results_csv, write_tracking_results_html_report, AllLandmarksSelector,
        AllMapSelector, BruteForceMatcher, Camera, CameraId, CameraModel, CandidateSelector,
        ColmapMapProvider, ConstantPoseMotionModel, ConstantVelocityMotionModel,
        CornerFeatureConfig, CornerFeatureError, CornerFeatureExtractor, Correspondence2D3D,
        CorrespondenceBuildError, CorrespondenceBuilder, CorrespondenceSet, CrossCheckMatcher,
        DescriptorMatch, DescriptorProvider, DltPnP, FeatureExtractor, FeatureSet, FeatureSetError,
        FixedLandmarkSelector, FixedLandmarkSubmapSelector, FnFeatureExtractor, Frame, FrameId,
        FrameLocalizationResult, FrameLocalizer, FramePriorSource, FrameTimestampIndex,
        GaussNewtonPoseRefiner, GnssMeasurement, GrayscaleImage, GrayscaleImageError,
        ImageLocalizer, ImageTracker, ImuMeasurement, InMemoryMapProvider,
        IntersectCandidateSelector, Keyframe, KeyframeDecision, KeyframePolicy,
        KeyframePolicyConfig, KittiTrajectoryParseError, Landmark, LandmarkCandidate,
        LandmarkCandidateId, LandmarkCandidateMappingFailure,
        LandmarkCandidateMappingFailureReason, LandmarkCandidateObservation,
        LandmarkCandidateValidationConfig, LandmarkCandidateValidationIssue,
        LandmarkCandidateValidationReport, LandmarkDescriptorStore, LandmarkId, LinearTriangulator,
        LocalMapWindow, LocalMapWindowConfig, LocalMappingPipeline, LocalMappingResult,
        LocalizationConfig, LocalizationFailureReason, LocalizationPipeline, LocalizationPrior,
        LocalizationPriorProvider, LocalizationResult, LocalizationSuccess, MapProvider,
        MapProviderStats, Matcher, MeasurementBuffer, MotionModel, NoopLocalRefiner, Observation,
        OnlineSlamConfig, OnlineSlamPipeline, OnlineSlamResult, PnPRansac, Pose, PoseCovariance,
        PoseCovarianceMatrix, PoseEstimationFailureDiagnostics, PoseEstimationFailureReason,
        PoseEstimator, PoseEstimatorDiagnostics, PosePriorMeasurement, PoseRefiner, PoseTrajectory,
        PositionCovariance, PriorConfig, PriorSubmapSelector, ProvidedFeatureExtractor, QueryImage,
        RadiusLandmarkSelector, RadiusSubmapSelector, RansacReport, RobustPoseEstimator,
        SelectableMapProvider, SimpleKeyframePolicy, StagedMapUpdate, SubmapSelector, TimeDelta,
        Timed, TimedFrame, TimedMeasurement, TimedPose, Timestamp, Tracker, TrackingConfig,
        TrackingEvaluationConfig, TrackingEvaluationFailure, TrackingEvaluationResult,
        TrackingEvent, TrackingFailureReason, TrackingResult, TrackingState, TrackingStats,
        TrajectoryAlignment, TrajectoryErrorSummary, TrajectoryEvaluationConfig,
        TrajectoryEvaluationFailure, TrajectoryEvaluationResult, TrajectoryFileError,
        TrajectorySample, TrajectorySummary, TrajectoryTranslationError, TriangulationConfig,
        TriangulationFailureReason, Triangulator, TumTrajectoryParseError, VisualMap,
        VisualMapValidationIssue, VisualMapValidationReport, SE3, SO3,
    };

    pub use crate::io::colmap::{
        read_colmap_binary_model, read_colmap_text_model, write_colmap_text_model, ColmapError,
        ColmapMapProviderError,
    };
    pub use crate::io::descriptors::{read_landmark_descriptors_txt, DescriptorStoreError};
    #[cfg(feature = "image-io")]
    pub use crate::io::images::{
        common_image_sequence_summary, decode_common_image, parse_timestamp_nanoseconds_txt,
        read_common_image, read_common_image_sequence, read_common_image_sequence_dir,
        read_common_image_sequence_dir_with_timestamp_file,
        read_common_image_sequence_dir_with_timestamps, read_common_image_sequence_with_timestamps,
        read_timestamp_nanoseconds_txt, validate_common_image_sequence_dimensions,
        validate_common_image_sequence_timestamps, write_png_gray, CommonImageError,
        ImageSequenceError, ImageSequenceSummary, ImageSequenceValidationIssue, LoadedImageFrame,
    };
    pub use crate::io::images::{
        parse_pgm, read_pgm, to_pgm_ascii, write_pgm_ascii, PgmImageError,
    };
    pub use crate::io::query_features::{read_query_features_txt, QueryFeatureError};
}
