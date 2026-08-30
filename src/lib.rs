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
pub use visloc_core::geometry::{reproject, Pose, Sim3, Sim3Tangent, SE3, SO3};
pub use visloc_core::types;
pub use visloc_core::types::{
    Camera, CameraId, CameraModel, Frame, FrameId, Keyframe, Landmark, LandmarkDescriptorStore,
    LandmarkId, LocalizationFailureReason, LocalizationResult, LocalizationSuccess, Observation,
    PoseEstimationFailureDiagnostics, PoseEstimationFailureReason, PoseEstimatorDiagnostics,
    QueryImage, VisualMap, VisualMapValidationIssue, VisualMapValidationReport,
};

/// Lossless verified-pair snapshot codec shared by the SfM diagnostics.
///
/// This lives in the library rather than directly under `examples/` so Cargo
/// does not discover the module-only implementation as a standalone example.
pub mod verified_pair_snapshot;

pub use visloc_fusion::{
    FramePriorSource, FramePriorSyncEvaluationConfig, FramePriorSyncEvaluationFailure,
    FramePriorSyncEvaluationResult, FramePriorSyncSummary, FrameTimestampIndex, GnssMeasurement,
    ImuMeasurement, LocalizationPriorProvider, MeasurementBuffer, PoseCovariance,
    PoseCovarianceMatrix, PosePriorMeasurement, PositionCovariance, PriorConfig, TimeDelta, Timed,
    TimedFrame, TimedMeasurement, TimedPose, Timestamp,
};
pub use visloc_io::calibration::{
    kitti_projection_to_pinhole_camera, parse_kitti_calibration_txt, read_kitti_calibration_txt,
    read_kitti_pinhole_camera, CalibrationError, KittiProjection,
};
pub use visloc_io::colmap::{
    write_colmap_binary_model_for_3dgs, write_colmap_reconstruction_for_3dgs,
    write_colmap_reconstruction_for_3dgs_with_cameras, write_colmap_text_model_for_3dgs,
    ColmapError, ColmapExportSummary, ColmapMapProvider,
};
pub use visloc_io::external_deep::{
    parse_external_deep_features_txt, parse_external_deep_matches_txt,
    read_external_deep_features_txt, read_external_deep_matches_txt, ExternalDeepError,
    ExternalDeepFeature, ExternalDeepFeatureSet, ExternalDeepMatch, ExternalDeepMatchSet,
};
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
#[cfg(feature = "image-io")]
pub use visloc_io::kitti::{
    read_kitti_image_sequence_dir, read_kitti_image_sequence_dir_with_timestamp_file,
    KittiDatasetError, KittiImageSequence,
};
pub use visloc_io::kitti_imu::{
    parse_kitti_oxts_sample, parse_kitti_oxts_timestamp_line, parse_kitti_oxts_timestamps_txt,
    read_kitti_oxts_dir, KittiOxtsError, KittiOxtsRecord, KittiOxtsSample, KITTI_OXTS_FIELD_COUNT,
};
pub use visloc_io::sensors::{
    parse_gnss_measurements_txt, read_gnss_measurements_txt, SensorLogError,
};
pub use visloc_io::two_view_matches::{
    parse_two_view_matches_txt, read_two_view_matches_txt, TwoViewFeatureMatch, TwoViewMatchError,
    TwoViewMatchSet,
};
pub use visloc_localization::{
    localize, localize_frame, localize_frame_with_descriptor_store, localize_frames,
    localize_frames_with_descriptor_store, localize_with_descriptor_store, map_provider_stats,
    AllLandmarksSelector, AllMapSelector, CandidateSelector, CorrespondenceBuildError,
    CorrespondenceBuilder, CorrespondenceSet, DescriptorProvider, FixedLandmarkSelector,
    FixedLandmarkSubmapSelector, FrameLocalizationResult, ImageLocalizer, InMemoryMapProvider,
    IntersectCandidateSelector, LocalizationConfig, LocalizationPipeline, LocalizationPrior,
    MapProvider, MapProviderStats, PriorSubmapSelector, ProjectionCorrespondenceBuilder,
    RadiusLandmarkSelector, RadiusSubmapSelector, SelectableMapProvider, SubmapSelector,
};
pub use visloc_mapping::{
    build_stereo_metric_points, build_stereo_replenish_candidates, AppliedMapUpdate,
    KeyframeDecision, KeyframeDecisionReason, KeyframePolicy, KeyframePolicyConfig,
    LandmarkCandidate, LandmarkCandidateId, LandmarkCandidateMappingFailure,
    LandmarkCandidateMappingFailureReason, LandmarkCandidateObservation,
    LandmarkCandidateValidationConfig, LandmarkCandidateValidationIssue,
    LandmarkCandidateValidationReport, LinearTriangulator, LocalMapWindow, LocalMapWindowConfig,
    LocalMappingPipeline, LocalMappingResult, LocalRefinementReason, LocalRefinementResult,
    LocalRefiner, MapUpdateValidationIssue, MapUpdateValidationReport, NoopLocalRefiner,
    SimpleKeyframePolicy, StagedMapUpdate, StereoReplenishConfig, TriangulatedLandmark,
    TriangulationConfig, TriangulationFailureReason, Triangulator,
};
pub use visloc_slam::{
    appearance_loop_candidate_descriptor_store, bearing_alignment_error_deg,
    build_appearance_loop_candidates, build_appearance_loop_candidates_with_diagnostics,
    close_loops_on_vo_trajectory, close_loops_on_vo_trajectory_with_globals,
    close_loops_on_vo_trajectory_with_globals_and_loop_matches,
    close_loops_on_vo_trajectory_with_loop_matches, correspondences_2d3d_for_loop_candidate,
    correspondences_for_loop_candidate, detect_loop_candidates,
    estimate_free_poses_from_prior_rays, estimate_gravity_and_velocities, estimate_gyro_bias,
    filter_pose_priors_by_edge_disagreement, filter_pose_priors_by_free_centre_residual,
    filter_pose_priors_by_track_quality, generate_ordered_pairs, gt_bearing_in_prior_frame,
    incremental_sfm, incremental_sfm_with_initial_poses, incremental_sfm_with_per_image_cameras,
    incremental_sfm_with_sequence_fallback_overrides, incremental_sfm_with_track_membership,
    loop_closure_constraints_from_candidates, online_ba_imu_state_rows,
    online_slam_results_to_html_report, pair_correspondences, pair_essential_mean_sampson_error,
    pairwise_pose_factors_from_loop_closures, parse_stereo_vo_imu_samples_txt,
    preview_track_build_stats, prior_free_essential_gt_bearing_error_deg, reconstruct_global_sfm,
    reconstruct_global_sfm_with_per_image_cameras, reconstruct_global_sfm_with_priors,
    reconstruct_stereo_vo_with_ba, refine_stereo_vo_with_ba,
    refine_visual_map_with_covisibility_ba,
    refine_visual_map_with_covisibility_ba_and_neighbor_allowlist, relative_pose_from_essential,
    relative_world_to_camera, rematch_essential_admission_ok, scan_pairwise_loop_closures,
    select_covisibility_local_ba_window,
    select_covisibility_local_ba_window_with_neighbor_allowlist, slice_imu_samples_for_keyframes,
    verify_loop_closure_candidates, verify_loop_closure_candidates_hybrid,
    verify_loop_closure_candidates_pnp, write_online_ba_imu_state_csv,
    write_online_slam_results_html_report, AdaptiveVelocityGateConfig,
    AppearanceLoopCandidateBuildResult, AppearanceLoopCandidateDiagnostic,
    AppearanceLoopScannerSettings, AtlasSubmap, BaConfig, BaError, BaIterationStats, BaObservation,
    BaResult, BaStereoObservation, BiasReleaseSchedule, BundleAdjustment, BundleAdjustmentRefiner,
    ChordalRotationInit, CovisibilityKeyframeScore, CovisibilityLocalBaConfig,
    CovisibilityLocalBaError, CovisibilityLocalBaResult, CovisibilityLocalBaSelection,
    CrossSubmapAlignmentConfig, CrossSubmapAlignmentResult, CrossSubmapBoundaryFactorResult,
    CrossSubmapCandidateDiagnostic, CrossSubmapCandidateFailureReason, CrossSubmapLandmarkMatch,
    CrossSubmapScaleEstimate, CrossSubmapWindowAlignmentResult, EssentialMatrixLoopClosureVerifier,
    GlobalReconstructionError, GlobalReconstructionTuning, GlobalSfmEdge, GlobalSfmPoses,
    GravityPrior, GravityVelocityAlignment, GyroBiasAlignment, HybridLoopClosureVerifier,
    HybridLoopClosureVerifierConfig, ImuPreintegratedDelta, ImuPreintegrationFactor,
    ImuPreintegrator, IncrementalSfmConfig, IncrementalSfmError, IncrementalSfmResult,
    LandmarkInit, LinearSolver, LoopAppearanceCandidateConfig, LoopCandidatePair,
    LoopCandidateVerificationDiagnostic, LoopClosureCandidate, LoopClosureCandidateSource,
    LoopClosureConfig, LoopClosureConstraint, LoopClosureVerification,
    LoopClosureVerificationFailureReason, LoopClosureVerifier, LoopClosureVerifierConfig,
    LoopRefinementSolver, LoopRefinementVerifier, MapAtlas, MapAtlasError, MaterializedAtlas,
    MotionBasedViInitializationResult, MotionBasedViInitializationStatus, MotionBasedViInitializer,
    MotionBasedViInitializerConfig, MotionBasedViRejectionReason, MotionViInitializationEvent,
    MotionViInitializationStatus, MotionViRawResidualActivationConfig, NextImagePolicy,
    OnlineBaImuStateRow, OnlineBaTriggerStats, OnlineSlamAdmittedLoopConstraint, OnlineSlamConfig,
    OnlineSlamConfigError, OnlineSlamCovisibilityLocalBaConfig, OnlineSlamCovisibilityLocalBaStats,
    OnlineSlamLocalBaConfig, OnlineSlamLoopClosureRefinementConfig,
    OnlineSlamLoopClosureRefinementState, OnlineSlamLoopClosureRefinementStats,
    OnlineSlamLoopConstraintRejectionReason, OnlineSlamMotionViInitConfig, OnlineSlamPipeline,
    OnlineSlamRejectedLoopConstraint, OnlineSlamRelocalizationAppearanceCandidate,
    OnlineSlamRelocalizationAppearanceConfig, OnlineSlamRelocalizationConfig,
    OnlineSlamRelocalizationCovisibilityConfig, OnlineSlamRelocalizationState,
    OnlineSlamRelocalizationStats, OnlineSlamResult, OnlineSlamViInitConfig, OnlineStereoVoBa,
    OnlineStereoVoBaConfig, OrderedPairCandidate, OrderedPairGeneratorConfig, OrderedPairHints,
    OrderedPairSource, PairwiseKeyframeView, PairwiseLoopClosureScannerConfig, PairwiseMatches,
    PairwisePoseFactor, PerImageCameraError, PerImageCameraGlobalError,
    PerImageCameraIncrementalError, PerImageCameras, PerPoseGravityObservation,
    PerPoseGravityPrior, PnPLoopClosureVerifier, PnPLoopClosureVerifierConfig, PoseGraph,
    PoseGraphEdge, PoseGraphEdgeKind, PoseGraphError, PoseGraphOptimizationStep,
    PoseGraphParseError, PoseGraphSe3Config, PoseGraphSe3IterationStats, PoseGraphSe3Result,
    PositionPrior, PositionPriorObservation, ReconstructedLandmark, RobustKernel, SfmTrack,
    Sim3Edge, Sim3Information, Sim3PoseGraph, Sim3PoseGraphConfig, Sim3PoseGraphIterationStats,
    Sim3PoseGraphResult, SparseFactorGraph, SparseFactorGraphConfig, SparseFactorGraphUpdateStats,
    SparseFactorInactiveReason, SparseFactorKey, SparseFactorKind, SparseFactorMeasurement,
    SparseFactorState, SparseKeyframeFactor, StationaryRejectionReason, StereoVoBaConfig,
    StereoVoBaError, StereoVoBaImuInput, StereoVoBaImuRefinement, StereoVoBaImuSample,
    StereoVoBaRefinement, StereoVoReconstruction, SubmapId, SubmapIdRemap, SubmapMergeEvidence,
    SubmapMergeQuality, SubmapMergeVerificationConfig, TrackBuildStats, TrackSource,
    VerifiedSubmapMerge, ViInitFallback, ViInitializationEvent, ViInitializationStatus,
    Viba2Config, Viba2Stats, VisualInertialInitializationResult, VisualInertialInitializer,
    VisualInertialInitializerConfig, VoLoopClosureConfig, VoLoopClosureError, VoLoopClosureResult,
};
pub use visloc_tracking::{
    tracking_results_to_csv, tracking_results_to_html_report, umeyama_similarity_transform,
    write_tracking_results_csv, write_tracking_results_html_report, AdaptiveImuPoseMotionModel,
    AdaptiveImuPoseMotionModelConfig, AdaptiveMotionMode, ConstantPoseMotionModel,
    ConstantVelocityMotionModel, CovisibilityLocalMapConfig, FrameLocalizer, ImageTracker,
    ImuPredictiveMotionModel, ImuPredictiveMotionModelConfig, ImuVelocityRefreshPolicy,
    KittiOdometryBenchmarkConfig, KittiOdometryBenchmarkSummary, KittiOdometrySegmentError,
    KittiTrajectoryParseError, MotionModel, NoopVisualOdometryFrontend,
    PosePriorVisualOverrideConfig, PoseTrajectory, ProjectionGuidedTrackingConfig,
    RelativePoseError, RelativePoseErrorConfig, RelativePoseErrorStatistics,
    RelativePoseErrorSummary, Tracker, TrackingConfig, TrackingEvaluationConfig,
    TrackingEvaluationFailure, TrackingEvaluationResult, TrackingEvent, TrackingFailureReason,
    TrackingResult, TrackingState, TrackingStats, TrajectoryAlignment, TrajectoryErrorSummary,
    TrajectoryEvaluationConfig, TrajectoryEvaluationFailure, TrajectoryEvaluationResult,
    TrajectoryFileError, TrajectorySample, TrajectorySimilarityTransform, TrajectorySummary,
    TrajectoryTranslationError, TumTrajectoryParseError, VisualOdometryEstimate,
    VisualOdometryFrontend, VisualOdometryPosePrior, VisualOdometryPriorProvider,
};
pub use visloc_vision::distortion::RadialTangential;
pub use visloc_vision::features::{
    build_pyramid, global_descriptor_onnx, superpoint_onnx, CornerDeepAdapter, CornerFeatureConfig,
    CornerFeatureError, CornerFeatureExtractor, DeepFeatureExtractor, DeepFeatureSet,
    DeepFeatureSetError, FeatureExtractor, FeatureSet, FeatureSetError, FnFeatureExtractor,
    GrayscaleImage, GrayscaleImageError, HogLikeFeatureConfig, HogLikeFeatureError,
    HogLikeFeatureExtractor, MultiScaleDeepConfig, MultiScaleDeepExtractor,
    ProvidedFeatureExtractor, HOG_BINS, HOG_CELLS_PER_SIDE, HOG_CELL_SIZE, HOG_DESCRIPTOR_DIM,
};
pub use visloc_vision::matching::{
    BruteForceMatcher, CrossCheckMatcher, DescriptorMatch, Matcher, MutualSoftmaxConfig,
    MutualSoftmaxMatcher,
};
pub use visloc_vision::pnp::{
    Correspondence2D3D, DltPnP, GaussNewtonPoseRefiner, PoseEstimator, PoseRefiner,
};
pub use visloc_vision::ransac::{PnPRansac, RansacReport, RobustPoseEstimator};
pub use visloc_vision::stereo::triangulate_stereo_pixel;
pub use visloc_vision::stereo_bootstrap::{
    bootstrap_stereo_landmarks, triangulate_two_view_left_frame, StereoBootstrapConfig,
    StereoBootstrapLandmark,
};
pub use visloc_vision::stereo_vo::{
    build_stereo_temporal_correspondences, estimate_relative_pose_kabsch_ransac,
    extend_stereo_tracks_via_projection, stereo_pair_correspondences,
    triangulate_stereo_feature_matches, triangulate_stereo_features, KabschRansacConfig,
    KabschRansacReport, StereoAdaptiveDepthGateConfig, StereoDepthGate, StereoDepthGateDiagnostics,
    StereoDepthGateState, StereoFeature, StereoFeatureConfig, StereoPairCorrespondence,
    StereoRelativePoseMode, StereoTrack, StereoTrackObservation, StereoVoError, StereoVoFrontend,
    StereoVoFrontendConfig, StereoVoPairDiagnostics, TrackExtensionConfig,
};

mod two_view_vo;
pub use two_view_vo::{
    EssentialMatrixVisualOdometryConfig, EssentialMatrixVisualOdometryFrontend,
    TwoViewMatchVisualOdometryConfig, TwoViewMatchVisualOdometryFrontend,
};

/// Common imports for applications using the default visual-localization stack.
///
/// The prelude is intentionally focused on the stable, high-level surface:
/// core map/query/pose types, the default localization pipeline, provider
/// traits, feature/matching/pose-estimation extension traits, and lightweight
/// tracking/mapping/fusion handles. Module paths remain available for callers
/// that prefer explicit imports.
pub mod prelude {
    pub use crate::{
        build_stereo_metric_points, build_stereo_replenish_candidates, localize, localize_frame,
        localize_frame_with_descriptor_store, localize_frames,
        localize_frames_with_descriptor_store, localize_with_descriptor_store, map_provider_stats,
        online_slam_results_to_html_report, parse_external_deep_features_txt,
        parse_external_deep_matches_txt, parse_two_view_matches_txt,
        read_external_deep_features_txt, read_external_deep_matches_txt, read_two_view_matches_txt,
        reproject, tracking_results_to_csv, tracking_results_to_html_report,
        triangulate_stereo_feature_matches, umeyama_similarity_transform,
        write_online_slam_results_html_report, write_tracking_results_csv,
        write_tracking_results_html_report, AllLandmarksSelector, AllMapSelector,
        BruteForceMatcher, Camera, CameraId, CameraModel, CandidateSelector, ChordalRotationInit,
        ColmapMapProvider, ConstantPoseMotionModel, ConstantVelocityMotionModel, CornerDeepAdapter,
        CornerFeatureConfig, CornerFeatureError, CornerFeatureExtractor, Correspondence2D3D,
        CorrespondenceBuildError, CorrespondenceBuilder, CorrespondenceSet, CrossCheckMatcher,
        DeepFeatureExtractor, DeepFeatureSet, DeepFeatureSetError, DescriptorMatch,
        DescriptorProvider, DltPnP, EssentialMatrixLoopClosureVerifier,
        EssentialMatrixVisualOdometryConfig, EssentialMatrixVisualOdometryFrontend,
        ExternalDeepError, ExternalDeepFeature, ExternalDeepFeatureSet, ExternalDeepMatch,
        ExternalDeepMatchSet, FeatureExtractor, FeatureSet, FeatureSetError, FixedLandmarkSelector,
        FixedLandmarkSubmapSelector, FnFeatureExtractor, Frame, FrameId, FrameLocalizationResult,
        FrameLocalizer, FramePriorSource, FramePriorSyncEvaluationConfig,
        FramePriorSyncEvaluationFailure, FramePriorSyncEvaluationResult, FramePriorSyncSummary,
        FrameTimestampIndex, GaussNewtonPoseRefiner, GnssMeasurement, GrayscaleImage,
        GrayscaleImageError, HogLikeFeatureConfig, HogLikeFeatureError, HogLikeFeatureExtractor,
        ImageLocalizer, ImageTracker, ImuMeasurement, ImuPreintegratedDelta,
        ImuPreintegrationFactor, ImuPreintegrator, InMemoryMapProvider, IntersectCandidateSelector,
        Keyframe, KeyframeDecision, KeyframePolicy, KeyframePolicyConfig,
        KittiOdometryBenchmarkConfig, KittiOdometryBenchmarkSummary, KittiOdometrySegmentError,
        KittiTrajectoryParseError, Landmark, LandmarkCandidate, LandmarkCandidateId,
        LandmarkCandidateMappingFailure, LandmarkCandidateMappingFailureReason,
        LandmarkCandidateObservation, LandmarkCandidateValidationConfig,
        LandmarkCandidateValidationIssue, LandmarkCandidateValidationReport,
        LandmarkDescriptorStore, LandmarkId, LinearTriangulator, LocalMapWindow,
        LocalMapWindowConfig, LocalMappingPipeline, LocalMappingResult, LocalizationConfig,
        LocalizationFailureReason, LocalizationPipeline, LocalizationPrior,
        LocalizationPriorProvider, LocalizationResult, LocalizationSuccess, LoopClosureCandidate,
        LoopClosureConfig, LoopClosureConstraint, LoopClosureVerification,
        LoopClosureVerificationFailureReason, LoopClosureVerifier, LoopClosureVerifierConfig,
        LoopRefinementSolver, LoopRefinementVerifier, MapProvider, MapProviderStats, Matcher,
        MeasurementBuffer, MotionModel, MultiScaleDeepConfig, MultiScaleDeepExtractor,
        MutualSoftmaxConfig, MutualSoftmaxMatcher, NoopLocalRefiner, NoopVisualOdometryFrontend,
        Observation, OnlineSlamConfig, OnlineSlamPipeline, OnlineSlamResult, PnPRansac, Pose,
        PoseCovariance, PoseCovarianceMatrix, PoseEstimationFailureDiagnostics,
        PoseEstimationFailureReason, PoseEstimator, PoseEstimatorDiagnostics, PoseGraph,
        PoseGraphEdge, PoseGraphEdgeKind, PoseGraphError, PoseGraphOptimizationStep,
        PosePriorMeasurement, PoseRefiner, PoseTrajectory, PositionCovariance, PriorConfig,
        PriorSubmapSelector, ProvidedFeatureExtractor, QueryImage, RadiusLandmarkSelector,
        RadiusSubmapSelector, RansacReport, RelativePoseError, RelativePoseErrorConfig,
        RelativePoseErrorStatistics, RelativePoseErrorSummary, RobustPoseEstimator,
        SelectableMapProvider, Sim3, Sim3Edge, Sim3Information, Sim3PoseGraph, Sim3PoseGraphConfig,
        Sim3PoseGraphIterationStats, Sim3PoseGraphResult, Sim3Tangent, SimpleKeyframePolicy,
        StagedMapUpdate, StationaryRejectionReason, StereoRelativePoseMode, StereoReplenishConfig,
        StereoVoPairDiagnostics, SubmapSelector, TimeDelta, Timed, TimedFrame, TimedMeasurement,
        TimedPose, Timestamp, Tracker, TrackingConfig, TrackingEvaluationConfig,
        TrackingEvaluationFailure, TrackingEvaluationResult, TrackingEvent, TrackingFailureReason,
        TrackingResult, TrackingState, TrackingStats, TrajectoryAlignment, TrajectoryErrorSummary,
        TrajectoryEvaluationConfig, TrajectoryEvaluationFailure, TrajectoryEvaluationResult,
        TrajectoryFileError, TrajectorySample, TrajectorySimilarityTransform, TrajectorySummary,
        TrajectoryTranslationError, TriangulationConfig, TriangulationFailureReason, Triangulator,
        TumTrajectoryParseError, TwoViewFeatureMatch, TwoViewMatchError, TwoViewMatchSet,
        TwoViewMatchVisualOdometryConfig, TwoViewMatchVisualOdometryFrontend,
        VisualInertialInitializationResult, VisualInertialInitializer,
        VisualInertialInitializerConfig, VisualMap, VisualMapValidationIssue,
        VisualMapValidationReport, VisualOdometryEstimate, VisualOdometryFrontend,
        VisualOdometryPosePrior, VisualOdometryPriorProvider, SE3, SO3,
    };

    pub use crate::io::calibration::{
        kitti_projection_to_pinhole_camera, parse_kitti_calibration_txt,
        read_kitti_calibration_txt, read_kitti_pinhole_camera, CalibrationError, KittiProjection,
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
    #[cfg(feature = "image-io")]
    pub use crate::io::kitti::{
        read_kitti_image_sequence_dir, read_kitti_image_sequence_dir_with_timestamp_file,
        KittiDatasetError, KittiImageSequence,
    };
    pub use crate::io::query_features::{read_query_features_txt, QueryFeatureError};
    pub use crate::io::sensors::{
        parse_gnss_measurements_txt, read_gnss_measurements_txt, SensorLogError,
    };
}
