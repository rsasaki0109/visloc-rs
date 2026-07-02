#![allow(dead_code, unused_imports)]

//! Compile-time public API allowlist.
//!
//! This test mirrors `docs/api_stability.md`: stable-intent types and
//! replaceable algorithm boundaries must keep their canonical import paths.
//! If a path moves, this test should fail until the migration and docs are
//! updated deliberately.

use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::{reproject, Pose, Sim3, SE3, SO3};
use visloc_rs::core::types::{
    Camera, CameraId, CameraModel, Frame, FrameId, Keyframe, Landmark, LandmarkDescriptorStore,
    LandmarkId, LocalizationFailureReason, LocalizationResult, LocalizationSuccess, Observation,
    PoseEstimationFailureDiagnostics, PoseEstimationFailureReason, PoseEstimatorDiagnostics,
    QueryImage, VisualMap, VisualMapValidationIssue, VisualMapValidationReport,
};
use visloc_rs::fusion::{
    FramePriorSource, GnssMeasurement, ImuMeasurement, MeasurementBuffer, PoseCovariance,
    PoseCovarianceMatrix, PosePriorMeasurement, PositionCovariance, PriorConfig, TimeDelta, Timed,
    TimedFrame, TimedMeasurement, TimedPose, Timestamp,
};
use visloc_rs::localization::{
    AllLandmarksSelector, AllMapSelector, CandidateSelector, CorrespondenceBuildError,
    CorrespondenceBuilder, CorrespondenceSet, DescriptorProvider, FixedLandmarkSelector,
    FixedLandmarkSubmapSelector, FrameLocalizationResult, ImageLocalizer, InMemoryMapProvider,
    IntersectCandidateSelector, LocalizationConfig, LocalizationPipeline, LocalizationPrior,
    MapProvider, MapProviderStats, PriorSubmapSelector, RadiusLandmarkSelector,
    RadiusSubmapSelector, SelectableMapProvider, SubmapSelector,
};
use visloc_rs::mapping::{
    LinearTriangulator, LocalMappingPipeline, LocalRefiner, NoopLocalRefiner, SimpleKeyframePolicy,
    Triangulator,
};
use visloc_rs::slam::{
    refine_visual_map_with_covisibility_ba, select_covisibility_local_ba_window,
    CovisibilityLocalBaConfig, CovisibilityLocalBaSelection, OnlineSlamConfig,
    OnlineSlamCovisibilityLocalBaConfig, OnlineSlamCovisibilityLocalBaStats, OnlineSlamPipeline,
};
use visloc_rs::tracking::{
    ConstantPoseMotionModel, MotionModel, PoseTrajectory, Tracker, TrackingEvaluationConfig,
    TrackingEvaluationResult, TrackingStats, TrajectoryErrorSummary,
};
use visloc_rs::vision::features::{
    FeatureExtractor, FeatureSet, GrayscaleImage, ProvidedFeatureExtractor,
};
use visloc_rs::vision::matching::{BruteForceMatcher, Matcher};
use visloc_rs::vision::pnp::{
    Correspondence2D3D, DltPnP, GaussNewtonPoseRefiner, PoseEstimator, PoseRefiner,
};
use visloc_rs::vision::ransac::{PnPRansac, RobustPoseEstimator};
use visloc_rs::vision::stereo_vo::{
    StereoAdaptiveDepthGateConfig, StereoDepthGate, StereoDepthGateDiagnostics,
    StereoDepthGateState, StereoFeatureConfig, StereoVoFrontendConfig,
};

fn assert_feature_extractor<T: FeatureExtractor>() {}
fn assert_matcher<T: Matcher>() {}
fn assert_pose_estimator<T: PoseEstimator>() {}
fn assert_pose_refiner<T: PoseRefiner>() {}
fn assert_robust_pose_estimator<T: RobustPoseEstimator>() {}
fn assert_candidate_selector<T: CandidateSelector>() {}
fn assert_map_provider<T: MapProvider>() {}
fn assert_descriptor_provider<T: DescriptorProvider>() {}
fn assert_submap_selector<T: SubmapSelector<InMemoryMapProvider>>() {}
fn assert_motion_model<T: MotionModel>() {}
fn assert_triangulator<T: Triangulator>() {}
fn assert_local_refiner<T: LocalRefiner>() {}

#[test]
fn core_stable_candidate_paths_are_usable() {
    let camera_id: CameraId = 1;
    let frame_id: FrameId = 10;
    let landmark_id: LandmarkId = 20;

    let camera = Camera::pinhole(camera_id, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::identity();
    let point_world = Point3::new(0.0, 0.0, 5.0);
    let pixel = reproject(&camera, &pose, &point_world).expect("pinhole reprojection");

    let frame = Frame::new(frame_id, camera_id);
    let keyframe = Keyframe {
        frame: frame.clone(),
        observations: Vec::new(),
    };
    let observation = Observation {
        frame_id,
        landmark_id,
        keypoint_index: 0,
        xy: pixel,
    };
    let query = QueryImage {
        camera: camera.clone(),
        keypoints: vec![pixel],
        descriptors: vec![vec![1.0]],
    };
    let mut landmark = Landmark::new(landmark_id, point_world);
    landmark.descriptor = Some(vec![1.0]);

    let mut map = VisualMap::new();
    map.cameras.insert(camera_id, camera);
    map.keyframes.insert(frame_id, keyframe);
    map.landmarks.insert(landmark_id, landmark);

    let mut descriptors = LandmarkDescriptorStore::new();
    descriptors.insert(landmark_id, vec![1.0]);

    let validation: VisualMapValidationReport = map.validate_with_descriptors(Some(&descriptors));
    assert!(validation.is_valid());
    assert_eq!(query.keypoints.len(), 1);
    assert_eq!(observation.landmark_id, landmark_id);

    let _se3 = SE3::identity();
    let _so3 = SO3::identity();
    let _sim3 = Sim3::identity();
    let _camera_model = CameraModel::Pinhole;
    let _map_issue: Option<VisualMapValidationIssue> = None;
    let _localization_result: Option<LocalizationResult> = None;
    let _localization_success: Option<LocalizationSuccess> = None;
    let _failure_reason: Option<LocalizationFailureReason> = None;
    let _pose_diagnostics: Option<PoseEstimatorDiagnostics> = None;
    let _pose_failure_diagnostics: Option<PoseEstimationFailureDiagnostics> = None;
    let _pose_failure_reason: Option<PoseEstimationFailureReason> = None;
}

#[test]
fn replaceable_algorithm_boundaries_have_canonical_paths() {
    assert_feature_extractor::<ProvidedFeatureExtractor>();
    assert_matcher::<BruteForceMatcher>();
    assert_pose_estimator::<DltPnP>();
    assert_pose_refiner::<GaussNewtonPoseRefiner>();
    assert_robust_pose_estimator::<PnPRansac>();
    assert_candidate_selector::<AllLandmarksSelector>();
    assert_map_provider::<InMemoryMapProvider>();
    assert_descriptor_provider::<InMemoryMapProvider>();
    assert_submap_selector::<AllMapSelector>();
    assert_motion_model::<ConstantPoseMotionModel>();
    assert_triangulator::<LinearTriangulator>();
    assert_local_refiner::<NoopLocalRefiner>();

    let features =
        FeatureSet::new(vec![Point2::new(1.0, 2.0)], vec![vec![1.0]]).expect("valid feature set");
    let extractor = ProvidedFeatureExtractor::new(features.clone());
    assert_eq!(extractor.extract(&()).unwrap(), features);

    let matcher = BruteForceMatcher::default();
    let matches = matcher.match_descriptors(&[vec![1.0]], &[vec![1.0]]);
    assert_eq!(matches.len(), 1);

    let _correspondence: Option<Correspondence2D3D> = None;
    let _correspondence_set: Option<CorrespondenceSet> = None;
    let _correspondence_error: Option<CorrespondenceBuildError> = None;
    let _builder: Option<CorrespondenceBuilder<BruteForceMatcher>> = None;
    let _localizer: Option<ImageLocalizer<ProvidedFeatureExtractor>> = None;
    let _selectable_provider: Option<SelectableMapProvider<InMemoryMapProvider, AllMapSelector>> =
        None;
    let _fixed_selector = FixedLandmarkSelector::new(vec![1]);
    let _intersect_selector =
        IntersectCandidateSelector::new(AllLandmarksSelector, AllLandmarksSelector);
    let _radius_selector = RadiusLandmarkSelector::new(Point3::origin(), 10.0);
    let _fixed_submap_selector = FixedLandmarkSubmapSelector::new(vec![1]);
    let _radius_submap_selector = RadiusSubmapSelector::new(Point3::origin(), 10.0);
    let _prior_submap_selector = PriorSubmapSelector {
        prior: LocalizationPrior::none(),
    };
    let _localization_config = LocalizationConfig::default();
    let _localization_pipeline = LocalizationPipeline::default();
    let _frame_result: Option<FrameLocalizationResult> = None;
    let _map_stats: Option<MapProviderStats> = None;
    let _grayscale: Option<GrayscaleImage> = None;
}

#[test]
fn experimental_composition_paths_remain_documented_imports() {
    let _covis_config = CovisibilityLocalBaConfig::default();
    let _covis_online_config = OnlineSlamCovisibilityLocalBaConfig::default();
    let _covis_stats: Option<OnlineSlamCovisibilityLocalBaStats> = None;
    let _covis_selection: Option<CovisibilityLocalBaSelection> = None;
    let _refine_fn = refine_visual_map_with_covisibility_ba;
    let _select_fn = select_covisibility_local_ba_window;

    let _fusion_timestamp = Timestamp::from_nanoseconds(0);
    let _fusion_delta = TimeDelta::from_nanoseconds(1_000_000);
    let _timed_pose = TimedPose::new(Timestamp::from_nanoseconds(0), Pose::identity());
    let _timed_frame = TimedFrame::new(Timestamp::from_nanoseconds(0), Frame::new(1, 1));
    let _timed_value = Timed::new(Timestamp::from_nanoseconds(0), 1_u32);
    let _gnss = GnssMeasurement::new(Timestamp::from_nanoseconds(0), Point3::new(0.0, 0.0, 0.0));
    let _pose_prior = PosePriorMeasurement::new(Timestamp::from_nanoseconds(0), Pose::identity());
    let _imu = ImuMeasurement::new(
        Timestamp::from_nanoseconds(0),
        Vector3::zeros(),
        Vector3::zeros(),
    );
    let _prior_config = PriorConfig::default();
    let _measurement_buffer: MeasurementBuffer<GnssMeasurement> = MeasurementBuffer::new();
    let _frame_prior_source: Option<FramePriorSource<GnssMeasurement>> = None;
    let _position_covariance: Option<PositionCovariance> = None;
    let _pose_covariance: Option<PoseCovariance> = None;
    let _pose_covariance_matrix: PoseCovarianceMatrix = PoseCovarianceMatrix::identity();

    let _tracking_stats = TrackingStats::default();
    let _tracking_eval_config = TrackingEvaluationConfig::default();
    let _tracking_eval_result: Option<TrackingEvaluationResult> = None;
    let _trajectory_summary: Option<TrajectoryErrorSummary> = None;
    let _pose_trajectory = PoseTrajectory::new();

    let _online_slam_config = OnlineSlamConfig::default();
    let _online_slam: OnlineSlamPipeline<
        Tracker<LocalizationPipeline, ConstantPoseMotionModel>,
        LocalMappingPipeline<SimpleKeyframePolicy, LinearTriangulator>,
    > = OnlineSlamPipeline::default();

    let _stereo_frontend_config = StereoVoFrontendConfig::default();
    let _stereo_feature_config = StereoFeatureConfig::default();
    let _adaptive_gate_config = StereoAdaptiveDepthGateConfig::default();
    let _adaptive_gate = StereoDepthGate::adaptive();
    let _fixed_gate = StereoDepthGate::fixed();
    let _depth_gate_state = StereoDepthGateState::default();
    let _depth_gate_diagnostics: Option<StereoDepthGateDiagnostics> = None;
}

#[test]
fn prelude_exposes_common_application_surface() {
    use visloc_rs::prelude::{
        Camera as PreludeCamera, Landmark as PreludeLandmark,
        LocalizationPipeline as PreludePipeline, Pose as PreludePose,
        QueryImage as PreludeQueryImage, VisualMap as PreludeVisualMap,
    };

    let camera = PreludeCamera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = PreludePose::identity();
    let query = PreludeQueryImage {
        camera,
        keypoints: Vec::new(),
        descriptors: Vec::new(),
    };
    let map = PreludeVisualMap::new();
    let _landmark = PreludeLandmark::new(1, Point3::new(0.0, 0.0, 1.0));
    let _pipeline = PreludePipeline::default();
    assert!(query.keypoints.is_empty());
    assert!(map.landmarks.is_empty());
    assert_eq!(pose, PreludePose::default());
}
