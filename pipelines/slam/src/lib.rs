#![forbid(unsafe_code)]
//! Minimal online SLAM orchestration.
//!
//! This crate wires tracking and local mapping together. It is not a full SLAM
//! system, but it now hosts both a sparse pose-graph optimizer (translation-
//! only and full SE(3), with dense or sparse Cholesky solves) and a Schur-
//! complement bundle adjustment that jointly refines poses and landmarks
//! from 2D reprojection residuals.

pub mod bundle;
pub use bundle::{
    BaConfig, BaError, BaGeneralStereoObservation, BaGncResult, BaIterationStats, BaObservation,
    BaResult, BaStereoObservation, BiasRandomWalkFactor, BundleAdjustment, BundleAdjustmentRefiner,
    GravityPrior, NavigationStatePrior, PairwisePoseFactor, PerPoseGravityObservation,
    PerPoseGravityPrior, PositionPrior, PositionPriorObservation,
};

pub mod covisibility_ba;
pub use covisibility_ba::{
    behind_camera_optimized_landmark_ratio, fixed_to_optimized_ratio_satisfied,
    mean_selected_reprojection_px, refine_visual_map_with_covisibility_ba,
    refine_visual_map_with_covisibility_ba_and_neighbor_allowlist, required_fixed_keyframes,
    select_covisibility_local_ba_window,
    select_covisibility_local_ba_window_with_neighbor_allowlist, CovisibilityKeyframeScore,
    CovisibilityLocalBaConfig, CovisibilityLocalBaError, CovisibilityLocalBaResult,
    CovisibilityLocalBaSelection,
};

pub mod incremental_sfm;
pub use incremental_sfm::{
    incremental_sfm, IncrementalSfmConfig, IncrementalSfmError, IncrementalSfmResult,
    PairwiseMatches, SfmTrack,
};

pub mod imu_preintegration;
pub use imu_preintegration::{
    ImuNoiseModel, ImuPreintegratedDelta, ImuPreintegrationFactor, ImuPreintegrator, Matrix9,
};

pub mod g2o;
pub use g2o::{read_g2o, write_g2o, G2oError};

mod block_cholesky;
pub mod covariance;
pub mod dpvo_patch_ba;
pub use dpvo_patch_ba::{
    dpvo_ba, dpvo_ba_step, dpvo_pose_from_se3, flow_mag, reproject_patch_grid, se3_from_dpvo_pose,
    transform_point, DpvoBaConfig, DpvoBaError, DpvoBaProblem, DpvoEdge, DpvoIntrinsics, DpvoPatch,
};
pub mod dpvo_patch_graph;
pub use dpvo_patch_graph::{
    active_edge_triples, DpvoGraphError, DpvoGraphEdge, DpvoGraphFrame, DpvoPatchGraph, DpvoVoConfig,
};
pub mod dpvo_vi_ba;
pub use dpvo_vi_ba::{dpvo_vi_ba, dpvo_vi_ba_step, DpvoImuFactor, DpvoViBaSolution, DpvoViWindow};
pub mod dpvo_vo;
mod finite_difference;
pub mod gnc;
pub mod incremental_pose_graph;
pub mod map_atlas;
pub mod marginalization;
pub mod pcm;
mod reordering;
pub mod sparse_factor_graph;
pub mod sparsification;

pub(crate) use finite_difference::central_difference_projection_jacobian;

pub mod sim3_pose_graph;
pub use sim3_pose_graph::{
    Sim3Edge, Sim3Information, Sim3PoseGraph, Sim3PoseGraphConfig, Sim3PoseGraphIterationStats,
    Sim3PoseGraphResult,
};

pub mod stereo_vo_ba;
pub use stereo_vo_ba::{
    parse_stereo_vo_imu_samples_txt, reconstruct_stereo_vo_with_ba, refine_stereo_vo_with_ba,
    slice_imu_samples_for_keyframes, LandmarkInit, ReconstructedLandmark, StereoVoBaConfig,
    StereoVoBaError, StereoVoBaImuInput, StereoVoBaImuRefinement, StereoVoBaImuSample,
    StereoVoBaRefinement, StereoVoReconstruction,
};

pub mod online_stereo_vo_ba;
pub use online_stereo_vo_ba::{
    online_ba_imu_state_rows, write_online_ba_imu_state_csv, OnlineBaImuStateRow,
    OnlineBaTriggerStats, OnlineStereoVoBa, OnlineStereoVoBaConfig,
};

pub mod vo_loop_closure;
pub use vo_loop_closure::{
    close_loops_on_vo_trajectory, close_loops_on_vo_trajectory_with_globals,
    close_loops_on_vo_trajectory_with_globals_and_loop_matches,
    close_loops_on_vo_trajectory_with_loop_matches, detect_loop_candidates, LoopCandidatePair,
    LoopCandidateVerificationDiagnostic, VoLoopClosureConfig, VoLoopClosureError,
    VoLoopClosureResult,
};

pub mod online_slam_vi_ba;
pub use online_slam_vi_ba::{
    estimate_scale_from_factors, run_inertial_only_vi_ba, run_local_vi_ba,
    run_viba2_inertial_with_scale, AdaptiveVelocityGateConfig, InertialOnlyViBaStats,
    KeyframeImuState, OnlineSlamLocalBaConfig, OnlineSlamLocalBaState, OnlineSlamLocalBaStats,
    Viba2Config, Viba2Stats,
};

pub mod vi_initializer;
pub use vi_initializer::{
    StationaryRejectionReason, VisualInertialInitializationResult, VisualInertialInitializer,
    VisualInertialInitializerConfig,
};

pub mod vi_motion_initializer;
pub use vi_motion_initializer::{
    estimate_gravity_and_velocities, estimate_gyro_bias, BiasReleaseSchedule,
    GravityVelocityAlignment, GyroBiasAlignment, MotionBasedViInitializationResult,
    MotionBasedViInitializationStatus, MotionBasedViInitializer, MotionBasedViInitializerConfig,
    MotionBasedViRejectionReason,
};

pub mod online_slam_vi_init;
use online_slam_vi_init::OnlineSlamViInitState;
pub use online_slam_vi_init::{
    OnlineSlamConfigError, OnlineSlamViInitConfig, ViInitFallback, ViInitializationEvent,
    ViInitializationStatus,
};

pub mod online_slam_motion_vi_init;
use online_slam_motion_vi_init::OnlineSlamMotionViInitState;
pub use online_slam_motion_vi_init::{
    MotionViInitializationEvent, MotionViInitializationStatus, OnlineSlamMotionViInitConfig,
};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

use nalgebra::{
    DMatrix, DVector, Matrix3, Matrix6, Point2, Point3, Quaternion, Rotation3, UnitQuaternion,
    Vector3, Vector6,
};
use visloc_core::geometry::{Pose, Sim3, SE3};
use visloc_core::types::{Camera, Frame, Keyframe, Observation, VisualMap};
use visloc_localization::LocalizationPipeline;
use visloc_mapping::{
    AppliedMapUpdate, KeyframePolicy, LandmarkCandidate, LinearTriangulator, LocalMappingPipeline,
    LocalMappingResult, SimpleKeyframePolicy, Triangulator,
};
use visloc_tracking::{
    ConstantPoseMotionModel, FrameLocalizer, MotionModel, Tracker, TrackingConfig, TrackingResult,
};
use visloc_vision::features::FeatureSet;
use visloc_vision::matching::Matcher;
use visloc_vision::pnp::Correspondence2D3D;
use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};
use visloc_vision::two_view::{
    EightPointEssentialMatrixEstimator, RelativePoseEstimator, TwoViewCorrespondence,
};

pub mod loop_closure;
pub use loop_closure::*;

mod loop_gating;
pub(crate) use loop_gating::*;

mod loop_pose_information;

pub mod online_slam;
pub use online_slam::*;

pub mod pose_graph;
pub use pose_graph::*;

pub use map_atlas::{
    AtlasSubmap, CrossSubmapAlignmentConfig, CrossSubmapAlignmentResult,
    CrossSubmapBoundaryFactorResult, CrossSubmapCandidateDiagnostic,
    CrossSubmapCandidateFailureReason, CrossSubmapLandmarkMatch, CrossSubmapScaleEstimate,
    CrossSubmapWindowAlignmentResult, MapAtlas, MapAtlasError, MaterializedAtlas, SubmapId,
    SubmapIdRemap, SubmapMergeEvidence, SubmapMergeQuality, SubmapMergeVerificationConfig,
    VerifiedSubmapMerge,
};

pub use sparse_factor_graph::{
    SparseFactorGraph, SparseFactorGraphConfig, SparseFactorGraphUpdateStats,
    SparseFactorInactiveReason, SparseFactorKey, SparseFactorKind, SparseFactorMeasurement,
    SparseFactorState, SparseKeyframeFactor,
};

pub mod report;
pub use report::*;
