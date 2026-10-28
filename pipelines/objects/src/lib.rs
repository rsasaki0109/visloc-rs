#![forbid(unsafe_code)]
//! Image-space multi-object tracking: the first stage of metric object
//! estimation from monocular VI-SLAM (`docs/object_metric_slam_plan.md`,
//! Phase 1).
//!
//! [`ByteTracker`] links per-frame detections into tracks with stable IDs,
//! the ByteTrack way: a box Kalman filter ([`BoxKalmanFilter`]), IoU costs,
//! an optimal assignment ([`assign_with_cost_limit`]), and a second pass over
//! the low-score detections that keeps partly occluded objects. It takes
//! [`Detection`]s from any [`ObjectDetector`]: [`ReplayDetector`] replays
//! stored boxes, and [`YoloxOnnxDetector`] runs a YOLOX model with ONNX
//! Runtime (the `onnx-inference` feature; [`yolox`] has its pre- and
//! post-processing). [`MotAccumulator`] scores tracks against ground truth
//! with the CLEAR MOT metrics.
//!
//! With metric camera poses from the VIO, the tracks become metric objects
//! (Phase 3): [`metric`] triangulates static objects and fits upright boxes,
//! and [`ground`] puts objects that stand on the ground, moving or not, on
//! it for their height, position and speed.
//!
//! ```
//! use visloc_objects::{BBox, ByteTracker, Detection};
//!
//! let mut tracker = ByteTracker::default();
//! for frame in 0..3 {
//!     let x = 100.0 + 5.0 * frame as f64;
//!     let detections = [Detection::new(BBox::from_xywh(x, 50.0, 40.0, 80.0), 0.9, 0)];
//!     let tracks = tracker.update(&detections);
//!     assert_eq!(tracks.len(), 1);
//!     assert_eq!(tracks[0].id.0, 1);
//! }
//! ```

pub mod assignment;
pub mod bbox;
pub mod byte_track;
pub mod detector;
pub mod ground;
pub mod image;
pub mod kalman;
pub mod metric;
pub mod mot_eval;
pub mod yolox;
pub mod yolox_onnx;

pub use assignment::{assign_with_cost_limit, Assignment};
pub use bbox::{BBox, Detection};
pub use byte_track::{ByteTrackConfig, ByteTracker, TrackId, TrackedObject};
pub use detector::{ObjectDetector, ReplayDetector, ReplayExhausted};
pub use ground::{
    estimate_ground_height, place_on_ground, place_on_local_ground, plausible_height,
    summarize_ground_track, BoxRays, GroundObservation, GroundParams, GroundTrack, GroundView,
};
pub use image::{ImageSizeError, RgbImage};
pub use kalman::{BoxKalmanFilter, BoxKalmanState};
pub use metric::{
    estimate_static_object, fit_upright_box, plausible_size, triangulate, BearingObservation,
    BoxFitParams, ObjectEstimate, ScaleSource, TriangulationParams, UprightBox,
};
pub use mot_eval::{MotAccumulator, MotSummary};
pub use yolox::{YoloxConfig, COCO_CLASSES, MOVABLE_COCO_CLASSES};
pub use yolox_onnx::{YoloxOnnxDetector, YoloxOnnxError};
