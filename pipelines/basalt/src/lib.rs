// Basalt parity code remains subject to the crate-wide unsafe-code lint. The
// pinned Eigen packet evaluation order is reproduced with safe scalar
// operations where bit-level compatibility requires it.
#![deny(unsafe_code)]
#![recursion_limit = "256"]
// `dead_code` and the style-only clippy lints this crate suppresses are
// configured package-wide in Cargo.toml's `[lints]` table (rather than here)
// so the same allow-list also covers this crate's `examples/`/`tests/`
// binaries, which are separate crate roots that inner `#![allow]` attributes
// in this file would not reach.
//! Basalt-compatible VI-SLAM foundations.
//!
//! The crate deliberately owns its calibration, camera, timestamp, and IMU
//! contracts so the later Basalt port does not accidentally inherit the
//! repository's generic descriptor/PnP pipeline conventions. Coordinate
//! directions and interval semantics are documented on the public types. The
//! raw-u16 pyramid, Pattern51 patch/Jacobian, and SE(2) update primitives feed
//! the dedicated direct KLT stream, which remains separate from the
//! repository's generic descriptor/PnP optical-flow path.
//!
//! Upstream provenance is recorded in [`provenance`] and
//! [`../PROVENANCE.md`](https://github.com/VladyslavUsenko/basalt).

pub mod adapter;
pub mod calibration;
pub mod camera;
pub mod config;
pub mod euroc;
pub mod fast;
pub mod imu;
pub mod initialization;
pub mod mapper;
// Physically lives under `mapper/` (`pipelines/basalt/src/mapper/online.rs`,
// per `docs/basalt_online_mapper_design.md`) but is declared here rather
// than with a `mod` line inside `mapper/mod.rs` so that file -- the pinned
// offline/parity module root -- stays byte-for-byte untouched.
#[path = "mapper/online.rs"]
pub mod mapper_online;
pub mod patch;
pub mod pattern;
pub mod provenance;
pub mod pyramid;
pub mod rt_priority;
pub mod stream;
pub mod streaming;
pub mod time;
pub mod timing;
pub mod types;
pub mod update;
pub mod vio;

pub use adapter::{
    direct_klt_config, vio_estimator_from_calibration, BasaltAdapterError, BasaltAdapterOutput,
    BasaltVioEstimatorAdapter,
};
pub use calibration::{BasaltCalibration, CalibrationError};
pub use camera::{CameraModelError, DoubleSphereCamera};
pub use euroc::{EurocImageEntry, EurocReaderError, EurocSensorDataset, EurocSensorFrame};
pub use fast::{GridFastConfig, GridFastDetector};
pub use patch::{
    MeanNormalizedPatch51, PatchData51, PatchInverseJacobian51, PatchJacobian51, PatchResidualError,
};
pub use pattern::Pattern51;
pub use pyramid::{ImageError, RawU16Image, RawU16Pyramid};
pub use stream::{
    DirectKltConfig, DirectKltStream, KltFailure, MultiCameraFlowOptions, RejectReason,
    RejectReasonCounters, StereoFrame, StereoMatchingGuess, StreamError, TrackFrameOutput,
    TrackStage,
};
pub use time::{select_imu_interval, ImuInterval, TimeInterval, TimeIntervalError};
pub use timing::{TimingBreakdown, TimingBucket, TimingStat};
pub use types::{
    BasaltFrame, BasaltNavState, CameraId, FrameId, ImuSample, TimestampNs, TrackId,
    TrackObservation,
};
pub use update::{AffineCompact2f, Se2, Se2UpdateError, Se2f};
