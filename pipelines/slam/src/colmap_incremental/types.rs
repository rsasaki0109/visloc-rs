//! Faithful port of COLMAP's rig/frame data model.
//!
//! Ported from (commit `64805cb870b574a569dccc34918d95a2db2b2fee`, pinned by
//! `docs/colmap_rig_mapper_port_plan.md` §0):
//! - `src/colmap/sensor/rig.h:49-222` — [`Rig`] (`sensor_t`/`data_t` id
//!   plumbing, reference-sensor convention, `SensorFromRig`).
//! - `src/colmap/scene/frame.h:44-228` and `frame.cc:34-119` — [`Frame`]
//!   (`SetCamFromWorld`, `SensorFromWorld`, `HasPose`/`ResetPose`).
//!
//! ## Pose convention
//!
//! Every pose in this module is `X_from_Y`, COLMAP's `Rigid3d` convention:
//! `X_from_Y.transform_point(p_Y) == p_X`. In particular `cam_from_world`
//! (a camera's absolute pose) maps a world point into that camera's frame,
//! exactly like COLMAP's `Image::CamFromWorld()` /
//! `Frame::SensorFromWorld()`. Composition `a.compose(b)` means "apply `b`
//! then `a`" (`(a ∘ b)(p) == a(b(p))`), matching COLMAP's `Rigid3d operator*`
//! — see [`visloc_core::geometry::SE3::compose`].
//!
//! This reuses the repository's existing SE(3) type
//! (`visloc_core::geometry::SE3`, already the pose representation for
//! `RigSensor::sensor_from_rig` in `crates/vision/src/pnp/generalized.rs`
//! and for rig transforms in `pipelines/slam/src/rig_sfm.rs`) rather than
//! inventing a new `Rigid3d` — see `docs/colmap_rig_mapper_port_plan.md`
//! §3.2's explicit instruction to reuse the repo's pose type.
//!
//! ## Deviation from COLMAP: no `rig_ptr_`
//!
//! COLMAP's `Frame` caches a raw, non-owning `Rig*` (`frame.h:131`,
//! `SetRigPtr`) so `SensorFromWorld`/`SetCamFromWorld` can be called with
//! just a `sensor_t`. Rust has no equivalent of a freely-aliased raw pointer
//! without `unsafe` (forbidden crate-wide, `#![forbid(unsafe_code)]` in
//! `pipelines/slam/src/lib.rs`) or a reference-counted/lifetime-carrying
//! design that would ripple through every caller. This port instead takes
//! the owning `&Rig` as an explicit parameter on [`Frame::sensor_from_world`]
//! and [`Frame::set_cam_from_world`] — semantically identical (same `Rig`
//! object, same lookup), just passed explicitly instead of cached.

use std::collections::{BTreeMap, BTreeSet};

use visloc_core::geometry::SE3;

/// `camera_t` (`util/types.h`): unique camera identifier.
pub type CameraT = u64;
/// `image_t`.
pub type ImageT = u64;
/// `frame_t`.
pub type FrameT = u64;
/// `rig_t`.
pub type RigT = u64;
/// `point3D_t`.
pub type Point3DT = u64;
/// `point2D_t` — a point2D's index within its image's keypoint list.
pub type Point2DT = usize;

/// `kInvalidCameraId` / `kInvalidImageId` / `kInvalidFrameId` / `kInvalidRigId`
/// (`util/types.h`): COLMAP uses `std::numeric_limits<T>::max()` as the
/// sentinel for "unset".
pub const INVALID_CAMERA_ID: CameraT = CameraT::MAX;
pub const INVALID_IMAGE_ID: ImageT = ImageT::MAX;
pub const INVALID_FRAME_ID: FrameT = FrameT::MAX;
pub const INVALID_RIG_ID: RigT = RigT::MAX;
pub const INVALID_POINT3D_ID: Point3DT = Point3DT::MAX;

/// `SensorType` (`sensor/rig.h`'s companion enum in `util/types.h`): COLMAP
/// distinguishes camera and IMU sensors within a rig; this port only ever
/// constructs `Camera` sensors (no IMU data in the visloc export this
/// module consumes), but keeps the full enum for a faithful `sensor_t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SensorType {
    Invalid,
    Camera,
    Imu,
}

/// `sensor_t` (`util/types.h`): a `(SensorType, id)` pair. `id` is the
/// underlying camera_t/imu_t within its type's own id space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SensorT {
    pub sensor_type: SensorType,
    pub id: u64,
}

impl SensorT {
    pub fn camera(camera_id: CameraT) -> Self {
        Self {
            sensor_type: SensorType::Camera,
            id: camera_id,
        }
    }

    pub const INVALID: SensorT = SensorT {
        sensor_type: SensorType::Invalid,
        id: u64::MAX,
    };
}

/// `data_t` (`util/types.h`): identifies one sensor measurement captured at
/// a frame instant — `sensor_id` (which sensor) plus `id` (that sensor's own
/// record id, here the `image_t` for a camera sensor).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DataT {
    pub sensor_id: SensorT,
    pub id: u64,
}

impl DataT {
    pub fn camera(camera_id: CameraT, image_id: ImageT) -> Self {
        Self {
            sensor_id: SensorT::camera(camera_id),
            id: image_id,
        }
    }
}

/// Port of `Rig` (`sensor/rig.h:49-222`): a set of rigidly-mounted sensors
/// with one reference sensor (identity pose in the rig frame) and, for every
/// other sensor, an optional `sensor_from_rig` extrinsic.
#[derive(Debug, Clone, PartialEq)]
pub struct Rig {
    rig_id: RigT,
    ref_sensor_id: SensorT,
    /// `sensors_from_rig_` (`rig.h:109`) — non-reference sensors only, exactly
    /// as COLMAP stores it (`NonRefSensors`).
    sensors_from_rig: BTreeMap<SensorT, Option<SE3>>,
}

impl Rig {
    pub fn new() -> Self {
        Self {
            rig_id: INVALID_RIG_ID,
            ref_sensor_id: SensorT::INVALID,
            sensors_from_rig: BTreeMap::new(),
        }
    }

    pub fn rig_id(&self) -> RigT {
        self.rig_id
    }

    pub fn set_rig_id(&mut self, rig_id: RigT) {
        self.rig_id = rig_id;
    }

    /// Port of `AddRefSensor` (`rig.h:57`). Must be called before any
    /// `add_sensor` call (COLMAP's own documented ordering requirement).
    pub fn add_ref_sensor(&mut self, ref_sensor_id: SensorT) {
        self.ref_sensor_id = ref_sensor_id;
    }

    /// Port of `AddSensor` (`rig.h:58-59`).
    pub fn add_sensor(&mut self, sensor_id: SensorT, sensor_from_rig: Option<SE3>) {
        self.sensors_from_rig.insert(sensor_id, sensor_from_rig);
    }

    /// Port of `HasSensor` (`rig.h:122-125`).
    pub fn has_sensor(&self, sensor_id: SensorT) -> bool {
        sensor_id == self.ref_sensor_id || self.sensors_from_rig.contains_key(&sensor_id)
    }

    /// Port of `NumSensors` (`rig.h:127-131`).
    pub fn num_sensors(&self) -> usize {
        let mut count = self.sensors_from_rig.len();
        if self.ref_sensor_id != SensorT::INVALID {
            count += 1;
        }
        count
    }

    /// Port of `RefSensorId` (`rig.h:133`).
    pub fn ref_sensor_id(&self) -> SensorT {
        self.ref_sensor_id
    }

    /// Port of `IsRefSensor` (`rig.h:135-137`).
    pub fn is_ref_sensor(&self, sensor_id: SensorT) -> bool {
        sensor_id == self.ref_sensor_id
    }

    /// Port of `HasSensorFromRig` (`rig.h:139-142`).
    pub fn has_sensor_from_rig(&self, sensor_id: SensorT) -> bool {
        sensor_id != self.ref_sensor_id
            && self
                .sensors_from_rig
                .get(&sensor_id)
                .is_some_and(Option::is_some)
    }

    /// Port of `SensorIds` (`rig.h:144-151`).
    pub fn sensor_ids(&self) -> BTreeSet<SensorT> {
        let mut ids: BTreeSet<SensorT> = self.sensors_from_rig.keys().copied().collect();
        ids.insert(self.ref_sensor_id);
        ids
    }

    /// Port of `NonRefSensors` (`rig.h:153-159`).
    pub fn non_ref_sensors(&self) -> &BTreeMap<SensorT, Option<SE3>> {
        &self.sensors_from_rig
    }

    /// Port of `SensorFromRig` (`rig.h:161-167`, `FindSensorFromRigOrThrow`).
    /// Panics for the reference sensor or an unknown/uncalibrated sensor,
    /// mirroring COLMAP's `THROW_CHECK`s.
    pub fn sensor_from_rig(&self, sensor_id: SensorT) -> SE3 {
        self.maybe_sensor_from_rig(sensor_id)
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "sensor ({:?}, {}) has no calibrated sensor_from_rig",
                    sensor_id.sensor_type, sensor_id.id
                )
            })
    }

    /// Port of `MaybeSensorFromRig` (`rig.h:169-176`,
    /// `FindSensorFromRigOrThrow`).
    pub fn maybe_sensor_from_rig(&self, sensor_id: SensorT) -> Option<&SE3> {
        assert!(
            sensor_id != self.ref_sensor_id,
            "the reference sensor does not have a SensorFromRig transformation, \
             which is fixed to identity"
        );
        self.sensors_from_rig
            .get(&sensor_id)
            .unwrap_or_else(|| {
                panic!(
                    "sensor ({:?}, {}) not found in the rig",
                    sensor_id.sensor_type, sensor_id.id
                )
            })
            .as_ref()
    }

    /// Port of `SetSensorFromRig` (`rig.h:178-185`).
    pub fn set_sensor_from_rig(&mut self, sensor_id: SensorT, sensor_from_rig: Option<SE3>) {
        assert!(
            sensor_id != self.ref_sensor_id,
            "the reference sensor does not have a SensorFromRig transformation, \
             which is fixed to identity"
        );
        assert!(
            self.sensors_from_rig.contains_key(&sensor_id),
            "sensor ({:?}, {}) not found in the rig",
            sensor_id.sensor_type,
            sensor_id.id
        );
        self.sensors_from_rig.insert(sensor_id, sensor_from_rig);
    }

    /// Port of `ResetSensorFromRig` (`rig.h:187-189`).
    pub fn reset_sensor_from_rig(&mut self, sensor_id: SensorT) {
        self.set_sensor_from_rig(sensor_id, None);
    }
}

impl Default for Rig {
    fn default() -> Self {
        Self::new()
    }
}

/// Port of `Frame` (`scene/frame.h:44-228`, `frame.cc:34-119`): the atomic
/// registration unit — one posed rig instant, with one `data_t` per captured
/// sensor measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    frame_id: FrameT,
    rig_id: RigT,
    data_ids: BTreeSet<DataT>,
    /// `rig_from_world_` (`frame.h:129`).
    rig_from_world: Option<SE3>,
}

impl Frame {
    pub fn new(frame_id: FrameT, rig_id: RigT) -> Self {
        Self {
            frame_id,
            rig_id,
            data_ids: BTreeSet::new(),
            rig_from_world: None,
        }
    }

    pub fn frame_id(&self) -> FrameT {
        self.frame_id
    }

    pub fn set_frame_id(&mut self, frame_id: FrameT) {
        self.frame_id = frame_id;
    }

    pub fn rig_id(&self) -> RigT {
        self.rig_id
    }

    pub fn set_rig_id(&mut self, rig_id: RigT) {
        self.rig_id = rig_id;
    }

    /// Port of `DataIds` (`frame.h:57`).
    pub fn data_ids(&self) -> &BTreeSet<DataT> {
        &self.data_ids
    }

    /// Port of `AddDataId` (`frame.h:58`, `frame.cc` inline). Does not
    /// re-implement COLMAP's `has_final_data_ids_` finalize-lock (no caller
    /// in this module needs it yet); see module doc.
    pub fn add_data_id(&mut self, data_id: DataT) {
        self.data_ids.insert(data_id);
    }

    pub fn num_data_ids(&self) -> usize {
        self.data_ids.len()
    }

    /// Port of `HasDataId` (`frame.h:62`).
    pub fn has_data_id(&self, data_id: DataT) -> bool {
        self.data_ids.contains(&data_id)
    }

    /// Port of `DataIds(SensorType)` (`frame.h:104-111`).
    pub fn data_ids_of_type(&self, sensor_type: SensorType) -> impl Iterator<Item = DataT> + '_ {
        self.data_ids
            .iter()
            .copied()
            .filter(move |data_id| data_id.sensor_id.sensor_type == sensor_type)
    }

    /// Port of `ImageIds` (`frame.h:114`): every `image_t` captured by this
    /// frame's camera sensors.
    pub fn image_ids(&self) -> impl Iterator<Item = ImageT> + '_ {
        self.data_ids_of_type(SensorType::Camera)
            .map(|data_id| data_id.id)
    }

    /// Port of `HasPose` (`frame.h:94`).
    pub fn has_pose(&self) -> bool {
        self.rig_from_world.is_some()
    }

    /// Port of `ResetPose` (`frame.h:95`).
    pub fn reset_pose(&mut self) {
        self.rig_from_world = None;
    }

    /// Port of `RigFromWorld` (`frame.h:88-89`, const accessor). Panics if
    /// unposed, mirroring COLMAP's `THROW_CHECK`.
    pub fn rig_from_world(&self) -> &SE3 {
        self.rig_from_world
            .as_ref()
            .expect("frame does not have a valid pose")
    }

    /// Port of `MaybeRigFromWorld` (`frame.h:90-91`).
    pub fn maybe_rig_from_world(&self) -> Option<&SE3> {
        self.rig_from_world.as_ref()
    }

    /// Port of `SetRigFromWorld` (`frame.h:92-93`).
    pub fn set_rig_from_world(&mut self, rig_from_world: SE3) {
        self.rig_from_world = Some(rig_from_world);
    }

    /// Port of `SensorFromWorld` (`frame.h:98`, `frame.cc:210-217`): the
    /// composed absolute pose of one of this frame's sensors. `rig` must be
    /// the `Rig` this frame's `rig_id` refers to (see module doc's "no
    /// `rig_ptr_`" note).
    pub fn sensor_from_world(&self, rig: &Rig, sensor_id: SensorT) -> SE3 {
        let rig_from_world = self.rig_from_world().clone();
        if rig.is_ref_sensor(sensor_id) {
            rig_from_world
        } else {
            rig.sensor_from_rig(sensor_id).compose(&rig_from_world)
        }
    }

    /// Port of `SetCamFromWorld` (`frame.cc:85-94`) — **verbatim**, the
    /// crux of COLMAP's metric guarantee (§1.1 of
    /// `docs/colmap_rig_mapper_port_plan.md`): a plain, single-camera
    /// absolute-pose estimate for a non-reference camera is converted into
    /// the frame's one shared `rig_from_world` by composing with the fixed,
    /// metric `sensor_from_rig` baseline — the composition cannot introduce
    /// or absorb a scale factor. `rig` must be the `Rig` this frame's
    /// `rig_id` refers to (see module doc's "no `rig_ptr_`" note).
    pub fn set_cam_from_world(&mut self, rig: &Rig, camera_id: CameraT, cam_from_world: SE3) {
        let sensor_id = SensorT::camera(camera_id);
        if rig.is_ref_sensor(sensor_id) {
            self.set_rig_from_world(cam_from_world);
        } else {
            let cam_from_rig = rig.sensor_from_rig(sensor_id);
            self.set_rig_from_world(cam_from_rig.inverse().compose(&cam_from_world));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    fn stereo_rig() -> Rig {
        // Mirrors the OpenLORIS manifest convention (S0 identity = ref,
        // S1 = calibrated stereo baseline), openloris-tier1000-rig-manifest-v1.txt.
        let mut rig = Rig::new();
        rig.set_rig_id(0);
        rig.add_ref_sensor(SensorT::camera(1));
        let cam2_from_rig = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.05, 0.0),
            Vector3::new(-0.06, 0.0, 0.0),
        );
        rig.add_sensor(SensorT::camera(2), Some(cam2_from_rig));
        rig
    }

    #[test]
    fn rig_sensor_lookups() {
        let rig = stereo_rig();
        assert_eq!(rig.num_sensors(), 2);
        assert!(rig.is_ref_sensor(SensorT::camera(1)));
        assert!(!rig.is_ref_sensor(SensorT::camera(2)));
        assert!(rig.has_sensor(SensorT::camera(1)));
        assert!(rig.has_sensor(SensorT::camera(2)));
        assert!(!rig.has_sensor(SensorT::camera(3)));
        assert!(rig.has_sensor_from_rig(SensorT::camera(2)));
        assert!(!rig.has_sensor_from_rig(SensorT::camera(1)));
        assert_eq!(
            rig.sensor_ids(),
            BTreeSet::from([SensorT::camera(1), SensorT::camera(2)])
        );
    }

    #[test]
    #[should_panic(expected = "reference sensor")]
    fn sensor_from_rig_panics_for_ref_sensor() {
        let rig = stereo_rig();
        let _ = rig.sensor_from_rig(SensorT::camera(1));
    }

    #[test]
    fn set_cam_from_world_ref_sensor_is_identity_composition() {
        // Frame.cc:88-89 — the reference sensor's cam_from_world IS
        // rig_from_world, no composition.
        let rig = stereo_rig();
        let mut frame = Frame::new(0, 0);
        let cam_from_world = SE3::new(
            UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3),
            Vector3::new(1.0, 2.0, 3.0),
        );
        frame.set_cam_from_world(&rig, 1, cam_from_world.clone());
        assert!(frame.has_pose());
        assert_se3_eq(frame.rig_from_world(), &cam_from_world);
        // SensorFromWorld for the ref sensor returns RigFromWorld() directly.
        let recovered = frame.sensor_from_world(&rig, SensorT::camera(1));
        assert_se3_eq(&recovered, &cam_from_world);
    }

    #[test]
    fn set_cam_from_world_non_ref_sensor_composes_fixed_baseline() {
        // Frame.cc:90-93 — RigFromWorld = Inverse(cam_from_rig) * cam_from_world.
        let rig = stereo_rig();
        let mut frame = Frame::new(0, 0);
        let cam2_from_world = SE3::new(
            UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3),
            Vector3::new(1.0, 2.0, 3.0),
        );
        frame.set_cam_from_world(&rig, 2, cam2_from_world.clone());
        assert!(frame.has_pose());

        // Hand-computed expectation: rig_from_world = inverse(cam2_from_rig) * cam2_from_world.
        let cam2_from_rig = rig.sensor_from_rig(SensorT::camera(2));
        let expected_rig_from_world = cam2_from_rig.inverse().compose(&cam2_from_world);
        assert_se3_eq(frame.rig_from_world(), &expected_rig_from_world);

        // Round trip: SensorFromWorld(cam2) recomposes back to cam2_from_world.
        let recovered = frame.sensor_from_world(&rig, SensorT::camera(2));
        assert_se3_eq(&recovered, &cam2_from_world);

        // And the reference camera's world pose is NOT cam2_from_world — it is
        // the rig pose (identity baseline), the other camera 6cm/ ~a few
        // degrees away.
        let cam1_from_world = frame.sensor_from_world(&rig, SensorT::camera(1));
        assert!((cam1_from_world.translation - cam2_from_world.translation).norm() > 1e-3);
    }

    #[test]
    fn reset_pose_clears_has_pose() {
        let rig = stereo_rig();
        let mut frame = Frame::new(0, 0);
        frame.set_cam_from_world(&rig, 1, SE3::identity());
        assert!(frame.has_pose());
        frame.reset_pose();
        assert!(!frame.has_pose());
        assert!(frame.maybe_rig_from_world().is_none());
    }

    #[test]
    fn frame_image_ids_filters_camera_sensors() {
        let mut frame = Frame::new(7, 0);
        frame.add_data_id(DataT::camera(1, 100));
        frame.add_data_id(DataT::camera(2, 101));
        let mut ids: Vec<ImageT> = frame.image_ids().collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![100, 101]);
        assert_eq!(frame.num_data_ids(), 2);
        assert!(frame.has_data_id(DataT::camera(1, 100)));
        assert!(!frame.has_data_id(DataT::camera(1, 999)));
    }

    fn assert_se3_eq(a: &SE3, b: &SE3) {
        let dt = (a.translation - b.translation).norm();
        let dr = (a.rotation.inverse() * b.rotation).angle();
        assert!(dt < 1e-9, "translation differs: {dt}");
        assert!(dr < 1e-9, "rotation differs: {dr}");
    }
}
