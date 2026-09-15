//! Faithful (partial, C1-scoped) port of `scene/reconstruction.h`'s
//! [`Reconstruction`] container plus the `scene/image.h`/`scene/point3d.h`
//! types it holds.
//!
//! `docs/colmap_rig_mapper_port_plan.md`'s Lead review scopes milestone C1 to
//! "data model and inputs" — this module ports the *type* (containers,
//! accessors, registration bookkeeping) exactly as declared in
//! `reconstruction.h:56-330` (fetched at the commit pinned by that plan's
//! §0), but does **not** port `reconstruction.cc`'s algorithmic bodies
//! (`Normalize`, `Crop`, `MergePoints3D`, colorization, `TearDown`, binary
//! I/O …) — those are out of C1's scope (`RegisterNextImage`/BA/filtering
//! land in C2/C3 per the Lead review's milestone table) and `reconstruction.cc`
//! was not part of the source set the task asked to read for C1 (only
//! `reconstruction.h`, for the types). `RegisterFrame`/`DeRegisterFrame` here
//! implement the header-documented contract (`reg_frame_ids_` bookkeeping,
//! `reconstruction.h:181-185,298-300`) without the additional point3D/track
//! side effects a full incremental mapper would need — those become relevant
//! only once C2 actually deletes/re-triangulates on deregistration.

use std::collections::BTreeMap;

use nalgebra::{Point2, Point3};

use visloc_core::geometry::Pose;
pub use visloc_core::types::Camera;
use visloc_io::colmap::{write_colmap_reconstruction_for_3dgs_with_cameras, ColmapError};
use visloc_vision::features::FeatureSet;

use super::types::{
    CameraT, Frame, FrameT, ImageT, Point2DT, Point3DT, Rig, RigT, SensorT, INVALID_POINT3D_ID,
};

/// `Point3D::error` sentinel for "not yet computed"
/// (COLMAP's `kInvalidPoint3DError = -1.0`, `scene/point3d.h`).
pub const INVALID_POINT3D_ERROR: f64 = -1.0;

/// Port of `Point2D` (`scene/point2d.h`): one keypoint plus its optional
/// track membership.
#[derive(Debug, Clone, PartialEq)]
pub struct Point2D {
    pub xy: Point2<f64>,
    pub point3d_id: Option<Point3DT>,
}

impl Point2D {
    pub fn new(xy: Point2<f64>) -> Self {
        Self {
            xy,
            point3d_id: None,
        }
    }

    /// Port of `Point2D::HasPoint3D` (`scene/point2d.h`).
    pub fn has_point3d(&self) -> bool {
        self.point3d_id.is_some()
    }
}

/// Port of `class Image` (`scene/image.h`): one camera exposure within a
/// [`Frame`], its detected keypoints, and each keypoint's optional 3D-point
/// membership.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    pub image_id: ImageT,
    pub camera_id: CameraT,
    pub frame_id: FrameT,
    pub name: String,
    pub points2d: Vec<Point2D>,
}

impl Image {
    pub fn new(image_id: ImageT, camera_id: CameraT, frame_id: FrameT, name: String) -> Self {
        Self {
            image_id,
            camera_id,
            frame_id,
            name,
            points2d: Vec::new(),
        }
    }

    pub fn num_points2d(&self) -> usize {
        self.points2d.len()
    }

    /// Port of `Image::NumPoints3D` (`scene/image.h`): keypoints that
    /// currently belong to a triangulated 3D point.
    pub fn num_points3d(&self) -> usize {
        self.points2d.iter().filter(|p| p.has_point3d()).count()
    }
}

/// Port of `TrackElement` (`scene/track.h`): one `(image, point2D)`
/// observation of a [`Point3D`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackElement {
    pub image_id: ImageT,
    pub point2d_idx: Point2DT,
}

/// Port of `struct Point3D` (`scene/point3d.h`): a triangulated landmark and
/// its observation track.
#[derive(Debug, Clone, PartialEq)]
pub struct Point3D {
    pub xyz: Point3<f64>,
    pub color: [u8; 3],
    pub track: Vec<TrackElement>,
    /// Mean reprojection error in pixels, or [`INVALID_POINT3D_ERROR`] if
    /// not yet computed.
    pub error: f64,
}

impl Point3D {
    pub fn new(xyz: Point3<f64>) -> Self {
        Self {
            xyz,
            color: [0, 0, 0],
            track: Vec::new(),
            error: INVALID_POINT3D_ERROR,
        }
    }
}

/// Port of `class Reconstruction` (`scene/reconstruction.h:56-330`),
/// C1-scoped per the module doc above.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Reconstruction {
    rigs: BTreeMap<RigT, Rig>,
    cameras: BTreeMap<CameraT, Camera>,
    frames: BTreeMap<FrameT, Frame>,
    images: BTreeMap<ImageT, Image>,
    points3d: BTreeMap<Point3DT, Point3D>,
    /// `reg_frame_ids_` (`reconstruction.h:298-300`): a vector (not a set),
    /// exactly as COLMAP documents, so registration order is preserved.
    reg_frame_ids: Vec<FrameT>,
    next_point3d_id: Point3DT,
}

impl Reconstruction {
    pub fn new() -> Self {
        Self {
            next_point3d_id: 1,
            ..Default::default()
        }
    }

    // ---- Counts (`reconstruction.h:64-71,322-328`) ----------------------

    pub fn num_rigs(&self) -> usize {
        self.rigs.len()
    }
    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }
    /// Port of `NumRegFrames` (`reconstruction.h:68,322`).
    pub fn num_reg_frames(&self) -> usize {
        self.reg_frame_ids.len()
    }
    /// Port of `NumRegImages` (`reconstruction.h:69`): images whose frame is
    /// registered.
    pub fn num_reg_images(&self) -> usize {
        self.images
            .values()
            .filter(|image| self.is_image_registered(image.image_id))
            .count()
    }
    pub fn num_images(&self) -> usize {
        self.images.len()
    }
    /// Port of `NumPoints3D` (`reconstruction.h:71,328`).
    pub fn num_points3d(&self) -> usize {
        self.points3d.len()
    }

    // ---- Accessors --------------------------------------------------

    pub fn rig(&self, rig_id: RigT) -> &Rig {
        self.rigs
            .get(&rig_id)
            .unwrap_or_else(|| panic!("rig {rig_id} does not exist"))
    }
    pub fn camera(&self, camera_id: CameraT) -> &Camera {
        self.cameras
            .get(&camera_id)
            .unwrap_or_else(|| panic!("camera {camera_id} does not exist"))
    }
    pub fn frame(&self, frame_id: FrameT) -> &Frame {
        self.frames
            .get(&frame_id)
            .unwrap_or_else(|| panic!("frame {frame_id} does not exist"))
    }
    pub fn frame_mut(&mut self, frame_id: FrameT) -> &mut Frame {
        self.frames
            .get_mut(&frame_id)
            .unwrap_or_else(|| panic!("frame {frame_id} does not exist"))
    }
    pub fn image(&self, image_id: ImageT) -> &Image {
        self.images
            .get(&image_id)
            .unwrap_or_else(|| panic!("image {image_id} does not exist"))
    }
    pub fn image_mut(&mut self, image_id: ImageT) -> &mut Image {
        self.images
            .get_mut(&image_id)
            .unwrap_or_else(|| panic!("image {image_id} does not exist"))
    }
    pub fn point3d(&self, point3d_id: Point3DT) -> &Point3D {
        self.points3d
            .get(&point3d_id)
            .unwrap_or_else(|| panic!("point3D {point3d_id} does not exist"))
    }
    pub fn point3d_mut(&mut self, point3d_id: Point3DT) -> &mut Point3D {
        self.points3d
            .get_mut(&point3d_id)
            .unwrap_or_else(|| panic!("point3D {point3d_id} does not exist"))
    }
    /// All current point3D ids, as a snapshot `Vec` (COLMAP's `Point3DIds()`
    /// returns a set; callers here only ever iterate it).
    pub fn point3d_ids(&self) -> Vec<Point3DT> {
        self.points3d.keys().copied().collect()
    }

    pub fn rigs(&self) -> &BTreeMap<RigT, Rig> {
        &self.rigs
    }
    pub fn cameras(&self) -> &BTreeMap<CameraT, Camera> {
        &self.cameras
    }
    pub fn frames(&self) -> &BTreeMap<FrameT, Frame> {
        &self.frames
    }
    pub fn images(&self) -> &BTreeMap<ImageT, Image> {
        &self.images
    }
    pub fn points3d(&self) -> &BTreeMap<Point3DT, Point3D> {
        &self.points3d
    }
    /// Port of `RegFrameIds` (`reconstruction.h:91`).
    pub fn reg_frame_ids(&self) -> &[FrameT] {
        &self.reg_frame_ids
    }

    pub fn exists_rig(&self, rig_id: RigT) -> bool {
        self.rigs.contains_key(&rig_id)
    }
    pub fn exists_camera(&self, camera_id: CameraT) -> bool {
        self.cameras.contains_key(&camera_id)
    }
    pub fn exists_frame(&self, frame_id: FrameT) -> bool {
        self.frames.contains_key(&frame_id)
    }
    pub fn exists_image(&self, image_id: ImageT) -> bool {
        self.images.contains_key(&image_id)
    }
    pub fn exists_point3d(&self, point3d_id: Point3DT) -> bool {
        self.points3d.contains_key(&point3d_id)
    }

    /// Not a literal COLMAP method name (COLMAP checks registration via
    /// `Frame::HasPose` on `image.FramePtr()`) — a direct, documented
    /// convenience matching the task's requested surface. Mirrors
    /// `incremental_mapper.cc`'s registration unit being the *frame*, not the
    /// image (§1.4 of the port plan): an image is registered iff its frame
    /// has a pose.
    pub fn is_image_registered(&self, image_id: ImageT) -> bool {
        let image = self.image(image_id);
        self.frames
            .get(&image.frame_id)
            .is_some_and(Frame::has_pose)
    }

    // ---- Mutators (`reconstruction.h:119-185`) ---------------------------

    pub fn add_rig(&mut self, rig: Rig) {
        let rig_id = rig.rig_id();
        let previous = self.rigs.insert(rig_id, rig);
        assert!(previous.is_none(), "rig {rig_id} already exists");
    }

    pub fn add_camera(&mut self, camera: Camera) {
        let camera_id = camera.id;
        let previous = self.cameras.insert(camera_id, camera);
        assert!(previous.is_none(), "camera {camera_id} already exists");
    }

    pub fn add_frame(&mut self, frame: Frame) {
        assert!(
            self.exists_rig(frame.rig_id()),
            "frame {}'s rig {} must be added before the frame",
            frame.frame_id(),
            frame.rig_id()
        );
        let frame_id = frame.frame_id();
        let previous = self.frames.insert(frame_id, frame);
        assert!(previous.is_none(), "frame {frame_id} already exists");
    }

    pub fn add_image(&mut self, image: Image) {
        assert!(
            self.exists_camera(image.camera_id),
            "image {}'s camera {} must be added before the image",
            image.image_id,
            image.camera_id
        );
        assert!(
            self.exists_frame(image.frame_id),
            "image {}'s frame {} must be added before the image",
            image.image_id,
            image.frame_id
        );
        let image_id = image.image_id;
        let previous = self.images.insert(image_id, image);
        assert!(previous.is_none(), "image {image_id} already exists");
    }

    /// Port of `AddPoint3D(point3D_id, point3D)` (`reconstruction.h:150`).
    pub fn add_point3d_with_id(&mut self, point3d_id: Point3DT, point3d: Point3D) {
        let previous = self.points3d.insert(point3d_id, point3d);
        assert!(previous.is_none(), "point3D {point3d_id} already exists");
        if point3d_id != INVALID_POINT3D_ID && point3d_id >= self.next_point3d_id {
            self.next_point3d_id = point3d_id + 1;
        }
        self.link_track(point3d_id);
    }

    /// Port of `AddPoint3D(xyz, track, color)` (`reconstruction.h:153-156`):
    /// assigns a fresh id and returns it.
    pub fn add_point3d(&mut self, xyz: Point3<f64>, track: Vec<TrackElement>) -> Point3DT {
        let point3d_id = self.next_point3d_id;
        self.next_point3d_id += 1;
        let mut point3d = Point3D::new(xyz);
        point3d.track = track;
        self.points3d.insert(point3d_id, point3d);
        self.link_track(point3d_id);
        point3d_id
    }

    fn link_track(&mut self, point3d_id: Point3DT) {
        let track = self.points3d.get(&point3d_id).unwrap().track.clone();
        for element in track {
            if let Some(image) = self.images.get_mut(&element.image_id) {
                if let Some(point2d) = image.points2d.get_mut(element.point2d_idx) {
                    point2d.point3d_id = Some(point3d_id);
                }
            }
        }
    }

    /// Port of `AddObservation` (`reconstruction.h:159`).
    pub fn add_observation(&mut self, point3d_id: Point3DT, track_el: TrackElement) {
        let point3d = self
            .points3d
            .get_mut(&point3d_id)
            .unwrap_or_else(|| panic!("point3D {point3d_id} does not exist"));
        point3d.track.push(track_el);
        if let Some(image) = self.images.get_mut(&track_el.image_id) {
            if let Some(point2d) = image.points2d.get_mut(track_el.point2d_idx) {
                point2d.point3d_id = Some(point3d_id);
            }
        }
    }

    /// Approximation of `Reconstruction::MergePoints3D`
    /// (`reconstruction.cc`, not fetched for this port — see module doc;
    /// `docs/colmap_rig_mapper_port_plan.md` scoped `reconstruction.cc`'s
    /// algorithmic bodies out of C1, and this port continues that for the
    /// one call site that needs it, `IncrementalTriangulator::MergeTracks`).
    /// Combines the two points' tracks and takes the track-length-weighted
    /// mean position/color, keeping `id1`'s slot; matches the documented
    /// COLMAP contract ("the merged point's position is the weighted
    /// average of the two previous points' positions, weighted by their
    /// track lengths") without reproducing its exact source.
    pub fn merge_points3d(&mut self, id1: Point3DT, id2: Point3DT) -> Point3DT {
        let p1 = self
            .points3d
            .remove(&id1)
            .unwrap_or_else(|| panic!("point3D {id1} does not exist"));
        let p2 = self
            .points3d
            .remove(&id2)
            .unwrap_or_else(|| panic!("point3D {id2} does not exist"));
        let n1 = p1.track.len() as f64;
        let n2 = p2.track.len() as f64;
        let total = n1 + n2;
        let xyz = Point3::from((p1.xyz.coords * n1 + p2.xyz.coords * n2) / total);
        let mut color = [0u8; 3];
        for ((out, &c1), &c2) in color.iter_mut().zip(&p1.color).zip(&p2.color) {
            *out = (((c1 as f64) * n1 + (c2 as f64) * n2) / total) as u8;
        }
        let mut merged = Point3D::new(xyz);
        merged.color = color;
        merged.track = p1.track;
        merged.track.extend(p2.track);
        self.points3d.insert(id1, merged);
        self.link_track(id1);
        id1
    }

    /// Port of `DeletePoint3D` (`reconstruction.h:167`): removes the point
    /// and clears every track observation's `point3D_id` back-reference.
    pub fn delete_point3d(&mut self, point3d_id: Point3DT) {
        let Some(point3d) = self.points3d.remove(&point3d_id) else {
            panic!("point3D {point3d_id} does not exist");
        };
        for element in point3d.track {
            if let Some(image) = self.images.get_mut(&element.image_id) {
                if let Some(point2d) = image.points2d.get_mut(element.point2d_idx) {
                    if point2d.point3d_id == Some(point3d_id) {
                        point2d.point3d_id = None;
                    }
                }
            }
        }
    }

    /// Port of `RegisterFrame` (`reconstruction.h:182`). Idempotent, exactly
    /// as COLMAP: registering an already-registered frame is a no-op on
    /// `reg_frame_ids_`.
    pub fn register_frame(&mut self, frame_id: FrameT) {
        assert!(
            self.exists_frame(frame_id),
            "frame {frame_id} does not exist"
        );
        if !self.reg_frame_ids.contains(&frame_id) {
            self.reg_frame_ids.push(frame_id);
        }
    }

    /// Port of `DeRegisterFrame` (`reconstruction.h:185`). C1-scoped (see
    /// module doc): resets the frame's pose and drops it from
    /// `reg_frame_ids_`; does not delete observed points3D (that side effect
    /// lives in `reconstruction.cc`, out of C1's ported surface).
    pub fn deregister_frame(&mut self, frame_id: FrameT) {
        assert!(
            self.exists_frame(frame_id),
            "frame {frame_id} does not exist"
        );
        self.reg_frame_ids.retain(|&id| id != frame_id);
        self.frame_mut(frame_id).reset_pose();
    }

    /// Port of `FindImageWithName` (`reconstruction.h:222-223`): linear
    /// search, exactly as COLMAP documents.
    pub fn find_image_with_name(&self, name: &str) -> Option<&Image> {
        self.images.values().find(|image| image.name == name)
    }

    /// COLMAP text-model export (`Reconstruction::WriteText` ->
    /// `cameras.txt`/`images.txt`/`points3D.txt`), reusing the existing
    /// writer `visloc_io::colmap::write_colmap_reconstruction_for_3dgs_with_cameras`
    /// (already used by `examples/generalized_rig_sfm.rs` for the same
    /// scoring pipeline, `scripts/score_openloris_model.py`) rather than a
    /// new formatter. Only *registered* images are exported (mirrors COLMAP:
    /// `images.txt` only ever lists images with a pose); `IMAGE_ID` in the
    /// output is the 0-based export-order index (the writer's own
    /// convention), not this reconstruction's `image_id` — callers that need
    /// to correlate the two should keep the `image_id -> name` mapping from
    /// [`Self::images`] (names round-trip; `score_openloris_model.py` keys
    /// off `NAME`, not `IMAGE_ID`, per `load_model_centres`).
    pub fn export_colmap_text(
        &self,
        out_dir: impl AsRef<std::path::Path>,
    ) -> Result<usize, ColmapError> {
        let mut registered_image_ids: Vec<ImageT> = self
            .images
            .keys()
            .copied()
            .filter(|&id| self.is_image_registered(id))
            .collect();
        registered_image_ids.sort_unstable();

        let mut cameras = Vec::with_capacity(registered_image_ids.len());
        let mut poses = Vec::with_capacity(registered_image_ids.len());
        let mut features = Vec::with_capacity(registered_image_ids.len());
        let mut names = Vec::with_capacity(registered_image_ids.len());
        let mut output_index_of: BTreeMap<ImageT, usize> = BTreeMap::new();

        for (output_index, &image_id) in registered_image_ids.iter().enumerate() {
            let image = self.image(image_id);
            let frame = self.frame(image.frame_id);
            let rig = self.rig(frame.rig_id());
            let cam_from_world = frame.sensor_from_world(rig, SensorT::camera(image.camera_id));
            cameras.push(self.camera(image.camera_id).clone());
            poses.push(Pose {
                world_to_camera: cam_from_world,
            });
            let keypoints: Vec<Point2<f64>> = image.points2d.iter().map(|p| p.xy).collect();
            let descriptors = vec![Vec::new(); keypoints.len()];
            features.push(FeatureSet {
                keypoints,
                descriptors,
            });
            names.push(image.name.clone());
            output_index_of.insert(image_id, output_index);
        }

        let mut landmarks = Vec::with_capacity(self.points3d.len());
        for point3d in self.points3d.values() {
            let observations: Vec<(usize, usize, Point2<f64>)> = point3d
                .track
                .iter()
                .filter_map(|element| {
                    let &output_index = output_index_of.get(&element.image_id)?;
                    let xy = self
                        .images
                        .get(&element.image_id)?
                        .points2d
                        .get(element.point2d_idx)?
                        .xy;
                    Some((output_index, element.point2d_idx, xy))
                })
                .collect();
            if observations.is_empty() {
                continue;
            }
            landmarks.push((point3d.xyz, observations));
        }

        let summary = write_colmap_reconstruction_for_3dgs_with_cameras(
            out_dir,
            &cameras,
            &poses,
            &features,
            &landmarks,
            |index| names[index].clone(),
        )?;
        Ok(summary.frame_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colmap_incremental::types::DataT;
    use visloc_core::geometry::SE3;

    fn trivial_reconstruction() -> (Reconstruction, ImageT, ImageT) {
        let mut recon = Reconstruction::new();
        let mut rig = Rig::new();
        rig.set_rig_id(0);
        rig.add_ref_sensor(SensorT::camera(1));
        rig.add_sensor(SensorT::camera(2), Some(SE3::identity()));
        recon.add_rig(rig);
        recon.add_camera(Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0));
        recon.add_camera(Camera::pinhole(2, 640, 480, 500.0, 500.0, 320.0, 240.0));

        let mut frame = Frame::new(0, 0);
        frame.add_data_id(DataT::camera(1, 10));
        frame.add_data_id(DataT::camera(2, 11));
        recon.add_frame(frame);

        let mut image_a = Image::new(10, 1, 0, "a.png".to_owned());
        image_a.points2d.push(Point2D::new(Point2::new(1.0, 2.0)));
        image_a.points2d.push(Point2D::new(Point2::new(3.0, 4.0)));
        recon.add_image(image_a);

        let mut image_b = Image::new(11, 2, 0, "b.png".to_owned());
        image_b.points2d.push(Point2D::new(Point2::new(5.0, 6.0)));
        recon.add_image(image_b);

        (recon, 10, 11)
    }

    #[test]
    fn register_deregister_frame_updates_bookkeeping() {
        let (mut recon, image_a, image_b) = trivial_reconstruction();
        assert_eq!(recon.num_reg_frames(), 0);
        assert!(!recon.is_image_registered(image_a));

        let rig = recon.rig(0).clone();
        recon
            .frame_mut(0)
            .set_cam_from_world(&rig, 1, SE3::identity());
        recon.register_frame(0);
        assert_eq!(recon.num_reg_frames(), 1);
        assert_eq!(recon.reg_frame_ids(), &[0]);
        assert!(recon.is_image_registered(image_a));
        assert!(recon.is_image_registered(image_b));
        assert_eq!(recon.num_reg_images(), 2);

        // Idempotent re-registration.
        recon.register_frame(0);
        assert_eq!(recon.num_reg_frames(), 1);

        recon.deregister_frame(0);
        assert_eq!(recon.num_reg_frames(), 0);
        assert!(!recon.is_image_registered(image_a));
    }

    #[test]
    fn add_point3d_and_observation_link_point2d_back_references() {
        let (mut recon, image_a, image_b) = trivial_reconstruction();
        let track = vec![
            TrackElement {
                image_id: image_a,
                point2d_idx: 0,
            },
            TrackElement {
                image_id: image_b,
                point2d_idx: 0,
            },
        ];
        let point3d_id = recon.add_point3d(Point3::new(1.0, 1.0, 5.0), track);
        assert_eq!(recon.num_points3d(), 1);
        assert!(recon.image(image_a).points2d[0].has_point3d());
        assert!(recon.image(image_b).points2d[0].has_point3d());

        recon.add_observation(
            point3d_id,
            TrackElement {
                image_id: image_a,
                point2d_idx: 1,
            },
        );
        assert_eq!(recon.point3d(point3d_id).track.len(), 3);
        assert!(recon.image(image_a).points2d[1].has_point3d());

        recon.delete_point3d(point3d_id);
        assert_eq!(recon.num_points3d(), 0);
        assert!(!recon.image(image_a).points2d[0].has_point3d());
        assert!(!recon.image(image_a).points2d[1].has_point3d());
        assert!(!recon.image(image_b).points2d[0].has_point3d());
    }

    #[test]
    #[should_panic(expected = "does not exist")]
    fn delete_missing_point3d_panics() {
        let (mut recon, _, _) = trivial_reconstruction();
        recon.delete_point3d(999);
    }

    #[test]
    fn export_colmap_text_writes_registered_images_only() {
        let (mut recon, image_a, _image_b) = trivial_reconstruction();
        let rig = recon.rig(0).clone();
        recon
            .frame_mut(0)
            .set_cam_from_world(&rig, 1, SE3::identity());
        recon.register_frame(0);
        let track = vec![TrackElement {
            image_id: image_a,
            point2d_idx: 0,
        }];
        recon.add_point3d(Point3::new(0.0, 0.0, 1.0), track);

        let dir = std::env::temp_dir().join(format!(
            "colmap_incremental_export_test_{}",
            std::process::id()
        ));
        let frame_count = recon.export_colmap_text(&dir).unwrap();
        assert_eq!(frame_count, 2);
        let images_txt = std::fs::read_to_string(dir.join("images.txt")).unwrap();
        assert!(images_txt.contains("a.png"));
        assert!(images_txt.contains("b.png"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
