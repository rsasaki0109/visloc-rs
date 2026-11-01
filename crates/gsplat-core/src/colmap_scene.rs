//! Build a 3DGS [`Scene`] seed and camera list from a COLMAP model.
//!
//! The SfM sparse cloud provides the *on-surface* initial points a 3DGS
//! optimizer needs (random volumetric init produces fog; see
//! `scripts/euroc_colmap_splat.sh`). Each landmark becomes a degree-0 Gaussian
//! with a small isotropic scale and neutral grey colour; SH and view-dependent
//! appearance are learned later. Cameras are converted from the COLMAP
//! world-to-camera convention into [`CameraView`]s for rendering.

use nalgebra::{Matrix3, Quaternion, Vector3};
use visloc_core::types::CameraModel;
use visloc_io::colmap::{read_colmap_binary_model, read_colmap_text_model, ColmapError};

use crate::camera::{CameraView, PinholeCamera};
use crate::gaussian::{Gaussian, Scene};
use crate::lens::{Lens, Projection};

/// Errors from loading a COLMAP model into a 3DGS scene.
#[derive(Debug, thiserror::Error)]
pub enum ColmapSceneError {
    #[error("colmap io error: {0}")]
    Colmap(#[from] ColmapError),
    #[error("camera {camera_id} is not a pinhole model (got {model:?}); only OPENCV/PINHOLE/SIMPLE_PINHOLE are supported")]
    UnsupportedCamera {
        camera_id: visloc_core::types::CameraId,
        model: CameraModel,
    },
    #[error("colmap model has no landmarks to seed gaussians")]
    NoLandmarks,
    #[error("colmap model has no registered images")]
    NoImages,
}

/// A COLMAP model loaded for 3DGS: the initial scene and its camera views.
#[derive(Debug, Clone)]
pub struct ColmapScene {
    pub scene: Scene,
    pub views: Vec<CameraView>,
    /// Placeholder names `frame_{id:06}.png` in the same order as `views`
    /// (the visloc map does not keep COLMAP image names; use `image_ids` with
    /// the model's `images.txt` to recover them).
    pub image_names: Vec<String>,
    /// COLMAP `IMAGE_ID` of each view, in the same order as `views`.
    pub image_ids: Vec<u64>,
}

/// Initial isotropic log-scale for the seed gaussians.
///
/// The value is a small world-space radius; a 3DGS optimizer refines it. It is
/// deliberately not derived from kNN distance here (that heuristic needs the
/// full point set and is better done by the trainer).
pub const DEFAULT_SEED_LOG_SCALE: f32 = -3.0;

/// Read a COLMAP model directory: text if `cameras.txt` is present, else
/// binary.
pub fn read_visual_map(
    model_dir: impl AsRef<std::path::Path>,
) -> Result<visloc_core::types::VisualMap, ColmapError> {
    let dir = model_dir.as_ref();
    if dir.join("cameras.txt").exists() {
        read_colmap_text_model(dir)
    } else {
        read_colmap_binary_model(dir)
    }
}

/// Load a COLMAP model from a directory (text if present, else binary) into a
/// 3DGS scene seed.
pub fn load_colmap_scene(
    model_dir: impl AsRef<std::path::Path>,
    seed_log_scale: f32,
) -> Result<ColmapScene, ColmapSceneError> {
    let map = read_visual_map(model_dir)?;
    scene_from_visual_map(&map, seed_log_scale)
}

/// Convert an already-loaded `VisualMap` into a [`ColmapScene`].
pub fn scene_from_visual_map(
    map: &visloc_core::types::VisualMap,
    seed_log_scale: f32,
) -> Result<ColmapScene, ColmapSceneError> {
    let scene = seed_scene_from_landmarks(map, seed_log_scale)?;
    if map.keyframes.is_empty() {
        return Err(ColmapSceneError::NoImages);
    }

    // Camera views from the registered keyframes.
    let mut keyframe_ids: Vec<_> = map.keyframes.keys().copied().collect();
    keyframe_ids.sort_unstable();
    let mut views = Vec::with_capacity(keyframe_ids.len());
    let mut image_names = Vec::with_capacity(keyframe_ids.len());
    let mut image_ids = Vec::with_capacity(keyframe_ids.len());
    for id in keyframe_ids {
        let kf = &map.keyframes[&id];
        let Some(pose) = kf.frame.pose.as_ref() else {
            continue;
        };
        let camera = match map.cameras.get(&kf.frame.camera_id) {
            Some(c) => c,
            None => continue,
        };
        let view = camera_view_from(camera, pose)?;
        views.push(view);
        image_names.push(format!("frame_{id:06}.png"));
        image_ids.push(id);
    }
    if views.is_empty() {
        return Err(ColmapSceneError::NoImages);
    }

    Ok(ColmapScene {
        scene,
        views,
        image_names,
        image_ids,
    })
}

/// The seed [`Scene`] of a map: one isotropic degree-0 gaussian per landmark
/// (neutral grey), in landmark-id order.
pub fn seed_scene_from_landmarks(
    map: &visloc_core::types::VisualMap,
    seed_log_scale: f32,
) -> Result<Scene, ColmapSceneError> {
    if map.landmarks.is_empty() {
        return Err(ColmapSceneError::NoLandmarks);
    }

    // Seed gaussians from the sparse landmarks (degree 0, neutral grey).
    let mut landmark_ids: Vec<_> = map.landmarks.keys().copied().collect();
    landmark_ids.sort_unstable();
    let mut gaussians = Vec::with_capacity(landmark_ids.len());
    for id in landmark_ids {
        let lm = &map.landmarks[&id];
        gaussians.push(Gaussian {
            mean: Vector3::new(
                lm.position.x as f32,
                lm.position.y as f32,
                lm.position.z as f32,
            ),
            scale_log: Vector3::new(seed_log_scale, seed_log_scale, seed_log_scale),
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            opacity_logit: 0.0,
            sh_dc: [0.0; 3],
            sh_rest: Vec::new(),
            sh_degree: 0,
        });
    }

    Ok(Scene::new(gaussians, 0))
}

/// Build a rendering [`CameraView`] from a visloc camera and pose.
///
/// `pose` is the visloc convention (`world_to_camera` SE3); its rotation is
/// converted to a `Matrix3<f32>`, the intrinsics are read from the COLMAP
/// parameter layout for the camera model, and its distortion becomes the
/// view's [`Lens`] ([`lens_of`]).
pub fn camera_view_from(
    camera: &visloc_core::types::Camera,
    pose: &visloc_core::geometry::Pose,
) -> Result<CameraView, ColmapSceneError> {
    let (fx, fy, cx, cy) = pinhole_intrinsics(camera)?;
    let lens = lens_of(camera)?;
    let r = pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let t = pose.world_to_camera.translation;
    let rotation = Matrix3::new(
        r[(0, 0)] as f32,
        r[(0, 1)] as f32,
        r[(0, 2)] as f32,
        r[(1, 0)] as f32,
        r[(1, 1)] as f32,
        r[(1, 2)] as f32,
        r[(2, 0)] as f32,
        r[(2, 1)] as f32,
        r[(2, 2)] as f32,
    );
    let translation = Vector3::new(t.x as f32, t.y as f32, t.z as f32);
    Ok(CameraView::new(
        rotation,
        translation,
        PinholeCamera::new(camera.width, camera.height, fx, fy, cx, cy).with_lens(lens),
    ))
}

/// The [`Lens`] of a COLMAP camera model, in gsplat-core's coefficient
/// order `[k1, k2, k3, k4, p1, p2, sx1, sy1]`:
///
/// - pinhole projection: `SIMPLE_PINHOLE` and `PINHOLE` (with visloc's
///   optional trailing `k1 k2`, [`visloc_core::types::Camera::pinhole_radial`]),
///   `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, and `FULL_OPENCV` without its
///   rational denominator (`k4 = k5 = k6 = 0`);
/// - equidistant projection: `SIMPLE_RADIAL_FISHEYE`, `RADIAL_FISHEYE`,
///   `OPENCV_FISHEYE`, `THIN_PRISM_FISHEYE`.
///
/// Other models (a rational `FULL_OPENCV`, `FOV`, Double Sphere,
/// equirectangular) are unsupported.
pub fn lens_of(camera: &visloc_core::types::Camera) -> Result<Lens, ColmapSceneError> {
    let unsupported = || ColmapSceneError::UnsupportedCamera {
        camera_id: camera.id,
        model: camera.model.clone(),
    };
    let at = |i: usize| camera.params.get(i).copied().unwrap_or(0.0) as f32;
    let (projection, coeffs) = match &camera.model {
        CameraModel::SimplePinhole => (Projection::Pinhole, [0.0; 8]),
        CameraModel::Pinhole => (
            Projection::Pinhole,
            [at(4), at(5), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        CameraModel::SimpleRadial => (
            Projection::Pinhole,
            [at(3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        CameraModel::Radial => (
            Projection::Pinhole,
            [at(3), at(4), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        CameraModel::OpenCv => (
            Projection::Pinhole,
            [at(4), at(5), 0.0, 0.0, at(6), at(7), 0.0, 0.0],
        ),
        CameraModel::FullOpenCv => {
            if [at(9), at(10), at(11)] != [0.0; 3] {
                return Err(unsupported());
            }
            (
                Projection::Pinhole,
                [at(4), at(5), at(8), 0.0, at(6), at(7), 0.0, 0.0],
            )
        }
        CameraModel::SimpleRadialFisheye => (
            Projection::Equidistant,
            [at(3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        CameraModel::RadialFisheye => (
            Projection::Equidistant,
            [at(3), at(4), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        CameraModel::OpenCvFisheye => (
            Projection::Equidistant,
            [at(4), at(5), at(6), at(7), 0.0, 0.0, 0.0, 0.0],
        ),
        CameraModel::ThinPrismFisheye => (
            Projection::Equidistant,
            [at(4), at(5), at(8), at(9), at(6), at(7), at(10), at(11)],
        ),
        _ => return Err(unsupported()),
    };
    Ok(Lens::new(projection, coeffs))
}

/// Extract `(fx, fy, cx, cy)` from a COLMAP camera parameter list (the
/// distortion is [`lens_of`]'s).
///
/// - `SIMPLE_PINHOLE`, `SIMPLE_RADIAL`, `RADIAL` and their fisheye
///   counterparts: `f, cx, cy, ...` (fy = fx)
/// - `PINHOLE`, `OPENCV`, `FULL_OPENCV`, `OPENCV_FISHEYE`,
///   `THIN_PRISM_FISHEYE`: `fx, fy, cx, cy, ...`
pub fn pinhole_intrinsics(
    camera: &visloc_core::types::Camera,
) -> Result<(f32, f32, f32, f32), ColmapSceneError> {
    let p = &camera.params;
    let need = |n: usize| -> Result<(), ColmapSceneError> {
        if p.len() < n {
            Err(ColmapSceneError::UnsupportedCamera {
                camera_id: camera.id,
                model: camera.model.clone(),
            })
        } else {
            Ok(())
        }
    };
    match &camera.model {
        CameraModel::SimplePinhole => {
            need(3)?;
            Ok((p[0] as f32, p[0] as f32, p[1] as f32, p[2] as f32))
        }
        CameraModel::SimpleRadial
        | CameraModel::Radial
        | CameraModel::SimpleRadialFisheye
        | CameraModel::RadialFisheye => {
            // [f, cx, cy, k1(, k2)]: one focal length.
            need(4)?;
            Ok((p[0] as f32, p[0] as f32, p[1] as f32, p[2] as f32))
        }
        CameraModel::Pinhole
        | CameraModel::OpenCv
        | CameraModel::FullOpenCv
        | CameraModel::OpenCvFisheye
        | CameraModel::ThinPrismFisheye => {
            need(4)?;
            Ok((p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32))
        }
        other => Err(ColmapSceneError::UnsupportedCamera {
            camera_id: camera.id,
            model: other.clone(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Point3};
    use visloc_core::geometry::Pose;
    use visloc_core::types::{Camera, Landmark, Observation, VisualMap};

    fn pinhole_camera() -> Camera {
        Camera {
            id: 1,
            model: CameraModel::Pinhole,
            width: 640,
            height: 480,
            params: vec![500.0, 500.0, 320.0, 240.0],
        }
    }

    fn identity_pose() -> Pose {
        Pose {
            world_to_camera: visloc_core::geometry::SE3::identity(),
        }
    }

    #[test]
    fn intrinsics_pinhole() {
        let cam = pinhole_camera();
        let (fx, fy, cx, cy) = pinhole_intrinsics(&cam).unwrap();
        assert!((fx - 500.0).abs() < 1e-4);
        assert!((fy - 500.0).abs() < 1e-4);
        assert!((cx - 320.0).abs() < 1e-4);
        assert!((cy - 240.0).abs() < 1e-4);
    }

    #[test]
    fn intrinsics_simple_pinhole_shares_focal() {
        let cam = Camera {
            id: 1,
            model: CameraModel::SimplePinhole,
            width: 640,
            height: 480,
            params: vec![400.0, 320.0, 240.0],
        };
        let (fx, fy, _, _) = pinhole_intrinsics(&cam).unwrap();
        assert!((fx - fy).abs() < 1e-6);
    }

    #[test]
    fn intrinsics_rejects_short_params() {
        let cam = Camera {
            id: 1,
            model: CameraModel::Pinhole,
            width: 640,
            height: 480,
            params: vec![500.0],
        };
        assert!(pinhole_intrinsics(&cam).is_err());
    }

    #[test]
    fn builds_scene_and_view_from_map() {
        let mut map = VisualMap::new();
        map.cameras.insert(1, pinhole_camera());
        map.landmarks.insert(
            10,
            Landmark {
                id: 10,
                position: Point3::new(0.0, 0.0, 5.0),
                descriptor: None,
                observations: vec![],
            },
        );
        let kf = visloc_core::types::Keyframe {
            frame: visloc_core::types::Frame {
                id: 0,
                camera_id: 1,
                keypoints: vec![Point2::new(320.0, 240.0)],
                descriptors: vec![],
                pose: Some(identity_pose()),
            },
            observations: vec![Observation {
                frame_id: 0,
                landmark_id: 10,
                keypoint_index: 0,
                xy: Point2::new(320.0, 240.0),
            }],
        };
        map.keyframes.insert(0, kf);
        let loaded = scene_from_visual_map(&map, DEFAULT_SEED_LOG_SCALE).unwrap();
        assert_eq!(loaded.scene.len(), 1);
        assert_eq!(loaded.views.len(), 1);
        assert_eq!(loaded.scene.gaussians[0].mean, Vector3::new(0.0, 0.0, 5.0));
    }

    #[test]
    fn empty_map_errors() {
        let map = VisualMap::new();
        assert!(matches!(
            scene_from_visual_map(&map, DEFAULT_SEED_LOG_SCALE),
            Err(ColmapSceneError::NoLandmarks)
        ));
    }

    fn camera(model: CameraModel, params: Vec<f64>) -> Camera {
        Camera {
            id: 1,
            model,
            width: 752,
            height: 480,
            params,
        }
    }

    #[test]
    fn lenses_follow_the_colmap_models() {
        let opencv = lens_of(&camera(
            CameraModel::OpenCv,
            vec![500.0, 500.0, 376.0, 240.0, -0.2, 0.05, 0.001, -0.002],
        ))
        .unwrap();
        assert_eq!(opencv.projection, Projection::Pinhole);
        assert_eq!(
            opencv.coeffs,
            [-0.2, 0.05, 0.0, 0.0, 0.001, -0.002, 0.0, 0.0]
        );
        // COLMAP's [.. k1 k2 p1 p2 k3 k4 sx1 sy1] to [k1 k2 k3 k4 p1 p2 sx1 sy1].
        let thin_prism = lens_of(&camera(
            CameraModel::ThinPrismFisheye,
            vec![
                540.0, 540.0, 413.0, 245.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            ],
        ))
        .unwrap();
        assert_eq!(thin_prism.projection, Projection::Equidistant);
        assert_eq!(thin_prism.coeffs, [1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
        let fisheye = lens_of(&camera(
            CameraModel::RadialFisheye,
            vec![300.0, 376.0, 240.0, 0.1, -0.01],
        ))
        .unwrap();
        assert_eq!(fisheye.coeffs[..2], [0.1, -0.01]);
        assert!(lens_of(&camera(
            CameraModel::SimplePinhole,
            vec![500.0, 376.0, 240.0]
        ))
        .unwrap()
        .is_pinhole());
        // A rational FULL_OPENCV, FOV and Double Sphere cannot be drawn.
        let rational = vec![
            500.0, 500.0, 376.0, 240.0, -0.2, 0.05, 0.0, 0.0, 0.01, 0.1, 0.0, 0.0,
        ];
        assert!(lens_of(&camera(CameraModel::FullOpenCv, rational)).is_err());
        assert!(lens_of(&camera(
            CameraModel::Fov,
            vec![500.0, 500.0, 376.0, 240.0, 0.9]
        ))
        .is_err());
    }

    #[test]
    fn fisheye_views_get_their_lens_and_field_of_view() {
        let cam = camera(
            CameraModel::OpenCvFisheye,
            vec![200.0, 200.0, 376.0, 240.0, 0.02, -0.01, 0.0, 0.0],
        );
        let view = camera_view_from(&cam, &identity_pose()).unwrap();
        assert_eq!(view.camera.lens.projection, Projection::Equidistant);
        assert_eq!((view.camera.fx, view.camera.cx), (200.0, 376.0));
        // The corners are ~2.2 rad off the axis at f = 200 px: max_theta is
        // fitted to the image, well past 90°.
        assert!(
            view.camera.lens.max_theta > 2.0,
            "{}",
            view.camera.lens.max_theta
        );
        assert!(view.pinhole_geometry().is_none());
    }
}
