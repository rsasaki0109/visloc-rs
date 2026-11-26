//! Diagnostics of one localization attempt against the map's own views:
//! how the inliers spread over the image, and which landmarks the camera
//! should have seen but did not match.
//!
//! Measurement only: nothing here changes a localization. It is what the
//! dynamic-obstacle work (`docs/dynamic_obstacle_localization_plan.md`)
//! measures first, to tell an occluded view (people in front of the mapped
//! structure: the expected landmarks go missing in a few image cells) from
//! wrong matches (inliers bunched in one blob) and from a view the map does
//! not cover.
//!
//! "Expected" landmarks are those observed by the map keyframes nearest the
//! camera (at most [`VisibilityGates::max_keyframes`], within
//! [`VisibilityGates::radius_m`] and [`VisibilityGates::max_angle_deg`] of
//! its optical axis) that project into the image in front of the camera.
//! Few keyframes, because a map built from a sequence keeps many landmarks
//! per physical point (one per stretch of keyframes that tracked it); the
//! cell counts are robust to that, the landmark counts are not. Using the keyframes' own observations,
//! rather than every landmark in the frustum, leaves out landmarks behind
//! walls that the map never saw from here.

use std::collections::HashMap;

use nalgebra::{Point2, Point3, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::{Camera, LandmarkId, VisualMap};

/// Which map keyframes count as seeing what the camera should see.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisibilityGates {
    /// A keyframe's camera centre within this distance of the camera (1 m).
    pub radius_m: f64,
    /// ...and its optical axis within this angle of the camera's (30°).
    pub max_angle_deg: f64,
    /// The nearest this many such keyframes (5).
    pub max_keyframes: usize,
}

impl Default for VisibilityGates {
    fn default() -> Self {
        Self {
            radius_m: 1.0,
            max_angle_deg: 30.0,
            max_keyframes: 5,
        }
    }
}

/// The largest grid side: the cells of a grid fit in one `u64` mask.
pub const MAX_GRID: usize = 8;

/// A `grid × grid` division of the image, cells as bits of a `u64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageGrid {
    grid: usize,
    width: f64,
    height: f64,
}

impl ImageGrid {
    /// `grid` is clamped to `1..=MAX_GRID`.
    pub fn new(camera: &Camera, grid: usize) -> Self {
        Self {
            grid: grid.clamp(1, MAX_GRID),
            width: f64::from(camera.width.max(1)),
            height: f64::from(camera.height.max(1)),
        }
    }

    pub fn side(&self) -> usize {
        self.grid
    }

    /// The cell of a pixel, `None` outside the image.
    pub fn cell(&self, pixel: &Point2<f64>) -> Option<usize> {
        if !(pixel.x >= 0.0 && pixel.y >= 0.0 && pixel.x < self.width && pixel.y < self.height) {
            return None;
        }
        let column = ((pixel.x / self.width) * self.grid as f64) as usize;
        let row = ((pixel.y / self.height) * self.grid as f64) as usize;
        Some(row.min(self.grid - 1) * self.grid + column.min(self.grid - 1))
    }

    /// The cells holding at least one of `pixels`, as a bit mask.
    pub fn occupied<'a>(&self, pixels: impl IntoIterator<Item = &'a Point2<f64>>) -> u64 {
        pixels
            .into_iter()
            .filter_map(|pixel| self.cell(pixel))
            .fold(0, |mask, cell| mask | (1 << cell))
    }
}

struct KeyframeView {
    center: Point3<f64>,
    axis: Vector3<f64>,
    /// Indices into `ExpectedVisibility::positions`.
    landmarks: Vec<u32>,
}

/// The map's keyframes and landmarks, indexed for [`ExpectedVisibility::expected`].
pub struct ExpectedVisibility {
    keyframes: Vec<KeyframeView>,
    ids: Vec<LandmarkId>,
    positions: Vec<Point3<f64>>,
}

impl ExpectedVisibility {
    /// Keyframes without a pose are left out. A keyframe's landmarks are its
    /// observations; when a map keeps them only on the landmarks, those are
    /// used instead.
    pub fn new(map: &VisualMap) -> Self {
        let mut ids: Vec<LandmarkId> = map.landmarks.keys().copied().collect();
        ids.sort_unstable();
        let index: HashMap<LandmarkId, u32> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i as u32))
            .collect();
        let positions = ids.iter().map(|id| map.landmarks[id].position).collect();
        let mut by_frame: HashMap<u64, Vec<u32>> = HashMap::new();
        for (id, landmark) in &map.landmarks {
            for observation in &landmark.observations {
                by_frame
                    .entry(observation.frame_id)
                    .or_default()
                    .push(index[id]);
            }
        }
        let mut frame_ids: Vec<u64> = map.keyframes.keys().copied().collect();
        frame_ids.sort_unstable();
        let keyframes = frame_ids
            .into_iter()
            .filter_map(|frame_id| {
                let keyframe = &map.keyframes[&frame_id];
                let pose = keyframe.frame.pose.as_ref()?;
                let mut landmarks: Vec<u32> = keyframe
                    .observations
                    .iter()
                    .filter_map(|observation| index.get(&observation.landmark_id).copied())
                    .collect();
                if landmarks.is_empty() {
                    landmarks = by_frame.get(&frame_id).cloned().unwrap_or_default();
                }
                landmarks.sort_unstable();
                landmarks.dedup();
                Some(KeyframeView {
                    center: pose.camera_center_world(),
                    axis: optical_axis(pose),
                    landmarks,
                })
            })
            .collect();
        Self {
            keyframes,
            ids,
            positions,
        }
    }

    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// The landmarks the camera at `pose` should see, each once, with its
    /// projection, in landmark-id order.
    pub fn expected(
        &self,
        camera: &Camera,
        pose: &Pose,
        gates: VisibilityGates,
    ) -> Vec<(LandmarkId, Point2<f64>)> {
        let center = pose.camera_center_world();
        let axis = optical_axis(pose);
        let min_cos = gates.max_angle_deg.to_radians().cos();
        let radius_squared = gates.radius_m * gates.radius_m;
        let mut nearby: Vec<(f64, usize)> = self
            .keyframes
            .iter()
            .enumerate()
            .filter_map(|(index, keyframe)| {
                let distance_squared = (keyframe.center - center).norm_squared();
                (distance_squared <= radius_squared && keyframe.axis.dot(&axis) >= min_cos)
                    .then_some((distance_squared, index))
            })
            .collect();
        nearby.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        nearby.truncate(gates.max_keyframes);
        let mut seen = vec![false; self.positions.len()];
        for (_, index) in nearby {
            for &landmark in &self.keyframes[index].landmarks {
                seen[landmark as usize] = true;
            }
        }
        let (width, height) = (f64::from(camera.width), f64::from(camera.height));
        seen.iter()
            .enumerate()
            .filter(|(_, &seen)| seen)
            .filter_map(|(landmark, _)| {
                let point_camera = pose.transform_world_point(&self.positions[landmark]);
                if point_camera.z <= 0.0 {
                    return None;
                }
                let pixel = camera.project(&point_camera)?;
                (pixel.x >= 0.0 && pixel.y >= 0.0 && pixel.x < width && pixel.y < height)
                    .then_some((self.ids[landmark], pixel))
            })
            .collect()
    }
}

fn optical_axis(pose: &Pose) -> Vector3<f64> {
    pose.world_to_camera
        .rotation
        .inverse_transform_vector(&Vector3::z())
}

/// The diagnostics of one attempt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ViewDiagnostics {
    /// Grid cells holding at least one inlier.
    pub inlier_cells: u32,
    /// Landmarks the camera should see ([`ExpectedVisibility::expected`]).
    pub expected: usize,
    /// ...of which are inliers.
    pub expected_inliers: usize,
    /// Cells holding at least `min_expected_per_cell` expected landmarks.
    pub expected_cells: u32,
    /// ...of which hold no inlier: where the view is occluded, or the map
    /// out of date, or the matching failed. By cell, not by landmark: an
    /// inlier on a duplicate of an expected landmark (another keyframe's
    /// copy of the same point) still counts.
    pub missing_cells: u32,
}

/// Puts one attempt's inliers (`inlier_pixels`, `inlier_ids`, aligned)
/// against the landmarks expected in view.
pub fn view_diagnostics(
    grid: &ImageGrid,
    inlier_pixels: &[Point2<f64>],
    inlier_ids: &[LandmarkId],
    expected: &[(LandmarkId, Point2<f64>)],
    min_expected_per_cell: usize,
) -> ViewDiagnostics {
    let inliers: std::collections::HashSet<LandmarkId> = inlier_ids.iter().copied().collect();
    let cells = grid.side() * grid.side();
    let mut expected_count = vec![0_usize; cells];
    let mut expected_inliers = 0;
    for (id, pixel) in expected {
        let Some(cell) = grid.cell(pixel) else {
            continue;
        };
        expected_count[cell] += 1;
        if inliers.contains(id) {
            expected_inliers += 1;
        }
    }
    let inlier_mask = grid.occupied(inlier_pixels);
    let expected_mask = expected_count
        .iter()
        .enumerate()
        .filter(|(_, &count)| count > 0 && count >= min_expected_per_cell)
        .fold(0_u64, |mask, (cell, _)| mask | (1 << cell));
    ViewDiagnostics {
        inlier_cells: inlier_mask.count_ones(),
        expected: expected.len(),
        expected_inliers,
        expected_cells: expected_mask.count_ones(),
        missing_cells: (expected_mask & !inlier_mask).count_ones(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};
    use visloc_core::types::{Frame, Keyframe, Landmark, Observation};

    fn camera() -> Camera {
        Camera::pinhole(1, 640, 480, 400.0, 400.0, 320.0, 240.0)
    }

    #[test]
    fn grid_cells_and_occupancy() {
        let grid = ImageGrid::new(&camera(), 4);
        assert_eq!(grid.cell(&Point2::new(0.0, 0.0)), Some(0));
        assert_eq!(grid.cell(&Point2::new(639.9, 479.9)), Some(15));
        assert_eq!(grid.cell(&Point2::new(160.0, 0.0)), Some(1));
        assert_eq!(grid.cell(&Point2::new(0.0, 120.0)), Some(4));
        assert_eq!(grid.cell(&Point2::new(640.0, 10.0)), None);
        assert_eq!(grid.cell(&Point2::new(-1.0, 10.0)), None);
        let pixels = [
            Point2::new(10.0, 10.0),
            Point2::new(20.0, 20.0),
            Point2::new(600.0, 400.0),
        ];
        assert_eq!(grid.occupied(&pixels).count_ones(), 2);
        assert_eq!(ImageGrid::new(&camera(), 99).side(), MAX_GRID);
    }

    /// A wall of landmarks 8 m in front of a keyframe at the origin, and one
    /// landmark seen only by a keyframe looking the other way.
    fn map() -> VisualMap {
        let mut map = VisualMap::new();
        let mut front = Keyframe {
            frame: Frame::new(1, 1),
            observations: Vec::new(),
        };
        front.frame.pose = Some(Pose::identity());
        let mut id = 0;
        for row in 0..6 {
            for column in 0..8 {
                let position =
                    Point3::new(-3.5 + column as f64 * 1.0, -2.5 + row as f64 * 1.0, 8.0);
                map.landmarks.insert(id, Landmark::new(id, position));
                front.observations.push(Observation {
                    frame_id: 1,
                    landmark_id: id,
                    keypoint_index: id as usize,
                    xy: Point2::origin(),
                });
                id += 1;
            }
        }
        map.landmarks
            .insert(1000, Landmark::new(1000, Point3::new(0.0, 0.0, -4.0)));
        let mut back = Keyframe {
            frame: Frame::new(2, 1),
            observations: vec![Observation {
                frame_id: 2,
                landmark_id: 1000,
                keypoint_index: 0,
                xy: Point2::origin(),
            }],
        };
        back.frame.pose = Some(Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.0, std::f64::consts::PI, 0.0),
            Vector3::zeros(),
        ));
        map.keyframes.insert(1, front);
        map.keyframes.insert(2, back);
        map
    }

    #[test]
    fn expected_comes_from_nearby_keyframes_facing_the_same_way() {
        let map = map();
        let visibility = ExpectedVisibility::new(&map);
        assert_eq!(visibility.keyframe_count(), 2);
        let expected =
            visibility.expected(&camera(), &Pose::identity(), VisibilityGates::default());
        // The wall's 48 landmarks project into the image; the one behind is
        // seen only by the keyframe facing away.
        assert_eq!(expected.len(), 48);
        assert!(expected.iter().all(|(id, _)| *id < 1000));
        // Far from both keyframes: nothing is expected.
        let far =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(5.0, 0.0, 0.0));
        assert!(visibility
            .expected(&camera(), &far, VisibilityGates::default())
            .is_empty());
    }

    #[test]
    fn only_the_nearest_keyframes_count() {
        let mut map = map();
        // A second keyframe facing the wall, 0.5 m to the side, sees one
        // more landmark that the nearest keyframe does not.
        map.landmarks
            .insert(2000, Landmark::new(2000, Point3::new(0.2, 0.1, 8.0)));
        let mut side = Keyframe {
            frame: Frame::new(3, 1),
            observations: vec![Observation {
                frame_id: 3,
                landmark_id: 2000,
                keypoint_index: 0,
                xy: Point2::origin(),
            }],
        };
        side.frame.pose = Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(-0.5, 0.0, 0.0),
        ));
        map.keyframes.insert(3, side);
        let visibility = ExpectedVisibility::new(&map);
        let one = VisibilityGates {
            max_keyframes: 1,
            ..VisibilityGates::default()
        };
        let nearest = visibility.expected(&camera(), &Pose::identity(), one);
        assert_eq!(nearest.len(), 48);
        let both = visibility.expected(&camera(), &Pose::identity(), VisibilityGates::default());
        assert_eq!(both.len(), 49);
    }

    #[test]
    fn an_occluded_half_shows_as_missing_cells() {
        let map = map();
        let visibility = ExpectedVisibility::new(&map);
        let camera = camera();
        let expected = visibility.expected(&camera, &Pose::identity(), VisibilityGates::default());
        let grid = ImageGrid::new(&camera, 4);
        // Only the left half of the view matched: someone stands on the right.
        let (pixels, ids): (Vec<_>, Vec<_>) = expected
            .iter()
            .filter(|(_, pixel)| pixel.x < 320.0)
            .map(|(id, pixel)| (*pixel, *id))
            .unzip();
        let diagnostics = view_diagnostics(&grid, &pixels, &ids, &expected, 1);
        assert_eq!(diagnostics.expected, 48);
        assert_eq!(diagnostics.expected_inliers, 24);
        assert!(diagnostics.expected_cells > 0);
        assert_eq!(diagnostics.inlier_cells * 2, diagnostics.expected_cells);
        assert_eq!(diagnostics.missing_cells * 2, diagnostics.expected_cells);
        // Everything matched: nothing missing.
        let (pixels, ids): (Vec<_>, Vec<_>) =
            expected.iter().map(|(id, pixel)| (*pixel, *id)).unzip();
        let full = view_diagnostics(&grid, &pixels, &ids, &expected, 1);
        assert_eq!(full.missing_cells, 0);
        assert_eq!(full.inlier_cells, full.expected_cells);
    }
}
