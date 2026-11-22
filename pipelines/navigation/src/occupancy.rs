//! Building a 2D occupancy grid for navigation from a sparse visual map.
//!
//! A visual map (a SLAM or SfM map: 3D landmarks, the camera poses that
//! observed them) says where there is *something* (a landmark) and, along
//! every ray from a camera to a landmark it observed, where there is
//! *nothing*. [`OccupancyMapper`] accumulates that evidence and projects it
//! to the 2D grid a ground robot plans on.
//!
//! The evidence is kept per cell *and per height slice* inside the obstacle
//! band (`[min_height, max_height]` above the floor), a 2.5D grid: a ray that
//! passes over a low obstacle clears the slices above it, not the slices the
//! obstacle fills. A column is then
//!
//! - **occupied** when any slice holds at least `min_hits` landmarks and its
//!   log-odds is above the occupied threshold;
//! - **free** when no slice is occupied and the lowest slice was seen
//!   through (obstacles stand on the floor: a column seen free only higher up,
//!   such as the inside of a sofa seen over its back, may still hold one), or
//!   the robot drove through it ([`OccupancyMapper::insert_free_disc`]);
//! - **unknown** otherwise.
//!
//! The map frame must be gravity-aligned with `z` up, as a visual-inertial
//! SLAM map is (the stereo-inertial SLAM's `--save-map` maps are). Landmarks
//! below or above the band, the floor and the ceiling, are not obstacles;
//! rays to them still clear the slices they pass through inside the band.

use std::collections::HashMap;

use nalgebra::Point3;
use visloc_core::types::{FrameId, LandmarkId, VisualMap};

use crate::grid::{GridGeometry, OccupancyGrid, FREE, OCCUPIED, UNKNOWN};

#[derive(Debug, Clone, PartialEq)]
pub struct OccupancyConfig {
    /// Cell edge length in metres.
    pub resolution: f64,
    /// Map-frame `z` of the floor.
    pub floor_z: f64,
    /// Lowest height above the floor that is an obstacle (below: floor).
    pub min_height: f64,
    /// Highest height above the floor that is an obstacle (above: the robot
    /// passes under it).
    pub max_height: f64,
    /// Height of one slice of the band.
    pub slice_height: f64,
    /// Rays and landmarks farther than this from the camera are not used.
    pub max_range: f64,
    /// Landmarks a slice needs to be occupied, so a lone wrong landmark
    /// does not block a corridor.
    pub min_hits: u32,
    /// Log-odds added by a landmark in a slice.
    pub hit_log_odds: f32,
    /// Log-odds added (negative) by a ray passing through a slice.
    pub miss_log_odds: f32,
    /// A slice above this log-odds (with `min_hits`) is occupied.
    pub occupied_log_odds: f32,
    /// Clamp of the log-odds, so later evidence can still change a slice.
    pub clamp_log_odds: f32,
    /// Landmarks observed from fewer cameras are not used.
    pub min_observations: usize,
    /// Cells a ray leaves untouched before its landmark, so a landmark's own
    /// cell is not cleared by rays that graze it.
    pub ray_end_margin: f64,
}

impl Default for OccupancyConfig {
    fn default() -> Self {
        Self {
            resolution: 0.05,
            floor_z: 0.0,
            min_height: 0.10,
            max_height: 1.50,
            slice_height: 0.25,
            max_range: 8.0,
            min_hits: 2,
            hit_log_odds: 0.85,
            miss_log_odds: -0.10,
            occupied_log_odds: 0.5,
            clamp_log_odds: 3.5,
            min_observations: 2,
            ray_end_margin: 0.15,
        }
    }
}

impl OccupancyConfig {
    fn slices(&self) -> usize {
        (((self.max_height - self.min_height) / self.slice_height).ceil() as usize).max(1)
    }

    /// The band slice holding map-frame height `z`, if it is in the band.
    fn slice_of(&self, z: f64) -> Option<usize> {
        let h = z - self.floor_z;
        if !(self.min_height..=self.max_height).contains(&h) {
            return None;
        }
        Some((((h - self.min_height) / self.slice_height) as usize).min(self.slices() - 1))
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Slice {
    log_odds: f32,
    hits: u32,
    seen: bool,
}

/// Counts of what [`OccupancyMapper`] was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OccupancyStats {
    pub rays: usize,
    pub landmarks_in_band: usize,
    pub landmarks_out_of_band: usize,
    pub rays_out_of_range: usize,
    pub free_discs: usize,
}

/// Accumulates landmarks, camera-to-landmark rays and driven footprints into
/// a 2.5D grid; [`Self::to_grid`] projects it to an [`OccupancyGrid`].
///
/// The grid grows to fit what is inserted: cells live in a hash map of
/// 32 x 32 tiles, so a map of a few kilometres at 5 cm stays cheap where
/// nothing was seen.
#[derive(Debug, Clone)]
pub struct OccupancyMapper {
    config: OccupancyConfig,
    slices: usize,
    tiles: HashMap<(i64, i64), Vec<Slice>>,
    driven: HashMap<(i64, i64), Vec<bool>>,
    stats: OccupancyStats,
}

const TILE: i64 = 32;

impl OccupancyMapper {
    pub fn new(config: OccupancyConfig) -> Self {
        assert!(config.resolution > 0.0, "resolution must be positive");
        assert!(config.slice_height > 0.0, "slice_height must be positive");
        assert!(
            config.max_height > config.min_height,
            "max_height must be above min_height"
        );
        let slices = config.slices();
        Self {
            config,
            slices,
            tiles: HashMap::new(),
            driven: HashMap::new(),
            stats: OccupancyStats::default(),
        }
    }

    pub const fn config(&self) -> &OccupancyConfig {
        &self.config
    }

    pub const fn stats(&self) -> OccupancyStats {
        self.stats
    }

    fn cell(&self, x: f64, y: f64) -> (i64, i64) {
        (
            (x / self.config.resolution).floor() as i64,
            (y / self.config.resolution).floor() as i64,
        )
    }

    fn slice_mut(&mut self, cell: (i64, i64), slice: usize) -> &mut Slice {
        let slices = self.slices;
        let tile = (cell.0.div_euclid(TILE), cell.1.div_euclid(TILE));
        let local = (cell.1.rem_euclid(TILE) * TILE + cell.0.rem_euclid(TILE)) as usize;
        let cells = self
            .tiles
            .entry(tile)
            .or_insert_with(|| vec![Slice::default(); (TILE * TILE) as usize * slices]);
        &mut cells[local * slices + slice]
    }

    /// Adds a landmark at a map-frame point (hit only, no ray).
    pub fn insert_landmark(&mut self, point: &Point3<f64>) {
        let Some(slice) = self.config.slice_of(point.z) else {
            self.stats.landmarks_out_of_band += 1;
            return;
        };
        self.stats.landmarks_in_band += 1;
        let cell = self.cell(point.x, point.y);
        let (hit, clamp) = (self.config.hit_log_odds, self.config.clamp_log_odds);
        let s = self.slice_mut(cell, slice);
        s.log_odds = (s.log_odds + hit).clamp(-clamp, clamp);
        s.hits += 1;
        s.seen = true;
    }

    /// Adds the ray from a camera centre to a landmark it observed: the
    /// in-band slices the ray passes through are seen free, and the landmark
    /// is a hit. A ray longer than `max_range` is ignored entirely.
    pub fn insert_ray(&mut self, camera: &Point3<f64>, landmark: &Point3<f64>) {
        let delta = landmark - camera;
        let length = delta.norm();
        if !length.is_finite() || length > self.config.max_range {
            self.stats.rays_out_of_range += 1;
            return;
        }
        self.stats.rays += 1;
        let horizontal = delta.xy().norm();
        let end_cell = self.cell(landmark.x, landmark.y);
        let step = self.config.resolution * 0.5;
        let free_length = horizontal - self.config.ray_end_margin;
        if free_length > 0.0 {
            let samples = (free_length / step).ceil() as usize;
            let mut previous = None;
            for i in 0..=samples {
                let t = (i as f64 * step).min(free_length) / horizontal;
                let p = camera + delta * t;
                let cell = self.cell(p.x, p.y);
                if cell == end_cell {
                    break;
                }
                let Some(slice) = self.config.slice_of(p.z) else {
                    continue;
                };
                if previous == Some((cell, slice)) {
                    continue;
                }
                previous = Some((cell, slice));
                let (miss, clamp) = (self.config.miss_log_odds, self.config.clamp_log_odds);
                let s = self.slice_mut(cell, slice);
                s.log_odds = (s.log_odds + miss).clamp(-clamp, clamp);
                s.seen = true;
            }
        }
        self.insert_landmark(landmark);
    }

    /// Marks every cell within `radius` of `(x, y)` as driven through: free
    /// whatever the landmarks say, since the robot's body was there. Pass
    /// the robot's ground position along the mapping trajectory.
    pub fn insert_free_disc(&mut self, x: f64, y: f64, radius: f64) {
        self.stats.free_discs += 1;
        let r = radius.max(0.0);
        let (x0, y0) = self.cell(x - r, y - r);
        let (x1, y1) = self.cell(x + r, y + r);
        let res = self.config.resolution;
        for cy in y0..=y1 {
            for cx in x0..=x1 {
                let (px, py) = ((cx as f64 + 0.5) * res, (cy as f64 + 0.5) * res);
                if (px - x).hypot(py - y) > r + 0.5 * res {
                    continue;
                }
                let tile = (cx.div_euclid(TILE), cy.div_euclid(TILE));
                let local = (cy.rem_euclid(TILE) * TILE + cx.rem_euclid(TILE)) as usize;
                self.driven
                    .entry(tile)
                    .or_insert_with(|| vec![false; (TILE * TILE) as usize])[local] = true;
            }
        }
    }

    /// Adds every landmark of a visual map with the rays from the cameras
    /// that observed it: the posed keyframes whose observations name it, or
    /// that its own observations name (a COLMAP text model read back keeps
    /// only the former). Landmarks observed by fewer than `min_observations`
    /// posed keyframes are skipped. Returns how many landmarks were used.
    pub fn insert_visual_map(&mut self, map: &VisualMap) -> usize {
        let mut observers: HashMap<LandmarkId, Vec<FrameId>> = HashMap::new();
        for keyframe in map.keyframes.values() {
            for observation in &keyframe.observations {
                observers
                    .entry(observation.landmark_id)
                    .or_default()
                    .push(keyframe.frame.id);
            }
        }
        for landmark in map.landmarks.values() {
            for observation in &landmark.observations {
                observers
                    .entry(landmark.id)
                    .or_default()
                    .push(observation.frame_id);
            }
        }
        let mut used = 0;
        let mut landmark_ids = map.landmarks.keys().copied().collect::<Vec<_>>();
        landmark_ids.sort_unstable();
        for id in landmark_ids {
            let Some(frames) = observers.get_mut(&id) else {
                continue;
            };
            frames.sort_unstable();
            frames.dedup();
            let centers = frames
                .iter()
                .filter_map(|frame_id| map.keyframes.get(frame_id))
                .filter_map(|keyframe| keyframe.frame.pose.as_ref())
                .map(|pose| pose.camera_center_world())
                .collect::<Vec<_>>();
            if centers.len() < self.config.min_observations {
                continue;
            }
            used += 1;
            let position = map.landmarks[&id].position;
            for center in &centers {
                // Each ray also counts the landmark once; a landmark seen
                // from n cameras is n hits, which is the evidence it has.
                self.insert_ray(center, &position);
            }
        }
        used
    }

    /// The occupancy of one column (cell), from its slices and whether it
    /// was driven through.
    fn column_value(&self, slices: Option<&[Slice]>, driven: bool) -> i8 {
        if driven {
            return FREE;
        }
        let Some(slices) = slices else {
            return UNKNOWN;
        };
        let occupied = slices
            .iter()
            .any(|s| s.hits >= self.config.min_hits && s.log_odds > self.config.occupied_log_odds);
        if occupied {
            OCCUPIED
        } else if slices[0].seen && slices[0].log_odds < 0.0 {
            // Free only where the bottom of the band was seen through:
            // obstacles stand on the floor, so a column seen free only above
            // something (the inside of a sofa, seen over its back) may still
            // hold it.
            FREE
        } else {
            UNKNOWN
        }
    }

    /// Projects the evidence to a 2D grid covering every cell touched,
    /// grown by `margin` metres of unknown on each side. `None` when nothing
    /// was inserted.
    pub fn to_grid(&self, margin: f64) -> Option<OccupancyGrid> {
        let keys = self.tiles.keys().chain(self.driven.keys());
        let (mut tx0, mut ty0, mut tx1, mut ty1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
        for &(tx, ty) in keys {
            tx0 = tx0.min(tx);
            ty0 = ty0.min(ty);
            tx1 = tx1.max(tx);
            ty1 = ty1.max(ty);
        }
        if tx0 == i64::MAX {
            return None;
        }
        let res = self.config.resolution;
        let pad = (margin.max(0.0) / res).ceil() as i64;
        let (cx0, cy0) = (tx0 * TILE - pad, ty0 * TILE - pad);
        let (cx1, cy1) = ((tx1 + 1) * TILE + pad, (ty1 + 1) * TILE + pad);
        let geometry = GridGeometry {
            resolution: res,
            origin_x: cx0 as f64 * res,
            origin_y: cy0 as f64 * res,
            width: (cx1 - cx0) as usize,
            height: (cy1 - cy0) as usize,
        };
        let mut grid = OccupancyGrid::unknown(geometry);
        for iy in 0..geometry.height {
            for ix in 0..geometry.width {
                let (cx, cy) = (cx0 + ix as i64, cy0 + iy as i64);
                let tile = (cx.div_euclid(TILE), cy.div_euclid(TILE));
                let local = (cy.rem_euclid(TILE) * TILE + cx.rem_euclid(TILE)) as usize;
                let slices = self
                    .tiles
                    .get(&tile)
                    .map(|cells| &cells[local * self.slices..(local + 1) * self.slices]);
                let driven = self.driven.get(&tile).is_some_and(|cells| cells[local]);
                grid.set(ix, iy, self.column_value(slices, driven));
            }
        }
        Some(grid.cropped_to_known((margin.max(0.0) / res).ceil() as usize))
    }
}

/// The floor height of a gravity-aligned map whose cameras rode
/// `camera_height` above the floor: the median camera-centre `z` minus
/// `camera_height`. `None` for a map without posed keyframes.
pub fn floor_z_from_cameras(map: &VisualMap, camera_height: f64) -> Option<f64> {
    let mut zs = map
        .keyframes
        .values()
        .filter_map(|keyframe| keyframe.frame.pose.as_ref())
        .map(|pose| pose.camera_center_world().z)
        .filter(|z| z.is_finite())
        .collect::<Vec<_>>();
    if zs.is_empty() {
        return None;
    }
    zs.sort_by(f64::total_cmp);
    Some(zs[zs.len() / 2] - camera_height)
}

/// Builds the grid of a visual map in one call: every landmark with its
/// rays, and a disc of `free_radius` (0 for none) around the ground point
/// of every posed keyframe's camera.
pub fn occupancy_from_visual_map(
    map: &VisualMap,
    config: OccupancyConfig,
    free_radius: f64,
    margin: f64,
) -> Option<(OccupancyGrid, OccupancyStats)> {
    let mut mapper = OccupancyMapper::new(config);
    mapper.insert_visual_map(map);
    if free_radius > 0.0 {
        let mut ids = map.keyframes.keys().copied().collect::<Vec<_>>();
        ids.sort_unstable();
        for id in ids {
            if let Some(pose) = map.keyframes[&id].frame.pose.as_ref() {
                let c = pose.camera_center_world();
                mapper.insert_free_disc(c.x, c.y, free_radius);
            }
        }
    }
    mapper.to_grid(margin).map(|grid| (grid, mapper.stats()))
}
