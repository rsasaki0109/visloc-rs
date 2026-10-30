//! Drop the observations of moving objects before the VIO uses them (Phase 2
//! of `docs/object_metric_slam_plan.md`).
//!
//! An object detector and tracker (for example `visloc-objects`) supplies boxes
//! in camera-0 pixels per frame, each with a stable object id. A box is only a
//! candidate: a parked car is static structure and keeps helping the VIO.
//!
//! The estimator scores each camera-0 observation against the pose the IMU
//! predicts for the frame:
//! - the reprojection error if the track has a landmark;
//! - otherwise the epipolar error, in pixels, against the track's previous
//!   observation.
//!
//! An object is **moving** when the median error of the observations inside
//! its box is both above `min_residual_px` and `background_ratio` times the
//! median of the observations outside every box. Its tracks are then left out
//! of the frame, in both cameras. It stays moving for `hold_ns`, so a brief
//! stop or a frame with too few points does not let it back in.
//!
//! The estimator runs this only once boxes are supplied (`set_object_boxes`).
//! Without boxes, it does exactly what it did before.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use nalgebra::Point2;

use crate::TrackId;

/// An object's box in camera-0 pixels at one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectBox {
    /// Stable across frames for the same object (a tracker's ID).
    pub object_id: u64,
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

impl ObjectBox {
    pub fn contains(&self, pixel: &Point2<f64>) -> bool {
        pixel.x >= self.x_min
            && pixel.x <= self.x_max
            && pixel.y >= self.y_min
            && pixel.y <= self.y_max
    }
}

/// Thresholds of the moving-object test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicObjectParams {
    /// Scored observations a box needs for a decision (5).
    pub min_points: usize,
    /// Scored observations outside the boxes needed for a reference (10).
    pub min_background_points: usize,
    /// A moving box's median error is above this (2 px)...
    pub min_residual_px: f64,
    /// ...and above this multiple of the background median (3).
    pub background_ratio: f64,
    /// An object stays moving this long after its last moving decision (0.5 s).
    pub hold_ns: i64,
}

impl Default for DynamicObjectParams {
    fn default() -> Self {
        Self {
            min_points: 5,
            min_background_points: 10,
            min_residual_px: 2.0,
            background_ratio: 3.0,
            hold_ns: 500_000_000,
        }
    }
}

/// Counts since the boxes were set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DynamicObjectStats {
    /// Frames that had boxes.
    pub frames_with_boxes: u64,
    pub boxes: u64,
    /// Boxes judged moving in their frame (including the hold).
    pub moving_boxes: u64,
    /// Boxes with too few scored observations, or frames without a background
    /// reference; such a box is moving only if its hold is still on.
    pub undecided_boxes: u64,
    /// Observations left out (all cameras).
    pub dropped_observations: u64,
}

/// A camera-0 observation and its error against the predicted pose; `None`
/// when it cannot be scored (a new track, or a projection failure).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ScoredObservation {
    pub track_id: TrackId,
    pub pixel: Point2<f64>,
    pub residual_px: Option<f64>,
}

/// Per-sequence state: the boxes, each object's last moving time, statistics.
#[derive(Debug, Clone, Default)]
pub(crate) struct DynamicObjectFilter {
    pub params: DynamicObjectParams,
    boxes: BTreeMap<i64, Vec<ObjectBox>>,
    last_moving_ns: HashMap<u64, i64>,
    pub stats: DynamicObjectStats,
}

impl DynamicObjectFilter {
    pub fn new(boxes: BTreeMap<i64, Vec<ObjectBox>>, params: DynamicObjectParams) -> Self {
        Self {
            params,
            boxes,
            ..Self::default()
        }
    }

    /// The boxes of the frame at `timestamp_ns`, if any.
    pub fn boxes_at(&self, timestamp_ns: i64) -> Option<&[ObjectBox]> {
        self.boxes
            .get(&timestamp_ns)
            .map(Vec::as_slice)
            .filter(|boxes| !boxes.is_empty())
    }

    /// The tracks to drop at `timestamp_ns`: those observed in camera 0 inside
    /// the box of a moving object.
    pub fn moving_tracks(
        &mut self,
        timestamp_ns: i64,
        boxes: &[ObjectBox],
        observations: &[ScoredObservation],
    ) -> BTreeSet<TrackId> {
        let params = self.params;
        let background = median(
            observations
                .iter()
                .filter(|o| !boxes.iter().any(|b| b.contains(&o.pixel)))
                .filter_map(|o| o.residual_px)
                .collect(),
            params.min_background_points,
        );
        self.stats.frames_with_boxes += 1;
        let mut dropped = BTreeSet::new();
        for b in boxes {
            self.stats.boxes += 1;
            let inside = || observations.iter().filter(|o| b.contains(&o.pixel));
            let own = median(
                inside().filter_map(|o| o.residual_px).collect(),
                params.min_points,
            );
            match (own, background) {
                (Some(own), Some(background)) => {
                    if own > params.min_residual_px && own > params.background_ratio * background {
                        self.last_moving_ns.insert(b.object_id, timestamp_ns);
                    }
                }
                _ => self.stats.undecided_boxes += 1,
            }
            let moving = self
                .last_moving_ns
                .get(&b.object_id)
                .is_some_and(|&last| timestamp_ns - last <= params.hold_ns);
            if moving {
                self.stats.moving_boxes += 1;
                dropped.extend(inside().map(|o| o.track_id));
            }
        }
        dropped
    }
}

/// The median of `values`, if there are at least `min_count` of them.
fn median(mut values: Vec<f64>, min_count: usize) -> Option<f64> {
    if values.is_empty() || values.len() < min_count {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    })
}

/// Read boxes from CSV with the header
/// `timestamp_ns,object_id,class_id,score,x_min,y_min,x_max,y_max`, one box
/// per line (as `yolox_track_demo --boxes-csv` writes). The class and score
/// are informational: choose which classes to write when producing the file.
pub fn read_object_boxes_csv(path: &Path) -> Result<BTreeMap<i64, Vec<ObjectBox>>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    parse_object_boxes_csv(&text).map_err(|error| format!("{}: {error}", path.display()))
}

/// See [`read_object_boxes_csv`].
pub fn parse_object_boxes_csv(text: &str) -> Result<BTreeMap<i64, Vec<ObjectBox>>, String> {
    let mut boxes: BTreeMap<i64, Vec<ObjectBox>> = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("timestamp_ns") {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let bad = || {
            format!(
                "line {}: expected timestamp_ns,object_id,class_id,score,x_min,y_min,x_max,y_max",
                index + 1
            )
        };
        if fields.len() != 8 {
            return Err(bad());
        }
        let timestamp_ns: i64 = fields[0].parse().map_err(|_| bad())?;
        let object_id: u64 = fields[1].parse().map_err(|_| bad())?;
        let corner = |i: usize| fields[i].parse::<f64>().map_err(|_| bad());
        let b = ObjectBox {
            object_id,
            x_min: corner(4)?,
            y_min: corner(5)?,
            x_max: corner(6)?,
            y_max: corner(7)?,
        };
        if !(b.x_max > b.x_min && b.y_max > b.y_min) {
            return Err(bad());
        }
        boxes.entry(timestamp_ns).or_default().push(b);
    }
    Ok(boxes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(x: f64) -> ObjectBox {
        ObjectBox {
            object_id: 7,
            x_min: x,
            y_min: 100.0,
            x_max: x + 100.0,
            y_max: 300.0,
        }
    }

    /// 20 background observations with `background` px error, and `inside`
    /// observations in `object(200)` with `own` px error.
    fn frame(background: f64, inside: usize, own: f64) -> Vec<ScoredObservation> {
        let mut observations: Vec<ScoredObservation> = (0..20)
            .map(|i| ScoredObservation {
                track_id: i,
                pixel: Point2::new(20.0 + 25.0 * i as f64, 400.0),
                residual_px: Some(background),
            })
            .collect();
        for i in 0..inside {
            observations.push(ScoredObservation {
                track_id: 100 + i as u64,
                pixel: Point2::new(210.0 + 10.0 * i as f64, 200.0),
                residual_px: Some(own),
            });
        }
        // A new track inside the box: unscored, but dropped with the object.
        observations.push(ScoredObservation {
            track_id: 999,
            pixel: Point2::new(250.0, 150.0),
            residual_px: None,
        });
        observations
    }

    #[test]
    fn a_moving_object_is_dropped_and_a_parked_one_kept() {
        let mut filter = DynamicObjectFilter::default();
        let dropped = filter.moving_tracks(0, &[object(200.0)], &frame(0.5, 8, 6.0));
        assert_eq!(dropped.len(), 9);
        assert!(dropped.contains(&999) && dropped.contains(&100));
        let mut filter = DynamicObjectFilter::default();
        assert!(filter
            .moving_tracks(0, &[object(200.0)], &frame(0.5, 8, 0.6))
            .is_empty());
        assert_eq!(filter.stats.moving_boxes, 0);
    }

    #[test]
    fn the_background_sets_the_bar() {
        // Inside 3 px: moving against a 0.5 px background, not against 1.5 px
        // (the prediction itself is off by that much).
        let mut filter = DynamicObjectFilter::default();
        assert!(!filter
            .moving_tracks(0, &[object(200.0)], &frame(0.5, 8, 3.0))
            .is_empty());
        let mut filter = DynamicObjectFilter::default();
        assert!(filter
            .moving_tracks(0, &[object(200.0)], &frame(1.5, 8, 3.0))
            .is_empty());
    }

    #[test]
    fn a_moving_object_stays_dropped_for_the_hold_time() {
        let mut filter = DynamicObjectFilter::default();
        assert!(!filter
            .moving_tracks(0, &[object(200.0)], &frame(0.5, 8, 6.0))
            .is_empty());
        // Too few points to decide, 0.3 s later: still moving.
        let dropped = filter.moving_tracks(300_000_000, &[object(200.0)], &frame(0.5, 2, 6.0));
        assert_eq!(dropped.len(), 3);
        assert_eq!(filter.stats.undecided_boxes, 1);
        // Stopped, 0.4 s after the last moving decision: held.
        assert!(!filter
            .moving_tracks(400_000_000, &[object(200.0)], &frame(0.5, 8, 0.5))
            .is_empty());
        // 0.6 s after: let back in.
        assert!(filter
            .moving_tracks(600_000_000, &[object(200.0)], &frame(0.5, 8, 0.5))
            .is_empty());
        assert_eq!(filter.stats.frames_with_boxes, 4);
        assert_eq!(filter.stats.moving_boxes, 3);
    }

    #[test]
    fn without_a_background_reference_nothing_new_is_decided() {
        let mut filter = DynamicObjectFilter::default();
        let few: Vec<ScoredObservation> = frame(0.5, 8, 6.0).into_iter().skip(15).collect();
        assert!(filter.moving_tracks(0, &[object(200.0)], &few).is_empty());
        assert_eq!(filter.stats.undecided_boxes, 1);
    }

    #[test]
    fn parses_the_box_csv() {
        let text = "timestamp_ns,object_id,class_id,score,x_min,y_min,x_max,y_max\n\
                    1000,3,0,0.91,10,20,50,120\n\
                    1000,4,2,0.80,200,20,300,90\n\
                    2000,3,0,0.88,12,20,52,121\n";
        let boxes = parse_object_boxes_csv(text).unwrap();
        assert_eq!(boxes.len(), 2);
        assert_eq!(boxes[&1000].len(), 2);
        assert_eq!(boxes[&2000][0].object_id, 3);
        assert!(boxes[&1000][1].contains(&Point2::new(250.0, 50.0)));
        assert!(parse_object_boxes_csv("1000,3,0,0.9,10,20,5,120\n").is_err());
        assert!(parse_object_boxes_csv("1000,3,0\n").is_err());
        let filter = DynamicObjectFilter::new(boxes, DynamicObjectParams::default());
        assert!(filter.boxes_at(1000).is_some() && filter.boxes_at(1500).is_none());
    }
}
