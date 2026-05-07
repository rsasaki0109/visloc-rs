use nalgebra::{Point2, Point3};
use visloc_core::types::{Camera, Frame, Keyframe, Landmark, Observation, VisualMap};
use visloc_mapping::{LocalMapWindow, LocalMapWindowConfig};

fn observation(frame_id: u64, landmark_id: u64, keypoint_index: usize) -> Observation {
    Observation {
        frame_id,
        landmark_id,
        keypoint_index,
        xy: Point2::new(100.0 + keypoint_index as f64, 120.0),
    }
}

fn keyframe(frame_id: u64, landmark_ids: &[u64]) -> Keyframe {
    let mut frame = Frame::new(frame_id, 1);
    for index in 0..landmark_ids.len() {
        frame
            .keypoints
            .push(Point2::new(100.0 + index as f64, 120.0));
    }
    Keyframe {
        frame,
        observations: landmark_ids
            .iter()
            .enumerate()
            .map(|(index, landmark_id)| observation(frame_id, *landmark_id, index))
            .collect(),
    }
}

fn map_with_keyframes() -> VisualMap {
    let mut map = VisualMap::new();
    map.cameras
        .insert(1, Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0));
    for landmark_id in 100..=104 {
        map.landmarks.insert(
            landmark_id,
            Landmark::new(landmark_id, Point3::new(landmark_id as f64, 0.0, 5.0)),
        );
    }
    map.keyframes.insert(1, keyframe(1, &[100, 101]));
    map.keyframes.insert(2, keyframe(2, &[101, 102]));
    map.keyframes.insert(3, keyframe(3, &[102, 103]));
    map.keyframes.insert(4, keyframe(4, &[103, 104]));
    map
}

#[test]
fn builds_recent_local_map_window() {
    let map = map_with_keyframes();

    let window = LocalMapWindow::from_recent(&map, &LocalMapWindowConfig { max_keyframes: 2 });

    assert_eq!(window.anchor_frame_id, Some(4));
    assert_eq!(window.keyframe_ids, vec![3, 4]);
    assert_eq!(window.landmark_ids, vec![102, 103, 104]);
    assert_eq!(window.keyframe_count(), 2);
    assert_eq!(window.landmark_count(), 3);
    assert_eq!(window.observation_count, 4);
    assert!(!window.is_empty());
}

#[test]
fn builds_anchor_window_from_keyframes_at_or_before_anchor() {
    let map = map_with_keyframes();

    let window = LocalMapWindow::from_anchor(&map, 3, &LocalMapWindowConfig { max_keyframes: 2 });

    assert_eq!(window.anchor_frame_id, Some(3));
    assert_eq!(window.keyframe_ids, vec![2, 3]);
    assert_eq!(window.landmark_ids, vec![101, 102, 103]);
}

#[test]
fn clamps_zero_max_keyframes_to_one() {
    let map = map_with_keyframes();

    let window = LocalMapWindow::from_recent(&map, &LocalMapWindowConfig { max_keyframes: 0 });

    assert_eq!(window.keyframe_ids, vec![4]);
    assert_eq!(window.landmark_ids, vec![103, 104]);
}

#[test]
fn from_keyframe_ids_deduplicates_sorts_and_skips_missing_keyframes() {
    let map = map_with_keyframes();

    let window = LocalMapWindow::from_keyframe_ids(&map, Some(4), vec![4, 2, 999, 2]);

    assert_eq!(window.anchor_frame_id, Some(4));
    assert_eq!(window.keyframe_ids, vec![2, 4]);
    assert_eq!(window.landmark_ids, vec![101, 102, 103, 104]);
    assert_eq!(window.observation_count, 4);
}

#[test]
fn empty_map_returns_empty_recent_window() {
    let map = VisualMap::new();

    let window = LocalMapWindow::from_recent(&map, &LocalMapWindowConfig::default());

    assert_eq!(window, LocalMapWindow::default());
    assert!(window.is_empty());
}
