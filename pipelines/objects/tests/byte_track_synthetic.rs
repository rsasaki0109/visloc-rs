//! ByteTracker on synthetic box sequences with known identities, scored with
//! CLEAR MOT: ID switches, occlusion, re-entry, false positives, classes.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use visloc_objects::{
    BBox, ByteTrackConfig, ByteTracker, Detection, MotAccumulator, MotSummary, TrackId,
    TrackedObject,
};

/// An object moving at constant velocity: centre `(x0, y0) + k (vx, vy)` in frame `k`.
#[derive(Debug, Clone, Copy)]
struct Mover {
    id: u64,
    class_id: u32,
    x0: f64,
    y0: f64,
    vx: f64,
    vy: f64,
    width: f64,
    height: f64,
}

impl Mover {
    fn new(id: u64, x0: f64, y0: f64, vx: f64, vy: f64, width: f64, height: f64) -> Self {
        Self {
            id,
            class_id: 0,
            x0,
            y0,
            vx,
            vy,
            width,
            height,
        }
    }

    fn bbox(&self, k: usize) -> BBox {
        let k = k as f64;
        BBox::from_center_size(
            self.x0 + self.vx * k,
            self.y0 + self.vy * k,
            self.width,
            self.height,
        )
    }
}

#[derive(Debug, Default)]
struct Frame {
    truth: Vec<(u64, BBox)>,
    detections: Vec<Detection>,
}

/// The box with its centre and size jittered by up to `fraction` of its height.
fn jitter(bbox: BBox, fraction: f64, rng: &mut SmallRng) -> BBox {
    if fraction == 0.0 {
        return bbox;
    }
    let (cx, cy) = bbox.center();
    let h = bbox.height();
    let mut d = || rng.gen_range(-fraction..fraction) * h;
    let (dx, dy, dw, dh) = (d(), d(), d(), d());
    BBox::from_center_size(cx + dx, cy + dy, bbox.width() + dw, h + dh)
}

struct Run {
    summary: MotSummary,
    outputs: Vec<Vec<TrackedObject>>,
}

impl Run {
    fn ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.outputs.iter().flatten().map(|o| o.id.0).collect();
        ids.sort();
        ids.dedup();
        ids
    }
}

fn run(tracker: &mut ByteTracker, frames: &[Frame]) -> Run {
    let mut accumulator = MotAccumulator::new(0.5);
    let mut outputs = Vec::new();
    for frame in frames {
        let tracks = tracker.update(&frame.detections);
        let hypotheses: Vec<(u64, BBox)> = tracks.iter().map(|t| (t.id.0, t.bbox)).collect();
        accumulator.add_frame(&frame.truth, &hypotheses);
        outputs.push(tracks);
    }
    Run {
        summary: accumulator.summary(),
        outputs,
    }
}

/// Frames where every visible mover is detected with score 0.9 and `noise`.
fn frames(movers: &[Mover], count: usize, noise: f64, seed: u64) -> Vec<Frame> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..count)
        .map(|k| {
            let mut frame = Frame::default();
            for m in movers {
                frame.truth.push((m.id, m.bbox(k)));
                frame.detections.push(Detection::new(
                    jitter(m.bbox(k), noise, &mut rng),
                    0.9,
                    m.class_id,
                ));
            }
            frame
        })
        .collect()
}

#[test]
fn steady_objects_keep_their_ids() {
    let movers: Vec<Mover> = (0..5)
        .map(|i| {
            Mover::new(
                i + 1,
                100.0 + 220.0 * i as f64,
                200.0 + 40.0 * i as f64,
                2.0,
                1.0,
                50.0,
                120.0,
            )
        })
        .collect();
    let frames = frames(&movers, 100, 0.02, 1);
    let result = run(&mut ByteTracker::default(), &frames);
    let s = result.summary;
    assert_eq!(
        (s.id_switches, s.misses, s.false_positives),
        (0, 0, 0),
        "{s:?}"
    );
    assert_eq!(result.ids(), vec![1, 2, 3, 4, 5]);
    assert!(s.motp_iou() > 0.9, "{s:?}");
}

#[test]
fn a_new_object_is_reported_from_its_second_frame_with_the_next_id() {
    let a = Mover::new(1, 100.0, 200.0, 3.0, 0.0, 50.0, 120.0);
    let b = Mover::new(2, 700.0, 300.0, -3.0, 0.0, 50.0, 120.0);
    let mut frames = frames(&[a], 30, 0.0, 2);
    for (k, frame) in frames.iter_mut().enumerate().skip(10) {
        frame.truth.push((b.id, b.bbox(k)));
        frame.detections.push(Detection::new(b.bbox(k), 0.9, 0));
    }
    let result = run(&mut ByteTracker::default(), &frames);
    assert!(result.outputs[10].iter().all(|o| o.id == TrackId(1)));
    let b_track = result.outputs[11]
        .iter()
        .find(|o| o.id != TrackId(1))
        .expect("b at frame 11");
    assert_eq!(b_track.id, TrackId(2));
    assert_eq!(b_track.start_frame, 11);
    assert_eq!(b_track.detection_index, 1);
    assert_eq!((result.summary.misses, result.summary.id_switches), (1, 0));
}

/// Two people cross; the farther one is partly hidden while they overlap and
/// its detector score drops to 0.3.
fn crossing_frames() -> Vec<Frame> {
    let near = Mover::new(1, 200.0, 300.0, 3.0, 0.0, 60.0, 150.0);
    let far = Mover::new(2, 600.0, 280.0, -3.0, 0.0, 40.0, 100.0);
    let mut rng = SmallRng::seed_from_u64(3);
    (0..120)
        .map(|k| {
            let occluded = near.bbox(k).iou(&far.bbox(k)) > 0.0;
            let far_score = if occluded { 0.3 } else { 0.9 };
            Frame {
                truth: vec![(near.id, near.bbox(k)), (far.id, far.bbox(k))],
                detections: vec![
                    Detection::new(jitter(near.bbox(k), 0.01, &mut rng), 0.9, 0),
                    Detection::new(jitter(far.bbox(k), 0.01, &mut rng), far_score, 0),
                ],
            }
        })
        .collect()
}

#[test]
fn low_score_detections_keep_an_occluded_object_tracked() {
    let frames = crossing_frames();
    let occluded = frames
        .iter()
        .filter(|f| f.detections[1].score < 0.5)
        .count();
    assert!(
        occluded >= 10,
        "the scenario should occlude for a while: {occluded}"
    );

    let result = run(&mut ByteTracker::default(), &frames);
    let s = result.summary;
    assert_eq!(
        (s.id_switches, s.misses, s.false_positives),
        (0, 0, 0),
        "{s:?}"
    );
    assert_eq!(result.ids(), vec![1, 2]);

    // Without the low-score pass the occluded object drops out for as long as
    // it is occluded, and is re-found afterwards under its old ID.
    let config = ByteTrackConfig {
        low_score_threshold: 0.5,
        ..ByteTrackConfig::default()
    };
    let result = run(&mut ByteTracker::new(config), &frames);
    let s = result.summary;
    assert_eq!(s.misses, occluded, "{s:?}");
    assert_eq!(s.id_switches, 0, "{s:?}");
    assert_eq!(result.ids(), vec![1, 2]);
}

fn gap_frames(gap: std::ops::Range<usize>, count: usize) -> Vec<Frame> {
    let mover = Mover::new(1, 100.0, 200.0, 4.0, 0.5, 50.0, 120.0);
    let mut frames = frames(&[mover], count, 0.01, 4);
    for k in gap {
        frames[k].detections.clear();
    }
    frames
}

#[test]
fn a_short_occlusion_keeps_the_id() {
    let frames = gap_frames(30..45, 80);
    let mut tracker = ByteTracker::default();
    let mut accumulator = MotAccumulator::new(0.5);
    for (k, frame) in frames.iter().enumerate() {
        let tracks = tracker.update(&frame.detections);
        if (30..45).contains(&k) {
            assert!(tracks.is_empty());
            assert_eq!(tracker.lost_track_ids(), vec![TrackId(1)], "frame {k}");
        }
        let hypotheses: Vec<(u64, BBox)> = tracks.iter().map(|t| (t.id.0, t.bbox)).collect();
        accumulator.add_frame(&frame.truth, &hypotheses);
    }
    let s = accumulator.summary();
    assert_eq!(
        (s.misses, s.id_switches, s.false_positives),
        (15, 0, 0),
        "{s:?}"
    );
}

#[test]
fn after_a_long_absence_the_object_gets_a_new_id() {
    // Lost for 40 frames, more than the 30 a lost track is kept.
    let frames = gap_frames(30..70, 100);
    let mut tracker = ByteTracker::default();
    let result = run(&mut tracker, &frames);
    assert_eq!(result.ids(), vec![1, 2]);
    let s = result.summary;
    // 40 frames hidden, plus the frame that starts the new track.
    assert_eq!((s.misses, s.id_switches), (41, 1), "{s:?}");
    assert!(tracker.lost_track_ids().is_empty());
}

#[test]
fn one_off_false_positives_are_never_reported() {
    let mover = Mover::new(1, 300.0, 300.0, 2.0, 0.0, 50.0, 120.0);
    let mut frames = frames(&[mover], 100, 0.01, 5);
    let mut rng = SmallRng::seed_from_u64(6);
    // From the second frame on (tracks started in the first frame are
    // confirmed at once, as in ByteTrack), a confident detection somewhere
    // else in every frame, never twice at the same place.
    for frame in frames.iter_mut().skip(1) {
        let (w, h) = (rng.gen_range(20.0..80.0), rng.gen_range(40.0..160.0));
        let x = rng.gen_range(600.0..1200.0);
        let y = rng.gen_range(0.0..600.0);
        frame
            .detections
            .push(Detection::new(BBox::from_xywh(x, y, w, h), 0.8, 0));
    }
    let result = run(&mut ByteTracker::default(), &frames);
    assert_eq!(result.summary.false_positives, 0, "{:?}", result.summary);
    assert_eq!(result.ids(), vec![1]);
}

#[test]
fn class_aware_association_keeps_classes_apart() {
    // The same box is a person (class 0) for ten frames, then a car (class 2).
    let bbox = BBox::from_xywh(200.0, 200.0, 60.0, 120.0);
    let detections: Vec<Vec<Detection>> = (0..20)
        .map(|k| vec![Detection::new(bbox, 0.9, if k < 10 { 0 } else { 2 })])
        .collect();

    let mut tracker = ByteTracker::default();
    let outputs: Vec<Vec<TrackedObject>> = detections.iter().map(|d| tracker.update(d)).collect();
    assert!(outputs[10].is_empty(), "the car track starts unconfirmed");
    assert_eq!(outputs[11][0].id, TrackId(2));
    assert_eq!(outputs[11][0].class_id, 2);
    assert_eq!(tracker.lost_track_ids(), vec![TrackId(1)]);

    let mut tracker = ByteTracker::new(ByteTrackConfig {
        class_aware: false,
        ..ByteTrackConfig::default()
    });
    let outputs: Vec<Vec<TrackedObject>> = detections.iter().map(|d| tracker.update(d)).collect();
    assert!(outputs
        .iter()
        .all(|o| o.len() == 1 && o[0].id == TrackId(1) && o[0].class_id == 0));
}

#[test]
fn invalid_detections_are_ignored_and_reset_starts_over() {
    let good = Detection::new(BBox::from_xywh(10.0, 10.0, 20.0, 40.0), 0.9, 0);
    let bad = [
        Detection::new(BBox::new(0.0, 0.0, f64::NAN, 10.0), 0.9, 0),
        Detection::new(BBox::new(5.0, 5.0, 5.0, 20.0), 0.9, 0),
        Detection::new(BBox::from_xywh(300.0, 10.0, 20.0, 40.0), f64::NAN, 0),
    ];
    let mut tracker = ByteTracker::default();
    let tracks = tracker.update(&[bad[0], good, bad[1], bad[2]]);
    assert_eq!(tracks.len(), 1);
    assert_eq!(tracks[0].detection_index, 1);
    tracker.update(&[good]);
    tracker.reset();
    assert_eq!(tracker.frame(), 0);
    assert_eq!(tracker.update(&[good])[0].id, TrackId(1));
}

#[test]
fn a_noisy_crowd_is_tracked_with_few_errors() {
    let mut rng = SmallRng::seed_from_u64(11);
    let movers: Vec<Mover> = (0..12)
        .map(|i| {
            let height = rng.gen_range(60.0..200.0);
            Mover {
                id: i + 1,
                class_id: if i % 3 == 0 { 2 } else { 0 },
                x0: rng.gen_range(100.0..1180.0),
                y0: rng.gen_range(150.0..570.0),
                vx: rng.gen_range(-4.0..4.0),
                vy: rng.gen_range(-1.0..1.0),
                width: height * rng.gen_range(0.35..0.6),
                height,
            }
        })
        .collect();
    let mut frames = Vec::new();
    for k in 0..200 {
        let mut frame = Frame::default();
        for m in &movers {
            frame.truth.push((m.id, m.bbox(k)));
            let roll: f64 = rng.gen();
            if roll < 0.05 {
                continue; // missed by the detector
            }
            let score = if roll < 0.15 {
                rng.gen_range(0.2..0.5)
            } else {
                rng.gen_range(0.6..0.95)
            };
            frame.detections.push(Detection::new(
                jitter(m.bbox(k), 0.03, &mut rng),
                score,
                m.class_id,
            ));
        }
        let x = rng.gen_range(0.0..1200.0);
        let y = rng.gen_range(0.0..600.0);
        frame
            .detections
            .push(Detection::new(BBox::from_xywh(x, y, 40.0, 90.0), 0.7, 0));
        frames.push(frame);
    }
    let result = run(&mut ByteTracker::default(), &frames);
    let s = result.summary;
    println!(
        "crowd: {s:?}, MOTA {:.3}, IoU {:.3}",
        s.mota(),
        s.motp_iou()
    );
    assert!(s.mota() > 0.9, "{s:?}");
    assert!(s.id_switches <= 3, "{s:?}");
}
