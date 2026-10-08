//! Transport-agnostic stereo pairing and IMU interval bookkeeping.
//!
//! Everything here is plain single-threaded data structures with no DDS or
//! threading dependency, so the exact policies the VIO node runs are unit
//! testable. The DDS shell wraps one [`VioInputs`] in a mutex + condvar.

use std::collections::VecDeque;

use visloc_basalt::ImuSample;

/// Pairs left/right images by header stamp.
///
/// Policy (equivalent to `message_filters` `ApproximateTime` restricted to a
/// hard tolerance, which is what hardware-triggered stereo needs):
///
/// * an incoming image is paired with the buffered image of the *other* side
///   whose stamp is closest, provided `|dt| <= tolerance_ns`;
/// * once a pair is emitted, every buffered image (both sides) at or before
///   the paired stamps is discarded: inputs are assumed time-ordered per
///   side, so those can never pair any more;
/// * images arriving at or before the last emitted pair are dropped as
///   stale/out-of-order;
/// * each side buffers at most `max_pending` images (drop-oldest), so a dead
///   camera can never grow memory.
///
/// The emitted pair carries the **left** stamp, the reference camera (cam0)
/// in Basalt's calibration.
#[derive(Debug)]
pub struct StereoSynchronizer<T> {
    tolerance_ns: i64,
    max_pending: usize,
    left: VecDeque<(i64, T)>,
    right: VecDeque<(i64, T)>,
    last_pair_left_ns: Option<i64>,
    last_pair_right_ns: Option<i64>,
    dropped: u64,
}

/// A matched stereo pair.
#[derive(Clone, Debug, PartialEq)]
pub struct StereoPair<T> {
    pub stamp_ns: i64,
    pub right_stamp_ns: i64,
    pub left: T,
    pub right: T,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

impl<T> StereoSynchronizer<T> {
    pub fn new(tolerance_ns: i64, max_pending: usize) -> Self {
        Self {
            tolerance_ns: tolerance_ns.max(0),
            max_pending: max_pending.max(1),
            left: VecDeque::new(),
            right: VecDeque::new(),
            last_pair_left_ns: None,
            last_pair_right_ns: None,
            dropped: 0,
        }
    }

    /// Images discarded so far (unmatched, stale, or evicted).
    pub const fn dropped(&self) -> u64 {
        self.dropped
    }

    pub fn pending(&self) -> (usize, usize) {
        (self.left.len(), self.right.len())
    }

    pub fn push_left(&mut self, stamp_ns: i64, image: T) -> Option<StereoPair<T>> {
        self.push(Side::Left, stamp_ns, image)
    }

    pub fn push_right(&mut self, stamp_ns: i64, image: T) -> Option<StereoPair<T>> {
        self.push(Side::Right, stamp_ns, image)
    }

    fn push(&mut self, side: Side, stamp_ns: i64, image: T) -> Option<StereoPair<T>> {
        let last = match side {
            Side::Left => self.last_pair_left_ns,
            Side::Right => self.last_pair_right_ns,
        };
        if last.is_some_and(|last| stamp_ns <= last) {
            self.dropped += 1;
            return None;
        }
        let (own, other) = match side {
            Side::Left => (&mut self.left, &mut self.right),
            Side::Right => (&mut self.right, &mut self.left),
        };
        let best = other
            .iter()
            .enumerate()
            .map(|(index, (other_ns, _))| (index, (other_ns - stamp_ns).abs()))
            .filter(|(_, dt)| *dt <= self.tolerance_ns)
            .min_by_key(|(_, dt)| *dt)
            .map(|(index, _)| index);
        let Some(index) = best else {
            // Keep ordered by stamp even if a side delivers slightly out of
            // order (e.g. two subscriptions racing).
            let position = own.partition_point(|(ns, _)| *ns <= stamp_ns);
            own.insert(position, (stamp_ns, image));
            while own.len() > self.max_pending {
                own.pop_front();
                self.dropped += 1;
            }
            return None;
        };
        // Everything older than the match on the other side can never pair.
        for _ in 0..index {
            other.pop_front();
            self.dropped += 1;
        }
        let (other_ns, other_image) = other.pop_front().expect("matched index exists");
        // Likewise for buffered images on this side older than this one.
        while own.front().is_some_and(|(ns, _)| *ns <= stamp_ns) {
            own.pop_front();
            self.dropped += 1;
        }
        let (left_ns, right_ns, left, right) = match side {
            Side::Left => (stamp_ns, other_ns, image, other_image),
            Side::Right => (other_ns, stamp_ns, other_image, image),
        };
        self.last_pair_left_ns = Some(left_ns);
        self.last_pair_right_ns = Some(right_ns);
        // Drop images of either side that are not newer than the pair.
        while self.left.front().is_some_and(|(ns, _)| *ns <= left_ns) {
            self.left.pop_front();
            self.dropped += 1;
        }
        while self.right.front().is_some_and(|(ns, _)| *ns <= right_ns) {
            self.right.pop_front();
            self.dropped += 1;
        }
        Some(StereoPair {
            stamp_ns: left_ns,
            right_stamp_ns: right_ns,
            left,
            right,
        })
    }
}

/// Time-ordered IMU sample buffer with Basalt's interval semantics.
#[derive(Debug)]
pub struct ImuBuffer {
    samples: VecDeque<ImuSample>,
    capacity: usize,
    dropped_out_of_order: u64,
    evicted: u64,
}

impl ImuBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::new(),
            capacity: capacity.max(2),
            dropped_out_of_order: 0,
            evicted: 0,
        }
    }

    /// Appends a sample. Non-increasing stamps are rejected (Basalt's IMU
    /// queue requires strictly increasing time) and counted.
    pub fn push(&mut self, sample: ImuSample) -> bool {
        if self
            .samples
            .back()
            .is_some_and(|last| sample.timestamp_ns <= last.timestamp_ns)
        {
            self.dropped_out_of_order += 1;
            return false;
        }
        self.samples.push_back(sample);
        while self.samples.len() > self.capacity {
            self.samples.pop_front();
            self.evicted += 1;
        }
        true
    }

    pub fn latest_ns(&self) -> Option<i64> {
        self.samples.back().map(|sample| sample.timestamp_ns)
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub const fn dropped_out_of_order(&self) -> u64 {
        self.dropped_out_of_order
    }

    pub const fn evicted(&self) -> u64 {
        self.evicted
    }

    /// Samples in the half-open interval `(after_ns, until_ns]`, or every
    /// buffered sample `<= until_ns` when `after_ns` is `None` (first frame),
    /// exactly like the EuRoC reader's `EurocSensorFrame::imu`. Samples
    /// `<= until_ns` are consumed.
    pub fn take_interval(&mut self, after_ns: Option<i64>, until_ns: i64) -> Vec<ImuSample> {
        let mut out = Vec::new();
        while let Some(sample) = self.samples.front() {
            if sample.timestamp_ns > until_ns {
                break;
            }
            let sample = self.samples.pop_front().expect("front exists");
            if after_ns.is_none_or(|after| sample.timestamp_ns > after) {
                out.push(sample);
            }
        }
        out
    }

    /// First buffered sample at or after `stamp_ns` (Basalt's
    /// initialization sample for the very first frame).
    pub fn first_at_or_after(&self, stamp_ns: i64) -> Option<ImuSample> {
        self.samples
            .iter()
            .find(|sample| sample.timestamp_ns >= stamp_ns)
            .copied()
    }
}

/// One estimator input: a stereo pair plus its IMU interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SensorPacket<T> {
    /// Frame time in the IMU clock (`left stamp + cam_time_offset_ns`).
    pub stamp_ns: i64,
    /// Original left-image header stamp (used for output headers).
    pub header_stamp_ns: i64,
    pub left: T,
    pub right: T,
    pub imu: Vec<ImuSample>,
    pub initialization_imu: Option<ImuSample>,
}

/// Counters exposed for diagnostics/logging.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputStats {
    pub pairs: u64,
    pub frames_dropped_behind: u64,
    pub sync_dropped: u64,
    pub imu_out_of_order: u64,
    pub imu_evicted: u64,
}

/// The VIO node's whole input side: stereo pairing, a bounded drop-oldest
/// queue of paired frames, and the IMU buffer. A frame is released only
/// once IMU data reaches its stamp, so every released interval is complete.
#[derive(Debug)]
pub struct VioInputs<T> {
    sync: StereoSynchronizer<T>,
    frames: VecDeque<StereoPair<T>>,
    max_frames: usize,
    imu: ImuBuffer,
    cam_time_offset_ns: i64,
    last_released_ns: Option<i64>,
    frames_dropped_behind: u64,
    pairs: u64,
}

impl<T> VioInputs<T> {
    pub fn new(
        stereo_tolerance_ns: i64,
        max_pending_images: usize,
        max_frames: usize,
        imu_capacity: usize,
        cam_time_offset_ns: i64,
    ) -> Self {
        Self {
            sync: StereoSynchronizer::new(stereo_tolerance_ns, max_pending_images),
            frames: VecDeque::new(),
            max_frames: max_frames.max(1),
            imu: ImuBuffer::new(imu_capacity),
            cam_time_offset_ns,
            last_released_ns: None,
            frames_dropped_behind: 0,
            pairs: 0,
        }
    }

    pub fn push_left(&mut self, stamp_ns: i64, image: T) {
        if let Some(pair) = self.sync.push_left(stamp_ns, image) {
            self.enqueue(pair);
        }
    }

    pub fn push_right(&mut self, stamp_ns: i64, image: T) {
        if let Some(pair) = self.sync.push_right(stamp_ns, image) {
            self.enqueue(pair);
        }
    }

    pub fn push_imu(&mut self, sample: ImuSample) -> bool {
        self.imu.push(sample)
    }

    fn enqueue(&mut self, pair: StereoPair<T>) {
        self.pairs += 1;
        self.frames.push_back(pair);
        while self.frames.len() > self.max_frames {
            // Drop-oldest: the estimator is behind. The next released frame's
            // IMU interval still starts at the last *released* frame, so the
            // preintegration stays contiguous across the gap.
            self.frames.pop_front();
            self.frames_dropped_behind += 1;
        }
    }

    pub fn queued_frames(&self) -> usize {
        self.frames.len()
    }

    /// Stamp (IMU clock) of the oldest queued frame, if any.
    pub fn front_stamp_ns(&self) -> Option<i64> {
        self.frames
            .front()
            .map(|pair| pair.stamp_ns + self.cam_time_offset_ns)
    }

    /// True when the oldest queued frame can be released.
    pub fn front_ready(&self) -> bool {
        match (self.front_stamp_ns(), self.imu.latest_ns()) {
            (Some(frame_ns), Some(imu_ns)) => imu_ns >= frame_ns,
            _ => false,
        }
    }

    /// Releases the oldest queued frame with its IMU interval once IMU data
    /// covers its stamp. Frames not newer than the last released one are
    /// discarded.
    pub fn pop_ready(&mut self) -> Option<SensorPacket<T>> {
        loop {
            let frame_ns = self.front_stamp_ns()?;
            if self.last_released_ns.is_some_and(|last| frame_ns <= last) {
                self.frames.pop_front();
                self.frames_dropped_behind += 1;
                continue;
            }
            if !self.front_ready() {
                return None;
            }
            let pair = self.frames.pop_front().expect("front exists");
            let first = self.last_released_ns.is_none();
            let initialization_imu = if first {
                self.imu.first_at_or_after(frame_ns)
            } else {
                None
            };
            let imu = self.imu.take_interval(self.last_released_ns, frame_ns);
            self.last_released_ns = Some(frame_ns);
            return Some(SensorPacket {
                stamp_ns: frame_ns,
                header_stamp_ns: pair.stamp_ns,
                left: pair.left,
                right: pair.right,
                imu,
                initialization_imu,
            });
        }
    }

    /// Forgets the released-frame history (after an estimator reset).
    pub fn reset_stream(&mut self) {
        self.last_released_ns = None;
    }

    pub fn stats(&self) -> InputStats {
        InputStats {
            pairs: self.pairs,
            frames_dropped_behind: self.frames_dropped_behind,
            sync_dropped: self.sync.dropped(),
            imu_out_of_order: self.imu.dropped_out_of_order(),
            imu_evicted: self.imu.evicted(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn imu(ns: i64) -> ImuSample {
        ImuSample::new(ns, Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81))
    }

    #[test]
    fn exact_stamps_pair_in_either_arrival_order() {
        let mut sync = StereoSynchronizer::new(0, 4);
        assert!(sync.push_left(100, "l100").is_none());
        let pair = sync.push_right(100, "r100").unwrap();
        assert_eq!(
            (pair.stamp_ns, pair.left, pair.right),
            (100, "l100", "r100")
        );
        assert!(sync.push_right(200, "r200").is_none());
        let pair = sync.push_left(200, "l200").unwrap();
        assert_eq!((pair.left, pair.right), ("l200", "r200"));
        assert_eq!(sync.pending(), (0, 0));
        assert_eq!(sync.dropped(), 0);
    }

    #[test]
    fn approximate_pairing_picks_closest_within_tolerance() {
        let mut sync = StereoSynchronizer::new(5, 8);
        sync.push_right(96, "r96");
        sync.push_right(103, "r103");
        let pair = sync.push_left(100, "l100").unwrap();
        // |103-100| = 3 < |96-100| = 4.
        assert_eq!(pair.right, "r103");
        assert_eq!(pair.stamp_ns, 100);
        assert_eq!(pair.right_stamp_ns, 103);
        // r96 is older than the match: discarded.
        assert_eq!(sync.pending(), (0, 0));
        assert_eq!(sync.dropped(), 1);
        // Outside tolerance: buffered, not paired.
        assert!(sync.push_left(200, "l200").is_none());
        assert!(sync.push_right(210, "r210").is_none());
        assert_eq!(sync.pending(), (1, 1));
    }

    #[test]
    fn unmatched_images_are_bounded_and_stale_dropped() {
        let mut sync = StereoSynchronizer::new(0, 3);
        for t in 0..10 {
            assert!(sync.push_left(t * 10, t).is_none());
        }
        assert_eq!(sync.pending(), (3, 0));
        assert_eq!(sync.dropped(), 7);
        let pair = sync.push_right(80, 80).unwrap();
        assert_eq!(pair.left, 8);
        // l70 older than the pair: dropped; l90 stays.
        assert_eq!(sync.pending(), (1, 0));
        // Right image older than the last pair: stale.
        assert!(sync.push_right(50, 50).is_none());
        assert_eq!(sync.pending(), (1, 0));
    }

    #[test]
    fn imu_interval_semantics_match_euroc_reader() {
        let mut buffer = ImuBuffer::new(100);
        for ns in [10, 20, 30, 40, 50] {
            assert!(buffer.push(imu(ns)));
        }
        assert!(!buffer.push(imu(50)), "duplicate stamp rejected");
        assert_eq!(buffer.first_at_or_after(25).unwrap().timestamp_ns, 30);
        let first: Vec<_> = buffer
            .take_interval(None, 30)
            .iter()
            .map(|s| s.timestamp_ns)
            .collect();
        assert_eq!(first, vec![10, 20, 30]);
        let next: Vec<_> = buffer
            .take_interval(Some(30), 50)
            .iter()
            .map(|s| s.timestamp_ns)
            .collect();
        assert_eq!(next, vec![40, 50]);
        assert!(buffer.is_empty());
    }

    #[test]
    fn imu_buffer_evicts_oldest() {
        let mut buffer = ImuBuffer::new(3);
        for ns in 1..=5 {
            buffer.push(imu(ns));
        }
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.evicted(), 2);
        let taken: Vec<_> = buffer
            .take_interval(None, 10)
            .iter()
            .map(|s| s.timestamp_ns)
            .collect();
        assert_eq!(taken, vec![3, 4, 5]);
    }

    #[test]
    fn frames_wait_for_imu_coverage() {
        let mut inputs = VioInputs::new(0, 4, 4, 1000, 0);
        for ns in (0..=100).step_by(5) {
            inputs.push_imu(imu(ns));
        }
        inputs.push_left(50, "l50");
        inputs.push_right(50, "r50");
        inputs.push_left(150, "l150");
        inputs.push_right(150, "r150");
        let first = inputs.pop_ready().unwrap();
        assert_eq!(first.stamp_ns, 50);
        assert_eq!(first.imu.len(), 11); // 0, 5, ..., 50
        assert_eq!(first.initialization_imu.unwrap().timestamp_ns, 50);
        // IMU only reaches 100 < 150: not ready yet.
        assert!(inputs.pop_ready().is_none());
        for ns in (105..=150).step_by(5) {
            inputs.push_imu(imu(ns));
        }
        let second = inputs.pop_ready().unwrap();
        assert_eq!(second.stamp_ns, 150);
        assert!(second.initialization_imu.is_none());
        let stamps: Vec<_> = second.imu.iter().map(|s| s.timestamp_ns).collect();
        assert_eq!(stamps.first(), Some(&55));
        assert_eq!(stamps.last(), Some(&150));
        assert_eq!(stamps.len(), 20);
    }

    #[test]
    fn drop_oldest_frames_keep_imu_contiguous() {
        let mut inputs = VioInputs::new(0, 4, 2, 1000, 0);
        for ns in (0..=40).step_by(10) {
            inputs.push_left(ns, ns);
            inputs.push_right(ns, ns);
        }
        // Five pairs into a two-frame queue: three dropped.
        assert_eq!(inputs.queued_frames(), 2);
        assert_eq!(inputs.stats().frames_dropped_behind, 3);
        for ns in 0..=40 {
            inputs.push_imu(imu(ns));
        }
        let a = inputs.pop_ready().unwrap();
        let b = inputs.pop_ready().unwrap();
        assert_eq!((a.stamp_ns, b.stamp_ns), (30, 40));
        assert_eq!(a.imu.len(), 31); // 0..=30 for the first frame
        let stamps: Vec<_> = b.imu.iter().map(|s| s.timestamp_ns).collect();
        assert_eq!(stamps, (31..=40).collect::<Vec<_>>());
        assert!(inputs.pop_ready().is_none());
    }

    #[test]
    fn camera_time_offset_shifts_into_imu_clock() {
        let mut inputs = VioInputs::new(0, 4, 4, 1000, 7);
        for ns in 0..=20 {
            inputs.push_imu(imu(ns));
        }
        inputs.push_left(10, ());
        inputs.push_right(10, ());
        let packet = inputs.pop_ready().unwrap();
        assert_eq!(packet.stamp_ns, 17);
        assert_eq!(packet.header_stamp_ns, 10);
        assert_eq!(packet.imu.last().unwrap().timestamp_ns, 17);
    }
}
