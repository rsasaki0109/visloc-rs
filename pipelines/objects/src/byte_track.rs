//! ByteTrack: multi-object tracking by associating every detection box.
//!
//! Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
//! Detection Box", ECCV 2022. Per frame:
//!
//! 1. Confirmed and lost tracks are predicted with a Kalman filter and matched
//!    to the high-score detections by IoU.
//! 2. Confirmed tracks left over are matched to the low-score detections, which
//!    are often the same objects partly occluded. Tracks still unmatched are
//!    lost; a lost track can be re-found in step 1 for `max_frames_lost` frames.
//! 3. Tracks started last frame are matched to the remaining high-score
//!    detections; they are confirmed on a match and dropped otherwise.
//! 4. Remaining high-score detections start new tracks.
//!
//! This follows the reference implementation (`yolox/tracker/byte_tracker.py`)
//! with two additions: association can be restricted to tracks and
//! detections of the same class, and a track gets its ID when it is
//! confirmed, so reported IDs have no gaps.

use crate::assignment::assign_with_cost_limit;
use crate::bbox::{BBox, Detection};
use crate::kalman::{BoxKalmanFilter, BoxKalmanState};

/// Settings of [`ByteTracker`]; the defaults are ByteTrack's for 30 fps video.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ByteTrackConfig {
    /// Detections with at least this score are associated first (`track_thresh`, 0.5).
    pub high_score_threshold: f64,
    /// Detections with a score in `[low_score_threshold, high_score_threshold)`
    /// are associated second, only with confirmed tracks (0.1).
    pub low_score_threshold: f64,
    /// A new track starts only from a detection with at least this score
    /// (`track_thresh + 0.1`, 0.6).
    pub new_track_score_threshold: f64,
    /// Largest IoU distance (`1 - IoU`) for the first association (`match_thresh`, 0.8).
    pub first_match_cost_limit: f64,
    /// Largest IoU distance for the low-score association (0.5).
    pub second_match_cost_limit: f64,
    /// Largest IoU distance for confirming new tracks (0.7).
    pub unconfirmed_match_cost_limit: f64,
    /// Weight IoU by the detection score in the first and the confirming
    /// association (ByteTrack's default; off for MOT20).
    pub fuse_score: bool,
    /// Frames a lost track is kept before it is dropped. ByteTrack uses
    /// `frame_rate / 30 * track_buffer`, one second of video.
    pub max_frames_lost: u64,
    /// Associate detections only with tracks of the same class. The reference
    /// tracker ignores classes.
    pub class_aware: bool,
    /// A tracked and a lost track closer than this IoU distance are the same
    /// object; the younger one is dropped (0.15).
    pub duplicate_cost_limit: f64,
    pub kalman: BoxKalmanFilter,
}

impl Default for ByteTrackConfig {
    fn default() -> Self {
        Self {
            high_score_threshold: 0.5,
            low_score_threshold: 0.1,
            new_track_score_threshold: 0.6,
            first_match_cost_limit: 0.8,
            second_match_cost_limit: 0.5,
            unconfirmed_match_cost_limit: 0.7,
            fuse_score: true,
            max_frames_lost: 30,
            class_aware: true,
            duplicate_cost_limit: 0.15,
            kalman: BoxKalmanFilter::default(),
        }
    }
}

/// Identifier of a confirmed track, unique within a sequence and starting at 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TrackId(pub u64);

/// A confirmed track that was matched to a detection in this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackedObject {
    pub id: TrackId,
    /// The class of the detection that started the track.
    pub class_id: u32,
    /// The Kalman-filtered box after this frame's update.
    pub bbox: BBox,
    /// Score of the matched detection.
    pub score: f64,
    /// Index of the matched detection in this frame's input.
    pub detection_index: usize,
    /// Frame (1-based) in which the track started.
    pub start_frame: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrackState {
    Tracked,
    Lost,
}

#[derive(Debug, Clone)]
struct Track {
    id: Option<TrackId>,
    class_id: u32,
    kalman: BoxKalmanState,
    state: TrackState,
    score: f64,
    start_frame: u64,
    /// Frame of the last matched detection.
    last_frame: u64,
    /// Detection matched in the current frame.
    detection_index: Option<usize>,
}

impl Track {
    fn bbox(&self) -> BBox {
        self.kalman.bbox()
    }
}

/// ByteTrack over a sequence of frames; see the module documentation.
#[derive(Debug, Clone)]
pub struct ByteTracker {
    config: ByteTrackConfig,
    frame: u64,
    next_id: u64,
    /// Tracks matched in the last frame, confirmed or not.
    tracked: Vec<Track>,
    lost: Vec<Track>,
}

impl Default for ByteTracker {
    fn default() -> Self {
        Self::new(ByteTrackConfig::default())
    }
}

impl ByteTracker {
    pub fn new(config: ByteTrackConfig) -> Self {
        Self {
            config,
            frame: 0,
            next_id: 1,
            tracked: Vec::new(),
            lost: Vec::new(),
        }
    }

    pub fn config(&self) -> &ByteTrackConfig {
        &self.config
    }

    /// Frames processed so far.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// IDs of the confirmed tracks that are lost and can still be re-found.
    pub fn lost_track_ids(&self) -> Vec<TrackId> {
        let mut ids: Vec<TrackId> = self.lost.iter().filter_map(|t| t.id).collect();
        ids.sort();
        ids
    }

    /// Forget all tracks and start a new sequence; IDs start again at 1.
    pub fn reset(&mut self) {
        *self = Self::new(self.config);
    }

    /// Process one frame's detections and return the confirmed tracks matched
    /// in it, sorted by ID. Detections with an invalid box or a non-finite
    /// score are ignored.
    pub fn update(&mut self, detections: &[Detection]) -> Vec<TrackedObject> {
        self.frame += 1;
        let frame = self.frame;
        let config = self.config;
        let kalman = config.kalman;

        let usable = |d: &Detection| d.bbox.is_valid() && d.score.is_finite();
        let high: Vec<usize> = (0..detections.len())
            .filter(|&i| {
                usable(&detections[i]) && detections[i].score >= config.high_score_threshold
            })
            .collect();
        let low: Vec<usize> = (0..detections.len())
            .filter(|&i| {
                let score = detections[i].score;
                usable(&detections[i])
                    && score >= config.low_score_threshold
                    && score < config.high_score_threshold
            })
            .collect();

        let (mut pool, mut unconfirmed): (Vec<Track>, Vec<Track>) =
            std::mem::take(&mut self.tracked)
                .into_iter()
                .partition(|t| t.id.is_some());
        pool.append(&mut self.lost);
        for track in pool.iter_mut().chain(unconfirmed.iter_mut()) {
            track.detection_index = None;
        }
        // Unconfirmed tracks are not predicted, as in the reference.
        for track in &mut pool {
            if track.state == TrackState::Lost {
                track.kalman.mean[7] = 0.0;
            }
            kalman.predict(&mut track.kalman);
        }

        // 1. Confirmed and lost tracks against high-score detections.
        let first = self.associate(
            &pool.iter().collect::<Vec<_>>(),
            &high,
            detections,
            config.fuse_score,
            config.first_match_cost_limit,
        );
        for &(t, d) in &first.matches {
            self.apply(&mut pool[t], detections, high[d]);
        }
        let remaining_high: Vec<usize> = first.unmatched_cols.iter().map(|&d| high[d]).collect();

        // 2. Confirmed tracks still tracked against low-score detections.
        let still_tracked: Vec<usize> = first
            .unmatched_rows
            .iter()
            .copied()
            .filter(|&t| pool[t].state == TrackState::Tracked)
            .collect();
        let second = self.associate(
            &still_tracked.iter().map(|&t| &pool[t]).collect::<Vec<_>>(),
            &low,
            detections,
            false,
            config.second_match_cost_limit,
        );
        for &(t, d) in &second.matches {
            self.apply(&mut pool[still_tracked[t]], detections, low[d]);
        }
        for &t in &second.unmatched_rows {
            pool[still_tracked[t]].state = TrackState::Lost;
        }

        // 3. Tracks started last frame against the remaining high-score detections.
        let third = self.associate(
            &unconfirmed.iter().collect::<Vec<_>>(),
            &remaining_high,
            detections,
            config.fuse_score,
            config.unconfirmed_match_cost_limit,
        );
        let mut confirmed = Vec::with_capacity(third.matches.len());
        for &(t, d) in &third.matches {
            let mut track = unconfirmed[t].clone();
            self.apply(&mut track, detections, remaining_high[d]);
            confirmed.push(track);
        }

        // 4. New tracks; on the first frame they are confirmed at once.
        let mut started = Vec::new();
        for &d in &third.unmatched_cols {
            let detection = &detections[remaining_high[d]];
            if detection.score < config.new_track_score_threshold {
                continue;
            }
            let mut track = Track {
                id: None,
                class_id: detection.class_id,
                kalman: kalman.initiate(&detection.bbox),
                state: TrackState::Tracked,
                score: detection.score,
                start_frame: frame,
                last_frame: frame,
                detection_index: Some(remaining_high[d]),
            };
            if frame == 1 {
                track.id = Some(self.take_id());
            }
            started.push(track);
        }

        let (mut tracked, lost): (Vec<Track>, Vec<Track>) = pool
            .into_iter()
            .partition(|t| t.state == TrackState::Tracked);
        tracked.extend(confirmed);
        tracked.extend(started);
        let lost: Vec<Track> = lost
            .into_iter()
            .filter(|t| frame - t.last_frame <= config.max_frames_lost)
            .collect();
        let (tracked, lost) = self.remove_duplicates(tracked, lost);
        self.tracked = tracked;
        self.lost = lost;

        let mut output: Vec<TrackedObject> = self
            .tracked
            .iter()
            .filter_map(|t| {
                Some(TrackedObject {
                    id: t.id?,
                    class_id: t.class_id,
                    bbox: t.bbox(),
                    score: t.score,
                    detection_index: t.detection_index?,
                    start_frame: t.start_frame,
                })
            })
            .collect();
        output.sort_by_key(|o| o.id);
        output
    }

    fn take_id(&mut self) -> TrackId {
        let id = TrackId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Update a track with a detection; this re-finds a lost track and
    /// confirms a new one.
    fn apply(&mut self, track: &mut Track, detections: &[Detection], index: usize) {
        let detection = &detections[index];
        self.config
            .kalman
            .update(&mut track.kalman, &detection.bbox);
        track.state = TrackState::Tracked;
        track.score = detection.score;
        track.last_frame = self.frame;
        track.detection_index = Some(index);
        if track.id.is_none() {
            track.id = Some(self.take_id());
        }
    }

    /// IoU-distance assignment of `tracks` to `detections[indices]`.
    fn associate(
        &self,
        tracks: &[&Track],
        indices: &[usize],
        detections: &[Detection],
        fuse_score: bool,
        cost_limit: f64,
    ) -> crate::assignment::Assignment {
        let mut costs = Vec::with_capacity(tracks.len() * indices.len());
        for track in tracks {
            let bbox = track.bbox();
            for &i in indices {
                let detection = &detections[i];
                let cost = if self.config.class_aware && track.class_id != detection.class_id {
                    f64::INFINITY
                } else {
                    let iou = bbox.iou(&detection.bbox);
                    1.0 - if fuse_score {
                        iou * detection.score
                    } else {
                        iou
                    }
                };
                costs.push(cost);
            }
        }
        assign_with_cost_limit(tracks.len(), indices.len(), &costs, cost_limit)
    }

    /// Of a tracked and a lost track that overlap almost completely, keep the
    /// one that has existed longer.
    fn remove_duplicates(&self, tracked: Vec<Track>, lost: Vec<Track>) -> (Vec<Track>, Vec<Track>) {
        let mut drop_tracked = vec![false; tracked.len()];
        let mut drop_lost = vec![false; lost.len()];
        for (p, a) in tracked.iter().enumerate() {
            for (q, b) in lost.iter().enumerate() {
                if self.config.class_aware && a.class_id != b.class_id {
                    continue;
                }
                if 1.0 - a.bbox().iou(&b.bbox()) < self.config.duplicate_cost_limit {
                    if a.last_frame - a.start_frame > b.last_frame - b.start_frame {
                        drop_lost[q] = true;
                    } else {
                        drop_tracked[p] = true;
                    }
                }
            }
        }
        let keep = |tracks: Vec<Track>, drop: &[bool]| {
            tracks
                .into_iter()
                .zip(drop)
                .filter_map(|(t, &d)| (!d).then_some(t))
                .collect::<Vec<_>>()
        };
        (keep(tracked, &drop_tracked), keep(lost, &drop_lost))
    }
}
