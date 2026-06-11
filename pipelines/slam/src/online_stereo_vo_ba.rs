//! Online sliding-window stereo VO + BA composition.
//!
//! Wraps a [`StereoVoFrontend`] and triggers
//! [`refine_stereo_vo_with_ba`] every `trigger_every_frames` processed
//! pairs over the trailing `window_size` frames. The first pose of each
//! triggered window is held fixed (the global anchor for the first
//! trigger, the previously-refined boundary pose for subsequent
//! triggers), so the refined window slides forward without
//! re-optimising already-frozen history.
//!
//! Keeping BA out of the frontend itself avoids a vision -> slam
//! dependency cycle. The frontend exposes the filtered temporal
//! matches it actually used via
//! [`StereoVoFrontend::temporal_matches_per_pair`]; the wrapper just
//! re-uses those captures plus the frontend's pose / feature buffers.
//!
//! Pose 0 of the trailing window is always treated as fixed because
//! `refine_stereo_vo_with_ba` fixes pose 0 internally. The caller does
//! not need to pin any pose explicitly.
//!
//! Memory footprint scales with the *whole* trajectory, not the
//! window: the frontend keeps every pose / feature set / stereo set /
//! temporal-match vector so far. For long real-time use, callers would
//! eventually need to evict old keyframes — that is out of scope here.

use std::io::Write;
use std::path::Path;

use visloc_vision::features::{FeatureExtractor, FeatureSet, GrayscaleImage};
use visloc_vision::matching::{DescriptorMatch, Matcher};
use visloc_vision::stereo_vo::{StereoVoError, StereoVoFrontend};

use crate::stereo_vo_ba::{
    refine_stereo_vo_with_ba, StereoVoBaConfig, StereoVoBaError, StereoVoBaImuInput,
    StereoVoBaRefinement,
};
use visloc_core::geometry::Pose;

/// Configuration for [`OnlineStereoVoBa`].
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineStereoVoBaConfig {
    /// Run BA every `trigger_every_frames` processed pairs (counted by
    /// the number of poses in the frontend, not by wall-clock frames).
    /// A value of `0` disables auto-triggering — callers can still
    /// invoke [`OnlineStereoVoBa::run_ba_now`] manually. Sensible
    /// default: `5..=10` for KITTI-rate stereo VO.
    pub trigger_every_frames: usize,
    /// Trailing window size passed to the refiner each trigger. The
    /// refiner sees the most recent `window_size` poses + features +
    /// stereo + temporal matches and treats its pose 0 as fixed (the
    /// boundary pose from the previous trigger, or the global anchor
    /// for the first trigger). Should be at least
    /// `ba_config.min_track_length`.
    pub window_size: usize,
    /// Underlying refiner config (track gates, Huber kernel, LM
    /// schedule, etc). The wrapper does NOT override
    /// `ba_config.window_size`; the refiner runs as a single joint BA
    /// over the trailing window slice every trigger.
    ///
    /// `ba_config.imu_input` must be `None`; the wrapper-level
    /// [`Self::imu_input`] is sliced per trigger and injected as the
    /// effective IMU input for the trailing window. Setting both at
    /// once raises `StereoVoBaError::InvalidImuInput` on the first
    /// trigger.
    pub ba_config: StereoVoBaConfig,
    /// Global IMU pre-integration input spanning the entire trajectory.
    /// `windows[i]` covers the integration interval between keyframes
    /// `i` and `i + 1`; on each trigger the wrapper slices
    /// `windows[start..end - 1]` to align with the trailing BA window
    /// and rebuilds a [`StereoVoBaImuInput`] (gravity / bias linearisation
    /// / weights / fix-first flags are passed through verbatim, so
    /// `fix_first_bias` / `fix_first_velocity` apply to the first
    /// keyframe of *each* trailing window, mirroring the post-process
    /// BA semantics). `None` runs the wrapper as visual-only sliding BA.
    pub imu_input: Option<StereoVoBaImuInput>,
    /// Extra backward "local map" history (in frames) prepended to the trailing
    /// optimisation window. `0` (default) is the classic window BA: optimise the
    /// trailing `window_size` poses, fixing only pose 0 of that window. When
    /// `> 0`, the BA window is extended back by this many frames, those older
    /// frames are held FIXED (anchoring long-baseline landmarks that persist from
    /// earlier windows), and only the trailing `window_size` poses are optimised
    /// — the fixed-keyframe local-mapping pattern. This lets a landmark constrain
    /// the recent poses over a baseline far longer than `window_size` without
    /// re-optimising (or deforming) the already-settled older trajectory.
    /// Incompatible with `imu_input` (the IMU slice aligns to the trailing
    /// window only); ignored when `imu_input` is set.
    pub local_map_history: usize,
    /// Exclude the temporal matches of frontend-rescued pairs from BA track
    /// building. When a pair's weak-consensus pose was clamped by a frontend
    /// rescue (translation-direction / rotation-spike / rotation-vector), its
    /// matches voted for a relative pose the motion model rejected as
    /// implausible — typically a dynamic object capturing the PnP consensus.
    /// Feeding those same matches to BA lets the optimiser re-impose the
    /// rejected motion onto the clamped poses (and smear it across the
    /// window). With this flag the rescued pair contributes no temporal
    /// matches: its tracks break at the contaminated pair and the clamped
    /// odometry carries the trajectory through the event.
    pub exclude_rescued_pair_matches: bool,
}

impl Default for OnlineStereoVoBaConfig {
    fn default() -> Self {
        Self {
            trigger_every_frames: 10,
            window_size: 20,
            ba_config: StereoVoBaConfig::default(),
            imu_input: None,
            local_map_history: 0,
            exclude_rescued_pair_matches: false,
        }
    }
}

/// Per-trigger BA outcome, exposed so callers can inspect the LM
/// trace, the number of tracks/observations refined, and how many
/// poses the trigger touched.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineBaTriggerStats {
    /// Index of the first pose in the refined window (held fixed).
    pub window_start: usize,
    /// Exclusive index of the last pose in the refined window.
    pub window_end: usize,
    /// Refiner result. `Ok` if BA actually solved; `Err` if track gates
    /// or "too few frames" stopped it.
    pub result: Result<StereoVoBaRefinement, StereoVoBaError>,
}

/// Sliding-window stereo VO + BA composition.
///
/// Owns the frontend and a history of triggered BA stats. After each
/// `process_pair_with_matches` call the wrapper checks whether
/// `trigger_every_frames` boundary has been hit and, if so, runs BA
/// over the trailing `window_size` frames and writes refined poses
/// back into the frontend.
pub struct OnlineStereoVoBa<
    E = visloc_vision::features::CornerFeatureExtractor,
    M = visloc_vision::matching::BruteForceMatcher,
> where
    E: FeatureExtractor<Image = GrayscaleImage>,
    E::Error: std::error::Error + Send + Sync + 'static,
    M: Matcher,
{
    /// The underlying classical/learned stereo VO frontend.
    pub frontend: StereoVoFrontend<E, M>,
    /// Configuration for online BA triggering.
    pub config: OnlineStereoVoBaConfig,
    /// Per-trigger BA history (one entry per fired trigger).
    pub trigger_history: Vec<OnlineBaTriggerStats>,
}

impl<E, M> OnlineStereoVoBa<E, M>
where
    E: FeatureExtractor<Image = GrayscaleImage>,
    E::Error: std::error::Error + Send + Sync + 'static,
    M: Matcher,
{
    pub fn new(frontend: StereoVoFrontend<E, M>, config: OnlineStereoVoBaConfig) -> Self {
        Self {
            frontend,
            config,
            trigger_history: Vec::new(),
        }
    }

    /// Process a stereo pair (caller-provided features + optional
    /// matches) through the frontend and, if the trigger boundary has
    /// been hit, run BA over the trailing window.
    pub fn process_pair_with_matches(
        &mut self,
        left_features: FeatureSet,
        right_features: FeatureSet,
        stereo_matches: Option<&[DescriptorMatch]>,
        temporal_matches: Option<&[DescriptorMatch]>,
    ) -> Result<Pose, StereoVoError> {
        let pose = self.frontend.process_feature_pair_with_matches(
            left_features,
            right_features,
            stereo_matches,
            temporal_matches,
        )?;
        self.maybe_trigger_ba();
        Ok(pose)
    }

    /// Run BA immediately over the trailing window (or the whole
    /// trajectory if fewer than `window_size` frames have been
    /// processed). Returns `None` when the trajectory is too short for
    /// any BA window at all.
    pub fn run_ba_now(&mut self) -> Option<OnlineBaTriggerStats> {
        let n = self.frontend.poses.len();
        if n < 2 {
            return None;
        }
        let window = self.config.window_size.max(2);
        let window = window.min(n);
        let start = n - window;
        let end = n;
        if end - start < self.config.ba_config.min_track_length {
            return None;
        }
        let stats = self.run_ba_window(start, end);
        let cloned = stats.clone();
        self.trigger_history.push(stats);
        Some(cloned)
    }

    fn maybe_trigger_ba(&mut self) {
        if self.config.trigger_every_frames == 0 {
            return;
        }
        let n = self.frontend.poses.len();
        // Need at least `window_size` frames before the first trigger;
        // require a full window so the BA fix-pose-0 anchor is meaningful.
        if n < self.config.window_size {
            return;
        }
        // Trigger on every `trigger_every_frames`-th *additional* pair
        // beyond the first full window, counting from the end of that
        // first window. This avoids back-to-back triggers and gives a
        // predictable cadence.
        let frames_since_first_window = n - self.config.window_size;
        if frames_since_first_window % self.config.trigger_every_frames != 0 {
            return;
        }
        let start = n - self.config.window_size;
        let end = n;
        let stats = self.run_ba_window(start, end);
        self.trigger_history.push(stats);
    }

    fn run_ba_window(&mut self, start: usize, end: usize) -> OnlineBaTriggerStats {
        // Optional "local map" backward history: extend the window back by
        // `local_map_history` frames and fix that prefix, so long-baseline
        // landmarks persisted from earlier windows anchor the recent poses
        // without re-optimising the older trajectory. Disabled (history = 0)
        // when IMU input is active, since the IMU slice aligns to the trailing
        // window only.
        let history = if self.config.imu_input.is_some() {
            0
        } else {
            self.config.local_map_history.min(start)
        };
        let ext_start = start - history;
        // The frontend's `temporal_matches_per_pair[i]` holds the matches
        // used between poses `i` and `i+1`, so the slice for poses
        // `ext_start..end` is `temporal_matches_per_pair[ext_start..end-1]`.
        // `pair_diagnostics[i]` describes the same pair, so rescued pairs can
        // be blanked index-aligned.
        let temporal_owned: Option<Vec<Vec<DescriptorMatch>>> =
            if self.config.exclude_rescued_pair_matches {
                Some(
                    (ext_start..end - 1)
                        .map(|i| {
                            let d = &self.frontend.pair_diagnostics[i];
                            if d.translation_direction_rescued
                                || d.rotation_spike_rescued
                                || d.rotation_vector_rescued
                            {
                                Vec::new()
                            } else {
                                self.frontend.temporal_matches_per_pair[i].clone()
                            }
                        })
                        .collect(),
                )
            } else {
                None
            };
        let temporal_slice: &[Vec<DescriptorMatch>] = match &temporal_owned {
            Some(owned) => owned,
            None => &self.frontend.temporal_matches_per_pair[ext_start..end - 1],
        };

        // Build the effective BA config for this trigger. The wrapper
        // owns the global IMU input and slices it to align with the
        // trailing window; the inner refiner sees a window-local
        // `imu_input` whose `windows.len() == end - start - 1`.
        let mut effective_config = self.config.ba_config.clone();
        // Fix the backward-history prefix (or just pose 0 when history == 0).
        effective_config.fix_pose_prefix = history.max(1);
        if self.config.imu_input.is_some() && effective_config.imu_input.is_some() {
            return OnlineBaTriggerStats {
                window_start: start,
                window_end: end,
                result: Err(StereoVoBaError::InvalidImuInput {
                    reason: "OnlineStereoVoBaConfig.imu_input and \
                             OnlineStereoVoBaConfig.ba_config.imu_input are both set; the wrapper \
                             cannot decide which one to slice. Move the IMU input to the wrapper \
                             level and leave ba_config.imu_input = None."
                        .to_string(),
                }),
            };
        }
        if let Some(global) = &self.config.imu_input {
            let expected = end - start - 1;
            if global.windows.len() < end - 1 {
                return OnlineBaTriggerStats {
                    window_start: start,
                    window_end: end,
                    result: Err(StereoVoBaError::InvalidImuInput {
                        reason: format!(
                            "imu_input.windows.len()={} does not cover the trailing BA window \
                             [{start},{end}); need at least {} windows total",
                            global.windows.len(),
                            end - 1
                        ),
                    }),
                };
            }
            let sliced_windows = global.windows[start..start + expected].to_vec();
            effective_config.imu_input = Some(StereoVoBaImuInput {
                windows: sliced_windows,
                gravity_world: global.gravity_world,
                bias_gyro_init: global.bias_gyro_init,
                bias_acc_init: global.bias_acc_init,
                weight_position: global.weight_position,
                weight_velocity: global.weight_velocity,
                weight_rotation: global.weight_rotation,
                bias_random_walk_weight: global.bias_random_walk_weight,
                fix_first_bias: global.fix_first_bias,
                fix_first_velocity: global.fix_first_velocity,
            });
        }

        // Refine over the (optionally history-extended) window.
        let result = refine_stereo_vo_with_ba(
            &self.frontend.camera,
            self.frontend.baseline,
            &self.frontend.poses[ext_start..end],
            &self.frontend.left_features[ext_start..end],
            &self.frontend.right_features[ext_start..end],
            &self.frontend.stereo_per_frame[ext_start..end],
            temporal_slice,
            &effective_config,
        );
        if let Ok(refinement) = &result {
            // Write back only the trailing (optimised) window; the fixed history
            // prefix [ext_start, start) is unchanged by construction.
            for frame in start..end {
                self.frontend.poses[frame] = refinement.refined_poses[frame - ext_start].clone();
            }
        }
        OnlineBaTriggerStats {
            window_start: ext_start,
            window_end: end,
            result,
        }
    }
}

/// Per-(trigger, keyframe) summary of the streaming IMU refinement
/// state — one entry per row produced by
/// [`write_online_ba_imu_state_csv`]. Exposed so callers that want to
/// post-process the same data programmatically don't have to re-parse
/// the CSV.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineBaImuStateRow {
    /// Zero-based index of the trigger inside
    /// [`OnlineStereoVoBa::trigger_history`].
    pub trigger_idx: usize,
    /// First (fixed-anchor) frame id of the refined window.
    pub window_start: usize,
    /// Exclusive last frame id of the refined window.
    pub window_end: usize,
    /// Offset of this row's keyframe inside the window, so the absolute
    /// frame id is `window_start + window_kf_offset`.
    pub window_kf_offset: usize,
    /// World-frame velocity at the keyframe after BA refinement (m/s).
    pub velocity: nalgebra::Vector3<f64>,
    /// Gyro bias at the keyframe after BA refinement (rad/s).
    pub bias_gyro: nalgebra::Vector3<f64>,
    /// Accelerometer bias at the keyframe after BA refinement (m/s²).
    pub bias_acc: nalgebra::Vector3<f64>,
}

/// Flatten a streaming trigger history into the per-(trigger, keyframe)
/// IMU-state rows that [`write_online_ba_imu_state_csv`] writes out.
/// Triggers whose [`OnlineBaTriggerStats::result`] is `Err` or whose
/// [`StereoVoBaRefinement::imu_refinement`] is `None` (i.e. the trigger
/// ran visual-only) contribute no rows. Exposed for tests and callers
/// that need the in-memory representation.
pub fn online_ba_imu_state_rows(
    trigger_history: &[OnlineBaTriggerStats],
) -> Vec<OnlineBaImuStateRow> {
    let mut rows = Vec::new();
    for (trigger_idx, stats) in trigger_history.iter().enumerate() {
        let Ok(refinement) = stats.result.as_ref() else {
            continue;
        };
        let Some(imu) = refinement.imu_refinement.as_ref() else {
            continue;
        };
        let n = imu
            .refined_velocities
            .len()
            .min(imu.refined_bias_gyro.len())
            .min(imu.refined_bias_acc.len());
        for kf in 0..n {
            rows.push(OnlineBaImuStateRow {
                trigger_idx,
                window_start: stats.window_start,
                window_end: stats.window_end,
                window_kf_offset: kf,
                velocity: imu.refined_velocities[kf],
                bias_gyro: imu.refined_bias_gyro[kf],
                bias_acc: imu.refined_bias_acc[kf],
            });
        }
    }
    rows
}

/// Write the per-(trigger, keyframe) IMU refinement state from a
/// streaming BA run to a CSV file. One row per keyframe inside each
/// trigger's window; triggers without an `imu_refinement` (visual-only,
/// or refiner `Err`) contribute nothing. Returns the number of data
/// rows written (header line excluded).
///
/// Schema (first line is a header):
///
/// ```text
/// trigger_idx,window_start,window_end,window_kf_offset,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz
/// ```
///
/// The absolute frame id for a row is `window_start + window_kf_offset`.
/// Velocity is in m/s in the world frame; gyro bias is in rad/s; accel
/// bias is in m/s². Numeric values are written with Rust's default
/// `f64` Display formatting (full round-trip precision).
pub fn write_online_ba_imu_state_csv(
    path: impl AsRef<Path>,
    trigger_history: &[OnlineBaTriggerStats],
) -> std::io::Result<usize> {
    let mut file = std::fs::File::create(path)?;
    writeln!(
        file,
        "trigger_idx,window_start,window_end,window_kf_offset,\
         vx,vy,vz,bgx,bgy,bgz,bax,bay,baz"
    )?;
    let rows = online_ba_imu_state_rows(trigger_history);
    for row in &rows {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            row.trigger_idx,
            row.window_start,
            row.window_end,
            row.window_kf_offset,
            row.velocity.x,
            row.velocity.y,
            row.velocity.z,
            row.bias_gyro.x,
            row.bias_gyro.y,
            row.bias_gyro.z,
            row.bias_acc.x,
            row.bias_acc.y,
            row.bias_acc.z,
        )?;
    }
    Ok(rows.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
    use visloc_core::geometry::SE3;
    use visloc_core::types::Camera;
    use visloc_vision::features::{CornerFeatureConfig, CornerFeatureExtractor, FeatureSet};
    use visloc_vision::matching::{BruteForceMatcher, DescriptorMatch};
    use visloc_vision::stereo_vo::StereoVoFrontendConfig;

    fn descriptor_match(query: usize, train: usize, confidence: f32) -> DescriptorMatch {
        DescriptorMatch {
            query_index: query,
            train_index: train,
            distance: 0.0,
            second_best_distance: None,
            ratio: None,
            confidence: Some(confidence),
        }
    }

    fn synthetic_camera(width: usize, height: usize) -> Camera {
        Camera::pinhole(
            0,
            width as u32,
            height as u32,
            500.0,
            500.0,
            width as f64 / 2.0,
            height as f64 / 2.0,
        )
    }

    /// Project a world point through `cam_to_world.inverse()` into the
    /// rectified left/right cameras and return the matched stereo
    /// observation `(u_l, v_l, u_r)`. Returns `None` for points behind
    /// the camera.
    fn project_stereo(
        camera: &Camera,
        baseline: f64,
        cam_pose_world_to_cam: &SE3,
        point_world: Point3<f64>,
    ) -> Option<(f64, f64, f64)> {
        let p_cam = cam_pose_world_to_cam.transform_point(&point_world);
        if p_cam.z <= 1e-3 {
            return None;
        }
        let (fx, fy, cx, cy) = camera.intrinsics()?;
        let u_l = fx * p_cam.x / p_cam.z + cx;
        let v_l = fy * p_cam.y / p_cam.z + cy;
        let u_r = fx * (p_cam.x - baseline) / p_cam.z + cx;
        Some((u_l, v_l, u_r))
    }

    /// Build a forward-translating stereo sequence with N world
    /// landmarks. Each frame translates +z by `step` metres in world.
    /// Features are emitted in *world-landmark order* across frames so
    /// stereo and temporal matches are trivial identity maps.
    // (camera, baseline, left features, right features, stereo matches per
    // frame, temporal matches per pair) for one synthetic forward-motion run.
    type SyntheticStereoSequence = (
        Camera,
        f64,
        Vec<FeatureSet>,
        Vec<FeatureSet>,
        Vec<Vec<DescriptorMatch>>,
        Vec<Vec<DescriptorMatch>>,
    );

    fn build_synthetic_sequence(n_frames: usize, step: f64) -> SyntheticStereoSequence {
        let camera = synthetic_camera(640, 480);
        let baseline = 0.5;
        // 24 landmarks scattered in a +z half-space.
        let mut landmarks = Vec::new();
        for ix in -2i32..=2 {
            for iz in 1..=5 {
                let x = ix as f64 * 1.2;
                let y = (ix.abs() as f64) * 0.2 - 0.4;
                let z = 6.0 + iz as f64 * 1.5;
                landmarks.push(Point3::new(x, y, z));
            }
        }

        let mut left_features = Vec::with_capacity(n_frames);
        let mut right_features = Vec::with_capacity(n_frames);
        let mut stereo_matches = Vec::with_capacity(n_frames);
        let mut temporal_matches = Vec::with_capacity(n_frames - 1);

        for frame in 0..n_frames {
            let cam_translation = Vector3::new(0.0, 0.0, step * frame as f64);
            let world_to_cam = SE3 {
                rotation: UnitQuaternion::identity(),
                translation: -cam_translation, // world->cam: subtract camera position
            };
            let mut left_kps = Vec::new();
            let mut right_kps = Vec::new();
            let mut left_desc = Vec::new();
            let mut right_desc = Vec::new();
            let mut stereo_pairs = Vec::new();
            for (lid, lp) in landmarks.iter().enumerate() {
                if let Some((u_l, v_l, u_r)) = project_stereo(&camera, baseline, &world_to_cam, *lp)
                {
                    let li = left_kps.len();
                    let ri = right_kps.len();
                    left_kps.push(Point2::new(u_l, v_l));
                    right_kps.push(Point2::new(u_r, v_l));
                    // Use a 4-d descriptor encoding the landmark id so brute force
                    // matching is deterministic when needed.
                    let d = vec![lid as f32, lp.x as f32, lp.y as f32, lp.z as f32];
                    left_desc.push(d.clone());
                    right_desc.push(d);
                    stereo_pairs.push(descriptor_match(li, ri, 1.0));
                }
            }
            left_features.push(FeatureSet::new(left_kps, left_desc).unwrap());
            right_features.push(FeatureSet::new(right_kps, right_desc).unwrap());
            stereo_matches.push(stereo_pairs);
            if frame > 0 {
                // Identity temporal mapping since indices coincide across frames
                // for any landmark that remains in view.
                let prev = &left_features[frame - 1];
                let curr = &left_features[frame];
                let n = prev.len().min(curr.len());
                let mut pair = Vec::with_capacity(n);
                for i in 0..n {
                    pair.push(descriptor_match(i, i, 1.0));
                }
                temporal_matches.push(pair);
            }
        }
        (
            camera,
            baseline,
            left_features,
            right_features,
            stereo_matches,
            temporal_matches,
        )
    }

    #[test]
    fn online_ba_keeps_clean_synthetic_trajectory_stable() {
        let (camera, baseline, left, right, stereo, temporal) = build_synthetic_sequence(15, 0.4);
        let extractor = CornerFeatureExtractor::new(CornerFeatureConfig::default());
        let matcher = BruteForceMatcher { ratio: Some(0.8) };
        let frontend = StereoVoFrontend::new_with(
            camera,
            baseline,
            StereoVoFrontendConfig::default(),
            extractor,
            matcher,
        );
        let mut online = OnlineStereoVoBa::new(
            frontend,
            OnlineStereoVoBaConfig {
                trigger_every_frames: 3,
                window_size: 5,
                ba_config: StereoVoBaConfig::default(),
                imu_input: None,
                local_map_history: 0,
                exclude_rescued_pair_matches: false,
            },
        );

        for i in 0..left.len() {
            let temp = if i == 0 {
                None
            } else {
                Some(temporal[i - 1].as_slice())
            };
            online
                .process_pair_with_matches(
                    left[i].clone(),
                    right[i].clone(),
                    Some(stereo[i].as_slice()),
                    temp,
                )
                .expect("frontend processes synthetic pair");
        }

        // Final pose should be near +z * (n-1) * step.
        let n = online.frontend.poses.len();
        let last = &online.frontend.poses[n - 1].world_to_camera;
        // World-to-camera translation should be approximately -(0, 0, 5.6).
        let expected_z = -(left.len() as f64 - 1.0) * 0.4;
        assert!(
            (last.translation.z - expected_z).abs() < 0.05,
            "expected z {} got {}",
            expected_z,
            last.translation.z
        );
        // At least one trigger should have fired with `window_size = 5`
        // and `trigger_every_frames = 3` over 15 frames.
        assert!(!online.trigger_history.is_empty());
    }

    #[test]
    fn online_ba_disabled_when_trigger_zero() {
        let (camera, baseline, left, right, stereo, temporal) = build_synthetic_sequence(8, 0.4);
        let extractor = CornerFeatureExtractor::new(CornerFeatureConfig::default());
        let matcher = BruteForceMatcher { ratio: Some(0.8) };
        let frontend = StereoVoFrontend::new_with(
            camera,
            baseline,
            StereoVoFrontendConfig::default(),
            extractor,
            matcher,
        );
        let mut online = OnlineStereoVoBa::new(
            frontend,
            OnlineStereoVoBaConfig {
                trigger_every_frames: 0,
                window_size: 4,
                ba_config: StereoVoBaConfig::default(),
                imu_input: None,
                local_map_history: 0,
                exclude_rescued_pair_matches: false,
            },
        );
        for i in 0..left.len() {
            let temp = if i == 0 {
                None
            } else {
                Some(temporal[i - 1].as_slice())
            };
            online
                .process_pair_with_matches(
                    left[i].clone(),
                    right[i].clone(),
                    Some(stereo[i].as_slice()),
                    temp,
                )
                .expect("frontend processes synthetic pair");
        }
        assert!(online.trigger_history.is_empty());
        // Manual trigger still works.
        let _ = online.run_ba_now();
        assert_eq!(online.trigger_history.len(), 1);
    }

    fn run_online_ba_with_imu(
        n_frames: usize,
        step: f64,
        imu_input: Option<StereoVoBaImuInput>,
    ) -> OnlineStereoVoBa<CornerFeatureExtractor, BruteForceMatcher> {
        let (camera, baseline, left, right, stereo, temporal) =
            build_synthetic_sequence(n_frames, step);
        let extractor = CornerFeatureExtractor::new(CornerFeatureConfig::default());
        let matcher = BruteForceMatcher { ratio: Some(0.8) };
        let frontend = StereoVoFrontend::new_with(
            camera,
            baseline,
            StereoVoFrontendConfig::default(),
            extractor,
            matcher,
        );
        let mut online = OnlineStereoVoBa::new(
            frontend,
            OnlineStereoVoBaConfig {
                trigger_every_frames: 3,
                window_size: 5,
                ba_config: StereoVoBaConfig::default(),
                imu_input,
                local_map_history: 0,
                exclude_rescued_pair_matches: false,
            },
        );
        for i in 0..left.len() {
            let temp = if i == 0 {
                None
            } else {
                Some(temporal[i - 1].as_slice())
            };
            online
                .process_pair_with_matches(
                    left[i].clone(),
                    right[i].clone(),
                    Some(stereo[i].as_slice()),
                    temp,
                )
                .expect("frontend processes synthetic pair");
        }
        online
    }

    /// `imu_input` on the wrapper is sliced to align with the trailing
    /// BA window every trigger. On a 15-frame +z constant-velocity scene
    /// the slicer feeds zero-accel / zero-gyro samples to each trigger;
    /// the refiner should converge with `imu_refinement` populated and
    /// the per-keyframe velocity within `0.05 m/s` of truth.
    #[test]
    fn online_ba_with_imu_input_refines_velocities() {
        // 15 frames at step = 0.4 m, dt_frame = 1 s → world velocity (0, 0, 0.4).
        let step = 0.4_f64;
        let dt_frame = 1.0_f64;
        let n_frames = 15usize;
        let zero = Vector3::<f64>::zeros();
        // 14 inter-frame windows; each window: one sample of dt = dt_frame, zero accel/gyro.
        let windows: Vec<Vec<crate::StereoVoBaImuSample>> = (0..n_frames - 1)
            .map(|_| {
                vec![crate::StereoVoBaImuSample {
                    dt: dt_frame,
                    gyro: zero,
                    accel: zero,
                }]
            })
            .collect();
        let imu_input = StereoVoBaImuInput::new(windows, Vector3::zeros(), 1.0, 1.0, 1.0);
        let online = run_online_ba_with_imu(n_frames, step, Some(imu_input));

        // With window_size=5 and trigger_every=3 over 15 frames, at least one
        // trigger should fire and succeed.
        assert!(
            !online.trigger_history.is_empty(),
            "expected at least one trigger"
        );
        let ok_triggers: Vec<&OnlineBaTriggerStats> = online
            .trigger_history
            .iter()
            .filter(|s| s.result.is_ok())
            .collect();
        assert!(
            !ok_triggers.is_empty(),
            "expected at least one successful trigger; got {:?}",
            online
                .trigger_history
                .iter()
                .map(|s| match &s.result {
                    Ok(_) => "ok".to_string(),
                    Err(e) => format!("err: {e}"),
                })
                .collect::<Vec<_>>(),
        );
        let last = ok_triggers
            .last()
            .expect("at least one ok trigger after filter");
        let imu = last
            .result
            .as_ref()
            .unwrap()
            .imu_refinement
            .as_ref()
            .expect("imu_refinement should be populated when imu_input is set");
        let window_len = last.window_end - last.window_start;
        assert_eq!(imu.refined_velocities.len(), window_len);
        // Every keyframe in the triggered window should recover the +z velocity.
        let truth_v = Vector3::new(0.0, 0.0, step / dt_frame);
        for (i, v) in imu.refined_velocities.iter().enumerate() {
            let err = (v - truth_v).norm();
            assert!(
                err < 0.05,
                "trailing-window keyframe {i} velocity {v:?} diverged from truth {truth_v:?}; err={err:.4}",
            );
        }
    }

    /// When the wrapper-level `imu_input.windows` is too short to cover
    /// the trailing BA window, the trigger should fail with a structured
    /// `InvalidImuInput` error instead of panicking.
    #[test]
    fn online_ba_imu_input_too_short_returns_invalid_imu_input() {
        let step = 0.4_f64;
        let n_frames = 15usize;
        let zero = Vector3::<f64>::zeros();
        // Only 3 windows but the wrapper will try to slice a 5-frame trailing
        // window (needs 4 windows worth of IMU on each trigger).
        let windows: Vec<Vec<crate::StereoVoBaImuSample>> = (0..3)
            .map(|_| {
                vec![crate::StereoVoBaImuSample {
                    dt: 1.0,
                    gyro: zero,
                    accel: zero,
                }]
            })
            .collect();
        let imu_input = StereoVoBaImuInput::new(windows, Vector3::zeros(), 1.0, 1.0, 1.0);
        let online = run_online_ba_with_imu(n_frames, step, Some(imu_input));

        assert!(!online.trigger_history.is_empty());
        let saw_invalid = online
            .trigger_history
            .iter()
            .any(|s| matches!(&s.result, Err(StereoVoBaError::InvalidImuInput { .. }),));
        assert!(
            saw_invalid,
            "expected at least one InvalidImuInput error; got {:?}",
            online
                .trigger_history
                .iter()
                .map(|s| match &s.result {
                    Ok(_) => "ok".to_string(),
                    Err(e) => format!("err: {e}"),
                })
                .collect::<Vec<_>>(),
        );
    }

    fn tempdir_for(label: &str) -> std::path::PathBuf {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "visloc_{}_{}_{}",
            label,
            std::process::id(),
            suffix
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// `write_online_ba_imu_state_csv` should produce one header line plus
    /// one row per (successful-trigger, in-window keyframe). The numeric
    /// columns must round-trip through Rust's f64 Display formatting.
    #[test]
    fn write_online_ba_imu_state_csv_emits_one_row_per_keyframe() {
        let step = 0.4_f64;
        let dt_frame = 1.0_f64;
        let n_frames = 15usize;
        let zero = Vector3::<f64>::zeros();
        let windows: Vec<Vec<crate::StereoVoBaImuSample>> = (0..n_frames - 1)
            .map(|_| {
                vec![crate::StereoVoBaImuSample {
                    dt: dt_frame,
                    gyro: zero,
                    accel: zero,
                }]
            })
            .collect();
        let imu_input = StereoVoBaImuInput::new(windows, Vector3::zeros(), 1.0, 1.0, 1.0);
        let online = run_online_ba_with_imu(n_frames, step, Some(imu_input));

        let rows = online_ba_imu_state_rows(&online.trigger_history);
        assert!(
            !rows.is_empty(),
            "expected at least one successful imu-refinement trigger"
        );
        // Every row's window_kf_offset must stay inside its window range.
        for row in &rows {
            assert!(row.window_start + row.window_kf_offset < row.window_end);
        }

        let dir = tempdir_for("online_ba_imu_csv");
        let csv_path = dir.join("online_ba_imu_state.csv");
        let written = write_online_ba_imu_state_csv(&csv_path, &online.trigger_history).unwrap();
        assert_eq!(written, rows.len());

        let body = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), rows.len() + 1, "header + one row per entry");
        assert_eq!(
            lines[0],
            "trigger_idx,window_start,window_end,window_kf_offset,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz"
        );
        // First data line should mirror the first row entry.
        let first = &rows[0];
        let expected_first = format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{}",
            first.trigger_idx,
            first.window_start,
            first.window_end,
            first.window_kf_offset,
            first.velocity.x,
            first.velocity.y,
            first.velocity.z,
            first.bias_gyro.x,
            first.bias_gyro.y,
            first.bias_gyro.z,
            first.bias_acc.x,
            first.bias_acc.y,
            first.bias_acc.z,
        );
        assert_eq!(lines[1], expected_first);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Triggers without an `imu_refinement` (visual-only or `Err`) must
    /// contribute zero data rows — the file should hold the header line
    /// only.
    #[test]
    fn write_online_ba_imu_state_csv_skips_triggers_without_imu_refinement() {
        // Visual-only path: no `imu_input` on the wrapper, so every
        // successful trigger has `imu_refinement = None`.
        let online = run_online_ba_with_imu(15, 0.4, None);
        assert!(
            !online.trigger_history.is_empty(),
            "wrapper should have produced at least one trigger"
        );
        let rows = online_ba_imu_state_rows(&online.trigger_history);
        assert!(rows.is_empty(), "visual-only triggers must not emit rows");

        let dir = tempdir_for("online_ba_imu_csv_empty");
        let csv_path = dir.join("online_ba_imu_state.csv");
        let written = write_online_ba_imu_state_csv(&csv_path, &online.trigger_history).unwrap();
        assert_eq!(written, 0);
        let body = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 1, "header line only");
        std::fs::remove_dir_all(&dir).ok();
    }
}
