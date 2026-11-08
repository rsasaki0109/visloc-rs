//! Opt-in progress diagnostics: temporal track-support histograms and
//! per-registration trace lines.

use super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct TemporalTrackSupportCount {
    pub(super) tracks: usize,
    pub(super) observations: usize,
    pub(super) metric_tracks: usize,
}

const fn temporal_track_span_class(span_frames: usize) -> &'static str {
    match span_frames {
        0 => "same-frame",
        1..=7 => "1-7",
        8..=15 => "8-15",
        16..=31 => "16-31",
        32..=127 => "32-127",
        _ => "128+",
    }
}

pub(super) fn temporal_track_support<I>(
    tracks: I,
    image_assignment: &[(usize, usize)],
    bin_frames: usize,
) -> BTreeMap<(usize, &'static str), TemporalTrackSupportCount>
where
    I: IntoIterator<Item = Vec<usize>>,
{
    let mut counts = BTreeMap::new();
    for images in tracks {
        if images.is_empty() {
            continue;
        }
        let mut assignments = images
            .iter()
            .filter_map(|&image| image_assignment.get(image).copied())
            .collect::<Vec<_>>();
        if assignments.is_empty() {
            continue;
        }
        assignments.sort_unstable();
        let anchor_frame = assignments[assignments.len() / 2].0;
        let span_frames = assignments.last().unwrap().0 - assignments[0].0;
        let metric = assignments
            .windows(2)
            .any(|pair| pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1);
        let count = counts
            .entry((
                anchor_frame / bin_frames,
                temporal_track_span_class(span_frames),
            ))
            .or_insert_with(TemporalTrackSupportCount::default);
        count.tracks += 1;
        count.observations += assignments.len();
        count.metric_tracks += usize::from(metric);
    }
    counts
}

pub(super) fn log_temporal_track_support<I>(
    stage: &str,
    tracks: I,
    image_assignment: &[(usize, usize)],
    bin_frames: usize,
) where
    I: IntoIterator<Item = Vec<usize>>,
{
    for ((frame_bin, span_class), count) in
        temporal_track_support(tracks, image_assignment, bin_frames)
    {
        eprintln!(
            "rig-temporal-support: stage={stage} frame_start={} frame_end_exclusive={} span_class={span_class} tracks={} observations={} metric_tracks={}",
            frame_bin * bin_frames,
            (frame_bin + 1) * bin_frames,
            count.tracks,
            count.observations,
            count.metric_tracks,
        );
    }
}

pub(super) fn temporal_support_bin_frames_from_env() -> Option<usize> {
    let value = std::env::var("VISLOC_SFM_TEMPORAL_SUPPORT_BIN_FRAMES").ok()?;
    match value.parse::<usize>() {
        Ok(value) if value > 0 => Some(value),
        _ => {
            eprintln!(
                "rig-temporal-support: disabled invalid VISLOC_SFM_TEMPORAL_SUPPORT_BIN_FRAMES={value:?}"
            );
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RegistrationTrace {
    pub(super) mode: &'static str,
    pub(super) order: usize,
    pub(super) frame: usize,
    pub(super) support: usize,
    pub(super) inliers: usize,
    pub(super) sensors: usize,
}

pub(super) fn registration_trace_line(trace: RegistrationTrace, pose: &Pose) -> String {
    let center = pose.camera_center_world();
    format!(
        "rig-registration-trace: mode={} order={} frame={} support={} inliers={} sensors={} cx={:.17e} cy={:.17e} cz={:.17e}",
        trace.mode,
        trace.order,
        trace.frame,
        trace.support,
        trace.inliers,
        trace.sensors,
        center.x, center.y, center.z
    )
}

pub(super) fn emit_registration_trace(enabled: bool, trace: RegistrationTrace, pose: &Pose) {
    if enabled {
        eprintln!("{}", registration_trace_line(trace, pose));
    }
}
