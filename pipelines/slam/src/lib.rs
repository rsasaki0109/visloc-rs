#![forbid(unsafe_code)]
//! Minimal online SLAM orchestration.
//!
//! This crate wires tracking and local mapping together. It is not a full SLAM
//! system: it can report lightweight loop-closure candidates, but global pose
//! graph optimization, dense mapping, and production bundle adjustment remain
//! outside this MVP layer.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

use visloc_core::types::{Frame, Keyframe, Observation, VisualMap};
use visloc_localization::LocalizationPipeline;
use visloc_mapping::{
    AppliedMapUpdate, KeyframePolicy, LandmarkCandidate, LinearTriangulator, LocalMappingPipeline,
    LocalMappingResult, SimpleKeyframePolicy, Triangulator,
};
use visloc_tracking::{
    ConstantPoseMotionModel, FrameLocalizer, MotionModel, Tracker, TrackingConfig, TrackingResult,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnlineSlamConfig {
    pub apply_map_updates: bool,
    pub loop_closure: LoopClosureConfig,
}

impl Default for OnlineSlamConfig {
    fn default() -> Self {
        Self {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopClosureConfig {
    pub enabled: bool,
    pub min_frame_id_gap: u64,
    pub min_shared_landmarks: usize,
    pub min_shared_landmark_ratio_percent: u8,
    pub max_candidates: usize,
}

impl Default for LoopClosureConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_frame_id_gap: 5,
            min_shared_landmarks: 12,
            min_shared_landmark_ratio_percent: 40,
            max_candidates: 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopClosureCandidate {
    pub query_frame_id: u64,
    pub matched_keyframe_id: u64,
    pub shared_landmark_count: usize,
    pub query_inlier_count: usize,
    pub keyframe_observation_count: usize,
    pub shared_landmark_ratio: f64,
    pub score: f64,
    pub geometrically_verified: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamPipeline<T, M> {
    pub map: VisualMap,
    pub tracker: T,
    pub mapper: M,
    pub config: OnlineSlamConfig,
}

impl Default
    for OnlineSlamPipeline<
        Tracker<LocalizationPipeline, ConstantPoseMotionModel>,
        LocalMappingPipeline<SimpleKeyframePolicy, LinearTriangulator>,
    >
{
    fn default() -> Self {
        Self {
            map: VisualMap::new(),
            tracker: Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            mapper: LocalMappingPipeline::default(),
            config: OnlineSlamConfig::default(),
        }
    }
}

impl<T, M> OnlineSlamPipeline<T, M> {
    pub fn new(map: VisualMap, tracker: T, mapper: M, config: OnlineSlamConfig) -> Self {
        Self {
            map,
            tracker,
            mapper,
            config,
        }
    }

    pub fn map(&self) -> &VisualMap {
        &self.map
    }

    pub fn map_mut(&mut self) -> &mut VisualMap {
        &mut self.map
    }
}

impl<P, Motion, K, Tri> OnlineSlamPipeline<Tracker<P, Motion>, LocalMappingPipeline<K, Tri>>
where
    P: FrameLocalizer,
    Motion: MotionModel,
    K: KeyframePolicy,
    Tri: Triangulator,
{
    pub fn process_frame<I>(&mut self, frame: &Frame, candidates: I) -> OnlineSlamResult
    where
        I: IntoIterator<Item = LandmarkCandidate>,
    {
        let tracking = self.tracker.track_frame(frame, &self.map);
        let mut mapping = None;
        let mut applied_update = None;
        let loop_closure_candidates =
            detect_loop_closure_candidates(frame, &tracking, &self.map, &self.config.loop_closure);

        if tracking.localization.success {
            let keyframe = keyframe_from_tracking_result(frame, &tracking);
            let mapping_result = self
                .mapper
                .process_keyframe(&self.map, &tracking, keyframe, candidates);
            if self.config.apply_map_updates && mapping_result.staged_update_validation.is_valid() {
                if let Ok(applied) = mapping_result.staged_update.clone().apply_to(&mut self.map) {
                    applied_update = Some(applied);
                }
            }
            mapping = Some(mapping_result);
        }

        OnlineSlamResult {
            tracking,
            mapping,
            applied_update,
            loop_closure_candidates,
            map_keyframe_count: self.map.keyframes.len(),
            map_landmark_count: self.map.landmarks.len(),
        }
    }

    pub fn reset_sequence_state(&mut self) {
        self.tracker.reset();
        self.mapper.reset();
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamResult {
    pub tracking: TrackingResult,
    pub mapping: Option<LocalMappingResult>,
    pub applied_update: Option<AppliedMapUpdate>,
    pub loop_closure_candidates: Vec<LoopClosureCandidate>,
    pub map_keyframe_count: usize,
    pub map_landmark_count: usize,
}

impl OnlineSlamResult {
    pub fn tracking_succeeded(&self) -> bool {
        self.tracking.localization.success
    }

    pub fn map_was_updated(&self) -> bool {
        self.applied_update.is_some()
    }

    pub fn has_loop_closure_candidate(&self) -> bool {
        !self.loop_closure_candidates.is_empty()
    }
}

pub fn online_slam_results_to_html_report(results: &[OnlineSlamResult]) -> String {
    let samples = slam_report_samples(results);
    let loop_candidates = results
        .iter()
        .flat_map(|result| result.loop_closure_candidates.iter())
        .collect::<Vec<_>>();
    let mut output = String::new();
    output.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    output.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    output.push_str("<title>visloc-rs online SLAM loop report</title>\n");
    output.push_str("<style>");
    output.push_str(
        "body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f7f9;color:#182026}\
         main{max-width:1120px;margin:0 auto;padding:28px}\
         h1{font-size:24px;margin:0 0 8px}\
         h2{font-size:18px;margin:0 0 10px}\
         .sub{margin:0 0 22px;color:#52616b}\
         .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}\
         .metric{background:white;border:1px solid #dde3ea;border-radius:8px;padding:12px}\
         .label{display:block;font-size:12px;color:#65727e}\
         .value{display:block;font-size:22px;font-weight:700;margin-top:4px}\
         .panel{background:white;border:1px solid #dde3ea;border-radius:8px;padding:16px;margin-top:14px}\
         table{width:100%;border-collapse:collapse;font-size:13px}\
         th,td{text-align:right;border-bottom:1px solid #e7ecf0;padding:6px 8px;vertical-align:top}\
         th:first-child,td:first-child{text-align:left}\
         .ok{color:#198754;font-weight:700}.warn{color:#a15c00;font-weight:700}\
         svg{width:100%;height:auto;display:block}",
    );
    output.push_str("</style>\n</head>\n<body>\n<main>\n");
    output.push_str("<h1>visloc-rs online SLAM loop report</h1>\n");
    output.push_str("<p class=\"sub\">Top-down tracked camera centers with diagnostic loop-closure candidate edges. This report does not imply global pose-graph optimization.</p>\n");
    output.push_str("<section class=\"grid\">\n");
    push_metric_card(&mut output, "Frames", &results.len().to_string());
    push_metric_card(&mut output, "Tracked poses", &samples.len().to_string());
    push_metric_card(
        &mut output,
        "Loop candidates",
        &loop_candidates.len().to_string(),
    );
    push_metric_card(
        &mut output,
        "Final keyframes",
        &results
            .last()
            .map(|result| result.map_keyframe_count.to_string())
            .unwrap_or_else(|| "0".to_string()),
    );
    push_metric_card(
        &mut output,
        "Final landmarks",
        &results
            .last()
            .map(|result| result.map_landmark_count.to_string())
            .unwrap_or_else(|| "0".to_string()),
    );
    output.push_str("</section>\n");
    output.push_str("<section class=\"panel\">\n");
    output.push_str(&online_slam_loop_svg(&samples, &loop_candidates));
    output.push_str("</section>\n");
    output.push_str("<section class=\"panel\">\n<h2>Loop Closure Candidates</h2>\n");
    output.push_str("<table><thead><tr><th>query frame</th><th>matched keyframe</th><th>shared landmarks</th><th>ratio</th><th>score</th><th>verified</th></tr></thead><tbody>\n");
    if loop_candidates.is_empty() {
        output.push_str("<tr><td colspan=\"6\">no loop candidates reported</td></tr>\n");
    }
    for candidate in &loop_candidates {
        let verified_class = if candidate.geometrically_verified {
            "ok"
        } else {
            "warn"
        };
        let verified_text = if candidate.geometrically_verified {
            "yes"
        } else {
            "candidate"
        };
        let _ = writeln!(
            output,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td class=\"{}\">{}</td></tr>",
            candidate.query_frame_id,
            candidate.matched_keyframe_id,
            candidate.shared_landmark_count,
            candidate.shared_landmark_ratio,
            candidate.score,
            verified_class,
            verified_text,
        );
    }
    output.push_str("</tbody></table>\n</section>\n</main>\n</body>\n</html>\n");
    output
}

pub fn write_online_slam_results_html_report(
    results: &[OnlineSlamResult],
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    std::fs::write(path, online_slam_results_to_html_report(results))
}

fn keyframe_from_tracking_result(frame: &Frame, tracking: &TrackingResult) -> Keyframe {
    let mut frame = frame.clone();
    frame.pose = tracking.localization.pose.clone();

    let observations = tracking
        .localization
        .inlier_query_indices
        .iter()
        .zip(tracking.localization.inlier_landmark_ids.iter())
        .filter_map(|(keypoint_index, landmark_id)| {
            frame.keypoints.get(*keypoint_index).map(|xy| Observation {
                frame_id: frame.id,
                landmark_id: *landmark_id,
                keypoint_index: *keypoint_index,
                xy: *xy,
            })
        })
        .collect();

    Keyframe {
        frame,
        observations,
    }
}

#[derive(Debug, Clone, Copy)]
struct SlamReportSample {
    frame_id: u64,
    x: f64,
    y: f64,
    z: f64,
}

fn slam_report_samples(results: &[OnlineSlamResult]) -> Vec<SlamReportSample> {
    results
        .iter()
        .filter_map(|result| {
            let pose = result.tracking.localization.pose.as_ref()?;
            let center = pose.camera_center_world();
            Some(SlamReportSample {
                frame_id: result.tracking.frame_id,
                x: center.x,
                y: center.y,
                z: center.z,
            })
        })
        .collect()
}

fn online_slam_loop_svg(
    samples: &[SlamReportSample],
    candidates: &[&LoopClosureCandidate],
) -> String {
    let projection = SlamReportProjection::from_samples(samples);
    let by_frame_id = samples
        .iter()
        .map(|sample| (sample.frame_id, *sample))
        .collect::<HashMap<_, _>>();
    let mut output = String::new();
    output.push_str("<svg viewBox=\"0 0 900 520\" role=\"img\" aria-label=\"online SLAM loop candidate plot\">\n");
    output.push_str("<rect x=\"0\" y=\"0\" width=\"900\" height=\"520\" fill=\"#fbfcfd\"/>\n");
    output.push_str("<g stroke=\"#e4e9ef\" stroke-width=\"1\">\n");
    for x in [80, 228, 376, 524, 672, 820] {
        let _ = writeln!(output, "<line x1=\"{x}\" y1=\"54\" x2=\"{x}\" y2=\"450\"/>");
    }
    for y in [54, 133, 212, 291, 370, 450] {
        let _ = writeln!(output, "<line x1=\"80\" y1=\"{y}\" x2=\"820\" y2=\"{y}\"/>");
    }
    output.push_str("</g>\n");
    output.push_str("<g fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n");
    if !samples.is_empty() {
        let points = samples
            .iter()
            .map(|sample| {
                let (x, y) = projection.project(sample);
                format!("{x:.2},{y:.2}")
            })
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(
            output,
            "<polyline points=\"{points}\" stroke=\"#2676c9\" stroke-width=\"4\"/>"
        );
    }
    output.push_str("</g>\n");
    output.push_str("<g fill=\"none\" stroke=\"#f0a202\" stroke-width=\"4\" stroke-dasharray=\"10 7\" stroke-linecap=\"round\">\n");
    for candidate in candidates {
        let (Some(query), Some(matched)) = (
            by_frame_id.get(&candidate.query_frame_id),
            by_frame_id.get(&candidate.matched_keyframe_id),
        ) else {
            continue;
        };
        let (qx, qy) = projection.project(query);
        let (mx, my) = projection.project(matched);
        let _ = writeln!(
            output,
            "<line x1=\"{qx:.2}\" y1=\"{qy:.2}\" x2=\"{mx:.2}\" y2=\"{my:.2}\"/>"
        );
    }
    output.push_str("</g>\n<g>\n");
    for sample in samples {
        let (x, y) = projection.project(sample);
        let _ = writeln!(
            output,
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"6\" fill=\"#2676c9\"/>"
        );
        let _ = writeln!(
            output,
            "<text x=\"{:.2}\" y=\"{:.2}\" fill=\"#52616b\" font-size=\"12\" text-anchor=\"middle\">{}</text>",
            x,
            y + 22.0,
            sample.frame_id
        );
    }
    output.push_str("</g>\n");
    output.push_str("<rect x=\"80\" y=\"468\" width=\"14\" height=\"6\" fill=\"#2676c9\"/>\n");
    output.push_str(
        "<text x=\"102\" y=\"476\" fill=\"#52616b\" font-size=\"13\">tracked camera path</text>\n",
    );
    output.push_str("<line x1=\"278\" y1=\"472\" x2=\"320\" y2=\"472\" stroke=\"#f0a202\" stroke-width=\"4\" stroke-dasharray=\"10 7\"/>\n");
    output.push_str(
        "<text x=\"330\" y=\"476\" fill=\"#52616b\" font-size=\"13\">loop candidate edge</text>\n",
    );
    output.push_str("</svg>\n");
    output
}

#[derive(Debug, Clone, Copy)]
struct SlamReportProjection {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    axis_y: usize,
}

impl SlamReportProjection {
    fn from_samples(samples: &[SlamReportSample]) -> Self {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for sample in samples {
            min[0] = min[0].min(sample.x);
            min[1] = min[1].min(sample.y);
            min[2] = min[2].min(sample.z);
            max[0] = max[0].max(sample.x);
            max[1] = max[1].max(sample.y);
            max[2] = max[2].max(sample.z);
        }

        if !min[0].is_finite() {
            return Self {
                min_x: -1.0,
                max_x: 1.0,
                min_y: -1.0,
                max_y: 1.0,
                axis_y: 2,
            };
        }

        let spread_y = max[1] - min[1];
        let spread_z = max[2] - min[2];
        let axis_y = if spread_z >= spread_y { 2 } else { 1 };
        let (mut min_x, mut max_x) = padded_range(min[0], max[0]);
        let (mut min_y, mut max_y) = padded_range(min[axis_y], max[axis_y]);
        let x_span = max_x - min_x;
        let y_span = max_y - min_y;
        if x_span > y_span {
            let delta = (x_span - y_span) * 0.5;
            min_y -= delta;
            max_y += delta;
        } else {
            let delta = (y_span - x_span) * 0.5;
            min_x -= delta;
            max_x += delta;
        }

        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            axis_y,
        }
    }

    fn project(&self, sample: &SlamReportSample) -> (f64, f64) {
        let plot_left = 80.0;
        let plot_top = 54.0;
        let plot_width = 740.0;
        let plot_height = 396.0;
        let horizontal = (sample.x - self.min_x) / (self.max_x - self.min_x);
        let vertical_value = if self.axis_y == 2 { sample.z } else { sample.y };
        let vertical = (vertical_value - self.min_y) / (self.max_y - self.min_y);
        (
            plot_left + horizontal * plot_width,
            plot_top + (1.0 - vertical) * plot_height,
        )
    }
}

fn padded_range(min: f64, max: f64) -> (f64, f64) {
    let span = max - min;
    if span.abs() < 1.0e-12 {
        (min - 1.0, max + 1.0)
    } else {
        let padding = span * 0.08;
        (min - padding, max + padding)
    }
}

fn push_metric_card(output: &mut String, label: &str, value: &str) {
    let _ = writeln!(
        output,
        "<div class=\"metric\"><span class=\"label\">{label}</span><span class=\"value\">{value}</span></div>"
    );
}

fn detect_loop_closure_candidates(
    frame: &Frame,
    tracking: &TrackingResult,
    map: &VisualMap,
    config: &LoopClosureConfig,
) -> Vec<LoopClosureCandidate> {
    if !config.enabled || !tracking.localization.success {
        return Vec::new();
    }

    let query_landmarks = tracking
        .localization
        .inlier_landmark_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    if query_landmarks.is_empty() {
        return Vec::new();
    }

    let mut candidates = map
        .keyframes
        .values()
        .filter_map(|keyframe| {
            if frame.id.abs_diff(keyframe.frame.id) < config.min_frame_id_gap {
                return None;
            }

            let keyframe_landmarks = keyframe
                .observations
                .iter()
                .map(|observation| observation.landmark_id)
                .collect::<HashSet<_>>();
            if keyframe_landmarks.is_empty() {
                return None;
            }

            let shared_landmark_count = query_landmarks.intersection(&keyframe_landmarks).count();
            if shared_landmark_count < config.min_shared_landmarks {
                return None;
            }

            let denominator = query_landmarks.len().min(keyframe_landmarks.len());
            let shared_landmark_ratio = shared_landmark_count as f64 / denominator as f64;
            let required_ratio = f64::from(config.min_shared_landmark_ratio_percent) / 100.0;
            if shared_landmark_ratio < required_ratio {
                return None;
            }

            let score = shared_landmark_ratio * shared_landmark_count as f64;
            Some(LoopClosureCandidate {
                query_frame_id: frame.id,
                matched_keyframe_id: keyframe.frame.id,
                shared_landmark_count,
                query_inlier_count: query_landmarks.len(),
                keyframe_observation_count: keyframe_landmarks.len(),
                shared_landmark_ratio,
                score,
                geometrically_verified: true,
            })
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.shared_landmark_count.cmp(&a.shared_landmark_count))
            .then_with(|| a.matched_keyframe_id.cmp(&b.matched_keyframe_id))
    });
    candidates.truncate(config.max_candidates);
    candidates
}
