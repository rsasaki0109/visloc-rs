//! CSV / SVG / HTML reporting for trajectories and tracking timelines.

use super::*;

pub(crate) fn optional_f64_csv(value: Option<f64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

pub(crate) fn push_metric_card(output: &mut String, label: &str, value: &str) {
    let _ = writeln!(
        output,
        "<div class=\"metric\"><span class=\"label\">{}</span><span class=\"value\">{}</span></div>",
        label, value
    );
}

pub(crate) fn format_optional_metric(value: Option<f64>, unit: &str) -> String {
    value
        .map(|value| format!("{value:.4} {unit}"))
        .unwrap_or_else(|| "n/a".to_string())
}

pub(crate) fn format_optional_count(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.1}"))
        .unwrap_or_else(|| "n/a".to_string())
}

pub(crate) fn format_optional_frame_id(value: Option<FrameId>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "n/a".to_string())
}

pub(crate) fn trajectory_svg(trajectory: &PoseTrajectory) -> String {
    let points = trajectory.camera_centers_world();
    let projection = TrajectorySvgProjection::from_points(&points, &[]);

    let mut output = String::new();
    output.push_str("<svg viewBox=\"0 0 900 520\" role=\"img\" aria-label=\"trajectory plot\">\n");
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
    push_polyline(&mut output, &points, &projection, "#e0574f", 4);
    output.push_str("</g>\n");
    push_points(&mut output, &points, &projection, "#e0574f");
    output.push_str(
        "<text x=\"80\" y=\"486\" fill=\"#65727e\" font-size=\"13\">top-down camera-center trajectory</text>\n",
    );
    output.push_str("</svg>\n");
    output
}

pub(crate) fn trajectory_comparison_svg(
    estimated: &PoseTrajectory,
    reference: &PoseTrajectory,
    alignment: TrajectoryAlignment,
) -> String {
    let reference_by_frame_id = reference
        .samples
        .iter()
        .map(|sample| (sample.frame_id, sample.camera_center_world()))
        .collect::<HashMap<_, _>>();
    let transform = estimated.trajectory_alignment_transform(&reference_by_frame_id, alignment);
    let estimated_points = estimated
        .samples
        .iter()
        .map(|sample| transform.apply(&sample.camera_center_world()))
        .collect::<Vec<_>>();
    let reference_points = reference.camera_centers_world();
    let projection = TrajectorySvgProjection::from_points(&estimated_points, &reference_points);

    let mut output = String::new();
    output.push_str("<svg viewBox=\"0 0 900 520\" role=\"img\" aria-label=\"trajectory plot\">\n");
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
    push_polyline(&mut output, &estimated_points, &projection, "#e0574f", 4);
    push_polyline(&mut output, &reference_points, &projection, "#2676c9", 4);
    output.push_str("</g>\n");

    output.push_str("<g stroke=\"#9aa7b2\" stroke-width=\"1.5\" stroke-dasharray=\"4 5\">\n");
    for sample in &estimated.samples {
        let Some(reference_center) = reference_by_frame_id.get(&sample.frame_id) else {
            continue;
        };
        let estimated_center = transform.apply(&sample.camera_center_world());
        let (ex, ey) = projection.project(&estimated_center);
        let (rx, ry) = projection.project(reference_center);
        let _ = writeln!(
            output,
            "<line x1=\"{ex:.2}\" y1=\"{ey:.2}\" x2=\"{rx:.2}\" y2=\"{ry:.2}\"/>"
        );
    }
    output.push_str("</g>\n");
    push_points(&mut output, &reference_points, &projection, "#2676c9");
    push_points(&mut output, &estimated_points, &projection, "#e0574f");
    output.push_str(
        "<text x=\"80\" y=\"486\" fill=\"#65727e\" font-size=\"13\">top-down camera-center trajectory</text>\n",
    );
    output.push_str("</svg>\n");
    output
}

fn push_polyline(
    output: &mut String,
    points: &[Point3<f64>],
    projection: &TrajectorySvgProjection,
    color: &str,
    stroke_width: usize,
) {
    if points.is_empty() {
        return;
    }

    let mut point_text = String::new();
    for point in points {
        let (x, y) = projection.project(point);
        let _ = write!(point_text, "{x:.2},{y:.2} ");
    }
    let _ = writeln!(
        output,
        "<polyline points=\"{}\" stroke=\"{}\" stroke-width=\"{}\"/>",
        point_text.trim_end(),
        color,
        stroke_width
    );
}

fn push_points(
    output: &mut String,
    points: &[Point3<f64>],
    projection: &TrajectorySvgProjection,
    color: &str,
) {
    output.push_str("<g>\n");
    for point in points {
        let (x, y) = projection.project(point);
        let _ = writeln!(
            output,
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"4\" fill=\"{color}\"/>"
        );
    }
    output.push_str("</g>\n");
}

#[derive(Debug, Clone, Copy)]
struct TrajectorySvgProjection {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    axis_y: usize,
}

impl TrajectorySvgProjection {
    fn from_points(estimated: &[Point3<f64>], reference: &[Point3<f64>]) -> Self {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for point in estimated.iter().chain(reference.iter()) {
            min[0] = min[0].min(point.x);
            min[1] = min[1].min(point.y);
            min[2] = min[2].min(point.z);
            max[0] = max[0].max(point.x);
            max[1] = max[1].max(point.y);
            max[2] = max[2].max(point.z);
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

    fn project(&self, point: &Point3<f64>) -> (f64, f64) {
        let plot_left = 80.0;
        let plot_top = 54.0;
        let plot_width = 740.0;
        let plot_height = 396.0;
        let horizontal = (point.x - self.min_x) / (self.max_x - self.min_x);
        let vertical_value = if self.axis_y == 2 { point.z } else { point.y };
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

pub(crate) fn parse_tum_frame_id(
    value: &str,
    line_number: usize,
    line: &str,
) -> Result<FrameId, TumTrajectoryParseError> {
    value.parse::<FrameId>().map_err(|error| {
        TumTrajectoryParseError::new(line_number, line, format!("invalid frame_id: {error}"))
    })
}

pub(crate) fn parse_tum_f64(
    value: &str,
    field_name: &str,
    line_number: usize,
    line: &str,
) -> Result<f64, TumTrajectoryParseError> {
    value.parse::<f64>().map_err(|error| {
        TumTrajectoryParseError::new(line_number, line, format!("invalid {field_name}: {error}"))
    })
}

pub(crate) fn parse_kitti_f64(
    value: &str,
    field_index: usize,
    line_number: usize,
    line: &str,
) -> Result<f64, KittiTrajectoryParseError> {
    value.parse::<f64>().map_err(|error| {
        KittiTrajectoryParseError::new(
            line_number,
            line,
            format!("invalid field {field_index}: {error}"),
        )
    })
}

pub(crate) fn cumulative_reference_distances(
    matched: &[(&TrajectorySample, &TrajectorySample)],
) -> Vec<f64> {
    let mut distances = Vec::with_capacity(matched.len());
    let mut cumulative = 0.0;
    for (index, (_, reference_sample)) in matched.iter().enumerate() {
        if index > 0 {
            let previous = matched[index - 1].1.camera_center_world();
            let current = reference_sample.camera_center_world();
            cumulative += (current - previous).norm();
        }
        distances.push(cumulative);
    }
    distances
}

pub(crate) fn first_index_for_distance(distances: &[f64], target_distance: f64) -> Option<usize> {
    distances
        .iter()
        .position(|distance| *distance >= target_distance)
}

pub(crate) fn relative_camera_to_world(first: &TrajectorySample, last: &TrajectorySample) -> SE3 {
    let first_camera_to_world = first.pose.world_to_camera.inverse();
    let last_camera_to_world = last.pose.world_to_camera.inverse();
    first_camera_to_world
        .inverse()
        .compose(&last_camera_to_world)
}

pub(crate) fn export_f64(value: f64) -> String {
    if value.abs() < 1.0e-15 {
        "0".to_string()
    } else {
        value.to_string()
    }
}

pub(crate) fn optional_f64_json(value: Option<f64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

pub(crate) fn optional_usize_json(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

pub(crate) fn relative_pose_error_statistics_json(
    stats: Option<RelativePoseErrorStatistics>,
) -> String {
    match stats {
        None => "null".to_string(),
        Some(s) => format!(
            "{{ \"rmse\": {}, \"mean\": {}, \"median\": {}, \"std\": {}, \"min\": {}, \"max\": {} }}",
            s.rmse, s.mean, s.median, s.std, s.min, s.max
        ),
    }
}

pub(crate) fn optional_frame_id_json(value: Option<FrameId>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

pub(crate) fn optional_vec3_json(value: Option<[f64; 3]>) -> String {
    value
        .map(|value| format!("[{}, {}, {}]", value[0], value[1], value[2]))
        .unwrap_or_else(|| "null".to_string())
}

pub(crate) fn trajectory_evaluation_failures_json(
    failures: &[TrajectoryEvaluationFailure],
) -> String {
    let mut output = String::from("[");
    for (index, failure) in failures.iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        output.push_str(&failure.to_json_inline());
    }
    output.push(']');
    output
}

fn tracking_evaluation_failures_json(failures: &[TrackingEvaluationFailure]) -> String {
    let mut output = String::from("[");
    for (index, failure) in failures.iter().enumerate() {
        if index > 0 {
            output.push_str(", ");
        }
        output.push_str(&failure.to_json_inline());
    }
    output.push(']');
    output
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TrackingStats {
    pub first_frame_id: Option<FrameId>,
    pub last_frame_id: Option<FrameId>,
    pub frame_count: usize,
    pub successful_frame_count: usize,
    pub failed_frame_count: usize,
    pub lost_count: usize,
    pub relocalization_count: usize,
    pub pose_prior_used_count: usize,
    pub external_localization_prior_used_count: usize,
    pub tracking_quality_gate_failure_count: usize,
    pub total_inlier_count: usize,
    pub total_correspondence_count: usize,
    pub covisibility_local_map_used_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackingEvaluationConfig {
    pub min_success_rate: Option<f64>,
    pub max_failure_rate: Option<f64>,
    pub max_lost_count: Option<usize>,
    pub max_tracking_quality_gate_failure_count: Option<usize>,
    pub min_external_localization_prior_usage_rate: Option<f64>,
    pub min_overall_inlier_ratio: Option<f64>,
    pub min_mean_inliers_per_successful_frame: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackingEvaluationResult {
    pub passed: bool,
    pub stats: TrackingStats,
    pub config: TrackingEvaluationConfig,
    pub failures: Vec<TrackingEvaluationFailure>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrackingEvaluationFailure {
    SuccessRateTooLow { actual: f64, minimum: f64 },
    FailureRateTooHigh { actual: f64, maximum: f64 },
    LostCountTooHigh { actual: usize, maximum: usize },
    QualityGateFailureCountTooHigh { actual: usize, maximum: usize },
    ExternalLocalizationPriorUsageRateTooLow { actual: f64, minimum: f64 },
    OverallInlierRatioTooLow { actual: f64, minimum: f64 },
    MeanInliersPerSuccessfulFrameTooLow { actual: f64, minimum: f64 },
}

impl TrackingStats {
    pub fn from_results(results: &[TrackingResult]) -> Self {
        let mut stats = Self::default();
        for result in results {
            if stats.first_frame_id.is_none() {
                stats.first_frame_id = Some(result.frame_id);
            }
            stats.last_frame_id = Some(result.frame_id);
            stats.frame_count += 1;
            if result.localization.success {
                stats.successful_frame_count += 1;
            } else {
                stats.failed_frame_count += 1;
            }
            if result.event == TrackingEvent::Lost {
                stats.lost_count += 1;
            }
            if result.event == TrackingEvent::Relocalized {
                stats.relocalization_count += 1;
            }
            if result.used_pose_prior {
                stats.pose_prior_used_count += 1;
            }
            if result.used_external_localization_prior {
                stats.external_localization_prior_used_count += 1;
            }
            if result.tracking_failure_reason.is_some() {
                stats.tracking_quality_gate_failure_count += 1;
            }
            stats.total_inlier_count += result.localization.inlier_count;
            stats.total_correspondence_count += result.localization.correspondence_count;
            if result.covisibility_local_map_size.is_some() {
                stats.covisibility_local_map_used_count += 1;
            }
        }
        stats
    }

    pub fn success_rate(&self) -> f64 {
        ratio(self.successful_frame_count, self.frame_count)
    }

    pub fn failure_rate(&self) -> f64 {
        ratio(self.failed_frame_count, self.frame_count)
    }

    pub fn pose_prior_usage_rate(&self) -> f64 {
        ratio(self.pose_prior_used_count, self.frame_count)
    }

    pub fn external_localization_prior_usage_rate(&self) -> f64 {
        ratio(
            self.external_localization_prior_used_count,
            self.frame_count,
        )
    }

    pub fn overall_inlier_ratio(&self) -> f64 {
        ratio(self.total_inlier_count, self.total_correspondence_count)
    }

    pub fn mean_inliers_per_successful_frame(&self) -> f64 {
        ratio(self.total_inlier_count, self.successful_frame_count)
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"first_frame_id\": {},\n",
                "  \"last_frame_id\": {},\n",
                "  \"frame_count\": {},\n",
                "  \"successful_frame_count\": {},\n",
                "  \"failed_frame_count\": {},\n",
                "  \"lost_count\": {},\n",
                "  \"relocalization_count\": {},\n",
                "  \"pose_prior_used_count\": {},\n",
                "  \"external_localization_prior_used_count\": {},\n",
                "  \"tracking_quality_gate_failure_count\": {},\n",
                "  \"total_inlier_count\": {},\n",
                "  \"total_correspondence_count\": {},\n",
                "  \"success_rate\": {},\n",
                "  \"failure_rate\": {},\n",
                "  \"pose_prior_usage_rate\": {},\n",
                "  \"external_localization_prior_usage_rate\": {},\n",
                "  \"overall_inlier_ratio\": {},\n",
                "  \"mean_inliers_per_successful_frame\": {}\n",
                "}}\n"
            ),
            optional_frame_id_json(self.first_frame_id),
            optional_frame_id_json(self.last_frame_id),
            self.frame_count,
            self.successful_frame_count,
            self.failed_frame_count,
            self.lost_count,
            self.relocalization_count,
            self.pose_prior_used_count,
            self.external_localization_prior_used_count,
            self.tracking_quality_gate_failure_count,
            self.total_inlier_count,
            self.total_correspondence_count,
            self.success_rate(),
            self.failure_rate(),
            self.pose_prior_usage_rate(),
            self.external_localization_prior_usage_rate(),
            self.overall_inlier_ratio(),
            self.mean_inliers_per_successful_frame(),
        )
    }

    fn to_json_inline(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"first_frame_id\": {}, ",
                "\"last_frame_id\": {}, ",
                "\"frame_count\": {}, ",
                "\"successful_frame_count\": {}, ",
                "\"failed_frame_count\": {}, ",
                "\"lost_count\": {}, ",
                "\"relocalization_count\": {}, ",
                "\"pose_prior_used_count\": {}, ",
                "\"external_localization_prior_used_count\": {}, ",
                "\"tracking_quality_gate_failure_count\": {}, ",
                "\"total_inlier_count\": {}, ",
                "\"total_correspondence_count\": {}, ",
                "\"success_rate\": {}, ",
                "\"failure_rate\": {}, ",
                "\"pose_prior_usage_rate\": {}, ",
                "\"external_localization_prior_usage_rate\": {}, ",
                "\"overall_inlier_ratio\": {}, ",
                "\"mean_inliers_per_successful_frame\": {}",
                "}}"
            ),
            optional_frame_id_json(self.first_frame_id),
            optional_frame_id_json(self.last_frame_id),
            self.frame_count,
            self.successful_frame_count,
            self.failed_frame_count,
            self.lost_count,
            self.relocalization_count,
            self.pose_prior_used_count,
            self.external_localization_prior_used_count,
            self.tracking_quality_gate_failure_count,
            self.total_inlier_count,
            self.total_correspondence_count,
            self.success_rate(),
            self.failure_rate(),
            self.pose_prior_usage_rate(),
            self.external_localization_prior_usage_rate(),
            self.overall_inlier_ratio(),
            self.mean_inliers_per_successful_frame(),
        )
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    pub fn evaluate(&self, config: TrackingEvaluationConfig) -> TrackingEvaluationResult {
        TrackingEvaluationResult::from_stats(self.clone(), config)
    }
}

impl TrackingEvaluationResult {
    pub fn from_stats(stats: TrackingStats, config: TrackingEvaluationConfig) -> Self {
        let mut failures = Vec::new();

        if let Some(minimum) = config.min_success_rate {
            let actual = stats.success_rate();
            if actual < minimum {
                failures.push(TrackingEvaluationFailure::SuccessRateTooLow { actual, minimum });
            }
        }

        if let Some(maximum) = config.max_failure_rate {
            let actual = stats.failure_rate();
            if actual > maximum {
                failures.push(TrackingEvaluationFailure::FailureRateTooHigh { actual, maximum });
            }
        }

        if let Some(maximum) = config.max_lost_count {
            if stats.lost_count > maximum {
                failures.push(TrackingEvaluationFailure::LostCountTooHigh {
                    actual: stats.lost_count,
                    maximum,
                });
            }
        }

        if let Some(maximum) = config.max_tracking_quality_gate_failure_count {
            if stats.tracking_quality_gate_failure_count > maximum {
                failures.push(TrackingEvaluationFailure::QualityGateFailureCountTooHigh {
                    actual: stats.tracking_quality_gate_failure_count,
                    maximum,
                });
            }
        }

        if let Some(minimum) = config.min_external_localization_prior_usage_rate {
            let actual = stats.external_localization_prior_usage_rate();
            if actual < minimum {
                failures.push(
                    TrackingEvaluationFailure::ExternalLocalizationPriorUsageRateTooLow {
                        actual,
                        minimum,
                    },
                );
            }
        }

        if let Some(minimum) = config.min_overall_inlier_ratio {
            let actual = stats.overall_inlier_ratio();
            if actual < minimum {
                failures
                    .push(TrackingEvaluationFailure::OverallInlierRatioTooLow { actual, minimum });
            }
        }

        if let Some(minimum) = config.min_mean_inliers_per_successful_frame {
            let actual = stats.mean_inliers_per_successful_frame();
            if actual < minimum {
                failures.push(
                    TrackingEvaluationFailure::MeanInliersPerSuccessfulFrameTooLow {
                        actual,
                        minimum,
                    },
                );
            }
        }

        let passed = failures.is_empty();
        Self {
            passed,
            stats,
            config,
            failures,
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"passed\": {},\n",
                "  \"config\": {},\n",
                "  \"stats\": {},\n",
                "  \"failures\": {}\n",
                "}}\n"
            ),
            self.passed,
            self.config.to_json_inline(),
            self.stats.to_json_inline(),
            tracking_evaluation_failures_json(&self.failures)
        )
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }
}

impl TrackingEvaluationConfig {
    fn to_json_inline(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"min_success_rate\": {}, ",
                "\"max_failure_rate\": {}, ",
                "\"max_lost_count\": {}, ",
                "\"max_tracking_quality_gate_failure_count\": {}, ",
                "\"min_external_localization_prior_usage_rate\": {}, ",
                "\"min_overall_inlier_ratio\": {}, ",
                "\"min_mean_inliers_per_successful_frame\": {}",
                "}}"
            ),
            optional_f64_json(self.min_success_rate),
            optional_f64_json(self.max_failure_rate),
            optional_usize_json(self.max_lost_count),
            optional_usize_json(self.max_tracking_quality_gate_failure_count),
            optional_f64_json(self.min_external_localization_prior_usage_rate),
            optional_f64_json(self.min_overall_inlier_ratio),
            optional_f64_json(self.min_mean_inliers_per_successful_frame)
        )
    }
}

impl TrackingEvaluationFailure {
    pub fn reason(&self) -> &'static str {
        match self {
            Self::SuccessRateTooLow { .. } => "success_rate_too_low",
            Self::FailureRateTooHigh { .. } => "failure_rate_too_high",
            Self::LostCountTooHigh { .. } => "lost_count_too_high",
            Self::QualityGateFailureCountTooHigh { .. } => "quality_gate_failure_count_too_high",
            Self::ExternalLocalizationPriorUsageRateTooLow { .. } => {
                "external_localization_prior_usage_rate_too_low"
            }
            Self::OverallInlierRatioTooLow { .. } => "overall_inlier_ratio_too_low",
            Self::MeanInliersPerSuccessfulFrameTooLow { .. } => {
                "mean_inliers_per_successful_frame_too_low"
            }
        }
    }

    fn to_json_inline(&self) -> String {
        match self {
            Self::SuccessRateTooLow { actual, minimum }
            | Self::ExternalLocalizationPriorUsageRateTooLow { actual, minimum }
            | Self::OverallInlierRatioTooLow { actual, minimum }
            | Self::MeanInliersPerSuccessfulFrameTooLow { actual, minimum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"minimum\": {}}}",
                self.reason(),
                actual,
                minimum
            ),
            Self::FailureRateTooHigh { actual, maximum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"maximum\": {}}}",
                self.reason(),
                actual,
                maximum
            ),
            Self::LostCountTooHigh { actual, maximum }
            | Self::QualityGateFailureCountTooHigh { actual, maximum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"maximum\": {}}}",
                self.reason(),
                actual,
                maximum
            ),
        }
    }
}

pub fn tracking_results_to_html_report(results: &[TrackingResult]) -> String {
    let stats = TrackingStats::from_results(results);
    let mut output = String::new();
    output.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
    output.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    output.push_str("<title>visloc-rs tracking report</title>\n");
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
         th:first-child,td:first-child,th:nth-child(3),td:nth-child(3),th:nth-child(4),td:nth-child(4),th:last-child,td:last-child{text-align:left}\
         .ok{color:#198754;font-weight:700}.fail{color:#c23b3b;font-weight:700}\
         svg{width:100%;height:auto;display:block}",
    );
    output.push_str("</style>\n</head>\n<body>\n<main>\n");
    output.push_str("<h1>visloc-rs tracking report</h1>\n");
    output.push_str("<p class=\"sub\">Frame-by-frame sequence-localization state, failures, priors, and inlier diagnostics.</p>\n");
    output.push_str("<section class=\"grid\">\n");
    push_metric_card(&mut output, "Frames", &stats.frame_count.to_string());
    push_metric_card(
        &mut output,
        "Success rate",
        &format!("{:.1}%", stats.success_rate() * 100.0),
    );
    push_metric_card(
        &mut output,
        "Failed frames",
        &stats.failed_frame_count.to_string(),
    );
    push_metric_card(&mut output, "Lost events", &stats.lost_count.to_string());
    push_metric_card(
        &mut output,
        "Relocalized",
        &stats.relocalization_count.to_string(),
    );
    push_metric_card(
        &mut output,
        "Motion prior",
        &format!("{:.1}%", stats.pose_prior_usage_rate() * 100.0),
    );
    push_metric_card(
        &mut output,
        "External prior",
        &format!(
            "{} ({:.1}%)",
            stats.external_localization_prior_used_count,
            stats.external_localization_prior_usage_rate() * 100.0
        ),
    );
    push_metric_card(
        &mut output,
        "Mean inliers",
        &format!("{:.1}", stats.mean_inliers_per_successful_frame()),
    );
    output.push_str("</section>\n");
    output.push_str("<section class=\"panel\">\n");
    output.push_str(&tracking_timeline_svg(results));
    output.push_str("</section>\n");
    output.push_str("<section class=\"panel\">\n<h2>Frames</h2>\n");
    output.push_str("<table><thead><tr><th>frame</th><th>success</th><th>state</th><th>event</th><th>inliers</th><th>ratio</th><th>reprojection</th><th>priors</th><th>reason</th></tr></thead><tbody>\n");
    for result in results.iter().take(160) {
        let success_class = if result.localization.success {
            "ok"
        } else {
            "fail"
        };
        let success_text = if result.localization.success {
            "ok"
        } else {
            "failed"
        };
        let reason = tracking_result_reason(result);
        let _ = writeln!(
            output,
            "<tr><td>{}</td><td class=\"{}\">{}</td><td>{:?}</td><td>{:?}</td><td>{}</td><td>{:.3}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            result.frame_id,
            success_class,
            success_text,
            result.state,
            result.event,
            result.localization.inlier_count,
            result.localization.inlier_ratio,
            format_optional_metric(result.localization.reprojection_error, "px"),
            tracking_prior_text(result),
            html_escape(&reason),
        );
    }
    if results.len() > 160 {
        let _ = writeln!(
            output,
            "<tr><td colspan=\"9\">{} more rows omitted</td></tr>",
            results.len() - 160
        );
    }
    output.push_str("</tbody></table>\n</section>\n</main>\n</body>\n</html>\n");
    output
}

pub fn write_tracking_results_html_report(
    results: &[TrackingResult],
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    std::fs::write(path, tracking_results_to_html_report(results))
}

pub fn tracking_results_to_csv(results: &[TrackingResult]) -> String {
    let mut output = String::from(
        "frame_id,state,event,success,successive_failures,used_pose_prior,used_external_localization_prior,external_localization_prior_radius,tracking_failure_reason,localization_failure_reason,candidate_landmark_count,match_count,correspondence_count,inlier_count,outlier_count,inlier_ratio,reprojection_error,median_reprojection_error,max_reprojection_error,map_cameras,map_keyframes,map_landmarks,map_descriptors\n",
    );
    for result in results {
        let _ = writeln!(
            output,
            "{},{:?},{:?},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            result.frame_id,
            result.state,
            result.event,
            result.localization.success,
            result.successive_failures,
            result.used_pose_prior,
            result.used_external_localization_prior,
            optional_f64_csv(result.external_localization_prior_radius),
            csv_escape(&format_optional_debug(&result.tracking_failure_reason)),
            csv_escape(&format_optional_debug(&result.localization.failure_reason)),
            result.localization.candidate_landmark_count,
            result.localization.match_count,
            result.localization.correspondence_count,
            result.localization.inlier_count,
            result.localization.outlier_count,
            result.localization.inlier_ratio,
            optional_f64_csv(result.localization.reprojection_error),
            optional_f64_csv(result.localization.median_reprojection_error),
            optional_f64_csv(result.localization.max_reprojection_error),
            result.map_stats.camera_count,
            result.map_stats.keyframe_count,
            result.map_stats.landmark_count,
            result.map_stats.descriptor_count,
        );
    }
    output
}

pub fn write_tracking_results_csv(
    results: &[TrackingResult],
    path: impl AsRef<Path>,
) -> std::io::Result<()> {
    std::fs::write(path, tracking_results_to_csv(results))
}

fn tracking_timeline_svg(results: &[TrackingResult]) -> String {
    let mut output = String::new();
    output
        .push_str("<svg viewBox=\"0 0 900 190\" role=\"img\" aria-label=\"tracking timeline\">\n");
    output.push_str("<rect x=\"0\" y=\"0\" width=\"900\" height=\"190\" fill=\"#fbfcfd\"/>\n");
    output.push_str(
        "<line x1=\"70\" y1=\"92\" x2=\"830\" y2=\"92\" stroke=\"#d8e0e7\" stroke-width=\"2\"/>\n",
    );
    if results.is_empty() {
        output.push_str(
            "<text x=\"70\" y=\"102\" fill=\"#65727e\" font-size=\"14\">no frames</text>\n",
        );
        output.push_str("</svg>\n");
        return output;
    }

    let denom = results.len().saturating_sub(1).max(1) as f64;
    for (index, result) in results.iter().enumerate() {
        let x = 70.0 + (index as f64 / denom) * 760.0;
        let radius = if result.localization.success {
            8.0
        } else {
            9.0
        };
        let color = tracking_event_color(result);
        let y = if result.localization.success {
            78.0
        } else {
            108.0
        };
        let _ = writeln!(
            output,
            "<line x1=\"{x:.2}\" y1=\"92\" x2=\"{x:.2}\" y2=\"{y:.2}\" stroke=\"{color}\" stroke-width=\"2\"/>"
        );
        let _ = writeln!(
            output,
            "<circle cx=\"{x:.2}\" cy=\"{y:.2}\" r=\"{radius}\" fill=\"{color}\"/>"
        );
        if index == 0 || index + 1 == results.len() || result.event != TrackingEvent::Tracked {
            let _ = writeln!(
                output,
                "<text x=\"{:.2}\" y=\"142\" fill=\"#52616b\" font-size=\"12\" text-anchor=\"middle\">{}</text>",
                x, result.frame_id
            );
            let _ = writeln!(
                output,
                "<text x=\"{:.2}\" y=\"158\" fill=\"#52616b\" font-size=\"11\" text-anchor=\"middle\">{:?}</text>",
                x, result.event
            );
        }
    }
    output.push_str(
        "<text x=\"70\" y=\"32\" fill=\"#52616b\" font-size=\"13\">success/relocalization</text>\n",
    );
    output.push_str(
        "<text x=\"70\" y=\"176\" fill=\"#52616b\" font-size=\"13\">failure/lost</text>\n",
    );
    output.push_str("</svg>\n");
    output
}

fn tracking_event_color(result: &TrackingResult) -> &'static str {
    match result.event {
        TrackingEvent::Initialized => "#2676c9",
        TrackingEvent::Tracked => "#198754",
        TrackingEvent::TrackingFailed => "#d9822b",
        TrackingEvent::Lost => "#c23b3b",
        TrackingEvent::Relocalized => "#7b4ab8",
    }
}

fn tracking_result_reason(result: &TrackingResult) -> String {
    if let Some(reason) = &result.tracking_failure_reason {
        format!("{reason:?}")
    } else if let Some(reason) = &result.localization.failure_reason {
        format!("{reason:?}")
    } else {
        String::new()
    }
}

fn tracking_prior_text(result: &TrackingResult) -> String {
    let mut priors = Vec::new();
    if result.used_pose_prior {
        priors.push("motion".to_string());
    }
    if result.used_external_localization_prior {
        let label = if let Some(radius) = result.external_localization_prior_radius {
            format!("external({radius:.3}m)")
        } else {
            "external".to_string()
        };
        priors.push(label);
    }

    if priors.is_empty() {
        "none".to_string()
    } else {
        priors.join(" + ")
    }
}

fn format_optional_debug<T: fmt::Debug>(value: &Option<T>) -> String {
    value
        .as_ref()
        .map(|value| format!("{value:?}"))
        .unwrap_or_default()
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}
