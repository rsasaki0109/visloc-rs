//! Trajectory representation, TUM/KITTI parsing, Umeyama alignment, and
//! ATE / RPE / KITTI-odometry evaluation.

use super::*;

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectorySample {
    pub frame_id: FrameId,
    pub pose: Pose,
    pub state: TrackingState,
    pub event: TrackingEvent,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub reprojection_error: Option<f64>,
}

impl TrajectorySample {
    pub fn from_tracking_result(result: &TrackingResult) -> Option<Self> {
        if !result.localization.success {
            return None;
        }

        Some(Self {
            frame_id: result.frame_id,
            pose: result.localization.pose.clone()?,
            state: result.state,
            event: result.event,
            inlier_count: result.localization.inlier_count,
            inlier_ratio: result.localization.inlier_ratio,
            reprojection_error: result.localization.reprojection_error,
        })
    }

    pub fn camera_center_world(&self) -> Point3<f64> {
        self.pose.camera_center_world()
    }

    pub fn to_csv_record(&self) -> String {
        let center = self.camera_center_world();
        let q = self.pose.world_to_camera.rotation.quaternion();
        let t = self.pose.world_to_camera.translation;
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{:?},{:?},{},{},{}",
            self.frame_id,
            center.x,
            center.y,
            center.z,
            q.w,
            q.i,
            q.j,
            q.k,
            t.x,
            t.y,
            t.z,
            self.state,
            self.event,
            self.inlier_count,
            self.inlier_ratio,
            optional_f64_csv(self.reprojection_error)
        )
    }

    pub fn to_kitti_pose_record(&self) -> String {
        let camera_to_world = self.pose.camera_to_world().matrix();
        let mut values = Vec::with_capacity(12);
        for row in 0..3 {
            for column in 0..4 {
                values.push(camera_to_world[(row, column)].to_string());
            }
        }
        values.join(" ")
    }

    pub fn to_tum_pose_record(&self) -> String {
        let camera_to_world = self.pose.camera_to_world();
        let q = camera_to_world.rotation.quaternion();
        let t = camera_to_world.translation;
        format!(
            "{} {} {} {} {} {} {} {}",
            self.frame_id,
            export_f64(t.x),
            export_f64(t.y),
            export_f64(t.z),
            export_f64(q.i),
            export_f64(q.j),
            export_f64(q.k),
            export_f64(q.w)
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PoseTrajectory {
    pub(crate) samples: Vec<TrajectorySample>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TumTrajectoryParseError {
    pub line_number: usize,
    pub line: String,
    pub message: String,
}

impl TumTrajectoryParseError {
    pub(crate) fn new(line_number: usize, line: &str, message: impl Into<String>) -> Self {
        Self {
            line_number,
            line: line.to_string(),
            message: message.into(),
        }
    }
}

impl fmt::Display for TumTrajectoryParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid TUM trajectory line {}: {} ({})",
            self.line_number, self.line, self.message
        )
    }
}

impl std::error::Error for TumTrajectoryParseError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KittiTrajectoryParseError {
    pub line_number: usize,
    pub line: String,
    pub message: String,
}

impl KittiTrajectoryParseError {
    pub(crate) fn new(line_number: usize, line: &str, message: impl Into<String>) -> Self {
        Self {
            line_number,
            line: line.to_string(),
            message: message.into(),
        }
    }
}

impl fmt::Display for KittiTrajectoryParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid KITTI trajectory line {}: {} ({})",
            self.line_number, self.line, self.message
        )
    }
}

impl std::error::Error for KittiTrajectoryParseError {}

#[derive(Debug)]
pub enum TrajectoryFileError {
    Io(std::io::Error),
    TumParse(TumTrajectoryParseError),
    KittiParse(KittiTrajectoryParseError),
}

impl fmt::Display for TrajectoryFileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "trajectory file I/O error: {error}"),
            Self::TumParse(error) => write!(formatter, "{error}"),
            Self::KittiParse(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for TrajectoryFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::TumParse(error) => Some(error),
            Self::KittiParse(error) => Some(error),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectorySummary {
    pub pose_count: usize,
    pub first_frame_id: Option<FrameId>,
    pub last_frame_id: Option<FrameId>,
    pub total_path_length: f64,
    pub mean_inlier_count: Option<f64>,
    pub mean_inlier_ratio: Option<f64>,
    pub mean_reprojection_error: Option<f64>,
    pub min_camera_center_world: Option<[f64; 3]>,
    pub max_camera_center_world: Option<[f64; 3]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryTranslationError {
    pub frame_id: FrameId,
    pub translation_error: f64,
}

impl TrajectoryTranslationError {
    pub fn to_csv_record(&self) -> String {
        format!("{},{}", self.frame_id, self.translation_error)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TrajectoryAlignment {
    #[default]
    None,
    FirstMatchedTranslation,
    /// Rigid SE(3) alignment via Umeyama (1991): finds the rotation +
    /// translation that minimises the sum of squared distances between
    /// frame-id-matched camera centres. Scale is fixed at 1.
    Umeyama,
    /// Similarity SO(3) + scale + translation alignment via Umeyama
    /// (1991): the standard ATE protocol used to compare monocular
    /// / scale-ambiguous SLAM systems against ground truth.
    UmeyamaWithScale,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectorySimilarityTransform {
    pub scale: f64,
    pub rotation: Rotation3<f64>,
    pub translation: Vector3<f64>,
}

impl TrajectorySimilarityTransform {
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            rotation: Rotation3::identity(),
            translation: Vector3::zeros(),
        }
    }

    pub fn pure_translation(translation: Vector3<f64>) -> Self {
        Self {
            scale: 1.0,
            rotation: Rotation3::identity(),
            translation,
        }
    }

    pub fn apply(&self, point: &Point3<f64>) -> Point3<f64> {
        let v = self.rotation * (point.coords * self.scale) + self.translation;
        Point3::from(v)
    }
}

/// Closed-form similarity registration (Umeyama 1991).
///
/// Returns the transform `T = (scale, R, t)` that minimises
/// `sum_i || T(source_i) - target_i ||^2`. When `with_scale` is false
/// the scale is fixed at 1 (rigid SE(3) Procrustes). Returns `None`
/// when there are fewer than two correspondences or the source point
/// cloud has zero spread.
pub fn umeyama_similarity_transform(
    source: &[Point3<f64>],
    target: &[Point3<f64>],
    with_scale: bool,
) -> Option<TrajectorySimilarityTransform> {
    if source.len() != target.len() || source.len() < 2 {
        return None;
    }
    let n = source.len() as f64;
    let source_centroid = source
        .iter()
        .fold(Vector3::<f64>::zeros(), |acc, p| acc + p.coords)
        / n;
    let target_centroid = target
        .iter()
        .fold(Vector3::<f64>::zeros(), |acc, p| acc + p.coords)
        / n;

    let mut sigma = Matrix3::<f64>::zeros();
    let mut source_variance = 0.0f64;
    for (s, t) in source.iter().zip(target.iter()) {
        let xs = s.coords - source_centroid;
        let xt = t.coords - target_centroid;
        sigma += xt * xs.transpose();
        source_variance += xs.norm_squared();
    }
    sigma /= n;
    source_variance /= n;
    if source_variance <= f64::EPSILON {
        return None;
    }

    let svd = sigma.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let singular_values = svd.singular_values;

    let det_uv = u.determinant() * v_t.determinant();
    let mut s_diag = Vector3::new(1.0, 1.0, 1.0);
    if det_uv < 0.0 {
        s_diag.z = -1.0;
    }
    let s_matrix = Matrix3::from_diagonal(&s_diag);
    let rotation_matrix = u * s_matrix * v_t;
    let rotation = Rotation3::from_matrix_unchecked(rotation_matrix);

    let scale = if with_scale {
        let trace_ds = singular_values.x * s_diag.x
            + singular_values.y * s_diag.y
            + singular_values.z * s_diag.z;
        trace_ds / source_variance
    } else {
        1.0
    };

    let translation = target_centroid - rotation * (source_centroid * scale);

    Some(TrajectorySimilarityTransform {
        scale,
        rotation,
        translation,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryErrorSummary {
    pub estimated_pose_count: usize,
    pub reference_pose_count: usize,
    pub matched_pose_count: usize,
    pub missing_reference_count: usize,
    pub missing_estimate_count: usize,
    pub mean_translation_error: Option<f64>,
    pub rmse_translation_error: Option<f64>,
    pub max_translation_error: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrajectoryEvaluationConfig {
    pub max_mean_translation_error: Option<f64>,
    pub max_rmse_translation_error: Option<f64>,
    pub max_max_translation_error: Option<f64>,
    pub min_matched_pose_count: Option<usize>,
    pub min_match_ratio: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KittiOdometryBenchmarkConfig {
    /// Segment lengths in meters used by the KITTI odometry leaderboard.
    /// The current public KITTI leaderboard reports `[100, 200, ..., 800]`.
    /// Older devkit/debug runs often use short windows such as
    /// `[5, 10, 50, 100, 150, ..., 400]`; pass those explicitly when a
    /// sequence is too short for the official windows.
    pub segment_lengths_m: Vec<f64>,
    /// Frame stride for start indices. KITTI's devkit uses every frame.
    pub start_frame_step: usize,
}

impl Default for KittiOdometryBenchmarkConfig {
    fn default() -> Self {
        Self {
            segment_lengths_m: vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
            start_frame_step: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KittiOdometrySegmentError {
    pub first_frame_id: FrameId,
    pub last_frame_id: FrameId,
    pub length_m: f64,
    /// KITTI translational error for this segment, as a fraction of segment
    /// length. Multiply by 100 for percent.
    pub translational_error_ratio: f64,
    /// KITTI rotational error for this segment in degrees per meter.
    pub rotational_error_deg_per_m: f64,
}

impl KittiOdometrySegmentError {
    pub fn translational_error_percent(&self) -> f64 {
        self.translational_error_ratio * 100.0
    }

    pub fn to_csv_record(&self) -> String {
        format!(
            "{},{},{},{},{},{}",
            self.first_frame_id,
            self.last_frame_id,
            self.length_m,
            self.translational_error_ratio,
            self.translational_error_percent(),
            self.rotational_error_deg_per_m
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct KittiOdometryBenchmarkSummary {
    pub estimated_pose_count: usize,
    pub reference_pose_count: usize,
    pub matched_pose_count: usize,
    pub segment_count: usize,
    pub mean_translational_error_percent: Option<f64>,
    pub mean_rotational_error_deg_per_m: Option<f64>,
    pub segment_errors: Vec<KittiOdometrySegmentError>,
}

impl KittiOdometryBenchmarkSummary {
    pub fn max_translational_error_percent(&self) -> Option<f64> {
        self.segment_errors
            .iter()
            .map(KittiOdometrySegmentError::translational_error_percent)
            .reduce(f64::max)
    }

    pub fn max_rotational_error_deg_per_m(&self) -> Option<f64> {
        self.segment_errors
            .iter()
            .map(|error| error.rotational_error_deg_per_m)
            .reduce(f64::max)
    }

    pub fn segment_errors_csv(&self) -> String {
        let mut output = String::from(
            "first_frame_id,last_frame_id,length_m,translational_error_ratio,translational_error_percent,rotational_error_deg_per_m\n",
        );
        for error in &self.segment_errors {
            output.push_str(&error.to_csv_record());
            output.push('\n');
        }
        output
    }

    pub fn to_json(&self) -> String {
        let mut output = String::new();
        writeln!(&mut output, "{{").unwrap();
        writeln!(
            &mut output,
            "  \"estimated_pose_count\": {},",
            self.estimated_pose_count
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"reference_pose_count\": {},",
            self.reference_pose_count
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"matched_pose_count\": {},",
            self.matched_pose_count
        )
        .unwrap();
        writeln!(&mut output, "  \"segment_count\": {},", self.segment_count).unwrap();
        writeln!(
            &mut output,
            "  \"mean_translational_error_percent\": {},",
            optional_f64_json(self.mean_translational_error_percent)
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"mean_rotational_error_deg_per_m\": {},",
            optional_f64_json(self.mean_rotational_error_deg_per_m)
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"max_translational_error_percent\": {},",
            optional_f64_json(self.max_translational_error_percent())
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"max_rotational_error_deg_per_m\": {}",
            optional_f64_json(self.max_rotational_error_deg_per_m())
        )
        .unwrap();
        output.push_str("}\n");
        output
    }

    pub fn write_segment_errors_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.segment_errors_csv())
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }
}

/// Configuration for the TUM-style relative pose error (RPE).
///
/// RPE (Sturm et al., 2012) measures *local* drift: for each pair of poses a
/// fixed number of steps apart it compares the relative motion of the estimate
/// against the reference. Unlike the absolute trajectory error (ATE), RPE needs
/// no global alignment — a relative motion is invariant to any rigid transform
/// applied to the whole trajectory — which makes it the canonical companion
/// metric to ATE for SLAM back-ends and visual odometry front-ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelativePoseErrorConfig {
    /// Step gap Δ between the two poses of each relative-motion pair, counted in
    /// matched (frame-id-intersected) trajectory steps. `delta = 1` reports the
    /// per-step drift; larger values probe drift accumulated over longer
    /// windows.
    pub delta: usize,
    /// Stride between successive start indices. `1` evaluates every pose.
    pub start_step: usize,
}

impl Default for RelativePoseErrorConfig {
    fn default() -> Self {
        Self {
            delta: 1,
            start_step: 1,
        }
    }
}

/// One relative-pose-error sample over a single Δ-spaced pair.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativePoseError {
    pub first_frame_id: FrameId,
    pub last_frame_id: FrameId,
    /// Translational drift of the relative motion, in meters.
    pub translation_error: f64,
    /// Rotational drift of the relative motion, in degrees.
    pub rotation_error_deg: f64,
}

impl RelativePoseError {
    pub fn to_csv_record(&self) -> String {
        format!(
            "{},{},{},{}",
            self.first_frame_id,
            self.last_frame_id,
            self.translation_error,
            self.rotation_error_deg
        )
    }
}

/// Distribution statistics for one RPE channel (translation or rotation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativePoseErrorStatistics {
    /// Root-mean-square error — the headline RPE figure reported in the TUM
    /// protocol.
    pub rmse: f64,
    pub mean: f64,
    pub median: f64,
    /// Population standard deviation (`rmse² = mean² + std²`).
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl RelativePoseErrorStatistics {
    /// Statistics of a non-empty error sample. Returns `None` when `values` is
    /// empty.
    fn from_values(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let sum_sq = values.iter().map(|v| v * v).sum::<f64>();
        let rmse = (sum_sq / n).sqrt();
        let std = (sum_sq / n - mean * mean).max(0.0).sqrt();
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let middle = sorted.len() / 2;
        let median = if sorted.len() % 2 == 1 {
            sorted[middle]
        } else {
            0.5 * (sorted[middle - 1] + sorted[middle])
        };
        Some(Self {
            rmse,
            mean,
            median,
            std,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
        })
    }
}

/// Aggregated TUM-style relative pose error over all Δ-spaced pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativePoseErrorSummary {
    pub estimated_pose_count: usize,
    pub reference_pose_count: usize,
    pub matched_pose_count: usize,
    /// The step gap Δ used (clamped to at least 1).
    pub delta: usize,
    pub pair_count: usize,
    /// Translational error statistics in meters (`None` when no pair exists).
    pub translation: Option<RelativePoseErrorStatistics>,
    /// Rotational error statistics in degrees (`None` when no pair exists).
    pub rotation_deg: Option<RelativePoseErrorStatistics>,
    pub errors: Vec<RelativePoseError>,
}

impl RelativePoseErrorSummary {
    pub fn errors_csv(&self) -> String {
        let mut output =
            String::from("first_frame_id,last_frame_id,translation_error,rotation_error_deg\n");
        for error in &self.errors {
            output.push_str(&error.to_csv_record());
            output.push('\n');
        }
        output
    }

    pub fn write_errors_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.errors_csv())
    }

    pub fn to_json(&self) -> String {
        let mut output = String::new();
        writeln!(&mut output, "{{").unwrap();
        writeln!(
            &mut output,
            "  \"estimated_pose_count\": {},",
            self.estimated_pose_count
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"reference_pose_count\": {},",
            self.reference_pose_count
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"matched_pose_count\": {},",
            self.matched_pose_count
        )
        .unwrap();
        writeln!(&mut output, "  \"delta\": {},", self.delta).unwrap();
        writeln!(&mut output, "  \"pair_count\": {},", self.pair_count).unwrap();
        writeln!(
            &mut output,
            "  \"translation_m\": {},",
            relative_pose_error_statistics_json(self.translation)
        )
        .unwrap();
        writeln!(
            &mut output,
            "  \"rotation_deg\": {}",
            relative_pose_error_statistics_json(self.rotation_deg)
        )
        .unwrap();
        output.push_str("}\n");
        output
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryEvaluationResult {
    pub passed: bool,
    pub summary: TrajectoryErrorSummary,
    pub config: TrajectoryEvaluationConfig,
    pub match_ratio: Option<f64>,
    pub failures: Vec<TrajectoryEvaluationFailure>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrajectoryEvaluationFailure {
    MeanTranslationErrorTooHigh { actual: f64, maximum: f64 },
    RmseTranslationErrorTooHigh { actual: f64, maximum: f64 },
    MaxTranslationErrorTooHigh { actual: f64, maximum: f64 },
    NotEnoughMatchedPoses { actual: usize, minimum: usize },
    MatchRatioTooLow { actual: f64, minimum: f64 },
    NoMatchedPoses,
}

impl TrajectoryErrorSummary {
    pub fn from_trajectories(estimated: &PoseTrajectory, reference: &PoseTrajectory) -> Self {
        let errors = estimated.translation_errors_against(reference);
        Self::from_translation_errors(estimated, reference, &errors)
    }

    fn from_translation_errors(
        estimated: &PoseTrajectory,
        reference: &PoseTrajectory,
        errors: &[TrajectoryTranslationError],
    ) -> Self {
        let matched_pose_count = errors.len();
        let estimated_ids = estimated.frame_id_set();
        let reference_ids = reference.frame_id_set();
        let missing_reference_count = estimated_ids.difference(&reference_ids).count();
        let missing_estimate_count = reference_ids.difference(&estimated_ids).count();

        let (mean_translation_error, rmse_translation_error, max_translation_error) =
            if errors.is_empty() {
                (None, None, None)
            } else {
                let sum = errors
                    .iter()
                    .map(|error| error.translation_error)
                    .sum::<f64>();
                let squared_sum = errors
                    .iter()
                    .map(|error| error.translation_error * error.translation_error)
                    .sum::<f64>();
                let max = errors
                    .iter()
                    .map(|error| error.translation_error)
                    .fold(0.0_f64, f64::max);
                (
                    Some(sum / matched_pose_count as f64),
                    Some((squared_sum / matched_pose_count as f64).sqrt()),
                    Some(max),
                )
            };

        Self {
            estimated_pose_count: estimated.len(),
            reference_pose_count: reference.len(),
            matched_pose_count,
            missing_reference_count,
            missing_estimate_count,
            mean_translation_error,
            rmse_translation_error,
            max_translation_error,
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"estimated_pose_count\": {},\n",
                "  \"reference_pose_count\": {},\n",
                "  \"matched_pose_count\": {},\n",
                "  \"missing_reference_count\": {},\n",
                "  \"missing_estimate_count\": {},\n",
                "  \"mean_translation_error\": {},\n",
                "  \"rmse_translation_error\": {},\n",
                "  \"max_translation_error\": {}\n",
                "}}\n"
            ),
            self.estimated_pose_count,
            self.reference_pose_count,
            self.matched_pose_count,
            self.missing_reference_count,
            self.missing_estimate_count,
            optional_f64_json(self.mean_translation_error),
            optional_f64_json(self.rmse_translation_error),
            optional_f64_json(self.max_translation_error)
        )
    }

    fn to_json_inline(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"estimated_pose_count\": {}, ",
                "\"reference_pose_count\": {}, ",
                "\"matched_pose_count\": {}, ",
                "\"missing_reference_count\": {}, ",
                "\"missing_estimate_count\": {}, ",
                "\"mean_translation_error\": {}, ",
                "\"rmse_translation_error\": {}, ",
                "\"max_translation_error\": {}",
                "}}"
            ),
            self.estimated_pose_count,
            self.reference_pose_count,
            self.matched_pose_count,
            self.missing_reference_count,
            self.missing_estimate_count,
            optional_f64_json(self.mean_translation_error),
            optional_f64_json(self.rmse_translation_error),
            optional_f64_json(self.max_translation_error)
        )
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    pub fn evaluate(&self, config: TrajectoryEvaluationConfig) -> TrajectoryEvaluationResult {
        TrajectoryEvaluationResult::from_summary(self.clone(), config)
    }
}

impl TrajectoryEvaluationResult {
    pub fn from_summary(
        summary: TrajectoryErrorSummary,
        config: TrajectoryEvaluationConfig,
    ) -> Self {
        let match_ratio = if summary.reference_pose_count == 0 {
            None
        } else {
            Some(summary.matched_pose_count as f64 / summary.reference_pose_count as f64)
        };
        let mut failures = Vec::new();

        if summary.matched_pose_count == 0 {
            failures.push(TrajectoryEvaluationFailure::NoMatchedPoses);
        }

        if let Some(maximum) = config.max_mean_translation_error {
            match summary.mean_translation_error {
                Some(actual) if actual > maximum => {
                    failures.push(TrajectoryEvaluationFailure::MeanTranslationErrorTooHigh {
                        actual,
                        maximum,
                    })
                }
                None => failures.push(TrajectoryEvaluationFailure::NoMatchedPoses),
                _ => {}
            }
        }

        if let Some(maximum) = config.max_rmse_translation_error {
            match summary.rmse_translation_error {
                Some(actual) if actual > maximum => {
                    failures.push(TrajectoryEvaluationFailure::RmseTranslationErrorTooHigh {
                        actual,
                        maximum,
                    })
                }
                None => failures.push(TrajectoryEvaluationFailure::NoMatchedPoses),
                _ => {}
            }
        }

        if let Some(maximum) = config.max_max_translation_error {
            match summary.max_translation_error {
                Some(actual) if actual > maximum => {
                    failures.push(TrajectoryEvaluationFailure::MaxTranslationErrorTooHigh {
                        actual,
                        maximum,
                    })
                }
                None => failures.push(TrajectoryEvaluationFailure::NoMatchedPoses),
                _ => {}
            }
        }

        if let Some(minimum) = config.min_matched_pose_count {
            if summary.matched_pose_count < minimum {
                failures.push(TrajectoryEvaluationFailure::NotEnoughMatchedPoses {
                    actual: summary.matched_pose_count,
                    minimum,
                });
            }
        }

        if let Some(minimum) = config.min_match_ratio {
            match match_ratio {
                Some(actual) if actual < minimum => {
                    failures.push(TrajectoryEvaluationFailure::MatchRatioTooLow { actual, minimum })
                }
                None => failures.push(TrajectoryEvaluationFailure::MatchRatioTooLow {
                    actual: 0.0,
                    minimum,
                }),
                _ => {}
            }
        }

        failures.dedup();
        let passed = failures.is_empty();

        Self {
            passed,
            summary,
            config,
            match_ratio,
            failures,
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"passed\": {},\n",
                "  \"match_ratio\": {},\n",
                "  \"config\": {},\n",
                "  \"summary\": {},\n",
                "  \"failures\": {}\n",
                "}}\n"
            ),
            self.passed,
            optional_f64_json(self.match_ratio),
            self.config.to_json_inline(),
            self.summary.to_json_inline(),
            trajectory_evaluation_failures_json(&self.failures)
        )
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }
}

impl TrajectoryEvaluationConfig {
    fn to_json_inline(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"max_mean_translation_error\": {}, ",
                "\"max_rmse_translation_error\": {}, ",
                "\"max_max_translation_error\": {}, ",
                "\"min_matched_pose_count\": {}, ",
                "\"min_match_ratio\": {}",
                "}}"
            ),
            optional_f64_json(self.max_mean_translation_error),
            optional_f64_json(self.max_rmse_translation_error),
            optional_f64_json(self.max_max_translation_error),
            optional_usize_json(self.min_matched_pose_count),
            optional_f64_json(self.min_match_ratio)
        )
    }
}

impl TrajectoryEvaluationFailure {
    pub fn reason(&self) -> &'static str {
        match self {
            Self::MeanTranslationErrorTooHigh { .. } => "mean_translation_error_too_high",
            Self::RmseTranslationErrorTooHigh { .. } => "rmse_translation_error_too_high",
            Self::MaxTranslationErrorTooHigh { .. } => "max_translation_error_too_high",
            Self::NotEnoughMatchedPoses { .. } => "not_enough_matched_poses",
            Self::MatchRatioTooLow { .. } => "match_ratio_too_low",
            Self::NoMatchedPoses => "no_matched_poses",
        }
    }

    pub(crate) fn to_json_inline(&self) -> String {
        match self {
            Self::MeanTranslationErrorTooHigh { actual, maximum }
            | Self::RmseTranslationErrorTooHigh { actual, maximum }
            | Self::MaxTranslationErrorTooHigh { actual, maximum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"maximum\": {}}}",
                self.reason(),
                actual,
                maximum
            ),
            Self::NotEnoughMatchedPoses { actual, minimum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"minimum\": {}}}",
                self.reason(),
                actual,
                minimum
            ),
            Self::MatchRatioTooLow { actual, minimum } => format!(
                "{{\"reason\": \"{}\", \"actual\": {}, \"minimum\": {}}}",
                self.reason(),
                actual,
                minimum
            ),
            Self::NoMatchedPoses => {
                format!("{{\"reason\": \"{}\"}}", self.reason())
            }
        }
    }
}

impl TrajectorySummary {
    pub fn from_trajectory(trajectory: &PoseTrajectory) -> Self {
        let pose_count = trajectory.len();
        let first_frame_id = trajectory.samples.first().map(|sample| sample.frame_id);
        let last_frame_id = trajectory.samples.last().map(|sample| sample.frame_id);
        let total_path_length = trajectory.total_path_length();
        let mean_reprojection_error = trajectory.mean_reprojection_error();

        let (mean_inlier_count, mean_inlier_ratio) = if pose_count == 0 {
            (None, None)
        } else {
            let inlier_count_sum: usize = trajectory
                .samples
                .iter()
                .map(|sample| sample.inlier_count)
                .sum();
            let inlier_ratio_sum: f64 = trajectory
                .samples
                .iter()
                .map(|sample| sample.inlier_ratio)
                .sum();
            (
                Some(inlier_count_sum as f64 / pose_count as f64),
                Some(inlier_ratio_sum / pose_count as f64),
            )
        };

        let (min_camera_center_world, max_camera_center_world) =
            trajectory.camera_center_bounds_world();

        Self {
            pose_count,
            first_frame_id,
            last_frame_id,
            total_path_length,
            mean_inlier_count,
            mean_inlier_ratio,
            mean_reprojection_error,
            min_camera_center_world,
            max_camera_center_world,
        }
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"pose_count\": {},\n",
                "  \"first_frame_id\": {},\n",
                "  \"last_frame_id\": {},\n",
                "  \"total_path_length\": {},\n",
                "  \"mean_inlier_count\": {},\n",
                "  \"mean_inlier_ratio\": {},\n",
                "  \"mean_reprojection_error\": {},\n",
                "  \"min_camera_center_world\": {},\n",
                "  \"max_camera_center_world\": {}\n",
                "}}\n"
            ),
            self.pose_count,
            optional_frame_id_json(self.first_frame_id),
            optional_frame_id_json(self.last_frame_id),
            self.total_path_length,
            optional_f64_json(self.mean_inlier_count),
            optional_f64_json(self.mean_inlier_ratio),
            optional_f64_json(self.mean_reprojection_error),
            optional_vec3_json(self.min_camera_center_world),
            optional_vec3_json(self.max_camera_center_world)
        )
    }
}

impl PoseTrajectory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_tracking_results(results: &[TrackingResult]) -> Self {
        let mut trajectory = Self::new();
        for result in results {
            trajectory.push_result(result);
        }
        trajectory
    }

    pub fn from_tum_poses_str(text: &str) -> Result<Self, TumTrajectoryParseError> {
        let mut trajectory = Self::new();
        for (line_index, line) in text.lines().enumerate() {
            let line_number = line_index + 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let fields = trimmed.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 8 {
                return Err(TumTrajectoryParseError::new(
                    line_number,
                    line,
                    format!("expected 8 fields, got {}", fields.len()),
                ));
            }

            let frame_id = parse_tum_frame_id(fields[0], line_number, line)?;
            let tx = parse_tum_f64(fields[1], "tx", line_number, line)?;
            let ty = parse_tum_f64(fields[2], "ty", line_number, line)?;
            let tz = parse_tum_f64(fields[3], "tz", line_number, line)?;
            let qx = parse_tum_f64(fields[4], "qx", line_number, line)?;
            let qy = parse_tum_f64(fields[5], "qy", line_number, line)?;
            let qz = parse_tum_f64(fields[6], "qz", line_number, line)?;
            let qw = parse_tum_f64(fields[7], "qw", line_number, line)?;
            let Some(rotation) = UnitQuaternion::try_new(Quaternion::new(qw, qx, qy, qz), 1.0e-12)
            else {
                return Err(TumTrajectoryParseError::new(
                    line_number,
                    line,
                    "quaternion norm is too small",
                ));
            };

            let camera_to_world = SE3::new(rotation, Vector3::new(tx, ty, tz));
            trajectory.push_sample(TrajectorySample {
                frame_id,
                pose: Pose {
                    world_to_camera: camera_to_world.inverse(),
                },
                state: TrackingState::Tracking,
                event: TrackingEvent::Tracked,
                inlier_count: 0,
                inlier_ratio: 0.0,
                reprojection_error: None,
            });
        }
        Ok(trajectory)
    }

    pub fn read_tum_poses(path: impl AsRef<Path>) -> Result<Self, TrajectoryFileError> {
        let text = std::fs::read_to_string(path).map_err(TrajectoryFileError::Io)?;
        Self::from_tum_poses_str(&text).map_err(TrajectoryFileError::TumParse)
    }

    pub fn from_kitti_poses_str(text: &str) -> Result<Self, KittiTrajectoryParseError> {
        let mut trajectory = Self::new();
        let mut pose_index = 0_u64;
        for (line_index, line) in text.lines().enumerate() {
            let line_number = line_index + 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let fields = trimmed.split_whitespace().collect::<Vec<_>>();
            if fields.len() != 12 {
                return Err(KittiTrajectoryParseError::new(
                    line_number,
                    line,
                    format!("expected 12 fields, got {}", fields.len()),
                ));
            }

            let mut values = [0.0_f64; 12];
            for (index, field) in fields.iter().enumerate() {
                values[index] = parse_kitti_f64(field, index, line_number, line)?;
            }

            let rotation = Matrix3::new(
                values[0], values[1], values[2], values[4], values[5], values[6], values[8],
                values[9], values[10],
            );
            let translation = Vector3::new(values[3], values[7], values[11]);
            let camera_to_world = SE3::new(
                UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation)),
                translation,
            );
            trajectory.push_sample(TrajectorySample {
                frame_id: pose_index,
                pose: Pose {
                    world_to_camera: camera_to_world.inverse(),
                },
                state: TrackingState::Tracking,
                event: TrackingEvent::Tracked,
                inlier_count: 0,
                inlier_ratio: 0.0,
                reprojection_error: None,
            });
            pose_index += 1;
        }
        Ok(trajectory)
    }

    pub fn read_kitti_poses(path: impl AsRef<Path>) -> Result<Self, TrajectoryFileError> {
        let text = std::fs::read_to_string(path).map_err(TrajectoryFileError::Io)?;
        Self::from_kitti_poses_str(&text).map_err(TrajectoryFileError::KittiParse)
    }

    pub fn push_result(&mut self, result: &TrackingResult) -> bool {
        if let Some(sample) = TrajectorySample::from_tracking_result(result) {
            self.samples.push(sample);
            true
        } else {
            false
        }
    }

    pub fn push_sample(&mut self, sample: TrajectorySample) {
        self.samples.push(sample);
    }

    pub fn samples(&self) -> &[TrajectorySample] {
        &self.samples
    }

    pub fn into_samples(self) -> Vec<TrajectorySample> {
        self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn frame_ids(&self) -> Vec<FrameId> {
        self.samples.iter().map(|sample| sample.frame_id).collect()
    }

    fn frame_id_set(&self) -> HashSet<FrameId> {
        self.samples.iter().map(|sample| sample.frame_id).collect()
    }

    pub fn camera_centers_world(&self) -> Vec<Point3<f64>> {
        self.samples
            .iter()
            .map(TrajectorySample::camera_center_world)
            .collect()
    }

    pub fn camera_center_bounds_world(&self) -> (Option<[f64; 3]>, Option<[f64; 3]>) {
        let Some(first) = self.samples.first() else {
            return (None, None);
        };

        let first_center = first.camera_center_world();
        let mut min = [first_center.x, first_center.y, first_center.z];
        let mut max = min;

        for sample in self.samples.iter().skip(1) {
            let center = sample.camera_center_world();
            min[0] = min[0].min(center.x);
            min[1] = min[1].min(center.y);
            min[2] = min[2].min(center.z);
            max[0] = max[0].max(center.x);
            max[1] = max[1].max(center.y);
            max[2] = max[2].max(center.z);
        }

        (Some(min), Some(max))
    }

    pub(crate) fn trajectory_alignment_transform(
        &self,
        reference_by_frame_id: &HashMap<FrameId, Point3<f64>>,
        alignment: TrajectoryAlignment,
    ) -> TrajectorySimilarityTransform {
        match alignment {
            TrajectoryAlignment::None => TrajectorySimilarityTransform::identity(),
            TrajectoryAlignment::FirstMatchedTranslation => self
                .samples
                .iter()
                .find_map(|sample| {
                    let reference_center = reference_by_frame_id.get(&sample.frame_id)?;
                    Some(TrajectorySimilarityTransform::pure_translation(
                        reference_center.coords - sample.camera_center_world().coords,
                    ))
                })
                .unwrap_or_else(TrajectorySimilarityTransform::identity),
            TrajectoryAlignment::Umeyama | TrajectoryAlignment::UmeyamaWithScale => {
                let mut source = Vec::with_capacity(self.samples.len());
                let mut target = Vec::with_capacity(self.samples.len());
                for sample in &self.samples {
                    if let Some(reference_center) = reference_by_frame_id.get(&sample.frame_id) {
                        source.push(sample.camera_center_world());
                        target.push(*reference_center);
                    }
                }
                let with_scale = matches!(alignment, TrajectoryAlignment::UmeyamaWithScale);
                umeyama_similarity_transform(&source, &target, with_scale)
                    .unwrap_or_else(TrajectorySimilarityTransform::identity)
            }
        }
    }

    pub fn total_path_length(&self) -> f64 {
        self.samples
            .windows(2)
            .map(|window| {
                (window[1].camera_center_world() - window[0].camera_center_world()).norm()
            })
            .sum()
    }

    pub fn kitti_odometry_benchmark_against(
        &self,
        reference: &PoseTrajectory,
        config: &KittiOdometryBenchmarkConfig,
    ) -> KittiOdometryBenchmarkSummary {
        let estimated_by_frame_id: HashMap<FrameId, &TrajectorySample> = self
            .samples
            .iter()
            .map(|sample| (sample.frame_id, sample))
            .collect();
        let matched: Vec<(&TrajectorySample, &TrajectorySample)> = reference
            .samples
            .iter()
            .filter_map(|reference_sample| {
                let estimated_sample = estimated_by_frame_id.get(&reference_sample.frame_id)?;
                Some((*estimated_sample, reference_sample))
            })
            .collect();

        let distances = cumulative_reference_distances(&matched);
        let mut segment_errors = Vec::new();
        let start_step = config.start_frame_step.max(1);
        for first_index in (0..matched.len()).step_by(start_step) {
            let first_distance = distances[first_index];
            for &length_m in &config.segment_lengths_m {
                if !(length_m.is_finite() && length_m > 0.0) {
                    continue;
                }
                let target_distance = first_distance + length_m;
                let Some(last_index) = first_index_for_distance(&distances, target_distance) else {
                    continue;
                };
                if last_index <= first_index {
                    continue;
                }
                let (estimated_first, reference_first) = matched[first_index];
                let (estimated_last, reference_last) = matched[last_index];
                let estimated_delta = relative_camera_to_world(estimated_first, estimated_last);
                let reference_delta = relative_camera_to_world(reference_first, reference_last);
                let error = estimated_delta.inverse().compose(&reference_delta);
                let translation_error = error.translation.norm() / length_m;
                let rotation_error = error.rotation.angle().to_degrees() / length_m;
                segment_errors.push(KittiOdometrySegmentError {
                    first_frame_id: reference_first.frame_id,
                    last_frame_id: reference_last.frame_id,
                    length_m,
                    translational_error_ratio: translation_error,
                    rotational_error_deg_per_m: rotation_error,
                });
            }
        }

        let mean_translational_error_percent = if segment_errors.is_empty() {
            None
        } else {
            Some(
                segment_errors
                    .iter()
                    .map(KittiOdometrySegmentError::translational_error_percent)
                    .sum::<f64>()
                    / segment_errors.len() as f64,
            )
        };
        let mean_rotational_error_deg_per_m = if segment_errors.is_empty() {
            None
        } else {
            Some(
                segment_errors
                    .iter()
                    .map(|error| error.rotational_error_deg_per_m)
                    .sum::<f64>()
                    / segment_errors.len() as f64,
            )
        };
        KittiOdometryBenchmarkSummary {
            estimated_pose_count: self.len(),
            reference_pose_count: reference.len(),
            matched_pose_count: matched.len(),
            segment_count: segment_errors.len(),
            mean_translational_error_percent,
            mean_rotational_error_deg_per_m,
            segment_errors,
        }
    }

    /// Compute the TUM-style relative pose error (RPE) of this trajectory
    /// against `reference`.
    ///
    /// Poses are matched by frame id; the i-th and (i+Δ)-th matched poses form
    /// each relative-motion pair, where Δ is `config.delta`. For each pair the
    /// relative camera motion of the estimate (`ΔP`) and of the reference
    /// (`ΔQ`) are compared via the residual `E = ΔQ⁻¹ · ΔP`, contributing its
    /// translation norm (m) and rotation angle (deg). Because relative motion
    /// is invariant to a global rigid transform of either trajectory, RPE needs
    /// no prior alignment — its defining advantage over the ATE.
    pub fn relative_pose_error_against(
        &self,
        reference: &PoseTrajectory,
        config: &RelativePoseErrorConfig,
    ) -> RelativePoseErrorSummary {
        let estimated_by_frame_id: HashMap<FrameId, &TrajectorySample> = self
            .samples
            .iter()
            .map(|sample| (sample.frame_id, sample))
            .collect();
        let matched: Vec<(&TrajectorySample, &TrajectorySample)> = reference
            .samples
            .iter()
            .filter_map(|reference_sample| {
                let estimated_sample = estimated_by_frame_id.get(&reference_sample.frame_id)?;
                Some((*estimated_sample, reference_sample))
            })
            .collect();

        let delta = config.delta.max(1);
        let start_step = config.start_step.max(1);
        let mut errors = Vec::new();
        let mut first_index = 0;
        while first_index + delta < matched.len() {
            let last_index = first_index + delta;
            let (estimated_first, reference_first) = matched[first_index];
            let (estimated_last, reference_last) = matched[last_index];
            let estimated_delta = relative_camera_to_world(estimated_first, estimated_last);
            let reference_delta = relative_camera_to_world(reference_first, reference_last);
            let residual = reference_delta.inverse().compose(&estimated_delta);
            errors.push(RelativePoseError {
                first_frame_id: reference_first.frame_id,
                last_frame_id: reference_last.frame_id,
                translation_error: residual.translation.norm(),
                rotation_error_deg: residual.rotation.angle().to_degrees(),
            });
            first_index += start_step;
        }

        let translation_values: Vec<f64> = errors.iter().map(|e| e.translation_error).collect();
        let rotation_values: Vec<f64> = errors.iter().map(|e| e.rotation_error_deg).collect();
        RelativePoseErrorSummary {
            estimated_pose_count: self.len(),
            reference_pose_count: reference.len(),
            matched_pose_count: matched.len(),
            delta,
            pair_count: errors.len(),
            translation: RelativePoseErrorStatistics::from_values(&translation_values),
            rotation_deg: RelativePoseErrorStatistics::from_values(&rotation_values),
            errors,
        }
    }

    pub fn mean_reprojection_error(&self) -> Option<f64> {
        let mut sum = 0.0;
        let mut count = 0;
        for sample in &self.samples {
            if let Some(error) = sample.reprojection_error {
                sum += error;
                count += 1;
            }
        }

        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }

    pub fn to_csv(&self) -> String {
        let mut output = String::from(
            "frame_id,camera_center_x,camera_center_y,camera_center_z,qw,qx,qy,qz,tx,ty,tz,state,event,inlier_count,inlier_ratio,reprojection_error\n",
        );
        for sample in &self.samples {
            output.push_str(&sample.to_csv_record());
            output.push('\n');
        }
        output
    }

    pub fn write_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_csv())
    }

    pub fn to_kitti_poses(&self) -> String {
        let mut output = String::new();
        for sample in &self.samples {
            output.push_str(&sample.to_kitti_pose_record());
            output.push('\n');
        }
        output
    }

    pub fn write_kitti_poses(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_kitti_poses())
    }

    pub fn to_tum_poses(&self) -> String {
        let mut output = String::new();
        for sample in &self.samples {
            output.push_str(&sample.to_tum_pose_record());
            output.push('\n');
        }
        output
    }

    pub fn write_tum_poses(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_tum_poses())
    }

    pub fn translation_errors_against(
        &self,
        reference: &PoseTrajectory,
    ) -> Vec<TrajectoryTranslationError> {
        self.translation_errors_against_with_alignment(reference, TrajectoryAlignment::None)
    }

    pub fn translation_errors_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
    ) -> Vec<TrajectoryTranslationError> {
        let reference_by_frame_id = reference
            .samples
            .iter()
            .map(|sample| (sample.frame_id, sample.camera_center_world()))
            .collect::<HashMap<_, _>>();
        let transform = self.trajectory_alignment_transform(&reference_by_frame_id, alignment);

        self.samples
            .iter()
            .filter_map(|sample| {
                let reference_center = reference_by_frame_id.get(&sample.frame_id)?;
                let aligned_center = transform.apply(&sample.camera_center_world());
                Some(TrajectoryTranslationError {
                    frame_id: sample.frame_id,
                    translation_error: (aligned_center - reference_center).norm(),
                })
            })
            .collect()
    }

    pub fn translation_error_summary_against(
        &self,
        reference: &PoseTrajectory,
    ) -> TrajectoryErrorSummary {
        TrajectoryErrorSummary::from_trajectories(self, reference)
    }

    pub fn translation_error_summary_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
    ) -> TrajectoryErrorSummary {
        let errors = self.translation_errors_against_with_alignment(reference, alignment);
        TrajectoryErrorSummary::from_translation_errors(self, reference, &errors)
    }

    pub fn translation_errors_csv_against(&self, reference: &PoseTrajectory) -> String {
        self.translation_errors_csv_against_with_alignment(reference, TrajectoryAlignment::None)
    }

    pub fn translation_errors_csv_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
    ) -> String {
        let mut output = String::from("frame_id,translation_error\n");
        for error in self.translation_errors_against_with_alignment(reference, alignment) {
            output.push_str(&error.to_csv_record());
            output.push('\n');
        }
        output
    }

    pub fn write_translation_errors_csv_against(
        &self,
        reference: &PoseTrajectory,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        std::fs::write(path, self.translation_errors_csv_against(reference))
    }

    pub fn write_translation_errors_csv_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        std::fs::write(
            path,
            self.translation_errors_csv_against_with_alignment(reference, alignment),
        )
    }

    pub fn write_translation_error_summary_json_against(
        &self,
        reference: &PoseTrajectory,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        self.translation_error_summary_against(reference)
            .write_json(path)
    }

    pub fn write_translation_error_summary_json_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        self.translation_error_summary_against_with_alignment(reference, alignment)
            .write_json(path)
    }

    pub fn to_html_report(&self) -> String {
        let summary = self.summary();
        let svg = trajectory_svg(self);
        let mut output = String::new();
        output.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
        output
            .push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
        output.push_str("<title>visloc-rs trajectory report</title>\n");
        output.push_str("<style>");
        output.push_str(
            "body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f7f9;color:#182026}\
             main{max-width:1080px;margin:0 auto;padding:28px}\
             h1{font-size:24px;margin:0 0 8px}\
             .sub{margin:0 0 22px;color:#52616b}\
             .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}\
             .metric{background:white;border:1px solid #dde3ea;border-radius:8px;padding:12px}\
             .label{display:block;font-size:12px;color:#65727e}\
             .value{display:block;font-size:22px;font-weight:700;margin-top:4px}\
             .panel{background:white;border:1px solid #dde3ea;border-radius:8px;padding:16px;margin-top:14px}\
             .legend{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 0;color:#52616b;font-size:13px}\
             .swatch{display:inline-block;width:26px;height:4px;border-radius:2px;vertical-align:middle;margin-right:6px}\
             code{background:#eef2f5;border-radius:4px;padding:2px 4px}\
             svg{width:100%;height:auto;display:block}",
        );
        output.push_str("</style>\n</head>\n<body>\n<main>\n");
        output.push_str("<h1>visloc-rs trajectory report</h1>\n");
        output.push_str(
            "<p class=\"sub\">Estimated camera-center trajectory from sequence localization.</p>\n",
        );
        output.push_str("<section class=\"grid\">\n");
        push_metric_card(&mut output, "Poses", &summary.pose_count.to_string());
        push_metric_card(
            &mut output,
            "First frame",
            &format_optional_frame_id(summary.first_frame_id),
        );
        push_metric_card(
            &mut output,
            "Last frame",
            &format_optional_frame_id(summary.last_frame_id),
        );
        push_metric_card(
            &mut output,
            "Path length",
            &format_optional_metric(Some(summary.total_path_length), "m"),
        );
        push_metric_card(
            &mut output,
            "Mean inliers",
            &format_optional_count(summary.mean_inlier_count),
        );
        push_metric_card(
            &mut output,
            "Mean reprojection",
            &format_optional_metric(summary.mean_reprojection_error, "px"),
        );
        output.push_str("</section>\n");
        output.push_str("<section class=\"panel\">\n");
        output.push_str(&svg);
        output.push_str(
            "<div class=\"legend\"><span><span class=\"swatch\" style=\"background:#e0574f\"></span>estimated trajectory</span></div>\n",
        );
        output.push_str("</section>\n</main>\n</body>\n</html>\n");
        output
    }

    pub fn write_html_report(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_html_report())
    }

    pub fn to_html_report_against(&self, reference: &PoseTrajectory) -> String {
        self.to_html_report_against_with_alignment(reference, TrajectoryAlignment::None)
    }

    pub fn to_html_report_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
    ) -> String {
        let summary = self.translation_error_summary_against_with_alignment(reference, alignment);
        let errors = self.translation_errors_against_with_alignment(reference, alignment);
        let svg = trajectory_comparison_svg(self, reference, alignment);
        let mut output = String::new();
        output.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
        output
            .push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
        output.push_str("<title>visloc-rs trajectory evaluation</title>\n");
        output.push_str("<style>");
        output.push_str(
            "body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f7f9;color:#182026}\
             main{max-width:1080px;margin:0 auto;padding:28px}\
             h1{font-size:24px;margin:0 0 8px}\
             .sub{margin:0 0 22px;color:#52616b}\
             .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}\
             .metric{background:white;border:1px solid #dde3ea;border-radius:8px;padding:12px}\
             .label{display:block;font-size:12px;color:#65727e}\
             .value{display:block;font-size:22px;font-weight:700;margin-top:4px}\
             .panel{background:white;border:1px solid #dde3ea;border-radius:8px;padding:16px;margin-top:14px}\
             .legend{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 0;color:#52616b;font-size:13px}\
             .swatch{display:inline-block;width:26px;height:4px;border-radius:2px;vertical-align:middle;margin-right:6px}\
             table{width:100%;border-collapse:collapse;font-size:13px}\
             th,td{text-align:right;border-bottom:1px solid #e7ecf0;padding:6px 8px}\
             th:first-child,td:first-child{text-align:left}\
             code{background:#eef2f5;border-radius:4px;padding:2px 4px}\
             svg{width:100%;height:auto;display:block}",
        );
        output.push_str("</style>\n</head>\n<body>\n<main>\n");
        output.push_str("<h1>visloc-rs trajectory evaluation</h1>\n");
        let _ = writeln!(
            output,
            "<p class=\"sub\">Estimated trajectory compared with reference poses. Alignment: <code>{alignment:?}</code></p>"
        );
        output.push_str("<section class=\"grid\">\n");
        push_metric_card(
            &mut output,
            "Estimated poses",
            &summary.estimated_pose_count.to_string(),
        );
        push_metric_card(
            &mut output,
            "Reference poses",
            &summary.reference_pose_count.to_string(),
        );
        push_metric_card(
            &mut output,
            "Matched poses",
            &summary.matched_pose_count.to_string(),
        );
        push_metric_card(
            &mut output,
            "Mean error",
            &format_optional_metric(summary.mean_translation_error, "m"),
        );
        push_metric_card(
            &mut output,
            "RMSE",
            &format_optional_metric(summary.rmse_translation_error, "m"),
        );
        push_metric_card(
            &mut output,
            "Max error",
            &format_optional_metric(summary.max_translation_error, "m"),
        );
        output.push_str("</section>\n");
        output.push_str("<section class=\"panel\">\n");
        output.push_str(&svg);
        output.push_str(
            "<div class=\"legend\"><span><span class=\"swatch\" style=\"background:#e0574f\"></span>estimated</span><span><span class=\"swatch\" style=\"background:#2676c9\"></span>reference</span><span><span class=\"swatch\" style=\"background:#9aa7b2\"></span>matched error</span></div>\n",
        );
        output.push_str("</section>\n");
        output.push_str("<section class=\"panel\">\n<h2>Matched translation errors</h2>\n");
        output.push_str(
            "<table><thead><tr><th>frame</th><th>translation error</th></tr></thead><tbody>\n",
        );
        for error in errors.iter().take(80) {
            let _ = writeln!(
                output,
                "<tr><td>{}</td><td>{}</td></tr>",
                error.frame_id,
                format_optional_metric(Some(error.translation_error), "m")
            );
        }
        if errors.len() > 80 {
            let _ = writeln!(
                output,
                "<tr><td colspan=\"2\">{} more rows omitted</td></tr>",
                errors.len() - 80
            );
        }
        output.push_str("</tbody></table>\n</section>\n</main>\n</body>\n</html>\n");
        output
    }

    pub fn write_html_report_against(
        &self,
        reference: &PoseTrajectory,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        std::fs::write(path, self.to_html_report_against(reference))
    }

    pub fn write_html_report_against_with_alignment(
        &self,
        reference: &PoseTrajectory,
        alignment: TrajectoryAlignment,
        path: impl AsRef<Path>,
    ) -> std::io::Result<()> {
        std::fs::write(
            path,
            self.to_html_report_against_with_alignment(reference, alignment),
        )
    }

    pub fn summary(&self) -> TrajectorySummary {
        TrajectorySummary::from_trajectory(self)
    }

    pub fn to_summary_json(&self) -> String {
        self.summary().to_json()
    }

    pub fn write_summary_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_summary_json())
    }
}

#[cfg(test)]
mod umeyama_alignment_tests {
    use super::*;
    use nalgebra::{Point3, Rotation3, UnitQuaternion, Vector3};

    fn make_source() -> Vec<Point3<f64>> {
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(2.5, -1.0, 0.5),
        ]
    }

    fn transform_points(
        points: &[Point3<f64>],
        scale: f64,
        rotation: &Rotation3<f64>,
        translation: &Vector3<f64>,
    ) -> Vec<Point3<f64>> {
        points
            .iter()
            .map(|p| Point3::from(rotation * (p.coords * scale) + translation))
            .collect()
    }

    #[test]
    fn umeyama_recovers_pure_translation() {
        let source = make_source();
        let translation = Vector3::new(5.0, -2.0, 1.5);
        let target = transform_points(&source, 1.0, &Rotation3::identity(), &translation);
        let transform = umeyama_similarity_transform(&source, &target, false).expect("transform");
        assert!((transform.scale - 1.0).abs() < 1e-12);
        assert!((transform.translation - translation).norm() < 1e-10);
        assert!(transform.rotation.matrix().relative_eq(
            Rotation3::<f64>::identity().matrix(),
            1e-10,
            1e-10
        ));
    }

    #[test]
    fn umeyama_recovers_rotation_and_translation() {
        let source = make_source();
        let rotation = Rotation3::from_euler_angles(0.3, -0.4, 0.5);
        let translation = Vector3::new(-1.0, 2.0, -3.0);
        let target = transform_points(&source, 1.0, &rotation, &translation);
        let transform = umeyama_similarity_transform(&source, &target, false).expect("transform");
        assert!((transform.scale - 1.0).abs() < 1e-10);
        assert!((transform.translation - translation).norm() < 1e-8);
        assert!(transform
            .rotation
            .matrix()
            .relative_eq(rotation.matrix(), 1e-8, 1e-8));
        let aligned: Vec<Point3<f64>> = source.iter().map(|p| transform.apply(p)).collect();
        for (a, t) in aligned.iter().zip(target.iter()) {
            assert!((a - t).norm() < 1e-8);
        }
    }

    #[test]
    fn umeyama_recovers_similarity_with_scale() {
        let source = make_source();
        let scale = 3.4;
        let rotation = Rotation3::from_euler_angles(-0.2, 0.8, 0.1);
        let translation = Vector3::new(2.0, 0.5, -1.0);
        let target = transform_points(&source, scale, &rotation, &translation);
        let transform = umeyama_similarity_transform(&source, &target, true).expect("transform");
        assert!((transform.scale - scale).abs() < 1e-8);
        assert!((transform.translation - translation).norm() < 1e-7);
        assert!(transform
            .rotation
            .matrix()
            .relative_eq(rotation.matrix(), 1e-8, 1e-8));
    }

    #[test]
    fn umeyama_returns_none_for_insufficient_points() {
        let source = vec![Point3::new(0.0, 0.0, 0.0)];
        let target = vec![Point3::new(1.0, 0.0, 0.0)];
        assert!(umeyama_similarity_transform(&source, &target, false).is_none());
    }

    #[test]
    fn umeyama_returns_none_for_zero_variance_source() {
        let source = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0)];
        let target = vec![Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)];
        assert!(umeyama_similarity_transform(&source, &target, false).is_none());
    }

    fn pose_with_camera_center(frame_id: FrameId, center: Point3<f64>) -> TrajectorySample {
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -center.coords);
        TrajectorySample {
            frame_id,
            pose,
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            inlier_count: 0,
            inlier_ratio: 0.0,
            reprojection_error: None,
        }
    }

    fn build_trajectory(centers: &[(FrameId, Point3<f64>)]) -> PoseTrajectory {
        let mut trajectory = PoseTrajectory::new();
        for (id, p) in centers {
            trajectory.push_sample(pose_with_camera_center(*id, *p));
        }
        trajectory
    }

    #[test]
    fn trajectory_alignment_umeyama_drives_error_to_zero_for_similarity_perturbation() {
        let scale = 2.0;
        let rotation = Rotation3::from_euler_angles(0.1, -0.2, 0.3);
        let translation = Vector3::new(7.0, -4.0, 2.0);
        let reference_centers: Vec<(FrameId, Point3<f64>)> = (0..6)
            .map(|i| {
                let p = Point3::new(i as f64, (i as f64).sin(), (i as f64).cos());
                (i as FrameId, p)
            })
            .collect();
        let estimated_centers: Vec<(FrameId, Point3<f64>)> = reference_centers
            .iter()
            .map(|(id, p)| {
                let inv_translation = -translation;
                let inv_rotation = rotation.inverse();
                let inv_scale = 1.0 / scale;
                let pre = p.coords + inv_translation;
                let body = inv_rotation * pre * inv_scale;
                (*id, Point3::from(body))
            })
            .collect();
        let reference = build_trajectory(&reference_centers);
        let estimated = build_trajectory(&estimated_centers);

        let aligned_summary = estimated.translation_error_summary_against_with_alignment(
            &reference,
            TrajectoryAlignment::UmeyamaWithScale,
        );
        let rmse = aligned_summary.rmse_translation_error.expect("rmse");
        assert!(
            rmse < 1e-7,
            "expected near-zero ATE after similarity alignment, got {rmse}"
        );

        let raw_summary = estimated.translation_error_summary_against_with_alignment(
            &reference,
            TrajectoryAlignment::None,
        );
        let raw_rmse = raw_summary.rmse_translation_error.expect("raw rmse");
        assert!(
            raw_rmse > 1.0,
            "expected substantial raw error before alignment, got {raw_rmse}"
        );
    }

    #[test]
    fn trajectory_alignment_umeyama_rigid_does_not_remove_scale() {
        let reference_centers: Vec<(FrameId, Point3<f64>)> = (0..5)
            .map(|i| (i as FrameId, Point3::new(i as f64, 0.0, 0.0)))
            .collect();
        let estimated_centers: Vec<(FrameId, Point3<f64>)> = reference_centers
            .iter()
            .map(|(id, p)| (*id, Point3::from(p.coords * 0.5)))
            .collect();
        let reference = build_trajectory(&reference_centers);
        let estimated = build_trajectory(&estimated_centers);

        let rigid = estimated.translation_error_summary_against_with_alignment(
            &reference,
            TrajectoryAlignment::Umeyama,
        );
        let scaled = estimated.translation_error_summary_against_with_alignment(
            &reference,
            TrajectoryAlignment::UmeyamaWithScale,
        );
        let rigid_rmse = rigid.rmse_translation_error.expect("rigid rmse");
        let scaled_rmse = scaled.rmse_translation_error.expect("scaled rmse");
        assert!(
            scaled_rmse < 1e-8,
            "expected near-zero ATE after similarity alignment, got {scaled_rmse}"
        );
        assert!(
            rigid_rmse > 0.1,
            "expected non-trivial residual under rigid alignment of half-scale trajectory, got {rigid_rmse}"
        );
    }

    fn rpe_sample(frame_id: FrameId, camera_to_world: SE3) -> TrajectorySample {
        let world_to_camera = camera_to_world.inverse();
        TrajectorySample {
            frame_id,
            pose: Pose::from_world_to_camera(world_to_camera.rotation, world_to_camera.translation),
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            inlier_count: 0,
            inlier_ratio: 0.0,
            reprojection_error: None,
        }
    }

    fn rpe_trajectory(poses: &[(FrameId, SE3)]) -> PoseTrajectory {
        let mut trajectory = PoseTrajectory::new();
        for (id, camera_to_world) in poses {
            trajectory.push_sample(rpe_sample(*id, camera_to_world.clone()));
        }
        trajectory
    }

    #[test]
    fn relative_pose_error_is_zero_when_estimate_matches_reference() {
        let poses: Vec<(FrameId, SE3)> = (0..8)
            .map(|i| {
                let c2w = SE3::new(
                    UnitQuaternion::from_euler_angles(0.0, 0.0, 0.05 * i as f64),
                    Vector3::new(i as f64, (i as f64 * 0.3).sin(), 0.0),
                );
                (i as FrameId, c2w)
            })
            .collect();
        let trajectory = rpe_trajectory(&poses);

        let summary = trajectory
            .relative_pose_error_against(&trajectory, &RelativePoseErrorConfig::default());

        assert_eq!(summary.pair_count, 7);
        let translation = summary.translation.expect("translation stats");
        let rotation = summary.rotation_deg.expect("rotation stats");
        assert!(
            translation.rmse < 1e-9,
            "translation rmse {}",
            translation.rmse
        );
        assert!(rotation.rmse < 1e-9, "rotation rmse {}", rotation.rmse);
    }

    #[test]
    fn relative_pose_error_is_invariant_to_a_global_rigid_transform() {
        // RPE compares *relative* motion, which is unchanged by left-multiplying
        // every pose by a fixed rigid transform — so a globally displaced and
        // rotated estimate scores ~0 RPE while its raw (unaligned) ATE is large.
        let global = SE3::new(
            UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3),
            Vector3::new(7.0, -4.0, 2.0),
        );
        let reference_poses: Vec<(FrameId, SE3)> = (0..10)
            .map(|i| {
                let c2w = SE3::new(
                    UnitQuaternion::from_euler_angles(0.0, 0.0, 0.1 * i as f64),
                    Vector3::new(i as f64, (i as f64).sin(), (i as f64).cos()),
                );
                (i as FrameId, c2w)
            })
            .collect();
        let estimated_poses: Vec<(FrameId, SE3)> = reference_poses
            .iter()
            .map(|(id, c2w)| (*id, global.compose(c2w)))
            .collect();
        let reference = rpe_trajectory(&reference_poses);
        let estimated = rpe_trajectory(&estimated_poses);

        let summary = estimated.relative_pose_error_against(
            &reference,
            &RelativePoseErrorConfig {
                delta: 2,
                start_step: 1,
            },
        );
        assert_eq!(summary.delta, 2);
        let translation = summary.translation.expect("translation stats");
        let rotation = summary.rotation_deg.expect("rotation stats");
        assert!(
            translation.rmse < 1e-9 && rotation.rmse < 1e-9,
            "RPE should vanish under a global rigid transform, got t={} r={}",
            translation.rmse,
            rotation.rmse
        );

        // Contrast: the unaligned absolute trajectory error is far from zero.
        let ate = estimated
            .translation_error_summary_against_with_alignment(&reference, TrajectoryAlignment::None)
            .rmse_translation_error
            .expect("ate rmse");
        assert!(
            ate > 1.0,
            "expected large raw ATE for the displaced estimate, got {ate}"
        );
    }

    #[test]
    fn relative_pose_error_recovers_a_constant_per_step_drift() {
        // Reference advances one meter per step with no rotation. The estimate's
        // per-step relative motion carries a fixed extra transform `drift`, so
        // every residual equals `drift`: translation 0.1 m, rotation 5 deg.
        let step = SE3::new(UnitQuaternion::identity(), Vector3::new(1.0, 0.0, 0.0));
        let drift = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 5.0_f64.to_radians()),
            Vector3::new(0.1, 0.0, 0.0),
        );

        let mut reference_poses = vec![(0u64, SE3::identity())];
        let mut estimated_poses = vec![(0u64, SE3::identity())];
        for i in 1..7u64 {
            let previous_reference = reference_poses[(i - 1) as usize].1.clone();
            reference_poses.push((i, previous_reference.compose(&step)));
            let previous_estimate = estimated_poses[(i - 1) as usize].1.clone();
            estimated_poses.push((i, previous_estimate.compose(&step).compose(&drift)));
        }
        let reference = rpe_trajectory(&reference_poses);
        let estimated = rpe_trajectory(&estimated_poses);

        let summary =
            estimated.relative_pose_error_against(&reference, &RelativePoseErrorConfig::default());
        let translation = summary.translation.expect("translation stats");
        let rotation = summary.rotation_deg.expect("rotation stats");
        // Tolerances absorb float accumulation through the multi-step compose
        // chain and quaternion angle extraction.
        assert!(
            (translation.mean - 0.1).abs() < 1e-6,
            "translation mean {}",
            translation.mean
        );
        assert!(
            translation.std < 1e-6,
            "translation std {}",
            translation.std
        );
        assert!(
            (rotation.mean - 5.0).abs() < 1e-6,
            "rotation mean {}",
            rotation.mean
        );
        assert!(rotation.std < 1e-6, "rotation std {}", rotation.std);
    }

    #[test]
    fn relative_pose_error_has_no_pairs_when_delta_exceeds_overlap() {
        let poses: Vec<(FrameId, SE3)> = (0..3)
            .map(|i| {
                (
                    i as FrameId,
                    SE3::new(UnitQuaternion::identity(), Vector3::new(i as f64, 0.0, 0.0)),
                )
            })
            .collect();
        let trajectory = rpe_trajectory(&poses);
        let summary = trajectory.relative_pose_error_against(
            &trajectory,
            &RelativePoseErrorConfig {
                delta: 5,
                start_step: 1,
            },
        );
        assert_eq!(summary.pair_count, 0);
        assert!(summary.translation.is_none());
        assert!(summary.rotation_deg.is_none());
    }

    fn fake_success_tracking_result(frame_id: u64) -> TrackingResult {
        fake_success_tracking_result_with_pose(
            frame_id,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros()),
        )
    }

    fn fake_success_tracking_result_with_pose(frame_id: u64, pose: Pose) -> TrackingResult {
        use visloc_core::types::LocalizationSuccess;
        let localization = LocalizationResult::success(LocalizationSuccess {
            pose,
            candidate_landmark_count: 4,
            match_count: 4,
            correspondence_count: 4,
            inliers: vec![0, 1, 2, 3],
            inlier_query_indices: vec![0, 1, 2, 3],
            inlier_landmark_ids: vec![1, 2, 3, 4],
            inlier_reprojection_errors: vec![0.0; 4],
            mean_reprojection_error: 0.0,
            median_reprojection_error: 0.0,
            max_reprojection_error: 0.0,
        });
        TrackingResult {
            frame_id,
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            successive_failures: 0,
            pose_prior: None,
            used_pose_prior: false,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason: None,
            map_landmark_count: 0,
            map_stats: MapProviderStats::default(),
            localization,
            covisibility_local_map_size: None,
        }
    }

    fn fake_failure_tracking_result(frame_id: u64) -> TrackingResult {
        let localization = LocalizationResult::failure(
            visloc_core::types::LocalizationFailureReason::QualityGateFailed,
            0,
            0,
            0,
        );
        TrackingResult {
            frame_id,
            state: TrackingState::Tracking,
            event: TrackingEvent::TrackingFailed,
            successive_failures: 1,
            pose_prior: None,
            used_pose_prior: false,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason: Some(TrackingFailureReason::InsufficientInliers {
                inlier_count: 0,
                min_inliers: 10,
            }),
            map_landmark_count: 0,
            map_stats: MapProviderStats::default(),
            localization,
            covisibility_local_map_size: None,
        }
    }

    #[test]
    fn adaptive_motion_starts_in_imu_mode() {
        let model = AdaptiveImuPoseMotionModel::with_defaults();
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_pose(), 0);
        assert_eq!(model.switches_to_imu(), 0);
    }

    #[test]
    fn adaptive_motion_switches_to_pose_after_consecutive_failures() {
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 2,
                successes_to_switch_to_imu: 5,
                ..AdaptiveImuPoseMotionModelConfig::default()
            },
        );
        model.observe(&fake_failure_tracking_result(1));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        model.observe(&fake_failure_tracking_result(2));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        assert_eq!(model.switches_to_pose(), 1);
    }

    #[test]
    fn adaptive_motion_switches_back_to_imu_after_consecutive_successes() {
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 3,
                ..AdaptiveImuPoseMotionModelConfig::default()
            },
        );
        model.observe(&fake_failure_tracking_result(1));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        // Three consecutive successes flip back to IMU.
        model.observe(&fake_success_tracking_result(2));
        model.observe(&fake_success_tracking_result(3));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        model.observe(&fake_success_tracking_result(4));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_pose(), 1);
        assert_eq!(model.switches_to_imu(), 1);
    }

    #[test]
    fn adaptive_motion_intermittent_failures_reset_pose_recovery_counter() {
        // While in Pose mode, a stray failure resets the
        // success-streak counter so the wrapper does not prematurely
        // switch back to IMU on a noisy regime.
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 3,
                ..AdaptiveImuPoseMotionModelConfig::default()
            },
        );
        model.observe(&fake_failure_tracking_result(1));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        model.observe(&fake_success_tracking_result(2));
        model.observe(&fake_success_tracking_result(3));
        model.observe(&fake_failure_tracking_result(4));
        model.observe(&fake_success_tracking_result(5));
        model.observe(&fake_success_tracking_result(6));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        model.observe(&fake_success_tracking_result(7));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
    }

    #[test]
    fn adaptive_motion_reset_clears_counters_and_mode() {
        let mut model = AdaptiveImuPoseMotionModel::with_defaults();
        // Force a switch.
        for f in 1..=3 {
            model.observe(&fake_failure_tracking_result(f));
        }
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        model.reset();
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_pose(), 0);
        assert_eq!(model.switches_to_imu(), 0);
    }

    // ---- Phase-24: IMU-velocity-refresh-on-switch-to-IMU ---------------

    /// World-frame camera-centre pose helper for adaptive Phase-24
    /// tests. `from_world_to_camera` takes `translation = -R · centre`,
    /// so for an identity rotation we negate the supplied `centre`.
    fn pose_at_world_centre(centre: Vector3<f64>) -> Pose {
        Pose::from_world_to_camera(UnitQuaternion::identity(), -centre)
    }

    #[test]
    fn adaptive_motion_refresh_imu_velocity_on_switch_recomputes_from_pose_diff() {
        // Refresh-on-switch enabled (the Phase-24 default). Drive a
        // failure to enter Pose mode, then two successes (each preceded
        // by an IMU sample so the pending-dt sum is well-defined). At
        // the second success we trip the switch-back threshold and the
        // refresh hook should rewrite IMU velocity_world to the visual
        // finite-difference (camera body moved +0.4m along world-x in
        // 0.1s ⇒ velocity_world ≈ (4, 0, 0)).
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 2,
                imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::FiniteDifference,
            },
        );
        // Seed a non-zero stale velocity so we can prove the refresh
        // overwrote it.
        model
            .imu_mut()
            .set_velocity_world(Vector3::new(99.0, 0.0, 0.0));
        model.observe(&fake_failure_tracking_result(1));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        // Push one IMU sample with dt=0.1, then a successful obs at the
        // origin → previous_successful_pose = origin, dt = 0.1.
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            2,
            pose_at_world_centre(Vector3::zeros()),
        ));
        // Push another IMU sample with dt=0.1, then a successful obs
        // displaced +0.4m along world-x. This is the second success →
        // mode switches back to IMU and the refresh hook fires.
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            3,
            pose_at_world_centre(Vector3::new(0.4, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_imu(), 1);
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 1);
        let v = model.imu().velocity_world();
        assert!(
            (v.x - 4.0).abs() < 1.0e-9 && v.y.abs() < 1.0e-9 && v.z.abs() < 1.0e-9,
            "refresh hook should overwrite stale velocity_world with visual \
             finite-difference (0.4 m / 0.1 s = 4 m/s along x); got {v:?}",
        );
    }

    #[test]
    fn adaptive_motion_refresh_disabled_leaves_imu_velocity_stale_on_switch() {
        // Same sequence as the previous test, but with the refresh
        // flag disabled. The seeded stale velocity_world must survive
        // the switch-back-to-IMU transition unchanged (this recovers
        // the Phase-23 #4 no-refresh behavior for A/B comparison).
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 2,
                imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::None,
            },
        );
        model
            .imu_mut()
            .set_velocity_world(Vector3::new(99.0, 0.0, 0.0));
        model.observe(&fake_failure_tracking_result(1));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            2,
            pose_at_world_centre(Vector3::zeros()),
        ));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            3,
            pose_at_world_centre(Vector3::new(0.4, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_imu(), 1);
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 0);
        let v = model.imu().velocity_world();
        assert_eq!(
            v,
            Vector3::new(99.0, 0.0, 0.0),
            "refresh disabled: stale velocity_world must survive the \
             switch-back unchanged; got {v:?}",
        );
    }

    #[test]
    fn adaptive_motion_reset_clears_refresh_state_and_counters() {
        // reset() must clear the Phase-24 recent-pose tracking state
        // (previous/latest successful pose, dt between them, and the
        // refresh counter) on top of the existing reset semantics.
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 2,
                imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::FiniteDifference,
            },
        );
        model.observe(&fake_failure_tracking_result(1));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            2,
            pose_at_world_centre(Vector3::zeros()),
        ));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            3,
            pose_at_world_centre(Vector3::new(0.4, 0.0, 0.0)),
        ));
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 1);
        model.reset();
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 0);
        // After reset, a fresh failure → switch sequence must NOT
        // trigger a refresh (no recent poses remembered).
        model.observe(&fake_failure_tracking_result(10));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            11,
            pose_at_world_centre(Vector3::zeros()),
        ));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        // This second success triggers a switch-back. At that moment
        // we only have ONE post-reset success on record before this
        // one — previous_successful_pose was set by frame 11 and
        // latest_successful_pose will be set by frame 12. Both
        // present + positive dt ⇒ refresh DOES fire. The point of
        // this assertion is that the per-instance counter restarts
        // from zero after reset(), not that refreshes stop.
        model.observe(&fake_success_tracking_result_with_pose(
            12,
            pose_at_world_centre(Vector3::new(0.2, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 1);
    }

    // ---- Phase-25: ZeroReset + ThreePoseSmoother A/B variants ----------

    #[test]
    fn adaptive_motion_refresh_zero_reset_zeroes_imu_velocity_on_switch() {
        // The ZeroReset policy must overwrite the IMU's
        // `velocity_world` with the zero vector at every switch-back,
        // independent of whether any visual poses are on record. The
        // counter must increment because the policy fired.
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 2,
                imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::ZeroReset,
            },
        );
        model
            .imu_mut()
            .set_velocity_world(Vector3::new(99.0, 0.0, 0.0));
        model.observe(&fake_failure_tracking_result(1));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            2,
            pose_at_world_centre(Vector3::zeros()),
        ));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            3,
            pose_at_world_centre(Vector3::new(0.4, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_imu(), 1);
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 1);
        let v = model.imu().velocity_world();
        assert_eq!(
            v,
            Vector3::zeros(),
            "ZeroReset policy must overwrite stale velocity_world with \
             zeros at switch; got {v:?}",
        );
    }

    #[test]
    fn adaptive_motion_refresh_three_pose_smoother_averages_two_finite_diffs() {
        // Three successive successful observations along world-x at
        //   t0: pose @ x=0          dt = irrelevant (first success)
        //   t1: pose @ x=0.4   dt1 = 0.1   ⇒ v_prev   = 4 m/s
        //   t2: pose @ x=1.2   dt2 = 0.1   ⇒ v_latest = 8 m/s
        // ThreePoseSmoother averages them ⇒ writes (4+8)/2 = 6 m/s.
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 3,
                imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::ThreePoseSmoother,
            },
        );
        model
            .imu_mut()
            .set_velocity_world(Vector3::new(99.0, 0.0, 0.0));
        model.observe(&fake_failure_tracking_result(1));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        // 1st success: oldest=None, prev=None, latest=pose@0
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            2,
            pose_at_world_centre(Vector3::zeros()),
        ));
        // 2nd success: oldest=None, prev=pose@0, latest=pose@0.4
        // dt_latest=0.1, dt_prev=0.1 (carried from the prior tick — but
        // there was no previous pose so the smoother's oldest branch
        // does not trigger). Counter still 0 (no switch yet).
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            3,
            pose_at_world_centre(Vector3::new(0.4, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Pose);
        // 3rd success: oldest=pose@0, prev=pose@0.4, latest=pose@1.2.
        // dt_prev=0.1 (was dt_latest one shift ago), dt_latest=0.1.
        // consecutive_successes reaches threshold → switch + smoother
        // fires with both 3-pose entries available.
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            4,
            pose_at_world_centre(Vector3::new(1.2, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.switches_to_imu(), 1);
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 1);
        let v = model.imu().velocity_world();
        assert!(
            (v.x - 6.0).abs() < 1.0e-9 && v.y.abs() < 1.0e-9 && v.z.abs() < 1.0e-9,
            "ThreePoseSmoother should average two finite-diffs \
             (4 m/s and 8 m/s along x ⇒ 6 m/s); got {v:?}",
        );
    }

    #[test]
    fn adaptive_motion_refresh_three_pose_smoother_falls_back_when_only_two_poses() {
        // Same setup as the FiniteDifference test (only TWO successful
        // poses before the switch), but with the policy set to
        // ThreePoseSmoother. The fallback must compute a single
        // finite-difference (4 m/s along x) rather than no-op.
        let mut model = AdaptiveImuPoseMotionModel::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig {
                failures_to_switch_to_pose: 1,
                successes_to_switch_to_imu: 2,
                imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::ThreePoseSmoother,
            },
        );
        model
            .imu_mut()
            .set_velocity_world(Vector3::new(99.0, 0.0, 0.0));
        model.observe(&fake_failure_tracking_result(1));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            2,
            pose_at_world_centre(Vector3::zeros()),
        ));
        model
            .imu_mut()
            .push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        model.observe(&fake_success_tracking_result_with_pose(
            3,
            pose_at_world_centre(Vector3::new(0.4, 0.0, 0.0)),
        ));
        assert_eq!(model.mode(), AdaptiveMotionMode::Imu);
        assert_eq!(model.velocity_refreshes_on_switch_to_imu(), 1);
        let v = model.imu().velocity_world();
        assert!(
            (v.x - 4.0).abs() < 1.0e-9 && v.y.abs() < 1.0e-9 && v.z.abs() < 1.0e-9,
            "ThreePoseSmoother fallback (only 2 poses) should match \
             single finite-difference; got {v:?}",
        );
    }
}
