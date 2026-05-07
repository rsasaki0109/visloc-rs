use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{
    parse_two_view_matches_txt, EssentialMatrixVisualOdometryConfig,
    EssentialMatrixVisualOdometryFrontend, InMemoryMapProvider, LocalizationPipeline,
    LocalizationPrior, Tracker, TrackingConfig, TrackingStats, TwoViewMatchVisualOdometryConfig,
    TwoViewMatchVisualOdometryFrontend, VisualOdometryFrontend, VisualOdometryPriorProvider,
};

const VO_PRIOR_RADIUS: f64 = 8.0;
const FRAME_PAIRS: [(u64, u64); 2] = [(100, 101), (101, 102)];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let output_dir = parse_output_dir(&mut args);
    if !args.is_empty() {
        eprintln!("usage: cargo run --example two_view_vo_compare -- [--out-dir <dir>]");
        std::process::exit(2);
    }

    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    // Map points span depth and X/Y so the essential-matrix solver gets a
    // well-conditioned configuration; for the same reason DLT PnP tracking
    // also benefits from non-coplanar map points.
    let map_points = [
        Point3::new(-1.0, -1.0, 4.5),
        Point3::new(1.0, -1.0, 4.6),
        Point3::new(-1.0, 1.0, 5.5),
        Point3::new(1.0, 1.0, 5.4),
        Point3::new(0.0, 0.0, 5.0),
        Point3::new(0.5, -0.25, 6.0),
        Point3::new(-0.6, 0.4, 4.8),
        Point3::new(0.4, 0.7, 5.2),
    ];
    let frame_specs: [(u64, Pose); 3] = [
        (
            100,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0)),
        ),
        (
            101,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.30, 0.0, 0.0)),
        ),
        (
            102,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.60, 0.0, -0.05)),
        ),
    ];
    let frames: Vec<Frame> = frame_specs
        .iter()
        .map(|(frame_id, pose)| {
            frame_from_projected_landmarks(*frame_id, &camera, pose, &map_points)
        })
        .collect();
    let provider = InMemoryMapProvider::new(build_map(&camera, &map_points));

    let two_view_matches = build_two_view_match_strings(&frames);
    let truth_translations = relative_truth_translations(&frame_specs);

    let flow_diagnostics =
        run_with_flow_frontend(&frames, &provider, &two_view_matches, &truth_translations);
    let essential_diagnostics = run_with_essential_frontend(
        &frames,
        &provider,
        &camera,
        &two_view_matches,
        &truth_translations,
    );

    println!("== Flow-only frontend (TwoViewMatchVisualOdometryFrontend) ==");
    print_diagnostics(&flow_diagnostics);
    println!("== Essential-matrix frontend (EssentialMatrixVisualOdometryFrontend) ==");
    print_diagnostics(&essential_diagnostics);

    if let Some(output_dir) = output_dir {
        fs::create_dir_all(&output_dir)?;
        let report_path = output_dir.join("two_view_vo_compare.txt");
        write_report(
            &report_path,
            &flow_diagnostics,
            &essential_diagnostics,
            &truth_translations,
        )?;
        println!(
            "wrote two-view VO comparison report: {}",
            report_path.display()
        );
    }

    Ok(())
}

#[derive(Debug)]
struct FrameDiagnostics {
    frame_id: u64,
    used_vo_prior: bool,
    match_count: usize,
    inlier_count: usize,
    estimated_relative_translation: Option<Vector3<f64>>,
    truth_relative_translation: Option<Vector3<f64>>,
    candidate_landmark_count: usize,
    localization_succeeded: bool,
    estimated_camera_center: Option<Point3<f64>>,
}

#[derive(Debug)]
struct RunDiagnostics {
    label: &'static str,
    frames: Vec<FrameDiagnostics>,
    stats: TrackingStats,
}

fn run_with_flow_frontend(
    frames: &[Frame],
    provider: &InMemoryMapProvider,
    two_view_matches: &[(u64, u64, String)],
    truth_translations: &[((u64, u64), Vector3<f64>)],
) -> RunDiagnostics {
    let mut frontend = TwoViewMatchVisualOdometryFrontend::new(TwoViewMatchVisualOdometryConfig {
        min_matches: 6,
        min_inliers: 6,
        max_residual_pixels: 4.0,
        pixel_translation_scale: 0.01,
        forward_translation: 0.0,
    });
    for (previous_id, current_id, contents) in two_view_matches {
        let matches =
            parse_two_view_matches_txt(contents).expect("flow frontend matches must parse");
        frontend.insert_matches(*previous_id, *current_id, matches);
    }
    drive_tracker_with_frontend("flow-only", frontend, frames, provider, truth_translations)
}

fn run_with_essential_frontend(
    frames: &[Frame],
    provider: &InMemoryMapProvider,
    camera: &Camera,
    two_view_matches: &[(u64, u64, String)],
    truth_translations: &[((u64, u64), Vector3<f64>)],
) -> RunDiagnostics {
    let mut frontend = EssentialMatrixVisualOdometryFrontend::new(
        camera.clone(),
        EssentialMatrixVisualOdometryConfig {
            ransac_iterations: 256,
            sampson_threshold: 5.0e-3,
            ransac_seed: 7,
            default_translation_scale: 1.0,
            min_inliers: 6,
        },
    );
    for (previous_id, current_id, contents) in two_view_matches {
        let matches =
            parse_two_view_matches_txt(contents).expect("essential frontend matches must parse");
        let scale = truth_translations
            .iter()
            .find(|((p, c), _)| *p == *previous_id && *c == *current_id)
            .map(|(_, translation)| translation.norm())
            .unwrap_or(1.0);
        frontend.insert_matches_with_scale(*previous_id, *current_id, matches, scale);
    }
    drive_tracker_with_frontend(
        "essential-matrix",
        frontend,
        frames,
        provider,
        truth_translations,
    )
}

fn drive_tracker_with_frontend<F>(
    label_text: &str,
    frontend: F,
    frames: &[Frame],
    provider: &InMemoryMapProvider,
    truth_translations: &[((u64, u64), Vector3<f64>)],
) -> RunDiagnostics
where
    F: VisualOdometryFrontend<Error = std::convert::Infallible>,
{
    let label: &'static str = match label_text {
        "flow-only" => "flow-only",
        "essential-matrix" => "essential-matrix",
        _ => "unknown",
    };
    let provider_vo = VisualOdometryPriorProvider::new(frontend);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());
    let mut previous_frame: Option<&Frame> = None;
    let mut previous_pose: Option<Pose> = None;
    let mut diagnostics: Vec<FrameDiagnostics> = Vec::new();

    for frame in frames {
        let vo_prior =
            previous_frame
                .zip(previous_pose.as_ref())
                .and_then(|(prev_frame, prev_pose)| {
                    provider_vo
                        .predict_pose_prior(prev_frame, prev_pose, frame)
                        .expect("two-view frontends are infallible")
                });

        let truth_translation = previous_frame.and_then(|prev| {
            truth_translations
                .iter()
                .find(|((p, c), _)| *p == prev.id && *c == frame.id)
                .map(|(_, translation)| *translation)
        });

        let (result, diagnostic) = if let Some(vo_prior) = vo_prior {
            let prior = LocalizationPrior::from_pose(vo_prior.pose.clone(), VO_PRIOR_RADIUS);
            let result = tracker
                .track_frame_with_localization_prior_submap_provider(frame, provider, &prior);
            let diagnostic = FrameDiagnostics {
                frame_id: result.frame_id,
                used_vo_prior: true,
                match_count: vo_prior.estimate.match_count,
                inlier_count: vo_prior.estimate.inlier_count,
                estimated_relative_translation: Some(
                    vo_prior.estimate.previous_to_current.translation,
                ),
                truth_relative_translation: truth_translation,
                candidate_landmark_count: result.localization.candidate_landmark_count,
                localization_succeeded: result.localization.success,
                estimated_camera_center: result
                    .localization
                    .pose
                    .as_ref()
                    .map(Pose::camera_center_world),
            };
            (result, diagnostic)
        } else {
            let result = tracker.track_frame_with_provider(frame, provider);
            let diagnostic = FrameDiagnostics {
                frame_id: result.frame_id,
                used_vo_prior: false,
                match_count: 0,
                inlier_count: 0,
                estimated_relative_translation: None,
                truth_relative_translation: truth_translation,
                candidate_landmark_count: result.localization.candidate_landmark_count,
                localization_succeeded: result.localization.success,
                estimated_camera_center: result
                    .localization
                    .pose
                    .as_ref()
                    .map(Pose::camera_center_world),
            };
            (result, diagnostic)
        };

        previous_frame = Some(frame);
        previous_pose = result.localization.pose.clone();
        diagnostics.push(diagnostic);
    }

    let stats = tracker.stats().clone();
    RunDiagnostics {
        label,
        frames: diagnostics,
        stats,
    }
}

fn build_map(camera: &Camera, points: &[Point3<f64>]) -> VisualMap {
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 9.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor);
        map.landmarks.insert(landmark.id, landmark);
    }
    map
}

fn frame_from_projected_landmarks(
    frame_id: u64,
    camera: &Camera,
    pose: &Pose,
    points: &[Point3<f64>],
) -> Frame {
    let mut frame = Frame::new(frame_id, camera.id);
    for (index, point) in points.iter().enumerate() {
        let keypoint = camera
            .project(&pose.transform_world_point(point))
            .expect("synthetic point must project in front of the camera");
        frame.keypoints.push(keypoint);
        frame.descriptors.push(vec![index as f32, 9.0]);
    }
    frame
}

fn build_two_view_match_strings(frames: &[Frame]) -> Vec<(u64, u64, String)> {
    let mut result = Vec::new();
    for (previous_id, current_id) in FRAME_PAIRS {
        let previous = frames
            .iter()
            .find(|frame| frame.id == previous_id)
            .expect("previous frame must exist");
        let current = frames
            .iter()
            .find(|frame| frame.id == current_id)
            .expect("current frame must exist");
        let mut contents = String::from("# PREV_IDX CURR_IDX PREV_X PREV_Y CURR_X CURR_Y\n");
        let count = previous.keypoints.len().min(current.keypoints.len());
        for index in 0..count {
            let prev_xy = previous.keypoints[index];
            let curr_xy = current.keypoints[index];
            writeln!(
                contents,
                "{index} {index} {:.4} {:.4} {:.4} {:.4}",
                prev_xy.x, prev_xy.y, curr_xy.x, curr_xy.y
            )
            .ok();
        }
        result.push((previous_id, current_id, contents));
    }
    result
}

fn relative_truth_translations(frame_specs: &[(u64, Pose)]) -> Vec<((u64, u64), Vector3<f64>)> {
    let mut translations = Vec::new();
    for window in frame_specs.windows(2) {
        let (prev_id, prev_pose) = (window[0].0, &window[0].1);
        let (curr_id, curr_pose) = (window[1].0, &window[1].1);
        // previous_to_current.translation = curr.t - rotation * prev.t with rotation = curr_R * prev_R^-1.
        // For identity rotations (this synthetic setup), it reduces to curr.t - prev.t.
        let prev_t = prev_pose.world_to_camera.translation;
        let curr_t = curr_pose.world_to_camera.translation;
        let relative_translation = curr_t - prev_t;
        translations.push(((prev_id, curr_id), relative_translation));
    }
    translations
}

fn print_diagnostics(run: &RunDiagnostics) {
    for frame in &run.frames {
        let estimated = frame
            .estimated_relative_translation
            .map(format_vector)
            .unwrap_or_else(|| "n/a".to_string());
        let truth = frame
            .truth_relative_translation
            .map(format_vector)
            .unwrap_or_else(|| "n/a".to_string());
        let center = frame
            .estimated_camera_center
            .map(|center| format!("[{:.3}, {:.3}, {:.3}]", center.x, center.y, center.z))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "frame={} vo_prior={} matches={} inliers={} relative_t_estimated={} relative_t_truth={} candidates={} success={} center={}",
            frame.frame_id,
            frame.used_vo_prior,
            frame.match_count,
            frame.inlier_count,
            estimated,
            truth,
            frame.candidate_landmark_count,
            frame.localization_succeeded,
            center,
        );
    }
    println!(
        "stats[{}] frames={} success_rate={:.3} external_prior_rate={:.3} external_prior_count={}",
        run.label,
        run.stats.frame_count,
        run.stats.success_rate(),
        run.stats.external_localization_prior_usage_rate(),
        run.stats.external_localization_prior_used_count,
    );
}

fn write_report(
    path: &Path,
    flow: &RunDiagnostics,
    essential: &RunDiagnostics,
    truth_translations: &[((u64, u64), Vector3<f64>)],
) -> std::io::Result<()> {
    let mut report = String::new();
    let _ = writeln!(report, "two-view VO comparison demo report");
    let _ = writeln!(report, "ground-truth relative translations:");
    for ((previous_id, current_id), translation) in truth_translations {
        let _ = writeln!(
            report,
            "  {previous_id} -> {current_id}: [{:.3}, {:.3}, {:.3}]",
            translation.x, translation.y, translation.z
        );
    }
    write_run(&mut report, flow);
    write_run(&mut report, essential);
    fs::write(path, report)
}

fn write_run(report: &mut String, run: &RunDiagnostics) {
    let _ = writeln!(report, "frontend={}", run.label);
    for frame in &run.frames {
        let estimated = frame
            .estimated_relative_translation
            .map(format_vector)
            .unwrap_or_else(|| "n/a".to_string());
        let truth = frame
            .truth_relative_translation
            .map(format_vector)
            .unwrap_or_else(|| "n/a".to_string());
        let center = frame
            .estimated_camera_center
            .map(|center| format!("[{:.3}, {:.3}, {:.3}]", center.x, center.y, center.z))
            .unwrap_or_else(|| "n/a".to_string());
        let _ = writeln!(
            report,
            "  frame={} vo_prior={} matches={} inliers={} relative_t_estimated={} relative_t_truth={} candidates={} success={} center={}",
            frame.frame_id,
            frame.used_vo_prior,
            frame.match_count,
            frame.inlier_count,
            estimated,
            truth,
            frame.candidate_landmark_count,
            frame.localization_succeeded,
            center,
        );
    }
    let _ = writeln!(
        report,
        "  stats frames={} success_rate={:.3} external_prior_rate={:.3} external_prior_count={}",
        run.stats.frame_count,
        run.stats.success_rate(),
        run.stats.external_localization_prior_usage_rate(),
        run.stats.external_localization_prior_used_count,
    );
}

fn format_vector(translation: Vector3<f64>) -> String {
    format!(
        "[{:.3}, {:.3}, {:.3}]",
        translation.x, translation.y, translation.z
    )
}

fn parse_output_dir(args: &mut Vec<String>) -> Option<PathBuf> {
    let output_flag_index = args.iter().position(|arg| arg == "--out-dir")?;
    if output_flag_index + 1 >= args.len() {
        eprintln!("--out-dir requires a directory path");
        std::process::exit(2);
    }
    let output_dir = PathBuf::from(args.remove(output_flag_index + 1));
    args.remove(output_flag_index);
    Some(output_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_two_view_match_strings_emits_lines_per_pair() {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let pose_a =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let pose_b =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        let pose_c =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.6, 0.0, 0.0));
        let points = [
            Point3::new(-1.0, -1.0, 5.0),
            Point3::new(1.0, -1.0, 5.0),
            Point3::new(0.0, 0.5, 5.5),
        ];
        let frames = vec![
            frame_from_projected_landmarks(100, &camera, &pose_a, &points),
            frame_from_projected_landmarks(101, &camera, &pose_b, &points),
            frame_from_projected_landmarks(102, &camera, &pose_c, &points),
        ];
        let strings = build_two_view_match_strings(&frames);
        assert_eq!(strings.len(), FRAME_PAIRS.len());
        for (_, _, contents) in &strings {
            let line_count = contents
                .lines()
                .filter(|line| !line.starts_with('#'))
                .count();
            assert_eq!(line_count, points.len());
        }
    }

    #[test]
    fn relative_truth_translations_subtracts_camera_translations() {
        let pose_a =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let pose_b =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        let translations = relative_truth_translations(&[(100, pose_a.clone()), (101, pose_b)]);
        assert_eq!(translations.len(), 1);
        assert_eq!(translations[0].0, (100, 101));
        assert!((translations[0].1 - Vector3::new(-0.3, 0.0, 0.0)).norm() < 1.0e-12);
    }
}
