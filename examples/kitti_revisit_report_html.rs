use std::fs;
use std::path::{Path, PathBuf};

use visloc_rs::EssentialMatrixLoopClosureVerifier;

use super::kitti_revisit_format::{html_escape, label_slug};
use super::kitti_revisit_frontend::FrontendReport;
use super::kitti_revisit_overlay::write_match_overlay_svg;
use super::kitti_revisit_report::ReportInputs;
use super::kitti_revisit_report_html_parts::{candidate_table_html, report_metrics_html};
use super::kitti_revisit_report_html_template::{render_report_page, ReportPageData};

pub(super) fn write_html_report(
    inputs: &ReportInputs<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let assets_dir = inputs.out_dir.join("assets");
    fs::create_dir_all(&assets_dir)?;

    let mut cards = String::new();
    let mut tables = String::new();
    for report in inputs.reports {
        cards.push_str(&report_card_html(
            report,
            inputs.frame_paths,
            inputs.frame_dimensions,
            inputs.camera,
            inputs.verifier,
            &assets_dir,
        )?);
        tables.push_str(&candidate_table_html(report));
    }

    let args = inputs.args;
    let command = format!(
        "cargo run --release --features image-io --example kitti_revisit_scanner_demo -- --segment-a {} --calib-a {} --segment-b {} --calib-b {} --projection {} --frontend {} --max-features {} --min-matches {} --min-inliers {} --min-inlier-ratio {:.3} --max-mean-sampson-error {:.5} --out-dir {}",
        args.segment_a.display(),
        args.calib_a.display(),
        args.segment_b.display(),
        args.calib_b.display(),
        args.projection_label,
        args.frontend.as_cli_label(),
        args.max_features,
        args.min_matches,
        args.min_inliers,
        args.min_inlier_ratio,
        args.max_mean_sampson_error,
        inputs.out_dir.display(),
    );
    let html = render_report_page(ReportPageData {
        segment_a_range: inputs.segment_a_range,
        segment_b_range: inputs.segment_b_range,
        frontend_label: args.frontend.as_cli_label(),
        report_count: inputs.reports.len(),
        max_features: args.max_features,
        min_gap: inputs.scanner_cfg.min_keyframe_id_gap,
        min_matches: inputs.scanner_cfg.min_matches,
        min_inliers: inputs.verifier_cfg.min_inliers,
        min_ratio: inputs.verifier_cfg.min_inlier_ratio,
        max_sampson: inputs.verifier_cfg.max_mean_sampson_error,
        command: &command,
        cards: &cards,
        tables: &tables,
    });
    fs::write(inputs.out_dir.join("index.html"), html)?;
    Ok(())
}

fn report_card_html(
    report: &FrontendReport,
    frame_paths: &[(u64, PathBuf)],
    frame_dimensions: &[(u64, usize, usize)],
    camera: &visloc_rs::Camera,
    verifier: &EssentialMatrixLoopClosureVerifier,
    assets_dir: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let Some(strongest) = report.strongest() else {
        return Ok(format!(
            r#"<article class="card"><h2>{}</h2><p class="empty">No accepted cross-segment loop candidate at current thresholds.</p>{}</article>"#,
            html_escape(report.label),
            report_metrics_html(report, None),
        ));
    };
    let verification = strongest
        .verification
        .as_ref()
        .expect("scanner populates verification");
    let from_src = find_frame_path(frame_paths, strongest.matched_keyframe_id)?;
    let to_src = find_frame_path(frame_paths, strongest.query_frame_id)?;
    let slug = label_slug(report.label);
    let from_asset = copy_report_image(
        from_src,
        assets_dir,
        &format!("{}_from_{}", slug, strongest.matched_keyframe_id),
    )?;
    let to_asset = copy_report_image(
        to_src,
        assets_dir,
        &format!("{}_to_{}", slug, strongest.query_frame_id),
    )?;
    let overlay_asset = write_match_overlay_svg(
        report,
        strongest,
        frame_dimensions,
        camera,
        verifier,
        assets_dir,
        &from_asset,
        &to_asset,
        &format!(
            "{}_matches_{}_{}",
            slug, strongest.matched_keyframe_id, strongest.query_frame_id
        ),
    )?;
    Ok(format!(
        r#"<article class="card">
  <h2>{label}</h2>
  {metrics}
  <p class="note">Strongest pair: frame {from_id} -> {to_id}; score {score:.3}; {inliers}/{matches} verified inliers; mean Sampson {sampson:.5}.</p>
  <figure class="overlay"><img src="{overlay_href}" alt="{label} verified inlier matches from frame {from_id} to {to_id}"><figcaption>essential-matrix verified inlier matches recovered from the same frontend correspondences</figcaption></figure>
  <div class="pair">
    <figure><img src="{from_href}" alt="{label} matched KITTI frame {from_id}"><figcaption>matched keyframe {from_id}</figcaption></figure>
    <figure><img src="{to_href}" alt="{label} query KITTI frame {to_id}"><figcaption>query frame {to_id}</figcaption></figure>
  </div>
</article>"#,
        label = html_escape(report.label),
        metrics = report_metrics_html(report, Some(strongest)),
        from_id = strongest.matched_keyframe_id,
        to_id = strongest.query_frame_id,
        score = strongest.score,
        inliers = verification.inlier_count,
        matches = verification.correspondence_count,
        sampson = verification.mean_sampson_error,
        overlay_href = html_escape(&format!("assets/{}", overlay_asset)),
        from_href = html_escape(&format!("assets/{}", from_asset)),
        to_href = html_escape(&format!("assets/{}", to_asset)),
    ))
}

fn find_frame_path(
    frame_paths: &[(u64, PathBuf)],
    frame_id: u64,
) -> Result<&Path, Box<dyn std::error::Error>> {
    frame_paths
        .iter()
        .find(|(id, _)| *id == frame_id)
        .map(|(_, path)| path.as_path())
        .ok_or_else(|| format!("no source image path for KITTI frame {frame_id}").into())
}

fn copy_report_image(
    source: &Path,
    assets_dir: &Path,
    stem: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let extension = source
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("png");
    let filename = format!("{stem}.{extension}");
    fs::copy(source, assets_dir.join(&filename))?;
    Ok(filename)
}
