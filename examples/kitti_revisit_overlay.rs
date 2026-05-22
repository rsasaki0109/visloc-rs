use std::fs;
use std::path::Path;

use visloc_rs::vision::features::FeatureSet;
use visloc_rs::vision::matching::{
    BruteForceMatcher, DescriptorMatch, Matcher, MutualSoftmaxConfig, MutualSoftmaxMatcher,
};
use visloc_rs::vision::two_view::TwoViewCorrespondence;
use visloc_rs::{EssentialMatrixLoopClosureVerifier, LoopClosureCandidate};

use super::kitti_revisit_cli::FrontendChoice;
use super::kitti_revisit_format::html_escape;
use super::kitti_revisit_frontend::FrontendReport;

pub(super) fn write_match_overlay_svg(
    report: &FrontendReport,
    strongest: &LoopClosureCandidate,
    frame_dimensions: &[(u64, usize, usize)],
    camera: &visloc_rs::Camera,
    verifier: &EssentialMatrixLoopClosureVerifier,
    assets_dir: &Path,
    from_asset: &str,
    to_asset: &str,
    stem: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let from_features = find_features(report, strongest.matched_keyframe_id)?;
    let to_features = find_features(report, strongest.query_frame_id)?;
    let (from_width, from_height) =
        find_dimensions(frame_dimensions, strongest.matched_keyframe_id)?;
    let (to_width, to_height) = find_dimensions(frame_dimensions, strongest.query_frame_id)?;

    let matches = match_descriptors_for_report(report, from_features, to_features);
    let total_matches = matches.len();
    let mut inlier_matches =
        verified_inlier_overlay_matches(&matches, from_features, to_features, camera, verifier)?;
    sort_overlay_matches(&mut inlier_matches);
    let shown = inlier_matches.len().min(80);
    let gap = 24.0;
    let width = from_width as f64 + gap + to_width as f64;
    let height = from_height.max(to_height) as f64;
    let right_x = from_width as f64 + gap;

    let mut lines = String::new();
    for (rank, m) in inlier_matches.iter().take(shown).enumerate() {
        let color = if rank % 2 == 0 { "#0f766e" } else { "#b45309" };
        let x1 = m.previous_x;
        let y1 = m.previous_y;
        let x2 = right_x + m.current_x;
        let y2 = m.current_y;
        lines.push_str(&format!(
            r#"<line x1="{x1:.2}" y1="{y1:.2}" x2="{x2:.2}" y2="{y2:.2}" stroke="{color}" stroke-width="1.3" stroke-opacity="0.42"/><circle cx="{x1:.2}" cy="{y1:.2}" r="2.0" fill="{color}" fill-opacity="0.75"/><circle cx="{x2:.2}" cy="{y2:.2}" r="2.0" fill="{color}" fill-opacity="0.75"/>"#
        ));
    }

    let filename = format!("{stem}.svg");
    let verification = strongest
        .verification
        .as_ref()
        .expect("scanner populates verification");
    let svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0}" height="{height:.0}" viewBox="0 0 {width:.0} {height:.0}">
  <rect width="100%" height="100%" fill="#e2e8f0"/>
  <image href="{from_href}" x="0" y="0" width="{from_width}" height="{from_height}" preserveAspectRatio="none"/>
  <image href="{to_href}" x="{right_x:.0}" y="0" width="{to_width}" height="{to_height}" preserveAspectRatio="none"/>
  <rect x="{from_width}" y="0" width="{gap:.0}" height="{height:.0}" fill="#f8fafc"/>
  <g>{lines}</g>
  <rect x="12" y="12" width="530" height="44" rx="4" fill="#0f172a" fill-opacity="0.82"/>
  <text x="24" y="31" fill="#ffffff" font-family="system-ui, Segoe UI, sans-serif" font-size="15" font-weight="700">frame {from_id} -> {to_id}: {shown}/{inliers} verified inlier matches</text>
  <text x="24" y="49" fill="#cbd5e1" font-family="system-ui, Segoe UI, sans-serif" font-size="12">essential verifier accepted {inliers}/{total} frontend correspondences; ratio {ratio:.3}; mean Sampson {sampson:.5}</text>
</svg>
"##,
        width = width,
        height = height,
        from_href = html_escape(from_asset),
        to_href = html_escape(to_asset),
        from_width = from_width,
        from_height = from_height,
        right_x = right_x,
        to_width = to_width,
        to_height = to_height,
        gap = gap,
        lines = lines,
        from_id = strongest.matched_keyframe_id,
        to_id = strongest.query_frame_id,
        shown = shown,
        total = total_matches,
        inliers = verification.inlier_count,
        ratio = verification.inlier_ratio,
        sampson = verification.mean_sampson_error,
    );
    fs::write(assets_dir.join(&filename), svg)?;
    Ok(filename)
}

#[derive(Debug, Clone)]
struct OverlayMatch {
    previous_x: f64,
    previous_y: f64,
    current_x: f64,
    current_y: f64,
    distance: f32,
    confidence: Option<f32>,
}

fn match_descriptors_for_report(
    report: &FrontendReport,
    from_features: &FeatureSet,
    to_features: &FeatureSet,
) -> Vec<DescriptorMatch> {
    match report.choice {
        FrontendChoice::Classical => {
            let matcher = BruteForceMatcher { ratio: Some(0.85) };
            matcher.match_descriptors(&from_features.descriptors, &to_features.descriptors)
        }
        FrontendChoice::Deep | FrontendChoice::DeepMultiScale => {
            let matcher = MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
                temperature: 25.0,
                min_confidence: 0.15,
                emit_ratio_metadata: false,
            });
            matcher.match_descriptors(&from_features.descriptors, &to_features.descriptors)
        }
        FrontendChoice::Both => unreachable!("expanded earlier"),
    }
}

fn verified_inlier_overlay_matches(
    matches: &[DescriptorMatch],
    from_features: &FeatureSet,
    to_features: &FeatureSet,
    camera: &visloc_rs::Camera,
    verifier: &EssentialMatrixLoopClosureVerifier,
) -> Result<Vec<OverlayMatch>, Box<dyn std::error::Error>> {
    let mut correspondences: Vec<TwoViewCorrespondence> = Vec::with_capacity(matches.len());
    let mut weights: Vec<f32> = Vec::with_capacity(matches.len());
    let mut overlay_matches: Vec<OverlayMatch> = Vec::with_capacity(matches.len());
    let mut any_confidence = false;
    for m in matches {
        let Some(prev) = from_features.keypoints.get(m.query_index) else {
            continue;
        };
        let Some(curr) = to_features.keypoints.get(m.train_index) else {
            continue;
        };
        correspondences.push(TwoViewCorrespondence {
            previous_xy: *prev,
            current_xy: *curr,
        });
        if let Some(confidence) = m.confidence {
            any_confidence = true;
            weights.push(confidence);
        } else {
            weights.push(1.0);
        }
        overlay_matches.push(OverlayMatch {
            previous_x: prev.x,
            previous_y: prev.y,
            current_x: curr.x,
            current_y: curr.y,
            distance: m.distance,
            confidence: m.confidence,
        });
    }

    let pose = if any_confidence {
        verifier.estimator.estimate_with_scale_and_weights(
            &correspondences,
            camera,
            verifier.config.default_translation_scale,
            &weights,
        )
    } else {
        verifier.estimator.estimate_with_scale(
            &correspondences,
            camera,
            verifier.config.default_translation_scale,
        )
    }
    .ok_or("could not recover verified inlier matches for overlay")?;

    let inliers = pose
        .inliers
        .iter()
        .filter_map(|&index| overlay_matches.get(index).cloned())
        .collect();
    Ok(inliers)
}

fn sort_overlay_matches(matches: &mut [OverlayMatch]) {
    matches.sort_by(|a, b| match (a.confidence, b.confidence) {
        (Some(lhs), Some(rhs)) => rhs.partial_cmp(&lhs).unwrap_or(std::cmp::Ordering::Equal),
        _ => a
            .distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal),
    });
}

fn find_features(
    report: &FrontendReport,
    frame_id: u64,
) -> Result<&FeatureSet, Box<dyn std::error::Error>> {
    report
        .features
        .iter()
        .find(|(id, _)| *id == frame_id)
        .map(|(_, features)| features)
        .ok_or_else(|| format!("no extracted features for KITTI frame {frame_id}").into())
}

pub(super) fn find_dimensions(
    frame_dimensions: &[(u64, usize, usize)],
    frame_id: u64,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    frame_dimensions
        .iter()
        .find(|(id, _, _)| *id == frame_id)
        .map(|(_, width, height)| (*width, *height))
        .ok_or_else(|| format!("no image dimensions for KITTI frame {frame_id}").into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_dimensions_reports_missing_frame() {
        let err = find_dimensions(&[(49, 1241, 376)], 4501)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no image dimensions for KITTI frame 4501"));
    }
}
