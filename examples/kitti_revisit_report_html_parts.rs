use visloc_rs::LoopClosureCandidate;

use super::kitti_revisit_format::html_escape;
use super::kitti_revisit_frontend::FrontendReport;

pub(super) fn report_metrics_html(
    report: &FrontendReport,
    strongest: Option<&LoopClosureCandidate>,
) -> String {
    let strongest_inliers = strongest
        .and_then(|candidate| candidate.verification.as_ref())
        .map(|verification| verification.inlier_count)
        .unwrap_or(0);
    let strongest_score = strongest.map(|candidate| candidate.score).unwrap_or(0.0);
    format!(
        r#"<div class="metrics">
  <div class="metric"><b>{}</b><span>accepted candidates</span></div>
  <div class="metric"><b>{}</b><span>best inliers</span></div>
  <div class="metric"><b>{:.3}</b><span>best score</span></div>
  <div class="metric"><b>{} ms</b><span>total runtime</span></div>
</div>"#,
        report.candidates.len(),
        strongest_inliers,
        strongest_score,
        report.elapsed_total_ms,
    )
}

pub(super) fn candidate_table_html(report: &FrontendReport) -> String {
    let mut rows = String::new();
    for candidate in report.candidates.iter().take(20) {
        let verification = candidate
            .verification
            .as_ref()
            .expect("scanner populates verification");
        rows.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.5}</td></tr>\n",
            candidate.matched_keyframe_id,
            candidate.query_frame_id,
            candidate.score,
            verification.correspondence_count,
            verification.inlier_count,
            verification.inlier_ratio,
            verification.mean_sampson_error,
        ));
    }
    if rows.is_empty() {
        rows.push_str("<tr><td colspan=\"7\" class=\"empty\">No accepted candidates</td></tr>\n");
    }
    format!(
        r#"<h2>{}</h2>
<table>
  <thead><tr><th>from</th><th>to</th><th>score</th><th>matches</th><th>inliers</th><th>ratio</th><th>mean Sampson</th></tr></thead>
  <tbody>
{}
  </tbody>
</table>
"#,
        html_escape(report.label),
        rows,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kitti_revisit_cli::FrontendChoice;
    use visloc_rs::{LoopClosureCandidate, LoopClosureVerification};

    fn candidate() -> LoopClosureCandidate {
        LoopClosureCandidate {
            query_frame_id: 4501,
            matched_keyframe_id: 49,
            shared_landmark_count: 0,
            query_inlier_count: 0,
            keyframe_observation_count: 0,
            shared_landmark_ratio: 0.0,
            score: 16083.0719,
            geometrically_verified: true,
            verification: Some(LoopClosureVerification {
                verified: true,
                correspondence_count: 95,
                inlier_count: 57,
                inlier_ratio: 0.6,
                mean_sampson_error: 0.00213,
                score: 16083.0719,
                failure_reason: None,
                relative_pose: None,
                mean_reprojection_error_px: None,
            }),
        }
    }

    fn report(candidates: Vec<LoopClosureCandidate>) -> FrontendReport {
        FrontendReport {
            choice: FrontendChoice::Deep,
            label: "deep-style <unsafe>",
            feature_min: 200,
            feature_max: 200,
            features: Vec::new(),
            candidates,
            elapsed_total_ms: 123,
            elapsed_extract_ms: 50,
            elapsed_scan_ms: 73,
        }
    }

    #[test]
    fn candidate_table_escapes_label_and_formats_candidate_metrics() {
        let html = candidate_table_html(&report(vec![candidate()]));

        assert!(html.contains("deep-style &lt;unsafe&gt;"));
        assert!(html.contains("<td>49</td><td>4501</td><td>16083.072</td>"));
        assert!(html.contains("<td>95</td><td>57</td><td>0.600</td><td>0.00213</td>"));
    }

    #[test]
    fn candidate_table_reports_empty_candidates() {
        let html = candidate_table_html(&report(Vec::new()));

        assert!(html.contains("No accepted candidates"));
        assert!(html.contains("colspan=\"7\""));
    }

    #[test]
    fn report_metrics_uses_zeroes_when_no_strongest_candidate_exists() {
        let html = report_metrics_html(&report(Vec::new()), None);

        assert!(html.contains("<b>0</b><span>accepted candidates</span>"));
        assert!(html.contains("<b>0.000</b><span>best score</span>"));
        assert!(html.contains("<b>123 ms</b><span>total runtime</span>"));
    }
}
