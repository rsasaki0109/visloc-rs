use std::fs;
use std::path::Path;

use super::kitti_revisit_format::csv_cell;
use super::kitti_revisit_frontend::FrontendReport;

pub(super) fn write_candidates_csv(
    path: &Path,
    reports: &[FrontendReport],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut csv = String::from(
        "frontend,matched_keyframe_id,query_frame_id,score,matches,inliers,inlier_ratio,mean_sampson_error,verified\n",
    );
    for report in reports {
        for candidate in &report.candidates {
            let verification = candidate
                .verification
                .as_ref()
                .expect("scanner populates verification");
            csv.push_str(&format!(
                "{},{},{},{:.6},{},{},{:.6},{:.8},{}\n",
                csv_cell(report.label),
                candidate.matched_keyframe_id,
                candidate.query_frame_id,
                candidate.score,
                verification.correspondence_count,
                verification.inlier_count,
                verification.inlier_ratio,
                verification.mean_sampson_error,
                verification.verified,
            ));
        }
    }
    fs::write(path, csv)?;
    Ok(())
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

    #[test]
    fn writes_candidate_rows_with_stable_numeric_precision() {
        let report = FrontendReport {
            choice: FrontendChoice::Deep,
            label: "deep-style",
            feature_min: 200,
            feature_max: 200,
            features: Vec::new(),
            candidates: vec![candidate()],
            elapsed_total_ms: 10,
            elapsed_extract_ms: 4,
            elapsed_scan_ms: 6,
        };
        let path = std::env::temp_dir().join("kitti_revisit_candidates_unit.csv");

        write_candidates_csv(&path, &[report]).unwrap();

        let csv = fs::read_to_string(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert!(csv.starts_with("frontend,matched_keyframe_id,query_frame_id"));
        assert!(csv.contains("deep-style,49,4501,16083.071900,95,57,0.600000,0.00213000,true"));
    }
}
