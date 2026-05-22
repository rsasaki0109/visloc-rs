use visloc_rs::vision::features::{
    CornerFeatureConfig, CornerFeatureExtractor, DeepFeatureExtractor, FeatureExtractor,
    FeatureSet, GrayscaleImage, HogLikeFeatureConfig, HogLikeFeatureExtractor,
    MultiScaleDeepConfig, MultiScaleDeepExtractor,
};
use visloc_rs::vision::matching::{
    BruteForceMatcher, Matcher, MutualSoftmaxConfig, MutualSoftmaxMatcher,
};
use visloc_rs::{
    scan_pairwise_loop_closures, EssentialMatrixLoopClosureVerifier, LoopClosureCandidate,
    PairwiseKeyframeView, PairwiseLoopClosureScannerConfig,
};

use super::kitti_revisit_cli::{CliArgs, FrontendChoice};
use super::kitti_revisit_data::RevisitDataset;

pub(super) struct FrontendReport {
    pub(super) choice: FrontendChoice,
    pub(super) label: &'static str,
    pub(super) feature_min: usize,
    pub(super) feature_max: usize,
    pub(super) features: Vec<(u64, FeatureSet)>,
    pub(super) candidates: Vec<LoopClosureCandidate>,
    pub(super) elapsed_total_ms: u128,
    pub(super) elapsed_extract_ms: u128,
    pub(super) elapsed_scan_ms: u128,
}

impl FrontendReport {
    pub(super) fn print(&self) {
        println!("== {} ==", self.label);
        println!(
            "  feature counts (per keyframe): min={} max={}",
            self.feature_min, self.feature_max
        );
        println!(
            "  timing: extract={} ms  scan={} ms  total={} ms",
            self.elapsed_extract_ms, self.elapsed_scan_ms, self.elapsed_total_ms
        );
        println!(
            "  loop scanner: {} cross-segment candidates",
            self.candidates.len()
        );
        for c in &self.candidates {
            let v = c
                .verification
                .as_ref()
                .expect("scanner populates verification on accepted pairs");
            println!(
                "    ({}, {}) inliers={} ratio={:.3} mean_sampson={:.5} score={:.3}",
                c.matched_keyframe_id,
                c.query_frame_id,
                v.inlier_count,
                v.inlier_ratio,
                v.mean_sampson_error,
                c.score,
            );
        }
        if let Some(strongest) = self.strongest() {
            let v = strongest
                .verification
                .as_ref()
                .expect("verification populated");
            println!(
                "  strongest pair: ({}, {}) inliers={} ratio={:.3} score={:.3}",
                strongest.matched_keyframe_id,
                strongest.query_frame_id,
                v.inlier_count,
                v.inlier_ratio,
                strongest.score,
            );
        } else {
            println!("  no cross-segment loop detected at current thresholds");
        }
    }

    pub(super) fn strongest(&self) -> Option<&LoopClosureCandidate> {
        self.candidates.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

pub(super) fn selected_frontends(frontend: FrontendChoice) -> Vec<FrontendChoice> {
    match frontend {
        FrontendChoice::Classical => vec![FrontendChoice::Classical],
        FrontendChoice::Deep => vec![FrontendChoice::Deep],
        FrontendChoice::DeepMultiScale => vec![FrontendChoice::DeepMultiScale],
        FrontendChoice::Both => vec![FrontendChoice::Classical, FrontendChoice::Deep],
    }
}

pub(super) fn print_compare(left: &FrontendReport, right: &FrontendReport) {
    println!("== compare {} vs {} ==", left.label, right.label);
    println!(
        "  candidates : {} -> {}",
        left.candidates.len(),
        right.candidates.len()
    );
    let left_score = left.strongest().map(|c| c.score).unwrap_or(0.0);
    let right_score = right.strongest().map(|c| c.score).unwrap_or(0.0);
    println!("  best score : {:.3} -> {:.3}", left_score, right_score);
    let left_inliers = left
        .strongest()
        .and_then(|c| c.verification.as_ref())
        .map(|v| v.inlier_count)
        .unwrap_or(0);
    let right_inliers = right
        .strongest()
        .and_then(|c| c.verification.as_ref())
        .map(|v| v.inlier_count)
        .unwrap_or(0);
    println!("  best inlrs : {} -> {}", left_inliers, right_inliers);
    println!(
        "  total time : {} ms -> {} ms",
        left.elapsed_total_ms, right.elapsed_total_ms
    );
}

pub(super) fn run_selected_frontends(
    args: &CliArgs,
    dataset: &RevisitDataset,
    images: &[&GrayscaleImage],
    scanner_cfg: &PairwiseLoopClosureScannerConfig,
    verifier: &EssentialMatrixLoopClosureVerifier,
) -> Result<Vec<FrontendReport>, Box<dyn std::error::Error>> {
    let mut reports = Vec::new();
    for choice in selected_frontends(args.frontend) {
        let report = run_frontend(
            choice,
            images,
            &dataset.frame_ids,
            scanner_cfg,
            verifier,
            &dataset.camera,
            args.max_features,
        )?;
        report.print();
        reports.push(report);
    }
    if reports.len() == 2 {
        print_compare(&reports[0], &reports[1]);
    }
    Ok(reports)
}

pub(super) fn run_frontend(
    choice: FrontendChoice,
    images: &[&GrayscaleImage],
    frame_ids: &[u64],
    scanner_cfg: &PairwiseLoopClosureScannerConfig,
    verifier: &EssentialMatrixLoopClosureVerifier,
    camera: &visloc_rs::Camera,
    max_features: usize,
) -> Result<FrontendReport, Box<dyn std::error::Error>> {
    use std::time::Instant;
    let label = match choice {
        FrontendChoice::Classical => "classical (Corner + BF ratio 0.85)",
        FrontendChoice::Deep => "deep-style (HogLike + MutualSoftmax)",
        FrontendChoice::DeepMultiScale => "deep-style ms (HogLike x3 + MutualSoftmax)",
        FrontendChoice::Both => unreachable!("expanded earlier"),
    };
    let total_start = Instant::now();
    let extract_start = Instant::now();
    let features: Vec<(u64, FeatureSet)> = match choice {
        FrontendChoice::Classical => extract_classical_features(images, frame_ids, max_features)?,
        FrontendChoice::Deep => extract_deep_features(images, frame_ids, max_features)?,
        FrontendChoice::DeepMultiScale => {
            extract_deep_multi_scale_features(images, frame_ids, max_features)?
        }
        FrontendChoice::Both => unreachable!(),
    };
    let elapsed_extract_ms = extract_start.elapsed().as_millis();
    let feature_min = features.iter().map(|(_, f)| f.len()).min().unwrap_or(0);
    let feature_max = features.iter().map(|(_, f)| f.len()).max().unwrap_or(0);
    let views: Vec<PairwiseKeyframeView> = features
        .iter()
        .map(|(id, fs)| PairwiseKeyframeView::from_features(*id, fs))
        .collect();

    let scan_start = Instant::now();
    let candidates = match choice {
        FrontendChoice::Classical => {
            let matcher = BruteForceMatcher { ratio: Some(0.85) };
            scan_pairwise_loop_closures(&views, &matcher, verifier, camera, scanner_cfg)
        }
        FrontendChoice::Deep | FrontendChoice::DeepMultiScale => {
            let matcher = MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
                temperature: 25.0,
                min_confidence: 0.15,
                emit_ratio_metadata: false,
            });
            run_scanner_with_matcher(&views, &matcher, verifier, camera, scanner_cfg)
        }
        FrontendChoice::Both => unreachable!(),
    };
    let elapsed_scan_ms = scan_start.elapsed().as_millis();
    let elapsed_total_ms = total_start.elapsed().as_millis();

    Ok(FrontendReport {
        choice,
        label,
        feature_min,
        feature_max,
        features,
        candidates,
        elapsed_total_ms,
        elapsed_extract_ms,
        elapsed_scan_ms,
    })
}

fn run_scanner_with_matcher<M: Matcher>(
    views: &[PairwiseKeyframeView],
    matcher: &M,
    verifier: &EssentialMatrixLoopClosureVerifier,
    camera: &visloc_rs::Camera,
    config: &PairwiseLoopClosureScannerConfig,
) -> Vec<LoopClosureCandidate> {
    scan_pairwise_loop_closures(views, matcher, verifier, camera, config)
}

fn extract_classical_features(
    images: &[&GrayscaleImage],
    frame_ids: &[u64],
    max_features: usize,
) -> Result<Vec<(u64, FeatureSet)>, Box<dyn std::error::Error>> {
    // FAST + intensity-patch descriptors with a configurable cap to keep
    // cross-segment brute-force matching tractable for README-scale demos.
    // KITTI 00 corner density is much higher than typical caps here.
    let extractor = CornerFeatureExtractor::new(CornerFeatureConfig {
        max_features,
        min_score: 0.02,
        descriptor_radius: 9,
    });
    let mut out = Vec::with_capacity(images.len());
    for (image, &id) in images.iter().zip(frame_ids.iter()) {
        let fs = extractor.extract(image)?;
        out.push((id, fs));
    }
    Ok(out)
}

fn extract_deep_features(
    images: &[&GrayscaleImage],
    frame_ids: &[u64],
    max_features: usize,
) -> Result<Vec<(u64, FeatureSet)>, Box<dyn std::error::Error>> {
    // Match the classical frontend's per-keyframe feature cap so the only
    // moving variable in the comparison is descriptor + matcher quality.
    let extractor = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
        max_features,
        min_corner_score: 0.02,
        descriptor_clip: 0.2,
        orient: false,
    });
    let mut out = Vec::with_capacity(images.len());
    for (image, &id) in images.iter().zip(frame_ids.iter()) {
        let deep = extractor.extract_deep(image)?;
        out.push((id, deep.into_feature_set()));
    }
    Ok(out)
}

fn extract_deep_multi_scale_features(
    images: &[&GrayscaleImage],
    frame_ids: &[u64],
    max_features: usize,
) -> Result<Vec<(u64, FeatureSet)>, Box<dyn std::error::Error>> {
    // A half-size per-octave cap keeps the merged 3-octave count near the
    // single-scale target while preserving coarse-scale candidates.
    let per_octave_cap = (max_features / 2).max(1);
    let inner = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
        max_features: per_octave_cap,
        min_corner_score: 0.02,
        descriptor_clip: 0.2,
        orient: false,
    });
    let extractor = MultiScaleDeepExtractor::new(
        inner,
        MultiScaleDeepConfig {
            octaves: 3,
            area_weighted_octave_cap: false,
        },
    );
    let mut out = Vec::with_capacity(images.len());
    for (image, &id) in images.iter().zip(frame_ids.iter()) {
        let deep = extractor.extract_deep(image)?;
        out.push((id, deep.into_feature_set()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_requested_frontend_selection() {
        assert_eq!(
            selected_frontends(FrontendChoice::Both),
            vec![FrontendChoice::Classical, FrontendChoice::Deep]
        );
        assert_eq!(
            selected_frontends(FrontendChoice::DeepMultiScale),
            vec![FrontendChoice::DeepMultiScale]
        );
    }
}
