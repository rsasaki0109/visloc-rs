//! Real-data KITTI 00 long-revisit appearance-scanner demo.
//!
//! Loads two segments of KITTI odometry seq 00 (e.g. frames 0–29 from
//! the start and frames 4500–4529 from the major loop-closure point),
//! extracts FAST corners + intensity-patch descriptors per frame with
//! the project's `CornerFeatureExtractor`, and runs the appearance
//! `scan_pairwise_loop_closures` over all keyframes from both segments
//! at once. `min_keyframe_id_gap` is set high enough that only
//! cross-segment pairs are eligible — same-segment pairs (which would
//! always pair as "loops" since they sit one frame apart) are filtered
//! before the verifier ever fires.
//!
//! Validates that the scanner's defaults pick up real KITTI revisits
//! end-to-end: descriptor matching across an actual driving loop, not
//! just synthetic features. Companion to the synthetic
//! `scanner_loop_closure_demo`.
//!
//! Usage (after fetching both segments via
//! `scripts/fetch_kitti_seq00_images.py --start-frame N --max-frames K
//! --stride 1 --cameras image_0 --out-dir ...`):
//!
//!     cargo run --release --features image-io --example kitti_revisit_scanner_demo -- \
//!         --segment-a ~/datasets/kitti_seq00_stride1_subset/image_0 \
//!         --calib-a   ~/datasets/kitti_seq00_stride1_subset/calib.txt \
//!         --segment-b ~/datasets/kitti_seq00_revisit_4500/image_0 \
//!         --calib-b   ~/datasets/kitti_seq00_revisit_4500/calib.txt
//!
//! Both calibration files are typically the same KITTI 00 `calib.txt`;
//! the demo accepts independent paths so a future user can compare
//! across recalibrated splits.

#[cfg(feature = "image-io")]
use std::fs;
#[cfg(feature = "image-io")]
use std::path::PathBuf;

#[cfg(feature = "image-io")]
use visloc_rs::io::kitti::read_kitti_image_sequence_dir;
#[cfg(feature = "image-io")]
use visloc_rs::vision::features::{
    CornerFeatureConfig, CornerFeatureExtractor, DeepFeatureExtractor, FeatureExtractor,
    FeatureSet, GrayscaleImage, HogLikeFeatureConfig, HogLikeFeatureExtractor,
    MultiScaleDeepConfig, MultiScaleDeepExtractor,
};
#[cfg(feature = "image-io")]
use visloc_rs::vision::matching::{
    BruteForceMatcher, Matcher, MutualSoftmaxConfig, MutualSoftmaxMatcher,
};
#[cfg(feature = "image-io")]
use visloc_rs::{
    scan_pairwise_loop_closures, EssentialMatrixLoopClosureVerifier, LoopClosureCandidate,
    LoopClosureVerifierConfig, PairwiseKeyframeView, PairwiseLoopClosureScannerConfig,
};

#[cfg(not(feature = "image-io"))]
fn main() {
    eprintln!("kitti_revisit_scanner_demo requires --features image-io");
    std::process::exit(2);
}

#[cfg(feature = "image-io")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrontendChoice {
    Classical,
    Deep,
    DeepMultiScale,
    Both,
}

#[cfg(feature = "image-io")]
impl FrontendChoice {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "classical" | "corner" => Ok(Self::Classical),
            "deep" | "hog" | "lightglue" => Ok(Self::Deep),
            "deep-ms" | "multiscale" | "deep-multi-scale" => Ok(Self::DeepMultiScale),
            "both" | "compare" => Ok(Self::Both),
            other => {
                Err(format!("--frontend must be classical|deep|deep-ms|both, got {other}").into())
            }
        }
    }
}

#[cfg(feature = "image-io")]
#[derive(Debug)]
struct CliArgs {
    segment_a: PathBuf,
    segment_b: PathBuf,
    calib_a: PathBuf,
    calib_b: PathBuf,
    projection_label: String,
    out_dir: Option<PathBuf>,
    frontend: FrontendChoice,
}

#[cfg(feature = "image-io")]
fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let mut segment_a: Option<PathBuf> = None;
    let mut segment_b: Option<PathBuf> = None;
    let mut calib_a: Option<PathBuf> = None;
    let mut calib_b: Option<PathBuf> = None;
    let mut projection_label = String::from("P0");
    let mut out_dir: Option<PathBuf> = None;
    let mut frontend = FrontendChoice::Classical;
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--segment-a" => segment_a = iter.next().map(PathBuf::from),
            "--segment-b" => segment_b = iter.next().map(PathBuf::from),
            "--calib-a" => calib_a = iter.next().map(PathBuf::from),
            "--calib-b" => calib_b = iter.next().map(PathBuf::from),
            "--projection" => {
                projection_label = iter
                    .next()
                    .ok_or("--projection requires a label like P0/P1")?;
            }
            "--out-dir" => out_dir = iter.next().map(PathBuf::from),
            "--frontend" => {
                let value = iter
                    .next()
                    .ok_or("--frontend requires classical|deep|both")?;
                frontend = FrontendChoice::parse(&value)?;
            }
            other => return Err(format!("unrecognised flag {other}").into()),
        }
    }
    Ok(CliArgs {
        segment_a: segment_a.ok_or("--segment-a is required")?,
        segment_b: segment_b.ok_or("--segment-b is required")?,
        calib_a: calib_a.ok_or("--calib-a is required")?,
        calib_b: calib_b.ok_or("--calib-b is required")?,
        projection_label,
        out_dir,
        frontend,
    })
}

#[cfg(feature = "image-io")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;

    let seq_a =
        read_kitti_image_sequence_dir(&args.segment_a, &args.calib_a, &args.projection_label, 0)?;
    let seq_b =
        read_kitti_image_sequence_dir(&args.segment_b, &args.calib_b, &args.projection_label, 0)?;
    if seq_a.frames.is_empty() || seq_b.frames.is_empty() {
        return Err(format!(
            "need at least 1 frame per segment (got A={}, B={})",
            seq_a.frames.len(),
            seq_b.frames.len(),
        )
        .into());
    }
    if seq_a.camera != seq_b.camera {
        eprintln!(
            "# warning: segment A camera differs from segment B camera; the scanner \
             assumes shared intrinsics — proceeding with segment A's camera."
        );
    }
    let camera = seq_a.camera.clone();
    println!(
        "segment_a frames={} (id_min={} id_max={})",
        seq_a.frames.len(),
        seq_a.frames.first().map(|f| f.frame_id).unwrap_or(0),
        seq_a.frames.last().map(|f| f.frame_id).unwrap_or(0),
    );
    println!(
        "segment_b frames={} (id_min={} id_max={})",
        seq_b.frames.len(),
        seq_b.frames.first().map(|f| f.frame_id).unwrap_or(0),
        seq_b.frames.last().map(|f| f.frame_id).unwrap_or(0),
    );

    // Per-frame KITTI id parsed from the filename stem (`004500.png` → 4500),
    // collected once and reused across every selected frontend.
    let mut frame_ids: Vec<u64> = Vec::with_capacity(seq_a.frames.len() + seq_b.frames.len());
    for frame in seq_a.frames.iter().chain(seq_b.frames.iter()) {
        let kitti_id = parse_kitti_frame_id(&frame.path).ok_or_else(|| {
            format!(
                "could not parse KITTI frame id from filename {:?}",
                frame.path
            )
        })?;
        frame_ids.push(kitti_id);
    }
    let id_a_min = frame_ids
        .iter()
        .take(seq_a.frames.len())
        .min()
        .copied()
        .unwrap_or(0);
    let id_a_max = frame_ids
        .iter()
        .take(seq_a.frames.len())
        .max()
        .copied()
        .unwrap_or(0);
    let id_b_min = frame_ids
        .iter()
        .skip(seq_a.frames.len())
        .min()
        .copied()
        .unwrap_or(0);
    let id_b_max = frame_ids
        .iter()
        .skip(seq_a.frames.len())
        .max()
        .copied()
        .unwrap_or(0);
    println!(
        "parsed KITTI ids: segment_a [{}..{}], segment_b [{}..{}]",
        id_a_min, id_a_max, id_b_min, id_b_max,
    );

    let segment_a_span = id_a_max - id_a_min + 1;
    let segment_b_span = id_b_max - id_b_min + 1;
    let min_gap = segment_a_span.max(segment_b_span);
    let scanner_cfg = PairwiseLoopClosureScannerConfig {
        min_keyframe_id_gap: min_gap,
        min_matches: 30,
    };
    println!(
        "scanner config: min_keyframe_id_gap={} min_matches={}",
        scanner_cfg.min_keyframe_id_gap, scanner_cfg.min_matches,
    );

    let images: Vec<&GrayscaleImage> = seq_a
        .frames
        .iter()
        .chain(seq_b.frames.iter())
        .map(|frame| &frame.image)
        .collect();

    let verifier = EssentialMatrixLoopClosureVerifier {
        config: LoopClosureVerifierConfig {
            min_inliers: 12,
            min_inlier_ratio: 0.4,
            max_mean_sampson_error: 5.0e-3,
            default_translation_scale: 1.0,
        },
        ..Default::default()
    };

    let frontends: Vec<FrontendChoice> = match args.frontend {
        FrontendChoice::Classical => vec![FrontendChoice::Classical],
        FrontendChoice::Deep => vec![FrontendChoice::Deep],
        FrontendChoice::DeepMultiScale => vec![FrontendChoice::DeepMultiScale],
        FrontendChoice::Both => vec![FrontendChoice::Classical, FrontendChoice::Deep],
    };

    let mut summary = String::new();
    summary.push_str(&format!(
        "min_keyframe_id_gap={} segment_a_frames={} segment_b_frames={}\n",
        scanner_cfg.min_keyframe_id_gap,
        seq_a.frames.len(),
        seq_b.frames.len(),
    ));

    let mut reports: Vec<FrontendReport> = Vec::new();
    for choice in frontends {
        let report = run_frontend(
            choice,
            &images,
            &frame_ids,
            &scanner_cfg,
            &verifier,
            &camera,
        )?;
        report.print();
        report.write_summary(&mut summary);
        reports.push(report);
    }

    if reports.len() == 2 {
        print_compare(&reports[0], &reports[1]);
    }

    if let Some(out_dir) = &args.out_dir {
        fs::create_dir_all(out_dir)?;
        fs::write(out_dir.join("summary.txt"), &summary)?;
        println!("wrote {}/summary.txt", out_dir.display());
    }
    Ok(())
}

#[cfg(feature = "image-io")]
struct FrontendReport {
    label: &'static str,
    feature_min: usize,
    feature_max: usize,
    candidates: Vec<LoopClosureCandidate>,
    elapsed_total_ms: u128,
    elapsed_extract_ms: u128,
    elapsed_scan_ms: u128,
}

#[cfg(feature = "image-io")]
impl FrontendReport {
    fn print(&self) {
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

    fn write_summary(&self, summary: &mut String) {
        summary.push_str(&format!("[{}]\n", self.label));
        summary.push_str(&format!(
            "  feature_count_min={} feature_count_max={}\n",
            self.feature_min, self.feature_max
        ));
        summary.push_str(&format!(
            "  extract_ms={} scan_ms={} total_ms={}\n",
            self.elapsed_extract_ms, self.elapsed_scan_ms, self.elapsed_total_ms
        ));
        summary.push_str(&format!("  candidates={}\n", self.candidates.len()));
        for c in &self.candidates {
            let v = c.verification.as_ref().expect("verification populated");
            summary.push_str(&format!(
                "  pair from={} to={} inliers={} ratio={:.4} mean_sampson={:.5} score={:.4}\n",
                c.matched_keyframe_id,
                c.query_frame_id,
                v.inlier_count,
                v.inlier_ratio,
                v.mean_sampson_error,
                c.score,
            ));
        }
        if let Some(strongest) = self.strongest() {
            summary.push_str(&format!(
                "  strongest_from={} strongest_to={} strongest_score={:.4}\n",
                strongest.matched_keyframe_id, strongest.query_frame_id, strongest.score,
            ));
        } else {
            summary.push_str("  strongest=None\n");
        }
        summary.push('\n');
    }

    fn strongest(&self) -> Option<&LoopClosureCandidate> {
        self.candidates.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

#[cfg(feature = "image-io")]
fn print_compare(left: &FrontendReport, right: &FrontendReport) {
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

#[cfg(feature = "image-io")]
fn run_frontend(
    choice: FrontendChoice,
    images: &[&GrayscaleImage],
    frame_ids: &[u64],
    scanner_cfg: &PairwiseLoopClosureScannerConfig,
    verifier: &EssentialMatrixLoopClosureVerifier,
    camera: &visloc_rs::Camera,
) -> Result<FrontendReport, Box<dyn std::error::Error>> {
    use std::time::Instant;
    let label = match choice {
        FrontendChoice::Classical => "classical (Corner + BF ratio 0.85)",
        FrontendChoice::Deep => "deep-style (HogLike + MutualSoftmax)",
        FrontendChoice::DeepMultiScale => "deep-style ms (HogLike×3 + MutualSoftmax)",
        FrontendChoice::Both => unreachable!("expanded earlier"),
    };
    let total_start = Instant::now();
    let extract_start = Instant::now();
    let features: Vec<(u64, FeatureSet)> = match choice {
        FrontendChoice::Classical => extract_classical_features(images, frame_ids)?,
        FrontendChoice::Deep => extract_deep_features(images, frame_ids)?,
        FrontendChoice::DeepMultiScale => extract_deep_multi_scale_features(images, frame_ids)?,
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
        label,
        feature_min,
        feature_max,
        candidates,
        elapsed_total_ms,
        elapsed_extract_ms,
        elapsed_scan_ms,
    })
}

#[cfg(feature = "image-io")]
fn run_scanner_with_matcher<M: Matcher>(
    views: &[PairwiseKeyframeView],
    matcher: &M,
    verifier: &EssentialMatrixLoopClosureVerifier,
    camera: &visloc_rs::Camera,
    config: &PairwiseLoopClosureScannerConfig,
) -> Vec<LoopClosureCandidate> {
    scan_pairwise_loop_closures(views, matcher, verifier, camera, config)
}

#[cfg(feature = "image-io")]
fn extract_classical_features(
    images: &[&GrayscaleImage],
    frame_ids: &[u64],
) -> Result<Vec<(u64, FeatureSet)>, Box<dyn std::error::Error>> {
    // FAST + intensity-patch descriptors with `max_features = 400` to keep
    // the cross-segment brute-force matcher under ~10 s of wall time. KITTI
    // 00 corner density is much higher than 400 per frame so the top-K
    // selection is still oversampled.
    let extractor = CornerFeatureExtractor::new(CornerFeatureConfig {
        max_features: 400,
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

#[cfg(feature = "image-io")]
fn extract_deep_features(
    images: &[&GrayscaleImage],
    frame_ids: &[u64],
) -> Result<Vec<(u64, FeatureSet)>, Box<dyn std::error::Error>> {
    // Match the classical frontend's per-keyframe feature cap so the only
    // moving variable in the comparison is descriptor + matcher quality.
    let extractor = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
        max_features: 400,
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

#[cfg(feature = "image-io")]
fn extract_deep_multi_scale_features(
    images: &[&GrayscaleImage],
    frame_ids: &[u64],
) -> Result<Vec<(u64, FeatureSet)>, Box<dyn std::error::Error>> {
    // Per-octave feature cap is 200 so the merged 3-octave feature count
    // averages around the 400-feature target the other frontends use,
    // keeping the comparison fair on cross-segment matcher cost.
    let inner = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
        max_features: 200,
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

#[cfg(feature = "image-io")]
fn parse_kitti_frame_id(path: &std::path::Path) -> Option<u64> {
    let stem = path.file_stem()?.to_str()?;
    stem.parse::<u64>().ok()
}
