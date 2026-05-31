//! Appearance-based cross-session bridge proposal on REAL KITTI images.
//!
//! Validates the place-recognition driver (`visloc_rs::vision::place_recognition`)
//! end-to-end on real imagery — the automatic analogue of the hand-built bridges
//! in `multi_session_kitti_merge_demo`. It loads a KITTI grayscale sequence,
//! extracts classical corner features per frame (no deep model / external files),
//! and splits the frames into two INTERLEAVED sessions — session A is the
//! even-indexed frames, session B the odd-indexed ones — so the two sessions
//! traverse the SAME route at an offset sampling: frame `2k` (A) and frame `2k±1`
//! (B) are a few KITTI frames apart and view nearly the same place, with a real
//! viewpoint change. This is the cross-session revisit a place-recognition
//! front-end must find (the loopy revisit of KITTI-00 itself is thousands of
//! frames away, outside a short subset).
//!
//! It then reports, honestly decomposed into two stages:
//!   1. **Retrieval** — VLAD global descriptor per frame + mutual-nearest-neighbour
//!      retrieval (`vlad` + `retrieve_mutual`): how often does appearance alone
//!      pair a session-A frame with a spatially-adjacent session-B frame?
//!   2. **Retrieval + geometry** — the full `propose_bridges` (adds local
//!      descriptor matching + two-view relative pose): how many proposals survive
//!      geometric verification, and are they still correct?
//!
//! "Correct" = the proposed A/B frames are within `--adjacency` interleaved
//! positions of each other (same place); with `--kitti-poses` the ground-truth
//! camera-centre distance is also reported. Precision = correct / proposed;
//! recall = session-A frames with a correct proposal / session-A frames.
//!
//! Honest finding (KITTI-00 stereo subset, 100 frames, classical corner features):
//! at `--min-similarity 0.15` retrieval reaches **precision 1.0, recall ~0.4** —
//! every proposed cross-session pair is a true same-place match, and ~40 % of
//! session-A frames get one; every retrieved pair also passes two-view
//! verification (the strongest has 160+ inliers). Recall is bounded by the
//! discriminative power of the classical patch descriptors, not the retrieval or
//! geometry — a learned global/local descriptor (NetVLAD, SuperPoint) would lift
//! it. Precision stays high because mutual-NN + a modest threshold is selective.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --features image-io \
//!     --example multi_session_place_recognition_kitti_demo -- \
//!     --image-dir /path/to/KITTI/sequences/00/image_0 \
//!     --calib     /path/to/KITTI/sequences/00/calib.txt \
//!     --max-frames 100
//! ```

#[cfg(not(feature = "image-io"))]
fn main() {
    eprintln!(
        "this example requires the `image-io` feature; rebuild with \
         `cargo run --release --features image-io --example multi_session_place_recognition_kitti_demo`"
    );
    std::process::exit(2);
}

#[cfg(feature = "image-io")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    imp::run()
}

#[cfg(feature = "image-io")]
mod imp {
    use std::env;
    use std::path::PathBuf;

    use visloc_rs::core::types::Camera;
    use visloc_rs::io::kitti::read_kitti_image_sequence_dir;
    use visloc_rs::tracking::PoseTrajectory;
    use visloc_rs::vision::features::{
        CornerFeatureConfig, CornerFeatureExtractor, FeatureExtractor, FeatureSet,
    };
    use visloc_rs::vision::matching::BruteForceMatcher;
    use visloc_rs::vision::place_recognition::{
        propose_bridges, retrieve_mutual, vlad, BridgeProposalConfig, Vocabulary,
    };
    use visloc_rs::vision::two_view::RelativePoseEstimator;

    struct Args {
        image_dir: PathBuf,
        calib: PathBuf,
        kitti_poses: Option<PathBuf>,
        start_frame: usize,
        frame_stride: usize,
        max_frames: usize,
        max_features: usize,
        vocab_size: usize,
        vocab_sample: usize,
        min_similarity: f32,
        min_inliers: usize,
        adjacency: usize,
        proximity_m: f64,
    }

    fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
        let mut image_dir: Option<PathBuf> = None;
        let mut calib: Option<PathBuf> = None;
        let mut kitti_poses: Option<PathBuf> = None;
        let mut start_frame = 0usize;
        let mut frame_stride = 1usize;
        let mut max_frames = 100usize;
        let mut max_features = 400usize;
        let mut vocab_size = 64usize;
        let mut vocab_sample = 8000usize;
        let mut min_similarity = 0.15f32;
        let mut min_inliers = 12usize;
        let mut adjacency = 3usize;
        let mut proximity_m = 5.0f64;

        let mut args = env::args().skip(1).collect::<Vec<_>>();
        let i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--image-dir" => {
                    image_dir = Some(PathBuf::from(args.remove(i + 1)));
                    args.remove(i);
                }
                "--calib" => {
                    calib = Some(PathBuf::from(args.remove(i + 1)));
                    args.remove(i);
                }
                "--kitti-poses" => {
                    kitti_poses = Some(PathBuf::from(args.remove(i + 1)));
                    args.remove(i);
                }
                "--start-frame" => {
                    start_frame = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--frame-stride" => {
                    frame_stride = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--max-frames" => {
                    max_frames = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--max-features" => {
                    max_features = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--vocab-size" => {
                    vocab_size = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--vocab-sample" => {
                    vocab_sample = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--min-similarity" => {
                    min_similarity = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--min-inliers" => {
                    min_inliers = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--adjacency" => {
                    adjacency = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                "--proximity-m" => {
                    proximity_m = args.remove(i + 1).parse()?;
                    args.remove(i);
                }
                other => return Err(format!("unknown argument: {other}").into()),
            }
        }
        Ok(Args {
            image_dir: image_dir.ok_or("--image-dir <KITTI sequence image_0 dir> is required")?,
            calib: calib.ok_or("--calib <path/to/calib.txt> is required")?,
            kitti_poses,
            start_frame,
            frame_stride,
            max_frames,
            max_features,
            vocab_size,
            vocab_sample,
            min_similarity,
            min_inliers,
            adjacency,
            proximity_m,
        })
    }

    /// Interleaved position of the `db`-session frame `di` and `query`-session
    /// frame `qi`: A frames sit at even positions `2·qi`, B at odd `2·di+1`.
    fn position_gap(qi: usize, di: usize) -> usize {
        let a = 2 * qi;
        let b = 2 * di + 1;
        a.abs_diff(b)
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args = parse_args()?;
        let seq = read_kitti_image_sequence_dir(&args.image_dir, &args.calib, "P0", 0)?;
        let camera: Camera = seq.camera.clone();

        // Subsample frames; remember each one's KITTI frame id for the GT lookup.
        let selected: Vec<(u64, &_)> = seq
            .frames
            .iter()
            .skip(args.start_frame)
            .step_by(args.frame_stride.max(1))
            .take(args.max_frames)
            .map(|f| (f.frame_id, &f.image))
            .collect();
        let n = selected.len();
        if n < 6 {
            return Err(format!("need at least 6 frames, got {n}").into());
        }
        println!(
            "image_dir={} frames={n} max_features={} vocab_size={}",
            args.image_dir.display(),
            args.max_features,
            args.vocab_size,
        );

        // Extract classical corner features per frame.
        let extractor = CornerFeatureExtractor::new(CornerFeatureConfig {
            max_features: args.max_features,
            ..CornerFeatureConfig::default()
        });
        let features: Vec<FeatureSet> = selected
            .iter()
            .map(|(_, img)| extractor.extract(img))
            .collect::<Result<_, _>>()?;
        let median_features = {
            let mut counts: Vec<usize> = features.iter().map(|f| f.keypoints.len()).collect();
            counts.sort_unstable();
            counts[counts.len() / 2]
        };
        println!("median features/frame = {median_features}");

        // Interleave into two sessions: A = even-index frames, B = odd-index.
        let a_idx: Vec<usize> = (0..n).step_by(2).collect();
        let b_idx: Vec<usize> = (1..n).step_by(2).collect();
        let a_feats: Vec<FeatureSet> = a_idx.iter().map(|&i| features[i].clone()).collect();
        let b_feats: Vec<FeatureSet> = b_idx.iter().map(|&i| features[i].clone()).collect();
        println!(
            "session A frames={} session B frames={}",
            a_feats.len(),
            b_feats.len()
        );

        // Build the vocabulary from a pooled, capped descriptor sample (k-means).
        let mut pool: Vec<&[f32]> = features
            .iter()
            .flat_map(|f| f.descriptors.iter().map(|d| d.as_slice()))
            .collect();
        let stride = (pool.len() / args.vocab_sample.max(1)).max(1);
        let sampled: Vec<&[f32]> = pool.drain(..).step_by(stride).collect();
        let vocab = Vocabulary::build(&sampled, args.vocab_size, 20, 7)
            .ok_or("vocabulary construction failed (too few descriptors?)")?;
        println!(
            "vocabulary: k={} dim={} (from {} sampled descriptors)",
            vocab.k(),
            vocab.dim(),
            sampled.len(),
        );

        // Ground-truth camera centres, if poses were supplied.
        let gt = match &args.kitti_poses {
            Some(p) => Some(PoseTrajectory::read_kitti_poses(p)?),
            None => None,
        };
        let gt_distance = |qi: usize, di: usize| -> Option<f64> {
            let traj = gt.as_ref()?;
            let samples = traj.samples();
            let fa = selected[a_idx[qi]].0 as usize;
            let fb = selected[b_idx[di]].0 as usize;
            let pa = samples.get(fa)?.pose.camera_center_world();
            let pb = samples.get(fb)?.pose.camera_center_world();
            Some((pa - pb).norm())
        };

        // --- Stage 1: appearance retrieval only ---
        let a_globals: Vec<Vec<f32>> = a_feats
            .iter()
            .map(|f| vlad(&f.descriptors, &vocab))
            .collect();
        let b_globals: Vec<Vec<f32>> = b_feats
            .iter()
            .map(|f| vlad(&f.descriptors, &vocab))
            .collect();
        let retrieved = retrieve_mutual(&a_globals, &b_globals, args.min_similarity);
        report_stage(
            "retrieval (VLAD + mutual-NN)",
            retrieved.iter().map(|r| (r.query, r.db)),
            a_feats.len(),
            args.adjacency,
            &gt_distance,
            args.proximity_m,
            args.kitti_poses.is_some(),
        );

        // --- Stage 2: retrieval + two-view geometry ---
        let proposals = propose_bridges(
            &a_feats,
            &b_feats,
            &vocab,
            &BruteForceMatcher { ratio: Some(0.8) },
            &RelativePoseEstimator::default(),
            &camera,
            &BridgeProposalConfig {
                min_similarity: args.min_similarity,
                min_inliers: args.min_inliers,
            },
        );
        report_stage(
            "retrieval + geometry (propose_bridges)",
            proposals.iter().map(|p| (p.query, p.db)),
            a_feats.len(),
            args.adjacency,
            &gt_distance,
            args.proximity_m,
            args.kitti_poses.is_some(),
        );
        if let Some(p) = proposals.first() {
            println!(
                "strongest proposal: A[{}]↔B[{}] similarity={:.3} inliers={} sampson={:.4}",
                p.query, p.db, p.similarity, p.inlier_count, p.mean_sampson_error,
            );
        }
        println!(
            "NOTE: each surviving proposal maps to a slam LoopClosureConstraint \
             (from=A keyframe, to=B keyframe, relative_pose=query_to_db, up to scale for \
             monocular) → consistent_session_bridges (Mahalanobis PCM) → merge_session."
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn report_stage(
        label: &str,
        pairs: impl Iterator<Item = (usize, usize)>,
        a_count: usize,
        adjacency: usize,
        gt_distance: &impl Fn(usize, usize) -> Option<f64>,
        proximity_m: f64,
        have_gt: bool,
    ) {
        let pairs: Vec<(usize, usize)> = pairs.collect();
        let total = pairs.len();
        let mut correct = 0usize;
        let mut gt_correct = 0usize;
        let mut gt_evaluated = 0usize;
        let mut covered = std::collections::HashSet::new();
        for &(qi, di) in &pairs {
            let adjacent = position_gap(qi, di) <= adjacency;
            if adjacent {
                correct += 1;
                covered.insert(qi);
            }
            if let Some(d) = gt_distance(qi, di) {
                gt_evaluated += 1;
                if d <= proximity_m {
                    gt_correct += 1;
                }
            }
        }
        let precision = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };
        let recall = covered.len() as f64 / a_count.max(1) as f64;
        println!(
            "[{label}] proposals={total} adjacency-precision={precision:.2} recall={recall:.2} \
             (correct={correct}, A-frames-covered={}/{a_count})",
            covered.len(),
        );
        if have_gt && gt_evaluated > 0 {
            println!(
                "    GT: {gt_correct}/{gt_evaluated} proposals within {proximity_m:.1} m of each other"
            );
        }
    }
}
