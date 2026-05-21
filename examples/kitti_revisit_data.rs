use std::path::{Path, PathBuf};

use visloc_rs::io::kitti::{read_kitti_image_sequence_dir, KittiImageSequence};
use visloc_rs::vision::features::GrayscaleImage;

use super::kitti_revisit_cli::CliArgs;

pub(super) struct RevisitDataset {
    pub(super) camera: visloc_rs::Camera,
    pub(super) images: Vec<GrayscaleImage>,
    pub(super) frame_ids: Vec<u64>,
    pub(super) frame_paths: Vec<(u64, PathBuf)>,
    pub(super) segment_a_len: usize,
    pub(super) segment_b_len: usize,
    pub(super) segment_a_range: (u64, u64),
    pub(super) segment_b_range: (u64, u64),
    pub(super) min_keyframe_id_gap: u64,
}

pub(super) fn load_revisit_dataset(
    args: &CliArgs,
) -> Result<RevisitDataset, Box<dyn std::error::Error>> {
    let seq_a =
        read_kitti_image_sequence_dir(&args.segment_a, &args.calib_a, &args.projection_label, 0)?;
    let seq_b =
        read_kitti_image_sequence_dir(&args.segment_b, &args.calib_b, &args.projection_label, 0)?;
    build_revisit_dataset(seq_a, seq_b)
}

fn build_revisit_dataset(
    seq_a: KittiImageSequence,
    seq_b: KittiImageSequence,
) -> Result<RevisitDataset, Box<dyn std::error::Error>> {
    validate_non_empty(&seq_a, &seq_b)?;
    warn_if_camera_mismatch(&seq_a, &seq_b);
    print_segment_summary("segment_a", &seq_a);
    print_segment_summary("segment_b", &seq_b);

    let segment_a_len = seq_a.frames.len();
    let segment_b_len = seq_b.frames.len();
    let camera = seq_a.camera.clone();
    let mut images = Vec::with_capacity(segment_a_len + segment_b_len);
    let mut frame_ids = Vec::with_capacity(segment_a_len + segment_b_len);
    let mut frame_paths = Vec::with_capacity(segment_a_len + segment_b_len);
    for frame in seq_a.frames.into_iter().chain(seq_b.frames.into_iter()) {
        let kitti_id = parse_kitti_frame_id(&frame.path).ok_or_else(|| {
            format!(
                "could not parse KITTI frame id from filename {:?}",
                frame.path
            )
        })?;
        frame_ids.push(kitti_id);
        frame_paths.push((kitti_id, frame.path));
        images.push(frame.image);
    }

    let segment_a_range = id_range(&frame_ids[..segment_a_len]);
    let segment_b_range = id_range(&frame_ids[segment_a_len..]);
    println!(
        "parsed KITTI ids: segment_a [{}..{}], segment_b [{}..{}]",
        segment_a_range.0, segment_a_range.1, segment_b_range.0, segment_b_range.1,
    );
    let min_keyframe_id_gap = segment_span(segment_a_range).max(segment_span(segment_b_range));

    Ok(RevisitDataset {
        camera,
        images,
        frame_ids,
        frame_paths,
        segment_a_len,
        segment_b_len,
        segment_a_range,
        segment_b_range,
        min_keyframe_id_gap,
    })
}

fn validate_non_empty(
    seq_a: &KittiImageSequence,
    seq_b: &KittiImageSequence,
) -> Result<(), Box<dyn std::error::Error>> {
    if seq_a.frames.is_empty() || seq_b.frames.is_empty() {
        return Err(format!(
            "need at least 1 frame per segment (got A={}, B={})",
            seq_a.frames.len(),
            seq_b.frames.len(),
        )
        .into());
    }
    Ok(())
}

fn warn_if_camera_mismatch(seq_a: &KittiImageSequence, seq_b: &KittiImageSequence) {
    if seq_a.camera != seq_b.camera {
        eprintln!(
            "# warning: segment A camera differs from segment B camera; the scanner \
             assumes shared intrinsics; proceeding with segment A's camera."
        );
    }
}

fn print_segment_summary(label: &str, sequence: &KittiImageSequence) {
    println!(
        "{} frames={} (id_min={} id_max={})",
        label,
        sequence.frames.len(),
        sequence.frames.first().map(|f| f.frame_id).unwrap_or(0),
        sequence.frames.last().map(|f| f.frame_id).unwrap_or(0),
    );
}

fn parse_kitti_frame_id(path: &Path) -> Option<u64> {
    let stem = path.file_stem()?.to_str()?;
    stem.parse::<u64>().ok()
}

fn id_range(frame_ids: &[u64]) -> (u64, u64) {
    let min = frame_ids.iter().min().copied().unwrap_or(0);
    let max = frame_ids.iter().max().copied().unwrap_or(0);
    (min, max)
}

fn segment_span(range: (u64, u64)) -> u64 {
    range.1.saturating_sub(range.0) + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kitti_frame_id_from_filename() {
        assert_eq!(parse_kitti_frame_id(Path::new("004501.png")), Some(4501));
        assert_eq!(parse_kitti_frame_id(Path::new("not-a-frame.png")), None);
    }

    #[test]
    fn computes_range_and_span() {
        assert_eq!(id_range(&[49, 38, 4501]), (38, 4501));
        assert_eq!(segment_span((38, 4501)), 4464);
        assert_eq!(segment_span((10, 10)), 1);
    }
}
