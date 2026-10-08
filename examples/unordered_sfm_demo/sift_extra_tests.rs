use super::{
    append_spatially_novel_keypoints, effective_extra_contrast_threshold,
    extract_sift_with_split_grayscale,
};
use visloc_rs::vision::features::sift::{
    describe_sift_keypoints, extract_sift, GrayImage, SiftConfig,
};
fn dot_texture(width: usize, height: usize) -> Vec<f32> {
    let mut state = 7u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut pixels = vec![0.15f32; width * height];
    for _ in 0..(width * height / 24) {
        let cx = next() * width as f64;
        let cy = next() * height as f64;
        let bright = 0.55 + 0.35 * next();
        let radius = 1.5 + 2.5 * next();
        let x0 = cx.floor() as i64 - 6;
        let y0 = cy.floor() as i64 - 6;
        for y in y0..=y0 + 12 {
            for x in x0..=x0 + 12 {
                if x < 0 || y < 0 || x >= width as i64 || y >= height as i64 {
                    continue;
                }
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let gaussian = (-(dx * dx + dy * dy) / (2.0 * radius * radius)).exp();
                let index = y as usize * width + x as usize;
                pixels[index] = (pixels[index] + (bright * gaussian) as f32).min(1.0);
            }
        }
    }
    pixels
}

#[test]
fn extra_threshold_preserves_primary_prefix_and_default_resolution() {
    let width = 128usize;
    let height = 128usize;
    let pixels = dot_texture(width, height);
    let image = GrayImage::new(width, height, &pixels).unwrap();
    let primary_config = SiftConfig {
        max_keypoints: 32,
        octaves: 2,
        contrast_threshold: 0.02,
        ..SiftConfig::default()
    };
    let dense_config = |threshold| SiftConfig {
        max_keypoints: usize::MAX,
        octaves: 2,
        contrast_threshold: threshold,
        ..SiftConfig::default()
    };
    let (primary_keypoints, primary_descriptors) = extract_sift(&image, &primary_config).unwrap();
    let (same_keypoints, same_descriptors) = extract_sift(&image, &dense_config(0.02)).unwrap();
    let (low_keypoints, low_descriptors) = extract_sift(&image, &dense_config(0.01)).unwrap();

    let mut merged_same_keypoints = primary_keypoints.clone();
    let mut merged_same_descriptors = primary_descriptors.clone();
    let same_added = append_spatially_novel_keypoints(
        &mut merged_same_keypoints,
        &mut merged_same_descriptors,
        same_keypoints,
        same_descriptors,
        32,
    );
    let mut merged_low_keypoints = primary_keypoints.clone();
    let mut merged_low_descriptors = primary_descriptors.clone();
    let low_added = append_spatially_novel_keypoints(
        &mut merged_low_keypoints,
        &mut merged_low_descriptors,
        low_keypoints,
        low_descriptors,
        32,
    );

    let primary_len = primary_keypoints.len();
    assert_eq!(&merged_same_keypoints[..primary_len], &primary_keypoints);
    assert_eq!(
        &merged_same_descriptors[..primary_len],
        &primary_descriptors
    );
    assert_eq!(&merged_low_keypoints[..primary_len], &primary_keypoints);
    assert_eq!(&merged_low_descriptors[..primary_len], &primary_descriptors);
    assert!(
        low_added >= same_added,
        "low threshold added {low_added}, same threshold added {same_added}"
    );
    assert_eq!(
        effective_extra_contrast_threshold(None, 0.02),
        0.02,
        "omitting the extra threshold must preserve legacy extraction"
    );
    assert_eq!(effective_extra_contrast_threshold(Some(0.01), 0.02), 0.01);
}

#[test]
fn split_grayscale_keeps_rounded_detector_and_floor_descriptors() {
    let width = 128usize;
    let height = 128usize;
    let source = dot_texture(width, height);
    let floor_pixels: Vec<f32> = source
        .iter()
        .map(|&value| ((value.clamp(0.0, 1.0) * 255.0).floor() as u8) as f32 / 255.0)
        .collect();
    let rounded_pixels: Vec<f32> = source
        .iter()
        .map(|&value| ((value.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8) as f32 / 255.0)
        .collect();
    let floor_image = GrayImage::new(width, height, &floor_pixels).unwrap();
    let rounded_image = GrayImage::new(width, height, &rounded_pixels).unwrap();
    let config = SiftConfig {
        max_keypoints: 64,
        octaves: 1,
        max_orientations: 2,
        vlfeat_compatible_detector: true,
        vlfeat_compatible_descriptor: true,
        vlfeat_bilinear_orientations: true,
        ..SiftConfig::default()
    };

    let (split_keypoints, split_descriptors) =
        extract_sift_with_split_grayscale(&rounded_image, &floor_image, &config).unwrap();
    let (expected_keypoints, _) = extract_sift(&rounded_image, &config).unwrap();
    let expected_descriptors = describe_sift_keypoints(&floor_image, &expected_keypoints, &config);

    assert_eq!(split_keypoints, expected_keypoints);
    assert_eq!(split_descriptors, expected_descriptors);
    assert_eq!(split_keypoints.len(), split_descriptors.len());
}
