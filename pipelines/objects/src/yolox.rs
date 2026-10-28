//! YOLOX (Ge et al., "YOLOX: Exceeding YOLO Series in 2021") pre- and
//! post-processing, independent of the inference runtime.
//!
//! This is the contract of the official ONNX exports (release `0.1.1rc0`,
//! Apache-2.0), checked against YOLOX's `demo/ONNXRuntime` pipeline:
//!
//! - **Input** `images`, `(1, 3, H, W)` f32: the image resized with its aspect
//!   ratio kept (bilinear) to fit `W × H`, placed at the top left, the rest
//!   filled with 114. Channels are **BGR**, values 0–255, not normalized.
//!   `yolox_nano` and `yolox_tiny` take 416 × 416, the larger models 640 × 640.
//! - **Output** `output`, `(1, N, 5 + C)`: per anchor `(dx, dy, log w, log h)`
//!   in its grid cell, then the objectness and `C` class probabilities, both
//!   already through a sigmoid. Anchors run over strides 8, 16 and 32, each
//!   grid row-major. C is 80 for the COCO models ([`COCO_CLASSES`]).
//!
//! The score of a class is objectness × class probability. Boxes above the
//! score threshold go through non-maximum suppression per class. The default
//! threshold is low (0.1) because [`ByteTracker`](crate::ByteTracker) uses
//! the low-score boxes in its second association.

use std::fmt;

use crate::bbox::{BBox, Detection};
use crate::image::RgbImage;

/// The 80 COCO classes of the official YOLOX models, by class index.
pub const COCO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// Input size and thresholds.
#[derive(Debug, Clone, PartialEq)]
pub struct YoloxConfig {
    pub input_width: usize,
    pub input_height: usize,
    /// Strides of the output levels, in anchor order.
    pub strides: Vec<usize>,
    /// Keep class scores above this (0.1).
    pub score_threshold: f64,
    /// Suppress a box overlapping a higher-scored box of the same class by
    /// more than this IoU (0.45, as in YOLOX's demo).
    pub nms_threshold: f64,
}

impl Default for YoloxConfig {
    fn default() -> Self {
        Self {
            input_width: 640,
            input_height: 640,
            strides: vec![8, 16, 32],
            score_threshold: 0.1,
            nms_threshold: 0.45,
        }
    }
}

impl YoloxConfig {
    /// Number of anchors the model outputs for this input size.
    pub fn anchor_count(&self) -> usize {
        self.strides
            .iter()
            .map(|s| (self.input_width / s) * (self.input_height / s))
            .sum()
    }
}

/// A preprocessed input tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Letterbox {
    /// `(3, H, W)` BGR, 0–255, row-major per channel.
    pub tensor: Vec<f32>,
    /// Factor from image to input pixels; detections are divided by it.
    pub scale: f64,
}

/// Resize `image` into the model input as described in the module docs.
///
/// The resize samples at pixel centres with edge clamping and no
/// anti-aliasing (OpenCV's `INTER_LINEAR` geometry, in floating point).
pub fn letterbox(image: &RgbImage, input_width: usize, input_height: usize) -> Letterbox {
    let (width, height) = (image.width(), image.height());
    let plane = input_width * input_height;
    let mut tensor = vec![114.0f32; 3 * plane];
    if width == 0 || height == 0 || plane == 0 {
        return Letterbox { tensor, scale: 1.0 };
    }
    let scale = (input_height as f64 / height as f64).min(input_width as f64 / width as f64);
    let resized_width = ((width as f64 * scale) as usize).min(input_width);
    let resized_height = ((height as f64 * scale) as usize).min(input_height);
    let xs = sample_positions(resized_width, width);
    let ys = sample_positions(resized_height, height);
    let data = image.data();
    for (y, &(y0, y1, fy)) in ys.iter().enumerate() {
        for (x, &(x0, x1, fx)) in xs.iter().enumerate() {
            for (bgr, rgb) in [2usize, 1, 0].into_iter().enumerate() {
                let at = |row: usize, col: usize| data[3 * (row * width + col) + rgb] as f32;
                let top = at(y0, x0) * (1.0 - fx) + at(y0, x1) * fx;
                let bottom = at(y1, x0) * (1.0 - fx) + at(y1, x1) * fx;
                tensor[bgr * plane + y * input_width + x] = top * (1.0 - fy) + bottom * fy;
            }
        }
    }
    Letterbox { tensor, scale }
}

/// For each output index: the two source indices and the weight of the second.
fn sample_positions(output: usize, input: usize) -> Vec<(usize, usize, f32)> {
    let ratio = input as f64 / output as f64;
    (0..output)
        .map(|i| {
            let position = ((i as f64 + 0.5) * ratio - 0.5).clamp(0.0, (input - 1) as f64);
            let first = position.floor() as usize;
            let second = (first + 1).min(input - 1);
            (first, second, (position - first as f64) as f32)
        })
        .collect()
}

/// The model output does not fit the configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YoloxOutputError {
    pub expected_anchors: usize,
    pub len: usize,
}

impl fmt::Display for YoloxOutputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "YOLOX output of {} values is not {} anchors × (5 + classes)",
            self.len, self.expected_anchors
        )
    }
}

impl std::error::Error for YoloxOutputError {}

/// Turn a raw output (`anchors × (5 + C)` values, the batch dimension
/// dropped) into detections in image pixels, sorted by score, highest first.
pub fn decode(
    output: &[f32],
    config: &YoloxConfig,
    scale: f64,
) -> Result<Vec<Detection>, YoloxOutputError> {
    let anchors = config.anchor_count();
    let error = YoloxOutputError {
        expected_anchors: anchors,
        len: output.len(),
    };
    if anchors == 0 || !output.len().is_multiple_of(anchors) || output.len() / anchors <= 5 {
        return Err(error);
    }
    let columns = output.len() / anchors;
    let mut candidates: Vec<Vec<Detection>> = vec![Vec::new(); columns - 5];
    let mut rows = output.chunks_exact(columns);
    for &stride in &config.strides {
        let grid_width = config.input_width / stride;
        let grid_height = config.input_height / stride;
        let s = stride as f32;
        for gy in 0..grid_height {
            for gx in 0..grid_width {
                let row = rows.next().ok_or_else(|| error.clone())?;
                let cx = (row[0] + gx as f32) * s;
                let cy = (row[1] + gy as f32) * s;
                let w = row[2].exp() * s;
                let h = row[3].exp() * s;
                let objectness = row[4];
                for (class, &probability) in row[5..].iter().enumerate() {
                    let score = objectness * probability;
                    if f64::from(score) <= config.score_threshold {
                        continue;
                    }
                    let bbox = BBox::new(
                        f64::from(cx - 0.5 * w) / scale,
                        f64::from(cy - 0.5 * h) / scale,
                        f64::from(cx + 0.5 * w) / scale,
                        f64::from(cy + 0.5 * h) / scale,
                    );
                    candidates[class].push(Detection::new(bbox, f64::from(score), class as u32));
                }
            }
        }
    }
    let mut detections: Vec<Detection> = candidates
        .into_iter()
        .flat_map(|boxes| non_maximum_suppression(boxes, config.nms_threshold))
        .collect();
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok(detections)
}

/// Greedy non-maximum suppression: keep the best box, drop the ones that
/// overlap it by more than `iou_threshold`, repeat.
pub fn non_maximum_suppression(mut boxes: Vec<Detection>, iou_threshold: f64) -> Vec<Detection> {
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for candidate in boxes {
        if kept
            .iter()
            .all(|k| k.bbox.iou(&candidate.bbox) <= iou_threshold)
        {
            kept.push(candidate);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_keeps_the_aspect_ratio_and_pads_with_114() {
        // A 4×2 image, red on the left half, blue on the right, into 8×8.
        let mut data = Vec::new();
        for _ in 0..2 {
            for x in 0..4 {
                data.extend_from_slice(if x < 2 { &[255, 0, 0] } else { &[0, 0, 255] });
            }
        }
        let image = RgbImage::new(4, 2, data).unwrap();
        let input = letterbox(&image, 8, 8);
        assert_eq!(input.scale, 2.0);
        let at = |c: usize, y: usize, x: usize| input.tensor[c * 64 + y * 8 + x];
        // Channel 0 is blue, channel 2 red.
        assert_eq!((at(0, 0, 0), at(2, 0, 0)), (0.0, 255.0));
        assert_eq!((at(0, 3, 7), at(2, 3, 7)), (255.0, 0.0));
        // Between the halves the resize blends.
        assert!(at(2, 1, 4) > 0.0 && at(2, 1, 4) < 255.0);
        // Rows 4.. are padding.
        assert!((0..3).all(|c| at(c, 4, 0) == 114.0 && at(c, 7, 7) == 114.0));
    }

    #[test]
    fn decode_places_a_box_from_its_grid_cell_and_stride() {
        let config = YoloxConfig {
            input_width: 64,
            input_height: 64,
            score_threshold: 0.3,
            ..YoloxConfig::default()
        };
        // 8×8 + 4×4 + 2×2 anchors, 5 + 2 columns.
        assert_eq!(config.anchor_count(), 84);
        let mut output = vec![0.0f32; 84 * 7];
        // Stride-16 level, cell (gx 1, gy 2): anchor 64 + 2 * 4 + 1 = 73.
        let row = &mut output[73 * 7..74 * 7];
        row.copy_from_slice(&[0.5, 0.25, 2.0f32.ln(), 1.0f32.ln(), 0.9, 0.2, 0.8]);
        let detections = decode(&output, &config, 2.0).unwrap();
        assert_eq!(detections.len(), 1, "{detections:?}");
        let d = detections[0];
        assert_eq!(d.class_id, 1);
        assert!((d.score - 0.72).abs() < 1e-6);
        // Centre (1.5 * 16, 2.25 * 16) = (24, 36), size 32 × 16, then halved.
        let expected = BBox::new(4.0, 14.0, 20.0, 22.0);
        for (a, b) in [
            (d.bbox.x_min, expected.x_min),
            (d.bbox.y_min, expected.y_min),
            (d.bbox.x_max, expected.x_max),
            (d.bbox.y_max, expected.y_max),
        ] {
            assert!((a - b).abs() < 1e-4, "{:?}", d.bbox);
        }
    }

    #[test]
    fn decode_rejects_an_output_of_the_wrong_size() {
        let config = YoloxConfig::default();
        assert!(decode(&[0.0; 100], &config, 1.0).is_err());
        assert!(decode(&vec![0.0; config.anchor_count() * 5], &config, 1.0).is_err());
    }

    #[test]
    fn suppression_is_per_class_and_keeps_the_best_box() {
        let b = |x: f64| BBox::from_xywh(x, 0.0, 10.0, 10.0);
        let boxes = vec![
            Detection::new(b(1.0), 0.6, 0),
            Detection::new(b(0.0), 0.9, 0),
            Detection::new(b(30.0), 0.5, 0),
        ];
        let kept = non_maximum_suppression(boxes, 0.45);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].score, 0.9);
        // The same box in another class survives: decode runs suppression per class.
        let config = YoloxConfig {
            input_width: 32,
            input_height: 32,
            strides: vec![32],
            score_threshold: 0.1,
            nms_threshold: 0.45,
        };
        let output = [0.5, 0.5, 0.0, 0.0, 1.0, 0.9, 0.8];
        let detections = decode(&output, &config, 1.0).unwrap();
        assert_eq!(
            detections.iter().map(|d| d.class_id).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }
}
