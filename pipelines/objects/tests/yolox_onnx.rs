//! YOLOX through ONNX Runtime on a real image. Needs the `onnx-inference`
//! feature, an official YOLOX model and the ONNX Runtime library, so it is
//! ignored by default:
//!
//! ```text
//! curl -LO https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
//! curl -LO https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets/dog.jpg
//! ORT_DYLIB_PATH=/path/to/libonnxruntime.so VISLOC_YOLOX_ONNX=yolox_s.onnx VISLOC_YOLOX_IMAGE=dog.jpg \
//!   cargo test -p visloc-objects --features onnx-inference --test yolox_onnx -- --ignored --nocapture
//! ```
//!
//! For `yolox_nano.onnx` or `yolox_tiny.onnx` also set `VISLOC_YOLOX_INPUT=416`.
#![cfg(feature = "onnx-inference")]

use visloc_objects::{ObjectDetector, RgbImage, YoloxConfig, YoloxOnnxDetector, COCO_CLASSES};

#[test]
#[ignore = "needs a YOLOX model, a test image and the ONNX Runtime library"]
fn yolox_finds_the_dog_the_bicycle_and_the_vehicle_in_yolox_demo_image() {
    let model = std::env::var("VISLOC_YOLOX_ONNX").expect("VISLOC_YOLOX_ONNX");
    let path = std::env::var("VISLOC_YOLOX_IMAGE").expect("VISLOC_YOLOX_IMAGE");
    let size: usize = std::env::var("VISLOC_YOLOX_INPUT")
        .map(|s| s.parse().expect("VISLOC_YOLOX_INPUT"))
        .unwrap_or(640);
    let decoded = image::open(&path).expect("image").to_rgb8();
    let (width, height) = (decoded.width() as usize, decoded.height() as usize);
    let image = RgbImage::new(width, height, decoded.into_raw()).unwrap();
    let config = YoloxConfig {
        input_width: size,
        input_height: size,
        ..YoloxConfig::default()
    };
    let mut detector = YoloxOnnxDetector::load_from_path(&model, config).expect("model");
    let detections = detector.detect(&image).expect("inference");
    for d in &detections {
        let b = d.bbox;
        println!(
            "{},{:.6},{:.3},{:.3},{:.3},{:.3}",
            d.class_id, d.score, b.x_min, b.y_min, b.x_max, b.y_max
        );
    }
    let confident: Vec<&str> = detections
        .iter()
        .filter(|d| d.score > 0.5)
        .map(|d| COCO_CLASSES[d.class_id as usize])
        .collect();
    for class in ["dog", "bicycle"] {
        assert!(
            confident.contains(&class),
            "{class} missing from {confident:?}"
        );
    }
    assert!(
        confident.contains(&"car") || confident.contains(&"truck"),
        "no vehicle in {confident:?}"
    );
    assert!(detections.windows(2).all(|w| w[0].score >= w[1].score));
}
