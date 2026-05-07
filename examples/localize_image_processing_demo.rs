use std::error::Error;
use std::fmt;
use std::fs::File;
use std::path::Path;

use image::codecs::gif::{GifEncoder, Repeat};
use image::imageops::{resize, FilterType};
use image::{Delay, DynamicImage, Frame, GrayImage, Rgb, RgbImage, RgbaImage};
use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Landmark, VisualMap};
use visloc_rs::{FeatureExtractor, FeatureSet, ImageLocalizer};

const IMAGE_WIDTH: u32 = 640;
const IMAGE_HEIGHT: u32 = 480;
const PATCH_SIZE: usize = 23;
const PATCH_RADIUS: i32 = (PATCH_SIZE as i32) / 2;

#[derive(Debug, Clone)]
struct PatchLandmark {
    id: u64,
}

impl PatchLandmark {
    fn template(&self) -> [[u8; PATCH_SIZE]; PATCH_SIZE] {
        let mut template = [[128_u8; PATCH_SIZE]; PATCH_SIZE];
        for (y, row) in template.iter_mut().enumerate() {
            for (x, value) in row.iter_mut().enumerate() {
                let border = x == 0 || y == 0 || x == PATCH_SIZE - 1 || y == PATCH_SIZE - 1;
                let center_line = x == PATCH_SIZE / 2 || y == PATCH_SIZE / 2;
                *value = if border {
                    245
                } else if center_line {
                    18
                } else if texture_bit(self.id, x as u32, y as u32) {
                    220
                } else {
                    42
                };
            }
        }
        template
    }

    fn descriptor(&self) -> Vec<f32> {
        self.template()
            .iter()
            .flatten()
            .map(|value| *value as f32 / 255.0)
            .collect()
    }
}

#[derive(Debug, Clone)]
struct PatchWindowExtractor {
    expected_feature_count: usize,
}

impl FeatureExtractor for PatchWindowExtractor {
    type Image = DynamicImage;
    type Error = DemoError;

    fn extract(&self, image: &Self::Image) -> Result<FeatureSet, Self::Error> {
        let gray = image.to_luma8();
        let centers = detect_patch_windows(&gray);
        if centers.len() != self.expected_feature_count {
            return Err(DemoError(format!(
                "expected {} visual patches, detected {}",
                self.expected_feature_count,
                centers.len()
            )));
        }

        let mut keypoints = Vec::with_capacity(centers.len());
        let mut descriptors = Vec::with_capacity(centers.len());
        for (x, y) in centers {
            keypoints.push(Point2::new(x as f64, y as f64));
            descriptors.push(extract_patch_descriptor(&gray, x, y)?);
        }

        FeatureSet::new(keypoints, descriptors).map_err(|error| DemoError(error.to_string()))
    }
}

#[derive(Debug, Clone)]
struct DemoError(String);

impl fmt::Display for DemoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Error for DemoError {}

fn main() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let query_path = root.join("examples/data/query_frame.png");
    let overlay_path = root.join("docs/assets/image-processing-demo.png");
    let gif_path = root.join("docs/assets/image-processing-demo.gif");

    let camera = Camera::pinhole(1, IMAGE_WIDTH, IMAGE_HEIGHT, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let landmarks = demo_landmarks();
    let points = demo_points();

    render_query_image(&query_path, &camera, &pose, &points, &landmarks)?;

    let query_image = image::open(&query_path)?;
    let extractor = PatchWindowExtractor {
        expected_feature_count: landmarks.len(),
    };
    let features = extractor.extract(&query_image)?;
    let map = build_demo_map(&camera, &points, &landmarks);
    let localizer = ImageLocalizer::new(extractor);
    let result = localizer.localize_image(&query_image, camera, &map)?;

    let raw = query_image.to_rgb8();
    let detected = draw_detected_features(&raw, &features.keypoints);
    let localized = draw_localization_overlay(
        &raw,
        &features.keypoints,
        result.success,
        result.inlier_count,
    );
    localized.save(&overlay_path)?;
    write_demo_gif(&gif_path, &raw, &detected, &localized)?;

    println!("input image: {}", query_path.display());
    println!("output overlay: {}", overlay_path.display());
    println!("output gif: {}", gif_path.display());
    println!("detected patch features: {}", features.keypoints.len());
    println!("success: {}", result.success);
    println!("matches: {}", result.match_count);
    println!("inliers: {}", result.inlier_count);
    println!("mean reprojection error: {:?}", result.reprojection_error);
    println!("pose: {:#?}", result.pose);

    Ok(())
}

fn build_demo_map(
    camera: &Camera,
    points: &[Point3<f64>],
    landmarks: &[PatchLandmark],
) -> VisualMap {
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    for (point, landmark_patch) in points.iter().zip(landmarks.iter()) {
        let mut landmark = Landmark::new(landmark_patch.id, *point);
        landmark.descriptor = Some(landmark_patch.descriptor());
        map.landmarks.insert(landmark.id, landmark);
    }
    map
}

fn render_query_image(
    path: &Path,
    camera: &Camera,
    pose: &Pose,
    points: &[Point3<f64>],
    landmarks: &[PatchLandmark],
) -> Result<(), Box<dyn Error>> {
    let mut image = RgbImage::from_pixel(IMAGE_WIDTH, IMAGE_HEIGHT, Rgb([219, 232, 240]));

    for y in 0..250 {
        let t = y as f32 / 250.0;
        let color = lerp_rgb(Rgb([132, 184, 218]), Rgb([235, 242, 247]), t);
        for x in 0..IMAGE_WIDTH {
            image.put_pixel(x, y, color);
        }
    }

    fill_polygon(
        &mut image,
        &[(0, 480), (170, 246), (470, 246), (640, 480)],
        Rgb([55, 64, 76]),
    );
    draw_line(&mut image, (238, 250), (170, 480), Rgb([230, 238, 246]), 5);
    draw_line(&mut image, (402, 250), (470, 480), Rgb([230, 238, 246]), 5);
    draw_dashed_line(
        &mut image,
        (320, 275),
        (320, 480),
        Rgb([249, 199, 79]),
        5,
        24,
        18,
    );

    draw_building(
        &mut image,
        42,
        120,
        168,
        255,
        Rgb([190, 168, 142]),
        Rgb([69, 111, 142]),
    );
    draw_building(
        &mut image,
        448,
        70,
        150,
        316,
        Rgb([170, 182, 194]),
        Rgb([71, 116, 147]),
    );
    draw_building(
        &mut image,
        236,
        162,
        170,
        116,
        Rgb([185, 193, 201]),
        Rgb([74, 118, 150]),
    );

    for (point, landmark_patch) in points.iter().zip(landmarks.iter()) {
        let pixel = camera
            .project(&pose.transform_world_point(point))
            .ok_or("demo point projected behind the camera")?;
        draw_patch(
            &mut image,
            (pixel.x.round() as i32, pixel.y.round() as i32),
            &landmark_patch.template(),
        );
    }

    image.save(path)?;
    Ok(())
}

fn extract_patch_descriptor(
    image: &GrayImage,
    center_x: i32,
    center_y: i32,
) -> Result<Vec<f32>, DemoError> {
    if center_x < PATCH_RADIUS
        || center_y < PATCH_RADIUS
        || center_x >= image.width() as i32 - PATCH_RADIUS
        || center_y >= image.height() as i32 - PATCH_RADIUS
    {
        return Err(DemoError(
            "detected patch is too close to the image border".to_owned(),
        ));
    }

    let mut descriptor = Vec::with_capacity(PATCH_SIZE * PATCH_SIZE);
    for y in center_y - PATCH_RADIUS..=center_y + PATCH_RADIUS {
        for x in center_x - PATCH_RADIUS..=center_x + PATCH_RADIUS {
            descriptor.push(image.get_pixel(x as u32, y as u32)[0] as f32 / 255.0);
        }
    }
    Ok(descriptor)
}

fn detect_patch_windows(image: &GrayImage) -> Vec<(i32, i32)> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut visited = vec![false; width * height];
    let mut centers = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if visited[index] || image.get_pixel(x as u32, y as u32)[0] < 242 {
                continue;
            }

            let mut stack = vec![(x as i32, y as i32)];
            visited[index] = true;
            let mut min_x = x as i32;
            let mut max_x = x as i32;
            let mut min_y = y as i32;
            let mut max_y = y as i32;
            let mut count = 0_usize;

            while let Some((cx, cy)) = stack.pop() {
                count += 1;
                min_x = min_x.min(cx);
                max_x = max_x.max(cx);
                min_y = min_y.min(cy);
                max_y = max_y.max(cy);

                for (nx, ny) in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)] {
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        continue;
                    }
                    let nindex = ny as usize * width + nx as usize;
                    if visited[nindex] || image.get_pixel(nx as u32, ny as u32)[0] < 242 {
                        continue;
                    }
                    visited[nindex] = true;
                    stack.push((nx, ny));
                }
            }

            let box_width = max_x - min_x + 1;
            let box_height = max_y - min_y + 1;
            if (26..=32).contains(&box_width)
                && (26..=32).contains(&box_height)
                && (220..=460).contains(&count)
            {
                centers.push(((min_x + max_x) / 2, (min_y + max_y) / 2));
            }
        }
    }

    centers.sort_by_key(|(x, y)| (*y, *x));
    centers
}

fn draw_patch(image: &mut RgbImage, center: (i32, i32), template: &[[u8; PATCH_SIZE]; PATCH_SIZE]) {
    draw_rect(
        image,
        center.0 - PATCH_RADIUS - 3,
        center.1 - PATCH_RADIUS - 3,
        PATCH_SIZE as i32 + 6,
        PATCH_SIZE as i32 + 6,
        Rgb([245, 247, 250]),
    );
    for (y, row) in template.iter().enumerate() {
        for (x, value) in row.iter().enumerate() {
            put_pixel_checked(
                image,
                center.0 + x as i32 - PATCH_RADIUS,
                center.1 + y as i32 - PATCH_RADIUS,
                Rgb([*value, *value, *value]),
            );
        }
    }
}

fn draw_detected_features(image: &RgbImage, keypoints: &[Point2<f64>]) -> RgbImage {
    let mut output = image.clone();
    for keypoint in keypoints {
        let center = (keypoint.x.round() as i32, keypoint.y.round() as i32);
        draw_circle(&mut output, center, 21, Rgb([53, 208, 186]), 3);
        draw_line(
            &mut output,
            (center.0 - 27, center.1),
            (center.0 + 27, center.1),
            Rgb([53, 208, 186]),
            2,
        );
        draw_line(
            &mut output,
            (center.0, center.1 - 27),
            (center.0, center.1 + 27),
            Rgb([53, 208, 186]),
            2,
        );
    }
    output
}

fn draw_localization_overlay(
    image: &RgbImage,
    keypoints: &[Point2<f64>],
    success: bool,
    inlier_count: usize,
) -> RgbImage {
    let mut output = draw_detected_features(image, keypoints);
    for keypoint in keypoints {
        let center = (keypoint.x.round() as i32, keypoint.y.round() as i32);
        draw_line(&mut output, center, (320, 430), Rgb([249, 199, 79]), 2);
    }

    draw_filled_circle(&mut output, (320, 430), 11, Rgb([112, 165, 255]));
    draw_line(&mut output, (320, 430), (390, 430), Rgb([239, 68, 68]), 5);
    draw_line(&mut output, (320, 430), (320, 360), Rgb([34, 197, 94]), 5);
    draw_line(&mut output, (320, 430), (366, 384), Rgb([59, 130, 246]), 5);

    let badge = if success {
        Rgb([16, 122, 89])
    } else {
        Rgb([140, 45, 45])
    };
    draw_rect(&mut output, 22, 22, 258, 62, Rgb([11, 18, 32]));
    draw_rect(&mut output, 31, 31, 48, 44, badge);
    draw_digit_text(
        &mut output,
        96,
        42,
        inlier_count as u32,
        Rgb([246, 247, 251]),
    );

    output
}

fn write_demo_gif(
    path: &Path,
    raw: &RgbImage,
    detected: &RgbImage,
    localized: &RgbImage,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    let mut encoder = GifEncoder::new(&mut file);
    encoder.set_repeat(Repeat::Infinite)?;
    let frames = [raw, detected, localized].into_iter().map(|image| {
        let resized = resize(image, 640, 480, FilterType::Lanczos3);
        Frame::from_parts(
            RgbaImage::from_fn(resized.width(), resized.height(), |x, y| {
                let p = resized.get_pixel(x, y);
                image::Rgba([p[0], p[1], p[2], 255])
            }),
            0,
            0,
            Delay::from_numer_denom_ms(900, 1),
        )
    });
    encoder.encode_frames(frames)?;
    Ok(())
}

fn demo_landmarks() -> Vec<PatchLandmark> {
    (1..=8).map(|id| PatchLandmark { id }).collect()
}

fn texture_bit(id: u64, x: u32, y: u32) -> bool {
    let mut value = id
        .wrapping_mul(0x9E37_79B1_85EB_CA87)
        .wrapping_add((x as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
        .wrapping_add((y as u64).wrapping_mul(0x1656_67B1_9E37_79F9));
    value ^= value >> 33;
    value = value.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    value ^= value >> 29;
    value & 1 == 1
}

fn demo_points() -> Vec<Point3<f64>> {
    vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
        Point3::new(-0.5, 0.4, 6.5),
        Point3::new(0.25, 0.75, 8.0),
    ]
}

fn lerp_rgb(a: Rgb<u8>, b: Rgb<u8>, t: f32) -> Rgb<u8> {
    Rgb([
        (a[0] as f32 * (1.0 - t) + b[0] as f32 * t) as u8,
        (a[1] as f32 * (1.0 - t) + b[1] as f32 * t) as u8,
        (a[2] as f32 * (1.0 - t) + b[2] as f32 * t) as u8,
    ])
}

fn fill_polygon(image: &mut RgbImage, points: &[(i32, i32)], color: Rgb<u8>) {
    let min_y = points.iter().map(|(_, y)| *y).min().unwrap_or(0).max(0);
    let max_y = points
        .iter()
        .map(|(_, y)| *y)
        .max()
        .unwrap_or(0)
        .min(image.height() as i32 - 1);
    for y in min_y..=max_y {
        let mut intersections = Vec::new();
        for index in 0..points.len() {
            let (x1, y1) = points[index];
            let (x2, y2) = points[(index + 1) % points.len()];
            if (y1 <= y && y < y2) || (y2 <= y && y < y1) {
                let t = (y - y1) as f64 / (y2 - y1) as f64;
                intersections.push((x1 as f64 + t * (x2 - x1) as f64).round() as i32);
            }
        }
        intersections.sort_unstable();
        for pair in intersections.chunks(2) {
            if let [start, end] = pair {
                for x in (*start).max(0)..=(*end).min(image.width() as i32 - 1) {
                    image.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
}

fn draw_building(
    image: &mut RgbImage,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    wall: Rgb<u8>,
    glass: Rgb<u8>,
) {
    draw_rect(image, x, y, width, height, wall);
    let window_width = (width / 5).max(16);
    let window_height = (height / 5).max(22);
    for row in 0..3 {
        for col in 0..3 {
            let wx = x + 18 + col * (window_width + 18);
            let wy = y + 22 + row * (window_height + 18);
            if wx + window_width < x + width - 10 && wy + window_height < y + height - 10 {
                draw_rect(image, wx, wy, window_width, window_height, glass);
            }
        }
    }
    draw_rect(
        image,
        x - 14,
        y + height,
        width + 28,
        34,
        darken(wall, 0.72),
    );
}

fn draw_rect(image: &mut RgbImage, x: i32, y: i32, width: i32, height: i32, color: Rgb<u8>) {
    for yy in y.max(0)..(y + height).min(image.height() as i32) {
        for xx in x.max(0)..(x + width).min(image.width() as i32) {
            image.put_pixel(xx as u32, yy as u32, color);
        }
    }
}

fn draw_filled_circle(image: &mut RgbImage, center: (i32, i32), radius: i32, color: Rgb<u8>) {
    let r2 = radius * radius;
    for y in center.1 - radius..=center.1 + radius {
        for x in center.0 - radius..=center.0 + radius {
            if (x - center.0).pow(2) + (y - center.1).pow(2) <= r2 {
                put_pixel_checked(image, x, y, color);
            }
        }
    }
}

fn draw_circle(image: &mut RgbImage, center: (i32, i32), radius: i32, color: Rgb<u8>, width: i32) {
    for offset in 0..width {
        let r = radius - offset;
        let r2 = r * r;
        for y in center.1 - r..=center.1 + r {
            for x in center.0 - r..=center.0 + r {
                let distance = (x - center.0).pow(2) + (y - center.1).pow(2);
                if (distance - r2).abs() <= r {
                    put_pixel_checked(image, x, y, color);
                }
            }
        }
    }
}

fn draw_dashed_line(
    image: &mut RgbImage,
    start: (i32, i32),
    end: (i32, i32),
    color: Rgb<u8>,
    width: i32,
    dash: i32,
    gap: i32,
) {
    let steps = (end.0 - start.0).abs().max((end.1 - start.1).abs()).max(1);
    for step in 0..=steps {
        let cycle = dash + gap;
        if step % cycle < dash {
            let t = step as f64 / steps as f64;
            let x = (start.0 as f64 + (end.0 - start.0) as f64 * t).round() as i32;
            let y = (start.1 as f64 + (end.1 - start.1) as f64 * t).round() as i32;
            draw_filled_circle(image, (x, y), width / 2, color);
        }
    }
}

fn draw_line(image: &mut RgbImage, start: (i32, i32), end: (i32, i32), color: Rgb<u8>, width: i32) {
    let steps = (end.0 - start.0).abs().max((end.1 - start.1).abs()).max(1);
    for step in 0..=steps {
        let t = step as f64 / steps as f64;
        let x = (start.0 as f64 + (end.0 - start.0) as f64 * t).round() as i32;
        let y = (start.1 as f64 + (end.1 - start.1) as f64 * t).round() as i32;
        draw_filled_circle(image, (x, y), width / 2, color);
    }
}

fn put_pixel_checked(image: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>) {
    if x >= 0 && y >= 0 && x < image.width() as i32 && y < image.height() as i32 {
        image.put_pixel(x as u32, y as u32, color);
    }
}

fn darken(color: Rgb<u8>, factor: f32) -> Rgb<u8> {
    Rgb([
        (color[0] as f32 * factor) as u8,
        (color[1] as f32 * factor) as u8,
        (color[2] as f32 * factor) as u8,
    ])
}

fn draw_digit_text(image: &mut RgbImage, x: i32, y: i32, value: u32, color: Rgb<u8>) {
    let text = format!("inliers {value}");
    for (index, byte) in text.bytes().enumerate() {
        draw_tiny_glyph(image, x + index as i32 * 14, y, byte, color);
    }
}

fn draw_tiny_glyph(image: &mut RgbImage, x: i32, y: i32, byte: u8, color: Rgb<u8>) {
    let pattern = match byte {
        b'0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        b'1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        b'2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        b'3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        b'4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        b'5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        b'6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        b'7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        b'8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        b'9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        b'i' => [0b010, 0b000, 0b010, 0b010, 0b010],
        b'n' => [0b000, 0b110, 0b101, 0b101, 0b101],
        b'l' => [0b010, 0b010, 0b010, 0b010, 0b011],
        b'e' => [0b000, 0b111, 0b110, 0b100, 0b111],
        b'r' => [0b000, 0b110, 0b101, 0b100, 0b100],
        b's' => [0b000, 0b111, 0b100, 0b111, 0b001],
        _ => [0; 5],
    };
    for (row, bits) in pattern.iter().enumerate() {
        for col in 0..3 {
            if bits & (1 << (2 - col)) != 0 {
                draw_rect(image, x + col * 4, y + row as i32 * 5, 3, 4, color);
            }
        }
    }
}
