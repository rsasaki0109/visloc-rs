use std::error::Error;
use std::fmt;
use std::fs::File;
use std::path::Path;

use image::codecs::gif::{GifEncoder, Repeat};
use image::imageops::{resize, FilterType};
use image::{Delay, DynamicImage, Frame, Rgb, RgbImage, RgbaImage};
use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Landmark, VisualMap};
use visloc_rs::{FeatureExtractor, FeatureSet, ImageLocalizer};

#[derive(Debug, Clone)]
struct MarkerSpec {
    id: u64,
    color: Rgb<u8>,
}

impl MarkerSpec {
    fn descriptor(&self) -> Vec<f32> {
        vec![
            self.id as f32,
            self.color[0] as f32,
            self.color[1] as f32,
            self.color[2] as f32,
        ]
    }
}

#[derive(Debug, Clone)]
struct ColorBlobExtractor {
    markers: Vec<MarkerSpec>,
}

impl FeatureExtractor for ColorBlobExtractor {
    type Image = DynamicImage;
    type Error = DemoError;

    fn extract(&self, image: &Self::Image) -> Result<FeatureSet, Self::Error> {
        let rgb = image.to_rgb8();
        let mut keypoints = Vec::with_capacity(self.markers.len());
        let mut descriptors = Vec::with_capacity(self.markers.len());

        for marker in &self.markers {
            let mut count = 0.0;
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;

            for (x, y, pixel) in rgb.enumerate_pixels() {
                if color_distance_sq(pixel, &marker.color) <= 26 * 26 {
                    count += 1.0;
                    sum_x += x as f64;
                    sum_y += y as f64;
                }
            }

            if count < 40.0 {
                return Err(DemoError(format!(
                    "marker {} was not detected strongly enough; matching pixels={count}",
                    marker.id
                )));
            }

            keypoints.push(Point2::new(sum_x / count, sum_y / count));
            descriptors.push(marker.descriptor());
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

    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let markers = demo_markers();
    let points = demo_points();

    render_query_image(&query_path, &camera, &pose, &points, &markers)?;

    let query_image = image::open(&query_path)?;
    let extractor = ColorBlobExtractor {
        markers: markers.clone(),
    };
    let features = extractor.extract(&query_image)?;
    let map = build_demo_map(&camera, &points, &markers);
    let localizer = ImageLocalizer::new(extractor);
    let result = localizer.localize_image(&query_image, camera, &map)?;

    let detected = draw_detected_features(&query_image.to_rgb8(), &features.keypoints);
    let localized = draw_localization_overlay(
        &query_image.to_rgb8(),
        &features.keypoints,
        result.success,
        result.inlier_count,
    );
    localized.save(&overlay_path)?;
    write_demo_gif(&gif_path, &query_image.to_rgb8(), &detected, &localized)?;

    println!("input image: {}", query_path.display());
    println!("output overlay: {}", overlay_path.display());
    println!("output gif: {}", gif_path.display());
    println!("detected image features: {}", features.keypoints.len());
    println!("success: {}", result.success);
    println!("matches: {}", result.match_count);
    println!("inliers: {}", result.inlier_count);
    println!("mean reprojection error: {:?}", result.reprojection_error);
    println!("pose: {:#?}", result.pose);

    Ok(())
}

fn build_demo_map(camera: &Camera, points: &[Point3<f64>], markers: &[MarkerSpec]) -> VisualMap {
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    for (point, marker) in points.iter().zip(markers.iter()) {
        let mut landmark = Landmark::new(marker.id, *point);
        landmark.descriptor = Some(marker.descriptor());
        map.landmarks.insert(landmark.id, landmark);
    }
    map
}

fn render_query_image(
    path: &Path,
    camera: &Camera,
    pose: &Pose,
    points: &[Point3<f64>],
    markers: &[MarkerSpec],
) -> Result<(), Box<dyn Error>> {
    let mut image = RgbImage::from_pixel(960, 640, Rgb([218, 232, 242]));

    for y in 0..330 {
        let t = y as f32 / 330.0;
        let color = lerp_rgb(Rgb([139, 190, 221]), Rgb([238, 244, 248]), t);
        for x in 0..960 {
            image.put_pixel(x, y, color);
        }
    }

    fill_polygon(
        &mut image,
        &[(0, 640), (260, 330), (700, 330), (960, 640)],
        Rgb([55, 64, 76]),
    );
    draw_line(&mut image, (333, 334), (250, 640), Rgb([230, 238, 246]), 5);
    draw_line(&mut image, (627, 334), (710, 640), Rgb([230, 238, 246]), 5);
    draw_dashed_line(
        &mut image,
        (480, 360),
        (480, 640),
        Rgb([249, 199, 79]),
        5,
        26,
        20,
    );

    draw_building(
        &mut image,
        70,
        155,
        230,
        325,
        Rgb([191, 166, 137]),
        Rgb([66, 111, 143]),
    );
    draw_building(
        &mut image,
        674,
        98,
        214,
        390,
        Rgb([172, 184, 196]),
        Rgb([71, 116, 147]),
    );
    draw_building(
        &mut image,
        355,
        210,
        253,
        150,
        Rgb([185, 193, 201]),
        Rgb([74, 118, 150]),
    );

    draw_line(&mut image, (110, 470), (420, 286), Rgb([27, 37, 52]), 2);
    draw_line(&mut image, (420, 286), (480, 343), Rgb([27, 37, 52]), 2);
    draw_line(&mut image, (480, 343), (548, 279), Rgb([27, 37, 52]), 2);
    draw_line(&mut image, (548, 279), (805, 482), Rgb([27, 37, 52]), 2);

    for (point, marker) in points.iter().zip(markers.iter()) {
        let pixel = camera
            .project(&pose.transform_world_point(point))
            .ok_or("demo point projected behind the camera")?;
        let center = (pixel.x.round() as i32, pixel.y.round() as i32);
        draw_filled_circle(&mut image, center, 14, Rgb([10, 16, 32]));
        draw_filled_circle(&mut image, center, 9, marker.color);
        draw_circle(&mut image, center, 18, Rgb([246, 247, 251]), 2);
    }

    image.save(path)?;
    Ok(())
}

fn draw_detected_features(image: &RgbImage, keypoints: &[Point2<f64>]) -> RgbImage {
    let mut output = image.clone();
    for keypoint in keypoints {
        let center = (keypoint.x.round() as i32, keypoint.y.round() as i32);
        draw_circle(&mut output, center, 24, Rgb([53, 208, 186]), 4);
        draw_line(
            &mut output,
            (center.0 - 30, center.1),
            (center.0 + 30, center.1),
            Rgb([53, 208, 186]),
            2,
        );
        draw_line(
            &mut output,
            (center.0, center.1 - 30),
            (center.0, center.1 + 30),
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
        draw_line(&mut output, center, (480, 560), Rgb([249, 199, 79]), 2);
    }

    draw_filled_circle(&mut output, (480, 560), 12, Rgb([112, 165, 255]));
    draw_line(&mut output, (480, 560), (560, 560), Rgb([239, 68, 68]), 5);
    draw_line(&mut output, (480, 560), (480, 480), Rgb([34, 197, 94]), 5);
    draw_line(&mut output, (480, 560), (532, 508), Rgb([59, 130, 246]), 5);

    let badge = if success {
        Rgb([16, 122, 89])
    } else {
        Rgb([140, 45, 45])
    };
    draw_rect(&mut output, 30, 28, 335, 74, Rgb([11, 18, 32]));
    draw_rect(&mut output, 38, 36, 68, 58, badge);
    draw_digit_text(
        &mut output,
        120,
        50,
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
        let resized = resize(image, 720, 480, FilterType::Lanczos3);
        Frame::from_parts(
            RgbaImage::from_fn(resized.width(), resized.height(), |x, y| {
                let p = resized.get_pixel(x, y);
                image::Rgba([p[0], p[1], p[2], 255])
            }),
            0,
            0,
            Delay::from_numer_denom_ms(950, 1),
        )
    });
    encoder.encode_frames(frames)?;
    Ok(())
}

fn demo_markers() -> Vec<MarkerSpec> {
    vec![
        MarkerSpec {
            id: 1,
            color: Rgb([239, 68, 68]),
        },
        MarkerSpec {
            id: 2,
            color: Rgb([53, 208, 186]),
        },
        MarkerSpec {
            id: 3,
            color: Rgb([112, 165, 255]),
        },
        MarkerSpec {
            id: 4,
            color: Rgb([249, 199, 79]),
        },
        MarkerSpec {
            id: 5,
            color: Rgb([168, 85, 247]),
        },
        MarkerSpec {
            id: 6,
            color: Rgb([14, 165, 233]),
        },
        MarkerSpec {
            id: 7,
            color: Rgb([34, 197, 94]),
        },
        MarkerSpec {
            id: 8,
            color: Rgb([251, 113, 133]),
        },
    ]
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

fn color_distance_sq(a: &Rgb<u8>, b: &Rgb<u8>) -> i32 {
    let dr = a[0] as i32 - b[0] as i32;
    let dg = a[1] as i32 - b[1] as i32;
    let db = a[2] as i32 - b[2] as i32;
    dr * dr + dg * dg + db * db
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
    let window_width = (width / 5).max(18);
    let window_height = (height / 5).max(26);
    for row in 0..3 {
        for col in 0..3 {
            let wx = x + 22 + col * (window_width + 22);
            let wy = y + 28 + row * (window_height + 22);
            if wx + window_width < x + width - 12 && wy + window_height < y + height - 12 {
                draw_rect(image, wx, wy, window_width, window_height, glass);
            }
        }
    }
    draw_rect(
        image,
        x - 18,
        y + height,
        width + 36,
        42,
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
