//! Axis-aligned image boxes and detections.

/// An axis-aligned box in pixel coordinates, `[x_min, x_max] × [y_min, y_max]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x_min: f64,
    pub y_min: f64,
    pub x_max: f64,
    pub y_max: f64,
}

impl BBox {
    pub const fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Self {
        Self {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    /// A box from its top-left corner and size.
    pub fn from_xywh(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self::new(x, y, x + width, y + height)
    }

    /// A box from its centre and size.
    pub fn from_center_size(cx: f64, cy: f64, width: f64, height: f64) -> Self {
        Self::new(
            cx - 0.5 * width,
            cy - 0.5 * height,
            cx + 0.5 * width,
            cy + 0.5 * height,
        )
    }

    pub fn width(&self) -> f64 {
        (self.x_max - self.x_min).max(0.0)
    }

    pub fn height(&self) -> f64 {
        (self.y_max - self.y_min).max(0.0)
    }

    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    pub fn center(&self) -> (f64, f64) {
        (
            0.5 * (self.x_min + self.x_max),
            0.5 * (self.y_min + self.y_max),
        )
    }

    /// Finite corners and a positive width and height.
    pub fn is_valid(&self) -> bool {
        [self.x_min, self.y_min, self.x_max, self.y_max]
            .iter()
            .all(|v| v.is_finite())
            && self.x_max > self.x_min
            && self.y_max > self.y_min
    }

    /// This box, found in an image turned `degrees` clockwise (0, 90, 180 or
    /// 270; other values leave it as it is), in the pixels of the original
    /// `width` × `height` image. For cameras mounted sideways, whose frames
    /// are turned upright before detection.
    pub fn unrotated(&self, degrees: u32, width: f64, height: f64) -> BBox {
        let b = self;
        match degrees {
            // Turned clockwise by 90: original (x, y) is at (height - y, x).
            90 => BBox::new(b.y_min, height - b.x_max, b.y_max, height - b.x_min),
            180 => BBox::new(
                width - b.x_max,
                height - b.y_max,
                width - b.x_min,
                height - b.y_min,
            ),
            // Turned clockwise by 270: original (x, y) is at (y, width - x).
            270 => BBox::new(width - b.y_max, b.x_min, width - b.y_min, b.x_max),
            _ => *b,
        }
    }

    /// Intersection over union, 0 when the boxes do not overlap or are empty.
    pub fn iou(&self, other: &BBox) -> f64 {
        let width = (self.x_max.min(other.x_max) - self.x_min.max(other.x_min)).max(0.0);
        let height = (self.y_max.min(other.y_max) - self.y_min.max(other.y_min)).max(0.0);
        let intersection = width * height;
        let union = self.area() + other.area() - intersection;
        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }
}

/// One detector output: a box, its confidence and its class.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub bbox: BBox,
    /// Detector confidence, in [0, 1].
    pub score: f64,
    /// The detector's class index (for example a COCO label).
    pub class_id: u32,
}

impl Detection {
    pub const fn new(bbox: BBox, score: f64, class_id: u32) -> Self {
        Self {
            bbox,
            score,
            class_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iou_of_identical_disjoint_and_half_overlapping_boxes() {
        let a = BBox::from_xywh(0.0, 0.0, 10.0, 10.0);
        assert_eq!(a.iou(&a), 1.0);
        assert_eq!(a.iou(&BBox::from_xywh(20.0, 0.0, 10.0, 10.0)), 0.0);
        // Overlap 5×10 = 50, union 150.
        let b = BBox::from_xywh(5.0, 0.0, 10.0, 10.0);
        assert!((a.iou(&b) - 50.0 / 150.0).abs() < 1e-12);
        assert_eq!(a.iou(&b), b.iou(&a));
    }

    #[test]
    fn unrotated_undoes_image_rotation() {
        // One lit pixel at (5, 2) in a 7 × 4 image; find it in each rotated
        // image and map its pixel square back.
        let mut original = image::GrayImage::new(7, 4);
        original.put_pixel(5, 2, image::Luma([255]));
        for degrees in [0, 90, 180, 270] {
            let rotated = match degrees {
                90 => image::imageops::rotate90(&original),
                180 => image::imageops::rotate180(&original),
                270 => image::imageops::rotate270(&original),
                _ => original.clone(),
            };
            let (u, v, _) = rotated
                .enumerate_pixels()
                .find(|(_, _, p)| p[0] == 255)
                .unwrap();
            let found = BBox::from_xywh(f64::from(u), f64::from(v), 1.0, 1.0);
            assert_eq!(
                found.unrotated(degrees, 7.0, 4.0),
                BBox::from_xywh(5.0, 2.0, 1.0, 1.0),
                "{degrees} degrees"
            );
        }
    }

    #[test]
    fn degenerate_boxes_are_invalid_and_have_zero_iou() {
        let empty = BBox::new(5.0, 5.0, 5.0, 9.0);
        assert!(!empty.is_valid());
        assert!(!BBox::new(0.0, 0.0, f64::NAN, 1.0).is_valid());
        assert_eq!(empty.iou(&empty), 0.0);
        assert!(BBox::from_center_size(1.0, 2.0, 3.0, 4.0).is_valid());
    }
}
