//! The object-detector boundary.

use std::fmt;

use crate::bbox::Detection;
use crate::image::RgbImage;

/// Finds objects in one image. The tracker only needs the resulting
/// [`Detection`]s, so any detector, learned or replayed, plugs in here.
pub trait ObjectDetector {
    type Error: std::error::Error + Send + Sync + 'static;

    /// The detections in `image`, in pixel coordinates of `image`.
    fn detect(&mut self, image: &RgbImage) -> Result<Vec<Detection>, Self::Error>;
}

/// Returns stored detections frame by frame and ignores the image: ground
/// truth, a detector's output saved earlier, or synthetic boxes in tests.
#[derive(Debug, Clone, Default)]
pub struct ReplayDetector {
    frames: Vec<Vec<Detection>>,
    next: usize,
}

impl ReplayDetector {
    pub fn new(frames: Vec<Vec<Detection>>) -> Self {
        Self { frames, next: 0 }
    }

    /// Frames not replayed yet.
    pub fn remaining(&self) -> usize {
        self.frames.len() - self.next
    }
}

/// [`ReplayDetector`] was asked for more frames than it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayExhausted {
    pub frames: usize,
}

impl fmt::Display for ReplayExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "all {} replayed frames were used", self.frames)
    }
}

impl std::error::Error for ReplayExhausted {}

impl ObjectDetector for ReplayDetector {
    type Error = ReplayExhausted;

    fn detect(&mut self, _image: &RgbImage) -> Result<Vec<Detection>, Self::Error> {
        let frame = self.frames.get(self.next).ok_or(ReplayExhausted {
            frames: self.frames.len(),
        })?;
        self.next += 1;
        Ok(frame.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbox::BBox;

    #[test]
    fn replays_frames_in_order_then_reports_exhaustion() {
        let a = Detection::new(BBox::from_xywh(0.0, 0.0, 10.0, 20.0), 0.9, 0);
        let mut detector = ReplayDetector::new(vec![vec![a], vec![]]);
        let image = RgbImage::from_gray(1, 1, &[0]).unwrap();
        assert_eq!(detector.detect(&image).unwrap(), vec![a]);
        assert_eq!(detector.remaining(), 1);
        assert!(detector.detect(&image).unwrap().is_empty());
        assert_eq!(detector.detect(&image), Err(ReplayExhausted { frames: 2 }));
    }
}
