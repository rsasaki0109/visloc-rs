//! Hamming matching against a map's binary landmark descriptors, packed once.
//!
//! [`CorrespondenceBuilder`](crate::CorrespondenceBuilder) and
//! [`ProjectionCorrespondenceBuilder`](crate::ProjectionCorrespondenceBuilder)
//! hand the map's descriptors to a generic [`Matcher`] as float vectors on
//! every query, so a binary matcher re-packs (and the builders re-copy) the
//! whole map per image. [`BinaryLandmarkIndex`] packs the landmark
//! descriptors to 64-bit words once and produces the same correspondences as
//! those builders with a [`HammingMatcher`], at a fraction of the cost:
//! [`BinaryLandmarkIndex::match_all`] for the appearance-only search and
//! [`BinaryLandmarkIndex::match_projected`] for a pose prior's projection
//! windows, which only compares each projected landmark with the keypoints
//! near it.
//!
//! [`Matcher`]: visloc_vision::matching::Matcher
//! [`HammingMatcher`]: visloc_vision::matching::HammingMatcher

use std::collections::HashSet;

use nalgebra::{Point2, Point3};
use visloc_core::geometry::Pose;
use visloc_core::types::{Camera, LandmarkDescriptorStore, LandmarkId, VisualMap};
use visloc_vision::matching::DescriptorMatch;
use visloc_vision::pnp::Correspondence2D3D;

use crate::CorrespondenceSet;

/// Acceptance gates of a Hamming match, as in
/// [`HammingMatcher`](visloc_vision::matching::HammingMatcher).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinaryMatchGates {
    /// Largest accepted Hamming distance.
    pub max_distance: u32,
    /// Accept only when `best < ratio * second_best`.
    pub ratio: Option<f32>,
}

/// Binary descriptors stored one bit per value (`>= 0.5` is a set bit), as
/// 64-bit words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedDescriptors {
    dimension: usize,
    words: usize,
    bits: Vec<u64>,
    /// Descriptors of another length than `dimension` never match.
    valid: Vec<bool>,
}

impl PackedDescriptors {
    /// Packs `descriptors`, which should all be `dimension` values long.
    pub fn pack<D: AsRef<[f32]>>(descriptors: &[D], dimension: usize) -> Self {
        let words = dimension.div_ceil(64);
        let mut bits = vec![0_u64; words * descriptors.len()];
        let mut valid = Vec::with_capacity(descriptors.len());
        for (index, descriptor) in descriptors.iter().enumerate() {
            let descriptor = descriptor.as_ref();
            let ok = descriptor.len() == dimension;
            valid.push(ok);
            if !ok {
                continue;
            }
            let packed = &mut bits[index * words..(index + 1) * words];
            for (bit, value) in descriptor.iter().enumerate() {
                if *value >= 0.5 {
                    packed[bit / 64] |= 1 << (bit % 64);
                }
            }
        }
        Self {
            dimension,
            words,
            bits,
            valid,
        }
    }

    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn len(&self) -> usize {
        self.valid.len()
    }

    pub fn is_empty(&self) -> bool {
        self.valid.is_empty()
    }

    fn get(&self, index: usize) -> Option<&[u64]> {
        self.valid[index].then(|| &self.bits[index * self.words..(index + 1) * self.words])
    }

    /// `(index, words)` of every valid descriptor.
    fn iter(&self) -> impl Iterator<Item = (usize, &[u64])> {
        self.bits
            .chunks_exact(self.words.max(1))
            .zip(&self.valid)
            .enumerate()
            .filter_map(|(index, (bits, valid))| valid.then_some((index, bits)))
    }
}

fn hamming(a: &[u64], b: &[u64]) -> u32 {
    // 256-bit descriptors (ORB) get a fixed-size kernel the compiler fully
    // unrolls.
    if let (Ok(a), Ok(b)) = (<&[u64; 4]>::try_from(a), <&[u64; 4]>::try_from(b)) {
        return (a[0] ^ b[0]).count_ones()
            + (a[1] ^ b[1]).count_ones()
            + (a[2] ^ b[2]).count_ones()
            + (a[3] ^ b[3]).count_ones();
    }
    a.iter().zip(b).map(|(a, b)| (a ^ b).count_ones()).sum()
}

/// Best and second-best distance of one search, with
/// [`HammingMatcher`](visloc_vision::matching::HammingMatcher)'s tie rule
/// (the first candidate at the lowest distance wins).
#[derive(Clone, Copy)]
struct Nearest {
    best: (u32, usize),
    second: u32,
}

impl Nearest {
    const fn new() -> Self {
        Self {
            best: (u32::MAX, 0),
            second: u32::MAX,
        }
    }

    fn offer(&mut self, distance: u32, index: usize) {
        if distance < self.best.0 {
            self.second = self.best.0;
            self.best = (distance, index);
        } else if distance < self.second {
            self.second = distance;
        }
    }

    /// The accepted `(distance, index, second)`, if any.
    fn accept(self, gates: BinaryMatchGates) -> Option<(u32, usize, Option<f32>)> {
        let (distance, index) = self.best;
        if distance > gates.max_distance {
            return None;
        }
        let second = (self.second != u32::MAX).then_some(self.second as f32);
        if let (Some(ratio), Some(second)) = (gates.ratio, second) {
            if distance as f32 >= ratio * second {
                return None;
            }
        }
        Some((distance, index, second))
    }
}

fn descriptor_match(
    query_index: usize,
    train_index: usize,
    distance: u32,
    second: Option<f32>,
) -> DescriptorMatch {
    DescriptorMatch {
        query_index,
        train_index,
        distance: distance as f32,
        second_best_distance: second,
        ratio: second.map(|second| distance as f32 / second.max(1.0)),
        confidence: None,
    }
}

/// A map's landmarks with their binary descriptors packed for matching.
#[derive(Debug, Clone)]
pub struct BinaryLandmarkIndex {
    landmark_ids: Vec<LandmarkId>,
    positions: Vec<Point3<f64>>,
    descriptors: PackedDescriptors,
}

impl BinaryLandmarkIndex {
    /// Indexes every landmark of `map` that has a descriptor in `store`, in
    /// landmark-id order. Descriptors whose length differs from the first
    /// one never match.
    pub fn new(map: &VisualMap, store: &LandmarkDescriptorStore) -> Self {
        let paired = store.ordered_landmark_descriptors(map);
        let dimension = paired.first().map_or(0, |(_, descriptor)| descriptor.len());
        let descriptors = paired
            .iter()
            .map(|(_, descriptor)| *descriptor)
            .collect::<Vec<_>>();
        Self {
            landmark_ids: paired.iter().map(|(landmark, _)| landmark.id).collect(),
            positions: paired
                .iter()
                .map(|(landmark, _)| landmark.position)
                .collect(),
            descriptors: PackedDescriptors::pack(&descriptors, dimension),
        }
    }

    pub fn len(&self) -> usize {
        self.landmark_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.landmark_ids.is_empty()
    }

    /// Descriptor length of the indexed landmarks.
    pub const fn dimension(&self) -> usize {
        self.descriptors.dimension
    }

    /// Packs query descriptors for [`Self::match_all`] /
    /// [`Self::match_projected`].
    pub fn pack_query<D: AsRef<[f32]>>(&self, descriptors: &[D]) -> PackedDescriptors {
        PackedDescriptors::pack(descriptors, self.dimension())
    }

    /// Appearance-only matching: every query descriptor against every
    /// landmark, split over up to `threads` threads. The same
    /// correspondences, in the same order, as
    /// [`CorrespondenceBuilder`](crate::CorrespondenceBuilder) with a
    /// [`HammingMatcher`](visloc_vision::matching::HammingMatcher) of the
    /// same gates: one per accepted keypoint, several keypoints may share a
    /// landmark.
    pub fn match_all(
        &self,
        keypoints: &[Point2<f64>],
        query: &PackedDescriptors,
        gates: BinaryMatchGates,
        threads: usize,
    ) -> CorrespondenceSet {
        let count = keypoints.len().min(query.len());
        let match_range = |range: std::ops::Range<usize>| {
            range
                .filter_map(|query_index| {
                    let bits = query.get(query_index)?;
                    let mut nearest = Nearest::new();
                    for (landmark, train) in self.descriptors.iter() {
                        nearest.offer(hamming(bits, train), landmark);
                    }
                    let (distance, landmark, second) = nearest.accept(gates)?;
                    Some(descriptor_match(query_index, landmark, distance, second))
                })
                .collect::<Vec<_>>()
        };
        let threads = threads.clamp(1, count.max(1));
        let matches = if threads == 1 {
            match_range(0..count)
        } else {
            let chunk = count.div_ceil(threads);
            std::thread::scope(|scope| {
                let handles = (0..count)
                    .step_by(chunk)
                    .map(|start| {
                        let match_range = &match_range;
                        scope.spawn(move || match_range(start..(start + chunk).min(count)))
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .flat_map(|handle| handle.join().expect("matcher thread panicked"))
                    .collect()
            })
        };
        self.correspondence_set(keypoints, matches, self.len())
    }

    /// Projection-guided matching: each landmark is projected with
    /// `pose_prior` and its descriptor compared only with the keypoints
    /// within `radius_px` of the projection (best and second best among
    /// them, gated by `gates`); overlapping windows are then resolved
    /// one-to-one, lowest distance first. The same correspondences as
    /// [`ProjectionCorrespondenceBuilder::build_with_pose_prior`](crate::ProjectionCorrespondenceBuilder::build_with_pose_prior)
    /// with a [`HammingMatcher`](visloc_vision::matching::HammingMatcher) of
    /// the same gates, except that landmarks behind the camera (`z <= 0`)
    /// are skipped before projecting.
    pub fn match_projected(
        &self,
        camera: &Camera,
        pose_prior: &Pose,
        keypoints: &[Point2<f64>],
        query: &PackedDescriptors,
        radius_px: f64,
        gates: BinaryMatchGates,
    ) -> CorrespondenceSet {
        let count = keypoints.len().min(query.len());
        let grid = KeypointGrid::new(&keypoints[..count], radius_px);
        let (width, height) = (f64::from(camera.width), f64::from(camera.height));
        let radius_squared = radius_px * radius_px;
        let mut candidate_count = 0;
        let mut proposals = Vec::new();
        for (landmark, position) in self.positions.iter().enumerate() {
            let Some(train) = self.descriptors.get(landmark) else {
                continue;
            };
            let point_camera = pose_prior.transform_world_point(position);
            if point_camera.z <= 0.0 {
                continue;
            }
            let Some(projected) = camera.project(&point_camera) else {
                continue;
            };
            if projected.x < 0.0
                || projected.y < 0.0
                || projected.x >= width
                || projected.y >= height
            {
                continue;
            }
            candidate_count += 1;
            // Window keypoints in ascending index order, as the builder's
            // linear scan visits them.
            let mut nearest = Nearest::new();
            for keypoint in grid.within(keypoints, projected, radius_squared) {
                if let Some(bits) = query.get(keypoint) {
                    nearest.offer(hamming(train, bits), keypoint);
                }
            }
            if let Some((distance, keypoint, second)) = nearest.accept(gates) {
                proposals.push(descriptor_match(keypoint, landmark, distance, second));
            }
        }
        // Landmark index order is landmark-id order, so this is the
        // builder's (distance, landmark id, keypoint) order.
        proposals.sort_by(|a, b| {
            a.distance
                .total_cmp(&b.distance)
                .then_with(|| a.train_index.cmp(&b.train_index))
                .then_with(|| a.query_index.cmp(&b.query_index))
        });
        let mut used_keypoints = HashSet::new();
        let mut used_landmarks = HashSet::new();
        proposals.retain(|m| {
            used_keypoints.insert(m.query_index) && used_landmarks.insert(m.train_index)
        });
        self.correspondence_set(keypoints, proposals, candidate_count)
    }

    fn correspondence_set(
        &self,
        keypoints: &[Point2<f64>],
        matches: Vec<DescriptorMatch>,
        candidate_landmark_count: usize,
    ) -> CorrespondenceSet {
        CorrespondenceSet {
            correspondences: matches
                .iter()
                .map(|m| Correspondence2D3D {
                    point2d: keypoints[m.query_index],
                    point3d: self.positions[m.train_index],
                    confidence: None,
                })
                .collect(),
            query_indices: matches.iter().map(|m| m.query_index).collect(),
            landmark_ids: matches
                .iter()
                .map(|m| self.landmark_ids[m.train_index])
                .collect(),
            candidate_landmark_count,
            match_count: matches.len(),
            descriptor_matches: matches,
        }
    }
}

/// Keypoints bucketed into square cells of the search radius, so a window
/// lookup visits only the 3x3 cells around the query point.
struct KeypointGrid {
    origin: Point2<f64>,
    cell: f64,
    columns: isize,
    rows: isize,
    /// Keypoint indices per cell, ascending.
    cells: Vec<Vec<usize>>,
}

impl KeypointGrid {
    fn new(keypoints: &[Point2<f64>], radius_px: f64) -> Self {
        let finite = || {
            keypoints
                .iter()
                .filter(|p| p.x.is_finite() && p.y.is_finite())
        };
        let fold = |init: f64, pick: fn(&Point2<f64>) -> f64, f: fn(f64, f64) -> f64| {
            finite().map(pick).fold(init, f)
        };
        let origin = Point2::new(
            fold(f64::INFINITY, |p| p.x, f64::min).min(0.0),
            fold(f64::INFINITY, |p| p.y, f64::min).min(0.0),
        );
        let extent = Point2::new(
            fold(0.0, |p| p.x, f64::max) - origin.x,
            fold(0.0, |p| p.y, f64::max) - origin.y,
        );
        // Any cell at least the radius wide keeps every window inside 3x3
        // cells; the cap only bounds the grid for far-flung coordinates.
        let cell = radius_px.max(1.0).max(extent.x.max(extent.y) / 1024.0);
        let columns = (extent.x / cell) as isize + 1;
        let rows = (extent.y / cell) as isize + 1;
        let mut grid = Self {
            origin,
            cell,
            columns,
            rows,
            cells: vec![Vec::new(); (columns * rows) as usize],
        };
        for (index, keypoint) in keypoints.iter().enumerate() {
            if keypoint.x.is_finite() && keypoint.y.is_finite() {
                let (column, row) = grid.cell_of(keypoint);
                grid.cells[(row * columns + column) as usize].push(index);
            }
        }
        grid
    }

    /// The (possibly out-of-grid) cell of a point.
    fn cell_of(&self, point: &Point2<f64>) -> (isize, isize) {
        (
            ((point.x - self.origin.x) / self.cell).floor() as isize,
            ((point.y - self.origin.y) / self.cell).floor() as isize,
        )
    }

    /// Indices of the keypoints within `sqrt(radius_squared)` of `center`,
    /// ascending.
    fn within(
        &self,
        keypoints: &[Point2<f64>],
        center: Point2<f64>,
        radius_squared: f64,
    ) -> Vec<usize> {
        let (column, row) = self.cell_of(&center);
        let mut found = Vec::new();
        for r in (row - 1).max(0)..=(row + 1).min(self.rows - 1) {
            for c in (column - 1).max(0)..=(column + 1).min(self.columns - 1) {
                for &index in &self.cells[(r * self.columns + c) as usize] {
                    let d = keypoints[index] - center;
                    if d.x * d.x + d.y * d.y <= radius_squared {
                        found.push(index);
                    }
                }
            }
        }
        found.sort_unstable();
        found
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{UnitQuaternion, Vector3};
    use visloc_core::types::Landmark;
    use visloc_vision::matching::HammingMatcher;

    use super::*;
    use crate::{CorrespondenceBuilder, ProjectionCorrespondenceBuilder};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn unit(&mut self) -> f64 {
            (self.next() % 1_000_000) as f64 / 1_000_000.0
        }

        fn bits(&mut self) -> Vec<f32> {
            (0..256).map(|_| (self.next() & 1) as f32).collect()
        }
    }

    fn flip(descriptor: &[f32], count: usize, rng: &mut Rng) -> Vec<f32> {
        let mut flipped = descriptor.to_vec();
        for _ in 0..count {
            let bit = (rng.next() % 256) as usize;
            flipped[bit] = 1.0 - flipped[bit];
        }
        flipped
    }

    struct Scene {
        map: VisualMap,
        store: LandmarkDescriptorStore,
        camera: Camera,
        prior: Pose,
        keypoints: Vec<Point2<f64>>,
        descriptors: Vec<Vec<f32>>,
    }

    /// Landmarks in front of and behind a camera; keypoints at the true
    /// projections of most visible ones (descriptors a few bits off, some
    /// landmarks sharing a descriptor so ratio tests bite) plus clutter. The
    /// prior is off by a few pixels.
    fn scene() -> Scene {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let camera = Camera::pinhole(1, 640, 480, 400.0, 400.0, 320.0, 240.0);
        let truth = Pose::identity();
        let prior = Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.004, -0.003, 0.002),
            Vector3::new(0.01, -0.02, 0.0),
        );
        let mut map = VisualMap::new();
        let mut store = LandmarkDescriptorStore::new();
        let mut keypoints = Vec::new();
        let mut descriptors = Vec::new();
        let mut shared = rng.bits();
        for id in 0..400_u64 {
            let position = Point3::new(
                rng.unit() * 8.0 - 4.0,
                rng.unit() * 6.0 - 3.0,
                if id % 10 == 0 {
                    -2.0
                } else {
                    1.5 + rng.unit() * 6.0
                },
            );
            if id % 37 == 0 {
                shared = rng.bits();
            }
            let descriptor = if id % 7 == 0 {
                shared.clone()
            } else {
                rng.bits()
            };
            map.landmarks.insert(id, Landmark::new(id, position));
            store.insert(id, descriptor.clone());
            if let Some(pixel) = camera.project(&truth.transform_world_point(&position)) {
                if position.z > 0.0 && id % 5 != 0 {
                    keypoints.push(pixel + nalgebra::Vector2::new(rng.unit(), rng.unit()));
                    descriptors.push(flip(&descriptor, (rng.next() % 30) as usize, &mut rng));
                }
            }
        }
        for _ in 0..150 {
            keypoints.push(Point2::new(rng.unit() * 640.0, rng.unit() * 480.0));
            descriptors.push(rng.bits());
        }
        // Interleave clutter with real keypoints.
        let mut order = (0..keypoints.len()).collect::<Vec<_>>();
        order.sort_by_key(|i| (i * 7919) % keypoints.len());
        Scene {
            map,
            store,
            camera,
            prior,
            keypoints: order.iter().map(|&i| keypoints[i]).collect(),
            descriptors: order.iter().map(|&i| descriptors[i].clone()).collect(),
        }
    }

    fn query(scene: &Scene) -> visloc_core::types::QueryImage {
        visloc_core::types::QueryImage {
            camera: scene.camera.clone(),
            keypoints: scene.keypoints.clone(),
            descriptors: scene.descriptors.clone(),
        }
    }

    const GATES: BinaryMatchGates = BinaryMatchGates {
        max_distance: 70,
        ratio: Some(0.8),
    };

    fn hamming_matcher() -> HammingMatcher {
        HammingMatcher {
            max_distance: GATES.max_distance,
            ratio: GATES.ratio,
        }
    }

    #[test]
    fn global_matching_equals_the_correspondence_builder() {
        let scene = scene();
        let index = BinaryLandmarkIndex::new(&scene.map, &scene.store);
        assert_eq!(index.len(), 400);
        let expected = CorrespondenceBuilder::new(hamming_matcher())
            .build(&query(&scene), &scene.map, &scene.store)
            .unwrap();
        assert!(expected.match_count > 100, "{}", expected.match_count);
        let packed = index.pack_query(&scene.descriptors);
        for threads in [1, 3] {
            let actual = index.match_all(&scene.keypoints, &packed, GATES, threads);
            assert_eq!(actual, expected, "threads={threads}");
        }
    }

    #[test]
    fn projection_matching_equals_the_projection_builder() {
        let scene = scene();
        let index = BinaryLandmarkIndex::new(&scene.map, &scene.store);
        let packed = index.pack_query(&scene.descriptors);
        for radius in [4.0, 12.0, 40.0] {
            let expected = ProjectionCorrespondenceBuilder::new(hamming_matcher())
                .build_with_pose_prior(
                    &query(&scene),
                    &scene.map,
                    &scene.store,
                    &scene.prior,
                    radius,
                )
                .unwrap();
            let actual = index.match_projected(
                &scene.camera,
                &scene.prior,
                &scene.keypoints,
                &packed,
                radius,
                GATES,
            );
            assert!(
                actual.match_count > 50,
                "radius {radius}: {}",
                actual.match_count
            );
            // The builder counts every map landmark as a candidate; the
            // index counts those projecting into the image.
            assert_eq!(expected.candidate_landmark_count, 400);
            assert!(actual.candidate_landmark_count < 400);
            assert_eq!(
                CorrespondenceSet {
                    candidate_landmark_count: 0,
                    ..actual
                },
                CorrespondenceSet {
                    candidate_landmark_count: 0,
                    ..expected
                },
                "radius {radius}"
            );
        }
    }

    #[test]
    fn mismatched_lengths_never_match() {
        let mut scene = scene();
        scene.descriptors[0] = vec![1.0; 8];
        let index = BinaryLandmarkIndex::new(&scene.map, &scene.store);
        let packed = index.pack_query(&scene.descriptors);
        let all = index.match_all(&scene.keypoints, &packed, GATES, 1);
        assert!(all.query_indices.iter().all(|&q| q != 0));
        let empty = index.match_all(&[], &index.pack_query::<Vec<f32>>(&[]), GATES, 4);
        assert_eq!(empty.match_count, 0);
    }
}
