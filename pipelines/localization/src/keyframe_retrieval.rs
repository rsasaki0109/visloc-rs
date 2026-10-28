//! Keyframe retrieval for the map-wide search.
//!
//! [`BinaryLandmarkIndex::match_all`] compares each query keypoint with
//! every landmark of the map, and its ratio test weighs the best landmark
//! against the second best anywhere in the map. The larger the map (a
//! merged one above all), the more often an unrelated landmark comes close
//! enough to reject a correct match. [`KeyframeRetrieval`] first picks the
//! map keyframes that look most like the query and hands only the
//! landmarks they observe to [`BinaryLandmarkIndex::match_among`].
//!
//! Each keyframe is described by the binary descriptors of the landmarks
//! it observes, hashed to visual words: the descriptor bits at fixed
//! positions (Basalt's `HashBow` permutation), as the SLAM mapper's own loop
//! detection does. A keyframe's bag of words is the L1-normalized histogram
//! of its words. A query scores each keyframe by histogram intersection
//! (`sum min(q_w, k_w)` over shared words, Basalt's `HashBow` score), through
//! an inverted index of the keyframes' words.

use std::collections::{BTreeSet, HashMap};

use visloc_core::types::VisualMap;

use crate::binary_index::{BinaryLandmarkIndex, PackedDescriptors};

/// Bits of a 256-bit descriptor taken, in order, for its visual word: the
/// permutation of Basalt's `HashBow` (and of the mapper's
/// `hash_descriptor`).
const WORD_BITS: [usize; 32] = [
    170, 215, 41, 38, 96, 172, 52, 1, 182, 89, 234, 98, 217, 73, 195, 113, 161, 247, 24, 75, 7,
    232, 49, 196, 144, 69, 3, 86, 94, 201, 107, 251,
];

/// The visual word of a packed 256-bit descriptor, from its first `bits`
/// permuted bits.
fn word(descriptor: &[u64], bits: u8) -> u32 {
    let mut word = 0_u32;
    for (index, &bit) in WORD_BITS.iter().take(usize::from(bits.min(32))).enumerate() {
        if descriptor
            .get(bit / 64)
            .is_some_and(|w| w & (1_u64 << (bit % 64)) != 0)
        {
            word |= 1 << index;
        }
    }
    word
}

/// L1-normalized histogram of the words of `descriptors`.
fn bag_of_words<'a>(descriptors: impl Iterator<Item = &'a [u64]>, bits: u8) -> HashMap<u32, f32> {
    let mut counts = HashMap::<u32, f32>::new();
    let mut total = 0.0_f32;
    for descriptor in descriptors {
        *counts.entry(word(descriptor, bits)).or_default() += 1.0;
        total += 1.0;
    }
    for weight in counts.values_mut() {
        *weight /= total;
    }
    counts
}

/// A map's keyframes, described by the words of the landmarks they observe.
#[derive(Debug, Clone)]
pub struct KeyframeRetrieval {
    bits: u8,
    keyframe_ids: Vec<u64>,
    /// Per keyframe, the indices (into the [`BinaryLandmarkIndex`]) of the
    /// landmarks it observes, ascending.
    landmarks: Vec<Vec<usize>>,
    /// Word -> `(keyframe, weight)` of every keyframe that has it.
    inverted: HashMap<u32, Vec<(u32, f32)>>,
}

impl KeyframeRetrieval {
    /// Describes every keyframe of `map` that observes landmarks of `index`,
    /// with `bits`-bit words (Basalt's mapper uses 16; fewer bits make
    /// coarser words that survive more descriptor noise).
    pub fn new(map: &VisualMap, index: &BinaryLandmarkIndex, bits: u8) -> Self {
        let mut keyframes = map.keyframes.iter().collect::<Vec<_>>();
        keyframes.sort_by_key(|(id, _)| **id);
        let mut retrieval = Self {
            bits,
            keyframe_ids: Vec::new(),
            landmarks: Vec::new(),
            inverted: HashMap::new(),
        };
        for (&id, keyframe) in keyframes {
            let landmarks = keyframe
                .observations
                .iter()
                .filter_map(|observation| index.landmark_index(observation.landmark_id))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            if landmarks.is_empty() {
                continue;
            }
            let words = bag_of_words(
                landmarks
                    .iter()
                    .filter_map(|&landmark| index.landmark_descriptor(landmark)),
                bits,
            );
            let keyframe_index = retrieval.keyframe_ids.len() as u32;
            for (word, weight) in words {
                retrieval
                    .inverted
                    .entry(word)
                    .or_default()
                    .push((keyframe_index, weight));
            }
            retrieval.keyframe_ids.push(id);
            retrieval.landmarks.push(landmarks);
        }
        retrieval
    }

    /// Number of described keyframes.
    pub fn len(&self) -> usize {
        self.keyframe_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keyframe_ids.is_empty()
    }

    /// The `top_k` keyframes most like `query`, best first, as
    /// `(position, score)`; ties go to the lower keyframe id.
    fn ranked(&self, query: &PackedDescriptors, top_k: usize) -> Vec<(usize, f32)> {
        let words = bag_of_words(
            (0..query.len()).filter_map(|index| query.get(index)),
            self.bits,
        );
        let mut scores = vec![0.0_f32; self.keyframe_ids.len()];
        for (word, query_weight) in words {
            for &(keyframe, weight) in self.inverted.get(&word).into_iter().flatten() {
                scores[keyframe as usize] += query_weight.min(weight);
            }
        }
        let mut ranked = scores
            .into_iter()
            .enumerate()
            .filter(|(_, score)| *score > 0.0)
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(top_k);
        ranked
    }

    /// The ids and scores of the `top_k` keyframes most like `query`, best
    /// first.
    pub fn query(&self, query: &PackedDescriptors, top_k: usize) -> Vec<(u64, f32)> {
        self.ranked(query, top_k)
            .into_iter()
            .map(|(keyframe, score)| (self.keyframe_ids[keyframe], score))
            .collect()
    }

    /// The landmarks (indices into the [`BinaryLandmarkIndex`]) observed by
    /// the `top_k` keyframes most like `query`, ascending, for
    /// [`BinaryLandmarkIndex::match_among`].
    pub fn candidate_landmarks(&self, query: &PackedDescriptors, top_k: usize) -> Vec<usize> {
        self.ranked(query, top_k)
            .into_iter()
            .flat_map(|(keyframe, _)| self.landmarks[keyframe].iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_index::BinaryMatchGates;
    use nalgebra::{Point2, Point3};
    use visloc_core::geometry::Pose;
    use visloc_core::types::{Frame, Keyframe, Landmark, LandmarkDescriptorStore, Observation};

    /// Deterministic, well-separated pseudo-random bits per seed.
    fn descriptor(seed: u64) -> Vec<f32> {
        let mut state = (seed + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        (0..256)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                f32::from(u8::from(state & 1 == 1))
            })
            .collect()
    }

    /// Three keyframes, each observing its own 40 landmarks.
    fn map() -> (VisualMap, LandmarkDescriptorStore) {
        let mut map = VisualMap::new();
        let mut store = LandmarkDescriptorStore::new();
        for keyframe in 0..3_u64 {
            let mut observations = Vec::new();
            for point in 0..40_u64 {
                let id = keyframe * 100 + point;
                map.landmarks.insert(
                    id,
                    Landmark {
                        id,
                        position: Point3::new(keyframe as f64, point as f64, 5.0),
                        descriptor: None,
                        observations: Vec::new(),
                    },
                );
                store.insert(id, descriptor(id));
                observations.push(Observation {
                    frame_id: keyframe,
                    landmark_id: id,
                    keypoint_index: point as usize,
                    xy: Point2::new(point as f64, 0.0),
                });
            }
            map.keyframes.insert(
                keyframe,
                Keyframe {
                    frame: Frame {
                        id: keyframe,
                        camera_id: 1,
                        keypoints: Vec::new(),
                        descriptors: Vec::new(),
                        pose: Some(Pose::identity()),
                    },
                    observations,
                },
            );
        }
        (map, store)
    }

    #[test]
    fn a_query_retrieves_the_keyframe_it_shares_landmarks_with() {
        let (map, store) = map();
        let index = BinaryLandmarkIndex::new(&map, &store);
        let retrieval = KeyframeRetrieval::new(&map, &index, 8);
        assert_eq!(retrieval.len(), 3);
        // The query sees 30 of keyframe 1's landmarks.
        let query = (0..30)
            .map(|point| descriptor(100 + point))
            .collect::<Vec<_>>();
        let packed = index.pack_query(&query);
        let ranked = retrieval.query(&packed, 3);
        assert_eq!(ranked[0].0, 1, "{ranked:?}");
        assert!(ranked.get(1).is_none_or(|second| second.1 < ranked[0].1));

        let landmarks = retrieval.candidate_landmarks(&packed, 1);
        assert_eq!(landmarks.len(), 40);
        let keypoints = (0..30)
            .map(|i| Point2::new(i as f64, 0.0))
            .collect::<Vec<_>>();
        let gates = BinaryMatchGates {
            max_distance: 10,
            ratio: Some(0.8),
        };
        let set = index.match_among(&keypoints, &packed, gates, &landmarks);
        assert_eq!(set.match_count, 30);
        assert_eq!(set.candidate_landmark_count, 40);
        assert!(set.landmark_ids.iter().all(|id| (100..140).contains(id)));
        // The same as the map-wide search, which finds the same landmarks.
        let all = index.match_all(&keypoints, &packed, gates, 1);
        assert_eq!(all.landmark_ids, set.landmark_ids);
    }

    #[test]
    fn landmarks_outside_the_index_and_empty_keyframes_are_skipped() {
        let (mut map, store) = map();
        map.keyframes.insert(
            9,
            Keyframe {
                frame: Frame {
                    id: 9,
                    camera_id: 1,
                    keypoints: Vec::new(),
                    descriptors: Vec::new(),
                    pose: Some(Pose::identity()),
                },
                observations: vec![Observation {
                    frame_id: 9,
                    landmark_id: 12_345,
                    keypoint_index: 0,
                    xy: Point2::new(0.0, 0.0),
                }],
            },
        );
        let index = BinaryLandmarkIndex::new(&map, &store);
        assert_eq!(index.landmark_index(12_345), None);
        assert_eq!(index.landmark_index(100), Some(40));
        let retrieval = KeyframeRetrieval::new(&map, &index, 16);
        assert_eq!(retrieval.len(), 3);
        let empty = index.pack_query::<Vec<f32>>(&[]);
        assert!(retrieval.query(&empty, 5).is_empty());
        assert!(retrieval.candidate_landmarks(&empty, 5).is_empty());
    }

    #[test]
    fn words_take_the_permuted_bits() {
        let mut descriptor = [0_u64; 4];
        descriptor[170 / 64] |= 1 << (170 % 64); // first permuted bit
        descriptor[38 / 64] |= 1 << (38 % 64); // fourth
        assert_eq!(word(&descriptor, 16), 0b1001);
        assert_eq!(word(&descriptor, 1), 0b1);
        assert_eq!(word(&descriptor, 0), 0);
    }
}
