//! Brute-force Hamming matching of binary descriptors.

use super::{DescriptorMatch, Matcher};

/// Brute-force Hamming matcher over binary descriptors stored as `0.0`/`1.0`
/// floats (any value `>= 0.5` is a set bit), such as ORB descriptors unpacked
/// one bit per value. For such descriptors squared L2 distance equals Hamming
/// distance, but packing them to 64-bit words and counting bits is far faster
/// than the float kernel. Queries are split across the available cores.
#[derive(Debug, Clone, PartialEq)]
pub struct HammingMatcher {
    /// Largest accepted Hamming distance.
    pub max_distance: u32,
    /// Accept only when `best < ratio * second_best`.
    pub ratio: Option<f32>,
}

fn pack_bits(descriptor: &[f32]) -> Vec<u64> {
    let mut words = vec![0_u64; descriptor.len().div_ceil(64)];
    for (bit, value) in descriptor.iter().enumerate() {
        if *value >= 0.5 {
            words[bit / 64] |= 1 << (bit % 64);
        }
    }
    words
}

impl HammingMatcher {
    fn best_match(
        &self,
        query_index: usize,
        bits: &[u64],
        train_bits: &[Option<Vec<u64>>],
    ) -> Option<DescriptorMatch> {
        let (mut best, mut second) = ((u32::MAX, 0_usize), u32::MAX);
        for (train_index, train) in train_bits.iter().enumerate() {
            let Some(train) = train else { continue };
            let distance = bits
                .iter()
                .zip(train)
                .map(|(a, b)| (a ^ b).count_ones())
                .sum::<u32>();
            if distance < best.0 {
                second = best.0;
                best = (distance, train_index);
            } else if distance < second {
                second = distance;
            }
        }
        let (distance, train_index) = best;
        if distance > self.max_distance {
            return None;
        }
        let second = (second != u32::MAX).then_some(second as f32);
        if let (Some(ratio), Some(second)) = (self.ratio, second) {
            if distance as f32 >= ratio * second {
                return None;
            }
        }
        Some(DescriptorMatch {
            query_index,
            train_index,
            distance: distance as f32,
            second_best_distance: second,
            ratio: second.map(|second| distance as f32 / second.max(1.0)),
            confidence: None,
        })
    }
}

impl Matcher for HammingMatcher {
    fn match_descriptors(&self, query: &[Vec<f32>], train: &[Vec<f32>]) -> Vec<DescriptorMatch> {
        let Some(dimension) = query.first().map(Vec::len) else {
            return Vec::new();
        };
        let train_bits = train
            .iter()
            .map(|descriptor| (descriptor.len() == dimension).then(|| pack_bits(descriptor)))
            .collect::<Vec<_>>();
        let threads = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(query.len());
        let chunk = query.len().div_ceil(threads.max(1));
        let match_chunk = |offset: usize, queries: &[Vec<f32>]| {
            queries
                .iter()
                .enumerate()
                .filter(|(_, descriptor)| descriptor.len() == dimension)
                .filter_map(|(index, descriptor)| {
                    self.best_match(offset + index, &pack_bits(descriptor), &train_bits)
                })
                .collect::<Vec<_>>()
        };
        if threads <= 1 {
            return match_chunk(0, query);
        }
        std::thread::scope(|scope| {
            let handles = query
                .chunks(chunk)
                .enumerate()
                .map(|(index, queries)| {
                    let match_chunk = &match_chunk;
                    scope.spawn(move || match_chunk(index * chunk, queries))
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("matcher thread panicked"))
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 256 deterministic, well-separated bits per seed, as 0/1 floats.
    fn descriptor(seed: usize) -> Vec<f32> {
        let mut state = (seed as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        (0..256)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 1) as f32
            })
            .collect()
    }

    #[test]
    fn applies_distance_cap_and_ratio() {
        let train = (0..4).map(descriptor).collect::<Vec<_>>();
        let mut near = descriptor(2);
        for value in near.iter_mut().take(3) {
            *value = 1.0 - *value;
        }
        let query = vec![near, descriptor(99)];
        let matcher = HammingMatcher {
            max_distance: 70,
            ratio: Some(0.8),
        };
        let matches = matcher.match_descriptors(&query, &train);
        assert_eq!(matches.len(), 1);
        assert_eq!((matches[0].query_index, matches[0].train_index), (0, 2));
        assert_eq!(matches[0].distance, 3.0);
    }

    #[test]
    fn many_queries_keep_their_indices_across_threads() {
        let train = (0..50).map(descriptor).collect::<Vec<_>>();
        let query = (0..37).rev().map(descriptor).collect::<Vec<_>>();
        let matcher = HammingMatcher {
            max_distance: 70,
            ratio: Some(0.8),
        };
        let mut matches = matcher.match_descriptors(&query, &train);
        matches.sort_by_key(|m| m.query_index);
        assert_eq!(matches.len(), 37);
        for m in &matches {
            assert_eq!(m.train_index, 36 - m.query_index);
            assert_eq!(m.distance, 0.0);
        }
    }

    #[test]
    fn mismatched_lengths_and_empty_input_match_nothing() {
        let matcher = HammingMatcher {
            max_distance: 70,
            ratio: None,
        };
        assert!(matcher.match_descriptors(&[], &[descriptor(0)]).is_empty());
        assert!(matcher
            .match_descriptors(&[vec![1.0; 8]], &[descriptor(0)])
            .is_empty());
    }
}
