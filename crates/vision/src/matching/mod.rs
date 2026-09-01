pub mod mutual_softmax;

pub use mutual_softmax::{MutualSoftmaxConfig, MutualSoftmaxMatcher};

#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorMatch {
    pub query_index: usize,
    pub train_index: usize,
    pub distance: f32,
    pub second_best_distance: Option<f32>,
    pub ratio: Option<f32>,
    /// Optional per-match confidence in `(0, 1]`. Populated by matchers that
    /// have a probabilistic interpretation of the match (e.g.
    /// [`mutual_softmax::MutualSoftmaxMatcher`] sets it to the dual-softmax
    /// confidence). Classical matchers leave it `None`. Downstream consumers
    /// (RANSAC sample weighting, scanner candidate ranking) treat `None` as
    /// "no signal" and fall back to uniform / unweighted behaviour.
    pub confidence: Option<f32>,
}

pub trait Matcher {
    fn match_descriptors(&self, query: &[Vec<f32>], train: &[Vec<f32>]) -> Vec<DescriptorMatch>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BruteForceMatcher {
    pub ratio: Option<f32>,
}

impl Default for BruteForceMatcher {
    fn default() -> Self {
        Self { ratio: Some(0.8) }
    }
}

impl BruteForceMatcher {
    pub fn l2_distance(lhs: &[f32], rhs: &[f32]) -> Option<f32> {
        if lhs.len() != rhs.len() {
            return None;
        }
        let squared = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<f32>();
        Some(squared.sqrt())
    }

    /// Match with a mutual-nearest-neighbour check using one descriptor
    /// matrix product for both directions.
    ///
    /// This is equivalent to wrapping this matcher in [`CrossCheckMatcher`],
    /// but avoids recomputing the transposed dot-product matrix.
    pub fn match_descriptors_cross_checked(
        &self,
        query: &[Vec<f32>],
        train: &[Vec<f32>],
    ) -> Vec<DescriptorMatch> {
        let n_query = query.len();
        let n_train = train.len();
        if n_query == 0 || n_train == 0 {
            return Vec::new();
        }
        let dim = query[0].len();
        let uniform = dim != 0
            && query.iter().all(|descriptor| descriptor.len() == dim)
            && train.iter().all(|descriptor| descriptor.len() == dim);
        if !uniform {
            let forward = self.match_descriptors_elementwise(query, train);
            let reverse = self.match_descriptors_elementwise(train, query);
            let mut reverse_best_query_by_train = vec![None; n_train];
            for descriptor_match in reverse {
                reverse_best_query_by_train[descriptor_match.query_index] =
                    Some(descriptor_match.train_index);
            }
            return forward
                .into_iter()
                .filter(|descriptor_match| {
                    reverse_best_query_by_train[descriptor_match.train_index]
                        == Some(descriptor_match.query_index)
                })
                .collect();
        }

        let q = nalgebra::DMatrix::from_fn(n_query, dim, |i, k| query[i][k]);
        let t = nalgebra::DMatrix::from_fn(n_train, dim, |j, k| train[j][k]);
        let dots = q * t.transpose();
        let query_norm_sq: Vec<f32> = query
            .iter()
            .map(|descriptor| descriptor.iter().map(|value| value * value).sum())
            .collect();
        let train_norm_sq: Vec<f32> = train
            .iter()
            .map(|descriptor| descriptor.iter().map(|value| value * value).sum())
            .collect();

        // nalgebra matrices are column-major. Walk every dot product once in
        // its contiguous order and update both directional top-2 states. The
        // query index still increases within each train column, and each
        // query observes train indices in increasing outer-loop order, so the
        // original strict-`<` first-wins tie behavior is preserved.
        let mut forward_best = vec![None; n_query];
        let mut forward_second = vec![None; n_query];
        let mut reverse_best = vec![None; n_train];
        let mut reverse_second = vec![None; n_train];
        for train_index in 0..n_train {
            for query_index in 0..n_query {
                let dot = dots[(query_index, train_index)];
                let reverse_score = query_norm_sq[query_index] - 2.0 * dot;
                if reverse_best[train_index]
                    .is_none_or(|(_, best_score)| reverse_score < best_score)
                {
                    reverse_second[train_index] =
                        reverse_best[train_index].map(|(_, best_score)| best_score);
                    reverse_best[train_index] = Some((query_index, reverse_score));
                } else if reverse_second[train_index]
                    .is_none_or(|second_score| reverse_score < second_score)
                {
                    reverse_second[train_index] = Some(reverse_score);
                }

                let forward_score = train_norm_sq[train_index] - 2.0 * dot;
                if forward_best[query_index]
                    .is_none_or(|(_, best_score)| forward_score < best_score)
                {
                    forward_second[query_index] = forward_best[query_index];
                    forward_best[query_index] = Some((train_index, forward_score));
                } else if forward_second[query_index]
                    .is_none_or(|(_, second_score)| forward_score < second_score)
                {
                    forward_second[query_index] = Some((train_index, forward_score));
                }
            }
        }

        let mut reverse_best_query_by_train = vec![None; n_train];
        for train_index in 0..n_train {
            let Some((query_index, best_score)) = reverse_best[train_index] else {
                continue;
            };
            let distance = (train_norm_sq[train_index] + best_score).max(0.0).sqrt();
            let second_distance = reverse_second[train_index]
                .map(|second_score| (train_norm_sq[train_index] + second_score).max(0.0).sqrt());
            if self
                .ratio
                .zip(second_distance)
                .is_some_and(|(ratio, second)| distance >= ratio * second)
            {
                continue;
            }
            reverse_best_query_by_train[train_index] = Some(query_index);
        }

        let mut matches = Vec::new();
        for query_index in 0..n_query {
            let Some((train_index, best_score)) = forward_best[query_index] else {
                continue;
            };
            if reverse_best_query_by_train[train_index] != Some(query_index) {
                continue;
            }
            let distance = (query_norm_sq[query_index] + best_score).max(0.0).sqrt();
            let second_distance = forward_second[query_index].map(|(_, second_score)| {
                (query_norm_sq[query_index] + second_score).max(0.0).sqrt()
            });
            if self
                .ratio
                .zip(second_distance)
                .is_some_and(|(ratio, second)| distance >= ratio * second)
            {
                continue;
            }
            matches.push(DescriptorMatch {
                query_index,
                train_index,
                distance,
                second_best_distance: second_distance,
                ratio: second_distance.map(|second| distance / second),
                confidence: None,
            });
        }
        matches
    }
}

impl Matcher for BruteForceMatcher {
    fn match_descriptors(&self, query: &[Vec<f32>], train: &[Vec<f32>]) -> Vec<DescriptorMatch> {
        // Nearest-neighbour by L2 is equivalent to nearest-neighbour by the
        // squared distance ‖q−t‖² = ‖q‖² + ‖t‖² − 2·q·t. The cross term q·t for
        // every (query, train) pair is one matrix product Q·Tᵀ, which nalgebra
        // dispatches to the blocked `matrixmultiply` kernel — orders of
        // magnitude faster than the element-wise double loop on the
        // 256-dimensional deep descriptors this matcher is fed. The selected
        // best/second distances are recovered as the actual L2 (sqrt), so the
        // returned `distance` / `ratio` fields are unchanged.
        let n_query = query.len();
        let n_train = train.len();
        if n_query == 0 || n_train == 0 {
            return Vec::new();
        }
        let dim = query[0].len();
        // The GEMM path assumes a single uniform descriptor dimension. Mixed
        // lengths (or a zero dimension) fall back to the exact element-wise
        // path, which skips mismatched pairs via `l2_distance`.
        let uniform = dim != 0
            && query.iter().all(|q| q.len() == dim)
            && train.iter().all(|t| t.len() == dim);
        if !uniform {
            return self.match_descriptors_elementwise(query, train);
        }

        let q = nalgebra::DMatrix::from_fn(n_query, dim, |i, k| query[i][k]);
        let t = nalgebra::DMatrix::from_fn(n_train, dim, |j, k| train[j][k]);
        let dots = q * t.transpose(); // (n_query × n_train), dots[(i, j)] = qᵢ·tⱼ
        let query_norm_sq: Vec<f32> = query
            .iter()
            .map(|q| q.iter().map(|x| x * x).sum())
            .collect();
        let train_norm_sq: Vec<f32> = train
            .iter()
            .map(|t| t.iter().map(|x| x * x).sum())
            .collect();

        let mut matches = Vec::new();
        for query_index in 0..n_query {
            // Rank by score_j = ‖t‖² − 2·q·t = ‖q−t‖² − ‖q‖²; ‖q‖² is constant
            // across the row, so this preserves the argmin and the strict-`<`
            // first-wins tie-break of the original loop.
            let mut best: Option<(usize, f32)> = None;
            let mut second_best: Option<(usize, f32)> = None;
            for train_index in 0..n_train {
                let score = train_norm_sq[train_index] - 2.0 * dots[(query_index, train_index)];
                if best.is_none_or(|(_, best_score)| score < best_score) {
                    second_best = best;
                    best = Some((train_index, score));
                } else if second_best.is_none_or(|(_, second_score)| score < second_score) {
                    second_best = Some((train_index, score));
                }
            }

            let Some((train_index, best_score)) = best else {
                continue;
            };
            let distance = (query_norm_sq[query_index] + best_score).max(0.0).sqrt();
            let second_distance = second_best.map(|(_, second_score)| {
                (query_norm_sq[query_index] + second_score).max(0.0).sqrt()
            });

            if let Some(ratio) = self.ratio {
                if let Some(second_distance) = second_distance {
                    if distance >= ratio * second_distance {
                        continue;
                    }
                }
            }

            matches.push(DescriptorMatch {
                query_index,
                train_index,
                distance,
                second_best_distance: second_distance,
                ratio: second_distance.map(|second_distance| distance / second_distance),
                confidence: None,
            });
        }

        matches
    }
}

impl BruteForceMatcher {
    /// Exact element-wise nearest-neighbour matching — the reference path used
    /// when descriptors have mixed dimensions (the GEMM path needs a uniform
    /// dimension). Identical in behaviour to the pre-GEMM implementation.
    fn match_descriptors_elementwise(
        &self,
        query: &[Vec<f32>],
        train: &[Vec<f32>],
    ) -> Vec<DescriptorMatch> {
        let mut matches = Vec::new();

        for (query_index, query_descriptor) in query.iter().enumerate() {
            let mut best: Option<(usize, f32)> = None;
            let mut second_best: Option<(usize, f32)> = None;

            for (train_index, train_descriptor) in train.iter().enumerate() {
                let Some(distance) = Self::l2_distance(query_descriptor, train_descriptor) else {
                    continue;
                };

                if best.is_none_or(|(_, best_distance)| distance < best_distance) {
                    second_best = best;
                    best = Some((train_index, distance));
                } else if second_best.is_none_or(|(_, second_distance)| distance < second_distance)
                {
                    second_best = Some((train_index, distance));
                }
            }

            let Some((train_index, distance)) = best else {
                continue;
            };

            if let Some(ratio) = self.ratio {
                if let Some((_, second_distance)) = second_best {
                    if distance >= ratio * second_distance {
                        continue;
                    }
                }
            }

            matches.push(DescriptorMatch {
                query_index,
                train_index,
                distance,
                second_best_distance: second_best.map(|(_, second_distance)| second_distance),
                ratio: second_best.map(|(_, second_distance)| distance / second_distance),
                confidence: None,
            });
        }

        matches
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CrossCheckMatcher<M = BruteForceMatcher> {
    pub inner: M,
}

impl<M> CrossCheckMatcher<M> {
    pub fn new(inner: M) -> Self {
        Self { inner }
    }
}

impl<M> Default for CrossCheckMatcher<M>
where
    M: Default,
{
    fn default() -> Self {
        Self {
            inner: M::default(),
        }
    }
}

impl<M> Matcher for CrossCheckMatcher<M>
where
    M: Matcher,
{
    fn match_descriptors(&self, query: &[Vec<f32>], train: &[Vec<f32>]) -> Vec<DescriptorMatch> {
        let forward = self.inner.match_descriptors(query, train);
        if forward.is_empty() {
            return forward;
        }

        let reverse = self.inner.match_descriptors(train, query);
        let mut reverse_best_query_by_train = vec![None; train.len()];
        for descriptor_match in reverse {
            if descriptor_match.query_index < reverse_best_query_by_train.len() {
                reverse_best_query_by_train[descriptor_match.query_index] =
                    Some(descriptor_match.train_index);
            }
        }

        forward
            .into_iter()
            .filter(|descriptor_match| {
                reverse_best_query_by_train
                    .get(descriptor_match.train_index)
                    .copied()
                    .flatten()
                    == Some(descriptor_match.query_index)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{BruteForceMatcher, CrossCheckMatcher, Matcher};

    #[test]
    fn single_gemm_cross_check_matches_two_pass_reference() {
        for (query_count, train_count, dim) in [(3, 5, 4), (17, 11, 32), (9, 13, 128)] {
            let descriptors = |count: usize, salt: usize| {
                (0..count)
                    .map(|row| {
                        (0..dim)
                            .map(|column| {
                                let value = (row * 131 + column * 17 + salt * 29) % 251;
                                (value as f32 - 125.0) / 127.0
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            };
            let query = descriptors(query_count, 3);
            let train = descriptors(train_count, 7);
            for ratio in [None, Some(0.8), Some(0.95)] {
                let matcher = BruteForceMatcher { ratio };
                let reference = CrossCheckMatcher::new(matcher).match_descriptors(&query, &train);
                assert_eq!(
                    matcher.match_descriptors_cross_checked(&query, &train),
                    reference
                );
            }
        }
    }

    #[test]
    fn single_gemm_cross_check_preserves_mixed_dimension_fallback() {
        let query = vec![vec![0.0, 1.0], vec![1.0]];
        let train = vec![vec![0.0, 1.0], vec![2.0]];
        let matcher = BruteForceMatcher { ratio: Some(0.8) };
        let reference = CrossCheckMatcher::new(matcher).match_descriptors(&query, &train);
        assert_eq!(
            matcher.match_descriptors_cross_checked(&query, &train),
            reference
        );
    }

    #[test]
    fn matches_nearest_descriptor_with_ratio_test() {
        let matcher = BruteForceMatcher { ratio: Some(0.9) };
        let query = vec![vec![1.0, 0.0]];
        let train = vec![vec![1.01, 0.0], vec![3.0, 0.0]];
        let matches = matcher.match_descriptors(&query, &train);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].train_index, 0);
        assert!(matches[0].second_best_distance.is_some());
        assert!(matches[0].ratio.unwrap() < 0.9);
    }

    #[test]
    fn cross_check_matcher_keeps_only_mutual_matches() {
        let matcher = CrossCheckMatcher::new(BruteForceMatcher { ratio: None });
        let query = vec![vec![0.0], vec![0.2]];
        let train = vec![vec![0.1], vec![5.0]];

        let matches = matcher.match_descriptors(&query, &train);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].query_index, 0);
        assert_eq!(matches[0].train_index, 0);
    }
}
