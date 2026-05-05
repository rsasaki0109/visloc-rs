#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorMatch {
    pub query_index: usize,
    pub train_index: usize,
    pub distance: f32,
    pub second_best_distance: Option<f32>,
    pub ratio: Option<f32>,
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
}

impl Matcher for BruteForceMatcher {
    fn match_descriptors(&self, query: &[Vec<f32>], train: &[Vec<f32>]) -> Vec<DescriptorMatch> {
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
