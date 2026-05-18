//! LightGlue-flavored mutual softmax matcher.
//!
//! Given two sets of unit-norm descriptors, compute the cosine similarity
//! matrix `S = Q T^T` (which equals the inner product because descriptors are
//! L2-normalized), then derive a per-pair confidence
//! `c[i, j] = sqrt(softmax_row(S)[i, j] * softmax_col(S)[i, j])`. Keep the
//! mutual nearest neighbour pair when the confidence exceeds a threshold.
//!
//! This is a deterministic, training-free emulation of LightGlue's matching
//! head. It is intended to be paired with a deep-style descriptor frontend
//! such as [`super::super::features::HogLikeFeatureExtractor`], but works
//! with *any* unit-norm descriptors. Inputs that are not L2-normalized still
//! work — the dual-softmax confidence is invariant to descriptor scale per
//! row/column — but the resulting confidence threshold then loses its
//! probabilistic interpretation.

use super::{DescriptorMatch, Matcher};

/// Configuration for [`MutualSoftmaxMatcher`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MutualSoftmaxConfig {
    /// Inverse softmax temperature (higher == peakier). 20.0 is a sane
    /// starting point for unit-norm descriptors with ~128 candidates.
    pub temperature: f32,
    /// Minimum dual-softmax confidence to keep a match. Range (0, 1].
    pub min_confidence: f32,
    /// If `true`, emit ratio-style metadata (distance / second-best distance)
    /// using the row-softmax probability as a proxy ratio. Disabled by
    /// default to keep the report focused on confidence.
    pub emit_ratio_metadata: bool,
}

impl Default for MutualSoftmaxConfig {
    fn default() -> Self {
        Self {
            temperature: 20.0,
            min_confidence: 0.2,
            emit_ratio_metadata: false,
        }
    }
}

/// LightGlue-style matcher operating on unit-norm descriptors. See module docs.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MutualSoftmaxMatcher {
    pub config: MutualSoftmaxConfig,
}

impl MutualSoftmaxMatcher {
    pub fn new(config: MutualSoftmaxConfig) -> Self {
        Self { config }
    }

    fn similarity(query: &[Vec<f32>], train: &[Vec<f32>]) -> Option<Vec<f32>> {
        let descriptor_dim = query.first().map(|q| q.len()).unwrap_or(0);
        if descriptor_dim == 0 {
            return None;
        }
        for descriptor in query.iter().chain(train.iter()) {
            if descriptor.len() != descriptor_dim {
                return None;
            }
        }
        let mut similarity = vec![0.0_f32; query.len() * train.len()];
        for (qi, q) in query.iter().enumerate() {
            for (tj, t) in train.iter().enumerate() {
                let mut dot = 0.0_f32;
                for (a, b) in q.iter().zip(t.iter()) {
                    dot += a * b;
                }
                similarity[qi * train.len() + tj] = dot;
            }
        }
        Some(similarity)
    }

    fn softmax_row(similarity: &[f32], rows: usize, cols: usize, temperature: f32) -> Vec<f32> {
        let mut out = vec![0.0_f32; rows * cols];
        for r in 0..rows {
            let row = &similarity[r * cols..(r + 1) * cols];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if !max.is_finite() {
                continue;
            }
            let mut sum = 0.0_f32;
            for (c, &value) in row.iter().enumerate() {
                let exp = ((value - max) * temperature).exp();
                out[r * cols + c] = exp;
                sum += exp;
            }
            if sum > 0.0 {
                for c in 0..cols {
                    out[r * cols + c] /= sum;
                }
            }
        }
        out
    }

    fn softmax_col(similarity: &[f32], rows: usize, cols: usize, temperature: f32) -> Vec<f32> {
        let mut out = vec![0.0_f32; rows * cols];
        for c in 0..cols {
            let mut max = f32::NEG_INFINITY;
            for r in 0..rows {
                let value = similarity[r * cols + c];
                if value > max {
                    max = value;
                }
            }
            if !max.is_finite() {
                continue;
            }
            let mut sum = 0.0_f32;
            for r in 0..rows {
                let exp = ((similarity[r * cols + c] - max) * temperature).exp();
                out[r * cols + c] = exp;
                sum += exp;
            }
            if sum > 0.0 {
                for r in 0..rows {
                    out[r * cols + c] /= sum;
                }
            }
        }
        out
    }
}

impl Matcher for MutualSoftmaxMatcher {
    fn match_descriptors(&self, query: &[Vec<f32>], train: &[Vec<f32>]) -> Vec<DescriptorMatch> {
        if query.is_empty() || train.is_empty() {
            return Vec::new();
        }
        let Some(similarity) = Self::similarity(query, train) else {
            return Vec::new();
        };
        let rows = query.len();
        let cols = train.len();
        let temperature = self.config.temperature;
        let row_soft = Self::softmax_row(&similarity, rows, cols, temperature);
        let col_soft = Self::softmax_col(&similarity, rows, cols, temperature);

        let mut confidence = vec![0.0_f32; rows * cols];
        for index in 0..rows * cols {
            confidence[index] = (row_soft[index] * col_soft[index]).sqrt();
        }

        // For each query, locate its argmax confidence column (and second).
        let mut row_best = vec![(0usize, f32::NEG_INFINITY); rows];
        let mut row_second = vec![f32::NEG_INFINITY; rows];
        for r in 0..rows {
            for c in 0..cols {
                let v = confidence[r * cols + c];
                if v > row_best[r].1 {
                    row_second[r] = row_best[r].1;
                    row_best[r] = (c, v);
                } else if v > row_second[r] {
                    row_second[r] = v;
                }
            }
        }

        // For each train, locate its argmax confidence row (mutual check).
        let mut col_best = vec![(0usize, f32::NEG_INFINITY); cols];
        for c in 0..cols {
            for r in 0..rows {
                let v = confidence[r * cols + c];
                if v > col_best[c].1 {
                    col_best[c] = (r, v);
                }
            }
        }

        let mut matches = Vec::new();
        for (qi, &(best_train, best_conf)) in row_best.iter().enumerate() {
            if best_conf < self.config.min_confidence {
                continue;
            }
            let (mutual_query, _) = col_best[best_train];
            if mutual_query != qi {
                continue;
            }
            let similarity_value = similarity[qi * cols + best_train];
            let descriptor_distance = ((1.0 - similarity_value).max(0.0) * 2.0).sqrt();
            let second_value = row_second[qi];
            let ratio = if self.config.emit_ratio_metadata && second_value > 0.0 {
                Some(second_value / best_conf)
            } else {
                None
            };
            matches.push(DescriptorMatch {
                query_index: qi,
                train_index: best_train,
                distance: descriptor_distance,
                second_best_distance: if self.config.emit_ratio_metadata {
                    Some(((1.0 - second_value).max(0.0) * 2.0).sqrt())
                } else {
                    None
                },
                ratio,
                confidence: Some(best_conf),
            });
        }
        matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(values: &[f32]) -> Vec<f32> {
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        values.iter().map(|v| v / norm.max(1e-12)).collect()
    }

    #[test]
    fn matches_identical_descriptors_one_to_one() {
        let descriptors = vec![
            unit(&[1.0, 0.0, 0.0, 0.0]),
            unit(&[0.0, 1.0, 0.0, 0.0]),
            unit(&[0.0, 0.0, 1.0, 0.0]),
        ];
        let matcher = MutualSoftmaxMatcher::default();
        let matches = matcher.match_descriptors(&descriptors, &descriptors);
        assert_eq!(matches.len(), 3);
        for m in &matches {
            assert_eq!(m.query_index, m.train_index);
            assert!(m.distance < 1e-3, "self-match distance should be near zero");
            // Identical descriptors → confidence saturates near 1.0 (3x3 dual-
            // softmax with one peaked column / row each).
            let conf = m.confidence.expect("MutualSoftmax populates confidence");
            assert!(
                conf > 0.99,
                "self-match confidence should be near 1.0, got {conf}"
            );
        }
    }

    #[test]
    fn confidence_is_strictly_higher_for_a_better_match_than_an_ambiguous_one() {
        // One distinctive query-train pair plus an ambiguous query. The
        // distinctive match should land with strictly higher confidence
        // than the ambiguous one, mirroring how MutualSoftmax "knows" how
        // sure it is per pair.
        let query = vec![
            unit(&[1.0, 0.0, 0.0, 0.0]),
            unit(&[0.7, 0.7, 0.0, 0.0]), // ambiguous between train[1] and train[2]
        ];
        let train = vec![
            unit(&[1.0, 0.0, 0.0, 0.0]),
            unit(&[1.0, 1.0, 0.0, 0.0]),
            unit(&[1.0, 0.9, 0.0, 0.0]),
        ];
        let matcher = MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
            temperature: 25.0,
            min_confidence: 0.0,
            emit_ratio_metadata: false,
        });
        let matches = matcher.match_descriptors(&query, &train);
        let conf_distinct = matches
            .iter()
            .find(|m| m.query_index == 0)
            .and_then(|m| m.confidence)
            .expect("query 0 matches the distinctive train descriptor");
        let conf_ambig = matches
            .iter()
            .find(|m| m.query_index == 1)
            .and_then(|m| m.confidence)
            .expect("query 1 still produces a match below the threshold");
        assert!(
            conf_distinct > conf_ambig,
            "distinctive match should have higher confidence: distinct={conf_distinct} ambiguous={conf_ambig}"
        );
    }

    #[test]
    fn rejects_low_confidence_when_train_is_ambiguous() {
        // Two indistinguishable train descriptors split the row-softmax mass:
        // the row gives 0.5/0.5, the col gives 1.0 (one-row col), so
        // confidence = sqrt(0.5 * 1.0) ~ 0.707. A 0.8 threshold must reject.
        let query = vec![unit(&[1.0, 0.5, 0.0, 0.0])];
        let train = vec![unit(&[1.0, 0.5, 0.0, 0.0]); 2];
        let matcher = MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
            temperature: 20.0,
            min_confidence: 0.8,
            emit_ratio_metadata: false,
        });
        let matches = matcher.match_descriptors(&query, &train);
        assert!(
            matches.is_empty(),
            "ambiguous row-softmax matches should be filtered by confidence threshold"
        );
    }

    #[test]
    fn rejects_low_confidence_when_both_directions_are_ambiguous() {
        // Two queries x two trains, all four descriptors collinear -> both
        // row and column softmax give 0.5, confidence sqrt(0.5*0.5)=0.5.
        let query = vec![unit(&[1.0, 0.5, 0.0, 0.0]); 2];
        let train = vec![unit(&[1.0, 0.5, 0.0, 0.0]); 2];
        let matcher = MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
            temperature: 20.0,
            min_confidence: 0.6,
            emit_ratio_metadata: false,
        });
        let matches = matcher.match_descriptors(&query, &train);
        assert!(matches.is_empty());
    }

    #[test]
    fn enforces_mutual_nearest_neighbour() {
        // q1 prefers t0 (best), but t0's best is q0 (more similar).
        let query = vec![unit(&[1.0, 0.0]), unit(&[0.95, 0.05])];
        let train = vec![unit(&[1.0, 0.0])];
        let matcher = MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
            temperature: 30.0,
            min_confidence: 0.05,
            emit_ratio_metadata: false,
        });
        let matches = matcher.match_descriptors(&query, &train);
        // Only the (0, 0) mutual pair should survive.
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].query_index, 0);
        assert_eq!(matches[0].train_index, 0);
    }

    #[test]
    fn handles_empty_inputs() {
        let matcher = MutualSoftmaxMatcher::default();
        assert!(matcher.match_descriptors(&[], &[]).is_empty());
        assert!(matcher
            .match_descriptors(&[unit(&[1.0, 0.0])], &[])
            .is_empty());
        assert!(matcher
            .match_descriptors(&[], &[unit(&[1.0, 0.0])])
            .is_empty());
    }

    #[test]
    fn temperature_zero_falls_back_to_uniform_filtering() {
        // With temperature 0, row/col softmax become uniform => confidence ~1/sqrt(rows*cols).
        // Set a min_confidence above that to verify everything is filtered out.
        let query = vec![unit(&[1.0, 0.0]), unit(&[0.0, 1.0])];
        let train = vec![unit(&[1.0, 0.0]), unit(&[0.0, 1.0])];
        let matcher = MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
            temperature: 0.0,
            min_confidence: 0.9,
            emit_ratio_metadata: false,
        });
        let matches = matcher.match_descriptors(&query, &train);
        assert!(matches.is_empty());
    }

    #[test]
    fn shape_mismatched_descriptors_yield_no_matches() {
        let query = vec![vec![1.0, 0.0]];
        let train = vec![vec![1.0, 0.0, 0.0]];
        let matcher = MutualSoftmaxMatcher::default();
        assert!(matcher.match_descriptors(&query, &train).is_empty());
    }
}
