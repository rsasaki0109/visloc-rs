//! CLEAR MOT metrics (Bernardin and Stiefelhagen, 2008) for 2D box tracks.

use std::collections::HashMap;

use crate::assignment::assign_with_cost_limit;
use crate::bbox::BBox;

/// Accumulates CLEAR MOT counts frame by frame.
///
/// In each frame a ground-truth object keeps the hypothesis it was matched to
/// in an earlier frame if they still overlap by at least the IoU threshold;
/// the rest are matched by maximum count, then minimum `1 - IoU`. A ground-truth
/// object matched to a different hypothesis than last time is an ID switch.
#[derive(Debug, Clone)]
pub struct MotAccumulator {
    iou_threshold: f64,
    last_match: HashMap<u64, u64>,
    summary: MotSummary,
}

/// Totals over the frames added so far.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MotSummary {
    pub frames: usize,
    pub ground_truth: usize,
    pub matches: usize,
    pub misses: usize,
    pub false_positives: usize,
    pub id_switches: usize,
    /// Sum of the IoU of all matches.
    pub iou_sum: f64,
}

impl MotSummary {
    /// `1 - (misses + false positives + ID switches) / ground truth`.
    pub fn mota(&self) -> f64 {
        if self.ground_truth == 0 {
            return 0.0;
        }
        1.0 - (self.misses + self.false_positives + self.id_switches) as f64
            / self.ground_truth as f64
    }

    /// Mean IoU of the matches.
    pub fn motp_iou(&self) -> f64 {
        if self.matches == 0 {
            0.0
        } else {
            self.iou_sum / self.matches as f64
        }
    }
}

impl MotAccumulator {
    /// `iou_threshold` is the least IoU for a match; 0.5 is the usual choice.
    pub fn new(iou_threshold: f64) -> Self {
        Self {
            iou_threshold,
            last_match: HashMap::new(),
            summary: MotSummary::default(),
        }
    }

    pub fn summary(&self) -> MotSummary {
        self.summary
    }

    /// Add one frame of `(ground-truth id, box)` and `(hypothesis id, box)`.
    pub fn add_frame(&mut self, ground_truth: &[(u64, BBox)], hypotheses: &[(u64, BBox)]) {
        let mut gt_to_hyp: Vec<Option<usize>> = vec![None; ground_truth.len()];
        let mut hyp_used = vec![false; hypotheses.len()];
        for (g, (gt_id, gt_box)) in ground_truth.iter().enumerate() {
            let Some(&previous) = self.last_match.get(gt_id) else {
                continue;
            };
            if let Some(h) = hypotheses.iter().position(|(id, _)| *id == previous) {
                if !hyp_used[h] && gt_box.iou(&hypotheses[h].1) >= self.iou_threshold {
                    gt_to_hyp[g] = Some(h);
                    hyp_used[h] = true;
                }
            }
        }

        let free_gt: Vec<usize> = (0..ground_truth.len())
            .filter(|&g| gt_to_hyp[g].is_none())
            .collect();
        let free_hyp: Vec<usize> = (0..hypotheses.len()).filter(|&h| !hyp_used[h]).collect();
        let mut costs = Vec::with_capacity(free_gt.len() * free_hyp.len());
        for &g in &free_gt {
            for &h in &free_hyp {
                let iou = ground_truth[g].1.iou(&hypotheses[h].1);
                costs.push(if iou >= self.iou_threshold {
                    1.0 - iou
                } else {
                    f64::INFINITY
                });
            }
        }
        // Every allowed pair costs at most 1, leaving a pair unmatched costs 2:
        // the assignment maximizes the number of matches first.
        let assignment = assign_with_cost_limit(free_gt.len(), free_hyp.len(), &costs, 2.0);
        for (g, h) in assignment.matches {
            gt_to_hyp[free_gt[g]] = Some(free_hyp[h]);
            hyp_used[free_hyp[h]] = true;
        }

        let summary = &mut self.summary;
        summary.frames += 1;
        summary.ground_truth += ground_truth.len();
        for (g, matched) in gt_to_hyp.iter().enumerate() {
            let (gt_id, gt_box) = ground_truth[g];
            let Some(h) = *matched else {
                summary.misses += 1;
                continue;
            };
            let (hyp_id, hyp_box) = hypotheses[h];
            summary.matches += 1;
            summary.iou_sum += gt_box.iou(&hyp_box);
            if let Some(previous) = self.last_match.insert(gt_id, hyp_id) {
                if previous != hyp_id {
                    summary.id_switches += 1;
                }
            }
        }
        summary.false_positives += hyp_used.iter().filter(|u| !**u).count();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(x: f64) -> BBox {
        BBox::from_xywh(x, 0.0, 10.0, 10.0)
    }

    #[test]
    fn perfect_tracking_scores_one() {
        let mut acc = MotAccumulator::new(0.5);
        for k in 0..5 {
            let x = k as f64;
            acc.add_frame(
                &[(1, b(x)), (2, b(50.0 + x))],
                &[(7, b(x)), (8, b(50.0 + x))],
            );
        }
        let s = acc.summary();
        assert_eq!(
            (s.matches, s.misses, s.false_positives, s.id_switches),
            (10, 0, 0, 0)
        );
        assert_eq!(s.mota(), 1.0);
        assert_eq!(s.motp_iou(), 1.0);
    }

    #[test]
    fn counts_misses_false_positives_and_switches() {
        let mut acc = MotAccumulator::new(0.5);
        acc.add_frame(&[(1, b(0.0))], &[(7, b(0.0))]);
        // Missed, plus a far-away false positive.
        acc.add_frame(&[(1, b(1.0))], &[(9, b(200.0))]);
        // Re-found under another ID: one switch.
        acc.add_frame(&[(1, b(2.0))], &[(8, b(2.0))]);
        let s = acc.summary();
        assert_eq!(
            (
                s.ground_truth,
                s.matches,
                s.misses,
                s.false_positives,
                s.id_switches
            ),
            (3, 2, 1, 1, 1)
        );
        assert!((s.mota() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn keeps_an_existing_match_over_a_slightly_better_one() {
        let mut acc = MotAccumulator::new(0.5);
        acc.add_frame(&[(1, b(0.0))], &[(7, b(0.0))]);
        // Hypothesis 8 fits better now, but 7 still overlaps enough: no switch.
        acc.add_frame(&[(1, b(1.0))], &[(7, b(3.0)), (8, b(1.0))]);
        let s = acc.summary();
        assert_eq!((s.id_switches, s.false_positives), (0, 1));
    }
}
