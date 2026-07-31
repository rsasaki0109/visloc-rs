//! Deterministic overlapping partitions for hierarchical ordered-image SfM.
//!
//! Boundaries are selected near a target window length, but may move within a
//! bounded search interval to a seam with stronger verified cross-boundary
//! support. An optional per-cut quality hint lets a frontend incorporate motion,
//! blur, or dynamic-region evidence without changing geometric acceptance gates.

use std::error::Error;
use std::fmt;
use std::ops::Range;

use crate::PairwiseMatches;

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveSubmapPartitionConfig {
    /// Smallest permitted window, including its overlap with the previous one.
    pub min_images: usize,
    /// Preferred window length before seam-quality adjustment.
    pub target_images: usize,
    /// Hard upper bound on a window's image count.
    pub max_images: usize,
    /// Images reconstructed independently in both adjacent submaps.
    pub overlap_images: usize,
    /// Maximum displacement of a boundary from `target_images`.
    pub boundary_search_radius: usize,
    /// When a submap window fails to build for a *widenable* reason — the
    /// hierarchical caller accepts
    /// [`IncrementalSfmError::NoSeedPair`](crate::IncrementalSfmError::NoSeedPair)
    /// and every build-time quality rejection — merge it with its successor
    /// (or, at the final window, its predecessor) and retry via
    /// [`widen_and_build`], instead of aborting the whole hierarchical run.
    /// The builder performs any applicable bounded same-window seed retry
    /// before this widening stage. This changes routing only: seeding and
    /// quality gates are untouched. Defaults to enabled because one weak
    /// window is a structural failure mode, not a tuning choice.
    pub widen_unseedable_windows: bool,
    /// Cap on how many neighbouring windows may be absorbed into one
    /// contiguous failing span before giving up and returning the original
    /// error (fail-fast for genuinely broken input — e.g. a sequence with no
    /// usable parallax anywhere). Chosen from the diagnosed MH_03 2700-frame
    /// failure: the strict 2.0-degree parallax / 30-match seed gates left a
    /// run of consecutive submap windows unseedable around one near-static
    /// hover span (window 12 of 164, `images 192..280`, and neighbouring
    /// windows sharing the same low-parallax signature); 16 gives that span
    /// more than 25% headroom while still bounding worst-case merged-window
    /// size (and thus reconstruction cost) for pathological all-static
    /// input, which still fails fast once the cap is spent.
    pub max_widen_merges: usize,
    /// Minimum number of images, in the overlap a build-stage-widened window
    /// shares with a still-independent predecessor, that must lie *outside*
    /// the original (pre-widen) failing window's own span.
    ///
    /// `widen_and_build` only ever grows a failing window *forward* (it
    /// absorbs its successor, never its predecessor, except when there is no
    /// successor at all — see that function's docs). That means the widened
    /// window's start is always the original failing window's own start, so
    /// the images it shares with its predecessor — `overlap_images` of them —
    /// are *entirely inside* a span this crate just proved has no seed pair
    /// anywhere inside it at its original size. A later seam alignment
    /// against that predecessor then has to rely on landmarks triangulated
    /// from that same suspect span, which is a real, diagnosed failure mode:
    /// the MH_03 2700-frame run widened `images 192..280` to `192..392`
    /// (finding a seed pair only in the added tail), and the *next* seam
    /// alignment — the widened submap against its untouched predecessor
    /// `176..264` — was rejected with `LowInlierRatio` on only 158
    /// correspondences (vs. 1294 at a healthy seam), 2h18m into the run.
    /// Note this is a distinct failure from a *short* residual overlap: the
    /// image *count* shared with the predecessor is always exactly
    /// `overlap_images`, by construction (every merge absorbs a whole
    /// neighbour window at its exact original cut point) — the defect is
    /// that those images are low-quality, not few.
    ///
    /// When, after a successful (possibly widened) build, the predecessor is
    /// still independent and fewer than this many of the shared images lie
    /// outside the original failing span, the predecessor is absorbed too —
    /// exactly like a widenable build failure would be — so the window ends
    /// up bordering whatever remains with frames a *different*,
    /// independently-successful window already vouches for. Bounded by the
    /// same `max_widen_merges` cap as ordinary widening (shared counter, same
    /// per-contiguous-span reset).
    ///
    /// Each absorption grows the merged window by roughly one more original
    /// window's width, and rebuilding a merged window means running
    /// incremental SfM over it *from scratch* (not incrementally over the
    /// previous rebuild) — measured on the diagnosed MH_03 span,
    /// successive predecessor rebuilds at 216, 232, 248, and 264 images each
    /// took on the order of 15-20 minutes, an order of magnitude worse than
    /// linear in image count. Requiring the *entire* `overlap_images`-wide
    /// zone to clear the diagnosed span (i.e. defaulting this to
    /// `overlap_images`) chains that many absorptions and was measured to
    /// make even a 600-frame fast-iteration validation run impractically
    /// slow. Defaults instead to `1`: the smallest nonzero value, so exactly
    /// *one* absorption fires whenever forward-only widening occurred (any
    /// positive threshold is satisfied once the boundary has moved at all),
    /// which already eliminates the *exact* diagnosed failing seam — the
    /// widened window no longer directly borders the untouched predecessor
    /// it was rejected against, because that predecessor is now inside it —
    /// without chaining further, unbounded-feeling rebuild cost for
    /// diminishing safety return. This is a deliberately cheap default, not
    /// a proof that one absorption is always enough; pair it with the
    /// exhaustive seam-failure reporting (`report_all_failing_seams` in
    /// `hierarchical_sfm.rs`) to see whether a still-weak seam remains
    /// afterward, and raise this (up to `overlap_images`, for the strongest
    /// guarantee: the *entire* shared overlap zone clear of the diagnosed
    /// span, matching what an ordinary never-widened boundary already has
    /// for free) if so and the extra rebuild cost is acceptable. Set to `0`
    /// to disable entirely (a widened window's predecessor-side boundary is
    /// then never treated specially).
    pub min_post_widen_overlap_images: usize,
}

impl Default for AdaptiveSubmapPartitionConfig {
    fn default() -> Self {
        Self {
            min_images: 24,
            target_images: 64,
            max_images: 96,
            overlap_images: 16,
            boundary_search_radius: 16,
            widen_unseedable_windows: true,
            max_widen_merges: 16,
            // See the field doc: kept small (not tied to `overlap_images`)
            // because each additional absorption's full incremental-SfM
            // rebuild was measured to cost far more than linearly in image
            // count.
            min_post_widen_overlap_images: 1,
        }
    }
}

/// Optional frontend evidence indexed by cut position. Entry `c` describes the
/// seam between images `c - 1` and `c`; larger finite values favor that seam.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdaptiveSubmapPartitionHints {
    pub boundary_quality_by_cut: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmapWindow {
    pub image_range: Range<usize>,
    /// Sum of verified correspondences on pair edges crossing the chosen end.
    /// The final window has no outgoing seam and therefore reports zero.
    pub outgoing_seam_support: usize,
}

impl SubmapWindow {
    pub fn image_count(&self) -> usize {
        self.image_range.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmapPartitionError {
    ZeroMinimum,
    InvalidSizeOrder,
    OverlapTooLarge,
    PairImageOutOfRange {
        pair_index: usize,
        image_index: usize,
        image_count: usize,
    },
}

impl fmt::Display for SubmapPartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMinimum => write!(f, "submap min_images must be positive"),
            Self::InvalidSizeOrder => write!(
                f,
                "submap sizes must satisfy min_images <= target_images <= max_images"
            ),
            Self::OverlapTooLarge => {
                write!(f, "submap overlap_images must be smaller than min_images")
            }
            Self::PairImageOutOfRange {
                pair_index,
                image_index,
                image_count,
            } => write!(
                f,
                "pair {pair_index} references image {image_index}; image count is {image_count}"
            ),
        }
    }
}

impl Error for SubmapPartitionError {}

/// Partition an ordered sequence into contiguous overlapping windows.
pub fn partition_ordered_submaps(
    image_count: usize,
    pairwise: &[PairwiseMatches],
    config: &AdaptiveSubmapPartitionConfig,
    hints: &AdaptiveSubmapPartitionHints,
) -> Result<Vec<SubmapWindow>, SubmapPartitionError> {
    validate_config(config)?;
    for (pair_index, pair) in pairwise.iter().enumerate() {
        for image_index in [pair.image_i, pair.image_j] {
            if image_index >= image_count {
                return Err(SubmapPartitionError::PairImageOutOfRange {
                    pair_index,
                    image_index,
                    image_count,
                });
            }
        }
    }
    if image_count == 0 {
        return Ok(Vec::new());
    }
    if image_count <= config.max_images {
        return Ok(vec![SubmapWindow {
            image_range: 0..image_count,
            outgoing_seam_support: 0,
        }]);
    }

    let seam_support = seam_support_by_cut(image_count, pairwise);
    let mut windows = Vec::new();
    let mut start = 0;
    while image_count - start > config.max_images {
        let target = (start + config.target_images).min(image_count);
        let lower =
            (target.saturating_sub(config.boundary_search_radius)).max(start + config.min_images);
        let upper = (target + config.boundary_search_radius)
            .min(start + config.max_images)
            .min(image_count);
        let end = (lower..=upper)
            .filter(|&cut| image_count - (cut - config.overlap_images) >= config.min_images)
            .max_by(|&left, &right| {
                seam_score(left, &seam_support, hints)
                    .total_cmp(&seam_score(right, &seam_support, hints))
                    .then_with(|| right.abs_diff(target).cmp(&left.abs_diff(target)))
                    .then_with(|| right.cmp(&left))
            })
            .unwrap_or_else(|| (start + config.max_images).min(image_count));
        windows.push(SubmapWindow {
            image_range: start..end,
            outgoing_seam_support: seam_support[end],
        });
        let next_start = end - config.overlap_images;
        debug_assert!(next_start > start, "validated overlap guarantees progress");
        start = next_start;
    }
    windows.push(SubmapWindow {
        image_range: start..image_count,
        outgoing_seam_support: 0,
    });
    Ok(windows)
}

/// Select and remap verified pairs into one window's local image indices.
pub fn remap_pairs_to_submap(
    pairwise: &[PairwiseMatches],
    image_range: Range<usize>,
) -> Vec<PairwiseMatches> {
    pairwise
        .iter()
        .filter(|pair| image_range.contains(&pair.image_i) && image_range.contains(&pair.image_j))
        .map(|pair| PairwiseMatches {
            image_i: pair.image_i - image_range.start,
            image_j: pair.image_j - image_range.start,
            matches: pair.matches.clone(),
        })
        .collect()
}

/// Why `widen_and_build` performed one merge; passed to `on_widen` purely for
/// observability (e.g. distinguishing log messages).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WidenMergeReason {
    /// The window itself failed `build` for a reason `is_widenable` accepted
    /// (or, for a tail window with no successor, its predecessor's merge was
    /// forced by having nowhere else to absorb).
    UnbuildableWindow,
    /// The window built successfully, but forward-only widening left its
    /// start at the original failing window's own start, so the images it
    /// shared with a still-independent predecessor were entirely inside a
    /// span already proven unseedable at its original size. See
    /// [`AdaptiveSubmapPartitionConfig::min_post_widen_overlap_images`].
    PostWidenOverlapSafety,
}

/// Build each window in `windows` via `build`, widening on a failure that
/// `is_widenable` accepts by merging the failing window with a neighbour and
/// retrying, instead of propagating the first error (the fail-fast behaviour
/// a plain `windows.iter().map(build).collect::<Result<Vec<_>, _>>()` would
/// give).
///
/// A failing window is merged with its successor — absorbed into one wider
/// window covering both original ranges — and rebuilt; if it is the final
/// window in the sequence (no successor to absorb), it is instead merged with
/// its predecessor, discarding that predecessor's already-accepted build,
/// since the predecessor's frames now belong to the wider tail window. This
/// repeats, so a run of several consecutive unseedable windows collapses into
/// one window wide enough to reach past the whole run, up to `max_merges`
/// absorptions *per contiguous failing span* (the counter resets once a
/// window succeeds and processing moves on). If a span is still failing once
/// its merge budget is spent, the triggering error is returned unchanged —
/// genuinely broken input still fails fast rather than growing without
/// bound.
///
/// Once a window (possibly after forward widening) builds successfully,
/// `min_post_widen_overlap_images` guards the *quality*, not just the
/// existence, of the boundary it exposes to a live predecessor: if forward
/// widening occurred this cycle (`merges > 0`) and fewer than
/// `min_post_widen_overlap_images` of the images shared with that
/// predecessor lie outside the *original* (pre-widen) failing window's own
/// span, the predecessor is absorbed too — spending one more merge from the
/// same `max_merges` budget — and the window is rebuilt again, repeating
/// until either the margin is satisfied, the predecessor budget runs out, or
/// there is no predecessor left. This never turns a success into a failure:
/// if the rebuilt (predecessor-absorbing) window itself fails to build, it
/// re-enters the ordinary widenable-failure handling above like any other
/// failing window. See
/// [`AdaptiveSubmapPartitionConfig::min_post_widen_overlap_images`] for why
/// this is a real, diagnosed failure mode and not a hypothetical one.
///
/// `on_widen(merge_number, absorbed_range, resulting_range, reason)` is
/// invoked once per merge, in order, purely for observability (e.g.
/// logging); pass `|_, _, _, _| {}` to ignore it.
///
/// Deterministic: windows are always visited left to right, a failing window
/// always merges forward first, and the post-widen safety absorption is a
/// pure function of the (deterministic) merged window and its unmerged
/// predecessor, so the same windows and the same sequence of build outcomes
/// always produce the same merges and the same output order.
pub fn widen_and_build<T, E>(
    mut windows: Vec<SubmapWindow>,
    max_merges: usize,
    min_post_widen_overlap_images: usize,
    mut build: impl FnMut(&SubmapWindow) -> Result<T, E>,
    is_widenable: impl Fn(&E) -> bool,
    mut on_widen: impl FnMut(usize, &Range<usize>, &Range<usize>, WidenMergeReason),
) -> Result<Vec<(SubmapWindow, T)>, E> {
    let mut outputs: Vec<(SubmapWindow, T)> = Vec::with_capacity(windows.len());
    let mut index = 0usize;
    while index < windows.len() {
        let mut merges = 0usize;
        // The start of the window this cycle began with, before any merge
        // (forward or backward) touched it. Forward merges never change a
        // window's start, so as long as it still equals this value the
        // window's predecessor-side boundary is still exactly the original
        // failing window's own edge.
        let original_start = windows[index].image_range.start;
        loop {
            match build(&windows[index]) {
                Ok(value) => {
                    let margin = original_start.saturating_sub(windows[index].image_range.start);
                    if merges > 0
                        && merges < max_merges
                        && index > 0
                        && margin < min_post_widen_overlap_images
                    {
                        // Safety absorption: this window only got here via
                        // widening, and the images it would share with its
                        // still-independent predecessor are (partly or
                        // wholly) inside the span this cycle already proved
                        // unseedable at its original size. Absorb the
                        // predecessor too, exactly like a widenable build
                        // failure would, and retry.
                        merges += 1;
                        let prev = windows.remove(index - 1);
                        outputs
                            .pop()
                            .expect("predecessor window was built before this one");
                        index -= 1;
                        let before = windows[index].image_range.clone();
                        windows[index] = SubmapWindow {
                            image_range: prev.image_range.start..windows[index].image_range.end,
                            outgoing_seam_support: windows[index].outgoing_seam_support,
                        };
                        on_widen(
                            merges,
                            &before,
                            &windows[index].image_range,
                            WidenMergeReason::PostWidenOverlapSafety,
                        );
                        continue;
                    }
                    outputs.push((windows[index].clone(), value));
                    break;
                }
                Err(error) if merges < max_merges && is_widenable(&error) => {
                    merges += 1;
                    if index + 1 < windows.len() {
                        // Absorb the successor: it has not been built yet
                        // (windows are only ever visited left to right), so
                        // nothing already accepted needs to be undone.
                        let next = windows.remove(index + 1);
                        let before = windows[index].image_range.clone();
                        windows[index] = SubmapWindow {
                            image_range: windows[index].image_range.start..next.image_range.end,
                            outgoing_seam_support: next.outgoing_seam_support,
                        };
                        on_widen(
                            merges,
                            &before,
                            &windows[index].image_range,
                            WidenMergeReason::UnbuildableWindow,
                        );
                    } else if index > 0 {
                        // Tail window with no successor: absorb the
                        // predecessor instead. Its build already succeeded
                        // and was pushed to `outputs`; undo that, since its
                        // frames now belong to the wider tail window.
                        let prev = windows.remove(index - 1);
                        outputs
                            .pop()
                            .expect("predecessor window was built before the tail window");
                        index -= 1;
                        let before = windows[index].image_range.clone();
                        windows[index] = SubmapWindow {
                            image_range: prev.image_range.start..windows[index].image_range.end,
                            outgoing_seam_support: windows[index].outgoing_seam_support,
                        };
                        on_widen(
                            merges,
                            &before,
                            &windows[index].image_range,
                            WidenMergeReason::UnbuildableWindow,
                        );
                    } else {
                        // Only one window left and it still fails: no
                        // neighbour left to absorb.
                        return Err(error);
                    }
                }
                Err(error) => return Err(error),
            }
        }
        index += 1;
    }
    Ok(outputs)
}

fn validate_config(config: &AdaptiveSubmapPartitionConfig) -> Result<(), SubmapPartitionError> {
    if config.min_images == 0 {
        return Err(SubmapPartitionError::ZeroMinimum);
    }
    if config.min_images > config.target_images || config.target_images > config.max_images {
        return Err(SubmapPartitionError::InvalidSizeOrder);
    }
    if config.overlap_images >= config.min_images {
        return Err(SubmapPartitionError::OverlapTooLarge);
    }
    Ok(())
}

fn seam_support_by_cut(image_count: usize, pairwise: &[PairwiseMatches]) -> Vec<usize> {
    let mut difference = vec![0_i64; image_count + 1];
    for pair in pairwise {
        let left = pair.image_i.min(pair.image_j);
        let right = pair.image_i.max(pair.image_j);
        let support = pair.matches.len() as i64;
        difference[left + 1] += support;
        difference[right + 1] -= support;
    }
    let mut active = 0_i64;
    difference
        .into_iter()
        .map(|delta| {
            active += delta;
            active.max(0) as usize
        })
        .collect()
}

fn seam_score(cut: usize, seam_support: &[usize], hints: &AdaptiveSubmapPartitionHints) -> f64 {
    let quality = hints
        .boundary_quality_by_cut
        .get(cut)
        .copied()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(1.0);
    seam_support[cut] as f64 * quality
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(i: usize, j: usize, count: usize) -> PairwiseMatches {
        PairwiseMatches {
            image_i: i,
            image_j: j,
            matches: (0..count).map(|k| (k, k)).collect(),
        }
    }

    fn config() -> AdaptiveSubmapPartitionConfig {
        AdaptiveSubmapPartitionConfig {
            min_images: 4,
            target_images: 6,
            max_images: 8,
            overlap_images: 2,
            boundary_search_radius: 2,
            ..AdaptiveSubmapPartitionConfig::default()
        }
    }

    #[test]
    fn partitions_cover_sequence_with_exact_overlap_and_no_short_tail() {
        let windows =
            partition_ordered_submaps(17, &[], &config(), &AdaptiveSubmapPartitionHints::default())
                .unwrap();
        assert_eq!(
            windows
                .iter()
                .map(|window| window.image_range.clone())
                .collect::<Vec<_>>(),
            vec![0..6, 4..10, 8..14, 12..17]
        );
        assert!(windows.iter().all(|window| window.image_count() >= 4));
        for adjacent in windows.windows(2) {
            assert_eq!(
                adjacent[0].image_range.end - adjacent[1].image_range.start,
                2
            );
        }
    }

    #[test]
    fn boundary_moves_to_the_best_supported_seam() {
        let pairs = vec![
            pair(3, 7, 100),
            pair(4, 7, 80),
            pair(6, 7, 100),
            pair(0, 1, 500),
        ];
        let windows = partition_ordered_submaps(
            14,
            &pairs,
            &config(),
            &AdaptiveSubmapPartitionHints::default(),
        )
        .unwrap();
        assert_eq!(windows[0].image_range, 0..7);
        assert_eq!(windows[0].outgoing_seam_support, 280);
    }

    #[test]
    fn motion_quality_hint_can_avoid_an_unsafe_seam() {
        let pairs = vec![pair(2, 6, 50), pair(3, 7, 50)];
        let mut hints = AdaptiveSubmapPartitionHints::default();
        hints.boundary_quality_by_cut = vec![1.0; 15];
        hints.boundary_quality_by_cut[6] = 0.0;
        hints.boundary_quality_by_cut[7] = 3.0;
        let windows = partition_ordered_submaps(14, &pairs, &config(), &hints).unwrap();
        assert_eq!(windows[0].image_range, 0..7);
    }

    #[test]
    fn remaps_only_internal_pairs_to_local_indices() {
        let pairs = vec![pair(1, 3, 2), pair(3, 5, 3), pair(5, 7, 4)];
        let local = remap_pairs_to_submap(&pairs, 3..7);
        assert_eq!(local, vec![pair(0, 2, 3)]);
    }

    #[test]
    fn rejects_invalid_configuration_and_pair_index() {
        let mut invalid = config();
        invalid.overlap_images = invalid.min_images;
        assert_eq!(
            partition_ordered_submaps(10, &[], &invalid, &Default::default()),
            Err(SubmapPartitionError::OverlapTooLarge)
        );
        assert!(matches!(
            partition_ordered_submaps(5, &[pair(0, 5, 1)], &config(), &Default::default()),
            Err(SubmapPartitionError::PairImageOutOfRange { .. })
        ));
    }

    mod widen_and_build_tests {
        use super::*;
        use std::cell::RefCell;

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct FakeSeedError;

        fn w(range: Range<usize>) -> SubmapWindow {
            SubmapWindow {
                image_range: range,
                outgoing_seam_support: 0,
            }
        }

        /// A window fails iff its range is entirely inside `bad` — the
        /// stand-in for "no seed pair anywhere in this window" — so widening
        /// only stops failing once a merged window extends past `bad.end`.
        fn build_failing_within<'a>(
            bad: Range<usize>,
            calls: &'a RefCell<Vec<Range<usize>>>,
        ) -> impl FnMut(&SubmapWindow) -> Result<usize, FakeSeedError> + 'a {
            move |window: &SubmapWindow| {
                calls.borrow_mut().push(window.image_range.clone());
                let r = &window.image_range;
                if r.start >= bad.start && r.end <= bad.end {
                    Err(FakeSeedError)
                } else {
                    Ok(r.len())
                }
            }
        }

        fn base_windows() -> Vec<SubmapWindow> {
            vec![w(0..10), w(10..20), w(20..30), w(30..40), w(40..50)]
        }

        #[test]
        fn merges_consecutive_unseedable_windows_into_one() {
            let calls = RefCell::new(Vec::new());
            let merges = RefCell::new(Vec::new());
            let outputs = widen_and_build(
                base_windows(),
                8,
                0, // post-widen safety absorption disabled: not under test here
                build_failing_within(10..40, &calls),
                |_error| true,
                |n, before, after, reason| {
                    assert_eq!(reason, WidenMergeReason::UnbuildableWindow);
                    merges.borrow_mut().push((n, before.clone(), after.clone()))
                },
            )
            .expect("wide enough merge budget reaches past the bad span");

            let ranges = outputs
                .iter()
                .map(|(window, _)| window.image_range.clone())
                .collect::<Vec<_>>();
            assert_eq!(ranges, vec![0..10, 10..50]);
            assert_eq!(
                outputs.iter().map(|(_, len)| *len).collect::<Vec<_>>(),
                vec![10, 40]
            );
            // Three neighbour absorptions: 10..20+20..30, then +30..40, then +40..50.
            assert_eq!(merges.borrow().len(), 3);
            assert_eq!(merges.borrow()[2].2, 10..50);
            // One build call for window 0, four for the failing/widening span
            // (10..20, 10..30, 10..40, 10..50).
            assert_eq!(calls.borrow().len(), 5);
        }

        #[test]
        fn cap_is_respected_and_original_error_returned_when_exhausted() {
            let calls = RefCell::new(Vec::new());
            let merges = RefCell::new(Vec::new());
            let error = widen_and_build(
                base_windows(),
                2, // one merge short of the three the bad span needs
                0, // post-widen safety absorption disabled: not under test here
                build_failing_within(10..40, &calls),
                |_error| true,
                |n, before, after, _reason| {
                    merges.borrow_mut().push((n, before.clone(), after.clone()))
                },
            )
            .unwrap_err();

            assert_eq!(error, FakeSeedError);
            assert_eq!(merges.borrow().len(), 2, "cap stops the third merge");
            // window 0 (1 call) + 10..20, 10..30, 10..40 (3 calls) before giving up.
            assert_eq!(calls.borrow().len(), 4);
        }

        #[test]
        fn tail_window_merges_backward_and_discards_the_predecessors_build() {
            let calls = RefCell::new(Vec::new());
            let build = |window: &SubmapWindow| {
                calls.borrow_mut().push(window.image_range.clone());
                if window.image_range == (20..30) {
                    Err(FakeSeedError)
                } else {
                    Ok(window.image_range.len())
                }
            };
            let outputs = widen_and_build(
                vec![w(0..10), w(10..20), w(20..30)],
                8,
                0, // post-widen safety absorption disabled: not under test here
                build,
                |_error| true,
                |_, _, _, _| {},
            )
            .expect("merging with the predecessor escapes the failing tail window");

            let ranges = outputs
                .iter()
                .map(|(window, _)| window.image_range.clone())
                .collect::<Vec<_>>();
            assert_eq!(ranges, vec![0..10, 10..30]);
            // window 0, window 1 (later discarded), failing 20..30, retried 10..30.
            assert_eq!(calls.borrow().len(), 4);
        }

        #[test]
        fn single_remaining_window_with_no_neighbour_fails_immediately() {
            let error = widen_and_build(
                vec![w(0..5)],
                8,
                0,
                |_window: &SubmapWindow| Err::<usize, _>(FakeSeedError),
                |_error| true,
                |_, _, _, _| {},
            )
            .unwrap_err();
            assert_eq!(error, FakeSeedError);
        }

        #[test]
        fn non_widenable_error_propagates_without_merging() {
            let calls = RefCell::new(Vec::new());
            let error = widen_and_build(
                base_windows(),
                8,
                0,
                build_failing_within(10..40, &calls),
                |_error| false, // this failure is never worth widening for
                |_, _, _, _| panic!("must not widen a non-widenable error"),
            )
            .unwrap_err();
            assert_eq!(error, FakeSeedError);
            // Window 0 succeeds, then window 1 (10..20) fails once and gives up.
            assert_eq!(calls.borrow().len(), 2);
        }

        #[test]
        fn windows_that_all_succeed_are_a_pass_through() {
            let outputs = widen_and_build(
                vec![w(0..5), w(5..10)],
                8,
                0,
                |window: &SubmapWindow| Ok::<_, FakeSeedError>(window.image_range.len()),
                |_error| true,
                |_, _, _, _| panic!("no failure means no widening"),
            )
            .unwrap();
            assert_eq!(
                outputs
                    .iter()
                    .map(|(window, len)| (window.image_range.clone(), *len))
                    .collect::<Vec<_>>(),
                vec![(0..5, 5), (5..10, 5)]
            );
        }

        #[test]
        fn merge_sequence_is_deterministic_across_runs() {
            let run = || {
                let calls = RefCell::new(Vec::new());
                let outputs = widen_and_build(
                    base_windows(),
                    8,
                    0,
                    build_failing_within(10..40, &calls),
                    |_error| true,
                    |_, _, _, _| {},
                )
                .unwrap();
                (
                    outputs
                        .into_iter()
                        .map(|(window, len)| (window.image_range, len))
                        .collect::<Vec<_>>(),
                    calls.into_inner(),
                )
            };
            assert_eq!(run(), run());
        }

        /// The exact structural shape of the diagnosed MH_03 2700-frame
        /// failure: a mid-sequence window (`bad`, `20..30`) needs one forward
        /// merge to build (`20..40`), which succeeds — but its start (`20`)
        /// is still the original failing window's own start, so all of its
        /// overlap with the untouched predecessor `10..20` is inside `bad`.
        /// With a nonzero `min_post_widen_overlap_images`, the predecessor
        /// must be absorbed too.
        fn thin_quality_windows() -> Vec<SubmapWindow> {
            vec![w(0..10), w(10..20), w(20..30), w(30..40), w(40..50)]
        }

        #[test]
        fn absorbs_predecessor_when_widened_window_still_borders_the_original_span() {
            let calls = RefCell::new(Vec::new());
            let merges = RefCell::new(Vec::new());
            // Fails only for 20..30 (needs exactly one forward merge to reach
            // 20..40, which is outside `bad`).
            let bad = 20..30;
            let build = |window: &SubmapWindow| {
                calls.borrow_mut().push(window.image_range.clone());
                let r = &window.image_range;
                if r.start >= bad.start && r.end <= bad.end {
                    Err(FakeSeedError)
                } else {
                    Ok(r.len())
                }
            };
            let outputs = widen_and_build(
                thin_quality_windows(),
                8,
                5, // require >= 5 clean images outside `bad` in the shared overlap
                build,
                |_error| true,
                |n, before, after, reason| {
                    merges
                        .borrow_mut()
                        .push((n, before.clone(), after.clone(), reason))
                },
            )
            .expect("forward widen succeeds, then the safety absorption also succeeds");

            let ranges = outputs
                .iter()
                .map(|(window, _)| window.image_range.clone())
                .collect::<Vec<_>>();
            // 10..20 (the predecessor) is gone, absorbed into the widened
            // window: final windows are 0..10, 10..40 (was 20..30, forward
            // merged to 20..40, then safety-absorbed 10..20), 40..50.
            assert_eq!(ranges, vec![0..10, 10..40, 40..50]);
            assert_eq!(merges.borrow().len(), 2);
            assert_eq!(
                merges.borrow()[0],
                (1, 20..30, 20..40, WidenMergeReason::UnbuildableWindow)
            );
            assert_eq!(
                merges.borrow()[1],
                (2, 20..40, 10..40, WidenMergeReason::PostWidenOverlapSafety)
            );
        }

        #[test]
        fn safety_absorption_cap_is_respected_and_the_partially_widened_window_is_kept() {
            let calls = RefCell::new(Vec::new());
            let merges = RefCell::new(Vec::new());
            let bad = 20..30;
            let build = |window: &SubmapWindow| {
                calls.borrow_mut().push(window.image_range.clone());
                let r = &window.image_range;
                if r.start >= bad.start && r.end <= bad.end {
                    Err(FakeSeedError)
                } else {
                    Ok(r.len())
                }
            };
            let outputs = widen_and_build(
                thin_quality_windows(),
                1, // exactly enough for the one forward merge, none left for safety
                5,
                build,
                |_error| true,
                |n, before, after, reason| {
                    merges
                        .borrow_mut()
                        .push((n, before.clone(), after.clone(), reason))
                },
            )
            .expect("the forward merge alone still succeeds; safety absorption is just skipped");

            let ranges = outputs
                .iter()
                .map(|(window, _)| window.image_range.clone())
                .collect::<Vec<_>>();
            // Cap spent on the forward merge: no budget left for the safety
            // absorption, so 10..20 survives untouched even though its
            // overlap with 20..40 is entirely inside `bad`. A capped-out
            // safety absorption must never turn this success into a failure.
            assert_eq!(ranges, vec![0..10, 10..20, 20..40, 40..50]);
            assert_eq!(merges.borrow().len(), 1);
            assert_eq!(merges.borrow()[0].3, WidenMergeReason::UnbuildableWindow);
        }

        #[test]
        fn no_safety_absorption_when_the_window_never_needed_widening() {
            let merges = RefCell::new(Vec::new());
            // Every window succeeds on the first try: `merges == 0` for all
            // of them, so the safety check must never fire even though
            // `min_post_widen_overlap_images` is large.
            let outputs = widen_and_build(
                thin_quality_windows(),
                8,
                1000,
                |window: &SubmapWindow| Ok::<_, FakeSeedError>(window.image_range.len()),
                |_error| true,
                |n, before, after, reason| {
                    merges
                        .borrow_mut()
                        .push((n, before.clone(), after.clone(), reason))
                },
            )
            .unwrap();
            assert_eq!(
                outputs
                    .iter()
                    .map(|(window, _)| window.image_range.clone())
                    .collect::<Vec<_>>(),
                vec![0..10, 10..20, 20..30, 30..40, 40..50]
            );
            assert!(merges.borrow().is_empty());
        }

        #[test]
        fn zero_threshold_disables_the_safety_absorption() {
            let calls = RefCell::new(Vec::new());
            let merges = RefCell::new(Vec::new());
            let bad = 20..30;
            let build = |window: &SubmapWindow| {
                calls.borrow_mut().push(window.image_range.clone());
                let r = &window.image_range;
                if r.start >= bad.start && r.end <= bad.end {
                    Err(FakeSeedError)
                } else {
                    Ok(r.len())
                }
            };
            let outputs = widen_and_build(
                thin_quality_windows(),
                8,
                0, // disabled
                build,
                |_error| true,
                |n, before, after, reason| {
                    merges
                        .borrow_mut()
                        .push((n, before.clone(), after.clone(), reason))
                },
            )
            .expect("forward widen alone succeeds");
            let ranges = outputs
                .iter()
                .map(|(window, _)| window.image_range.clone())
                .collect::<Vec<_>>();
            assert_eq!(ranges, vec![0..10, 10..20, 20..40, 40..50]);
            assert_eq!(merges.borrow().len(), 1);
            assert_eq!(merges.borrow()[0].3, WidenMergeReason::UnbuildableWindow);
        }

        #[test]
        fn safety_absorption_sequence_is_deterministic_across_runs() {
            let run = || {
                let calls = RefCell::new(Vec::new());
                let merges = RefCell::new(Vec::new());
                let bad = 20..30;
                let build = |window: &SubmapWindow| {
                    calls.borrow_mut().push(window.image_range.clone());
                    let r = &window.image_range;
                    if r.start >= bad.start && r.end <= bad.end {
                        Err(FakeSeedError)
                    } else {
                        Ok(r.len())
                    }
                };
                let outputs = widen_and_build(
                    thin_quality_windows(),
                    8,
                    5,
                    build,
                    |_error| true,
                    |n, before, after, reason| {
                        merges
                            .borrow_mut()
                            .push((n, before.clone(), after.clone(), reason))
                    },
                )
                .unwrap();
                (
                    outputs
                        .into_iter()
                        .map(|(window, len)| (window.image_range, len))
                        .collect::<Vec<_>>(),
                    calls.into_inner(),
                    merges.into_inner(),
                )
            };
            assert_eq!(run(), run());
        }
    }
}
