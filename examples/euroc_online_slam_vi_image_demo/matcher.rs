//! The runtime-dispatched descriptor matcher (`DemoMatcher`).

use super::*;

/// Runtime-dispatched descriptor matcher. `LocalizationPipeline<M, ...>`
/// is generic in `M` and monomorphisation fixes a single concrete type
/// for the lifetime of the pipeline; wrap `BruteForceMatcher` and
/// `CrossCheckMatcher<BruteForceMatcher>` so a single CLI flag can
/// pick between them without duplicating the pipeline construction.
#[cfg(feature = "image-io")]
#[derive(Debug, Clone)]
pub(super) enum DemoMatcher {
    BruteForce(BruteForceMatcher),
    CrossCheck(CrossCheckMatcher<BruteForceMatcher>),
    MutualSoftmax(MutualSoftmaxMatcher),
    /// Brute-force NN then drop matches above `max_distance` (optical-flow
    /// track-id descriptors: only distance≈0 is a real association).
    MaxDistance {
        inner: BruteForceMatcher,
        max_distance: f32,
    },
    /// Exact nearest-neighbour on the GPU (CUDA top-2 kernel), wrapping the
    /// same ratio policy as `BruteForce`. Falls back to the CPU path when the
    /// library is missing or a device call fails. Requires the `gpu-matcher`
    /// feature; absent otherwise.
    #[cfg(feature = "gpu-matcher")]
    Gpu(visloc_rs::vision::matching::GpuDescriptorMatcher),
}

#[cfg(feature = "image-io")]
impl Matcher for DemoMatcher {
    fn match_descriptors(&self, query: &[Vec<f32>], train: &[Vec<f32>]) -> Vec<DescriptorMatch> {
        match self {
            DemoMatcher::BruteForce(m) => m.match_descriptors(query, train),
            DemoMatcher::CrossCheck(m) => m.match_descriptors(query, train),
            DemoMatcher::MutualSoftmax(m) => m.match_descriptors(query, train),
            DemoMatcher::MaxDistance {
                inner,
                max_distance,
            } => inner
                .match_descriptors(query, train)
                .into_iter()
                .filter(|m| m.distance <= *max_distance)
                .collect(),
            #[cfg(feature = "gpu-matcher")]
            DemoMatcher::Gpu(m) => m.match_descriptors(query, train),
        }
    }
}
