//! [`Matcher`] adapter over the wgpu nearest-neighbour matcher, so pipelines
//! written against the CPU [`visloc_vision::matching::BruteForceMatcher`]
//! (e.g. map-based localization) can run their descriptor search on the GPU.

use std::sync::Arc;

use visloc_vision::matching::{DescriptorMatch, Matcher};

use crate::{FeatureBank, GpuContext, GpuMatcher};

/// GPU brute-force matcher with an optional Lowe ratio test: the same
/// result as `BruteForceMatcher { ratio }` up to f32 summation order.
/// Sets it cannot take (empty, or a descriptor dimension that is not a
/// multiple of 32) fall back to the CPU matcher. Cheap to clone.
#[derive(Clone)]
pub struct WgpuDescriptorMatcher {
    inner: Arc<(GpuContext, GpuMatcher)>,
    pub ratio: Option<f32>,
}

impl WgpuDescriptorMatcher {
    pub fn new(ctx: GpuContext, ratio: Option<f32>) -> Self {
        let matcher = GpuMatcher::new(&ctx);
        Self {
            inner: Arc::new((ctx, matcher)),
            ratio,
        }
    }
}

impl Matcher for WgpuDescriptorMatcher {
    fn match_descriptors(&self, query: &[Vec<f32>], train: &[Vec<f32>]) -> Vec<DescriptorMatch> {
        let cpu = || {
            visloc_vision::matching::BruteForceMatcher { ratio: self.ratio }
                .match_descriptors(query, train)
        };
        if query.is_empty() || train.is_empty() {
            return Vec::new();
        }
        let (ctx, matcher) = &*self.inner;
        match FeatureBank::upload(ctx, &[query, train])
            .and_then(|bank| matcher.match_pairs(ctx, &bank, &[(0, 1)], self.ratio, false))
        {
            Ok(mut matches) => matches.pop().unwrap_or_default(),
            Err(_) => cpu(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same matches as the CPU brute force on random 128-d descriptors.
    #[test]
    fn matches_cpu_brute_force() {
        let Some(ctx) = crate::try_context() else {
            return;
        };
        let mut st = 7u32;
        let mut rnd = move || {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (st >> 8) as f32 / (1u32 << 24) as f32
        };
        let mut set = |n: usize| -> Vec<Vec<f32>> {
            (0..n).map(|_| (0..128).map(|_| rnd()).collect()).collect()
        };
        let (query, train) = (set(300), set(2000));
        let gpu = WgpuDescriptorMatcher::new(ctx, Some(0.9)).match_descriptors(&query, &train);
        let cpu = visloc_vision::matching::BruteForceMatcher { ratio: Some(0.9) }
            .match_descriptors(&query, &train);
        let key = |m: &DescriptorMatch| (m.query_index, m.train_index);
        let (mut g, mut c): (Vec<_>, Vec<_>) =
            (gpu.iter().map(key).collect(), cpu.iter().map(key).collect());
        g.sort_unstable();
        c.sort_unstable();
        assert!(!c.is_empty());
        let same = g.iter().filter(|m| c.binary_search(m).is_ok()).count();
        // f32 summation order may flip a ratio test at the margin.
        assert!(
            same * 100 >= c.len() * 98,
            "{same} of {} CPU matches",
            c.len()
        );
    }
}
