//! Descriptor matching on the GPU for `build_photo_dataset`
//! ([`CudaPairMatcher`], a [`PairMatcher`]): the CPU matcher's
//! ratio-tested mutual nearest neighbours
//! (`BruteForceMatcher::match_descriptors_cross_checked`), bit for bit.
//!
//! The descriptors must be 128 bytes over a scale of 512 (the RootSIFT bytes
//! `gsplat_photos` keeps) or 1, the case the CPU matcher computes exactly
//! too: every score is an exact integer in byte units, so the device's top-2
//! states ([`CudaGsplat::match_top2`]) are the CPU's, and the final filter
//! below repeats the CPU's f32 arithmetic on exactly the same values.

use std::fmt;
use std::sync::Mutex;

use visloc_gsplat_train::photos::PairMatcher;
use visloc_vision::matching::DescriptorMatch;

use crate::{CudaGsplat, GsplatCudaError, Top2, MATCH_DESCRIPTOR_BYTES, NO_SCORE};

/// [`PairMatcher`] on the CUDA library (see the module docs).
pub struct CudaPairMatcher {
    state: Mutex<State>,
    ratio: f32,
    /// Pairs sent to the device at a time.
    batch: usize,
}

struct State {
    gs: CudaGsplat,
    /// Per image, its descriptors' squared norms in byte units.
    norms: Vec<Vec<i32>>,
    /// 1 / scale^2: byte units to the descriptors' units.
    unit: f32,
}

impl fmt::Debug for CudaPairMatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the CUDA matcher (ratio {})", self.ratio)
    }
}

impl CudaPairMatcher {
    /// The matcher `build_photo_dataset`'s CPU path uses: ratio 0.8.
    pub fn new(gs: CudaGsplat) -> Self {
        Self {
            state: Mutex::new(State {
                gs,
                norms: Vec::new(),
                unit: 1.0,
            }),
            ratio: 0.8,
            batch: 256,
        }
    }
}

/// The scale that maps every value to an integer byte (512, then 1), with
/// every descriptor 128 long.
fn byte_scale(descriptors: &[&[Vec<f32>]]) -> Option<f32> {
    let rows = || descriptors.iter().flat_map(|d| d.iter());
    if !rows().all(|r| r.len() == MATCH_DESCRIPTOR_BYTES) {
        return None;
    }
    [512.0f32, 1.0].into_iter().find(|&scale| {
        rows().all(|r| {
            r.iter().all(|&v| {
                let s = v * scale;
                (0.0..=255.0).contains(&s) && s.fract() == 0.0
            })
        })
    })
}

/// The CPU matcher's selection from the top-2 states of one pair: train
/// rows whose best passes the ratio test name their query; a query keeps
/// its best when that train row names it back and its own ratio test
/// passes. Norms and scores in byte units, `unit` = 1 / scale^2.
pub fn cross_checked(
    ratio: Option<f32>,
    unit: f32,
    query_norms: &[i32],
    train_norms: &[i32],
    fwd: &[Top2],
    rev: &[Top2],
) -> Vec<DescriptorMatch> {
    let f = |x: i32| x as f32 * unit;
    let fails = |distance: f32, second: Option<f32>| {
        ratio.zip(second).is_some_and(|(r, s)| distance >= r * s)
    };
    let mut named = vec![None; train_norms.len()];
    for (t, r) in rev.iter().enumerate() {
        if r.best == NO_SCORE {
            continue;
        }
        let distance = (f(train_norms[t]) + f(r.best)).max(0.0).sqrt();
        let second =
            (r.second != NO_SCORE).then(|| (f(train_norms[t]) + f(r.second)).max(0.0).sqrt());
        if !fails(distance, second) {
            named[t] = Some(r.index as usize);
        }
    }
    let mut matches = Vec::new();
    for (q, s) in fwd.iter().enumerate() {
        if s.best == NO_SCORE {
            continue;
        }
        let t = s.index as usize;
        if named[t] != Some(q) {
            continue;
        }
        let distance = (f(query_norms[q]) + f(s.best)).max(0.0).sqrt();
        let second =
            (s.second != NO_SCORE).then(|| (f(query_norms[q]) + f(s.second)).max(0.0).sqrt());
        if fails(distance, second) {
            continue;
        }
        matches.push(DescriptorMatch {
            query_index: q,
            train_index: t,
            distance,
            second_best_distance: second,
            ratio: second.map(|s| distance / s),
            confidence: None,
        });
    }
    matches
}

impl PairMatcher for CudaPairMatcher {
    fn set_features(&self, descriptors: &[&[Vec<f32>]]) -> Result<(), String> {
        let scale = byte_scale(descriptors)
            .ok_or("descriptors are not 128 bytes over a scale of 512 or 1")?;
        let mut counts = Vec::with_capacity(descriptors.len());
        let mut bytes = Vec::new();
        let mut norms = Vec::with_capacity(descriptors.len());
        for d in descriptors {
            counts.push(u32::try_from(d.len()).map_err(|_| "too many descriptors")?);
            let mut n = Vec::with_capacity(d.len());
            for row in d.iter() {
                let mut sq = 0i32;
                for &v in row {
                    let b = (v * scale) as u8;
                    sq += i32::from(b) * i32::from(b);
                    bytes.push(b);
                }
                n.push(sq);
            }
            norms.push(n);
        }
        let mut st = self.state.lock().map_err(|_| "matcher poisoned")?;
        st.gs
            .set_match_descriptors(&counts, &bytes)
            .map_err(|e: GsplatCudaError| e.to_string())?;
        st.norms = norms;
        st.unit = 1.0 / (scale * scale);
        Ok(())
    }

    fn match_pairs(&self, pairs: &[(usize, usize)]) -> Result<Vec<Vec<DescriptorMatch>>, String> {
        let mut st = self.state.lock().map_err(|_| "matcher poisoned")?;
        let mut out = Vec::with_capacity(pairs.len());
        for chunk in pairs.chunks(self.batch) {
            let ids: Vec<(u32, u32)> = chunk.iter().map(|&(i, j)| (i as u32, j as u32)).collect();
            let (fwd, rev) = st.gs.match_top2(&ids).map_err(|e| e.to_string())?;
            let (mut fo, mut ro) = (0, 0);
            for &(i, j) in chunk {
                let (qn, tn) = (&st.norms[i], &st.norms[j]);
                let (fq, rt) = (&fwd[fo..fo + qn.len()], &rev[ro..ro + tn.len()]);
                fo += qn.len();
                ro += tn.len();
                out.push(if qn.is_empty() || tn.is_empty() {
                    Vec::new()
                } else {
                    cross_checked(Some(self.ratio), st.unit, qn, tn, fq, rt)
                });
            }
        }
        Ok(out)
    }
}
