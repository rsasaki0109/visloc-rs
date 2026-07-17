//! Persistent correspondence graph — COLMAP's view-graph object.
//!
//! Port of `src/colmap/scene/correspondence_graph.{h,cc}` (BSD-3-Clause, ETH
//! Zurich / UNC Chapel Hill): the graph of image-to-image and feature-to-
//! feature correspondences a two-view-verified image collection induces.
//! COLMAP's own incremental mapper, pair generators (`TransitivePairGenerator`),
//! and track/point registration all query this structure rather than
//! re-deriving connectivity ad hoc — `docs/colmap_port_plan.md`'s §2 component
//! table calls the pre-M2 state of this repo exactly that: "Ad hoc union-find
//! over `PairwiseMatches` inside `incremental_sfm()`, rebuilt per call ...
//! **PARTIAL → gap.**" This module is the M2 fix for that gap.
//!
//! Ported surface, each cited to its COLMAP source line (`main` branch,
//! fetched 2026-07-17):
//! - [`Correspondence`] — `CorrespondenceGraph::Correspondence`
//!   (`correspondence_graph.h:47-58`): a `(image_id, point2D_idx)` pair.
//! - [`CorrespondenceGraph::add_image`] — `AddImage`
//!   (`.h:103`, `.cc:94-98`): declare an image's point-2D capacity before any
//!   geometry referencing it can be added.
//! - [`CorrespondenceGraph::add_two_view_geometry`] — `AddTwoViewGeometry`
//!   (`.h:109-111`, `.cc:100-201`): ingest inlier matches bidirectionally,
//!   dropping self-matches, out-of-bounds point indices, and duplicate
//!   correspondences. COLMAP logs each dropped case as `LOG(WARNING)`; this
//!   port has no logging dependency in this crate, so it returns an
//!   [`IngestStats`] tally instead — same information, inspectable in tests.
//! - [`CorrespondenceGraph::find_correspondences`] /
//!   [`CorrespondenceGraph::extract_correspondences`] — `FindCorrespondences`
//!   / `ExtractCorrespondences` (`.cc:203-228`): direct (one-hop)
//!   correspondences of a single `(image, point2D)` observation.
//! - [`CorrespondenceGraph::extract_transitive_correspondences`] —
//!   `ExtractTransitiveCorrespondences` (`.cc:230-291`): the BFS-style
//!   multi-hop closure, bounded by a `transitivity` level count, that is the
//!   actual track-building primitive (a track is the closure at "however many
//!   levels it takes to stop growing" — see [`CorrespondenceGraph::track_of`]).
//! - [`CorrespondenceGraph::num_observations_for_image`],
//!   [`CorrespondenceGraph::num_correspondences_for_image`],
//!   [`CorrespondenceGraph::num_matches_between_images`],
//!   [`CorrespondenceGraph::num_matches_between_all_images`] — the
//!   connectivity accessors (`.h:82-94`) COLMAP's initial-pair heuristic and
//!   next-best-view ranking read.
//! - [`CorrespondenceGraph::is_two_view_observation`] — `IsTwoViewObservation`
//!   (`.h:157`, `.cc:354-363`): true iff an observation's *only*
//!   correspondence is itself a two-view-only observation (a strict two-image
//!   track, as opposed to a multi-view one).
//! - [`CorrespondenceGraph::finalize`] — `Finalize` (`.h:68-74`, `.cc:57-92`).
//!   **Documented discrepancy**, found by reading the current `main`-branch
//!   `.cc` directly rather than trusting the header comment: the header
//!   claims `Finalize()` "Deletes images without observations, as they are
//!   useless for SfM," but the `.cc` body only flattens each image's
//!   `corrs` into `flat_corrs`/`flat_corr_begs` — it never actually removes
//!   an entry from `images_`. `docs/colmap_port_plan.md`'s M2 scope
//!   explicitly asks for "drop images without correspondences, compute
//!   per-image counts" as part of `Finalize()` semantics, so this port
//!   implements the **documented** contract (drop zero-observation images),
//!   not the apparently-stale-doc `.cc` behaviour. Practical effect: after
//!   `finalize()`, [`CorrespondenceGraph::exists_image`] can go from `true`
//!   to `false` for an image that was added but never received a
//!   correspondence, and every accessor that looks an image up by id follows
//!   COLMAP's own `.at()`-throws-if-missing convention (this port panics with
//!   a descriptive message, since Rust has no direct `std::out_of_range`
//!   analogue worth threading through every call site of an internal engine
//!   structure).
//!
//! Not ported: `ExtractMatchesBetweenImages`/`ExtractTwoViewGeometry`/
//! `UpdateTwoViewGeometry`'s COLMAP-database-specific `Invert()`/
//! `ShouldSwapImagePair()` direction bookkeeping (`.cc:196-201, 322-352`) is
//! **not** faithfully reproduced: this repo's current downstream consumer
//! (`pipelines/slam/src/incremental_sfm.rs`'s seed-pair placement) always
//! *recomputes* its own relative pose from raw correspondences via
//! `RelativePoseEstimator` rather than reusing a verifier's stored E/F/H, so
//! no caller needs direction-consistent matrix inversion on a swapped-order
//! query yet. This graph stores each pair's edge metadata
//! ([`ConfigurationType`] + match count) exactly as inserted and documents
//! (see [`EdgeMetadata`]) that a query in the opposite direction from
//! insertion returns the same metadata unchanged, not inverted — a scope
//! reduction in the same spirit as M1's own documented substitutions.
//! `operator<<`/`Print` stream helpers (`.cc:365-379`) are skipped (no
//! `Display` need identified in this repo). COLMAP's `image_pair_t` bit-packed
//! Cantor-pairing pair-id encoding (`ImagePairToPairId`) is replaced by a
//! plain canonical-order `(usize, usize)` tuple key — this repo's images are
//! dense `0..n` indices, not COLMAP's arbitrary database row ids, so no
//! bit-packing is needed for a fast/collision-free key.
//!
//! **Degenerate-pair policy** (`docs/colmap_port_plan.md`'s M2 scope item 3).
//! Mirroring COLMAP exactly: this graph type does **not** gate ingestion by
//! [`ConfigurationType`] internally — `AddTwoViewGeometry` in
//! `correspondence_graph.cc` has no `config`-based branch at all. The decision
//! of *which* two-view geometries are worth adding is made by the **caller**,
//! in `colmap/scene/database_cache.cc`'s `UseInlierMatchesCheck`
//! (`database_cache.cc:40-46`, fetched 2026-07-17):
//! ```text
//! bool UseInlierMatchesCheck(options, config, num_matches) {
//!   return num_matches >= options.min_num_matches &&
//!          (!options.ignore_watermarks || config != TwoViewGeometry::WATERMARK);
//! }
//! ```
//! called once per verified pair before `AddTwoViewGeometry` is invoked at
//! all (`database_cache.cc:284-300`). So in real COLMAP: `WATERMARK` pairs are
//! dropped only if `ignore_watermarks` is set (COLMAP's own default); every
//! *other* configuration, **including `PLANAR`/`PANORAMIC`/`PLANAR_OR_PANORAMIC`**,
//! is added as long as it clears `min_num_matches`. `DEGENERATE`-classified
//! geometries contribute nothing not because of an explicit `config ==
//! DEGENERATE` check anywhere, but because
//! [`crate::two_view::colmap_verification::TwoViewGeometryVerifier`] (this
//! repo's M1 port) always returns an **empty** inlier list for `DEGENERATE`
//! (`colmap_verification.rs`'s `degenerate_report()`) — there is nothing to
//! add, the same way COLMAP's own degenerate branch never populates
//! `inlier_matches`. This repo's current wiring
//! (`examples/unordered_sfm_demo.rs`'s `verify_pairs`) is **stricter than
//! COLMAP** here: its keep-list is `Calibrated | Uncalibrated | Planar |
//! Multiple`, dropping `Panoramic` (and unresolved `PlanarOrPanoramic`)
//! entirely before a [`PairwiseMatches`](crate::two_view::TwoViewCorrespondence)
//! list — let alone this graph — ever sees them. Loosening that keep-list to
//! match COLMAP (`Panoramic` pairs *do* carry real, useful correspondences for
//! transitive track-building even though their own two-view geometry has no
//! triangulatable baseline) is a real, identified lever, deliberately **not**
//! pulled in this milestone: it is a change to *which pairs reach
//! `PairwiseMatches` at all*, independent of *which algorithm builds tracks
//! from them* (this milestone's actual scope), and pulling it now would
//! confound the M2 accuracy A/B (legacy union-find vs. this graph) with an
//! unrelated M1.1-adjacent change. Flagged as a follow-up.

use std::collections::{HashMap, HashSet};

use super::colmap_verification::ConfigurationType;

/// One `(image, point2D)` correspondence target. Port of
/// `CorrespondenceGraph::Correspondence` (`correspondence_graph.h:47-58`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Correspondence {
    pub image_id: usize,
    pub point2d_idx: usize,
}

impl Correspondence {
    pub fn new(image_id: usize, point2d_idx: usize) -> Self {
        Self {
            image_id,
            point2d_idx,
        }
    }
}

/// Per-pair edge metadata: match count + the M1 [`ConfigurationType`]. Port of
/// `CorrespondenceGraph::ImagePair` (`correspondence_graph.h:180-185`), minus
/// the full `TwoViewGeometry` payload (COLMAP stores it "without matches";
/// this port stores `config` only — see the module doc's "Not ported" note
/// on why direction-dependent E/F/H/pose data isn't carried here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeMetadata {
    /// Number of correspondences actually ingested for this pair (after
    /// dropping self-matches/out-of-bounds/duplicates) — COLMAP's
    /// `ImagePair::num_matches`.
    pub num_matches: usize,
    /// The [`ConfigurationType`] the caller classified this pair as. Stored
    /// exactly as inserted; see the module doc for why a swapped-direction
    /// query does not invert anything (there is nothing direction-dependent
    /// left to invert once only `config` + a match count are kept).
    pub config: ConfigurationType,
}

/// Outcome of one [`CorrespondenceGraph::add_two_view_geometry`] call: how
/// many of the input matches were actually ingested vs. dropped, and why.
/// Stands in for COLMAP's `LOG(WARNING)` calls in `AddTwoViewGeometry`
/// (`correspondence_graph.cc:151-190`) — this crate has no logging
/// dependency, so the same information is returned as data instead, callable
/// and assertable from tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IngestStats {
    /// Correspondences added bidirectionally (`.cc:163-173`).
    pub added: usize,
    /// Matches referencing a `point2D_idx` outside the declared capacity from
    /// [`CorrespondenceGraph::add_image`] (`.cc:134-135, 174-189`).
    pub out_of_bounds: usize,
    /// Matches that duplicate a correspondence already present, checked from
    /// one side only since correspondences are always added bidirectionally
    /// (`.cc:141-162`, comment: "checking from only one side is sufficient").
    pub duplicate: usize,
}

impl IngestStats {
    /// Total matches presented to `add_two_view_geometry`, i.e.
    /// `added + out_of_bounds + duplicate`.
    pub fn total(&self) -> usize {
        self.added + self.out_of_bounds + self.duplicate
    }
}

/// Why [`CorrespondenceGraph::add_two_view_geometry`] could not be applied at
/// all (as opposed to [`IngestStats`], which reports per-match outcomes for a
/// call that *did* proceed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrespondenceGraphError {
    /// `image_id1 == image_id2`. COLMAP logs a warning and returns
    /// (`correspondence_graph.cc:104-108`) rather than treating this as fatal;
    /// this port surfaces it as a distinguishable error instead so a caller
    /// can decide whether that's a bug in its own pair generation.
    SelfMatch(usize),
    /// Either image was never registered via [`CorrespondenceGraph::add_image`].
    /// COLMAP's `images_.at(image_id)` throws `std::out_of_range` here
    /// (`.cc:111-112`).
    UnknownImage(usize),
    /// This exact `(image_id1, image_id2)` pair (in either order) was already
    /// added. COLMAP `THROW_CHECK(inserted)`s
    /// ("Two view geometry for image pair was already added",
    /// `.cc:121-124`) — geometry for a pair is meant to be added exactly
    /// once; use [`CorrespondenceGraph::update_edge_config`] to revise it in
    /// place afterwards (mirrors `UpdateTwoViewGeometry`, `.h:147-149`).
    DuplicatePair(usize, usize),
    /// [`CorrespondenceGraph::finalize`] has already run; the flattened
    /// representation no longer supports incremental inserts (mirrors
    /// COLMAP's `corrs`-cleared-after-`Finalize` invariant, `.cc:90`).
    AlreadyFinalized,
}

impl std::fmt::Display for CorrespondenceGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SelfMatch(id) => write!(f, "cannot add a self-match for image {id}"),
            Self::UnknownImage(id) => write!(f, "image {id} was never added to the graph"),
            Self::DuplicatePair(a, b) => {
                write!(f, "two-view geometry for pair ({a}, {b}) was already added")
            }
            Self::AlreadyFinalized => write!(f, "graph is already finalized"),
        }
    }
}

impl std::error::Error for CorrespondenceGraphError {}

/// Canonical (order-independent) key for an unordered image pair.
fn pair_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// Per-image bookkeeping. Port of `CorrespondenceGraph::Image`
/// (`correspondence_graph.h:160-178`).
#[derive(Debug, Clone, PartialEq)]
struct ImageEntry {
    num_observations: usize,
    num_correspondences: usize,
    /// Pre-finalize: correspondences per point2D index, grown lazily to
    /// `num_points2d` entries.
    corrs: Vec<Vec<Correspondence>>,
    /// Post-finalize flattened form (`flat_corrs`/`flat_corr_begs` in COLMAP).
    flat_corrs: Vec<Correspondence>,
    flat_corr_begs: Vec<usize>,
}

impl ImageEntry {
    fn new(num_points2d: usize) -> Self {
        Self {
            num_observations: 0,
            num_correspondences: 0,
            corrs: vec![Vec::new(); num_points2d],
            flat_corrs: Vec::new(),
            flat_corr_begs: Vec::new(),
        }
    }

    fn num_points2d(&self) -> usize {
        if self.flat_corr_begs.is_empty() {
            self.corrs.len()
        } else {
            self.flat_corr_begs.len() - 1
        }
    }
}

/// The persistent correspondence graph. Port of `CorrespondenceGraph`
/// (`src/colmap/scene/correspondence_graph.h/.cc`) — see the module doc for
/// the full ported-surface table and citations.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CorrespondenceGraph {
    finalized: bool,
    images: HashMap<usize, ImageEntry>,
    image_pairs: HashMap<(usize, usize), EdgeMetadata>,
}

impl CorrespondenceGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Port of `AddImage` (`.h:103`, `.cc:94-98`). Declares an image's
    /// point-2D capacity so later [`Self::add_two_view_geometry`] calls can
    /// validate indices against it. Panics if `image_id` was already added
    /// (COLMAP: `THROW_CHECK(!ExistsImage(image_id))`).
    pub fn add_image(&mut self, image_id: usize, num_points2d: usize) {
        assert!(
            !self.images.contains_key(&image_id),
            "image {image_id} already added to the correspondence graph"
        );
        self.images.insert(image_id, ImageEntry::new(num_points2d));
    }

    /// Port of `NumImages` (`.h:77`).
    pub fn num_images(&self) -> usize {
        self.images.len()
    }

    /// Port of `NumImagePairs` (`.h:80`).
    pub fn num_image_pairs(&self) -> usize {
        self.image_pairs.len()
    }

    /// Port of `ExistsImage` (`.h:97`, `.cc:212-214`).
    pub fn exists_image(&self, image_id: usize) -> bool {
        self.images.contains_key(&image_id)
    }

    fn image(&self, image_id: usize) -> &ImageEntry {
        self.images
            .get(&image_id)
            .unwrap_or_else(|| panic!("image {image_id} does not exist"))
    }

    /// Port of `NumObservationsForImage` (`.h:84`, `.cc:216-224`): the number
    /// of this image's point2Ds that have at least one correspondence.
    pub fn num_observations_for_image(&self, image_id: usize) -> usize {
        self.image(image_id).num_observations
    }

    /// Port of `NumCorrespondencesForImage` (`.h:87`, `.cc:226-234`): total
    /// correspondences (not deduplicated by point2D) to other images.
    pub fn num_correspondences_for_image(&self, image_id: usize) -> usize {
        self.image(image_id).num_correspondences
    }

    /// Port of `NumMatchesBetweenImages` (`.h:90-91`, `.cc:236-245`). Returns
    /// `0` for a pair that was never added (COLMAP: same — `find` miss).
    pub fn num_matches_between_images(&self, image_id1: usize, image_id2: usize) -> usize {
        self.image_pairs
            .get(&pair_key(image_id1, image_id2))
            .map_or(0, |e| e.num_matches)
    }

    /// Port of `NumMatchesBetweenAllImages` (`.h:94`, `.cc:38-46`).
    pub fn num_matches_between_all_images(&self) -> HashMap<(usize, usize), usize> {
        self.image_pairs
            .iter()
            .map(|(&pair, edge)| (pair, edge.num_matches))
            .collect()
    }

    /// Port of `ImagePairs` (`.h:100`, `.cc:48-55`).
    pub fn image_pairs(&self) -> Vec<(usize, usize)> {
        self.image_pairs.keys().copied().collect()
    }

    /// The stored [`EdgeMetadata`] for a pair, if it was added. Lighter-weight
    /// analogue of `ExtractTwoViewGeometry` (`.h:142-143`) — this port has no
    /// direction-dependent payload to invert (see module doc), so unlike
    /// COLMAP's version this is a plain lookup, order-insensitive.
    pub fn edge(&self, image_id1: usize, image_id2: usize) -> Option<EdgeMetadata> {
        self.image_pairs.get(&pair_key(image_id1, image_id2)).copied()
    }

    /// Port of `UpdateTwoViewGeometry` (`.h:147-149`, `.cc:340-352`), reduced
    /// to updating just the stored [`ConfigurationType`] (see module doc for
    /// why no direction-dependent geometry is stored to update here).
    /// Returns `false` if the pair was never added.
    pub fn update_edge_config(
        &mut self,
        image_id1: usize,
        image_id2: usize,
        config: ConfigurationType,
    ) -> bool {
        match self.image_pairs.get_mut(&pair_key(image_id1, image_id2)) {
            Some(edge) => {
                edge.config = config;
                true
            }
            None => false,
        }
    }

    /// Port of `AddTwoViewGeometry` (`.h:109-111`, `.cc:100-201`): ingest
    /// verified matches between `image_id1` and `image_id2` bidirectionally.
    /// `matches` are `(point2d_idx_in_image1, point2d_idx_in_image2)` pairs,
    /// already restricted to the winning model's inliers by the caller (this
    /// function does not know or care about [`ConfigurationType`] beyond
    /// storing it as edge metadata — see the module doc's "Degenerate-pair
    /// policy" section for why that gating lives at the call site, exactly as
    /// in COLMAP).
    ///
    /// Returns [`IngestStats`] tallying how many matches were added vs.
    /// dropped as self-matches/out-of-bounds/duplicates (COLMAP logs each
    /// dropped case; this port reports the same information as data).
    pub fn add_two_view_geometry(
        &mut self,
        image_id1: usize,
        image_id2: usize,
        matches: &[(usize, usize)],
        config: ConfigurationType,
    ) -> Result<IngestStats, CorrespondenceGraphError> {
        if self.finalized {
            return Err(CorrespondenceGraphError::AlreadyFinalized);
        }
        // `.cc:104-108`: self-matches are rejected outright.
        if image_id1 == image_id2 {
            return Err(CorrespondenceGraphError::SelfMatch(image_id1));
        }
        if !self.images.contains_key(&image_id1) {
            return Err(CorrespondenceGraphError::UnknownImage(image_id1));
        }
        if !self.images.contains_key(&image_id2) {
            return Err(CorrespondenceGraphError::UnknownImage(image_id2));
        }
        let key = pair_key(image_id1, image_id2);
        // `.cc:121-124`: THROW_CHECK(inserted) — a pair may only be added once.
        if self.image_pairs.contains_key(&key) {
            return Err(CorrespondenceGraphError::DuplicatePair(image_id1, image_id2));
        }

        let mut stats = IngestStats::default();
        for &(k1, k2) in matches {
            let valid1 = k1 < self.images[&image_id1].num_points2d();
            let valid2 = k2 < self.images[&image_id2].num_points2d();
            // `.cc:134-136, 174-189`: out-of-bounds indices are dropped.
            if !valid1 || !valid2 {
                stats.out_of_bounds += 1;
                continue;
            }
            // `.cc:143-149`: duplicate check from image1's side only — valid
            // because correspondences are always added bidirectionally below.
            let duplicate = self.images[&image_id1].corrs[k1]
                .iter()
                .any(|c| c.image_id == image_id2 && c.point2d_idx == k2);
            if duplicate {
                stats.duplicate += 1;
                continue;
            }
            // `.cc:163-173`: add bidirectionally; the first correspondence at
            // a point2D promotes it to an "observation".
            {
                let img1 = self.images.get_mut(&image_id1).unwrap();
                img1.corrs[k1].push(Correspondence::new(image_id2, k2));
                if img1.corrs[k1].len() == 1 {
                    img1.num_observations += 1;
                }
            }
            {
                let img2 = self.images.get_mut(&image_id2).unwrap();
                img2.corrs[k2].push(Correspondence::new(image_id1, k1));
                if img2.corrs[k2].len() == 1 {
                    img2.num_observations += 1;
                }
            }
            stats.added += 1;
        }

        // `.cc:114-116`: num_correspondences increases by the number of
        // correspondences actually recorded for this pair (COLMAP increments
        // by the raw match count up front, then decrements per dropped match;
        // net effect is the same as incrementing by `stats.added` directly).
        self.images.get_mut(&image_id1).unwrap().num_correspondences += stats.added;
        self.images.get_mut(&image_id2).unwrap().num_correspondences += stats.added;

        self.image_pairs.insert(
            key,
            EdgeMetadata {
                num_matches: stats.added,
                config,
            },
        );

        Ok(stats)
    }

    /// Port of `HasCorrespondences` (`.h:152`, `.cc:247-251`).
    pub fn has_correspondences(&self, image_id: usize, point2d_idx: usize) -> bool {
        !self.find_correspondences(image_id, point2d_idx).is_empty()
    }

    /// Port of `FindCorrespondences` (`.h:114-115`, `.cc:203-216`): the
    /// direct (one-hop) correspondences of one `(image, point2D)`
    /// observation. COLMAP returns a `[begin, end)` pointer range into either
    /// the pre-finalize `corrs` or the post-finalize `flat_corrs`; this port
    /// returns a plain slice, which serves the same purpose without exposing
    /// raw pointers.
    pub fn find_correspondences(&self, image_id: usize, point2d_idx: usize) -> &[Correspondence] {
        let image = self.image(image_id);
        if self.finalized {
            let beg = image.flat_corr_begs[point2d_idx];
            let end = image.flat_corr_begs[point2d_idx + 1];
            &image.flat_corrs[beg..end]
        } else {
            &image.corrs[point2d_idx]
        }
    }

    /// Port of `ExtractCorrespondences` (`.h:117-120`, `.cc:218-228`): the
    /// owned-`Vec` variant of [`Self::find_correspondences`].
    pub fn extract_correspondences(&self, image_id: usize, point2d_idx: usize) -> Vec<Correspondence> {
        self.find_correspondences(image_id, point2d_idx).to_vec()
    }

    /// Port of `IsTwoViewObservation` (`.h:157`, `.cc:354-363`): true iff
    /// `(image_id, point2d_idx)` has exactly one correspondence, and that
    /// correspondence's own correspondence set is *also* exactly that one
    /// observation back — i.e. a strict, mutually-exclusive two-image track.
    pub fn is_two_view_observation(&self, image_id: usize, point2d_idx: usize) -> bool {
        let corrs = self.find_correspondences(image_id, point2d_idx);
        if corrs.len() != 1 {
            return false;
        }
        let other = corrs[0];
        self.find_correspondences(other.image_id, other.point2d_idx).len() == 1
    }

    /// Port of `ExtractTransitiveCorrespondences` (`.h:130-134`,
    /// `.cc:230-291`): breadth-first, level-by-level closure over
    /// correspondences reachable from `(image_id, point2d_idx)`, stopping
    /// after `transitivity` levels or when a level adds nothing new. The
    /// returned list never contains the seed observation itself and has no
    /// duplicates (matches COLMAP's own contract, `.h:128-129`).
    ///
    /// `transitivity == 1` is a direct alias for
    /// [`Self::extract_correspondences`] (`.cc:235-238`). Passing
    /// `usize::MAX` reproduces the *unbounded* closure — the full connected
    /// component — which is what a legacy union-find track builder computes;
    /// see `pipelines/slam/src/incremental_sfm.rs`'s `build_tracks_via_graph`
    /// for exactly that use.
    pub fn extract_transitive_correspondences(
        &self,
        image_id: usize,
        point2d_idx: usize,
        transitivity: usize,
    ) -> Vec<Correspondence> {
        if transitivity == 1 {
            return self.extract_correspondences(image_id, point2d_idx);
        }
        if !self.exists_image(image_id) || !self.has_correspondences(image_id, point2d_idx) {
            return Vec::new();
        }

        // `.cc:246-253`: seed the queue with the requested observation itself
        // (removed again at the end), and a visited-set for dedup.
        let mut corrs = vec![Correspondence::new(image_id, point2d_idx)];
        let mut visited: HashSet<(usize, usize)> = HashSet::new();
        visited.insert((image_id, point2d_idx));

        let mut queue_beg = 0usize;
        let mut queue_end = 1usize;

        for _level in 0..transitivity {
            for i in queue_beg..queue_end {
                let reference = corrs[i];
                for &next in self.find_correspondences(reference.image_id, reference.point2d_idx) {
                    if visited.insert((next.image_id, next.point2d_idx)) {
                        corrs.push(next);
                    }
                }
            }
            queue_beg = queue_end;
            queue_end = corrs.len();
            // `.cc:279-282`: no growth this level — the closure is complete.
            if queue_beg == queue_end {
                break;
            }
        }

        // `.cc:285-290`: drop the seed observation (swap-remove, order of the
        // remaining elements is otherwise the discovery order, same as COLMAP).
        if corrs.len() > 1 {
            let last = corrs.len() - 1;
            corrs.swap(0, last);
        }
        corrs.pop();
        corrs
    }

    /// Port of `Finalize` (`.h:68-74`, `.cc:57-92`): flattens each image's
    /// per-point2D correspondence lists into the compact `flat_corrs`/
    /// `flat_corr_begs` form, and — per this port's documented-contract choice
    /// (see module doc) — drops every image with zero observations. Panics if
    /// already finalized (COLMAP: `THROW_CHECK(!finalized_)`).
    pub fn finalize(&mut self) {
        assert!(!self.finalized, "correspondence graph is already finalized");
        self.finalized = true;

        self.images.retain(|_, image| image.num_observations > 0);

        for image in self.images.values_mut() {
            let num_points2d = image.corrs.len();
            let mut flat_corrs = Vec::with_capacity(image.num_correspondences);
            let mut flat_corr_begs = Vec::with_capacity(num_points2d + 1);
            for point2d_idx in 0..num_points2d {
                flat_corr_begs.push(flat_corrs.len());
                flat_corrs.extend(image.corrs[point2d_idx].iter().copied());
            }
            flat_corr_begs.push(flat_corrs.len());
            image.flat_corrs = flat_corrs;
            image.flat_corr_begs = flat_corr_begs;
            image.corrs = Vec::new();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Port of `TEST(CorrespondenceGraph, Empty)` (`correspondence_graph_test.cc:57-63`).
    #[test]
    fn empty_graph_reports_zero_everything() {
        let graph = CorrespondenceGraph::new();
        assert_eq!(graph.num_images(), 0);
        assert_eq!(graph.num_image_pairs(), 0);
        assert_eq!(graph.num_matches_between_all_images().len(), 0);
        assert_eq!(graph.image_pairs().len(), 0);
    }

    /// Port of `TEST_P(CorrespondenceGraphFinalizeTest, TwoView)`
    /// (`correspondence_graph_test.cc:78-218`), covering both the
    /// not-yet-finalized and finalized code paths (COLMAP parametrizes the
    /// same test body over both; this port just calls it twice).
    fn two_view_case(finalize: bool) {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(0, 10);
        graph.add_image(1, 10);
        assert!(graph.exists_image(0));
        assert!(graph.exists_image(1));
        assert!(!graph.exists_image(2));
        assert_eq!(graph.num_images(), 2);

        let matches = vec![(0, 0), (1, 2), (3, 7), (4, 8)];
        let stats = graph
            .add_two_view_geometry(0, 1, &matches, ConfigurationType::Calibrated)
            .expect("valid pair must be added");
        assert_eq!(stats.added, 4);
        assert_eq!(stats.out_of_bounds, 0);
        assert_eq!(stats.duplicate, 0);

        if finalize {
            graph.finalize();
        }

        assert_eq!(graph.num_correspondences_for_image(0), 4);
        assert_eq!(graph.num_correspondences_for_image(1), 4);
        assert_eq!(graph.num_matches_between_all_images().len(), 1);
        assert_eq!(graph.num_matches_between_images(0, 1), 4);
        assert_eq!(graph.image_pairs(), vec![(0, 1)]);
        assert_eq!(
            graph.edge(0, 1).unwrap().config,
            ConfigurationType::Calibrated
        );

        assert_eq!(graph.extract_correspondences(0, 0), vec![Correspondence::new(1, 0)]);
        assert!(graph.has_correspondences(0, 0));
        assert!(graph.is_two_view_observation(0, 0));

        assert_eq!(graph.extract_correspondences(1, 0), vec![Correspondence::new(0, 0)]);
        assert!(graph.is_two_view_observation(1, 0));

        assert_eq!(graph.extract_correspondences(0, 1), vec![Correspondence::new(1, 2)]);
        assert_eq!(graph.extract_correspondences(1, 2), vec![Correspondence::new(0, 1)]);
        assert_eq!(graph.extract_correspondences(0, 4), vec![Correspondence::new(1, 8)]);
        assert_eq!(graph.extract_correspondences(0, 3), vec![Correspondence::new(1, 7)]);
        assert_eq!(graph.extract_correspondences(1, 7), vec![Correspondence::new(0, 3)]);
        assert_eq!(graph.extract_correspondences(1, 8), vec![Correspondence::new(0, 4)]);

        // Transitivity 0 finds nothing; transitivity 2 (there is no third
        // image to chain through) matches the direct one-hop set exactly.
        for point2d_idx in 0..10 {
            assert_eq!(
                graph.extract_transitive_correspondences(0, point2d_idx, 0).len(),
                0
            );
            assert_eq!(
                graph.extract_correspondences(0, point2d_idx).len(),
                graph.extract_transitive_correspondences(0, point2d_idx, 2).len()
            );
            assert_eq!(
                graph.extract_transitive_correspondences(1, point2d_idx, 0).len(),
                0
            );
            assert_eq!(
                graph.extract_correspondences(1, point2d_idx).len(),
                graph.extract_transitive_correspondences(1, point2d_idx, 2).len()
            );
        }

        assert_eq!(graph.num_observations_for_image(0), 4);
        assert_eq!(graph.num_observations_for_image(1), 4);
    }

    #[test]
    fn two_view_not_finalized() {
        two_view_case(false);
    }

    #[test]
    fn two_view_finalized() {
        two_view_case(true);
    }

    /// Port of `TEST_P(CorrespondenceGraphFinalizeTest, ThreeView)`
    /// (`correspondence_graph_test.cc:220-298`): a genuine transitive chain
    /// (image 0 <-> 1 <-> 2 sharing point 0) that exercises multi-hop closure.
    fn three_view_case(finalize: bool) {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(0, 10);
        graph.add_image(1, 10);
        graph.add_image(2, 10);
        graph
            .add_two_view_geometry(0, 1, &[(0, 0)], ConfigurationType::Calibrated)
            .unwrap();
        graph
            .add_two_view_geometry(0, 2, &[(0, 0)], ConfigurationType::Calibrated)
            .unwrap();
        graph
            .add_two_view_geometry(1, 2, &[(0, 0), (5, 5)], ConfigurationType::Calibrated)
            .unwrap();
        if finalize {
            graph.finalize();
        }

        assert_eq!(graph.num_observations_for_image(0), 1);
        assert_eq!(graph.num_observations_for_image(1), 2);
        assert_eq!(graph.num_observations_for_image(2), 2);
        assert_eq!(graph.num_correspondences_for_image(0), 2);
        assert_eq!(graph.num_correspondences_for_image(1), 3);
        assert_eq!(graph.num_correspondences_for_image(2), 3);

        assert_eq!(graph.num_matches_between_all_images().len(), 3);
        assert_eq!(graph.num_matches_between_images(0, 1), 1);
        assert_eq!(graph.num_matches_between_images(0, 2), 1);
        assert_eq!(graph.num_matches_between_images(1, 2), 2);

        let corrs0 = graph.extract_correspondences(0, 0);
        assert_eq!(corrs0, vec![Correspondence::new(1, 0), Correspondence::new(2, 0)]);
        let corrs1 = graph.extract_correspondences(1, 0);
        assert_eq!(corrs1, vec![Correspondence::new(0, 0), Correspondence::new(2, 0)]);
        let corrs2 = graph.extract_correspondences(2, 0);
        assert_eq!(corrs2, vec![Correspondence::new(0, 0), Correspondence::new(1, 0)]);

        assert_eq!(graph.extract_correspondences(1, 5), vec![Correspondence::new(2, 5)]);
        assert_eq!(graph.extract_correspondences(2, 5), vec![Correspondence::new(1, 5)]);

        // Transitive closure of point 0 across all three images: 2 other
        // observations reachable regardless of which image you start from,
        // and regardless of whether transitivity is 2 or 3 (closure
        // completes after 2 levels here — a third level adds nothing new).
        for image_id in [0usize, 1, 2] {
            assert_eq!(
                graph.extract_transitive_correspondences(image_id, 0, 2).len(),
                2
            );
            assert_eq!(
                graph.extract_transitive_correspondences(image_id, 0, 3).len(),
                2
            );
        }
    }

    #[test]
    fn three_view_not_finalized() {
        three_view_case(false);
    }

    #[test]
    fn three_view_finalized() {
        three_view_case(true);
    }

    /// Port of `TEST(CorrespondenceGraph, OutOfBounds)`
    /// (`correspondence_graph_test.cc:307-322`).
    #[test]
    fn out_of_bounds_matches_are_dropped() {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(0, 10);
        graph.add_image(1, 4);
        let matches = vec![(9, 3), (10, 3), (9, 4)];
        let stats = graph
            .add_two_view_geometry(0, 1, &matches, ConfigurationType::Calibrated)
            .unwrap();
        // (9,3): both indices within capacity (image 0 has 10, image 1 has
        // 4) -> valid. (10,3): image 0's capacity is 10 (indices 0..9), so
        // point2d_idx1=10 is out of bounds. (9,4): image 1's capacity is 4
        // (indices 0..3), so point2d_idx2=4 is out of bounds.
        assert_eq!(stats.added, 1);
        assert_eq!(stats.out_of_bounds, 2);
        assert_eq!(graph.num_correspondences_for_image(0), 1);
        assert_eq!(graph.num_correspondences_for_image(1), 1);
        assert_eq!(graph.num_matches_between_images(0, 1), 1);
    }

    /// Port of `TEST(CorrespondenceGraph, Duplicate)`
    /// (`correspondence_graph_test.cc:324-341`).
    #[test]
    fn duplicate_matches_are_dropped() {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(0, 10);
        graph.add_image(1, 10);
        let matches = vec![(0, 0), (1, 1), (1, 1), (3, 3), (3, 4)];
        let stats = graph
            .add_two_view_geometry(0, 1, &matches, ConfigurationType::Calibrated)
            .unwrap();
        assert_eq!(stats.added, 4);
        assert_eq!(stats.duplicate, 1);
        assert_eq!(graph.num_correspondences_for_image(0), 4);
        assert_eq!(graph.num_correspondences_for_image(1), 4);
        assert_eq!(graph.num_matches_between_images(0, 1), 4);
    }

    /// Port of `TEST(CorrespondenceGraph, UpdateTwoViewGeometry)`
    /// (`correspondence_graph_test.cc:343-377`), reduced to this port's
    /// config-only edge payload (see module doc).
    #[test]
    fn update_edge_config_replaces_stored_configuration() {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(0, 10);
        graph.add_image(1, 10);
        graph
            .add_two_view_geometry(0, 1, &[(0, 0), (1, 2), (3, 7)], ConfigurationType::Calibrated)
            .unwrap();
        graph.finalize();

        assert_eq!(graph.edge(0, 1).unwrap().config, ConfigurationType::Calibrated);
        assert!(graph.update_edge_config(0, 1, ConfigurationType::Planar));
        assert_eq!(graph.edge(0, 1).unwrap().config, ConfigurationType::Planar);
        // Order-insensitive, unlike COLMAP's direction-aware version (module doc).
        assert_eq!(graph.edge(1, 0).unwrap().config, ConfigurationType::Planar);
        // Matches are untouched by a config-only update.
        assert_eq!(graph.extract_correspondences(0, 0), vec![Correspondence::new(1, 0)]);
    }

    /// Self-matches are rejected, not silently ignored (this port's
    /// deliberate deviation from COLMAP's log-and-return — see
    /// [`CorrespondenceGraphError::SelfMatch`]'s doc).
    #[test]
    fn self_match_is_rejected() {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(0, 10);
        let err = graph
            .add_two_view_geometry(0, 0, &[(0, 0)], ConfigurationType::Calibrated)
            .unwrap_err();
        assert_eq!(err, CorrespondenceGraphError::SelfMatch(0));
    }

    /// Adding the same pair twice is rejected (COLMAP `THROW_CHECK`s this).
    #[test]
    fn duplicate_pair_add_is_rejected() {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(0, 10);
        graph.add_image(1, 10);
        graph
            .add_two_view_geometry(0, 1, &[(0, 0)], ConfigurationType::Calibrated)
            .unwrap();
        let err = graph
            .add_two_view_geometry(0, 1, &[(1, 1)], ConfigurationType::Calibrated)
            .unwrap_err();
        assert_eq!(err, CorrespondenceGraphError::DuplicatePair(0, 1));
        // Also rejected in swapped order — the pair key is order-independent.
        let err_swapped = graph
            .add_two_view_geometry(1, 0, &[(1, 1)], ConfigurationType::Calibrated)
            .unwrap_err();
        assert_eq!(err_swapped, CorrespondenceGraphError::DuplicatePair(1, 0));
    }

    /// Finalize drops images that never received a correspondence — this
    /// port's documented-contract choice (module doc "Documented discrepancy").
    #[test]
    fn finalize_drops_images_without_observations() {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(0, 10);
        graph.add_image(1, 10);
        graph.add_image(2, 10); // never referenced by any match
        graph
            .add_two_view_geometry(0, 1, &[(0, 0)], ConfigurationType::Calibrated)
            .unwrap();
        assert!(graph.exists_image(2));
        graph.finalize();
        assert!(graph.exists_image(0));
        assert!(graph.exists_image(1));
        assert!(!graph.exists_image(2));
        assert_eq!(graph.num_images(), 2);
    }

    /// A four-image chain (0-1-2-3, each link sharing one point) exercises
    /// transitivity bounds beyond the three-view case: transitivity=1 sees
    /// only the direct neighbour, transitivity=2 reaches two hops, and
    /// unbounded transitivity reaches the whole chain — the exact behaviour
    /// `build_tracks_via_graph` in `incremental_sfm.rs` relies on to
    /// reproduce the legacy union-find's full-closure tracks.
    #[test]
    fn transitivity_parameter_bounds_the_closure_depth() {
        let mut graph = CorrespondenceGraph::new();
        for id in 0..4 {
            graph.add_image(id, 5);
        }
        graph
            .add_two_view_geometry(0, 1, &[(0, 0)], ConfigurationType::Calibrated)
            .unwrap();
        graph
            .add_two_view_geometry(1, 2, &[(0, 0)], ConfigurationType::Calibrated)
            .unwrap();
        graph
            .add_two_view_geometry(2, 3, &[(0, 0)], ConfigurationType::Calibrated)
            .unwrap();

        assert_eq!(graph.extract_transitive_correspondences(0, 0, 1).len(), 1); // just image 1
        assert_eq!(graph.extract_transitive_correspondences(0, 0, 2).len(), 2); // + image 2
        assert_eq!(graph.extract_transitive_correspondences(0, 0, 3).len(), 3); // + image 3
        assert_eq!(graph.extract_transitive_correspondences(0, 0, usize::MAX).len(), 3);
        // The full closure is order-independent of which endpoint you start from.
        assert_eq!(graph.extract_transitive_correspondences(3, 0, usize::MAX).len(), 3);
    }

    /// M2.1 acceptance (`docs/colmap_port_plan.md`): reproduces
    /// `examples/unordered_sfm_demo.rs`'s `verify_pairs` end-to-end for a
    /// pure-rotation pair — classify with [`TwoViewGeometryVerifier`], then
    /// feed the winning model's inliers into this graph exactly as the demo
    /// now does since M2.1 widened its keep-list to admit `PANORAMIC`
    /// (previously dropped, stricter than COLMAP's own
    /// `database_cache.cc` `UseInlierMatchesCheck`, which only excludes
    /// `WATERMARK`/too-few-matches). Confirms the graph itself never gated on
    /// `ConfigurationType` in the first place (`add_two_view_geometry`
    /// doesn't inspect `config` beyond storing it), so a `PANORAMIC` pair's
    /// correspondences reach the graph/track-building layer just like any
    /// other non-degenerate configuration.
    #[test]
    fn panoramic_classified_pair_contributes_correspondences_to_graph() {
        use super::super::colmap_verification::{TwoViewGeometryOptions, TwoViewGeometryVerifier};
        use super::super::TwoViewCorrespondence;
        use nalgebra::{Point3, UnitQuaternion, Vector3};
        use visloc_core::geometry::Pose;
        use visloc_core::types::Camera;

        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        // Same scattered, real-depth-variation point cloud as
        // `colmap_verification.rs`'s `general_scene_points` fixture.
        let mut points = Vec::new();
        for i in 0..6 {
            for j in 0..4 {
                points.push(Point3::new(
                    -1.5 + 0.6 * i as f64,
                    -1.0 + 0.7 * j as f64,
                    3.0 + 0.8 * ((i + j) % 5) as f64,
                ));
            }
        }
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.12);
        // Pure rotation: same camera center as `previous`, only re-oriented.
        let current = Pose::from_world_to_camera(yaw, Vector3::new(0.0, 0.0, 0.0));

        let correspondences: Vec<TwoViewCorrespondence> = points
            .iter()
            .filter_map(|p| {
                let p1 = camera.project(&previous.transform_world_point(p))?;
                let p2 = camera.project(&current.transform_world_point(p))?;
                Some(TwoViewCorrespondence::new(p1, p2))
            })
            .collect();
        assert!(correspondences.len() >= 20, "fixture sanity");

        let verifier = TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(&camera, 4.0));
        let report = verifier.classify(&correspondences, &camera);
        assert_eq!(
            report.config,
            ConfigurationType::Panoramic,
            "fixture sanity: pure rotation must classify PANORAMIC"
        );
        assert!(
            !report.inliers.is_empty(),
            "a PANORAMIC report must carry the winning homography's inliers, \
             not an empty list like DEGENERATE"
        );

        // Mirror `verify_pairs`'s keep-list (M2.1): PANORAMIC is now kept.
        let keep = matches!(
            report.config,
            ConfigurationType::Calibrated
                | ConfigurationType::Uncalibrated
                | ConfigurationType::Planar
                | ConfigurationType::Panoramic
                | ConfigurationType::PlanarOrPanoramic
                | ConfigurationType::Multiple
        );
        assert!(keep, "M2.1: PANORAMIC must be kept by the demo's keep-list");

        // Synthetic point2D indices: correspondence i <-> point2D i in both images.
        let matches: Vec<(usize, usize)> = report.inliers.iter().map(|&i| (i, i)).collect();
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(0, correspondences.len());
        graph.add_image(1, correspondences.len());
        let stats = graph
            .add_two_view_geometry(0, 1, &matches, report.config)
            .expect("PANORAMIC pair must be a valid graph edge, same as any other configuration");
        assert_eq!(stats.added, matches.len());
        assert!(
            graph.num_correspondences_for_image(0) > 0,
            "PANORAMIC pair's correspondences must reach the graph/track-building layer"
        );
        assert_eq!(graph.edge(0, 1).unwrap().config, ConfigurationType::Panoramic);
    }

    /// M2.1 acceptance, the negative case: a `DEGENERATE` classification
    /// (too few raw correspondences, per `colmap_verification.rs`'s
    /// `too_few_correspondences_is_degenerate`) must still contribute
    /// nothing — unaffected by M2.1's keep-list widening. Not because the
    /// graph gates on `ConfigurationType` (it never has), but because
    /// [`TwoViewGeometryVerifier`] always returns an empty inlier list for
    /// `DEGENERATE`, the same reason COLMAP's own degenerate branch never
    /// populates `inlier_matches` (`two_view_geometry.cc`'s degenerate
    /// returns).
    #[test]
    fn degenerate_classified_pair_contributes_no_correspondences() {
        use super::super::colmap_verification::TwoViewGeometryVerifier;
        use super::super::TwoViewCorrespondence;
        use nalgebra::{Point3, UnitQuaternion, Vector3};
        use visloc_core::geometry::Pose;
        use visloc_core::types::Camera;

        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        // Below the default `min_num_inliers = 15` gate.
        let points: Vec<Point3<f64>> = (0..10)
            .map(|i| Point3::new(-1.0 + 0.2 * i as f64, 0.0, 3.0))
            .collect();
        let correspondences: Vec<TwoViewCorrespondence> = points
            .iter()
            .filter_map(|p| {
                let p1 = camera.project(&previous.transform_world_point(p))?;
                let p2 = camera.project(&current.transform_world_point(p))?;
                Some(TwoViewCorrespondence::new(p1, p2))
            })
            .collect();

        let verifier = TwoViewGeometryVerifier::default();
        let report = verifier.classify(&correspondences, &camera);
        assert_eq!(report.config, ConfigurationType::Degenerate, "fixture sanity");
        assert!(report.inliers.is_empty());

        let keep = matches!(
            report.config,
            ConfigurationType::Calibrated
                | ConfigurationType::Uncalibrated
                | ConfigurationType::Planar
                | ConfigurationType::Panoramic
                | ConfigurationType::PlanarOrPanoramic
                | ConfigurationType::Multiple
        );
        assert!(!keep, "DEGENERATE must never be kept by the demo's keep-list");

        // Even if a caller ignored `keep` and tried to add the (empty)
        // inlier set anyway, the graph would record zero correspondences.
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(0, correspondences.len());
        graph.add_image(1, correspondences.len());
        let matches: Vec<(usize, usize)> = report.inliers.iter().map(|&i| (i, i)).collect();
        assert!(matches.is_empty());
        let stats = graph
            .add_two_view_geometry(0, 1, &matches, report.config)
            .expect("adding an empty match list is still a valid (if useless) edge");
        assert_eq!(stats.added, 0);
        assert_eq!(graph.num_correspondences_for_image(0), 0);
        assert_eq!(graph.num_correspondences_for_image(1), 0);
    }
}
