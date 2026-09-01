//! Incremental structure-from-motion from an **unordered** image set — the
//! COLMAP-style SfM pillar of visloc-rs.
//!
//! Unlike the stereo-VO SfM path (`--sfm-colmap-out` on
//! `stereo_vo_external_deep_files`), which needs an *ordered* video with
//! frame→frame matches, this demo takes a directory of per-image deep features
//! with **no temporal order**, builds its own view graph, and grows one
//! reconstruction:
//!
//! 1. **View graph.** `--pair-source` (default `vlad`) selects how candidate
//!    pairs are proposed:
//!    - `vlad`: a VLAD vocabulary over all descriptors gives each image a
//!      global descriptor; the top-K most similar images per image become
//!      candidate pairs.
//!    - `vlad-union`: a deterministic union of a numeric-stem local-overlap
//!      schedule (`--local-stem-window`) and the VLAD retrieval pairs.  The
//!      optional `--candidate-budget` retains local pairs first, then the
//!      highest-scoring retrieval pairs.  It is an explicit, bounded M3
//!      schedule and never consults raw matches or verification outcomes.
//!      `--rig-local-grouping` makes the local schedule camera-aware for
//!      names of the form `<camera-prefix>_<numeric-timestamp>`: temporal
//!      edges stay within a camera and bounded same-timestamp cross-camera
//!      rig edges are added deterministically.
//!    - `temporal-pyramid`: a rig-aware temporal pyramid.  Within each
//!      camera it proposes positional offsets 1, 2, 4, … up to
//!      `--temporal-pyramid-max-offset` (default 32), then adds pairs with
//!      the same timestamp across cameras, and finally fills a bounded
//!      `--candidate-budget` with highest-scoring VLAD retrieval pairs.
//!      This is deterministic and GT-free; it is useful when numeric
//!      timestamps are irregular or have large nanosecond gaps.
//!    - `vocab-tree`: `visloc_rs::vision::vocab_tree`'s hierarchical-k-means
//!      vocabulary + TF-IDF/Hamming-embedding inverted-file retrieval
//!      (COLMAP's `VocabTreePairGenerator`-equivalent, M3 in
//!      `docs/colmap_port_plan.md`) — `--vocab-tree-branching`/
//!      `--vocab-tree-depth` size the tree, `--vocab-tree-num-images` is the
//!      top-N retrieved per query image (COLMAP default 100).
//!
//!    `--exhaustive` overrides either source with all pairs.
//! 2. **Verified matches.** Each candidate pair is matched — `--matcher`
//!    (default `nn`) selects the algorithm:
//!    - `nn`: cross-checked brute-force nearest-neighbour + Lowe ratio
//!      (pre-M6 behaviour, unchanged).
//!    - `lightglue` (M6, `docs/colmap_port_plan.md`): the learned LightGlue
//!      matcher (SuperPoint variant), run in-process via ONNX Runtime
//!      (`visloc_rs::vision::features::lightglue_onnx::LightGlueOnnxMatcher`,
//!      `--lightglue-model PATH`, `onnx-inference` feature required). Unlike
//!      NN+ratio's independent per-descriptor search, LightGlue attends over
//!      *both* images' descriptors jointly — the lever M5's diagnosis
//!      motivated: ETH3D `courtyard`'s cross-component bridge pairs carry
//!      real but sparse correspondence signal that a per-descriptor ratio
//!      test cannot safely extract on a repeated-texture scene (M5's "naive
//!      rescue" experiment found classifier-passing *false* bridges from
//!      over-relaxing the ratio test — evidence the matcher itself, not just
//!      its threshold, was the bottleneck). `--matcher lightglue` replaces
//!      the matching step in *both* the main pass below and the M5
//!      rescue-bridging pass (step 4); `--rescue-match-ratio`/
//!      `--rescue-cross-check` are `nn`-only knobs, ignored under
//!      `lightglue` (see `PairMatcher::match_pair`'s doc comment). One ONNX
//!      graph is exported per camera resolution
//!      (`scripts/export_lightglue_onnx.py --width --height`); re-export for
//!      a different scene's intrinsics.
//!
//!    Every matched pair is then geometrically verified per
//!    `--verification-mode` (default `legacy`):
//!    - `legacy`: essential-matrix-only RANSAC, COLMAP's legacy fixed
//!      `5e-3`-normalized Sampson threshold (`RelativePoseEstimator`,
//!      unchanged since before M1).
//!    - `threshold-only`: the same single-model essential-matrix-only RANSAC
//!      as `legacy`, but with the per-camera pixel-derived Sampson threshold
//!      (`TwoViewGeometryOptions::for_camera`'s ≈4px-equivalent bound) instead
//!      of the fixed `5e-3` default — isolates the threshold half of the M1
//!      confound (see `docs/colmap_port_plan.md`'s "M1.1 results").
//!    - `full`: COLMAP-style multi-model (essential / fundamental /
//!      homography) verification with `ConfigurationType` classification
//!      (`visloc_rs::vision::two_view::colmap_verification`, ported from
//!      `src/colmap/estimators/two_view_geometry.cc`): only `DEGENERATE` and
//!      `WATERMARK` pairs are dropped before `incremental_sfm` ever sees
//!      them, matching COLMAP's real admission gate
//!      (`database_cache.cc`'s `UseInlierMatchesCheck`, M2.1 — see
//!      `docs/colmap_port_plan.md`). `PANORAMIC` (pure rotation, no
//!      triangulatable baseline) and unresolved `PLANAR_OR_PANORAMIC` pairs
//!      *do* contribute their homography inliers to `PairwiseMatches`, same
//!      as `PLANAR`; they just never become a *seed* pair, because
//!      `pipelines/slam/src/incremental_sfm.rs`'s own parallax gate
//!      (`place_seed_pair`) independently rejects near-zero-baseline pairs at
//!      growth time — the same "recompute and gate on triangulation angle,
//!      don't consult the stored classification" design COLMAP's own
//!      `IncrementalMapperImpl::EstimateInitialTwoViewGeometry` uses. Every
//!      other configuration (`CALIBRATED`/`UNCALIBRATED`/`PLANAR`/`MULTIPLE`)
//!      keeps its winning model's own inliers (which need not be the
//!      essential matrix's).
//!
//!    A per-`ConfigurationType` count is printed under `full`, so all three
//!    modes can be A/B'd on the same view graph — this is the M1/M1.1
//!    acceptance experiments' switch (see `docs/colmap_port_plan.md`). The
//!    legacy `--colmap-verification` boolean flag still works as a shorthand
//!    for `--verification-mode full`.
//! 3. **Incremental SfM.** [`visloc_rs::slam::incremental_sfm`] seeds from the
//!    strongest pair, registers images by PnP, triangulates tracks, and bundle-
//!    adjusts. Its first internal step — building feature tracks out of the
//!    verified pairs above — is itself an M2 A/B switch: `--track-source
//!    union-find` (default) is the original ad hoc union-find, `--track-source
//!    graph` routes through COLMAP's persistent `CorrespondenceGraph`
//!    (`visloc_rs::vision::two_view::correspondence_graph`, ported from
//!    `src/colmap/scene/correspondence_graph.{h,cc}`) instead. Both are proven
//!    to produce byte-identical tracks (see `pipelines/slam/src/
//!    incremental_sfm.rs`'s `graph_tracks_match_union_find_tracks_*` tests),
//!    so this flag is the M2 acceptance experiment's switch, not a behaviour
//!    change — see `docs/colmap_port_plan.md`'s "M2 results".
//!    `--union-traversal-order reverse-pairs|reverse-matches|reverse-both` is
//!    a separate default-off diagnostic that reorders only the accepted
//!    pair/match stream after verification; `original` is the no-op default.
//!    `physical-hash:SEED` and `physical-hash-reverse:SEED` provide a
//!    deterministic physical-edge traversal (coordinates, not row indices)
//!    while preserving the exact verified edge multiset.
//! 4. **Rescue-bridging (opt-in, `--rescue-bridging`, M5 in
//!    `docs/colmap_port_plan.md`).** Runs after the initial verification pass
//!    above. Detects whether the verified-pair graph is disconnected
//!    (`visloc_rs::vision::two_view::connected_components`) — the diagnosed
//!    ETH3D `courtyard` failure mode (images 0-24 vs 25-37 never verify a
//!    single pair against each other at any pair budget M3/M4 tried). If so,
//!    it proposes cross-component candidate pairs, ranked by a fresh VLAD
//!    global-descriptor similarity and budget-capped
//!    (`generate_bridge_candidates`), rematches each with a deliberately
//!    relaxed profile (`--rescue-match-ratio`, default a looser Lowe ratio
//!    than `--match-ratio`, and mutual-NN instead of strict cross-check
//!    unless `--rescue-cross-check` is set), and re-verifies every candidate
//!    with the *same* full [`TwoViewGeometryVerifier`] every other pair goes
//!    through — a relaxed matcher only ever *proposes* a bridge, the
//!    classifier still decides what's *admitted* (the M1.1 lesson: loose
//!    thresholds are only safe when a real classifier gates the result).
//!    Admitted pairs are appended to the same `PairwiseMatches` list that
//!    feeds `incremental_sfm`, so a successful bridge participates in track
//!    building exactly like any other verified pair.
//! 5. **Export.** The registered poses + merged multi-view tracks are written as
//!    a COLMAP text model (`cameras.txt` / `images.txt` / `points3D.txt`),
//!    ready for 3DGS / NeRF training.
//!
//! Feature-file format is the same `X Y SCORE D0 D1 …` per keypoint used by
//! `read_external_deep_features_txt` (export SuperPoint with the repo's helper
//! scripts). The image set is every file in `--features-dir` ending with
//! `--feature-suffix`, sorted lexically; each image's COLMAP name is that file
//! with the suffix replaced by `--image-suffix`.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --example unordered_sfm_demo -- \
//!     --features-dir /tmp/sp_photos \
//!     --feature-suffix _features.txt --image-suffix .png \
//!     --width 752 --height 480 --fx 458.6 --fy 457.3 --cx 367.2 --cy 248.4 \
//!     --retrieval-topk 12 --min-matches 30 \
//!     --out-colmap /tmp/photos_sfm_colmap
//! ```
//!
//! For an undistorted COLMAP model with per-image `PINHOLE` assignments, use
//! `--input-colmap-calibration MODEL_DIR` instead of the six scalar camera
//! flags.  The loader validates `cameras.txt`/`images.txt`, maps names by exact
//! name/basename/unique stem, validates decoded image dimensions when
//! `--images-dir` is available, and keeps each image's intrinsics fixed.  SfM
//! internally uses a lossless normalized-ray conversion to the first image's
//! camera convention; the exported model restores the original camera IDs and
//! native feature pixels.
//!
//! For high-resolution feature generation, add `--sift-stream-export` together
//! with `--export-features-dir DIR --export-features-only`.  This default-off
//! mode walks source images lexically, decodes and extracts one image, writes
//! its feature and `_loci.txt` files through same-directory atomic renames,
//! then releases that image before continuing.  It accepts the same optional
//! per-image calibration and validates each decoded image's dimensions.
//! Add `--sift-stream-resume` to opt into per-image completion sidecars.  A
//! sidecar is published only after both output files are complete and records
//! a stable extractor/configuration hash plus source, feature, and locus-file
//! byte hashes.  A later run resumes only when every recorded value validates;
//! missing, malformed, stale, or tampered sidecars are re-extracted.
//!
//! The optional Python `scripts/export_superpoint_lightglue.py` helper supports
//! the same safe resume contract for stereo/mono exports: use
//! `--start-index I --end-index J` for an explicit source range `[I,J)` (output
//! names retain source indices), `--skip-existing` for structural rather than
//! size-only validation, and `--manifest PATH` for an atomic SHA-256 manifest.
//! `--validate-only --manifest PATH` validates an existing range without
//! loading the optional LightGlue stack.  A range worker emits its first
//! boundary temporal match from the predecessor frame, so adjacent workers do
//! not duplicate that edge.
//!
//! `--pair-stem-window N` is an explicit, default-off candidate restriction:
//! after validating that every loaded image has a unique trailing numeric
//! stem, only pairs whose numeric stem difference is at most `N` are matched
//! and verified.  It applies to imported verified pairs and all opt-in pair
//! expansion paths as well; omitted means the historical candidate set.
//!
//! `--rig-local-grouping` is an explicit `vlad-union` option for multi-camera
//! names such as `cam4_1474975187520882738.png`.  It avoids treating the same
//! timestamp in different cameras as a duplicate global stem: local edges
//! connect timestamps within each camera (up to `--local-stem-window`) and
//! each timestamp contributes at most one canonical edge per camera pair.
//! The generated candidate manifest records this policy in validated metadata.
//!
//! `--candidate-manifest PATH` imports a versioned, image-name-bound list of
//! candidate pairs and bypasses pair generation.  `--export-candidate-manifest
//! PATH` writes the generated list atomically and exits before matching.  The
//! manifest is deliberately small and hashable, so a benchmark can cache the
//! cheap retrieval result while validating that it still belongs to the same
//! image order.  Candidate manifests contain no descriptors, raw matches, or
//! ground-truth information.
//!
//! `--max-mapper-matches-per-pair N` is an explicit resource guard for dense
//! feature sets.  It keeps the verifier/snapshot stream complete, then passes
//! only the first `N` deterministic inliers from each pair to the mapper;
//! omitted means the historical unbounded mapper input.
//!
//! `--initial-poses MODEL/images.txt` is an explicit, default-off staged
//! incremental seed.  The model's image names are matched by stem and its
//! sibling `cameras.txt` must describe the same shared pinhole calibration, or
//! the per-image calibrations supplied by `--input-colmap-calibration`.  At
//! least two supplied poses are required; they are held fixed while the full
//! loaded track graph is triangulated and missing images are grown by PnP,
//! then the ordinary final BA is allowed to release them (apart from its
//! usual gauge anchors).  It is valid only with `--mapper incremental` and
//! cannot be combined with `--seed-pair`.
//!
//! `--next-image-policy auto|count|visibility` controls the next-PnP ranking.
//! The demo default is the conservative `auto` policy: it tries visibility
//! first, compares the historical count ranking whenever visibility is
//! incomplete, and may run the existing post-refinement completion pass when
//! the selected candidate is still incomplete.  A post candidate is adopted
//! only when it strictly adds registered images. `count` remains the library
//! default and is forced for snapshot replay unless a policy is explicitly
//! requested.
//!
//! `--export-verified-pairs-snapshot PATH` writes a lossless, versioned
//! snapshot of the accepted pair/match stream after verification and all
//! configured stream-order transforms.  `--import-verified-pairs-snapshot
//! PATH` validates that snapshot against the loaded image/feature manifest and
//! camera, then bypasses matching and verification without reordering the
//! stored stream.  The older `--import-verified-pairs-file` text format is
//! unchanged.
//! `--export-verified-pairs-only` pairs with the export flag for resumable
//! matching shards: it writes the snapshot and exits before track building or
//! mapping.  It is default-off and requires an export path.
//! `--snapshot-coordinate-override-dir DIR` is a diagnostic-only companion to
//! snapshot import: after the snapshot validates against the base features, it
//! checks that `DIR` has the same image names, row counts, and descriptor bits,
//! then replaces only keypoint `(x,y)` coordinates.  Pair order, indices,
//! models, and hashes remain those of the immutable snapshot.  It requires
//! `--import-verified-pairs-snapshot` and is default-off.
//! `--snapshot-keypoints-only` is an explicit memory-saving replay mode for
//! `--import-verified-pairs-snapshot`: it is limited to file-backed features
//! and the plain incremental mapper, keeps only keypoints/row counts in the
//! mapper feature bank, and re-reads one feature file at a time to reproduce
//! the exact descriptor-bound manifest hash.  It cannot be combined with
//! coordinate overrides, feature/snapshot export, canonical row ordering,
//! orientation-locus canonicalization, or model-score diagnostics.
//! `--persistent-match-worker-plan PLAN` is the default-off M4 worker mode:
//! it consumes a versioned, image-order-bound candidate/shard plan, loads the
//! file feature bank and NN matcher once, and atomically publishes one
//! lossless verified-pair snapshot per shard.  It is restricted to
//! `files` + `nn` + `full` + plain `incremental`; the Python electro runner's
//! `--persistent-matcher` option generates and validates this plan.
//!
//! For machine-readable NN diagnostics (and an early exit without
//! reconstruction), add `--diagnose-pairs-csv /tmp/pairs.csv`; it uses the
//! existing candidate source, or `--exhaustive` for every image pair. To
//! inspect every pair touching selected image stems, use
//! `--diagnose-pair-stems DSC_0297,DSC_0309`. An optional
//! `--import-matches-file` (or supplement file) adds COLMAP pair/index overlap
//! columns to the CSV; `--import-verified-pairs-file` adds COLMAP verified
//! inlier/configuration columns and may be combined with the raw import here.
//! In `--feature-extractor sift` mode, `--sift-scale-adaptive-gradients`
//! enables the opt-in VLFeat-style scale-space gradient descriptor path;
//! leaving it out preserves the historical direct-source gradients.
//! `--sift-vlfeat-compatible-descriptor` instead selects the complete
//! VLFeat/COLMAP-compatible descriptor convention (m=3, octave gradients,
//! UBC orientation layout and 512-equivalent quantization); it is default-off
//! and cannot be combined with the partial experimental descriptor flags.
//! `--sift-dsp` enables the corrected descriptor's published DSP-SIFT preset
//! (uniform domain-size samples `1/6…4/3 × σ`, 15 samples); it is default-off
//! and requires `--sift-vlfeat-compatible-descriptor`. The existing
//! `--sift-dsp-num-scales` override is retained only for bounded experiments.
//! `--sift-vlfeat-compatible-detector` independently selects the matching
//! VLFeat/COLMAP DoG detector contract (first octave -1, quadratic localization,
//! source orientation assignment and large-scale-first capping); it is also
//! default-off and requires isotropic keypoints.
//! `--sift-vlfeat-bilinear-orientations` additionally enables the bilinear
//! orientation-bin switch used by COLMAP's vendored VLFeat build; it requires
//! the compatible detector flag and remains default-off.
//! `--sift-vlfeat-compatible-output-order` makes the compatible detector's
//! COLMAP source-order contract explicit (ascending retained octave/level,
//! then VLFeat scan/orientation order); it requires the detector flag and is
//! default-off because the current compatible detector already emits it.
//! `--sift-colmap-compatible-grayscale` keeps the legacy decoder unchanged
//! but applies COLMAP's float32 RGB-to-gray rounding (and ignores alpha) to
//! SIFT input images; it is useful for preprocessing-parity experiments.
//! `--sift-split-colmap-detector-grayscale` is a stricter diagnostic split:
//! it detects/orients on that rounded image but computes compatible descriptors
//! from the legacy floor image. It requires both compatible SIFT modes and is
//! mutually exclusive with the all-rounded flag.
//! `--stable-track-order` makes track/observation traversal use physical
//! keypoint coordinates (and descriptor contents only for co-located ties),
//! so a permutation of feature rows cannot change mapper landmark/PnP order.
//! It is default-off and does not alter matching or legacy output.
//! `--canonical-feature-order` additionally rewrites each feature file into
//! that physical order before matching (and remaps imported indices), making
//! the complete NN/mapping path permutation-invariant. It is default-off.
//! `--orientation-locus-canonicalization` retains all orientation rows during
//! NN matching but remaps verified correspondences to one deterministic
//! `(x,y,scale)` representative per image locus before track construction.
//! SIFT extraction carries this metadata in memory (and in `_loci.txt`
//! sidecars on export); six-column COLMAP affine rows are recognized on file
//! import. Metadata-free legacy files are unchanged. It is default-off.
//! `--incremental-correspondence-triangulation` is a separate default-off
//! mapper path: it builds conflict-free tracks with an explicit
//! observation-to-point map and re-triangulates live points after each PnP
//! registration, while retaining the plain seed/growth schedule. It cannot
//! be combined with `--colmap-style`.
//! `--diagnose-colmap-track-membership MODEL/points3D.txt` is a separate
//! default-off oracle diagnostic: it imports only the validated
//! `(IMAGE_ID, POINT2D_IDX)` partitions from that sparse model (using sibling
//! `images.txt` for names/row counts), ignores COLMAP XYZ/poses, and reruns
//! the plain incremental mapper with fresh triangulation/BA. Historical
//! source tracks containing multiple observations from one image are skipped
//! and counted explicitly.
//! `--pose-guided-track-splitting` is a separate default-off diagnostic that
//! waits for a complete posed model, then splits legacy union components
//! (including same-image-conflict components) by deterministic wide-baseline
//! 3-D hypotheses, one observation per image, cheirality/reprojection gates,
//! and fixed-pose local point refinement before one guarded final BA. It is
//! intentionally incompatible with imported oracle memberships and alternate
//! track builders; it may be composed after geometry conflict recovery, while
//! incomplete pose models leave the legacy result unchanged.
//! `--pose-guided-track-splitting-graph-support` is a separate default-off
//! admission rule layered on that diagnostic: after the two-view anchor, each
//! added observation must have direct verified edges to at least two distinct
//! images already in the hypothesis, and multi-view emissions need two
//! independent cross-image supports. Two-view hypotheses remain valid and the
//! original pose-guided strategy is unchanged when this subflag is omitted.
//! `--pose-guided-track-splitting-bridge-cuts` is a separate default-off
//! refinement before that split: Tarjan bridge candidates are cut only when
//! both sides have at least two images and independently valid posed
//! triangulations, while the combined observations cannot fit one point.
//! Singleton/invalid sides and geometrically valid sparse chains remain intact;
//! the resulting subcomponents then use the ordinary pose-guided splitter.
//! `--pose-guided-split-max-reproj PX` optionally narrows only the pose-guided
//! split's candidate observation/point gate; omitted reuses the ordinary
//! `--max-reproj` value and does not alter mapper/PnP/BA thresholds.
//! `--pose-guided-track-splitting-iterations N` bounds repeated split passes
//! from the original components (default `1` when splitting is enabled); it
//! accepts `1..=8` and stops/rolls back on a non-improving pass.
//! `--pose-guided-track-merging` is a separate default-off post-split pass:
//! complementary split tracks may be merged only across a verified edge when
//! their image sets are disjoint and their complete union fits one posed point
//! under the split reprojection gate.  Candidate unions are deterministic and
//! recomputed after every accepted merge; it requires pose-guided splitting.
//! `--pose-guided-merge-max-reproj PX` optionally widens only that union-fit
//! gate; omitted inherits the split gate, while post-BA validation still uses
//! the ordinary `--max-reproj` hard bound.
//! `--final-min-track-length 3` is a separate default-off final-support gate:
//! after registration and all splitting/recovery passes, length-2 landmarks
//! are removed, the remaining points are re-triangulated and BA-refined, and
//! the complete pre-gate state is restored if registered-camera support or the
//! remaining-support objective becomes invalid. It never changes growth/PnP.
//! `--cycle-supported-tracks` is a separate default-off track strategy: it
//! ranks accepted correspondences by exact three-view cycle support, then by
//! retained geometric/pair confidence and stable physical keys, while
//! enforcing one observation per image per track. It does not replace the
//! legacy or stable strategies unless explicitly selected.
//! For a controlled mapper seed replay, `--seed-pair I,J` restricts the
//! otherwise unchanged seed candidate list to that normalized image-index
//! pair; it is default-off and intended for diagnostics.
//! `--sequence-fallback-carry-scale` is a default-off after-post policy that
//! carries the accepted baseline magnitude across consecutive provisional
//! registrations; it requires the relaxed projection and after-post flags.
//! When investigating the opt-in calibrated F→E path,
//! `VISLOC_SFM_DEBUG_DUMP_F2E_DIAGNOSTICS=1` emits the calibrated-F singular
//! values, essential-manifold projection distortion, F/E residual agreement,
//! cheirality margin, and deterministic subset-refit pose spread for every
//! `UNCALIBRATED` candidate.
//! `--strict-uncalibrated-f-to-essential` reuses that gate but drops failing
//! known-intrinsics F-winning edges instead of falling back to their F
//! correspondences; it is default-off and has no rotation-only edge fallback.
//! `--calibrated-essential-primary` is a separate default-off policy that
//! promotes a robust, sufficiently supported direct-E estimate to the primary
//! track model for known-intrinsics F-winning pairs; F/H remain diagnostics.
//! `--final-ba-polish-iterations N` optionally runs a fixed-support pure-L2
//! polish after all registration/refinement passes; `0` (the default) is a
//! no-op and any worsening/non-finite solve is rolled back.
//! `--ba-huber-delta PX` is a default-off override for the shared periodic and
//! final/global BA Huber threshold; omission preserves the historical `3 px`
//! setting and the flag requires a positive finite pixel value.
//! `--geometry-weighted-ba` adds a separate default-off final fixed-support
//! solve whose observation weights are a pre-BA, clamped `sin²(parallax)` proxy;
//! track/observation support and registration are unchanged.
//! `--freeze-ill-conditioned-landmarks` applies a default-off conditioning
//! safeguard: weak pre-BA point blocks with an already-bad reprojection are
//! omitted from that BA's residual rows, while well-fitting weak points remain
//! variables. This avoids using a frozen, wrong point as a camera constraint.
//! `--landmark-ba-warm-start-iterations N` runs a default-off, camera-fixed
//! point-only BA before each global/periodic joint BA; `0` is a no-op and any
//! non-finite or cost-increasing warm start is rolled back.
//! `--landmark-ba-warm-start-min-registered-images N` optionally scopes that
//! experiment to BA calls with at least `N` registered cameras (`0` means all).
//! `--periodic-ba-min-registered-images N` is a default-off plain-growth
//! schedule diagnostic: it defers periodic BA until `N` cameras are registered
//! (`0` keeps the historical schedule), without suppressing the configured
//! final BA.
//! `--global-ba-max-refinements N` overrides the maximum number of follow-up
//! global BA → complete → filter rounds used by `--colmap-style` or
//! `--final-iterative-refinement`; `0` keeps the initial global BA and skips
//! follow-up rounds.  Omission preserves [`IncrementalSfmConfig`]'s default of
//! `5`, and this control does not affect the ordinary one-shot final BA.
//! `--ba-linear-solver dense|sparse` is a default-off solver A/B for the
//! Schur-reduced BA system; omission keeps the historical dense backend.
//! `--diagnose-model-score MODEL/images.txt` reads a completed COLMAP model and
//! scores every imported verified correspondence against its pose-induced
//! calibrated epipolar geometry, including a deterministic hash-held-out
//! subset, then exits without matching or reconstruction. It requires
//! `--import-verified-pairs-file` and is default-off.
//! For a numerical BA audit, combine `VISLOC_SFM_DEBUG=1`,
//! `VISLOC_SFM_DEBUG_BA=1`, and optionally `VISLOC_SFM_DEBUG_BA_STEPS=1`;
//! adding `VISLOC_SFM_DEBUG_BA_JACOBIANS=1` compares a bounded live-state
//! sample of visual Jacobians to central differences. These environment gates
//! are diagnostic-only and do not alter reconstruction behavior.
//!
//! Add `--verification-mode threshold-only` or `--verification-mode full`
//! (or the legacy `--colmap-verification` boolean, equivalent to `full`) to
//! swap in the COLMAP-style two-view verification paths described above
//! instead of the default legacy essential-matrix-only path; see
//! `verify_pairs`'s doc comment and `docs/colmap_port_plan.md`'s M1/M1.1
//! sections.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
#[cfg(feature = "image-io")]
use std::io::Read;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use nalgebra::{Matrix3, Point2, Point3, UnitQuaternion, Vector3};
use rayon::prelude::*;
use visloc_rs::slam::{
    run_fixed_rotation_support_bundle_adjustment, run_fixed_support_bundle_adjustment,
};
#[cfg(feature = "onnx-inference")]
use visloc_rs::vision::features::lightglue_onnx::LightGlueOnnxMatcher;
#[cfg(feature = "image-io")]
use visloc_rs::vision::features::sift::{
    describe_sift_keypoints, extract_sift, GrayImage, SiftConfig, SiftError, SiftKeypoint,
};
#[cfg(feature = "onnx-inference")]
use visloc_rs::vision::features::superpoint_onnx::OnnxBackend;
use visloc_rs::vision::place_recognition::{cosine_similarity, vlad, Vocabulary};
use visloc_rs::vision::two_view::{
    connected_components, estimate_fundamental_dlt, fundamental_squared_sampson_error,
    generate_bridge_candidates, homography_squared_error, recover_relative_pose_with_options,
    BridgeCandidateOptions, CheiralityOptions, ConfigurationType,
    EightPointEssentialMatrixEstimator, EssentialMatrixEstimator, EssentialRansac,
    EssentialRansacConfig, RelativePoseEstimator, TwoViewCorrespondence, TwoViewGeometryOptions,
    TwoViewGeometryReport, TwoViewGeometryVerifier,
};
use visloc_rs::vision::vocab_tree::{
    generate_pairs, HkmBuildOptions, VocabTree, VocabTreeOptions, VocabTreePairGeneratorOptions,
};
use visloc_rs::{
    bearing_alignment_error_deg, estimate_free_poses_from_prior_rays,
    filter_pose_priors_by_track_quality, gt_bearing_in_prior_frame, incremental_sfm,
    incremental_sfm_with_initial_poses, incremental_sfm_with_sequence_fallback_overrides,
    incremental_sfm_with_track_membership, pair_correspondences, pair_essential_mean_sampson_error,
    prior_free_essential_gt_bearing_error_deg, read_external_deep_features_txt,
    reconstruct_global_sfm, reconstruct_global_sfm_with_priors, relative_pose_from_essential,
    rematch_essential_admission_ok, triangulate_two_view_left_frame,
    write_colmap_reconstruction_for_3dgs, write_colmap_reconstruction_for_3dgs_with_cameras,
    BaConfig, BruteForceMatcher, Camera, CameraModel, DescriptorMatch, FeatureSet,
    GlobalReconstructionTuning, IncrementalSfmConfig, LinearSolver, Matcher, NextImagePolicy,
    PairwiseMatches, PerImageCameras, Pose, RobustKernel, TrackSource, SE3,
};

use visloc_rs::slam::incremental_sfm::log_process_memory;
use visloc_rs::verified_pair_snapshot::{
    self, PairRecord as SnapshotPairRecord, Snapshot as VerifiedPairSnapshot,
};

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn trim_process_allocator() {
    unsafe extern "C" {
        fn malloc_trim(pad: usize) -> std::ffi::c_int;
    }
    // SAFETY: `malloc_trim(0)` has no pointer arguments and only asks glibc
    // to return currently unused allocator pages to the operating system.
    unsafe {
        malloc_trim(0);
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn trim_process_allocator() {}

/// A COLMAP-export landmark: world position + `(image, keypoint, pixel)` track.
type ExportLandmark = (Point3<f64>, Vec<(usize, usize, Point2<f64>)>);

/// The M1/M1.1 two-view verification A/B switch — see the file header and
/// `verify_pairs`'s doc comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerificationMode {
    /// Essential-matrix-only RANSAC, legacy fixed `5e-3`-normalized Sampson
    /// threshold. The M1 "OFF" path, byte-identical to pre-M1 behaviour.
    Legacy,
    /// Essential-matrix-only RANSAC (same single-model estimator as
    /// `Legacy`), but with the per-camera pixel-derived Sampson threshold
    /// (`TwoViewGeometryOptions::for_camera`) instead of the fixed default.
    /// No fundamental/homography models, no `ConfigurationType`
    /// classification, no watermark detection. The M1.1 ablation mode.
    ThresholdOnly,
    /// Full COLMAP-style `TwoViewGeometryVerifier` (E/F/H + classification).
    /// The M1 "ON" path, byte-identical to pre-M1.1 `--colmap-verification`.
    Full,
}

impl std::str::FromStr for VerificationMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "legacy" => Ok(Self::Legacy),
            "threshold-only" => Ok(Self::ThresholdOnly),
            "full" => Ok(Self::Full),
            other => Err(format!(
                "unknown --verification-mode {other:?} (expected legacy|threshold-only|full)"
            )),
        }
    }
}

/// The M3 pair-generation A/B switch (`docs/colmap_port_plan.md`'s "M3
/// results"): which candidate-pair source feeds two-view verification.
/// [`PairSource::Vlad`] (default) is the pre-M3 flat-VLAD top-K path,
/// unchanged (`candidate_pairs_vlad`, formerly this file's only
/// `candidate_pairs`). [`PairSource::VocabTree`] routes through
/// `visloc_rs::vision::vocab_tree`'s hierarchical-k-means +
/// TF-IDF/Hamming-embedding retrieval instead (COLMAP's
/// `VocabTreePairGenerator`-equivalent, `src/colmap/controllers/pairing.h`)
/// — see `candidate_pairs_vocab_tree`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairSource {
    /// Flat-VLAD top-K cosine retrieval (pre-M3 behaviour, unchanged).
    Vlad,
    /// VLAD top-K pairs retained only when retrieval is mutual.
    VladMutual,
    /// Union of a bounded numeric-stem local schedule and flat-VLAD
    /// retrieval.  Local edges are retained first when a budget is applied.
    VladUnion,
    /// Rig-aware temporal-pyramid offsets, same-timestamp cross-camera edges,
    /// and a VLAD fill pass under an optional candidate budget.
    TemporalPyramid,
    /// Hierarchical-k-means vocab-tree retrieval (M3).
    VocabTree,
    /// COLMAP's `TransitivePairGenerator` port: propose pairs through the
    /// *verified-match* graph — images that share a matched partner but have
    /// no direct pair yet get proposed (`pairing.cc`). Runs a vocab-tree
    /// base pass, then expands transitively for
    /// [`TRANSITIVE_ROUNDS`] rounds.
    Transitive,
}

impl std::str::FromStr for PairSource {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "vlad" => Ok(Self::Vlad),
            "vlad-mutual" => Ok(Self::VladMutual),
            "vlad-union" => Ok(Self::VladUnion),
            "temporal-pyramid" => Ok(Self::TemporalPyramid),
            "vocab-tree" => Ok(Self::VocabTree),
            "transitive" => Ok(Self::Transitive),
            other => Err(format!(
                "unknown --pair-source {other:?} (expected vlad|vlad-mutual|vlad-union|temporal-pyramid|vocab-tree|transitive)"
            )),
        }
    }
}

/// Essential matrix `E` with `x_jᵀ E x_i = 0` from absolute poses
/// (`X_j = R X_i + t`, `E = [t]× R`).
fn essential_from_absolute_poses(pose_i: &Pose, pose_j: &Pose) -> Option<Matrix3<f64>> {
    let rel = pose_j
        .world_to_camera
        .compose(&pose_i.world_to_camera.inverse());
    let t = rel.translation;
    if t.norm() < 1e-9 {
        return None;
    }
    let r = rel.rotation.to_rotation_matrix().into_inner();
    let t_skew = Matrix3::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0);
    Some(t_skew * r)
}

/// Parse COLMAP text `images.txt` into `{stem → Pose}` (world-to-camera).
fn poses_from_colmap_images_txt(path: &Path) -> Result<HashMap<String, Pose>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let mut out = HashMap::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        // IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        if parts.len() < 10 {
            continue;
        }
        let name = parts[9];
        if Path::new(name).extension().is_none() {
            continue;
        }
        let qw: f64 = parts[1].parse().map_err(|e| format!("{e}"))?;
        let qx: f64 = parts[2].parse().map_err(|e| format!("{e}"))?;
        let qy: f64 = parts[3].parse().map_err(|e| format!("{e}"))?;
        let qz: f64 = parts[4].parse().map_err(|e| format!("{e}"))?;
        let tx: f64 = parts[5].parse().map_err(|e| format!("{e}"))?;
        let ty: f64 = parts[6].parse().map_err(|e| format!("{e}"))?;
        let tz: f64 = parts[7].parse().map_err(|e| format!("{e}"))?;
        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz));
        let pose = Pose::from_world_to_camera(q, Vector3::new(tx, ty, tz));
        out.insert(
            Path::new(name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(name)
                .to_string(),
            pose,
        );
        // Skip points2D row when present.
        if let Some(nxt) = lines.peek() {
            let n = nxt.trim();
            if !n.is_empty() && !n.starts_with('#') {
                let np: Vec<&str> = n.split_whitespace().collect();
                let looks_image = np.len() >= 10
                    && np[0].parse::<i64>().is_ok()
                    && Path::new(np[9]).extension().is_some();
                if !looks_image {
                    lines.next();
                }
            }
        }
    }
    Ok(out)
}

/// Parse and validate a partial COLMAP pose model for the opt-in staged
/// incremental path. Unlike the diagnostics-only parser above, this helper
/// rejects duplicate/unknown image stems and checks the sibling camera model
/// before returning an index-aligned pose vector.
fn initial_poses_from_colmap_images_txt(
    path: &Path,
    image_names: &[String],
    camera: &Camera,
) -> Result<Vec<Option<Pose>>, String> {
    initial_poses_from_colmap_images_txt_with_expected_cameras(path, image_names, camera, None)
}

/// Parse the opt-in initial-pose model while validating each pose's camera
/// against the loaded image calibration.  `expected_cameras` is `None` for
/// the historical shared-camera path; when present it is indexed like
/// `image_names` and permits a pose model to use several COLMAP camera IDs.
fn initial_poses_from_colmap_images_txt_with_expected_cameras(
    path: &Path,
    image_names: &[String],
    camera: &Camera,
    expected_cameras: Option<&[Camera]>,
) -> Result<Vec<Option<Pose>>, String> {
    if let Some(expected_cameras) = expected_cameras {
        if expected_cameras.len() != image_names.len() {
            return Err(format!(
                "--initial-poses per-image camera count {} does not match loaded image count {}",
                expected_cameras.len(),
                image_names.len()
            ));
        }
    }
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read --initial-poses model {path:?}: {error}"))?;
    let mut entries: Vec<(String, u64, Pose)> = Vec::new();
    let mut source_stems = HashSet::new();
    for (line_number, line) in text.lines().enumerate() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 10 || parts[0].starts_with('#') {
            continue;
        }
        // A COLMAP points2D row can also contain many tokens. The image
        // header is distinguished by its integer image/camera ids and a
        // filename-like final token.
        let Ok(_image_id) = parts[0].parse::<u64>() else {
            continue;
        };
        let Ok(camera_id) = parts[8].parse::<u64>() else {
            continue;
        };
        let name = parts[9];
        if Path::new(name).extension().is_none() {
            continue;
        }
        let values = parts[1..8]
            .iter()
            .map(|value| value.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                format!(
                    "invalid pose values in --initial-poses {path:?} line {}: {error}",
                    line_number + 1
                )
            })?;
        if !values.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "non-finite pose in --initial-poses {path:?} line {}",
                line_number + 1
            ));
        }
        let quaternion_norm = values[0..4].iter().map(|value| value * value).sum::<f64>();
        if quaternion_norm <= 1.0e-24 {
            return Err(format!(
                "zero quaternion in --initial-poses {path:?} line {}",
                line_number + 1
            ));
        }
        let stem = image_stem(name).to_owned();
        if !source_stems.insert(stem.clone()) {
            return Err(format!(
                "duplicate image stem {stem:?} in --initial-poses model {path:?}"
            ));
        }
        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            values[0], values[1], values[2], values[3],
        ));
        let pose = Pose::from_world_to_camera(q, Vector3::new(values[4], values[5], values[6]));
        entries.push((stem, camera_id, pose));
    }
    if entries.len() < 2 {
        return Err(format!(
            "--initial-poses model {path:?} contains only {} usable image poses; at least 2 are required",
            entries.len()
        ));
    }

    let mut loaded_by_stem = HashMap::new();
    for (image, name) in image_names.iter().enumerate() {
        let stem = image_stem(name).to_owned();
        if loaded_by_stem.insert(stem.clone(), image).is_some() {
            return Err(format!(
                "loaded images contain duplicate stem {stem:?}; --initial-poses cannot map it uniquely"
            ));
        }
    }
    let mut used_camera_ids = HashSet::new();
    let mut poses = vec![None; image_names.len()];
    let mut camera_id_by_image = vec![None; image_names.len()];
    for (stem, camera_id, pose) in entries {
        let Some(&image) = loaded_by_stem.get(&stem) else {
            return Err(format!(
                "--initial-poses model contains unknown image stem {stem:?}"
            ));
        };
        used_camera_ids.insert(camera_id);
        poses[image] = Some(pose);
        camera_id_by_image[image] = Some(camera_id);
    }
    let seeded = poses.iter().filter(|pose| pose.is_some()).count();
    if seeded < 2 {
        return Err(format!(
            "--initial-poses model overlaps loaded images at only {seeded} pose(s); at least 2 are required"
        ));
    }

    let camera_path = path
        .parent()
        .map(|parent| parent.join("cameras.txt"))
        .ok_or_else(|| format!("--initial-poses path {path:?} has no model directory"))?;
    let camera_text = std::fs::read_to_string(&camera_path).map_err(|error| {
        format!("--initial-poses requires readable sibling cameras.txt at {camera_path:?}: {error}")
    })?;
    if let Some((k1, k2)) = camera.radial_distortion() {
        if k1.abs() > 1.0e-12 || k2.abs() > 1.0e-12 {
            return Err(
                "--initial-poses COLMAP PINHOLE validation does not support nonzero input distortion"
                    .into(),
            );
        }
    }
    let mut expected_by_camera_id: HashMap<u64, &Camera> = HashMap::new();
    if let Some(expected_cameras) = expected_cameras {
        for (image, camera_id) in camera_id_by_image.iter().enumerate() {
            let Some(camera_id) = camera_id else {
                continue;
            };
            let expected_camera = &expected_cameras[image];
            if let Some(previous) = expected_by_camera_id.insert(*camera_id, expected_camera) {
                if previous != expected_camera {
                    return Err(format!(
                        "--initial-poses camera id {camera_id} maps to incompatible loaded per-image calibrations"
                    ));
                }
            }
        }
    }
    let mut matched_camera_ids = HashSet::new();
    for (line_number, line) in camera_text.lines().enumerate() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 8 || line.trim_start().starts_with('#') {
            continue;
        }
        let Ok(camera_id) = parts[0].parse::<u64>() else {
            continue;
        };
        if !used_camera_ids.contains(&camera_id) {
            continue;
        }
        if parts[1] != "PINHOLE" {
            return Err(format!(
                "--initial-poses camera {camera_id} uses {}, expected PINHOLE (line {})",
                parts[1],
                line_number + 1
            ));
        }
        let expected_camera = expected_by_camera_id
            .get(&camera_id)
            .copied()
            .unwrap_or(camera);
        let expected = expected_camera.intrinsics().ok_or_else(|| {
            format!(
                "--initial-poses expected camera for CAMERA_ID {camera_id} has no finite pinhole intrinsics"
            )
        })?;
        let width = parts[2].parse::<u32>().map_err(|error| {
            format!(
                "invalid width in --initial-poses cameras.txt line {}: {error}",
                line_number + 1
            )
        })?;
        let height = parts[3].parse::<u32>().map_err(|error| {
            format!(
                "invalid height in --initial-poses cameras.txt line {}: {error}",
                line_number + 1
            )
        })?;
        if width != expected_camera.width || height != expected_camera.height {
            return Err(format!(
                "--initial-poses camera {camera_id} dimensions {width}x{height} disagree with input camera {}x{}",
                expected_camera.width, expected_camera.height
            ));
        }
        let params = parts[4..8]
            .iter()
            .map(|value| value.parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                format!("invalid intrinsics in --initial-poses cameras.txt: {error}")
            })?;
        for (actual, expected_value) in params
            .iter()
            .zip([expected.0, expected.1, expected.2, expected.3])
        {
            let tolerance = 1.0e-6 * expected_value.abs().max(1.0);
            if !actual.is_finite() || (actual - expected_value).abs() > tolerance {
                return Err(format!(
                    "--initial-poses camera {camera_id} intrinsics disagree with input camera: model={params:?}, input=({:.9},{:.9},{:.9},{:.9})",
                    expected.0, expected.1, expected.2, expected.3
                ));
            }
        }
        if !matched_camera_ids.insert(camera_id) {
            return Err(format!(
                "duplicate camera id {camera_id} in --initial-poses cameras.txt"
            ));
        }
    }
    if matched_camera_ids.len() != used_camera_ids.len() {
        let missing: Vec<u64> = used_camera_ids
            .difference(&matched_camera_ids)
            .copied()
            .collect();
        return Err(format!(
            "--initial-poses cameras.txt is missing camera id(s) used by poses: {missing:?}"
        ));
    }
    Ok(poses)
}

/// Parsed observation-only membership from a COLMAP sparse model.  The
/// exporter deliberately discards point coordinates, colors, reprojection
/// errors, and camera poses: the mapper must re-triangulate this partition
/// using the currently loaded feature pixels and intrinsics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ColmapTrackMembership {
    tracks: Vec<Vec<(usize, usize)>>,
    source_points: usize,
    source_observations: usize,
    retained_observations: usize,
    skipped_conflicting_points: usize,
    skipped_conflicting_observations: usize,
}

/// Read COLMAP `points3D.txt` membership and validate it against the loaded
/// feature manifest.  COLMAP's text model identifies observations by
/// `(IMAGE_ID, POINT2D_IDX)`, so the sibling `images.txt` is required to map
/// image IDs to the loaded image names and to validate the point2D row count.
/// A few historical sparse models contain a point with two observations from
/// one image.  Such a point cannot be represented by the mapper's
/// one-observation-per-image invariant; it is excluded as an invalid source
/// track and counted explicitly instead of silently selecting one row.
fn parse_colmap_track_membership(
    points_path: &Path,
    image_names: &[String],
    features: &[FeatureSet],
) -> Result<ColmapTrackMembership, String> {
    if image_names.len() != features.len() {
        return Err(format!(
            "COLMAP track membership manifest mismatch: {} image names vs {} feature sets",
            image_names.len(),
            features.len()
        ));
    }
    let images_path = points_path
        .parent()
        .map(|parent| parent.join("images.txt"))
        .ok_or_else(|| format!("{points_path:?} has no sibling images.txt directory"))?;
    let images_file = std::fs::File::open(&images_path)
        .map_err(|error| format!("cannot read COLMAP track sibling {images_path:?}: {error}"))?;
    let mut image_entries: HashMap<u64, (String, usize)> = HashMap::new();
    let mut image_names_seen = HashSet::new();
    let mut image_lines = BufReader::new(images_file).lines().enumerate();
    while let Some((line_index, line_result)) = image_lines.next() {
        let line = line_result.map_err(|error| {
            format!(
                "cannot read COLMAP images.txt line {}: {error}",
                line_index + 1
            )
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 10 {
            continue;
        }
        let Ok(image_id) = parts[0].parse::<u64>() else {
            continue;
        };
        if parts[8].parse::<u64>().is_err() {
            continue;
        }
        let name = parts[9..].join(" ");
        if Path::new(&name).extension().is_none() {
            continue;
        }
        if image_entries.contains_key(&image_id) {
            return Err(format!(
                "duplicate IMAGE_ID {image_id} in COLMAP images.txt line {}",
                line_index + 1
            ));
        }
        if !image_names_seen.insert(name.clone()) {
            return Err(format!(
                "duplicate image name {name:?} in COLMAP images.txt line {}",
                line_index + 1
            ));
        }
        let points_line = image_lines.next().ok_or_else(|| {
            format!(
                "COLMAP images.txt has no POINTS2D row after IMAGE_ID {image_id} line {}",
                line_index + 1
            )
        })?;
        let points_line_number = points_line.0 + 1;
        let points_line = points_line.1.map_err(|error| {
            format!("cannot read COLMAP images.txt line {points_line_number}: {error}")
        })?;
        let point_tokens: Vec<&str> = points_line.split_whitespace().collect();
        if point_tokens.len() % 3 != 0 {
            return Err(format!(
                "COLMAP images.txt POINTS2D row after IMAGE_ID {image_id} has {} fields, not a multiple of 3",
                point_tokens.len()
            ));
        }
        for chunk in point_tokens.chunks_exact(3) {
            chunk[0].parse::<f64>().map_err(|error| {
                format!(
                    "invalid POINTS2D x in COLMAP images.txt line {points_line_number}: {error}"
                )
            })?;
            chunk[1].parse::<f64>().map_err(|error| {
                format!(
                    "invalid POINTS2D y in COLMAP images.txt line {points_line_number}: {error}"
                )
            })?;
            chunk[2].parse::<i64>().map_err(|error| {
                format!(
                    "invalid POINTS2D point id in COLMAP images.txt line {points_line_number}: {error}"
                )
            })?;
        }
        image_entries.insert(image_id, (name, point_tokens.len() / 3));
    }
    if image_entries.len() != image_names.len() {
        return Err(format!(
            "COLMAP images.txt contains {} usable images, loaded feature manifest contains {}",
            image_entries.len(),
            image_names.len()
        ));
    }
    let mut loaded_by_name = HashMap::new();
    for (image, name) in image_names.iter().enumerate() {
        if loaded_by_name.insert(name.as_str(), image).is_some() {
            return Err(format!(
                "loaded feature manifest repeats image name {name:?}"
            ));
        }
    }
    let source_names: HashSet<&str> = image_entries
        .values()
        .map(|(name, _)| name.as_str())
        .collect();
    let missing: Vec<&str> = image_names
        .iter()
        .filter_map(|name| (!source_names.contains(name.as_str())).then_some(name.as_str()))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "COLMAP image/name manifest does not cover loaded images; missing={missing:?}"
        ));
    }
    let mut image_id_to_index = HashMap::new();
    for (image_id, (name, row_count)) in image_entries {
        let Some(&image) = loaded_by_name.get(name.as_str()) else {
            return Err(format!(
                "COLMAP images.txt image {name:?} (IMAGE_ID {image_id}) is absent from loaded feature manifest"
            ));
        };
        if row_count != features[image].keypoints.len()
            || row_count != features[image].descriptors.len()
        {
            return Err(format!(
                "POINTS2D row count for {name:?} is {row_count}, loaded feature set has {} keypoints / {} descriptors",
                features[image].keypoints.len(),
                features[image].descriptors.len()
            ));
        }
        image_id_to_index.insert(image_id, image);
    }
    debug_assert_eq!(image_id_to_index.len(), image_names.len());

    let points_file = std::fs::File::open(points_path)
        .map_err(|error| format!("cannot read COLMAP points3D file {points_path:?}: {error}"))?;
    let mut result = ColmapTrackMembership::default();
    let mut owned_observations = HashSet::new();
    for (line_index, line_result) in BufReader::new(points_file).lines().enumerate() {
        let line = line_result.map_err(|error| {
            format!(
                "cannot read COLMAP points3D line {}: {error}",
                line_index + 1
            )
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 8 || (parts.len() - 8) % 2 != 0 {
            return Err(format!(
                "COLMAP points3D line {} has malformed TRACK[] fields",
                line_index + 1
            ));
        }
        parts[0].parse::<u64>().map_err(|error| {
            format!(
                "invalid POINT3D_ID in COLMAP points3D line {}: {error}",
                line_index + 1
            )
        })?;
        for (column, value) in parts[1..8].iter().enumerate() {
            value.parse::<f64>().map_err(|error| {
                format!(
                    "invalid point metadata column {} in COLMAP points3D line {}: {error}",
                    column + 1,
                    line_index + 1
                )
            })?;
        }
        result.source_points += 1;
        let mut track = Vec::with_capacity((parts.len() - 8) / 2);
        let mut track_images = HashSet::new();
        let mut conflicting = false;
        for pair in parts[8..].chunks_exact(2) {
            let image_id = pair[0].parse::<u64>().map_err(|error| {
                format!(
                    "invalid IMAGE_ID in COLMAP points3D line {}: {error}",
                    line_index + 1
                )
            })?;
            let keypoint = pair[1].parse::<usize>().map_err(|error| {
                format!(
                    "invalid POINT2D_IDX in COLMAP points3D line {}: {error}",
                    line_index + 1
                )
            })?;
            let Some(&image) = image_id_to_index.get(&image_id) else {
                return Err(format!(
                    "COLMAP points3D line {} references unknown IMAGE_ID {image_id}",
                    line_index + 1
                ));
            };
            if keypoint >= features[image].keypoints.len() {
                return Err(format!(
                    "COLMAP points3D line {} references {name} keypoint {keypoint}, outside loaded feature rows",
                    line_index + 1,
                    name = image_names[image]
                ));
            }
            if !track_images.insert(image) {
                conflicting = true;
            }
            track.push((image, keypoint));
        }
        result.source_observations += track.len();
        if conflicting {
            result.skipped_conflicting_points += 1;
            result.skipped_conflicting_observations += track.len();
            continue;
        }
        for &observation in &track {
            if !owned_observations.insert(observation) {
                return Err(format!(
                    "COLMAP points3D contains observation ({},{}) in more than one point",
                    observation.0, observation.1
                ));
            }
        }
        result.retained_observations += track.len();
        result.tracks.push(track);
    }
    Ok(result)
}

/// Estimate the Sim(3) that maps one camera-centre set into another. This is
/// only used by the opt-in COLMAP-basin BA probe to put our reconstructed
/// landmarks in the injected pose frame; it follows the same Umeyama
/// convention as `scripts/score_umeyama_centers.py`.
fn umeyama_centres(
    source: &[Vector3<f64>],
    target: &[Vector3<f64>],
) -> Result<(f64, Matrix3<f64>, Vector3<f64>), String> {
    if source.len() != target.len() || source.len() < 3 {
        return Err(format!(
            "Sim(3) alignment needs at least 3 paired centres, got {} and {}",
            source.len(),
            target.len()
        ));
    }
    let n = source.len() as f64;
    let source_mean = source.iter().copied().sum::<Vector3<f64>>() / n;
    let target_mean = target.iter().copied().sum::<Vector3<f64>>() / n;
    let source_zero: Vec<Vector3<f64>> = source.iter().map(|p| *p - source_mean).collect();
    let target_zero: Vec<Vector3<f64>> = target.iter().map(|p| *p - target_mean).collect();
    let mut covariance = Matrix3::zeros();
    let mut source_variance = 0.0;
    for (src, dst) in source_zero.iter().zip(&target_zero) {
        covariance += dst * src.transpose();
        source_variance += src.norm_squared();
    }
    covariance /= n;
    source_variance /= n;
    if !source_variance.is_finite() || source_variance <= f64::EPSILON {
        return Err("Sim(3) source centres have zero variance".into());
    }
    let svd = covariance.svd(true, true);
    let u = svd.u.ok_or("Sim(3) SVD did not return U")?;
    let v_t = svd.v_t.ok_or("Sim(3) SVD did not return V^T")?;
    let mut correction = Matrix3::identity();
    if u.determinant() * v_t.determinant() < 0.0 {
        correction[(2, 2)] = -1.0;
    }
    let rotation = u * correction * v_t;
    let numerator = svd.singular_values[0] * correction[(0, 0)]
        + svd.singular_values[1] * correction[(1, 1)]
        + svd.singular_values[2] * correction[(2, 2)];
    let scale = numerator / source_variance;
    let translation = target_mean - scale * (rotation * source_mean);
    if !scale.is_finite() || scale <= 0.0 || !translation.iter().all(|v| v.is_finite()) {
        return Err("Sim(3) alignment produced a non-finite or non-positive scale".into());
    }
    Ok((scale, rotation, translation))
}

fn transform_point_by_sim3(
    point: Point3<f64>,
    scale: f64,
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
) -> Point3<f64> {
    Point3::from(scale * (rotation * point.coords) + translation)
}

fn mean_track_reprojection(
    camera: &Camera,
    tracks: &[visloc_rs::slam::SfmTrack],
    poses: &[Option<Pose>],
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for track in tracks {
        for &(image, _, observed) in &track.observations {
            let Some(Some(pose)) = poses.get(image) else {
                continue;
            };
            let Some(projected) = camera.project(&pose.transform_world_point(&track.position))
            else {
                continue;
            };
            let error = (projected - observed).norm();
            if error.is_finite() {
                sum += error;
                count += 1;
            }
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Replace an already-complete incremental result with an external COLMAP
/// pose basin and run one fixed-support BA solve. The mapper's track topology
/// and observations remain the support under test; only the world-frame point
/// coordinates are carried through the same Sim(3) used to align the two pose
/// sets. This function is called only by `--diagnose-ba-oracle-poses`.
fn run_oracle_pose_ba_probe(
    result: &mut visloc_rs::slam::IncrementalSfmResult,
    features: &[FeatureSet],
    image_names: &[String],
    camera: &Camera,
    config: &IncrementalSfmConfig,
    oracle_path: &Path,
) -> Result<(f64, f64, f64, visloc_rs::BaResult), String> {
    let oracle_by_stem = poses_from_colmap_images_txt(oracle_path)?;
    let mut source_centres = Vec::with_capacity(image_names.len());
    let mut target_centres = Vec::with_capacity(image_names.len());
    let mut oracle_poses = Vec::with_capacity(image_names.len());
    for (image, name) in image_names.iter().enumerate() {
        let Some(estimated) = result.poses.get(image).and_then(Option::as_ref) else {
            return Err(format!(
                "oracle BA probe requires every mapper pose; image {image} ({name}) is missing"
            ));
        };
        let stem = image_stem(name);
        let Some(oracle) = oracle_by_stem.get(stem) else {
            return Err(format!("oracle pose is missing image stem {stem:?}"));
        };
        source_centres.push(estimated.camera_center_world().coords);
        target_centres.push(oracle.camera_center_world().coords);
        oracle_poses.push(Some(oracle.clone()));
    }
    let (scale, rotation, translation) = umeyama_centres(&source_centres, &target_centres)?;
    let mut oracle_tracks = result.tracks.clone();
    for track in &mut oracle_tracks {
        track.position = transform_point_by_sim3(track.position, scale, &rotation, &translation);
    }
    let initial_reprojection = mean_track_reprojection(camera, &oracle_tracks, &oracle_poses);
    let mut probe_config = config.clone();
    // The caller wants the same ordinary BA objective/schedule as the mapper;
    // this probe is not allowed to trigger the separate optional polish pass.
    probe_config.final_ba_polish_iterations = 0;
    let (ba_result, refined_camera) = run_fixed_support_bundle_adjustment(
        camera,
        features,
        &mut oracle_tracks,
        &probe_config,
        &mut oracle_poses,
    )
    .map_err(|error| format!("oracle BA probe failed: {error:?}"))?;
    let final_reprojection = mean_track_reprojection(camera, &oracle_tracks, &oracle_poses);
    result.poses = oracle_poses;
    result.tracks = oracle_tracks;
    result.mean_reprojection_px = final_reprojection;
    result.ba_result = Some(ba_result.clone());
    result.refined_camera = refined_camera;
    Ok((scale, initial_reprojection, final_reprojection, ba_result))
}

/// Build rotations for the fixed-rotation BA diagnostic.  A source path is
/// parsed as COLMAP `images.txt`; `current`/`champion` use the completed
/// incremental rotations themselves.  External rotations are right-aligned
/// to the current world gauge using the lowest-index registered image, while
/// the current translations are deliberately retained.
fn fixed_rotation_targets(
    result: &visloc_rs::slam::IncrementalSfmResult,
    image_names: &[String],
    source: &str,
) -> Result<(Vec<Option<Pose>>, String), String> {
    let mut targets = vec![None; result.poses.len()];
    let registered: Vec<usize> = result
        .poses
        .iter()
        .enumerate()
        .filter_map(|(image, pose)| pose.as_ref().map(|_| image))
        .collect();
    if registered.is_empty() {
        return Err("fixed-rotation BA requires at least one registered pose".into());
    }
    if source == "current" || source == "champion" {
        for &image in &registered {
            targets[image] = result.poses[image].clone();
        }
        return Ok((targets, source.to_owned()));
    }

    let source_by_stem = poses_from_colmap_images_txt(Path::new(source))?;
    let anchor = registered[0];
    let anchor_stem = image_stem(&image_names[anchor]);
    let source_anchor = source_by_stem
        .get(anchor_stem)
        .ok_or_else(|| format!("fixed-rotation source is missing anchor stem {anchor_stem:?}"))?;
    let current_anchor = result.poses[anchor]
        .as_ref()
        .expect("registered pose exists");
    // For world-to-camera rotations, a global world-frame change acts on the
    // right.  Q_inv maps the source frame into the current mapper gauge.
    let q_inv =
        source_anchor.world_to_camera.rotation.inverse() * current_anchor.world_to_camera.rotation;
    for &image in &registered {
        let stem = image_stem(&image_names[image]);
        let source_pose = source_by_stem
            .get(stem)
            .ok_or_else(|| format!("fixed-rotation source is missing stem {stem:?}"))?;
        let current_pose = result.poses[image]
            .as_ref()
            .expect("registered pose exists");
        targets[image] = Some(Pose::from_world_to_camera(
            source_pose.world_to_camera.rotation * q_inv,
            current_pose.world_to_camera.translation,
        ));
    }
    Ok((targets, source.to_owned()))
}

/// Replace the incremental model's rotations by the requested diagnostic
/// targets and run translation/landmark-only BA on its exact support.
fn run_fixed_rotation_ba_probe(
    result: &mut visloc_rs::slam::IncrementalSfmResult,
    features: &[FeatureSet],
    image_names: &[String],
    camera: &Camera,
    config: &IncrementalSfmConfig,
    source: &str,
) -> Result<(String, usize, f64, f64, f64, visloc_rs::BaResult), String> {
    let (targets, label) = fixed_rotation_targets(result, image_names, source)?;
    let fixed_count = targets.iter().filter(|pose| pose.is_some()).count();
    let initial_reprojection = mean_track_reprojection(camera, &result.tracks, &targets);
    let mut probe_config = config.clone();
    // The fixed-rotation diagnostic is a pose/structure decomposition.  Keep
    // the intrinsics and separate final polish schedule out of the probe so
    // only the requested pose constraint changes the experiment.
    probe_config.refine_intrinsics = false;
    probe_config.final_ba_polish_iterations = 0;
    probe_config.geometry_weighted_ba = false;
    let (ba_result, refined_camera) = run_fixed_rotation_support_bundle_adjustment(
        camera,
        features,
        &mut result.tracks,
        &probe_config,
        &mut result.poses,
        &targets,
    )
    .map_err(|error| format!("fixed-rotation BA probe failed: {error:?}"))?;
    let final_reprojection = mean_track_reprojection(camera, &result.tracks, &result.poses);
    let mut max_rotation_delta = 0.0f64;
    for (after, target) in result.poses.iter().zip(&targets) {
        let (Some(after), Some(target)) = (after.as_ref(), target.as_ref()) else {
            continue;
        };
        let delta = (target.world_to_camera.rotation.inverse() * after.world_to_camera.rotation)
            .angle()
            .to_degrees();
        if delta.is_finite() {
            max_rotation_delta = max_rotation_delta.max(delta);
        }
    }
    result.mean_reprojection_px = final_reprojection;
    result.ba_result = Some(ba_result.clone());
    result.refined_camera = refined_camera;
    Ok((
        label,
        fixed_count,
        initial_reprojection,
        final_reprojection,
        max_rotation_delta,
        ba_result,
    ))
}

/// COLMAP-style guided matching (`FeaturePairsMatching`'s
/// `FindGuidedMatches`): given the pair's verified essential geometry,
/// rematch descriptors that the initial NN+ratio pass missed under an
/// epipolar constraint. For every not-yet-matched query descriptor the best
/// unused train descriptor is accepted only when **both** the Lowe ratio
/// (`0.9`, looser than the main pass) and the squared Sampson distance
/// (`guided_max_error_px`) pass — pure geometric admission without a ratio
/// gate is what produced M5's false-bridge failure, so this stays
/// deliberately conservative. Conflicts (two queries claiming one train)
/// resolve to the smaller descriptor distance, greedy by distance order.
///
/// When `pose_essential` is `Some`, that matrix is used for the Sampson gate
/// (pose-guided rematch after global); otherwise E is estimated from
/// `inlier_corrs` via normalized eight-point.
fn guided_epipolar_matches(
    camera: &Camera,
    features_i: &FeatureSet,
    features_j: &FeatureSet,
    initial: &[DescriptorMatch],
    inlier_corrs: &[TwoViewCorrespondence],
    max_error_px: f64,
    pose_essential: Option<Matrix3<f64>>,
    max_lowe_ratio: f64,
) -> Vec<DescriptorMatch> {
    let essential = if let Some(e) = pose_essential {
        e
    } else {
        let Some(e) = EssentialMatrixEstimator::estimate(
            &EightPointEssentialMatrixEstimator::default(),
            inlier_corrs,
            camera,
        ) else {
            return Vec::new();
        };
        e
    };
    let (fx, fy, _, _) = camera.intrinsics().unwrap_or((1.0, 1.0, 0.0, 0.0));
    let focal = 0.5 * (fx + fy);
    let max_sq_norm = (max_error_px / focal).powi(2);

    let normalize_all = |keypoints: &[Point2<f64>]| -> Vec<Option<[f64; 3]>> {
        keypoints
            .iter()
            .map(|p| camera.normalize_pixel(p).map(|n| [n.x, n.y, 1.0]))
            .collect()
    };
    let norm_i = normalize_all(&features_i.keypoints);
    let norm_j = normalize_all(&features_j.keypoints);
    let sampson_sq = |ni: &[f64; 3], nj: &[f64; 3]| -> Option<f64> {
        let e_ni = essential * nalgebra::Vector3::new(ni[0], ni[1], ni[2]);
        let et_nj = essential.transpose() * nalgebra::Vector3::new(nj[0], nj[1], nj[2]);
        let numerator = nalgebra::Vector3::new(nj[0], nj[1], nj[2])
            .dot(&e_ni)
            .powi(2);
        let denominator = e_ni.x * e_ni.x + e_ni.y * e_ni.y + et_nj.x * et_nj.x + et_nj.y * et_nj.y;
        if denominator < 1e-18 {
            None
        } else {
            Some(numerator / denominator)
        }
    };

    let mut used_query = vec![false; features_i.descriptors.len()];
    let mut used_train = vec![false; features_j.descriptors.len()];
    for m in initial {
        used_query[m.query_index] = true;
        used_train[m.train_index] = true;
    }

    // Descriptor-distance matrix over the full pair (one GEMM), rows =
    // queries, cols = trains.
    let n_q = features_i.descriptors.len();
    let n_t = features_j.descriptors.len();
    if n_q == 0 || n_t == 0 || features_i.descriptors[0].is_empty() {
        return Vec::new();
    }
    let dim = features_i.descriptors[0].len();
    let q = nalgebra::DMatrix::from_fn(n_q, dim, |a, k| features_i.descriptors[a][k] as f64);
    let t = nalgebra::DMatrix::from_fn(n_t, dim, |b, k| features_j.descriptors[b][k] as f64);
    let dist = &q * &t.transpose();

    struct Candidate {
        query: usize,
        train: usize,
        distance: f32,
    }
    let mut candidates: Vec<Candidate> = Vec::new();
    for qi in 0..n_q {
        if used_query[qi] || norm_i[qi].is_none() {
            continue;
        }
        let mut best: Option<(usize, f64)> = None;
        let mut second: f64 = f64::INFINITY;
        for tj in 0..n_t {
            if used_train[tj] {
                continue;
            }
            let d = ((dist[(qi, tj)]).max(0.0)).sqrt();
            if d < second {
                if d < best.map_or(f64::INFINITY, |(_, bd)| bd) {
                    second = best.map_or(f64::INFINITY, |(_, bd)| bd);
                    best = Some((tj, d));
                } else {
                    second = d;
                }
            }
        }
        let Some((tj, d)) = best else { continue };
        if d <= 0.0 || d >= second {
            continue;
        }
        if d / second > max_lowe_ratio {
            continue;
        }
        let Some(nj) = norm_j[tj] else { continue };
        let Some(sq) = sampson_sq(&norm_i[qi].unwrap(), &nj) else {
            continue;
        };
        if sq <= max_sq_norm {
            candidates.push(Candidate {
                query: qi,
                train: tj,
                distance: d as f32,
            });
        }
    }
    candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    let mut taken_train = used_train;
    let mut out = Vec::new();
    for c in candidates {
        if taken_train[c.train] {
            continue;
        }
        taken_train[c.train] = true;
        out.push(DescriptorMatch {
            query_index: c.query,
            train_index: c.train,
            distance: c.distance,
            second_best_distance: None,
            ratio: None,
            confidence: None,
        });
    }
    out
}

/// The model that COLMAP's `FindGuidedMatches` selects from a verified
/// `TwoViewGeometry`.  In particular, an `UNCALIBRATED` report means that F
/// won model selection (the report has no camera-specific calibration object),
/// while `CALIBRATED` selects E.  Homography-only configurations use H and
/// unresolved/multiple configurations are deliberately not guided.
#[derive(Debug, Clone, Copy)]
enum ColmapGuidedGeometry {
    Essential(Matrix3<f64>),
    Fundamental(Matrix3<f64>),
    Homography(Matrix3<f64>),
}

fn colmap_guided_geometry(report: &TwoViewGeometryReport) -> Option<ColmapGuidedGeometry> {
    match report.config {
        ConfigurationType::Calibrated => report.essential.map(ColmapGuidedGeometry::Essential),
        ConfigurationType::Uncalibrated => {
            report.fundamental.map(ColmapGuidedGeometry::Fundamental)
        }
        ConfigurationType::Planar
        | ConfigurationType::Panoramic
        | ConfigurationType::PlanarOrPanoramic => {
            report.homography.map(ColmapGuidedGeometry::Homography)
        }
        ConfigurationType::Undefined
        | ConfigurationType::Degenerate
        | ConfigurationType::Watermark
        | ConfigurationType::Multiple => None,
    }
}

fn colmap_guided_geometry_name(geometry: Option<ColmapGuidedGeometry>) -> &'static str {
    match geometry {
        Some(ColmapGuidedGeometry::Essential(_)) => "E",
        Some(ColmapGuidedGeometry::Fundamental(_)) => "F",
        Some(ColmapGuidedGeometry::Homography(_)) => "H",
        None => "none",
    }
}

/// Append-only guided matching with COLMAP's geometry and descriptor rules.
///
/// COLMAP first masks a full descriptor-distance matrix using the selected
/// E/F/H model, then applies the ordinary two-nearest-neighbour matcher in
/// both directions.  The production demo's historical guided path predates
/// this compatibility mode and remains untouched; it has a known dot-product
/// distance quirk and is kept for reproducibility.  This function fixes that
/// mismatch behind `--colmap-guided-matching`, while retaining every endpoint
/// used by the initial match set and never replacing an initial match.
fn colmap_guided_matches(
    camera: &Camera,
    features_i: &FeatureSet,
    features_j: &FeatureSet,
    initial: &[DescriptorMatch],
    report: &TwoViewGeometryReport,
    max_error_px: f64,
    max_lowe_ratio: f64,
    cross_check: bool,
) -> Vec<DescriptorMatch> {
    let geometry = colmap_guided_geometry(report);
    let Some(geometry) = geometry else {
        return Vec::new();
    };
    if !max_error_px.is_finite() || max_error_px < 0.0 {
        return Vec::new();
    }
    if !max_lowe_ratio.is_finite() || max_lowe_ratio <= 0.0 {
        return Vec::new();
    }

    let n_q = features_i.keypoints.len().min(features_i.descriptors.len());
    let n_t = features_j.keypoints.len().min(features_j.descriptors.len());
    if n_q == 0 || n_t == 0 {
        return Vec::new();
    }

    let essential_threshold =
        TwoViewGeometryOptions::for_camera(camera, max_error_px).essential_sampson_threshold;
    let essential_threshold_sq = essential_threshold * essential_threshold;
    let pixel_threshold_sq = max_error_px * max_error_px;
    let geometry_accepts = |query: usize, train: usize| -> bool {
        let Some(previous_xy) = features_i.keypoints.get(query).copied() else {
            return false;
        };
        let Some(current_xy) = features_j.keypoints.get(train).copied() else {
            return false;
        };
        let correspondence = TwoViewCorrespondence::new(previous_xy, current_xy);
        match geometry {
            ColmapGuidedGeometry::Essential(essential) => {
                normalized_essential_squared_sampson_error(&essential, &correspondence, camera)
                    .is_some_and(|error| error <= essential_threshold_sq)
            }
            ColmapGuidedGeometry::Fundamental(fundamental) => {
                let error = fundamental_squared_sampson_error(&fundamental, &correspondence);
                error.is_finite() && error <= pixel_threshold_sq
            }
            ColmapGuidedGeometry::Homography(homography) => {
                homography_squared_error(&homography, &correspondence)
                    .is_some_and(|error| error.is_finite() && error <= pixel_threshold_sq)
            }
        }
    };

    let mut used_query = vec![false; n_q];
    let mut used_train = vec![false; n_t];
    for descriptor_match in initial {
        if descriptor_match.query_index < n_q && descriptor_match.train_index < n_t {
            used_query[descriptor_match.query_index] = true;
            used_train[descriptor_match.train_index] = true;
        }
    }

    // COLMAP's SIFT descriptors are byte-equivalent vectors with an L2 norm
    // near 512, and its default max-distance is 0.7 in that scale.  Keep the
    // threshold fixed in this compatibility mode rather than silently tying
    // it to the main pass's Lowe ratio.
    const COLMAP_MAX_DESCRIPTOR_DISTANCE: f64 = 512.0 * 0.7;

    #[derive(Debug, Clone, Copy)]
    struct Candidate {
        query: usize,
        train: usize,
        distance_sq: f64,
    }

    let nearest = |query: usize, train_filter: Option<usize>| -> Option<(usize, f64, f64)> {
        let mut best: Option<(usize, f64)> = None;
        let mut second: Option<(usize, f64)> = None;
        for train in 0..n_t {
            if train_filter == Some(train) {
                continue;
            }
            if !geometry_accepts(query, train) {
                continue;
            }
            let distance_sq = descriptor_squared_distance(
                features_i.descriptors.get(query)?,
                features_j.descriptors.get(train)?,
            );
            if !distance_sq.is_finite() {
                continue;
            }
            if best.is_none_or(|(_, current)| distance_sq < current) {
                second = best;
                best = Some((train, distance_sq));
            } else if second.is_none_or(|(_, current)| distance_sq < current) {
                second = Some((train, distance_sq));
            }
        }
        let (train, best_distance_sq) = best?;
        Some((
            train,
            best_distance_sq,
            second.map_or(f64::INFINITY, |(_, distance_sq)| distance_sq),
        ))
    };

    let passes_ratio_and_distance = |distance_sq: f64, second_sq: f64| -> bool {
        if !distance_sq.is_finite() || distance_sq > COLMAP_MAX_DESCRIPTOR_DISTANCE.powi(2) {
            return false;
        }
        if second_sq.is_finite() {
            let distance = distance_sq.sqrt();
            let second = second_sq.sqrt();
            distance.is_finite() && second.is_finite() && distance < max_lowe_ratio * second
        } else {
            true
        }
    };

    let mut forward = Vec::new();
    for (query, used) in used_query.iter().enumerate().take(n_q) {
        if *used {
            continue;
        }
        let Some((train, distance_sq, second_sq)) = nearest(query, None) else {
            continue;
        };
        if passes_ratio_and_distance(distance_sq, second_sq) {
            forward.push(Candidate {
                query,
                train,
                distance_sq,
            });
        }
    }

    if cross_check {
        // This is the same mutual-NN test as COLMAP's second
        // `FindBestMatchesIndex` call.  A train descriptor's reverse nearest
        // query is found under the same geometry mask and ratio/distance bar.
        let mut reverse_best = vec![None; n_t];
        for (train, reverse_slot) in reverse_best.iter_mut().enumerate().take(n_t) {
            let mut best: Option<(usize, f64)> = None;
            let mut second: Option<(usize, f64)> = None;
            for query in 0..n_q {
                if !geometry_accepts(query, train) {
                    continue;
                }
                let distance_sq = descriptor_squared_distance(
                    features_i.descriptors.get(query).unwrap_or(&Vec::new()),
                    features_j.descriptors.get(train).unwrap_or(&Vec::new()),
                );
                if !distance_sq.is_finite() {
                    continue;
                }
                if best.is_none_or(|(_, current)| distance_sq < current) {
                    second = best;
                    best = Some((query, distance_sq));
                } else if second.is_none_or(|(_, current)| distance_sq < current) {
                    second = Some((query, distance_sq));
                }
            }
            if let Some((query, distance_sq)) = best {
                let second_sq = second.map_or(f64::INFINITY, |(_, value)| value);
                if passes_ratio_and_distance(distance_sq, second_sq) {
                    *reverse_slot = Some(query);
                }
            }
        }
        forward.retain(|candidate| reverse_best[candidate.train] == Some(candidate.query));
    }

    // FindGuidedMatches itself has no append-only conflict stage, but the
    // demo must not replace an existing match.  Resolve any residual conflicts
    // deterministically by distance, then physical row indices.
    forward.sort_by(|lhs, rhs| {
        lhs.distance_sq
            .total_cmp(&rhs.distance_sq)
            .then_with(|| lhs.query.cmp(&rhs.query))
            .then_with(|| lhs.train.cmp(&rhs.train))
    });
    let mut taken_train = used_train;
    let mut out = Vec::with_capacity(forward.len());
    for candidate in forward {
        if taken_train[candidate.train] {
            continue;
        }
        taken_train[candidate.train] = true;
        let distance = candidate.distance_sq.sqrt();
        out.push(DescriptorMatch {
            query_index: candidate.query,
            train_index: candidate.train,
            distance: distance as f32,
            second_best_distance: None,
            ratio: None,
            confidence: None,
        });
    }
    out.sort_by_key(|candidate| candidate.query_index);
    out
}

/// The M6 pair-*matching* A/B switch (`docs/colmap_port_plan.md`'s "M6
/// results"): which algorithm turns two images' descriptor sets into
/// candidate correspondences, **before** two-view geometric verification
/// ([`VerificationMode`]) ever runs. Orthogonal to [`VerificationMode`] and
/// [`PairSource`] — this only changes how a *given* candidate pair's raw
/// matches are produced, not which pairs are proposed or how they're
/// classified afterwards.
///
/// [`MatcherKind::Nn`] (default) is the pre-M6 nearest-neighbour + Lowe-ratio
/// path (`BruteForceMatcher`/`CrossCheckMatcher`), unchanged.
/// [`MatcherKind::LightGlue`] routes through
/// [`visloc_rs::vision::features::lightglue_onnx::LightGlueOnnxMatcher`] — a
/// learned, *joint* matcher that attends over both images' descriptors
/// together (as opposed to NN+ratio's independent per-descriptor nearest-
/// neighbour search) — motivated directly by M5's diagnosis
/// (`docs/colmap_port_plan.md`'s "M5 results"): ETH3D `courtyard`'s
/// cross-component bridge pairs carry real but very sparse correspondence
/// signal that a per-descriptor ratio test cannot safely extract from a
/// repeated-texture scene (M5's own "naive rescue" experiment showed a
/// *classifier-passing* false-bridge failure mode from over-relaxing the
/// NN+ratio matcher — the concrete evidence that the matcher itself, not
/// just its threshold, needed to change). Requires the `onnx-inference`
/// feature; `--matcher lightglue` without it is a hard runtime error (see
/// `parse_args`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatcherKind {
    /// Nearest-neighbour + Lowe-ratio test, optionally bidirectional
    /// cross-checked. Pre-M6 behaviour, unchanged.
    Nn,
    /// LightGlue (SuperPoint variant), run in-process via ONNX Runtime.
    LightGlue,
}

impl std::str::FromStr for MatcherKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "nn" => Ok(Self::Nn),
            "lightglue" => Ok(Self::LightGlue),
            other => Err(format!(
                "unknown --matcher {other:?} (expected nn|lightglue)"
            )),
        }
    }
}

/// Diagnostic ordering applied to the verified pair/match stream immediately
/// before the mapper consumes it.  The default preserves the historical
/// traversal exactly; the reverse variants only reorder existing entries and
/// never add, remove, or rewrite a correspondence index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum UnionTraversalOrder {
    #[default]
    Original,
    ReversePairs,
    ReverseMatches,
    ReverseBoth,
    /// Sort pair and correspondence traversal by a stable hash of the
    /// physical endpoint coordinates.  The seed makes independent replay
    /// orders possible without changing matching or verification.
    PhysicalHash(u64),
    /// Descending counterpart of [`Self::PhysicalHash`].
    PhysicalHashReverse(u64),
}

impl UnionTraversalOrder {
    fn as_string(self) -> String {
        match self {
            Self::Original => "original".to_string(),
            Self::ReversePairs => "reverse-pairs".to_string(),
            Self::ReverseMatches => "reverse-matches".to_string(),
            Self::ReverseBoth => "reverse-both".to_string(),
            Self::PhysicalHash(seed) => format!("physical-hash:{seed}"),
            Self::PhysicalHashReverse(seed) => format!("physical-hash-reverse:{seed}"),
        }
    }
}

impl std::str::FromStr for UnionTraversalOrder {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "original" => Ok(Self::Original),
            "reverse-pairs" => Ok(Self::ReversePairs),
            "reverse-matches" => Ok(Self::ReverseMatches),
            "reverse-both" => Ok(Self::ReverseBoth),
            other => {
                let (kind, raw_seed) = if let Some(seed) = other.strip_prefix("physical-hash:") {
                    ("physical-hash", seed)
                } else if let Some(seed) = other.strip_prefix("physical-hash-reverse:") {
                    ("physical-hash-reverse", seed)
                } else {
                    return Err(format!(
                        "unknown --union-traversal-order {other:?} (expected original|reverse-pairs|reverse-matches|reverse-both|physical-hash:SEED|physical-hash-reverse:SEED)"
                    ));
                };
                let seed = if let Some(hex) = raw_seed
                    .strip_prefix("0x")
                    .or_else(|| raw_seed.strip_prefix("0X"))
                {
                    u64::from_str_radix(hex, 16)
                } else {
                    raw_seed.parse::<u64>()
                }
                .map_err(|_| {
                    format!(
                        "invalid {kind} seed {raw_seed:?}; use an unsigned decimal or 0x-prefixed hexadecimal integer"
                    )
                })?;
                Ok(if kind == "physical-hash" {
                    Self::PhysicalHash(seed)
                } else {
                    Self::PhysicalHashReverse(seed)
                })
            }
        }
    }
}

#[derive(Debug)]
struct Args {
    /// `Files` (default): read precomputed `X Y SCORE D…` feature files from
    /// `features_dir`. `Sift`: run the pure-Rust SIFT frontend in-process on
    /// every image in `images_dir` (requires `--images-dir`; ignores
    /// `--features-dir`). See `visloc_vision::features::sift`.
    feature_extractor: FeatureExtractorKind,
    features_dir: PathBuf,
    images_dir: Option<PathBuf>,
    feature_suffix: String,
    image_suffix: String,
    sift_max_keypoints: usize,
    /// Enable SIFT affine shape adaptation (descriptor-side Baumberg).
    sift_affine: bool,
    /// Interest-point operator: `dog` (default) or `hessian-laplace`.
    sift_detector: String,
    /// Multi-anisotropy detection proposals (requires `--sift-affine`).
    sift_multi_anisotropy: bool,
    /// Domain-size pooling (DSP-SIFT / Dong & Soatto).
    sift_dsp: bool,
    /// DSP scale count used by explicit experimental overrides. The
    /// paper-standard `--sift-dsp` preset is 15 samples.
    sift_dsp_num_scales: usize,
    /// COLMAP-style L1-root (RootSIFT) descriptor normalization.
    sift_l1_root: bool,
    /// Cap orientations per keypoint (COLMAP default 2). `0` = unlimited.
    sift_max_orientations: usize,
    /// Use circularly smoothed strict-local-maximum orientation peaks.
    /// Default off preserves the historical threshold-bin selector.
    sift_standard_orientations: bool,
    /// COLMAP-style: keep larger-σ features when capping keypoints.
    sift_prefer_larger_scale: bool,
    /// Walk every octave before max-keypoint truncation.
    sift_full_pyramid: bool,
    /// DoG contrast / peak gate (COLMAP often uses `0.02/octave_resolution` ≈
    /// 0.0067). Default `0.02` = legacy Lowe-ish threshold.
    sift_contrast_threshold: f64,
    /// Spatial SIFT descriptor magnification in units of keypoint σ. The
    /// legacy descriptor uses 8.0; COLMAP/VLFeat-style sampling is ~3.0.
    sift_descriptor_magnification: f64,
    /// Use scale-adaptive Gaussian-pyramid gradients for SIFT descriptors.
    /// Default off preserves the historical direct-source gradient path.
    sift_scale_adaptive_gradients: bool,
    /// Use one cohesive VLFeat/COLMAP-compatible descriptor convention
    /// (octave gradient, m=3 support, histogram layout, normalization and
    /// 512-equivalent quantization). Default off preserves legacy SIFT.
    sift_vlfeat_compatible_descriptor: bool,
    /// Use the cohesive VLFeat/COLMAP DoG detector contract (first octave -1,
    /// subpixel localization, source edge test/orientations, large-scale cap).
    /// Default off preserves the historical detector.
    sift_vlfeat_compatible_detector: bool,
    /// Use COLMAP's vendored VLFeat bilinear orientation-bin accumulation.
    /// Requires `--sift-vlfeat-compatible-detector`; default off preserves the
    /// existing nearest-bin compatible-detector experiment.
    sift_vlfeat_bilinear_orientations: bool,
    /// Emit compatible-detector rows in COLMAP CPU SIFT source order. This is
    /// a narrow default-off ordering contract; the current detector already
    /// emits this order, so enabling it should be a no-op on ordinary inputs.
    sift_vlfeat_compatible_output_order: bool,
    /// Decode SIFT inputs with COLMAP's float32 RGB-to-gray rounding. Default
    /// off preserves the image crate's historical integer/floor conversion.
    sift_colmap_compatible_grayscale: bool,
    /// Detect/orient on COLMAP-rounded gray but describe fixed keypoints on
    /// legacy floor gray. Requires both VLFeat-compatible SIFT modes.
    sift_split_colmap_detector_grayscale: bool,
    /// Compute a second descriptor bank on the same keypoints and append its
    /// non-conflicting NN matches without replacing the primary matches.
    /// `None` keeps the single-bank legacy path.
    sift_append_descriptor_magnification: Option<f64>,
    /// Raise SIFT budget by [`Self::sift_extra_keypoints`] for these stems only
    /// (preserves the global 4096 contrast set on every other image).
    sift_extra_keypoints_stems: Vec<String>,
    /// Extra keypoints appended to `--sift-max-keypoints` for
    /// [`Self::sift_extra_keypoints_stems`] (fine-octave densification).
    sift_extra_keypoints: usize,
    /// Contrast threshold used only by the dense extra-keypoint extraction.
    /// `None` reuses [`Self::sift_contrast_threshold`] (legacy behavior).
    sift_extra_contrast_threshold: Option<f64>,
    /// Preserve NN matches computed on each image's primary SIFT prefix, then
    /// append only non-conflicting candidates that involve extra keypoints.
    /// Default off; this is meaningful when extra SIFT keypoints are enabled.
    sift_extra_matches_append_only: bool,
    /// Build conflict-free correspondence tracks and re-triangulate live
    /// points after each PnP registration.  This keeps the plain incremental
    /// schedule and is intentionally default-off.
    incremental_correspondence_triangulation: bool,
    /// Collapse orientation-expanded rows that share one physical
    /// `(x,y,scale)` locus before track construction.  Matching and geometric
    /// verification still see all descriptor alternatives; only the accepted
    /// pair stream is remapped/deduplicated.  Default off preserves legacy
    /// row-level behavior, and feature files without locus metadata are a
    /// deliberate no-op.
    orientation_locus_canonicalization: bool,
    /// Precomputed `(query_idx, train_idx)` correspondences per image pair
    /// (COLMAP `export_colmap_matches.py` format). Skips NN matching on the
    /// main verification pass; still runs the two-view verifier.
    import_matches_file: Option<PathBuf>,
    /// Bridge/supplement oracle: use imported raw matches when a pair is listed,
    /// otherwise fall back to NN+ratio on our features. Still runs the verifier.
    import_matches_supplement_file: Option<PathBuf>,
    /// Write loaded/extracted features to `DIR/{stem}_features.txt` (external-deep
    /// format) after the frontend pass. With `--export-features-only`, exit before
    /// matching (oracle tooling for spatial match transfer).
    export_features_dir: Option<PathBuf>,
    export_features_only: bool,
    /// Decode/extract/export SIFT sequentially, writing each feature file via
    /// atomic rename. Requires `--feature-extractor sift`,
    /// `--export-features-dir`, and `--export-features-only`; omitted keeps the
    /// historical in-memory batch extractor.
    sift_stream_export: bool,
    /// Resume an SIFT stream export only from per-image sidecars whose
    /// extractor configuration, source bytes, and both output hashes validate.
    /// Requires `--sift-stream-export`; invalid or missing sidecars are
    /// re-extracted and republished atomically.
    #[cfg_attr(not(feature = "image-io"), allow(dead_code))]
    sift_stream_resume: bool,
    /// COLMAP `two_view_geometries` oracle (`export_colmap_verified_pairs.py`):
    /// bypasses NN matching and verification; feeds inliers + config + E
    /// directly into the mapper.
    import_verified_pairs_file: Option<PathBuf>,
    /// Export the exact post-verification pair/match stream to a checksummed
    /// versioned snapshot.  Default None; does not alter reconstruction.
    export_verified_pairs_snapshot: Option<PathBuf>,
    /// Import a checksummed verified-pair snapshot and bypass matching and
    /// verification.  The loaded image/feature manifest and camera must match.
    import_verified_pairs_snapshot: Option<PathBuf>,
    /// Replay an imported snapshot while retaining keypoints and zero-sized
    /// descriptor rows only.  The original descriptor files are re-read one
    /// at a time after calibration to reproduce the exact feature manifest
    /// hash.  This is opt-in because descriptor-dependent diagnostics and
    /// alternate mapper modes are intentionally unavailable.
    snapshot_keypoints_only: bool,
    /// Write a verified-pair snapshot and exit before track building/mapping.
    /// This is the opt-in match-shard worker mode and requires
    /// `export_verified_pairs_snapshot`.
    export_verified_pairs_only: bool,
    /// Run one persistent, plan-driven match worker.  The worker loads the
    /// file-backed feature bank and matcher once, then writes one atomic
    /// snapshot per plan shard.  This is default-off and intentionally
    /// restricted to the frozen NN/full/plain incremental match path.
    persistent_match_worker_plan: Option<PathBuf>,
    /// Diagnostic-only coordinate replacement after a validated snapshot
    /// import.  The override directory must have the same image names, row
    /// counts, and descriptor bit patterns as the snapshot's base features;
    /// only keypoint x/y values are copied.  Default None.
    snapshot_coordinate_override_dir: Option<PathBuf>,
    /// Diagnostic only: after the ordinary mapper has produced its tracks,
    /// replace its poses with the poses parsed from this COLMAP `images.txt`,
    /// Sim(3)-align the mapper's points into that frame, and run one ordinary
    /// fixed-support BA solve. Omitted by default; never affects reconstruction.
    diagnose_ba_oracle_poses_file: Option<PathBuf>,
    /// Diagnostic only: after ordinary mapping, fix each registered pose's
    /// rotation and optimize translations/landmarks on the same support.
    /// `current` keeps the champion rotations; a path to COLMAP `images.txt`
    /// supplies rotations after deterministic gauge alignment.
    diagnose_fixed_rotation_ba: Option<String>,
    /// Diagnostic only: score a completed COLMAP `images.txt` against every
    /// correspondence in `--import-verified-pairs-file`, including a stable
    /// hash-held-out subset. Omitted by default and never affects mapping.
    diagnose_model_score_file: Option<PathBuf>,
    /// Seed incremental growth from a partial COLMAP `images.txt` model.
    /// Supplied poses are fixed during initial triangulation/PnP growth and
    /// released for the normal final BA gauge handling. Incremental-only.
    initial_poses_file: Option<PathBuf>,
    /// Diagnostic-only: import observation membership from a COLMAP
    /// `points3D.txt`, while ignoring its XYZ/color/error and camera poses.
    /// The sibling `images.txt` supplies IMAGE_ID/name and point2D-row
    /// validation. Requires the plain incremental mapper.
    diagnose_colmap_track_membership: Option<PathBuf>,
    /// Diagnostic-only pose-guided multi-model splitting of legacy union
    /// components after a complete initial reconstruction. Default off.
    pose_guided_track_splitting: bool,
    /// Require direct verified graph support from two existing hypothesis
    /// images for each observation added by pose-guided splitting. Default off.
    pose_guided_track_splitting_graph_support: bool,
    /// Run the deterministic Tarjan bridge-cut refinement before splitting.
    /// Requires pose-guided track splitting and is default-off.
    pose_guided_track_splitting_bridge_cuts: bool,
    /// Optional split-only reprojection gate; None reuses max_reproj.
    pose_guided_split_max_reproj: Option<f64>,
    /// Optional number of pose-guided split passes; None uses one pass.
    pose_guided_track_splitting_iterations: Option<usize>,
    /// Merge complementary posed split tracks across verified edges. Default
    /// off; requires pose-guided splitting.
    pose_guided_track_merging: bool,
    /// Optional reprojection gate for post-split merge fitting.
    pose_guided_merge_max_reproj: Option<f64>,
    out_colmap: PathBuf,
    /// Optional COLMAP text model directory whose PINHOLE camera is selected
    /// independently for each loaded image.  Omitted preserves the historical
    /// scalar `--width/--height/--fx/--fy/--cx/--cy` camera.
    input_colmap_calibration: Option<PathBuf>,
    camera: Camera,
    vocab_size: usize,
    retrieval_topk: usize,
    exhaustive: bool,
    /// Restrict candidate/imported pair endpoints to a unique numeric stem
    /// window. `None` preserves the historical all-candidate behavior.
    pair_stem_window: Option<u64>,
    match_ratio: f32,
    min_matches: usize,
    min_pnp_inliers: usize,
    /// Optional deterministic cap on accepted correspondences from each
    /// verified pair before the mapper consumes the stream.  Matching,
    /// verification, and an exported verified-pair snapshot remain complete;
    /// `None` preserves the historical mapper input exactly.
    max_mapper_matches_per_pair: Option<usize>,
    max_reproj: f64,
    next_image_policy: NextImagePolicy,
    final_ba: bool,
    /// Final-only track-length gate. The first guarded value is exactly 3;
    /// omitted keeps the historical registration and support unchanged.
    final_min_track_length: Option<usize>,
    seed_trials: usize,
    /// Optional diagnostic restriction to one seed pair (`I,J`).
    seed_pair: Option<(usize, usize)>,
    refine_intrinsics: bool,
    refine_distortion: bool,
    colmap_style: bool,
    /// Plain-growth final pass: iterative global BA + filter + re-triangulate
    /// (COLMAP final polish without colmap-style per-registration local BA).
    final_iterative_global_refinement: bool,
    /// Optional cap on follow-up global BA → complete → filter rounds in the
    /// iterative refinement schedule. `None` preserves the library default.
    global_ba_max_refinements: Option<usize>,
    /// After final refinement, give each missing image one bounded PnP attempt
    /// against the tightened structure. Default off.
    post_refinement_registration: bool,
    structureless_registration: bool,
    /// Raise above 128 to opt into the COLMAP-style confidence-based
    /// adaptive PnP RANSAC budget for large correspondence sets.
    pnp_max_iterations: usize,
    /// Optional Levenberg–Marquardt iteration cap for each bundle-adjustment
    /// solve. `None` keeps the library default (20); an explicit value is an
    /// opt-in convergence experiment and does not alter the normal path.
    ba_max_iterations: Option<usize>,
    /// Optional final/global BA Huber threshold in input pixels. `None`
    /// preserves the historical 3 px Huber configuration.
    ba_huber_delta: Option<f64>,
    /// Optional Schur-reduced BA linear solver. `None` preserves the dense
    /// historical backend; `sparse` is an explicit large-model experiment.
    ba_linear_solver: Option<LinearSolver>,
    /// Defer plain-growth periodic BA until this many cameras are registered;
    /// `0` preserves the historical `ba_every` schedule.
    periodic_ba_min_registered_images: usize,
    /// Optional final fixed-support pure-L2 BA polish iteration cap. `0` is the
    /// default no-op; support membership and fixed intrinsics are preserved.
    final_ba_polish_iterations: usize,
    /// Run a final fixed-support BA with pre-BA track-parallax information
    /// weights. Default off; support membership and registration are unchanged.
    geometry_weighted_ba: bool,
    /// Exclude weak, already-bad pre-BA landmark residual rows from BA.
    /// Default off; well-fitting weak points remain ordinary variables.
    freeze_ill_conditioned_landmarks: bool,
    /// Run a camera-fixed point-only BA before each global/periodic joint BA.
    /// `0` preserves the historical schedule exactly.
    landmark_ba_warm_start_iterations: usize,
    /// Minimum registered-camera count for the warm start; `0` applies it to
    /// every global/periodic BA.
    landmark_ba_warm_start_min_registered_images: usize,
    filter_images: bool,
    verification_mode: VerificationMode,
    /// COLMAP-style guided matching: after a pair verifies, rematch
    /// descriptors missed by the initial NN+ratio pass under the verified
    /// epipolar geometry, then re-verify. Off by default (byte-identical
    /// legacy behaviour when off).
    guided_matching: bool,
    /// Use COLMAP's E/F/H geometry selection and true descriptor-distance
    /// semantics for guided rematching.  This is an opt-in, append-only
    /// compatibility path and requires [`Self::guided_matching`].
    colmap_guided_matching: bool,
    /// Full verifier: COLMAP `multiple_models` — peel multiple two-view
    /// geometries per pair; keep the strongest Calibrated (else largest)
    /// sub-model's inliers. Off by default.
    multiple_models: bool,
    /// Override COLMAP `min_e_f_inlier_ratio` (default 0.95 when unset).
    min_e_f_inlier_ratio: Option<f64>,
    /// When Calibrated, keep E inliers even if F has more (COLMAP: max(E,F)).
    calibrated_prefer_essential: bool,
    /// For F-winning `Uncalibrated` pairs only, project F through the known
    /// intrinsics, re-score every candidate with the calibrated E threshold,
    /// and use the E_F inlier set when conservative support/cheirality guards
    /// pass. Default off; calibrated E winners and all other configurations
    /// are unchanged.
    refine_uncalibrated_f_to_essential: bool,
    /// In addition to the opt-in F→E refinement, drop an uncalibrated
    /// F-winning pair when that refinement fails the strict gate. This is a
    /// separate, default-off strategy: unlike the refinement flag it does
    /// not fall back to F matches for translation/track construction.
    strict_uncalibrated_f_to_essential: bool,
    /// For known-intrinsics F-winning pairs, use a robust/refit/rescored
    /// essential estimate as the primary track model when it passes the
    /// calibrated support and cheirality gates. Default off.
    calibrated_essential_primary: bool,
    /// Prefer essential-matrix inliers for global/hybrid *edges* when the full
    /// verifier estimated E (tracks / incremental still use the winning F/H
    /// set). Off by default.
    prefer_essential_inliers: bool,
    /// Like [`Self::prefer_essential_inliers`], but only on edges where at
    /// least one endpoint lacks a hybrid pose prior (free camera). Off by default.
    prefer_essential_free_endpoints: bool,
    /// Prefer E inliers only on edges incident to these image stems
    /// (comma-separated). Empty = unused.
    prefer_essential_stems: Vec<String>,
    /// With `--prefer-essential-stems`, require both endpoints in the set.
    prefer_essential_stem_clique: bool,
    /// Prefer E only on explicit index pairs `I-J,K-L,…`. Empty = unused.
    prefer_essential_pairs: Vec<(usize, usize)>,
    /// Drop selected stem/pair edges that lack strong E inliers (no F fallback).
    require_essential_selected_edges: bool,
    /// Drop edges incident to these stems unless strong E exists (isolation).
    require_essential_stems: Vec<String>,
    /// Min E inliers for `--require-essential-stems` (0 = min-matches).
    require_essential_min_e_inliers: usize,
    /// Re-match pairs incident to these stems at `--rematch-ratio`.
    rematch_stems: Vec<String>,
    /// Lowe ratio for `--rematch-stems` (default 0.9).
    rematch_ratio: f32,
    /// When rematching stems, skip mutual-NN cross-check (diagnose showed
    /// some 0297↔far pairs only densify without it). Default true = keep CC.
    rematch_cross_check: bool,
    /// Run COLMAP-style epipolar guided rematch only on `--rematch-stems`
    /// pairs (main pass stays unguided). Default false.
    rematch_guided: bool,
    /// After hybrid incremental priors are known, rematch `--rematch-stems`
    /// (or all non-prior images if stems empty) only against prior cameras —
    /// targets prior↔hub bridges rather than free–free densification.
    rematch_free_vs_priors: bool,
    /// Min E inliers for auto-`prefer-essential-pairs` from free↔prior rematch
    /// gains (`0` = prefer every E-gain pair).
    rematch_prefer_min_e_inliers: usize,
    /// Stems that need a higher E bar for auto prefer-E (comma-separated).
    rematch_prefer_strong_stems: Vec<String>,
    /// Min E for pairs incident to [`Self::rematch_prefer_strong_stems`]
    /// (default 50). Ignored when strong-stems empty.
    rematch_prefer_strong_min_e: usize,
    /// When free↔prior rematch gains E, also replace primary `matches` with
    /// essential inliers so tracks (not only view-graph edges) use E.
    rematch_tracks_use_essential: bool,
    /// Min essential chirality margin `(best-second)/best` to accept rematch
    /// E-gains (`0` = off).
    rematch_min_chirality_margin: f64,
    /// Reject rematch E-gains whose primary chirality disagrees with a
    /// triangulation anchor from two other prior↔free essentials.
    rematch_prior_anchor: bool,
    /// Min E inliers on anchor prior↔free pairs for [`Self::rematch_prior_anchor`].
    rematch_anchor_min_e_inliers: usize,
    /// Override COLMAP `min_e_f_inlier_ratio` for free↔prior rematch only
    /// (`None` = use [`Self::min_e_f_inlier_ratio`] / verifier default 0.95).
    rematch_min_e_f_inlier_ratio: Option<f64>,
    /// On free↔prior rematch, keep E inliers when Calibrated even if F has more.
    rematch_calibrated_prefer_essential: bool,
    /// Guide free↔prior rematch epipolar geometry from incremental pose priors
    /// plus free centres triangulated from prior rays (no GT). Default off.
    rematch_prior_ray_guided: bool,
    /// Min prior↔free rays to triangulate a free centre for prior-ray guide.
    rematch_prior_ray_min_rays: usize,
    /// Min E inliers on anchor prior↔free edge for prior-ray guide.
    rematch_prior_ray_min_e_inliers: usize,
    /// Override two-view verification on free↔prior rematch only (`None` = same as
    /// [`Self::verification_mode`]). `threshold-only` skips F/H model selection.
    rematch_verification_mode: Option<VerificationMode>,
    /// After the first hybrid global solve, rematch free↔prior again using
    /// essential matrices from the estimated absolute poses (pose-guided
    /// epipolar), accept E-gain pairs, then re-run global. Default off.
    rematch_pose_guided_after_global: bool,
    /// Optional COLMAP `images.txt` whose poses replace the estimated ones
    /// **only** for pose-guided rematch E (GT/oracle probe). Empty = use est.
    rematch_pose_guided_gt: Option<PathBuf>,
    /// Weight multiplier for edges built from essential inliers (default 1.0).
    essential_edge_weight_boost: f64,
    /// Full verifier: when E inliers clear `min_matches` *and* E/F inlier
    /// ratio ≥ [`Self::force_essential_min_ef_ratio`], use E as the primary
    /// match set (tracks+edges). Avoids forcing weak-E pairs that hurt more
    /// than F. Off by default.
    force_essential_matches: bool,
    /// Minimum E/F inlier ratio for `--force-essential-matches` (default 0.7).
    force_essential_min_ef_ratio: f64,
    /// Minimum absolute E inlier count for `--force-essential-matches`
    /// (default 0 = only `min_matches`). Raise to e.g. 100 for hub-like pairs.
    force_essential_min_e_inliers: usize,
    /// Only apply force-E on `Uncalibrated` pairs (F-won model selection).
    /// Calibrated pairs already passed E/F agreement. Default false.
    force_essential_uncalibrated_only: bool,
    /// After hybrid BA, re-PnP free cameras against prior-anchored tracks.
    repnp_free_from_priors: bool,
    /// Min prior-anchored corrs for re-PnP (0 = min-pnp-inliers).
    repnp_free_min_corrs: usize,
    /// Before hybrid global, PnP free cams into prior-only structure and pin
    /// successes as pose priors. Default off.
    repnp_seed_free_as_priors: bool,
    /// Hybrid: rewrite prior–prior edge R/t from the incremental pose priors.
    repair_prior_edges: bool,
    /// After pass-1 global, rewrite free-incident edges from solved poses and
    /// re-average. Default off.
    repair_free_edges_from_solved: bool,
    /// With repair: only edges antipodal to pass-1 pose bearings.
    repair_free_edges_only_flipped: bool,
    /// Limit repair to edges incident to these stems (comma-separated).
    repair_free_edges_stems: Vec<String>,
    /// Drop free-incident edges antipodal to pass-1 poses (not rewrite).
    drop_free_edges_antipodal: bool,
    /// Flip prior↔free edge chirality when multi-view prior rays agree better.
    prior_guided_free_chirality: bool,
    /// Flip prior↔free edge chirality using triangulated free centres from
    /// incremental pose priors (metric frame anchor).
    metric_prior_chirality_edges: bool,
    /// Min prior↔free rays to anchor a free centre for metric chirality.
    metric_prior_chirality_min_rays: usize,
    /// COLMAP `images.txt` GT poses for bearing-vs-GT diagnostic output.
    diagnose_bearing_gt: Option<PathBuf>,
    /// Limit `--diagnose-bearing-gt` to pairs incident to these stems (empty=all).
    diagnose_bearing_stems: Vec<String>,
    /// Oracle ceiling: flip edge chirality to match GT bearings at build time.
    gt_chirality_oracle: bool,
    /// GT poses for [`Self::gt_chirality_oracle`] (same file as diagnostic).
    gt_chirality_oracle_path: Option<PathBuf>,
    /// Reject rematch E-gains whose essential bearing vs GT exceeds this (deg).
    /// `0` = off. Requires `--rematch-gt-bearing-path` or `--diagnose-bearing-gt`.
    rematch_max_gt_bearing_deg: f64,
    /// COLMAP `images.txt` for [`Self::rematch_max_gt_bearing_deg`].
    rematch_gt_bearing_path: Option<PathBuf>,
    /// Tighter Sampson gate (px) for guided rematch only (`None` = 2.0 px).
    rematch_guided_max_error_px: Option<f64>,
    /// Lowe ratio for guided epipolar densify on rematch (`None` = 0.8).
    rematch_guided_lowe_ratio: Option<f64>,
    /// Reject rematch E-gains whose two-view config is not `Calibrated`.
    rematch_require_calibrated: bool,
    /// Reject rematch E-gains whose mean essential Sampson exceeds this
    /// (normalized coords; `0` = off).
    rematch_max_mean_sampson: f64,
    /// Hybrid: position-averaging scale from prior–prior metric length.
    metric_prior_scale: bool,
    /// `Incremental` (default): the existing grow-from-seed mapper.
    /// `Global`: GLOMAP-style — per-pair essential relative poses, rotation +
    /// position averaging, track triangulation, one joint BA
    /// (`visloc_slam::global_sfm::reconstruct_global_sfm`).
    /// `Hybrid`: incremental first, then global with those poses pinned as
    /// absolute priors for the leftover images.
    mapper: MapperKind,
    /// Global mapper only: harden essential cheirality (min tri-angle,
    /// ambiguity rejection). Default off = byte-identical legacy edges.
    chirality_harden: bool,
    /// Global mapper only: try this many high-degree rotation seeds and keep
    /// the best. `1` = legacy single-seed.
    rotation_seed_trials: usize,
    /// Global mapper: re-estimate edge translations under consensus rotations.
    refine_global_translations: bool,
    /// Global mapper: solve camera centres with one unknown scale per E edge
    /// instead of forcing unit displacement on every edge. Default off.
    global_independent_edge_scales: bool,
    /// Global mapper: keep ambiguous essentials as primary+alternate edges.
    multi_hypothesis_edges: bool,
    /// Global mapper: minimum essential inliers for a view-graph edge.
    min_edge_inliers: usize,
    /// Global mapper: drop edges whose median triangulation angle is below
    /// this (degrees). `0` disables.
    min_edge_parallax_deg: f64,
    /// Global mapper: down-weight chirality-ambiguous edges.
    weight_by_chirality_margin: bool,
    /// Hybrid mapper: drop incremental priors with thin track support or high
    /// local mean reprojection before global placement.
    hybrid_filter_priors: bool,
    /// Hybrid + `--hybrid-filter-priors`: minimum track observations.
    hybrid_prior_min_obs: usize,
    /// Hybrid + `--hybrid-filter-priors`: maximum per-image mean reprojection.
    hybrid_prior_max_reproj: f64,
    /// Hybrid mapper: clear pose priors whose image stem matches (comma-separated).
    /// A/B for surgically unpinning bent hubs (e.g. `DSC_0296`) without the
    /// quality filter's mass drop.
    hybrid_drop_prior_stems: Vec<String>,
    /// Hybrid: drop priors that disagree with free-centre probe.
    hybrid_drop_inconsistent_priors: bool,
    /// Incremental: reject PnP poses that flip vs two-view neighbours.
    verify_registration_two_view: bool,
    /// After ordinary PnP stalls, use a validated relative pose between
    /// numeric consecutive stems and the robust recent step scale. Default
    /// off; this is restricted to the plain incremental path.
    sequence_relative_pose_fallback: bool,
    /// Defer sequence fallback until ordinary post-refinement registration
    /// stalls, then resume ordinary PnP after each provisional pose.
    sequence_fallback_after_post: bool,
    /// Under sequence fallback, project a robust recent world-frame velocity
    /// onto the candidate translation direction. Default off keeps the median
    /// step-magnitude estimator.
    sequence_constant_velocity_scale: bool,
    /// Under sequence fallback, use the projected scale with only broad
    /// 0.25x..4x recent-median bounds. Default off preserves the strict
    /// projected policy and the historical median estimator.
    sequence_relaxed_constant_velocity_scale: bool,
    /// Under after-post sequence fallback, carry an accepted provisional
    /// baseline magnitude to the next consecutive fallback. Default off.
    sequence_fallback_carry_scale: bool,
    /// Hybrid mapper: pin incremental orientations only; centres from global
    /// bearing averaging (not incremental centres).
    hybrid_rotation_priors_only: bool,
    /// GLOMAP-style joint camera+point positioning from feature-track rays.
    joint_global_positioning: bool,
    /// Global/hybrid: keep only CALIBRATED (or MULTIPLE) two-view configs.
    calibrated_view_edges_only: bool,
    /// M2 A/B switch: which algorithm builds feature tracks from the verified
    /// pairs (`docs/colmap_port_plan.md`'s M2 milestone) — the legacy ad hoc
    /// union-find (default) or COLMAP's persistent `CorrespondenceGraph`.
    track_source: TrackSource,
    /// Process verified correspondences in descending retained geometric
    /// support and skip same-image-conflicting merges. Default off.
    confidence_ordered_tracks: bool,
    /// Opt-in per-correspondence normalized Sampson ordering for finite
    /// E-supported, calibrated pairs; other model configurations use the
    /// pair-level confidence fallback. Default off.
    geometric_confidence_tracks: bool,
    /// Canonicalize mapper track/observation traversal by physical feature
    /// keys instead of input keypoint indices. Default off.
    stable_track_order: bool,
    /// Prefer accepted correspondences with distinct third-view cycle support
    /// before pair/geometric confidence; deterministic and default off.
    cycle_supported_tracks: bool,
    /// Canonicalize each image's keypoint/descriptor row order by the same
    /// physical key before matching; imported match indices are remapped.
    canonical_feature_order: bool,
    /// Diagnostic-only ordering of the already-verified stream before legacy
    /// union-find traversal. `original` is the default/no-op.
    union_traversal_order: UnionTraversalOrder,
    /// Incremental: revisit same-image-conflicted tracks after final refinement
    /// with the guarded geometry-guided recovery pass. Default off.
    geometry_guided_conflict_recovery: bool,
    /// M3 A/B switch: which candidate-pair source feeds verification — flat
    /// VLAD top-K (default), a bounded local+VLAD union, a rig-aware temporal
    /// pyramid plus VLAD fill, or the hierarchical vocab-tree
    /// (`docs/colmap_port_plan.md`'s M3 milestone).
    pair_source: PairSource,
    /// Numeric-stem local overlap window for `--pair-source vlad-union`.
    /// This schedule is cheap and is evaluated before any pair matching.
    local_stem_window: Option<u64>,
    /// Interpret image names as `<camera-prefix>_<numeric-timestamp>` for the
    /// `vlad-union` local schedule.  This opt-in keeps temporal edges within
    /// each camera and adds only same-timestamp cross-camera rig edges.
    rig_local_grouping: bool,
    /// Maximum positional offset for the rig-aware temporal-pyramid
    /// candidate source.  Powers of two from 1 through this value are used;
    /// the default is 32.  This is deliberately a positional offset rather
    /// than a raw timestamp difference because ETH3D timestamps are in
    /// nanoseconds and are not consecutive integers.
    temporal_pyramid_max_offset: u64,
    /// Optional upper bound on generated candidate pairs.  Under
    /// `vlad-union`, local pairs have priority, then retrieval pairs by
    /// descending similarity and stable pair key.  `None` preserves the full
    /// generated set.
    candidate_budget: Option<usize>,
    /// Vocab-tree hierarchical-k-means branching factor (M3; ignored under
    /// `--pair-source vlad`). See `vocab_tree::hkm::HkmBuildOptions`.
    vocab_tree_branching: usize,
    /// Vocab-tree hierarchical-k-means depth (M3; ignored under
    /// `--pair-source vlad`).
    vocab_tree_depth: usize,
    /// Vocab-tree pair generator's `num_images` (top-N retrieved per query
    /// image before dedup) — COLMAP default 100
    /// (`VocabTreePairingOptions::num_images`). Ignored under
    /// `--pair-source vlad`.
    vocab_tree_num_images: usize,
    /// Import a validated, image-name-bound candidate manifest and bypass
    /// candidate generation.  Matching/verification still run for its pairs.
    candidate_manifest: Option<PathBuf>,
    /// Export the generated candidate manifest and exit before matching.
    export_candidate_manifest: Option<PathBuf>,
    /// M5 (`docs/colmap_port_plan.md`): run the opt-in rescue-bridging pass
    /// after initial verification (see the file header's step 4).
    rescue_bridging: bool,
    /// Rescue pass's relaxed Lowe ratio (looser than `--match-ratio`) — the
    /// M5 "matching relaxation" lever.
    rescue_match_ratio: f32,
    /// Rescue pass's minimum raw-match / verified-inlier floor. Deliberately
    /// independent of `--min-matches`: rescue candidates are, by
    /// construction, the pairs the main pass already couldn't reach, so this
    /// is the floor the M5 brief's "cheapest lever first" default should use
    /// (COLMAP's own `min_num_inliers` default, 15) rather than inheriting
    /// whatever (possibly stricter) floor the main pass used.
    rescue_min_matches: usize,
    /// Maximum number of cross-component candidate pairs the rescue pass will
    /// attempt (budget cap, `BridgeCandidateOptions::max_candidates`).
    rescue_max_candidates: usize,
    /// Whether the rescue pass's relaxed matcher also applies strict
    /// bidirectional cross-check (default `false`: mutual-NN + ratio only,
    /// per the M5 brief's "mutual-NN with Lowe ratio *instead of* ... strict
    /// cross-check").
    rescue_cross_check: bool,
    /// M5 diagnosis tool (`--diagnose-pair I,J`, repeatable): dump raw match
    /// counts and verification outcomes for specific `(i, j)` image-index
    /// pairs across a battery of matching profiles, then exit without
    /// running the reconstruction. Used to inspect the exact bridge
    /// candidates the M5 brief asks for (e.g. the boundary pair) by hand.
    diagnose_pairs: Vec<(usize, usize)>,
    /// Write one machine-readable row per diagnosis profile and image pair.
    /// Without [`Self::diagnose_pair_stems`], uses the normal candidate-pair
    /// source; pass `--exhaustive` to cover every pair.
    diagnose_pairs_csv: Option<PathBuf>,
    /// Comma-separated image stems to diagnose. When set, the CSV covers every
    /// pair incident to one of these stems, including pairs omitted by the
    /// normal retrieval candidate source. Requires `--diagnose-pairs-csv`.
    diagnose_pair_stems: Vec<String>,
    /// M6 (`docs/colmap_port_plan.md`): which algorithm produces raw
    /// descriptor matches for a candidate pair, before two-view verification
    /// — `nn` (default, pre-M6 NN+ratio behaviour) or `lightglue` (learned
    /// joint matcher, `onnx-inference`-gated). See [`MatcherKind`].
    matcher: MatcherKind,
    /// Path to the exported LightGlue ONNX graph (`--matcher lightglue`
    /// only; see `scripts/export_lightglue_onnx.py`). One graph per camera
    /// resolution — re-export for a different `--width`/`--height`. Only
    /// read from [`build_matcher`]'s `onnx-inference`-gated branch — the
    /// `#[allow(dead_code)]` covers the default (feature-off) build, where
    /// `--matcher lightglue` is rejected before this field would be read.
    #[cfg_attr(not(feature = "onnx-inference"), allow(dead_code))]
    lightglue_model: Option<PathBuf>,
    /// ONNX execution provider for `--matcher lightglue`: `auto` (CUDA then
    /// CPU), `cuda`, or `cpu`. Machines without an NVIDIA driver must use
    /// `cpu` — `auto` can hang in CUDA EP registration.
    #[cfg_attr(not(feature = "onnx-inference"), allow(dead_code))]
    onnx_backend: String,
    /// Cap keypoints per image fed to LightGlue (score-sorted prefix). `0` =
    /// use all. CPU ONNX at 4096×4096 is multi-minute per pair; 512–1024 is
    /// the practical courtyard A/B budget.
    #[cfg_attr(not(feature = "onnx-inference"), allow(dead_code))]
    lightglue_max_keypoints: usize,
}

/// Serialize the complete parsed command line in declaration order.  Keeping
/// this as `Debug` output is intentional: unlike a map-based representation it
/// has a fixed field order, and it automatically includes newly added flags
/// once they are added to [`Args`].  The snapshot is diagnostic only and never
/// participates in reconstruction decisions.
fn effective_config_snapshot(args: &Args) -> String {
    let orientation_cap = if args.sift_vlfeat_compatible_detector {
        if args.sift_max_orientations == 0 {
            "2 (compatible-mode default)".to_owned()
        } else {
            args.sift_max_orientations.min(4).to_string()
        }
    } else if args.sift_max_orientations == 0 {
        "unlimited (legacy mode)".to_owned()
    } else {
        args.sift_max_orientations.to_string()
    };
    let descriptor_magnification = if args.sift_vlfeat_compatible_descriptor {
        "3.0 (compatible-mode fixed)".to_owned()
    } else {
        args.sift_descriptor_magnification.to_string()
    };
    format!(
        "raw={args:?};effective_sift_orientation_cap={orientation_cap};\
         effective_sift_descriptor_magnification={descriptor_magnification}"
    )
}

/// Stable, dependency-free hash for the effective command-line snapshot.
/// `DefaultHasher` is deliberately avoided because its implementation is not
/// a public cross-version serialization contract.  FNV-1a is sufficient here:
/// this is a reproducibility label, not a cryptographic integrity check.
fn fnv1a64_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3u64);
    }
    hash
}

fn effective_config_hash(snapshot: &str) -> u64 {
    fnv1a64_bytes(snapshot.as_bytes())
}

/// Reorder only the already-verified traversal stream.  This deliberately
/// leaves image/keypoint indices and every correspondence value untouched so
/// an A/B isolates legacy union traversal rather than matching or geometric
/// verification.  The two-argument wrapper is retained for the old index-only
/// controls and for unit tests; the executable uses the feature-aware helper
/// below for physical hashing.
#[cfg(test)]
fn apply_union_traversal_order(pairwise: &mut [PairwiseMatches], order: UnionTraversalOrder) {
    apply_union_traversal_order_with_features(pairwise, order, &[]);
}

fn quantized_physical_coordinate(value: f64) -> i64 {
    let scaled = value * 1_000_000.0;
    if scaled.is_finite() {
        scaled.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64
    } else {
        i64::MIN
    }
}

fn physical_hash_mix(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Return a deterministic ordering key for one verified physical edge.  The
/// coordinate quantization makes the primary key independent of feature-row
/// order while retaining feature indices only as a final collision breaker.
fn physical_edge_order_key(
    pair: &PairwiseMatches,
    keypoint_i: usize,
    keypoint_j: usize,
    seed: u64,
    features: &[FeatureSet],
) -> (u64, usize, usize, usize, usize) {
    let (image_i, image_j, keypoint_i, keypoint_j) = if pair.image_i <= pair.image_j {
        (pair.image_i, pair.image_j, keypoint_i, keypoint_j)
    } else {
        (pair.image_j, pair.image_i, keypoint_j, keypoint_i)
    };
    let point_i = features
        .get(image_i)
        .and_then(|set| set.keypoints.get(keypoint_i));
    let point_j = features
        .get(image_j)
        .and_then(|set| set.keypoints.get(keypoint_j));
    let (x_i, y_i) = point_i
        .map(|point| {
            (
                quantized_physical_coordinate(point.x),
                quantized_physical_coordinate(point.y),
            )
        })
        .unwrap_or((i64::MIN, i64::MIN));
    let (x_j, y_j) = point_j
        .map(|point| {
            (
                quantized_physical_coordinate(point.x),
                quantized_physical_coordinate(point.y),
            )
        })
        .unwrap_or((i64::MIN, i64::MIN));
    let mut hash = 0xcbf29ce484222325u64 ^ seed;
    for value in [
        image_i as u64,
        image_j as u64,
        x_i as u64,
        y_i as u64,
        x_j as u64,
        y_j as u64,
    ] {
        hash = physical_hash_mix(hash, value);
    }
    (hash, image_i, image_j, keypoint_i, keypoint_j)
}

fn physical_pair_order_key(
    pair: &PairwiseMatches,
    seed: u64,
    features: &[FeatureSet],
) -> (u64, usize, usize) {
    pair.matches
        .iter()
        .map(|&(keypoint_i, keypoint_j)| {
            let edge = physical_edge_order_key(pair, keypoint_i, keypoint_j, seed, features);
            (edge.0, edge.1, edge.2)
        })
        .min()
        .unwrap_or_else(|| {
            let mut hash = 0xcbf29ce484222325u64 ^ seed;
            hash = physical_hash_mix(hash, pair.image_i as u64);
            hash = physical_hash_mix(hash, pair.image_j as u64);
            (
                hash,
                pair.image_i.min(pair.image_j),
                pair.image_i.max(pair.image_j),
            )
        })
}

/// Feature-aware variant of [`apply_union_traversal_order`].  Physical-hash
/// modes sort matches inside each pair and then pairs by their first physical
/// edge.  No correspondence is added, removed, or rewritten.
fn apply_union_traversal_order_with_features(
    pairwise: &mut [PairwiseMatches],
    order: UnionTraversalOrder,
    features: &[FeatureSet],
) {
    if matches!(
        order,
        UnionTraversalOrder::ReverseMatches | UnionTraversalOrder::ReverseBoth
    ) {
        for pair in pairwise.iter_mut() {
            pair.matches.reverse();
        }
    }
    if matches!(
        order,
        UnionTraversalOrder::ReversePairs | UnionTraversalOrder::ReverseBoth
    ) {
        pairwise.reverse();
    }
    let (seed, descending) = match order {
        UnionTraversalOrder::PhysicalHash(seed) => (seed, false),
        UnionTraversalOrder::PhysicalHashReverse(seed) => (seed, true),
        _ => return,
    };
    for pair in pairwise.iter_mut() {
        let mut keyed_matches: Vec<_> = pair
            .matches
            .iter()
            .copied()
            .map(|(keypoint_i, keypoint_j)| {
                (
                    physical_edge_order_key(pair, keypoint_i, keypoint_j, seed, features),
                    (keypoint_i, keypoint_j),
                )
            })
            .collect();
        keyed_matches.sort_unstable_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        pair.matches = keyed_matches.into_iter().map(|(_, edge)| edge).collect();
    }
    pairwise.sort_unstable_by_key(|pair| physical_pair_order_key(pair, seed, features));
    if descending {
        for pair in pairwise.iter_mut() {
            pair.matches.reverse();
        }
        pairwise.reverse();
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MapperMatchCapStats {
    pairs_capped: usize,
    matches_before: usize,
    matches_after: usize,
    essential_before: usize,
    essential_after: usize,
}

/// Bound the correspondence stream consumed by the mapper while preserving
/// the complete verified stream for diagnostics and snapshot export.  The
/// verifier's deterministic inlier order is retained (rather than choosing
/// by a post-hoc score), so this option cannot introduce a second geometry
/// ranking policy.  `None`/the caller's omission is the historical path.
fn cap_mapper_pair_matches(pairwise: &mut [PairwiseMatches], limit: usize) -> MapperMatchCapStats {
    let mut stats = MapperMatchCapStats::default();
    for pair in pairwise {
        stats.matches_before += pair.matches.len();
        stats.essential_before += pair.essential_matches.as_ref().map_or(0, Vec::len);
        let pair_matches_capped = pair.matches.len() > limit;
        let essential_capped = pair
            .essential_matches
            .as_ref()
            .is_some_and(|matches| matches.len() > limit);
        if pair_matches_capped || essential_capped {
            stats.pairs_capped += 1;
        }
        pair.matches.truncate(limit);
        if let Some(matches) = pair.essential_matches.as_mut() {
            matches.truncate(limit);
        }
        stats.matches_after += pair.matches.len();
        stats.essential_after += pair.essential_matches.as_ref().map_or(0, Vec::len);
    }
    stats
}

/// Stable FNV-1a hash of the multiset of verified `(image, keypoint)` edges.
/// Pair direction and traversal order are normalized, while duplicate edges
/// remain duplicated in the sorted stream.  It is a diagnostic integrity label
/// rather than a cryptographic digest.
fn unordered_pairwise_edge_hash(pairwise: &[PairwiseMatches]) -> u64 {
    let mut edges = Vec::new();
    for pair in pairwise {
        let (image_i, image_j, swapped) = if pair.image_i <= pair.image_j {
            (pair.image_i, pair.image_j, false)
        } else {
            (pair.image_j, pair.image_i, true)
        };
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let (keypoint_i, keypoint_j) = if swapped {
                (keypoint_j, keypoint_i)
            } else {
                (keypoint_i, keypoint_j)
            };
            edges.push((image_i, image_j, keypoint_i, keypoint_j));
        }
    }
    edges.sort_unstable();
    let mut hash = 0xcbf29ce484222325u64;
    for (image_i, image_j, keypoint_i, keypoint_j) in edges {
        for value in [image_i, image_j, keypoint_i, keypoint_j] {
            for byte in (value as u64).to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3u64);
            }
        }
    }
    hash
}

/// Parse `--track-source`'s value into the M2 [`TrackSource`] A/B switch.
/// `TrackSource` lives in `visloc-slam` and has no `FromStr` of its own (it's
/// a plain engine config knob, not a CLI type), so this demo owns the string
/// mapping.
fn parse_track_source(s: &str) -> Result<TrackSource, String> {
    match s {
        "union-find" => Ok(TrackSource::UnionFind),
        "graph" => Ok(TrackSource::CorrespondenceGraph),
        other => Err(format!(
            "unknown --track-source {other:?} (expected union-find|graph)"
        )),
    }
}

fn parse_diagnose_stems(raw: &str) -> Result<Vec<String>, String> {
    let stems: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|stem| !stem.is_empty())
        .map(str::to_owned)
        .collect();
    if stems.is_empty() {
        return Err("--diagnose-pair-stems requires at least one non-empty stem".into());
    }
    Ok(stems)
}

fn image_stem(name: &str) -> &str {
    Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(name)
}

/// Camera assignment retained alongside the validated, index-aligned rig.
/// The native camera rows are kept for the multi-camera COLMAP exporter while
/// the rig's first camera is used as the internal ray convention.
#[derive(Debug, Clone)]
struct LoadedPerImageCalibration {
    rig: PerImageCameras,
    native_cameras: Vec<Camera>,
}

/// Parse the `IMAGE_ID ... CAMERA_ID NAME` headers from COLMAP text
/// `images.txt`.  The following POINTS2D line is skipped, so a point record
/// whose first token happens to be an integer cannot be mistaken for an image.
fn parse_colmap_image_camera_assignments(contents: &str) -> Result<HashMap<String, u64>, String> {
    let mut assignments = HashMap::new();
    let mut lines = contents.lines();
    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 10
            || tokens[0].parse::<u64>().is_err()
            || tokens[8].parse::<u64>().is_err()
            || tokens[9].parse::<f64>().is_ok()
        {
            // A valid image header has ten fields.  Invalid non-header lines
            // are ignored for compatibility with COLMAP comments/exports.
            continue;
        }
        let camera_id = tokens[8]
            .parse::<u64>()
            .map_err(|error| format!("invalid CAMERA_ID {:?} in images.txt: {error}", tokens[8]))?;
        let name = tokens[9..].join(" ");
        if name.is_empty() {
            return Err("images.txt contains an image header with an empty NAME".into());
        }
        if assignments.insert(name.clone(), camera_id).is_some() {
            return Err(format!("duplicate image NAME {name:?} in images.txt"));
        }
        // COLMAP always writes the points row, including when it is empty.
        let _ = lines.next();
    }
    if assignments.is_empty() {
        return Err("images.txt contains no usable image headers".into());
    }
    Ok(assignments)
}

fn calibration_name_candidates(name: &str) -> Vec<String> {
    let path = Path::new(name);
    let mut candidates = vec![name.to_owned()];
    if let Some(base) = path.file_name().and_then(|value| value.to_str()) {
        if !candidates.iter().any(|candidate| candidate == base) {
            candidates.push(base.to_owned());
        }
    }
    let stem = image_stem(name);
    if !candidates
        .iter()
        .any(|candidate| image_stem(candidate) == stem)
    {
        candidates.push(stem.to_owned());
    }
    candidates
}

/// Resolve each loaded image name to the camera id declared by COLMAP and
/// validate the supported PINHOLE parameter contract.  Resolution accepts an
/// exact path, basename, or unique stem, which covers feature files that use a
/// different image extension while still rejecting ambiguous mappings.
fn resolve_input_colmap_calibration(
    model_dir: &Path,
    image_names: &[String],
) -> Result<LoadedPerImageCalibration, String> {
    let cameras_path = model_dir.join("cameras.txt");
    let images_path = model_dir.join("images.txt");
    let camera_text = std::fs::read_to_string(&cameras_path)
        .map_err(|error| format!("cannot read calibration {cameras_path:?}: {error}"))?;
    let image_text = std::fs::read_to_string(&images_path)
        .map_err(|error| format!("cannot read calibration {images_path:?}: {error}"))?;
    let parsed_cameras = visloc_io::colmap::parse_cameras_txt(&camera_text)
        .map_err(|error| format!("cannot parse calibration cameras.txt: {error}"))?;
    let mut cameras_by_id = HashMap::new();
    for camera in parsed_cameras {
        if cameras_by_id.insert(camera.id, camera).is_some() {
            return Err("cameras.txt contains duplicate CAMERA_ID".into());
        }
    }
    let assignments = parse_colmap_image_camera_assignments(&image_text)?;
    let mut by_basename: HashMap<String, Vec<(String, u64)>> = HashMap::new();
    let mut by_stem: HashMap<String, Vec<(String, u64)>> = HashMap::new();
    for (name, camera_id) in &assignments {
        if let Some(base) = Path::new(name).file_name().and_then(|value| value.to_str()) {
            by_basename
                .entry(base.to_owned())
                .or_default()
                .push((name.clone(), *camera_id));
        }
        by_stem
            .entry(image_stem(name).to_owned())
            .or_default()
            .push((name.clone(), *camera_id));
    }
    let mut native_cameras = Vec::with_capacity(image_names.len());
    for name in image_names {
        let mut resolved = None;
        for candidate in calibration_name_candidates(name) {
            if let Some(&camera_id) = assignments.get(&candidate) {
                resolved = Some((candidate, camera_id));
                break;
            }
            if let Some(entries) = by_basename.get(&candidate) {
                if entries.len() > 1 {
                    return Err(format!(
                        "image {name:?} matches ambiguous calibration basenames {:?}",
                        entries.iter().map(|(entry, _)| entry).collect::<Vec<_>>()
                    ));
                }
                if let Some((entry, camera_id)) = entries.first() {
                    resolved = Some((entry.clone(), *camera_id));
                    break;
                }
            }
            if let Some(entries) = by_stem.get(image_stem(&candidate)) {
                if entries.len() > 1 {
                    return Err(format!(
                        "image {name:?} matches ambiguous calibration stems {:?}",
                        entries.iter().map(|(entry, _)| entry).collect::<Vec<_>>()
                    ));
                }
                if let Some((entry, camera_id)) = entries.first() {
                    resolved = Some((entry.clone(), *camera_id));
                    break;
                }
            }
        }
        let Some((resolved_name, camera_id)) = resolved else {
            return Err(format!(
                "calibration images.txt has no camera assignment for loaded image {name:?}"
            ));
        };
        let camera = cameras_by_id.get(&camera_id).ok_or_else(|| {
            format!("calibration image {resolved_name:?} refers to missing CAMERA_ID {camera_id}")
        })?;
        if camera.model != CameraModel::Pinhole {
            return Err(format!(
                "calibration CAMERA_ID {camera_id} uses {:?}; only PINHOLE is supported",
                camera.model
            ));
        }
        if camera.params.len() != 4 {
            return Err(format!(
                "calibration CAMERA_ID {camera_id} has {} parameters; PINHOLE requires 4",
                camera.params.len()
            ));
        }
        native_cameras.push(camera.clone());
    }
    let rig = PerImageCameras::new(native_cameras.clone()).map_err(|error| error.to_string())?;
    Ok(LoadedPerImageCalibration {
        rig,
        native_cameras,
    })
}

/// Resolve and validate a per-image calibration for an already-loaded feature
/// set.  The feature-file path keeps this wrapper so it can retain the exact
/// historical validation order; streaming extraction uses the resolver above
/// and validates each decoded image before writing its result.
fn load_input_colmap_calibration(
    model_dir: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    image_dir: Option<&Path>,
) -> Result<LoadedPerImageCalibration, String> {
    let loaded = resolve_input_colmap_calibration(model_dir, image_names)?;
    loaded
        .rig
        .validate_features(features)
        .map_err(|error| format!("calibration feature validation failed: {error}"))?;
    validate_calibration_image_dimensions(&loaded.rig, image_names, image_dir)?;
    Ok(loaded)
}

#[cfg(feature = "image-io")]
fn validate_calibration_image_dimensions(
    rig: &PerImageCameras,
    image_names: &[String],
    image_dir: Option<&Path>,
) -> Result<(), String> {
    let Some(image_dir) = image_dir else {
        return Ok(());
    };
    let mut dimensions = Vec::with_capacity(image_names.len());
    for name in image_names {
        let direct = image_dir.join(name);
        let path = if direct.is_file() {
            direct
        } else {
            let stem = image_stem(name);
            let mut matches = std::fs::read_dir(image_dir)
                .map_err(|error| format!("cannot scan image directory {image_dir:?}: {error}"))?
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .filter(|path| {
                    path.file_stem()
                        .and_then(|value| value.to_str())
                        .is_some_and(|candidate| candidate == stem)
                });
            let Some(path) = matches.next() else {
                return Err(format!(
                    "cannot find source image {name:?} under {image_dir:?} for calibration dimension validation"
                ));
            };
            if matches.next().is_some() {
                return Err(format!(
                    "multiple source images match {name:?} under {image_dir:?}"
                ));
            }
            path
        };
        let image = visloc_io::images::read_common_image(&path)
            .map_err(|error| format!("cannot decode source image {path:?}: {error}"))?;
        dimensions.push((image.width() as u32, image.height() as u32));
    }
    rig.validate_image_dimensions(&dimensions)
        .map_err(|error| error.to_string())
}

#[cfg(not(feature = "image-io"))]
fn validate_calibration_image_dimensions(
    _rig: &PerImageCameras,
    _image_names: &[String],
    image_dir: Option<&Path>,
) -> Result<(), String> {
    if image_dir.is_some() {
        return Err(
            "--input-colmap-calibration image-dimension validation requires the image-io feature"
                .into(),
        );
    }
    Ok(())
}

/// Parse the trailing decimal run from an image stem.  A pair-window run is
/// intentionally strict: silently falling back to lexical order would make a
/// supposedly sequence-local reconstruction depend on an unrelated filename
/// convention.
fn trailing_numeric_stem(name: &str) -> Result<u64, String> {
    let stem = image_stem(name);
    let bytes = stem.as_bytes();
    let mut start = bytes.len();
    while start > 0 && bytes[start - 1].is_ascii_digit() {
        start -= 1;
    }
    if start == bytes.len() {
        return Err(format!(
            "image {name:?} has no trailing numeric stem (expected e.g. DSC_0001)"
        ));
    }
    stem[start..].parse::<u64>().map_err(|error| {
        format!(
            "image {name:?} has an invalid trailing numeric stem {:?}: {error}",
            &stem[start..]
        )
    })
}

/// Validate and return the numeric stem for every loaded image.  Duplicate
/// numeric suffixes are rejected because they would make the window relation
/// ambiguous even when the lexical filenames differ.
fn numeric_stem_values(image_names: &[String]) -> Result<Vec<u64>, String> {
    let mut seen: HashMap<u64, (usize, String)> = HashMap::new();
    let mut values = Vec::with_capacity(image_names.len());
    for (index, name) in image_names.iter().enumerate() {
        let value = trailing_numeric_stem(name)?;
        if let Some((other_index, other_name)) = seen.insert(value, (index, name.clone())) {
            return Err(format!(
                "duplicate numeric image stem {value} in images {other_index} ({other_name:?}) and {index} ({name:?})"
            ));
        }
        values.push(value);
    }
    Ok(values)
}

/// Return the camera-prefix and trailing timestamp from a rig image name.
///
/// Rig-aware local grouping is deliberately opt-in because a flat image set
/// does not otherwise have a reliable camera-name convention.  The accepted
/// form is `<prefix>_<decimal-timestamp>` (with the normal image extension
/// still present in `name`); the prefix is retained verbatim and is compared
/// case-sensitively.  A duplicate `(prefix, timestamp)` is rejected, while
/// the same timestamp across distinct prefixes is the expected stereo/rig
/// case.
fn rig_camera_timestamp(name: &str) -> Result<(String, u64), String> {
    let stem = image_stem(name);
    let Some(separator) = stem.rfind('_') else {
        return Err(format!(
            "rig-local grouping requires image {name:?} to use <camera-prefix>_<numeric-timestamp>"
        ));
    };
    let prefix = &stem[..separator];
    let timestamp = &stem[separator + 1..];
    if prefix.is_empty() || timestamp.is_empty() {
        return Err(format!(
            "rig-local grouping requires image {name:?} to use <camera-prefix>_<numeric-timestamp>"
        ));
    }
    let timestamp = timestamp.parse::<u64>().map_err(|error| {
        format!(
            "rig-local grouping image {name:?} has invalid numeric timestamp {timestamp:?}: {error}"
        )
    })?;
    Ok((prefix.to_owned(), timestamp))
}

/// Build deterministic camera-aware local edges for a multi-camera rig.
///
/// Within each camera prefix, timestamps within `window` are connected.  At
/// each timestamp, every pair of distinct camera prefixes is connected once;
/// this is bounded by the number of camera pairs at that instant and does not
/// create the quadratic cross-product between neighbouring timestamps.  The
/// returned pairs use canonical image indices and are sorted by pair key.
fn rig_local_pairs(image_names: &[String], window: u64) -> Result<Vec<(usize, usize)>, String> {
    if window == 0 {
        return Err("--local-stem-window must be at least 1".into());
    }
    let mut by_camera = BTreeMap::<String, Vec<(u64, usize)>>::new();
    let mut by_timestamp = BTreeMap::<u64, Vec<(String, usize)>>::new();
    let mut seen = HashSet::<(String, u64)>::new();
    for (index, name) in image_names.iter().enumerate() {
        let (camera, timestamp) = rig_camera_timestamp(name)?;
        if !seen.insert((camera.clone(), timestamp)) {
            return Err(format!(
                "rig-local grouping repeats timestamp {timestamp} for camera prefix {camera:?}"
            ));
        }
        by_camera
            .entry(camera.clone())
            .or_default()
            .push((timestamp, index));
        by_timestamp
            .entry(timestamp)
            .or_default()
            .push((camera, index));
    }

    let mut pairs = HashSet::<(usize, usize)>::new();
    for entries in by_camera.values_mut() {
        entries.sort_unstable_by_key(|&(timestamp, index)| (timestamp, index));
        let mut first = 0usize;
        for right in 0..entries.len() {
            let right_timestamp = entries[right].0;
            while first < right && right_timestamp.abs_diff(entries[first].0) > window {
                first += 1;
            }
            for left in first..right {
                let pair = (
                    entries[left].1.min(entries[right].1),
                    entries[left].1.max(entries[right].1),
                );
                pairs.insert(pair);
            }
        }
    }
    for entries in by_timestamp.values_mut() {
        entries.sort_unstable();
        for left in 0..entries.len() {
            for right in left + 1..entries.len() {
                // Duplicate `(camera,timestamp)` values were rejected above,
                // so every pair here is a distinct-camera rig edge.
                let pair = (
                    entries[left].1.min(entries[right].1),
                    entries[left].1.max(entries[right].1),
                );
                pairs.insert(pair);
            }
        }
    }
    let mut pairs: Vec<_> = pairs.into_iter().collect();
    pairs.sort_unstable();
    Ok(pairs)
}

/// Build the two deterministic rig-aware components of the temporal-pyramid
/// schedule.  Temporal offsets are positions in a camera's timestamp-sorted
/// sequence, not differences between the nanosecond timestamp values.  This
/// distinction matters for ETH3D, where captures are not numbered by a dense
/// integer frame counter.  Same-timestamp cross-camera pairs are returned
/// separately so the caller can give them a stable priority after temporal
/// edges and before VLAD fill edges.
fn rig_temporal_pyramid_pairs(
    image_names: &[String],
    max_offset: u64,
) -> Result<(Vec<(usize, usize)>, Vec<(usize, usize)>), String> {
    if max_offset == 0 {
        return Err("--temporal-pyramid-max-offset must be at least 1".into());
    }
    let mut by_camera = BTreeMap::<String, Vec<(u64, usize)>>::new();
    let mut by_timestamp = BTreeMap::<u64, Vec<(String, usize)>>::new();
    let mut seen = HashSet::<(String, u64)>::new();
    for (index, name) in image_names.iter().enumerate() {
        let (camera, timestamp) = rig_camera_timestamp(name)?;
        if !seen.insert((camera.clone(), timestamp)) {
            return Err(format!(
                "temporal-pyramid grouping repeats timestamp {timestamp} for camera prefix {camera:?}"
            ));
        }
        by_camera
            .entry(camera.clone())
            .or_default()
            .push((timestamp, index));
        by_timestamp
            .entry(timestamp)
            .or_default()
            .push((camera, index));
    }

    // Generate adjacent edges first, then progressively longer pyramid
    // levels.  That gives a bounded budget the most local support while
    // retaining every level when no budget is requested.
    let mut offsets = Vec::new();
    let mut offset = 1u64;
    loop {
        offsets.push(offset as usize);
        if offset > max_offset / 2 {
            break;
        }
        offset *= 2;
    }
    let mut temporal = Vec::new();
    let mut temporal_seen = HashSet::<(usize, usize)>::new();
    for &offset in &offsets {
        for entries in by_camera.values() {
            for left in 0..entries.len().saturating_sub(offset) {
                let right = left + offset;
                let pair = (
                    entries[left].1.min(entries[right].1),
                    entries[left].1.max(entries[right].1),
                );
                if temporal_seen.insert(pair) {
                    temporal.push(pair);
                }
            }
        }
    }

    let mut cross_camera = Vec::new();
    let mut cross_seen = HashSet::<(usize, usize)>::new();
    for entries in by_timestamp.values_mut() {
        entries.sort_unstable();
        for left in 0..entries.len() {
            for right in left + 1..entries.len() {
                // A duplicate `(camera,timestamp)` was rejected above, so
                // every edge here is between distinct camera prefixes.
                let pair = (
                    entries[left].1.min(entries[right].1),
                    entries[left].1.max(entries[right].1),
                );
                if cross_seen.insert(pair) {
                    cross_camera.push(pair);
                }
            }
        }
    }
    Ok((temporal, cross_camera))
}

fn temporal_pyramid_offsets_string(max_offset: u64) -> String {
    let mut values = Vec::new();
    let mut offset = 1u64;
    loop {
        values.push(offset.to_string());
        if offset > max_offset / 2 {
            break;
        }
        offset *= 2;
    }
    values.join(",")
}

/// Candidate pairs from a rig-aware temporal pyramid plus a deterministic
/// VLAD fill.  Temporal edges are selected first, same-timestamp rig edges
/// second, and retrieval-only edges by descending VLAD score last.  The
/// final stream is sorted by canonical pair key after this priority selection
/// so the archived manifest is stable and independent of map iteration.
fn candidate_pairs_temporal_pyramid(
    features: &[FeatureSet],
    image_names: &[String],
    vocab_size: usize,
    topk: usize,
    max_offset: u64,
    budget: Option<usize>,
) -> Result<Vec<(usize, usize)>, String> {
    let (temporal, cross_camera) = rig_temporal_pyramid_pairs(image_names, max_offset)?;
    let retrieval = candidate_pairs_vlad_scored(features, vocab_size, topk, false, false);
    let mut selected = Vec::new();
    let mut seen = HashSet::<(usize, usize)>::new();
    for pair in temporal.into_iter().chain(cross_camera) {
        if seen.insert(pair) {
            selected.push(pair);
        }
    }
    if let Some(budget) = budget {
        selected.truncate(budget);
        if selected.len() < budget {
            for (pair, _) in retrieval {
                if seen.insert(pair) {
                    selected.push(pair);
                    if selected.len() == budget {
                        break;
                    }
                }
            }
        }
    } else {
        for (pair, _) in retrieval {
            if seen.insert(pair) {
                selected.push(pair);
            }
        }
    }
    selected.sort_unstable();
    Ok(selected)
}

fn pair_within_stem_window(
    pair: (usize, usize),
    stem_values: &[u64],
    window: u64,
) -> Result<bool, String> {
    let (i, j) = pair;
    let (&left, &right) = (
        stem_values.get(i).ok_or_else(|| {
            format!(
                "pair index {i} is outside the loaded image range 0..{}",
                stem_values.len()
            )
        })?,
        stem_values.get(j).ok_or_else(|| {
            format!(
                "pair index {j} is outside the loaded image range 0..{}",
                stem_values.len()
            )
        })?,
    );
    let difference = left.abs_diff(right);
    Ok(difference <= window)
}

/// Apply the validated numeric-stem window while preserving the input order.
/// Candidate generators already provide deterministic order (and imported
/// records have a deterministic file order), so filtering must not introduce
/// a second traversal policy.
fn filter_pairs_by_stem_window(
    pairs: Vec<(usize, usize)>,
    image_names: &[String],
    window: Option<u64>,
) -> Result<Vec<(usize, usize)>, String> {
    let Some(window) = window else {
        return Ok(pairs);
    };
    if window == 0 {
        return Err("--pair-stem-window must be at least 1".into());
    }
    let stem_values = numeric_stem_values(image_names)?;
    pairs
        .into_iter()
        .filter_map(
            |pair| match pair_within_stem_window(pair, &stem_values, window) {
                Ok(true) => Some(Ok(pair)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            },
        )
        .collect()
}

const CANDIDATE_MANIFEST_MAGIC: &str = "visloc_candidate_manifest_v1";

/// Parse a small, image-name-bound candidate-pair manifest.  The format is
/// intentionally line-oriented so it can be inspected, hashed, and generated
/// without a JSON dependency in the Rust example:
///
/// visloc_candidate_manifest_v1
/// images 2
/// image 0 first.JPG
/// image 1 second.JPG
/// pairs 1
/// pair 0 1
///
/// Pair order is preserved, while duplicate/reversed pairs are rejected.  A
/// manifest is a candidate schedule only; it contains no raw matches or
/// verification outcomes.
fn parse_candidate_manifest(
    path: &Path,
    image_names: &[String],
) -> Result<Vec<(usize, usize)>, String> {
    parse_candidate_manifest_with_metadata(path, image_names).map(|(pairs, _)| pairs)
}

/// Parse a candidate manifest and retain its optional deterministic policy
/// metadata.  Metadata is deliberately a tiny `metadata KEY VALUE` block so
/// older readers can still reject unsupported extensions rather than silently
/// changing the candidate schedule.
fn parse_candidate_manifest_with_metadata(
    path: &Path,
    image_names: &[String],
) -> Result<(Vec<(usize, usize)>, BTreeMap<String, String>), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read candidate manifest {path:?}: {error}"))?;
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let mut cursor = 0usize;
    let next = |cursor: &mut usize, label: &str| -> Result<&str, String> {
        let line = lines.get(*cursor).copied().ok_or_else(|| {
            format!("candidate manifest {path:?} is truncated while reading {label}")
        })?;
        *cursor += 1;
        Ok(line)
    };
    if next(&mut cursor, "header")? != CANDIDATE_MANIFEST_MAGIC {
        return Err(format!(
            "candidate manifest {path:?} has unsupported header (expected {CANDIDATE_MANIFEST_MAGIC})"
        ));
    }
    let image_header = next(&mut cursor, "image count")?;
    let image_fields: Vec<&str> = image_header.split_whitespace().collect();
    if image_fields.len() != 2 || image_fields[0] != "images" {
        return Err(format!(
            "candidate manifest {path:?} image count must be images N"
        ));
    }
    let image_count: usize = image_fields[1].parse().map_err(|error| {
        format!("candidate manifest {path:?} image count is not numeric: {error}")
    })?;
    if image_count != image_names.len() {
        return Err(format!(
            "candidate manifest {path:?} image count {} differs from loaded {}",
            image_count,
            image_names.len()
        ));
    }
    for expected_index in 0..image_count {
        let line = next(&mut cursor, "image entry")?;
        let mut fields = line.splitn(3, char::is_whitespace);
        let kind = fields.next().unwrap_or_default();
        let index = fields.next().unwrap_or_default();
        let name = fields.next().unwrap_or_default().trim();
        if kind != "image" || name.is_empty() {
            return Err(format!(
                "candidate manifest {path:?} image entry must be image INDEX NAME"
            ));
        }
        let index: usize = index.parse().map_err(|error| {
            format!("candidate manifest {path:?} image index is not numeric: {error}")
        })?;
        if index != expected_index || name != image_names[index] {
            return Err(format!(
                "candidate manifest {path:?} image entry {expected_index} does not match loaded image order"
            ));
        }
    }
    let mut metadata = BTreeMap::new();
    while let Some(line) = lines.get(cursor).copied() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.first().copied() != Some("metadata") {
            break;
        }
        if fields.len() != 3 || fields[1].is_empty() || fields[2].is_empty() {
            return Err(format!(
                "candidate manifest {path:?} metadata must be metadata KEY VALUE"
            ));
        }
        if metadata
            .insert(fields[1].to_owned(), fields[2].to_owned())
            .is_some()
        {
            return Err(format!(
                "candidate manifest {path:?} repeats metadata key {:?}",
                fields[1]
            ));
        }
        cursor += 1;
    }
    let pair_header = next(&mut cursor, "pair count")?;
    let pair_fields: Vec<&str> = pair_header.split_whitespace().collect();
    if pair_fields.len() != 2 || pair_fields[0] != "pairs" {
        return Err(format!(
            "candidate manifest {path:?} pair count must be pairs N"
        ));
    }
    let pair_count: usize = pair_fields[1].parse().map_err(|error| {
        format!("candidate manifest {path:?} pair count is not numeric: {error}")
    })?;
    let mut pairs = Vec::with_capacity(pair_count);
    let mut seen = HashSet::with_capacity(pair_count);
    for pair_number in 0..pair_count {
        let line = next(&mut cursor, "pair entry")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 3 || fields[0] != "pair" {
            return Err(format!(
                "candidate manifest {path:?} pair {pair_number} must be pair I J"
            ));
        }
        let i: usize = fields[1].parse().map_err(|error| {
            format!("candidate manifest {path:?} pair {pair_number} first index is not numeric: {error}")
        })?;
        let j: usize = fields[2].parse().map_err(|error| {
            format!("candidate manifest {path:?} pair {pair_number} second index is not numeric: {error}")
        })?;
        if i >= image_names.len() || j >= image_names.len() || i >= j {
            return Err(format!(
                "candidate manifest {path:?} pair {pair_number} must satisfy 0 <= I < J < {}",
                image_names.len()
            ));
        }
        if !seen.insert((i, j)) {
            return Err(format!(
                "candidate manifest {path:?} repeats pair ({i},{j})"
            ));
        }
        pairs.push((i, j));
    }
    if cursor != lines.len() {
        return Err(format!(
            "candidate manifest {path:?} has unexpected trailing data"
        ));
    }
    Ok((pairs, metadata))
}

const PERSISTENT_MATCH_WORKER_PLAN_MAGIC: &str = "visloc_match_worker_plan_v1";

#[derive(Debug, Clone)]
struct PersistentMatchWorkerShard {
    id: usize,
    candidate_path: PathBuf,
    snapshot_path: PathBuf,
    candidate_sha256: String,
}

#[derive(Debug, Clone)]
struct PersistentMatchWorkerPlan {
    root: PathBuf,
    image_names: Vec<String>,
    pair_count: usize,
    candidate_index_sha256: String,
    feature_manifest_sha256: String,
    shards: Vec<PersistentMatchWorkerShard>,
}

/// Reject plan paths that could escape the plan directory.  The external
/// runner writes plans at the artifact root, so both candidate and snapshot
/// paths are intentionally simple POSIX-style relative paths.
fn persistent_plan_relative_path(raw: &str, label: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(raw);
    if raw.is_empty()
        || raw.contains('\\')
        || path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(format!(
            "persistent match worker {label} must be a simple relative path: {raw:?}"
        ));
    }
    Ok(path)
}

/// Parse the dependency-free, versioned plan consumed by the persistent match
/// worker.  The candidate files remain the source of truth for pair metadata;
/// this plan only binds image order, total coverage, and each input/output
/// path.  Python performs the stronger SHA-256/index validation before launch.
fn parse_persistent_match_worker_plan(path: &Path) -> Result<PersistentMatchWorkerPlan, String> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "cannot read persistent match worker plan {}: {error}",
            path.display()
        )
    })?;
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let mut cursor = 0usize;
    let next = |cursor: &mut usize, label: &str| -> Result<&str, String> {
        let line = lines.get(*cursor).copied().ok_or_else(|| {
            format!(
                "persistent match worker plan {} is truncated while reading {label}",
                path.display()
            )
        })?;
        *cursor += 1;
        Ok(line)
    };
    if next(&mut cursor, "header")? != PERSISTENT_MATCH_WORKER_PLAN_MAGIC {
        return Err(format!(
            "persistent match worker plan {} has unsupported header (expected {PERSISTENT_MATCH_WORKER_PLAN_MAGIC})",
            path.display()
        ));
    }
    let parse_count = |line: &str, kind: &str| -> Result<usize, String> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 2 || fields[0] != kind {
            return Err(format!(
                "persistent match worker plan {} requires `{kind} N`",
                path.display()
            ));
        }
        fields[1].parse::<usize>().map_err(|error| {
            format!(
                "persistent match worker plan {} {kind} count is not numeric: {error}",
                path.display()
            )
        })
    };
    let image_count = parse_count(next(&mut cursor, "image count")?, "images")?;
    if image_count < 2 {
        return Err(format!(
            "persistent match worker plan {} needs at least two images",
            path.display()
        ));
    }
    let mut image_names = Vec::with_capacity(image_count);
    for expected_index in 0..image_count {
        let line = next(&mut cursor, "image entry")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 3 || fields[0] != "image" {
            return Err(format!(
                "persistent match worker plan {} image entry must be image INDEX NAME",
                path.display()
            ));
        }
        let index = fields[1].parse::<usize>().map_err(|error| {
            format!(
                "persistent match worker plan {} image index is not numeric: {error}",
                path.display()
            )
        })?;
        if index != expected_index || fields[2].is_empty() {
            return Err(format!(
                "persistent match worker plan {} image entry {expected_index} is not ordered",
                path.display()
            ));
        }
        image_names.push(fields[2].to_owned());
    }
    let parse_hash = |line: &str, kind: &str| -> Result<String, String> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let value = fields.get(1).copied().unwrap_or_default();
        if fields.len() != 2
            || fields[0] != kind
            || value.len() != 64
            || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(format!(
                "persistent match worker plan {} requires `{kind} SHA256`",
                path.display()
            ));
        }
        Ok(value.to_ascii_lowercase())
    };
    let candidate_index_sha256 = parse_hash(
        next(&mut cursor, "candidate index hash")?,
        "candidate_index_sha256",
    )?;
    let feature_manifest_sha256 = parse_hash(
        next(&mut cursor, "feature manifest hash")?,
        "feature_manifest_sha256",
    )?;
    let pair_count = parse_count(next(&mut cursor, "pair count")?, "pairs")?;
    if pair_count == 0 {
        return Err(format!(
            "persistent match worker plan {} must contain at least one pair",
            path.display()
        ));
    }
    let shard_count = parse_count(next(&mut cursor, "shard count")?, "shards")?;
    if shard_count == 0 {
        return Err(format!(
            "persistent match worker plan {} must contain at least one shard",
            path.display()
        ));
    }
    let mut shards = Vec::with_capacity(shard_count);
    let mut all_paths = HashSet::with_capacity(shard_count * 2);
    let mut previous_id = None;
    for _shard_index in 0..shard_count {
        let line = next(&mut cursor, "shard entry")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 5 || fields[0] != "shard" {
            return Err(format!(
                "persistent match worker plan {} shard entry must be shard ID CANDIDATE SNAPSHOT CANDIDATE_SHA256",
                path.display()
            ));
        }
        let id = fields[1].parse::<usize>().map_err(|error| {
            format!(
                "persistent match worker plan {} shard id is not numeric: {error}",
                path.display()
            )
        })?;
        if previous_id.is_some_and(|previous| id <= previous) {
            return Err(format!(
                "persistent match worker plan {} shard IDs must be strictly increasing",
                path.display()
            ));
        }
        previous_id = Some(id);
        let candidate_path =
            persistent_plan_relative_path(fields[2], &format!("shard {id} candidate path"))?;
        let snapshot_path =
            persistent_plan_relative_path(fields[3], &format!("shard {id} snapshot path"))?;
        let candidate_sha256 = parse_hash(
            &format!("candidate_sha256 {}", fields[4]),
            "candidate_sha256",
        )?;
        if candidate_path == snapshot_path {
            return Err(format!(
                "persistent match worker plan {} shard {id} reuses one path for candidate and snapshot",
                path.display()
            ));
        }
        if !all_paths.insert(candidate_path.clone()) {
            return Err(format!(
                "persistent match worker plan {} repeats candidate or snapshot path {}",
                path.display(),
                candidate_path.display()
            ));
        }
        if !all_paths.insert(snapshot_path.clone()) {
            return Err(format!(
                "persistent match worker plan {} repeats candidate or snapshot path {}",
                path.display(),
                snapshot_path.display()
            ));
        }
        shards.push(PersistentMatchWorkerShard {
            id,
            candidate_path,
            snapshot_path,
            candidate_sha256,
        });
    }
    if cursor != lines.len() {
        return Err(format!(
            "persistent match worker plan {} has unexpected trailing data",
            path.display()
        ));
    }
    let root = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    Ok(PersistentMatchWorkerPlan {
        root,
        image_names,
        pair_count,
        candidate_index_sha256,
        feature_manifest_sha256,
        shards,
    })
}

/// Write a candidate manifest through a same-directory temporary file and
/// rename.  This keeps an interrupted cheap retrieval pass from leaving a
/// file that a later benchmark could mistake for a complete schedule.
#[cfg(test)]
fn write_candidate_manifest(
    path: &Path,
    image_names: &[String],
    pairs: &[(usize, usize)],
) -> Result<(), String> {
    write_candidate_manifest_with_metadata(path, image_names, pairs, &BTreeMap::new())
}

/// Write a candidate manifest with a canonical metadata block.  Keys are
/// sorted by `BTreeMap`, making the bytes stable across runs and suitable for
/// the hash-bound shard index.
fn write_candidate_manifest_with_metadata(
    path: &Path,
    image_names: &[String],
    pairs: &[(usize, usize)],
    metadata: &BTreeMap<String, String>,
) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|error| {
        format!("cannot create candidate manifest directory {parent:?}: {error}")
    })?;
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("candidate manifest path has no valid filename: {path:?}"))?;
    let temporary = parent.join(format!(".{file_name}.tmp"));
    let mut text = String::new();
    text.push_str(CANDIDATE_MANIFEST_MAGIC);
    text.push('\n');
    text.push_str(&format!("images {}\n", image_names.len()));
    for (index, name) in image_names.iter().enumerate() {
        if name.chars().any(char::is_whitespace) {
            return Err(format!(
                "candidate manifest cannot encode whitespace in image name {name:?}"
            ));
        }
        text.push_str(&format!("image {index} {name}\n"));
    }
    for (key, value) in metadata {
        if key.is_empty()
            || value.is_empty()
            || key.chars().any(char::is_whitespace)
            || value.chars().any(char::is_whitespace)
        {
            return Err(format!(
                "candidate manifest metadata must use non-empty whitespace-free KEY VALUE (got {key:?}={value:?})"
            ));
        }
        text.push_str(&format!("metadata {key} {value}\n"));
    }
    text.push_str(&format!("pairs {}\n", pairs.len()));
    let mut seen = HashSet::with_capacity(pairs.len());
    for &(i, j) in pairs {
        if i >= image_names.len() || j >= image_names.len() || i >= j {
            return Err(format!(
                "candidate pair ({i},{j}) is outside canonical image order"
            ));
        }
        if !seen.insert((i, j)) {
            return Err(format!("candidate pair ({i},{j}) is duplicated"));
        }
        text.push_str(&format!("pair {i} {j}\n"));
    }
    std::fs::write(&temporary, text).map_err(|error| {
        format!("cannot write temporary candidate manifest {temporary:?}: {error}")
    })?;
    std::fs::rename(&temporary, path).map_err(|error| {
        format!("cannot atomically install candidate manifest {path:?}: {error}")
    })?;
    Ok(())
}

/// Add the numeric-consecutive edges required by the opt-in sequence
/// relative-pose fallback.  Retrieval remains the normal source for every
/// other edge; only missing consecutive pairs are appended, so the flag does
/// not silently turn a bounded retrieval run into an exhaustive matcher.
fn append_consecutive_stem_candidates(
    pairs: &mut Vec<(usize, usize)>,
    image_names: &[String],
) -> Result<usize, String> {
    let stem_values = numeric_stem_values(image_names)?;
    let mut by_stem: Vec<(u64, usize)> = stem_values
        .into_iter()
        .enumerate()
        .map(|(image, stem)| (stem, image))
        .collect();
    by_stem.sort_unstable();

    let existing: HashSet<(usize, usize)> = pairs.iter().copied().collect();
    let mut added = 0usize;
    for window in by_stem.windows(2) {
        let [(left_stem, left_image), (right_stem, right_image)] = window else {
            unreachable!("windows(2) always has two entries");
        };
        if right_stem.saturating_sub(*left_stem) != 1 {
            continue;
        }
        let pair = (*left_image, *right_image);
        if !existing.contains(&pair) {
            pairs.push(pair);
            added += 1;
        }
    }
    Ok(added)
}

fn filter_imported_verified_pairs_by_stem_window(
    imported: Vec<ImportedVerifiedPair>,
    image_names: &[String],
    window: Option<u64>,
) -> Result<Vec<ImportedVerifiedPair>, String> {
    let Some(window) = window else {
        return Ok(imported);
    };
    let stem_values = numeric_stem_values(image_names)?;
    imported
        .into_iter()
        .filter_map(|pair| {
            match pair_within_stem_window((pair.image_i, pair.image_j), &stem_values, window) {
                Ok(true) => Some(Ok(pair)),
                Ok(false) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .collect()
}

/// Validate the mutually-exclusive diagnostic output modes before loading
/// features, and optionally validate image indices once the image count is
/// known. Keeping this separate from `parse_args` makes the CLI contract
/// testable without synthesizing process arguments.
fn validate_diagnose_options(
    pairs_csv: Option<&Path>,
    pair_stems: &[String],
    pairs: &[(usize, usize)],
    image_count: Option<usize>,
) -> Result<(), String> {
    if pairs_csv.is_none() && !pair_stems.is_empty() {
        return Err("--diagnose-pair-stems requires --diagnose-pairs-csv PATH".into());
    }
    if pairs_csv.is_some() && !pairs.is_empty() {
        return Err("--diagnose-pairs-csv and --diagnose-pair are mutually exclusive".into());
    }
    if let Some(n) = image_count {
        for &(i, j) in pairs {
            if i == j {
                return Err(format!(
                    "--diagnose-pair requires distinct image indices, got {i},{j}"
                ));
            }
            if i >= n || j >= n {
                return Err(format!(
                    "--diagnose-pair index out of range: {i},{j} for {n} images"
                ));
            }
        }
    }
    Ok(())
}

fn validate_diagnose_stems(image_names: &[String], stems: &[String]) -> Result<(), String> {
    for stem in stems {
        if !image_names.iter().any(|name| image_stem(name) == stem) {
            return Err(format!(
                "--diagnose-pair-stems stem {stem:?} does not match a loaded image"
            ));
        }
    }
    Ok(())
}

fn parse_seed_pair(raw: &str) -> Result<(usize, usize), String> {
    let (left, right) = raw
        .split_once(',')
        .ok_or_else(|| format!("--seed-pair expects I,J, got {raw:?}"))?;
    let i: usize = left
        .trim()
        .parse()
        .map_err(|e| format!("--seed-pair invalid first index in {raw:?}: {e}"))?;
    let j: usize = right
        .trim()
        .parse()
        .map_err(|e| format!("--seed-pair invalid second index in {raw:?}: {e}"))?;
    if i == j {
        return Err(format!(
            "--seed-pair requires two distinct images, got {i},{j}"
        ));
    }
    Ok((i.min(j), i.max(j)))
}

fn parse_args() -> Result<Args, String> {
    parse_args_from(env::args().skip(1))
}

fn parse_args_from<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut features_dir = None;
    let mut feature_suffix = String::from("_features.txt");
    let mut image_suffix = String::from(".png");
    let mut out_colmap = None;
    let mut input_colmap_calibration: Option<PathBuf> = None;
    let (mut width, mut height) = (None, None);
    let (mut fx, mut fy, mut cx, mut cy) = (None, None, None, None);
    let mut vocab_size = 64usize;
    let mut retrieval_topk = 12usize;
    let mut exhaustive = false;
    let mut match_ratio = 0.8f32;
    let mut min_matches = 30usize;
    let mut min_pnp_inliers = 12usize;
    let mut max_mapper_matches_per_pair: Option<usize> = None;
    let mut max_reproj = 4.0f64;
    // The demo's no-flag workflow uses the robust two-stage policy; the public
    // library default remains CorrespondenceCount for API/snapshot identity.
    let mut next_image_policy = NextImagePolicy::Auto;
    let mut next_image_policy_explicit = false;
    let mut final_ba = true;
    let mut final_min_track_length: Option<usize> = None;
    let mut seed_trials = 12usize;
    let mut seed_pair: Option<(usize, usize)> = None;
    let mut pair_stem_window: Option<u64> = None;
    let mut local_stem_window: Option<u64> = None;
    let mut rig_local_grouping = false;
    let mut temporal_pyramid_max_offset = 32u64;
    let mut candidate_budget: Option<usize> = None;
    let mut refine_intrinsics = false;
    let mut refine_distortion = false;
    let mut colmap_style = false;
    let mut final_iterative_global_refinement = false;
    let mut global_ba_max_refinements: Option<usize> = None;
    let mut post_refinement_registration = false;
    let mut structureless_registration = false;
    let mut guided_matching = false;
    let mut colmap_guided_matching = false;
    let mut multiple_models = false;
    let mut min_e_f_inlier_ratio: Option<f64> = None;
    let mut calibrated_prefer_essential = false;
    let mut refine_uncalibrated_f_to_essential = false;
    let mut strict_uncalibrated_f_to_essential = false;
    let mut calibrated_essential_primary = false;
    let mut prefer_essential_inliers = false;
    let mut prefer_essential_free_endpoints = false;
    let mut prefer_essential_stems: Vec<String> = Vec::new();
    let mut prefer_essential_stem_clique = false;
    let mut prefer_essential_pairs: Vec<(usize, usize)> = Vec::new();
    let mut require_essential_selected_edges = false;
    let mut require_essential_stems: Vec<String> = Vec::new();
    let mut require_essential_min_e_inliers = 0usize;
    let mut rematch_stems: Vec<String> = Vec::new();
    let mut rematch_ratio = 0.9f32;
    let mut rematch_cross_check = true;
    let mut rematch_guided = false;
    let mut rematch_free_vs_priors = false;
    let mut rematch_prefer_min_e_inliers = 0usize;
    let mut rematch_prefer_strong_stems: Vec<String> = Vec::new();
    let mut rematch_prefer_strong_min_e = 50usize;
    let mut rematch_tracks_use_essential = false;
    let mut rematch_min_chirality_margin = 0.0f64;
    let mut rematch_prior_anchor = false;
    let mut rematch_anchor_min_e_inliers = 25usize;
    let mut rematch_min_e_f_inlier_ratio: Option<f64> = None;
    let mut rematch_calibrated_prefer_essential = false;
    let mut rematch_prior_ray_guided = false;
    let mut rematch_prior_ray_min_rays = 2usize;
    let mut rematch_prior_ray_min_e_inliers = 25usize;
    let mut rematch_verification_mode: Option<VerificationMode> = None;
    let mut rematch_pose_guided_after_global = false;
    let mut rematch_pose_guided_gt: Option<PathBuf> = None;
    let mut essential_edge_weight_boost = 1.0f64;
    let mut force_essential_matches = false;
    let mut force_essential_min_ef_ratio = 0.7f64;
    let mut force_essential_min_e_inliers = 0usize;
    let mut force_essential_uncalibrated_only = false;
    let mut repnp_free_from_priors = false;
    let mut repnp_free_min_corrs = 0usize;
    let mut repnp_seed_free_as_priors = false;
    let mut repair_prior_edges = false;
    let mut repair_free_edges_from_solved = false;
    let mut repair_free_edges_only_flipped = false;
    let mut repair_free_edges_stems: Vec<String> = Vec::new();
    let mut drop_free_edges_antipodal = false;
    let mut prior_guided_free_chirality = false;
    let mut metric_prior_chirality_edges = false;
    let mut metric_prior_chirality_min_rays = 3usize;
    let mut diagnose_bearing_gt: Option<PathBuf> = None;
    let mut diagnose_bearing_stems: Vec<String> = Vec::new();
    let mut gt_chirality_oracle = false;
    let mut gt_chirality_oracle_path: Option<PathBuf> = None;
    let mut rematch_max_gt_bearing_deg = 0.0f64;
    let mut rematch_gt_bearing_path: Option<PathBuf> = None;
    let mut rematch_guided_max_error_px: Option<f64> = None;
    let mut rematch_guided_lowe_ratio: Option<f64> = None;
    let mut rematch_require_calibrated = false;
    let mut rematch_max_mean_sampson = 0.0f64;
    let mut metric_prior_scale = false;
    let mut sequence_relative_pose_fallback = false;
    let mut sequence_fallback_after_post = false;
    let mut sequence_constant_velocity_scale = false;
    let mut sequence_relaxed_constant_velocity_scale = false;
    let mut sequence_fallback_carry_scale = false;
    let mut pnp_max_iterations = 128usize;
    let mut ba_max_iterations: Option<usize> = None;
    let mut ba_huber_delta: Option<f64> = None;
    let mut ba_linear_solver: Option<LinearSolver> = None;
    let mut periodic_ba_min_registered_images = 0usize;
    let mut final_ba_polish_iterations = 0usize;
    let mut geometry_weighted_ba = false;
    let mut freeze_ill_conditioned_landmarks = false;
    let mut landmark_ba_warm_start_iterations = 0usize;
    let mut landmark_ba_warm_start_min_registered_images = 0usize;
    let mut filter_images = false;
    let mut verification_mode = VerificationMode::Legacy;
    let mut track_source = TrackSource::UnionFind;
    let mut confidence_ordered_tracks = false;
    let mut geometric_confidence_tracks = false;
    let mut stable_track_order = false;
    let mut cycle_supported_tracks = false;
    let mut canonical_feature_order = false;
    let mut union_traversal_order = UnionTraversalOrder::Original;
    let mut geometry_guided_conflict_recovery = false;
    let mut pair_source = PairSource::Vlad;
    let mut vocab_tree_branching = 10usize;
    let mut vocab_tree_depth = 3usize;
    let mut vocab_tree_num_images = 100usize;
    let mut rescue_bridging = false;
    let mut rescue_match_ratio = 0.95f32;
    let mut rescue_min_matches = 15usize;
    let mut rescue_max_candidates = 200usize;
    let mut rescue_cross_check = false;
    let mut diagnose_pairs: Vec<(usize, usize)> = Vec::new();
    let mut diagnose_pairs_csv: Option<PathBuf> = None;
    let mut diagnose_pair_stems: Vec<String> = Vec::new();
    let mut matcher = MatcherKind::Nn;
    let mut lightglue_model: Option<PathBuf> = None;
    let mut onnx_backend = String::from("auto");
    let mut lightglue_max_keypoints = 0usize;
    let mut import_matches_file: Option<PathBuf> = None;
    let mut import_matches_supplement_file: Option<PathBuf> = None;
    let mut export_features_dir: Option<PathBuf> = None;
    let mut export_features_only = false;
    let mut sift_stream_export = false;
    let mut sift_stream_resume = false;
    let mut import_verified_pairs_file: Option<PathBuf> = None;
    let mut export_verified_pairs_snapshot: Option<PathBuf> = None;
    let mut import_verified_pairs_snapshot: Option<PathBuf> = None;
    let mut snapshot_keypoints_only = false;
    let mut export_verified_pairs_only = false;
    let mut persistent_match_worker_plan: Option<PathBuf> = None;
    let mut candidate_manifest: Option<PathBuf> = None;
    let mut export_candidate_manifest: Option<PathBuf> = None;
    let mut snapshot_coordinate_override_dir: Option<PathBuf> = None;
    let mut diagnose_ba_oracle_poses_file: Option<PathBuf> = None;
    let mut diagnose_fixed_rotation_ba: Option<String> = None;
    let mut diagnose_model_score_file: Option<PathBuf> = None;
    let mut initial_poses_file: Option<PathBuf> = None;
    let mut diagnose_colmap_track_membership: Option<PathBuf> = None;
    let mut pose_guided_track_splitting = false;
    let mut pose_guided_track_splitting_graph_support = false;
    let mut pose_guided_track_splitting_bridge_cuts = false;
    let mut pose_guided_split_max_reproj: Option<f64> = None;
    let mut pose_guided_track_splitting_iterations: Option<usize> = None;
    let mut pose_guided_track_merging = false;
    let mut pose_guided_merge_max_reproj: Option<f64> = None;
    let mut feature_extractor = FeatureExtractorKind::Files;
    let mut mapper = MapperKind::Incremental;
    let mut chirality_harden = false;
    let mut rotation_seed_trials = 1usize;
    let mut refine_global_translations = false;
    let mut global_independent_edge_scales = false;
    let mut multi_hypothesis_edges = false;
    let mut min_edge_inliers = 15usize;
    let mut min_edge_parallax_deg = 2.0f64;
    let mut weight_by_chirality_margin = false;
    let mut hybrid_filter_priors = false;
    let mut hybrid_prior_min_obs = 50usize;
    let mut hybrid_prior_max_reproj = 0.45f64;
    let mut hybrid_drop_prior_stems: Vec<String> = Vec::new();
    let mut hybrid_drop_inconsistent_priors = false;
    let mut verify_registration_two_view = false;
    let mut hybrid_rotation_priors_only = false;
    let mut joint_global_positioning = false;
    let mut calibrated_view_edges_only = false;
    let mut images_dir: Option<PathBuf> = None;
    let mut sift_max_keypoints = 2048usize;
    let mut sift_affine = false;
    let mut sift_detector = String::from("dog");
    let mut sift_multi_anisotropy = false;
    let mut sift_dsp = false;
    let mut sift_dsp_num_scales = 15usize;
    let mut sift_l1_root = false;
    let mut sift_max_orientations = 0usize;
    let mut sift_standard_orientations = false;
    let mut sift_prefer_larger_scale = false;
    let mut sift_full_pyramid = false;
    let mut sift_contrast_threshold = 0.02f64;
    let mut sift_descriptor_magnification = 8.0f64;
    let mut sift_descriptor_magnification_explicit = false;
    let mut sift_scale_adaptive_gradients = false;
    let mut sift_vlfeat_compatible_descriptor = false;
    let mut sift_vlfeat_compatible_detector = false;
    let mut sift_vlfeat_bilinear_orientations = false;
    let mut sift_vlfeat_compatible_output_order = false;
    let mut sift_colmap_compatible_grayscale = false;
    let mut sift_split_colmap_detector_grayscale = false;
    let mut sift_append_descriptor_magnification: Option<f64> = None;
    let mut sift_extra_keypoints_stems: Vec<String> = Vec::new();
    let mut sift_extra_keypoints = 0usize;
    let mut sift_extra_contrast_threshold: Option<f64> = None;
    let mut sift_extra_matches_append_only = false;
    let mut incremental_correspondence_triangulation = false;
    let mut orientation_locus_canonicalization = false;

    let mut a: Vec<String> = args.into_iter().collect();
    let mut i = 0;
    while i < a.len() {
        match a[i].as_str() {
            "--features-dir" => features_dir = Some(PathBuf::from(a.remove(i + 1))),
            "--hybrid-filter-priors" => hybrid_filter_priors = true,
            "--hybrid-prior-min-obs" => {
                hybrid_prior_min_obs = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--hybrid-prior-max-reproj" => {
                hybrid_prior_max_reproj = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--hybrid-drop-prior-stems" => {
                hybrid_drop_prior_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
            }
            "--hybrid-drop-inconsistent-priors" => hybrid_drop_inconsistent_priors = true,
            "--verify-registration-two-view" => verify_registration_two_view = true,
            "--hybrid-rotation-priors-only" => hybrid_rotation_priors_only = true,
            "--joint-global-positioning" => joint_global_positioning = true,
            "--calibrated-view-edges-only" => calibrated_view_edges_only = true,
            "--images-dir" => images_dir = Some(PathBuf::from(a.remove(i + 1))),
            "--feature-extractor" => {
                feature_extractor = match a.remove(i + 1).as_str() {
                    "files" => FeatureExtractorKind::Files,
                    "sift" => FeatureExtractorKind::Sift,
                    other => {
                        return Err(format!(
                            "--feature-extractor must be files|sift, got {other}"
                        ))
                    }
                };
            }
            "--sift-max-keypoints" => {
                sift_max_keypoints = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--sift-affine" => sift_affine = true,
            "--sift-multi-anisotropy" => sift_multi_anisotropy = true,
            "--sift-dsp" => sift_dsp = true,
            "--sift-dsp-num-scales" => {
                sift_dsp_num_scales = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--sift-l1-root" => sift_l1_root = true,
            "--sift-max-orientations" => {
                sift_max_orientations = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--sift-standard-orientations" => sift_standard_orientations = true,
            "--sift-prefer-larger-scale" => sift_prefer_larger_scale = true,
            "--sift-full-pyramid" => sift_full_pyramid = true,
            "--sift-contrast-threshold" => {
                sift_contrast_threshold = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--sift-descriptor-magnification" => {
                sift_descriptor_magnification_explicit = true;
                sift_descriptor_magnification =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--sift-scale-adaptive-gradients" => sift_scale_adaptive_gradients = true,
            "--sift-vlfeat-compatible-descriptor" => sift_vlfeat_compatible_descriptor = true,
            "--sift-vlfeat-compatible-detector" => sift_vlfeat_compatible_detector = true,
            "--sift-vlfeat-bilinear-orientations" => sift_vlfeat_bilinear_orientations = true,
            "--sift-vlfeat-compatible-output-order" => sift_vlfeat_compatible_output_order = true,
            "--sift-colmap-compatible-grayscale" => sift_colmap_compatible_grayscale = true,
            "--sift-split-colmap-detector-grayscale" => sift_split_colmap_detector_grayscale = true,
            "--sift-append-descriptor-magnification" => {
                let magnification: f64 = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
                if !magnification.is_finite() || magnification <= 0.0 {
                    return Err(format!(
                        "--sift-append-descriptor-magnification must be finite and > 0, got {magnification}"
                    ));
                }
                sift_append_descriptor_magnification = Some(magnification);
            }
            "--sift-extra-keypoints-stems" => {
                sift_extra_keypoints_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--sift-extra-keypoints" => {
                sift_extra_keypoints = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--sift-extra-contrast-threshold" => {
                let threshold: f64 = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
                if !threshold.is_finite() || threshold < 0.0 {
                    return Err(format!(
                        "--sift-extra-contrast-threshold must be finite and >= 0, got {threshold}"
                    ));
                }
                sift_extra_contrast_threshold = Some(threshold);
            }
            "--sift-extra-matches-append-only" => sift_extra_matches_append_only = true,
            "--incremental-correspondence-triangulation" => {
                incremental_correspondence_triangulation = true
            }
            "--orientation-locus-canonicalization" => orientation_locus_canonicalization = true,
            "--sift-detector" => sift_detector = a.remove(i + 1),
            "--feature-suffix" => feature_suffix = a.remove(i + 1),
            "--image-suffix" => image_suffix = a.remove(i + 1),
            "--out-colmap" => out_colmap = Some(PathBuf::from(a.remove(i + 1))),
            "--input-colmap-calibration" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--input-colmap-calibration requires MODEL_DIR")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--input-colmap-calibration requires a non-empty MODEL_DIR".into());
                }
                a.remove(i + 1);
                input_colmap_calibration = Some(PathBuf::from(raw));
            }
            "--width" => width = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?),
            "--height" => height = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?),
            "--fx" => fx = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?),
            "--fy" => fy = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?),
            "--cx" => cx = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?),
            "--cy" => cy = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?),
            "--vocab-size" => vocab_size = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--retrieval-topk" => {
                retrieval_topk = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--exhaustive" => exhaustive = true,
            "--pair-stem-window" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--pair-stem-window requires a positive integer N")?
                    .clone();
                let window: u64 = raw.parse().map_err(|error| {
                    format!("--pair-stem-window must be a positive integer: {error}")
                })?;
                if window == 0 {
                    return Err("--pair-stem-window must be at least 1".into());
                }
                a.remove(i + 1);
                pair_stem_window = Some(window);
            }
            "--local-stem-window" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--local-stem-window requires a positive integer N")?
                    .clone();
                let window: u64 = raw.parse().map_err(|error| {
                    format!("--local-stem-window must be a positive integer: {error}")
                })?;
                if window == 0 {
                    return Err("--local-stem-window must be at least 1".into());
                }
                a.remove(i + 1);
                local_stem_window = Some(window);
            }
            "--rig-local-grouping" => rig_local_grouping = true,
            "--temporal-pyramid-max-offset" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--temporal-pyramid-max-offset requires a positive integer")?
                    .clone();
                let offset: u64 = raw.parse().map_err(|error| {
                    format!("--temporal-pyramid-max-offset must be a positive integer: {error}")
                })?;
                if offset == 0 {
                    return Err("--temporal-pyramid-max-offset must be at least 1".into());
                }
                a.remove(i + 1);
                temporal_pyramid_max_offset = offset;
            }
            "--candidate-budget" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--candidate-budget requires a positive integer")?
                    .clone();
                let budget: usize = raw.parse().map_err(|error| {
                    format!("--candidate-budget must be a positive integer: {error}")
                })?;
                if budget == 0 {
                    return Err("--candidate-budget must be at least 1".into());
                }
                a.remove(i + 1);
                candidate_budget = Some(budget);
            }
            "--match-ratio" => match_ratio = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--min-matches" => min_matches = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--min-pnp-inliers" => {
                min_pnp_inliers = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--max-mapper-matches-per-pair" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--max-mapper-matches-per-pair requires a positive integer")?
                    .clone();
                let limit: usize = raw.parse().map_err(|error| {
                    format!("--max-mapper-matches-per-pair must be a positive integer: {error}")
                })?;
                if limit == 0 {
                    return Err("--max-mapper-matches-per-pair must be at least 1".into());
                }
                a.remove(i + 1);
                max_mapper_matches_per_pair = Some(limit);
            }
            "--max-reproj" => max_reproj = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--next-image-policy" => {
                let value = a
                    .get(i + 1)
                    .ok_or("--next-image-policy requires auto, count, or visibility")?
                    .clone();
                next_image_policy_explicit = true;
                next_image_policy = match value.as_str() {
                    "auto" => NextImagePolicy::Auto,
                    "count" => NextImagePolicy::CorrespondenceCount,
                    "visibility" => NextImagePolicy::VisibilityPyramid,
                    other => {
                        return Err(format!(
                            "--next-image-policy must be auto, count, or visibility, got {other}"
                        ));
                    }
                };
                a.remove(i + 1);
            }
            "--no-final-ba" => final_ba = false,
            "--final-min-track-length" => {
                let value: usize = a
                    .get(i + 1)
                    .ok_or("--final-min-track-length requires 3")?
                    .parse()
                    .map_err(|error| {
                        format!("--final-min-track-length must be an integer: {error}")
                    })?;
                if value != 3 {
                    return Err(format!(
                        "--final-min-track-length currently supports only 3, got {value}"
                    ));
                }
                a.remove(i + 1);
                final_min_track_length = Some(value);
            }
            "--seed-trials" => seed_trials = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--seed-pair" => seed_pair = Some(parse_seed_pair(&a.remove(i + 1))?),
            "--refine-intrinsics" => refine_intrinsics = true,
            "--refine-distortion" => refine_distortion = true,
            "--colmap-style" => colmap_style = true,
            "--final-iterative-refinement" => final_iterative_global_refinement = true,
            "--global-ba-max-refinements" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--global-ba-max-refinements requires a non-negative integer")?
                    .clone();
                let refinements: usize = raw.parse().map_err(|error| {
                    format!("--global-ba-max-refinements must be a non-negative integer: {error}")
                })?;
                a.remove(i + 1);
                global_ba_max_refinements = Some(refinements);
            }
            "--post-refinement-registration" => post_refinement_registration = true,
            "--structureless-registration" => structureless_registration = true,
            "--sequence-relative-pose-fallback" => sequence_relative_pose_fallback = true,
            "--sequence-fallback-after-post" => sequence_fallback_after_post = true,
            "--sequence-constant-velocity-scale" => sequence_constant_velocity_scale = true,
            "--sequence-relaxed-constant-velocity-scale" => {
                sequence_relaxed_constant_velocity_scale = true
            }
            "--sequence-fallback-carry-scale" => sequence_fallback_carry_scale = true,
            "--guided-matching" => guided_matching = true,
            "--colmap-guided-matching" => colmap_guided_matching = true,
            "--multiple-models" => multiple_models = true,
            "--min-e-f-inlier-ratio" => {
                min_e_f_inlier_ratio = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--calibrated-prefer-essential" => calibrated_prefer_essential = true,
            "--refine-uncalibrated-f-to-essential" => refine_uncalibrated_f_to_essential = true,
            "--strict-uncalibrated-f-to-essential" => strict_uncalibrated_f_to_essential = true,
            "--calibrated-essential-primary" => calibrated_essential_primary = true,
            "--prefer-essential-inliers" => prefer_essential_inliers = true,
            "--prefer-essential-free-endpoints" => prefer_essential_free_endpoints = true,
            "--prefer-essential-stems" => {
                prefer_essential_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--prefer-essential-stem-clique" => prefer_essential_stem_clique = true,
            "--prefer-essential-pairs" => {
                let raw = a.remove(i + 1);
                prefer_essential_pairs = raw
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        let (l, r) = s.split_once('-').ok_or_else(|| {
                            format!("--prefer-essential-pairs expects I-J, got {s:?}")
                        })?;
                        Ok::<_, String>((
                            l.parse().map_err(|e| format!("{e}"))?,
                            r.parse().map_err(|e| format!("{e}"))?,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--require-essential-selected-edges" => require_essential_selected_edges = true,
            "--require-essential-stems" => {
                require_essential_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--require-essential-min-e-inliers" => {
                require_essential_min_e_inliers =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-stems" => {
                rematch_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--rematch-ratio" => {
                rematch_ratio = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-no-cross-check" => rematch_cross_check = false,
            "--rematch-guided" => rematch_guided = true,
            "--rematch-free-vs-priors" => rematch_free_vs_priors = true,
            "--rematch-prefer-min-e-inliers" => {
                rematch_prefer_min_e_inliers =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-prefer-strong-stems" => {
                rematch_prefer_strong_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--rematch-prefer-strong-min-e" => {
                rematch_prefer_strong_min_e = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-tracks-use-essential" => rematch_tracks_use_essential = true,
            "--rematch-min-chirality-margin" => {
                rematch_min_chirality_margin =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-prior-anchor" => rematch_prior_anchor = true,
            "--rematch-anchor-min-e-inliers" => {
                rematch_anchor_min_e_inliers =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-min-e-f-inlier-ratio" => {
                rematch_min_e_f_inlier_ratio =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--rematch-calibrated-prefer-essential" => rematch_calibrated_prefer_essential = true,
            "--rematch-prior-ray-guided" => rematch_prior_ray_guided = true,
            "--rematch-prior-ray-min-rays" => {
                rematch_prior_ray_min_rays = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-prior-ray-min-e-inliers" => {
                rematch_prior_ray_min_e_inliers =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-verification-mode" => {
                rematch_verification_mode = Some(a.remove(i + 1).parse().map_err(|e: String| e)?)
            }
            "--rematch-pose-guided-after-global" => rematch_pose_guided_after_global = true,
            "--rematch-pose-guided-gt" => {
                rematch_pose_guided_gt = Some(PathBuf::from(a.remove(i + 1)));
                rematch_pose_guided_after_global = true;
            }
            "--essential-edge-weight-boost" => {
                essential_edge_weight_boost = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--force-essential-matches" => force_essential_matches = true,
            "--force-essential-min-ef-ratio" => {
                force_essential_min_ef_ratio =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--force-essential-min-e-inliers" => {
                force_essential_min_e_inliers =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--force-essential-uncalibrated-only" => force_essential_uncalibrated_only = true,
            "--repnp-free-from-priors" => repnp_free_from_priors = true,
            "--repnp-free-min-corrs" => {
                repnp_free_min_corrs = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--repnp-seed-free-as-priors" => repnp_seed_free_as_priors = true,
            "--repair-prior-edges" => repair_prior_edges = true,
            "--repair-free-edges-from-solved" => repair_free_edges_from_solved = true,
            "--repair-free-edges-only-flipped" => repair_free_edges_only_flipped = true,
            "--repair-free-edges-stems" => {
                repair_free_edges_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--drop-free-edges-antipodal" => drop_free_edges_antipodal = true,
            "--prior-guided-free-chirality" => prior_guided_free_chirality = true,
            "--metric-prior-chirality-edges" => metric_prior_chirality_edges = true,
            "--metric-prior-chirality-min-rays" => {
                metric_prior_chirality_min_rays =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--diagnose-bearing-gt" => {
                diagnose_bearing_gt = Some(PathBuf::from(a.remove(i + 1)));
            }
            "--diagnose-bearing-stems" => {
                diagnose_bearing_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--gt-chirality-oracle" => gt_chirality_oracle = true,
            "--gt-chirality-oracle-path" => {
                gt_chirality_oracle_path = Some(PathBuf::from(a.remove(i + 1)));
            }
            "--rematch-max-gt-bearing-deg" => {
                rematch_max_gt_bearing_deg = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-gt-bearing-path" => {
                rematch_gt_bearing_path = Some(PathBuf::from(a.remove(i + 1)));
            }
            "--rematch-guided-max-error-px" => {
                rematch_guided_max_error_px =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--rematch-guided-lowe-ratio" => {
                rematch_guided_lowe_ratio =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--rematch-require-calibrated" => rematch_require_calibrated = true,
            "--rematch-max-mean-sampson" => {
                rematch_max_mean_sampson = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--metric-prior-scale" => metric_prior_scale = true,
            "--pnp-max-iterations" => {
                pnp_max_iterations = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--ba-max-iterations" => {
                let iterations: usize = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
                if iterations == 0 {
                    return Err("--ba-max-iterations must be at least 1".into());
                }
                ba_max_iterations = Some(iterations);
            }
            "--ba-huber-delta" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--ba-huber-delta requires a positive finite pixel value")?
                    .clone();
                let delta: f64 = raw
                    .parse()
                    .map_err(|error| format!("--ba-huber-delta must be numeric: {error}"))?;
                if !delta.is_finite() || delta <= 0.0 {
                    return Err("--ba-huber-delta must be a positive finite pixel value".into());
                }
                a.remove(i + 1);
                ba_huber_delta = Some(delta);
            }
            "--ba-linear-solver" => {
                let solver = a
                    .get(i + 1)
                    .ok_or("--ba-linear-solver requires dense or sparse")?
                    .as_str();
                let parsed = match solver {
                    "dense" => LinearSolver::Dense,
                    "sparse" => LinearSolver::Sparse,
                    other => {
                        return Err(format!(
                            "--ba-linear-solver must be dense or sparse, got {other:?}"
                        ))
                    }
                };
                a.remove(i + 1);
                ba_linear_solver = Some(parsed);
            }
            "--periodic-ba-min-registered-images" => {
                periodic_ba_min_registered_images =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
            }
            "--final-ba-polish-iterations" => {
                final_ba_polish_iterations = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
            }
            "--geometry-weighted-ba" => geometry_weighted_ba = true,
            "--freeze-ill-conditioned-landmarks" => freeze_ill_conditioned_landmarks = true,
            "--landmark-ba-warm-start-iterations" => {
                landmark_ba_warm_start_iterations =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
            }
            "--landmark-ba-warm-start-min-registered-images" => {
                landmark_ba_warm_start_min_registered_images =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
            }
            "--mapper" => {
                mapper = match a.remove(i + 1).as_str() {
                    "incremental" => MapperKind::Incremental,
                    "global" => MapperKind::Global,
                    "hybrid" => MapperKind::Hybrid,
                    other => {
                        return Err(format!(
                            "--mapper must be incremental|global|hybrid, got {other}"
                        ))
                    }
                };
            }
            "--chirality-harden" => chirality_harden = true,
            "--rotation-seed-trials" => {
                rotation_seed_trials = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--refine-global-translations" => refine_global_translations = true,
            "--global-independent-edge-scales" => global_independent_edge_scales = true,
            "--multi-hypothesis-edges" => multi_hypothesis_edges = true,
            "--min-edge-inliers" => {
                min_edge_inliers = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--min-edge-parallax-deg" => {
                min_edge_parallax_deg = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--weight-by-chirality-margin" => weight_by_chirality_margin = true,
            "--filter-images" => filter_images = true,
            "--colmap-verification" => verification_mode = VerificationMode::Full,
            "--verification-mode" => {
                verification_mode = a.remove(i + 1).parse().map_err(|e: String| e)?
            }
            "--track-source" => track_source = parse_track_source(&a.remove(i + 1))?,
            "--confidence-ordered-tracks" => confidence_ordered_tracks = true,
            "--geometric-confidence-tracks" => geometric_confidence_tracks = true,
            "--stable-track-order" => stable_track_order = true,
            "--cycle-supported-tracks" => cycle_supported_tracks = true,
            "--canonical-feature-order" => canonical_feature_order = true,
            "--union-traversal-order" => {
                union_traversal_order = a.remove(i + 1).parse().map_err(|e: String| e)?
            }
            "--geometry-guided-conflict-recovery" => geometry_guided_conflict_recovery = true,
            "--pose-guided-track-splitting" => pose_guided_track_splitting = true,
            "--pose-guided-track-splitting-graph-support" => {
                pose_guided_track_splitting_graph_support = true
            }
            "--pose-guided-track-splitting-bridge-cuts" => {
                pose_guided_track_splitting_bridge_cuts = true
            }
            "--pose-guided-split-max-reproj" => {
                pose_guided_split_max_reproj =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--pose-guided-track-splitting-iterations" => {
                pose_guided_track_splitting_iterations =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--pose-guided-track-merging" => pose_guided_track_merging = true,
            "--pose-guided-merge-max-reproj" => {
                pose_guided_merge_max_reproj =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--pair-source" => pair_source = a.remove(i + 1).parse().map_err(|e: String| e)?,
            "--vocab-tree-branching" => {
                vocab_tree_branching = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--vocab-tree-depth" => {
                vocab_tree_depth = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--vocab-tree-num-images" => {
                vocab_tree_num_images = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rescue-bridging" => rescue_bridging = true,
            "--rescue-match-ratio" => {
                rescue_match_ratio = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rescue-min-matches" => {
                rescue_min_matches = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rescue-max-candidates" => {
                rescue_max_candidates = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rescue-cross-check" => rescue_cross_check = true,
            "--diagnose-pair" => {
                let raw = a.remove(i + 1);
                let (lhs, rhs) = raw
                    .split_once(',')
                    .ok_or_else(|| format!("--diagnose-pair expects I,J, got {raw:?}"))?;
                let i_idx: usize = lhs.trim().parse().map_err(|e| format!("{e}"))?;
                let j_idx: usize = rhs.trim().parse().map_err(|e| format!("{e}"))?;
                diagnose_pairs.push((i_idx, j_idx));
            }
            "--diagnose-pairs-csv" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--diagnose-pairs-csv requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--diagnose-pairs-csv requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                diagnose_pairs_csv = Some(PathBuf::from(raw));
            }
            "--diagnose-pair-stems" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--diagnose-pair-stems requires STEM[,STEM…]")?
                    .clone();
                a.remove(i + 1);
                diagnose_pair_stems = parse_diagnose_stems(&raw)?;
            }
            "--matcher" => matcher = a.remove(i + 1).parse().map_err(|e: String| e)?,
            "--lightglue-model" => lightglue_model = Some(PathBuf::from(a.remove(i + 1))),
            "--onnx-backend" => onnx_backend = a.remove(i + 1),
            "--lightglue-max-keypoints" => {
                lightglue_max_keypoints = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--import-matches-file" => import_matches_file = Some(PathBuf::from(a.remove(i + 1))),
            "--import-matches-supplement-file" => {
                import_matches_supplement_file = Some(PathBuf::from(a.remove(i + 1)))
            }
            "--export-features-dir" => export_features_dir = Some(PathBuf::from(a.remove(i + 1))),
            "--export-features-only" => export_features_only = true,
            "--sift-stream-export" => sift_stream_export = true,
            "--sift-stream-resume" => sift_stream_resume = true,
            "--import-verified-pairs-file" => {
                import_verified_pairs_file = Some(PathBuf::from(a.remove(i + 1)))
            }
            "--export-verified-pairs-snapshot" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--export-verified-pairs-snapshot requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--export-verified-pairs-snapshot requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                export_verified_pairs_snapshot = Some(PathBuf::from(raw));
            }
            "--import-verified-pairs-snapshot" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--import-verified-pairs-snapshot requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--import-verified-pairs-snapshot requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                import_verified_pairs_snapshot = Some(PathBuf::from(raw));
            }
            "--snapshot-keypoints-only" => snapshot_keypoints_only = true,
            "--export-verified-pairs-only" => export_verified_pairs_only = true,
            "--persistent-match-worker-plan" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--persistent-match-worker-plan requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--persistent-match-worker-plan requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                persistent_match_worker_plan = Some(PathBuf::from(raw));
            }
            "--candidate-manifest" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--candidate-manifest requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--candidate-manifest requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                candidate_manifest = Some(PathBuf::from(raw));
            }
            "--export-candidate-manifest" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--export-candidate-manifest requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--export-candidate-manifest requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                export_candidate_manifest = Some(PathBuf::from(raw));
            }
            "--snapshot-coordinate-override-dir" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--snapshot-coordinate-override-dir requires DIR")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err(
                        "--snapshot-coordinate-override-dir requires a non-empty DIR".into(),
                    );
                }
                a.remove(i + 1);
                snapshot_coordinate_override_dir = Some(PathBuf::from(raw));
            }
            "--diagnose-ba-oracle-poses" => {
                diagnose_ba_oracle_poses_file = Some(PathBuf::from(a.remove(i + 1)))
            }
            "--diagnose-fixed-rotation-ba" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--diagnose-fixed-rotation-ba requires current or MODEL/images.txt")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--diagnose-fixed-rotation-ba requires a non-empty source".into());
                }
                a.remove(i + 1);
                diagnose_fixed_rotation_ba = Some(raw);
            }
            "--diagnose-model-score" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--diagnose-model-score requires MODEL/images.txt")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err(
                        "--diagnose-model-score requires a non-empty MODEL/images.txt".into(),
                    );
                }
                a.remove(i + 1);
                diagnose_model_score_file = Some(PathBuf::from(raw));
            }
            "--initial-poses" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--initial-poses requires MODEL/images.txt")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--initial-poses requires a non-empty MODEL/images.txt".into());
                }
                a.remove(i + 1);
                initial_poses_file = Some(PathBuf::from(raw));
            }
            "--diagnose-colmap-track-membership" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--diagnose-colmap-track-membership requires MODEL/points3D.txt")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err(
                        "--diagnose-colmap-track-membership requires a non-empty points3D.txt path"
                            .into(),
                    );
                }
                a.remove(i + 1);
                diagnose_colmap_track_membership = Some(PathBuf::from(raw));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    let camera = if input_colmap_calibration.is_some() {
        if width.is_some()
            || height.is_some()
            || fx.is_some()
            || fy.is_some()
            || cx.is_some()
            || cy.is_some()
        {
            return Err(
                "--input-colmap-calibration cannot be combined with --width/--height/--fx/--fy/--cx/--cy"
                    .into(),
            );
        }
        // The real reference camera is loaded after feature/image names are
        // known.  This finite placeholder keeps the parsed Args total and is
        // replaced before any matching, verification, or mapping work.
        Camera::pinhole(0, 1, 1, 1.0, 1.0, 0.0, 0.0)
    } else {
        let width = width.ok_or("--width is required")?;
        let height = height.ok_or("--height is required")?;
        Camera::pinhole(
            0,
            width,
            height,
            fx.ok_or("--fx is required")?,
            fy.ok_or("--fy is required")?,
            cx.ok_or("--cx is required")?,
            cy.ok_or("--cy is required")?,
        )
    };

    if !sift_descriptor_magnification.is_finite() || sift_descriptor_magnification <= 0.0 {
        return Err(format!(
            "--sift-descriptor-magnification must be finite and > 0, got {sift_descriptor_magnification}"
        ));
    }
    if colmap_guided_matching && !guided_matching {
        return Err(
            "--colmap-guided-matching requires --guided-matching so the guided pass is explicit"
                .into(),
        );
    }
    if sift_vlfeat_compatible_descriptor && sift_scale_adaptive_gradients {
        return Err(
            "--sift-vlfeat-compatible-descriptor already selects the complete scale-adaptive descriptor; remove --sift-scale-adaptive-gradients".into(),
        );
    }
    if sift_dsp && !sift_vlfeat_compatible_descriptor {
        return Err(
            "--sift-dsp requires --sift-vlfeat-compatible-descriptor so domain-size pooling uses the corrected VLFeat/COLMAP descriptor".into(),
        );
    }
    if sift_dsp && sift_dsp_num_scales == 0 {
        return Err("--sift-dsp requires at least one domain-size sample".into());
    }
    if sift_vlfeat_compatible_descriptor && sift_affine {
        return Err(
            "--sift-vlfeat-compatible-descriptor currently requires isotropic keypoints; remove --sift-affine".into(),
        );
    }
    if sift_vlfeat_compatible_detector && sift_affine {
        return Err(
            "--sift-vlfeat-compatible-detector currently requires isotropic keypoints; remove --sift-affine".into(),
        );
    }
    if sift_vlfeat_compatible_detector && sift_standard_orientations {
        return Err(
            "--sift-vlfeat-compatible-detector already selects VLFeat orientation peaks; remove --sift-standard-orientations".into(),
        );
    }
    if sift_vlfeat_bilinear_orientations && !sift_vlfeat_compatible_detector {
        return Err(
            "--sift-vlfeat-bilinear-orientations requires --sift-vlfeat-compatible-detector".into(),
        );
    }
    if sift_vlfeat_compatible_output_order && !sift_vlfeat_compatible_detector {
        return Err(
            "--sift-vlfeat-compatible-output-order requires --sift-vlfeat-compatible-detector"
                .into(),
        );
    }
    if sift_vlfeat_compatible_descriptor && sift_append_descriptor_magnification.is_some() {
        return Err(
            "--sift-vlfeat-compatible-descriptor cannot be combined with --sift-append-descriptor-magnification".into(),
        );
    }
    if sift_vlfeat_compatible_descriptor
        && sift_descriptor_magnification_explicit
        && (sift_descriptor_magnification - 3.0).abs() > 1e-12
    {
        return Err(
            "--sift-vlfeat-compatible-descriptor fixes descriptor magnification at 3.0; omit --sift-descriptor-magnification or use 3.0".into(),
        );
    }
    if sift_split_colmap_detector_grayscale
        && (!sift_vlfeat_compatible_detector || !sift_vlfeat_compatible_descriptor)
    {
        return Err(
            "--sift-split-colmap-detector-grayscale requires --sift-vlfeat-compatible-detector and --sift-vlfeat-compatible-descriptor".into(),
        );
    }
    if sift_split_colmap_detector_grayscale && sift_colmap_compatible_grayscale {
        return Err(
            "--sift-split-colmap-detector-grayscale cannot be combined with --sift-colmap-compatible-grayscale".into(),
        );
    }
    if initial_poses_file.is_some() && mapper != MapperKind::Incremental {
        return Err("--initial-poses currently requires --mapper incremental".into());
    }
    if global_independent_edge_scales && mapper != MapperKind::Global {
        return Err("--global-independent-edge-scales currently requires --mapper global".into());
    }
    if initial_poses_file.is_some() && seed_pair.is_some() {
        return Err("--initial-poses cannot be combined with --seed-pair".into());
    }
    if input_colmap_calibration.is_some() && (refine_intrinsics || refine_distortion) {
        return Err(
            "--input-colmap-calibration currently keeps per-image PINHOLE intrinsics fixed; remove --refine-intrinsics/--refine-distortion"
                .into(),
        );
    }
    if sift_stream_export && feature_extractor != FeatureExtractorKind::Sift {
        return Err("--sift-stream-export requires --feature-extractor sift".into());
    }
    if sift_stream_export && export_features_dir.is_none() {
        return Err("--sift-stream-export requires --export-features-dir DIR".into());
    }
    if sift_stream_export && !export_features_only {
        return Err(
            "--sift-stream-export requires --export-features-only so extracted banks are not retained in memory"
                .into(),
        );
    }
    if sift_stream_resume && !sift_stream_export {
        return Err("--sift-stream-resume requires --sift-stream-export".into());
    }
    if export_verified_pairs_only && export_verified_pairs_snapshot.is_none() {
        return Err(
            "--export-verified-pairs-only requires --export-verified-pairs-snapshot PATH".into(),
        );
    }
    if export_verified_pairs_only && import_verified_pairs_snapshot.is_some() {
        return Err(
            "--export-verified-pairs-only cannot be combined with --import-verified-pairs-snapshot"
                .into(),
        );
    }
    if candidate_manifest.is_some() && export_candidate_manifest.is_some() {
        return Err(
            "--candidate-manifest and --export-candidate-manifest are mutually exclusive".into(),
        );
    }
    if local_stem_window.is_some() && pair_source != PairSource::VladUnion {
        return Err("--local-stem-window requires --pair-source vlad-union".into());
    }
    if rig_local_grouping && pair_source != PairSource::VladUnion {
        return Err("--rig-local-grouping requires --pair-source vlad-union".into());
    }
    if pair_source == PairSource::TemporalPyramid
        && (local_stem_window.is_some() || rig_local_grouping)
    {
        return Err(
            "--pair-source temporal-pyramid owns rig grouping; do not combine it with --local-stem-window or --rig-local-grouping".into(),
        );
    }
    if pair_source == PairSource::VladUnion && local_stem_window.is_none() {
        return Err("--pair-source vlad-union requires --local-stem-window N".into());
    }
    if pair_source == PairSource::VladUnion && exhaustive {
        return Err("--pair-source vlad-union cannot be combined with --exhaustive".into());
    }
    if pair_source == PairSource::TemporalPyramid && exhaustive {
        return Err("--pair-source temporal-pyramid cannot be combined with --exhaustive".into());
    }
    if candidate_budget.is_some()
        && !matches!(
            pair_source,
            PairSource::VladUnion | PairSource::TemporalPyramid
        )
    {
        return Err(
            "--candidate-budget currently requires --pair-source vlad-union or temporal-pyramid"
                .into(),
        );
    }
    if pair_source == PairSource::VladUnion && pair_stem_window.is_some() {
        return Err(
            "--pair-source vlad-union uses --local-stem-window; do not combine it with --pair-stem-window".into(),
        );
    }
    if pair_source == PairSource::TemporalPyramid && pair_stem_window.is_some() {
        return Err(
            "--pair-source temporal-pyramid uses rig timestamps; do not combine it with --pair-stem-window".into(),
        );
    }
    if candidate_manifest.is_some()
        && (exhaustive
            || local_stem_window.is_some()
            || rig_local_grouping
            || candidate_budget.is_some()
            || pair_stem_window.is_some()
            || pair_source == PairSource::Transitive)
    {
        return Err(
            "--candidate-manifest cannot be combined with generated candidate filters or transitive expansion".into(),
        );
    }
    if export_candidate_manifest.is_some()
        && (import_matches_file.is_some()
            || import_matches_supplement_file.is_some()
            || import_verified_pairs_file.is_some()
            || import_verified_pairs_snapshot.is_some())
    {
        return Err(
            "--export-candidate-manifest cannot be combined with imported pair streams".into(),
        );
    }
    if diagnose_colmap_track_membership.is_some() && mapper != MapperKind::Incremental {
        return Err(
            "--diagnose-colmap-track-membership currently requires --mapper incremental".into(),
        );
    }
    if diagnose_colmap_track_membership.is_some()
        && feature_extractor != FeatureExtractorKind::Files
    {
        return Err(
            "--diagnose-colmap-track-membership currently requires --feature-extractor files"
                .into(),
        );
    }
    if diagnose_colmap_track_membership.is_some() && initial_poses_file.is_some() {
        return Err(
            "--diagnose-colmap-track-membership cannot be combined with --initial-poses".into(),
        );
    }
    if diagnose_colmap_track_membership.is_some() && colmap_style {
        return Err(
            "--diagnose-colmap-track-membership uses the plain incremental schedule and cannot be combined with --colmap-style".into(),
        );
    }
    if diagnose_colmap_track_membership.is_some()
        && (incremental_correspondence_triangulation
            || confidence_ordered_tracks
            || geometric_confidence_tracks
            || stable_track_order
            || cycle_supported_tracks
            || canonical_feature_order)
    {
        return Err(
            "--diagnose-colmap-track-membership cannot be combined with an alternate track strategy/order".into(),
        );
    }
    if pose_guided_track_splitting && mapper != MapperKind::Incremental {
        return Err("--pose-guided-track-splitting currently requires --mapper incremental".into());
    }
    if pose_guided_track_splitting_graph_support && !pose_guided_track_splitting {
        return Err(
            "--pose-guided-track-splitting-graph-support requires --pose-guided-track-splitting"
                .into(),
        );
    }
    if pose_guided_track_splitting_bridge_cuts && !pose_guided_track_splitting {
        return Err(
            "--pose-guided-track-splitting-bridge-cuts requires --pose-guided-track-splitting"
                .into(),
        );
    }
    if pose_guided_split_max_reproj.is_some() && !pose_guided_track_splitting {
        return Err("--pose-guided-split-max-reproj requires --pose-guided-track-splitting".into());
    }
    if pose_guided_track_merging && !pose_guided_track_splitting {
        return Err("--pose-guided-track-merging requires --pose-guided-track-splitting".into());
    }
    if pose_guided_merge_max_reproj.is_some() && !pose_guided_track_merging {
        return Err("--pose-guided-merge-max-reproj requires --pose-guided-track-merging".into());
    }
    if let Some(value) = pose_guided_merge_max_reproj {
        if !value.is_finite() || value <= 0.0 {
            return Err("--pose-guided-merge-max-reproj must be finite and positive".into());
        }
    }
    if let Some(value) = pose_guided_split_max_reproj {
        if !value.is_finite() || value <= 0.0 {
            return Err("--pose-guided-split-max-reproj must be finite and positive".into());
        }
    }
    if final_min_track_length.is_some() && !final_ba {
        return Err("--final-min-track-length requires final bundle adjustment".into());
    }
    if let Some(iterations) = pose_guided_track_splitting_iterations {
        if !(1..=8).contains(&iterations) {
            return Err("--pose-guided-track-splitting-iterations must be between 1 and 8".into());
        }
        if !pose_guided_track_splitting {
            return Err(
                "--pose-guided-track-splitting-iterations requires --pose-guided-track-splitting"
                    .into(),
            );
        }
    }
    if pose_guided_track_splitting && colmap_style {
        return Err(
            "--pose-guided-track-splitting uses the plain incremental schedule and cannot be combined with --colmap-style".into(),
        );
    }
    if pose_guided_track_splitting && diagnose_colmap_track_membership.is_some() {
        return Err(
            "--pose-guided-track-splitting cannot be combined with imported COLMAP track membership".into(),
        );
    }
    if pose_guided_track_splitting
        && (track_source != TrackSource::UnionFind
            || incremental_correspondence_triangulation
            || confidence_ordered_tracks
            || geometric_confidence_tracks
            || stable_track_order
            || cycle_supported_tracks
            || canonical_feature_order)
    {
        return Err(
            "--pose-guided-track-splitting requires the legacy union-find track builder without another track strategy flag".into(),
        );
    }
    if incremental_correspondence_triangulation && mapper != MapperKind::Incremental {
        return Err(
            "--incremental-correspondence-triangulation currently requires --mapper incremental"
                .into(),
        );
    }
    if incremental_correspondence_triangulation && colmap_style {
        return Err(
            "--incremental-correspondence-triangulation keeps the plain growth schedule and cannot be combined with --colmap-style".into(),
        );
    }
    if sequence_relative_pose_fallback && mapper != MapperKind::Incremental {
        return Err(
            "--sequence-relative-pose-fallback currently requires --mapper incremental".into(),
        );
    }
    if sequence_relative_pose_fallback && colmap_style {
        return Err(
            "--sequence-relative-pose-fallback currently requires the plain incremental schedule; remove --colmap-style".into(),
        );
    }
    if sequence_relative_pose_fallback && initial_poses_file.is_some() {
        return Err(
            "--sequence-relative-pose-fallback cannot be combined with --initial-poses".into(),
        );
    }
    if sequence_fallback_after_post && !sequence_relative_pose_fallback {
        return Err(
            "--sequence-fallback-after-post requires --sequence-relative-pose-fallback".into(),
        );
    }
    if sequence_fallback_after_post && !post_refinement_registration {
        return Err(
            "--sequence-fallback-after-post requires --post-refinement-registration".into(),
        );
    }
    if sequence_fallback_carry_scale && !sequence_relative_pose_fallback {
        return Err(
            "--sequence-fallback-carry-scale requires --sequence-relative-pose-fallback".into(),
        );
    }
    if sequence_fallback_carry_scale && !sequence_fallback_after_post {
        return Err(
            "--sequence-fallback-carry-scale requires --sequence-fallback-after-post".into(),
        );
    }
    if sequence_fallback_carry_scale && !sequence_relaxed_constant_velocity_scale {
        return Err(
            "--sequence-fallback-carry-scale requires --sequence-relaxed-constant-velocity-scale"
                .into(),
        );
    }
    if sequence_constant_velocity_scale && !sequence_relative_pose_fallback {
        return Err(
            "--sequence-constant-velocity-scale requires --sequence-relative-pose-fallback".into(),
        );
    }
    if sequence_relaxed_constant_velocity_scale && !sequence_relative_pose_fallback {
        return Err(
            "--sequence-relaxed-constant-velocity-scale requires --sequence-relative-pose-fallback"
                .into(),
        );
    }
    if sequence_constant_velocity_scale && sequence_relaxed_constant_velocity_scale {
        return Err(
            "--sequence-constant-velocity-scale and --sequence-relaxed-constant-velocity-scale are mutually exclusive".into(),
        );
    }
    if import_verified_pairs_file.is_some() && import_verified_pairs_snapshot.is_some() {
        return Err(
            "--import-verified-pairs-file and --import-verified-pairs-snapshot are mutually exclusive"
                .into(),
        );
    }
    if snapshot_coordinate_override_dir.is_some() && import_verified_pairs_snapshot.is_none() {
        return Err(
            "--snapshot-coordinate-override-dir requires --import-verified-pairs-snapshot".into(),
        );
    }
    if snapshot_coordinate_override_dir.is_some()
        && feature_extractor != FeatureExtractorKind::Files
    {
        return Err(
            "--snapshot-coordinate-override-dir currently requires --feature-extractor files"
                .into(),
        );
    }
    if snapshot_coordinate_override_dir.is_some() && export_verified_pairs_snapshot.is_some() {
        return Err(
            "--snapshot-coordinate-override-dir cannot be combined with --export-verified-pairs-snapshot"
                .into(),
        );
    }
    if snapshot_keypoints_only {
        if import_verified_pairs_snapshot.is_none() {
            return Err(
                "--snapshot-keypoints-only requires --import-verified-pairs-snapshot PATH".into(),
            );
        }
        if feature_extractor != FeatureExtractorKind::Files {
            return Err(
                "--snapshot-keypoints-only currently requires --feature-extractor files".into(),
            );
        }
        if mapper != MapperKind::Incremental || colmap_style {
            return Err(
                "--snapshot-keypoints-only currently requires the plain incremental mapper (remove --mapper global|hybrid and --colmap-style)".into(),
            );
        }
        if snapshot_coordinate_override_dir.is_some() {
            return Err(
                "--snapshot-keypoints-only cannot be combined with --snapshot-coordinate-override-dir".into(),
            );
        }
        if export_features_dir.is_some() || export_features_only {
            return Err("--snapshot-keypoints-only cannot be combined with feature export".into());
        }
        if export_verified_pairs_snapshot.is_some() {
            return Err(
                "--snapshot-keypoints-only cannot be combined with --export-verified-pairs-snapshot".into(),
            );
        }
        if canonical_feature_order || orientation_locus_canonicalization {
            return Err(
                "--snapshot-keypoints-only cannot be combined with feature-order or orientation-locus canonicalization".into(),
            );
        }
        if diagnose_model_score_file.is_some() {
            return Err(
                "--snapshot-keypoints-only cannot be combined with --diagnose-model-score".into(),
            );
        }
        if stable_track_order {
            return Err(
                "--snapshot-keypoints-only cannot be combined with --stable-track-order (descriptor tie-breaks require descriptor payloads)".into(),
            );
        }
    }
    if import_verified_pairs_snapshot.is_some() {
        if import_matches_file.is_some() || import_matches_supplement_file.is_some() {
            return Err(
                "--import-verified-pairs-snapshot cannot be combined with raw match imports".into(),
            );
        }
        if pair_stem_window.is_some()
            || local_stem_window.is_some()
            || rig_local_grouping
            || candidate_budget.is_some()
            || candidate_manifest.is_some()
            || export_candidate_manifest.is_some()
            || pair_source == PairSource::Transitive
        {
            return Err(
                "--import-verified-pairs-snapshot cannot generate, filter, or transitively expand pairs"
                    .into(),
            );
        }
        if !rematch_stems.is_empty()
            || rematch_free_vs_priors
            || rescue_bridging
            || orientation_locus_canonicalization
            || canonical_feature_order
            || union_traversal_order != UnionTraversalOrder::Original
        {
            return Err(
                "--import-verified-pairs-snapshot cannot be combined with pair-stream rematching, reordering, or canonicalization flags"
                    .into(),
            );
        }
        if !diagnose_pairs.is_empty() || diagnose_pairs_csv.is_some() {
            return Err(
                "--import-verified-pairs-snapshot cannot be combined with matching diagnostics"
                    .into(),
            );
        }
    }
    if import_verified_pairs_snapshot.is_some() && !next_image_policy_explicit {
        // Snapshot v1 records the historical Count-default replay semantics.
        // Keep an explicit user policy authoritative, but do not silently
        // change model bytes merely because the demo default is Auto.
        next_image_policy = NextImagePolicy::CorrespondenceCount;
    }
    validate_diagnose_options(
        diagnose_pairs_csv.as_deref(),
        &diagnose_pair_stems,
        &diagnose_pairs,
        None,
    )?;

    let parsed = Args {
        feature_extractor,
        features_dir: features_dir.unwrap_or_default(),
        hybrid_filter_priors,
        hybrid_prior_min_obs,
        hybrid_prior_max_reproj,
        hybrid_drop_prior_stems,
        hybrid_drop_inconsistent_priors,
        verify_registration_two_view,
        hybrid_rotation_priors_only,
        joint_global_positioning,
        calibrated_view_edges_only,
        images_dir,
        feature_suffix,
        image_suffix,
        out_colmap: out_colmap.ok_or("--out-colmap is required")?,
        input_colmap_calibration,
        camera,
        vocab_size,
        retrieval_topk,
        exhaustive,
        pair_stem_window,
        local_stem_window,
        rig_local_grouping,
        temporal_pyramid_max_offset,
        candidate_budget,
        match_ratio,
        min_matches,
        min_pnp_inliers,
        max_mapper_matches_per_pair,
        max_reproj,
        next_image_policy,
        final_ba,
        final_min_track_length,
        seed_trials,
        seed_pair,
        refine_intrinsics,
        refine_distortion,
        colmap_style,
        final_iterative_global_refinement,
        global_ba_max_refinements,
        post_refinement_registration,
        structureless_registration,
        pnp_max_iterations,
        ba_max_iterations,
        ba_huber_delta,
        ba_linear_solver,
        periodic_ba_min_registered_images,
        final_ba_polish_iterations,
        geometry_weighted_ba,
        freeze_ill_conditioned_landmarks,
        landmark_ba_warm_start_iterations,
        landmark_ba_warm_start_min_registered_images,
        mapper,
        chirality_harden,
        rotation_seed_trials,
        refine_global_translations,
        global_independent_edge_scales,
        multi_hypothesis_edges,
        min_edge_inliers,
        min_edge_parallax_deg,
        weight_by_chirality_margin,
        filter_images,
        verification_mode,
        guided_matching,
        colmap_guided_matching,
        multiple_models,
        min_e_f_inlier_ratio,
        calibrated_prefer_essential,
        refine_uncalibrated_f_to_essential,
        strict_uncalibrated_f_to_essential,
        calibrated_essential_primary,
        prefer_essential_inliers,
        prefer_essential_free_endpoints,
        prefer_essential_stems,
        prefer_essential_stem_clique,
        prefer_essential_pairs,
        require_essential_selected_edges,
        require_essential_stems,
        require_essential_min_e_inliers,
        rematch_stems,
        rematch_ratio,
        rematch_cross_check,
        rematch_guided,
        rematch_free_vs_priors,
        rematch_prefer_min_e_inliers,
        rematch_prefer_strong_stems,
        rematch_prefer_strong_min_e,
        rematch_tracks_use_essential,
        rematch_min_chirality_margin,
        rematch_prior_anchor,
        rematch_anchor_min_e_inliers,
        rematch_min_e_f_inlier_ratio,
        rematch_calibrated_prefer_essential,
        rematch_prior_ray_guided,
        rematch_prior_ray_min_rays,
        rematch_prior_ray_min_e_inliers,
        rematch_verification_mode,
        rematch_pose_guided_after_global,
        rematch_pose_guided_gt,
        essential_edge_weight_boost,
        force_essential_matches,
        force_essential_min_ef_ratio,
        force_essential_min_e_inliers,
        force_essential_uncalibrated_only,
        repnp_free_from_priors,
        repnp_free_min_corrs,
        repnp_seed_free_as_priors,
        repair_prior_edges,
        repair_free_edges_from_solved,
        repair_free_edges_only_flipped,
        repair_free_edges_stems,
        drop_free_edges_antipodal,
        prior_guided_free_chirality,
        metric_prior_chirality_edges,
        metric_prior_chirality_min_rays,
        diagnose_bearing_gt,
        diagnose_bearing_stems,
        gt_chirality_oracle,
        gt_chirality_oracle_path,
        rematch_max_gt_bearing_deg,
        rematch_gt_bearing_path,
        rematch_guided_max_error_px,
        rematch_guided_lowe_ratio,
        rematch_require_calibrated,
        rematch_max_mean_sampson,
        metric_prior_scale,
        sequence_relative_pose_fallback,
        sequence_fallback_after_post,
        sequence_constant_velocity_scale,
        sequence_relaxed_constant_velocity_scale,
        sequence_fallback_carry_scale,
        track_source,
        confidence_ordered_tracks,
        geometric_confidence_tracks,
        stable_track_order,
        cycle_supported_tracks,
        canonical_feature_order,
        union_traversal_order,
        geometry_guided_conflict_recovery,
        pose_guided_track_splitting,
        pose_guided_track_splitting_graph_support,
        pose_guided_track_splitting_bridge_cuts,
        pose_guided_split_max_reproj,
        pose_guided_track_splitting_iterations,
        pose_guided_track_merging,
        pose_guided_merge_max_reproj,
        pair_source,
        candidate_manifest,
        export_candidate_manifest,
        vocab_tree_branching,
        vocab_tree_depth,
        vocab_tree_num_images,
        rescue_bridging,
        rescue_match_ratio,
        rescue_min_matches,
        rescue_max_candidates,
        rescue_cross_check,
        diagnose_pairs,
        diagnose_pairs_csv,
        diagnose_pair_stems,
        matcher,
        lightglue_model,
        onnx_backend,
        lightglue_max_keypoints,
        sift_max_keypoints,
        sift_affine,
        sift_detector,
        sift_multi_anisotropy,
        sift_dsp,
        sift_dsp_num_scales,
        sift_l1_root,
        sift_max_orientations,
        sift_standard_orientations,
        sift_prefer_larger_scale,
        sift_full_pyramid,
        sift_contrast_threshold,
        sift_descriptor_magnification,
        sift_scale_adaptive_gradients,
        sift_vlfeat_compatible_descriptor,
        sift_vlfeat_compatible_detector,
        sift_vlfeat_bilinear_orientations,
        sift_vlfeat_compatible_output_order,
        sift_colmap_compatible_grayscale,
        sift_split_colmap_detector_grayscale,
        sift_append_descriptor_magnification,
        sift_extra_keypoints_stems,
        sift_extra_keypoints,
        sift_extra_contrast_threshold,
        sift_extra_matches_append_only,
        incremental_correspondence_triangulation,
        orientation_locus_canonicalization,
        import_matches_file,
        import_matches_supplement_file,
        export_features_dir,
        export_features_only,
        sift_stream_export,
        sift_stream_resume,
        import_verified_pairs_file,
        export_verified_pairs_snapshot,
        import_verified_pairs_snapshot,
        snapshot_keypoints_only,
        export_verified_pairs_only,
        persistent_match_worker_plan,
        snapshot_coordinate_override_dir,
        diagnose_ba_oracle_poses_file,
        diagnose_fixed_rotation_ba,
        diagnose_model_score_file,
        initial_poses_file,
        diagnose_colmap_track_membership,
    };
    validate_persistent_match_worker_args(&parsed)?;
    Ok(parsed)
}

/// Keep the plan-driven worker as a narrow, reproducible matching path.  The
/// worker exits before mapper construction, but accepting mapper/diagnostic
/// switches here would make a typo look like a successful persistent A/B.
fn validate_persistent_match_worker_args(args: &Args) -> Result<(), String> {
    if args.persistent_match_worker_plan.is_none() {
        return Ok(());
    }
    if args.feature_extractor != FeatureExtractorKind::Files {
        return Err(
            "--persistent-match-worker-plan currently requires --feature-extractor files".into(),
        );
    }
    if args.input_colmap_calibration.is_none() {
        return Err(
            "--persistent-match-worker-plan requires --input-colmap-calibration for the frozen per-image camera contract".into(),
        );
    }
    if args.mapper != MapperKind::Incremental || args.colmap_style {
        return Err(
            "--persistent-match-worker-plan currently requires the plain incremental mapper (remove --mapper global|hybrid and --colmap-style)".into(),
        );
    }
    if args.matcher != MatcherKind::Nn {
        return Err("--persistent-match-worker-plan currently requires --matcher nn".into());
    }
    if args.verification_mode != VerificationMode::Full {
        return Err(
            "--persistent-match-worker-plan currently requires --verification-mode full".into(),
        );
    }
    if args.persistent_match_worker_plan.is_some()
        && (args.candidate_manifest.is_some()
            || args.export_candidate_manifest.is_some()
            || args.exhaustive
            || args.pair_stem_window.is_some()
            || args.local_stem_window.is_some()
            || args.rig_local_grouping
            || args.candidate_budget.is_some()
            || args.pair_source != PairSource::Vlad
            || args.retrieval_topk != 12
            || args.vocab_size != 64
            || args.vocab_tree_branching != 10
            || args.vocab_tree_depth != 3
            || args.vocab_tree_num_images != 100)
    {
        return Err(
            "--persistent-match-worker-plan owns the candidate shard schedule; remove candidate generation/filter flags".into(),
        );
    }
    if args.export_verified_pairs_snapshot.is_some()
        || args.export_verified_pairs_only
        || args.import_verified_pairs_file.is_some()
        || args.import_verified_pairs_snapshot.is_some()
        || args.snapshot_keypoints_only
        || args.snapshot_coordinate_override_dir.is_some()
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with snapshot/raw-pair import or export modes".into(),
        );
    }
    if args.import_matches_file.is_some() || args.import_matches_supplement_file.is_some() {
        return Err(
            "--persistent-match-worker-plan cannot be combined with raw match imports".into(),
        );
    }
    if args.export_features_dir.is_some()
        || args.export_features_only
        || args.sift_stream_export
        || args.sift_stream_resume
        || args.incremental_correspondence_triangulation
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with feature export or alternate incremental modes".into(),
        );
    }
    if args.guided_matching
        || args.colmap_guided_matching
        || args.multiple_models
        || args.min_e_f_inlier_ratio.is_some()
        || args.calibrated_prefer_essential
        || args.refine_uncalibrated_f_to_essential
        || args.strict_uncalibrated_f_to_essential
        || args.calibrated_essential_primary
        || args.force_essential_matches
        || args.force_essential_uncalibrated_only
        || args.prefer_essential_inliers
        || args.prefer_essential_free_endpoints
        || !args.prefer_essential_stems.is_empty()
        || args.prefer_essential_stem_clique
        || !args.prefer_essential_pairs.is_empty()
        || args.require_essential_selected_edges
        || !args.require_essential_stems.is_empty()
        || args.require_essential_min_e_inliers != 0
        || args.essential_edge_weight_boost != 1.0
    {
        return Err(
            "--persistent-match-worker-plan currently supports only the frozen simple NN/full verifier settings".into(),
        );
    }
    if args.sift_append_descriptor_magnification.is_some()
        || args.sift_extra_matches_append_only
        || args.canonical_feature_order
        || args.orientation_locus_canonicalization
        || args.union_traversal_order != UnionTraversalOrder::Original
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with alternate descriptor, feature-order, or union-traversal modes".into(),
        );
    }
    if args.rescue_bridging
        || args.rescue_cross_check
        || args.rescue_match_ratio != 0.95
        || args.rescue_min_matches != 15
        || args.rescue_max_candidates != 200
        || args.sequence_relative_pose_fallback
        || args.sequence_fallback_after_post
        || args.sequence_constant_velocity_scale
        || args.sequence_relaxed_constant_velocity_scale
        || args.sequence_fallback_carry_scale
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with rematch, rescue, or sequence-fallback modes".into(),
        );
    }
    if !args.rematch_stems.is_empty()
        || args.rematch_ratio != 0.9
        || !args.rematch_cross_check
        || args.rematch_guided
        || args.rematch_free_vs_priors
        || args.rematch_prefer_min_e_inliers != 0
        || !args.rematch_prefer_strong_stems.is_empty()
        || args.rematch_tracks_use_essential
        || args.rematch_min_chirality_margin != 0.0
        || args.rematch_prior_anchor
        || args.rematch_min_e_f_inlier_ratio.is_some()
        || args.rematch_calibrated_prefer_essential
        || args.rematch_prior_ray_guided
        || args.rematch_prior_ray_min_rays != 2
        || args.rematch_prior_ray_min_e_inliers != 25
        || args.rematch_anchor_min_e_inliers != 25
        || args.rematch_prefer_strong_min_e != 50
        || args.rematch_verification_mode.is_some()
        || args.rematch_pose_guided_after_global
        || args.rematch_pose_guided_gt.is_some()
        || args.rematch_max_gt_bearing_deg != 0.0
        || args.rematch_gt_bearing_path.is_some()
        || args.rematch_guided_max_error_px.is_some()
        || args.rematch_guided_lowe_ratio.is_some()
        || args.rematch_require_calibrated
        || args.rematch_max_mean_sampson != 0.0
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with rematch configuration".into(),
        );
    }
    if args.diagnose_ba_oracle_poses_file.is_some()
        || args.diagnose_fixed_rotation_ba.is_some()
        || args.diagnose_model_score_file.is_some()
        || args.initial_poses_file.is_some()
        || args.diagnose_colmap_track_membership.is_some()
        || !args.diagnose_pairs.is_empty()
        || args.diagnose_pairs_csv.is_some()
        || !args.diagnose_pair_stems.is_empty()
        || args.diagnose_bearing_gt.is_some()
        || !args.diagnose_bearing_stems.is_empty()
        || args.gt_chirality_oracle
        || args.gt_chirality_oracle_path.is_some()
        || args.rematch_gt_bearing_path.is_some()
        || args.rematch_pose_guided_gt.is_some()
    {
        return Err(
            "--persistent-match-worker-plan cannot consume GT/oracle or matching diagnostic inputs"
                .into(),
        );
    }
    if args.track_source != TrackSource::UnionFind
        || args.confidence_ordered_tracks
        || args.geometric_confidence_tracks
        || args.stable_track_order
        || args.cycle_supported_tracks
        || args.geometry_guided_conflict_recovery
        || args.pose_guided_track_splitting
        || args.pose_guided_track_merging
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with alternate mapper track/recovery modes".into(),
        );
    }
    if args.final_min_track_length.is_some()
        || args.seed_pair.is_some()
        || args.seed_trials != 12
        || !args.final_ba
        || args.refine_intrinsics
        || args.refine_distortion
        || args.final_iterative_global_refinement
        || args.global_ba_max_refinements.is_some()
        || args.post_refinement_registration
        || args.structureless_registration
        || args.pnp_max_iterations != 128
        || args.ba_max_iterations.is_some()
        || args.ba_huber_delta.is_some()
        || args.ba_linear_solver.is_some()
        || args.periodic_ba_min_registered_images != 0
        || args.final_ba_polish_iterations != 0
        || args.geometry_weighted_ba
        || args.freeze_ill_conditioned_landmarks
        || args.landmark_ba_warm_start_iterations != 0
        || args.landmark_ba_warm_start_min_registered_images != 0
        || args.filter_images
        || args.min_pnp_inliers != 12
        || args.max_mapper_matches_per_pair.is_some()
        || args.max_reproj != 4.0
        || args.next_image_policy != NextImagePolicy::Auto
        || args.chirality_harden
        || args.rotation_seed_trials != 1
        || args.refine_global_translations
        || args.global_independent_edge_scales
        || args.multi_hypothesis_edges
        || args.min_edge_inliers != 15
        || args.min_edge_parallax_deg != 2.0
        || args.weight_by_chirality_margin
        || args.hybrid_filter_priors
        || args.hybrid_drop_inconsistent_priors
        || !args.hybrid_drop_prior_stems.is_empty()
        || args.verify_registration_two_view
        || args.hybrid_rotation_priors_only
        || args.joint_global_positioning
        || args.calibrated_view_edges_only
        || args.pose_guided_track_splitting
        || args.pose_guided_track_splitting_graph_support
        || args.pose_guided_track_splitting_bridge_cuts
        || args.pose_guided_split_max_reproj.is_some()
        || args.pose_guided_track_splitting_iterations.is_some()
        || args.pose_guided_track_merging
        || args.pose_guided_merge_max_reproj.is_some()
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with mapper/refinement options"
                .into(),
        );
    }
    if args.lightglue_model.is_some()
        || args.onnx_backend != "auto"
        || args.lightglue_max_keypoints != 0
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with learned-matcher options".into(),
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MapperKind {
    Incremental,
    Global,
    /// Incremental first, then global with those poses as absolute priors.
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum FeatureExtractorKind {
    Files,
    Sift,
}

/// Per-row SIFT metadata retained only for the opt-in orientation-locus
/// canonicalizer.  `FeatureSet` deliberately remains the public, compact
/// `(x,y)+descriptor` representation used by every mapper path.  A detector
/// extremum that emitted several orientation rows has identical `(x,y,scale)`
/// metadata; rows with different scales remain different loci.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FeatureLocusMetadata {
    x: f64,
    y: f64,
    scale: f64,
    orientation: f64,
}

/// Quantized physical identity of one detector locus.  The key is derived
/// from metadata rather than the source row index, so it survives feature
/// permutations and feature-file round trips.  Orientation is intentionally
/// absent: orientation copies are alternatives of one locus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct FeatureLocusKey {
    x: i64,
    y: i64,
    scale: i64,
}

const FEATURE_LOCUS_COORD_SCALE: f64 = 1_000_000.0;

fn feature_locus_key(metadata: FeatureLocusMetadata) -> Option<FeatureLocusKey> {
    let quantize = |value: f64| {
        let scaled = value * FEATURE_LOCUS_COORD_SCALE;
        (scaled.is_finite() && scaled >= i64::MIN as f64 && scaled <= i64::MAX as f64)
            .then(|| scaled.round() as i64)
    };
    Some(FeatureLocusKey {
        x: quantize(metadata.x)?,
        y: quantize(metadata.y)?,
        scale: quantize(metadata.scale.abs())?,
    })
}

/// Extract SIFT features for one image path.
///
/// When `extra_keypoints > 0`, extracts a primary set at `max_keypoints` then
/// appends spatially novel survivors from a denser extraction so the primary
/// contrast-ranked prefix stays identical to a plain `--sift-max-keypoints`
/// run (load-bearing for courtyard hub edge `10-11`).
#[cfg(feature = "image-io")]
fn extract_sift_for_image(
    path: &Path,
    max_keypoints: usize,
    affine: bool,
    detector: &str,
    multi_anisotropy: bool,
    dsp: bool,
    dsp_num_scales: usize,
    l1_root: bool,
    max_orientations: usize,
    standard_orientations: bool,
    prefer_larger_scale: bool,
    full_pyramid: bool,
    contrast_threshold: f64,
    descriptor_magnification: f64,
    scale_adaptive_gradients: bool,
    vlfeat_compatible_descriptor: bool,
    vlfeat_compatible_detector: bool,
    vlfeat_bilinear_orientations: bool,
    vlfeat_compatible_output_order: bool,
    colmap_compatible_grayscale: bool,
    split_colmap_detector_grayscale: bool,
    append_descriptor_magnification: Option<f64>,
    extra_keypoints: usize,
    extra_contrast_threshold: Option<f64>,
    expected_dimensions: Option<(u32, u32)>,
) -> Result<
    (
        FeatureSet,
        usize,
        Option<Vec<Vec<f32>>>,
        Vec<FeatureLocusMetadata>,
    ),
    Box<dyn std::error::Error>,
> {
    use visloc_rs::vision::features::sift::{
        describe_sift_keypoints, extract_sift, GrayImage, SiftConfig, SiftDetector,
        SiftNormalization,
    };
    let detector_grayscale = if colmap_compatible_grayscale || split_colmap_detector_grayscale {
        visloc_io::images::read_common_image_colmap_grayscale(path)?
    } else {
        visloc_io::images::read_common_image(path)?
    };
    if let Some((expected_width, expected_height)) = expected_dimensions {
        let actual = (
            detector_grayscale.width() as u32,
            detector_grayscale.height() as u32,
        );
        if actual != (expected_width, expected_height) {
            return Err(format!(
                "source image {path:?} dimensions {}x{} do not match calibration {}x{}",
                actual.0, actual.1, expected_width, expected_height
            )
            .into());
        }
    }
    let descriptor_grayscale = if split_colmap_detector_grayscale {
        Some(visloc_io::images::read_common_image(path)?)
    } else {
        None
    };
    let image = GrayImage::new(
        detector_grayscale.width(),
        detector_grayscale.height(),
        detector_grayscale.pixels(),
    )?;
    let descriptor_image = if let Some(grayscale) = descriptor_grayscale.as_ref() {
        GrayImage::new(grayscale.width(), grayscale.height(), grayscale.pixels())?
    } else {
        GrayImage::new(image.width, image.height, image.pixels)?
    };
    let detector = match detector {
        "dog" => SiftDetector::Dog,
        "hessian-laplace" | "hessian" => SiftDetector::HessianLaplace,
        other => {
            return Err(format!("unknown --sift-detector {other} (dog|hessian-laplace)").into())
        }
    };
    let make_cfg = |cap: usize, threshold: f64| SiftConfig {
        max_keypoints: cap,
        affine,
        detector,
        multi_anisotropy: multi_anisotropy && affine,
        domain_size_pooling: dsp,
        dsp_num_scales: if dsp { dsp_num_scales.max(1) } else { 15 },
        normalization: if l1_root {
            SiftNormalization::L1Root
        } else {
            SiftNormalization::L2
        },
        max_orientations,
        standard_orientation_peaks: standard_orientations,
        prefer_larger_scale,
        full_pyramid,
        descriptor_magnification,
        scale_adaptive_gradients,
        vlfeat_compatible_descriptor,
        vlfeat_compatible_detector,
        vlfeat_bilinear_orientations,
        vlfeat_compatible_output_order,
        contrast_threshold: threshold,
        ..SiftConfig::default()
    };
    let primary_config = make_cfg(max_keypoints, contrast_threshold);
    let (mut keypoints, mut descriptors) = if split_colmap_detector_grayscale {
        extract_sift_with_split_grayscale(&image, &descriptor_image, &primary_config)?
    } else {
        extract_sift(&image, &primary_config)?
    };
    let primary_keypoint_count = keypoints.len();
    if extra_keypoints > 0 {
        let dense_cap =
            max_keypoints.saturating_add(extra_keypoints.saturating_mul(2).max(extra_keypoints));
        let dense_threshold =
            effective_extra_contrast_threshold(extra_contrast_threshold, contrast_threshold);
        let (dense_kp, dense_desc) = extract_sift(&image, &make_cfg(dense_cap, dense_threshold))?;
        append_spatially_novel_keypoints(
            &mut keypoints,
            &mut descriptors,
            dense_kp,
            dense_desc,
            extra_keypoints,
        );
        if split_colmap_detector_grayscale {
            descriptors = describe_sift_keypoints(
                &descriptor_image,
                &keypoints,
                &make_cfg(keypoints.len(), contrast_threshold),
            );
        }
    }
    let alternate_descriptors = append_descriptor_magnification.map(|magnification| {
        let alternate_config = SiftConfig {
            descriptor_magnification: magnification,
            max_keypoints: keypoints.len(),
            ..make_cfg(keypoints.len(), contrast_threshold)
        };
        let descriptor_source = if split_colmap_detector_grayscale {
            &descriptor_image
        } else {
            &image
        };
        let descriptors = describe_sift_keypoints(descriptor_source, &keypoints, &alternate_config);
        assert_eq!(
            descriptors.len(),
            keypoints.len(),
            "alternate descriptor bank must preserve keypoint indices"
        );
        descriptors
    });
    let features = FeatureSet::new(
        keypoints.iter().map(|k| Point2::new(k.x, k.y)).collect(),
        descriptors,
    )?;
    let locus_metadata = keypoints
        .iter()
        .map(|keypoint| FeatureLocusMetadata {
            x: keypoint.x,
            y: keypoint.y,
            scale: keypoint.sigma,
            orientation: keypoint.orientation,
        })
        .collect();
    Ok((
        features,
        primary_keypoint_count,
        alternate_descriptors,
        locus_metadata,
    ))
}

/// Detect keypoints on one grayscale image and describe those exact
/// keypoints on another.  The split is intentionally narrow: it is used only
/// by the opt-in COLMAP preprocessing experiment, and never redetects or
/// changes keypoint order while switching the descriptor source.
#[cfg(feature = "image-io")]
fn extract_sift_with_split_grayscale(
    detector_image: &GrayImage<'_>,
    descriptor_image: &GrayImage<'_>,
    config: &SiftConfig,
) -> Result<(Vec<SiftKeypoint>, Vec<Vec<f32>>), SiftError> {
    let (keypoints, _) = extract_sift(detector_image, config)?;
    let descriptors = describe_sift_keypoints(descriptor_image, &keypoints, config);
    Ok((keypoints, descriptors))
}

/// Resolve the optional extra-extraction threshold without changing the
/// legacy path: absent means exactly the primary SIFT threshold.
#[cfg(feature = "image-io")]
fn effective_extra_contrast_threshold(extra: Option<f64>, primary: f64) -> f64 {
    extra.unwrap_or(primary)
}

/// Append at most `max_extra` dense SIFT detections that are novel on the
/// existing 0.5 px spatial grid. The primary vectors are mutated only by
/// appending, so their prefix remains byte-identical to the primary extraction.
#[cfg(feature = "image-io")]
fn append_spatially_novel_keypoints(
    keypoints: &mut Vec<visloc_rs::vision::features::sift::SiftKeypoint>,
    descriptors: &mut Vec<Vec<f32>>,
    dense_keypoints: Vec<visloc_rs::vision::features::sift::SiftKeypoint>,
    dense_descriptors: Vec<Vec<f32>>,
    max_extra: usize,
) -> usize {
    let mut seen: HashSet<(i32, i32)> = keypoints
        .iter()
        .map(|k| ((k.x * 2.0).round() as i32, (k.y * 2.0).round() as i32))
        .collect();
    let mut added = 0usize;
    for (keypoint, descriptor) in dense_keypoints
        .into_iter()
        .zip(dense_descriptors.into_iter())
    {
        if added >= max_extra {
            break;
        }
        let key = (
            (keypoint.x * 2.0).round() as i32,
            (keypoint.y * 2.0).round() as i32,
        );
        if seen.insert(key) {
            keypoints.push(keypoint);
            descriptors.push(descriptor);
            added += 1;
        }
    }
    added
}

/// Enumerate supported SIFT source images in the same lexical order used by
/// the historical batch extractor.  Keeping this as a shared helper makes the
/// streaming export byte-comparable with the ordinary path.
#[cfg(feature = "image-io")]
fn list_sift_image_paths(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| {
                    matches!(
                        extension.to_ascii_lowercase().as_str(),
                        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp"
                    )
                })
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// In-process SIFT over every common-format image in `dir`, sorted lexically.
#[cfg(not(feature = "image-io"))]
fn load_images_with_sift(
    _dir: &Path,
    _max_keypoints: usize,
    _affine: bool,
    _detector: &str,
    _multi_anisotropy: bool,
    _dsp: bool,
    _dsp_num_scales: usize,
    _l1_root: bool,
    _max_orientations: usize,
    _standard_orientations: bool,
    _prefer_larger_scale: bool,
    _full_pyramid: bool,
    _contrast_threshold: f64,
    _descriptor_magnification: f64,
    _scale_adaptive_gradients: bool,
    _vlfeat_compatible_descriptor: bool,
    _vlfeat_compatible_detector: bool,
    _vlfeat_bilinear_orientations: bool,
    _vlfeat_compatible_output_order: bool,
    _colmap_compatible_grayscale: bool,
    _split_colmap_detector_grayscale: bool,
    _append_descriptor_magnification: Option<f64>,
    _extra_stems: &[String],
    _extra_keypoints: usize,
    _extra_contrast_threshold: Option<f64>,
) -> Result<
    (
        Vec<FeatureSet>,
        Vec<String>,
        Vec<usize>,
        Vec<Option<Vec<Vec<f32>>>>,
        Vec<Option<Vec<FeatureLocusMetadata>>>,
    ),
    Box<dyn std::error::Error>,
> {
    Err("--feature-extractor sift requires building with --features image-io".into())
}

/// In-process SIFT over every common-format image in `dir`, sorted lexically.
#[cfg(feature = "image-io")]
fn load_images_with_sift(
    dir: &Path,
    max_keypoints: usize,
    affine: bool,
    detector: &str,
    multi_anisotropy: bool,
    dsp: bool,
    dsp_num_scales: usize,
    l1_root: bool,
    max_orientations: usize,
    standard_orientations: bool,
    prefer_larger_scale: bool,
    full_pyramid: bool,
    contrast_threshold: f64,
    descriptor_magnification: f64,
    scale_adaptive_gradients: bool,
    vlfeat_compatible_descriptor: bool,
    vlfeat_compatible_detector: bool,
    vlfeat_bilinear_orientations: bool,
    vlfeat_compatible_output_order: bool,
    colmap_compatible_grayscale: bool,
    split_colmap_detector_grayscale: bool,
    append_descriptor_magnification: Option<f64>,
    extra_stems: &[String],
    extra_keypoints: usize,
    extra_contrast_threshold: Option<f64>,
) -> Result<
    (
        Vec<FeatureSet>,
        Vec<String>,
        Vec<usize>,
        Vec<Option<Vec<Vec<f32>>>>,
        Vec<Option<Vec<FeatureLocusMetadata>>>,
    ),
    Box<dyn std::error::Error>,
> {
    let paths = list_sift_image_paths(dir)?;
    let total = paths.len();
    let extra_want: HashSet<&str> = extra_stems.iter().map(String::as_str).collect();
    eprintln!(
        "sift: extracting {total} image(s) (dsp={dsp}, dsp_scales={}, l1_root={l1_root}, contrast={contrast_threshold}, vlfeat_detector={vlfeat_compatible_detector}, colmap_gray={colmap_compatible_grayscale}, split_colmap_gray={split_colmap_detector_grayscale}, extra_kp={extra_keypoints} stems={extra_stems:?})",
        if dsp { dsp_num_scales } else { 0 }
    );
    let results: Result<Vec<_>, Box<dyn std::error::Error + Send>> = paths
        .par_iter()
        .enumerate()
        .map(|(idx, path)| {
            let started = std::time::Instant::now();
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let extra = if extra_keypoints > 0 && extra_want.contains(stem) {
                extra_keypoints
            } else {
                0
            };
            let (feat, primary_keypoint_count, alternate_descriptors, locus_metadata) =
                extract_sift_for_image(
                    path,
                    max_keypoints,
                    affine,
                    detector,
                    multi_anisotropy,
                    dsp,
                    dsp_num_scales,
                    l1_root,
                    max_orientations,
                    standard_orientations,
                    prefer_larger_scale,
                    full_pyramid,
                    contrast_threshold,
                    descriptor_magnification,
                    scale_adaptive_gradients,
                    vlfeat_compatible_descriptor,
                    vlfeat_compatible_detector,
                    vlfeat_bilinear_orientations,
                    vlfeat_compatible_output_order,
                    colmap_compatible_grayscale,
                    split_colmap_detector_grayscale,
                    append_descriptor_magnification,
                    extra,
                    extra_contrast_threshold,
                    None,
                )
                .map_err(|e| -> Box<dyn std::error::Error + Send> {
                    Box::new(std::io::Error::other(e.to_string()))
                })?;
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("image")
                .to_string();
            eprintln!(
                "sift: [{}/{}] {} -> {} kp (primary {}) in {:.1}s",
                idx + 1,
                total,
                name,
                feat.keypoints.len(),
                primary_keypoint_count,
                started.elapsed().as_secs_f64()
            );
            Ok((
                feat,
                name,
                primary_keypoint_count,
                alternate_descriptors,
                locus_metadata,
            ))
        })
        .collect();
    let results = results.map_err(|e| -> Box<dyn std::error::Error> { e })?;
    let mut features = Vec::with_capacity(results.len());
    let mut names = Vec::with_capacity(results.len());
    let mut primary_keypoint_counts = Vec::with_capacity(results.len());
    let mut alternate_descriptors = Vec::with_capacity(results.len());
    let mut locus_metadata = Vec::with_capacity(results.len());
    for (feature, name, primary_keypoint_count, alternate, loci) in results {
        features.push(feature);
        names.push(name);
        primary_keypoint_counts.push(primary_keypoint_count);
        alternate_descriptors.push(alternate);
        locus_metadata.push(Some(loci));
    }
    Ok((
        features,
        names,
        primary_keypoint_counts,
        alternate_descriptors,
        locus_metadata,
    ))
}

/// Extract and export SIFT one image at a time.  This intentionally shares
/// the same per-image extractor and serializers as the batch path, but avoids
/// Rayon and does not retain completed source images or descriptor banks.
#[cfg(feature = "image-io")]
fn stream_export_images_with_sift(
    dir: &Path,
    output_dir: &Path,
    calibration_model_dir: Option<&Path>,
    max_keypoints: usize,
    affine: bool,
    detector: &str,
    multi_anisotropy: bool,
    dsp: bool,
    dsp_num_scales: usize,
    l1_root: bool,
    max_orientations: usize,
    standard_orientations: bool,
    prefer_larger_scale: bool,
    full_pyramid: bool,
    contrast_threshold: f64,
    descriptor_magnification: f64,
    scale_adaptive_gradients: bool,
    vlfeat_compatible_descriptor: bool,
    vlfeat_compatible_detector: bool,
    vlfeat_bilinear_orientations: bool,
    vlfeat_compatible_output_order: bool,
    colmap_compatible_grayscale: bool,
    split_colmap_detector_grayscale: bool,
    append_descriptor_magnification: Option<f64>,
    extra_stems: &[String],
    extra_keypoints: usize,
    extra_contrast_threshold: Option<f64>,
    resume: bool,
) -> Result<usize, Box<dyn std::error::Error>> {
    let paths = list_sift_image_paths(dir)?;
    let image_names = paths
        .iter()
        .map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("source image {path:?} has no UTF-8 filename"))
                .map(str::to_owned)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let calibration = calibration_model_dir
        .map(|model_dir| resolve_input_colmap_calibration(model_dir, &image_names))
        .transpose()
        .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
    let extra_want: HashSet<&str> = extra_stems.iter().map(String::as_str).collect();
    eprintln!(
        "sift-stream: extracting {} image(s) (output={output_dir:?}, calibration={}, resume={resume}, dsp={dsp}, contrast={contrast_threshold}, vlfeat_detector={vlfeat_compatible_detector})",
        image_names.len(),
        calibration.is_some(),
    );
    let mut stems = HashSet::new();
    for name in &image_names {
        let stem = image_stem(name);
        if !stems.insert(stem.to_owned()) {
            return Err(format!("duplicate source image stem {stem:?}").into());
        }
    }
    std::fs::create_dir_all(output_dir)?;
    let mut total_keypoints = 0usize;
    for (index, path) in paths.iter().enumerate() {
        let stem = image_stem(&image_names[index]);
        let extra = if extra_keypoints > 0 && extra_want.contains(stem) {
            extra_keypoints
        } else {
            0
        };
        let expected_dimensions = calibration.as_ref().map(|loaded| {
            let camera = &loaded.native_cameras[index];
            (camera.width, camera.height)
        });
        let expected_camera = calibration
            .as_ref()
            .map(|loaded| &loaded.native_cameras[index]);
        let config_hash = sift_stream_config_hash(
            &image_names[index],
            max_keypoints,
            affine,
            detector,
            multi_anisotropy,
            dsp,
            dsp_num_scales,
            l1_root,
            max_orientations,
            standard_orientations,
            prefer_larger_scale,
            full_pyramid,
            contrast_threshold,
            descriptor_magnification,
            scale_adaptive_gradients,
            vlfeat_compatible_descriptor,
            vlfeat_compatible_detector,
            vlfeat_bilinear_orientations,
            vlfeat_compatible_output_order,
            colmap_compatible_grayscale,
            split_colmap_detector_grayscale,
            append_descriptor_magnification,
            extra,
            extra_contrast_threshold,
            expected_dimensions,
            expected_camera,
        );
        let feature_path = output_dir.join(format!("{stem}_features.txt"));
        let metadata_path = output_dir.join(format!("{stem}_loci.txt"));
        let manifest_path = sift_stream_manifest_path(output_dir, stem);
        let source_digest = if resume {
            Some(file_fnv1a64(path)?)
        } else {
            None
        };
        if let Some(source_digest) = source_digest {
            if let Some(rows) = validate_sift_stream_manifest(
                &manifest_path,
                config_hash,
                source_digest,
                &feature_path,
                &metadata_path,
            )? {
                total_keypoints = total_keypoints.saturating_add(rows);
                eprintln!(
                    "sift-stream: [{}/{}] {} -> {} kp (resumed)",
                    index + 1,
                    image_names.len(),
                    image_names[index],
                    rows,
                );
                continue;
            }
            eprintln!(
                "sift-stream: [{}/{}] {} has no valid completion sidecar; re-extracting",
                index + 1,
                image_names.len(),
                image_names[index],
            );
        }
        let started = std::time::Instant::now();
        let (features, _primary_count, _alternate, loci) = extract_sift_for_image(
            path,
            max_keypoints,
            affine,
            detector,
            multi_anisotropy,
            dsp,
            dsp_num_scales,
            l1_root,
            max_orientations,
            standard_orientations,
            prefer_larger_scale,
            full_pyramid,
            contrast_threshold,
            descriptor_magnification,
            scale_adaptive_gradients,
            vlfeat_compatible_descriptor,
            vlfeat_compatible_detector,
            vlfeat_bilinear_orientations,
            vlfeat_compatible_output_order,
            colmap_compatible_grayscale,
            split_colmap_detector_grayscale,
            append_descriptor_magnification,
            extra,
            extra_contrast_threshold,
            expected_dimensions,
        )?;
        if features.keypoints.len() != loci.len() {
            return Err(format!(
                "SIFT extractor returned {} keypoints but {} locus rows for {path:?}",
                features.keypoints.len(),
                loci.len()
            )
            .into());
        }
        write_stream_file_atomically(&feature_path, &feature_export_text(&features))?;
        write_stream_file_atomically(&metadata_path, &locus_metadata_text(&loci))?;
        if let Some(source_digest) = source_digest {
            let feature_digest = file_fnv1a64(&feature_path)?;
            let metadata_digest = file_fnv1a64(&metadata_path)?;
            write_sift_stream_manifest_atomically(
                &manifest_path,
                config_hash,
                source_digest,
                features.len(),
                feature_digest,
                metadata_digest,
            )?;
        }
        total_keypoints = total_keypoints.saturating_add(features.len());
        eprintln!(
            "sift-stream: [{}/{}] {} -> {} kp in {:.1}s",
            index + 1,
            image_names.len(),
            image_names[index],
            features.len(),
            started.elapsed().as_secs_f64()
        );
    }
    Ok(total_keypoints)
}

fn image_name_for(feat_filename: &str, feat_suffix: &str, image_suffix: &str) -> String {
    match feat_filename.strip_suffix(feat_suffix) {
        Some(stem) => format!("{stem}{image_suffix}"),
        None => feat_filename.to_string(),
    }
}

fn export_features_to_dir(
    dir: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    locus_metadata: &[Option<Vec<FeatureLocusMetadata>>],
) -> Result<(), Box<dyn std::error::Error>> {
    export_features_to_dir_impl(dir, image_names, features, None, locus_metadata)
}

/// Export canonical descriptors with a parallel native-pixel keypoint sidecar.
/// This is the calibration-aware counterpart to [`export_features_to_dir`];
/// descriptors remain owned by the mapper feature bank and are never cloned
/// merely to restore native coordinates for export.
fn export_features_to_dir_with_native_keypoints(
    dir: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    native_keypoints: &[Vec<Point2<f64>>],
    locus_metadata: &[Option<Vec<FeatureLocusMetadata>>],
) -> Result<(), Box<dyn std::error::Error>> {
    export_features_to_dir_impl(
        dir,
        image_names,
        features,
        Some(native_keypoints),
        locus_metadata,
    )
}

fn export_features_to_dir_impl(
    dir: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    native_keypoints: Option<&[Vec<Point2<f64>>]>,
    locus_metadata: &[Option<Vec<FeatureLocusMetadata>>],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(native_keypoints) = native_keypoints {
        if native_keypoints.len() != features.len() {
            return Err(format!(
                "native feature export: {} keypoint sets but {} feature sets",
                native_keypoints.len(),
                features.len()
            )
            .into());
        }
        for (image, (feature_set, keypoints)) in features.iter().zip(native_keypoints).enumerate() {
            if feature_set.descriptors.len() != keypoints.len() {
                return Err(format!(
                    "native feature export: image {image} has {} descriptors but {} keypoints",
                    feature_set.descriptors.len(),
                    keypoints.len()
                )
                .into());
            }
        }
    }
    std::fs::create_dir_all(dir)?;
    for (image_index, (name, feat)) in image_names.iter().zip(features).enumerate() {
        let stem = Path::new(name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(name);
        let path = dir.join(format!("{stem}_features.txt"));
        let keypoints = native_keypoints
            .map(|all| all[image_index].as_slice())
            .unwrap_or(feat.keypoints.as_slice());
        std::fs::write(
            &path,
            feature_export_text_with_keypoints(keypoints, &feat.descriptors),
        )?;
        if let Some(loci) = locus_metadata
            .get(image_index)
            .and_then(|loci| loci.as_ref())
        {
            let metadata_path = dir.join(format!("{stem}_loci.txt"));
            std::fs::write(metadata_path, locus_metadata_text(loci))?;
        }
    }
    Ok(())
}

#[cfg_attr(not(feature = "image-io"), allow(dead_code))]
fn feature_export_text(features: &FeatureSet) -> String {
    feature_export_text_with_keypoints(&features.keypoints, &features.descriptors)
}

fn feature_export_text_with_keypoints(
    keypoints: &[Point2<f64>],
    descriptors: &[Vec<f32>],
) -> String {
    let mut out = String::from("# visloc external-deep feature export\n");
    for (kp, desc) in keypoints.iter().zip(descriptors.iter()) {
        out.push_str(&format!("{:.6} {:.6} {:.6}", kp.x, kp.y, 1.0));
        for value in desc.as_slice() {
            out.push_str(&format!(" {value:.6}"));
        }
        out.push('\n');
    }
    out
}

fn locus_metadata_text(loci: &[FeatureLocusMetadata]) -> String {
    let mut out = String::from("# visloc orientation-locus metadata: x y scale orientation\n");
    for locus in loci {
        out.push_str(&format!(
            "{:.17e} {:.17e} {:.17e} {:.17e}\n",
            locus.x, locus.y, locus.scale, locus.orientation
        ));
    }
    out
}

/// Write one completed feature result with a same-directory temporary file
/// followed by an atomic rename.  A failed extractor therefore cannot leave a
/// truncated final feature file in the requested export directory.
#[cfg(feature = "image-io")]
fn write_stream_file_atomically(
    path: &Path,
    contents: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid feature export path {path:?}"))?;
    let temporary = path.with_file_name(format!(".{file_name}.tmp"));
    std::fs::write(&temporary, contents)?;
    if let Err(error) = std::fs::rename(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

/// Per-image completion marker for resumable SIFT stream exports.  The
/// marker is deliberately separate from the feature/locus files: it is
/// atomically installed last, so its presence never by itself makes a
/// partially written pair look complete.
#[cfg(feature = "image-io")]
const SIFT_STREAM_MANIFEST_MAGIC: &str = "visloc_sift_stream_manifest_v1";

#[cfg(feature = "image-io")]
fn sift_stream_manifest_path(output_dir: &Path, stem: &str) -> PathBuf {
    output_dir.join(format!("{stem}_sift_stream_manifest.txt"))
}

/// Return `(byte_count, FNV-1a-64)` for one completed source or output file.
/// This is a stable corruption/configuration guard, not a cryptographic
/// authenticity claim; the benchmark's top-level manifests use SHA-256 when
/// a stronger artifact identity is required.
#[cfg(feature = "image-io")]
fn file_fnv1a64(path: &Path) -> Result<(u64, u64), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buffer = [0u8; 1024 * 1024];
    let mut bytes = 0u64;
    let mut hash = 0xcbf29ce484222325u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(read as u64)
            .ok_or_else(|| format!("file {path:?} is too large to hash"))?;
        hash = fnv1a64_bytes_with_seed(hash, &buffer[..read]);
    }
    Ok((bytes, hash))
}

#[cfg(feature = "image-io")]
fn fnv1a64_bytes_with_seed(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3u64);
    }
    hash
}

#[cfg(feature = "image-io")]
#[allow(clippy::too_many_arguments)]
fn sift_stream_config_hash(
    image_name: &str,
    max_keypoints: usize,
    affine: bool,
    detector: &str,
    multi_anisotropy: bool,
    dsp: bool,
    dsp_num_scales: usize,
    l1_root: bool,
    max_orientations: usize,
    standard_orientations: bool,
    prefer_larger_scale: bool,
    full_pyramid: bool,
    contrast_threshold: f64,
    descriptor_magnification: f64,
    scale_adaptive_gradients: bool,
    vlfeat_compatible_descriptor: bool,
    vlfeat_compatible_detector: bool,
    vlfeat_bilinear_orientations: bool,
    vlfeat_compatible_output_order: bool,
    colmap_compatible_grayscale: bool,
    split_colmap_detector_grayscale: bool,
    append_descriptor_magnification: Option<f64>,
    extra_keypoints: usize,
    extra_contrast_threshold: Option<f64>,
    expected_dimensions: Option<(u32, u32)>,
    expected_camera: Option<&Camera>,
) -> u64 {
    // Keep this explicit rather than hashing `Args`: `--sift-stream-resume`
    // itself must not make a previously completed export stale, while every
    // extraction-affecting option and per-image camera assignment must.
    let snapshot = format!(
        "visloc_sift_stream_resume_v1;image_name={image_name:?};max_keypoints={max_keypoints};affine={affine};detector={detector:?};multi_anisotropy={multi_anisotropy};dsp={dsp};dsp_num_scales={dsp_num_scales};l1_root={l1_root};max_orientations={max_orientations};standard_orientations={standard_orientations};prefer_larger_scale={prefer_larger_scale};full_pyramid={full_pyramid};contrast_threshold={contrast_threshold:?};descriptor_magnification={descriptor_magnification:?};scale_adaptive_gradients={scale_adaptive_gradients};vlfeat_compatible_descriptor={vlfeat_compatible_descriptor};vlfeat_compatible_detector={vlfeat_compatible_detector};vlfeat_bilinear_orientations={vlfeat_bilinear_orientations};vlfeat_compatible_output_order={vlfeat_compatible_output_order};colmap_compatible_grayscale={colmap_compatible_grayscale};split_colmap_detector_grayscale={split_colmap_detector_grayscale};append_descriptor_magnification={append_descriptor_magnification:?};extra_keypoints={extra_keypoints};extra_contrast_threshold={extra_contrast_threshold:?};expected_dimensions={expected_dimensions:?};expected_camera={expected_camera:?}"
    );
    effective_config_hash(&snapshot)
}

#[cfg(feature = "image-io")]
fn parse_sift_stream_u64(fields: &HashMap<String, String>, key: &str) -> Option<u64> {
    fields.get(key)?.parse().ok()
}

#[cfg(feature = "image-io")]
fn parse_sift_stream_hex(fields: &HashMap<String, String>, key: &str) -> Option<u64> {
    u64::from_str_radix(fields.get(key)?, 16).ok()
}

#[cfg(feature = "image-io")]
fn count_sift_stream_rows(path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mut rows = 0usize;
    for line in BufReader::new(file).lines() {
        let line = line?;
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('#') {
            rows = rows
                .checked_add(1)
                .ok_or_else(|| format!("row count overflow in {path:?}"))?;
        }
    }
    Ok(rows)
}

#[cfg(feature = "image-io")]
fn write_sift_stream_manifest_atomically(
    path: &Path,
    config_hash: u64,
    source_digest: (u64, u64),
    feature_rows: usize,
    feature_digest: (u64, u64),
    loci_digest: (u64, u64),
) -> Result<(), Box<dyn std::error::Error>> {
    let contents = format!(
        "{SIFT_STREAM_MANIFEST_MAGIC}\nconfig_fnv1a64={config_hash:016x}\nsource_bytes={}\nsource_fnv1a64={:016x}\nfeature_rows={feature_rows}\nfeature_bytes={}\nfeature_fnv1a64={:016x}\nloci_bytes={}\nloci_fnv1a64={:016x}\n",
        source_digest.0,
        source_digest.1,
        feature_digest.0,
        feature_digest.1,
        loci_digest.0,
        loci_digest.1,
    );
    write_stream_file_atomically(path, &contents)
}

/// Validate a completion marker and all files it covers.  `Ok(None)` means
/// that the output is absent, malformed, stale, or tampered and must be
/// re-extracted; filesystem errors unrelated to absence remain hard errors.
#[cfg(feature = "image-io")]
fn validate_sift_stream_manifest(
    path: &Path,
    expected_config_hash: u64,
    expected_source_digest: (u64, u64),
    feature_path: &Path,
    loci_path: &Path,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::InvalidData => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mut lines = text.lines();
    if lines.next() != Some(SIFT_STREAM_MANIFEST_MAGIC) {
        return Ok(None);
    }
    let mut fields = HashMap::new();
    for line in lines {
        let Some((key, value)) = line.split_once('=') else {
            return Ok(None);
        };
        if key.is_empty()
            || value.is_empty()
            || fields.insert(key.to_owned(), value.to_owned()).is_some()
        {
            return Ok(None);
        }
    }
    if fields.len() != 8 {
        return Ok(None);
    }
    let config_hash = parse_sift_stream_hex(&fields, "config_fnv1a64");
    let source_bytes = parse_sift_stream_u64(&fields, "source_bytes");
    let source_hash = parse_sift_stream_hex(&fields, "source_fnv1a64");
    let rows = fields
        .get("feature_rows")
        .and_then(|value| value.parse().ok());
    let feature_bytes = parse_sift_stream_u64(&fields, "feature_bytes");
    let feature_hash = parse_sift_stream_hex(&fields, "feature_fnv1a64");
    let loci_bytes = parse_sift_stream_u64(&fields, "loci_bytes");
    let loci_hash = parse_sift_stream_hex(&fields, "loci_fnv1a64");
    let (
        Some(config_hash),
        Some(source_bytes),
        Some(source_hash),
        Some(rows),
        Some(feature_bytes),
        Some(feature_hash),
        Some(loci_bytes),
        Some(loci_hash),
    ) = (
        config_hash,
        source_bytes,
        source_hash,
        rows,
        feature_bytes,
        feature_hash,
        loci_bytes,
        loci_hash,
    )
    else {
        return Ok(None);
    };
    if config_hash != expected_config_hash || (source_bytes, source_hash) != expected_source_digest
    {
        return Ok(None);
    }
    let actual_feature = match file_fnv1a64(feature_path) {
        Ok(digest) => digest,
        Err(error)
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) =>
        {
            return Ok(None)
        }
        Err(error) => return Err(error),
    };
    let actual_loci = match file_fnv1a64(loci_path) {
        Ok(digest) => digest,
        Err(error)
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::NotFound) =>
        {
            return Ok(None)
        }
        Err(error) => return Err(error),
    };
    if actual_feature != (feature_bytes, feature_hash) || actual_loci != (loci_bytes, loci_hash) {
        return Ok(None);
    }
    if count_sift_stream_rows(feature_path)? != rows || count_sift_stream_rows(loci_path)? != rows {
        return Ok(None);
    }
    Ok(Some(rows))
}

/// Streaming counterpart to [`export_features_to_dir`].  The loader callback
/// is invoked exactly once per path and its result is consumed and dropped
/// before the next callback, so source images, pyramids, and descriptor banks
/// are never accumulated across the directory.
#[cfg(feature = "image-io")]
#[cfg_attr(not(test), allow(dead_code))]
fn stream_export_features_with_loader<F>(
    paths: &[PathBuf],
    output_dir: &Path,
    mut loader: F,
) -> Result<usize, Box<dyn std::error::Error>>
where
    F: FnMut(
        usize,
        &Path,
    ) -> Result<(FeatureSet, Vec<FeatureLocusMetadata>), Box<dyn std::error::Error>>,
{
    let mut stems = HashSet::new();
    let mut stem_names = Vec::with_capacity(paths.len());
    for path in paths {
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or_else(|| format!("source image {path:?} has no UTF-8 stem"))?
            .to_owned();
        if !stems.insert(stem.clone()) {
            return Err(format!("duplicate source image stem {stem:?}").into());
        }
        stem_names.push(stem);
    }
    std::fs::create_dir_all(output_dir)?;
    let mut total_keypoints = 0usize;
    for (index, path) in paths.iter().enumerate() {
        let stem = &stem_names[index];
        let (features, loci) = loader(index, path)?;
        if features.keypoints.len() != loci.len() {
            return Err(format!(
                "stream loader returned {} keypoints but {} locus rows for {path:?}",
                features.keypoints.len(),
                loci.len()
            )
            .into());
        }
        let feature_path = output_dir.join(format!("{stem}_features.txt"));
        let metadata_path = output_dir.join(format!("{stem}_loci.txt"));
        write_stream_file_atomically(&feature_path, &feature_export_text(&features))?;
        write_stream_file_atomically(&metadata_path, &locus_metadata_text(&loci))?;
        total_keypoints = total_keypoints.saturating_add(features.len());
        // `features` owns all descriptors returned by this callback.  It is
        // deliberately scoped to one loop iteration and dropped here before
        // the next image is decoded.
    }
    Ok(total_keypoints)
}

const CANONICAL_FEATURE_COORD_SCALE: f64 = 1_000_000.0;

fn canonical_coordinate_key(value: f64) -> (u8, i64) {
    if value.is_finite() {
        (0, (value * CANONICAL_FEATURE_COORD_SCALE).round() as i64)
    } else if value.is_nan() {
        (2, 0)
    } else if value.is_sign_negative() {
        (1, 0)
    } else {
        (3, 0)
    }
}

fn canonical_descriptor_cmp(lhs: &[f32], rhs: &[f32]) -> CmpOrdering {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| a.total_cmp(b))
        .find(|ordering| *ordering != CmpOrdering::Equal)
        .unwrap_or_else(|| lhs.len().cmp(&rhs.len()))
}

fn canonical_feature_cmp(set: &FeatureSet, lhs: usize, rhs: usize) -> CmpOrdering {
    let lhs_point = set.keypoints[lhs];
    let rhs_point = set.keypoints[rhs];
    canonical_coordinate_key(lhs_point.x)
        .cmp(&canonical_coordinate_key(rhs_point.x))
        .then_with(|| {
            canonical_coordinate_key(lhs_point.y).cmp(&canonical_coordinate_key(rhs_point.y))
        })
        .then_with(|| canonical_descriptor_cmp(&set.descriptors[lhs], &set.descriptors[rhs]))
        .then_with(|| lhs.cmp(&rhs))
}

/// Put every feature/descriptor row into a deterministic physical order and
/// return an old-index → new-index map per image. The descriptor tie-break is
/// only used for co-located rows; exact duplicate rows remain physically
/// indistinguishable and use their original index as the final deterministic
/// fallback. Alternate SIFT banks follow the same permutation one-for-one.
fn canonicalize_feature_order(
    features: &mut [FeatureSet],
    alternate_descriptors: &mut [Option<Vec<Vec<f32>>>],
) -> Result<Vec<Vec<usize>>, String> {
    if features.len() != alternate_descriptors.len() {
        return Err(format!(
            "canonical feature order: {} feature sets but {} alternate banks",
            features.len(),
            alternate_descriptors.len()
        ));
    }
    let mut old_to_new = Vec::with_capacity(features.len());
    for image in 0..features.len() {
        let set = &features[image];
        let mut order: Vec<usize> = (0..set.len()).collect();
        order.sort_by(|lhs, rhs| canonical_feature_cmp(set, *lhs, *rhs));
        let mut inverse = vec![0usize; order.len()];
        for (new_index, &old_index) in order.iter().enumerate() {
            inverse[old_index] = new_index;
        }
        let set = &mut features[image];
        let old_keypoints = std::mem::take(&mut set.keypoints);
        let old_descriptors = std::mem::take(&mut set.descriptors);
        set.keypoints = order.iter().map(|&index| old_keypoints[index]).collect();
        set.descriptors = order
            .iter()
            .map(|&index| old_descriptors[index].clone())
            .collect();
        if let Some(bank) = alternate_descriptors[image].as_mut() {
            if bank.len() != order.len() {
                return Err(format!(
                    "canonical feature order: image {image} alternate bank has {} rows, expected {}",
                    bank.len(),
                    order.len()
                ));
            }
            let old_bank = std::mem::take(bank);
            *bank = order.iter().map(|&index| old_bank[index].clone()).collect();
        }
        old_to_new.push(inverse);
    }
    Ok(old_to_new)
}

/// Apply the mapper's old-index → new-index permutation to the native pixel
/// sidecar used by the multi-camera exporter.  Descriptors are intentionally
/// absent from this sidecar: canonicalization changes only keypoint pixels,
/// so the mapper's descriptor bank remains the single owner of each row.
fn remap_feature_keypoints_by_old_to_new(
    keypoints: &mut [Vec<Point2<f64>>],
    old_to_new: &[Vec<usize>],
) -> Result<(), String> {
    if keypoints.len() != old_to_new.len() {
        return Err(format!(
            "native keypoint order: {} feature sets but {} index maps",
            keypoints.len(),
            old_to_new.len()
        ));
    }
    for (image, (image_keypoints, map)) in keypoints.iter_mut().zip(old_to_new).enumerate() {
        if image_keypoints.len() != map.len() || map.iter().any(|&new| new >= map.len()) {
            return Err(format!(
                "native keypoint order: image {image} has {} rows but map has {} entries",
                image_keypoints.len(),
                map.len()
            ));
        }
        let mut seen = vec![false; map.len()];
        for &new_index in map {
            if std::mem::replace(&mut seen[new_index], true) {
                return Err(format!(
                    "native keypoint order: image {image} permutation contains duplicate index {new_index}"
                ));
            }
        }
        let old_keypoints = std::mem::take(image_keypoints);
        let mut reordered = vec![Point2::new(0.0, 0.0); map.len()];
        for (old_index, &new_index) in map.iter().enumerate() {
            reordered[new_index] = old_keypoints[old_index];
        }
        *image_keypoints = reordered;
    }
    Ok(())
}

/// Replace compacted mapper feature keypoints with native-pixel coordinates
/// for COLMAP's multi-camera exporter.  The descriptor vectors are retained
/// from `output_features`; the writer consumes only the keypoint coordinates.
fn replace_feature_keypoints_from_native(
    output_features: &mut [FeatureSet],
    source_indices: &[usize],
    native_keypoints: &[Vec<Point2<f64>>],
) -> Result<(), String> {
    if output_features.len() != source_indices.len() {
        return Err(format!(
            "native export: {} output feature sets but {} source indices",
            output_features.len(),
            source_indices.len()
        ));
    }
    for (output_index, (&source_index, output)) in source_indices
        .iter()
        .zip(output_features.iter_mut())
        .enumerate()
    {
        let source = native_keypoints.get(source_index).ok_or_else(|| {
            format!(
                "native export: source image {source_index} is outside 0..{}",
                native_keypoints.len()
            )
        })?;
        if source.len() != output.descriptors.len() {
            return Err(format!(
                "native export: output image {output_index} has {} descriptors but source image {source_index} has {} keypoints",
                output.descriptors.len(),
                source.len()
            ));
        }
        output.keypoints = source.clone();
    }
    Ok(())
}

fn remap_locus_metadata(
    metadata: &mut [Option<Vec<FeatureLocusMetadata>>],
    old_to_new: &[Vec<usize>],
) -> Result<(), String> {
    if metadata.len() != old_to_new.len() {
        return Err(format!(
            "canonical feature order: {} metadata sets but {} index maps",
            metadata.len(),
            old_to_new.len()
        ));
    }
    for (image, (loci, map)) in metadata.iter_mut().zip(old_to_new).enumerate() {
        let Some(loci) = loci.as_mut() else {
            continue;
        };
        if loci.len() != map.len() {
            return Err(format!(
                "canonical feature order: image {image} metadata has {} rows, expected {}",
                loci.len(),
                map.len()
            ));
        }
        let old_loci = std::mem::take(loci);
        let mut new_loci = vec![
            FeatureLocusMetadata {
                x: f64::NAN,
                y: f64::NAN,
                scale: f64::NAN,
                orientation: f64::NAN,
            };
            old_loci.len()
        ];
        for (old_index, &new_index) in map.iter().enumerate() {
            let Some(slot) = new_loci.get_mut(new_index) else {
                return Err(format!(
                    "canonical feature order: image {image} invalid metadata map {old_index}->{new_index}"
                ));
            };
            *slot = old_loci[old_index];
        }
        *loci = new_loci;
    }
    Ok(())
}

fn remap_imported_matches(
    imported: &mut HashMap<(usize, usize), Vec<(usize, usize)>>,
    old_to_new: &[Vec<usize>],
) -> Result<(), String> {
    for (&(image_i, image_j), matches) in imported.iter_mut() {
        for (keypoint_i, keypoint_j) in matches {
            *keypoint_i = *old_to_new
                .get(image_i)
                .and_then(|indices| indices.get(*keypoint_i))
                .ok_or_else(|| {
                    format!(
                        "canonical feature order: invalid imported index ({image_i},{keypoint_i})"
                    )
                })?;
            *keypoint_j = *old_to_new
                .get(image_j)
                .and_then(|indices| indices.get(*keypoint_j))
                .ok_or_else(|| {
                    format!(
                        "canonical feature order: invalid imported index ({image_j},{keypoint_j})"
                    )
                })?;
        }
    }
    Ok(())
}

fn remap_imported_verified_pairs(
    imported: &mut [ImportedVerifiedPair],
    old_to_new: &[Vec<usize>],
) -> Result<(), String> {
    for pair in imported {
        for (keypoint_i, keypoint_j) in &mut pair.matches {
            *keypoint_i = *old_to_new
                .get(pair.image_i)
                .and_then(|indices| indices.get(*keypoint_i))
                .ok_or_else(|| {
                    format!(
                        "canonical feature order: invalid verified index ({},{})",
                        pair.image_i, *keypoint_i
                    )
                })?;
            *keypoint_j = *old_to_new
                .get(pair.image_j)
                .and_then(|indices| indices.get(*keypoint_j))
                .ok_or_else(|| {
                    format!(
                        "canonical feature order: invalid verified index ({},{})",
                        pair.image_j, *keypoint_j
                    )
                })?;
        }
    }
    Ok(())
}

/// Parse `export_colmap_matches.py` output: image count, names, then per-pair
/// `(i, j, count)` + `count` lines of `qi tj`. Pair keys are normalized to
/// `(min(i,j), max(i,j))`.
fn parse_imported_matches_file(
    path: &Path,
    image_names: &[String],
) -> Result<HashMap<(usize, usize), Vec<(usize, usize)>>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    let name_count: usize = lines
        .next()
        .ok_or("import matches: missing image count")?
        .parse()?;
    if name_count != image_names.len() {
        return Err(format!(
            "import matches: file has {name_count} names, run has {}",
            image_names.len()
        )
        .into());
    }
    for (idx, expected_name) in image_names.iter().enumerate().take(name_count) {
        let file_name = lines
            .next()
            .ok_or("import matches: truncated image name list")?;
        if file_name != expected_name {
            return Err(format!(
                "import matches: name mismatch at {idx}: file {file_name:?} vs run {:?}",
                expected_name
            )
            .into());
        }
    }
    let pair_count: usize = lines
        .next()
        .ok_or("import matches: missing pair count")?
        .parse()?;
    let mut out = HashMap::new();
    for _ in 0..pair_count {
        let head: Vec<usize> = lines
            .next()
            .ok_or("import matches: truncated pair header")?
            .split_whitespace()
            .map(|t| t.parse())
            .collect::<Result<_, _>>()?;
        if head.len() != 3 {
            return Err("import matches: pair header needs i j count".into());
        }
        let (mut i, mut j, count) = (head[0], head[1], head[2]);
        if i > j {
            std::mem::swap(&mut i, &mut j);
        }
        let mut matches = Vec::with_capacity(count);
        for _ in 0..count {
            let m: Vec<usize> = lines
                .next()
                .ok_or("import matches: truncated correspondence")?
                .split_whitespace()
                .map(|t| t.parse())
                .collect::<Result<_, _>>()?;
            if m.len() != 2 {
                return Err("import matches: correspondence needs qi tj".into());
            }
            let (qi, tj) = if head[0] <= head[1] {
                (m[0], m[1])
            } else {
                (m[1], m[0])
            };
            matches.push((qi, tj));
        }
        out.insert((i, j), matches);
    }
    Ok(out)
}

struct ImportedVerifiedPair {
    image_i: usize,
    image_j: usize,
    matches: Vec<(usize, usize)>,
    config: ConfigurationType,
    essential_matrix: Option<Matrix3<f64>>,
}

/// Verifier fields which are not currently consumed by PairwiseMatches but
/// are required to make a verified stream auditable and losslessly replayable.
#[derive(Debug, Clone, Default)]
struct SnapshotPairMetadata {
    raw_match_count: usize,
    raw_matches: Vec<(usize, usize)>,
    accepted_inlier_indices: Vec<usize>,
    essential_inlier_indices: Vec<usize>,
    e_inlier_count: usize,
    f_inlier_count: usize,
    h_inlier_count: usize,
    fundamental: Option<Matrix3<f64>>,
    homography: Option<Matrix3<f64>>,
    relative_pose: Option<(Matrix3<f64>, Vector3<f64>)>,
}

fn configuration_code(config: Option<ConfigurationType>) -> u8 {
    match config {
        None => 255,
        Some(ConfigurationType::Undefined) => 0,
        Some(ConfigurationType::Degenerate) => 1,
        Some(ConfigurationType::Uncalibrated) => 2,
        Some(ConfigurationType::Calibrated) => 3,
        Some(ConfigurationType::Planar) => 4,
        Some(ConfigurationType::Panoramic) => 5,
        Some(ConfigurationType::PlanarOrPanoramic) => 6,
        Some(ConfigurationType::Watermark) => 7,
        Some(ConfigurationType::Multiple) => 8,
    }
}

fn configuration_from_code(code: u8) -> Result<Option<ConfigurationType>, String> {
    Ok(match code {
        0 => Some(ConfigurationType::Undefined),
        1 => Some(ConfigurationType::Degenerate),
        2 => Some(ConfigurationType::Uncalibrated),
        3 => Some(ConfigurationType::Calibrated),
        4 => Some(ConfigurationType::Planar),
        5 => Some(ConfigurationType::Panoramic),
        6 => Some(ConfigurationType::PlanarOrPanoramic),
        7 => Some(ConfigurationType::Watermark),
        8 => Some(ConfigurationType::Multiple),
        255 => None,
        other => {
            return Err(format!(
                "verified-pair snapshot has unknown configuration code {other}"
            ))
        }
    })
}

fn matrix_bits(matrix: Option<&Matrix3<f64>>) -> Option<[u64; 9]> {
    matrix.map(|matrix| std::array::from_fn(|index| matrix.as_slice()[index].to_bits()))
}

fn matrix_from_bits(bits: Option<[u64; 9]>) -> Option<Matrix3<f64>> {
    bits.map(|bits| Matrix3::from_column_slice(&bits.map(f64::from_bits)))
}

fn vector_bits(vector: Option<&Vector3<f64>>) -> Option<[u64; 3]> {
    vector.map(|vector| std::array::from_fn(|index| vector[index].to_bits()))
}

fn snapshot_metadata_from_report(
    raw_matches: &[DescriptorMatch],
    report: &TwoViewGeometryReport,
) -> SnapshotPairMetadata {
    SnapshotPairMetadata {
        raw_match_count: raw_matches.len(),
        raw_matches: raw_matches
            .iter()
            .map(|m| (m.query_index, m.train_index))
            .collect(),
        accepted_inlier_indices: report.inliers.clone(),
        essential_inlier_indices: report.essential_inliers.clone(),
        e_inlier_count: report.e_inlier_count,
        f_inlier_count: report.f_inlier_count,
        h_inlier_count: report.h_inlier_count,
        fundamental: report.fundamental,
        homography: report.homography,
        relative_pose: report.relative_pose,
    }
}

fn snapshot_pair_record(
    pair: &PairwiseMatches,
    metadata: Option<&SnapshotPairMetadata>,
) -> SnapshotPairRecord {
    let fallback_accepted: Vec<u64> = (0..pair.matches.len() as u64).collect();
    let fallback_essential: Vec<u64> = pair
        .essential_matches
        .as_ref()
        .map(|matches| (0..matches.len() as u64).collect())
        .unwrap_or_default();
    let metadata = metadata.cloned().unwrap_or_default();
    SnapshotPairRecord {
        image_i: pair.image_i as u64,
        image_j: pair.image_j as u64,
        raw_match_count: if metadata.raw_match_count == 0 {
            pair.matches.len() as u64
        } else {
            metadata.raw_match_count as u64
        },
        raw_matches: if metadata.raw_matches.is_empty() {
            pair.matches
                .iter()
                .map(|&(left, right)| (left as u64, right as u64))
                .collect()
        } else {
            metadata
                .raw_matches
                .iter()
                .map(|&(left, right)| (left as u64, right as u64))
                .collect()
        },
        accepted_inlier_indices: if metadata.accepted_inlier_indices.is_empty() {
            fallback_accepted
        } else {
            metadata
                .accepted_inlier_indices
                .iter()
                .map(|&value| value as u64)
                .collect()
        },
        essential_inlier_indices: if metadata.essential_inlier_indices.is_empty() {
            fallback_essential
        } else {
            metadata
                .essential_inlier_indices
                .iter()
                .map(|&value| value as u64)
                .collect()
        },
        matches: pair
            .matches
            .iter()
            .map(|&(left, right)| (left as u64, right as u64))
            .collect(),
        essential_matches: pair.essential_matches.as_ref().map(|matches| {
            matches
                .iter()
                .map(|&(left, right)| (left as u64, right as u64))
                .collect()
        }),
        config: configuration_code(pair.two_view_config),
        calibrated: pair.two_view_config == Some(ConfigurationType::Calibrated),
        e_inlier_count: if metadata.e_inlier_count == 0 {
            pair.essential_matches.as_ref().map_or(0, Vec::len) as u64
        } else {
            metadata.e_inlier_count as u64
        },
        f_inlier_count: metadata.f_inlier_count as u64,
        h_inlier_count: metadata.h_inlier_count as u64,
        essential_matrix_bits: matrix_bits(pair.essential_matrix.as_ref()),
        fundamental_matrix_bits: matrix_bits(metadata.fundamental.as_ref()),
        homography_matrix_bits: matrix_bits(metadata.homography.as_ref()),
        relative_rotation_bits: metadata
            .relative_pose
            .as_ref()
            .and_then(|(rotation, _)| matrix_bits(Some(rotation))),
        relative_translation_bits: metadata
            .relative_pose
            .as_ref()
            .and_then(|(_, translation)| vector_bits(Some(translation))),
    }
}

fn snapshot_metadata_map_from_pairs(
    pairwise: &[PairwiseMatches],
) -> HashMap<(usize, usize), SnapshotPairMetadata> {
    pairwise
        .iter()
        .map(|pair| {
            (
                (
                    pair.image_i.min(pair.image_j),
                    pair.image_i.max(pair.image_j),
                ),
                SnapshotPairMetadata::default(),
            )
        })
        .collect()
}

fn snapshot_metadata_from_record(record: &SnapshotPairRecord) -> SnapshotPairMetadata {
    SnapshotPairMetadata {
        raw_match_count: record.raw_match_count as usize,
        raw_matches: record
            .raw_matches
            .iter()
            .map(|&(left, right)| (left as usize, right as usize))
            .collect(),
        accepted_inlier_indices: record
            .accepted_inlier_indices
            .iter()
            .map(|&value| value as usize)
            .collect(),
        essential_inlier_indices: record
            .essential_inlier_indices
            .iter()
            .map(|&value| value as usize)
            .collect(),
        e_inlier_count: record.e_inlier_count as usize,
        f_inlier_count: record.f_inlier_count as usize,
        h_inlier_count: record.h_inlier_count as usize,
        fundamental: matrix_from_bits(record.fundamental_matrix_bits),
        homography: matrix_from_bits(record.homography_matrix_bits),
        relative_pose: matrix_from_bits(record.relative_rotation_bits).and_then(|rotation| {
            vector_from_bits(record.relative_translation_bits)
                .map(|translation| (rotation, translation))
        }),
    }
}

fn snapshot_metadata_map_from_snapshot(
    snapshot: &VerifiedPairSnapshot,
) -> HashMap<(usize, usize), SnapshotPairMetadata> {
    snapshot
        .pairs
        .iter()
        .map(|record| {
            (
                (
                    record.image_i.min(record.image_j) as usize,
                    record.image_i.max(record.image_j) as usize,
                ),
                snapshot_metadata_from_record(record),
            )
        })
        .collect()
}

/// Promote only stable F-winning edges to a calibrated E for the opt-in
/// sequence fallback.  The normal pair stream remains F-winning and keeps its
/// original matrix; this pass is deliberately performed after verification,
/// using the lossless raw-match metadata already captured for diagnostics.
/// Thus a sequence fallback can use `Kᵀ F K` when the direct E estimate is a
/// façade-biased outlier without changing ordinary track construction.
#[derive(Debug, Clone, Default)]
struct SequenceFToEPromotionStats {
    promoted: usize,
    high_support_overrides: usize,
    high_support_override_pair_indices: Vec<usize>,
}

fn promote_sequence_fundamentals_to_essentials(
    pairwise: &mut [PairwiseMatches],
    metadata_by_pair: &HashMap<(usize, usize), SnapshotPairMetadata>,
    features: &[FeatureSet],
    camera: &Camera,
) -> SequenceFToEPromotionStats {
    let mut stats = SequenceFToEPromotionStats::default();
    for (pair_index, pair) in pairwise.iter_mut().enumerate() {
        if pair.two_view_config != Some(ConfigurationType::Uncalibrated)
            || pair.essential_matches.is_some()
        {
            continue;
        }
        let key = (
            pair.image_i.min(pair.image_j),
            pair.image_i.max(pair.image_j),
        );
        let Some(metadata) = metadata_by_pair.get(&key) else {
            continue;
        };
        let Some(fundamental) = metadata.fundamental else {
            continue;
        };
        if metadata.raw_matches.len() < 8
            || pair.image_i >= features.len()
            || pair.image_j >= features.len()
        {
            continue;
        }
        let correspondences = metadata
            .raw_matches
            .iter()
            .filter_map(|&(keypoint_i, keypoint_j)| {
                Some(TwoViewCorrespondence::new(
                    *features[pair.image_i].keypoints.get(keypoint_i)?,
                    *features[pair.image_j].keypoints.get(keypoint_j)?,
                ))
            })
            .collect::<Vec<_>>();
        if correspondences.len() < 8 {
            continue;
        }
        let report = TwoViewGeometryReport {
            config: ConfigurationType::Uncalibrated,
            inliers: metadata.accepted_inlier_indices.clone(),
            essential: pair.essential_matrix,
            fundamental: Some(fundamental),
            homography: metadata.homography,
            relative_pose: metadata.relative_pose,
            essential_inliers: metadata.essential_inlier_indices.clone(),
            e_inlier_count: metadata.e_inlier_count,
            f_inlier_count: metadata.f_inlier_count,
            h_inlier_count: metadata.h_inlier_count,
        };
        let Some(diagnostics) = f_to_e_candidate_diagnostics(&report, &correspondences, camera)
        else {
            continue;
        };
        let strict_sequence_gate = sequence_f_to_e_stability_gate(&diagnostics);
        let high_support_override =
            !strict_sequence_gate && sequence_f_to_e_high_support_override_gate(&diagnostics);
        if !strict_sequence_gate && !high_support_override {
            continue;
        }
        let Some(essential) = project_fundamental_to_essential(&fundamental, camera) else {
            continue;
        };
        pair.essential_matrix = Some(essential);
        stats.promoted += 1;
        if high_support_override {
            stats.high_support_overrides += 1;
            stats.high_support_override_pair_indices.push(pair_index);
        }
        if std::env::var_os("VISLOC_SFM_DEBUG").is_some() {
            eprintln!(
                "sfm-debug-sequence-f2e: {} {} gate={} f_inliers={} ef_inliers={} overlap={:.6} cheirality={:.6} margin={:.6} angle_p25_deg={:.6} rotation_spread_deg={:.6} translation_spread_deg={:.6}",
                pair.image_i,
                pair.image_j,
                if high_support_override {
                    "high_support_translation_spread_override"
                } else {
                    "strict"
                },
                diagnostics.f_inliers,
                diagnostics.ef_inliers,
                diagnostics.ef_overlap_on_f,
                diagnostics.cheirality_ratio,
                diagnostics.cheirality_margin,
                diagnostics.ef_angle_p25_deg,
                diagnostics.pose_rotation_spread_deg,
                diagnostics.pose_translation_spread_deg,
            );
        }
    }
    stats
}

fn snapshot_indices_for_matches(
    raw_matches: &[(usize, usize)],
    matches: &[(usize, usize)],
) -> Option<Vec<usize>> {
    let mut used = vec![false; raw_matches.len()];
    let mut indices = Vec::with_capacity(matches.len());
    for &needle in matches {
        let index = raw_matches
            .iter()
            .enumerate()
            .find(|(index, value)| **value == needle && !used[*index])
            .map(|(index, _)| index)?;
        used[index] = true;
        indices.push(index);
    }
    Some(indices)
}

fn snapshot_metadata_matches_pair(pair: &PairwiseMatches, metadata: &SnapshotPairMetadata) -> bool {
    if metadata.raw_match_count != metadata.raw_matches.len()
        || metadata.accepted_inlier_indices.len() != pair.matches.len()
    {
        return false;
    }
    let Some(accepted) = snapshot_indices_for_matches(&metadata.raw_matches, &pair.matches) else {
        return false;
    };
    if accepted != metadata.accepted_inlier_indices {
        return false;
    }
    match (
        &pair.essential_matches,
        metadata.essential_inlier_indices.as_slice(),
    ) {
        (Some(matches), indices) => {
            snapshot_indices_for_matches(&metadata.raw_matches, matches).as_deref() == Some(indices)
        }
        (None, []) => true,
        (None, _) => false,
    }
}

fn write_verified_pair_snapshot(
    path: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    camera: &Camera,
    pairwise: &[PairwiseMatches],
    metadata_by_pair: &HashMap<(usize, usize), SnapshotPairMetadata>,
    args: &Args,
) -> Result<(), String> {
    write_verified_pair_snapshot_with_validation(
        path,
        image_names,
        features,
        camera,
        pairwise,
        metadata_by_pair,
        args,
        None,
        false,
    )
}

fn write_verified_pair_snapshot_atomic(
    path: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    camera: &Camera,
    pairwise: &[PairwiseMatches],
    metadata_by_pair: &HashMap<(usize, usize), SnapshotPairMetadata>,
    args: &Args,
    feature_validation: &SnapshotFeatureValidation,
) -> Result<(), String> {
    write_verified_pair_snapshot_with_validation(
        path,
        image_names,
        features,
        camera,
        pairwise,
        metadata_by_pair,
        args,
        Some(feature_validation),
        true,
    )
}

fn write_verified_pair_snapshot_with_validation(
    path: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    camera: &Camera,
    pairwise: &[PairwiseMatches],
    metadata_by_pair: &HashMap<(usize, usize), SnapshotPairMetadata>,
    args: &Args,
    feature_validation: Option<&SnapshotFeatureValidation>,
    atomic: bool,
) -> Result<(), String> {
    let feature_counts: Vec<u64> = feature_validation
        .map(|validation| {
            validation
                .feature_counts
                .iter()
                .map(|&count| count as u64)
                .collect()
        })
        .unwrap_or_else(|| {
            features
                .iter()
                .map(|features| features.keypoints.len() as u64)
                .collect()
        });
    let records: Vec<SnapshotPairRecord> = pairwise
        .iter()
        .map(|pair| {
            let key = (
                pair.image_i.min(pair.image_j),
                pair.image_i.max(pair.image_j),
            );
            let metadata = metadata_by_pair
                .get(&key)
                .filter(|metadata| snapshot_metadata_matches_pair(pair, metadata));
            snapshot_pair_record(pair, metadata)
        })
        .collect();
    let verifier_config = snapshot_verifier_config(args);
    let snapshot_config = snapshot_export_config(args);
    let snapshot = VerifiedPairSnapshot {
        schema_version: verified_pair_snapshot::SCHEMA_VERSION,
        image_names: image_names.to_vec(),
        image_manifest_hash: snapshot_image_manifest_hash(image_names),
        feature_manifest_hash: feature_validation.map_or_else(
            || snapshot_feature_manifest_hash(features),
            |validation| validation.feature_manifest_hash,
        ),
        feature_counts,
        width: u64::from(camera.width),
        height: u64::from(camera.height),
        intrinsics_bits: snapshot_intrinsics_bits(camera)?,
        // The full Args debug snapshot is already in the phase log. It
        // contains candidate/output paths, so storing it here made otherwise
        // identical snapshots differ across resumable run roots. Keep the
        // binary envelope path-independent; pair hashes and the runner index
        // retain the data/provenance bindings.
        effective_config_hash: effective_config_hash(&snapshot_config),
        effective_config: snapshot_config,
        verifier_config_hash: effective_config_hash(&verifier_config),
        verifier_config,
        pair_order_hash: ordered_pairwise_edge_hash(pairwise),
        unordered_edge_hash: unordered_pairwise_edge_hash(pairwise),
        accepted_match_count: pairwise
            .iter()
            .map(|pair| pair.matches.len())
            .sum::<usize>() as u64,
        pairs: records,
    };
    if atomic {
        verified_pair_snapshot::write_atomic(path, &snapshot)
    } else {
        verified_pair_snapshot::write(path, &snapshot)
    }
}

fn vector_from_bits(bits: Option<[u64; 3]>) -> Option<Vector3<f64>> {
    bits.map(|bits| Vector3::from_column_slice(&bits.map(f64::from_bits)))
}

fn snapshot_image_manifest_hash(image_names: &[String]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    hash = physical_hash_mix(hash, image_names.len() as u64);
    for name in image_names {
        hash = physical_hash_mix(hash, name.len() as u64);
        for byte in name.as_bytes() {
            hash = physical_hash_mix(hash, u64::from(*byte));
        }
    }
    hash
}

fn snapshot_feature_manifest_hash(features: &[FeatureSet]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    hash = physical_hash_mix(hash, features.len() as u64);
    for feature_set in features {
        hash = physical_hash_mix(hash, feature_set.keypoints.len() as u64);
        hash = physical_hash_mix(hash, feature_set.descriptors.len() as u64);
        for keypoint in &feature_set.keypoints {
            hash = physical_hash_mix(hash, keypoint.x.to_bits());
            hash = physical_hash_mix(hash, keypoint.y.to_bits());
        }
        for descriptor in &feature_set.descriptors {
            hash = physical_hash_mix(hash, descriptor.len() as u64);
            for value in descriptor {
                hash = physical_hash_mix(hash, u64::from(value.to_bits()));
            }
        }
    }
    hash
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SnapshotFeatureValidation {
    feature_counts: Vec<usize>,
    feature_manifest_hash: u64,
}

/// Reconstruct the exact v1 feature-manifest hash without retaining the
/// descriptor bank.  The first pass records a compact bitwise fingerprint for
/// every source file before dropping its descriptors.  This pass re-parses one
/// file at a time, verifies that the source is unchanged, and feeds the
/// calibrated in-memory keypoints plus the original descriptor bits through
/// the same hash stream as [`snapshot_feature_manifest_hash`].
fn snapshot_feature_validation_from_files(
    paths: &[PathBuf],
    features: &[FeatureSet],
    fingerprints: &[SnapshotFeatureFileFingerprint],
) -> Result<SnapshotFeatureValidation, String> {
    if paths.len() != features.len() || paths.len() != fingerprints.len() {
        return Err(format!(
            "snapshot feature replay manifest mismatch: {} source paths, {} feature sets, {} fingerprints",
            paths.len(),
            features.len(),
            fingerprints.len()
        ));
    }
    let mut hash = 0xcbf29ce484222325u64;
    hash = physical_hash_mix(hash, features.len() as u64);
    for (image, ((path, feature_set), expected)) in
        paths.iter().zip(features).zip(fingerprints).enumerate()
    {
        if feature_set.keypoints.len() != feature_set.descriptors.len() {
            return Err(format!(
                "snapshot keypoint-only image {image} has {} keypoints but {} placeholder descriptor rows",
                feature_set.keypoints.len(),
                feature_set.descriptors.len()
            ));
        }
        let source = read_feature_set(path).map_err(|error| {
            format!("cannot re-read snapshot feature source image {image} ({path:?}): {error}")
        })?;
        let observed = snapshot_feature_file_fingerprint(&source);
        if observed != *expected {
            return Err(format!(
                "snapshot feature source image {image} ({path:?}) changed between loads"
            ));
        }
        if observed.keypoint_count != feature_set.keypoints.len()
            || observed.descriptor_count != feature_set.descriptors.len()
        {
            return Err(format!(
                "snapshot feature source image {image} ({path:?}) has {} keypoints / {} descriptor rows, loaded {} / {}",
                observed.keypoint_count,
                observed.descriptor_count,
                feature_set.keypoints.len(),
                feature_set.descriptors.len(),
            ));
        }
        hash = physical_hash_mix(hash, feature_set.keypoints.len() as u64);
        hash = physical_hash_mix(hash, source.descriptors.len() as u64);
        for keypoint in &feature_set.keypoints {
            hash = physical_hash_mix(hash, keypoint.x.to_bits());
            hash = physical_hash_mix(hash, keypoint.y.to_bits());
        }
        for descriptor in &source.descriptors {
            hash = physical_hash_mix(hash, descriptor.len() as u64);
            for value in descriptor {
                hash = physical_hash_mix(hash, u64::from(value.to_bits()));
            }
        }
    }
    Ok(SnapshotFeatureValidation {
        feature_counts: features.iter().map(FeatureSet::len).collect(),
        feature_manifest_hash: hash,
    })
}

fn snapshot_intrinsics_bits(camera: &Camera) -> Result<[u64; 4], String> {
    let Some((fx, fy, cx, cy)) = camera.intrinsics() else {
        return Err("verified-pair snapshot requires a camera with intrinsics".into());
    };
    let values = [fx, fy, cx, cy];
    if values.iter().any(|value| !value.is_finite()) {
        return Err("camera intrinsics contain a non-finite value".into());
    }
    Ok(values.map(f64::to_bits))
}

/// The verifier knobs which affect the initial pair stream.  Paths and mapper
/// knobs are intentionally absent: importing a snapshot must not require the
/// original matcher input files or rerun any of those decisions.
fn snapshot_verifier_config(args: &Args) -> String {
    format!(
        "mode={:?};ratio_bits={:08x};min_matches={};cross_check=1;guided={};multiple_models={};min_e_f={:?};calibrated_prefer_essential={};refine_f2e={};strict_f2e={};calibrated_essential_primary={};force_essential={};force_essential_ratio_bits={:016x};force_essential_min={};force_essential_uncalibrated_only={};colmap_guided={}",
        args.verification_mode,
        args.match_ratio.to_bits(),
        args.min_matches,
        args.guided_matching,
        args.multiple_models,
        args.min_e_f_inlier_ratio,
        args.calibrated_prefer_essential,
        args.refine_uncalibrated_f_to_essential,
        args.strict_uncalibrated_f_to_essential,
        args.calibrated_essential_primary,
        args.force_essential_matches,
        args.force_essential_min_ef_ratio.to_bits(),
        args.force_essential_min_e_inliers,
        args.force_essential_uncalibrated_only,
        args.colmap_guided_matching,
    )
}

fn snapshot_export_config(args: &Args) -> String {
    format!("verified-pair-export-v1;{}", snapshot_verifier_config(args))
}

/// Hash the exact pair and correspondence order consumed by track building.
/// This is deliberately stronger than [`unordered_pairwise_edge_hash`]: pair
/// order, direction, accepted order, essential subset order, configuration,
/// and the stored essential matrix all contribute.
fn ordered_pairwise_edge_hash(pairwise: &[PairwiseMatches]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    hash = physical_hash_mix(hash, pairwise.len() as u64);
    for pair in pairwise {
        hash = physical_hash_mix(hash, pair.image_i as u64);
        hash = physical_hash_mix(hash, pair.image_j as u64);
        hash = physical_hash_mix(hash, configuration_code(pair.two_view_config) as u64);
        hash = physical_hash_mix(hash, pair.matches.len() as u64);
        for &(left, right) in &pair.matches {
            hash = physical_hash_mix(hash, left as u64);
            hash = physical_hash_mix(hash, right as u64);
        }
        match &pair.essential_matches {
            Some(matches) => {
                hash = physical_hash_mix(hash, 1);
                hash = physical_hash_mix(hash, matches.len() as u64);
                for &(left, right) in matches {
                    hash = physical_hash_mix(hash, left as u64);
                    hash = physical_hash_mix(hash, right as u64);
                }
            }
            None => hash = physical_hash_mix(hash, 0),
        }
        match &pair.essential_matrix {
            Some(matrix) => {
                hash = physical_hash_mix(hash, 1);
                for value in matrix.as_slice() {
                    hash = physical_hash_mix(hash, value.to_bits());
                }
            }
            None => hash = physical_hash_mix(hash, 0),
        }
    }
    hash
}

/// Run a versioned, plan-driven match worker without reloading the feature
/// bank between candidate shards.  Candidate manifests are preflighted before
/// the first snapshot is published so a duplicate pair or metadata mismatch
/// cannot leave a prefix that looks complete.  Verification itself remains
/// the existing `verify_pairs` implementation and each shard's temporary
/// result is dropped before the next shard starts.
fn run_persistent_match_worker(
    plan: &PersistentMatchWorkerPlan,
    features: &[FeatureSet],
    image_names: &[String],
    camera: &Camera,
    matcher: &PairMatcher,
    args: &Args,
    feature_validation: &SnapshotFeatureValidation,
) -> Result<(), Box<dyn std::error::Error>> {
    if plan.image_names != image_names {
        return Err("persistent match worker plan image order differs from loaded features".into());
    }
    let mut seen_pairs = HashSet::with_capacity(plan.pair_count);
    let mut expected_metadata: Option<BTreeMap<String, String>> = None;
    let mut total_pairs = 0usize;
    for shard in &plan.shards {
        let candidate_path = plan.root.join(&shard.candidate_path);
        let (candidates, metadata) =
            parse_candidate_manifest_with_metadata(&candidate_path, image_names)
                .map_err(std::io::Error::other)?;
        if let Some(expected) = expected_metadata.as_ref() {
            if expected != &metadata {
                return Err(format!(
                    "persistent match worker candidate shard {} metadata differs from shard 0",
                    shard.id
                )
                .into());
            }
        } else {
            expected_metadata = Some(metadata);
        }
        for &pair in &candidates {
            if !seen_pairs.insert(pair) {
                return Err(format!(
                    "persistent match worker candidate shards overlap at pair ({},{})",
                    pair.0, pair.1
                )
                .into());
            }
        }
        total_pairs = total_pairs
            .checked_add(candidates.len())
            .ok_or("persistent match worker candidate pair count overflow")?;
        // This pass only validates coverage/metadata.  Do not retain any
        // candidate vectors while the feature bank is resident; each shard is
        // parsed again immediately before verification below.
        drop(candidates);
    }
    if total_pairs != plan.pair_count {
        return Err(format!(
            "persistent match worker plan declares {} pairs but candidate shards contain {}",
            plan.pair_count, total_pairs
        )
        .into());
    }

    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "persistent-match-plan candidate_index_sha256={} feature_manifest_sha256={}",
        plan.candidate_index_sha256, plan.feature_manifest_sha256,
    )?;
    stdout.flush()?;
    for shard in &plan.shards {
        let candidate_path = plan.root.join(&shard.candidate_path);
        let (candidates, _candidate_metadata) =
            parse_candidate_manifest_with_metadata(&candidate_path, image_names)
                .map_err(std::io::Error::other)?;
        let started = std::time::Instant::now();
        let (mut pairwise, _stats, metadata) = verify_pairs(
            features,
            camera,
            &candidates,
            args.match_ratio,
            args.min_matches,
            args.verification_mode,
            matcher,
            true,
            args.guided_matching,
            args.multiple_models,
            args.min_e_f_inlier_ratio,
            args.calibrated_prefer_essential,
            args.refine_uncalibrated_f_to_essential,
            args.strict_uncalibrated_f_to_essential,
            args.calibrated_essential_primary,
            args.force_essential_matches,
            args.force_essential_min_ef_ratio,
            args.force_essential_min_e_inliers,
            args.force_essential_uncalibrated_only,
            None,
            None,
            None,
            None,
            args.colmap_guided_matching,
        );
        if pairwise.is_empty() {
            return Err(format!(
                "persistent match worker shard {} has no verified pair (lower --min-matches?)",
                shard.id
            )
            .into());
        }
        let edge_hash_before = unordered_pairwise_edge_hash(&pairwise);
        apply_union_traversal_order_with_features(
            &mut pairwise,
            args.union_traversal_order,
            features,
        );
        let edge_hash_after = unordered_pairwise_edge_hash(&pairwise);
        if edge_hash_before != edge_hash_after {
            return Err(format!(
                "persistent match worker shard {} changed the verified edge multiset: before={edge_hash_before:016x} after={edge_hash_after:016x}",
                shard.id
            )
            .into());
        }
        let accepted = pairwise
            .iter()
            .map(|pair| pair.matches.len())
            .sum::<usize>();
        let ordered_hash = ordered_pairwise_edge_hash(&pairwise);
        let unordered_hash = edge_hash_after;
        let snapshot_path = plan.root.join(&shard.snapshot_path);
        write_verified_pair_snapshot_atomic(
            &snapshot_path,
            image_names,
            features,
            camera,
            &pairwise,
            &metadata,
            args,
            feature_validation,
        )?;
        let elapsed = started.elapsed().as_secs_f64();
        writeln!(
            stdout,
            "persistent-match-complete shard_id={} candidate_path={} snapshot_path={} candidate_sha256={} candidate_pairs={} pairs={} accepted={} ordered_edge_fnv1a64={ordered_hash:016x} unordered_edge_fnv1a64={unordered_hash:016x} elapsed_s={elapsed:.9}",
            shard.id,
            shard.candidate_path.display(),
            shard.snapshot_path.display(),
            shard.candidate_sha256,
            candidates.len(),
            pairwise.len(),
            accepted,
        )?;
        stdout.flush()?;
        // `pairwise`, metadata, and the candidate vector are all shard-local;
        // dropping them at this boundary keeps the worker's result buffers
        // bounded even when the feature bank is large.
        drop(_candidate_metadata);
        drop(candidates);
        drop(metadata);
        drop(pairwise);
        trim_process_allocator();
    }
    Ok(())
}

fn verification_stats_from_snapshot(
    snapshot: &VerifiedPairSnapshot,
) -> Result<VerificationStats, String> {
    let mut stats = VerificationStats::default();
    for (index, pair) in snapshot.pairs.iter().enumerate() {
        let config = configuration_from_code(pair.config)?;
        if let Some(config) = config {
            stats.record(config);
        } else if pair.accepted_inlier_indices.is_empty() && !pair.matches.is_empty() {
            return Err(format!(
                "snapshot pair {index} has accepted matches but no configuration"
            ));
        }
    }
    Ok(stats)
}

fn validate_snapshot_for_run(
    snapshot: &VerifiedPairSnapshot,
    image_names: &[String],
    features: &[FeatureSet],
    camera: &Camera,
    precomputed_feature_validation: Option<&SnapshotFeatureValidation>,
) -> Result<Vec<PairwiseMatches>, String> {
    if snapshot.schema_version != verified_pair_snapshot::SCHEMA_VERSION {
        return Err(format!(
            "unsupported verified-pair snapshot schema {}",
            snapshot.schema_version
        ));
    }
    if snapshot.image_names != image_names {
        return Err(
            "verified-pair snapshot image manifest names do not match loaded images".into(),
        );
    }
    let computed_feature_counts: Vec<usize> = features.iter().map(|f| f.keypoints.len()).collect();
    let feature_counts = precomputed_feature_validation
        .map(|validation| validation.feature_counts.as_slice())
        .unwrap_or(computed_feature_counts.as_slice());
    let snapshot_counts: Vec<usize> = snapshot
        .feature_counts
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .map_err(|_| format!("snapshot feature count {value} does not fit usize"))
        })
        .collect::<Result<_, _>>()?;
    if snapshot_counts != feature_counts {
        return Err(format!(
            "verified-pair snapshot feature counts do not match loaded features ({snapshot_counts:?} vs {feature_counts:?})"
        ));
    }
    let image_hash = snapshot_image_manifest_hash(image_names);
    if snapshot.image_manifest_hash != image_hash {
        return Err(format!(
            "verified-pair snapshot image manifest hash mismatch: stored {:016x}, loaded {image_hash:016x}",
            snapshot.image_manifest_hash
        ));
    }
    let feature_hash = precomputed_feature_validation.map_or_else(
        || snapshot_feature_manifest_hash(features),
        |validation| validation.feature_manifest_hash,
    );
    if snapshot.feature_manifest_hash != feature_hash {
        return Err(format!(
            "verified-pair snapshot feature manifest hash mismatch: stored {:016x}, loaded {feature_hash:016x}",
            snapshot.feature_manifest_hash
        ));
    }
    if snapshot.width != u64::from(camera.width) || snapshot.height != u64::from(camera.height) {
        return Err(format!(
            "verified-pair snapshot camera dimensions {}x{} do not match loaded {}x{}",
            snapshot.width, snapshot.height, camera.width, camera.height
        ));
    }
    if snapshot.intrinsics_bits != snapshot_intrinsics_bits(camera)? {
        return Err("verified-pair snapshot camera intrinsics do not match loaded camera".into());
    }
    if effective_config_hash(&snapshot.effective_config) != snapshot.effective_config_hash {
        return Err("verified-pair snapshot effective-config checksum is invalid".into());
    }
    if effective_config_hash(&snapshot.verifier_config) != snapshot.verifier_config_hash {
        return Err("verified-pair snapshot verifier-config checksum is invalid".into());
    }
    let pairwise = pairwise_from_snapshot(snapshot, feature_counts)?;
    let ordered_hash = ordered_pairwise_edge_hash(&pairwise);
    if snapshot.pair_order_hash != ordered_hash {
        return Err(format!(
            "verified-pair snapshot pair-order hash mismatch: stored {:016x}, loaded {ordered_hash:016x}",
            snapshot.pair_order_hash
        ));
    }
    let unordered_hash = unordered_pairwise_edge_hash(&pairwise);
    if snapshot.unordered_edge_hash != unordered_hash {
        return Err(format!(
            "verified-pair snapshot unordered-edge hash mismatch: stored {:016x}, loaded {unordered_hash:016x}",
            snapshot.unordered_edge_hash
        ));
    }
    let accepted_match_count: usize = pairwise.iter().map(|pair| pair.matches.len()).sum();
    if snapshot.accepted_match_count != accepted_match_count as u64 {
        return Err(format!(
            "verified-pair snapshot accepted-match count {} does not match loaded {accepted_match_count}",
            snapshot.accepted_match_count
        ));
    }
    Ok(pairwise)
}

fn pairwise_from_snapshot(
    snapshot: &VerifiedPairSnapshot,
    feature_counts: &[usize],
) -> Result<Vec<PairwiseMatches>, String> {
    let mut pairs = Vec::with_capacity(snapshot.pairs.len());
    let mut seen = HashSet::new();
    for (pair_number, record) in snapshot.pairs.iter().enumerate() {
        let image_i = usize::try_from(record.image_i)
            .map_err(|_| format!("snapshot pair {pair_number} image_i does not fit usize"))?;
        let image_j = usize::try_from(record.image_j)
            .map_err(|_| format!("snapshot pair {pair_number} image_j does not fit usize"))?;
        if image_i == image_j || image_i >= feature_counts.len() || image_j >= feature_counts.len()
        {
            return Err(format!(
                "snapshot pair {pair_number} has invalid image indices ({image_i},{image_j})"
            ));
        }
        let key = (image_i.min(image_j), image_i.max(image_j));
        if !seen.insert(key) {
            return Err(format!(
                "snapshot contains duplicate image pair ({},{})",
                key.0, key.1
            ));
        }
        let config = configuration_from_code(record.config)?;
        if record.calibrated != (config == Some(ConfigurationType::Calibrated)) {
            return Err(format!(
                "snapshot pair {pair_number} calibrated flag disagrees with configuration code"
            ));
        }
        let validate_indices = |label: &str, values: &[u64], limit: usize| -> Result<(), String> {
            for &value in values {
                let index = usize::try_from(value).map_err(|_| {
                    format!("snapshot pair {pair_number} {label} index does not fit usize")
                })?;
                if index >= limit {
                    return Err(format!(
                        "snapshot pair {pair_number} {label} index {index} is outside 0..{limit}"
                    ));
                }
            }
            Ok(())
        };
        if let Some(max_index) = record.accepted_inlier_indices.iter().max() {
            if *max_index >= record.raw_match_count {
                return Err(format!(
                    "snapshot pair {pair_number} accepted inlier index {max_index} >= raw match count {}",
                    record.raw_match_count
                ));
            }
        }
        if let Some(max_index) = record.essential_inlier_indices.iter().max() {
            if *max_index >= record.raw_match_count {
                return Err(format!(
                    "snapshot pair {pair_number} essential inlier index {max_index} >= raw match count {}",
                    record.raw_match_count
                ));
            }
        }
        let raw_match_count = usize::try_from(record.raw_match_count).map_err(|_| {
            format!(
                "snapshot pair {pair_number} raw match count {} does not fit usize",
                record.raw_match_count
            )
        })?;
        if raw_match_count != record.raw_matches.len() {
            return Err(format!(
                "snapshot pair {pair_number} raw match count {} does not match stream length {}",
                raw_match_count,
                record.raw_matches.len()
            ));
        }
        let raw_matches = record
            .raw_matches
            .iter()
            .map(|&(left, right)| {
                let left = usize::try_from(left).map_err(|_| {
                    format!("snapshot pair {pair_number} raw query index does not fit usize")
                })?;
                let right = usize::try_from(right).map_err(|_| {
                    format!("snapshot pair {pair_number} raw train index does not fit usize")
                })?;
                if left >= feature_counts[image_i] || right >= feature_counts[image_j] {
                    return Err(format!(
                        "snapshot pair {pair_number} raw match ({left},{right}) is outside feature counts ({},{})",
                        feature_counts[image_i], feature_counts[image_j]
                    ));
                }
                Ok((left, right))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let matches = record
            .matches
            .iter()
            .map(|&(left, right)| {
                let left = usize::try_from(left)
                    .map_err(|_| format!("snapshot pair {pair_number} query index does not fit usize"))?;
                let right = usize::try_from(right)
                    .map_err(|_| format!("snapshot pair {pair_number} train index does not fit usize"))?;
                if left >= feature_counts[image_i] || right >= feature_counts[image_j] {
                    return Err(format!(
                        "snapshot pair {pair_number} accepted match ({left},{right}) is outside feature counts ({},{})",
                        feature_counts[image_i], feature_counts[image_j]
                    ));
                }
                Ok((left, right))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let essential_matches = record
            .essential_matches
            .as_ref()
            .map(|values| {
                values
                    .iter()
                    .map(|&(left, right)| {
                        let left = usize::try_from(left).map_err(|_| {
                            format!("snapshot pair {pair_number} essential query index does not fit usize")
                        })?;
                        let right = usize::try_from(right).map_err(|_| {
                            format!("snapshot pair {pair_number} essential train index does not fit usize")
                        })?;
                        if left >= feature_counts[image_i] || right >= feature_counts[image_j] {
                            return Err(format!(
                                "snapshot pair {pair_number} essential match ({left},{right}) is outside feature counts"
                            ));
                        }
                        Ok((left, right))
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .transpose()?;
        validate_indices(
            "accepted inlier",
            &record.accepted_inlier_indices,
            raw_match_count,
        )?;
        validate_indices(
            "essential inlier",
            &record.essential_inlier_indices,
            raw_match_count,
        )?;
        if record.accepted_inlier_indices.len() != matches.len() {
            return Err(format!(
                "snapshot pair {pair_number} has {} accepted indices but {} accepted matches",
                record.accepted_inlier_indices.len(),
                matches.len()
            ));
        }
        for (position, &raw_index) in record.accepted_inlier_indices.iter().enumerate() {
            let raw_index = usize::try_from(raw_index).map_err(|_| {
                format!("snapshot pair {pair_number} accepted index does not fit usize")
            })?;
            if raw_matches[raw_index] != matches[position] {
                return Err(format!(
                    "snapshot pair {pair_number} accepted match at position {position} disagrees with raw index {raw_index}"
                ));
            }
        }
        if let Some(essential_matches) = &essential_matches {
            if record.essential_inlier_indices.len() != essential_matches.len() {
                return Err(format!(
                    "snapshot pair {pair_number} has {} essential indices but {} essential matches",
                    record.essential_inlier_indices.len(),
                    essential_matches.len()
                ));
            }
            for (position, &raw_index) in record.essential_inlier_indices.iter().enumerate() {
                let raw_index = usize::try_from(raw_index).map_err(|_| {
                    format!("snapshot pair {pair_number} essential index does not fit usize")
                })?;
                if raw_matches[raw_index] != essential_matches[position] {
                    return Err(format!(
                        "snapshot pair {pair_number} essential match at position {position} disagrees with raw index {raw_index}"
                    ));
                }
            }
        } else if !record.essential_inlier_indices.is_empty() {
            return Err(format!(
                "snapshot pair {pair_number} has essential indices but no essential matches"
            ));
        }
        pairs.push(PairwiseMatches {
            image_i,
            image_j,
            matches,
            two_view_config: config,
            essential_matches,
            essential_matrix: matrix_from_bits(record.essential_matrix_bits),
        });
    }
    Ok(pairs)
}

fn parse_config_token(tok: &str) -> Result<ConfigurationType, Box<dyn std::error::Error>> {
    if let Ok(n) = tok.parse::<usize>() {
        return Ok(match n {
            0 => ConfigurationType::Undefined,
            1 => ConfigurationType::Degenerate,
            2 => ConfigurationType::Uncalibrated,
            3 => ConfigurationType::Calibrated,
            4 => ConfigurationType::Planar,
            5 => ConfigurationType::Panoramic,
            6 => ConfigurationType::PlanarOrPanoramic,
            7 => ConfigurationType::Watermark,
            8 => ConfigurationType::Multiple,
            other => return Err(format!("unknown config code {other}").into()),
        });
    }
    Ok(match tok {
        "Undefined" => ConfigurationType::Undefined,
        "Degenerate" => ConfigurationType::Degenerate,
        "Uncalibrated" => ConfigurationType::Uncalibrated,
        "Calibrated" => ConfigurationType::Calibrated,
        "Planar" => ConfigurationType::Planar,
        "Panoramic" => ConfigurationType::Panoramic,
        "PlanarOrPanoramic" => ConfigurationType::PlanarOrPanoramic,
        "Watermark" => ConfigurationType::Watermark,
        "Multiple" => ConfigurationType::Multiple,
        other => return Err(format!("unknown config {other:?}").into()),
    })
}

fn parse_imported_verified_pairs_file(
    path: &Path,
    image_names: &[String],
) -> Result<Vec<ImportedVerifiedPair>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    let name_count: usize = lines
        .next()
        .ok_or("import verified: missing image count")?
        .parse()?;
    if name_count != image_names.len() {
        return Err(format!(
            "import verified: file has {name_count} names, run has {}",
            image_names.len()
        )
        .into());
    }
    for (idx, expected_name) in image_names.iter().enumerate().take(name_count) {
        let file_name = lines
            .next()
            .ok_or("import verified: truncated image name list")?;
        if file_name != expected_name {
            return Err(format!(
                "import verified: name mismatch at {idx}: file {file_name:?} vs run {:?}",
                expected_name
            )
            .into());
        }
    }
    let pair_count: usize = lines
        .next()
        .ok_or("import verified: missing pair count")?
        .parse()?;
    let mut out = Vec::with_capacity(pair_count);
    for _ in 0..pair_count {
        let head: Vec<&str> = lines
            .next()
            .ok_or("import verified: truncated pair header")?
            .split_whitespace()
            .collect();
        if head.len() < 13 {
            return Err("import verified: pair header needs i j count config e(9)".into());
        }
        let i: usize = head[0].parse()?;
        let j: usize = head[1].parse()?;
        let count: usize = head[2].parse()?;
        let config = parse_config_token(head[3])?;
        let e_vals: Vec<f64> = head[4..13]
            .iter()
            .map(|t| t.parse())
            .collect::<Result<_, _>>()?;
        let essential_matrix = if e_vals.iter().any(|v| v.abs() > 1e-15) {
            Some(Matrix3::from_row_slice(&e_vals))
        } else {
            None
        };
        let mut matches = Vec::with_capacity(count);
        for _ in 0..count {
            let m: Vec<usize> = lines
                .next()
                .ok_or("import verified: truncated correspondence")?
                .split_whitespace()
                .map(|t| t.parse())
                .collect::<Result<_, _>>()?;
            if m.len() != 2 {
                return Err("import verified: correspondence needs qi tj".into());
            }
            matches.push((m[0], m[1]));
        }
        out.push(ImportedVerifiedPair {
            image_i: i,
            image_j: j,
            matches,
            config,
            essential_matrix,
        });
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VerifiedPairOracle {
    inliers: usize,
    config: ConfigurationType,
}

fn verified_pair_oracle_map(
    imported: &[ImportedVerifiedPair],
) -> HashMap<(usize, usize), VerifiedPairOracle> {
    imported
        .iter()
        .map(|pair| {
            (
                (
                    pair.image_i.min(pair.image_j),
                    pair.image_i.max(pair.image_j),
                ),
                VerifiedPairOracle {
                    inliers: pair.matches.len(),
                    config: pair.config,
                },
            )
        })
        .collect()
}

fn verified_pairs_to_pairwise(imported: Vec<ImportedVerifiedPair>) -> Vec<PairwiseMatches> {
    imported
        .into_iter()
        .map(|p| {
            let essential_matches = match p.config {
                ConfigurationType::Calibrated | ConfigurationType::Multiple
                    if p.essential_matrix.is_some() =>
                {
                    Some(p.matches.clone())
                }
                _ => None,
            };
            PairwiseMatches {
                image_i: p.image_i,
                image_j: p.image_j,
                matches: p.matches,
                two_view_config: Some(p.config),
                essential_matches,
                essential_matrix: p.essential_matrix,
            }
        })
        .collect()
}

/// Read the optional four-column sidecar emitted by `export_features_to_dir`.
/// A missing sidecar is intentional for legacy feature dumps and means that
/// every row remains its own locus.
fn read_locus_sidecar(
    path: &Path,
    expected_rows: usize,
) -> Result<Option<Vec<FeatureLocusMetadata>>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values = line
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                format!(
                    "{}:{}: invalid locus metadata: {error}",
                    path.display(),
                    line_number + 1
                )
            })?;
        if values.len() != 4 {
            return Err(format!(
                "{}:{}: locus metadata needs x y scale orientation",
                path.display(),
                line_number + 1
            )
            .into());
        }
        rows.push(FeatureLocusMetadata {
            x: values[0],
            y: values[1],
            scale: values[2],
            orientation: values[3],
        });
    }
    if rows.len() != expected_rows {
        return Err(format!(
            "{}: {} metadata rows, expected {} feature rows",
            path.display(),
            rows.len(),
            expected_rows
        )
        .into());
    }
    Ok(Some(rows))
}

/// Parse the compact COLMAP six-column keypoint representation when the
/// descriptor payload has the usual 128 dimensions:
/// `x y a11 a12 a21 a22 d0 ... d127`.  Existing external files use
/// `x y score descriptor...` and take the shared parser path instead.
fn read_six_column_locus_features(
    path: &Path,
) -> Result<Option<(FeatureSet, Vec<FeatureLocusMetadata>)>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let rows = text
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            (!line.is_empty() && !line.starts_with('#'))
                .then(|| line.split_whitespace().collect::<Vec<_>>())
        })
        .collect::<Vec<_>>();
    if rows.is_empty() || rows[0].len() != 134 {
        return Ok(None);
    }
    if rows.iter().any(|row| row.len() != 134) {
        return Err(format!(
            "{}: six-column feature rows have inconsistent field counts",
            path.display()
        )
        .into());
    }
    let mut keypoints = Vec::with_capacity(rows.len());
    let mut descriptors = Vec::with_capacity(rows.len());
    let mut metadata = Vec::with_capacity(rows.len());
    for (row_number, row) in rows.iter().enumerate() {
        let parse = |column: usize| {
            row[column].parse::<f64>().map_err(|error| {
                format!(
                    "{} row {}: invalid numeric field {column}: {error}",
                    path.display(),
                    row_number + 1
                )
            })
        };
        let x = parse(0)?;
        let y = parse(1)?;
        let a11 = parse(2)?;
        let a12 = parse(3)?;
        let a21 = parse(4)?;
        let a22 = parse(5)?;
        let descriptor = row[6..]
            .iter()
            .map(|value| {
                value.parse::<f32>().map_err(|error| {
                    format!(
                        "{} row {}: invalid descriptor: {error}",
                        path.display(),
                        row_number + 1
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        keypoints.push(Point2::new(x, y));
        descriptors.push(descriptor);
        metadata.push(FeatureLocusMetadata {
            x,
            y,
            scale: (a11 * a22 - a12 * a21).abs().sqrt(),
            orientation: a21.atan2(a11).rem_euclid(std::f64::consts::TAU),
        });
    }
    Ok(Some((FeatureSet::new(keypoints, descriptors)?, metadata)))
}

/// Read one feature file with the same parser used by the historical batch
/// loader.  The keypoint-only snapshot replay deliberately calls this helper
/// once per file, so the descriptor payload is released before the next image
/// is parsed.
fn read_feature_set(path: &Path) -> Result<FeatureSet, Box<dyn std::error::Error>> {
    if let Some((feature_set, _)) = read_six_column_locus_features(path)? {
        return Ok(feature_set);
    }
    Ok(read_external_deep_features_txt(path)?.into_feature_set()?)
}

fn list_feature_files(
    dir: &Path,
    feature_suffix: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut files: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.ends_with(feature_suffix))
        .collect();
    files.sort();
    Ok(files)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SnapshotFeatureFileFingerprint {
    keypoint_hash: u64,
    descriptor_hash: u64,
    keypoint_count: usize,
    descriptor_count: usize,
}

fn snapshot_feature_file_fingerprint(feature_set: &FeatureSet) -> SnapshotFeatureFileFingerprint {
    let mut keypoint_hash = 0xcbf29ce484222325u64;
    keypoint_hash = physical_hash_mix(keypoint_hash, feature_set.keypoints.len() as u64);
    for keypoint in &feature_set.keypoints {
        keypoint_hash = physical_hash_mix(keypoint_hash, keypoint.x.to_bits());
        keypoint_hash = physical_hash_mix(keypoint_hash, keypoint.y.to_bits());
    }
    let mut descriptor_hash = 0xcbf29ce484222325u64;
    descriptor_hash = physical_hash_mix(descriptor_hash, feature_set.descriptors.len() as u64);
    for descriptor in &feature_set.descriptors {
        descriptor_hash = physical_hash_mix(descriptor_hash, descriptor.len() as u64);
        for value in descriptor {
            descriptor_hash = physical_hash_mix(descriptor_hash, u64::from(value.to_bits()));
        }
    }
    SnapshotFeatureFileFingerprint {
        keypoint_hash,
        descriptor_hash,
        keypoint_count: feature_set.keypoints.len(),
        descriptor_count: feature_set.descriptors.len(),
    }
}

/// Read every `*<feature_suffix>` file in `dir`, sorted lexically, returning the
/// per-image feature sets and their COLMAP image names.
fn load_images(
    dir: &Path,
    feature_suffix: &str,
    image_suffix: &str,
) -> Result<
    (
        Vec<FeatureSet>,
        Vec<String>,
        Vec<Option<Vec<FeatureLocusMetadata>>>,
    ),
    Box<dyn std::error::Error>,
> {
    let files = list_feature_files(dir, feature_suffix)?;
    let mut features = Vec::new();
    let mut names = Vec::new();
    let mut locus_metadata = Vec::new();
    for f in &files {
        let feature_path = dir.join(f);
        let (feature_set, metadata) = if let Some(parsed) =
            read_six_column_locus_features(&feature_path)?
        {
            parsed
        } else {
            let feature_set = read_external_deep_features_txt(&feature_path)?.into_feature_set()?;
            let stem = f.strip_suffix(feature_suffix).unwrap_or(f);
            let metadata_path = dir.join(format!("{stem}_loci.txt"));
            let metadata = read_locus_sidecar(&metadata_path, feature_set.len())?;
            (feature_set, metadata.unwrap_or_default())
        };
        features.push(feature_set);
        locus_metadata.push((!metadata.is_empty()).then_some(metadata));
        names.push(image_name_for(f, feature_suffix, image_suffix));
    }
    Ok((features, names, locus_metadata))
}

/// The memory-bounded feature representation used by explicit snapshot
/// replay.  `paths` preserves the exact lexical source order so the original
/// descriptor-bound feature manifest can be recomputed after calibration.
#[derive(Debug)]
struct SnapshotKeypointsOnlyLoad {
    features: Vec<FeatureSet>,
    image_names: Vec<String>,
    locus_metadata: Vec<Option<Vec<FeatureLocusMetadata>>>,
    paths: Vec<PathBuf>,
    fingerprints: Vec<SnapshotFeatureFileFingerprint>,
}

/// Load file-backed features one image at a time while retaining only pixels,
/// locus metadata, and one empty descriptor row per keypoint.  Keeping the
/// outer descriptor row count is intentional: downstream row-index validation
/// treats it as part of the feature shape, while ordinary incremental mapping
/// never reads descriptor values after an imported snapshot.
fn load_images_keypoints_only(
    dir: &Path,
    feature_suffix: &str,
    image_suffix: &str,
) -> Result<SnapshotKeypointsOnlyLoad, Box<dyn std::error::Error>> {
    let files = list_feature_files(dir, feature_suffix)?;
    let mut features = Vec::with_capacity(files.len());
    let mut image_names = Vec::with_capacity(files.len());
    let mut locus_metadata = Vec::with_capacity(files.len());
    let mut paths = Vec::with_capacity(files.len());
    let mut fingerprints = Vec::with_capacity(files.len());
    for file_name in files {
        let feature_path = dir.join(&file_name);
        let (feature_set, metadata) = if let Some(parsed) =
            read_six_column_locus_features(&feature_path)?
        {
            parsed
        } else {
            let feature_set = read_external_deep_features_txt(&feature_path)?.into_feature_set()?;
            let stem = file_name.strip_suffix(feature_suffix).unwrap_or(&file_name);
            let metadata_path = dir.join(format!("{stem}_loci.txt"));
            let metadata = read_locus_sidecar(&metadata_path, feature_set.len())?;
            (feature_set, metadata.unwrap_or_default())
        };
        let fingerprint = snapshot_feature_file_fingerprint(&feature_set);
        let FeatureSet {
            keypoints,
            descriptors,
        } = feature_set;
        let descriptor_rows = descriptors.len();
        if descriptor_rows != keypoints.len() {
            return Err(format!(
                "{}: parser returned {} descriptors for {} keypoints",
                feature_path.display(),
                descriptor_rows,
                keypoints.len()
            )
            .into());
        }
        // Moving `keypoints` out and replacing the descriptor rows drops the
        // parsed payload before the next loop iteration.
        let row_count = keypoints.len();
        features.push(FeatureSet {
            keypoints,
            descriptors: (0..row_count).map(|_| Vec::new()).collect(),
        });
        locus_metadata.push((!metadata.is_empty()).then_some(metadata));
        image_names.push(image_name_for(&file_name, feature_suffix, image_suffix));
        paths.push(feature_path);
        fingerprints.push(fingerprint);
    }
    Ok(SnapshotKeypointsOnlyLoad {
        features,
        image_names,
        locus_metadata,
        paths,
        fingerprints,
    })
}

/// Replace only keypoint coordinates after a verified-pair snapshot has been
/// validated against the base feature directory.
///
/// The snapshot's correspondence indices refer to rows, so this diagnostic
/// path must never silently reorder rows or accept a descriptor mismatch.  A
/// bitwise descriptor comparison (rather than an approximate float comparison)
/// makes that contract explicit, including signed zero and NaN payloads.  The
/// caller loads the replacement directory with [`load_images`], which applies
/// the same lexical image ordering and feature parser as the base directory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SnapshotCoordinateOverrideStats {
    images: usize,
    rows: usize,
    changed_rows: usize,
}

fn apply_snapshot_coordinate_override(
    base_features: &mut [FeatureSet],
    base_image_names: &[String],
    override_features: &[FeatureSet],
    override_image_names: &[String],
) -> Result<SnapshotCoordinateOverrideStats, String> {
    if base_features.len() != base_image_names.len() {
        return Err(format!(
            "base feature/image manifest mismatch: {} feature sets vs {} names",
            base_features.len(),
            base_image_names.len()
        ));
    }
    if override_features.len() != override_image_names.len() {
        return Err(format!(
            "coordinate override feature/image manifest mismatch: {} feature sets vs {} names",
            override_features.len(),
            override_image_names.len()
        ));
    }
    if base_image_names != override_image_names {
        let first_difference = base_image_names
            .iter()
            .zip(override_image_names)
            .position(|(base, replacement)| base != replacement)
            .unwrap_or(base_image_names.len().min(override_image_names.len()));
        return Err(format!(
            "coordinate override image names/order do not match at row {first_difference}: base={:?}, override={:?}",
            base_image_names.get(first_difference),
            override_image_names.get(first_difference),
        ));
    }
    let mut stats = SnapshotCoordinateOverrideStats {
        images: base_features.len(),
        ..SnapshotCoordinateOverrideStats::default()
    };
    for (image_index, (base, replacement)) in
        base_features.iter_mut().zip(override_features).enumerate()
    {
        if base.keypoints.len() != replacement.keypoints.len()
            || base.descriptors.len() != replacement.descriptors.len()
        {
            return Err(format!(
                "coordinate override row count mismatch for {}: base keypoints/descriptors={}/{}, override={}/{}",
                base_image_names[image_index],
                base.keypoints.len(),
                base.descriptors.len(),
                replacement.keypoints.len(),
                replacement.descriptors.len(),
            ));
        }
        for (row, ((base_descriptor, replacement_descriptor), replacement_keypoint)) in base
            .descriptors
            .iter()
            .zip(&replacement.descriptors)
            .zip(&replacement.keypoints)
            .enumerate()
        {
            if base_descriptor.len() != replacement_descriptor.len()
                || base_descriptor.iter().zip(replacement_descriptor).any(
                    |(base_value, replacement_value)| {
                        base_value.to_bits() != replacement_value.to_bits()
                    },
                )
            {
                return Err(format!(
                    "coordinate override descriptor/index mismatch at {} row {}",
                    base_image_names[image_index], row
                ));
            }
            if !replacement_keypoint.x.is_finite() || !replacement_keypoint.y.is_finite() {
                return Err(format!(
                    "coordinate override has non-finite keypoint at {} row {}",
                    base_image_names[image_index], row
                ));
            }
            if base.keypoints[row] != *replacement_keypoint {
                stats.changed_rows += 1;
            }
            stats.rows += 1;
        }
        // The descriptor vectors are deliberately left untouched.  Replacing
        // the keypoint vector only is what keeps every snapshot feature index
        // and descriptor byte identical.
        base.keypoints.clone_from(&replacement.keypoints);
    }
    Ok(stats)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct LocusCanonicalizationStats {
    metadata_images: usize,
    metadata_rows: usize,
    physical_loci: usize,
    collapsed_rows: usize,
    input_matches: usize,
    output_matches: usize,
    deduplicated_matches: usize,
    changed_pairs: usize,
}

fn finite_order(value: f64) -> (u8, f64) {
    if value.is_finite() {
        (0, value)
    } else if value.is_nan() {
        (2, 0.0)
    } else if value.is_sign_negative() {
        (1, 0.0)
    } else {
        (3, 0.0)
    }
}

fn finite_order_cmp(lhs: f64, rhs: f64) -> CmpOrdering {
    let (lhs_class, lhs_value) = finite_order(lhs);
    let (rhs_class, rhs_value) = finite_order(rhs);
    lhs_class
        .cmp(&rhs_class)
        .then_with(|| lhs_value.total_cmp(&rhs_value))
}

fn locus_row_cmp(
    features: &[FeatureSet],
    metadata: &[Option<Vec<FeatureLocusMetadata>>],
    image: usize,
    lhs: usize,
    rhs: usize,
) -> CmpOrdering {
    let lhs_metadata = metadata
        .get(image)
        .and_then(|rows| rows.as_ref())
        .and_then(|rows| rows.get(lhs));
    let rhs_metadata = metadata
        .get(image)
        .and_then(|rows| rows.as_ref())
        .and_then(|rows| rows.get(rhs));
    let metadata_order = match (lhs_metadata, rhs_metadata) {
        (Some(lhs), Some(rhs)) => feature_locus_key(*lhs)
            .cmp(&feature_locus_key(*rhs))
            .then_with(|| finite_order_cmp(lhs.orientation, rhs.orientation)),
        (Some(_), None) => CmpOrdering::Less,
        (None, Some(_)) => CmpOrdering::Greater,
        (None, None) => CmpOrdering::Equal,
    };
    metadata_order
        .then_with(|| {
            let lhs = features.get(image).and_then(|set| set.keypoints.get(lhs));
            let rhs = features.get(image).and_then(|set| set.keypoints.get(rhs));
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => {
                    finite_order_cmp(lhs.x, rhs.x).then_with(|| finite_order_cmp(lhs.y, rhs.y))
                }
                (Some(_), None) => CmpOrdering::Less,
                (None, Some(_)) => CmpOrdering::Greater,
                (None, None) => CmpOrdering::Equal,
            }
        })
        .then_with(|| {
            let lhs = features.get(image).and_then(|set| set.descriptors.get(lhs));
            let rhs = features.get(image).and_then(|set| set.descriptors.get(rhs));
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => canonical_descriptor_cmp(lhs, rhs),
                (Some(_), None) => CmpOrdering::Less,
                (None, Some(_)) => CmpOrdering::Greater,
                (None, None) => CmpOrdering::Equal,
            }
        })
        .then_with(|| lhs.cmp(&rhs))
}

/// Build an old-row → representative-row map for every image.  Only rows
/// with complete finite metadata participate in physical grouping; absent or
/// malformed metadata intentionally falls back to identity for that row.
fn build_locus_representatives(
    features: &[FeatureSet],
    metadata: &[Option<Vec<FeatureLocusMetadata>>],
) -> Result<(Vec<Vec<usize>>, LocusCanonicalizationStats), String> {
    if features.len() != metadata.len() {
        return Err(format!(
            "orientation locus canonicalization: {} feature sets but {} metadata sets",
            features.len(),
            metadata.len()
        ));
    }
    let mut maps = Vec::with_capacity(features.len());
    let mut stats = LocusCanonicalizationStats::default();
    for (image, set) in features.iter().enumerate() {
        let mut representatives: Vec<usize> = (0..set.len()).collect();
        let Some(rows) = metadata[image].as_ref() else {
            maps.push(representatives);
            continue;
        };
        if rows.len() != set.len() {
            return Err(format!(
                "orientation locus canonicalization: image {image} has {} metadata rows but {} features",
                rows.len(),
                set.len()
            ));
        }
        stats.metadata_images += 1;
        stats.metadata_rows += rows.len();
        let mut groups: HashMap<FeatureLocusKey, Vec<usize>> = HashMap::new();
        for (row, &row_metadata) in rows.iter().enumerate() {
            if let Some(key) = feature_locus_key(row_metadata) {
                groups.entry(key).or_default().push(row);
            }
        }
        stats.physical_loci += groups.len();
        for rows in groups.values() {
            let representative = rows
                .iter()
                .copied()
                .min_by(|lhs, rhs| locus_row_cmp(features, metadata, image, *lhs, *rhs))
                .expect("non-empty locus group");
            for &row in rows {
                representatives[row] = representative;
            }
            stats.collapsed_rows += rows.len().saturating_sub(1);
        }
        maps.push(representatives);
    }
    Ok((maps, stats))
}

fn descriptor_squared_distance(lhs: &[f32], rhs: &[f32]) -> f64 {
    if lhs.len() != rhs.len() {
        return f64::INFINITY;
    }
    let mut sum = 0.0;
    for (&lhs, &rhs) in lhs.iter().zip(rhs) {
        let delta = f64::from(lhs) - f64::from(rhs);
        if !delta.is_finite() {
            return f64::INFINITY;
        }
        sum += delta * delta;
    }
    if sum.is_finite() {
        sum
    } else {
        f64::INFINITY
    }
}

fn match_geometry_residual(
    features: &[FeatureSet],
    pair: &PairwiseMatches,
    keypoint_i: usize,
    keypoint_j: usize,
    camera: Option<&Camera>,
) -> Option<f64> {
    let essential = pair.essential_matrix.as_ref()?;
    let camera = camera?;
    let point_i = features.get(pair.image_i)?.keypoints.get(keypoint_i)?;
    let point_j = features.get(pair.image_j)?.keypoints.get(keypoint_j)?;
    normalized_essential_squared_sampson_error(
        essential,
        &TwoViewCorrespondence::new(*point_i, *point_j),
        camera,
    )
}

fn match_candidate_cmp(
    lhs: &(usize, usize, f64, Option<f64>),
    rhs: &(usize, usize, f64, Option<f64>),
) -> CmpOrdering {
    let distance_order = match (lhs.2.is_finite(), rhs.2.is_finite()) {
        (true, true) => lhs.2.total_cmp(&rhs.2),
        (true, false) => CmpOrdering::Less,
        (false, true) => CmpOrdering::Greater,
        (false, false) => CmpOrdering::Equal,
    };
    distance_order
        .then_with(|| match (lhs.3, rhs.3) {
            (Some(lhs), Some(rhs)) => match (lhs.is_finite(), rhs.is_finite()) {
                (true, true) => lhs.total_cmp(&rhs),
                (true, false) => CmpOrdering::Less,
                (false, true) => CmpOrdering::Greater,
                (false, false) => CmpOrdering::Equal,
            },
            (Some(_), None) => CmpOrdering::Less,
            (None, Some(_)) => CmpOrdering::Greater,
            (None, None) => CmpOrdering::Equal,
        })
        .then_with(|| lhs.0.cmp(&rhs.0))
        .then_with(|| lhs.1.cmp(&rhs.1))
}

/// Remap accepted correspondences from orientation rows to one representative
/// row per physical locus.  Matching still ran on every descriptor variant;
/// this post-verification step only prevents two orientations of one locus
/// from entering one multi-view track as separate same-image observations.
/// Duplicate locus-pairs choose the lowest primary descriptor distance, then
/// calibrated geometric residual when available, then stable source indices.
fn canonicalize_pairwise_loci(
    features: &[FeatureSet],
    metadata: &[Option<Vec<FeatureLocusMetadata>>],
    pairwise: &mut [PairwiseMatches],
    camera: Option<&Camera>,
) -> Result<LocusCanonicalizationStats, String> {
    let (representatives, mut stats) = build_locus_representatives(features, metadata)?;
    // A legacy feature dump has no physical-locus metadata.  In that case the
    // opt-in flag must be a strict no-op: even sorting an identity-mapped
    // match stream would change the legacy UnionFind traversal basin.
    if stats.metadata_images == 0 {
        for pair in pairwise {
            stats.input_matches += pair.matches.len();
            stats.output_matches += pair.matches.len();
        }
        return Ok(stats);
    }
    for pair in pairwise {
        let map_i = representatives.get(pair.image_i).ok_or_else(|| {
            format!(
                "orientation locus canonicalization: invalid image index {}",
                pair.image_i
            )
        })?;
        let map_j = representatives.get(pair.image_j).ok_or_else(|| {
            format!(
                "orientation locus canonicalization: invalid image index {}",
                pair.image_j
            )
        })?;
        let before = pair.matches.len();
        stats.input_matches += before;
        let mut selected: HashMap<(usize, usize), (usize, usize, f64, Option<f64>)> =
            HashMap::new();
        for &(raw_i, raw_j) in &pair.matches {
            let Some(&canonical_i) = map_i.get(raw_i) else {
                continue;
            };
            let Some(&canonical_j) = map_j.get(raw_j) else {
                continue;
            };
            let distance = features
                .get(pair.image_i)
                .and_then(|set| set.descriptors.get(raw_i))
                .zip(
                    features
                        .get(pair.image_j)
                        .and_then(|set| set.descriptors.get(raw_j)),
                )
                .map_or(f64::INFINITY, |(lhs, rhs)| {
                    descriptor_squared_distance(lhs, rhs)
                });
            let candidate = (
                raw_i,
                raw_j,
                distance,
                match_geometry_residual(features, pair, raw_i, raw_j, camera),
            );
            let key = (canonical_i, canonical_j);
            if selected
                .get(&key)
                .is_none_or(|current| match_candidate_cmp(&candidate, current) == CmpOrdering::Less)
            {
                selected.insert(key, candidate);
            }
        }
        let mut selected_entries: Vec<_> = selected.into_iter().collect();
        selected_entries.sort_by(|lhs, rhs| {
            locus_row_cmp(features, metadata, pair.image_i, lhs.0 .0, rhs.0 .0)
                .then_with(|| locus_row_cmp(features, metadata, pair.image_j, lhs.0 .1, rhs.0 .1))
        });
        pair.matches = selected_entries
            .iter()
            .map(|&((canonical_i, canonical_j), _)| (canonical_i, canonical_j))
            .collect();

        // The essential subset is an independently consumed endpoint list in
        // the mapper.  Apply the same physical representative map and retain
        // one deterministic endpoint pair per locus, without assuming that
        // every imported E row also occurs in the winning `matches` vector.
        if let Some(essential_matches) = pair.essential_matches.as_mut() {
            let mut essential = HashMap::<(usize, usize), (usize, usize)>::new();
            for &(raw_i, raw_j) in essential_matches.iter() {
                let (Some(&canonical_i), Some(&canonical_j)) = (map_i.get(raw_i), map_j.get(raw_j))
                else {
                    continue;
                };
                essential
                    .entry((canonical_i, canonical_j))
                    .or_insert((canonical_i, canonical_j));
            }
            let mut essential_entries: Vec<_> = essential.into_values().collect();
            essential_entries.sort_by(|lhs, rhs| {
                locus_row_cmp(features, metadata, pair.image_i, lhs.0, rhs.0)
                    .then_with(|| locus_row_cmp(features, metadata, pair.image_j, lhs.1, rhs.1))
            });
            *essential_matches = essential_entries;
        }
        let after = pair.matches.len();
        stats.output_matches += after;
        stats.deduplicated_matches += before.saturating_sub(after);
        if before != after {
            stats.changed_pairs += 1;
        }
    }
    Ok(stats)
}

/// Bounded, deterministic descriptor sample for training a retrieval
/// vocabulary — k-means over *every* descriptor (262 k for 128×2048-kpt
/// images) is the pipeline's bottleneck and unnecessary for either VLAD or
/// the vocab-tree: both only need a representative sample. Strides the full
/// descriptor list down to ~`VOCAB_SAMPLE`. Shared by
/// [`candidate_pairs_vlad`] and [`candidate_pairs_vocab_tree`] (M3).
fn sampled_training_descriptors(features: &[FeatureSet]) -> Vec<&[f32]> {
    const VOCAB_SAMPLE: usize = 40_000;
    let all_desc: Vec<&[f32]> = features
        .iter()
        .flat_map(|f| f.descriptors.iter().map(|d| d.as_slice()))
        .collect();
    let stride = (all_desc.len() / VOCAB_SAMPLE).max(1);
    all_desc.iter().step_by(stride).copied().collect()
}

/// All `(i, j)` pairs with `i < j` — the exhaustive fallback shared by both
/// pair sources.
fn all_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }
    pairs
}

/// Candidate image pairs `(i, j)` with `i < j` from flat-VLAD top-K cosine
/// retrieval (or all pairs when `exhaustive`) — the pre-M3 pair source,
/// unchanged.
fn candidate_pairs_vlad_scored(
    features: &[FeatureSet],
    vocab_size: usize,
    topk: usize,
    exhaustive: bool,
    mutual: bool,
) -> Vec<((usize, usize), f32)> {
    let n = features.len();
    if exhaustive || n <= topk + 1 {
        return all_pairs(n).into_iter().map(|pair| (pair, 0.0)).collect();
    }

    let sample = sampled_training_descriptors(features);
    let Some(vocab) = Vocabulary::build(&sample, vocab_size, 10, 0) else {
        // Fall back to exhaustive if the vocabulary cannot be built.
        return all_pairs(n).into_iter().map(|pair| (pair, 0.0)).collect();
    };
    let globals: Vec<Vec<f32>> = features
        .iter()
        .map(|f| vlad(&f.descriptors, &vocab))
        .collect();

    let mut scores = std::collections::BTreeMap::<(usize, usize), f32>::new();
    let mut neighbors = vec![HashSet::<usize>::new(); n];
    for i in 0..n {
        let mut sims: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine_similarity(&globals[i], &globals[j])))
            .collect();
        sims.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        for &(j, score) in sims.iter().take(topk) {
            neighbors[i].insert(j);
            let pair = (i.min(j), i.max(j));
            if !mutual || neighbors[j].contains(&i) {
                scores
                    .entry(pair)
                    .and_modify(|best| *best = best.max(score))
                    .or_insert(score);
            }
        }
    }
    if mutual {
        // The first pass can see only one side of a pair.  Rebuild the score
        // map from the completed neighbour sets so pair admission is exactly
        // symmetric and independent of image traversal order.
        scores.clear();
        for i in 0..n {
            let mut sims: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i && neighbors[i].contains(&j) && neighbors[j].contains(&i))
                .map(|j| (j, cosine_similarity(&globals[i], &globals[j])))
                .collect();
            sims.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for (j, score) in sims {
                let pair = (i.min(j), i.max(j));
                scores
                    .entry(pair)
                    .and_modify(|best| *best = best.max(score))
                    .or_insert(score);
            }
        }
    }
    scores.into_iter().collect()
}

fn candidate_pairs_vlad(
    features: &[FeatureSet],
    vocab_size: usize,
    topk: usize,
    exhaustive: bool,
) -> Vec<(usize, usize)> {
    candidate_pairs_vlad_scored(features, vocab_size, topk, exhaustive, false)
        .into_iter()
        .map(|(pair, _)| pair)
        .collect()
}

fn candidate_pairs_vlad_mutual(
    features: &[FeatureSet],
    vocab_size: usize,
    topk: usize,
    exhaustive: bool,
) -> Vec<(usize, usize)> {
    candidate_pairs_vlad_scored(features, vocab_size, topk, exhaustive, true)
        .into_iter()
        .map(|(pair, _)| pair)
        .collect()
}

/// Candidate pairs from a bounded union of local numeric-stem overlap and
/// VLAD retrieval. Local edges are selected before retrieval edges under a
/// budget, then retrieval-only edges are ranked by pre-match VLAD score.
#[cfg(test)]
fn candidate_pairs_vlad_union(
    features: &[FeatureSet],
    image_names: &[String],
    vocab_size: usize,
    topk: usize,
    local_window: u64,
    budget: Option<usize>,
) -> Result<Vec<(usize, usize)>, String> {
    candidate_pairs_vlad_union_with_grouping(
        features,
        image_names,
        vocab_size,
        topk,
        local_window,
        budget,
        false,
    )
}

fn candidate_pairs_vlad_union_with_grouping(
    features: &[FeatureSet],
    image_names: &[String],
    vocab_size: usize,
    topk: usize,
    local_window: u64,
    budget: Option<usize>,
    rig_local_grouping: bool,
) -> Result<Vec<(usize, usize)>, String> {
    let local = if rig_local_grouping {
        rig_local_pairs(image_names, local_window)?
    } else {
        filter_pairs_by_stem_window(all_pairs(features.len()), image_names, Some(local_window))?
    };
    let retrieval = candidate_pairs_vlad_scored(features, vocab_size, topk, false, false);
    let local_set: HashSet<(usize, usize)> = local.iter().copied().collect();
    let mut ranked: Vec<((usize, usize), bool, f32)> = retrieval
        .into_iter()
        .map(|(pair, score)| (pair, local_set.contains(&pair), score))
        .collect();
    let mut seen: HashSet<(usize, usize)> = ranked.iter().map(|(pair, _, _)| *pair).collect();
    for pair in local {
        if seen.insert(pair) {
            ranked.push((pair, true, f32::NEG_INFINITY));
        }
    }
    ranked.sort_by(|lhs, rhs| {
        rhs.1
            .cmp(&lhs.1)
            .then_with(|| rhs.2.total_cmp(&lhs.2))
            .then_with(|| lhs.0.cmp(&rhs.0))
    });
    if let Some(budget) = budget {
        ranked.truncate(budget);
    }
    ranked.sort_unstable_by_key(|(pair, _, _)| *pair);
    Ok(ranked.into_iter().map(|(pair, _, _)| pair).collect())
}

/// Candidate image pairs `(i, j)` with `i < j` from the M3 hierarchical
/// vocab-tree (`visloc_rs::vision::vocab_tree`, COLMAP's
/// `VocabTreePairGenerator`-equivalent, `docs/colmap_port_plan.md`'s M3
/// milestone), or all pairs when `exhaustive`.
///
/// Trains the hierarchical vocabulary on the same bounded descriptor sample
/// [`candidate_pairs_vlad`] uses, indexes every image's *full* descriptor
/// set (unsampled — retrieval quality for images the tree has never seen
/// depends on it having every one of their features, unlike the shared
/// training sample which only needs to be representative), then queries each
/// image against the finalized tree with its own descriptors, keeping the
/// top `vocab_tree_num_images` other images per query
/// ([`generate_pairs`]/[`VocabTreePairGeneratorOptions`]).
fn candidate_pairs_vocab_tree(
    features: &[FeatureSet],
    branching_factor: usize,
    depth: usize,
    num_images: usize,
    exhaustive: bool,
) -> Vec<(usize, usize)> {
    let n = features.len();
    if exhaustive {
        return all_pairs(n);
    }

    let sample = sampled_training_descriptors(features);
    let hkm_options = HkmBuildOptions {
        branching_factor,
        depth,
        ..HkmBuildOptions::default()
    };
    let vocab_tree_options = VocabTreeOptions::default();
    let Some(mut tree) = VocabTree::build(&sample, &hkm_options, &vocab_tree_options) else {
        // Fall back to exhaustive if the vocabulary cannot be built (mirrors
        // candidate_pairs_vlad's own degenerate-input fallback).
        return all_pairs(n);
    };
    for (i, f) in features.iter().enumerate() {
        tree.add_image(i, &f.descriptors);
    }
    tree.finalize();
    println!(
        "vocab-tree: {} leaf words (requested {}^{}={}), {} images indexed",
        tree.num_words(),
        branching_factor,
        depth,
        branching_factor.pow(depth as u32),
        tree.num_images(),
    );

    let image_descriptors: Vec<Vec<Vec<f32>>> =
        features.iter().map(|f| f.descriptors.clone()).collect();
    generate_pairs(
        &tree,
        &image_descriptors,
        &VocabTreePairGeneratorOptions { num_images },
    )
}

/// How many transitive-expansion rounds
/// ([`PairSource::Transitive`], COLMAP's `TransitivePairGenerator`) run
/// after the vocab-tree base pass. Two rounds cover the common
/// "bridge image chains a-b-c and b-d-e" real-scene topology; each round
/// only proposes pairs not proposed before, so cost is bounded by the
/// verified-graph neighbourhood size.
const TRANSITIVE_ROUNDS: usize = 2;

/// Candidate image pairs `(i, j)` with `i < j` — dispatches on
/// [`PairSource`] (`docs/colmap_port_plan.md`'s M3 A/B switch); `exhaustive`
/// overrides either source, matching pre-M3 behaviour.
/// [`PairSource::Transitive`] returns its *base* pass here (vocab-tree);
/// the transitive expansion happens in [`expand_transitive`] after those
/// base pairs are verified, mirroring COLMAP's generator running against
/// an existing match table.
fn candidate_pairs(
    features: &[FeatureSet],
    image_names: &[String],
    args: &Args,
) -> Result<Vec<(usize, usize)>, String> {
    match args.pair_source {
        PairSource::Vlad => Ok(candidate_pairs_vlad(
            features,
            args.vocab_size,
            args.retrieval_topk,
            args.exhaustive,
        )),
        PairSource::VladMutual => Ok(candidate_pairs_vlad_mutual(
            features,
            args.vocab_size,
            args.retrieval_topk,
            args.exhaustive,
        )),
        PairSource::VladUnion => candidate_pairs_vlad_union_with_grouping(
            features,
            image_names,
            args.vocab_size,
            args.retrieval_topk,
            args.local_stem_window
                .expect("validated vlad-union local window"),
            args.candidate_budget,
            args.rig_local_grouping,
        ),
        PairSource::TemporalPyramid => candidate_pairs_temporal_pyramid(
            features,
            image_names,
            args.vocab_size,
            args.retrieval_topk,
            args.temporal_pyramid_max_offset,
            args.candidate_budget,
        ),
        PairSource::VocabTree | PairSource::Transitive => Ok(candidate_pairs_vocab_tree(
            features,
            args.vocab_tree_branching,
            args.vocab_tree_depth,
            args.vocab_tree_num_images,
            args.exhaustive,
        )),
    }
}

/// Metadata written beside generated candidate pairs.  It is intentionally
/// descriptive rather than used to reconstruct pairs: the image-name-bound
/// pair list remains the authority, while this block makes an archived
/// manifest auditable and lets sharding tools preserve the exact schedule.
fn candidate_manifest_metadata(args: &Args) -> BTreeMap<String, String> {
    let mut metadata = BTreeMap::new();
    let (policy, pair_source) = match args.pair_source {
        PairSource::Vlad => ("vlad-topk-v1", "vlad"),
        PairSource::VladMutual => ("vlad-mutual-v1", "vlad-mutual"),
        PairSource::VladUnion => ("vlad-union-v1", "vlad-union"),
        PairSource::TemporalPyramid => ("temporal-pyramid-v1", "temporal-pyramid"),
        PairSource::VocabTree => ("vocab-tree-v1", "vocab-tree"),
        PairSource::Transitive => ("transitive-v1", "transitive"),
    };
    metadata.insert("candidate_policy".to_owned(), policy.to_owned());
    metadata.insert("pair_source".to_owned(), pair_source.to_owned());
    metadata.insert("retrieval_topk".to_owned(), args.retrieval_topk.to_string());
    if args.pair_source == PairSource::VladUnion {
        metadata.insert(
            "local_grouping".to_owned(),
            if args.rig_local_grouping {
                "rig-prefix-timestamp-v1"
            } else {
                "unique-numeric-stem-v1"
            }
            .to_owned(),
        );
        metadata.insert(
            "cross_camera_rule".to_owned(),
            if args.rig_local_grouping {
                "same-timestamp"
            } else {
                "none"
            }
            .to_owned(),
        );
        metadata.insert(
            "local_stem_window".to_owned(),
            args.local_stem_window
                .expect("validated vlad-union local window")
                .to_string(),
        );
        metadata.insert(
            "candidate_budget".to_owned(),
            args.candidate_budget
                .map_or_else(|| "none".to_owned(), |budget| budget.to_string()),
        );
    }
    if args.pair_source == PairSource::TemporalPyramid {
        metadata.insert(
            "local_grouping".to_owned(),
            "rig-prefix-timestamp-v1".to_owned(),
        );
        metadata.insert("cross_camera_rule".to_owned(), "same-timestamp".to_owned());
        metadata.insert(
            "temporal_offsets".to_owned(),
            temporal_pyramid_offsets_string(args.temporal_pyramid_max_offset),
        );
        metadata.insert(
            "temporal_pyramid_max_offset".to_owned(),
            args.temporal_pyramid_max_offset.to_string(),
        );
        metadata.insert(
            "candidate_budget".to_owned(),
            args.candidate_budget
                .map_or_else(|| "none".to_owned(), |budget| budget.to_string()),
        );
    }
    metadata
}

/// One round of COLMAP's `TransitivePairGenerator` (`src/colmap/pairing.cc`):
/// from the verified-match adjacency, propose every `(i, k)` with `i < k`
/// that shares a common matched partner `j` but has no direct pair yet.
fn expand_transitive(
    pairwise: &[PairwiseMatches],
    already_proposed: &HashSet<(usize, usize)>,
) -> Vec<(usize, usize)> {
    let mut neighbors: HashMap<usize, HashSet<usize>> = HashMap::new();
    for p in pairwise {
        if p.matches.is_empty() {
            continue;
        }
        neighbors.entry(p.image_i).or_default().insert(p.image_j);
        neighbors.entry(p.image_j).or_default().insert(p.image_i);
    }
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut seen = already_proposed.clone();
    for (&i, ni) in &neighbors {
        for &j in ni {
            // Partners of partners.
            let Some(nj) = neighbors.get(&j) else {
                continue;
            };
            for &k in nj {
                if k == i {
                    continue;
                }
                let key = if i < k { (i, k) } else { (k, i) };
                if seen.insert(key) {
                    out.push(key);
                }
            }
        }
    }
    out.sort_unstable();
    out
}

/// Per-`ConfigurationType` pair counts from the COLMAP-style verifier, for
/// the M1 acceptance experiment's pair-rejection report (how many VLAD
/// candidate pairs got reclassified away from a naive essential-matrix
/// accept). Unused (stays all-zero) when `--colmap-verification` is off.
#[derive(Debug, Default, Clone, Copy)]
struct VerificationStats {
    calibrated: usize,
    uncalibrated: usize,
    planar: usize,
    panoramic: usize,
    planar_or_panoramic: usize,
    watermark: usize,
    degenerate: usize,
    multiple: usize,
    /// Pairs whose primary match set was swapped to E inliers.
    force_essential_swaps: usize,
    /// F-winning uncalibrated pairs whose accepted matches were replaced by
    /// the guarded calibrated F→E refinement.
    uncalibrated_f_to_essential_refinements: usize,
    /// F-winning uncalibrated pairs rejected by the opt-in strict strategy
    /// instead of falling back to their F inliers.
    strict_uncalibrated_f_to_essential_exclusions: usize,
    /// Accepted F inliers removed by the strict strategy.
    strict_uncalibrated_f_to_essential_excluded_inliers: usize,
    /// F-winning pairs promoted to a direct calibrated-essential track model.
    calibrated_essential_primary_promotions: usize,
}

impl VerificationStats {
    fn record(&mut self, config: ConfigurationType) {
        match config {
            ConfigurationType::Calibrated => self.calibrated += 1,
            ConfigurationType::Uncalibrated => self.uncalibrated += 1,
            ConfigurationType::Planar => self.planar += 1,
            ConfigurationType::Panoramic => self.panoramic += 1,
            ConfigurationType::PlanarOrPanoramic => self.planar_or_panoramic += 1,
            ConfigurationType::Watermark => self.watermark += 1,
            ConfigurationType::Degenerate => self.degenerate += 1,
            ConfigurationType::Multiple => self.multiple += 1,
            ConfigurationType::Undefined => {}
        }
    }

    fn merge(&mut self, other: &VerificationStats) {
        self.calibrated += other.calibrated;
        self.uncalibrated += other.uncalibrated;
        self.planar += other.planar;
        self.panoramic += other.panoramic;
        self.planar_or_panoramic += other.planar_or_panoramic;
        self.watermark += other.watermark;
        self.degenerate += other.degenerate;
        self.multiple += other.multiple;
        self.force_essential_swaps += other.force_essential_swaps;
        self.uncalibrated_f_to_essential_refinements +=
            other.uncalibrated_f_to_essential_refinements;
        self.strict_uncalibrated_f_to_essential_exclusions +=
            other.strict_uncalibrated_f_to_essential_exclusions;
        self.strict_uncalibrated_f_to_essential_excluded_inliers +=
            other.strict_uncalibrated_f_to_essential_excluded_inliers;
        self.calibrated_essential_primary_promotions +=
            other.calibrated_essential_primary_promotions;
    }

    fn total(&self) -> usize {
        self.calibrated
            + self.uncalibrated
            + self.planar
            + self.panoramic
            + self.planar_or_panoramic
            + self.watermark
            + self.degenerate
            + self.multiple
    }
}

/// The M6 pair-matching backend, dispatched on [`MatcherKind`]. Holds the
/// loaded LightGlue ONNX session (cheap to `Clone`: it wraps an
/// `Arc<Mutex<ort::session::Session>>`, same as
/// [`visloc_rs::vision::features::lightglue_onnx::LightGlueOnnxMatcher`]'s
/// own doc comment explains) so it can be shared across the rayon-parallel
/// per-pair closures in [`verify_pairs`] and [`rescue_bridging`] without
/// re-loading the model.
enum PairMatcher {
    /// Pre-M6 nearest-neighbour + Lowe-ratio matcher.
    Nn,
    /// NN matcher that preserves the matches obtained from each image's
    /// primary feature prefix before appending non-conflicting extra matches.
    NnAppendOnly { primary_keypoint_counts: Vec<usize> },
    /// NN matcher with an alternate descriptor bank for the same keypoint
    /// indices. Alternate matches are append-only and cannot replace the
    /// primary result.
    NnDescriptorEnsemble {
        primary_keypoint_counts: Option<Vec<usize>>,
        alternate_descriptors: Vec<Option<Vec<Vec<f32>>>>,
    },
    /// LightGlue (SuperPoint variant), in-process via ONNX Runtime.
    /// `max_keypoints` truncates each side to a score-sorted prefix (`0` = all).
    #[cfg(feature = "onnx-inference")]
    LightGlue {
        matcher: LightGlueOnnxMatcher,
        max_keypoints: usize,
    },
}

impl PairMatcher {
    /// Raw descriptor matches for one candidate pair `(features_i,
    /// features_j)`. `ratio`/`cross_check` are [`MatcherKind::Nn`]-only
    /// knobs (Lowe ratio test / bidirectional mutual-NN confirmation); they
    /// are silently ignored under [`MatcherKind::LightGlue`], which has no
    /// equivalent parameters of its own — LightGlue's matching decision is
    /// the learned assignment-matrix + `filter_threshold` cut baked into the
    /// exported ONNX graph (see `scripts/export_lightglue_onnx.py`), not a
    /// per-descriptor ratio the caller can tune. This is a deliberate M6
    /// design choice (see the file header and `docs/colmap_port_plan.md`'s
    /// "M6 results"): LightGlue *replaces* the NN+ratio matcher rather than
    /// taking its knobs as a compatibility shim.
    fn match_pair(
        &self,
        ratio: f32,
        cross_check: bool,
        image_i: usize,
        image_j: usize,
        features_i: &FeatureSet,
        features_j: &FeatureSet,
    ) -> Vec<DescriptorMatch> {
        match self {
            PairMatcher::Nn => nn_matches(
                ratio,
                cross_check,
                &features_i.descriptors,
                &features_j.descriptors,
            ),
            PairMatcher::NnAppendOnly {
                primary_keypoint_counts,
            } => {
                let primary_i = primary_keypoint_counts
                    .get(image_i)
                    .copied()
                    .unwrap_or(features_i.descriptors.len());
                let primary_j = primary_keypoint_counts
                    .get(image_j)
                    .copied()
                    .unwrap_or(features_j.descriptors.len());
                append_only_nn_matches(
                    ratio,
                    cross_check,
                    features_i,
                    features_j,
                    primary_i,
                    primary_j,
                )
            }
            PairMatcher::NnDescriptorEnsemble {
                primary_keypoint_counts,
                alternate_descriptors,
            } => {
                let baseline = if let Some(counts) = primary_keypoint_counts {
                    let primary_i = counts
                        .get(image_i)
                        .copied()
                        .unwrap_or(features_i.descriptors.len());
                    let primary_j = counts
                        .get(image_j)
                        .copied()
                        .unwrap_or(features_j.descriptors.len());
                    append_only_nn_matches(
                        ratio,
                        cross_check,
                        features_i,
                        features_j,
                        primary_i,
                        primary_j,
                    )
                } else {
                    nn_matches(
                        ratio,
                        cross_check,
                        &features_i.descriptors,
                        &features_j.descriptors,
                    )
                };
                let Some(alternate_i) = alternate_descriptors
                    .get(image_i)
                    .and_then(|descriptors| descriptors.as_ref())
                else {
                    return baseline;
                };
                let Some(alternate_j) = alternate_descriptors
                    .get(image_j)
                    .and_then(|descriptors| descriptors.as_ref())
                else {
                    return baseline;
                };
                assert_eq!(
                    alternate_i.len(),
                    features_i.descriptors.len(),
                    "alternate descriptor indices must match image-i keypoints"
                );
                assert_eq!(
                    alternate_j.len(),
                    features_j.descriptors.len(),
                    "alternate descriptor indices must match image-j keypoints"
                );
                let alternate = nn_matches(ratio, cross_check, alternate_i, alternate_j);
                append_nonconflicting_matches(baseline, alternate)
            }
            #[cfg(feature = "onnx-inference")]
            PairMatcher::LightGlue {
                matcher,
                max_keypoints,
            } => {
                let (kp_i, desc_i) = truncate_features(features_i, *max_keypoints);
                let (kp_j, desc_j) = truncate_features(features_j, *max_keypoints);
                match matcher.match_features(kp_i, desc_i, kp_j, desc_j) {
                    Ok(matches) => matches
                        .into_iter()
                        .map(|m| DescriptorMatch {
                            query_index: m.query_index,
                            train_index: m.train_index,
                            // LightGlue's assignment matrix has no notion of an
                            // L2 descriptor "distance" the way NN+ratio does —
                            // its own `score` (the assignment-matrix confidence)
                            // is carried in `confidence` instead, which is what
                            // every downstream consumer here actually reads.
                            // `distance = 1.0 - score` keeps this field
                            // orderable (lower = better) for any generic caller
                            // that still sorts on it, without claiming a false
                            // Euclidean-distance semantics.
                            distance: 1.0 - m.score,
                            second_best_distance: None,
                            ratio: None,
                            confidence: Some(m.score),
                        })
                        .collect(),
                    Err(error) => {
                        eprintln!("lightglue match error (treated as zero matches for this pair): {error}");
                        Vec::new()
                    }
                }
            }
        }
    }
}

/// Run the legacy NN+ratio matcher on descriptor slices, preserving the
/// exact cross-check and tie-breaking behavior used when append-only mode is
/// disabled.
fn nn_matches(
    ratio: f32,
    cross_check: bool,
    query: &[Vec<f32>],
    train: &[Vec<f32>],
) -> Vec<DescriptorMatch> {
    if cross_check {
        BruteForceMatcher { ratio: Some(ratio) }.match_descriptors_cross_checked(query, train)
    } else {
        BruteForceMatcher { ratio: Some(ratio) }.match_descriptors(query, train)
    }
}

/// Preserve the primary-prefix NN matches exactly, then append only matches
/// from the full descriptor set that involve at least one extra descriptor
/// and do not reuse a primary query or train endpoint. The full matcher is
/// still used to rank extra candidates, but it can never replace a primary
/// match whose Lowe decision changed after extras were appended.
fn append_only_nn_matches(
    ratio: f32,
    cross_check: bool,
    features_i: &FeatureSet,
    features_j: &FeatureSet,
    primary_i: usize,
    primary_j: usize,
) -> Vec<DescriptorMatch> {
    let primary_i = primary_i.min(features_i.descriptors.len());
    let primary_j = primary_j.min(features_j.descriptors.len());
    if primary_i == features_i.descriptors.len() && primary_j == features_j.descriptors.len() {
        return nn_matches(
            ratio,
            cross_check,
            &features_i.descriptors,
            &features_j.descriptors,
        );
    }
    let primary = nn_matches(
        ratio,
        cross_check,
        &features_i.descriptors[..primary_i],
        &features_j.descriptors[..primary_j],
    );
    let full = nn_matches(
        ratio,
        cross_check,
        &features_i.descriptors,
        &features_j.descriptors,
    );

    let mut used_queries = HashSet::new();
    let mut used_trains = HashSet::new();
    let mut seen_pairs = HashSet::new();
    for m in &primary {
        used_queries.insert(m.query_index);
        used_trains.insert(m.train_index);
        seen_pairs.insert((m.query_index, m.train_index));
    }

    let mut out = primary;
    for m in full {
        // A match entirely inside the primary prefix is represented by the
        // preserved baseline result, even if the full set picked another
        // primary-to-primary neighbour after extras were added.
        if m.query_index < primary_i && m.train_index < primary_j {
            continue;
        }
        if used_queries.contains(&m.query_index) || used_trains.contains(&m.train_index) {
            continue;
        }
        if !seen_pairs.insert((m.query_index, m.train_index)) {
            continue;
        }
        used_queries.insert(m.query_index);
        used_trains.insert(m.train_index);
        out.push(m);
    }
    out
}

/// Append alternate-bank matches without reusing either endpoint claimed by
/// the baseline. Both banks share the original keypoint-index space, so this
/// only augments correspondences and never creates duplicate track nodes.
fn append_nonconflicting_matches(
    baseline: Vec<DescriptorMatch>,
    alternate: Vec<DescriptorMatch>,
) -> Vec<DescriptorMatch> {
    let mut used_queries = HashSet::new();
    let mut used_trains = HashSet::new();
    for m in &baseline {
        used_queries.insert(m.query_index);
        used_trains.insert(m.train_index);
    }
    let mut out = baseline;
    for m in alternate {
        if used_queries.contains(&m.query_index) || used_trains.contains(&m.train_index) {
            continue;
        }
        used_queries.insert(m.query_index);
        used_trains.insert(m.train_index);
        out.push(m);
    }
    out
}

/// Score-sorted prefix for LightGlue (external SuperPoint dumps are already
/// descending by score). `0` or oversized caps keep the full set.
#[cfg(feature = "onnx-inference")]
fn truncate_features(features: &FeatureSet, max_keypoints: usize) -> (&[Point2<f64>], &[Vec<f32>]) {
    let n = features.keypoints.len();
    let take = if max_keypoints == 0 || max_keypoints >= n {
        n
    } else {
        max_keypoints
    };
    (&features.keypoints[..take], &features.descriptors[..take])
}

/// Build the [`PairMatcher`] `--matcher` selects. Fails fast (before any
/// pair is processed) if `--matcher lightglue` is requested without either
/// the `onnx-inference` feature compiled in or a `--lightglue-model` path.
fn build_matcher(
    args: &Args,
    primary_keypoint_counts: &[usize],
    alternate_descriptors: Vec<Option<Vec<Vec<f32>>>>,
) -> Result<PairMatcher, Box<dyn std::error::Error>> {
    if args.sift_append_descriptor_magnification.is_some() {
        if matches!(args.matcher, MatcherKind::LightGlue) {
            return Err(
                "--sift-append-descriptor-magnification requires --matcher nn (LightGlue has no NN descriptor bank)".into(),
            );
        }
        if alternate_descriptors.len() != primary_keypoint_counts.len()
            || alternate_descriptors.iter().any(Option::is_none)
        {
            return Err(
                "--sift-append-descriptor-magnification requires --feature-extractor sift".into(),
            );
        }
        return Ok(PairMatcher::NnDescriptorEnsemble {
            primary_keypoint_counts: args
                .sift_extra_matches_append_only
                .then(|| primary_keypoint_counts.to_vec()),
            alternate_descriptors,
        });
    }
    match args.matcher {
        MatcherKind::Nn if args.sift_extra_matches_append_only => Ok(PairMatcher::NnAppendOnly {
            primary_keypoint_counts: primary_keypoint_counts.to_vec(),
        }),
        MatcherKind::Nn => Ok(PairMatcher::Nn),
        MatcherKind::LightGlue => {
            if args.sift_extra_matches_append_only {
                return Err(
                    "--sift-extra-matches-append-only requires --matcher nn (LightGlue has no NN prefix)".into(),
                );
            }
            #[cfg(feature = "onnx-inference")]
            {
                let path = args
                    .lightglue_model
                    .as_ref()
                    .ok_or("--matcher lightglue requires --lightglue-model PATH")?;
                let backend = match args.onnx_backend.as_str() {
                    "auto" => OnnxBackend::CudaThenCpu,
                    "cuda" => OnnxBackend::Cuda,
                    "cpu" => OnnxBackend::Cpu,
                    other => {
                        return Err(format!(
                            "unknown --onnx-backend {other:?} (expected auto|cuda|cpu)"
                        )
                        .into())
                    }
                };
                eprintln!(
                    "loading LightGlue ONNX from {path:?} (backend={:?})…",
                    backend
                );
                let matcher = LightGlueOnnxMatcher::load_from_path_with_backend(path, backend)
                    .map_err(|error| {
                        format!("failed to load LightGlue ONNX model {path:?}: {error}")
                    })?;
                eprintln!("LightGlue ONNX loaded");
                Ok(PairMatcher::LightGlue {
                    matcher,
                    max_keypoints: args.lightglue_max_keypoints,
                })
            }
            #[cfg(not(feature = "onnx-inference"))]
            {
                Err(
                    "--matcher lightglue requires rebuilding with --features onnx-inference \
                     (see docs/colmap_port_plan.md's M6 results)"
                        .into(),
                )
            }
        }
    }
}

/// Match and geometrically verify each candidate pair into `PairwiseMatches`.
/// Candidate pairs are independent, so the (descriptor-matching dominated) loop
/// is run across cores with rayon.
///
/// `mode` is the M1/M1.1 A/B switch:
/// - [`VerificationMode::Legacy`] (default) reproduces the exact pre-M1
///   essential-matrix-only path byte-for-byte (same estimator, same fixed
///   `5e-3` threshold, same call, same acceptance test) — the "flag off means
///   unchanged behaviour" guarantee `docs/colmap_port_plan.md` asks for.
/// - [`VerificationMode::ThresholdOnly`] runs the *same* single-model
///   essential-matrix-only estimator, but with the per-camera pixel-derived
///   Sampson threshold — isolates the "tighter threshold" half of the M1
///   confound from the "E/F/H classification" half (M1.1).
/// - [`VerificationMode::Full`] goes through [`TwoViewGeometryVerifier`]
///   instead: only `DEGENERATE` and `WATERMARK` pairs are dropped rather than
///   handed to `incremental_sfm` — COLMAP's own admission gate
///   (`database_cache.cc`'s `UseInlierMatchesCheck`) keeps everything else,
///   including `PANORAMIC` (pure rotation — no baseline to triangulate from)
///   and unresolved `PLANAR_OR_PANORAMIC` (M2.1 parity fix; see
///   `docs/colmap_port_plan.md`'s "M2.1 results" — previously this demo
///   dropped both, stricter than real COLMAP). `CALIBRATED` / `UNCALIBRATED`
///   / `PLANAR` / `PANORAMIC` / `PLANAR_OR_PANORAMIC` / `MULTIPLE` pairs all
///   keep their winning model's own inliers (which need not be the essential
///   matrix's); a `PANORAMIC`/`PLANAR_OR_PANORAMIC` pair's correspondences
///   can still help track connectivity and BA even though the pair itself
///   can never become a seed (`incremental_sfm`'s parallax gate at growth
///   time excludes near-zero-baseline pairs independently of this
///   classification, mirroring how COLMAP's own init-pair search
///   recomputes and gates on triangulation angle rather than consulting the
///   stored `ConfigurationType`).
///
/// Re-match pairs incident to `stems` at a looser Lowe ratio; keep the new
/// geometry when it exposes more essential inliers (hub densification without
/// changing the global keypoint set).
#[allow(clippy::too_many_arguments)]
fn rematch_stem_pairs(
    features: &[FeatureSet],
    image_names: &[String],
    pairwise: &mut [PairwiseMatches],
    camera: &Camera,
    stems: &[String],
    rematch_ratio: f32,
    rematch_cross_check: bool,
    min_matches: usize,
    mode: VerificationMode,
    matcher: &PairMatcher,
    guided_matching: bool,
    multiple_models: bool,
    min_e_f_inlier_ratio: Option<f64>,
    calibrated_prefer_essential: bool,
    force_essential_min_ef_ratio: f64,
    force_essential_min_e_inliers: usize,
    guided_max_error_px: Option<f64>,
    guided_lowe_ratio: Option<f64>,
) -> usize {
    let want: HashSet<&str> = stems.iter().map(String::as_str).collect();
    let stem_of = |idx: usize| -> &str {
        Path::new(&image_names[idx])
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(image_names[idx].as_str())
    };
    let targets: Vec<(usize, usize)> = pairwise
        .iter()
        .filter(|p| want.contains(stem_of(p.image_i)) || want.contains(stem_of(p.image_j)))
        .map(|p| (p.image_i, p.image_j))
        .collect();
    if targets.is_empty() || mode != VerificationMode::Full {
        return 0;
    }
    let (fresh, _, _) = verify_pairs(
        features,
        camera,
        &targets,
        rematch_ratio,
        min_matches,
        mode,
        matcher,
        rematch_cross_check,
        guided_matching,
        multiple_models,
        min_e_f_inlier_ratio,
        calibrated_prefer_essential,
        false, // F→E refinement is only enabled on the main configured pass
        false, // strict F→E exclusion is only enabled on the main configured pass
        false, // calibrated-essential promotion is only enabled on the main pass
        false, // do not force-swap primary matches on rematch
        force_essential_min_ef_ratio,
        force_essential_min_e_inliers,
        false,
        guided_max_error_px,
        guided_lowe_ratio,
        None,
        None,
        false,
    );
    let mut improved = 0usize;
    for new in fresh {
        let key = (new.image_i.min(new.image_j), new.image_i.max(new.image_j));
        if let Some(old) = pairwise
            .iter_mut()
            .find(|p| (p.image_i.min(p.image_j), p.image_i.max(p.image_j)) == key)
        {
            let old_e = old.essential_matches.as_ref().map(|e| e.len()).unwrap_or(0);
            let new_e = new.essential_matches.as_ref().map(|e| e.len()).unwrap_or(0);
            if new_e > old_e || (old_e == 0 && new_e >= min_matches) {
                let name = |idx: usize| {
                    Path::new(&image_names[idx])
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or(image_names[idx].as_str())
                };
                eprintln!(
                    "rematch: improved {}-{} E {} -> {} (inliers {} -> {})",
                    name(new.image_i),
                    name(new.image_j),
                    old_e,
                    new_e,
                    old.matches.len(),
                    new.matches.len()
                );
                *old = new;
                improved += 1;
            }
        }
    }
    improved
}

/// Single-pair pose-guided verify (COLMAP FindGuidedMatches under known E).
fn verify_pose_guided_pair(
    features: &[FeatureSet],
    camera: &Camera,
    i: usize,
    j: usize,
    pose_i: &Pose,
    pose_j: &Pose,
    rematch_ratio: f32,
    rematch_cross_check: bool,
    min_matches: usize,
    matcher: &PairMatcher,
    guided_max_error_px: f64,
    guided_lowe_ratio: f64,
    calibrated_prefer_essential: bool,
) -> Option<PairwiseMatches> {
    let pose_e = essential_from_absolute_poses(pose_i, pose_j)?;
    let dm0 = matcher.match_pair(
        rematch_ratio,
        rematch_cross_check,
        i,
        j,
        &features[i],
        &features[j],
    );
    let extra = guided_epipolar_matches(
        camera,
        &features[i],
        &features[j],
        &dm0,
        &[],
        guided_max_error_px,
        Some(pose_e),
        guided_lowe_ratio,
    );
    let mut dm = dm0;
    dm.extend(extra);
    if dm.len() < min_matches {
        return None;
    }
    let corrs: Vec<TwoViewCorrespondence> = dm
        .iter()
        .map(|m| {
            TwoViewCorrespondence::new(
                features[i].keypoints[m.query_index],
                features[j].keypoints[m.train_index],
            )
        })
        .collect();
    let mut opts = TwoViewGeometryOptions::for_camera(camera, 4.0);
    opts.calibrated_prefer_essential = calibrated_prefer_essential;
    let verifier = TwoViewGeometryVerifier::new(opts);
    let report = verifier.classify(&corrs, camera);
    let keep = matches!(
        report.config,
        ConfigurationType::Calibrated
            | ConfigurationType::Uncalibrated
            | ConfigurationType::Planar
            | ConfigurationType::Panoramic
            | ConfigurationType::PlanarOrPanoramic
            | ConfigurationType::Multiple
    );
    if !keep || report.inliers.len() < min_matches {
        return None;
    }
    let essential_matches = if report.essential_inliers.len() >= min_matches {
        Some(
            report
                .essential_inliers
                .iter()
                .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let matches: Vec<(usize, usize)> = report
        .inliers
        .iter()
        .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
        .collect();
    Some(PairwiseMatches {
        image_i: i,
        image_j: j,
        matches,
        two_view_config: Some(report.config),
        essential_matches,
        essential_matrix: report.essential,
    })
}

/// Post-incremental: rematch free hub stems only against cameras that have
/// pose priors — so densification targets prior↔hub bridges (the courtyard
/// unlock), not free–free pairs like `0297–0298`.
#[allow(clippy::too_many_arguments)]
fn rematch_free_against_priors(
    features: &[FeatureSet],
    image_names: &[String],
    pairwise: &mut Vec<PairwiseMatches>,
    camera: &Camera,
    pose_priors: &[Option<Pose>],
    free_stems: &[String],
    rematch_ratio: f32,
    rematch_cross_check: bool,
    min_matches: usize,
    mode: VerificationMode,
    matcher: &PairMatcher,
    guided_matching: bool,
    multiple_models: bool,
    min_e_f_inlier_ratio: Option<f64>,
    calibrated_prefer_essential: bool,
    _force_essential_min_ef_ratio: f64,
    _force_essential_min_e_inliers: usize,
    tracks_use_essential: bool,
    min_chirality_margin: f64,
    require_prior_anchor: bool,
    anchor_min_e_inliers: usize,
    gt_by_stem: Option<&HashMap<String, Pose>>,
    max_gt_bearing_deg: f64,
    guided_max_error_px: Option<f64>,
    guided_lowe_ratio: Option<f64>,
    require_calibrated: bool,
    max_mean_sampson: f64,
    prior_ray_guided: bool,
    prior_ray_min_rays: usize,
    prior_ray_min_e_inliers: usize,
    rematch_verification_mode: Option<VerificationMode>,
    pair_stem_window: Option<u64>,
) -> (usize, Vec<((usize, usize), usize)>) {
    let rematch_mode = rematch_verification_mode.unwrap_or(mode);
    match rematch_mode {
        VerificationMode::Full | VerificationMode::ThresholdOnly => {}
        VerificationMode::Legacy => return (0, Vec::new()),
    }
    let prior_idx: HashSet<usize> = pose_priors
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|_| i))
        .collect();
    if prior_idx.is_empty() {
        return (0, Vec::new());
    }
    let stem_of = |idx: usize| -> &str {
        Path::new(&image_names[idx])
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(image_names[idx].as_str())
    };
    let free_want: HashSet<&str> = free_stems.iter().map(String::as_str).collect();
    let free_idx: HashSet<usize> = (0..features.len())
        .filter(|&i| {
            if prior_idx.contains(&i) {
                return false;
            }
            if free_want.is_empty() {
                true
            } else {
                free_want.contains(stem_of(i))
            }
        })
        .collect();
    if free_idx.is_empty() {
        return (0, Vec::new());
    }
    // All free×prior pairs (exhaustive within the bipartite cut) — includes
    // pairs the main pass never admitted, which is the point for bridges.
    let mut targets: Vec<(usize, usize)> = Vec::new();
    for &f in &free_idx {
        for &p in &prior_idx {
            targets.push((f.min(p), f.max(p)));
        }
    }
    targets.sort_unstable();
    targets.dedup();
    if let Some(window) = pair_stem_window {
        let stem_values = numeric_stem_values(image_names)
            .expect("pair stem window was validated before rematch-free-vs-priors");
        targets.retain(|&pair| {
            pair_within_stem_window(pair, &stem_values, window)
                .expect("rematch pair indices are loaded image indices")
        });
    }
    eprintln!(
        "rematch-free-vs-priors: {} free × {} prior → {} candidate pairs (ratio={:.2}, guided={})",
        free_idx.len(),
        prior_idx.len(),
        targets.len(),
        rematch_ratio,
        guided_matching
    );
    let guided_max_px = guided_max_error_px.unwrap_or(2.0);
    let guided_lowe = guided_lowe_ratio.unwrap_or(0.8);
    let fresh = if prior_ray_guided {
        let free_poses = estimate_free_poses_from_prior_rays(
            pairwise,
            features,
            camera,
            pose_priors,
            prior_ray_min_rays,
            prior_ray_min_e_inliers,
        );
        eprintln!(
            "rematch prior-ray-guided: {} free pose(s) from incremental rays (min_rays={}, min_e={})",
            free_poses.len(),
            prior_ray_min_rays,
            prior_ray_min_e_inliers
        );
        let pose_at = |idx: usize| -> Option<Pose> {
            pose_priors
                .get(idx)
                .and_then(|p| p.as_ref())
                .cloned()
                .or_else(|| free_poses.get(&idx).cloned())
        };
        let mut out = Vec::new();
        let mut std_targets: Vec<(usize, usize)> = Vec::new();
        let mut guided_attempts = 0usize;
        for &(i, j) in &targets {
            let (Some(pi), Some(pj)) = (pose_at(i), pose_at(j)) else {
                std_targets.push((i, j));
                continue;
            };
            guided_attempts += 1;
            if let Some(pm) = verify_pose_guided_pair(
                features,
                camera,
                i,
                j,
                &pi,
                &pj,
                rematch_ratio,
                rematch_cross_check,
                min_matches,
                matcher,
                guided_max_px,
                guided_lowe,
                calibrated_prefer_essential,
            ) {
                out.push(pm);
            } else {
                std_targets.push((i, j));
            }
        }
        eprintln!(
            "rematch prior-ray-guided: pose-guided {}/{} pair(s) ok; {} fallback to standard verify",
            out.len(),
            guided_attempts,
            std_targets.len()
        );
        if !std_targets.is_empty() {
            let (more, _, _) = verify_pairs(
                features,
                camera,
                &std_targets,
                rematch_ratio,
                min_matches,
                rematch_mode,
                matcher,
                rematch_cross_check,
                guided_matching,
                multiple_models,
                min_e_f_inlier_ratio,
                calibrated_prefer_essential,
                false,
                false,
                false,
                false,
                0.0,
                0,
                false,
                guided_max_error_px,
                guided_lowe_ratio,
                None,
                None,
                false,
            );
            out.extend(more);
        }
        out
    } else {
        verify_pairs(
            features,
            camera,
            &targets,
            rematch_ratio,
            min_matches,
            rematch_mode,
            matcher,
            rematch_cross_check,
            guided_matching,
            multiple_models,
            min_e_f_inlier_ratio,
            calibrated_prefer_essential,
            false,
            false,
            false,
            false,
            0.0,
            0,
            false,
            guided_max_error_px,
            guided_lowe_ratio,
            None,
            None,
            false,
        )
        .0
    };
    let mut changed = 0usize;
    let mut gained_e_pairs: Vec<((usize, usize), usize)> = Vec::new();
    let mut rejected = 0usize;
    let mut existing: HashMap<(usize, usize), usize> = HashMap::new();
    for (idx, p) in pairwise.iter().enumerate() {
        existing.insert((p.image_i.min(p.image_j), p.image_i.max(p.image_j)), idx);
    }
    for new in fresh {
        let key = (new.image_i.min(new.image_j), new.image_i.max(new.image_j));
        let new_e = new.essential_matches.as_ref().map(|e| e.len()).unwrap_or(0);
        let name = |idx: usize| stem_of(idx);
        let mut new = new;
        if tracks_use_essential {
            if let Some(ess) = new.essential_matches.clone() {
                if ess.len() >= min_matches {
                    new.matches = ess;
                }
            }
        }
        let (prior_cam, free_cam) = if prior_idx.contains(&new.image_i) {
            (new.image_i, new.image_j)
        } else {
            (new.image_j, new.image_i)
        };
        if !rematch_essential_admission_ok(
            &new,
            prior_cam,
            free_cam,
            features,
            camera,
            pose_priors,
            pairwise,
            min_chirality_margin,
            require_prior_anchor,
            anchor_min_e_inliers,
        ) {
            rejected += 1;
            continue;
        }
        if require_calibrated
            && !matches!(
                new.two_view_config,
                Some(visloc_vision::two_view::ConfigurationType::Calibrated)
            )
        {
            eprintln!(
                "rematch-free-vs-priors: reject {}-{} config={:?} (require Calibrated)",
                name(prior_cam),
                name(free_cam),
                new.two_view_config
            );
            rejected += 1;
            continue;
        }
        if max_mean_sampson > 0.0 {
            if let Some(ms) = pair_essential_mean_sampson_error(&new, features, camera) {
                if ms > max_mean_sampson {
                    eprintln!(
                        "rematch-free-vs-priors: Sampson reject {}-{} mean={:.5} > {:.5}",
                        name(prior_cam),
                        name(free_cam),
                        ms,
                        max_mean_sampson
                    );
                    rejected += 1;
                    continue;
                }
            }
        }
        if max_gt_bearing_deg > 0.0 {
            if let Some(gt) = gt_by_stem {
                let stem_of = |idx: usize| -> Option<&str> {
                    Path::new(&image_names[idx])
                        .file_stem()
                        .and_then(|s| s.to_str())
                };
                if let (Some(ps), Some(fs)) = (stem_of(prior_cam), stem_of(free_cam)) {
                    if let (Some(gt_p), Some(gt_f)) = (gt.get(ps), gt.get(fs)) {
                        if let Some(err) = prior_free_essential_gt_bearing_error_deg(
                            &new, prior_cam, free_cam, features, camera, gt_p, gt_f,
                        ) {
                            if err > max_gt_bearing_deg {
                                eprintln!(
                                    "rematch-free-vs-priors: GT-bearing reject {}-{} err={:.1}° > {:.1}°",
                                    ps, fs, err, max_gt_bearing_deg
                                );
                                rejected += 1;
                                continue;
                            }
                        }
                    }
                }
            }
        }
        if let Some(&idx) = existing.get(&key) {
            let old = &pairwise[idx];
            let old_e = old.essential_matches.as_ref().map(|e| e.len()).unwrap_or(0);
            // Prior↔hub unlock needs *essential* support; F-only densification
            // (E=0) poisons tracks without a calibrated bridge.
            if new_e > old_e {
                eprintln!(
                    "rematch-free-vs-priors: improved {}-{} E {} -> {} config={:?} (inliers {} -> {}, tracks_e={})",
                    name(new.image_i),
                    name(new.image_j),
                    old_e,
                    new_e,
                    new.two_view_config,
                    old.matches.len(),
                    new.matches.len(),
                    tracks_use_essential
                );
                pairwise[idx] = new;
                gained_e_pairs.push((key, new_e));
                changed += 1;
            }
        } else if new_e >= min_matches {
            eprintln!(
                "rematch-free-vs-priors: new bridge {}-{} E={} config={:?} inliers={} tracks_e={}",
                name(new.image_i),
                name(new.image_j),
                new_e,
                new.two_view_config,
                new.matches.len(),
                tracks_use_essential
            );
            existing.insert(key, pairwise.len());
            pairwise.push(new);
            gained_e_pairs.push((key, new_e));
            changed += 1;
        }
    }
    if rejected > 0 {
        eprintln!(
            "rematch-free-vs-priors: rejected {rejected} E-gain(s) (margin>={min_chirality_margin:.2}, prior_anchor={require_prior_anchor})"
        );
    }
    (changed, gained_e_pairs)
}

/// Post-global free↔prior rematch seeded by absolute-pose essentials.
/// For each free×prior pair where both cameras registered, expand NN matches
/// under the pose-derived E Sampson gate, re-verify, and keep pairs whose
/// essential inlier count rises (same accept gate as pre-global rematch).
fn rematch_pose_guided_free_vs_priors(
    features: &[FeatureSet],
    image_names: &[String],
    pairwise: &mut Vec<PairwiseMatches>,
    camera: &Camera,
    poses: &[Option<Pose>],
    pose_priors: &[Option<Pose>],
    // When set, replace guidance poses by stem (GT oracle); reconstruction
    // poses stay untouched.
    guidance_poses_by_stem: Option<&HashMap<String, Pose>>,
    free_stems: &[String],
    rematch_ratio: f32,
    rematch_cross_check: bool,
    min_matches: usize,
    matcher: &PairMatcher,
    tracks_use_essential: bool,
    pair_stem_window: Option<u64>,
) -> (usize, Vec<((usize, usize), usize)>) {
    let prior_idx: HashSet<usize> = pose_priors
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|_| i))
        .collect();
    if prior_idx.is_empty() {
        return (0, Vec::new());
    }
    let stem_of = |idx: usize| -> &str {
        Path::new(&image_names[idx])
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(image_names[idx].as_str())
    };
    let guide_pose = |idx: usize| -> Option<Pose> {
        if let Some(by_stem) = guidance_poses_by_stem {
            if let Some(p) = by_stem.get(stem_of(idx)) {
                return Some(p.clone());
            }
        }
        poses.get(idx).and_then(|p| p.clone())
    };
    let free_want: HashSet<&str> = free_stems.iter().map(String::as_str).collect();
    let free_idx: HashSet<usize> = (0..features.len())
        .filter(|&i| {
            if prior_idx.contains(&i) {
                return false;
            }
            if guide_pose(i).is_none() {
                return false;
            }
            if free_want.is_empty() {
                true
            } else {
                free_want.contains(stem_of(i))
            }
        })
        .collect();
    if free_idx.is_empty() {
        return (0, Vec::new());
    }
    let mut targets: Vec<(usize, usize)> = Vec::new();
    for &f in &free_idx {
        for &p in &prior_idx {
            if guide_pose(p).is_none() {
                continue;
            }
            targets.push((f.min(p), f.max(p)));
        }
    }
    targets.sort_unstable();
    targets.dedup();
    if let Some(window) = pair_stem_window {
        let stem_values = numeric_stem_values(image_names)
            .expect("pair stem window was validated before pose-guided rematch");
        targets.retain(|&pair| {
            pair_within_stem_window(pair, &stem_values, window)
                .expect("rematch pair indices are loaded image indices")
        });
    }
    eprintln!(
        "rematch-pose-guided: {} free × {} prior → {} candidate pairs (ratio={:.2}, gt_guide={})",
        free_idx.len(),
        prior_idx.len(),
        targets.len(),
        rematch_ratio,
        guidance_poses_by_stem.is_some()
    );

    let mut opts = TwoViewGeometryOptions::for_camera(camera, 4.0);
    opts.calibrated_prefer_essential = true;
    let verifier = TwoViewGeometryVerifier::new(opts);

    let results: Vec<Option<PairwiseMatches>> = targets
        .par_iter()
        .map(|&(i, j)| {
            let (Some(pi), Some(pj)) = (guide_pose(i), guide_pose(j)) else {
                return None;
            };
            let pose_e = essential_from_absolute_poses(&pi, &pj)?;
            let dm0 = matcher.match_pair(
                rematch_ratio,
                rematch_cross_check,
                i,
                j,
                &features[i],
                &features[j],
            );
            let extra = guided_epipolar_matches(
                camera,
                &features[i],
                &features[j],
                &dm0,
                &[],
                2.0,
                Some(pose_e),
                0.8,
            );
            let mut dm = dm0;
            dm.extend(extra);
            if dm.len() < min_matches {
                return None;
            }
            let corrs: Vec<TwoViewCorrespondence> = dm
                .iter()
                .map(|m| {
                    TwoViewCorrespondence::new(
                        features[i].keypoints[m.query_index],
                        features[j].keypoints[m.train_index],
                    )
                })
                .collect();
            let report = verifier.classify(&corrs, camera);
            let keep = matches!(
                report.config,
                ConfigurationType::Calibrated
                    | ConfigurationType::Uncalibrated
                    | ConfigurationType::Planar
                    | ConfigurationType::Panoramic
                    | ConfigurationType::PlanarOrPanoramic
                    | ConfigurationType::Multiple
            );
            if !keep || report.inliers.len() < min_matches {
                return None;
            }
            let essential_matches = if report.essential_inliers.len() >= min_matches {
                Some(
                    report
                        .essential_inliers
                        .iter()
                        .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let matches: Vec<(usize, usize)> = report
                .inliers
                .iter()
                .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                .collect();
            Some(PairwiseMatches {
                image_i: i,
                image_j: j,
                matches,
                two_view_config: Some(report.config),
                essential_matches,
                essential_matrix: report.essential,
            })
        })
        .collect();

    let mut changed = 0usize;
    let mut gained_e_pairs: Vec<((usize, usize), usize)> = Vec::new();
    let mut existing: HashMap<(usize, usize), usize> = HashMap::new();
    for (idx, p) in pairwise.iter().enumerate() {
        existing.insert((p.image_i.min(p.image_j), p.image_i.max(p.image_j)), idx);
    }
    for new in results.into_iter().flatten() {
        let key = (new.image_i.min(new.image_j), new.image_i.max(new.image_j));
        let new_e = new.essential_matches.as_ref().map(|e| e.len()).unwrap_or(0);
        let mut new = new;
        if tracks_use_essential {
            if let Some(ess) = new.essential_matches.clone() {
                if ess.len() >= min_matches {
                    new.matches = ess;
                }
            }
        }
        if let Some(&idx) = existing.get(&key) {
            let old = &pairwise[idx];
            let old_e = old.essential_matches.as_ref().map(|e| e.len()).unwrap_or(0);
            if new_e > old_e {
                eprintln!(
                    "rematch-pose-guided: improved {}-{} E {} -> {} (inliers {} -> {})",
                    stem_of(new.image_i),
                    stem_of(new.image_j),
                    old_e,
                    new_e,
                    old.matches.len(),
                    new.matches.len()
                );
                pairwise[idx] = new;
                gained_e_pairs.push((key, new_e));
                changed += 1;
            }
        } else if new_e >= min_matches {
            eprintln!(
                "rematch-pose-guided: new bridge {}-{} E={} inliers={}",
                stem_of(new.image_i),
                stem_of(new.image_j),
                new_e,
                new.matches.len()
            );
            existing.insert(key, pairwise.len());
            pairwise.push(new);
            gained_e_pairs.push((key, new_e));
            changed += 1;
        }
    }
    (changed, gained_e_pairs)
}

fn verify_pairs(
    features: &[FeatureSet],
    camera: &Camera,
    candidates: &[(usize, usize)],
    match_ratio: f32,
    min_matches: usize,
    mode: VerificationMode,
    matcher: &PairMatcher,
    cross_check: bool,
    guided_matching: bool,
    multiple_models: bool,
    min_e_f_inlier_ratio: Option<f64>,
    calibrated_prefer_essential: bool,
    refine_uncalibrated_f_to_essential: bool,
    strict_uncalibrated_f_to_essential: bool,
    calibrated_essential_primary: bool,
    force_essential_matches: bool,
    force_essential_min_ef_ratio: f64,
    force_essential_min_e_inliers: usize,
    force_essential_uncalibrated_only: bool,
    guided_max_error_px: Option<f64>,
    guided_lowe_ratio: Option<f64>,
    imported_matches: Option<&HashMap<(usize, usize), Vec<(usize, usize)>>>,
    imported_matches_supplement: Option<&HashMap<(usize, usize), Vec<(usize, usize)>>>,
    colmap_guided_matching: bool,
) -> (
    Vec<PairwiseMatches>,
    VerificationStats,
    HashMap<(usize, usize), SnapshotPairMetadata>,
) {
    let verifier = (mode == VerificationMode::Full).then(|| {
        let mut opts = TwoViewGeometryOptions::for_camera(camera, 4.0);
        opts.multiple_models = multiple_models;
        if let Some(r) = min_e_f_inlier_ratio {
            opts.min_e_f_inlier_ratio = r;
        }
        opts.calibrated_prefer_essential = calibrated_prefer_essential;
        TwoViewGeometryVerifier::new(opts)
    });
    // Same single-model essential-only estimator as the legacy path, just
    // with `for_camera`'s per-camera pixel-derived Sampson threshold swapped
    // in for the fixed `5e-3` default — everything else (iterations, seed,
    // translation scale) stays at `EssentialRansacConfig`/`RelativePoseEstimator`
    // defaults, matching the legacy path field-for-field.
    let threshold_only_estimator = (mode == VerificationMode::ThresholdOnly).then(|| {
        let sampson_threshold =
            TwoViewGeometryOptions::for_camera(camera, 4.0).essential_sampson_threshold;
        RelativePoseEstimator {
            ransac: EssentialRansac {
                estimator: EightPointEssentialMatrixEstimator::default(),
                config: EssentialRansacConfig {
                    sampson_threshold,
                    ..EssentialRansacConfig::default()
                },
            },
            default_translation_scale: 1.0,
            ..RelativePoseEstimator::default()
        }
    });

    #[cfg(feature = "onnx-inference")]
    let sequential = matches!(matcher, PairMatcher::LightGlue { .. });
    #[cfg(not(feature = "onnx-inference"))]
    let sequential = false;
    let force_e_swaps = AtomicUsize::new(0);
    let f_to_e_refinements = AtomicUsize::new(0);
    let strict_f_to_e_exclusions = AtomicUsize::new(0);
    let strict_f_to_e_excluded_inliers = AtomicUsize::new(0);
    let calibrated_essential_primary_promotions = AtomicUsize::new(0);
    let dump_match_stats = std::env::var_os("VISLOC_SFM_DEBUG_DUMP_MATCH_STATS").is_some();
    let dump_pair_outcomes = std::env::var_os("VISLOC_SFM_DEBUG_DUMP_PAIR_OUTCOMES").is_some();
    let dump_match_indices = std::env::var_os("VISLOC_SFM_DEBUG_DUMP_MATCH_INDICES").is_some();
    let dump_guided_matches = std::env::var_os("VISLOC_SFM_DEBUG_DUMP_GUIDED_MATCHES").is_some();
    // Full-graph, GT-independent E quality probe.  This is intentionally
    // environment-gated: it adds bounded triangulation work and must never
    // alter the ordinary verification or mapper path.  When a refined F is
    // available, the same row also contains a diagnostics-only `Kᵀ F K` → E
    // comparison (`f2e_*` fields); it is never fed back into verification.
    let dump_essential_quality =
        std::env::var_os("VISLOC_SFM_DEBUG_DUMP_ESSENTIAL_QUALITY").is_some();
    let dump_f2e_diagnostics = std::env::var_os("VISLOC_SFM_DEBUG_DUMP_F2E_DIAGNOSTICS").is_some();
    let verify_one = |&(i, j): &(usize, usize)| {
        let dm: Vec<DescriptorMatch> = if let Some(imp) = imported_matches {
            let key = (i.min(j), i.max(j));
            let Some(raw) = imp.get(&key) else {
                return (None, None, None);
            };
            let flip = i > j;
            raw.iter()
                .map(|&(a, b)| {
                    let (qi, tj) = if flip { (b, a) } else { (a, b) };
                    DescriptorMatch {
                        query_index: qi,
                        train_index: tj,
                        distance: 0.0,
                        second_best_distance: None,
                        ratio: None,
                        confidence: None,
                    }
                })
                .collect()
        } else if let Some(supp) = imported_matches_supplement {
            let key = (i.min(j), i.max(j));
            if let Some(raw) = supp.get(&key) {
                let flip = i > j;
                raw.iter()
                    .map(|&(a, b)| {
                        let (qi, tj) = if flip { (b, a) } else { (a, b) };
                        DescriptorMatch {
                            query_index: qi,
                            train_index: tj,
                            distance: 0.0,
                            second_best_distance: None,
                            ratio: None,
                            confidence: None,
                        }
                    })
                    .collect()
            } else {
                matcher.match_pair(match_ratio, cross_check, i, j, &features[i], &features[j])
            }
        } else {
            matcher.match_pair(match_ratio, cross_check, i, j, &features[i], &features[j])
        };
        if dump_match_stats {
            eprintln!("sfm-debug-raw: {} {} matches={}", i, j, dm.len());
        }
        if dm.len() < min_matches {
            if dump_pair_outcomes {
                eprintln!(
                        "sfm-debug-outcome: {i} {j} raw={} config=TOO_FEW accepted=0 e=0 f=0 h=0 reason=too_few_raw",
                        dm.len(),
                    );
            }
            return (None, None, None);
        }
        let corrs: Vec<TwoViewCorrespondence> = dm
            .iter()
            .map(|m| {
                TwoViewCorrespondence::new(
                    features[i].keypoints[m.query_index],
                    features[j].keypoints[m.train_index],
                )
            })
            .collect();

        if let Some(verifier) = &verifier {
            let report = verifier.classify(&corrs, camera);
            // M2.1: mirror COLMAP's real gate (`database_cache.cc`'s
            // `UseInlierMatchesCheck`), which is `num_matches >=
            // min_num_matches && (!ignore_watermarks || config !=
            // WATERMARK)` — i.e. every non-`DEGENERATE`, non-`WATERMARK`
            // configuration contributes its inlier matches, including
            // `PLANAR_OR_PANORAMIC`/`PANORAMIC` (homography-only, no
            // triangulatable baseline). `DEGENERATE` needs no explicit
            // arm here because [`TwoViewGeometryVerifier`] already
            // returns an empty inlier list for it (`degenerate_report()`
            // in `colmap_verification.rs`), the same reason COLMAP's own
            // degenerate branch never populates `inlier_matches`.
            let keep = matches!(
                report.config,
                ConfigurationType::Calibrated
                    | ConfigurationType::Uncalibrated
                    | ConfigurationType::Planar
                    | ConfigurationType::Panoramic
                    | ConfigurationType::PlanarOrPanoramic
                    | ConfigurationType::Multiple
            );
            if !keep || report.inliers.len() < min_matches {
                if dump_pair_outcomes {
                    let reason = if !keep {
                        "configuration_rejected"
                    } else {
                        "inliers_below_min"
                    };
                    eprintln!(
                            "sfm-debug-outcome: {i} {j} raw={} config={} accepted={} e={} f={} h={} reason={reason}",
                            dm.len(),
                            configuration_name(report.config),
                            report.inliers.len(),
                            report.e_inlier_count,
                            report.f_inlier_count,
                            report.h_inlier_count,
                        );
                }
                if dump_essential_quality {
                    let quality = essential_pair_quality(&report, &corrs, camera);
                    let f2e_quality = fundamental_to_essential_quality(&report, &corrs, camera);
                    eprintln!(
                            "sfm-debug-essential-quality: {i} {j} raw={} config={} accepted={} e={} f={} h={}{}{}",
                            dm.len(),
                            configuration_name(report.config),
                            report.inliers.len(),
                            report.e_inlier_count,
                            report.f_inlier_count,
                            report.h_inlier_count,
                            format_essential_pair_quality(quality),
                            format_fundamental_to_essential_quality(f2e_quality),
                        );
                }
                return (None, Some(report.config), None);
            }
            // Guided matching (COLMAP FindGuidedMatches): expand the
            // match set under the verified epipolar geometry, then
            // re-verify so config/inliers describe the final set.
            let (dm, report, report_corrs) = if guided_matching {
                let original_report = report.clone();
                // Prefer E inliers as the epipolar seed when available —
                // F-seeded guided matching densifies Uncalibrated façades
                // without raising calibrated bridges (courtyard prior↔hub).
                let seed_idx: &[usize] = if report.essential_inliers.len() >= 8 {
                    &report.essential_inliers
                } else {
                    &report.inliers
                };
                let inlier_corrs: Vec<TwoViewCorrespondence> = seed_idx
                    .iter()
                    .filter_map(|&idx| corrs.get(idx).copied())
                    .collect();
                let guided_max_error = guided_max_error_px.unwrap_or(2.0);
                let guided_ratio = guided_lowe_ratio.unwrap_or(0.8);
                let extra = if colmap_guided_matching {
                    colmap_guided_matches(
                        camera,
                        &features[i],
                        &features[j],
                        &dm,
                        &report,
                        guided_max_error,
                        guided_ratio,
                        cross_check,
                    )
                } else {
                    guided_epipolar_matches(
                        camera,
                        &features[i],
                        &features[j],
                        &dm,
                        &inlier_corrs,
                        guided_max_error,
                        None,
                        guided_ratio,
                    )
                };
                if dump_guided_matches {
                    eprintln!(
                        "sfm-debug-guided: {i} {j} model={} base={} extra={} ratio={:.3} max_error_px={:.3} cross_check={}",
                        colmap_guided_geometry_name(colmap_guided_geometry(&report)),
                        dm.len(),
                        extra.len(),
                        guided_ratio,
                        guided_max_error,
                        cross_check,
                    );
                    for descriptor_match in &extra {
                        eprintln!(
                            "sfm-debug-guided-match: {i} {j} query={} train={} distance={:.9e}",
                            descriptor_match.query_index,
                            descriptor_match.train_index,
                            descriptor_match.distance,
                        );
                    }
                }
                if extra.is_empty() {
                    (dm, report, corrs)
                } else {
                    let mut expanded = dm.clone();
                    expanded.extend(extra);
                    let new_corrs: Vec<TwoViewCorrespondence> = expanded
                        .iter()
                        .map(|m| {
                            TwoViewCorrespondence::new(
                                features[i].keypoints[m.query_index],
                                features[j].keypoints[m.train_index],
                            )
                        })
                        .collect();
                    let new_report = verifier.classify(&new_corrs, camera);
                    if new_report.inliers.len() >= min_matches {
                        if colmap_guided_matching {
                            // The compatibility mode is append-only by
                            // contract: a new model is allowed to add
                            // verified inliers, but it must not make a
                            // previously accepted baseline correspondence
                            // disappear merely because model selection changed
                            // after expansion.
                            let mut preserved_report = new_report;
                            let mut inliers = preserved_report.inliers.clone();
                            for &index in &original_report.inliers {
                                if !inliers.contains(&index) {
                                    inliers.push(index);
                                }
                            }
                            inliers.sort_unstable();
                            preserved_report.inliers = inliers;
                            (expanded, preserved_report, new_corrs)
                        } else {
                            (expanded, new_report, new_corrs)
                        }
                    } else {
                        (dm, report, corrs)
                    }
                }
            } else {
                (dm, report, corrs)
            };
            if dump_f2e_diagnostics && report.config == ConfigurationType::Uncalibrated {
                if let Some(diagnostics) =
                    f_to_e_candidate_diagnostics(&report, &report_corrs, camera)
                {
                    eprintln!(
                            "sfm-debug-f2e-candidate: {i} {j} s1={:.9e} s2={:.9e} s3={:.9e} projection_distortion={:.6} s1_s2_mismatch={:.6} s3_s2={:.6} f_inliers={} ef_inliers={} ef_overlap_on_f={:.6} f_norm_residual={:.6e} ef_norm_residual_on_f={:.6e} ef_to_f_residual_ratio={:.6} cheirality_ratio={:.6} cheirality_margin={:.6} ef_angle_p25_deg={:.6} stable_refits={} pose_rotation_spread_deg={:.6} pose_translation_spread_deg={:.6}",
                            diagnostics.calibrated_s1,
                            diagnostics.calibrated_s2,
                            diagnostics.calibrated_s3,
                            diagnostics.projection_distortion,
                            diagnostics.s1_s2_mismatch,
                            diagnostics.s3_s2_ratio,
                            diagnostics.f_inliers,
                            diagnostics.ef_inliers,
                            diagnostics.ef_overlap_on_f,
                            diagnostics.f_normalized_residual,
                            diagnostics.ef_normalized_residual_on_f,
                            diagnostics.ef_to_f_residual_ratio,
                            diagnostics.cheirality_ratio,
                            diagnostics.cheirality_margin,
                            diagnostics.ef_angle_p25_deg,
                            diagnostics.stable_refits,
                            diagnostics.pose_rotation_spread_deg,
                            diagnostics.pose_translation_spread_deg,
                        );
                } else {
                    eprintln!("sfm-debug-f2e-candidate: {i} {j} invalid=1");
                }
            }
            if dump_essential_quality {
                let quality = essential_pair_quality(&report, &report_corrs, camera);
                let f2e_quality = fundamental_to_essential_quality(&report, &report_corrs, camera);
                eprintln!(
                        "sfm-debug-essential-quality: {i} {j} raw={} config={} accepted={} e={} f={} h={}{}{}",
                        dm.len(),
                        configuration_name(report.config),
                        report.inliers.len(),
                        report.e_inlier_count,
                        report.f_inlier_count,
                        report.h_inlier_count,
                        format_essential_pair_quality(quality),
                        format_fundamental_to_essential_quality(f2e_quality),
                    );
            }
            let direct_essential_primary = if calibrated_essential_primary {
                select_calibrated_essential_primary(&report, &report_corrs, camera, min_matches)
            } else {
                None
            };
            if let Some(selection) = &direct_essential_primary {
                calibrated_essential_primary_promotions.fetch_add(1, Ordering::Relaxed);
                if dump_pair_outcomes {
                    eprintln!(
                            "sfm-debug-calibrated-essential-primary: {i} {j} f_inliers={} initial_e_inliers={} rescored_e_inliers={} cheirality={}/{} mean_sampson={:.6e}",
                            report.f_inlier_count,
                            selection.initial_inlier_count,
                            selection.inlier_indices.len(),
                            selection.quality.best_cheirality,
                            selection.inlier_indices.len(),
                            selection.quality.mean_sampson,
                        );
                }
            }
            let f_to_e_refinement = if direct_essential_primary.is_none()
                && (refine_uncalibrated_f_to_essential || strict_uncalibrated_f_to_essential)
            {
                refine_uncalibrated_f_winner(&report, &report_corrs, camera, min_matches)
            } else {
                None
            };
            if let Some(refinement) = &f_to_e_refinement {
                f_to_e_refinements.fetch_add(1, Ordering::Relaxed);
                if dump_pair_outcomes {
                    let cheirality_ratio = refinement.quality.best_cheirality as f64
                        / refinement.inlier_indices.len() as f64;
                    let second_over_best = if refinement.quality.best_cheirality > 0 {
                        refinement.quality.second_cheirality as f64
                            / refinement.quality.best_cheirality as f64
                    } else {
                        f64::NAN
                    };
                    eprintln!(
                            "sfm-debug-f2e-refinement: {i} {j} f_inliers={} ef_inliers={} cheirality={}/{} cheirality_ratio={:.6} second_over_best={:.6}",
                            refinement.f_inlier_count,
                            refinement.inlier_indices.len(),
                            refinement.quality.best_cheirality,
                            refinement.inlier_indices.len(),
                            cheirality_ratio,
                            second_over_best,
                        );
                }
            }
            if should_exclude_strict_uncalibrated_f_winner(
                strict_uncalibrated_f_to_essential,
                camera,
                &report,
                f_to_e_refinement.as_ref(),
            ) && direct_essential_primary.is_none()
            {
                strict_f_to_e_exclusions.fetch_add(1, Ordering::Relaxed);
                strict_f_to_e_excluded_inliers.fetch_add(report.inliers.len(), Ordering::Relaxed);
                if dump_pair_outcomes {
                    eprintln!(
                            "sfm-debug-outcome: {i} {j} raw={} config={} accepted={} e={} f={} h={} reason=strict_f2e_gate_rejected",
                            dm.len(),
                            configuration_name(report.config),
                            report.inliers.len(),
                            report.e_inlier_count,
                            report.f_inlier_count,
                            report.h_inlier_count,
                        );
                }
                // There is no rotation-only PairwiseMatches representation;
                // omit this edge so its uncalibrated F observations cannot
                // enter translation initialization or track construction.
                return (None, Some(report.config), None);
            }
            let essential_matches = {
                let ef_ratio = if report.f_inlier_count > 0 {
                    report.e_inlier_count as f64 / report.f_inlier_count as f64
                } else if report.e_inlier_count > 0 {
                    f64::INFINITY
                } else {
                    0.0
                };
                // Always require the strong-E gate before exposing E inliers
                // to prefer-essential edge construction (weak E poisons bearings).
                let strong = report.e_inlier_count >= force_essential_min_e_inliers
                    && ef_ratio >= force_essential_min_ef_ratio
                    && report.essential_inliers.len() >= min_matches;
                if strong {
                    Some(
                        report
                            .essential_inliers
                            .iter()
                            .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                }
            };
            let refined_matches = f_to_e_refinement.as_ref().map(|refinement| {
                refinement
                    .inlier_indices
                    .iter()
                    .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                    .collect::<Vec<_>>()
            });
            let direct_matches = direct_essential_primary.as_ref().map(|selection| {
                selection
                    .inlier_indices
                    .iter()
                    .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                    .collect::<Vec<_>>()
            });
            let winning: Vec<(usize, usize)> = report
                .inliers
                .iter()
                .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                .collect();
            let uncalibrated_ok = !force_essential_uncalibrated_only
                || matches!(report.config, ConfigurationType::Uncalibrated);
            let matches: Vec<(usize, usize)> = if let Some(primary) = &direct_matches {
                primary.clone()
            } else if let Some(refined) = &refined_matches {
                refined.clone()
            } else {
                match (
                    force_essential_matches && uncalibrated_ok && essential_matches.is_some(),
                    essential_matches.as_ref(),
                ) {
                    (true, Some(ess)) => {
                        force_e_swaps.fetch_add(1, Ordering::Relaxed);
                        ess.clone()
                    }
                    _ => winning,
                }
            };
            let essential_matrix = direct_essential_primary
                .as_ref()
                .map(|selection| selection.essential)
                .or_else(|| {
                    f_to_e_refinement
                        .as_ref()
                        .map(|refinement| refinement.essential)
                })
                .or(report.essential);
            let output_config = if direct_essential_primary.is_some() {
                ConfigurationType::Calibrated
            } else {
                report.config
            };
            if dump_pair_outcomes {
                eprintln!(
                        "sfm-debug-outcome: {i} {j} raw={} config={} accepted={} e={} f={} h={} reason=accepted",
                        dm.len(),
                        configuration_name(output_config),
                        matches.len(),
                        report.e_inlier_count,
                        report.f_inlier_count,
                        report.h_inlier_count,
                    );
                // Keep the accepted-set dump self-contained for an
                // order-only replay.  The verified E is diagnostics
                // output only; ordinary reconstruction never serializes
                // or consumes this line.
                let e_values = essential_matrix
                    .as_ref()
                    .map(|matrix| {
                        matrix
                            .as_slice()
                            .iter()
                            .map(|value| format!("{value:.17e}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_else(|| "0 0 0 0 0 0 0 0 0".to_string());
                eprintln!("sfm-debug-essential-matrix: {i} {j} values={e_values}");
            }
            if dump_match_indices {
                for &(query_index, train_index) in &matches {
                    eprintln!("sfm-debug-match: {i} {j} query={query_index} train={train_index}");
                }
            }
            let pair = PairwiseMatches {
                image_i: i,
                image_j: j,
                matches,
                two_view_config: Some(output_config),
                essential_matches: direct_matches.or(refined_matches).or(essential_matches),
                essential_matrix,
            };
            let mut metadata = snapshot_metadata_from_report(&dm, &report);
            metadata.accepted_inlier_indices =
                snapshot_indices_for_matches(&metadata.raw_matches, &pair.matches)
                    .unwrap_or_default();
            metadata.essential_inlier_indices = pair
                .essential_matches
                .as_ref()
                .and_then(|matches| snapshot_indices_for_matches(&metadata.raw_matches, matches))
                .unwrap_or_default();
            (Some(pair), Some(output_config), Some(metadata))
        } else {
            let estimator = match &threshold_only_estimator {
                Some(e) => *e,
                None => RelativePoseEstimator::default(),
            };
            let Some(rel) = estimator.estimate(&corrs, camera) else {
                return (None, None, None);
            };
            if rel.inliers.len() < min_matches {
                return (None, None, None);
            }
            let matches: Vec<(usize, usize)> = rel
                .inliers
                .iter()
                .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                .collect();
            let pair = PairwiseMatches {
                image_i: i,
                image_j: j,
                matches,
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            };
            let metadata = SnapshotPairMetadata {
                raw_match_count: dm.len(),
                raw_matches: dm.iter().map(|m| (m.query_index, m.train_index)).collect(),
                accepted_inlier_indices: rel.inliers.clone(),
                essential_inlier_indices: Vec::new(),
                e_inlier_count: rel.inliers.len(),
                relative_pose: Some((
                    rel.previous_to_current
                        .rotation
                        .to_rotation_matrix()
                        .into_inner(),
                    rel.previous_to_current.translation,
                )),
                ..SnapshotPairMetadata::default()
            };
            (Some(pair), None, Some(metadata))
        }
    };

    // LightGlue holds a single Mutex'd ORT session. Rayon + ORT's internal
    // thread pool deadlocks on this machine; keep LightGlue sequential with
    // progress logs. NN matching stays parallel.
    let results: Vec<(
        Option<PairwiseMatches>,
        Option<ConfigurationType>,
        Option<SnapshotPairMetadata>,
    )> = if sequential {
        let total = candidates.len();
        candidates
            .iter()
            .enumerate()
            .map(|(k, pair)| {
                if k % 25 == 0 || k + 1 == total {
                    eprintln!("lightglue verify: {} / {} pairs", k + 1, total);
                }
                verify_one(pair)
            })
            .collect()
    } else {
        candidates.par_iter().map(verify_one).collect()
    };

    let mut stats = VerificationStats::default();
    let mut pairwise = Vec::with_capacity(results.len());
    let mut metadata_by_pair = HashMap::new();
    for (pair, config, metadata) in results {
        if let Some(config) = config {
            stats.record(config);
        }
        if let Some(pair) = pair {
            let key = (
                pair.image_i.min(pair.image_j),
                pair.image_i.max(pair.image_j),
            );
            if let Some(metadata) = metadata {
                metadata_by_pair.insert(key, metadata);
            }
            pairwise.push(pair);
        }
    }
    stats.force_essential_swaps = force_e_swaps.load(Ordering::Relaxed);
    stats.uncalibrated_f_to_essential_refinements = f_to_e_refinements.load(Ordering::Relaxed);
    stats.strict_uncalibrated_f_to_essential_exclusions =
        strict_f_to_e_exclusions.load(Ordering::Relaxed);
    stats.strict_uncalibrated_f_to_essential_excluded_inliers =
        strict_f_to_e_excluded_inliers.load(Ordering::Relaxed);
    stats.calibrated_essential_primary_promotions =
        calibrated_essential_primary_promotions.load(Ordering::Relaxed);
    (pairwise, stats, metadata_by_pair)
}

/// One rescue-pass candidate's outcome, kept for reporting regardless of
/// whether it was admitted — `main`'s acceptance report (M5,
/// `docs/colmap_port_plan.md`) needs both "which bridges were found" and,
/// in the honest-negative case, "how close did the closest attempt get".
#[derive(Debug, Clone, Copy)]
struct RescueAttempt {
    pair: (usize, usize),
    raw_matches: usize,
    config: ConfigurationType,
    inliers: usize,
}

/// M5 (`docs/colmap_port_plan.md`): opt-in rescue-bridging pass, run after
/// the initial [`verify_pairs`] call. Detects whether the resulting
/// verified-pair graph (`pairwise`) is disconnected
/// (`visloc_rs::vision::two_view::connected_components`); if so, proposes
/// cross-component candidate pairs ranked by a fresh VLAD global-descriptor
/// similarity and budget-capped (`generate_bridge_candidates`), rematches
/// each with the relaxed `--rescue-*` profile, and re-verifies with the same
/// [`TwoViewGeometryVerifier`] / keep-list [`verify_pairs`] itself uses under
/// `--verification-mode full` — a looser matcher only ever *proposes* a
/// bridge here, never *admits* one unverified (the M1.1 lesson).
///
/// Returns the admitted bridge pairs, already in `PairwiseMatches` form and
/// ready to append to the caller's verified-pair list (every attempt's
/// [`RescueAttempt`] outcome — admitted or not — is reported via `println!`
/// as it's produced, per this milestone's acceptance-report requirement).
fn rescue_bridging(
    features: &[FeatureSet],
    image_names: &[String],
    camera: &Camera,
    pairwise: &[PairwiseMatches],
    args: &Args,
    matcher: &PairMatcher,
) -> Result<Vec<PairwiseMatches>, String> {
    let n = features.len();
    let edges: Vec<(usize, usize)> = pairwise.iter().map(|p| (p.image_i, p.image_j)).collect();
    let components = connected_components(n, &edges);
    println!(
        "rescue-bridging: view graph has {} connected component(s) (sizes {})",
        components.len(),
        components
            .iter()
            .map(|c| c.len().to_string())
            .collect::<Vec<_>>()
            .join("+"),
    );
    if components.len() <= 1 {
        println!("rescue-bridging: graph is already connected, nothing to bridge");
        return Ok(Vec::new());
    }

    // Retrieval score for ranking cross-component candidates: a fresh VLAD
    // vocabulary/global descriptor per image, independent of whichever
    // `--pair-source` built the *initial* graph (so this still works under
    // `--pair-source vocab-tree`). Falls back to a uniform (unranked) score
    // if the vocabulary cannot be built — the candidate generator itself
    // still enforces "cross-component only, budget-capped" either way.
    let sample = sampled_training_descriptors(features);
    let globals: Option<Vec<Vec<f32>>> =
        Vocabulary::build(&sample, args.vocab_size, 10, 0).map(|vocab| {
            features
                .iter()
                .map(|f| vlad(&f.descriptors, &vocab))
                .collect()
        });
    let similarity = |i: usize, j: usize| -> f32 {
        match &globals {
            Some(g) => cosine_similarity(&g[i], &g[j]),
            None => 0.0,
        }
    };

    let all_candidates = generate_bridge_candidates(
        &components,
        similarity,
        &BridgeCandidateOptions {
            max_candidates: args.rescue_max_candidates,
        },
    );
    let candidates =
        filter_pairs_by_stem_window(all_candidates, image_names, args.pair_stem_window)?;
    println!(
        "rescue-bridging: {} cross-component candidate pair(s) proposed (ratio={}, cross_check={}, min_matches={})",
        candidates.len(),
        args.rescue_match_ratio,
        args.rescue_cross_check,
        args.rescue_min_matches,
    );

    let verifier = TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(camera, 4.0));

    let results: Vec<(Option<PairwiseMatches>, RescueAttempt)> = candidates
        .par_iter()
        .map(|&(i, j)| {
            let dm = matcher.match_pair(
                args.rescue_match_ratio,
                args.rescue_cross_check,
                i,
                j,
                &features[i],
                &features[j],
            );
            let raw_matches = dm.len();
            if raw_matches < args.rescue_min_matches {
                return (
                    None,
                    RescueAttempt {
                        pair: (i, j),
                        raw_matches,
                        config: ConfigurationType::Degenerate,
                        inliers: 0,
                    },
                );
            }

            let corrs: Vec<TwoViewCorrespondence> = dm
                .iter()
                .map(|m| {
                    TwoViewCorrespondence::new(
                        features[i].keypoints[m.query_index],
                        features[j].keypoints[m.train_index],
                    )
                })
                .collect();
            let report = verifier.classify(&corrs, camera);
            let attempt = RescueAttempt {
                pair: (i, j),
                raw_matches,
                config: report.config,
                inliers: report.inliers.len(),
            };
            // Same keep-list `verify_pairs`'s `full` mode uses (M2.1): every
            // non-DEGENERATE, non-WATERMARK configuration is admissible.
            let keep = matches!(
                report.config,
                ConfigurationType::Calibrated
                    | ConfigurationType::Uncalibrated
                    | ConfigurationType::Planar
                    | ConfigurationType::Panoramic
                    | ConfigurationType::PlanarOrPanoramic
                    | ConfigurationType::Multiple
            );
            if !keep || report.inliers.len() < args.rescue_min_matches {
                return (None, attempt);
            }
            let matches: Vec<(usize, usize)> = report
                .inliers
                .iter()
                .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                .collect();
            let essential_matches = if report.essential_inliers.len() >= args.rescue_min_matches {
                Some(
                    report
                        .essential_inliers
                        .iter()
                        .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                        .collect(),
                )
            } else {
                None
            };
            (
                Some(PairwiseMatches {
                    image_i: i,
                    image_j: j,
                    matches,
                    two_view_config: Some(report.config),
                    essential_matches,
                    essential_matrix: report.essential,
                }),
                attempt,
            )
        })
        .collect();

    let mut admitted = Vec::new();
    let mut attempts = Vec::with_capacity(results.len());
    for (pair, attempt) in results {
        if let Some(pair) = &pair {
            println!(
                "rescue-bridging: BRIDGE admitted ({}, {}) raw_matches={} inliers={} config={:?}",
                attempt.pair.0,
                attempt.pair.1,
                attempt.raw_matches,
                attempt.inliers,
                attempt.config,
            );
            admitted.push(pair.clone());
        }
        attempts.push(attempt);
    }

    if let Some(best) = attempts.iter().max_by_key(|a| a.inliers) {
        println!(
            "rescue-bridging: best cross-component attempt ({}, {}) raw_matches={} inliers={} config={:?}",
            best.pair.0, best.pair.1, best.raw_matches, best.inliers, best.config,
        );
    }
    println!(
        "rescue-bridging: {} bridge pair(s) admitted out of {} attempted",
        admitted.len(),
        candidates.len(),
    );

    if !admitted.is_empty() {
        let mut all_edges = edges.clone();
        all_edges.extend(admitted.iter().map(|p| (p.image_i, p.image_j)));
        let components_after = connected_components(n, &all_edges);
        println!(
            "rescue-bridging: view graph now has {} connected component(s) after admission",
            components_after.len(),
        );
    }

    Ok(admitted)
}

/// The profiles shared by the human-readable `--diagnose-pair` output and the
/// machine-readable CSV export. Keeping this list in one place prevents the
/// two diagnostics from silently measuring different NN match sets.
const DIAGNOSE_PROFILES: [(f32, bool); 4] = [(0.8, true), (0.9, true), (0.95, true), (0.95, false)];

struct DiagnosePairProfile {
    matches: Vec<DescriptorMatch>,
    valid_matches: Vec<(usize, usize)>,
    correspondences: Vec<TwoViewCorrespondence>,
    report: Option<TwoViewGeometryReport>,
}

fn diagnose_pair_profile(
    features: &[FeatureSet],
    camera: &Camera,
    verifier: &TwoViewGeometryVerifier,
    matcher: &PairMatcher,
    i: usize,
    j: usize,
    ratio: f32,
    cross_check: bool,
) -> DiagnosePairProfile {
    let matches = matcher.match_pair(ratio, cross_check, i, j, &features[i], &features[j]);
    let valid_matches: Vec<(usize, usize)> = matches
        .iter()
        .filter_map(|m| {
            features[i]
                .keypoints
                .get(m.query_index)
                .and_then(|_| features[j].keypoints.get(m.train_index))
                .map(|_| (m.query_index, m.train_index))
        })
        .collect();
    let correspondences: Vec<TwoViewCorrespondence> = valid_matches
        .iter()
        .map(|&(query_index, train_index)| {
            TwoViewCorrespondence::new(
                features[i].keypoints[query_index],
                features[j].keypoints[train_index],
            )
        })
        .collect();
    let report = (correspondences.len() >= 8).then(|| verifier.classify(&correspondences, camera));
    DiagnosePairProfile {
        matches,
        valid_matches,
        correspondences,
        report,
    }
}

struct DiagnoseImportedRaw {
    valid_matches: Vec<(usize, usize)>,
    report: Option<TwoViewGeometryReport>,
}

/// Deterministic, GT-independent diagnostics for an essential-matrix report.
///
/// The verifier already exposes the four-hypothesis cheirality scores through
/// `recover_relative_pose_with_options`.  The remaining values are computed
/// from a bounded, deterministic prefix of the E inliers so a full courtyard
/// graph can be audited without making the normal mapper pay for another
/// triangulation pass.  `depth_ratio` is `min(z1,z2)/max(z1,z2)` for positive
/// depths; it is a scale-free conditioning proxy, not a metric depth claim.
#[derive(Debug, Clone, Copy)]
struct EssentialPairQuality {
    best_cheirality: i64,
    second_cheirality: i64,
    cheirality_ratio: f64,
    mean_sampson: f64,
    /// Winning `R` as `(w,x,y,z)` and camera-2 centre direction in camera-1
    /// coordinates.  These are emitted only as diagnostic fields so the
    /// optional GT audit can compare direct-E and F→E poses without changing
    /// the mapper data path.
    rotation_quaternion: [f64; 4],
    center_direction: [f64; 3],
    angle_samples: usize,
    angle_ge_1deg: usize,
    angle_p10_deg: f64,
    angle_p25_deg: f64,
    angle_median_deg: f64,
    depth_ratio_p10: f64,
    depth_ratio_p25: f64,
    depth_ratio_median: f64,
}

fn quantile_nearest_rank(values: &mut [f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let rank = (fraction.clamp(0.0, 1.0) * values.len() as f64).ceil() as usize;
    values[rank.saturating_sub(1).min(values.len() - 1)]
}

fn essential_pair_quality(
    report: &TwoViewGeometryReport,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<EssentialPairQuality> {
    essential_pair_quality_for_inliers_with_options(
        report.essential.as_ref()?,
        &report.essential_inliers,
        correspondences,
        camera,
        &CheiralityOptions::default(),
    )
}

fn essential_pair_quality_for_inliers(
    essential: &Matrix3<f64>,
    e_inliers: &[usize],
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<EssentialPairQuality> {
    essential_pair_quality_for_inliers_with_options(
        essential,
        e_inliers,
        correspondences,
        camera,
        &CheiralityOptions::default(),
    )
}

fn essential_pair_quality_for_inliers_with_options(
    essential: &Matrix3<f64>,
    e_inliers: &[usize],
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    cheirality_options: &CheiralityOptions,
) -> Option<EssentialPairQuality> {
    if e_inliers.len() < 8 {
        return None;
    }
    let recovery = recover_relative_pose_with_options(
        essential,
        correspondences,
        camera,
        e_inliers,
        cheirality_options,
    )?;
    let cheirality_ratio = if e_inliers.is_empty() {
        f64::NAN
    } else {
        recovery.best_score as f64 / e_inliers.len() as f64
    };
    let mean_sampson =
        mean_normalized_essential_sampson_error(essential, correspondences, camera, e_inliers);
    let q = recovery.rotation.quaternion();
    let center = -recovery
        .rotation
        .inverse()
        .transform_vector(&recovery.translation_unit);
    let center_direction = center
        .try_normalize(1.0e-12)
        .map(|value| [value.x, value.y, value.z])
        .unwrap_or([f64::NAN; 3]);
    let rotation_quaternion = [q.w, q.i, q.j, q.k];

    const MAX_TRIANGULATION_SAMPLES: usize = 256;
    let stride = e_inliers.len().div_ceil(MAX_TRIANGULATION_SAMPLES);
    let left_to_right = SE3::new(recovery.rotation, recovery.translation_unit);
    let camera_2_center = -recovery
        .rotation
        .inverse()
        .transform_vector(&recovery.translation_unit);
    let mut angles = Vec::new();
    let mut depth_ratios = Vec::new();
    for &inlier_index in e_inliers.iter().step_by(stride.max(1)) {
        let Some(corr) = correspondences.get(inlier_index) else {
            continue;
        };
        let Some(point) = triangulate_two_view_left_frame(
            camera,
            camera,
            &left_to_right,
            &corr.previous_xy,
            &corr.current_xy,
        ) else {
            continue;
        };
        let point_2 = left_to_right.transform_point(&point);
        if !point.coords.iter().all(|v| v.is_finite())
            || !point_2.coords.iter().all(|v| v.is_finite())
            || point.z <= 0.0
            || point_2.z <= 0.0
        {
            continue;
        }
        let Some(ray_1) = point.coords.try_normalize(1.0e-12) else {
            continue;
        };
        let Some(ray_2) = (point.coords - camera_2_center).try_normalize(1.0e-12) else {
            continue;
        };
        let angle = ray_1.dot(&ray_2).clamp(-1.0, 1.0).acos().to_degrees();
        if !angle.is_finite() {
            continue;
        }
        angles.push(angle);
        let z1 = point.z;
        let z2 = point_2.z;
        let depth_ratio = z1.min(z2) / z1.max(z2);
        if depth_ratio.is_finite() {
            depth_ratios.push(depth_ratio);
        }
    }
    if angles.is_empty() || depth_ratios.is_empty() {
        return Some(EssentialPairQuality {
            best_cheirality: recovery.best_score,
            second_cheirality: recovery.second_score,
            cheirality_ratio,
            mean_sampson,
            rotation_quaternion,
            center_direction,
            angle_samples: 0,
            angle_ge_1deg: 0,
            angle_p10_deg: f64::NAN,
            angle_p25_deg: f64::NAN,
            angle_median_deg: f64::NAN,
            depth_ratio_p10: f64::NAN,
            depth_ratio_p25: f64::NAN,
            depth_ratio_median: f64::NAN,
        });
    }
    let angle_ge_1deg = angles.iter().filter(|&&angle| angle >= 1.0).count();
    let angle_p10_deg = quantile_nearest_rank(&mut angles, 0.10);
    let angle_p25_deg = quantile_nearest_rank(&mut angles, 0.25);
    let angle_median_deg = quantile_nearest_rank(&mut angles, 0.50);
    let depth_ratio_p10 = quantile_nearest_rank(&mut depth_ratios, 0.10);
    let depth_ratio_p25 = quantile_nearest_rank(&mut depth_ratios, 0.25);
    let depth_ratio_median = quantile_nearest_rank(&mut depth_ratios, 0.50);
    Some(EssentialPairQuality {
        best_cheirality: recovery.best_score,
        second_cheirality: recovery.second_score,
        cheirality_ratio,
        mean_sampson,
        rotation_quaternion,
        center_direction,
        angle_samples: angles.len(),
        angle_ge_1deg,
        angle_p10_deg,
        angle_p25_deg,
        angle_median_deg,
        depth_ratio_p10,
        depth_ratio_p25,
        depth_ratio_median,
    })
}

/// Normalized-coordinate Sampson distance for an essential matrix.  This is
/// intentionally kept local to the read-only diagnostic: the production
/// verifier already uses the same expression in `two_view::sampson_distance`
/// and its threshold units are unchanged here.
fn normalized_essential_squared_sampson_error(
    essential: &Matrix3<f64>,
    correspondence: &TwoViewCorrespondence,
    camera: &Camera,
) -> Option<f64> {
    let previous = camera.normalize_pixel(&correspondence.previous_xy)?;
    let current = camera.normalize_pixel(&correspondence.current_xy)?;
    let previous_h = Vector3::new(previous.x, previous.y, 1.0);
    let current_h = Vector3::new(current.x, current.y, 1.0);
    let e_previous = essential * previous_h;
    let et_current = essential.transpose() * current_h;
    let numerator = current_h.dot(&e_previous).powi(2);
    let denominator =
        e_previous.x.powi(2) + e_previous.y.powi(2) + et_current.x.powi(2) + et_current.y.powi(2);
    if denominator < 1.0e-18 {
        return None;
    }
    let error = numerator / denominator;
    error.is_finite().then_some(error)
}

fn mean_normalized_essential_sampson_error(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    indices: &[usize],
) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for &index in indices {
        let Some(correspondence) = correspondences.get(index) else {
            continue;
        };
        let Some(error_sq) =
            normalized_essential_squared_sampson_error(essential, correspondence, camera)
        else {
            continue;
        };
        total += error_sq.sqrt();
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        total / count as f64
    }
}

/// Deterministic held-out partition for the completed-model cross-validation
/// probe.  The hash depends only on the imported pair and feature indices, so
/// it is independent of mapper traversal, track conflicts, and the candidate
/// order used to produce the model.
#[cfg(test)]
fn model_cross_validation_is_held_out(
    image_i: usize,
    image_j: usize,
    keypoint_i: usize,
    keypoint_j: usize,
) -> bool {
    let mut hash = 0xcbf29ce484222325u64;
    for value in [image_i, image_j, keypoint_i, keypoint_j] {
        for byte in (value as u64).to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3u64);
        }
    }
    hash % 5 == 0
}

/// Order-independent form of the held-out hash.  Feature permutations preserve
/// the physical endpoint coordinates, while keypoint indices need not remain
/// stable; quantizing at 1/1000 pixel keeps decimal feature-file round trips
/// in the same partition without making co-located duplicate rows diverge.
fn model_cross_validation_is_held_out_for_pixels(
    image_i: usize,
    image_j: usize,
    pixel_i: &Point2<f64>,
    pixel_j: &Point2<f64>,
) -> bool {
    let quantize = |value: f64| {
        if value.is_finite() {
            (value * 1_000.0).round() as i64 as u64
        } else {
            u64::MAX
        }
    };
    let mut hash = 0xcbf29ce484222325u64;
    for value in [
        image_i as u64,
        image_j as u64,
        quantize(pixel_i.x),
        quantize(pixel_i.y),
        quantize(pixel_j.x),
        quantize(pixel_j.y),
    ] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3u64);
        }
    }
    hash % 5 == 0
}

#[derive(Debug, Default)]
struct ModelCrossValidationBucket {
    observations: usize,
    residuals: Vec<f64>,
    under_threshold: usize,
    triangulated: usize,
    positive_depth: usize,
    angle_ge_one_degree: usize,
}

impl ModelCrossValidationBucket {
    fn record(
        &mut self,
        residual: Option<f64>,
        threshold: f64,
        triangulated: bool,
        positive_depth: bool,
        angle_ge_one_degree: bool,
    ) {
        self.observations += 1;
        if let Some(residual) = residual.filter(|value| value.is_finite()) {
            self.residuals.push(residual);
            if residual <= threshold {
                self.under_threshold += 1;
            }
        }
        if triangulated {
            self.triangulated += 1;
        }
        if positive_depth {
            self.positive_depth += 1;
        }
        if angle_ge_one_degree {
            self.angle_ge_one_degree += 1;
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ModelCrossValidationBucketSummary {
    observations: usize,
    residual_samples: usize,
    under_threshold: usize,
    triangulated: usize,
    positive_depth: usize,
    angle_ge_one_degree: usize,
    mean_sampson_root: f64,
    median_sampson_root: f64,
    p90_sampson_root: f64,
    under_fraction: f64,
    positive_fraction: f64,
    angle_fraction: f64,
}

fn fraction_or_nan(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        f64::NAN
    } else {
        numerator as f64 / denominator as f64
    }
}

fn summarize_model_cross_validation_bucket(
    bucket: &mut ModelCrossValidationBucket,
) -> ModelCrossValidationBucketSummary {
    bucket.residuals.sort_by(f64::total_cmp);
    let residual_samples = bucket.residuals.len();
    let mean_sampson_root = if residual_samples == 0 {
        f64::NAN
    } else {
        bucket.residuals.iter().sum::<f64>() / residual_samples as f64
    };
    let quantile = |fraction: f64| {
        if residual_samples == 0 {
            f64::NAN
        } else {
            let index = ((residual_samples as f64 * fraction).ceil() as usize)
                .saturating_sub(1)
                .min(residual_samples - 1);
            bucket.residuals[index]
        }
    };
    ModelCrossValidationBucketSummary {
        observations: bucket.observations,
        residual_samples,
        under_threshold: bucket.under_threshold,
        triangulated: bucket.triangulated,
        positive_depth: bucket.positive_depth,
        angle_ge_one_degree: bucket.angle_ge_one_degree,
        mean_sampson_root,
        median_sampson_root: quantile(0.5),
        p90_sampson_root: quantile(0.9),
        under_fraction: fraction_or_nan(bucket.under_threshold, residual_samples),
        positive_fraction: fraction_or_nan(bucket.positive_depth, bucket.triangulated),
        angle_fraction: fraction_or_nan(bucket.angle_ge_one_degree, bucket.triangulated),
    }
}

/// Huber location estimate for a small set of already pair-balanced angular
/// errors.  The scale comes from the median absolute deviation and the
/// conventional 1.345 Huber tuning constant; no scene/GT-specific threshold
/// is used.  A zero-MAD set is already robustly constant, so its median is
/// returned directly.
fn robust_huber_mean(values: &[f64]) -> f64 {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.sort_by(f64::total_cmp);
    let median = finite[finite.len() / 2];
    let mut deviations = finite
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    let mad = deviations[deviations.len() / 2];
    if mad <= 1.0e-12 {
        return median;
    }
    let scale = 1.4826 * mad;
    let delta = 1.345 * scale;
    let mut location = median;
    for _ in 0..8 {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        for value in &finite {
            let residual = *value - location;
            let weight = if residual.abs() <= delta {
                1.0
            } else {
                delta / residual.abs()
            };
            weighted_sum += weight * *value;
            weight_sum += weight;
        }
        if !(weight_sum.is_finite() && weight_sum > 0.0) {
            return f64::NAN;
        }
        let next = weighted_sum / weight_sum;
        if !next.is_finite() {
            return f64::NAN;
        }
        if (next - location).abs() <= 1.0e-12 {
            location = next;
            break;
        }
        location = next;
    }
    location
}

#[derive(Debug, Clone)]
struct ModelCrossValidationPairScore {
    image_i: usize,
    image_j: usize,
    config: ConfigurationType,
    verified_inliers: usize,
    /// Rotation disagreement to the imported pair E, when the file carries
    /// one. This is a diagnostic reference for the rotation-only alternative;
    /// it is not used by the mapper or the calibrated residual score.
    rotation_disagreement_deg: f64,
    /// Rotation disagreement on the subset of imported-E references that
    /// passed the strong cheirality/parallax/stability gate below.
    stable_rotation_disagreement_deg: f64,
    /// Signed (cheirality-selected) camera-centre direction disagreement to
    /// the imported calibrated-E reference, in degrees.
    translation_disagreement_deg: f64,
    /// Quality of the imported-E reference used for the two direction fields.
    reference_cheirality_margin: f64,
    reference_angle_p25_deg: f64,
    reference_stable_refits: usize,
    reference_rotation_spread_deg: f64,
    reference_translation_spread_deg: f64,
    all: ModelCrossValidationBucketSummary,
    held_out: ModelCrossValidationBucketSummary,
}

#[derive(Debug, Clone, Copy)]
struct ImportedEssentialReferenceQuality {
    rotation: UnitQuaternion<f64>,
    center_direction: Vector3<f64>,
    cheirality_margin: f64,
    angle_p25_deg: f64,
    stable_refits: usize,
    rotation_spread_deg: f64,
    translation_spread_deg: f64,
}

#[derive(Debug, Clone, Default)]
struct ModelCrossValidationSummary {
    imported_pairs: usize,
    registered_images: usize,
    registered_pairs: usize,
    invalid_correspondences: usize,
    normalized_threshold: f64,
    all: ModelCrossValidationBucketSummary,
    held_out: ModelCrossValidationBucketSummary,
    pair_balanced_mean_sampson_root: f64,
    pair_balanced_median_sampson_root: f64,
    pair_balanced_p90_sampson_root: f64,
    pair_balanced_under_fraction: f64,
    pair_balanced_positive_fraction: f64,
    pair_balanced_angle_fraction: f64,
    pair_balanced_rotation_disagreement_deg: f64,
    rotation_reference_pairs: usize,
    pair_balanced_stable_rotation_disagreement_deg: f64,
    pair_balanced_translation_disagreement_deg: f64,
    pair_balanced_translation_median_deg: f64,
    pair_balanced_translation_p90_deg: f64,
    pair_balanced_translation_huber_deg: f64,
    stable_rotation_reference_pairs: usize,
    translation_reference_pairs: usize,
    translation_reference_coverage: f64,
    image_balanced_under_fraction: f64,
    image_balanced_positive_fraction: f64,
    image_balanced_angle_fraction: f64,
    image_balanced_median_sampson_root: f64,
    image_balanced_translation_disagreement_deg: f64,
    image_translation_reference_coverage: f64,
    pairs: Vec<ModelCrossValidationPairScore>,
}

/// GT-independent ranking score for multi-hypothesis diagnostics.  This is
/// intentionally only available when at least three calibrated imported-E
/// pair references are present; callers must compare models scored against the
/// same verified multiset and camera.  Lower is better.  It is reported only
/// and never selects or mutates the normal reconstruction path.
fn model_cross_validation_selection_score(summary: &ModelCrossValidationSummary) -> Option<f64> {
    (summary.rotation_reference_pairs >= 3
        && summary.pair_balanced_rotation_disagreement_deg.is_finite())
    .then_some(summary.pair_balanced_rotation_disagreement_deg)
}

#[derive(Debug, Default)]
struct ModelCrossValidationImageAccumulator {
    median_sampson_sum: f64,
    under_sum: f64,
    positive_sum: f64,
    angle_sum: f64,
    translation_sum: f64,
    translation_count: usize,
    reference_attempted: usize,
    reference_valid: usize,
    count: usize,
}

/// Score a completed pose model against the complete imported verified set.
///
/// This is intentionally a post-hoc diagnostic: it does not inspect the
/// reconstruction's retained tracks, so pair edges and observations rejected
/// by union-find conflicts still contribute.  Residuals use the pose-induced
/// calibrated essential matrix and the same normalized threshold family as
/// the full verifier.  Positive depth and a one-degree triangulation-angle
/// gate are reported separately rather than silently folded into the residual
/// score.  Pair and image summaries are balanced so a dense pair cannot
/// dominate the model comparison.
fn score_model_against_verified_pairs(
    model_images_path: &Path,
    imported: &[ImportedVerifiedPair],
    features: &[FeatureSet],
    image_names: &[String],
    camera: &Camera,
) -> Result<ModelCrossValidationSummary, Box<dyn std::error::Error>> {
    let poses_by_stem = poses_from_colmap_images_txt(model_images_path)
        .map_err(|error| format!("model cross-validation: {error}"))?;
    let registered_images = image_names
        .iter()
        .filter(|name| poses_by_stem.contains_key(image_stem(name)))
        .count();
    let normalized_threshold = TwoViewGeometryOptions::for_camera(camera, 4.0)
        .essential_sampson_threshold
        .sqrt();
    let mut all_bucket = ModelCrossValidationBucket::default();
    let mut held_out_bucket = ModelCrossValidationBucket::default();
    let mut pair_scores = Vec::new();
    let mut invalid_correspondences = 0usize;
    let mut image_accumulators: HashMap<usize, ModelCrossValidationImageAccumulator> =
        HashMap::new();
    // References are counted only after both candidate cameras are present;
    // this makes coverage describe the diagnostic population that could
    // actually be compared, rather than all rows in the replay file.
    let mut reference_attempted_pairs = 0usize;

    for pair in imported {
        let Some(pose_i) = image_names
            .get(pair.image_i)
            .and_then(|name| poses_by_stem.get(image_stem(name)))
        else {
            continue;
        };
        let Some(pose_j) = image_names
            .get(pair.image_j)
            .and_then(|name| poses_by_stem.get(image_stem(name)))
        else {
            continue;
        };
        let Some(essential) = essential_from_absolute_poses(pose_i, pose_j) else {
            continue;
        };
        let left_to_right = pose_j
            .world_to_camera
            .compose(&pose_i.world_to_camera.inverse());
        let camera_2_center = -left_to_right
            .rotation
            .inverse()
            .transform_vector(&left_to_right.translation);
        let mut pair_all = ModelCrossValidationBucket::default();
        let mut pair_held_out = ModelCrossValidationBucket::default();
        let mut valid_correspondences = Vec::with_capacity(pair.matches.len());

        for &(keypoint_i, keypoint_j) in &pair.matches {
            let Some(pixel_i) = features
                .get(pair.image_i)
                .and_then(|feature_set| feature_set.keypoints.get(keypoint_i))
            else {
                invalid_correspondences += 1;
                continue;
            };
            let Some(pixel_j) = features
                .get(pair.image_j)
                .and_then(|feature_set| feature_set.keypoints.get(keypoint_j))
            else {
                invalid_correspondences += 1;
                continue;
            };
            let correspondence = TwoViewCorrespondence::new(*pixel_i, *pixel_j);
            valid_correspondences.push(correspondence);
            let residual =
                normalized_essential_squared_sampson_error(&essential, &correspondence, camera)
                    .map(f64::sqrt);
            let triangulated_point =
                triangulate_two_view_left_frame(camera, camera, &left_to_right, pixel_i, pixel_j);
            let triangulated = triangulated_point.as_ref().is_some_and(|point| {
                point.coords.iter().all(|value| value.is_finite())
                    && left_to_right
                        .transform_point(point)
                        .coords
                        .iter()
                        .all(|value| value.is_finite())
            });
            let positive_depth = triangulated
                && triangulated_point.as_ref().is_some_and(|point| {
                    let point_j = left_to_right.transform_point(point);
                    point.z > 0.0 && point_j.z > 0.0
                });
            let angle_ge_one_degree = triangulated_point.as_ref().is_some_and(|point| {
                if !positive_depth {
                    return false;
                }
                let Some(ray_i) = point.coords.try_normalize(1.0e-12) else {
                    return false;
                };
                let Some(ray_j) = (point.coords - camera_2_center).try_normalize(1.0e-12) else {
                    return false;
                };
                let angle = ray_i.dot(&ray_j).clamp(-1.0, 1.0).acos().to_degrees();
                angle.is_finite() && angle >= 1.0
            });
            pair_all.record(
                residual,
                normalized_threshold,
                triangulated,
                positive_depth,
                angle_ge_one_degree,
            );
            all_bucket.record(
                residual,
                normalized_threshold,
                triangulated,
                positive_depth,
                angle_ge_one_degree,
            );
            if model_cross_validation_is_held_out_for_pixels(
                pair.image_i,
                pair.image_j,
                pixel_i,
                pixel_j,
            ) {
                pair_held_out.record(
                    residual,
                    normalized_threshold,
                    triangulated,
                    positive_depth,
                    angle_ge_one_degree,
                );
                held_out_bucket.record(
                    residual,
                    normalized_threshold,
                    triangulated,
                    positive_depth,
                    angle_ge_one_degree,
                );
            }
        }

        let all = summarize_model_cross_validation_bucket(&mut pair_all);
        let held_out = summarize_model_cross_validation_bucket(&mut pair_held_out);
        // The imported E is a calibrated reference only for configurations in
        // which the verifier actually selected a calibrated model.  F-winning
        // and planar rows may carry an auxiliary E diagnostic, but treating it
        // as a pose reference would mix incomparable model hypotheses.
        let has_calibrated_reference = matches!(
            pair.config,
            ConfigurationType::Calibrated | ConfigurationType::Multiple
        ) && pair.essential_matrix.is_some();
        if has_calibrated_reference {
            reference_attempted_pairs += 1;
        }
        let reference_quality = if has_calibrated_reference {
            pair.essential_matrix
                .as_ref()
                .and_then(|imported_essential| {
                    imported_essential_reference_quality(
                        imported_essential,
                        &valid_correspondences,
                        camera,
                    )
                })
        } else {
            None
        };
        let rotation_disagreement_deg = if matches!(
            pair.config,
            ConfigurationType::Calibrated | ConfigurationType::Multiple
        ) {
            pair.essential_matrix
                .as_ref()
                .and_then(|imported_essential| {
                    relative_pose_from_essential(imported_essential, &valid_correspondences, camera)
                })
                .map(|imported_pose| {
                    (left_to_right.rotation.inverse() * imported_pose.previous_to_current.rotation)
                        .angle()
                        .to_degrees()
                })
                .filter(|value| value.is_finite())
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };
        let stable_rotation_disagreement_deg = reference_quality
            .as_ref()
            .map(|reference| {
                (left_to_right.rotation.inverse() * reference.rotation)
                    .angle()
                    .to_degrees()
            })
            .filter(|value| value.is_finite())
            .unwrap_or(f64::NAN);
        let candidate_center_direction = camera_2_center.try_normalize(1.0e-12);
        let translation_disagreement_deg = reference_quality
            .as_ref()
            .zip(candidate_center_direction.as_ref())
            .map(|(reference, candidate)| {
                translation_direction_delta_deg(candidate, &reference.center_direction)
            })
            .filter(|value| value.is_finite())
            .unwrap_or(f64::NAN);
        for image_index in [pair.image_i, pair.image_j] {
            let accumulator = image_accumulators.entry(image_index).or_default();
            if has_calibrated_reference {
                accumulator.reference_attempted += 1;
            }
            if reference_quality.is_some() {
                accumulator.reference_valid += 1;
            }
            if translation_disagreement_deg.is_finite() {
                accumulator.translation_sum += translation_disagreement_deg;
                accumulator.translation_count += 1;
            }
            if all.residual_samples > 0 {
                accumulator.median_sampson_sum += all.median_sampson_root;
                accumulator.under_sum += all.under_fraction;
                accumulator.positive_sum += all.positive_fraction;
                accumulator.angle_sum += all.angle_fraction;
                accumulator.count += 1;
            }
        }
        pair_scores.push(ModelCrossValidationPairScore {
            image_i: pair.image_i,
            image_j: pair.image_j,
            config: pair.config,
            verified_inliers: pair.matches.len(),
            rotation_disagreement_deg,
            stable_rotation_disagreement_deg,
            translation_disagreement_deg,
            reference_cheirality_margin: reference_quality
                .as_ref()
                .map(|reference| reference.cheirality_margin)
                .unwrap_or(f64::NAN),
            reference_angle_p25_deg: reference_quality
                .as_ref()
                .map(|reference| reference.angle_p25_deg)
                .unwrap_or(f64::NAN),
            reference_stable_refits: reference_quality
                .as_ref()
                .map(|reference| reference.stable_refits)
                .unwrap_or(0),
            reference_rotation_spread_deg: reference_quality
                .as_ref()
                .map(|reference| reference.rotation_spread_deg)
                .unwrap_or(f64::NAN),
            reference_translation_spread_deg: reference_quality
                .as_ref()
                .map(|reference| reference.translation_spread_deg)
                .unwrap_or(f64::NAN),
            all,
            held_out,
        });
    }

    let all = summarize_model_cross_validation_bucket(&mut all_bucket);
    let held_out = summarize_model_cross_validation_bucket(&mut held_out_bucket);
    let finite_pair_values = |select: fn(&ModelCrossValidationBucketSummary) -> f64| {
        let mut values: Vec<f64> = pair_scores
            .iter()
            .map(|pair| select(&pair.all))
            .filter(|value| value.is_finite())
            .collect();
        if values.is_empty() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        values.sort_by(f64::total_cmp);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let median = values[values.len() / 2];
        let p90 = values[((values.len() * 9).saturating_sub(1) / 10).min(values.len() - 1)];
        (mean, median, p90)
    };
    let (
        pair_balanced_mean_sampson_root,
        pair_balanced_median_sampson_root,
        pair_balanced_p90_sampson_root,
    ) = finite_pair_values(|summary| summary.mean_sampson_root);
    let (pair_balanced_under_fraction, _, _) = finite_pair_values(|summary| summary.under_fraction);
    let (pair_balanced_positive_fraction, _, _) =
        finite_pair_values(|summary| summary.positive_fraction);
    let (pair_balanced_angle_fraction, _, _) = finite_pair_values(|summary| summary.angle_fraction);
    let (rotation_sum, rotation_count) = pair_scores
        .iter()
        .map(|pair| pair.rotation_disagreement_deg)
        .filter(|value| value.is_finite())
        .fold((0.0, 0usize), |(sum, count), value| {
            (sum + value, count + 1)
        });
    let pair_balanced_rotation_disagreement_deg = if rotation_count == 0 {
        f64::NAN
    } else {
        rotation_sum / rotation_count as f64
    };
    let stable_rotation_values = pair_scores
        .iter()
        .map(|pair| pair.stable_rotation_disagreement_deg)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let translation_values = pair_scores
        .iter()
        .map(|pair| pair.translation_disagreement_deg)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let pair_balanced_stable_rotation_disagreement_deg = robust_huber_mean(&stable_rotation_values);
    let pair_balanced_translation_disagreement_deg = if translation_values.is_empty() {
        f64::NAN
    } else {
        translation_values.iter().sum::<f64>() / translation_values.len() as f64
    };
    let pair_balanced_translation_median_deg = if translation_values.is_empty() {
        f64::NAN
    } else {
        let mut sorted = translation_values.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    };
    let pair_balanced_translation_p90_deg = if translation_values.is_empty() {
        f64::NAN
    } else {
        let mut sorted = translation_values.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[((sorted.len() * 9).saturating_sub(1) / 10).min(sorted.len() - 1)]
    };
    let pair_balanced_translation_huber_deg = robust_huber_mean(&translation_values);
    let mut image_under = Vec::new();
    let mut image_positive = Vec::new();
    let mut image_angle = Vec::new();
    let mut image_median_sampson = Vec::new();
    let mut image_translation = Vec::new();
    let mut image_translation_coverage = Vec::new();
    for accumulator in image_accumulators.values() {
        if accumulator.count > 0 {
            let count = accumulator.count as f64;
            image_median_sampson.push(accumulator.median_sampson_sum / count);
            image_under.push(accumulator.under_sum / count);
            image_positive.push(accumulator.positive_sum / count);
            image_angle.push(accumulator.angle_sum / count);
        }
        if accumulator.translation_count > 0 {
            image_translation
                .push(accumulator.translation_sum / accumulator.translation_count as f64);
        }
        if accumulator.reference_attempted > 0 {
            image_translation_coverage
                .push(accumulator.reference_valid as f64 / accumulator.reference_attempted as f64);
        }
    }
    let mean = |values: &[f64]| {
        if values.is_empty() {
            f64::NAN
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    };
    Ok(ModelCrossValidationSummary {
        imported_pairs: imported.len(),
        registered_images,
        registered_pairs: pair_scores.len(),
        invalid_correspondences,
        normalized_threshold,
        all,
        held_out,
        pair_balanced_mean_sampson_root,
        pair_balanced_median_sampson_root,
        pair_balanced_p90_sampson_root,
        pair_balanced_under_fraction,
        pair_balanced_positive_fraction,
        pair_balanced_angle_fraction,
        pair_balanced_rotation_disagreement_deg,
        rotation_reference_pairs: rotation_count,
        pair_balanced_stable_rotation_disagreement_deg,
        pair_balanced_translation_disagreement_deg,
        pair_balanced_translation_median_deg,
        pair_balanced_translation_p90_deg,
        pair_balanced_translation_huber_deg,
        stable_rotation_reference_pairs: stable_rotation_values.len(),
        translation_reference_pairs: translation_values.len(),
        translation_reference_coverage: if reference_attempted_pairs == 0 {
            f64::NAN
        } else {
            translation_values.len() as f64 / reference_attempted_pairs as f64
        },
        image_balanced_under_fraction: mean(&image_under),
        image_balanced_positive_fraction: mean(&image_positive),
        image_balanced_angle_fraction: mean(&image_angle),
        image_balanced_median_sampson_root: mean(&image_median_sampson),
        image_balanced_translation_disagreement_deg: robust_huber_mean(&image_translation),
        image_translation_reference_coverage: mean(&image_translation_coverage),
        pairs: pair_scores,
    })
}

fn print_model_cross_validation_summary(
    summary: &ModelCrossValidationSummary,
    model_images_path: &Path,
    verified_pairs_path: &Path,
    image_names: &[String],
) {
    for pair in &summary.pairs {
        println!(
            "model-xval-pair: i={} j={} image_i_name={} image_j_name={} config={:?} verified={} rotation_ref_deg={:.6} stable_rotation_ref_deg={:.6} translation_ref_deg={:.6} ref_cheirality_margin={:.6} ref_angle_p25_deg={:.6} ref_stable_refits={} ref_rotation_spread_deg={:.6} ref_translation_spread_deg={:.6} observations={} residual_n={} mean={:.9e} median={:.9e} p90={:.9e} under={:.6} under_n={} triangulated={} positive={:.6} positive_n={} angle_ge_1deg={:.6} angle_n={} heldout_n={} heldout_under={:.6} heldout_positive={:.6} heldout_angle_ge_1deg={:.6}",
            pair.image_i,
            pair.image_j,
            image_names
                .get(pair.image_i)
                .map(String::as_str)
                .unwrap_or("<unknown>"),
            image_names
                .get(pair.image_j)
                .map(String::as_str)
                .unwrap_or("<unknown>"),
            pair.config,
            pair.verified_inliers,
            pair.rotation_disagreement_deg,
            pair.stable_rotation_disagreement_deg,
            pair.translation_disagreement_deg,
            pair.reference_cheirality_margin,
            pair.reference_angle_p25_deg,
            pair.reference_stable_refits,
            pair.reference_rotation_spread_deg,
            pair.reference_translation_spread_deg,
            pair.all.observations,
            pair.all.residual_samples,
            pair.all.mean_sampson_root,
            pair.all.median_sampson_root,
            pair.all.p90_sampson_root,
            pair.all.under_fraction,
            pair.all.under_threshold,
            pair.all.triangulated,
            pair.all.positive_fraction,
            pair.all.positive_depth,
            pair.all.angle_fraction,
            pair.all.angle_ge_one_degree,
            pair.held_out.residual_samples,
            pair.held_out.under_fraction,
            pair.held_out.positive_fraction,
            pair.held_out.angle_fraction,
        );
    }
    println!(
        "model-xval-summary: model={} verified_file={} imported_pairs={} registered_images={} registered_pairs={} invalid_correspondences={} normalized_threshold={:.9e} all_observations={} all_residual_n={} all_mean={:.9e} all_median={:.9e} all_p90={:.9e} all_under={:.6} all_positive={:.6} all_angle_ge_1deg={:.6} pair_mean={:.9e} pair_median={:.9e} pair_p90={:.9e} pair_under={:.6} pair_positive={:.6} pair_angle_ge_1deg={:.6} pair_rotation_ref_deg={:.6} rotation_ref_pairs={} pair_stable_rotation_ref_deg={:.6} stable_rotation_ref_pairs={} pair_translation_ref_deg={:.6} pair_translation_median_deg={:.6} pair_translation_p90_deg={:.6} pair_translation_huber_deg={:.6} translation_ref_pairs={} translation_ref_coverage={:.6} selection_score_deg={:.6} image_median={:.9e} image_under={:.6} image_positive={:.6} image_angle_ge_1deg={:.6} image_translation_ref_deg={:.6} image_translation_ref_coverage={:.6} heldout_observations={} heldout_residual_n={} heldout_under={:.6} heldout_positive={:.6} heldout_angle_ge_1deg={:.6}",
        model_images_path.display(),
        verified_pairs_path.display(),
        summary.imported_pairs,
        summary.registered_images,
        summary.registered_pairs,
        summary.invalid_correspondences,
        summary.normalized_threshold,
        summary.all.observations,
        summary.all.residual_samples,
        summary.all.mean_sampson_root,
        summary.all.median_sampson_root,
        summary.all.p90_sampson_root,
        summary.all.under_fraction,
        summary.all.positive_fraction,
        summary.all.angle_fraction,
        summary.pair_balanced_mean_sampson_root,
        summary.pair_balanced_median_sampson_root,
        summary.pair_balanced_p90_sampson_root,
        summary.pair_balanced_under_fraction,
        summary.pair_balanced_positive_fraction,
        summary.pair_balanced_angle_fraction,
        summary.pair_balanced_rotation_disagreement_deg,
        summary.rotation_reference_pairs,
        summary.pair_balanced_stable_rotation_disagreement_deg,
        summary.stable_rotation_reference_pairs,
        summary.pair_balanced_translation_disagreement_deg,
        summary.pair_balanced_translation_median_deg,
        summary.pair_balanced_translation_p90_deg,
        summary.pair_balanced_translation_huber_deg,
        summary.translation_reference_pairs,
        summary.translation_reference_coverage,
        model_cross_validation_selection_score(summary).unwrap_or(f64::NAN),
        summary.image_balanced_median_sampson_root,
        summary.image_balanced_under_fraction,
        summary.image_balanced_positive_fraction,
        summary.image_balanced_angle_fraction,
        summary.image_balanced_translation_disagreement_deg,
        summary.image_translation_reference_coverage,
        summary.held_out.observations,
        summary.held_out.residual_samples,
        summary.held_out.under_fraction,
        summary.held_out.positive_fraction,
        summary.held_out.angle_fraction,
    );
}

/// Express a pixel-space fundamental matrix in normalized coordinates.  For
/// `x_jᵀ F x_i = 0` and `x_norm = K⁻¹x`, the corresponding relation is
/// `x_norm,jᵀ (K_jᵀ F K_i) x_norm,i = 0`.
fn calibrated_fundamental(fundamental: &Matrix3<f64>, camera: &Camera) -> Option<Matrix3<f64>> {
    let (fx, fy, cx, cy) = camera.intrinsics()?;
    if ![fx, fy, cx, cy].iter().all(|value| value.is_finite())
        || fx.abs() < 1.0e-12
        || fy.abs() < 1.0e-12
    {
        return None;
    }
    let k = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    let calibrated = k.transpose() * fundamental * k;
    calibrated
        .iter()
        .all(|value| value.is_finite())
        .then_some(calibrated)
}

/// Project a pixel-space fundamental matrix into the calibrated essential
/// manifold by equalizing its two largest singular values and zeroing the
/// third.  This is the closest essential-manifold projection in Frobenius
/// norm for the fixed singular vectors.
fn project_fundamental_to_essential(
    fundamental: &Matrix3<f64>,
    camera: &Camera,
) -> Option<Matrix3<f64>> {
    let calibrated = calibrated_fundamental(fundamental, camera)?;
    let svd = calibrated.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let singular_values = svd.singular_values;
    let scale = 0.5 * (singular_values[0] + singular_values[1]);
    if !scale.is_finite() || scale < 1.0e-12 {
        return None;
    }
    let essential = u * Matrix3::from_diagonal(&Vector3::new(scale, scale, 0.0)) * v_t;
    essential
        .iter()
        .all(|value| value.is_finite())
        .then_some(essential)
}

/// Conservative, opt-in repair for the specific COLMAP case where the full
/// verifier selected a fundamental matrix (`UNCALIBRATED`) even though this
/// caller has valid shared intrinsics.  The verifier's F inliers are
/// recomputed from *all* candidate correspondences, then the projected
/// `E_F = Kᵀ F K` is rescored in normalized coordinates. The repair is
/// accepted only when it retains at least half of the F support, clears the
/// caller's minimum support, has an unambiguous positive-depth solution, and
/// passes the strict manifold, residual-agreement, and deterministic
/// subset-refit stability gate below.
///
/// This is deliberately separate from the diagnostics-only
/// [`fundamental_to_essential_quality`]: callers can replace their accepted
/// match set only through the explicit CLI gate, while the default path never
/// invokes this function.
#[derive(Debug, Clone)]
struct UncalibratedFToEssentialRefinement {
    essential: Matrix3<f64>,
    inlier_indices: Vec<usize>,
    f_inlier_count: usize,
    quality: EssentialPairQuality,
}

/// Direct calibrated-essential candidate used by the opt-in primary-model
/// policy. The full verifier already performs robust E RANSAC and an inlier
/// refit; this helper performs one deterministic refit on that E support and
/// rescored all candidate correspondences before admitting the result.
#[derive(Debug, Clone)]
struct CalibratedEssentialPrimarySelection {
    essential: Matrix3<f64>,
    inlier_indices: Vec<usize>,
    initial_inlier_count: usize,
    quality: EssentialPairQuality,
}

/// Select a direct-E model for a known-intrinsics F-winning pair.
///
/// The support floor follows COLMAP's minimum-inlier gate. The `0.5` E/F
/// support floor is the same conservative floor used by the existing guarded
/// F→E path, while the pose check uses the source-derived hardened cheirality
/// policy (≥1° triangulation angle, ≤0.85 ambiguity, ≥50% positive-depth
/// support). Thus an F model may win on raw support, but a weak/planar/pure-
/// rotation E candidate cannot displace it merely because calibration exists.
fn select_calibrated_essential_primary(
    report: &TwoViewGeometryReport,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    min_matches: usize,
) -> Option<CalibratedEssentialPrimarySelection> {
    if report.config != ConfigurationType::Uncalibrated || camera.intrinsics().is_none() {
        return None;
    }
    let initial_essential = *report.essential.as_ref()?;
    let initial_inlier_count = report.essential_inliers.len();
    let required_support = min_matches.max(8);
    if initial_inlier_count < required_support {
        return None;
    }
    if report.f_inlier_count == 0
        || initial_inlier_count as f64 / (report.f_inlier_count as f64) < 0.5
    {
        return None;
    }
    // A report labelled Uncalibrated already excludes a homography that is
    // close to F, but keep the degeneracy guard explicit at this policy
    // boundary so a future classifier change cannot promote a planar edge.
    if report.h_inlier_count as f64 / report.f_inlier_count as f64 >= 0.8 {
        return None;
    }

    let initial_correspondences: Vec<TwoViewCorrespondence> = report
        .essential_inliers
        .iter()
        .filter_map(|&index| correspondences.get(index).copied())
        .collect();
    if initial_correspondences.len() != initial_inlier_count {
        return None;
    }
    let refit = EightPointEssentialMatrixEstimator::default()
        .estimate(&initial_correspondences, camera)
        .unwrap_or(initial_essential);
    if !refit.iter().all(|value| value.is_finite()) {
        return None;
    }
    let normalized_threshold =
        TwoViewGeometryOptions::for_camera(camera, 4.0).essential_sampson_threshold;
    let threshold_sq = normalized_threshold * normalized_threshold;
    let inlier_indices: Vec<usize> = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            normalized_essential_squared_sampson_error(&refit, correspondence, camera)
                .is_some_and(|error| error <= threshold_sq)
                .then_some(index)
        })
        .collect();
    let retained_floor = ((initial_inlier_count as f64) * 0.8).ceil() as usize;
    if inlier_indices.len() < required_support
        || inlier_indices.len() < retained_floor
        || inlier_indices.len() as f64 / (report.f_inlier_count as f64) < 0.5
    {
        return None;
    }

    let hardened = recover_relative_pose_with_options(
        &refit,
        correspondences,
        camera,
        &inlier_indices,
        &CheiralityOptions::hardened(),
    )?;
    let quality =
        essential_pair_quality_for_inliers(&refit, &inlier_indices, correspondences, camera)?;
    // `hardened` is the acceptance authority. The explicit finite checks keep
    // diagnostics and future callers from accepting an invalid quality row.
    if hardened.best_score <= 0 || !quality.mean_sampson.is_finite() || quality.angle_samples == 0 {
        return None;
    }
    Some(CalibratedEssentialPrimarySelection {
        essential: refit,
        inlier_indices,
        initial_inlier_count,
        quality,
    })
}

/// Decide whether the opt-in strict strategy must omit an F-winning pair.
///
/// A camera without usable intrinsics is deliberately left on the historical
/// F path: strict F→E is meaningful only when the caller supplied calibration.
/// Keeping this predicate separate makes the default-off and strict-pass/fail
/// behavior directly testable without invoking the parallel verifier.
fn should_exclude_strict_uncalibrated_f_winner(
    strict: bool,
    camera: &Camera,
    report: &TwoViewGeometryReport,
    refinement: Option<&UncalibratedFToEssentialRefinement>,
) -> bool {
    strict
        && camera.intrinsics().is_some()
        && report.config == ConfigurationType::Uncalibrated
        && refinement.is_none()
}

fn refine_uncalibrated_f_winner(
    report: &TwoViewGeometryReport,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    min_matches: usize,
) -> Option<UncalibratedFToEssentialRefinement> {
    if report.config != ConfigurationType::Uncalibrated {
        return None;
    }
    let fundamental = report.fundamental.as_ref()?;
    let required_support = min_matches.max(8);
    let pixel_threshold = TwoViewGeometryOptions::for_camera(camera, 4.0).max_error_px;
    let pixel_threshold_sq = pixel_threshold * pixel_threshold;
    let f_inliers: Vec<usize> = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            (fundamental_squared_sampson_error(fundamental, correspondence) <= pixel_threshold_sq)
                .then_some(index)
        })
        .collect();
    if f_inliers.len() < required_support {
        return None;
    }

    let essential = project_fundamental_to_essential(fundamental, camera)?;
    let normalized_threshold =
        TwoViewGeometryOptions::for_camera(camera, 4.0).essential_sampson_threshold;
    let normalized_threshold_sq = normalized_threshold * normalized_threshold;
    let inlier_indices: Vec<usize> = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            (normalized_essential_squared_sampson_error(&essential, correspondence, camera)
                .is_some_and(|error| error <= normalized_threshold_sq))
            .then_some(index)
        })
        .collect();
    if inlier_indices.len() < required_support
        || (inlier_indices.len() as f64 / f_inliers.len() as f64) < 0.5
    {
        return None;
    }

    let quality =
        essential_pair_quality_for_inliers(&essential, &inlier_indices, correspondences, camera)?;
    let cheirality_ratio = quality.best_cheirality as f64 / inlier_indices.len() as f64;
    let second_over_best = if quality.best_cheirality > 0 {
        quality.second_cheirality as f64 / quality.best_cheirality as f64
    } else {
        f64::INFINITY
    };
    // A positive-depth count alone can be misleading for an almost-pure
    // rotation. Require both a strong winner and at least one valid
    // triangulation sample; the thresholds are fixed structural guards, not
    // scene/GT-derived tuning knobs.
    const MIN_CHEIRALITY_RATIO: f64 = 0.75;
    const MAX_SECOND_OVER_BEST: f64 = 0.25;
    if quality.angle_samples == 0
        || !cheirality_ratio.is_finite()
        || cheirality_ratio < MIN_CHEIRALITY_RATIO
        || !second_over_best.is_finite()
        || second_over_best > MAX_SECOND_OVER_BEST
    {
        return None;
    }

    // A numerically plausible E_F is not necessarily a calibrated F.  Keep
    // the behavioral switch deliberately strict: the conversion must stay
    // close to the essential manifold, retain nearly all of the F support,
    // preserve its normalized residuals, and give the same pose under the
    // deterministic subset refits below.  These are geometry-consistency
    // checks, not scene-specific image/stem rules.
    let diagnostics = f_to_e_candidate_diagnostics(report, correspondences, camera)?;
    if !f_to_e_stability_gate(&diagnostics) {
        return None;
    }

    Some(UncalibratedFToEssentialRefinement {
        essential,
        inlier_indices,
        f_inlier_count: f_inliers.len(),
        quality,
    })
}

/// Diagnostics used to decide whether a calibrated F→E conversion is stable
/// enough to feed into tracks.  These values are all computed without GT: the
/// singular-value fields describe the calibration-induced manifold projection,
/// the residual fields compare the two algebraic models on the same pixels,
/// and the pose-spread fields compare deterministic F refits on F inlier
/// subsets.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FToECandidateDiagnostics {
    calibrated_s1: f64,
    calibrated_s2: f64,
    calibrated_s3: f64,
    projection_distortion: f64,
    s1_s2_mismatch: f64,
    s3_s2_ratio: f64,
    f_inliers: usize,
    ef_inliers: usize,
    ef_overlap_on_f: f64,
    f_normalized_residual: f64,
    ef_normalized_residual_on_f: f64,
    ef_to_f_residual_ratio: f64,
    cheirality_ratio: f64,
    cheirality_margin: f64,
    ef_angle_p25_deg: f64,
    stable_refits: usize,
    pose_rotation_spread_deg: f64,
    pose_translation_spread_deg: f64,
}

/// Strict, GT-independent eligibility gate for replacing an F winner with
/// `E_F`.  An F that is genuinely explained by the known calibration should
/// already be close to rank-2/equal-singular essential geometry, and a stable
/// refit should not move its relative pose by several degrees.  The thresholds
/// are intentionally conservative so the opt-in path cannot broadly rewrite
/// a COLMAP-quality F graph.
fn f_to_e_stability_gate(diagnostics: &FToECandidateDiagnostics) -> bool {
    f_to_e_stability_gate_with_max_pose_spread(diagnostics, 5.0)
}

/// Sequence registration has an independent consecutive-stem constraint and
/// only uses this conversion to recover a missing pose; it does not rewrite
/// the ordinary F-winning graph.  The existing sequential-SfM quality gate
/// treats ten degrees as the upper bound for a PnP/E pose disagreement, so
/// allow that same bound here while retaining every other strict F→E check.
fn sequence_f_to_e_stability_gate(diagnostics: &FToECandidateDiagnostics) -> bool {
    f_to_e_stability_gate_with_max_pose_spread(diagnostics, 10.0)
}

/// High-support sequence-only exception for an otherwise strict F→E candidate.
///
/// A large translation spread is the sole field allowed to exceed the normal
/// sequence bound.  The exception is intentionally narrower than the ordinary
/// F→E gate: it needs at least 100 F and E-support rows, near-complete overlap,
/// an unambiguous positive-depth solution, and at least one degree of robust
/// fourth-view-free parallax.  The call with an infinite translation limit
/// still applies the ordinary manifold, residual, refit-count, finite-value,
/// and five-degree rotation-spread limits.
fn sequence_f_to_e_high_support_override_gate(diagnostics: &FToECandidateDiagnostics) -> bool {
    const MIN_SUPPORT: usize = 100;
    const MIN_EF_OVERLAP_ON_F: f64 = 0.95;
    const MIN_CHEIRALITY_RATIO: f64 = 0.95;
    // This is the same second-solution exclusion used by the guarded F→E
    // refiner (`second / best <= 0.25`), expressed as a winner margin.
    const MIN_CHEIRALITY_MARGIN: f64 = 0.75;
    const MIN_ANGLE_P25_DEG: f64 = 1.0;
    diagnostics.pose_translation_spread_deg.is_finite()
        && diagnostics.pose_translation_spread_deg > 10.0
        && diagnostics.f_inliers >= MIN_SUPPORT
        && diagnostics.ef_inliers >= MIN_SUPPORT
        && diagnostics.ef_overlap_on_f.is_finite()
        && diagnostics.ef_overlap_on_f >= MIN_EF_OVERLAP_ON_F
        && diagnostics.cheirality_ratio.is_finite()
        && diagnostics.cheirality_ratio >= MIN_CHEIRALITY_RATIO
        && diagnostics.cheirality_margin.is_finite()
        && diagnostics.cheirality_margin >= MIN_CHEIRALITY_MARGIN
        && diagnostics.ef_angle_p25_deg.is_finite()
        && diagnostics.ef_angle_p25_deg >= MIN_ANGLE_P25_DEG
        && f_to_e_stability_gate_with_max_pose_spread(diagnostics, f64::INFINITY)
}

fn f_to_e_stability_gate_with_max_pose_spread(
    diagnostics: &FToECandidateDiagnostics,
    max_pose_spread_deg: f64,
) -> bool {
    const MAX_MANIFOLD_DISTORTION: f64 = 0.01;
    const MAX_S1_S2_MISMATCH: f64 = 0.02;
    const MAX_S3_S2_RATIO: f64 = 0.05;
    const MIN_EF_OVERLAP_ON_F: f64 = 0.90;
    const MAX_EF_TO_F_RESIDUAL_RATIO: f64 = 3.0;
    const MAX_POSE_SPREAD_DEG: f64 = 5.0;
    const MIN_STABLE_REFITS: usize = 2;

    let residual_agrees = (diagnostics.ef_to_f_residual_ratio.is_finite()
        && diagnostics.ef_to_f_residual_ratio <= MAX_EF_TO_F_RESIDUAL_RATIO)
        || (diagnostics.f_normalized_residual.is_finite()
            && diagnostics.ef_normalized_residual_on_f.is_finite()
            && diagnostics.f_normalized_residual <= 1.0e-8
            && diagnostics.ef_normalized_residual_on_f <= 1.0e-8);
    diagnostics.calibrated_s1.is_finite()
        && diagnostics.calibrated_s1 > 1.0e-12
        && diagnostics.calibrated_s2.is_finite()
        && diagnostics.calibrated_s2 > 1.0e-12
        && diagnostics.calibrated_s3.is_finite()
        && diagnostics.calibrated_s3 >= 0.0
        && diagnostics.projection_distortion.is_finite()
        && diagnostics.projection_distortion <= MAX_MANIFOLD_DISTORTION
        && diagnostics.s1_s2_mismatch.is_finite()
        && diagnostics.s1_s2_mismatch <= MAX_S1_S2_MISMATCH
        && diagnostics.s3_s2_ratio.is_finite()
        && diagnostics.s3_s2_ratio <= MAX_S3_S2_RATIO
        && diagnostics.f_inliers >= 8
        && diagnostics.ef_inliers >= 8
        && diagnostics.ef_overlap_on_f.is_finite()
        && diagnostics.ef_overlap_on_f >= MIN_EF_OVERLAP_ON_F
        && residual_agrees
        && diagnostics.cheirality_ratio.is_finite()
        && diagnostics.cheirality_margin.is_finite()
        && diagnostics.stable_refits >= MIN_STABLE_REFITS
        && diagnostics.pose_rotation_spread_deg.is_finite()
        && diagnostics.pose_rotation_spread_deg <= max_pose_spread_deg.min(MAX_POSE_SPREAD_DEG)
        && diagnostics.pose_translation_spread_deg.is_finite()
        && diagnostics.pose_translation_spread_deg <= max_pose_spread_deg
}

fn quaternion_delta_deg(a: &UnitQuaternion<f64>, b: &UnitQuaternion<f64>) -> f64 {
    (a.inverse() * *b).angle().abs().to_degrees()
}

fn translation_direction_delta_deg(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let Some(a) = a.try_normalize(1.0e-12) else {
        return f64::NAN;
    };
    let Some(b) = b.try_normalize(1.0e-12) else {
        return f64::NAN;
    };
    a.dot(&b).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Return whether an imported calibrated-E reference is sufficiently
/// constrained to act as a diagnostic pose reference.  The thresholds mirror
/// the existing hardened cheirality policy (50% positive, one-degree
/// parallax, 15% winner margin) and add a conservative cap on deterministic
/// subset/refit translation spread.  This is report-only; it never gates the
/// normal mapper.
fn imported_reference_quality_is_strong(
    quality: &EssentialPairQuality,
    stable_refits: usize,
    translation_spread_deg: f64,
) -> bool {
    const MIN_CHEIRALITY_RATIO: f64 = 0.5;
    const MIN_CHEIRALITY_MARGIN: f64 = 0.15;
    const MIN_ANGLE_P25_DEG: f64 = 1.0;
    const MIN_STABLE_REFITS: usize = 2;
    const MAX_TRANSLATION_SPREAD_DEG: f64 = 20.0;
    quality.cheirality_ratio.is_finite()
        && quality.cheirality_ratio >= MIN_CHEIRALITY_RATIO
        && quality.cheirality_margin().is_finite()
        && quality.cheirality_margin() >= MIN_CHEIRALITY_MARGIN
        && quality.angle_p25_deg.is_finite()
        && quality.angle_p25_deg >= MIN_ANGLE_P25_DEG
        && stable_refits >= MIN_STABLE_REFITS
        && translation_spread_deg.is_finite()
        && translation_spread_deg <= MAX_TRANSLATION_SPREAD_DEG
}

impl EssentialPairQuality {
    fn cheirality_margin(&self) -> f64 {
        if self.best_cheirality <= 0 {
            0.0
        } else {
            (self.best_cheirality - self.second_cheirality) as f64 / self.best_cheirality as f64
        }
    }
}

/// Re-estimate an imported calibrated-E reference on three deterministic
/// subsets.  Every refit uses the same hardened cheirality selector, so the
/// sign of the camera-centre direction is resolved geometrically rather than
/// by taking an absolute dot product.  The result is a diagnostic stability
/// measure, not a replacement for the imported E.
fn imported_reference_pose_stability(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> (usize, f64, f64) {
    if correspondences.len() < 8 {
        return (0, f64::NAN, f64::NAN);
    }
    let full_indices: Vec<usize> = (0..correspondences.len()).collect();
    let Some(full_quality) = essential_pair_quality_for_inliers_with_options(
        essential,
        &full_indices,
        correspondences,
        camera,
        &CheiralityOptions::hardened(),
    ) else {
        return (0, f64::NAN, f64::NAN);
    };
    let full_rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        full_quality.rotation_quaternion[0],
        full_quality.rotation_quaternion[1],
        full_quality.rotation_quaternion[2],
        full_quality.rotation_quaternion[3],
    ));
    let full_center = Vector3::from_row_slice(&full_quality.center_direction);
    let subset_size = if correspondences.len() > 8 {
        correspondences
            .len()
            .min(64)
            .min(correspondences.len() - 1)
            .max(8)
    } else {
        correspondences.len()
    };
    let prefix = (0..subset_size).collect::<Vec<_>>();
    let suffix = (correspondences.len() - subset_size..correspondences.len()).collect::<Vec<_>>();
    let stride = correspondences.len().div_ceil(subset_size);
    let evenly_spaced = (0..correspondences.len())
        .step_by(stride.max(1))
        .take(subset_size)
        .collect::<Vec<_>>();

    let mut valid = 0usize;
    let mut max_rotation = 0.0f64;
    let mut max_translation = 0.0f64;
    for subset in [prefix, suffix, evenly_spaced] {
        if subset.len() < 8 {
            continue;
        }
        let subset_correspondences = subset
            .iter()
            .filter_map(|&index| correspondences.get(index).copied())
            .collect::<Vec<_>>();
        if subset_correspondences.len() < 8 {
            continue;
        }
        let Some(refit_essential) =
            EightPointEssentialMatrixEstimator::default().estimate(&subset_correspondences, camera)
        else {
            continue;
        };
        let refit_indices: Vec<usize> = (0..subset_correspondences.len()).collect();
        let Some(refit_recovery) = recover_relative_pose_with_options(
            &refit_essential,
            &subset_correspondences,
            camera,
            &refit_indices,
            &CheiralityOptions::hardened(),
        ) else {
            continue;
        };
        let refit_center = -refit_recovery
            .rotation
            .inverse()
            .transform_vector(&refit_recovery.translation_unit);
        let rotation_delta = quaternion_delta_deg(&full_rotation, &refit_recovery.rotation);
        let translation_delta = translation_direction_delta_deg(&full_center, &refit_center);
        if !rotation_delta.is_finite() || !translation_delta.is_finite() {
            continue;
        }
        valid += 1;
        max_rotation = max_rotation.max(rotation_delta);
        max_translation = max_translation.max(translation_delta);
    }
    (valid, max_rotation, max_translation)
}

/// Extract a strong, calibrated imported-E reference and its deterministic
/// stability diagnostics.  The accepted match order is the replay-file order
/// and is independent of the candidate mapper traversal.
fn imported_essential_reference_quality(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<ImportedEssentialReferenceQuality> {
    let indices: Vec<usize> = (0..correspondences.len()).collect();
    let quality = essential_pair_quality_for_inliers_with_options(
        essential,
        &indices,
        correspondences,
        camera,
        &CheiralityOptions::hardened(),
    )?;
    let (stable_refits, rotation_spread_deg, translation_spread_deg) =
        imported_reference_pose_stability(essential, correspondences, camera);
    if !imported_reference_quality_is_strong(&quality, stable_refits, translation_spread_deg) {
        return None;
    }
    let center_direction = Vector3::from_row_slice(&quality.center_direction);
    if !center_direction.iter().all(|value| value.is_finite()) {
        return None;
    }
    let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        quality.rotation_quaternion[0],
        quality.rotation_quaternion[1],
        quality.rotation_quaternion[2],
        quality.rotation_quaternion[3],
    ));
    Some(ImportedEssentialReferenceQuality {
        rotation,
        center_direction,
        cheirality_margin: quality.cheirality_margin(),
        angle_p25_deg: quality.angle_p25_deg,
        stable_refits,
        rotation_spread_deg,
        translation_spread_deg,
    })
}

/// Refit F on three deterministic subsets of its inliers and compare the
/// resulting calibrated poses to the full F→E pose.  The subset construction
/// is deliberately fixed (prefix, suffix, evenly-spaced) so a run can be
/// reproduced byte-for-byte and does not consume random state.
fn f_to_e_pose_stability(
    essential: &Matrix3<f64>,
    f_inliers: &[usize],
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> (usize, f64, f64) {
    if f_inliers.len() < 8 {
        return (0, f64::NAN, f64::NAN);
    }
    let Some(full_recovery) = recover_relative_pose_with_options(
        essential,
        correspondences,
        camera,
        f_inliers,
        &CheiralityOptions::default(),
    ) else {
        return (0, f64::NAN, f64::NAN);
    };
    let full_center = -full_recovery
        .rotation
        .inverse()
        .transform_vector(&full_recovery.translation_unit);
    let subset_size = if f_inliers.len() > 8 {
        f_inliers.len().min(64).min(f_inliers.len() - 1).max(8)
    } else {
        f_inliers.len()
    };
    let prefix = f_inliers
        .iter()
        .take(subset_size)
        .copied()
        .collect::<Vec<_>>();
    let suffix = f_inliers
        .iter()
        .rev()
        .take(subset_size)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let stride = f_inliers.len().div_ceil(subset_size);
    let evenly_spaced = f_inliers
        .iter()
        .step_by(stride.max(1))
        .take(subset_size)
        .copied()
        .collect::<Vec<_>>();
    let subsets = [prefix, suffix, evenly_spaced];

    let mut valid = 0usize;
    let mut max_rotation = 0.0f64;
    let mut max_translation = 0.0f64;
    for subset in subsets {
        if subset.len() < 8 {
            continue;
        }
        let subset_corrs = subset
            .iter()
            .filter_map(|&index| correspondences.get(index).copied())
            .collect::<Vec<_>>();
        if subset_corrs.len() < 8 {
            continue;
        }
        let Some(refit_f) = estimate_fundamental_dlt(&subset_corrs) else {
            continue;
        };
        let Some(refit_e) = project_fundamental_to_essential(&refit_f, camera) else {
            continue;
        };
        let Some(refit_recovery) = recover_relative_pose_with_options(
            &refit_e,
            correspondences,
            camera,
            &subset,
            &CheiralityOptions::default(),
        ) else {
            continue;
        };
        let rotation_delta =
            quaternion_delta_deg(&full_recovery.rotation, &refit_recovery.rotation);
        let refit_center = -refit_recovery
            .rotation
            .inverse()
            .transform_vector(&refit_recovery.translation_unit);
        let translation_delta = translation_direction_delta_deg(&full_center, &refit_center);
        if !rotation_delta.is_finite() || !translation_delta.is_finite() {
            continue;
        }
        valid += 1;
        max_rotation = max_rotation.max(rotation_delta);
        max_translation = max_translation.max(translation_delta);
    }
    (valid, max_rotation, max_translation)
}

/// Compute all GT-independent F→E candidate diagnostics.  Returning `None`
/// means that the pixel F or its calibrated representation was invalid; the
/// caller logs that case separately and keeps the legacy F result.
fn f_to_e_candidate_diagnostics(
    report: &TwoViewGeometryReport,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<FToECandidateDiagnostics> {
    if report.config != ConfigurationType::Uncalibrated {
        return None;
    }
    let fundamental = report.fundamental.as_ref()?;
    let calibrated = calibrated_fundamental(fundamental, camera)?;
    let singular_values = calibrated.svd(false, false).singular_values;
    let calibrated_norm = calibrated.norm();
    let essential = project_fundamental_to_essential(fundamental, camera)?;
    let projection_distortion = if calibrated_norm > 1.0e-12 {
        (calibrated - essential).norm() / calibrated_norm
    } else {
        f64::NAN
    };
    let s1_s2_mismatch = if singular_values[0] + singular_values[1] > 1.0e-12 {
        (singular_values[0] - singular_values[1]).abs()
            / (0.5 * (singular_values[0] + singular_values[1]))
    } else {
        f64::NAN
    };
    let s3_s2_ratio = if singular_values[1] > 1.0e-12 {
        singular_values[2] / singular_values[1]
    } else {
        f64::NAN
    };
    let pixel_threshold = TwoViewGeometryOptions::for_camera(camera, 4.0).max_error_px;
    let f_threshold_sq = pixel_threshold * pixel_threshold;
    let f_inliers = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            (fundamental_squared_sampson_error(fundamental, correspondence) <= f_threshold_sq)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let normalized_threshold =
        TwoViewGeometryOptions::for_camera(camera, 4.0).essential_sampson_threshold;
    let normalized_threshold_sq = normalized_threshold * normalized_threshold;
    let ef_inliers = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            normalized_essential_squared_sampson_error(&essential, correspondence, camera)
                .is_some_and(|error| error <= normalized_threshold_sq)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let ef_set = ef_inliers.iter().copied().collect::<HashSet<_>>();
    let ef_overlap_on_f = if f_inliers.is_empty() {
        f64::NAN
    } else {
        f_inliers
            .iter()
            .filter(|index| ef_set.contains(index))
            .count() as f64
            / f_inliers.len() as f64
    };
    let f_normalized_residual =
        mean_normalized_essential_sampson_error(&calibrated, correspondences, camera, &f_inliers);
    let ef_normalized_residual_on_f =
        mean_normalized_essential_sampson_error(&essential, correspondences, camera, &f_inliers);
    let ef_to_f_residual_ratio = if f_normalized_residual > 1.0e-12 {
        ef_normalized_residual_on_f / f_normalized_residual
    } else {
        f64::NAN
    };
    let quality =
        essential_pair_quality_for_inliers(&essential, &ef_inliers, correspondences, camera);
    let ef_angle_p25_deg = quality
        .as_ref()
        .map(|quality| quality.angle_p25_deg)
        .unwrap_or(f64::NAN);
    let (cheirality_ratio, cheirality_margin) = if let Some(quality) = quality.as_ref() {
        let ratio = if ef_inliers.is_empty() {
            f64::NAN
        } else {
            quality.best_cheirality as f64 / ef_inliers.len() as f64
        };
        let margin = if quality.best_cheirality > 0 {
            (quality.best_cheirality - quality.second_cheirality) as f64
                / quality.best_cheirality as f64
        } else {
            f64::NAN
        };
        (ratio, margin)
    } else {
        (f64::NAN, f64::NAN)
    };
    let (stable_refits, pose_rotation_spread_deg, pose_translation_spread_deg) =
        f_to_e_pose_stability(&essential, &f_inliers, correspondences, camera);
    Some(FToECandidateDiagnostics {
        calibrated_s1: singular_values[0],
        calibrated_s2: singular_values[1],
        calibrated_s3: singular_values[2],
        projection_distortion,
        s1_s2_mismatch,
        s3_s2_ratio,
        f_inliers: f_inliers.len(),
        ef_inliers: ef_inliers.len(),
        ef_overlap_on_f,
        f_normalized_residual,
        ef_normalized_residual_on_f,
        ef_to_f_residual_ratio,
        cheirality_ratio,
        cheirality_margin,
        ef_angle_p25_deg,
        stable_refits,
        pose_rotation_spread_deg,
        pose_translation_spread_deg,
    })
}

#[derive(Debug, Clone)]
struct FundamentalToEssentialQuality {
    f_inliers: usize,
    ef_inliers: usize,
    f_mean_sampson_px: f64,
    direct_mean_sampson_on_f: f64,
    ef_mean_sampson_on_f: f64,
    ef_mean_sampson_on_direct: f64,
    ef_mean_sampson: f64,
    ef_quality: Option<EssentialPairQuality>,
}

/// Recompute the F inlier set from the report's refined F, then evaluate the
/// calibrated E obtained from that F.  The report does not expose F indices,
/// so recomputation is necessary and is deterministic.  This never feeds the
/// derived model back into verification or mapping; it is a diagnostics-only
/// comparison of the two calibrated pose hypotheses.
fn fundamental_to_essential_quality(
    report: &TwoViewGeometryReport,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<FundamentalToEssentialQuality> {
    let fundamental = report.fundamental.as_ref()?;
    let pixel_threshold = TwoViewGeometryOptions::for_camera(camera, 4.0).max_error_px;
    let pixel_threshold_sq = pixel_threshold * pixel_threshold;
    let f_inliers: Vec<usize> = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            (fundamental_squared_sampson_error(fundamental, correspondence) <= pixel_threshold_sq)
                .then_some(index)
        })
        .collect();
    if f_inliers.len() < 8 {
        return Some(FundamentalToEssentialQuality {
            f_inliers: f_inliers.len(),
            ef_inliers: 0,
            f_mean_sampson_px: f64::NAN,
            direct_mean_sampson_on_f: f64::NAN,
            ef_mean_sampson_on_f: f64::NAN,
            ef_mean_sampson_on_direct: f64::NAN,
            ef_mean_sampson: f64::NAN,
            ef_quality: None,
        });
    }
    let essential = project_fundamental_to_essential(fundamental, camera)?;
    let normalized_threshold =
        TwoViewGeometryOptions::for_camera(camera, 4.0).essential_sampson_threshold;
    let normalized_threshold_sq = normalized_threshold * normalized_threshold;
    let ef_inliers: Vec<usize> = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            (normalized_essential_squared_sampson_error(&essential, correspondence, camera)
                .is_some_and(|error| error <= normalized_threshold_sq))
            .then_some(index)
        })
        .collect();
    let f_mean_sampson_px = {
        let mut total = 0.0;
        let mut count = 0usize;
        for &index in &f_inliers {
            let error = fundamental_squared_sampson_error(fundamental, &correspondences[index]);
            if error.is_finite() {
                total += error.sqrt();
                count += 1;
            }
        }
        if count == 0 {
            f64::NAN
        } else {
            total / count as f64
        }
    };
    Some(FundamentalToEssentialQuality {
        f_inliers: f_inliers.len(),
        ef_inliers: ef_inliers.len(),
        f_mean_sampson_px,
        direct_mean_sampson_on_f: report
            .essential
            .as_ref()
            .map(|essential| {
                mean_normalized_essential_sampson_error(
                    essential,
                    correspondences,
                    camera,
                    &f_inliers,
                )
            })
            .unwrap_or(f64::NAN),
        ef_mean_sampson_on_f: mean_normalized_essential_sampson_error(
            &essential,
            correspondences,
            camera,
            &f_inliers,
        ),
        ef_mean_sampson_on_direct: mean_normalized_essential_sampson_error(
            &essential,
            correspondences,
            camera,
            &report.essential_inliers,
        ),
        ef_mean_sampson: mean_normalized_essential_sampson_error(
            &essential,
            correspondences,
            camera,
            &ef_inliers,
        ),
        ef_quality: essential_pair_quality_for_inliers(
            &essential,
            &ef_inliers,
            correspondences,
            camera,
        ),
    })
}

fn format_essential_pair_quality(quality: Option<EssentialPairQuality>) -> String {
    let Some(q) = quality else {
        return " cheirality_best=NA cheirality_second=NA cheirality_ratio=NA cheirality_second_over_best=NA sampson_mean=NA pose_q=NA center_dir=NA angle_samples=0 angle_ge_1deg=0 angle_p10_deg=NA angle_p25_deg=NA angle_median_deg=NA depth_ratio_p10=NA depth_ratio_p25=NA depth_ratio_median=NA".to_owned();
    };
    let second_over_best = if q.best_cheirality > 0 {
        q.second_cheirality as f64 / q.best_cheirality as f64
    } else {
        f64::NAN
    };
    format!(
        " cheirality_best={} cheirality_second={} cheirality_ratio={:.6} cheirality_second_over_best={:.6} sampson_mean={:.8} pose_q={:.9},{:.9},{:.9},{:.9} center_dir={:.9},{:.9},{:.9} angle_samples={} angle_ge_1deg={} angle_p10_deg={:.6} angle_p25_deg={:.6} angle_median_deg={:.6} depth_ratio_p10={:.6} depth_ratio_p25={:.6} depth_ratio_median={:.6}",
        q.best_cheirality,
        q.second_cheirality,
        q.cheirality_ratio,
        second_over_best,
        q.mean_sampson,
        q.rotation_quaternion[0],
        q.rotation_quaternion[1],
        q.rotation_quaternion[2],
        q.rotation_quaternion[3],
        q.center_direction[0],
        q.center_direction[1],
        q.center_direction[2],
        q.angle_samples,
        q.angle_ge_1deg,
        q.angle_p10_deg,
        q.angle_p25_deg,
        q.angle_median_deg,
        q.depth_ratio_p10,
        q.depth_ratio_p25,
        q.depth_ratio_median,
    )
}

fn format_fundamental_to_essential_quality(
    quality: Option<FundamentalToEssentialQuality>,
) -> String {
    let Some(q) = quality else {
        return " f2e_f_inliers=NA f2e_ef_inliers=NA f2e_f_mean_sampson_px=NA f2e_direct_mean_sampson_on_f=NA f2e_ef_mean_sampson_on_f=NA f2e_ef_mean_sampson_on_direct=NA f2e_ef_mean_sampson=NA f2e_ef_cheirality_best=NA f2e_ef_cheirality_second=NA f2e_ef_cheirality_ratio=NA f2e_ef_pose_q=NA f2e_ef_center_dir=NA f2e_ef_angle_p10_deg=NA f2e_ef_angle_p25_deg=NA f2e_ef_angle_median_deg=NA f2e_ef_depth_ratio_p10=NA f2e_ef_sampson_quality=NA".to_owned();
    };
    let (best, second, ratio) = q
        .ef_quality
        .as_ref()
        .map(|value| {
            (
                value.best_cheirality.to_string(),
                value.second_cheirality.to_string(),
                format!("{:.6}", value.cheirality_ratio),
            )
        })
        .unwrap_or_else(|| ("NA".to_owned(), "NA".to_owned(), "NA".to_owned()));
    format!(
        " f2e_f_inliers={} f2e_ef_inliers={} f2e_f_mean_sampson_px={:.6} f2e_direct_mean_sampson_on_f={:.8} f2e_ef_mean_sampson_on_f={:.8} f2e_ef_mean_sampson_on_direct={:.8} f2e_ef_mean_sampson={:.8} f2e_ef_cheirality_best={} f2e_ef_cheirality_second={} f2e_ef_cheirality_ratio={} f2e_ef_pose_q={:.9},{:.9},{:.9},{:.9} f2e_ef_center_dir={:.9},{:.9},{:.9} f2e_ef_angle_p10_deg={:.6} f2e_ef_angle_p25_deg={:.6} f2e_ef_angle_median_deg={:.6} f2e_ef_depth_ratio_p10={:.6} f2e_ef_sampson_quality={}",
        q.f_inliers,
        q.ef_inliers,
        q.f_mean_sampson_px,
        q.direct_mean_sampson_on_f,
        q.ef_mean_sampson_on_f,
        q.ef_mean_sampson_on_direct,
        q.ef_mean_sampson,
        best,
        second,
        ratio,
        q.ef_quality
            .as_ref()
            .map(|value| value.rotation_quaternion[0])
            .unwrap_or(f64::NAN),
        q.ef_quality
            .as_ref()
            .map(|value| value.rotation_quaternion[1])
            .unwrap_or(f64::NAN),
        q.ef_quality
            .as_ref()
            .map(|value| value.rotation_quaternion[2])
            .unwrap_or(f64::NAN),
        q.ef_quality
            .as_ref()
            .map(|value| value.rotation_quaternion[3])
            .unwrap_or(f64::NAN),
        q.ef_quality
            .as_ref()
            .map(|value| value.center_direction[0])
            .unwrap_or(f64::NAN),
        q.ef_quality
            .as_ref()
            .map(|value| value.center_direction[1])
            .unwrap_or(f64::NAN),
        q.ef_quality
            .as_ref()
            .map(|value| value.center_direction[2])
            .unwrap_or(f64::NAN),
        q.ef_quality
            .as_ref()
            .map(|value| value.angle_p10_deg)
            .unwrap_or(f64::NAN),
        q.ef_quality
            .as_ref()
            .map(|value| value.angle_p25_deg)
            .unwrap_or(f64::NAN),
        q.ef_quality
            .as_ref()
            .map(|value| value.angle_median_deg)
            .unwrap_or(f64::NAN),
        q.ef_quality
            .as_ref()
            .map(|value| value.depth_ratio_p10)
            .unwrap_or(f64::NAN),
        q.ef_quality.is_some(),
    )
}

fn diagnose_imported_raw(
    features: &[FeatureSet],
    camera: &Camera,
    verifier: &TwoViewGeometryVerifier,
    i: usize,
    j: usize,
    raw_matches: &[(usize, usize)],
) -> DiagnoseImportedRaw {
    let valid_matches: Vec<(usize, usize)> = raw_matches
        .iter()
        .copied()
        .filter(|&(query_index, train_index)| {
            features[i].keypoints.get(query_index).is_some()
                && features[j].keypoints.get(train_index).is_some()
        })
        .collect();
    let correspondences: Vec<TwoViewCorrespondence> = valid_matches
        .iter()
        .map(|&(query_index, train_index)| {
            TwoViewCorrespondence::new(
                features[i].keypoints[query_index],
                features[j].keypoints[train_index],
            )
        })
        .collect();
    let report = (correspondences.len() >= 8).then(|| verifier.classify(&correspondences, camera));
    DiagnoseImportedRaw {
        valid_matches,
        report,
    }
}

#[derive(Debug, Clone, Copy)]
struct DiagnosePairRow {
    ratio: f32,
    cross_check: bool,
    raw_matches: usize,
    valid_matches: usize,
    config: ConfigurationType,
    accepted_inliers: usize,
    e_inliers: usize,
    f_inliers: usize,
    h_inliers: usize,
    colmap_pair_present: bool,
    colmap_raw_matches: usize,
    colmap_index_overlap: usize,
    colmap_verified_present: bool,
    colmap_verified_inliers: usize,
    colmap_verified_config: Option<ConfigurationType>,
    imported_config: Option<ConfigurationType>,
    imported_accepted_inliers: usize,
    imported_e_inliers: usize,
    imported_f_inliers: usize,
    imported_h_inliers: usize,
    imported_accepted_index_overlap: usize,
}

fn configuration_name(config: ConfigurationType) -> &'static str {
    match config {
        ConfigurationType::Undefined => "UNDEFINED",
        ConfigurationType::Degenerate => "DEGENERATE",
        ConfigurationType::Uncalibrated => "UNCALIBRATED",
        ConfigurationType::Calibrated => "CALIBRATED",
        ConfigurationType::Planar => "PLANAR",
        ConfigurationType::Panoramic => "PANORAMIC",
        ConfigurationType::PlanarOrPanoramic => "PLANAR_OR_PANORAMIC",
        ConfigurationType::Watermark => "WATERMARK",
        ConfigurationType::Multiple => "MULTIPLE",
    }
}

fn diagnose_pair_row(
    features: &[FeatureSet],
    camera: &Camera,
    verifier: &TwoViewGeometryVerifier,
    matcher: &PairMatcher,
    i: usize,
    j: usize,
    ratio: f32,
    cross_check: bool,
    colmap_matches: Option<&HashMap<(usize, usize), Vec<(usize, usize)>>>,
    colmap_verified: Option<&HashMap<(usize, usize), VerifiedPairOracle>>,
    imported_raw: Option<&DiagnoseImportedRaw>,
) -> DiagnosePairRow {
    let profile = diagnose_pair_profile(
        features,
        camera,
        verifier,
        matcher,
        i,
        j,
        ratio,
        cross_check,
    );
    let (config, accepted_inliers, e_inliers, f_inliers, h_inliers) = profile
        .report
        .as_ref()
        .map(|report| {
            (
                report.config,
                report.inliers.len(),
                report.e_inlier_count,
                report.f_inlier_count,
                report.h_inlier_count,
            )
        })
        .unwrap_or((ConfigurationType::Undefined, 0, 0, 0, 0));

    let colmap = colmap_matches.and_then(|matches| matches.get(&(i, j)));
    let colmap_set: HashSet<(usize, usize)> = colmap
        .into_iter()
        .flat_map(|matches| matches.iter().copied())
        .collect();
    let visloc_set: HashSet<(usize, usize)> = profile
        .matches
        .iter()
        .map(|m| (m.query_index, m.train_index))
        .collect();
    let colmap_index_overlap = visloc_set.intersection(&colmap_set).count();
    let verified = colmap_verified.and_then(|matches| matches.get(&(i, j)));
    let (
        imported_config,
        imported_accepted_inliers,
        imported_e_inliers,
        imported_f_inliers,
        imported_h_inliers,
    ) = imported_raw
        .and_then(|raw| raw.report.as_ref())
        .map(|report| {
            (
                Some(report.config),
                report.inliers.len(),
                report.e_inlier_count,
                report.f_inlier_count,
                report.h_inlier_count,
            )
        })
        .unwrap_or((None, 0, 0, 0, 0));
    let profile_accepted_set: HashSet<(usize, usize)> = profile
        .report
        .as_ref()
        .map(|report| {
            report
                .inliers
                .iter()
                .filter_map(|&idx| profile.valid_matches.get(idx).copied())
                .collect()
        })
        .unwrap_or_default();
    let imported_accepted_set: HashSet<(usize, usize)> = imported_raw
        .map(|raw| {
            raw.report
                .as_ref()
                .map(|report| {
                    report
                        .inliers
                        .iter()
                        .filter_map(|&idx| raw.valid_matches.get(idx).copied())
                        .collect()
                })
                .unwrap_or_default()
        })
        .unwrap_or_default();
    let imported_accepted_index_overlap = profile_accepted_set
        .intersection(&imported_accepted_set)
        .count();

    DiagnosePairRow {
        ratio,
        cross_check,
        raw_matches: profile.matches.len(),
        valid_matches: profile.correspondences.len(),
        config,
        accepted_inliers,
        e_inliers,
        f_inliers,
        h_inliers,
        colmap_pair_present: colmap.is_some(),
        colmap_raw_matches: colmap.map_or(0, |matches| matches.len()),
        colmap_index_overlap,
        colmap_verified_present: verified.is_some(),
        colmap_verified_inliers: verified.map_or(0, |oracle| oracle.inliers),
        colmap_verified_config: verified.map(|oracle| oracle.config),
        imported_config,
        imported_accepted_inliers,
        imported_e_inliers,
        imported_f_inliers,
        imported_h_inliers,
        imported_accepted_index_overlap,
    }
}

fn diagnose_pairs_for_csv(
    features: &[FeatureSet],
    image_names: &[String],
    args: &Args,
) -> Result<Vec<(usize, usize)>, String> {
    if args.diagnose_pair_stems.is_empty() {
        let generated = if let Some(path) = args.candidate_manifest.as_deref() {
            parse_candidate_manifest(path, image_names)?
        } else {
            candidate_pairs(features, image_names, args)?
        };
        return filter_pairs_by_stem_window(generated, image_names, args.pair_stem_window);
    }
    filter_pairs_by_stem_window(
        all_pairs(features.len())
            .into_iter()
            .filter(|&(i, j)| {
                args.diagnose_pair_stems.iter().any(|stem| {
                    image_stem(&image_names[i]) == stem || image_stem(&image_names[j]) == stem
                })
            })
            .collect(),
        image_names,
        args.pair_stem_window,
    )
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn write_diagnose_pairs_csv(
    path: &Path,
    features: &[FeatureSet],
    image_names: &[String],
    pairs: &[(usize, usize)],
    camera: &Camera,
    matcher: &PairMatcher,
    colmap_matches: Option<&HashMap<(usize, usize), Vec<(usize, usize)>>>,
    colmap_verified: Option<&HashMap<(usize, usize), VerifiedPairOracle>>,
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(std::fs::File::create(path)?);
    writeln!(
        writer,
        "image_i,image_j,image_name_i,image_name_j,kp_i,kp_j,ratio,cross_check,raw_matches,valid_matches,config,accepted_inliers,e_inliers,f_inliers,h_inliers,colmap_pair_present,colmap_raw_matches,colmap_index_overlap,colmap_verified_present,colmap_verified_inliers,colmap_verified_config,imported_config,imported_accepted_inliers,imported_e_inliers,imported_f_inliers,imported_h_inliers,imported_accepted_index_overlap"
    )?;
    let verifier = TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(camera, 4.0));
    let mut rows = 0usize;
    for &(i, j) in pairs {
        let imported_raw = colmap_matches
            .and_then(|matches| matches.get(&(i, j)))
            .map(|matches| diagnose_imported_raw(features, camera, &verifier, i, j, matches));
        for &(ratio, cross_check) in &DIAGNOSE_PROFILES {
            let row = diagnose_pair_row(
                features,
                camera,
                &verifier,
                matcher,
                i,
                j,
                ratio,
                cross_check,
                colmap_matches,
                colmap_verified,
                imported_raw.as_ref(),
            );
            writeln!(
                writer,
                "{i},{j},{},{},{},{},{:.2},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                csv_escape(&image_names[i]),
                csv_escape(&image_names[j]),
                features[i].keypoints.len(),
                features[j].keypoints.len(),
                row.ratio,
                if row.cross_check { 1 } else { 0 },
                row.raw_matches,
                row.valid_matches,
                configuration_name(row.config),
                row.accepted_inliers,
                row.e_inliers,
                row.f_inliers,
                row.h_inliers,
                if row.colmap_pair_present { 1 } else { 0 },
                row.colmap_raw_matches,
                row.colmap_index_overlap,
                if row.colmap_verified_present { 1 } else { 0 },
                row.colmap_verified_inliers,
                row.colmap_verified_config
                    .map(configuration_name)
                    .unwrap_or("NONE"),
                row.imported_config
                    .map(configuration_name)
                    .unwrap_or("NONE"),
                row.imported_accepted_inliers,
                row.imported_e_inliers,
                row.imported_f_inliers,
                row.imported_h_inliers,
                row.imported_accepted_index_overlap,
            )?;
            rows += 1;
        }
    }
    writer.flush()?;
    Ok(rows)
}

/// M5 diagnosis tool (`--diagnose-pair I,J`): dump raw match counts and
/// [`TwoViewGeometryVerifier`] outcomes for one specific `(i, j)` image pair
/// across the same fixed battery used by `--diagnose-pairs-csv`.
fn diagnose_pair(
    features: &[FeatureSet],
    camera: &Camera,
    matcher: &PairMatcher,
    i: usize,
    j: usize,
) {
    println!("=== diagnose-pair ({i}, {j}) ===");
    let verifier = TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(camera, 4.0));
    for &(ratio, cross_check) in &DIAGNOSE_PROFILES {
        let profile = diagnose_pair_profile(
            features,
            camera,
            &verifier,
            matcher,
            i,
            j,
            ratio,
            cross_check,
        );
        let Some(report) = profile.report.as_ref() else {
            let detail = if profile.matches.len() < 8 {
                "too few to classify"
            } else {
                "invalid keypoint index"
            };
            println!(
                "  ratio={ratio:.2} cross_check={cross_check:<5} raw_matches={:<5} ({detail})",
                profile.matches.len()
            );
            continue;
        };
        // Recover a translation direction from E inliers when present, for
        // façade chirality diagnosis against GT.
        let mut pose_note = String::new();
        if report.e_inlier_count >= 8 {
            let e_corrs: Vec<TwoViewCorrespondence> = report
                .essential_inliers
                .iter()
                .filter_map(|&idx| profile.correspondences.get(idx).copied())
                .collect();
            if let Some(rel) = RelativePoseEstimator::default().estimate(&e_corrs, camera) {
                let t = rel.previous_to_current.translation;
                let r = rel.previous_to_current.rotation;
                if let Some(d) = (-r.inverse().transform_vector(&t)).try_normalize(1e-12) {
                    pose_note = format!(" E_dir=[{:.3},{:.3},{:.3}]", d.x, d.y, d.z);
                }
            }
        }
        println!(
            "  ratio={ratio:.2} cross_check={cross_check:<5} raw_matches={:<5} config={:?} \
             inliers={} (E={} F={} H={}){pose_note}",
            profile.matches.len(),
            report.config,
            report.inliers.len(),
            report.e_inlier_count,
            report.f_inlier_count,
            report.h_inlier_count,
        );
    }
}

fn gt_poses_aligned(image_names: &[String], by_stem: &HashMap<String, Pose>) -> Vec<Option<Pose>> {
    image_names
        .iter()
        .map(|name| {
            Path::new(name)
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|stem| by_stem.get(stem).cloned())
        })
        .collect()
}

/// Compare essential bearings against GT centres (`--diagnose-bearing-gt`).
fn diagnose_bearing_vs_gt(
    label: &str,
    pairwise: &[PairwiseMatches],
    features: &[FeatureSet],
    camera: &Camera,
    image_names: &[String],
    gt_by_stem: &HashMap<String, Pose>,
    stem_filter: &HashSet<&str>,
) {
    let stem_of = |idx: usize| -> &str {
        Path::new(&image_names[idx])
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(image_names[idx].as_str())
    };
    let mut rows: Vec<(String, usize, f64, f64, f64, bool)> = Vec::new();
    for pair in pairwise {
        if !stem_filter.is_empty()
            && !stem_filter.contains(stem_of(pair.image_i))
            && !stem_filter.contains(stem_of(pair.image_j))
        {
            continue;
        }
        let e_count = pair
            .essential_matches
            .as_ref()
            .map(|e| e.len())
            .unwrap_or(0);
        // Some verifier winners keep an essential matrix for diagnostics but
        // do not retain an E-inlier index list when F/H won the accepted
        // configuration.  In that case use the accepted correspondence list
        // for pose decomposition instead of silently dropping the edge.
        if e_count < 8 && pair.essential_matrix.is_none() {
            continue;
        }
        let Some(gti) = gt_by_stem.get(stem_of(pair.image_i)) else {
            continue;
        };
        let Some(gtj) = gt_by_stem.get(stem_of(pair.image_j)) else {
            continue;
        };
        let Some(gt_in_i) = gt_bearing_in_prior_frame(gti, gtj) else {
            continue;
        };
        let corrs = pair_correspondences(pair, features, true);
        if corrs.len() < 8 {
            continue;
        }
        let rel = if let Some(essential) = pair.essential_matrix.as_ref() {
            relative_pose_from_essential(essential, &corrs, camera)
        } else {
            RelativePoseEstimator::default().estimate(&corrs, camera)
        };
        let Some(rel) = rel else {
            continue;
        };
        let r = rel.previous_to_current.rotation;
        let t = rel.previous_to_current.translation;
        let gt_rel = gtj.world_to_camera.compose(&gti.world_to_camera.inverse());
        let rotation_error = (r.inverse() * gt_rel.rotation).angle().to_degrees();
        let Some(est) = (-r.inverse().transform_vector(&t)).try_normalize(1e-12) else {
            continue;
        };
        let err_pri = bearing_alignment_error_deg(&est, &gt_in_i);
        let (err_alt, alt_wins) = if let Some((r_alt, t_alt)) = rel.alternate.as_ref() {
            let t_a = t_alt * rel.translation_scale;
            let d_alt: Vector3<f64> = (-r_alt.inverse().transform_vector(&t_a))
                .try_normalize(1e-12)
                .unwrap_or(est);
            let err = bearing_alignment_error_deg(&d_alt, &gt_in_i);
            (err, err + 1e-3 < err_pri)
        } else {
            (f64::NAN, false)
        };
        rows.push((
            format!("{}-{}", stem_of(pair.image_i), stem_of(pair.image_j)),
            e_count,
            err_pri,
            err_alt,
            rotation_error,
            alt_wins,
        ));
    }
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    println!(
        "=== diagnose-bearing-gt ({label}) {} pair(s) ===",
        rows.len()
    );
    let mut alt_wins = 0usize;
    let mut sum_pri = 0.0f64;
    let mut sum_rotation = 0.0f64;
    for (name, e, pri, alt, rotation, wins) in &rows {
        if *wins {
            alt_wins += 1;
        }
        sum_pri += pri;
        sum_rotation += rotation;
        let alt_s = if alt.is_finite() {
            format!(" alt={alt:.1}°{}", if *wins { " *" } else { "" })
        } else {
            String::new()
        };
        println!("  {name} E={e} R={rotation:.1}° pri={pri:.1}°{alt_s}");
    }
    if !rows.is_empty() {
        println!(
            "  summary: mean_R={:.1}° mean_pri={:.1}° alt_would_help={}/{}",
            sum_rotation / rows.len() as f64,
            sum_pri / rows.len() as f64,
            alt_wins,
            rows.len()
        );
    }
}

/// Dump a GT-independent rotation-cycle diagnostic for the verified view graph.
///
/// A pairwise essential estimate is only a local constraint; a wrong
/// façade/chirality solution can still pass its own RANSAC.  For every
/// available triangle `(i,j,k)`, compare `R_jk * R_ij` with `R_ik` and attach
/// the resulting cycle error to all three edges.  This is deliberately an
/// environment-gated diagnostic rather than a mapper policy: the latter
/// needs an explicit A/B threshold and must not silently change legacy tracks.
fn dump_rotation_cycle_diagnostics(
    pairwise: &[PairwiseMatches],
    features: &[FeatureSet],
    camera: &Camera,
    image_names: &[String],
) {
    let mut rotations: HashMap<(usize, usize), UnitQuaternion<f64>> = HashMap::new();
    for pair in pairwise {
        let Some(essential) = pair.essential_matrix.as_ref() else {
            continue;
        };
        let corrs = pair_correspondences(pair, features, true);
        let Some(relative) = relative_pose_from_essential(essential, &corrs, camera) else {
            continue;
        };
        let (i, j, rotation) = if pair.image_i < pair.image_j {
            (
                pair.image_i,
                pair.image_j,
                relative.previous_to_current.rotation,
            )
        } else {
            (
                pair.image_j,
                pair.image_i,
                relative.previous_to_current.rotation.inverse(),
            )
        };
        rotations.insert((i, j), rotation);
    }
    let mut edge_errors: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
    let n = features.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let Some(r_ij) = rotations.get(&(i, j)).copied() else {
                continue;
            };
            for k in (j + 1)..n {
                let (Some(r_jk), Some(r_ik)) = (
                    rotations.get(&(j, k)).copied(),
                    rotations.get(&(i, k)).copied(),
                ) else {
                    continue;
                };
                let predicted = r_jk * r_ij;
                let error_deg = (predicted.inverse() * r_ik).angle().to_degrees();
                if !error_deg.is_finite() {
                    continue;
                }
                edge_errors.entry((i, j)).or_default().push(error_deg);
                edge_errors.entry((j, k)).or_default().push(error_deg);
                edge_errors.entry((i, k)).or_default().push(error_deg);
            }
        }
    }
    let stem = |idx: usize| {
        Path::new(&image_names[idx])
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(image_names[idx].as_str())
            .to_owned()
    };
    let mut rows: Vec<((usize, usize), Vec<f64>)> = edge_errors.into_iter().collect();
    for (_, errors) in &mut rows {
        errors.sort_by(f64::total_cmp);
    }
    rows.sort_by(|a, b| {
        b.1.get(b.1.len() / 2)
            .unwrap_or(&f64::NEG_INFINITY)
            .total_cmp(a.1.get(a.1.len() / 2).unwrap_or(&f64::NEG_INFINITY))
    });
    eprintln!(
        "sfm-debug: rotation cycles edges={} essential_rotations={} triangles={}",
        rows.len(),
        rotations.len(),
        rows.iter().map(|(_, errors)| errors.len()).sum::<usize>() / 3,
    );
    for ((i, j), errors) in rows.iter().take(30) {
        let median = errors[errors.len() / 2];
        let p90 = errors[((errors.len() * 9).saturating_sub(1) / 10).min(errors.len() - 1)];
        let max = *errors.last().unwrap_or(&f64::NAN);
        eprintln!(
            "sfm-debug: rotation-cycle {}-{} triangles={} median={median:.1}deg p90={p90:.1}deg max={max:.1}deg",
            stem(*i),
            stem(*j),
            errors.len(),
        );
    }
    eprintln!("sfm-debug: rotation-cycle lowest-consistency edges:");
    for ((i, j), errors) in rows.iter().rev().take(20) {
        let median = errors[errors.len() / 2];
        let p90 = errors[((errors.len() * 9).saturating_sub(1) / 10).min(errors.len() - 1)];
        eprintln!(
            "sfm-debug: rotation-cycle {}-{} triangles={} median={median:.1}deg p90={p90:.1}deg",
            stem(*i),
            stem(*j),
            errors.len(),
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\nsee the file header for usage.");
            std::process::exit(2);
        }
    };
    if args.feature_extractor == FeatureExtractorKind::Files
        && args.features_dir.as_os_str().is_empty()
    {
        return Err(
            "--features-dir is required (or use --feature-extractor sift with --images-dir)".into(),
        );
    }
    if args.sift_stream_export {
        #[cfg(feature = "image-io")]
        {
            let dir = args.images_dir.as_deref().ok_or(
                "--sift-stream-export requires --images-dir with --feature-extractor sift",
            )?;
            let output_dir = args
                .export_features_dir
                .as_deref()
                .ok_or("--sift-stream-export requires --export-features-dir DIR")?;
            let total_keypoints = stream_export_images_with_sift(
                dir,
                output_dir,
                args.input_colmap_calibration.as_deref(),
                args.sift_max_keypoints,
                args.sift_affine,
                &args.sift_detector,
                args.sift_multi_anisotropy,
                args.sift_dsp,
                args.sift_dsp_num_scales,
                args.sift_l1_root,
                args.sift_max_orientations,
                args.sift_standard_orientations,
                args.sift_prefer_larger_scale,
                args.sift_full_pyramid,
                args.sift_contrast_threshold,
                args.sift_descriptor_magnification,
                args.sift_scale_adaptive_gradients,
                args.sift_vlfeat_compatible_descriptor,
                args.sift_vlfeat_compatible_detector,
                args.sift_vlfeat_bilinear_orientations,
                args.sift_vlfeat_compatible_output_order,
                args.sift_colmap_compatible_grayscale,
                args.sift_split_colmap_detector_grayscale,
                args.sift_append_descriptor_magnification,
                &args.sift_extra_keypoints_stems,
                args.sift_extra_keypoints,
                args.sift_extra_contrast_threshold,
                args.sift_stream_resume,
            )?;
            println!(
                "streaming SIFT export complete: {} image(s), {} keypoints -> {}",
                list_sift_image_paths(dir)?.len(),
                total_keypoints,
                output_dir.display(),
            );
            return Ok(());
        }
        #[cfg(not(feature = "image-io"))]
        {
            return Err("--sift-stream-export requires building with --features image-io".into());
        }
    }
    let mut snapshot_feature_paths: Option<Vec<PathBuf>> = None;
    let mut snapshot_feature_fingerprints: Option<Vec<SnapshotFeatureFileFingerprint>> = None;
    let (
        mut features,
        image_names,
        primary_keypoint_counts,
        mut alternate_descriptors,
        mut locus_metadata,
    ) = match args.feature_extractor {
        FeatureExtractorKind::Files => {
            let (features, image_names, locus_metadata) = if args.snapshot_keypoints_only {
                let loaded = load_images_keypoints_only(
                    &args.features_dir,
                    &args.feature_suffix,
                    &args.image_suffix,
                )?;
                snapshot_feature_paths = Some(loaded.paths);
                snapshot_feature_fingerprints = Some(loaded.fingerprints);
                (loaded.features, loaded.image_names, loaded.locus_metadata)
            } else {
                load_images(&args.features_dir, &args.feature_suffix, &args.image_suffix)?
            };
            let primary_keypoint_counts = features.iter().map(FeatureSet::len).collect();
            let alternate_descriptors = vec![None; features.len()];
            (
                features,
                image_names,
                primary_keypoint_counts,
                alternate_descriptors,
                locus_metadata,
            )
        }
        FeatureExtractorKind::Sift => {
            let dir = args.images_dir.clone().unwrap_or_else(|| {
                eprintln!("error: --feature-extractor sift requires --images-dir");
                std::process::exit(2);
            });
            load_images_with_sift(
                &dir,
                args.sift_max_keypoints,
                args.sift_affine,
                &args.sift_detector,
                args.sift_multi_anisotropy,
                args.sift_dsp,
                args.sift_dsp_num_scales,
                args.sift_l1_root,
                args.sift_max_orientations,
                args.sift_standard_orientations,
                args.sift_prefer_larger_scale,
                args.sift_full_pyramid,
                args.sift_contrast_threshold,
                args.sift_descriptor_magnification,
                args.sift_scale_adaptive_gradients,
                args.sift_vlfeat_compatible_descriptor,
                args.sift_vlfeat_compatible_detector,
                args.sift_vlfeat_bilinear_orientations,
                args.sift_vlfeat_compatible_output_order,
                args.sift_colmap_compatible_grayscale,
                args.sift_split_colmap_detector_grayscale,
                args.sift_append_descriptor_magnification,
                &args.sift_extra_keypoints_stems,
                args.sift_extra_keypoints,
                args.sift_extra_contrast_threshold,
            )?
        }
    };
    log_process_memory("example-after-feature-load");
    // A per-image COLMAP calibration is represented internally by converting
    // each native pixel to the first image's pinhole convention.  This keeps
    // descriptor/index identity and the established mapper API intact while
    // making every normalized ray use its own focal length/principal point.
    // Retain only native feature pixels for the multi-camera COLMAP export.
    // Descriptors are unchanged by calibration and remain in `features`.
    let mut per_image_calibration: Option<LoadedPerImageCalibration> = None;
    let mut native_keypoints_for_export: Option<Vec<Vec<Point2<f64>>>> = None;
    if let Some(model_dir) = args.input_colmap_calibration.as_deref() {
        let loaded = load_input_colmap_calibration(
            model_dir,
            &image_names,
            &features,
            args.images_dir.as_deref(),
        )?;
        if !loaded.rig.has_shared_geometry() {
            native_keypoints_for_export = Some(
                features
                    .iter()
                    .map(|feature_set| feature_set.keypoints.clone())
                    .collect(),
            );
        }
        loaded.rig.canonicalize_features_in_place(&mut features)?;
        args.camera = loaded.rig.reference_camera().clone();
        println!(
            "per-image calibration: {} image cameras, {} unique camera definitions, reference CAMERA_ID={} (internal ray canonicalization; intrinsics fixed)",
            loaded.rig.len(),
            loaded
                .native_cameras
                .iter()
                .map(|camera| camera.id)
                .collect::<HashSet<_>>()
                .len(),
            args.camera.id,
        );
        per_image_calibration = Some(loaded);
        log_process_memory("example-after-calibration-canonicalization");
    }
    let snapshot_feature_validation = if args.snapshot_keypoints_only {
        let paths = snapshot_feature_paths
            .as_deref()
            .ok_or("--snapshot-keypoints-only did not retain feature source paths")?;
        let fingerprints = snapshot_feature_fingerprints
            .as_deref()
            .ok_or("--snapshot-keypoints-only did not retain feature source fingerprints")?;
        let validation = snapshot_feature_validation_from_files(paths, &features, fingerprints)
            .map_err(std::io::Error::other)?;
        println!(
            "snapshot keypoints-only replay: retained {} keypoint sets; descriptor payloads re-read one file at a time (feature-manifest-fnv1a64={:016x})",
            features.len(),
            validation.feature_manifest_hash,
        );
        log_process_memory("example-after-keypoints-only-feature-fingerprint");
        Some(validation)
    } else {
        None
    };
    let config_snapshot = effective_config_snapshot(&args);
    println!(
        "effective-config: fnv1a64={:016x} {config_snapshot}",
        effective_config_hash(&config_snapshot)
    );
    // A locus-aware run needs stable physical row IDs as well as stable
    // representative endpoints; otherwise union-find's root tie-break still
    // observes the source orientation-row order.  Metadata-free legacy files
    // deliberately skip this implicit reorder, so the opt-in remains a true
    // no-op for old dumps.
    let canonicalize_locus_feature_order =
        args.orientation_locus_canonicalization && locus_metadata.iter().any(Option::is_some);
    let canonical_feature_index_map =
        if args.canonical_feature_order || canonicalize_locus_feature_order {
            let map = canonicalize_feature_order(&mut features, &mut alternate_descriptors)?;
            if let Some(native_keypoints) = native_keypoints_for_export.as_mut() {
                remap_feature_keypoints_by_old_to_new(native_keypoints, &map)?;
            }
            remap_locus_metadata(&mut locus_metadata, &map).map_err(|error| error.to_string())?;
            println!(
                "feature order: canonical physical key ({} image(s){})",
                map.len(),
                if canonicalize_locus_feature_order && !args.canonical_feature_order {
                    ", locus-aware"
                } else {
                    ""
                }
            );
            Some(map)
        } else {
            None
        };
    if features.len() < 2 {
        return Err(format!("need ≥2 images, found {}", features.len()).into());
    }
    if let Some((i, j)) = args.seed_pair {
        if i >= features.len() || j >= features.len() {
            return Err(format!(
                "--seed-pair {i},{j} is outside the loaded image range 0..{}",
                features.len()
            )
            .into());
        }
    }
    if let Some(window) = args.pair_stem_window {
        // Fail before any descriptor vocabulary or matcher work if the
        // sequence naming contract is not satisfied.
        numeric_stem_values(&image_names)?;
        println!(
            "pair stem window enabled: |stem_i-stem_j| <= {window} (unique numeric suffixes validated)"
        );
    }
    let sequence_stem_values = if args.sequence_relative_pose_fallback {
        let values = numeric_stem_values(&image_names)?;
        println!(
            "sequence relative-pose fallback enabled: unique numeric stems validated ({} images)",
            values.len()
        );
        Some(values)
    } else {
        None
    };
    let initial_poses = if let Some(path) = args.initial_poses_file.as_deref() {
        let poses = if let Some(calibration) = per_image_calibration.as_ref() {
            initial_poses_from_colmap_images_txt_with_expected_cameras(
                path,
                &image_names,
                &args.camera,
                Some(&calibration.native_cameras),
            )?
        } else {
            initial_poses_from_colmap_images_txt(path, &image_names, &args.camera)?
        };
        println!(
            "initial poses: {} / {} image poses imported from {:?}; fixed during initial growth",
            poses.iter().filter(|pose| pose.is_some()).count(),
            poses.len(),
            path,
        );
        Some(poses)
    } else {
        None
    };
    let total_kp: usize = features.iter().map(|f| f.keypoints.len()).sum();
    println!(
        "loaded {} images, {} keypoints total, camera {}x{}",
        features.len(),
        total_kp,
        args.camera.width,
        args.camera.height,
    );

    if let Some(dir) = &args.export_features_dir {
        if let Some(native_keypoints) = native_keypoints_for_export.as_ref() {
            export_features_to_dir_with_native_keypoints(
                dir,
                &image_names,
                &features,
                native_keypoints,
                &locus_metadata,
            )?;
        } else {
            export_features_to_dir(dir, &image_names, &features, &locus_metadata)?;
        }
        println!(
            "export features: {} file(s) -> {}",
            image_names.len(),
            dir.display()
        );
        if args.export_features_only {
            return Ok(());
        }
    }

    // A completed-model cross-validation probe is deliberately before matcher
    // construction and all reconstruction decisions.  It consumes the full
    // imported verified correspondence multiset (including matches that the
    // mapper later drops during track conflict resolution), then exits.
    if let Some(model_path) = args.diagnose_model_score_file.as_deref() {
        let verified_path = args
            .import_verified_pairs_file
            .as_deref()
            .ok_or("--diagnose-model-score requires --import-verified-pairs-file")?;
        let mut imported = parse_imported_verified_pairs_file(verified_path, &image_names)?;
        if let Some(map) = canonical_feature_index_map.as_ref() {
            remap_imported_verified_pairs(&mut imported, map).map_err(|error| error.to_string())?;
        }
        imported = filter_imported_verified_pairs_by_stem_window(
            imported,
            &image_names,
            args.pair_stem_window,
        )?;
        let summary = score_model_against_verified_pairs(
            model_path,
            &imported,
            &features,
            &image_names,
            &args.camera,
        )?;
        print_model_cross_validation_summary(&summary, model_path, verified_path, &image_names);
        return Ok(());
    }

    // Snapshot import is deliberately resolved before matcher/candidate
    // construction.  Validation covers the loaded image/feature manifests,
    // camera, pair order, and correspondence hashes; once it succeeds the
    // mapper receives the stored stream verbatim and no descriptor matcher or
    // verifier is consulted.
    let mut imported_snapshot = if let Some(path) = args.import_verified_pairs_snapshot.as_deref() {
        let snapshot = verified_pair_snapshot::read(path).map_err(std::io::Error::other)?;
        let pairwise = validate_snapshot_for_run(
            &snapshot,
            &image_names,
            &features,
            &args.camera,
            snapshot_feature_validation.as_ref(),
        )
        .map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid verified-pair snapshot {}: {error}", path.display()),
            )
        })?;
        println!(
            "import verified-pair snapshot: {} pairs, {} accepted correspondences from {} (matching/verifier bypassed; ordered-edge-fnv1a64={:016x}, unordered-edge-fnv1a64={:016x})",
            pairwise.len(),
            pairwise.iter().map(|pair| pair.matches.len()).sum::<usize>(),
            path.display(),
            ordered_pairwise_edge_hash(&pairwise),
            unordered_pairwise_edge_hash(&pairwise),
        );
        let stats = verification_stats_from_snapshot(&snapshot)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        // Snapshot metadata is needed only when the caller explicitly asks to
        // re-export a snapshot.  Ordinary replay feeds PairwiseMatches to the
        // mapper and must not retain the lossless raw-match/index streams.
        let metadata = if args.export_verified_pairs_snapshot.is_some() {
            snapshot_metadata_map_from_snapshot(&snapshot)
        } else {
            HashMap::new()
        };
        // `validate_snapshot_for_run` has completed all manifest, camera,
        // configuration, and pair-order/hash checks.  PairwiseMatches owns
        // the mapper stream now; release the decoded snapshot before any
        // candidate/matcher/mapper state is built.
        drop(snapshot);
        log_process_memory("example-after-snapshot-release");
        Some((pairwise, stats, metadata))
    } else {
        None
    };
    let snapshot_imported = imported_snapshot.is_some();
    log_process_memory("example-after-snapshot-import");

    // Coordinate overrides are intentionally applied only after the immutable
    // snapshot has validated against the base features.  The replacement
    // directory is checked row-for-row and descriptor-bit-for-bit, so the
    // imported pair stream remains an exact topology/model control: only the
    // pixels used for track triangulation/BA change.
    if let Some(override_dir) = args.snapshot_coordinate_override_dir.as_deref() {
        let Some((imported_pairs, _, _)) = imported_snapshot.as_ref() else {
            return Err(
                "--snapshot-coordinate-override-dir requires a validated snapshot import".into(),
            );
        };
        let (override_features, override_names, _) =
            load_images(override_dir, &args.feature_suffix, &args.image_suffix)?;
        let stats = apply_snapshot_coordinate_override(
            &mut features,
            &image_names,
            &override_features,
            &override_names,
        )
        .map_err(std::io::Error::other)?;
        let ordered_hash = ordered_pairwise_edge_hash(imported_pairs);
        let unordered_hash = unordered_pairwise_edge_hash(imported_pairs);
        println!(
            "snapshot coordinate override: {} image(s), {} row(s), {} coordinate row(s) changed from {}; descriptors/index rows validated bitwise; ordered-edge-fnv1a64={ordered_hash:016x} unordered-edge-fnv1a64={unordered_hash:016x} (unchanged)",
            stats.images,
            stats.rows,
            stats.changed_rows,
            override_dir.display(),
        );
    }

    // A COLMAP point-membership oracle is loaded only after the feature
    // manifest (and any validated coordinate override) is final.  Its XYZ,
    // camera poses, colors, and reprojection errors are intentionally never
    // passed to the mapper.
    let colmap_track_membership = if let Some(path) =
        args.diagnose_colmap_track_membership.as_deref()
    {
        let membership = parse_colmap_track_membership(path, &image_names, &features)?;
        println!(
            "COLMAP track-membership oracle: source_points={} source_observations={} retained_tracks={} retained_observations={} skipped_conflicting_points={} skipped_conflicting_observations={} (XYZ/poses ignored; fresh triangulation)",
            membership.source_points,
            membership.source_observations,
            membership.tracks.len(),
            membership.retained_observations,
            membership.skipped_conflicting_points,
            membership.skipped_conflicting_observations,
        );
        Some(membership)
    } else {
        None
    };

    // Descriptor-ensemble diagnostics need the same matcher instance as the
    // reconstruction path. Build it only for the opt-in ensemble so ordinary
    // diagnostics keep their historical no-model/no-extra-work behavior.
    let mut alternate_descriptors = Some(alternate_descriptors);
    let mut prebuilt_pair_matcher =
        if !snapshot_imported && args.sift_append_descriptor_magnification.is_some() {
            Some(build_matcher(
                &args,
                &primary_keypoint_counts,
                alternate_descriptors.take().unwrap(),
            )?)
        } else {
            None
        };

    validate_diagnose_options(
        args.diagnose_pairs_csv.as_deref(),
        &args.diagnose_pair_stems,
        &args.diagnose_pairs,
        Some(features.len()),
    )?;
    validate_diagnose_stems(&image_names, &args.diagnose_pair_stems)?;

    if let Some(path) = &args.diagnose_pairs_csv {
        if args.import_matches_file.is_some() && args.import_matches_supplement_file.is_some() {
            return Err(
                "use only one of --import-matches-file or --import-matches-supplement-file \
                 with --diagnose-pairs-csv"
                    .to_string()
                    .into(),
            );
        }
        let imported_matches = if let Some(import_path) = &args.import_matches_file {
            let mut matches = parse_imported_matches_file(import_path, &image_names)?;
            if let Some(map) = canonical_feature_index_map.as_ref() {
                remap_imported_matches(&mut matches, map).map_err(|error| error.to_string())?;
            }
            println!(
                "diagnose CSV: COLMAP raw matches {} pair(s) from {}",
                matches.len(),
                import_path.display()
            );
            Some(matches)
        } else if let Some(import_path) = &args.import_matches_supplement_file {
            let mut matches = parse_imported_matches_file(import_path, &image_names)?;
            if let Some(map) = canonical_feature_index_map.as_ref() {
                remap_imported_matches(&mut matches, map).map_err(|error| error.to_string())?;
            }
            println!(
                "diagnose CSV: COLMAP raw matches {} pair(s) from {}",
                matches.len(),
                import_path.display()
            );
            Some(matches)
        } else {
            None
        };
        let imported_verified = if let Some(import_path) = &args.import_verified_pairs_file {
            let mut imported = parse_imported_verified_pairs_file(import_path, &image_names)?;
            if let Some(map) = canonical_feature_index_map.as_ref() {
                remap_imported_verified_pairs(&mut imported, map)
                    .map_err(|error| error.to_string())?;
            }
            let imported_before_window = imported.len();
            imported = filter_imported_verified_pairs_by_stem_window(
                imported,
                &image_names,
                args.pair_stem_window,
            )?;
            if let Some(window) = args.pair_stem_window {
                println!(
                    "diagnose CSV: pair stem window |stem_i-stem_j| <= {window} retained {} / {} imported verified pairs",
                    imported.len(),
                    imported_before_window,
                );
            }
            println!(
                "diagnose CSV: COLMAP verified pairs {} from {}",
                imported.len(),
                import_path.display()
            );
            Some(verified_pair_oracle_map(&imported))
        } else {
            None
        };
        let default_diagnose_matcher = PairMatcher::Nn;
        let diagnose_matcher = prebuilt_pair_matcher
            .as_ref()
            .unwrap_or(&default_diagnose_matcher);
        let pairs = diagnose_pairs_for_csv(&features, &image_names, &args)?;
        let rows = write_diagnose_pairs_csv(
            path,
            &features,
            &image_names,
            &pairs,
            &args.camera,
            diagnose_matcher,
            imported_matches.as_ref(),
            imported_verified.as_ref(),
        )?;
        println!(
            "diagnose CSV: {} pair(s) × {} profiles = {} row(s) -> {}",
            pairs.len(),
            DIAGNOSE_PROFILES.len(),
            rows,
            path.display()
        );
        return Ok(());
    }

    if !args.diagnose_pairs.is_empty() {
        let default_diagnose_matcher = PairMatcher::Nn;
        let diagnose_matcher = prebuilt_pair_matcher
            .as_ref()
            .unwrap_or(&default_diagnose_matcher);
        for &(i, j) in &args.diagnose_pairs {
            diagnose_pair(&features, &args.camera, diagnose_matcher, i, j);
        }
        return Ok(());
    }

    // M6 (`docs/colmap_port_plan.md`): built once, up front, so a bad
    // `--matcher lightglue` invocation (missing feature / missing model
    // path) fails immediately rather than after the (potentially expensive)
    // candidate-pair generation step below.
    let pair_matcher = match prebuilt_pair_matcher.take() {
        Some(matcher) => matcher,
        None if snapshot_imported => PairMatcher::Nn,
        None => build_matcher(
            &args,
            &primary_keypoint_counts,
            alternate_descriptors.take().unwrap(),
        )?,
    };
    println!(
        "pair matcher: {}",
        if snapshot_imported {
            "snapshot (matching/verifier bypassed)"
        } else {
            match args.matcher {
                MatcherKind::Nn if args.sift_append_descriptor_magnification.is_some() => {
                    if args.sift_extra_matches_append_only {
                        "nn (primary-preserving extras + descriptor ensemble append-only)"
                    } else {
                        "nn (descriptor ensemble append-only)"
                    }
                }
                MatcherKind::Nn if args.sift_extra_matches_append_only => {
                    "nn (NN + Lowe ratio, primary-preserving append-only extras)"
                }
                MatcherKind::Nn => "nn (NN + Lowe ratio)",
                MatcherKind::LightGlue => "lightglue (learned joint matcher, ONNX)",
            }
        },
    );

    if let Some(plan_path) = args.persistent_match_worker_plan.as_deref() {
        let plan = parse_persistent_match_worker_plan(plan_path).map_err(std::io::Error::other)?;
        // The worker returns immediately after matching and never exports a
        // model or features. Release mapper/export-only calibration state
        // before allocating the first shard result; on multi-camera inputs
        // the retained native keypoint copy alone is tens of MiB.
        drop(native_keypoints_for_export.take());
        drop(per_image_calibration.take());
        drop(locus_metadata);
        drop(canonical_feature_index_map);
        drop(initial_poses);
        drop(imported_snapshot);
        trim_process_allocator();
        let feature_validation = SnapshotFeatureValidation {
            feature_counts: features.iter().map(FeatureSet::len).collect(),
            feature_manifest_hash: snapshot_feature_manifest_hash(&features),
        };
        println!(
            "persistent match worker: {} shard(s), {} candidate pairs, feature-manifest-fnv1a64={:016x}",
            plan.shards.len(),
            plan.pair_count,
            feature_validation.feature_manifest_hash,
        );
        run_persistent_match_worker(
            &plan,
            &features,
            &image_names,
            &args.camera,
            &pair_matcher,
            &args,
            &feature_validation,
        )?;
        return Ok(());
    }

    if let Some(path) = args.export_candidate_manifest.as_deref() {
        let generated = candidate_pairs(&features, &image_names, &args)?;
        let generated =
            filter_pairs_by_stem_window(generated, &image_names, args.pair_stem_window)?;
        let metadata = candidate_manifest_metadata(&args);
        write_candidate_manifest_with_metadata(path, &image_names, &generated, &metadata)?;
        println!(
            "candidate manifest: exported {} pairs for {} images to {}",
            generated.len(),
            image_names.len(),
            path.display()
        );
        return Ok(());
    }
    let mut all_candidates = if snapshot_imported {
        Vec::new()
    } else if let Some(path) = args.candidate_manifest.as_deref() {
        parse_candidate_manifest(path, &image_names)?
    } else {
        candidate_pairs(&features, &image_names, &args)?
    };
    if !snapshot_imported && args.sequence_relative_pose_fallback {
        let before = all_candidates.len();
        let added = append_consecutive_stem_candidates(&mut all_candidates, &image_names)?;
        if added > 0 {
            println!(
                "sequence relative-pose fallback: appended {added} missing consecutive candidate pair(s) ({before} -> {} before stem filtering)",
                all_candidates.len(),
            );
        }
    }
    let candidate_count_before_window = all_candidates.len();
    let candidates = if snapshot_imported {
        Vec::new()
    } else {
        filter_pairs_by_stem_window(all_candidates, &image_names, args.pair_stem_window)?
    };
    if let Some(window) = args.pair_stem_window {
        println!(
            "pair stem window: |stem_i-stem_j| <= {window} retained {} / {} candidate pairs",
            candidates.len(),
            candidate_count_before_window,
        );
    }
    println!(
        "view graph: {} candidate pairs ({})",
        candidates.len(),
        if args.candidate_manifest.is_some() {
            "candidate manifest"
        } else if args.exhaustive {
            if args.pair_stem_window.is_some() {
                "exhaustive + stem window"
            } else {
                "exhaustive"
            }
        } else {
            match args.pair_source {
                PairSource::Vlad => "VLAD top-k",
                PairSource::VladMutual => "VLAD mutual top-k",
                PairSource::VladUnion => {
                    if args.rig_local_grouping {
                        "rig-local stem + VLAD union"
                    } else {
                        "local stem + VLAD union"
                    }
                }
                PairSource::TemporalPyramid => "temporal pyramid + VLAD fill",
                PairSource::VocabTree => "vocab-tree",
                PairSource::Transitive => "transitive (vocab-tree base)",
            }
        },
    );

    let (mut pairwise, mut verification_stats, mut snapshot_metadata) = if snapshot_imported {
        imported_snapshot
            .take()
            .expect("snapshot_imported is true only when the import state exists")
    } else if let Some(path) = &args.import_verified_pairs_file {
        if args.import_matches_file.is_some() || args.import_matches_supplement_file.is_some() {
            return Err(
                    "use only one of --import-matches-file, --import-matches-supplement-file, or --import-verified-pairs-file".into(),
                );
        }
        let mut imported = parse_imported_verified_pairs_file(path, &image_names)?;
        if let Some(map) = canonical_feature_index_map.as_ref() {
            remap_imported_verified_pairs(&mut imported, map).map_err(|error| error.to_string())?;
        }
        let imported_before_window = imported.len();
        imported = filter_imported_verified_pairs_by_stem_window(
            imported,
            &image_names,
            args.pair_stem_window,
        )?;
        if let Some(window) = args.pair_stem_window {
            println!(
                "pair stem window: |stem_i-stem_j| <= {window} retained {} / {} imported verified pairs",
                imported.len(),
                imported_before_window,
            );
        }
        let mut stats = VerificationStats::default();
        for p in &imported {
            stats.record(p.config);
        }
        println!(
            "import verified pairs: {} pairs from {} (verification bypassed)",
            imported.len(),
            path.display()
        );
        let pairwise = verified_pairs_to_pairwise(imported);
        let metadata = snapshot_metadata_map_from_pairs(&pairwise);
        (pairwise, stats, metadata)
    } else {
        if args.import_matches_file.is_some() && args.import_matches_supplement_file.is_some() {
            return Err(
                "use only one of --import-matches-file or --import-matches-supplement-file".into(),
            );
        }
        let imported_matches = if let Some(path) = &args.import_matches_file {
            let mut imp = parse_imported_matches_file(path, &image_names)?;
            if let Some(map) = canonical_feature_index_map.as_ref() {
                remap_imported_matches(&mut imp, map).map_err(|error| error.to_string())?;
            }
            println!(
                "import matches: {} pairs from {}",
                imp.len(),
                path.display()
            );
            Some(imp)
        } else {
            None
        };
        let imported_supplement = if let Some(path) = &args.import_matches_supplement_file {
            let mut imp = parse_imported_matches_file(path, &image_names)?;
            if let Some(map) = canonical_feature_index_map.as_ref() {
                remap_imported_matches(&mut imp, map).map_err(|error| error.to_string())?;
            }
            println!(
                "import matches supplement: {} pairs from {} (NN fallback elsewhere)",
                imp.len(),
                path.display()
            );
            Some(imp)
        } else {
            None
        };
        let imported_ref = imported_matches.as_ref();
        let supplement_ref = imported_supplement.as_ref();
        let (pairwise, stats, metadata) = verify_pairs(
            &features,
            &args.camera,
            &candidates,
            args.match_ratio,
            args.min_matches,
            args.verification_mode,
            &pair_matcher,
            true,
            args.guided_matching,
            args.multiple_models,
            args.min_e_f_inlier_ratio,
            args.calibrated_prefer_essential,
            args.refine_uncalibrated_f_to_essential,
            args.strict_uncalibrated_f_to_essential,
            args.calibrated_essential_primary,
            args.force_essential_matches,
            args.force_essential_min_ef_ratio,
            args.force_essential_min_e_inliers,
            args.force_essential_uncalibrated_only,
            None,
            None,
            imported_ref,
            supplement_ref,
            args.colmap_guided_matching,
        );
        (pairwise, stats, metadata)
    };
    log_process_memory("example-after-pairwise-materialization");
    if std::env::var_os("VISLOC_SFM_DEBUG_DUMP_ESSENTIAL_QUALITY").is_some() {
        // `verify_pairs` emitted one bounded, machine-readable quality row per
        // attempted report.  Stop before rematching/track construction so the
        // probe remains a read-only verification diagnostic.
        println!(
            "essential-quality diagnostic: {} verified pair(s); mapper skipped; strict_f2e_excluded_pairs={} strict_f2e_excluded_inliers={} calibrated_essential_promotions={}",
            pairwise.len(),
            verification_stats.strict_uncalibrated_f_to_essential_exclusions,
            verification_stats.strict_uncalibrated_f_to_essential_excluded_inliers,
            verification_stats.calibrated_essential_primary_promotions,
        );
        return Ok(());
    }
    if !snapshot_imported && !args.rematch_stems.is_empty() && !args.rematch_free_vs_priors {
        let n = rematch_stem_pairs(
            &features,
            &image_names,
            &mut pairwise,
            &args.camera,
            &args.rematch_stems,
            args.rematch_ratio,
            args.rematch_cross_check,
            args.min_matches,
            args.verification_mode,
            &pair_matcher,
            args.rematch_guided || args.guided_matching,
            args.multiple_models,
            args.min_e_f_inlier_ratio,
            args.calibrated_prefer_essential,
            args.force_essential_min_ef_ratio,
            args.force_essential_min_e_inliers,
            args.rematch_guided_max_error_px,
            args.rematch_guided_lowe_ratio,
        );
        println!(
            "rematch: stems {:?} ratio={:.2} cross_check={} improved {} pair(s)",
            args.rematch_stems, args.rematch_ratio, args.rematch_cross_check, n
        );
    }
    if !snapshot_imported && args.pair_source == PairSource::Transitive {
        let mut all_proposed: HashSet<(usize, usize)> = candidates.iter().copied().collect();
        for _ in 0..TRANSITIVE_ROUNDS {
            let extension = filter_pairs_by_stem_window(
                expand_transitive(&pairwise, &all_proposed),
                &image_names,
                args.pair_stem_window,
            )?;
            if extension.is_empty() {
                break;
            }
            println!("transitive expansion: {} new pairs", extension.len());
            extension.iter().for_each(|p| {
                all_proposed.insert(*p);
            });
            let (more, stats, more_metadata) = verify_pairs(
                &features,
                &args.camera,
                &extension,
                args.match_ratio,
                args.min_matches,
                args.verification_mode,
                &pair_matcher,
                true,
                args.guided_matching,
                args.multiple_models,
                args.min_e_f_inlier_ratio,
                args.calibrated_prefer_essential,
                args.refine_uncalibrated_f_to_essential,
                args.strict_uncalibrated_f_to_essential,
                args.calibrated_essential_primary,
                args.force_essential_matches,
                args.force_essential_min_ef_ratio,
                args.force_essential_min_e_inliers,
                args.force_essential_uncalibrated_only,
                None,
                None,
                None,
                None,
                args.colmap_guided_matching,
            );
            verification_stats.merge(&stats);
            snapshot_metadata.extend(more_metadata);
            pairwise.extend(more);
        }
    }
    let sequence_fallback_high_support_override_pair_indices = if !snapshot_imported
        && args.sequence_relative_pose_fallback
    {
        if args.sequence_constant_velocity_scale {
            println!(
                "sequence fallback: scale estimator=constant-velocity projection (positive projected scale within recent median/MAD fence)"
            );
        } else if args.sequence_relaxed_constant_velocity_scale {
            println!(
                "sequence fallback: scale estimator=constant-velocity projection (positive projected scale within broad 0.25x..4x recent-median bounds)"
            );
        }
        if args.sequence_fallback_after_post {
            println!(
                "sequence fallback: scheduling=after ordinary post-refinement registration (one provisional pose per stalled stage)"
            );
        }
        if args.sequence_fallback_carry_scale {
            println!(
                "sequence fallback: consecutive provisional scale carry enabled (reuse previous accepted baseline within broad 0.25x..4x bounds)"
            );
        }
        let promotion = promote_sequence_fundamentals_to_essentials(
            &mut pairwise,
            &snapshot_metadata,
            &features,
            &args.camera,
        );
        println!(
            "sequence fallback: promoted {} stable uncalibrated F→E edge(s) ({} high-support translation-spread override(s)) for consecutive-pose recovery (sequence-only 10° refit spread bound)",
            promotion.promoted,
            promotion.high_support_overrides,
        );
        promotion.high_support_override_pair_indices
    } else {
        Vec::new()
    };
    let verified_matches: usize = pairwise.iter().map(|p| p.matches.len()).sum();
    let attempted_pairs = if snapshot_imported {
        pairwise.len()
    } else {
        candidates.len()
    };
    println!(
        "verified {} / {} pairs, {} inlier correspondences",
        pairwise.len(),
        attempted_pairs,
        verified_matches,
    );
    // M4 diagnosis probe (docs/colmap_port_plan.md): dump the raw verified-pair
    // image-index graph so the connected-component structure can be inspected
    // directly (temporary, env-gated; not part of the milestone's shipped
    // behaviour).
    if std::env::var_os("VISLOC_SFM_DEBUG_DUMP_PAIRS").is_some() {
        for p in &pairwise {
            eprintln!(
                "sfm-debug-pairs: {} {} matches={}",
                p.image_i,
                p.image_j,
                p.matches.len()
            );
        }
    }
    if args.verification_mode == VerificationMode::Full {
        if args.multiple_models {
            println!(
                "verification: multiple_models on (keep strongest Calibrated sub-model per pair)"
            );
        }
        println!(
            "colmap-style verification: {} pairs classified (CALIBRATED={} UNCALIBRATED={} \
             PLANAR={} PANORAMIC={} PLANAR_OR_PANORAMIC={} WATERMARK={} DEGENERATE={} MULTIPLE={})",
            verification_stats.total(),
            verification_stats.calibrated,
            verification_stats.uncalibrated,
            verification_stats.planar,
            verification_stats.panoramic,
            verification_stats.planar_or_panoramic,
            verification_stats.watermark,
            verification_stats.degenerate,
            verification_stats.multiple,
        );
        if args.refine_uncalibrated_f_to_essential || args.strict_uncalibrated_f_to_essential {
            println!(
                "verification: guarded uncalibrated-F→E refinement accepted {} pair(s) \
                 (UNCALIBRATED only; manifold/support/residual/refit-stability gate)",
                verification_stats.uncalibrated_f_to_essential_refinements,
            );
        }
        if args.strict_uncalibrated_f_to_essential {
            println!(
                "verification: strict uncalibrated-F→E strategy excluded {} pair(s), {} F inlier(s) \
                 (no rotation-only edge retained)",
                verification_stats.strict_uncalibrated_f_to_essential_exclusions,
                verification_stats.strict_uncalibrated_f_to_essential_excluded_inliers,
            );
        }
        if args.calibrated_essential_primary {
            println!(
                "verification: calibrated-essential-primary promoted {} F-winning pair(s) \
                 to direct E after robust refit/rescore and hardened cheirality gates",
                verification_stats.calibrated_essential_primary_promotions,
            );
        }
    }
    if args.prefer_essential_inliers {
        println!(
            "verification: prefer-essential-inliers (global/hybrid edges use E inliers; tracks keep winning set)"
        );
    }
    if args.prefer_essential_free_endpoints {
        println!(
            "verification: prefer-essential-free-endpoints (E inliers only on edges with a free camera)"
        );
    }
    if !args.prefer_essential_stems.is_empty() {
        println!(
            "verification: prefer-essential-stems {:?}{} (E inliers on matching edges)",
            args.prefer_essential_stems,
            if args.prefer_essential_stem_clique {
                " [clique]"
            } else {
                ""
            }
        );
    }
    if !args.prefer_essential_pairs.is_empty() {
        println!(
            "verification: prefer-essential-pairs {:?} (E inliers only on these index pairs)",
            args.prefer_essential_pairs
        );
    }
    if args.require_essential_selected_edges {
        println!(
            "verification: require-essential-selected-edges (drop selected pairs without strong E)"
        );
    }
    if !args.require_essential_stems.is_empty() {
        println!(
            "verification: require-essential-stems {:?} (drop incident edges without strong E; min_e={})",
            args.require_essential_stems, args.require_essential_min_e_inliers
        );
    }
    if (args.essential_edge_weight_boost - 1.0).abs() > 1e-12 {
        println!(
            "verification: essential-edge-weight-boost={}",
            args.essential_edge_weight_boost
        );
    }
    if args.force_essential_matches {
        println!(
            "verification: force-essential-matches when E/F≥{:.2}, E≥{}{} \
             (swapped {} pairs)",
            args.force_essential_min_ef_ratio,
            args.force_essential_min_e_inliers,
            if args.force_essential_uncalibrated_only {
                ", uncalibrated-only"
            } else {
                ""
            },
            verification_stats.force_essential_swaps,
        );
    }
    if pairwise.is_empty() {
        return Err("no pair survived geometric verification — lower --min-matches?".into());
    }
    if std::env::var_os("VISLOC_SFM_DEBUG_DUMP_ROTATION_CYCLES").is_some() {
        dump_rotation_cycle_diagnostics(&pairwise, &features, &args.camera, &image_names);
    }

    // M5 (`docs/colmap_port_plan.md`): opt-in rescue-bridging pass. Runs
    // after the standard verification above, strictly additive — admitted
    // bridge pairs are appended to `pairwise`, the same list `incremental_sfm`
    // consumes below, so a successful bridge participates in track building
    // exactly like any other verified pair.
    if !snapshot_imported && args.rescue_bridging {
        let bridges = rescue_bridging(
            &features,
            &image_names,
            &args.camera,
            &pairwise,
            &args,
            &pair_matcher,
        )?;
        pairwise.extend(bridges);
    }

    if !snapshot_imported && args.orientation_locus_canonicalization {
        let locus_stats = canonicalize_pairwise_loci(
            &features,
            &locus_metadata,
            &mut pairwise,
            Some(&args.camera),
        )
        .map_err(|error| format!("orientation locus canonicalization failed: {error}"))?;
        println!(
            "orientation loci: metadata_images={} metadata_rows={} physical_loci={} collapsed_rows={} matches={} -> {} deduplicated={} changed_pairs={}",
            locus_stats.metadata_images,
            locus_stats.metadata_rows,
            locus_stats.physical_loci,
            locus_stats.collapsed_rows,
            locus_stats.input_matches,
            locus_stats.output_matches,
            locus_stats.deduplicated_matches,
            locus_stats.changed_pairs,
        );
    }

    // Keep the integrity label independent of pair/match traversal order, then
    // apply the explicit legacy-union diagnostic before any mapper consumes
    // the verified stream.  The default `Original` path is a no-op.
    let edge_hash_before = unordered_pairwise_edge_hash(&pairwise);
    if !snapshot_imported {
        apply_union_traversal_order_with_features(
            &mut pairwise,
            args.union_traversal_order,
            &features,
        );
    }
    let edge_hash_after = unordered_pairwise_edge_hash(&pairwise);
    if edge_hash_before != edge_hash_after {
        return Err(format!(
            "union traversal reordered the verified edge multiset: before={edge_hash_before:016x} after={edge_hash_after:016x}"
        )
        .into());
    }
    println!(
        "union traversal: order={} unordered-edge-fnv1a64={edge_hash_after:016x}",
        args.union_traversal_order.as_string(),
    );

    // Reuse the existing COLMAP-pose diagnostic input for registration-time
    // transition logs.  The library receives only an index-aligned optional
    // vector; missing stems remain `None` and never affect the mapper.
    let debug_oracle_poses = if let Some(path) = args.diagnose_ba_oracle_poses_file.as_ref() {
        let oracle_by_stem = poses_from_colmap_images_txt(path)?;
        Some(
            image_names
                .iter()
                .map(|name| oracle_by_stem.get(image_stem(name)).cloned())
                .collect(),
        )
    } else {
        None
    };
    let default_sfm_config = IncrementalSfmConfig::default();
    let default_ba_config = default_sfm_config.ba_config;
    let config = IncrementalSfmConfig {
        min_seed_matches: args.min_matches,
        min_pnp_inliers: args.min_pnp_inliers,
        max_reprojection_error_px: args.max_reproj,
        next_image_policy: args.next_image_policy,
        final_global_ba: args.final_ba,
        final_min_track_length: args.final_min_track_length,
        seed_trials: args.seed_trials,
        seed_pair: args.seed_pair,
        // Distortion self-calibration runs inside the joint intrinsics BA, so it
        // implies intrinsics refinement; the (k1, k2) flag rides on `ba_config`.
        refine_intrinsics: args.refine_intrinsics || args.refine_distortion,
        ba_config: BaConfig {
            max_iterations: args
                .ba_max_iterations
                .unwrap_or(default_ba_config.max_iterations),
            robust_kernel: args
                .ba_huber_delta
                .map_or(default_ba_config.robust_kernel, |delta| {
                    RobustKernel::Huber { delta }
                }),
            linear_solver: args
                .ba_linear_solver
                .unwrap_or(default_ba_config.linear_solver),
            refine_distortion: args.refine_distortion,
            ..default_ba_config
        },
        periodic_ba_min_registered_images: args.periodic_ba_min_registered_images,
        final_ba_polish_iterations: args.final_ba_polish_iterations,
        colmap_style_mapper: args.colmap_style,
        final_iterative_global_refinement: args.final_iterative_global_refinement,
        global_ba_max_refinements: args
            .global_ba_max_refinements
            .unwrap_or(default_sfm_config.global_ba_max_refinements),
        structureless_registration: args.structureless_registration,
        verify_registration_two_view: args.verify_registration_two_view,
        sequence_relative_pose_fallback: args.sequence_relative_pose_fallback,
        sequence_fallback_after_post: args.sequence_fallback_after_post,
        sequence_constant_velocity_scale: args.sequence_constant_velocity_scale,
        sequence_relaxed_constant_velocity_scale: args.sequence_relaxed_constant_velocity_scale,
        sequence_fallback_carry_scale: args.sequence_fallback_carry_scale,
        sequence_stem_values,
        pnp_max_iterations: args.pnp_max_iterations,
        filter_images: args.filter_images,
        track_source: args.track_source,
        incremental_correspondence_triangulation: args.incremental_correspondence_triangulation,
        confidence_ordered_tracks: args.confidence_ordered_tracks,
        geometric_confidence_tracks: args.geometric_confidence_tracks,
        stable_track_order: args.stable_track_order || args.canonical_feature_order,
        cycle_supported_tracks: args.cycle_supported_tracks,
        geometry_weighted_ba: args.geometry_weighted_ba,
        freeze_ill_conditioned_landmarks: args.freeze_ill_conditioned_landmarks,
        landmark_ba_warm_start_iterations: args.landmark_ba_warm_start_iterations,
        landmark_ba_warm_start_min_registered_images: args
            .landmark_ba_warm_start_min_registered_images,
        debug_oracle_poses,
        // The staged path is passed separately to the mapper below so the
        // ordinary config/default snapshot remains unchanged.
        geometry_guided_conflict_recovery: args.geometry_guided_conflict_recovery,
        pose_guided_track_splitting: args.pose_guided_track_splitting,
        pose_guided_track_splitting_iterations: args
            .pose_guided_track_splitting_iterations
            .unwrap_or(1),
        pose_guided_graph_support: args.pose_guided_track_splitting_graph_support,
        pose_guided_bridge_cuts: args.pose_guided_track_splitting_bridge_cuts,
        pose_guided_split_max_reprojection_error_px: args.pose_guided_split_max_reproj,
        pose_guided_track_merging: args.pose_guided_track_merging,
        pose_guided_merge_max_reprojection_error_px: args.pose_guided_merge_max_reproj,
        post_refinement_registration: args.post_refinement_registration,
        ..IncrementalSfmConfig::default()
    };
    if let Some(path) = args.export_verified_pairs_snapshot.as_deref() {
        write_verified_pair_snapshot(
            path,
            &image_names,
            &features,
            &args.camera,
            &pairwise,
            &snapshot_metadata,
            &args,
        )?;
        println!(
            "export verified-pair snapshot: {} pairs, {} accepted correspondences, ordered-edge-fnv1a64={:016x}, unordered-edge-fnv1a64={:016x} -> {}",
            pairwise.len(),
            pairwise.iter().map(|pair| pair.matches.len()).sum::<usize>(),
            ordered_pairwise_edge_hash(&pairwise),
            unordered_pairwise_edge_hash(&pairwise),
            path.display(),
        );
    }
    // Snapshot metadata is needed for export and sequence-fallback decisions
    // only.  Both have completed before mapping starts; do not carry the
    // lossless raw-match/index copies into track building or BA.
    drop(snapshot_metadata);
    if snapshot_imported {
        log_process_memory("example-after-snapshot-metadata-release");
    }
    if args.export_verified_pairs_only {
        // This mode is the resumable match-shard worker.  The complete
        // verified stream is durable at this point; track construction and
        // mapping intentionally happen only once after all shards merge.
        println!(
            "verified-pair export-only: mapping skipped; snapshot is complete at {}",
            args.export_verified_pairs_snapshot
                .as_deref()
                .expect("validated export-only path")
                .display(),
        );
        return Ok(());
    }
    if let Some(limit) = args.max_mapper_matches_per_pair {
        let cap_stats = cap_mapper_pair_matches(&mut pairwise, limit);
        println!(
            "mapper match cap: limit={} pairs_capped={} matches {}=>{} essential {}=>{} (verified snapshot/diagnostics retain full stream)",
            limit,
            cap_stats.pairs_capped,
            cap_stats.matches_before,
            cap_stats.matches_after,
            cap_stats.essential_before,
            cap_stats.essential_after,
        );
    }
    log_process_memory("example-after-mapper-cap");
    let gt_path = args
        .gt_chirality_oracle_path
        .as_ref()
        .or(args.diagnose_bearing_gt.as_ref())
        .or(args.rematch_gt_bearing_path.as_ref());
    let gt_by_stem = if let Some(path) = gt_path {
        Some(poses_from_colmap_images_txt(path)?)
    } else {
        None
    };
    let diagnose_stems: HashSet<&str> = if args.diagnose_bearing_stems.is_empty() {
        [
            "DSC_0297", "DSC_0320", "DSC_0321", "DSC_0322", "DSC_0323", "DSC_0296",
        ]
        .iter()
        .copied()
        .collect()
    } else {
        args.diagnose_bearing_stems
            .iter()
            .map(String::as_str)
            .collect()
    };
    if args.diagnose_bearing_gt.is_some() {
        if let Some(ref gt) = gt_by_stem {
            diagnose_bearing_vs_gt(
                "pre-mapper",
                &pairwise,
                &features,
                &args.camera,
                &image_names,
                gt,
                &diagnose_stems,
            );
        }
    }
    let gt_poses_aligned = gt_by_stem
        .as_ref()
        .map(|gt| gt_poses_aligned(&image_names, gt));
    if args.mapper == MapperKind::Global || args.mapper == MapperKind::Hybrid {
        let mut rematch_prefer_e_pairs: Vec<(usize, usize)> = Vec::new();
        let pose_priors = if args.mapper == MapperKind::Hybrid {
            log_process_memory("example-before-incremental-mapper");
            let inc = incremental_sfm(&args.camera, &features, &pairwise, &config)?;
            let mut priors = if args.hybrid_filter_priors {
                let (filtered, kept) = filter_pose_priors_by_track_quality(
                    &args.camera,
                    &inc.poses,
                    &inc.tracks,
                    args.hybrid_prior_min_obs,
                    args.hybrid_prior_max_reproj,
                );
                println!(
                    "hybrid: filtered incremental priors {kept} / {} (min_obs={}, max_reproj={:.2} px)",
                    inc.registered_images,
                    args.hybrid_prior_min_obs,
                    args.hybrid_prior_max_reproj,
                );
                filtered
            } else {
                inc.poses
            };
            if !args.hybrid_drop_prior_stems.is_empty() {
                let drop: HashSet<&str> = args
                    .hybrid_drop_prior_stems
                    .iter()
                    .map(String::as_str)
                    .collect();
                let mut cleared = 0usize;
                for (i, pose) in priors.iter_mut().enumerate() {
                    if pose.is_none() {
                        continue;
                    }
                    let stem = Path::new(&image_names[i])
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or(image_names[i].as_str());
                    if drop.contains(stem) {
                        *pose = None;
                        cleared += 1;
                    }
                }
                println!(
                    "hybrid: dropped {} prior(s) by stem {:?}",
                    cleared, args.hybrid_drop_prior_stems
                );
            }
            println!(
                "hybrid: incremental priors {} / {} images (mean reproj {:.3} px)",
                priors.iter().filter(|p| p.is_some()).count(),
                features.len(),
                inc.mean_reprojection_px,
            );
            if args.rematch_free_vs_priors {
                let (n, gained) = rematch_free_against_priors(
                    &features,
                    &image_names,
                    &mut pairwise,
                    &args.camera,
                    &priors,
                    &args.rematch_stems,
                    args.rematch_ratio,
                    args.rematch_cross_check,
                    args.min_matches,
                    args.verification_mode,
                    &pair_matcher,
                    args.rematch_guided || args.guided_matching,
                    args.multiple_models,
                    args.rematch_min_e_f_inlier_ratio
                        .or(args.min_e_f_inlier_ratio),
                    args.rematch_calibrated_prefer_essential || args.calibrated_prefer_essential,
                    args.force_essential_min_ef_ratio,
                    args.force_essential_min_e_inliers,
                    args.rematch_tracks_use_essential,
                    args.rematch_min_chirality_margin,
                    args.rematch_prior_anchor,
                    args.rematch_anchor_min_e_inliers,
                    gt_by_stem.as_ref(),
                    args.rematch_max_gt_bearing_deg,
                    args.rematch_guided_max_error_px,
                    args.rematch_guided_lowe_ratio,
                    args.rematch_require_calibrated,
                    args.rematch_max_mean_sampson,
                    args.rematch_prior_ray_guided,
                    args.rematch_prior_ray_min_rays,
                    args.rematch_prior_ray_min_e_inliers,
                    args.rematch_verification_mode,
                    args.pair_stem_window,
                );
                let min_e = args.rematch_prefer_min_e_inliers;
                let strong_want: HashSet<&str> = args
                    .rematch_prefer_strong_stems
                    .iter()
                    .map(String::as_str)
                    .collect();
                let strong_idx: HashSet<usize> = image_names
                    .iter()
                    .enumerate()
                    .filter_map(|(i, name)| {
                        let stem = Path::new(name)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or(name.as_str());
                        strong_want.contains(stem).then_some(i)
                    })
                    .collect();
                let strong_min = args.rematch_prefer_strong_min_e;
                rematch_prefer_e_pairs = gained
                    .into_iter()
                    .filter_map(|(pair, e)| {
                        let needs_strong =
                            strong_idx.contains(&pair.0) || strong_idx.contains(&pair.1);
                        let thr = if needs_strong {
                            strong_min.max(min_e)
                        } else {
                            min_e
                        };
                        (e >= thr).then_some(pair)
                    })
                    .collect();
                println!(
                    "hybrid: rematch-free-vs-priors changed {} pair(s) (stems {:?}, ratio={:.2}); prefer-E (min_e={}, strong_min={} on {:?}) on {:?}",
                    n, args.rematch_stems, args.rematch_ratio, min_e, strong_min, args.rematch_prefer_strong_stems, rematch_prefer_e_pairs
                );
                if args.diagnose_bearing_gt.is_some() {
                    if let Some(ref gt) = gt_by_stem {
                        diagnose_bearing_vs_gt(
                            "post-rematch",
                            &pairwise,
                            &features,
                            &args.camera,
                            &image_names,
                            gt,
                            &diagnose_stems,
                        );
                    }
                }
            }
            Some(priors)
        } else {
            None
        };
        let mut prefer_essential_edge_pairs = args.prefer_essential_pairs.clone();
        prefer_essential_edge_pairs.extend(rematch_prefer_e_pairs);
        prefer_essential_edge_pairs.sort_unstable();
        prefer_essential_edge_pairs.dedup();
        let mut tuning = GlobalReconstructionTuning {
            min_pair_matches: args.min_matches,
            min_edge_inliers: args.min_edge_inliers,
            min_edge_parallax_deg: args.min_edge_parallax_deg,
            chirality_harden_edges: args.chirality_harden,
            rotation_seed_trials: args.rotation_seed_trials,
            refine_translations_with_global_rotations: args.refine_global_translations,
            independent_edge_scales: args.global_independent_edge_scales,
            multi_hypothesis_edges: args.multi_hypothesis_edges,
            weight_edges_by_chirality_margin: args.weight_by_chirality_margin,
            hybrid_rotation_priors_only: args.hybrid_rotation_priors_only,
            joint_global_positioning: args.joint_global_positioning,
            calibrated_view_edges_only: args.calibrated_view_edges_only,
            prefer_essential_edge_matches: args.prefer_essential_inliers,
            prefer_essential_edge_matches_free_endpoints: args.prefer_essential_free_endpoints,
            prefer_essential_edge_image_indices: {
                let want: HashSet<&str> = args
                    .prefer_essential_stems
                    .iter()
                    .map(String::as_str)
                    .collect();
                if want.is_empty() {
                    Vec::new()
                } else {
                    image_names
                        .iter()
                        .enumerate()
                        .filter_map(|(i, name)| {
                            let stem = Path::new(name)
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or(name.as_str());
                            want.contains(stem).then_some(i)
                        })
                        .collect()
                }
            },
            prefer_essential_edge_stem_clique: args.prefer_essential_stem_clique,
            prefer_essential_edge_pairs: prefer_essential_edge_pairs.clone(),
            require_essential_for_selected_edges: args.require_essential_selected_edges,
            require_essential_edge_image_indices: {
                let want: HashSet<&str> = args
                    .require_essential_stems
                    .iter()
                    .map(String::as_str)
                    .collect();
                if want.is_empty() {
                    Vec::new()
                } else {
                    image_names
                        .iter()
                        .enumerate()
                        .filter_map(|(i, name)| {
                            let stem = Path::new(name)
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or(name.as_str());
                            want.contains(stem).then_some(i)
                        })
                        .collect()
                }
            },
            require_essential_min_e_inliers: args.require_essential_min_e_inliers,
            essential_edge_weight_boost: args.essential_edge_weight_boost,
            repair_edges_from_pose_priors: args.repair_prior_edges,
            repair_free_edges_from_solved_poses: args.repair_free_edges_from_solved,
            repair_free_edges_only_flipped: args.repair_free_edges_only_flipped,
            repair_free_edges_image_indices: {
                let want: HashSet<&str> = args
                    .repair_free_edges_stems
                    .iter()
                    .map(String::as_str)
                    .collect();
                if want.is_empty() {
                    Vec::new()
                } else {
                    image_names
                        .iter()
                        .enumerate()
                        .filter_map(|(i, name)| {
                            let stem = Path::new(name)
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or(name.as_str());
                            want.contains(stem).then_some(i)
                        })
                        .collect()
                }
            },
            drop_free_edges_antipodal_to_solved: args.drop_free_edges_antipodal,
            prior_guided_free_chirality: args.prior_guided_free_chirality,
            metric_prior_chirality_edges: args.metric_prior_chirality_edges,
            metric_prior_chirality_min_rays: args.metric_prior_chirality_min_rays,
            gt_chirality_oracle: args.gt_chirality_oracle,
            metric_scale_from_pose_priors: args.metric_prior_scale,
            drop_inconsistent_pose_priors: args.hybrid_drop_inconsistent_priors,
            repnp_free_cameras_from_priors: args.repnp_free_from_priors,
            repnp_free_min_corrs: args.repnp_free_min_corrs,
            repnp_seed_free_as_priors: args.repnp_seed_free_as_priors,
            // Never re-pin stems the user explicitly dropped as priors.
            repnp_seed_exclude_image_indices: {
                let want: HashSet<&str> = args
                    .hybrid_drop_prior_stems
                    .iter()
                    .map(String::as_str)
                    .collect();
                if want.is_empty() {
                    Vec::new()
                } else {
                    image_names
                        .iter()
                        .enumerate()
                        .filter_map(|(i, name)| {
                            let stem = Path::new(name)
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or(name.as_str());
                            want.contains(stem).then_some(i)
                        })
                        .collect()
                }
            },
            ..GlobalReconstructionTuning::default()
        };
        if args.hybrid_rotation_priors_only && args.mapper == MapperKind::Hybrid {
            println!("hybrid: rotation-only priors (centres from global averaging)");
        }
        if args.joint_global_positioning {
            println!("global: joint track positioning (GLOMAP-style ray IRLS)");
        }
        if args.calibrated_view_edges_only {
            println!(
                "global: calibrated two-view configs only (drop uncalibrated/planar/panoramic)"
            );
        }
        if args.hybrid_drop_inconsistent_priors {
            println!("hybrid: drop priors inconsistent with free-centre probe (Sim3 residual)");
        }
        if args.repnp_free_from_priors {
            println!("hybrid: re-PnP free cameras against prior-anchored tracks");
        }
        if args.repnp_seed_free_as_priors {
            println!("hybrid: pre-global PnP seed free cameras as pose priors");
        }
        if args.repair_prior_edges {
            println!("hybrid: repairing prior–prior edges from incremental pose priors");
        }
        if args.repair_free_edges_from_solved {
            println!("hybrid: pass-2 free-incident edge repair from solved poses");
        }
        if args.metric_prior_scale {
            println!("hybrid: metric scale row from prior–prior baseline");
        }
        let gt_slice = gt_poses_aligned.as_deref();
        log_process_memory("example-before-global-mapper");
        let (mut poses, mut tracks, mut mean_reproj) = match pose_priors.as_ref() {
            Some(priors) => reconstruct_global_sfm_with_priors(
                &args.camera,
                &features,
                &pairwise,
                &tuning,
                &config,
                Some(priors.as_slice()),
                gt_slice,
            )?,
            None => reconstruct_global_sfm(&args.camera, &features, &pairwise, &tuning, &config)?,
        };
        log_process_memory("example-after-global-mapper");
        if args.rematch_pose_guided_after_global {
            if let Some(priors) = pose_priors.as_ref() {
                let gt_guide = match args.rematch_pose_guided_gt.as_ref() {
                    Some(path) => Some(poses_from_colmap_images_txt(path)?),
                    None => None,
                };
                if let Some(ref g) = gt_guide {
                    println!(
                        "hybrid: pose-guided rematch using GT poses from {:?} ({} stems)",
                        args.rematch_pose_guided_gt.as_ref().unwrap(),
                        g.len()
                    );
                }
                let (n, gained) = rematch_pose_guided_free_vs_priors(
                    &features,
                    &image_names,
                    &mut pairwise,
                    &args.camera,
                    &poses,
                    priors,
                    gt_guide.as_ref(),
                    &args.rematch_stems,
                    args.rematch_ratio,
                    args.rematch_cross_check,
                    args.min_matches,
                    &pair_matcher,
                    args.rematch_tracks_use_essential,
                    args.pair_stem_window,
                );
                let min_e = args.rematch_prefer_min_e_inliers;
                let strong_want: HashSet<&str> = args
                    .rematch_prefer_strong_stems
                    .iter()
                    .map(String::as_str)
                    .collect();
                let strong_idx: HashSet<usize> = image_names
                    .iter()
                    .enumerate()
                    .filter_map(|(i, name)| {
                        let stem = Path::new(name)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or(name.as_str());
                        strong_want.contains(stem).then_some(i)
                    })
                    .collect();
                let strong_min = args.rematch_prefer_strong_min_e;
                let extra_prefer: Vec<(usize, usize)> = gained
                    .into_iter()
                    .filter_map(|(pair, e)| {
                        let needs_strong =
                            strong_idx.contains(&pair.0) || strong_idx.contains(&pair.1);
                        let thr = if needs_strong {
                            strong_min.max(min_e)
                        } else {
                            min_e
                        };
                        (e >= thr).then_some(pair)
                    })
                    .collect();
                println!(
                    "hybrid: rematch-pose-guided-after-global changed {} pair(s); prefer-E +{:?}",
                    n, extra_prefer
                );
                if n > 0 {
                    prefer_essential_edge_pairs.extend(extra_prefer);
                    prefer_essential_edge_pairs.sort_unstable();
                    prefer_essential_edge_pairs.dedup();
                    tuning.prefer_essential_edge_pairs = prefer_essential_edge_pairs.clone();
                    let (p2, t2, r2) = reconstruct_global_sfm_with_priors(
                        &args.camera,
                        &features,
                        &pairwise,
                        &tuning,
                        &config,
                        Some(priors.as_slice()),
                        gt_slice,
                    )?;
                    poses = p2;
                    tracks = t2;
                    mean_reproj = r2;
                }
            } else {
                println!("hybrid: rematch-pose-guided-after-global skipped (no pose priors)");
            }
        }
        let registered_indices: Vec<usize> = poses
            .iter()
            .enumerate()
            .filter_map(|(image, pose)| pose.is_some().then_some(image))
            .collect();
        let registered = registered_indices.len();
        println!(
            "reconstruction: mapper={}: {} / {} images registered, {} tracks, mean reproj {:.3} px",
            match args.mapper {
                MapperKind::Global => "global",
                MapperKind::Hybrid => "hybrid",
                MapperKind::Incremental => unreachable!(),
            },
            registered,
            features.len(),
            tracks.len(),
            mean_reproj
        );
        // Compact the pose list to only registered images so the shared
        // COLMAP export path applies unchanged.
        let mut remap = HashMap::with_capacity(registered_indices.len());
        for (output_index, &image) in registered_indices.iter().enumerate() {
            remap.insert(image, output_index);
        }
        let poses_out: Vec<Pose> = registered_indices
            .iter()
            .map(|&image| poses[image].clone().expect("registered pose is present"))
            .collect();
        let mut features_out: Vec<FeatureSet> = registered_indices
            .iter()
            .map(|&image| features[image].clone())
            .collect();
        if let Some(native_keypoints) = native_keypoints_for_export.as_ref() {
            replace_feature_keypoints_from_native(
                &mut features_out,
                &registered_indices,
                native_keypoints,
            )
            .map_err(std::io::Error::other)?;
        }
        let export_cameras_out: Option<Vec<Camera>> =
            per_image_calibration.as_ref().map(|loaded| {
                registered_indices
                    .iter()
                    .map(|&image| loaded.native_cameras[image].clone())
                    .collect()
            });
        let names_out: Vec<String> = registered_indices
            .iter()
            .map(|&image| image_names[image].clone())
            .collect();
        let landmarks_out: Vec<ExportLandmark> = tracks
            .iter()
            .map(|t| {
                let obs = t
                    .observations
                    .iter()
                    .filter_map(|&(img, kp, px)| remap.get(&img).map(|&ni| (ni, kp, px)))
                    .collect();
                (t.position, obs)
            })
            .collect();
        let summary = if let Some(cameras_out) = export_cameras_out.as_ref() {
            write_colmap_reconstruction_for_3dgs_with_cameras(
                &args.out_colmap,
                cameras_out,
                &poses_out,
                &features_out,
                &landmarks_out,
                |k| names_out[k].clone(),
            )?
        } else {
            write_colmap_reconstruction_for_3dgs(
                &args.out_colmap,
                &args.camera,
                &poses_out,
                &features_out,
                &landmarks_out,
                |k| names_out[k].clone(),
            )?
        };
        println!(
            "wrote COLMAP model to {} ({} images, {} points, {} observations)",
            args.out_colmap.display(),
            summary.frame_count,
            summary.landmark_count,
            summary.observation_count,
        );
        return Ok(());
    }
    log_process_memory("example-before-incremental-mapper");
    let mut result = if let Some(membership) = colmap_track_membership.as_ref() {
        incremental_sfm_with_track_membership(
            &args.camera,
            &features,
            &pairwise,
            &config,
            &membership.tracks,
        )?
    } else if args.sequence_relative_pose_fallback
        && initial_poses.is_none()
        && !sequence_fallback_high_support_override_pair_indices.is_empty()
    {
        incremental_sfm_with_sequence_fallback_overrides(
            &args.camera,
            &features,
            &pairwise,
            &config,
            &sequence_fallback_high_support_override_pair_indices,
        )?
    } else {
        incremental_sfm_with_initial_poses(
            &args.camera,
            &features,
            &pairwise,
            &config,
            initial_poses.as_deref(),
        )?
    };
    log_process_memory("example-after-incremental-mapper");
    if let Some(oracle_path) = args.diagnose_ba_oracle_poses_file.as_deref() {
        let (scale, initial_reprojection, final_reprojection, ba_result) =
            run_oracle_pose_ba_probe(
                &mut result,
                &features,
                &image_names,
                &args.camera,
                &config,
                oracle_path,
            )?;
        println!(
            "oracle-ba probe: injected {} poses, support={} tracks, sim3_point_scale={scale:.6}, \
             reproj={initial_reprojection:.6}->{final_reprojection:.6} px, \
             ba_cost={:.9e}->{:.9e}, iterations={}, converged={}",
            result.registered_images,
            result.tracks.len(),
            ba_result.initial_cost,
            ba_result.final_cost,
            ba_result.iterations.len(),
            ba_result.converged,
        );
    }
    if let Some(source) = args.diagnose_fixed_rotation_ba.as_deref() {
        if args.diagnose_ba_oracle_poses_file.is_some() {
            return Err(
                "--diagnose-fixed-rotation-ba cannot be combined with --diagnose-ba-oracle-poses"
                    .into(),
            );
        }
        let (
            source_label,
            fixed_count,
            initial_reprojection,
            final_reprojection,
            max_rotation_delta,
            ba_result,
        ) = run_fixed_rotation_ba_probe(
            &mut result,
            &features,
            &image_names,
            &args.camera,
            &config,
            source,
        )?;
        println!(
            concat!(
                "fixed-rotation-ba probe: source={:?} fixed_rotations={} support={} ",
                "reproj={:.6}->{:.6} px, max_rotation_delta={:.3e} deg, ",
                "ba_cost={:.9e}->{:.9e}, iterations={}, converged={}"
            ),
            source_label,
            fixed_count,
            result.tracks.len(),
            initial_reprojection,
            final_reprojection,
            max_rotation_delta,
            ba_result.initial_cost,
            ba_result.final_cost,
            ba_result.iterations.len(),
            ba_result.converged,
        );
    }
    let track_label = if args.diagnose_colmap_track_membership.is_some() {
        "oracle-colmap-track-membership"
    } else if args.pose_guided_track_splitting
        && args.geometry_guided_conflict_recovery
        && args.pose_guided_track_merging
    {
        "geometry-recovery+pose-guided-track-splitting+merging"
    } else if args.pose_guided_track_splitting && args.geometry_guided_conflict_recovery {
        "geometry-recovery+pose-guided-track-splitting"
    } else if args.pose_guided_track_splitting && args.pose_guided_track_merging {
        "pose-guided-track-splitting+merging"
    } else if args.pose_guided_track_splitting && args.pose_guided_track_splitting_graph_support {
        "pose-guided-track-splitting+graph-support"
    } else if args.pose_guided_track_splitting {
        "pose-guided-track-splitting"
    } else if args.incremental_correspondence_triangulation {
        "incremental-correspondence-triangulation"
    } else if args.cycle_supported_tracks && args.canonical_feature_order {
        "cycle-supported+canonical-feature-order"
    } else if args.cycle_supported_tracks {
        "cycle-supported"
    } else if args.canonical_feature_order {
        "canonical-feature-order"
    } else if args.stable_track_order {
        "stable-track-order"
    } else if args.geometric_confidence_tracks {
        "track-geometric-confidence"
    } else if args.confidence_ordered_tracks {
        "track-confidence-ordered"
    } else {
        match args.track_source {
            TrackSource::UnionFind => "track-source=union-find",
            TrackSource::CorrespondenceGraph => "track-source=graph",
        }
    };
    println!(
        "reconstruction ({}): {} / {} images registered, {} tracks, mean reproj {:.3} px",
        track_label,
        result.registered_images,
        features.len(),
        result.tracks.len(),
        result.mean_reprojection_px,
    );

    // When intrinsics were refined, export with the refined camera (and report the
    // before→after pull — on observable, wide-parallax capture this is where focal
    // length is recoverable, unlike low-parallax forward video).
    let export_camera = result.refined_camera.clone().unwrap_or(args.camera.clone());
    if let (Some(i0), Some(i1)) = (args.camera.intrinsics(), export_camera.intrinsics()) {
        if result.refined_camera.is_some() {
            println!(
                "refined intrinsics: fx {:.2}->{:.2}  fy {:.2}->{:.2}  cx {:.2}->{:.2}  cy {:.2}->{:.2}",
                i0.0, i1.0, i0.1, i1.1, i0.2, i1.2, i0.3, i1.3,
            );
            if let Some((k1, k2)) = export_camera.radial_distortion() {
                let (k1_0, k2_0) = args.camera.radial_distortion().unwrap_or((0.0, 0.0));
                println!("refined distortion: k1 {k1_0:.5}->{k1:.5}  k2 {k2_0:.5}->{k2:.5}");
            }
        }
    }

    // Compact to registered images (the COLMAP writer expects a dense pose list)
    // and remap each track observation's image index.
    let registered: Vec<usize> = (0..features.len())
        .filter(|&i| result.poses[i].is_some())
        .collect();
    let remap: HashMap<usize, usize> = registered
        .iter()
        .enumerate()
        .map(|(new_idx, &old)| (old, new_idx))
        .collect();
    let poses_out: Vec<Pose> = registered
        .iter()
        .map(|&i| result.poses[i].clone().unwrap())
        .collect();
    let mut features_out: Vec<FeatureSet> =
        registered.iter().map(|&i| features[i].clone()).collect();
    if let Some(native_keypoints) = native_keypoints_for_export.as_ref() {
        replace_feature_keypoints_from_native(&mut features_out, &registered, native_keypoints)
            .map_err(std::io::Error::other)?;
    }
    let names_out: Vec<String> = registered.iter().map(|&i| image_names[i].clone()).collect();
    let export_cameras_out: Option<Vec<Camera>> = per_image_calibration.as_ref().map(|loaded| {
        registered
            .iter()
            .map(|&i| loaded.native_cameras[i].clone())
            .collect()
    });
    let landmarks_out: Vec<ExportLandmark> = result
        .tracks
        .iter()
        .map(|t| {
            let obs = t
                .observations
                .iter()
                .filter_map(|&(img, kp, px)| remap.get(&img).map(|&ni| (ni, kp, px)))
                .collect();
            (t.position, obs)
        })
        .collect();

    let summary = if let Some(cameras_out) = export_cameras_out.as_ref() {
        write_colmap_reconstruction_for_3dgs_with_cameras(
            &args.out_colmap,
            cameras_out,
            &poses_out,
            &features_out,
            &landmarks_out,
            |k| names_out[k].clone(),
        )?
    } else {
        write_colmap_reconstruction_for_3dgs(
            &args.out_colmap,
            &export_camera,
            &poses_out,
            &features_out,
            &landmarks_out,
            |k| names_out[k].clone(),
        )?
    };
    println!(
        "wrote COLMAP model to {} ({} images, {} points, {} observations)",
        args.out_colmap.display(),
        summary.frame_count,
        summary.landmark_count,
        summary.observation_count,
    );
    Ok(())
}

#[cfg(test)]
mod diagnose_cli_tests {
    use super::{
        apply_snapshot_coordinate_override, apply_union_traversal_order,
        apply_union_traversal_order_with_features, candidate_pairs_vlad_union,
        canonicalize_feature_order, canonicalize_pairwise_loci, cap_mapper_pair_matches,
        colmap_guided_geometry, colmap_guided_matches, descriptor_squared_distance,
        effective_config_hash, effective_config_snapshot,
        filter_imported_verified_pairs_by_stem_window, filter_pairs_by_stem_window,
        imported_reference_quality_is_strong, initial_poses_from_colmap_images_txt,
        initial_poses_from_colmap_images_txt_with_expected_cameras, load_images,
        load_images_keypoints_only, load_input_colmap_calibration, match_candidate_cmp,
        model_cross_validation_is_held_out, model_cross_validation_is_held_out_for_pixels,
        model_cross_validation_selection_score, parse_args_from, parse_candidate_manifest,
        parse_candidate_manifest_with_metadata, parse_colmap_image_camera_assignments,
        parse_colmap_track_membership, parse_diagnose_stems, remap_feature_keypoints_by_old_to_new,
        replace_feature_keypoints_from_native, rig_local_pairs, rig_temporal_pyramid_pairs,
        robust_huber_mean, snapshot_export_config, snapshot_feature_manifest_hash,
        snapshot_feature_validation_from_files, summarize_model_cross_validation_bucket,
        temporal_pyramid_offsets_string, translation_direction_delta_deg,
        unordered_pairwise_edge_hash, validate_diagnose_options, verified_pair_oracle_map,
        write_candidate_manifest, write_candidate_manifest_with_metadata, Camera,
        ColmapTrackMembership, ConfigurationType, EssentialPairQuality, FeatureLocusMetadata,
        ImportedVerifiedPair, IncrementalSfmConfig, LinearSolver, ModelCrossValidationBucket,
        ModelCrossValidationSummary, NextImagePolicy, PairwiseMatches, PerImageCameras,
        RobustKernel, TwoViewGeometryReport, UnionTraversalOrder,
    };
    use nalgebra::{Matrix3, Point2, Vector3};
    use std::cmp::Ordering as CmpOrdering;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use visloc_rs::vision::features::FeatureSet;
    use visloc_rs::DescriptorMatch;

    fn minimal_args(extra: &[&str]) -> Vec<String> {
        let mut args = [
            "--width",
            "1600",
            "--height",
            "1066",
            "--fx",
            "879.4",
            "--fy",
            "879.4",
            "--cx",
            "803.4",
            "--cy",
            "532.6",
            "--out-colmap",
            "/tmp/diagnose-cli-test",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        args.extend(extra.iter().map(|arg| (*arg).to_owned()));
        args
    }

    #[test]
    fn next_image_policy_defaults_to_auto_and_snapshot_replay_stays_count() {
        let defaults = parse_args_from(minimal_args(&[])).unwrap();
        assert_eq!(defaults.next_image_policy, NextImagePolicy::Auto);

        let count = parse_args_from(minimal_args(&["--next-image-policy", "count"])).unwrap();
        assert_eq!(
            count.next_image_policy,
            NextImagePolicy::CorrespondenceCount
        );
        let visibility =
            parse_args_from(minimal_args(&["--next-image-policy", "visibility"])).unwrap();
        assert_eq!(
            visibility.next_image_policy,
            NextImagePolicy::VisibilityPyramid
        );
        let auto = parse_args_from(minimal_args(&["--next-image-policy", "auto"])).unwrap();
        assert_eq!(auto.next_image_policy, NextImagePolicy::Auto);
        let snapshot_default = parse_args_from(minimal_args(&[
            "--import-verified-pairs-snapshot",
            "/tmp/pairs.vps",
        ]))
        .unwrap();
        assert_eq!(
            snapshot_default.next_image_policy,
            NextImagePolicy::CorrespondenceCount
        );
        let snapshot_auto = parse_args_from(minimal_args(&[
            "--import-verified-pairs-snapshot",
            "/tmp/pairs.vps",
            "--next-image-policy",
            "auto",
        ]))
        .unwrap();
        assert_eq!(snapshot_auto.next_image_policy, NextImagePolicy::Auto);
        assert!(parse_args_from(minimal_args(&["--next-image-policy", "pyramid"])).is_err());
        assert!(parse_args_from(minimal_args(&["--next-image-policy"])).is_err());
    }

    #[test]
    fn mapper_match_cap_is_opt_in_and_preserves_verified_prefixes() {
        let defaults = parse_args_from(minimal_args(&[])).unwrap();
        assert!(defaults.max_mapper_matches_per_pair.is_none());
        assert!(parse_args_from(minimal_args(&["--max-mapper-matches-per-pair", "0",])).is_err());

        let mut pairwise = vec![PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 10), (1, 11), (2, 12)],
            two_view_config: Some(ConfigurationType::Calibrated),
            essential_matches: Some(vec![(0, 10), (2, 12)]),
            essential_matrix: Some(Matrix3::identity()),
        }];
        let stats = cap_mapper_pair_matches(&mut pairwise, 2);
        assert_eq!(stats.pairs_capped, 1);
        assert_eq!(stats.matches_before, 3);
        assert_eq!(stats.matches_after, 2);
        assert_eq!(stats.essential_before, 2);
        assert_eq!(stats.essential_after, 2);
        assert_eq!(pairwise[0].matches, vec![(0, 10), (1, 11)]);
        assert_eq!(
            pairwise[0].essential_matches.as_deref(),
            Some(&[(0, 10), (2, 12)][..])
        );
    }

    #[test]
    fn candidate_manifest_round_trip_binds_image_order_and_rejects_duplicates() {
        let root =
            std::env::temp_dir().join(format!("visloc_candidate_manifest_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("pairs.txt");
        let names = vec![
            "DSC_0001.JPG".to_owned(),
            "DSC_0002.JPG".to_owned(),
            "DSC_0003.JPG".to_owned(),
        ];
        let pairs = vec![(0, 2), (0, 1)];
        write_candidate_manifest(&path, &names, &pairs).unwrap();
        assert_eq!(parse_candidate_manifest(&path, &names).unwrap(), pairs);

        let mut wrong_names = names.clone();
        wrong_names.swap(0, 1);
        assert!(parse_candidate_manifest(&path, &wrong_names)
            .unwrap_err()
            .contains("image entry"));

        std::fs::write(
            &path,
            "visloc_candidate_manifest_v1\nimages 3\n\
             image 0 DSC_0001.JPG\nimage 1 DSC_0002.JPG\nimage 2 DSC_0003.JPG\n\
             pairs 2\npair 0 1\npair 1 0\n",
        )
        .unwrap();
        assert!(parse_candidate_manifest(&path, &names)
            .unwrap_err()
            .contains("must satisfy"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn candidate_schedule_flags_are_explicit_and_validated() {
        let union = parse_args_from(minimal_args(&[
            "--pair-source",
            "vlad-union",
            "--local-stem-window",
            "3",
            "--candidate-budget",
            "200",
        ]))
        .unwrap();
        assert!(matches!(union.pair_source, super::PairSource::VladUnion));
        assert_eq!(union.local_stem_window, Some(3));
        assert_eq!(union.candidate_budget, Some(200));
        let rig = parse_args_from(minimal_args(&[
            "--pair-source",
            "vlad-union",
            "--local-stem-window",
            "3",
            "--rig-local-grouping",
        ]))
        .unwrap();
        assert!(rig.rig_local_grouping);
        assert!(parse_args_from(minimal_args(&["--pair-source", "vlad-union",])).is_err());
        assert!(parse_args_from(minimal_args(&["--local-stem-window", "3",])).is_err());
        assert!(parse_args_from(minimal_args(&["--rig-local-grouping",])).is_err());
        assert!(parse_args_from(minimal_args(&["--candidate-budget", "0",])).is_err());

        let temporal = parse_args_from(minimal_args(&[
            "--pair-source",
            "temporal-pyramid",
            "--temporal-pyramid-max-offset",
            "64",
            "--candidate-budget",
            "12000",
        ]))
        .unwrap();
        assert!(matches!(
            temporal.pair_source,
            super::PairSource::TemporalPyramid
        ));
        assert_eq!(temporal.temporal_pyramid_max_offset, 64);
        assert_eq!(temporal.candidate_budget, Some(12000));
        assert!(parse_args_from(minimal_args(&[
            "--pair-source",
            "temporal-pyramid",
            "--temporal-pyramid-max-offset",
            "0",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&[
            "--pair-source",
            "temporal-pyramid",
            "--rig-local-grouping",
        ]))
        .is_err());
    }

    #[test]
    fn bounded_local_vlad_schedule_is_deterministic_and_respects_budget() {
        let features: Vec<FeatureSet> = (0..5)
            .map(|index| {
                FeatureSet::new(
                    vec![Point2::new(index as f64, 0.0)],
                    vec![vec![1.0, index as f32 + 1.0]],
                )
                .unwrap()
            })
            .collect();
        let names: Vec<String> = (0..5).map(|index| format!("DSC_{index:04}.JPG")).collect();
        let first = candidate_pairs_vlad_union(&features, &names, 4, 12, 1, Some(3)).unwrap();
        let second = candidate_pairs_vlad_union(&features, &names, 4, 12, 1, Some(3)).unwrap();
        assert_eq!(first, second);
        assert!(first.len() <= 3);
        assert!(first.iter().all(|&(i, j)| i < j && j < features.len()));
    }

    #[test]
    fn rig_local_grouping_keeps_temporal_edges_per_camera_and_adds_same_timestamp_edges() {
        let names = vec![
            "cam4_100.png".to_owned(),
            "cam4_102.png".to_owned(),
            "cam4_110.png".to_owned(),
            "cam5_100.png".to_owned(),
            "cam5_103.png".to_owned(),
            "cam6_100.png".to_owned(),
        ];
        let pairs = rig_local_pairs(&names, 3).unwrap();
        assert_eq!(
            pairs,
            vec![
                (0, 1), // cam4 temporal, difference 2
                (0, 3), // cam4/cam5 same timestamp
                (0, 5), // cam4/cam6 same timestamp
                (3, 4), // cam5 temporal, difference 3
                (3, 5), // cam5/cam6 same timestamp
            ]
        );
        assert!(!pairs.contains(&(1, 3))); // different timestamps/cameras
        assert!(!pairs.contains(&(2, 3))); // outside cam4 temporal window
    }

    #[test]
    fn rig_local_grouping_rejects_duplicate_timestamp_within_one_camera() {
        let names = vec![
            "cam4_100.png".to_owned(),
            "cam4_100.jpg".to_owned(),
            "cam5_100.png".to_owned(),
        ];
        let error = rig_local_pairs(&names, 3).unwrap_err();
        assert!(error.contains("repeats timestamp 100"));
    }

    #[test]
    fn temporal_pyramid_uses_positional_offsets_and_same_timestamp_rig_edges() {
        let names = vec![
            "cam4_100.png".to_owned(),
            "cam4_300.png".to_owned(),
            "cam4_900.png".to_owned(),
            "cam4_1400.png".to_owned(),
            "cam5_100.png".to_owned(),
            "cam5_900.png".to_owned(),
        ];
        let (temporal, cross) = rig_temporal_pyramid_pairs(&names, 2).unwrap();
        // Offset 1 connects adjacent positions even though timestamp gaps
        // are irregular; offset 2 connects the first and third positions.
        assert_eq!(
            temporal,
            vec![(0, 1), (1, 2), (2, 3), (4, 5), (0, 2), (1, 3)]
        );
        assert_eq!(cross, vec![(0, 4), (2, 5)]);
        assert_eq!(temporal_pyramid_offsets_string(64), "1,2,4,8,16,32,64");
    }

    #[test]
    fn temporal_pyramid_rejects_duplicate_timestamp_within_camera_but_allows_rig_duplicate() {
        let names = vec![
            "cam4_100.png".to_owned(),
            "cam4_100.jpg".to_owned(),
            "cam5_100.png".to_owned(),
        ];
        let error = rig_temporal_pyramid_pairs(&names, 32).unwrap_err();
        assert!(error.contains("repeats timestamp 100"));
        let names = vec!["cam4_100.png".to_owned(), "cam5_100.png".to_owned()];
        let (temporal, cross) = rig_temporal_pyramid_pairs(&names, 32).unwrap();
        assert!(temporal.is_empty());
        assert_eq!(cross, vec![(0, 1)]);
    }

    #[test]
    fn candidate_manifest_metadata_is_canonical_and_round_trips() {
        let root = std::env::temp_dir().join(format!(
            "visloc_candidate_manifest_metadata_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("pairs.txt");
        let names = vec!["cam4_100.png".to_owned(), "cam5_100.png".to_owned()];
        let mut metadata = BTreeMap::new();
        metadata.insert("pair_source".to_owned(), "vlad-union".to_owned());
        metadata.insert(
            "local_grouping".to_owned(),
            "rig-prefix-timestamp-v1".to_owned(),
        );
        write_candidate_manifest_with_metadata(&path, &names, &[(0, 1)], &metadata).unwrap();
        let (pairs, parsed) = parse_candidate_manifest_with_metadata(&path, &names).unwrap();
        assert_eq!(pairs, vec![(0, 1)]);
        assert_eq!(parsed, metadata);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("metadata local_grouping rig-prefix-timestamp-v1\n"));
        assert!(text.contains("metadata pair_source vlad-union\n"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn diagnose_stems_are_trimmed_and_require_a_value() {
        assert_eq!(
            parse_diagnose_stems(" DSC_0297, ,DSC_0309 ").unwrap(),
            vec!["DSC_0297", "DSC_0309"]
        );
        assert!(parse_diagnose_stems(" , \t").is_err());
    }

    #[test]
    fn per_image_calibration_maps_by_stem_and_rejects_shared_camera_flags() {
        let root = std::env::temp_dir().join(format!(
            "visloc_per_image_calibration_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("cameras.txt"),
            concat!(
                "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n",
                "7 PINHOLE 100 80 50 50 50 40\n",
                "9 PINHOLE 200 160 100 80 100 80\n",
            ),
        )
        .unwrap();
        std::fs::write(
            root.join("images.txt"),
            concat!(
                "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n",
                "1 1 0 0 0 0 0 0 7 sub/a.jpg\n",
                "0 0 -1\n",
                "2 1 0 0 0 0 0 0 9 b.jpg\n",
                "0 0 -1\n",
            ),
        )
        .unwrap();
        let names = vec!["a.png".to_owned(), "b.png".to_owned()];
        let features = vec![
            FeatureSet::new(vec![Point2::new(50.0, 40.0)], vec![vec![1.0]]).unwrap(),
            FeatureSet::new(vec![Point2::new(100.0, 80.0)], vec![vec![2.0]]).unwrap(),
        ];
        let loaded = load_input_colmap_calibration(&root, &names, &features, None).unwrap();
        assert_eq!(
            loaded
                .native_cameras
                .iter()
                .map(|camera| camera.id)
                .collect::<Vec<_>>(),
            vec![7, 9]
        );
        assert_eq!(
            parse_colmap_image_camera_assignments(
                &std::fs::read_to_string(root.join("images.txt")).unwrap()
            )
            .unwrap()
            .len(),
            2
        );
        let poses = initial_poses_from_colmap_images_txt_with_expected_cameras(
            &root.join("images.txt"),
            &names,
            loaded.rig.reference_camera(),
            Some(&loaded.native_cameras),
        )
        .unwrap();
        assert_eq!(poses.iter().filter(|pose| pose.is_some()).count(), 2);
        let parsed = parse_args_from(vec![
            "--input-colmap-calibration".into(),
            root.display().to_string(),
            "--width".into(),
            "100".into(),
            "--out-colmap".into(),
            "/tmp/per-image-calibration-test".into(),
        ]);
        assert!(parsed.is_err(), "manual scalar intrinsics must be rejected");
        let parsed = parse_args_from(vec![
            "--input-colmap-calibration".into(),
            root.display().to_string(),
            "--out-colmap".into(),
            "/tmp/per-image-calibration-test".into(),
        ])
        .unwrap();
        assert_eq!(parsed.camera.width, 1, "placeholder is replaced in main");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn diagnose_modes_and_indices_are_validated() {
        assert!(validate_diagnose_options(None, &["DSC_0297".into()], &[], None).is_err());
        assert!(
            validate_diagnose_options(Some(Path::new("pairs.csv")), &[], &[(0, 1)], None,).is_err()
        );
        assert!(validate_diagnose_options(None, &[], &[(1, 1)], Some(2)).is_err());
        assert!(validate_diagnose_options(None, &[], &[(0, 2)], Some(2)).is_err());
        assert!(validate_diagnose_options(None, &[], &[(0, 1)], Some(2)).is_ok());
    }

    #[test]
    fn initial_pose_model_is_mapped_by_stem_and_validated_against_camera() {
        let root =
            std::env::temp_dir().join(format!("visloc_initial_pose_model_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("temporary model directory");
        std::fs::write(
            root.join("cameras.txt"),
            "# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n0 PINHOLE 1600 1066 879.4 879.4 803.4 532.6\n",
        )
        .expect("camera model");
        std::fs::write(
            root.join("images.txt"),
            concat!(
                "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n",
                "1 1 0 0 0 0 0 0 0 DSC_0001.png\n",
                "0 0 -1\n",
                "2 1 0 0 0 1 2 3 0 DSC_0002.png\n",
                "0 0 -1\n",
            ),
        )
        .expect("partial image model");
        let camera = Camera::pinhole(0, 1600, 1066, 879.4, 879.4, 803.4, 532.6);
        let names = vec![
            "DSC_0002.png".to_owned(),
            "DSC_0003.png".to_owned(),
            "DSC_0001.png".to_owned(),
        ];
        let poses = initial_poses_from_colmap_images_txt(&root.join("images.txt"), &names, &camera)
            .expect("valid partial model");
        assert!(
            poses[0].is_some(),
            "matching is by image stem, not row order"
        );
        assert!(poses[1].is_none(), "unseeded loaded images stay None");
        assert!(poses[2].is_some());
        assert_eq!(
            poses[0].as_ref().unwrap().world_to_camera.translation,
            Vector3::new(1.0, 2.0, 3.0)
        );
        let unknown = vec!["DSC_0002.png".to_owned(), "DSC_0009.png".to_owned()];
        assert!(
            initial_poses_from_colmap_images_txt(&root.join("images.txt"), &unknown, &camera,)
                .is_err()
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn colmap_track_membership_maps_rows_and_skips_same_image_points() {
        let root = std::env::temp_dir().join(format!(
            "visloc_colmap_track_membership_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("temporary model directory");
        std::fs::write(
            root.join("images.txt"),
            concat!(
                "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n",
                "1 1 0 0 0 0 0 0 0 DSC_0001.png\n",
                "0 0 -1 1 1 -1 2 2 -1\n",
                "2 1 0 0 0 1 0 0 0 DSC_0002.png\n",
                "0 0 -1 1 1 -1 2 2 -1\n",
                "3 1 0 0 0 2 0 0 0 DSC_0003.png\n",
                "0 0 -1 1 1 -1 2 2 -1\n",
            ),
        )
        .expect("COLMAP image manifest");
        std::fs::write(
            root.join("points3D.txt"),
            concat!(
                "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n",
                "1 0 0 5 1 2 3 0.1 1 0 2 1 3 2\n",
                "2 0 0 5 1 2 3 0.1 1 0 1 1\n",
            ),
        )
        .expect("COLMAP point membership");
        let features = (0..3)
            .map(|_| {
                FeatureSet::new(
                    vec![
                        Point2::new(0.0, 0.0),
                        Point2::new(1.0, 1.0),
                        Point2::new(2.0, 2.0),
                    ],
                    vec![vec![1.0], vec![2.0], vec![3.0]],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let names = vec![
            "DSC_0001.png".to_owned(),
            "DSC_0002.png".to_owned(),
            "DSC_0003.png".to_owned(),
        ];
        let membership =
            parse_colmap_track_membership(&root.join("points3D.txt"), &names, &features)
                .expect("valid observation-only membership");
        assert_eq!(
            membership,
            ColmapTrackMembership {
                tracks: vec![vec![(0, 0), (1, 1), (2, 2)]],
                source_points: 2,
                source_observations: 5,
                retained_observations: 3,
                skipped_conflicting_points: 1,
                skipped_conflicting_observations: 2,
            }
        );

        std::fs::write(
            root.join("points3D_duplicate.txt"),
            concat!(
                "1 0 0 5 1 2 3 0.1 1 0 2 1 3 2\n",
                "2 0 0 5 1 2 3 0.1 1 0 2 1 3 2\n",
            ),
        )
        .expect("duplicate observation membership");
        let error =
            parse_colmap_track_membership(&root.join("points3D_duplicate.txt"), &names, &features)
                .expect_err("an observation cannot belong to two points");
        assert!(error.contains("more than one point"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn pair_stem_window_is_strict_deterministic_and_default_off() {
        let names = vec![
            "DSC_0001.png".to_owned(),
            "DSC_0003.png".to_owned(),
            "DSC_0004.png".to_owned(),
            "DSC_0010.png".to_owned(),
        ];
        let pairs = vec![(0, 1), (0, 2), (1, 2), (2, 3)];
        assert_eq!(
            filter_pairs_by_stem_window(pairs.clone(), &names, Some(2)).unwrap(),
            vec![(0, 1), (1, 2)]
        );
        assert_eq!(
            filter_pairs_by_stem_window(pairs.clone(), &names, None).unwrap(),
            pairs,
            "omitting the flag must preserve the candidate stream exactly"
        );

        let parsed = parse_args_from(minimal_args(&["--pair-stem-window", "3"])).unwrap();
        assert_eq!(parsed.pair_stem_window, Some(3));
        assert!(parse_args_from(minimal_args(&["--pair-stem-window", "0"])).is_err());
        assert!(parse_args_from(minimal_args(&["--pair-stem-window", "NaN"])).is_err());
        assert!(parse_args_from(minimal_args(&["--pair-stem-window"])).is_err());
        assert!(parse_args_from(minimal_args(&[]))
            .unwrap()
            .pair_stem_window
            .is_none());

        assert!(filter_pairs_by_stem_window(
            vec![(0, 1)],
            &["left.png".to_owned(), "right.png".to_owned()],
            Some(2),
        )
        .is_err());
        assert!(filter_pairs_by_stem_window(
            vec![(0, 1)],
            &["DSC_0001.png".to_owned(), "other_0001.png".to_owned()],
            Some(2),
        )
        .is_err());

        let imported = vec![
            ImportedVerifiedPair {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
                config: ConfigurationType::Calibrated,
                essential_matrix: None,
            },
            ImportedVerifiedPair {
                image_i: 0,
                image_j: 3,
                matches: vec![(0, 0)],
                config: ConfigurationType::Calibrated,
                essential_matrix: None,
            },
        ];
        let imported_filtered =
            filter_imported_verified_pairs_by_stem_window(imported, &names, Some(2)).unwrap();
        assert_eq!(imported_filtered.len(), 1);
        assert_eq!(
            (imported_filtered[0].image_i, imported_filtered[0].image_j),
            (0, 1)
        );
    }

    #[test]
    fn sequence_fallback_appends_only_missing_consecutive_candidates() {
        let names = vec![
            "DSC_0001.png".to_owned(),
            "DSC_0002.png".to_owned(),
            "DSC_0003.png".to_owned(),
            "DSC_0005.png".to_owned(),
        ];
        let mut pairs = vec![(0, 1), (0, 3)];
        assert_eq!(
            super::append_consecutive_stem_candidates(&mut pairs, &names).unwrap(),
            1
        );
        assert_eq!(pairs, vec![(0, 1), (0, 3), (1, 2)]);

        let before = pairs.clone();
        assert_eq!(
            super::append_consecutive_stem_candidates(&mut pairs, &names).unwrap(),
            0
        );
        assert_eq!(pairs, before, "the opt-in augmentation is deterministic");
        assert!(super::append_consecutive_stem_candidates(
            &mut Vec::new(),
            &["left.png".to_owned(), "right.png".to_owned()]
        )
        .is_err());
    }

    #[test]
    fn canonical_feature_order_is_deterministic_and_reorders_alternate_bank() {
        let make_features = |reverse: bool| {
            let (keypoints, descriptors) = if reverse {
                (
                    vec![Point2::new(10.0, 0.0), Point2::new(20.0, 0.0)],
                    vec![vec![10.0], vec![20.0]],
                )
            } else {
                (
                    vec![Point2::new(20.0, 0.0), Point2::new(10.0, 0.0)],
                    vec![vec![20.0], vec![10.0]],
                )
            };
            let alternate = if reverse {
                vec![vec![100.0], vec![200.0]]
            } else {
                vec![vec![200.0], vec![100.0]]
            };
            (
                vec![FeatureSet::new(keypoints, descriptors).unwrap()],
                vec![Some(alternate)],
            )
        };
        let (mut first, mut first_alternate) = make_features(false);
        let (mut second, mut second_alternate) = make_features(true);
        let first_map = canonicalize_feature_order(&mut first, &mut first_alternate).unwrap();
        let second_map = canonicalize_feature_order(&mut second, &mut second_alternate).unwrap();
        assert_eq!(first, second);
        assert_eq!(first_alternate, second_alternate);
        assert_eq!(first_map[0], vec![1, 0]);
        assert_eq!(second_map[0], vec![0, 1]);
    }

    #[test]
    fn native_keypoint_sidecar_reorders_without_copying_descriptors() {
        let mut native = vec![vec![Point2::new(100.0, 1.0), Point2::new(200.0, 2.0)]];
        remap_feature_keypoints_by_old_to_new(&mut native, &[vec![1, 0]]).unwrap();
        assert_eq!(
            native[0],
            vec![Point2::new(200.0, 2.0), Point2::new(100.0, 1.0)]
        );

        let mut output = vec![FeatureSet::new(
            vec![Point2::new(10.0, 1.0), Point2::new(20.0, 2.0)],
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
        )
        .unwrap()];
        let descriptor_storage = output[0].descriptors.as_ptr();
        let descriptors = output[0].descriptors.clone();
        replace_feature_keypoints_from_native(&mut output, &[0], &native).unwrap();
        assert_eq!(output[0].keypoints, native[0]);
        assert_eq!(output[0].descriptors, descriptors);
        assert_eq!(output[0].descriptors.as_ptr(), descriptor_storage);
    }

    #[test]
    fn orientation_locus_canonicalization_collapses_variants_and_keeps_best_distance() {
        let features = vec![
            FeatureSet::new(
                vec![
                    Point2::new(10.0, 10.0),
                    Point2::new(10.0, 10.0),
                    Point2::new(20.0, 20.0),
                ],
                vec![vec![0.0], vec![1.0], vec![5.0]],
            )
            .unwrap(),
            FeatureSet::new(
                vec![
                    Point2::new(11.0, 11.0),
                    Point2::new(11.0, 11.0),
                    Point2::new(21.0, 21.0),
                ],
                vec![vec![1.0], vec![1.0], vec![5.0]],
            )
            .unwrap(),
        ];
        let metadata = vec![
            Some(vec![
                FeatureLocusMetadata {
                    x: 10.0,
                    y: 10.0,
                    scale: 2.0,
                    orientation: 0.0,
                },
                FeatureLocusMetadata {
                    x: 10.0,
                    y: 10.0,
                    scale: 2.0,
                    orientation: 1.0,
                },
                FeatureLocusMetadata {
                    x: 20.0,
                    y: 20.0,
                    scale: 3.0,
                    orientation: 0.0,
                },
            ]),
            Some(vec![
                FeatureLocusMetadata {
                    x: 11.0,
                    y: 11.0,
                    scale: 2.0,
                    orientation: 0.0,
                },
                FeatureLocusMetadata {
                    x: 11.0,
                    y: 11.0,
                    scale: 2.0,
                    orientation: 1.0,
                },
                FeatureLocusMetadata {
                    x: 21.0,
                    y: 21.0,
                    scale: 3.0,
                    orientation: 0.0,
                },
            ]),
        ];
        let mut pair = PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1), (2, 2)]);
        let stats =
            canonicalize_pairwise_loci(&features, &metadata, std::slice::from_mut(&mut pair), None)
                .unwrap();
        assert_eq!(stats.metadata_images, 2);
        assert_eq!(stats.physical_loci, 4);
        assert_eq!(stats.collapsed_rows, 2);
        assert_eq!(stats.input_matches, 3);
        assert_eq!(stats.output_matches, 2);
        assert_eq!(stats.deduplicated_matches, 1);
        assert_eq!(
            pair.matches.len(),
            2,
            "two orientations must not create duplicate endpoint pairs"
        );
        assert_eq!(descriptor_squared_distance(&[1.0], &[1.0]), 0.0);
        assert!(
            match_candidate_cmp(&(1, 1, 0.0, None), &(0, 0, 1.0, None)) == CmpOrdering::Less,
            "the lower descriptor distance must win before stable tie-breaks"
        );
        assert!(
            pair.matches.iter().any(|&(i, j)| (i, j) == (2, 2)),
            "different scales at the same image location remain separate loci"
        );
    }

    #[test]
    fn orientation_locus_canonicalization_is_permutation_invariant_and_default_noop() {
        let make = |reverse: bool| {
            let rows = [
                (Point2::new(10.0, 10.0), vec![0.0], 2.0, 0.0),
                (Point2::new(10.0, 10.0), vec![1.0], 2.0, 1.0),
                (Point2::new(20.0, 20.0), vec![5.0], 3.0, 0.0),
            ];
            let order = if reverse {
                vec![2, 1, 0]
            } else {
                vec![0, 1, 2]
            };
            let keypoints: Vec<Point2<f64>> = order.iter().map(|&i| rows[i].0).collect();
            let descriptors: Vec<Vec<f32>> = order.iter().map(|&i| rows[i].1.clone()).collect();
            let metadata: Vec<FeatureLocusMetadata> = order
                .iter()
                .map(|&i| FeatureLocusMetadata {
                    x: rows[i].0.x,
                    y: rows[i].0.y,
                    scale: rows[i].2,
                    orientation: rows[i].3,
                })
                .collect();
            (
                vec![
                    FeatureSet::new(keypoints.clone(), descriptors.clone()).unwrap(),
                    FeatureSet::new(keypoints, descriptors).unwrap(),
                ],
                vec![Some(metadata.clone()), Some(metadata)],
                PairwiseMatches::new(0, 1, order.iter().map(|&i| (i, i)).collect()),
            )
        };
        let (first_features, first_metadata, mut first_pair) = make(false);
        let (second_features, second_metadata, mut second_pair) = make(true);
        canonicalize_pairwise_loci(
            &first_features,
            &first_metadata,
            std::slice::from_mut(&mut first_pair),
            None,
        )
        .unwrap();
        canonicalize_pairwise_loci(
            &second_features,
            &second_metadata,
            std::slice::from_mut(&mut second_pair),
            None,
        )
        .unwrap();
        let physical = |features: &[FeatureSet], pair: &PairwiseMatches| {
            pair.matches
                .iter()
                .map(|&(i, j)| {
                    (
                        features[pair.image_i].keypoints[i],
                        features[pair.image_j].keypoints[j],
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            physical(&first_features, &first_pair),
            physical(&second_features, &second_pair),
            "canonical output must depend on physical loci, not source row order"
        );

        let legacy_features = vec![
            FeatureSet::new(
                vec![Point2::new(1.0, 1.0), Point2::new(1.0, 1.0)],
                vec![vec![0.0], vec![1.0]],
            )
            .unwrap(),
            FeatureSet::new(
                vec![Point2::new(2.0, 2.0), Point2::new(2.0, 2.0)],
                vec![vec![0.0], vec![1.0]],
            )
            .unwrap(),
        ];
        let legacy_pair = PairwiseMatches::new(0, 1, vec![(1, 1), (0, 0)]);
        let mut unchanged = vec![legacy_pair.clone()];
        let stats =
            canonicalize_pairwise_loci(&legacy_features, &[None, None], &mut unchanged, None)
                .unwrap();
        assert_eq!(unchanged, vec![legacy_pair]);
        assert_eq!(stats.metadata_images, 0);
        assert_eq!(stats.physical_loci, 0);
        assert_eq!(stats.collapsed_rows, 0);
        assert_eq!(stats.input_matches, 2);
        assert_eq!(stats.output_matches, 2);
        assert_eq!(stats.deduplicated_matches, 0);
        assert_eq!(
            stats.changed_pairs, 0,
            "metadata-free files must retain legacy identity"
        );
    }

    #[test]
    fn union_traversal_controls_preserve_edge_multiset_and_default_identity() {
        let original = vec![
            PairwiseMatches::new(2, 0, vec![(4, 5), (3, 2)]),
            PairwiseMatches::new(0, 1, vec![(7, 8)]),
        ];
        let hash = unordered_pairwise_edge_hash(&original);

        let mut unchanged = original.clone();
        apply_union_traversal_order(&mut unchanged, UnionTraversalOrder::Original);
        assert_eq!(unchanged, original);

        for order in [
            UnionTraversalOrder::ReversePairs,
            UnionTraversalOrder::ReverseMatches,
            UnionTraversalOrder::ReverseBoth,
        ] {
            let mut reordered = original.clone();
            apply_union_traversal_order(&mut reordered, order);
            assert_eq!(unordered_pairwise_edge_hash(&reordered), hash);
            let original_edges: Vec<_> = original
                .iter()
                .flat_map(|pair| pair.matches.iter().copied())
                .collect();
            let reordered_edges: Vec<_> = reordered
                .iter()
                .flat_map(|pair| pair.matches.iter().copied())
                .collect();
            assert_eq!(
                original_edges.len(),
                reordered_edges.len(),
                "{order:?} must preserve correspondence count"
            );
        }

        let mut reversed_pairs = original.clone();
        apply_union_traversal_order(&mut reversed_pairs, UnionTraversalOrder::ReversePairs);
        assert_eq!(reversed_pairs[0], original[1]);
        let mut reversed_matches = original.clone();
        apply_union_traversal_order(&mut reversed_matches, UnionTraversalOrder::ReverseMatches);
        assert_eq!(reversed_matches[0].matches, vec![(3, 2), (4, 5)]);
        assert!(
            parse_args_from(minimal_args(&["--union-traversal-order", "not-a-mode",])).is_err()
        );
    }

    #[test]
    fn physical_hash_traversal_is_seeded_and_preserves_the_edge_multiset() {
        let features = (0..3)
            .map(|image| {
                let keypoints = (0..10)
                    .map(|keypoint| {
                        Point2::new(
                            (image * 100 + keypoint * 7) as f64,
                            (image * 10 + keypoint * 3) as f64,
                        )
                    })
                    .collect();
                let descriptors = (0..10).map(|value| vec![value as f32]).collect();
                FeatureSet::new(keypoints, descriptors).unwrap()
            })
            .collect::<Vec<_>>();
        let original = vec![
            PairwiseMatches::new(2, 0, vec![(4, 5), (3, 2)]),
            PairwiseMatches::new(0, 1, vec![(7, 8)]),
        ];
        let hash = unordered_pairwise_edge_hash(&original);
        let mut first = original.clone();
        apply_union_traversal_order_with_features(
            &mut first,
            UnionTraversalOrder::PhysicalHash(17),
            &features,
        );
        assert_eq!(unordered_pairwise_edge_hash(&first), hash);
        assert_eq!(
            UnionTraversalOrder::PhysicalHash(17).as_string(),
            "physical-hash:17"
        );
        assert_eq!(
            "physical-hash:0x11".parse::<UnionTraversalOrder>().unwrap(),
            UnionTraversalOrder::PhysicalHash(17)
        );
        assert_eq!(
            "physical-hash-reverse:17"
                .parse::<UnionTraversalOrder>()
                .unwrap(),
            UnionTraversalOrder::PhysicalHashReverse(17)
        );
        let parsed = parse_args_from(minimal_args(&[
            "--union-traversal-order",
            "physical-hash:0x11",
        ]))
        .unwrap();
        assert_eq!(
            parsed.union_traversal_order,
            UnionTraversalOrder::PhysicalHash(17)
        );
        assert!("physical-hash:".parse::<UnionTraversalOrder>().is_err());
        assert!("physical-hash:nope".parse::<UnionTraversalOrder>().is_err());

        let mut second = original.clone();
        apply_union_traversal_order_with_features(
            &mut second,
            UnionTraversalOrder::PhysicalHash(17),
            &features,
        );
        assert_eq!(
            first, second,
            "same physical hash seed must be deterministic"
        );
        let mut different_seed = original.clone();
        apply_union_traversal_order_with_features(
            &mut different_seed,
            UnionTraversalOrder::PhysicalHash(18),
            &features,
        );
        assert_ne!(
            first, different_seed,
            "the seed must parameterize the physical traversal order"
        );
        let mut descending = original.clone();
        apply_union_traversal_order_with_features(
            &mut descending,
            UnionTraversalOrder::PhysicalHashReverse(17),
            &features,
        );
        assert_eq!(unordered_pairwise_edge_hash(&descending), hash);
    }

    #[test]
    fn verified_oracle_map_normalizes_pair_and_keeps_config() {
        let imported = vec![ImportedVerifiedPair {
            image_i: 3,
            image_j: 1,
            matches: vec![(0, 0), (1, 1)],
            config: ConfigurationType::Calibrated,
            essential_matrix: None,
        }];
        let oracle = verified_pair_oracle_map(&imported);
        assert_eq!(
            oracle.get(&(1, 3)),
            Some(&super::VerifiedPairOracle {
                inliers: 2,
                config: ConfigurationType::Calibrated,
            })
        );
    }

    #[test]
    fn model_cross_validation_holdout_is_stable_and_bucket_metrics_are_bounded() {
        let first = model_cross_validation_is_held_out(3, 7, 11, 19);
        assert_eq!(first, model_cross_validation_is_held_out(3, 7, 11, 19));
        assert_ne!(
            first,
            model_cross_validation_is_held_out(3, 7, 11, 20),
            "the deterministic partition should not collapse nearby match keys"
        );
        let pixel_i = Point2::new(12.3456, 78.9012);
        let pixel_j = Point2::new(98.7654, 32.1098);
        assert_eq!(
            model_cross_validation_is_held_out_for_pixels(3, 7, &pixel_i, &pixel_j),
            model_cross_validation_is_held_out_for_pixels(3, 7, &pixel_i, &pixel_j)
        );
        let mut bucket = ModelCrossValidationBucket::default();
        bucket.record(Some(0.01), 0.02, true, true, true);
        bucket.record(Some(0.03), 0.02, true, false, false);
        bucket.record(None, 0.02, false, false, false);
        let summary = summarize_model_cross_validation_bucket(&mut bucket);
        assert_eq!(summary.observations, 3);
        assert_eq!(summary.residual_samples, 2);
        assert_eq!(summary.under_threshold, 1);
        assert_eq!(summary.triangulated, 2);
        assert_eq!(summary.positive_depth, 1);
        assert_eq!(summary.angle_ge_one_degree, 1);
        assert!((summary.under_fraction - 0.5).abs() < 1.0e-12);
        assert!((summary.positive_fraction - 0.5).abs() < 1.0e-12);
        assert!((summary.angle_fraction - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn model_cross_validation_selection_score_requires_shared_calibrated_references() {
        let mut summary = ModelCrossValidationSummary::default();
        summary.pair_balanced_rotation_disagreement_deg = 2.5;
        assert!(model_cross_validation_selection_score(&summary).is_none());
        summary.rotation_reference_pairs = 3;
        assert_eq!(model_cross_validation_selection_score(&summary), Some(2.5));
        summary.pair_balanced_rotation_disagreement_deg = f64::NAN;
        assert!(model_cross_validation_selection_score(&summary).is_none());
    }

    #[test]
    fn calibrated_reference_direction_uses_cheirality_and_stability_gates() {
        let forward = Vector3::new(1.0, 0.0, 0.0);
        let opposite = Vector3::new(-1.0, 0.0, 0.0);
        assert!(translation_direction_delta_deg(&forward, &forward) < 1.0e-9);
        assert!(translation_direction_delta_deg(&forward, &opposite) > 179.9);

        let quality = EssentialPairQuality {
            best_cheirality: 80,
            second_cheirality: 40,
            cheirality_ratio: 0.8,
            mean_sampson: 0.01,
            rotation_quaternion: [1.0, 0.0, 0.0, 0.0],
            center_direction: [1.0, 0.0, 0.0],
            angle_samples: 80,
            angle_ge_1deg: 70,
            angle_p10_deg: 1.2,
            angle_p25_deg: 2.0,
            angle_median_deg: 4.0,
            depth_ratio_p10: 0.5,
            depth_ratio_p25: 0.7,
            depth_ratio_median: 1.0,
        };
        assert!(imported_reference_quality_is_strong(&quality, 3, 12.0));
        assert!(!imported_reference_quality_is_strong(&quality, 1, 12.0));
        assert!(!imported_reference_quality_is_strong(&quality, 3, 20.0001));
        assert!(!imported_reference_quality_is_strong(&quality, 3, f64::NAN));

        // The Huber location must not be pulled to a single angular outlier;
        // this is the robust pair-balanced statistic used by the diagnostic.
        let robust = robust_huber_mean(&[1.0, 1.1, 0.9, 1.2, 100.0]);
        assert!(robust.is_finite());
        assert!(robust < 5.0, "robust location was {robust}");
    }

    #[test]
    fn geometry_conflict_recovery_flag_is_default_off_and_parseable() {
        let defaults = parse_args_from(minimal_args(&[])).unwrap();
        assert!(!defaults.geometry_guided_conflict_recovery);
        assert!(!defaults.post_refinement_registration);
        assert!(!defaults.sequence_relative_pose_fallback);
        assert!(!defaults.sequence_fallback_after_post);
        assert!(!defaults.sequence_constant_velocity_scale);
        assert!(!defaults.sequence_relaxed_constant_velocity_scale);
        assert!(!defaults.sequence_fallback_carry_scale);
        assert!(!defaults.refine_uncalibrated_f_to_essential);
        assert!(!defaults.strict_uncalibrated_f_to_essential);
        assert!(!defaults.calibrated_essential_primary);
        assert_eq!(defaults.sift_descriptor_magnification, 8.0);
        assert!(!defaults.sift_scale_adaptive_gradients);
        assert!(!defaults.sift_vlfeat_compatible_descriptor);
        assert!(!defaults.sift_vlfeat_compatible_detector);
        assert!(!defaults.sift_vlfeat_bilinear_orientations);
        assert!(!defaults.sift_vlfeat_compatible_output_order);
        assert!(!defaults.sift_colmap_compatible_grayscale);
        assert!(!defaults.sift_split_colmap_detector_grayscale);
        assert_eq!(defaults.sift_append_descriptor_magnification, None);
        assert!(!defaults.sift_standard_orientations);
        assert_eq!(defaults.sift_extra_contrast_threshold, None);
        assert!(!defaults.sift_extra_matches_append_only);
        assert!(!defaults.orientation_locus_canonicalization);
        assert!(!defaults.incremental_correspondence_triangulation);
        assert!(!defaults.confidence_ordered_tracks);
        assert!(!defaults.geometric_confidence_tracks);
        assert!(!defaults.stable_track_order);
        assert!(!defaults.cycle_supported_tracks);
        assert!(!defaults.canonical_feature_order);
        assert_eq!(
            defaults.union_traversal_order,
            UnionTraversalOrder::Original
        );
        assert!(!defaults.geometry_weighted_ba);
        assert!(!defaults.freeze_ill_conditioned_landmarks);
        assert_eq!(defaults.landmark_ba_warm_start_iterations, 0);
        assert_eq!(defaults.landmark_ba_warm_start_min_registered_images, 0);
        assert_eq!(defaults.ba_max_iterations, None);
        assert_eq!(defaults.ba_huber_delta, None);
        assert_eq!(defaults.final_min_track_length, None);
        assert_eq!(
            IncrementalSfmConfig::default().ba_config.robust_kernel,
            RobustKernel::Huber { delta: 3.0 }
        );
        assert_eq!(defaults.periodic_ba_min_registered_images, 0);
        assert_eq!(defaults.final_ba_polish_iterations, 0);
        assert_eq!(defaults.diagnose_ba_oracle_poses_file, None);
        assert_eq!(defaults.diagnose_fixed_rotation_ba, None);
        assert_eq!(defaults.diagnose_model_score_file, None);
        assert_eq!(defaults.initial_poses_file, None);
        assert_eq!(defaults.diagnose_colmap_track_membership, None);
        assert!(!defaults.pose_guided_track_splitting);
        assert!(!defaults.pose_guided_track_splitting_graph_support);
        assert!(!defaults.pose_guided_track_splitting_bridge_cuts);
        assert!(!defaults.pose_guided_track_merging);
        assert_eq!(defaults.pose_guided_merge_max_reproj, None);
        assert_eq!(defaults.pose_guided_split_max_reproj, None);
        assert_eq!(defaults.pose_guided_track_splitting_iterations, None);
        assert_eq!(defaults.seed_pair, None);

        let seed_pair = parse_args_from(minimal_args(&["--seed-pair", "9,8"])).unwrap();
        assert_eq!(seed_pair.seed_pair, Some((8, 9)));
        assert!(parse_args_from(minimal_args(&["--seed-pair", "8,8"])).is_err());
        assert!(parse_args_from(minimal_args(&["--seed-pair", "8-9"])).is_err());

        let edge_scales = parse_args_from(minimal_args(&[
            "--mapper",
            "global",
            "--global-independent-edge-scales",
        ]))
        .unwrap();
        assert!(edge_scales.global_independent_edge_scales);
        assert!(parse_args_from(minimal_args(&["--global-independent-edge-scales"])).is_err());

        let enabled = parse_args_from(minimal_args(&[
            "--geometry-guided-conflict-recovery",
            "--post-refinement-registration",
            "--sequence-relative-pose-fallback",
            "--sequence-fallback-after-post",
            "--sequence-constant-velocity-scale",
            "--guided-matching",
            "--colmap-guided-matching",
            "--refine-uncalibrated-f-to-essential",
            "--strict-uncalibrated-f-to-essential",
            "--calibrated-essential-primary",
            "--sift-extra-keypoints-stems",
            "DSC_0299,DSC_0306",
            "--sift-extra-keypoints",
            "2048",
            "--sift-extra-contrast-threshold",
            "0.01",
            "--sift-extra-matches-append-only",
            "--orientation-locus-canonicalization",
            "--incremental-correspondence-triangulation",
            "--sift-descriptor-magnification",
            "3.0",
            "--sift-scale-adaptive-gradients",
            "--sift-standard-orientations",
            "--sift-append-descriptor-magnification",
            "3.0",
            "--confidence-ordered-tracks",
            "--geometric-confidence-tracks",
            "--stable-track-order",
            "--cycle-supported-tracks",
            "--canonical-feature-order",
            "--union-traversal-order",
            "reverse-both",
            "--geometry-weighted-ba",
            "--freeze-ill-conditioned-landmarks",
            "--landmark-ba-warm-start-iterations",
            "3",
            "--landmark-ba-warm-start-min-registered-images",
            "27",
        ]))
        .unwrap();
        assert!(enabled.geometry_guided_conflict_recovery);
        assert!(enabled.post_refinement_registration);
        assert!(enabled.sequence_relative_pose_fallback);
        assert!(enabled.sequence_fallback_after_post);
        assert!(enabled.sequence_constant_velocity_scale);
        assert!(!enabled.sequence_relaxed_constant_velocity_scale);
        assert!(enabled.guided_matching);
        assert!(enabled.colmap_guided_matching);
        assert!(parse_args_from(minimal_args(&["--sequence-constant-velocity-scale",])).is_err());
        let relaxed = parse_args_from(minimal_args(&[
            "--sequence-relative-pose-fallback",
            "--sequence-relaxed-constant-velocity-scale",
        ]))
        .unwrap();
        assert!(relaxed.sequence_relative_pose_fallback);
        assert!(relaxed.sequence_relaxed_constant_velocity_scale);
        assert!(!relaxed.sequence_constant_velocity_scale);
        let carried = parse_args_from(minimal_args(&[
            "--post-refinement-registration",
            "--sequence-relative-pose-fallback",
            "--sequence-fallback-after-post",
            "--sequence-relaxed-constant-velocity-scale",
            "--sequence-fallback-carry-scale",
        ]))
        .unwrap();
        assert!(carried.sequence_fallback_carry_scale);
        assert!(parse_args_from(minimal_args(&[
            "--sequence-relative-pose-fallback",
            "--sequence-fallback-carry-scale",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&[
            "--post-refinement-registration",
            "--sequence-relative-pose-fallback",
            "--sequence-fallback-after-post",
            "--sequence-fallback-carry-scale",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&[
            "--sequence-relaxed-constant-velocity-scale",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&[
            "--sequence-relative-pose-fallback",
            "--sequence-constant-velocity-scale",
            "--sequence-relaxed-constant-velocity-scale",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&[
            "--sequence-relative-pose-fallback",
            "--sequence-fallback-after-post",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&[
            "--post-refinement-registration",
            "--sequence-fallback-after-post",
        ]))
        .is_err());
        let composed = parse_args_from(minimal_args(&[
            "--geometry-guided-conflict-recovery",
            "--pose-guided-track-splitting",
            "--pose-guided-split-max-reproj",
            "1.0",
        ]))
        .unwrap();
        assert!(composed.geometry_guided_conflict_recovery);
        assert!(composed.pose_guided_track_splitting);
        assert!(!composed.pose_guided_track_splitting_bridge_cuts);
        assert_eq!(composed.pose_guided_split_max_reproj, Some(1.0));
        let bridge_cuts = parse_args_from(minimal_args(&[
            "--pose-guided-track-splitting",
            "--pose-guided-track-splitting-bridge-cuts",
        ]))
        .unwrap();
        assert!(bridge_cuts.pose_guided_track_splitting_bridge_cuts);
        assert!(parse_args_from(minimal_args(
            &["--pose-guided-track-splitting-bridge-cuts",]
        ))
        .is_err());
        let final_track_gate =
            parse_args_from(minimal_args(&["--final-min-track-length", "3"])).unwrap();
        assert_eq!(final_track_gate.final_min_track_length, Some(3));
        for invalid_length in ["0", "2", "4"] {
            assert!(
                parse_args_from(minimal_args(&["--final-min-track-length", invalid_length]))
                    .is_err(),
                "unsupported final track length {invalid_length:?} was accepted"
            );
        }
        assert!(parse_args_from(minimal_args(&[
            "--final-min-track-length",
            "3",
            "--no-final-ba",
        ]))
        .is_err());
        let merging = parse_args_from(minimal_args(&[
            "--pose-guided-track-splitting",
            "--pose-guided-track-merging",
        ]))
        .unwrap();
        assert!(merging.pose_guided_track_merging);
        assert_eq!(merging.pose_guided_merge_max_reproj, None);
        assert!(parse_args_from(minimal_args(&["--pose-guided-track-merging"])).is_err());
        let merging_gate = parse_args_from(minimal_args(&[
            "--pose-guided-track-splitting",
            "--pose-guided-track-merging",
            "--pose-guided-merge-max-reproj",
            "4.0",
        ]))
        .unwrap();
        assert_eq!(merging_gate.pose_guided_merge_max_reproj, Some(4.0));
        assert!(parse_args_from(minimal_args(&[
            "--pose-guided-track-splitting",
            "--pose-guided-merge-max-reproj",
            "4.0",
        ]))
        .is_err());
        for invalid_merge_gate in ["0", "-1", "NaN", "inf"] {
            assert!(
                parse_args_from(minimal_args(&[
                    "--pose-guided-track-splitting",
                    "--pose-guided-track-merging",
                    "--pose-guided-merge-max-reproj",
                    invalid_merge_gate,
                ]))
                .is_err(),
                "invalid merge gate {invalid_merge_gate:?} was accepted"
            );
        }
        let pose_split = parse_args_from(minimal_args(&[
            "--pose-guided-track-splitting",
            "--pose-guided-track-splitting-graph-support",
        ]))
        .unwrap();
        assert!(pose_split.pose_guided_track_splitting);
        assert!(pose_split.pose_guided_track_splitting_graph_support);
        let split_gate = parse_args_from(minimal_args(&[
            "--pose-guided-track-splitting",
            "--pose-guided-split-max-reproj",
            "1.0",
        ]))
        .unwrap();
        assert_eq!(split_gate.pose_guided_split_max_reproj, Some(1.0));
        let split_iterations = parse_args_from(minimal_args(&[
            "--pose-guided-track-splitting",
            "--pose-guided-track-splitting-iterations",
            "2",
        ]))
        .unwrap();
        assert_eq!(
            split_iterations.pose_guided_track_splitting_iterations,
            Some(2)
        );
        assert!(
            parse_args_from(minimal_args(&["--pose-guided-split-max-reproj", "1.0",])).is_err()
        );
        assert!(parse_args_from(minimal_args(&[
            "--pose-guided-track-splitting-iterations",
            "2",
        ]))
        .is_err());
        for invalid_iterations in ["0", "9"] {
            assert!(
                parse_args_from(minimal_args(&[
                    "--pose-guided-track-splitting",
                    "--pose-guided-track-splitting-iterations",
                    invalid_iterations,
                ]))
                .is_err(),
                "invalid pose-guided split iterations {invalid_iterations:?} was accepted"
            );
        }
        for invalid_gate in ["0", "-1", "NaN", "inf"] {
            assert!(
                parse_args_from(minimal_args(&[
                    "--pose-guided-track-splitting",
                    "--pose-guided-split-max-reproj",
                    invalid_gate,
                ]))
                .is_err(),
                "invalid pose-guided split gate {invalid_gate:?} was accepted"
            );
        }
        assert!(parse_args_from(minimal_args(&[
            "--pose-guided-track-splitting-graph-support",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&[
            "--pose-guided-track-splitting",
            "--colmap-style",
        ]))
        .is_err());
        let membership = parse_args_from(minimal_args(&[
            "--diagnose-colmap-track-membership",
            "/tmp/model/points3D.txt",
        ]))
        .unwrap();
        assert_eq!(
            membership.diagnose_colmap_track_membership,
            Some(PathBuf::from("/tmp/model/points3D.txt"))
        );
        assert!(parse_args_from(minimal_args(&[
            "--diagnose-colmap-track-membership",
            "/tmp/model/points3D.txt",
            "--mapper",
            "global",
        ]))
        .is_err());
        let ba_enabled = parse_args_from(minimal_args(&["--ba-max-iterations", "40"])).unwrap();
        assert_eq!(ba_enabled.ba_max_iterations, Some(40));
        assert!(parse_args_from(minimal_args(&["--ba-max-iterations", "0"])).is_err());
        let global_ba_rounds =
            parse_args_from(minimal_args(&["--global-ba-max-refinements", "0"])).unwrap();
        assert_eq!(global_ba_rounds.global_ba_max_refinements, Some(0));
        let global_ba_rounds_explicit =
            parse_args_from(minimal_args(&["--global-ba-max-refinements", "3"])).unwrap();
        assert_eq!(global_ba_rounds_explicit.global_ba_max_refinements, Some(3));
        for invalid_rounds in ["-1", "not-a-number"] {
            assert!(
                parse_args_from(minimal_args(&[
                    "--global-ba-max-refinements",
                    invalid_rounds
                ]))
                .is_err(),
                "invalid global BA refinement cap {invalid_rounds:?} was accepted"
            );
        }
        assert!(parse_args_from(minimal_args(&["--global-ba-max-refinements"])).is_err());
        let huber_enabled = parse_args_from(minimal_args(&["--ba-huber-delta", "1.0"])).unwrap();
        assert_eq!(huber_enabled.ba_huber_delta, Some(1.0));
        for invalid_delta in ["0", "-1", "NaN", "inf"] {
            assert!(
                parse_args_from(minimal_args(&["--ba-huber-delta", invalid_delta])).is_err(),
                "invalid Huber delta {invalid_delta:?} was accepted"
            );
        }
        assert!(parse_args_from(minimal_args(&["--ba-huber-delta"])).is_err());
        let sparse_solver =
            parse_args_from(minimal_args(&["--ba-linear-solver", "sparse"])).unwrap();
        assert_eq!(sparse_solver.ba_linear_solver, Some(LinearSolver::Sparse));
        let dense_solver = parse_args_from(minimal_args(&["--ba-linear-solver", "dense"])).unwrap();
        assert_eq!(dense_solver.ba_linear_solver, Some(LinearSolver::Dense));
        for invalid_solver in ["", "foo", "DENSE"] {
            assert!(
                parse_args_from(minimal_args(&["--ba-linear-solver", invalid_solver])).is_err(),
                "invalid BA linear solver {invalid_solver:?} was accepted"
            );
        }
        assert!(parse_args_from(minimal_args(&["--ba-linear-solver"])).is_err());
        let periodic_deferred =
            parse_args_from(minimal_args(&["--periodic-ba-min-registered-images", "32"])).unwrap();
        assert_eq!(periodic_deferred.periodic_ba_min_registered_images, 32);
        let polish_enabled =
            parse_args_from(minimal_args(&["--final-ba-polish-iterations", "10"])).unwrap();
        assert_eq!(polish_enabled.final_ba_polish_iterations, 10);
        let polish_disabled =
            parse_args_from(minimal_args(&["--final-ba-polish-iterations", "0"])).unwrap();
        assert_eq!(polish_disabled.final_ba_polish_iterations, 0);
        let oracle_probe = parse_args_from(minimal_args(&[
            "--diagnose-ba-oracle-poses",
            "/tmp/oracle/images.txt",
        ]))
        .unwrap();
        assert_eq!(
            oracle_probe.diagnose_ba_oracle_poses_file,
            Some(Path::new("/tmp/oracle/images.txt").to_path_buf())
        );
        let fixed_rotation =
            parse_args_from(minimal_args(&["--diagnose-fixed-rotation-ba", "current"])).unwrap();
        assert_eq!(
            fixed_rotation.diagnose_fixed_rotation_ba,
            Some("current".to_owned())
        );
        assert!(parse_args_from(minimal_args(&["--diagnose-fixed-rotation-ba"])).is_err());
        let model_score = parse_args_from(minimal_args(&[
            "--diagnose-model-score",
            "/tmp/model/images.txt",
        ]))
        .unwrap();
        assert_eq!(
            model_score.diagnose_model_score_file,
            Some(Path::new("/tmp/model/images.txt").to_path_buf())
        );
        assert!(parse_args_from(minimal_args(&["--diagnose-model-score"])).is_err());
        let initial_poses = parse_args_from(minimal_args(&[
            "--initial-poses",
            "/tmp/partial-model/images.txt",
        ]))
        .unwrap();
        assert_eq!(
            initial_poses.initial_poses_file,
            Some(Path::new("/tmp/partial-model/images.txt").to_path_buf())
        );
        assert!(parse_args_from(minimal_args(&["--initial-poses"])).is_err());
        assert!(parse_args_from(minimal_args(&["--initial-poses", "",])).is_err());
        assert!(parse_args_from(minimal_args(&[
            "--initial-poses",
            "/tmp/partial-model/images.txt",
            "--seed-pair",
            "8,9",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&[
            "--initial-poses",
            "/tmp/partial-model/images.txt",
            "--mapper",
            "global",
        ]))
        .is_err());
        assert!(enabled.refine_uncalibrated_f_to_essential);
        assert!(enabled.strict_uncalibrated_f_to_essential);
        assert!(enabled.calibrated_essential_primary);
        assert_eq!(enabled.sift_descriptor_magnification, 3.0);
        assert!(enabled.sift_scale_adaptive_gradients);
        assert_eq!(enabled.sift_append_descriptor_magnification, Some(3.0));
        assert!(enabled.sift_standard_orientations);
        assert!(enabled.confidence_ordered_tracks);
        assert!(enabled.geometric_confidence_tracks);
        assert!(enabled.stable_track_order);
        assert!(enabled.cycle_supported_tracks);
        assert!(enabled.canonical_feature_order);
        assert_eq!(
            enabled.union_traversal_order,
            UnionTraversalOrder::ReverseBoth
        );
        assert!(enabled.geometry_weighted_ba);
        assert!(enabled.freeze_ill_conditioned_landmarks);
        assert_eq!(enabled.landmark_ba_warm_start_iterations, 3);
        assert_eq!(enabled.landmark_ba_warm_start_min_registered_images, 27);
        assert_eq!(
            enabled.sift_extra_keypoints_stems,
            vec!["DSC_0299".to_owned(), "DSC_0306".to_owned()]
        );
        assert_eq!(enabled.sift_extra_keypoints, 2048);
        assert_eq!(enabled.sift_extra_contrast_threshold, Some(0.01));
        assert!(enabled.sift_extra_matches_append_only);
        assert!(enabled.orientation_locus_canonicalization);
        assert!(enabled.incremental_correspondence_triangulation);
        assert!(parse_args_from(minimal_args(&[
            "--incremental-correspondence-triangulation",
            "--colmap-style",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&[
            "--incremental-correspondence-triangulation",
            "--mapper",
            "global",
        ]))
        .is_err());
        let dsp = parse_args_from(minimal_args(&[
            "--sift-dsp",
            "--sift-vlfeat-compatible-descriptor",
        ]))
        .unwrap();
        assert!(dsp.sift_dsp);
        assert_eq!(dsp.sift_dsp_num_scales, 15);
        assert!(parse_args_from(minimal_args(&["--sift-dsp"])).is_err());
        assert!(parse_args_from(minimal_args(&[
            "--sift-dsp",
            "--sift-vlfeat-compatible-descriptor",
            "--sift-dsp-num-scales",
            "0",
        ]))
        .is_err());
        let rounded =
            parse_args_from(minimal_args(&["--sift-colmap-compatible-grayscale"])).unwrap();
        assert!(rounded.sift_colmap_compatible_grayscale);
        let split = parse_args_from(minimal_args(&[
            "--sift-vlfeat-compatible-detector",
            "--sift-vlfeat-compatible-descriptor",
            "--sift-split-colmap-detector-grayscale",
        ]))
        .unwrap();
        assert!(split.sift_split_colmap_detector_grayscale);
        assert!(
            parse_args_from(minimal_args(&["--sift-split-colmap-detector-grayscale",])).is_err()
        );
        assert!(parse_args_from(minimal_args(&[
            "--sift-vlfeat-compatible-detector",
            "--sift-vlfeat-compatible-descriptor",
            "--sift-colmap-compatible-grayscale",
            "--sift-split-colmap-detector-grayscale",
        ]))
        .is_err());
        assert!(
            parse_args_from(minimal_args(&["--sift-extra-contrast-threshold", "-0.01",])).is_err()
        );
        assert!(
            parse_args_from(minimal_args(&["--sift-extra-contrast-threshold", "NaN",])).is_err()
        );
        assert!(parse_args_from(minimal_args(&["--sift-descriptor-magnification", "0",])).is_err());
        assert!(
            parse_args_from(minimal_args(&["--sift-descriptor-magnification", "NaN",])).is_err()
        );
        assert!(parse_args_from(minimal_args(&[
            "--sift-append-descriptor-magnification",
            "0",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&[
            "--sift-append-descriptor-magnification",
            "NaN",
        ]))
        .is_err());
        assert!(
            parse_args_from(minimal_args(&[
                "--sift-vlfeat-compatible-descriptor",
                "--sift-descriptor-magnification",
                "3.0",
            ]))
            .unwrap()
            .sift_vlfeat_compatible_descriptor
        );
        assert!(parse_args_from(minimal_args(&[
            "--sift-vlfeat-compatible-descriptor",
            "--sift-scale-adaptive-gradients",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&[
            "--sift-vlfeat-compatible-descriptor",
            "--sift-descriptor-magnification",
            "8.0",
        ]))
        .is_err());
        assert!(
            parse_args_from(minimal_args(&["--sift-vlfeat-compatible-detector"]))
                .unwrap()
                .sift_vlfeat_compatible_detector
        );
        let source_order = parse_args_from(minimal_args(&[
            "--sift-vlfeat-compatible-detector",
            "--sift-vlfeat-compatible-output-order",
        ]))
        .unwrap();
        assert!(source_order.sift_vlfeat_compatible_output_order);
        let bilinear = parse_args_from(minimal_args(&[
            "--sift-vlfeat-compatible-detector",
            "--sift-vlfeat-bilinear-orientations",
        ]))
        .unwrap();
        assert!(bilinear.sift_vlfeat_bilinear_orientations);
        assert!(parse_args_from(minimal_args(&[
            "--sift-vlfeat-compatible-detector",
            "--sift-affine",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&[
            "--sift-vlfeat-compatible-detector",
            "--sift-standard-orientations",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&["--sift-vlfeat-bilinear-orientations"])).is_err());
        assert!(parse_args_from(minimal_args(&["--sift-vlfeat-compatible-output-order"])).is_err());
        assert!(parse_args_from(minimal_args(&["--colmap-guided-matching"])).is_err());
    }

    #[test]
    fn verified_pair_snapshot_flags_are_explicit_and_mutually_exclusive() {
        let exported = parse_args_from(minimal_args(&[
            "--export-verified-pairs-snapshot",
            "/tmp/pairs.vps",
        ]))
        .unwrap();
        assert_eq!(
            exported.export_verified_pairs_snapshot,
            Some(PathBuf::from("/tmp/pairs.vps"))
        );
        let imported = parse_args_from(minimal_args(&[
            "--import-verified-pairs-snapshot",
            "/tmp/pairs.vps",
        ]))
        .unwrap();
        assert_eq!(
            imported.import_verified_pairs_snapshot,
            Some(PathBuf::from("/tmp/pairs.vps"))
        );
        assert!(parse_args_from(minimal_args(&[
            "--import-verified-pairs-file",
            "/tmp/legacy.txt",
            "--import-verified-pairs-snapshot",
            "/tmp/pairs.vps",
        ]))
        .is_err());
        assert!(parse_args_from(minimal_args(&["--import-verified-pairs-snapshot", "",])).is_err());
        let export_only = parse_args_from(minimal_args(&[
            "--export-verified-pairs-snapshot",
            "/tmp/pairs.vps",
            "--export-verified-pairs-only",
        ]))
        .unwrap();
        assert!(export_only.export_verified_pairs_only);
        assert!(parse_args_from(minimal_args(&["--export-verified-pairs-only"])).is_err());
    }

    #[test]
    fn snapshot_keypoints_only_is_opt_in_and_rejects_unsafe_combinations() {
        let defaults = parse_args_from(minimal_args(&[])).unwrap();
        assert!(!defaults.snapshot_keypoints_only);
        let enabled = parse_args_from(minimal_args(&[
            "--import-verified-pairs-snapshot",
            "/tmp/pairs.vps",
            "--snapshot-keypoints-only",
        ]))
        .unwrap();
        assert!(enabled.snapshot_keypoints_only);

        for extra in [
            vec!["--snapshot-keypoints-only", "--feature-extractor", "sift"],
            vec!["--snapshot-keypoints-only", "--mapper", "global"],
            vec!["--snapshot-keypoints-only", "--mapper", "hybrid"],
            vec!["--snapshot-keypoints-only", "--colmap-style"],
            vec![
                "--snapshot-keypoints-only",
                "--snapshot-coordinate-override-dir",
                "/tmp/override",
            ],
            vec![
                "--snapshot-keypoints-only",
                "--export-features-dir",
                "/tmp/features",
            ],
            vec!["--snapshot-keypoints-only", "--export-features-only"],
            vec![
                "--snapshot-keypoints-only",
                "--export-verified-pairs-snapshot",
                "/tmp/other.vps",
            ],
            vec!["--snapshot-keypoints-only", "--canonical-feature-order"],
            vec![
                "--snapshot-keypoints-only",
                "--orientation-locus-canonicalization",
            ],
            vec![
                "--snapshot-keypoints-only",
                "--diagnose-model-score",
                "/tmp/model/images.txt",
            ],
            vec!["--snapshot-keypoints-only", "--stable-track-order"],
        ] {
            let mut args = vec!["--import-verified-pairs-snapshot", "/tmp/pairs.vps"];
            args.extend(extra);
            assert!(
                parse_args_from(minimal_args(&args)).is_err(),
                "unsafe snapshot-keypoints-only combination was accepted: {args:?}"
            );
        }
    }

    #[test]
    fn snapshot_keypoints_only_loader_matches_full_feature_hash_and_geometry_shape() {
        let root = std::env::temp_dir().join(format!(
            "visloc_snapshot_keypoints_only_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("a_features.txt"),
            "# feature file\n1.0 2.0 0.9 0.0 -0.0 0.25\n3.0 4.0 0.8 1.0 2.0 3.0\n",
        )
        .unwrap();
        std::fs::write(root.join("b_features.txt"), "5.0 6.0 0.7 -1.0 4.0 8.0\n").unwrap();

        let (full, names, _) = load_images(&root, "_features.txt", ".png").unwrap();
        let loaded = load_images_keypoints_only(&root, "_features.txt", ".png").unwrap();
        assert_eq!(loaded.image_names, names);
        assert_eq!(loaded.features.len(), full.len());
        assert!(loaded
            .features
            .iter()
            .all(|set| set.keypoints.len() == set.descriptors.len()
                && set.descriptors.iter().all(Vec::is_empty)));
        let validation = snapshot_feature_validation_from_files(
            &loaded.paths,
            &loaded.features,
            &loaded.fingerprints,
        )
        .unwrap();
        assert_eq!(
            validation.feature_counts,
            full.iter().map(FeatureSet::len).collect::<Vec<_>>()
        );
        assert_eq!(
            validation.feature_manifest_hash,
            snapshot_feature_manifest_hash(&full)
        );

        // Per-image calibration changes only the in-memory keypoints; the
        // descriptor re-read must still produce the same exact stream as a
        // full feature bank carrying those transformed keypoints.
        let mut calibrated = loaded.features;
        calibrated[0].keypoints[0].x += 0.125;
        let mut full_calibrated = full.clone();
        full_calibrated[0].keypoints[0].x += 0.125;
        assert_eq!(
            snapshot_feature_validation_from_files(
                &loaded.paths,
                &calibrated,
                &loaded.fingerprints,
            )
            .unwrap()
            .feature_manifest_hash,
            snapshot_feature_manifest_hash(&full_calibrated)
        );

        // Empty descriptor rows remain a valid feature shape for the camera
        // rig, preserving the mapper's row-index geometry contract.
        let rig = PerImageCameras::new(vec![
            Camera::pinhole(0, 100, 100, 50.0, 50.0, 50.0, 50.0),
            Camera::pinhole(1, 100, 100, 50.0, 50.0, 50.0, 50.0),
        ])
        .unwrap();
        rig.validate_features(&calibrated).unwrap();

        std::fs::write(
            root.join("a_features.txt"),
            "# feature file\n1.0 2.0 0.9 9.0 -0.0 0.25\n3.0 4.0 0.8 1.0 2.0 3.0\n",
        )
        .unwrap();
        let error = snapshot_feature_validation_from_files(
            &loaded.paths,
            &calibrated,
            &loaded.fingerprints,
        )
        .unwrap_err();
        assert!(error.contains("changed between loads"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn snapshot_coordinate_override_requires_import_and_preserves_default() {
        let default_args = parse_args_from(minimal_args(&[])).unwrap();
        assert!(default_args.snapshot_coordinate_override_dir.is_none());
        assert!(parse_args_from(minimal_args(&[
            "--snapshot-coordinate-override-dir",
            "/tmp/subpixel",
        ]))
        .is_err());
        let args = parse_args_from(minimal_args(&[
            "--import-verified-pairs-snapshot",
            "/tmp/pairs.vps",
            "--snapshot-coordinate-override-dir",
            "/tmp/subpixel",
        ]))
        .unwrap();
        assert_eq!(
            args.snapshot_coordinate_override_dir,
            Some(PathBuf::from("/tmp/subpixel"))
        );
        assert!(parse_args_from(minimal_args(&[
            "--import-verified-pairs-snapshot",
            "/tmp/pairs.vps",
            "--snapshot-coordinate-override-dir",
            "/tmp/subpixel",
            "--export-verified-pairs-snapshot",
            "/tmp/other.vps",
        ]))
        .is_err());
    }

    #[test]
    fn snapshot_coordinate_override_is_descriptor_exact_and_coordinate_only() {
        let names = vec!["left.png".to_owned(), "right.png".to_owned()];
        let mut base = vec![
            FeatureSet::new(
                vec![Point2::new(1.0, 2.0), Point2::new(3.0, 4.0)],
                vec![vec![1.0, -0.0], vec![f32::from_bits(0x7fc0_0042), 5.0]],
            )
            .unwrap(),
            FeatureSet::new(vec![Point2::new(5.0, 6.0)], vec![vec![7.0, 8.0]]).unwrap(),
        ];
        let replacement = vec![
            FeatureSet::new(
                vec![Point2::new(1.25, 2.5), Point2::new(3.0, 4.0)],
                vec![vec![1.0, -0.0], vec![f32::from_bits(0x7fc0_0042), 5.0]],
            )
            .unwrap(),
            FeatureSet::new(vec![Point2::new(5.0, 6.5)], vec![vec![7.0, 8.0]]).unwrap(),
        ];
        let before_descriptor_bits = base
            .iter()
            .map(|features| {
                features
                    .descriptors
                    .iter()
                    .map(|descriptor| descriptor.iter().map(|value| value.to_bits()).collect())
                    .collect::<Vec<Vec<u32>>>()
            })
            .collect::<Vec<_>>();
        let descriptor_bits = |sets: &[FeatureSet]| {
            sets.iter()
                .map(|features| {
                    features
                        .descriptors
                        .iter()
                        .map(|descriptor| descriptor.iter().map(|value| value.to_bits()).collect())
                        .collect::<Vec<Vec<u32>>>()
                })
                .collect::<Vec<_>>()
        };
        let stats =
            apply_snapshot_coordinate_override(&mut base, &names, &replacement, &names).unwrap();
        assert_eq!(stats.images, 2);
        assert_eq!(stats.rows, 3);
        assert_eq!(stats.changed_rows, 2);
        assert_eq!(base[0].keypoints, replacement[0].keypoints);
        assert_eq!(base[1].keypoints, replacement[1].keypoints);
        assert_eq!(descriptor_bits(&base), before_descriptor_bits);

        let mut changed_descriptor = replacement.clone();
        changed_descriptor[0].descriptors[0][0] = 1.5;
        let error = apply_snapshot_coordinate_override(
            &mut base.clone(),
            &names,
            &changed_descriptor,
            &names,
        )
        .unwrap_err();
        assert!(error.contains("descriptor/index mismatch"));
        let error = apply_snapshot_coordinate_override(
            &mut base.clone(),
            &names,
            &replacement,
            &["right.png".to_owned(), "left.png".to_owned()],
        )
        .unwrap_err();
        assert!(error.contains("image names/order"));
    }

    fn guided_report(
        config: ConfigurationType,
        essential: Option<Matrix3<f64>>,
        fundamental: Option<Matrix3<f64>>,
        homography: Option<Matrix3<f64>>,
    ) -> TwoViewGeometryReport {
        TwoViewGeometryReport {
            config,
            inliers: vec![0],
            essential,
            fundamental,
            homography,
            relative_pose: None,
            essential_inliers: vec![0],
            e_inlier_count: 1,
            f_inlier_count: 1,
            h_inlier_count: 1,
        }
    }

    fn guided_camera() -> Camera {
        Camera::pinhole(0, 640, 480, 100.0, 100.0, 0.0, 0.0)
    }

    #[test]
    fn colmap_guided_geometry_uses_reported_e_f_or_h_model() {
        let identity = Matrix3::identity();
        assert!(matches!(
            colmap_guided_geometry(&guided_report(
                ConfigurationType::Calibrated,
                Some(identity),
                None,
                None,
            )),
            Some(super::ColmapGuidedGeometry::Essential(_))
        ));
        assert!(matches!(
            colmap_guided_geometry(&guided_report(
                ConfigurationType::Uncalibrated,
                None,
                Some(identity),
                None,
            )),
            Some(super::ColmapGuidedGeometry::Fundamental(_))
        ));
        assert!(matches!(
            colmap_guided_geometry(&guided_report(
                ConfigurationType::Planar,
                None,
                None,
                Some(identity),
            )),
            Some(super::ColmapGuidedGeometry::Homography(_))
        ));
        assert!(colmap_guided_geometry(&guided_report(
            ConfigurationType::Multiple,
            Some(identity),
            Some(identity),
            Some(identity),
        ))
        .is_none());
    }

    #[test]
    fn colmap_guided_epipolar_gate_admits_only_geometrically_valid_candidate() {
        // This F has horizontal epipolar lines: equal y coordinates have zero
        // algebraic/Sampson residual, while the distractor is outside the
        // pixel-unit guided gate.  Its descriptor is otherwise equally good.
        let fundamental = Matrix3::new(
            0.0, 0.0, 0.0, //
            0.0, 0.0, -1.0, //
            0.0, 1.0, 0.0,
        );
        let features_i =
            FeatureSet::new(vec![Point2::new(10.0, 20.0)], vec![vec![1.0, 0.0]]).unwrap();
        let features_j = FeatureSet::new(
            vec![Point2::new(30.0, 20.0), Point2::new(30.0, 24.0)],
            vec![vec![1.0, 0.0], vec![1.0, 0.0]],
        )
        .unwrap();
        let extras = colmap_guided_matches(
            &guided_camera(),
            &features_i,
            &features_j,
            &[],
            &guided_report(
                ConfigurationType::Uncalibrated,
                None,
                Some(fundamental),
                None,
            ),
            1.0,
            0.9,
            true,
        );
        assert_eq!(
            extras
                .iter()
                .map(|m| (m.query_index, m.train_index))
                .collect::<Vec<_>>(),
            vec![(0, 0)]
        );
    }

    #[test]
    fn colmap_guided_cross_check_is_unique_and_initial_matches_are_preserved() {
        let features_i = FeatureSet::new(
            vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .unwrap();
        let features_j = FeatureSet::new(
            vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .unwrap();
        let initial = vec![DescriptorMatch {
            query_index: 0,
            train_index: 0,
            distance: 0.0,
            second_best_distance: None,
            ratio: None,
            confidence: None,
        }];
        let before = initial.clone();
        let extras = colmap_guided_matches(
            &guided_camera(),
            &features_i,
            &features_j,
            &initial,
            &guided_report(
                ConfigurationType::Planar,
                None,
                None,
                Some(Matrix3::identity()),
            ),
            1.0,
            0.9,
            true,
        );
        // Query 1 has a distinct descriptor/geometry-consistent endpoint, so
        // the append-only helper may add it while retaining the baseline.
        assert_eq!(
            extras
                .iter()
                .map(|m| (m.query_index, m.train_index))
                .collect::<Vec<_>>(),
            vec![(1, 1)]
        );
        assert_eq!(initial, before);
        let mut expanded = initial.clone();
        expanded.extend(extras);
        assert_eq!(expanded.len(), 2);
        assert_eq!((expanded[0].query_index, expanded[0].train_index), (0, 0));
    }

    #[test]
    fn effective_config_snapshot_is_stable_and_experimental_defaults_are_off() {
        let first = parse_args_from(minimal_args(&[])).unwrap();
        let second = parse_args_from(minimal_args(&[])).unwrap();
        let first_snapshot = effective_config_snapshot(&first);
        let second_snapshot = effective_config_snapshot(&second);
        assert_eq!(first_snapshot, second_snapshot);
        assert_eq!(
            effective_config_hash(&first_snapshot),
            effective_config_hash(&second_snapshot)
        );
        assert_eq!(
            effective_config_hash(""),
            0xcbf29ce484222325,
            "the snapshot label uses the stable FNV-1a offset basis"
        );
        let mut path_a = parse_args_from(minimal_args(&[])).unwrap();
        let mut path_b = parse_args_from(minimal_args(&[])).unwrap();
        path_a.out_colmap = PathBuf::from("/tmp/electro-repeat-a/model");
        path_b.out_colmap = PathBuf::from("/tmp/electro-repeat-b/model");
        assert_ne!(
            effective_config_snapshot(&path_a),
            effective_config_snapshot(&path_b)
        );
        assert_eq!(
            snapshot_export_config(&path_a),
            snapshot_export_config(&path_b)
        );

        // Every experimental mapper/verification switch is opt-in. Values
        // such as the ordinary matcher, mapper, and final BA are checked here
        // too so an omitted flag cannot silently select a newer path.
        assert!(!first.refine_intrinsics);
        assert!(!first.refine_distortion);
        assert!(!first.colmap_style);
        assert!(!first.final_iterative_global_refinement);
        assert_eq!(first.global_ba_max_refinements, None);
        assert!(!first.post_refinement_registration);
        assert!(!first.structureless_registration);
        assert!(!first.guided_matching);
        assert!(!first.colmap_guided_matching);
        assert!(!first.multiple_models);
        assert!(first.min_e_f_inlier_ratio.is_none());
        assert!(!first.calibrated_prefer_essential);
        assert!(!first.refine_uncalibrated_f_to_essential);
        assert!(!first.strict_uncalibrated_f_to_essential);
        assert!(!first.calibrated_essential_primary);
        assert!(!first.prefer_essential_inliers);
        assert!(!first.prefer_essential_free_endpoints);
        assert!(first.prefer_essential_stems.is_empty());
        assert!(first.prefer_essential_pairs.is_empty());
        assert!(!first.require_essential_selected_edges);
        assert!(first.require_essential_stems.is_empty());
        assert!(first.rematch_stems.is_empty());
        assert!(!first.rematch_guided);
        assert!(!first.rematch_free_vs_priors);
        assert!(!first.rematch_tracks_use_essential);
        assert_eq!(first.rematch_prefer_min_e_inliers, 0);
        assert!(first.rematch_prefer_strong_stems.is_empty());
        assert!((first.rematch_min_chirality_margin - 0.0).abs() < f64::EPSILON);
        assert!(!first.rematch_prior_anchor);
        assert!(first.rematch_min_e_f_inlier_ratio.is_none());
        assert!(!first.rematch_calibrated_prefer_essential);
        assert!(!first.rematch_prior_ray_guided);
        assert_eq!(first.rematch_prior_ray_min_rays, 2);
        assert_eq!(first.rematch_prior_ray_min_e_inliers, 25);
        assert!(first.rematch_verification_mode.is_none());
        assert!(!first.rematch_pose_guided_after_global);
        assert!((first.rematch_max_gt_bearing_deg - 0.0).abs() < f64::EPSILON);
        assert!(first.rematch_gt_bearing_path.is_none());
        assert!(first.rematch_guided_max_error_px.is_none());
        assert!(first.rematch_guided_lowe_ratio.is_none());
        assert!(!first.rematch_require_calibrated);
        assert!((first.rematch_max_mean_sampson - 0.0).abs() < f64::EPSILON);
        assert!((first.essential_edge_weight_boost - 1.0).abs() < f64::EPSILON);
        assert!(!first.force_essential_matches);
        assert!((first.force_essential_min_ef_ratio - 0.7).abs() < f64::EPSILON);
        assert_eq!(first.force_essential_min_e_inliers, 0);
        assert!(!first.force_essential_uncalibrated_only);
        assert!(!first.repnp_free_from_priors);
        assert_eq!(first.repnp_free_min_corrs, 0);
        assert!(!first.repnp_seed_free_as_priors);
        assert!(!first.repair_prior_edges);
        assert!(!first.repair_free_edges_from_solved);
        assert!(!first.drop_free_edges_antipodal);
        assert!(!first.prior_guided_free_chirality);
        assert!(!first.metric_prior_chirality_edges);
        assert!(!first.metric_prior_scale);
        assert!(!first.chirality_harden);
        assert!(!first.refine_global_translations);
        assert!(!first.global_independent_edge_scales);
        assert!(!first.multi_hypothesis_edges);
        assert!(!first.weight_by_chirality_margin);
        assert!(!first.hybrid_filter_priors);
        assert!(!first.hybrid_drop_inconsistent_priors);
        assert!(!first.verify_registration_two_view);
        assert!(!first.hybrid_rotation_priors_only);
        assert!(!first.joint_global_positioning);
        assert!(!first.calibrated_view_edges_only);
        assert!(!first.filter_images);
        assert!(!first.confidence_ordered_tracks);
        assert!(!first.geometric_confidence_tracks);
        assert!(!first.stable_track_order);
        assert!(!first.cycle_supported_tracks);
        assert!(!first.canonical_feature_order);
        assert_eq!(first.union_traversal_order, UnionTraversalOrder::Original);
        assert!(!first.geometry_guided_conflict_recovery);
        assert!(!first.rescue_bridging);
        assert!(!first.rescue_cross_check);
        assert!(first.diagnose_pairs.is_empty());
        assert!(first.diagnose_pairs_csv.is_none());
        assert!(first.diagnose_pair_stems.is_empty());
        assert!(matches!(first.matcher, super::MatcherKind::Nn));
        assert!(matches!(first.mapper, super::MapperKind::Incremental));
        assert!(matches!(first.pair_source, super::PairSource::Vlad));
        assert!(matches!(first.track_source, super::TrackSource::UnionFind));
        assert!(!first.export_features_only);
        assert!(!first.sift_stream_export);
        assert!(!first.sift_stream_resume);
        assert!(first.import_matches_file.is_none());
        assert!(first.import_matches_supplement_file.is_none());
        assert!(!first.sift_stream_export);
        assert!(first.import_verified_pairs_file.is_none());
        assert!(first.export_verified_pairs_snapshot.is_none());
        assert!(first.import_verified_pairs_snapshot.is_none());
        assert!(first.snapshot_coordinate_override_dir.is_none());
        assert!(first.diagnose_ba_oracle_poses_file.is_none());
        assert!(first.diagnose_fixed_rotation_ba.is_none());
        assert!(first.ba_max_iterations.is_none());
        assert_eq!(first.periodic_ba_min_registered_images, 0);
        assert!(first.pair_stem_window.is_none());
        assert_eq!(first.final_ba_polish_iterations, 0);
        assert!(!first.geometry_weighted_ba);
        assert!(!first.freeze_ill_conditioned_landmarks);
        assert_eq!(first.landmark_ba_warm_start_iterations, 0);
        assert_eq!(first.landmark_ba_warm_start_min_registered_images, 0);
        assert!(!first.sift_affine);
        assert!(!first.sift_multi_anisotropy);
        assert!(!first.sift_dsp);
        assert!(!first.sift_l1_root);
        assert!(!first.sift_standard_orientations);
        assert!(!first.sift_prefer_larger_scale);
        assert!(!first.sift_full_pyramid);
        assert!(!first.sift_scale_adaptive_gradients);
        assert!(!first.sift_vlfeat_compatible_descriptor);
        assert!(!first.sift_vlfeat_compatible_detector);
        assert!(!first.sift_vlfeat_bilinear_orientations);
        assert!(!first.sift_vlfeat_compatible_output_order);
        assert!(!first.sift_colmap_compatible_grayscale);
        assert!(!first.sift_split_colmap_detector_grayscale);
        assert!(first.sift_append_descriptor_magnification.is_none());
        assert!(first.sift_extra_keypoints_stems.is_empty());
        assert_eq!(first.sift_extra_keypoints, 0);
        assert!(first.sift_extra_contrast_threshold.is_none());
        assert!(!first.sift_extra_matches_append_only);

        let stream = parse_args_from(minimal_args(&[
            "--feature-extractor",
            "sift",
            "--images-dir",
            "/tmp/sift-images",
            "--export-features-dir",
            "/tmp/sift-features",
            "--export-features-only",
            "--sift-stream-export",
            "--sift-stream-resume",
        ]))
        .unwrap();
        assert!(stream.sift_stream_export);
        assert!(stream.sift_stream_resume);
        assert!(parse_args_from(minimal_args(&["--sift-stream-resume"])).is_err());
        assert!(parse_args_from(minimal_args(&["--sift-stream-export"])).is_err());
        assert!(parse_args_from(minimal_args(&[
            "--export-features-dir",
            "/tmp/sift-features",
            "--export-features-only",
            "--sift-stream-export",
        ]))
        .is_err());
    }
}

#[cfg(all(test, feature = "image-io"))]
mod sift_extra_tests {
    use super::{
        append_spatially_novel_keypoints, effective_extra_contrast_threshold,
        extract_sift_with_split_grayscale,
    };
    use visloc_rs::vision::features::sift::{
        describe_sift_keypoints, extract_sift, GrayImage, SiftConfig,
    };
    fn dot_texture(width: usize, height: usize) -> Vec<f32> {
        let mut state = 7u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut pixels = vec![0.15f32; width * height];
        for _ in 0..(width * height / 24) {
            let cx = next() * width as f64;
            let cy = next() * height as f64;
            let bright = 0.55 + 0.35 * next();
            let radius = 1.5 + 2.5 * next();
            let x0 = cx.floor() as i64 - 6;
            let y0 = cy.floor() as i64 - 6;
            for y in y0..=y0 + 12 {
                for x in x0..=x0 + 12 {
                    if x < 0 || y < 0 || x >= width as i64 || y >= height as i64 {
                        continue;
                    }
                    let dx = x as f64 - cx;
                    let dy = y as f64 - cy;
                    let gaussian = (-(dx * dx + dy * dy) / (2.0 * radius * radius)).exp();
                    let index = y as usize * width + x as usize;
                    pixels[index] = (pixels[index] + (bright * gaussian) as f32).min(1.0);
                }
            }
        }
        pixels
    }

    #[test]
    fn extra_threshold_preserves_primary_prefix_and_default_resolution() {
        let width = 128usize;
        let height = 128usize;
        let pixels = dot_texture(width, height);
        let image = GrayImage::new(width, height, &pixels).unwrap();
        let primary_config = SiftConfig {
            max_keypoints: 32,
            octaves: 2,
            contrast_threshold: 0.02,
            ..SiftConfig::default()
        };
        let dense_config = |threshold| SiftConfig {
            max_keypoints: usize::MAX,
            octaves: 2,
            contrast_threshold: threshold,
            ..SiftConfig::default()
        };
        let (primary_keypoints, primary_descriptors) =
            extract_sift(&image, &primary_config).unwrap();
        let (same_keypoints, same_descriptors) = extract_sift(&image, &dense_config(0.02)).unwrap();
        let (low_keypoints, low_descriptors) = extract_sift(&image, &dense_config(0.01)).unwrap();

        let mut merged_same_keypoints = primary_keypoints.clone();
        let mut merged_same_descriptors = primary_descriptors.clone();
        let same_added = append_spatially_novel_keypoints(
            &mut merged_same_keypoints,
            &mut merged_same_descriptors,
            same_keypoints,
            same_descriptors,
            32,
        );
        let mut merged_low_keypoints = primary_keypoints.clone();
        let mut merged_low_descriptors = primary_descriptors.clone();
        let low_added = append_spatially_novel_keypoints(
            &mut merged_low_keypoints,
            &mut merged_low_descriptors,
            low_keypoints,
            low_descriptors,
            32,
        );

        let primary_len = primary_keypoints.len();
        assert_eq!(&merged_same_keypoints[..primary_len], &primary_keypoints);
        assert_eq!(
            &merged_same_descriptors[..primary_len],
            &primary_descriptors
        );
        assert_eq!(&merged_low_keypoints[..primary_len], &primary_keypoints);
        assert_eq!(&merged_low_descriptors[..primary_len], &primary_descriptors);
        assert!(
            low_added >= same_added,
            "low threshold added {low_added}, same threshold added {same_added}"
        );
        assert_eq!(
            effective_extra_contrast_threshold(None, 0.02),
            0.02,
            "omitting the extra threshold must preserve legacy extraction"
        );
        assert_eq!(effective_extra_contrast_threshold(Some(0.01), 0.02), 0.01);
    }

    #[test]
    fn split_grayscale_keeps_rounded_detector_and_floor_descriptors() {
        let width = 128usize;
        let height = 128usize;
        let source = dot_texture(width, height);
        let floor_pixels: Vec<f32> = source
            .iter()
            .map(|&value| ((value.clamp(0.0, 1.0) * 255.0).floor() as u8) as f32 / 255.0)
            .collect();
        let rounded_pixels: Vec<f32> = source
            .iter()
            .map(|&value| ((value.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8) as f32 / 255.0)
            .collect();
        let floor_image = GrayImage::new(width, height, &floor_pixels).unwrap();
        let rounded_image = GrayImage::new(width, height, &rounded_pixels).unwrap();
        let config = SiftConfig {
            max_keypoints: 64,
            octaves: 1,
            max_orientations: 2,
            vlfeat_compatible_detector: true,
            vlfeat_compatible_descriptor: true,
            vlfeat_bilinear_orientations: true,
            ..SiftConfig::default()
        };

        let (split_keypoints, split_descriptors) =
            extract_sift_with_split_grayscale(&rounded_image, &floor_image, &config).unwrap();
        let (expected_keypoints, _) = extract_sift(&rounded_image, &config).unwrap();
        let expected_descriptors =
            describe_sift_keypoints(&floor_image, &expected_keypoints, &config);

        assert_eq!(split_keypoints, expected_keypoints);
        assert_eq!(split_descriptors, expected_descriptors);
        assert_eq!(split_keypoints.len(), split_descriptors.len());
    }
}

#[cfg(test)]
mod append_only_matcher_tests {
    use super::{
        append_only_nn_matches, f_to_e_stability_gate, parse_args_from,
        parse_persistent_match_worker_plan, project_fundamental_to_essential,
        refine_uncalibrated_f_winner, select_calibrated_essential_primary,
        sequence_f_to_e_high_support_override_gate, sequence_f_to_e_stability_gate,
        should_exclude_strict_uncalibrated_f_winner, snapshot_feature_manifest_hash,
        write_verified_pair_snapshot, write_verified_pair_snapshot_atomic,
        FToECandidateDiagnostics, PairMatcher, PairwiseMatches, SnapshotFeatureValidation,
        PERSISTENT_MATCH_WORKER_PLAN_MAGIC,
    };
    use nalgebra::{Matrix3, Point2, Point3, UnitQuaternion, Vector3};
    use std::collections::HashMap;
    use std::path::PathBuf;
    use visloc_rs::vision::two_view::{
        ConfigurationType, TwoViewCorrespondence, TwoViewGeometryReport,
    };
    use visloc_rs::FeatureSet;
    use visloc_rs::{Camera, CameraModel};

    fn minimal_args(extra: &[&str]) -> Vec<String> {
        let mut args = [
            "--width",
            "1600",
            "--height",
            "1066",
            "--fx",
            "879.4",
            "--fy",
            "879.4",
            "--cx",
            "803.4",
            "--cy",
            "532.6",
            "--out-colmap",
            "/tmp/persistent-cli-test",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        args.extend(extra.iter().map(|arg| (*arg).to_owned()));
        args
    }

    fn features(descriptors: Vec<Vec<f32>>) -> FeatureSet {
        let keypoints = (0..descriptors.len())
            .map(|index| Point2::new(index as f64, 0.0))
            .collect();
        FeatureSet::new(keypoints, descriptors).unwrap()
    }

    #[test]
    fn append_only_preserves_primary_match_against_extra_distractor() {
        let query = features(vec![vec![0.0]]);
        // The primary prefix has a decisive 0.1-vs-1.0 Lowe match. The extra
        // 0.05 descriptor wins when matching the full set, replacing it in
        // the ordinary matcher.
        let train = features(vec![vec![0.1], vec![1.0], vec![0.05]]);
        let normal = PairMatcher::Nn.match_pair(0.8, true, 0, 1, &query, &train);
        assert_eq!(normal.len(), 1);
        assert_eq!(normal[0].train_index, 2);

        let append_only = append_only_nn_matches(0.8, true, &query, &train, 1, 2);
        assert_eq!(append_only.len(), 1);
        assert_eq!(append_only[0].query_index, 0);
        assert_eq!(append_only[0].train_index, 0);

        let configured = PairMatcher::NnAppendOnly {
            primary_keypoint_counts: vec![1, 2],
        };
        assert_eq!(
            configured.match_pair(0.8, true, 0, 1, &query, &train),
            append_only
        );
    }

    #[test]
    fn descriptor_ensemble_appends_alternate_only_match_without_replacing_primary() {
        let query = features(vec![vec![0.0], vec![10.0]]);
        let train = features(vec![vec![0.1], vec![100.0]]);
        let primary = PairMatcher::Nn.match_pair(0.8, true, 0, 1, &query, &train);
        assert_eq!(primary.len(), 1);
        assert_eq!((primary[0].query_index, primary[0].train_index), (0, 0));

        let ensemble = PairMatcher::NnDescriptorEnsemble {
            primary_keypoint_counts: None,
            alternate_descriptors: vec![
                Some(vec![vec![0.0], vec![1.0]]),
                Some(vec![vec![0.1], vec![1.1]]),
            ],
        };
        let matches = ensemble.match_pair(0.8, true, 0, 1, &query, &train);
        assert_eq!(matches.len(), 2);
        assert_eq!((matches[0].query_index, matches[0].train_index), (0, 0));
        assert_eq!((matches[1].query_index, matches[1].train_index), (1, 1));
    }

    #[test]
    fn fundamental_projection_uses_k_transpose_f_k_convention() {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 510.0, 320.0, 240.0);
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 510.0, 240.0, 0.0, 0.0, 1.0);
        let rotation = UnitQuaternion::from_euler_angles(0.08, -0.12, 0.17)
            .to_rotation_matrix()
            .into_inner();
        let translation = Vector3::new(0.3, -0.2, 0.7);
        let skew = Matrix3::new(
            0.0,
            -translation.z,
            translation.y,
            translation.z,
            0.0,
            -translation.x,
            -translation.y,
            translation.x,
            0.0,
        );
        let essential = skew * rotation;
        let k_inverse = k.try_inverse().expect("synthetic K is invertible");
        let fundamental = k_inverse.transpose() * essential * k_inverse;
        let projected = project_fundamental_to_essential(&fundamental, &camera)
            .expect("synthetic F must project to E");
        let scale = projected.dot(&essential) / essential.dot(&essential);
        let relative_error = (projected - scale * essential).norm() / essential.norm();
        assert!(relative_error < 1.0e-10, "relative error={relative_error}");
    }

    fn synthetic_uncalibrated_f_winner() -> (Camera, Vec<TwoViewCorrespondence>, Matrix3<f64>) {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 510.0, 320.0, 240.0);
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 510.0, 240.0, 0.0, 0.0, 1.0);
        let rotation = UnitQuaternion::from_euler_angles(0.04, -0.06, 0.03)
            .to_rotation_matrix()
            .into_inner();
        let translation = Vector3::new(0.35, -0.08, 0.12);
        let skew = Matrix3::new(
            0.0,
            -translation.z,
            translation.y,
            translation.z,
            0.0,
            -translation.x,
            -translation.y,
            translation.x,
            0.0,
        );
        let essential = skew * rotation;
        let k_inverse = k.try_inverse().expect("synthetic K is invertible");
        let fundamental = k_inverse.transpose() * essential * k_inverse;
        let points = (0..32)
            .map(|index| {
                let phase = index as f64;
                let x = 0.95 * (phase * 0.71).sin();
                let y = 0.70 * (phase * 1.13).cos();
                let z = 4.0 + 0.35 * (phase * 0.37).sin() + 0.01 * phase;
                Point3::new(x, y, z)
            })
            .collect::<Vec<_>>();
        let correspondences = points
            .iter()
            .map(|point| {
                let moved = rotation * point.coords + translation;
                TwoViewCorrespondence::new(
                    camera.project(point).expect("left point in front"),
                    camera
                        .project(&Point3::new(moved.x, moved.y, moved.z))
                        .expect("right point in front"),
                )
            })
            .collect();
        (camera, correspondences, fundamental)
    }

    #[test]
    fn guarded_f_winner_refinement_recovers_calibrated_inliers() {
        let (camera, correspondences, fundamental) = synthetic_uncalibrated_f_winner();
        let report = TwoViewGeometryReport {
            config: ConfigurationType::Uncalibrated,
            inliers: (0..correspondences.len()).collect(),
            essential: None,
            fundamental: Some(fundamental),
            homography: None,
            relative_pose: None,
            essential_inliers: Vec::new(),
            e_inlier_count: 0,
            f_inlier_count: correspondences.len(),
            h_inlier_count: 0,
        };
        let refinement = refine_uncalibrated_f_winner(&report, &correspondences, &camera, 12)
            .expect("exact calibrated F should produce a robust E_F");
        assert_eq!(
            refinement.inlier_indices,
            (0..correspondences.len()).collect::<Vec<_>>()
        );
        assert_eq!(refinement.f_inlier_count, correspondences.len());
        assert!(refinement.quality.cheirality_ratio >= 0.75);
        assert!(refinement.quality.angle_samples > 0);
    }

    #[test]
    fn calibrated_essential_primary_can_beat_f_support_with_healthy_e() {
        let (camera, correspondences, fundamental) = synthetic_uncalibrated_f_winner();
        let essential = project_fundamental_to_essential(&fundamental, &camera)
            .expect("synthetic F must provide a calibrated E");
        let count = correspondences.len();
        let report = TwoViewGeometryReport {
            config: ConfigurationType::Uncalibrated,
            inliers: (0..count).collect(),
            essential: Some(essential),
            fundamental: Some(fundamental),
            homography: None,
            relative_pose: None,
            essential_inliers: (0..count).collect(),
            e_inlier_count: count,
            f_inlier_count: count + 8,
            h_inlier_count: 0,
        };
        let selected = select_calibrated_essential_primary(&report, &correspondences, &camera, 12)
            .expect("a healthy calibrated E should be selected even when F has more support");
        assert!(report.f_inlier_count > report.e_inlier_count);
        assert_eq!(selected.initial_inlier_count, count);
        assert!(selected.inlier_indices.len() >= count * 4 / 5);
        assert!(selected.quality.mean_sampson.is_finite());
    }

    #[test]
    fn calibrated_essential_primary_rejects_degenerate_evidence() {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 510.0, 320.0, 240.0);
        let correspondences = (0..32)
            .map(|_| {
                TwoViewCorrespondence::new(Point2::new(320.0, 240.0), Point2::new(320.0, 240.0))
            })
            .collect::<Vec<_>>();
        let report = TwoViewGeometryReport {
            config: ConfigurationType::Uncalibrated,
            inliers: (0..correspondences.len()).collect(),
            essential: Some(Matrix3::identity()),
            fundamental: Some(Matrix3::identity()),
            homography: None,
            relative_pose: None,
            essential_inliers: (0..correspondences.len()).collect(),
            e_inlier_count: correspondences.len(),
            f_inlier_count: correspondences.len() + 8,
            h_inlier_count: 0,
        };
        assert!(
            select_calibrated_essential_primary(&report, &correspondences, &camera, 12).is_none(),
            "coincident observations must not create a calibrated primary edge"
        );
    }

    #[test]
    fn guarded_f_winner_refinement_falls_back_for_invalid_weak_or_calibrated_input() {
        let (camera, correspondences, fundamental) = synthetic_uncalibrated_f_winner();
        let base = TwoViewGeometryReport {
            config: ConfigurationType::Uncalibrated,
            inliers: (0..correspondences.len()).collect(),
            essential: None,
            fundamental: Some(fundamental),
            homography: None,
            relative_pose: None,
            essential_inliers: Vec::new(),
            e_inlier_count: 0,
            f_inlier_count: correspondences.len(),
            h_inlier_count: 0,
        };
        let mut invalid = base.clone();
        invalid.fundamental = Some(Matrix3::zeros());
        assert!(refine_uncalibrated_f_winner(&invalid, &correspondences, &camera, 12).is_none());
        assert!(refine_uncalibrated_f_winner(&base, &correspondences[..8], &camera, 12).is_none());
        let mut calibrated = base;
        calibrated.config = ConfigurationType::Calibrated;
        assert!(refine_uncalibrated_f_winner(&calibrated, &correspondences, &camera, 12).is_none());
    }

    #[test]
    fn strict_f_to_e_exclusion_is_opt_in_and_keeps_no_calibration_unchanged() {
        let (camera, correspondences, fundamental) = synthetic_uncalibrated_f_winner();
        let report = TwoViewGeometryReport {
            config: ConfigurationType::Uncalibrated,
            inliers: (0..correspondences.len()).collect(),
            essential: None,
            fundamental: Some(fundamental),
            homography: None,
            relative_pose: None,
            essential_inliers: Vec::new(),
            e_inlier_count: 0,
            f_inlier_count: correspondences.len(),
            h_inlier_count: 0,
        };
        let refinement = refine_uncalibrated_f_winner(&report, &correspondences, &camera, 12)
            .expect("synthetic calibrated F should pass the strict gate");
        assert!(!should_exclude_strict_uncalibrated_f_winner(
            false, &camera, &report, None,
        ));
        assert!(!should_exclude_strict_uncalibrated_f_winner(
            true,
            &camera,
            &report,
            Some(&refinement),
        ));
        assert!(should_exclude_strict_uncalibrated_f_winner(
            true, &camera, &report, None,
        ));

        let mut calibrated = report.clone();
        calibrated.config = ConfigurationType::Calibrated;
        assert!(!should_exclude_strict_uncalibrated_f_winner(
            true,
            &camera,
            &calibrated,
            None,
        ));

        let no_calibration = Camera {
            id: camera.id,
            model: CameraModel::Unknown("NONE".to_owned()),
            width: camera.width,
            height: camera.height,
            params: Vec::new(),
        };
        assert!(!should_exclude_strict_uncalibrated_f_winner(
            true,
            &no_calibration,
            &report,
            None,
        ));
    }

    #[test]
    fn f_to_e_stability_gate_is_strict_and_deterministic() {
        let stable = FToECandidateDiagnostics {
            calibrated_s1: 1.0,
            calibrated_s2: 0.999,
            calibrated_s3: 1.0e-12,
            projection_distortion: 0.001,
            s1_s2_mismatch: 0.002,
            s3_s2_ratio: 1.0e-12,
            f_inliers: 100,
            ef_inliers: 98,
            ef_overlap_on_f: 0.98,
            f_normalized_residual: 0.001,
            ef_normalized_residual_on_f: 0.002,
            ef_to_f_residual_ratio: 2.0,
            cheirality_ratio: 0.95,
            cheirality_margin: 0.9,
            ef_angle_p25_deg: 1.5,
            stable_refits: 3,
            pose_rotation_spread_deg: 1.0,
            pose_translation_spread_deg: 2.0,
        };
        assert!(f_to_e_stability_gate(&stable));
        assert!(f_to_e_stability_gate(&stable));

        let mut unstable_pose = stable;
        unstable_pose.pose_translation_spread_deg = 5.01;
        assert!(!f_to_e_stability_gate(&unstable_pose));
        assert!(sequence_f_to_e_stability_gate(&unstable_pose));
        unstable_pose.pose_translation_spread_deg = 10.01;
        assert!(!sequence_f_to_e_stability_gate(&unstable_pose));
        unstable_pose.pose_translation_spread_deg = 2.0;
        unstable_pose.pose_rotation_spread_deg = 5.01;
        assert!(!sequence_f_to_e_stability_gate(&unstable_pose));
        let mut non_essential = stable;
        non_essential.projection_distortion = 0.0101;
        assert!(!f_to_e_stability_gate(&non_essential));
        let mut weak_overlap = stable;
        weak_overlap.ef_overlap_on_f = 0.899;
        assert!(!f_to_e_stability_gate(&weak_overlap));

        let mut high_support_override = stable;
        high_support_override.f_inliers = 556;
        high_support_override.ef_inliers = 555;
        high_support_override.ef_overlap_on_f = 555.0 / 556.0;
        high_support_override.cheirality_ratio = 555.0 / 555.0;
        high_support_override.cheirality_margin = 1.0;
        high_support_override.ef_angle_p25_deg = 1.528;
        high_support_override.pose_translation_spread_deg = 49.094;
        assert!(!sequence_f_to_e_stability_gate(&high_support_override));
        assert!(sequence_f_to_e_high_support_override_gate(
            &high_support_override
        ));

        let mut poor_override_overlap = high_support_override;
        poor_override_overlap.ef_overlap_on_f = 0.949;
        assert!(!sequence_f_to_e_high_support_override_gate(
            &poor_override_overlap
        ));
        let mut poor_override_cheirality = high_support_override;
        poor_override_cheirality.cheirality_ratio = 0.949;
        assert!(!sequence_f_to_e_high_support_override_gate(
            &poor_override_cheirality
        ));
        let mut poor_override_margin = high_support_override;
        poor_override_margin.cheirality_margin = 0.749;
        assert!(!sequence_f_to_e_high_support_override_gate(
            &poor_override_margin
        ));
        let mut poor_override_manifold = high_support_override;
        poor_override_manifold.projection_distortion = 0.0101;
        assert!(!sequence_f_to_e_high_support_override_gate(
            &poor_override_manifold
        ));
        let mut poor_override_parallax = high_support_override;
        poor_override_parallax.ef_angle_p25_deg = 0.999;
        assert!(!sequence_f_to_e_high_support_override_gate(
            &poor_override_parallax
        ));
    }

    #[test]
    fn persistent_worker_plan_parser_rejects_path_traversal_and_reuse() {
        let root = std::env::temp_dir().join(format!(
            "visloc_persistent_plan_parser_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let hash = "a".repeat(64);
        let valid = format!(
            "{PERSISTENT_MATCH_WORKER_PLAN_MAGIC}\n\
             images 2\n\
             image 0 left.png\n\
             image 1 right.png\n\
             candidate_index_sha256 {hash}\n\
             feature_manifest_sha256 {hash}\n\
             pairs 1\n\
             shards 1\n\
             shard 4 candidates/candidate-000004.txt matches/verified-000004.vps {hash}\n"
        );
        let path = root.join("valid.plan");
        std::fs::write(&path, valid).unwrap();
        let parsed = parse_persistent_match_worker_plan(&path).unwrap();
        assert_eq!(parsed.image_names, ["left.png", "right.png"]);
        assert_eq!(parsed.shards[0].id, 4);
        assert_eq!(parsed.root, root);

        let traversal = root.join("traversal.plan");
        std::fs::write(
            &traversal,
            format!(
                "{PERSISTENT_MATCH_WORKER_PLAN_MAGIC}\n\
                 images 2\nimage 0 left.png\nimage 1 right.png\n\
                 candidate_index_sha256 {hash}\nfeature_manifest_sha256 {hash}\n\
                 pairs 1\nshards 1\n\
                 shard 0 ../candidate.txt matches/out.vps {hash}\n"
            ),
        )
        .unwrap();
        assert!(parse_persistent_match_worker_plan(&traversal)
            .unwrap_err()
            .contains("relative path"));

        let duplicate = root.join("duplicate.plan");
        std::fs::write(
            &duplicate,
            format!(
                "{PERSISTENT_MATCH_WORKER_PLAN_MAGIC}\n\
                 images 2\nimage 0 left.png\nimage 1 right.png\n\
                 candidate_index_sha256 {hash}\nfeature_manifest_sha256 {hash}\n\
                 pairs 2\nshards 2\n\
                 shard 0 candidates/a.txt matches/a.vps {hash}\n\
                 shard 1 candidates/a.txt matches/b.vps {hash}\n"
            ),
        )
        .unwrap();
        assert!(parse_persistent_match_worker_plan(&duplicate)
            .unwrap_err()
            .contains("repeats"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn persistent_worker_cli_is_opt_in_and_fail_closed() {
        let defaults = parse_args_from(minimal_args(&[])).unwrap();
        assert!(defaults.persistent_match_worker_plan.is_none());
        let valid = parse_args_from(vec![
            "--input-colmap-calibration".to_owned(),
            "/tmp/calibration".to_owned(),
            "--verification-mode".to_owned(),
            "full".to_owned(),
            "--persistent-match-worker-plan".to_owned(),
            "/tmp/match-worker.plan".to_owned(),
            "--out-colmap".to_owned(),
            "/tmp/persistent-model".to_owned(),
        ])
        .unwrap();
        assert_eq!(
            valid.persistent_match_worker_plan,
            Some(PathBuf::from("/tmp/match-worker.plan"))
        );

        for extra in [
            vec!["--mapper", "global"],
            vec!["--matcher", "lightglue"],
            vec!["--verification-mode", "legacy"],
            vec!["--guided-matching"],
            vec!["--candidate-manifest", "/tmp/candidate.txt"],
            vec!["--import-verified-pairs-snapshot", "/tmp/pairs.vps"],
            vec!["--canonical-feature-order"],
            vec!["--union-traversal-order", "reverse-both"],
            vec!["--rematch-stems", "foo"],
            vec!["--diagnose-bearing-gt", "/tmp/gt/images.txt"],
            vec!["--global-ba-max-refinements", "1"],
        ] {
            let mut args = vec![
                "--input-colmap-calibration".to_owned(),
                "/tmp/calibration".to_owned(),
                "--verification-mode".to_owned(),
                "full".to_owned(),
                "--persistent-match-worker-plan".to_owned(),
                "/tmp/match-worker.plan".to_owned(),
                "--out-colmap".to_owned(),
                "/tmp/persistent-model".to_owned(),
            ];
            args.extend(extra.iter().map(|arg| (*arg).to_owned()));
            assert!(
                parse_args_from(args).is_err(),
                "persistent worker accepted unsupported options: {extra:?}"
            );
        }
    }

    #[test]
    fn cached_snapshot_feature_validation_is_byte_identical_to_default_writer() {
        let root = std::env::temp_dir().join(format!(
            "visloc_persistent_snapshot_writer_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let features = vec![
            FeatureSet::new(vec![Point2::new(10.0, 20.0)], vec![vec![0.1, 0.2, 0.3]]).unwrap(),
            FeatureSet::new(vec![Point2::new(30.0, 40.0)], vec![vec![0.4, 0.5, 0.6]]).unwrap(),
        ];
        let image_names = vec!["left.png".to_owned(), "right.png".to_owned()];
        let pairwise = vec![PairwiseMatches::new(0, 1, vec![(0, 0)])];
        let metadata = HashMap::new();
        let args = parse_args_from(minimal_args(&[])).unwrap();
        let validation = SnapshotFeatureValidation {
            feature_counts: features.iter().map(FeatureSet::len).collect(),
            feature_manifest_hash: snapshot_feature_manifest_hash(&features),
        };
        write_verified_pair_snapshot(
            &root.join("default.vps"),
            &image_names,
            &features,
            &args.camera,
            &pairwise,
            &metadata,
            &args,
        )
        .unwrap();
        write_verified_pair_snapshot_atomic(
            &root.join("cached.vps"),
            &image_names,
            &features,
            &args.camera,
            &pairwise,
            &metadata,
            &args,
            &validation,
        )
        .unwrap();
        assert_eq!(
            std::fs::read(root.join("default.vps")).unwrap(),
            std::fs::read(root.join("cached.vps")).unwrap()
        );
        let _ = std::fs::remove_dir_all(root);
    }
}

#[cfg(all(test, feature = "image-io"))]
mod sift_stream_tests {
    use super::{
        export_features_to_dir, export_features_to_dir_with_native_keypoints, feature_export_text,
        feature_export_text_with_keypoints, file_fnv1a64, locus_metadata_text,
        sift_stream_manifest_path, stream_export_features_with_loader,
        validate_sift_stream_manifest, write_sift_stream_manifest_atomically, FeatureLocusMetadata,
    };
    use nalgebra::Point2;
    use std::cell::Cell;
    use std::fs;
    use std::path::PathBuf;
    use visloc_rs::FeatureSet;

    fn test_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "visloc_sift_stream_{label}_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("thread")
        ));
        let _ = fs::remove_dir_all(&root);
        root
    }

    fn sample_features(index: usize) -> (FeatureSet, Vec<FeatureLocusMetadata>) {
        let x = 10.0 + index as f64;
        let y = 20.0 + index as f64;
        let feature = FeatureSet::new(
            vec![Point2::new(x, y), Point2::new(x + 1.0, y + 1.0)],
            vec![vec![0.1 + index as f32, 0.2], vec![0.3, 0.4]],
        )
        .unwrap();
        let loci = feature
            .keypoints
            .iter()
            .enumerate()
            .map(|(row, point)| FeatureLocusMetadata {
                x: point.x,
                y: point.y,
                scale: 1.0 + row as f64,
                orientation: 0.25 * row as f64,
            })
            .collect();
        (feature, loci)
    }

    #[test]
    fn stream_export_is_ordered_single_pass_and_byte_identical() {
        let root = test_root("identity");
        let stream_dir = root.join("stream");
        let batch_dir = root.join("batch");
        let mut paths = vec![root.join("b.png"), root.join("a.png")];
        paths.sort();
        let samples = vec![sample_features(0), sample_features(1)];
        let active = Cell::new(0usize);
        let peak = Cell::new(0usize);
        let order = Cell::new(0usize);
        let total = stream_export_features_with_loader(&paths, &stream_dir, |index, path| {
            assert_eq!(index, order.get());
            assert_eq!(path, &paths[index]);
            order.set(order.get() + 1);
            active.set(active.get() + 1);
            peak.set(peak.get().max(active.get()));
            let sample = samples[index].clone();
            active.set(active.get() - 1);
            Ok(sample)
        })
        .unwrap();
        assert_eq!(total, 4);
        assert_eq!(order.get(), 2);
        assert_eq!(active.get(), 0);
        assert_eq!(peak.get(), 1, "the loader must never overlap images");

        let names = vec!["a.png".to_owned(), "b.png".to_owned()];
        let features: Vec<FeatureSet> =
            samples.iter().map(|(feature, _)| feature.clone()).collect();
        let loci: Vec<Option<Vec<FeatureLocusMetadata>>> =
            samples.iter().map(|(_, loci)| Some(loci.clone())).collect();
        export_features_to_dir(&batch_dir, &names, &features, &loci).unwrap();
        for stem in ["a", "b"] {
            assert_eq!(
                fs::read(stream_dir.join(format!("{stem}_features.txt"))).unwrap(),
                fs::read(batch_dir.join(format!("{stem}_features.txt"))).unwrap()
            );
            assert_eq!(
                fs::read(stream_dir.join(format!("{stem}_loci.txt"))).unwrap(),
                fs::read(batch_dir.join(format!("{stem}_loci.txt"))).unwrap()
            );
            assert!(!stream_dir
                .join(format!(".{stem}_features.txt.tmp"))
                .exists());
            assert!(!stream_dir.join(format!(".{stem}_loci.txt.tmp")).exists());
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn stream_failure_keeps_existing_output_and_writes_no_partial_file() {
        let root = test_root("failure");
        let output_dir = root.join("features");
        fs::create_dir_all(&output_dir).unwrap();
        let existing = output_dir.join("a_features.txt");
        fs::write(&existing, b"previous-complete-file\n").unwrap();
        let paths = vec![root.join("a.png"), root.join("b.png")];
        let error = stream_export_features_with_loader(&paths, &output_dir, |index, _| {
            if index == 0 {
                return Err("synthetic extraction failure".into());
            }
            Ok(sample_features(index))
        })
        .unwrap_err();
        assert!(error.to_string().contains("synthetic extraction failure"));
        assert_eq!(fs::read(&existing).unwrap(), b"previous-complete-file\n");
        assert!(!output_dir.join("b_features.txt").exists());
        assert!(!output_dir.join("a_loci.txt").exists());
        assert!(!output_dir.join(".a_features.txt.tmp").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn native_coordinate_export_reuses_canonical_descriptor_rows() {
        let root = test_root("native-coordinate-export");
        let output_dir = root.join("features");
        let names = vec!["image.png".to_owned()];
        let features = vec![FeatureSet::new(
            vec![Point2::new(10.0, 20.0), Point2::new(30.0, 40.0)],
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        )
        .unwrap()];
        let native_keypoints = vec![vec![Point2::new(100.0, 200.0), Point2::new(300.0, 400.0)]];
        let loci = vec![None];
        export_features_to_dir_with_native_keypoints(
            &output_dir,
            &names,
            &features,
            &native_keypoints,
            &loci,
        )
        .unwrap();
        let exported = fs::read_to_string(output_dir.join("image_features.txt")).unwrap();
        assert_eq!(
            exported,
            feature_export_text_with_keypoints(&native_keypoints[0], &features[0].descriptors)
        );
        assert!(exported.contains("100.000000 200.000000"));
        assert!(exported.contains("0.100000 0.200000"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn resume_manifest_requires_matching_config_source_and_outputs() {
        let root = test_root("resume-manifest");
        let source = root.join("source.png");
        let output_dir = root.join("features");
        fs::create_dir_all(&output_dir).unwrap();
        fs::write(&source, b"source-image-bytes").unwrap();
        let (features, loci) = sample_features(3);
        let feature_path = output_dir.join("source_features.txt");
        let loci_path = output_dir.join("source_loci.txt");
        fs::write(&feature_path, feature_export_text(&features)).unwrap();
        fs::write(&loci_path, locus_metadata_text(&loci)).unwrap();
        let source_digest = file_fnv1a64(&source).unwrap();
        let feature_digest = file_fnv1a64(&feature_path).unwrap();
        let loci_digest = file_fnv1a64(&loci_path).unwrap();
        let manifest = sift_stream_manifest_path(&output_dir, "source");
        write_sift_stream_manifest_atomically(
            &manifest,
            0x1234,
            source_digest,
            features.len(),
            feature_digest,
            loci_digest,
        )
        .unwrap();
        assert_eq!(
            validate_sift_stream_manifest(
                &manifest,
                0x1234,
                source_digest,
                &feature_path,
                &loci_path,
            )
            .unwrap(),
            Some(features.len())
        );

        let mut tampered = fs::read(&feature_path).unwrap();
        let last = tampered.len() - 1;
        tampered[last] = if tampered[last] == b'\n' { b' ' } else { b'\n' };
        fs::write(&feature_path, tampered).unwrap();
        assert_eq!(
            validate_sift_stream_manifest(
                &manifest,
                0x1234,
                source_digest,
                &feature_path,
                &loci_path,
            )
            .unwrap(),
            None
        );
        fs::write(&feature_path, feature_export_text(&features)).unwrap();
        assert!(validate_sift_stream_manifest(
            &manifest,
            0x9999,
            source_digest,
            &feature_path,
            &loci_path,
        )
        .unwrap()
        .is_none());
        assert!(validate_sift_stream_manifest(
            &manifest,
            0x1234,
            (source_digest.0 + 1, source_digest.1),
            &feature_path,
            &loci_path,
        )
        .unwrap()
        .is_none());
        let mut changed_source = fs::read(&source).unwrap();
        changed_source[0] ^= 1;
        fs::write(&source, changed_source).unwrap();
        let changed_source_digest = file_fnv1a64(&source).unwrap();
        assert!(validate_sift_stream_manifest(
            &manifest,
            0x1234,
            changed_source_digest,
            &feature_path,
            &loci_path,
        )
        .unwrap()
        .is_none());
        assert!(!manifest
            .with_file_name(format!(
                ".{}.tmp",
                manifest.file_name().unwrap().to_string_lossy()
            ))
            .exists());
        let _ = fs::remove_dir_all(root);
    }
}
