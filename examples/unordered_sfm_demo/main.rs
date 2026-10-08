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
//!      Levels through offset 32 stay dense; longer levels are sampled with
//!      stride `offset / 16` after the cross-camera edges. This reaches wide
//!      gaps without making every long level linear in the sequence length.
//!      This is deterministic and GT-free; it is useful when numeric
//!      timestamps are irregular or have large nanosecond gaps.
//!      `--rig-frame-manifest` accepts the `generalized-rig-manifest-v1`
//!      used by `generalized_rig_sfm`; its explicit `F frame image sensor`
//!      rows override filename-derived grouping for datasets whose synchronized
//!      cameras use different aliases or timestamps.
//!      `--retrieval-min-frame-gap` applies a rig-frame temporal exclusion
//!      only to appearance fill, reserving it for loop-closure baselines while
//!      leaving the temporal pyramid and same-frame rig edges unchanged.
//!      File-backed exports can add `--stream-candidate-features` to release
//!      local descriptors per image. `--retrieval-backend lsh` enables the
//!      deterministic approximate index; `--ann-bits 0` scales its signature
//!      width with the image count.
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
//! `--shared-snapshot-envelope` opts exported snapshots into compact pair chunks
//! with a content-addressed `.vpe` envelope in the same directory. Copy both
//! chunks and envelopes together; legacy snapshot import remains supported.
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
//! `--component-model-min-images N` instead maps every verified-view-graph
//! component with at least `N` images, largest first, and writes independent
//! `component-NNN` COLMAP models plus `components.tsv` below `--out-colmap`.
//! `--component-model-max-count N` bounds the number of models (default 16).
//! Components run sequentially from one feature load, so their reconstruction
//! state is not retained together; the outputs have independent gauges and are
//! not one connected model. This mode cannot be combined with a fixed seed or
//! initial poses and is default-off.
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
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::io::Read;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use nalgebra::{Matrix3, Point2, Point3, UnitQuaternion, Vector3};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
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
    GlobalReconstructionTuning, IncrementalSfmConfig, IncrementalSfmResult, LinearSolver, Matcher,
    NextImagePolicy, PairwiseMatches, PerImageCameras, Pose, RobustKernel, TrackSource, SE3,
};

use visloc_rs::slam::incremental_sfm::log_process_memory;
use visloc_rs::verified_pair_snapshot::{
    self, PairRecord as SnapshotPairRecord, Snapshot as VerifiedPairSnapshot,
};

mod candidates;
mod cli;
mod colmap_prior;
mod diagnose;
mod features;
mod mapper_input;
mod match_worker;
mod matching;
mod pair_quality;
mod snapshot;

use candidates::*;
use cli::*;
use colmap_prior::*;
use diagnose::*;
use features::*;
use mapper_input::*;
use match_worker::*;
use matching::*;
use pair_quality::*;
use snapshot::*;

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

#[derive(Debug)]
struct Args {
    /// `--gpu-match`: batched GPU descriptor matching (wgpu) for the plain
    /// `nn` matcher in pair verification. Same decisions as the CPU
    /// matcher up to f32 summation order; needs `--features gpu`.
    gpu_match: bool,
    /// `--gpu-sift`: run the wgpu SIFT extractor for configurations it
    /// supports (DoG, no affine/DSP/VLFeat modes); others stay on the CPU.
    /// Not bit-identical to the CPU extractor (>= 98% identical keypoints,
    /// descriptor dot > 0.999 in its parity test); needs `--features gpu`.
    gpu_sift: bool,
    /// `--gpu-ba`: run the mapper's global bundle adjustments on the wgpu
    /// LM solver (`visloc-ba-gpu`); local BA stays on the CPU. Same final
    /// cost as the dense solver up to f32 linearisation, not bit-identical;
    /// needs `--features gpu`.
    gpu_ba: bool,
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
    /// Store one content-addressed envelope alongside compact pair chunks.
    shared_snapshot_envelope: bool,
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
    /// Keep only calibrated keypoints resident in a persistent match worker
    /// and hydrate descriptors for the current candidate shard on demand.
    /// This bounds descriptor memory by shard incidence instead of image
    /// count, at the cost of re-reading images shared by multiple shards.
    stream_match_features: bool,
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
    /// Maximum number of seed *growth attempts* (`0` = use `seed_trials`).
    seed_attempts: usize,
    /// Optional diagnostic restriction to one seed pair (`I,J`).
    seed_pair: Option<(usize, usize)>,
    /// Map each sufficiently large verified-view-graph component independently
    /// and write one COLMAP model per component below `out_colmap`.
    component_model_min_images: Option<usize>,
    /// Bound the number of independently reconstructed components.
    component_model_max_count: usize,
    refine_intrinsics: bool,
    refine_distortion: bool,
    /// `--refine-tangential-distortion`: also self-calibrate `(p1, p2)`
    /// (the camera becomes OPENCV). Requires `--refine-distortion`.
    refine_tangential_distortion: bool,
    /// `--shared-focal`: constrain the refined focal to `fx == fy`.
    shared_focal: bool,
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
    /// Solve the pure-visual bundle adjustments with the matrix-free
    /// implicit-Schur PCG backend. The default dense/sparse block-Cholesky
    /// reduced system fills in when tracks are long, so a large rig or
    /// temporal-pyramid run spends hours in `linear_solve`; the matrix-free
    /// operator tracks the observation count instead. Ineligible problems
    /// (intrinsics refinement, non-visual states, no gauge anchor) fall back
    /// to the ordinary solver. Default off.
    matrix_free_ba: bool,
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
    /// Explicit generalized-rig frame/image/sensor assignments for temporal
    /// candidate generation. This avoids inferring synchronization from image
    /// aliases when different sensors use different names or timestamps.
    rig_frame_manifest: Option<PathBuf>,
    /// Optional labels from a completed first-pass reconstruction. When
    /// present, appearance fill prioritizes pairs whose registered endpoints
    /// belong to different components. Ground truth is never consulted.
    retrieval_component_manifest: Option<PathBuf>,
    /// Reject appearance-retrieval fill edges whose rig frame ids differ by
    /// less than this value. Temporal-pyramid and same-frame rig edges are
    /// unaffected; this keeps a bounded retrieval budget for loop closure.
    retrieval_min_frame_gap: Option<u64>,
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
    /// File-backed candidate-export mode that retains only the bounded
    /// vocabulary sample and one VLAD descriptor per image. Local descriptor
    /// payloads are parsed in deterministic passes and released per file.
    stream_candidate_features: bool,
    /// Global-descriptor neighbour search used by streamed temporal-pyramid
    /// candidate export. Exact preserves historical output; LSH is opt-in.
    retrieval_backend: RetrievalBackend,
    /// Number of deterministic feature-hash projection tables for LSH.
    ann_tables: usize,
    /// Signature width per LSH table (1..=63, or 0 for scale-aware auto).
    ann_bits: usize,
    /// Lowest-margin one-bit buckets probed per table.
    ann_probes: usize,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\nsee the file header for usage.");
            std::process::exit(2);
        }
    };
    // Parsing rejects the GPU flags on builds without the `gpu` feature.
    #[cfg(not(feature = "gpu"))]
    let _ = (args.gpu_match, args.gpu_sift, args.gpu_ba);
    #[cfg(feature = "gpu")]
    if args.gpu_match {
        let ctx = visloc_sift_gpu::GpuContext::new().map_err(|e| format!("gpu: {e}"))?;
        let matcher = visloc_sift_gpu::GpuMatcher::new(&ctx);
        let _ = GPU_NN.set((ctx, matcher));
    }
    #[cfg(feature = "gpu")]
    if args.gpu_sift {
        let ctx = visloc_sift_gpu::GpuContext::new().map_err(|e| format!("gpu: {e}"))?;
        let _ = GPU_SIFT.set(std::sync::Mutex::new(visloc_sift_gpu::SiftGpu::new(ctx)));
    }
    #[cfg(feature = "gpu")]
    if args.gpu_ba {
        let ctx = visloc_ba_gpu::GpuContext::new().map_err(|e| format!("gpu: {e}"))?;
        visloc_rs::slam::set_ba_accelerator(Box::new(visloc_ba_gpu::GpuBundleAdjuster::new(ctx)));
    }
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
    if args.stream_candidate_features {
        let files = list_feature_files(&args.features_dir, &args.feature_suffix)?;
        let image_names = files
            .iter()
            .map(|file| image_name_for(file, &args.feature_suffix, &args.image_suffix))
            .collect::<Vec<_>>();
        let calibration_root = args
            .input_colmap_calibration
            .as_deref()
            .ok_or("--stream-candidate-features requires --input-colmap-calibration")?;
        let calibration = resolve_input_colmap_calibration(calibration_root, &image_names)?;
        validate_calibration_image_dimensions(
            &calibration.rig,
            &image_names,
            args.images_dir.as_deref(),
        )?;
        let streamed = stream_vlad_globals_from_feature_files(
            &args.features_dir,
            &files,
            &calibration.rig,
            args.vocab_size,
        )?;
        log_process_memory("example-after-streamed-vlad-globals");
        // A missing vocabulary must not allocate N*(N-1)/2 pairs before the
        // temporal-pyramid budget is applied. Fail closed instead.
        let globals = streamed.appearance_globals()?;
        let retrieval = match args.retrieval_backend {
            RetrievalBackend::Exact => {
                candidate_pairs_vlad_scored_from_globals(globals, args.retrieval_topk, false)
            }
            RetrievalBackend::Lsh => {
                let bits = effective_ann_bits(args.ann_bits, globals.len());
                if args.ann_probes > bits {
                    return Err(format!(
                        "--ann-probes {} exceeds effective --ann-bits {bits}",
                        args.ann_probes
                    )
                    .into());
                }
                candidate_pairs_vlad_lsh_scored(
                    globals,
                    args.retrieval_topk,
                    args.ann_tables,
                    bits,
                    args.ann_probes,
                )
            }
        };
        let generated = candidate_pairs_temporal_pyramid_from_retrieval(
            &image_names,
            retrieval,
            args.temporal_pyramid_max_offset,
            args.candidate_budget,
            args.rig_frame_manifest.as_deref(),
            args.retrieval_component_manifest.as_deref(),
            args.retrieval_min_frame_gap,
        )?;
        let path = args
            .export_candidate_manifest
            .as_deref()
            .ok_or("streamed candidate export lost its output path")?;
        let metadata = candidate_manifest_metadata(&args, image_names.len());
        write_candidate_manifest_with_metadata(path, &image_names, &generated, &metadata)?;
        println!(
            "streamed candidate manifest: {} pairs / {} images / {} local descriptors / {} vocabulary samples -> {}",
            generated.len(),
            image_names.len(),
            streamed.total_descriptors,
            streamed.sampled_descriptors,
            path.display(),
        );
        return Ok(());
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
            let (features, image_names, locus_metadata) =
                if args.snapshot_keypoints_only || args.stream_match_features {
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
    let snapshot_feature_validation = if args.snapshot_keypoints_only || args.stream_match_features
    {
        let paths = snapshot_feature_paths
            .as_deref()
            .ok_or("--snapshot-keypoints-only did not retain feature source paths")?;
        let fingerprints = snapshot_feature_fingerprints
            .as_deref()
            .ok_or("--snapshot-keypoints-only did not retain feature source fingerprints")?;
        let validation = snapshot_feature_validation_from_files(paths, &features, fingerprints)
            .map_err(std::io::Error::other)?;
        println!(
            "memory-bounded feature load: retained {} keypoint sets; descriptor payloads re-read one file at a time (feature-manifest-fnv1a64={:016x})",
            features.len(), validation.feature_manifest_hash,
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
            remap_locus_metadata(&mut locus_metadata, &map)?;
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
            remap_imported_verified_pairs(&mut imported, map)?;
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
        let compact_mapper_replay = args.export_verified_pairs_snapshot.is_none();
        let snapshot = if compact_mapper_replay {
            verified_pair_snapshot::read_mapper_compact(path)
        } else {
            verified_pair_snapshot::read(path)
        }
        .map_err(std::io::Error::other)?;
        let pairwise = validate_snapshot_for_run(
            &snapshot,
            &image_names,
            &features,
            &args.camera,
            snapshot_feature_validation.as_ref(),
            compact_mapper_replay,
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
        let stats = verification_stats_from_snapshot(&snapshot, compact_mapper_replay)
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
                remap_imported_matches(&mut matches, map)?;
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
                remap_imported_matches(&mut matches, map)?;
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
                remap_imported_verified_pairs(&mut imported, map)?;
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
        let feature_validation =
            snapshot_feature_validation.unwrap_or_else(|| SnapshotFeatureValidation {
                feature_counts: features.iter().map(FeatureSet::len).collect(),
                feature_manifest_hash: snapshot_feature_manifest_hash(&features),
            });
        let stream_sources = if args.stream_match_features {
            Some((
                snapshot_feature_paths
                    .as_deref()
                    .ok_or("--stream-match-features lost its feature source paths")?,
                snapshot_feature_fingerprints
                    .as_deref()
                    .ok_or("--stream-match-features lost its feature fingerprints")?,
            ))
        } else {
            None
        };
        println!(
            "persistent match worker: {} shard(s), {} candidate pairs, feature-manifest-fnv1a64={:016x}, streamed_features={}, descriptor_cache_rows={}",
            plan.shards.len(),
            plan.pair_count,
            feature_validation.feature_manifest_hash,
            args.stream_match_features,
            if args.stream_match_features {
                STREAM_DESCRIPTOR_CACHE_ROWS
            } else {
                0
            },
        );
        run_persistent_match_worker(
            &plan,
            &mut features,
            &image_names,
            &args.camera,
            &pair_matcher,
            &args,
            &feature_validation,
            stream_sources,
        )?;
        return Ok(());
    }

    if let Some(path) = args.export_candidate_manifest.as_deref() {
        let generated = candidate_pairs(&features, &image_names, &args)?;
        let generated =
            filter_pairs_by_stem_window(generated, &image_names, args.pair_stem_window)?;
        let metadata = candidate_manifest_metadata(&args, image_names.len());
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
            remap_imported_verified_pairs(&mut imported, map)?;
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
                remap_imported_matches(&mut imp, map)?;
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
                remap_imported_matches(&mut imp, map)?;
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
        seed_attempts: args.seed_attempts,
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
            matrix_free_ba: args.matrix_free_ba,
            refine_distortion: args.refine_distortion,
            refine_tangential_distortion: args.refine_tangential_distortion,
            shared_focal: args.shared_focal,
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
    if let Some(min_images) = args.component_model_min_images {
        map_verified_view_graph_components(
            &args.out_colmap,
            min_images,
            args.component_model_max_count,
            &args.camera,
            &features,
            &image_names,
            &pairwise,
            &config,
            native_keypoints_for_export.as_deref(),
            per_image_calibration.as_ref(),
        )?;
        return Ok(());
    }
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
mod diagnose_cli_tests;

#[cfg(all(test, feature = "image-io"))]
mod sift_extra_tests;

#[cfg(test)]
mod append_only_matcher_tests;

#[cfg(all(test, feature = "image-io"))]
mod sift_stream_tests;
