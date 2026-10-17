# Changelog

All notable changes to `visloc-rs` will be documented here.

## Unreleased

### Added

- **`gsplat_photos --equirect`: 360° panoramas as cube faces** (#301).
  `PhotoSfmConfig::equirect` cuts each 2:1 equirectangular panorama into
  six 90° cube faces (`visloc_gsplat_train::equirect`), at most `max_size`
  square, before SfM and training. Their focal length `size / 2` is exact
  and stays fixed. Faces of one panorama are not matched with each other,
  and the window counts panoramas. The faces are still separate cameras.
  The log reports how far apart each panorama's face centres land. Off by
  default.
  - Five synthetic room panoramas (`equirect_end_to_end`, an ignored
    test): all 30 faces register. Relative rotations are within 0.13° of
    the truth, and panorama distances within 0.4 mm over 0.58 m spacing.
  - The five real panoramas of the
    [ScanVan example](https://github.com/ScanVan/sfs-framework-example)
    (CPU, `--steps 0`): the four side faces of every panorama register;
    the up and down faces (sky, the camera mount) do not. Panorama
    distances match ScanVan's own trajectory within 0.4%. As pinhole
    images the same panoramas reconstruct nothing.
  - The same five on an RTX 5080 (GPU SIFT and matching, 30k steps): the
    same 20 faces register, and SfM, training and the mesh take 5 to
    14 min depending on the host. The held-out panorama (the first, a
    step behind every training one) renders at 17.8 to 22.2 dB per face
    (19.2 dB mean), the training faces at 35 to 38 dB. The buildings
    land where they should. The losses are thin branches, and the lens
    rims and camera mount that move with the camera (`gsplat_photos`
    takes no mask for them).
  - With `equirect`, the held-out views are whole panoramas. Every 8th
    registered face would have held out the same face of every
    panorama: on ScanVan it took 3 of the 5 front faces, which left the
    view ahead with two training images.
- **A warning for panoramas fed as pinhole photos.** When at least half the
  images look equirectangular, `build_photo_dataset` logs a warning that
  points to `--equirect`. An image counts when it is 2:1 with left and
  right edges that meet and smooth pole rows, or when its metadata says
  equirectangular (Photo Sphere XMP). Outputs are unchanged. It flags
  three stitched Poly Haven panoramas and none of the 128 south-building
  photos or their 2:1 crops. It misses the ScanVan panoramas, whose left
  and right edges are lens rims that do not meet.

### Fixed

- **COLMAP-port mapper: resuming a reconstruction with 3D points
  panicked.** `ObservationManager::new` counted the existing observations
  before the visibility pyramids were sized. The constructor now sizes
  them, as COLMAP's `InitImageStat` does; on an empty reconstruction the
  state is unchanged.

- **COLMAP-port mapper: the converged final BA stalled on two of three
  SmallCity paths.** Two LM rules now apply to solves with the absolute
  gradient criterion. In production that means only the converged final BA
  (under Added).
  - **A trial is judged the way the model sees it.** The trial cost now
    leaves out observations whose point is behind its camera at the
    current state, as the linearization already does. Before, a point that
    one step had carried behind a camera came back in front on a later
    trial with a residual of hundreds of pixels. On mapper seed 1 a single
    two-view point at a camera centre did this to 7 of the 18 trials.
    Each one raised the cost by 14k to 869k and was rejected, so the trust
    radius never grew past about 6e6 and the drift stayed.
  - **A small decrease ends the solve only if the model predicted a small
    one too.** On seed 2 the final BA stopped after a step that lowered the
    cost by 4% of its predicted decrease (rho 0.04).
  - **SmallCity** (RTX 3090 pod replays with the same matches; every model
    before the final BA is byte-identical to the earlier replays). Median
    centre error:

    | seed | before the final BA | final BA before | now |
    |---|---|---|---|
    | 0 | 0.611 m | 0.146 m | 0.147 m |
    | 1 | 0.496 m | 0.366 m | 0.161 m |
    | 2 | 0.840 m | 0.711 m | 0.648 m |

    - Seed 1's p90 goes from 1.29 m to 0.35 m. The fitted scales of its
      four passes go from 3.29–3.43 to 3.31–3.33. Its final BA accepts all
      16 steps (97 s), and the trust radius reaches 1.3e8.
    - Seed 2's remaining drift (3.9% between passes) is in the BA optimum
      itself. Run with no function tolerance, the same solve also ends at
      0.650 m.
    - Seed 0 is unchanged within 1 mm (p90 0.281 m to 0.283 m).
  - **What did not help:** more refinement rounds (BA, merge, filter) after
    the converged BA. Restarted at the initial trust radius, each round
    stops after one or two steps. Started from the previous round's
    radius, four more rounds took seed 0 from 0.147 m to 0.328 m and seed
    1 from 0.161 m to 0.352 m, worse with every round.
  - **Unchanged:** the one- or two-step incremental BAs keep the plain
    evaluation. All 16 ETH3D models are byte-identical.

- **`colmap_incremental_mapper` replays a `gsplat_photos`/`gsplat_euroc`
  export with the options it ran with.** Those pipelines use COLMAP's
  single-camera `abs_pose_min_num_inliers=30`, but a replay of their
  `port/` directory used the port's rig-control default of 8 unless
  `VISLOC_ABS_POSE_MIN_INLIERS=30` was set. That registered images the
  pipeline rejects; ETH3D meadow, for one, replayed at 15/15 images with a
  26.6 cm centre RMSE instead of 14/15 at 5.9 cm. The export now writes
  `port/mapper_options.txt` (the options applied, including `--mapper-opt`
  overrides), and the replay applies it, so a replay reproduces the
  pipeline's model byte for byte. Earlier replay-based comparisons in this
  log ran with 8 on both sides. `MapperOptions::apply_override` now holds
  the `key=value` parser that `--mapper-opt` uses.

- **PnP RANSAC no longer throws away its best pose.** After the search,
  `PnPRansac` refit the pose estimator on all the inliers and refined
  from that refit alone. With P3P the refit solves from three of the
  inliers and can land far from the search's winner, and the refinement
  then started from the bad pose: on the Courthouse replay 50 image
  registrations failed this way (around 700 correspondences, 3-6 inliers
  left), leaving 544 of 553 images registered. It now starts from the
  better of the winner and the refit. Courthouse registers all 553 with
  no failed attempt; the ETH3D courtyard, pipes, kicker and terrains
  replays are unchanged (same images, centre RMSE within 0.07 cm).

- **`gsplat_photos` GPU matching saturated VLFeat-compatible descriptors.**
  Those descriptors come back already in COLMAP's byte scale (norm about
  512), and the default u8 GPU bank quantises with `round(512 x)` again,
  so every non-zero value clipped to 255 and the descriptors were nearly
  binary. On SmallCity (GPU matching) this cut the verified pairs by 12%
  and the 3D points by 70% (292k instead of 958k), and pass 4 drifted
  (own-fit median 0.62 m instead of 0.13 m). They are now divided by 512
  before matching, which is exact: the bank recovers the same bytes and
  CPU matching is bit-identical. CPU-matched runs (`--cpu`, the ETH3D
  numbers above) were not affected.
- **`gsplat_photos` SIFT finds matches on high-resolution photos.** It now
  uses COLMAP's VLFeat detector (subpixel localization) and descriptor, and
  feeds SIFT grey values in [0, 1] as `GrayImage` documents (it passed
  0–255, which made the contrast threshold 255× too permissive). The default
  detector reports integer pixel positions. On one ETH3D courtyard pair with
  1,749 points in common it found 37 matches; the new path finds 721. ETH3D
  courtyard (1,600 px) registers 38/38 photos instead of 15, and pipes 11/14
  instead of 2. The VLFeat path runs on the CPU (matching stays on the GPU);
  `--sift-opt vlfeat_compatible_detector=0 --sift-opt
  vlfeat_compatible_descriptor=0 --sift-opt vlfeat_bilinear_orientations=0`
  restores GPU extraction.
- **`gsplat_photos` no longer drops photos a pixel off the common size.**
  Photos whose scaled size is within 1% of the most common one are resized
  to it instead of being dropped. ETH3D courtyard's undistorted images come
  in four sizes a few pixels apart, and 14 of its 38 photos were lost,
  splitting the scene into fragments.
- **`gsplat_photos` panicked on large photo sets** (wgpu: descriptor bank
  over `max_storage_buffer_binding_size`, e.g. 1,940 photos x 4,000 SIFT);
  it now matches in blocks. The trainer's logged L1 no longer rounds to 0
  on large images (2.8 MP), gradients were unaffected.
- **README hero splat filter read the wrong PLY columns.** Its
  large-and-transparent rule used SH coefficients as opacity and scale; the
  columns are now looked up by name.
- **`config.optical_flow_imu_seed_rotation` can be set from a Basalt config
  JSON.** The adapter read the key, but `BasaltConfig::from_json` rejected it
  as unknown, so only code could enable it.
- **Bundle adjustment uses the camera's full lens model.** Pose/structure BA
  (serial, parallel, dense-QR assembly and the GNC residual vector) now
  linearises the same lens that `Camera::project` measures, with analytic
  Jacobians for OPENCV (k1, k2, p1, p2), FULL_OPENCV, OPENCV_FISHEYE,
  SIMPLE_RADIAL_FISHEYE, RADIAL_FISHEYE, FOV and Double Sphere. These cameras
  were previously rejected with `UnsupportedCameraModel`; SIMPLE_RADIAL and
  RADIAL are now accepted too. Pipelines that used to skip BA silently on
  such a camera (they swallow the error) now run it. Unknown models stay
  rejected, and stereo observations with any camera other than PINHOLE /
  SIMPLE_PINHOLE now return `UnsupportedCameraModel` instead of being
  modelled as a pinhole. Distortion-free and radial-only cameras are
  bit-for-bit unchanged (pinned by tests). Not yet covered: rig observations
  still project with a pinhole, and the matrix-free / GPU backends still
  decline distorted cameras and fall back to the regular solver.

### Added

- **Multi-view stereo: `visloc-mvs`.** Dense point clouds from a COLMAP
  sparse model with undistorted images: PatchMatch depth and normal maps
  (ACMH, with ACMP's planar prior and ACMM's geometric consistency and
  multi-scale scheme), fused into `fused.ply`, on the CPU or on the GPU
  through wgpu (`--features gpu`). Details and scores in `docs/mvs.md`.
  - **ETH3D high-res training, all 13 scenes at 3200 px, one NVIDIA L4:**
    F1 81.4% at 2 cm on average (accuracy 87.7%, completeness 76.9%) with
    the example's defaults (one scale, the planar prior pass, two
    geometric passes). On the same scenes ACMMP reports 83.42%, ACMP
    79.79%, ACMM 78.86% and COLMAP 67.66%.
  - **Planar prior** (ACMP): credible pixels are triangulated into
    planes that a prior pass favours where the costs cannot tell planes
    apart. It raises F1 on all 13 scenes, from 79.5% to 81.4%, most on
    office (+7.2 points), meadow (+3.8) and pipes (+3.4);
    `--no-planar-prior` leaves it out. ACMM's multi-scale scheme
    (`--scales 3`, without the prior) scores 78.0%: it helps weakly
    textured scenes and costs accuracy on detailed ones.
  - **Time:** 77 minutes for all 13 scenes, 1:54 to 14:17 per scene,
    loading and fusion included (65 minutes without the planar prior).
  - **Fusion:** one agreeing source view within 10° keeps a point by
    default (ACMM and ACMP ask for two): over the 13 scenes, 81.4%
    against 79.3% with two views at 30°.
  - **Memory:** one set of depth maps at a time (a geometric pass and
    fusion drop each map after the last view that reads it), costs only
    with `--save-maps`, grey levels computed from the colours. On ETH3D
    facade (76 images at 3200 px, an L4), peak memory falls from 24.2 to
    12.0 GiB with one scale and from 28.7 to 12.4 GiB with three;
    `fused.ply` is byte-identical on pipes (CPU and GPU).
  - **Run it:** `cargo run --release -p visloc-mvs --features gpu
    --example mvs -- --sparse <model> --images <dir> --out <dir> --gpu`.

- **COLMAP-port mapper: complete the largest model
  (`complete_largest_model=1`).** Off by default in the mapper; on in
  `gsplat_photos` (`--no-complete-model` turns it off, see Changed).
  - **When it runs:** once every model is built. It extends the largest
    model with the images it left out.
  - **Candidates:** images with at least `2 * abs_pose_min_num_inliers`
    2D–2D correspondences to registered images.
  - **Why COLMAP misses them:** its structure-less fallback asks the same
    candidates for `abs_pose_min_num_inliers` 3D points. That filter keeps
    out the images that see too few of them.
  - **Each registration** is triangulated and locally refined. It is undone
    if the image then sees fewer than `abs_pose_min_num_inliers` 3D points.
  - **No other effect:** the pass cannot change which models are built, and
    a model it does not extend stays bit-identical.
  - **ETH3D, 13 DSLR training scenes:**
    - On COLMAP's matches: mean AUC@1°/3°/5° 51.97 / 76.75 / 83.21 (445 of
      454 registered) → 53.69 / 79.94 / 86.90 (452). That is ahead of
      COLMAP 4.2.1's incremental mapper (51.34 / 76.25 / 82.88) and of its
      global mapper (48.65 / 70.41 / 76.36), and ahead of the global mapper
      without pipes too.
    - On visloc-rs's matches: 53.09 / 78.85 / 85.69 (448) → 53.25 / 79.66 /
      86.90 (452).
    - Details: `docs/eth3d_auc_benchmark.md`.

- **VLFeat-compatible SIFT on the GPU (`gsplat_photos`).**
  `visloc_sift_gpu::VlfeatSiftGpu` ports the photo pipeline's SIFT (the
  VLFeat-compatible detector and descriptor with COLMAP's byte
  descriptors) to f32 WGSL. `gsplat_photos` uses it by default
  (`--cpu-sift` for the CPU extractor, see Changed).
  - **On the GPU:** both scale spaces, the DoG extrema with VLFeat's
    five-step refinement, the orientation histograms, the descriptor
    pyramid's gradients and the descriptors.
  - **On the host, as the CPU path does:** the scan order, the per-locus
    orientation cap, the per-level `max_keypoints` cap and each keypoint's
    descriptor level, in f64.
  - **Against the CPU extractor (f64):** on 50 SmallCity photos and an RTX
    3090, 99.97% of the keypoints pair up both ways; descriptor bytes
    differ by 0.0007 on average and by at most 35 (at most one on
    llvmpipe). On 40 SmallCity photos `gsplat_photos` verifies 700 of 780
    pairs and registers 40/40 either way.
  - **Speed:** on an RTX 3090, SmallCity's 5,822 photos at 1,024 px take
    43.8 s (7.5 ms a photo), against 201.9 s for the CPU extractor on 23
    threads. Images of one size are pipelined: the GPU detects the next
    image while the host orders and caps the current one's candidates.
  - **ETH3D, the 13 DSLR training scenes** (an RTX 3090, 13 CPU threads):
    extraction takes 10.0 s against 92.4 s, and the visloc-rs mapper
    scores the same: mean AUC@1°/3°/5° 53.08 / 78.85 / 85.68 against
    53.09 / 78.85 / 85.69, 448/454 registered either way.
  - **ETH3D on an L4**, the GPU of the COLMAP 4.2.1 CUDA runs (13 CPU
    threads, COLMAP's 1,600 px input): 16.3 s with decoding, against
    22.2 s for COLMAP's CUDA extractor. Features, matches and the mapper
    together took 73.4 s, against 225.0 s for COLMAP with its global
    mapper.
  - Each pyramid level is its own buffer, so no binding exceeds one level;
    an image past the device's limit is extracted on the CPU.
    `sift_vlfeat_gpu_bench` compares the two extractors on a directory of
    images.

- **COLMAP-port mapper: large bundle adjustments on the GPU.**
  `colmap_incremental_mapper --gpu-ba 1` and `gsplat_photos --gpu-ba` (both
  `--features gpu` builds) solve the reduced camera system of large bundle
  adjustments through wgpu (`visloc_ba_gpu::GpuReducedSolver`). Without the
  flag nothing changes: the CPU path stays bit-identical.
  - **What moves:** bundle adjustments with at least 1,000 free frames. The
    GPU runs the preconditioned conjugate gradients on the reduced camera
    system. Its product reads each off-diagonal block once, as the CPU's
    does: one workgroup per frame over the contiguous blocks of its edges,
    which writes the other half of each edge's product for a second pass
    to gather. The CPU keeps the linearization, the point elimination, the
    preconditioner's factorization and the back-substitution, in f64, and
    uploads the blocks in f32 for each solve.
  - **GPU elimination, opt-in** (`VISLOC_GPU_ELIMINATION=1`): the GPU forms
    the off-diagonal blocks itself, as exact sums (64-bit fixed point over
    two 32-bit atomics), and keeps them for the solve.
  - **Deterministic:** the conjugate gradients reduce in a fixed order, so
    a solve repeats exactly on a given device. It is not bit-identical to
    the CPU path.
  - **Validation:** on a SmallCity pass-1 replay (1,499 photos, llvmpipe)
    the large bundle adjustments take about the CPU's PCG iteration counts,
    and the camera centres differ from the CPU model by a median of 4e-5
    and at most 0.005% of the scene extent.
  - **Speed (SmallCity on an NVIDIA L4, 15 CPU threads):** the mapper takes
    845.0 s with it and 846.7 s without. Its conjugate-gradient iterations
    take 3.4 ms against the CPU's 6.8 ms at 3.4M edges, but each solve's
    upload cost 0.59 s, and those solves are a tenth of the mapper's time.
    Uploads now go through a buffer kept between solves (not yet timed on a
    GPU). The first version (each block read twice, the blocks formed on
    the GPU) took 1,193 s against 1,168 s on an L4 with 21 threads.
  - `visloc_slam::ReducedSolveAccelerator` is the hook another backend can
    implement. Details: `docs/gpu_bundle_adjustment.md`.

- **ETH3D benchmark against COLMAP 4.2 and GLOMAP.**
  - **What it runs:** `scripts/benchmark_eth3d_auc.sh` runs the visloc-rs
    mapper, COLMAP's incremental mapper and COLMAP's global mapper (GLOMAP,
    via pycolmap 4.2.1) on the 13 ETH3D DSLR training scenes.
  - **Inputs:** the same inputs for all three, either COLMAP's SIFT and
    exhaustive matches (the GLOMAP paper's setup) or visloc-rs's own.
  - **Scoring:** as in the GLOMAP paper, AUC at 1°, 3° and 5° of the pairwise
    relative pose error.
  - **Result on COLMAP's matches (mean AUC@1°/3°/5°):**

    | Mapper | AUC | Mapper time |
    |---|---|---|
    | visloc-rs | 50.9 / 75.5 / 82.1 | 28.4 s |
    | COLMAP incremental | 50.4 / 75.2 / 81.7 | 200.7 s |
    | COLMAP global | 51.6 / 75.5 / 81.7 | 184.4 s |

  - **On visloc-rs's own matches** it leads both at every threshold: 53.1 /
    78.9 / 85.7, against COLMAP's three-run means of 50.2 / 75.1 / 82.0 and
    51.3 / 75.0 / 81.1.
  - **Mapper options:** visloc-rs runs as `gsplat_photos` configures it, with
    COLMAP 4's structure-less fallback on. On the port's default, which
    leaves it off, it registers 7 and 3 fewer images and scores 50.7 / 74.5
    / 80.5 and 51.7 / 76.5 / 83.1.
  - **SmallCity:** on the 5,822-photo capture, both mappers get visloc-rs's
    matches and 32 CPU threads. visloc-rs maps it in 21.7 min. COLMAP's
    global mapper had not finished when a 100-minute cap stopped it, 94 min
    into mapping.
    `scripts/benchmark_smallcity_glomap.py` runs COLMAP's side.
  - Tables and caveats: `docs/eth3d_auc_benchmark.md`. Raw results:
    `benchmarks/eth3d_auc/` and `benchmarks/smallcity_glomap/`.

- **COLMAP-port BA: two-level PCG preconditioner for converging
  solves.** Once a large global BA (1,000+ free frames) is past its first
  three LM steps, its iterative solve adds a coarse correction to the
  run-of-frames preconditioner. In practice that means the converged
  final BA.
  - **What it does:** one shared 6-vector step per aggregate of 32
    consecutive frames, solved densely (`P (PᵀAP)⁻¹ Pᵀ r`). A frame's step
    is a world-frame motion, so a shared step moves the aggregate rigidly.
  - **Why it helps:** smooth deformations of a long capture, such as scale
    drifting along a street, are close to rigid at that scale. They are
    the low-curvature modes conjugate gradients converge on slowest, and
    the run-of-frames level cannot see them. Under heavy LM damping the
    damping holds them down, so the correction waits until the damping is
    at most 1e-5 (trust radius 1e5 or more). From the first step on, it
    made SmallCity's one-step incremental BAs run up to 13 steps and
    changed the reconstruction's path.
  - **Cost:** at most 256 aggregates. SmallCity's coarse factor (1,092
    rows) takes 55 ms a solve and 0.5 ms a PCG iteration, against about
    26 ms for one PCG product. `VISLOC_PORT_PCG_COARSE=0` turns it off.
  - **PCG iterations, 4-thread benches:**

    | Bench | without | with |
    |---|---|---|
    | 1,500-frame drift test, converged solve | 3,620 (5.6 s) | 1,165 (2.1 s) |
    | `large_global_ba`, 1,200 frames, one pass | 803 | 256 |
    | `large_global_ba`, 1,200 frames, four passes (3 steps) | 149 | 149 (identical) |

    The final costs match.
  - **SmallCity** (one RTX 3090 pod replay). The model before the final
    BA is byte-identical to the replay without the coarse level:

    | | without | with |
    |---|---|---|
    | Final BA LM steps | 41 | 14 |
    | Final BA PCG iterations | 20,102 | 6,425 |
    | Final BA time | 661 s | 204 s |
    | Mapper wall time | 30:26 | 23:20 |
    | Median centre error | 0.178 m | 0.146 m |
    | p90 centre error | 0.354 m | 0.281 m |
    | Max centre error | 1.12 m | 0.76 m |

    Applying the coarse level from the first step instead changed the
    path: the mapper took 27:46 and ended at 0.524 m, with a 2% scale step
    between passes 2 and 3 left in place.
  - **Smaller problems are unchanged:** all 15 ETH3D models are
    byte-identical, since none reaches the iterative solve.

- **COLMAP-port mapper: a converged final BA on large models.** This is
  not in COLMAP.
  - **What it does:** each model of 1,000 or more registered frames ends
    with one more global refinement. Its BA runs to convergence (absolute
    gradient stop, up to 50 LM steps).
  - **When it runs:** 1,000 frames is where global BAs switch to the
    iterative solver. Smaller models, such as the ETH3D scenes and
    Courthouse, are unchanged.
  - **Setting it:** `ba_global_final_converge_min_frames=N` sets the
    threshold. Pass it as `gsplat_euroc --mapper-opt`, or as a line in a
    replay's `mapper_options.txt`. `0` runs it on every model, and a
    number above any model's size turns it off.
  - **Why it is needed:** COLMAP's global BAs stop on the relative
    `gradient_tolerance=1.0` after one or two steps. Each restarts from
    Ceres' initial trust radius, whose damping holds back smooth,
    low-curvature modes. Scale drift along a street is such a mode, so no
    global BA of the incremental loop corrects it.
  - **Synthetic test** (`global_ba_drift_needs_a_converged_solve`): a
    1,500-frame street with an 8% scale drift.
    - One step leaves 6.7% of the drift.
    - The converged solve leaves 0.01%.
  - **SmallCity:** its four passes are the four sides of the block, an
    open path that ends about 26 m from its start. Its scale drifts
    smoothly along the path, by about ±4% against the reference. One
    replay on an RTX 3090 pod (32 vCPUs) compared the model before and
    after the final converged BA (`VISLOC_SNAPSHOT_BEFORE_FINAL`):

    | | before | after |
    |---|---|---|
    | Median centre error | 0.611 m | 0.178 m |
    | p90 centre error | 2.10 m | 0.354 m |
    | Max centre error | 2.92 m | 1.12 m |
    | Scale drift along the path | ±4% | about ±1% |
    | Per-pass fitted scales | 3.92 to 4.19 | 3.92 to 3.98 |

    - The "before" model matches the earlier replay without this BA byte
      for byte.
    - The final BA ran 41 LM steps (20,102 PCG iterations) and stopped on
      the function tolerance. It took 11 minutes, so the mapper's wall
      time went from 21:00 to 30:26.
    - With the two-level PCG preconditioner (the entry above), the same
      final BA takes 14 steps and 204 s and reaches 0.146 m. The mapper
      then takes 23:20.
    - **Follow-up: the gain depends on the reconstruction path.** Three
      replays with the same matches and `--random-seed` 0, 1 and 2 give
      these median centre errors:

      | seed | before the final BA | after |
      |---|---|---|
      | 0 | 0.611 m | 0.146 m |
      | 1 | 0.496 m | 0.366 m |
      | 2 | 0.840 m | 0.711 m |

      The final BA improves every path, but it removes the scale drift
      (about ±4% along the path) only on seed 0. Seed 2's BA stops after
      seven accepted steps on the function tolerance, while its trust
      radius is still small. In seed 1's BA, 7 of 18 trials are
      rejected. Both causes are fixed (under Fixed: "the converged final
      BA stalled"): seeds 0, 1 and 2 now reach 0.147 m, 0.161 m and
      0.648 m.
  - **ETH3D, 13 scenes, forced on with a threshold of 0:** the change is
    within noise. Meadow goes from 5.73 to 4.94 cm and facade from 1.67
    to 1.77 cm. That is why small models are left out. At the default
    threshold, all 15 ETH3D models are byte-identical to before.
  - `VISLOC_SNAPSHOT_BEFORE_FINAL=<dir>` writes the model as the
    incremental loop left it, so one run gives both models.

- **`visloc_web`: run the photo pipeline from a web browser.**
  `cargo run --release --example visloc_web` serves a page on
  127.0.0.1:8080. From it you can:
  - pick a photo folder in a folder browser;
  - choose SfM only, splats, or splats and a mesh, plus the main
    `gsplat_photos` options;
  - start a run, follow a progress bar parsed from the run's output, and
    see the log without the mapper's per-step diagnostics;
  - cancel a run;
  - turn the reconstruction in an in-page 3D viewer (sparse points in
    their colours, cameras as frustums);
  - download `scene.ply`, `mesh.ply`, the COLMAP text model and the full
    log;
  - open an existing output folder just to view it.

  Each run is a `gsplat_photos` child process, one at a time. The server
  uses only the standard library, so no new dependencies. Downloads are
  limited to the run's output folder.

- **The COLMAP-port BA refines rig calibrations
  (`ba_refine_sensor_from_rig`).** The non-reference sensors'
  `sensor_from_rig` poses of the observed rigs become 6-DoF nodes of the
  reduced system (stepped as `sensor_from_rig ∘ exp(ξ)`), coupled directly
  to the frames observing through them and, through the points, to
  everything else, on both solve paths. Local BA refines a rig's sensors
  only while every registered frame of it is in the window
  (`incremental_mapper.cc:1036-1065`). Off by default
  (`--mapper-opt ba_refine_sensor_from_rig=1`). A synthetic 3-sensor rig
  is recovered from a perturbed calibration on both paths. SmallCity's
  six cameras are not rigid enough for it: its reference poses put
  cameras 3 and 5 about 1° (median, up to 15° in turns) off any fixed
  calibration, ~8 px at its focal length, so a rig model would fight the
  data there.

- **The COLMAP-port mapper self-calibrates the focal length and principal
  point in its bundle adjustments, and `gsplat_photos` turns it on.**
  COLMAP's `ba_refine_focal_length` / `ba_refine_principal_point`
  (`--mapper-opt ba_refine_focal_length=1` etc., recorded in
  `mapper_options.txt`) add the camera to the native BA as one more
  6-dim node of the reduced system (`[focal, cx, cy]` plus three fixed
  slots), so the direct and the iterative (PCG, 1000+ free frames) solves
  take it like a frame; forcing every ETH3D solve onto the iterative path
  reproduces the direct solves' errors to 0.002 cm. One focal step moves
  `fx` and `fy` together (square pixels). Local BA refines a camera only
  while every registered image of it is in the window, as COLMAP does.
  Off by default in the mapper; `gsplat_photos` enables it unless
  `--no-refine-intrinsics`. On the 13 ETH3D DSLR scenes the centred
  principal point prior is ~4 px off, and fixed intrinsics had inflated
  the camera-centre errors 2-5x. Replaying the 8192-detection exports
  with the refinement on (centre RMSE, cm; COLMAP 4.2 from the same prior
  with focal and principal point refined):

  | scene | fixed | refined in the mapper | COLMAP refined |
  |---|---|---|---|
  | facade | 6.688 | **1.673** | |
  | courtyard | 4.431 | **1.420** | 1.680 |
  | electro | 3.506 | **0.523** | |
  | delivery_area | 2.354 | 0.502 | 0.464 |
  | kicker | 2.075 | 0.744 | 0.538 |
  | terrace | 2.049 | **0.380** | 0.550 |
  | office | 0.997 | **0.565** | 0.800 (20/26) |
  | relief | 0.855 | 0.429 | 0.281 |
  | relief_2 | 0.730 | 0.228 | 0.201 |
  | playground | 0.628 | **0.386** | 123.7 |
  | terrains | 0.526 | **0.244** | 0.314 |
  | pipes | 0.699 | **0.326** | 0.392 |
  | meadow | 5.839 | 5.732 | 5.312 |

- **COLMAP 4's structure-less registration fallback in the COLMAP-port
  mapper.** When no candidate image registers from 2D-3D correspondences,
  the mapper now retries the candidates from 2D-2D correspondences to the
  registered images. The registered cameras form one generalized camera
  in world coordinates. GR6P RANSAC between it and the new image gives
  the pose with metric scale. The inlier matches then continue existing
  tracks or triangulate new ones, and a Cauchy-loss BA refines the pose.
  `MapperOptions::structure_less_registration_fallback` turns it on.
  `gsplat_photos` and `gsplat_euroc` enable it, and the port's own controls
  keep it off. Results with it on, from the same features:
  - ETH3D pipes registers 14/14 images instead of 11/14, as COLMAP 4.2's
    mapper does on these matches (COLMAP also registers 11/14 with the
    fallback off).
  - The other 12 ETH3D DSLR training scenes are unchanged.
  - The Courthouse replay with it off is byte-identical.

  Deviations from COLMAP 4: GR6P only, with no 5+1 solver, grouped
  sampler or LO step; a bounded number of trials; and an angular epipolar
  threshold in place of the tangent Sampson error.

- **`gsplat_photos --mask-black N` trains around blacked-out pixels.**
  Near-black regions (people and cars some datasets black out) get alpha 0
  in the training images; `TrainConfig::alpha_mask` leaves pixels with alpha
  below 128 out of the L1 and SSIM losses (no gradient, no window centre),
  and the held-out PSNR skips them (`metrics::psnr_masked`).
- **README hero on a city block: Hierarchical 3DGS SmallCity.** 5,822
  photos, 5,583 registered with 957,418 points, held-out PSNR 24.09 (masked),
  4.7M gaussians and a 5.9M-triangle mesh from one `gsplat_photos` command
  in 2.5 h on a Google Colab A100 (5 h before this cycle's speed-ups;
  earlier in this cycle: Tanks and Temples Courthouse, 553 frames). The orbit
  shows the roads: training leaves the dataset's blacked-out people and cars
  out of the loss, and the hero raises the street splats' opacity
  (`--ground-radius`, `--ground-min-alpha`). `scripts/colab/hero_pipeline.py` runs the whole job on a
  Colab GPU (Tanks and Temples, Mill-19 and Hierarchical 3DGS scenes; from
  the terminal with the Colab CLI or from
  `scripts/colab/readme_hero_courthouse.ipynb`), adding the NVIDIA Vulkan
  user-space driver when the VM lacks it. `scripts/make_readme_hero.py` is
  scene-agnostic (intrinsics from `cameras.txt`, margins relative to the
  camera ring, nadir-aware up axis, orbit / filter flags, Linux + EGL).
- **`gsplat_photos` for large photo sets.** `--retrieval K` (VLAD top-K pairs
  past `--exhaustive-max`), `--exhaustive-max`, `--window`,
  `--max-keypoints`, `--sift-opt key=value` (wide-baseline `affine` /
  `domain_size_pooling` run on the CPU) and `--refine-intrinsics-max`;
  GPU matching splits the descriptor bank into blocks past the ~2 GiB
  binding limit.
- **Faster mapper and BA on large models.** The COLMAP-port mapper filters
  points and completes / merges tracks in parallel (byte-identical
  output), builds BA problems with less overhead, and factors reduced
  camera systems of 64+ frames in a fill-reducing order (nested
  dissection / RCM). The joint pose + intrinsics BA gained a block-sparse,
  parallel solve (`LinearSolver::Sparse`, used by `gsplat_photos`).

- **Multi-camera Basalt VIO for divergent rigs (opt-in, aimed at LaMAria /
  Project Aria).** The VIO used only cam0 for new keypoints, landmark hosting
  and keyframe decisions, and seeded the cam0→cam1 stereo KLT at the same
  pixel, which on Aria's ~75°-divergent cameras found almost no matches. New
  optional keys: `optical_flow_matching_guess_type = "REPROJ_FIX_DEPTH"` with
  `optical_flow_matching_default_depth` (reprojection seed plus a predicted
  patch warp), `optical_flow_detect_all_cameras` (cam1 keypoint
  replenishment), `vio_landmarks_all_cameras` (cam1-hosted landmarks for
  tracks cam0 never sees) and `vio_kf_connectivity_all_cameras`. LaMAria
  variant `configs/basalt/variants/lamaria/euroc_config_big_window_multicam.json`.
  Synthetic Aria-like rig: RMS error 7.1 → 3.3 mm on a textured room, and
  23.5 → 5.1 mm when cam0 faces a blank wall for 3 s. Not yet measured on
  LaMAria; see `docs/lamaria_multicam.md`. Defaults are unchanged.
- **Joint GNSS + visual-odometry fusion** (`visloc_slam::gnss_fusion`,
  `visloc_slam::gnss_pose_graph`). VO relative-pose factors and GNSS
  position factors are optimised together (full batch or a causal sliding
  window) with an estimated GNSS-ENU-to-map alignment (yaw-only or full
  rotation plus translation; scale and per-frame scale drift for monocular
  VO), antenna lever arm, per-fix covariance, GNC outlier rejection with a
  χ² gate, and dropout bridging. Analytic Jacobians with finite-difference
  tests; RANSAC alignment bootstrap. GNSS fixes are matched to frame times by
  `visloc_fusion::interpolate_gnss_fix` / `gnss_fix_covariance` (plus
  `MeasurementBuffer::as_slice`). New `gnss_vo_fusion_demo` example and
  `scripts/check_gnss_fusion_demo_outputs.sh`. Synthetic ~1 km loop: ATE
  5.25 m (VO only) → 0.90 m fused (metric); 3.22 m (Sim(3)-aligned VO) →
  0.37 m (monocular). Loosely coupled: receiver ENU positions only, no raw
  pseudoranges and no IMU. See `docs/gnss_fusion.md`. `visloc-slam` now
  depends directly on `visloc-fusion` (it already did through `visloc-io`).
- **Monocular-inertial VIO on the Basalt port (experimental).**
  `basalt_euroc_vio_demo --mono` replays cam0 + IMU with the stereo
  calibration reduced to camera 0. The estimator needed no change: it already
  triangulates new keyframe landmarks across frames, initialises from IMU
  gravity alignment, and its KLT frontend treats cam1 as optional; the gap was
  the EuRoC reader and rig construction. New `EurocSensorDataset::open_monocular`,
  `BasaltCalibration::retain_cameras` and
  `visloc_basalt::vio_estimator_from_calibration`. The stereo-inertial path is
  unchanged. Synthetic tests (including rendered images through the real KLT
  frontend) recover metric scale within 0.3% at 5–9 mm RMS; EuRoC is still to
  be measured, and position drifts with IMU bias while the platform is static
  before the first motion. See `docs/mono_inertial_vio.md`.
- **Python bindings (`bindings/python`, package `visloc`).** pyo3/maturin
  extension with NumPy interop: `Camera` (COLMAP models + Double Sphere),
  `Pose`/`SE3`, COLMAP text/binary `Reconstruction` read/write, `localize` and
  `estimate_pose_pnp_ransac`, and `evaluate_ate` / `evaluate_rpe` /
  `umeyama_alignment`. Ships `.pyi` stubs and pytest tests; built and tested
  by a new `python-bindings` CI job. The crate is excluded from the root
  workspace (own `Cargo.lock`), so pyo3/numpy do not enter the Rust gates.
- **`visloc_io::colmap::write_colmap_binary_model`** for a `VisualMap`.
- **ROS 2 nodes (`ros2/visloc-ros2`)** over pure-Rust DDS (`ros2-client` /
  RustDDS; no ROS install needed to build). `visloc_vio_node` runs the Basalt
  stereo-inertial VIO on two `sensor_msgs/Image` topics plus
  `sensor_msgs/Imu` and publishes `nav_msgs/Odometry`,
  `geometry_msgs/PoseStamped`, `nav_msgs/Path` and `/tf` (VIO only; the
  online mapper is not attached). `visloc_localize_node` localizes
  `sensor_msgs/Image` frames against a COLMAP map plus landmark descriptors
  and publishes `PoseWithCovarianceStamped`, an inlier count and
  diagnostics. The crate is excluded from the root workspace, has its own
  lockfile and a dedicated CI job, and is tested end to end over RTPS
  (RustDDS to RustDDS), not yet against a ROS 2 install.
- **`Camera::project_with_point_jacobian`** (visloc-core): the pixel from
  `Camera::project` plus its analytic 2×3 Jacobian with respect to the
  camera-frame point, for every lens model.
- **`BaConfig::refine_tangential_distortion` / `--refine-tangential-distortion`**
  (unordered SfM demo): opt-in self-calibration of p1/p2 on top of
  `refine_distortion`. A PINHOLE camera is promoted to OPENCV
  `[fx, fy, cx, cy, k1, k2, p1, p2]` starting from p1 = p2 = 0.
  `refine_intrinsics` now also refines OPENCV cameras (their lens terms stay
  fixed unless `refine_distortion` is set). Default off.

### Changed

- **`gsplat_photos` extracts SIFT on the GPU and completes the largest
  model by default.** `--cpu-sift` and `--no-complete-model` restore the
  previous behaviour; `--cpu` also extracts on the CPU.
  - **Why:** on ETH3D the mapper scores the same with GPU SIFT (mean
    AUC@1°/3°/5° 53.08 / 78.85 / 85.68 against 53.09 / 78.85 / 85.69), and
    completion raises it (452 of 454 images registered instead of 448).
  - **SmallCity, one RTX 3090 run with 23 CPU threads:**
    - The CPU-SIFT model reproduced the earlier run byte for byte
      (5,816 / 5,822 registered, median centre error 0.147 m).
    - Completion adds nothing there: its six left-out photos do not
      register. The mapper took 1% longer.
    - With GPU SIFT, extraction took 40.2 s instead of 331.2 s, and the
      model registers 5,816 photos with a median of 0.184 m (p90 0.276 m,
      against 0.283 m). That is within the spread of mapper seeds on this
      capture, 0.15 to 0.65 m.
  - The ETH3D and SmallCity numbers documented before this change were
    measured with CPU SIFT and without completion, unless noted.

- **`gsplat_photos` logs how long decoding the photos took**, on its
  `N images at WxH` line. COLMAP's extraction times include decoding;
  `gsplat_photos`' SIFT times do not.

- **COLMAP-port mapper: faster loading and model export, byte-identical.**
  - **Loading:** the replay export's feature files are parsed in
    parallel, and the correspondence graph looks up a pair's two images
    once instead of per match and flattens images in parallel.
  - **Export:** the COLMAP text writer
    (`write_colmap_reconstruction_for_3dgs_with_cameras`) appends numbers
    to its buffers instead of allocating a string each, keeps keypoint ids
    in dense arrays instead of maps, and formats `points3D.txt` and
    `images.txt` in parallel chunks joined in order.
  - **SmallCity pass 1** (1,499 photos, 4 threads): loading 6.3 s instead
    of 8.8 s, export 3.1 s instead of 8.4 s, and the model files are
    byte-identical.
  - `colmap_incremental_mapper` now logs both times.

- **COLMAP-port mapper: about half the memory, byte-identical.** SmallCity
  pass 1 (1,499 photos, 4 threads) writes the same model files as before:

  | | before | now |
  |---|---|---|
  | Peak memory | 4.90 GB | **2.60 GB** |
  | Memory after loading | 2.94 GB | **0.70 GB** |
  | Loading | 6.5 s | **0.9 s** |
  | Mapper wall time | 5:03 | **4:25** |

  - **All of SmallCity** (5,822 photos, one run on an NVIDIA L4 pod with
    13 CPU threads): peak memory **9.0 GiB** instead of 18.9 GiB (the run
    in `docs/gpu_bundle_adjustment.md`, an L4 pod with 15 threads), and
    2.7 GiB instead of 11.3 GiB after loading.
    - **Accuracy:** the model registers 5,816 photos with a median
      camera-centre error of 0.184 m (p90 0.276 m) against the release's
      model.
    - **Why the models differ:** the two runs' VLAD retrieval picked
      slightly different candidate pairs (242,917 against 242,592), so the
      mapper's input differed. The earlier model scored 0.297 m. Both lie
      within the 0.15 to 0.65 m that mapper seeds alone give on this
      capture.
    - **Wall time:** the newer pod ran the unchanged stages 1.6 to 2.0
      times slower, so the wall times (1,380 s against 847 s) do not
      compare. The pass-1 replay above times both versions on one machine.
    - **What remains:** the largest working sets are the last global
      bundle adjustments', on 15.1M observations: 2.1 GiB of per-point
      linearizations and 0.9 GiB for the reduced camera system.
  - **Correspondence graph.** The replay export's graph is built in one
    pass (`CorrespondenceGraph::finalized_from_pairs`), equal field for
    field to `add_image`, `add_two_view_geometry` and `finalize`, without
    the per-keypoint lists the incremental path keeps until `finalize`. The
    export's matches stay in one `u32` buffer. Correspondences and each
    image's offsets are 32-bit, as COLMAP stores them: `Correspondence`'s
    fields are now `u32`, and `add_image` panics on an id or a keypoint
    count past `u32::MAX`.
  - **Bundle adjustment:**
    - The elimination finds each pair's block through a per-row table of
      partner positions instead of an index over every point's pairs
      (1.9 GB on SmallCity's last bundle adjustment).
    - Observations name their camera in a table instead of carrying its
      intrinsics and sensor pose: 24 bytes each instead of 112 (1.3 GB
      there).
    - Systems from 20,000 off-diagonal blocks are eliminated by output, as
      those from 1,000 frames already were. The sharded loop kept 8 copies
      of every block: 825 MB at 358k blocks, pass 1's peak. By output, pass
      1's eliminations from 20k to 400k blocks took 0.59 to 0.90 of the
      time.
    - Each row of the reduced system is written into the block array in
      place, instead of into its own buffer and then concatenated.

- **`gsplat_photos` loads large photos faster, with the same pixels.** Its
  photos go to the working size through a Triangle resize that performs
  `image::imageops::resize`'s arithmetic in the same order, so the working
  images are byte-identical. It sums blocks of source rows in registers and
  reads the intermediate row by row, where the `image` crate reads one
  pixel at a time and walks its intermediate down columns.
  - **Other changes:** the decoded RGB image is moved, not copied, and
    each file is read once for both the decoder and the EXIF focal.
  - **One 24-megapixel ETH3D photo to 1,600 px, one thread:** the resize
    takes 108 ms instead of 261 ms, and decoding plus resizing 325 ms
    instead of 552 ms.
  - **ETH3D pipes on 4 threads:** loading takes 1.3 s instead of 2.0 s, and
    every output file is byte-identical.

- **COLMAP-port BA: cheaper setup of large bundle adjustments.** Results
  are bit-identical: the SmallCity pass-1 model matches byte for byte.
  - **First linearization.** A solve's first linearization zeroed its
    camera-point block array on one thread (2 GB on SmallCity's last
    bundle adjustments). It is now zeroed in parallel, which also spreads
    the first touch of its pages over the threads.
  - **Slot index.** The reduced system's pattern and the elimination's
    contribution index each built the same list of observing points per
    frame. It is now built once, in parallel over chunks of points.
  - **SmallCity pass 1** (1,499 photos, 4 threads, one run each; the
    machine varies by about 10% between runs): the large bundle
    adjustments' linearization takes 5.8 s instead of 13.5 s, all global
    bundle adjustments 208 s instead of 258 s, and the mapper 364.4 s
    instead of 391.6 s.

- **COLMAP-port mapper: global track completion skips tracks it cannot
  extend.** Results are bit-identical: all 13 local ETH3D scenes and the
  SmallCity pass-1 model match byte for byte, at 4 threads and at 1.
  - **Which tracks change:** a track can only gain a 2D point that one of
    its observations corresponds to, of a registered image and without a
    3D point. A completion pass only adds observations.
  - **Change:** `CompleteAllTracks` now runs only on the tracks that have
    such a correspondence. It finds them in one pass from those 2D
    points, since the correspondence graph stores every match both ways.
  - **SmallCity pass 1** (1,499 photos, 4 threads): 15% of the tracks
    qualify. Global track completion takes 10.4 s instead of 19.4 s,
    including the 3.3 s scan.

- **COLMAP-port mapper: cheaper triangulation, track completion and
  large-BA setup.** Results are bit-identical: all 15 ETH3D models (13
  scenes) and the SmallCity pass-1 model below match byte for byte
  (`cameras.txt`, `images.txt` and `points3D.txt`).
  - **Why:** a perf profile of a SmallCity replay (pass 1, 1,499 photos,
    4 threads) ranked three costs first:
    - the track-completion search, 15% of CPU time;
    - the DLT triangulation's SVD, 13%;
    - one part of a large global BA's setup, 8%: finding each point's
      frame pairs in the edge rows of the reduced camera system.
  - **DLT triangulation.** nalgebra's SVD gives the same `V` without `U`,
    and on a fixed-size matrix as on a `DMatrix`. A check on 1,000,000
    random systems of 2 to 8 views found them bit-identical. Systems of
    up to four views now skip `U` and the heap. A 2-view SVD takes
    1.33 µs instead of 1.96 µs.
  - **Track completion.** The search now marks a 2D point visited only
    after the registration and 3D-point checks. Those checks reject the
    track and every triangulated 2D point on every visit, so the visited
    set holds only the candidates.
  - **Large global BA setup.** Up to 16,384 frames, the edge rows are
    marked straight into a bitmap per row over the frame slots. They were
    gathered, sorted and deduplicated. A frame pair's edge index is now
    its row's offset plus a popcount rank, not a binary search of the row.
  - **SmallCity pass 1, before and after** (two runs each; the machine
    varies by about 10% between runs):
    - mapper: 378.7 and 412.7 s before, 337.7 and 342.6 s after;
    - setup of the 1,498-frame global BA: 1.19 to 1.24 s before, 0.27 to
      0.31 s after;
    - global BA: 132 to 151 s before, 108 s after;
    - local track merging and completion: 73 to 78 s before, 65 to 66 s
      after;
    - triangulation: 24 to 26 s before, 20 s after.
  - **ETH3D scenes** are small, and their times do not change measurably
    (facade: 11.2 s before and after).

- **COLMAP-port mapper: cheaper local refinement.** Results are
  bit-identical: all 15 ETH3D models match (`images.txt` and
  `points3D.txt`).
  - **Why:** on SmallCity, local refinement is a third of the mapper's
    time (465 s of 23:20):
    - local BA 177 s;
    - track merging and completion 124 s;
    - point filtering 112 s.

    A callgrind profile of ETH3D facade put the track-completion search
    (`plan_complete`) first, at 10% of all instructions.
  - **Track completion.**
    - `CompleteTracks` and `CompleteImage` run back to back. The second
      searched again from most of the new image's 3D points, which the
      first had just searched.
    - A search that added nothing, and examined no 2D point the first
      pass then gave a 3D point, would add nothing again. The image pass
      now reuses it. On facade, 88% of these searches are reused.
    - A completion plan now records only the 2D points that can still
      gain a 3D point during the pass: those of registered images
      without one. These are the only ones that can make the plan stale.
      The search also checks a 2D point's 3D point before its image's
      registration.
    - A 2D point without a 3D point builds its correspondences' data (a
      camera copy and a pose each) only when it is about to be
      triangulated. Most stop earlier, because a correspondence already
      has a 3D point.
  - **Point filtering.**
    - `AdjustLocalBundle` filters the points of the window images, then
      the modified points, and most of the second set is in the first.
    - A point the first filter left untouched would be untouched again,
      because filtering changes no other point and moves nothing. The
      second filter now skips those points: on facade it checks 38
      points instead of 4,197.
    - The filters' pose table holds a 4-byte slot per image id instead
      of the poses themselves.
  - **Local BA setup.** A window's points are deduplicated with a bitmap
    over point ids before sorting, instead of sorting every sighting.
  - **ETH3D, before and after:**
    - On facade (76 images), local refinement takes 3.63 s instead of
      4.05 s, global refinement 3.99 s instead of 4.26 s, and the mapper
      11.6 s instead of 12.5 s.
    - The smaller scenes change only within noise.
    - Per facade local refinement: track completion takes 3.7 ms instead
      of 5.8 ms, filtering 4.6 ms instead of 6.2 ms, and local BA setup
      2.7 ms instead of 3.3 ms.

- **COLMAP-port BA: cheaper gradient stop and PCG product.**
  - **Gradient stop.** After an accepted step, nearly every solve stops
    on the gradient tolerance at the top of the next iteration. That
    check reads only `Jᵀr`, but it used to come after a full
    re-linearization. A gradient-only pass now decides it first. It
    runs the same code, so the sums and their order are the same. The
    `JᵀJ` blocks are built only when the solve goes on, and not at all
    after the last iteration. Results are bit-identical: ETH3D outputs
    match on all 13 scenes, and the mapper is about 6% faster there
    (37.5 s → 35.3 s). `BA_PHASES` reports the passes as `gradient_ms`.
  - **PCG product.** The edge-list PCG of large global BAs (1,000+
    free frames) computes each off-diagonal block's two products
    (`B x_a` and `Bᵀ x_b`) in one fused pass over the block. It no
    longer converts every block to an f64 matrix first. Each iteration
    is about 25% faster on the 4-thread `large_global_ba` bench.
  - **SmallCity measurement** (#276–#278, before and after on one pod):
    - Mapper time 27:53 → 27:01. Per step, elimination is 3.8× and
      linearization 1.9× faster.
    - The rounding changes moved the reconstruction to a different
      path. One 2,374-frame global BA then ran 25 LM steps instead of
      one, and PCG iterations doubled, eating most of the gain.
    - The median camera-centre error went from 0.640 m to 0.210 m
      (p90 2.19 m → 0.75 m). The previous binary reproduced 0.640 m
      exactly, so the mapper is deterministic and the change comes from
      the code. A single run cannot say whether the gain is systematic.
    - **Follow-up: the gain was not systematic.** The same replay
      with this release (#280) gives 0.611 m (p90 2.10 m) in 21:00. Each
      of the four passes alone fits the reference to about 0.1 m, but
      their fitted scales differ by up to 7% (3.92 to 4.19). So the
      centre error is the drift between passes, not BA precision, and
      it moves with the reconstruction path.
    - **PCG tolerance 0.1 tried and rejected** (`VISLOC_PORT_PCG_TOL`,
      same pod). The looser solves made the large BAs take 143 LM steps
      instead of 77, and their PCG iterations went from 10,249 to
      22,972. Wall time rose from 21:00 to 24:03. The median error was
      0.522 m, inside the run-to-run spread. The default stays 1e-2.

- **COLMAP-port mapper: fewer allocations and cheaper scoring.** All
  three changes are bit-identical; ETH3D outputs match the previous
  version on all 13 scenes.
  - A linearization keeps every point's `Jpose^T Jpoint` blocks in one
    array instead of one vector per point (hundreds of thousands of
    allocations per global BA), and the LM loop reuses the previous
    array. One linearization of a 3M-observation problem went from
    190 ms to 132 ms; the `large_global_ba` solve from about 8 s to
    6.5 s.
  - PnP RANSAC scores each hypothesis by inlier count and mean error
    only. It used to sort every inlier error per hypothesis for a median
    and maximum that only the final pose reports.
  - Track completion and merging look up their claimed and visited sets
    with the crate's FxHash-style hasher instead of SipHash, about 15%
    faster.

- **Less per-image overhead in the COLMAP-port mapper on large models.**
  The SmallCity replay also spent 435 s in local refinement outside the
  solver core: point filtering 107 s, choosing the local window 63 s,
  and linearizations in every solve. The filter and window changes are
  bit-identical. The linearization rounds differently: its sums are
  grouped by point shard, and the pose Jacobian is formed as
  `[J_point | -J_point [p]x]`. It stays deterministic at any thread count
  (new test).
  - **Linearization** runs in a fixed number of point shards. Each shard
    adds its per-frame blocks into its own accumulators, and the shards
    are summed in order. The old code wrote 336 bytes per observation to
    a buffer and folded it serially, which was a third of the time. A
    point's linearization also no longer copies its frame list. On the
    `large_global_ba_phases` bench (3M observations) one linearization
    went from 350 ms to 190 ms.
  - **Point filters** compute each observing image's pose and
    projection centre once per pass, in one table shared by both
    filters. They used to compose every observation's pose through its
    frame and rig. The triangulation-angle filter also no longer
    collects and sorts every observation's image id.
  - **`find_local_bundle`** counts shared points through a table indexed
    by image id instead of a `BTreeMap` and a `HashMap`, and takes the
    75th-percentile angle with a selection instead of a sort.

  On the 13 ETH3D scenes every centre error is unchanged to the reported
  digit; mapper time is 41.1 s → 39.6 s (facade, the largest, 17.2 s →
  15.6 s).

- **Faster COLMAP-port bundle adjustment.** Profiling a SmallCity replay
  (5,816 images, 18.5 min mapper) put 400 s in global-BA solves and
  76 s in local-BA solves, nearly all of them a single LM step. Three
  changes:
  - A solve that stops on the function tolerance no longer
    re-linearizes at the accepted point first. Ceres does not either. The
    old order spent one of every two linearizations of a one-step solve
    on nothing.
  - The Schur complement's per-pair term reuses the point's
    inverse-Hessian-scaled block and accumulates with fused
    multiply-adds (it recomputed the scaled block per pair).
  - A point's frame pairs are stored frame-major, so the per-row
    elimination of the iterative path reads them contiguously. It also
    adds each point shard's partial sums once at the shard's end instead
    of tracking shards per block.

  Both elimination paths stay bit-identical to each other (new test). On
  a synthetic 1,200-frame / 3M-observation global BA (`large_global_ba`
  bench, 4 threads) the solve went from 11.9 s to 8.1 s: elimination
  6.1 → 3.1 s, linearization 3.0 → 2.0 s. On the 13 ETH3D DSLR scenes
  (intrinsics refined) every centre error is unchanged to the reported
  digit, and mapper wall time fell from 51.5 s to 41.1 s.

- **`gsplat_photos` keeps up to 8192 SIFT detections per image (was
  4000), COLMAP's `max_num_features`.** On textured scenes 4000 kept
  about half of COLMAP's features (playground: 5,934 against 10,000-13,500
  per image), and the photos the port could not register were the ones
  with the fewest verified matches. COLMAP's own mapper on our matches
  dropped the same photos, so the gap was in the features. On the 13
  ETH3D DSLR scenes (1600 px, fixed focal), 8192 registers what COLMAP
  4.2 registers on every scene:

  | scene | 4000 | 8192 | COLMAP 4.2 |
  |---|---|---|---|
  | playground | 35/38, 0.784 cm | 38/38, 0.628 cm | 38/38, 0.619 cm |
  | facade | 75/76, 6.835 cm | 76/76, 6.688 cm | 76/76, 6.541 cm |
  | delivery_area | 2.542 cm | 2.354 cm | 2.567 cm |
  | courtyard | 4.568 cm | 4.431 cm | 4.427 cm |
  | terrains | 0.551 cm | 0.526 cm | 0.580 cm |
  | meadow | 5.878 cm | 5.839 cm | 5.009 cm |
  | terrace | 1.949 cm | 2.049 cm | 1.867 cm |
  | relief | 0.782 cm | 0.855 cm | 0.800 cm |
  | electro | 3.431 cm | 3.506 cm | 3.431 cm |

  Centre RMSE after a Sim(3) fit; office, pipes, relief_2 and kicker
  (under 4000 detections an image) are unchanged. The 13-scene mean RMSE
  goes from 2.45 to 2.41 cm. An earlier SmallCity run with 8000 (before
  the PnP fix) cut the median centre error from 1.66 m to 0.65 m. CPU
  matching takes 2-3.5x longer (facade 317 s -> 1100 s). `--max-keypoints
  4000` restores the old cap.

- **CPU descriptor matching runs on a 16-bit integer kernel for byte
  descriptors, about 2x faster.** `BruteForceMatcher::
  match_descriptors_cross_checked` (the CPU matcher of `gsplat_photos` and
  `gsplat_euroc`) computed every dot product with an f32 matrix product,
  stored the `n x m` result and scanned it for the top two in both
  directions. Descriptors that are bytes over a power-of-two scale
  (`gsplat_photos` keeps COLMAP's bytes / 512) now go through an AVX2
  `pmaddwd` kernel on 4 x 16 tiles with the top-2 scan fused in. Such dot
  products are exact in both f32 and integers, and rows are visited in the
  same order, so the matches are bit-identical; other descriptors keep the
  f32 path. 11,000 x 11,000 descriptors: 1.06 s -> 0.50 s a pair on one
  core (`match_bench` example).

- **COLMAP-port BA builds its problem point by point.** The native solve
  used to list every observation image by image (with a camera id each),
  copy that list twice, and have the solver regroup it by point with a
  stable sort. It now builds each point's observations straight from its
  track, in the order the regrouping produced (in-window elements by
  image and 2D point, then the pulled-in ones), so the problem and the
  results are bit-identical. On the Courthouse replay this takes BA setup
  and problem build from 5.7 s to 4.0 s, and peak memory from 618 MB to
  550 MB. Courthouse output is byte-identical.
- **The correspondence graph looks images up with an integer hash.** It
  used SipHash on the image id, which took most of `find_correspondences`,
  the inner loop of track completion and merging. Courthouse
  complete/merge in global refinement takes 5.9 s instead of 7.0 s.

- **The COLMAP-port BA's PCG is preconditioned with runs of 32 frames.**
  Block-Jacobi dropped every coupling between cameras. Cameras next to
  each other in capture (frame-id) order share most points, so the
  preconditioner is now a dense inverse over each run of 32 consecutive
  free frames. It falls back to block-Jacobi when a run is not positive
  definite. On a SmallCity mapper replay (same machine and inputs), PCG
  took 30,634 iterations instead of 50,393, the solves 405 s instead of
  612 s, and the mapper 904 s instead of 1,088 s, with a comparable result
  (centre error median 0.87 m against 0.96 m). `VISLOC_PORT_PCG_CLUSTER=k`
  changes the run length; `1` restores block-Jacobi. Systems under 1,000
  free frames do not use PCG and are unchanged.
- **VLFeat-compatible SIFT is 2.2x faster on the CPU, bit-identically.** The
  separable blur now runs a kernel tap at a time over whole rows (contiguous
  and vectorizable, same per-sample summation order), and the extremum scan
  stops as soon as a sample fails the threshold or both neighbour tests.
  One 1,600 x 1,066 photo: 5.72 s -> 2.64 s, with the same keypoints and
  descriptors bit for bit (`crates/vision/examples/sift_vlfeat_bench.rs`
  prints a checksum).
- **`gsplat_photos` CPU matching computes one descriptor product per pair**
  instead of two (`match_descriptors_cross_checked` instead of
  `CrossCheckMatcher`): 3.6x faster on 8,000-keypoint photos, identical
  matches.
- **`gsplat_photos` frees SfM inputs it no longer needs.** Byte-identical
  models. Grey images for SIFT are made 256 at a time instead of all at
  once (4 bytes a pixel: 16.5 GB for SmallCity's 5,822 photos). With
  `--no-refine-intrinsics` the SIFT descriptors are freed before the mapper
  (13 GB at 4,000 keypoints). The port's in-memory `pairs.bin` is dropped
  once written. A 553-image Courthouse run (CPU, 800 keypoints) peaks at
  972 MB instead of 1,479 MB. SmallCity with `--retrieval 60` had been
  killed on a 50 GB machine right after matching.
- **`gsplat_photos --retrieval` skips the window's own neighbours.** The
  VLAD top-k used to rank every image. On the SmallCity capture, three
  quarters of the k slots went to images already paired by the sliding
  window, leaving the four passes around the block weakly tied. They drifted
  apart in scale by up to 11%, and passes 2 and 3 bent by metres. Retrieval
  now picks its k among images more than `--window` positions away, which
  adds 126,641 pairs instead of 44,063, with matching time unchanged. Scored
  against the release's own COLMAP model by camera centres after a Sim(3)
  alignment: median error 1.82 m -> 0.88 m, p90 3.49 m -> 2.03 m, max
  8.71 m -> 3.36 m. Within-pass error drops from 1.80 m / 1.11 m to
  0.15 m / 0.05 m on passes 2 and 3. 5,549 instead of 5,581 photos are
  registered. `scripts/score_smallcity_reference.py` fetches that reference
  (about 300 MB of the 3 GB release, by HTTP range) and scores a model.
- **`gsplat_photos` matches COLMAP-style u8 descriptors on the GPU by
  default** (`FeatureBank::upload_u8`: round(512 x) clamped to 255, exact
  integer dot products), 2.2x faster than the previous f32 kernel on an
  NVIDIA L4 (0.92 -> 0.41 ms per cross-checked pair at 4,000 SIFT) with a
  quarter of the descriptor memory. It keeps 99.5% of the f32 matches on
  Tanks and Temples Courthouse; over three SfM settings there, it
  registered 553, 551 and 441 of 553 frames against f32's 550, 439 and
  552 (the mapper's model splits are sensitive to tiny match changes).
  `--match-f32` restores f32 matching.
- **Faster large-scene reconstruction.** Same results: the f32 GPU
  matcher is 1.35x faster (generated by `scripts/gen_match_wgsl.py`), TSDF
  fusion skips blocks outside each view's frustum, VLAD retrieval computes
  its similarities as GEMMs, k-means++ seeding is O(k), and match
  verification overlaps GPU matching. Two-view verification is 2.1x faster:
  the five-point RANSAC's minimal-sample cheirality test is PoseLib's closed
  form instead of one SVD per point (same reports on the benchmarks).
  Approximate: past 1,000 free frames,
  global BA solves the reduced camera system with preconditioned CG
  (relative residual 1e-2; Ceres' `ITERATIVE_SCHUR` default is 0.1);
  smaller problems are unchanged.
- **COLMAP-port bundle adjustment hands the solver arrays, not maps.**
  Byte-identical models. The problem used to be built as a
  `BundleAdjustment`, with a `Camera` clone (a heap-allocated parameter
  vector) per observation and `BTreeMap`s of every pose and landmark, which
  the native solver then converted back into arrays. It is now built
  directly as compact observations plus frame and point arrays. The
  Courthouse mapper takes 24.9 s instead of 29.1 s (mean of two runs, 4
  cores, iterative solve forced on).
- **COLMAP-port mapper 2.2x faster on SmallCity** (5,822 photos: 3,866 ->
  1,782 s). Most of it keeps the model byte-identical. Global BA's Schur
  elimination now runs row by row, in parallel over its outputs, and its
  problem build is 2.4x faster. `CompleteImage`, `TriangulateImage` and the
  negative-depth filter are speculatively parallel. `FindLocalBundle`
  indexes the image's shared points once. Together these bring the mapper
  to 2,477 s. Two BA changes alter results:
  - A failed linear solve is now an invalid step, as in Ceres: the trust
    region shrinks and the solve continues, and five failures in a row end
    it. Before, one failure discarded the whole BA, including every
    accepted step.
  - LM stops once an accepted step lowers the cost by less than 1e-5. This
    deliberately deviates from COLMAP's `function_tolerance=0`; several
    global BAs had crept to the 50-iteration cap at ~1e-6 relative decrease
    per step. `VISLOC_PORT_BA_FUNCTION_TOLERANCE=0` restores COLMAP's
    behaviour.
  - The PCG relative residual tolerance is 1e-2 instead of 1e-3. On a
    SmallCity mapper replay this halves the PCG iterations per global solve
    (356 -> 190) and the mapper takes 1,184 s instead of 1,686 s. The main
    model registers 5,599 images instead of 5,606, with a median
    camera-centre difference of 0.2% of the scene extent, which is within
    the spread of the other solver variants tried.
    `VISLOC_PORT_PCG_TOL=1e-3` restores the previous tolerance.
  - The PCG product reads each camera-pair block once, in f32, instead of
    reading both triangles in f64. On a SmallCity mapper replay a PCG
    iteration takes 5.6 ms instead of 18.5 ms, PCG time drops from 414 s
    to 144 s and the mapper from 1,122 s to 836 s. The main model
    registers 5,581 images instead of 5,599, with a median camera-centre
    difference of 0.19% of the scene extent.
    `VISLOC_PORT_PCG_F64_BLOCKS=1` keeps f64 blocks.

  On SmallCity the main model keeps 5,569 of 5,583 images (scattered
  marginal frames swap in and out) and 956,701 of 957,418 points, with
  camera centres within 0.12 % (median) / 0.96 % (max) of the scene extent.
  On Courthouse the one global BA that used to fail now succeeds: 544
  images registered (was 543).
- **Further mapper speed-ups (byte-identical output).** Courthouse replay:
  47.6 -> ~34 s.
  - The reconstruction's images, frames and 3D points live in id-indexed
    tables instead of `BTreeMap`s. `Reconstruction::{images,frames,points3d}()`
    now return `&IdMap<_>`, with the same read API.
  - Global BA's linearization runs in cache-sized chunks, its edge pattern
    is built per frame, and its elimination computes each row of the
    reduced system in one pass. On a half-scale SmallCity-shaped benchmark
    (`profile_synthetic_city`), linearization is 3-4x faster.
  - The track-completion BFS uses a hashed visited set.
  - A fixed-budget PnP RANSAC scores its hypotheses in parallel, behind
    `visloc-vision`'s new `parallel` feature. `PoseEstimator` now requires
    `Sync`.
- **Large source files split into submodules (no behavior change).**
  `pipelines/slam/src/incremental_sfm.rs` (18.5k lines), `bundle.rs` (17k),
  `examples/unordered_sfm_demo.rs` (22.5k, now
  `examples/unordered_sfm_demo/main.rs` plus modules) and
  `pipelines/basalt/src/vio/aom.rs` (21.5k) are now directories of files
  mostly under 3k lines. Pure code moves: public paths, test names and test
  counts are unchanged, and the Basalt parity arithmetic is untouched.
- **Planning docs archived.** The long `PLAN.md` log and the itemized 0.2.0
  change log moved to `docs/archive/`; `PLAN.md` is now a one-page handoff.

## 0.2.1 - 2026-10-08

Patch release: camera-distortion correctness fixes. Each fix below corrected
results that were silently wrong, not just errors that were reported.

### Fixed

- **COLMAP camera export with self-calibrated distortion.** A `Pinhole`
  camera carrying the refined radial `[k1, k2]` tail
  (`--refine-distortion`) was written as `PINHOLE` with 6 params, which is
  not valid COLMAP (`PINHOLE` has 4); COLMAP rejects it and other readers
  silently drop the distortion. Every COLMAP writer (text and binary) now
  exports it as `OPENCV` with `p1 = p2 = 0` (the identical projection), and
  fallible writers reject any camera whose parameter count does not match its
  COLMAP model. New public helper `visloc_io::colmap::colmap_camera_record`.
- **OPENCV tangential distortion.** `Camera::project` / `normalize_pixel` /
  `unit_ray_from_pixel` ignored `p1, p2` of an `OpenCv` camera, so a COLMAP
  `OPENCV` model with non-zero tangential terms was silently treated as
  radial-only. Such cameras now use the full Brown-Conrady model (shared with
  `FULL_OPENCV`, k3..k6 = 0); results with `p1 = p2 = 0` are bit-identical to
  before. New `Camera::tangential_distortion`.
- **Pose/structure BA ignored self-calibrated radial distortion.** With
  `refine_intrinsics = false` (e.g. the SfM polish / warm-start passes after
  `--refine-distortion`), `optimize_weighted` built its residuals and
  Jacobians with an undistorted pinhole while the cost it accepts steps
  against used the distorted `Camera::project`, so LM optimised a different
  model than it measured and could stall. The monocular residual and Jacobian
  (serial, parallel, and dense-QR assembly, plus the GNC residual vector) now
  apply `1 + k1·r² + k2·r⁴` with an analytic Jacobian; distortion-free cameras
  keep the historical closed form bit-for-bit.

### Added

- **`BaConfig::shared_focal` / `--shared-focal`** (unordered and sequential
  SfM demos): constrain intrinsics self-calibration to `fx == fy`. Previously
  `fx` and `fy` were always refined independently, so weakly observable
  motion (near-pure forward translation) could settle on a spurious aspect
  ratio. Default off; the joint solve is unchanged when it is not set.

## 0.2.0 - 2026-09-15

### Highlights

- **Real-data SfM at scale.** All ten ETH3D low-resolution many-view scenes:
  **9,996/10,008 cameras registered (99.88%)**, with no mapper run above
  3.32 GiB. Same-input ETH3D Electro 1,200 runs **3.46× faster than COLMAP**
  with **25.2% lower** camera-centre RMSE, and the 38-image courtyard control
  reaches **0.5379 cm vs COLMAP's 1.6166 cm**.
- **Visual-Inertial SLAM (faithful Basalt Rust port).** A tightly-coupled
  stereo-inertial VIO estimator plus offline mapper that matches native
  Basalt's ATE within **0.1%** on all 11 EuRoC sequences and, with EuRoC's
  official calibration, beats measured ORB-SLAM3 on **8 of 11** sequences.
- **COLMAP-compatible unordered SfM** with VLAD retrieval, essential-matrix
  verification, incremental mapping, connected-component models, and COLMAP
  text export for downstream 3DGS / MVS.
- **Online stereo SLAM** with loop closure, SE(3)/Sim(3) pose-graph
  optimization, GNC outlier rejection, and bundle adjustment, benchmarked on
  KITTI, EuRoC, and TUM RGB-D.
- **Opt-in deep frontend** (SuperPoint / LightGlue via in-Rust ONNX Runtime,
  CUDA-accelerated), plus public-data map-reuse localization and GNSS-prior
  tracking demos.

### Full change log

The itemized 0.2.0 change log (every Added / Changed / Fixed entry from
2026-05-07 to 2026-09-15, about 4,800 lines) is archived in
[`docs/archive/changelog_0.2.0_full.md`](docs/archive/changelog_0.2.0_full.md).

## 0.1.0 - 2026-05-07

### Added

- Workspace split into core, vision, IO, localization pipeline, and tracking pipeline crates.
- Core visual localization types: `Frame`, `Keyframe`, `VisualMap`, `Landmark`, `Observation`, `Camera`, `Pose`, and `LocalizationResult`.
- `SO3` / `SE3` pose wrappers and reprojection utilities built on `nalgebra`.
- Brute-force descriptor matching with L2 distance, ratio test, optional cross-checking, and match diagnostics.
- Minimal DLT PnP estimator, PnP RANSAC, pose-estimation diagnostics, and optional Gauss-Newton pose refinement.
- COLMAP text and binary map parsers for `cameras`, `images`, and `points3D`.
- Text parsers for landmark descriptors and query features.
- Localization pipeline over query descriptors and visual-map landmarks.
- Map providers, submap selectors, priors, localization quality gates, and map validation reports.
- Tracking skeleton with motion models, state transitions, and sequence examples.
- Local mapping skeleton with keyframe policy, local map windows, landmark candidates, linear triangulation, staged map updates, and local refinement hooks.
- Online SLAM MVP composition that combines tracking and local mapping without loop closure or global optimization.
- COLMAP text model writer for saving reusable sparse maps.
- Sensor-fusion foundation crate with timestamped frames/poses, GNSS/pose/IMU measurements, covariance types, measurement buffers, frame prior sources, and external localization-prior tracking hooks.
- GNSS-prior tracking example showing radius-submap narrowing before localization.
- COLMAP compatibility notes covering supported sparse model inputs, descriptor handling, writer behavior, and current limitations.
- Root crate prelude and top-level re-exports for common application-facing localization APIs.
- Pre-1.0 to v1.0 migration guide covering recommended imports, localization boundaries, COLMAP descriptor handling, tracking priors, and experimental layers.
- Package metadata and crate-content checks in the local quality gate and CI.
- crates.io package metadata now includes project homepage and repository URLs.
- Workspace member crates now use crate-specific descriptions and docs.rs URLs.
- Publishing guide documenting workspace crate publish order and package-check workflow.
- Examples, integration tests, design docs, local check script, and GitHub Actions CI.

### Not Yet Implemented

- Full Visual SLAM.
- Full SfM.
- Loop closure.
- Dense mapping.
- Full bundle adjustment.
- Full tightly-coupled visual-inertial or GNSS/INS fusion.
