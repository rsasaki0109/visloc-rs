# Visual SLAM literature survey and visloc-rs research map

This is a living, implementation-oriented survey. It records primary papers and
official implementations, the design claim each source supports, and the
experiment needed before that idea can be adopted in `visloc-rs`. A paper's
reported benchmark result is not treated as evidence for this repository until
the corresponding ablation is reproduced here.

## Research questions

1. Which data associations keep tracking continuous through blur, low texture,
   and revisits?
2. Which metric-stereo residuals preserve scale without turning depth noise into
   global deformation?
3. Which loop-verification sequence prevents a false closure from entering the
   graph while retaining useful recall?
4. Which combination of pose-graph optimization, map-point fusion, and bundle
   adjustment improves the full trajectory rather than only a prefix?
5. Which visual-inertial formulation makes scale, gravity, velocity, and biases
   observable and keeps their priors consistent after marginalization?

## Classical feature-based systems

| Work | Central contribution | visloc-rs hypothesis |
| --- | --- | --- |
| [MonoSLAM (2007)](https://www.robots.ox.ac.uk/ActiveVision/Papers/davison_etal_pami2007/) | Persistent probabilistic sparse map and active measurement in a real-time monocular EKF | Track/map uncertainty should drive search radii and gating rather than fixed pixel thresholds |
| [PTAM (2007)](https://ugweb.cs.ualberta.ca/~vis/courses/CompVis/readings/3DReconstruction/ptam.pdf) | Parallel tracking and mapping, keyframes, and bundle adjustment | Separate latency-sensitive tracking from local/global map optimization |
| [ORB-SLAM (2015)](https://arxiv.org/abs/1502.00956) | One repeatable feature family for tracking, mapping, relocalization, and loop closing; covisibility graph and keyframe culling | Reuse one descriptor identity across every association stage and aggressively fuse/cull redundant map elements |
| [ORB-SLAM2 (2017)](https://arxiv.org/abs/1610.06475) | Monocular, stereo, and RGB-D SLAM with monocular/stereo BA residuals and metric scale | Stereo loops must fix scale to one; optimize stereo reprojection residuals instead of recovering scale from two noisy 3-D clouds |
| [ORB-SLAM3 (2021)](https://arxiv.org/abs/2007.11898) | MAP visual-inertial estimation, Atlas multi-map recovery, and improved-recall place recognition | Verify a geometrically matched place through its local covisible keyframes, then fuse map points and optimize the affected map |

## Direct and hybrid systems

| Work | Central contribution | visloc-rs hypothesis |
| --- | --- | --- |
| [DTAM (2011)](https://www.doc.ic.ac.uk/~ajd/Publications/newcombe_etal_iccv2011.pdf) | Dense depth-map construction and whole-image tracking | A photometric rescue path can cover low-corner regions missed by feature-only tracking |
| [LSD-SLAM (2014)](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Engel2014ECCV.pdf) | Probabilistic semi-dense depth maps and Sim(3) pose graph for monocular scale drift | Depth variance must participate in alignment; Sim(3) belongs to scale-unobservable maps, not metric stereo |
| [DSO (2016/2018)](https://arxiv.org/abs/1607.02565) | Joint photometric window optimization with inverse depth, exposure, response, and vignetting calibration | Calibration and joint window optimization are stronger levers than adding more independent frame-to-frame matches |
| [Stereo DSO (ICCV 2017)](https://openaccess.thecvf.com/content_iccv_2017/html/Wang_Stereo_DSO_Large-Scale_ICCV_2017_paper.html) | Static-stereo photometric constraints and temporal multi-view constraints are jointly optimized in one active window; fixed baseline observes metric scale | Compare a joint left/right/temporal window residual against independent stereo triangulation plus monocular PnP, especially where the present similarity scale departs from one |
| [LDSO (2018)](https://arxiv.org/abs/1808.01111) | Direct odometry with repeatable feature-biased points, BoW loop detection, geometric Sim(3) verification, and PGO | Hybrid direct tracking plus descriptor-based long-term association is preferable to forcing one representation to do both jobs |
| [SVO (ICRA 2014)](https://doi.org/10.1109/ICRA.2014.6906584) | Feature detection for map structure combined with direct patch alignment and probabilistic depth filtering | Projection-guided tracking should refine patches photometrically at subpixel precision instead of relying only on descriptor rematching |

## Visual-inertial estimation

| Work | Central contribution | visloc-rs hypothesis |
| --- | --- | --- |
| [MSCKF (ICRA 2007)](https://www-users.cse.umn.edu/~stergios/papers/ICRA07-MSCKF.pdf) | Clone a sliding set of camera poses and eliminate feature positions to impose multi-state visual constraints in an EKF | Add a feature-elimination control arm and consistency/NEES tests; it is a bounded-compute VIO baseline, not a drop-in substitute for the requested mapping back end |
| [OKVIS (IJRR 2015)](https://github.com/ethz-asl/okvis) | Keyframe-based nonlinear optimization jointly estimates multi-camera reprojection and inertial states | Replace the one-shot IMU seed with visual and preintegrated inertial residuals in the same marginalizing window, then compare pose, velocity, bias, and runtime against visual-only |
| [OKVIS2 (2022)](https://github.com/smartroboticslab/okvis2) | Extends keyframe VIO to scalable SLAM with loop closure in the authors' implementation | Test loop correction through the VIO factor history and marginalization prior, rather than correcting poses while leaving inertial states locally inconsistent |
| [VINS-Mono (2018)](https://arxiv.org/abs/1708.03852) | Robust initialization, tightly coupled nonlinear VIO, failure recovery, loop detection, and gravity-aligned 4-DoF PGO | IMU factors and visual residuals must share the same optimization window; recovery must restore velocity and biases as well as pose |
| [ORB-SLAM3 (2021)](https://arxiv.org/abs/2007.11898) | MAP-based inertial initialization and continued visual-inertial BA | A one-shot IMU seed is insufficient; scale and inertial parameters need scheduled refinement after initialization |
| [Inertial-Only Optimization for Visual-Inertial Initialization (ICRA 2020)](https://arxiv.org/abs/2003.05766) | Holds the visual trajectory fixed and estimates velocities, gravity direction, scale, and IMU biases by MAP optimization before joint VI-BA | The stereo/known-scale initializer must convert calibrated camera poses to body poses and must not let free visual poses or independent per-keyframe biases absorb inertial residuals |
| [DM-VIO (2022)](https://arxiv.org/abs/2201.04114) | Delayed marginalization and pose-graph BA for consistent relinearization and scale/gravity updates | Keep a delayed factor history so initialization or loop corrections can rebuild inconsistent marginalization priors |
| [OpenVINS (ICRA 2020)](https://yangyulin.net/papers/2020_icra_ov.pdf) | FEJ-MSCKF research platform with explicit calibration, consistency, simulation, and Monte-Carlo evaluation | Add NEES/consistency tests and preserve unobservable directions; low ATE alone does not prove a consistent estimator |
| [Towards Consistent Visual-Inertial Navigation (ICRA 2014)](https://www.cs.cmu.edu/~kaess/pub/Huang14icra.html) | Projects linearized measurement Jacobians onto the correct observable subspace so estimation cannot gain fictitious information along gauge directions | Add invariant/gauge tests under global translation and yaw, and reject a VIO marginalization scheme that becomes overconfident even if its single-run ATE is low |
| [BASALT / Non-linear Factor Recovery (2019)](https://arxiv.org/abs/1904.06504) | Recover non-linear factors from marginalized VIO states for later visual-inertial mapping | Store recoverable factor summaries so global map corrections do not discard the local estimator's information |
| [Kimera (ICRA 2020)](https://arxiv.org/abs/1910.02490) and [Kimera2 (2024)](https://arxiv.org/abs/2401.06323) | Modular stereo VIO, robust PGO, meshing/semantics; later adds stronger tracking, keyframing, modalities, and GNC | Keep localization, robust global estimation, and dense mapping modular; benchmark each module and their integration separately |

## Learned correspondence and optimization

| Work | Central contribution | visloc-rs hypothesis |
| --- | --- | --- |
| [SuperPoint (CVPRW 2018)](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf) | One fully convolutional pass jointly detects points and descriptors; homographic adaptation supplies self-supervised repeatability | The current ONNX front end must be ablated by repeatability, match inliers, tracking continuity, and CPU latency—not credited merely because it is learned |
| [SuperGlue (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html) | Attention over both feature sets predicts a partial assignment through differentiable optimal transport, including unmatched points | Compare against brute-force cross-check specifically on loop/recovery pairs; adopt only if geometric recall rises enough to pay its latency |
| [DROID-SLAM (2021)](https://arxiv.org/abs/2108.10869) | Recurrent correspondence updates coupled to differentiable dense BA | Learned features alone are not the main gain; iterating correspondence and geometry is |
| [DPVO / Deep Patch VO (2023)](https://arxiv.org/abs/2208.04726) | Sparse learned patch tracking with recurrent updates and differentiable BA | A bounded sparse patch graph may deliver most of DROID's robustness at lower memory cost |
| [MASt3R-SLAM (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Murai_MASt3R-SLAM_Real-Time_Dense_SLAM_with_3D_Reconstruction_Priors_CVPR_2025_paper.pdf) | Pointmap priors, dense matching, graph construction, loop closure, and second-order global optimization | Foundation-model pointmaps are a future front-end/initialization option, but must be evaluated separately from the pure-Rust core |

## Place recognition and robust graph estimation

| Work | Central contribution | visloc-rs hypothesis |
| --- | --- | --- |
| [ORB-SLAM3 (2021)](https://arxiv.org/abs/2007.11898) | Geometry-first candidate check followed by local consistency in three covisible keyframes | Replace the current temporal-confirmation proxy with simultaneous candidate-neighborhood verification |
| [PCM (ICRA 2018)](https://par.nsf.gov/biblio/10354834) | Maximum pairwise-consistent measurement set as a maximum-clique problem | Batch PCM should select a consensus before any cold-start loop can poison the graph |
| [Switchable Constraints (IROS 2012)](https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf) | Jointly optimize pose and latent switches that disable false loop factors | Compare an explicit switch variable against GNC weights when false constraints form a coherent cluster |
| [GNC for Robust Spatial Perception (2020)](https://arxiv.org/abs/1909.08605) | Graduated non-convex robust estimation via Black-Rangarajan duality | GNC is a back-end safeguard, not a substitute for geometric/covisibility front-end verification |
| [TEASER (2020)](https://arxiv.org/abs/2001.07715) | Certifiably robust scale/rotation/translation registration under extreme correspondence outliers | Reserve for unknown-scale map merging or monocular registration; do not use it to re-estimate metric stereo scale |

The place-recognition front end has its own coarse-to-fine lineage:

The [official ORB-SLAM3 loop-closing implementation](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/src/LoopClosing.cc)
makes the paper's local-consistency step concrete: it gathers map points from
the candidate's ten best covisible keyframes, requires at least 20 BoW matches,
15 RANSAC inliers and 20 Sim(3) inliers, expands to at least 50 projection
matches and 80 after optimization, then checks the recovered transform against
the current keyframe's covisible neighbors until three keyframes agree. If the
region is not yet confirmed, it propagates the recovered transform into the
next current keyframe and permits two misses before resetting the hypothesis.
For stereo/RGB-D the solver fixes scale. visloc-rs now performs independent PnP
checks across current covisible keyframes against the candidate's covisible
landmark region and carries an unconfirmed region hypothesis across successive
current keyframes, allowing two misses before reset and requiring three
consistent detections. This is closer to ORB-SLAM3's local-consistency test,
The candidate can now optionally run a one-to-one projection-window rematch
from the primary PnP pose and a refined PnP gate. The pre-refinement match
floor defaults to ORB-SLAM3's 50 for the experiment runner, but remains
configurable because SuperPoint correspondence counts must be measured. The
second post-optimization 80-match stage and map-point fusion are not yet
equivalent, and this new path is not adopted until EuRoC evidence passes the
promotion gate.

| Work | Central contribution | visloc-rs hypothesis |
| --- | --- | --- |
| [FAB-MAP (IJRR 2008)](https://www.robots.ox.ac.uk/~pnewman/papers/IJRRFabMap.pdf) | A generative appearance model explicitly reasons about perceptual aliasing and the probability that an observation is a new place | Calibrate retrieval likelihood/novel-place rejection separately from geometry; a high cosine score must not be treated as a loop probability |
| [DBoW2 / Bags of Binary Words (2012)](http://doriangalvez.com/papers/GalvezTRO12.pdf) | Fast binary-descriptor vocabulary tree, inverted retrieval, and geometric verification | A mean local descriptor is a weak global image representation; compare it with a vocabulary/inverted index using the same local evidence |
| [NetVLAD (CVPR 2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.html) | End-to-end place-specific global descriptor trained with weak geographic supervision | Retrieval recall should be measured independently of local geometric verification |
| [HF-Net (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sarlin_From_Coarse_to_Fine_Robust_Hierarchical_Localization_at_Large_Scale_CVPR_2019_paper.pdf) | Shared network for global retrieval and local features in a coarse-to-fine localization pipeline | Cache one global descriptor per keyframe, then spend local matching only on retrieved candidate regions |
| [LightGlue (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html) | Adaptive-depth sparse feature matching with early stopping and point pruning | Replace brute-force loop matching only after measuring match recall, pose accuracy, and adaptive runtime on EuRoC revisits |

## Dense and neural map representations

| Work | Central contribution | Relevance now |
| --- | --- | --- |
| [iMAP (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/papers/Sucar_iMAP_Implicit_Mapping_and_Positioning_in_Real-Time_ICCV_2021_paper.pdf) | Online implicit neural scene representation jointly used for mapping and tracking | Dense-map research track; not a remedy for the current sparse trajectory error |
| [NICE-SLAM (CVPR 2022)](https://arxiv.org/abs/2112.12130) | Hierarchical local neural encoding for scalable dense RGB-D SLAM | Candidate representation for bounded local dense maps |
| [SplaTAM (CVPR 2024)](https://arxiv.org/abs/2312.02126) | Explicit 3-D Gaussian map used for RGB-D tracking, mapping, and rendering | Useful after metric pose quality is stable; rendering quality is not trajectory evidence |
| [Photo-SLAM (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_Photo-SLAM_Real-time_Simultaneous_Localization_and_Photorealistic_Mapping_for_Monocular_Stereo_CVPR_2024_paper.pdf) | Feature SLAM plus photorealistic Gaussian map for monocular/stereo/RGB-D | Supports keeping sparse geometric localization separate from photorealistic map optimization |

## Calibration and non-ideal sensing

These works define separate robustness tracks. EuRoC Machine Hall uses
calibrated global-shutter cameras and cannot validate rolling-shutter, event,
or dynamic-object claims; those changes require an appropriate dataset rather
than being inferred from an MH result.

| Work | Central contribution | visloc-rs hypothesis / required gate |
| --- | --- | --- |
| [Kalibr](https://github.com/ethz-asl/kalibr) | Joint spatial and temporal camera/IMU calibration, IMU intrinsics, multi-camera models, and rolling-shutter calibration | Import and report calibration covariance/time offset, then perturb extrinsics, delay, noise density, and bias random walk in simulation before trusting tight-VI gains |
| [Direct Sparse Odometry with Rolling Shutter (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/David_Schubert_Direct_Sparse_Odometry_ECCV_2018_paper.html) | Optimizes per-keyframe velocity with a constant-velocity prior so each image row has the correct capture pose | Add a row-time camera model only behind a calibrated readout-time flag and evaluate on rolling-shutter data; a global-shutter EuRoC improvement is no evidence for it |
| [UltimateSLAM (2017)](https://arxiv.org/abs/1709.06310) | Tightly couples events, conventional frames, and IMU to exploit complementary high-speed/HDR and low-motion regimes | Keep events as an optional measurement factor sharing the inertial state; test HDR, blur, and stationary intervals separately from standard-frame EuRoC |
| [DynaSLAM (2018)](https://arxiv.org/abs/1806.05620) | Semantic and multi-view motion detection excludes dynamic content and preserves a static map | Add geometric residual-consistency masks before a semantic dependency; measure static inliers, continuity, and false masking on dynamic-scene datasets |

## Multi-map, multi-session, and collaborative mapping

| Work | Central contribution | visloc-rs hypothesis / required gate |
| --- | --- | --- |
| [ORBSLAM-Atlas (2019)](https://arxiv.org/abs/1908.11585) | Starts a new accurate submap after ill-conditioned tracking loss, detects common regions, and nonlinearly merges/fuses disconnected maps | Preserve failed-session maps instead of forcing dubious recovery into one graph; require correct merge precision, metric scale consistency, duplicate-point fusion, and lower post-merge ATE |
| [maplab (2018)](https://tisl.cs.utoronto.ca/publications/maplab%3A%20An%20Open%20Framework%20for%20Research%20in%20Visual-inertial%20Mapping%20and%20Localization) | Provides multi-session visual-inertial map merging, loop closure, and batch optimization as an offline research framework | Export lossless factor/map state and compare online correction with a batch multi-session upper bound; pose-only CSV export is insufficient for VI re-optimization |
| [CCM-SLAM (JFR 2019)](https://www.research-collection.ethz.ch/handle/20.500.11850/313259) | Centralized collaborative monocular SLAM co-localizes agents and fuses shared map experience | Treat bandwidth, delayed/out-of-order updates, agent identity, and inter-map false closures as first-class tests before attempting distributed operation |

## Evaluation protocol

Use both absolute trajectory error and relative pose error; ATE alone can hide
local drift and a similarity alignment can hide metric-scale failure. See the
[TUM trajectory evaluation paper](https://jsturm.de/publications/data/sturm12iros_ws.pdf).

Every promoted configuration must report:

- rigid and similarity ATE, similarity scale, translational and rotational RPE;
- tracking coverage, longest continuous run, failure/recovery counts;
- retrieval candidates, PnP verification, covisibility confirmations, accepted
  loops, and post-hoc ground-truth loop precision;
- pose jump and landmark displacement mean/p95/max after every graph solve;
- wall-clock time, mean/p95 frame time, memory, and five-run variation;
- full-sequence MH_01, MH_03, and MH_05 results against a no-loop binary-identical
  baseline.

Use the same evaluator for every A/B directory:

```powershell
python scripts/summarize_visual_slam_evaluation.py `
  --run no_loop=E:/visloc_archive/<no-loop-run> `
  --run candidate=E:/visloc_archive/<candidate-run> `
  --output docs/generated/<sequence>-visual-slam-ab.md `
  --json E:/visloc_archive/<sequence>-visual-slam-ab.json
```

The candidate is promotable only when its accepted loops have 100% measured
precision at the documented 0.5 m / 10 degree GT threshold, rigid ATE is below
the binary-identical no-loop baseline, tracking coverage and longest continuous
run do not regress materially, and the runtime cost is explicitly reported.
Zero accepted loops proves safety but not improvement.

Run the binary-identical repeated experiment matrix sequentially so concurrent
inference does not contaminate runtime measurements. Keep the host free of
unrelated compute-heavy jobs for the entire matrix: the memory gate below
prevents allocation-starved runs from being accepted, but it cannot by itself
prove CPU exclusivity. Any overlap with an unrelated compute job invalidates
the affected runtime samples and requires a clean rerun. The runner records the
Git revision, executable and model hashes, exact argument vector, timestamps,
elapsed time, and exit status for every run:

```powershell
$env:ORT_DYLIB_PATH = 'E:/tools/onnxruntime-win-x64-1.23.2/lib/onnxruntime.dll'
./scripts/run_visual_slam_euroc_ab.ps1 `
  -DatasetRoot C:/Users/rsasa/Workspace/old/simple_visual_slam/euroc_mav/machine_hall/machine_hall `
  -OutRoot E:/visloc_archive/visual_slam_ab_20260713 `
  -Repetitions 5
```

The runner always passes `--max-frames`; its default `-MaxFrames 0` means every
available frame. This is intentional because omitting the demo option would use
the demo's 400-frame smoke-test default rather than a full sequence.
It fails before launching when the dynamic ONNX runtime DLL is unavailable and
records the Git revision, dirty status and diff fingerprint, executable/model/
runtime DLL hashes, exact arguments, Rust toolchain, OS, CPU, logical processor
count, installed memory, and 250-ms-sampled peak process working set. It also
requires at least 4 GiB of available physical memory and 4 GiB of Windows
commit headroom before each run, samples both every five seconds, and records
the preflight and minimum values in each run manifest. If either reserve is
crossed during a run, the runner stops only its benchmark child and records an
environmental validation error separately from the child's real exit code.
Such a run is not an accuracy or runtime sample. The thresholds and sampling
period can be changed explicitly with `-MinAvailableMemoryGiB`,
`-MinCommitHeadroomGiB`, and `-ResourceSampleIntervalSeconds` when the host has
a documented resource policy.

A dirty-tree run is therefore identifiable, while
the executable hash remains the authoritative identity of what actually ran.
For a long interrupted matrix, repeat the same command with `-Resume`.
Only runs whose per-run manifest has exit code zero and whose `summary.txt`
exists are skipped; an incomplete or failed directory is never overwritten
implicitly. Resume is also refused when executable, SuperPoint model, or ORT
DLL hashes, dataset root, sequence list, repetition count, or frame cap differ
from the existing experiment. The schema-v3 manifest also fingerprints the
common and per-variant argument protocol and the resource gate, preventing a
changed treatment or environmental validity rule from being resumed into an
older matrix. Each successful run is rejected by the
runner if `summary.txt` does not report the expected
`pose_graph_refinement=false` for `no_loop` or `true` for `appearance_loop`.

The matrix counterbalances order for each sequence: odd repetitions run
`no_loop` then `appearance_loop`, and even repetitions reverse that order to
reduce cache, thermal, and time-of-run bias. Both variants use the same
executable, feature tracking, stereo mapping, and localization settings. The
`no_loop` arm deliberately omits pose-graph refinement altogether: merely
disabling appearance retrieval is not a no-loop control because the graph
refiner also consumes shared-landmark loop candidates. The `appearance_loop`
arm adds the complete robust loop subsystem (PnP verification, PCM, GNC,
correction propagation, appearance retrieval), including a 15-pixel/50-match
projection-rematch gate and its three-keyframe/two-miss confirmation state.
PCM uses pairwise-only consistency in this candidate because MH_01 directly
demonstrates that the additional individual-vs-drifted-odometry gate rejects
true loops; the default library behavior remains unchanged.
Thus the measured treatment is the deployable loop-closure subsystem, while
“binary-identical” means the exact same executable rather than identical CLI
flags.
After the matrix completes, generate both per-run rows and sequence/variant
mean plus sample-standard-deviation rows, with loop precision pooled over all
ground-truth-evaluated accepted constraints:

```powershell
python scripts/summarize_visual_slam_evaluation.py `
  --matrix-root E:/visloc_archive/visual_slam_ab_20260713 `
  --output docs/generated/euroc-visual-slam-ab.md `
  --json E:/visloc_archive/visual_slam_ab_20260713/results.json
```

The generated conservative gate does not promote on a favorable mean alone:
the worst candidate rigid ATE must beat the best no-loop rigid ATE across the
recorded repetitions, every accepted loop must have a ground-truth verdict and
all must be correct, candidate tracking coverage may fall by at most 0.005,
the longest continuous run may fall by at most 1%, and runtime must be present
for every run. The sampled peak working set must also be present for every run.
`INCOMPLETE` means evidence is missing; `REJECT` means complete
evidence violates at least one safety or improvement gate.

## Ordered implementation experiments

1. Instrument every appearance candidate and graph solve before changing more
   thresholds.
2. Reject candidate regions connected to the current local graph, verify the
   candidate's covisible landmark region, and propagate an unconfirmed region
   across current keyframes until three geometrically consistent detections.
3. Add bidirectional, pyramid/noise-weighted reprojection validation and guided
   map-point matching.
4. Batch loops before optimization; compare batch PCM, GNC, and explicit switch
   variables on injected and naturally occurring false closures.
5. Fuse duplicate map points and run loop-side covisibility BA after SE(3) PGO.
6. Propagate stereo disparity uncertainty into triangulation, PnP, and BA.
7. Build a genuinely tightly coupled stereo-inertial window with consistent
   relinearization and delayed prior reconstruction.
8. Only then benchmark learned recurrent patch/pointmap front ends and dense map
   representations.

## Recorded diagnostic result

The EuRoC demo previously advanced past every IMU sample before the visual seed
without forwarding those samples to the stationary initializer. Its first
initialization window therefore began near takeoff and repeatedly reported
`GyroNoiseTooHigh` (roughly 0.08--0.25 rad/s observed versus a 0.05 limit)
before giving up. The demo now forwards pre-seed samples through an
initializer-only API: they estimate stationary gravity/bias but never enter
the first inter-keyframe preintegration factor. A regression test pins both
the buffered sample/duration count and zero preintegrator elapsed time. This is
an implementation correction, not yet evidence of better EuRoC ATE; the next
rebuilt run must report initialization status and downstream VI metrics.

The discarded temporal-confirmation proxy was run on all 3,660 MH_01 frames
with the same SuperPoint, stereo replenishment, projection-guided tracking,
SE(3), PCM, and GNC configuration used by the no-loop comparison. Tracking
coverage was 0.898. Of 976 ranked appearance candidates, 19 passed primary PnP;
17 remained deferred by the temporal gate and the remaining two were rejected
by PCM, so no loop entered the graph. Final-keyframe rigid ATE was 2.5740 m and
similarity ATE was 2.1022 m (scale 1.827028), matching the no-loop outcome. This
negative result supports testing spatial covisibility agreement rather than
waiting for consecutive frames to repeat a retrieval.

The first spatial-covisibility prefix run processed 1,000 MH_01 frames with
0.933 tracking coverage and 976.3 ms/frame wall-clock runtime. Sixty-seven
appearance candidates passed retrieval, five passed primary PnP, and all five
failed the three-current-keyframe covisibility gate; consequently no loop was
admitted. Live rigid ATE was 0.3020 m and final-keyframe rigid ATE was 0.3071 m.
This configuration is safe but does not beat the no-loop prefix and is not an
adoption candidate. Candidate-level neighbor-PnP and SE(3)-disagreement logs
are required before changing its thresholds.

A follow-up keyframe-density experiment forced a quality-gated keyframe every
30 frames (minimum 20 PnP inliers). It increased keyframes from 44 to 68 and
the mean covisibility local-map size from 69.4 to 165.7, but reduced tracking
coverage from 0.933 to 0.886, worsened final-keyframe rigid ATE from 0.3071 m
to 0.3491 m, and cost 959.3 ms/frame. Thirteen primary PnP hypotheses still
produced no accepted loop. This configuration is rejected: manufacturing a
denser graph does not repair inconsistent loop geometry and damages tracking.

With ORB-SLAM3's connected-region rejection added, the original 1,000-frame
tracking configuration rejected 39 of 67 retrieval candidates before PnP.
Only two primary PnP hypotheses remained; both belonged to the same region at
keyframe 893 and seeded one of three required cross-keyframe confirmations.
No loop was admitted, while tracking coverage (0.933), rigid ATE (0.3020 m),
final-keyframe rigid ATE (0.3071 m), and RPE remained identical to the safe
no-loop trajectory. A full-sequence run is required because the prefix ends
before that pending region can be confirmed or rejected.

The corresponding 3,660-frame MH_01 full run completed in 3,226 seconds
(881.4 ms/frame), with 0.898 tracking coverage and a 513-frame longest
continuous segment. Of 976 ranked candidates, connected-region screening
rejected 157 and 13 passed primary PnP. Eleven detections carried a pending
region forward and one region reached the three-keyframe confirmation gate,
but PCM rejected that confirmed constraint; zero loop constraints entered the
graph and no PGO solve ran. Live rigid ATE was 2.9940 m (similarity ATE
2.3053 m, scale 2.0896); the final-keyframe trajectory remained at rigid ATE
2.5740 m and similarity ATE 2.1022 m (scale 1.827028), with delta-1 RPE
0.3723 m / 6.89 degrees and delta-10 RPE 1.7562 m / 27.28 degrees. This proves
that connected-region rejection plus cross-keyframe continuation is safe on
this run but not an improvement. The run used the pre-fix binary whose VI
initializer still discarded stationary pre-seed IMU samples and gave up, so
both arms must be rebuilt and rerun after the initializer correction; its
trajectory cannot serve as the final binary-identical baseline for the new
candidate.

A rebuilt 1,000-frame MH_01 projection diagnostic used a deliberately
non-binding one-match minimum to measure the post-PnP rematch distribution.
All 29 primary-PnP candidates produced 232--606 one-to-one projection-guided
correspondences, so the ORB-SLAM3-inspired 50-match gate would not remove any
candidate on this prefix. Six refined PnP solves were rejected; their inlier
ratios (57/232 through 164/342) were all below the configured 0.5 threshold.
The other 23 passed projection refinement, three regions reached cross-
keyframe confirmation, and PCM rejected all three, leaving zero admitted loops
and zero PGO solves. The run tracked 0.947 of frames and had rigid ATE 0.5289 m
and final-keyframe rigid ATE 0.6911 m. The subsequently completed same-binary
no-loop prefix produced exactly the same tracking, ATE, RPE, and longest
continuous segment. `slam_trajectory.csv`, `keyframe_trajectory.csv`, and
`final_keyframe_errors.csv` were SHA-256-identical between arms, proving that a
candidate run with no admitted constraint is trajectory-side-effect-free.
It is nevertheless not promotable: mean runtime rose from 669.5 to
723.5 ms/frame (+8.1%) and p95 from 789.2 to 949.7 ms, with zero accuracy gain
and zero accepted loops. Both new-binary arms differ from the older
0.3020/0.3071 m prefix because correcting the shared projection matcher changed
the common frontend trajectory; the same-binary comparison is authoritative.
The run also buffered 216 pre-seed IMU samples but static VI initialization
still gave up. Direct IMU analysis shows the pre-seed 1.1-second interval has
per-axis gyro standard deviations 0.066/0.186/0.064 rad/s, above the 0.05
gate. Later 0.5-second low-variance windows are not truly stationary (for
example about 0.14 rad/s mean rotation), so shortening the detector window
would risk treating real motion as gyro bias and is not an acceptable fix.
MH_01 therefore needs motion-based visual-inertial alignment rather than a
forced static initializer success; the pre-seed API remains useful for
datasets that actually contain a stationary startup.

The pipeline has an explicit motion-start fallback for this case. With
`--motion-vi-init --motion-vi-init-after-static-give-up`, the motion-VI stage
remains gated until the static stage terminally gives up, then starts banking
keyframes and preintegration factors from the configured running IMU bias
linearisation instead of fabricating a stationary result. The existing default
remains off and still requires static success. The motion initializer's public
explicit-bias entry point reuses the same translation/keyframe excitation,
velocity, and nonlinear-solver gates; a regression test proves the opt-in path
collects post-give-up factors while the legacy test proves default gating is
unchanged.

That audit also found that the motion initializer's velocity sanity gate was
not transactional: the inner inertial BA wrote candidate poses into the live
map before the post-solve velocity limit rejected its state. The solve now runs
on a cloned map and commits only after every post-solve gate passes. The
velocity-rejection regression test snapshots the complete `VisualMap` and
proves it is unchanged after rejection. This correction applies to both VIBA1
and the optional VIBA2 scale loop and prevents a nominally rejected inertial
initialization from silently perturbing later visual tracking.

The first 1,000-frame MH_01 motion-start diagnostic still failed physically:
with every visual pose and every keyframe bias free, the solve absorbed visual
rotation into gyro biases of 1.65--5.72 rad/s. Campos, Montiel, and Tardos's
inertial-only formulation instead treats the already estimated visual
trajectory as fixed. After fixing all candidate poses and imposing a shared
short-window bias, the gyro estimate improved to roughly 0.25--0.46 rad/s, but
the accelerometer bias became 10.5--11.9 m/s2 (approximately one gravity). This
second failure exposed a frame error: the map stores camera poses, while the
IMU factor residual is defined on body poses. The initializer had omitted the
calibrated camera-to-body transform. This is load-bearing because EuRoC's
[`T_BS`](https://www.research-collection.ethz.ch/bitstreams/d861e63b-cfa9-4411-85a5-5ad6b3526e44/download)
specifies the sensor pose relative to the body/system frame.

The third same-trajectory run converted every fixed candidate camera pose with
the EuRoC `T_BS` before constructing inertial residuals. It succeeded at frame
117 using three keyframes and two preintegration factors: the shared gyro-bias
magnitude was 0.0854 rad/s, accelerometer-bias magnitude was 0.250 m/s2, and
the largest recovered velocity was 0.557 m/s. These are inside the deliberately
conservative diagnostic limits of 0.2 rad/s, 1.0 m/s2, and 10 m/s. Its
`slam_trajectory.csv` SHA-256 is exactly identical to both rejected runs and
the no-motion visual baseline (`7E6163AC...F19A6BE`); rigid live/final-keyframe
ATE therefore remained 0.5289/0.6911 m. This validates the frame conversion
and transactional non-write-back, but it does not promote tight VI fusion:
the three-keyframe trigger was intentionally aggressive and this single run
cost 803.1 ms/frame with a 1,568.2 ms p95.

The downstream local VI-BA audit subsequently found and corrected the same
frame conflation without changing visual reprojection semantics. Bundle pose
variables remain camera poses for feature residuals; only each IMU cost and
Jacobian composes `T_bw = T_bc T_cw` and uses the resulting body rotation and
centre. Right perturbations remain valid because
`T_bc T_cw Exp(xi) = T_bw Exp(xi)`. A non-identity rotation/lever-arm test now
proves both zero cost at the calibrated truth and convergence of a drifted
camera pose back to the true body pose. Pipeline validation also rejects
static, motion, and local VI stages configured with different extrinsics.

The audit also exposed unconstrained per-keyframe bias slots in local VI-BA.
The BA bias random-walk factor now carries separate gyro and accelerometer
information weights, and local VI-BA can attach those factors to every
in-window IMU edge. It remains explicit opt-in: EuRoC provides both random-walk
densities, but automatically using `1/(sigma^2 dt)` would be false precision
while the rotation/velocity/position IMU blocks still use manual scalar
weights rather than propagated covariance. Tight VI therefore remains
disabled in the promotion matrix until all inertial residual blocks share a
calibrated covariance model and repeated A/B evaluation passes.

The same diagnostic exposed an observability gap in the robust front end:
aggregate PCM rejection counts discarded the rejected relative pose, so an
offline evaluator could not distinguish correct false-positive rejection from
over-rejection of a true loop. The per-frame result now retains every PCM- or
covariance-rejected constraint (source, inliers, full SE(3), and reason), and
the EuRoC demo writes `loop_rejected_constraints.csv` with the same 0.5 m /
10 degree ground-truth classification used for admitted loops. Summary fields
separately report how many rejected constraints were GT-evaluable and how many
were actually correct. The rebuilt 1,000-frame MH_01 prefix classified all
three PCM rejections as true loops: their translation errors were 0.0440,
0.0235, and 0.0218 m and their rotation errors were 0.456, 0.502, and 0.581
degrees. All are far inside the 0.5 m / 10 degree correctness gate. The three
trajectory exports remained SHA-256-identical to the no-loop control because
no constraint was admitted. This localizes the present failure to PCM's
odometry-consistency model or threshold, rather than appearance retrieval or
PnP. The next loop experiment must inspect the PCM cycle residuals and compare
an uncertainty-normalized gate before relaxing admission.

That residual audit is now complete. The three correct closures have raw
single-loop SE(3) residuals 1.681, 1.677, and 1.530 against the drifted visual
odometry, so the legacy require-individual threshold of 1.0 rejects all of
them. In contrast, their three pairwise cycle residuals are 0.025, 0.216, and
0.197: all three measurements form a mutually consistent clique by a wide
margin. This matches Mangelson et al.'s formulation, which tests a
covariance-normalized pairwise cycle and selects the maximum clique; the
single-loop odometry pre-filter is an extra heuristic and is unsafe under
accumulated drift. The online incremental gate also ignored the configured PCM
noise model even though batch PCM honored it; that implementation mismatch is
corrected and regression-tested. A new explicit diagnostic flag disables only
the individual pre-filter while leaving the default unchanged.

This choice is also supported by the official
[Kimera-RPGO implementation](https://github.com/MIT-SPARK/Kimera-RPGO):
its odometry check and PCM consistency check have separate thresholds, and
either check can be disabled independently. The visloc-rs experiment therefore
uses the explicit pairwise-only flag rather than silently widening the mixed-
unit raw threshold.

A same-binary 1,000-frame MH_01 A/B then compared no-loop against this
pairwise-only candidate. The candidate admitted one appearance loop at frame
939 (408 PnP inliers; GT error 0.0440 m / 0.456 degrees), ran one SE(3) GNC PGO
solve, moved 4,870 anchored landmarks, and propagated one tracker correction.
Loop precision was 1/1. Final-keyframe rigid ATE improved from 0.6911 to
0.3343 m and delta-10 RPE improved from 0.5615 m / 12.18 degrees to
0.4719 m / 11.93 degrees. Tracking coverage rose from 0.947 to 0.948 with the
same 449-frame longest segment. The causal live rigid ATE changed from 0.5289
to 0.5315 m (+0.5%); this CSV cannot retroactively reflect a loop that closes
at frame 939, so it is retained as a bounded regression check rather than the
primary loop-SLAM accuracy metric. Runtime was 696.9 ms/frame for no-loop and
665.4 ms/frame for the candidate in this single order, but one repetition is
insufficient to claim a speedup. The promotion evaluator now uses final
optimized keyframe rigid ATE as its primary accuracy gate, permits at most 1%
live-ATE regression, caps runtime and peak-memory regression at 25%, and
requires at least three paired counterbalanced repetitions. This diagnostic is
therefore INCOMPLETE, not promoted.

The motion initializer now also supports optional post-solve gyro- and
accelerometer-bias magnitude gates, alongside the existing velocity limit.
All solves remain speculative until velocity and both bias bounds pass; a
rejection leaves the live `VisualMap` unchanged. These thresholds are
diagnostic controls, not promoted EuRoC defaults: their values still require
sensor-specification or calibration evidence and a same-binary prefix run.

## Literature backlog

The next survey pass deepens calibration/observability analysis and covers
line/plane/object landmarks and long-term map maintenance. Each addition must
state a testable repository hypothesis rather than only summarize the abstract.
