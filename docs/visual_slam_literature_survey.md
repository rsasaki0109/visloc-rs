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

## Correspondence, triangulation, and sparse optimization foundations

| Work | Central contribution | visloc-rs hypothesis |
| --- | --- | --- |
| [Lucas--Kanade (1981)](https://idl.uw.edu/living-papers-paper/lucas-kanade/index.pdf) | Iterative gradient-based image registration obtains subpixel displacement without an exhaustive correspondence search | Compare descriptor-only projection rematching with bidirectional patch refinement; gate the refined track on photometric residual, convergence, and forward/backward consistency |
| [Shi--Tomasi, Good Features to Track (CVPR 1994)](https://publications.ri.cmu.edu/good-features-to-track) | Selects patches from the smaller structure-tensor eigenvalue and monitors affine tracking failure, occlusion, and disocclusion | Replenish tracks by predicted trackability and spatial coverage rather than detector score alone, and retire tracks when their local appearance ceases to support one physical point |
| [Hartley--Sturm triangulation (CVIU 1997)](https://perception.inrialpes.fr/Publications/1997/HS97/HartleySturm-cviu97.pdf) | Finds the global minimum of a Gaussian image-coordinate correction instead of intersecting noisy rays or accepting a local minimum | Replace midpoint-style stereo depth confidence with reprojection-domain triangulation, cheirality, parallax, and propagated pixel covariance; never use raw 3-D distance as though depth noise were isotropic |
| [MLPnP (ISPRS 2016)](https://arxiv.org/abs/1607.08112) and [authors' C++/MATLAB implementations](https://www.ipf.kit.edu/english/code_1840.php) | Propagates image-observation covariance to bearing-vector tangent space, solves a maximum-likelihood PnP problem, and recovers internal rotation/translation accuracy | After robust PnP refinement, derive a 6-DoF loop information matrix from the inlier reprojection Jacobian and observation covariances; reject rank-deficient/ill-conditioned geometry instead of replacing uncertainty with inlier count |
| [Uncertainty-Aware Camera Pose Estimation from Points and Lines (CVPR 2021)](https://arxiv.org/abs/2107.03890) and [authors' implementation](https://alexandervakhitov.github.io/uncertain-pnp/) | Extends uncertainty-aware PnP and motion-only BA to model covariance in both the 2-D detections and the reconstructed 3-D points | Propagate rectified-stereo pixel/disparity covariance into each loop landmark's anisotropic 3-D covariance, then include both 2-D and 3-D uncertainty in pose refinement; MLPnP's 2-D-only model is insufficient for noisy stereo depth |
| [Bundle Adjustment -- A Modern Synthesis (2000)](https://hal.inria.fr/inria-00548290) | Defines joint robust structure/camera refinement, gauge freedom, quality control, and sparse second-order methods | Every BA fixture must verify a fixed gauge, robust reprojection weighting, Schur landmark elimination, rejected-step rollback, and a decrease in the exact cost reported to the caller |
| [g2o paper (ICRA 2011)](https://ais.informatik.uni-freiburg.de/publications/papers/kuemmerle11icra.pdf) and [official implementation](https://github.com/RainerKuemmerle/g2o) | Expresses pose SLAM and BA as extensible sparse nonlinear graph optimization and exploits each graph's block structure | Cross-check SE(3) residual direction, analytic/numeric Jacobians, gauge fixing, and LM/GN convergence on exported visloc-rs graphs before attributing a trajectory change to loop selection |
| [iSAM2 (IJRR 2012)](https://www.cs.cmu.edu/~kaess/pub/Kaess12ijrr.html) and [GTSAM ISAM2](https://gtsam.org/doxygen/a04947.html) | Bayes-tree updates provide incremental variable reordering and selective relinearization without periodic full batch solves | Benchmark an incremental factor-history back end against batch PGO/BA, including loop-factor removal and delayed relinearization after GNC/PCM changes the accepted set; equal final cost is required before runtime can decide |
| [Ceres covariance estimation](https://ceres-solver.readthedocs.io/latest/nnls_covariance.html) (official implementation documentation) | Computes selected covariance blocks from the solved Jacobian through sparse QR, or a pseudoinverse through dense SVD when rank deficiency must be represented | Cross-check each visloc-rs PnP covariance against an independently assembled finite-difference Jacobian; expose rank/condition diagnostics and never silently invert a near-singular normal matrix |

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
prevents allocation-starved runs, while the external-CPU gate rejects sustained
CPU time consumed outside the benchmark process. This sampling still does not
replace reserving the host: any known overlap with an unrelated compute job
invalidates the affected runtime samples and requires a clean rerun. The runner records the
Git revision, executable and model hashes, exact argument vector, timestamps,
elapsed time, and exit status for every run:

```powershell
$env:ORT_DYLIB_PATH = 'E:/tools/colmap/bin/onnxruntime.dll'
$env:Path = 'E:\tools\colmap\bin;E:\tools\venv-cu\Lib\site-packages\torch\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;' + $env:Path
& "$HOME/.cargo/bin/cargo.exe" build --release `
  --features 'image-io onnx-cuda' `
  --example euroc_online_slam_vi_image_demo
./scripts/run_visual_slam_euroc_ab.ps1 `
  -DatasetRoot E:/datasets/euroc_mav/machine_hall `
  -OutRoot E:/visloc_archive/visual_slam_ab_20260713 `
  -OnnxBackend cuda `
  -Repetitions 5
```

The runner defaults to strict `-OnnxBackend cuda`: a missing provider or CUDA
dependency aborts model loading instead of silently falling back to CPU. Use
`cpu` only for an explicitly labelled CPU experiment. It always passes
`--max-frames`; its default `-MaxFrames 0` means every
available frame. This is intentional because omitting the demo option would use
the demo's 400-frame smoke-test default rather than a full sequence.
It fails before launching when the dynamic ONNX runtime DLL is unavailable and
records the Git revision, dirty status and diff fingerprint, executable/model/
runtime DLL hashes, exact arguments, Rust toolchain, OS, CPU, logical processor
count, installed memory, and 250-ms-sampled peak process working set. It also
re-hashes the executable, SuperPoint model, and ONNX Runtime DLL immediately
before every run, stores those hashes in the run manifest, and refuses to mix a
changed artifact into an existing matrix. It also
requires at least 4 GiB of available physical memory and 4 GiB of Windows
commit headroom before each run, samples both every five seconds, and records
the preflight and minimum values in each run manifest. It also measures external
process CPU for two seconds before launch and every resource interval during the
run, excluding the runner and benchmark child; the default maximum is 0.5 CPU
cores. Preflight rejects one contaminated sample, while an active run requires
three consecutive over-limit samples before termination so a scheduler spike
is recorded but is not mislabeled as sustained contention. The runner's
ancestor chain (the waiting shell/orchestrator) is also
excluded and recorded by PID so measurement overhead is not mistaken for a
competing workload. If either memory reserve is crossed or external CPU exceeds the limit,
the runner stops only its benchmark child and records an environmental
validation error separately from the child's real exit code.
Such a run is not an accuracy or runtime sample. The thresholds and sampling
period can be changed explicitly with `-MinAvailableMemoryGiB`,
`-MinCommitHeadroomGiB`, `-MaxExternalCpuCores`,
`-ExternalCpuViolationSamples`, and
`-ResourceSampleIntervalSeconds` when the host has a documented resource
policy.

A dirty-tree run is therefore identifiable, while
the executable hash remains the authoritative identity of what actually ran.
For a long interrupted matrix, repeat the same command with `-Resume`.
Only runs whose per-run manifest has exit code zero and whose `summary.txt`
exists are skipped; an incomplete or failed directory is never overwritten
implicitly. Resume is also refused when executable, SuperPoint model, or ORT
DLL hashes, dataset root, sequence list, or frame cap differ from the existing
experiment. The requested repetition count may only increase: this safely
extends a completed prefix while a decrease is refused so recorded runs cannot
be orphaned. The schema-v9 manifest also fingerprints the
common and per-variant argument protocol plus the memory and external-CPU
resource gates, preventing a changed treatment or environmental validity rule
from being resumed into an older matrix. Each successful run is rejected by the
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
Its loop edge uses the fixed scalar selected by the documented MH_01 prefix
sweep (`-CandidateLoopEdgeWeight`) rather than treating the 408 PnP inliers
observed on MH_01 as 408 times the inverse pose covariance. Inlier count remains
evidence for verifier acceptance, not a dimensionally valid uncertainty
estimate; calibrated anisotropic PnP information is the eventual replacement
for every scalar choice.
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
when loops are accepted, the worst candidate rigid ATE must beat the best
no-loop rigid ATE across the recorded repetitions; every accepted loop must
have a ground-truth verdict and all must be correct. A zero-loop sequence can
instead be classified `SAFE_NO_OP`, but only when each paired repetition has
identical SHA-256 hashes for `slam_trajectory.csv`, `slam_errors.csv`,
`keyframe_trajectory.csv`, `frame_groundtruth.csv`, and
`final_keyframe_errors.csv`. Rounded metric equality is not sufficient.
Candidate tracking coverage may fall by at most 0.005, the longest continuous
run may fall by at most 1%, and the worst candidate delta-1/delta-10 translation
and rotation RPE may exceed the best baseline by at most 1%. Runtime and sampled
peak working set must be present for every run, and the candidate may exceed its
same-repetition control by at most 25% in either measure. Comparing the slowest
candidate from one repetition against the fastest control from another would
undo the matrix's counterbalancing and confound treatment overhead with host
state. `INCOMPLETE` means evidence is missing; `REJECT` means complete evidence
violates at least one safety or improvement gate.
The report also emits one matrix decision. It is `PROMOTE` only when at least
one declared sequence improves and every other declared sequence is either
`PROMOTE` or a hash-proven `SAFE_NO_OP`; an all-no-op matrix cannot claim an
improvement. Missing declared sequences or repetitions keep the matrix
`INCOMPLETE`.

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
7. Export deterministic graph fixtures, cross-check gauge/Jacobian/final cost
   against g2o or GTSAM, then compare batch solves with an incremental Bayes-tree
   factor history that can remove rejected loop factors.
8. Build a genuinely tightly coupled stereo-inertial window with consistent
   relinearization and delayed prior reconstruction.
9. Only then benchmark learned recurrent patch/pointmap front ends and dense map
   representations.

Step 3 now has a stricter descriptor-level first stage. Lucas--Kanade assumes
an approximate registration and iteratively reduces a local photometric
residual; Shi--Tomasi additionally monitors appearance dissimilarity because a
numerically trackable patch can still drift or become occluded. The current
`Frame` API carries keypoints/descriptors but no source image, so it would be
incorrect to label descriptor filtering as LK. Instead, matching now mirrors
the complementary check visible in ORB-SLAM3's official
`ORBmatcher::SearchByProjection`: retain the existing landmark-to-window
best/second ratio, then also require the selected query keypoint to prefer its
best competing projected landmark by a configurable reverse ratio. Exact or
near descriptor collisions from overlapping windows are rejected before PnP;
the appearance-global fallback remains available if too few survive. True
pyramidal photometric refinement remains an explicit later experiment that
requires retaining image pyramids alongside `Frame`.

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

Two completed full-sequence MH_01 repetitions were deterministic and exposed a
local-accuracy regression hidden by the ATE improvement. Pairwise-only PCM
improved final-keyframe rigid ATE from 3.9434 m to 3.2506 m and live rigid ATE
from 4.0558 m to 3.1126 m, but delta-1 translation/rotation RPE worsened from
0.2127 m / 4.27 degrees to 0.2441 m / 4.54 degrees and delta-10 RPE worsened
from 1.2667 m / 18.22 degrees to 1.4483 m / 19.06 degrees. The promotion gate
now rejects a greater-than-1% regression in any of these four RPE metrics.
Recomputing every RPE pair from the raw EuRoC ground truth and cam0 `T_BS`
reproduced the summary values. Before frame 939, candidate delta-10 translation
RPE was better (0.459 m versus 0.568 m); after closure it was worse in every
segment, reaching 1.404 m versus 0.990 m after frame 2500. The degradation is
therefore persistent post-correction tracking/map behaviour, not one RPE pair
crossing the closure instant.

The back end revealed a dimensionally invalid confidence mismatch: every
sequential edge has scalar weight `1.0`, while
`PoseGraph::add_loop_closure_constraint` used the PnP inlier count directly, so
the accepted loop had weight `408.0`. Correspondence count is not inverse pose
covariance. The legacy API remains unchanged, but online refinement can now use
an explicit fixed positive loop weight, recorded in the run summary and checked
by the matrix runner. A 1.0/0.1/0.01 MH_01 prefix sweep must establish whether a
softer loop preserves the ATE gain without the RPE regression; none of these
weights is promoted in advance of that measurement. The long-term replacement
is a calibrated anisotropic PnP information matrix, not another universal
scalar. MLPnP supplies the observation-side model: propagate each feature's
pixel covariance to its bearing tangent plane. At the refined pose, assemble
the inlier residual Jacobian `J` and block observation precision `W`, then use
`J^T W J` as the local 6-DoF pose information only when its numerical rank and
condition number pass an explicit gate. The matrix must be transformed into
the pose graph edge's exact tangent/residual convention and checked against a
finite-difference Jacobian. Ceres' covariance implementation provides the
independent reference behaviour: QR for a full-rank Jacobian, or an SVD
pseudoinverse when rank deficiency is deliberately represented. A synthetic
geometry suite must cover well-spread points, near-planar points, narrow image
support, depth imbalance, and pixel-noise rescaling before this information
matrix is allowed into EuRoC evaluation.

The repository already has the correct narrow integration seam:
`crates/vision/src/pnp/mod.rs`'s `GaussNewtonPoseRefiner` assembles the final
`2N x 6` finite-difference reprojection Jacobian, while
`crates/vision/src/ransac/mod.rs`'s `RansacReport` retains the final inlier
indices and per-inlier pixel errors. The next implementation should return a
typed pose-information/covariance diagnostic from that final inlier-only
refinement and carry it through `PnPLoopClosureVerifier` into
`PoseGraphEdge::information`. It must not estimate information from all
pre-RANSAC matches, from the RANSAC threshold, or from `inliers.len()`.

There is an important non-reference implementation already in
`pipelines/slam/src/vo_loop_closure.rs`: its optional offline
`loop_edge_information` also forms a reprojection `J^T J`, but then
trace-normalises the matrix to `6 * inlier_count` and adds a ridge to force
positive definiteness. That preserves the same uncalibrated 408-versus-1 loop
magnitude exposed by MH_01, and the ridge can hide deficient geometry instead
of rejecting it. Its directional shape is useful test material, but neither
its magnitude nor its degeneracy policy may be copied into the online path.
The controlled replacement must separate (a) query-image 2-D observation-noise
calibration, (b) older-keyframe stereo landmark covariance propagated from
left/right pixel and disparity noise, (c) geometric rank/condition validation,
and (d) any experimental global loop-strength multiplier, and report all four
independently. The CVPR 2021 uncertainty-aware PnP refinement is the closer
reference for this stereo case than 2-D-only MLPnP: treating an uncertain-depth
landmark as a fixed 3-D point makes the resulting pose information
overconfident precisely along the optical-axis direction that matters here.

The fixed-weight diagnostic is deliberately long enough to observe the
post-loop failure, not merely the closure instant. Run 2,800 MH_01 frames (the
accepted loop occurs near frame 939 and the largest measured late RPE gap is
after frame 2,500), one paired baseline/candidate run per weight:

```powershell
$weights = @(1.0, 0.1, 0.01)
foreach ($weight in $weights) {
  $tag = $weight.ToString('R', [Globalization.CultureInfo]::InvariantCulture).Replace('.', 'p')
  $root = "E:/visloc_archive/mh01_loop_weight_sweep_20260713/weight_$tag"
  ./scripts/run_visual_slam_euroc_ab.ps1 `
    -DatasetRoot E:/datasets/euroc_mav/machine_hall `
    -OutRoot $root `
    -Sequences MH_01_easy `
    -MaxFrames 2800 `
    -Repetitions 1 `
    -CandidateLoopEdgeWeight $weight
  python scripts/summarize_visual_slam_evaluation.py `
    --matrix-root $root `
    --output "$root/summary.md" `
    --json "$root/summary.json"
}
```

This is a parameter screen only: its one repetition must remain `INCOMPLETE`
under the promotion policy. The selected weight advances only if it has a
correct accepted loop, improves rigid ATE, and does not worsen any delta-1 or
delta-10 translation/rotation RPE metric by more than 1% on this prefix. It
then receives the full three-repetition MH_01/MH_03/MH_05 promotion matrix;
prefix evidence alone can never adopt it.

The completed 2,800-frame screen rejected all three fixed weights. The
same-binary no-loop baseline was deterministic across the three paired runs:
tracking coverage `0.856`, longest segment `449`, live/final-keyframe rigid ATE
`2.6562/2.7093 m`, delta-1 RPE `0.2334 m / 4.564 degrees`, and delta-10 RPE
`1.4361 m / 19.744 degrees`. Each candidate's accepted loops were
ground-truth-correct (precision `1.0`), but no scalar preserved every local
accuracy metric:

| fixed loop weight | accepted loops | live/final ATE (m) | delta-1 RPE (m / deg) | delta-10 RPE (m / deg) | decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| `1.0` | 3/3 | 2.7828 / 2.7958 | 0.2569 / 5.336 | 1.5370 / 23.174 | reject: ATE and all RPE regress |
| `0.1` | 2/2 | 2.4745 / 2.4775 | 0.2469 / 4.740 | 1.4692 / 21.410 | reject: ATE improves, all RPE regress by more than 1% |
| `0.01` | 3/3 | 3.0118 / 2.9194 | 0.2475 / 4.580 | 1.4418 / 18.660 | reject: ATE and delta-1 translation regress |

These are deliberately diagnostic, single-repetition results, so the runner's
promotion verdict remains `INCOMPLETE` independently of the metric failures.
Their purpose is nevertheless complete: a universal isotropic loop strength
cannot satisfy the MH_01 global- and local-accuracy gates. No fixed weight
advances to the full sequence matrix. The next candidate must use the
inlier-only reprojection Jacobian and observation/landmark covariance to retain
directional observability, reject rank-deficient or ill-conditioned geometry,
and expose any global strength cap separately from the measured matrix.

The first controlled replacement is now implemented behind
`--pose-graph-refinement-loop-pose-information`. It consumes only the final
appearance-PnP inlier pairs retained by `LoopClosureCandidate`, never all
descriptor matches. For each older-map landmark it forms a 3-D covariance by
inverting the multi-view reprojection Hessian over that landmark's existing map
observations. A one-view point remains rank deficient and is discarded rather
than made invertible with a ridge. The covariance is rotated into the older
camera frame and propagated through the loop projection; with query pixel
noise `sigma_px`, each inlier contributes
`J_pose^T (sigma_px^2 I + J_point Sigma_point J_point^T)^-1 J_pose`. The sum is
accepted only when enough landmarks survive and the 6-DoF matrix is positive
full-rank with condition number below the configured ceiling. Synthetic tests
cover a well-spread, multi-depth scene and the single-view ray degeneracy.

The estimator now returns a typed failure instead of collapsing every gate to
`None`. Full-run summaries separately count invalid configuration, missing
keyframe/pose state, insufficient usable correspondences, rank deficiency,
ill-conditioning, and an unsupported solver. Loop-edge rejection and
sequential-edge identity fallback therefore remain safe while EuRoC logs show
whether more correspondence support or better-conditioned geometry is needed.

Absolute pixel-domain curvature is not dimensionally comparable to the current
unit-information odometry chain. The implementation therefore exposes a
separate maximum-eigenvalue cap (default `1.0`) instead of trace-normalising to
the inlier count. It records the raw condition number, used correspondence
count, and applied spectral scale in the EuRoC summary so the cap cannot be
mistaken for measured covariance. The vision-layer stereo bootstrap now
propagates independent `(u_l, v_l, u_r, v_r)` pixel noise through its general
6-DoF DLT triangulator with a central-difference Jacobian and retains the
resulting anisotropic 3-D covariance on each survivor. A rectified fixture
checks the analytic `sigma_Z = sqrt(2) Z^2 sigma_px / (f b)` law, including its
quadratic growth with range. This implements the standard post-triangulation
error propagation required in Section 3.6 of Vakhitov et al.; their Eq. 21 then
defines how the 3-D covariance must enter the pose residual covariance.
`VisualMap` now carries these world-frame matrices in a validated landmark
covariance sidecar. Seed and stereo re-bootstrap insertion rotate the
left-camera covariance into the world frame, submap extraction preserves it,
and loop pose-information prefers it over the left-view Hessian fallback. A
single-left-view fixture that is correctly rank deficient without metadata
becomes usable with a calibrated stereo covariance. BA still needs to propagate
or relinearize the covariance when it moves a landmark, so full
uncertainty-aware stereo-PnP remains an experimental path rather than a promoted
default.

The first 1,000-frame MH_01 same-binary diagnostic exercised this path in the
real online pipeline. All three accepted appearance loops received a 6×6
matrix (zero pose-information rejections), using 637 final PnP inliers in
total; all three were ground-truth-correct. The worst raw matrix condition
number was `564.13`, and the smallest spectral scale needed to impose the
unit-eigenvalue cap was `2.41e-8`, confirming that raw pixel curvature and the
legacy unit odometry edges differ by many orders of magnitude. Relative to the
deterministic no-loop baseline, tracking changed `0.947 → 0.944` with the same
449-frame longest segment, live rigid ATE improved `0.5289 → 0.5015 m`, final
keyframe ATE improved `0.6911 → 0.4287 m`, delta-1 translation RPE was
effectively unchanged (`0.184407 → 0.184445 m`), and delta-10 translation RPE
improved `0.5615 → 0.4812 m`. Rotation RPE nevertheless regressed from
`5.958 → 6.163 degrees` at delta 1 and `12.177 → 12.600 degrees` at delta 10.
The promotion gate therefore rejects this loop-only information variant even
though its ATE and translation RPE improve.

The failure isolates the next correction: applying a covariance-shaped matrix
only to loops while sequential edges remain isotropic identity still compares
different information conventions. The online graph now applies the same
inlier/covariance estimator and the same explicit spectral cap to sequential
PnP edges, falling back to identity only when the chain has too little
multi-view landmark support. While implementing that path, a map invariant bug
was exposed and fixed: tracking-produced keyframes already contained their
inlier observations, but `StagedMapUpdate::apply_to` did not mirror those
relations into `Landmark::observations`. The two VisualMap observation indices
are now synchronized without duplicates, which is required for covariance and
covisibility consumers alike. This matched-information graph needs a fresh
same-binary diagnostic; the loop-only result above must not be used to claim
promotion.

That fresh 1,000-frame MH_01 diagnostic is now complete, after the observation
index repair and with the same executable for both variants. The no-loop
baseline achieved tracking `0.934`, longest segment `307`, live/final-keyframe
rigid ATE `0.2513/0.3457 m`, delta-1 RPE `0.1756 m / 6.074 degrees`, and
delta-10 RPE `0.4649 m / 18.589 degrees`. The matched-information candidate
put covariance-shaped matrices on 58 sequential edges (two early identity
fallbacks) and its one accepted, ground-truth-correct loop (432 PnP inliers,
raw condition number `222.52`, spectral cap scale `8.79e-9`). Rotation RPE
improved to `5.916/16.667 degrees`, but every safety-critical translation or
coverage headline regressed: tracking `0.926`, live/final ATE
`0.2771/0.3750 m`, delta-1 translation RPE `0.1920 m`, and delta-10
translation RPE `0.4809 m`. It is rejected and does not advance to 2,800
frames. Matching the matrix *form* is insufficient when the relative-edge
covariance still treats the older pose and its correlated map points as
fixed; exact stereo seed uncertainty and pose/landmark correlation remain
unmodelled.

The failure audit also exposed a separate measurement-integrity defect in
stereo replenishment. Candidate association selected a real anchor-keypoint
index within the reprojection gate, but stored the stereo point's synthetic
anchor reprojection as `Observation.xy`. Thus index and pixel described
different measurements, and downstream two-view triangulation received a
self-consistent prediction in place of image evidence. Replenishment now uses
the selected keypoint's actual pixel coordinate; the stereo reprojection is
only the search/gating prediction. A regression test deliberately offsets the
real detection from the prediction and proves that the measured coordinate is
retained. Since this changes map geometry for both variants, all subsequent
EuRoC comparisons require a new same-binary baseline/candidate pair.

The first post-fix 1,000-frame pair improved the no-loop tracking coverage to
`0.959` and the longest segment to `527`, but it cannot evaluate loop quality:
the candidate was bit-identical in every accuracy metric because no loop was
confirmed. Appearance retrieval ranked 75 regions, rejected 60 connected
regions, PnP-verified two candidates, and left three region tracks waiting for
the configured three-keyframe confirmation; zero reached confirmation before
the prefix ended. This is not a pass or a loop regression. The next diagnostic
must retain the safety confirmation and extend MH_01 to 2,800 frames. It also
records sequential-edge correspondence, condition-number, and spectral-cap
aggregates so a later accepted loop can be interpreted against the odometry
information distribution rather than only the loop matrix.

The 2,800-frame extension still admitted zero loops and was therefore exactly
equal to its paired baseline (`0.855` tracking, longest `527`, live/final ATE
`3.4996/2.9476 m`, delta-1 RPE `0.2620 m / 4.802 degrees`, delta-10 RPE
`1.5335 m / 19.728 degrees`). This was not a lack of loop evidence. At frame
960, regions rooted at keyframes 58 and 75 had appearance similarity above
`0.90`, primary PnP support of `300/360` inliers, and projection-rematched PnP
support of `334/381` inliers. Both started temporal confirmation, but later
keyframes did not independently retrieve and globally rematch the region, so
the pending state expired. Across the run, 322 sequential edges received pose
information (13 identity fallbacks, 33,042 used correspondences); the maximum
raw condition number was `4919.22` and minimum spectral scale `5.35e-9`.

That audit exposed a mismatch between the implementation and the ORB-SLAM3
temporal-consistency procedure already cited above. The code comment said that
one strong current-to-region pose is carried into later keyframes, but the
confirmation implementation required appearance retrieval and global PnP to
succeed from scratch on every later keyframe. A pending region is now actively
carried forward: compose its recovered root-to-last pose with intervening
odometry, projection-match the same covisible region into the new keyframe,
and run refined PnP before incrementing confirmation. The original
three-keyframe requirement and pose-disagreement gates remain intact. The
summary separately reports pending-projection attempts and successful PnP
verifications. A synthetic 64-point test proves that confirmation evidence can
be recovered by projection without another global retrieval; the adaptive
`5/10/15 px` search-radius schedule is unit-tested, but EuRoC must still
show that this produces correct loops and safe ATE/RPE before promotion.

The motion initializer now also supports optional post-solve gyro- and
accelerometer-bias magnitude gates, alongside the existing velocity limit.
All solves remain speculative until velocity and both bias bounds pass; a
rejection leaves the live `VisualMap` unchanged. These thresholds are
diagnostic controls, not promoted EuRoC defaults: their values still require
sensor-specification or calibration evidence and a same-binary prefix run.

### Projection query-to-landmark ambiguity gate: MH_01 rejection result (2026-07-14)

The query-centric nearest/second-nearest descriptor ratio gate is available as
an opt-in projection-tracking experiment, but is deliberately disabled by
default. A deterministic 300-frame MH_01 comparison used the same SuperPoint
ONNX input, projection window, covisibility local map, and stereo-replenishment
settings. The legacy `None` configuration achieved tracking coverage `0.933`,
rigid ATE `0.2340 m`, final-keyframe rigid ATE `0.2255 m`, and `601.10
ms/frame`. A ratio of `0.9` improved rigid ATE to `0.1911 m` and
final-keyframe rigid ATE to `0.2024 m`, but reduced tracking coverage to
`0.837`, increased runtime to `615.01 ms/frame`, worsened delta-1 translation
RPE (`0.1771 -> 0.1844 m`) and delta-10 rotation RPE (`8.27 -> 9.36 deg`).
Because continuity is a primary adoption gate and the RPE result is mixed,
`0.9` is rejected as a production default pending a wider sweep or a
confidence-aware formulation. Reproduce with
`--projection-query-landmark-distance-ratio none` and `0.9`, respectively.

### Strict-CUDA three-sequence prefix baseline (2026-07-14)

The CUDA-enabled release binary was then run for 300 frames on every required
Machine Hall sequence with the rejected ambiguity gate disabled. Each run used
`--superpoint-onnx-backend cuda`, so provider registration could not fall back
to CPU. These are prefix diagnostics, not full-sequence promotion evidence:

| sequence | tracking | rigid ATE (m) | final-KF rigid ATE (m) | d1 RPE (m / deg) | d10 RPE (m / deg) | ms/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MH_01 | 0.917 | 0.2603 | 0.2483 | 0.1629 / 5.30 | 0.2786 / 11.49 | 129.63 |
| MH_03 | 0.903 | 0.1718 | 0.2626 | 0.2844 / 6.20 | 0.4194 / 7.06 | 178.76 |
| MH_05 | 0.920 | 0.3019 | 0.2916 | 0.2013 / 22.96 | 0.7346 / 27.66 | 136.82 |

The corresponding MH_01 CPU-provider prefix took `601.10 ms/frame`; strict
CUDA therefore reduced end-to-end time by 4.64x on that comparison. Its
tracking and trajectory were not bit-identical to CUDA, so provider changes are
treated as experiment changes: candidate and no-loop baseline must use the
same binary, ONNX Runtime DLL, CUDA provider, and model hashes.

### Loop welding and post-PGO fusion audit (2026-07-14)

ORB-SLAM2 describes a fourth full-BA thread launched after loop closure, while
ORB-SLAM3 formulates map estimation as BA and likewise runs full BA after a
loop. The [official ORB-SLAM2 loop-closing source](https://github.com/raulmur/ORB_SLAM2/blob/master/src/LoopClosing.cc)
makes the ordering concrete: `SearchAndFuse`, essential-graph optimization,
then global BA. This is stronger than visloc-rs's historical pose-edge-only
write-back, because the accepted image evidence becomes map observations that
BA can optimize.

An initial implementation persisted all accepted PnP pairs before PGO and ran
a synchronous covisibility welding BA afterward. On the deterministic strict-
CUDA MH_01 1,000-frame pair, the no-loop baseline was tracking `0.966`, live /
final-keyframe rigid ATE `0.2737 / 0.3040 m`, d1 RPE `0.1743 m / 6.00 deg`, and
d10 RPE `0.3576 m / 16.85 deg`. Welding inserted 204 observations (three
reassignments) and optimized 21 keyframes / 1,942 landmarks. Live ATE improved
to `0.2422 m`, but final-keyframe ATE regressed to `0.3334 m`, d1 to
`0.2115 m / 7.68 deg`, and d10 translation to `0.4718 m`; it is rejected.
Fusion without the immediate welding solve was also rejected: live ATE
`0.2252 m`, final-keyframe ATE `0.3281 m`, and d10 translation RPE `0.5290 m`.
The map mutation itself therefore disturbed subsequent local tracking, not
only the explicit BA solve.

The repository now defers fusion until a PGO solve succeeds. It also refuses
to fuse a loop edge that final GNC classifies below its `0.5` inlier threshold,
and reprojects every older landmark with the corrected query pose, retaining
only pairs within the appearance PnP verifier's pixel-error threshold. The
first fixed-weight `0.001` screen admitted three ground-truth-correct loops but
GNC retained none for fusion: all 928 PnP pairs were robust-rejected, so zero
observations were mutated and the result returned exactly to the prior PGO-only
metrics (`0.2379 / 0.2877 m` live/final ATE).

That diagnostic exposed a scalar/information inconsistency in GNC. Its inner
solve multiplied isotropic edges by `edge.weight`, but its robust classifier
used unweighted `||r||^2`; the equivalent full-matrix path correctly used
`r^T Omega r`. Classification now uses `edge.weight * ||r||^2`, with a
regression proving an isotropic scalar edge and `weight * I` receive identical
GNC weights. On the follow-up screen GNC retained all three correct loops, but
all 800 resulting pairs failed the corrected-pose 4 px reprojection gate, so
fusion again safely mutated zero observations. PGO-only live/final ATE was
`0.2596 / 0.3026 m`, d1 RPE `0.1898 m / 6.00 deg`, and d10 RPE
`0.4199 m / 16.03 deg`; translation RPE still regressed against no-loop. This
is a safe no-op, not a promotion. The remaining backend defect is that the
weak loop PGO solution does not make even its accepted PnP map points mutually
reprojection-consistent. Fusion must remain opt-in until pose correction and
local-RPE gates agree.

The next implementation follows the official ordering more closely. It first
applies the accepted loop transform to the current covisibility region, moves
that region's landmarks with the same rigid correction, transactionally fuses
the geometrically consistent observations, runs a second essential-graph
optimization, and finally runs local BA. Corrections above `0.5 m` or `0.2 rad`
are rejected before map mutation, and every stage can restore the original map
and graph. This made one fixed-weight `0.1` loop weld internally consistent,
but exposed a generic BA defect: reprojection cost skipped observations behind
the camera, allowing LM to improve its objective by moving 6.61% of selected
landmark observations behind a camera.

LM acceptance now separately counts non-projectable visual observations and
rejects any candidate step that increases that count. A synthetic regression
reproduces the former zero-cost-by-negative-depth failure. On the identical
strict-CUDA MH_01 960-frame run, the behind-camera ratio fell from `6.61%` to
`0%`; the transaction then committed 307 inserted and 61 reassigned
observations while updating 21 keyframes and 2,056 landmarks. At 1,000 frames,
the welded result improved final-keyframe rigid ATE from the no-loop `0.3040 m`
to `0.2329 m`, and d10 RPE from `0.3576 m / 16.85 deg` to
`0.3369 m / 14.44 deg`. However, d1 rotation RPE changed from `6.00 deg` to
`6.11 deg` (about 1.9% worse), outside the current 1% non-regression margin.
The cheirality fix is retained as a general optimizer invariant, while the
20 px post-BA kernel remains rejected.

Reducing the post-welding Huber transition to the same 5 px scale as the map's
outlier gate removed that last MH_01 regression. On the strict-CUDA 1,000-frame
screen, tracking was `0.965`; final-keyframe rigid ATE was `0.2543 m`; d1 RPE
was `0.1734 m / 5.99 deg`; and d10 RPE was `0.3545 m / 14.20 deg`. These all
meet the 1% non-regression margin against no-loop (`0.966`, `0.3040 m`,
`0.1743 m / 6.00 deg`, and `0.3576 m / 16.85 deg`). The weld again committed
307 inserted and 61 reassigned observations, optimized 21 keyframes / 2,056
landmarks, and kept the behind-camera ratio at zero.

The same candidate and an identical-binary no-loop control were then run for
1,000 frames on MH_03 and MH_05. Neither prefix produced a verified loop, so
all three trajectory/error CSV files were SHA-256-identical between candidate
and control on both sequences. This is the required side-effect-free behavior,
although it also shows that loop closure cannot repair their earlier tracking
failure in this prefix:

| sequence | candidate/control tracking | verified loops | final-KF ATE (m) | d1 RPE (m / deg) | d10 RPE (m / deg) | candidate / control ms/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MH_03 | 0.872 | 0 | 2.7717 | 0.4272 / 6.29 | 3.2076 / 31.82 | 145.87 / 114.73 |
| MH_05 | 0.889 | 0 | 5.3535 | 0.4153 / 12.46 | 3.3539 / 28.48 | 132.58 / 113.27 |

Thus the 5 px configuration is the first screened loop-welding candidate that
strictly improves MH_01 without changing MH_03/MH_05 trajectories when no loop
is verified. The 17-27% candidate-discovery runtime overhead and the difficult-
sequence tracking cliffs remain separate optimization targets; this evidence
is a three-sequence 1,000-frame promotion screen, not a complete-sequence
benchmark.

### MH_03 tracking-cliff motion-model audit (2026-07-15)

The [official ORB-SLAM2 tracker](https://github.com/raulmur/ORB_SLAM2/blob/master/src/Tracking.cc)
uses a constant-velocity prediction only while the previous tracking state is
healthy and otherwise falls back to reference-keyframe tracking. ORB-SLAM3's
[visual-inertial formulation](https://arxiv.org/abs/2007.11898) and
[VINS-Mono](https://arxiv.org/abs/1708.03852) jointly estimate pose, velocity,
gravity, and IMU biases; raw strapdown integration with stale or zero biases is
not equivalent to either system. This distinction predicts that an uncorrected
IMU prior can look excellent briefly and then diverge.

Strict-CUDA MH_03 experiments confirmed that prediction. IMU propagation with
the calibrated cam0/body transform tracked all of the first 400 frames with
`0.0114 m` final-keyframe rigid ATE, but at 1,000 frames coverage fell to
`0.559` and final-keyframe ATE exploded to `41.95 m`. Adaptive IMU/pose fallback
reached coverage `0.849` and ATE `4.2047 m`; a long-latched variant reached
`0.906` and `3.0026 m`, but both failed the no-loop baseline's `0.872` coverage
and `2.7717 m` ATE joint gate. Constant velocity was already unusable at 400
frames (`0.287`, `14.49 m`). A larger pose-jump gate (`0.2 -> 0.3 m`) raised
coverage to `0.911` but worsened final ATE to `3.1301 m`. Strict stereo removed
the fixed-depth fallback but reduced coverage to `0.767` and yielded `3.1643 m`
ATE. None is promoted.

The failure CSVs show that MH_03/MH_05 are not primarily starved of descriptor
matches: many rejected poses still have 90--160 PnP inliers. The common failure
is disagreement with the pose prior while the metric map scale is already
drifting. The tracker now refuses to feed a constant-pose fallback back into
PnP as a warm start; adaptive tracking enables warm start only while its IMU
mode is healthy. This is a safety invariant, not a claimed EuRoC gain: the
projection-guided path handled the screened frames before the global fallback,
so the 1,000-frame trajectory did not change.

### Stereo covariance continuity and rejected PnP ranking (2026-07-15)

The open-access primary study
[Principled Uncertainty Propagation for Stereo Visual Odometry](https://link.springer.com/article/10.1007/s10846-026-02358-0)
propagates per-feature pixel covariance through stereo triangulation and uses
measurement covariance in maximum-likelihood pose refinement and graph
factors. visloc-rs already propagated seed stereo covariance, but normal stereo
replenishment discarded it when a `LandmarkCandidate` was staged. Replenished
candidates now rotate the same-instant cam0/cam1 covariance into the world
frame, carry it through temporal triangulation and the transactional map
update, and validate finiteness, symmetry, positive semidefiniteness, entity
existence, and uniqueness. A regression proves the sidecar survives candidate
creation and map application. This makes fresh metric points available to the
existing covariance-aware loop-edge estimator instead of leaving only the 412
seed points annotated.

An inverse-projected-covariance PROSAC experiment was also implemented and
then removed after controlled rejection. With covariance only on the 412 seed
points, MH_03 translation RPE improved (`d1 0.4272 -> 0.3902 m`, `d10 3.2076 ->
2.9716 m`), but coverage regressed `0.872 -> 0.843`. After covariance continuity
raised the final annotated map from 412 to 7,876 points, coverage fell further
to `0.752`, final-keyframe ATE rose `2.7717 -> 3.1578 m`, and translation RPE
also regressed. Stereo depth precision is not descriptor-match correctness;
using it only to order RANSAC hypotheses misuses the uncertainty model. The
paper's covariance propagation and map sidecar remain, while PnP sampling is
restored to the baseline policy. A future uncertainty-aware PnP must use the
full residual covariance in consensus scoring and refinement, with an explicit
outlier model, rather than treating inverse depth variance as match confidence.

A second experiment tested whether the calibrated same-instant stereo point
should directly replace the temporally triangulated replenishment mean, as
suggested by ORB-SLAM2's [stereo/RGB-D formulation](https://arxiv.org/abs/1610.06475)
and official [`Frame::UnprojectStereo`](https://github.com/raulmur/ORB_SLAM2/blob/master/src/Frame.cc).
The naive mean replacement was also removed: MH_03 coverage fell `0.872 ->
0.808`, final-keyframe ATE rose `2.7717 -> 3.0321 m`, and the final map shrank
from 9,306 to 6,702 points. The stereo point, lifted through the current
estimated pose, often disagreed with the older anchor measurement under its
already-drifted pose and failed the 2 px triangulation validation. The correct
next design is an explicit right-image/depth observation factor jointly
optimized with both poses and the landmark; replacing the mean while retaining
a monocular two-view factor is not equivalent and is not kept.

After removing both rejected consumers, a strict-CUDA 300-frame MH_03
non-regression run reproduced the 2026-07-14 baseline exactly: tracking
`0.903`, rigid/final-keyframe ATE `0.1718/0.2626 m`, d1 RPE `0.2844 m / 6.20
deg`, and d10 RPE `0.4194 m / 7.06 deg`. All five trajectory, error, ground
truth, keyframe-trajectory, and final-keyframe-error CSV SHA-256 hashes were
identical. The only intended difference is map metadata: 1,563 of 2,349 final
landmarks now carry validated stereo covariance instead of only the seed
subset. Thus covariance continuity is retained without changing default
tracking; it becomes active only for explicit covariance-aware backend
consumers.

### Non-rectified stereo BA factor and MH_03 rejection (2026-07-15)

The next design was implemented as an actual observation factor rather than a
replacement landmark mean. `BaGeneralStereoObservation` retains calibrated
left and right pixels, the right camera intrinsics, and the fixed
`T_right<-left` rig transform. Its residual is the four-vector
`(u_l,v_l,u_r,v_r)` and its analytic pose/landmark Jacobians explicitly pass
the right branch through the rotational cam0-to-cam1 extrinsic. Synthetic
tests cover zero residual at truth, recovery of a perturbed landmark, and
recovery of a perturbed six-DoF pose with unequal intrinsics, nonzero vertical
translation, and a rotated right camera. This is necessary for EuRoC: reducing
the rig to rectified `(u_l,v_l,u_r)` plus a scalar baseline discards real
calibration geometry.

The right measurement now survives seed bootstrap, stereo replenishment,
candidate triangulation, transactional map staging, loop-observation
reassignment, and covisibility/post-welding BA through a validated
`StereoObservation` sidecar. The BA builder replaces the left monocular factor
rather than adding it a second time. Missing or rejected right measurements
fall back to the historical monocular factor, and the entire consumer remains
opt-in through `--covisibility-local-ba-general-stereo`.

Correct geometry did not imply a safe online policy. On the same strict-CUDA
MH_03 300-frame run, mono covisibility BA achieved tracking `0.963`, rigid /
final-keyframe ATE `0.1504 / 0.1636 m`, d1 RPE `0.2103 m / 4.16 deg`, and d10
RPE `0.2893 m / 4.79 deg`. Unconditionally consuming the right sidecars reduced
tracking to `0.873`, changed ATE to `0.1554 / 0.1946 m`, and worsened both
rotation RPEs. A 5 px initial right-reprojection gate was worse still:
tracking `0.807`, ATE `0.2515 / 0.3081 m`, d1 `0.2551 m / 4.92 deg`, and d10
`0.4813 m / 8.04 deg`. Both stereo consumers are rejected as defaults. The
factor and data path remain an explicit research control; the production
default is unchanged. The result indicates that repeatedly constraining a
temporally re-triangulated landmark with its original same-instant stereo
pixel requires a better observation-lifetime/outlier model, not merely a
tighter pixel gate.

With the stereo consumer disabled, the new measurement sidecar is behaviorally
inert. A strict-CUDA MH_03 300-frame rerun stored 1,563 right observations and
still reproduced the pre-sidecar run exactly: all five trajectory, frame-error,
ground-truth, keyframe-trajectory, and final-keyframe-error CSV SHA-256 hashes
matched. Tracking remained `0.903`, rigid/final-keyframe ATE `0.1718/0.2626
m`, d1 `0.2844 m / 6.20 deg`, and d10 `0.4194 m / 7.06 deg`.

The same A/B also rechecked whether the historical monocular covisibility BA
should be promoted independently of the new stereo factor. It looked strong
at 300 frames (`0.963` tracking and `0.1636 m` final-keyframe ATE versus the
no-BA prefix's `0.903` and `0.2626 m`), but reversed by 1,000 frames. Ungated
mono BA reached tracking `0.892` yet worsened final-keyframe ATE from the no-BA
`2.7717 m` to `4.0105 m`; d1 translation RPE exploded from `0.4272 m` to
`4.4662 m`. Its first solve had no fixed boundary keyframe and reduced a
`70.19 px` reprojection objective to `0.76 px`, exposing a locally valid but
globally gauge-unsafe write-back. Enabling the existing transactional fixed-
boundary (`fixed >= ceil(0.34 * optimized)`) and behind-camera (`<= 0.3`)
gates prevented that collapse, but still failed the joint promotion gate:
tracking `0.866`, final-keyframe ATE `2.7983 m`, d1 RPE `0.4790 m / 6.79 deg`,
and d10 RPE `3.2106 m / 33.38 deg`, versus no-BA `0.872`, `2.7717 m`, `0.4272
m / 6.29 deg`, and `3.2076 m / 31.82 deg`. Online covisibility BA therefore
remains disabled in the adopted baseline; short-prefix reprojection gains are
not treated as trajectory evidence.

### Three-sequence counterbalanced promotion matrix (2026-07-15)

The adopted loop-welding configuration completed the declared strict-CUDA
matrix at
`E:/visloc_archive/final_loop_welding_3seq_1000_20260715_pwsh`: MH_01,
MH_03, and MH_05, no-loop control versus appearance-loop treatment, three
counterbalanced repetitions, and 1,000 frames per run. All 18 root run
manifests have exit code zero, no validation error, the same executable/model/
ONNX Runtime hashes, and unique sequence/variant/repetition identities. The
machine-readable and rendered reports are
`docs/generated/visual_slam_final_3seq_1000_20260715.json` and
`docs/generated/visual_slam_final_3seq_1000_20260715.md`.

| sequence | variant | tracking / longest | live / final-KF rigid ATE m | d1 RPE m / deg | d10 RPE m / deg | accepted correct/evaluated | mean runtime ms/frame | decision |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| MH_01 | no loop | 0.966 / 558 | 0.2737 / 0.3040 | 0.1743 / 6.00 | 0.3576 / 16.85 | 0/0 | 131.8 +/- 28.6 | control |
| MH_01 | loop welding | 0.965 / 558 | 0.2492 / 0.2543 | 0.1734 / 5.99 | 0.3545 / 14.20 | 6/6 | 127.0 +/- 12.9 | `PROMOTE` |
| MH_03 | no loop | 0.872 / 215 | 2.5609 / 2.7717 | 0.4272 / 6.29 | 3.2076 / 31.82 | 0/0 | 135.9 +/- 13.5 | control |
| MH_03 | loop welding | 0.872 / 215 | 2.5609 / 2.7717 | 0.4272 / 6.29 | 3.2076 / 31.82 | 0/0 | 140.8 +/- 13.5 | `SAFE_NO_OP` |
| MH_05 | no loop | 0.889 / 144 | 5.1402 / 5.3535 | 0.4153 / 12.46 | 3.3539 / 28.48 | 0/0 | 144.6 +/- 19.7 | control |
| MH_05 | loop welding | 0.889 / 144 | 5.1402 / 5.3535 | 0.4153 / 12.46 | 3.3539 / 28.48 | 0/0 | 127.7 +/- 12.9 | `SAFE_NO_OP` |

MH_01's final-keyframe rigid ATE improves by 16.3% without reducing its
longest continuous segment; delta-10 rotation RPE improves from 16.85 to 14.20
degrees. All six accepted constraints are ground-truth-correct. MH_03 and
MH_05 accept no loop, so numerical equality alone is not used as safety
evidence: all 30 SHA-256 comparisons covering five accuracy artifacts, three
repetitions, and two sequences match between treatment and paired control. The
matrix-level decision is therefore `PROMOTE`: one declared sequence improves
and the other two are hash-proven no-ops. Runtime and working-set gates use
same-repetition ratios and remain within their 25% limits; the noisy runtime
means are reported but are not claimed as a speedup.

One first attempt at `MH_03_medium_appearance_loop_r03` was killed before it
could produce a summary after unrelated external CPU load exceeded four cores
for three consecutive resource samples. Its manifest is preserved under the
matrix `_failed_attempts` directory and is not an accuracy/runtime sample. A
same-protocol retry completed with exit code zero and is the only r03 candidate
included in the 18-run report.

The promoted treatment is the fixed-weight 0.1 SE(3) loop edge with PnP
verification, pairwise-only PCM, GNC, correction propagation, transactional
loop-observation fusion, and loop-welding BA. The generalized non-rectified
stereo local-BA factor and continuous covisibility local BA remain opt-in and
disabled because their MH_03 screens regressed the joint trajectory metrics.
This is a reproducible 1,000-frame prefix decision, not a full-sequence EuRoC
or ORB-SLAM3-parity claim; longer-horizon tracking and tight-VI promotion remain
separate research work.

### ROBOMECH 2026 overview cross-check (2026-07-15)

The [SfM / Visual SLAM / Visual Localization overview presented for ROBOMECH
2026](https://docswell.com/s/ystk_hara/K4N93D-sfm-vslam-vloc-robomech2026)
is a useful secondary map of the field, especially its separation of local
optimization from global BA/pose-graph correction and its emphasis on ATE,
runtime, tracking interruptions, and learned local features. Repository
decisions still use the cited primary papers for algorithmic claims. Applied to
the current EuRoC evidence, the overview reinforces two concrete rules: a local
reprojection reduction is not global trajectory evidence, and a tracking-cliff
candidate must report continuity and recovery behavior alongside ATE. The
selective tight-VI falsification screen and resulting next hypothesis are
recorded in
[`generated/selective_tight_vi_diagnostics_20260715.md`](generated/selective_tight_vi_diagnostics_20260715.md).

The subsequent full-sequence result closes that experiment. Following
[Forster et al.](https://arxiv.org/abs/1512.02363) for on-manifold preintegration,
[VINS-Mono](https://arxiv.org/abs/1708.03852) and
[OKVIS](https://github.com/ethz-asl/okvis) for a joint pose/velocity/bias visual-
inertial window, and [ORB-SLAM3](https://arxiv.org/abs/2007.11898) for continued
visual-inertial optimization and recovery context, the implementation now uses
covariance-bounded continuation, strict initializer/local-NIS write-back gates,
and a carried marginalization prior. A support-selective configuration was run
on all MH_01/MH_03/MH_05 frames for three counterbalanced repetitions. MH_01
and MH_03 are hash-exact control no-ops; MH_05 improves tracking coverage,
longest continuity, rigid ATE, and both translation RPE horizons while its two
rotation RPE regressions remain below the declared 2% non-inferiority limit.
The reproducible report is
[`generated/tracking_cliff_tight_vi_full_3rep_20260715.md`](generated/tracking_cliff_tight_vi_full_3rep_20260715.md).

## Literature backlog

The next survey pass deepens calibration/observability analysis and covers
line/plane/object landmarks and long-term map maintenance. Each addition must
state a testable repository hypothesis rather than only summarize the abstract.
