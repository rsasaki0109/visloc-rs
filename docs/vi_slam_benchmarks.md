# Visual-Inertial SLAM (Basalt Rust port) — benchmark details

This page preserves the detailed VI-SLAM measurements relocated from the
[README](../README.md) front page. The estimator is a separate, faithful Rust
port of upstream [Basalt](https://github.com/VladyslavUsenko/basalt) commit
`0f3b2b52` — a tightly-coupled stereo-inertial VIO estimator plus an offline
structure-from-motion mapper — living in [`pipelines/basalt`](../pipelines/basalt)
and exercised end to end by
[`examples/basalt_euroc_vio_demo.rs`](../examples/basalt_euroc_vio_demo.rs).
It is not the vision-only stereo SLAM stack used in the SfM sections: Basalt
fuses IMU preintegration directly into the sliding-window optimizer, and this
port targets upstream numerical and structural fidelity (matching Basalt's own
frontend, factors, and marginalization) on the standard EuRoC benchmark, not a
from-scratch design.

<p align="center">
  <img src="assets/basalt_online_vs_orbslam3_trajectories.png" alt="2x3 grid of EuRoC top-down trajectories (visloc-rs online VI-SLAM vs ORB-SLAM3 vs ground truth, all SE(3)-aligned) for MH_01, MH_03, V1_02, V1_03, V2_01, V2_02" width="820">
</p>

<p align="center"><sub>Six of the eight EuRoC sequences where this
repository's online VI-SLAM (blue) beats measured ORB-SLAM3
(red), both SE(3)-Umeyama-aligned to EuRoC ground truth (black). Full
results and protocol below.</sub></p>

## Beating ORB-SLAM3 on EuRoC

Same evaluator, one run per sequence, full-trajectory ATE with SE(3) Umeyama
alignment; ORB-SLAM3 measured on this machine (not a paper number). Both
systems use EuRoC's official per-sequence calibration — ORB-SLAM3 always did;
this repository switches its Basalt VIO's input calibration to match (see
"How this works" below). Estimator code is otherwise unchanged from the
faithful-port parity result linked at the end of this section. The mapper
runs **online**: a dedicated thread ingests each keyframe as the VIO
produces it (incremental detect/match/loop-check, a rate-limited
background-thread bundle adjustment, one final full pass at the end) instead
of a separate offline batch job over the whole sequence; see
[the online mapper design](basalt_online_mapper_design.md) and
[the global-consistency plan §1.5](vi_slam_global_consistency_plan.md#15-result-2026-09-16-the-offline-mappers-811-result-reproduced-online)
for the architecture and the full before/after evidence. The original
offline batch mapper (2-18 min / 3-6 GB peak RSS per sequence) still exists,
unchanged, for parity/comparison — see "How this works" and "Run it" below.

| Sequence | visloc-rs (online VI-SLAM) | ORB-SLAM3 stereo-inertial | Winner |
| --- | ---: | ---: | :---: |
| MH_01_easy | 0.0155 | 0.0363 | visloc-rs |
| MH_02_easy | 0.0254 | 0.0334 | visloc-rs |
| MH_03_medium | 0.0256 | 0.0283 | visloc-rs |
| MH_04_difficult | 0.0702 | 0.0428 | ORB-SLAM3 |
| MH_05_difficult | 0.0633 | 0.0546 | ORB-SLAM3 |
| V1_01_easy | 0.0355 | 0.0380 | visloc-rs |
| V1_02_medium | 0.0139 | 0.0170 | visloc-rs |
| V1_03_difficult | 0.0213 | 0.0287 | visloc-rs |
| V2_01_easy | 0.0171 | 0.0390 | visloc-rs |
| V2_02_medium | 0.0119 | 0.0140 | visloc-rs |
| V2_03_difficult | 0.0445 | 0.0563 | visloc-rs |

<p align="center"><sub>Re-measured 2026-09-28 with the real-time work
below (`vio_max_iterations` 7 -> 5): 9/11 wins held, V2_01_easy's ATE grew
from 0.0153 to 0.0171 m (+11.8%, 3-run median) and V1_03_difficult's from
0.0204 to 0.0215 m (+5.4%, 3-run median) — both still comfortably beat
ORB-SLAM3 (2.3x and 1.3x margin). Every other sequence moved by <5%,
mostly within run-to-run noise. See "VIO speed: real time on all 11/11
sequences" below for the full accounting.</sub></p>

<p align="center"><sub>9/11 wins (2026-09-28 re-measurement, up from 8/11):
V2_03_difficult flipped from a loss (0.1078 m) to a win as a side effect of
the VIO speed work below — see "VIO speed: `--pipeline` becomes the online
demo's default" for why. Because the online mapper's propagated trajectory
is not reproducible run-to-run (live-threaded optimizer timing; see "Honest
caveats"), the flip was checked for luck: 3 additional runs with the same
exe gave V2_03_difficult 0.0446/0.0450/0.0447 m (median across all 4 runs
including the table's 0.0451 m: **0.0449 m**, range 0.0446-0.0451 m), and
V1_03_difficult (the table's other trigger-count-sensitive sequence)
0.0193/0.0193/0.0214 m (median across all 4: **0.0204 m**, range
0.0193-0.0214 m) — both comfortably below ORB-SLAM3 (0.0563 m, 0.0287 m) on
every single run, not just on average. Evidence:
`E:\visloc-rs-runs\vio_rt_runs\repro_check\`. All values are full-trajectory
ATE translation RMSE in metres, lower is better, driver
<code>scripts/run_basalt_online_all11.py</code> equivalent (same protocol,
<code>--pipeline --threads 12</code>), artifacts
<code>E:\visloc-rs-runs\vio_rt_runs\online_all11_pipeline\{summary.json,status/,runs/}</code>.
The estimator's own VIO trajectory is unchanged bit-for-bit by this
re-measurement (`--pipeline` is byte-identical to serial, see below); only
the mapper's live-threaded correction timing shifted, which is why most
sequences moved by only a few percent while V2_03 moved by more. VIO alone,
before the mapper's corrections, already beats ORB-SLAM3 on MH_01_easy
(0.030 vs 0.036 m) and V2_01_easy (0.027 vs 0.039 m).</sub></p>

<p align="center">
  <img src="assets/basalt_online_vs_orbslam3.png" alt="Bar chart of full-trajectory SE(3) ATE, visloc-rs online VI-SLAM vs measured ORB-SLAM3, across all 11 EuRoC sequences" width="820">
</p>

### Online-specific numbers: speed, and honest gaps vs the offline number

The offline-mapper table above is what "online" is measured against
per-sequence; here is that comparison plus the operational numbers (real-time
factor, mapper queue lag, loop/trigger counts, peak RSS) the offline table
never had because the offline mapper is not a live process:

| Sequence | Online SE3 | Offline SE3 | vs offline | RTF | Max queue lag | Loop pairs | Optimizer triggers | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MH_01_easy | 0.0155 | 0.0154 | +0.6% | 0.601 | 95.08s | 600 | 4 | 753 MB |
| MH_02_easy | 0.0256 | 0.0244 | +4.9% | 0.493 | 44.23s | 467 | 4 | 567 MB |
| MH_03_medium | 0.0256 | 0.0264 | -3.0% | 0.571 | 13.42s | 477 | 4 | 448 MB |
| MH_04_difficult | 0.0702 | 0.0849 | -17.3% | 0.522 | 2.68s | 246 | 3 | 313 MB |
| MH_05_difficult | 0.0633 | 0.0611 | +3.6% | 0.623 | 1.82s | 202 | 3 | 310 MB |
| V1_01_easy | 0.0354 | 0.0352 | +0.6% | 0.893 | 11.38s | 195 | 4 | 457 MB |
| V1_02_medium | 0.0138 | 0.0139 | -0.7% | 0.983 | 3.28s | 119 | 3 | 275 MB |
| V1_03_difficult | 0.0214 | 0.0177 | **+20.9%** | **1.199** | 11.62s | 89 | 5 | 360 MB |
| V2_01_easy | 0.0153 | 0.0162 | -5.6% | **1.076** | 9.48s | 42 | 3 | 354 MB |
| V2_02_medium | 0.0119 | 0.0103 | **+15.5%** | 0.932 | 10.94s | 125 | 4 | 324 MB |
| V2_03_difficult | 0.0451 | 0.0653 | **-30.9%** | **1.242** | 2.24s | 114 | 4 | 247 MB |

Re-measured 2026-09-28 on the same protocol, with `--pipeline --threads 12`
(see below). **Speed: 3/11 sequences now clear real-time (RTF >= 1.0) —
V1_03_difficult, V2_01_easy, V2_03_difficult — and two more are within 7%
(V1_02_medium 0.983, V2_02_medium 0.932), all measured under CPU contention
from another concurrent job on the same machine, so idle numbers are
expected to be higher still.** RTF rose 3.4-7.1x per sequence versus the
prior measurement (0.088-0.379) purely from parallelization that had already
landed on `main` since that measurement plus this session's `--pipeline`
default flip — no algorithm or accuracy-affecting change (§ below).
**Accuracy: still 8/11 sequences within ~20% of the offline mapper's own
number; V1_03 and V2_02 remain outside it, both for the same reason as
before (fewer optimizer triggers than average). V2_03 flipped from
+65.2% (a real gap) to -30.9% (now *beats* its own offline number)** — its
optimizer trigger count went from 2 (the fewest of all 11, the suspected
root cause named in the prior measurement) to 4, because the now-faster
VIO/mapper pair gives the rate-limited background-optimizer trigger more
real-time chances to fire over the same sequence. This is a live-threading
timing effect, not a code change to the mapper: the trigger logic, its rate
limit, and the offline mapper itself are byte-for-byte untouched. On every
sequence the mapper thread still kept up with the VIO thread (no sequence's
total wall time was gated by the mapper falling behind faster than its
queue capacity). Full per-sequence artifacts:
`E:\visloc-rs-runs\vio_rt_runs\online_all11_pipeline\{summary.json,status/,runs/}`.

### VIO speed: `--pipeline` becomes the online demo's default

The online demo's two-thread frontend/estimator overlap (`--pipeline`,
added by PR #153) was previously opt-in and unused by the benchmark driver,
so every number above through 2026-09-20 was measured on the serial path.
It is now the default (`--no-pipeline` restores serial; `--realtime`'s
dataset-paced mode still only exists on the serial path and now falls back
to it automatically instead of erroring, unless `--pipeline` was passed
explicitly). This is a pure scheduling change: verified byte-identical
`trajectory.tum` SHA-256 and byte-identical `marg_data/` between serial and
`--pipeline --threads 12` on MH_03_medium and MH_04_difficult (300 frames
each, `basalt_euroc_vio_demo`, which has no mapper thread to introduce its
own timing variance). The **online** demo's mapper-corrected
`trajectory_online.tum` is not expected to be bit-identical between the two
modes — that file already depends on live mapper-thread scheduling and was
already documented as non-reproducible run-to-run (see "Honest caveats"
below); the underlying VIO computation it is built from did not change.

Separately, the landmark-reduction hot path
(`reduce_landmark_factors_f32_checked_with_options`, ~25% of instrumented
VIO wall time, already `par_iter`-parallelized across factors per §1.6/7c)
was re-investigated for further bit-identical wins: a rayon `with_min_len`
scheduling tweak and a thread-local QR-workspace-reuse variant were both
built, both verified byte-identical, and both measured with paired
alternating timing against the unmodified baseline — neither showed a
reproducible speedup (median 2-3% *slower*, within the run-to-run
contention noise on this machine). Both changes were reverted; this matches
the plan doc's own stated kill criterion for stage 7c ("if reduction is
already negligible after 7b, stop"). Full evidence in
`E:\visloc-rs-runs\vio_rt_runs\codex_work\reduction_investigation\report.md`.

Real-time factor (dataset duration / VIO wall time) is still bounded by the
VIO estimator's own per-frame compute; getting the remaining 8 sequences to
RTF >= 1.0 next requires either idle-machine measurement (this session's
numbers are all under contention) or non-bit-identical, accuracy-gated
levers (fewer LM iterations, cheaper frontend settings, smaller window) —
see [the global-consistency plan §1.6/7c](vi_slam_global_consistency_plan.md)
for the current status of that follow-up.

### VIO speed: real time on all 11/11 sequences (2026-09-28)

**Result: RTF >= 1.0 on every one of the 11 EuRoC sequences**, measured
clean (no other job sharing the machine — verified by polling for
`gsplat_euroc.exe`, the concurrent SfM benchmark process, before and
throughout each run; a shared-machine `TIMING_LOCK` file coordinates this
across agents). Three levers, applied in this order:

**1. `vio_max_iterations` 7 -> 5 (accuracy-gated, config-only).**
A timing-bucket breakdown (`basalt-timing-breakdown` feature, MH_02/MH_04,
300 frames, `--pipeline`) showed `estimator_lm_solve` at 78-82% of real VIO
wall time on both, and `lm_landmark_reduction` (already `par_iter`
parallel) a consistent 44-45% of that — i.e. most of the remaining cost is
the LM outer loop's own per-trial bookkeeping (linearization, cost
evaluation, model-decrease checks), which scales with the iteration count,
not something further parallelism can remove (matches the negative
scheduling-tweak result above). Cutting the iteration budget cuts all of
it proportionally. Full 11-sequence gate: **9/11 wins held** (same two
losses, MH_04/MH_05); ATE moved <5% on 8/11 sequences, but V1_03_difficult
grew from 0.0204 to 0.0215 m (+5.4%, 3-run median; V1_03 was already
documented as run-to-run non-reproducible) and V2_01_easy — not previously
flagged as sensitive — grew from 0.0153 to 0.0171 m (+11.8%, 3-run
median). Both remain decisive wins vs ORB-SLAM3 (1.3x and 2.3x margin).
Config: `configs/basalt/variants/official_euroc_ds/euroc_config.json`.
Evidence: `E:\visloc-rs-runs\vio_rt_runs\online_all11_maxiter5\` and
`repro_check_maxiter5\`.

Two other accuracy-gated levers were tried and **rejected** — both broke
the 9/11 gate by flipping V2_03_difficult to a loss: `optical_flow_
detection_grid_size` 50 -> 55/60 (sparser point detection; 60 flipped V2_03
to a reproducible loss, median 0.0601 m vs ORB-SLAM3's 0.0563 m over 3
runs; also grew MH_04's ATE 24-39%) and `vio_max_kfs` 7 -> 5 (smaller
sliding window; flipped V2_03 to 0.0572 m median over 3 runs, also grew
V1_03 by 26%). Both are bit-identical-safe wins on most other sequences
(6-11% ATE improvement on several, real RTF gains) but V2_03_difficult
appears structurally fragile: it broke under every VIO-numerics-changing
lever tried this session, apparently because the online mapper's
rate-limited background-optimizer trigger count/timing (already the
documented cause of V2_03's run-to-run non-reproducibility) is sensitive
to *any* change in VIO output timing, not something specific to grid size
or window size. Neither lever was needed once the two levers below were
found, so both stay reverted; the checked-in config is unchanged except
for `vio_max_iterations`.

**2. `release-rt` Cargo build profile (bit-identical, zero accuracy
risk).** The online demo's estimator thread runs at only 0.2-2.2 average
CPU cores despite `--threads 12` and an already-parallel landmark-reduction
hot path — investigated and ruled out as a real "stall": not disk I/O (a
same-sequence twice-in-a-row cache test showed no meaningful wall-time
benefit), not mapper-queue backpressure (`max_mapper_queue_depth` stayed
far under `mapper_queue_capacity` on every sequence), not `--realtime`
pacing (default is `as_fast_as_possible`), and not a lock held by the
mapper thread (only small stats-bookkeeping locks exist on that path). The
low average instead reflects each `lm_landmark_reduction` call being only
~7-8ms — far below per-second CPU sampling granularity — diluting real but
brief parallel bursts; the estimator thread is genuinely compute-bound.
Given that, the codegen itself was the remaining lever: a `[profile.
release-rt]` in the workspace `Cargo.toml` (`inherits = "release"`,
`lto = "fat"`, `codegen-units = 1`, `panic = "abort"`) — a separate opt-in
profile so the historical `release` profile used by tests and other
binaries is untouched. **Verified bit-identical** against `release`:
`trajectory.tum`/`trajectory_vio.tum` SHA-256 and `marg_data/` contents
match exactly (`basalt_euroc_vio_demo` and `basalt_euroc_online_slam_demo`,
MH_03_medium, 300 frames). Clean alternating-pair timing (600 frames,
`vio_wall_seconds`, current `vio_max_iterations=5` config): MH_03_medium
RTF `release` [0.402, 0.509, 0.569] (median 0.509) -> `release-rt` [1.000,
0.827, 1.092] (median 1.00), **+96%**; MH_02_easy `release` [0.663, 0.767,
0.989] (median 0.767) -> `release-rt` [1.279, 1.163, 0.501*] (median
1.163), **+52%** (*one `release-rt` run coincided with unrelated
`chrome-headless-shell` processes consuming CPU on the shared machine; the
other 5/6 measurements were consistent).

**3. `mimalloc` global allocator (bit-identical, zero accuracy risk,
opt-in).** The VIO estimator's LM trial loop allocates a scratch buffer
per factor per trial, so a faster allocator was worth trying on top of
`release-rt`. Added `mimalloc` as an optional dependency and a
`mimalloc-global` feature that installs it as the online demo's
`#[global_allocator]`. **Verified bit-identical**: `trajectory_vio.tum`
SHA-256 matches the plain `release-rt` build exactly (MH_03_medium, 300
frames). Clean, two alternating-pair rounds, 600 frames, on top of
`release-rt`: MH_02_easy RTF [1.516, 1.562] (mean 1.539) -> +mimalloc
[1.716, 1.695] (mean 1.706), **+10.8%**; MH_03_medium [1.239, 1.317] (mean
1.278) -> +mimalloc [1.406, 1.408] (mean 1.407), **+10.1%**.

**Final full-11 clean RTF table** (`release-rt` + `mimalloc-global`,
`vio_max_iterations=5`, `TIMING_LOCK` held throughout, `gsplat_euroc.exe`
confirmed absent for the full run in each case):

| Sequence | ATE (m) | RTF | vs ORB-SLAM3 |
| --- | ---: | ---: | :---: |
| MH_01_easy | 0.0155 | 1.121 | WIN |
| MH_02_easy | 0.0254 | 1.058 | WIN |
| MH_03_medium | 0.0256 | 1.069 | WIN |
| MH_04_difficult | 0.0702 | 1.092 | LOSS |
| MH_05_difficult | 0.0633 | 1.073 | LOSS |
| V1_01_easy | 0.0355 | 1.114 | WIN |
| V1_02_medium | 0.0139 | 1.113 | WIN |
| V1_03_difficult | 0.0213 | 1.222 | WIN |
| V2_01_easy | 0.0171 | 1.098 | WIN |
| V2_02_medium | 0.0119 | 1.230 | WIN |
| V2_03_difficult | 0.0445 | 1.682 | WIN |

**RTF >= 1.0 on 11/11 sequences (range 1.06-1.68x); 9/11 wins vs
ORB-SLAM3 unchanged** (bit-identical accuracy — RTF here is a pure
codegen/allocator change on top of the `vio_max_iterations=5` config
already gated above). Combined `release-rt` + `mimalloc-global` speedup
over the pre-session `release` build is roughly 2-2.5x on this machine.
Build: `RUSTFLAGS="-C target-feature=+avx2,+fma" cargo build --profile
release-rt --example basalt_euroc_online_slam_demo --features
basalt-lm-workspace-reuse,mimalloc-global` (see "Run it" below). Full
evidence: `E:\visloc-rs-runs\vio_rt_runs\online_all11_release_rt_mimalloc\
{summary.json,status/,runs/}`, `E:\visloc-rs-runs\vio_rt_runs\
buildlever_ab\`, `E:\visloc-rs-runs\vio_rt_runs\mimalloc_ab\`.

`target-cpu=native` and profile-guided optimization were both left
untried this session in the interest of keeping the default/documented
build portable across machines (a `target-cpu=native` binary is only
valid on the machine that built it); the two levers above already closed
the remaining gap to real time on every sequence without that tradeoff.

**Speed (2026-09-20): compact MargData LM path.** The VIO estimator now has an
opt-in `lean_marg_data` path that keeps the post-solve factor snapshot MargData
needs but skips the diagnostic LM payloads: no per-trial landmark
re-factorization, no duplicate pre-solve diagnostic linearization. Outputs are
byte-identical — `trajectory.tum` SHA-256 and the whole `marg_data/` tree match
the diagnostic path on 200- and 400-frame MH_03 runs. On 400 MH_03 frames the
total VIO wall time fell from 127-131 s to 39-40 s (**3.2x**), with the LM
solver itself **4.2x** faster (116-119 s -> 28 s); end-to-end online-mapper RTF
rose from 0.128 to **0.349**. The VIO demo and online demo expose
`--retained-marg-diagnostics` to restore the old path; the online demo enables
the compact path by default. The lean loop also caches the linearization and
landmark reduction across rejected damping trials and evaluates the model
decrease from the retained Q1/Q2 payload instead of re-factoring landmarks.
Across MH_01/02/03/04/05 (300-400 frames each) it is **3.6-4.3x** faster than
the diagnostic path with byte-identical trajectory and MargData (MH_04, the
rejection-heavy sequence, is 4.26x; its LM solve is 5.4x). Full evidence:
[lean MargData LM speedup](../work/vi_slam_lean_margdata_lm_speedup_20260920.md).
This does not change any algorithm, window setting, or calibration, so the
accuracy tables above are unaffected.

Getting an online mapper that keeps up with the VIO took three real bugs
found and fixed via live full-sequence reruns: a bounded channel that
blocked the VIO thread whenever the mapper fell behind (replaced with an
unbounded one — a queued keyframe packet is cheap once its raw pixels are
dropped, which happens in the same call that receives it); the mapper
re-processing a keyframe packet's *entire* marginalization window every
time instead of only the images newly introduced since the last packet
(consecutive packets overlap heavily); and a "trigger on any accepted loop"
policy that fired on nearly every packet through a loop-revisited corridor,
replaced with a rate-limited trigger that runs each periodic optimization on
a cloned snapshot in its own thread so ingestion is never blocked by it.
Full detail: [the online mapper design](basalt_online_mapper_design.md) and
[the global-consistency plan §1.5](vi_slam_global_consistency_plan.md).

**How this works.** Basalt's shipped calibration is its own from-scratch DS
recalibration of EuRoC's raw images, not EuRoC's official factory pinhole
calibration (`mav0/cam{0,1}/sensor.yaml`) that ORB-SLAM3 uses — and it
carries a ~1.4% metric-scale bias, isolated by a visual-side sensitivity
probe (not IMU noise, which was tested and ruled out) to the stereo
calibration itself (~+0.45% effective focal length, ~+0.15% baseline vs the
official calibration).
[`euroc_official_to_ds_calib.py`](../scripts/euroc_official_to_ds_calib.py)
converts EuRoC's official calibration directly into Basalt's DS model
(GT-free; see
[`configs/basalt/variants/official_euroc_ds/README.txt`](../configs/basalt/variants/official_euroc_ds/README.txt)
for the fit-region rationale), which removes most of the bias; the
unchanged faithful NFR mapper then closes loops and optimises globally on
top of the corrected VIO output — now running online (a dedicated thread
ingesting each keyframe as the VIO produces it) rather than as a separate
offline batch job, see the "Online-specific numbers" section above. Full
root-cause diagnostics (E1/E2 visual-vs-IMU decomposition, IMU-noise
negative controls) are in [the plan doc](vi_slam_global_consistency_plan.md).

Two things this result is **not**: (a) the mapper's own detection/matching/
triangulation/bundle-adjustment code is unchanged from the offline port —
online-izing it added incremental scheduling and threading around that
code, not a new algorithm, and the offline batch path (`pipelines/basalt/src/
mapper/{mod.rs,session.rs,features.rs,triangulation.rs}`) remains
byte-for-byte untouched; (b) real-time VIO — **now achieved on all 11/11
sequences** (real-time factor 1.06-1.68x, clean/idle measurement, see "VIO
speed: real time on all 11/11 sequences" above) via two bit-identical
build-level changes (a `release-rt` LTO/codegen-units/panic-abort profile
and the `mimalloc` global allocator) plus one accuracy-gated config change
(`vio_max_iterations` 7 -> 5, 9/11 wins held). The two
remaining losses (MH_04, MH_05) are VIO tracking-robustness limits on
fast/motion-blurred sequences, not mapper or calibration limits — see
[`vi_slam_global_consistency_plan.md`](vi_slam_global_consistency_plan.md)
for next steps.

### VIO accuracy: frontend robustness on MH_04/MH_05 (2026-09-29)

Per-frame frontend diagnostics (`--frontend-stats-csv`, new opt-in flag on
`basalt_euroc_online_slam_demo`) correlated against per-frame consecutive-pose
RPE showed MH_04/MH_05's accuracy gap is driven by track *churn* under
motion blur (rising `RejectReason::FrameFbSquared` — forward-backward KLT
inconsistency — and replacement-track creation during fast-motion bursts),
not by point starvation (`num_observations` stays well above grid capacity
throughout; `FastNoCandidate`, FAST finding no corner anywhere, fires under
40 times total per ~2000-frame sequence). Full diagnosis methodology and
numbers: [the global-consistency plan §1.8](vi_slam_global_consistency_plan.md).

**`optical_flow_levels` 3 -> 4 (config-only)** targets this directly by
widening the KLT pyramid's coarse-to-fine displacement range. Definitive
measurement (same exe, all 11 sequences, baseline immediately followed by
`levels4` per sequence so both share the same machine conditions;
`E:\visloc-rs-runs\vio_fe_runs\gate_baseline_vs_levels4_all11\summary.json`):

| Sequence | Baseline ATE | `levels4` ATE | ATE change | Baseline RTF | `levels4` RTF | vs ORB-SLAM3 |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| MH_01_easy | 0.0154 | 0.0138 | -10.4% | 0.947 | 0.879 | WIN / WIN |
| MH_02_easy | 0.0256 | 0.0233 | -8.8% | 0.309* | 1.030 | WIN / WIN |
| MH_03_medium | 0.0255 | 0.0237 | -6.9% | 0.646 | 0.587 | WIN / WIN |
| MH_04_difficult | 0.0702 | 0.0618 | **-12.0%** | 1.037 | 1.095 | LOSS / LOSS (gap 64%->44%) |
| MH_05_difficult | 0.0632 | 0.0570 | **-9.7%** | 1.006 | 1.036 | LOSS / LOSS (gap 16%->4%) |
| V1_01_easy | 0.0354 | 0.0361 | +1.7% | 0.990 | 0.838† | WIN / WIN |
| V1_02_medium | 0.0138 | 0.0120 | -13.1% | 1.021 | 1.084 | WIN / WIN |
| V1_03_difficult | 0.0215 | 0.0244 | +13.6% | 0.849 | 0.904 | WIN / WIN |
| V2_01_easy | 0.0165 | 0.0173 | +4.3% | 1.093† | 0.813 | WIN / WIN |
| V2_02_medium | 0.0119 | 0.0107 | -10.0% | 0.783 | 0.647 | WIN / WIN |
| V2_03_difficult | 0.0447 | 0.0380 | -15.0% | 1.092 | 1.009 | WIN / WIN |

**9/11 wins held for both configs, identical pattern** (only MH_04/MH_05
lose) — both loss margins narrow substantially (64%->44%, 16%->4%) but
neither flips, so this does not clear the ">9/11" bar and the checked-in
default config is unchanged; `levels4` stays an available variant
(`configs/basalt/variants/official_euroc_ds/euroc_config_levels4.json`).

**RTF is not clean in this measurement and the gate's real-time bar
("RTF >= 1.0 on all 11") is not met by *either* config**: baseline itself
is below 1.0 on 6/11 sequences here, well under the historically documented
1.06-1.68x clean table. Root-caused to two distinct sources, both logged by
a 15-second-interval process-activity watchdog held for the whole run: (1)
a marked at † above ran fully or partly concurrent with a recurring,
low-footprint `cargo.exe`/`rustc.exe` build cycle in Windows Session 0
(services context, not the console session either this work or the
already-waited-out OpenLORIS build used) that appeared repeatedly
throughout the run and was not caught by the initial pre-run check alone —
likely another agent's own Codex CLI-driven `cargo check`/test cycle on
this shared machine; (2) the entry marked `*` (MH_02 baseline, RTF 0.309)
shows no watchdog-detected external process at all and is unexplained —
general session/machine slowness beyond what was tracked. Because baseline
and `levels4` shared the same run-to-run conditions per sequence (immediate
back-to-back), the *relative* comparison above (ATE and the ~similar RTF
degradation pattern on both configs) is still informative, but the
*absolute* RTF column should not be read as a clean, final answer on
whether `levels4` costs real time — a dedicated re-run on a verified-idle
machine (not just checked once before starting) is needed to settle that
question. `optical_flow_levels=5` was also tried in an earlier, separate
measurement: MH_04 improves further (0.0533, -24.1%) but MH_05 is worse
than `levels4` (0.0595, -6.0% vs. baseline) — a real trade-off, not a
strict further win, so `levels4` remains the better single fixed config of
the two.

An IMU/gyro-rotation-seeded KLT initialization lever (`--imu-seed-klt`,
predicting each track's new-frame search position from integrated gyro
instead of a zero-motion seed) was also tried and is a verified honest
negative after three variants (sign, extrinsic frame convention, and gyro
bias correction were each checked against this codebase's own conventions
and the real EuRoC calibration, not just re-derived) — see §1.8 of the
plan doc for the full investigation.

## Pipeline

```mermaid
flowchart LR
    IMG["Stereo cam0 / cam1 PNGs<br/>euroc.rs"] --> PYR["Basalt image pyramid<br/>pyramid.rs"]
    PYR --> FAST["FAST-9 grid detector<br/>fast.rs"]
    FAST --> OF["Frame-to-frame optical flow<br/>Pattern51 patches<br/>stream.rs / patch.rs / pattern.rs"]
    OF --> EPI["Stereo epipolar filter<br/>adapter.rs"]
    IMU["IMU samples"] --> PRE["IMU preintegration<br/>imu/preintegration.rs, imu/sampling.rs"]
    EPI --> INIT["Initialization<br/>initialization.rs"]
    PRE --> INIT
    INIT --> LM["ABS_QR LM sliding window + FEJ<br/>vio/estimator.rs, vio/window/, vio/aom/"]
    LM --> MARG["Square-root marginalization<br/>vio/margdata.rs"]
    MARG --> OUT["Trajectory (TUM / CSV) + MargData packets<br/>adapter.rs"]
    OUT --> MAP["Offline mapper<br/>mapper/mod.rs, mapper/session.rs"]
    MAP --> MATCH["Keyframe matching + 5-pt Stewenius RANSAC<br/>mapper/features.rs"]
    MATCH --> TRI["Triangulation + bundle adjustment<br/>mapper/triangulation.rs"]
    TRI --> PTS["map.json / points.json / poses.json"]
```

<p align="center"><sub>Node labels are the actual source modules under
<a href="../pipelines/basalt/src">pipelines/basalt/src</a>. Marginalization
(<code>sqrt_to_sqrt_marginalize</code> in <code>vio/margdata.rs</code>) runs
every frame; a MargData packet is only written to disk when Basalt selects a
keyframe for removal, which is what feeds the offline mapper.</sub></p>

## Run it

Build with AVX2/FMA, the LM-workspace-reuse optimization, the `release-rt`
profile (fat LTO, one codegen unit, no unwind tables) and `mimalloc` — the
build the real-time numbers above were measured with, verified
bit-identical to plain `--release` (see above) — then replay one EuRoC
sequence through the **online** VIO + mapper (`--calibration`/`--config`
below are checked into this repository; `--euroc-dir` is an external
dataset path) — this reproduces the headline table:

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" \
  cargo build --profile release-rt --example basalt_euroc_online_slam_demo \
  --features basalt-lm-workspace-reuse,mimalloc-global

RUSTFLAGS="-C target-feature=+avx2,+fma" \
  cargo run --profile release-rt --example basalt_euroc_online_slam_demo \
  --features basalt-lm-workspace-reuse,mimalloc-global -- \
  --euroc-dir /path/to/MH_01_easy \
  --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
  --out-dir target/basalt_mh01_online \
  --optimize-every-k 100 --periodic-iterations 4
```

A plain `cargo build --release --example basalt_euroc_online_slam_demo
--features basalt-lm-workspace-reuse` (the historical command, no
`--profile release-rt`, no `mimalloc-global`) is still correct — same
`trajectory_online.tum` bit-for-bit modulo the mapper's live-threaded
timing — just roughly 2-2.5x slower on this machine.

This writes `trajectory_online.tum` (full-frame, mapper corrections
propagated to every VIO frame — the file scored above),
`trajectory_online_kf.tum` (keyframes only), and
`timing_breakdown_online.json` (RTF, mapper queue lag, loop/trigger counts,
per-stage optimize timing, peak RSS). `--realtime` paces input at dataset
rate instead of as fast as possible; `--help` lists every flag. The example
never reads a ground-truth file — all outputs are produced causally from
sensor data and estimator state.
[`scripts/run_basalt_online_all11.py`](../scripts/run_basalt_online_all11.py)
drives this across all 11 EuRoC sequences (detached, resumable, live
`summary.md`/`summary.json`).

The original **offline** VIO + batch-mapper path
(`examples/basalt_euroc_vio_demo.rs` +
`examples/basalt_mapper_offline_demo.rs`) is unchanged and still available
for parity/comparison runs:

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" \
  cargo build --release --example basalt_euroc_vio_demo --features basalt-lm-workspace-reuse
```

Then replay one EuRoC sequence (`--calibration` and `--config` below are
checked into this repository; `--euroc-dir` is an external dataset path):

```bash
cargo run --release --example basalt_euroc_vio_demo --features basalt-lm-workspace-reuse -- \
  --euroc-dir /path/to/MH_01_easy \
  --calibration benchmarks/basalt/release_inputs/euroc_ds_calib.json \
  --config configs/basalt/euroc_config.json \
  --out-dir target/basalt_mh01 \
  --max-frames 80
```

This writes `trajectory.tum`, `trajectory.csv`, `trace.jsonl`, a
`marg_data/` directory of per-keyframe MargData JSON packets (the offline
mapper's input), and `summary.txt` under `--out-dir`. Pass `--no-trace`
and/or `--no-marg-data` to drop the diagnostic trace and MargData output
respectively; `--help` lists every flag. The example never reads a ground-truth
file — all outputs are produced causally from sensor data and estimator state.

To reproduce the same result via the offline batch mapper instead (the
result this branch's online path was checked against, not the headline
table above), swap in the official-calibration variant (keep
`--no-marg-data` off, since the mapper needs it) and then run the offline
mapper on the resulting `marg_data/` directory:

```bash
cargo run --release --example basalt_euroc_vio_demo --features basalt-lm-workspace-reuse -- \
  --euroc-dir /path/to/MH_01_easy \
  --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
  --out-dir target/basalt_mh01_official

cargo run --release --example basalt_mapper_offline_demo -- \
  --marg-dir target/basalt_mh01_official/marg_data \
  --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
  --out-dir target/basalt_mh01_official_mapper
```

`configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json` is produced
from EuRoC's own factory pinhole-radtan calibration by
[`euroc_official_to_ds_calib.py`](../scripts/euroc_official_to_ds_calib.py)
(GT-free; see its
[README](../configs/basalt/variants/official_euroc_ds/README.txt) for the fit
decision). The mapper writes `trajectory.tum`/`trajectory.csv` (keyframe
poses) plus `poses.json`/`points.json`/`map.json`/`mapper_report.json`; to
propagate the mapper's corrections onto the full (non-keyframe) VIO
trajectory for scoring, run
[`propagate_basalt_mapper_corrections.py`](../scripts/propagate_basalt_mapper_corrections.py):

```bash
python3 scripts/propagate_basalt_mapper_corrections.py \
  --vio-trajectory-csv target/basalt_mh01_official/trajectory.csv \
  --mapper-poses-json target/basalt_mh01_official_mapper/poses.json \
  --out-tum target/basalt_mh01_official_mapper/full_trajectory.tum
```

`scripts/run_basalt_official_calib_all11.py` drives all three steps above
plus evaluation across all 11 EuRoC sequences (detached, resumable, writes a
live-updating `summary.md`/`summary.json`); see its module docstring.

## Honest caveats

- The mapper (offline or online) matches native's match graph exactly but
  its final point coordinates are only verified within 1 mm of native, not
  bit-exact.
- "Online" means the mapper thread keeps pace with the VIO thread (never
  blocking it); the VIO estimator itself is now parallelized, defaults to
  a two-thread frontend/estimator pipeline, and (as of 2026-09-28, with the
  `release-rt` build profile + `mimalloc`) runs at wall-clock real time
  (RTF >= 1.0) on all 11/11 sequences, clean/idle-measured, range
  1.06-1.68x — see the "VIO speed: real time on all 11/11 sequences"
  subsection above for the full breakdown and the accuracy-gated
  `vio_max_iterations` change that was needed alongside the two
  bit-identical build changes.
- V1_03_difficult and V2_02_medium's online ATE remains a real, reported gap
  from the offline mapper's own number (+20.9% and +15.5% respectively, both
  above the ~10% target) — likely because the rate-limited background
  optimizer trigger gets fewer chances to run on sequences with fewer loop
  events. V2_03_difficult had the same suspected cause and previously the
  largest gap (+65.2%); after the VIO speed work below changed its live
  optimizer-trigger timing (2 triggers -> 4), it now *beats* its offline
  number by 30.9% and its online ATE beats ORB-SLAM3 too. Not fully
  explained or deliberately tuned; see
  [the plan doc §1.5](vi_slam_global_consistency_plan.md) for the
  candidate cause and the branch history for the debugging record.
- MH_04 and MH_05 remain losses vs ORB-SLAM3 (online and offline alike):
  these are VIO tracking-robustness limits on fast/motion-blurred
  sequences, not calibration or mapper limits — see
  [the plan doc](vi_slam_global_consistency_plan.md) for next steps.
- The online-mapper **propagated** trajectory is not reproducible across
  identical runs (background-optimizer threading); the estimator-level VIO
  trajectory and the `MargData` bytes are deterministic and byte-identical.
- The `lean_marg_data` speed path (2026-09-20) is byte-identical to the
  diagnostic path but omits the `window.lm` diagnostic trace payload; callers
  that consume that trace must keep the default or pass
  `--retained-marg-diagnostics`.
- V2_02_medium's *offline* reference number (used only for the "vs offline"
  comparison, not the headline ORB-SLAM3 table) is from a manual rerun
  after the offline all-11 sweep's own attempt for that sequence hit a
  driver wall-time-monitor artefact.
- The offline batch mapper (2-18 minutes, 3-6 GB peak RSS per sequence) is
  unchanged and still available; peak RSS for the online path is reported
  per-sequence above (147-528 MB) but was not a tuning target for this
  stage — accuracy and keeping pace with the VIO were.
- The Cargo license inventory resolves 119/119 package licenses, but legal
  clearance was not sought or claimed — this is an engineering audit, not a
  legal one.

Separately, on upstream Basalt's own shipped calibration/config inputs (not
the official-calibration result above), the Rust port matches native
Basalt's ATE to within **0.1%** on all 11 EuRoC sequences and is byte-exact
cross-target (Windows ↔ Linux) at the trajectory/lifecycle level, with
**1.13×** runtime and **0.56×** peak RSS vs native on the same-domain Linux
measurement — this is the separate faithful-port parity claim; see the
[faithful-port closure report](../work/m11_basalt_faithful_port_final_closure_20260914.md)
and [upstream oracle / provenance](../benchmarks/basalt/README.md) for the full
parity evidence.
