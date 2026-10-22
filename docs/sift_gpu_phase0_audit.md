# GPU SIFT Phase 0 audit and Phase 1 baseline

Date: 2026-10-19  
Audited implementation revision: `2bdf090e2467835894d300888ecfda3ab6595961`  
Toolchain: repository-pinned Rust/Cargo 1.94.0; wgpu 30.0.1  
Development host: Apple M1, integrated GPU, Metal  
Production L40S/Vulkan: not available for this audit.  
No SIFT algorithm or feature selection changes were made as part of the audit.

## Reproducible baseline

The benchmark reports JSON Lines on stdout with `--jsonl`; `--gpu-timestamps` additionally requests optional pass-boundary timestamp queries, supported on this Apple M1 Metal adapter. Example full CPU/GPU correctness and latency sample:

```text
rustup run 1.94.0 cargo run --release -p visloc-sift-gpu --features gpu --example sift_gpu_bench -- --images /Users/marvin/Coding/GitHub/visloc-test/south-building/south-building/images --max-images 5 --jsonl
```

GPU-only sustained sample (all images available in this dataset directory):

```text
rustup run 1.94.0 cargo run --release -p visloc-sift-gpu --features gpu --example sift_gpu_bench -- --images /Users/marvin/Coding/GitHub/visloc-test/south-building/south-building/images --max-images 128 --no-cpu --jsonl
```

Optional pass timings, resolving timestamps once after each image's measured extraction interval:

```text
rustup run 1.94.0 cargo run --release -p visloc-sift-gpu --features gpu --example sift_gpu_bench -- --images /Users/marvin/Coding/GitHub/visloc-test/south-building/south-building/images --max-images 5 --no-cpu --jsonl --gpu-timestamps
```

Image-directory order is lexically sorted; each benchmark invocation warms up on its first selected image before timing it. `--max-keypoints` defaults to 8,192 in this example to match the production photo configuration; the library `SiftConfig::default()` remains unlimited. Machine-readable metadata includes adapter/backend, feature cap, CPU enablement, and whether GPU timestamp queries are supported/enabled. Each image record includes dimensions, timings, feature counts, CPU comparison, and descriptor dot similarity where applicable. Summary records include total/mean/median/p95 and images/s. The CPU/GPU sample's descriptor distribution aggregates only paired keypoints.

### Baseline measurements

South Building contains 128 JPEG images in the selected directory, each of the first 128 measuring 3072×2304 (3K, not 4K). At 8,192 returned features/image, release mode, Apple M1 / Metal:

- Pinned-toolchain GPU-only 128-image run (authoritative): 90.449265 s total, 0.706635 s mean, 0.706417 s median, 0.750196 s p95, 1.415158 images/s. All images returned 8,192 features. An earlier run with unpinned Homebrew rustc 1.99.1 measured 87.480505 s total / 0.683678 s median / 0.721475 s p95; excluded from authoritative figures because it violates the toolchain pin.
- Five-image CPU/GPU correctness run: CPU total 40.464384 s (8.092877 s mean); GPU total 5.075490 s (1.015098 s mean). GPU feature pairing with CPU was 99.96–100.00%; descriptor dot p0.1 0.99999946, p1 0.99999958, median 1.0.
- Five-image timed stage sample with timestamp queries: detection+scale-space pass median 460.175 ms, p95 472.174 ms; descriptor pass median 40.532 ms, p95 41.634 ms. Corresponding extraction wall median/p95 were 711.726/778.817 ms. Pass timestamps do not include host sorting, readback waits/copies, command submission overhead outside the pass, or CPU feature construction. Timestamp instrumentation also perturbs runtime slightly; do not compare its wall times directly with the non-instrumented sustained run.
- The 128-image workload is a sustained sample only for this local M1/3K dataset. At its measured 0.707 s/image, linear extrapolation would be about 38.9 minutes for 3,300 images of this size; this is not a 4K estimate and is not an L40S result. The 10-minute production goal remains unvalidated and likely requires substantial improvement. The full target 4K workload must be measured on the L40S.

At audit time, `rustc --version --verbose` reported Homebrew Rust 1.99.1, but that was not the repository toolchain: `rust-toolchain.toml` pins 1.94.0, and all authoritative validation/benchmark commands above explicitly use `rustup run 1.94.0`.

## Execution flow and implementation map

| Stage | Implementation | Work and output |
|---|---|---|
| Context and pipelines | `crates/gsplat-render/src/gpu.rs`; `crates/sift-gpu/src/extractor.rs` `SiftGpu::new` | Creates the high-performance adapter/device/queue; compiles 10 compute pipelines (upsample, aligned upsample, horizontal/vertical blur, halve, DoG, extrema, indirect-args, orientation, descriptor). |
| Input validation/topology | `extract`, `detect` | Rejects unsupported SIFT configs and images smaller than 16×16; derives interval count and automatic octave sizes. |
| Gaussian/DoG scale space | `detect`; `shaders/image.wgsl` | Uploads grayscale f32 input and Gaussian weights; doubles octave 0, blurs levels, halves later octaves, subtracts adjacent Gaussian levels into DoG. Operations are ordered in one compute pass. |
| Extrema detection | `shaders/detect.wgsl` `extrema` | One logical invocation per interior pixel and DoG interval; 3×3×3 extrema check, contrast and edge filtering; atomically appends a 7-word candidate per octave. Optional subpixel fit is configuration-controlled and off by default. |
| Indirect args/orientations | `orientation_args`, `orientation` in `detect.wgsl` | A 1-thread dispatch derives per-octave indirect dimensions from atomic candidate counts; a 64-thread workgroup per candidate builds the orientation histogram and atomically appends 9-word oriented-keypoint records. |
| Detection readbacks | `read_bytes` in `extractor.rs` | Blocking reads of counters (80 bytes) then the active oriented-keypoint prefix (`36 × detected_oriented_count` bytes). Detect retries with smaller candidate capacity after overflow. |
| Host order and selection | `select_keypoints` in `extractor.rs` | CPU sorts canonical `(octave, level, y, x, bin)` order; maps coordinates/scales to source image; applies legacy early-octave cutoff and any contrast/scale ranking/truncation. |
| Descriptor input/compute | `describe`; `shaders/describe.wgsl` | Uploads the source image again and selected `(x,y,sigma,orientation)` f32 records; one 256-thread workgroup per output descriptor computes 4×4×8 values and L2 or L1-root normalization. |
| Final output | `read_bytes`, `extract` | Blocking read of `512 × selected_count` descriptor bytes; host creates returned `Vec<SiftKeypoint>` and one 128-float `Vec` per descriptor. Public return type and ordering are paired vectors. |

### Dispatch and submission accounting

For `I = max(config.intervals, 1)` and `n` active octaves, detection-related dispatch commands in the current source total `n(2I+9)+2`:

- Octave zero: upsample + initial blur pair + `I+2` blur pairs + DoG = `2I+8` dispatches.
- Each later octave: halve + `I+2` blur pairs + DoG = `2I+6` dispatches.
- Each octave also has extrema, orientation-args, and orientation dispatch commands.

For default `I=3` and the 7 octaves used by 3072×2304 and 4096×3072 images: **107 detection-related dispatch commands** in one explicit detection queue submission. Of these, 86 are scale-space/DoG and 21 are detector/orientation commands. The indirect orientation dispatch launches up to one workgroup per detected candidate, while each extrema invocation checks a pixel at each DoG level; those are workgroups/invocations, not extra host dispatch commands. A nonempty extraction adds one descriptor dispatch command.

Successful nonempty extraction uses 5 explicit queue submissions: detection; counter readback copy; keypoint readback copy; descriptor pass; descriptor readback copy. `read_bytes` blocks with `poll(WaitIndefinitely)` and maps for each of the three reads. There is no host readback between scale-space dispatches. Overflow retries add detection and counter-readback submissions/waits.

### Per-image GPU resources

The `storage` helper allocates storage buffers with `STORAGE | COPY_DST | COPY_SRC` and 4-byte rounded sizes (minimum 16 bytes); the indirect args buffer adds `INDIRECT`. Detection parameters use `UNIFORM | COPY_DST`, with 256-byte slots and 64-byte bindings. Readback buffers use `MAP_READ | COPY_DST`. Image, Gaussian, DoG, weights, candidates, counters, args, and descriptors are all f32/u32 storage; there are no intermediate textures.

Let `P_o = w_o h_o` be pixels in octave `o`, `L=I+3` Gaussian levels, `C_o=max(floor(P_o/cand_div),4096)`, and `Kcap=2ΣC_o`. One detection attempt allocates input image `4WH`, weights `4 × taps`, scratch `4P_0`, Gaussian buffers `4P_oL` per octave, DoG buffers `4P_o(L−1)` per octave, candidate buffers `28C_o`, oriented keypoints `36Kcap`, 80-byte counters, 192-byte indirect args, and one uniform parameter buffer. Descriptor setup additionally allocates a second image buffer `4WH`, selected-keypoint buffer `16N`, descriptor output `512N`, and a 32-byte uniform buffer. Each readback creates a fresh staging buffer. No workspace buffers are retained across calls in the current SiftGpu path.

For 3072×2304 input, the first octave is 6144×4608. With seven octaves, default intervals, `cand_div=32`, and the 4,096 candidate minimum, the detection buffers are approximately 2.01 GB decimal (about 1.87 GiB) before staging, allocator overhead, retries, and transient descriptor resources. For 4096×3072 input, the analogous rough allocation is 3.57 GB decimal (about 3.32 GiB). These are estimates from declared full-resolution f32 pyramids and capacities, not measured process/GPU peak-memory readings. The full 4K path's capacity and adapter memory headroom need dedicated L40S and Metal tests before changes.

A successful call allocates approximately 28 detection buffers plus two staging buffers, then four descriptor buffers plus a third staging buffer (about 35 GPU buffer objects, excluding driver internals). The image is uploaded twice. Additional CPU transient allocations build dimensions, kernels, operation vectors/bind groups, parameter vectors, the raw keypoint vector, sorted/selected keypoints, flattened descriptor inputs, mapped byte vectors, and output vectors.

### Output and selection semantics

The library API returns all features unless `max_keypoints` is set; this standalone benchmark defaults to 8,192, matching production settings. Orientation count defaults to unlimited. The GPU's raw orientation counter can exceed the selected output cap: host selection is after detection, before descriptor calculation. It cannot simply truncate orientation append order: atomic append ordering is not canonical and would change the existing CPU-compatible sort/ranking policy. In default non-full-pyramid capped mode, selection retains whole octaves through the first octave that reaches the budget; a later cap can rank by contrast or larger scale. Descriptor order follows the final selected keypoint order.

The CPU SIFT implementation in `crates/vision/src/features/sift.rs` is the reference. Existing GPU extractor tests compare default and capped L1-root extraction on a deterministic small texture. They require ≥98% keypoint pairing and descriptor dot >0.999 for matched features; exact ordering and all output fields are not asserted.

## Profiling gates and limitations

- The optional timestamp-query implementation timestamps the complete detection compute pass and the descriptor compute pass. Device timestamp support is feature-negotiated; when unsupported or not requested, the benchmark still produces CPU wall-time JSONL and `null` stage times. Query results are read once after extraction timing rather than waited on at each intermediate GPU stage. The existing detection readbacks remain blocking as required by host selection.
- No memory-allocation counter or peak GPU memory instrumentation is present; resource object counts and buffer sizes above are source-derived. GPU process memory was not reported as a measured value.
- The full crate test command had 3 matcher test failures on the Apple M1: `matcher_matches_cpu_cross_check`, `u8_matcher_matches_exact_cpu_reference`, and `descriptor_matcher::tests::matches_cpu_brute_force`. The two SIFT-vs-CPU tests passed individually, and all SIFT/VLFeat extraction tests shown in the full run passed. Matcher failures are outside this SIFT extraction audit and should be kept visible as baseline failures, not attributed to future SIFT changes.
- No 4K images or 3,300-image dataset are available in this workspace. No L40S/Vulkan adapter is available. No downstream matching/RANSAC/reconstruction completeness test was run. Therefore Phase 1 is a local reference profile and gate input, not evidence the production target is achieved.
- Timestamp run detail for five 3072×2304 images (Apple M1 Metal, 8,192 features): extraction wall times 699.470, 673.538, 711.726, 729.937, 778.817 ms; detection pass 446.100, 453.295, 460.175, 472.174, 471.822 ms; descriptor pass 39.158, 37.860, 40.532, 41.634, 41.268 ms.

## Bottleneck report / decision gate

The measured stage timings identify scale-space plus detection as the largest GPU compute pass (roughly 0.45–0.47 s of the 0.67–0.78 s extraction wall time on the local 3K run); descriptors take about 38–42 ms for 8,192 features. This makes descriptor-only shader tuning an unlikely first win. However, the current single pass combines the complete 86-command scale-space sequence and detector operations, so these numbers **do not** separate blur/DoG from extrema/orientation, and cannot yet justify a specific kernel rewrite. The two blocking detection readbacks and repeated resource construction remain measured-by-inspection costs, but their wall-time share is not isolated.

Profiling was extended to timestamp scale-space, extrema, orientation, and descriptor in separate ordered passes, without changing the non-profiled path. On the 5-image 3072×2304 M1/Metal sample, medians were approximately 238–263 ms scale-space, 138–151 ms extrema, 29–33 ms orientation, and 37–41 ms descriptors. Orientation therefore was not the major share despite its high per-candidate cost.

### Retained shader change: extrema early rejection

The extrema kernel tests a strict minimum and maximum by examining the 26 neighbors in the adjacent 3×3×3 DoG neighborhood. Once it has observed at least one neighbor greater than or equal to the candidate and one less than or equal, neither strict-extremum hypothesis can recover. It now returns immediately at that point, avoiding remaining storage loads/comparisons for rejected pixels. This preserves the existing strict comparisons and output behavior; a pixel returns only after both boolean predicates are already false.

On 30 South Building 3072×2304 images on Apple M1/Metal at 8,192 output features, timestamped extrema median/p95 fell from 177.659/189.850 ms to 112.278/121.023 ms (about 37% lower median). Scale-space remained about 253 ms median. Timestamped end-to-end and descriptor timings are perturbed and are not the throughput claim. Two non-instrumented runs after the change measured 18.598 s total, 0.622 s median, 0.650 s p95 and 19.266 s total, 0.647 s median, 0.662 s p95. The prior non-instrumented reference run was 20.951 s total, 0.699 s median, 0.749 s p95. Both post-change runs are faster, with end-to-end medians 7.5–11.1% lower; these remain separate runs, so sustained/thermal effects warrant a longer follow-up. Stage comparison supports retaining this safe optimization.

The two focused SIFT-vs-CPU tests passed after the change. Their existing criteria are ≥98% feature pairing and >0.999 descriptor dot products, and the South Building CPU/GPU comparison before this patch showed 99.96–100% pairing and descriptor similarity essentially 1.0. A quality comparison over the 30-image set with CPU enabled remains useful follow-up.

Other shader experiments evaluated and rejected:

- Workgroup-shared-memory tiled separable blur, with both fixed and active halos: CPU/reference feature pairing and descriptor metrics remained unchanged on the five-image test, but timestamped scale-space median rose from about 263 ms reference to 338–346 ms tiled. An uninstrumented 128-image sample measured 91.72 s tiled; a comparable reference sample measured 90.45 s, so no speedup was demonstrated (runs were separate and affected by runtime variability). The experiment was removed.
- Compute the orientation histogram's Gaussian column sum once per 64-thread candidate workgroup instead of identically in every lane. On a five-image timestamp sample, orientation median moved from 33.14 ms to 31.58 ms. But uninstrumented 30-image runs were noisy: reference 20.951 s (median 0.699 s, p95 0.749 s), optimization 20.649 s (median 0.691 s, p95 0.722 s), and repeated optimization 20.887 s (median 0.702 s, p95 0.745 s). This small effect was not repeatable enough to retain; the shader is restored to reference.

A single-variable 8×8 image-kernel workgroup-size test also passed the two focused SIFT-vs-CPU tests but did not show a clear scale-space or end-to-end improvement in the 30-image timestamp sample; it was reverted. A second 32×8 image workgroup layout passed the default differential test, but its 30-image run measured 21.64 s total, 0.723 s median, and 0.772 s p95, worse than the 16×16 reference; it too was reverted. No shader experiment is retained as a claimed optimization.

### Detection readback synchronization experiment

Detection needs host-visible counter values before checking overflow and the active oriented-keypoint count; it then needs only the active oriented-keypoint prefix so the CPU can preserve canonical ordering/ranking before descriptors. The old path used separate counter and keypoint staging allocations, copies, submissions, and waits. The current experiment reuses grow-on-demand staging buffers; it puts the 80-byte counter copy in the detection command submission, then maps/waits once for counts. After reading the exact count, it copies only the active keypoint prefix into a separate reusable staging buffer and performs the second necessary map/wait. This preserves overflow checking and avoids transferring the unused keypoint capacity.

On M1/Metal, two 30-image runs of the combined staging attempt that mapped a packet-capacity-sized buffer regressed badly (24.21 s total, 0.704 s median / 1.237 s p95; an earlier outlier also regressed), so the full packet mapping was rejected. The corrected small-counter plus active-prefix staging version returned to 19.16 and 19.17 s total, 0.640 and 0.636 s median, 0.658 and 0.670 s p95, comparable to the preceding extrema-optimized runs. Therefore these measurements do **not** establish a synchronization speedup; the implementation is retained only as a reduction in per-image staging allocation and one fewer explicit queue submission, while still requiring two host wait points due to CPU ordering/selection semantics.

Five-image CPU/GPU differential after this readback change: 99.96–100% feature pairing, descriptor dot p0.1 0.99999946 and median 1.0. The focused default and capped L1-root GPU tests passed. The active prefix is zero-safe; counter overflow still causes the existing retry. Further wait removal needs device-side canonical selection/ranking plus descriptor input generation (and careful tie preservation), or a redesigned asynchronous API. It is not justified to claim this is already accomplished.

### Async extraction API

`SiftGpu::extract_async` now awaits wgpu map callbacks for counters, active keypoints, and descriptors. A callback-backed future is driven by a helper thread polling the device, waking the executor on completion; awaiting tasks are not parked in `WaitIndefinitely`. The original `extract` remains source-compatible and is implemented as `pollster::block_on(extract_async(...))`. Preparation/encoding, CPU ordering, feature selection, result materialization, and sequential dependencies still execute in the future's polling task; this does not make dependent stages overlap or eliminate the three map points. It allows an async runtime to schedule other work while GPU readbacks are pending.

The new parity test compares keypoints and all descriptor values exactly between sync and async APIs; it passes, as do default and capped L1-root CPU differential tests. Thirty-image M1/Metal run via the sync compatibility wrapper after the conversion: 18.685 s total, 0.622 s median, 0.654 s p95 (comparable to pre-conversion runs around 19.16 s / 0.640 s median). This is a compatibility/per-scheduler behavior change, not a claimed raw throughput gain. A real caller must await `extract_async` on its executor to gain nonblocking behavior; the existing sync benchmark deliberately remains sync.

### GPU utilization and worker-queue experiment

Hardware utilization was sampled from macOS `IOAccelerator` `PerformanceStatistics` (`Device Utilization %`) while the release benchmark ran. `powermetrics` also exposes Apple GPU power/frequency sampling, but requires superuser and was not used. `nvidia-smi` is unavailable on this M1. During one 30-image one-worker run, 893 readings averaged 77.02% device utilization (min/max 0/100%). During a 30-image two-worker run, 617 readings averaged 94.56% (min/max 0/100%). These are OS-provided utilization percentages, sampled by polling `ioreg`; the extrema include idle/transient periods and are less representative than the mean. This is a local utilization indicator, not a cross-platform GPU-busy metric or an NVIDIA L40S measurement.

The benchmark has an opt-in `--workers N` experiment that creates N independent wgpu instances/devices/queues and SIFT extractors, warms each before the timer barrier, then distributes the same sorted image list across a bounded number of host workers. Run it with:

```text
rustup run 1.94.0 cargo run --release -p visloc-sift-gpu --features gpu --example sift_gpu_bench -- --images /Users/marvin/Coding/GitHub/visloc-test/south-building/south-building/images --max-images 30 --no-cpu --jsonl --workers 1
rustup run 1.94.0 cargo run --release -p visloc-sift-gpu --features gpu --example sift_gpu_bench -- --images /Users/marvin/Coding/GitHub/visloc-test/south-building/south-building/images --max-images 30 --no-cpu --jsonl --workers 2
```

The one-worker queue measured 18.544 s wall (1.618 images/s) in one run. Three two-worker runs measured 17.194, 18.349, and 17.130 s wall (1.745, 1.635, and 1.751 images/s); median wall throughput is about 1.745 images/s, 7.9% above that one-worker run. Per-image median latency increased from roughly 0.618 s to about 1.0–1.10 s under two workers. So there is a modest, variable aggregate-throughput opportunity and stronger GPU occupancy, but each image completes slower. The system is already fairly busy with one worker; multi-worker saturation does not multiply throughput.

This experiment uses separate wgpu device contexts per worker, so it duplicates the large scale-space allocations (source-estimated ~2 GB per 3K active image, excluding driver/system allocation behavior) and can quickly pressure the 16 GB unified-memory machine. It is an upper-bound style concurrency probe, not the recommended production scheduler. A bounded multi-image pipeline on one shared device, using independent per-image workspaces and asynchronous completion, may achieve similar utilization with less duplicated device/context overhead; it needs careful memory-budgeting and queue-depth measurement. Current public extractor still serializes each image's candidate readback → CPU selection → descriptor dispatch; async map awaiting frees the host executor thread but doesn't itself enqueue the next image or pipeline phases.

The measurements suggest remaining throughput potential is primarily from hiding host/device gaps and feeding the device continuously, not blindly adding CPU threads. Evaluate a bounded shared-device queue (start at depth 2), keep outputs ordered, measure memory and GPU utilization, and compare total throughput, per-image latency, and quality. On this M1, the simple two-context worker queue only shows a ~8% median throughput gain and is not sufficient evidence for retaining multi-context production behavior.

## End-to-end resource reuse and scale-space fusion follow-up

The standalone SiftGpu path now has a keyed grow-only buffer pool. It retains the image input, Gaussian weights, scratch, per-octave Gaussian/DoG/candidate buffers, counters, indirect args, detection uniforms, selected descriptor-keypoint/output buffers, descriptor uniforms, and descriptor readback buffer across synchronous extractions. Buffer IDs are keyed by role and octave; capacities double when needed. The detection image buffer is reused for descriptor generation, removing the duplicate image buffer allocation and image upload. Counter and active-keypoint staging buffers also persist. Compute pipelines were already retained in SiftGpu.

This is a sequential single-workspace pool. It intentionally does not reuse an allocation while another extraction can be in flight; the current `&mut self` extraction completes all reads before returning. A multi-image queue needs independent workspaces (or a fence-aware pool), bounded depth, and ordered output. The 30-image 3072×2304 run reports **1,929,675,888 bytes** pooled GPU buffer capacity (~1.80 GiB) at the largest capacity reached; descriptor readback and small dedicated staging allocations are excluded from this API estimate. This large footprint matters for two concurrent images and will vary by image dimensions and candidate-capacity retries.

The retained scale-space fusion writes a Gaussian level and its corresponding adjacent DoG plane in the same vertical blur dispatch. It removes the separate DoG read/subtract/write dispatch for each octave: seven fewer dispatch commands out of 107 for the default 7-octave 3K/4K path. The prior Gaussian level is read from the same octave buffer, while the new Gaussian and DoG value are written once. Existing explicit Gaussian and DoG buffers remain the reference-compatible representation; no precision change was made. The first Gaussian level is left on the original blur path because it has no preceding Gaussian level/DoG output.

The separate nearest-neighbor upsample + horizontal-blur fusion passed the small parity tests but did not improve end-to-end timing on M1, so it was removed. A second experiment fused each later-octave halve with its first horizontal blur. It appeared correct on a one-octave image, but a multi-octave diagnostic showed GPU keypoint counts dropping from CPU 87 to GPU 1 at three octaves, so that fusion was reverted without benchmarking. Preserve this negative correctness result; any reattempt must first compare per-octave Gaussian and DoG planes, not only final aggregate matching.

### Measured end-to-end result

All measurements use Rust 1.94.0 release, Apple M1/Metal, South Building 3072×2304 images, 8,192 feature cap, single worker, CPU comparison disabled. Baseline before resource reuse/fusion: 30 images, 19.324 s total, 0.648 s median, 0.676 s p95 (from the warm single-worker benchmark). After buffer pooling and DoG fusion, runs measured 13.534 s total / 0.450 s median / 0.468 s p95, 13.768 s / 0.463 s / 0.482 s, and 14.016 s / 0.470 s / 0.494 s. The first two outputs predate the pooled-memory summary field; the last reports retained pool capacity 1,929,675,888 bytes. This is approximately **27.5–30.6% lower median latency** and **1.38–1.43× aggregate throughput** across separate runs. Per-image feature cap and ordering are unchanged.

The five-image CPU differential after pooling/fusion had feature pairing 99.96–100%, descriptor dot p0.1 0.99999946 and median 1.0. Default, capped L1-root, and async/sync parity tests passed. These tests do not replace a 30-image CPU quality comparison or downstream SfM validation. This measured gain is on Apple M1 only; verify Metal low-memory behavior and Vulkan L40S independently before deployment.

Additional dispatch reductions that remain plausible: fuse octave halving into the first blur (one dispatch per later octave), and consider GPU-side selection/descriptor input generation to reduce CPU ordering and readback waits. Both need dedicated intermediate-value and full-feature differential tests before implementation. Keep an eye on memory: buffer reuse cuts allocation churn but retains the high-water ~1.8 GiB pool, while an in-flight double buffer could roughly double image-dependent workspace memory.

## Phase 2: throughput optimizations and final local results

Date: 2026-10-20  
Toolchain: Rust 1.94.0 release; wgpu 30.0.1; Apple M1 / Metal  
Workload: South Building, 8,192 returned features per image. The 4K rows use
4096x3072 images produced by `sips` upscaling the 3072x2304 source images;
they exercise the 4K buffer sizes and pixel counts but are not native 4K
captures. No L40S/Vulkan adapter was available, so none of these numbers are
an L40S claim.

### Retained changes

1. **Reused high-water buffer pool.** Image, weights, Gaussian/DoG planes,
   candidates, oriented keypoints, counters, indirect args, descriptor input,
   output and staging buffers persist across extractions. Capacities grow
   geometrically. On 3072x2304 the pool reports 1,929,675,888 bytes; on
   4096x3072 it reports 3,423,139,952 bytes.
2. **Fused Gaussian/DoG writing.** `blur_v_dog` writes the new Gaussian plane
   and the adjacent DoG plane in one vertical-blur dispatch, replacing the
   separate DoG pass. The first Gaussian level retains the original path.
3. **Extrema early rejection and unclamped interior reads.** The 3x3x3 test
   exits as soon as both strict-extremum hypotheses are impossible, and the
   interior fast path indexes DoG planes directly instead of clamping every
   neighbor coordinate.
4. **Four output pixels per blur invocation.** `blur_h` computes four adjacent
   columns and `blur_v` / `blur_v_dog` compute four adjacent rows per thread.
   Taps and weight loads are shared across the four outputs; border columns
   and rows keep the original clamped arithmetic. The extraction dispatch
   divides the corresponding workgroup dimension by four. Arithmetic order
   per output is unchanged, so results remain within the existing CPU
   differential tolerance.
5. **Linear-time capped selection.** The old host path sorted every oriented
   keypoint canonically, truncated by whole octaves, mapped every record to a
   `SiftKeypoint`, then sorted all of them by contrast. The new path computes
   the octave cutoff from per-octave counts, runs `select_nth_unstable_by`
   directly on compact raw records, and sorts only the selected 8,192 prefix.
   This changed measured host selection from roughly 65-75 ms to roughly
   4-7 ms per 3K image while preserving the CPU's set and stable ranking
   (canonical order is used as the explicit tie-break).

A shared-memory tiled extrema experiment was measured and rejected: on 30
3072x2304 images its extrema pass was about 145 ms versus about 101 ms for the
global-read version, and end-to-end median rose from about 0.45 s to about
0.51 s. Tiled separable blur had already been rejected earlier for the same
reason on this adapter.

### Final measurements

GPU-only sustained runs, single worker, no CPU comparison:

| Input | Images | Total | Mean | Median | p95 | Throughput | Pool |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3072x2304 baseline `2bdf090` | 30 | 21.629 s | 0.721 s | - | - | 1.39 img/s | - |
| 3072x2304 baseline, 128-image audit | 128 | 90.449 s | 0.707 s | 0.706 s | 0.750 s | 1.42 img/s | - |
| 3072x2304 final | 128 | 39.679 s | 0.310 s | 0.310 s | 0.334 s | 3.23 img/s | 1.93 GB |
| 4096x3072 upscaled final | 30 | 16.459 s | 0.549 s | 0.550 s | 0.578 s | 1.82 img/s | 3.42 GB |
| 4096x3072 upscaled final | 128 | 69.773 s | 0.545 s | 0.536 s | 0.668 s | 1.83 img/s | 3.42 GB |

The 3K sustained median is 2.28x lower than the audit baseline and throughput
is 2.28x higher. Timestamped 3K stage medians after the retained changes were
approximately 123 ms scale space, 102 ms extrema, 40 ms orientation, and
38 ms descriptor (about 318 ms extraction wall median in that instrumented
sample). At 4K the corresponding medians were approximately 217 ms scale
space, 191 ms extrema, 68 ms orientation, and 44 ms descriptor, with the
remaining wall time in host selection, staging readbacks, uploads and command
submission.

The `--workers 2` probe on this M1 did not improve folder throughput over one
worker (2.48 versus 2.69 img/s in the worker-mode control runs; the final
single-context folder run reached 3.23 img/s). Separate wgpu contexts duplicate
the multi-gigabyte pool and contend for the same integrated GPU. The
single-workspace synchronous path remains the recommended local configuration.

CPU/GPU quality on the final 3K code, three South Building images, 8,192
features: 100.00% of CPU keypoints paired, descriptor dot p0.1 = 0.99999958,
median = 1.0. CPU extraction measured about 9.85 s/image versus 0.31 s/image
for sustained GPU-only extraction. The focused CPU differential tests
(default, capped L1-root, unaligned 193x129, full pyramid, larger-scale
preference) and the sync/async and alternating-size pooling tests pass. The
three matcher tests remain the pre-existing M1 failures recorded in Phase 0;
they are outside SIFT extraction.

## Phase 3: candidate pre-capping before orientation

Date: 2026-10-21  
Toolchain: Rust 1.94.0 release; wgpu 30.0.1; Apple M1 / Metal  
Workload: South Building 3072x2304, 8,192 returned features per image.

### Retained change

Detection is now submitted in two phases. Phase A runs scale space and
extrema and copies the per-octave counters back. If the raw candidate count
exceeds the output cap, the host reads the active candidate prefixes and
preselects the candidates that can still reach the final ordering cutoff.
Phase B orients only that compact set. The existing CPU `select_keypoints`
still performs the final f64 sigma mapping, canonical ordering, and exact
`max_keypoints` truncation, so returned coordinates, scales, orientations,
descriptors, and order are unchanged.

The fast path is used only when the compact orientations produce at least
`max_keypoints` records; otherwise detection falls back to orienting every
candidate. On the default capped path the CPU's early octave cutoff is
provably octave 0, so preselection is restricted to octave 0. `full_pyramid`
and `prefer_larger_scale` preselect globally by contrast, or by
sigma-then-contrast, with the canonical `(octave, level, y, x)` tie-break.
All candidates tied at the cutoff primary key are retained.

### Byte-identical validation

All comparisons use the GPU example dump at 8,192 features against fresh
`2bdf090` output. The default five-frame set (`0,10,25,50,100`) matched
byte-for-byte, as did frame 0 under:

- `--aligned-octave0`
- `--subpixel`
- `--max-orientations 2`
- `--l1-root --magnification 3 --max-orientations 2`
- `--prefer-larger-scale --subpixel`

The focused CPU differential tests for default, capped L1-root, unaligned
193x129, full pyramid, larger-scale preference, sync/async parity, and
alternating-size pooling pass. The three matcher failures remain the known
pre-existing M1 failures.

### Final measurements

GPU-only sustained runs, single worker, no CPU comparison:

| Configuration | Images | Total | Mean | Median | p95 | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| Phase 2 (before pre-cap) | 128 | 39.679 s | 0.310 s | 0.310 s | 0.334 s | 3.23 img/s |
| Phase 3 (pre-cap) | 128 | 33.640 s | 0.263 s | 0.263 s | 0.279 s | 3.80 img/s |
| Audit baseline `2bdf090` | 128 | 90.449 s | 0.707 s | 0.706 s | 0.750 s | 1.42 img/s |

The 128-image total is 15.2% lower than Phase 2 and 2.69x faster than the
audit baseline. A 32-image timestamped run measured stage medians of
117.9 ms scale space, 98.4 ms extrema, **1.49 ms orientation**, and 37.1 ms
descriptor. Orientation is no longer a meaningful share of extraction; the
remaining time is dominated by scale space, extrema, and descriptor work plus
host readback/selection gaps.

### Rejected experiments

- **Fused horizontal + vertical blur with a shared-memory tile.** The shader
  produced byte-identical output on the five-frame set, but the scale-space
  pass regressed from about 118 ms to about 8.3 s per image on this M1/Metal
  adapter. The change was removed rather than investigated further.
- **2x2 pixels per extrema invocation.** This reduced workgroup count 4x and
  preserved byte-identical output, but the extrema median rose from 98.4 ms
  to 107.8 ms. It was reverted.

No shader experiment from this phase is retained as a claimed optimization;
only the host-side candidate pre-cap remains. The next plausible gains are
GPU-side preselection/compaction to remove the candidate readback and a
single-sync selection path, but those need dedicated tie-preserving and
full-feature differential tests before implementation.

## Phase 4: deferred later octaves and extrema scheduling

Date: 2026-10-21
Toolchain: Rust 1.94.0 release; wgpu 30.0.1; Apple M1 / Metal
Workload: South Building 3072x2304 (3K) and 4096x3072 upscaled (4K),
8,192 returned features per image. Commits: `c00ca6a` (octave-0-only fast
path), `aed6925` (axis-first extrema), `543a139` (z-loop extrema).

### Retained change: build only octave 0 on the capped early-break path

`c00ca6a` gates the octave-0-only path on all of the following:

```text
!prefer_larger_scale
&& !full_pyramid
&& max_keypoints != usize::MAX
&& num_octaves > 1
```

Phase A then builds only octave 0. The host preselects up to
`max_keypoints` octave-0 candidates plus any ties at the cutoff, orients that
compact set, and returns immediately when the oriented count reaches the
cap. Otherwise it builds octaves 1..n-1 and runs the existing full-orientation
fallback. The default capped CPU selection already stops at octave 0 when
that octave fills the budget, so later octaves cannot affect the returned
prefix. Full-pyramid, larger-scale preference, and unlimited-cap runs keep
the original full scale space and global orientation path.

This removes roughly three quarters of scale-space and extrema pixel work on
the common 3K/4K capped path. It does not add a global contrast pre-cap and
does not change the CPU-side f64 sigma mapping, canonical ordering, tie
handling, or final truncation.

### Retained change: extrema neighbor order and z-loop

`aed6925` visits the center DoG level first and checks same-level axis
neighbors before the vertical neighbor and diagonals. The original
z-major/dy-major nested loops spent longer finding witnesses for both the
min and max rejection conditions. `543a139` then assigns one thread per
interior pixel and walks the overlapping DoG levels in z, reusing the cache
footprint of adjacent planes. Early exits remain per level and the candidate
set is unchanged.

An early axis-first draft omitted the two vertical `(0,0)` neighbors on the
adjacent DoG levels; octave-0 candidates on frame 0 inflated from 135,452 to
173,125. Adding the full 3x3x3 neighborhood back (skipping only the true
center) restored the count before any benchmark or validation result was
accepted.

### Byte-identical validation

All comparisons use `sift_dump_features` and retained `2bdf090` output.
The default five-frame set (`0,10,25,50,100`) matched byte-for-byte after both
`c00ca6a` and the final extrema changes. Frame 0 also matched under:

- `--aligned-octave0`
- `--subpixel`
- `--max-orientations 2`
- `--l1-root --magnification 3 --max-orientations 2`
- `--prefer-larger-scale --subpixel`
- `--full-pyramid` on the default five-frame set
- `--full-pyramid --prefer-larger-scale --subpixel`
- `--keypoints 1000000` (fallback path)
- `--keypoints 18446744073709551615` on the low-texture image (unlimited)
- a low-texture 512x512 fallback image
- 4096x3072 frame 0 against the `2bdf090` binary, default and full pyramid

The focused CPU differential tests for default, capped L1-root, unaligned
193x129, full pyramid, larger-scale preference, sync/async parity, and
alternating-size pooling pass. In the final validation runs the full
`visloc-sift-gpu` suite passed 15/15; the three matcher failures recorded in
Phase 0 did not reproduce.

### Final measurements

GPU-only sustained runs, single worker, no CPU comparison, 3K:

| Configuration | Images | Total | Mean | Median | p95 | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| Phase 3 (pre-cap) | 128 | 33.640 s | 0.263 s | 0.263 s | 0.279 s | 3.80 img/s |
| Octave-0-only fast path | 128 | 26.537 s | 0.207 s | 0.206 s | 0.220 s | 4.82 img/s |
| Final extrema changes | 128 | 25.297 s | 0.198 s | 0.196 s | 0.217 s | 5.06 img/s |

4K upscaled, 32 images:

| Configuration | Total | Median | p95 | Throughput |
|---|---:|---:|---:|---:|
| Phase 3 | - | - | - | 2.21 img/s |
| Octave-0-only fast path | 11.432 s | 0.358 s | 0.378 s | 2.80 img/s |
| Final extrema changes | 10.000 s | 0.312 s | 0.335 s | 3.20 img/s |

Timestamped 3K stage medians (32 images):

| Stage | Phase 3 | Octave-0-only | Final |
|---|---:|---:|---:|
| Scale space | 117.9 ms | 80.4 ms | 79.8 ms |
| Extrema | 98.4 ms | 71.7 ms | **61.2 ms** |
| Orientation | 1.49 ms | 1.49 ms | 1.48 ms |
| Descriptor | 37.1 ms | 35.4 ms | 35.5 ms |

The final 3K total is 24.8% below Phase 3 and 3.58x faster than the
`2bdf090` audit baseline. The final 4K throughput is 45% above the Phase 3
2.21 img/s sample. Across 128 images the 3K total fell from 26.537 s to
25.297 s after the extrema changes; most of that is the axis-first neighbor
order, while the incremental z-loop gain is small and within 3K run-to-run
noise. At 4K the final run measured 12.5% lower total than the octave-0-only
run, but these are separate runs and thermal variance is possible.

### Rejected experiments

- **Descriptor level/spatial sorting.** Keypoints were sorted by sigma
  bucket then spatial tile, with an index map restoring output order.
  Output was byte-identical, but 3K descriptor time was 35.0 ms unsorted
  versus 35.0-35.3 ms sorted (tiles 64/128/256), and 4K was 41.3 ms versus
  41.4 ms. Spatial-first, sigma-only, and space-only keys were also no
  faster. Reverted.
- **Center-level-first with the original nested loops.** Extrema median was
  72.2 ms versus 71.7 ms control. Reverted.
- **32x8 extrema workgroups.** Extrema median rose to 75.6 ms. Reverted.
- **Vertical neighbor first.** Extrema median was 67.9 ms versus 66.6 ms for
  axis-first. Reverted.
- **Axis-first without vertical neighbors.** Incorrect candidate set, as
  noted above. Fixed, not retained as an optimization.

The remaining measured cost is scale space (~80 ms), extrema (~61 ms), and
descriptors (~35 ms). Further extrema work should target the remaining
scale-space/DoG read traffic or a shared-memory design that beats the global
read path; the previously tested shared-memory extrema tile regressed on
this adapter. Descriptor changes need a new hypothesis beyond reordering
keypoints, because locality sorting did not move the pass.
