# Single-binary deep stereo SLAM — in-process front-end, end-to-end

The [SuperPoint](superpoint_onnx_cuda_benchmark.md) and
[LightGlue](lightglue_onnx_benchmark.md) ONNX benchmarks measured the deep
front-end in isolation. This one wires it into the **actual SLAM pipeline**: the
stereo-VO example (`stereo_vo_external_deep_files`) gains an `--in-process-onnx`
mode that computes SuperPoint + LightGlue per frame via ONNX Runtime, in place
of reading the pre-exported `--features-dir`. Everything downstream — windowed
online BA, loop closure, pose-graph optimization — is byte-for-byte the same
code path; only where the features and matches come from changes.

The result is a **single Rust binary** that runs a full deep-frontend stereo
SLAM from raw rectified images — no Python, no PyTorch, no ~30 GB on-disk feature
dump.

## Result — EuRoC MH_03, full 2700 frames, identical VO config

Both runs use the same loop-closure stack (`--online-ba --online-ba-window 10
--online-ba-history 20 --loop-closure --loop-min-frame-gap 200 --loop-two-view-ba
--loop-edge-information`), same machine (RTX 4070 Ti SUPER), scored with the same
`evo_ape` against the timestamped Vicon/Leica ground truth.

| front-end | dependency | wall-clock | verified loops | ATE SE(3) | ATE Sim(3) |
|---|---|---|---|---|---|
| **in-process ONNX** (SuperPoint + LightGlue, GPU) | **single Rust binary** | **3 min 19 s** | 306 | **0.051 m** | 0.047 m |
| file-based pre-export | Python + PyTorch + ~30 GB feature dump | 4 min 49 s | 319 | 0.066 m | 0.057 m |

The in-process pipeline is **1.45× faster end-to-end *and* at least as accurate**
— it computes features on the GPU faster than the file-based path reads the
pre-exported features back from disk, and it drops the Python export stage (and
its 30 GB) entirely.

The two are **not** bit-identical: the file-based features were exported by a
separate Python SuperPoint pass, so the keypoint *sets* differ slightly (the
ONNX export keeps the top-1500 above a 0.005 score gate). Given the *same*
features, the ONNX LightGlue matches are bit-identical to the Python reference
([1500/1500 indices agree](lightglue_onnx_benchmark.md#parity--bit-identical-to-the-python-reference));
the small ATE difference here is the front-end's keypoint selection, not the
matcher. Both land in the same accuracy regime (within ~2.4× of ORB-SLAM3 on
this flight), confirming the in-process front-end drives real SLAM at the
file-based path's accuracy.

## Generalizes to KITTI — seq00, full 4541 frames

The same single binary, on the KITTI seq00 driving sequence (1241×376, exported a
LightGlue model at that resolution; SuperPoint is resolution-dynamic), with the
same loop-closure config as the
[published KITTI loop-closure benchmark](kitti_loop_closure_benchmark.md):

| front-end | verified loops | ATE Sim(3) |
|---|---|---|
| **in-process ONNX** (single binary) | 73 | **2.18 m** |
| file-based pre-export (same machine, same config) | 35 | 2.49 m |
| file-based pre-export (published benchmark) | — | 2.57 m |

The in-process front-end lands **2.18 m Sim(3) ATE on the full 4541-frame
seq00**. Re-running the file-based path on the *same machine* with the *same*
loop-closure config (the published-benchmark settings, scored with the same
`evo_ape -as`) gives 2.49 m — so the in-process binary is **~12 % better, not
merely on par**, and consistent with the 2.57 m the published benchmark records.

The edge is a real front-end difference, not noise: the in-process SuperPoint
(top-1500 above a 0.005 score gate) selects a slightly different keypoint set
than the pre-exported features (top-2048), and on this sequence that set yields
**more verified loops (73 vs 35)** — more loop constraints feeding the GNC
pose-graph optimization, hence a tighter trajectory. The result confirms the
in-process front-end generalizes from the EuRoC MAV flight to the KITTI car
sequence at, or above, the file-based path's accuracy — from a single binary
with no Python and no 42 GB feature dump.

## Reproduce

```sh
# 1. Export the two ONNX models (once).
scripts/export_superpoint_onnx.py --out models/superpoint_1500.onnx
scripts/export_lightglue_onnx.py  --out models/lightglue.onnx

# 2. Run single-binary deep SLAM on the rectified stereo images.
#    (Build with --features "image-io onnx-cuda"; the CUDA-provider runtime
#     setup is the same as the SuperPoint benchmark runner — provider shared
#     libs next to the binary, cuDNN on LD_LIBRARY_PATH.)
target/release/examples/stereo_vo_external_deep_files \
    --in-process-onnx --images-dir /tmp/MH_03_rect \
    --superpoint-model models/superpoint_1500.onnx \
    --lightglue-model models/lightglue.onnx \
    --frames 2700 --calib /tmp/MH_03_rect/calib.txt \
    --projection-left P0 --projection-right P1 --width 752 --height 480 \
    --online-ba --online-ba-window 10 --online-ba-history 20 \
    --loop-closure --loop-min-frame-gap 200 --loop-two-view-ba --loop-edge-information \
    --out-dir target/inproc_mh03

# 3. Score (KITTI 3x4 vo_poses -> TUM via timestamps -> evo).
python3 scripts/kitti_poses_to_tum.py target/inproc_mh03/vo_poses.txt \
    /tmp/MH_03_rect/timestamps.txt target/inproc_mh03/inproc.tum
evo_ape tum /tmp/MH_03_gt.tum target/inproc_mh03/inproc.tum -a    # SE(3)
```

`--in-process-onnx` takes `--images-dir` (with `--left-subdir`/`--right-subdir`,
default `image_0`/`image_1`), `--superpoint-model`, `--lightglue-model`,
`--onnx-cpu` (force CPU), and `--onnx-max-keypoints`. The match confidence
carried in `DescriptorMatch::confidence` is the LightGlue score, so the same
`--min-stereo-confidence` / `--min-temporal-confidence` gates apply to both
front-ends. Without the `onnx-inference`+`image-io` features the flag errors and
the binary stays the file-based reader (the default build is unchanged).
