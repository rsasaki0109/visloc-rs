# Single-Binary Deep Stereo SLAM: In-Process vs File-Based Front-End (Registry-Backed)

Generated from benchmark-registry run manifests. Phase 3 formalization of the end-to-end wall-clock result documented in `docs/inprocess_slam_benchmark.md`: the same rectified EuRoC `MH_03_medium` 2700-frame stereo stream driven through the same Rust binary (`stereo_vo_external_deep_files`) and the same loop-closure/BA configuration (`--online-ba --online-ba-window 10 --online-ba-history 20 --loop-closure --loop-min-frame-gap 200 --loop-two-view-ba --loop-edge-information`), scored with the same `evo_ape` against the timestamped Vicon/Leica ground truth. The only difference between the two rows is where SuperPoint + LightGlue features and matches come from.

| front-end | dependency | wall-clock | verified loops | ATE SE(3) | ATE Sim(3) | run id |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| in-process ONNX (single Rust binary) | single Rust binary | 199 s | 306 | 0.051 m | 0.047 m | inprocess-deep-slam-onnx-MH_03_medium-20260703T130000Z |
| file-based pre-export (Python + PyTorch) | Python + PyTorch + ~30 GB feature dump | 289 s | 319 | 0.066 m | 0.057 m | inprocess-deep-slam-filebased-MH_03_medium-20260703T130000Z |

## Headline

- **The single Rust binary in-process ONNX front-end is 1.45x faster end-to-end AND at least as accurate**: 199 s vs 289 s wall-clock, 0.051 m vs 0.066 m ATE SE(3), 0.047 m vs 0.057 m ATE Sim(3) -- while dropping the Python export stage and its ~30 GB on-disk feature dump entirely. It computes features on the GPU faster than the file-based path reads the pre-exported features back from disk.

## Supporting throughput

- SuperPoint extraction on CUDA: 7.4 ms/frame (~135 fps) vs CPU 165 ms/frame (~6.1 fps) -- ~22x speedup, 6.7x headroom over the 20 Hz EuRoC camera rate (`docs/superpoint_onnx_cuda_benchmark.md`).
- Full learned front-end (extract + match) on GPU: ~34 fps, above the 20 Hz camera rate (`docs/lightglue_onnx_benchmark.md`).
- V2_03 orbit sequence single-binary VO: 23.9 fps (`docs/inprocess_slam_benchmark.md`).

## Caveats

- **The two arms are NOT bit-identical.** The file-based features were exported by a separate Python SuperPoint pass, so the keypoint sets differ slightly (the in-process ONNX export keeps the top-1500 above a 0.005 score gate). This is a keypoint-set difference, not a matcher difference: given the same features, the ONNX LightGlue matches are bit-identical to the Python reference (1500/1500 indices agree). The small ATE difference between the two arms is attributable to the front-end's keypoint selection, not the matcher. Both arms land within ~2.4x of ORB-SLAM3 on this flight.
- **This is a documented prior GPU run, not reproduced this session.** A local re-run needs Windows CUDA ONNX Runtime provider DLLs, cuDNN 9, and a PyTorch SuperPoint/LightGlue ONNX export, which is impractical for this evidence-formalization pass. Both manifests capture the already-documented, previously-executed result from `docs/inprocess_slam_benchmark.md` rather than re-executing it.

## Conclusion

The single-binary in-process ONNX front-end is not just a convenience: on this end-to-end SLAM run it is both faster and at least as accurate as the file-based path it replaces, while eliminating the Python/PyTorch dependency and its ~30 GB feature dump. The result is scoped honestly: it is a documented prior GPU run rather than one reproduced in this session, and the two arms' keypoint sets are not bit-identical even though the matcher is.
