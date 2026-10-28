# 2D object detection and tracking

`visloc-objects` (`pipelines/objects`, also `visloc_rs::objects`) finds and
tracks objects in image space. It is Phase 1 of the
[metric-object plan](object_metric_slam_plan.md). The later phases turn
these tracks into metric 3D objects with the VIO.

- **Detection:** `ObjectDetector` is the boundary.
  - `YoloxOnnxDetector` runs an official YOLOX model with ONNX Runtime
    (feature `onnx-inference`).
  - `ReplayDetector` returns stored boxes: ground truth, a detector's saved
    output, or synthetic boxes in tests.
- **Tracking:** `ByteTracker` is ByteTrack with class-aware association; see
  the crate documentation.
- **Scoring:** `MotAccumulator` computes CLEAR MOT (MOTA, ID switches,
  misses, false positives, mean IoU).

## Examples

- `cargo run --example track_objects_dummy` tracks three synthetic people.
  Two of them cross, and the hidden one's score drops to 0.3. The example
  checks that nothing is missed or switched (MOTA 1.0) and writes
  MOTChallenge-format tracks. CI runs it (`scripts/run_examples.sh`).
- `yolox_track_demo` runs YOLOX and ByteTrack on a directory of frames. It
  writes `tracks.txt` (`frame,id,x,y,w,h,score,class,-1,-1`) and, with
  `--draw`, every frame with its track boxes and IDs:

  ```sh
  curl -LO https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
  ORT_DYLIB_PATH=/path/to/libonnxruntime.so cargo run --release \
    --features image-io,onnx-inference --example yolox_track_demo -- \
    --model yolox_s.onnx --images frames/ --out-dir target/yolox_track --draw
  ```

  For `yolox_nano` and `yolox_tiny`, add `--input-size 416`. `ort` loads ONNX
  Runtime at run time: set `ORT_DYLIB_PATH`, for example to the
  `libonnxruntime.so` that the `onnxruntime` Python wheel ships.

## YOLOX

The models are Apache-2.0 and are not bundled. The two that were checked are:

| Model | Input | sha256 |
| --- | --- | --- |
| `yolox_nano.onnx` | 416 × 416 | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` |
| `yolox_s.onnx` | 640 × 640 | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` |

Both come from release `0.1.1rc0`. Pre- and post-processing are in
`visloc_objects::yolox` and follow YOLOX's `demo/ONNXRuntime`:

- **Pre-processing:** a letterbox (aspect ratio kept, top left, grey 114),
  BGR, values 0–255.
- **Post-processing:** decode the grid, score = objectness × class
  probability, then per-class NMS at IoU 0.45.

The score threshold is 0.1, low on purpose, because ByteTrack's second
association uses the low-score boxes.

The results match a NumPy re-implementation of that pipeline running on the
Python `onnxruntime` 1.29, on YOLOX's `assets/dog.jpg` (decoded once to PNG,
so both sides see the same pixels):

| Model | Detections (Rust / Python) | Largest box difference | Largest score difference |
| --- | --- | --- | --- |
| yolox_s | 7 / 7 | 4.8e-4 px | 4.0e-7 |
| yolox_nano | 8 / 8 | 5.0e-4 px | 4.0e-7 |

The ignored test `pipelines/objects/tests/yolox_onnx.rs` runs the detector on
that image. Its header has the command, and it checks that the dog, the
bicycle and the vehicle are found with a score above 0.5. On 40 frames panned
across the same image, `yolox_track_demo` with `yolox_s` keeps one ID each for
the bicycle, the dog and the car through all 40 frames, with no ID switches.
YOLOX also reports the car as a truck in 34 frames, as a separate track
because association is class-aware. On the development container's CPU,
detection takes about 85 ms per frame and tracking 0.03 ms.
