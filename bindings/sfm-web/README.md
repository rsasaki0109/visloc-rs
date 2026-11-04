# visloc-sfm-web

visloc-rs's photo SfM in the browser: photos to camera poses and a sparse
point cloud, computed in a Web Worker, so the photos never leave the device.
It runs the pieces `gsplat_photos` runs natively: VLFeat-compatible SIFT,
ratio-tested mutual nearest-neighbour matching, two-view geometric
verification, and the COLMAP-port incremental mapper, which reads its input
from memory (`DatabaseCache::from_memory`). Where the browser has WebGPU,
SIFT and matching run on the GPU (`visloc-sift-gpu`'s `VlfeatSiftGpu` and
`GpuMatcher`, through their non-blocking `extract_async` and
`match_pairs_async`); verification and the mapper run in WebAssembly on the
CPU. Without WebGPU, everything runs on the CPU.

The site carries it at <https://rsasaki0109.github.io/visloc-rs/sfm/>
(`.github/workflows/pages.yml` builds it on every push to `main`). Add
`?cpu` to the address to keep SIFT and matching off the GPU.

```bash
./build.sh                      # web/pkg (the wasm and its glue), web/vendor (three.js)
python3 -m http.server -d web   # then open http://localhost:8000
```

`build.sh` needs the `wasm32-unknown-unknown` target and `wasm-bindgen-cli`
at Cargo.lock's `wasm-bindgen` (the lockfile was seeded from
`bindings/wasm`'s, so one CLI serves all three). It copies three.js from
`bindings/wasm/web/vendor`. One thread, so no COOP/COEP headers are needed.

- `web/app.js`: the page. It runs the sample photos (`web/samples/`) or
  photos the visitor picks, shows the cameras and points with three.js as
  they come, and offers the model as PLY (points) and JSON (cameras,
  intrinsics and points).
- `web/worker.js`: decodes each photo with the browser's decoder (which
  applies its EXIF orientation), drives a `Session` (`init_gpu`,
  `add_photo`, `plan_pairs`, `match_some`, `reconstruct`) and posts progress
  back. It sends the GPU 16 pairs per dispatch, so that each dispatch ends
  well inside the OS's GPU watchdog (2 s on Windows) on integrated GPUs too.
- The photos are shrunk to 800 px on the long side, with up to 2000
  keypoints each; each photo is matched with all others up to 40 photos and
  with the next 8 in name order beyond that. All photos must be the
  same size (one camera). Without a known focal length the mapper starts
  from 1.2 times the long side and refines it, with the principal point.
- The GPU matches f32 descriptors, not the u8 ones `gsplat_photos` matches
  natively: in a browser `dot4U8Packed` can be the slower of the two (2.4
  times on an AMD RDNA 2 iGPU).

On the ETH3D courtyard sample (38 photos), headless Edge placed all 38
photos:

| Device | SIFT | Matching and verification | Mapper | In all |
|---|---|---|---|---|
| An 8 GB NVIDIA laptop GPU (WebGPU) | 2.8 s | 3.9 s | 3.1 s | 10 s |
| An AMD RDNA 2 iGPU (WebGPU) | 5.2 s | 14.1 s | 3.6 s | 23 s |
| CPU (`?cpu`) | 17.3 s | 31.9 s | 3.0 s | 52 s |

On a laptop with both, browsers usually take the integrated GPU; the
operating system's graphics settings can give the browser the other one.

Natively, on one thread and the CPU, SIFT takes 14.9 s, matching 13.1 s and
verification 2.7 s:

```bash
cargo run --release --example photos -- web/samples/courtyard --focal 440
```

`cargo test` runs the mapper on a synthetic scene and checks the pair plan.
