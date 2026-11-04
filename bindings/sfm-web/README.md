# visloc-sfm-web

visloc-rs's photo SfM in the browser: photos to camera poses and a sparse
point cloud, computed in a Web Worker in WebAssembly, so the photos never
leave the device. It runs the pieces `gsplat_photos --cpu` runs natively:
VLFeat-compatible SIFT, ratio-tested mutual nearest-neighbour matching,
two-view geometric verification, and the COLMAP-port incremental mapper,
which reads its input from memory (`DatabaseCache::from_memory`).

The site carries it at <https://rsasaki0109.github.io/visloc-rs/sfm/>
(`.github/workflows/pages.yml` builds it on every push to `main`).

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
- `web/worker.js`: drives a `Session` (`add_photo`, `plan_pairs`,
  `match_some`, `reconstruct`) and posts progress back.
- The photos are shrunk to 800 px on the long side, with up to 2000
  keypoints each; each photo is matched with all others up to 40 photos and
  with the next 8 in name order beyond that. All photos must be the
  same size (one camera). Without a known focal length the mapper starts
  from 1.2 times the long side and refines it, with the principal point.

On the ETH3D courtyard sample (38 photos) a headless Edge took 50 s and
placed all 38 photos; natively on one thread the same run takes 31 s, most
of it SIFT and matching:

```bash
cargo run --release --example photos -- web/samples/courtyard --focal 440
```

`cargo test` runs the mapper on a synthetic scene and checks the pair plan.
