# Web (WASM + WebGPU) in-browser SfM — plan / feasibility record

Status: **parked**. The mapper core was proven to compile and run on
`wasm32-unknown-unknown`, but the product is on hold (the only hard
requirement it adds is a static host that can send `COOP`/`COEP` response
headers; GitHub Pages cannot). This document is the design + evidence record
so the work can be resumed without repeating the spike.

Since then, two pieces of it have landed. The browser VIO (`bindings/wasm`)
runs with wasm threads on GitHub Pages through a service worker that adds
the headers (`web/coi-sw.js`), so the hosting question below has an answer.
And the viewer end of the pipeline runs there: `bindings/splat-web` draws
visloc-rs's 3DGS rasterizer on WebGPU, showing a scene that visloc-rs posed
and trained from photos (natively, for now).

Now W1 and W2 (§6) are done too, on the CPU and on one thread:
`bindings/sfm-web` runs SIFT, matching, two-view verification and the
mapper in a page's worker on GitHub Pages (`sfm/`), and shows the cameras
and points with three.js. On the ETH3D courtyard (38 photos at 800 px) a
headless Edge took 50 s and placed all 38. The mapper reads its input from
memory (`DatabaseCache::from_memory`); the §2 changes are in, with a clock
that stands still on wasm (`visloc-slam`'s `time_compat`) instead of
`web-time`.

And the WebGPU half of W3: where the browser has WebGPU, SIFT and matching
run on the GPU (`visloc-sift-gpu`'s ports, through non-blocking
`extract_async` / `match_pairs_async`). The same 38 photos take 10 s on an
8 GB NVIDIA laptop GPU and 23 s on an AMD RDNA 2 iGPU, which is what
browsers on such a laptop take by default (52 s on the CPU). There
verification and the mapper (CPU, one thread) take 7 s of the 10. Still
open: threads, video, focal lengths from EXIF, and the COLMAP text export.

## 0. Product UX (the target)

> No upload, no install, no setup. Drop images or a video, run the pipeline in
> the browser, export, and drop the result into your favourite 3DGS tool.

- 100% client-side: image bytes never leave the machine.
- Input: image set or video (frames extracted in-browser).
- Output: COLMAP text model (`cameras.txt` / `images.txt` / `points3D.txt`)
  that nerfstudio / gsplat / Postshot / etc. can consume directly.
- Scope is deliberately small (tens of images, one scene): browsers cannot
  host the 10k-image / multi-GB native runs.

## 1. What is already proven (spike evidence, 2026-09-16)

Environment: `rustc 1.94.0`, `wasm32-unknown-unknown` target, Node v22.

1. **Compiles**: `visloc-core`, `visloc-vision`, and `visloc-slam` (including
   the whole `colmap_incremental` mapper port) build for
   `wasm32-unknown-unknown` after two small changes (§2).
2. **Runs**: a temporary `#[wasm_bindgen]` cdylib called
   `colmap_incremental::spike::run_synthetic(n)`, which builds the synthetic
   two-camera rig scene (using the crate's `test_support`) and runs the full
   `pipeline::run` (initial-pair search → incremental registration →
   triangulation → local BA → global BA, i.e. GP3P and the native
   Schur-complement solver). Results under Node:

   | frames | registered | wall |
   |---|---|---|
   | 10 | 10 | 370 ms |
   | 20 | 20 | 336 ms |
   | 40 | 40 | 1017 ms |
   | 80 | 80 | 2316 ms |

   Release wasm artifact: ~2.4 MB (before `wasm-opt`).
3. **rayon is not a blocker**: `par_iter()` / `into_par_iter()` degrade to
   single-threaded execution on `wasm32-unknown-unknown` and produce correct
   results. Only a custom `rayon::ThreadPoolBuilder` fails (`build()` returns
   `Err`); the mapper path does not use one. A minimal standalone check
   produced `run_parallel -> 999000`, `run_range -> 999000`,
   `run_threadpool -> -1`.
4. `onnx-inference` is off by default, so no ONNX Runtime / native libraries
   are pulled into the wasm build.
5. `pipeline::run` is filesystem-free: `std::fs` in this crate is confined to
   load/export helpers (`database_cache.rs`, `reconstruction.rs`), so the
   mapper can be driven entirely from in-memory data.

## 2. Required source changes (proven during the spike, reverted when parked)

- **RNG.** `rand`/`getrandom` need the `js` backend on
  `wasm32-unknown-unknown` (uses `crypto.getRandomValues`). Add a
  target-scoped dependency (e.g. in `crates/vision/Cargo.toml`):
  ```toml
  [target.'cfg(target_arch = "wasm32")'.dependencies]
  getrandom = { version = "0.2", features = ["js"] }
  ```
  Feature unification enables it for the whole graph. (RANSAC itself uses
  seeded `StdRng`, so OS entropy is only needed if some path calls it.)
- **Clock.** `std::time::Instant::now()` panics on `wasm32-unknown-unknown`
  with `time not implemented on this platform`. The mapper path has ~11 call
  sites (`bundle_adjustment.rs`, `incremental_triangulator.rs`,
  `pipeline.rs`, `rig_ba_solver.rs`). Add a target-scoped `web-time` dependency
  and a small alias module:
  ```rust
  pub(crate) mod time_compat {
      #[cfg(not(target_arch = "wasm32"))]
      pub(crate) use std::time::{Duration, Instant};
      #[cfg(target_arch = "wasm32")]
      pub(crate) use web_time::{Duration, Instant};
  }
  ```
  and import `Instant` from it instead of `std::time`.
- Do **not** use custom rayon thread pools in the wasm path.

## 3. Hosting: COOP/COEP is mandatory for threads

Prior art `https://offlinetools.io/colmap-landing/` (an unofficial browser
port of COLMAP) is **not** single-threaded: its `app.js` refuses to start
without `window.crossOriginIsolated` and `SharedArrayBuffer`, requires
COOP/COEP headers for "WebAssembly threads", runs the engine in a `Worker`,
derives thread counts from `navigator.hardwareConcurrency`, and uses WebGPU
for SIFT. It is served from Cloudflare with
`cross-origin-opener-policy: same-origin` and
`cross-origin-embedder-policy: require-corp`.

Consequences:

| host | COOP/COEP | wasm threads | verdict |
|---|---|---|---|
| GitHub Pages | no custom headers | no | serial only |
| Cloudflare Pages / Netlify / Vercel | `_headers` / `netlify.toml` / `vercel.json` | yes | **recommended** |
| GitHub Pages + `coi-serviceworker` | injected by a SW | yes (hacky) | fallback |

Parked decision: use a header-capable static host (Cloudflare Pages is the
cheapest equivalent to the prior art). A single-threaded GitHub Pages build
remains possible as a degraded mode but severely limits image count.

## 4. Architecture

```text
browser (cross-origin isolated)
├── UI / viewer (TS + three.js or a wgpu renderer)
├── Web Worker  ── owns the wasm engine (SharedArrayBuffer, threads)
│   ├── feature extraction   (WebGPU compute; fallback: wasm SIFT, serial)
│   ├── matching + ratio + geometric verification (WebGPU / wasm)
│   └── visloc mapper (wasm): DatabaseCache -> pipeline::run -> model
└── main thread ── image decode (createImageBitmap), video frames (WebCodecs),
                   COLMAP text export as strings -> download / File System API
```

- New wasm-facing API (e.g. `crates/visloc-wasm`): `add_image(keypoints,
  descriptors)`, `add_matches(pair, correspondences)`, `run(options)`,
  `export_colmap() -> String` (do **not** go through `std::fs`), plus progress
  callbacks.
- Feature extraction/matching must be built for the browser (the native
  pipeline consumes pre-extracted features + a verified pair graph). Options:
  1. port visloc's own SIFT + matcher to wasm/WebGPU (CPU serial first),
  2. use ONNX Runtime Web (WebGPU EP) with the exported SuperPoint/LightGlue
     models the repo already ships tooling for.
- Export: add a string-returning variant of `Reconstruction::export_colmap_text`
  (the current one writes files) so results can be zipped in-browser.
- Video: `WebCodecs` `VideoDecoder` (or `<video>` + `canvas`) to sample frames
  at a chosen stride.

## 5. Performance hypothesis (to be measured, not assumed)

Claim under test: with wasm threads + WebGPU, the browser can beat
single-threaded native, and can approach multi-threaded native CPU on the
regular stages. It will not beat native + rayon + SIMD + (CUDA) on the
sequential mapper.

Amdahl breakdown of this repo's mapper: feature extraction/matching are
embarrassingly parallel (WebGPU win), BA is parallel/regular (WebGPU win),
but next-image selection + registration + triangulation scheduling are
sequential (native/CPU-bound). Small BAs are launch-overhead-bound and should
stay on the CPU.

Minimum microbenchmarks before any performance claim:

1. One fixed BA problem: native (rayon, all cores) vs wasm serial vs (later)
   WebGPU.
2. Feature extraction + matching: native CPU vs WebGPU, same images.
3. Whole-pipeline wall with a per-stage breakdown (extraction / matching /
   registration / BA) so the sequential floor is known.

## 6. Milestones

- **W1 — engine in a worker.** `crates/visloc-wasm` (wasm-bindgen) + `_headers`
  COOP/COEP on Cloudflare Pages; run the synthetic pipeline in a browser
  worker; threads enabled and thread count reported.
- **W2 — real images, CPU.** In-browser SIFT (wasm) + matcher + verification
  feeding `DatabaseCache`; SfM on ~20–50 images; string-based COLMAP export.
- **W3 — WebGPU.** Move feature extraction/matching to WebGPU compute; measure
  §5.1/§5.2.
- **W4 — UX.** drag-and-drop, video frame extraction, viewer, export to 3DGS
  tools, `wasm-opt` size reduction.

## 7. Open questions / risks

- Feature extraction parity with the COLMAP SIFT the native benchmarks use
  (keypoint space / thresholds) — the native parity work assumes COLMAP-style
  SIFT; a different extractor changes the graph.
- Memory ceiling: the native 10k run peaks ~4.8 GB; browser tabs cannot. Hard
  cap the image count and stream/evict data.
- Determinism: the port claims byte-identical repeats natively; wasm floats
  are IEEE-754 and should match, but thread scheduling must not affect the
  deterministic reductions (the port already restricts to geometry-indexed
  parallel iterators).
- Bundle size and first-load time; `wasm-opt -Oz`, lazy-load the engine.

## 8. Not done (parked)

- No committed wasm scaffolding; the `spike` feature / `test_support`
  exposure / `time_compat` alias and the target-scoped `getrandom`/`web-time`
  changes were reverted when parking. Re-apply per §2 to resume.
- No browser feature extractor/matcher, no string export, no UI, no host
  config. README is intentionally unchanged.

(Both since superseded by `bindings/sfm-web`, except the string export; see
the status at the top.)
