# visloc-splat-web

visloc-rs's 3D Gaussian Splatting rasterizer (`visloc-gsplat-render`) in the
browser: WebGPU through wgpu, drawn into a canvas. The page gives it a scene
(`.splat` bytes) and a camera; `Renderer::render_async` renders without
blocking (it awaits the compaction counts instead of polling for them) and a
full-screen pass draws the image from the GPU buffer, so the frame never
comes back to the CPU.

The site carries it at <https://rsasaki0109.github.io/visloc-rs/splat/>
(`.github/workflows/pages.yml` builds it on every push to `main`).

```bash
./build.sh                      # web/pkg: the wasm and its wasm-bindgen glue
python3 -m http.server -d web   # then open http://localhost:8000
```

`build.sh` needs the `wasm32-unknown-unknown` target and `wasm-bindgen-cli`
at Cargo.lock's `wasm-bindgen` (the lockfile was seeded from
`bindings/wasm`'s, so one CLI serves both). No threads, so no COOP/COEP
headers are needed.

- `web/viewer.js`: the orbit camera (drag, right-drag or shift to pan,
  scroll or pinch), the scene list (`web/scenes.json`), and until the first
  drag a glide through the capture viewpoints (`gsplat_export_web
  --cameras`), where a splat trained from photos looks right.
- `web/scenes/`: the scenes, made by visloc-rs from photos (see its README
  for how, and for their licenses).

`cargo test` checks the camera maths and the presenting pass natively (the
GPU part skips without an adapter).
