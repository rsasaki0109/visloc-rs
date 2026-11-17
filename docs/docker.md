# Container image

Each release's command-line tools also come as a container image:

```sh
docker pull ghcr.io/rsasaki0109/visloc-rs:latest     # or :0.4, :0.4.0
```

It holds the same tools as the release archives, on Debian 12 (x86_64):

| Tool | What it does |
| --- | --- |
| `gsplat_photos` | Photos → SfM (COLMAP model), 3D Gaussian splats and a mesh ([photos_to_splat.md](photos_to_splat.md)) |
| `visloc_mvs` | Dense multi-view stereo from a COLMAP model ([mvs.md](mvs.md)) |
| `colmap_incremental_mapper` | The COLMAP-port incremental mapper |
| `visloc_web` | Browser UI for `gsplat_photos` |

Without a command the container prints `gsplat_photos --help`.

## Photos to a splat

Mount a folder at `/data` and point the tools at it:

```sh
docker run --rm -v "$PWD:/data" ghcr.io/rsasaki0109/visloc-rs \
  gsplat_photos --images /data/photos --out /data/run
```

- In Windows PowerShell, write the mount as `-v "${PWD}:/data"`.
- On Linux, add `--user "$(id -u):$(id -g)"` so the outputs are yours
  rather than root's.

## The GPU

The GPU paths use wgpu through Vulkan.

- **Linux with an NVIDIA GPU:** install the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  and add `--gpus all`. The toolkit mounts NVIDIA's Vulkan driver into the
  container, which the image allows (`NVIDIA_DRIVER_CAPABILITIES` includes
  `graphics`).
- **Everywhere else**, including Docker Desktop on Windows and macOS, the
  container sees no GPU. wgpu then runs on Mesa's software Vulkan device
  (llvmpipe) on the CPU: correct, but slow. On those machines the native
  binaries from the
  [release page](https://github.com/rsasaki0109/visloc-rs/releases) use the
  GPU (DirectX 12, Vulkan or Metal) and are much faster.

On the CPU, the 38 courtyard photos in
`bindings/sfm-web/web/samples/courtyard` (800×534) took 300 s on 4 cores,
with `--max-size 400 --steps 300`:

- SIFT: 23 s.
- SfM: 78 s, all 38 images registered.
- Training: 212 s.
- The mesh: 9 s.

## Browser UI

`visloc_web` listens on 127.0.0.1 by default. In a container, bind it to
every interface and publish the port:

```sh
docker run --rm -p 127.0.0.1:8080:8080 -v "$PWD:/data" ghcr.io/rsasaki0109/visloc-rs \
  visloc_web --host 0.0.0.0
```

Then open <http://127.0.0.1:8080> and choose folders under `/data`.
`-p 127.0.0.1:8080:8080` keeps the port on this machine, because whoever
can reach it can start runs and read their outputs.

## Building the image

```sh
docker build -t visloc-rs .
```

The `Dockerfile` builds the four tools in `rust:1.94.0-slim-bookworm` and
copies them into `debian:bookworm-slim` with the Vulkan loader and Mesa's
drivers (523 MB). The release workflow builds it for every `v*` tag and
pushes `ghcr.io/rsasaki0109/visloc-rs:<version>` and `:<major>.<minor>`. It
also pushes `:latest`, but only when the tag is the newest version tag
([publishing.md](publishing.md)).
