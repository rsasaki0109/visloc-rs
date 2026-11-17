# The release's command-line tools as a container image:
#   gsplat_photos              photos -> SfM (COLMAP model), 3D Gaussian splats and a mesh
#   visloc_mvs                 dense multi-view stereo from a COLMAP model
#   colmap_incremental_mapper  the COLMAP-port incremental mapper
#   visloc_web                 browser UI for gsplat_photos (run with --host 0.0.0.0)
#
# Published as ghcr.io/rsasaki0109/visloc-rs by .github/workflows/release.yml.
# Build locally with `docker build -t visloc-rs .`; see docs/docker.md.

FROM rust:1.94.0-slim-bookworm AS build
# The image's toolchain is the one rust-toolchain.toml pins; skip the
# toolchain file so rustup does not fetch rustfmt and clippy for a build.
ENV RUSTUP_TOOLCHAIN=1.94.0
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p visloc-gsplat-train --features "gpu euroc" --example gsplat_photos \
 && cargo build --release --locked -p visloc-mvs --features gpu --example mvs \
 && cargo build --release --locked -p visloc-rs --example colmap_incremental_mapper --example visloc_web \
 && mkdir /out \
 && cp target/release/examples/gsplat_photos target/release/examples/colmap_incremental_mapper \
       target/release/examples/visloc_web /out/ \
 && cp target/release/examples/mvs /out/visloc_mvs

FROM debian:bookworm-slim
# The GPU paths use wgpu through Vulkan: libvulkan1 is the loader, and Mesa's
# lavapipe is a software Vulkan device for machines without a GPU in the
# container. With the NVIDIA Container Toolkit (`docker run --gpus all`) the
# NVIDIA Vulkan driver is mounted in and used instead.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers ca-certificates \
 && rm -rf /var/lib/apt/lists/*
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility
# Mesa's Vulkan drivers warn on every start without a runtime directory.
ENV XDG_RUNTIME_DIR=/tmp
COPY --from=build /out/ /usr/local/bin/
COPY LICENSE-MIT LICENSE-APACHE /usr/share/doc/visloc-rs/
WORKDIR /data
CMD ["gsplat_photos", "--help"]
