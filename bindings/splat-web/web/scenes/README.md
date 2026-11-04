# Scenes

Each scene is visloc-rs's own: posed by its SfM and trained by its 3DGS
trainer from the photos alone (only the dataset's focal length is given),
then written for the web by `gsplat_export_web` (`<scene>.splat`, the most
important gaussians first, and the capture viewpoints in
`<scene>.cameras.json`).

| Scene | Photos | SfM | Training |
|---|---|---|---|
| `courtyard` | ETH3D high-resolution multi-view, *courtyard*: 38 DSLR photos (undistorted), at 1600 px | `gsplat_photos_cuda` (CPU SIFT, CUDA matching, the COLMAP-port mapper; `--focal 880`): 38/38 registered | `gsplat_train_cuda --steps 30000 --max-gaussians 1500000` |

## License

The scenes are derived from their datasets and keep their licenses, not the
repository's.

- `courtyard`: ETH3D, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  T. Schöps, J. L. Schönberger, S. Galliani, T. Sattler, K. Schindler,
  M. Pollefeys and A. Geiger, "A Multi-View Stereo Benchmark with
  High-Resolution Images and Multi-Camera Videos", CVPR 2017.
  <https://www.eth3d.net>
