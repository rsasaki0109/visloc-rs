# Photos to splat and mesh

`gsplat_photos` turns a folder of photos into camera poses, a trained 3D
Gaussian Splatting scene and a coloured mesh, in Rust + wgpu with no COLMAP
or Python. This page holds the details that used to live in the README.

```bash
cargo run --release -p visloc-gsplat-train --features gpu,euroc --example gsplat_photos -- \
  --images /path/to/my_photos --out /path/to/runs/my-scene
```

## From a web browser

The same pipeline can also be run from a web browser. A small local server
(`visloc_web`, standard library only) lets you pick the photo folder, start a
run, and follow its progress and log. When the run ends you can turn the
recovered cameras and points in 3D and download `scene.ply`, `mesh.ply` and
the COLMAP model. It can also open an existing output folder just to view it.

```bash
cargo build --release -p visloc-gsplat-train --features gpu,euroc --example gsplat_photos
cargo run --release --example visloc_web -- --port 8080   # then open http://127.0.0.1:8080
```

The server listens on 127.0.0.1 only. Pass `--host` to serve on another
address, but anyone who can reach the server can start runs and download run
outputs.

## The README's GIFs

![south-building: each raw input photo next to the trained Gaussian splat and the extracted mesh rendered from the same recovered pose](assets/photos_to_mesh.gif)

**south-building.** The 128 raw JPGs, each next to the trained Gaussian splat and the extracted mesh rendered from that photo's own recovered pose. Every image registered, focal refined from EXIF 796 px to 847.0 px (COLMAP: 847.2), held-out PSNR 22.77 at 30k steps, 37 min end to end ([GIF script](../scripts/make_photos_demo_gif.py)).

**SmallCity (the README's orbiting hero).** A whole city block: the 5,822 photos of the Hierarchical 3DGS SmallCity capture (six helmet-mounted GoPros on a bicycle, four passes around the block; the release ships them undistorted to one pinhole camera with people and cars blacked out, and its focal length is used as the prior). `gsplat_photos` registered 5,583 / 5,822 photos with 957,418 points (window + VLAD retrieval matching, `--retrieval 30`), trained a 4.7M-gaussian splat to held-out PSNR 24.09 over 698 views (`--mask-black 12` leaves the blacked-out pixels out of the loss and the PSNR) and fused a 5.9M-triangle mesh: 2.5 h end to end on a Google Colab A100 (SfM 82 min, training 30 min, mesh 34 min; [log](media/hero/small_city_gsplat_photos.log)), half the 5 h of the same job before this release's matching, mapper, bundle-adjustment and TSDF speed-ups.

Replaying the current mapper on `gsplat_photos`' matches registers 5,816 photos, and against the release's own metric COLMAP model (a reference, not ground truth; [`scripts/score_smallcity_reference.py`](../scripts/score_smallcity_reference.py), after a Sim(3) fit) the camera centres land within a median 0.15 m, 0.16 m and 0.65 m for mapper seeds 0, 1 and 2 (p90 0.28 m, 0.35 m and 1.52 m); seed 2 keeps a 4% scale drift between the four passes.

Trained from street level, the road is thin splats that turn opaque only at grazing angles, so the orbit shows the street band around the camera path at no less than 70% opacity (`--ground-min-alpha 0.7`); that is a display choice, not the trained scene. [`scripts/colab/hero_pipeline.py`](../scripts/colab/hero_pipeline.py) reproduces it on a Colab GPU (from the terminal with the Colab CLI, or via [the notebook](../scripts/colab/readme_hero_courthouse.ipynb)) and [`scripts/make_readme_hero.py`](../scripts/make_readme_hero.py) renders the orbit; the previous hero, Tanks and Temples Courthouse (553 frames, 34 min on an L4), is kept in [its log](media/hero/courthouse_gsplat_photos.log).

To reproduce the south-building run and its GIF, use the raw `images/` of COLMAP's
south-building dataset. `gsplat_photos` prints the focal refinement and
the held-out PSNR:

```bash
cargo run --release -p visloc-gsplat-train --features gpu,euroc --example gsplat_photos -- \
  --images <south-building>/images --out runs/sb --max-size 1024 --steps 30000 --normal-weight 0.005
cargo run --release -p visloc-gsplat-train --features gpu --example gsplat_eval -- \
  --ply runs/sb/scene.ply --data runs/sb --eval-every 1 --save-dir runs/sb/renders
python scripts/make_photos_demo_gif.py --run runs/sb --out photos_to_mesh.gif --every 3 --panel-width 300 --fps 6
```

## The pipeline

The pipeline has three stages:

1. **SfM.** Takes the focal length from EXIF, then runs SIFT and matching
   on the GPU (`--cpu-sift` extracts on the CPU), two-view verification
   and the COLMAP-port mapper (the configuration that beats COLMAP on EuRoC in
   the README). A final bundle adjustment refines the intrinsics. A COLMAP text
   model is written to `sparse/0`, so brush and Inria 3DGS can use the
   result too.
2. **Training.** brush's refine strategy on the hand-written wgpu forward
   and backward passes. On the same GPU, at 30k steps, it is faster than
   brush 0.3 on all five benchmark scenes (14–30%). PSNR is within
   0.05 dB of brush on four of them; details in
   [the 3DGS plan](rust_3dgs_plan.md). Reproduce the table with
   [`scripts/benchmark_gsplat_vs_brush.py`](../scripts/benchmark_gsplat_vs_brush.py);
   the dataset layout and the command are in its header.
3. **Mesh.** Median-depth rendering, TSDF fusion and surface nets. Surfaces
   with no SfM point nearby, such as the sky, are dropped. The
   `--normal-weight 0.005` depth-normal loss makes surfaces smoother for
   -0.2 dB.

## 360° cameras

`--equirect` takes a folder of 2:1 equirectangular
panoramas and cuts each one into its six 90° cube faces, at most
`--max-size` square. The faces then go through the three stages as pinhole
images with their exact focal length held fixed, the six of a panorama as
one rig: one pose per panorama, the faces at its centre
(`--equirect-independent-faces` solves them as separate cameras). Faces of
one panorama are never matched with each other, and `--window` counts
panoramas. `--equirect-native` runs SfM on the panoramas themselves as
equirectangular cameras (COLMAP's `EQUIRECTANGULAR`), at most
`2 x --max-size` wide: several times faster, with fewer points. Either way
training sees the cube faces that see SfM points; a camera mount or selfie
stick below the camera is left out. Panoramas taken from almost the same
spot (slow video) have too little parallax to reconstruct: space the
frames out. Without `--equirect`, a folder
that looks like panoramas gets a warning: 2:1 images whose left and right
edges meet and whose top and bottom rows are poles, or whose metadata says
equirectangular. The pinhole model gives no usable result on them.

`--insv VIDEO.insv` takes an Insta360's raw dual-fisheye frames instead of
panoramas. Each image has the two lenses side by side, front lens on the
left, as ffmpeg's `hstack` of the video's two streams produces.

- **Faces.** Each lens is cut into five 90° faces, at most `--max-size`
  square: one along its axis and four tilted 45° from it. All of them stay
  within 90° of the axis. The cut uses the lens calibration in the video's
  metadata ([insta360_lens.md](insta360_lens.md)).
- **Rigs.** The faces of a lens are solved as one rig. The two lenses are
  separate cameras, so nothing is stitched, and `--window` counts lenses
  (two per frame).
- **Scale.** The video's IMU scales the model to metres (next section).
- **Cameras.** The calibration is decoded for the X5; videos from other
  cameras stop with an error.

## Metric scale from the camera's IMU

SfM leaves the model in arbitrary units. When the photos are frames of a
video and the camera logged its IMU, `--imu FILE` scales the model to
metres right after SfM: sparse/0, the splats and the mesh all come out in
metres (`<out>/port` keeps the mapper's units). It is the per-run
visual-inertial alignment of arXiv 2610.35554 with a rolling-shutter
term, validated on the 50 OverMaps360 walks
([insta360_imu_scale.md](insta360_imu_scale.md); Rust:
`visloc_slam::estimate_imu_scale`).

- **The IMU**: an Insta360 `.insv` (the IMU and frame timestamps in its
  trailer; only the file's end is read), or a CSV file
  `time, gx, gy, gz, ax, ay, az` in seconds, rad/s and m/s² (EuRoC's
  `imu0/data.csv` and OverMaps360's `insta360_imu.csv` are recognised by
  their headers).
- **The frames**: each photo's file name must end in its frame number
  (the last run of digits, e.g. ffmpeg's `%06d.jpg`). The frame period and
  the offset to the IMU clock are searched from the gyro against the
  model's rotations; `--imu-frame-period S` gives the search a starting
  period when the frames cover only part of the IMU log. With `--equirect`
  the panoramas are the frames.
- **The output**: the log prints the timing, the rotation residual and the
  scale; `<out>/imu_scale.txt` lists every run. A rotation residual of more
  than a few tenths of a degree means the frames and the IMU do not match
  (wrong numbering, stabilised video).

**From an Insta360 video** (not tried here: it needs Insta360 Studio, the
maker's desktop application). The `.insv` keeps the IMU; the export does
not, so pass the original:

1. In Insta360 Studio, export the clip as a 360° (equirectangular) video
   with FlowState stabilisation, direction lock and horizon levelling off.
   Stabilised frames turn independently of the camera, and the gyro no
   longer matches them.
2. Cut every 10th frame (3 fps at 29.97 fps), numbered in order:
   `ffmpeg -i export.mp4 -vf "select=not(mod(n\,10))" -fps_mode vfr frames/%06d.jpg`.
   Frame `k` is then video frame `10 (k - 1)`; any numbering linear in time
   works, since the period and offset are searched.
3. `gsplat_photos --images frames --out run --equirect --imu original.insv`.

On 136 frames (67 s) of the OverMaps360 walk 0313afa5, view 0 of lens 0 at
1600 px, gsplat_photos registered all frames, found the dataset's frame
timing (0.333683 s per frame) and scaled its model by 6.58. Aligned to
the dataset's own model, it implies 1.0402 m per dataset unit, against
1.0401 from the dataset's model over the whole walk (1.1 cm median
residual).

To train on panoramas that another tool already solved (such as COLMAP
4.1), point `gsplat_train --data <colmap_root>` at the model: for
an `EQUIRECTANGULAR` camera it cuts the same cube faces (`--max-face-size`,
`--min-face-points`, `--faces-dir`), holds out every 8th panorama whole, and
reads `sparse/0` as text or binary.
