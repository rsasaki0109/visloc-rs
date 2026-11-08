# Changelog

All notable changes to `visloc-rs` will be documented here.

## Unreleased

### Added

- **`gsplat_extract`: classes or single objects cut out of a labelled
  splat** (`visloc-gsplat-core` example, no GPU).
  - **Input.** It reads `scene.ply` for the colours, and
    `gsplat_semantic_cuda`'s `--out` for each gaussian's class and object.
  - **Output.** `--select chair:1` (or a whole class, `--select chair`,
    repeated for more) goes to a PLY with `class` and `instance`, and to a
    `.splat` for the web viewer.
  - **`--list`** prints each class's objects with their gaussians, centre
    and size.
  - **MeetingRoom_In.** Chair 1 is 1,041 of the 88,309 gaussians. The big
    table with all the chairs is 15,027 (docs/semantic_splat.md, "One
    object on its own").

- **Labels move the splats' opacities and centres, not their shapes (CUDA
  ABI 15).**
  - **Why.** Through the shape, the labels' cross-entropy spread splats
    over the pixels of their class. On MeetingRoom_In the held-out views
    saw 95–156 splats with a footprint above 2,000 px², against 22–45
    without the labels. In 2 of 16 runs some showed as floaters (27.69 and
    30.63 dB, against 32.35–32.73 dB for the others).
  - **Now.** `gsplat_train_cuda --labels` leaves the conic out of the
    labels' gradient. In eight MeetingRoom_In runs: 31–51 large splats, no
    floaters, 32.46–32.75 dB (32.57–32.77 dB without the labels), and
    87.9–88.2% of the held-out pixels (89.6–89.9% with the shapes,
    79.4–84.5% without the labels). Building_In: 91.3% (92.4% with the
    shapes, 87.2% without) at 32.00 and 32.05 dB.
  - **Outdoors.** On FIORD's Bridge_Out (sky, snow, buildings), with
    eleven SAM 3 prompts, four runs at the default:
    - 95.2% of the held-out pixels, against 93.7–93.8% without the labels
      and 95.7% with the shapes.
    - Windows' IoU goes from 49–53% to 71%.
    - No floaters, at 30.91–31.17 dB (31.17–31.20 dB without the labels).
    - Lamp posts and trees gain little, and the splat keeps none of the
      people walking past (docs/semantic_splat.md, "Bridge_Out").
  - **`--semantic-shape`** keeps the previous behaviour.
    `--semantic-logits-only` is unchanged.
  - **What did not help:** a cross-entropy linear below P_y / A = 0.1 or
    0.5, or the labels left out of the refine weight (96–126 large
    splats). Letting the labels only shrink a shape removed the large
    splats and the 1.8 points with them.
  - **API.** `LabelGeometry` (`LogitsOnly`, `OpacityPosition`, the
    default, `Full`) for `SemanticTraining::geometry`,
    `StepParams::semantic_geometry` (as `u32`) and
    `CudaGsplat::label_backward`.
  - **Tests.** The oracle test checks that the default's conic, rotation
    and scale gradients are exactly zero, and that its other gradients
    match the full gradient's. `labels_train_the_logits` fits all three.

- **`sam3_labels.py --prompt-threshold`.** `--prompt-threshold window=0.4`
  keeps one prompt's instances from its own score instead of
  `--threshold`'s (docs/semantic_splat.md, "Windows alone at 0.4").
  - **MeetingRoom_In.** With the windows alone at 0.4, SAM 3 keeps the
    glass walls it dropped at 0.5: window IoU 34.2% → 63.6%, mIoU 73.8% →
    75.5%. With every prompt at 0.4 the walls lost more.
  - **Building_In.** The windows alone at 0.4 change little: window IoU
    48.1% → 49.7%, mIoU 58.5% → 58.1%. Every prompt at 0.4 costs 3.1
    points of mIoU.
  - The default stays 0.5 for every prompt.

- **Training with the labels inside the colour kernels (CUDA ABI 14).**
  - **The label channel.** The cross-entropy needs only each pixel's
    label-class channel P_y and its coverage 1 - T. `raster_fwd_kernel`
    now blends P_y alongside the colour. `raster_bwd_kernel` takes the
    loss's gradient, two floats per pixel from `label_loss_kernel`, with
    the colour's. The logits' gradient leaves out the part every class
    shares, which the softmax's backward cancels.
  - **Time.** The labels add about 1 ms per step instead of 5.6 ms. In one
    session, MeetingRoom_In's 30k steps took 192 s without them, a median
    of 216 s with them (eight runs) and 372–378 s with the separate passes.
    On Building_In they took 335 s against 298 s.
  - **Results** as before: seven of the eight runs give 32.50–32.60 dB
    and 89.7–89.9% of the held-out pixels. In the eighth, floaters
    appeared in front of two held-out views (27.69 dB).
  - **Test.** `label_loss_matches_the_class_channels` checks the loss and
    every gradient against the same loss through every class channel
    (`render_classes`, the gradient on the CPU, `classes_backward`), for 5
    and 16 classes. Worst error 2.6e-6.
  - **API.** `CudaGsplat::label_backward`. `render_classes` and
    `classes_backward` keep separate passes, now over every pixel.
  - **Fix.** Destroying a context now frees its class probabilities.

- **Where each class's pixels go (`gsplat_semantic_cuda --check`).**
  - **Per class** the check prints the recall and the precision, each with
    the classes that took the most of the rest, and the run prints each
    class's mean confidence over its gaussians.
  - **`--check-images <dir>`** writes each held-out view's classes as
    rendered and as SAM 3 labelled them, in the class colours.
  - **Windows** (docs/semantic_splat.md, "Windows"): their errors are
    pixels SAM 3 calls "other", not walls. SAM 3 drops whole windows in some
    photos: in two held-out MeetingRoom_In photos its best "window" scores
    0.47, under the 0.5 cut, and the splats carry the windows in from the
    other photos. The window IoU understates them.
  - **A lower SAM 3 threshold.** At `--threshold 0.4` SAM 3 keeps those
    windows on MeetingRoom_In. The window votes' confidence goes from 0.63
    to 0.77, and the other named classes move by 2.3 points at most. At
    0.3 walls and ceiling lights lose.
  - **A before/after figure**: two held-out MeetingRoom_In views, the vote
    on a plain splat against a splat trained with the labels (74.9% →
    90.1% and 80.9% → 90.8% of their pixels).

- **Training with the labels (`gsplat_train_cuda --labels`, CUDA ABI 13).**
  - **Classes in training.** Every gaussian gets a logit per class (16 at
    most). The softmax of its logits, its class probabilities, is blended
    like colour.
  - **The loss.** `--semantic-weight` (0.02) times the mean cross-entropy
    of each labelled pixel's normalised blend against SAM 3's label.
  - **The gradient.** It reaches the logits and, through alpha, the
    gaussians' geometry, opacity and brush's refine weight.
    `--semantic-logits-only` keeps it to the logits.
  - **Output.** `scene.ply` gains `class_logit_<c>` properties
    (`ply::ply_properties` reads them back), and `gsplat_semantic_cuda
    --logits` takes the classes from them.
  - **API.** `CudaGsplat::set_labels`, `set_logits`, `logits`,
    `render_classes` and `classes_backward`, `CudaTrainer::set_labels` with
    `SemanticTraining`, and `semantic::label_from_logits`.
  - **FIORD MeetingRoom_In** (held-out views, against SAM 3's labels):
    pixel accuracy 79.4–84.5% over four runs without the labels, 89.8% at
    the default weight. mIoU rises from 68.9–73.1% to 81.5%, and held-out
    instances matched at IoU 0.5 from 64.4–66.1% to 76.8%, at the same
    PSNR (32.69 dB).
  - **Building_In:** pixel accuracy 87.2–87.3% → 92.4%, mIoU 57.8–58.2% →
    71.9%, instances 26.1–26.2% → 45.4%, at the same PSNR. Its ceiling
    lights' IoU goes from 19.6% to 81.5% at weight 0.05.
  - **Cost.** About 12% more training time (the entry above; separate
    passes doubled it). Details in docs/semantic_splat.md.
  - **Building_In with the vote alone** is in the doc too: 87.2% of the
    held-out pixels and 26.1% of the instances.

- **Objects in semantic splats (`gsplat_semantic_cuda --instances`,
  `sam3_labels.py --instances`).**
  - **Instances.** `sam3_labels.py --instances` writes each pixel's SAM 3
    instance as well as its class.
  - **Votes.** Each view's instance image votes on its own
    (`vote_labels`). An instance's members are the gaussians that give it
    at least half of their weight there (`semantic::view_instance_members`).
  - **Merging.** Instances of one class in different views join when their
    normalised member weights overlap by `--instance-overlap` (0.5),
    chained by union-find. Each gaussian takes the object of its class with
    the most weight. Objects seen in fewer than `--min-views` (3) views
    drop (`semantic::merge_instances`).
  - **Outputs.** `--out` gains an `instance` property. `--web` writes
    `instances.splat`, each object its own colour (`instance_color`), and
    `instances.json`. The viewer takes `&instances=`, and O switches
    between classes and objects.
  - **FIORD MeetingRoom_In.** 3,728 instances in 106 photos become 65
    objects (25 chairs) in 2.6 s. `--check` merges the training views' and
    matches 64.4% of the 525 held-out instances to an object at IoU 0.5
    (weighted pixels): chairs 68.8%, tables 74.5%, windows 8.0%. Details in
    docs/semantic_splat.md.

- **A class per gaussian, voted from SAM 3 labels (`gsplat_semantic_cuda`,
  `scripts/sam3_labels.py`).**
  - **Labels.** `sam3_labels.py` runs SAM 3 (transformers `Sam3Model`) on a
    run's images with text prompts. A pixel takes the prompt of the
    highest-scoring instance over it, 0 ("other") where none, and 255 (no
    vote) where the image's alpha masks it out. `--rotate` turns each image
    upright for SAM 3: FIORD's upside-down photos otherwise swap floor and
    ceiling.
  - **Votes.** `CudaGsplat::vote_labels` renders a view and adds each
    splat's blending weight (`T_i a_i`) at every labelled pixel to that
    pixel's class, through the view's lens; `votes` returns the sums (ABI
    12). They match the CPU f64 oracle to 1.7e-5, through a pinhole and
    fisheyes.
  - **Classes.** `semantic::label_gaussians` gives each gaussian its class
    with the most weight and that class's share as its confidence.
    `colour_by_class` paints the splat by class. `ply::encode_ply_with`
    adds float properties (`class`, `class_confidence`).
  - **Web.** `--web` writes the RGB and the class splats with the same
    gaussians in the same order (`splat::importance_order`, now also
    `gsplat_export_web`'s), the legend and the capture cameras. The viewer
    in `docs/euroc_splat` takes `&semantic=` and blends the two with a
    slider or M; `&cameras=` starts at the first capture.
  - **FIORD MeetingRoom_In.** Ten prompts, 0.75 s per photo with SAM 3 on
    an 8 GB laptop GPU, 0.5 s to vote 106 views. `--check` votes with the
    training views only and scores the painted splat at the 14 held-out
    views against SAM 3's labels of those photos: 82.6% of the pixels, mIoU
    71.3% over 11 classes. Details in docs/semantic_splat.md.

- **`gsplat_photos --stream-images`, `--sift-threads`.** `--stream-images`
  keeps one working image in memory at a time (they are written to
  `<out>/images`, SIFT reads each back and drops it), so peak RAM no longer
  scales with the number of photos; the model is byte-identical to the
  in-memory run (tested). `--sift-threads N` caps the SIFT extraction
  threads, each of which holds one image's scale space; with
  `--stream-images` and no `--sift-threads` it defaults to the number of scale
  spaces that fit in half the machine's memory.
- **Per-image masks in SfM (`gsplat_photos --masks DIR`).** COLMAP's
  `ImageReader.mask_path` for the photos pipeline: a PNG per photo, zero
  masked out, keeps SIFT keypoints off the masked-out pixels on the CPU and
  both GPU extractors, and gives the training images alpha 0 there.
  `PhotoSfmConfig::mask_dir` / `mask_flip` (`--flip-mask`), the `masks`
  module. A mask with alpha is read through its alpha, or through its
  colour when the alpha is opaque everywhere (an RGBA export of a
  black-and-white mask, which would otherwise mask nothing). A mask is
  resized to the working images with nearest-neighbour sampling; a missing
  mask or one of another aspect is an error before any image is decoded.
  Not with `equirect` or `insv`. Also `gsplat_photos_cuda --masks`. Details in
  docs/photos_to_splat.md.
- **One mask for every photo (`gsplat_photos --camera-mask FILE`).**
  COLMAP's `ImageReader.camera_mask_path`, for what is fixed to the camera
  (a tripod, a monopod): `PhotoSfmConfig::camera_mask`,
  `ImageMasks::new`. It is read and resized like a `--masks` mask; with
  `--masks` as well, a pixel is masked out where either mask has it out.
  Also `gsplat_photos_cuda --camera-mask`. On FIORD's 200° scenes, whose
  tripod sits at the top of every photo, a mask of it (1.8–2.0% of the
  image circle) gains 0.3–0.4 dB of held-out PSNR off the tripod in the two
  indoor scenes, 0.3–1.4 dB at 75–95° off the axis; the outdoor scene is
  within run-to-run spread. Details in docs/fisheye.md.
- **A dual fisheye's two lenses as one rig (`gsplat_photos --back-images
  DIR --back-camera "MODEL W H PARAMS..."`).** The back lens's photos join
  the front lens's (`--images`, `--camera`), each through its own
  calibration, with no faces cut (`PhotoSfmConfig::back_lens`, `BackLens`).
  Also `gsplat_photos_cuda`.
  - **Frames.** A back photo pairs with the front photo of the same name
    once each folder's common ending is cut off, from the separator it
    starts at (`pair_lens_photos`: FIORD's `..._fisheye1` with
    `..._fisheye2`, or the same names). A frame may lack either photo.
  - **Rig.** A frame's photos are one rig. The back lens starts half a turn
    about the vertical and is refined (`<out>/back_lens.txt`); both cameras
    are held fixed.
  - **Per lens.** Each lens has its own `--max-angle` cone and camera mask
    (`--back-camera-mask`, `ImageMasks::with_lenses`). A pair of the two
    lenses is verified on unit rays (`verify_pair` takes both cameras).
  - **Model.** `sparse/0` has both cameras. `<out>/image_frames.txt`
    (`dataset::FRAMES_FILE`) makes the loader hold out whole frames.
  - **Mapper.** A central rig of fisheyes past 180° (every sensor at one
    centre) now starts on unit rays, as a single camera does. A rig of
    sensors apart starts on the rays ahead of both image planes. Before,
    one match past 90° kept such a rig from starting.
  - **Tested.** A synthetic room through both lenses, one frame without its
    back photo: every photo registers at the true poses (within 0.1°), and
    the back lens settles 0.005° from the half turn.
  - **FIORD.** On MeetingRoom_In and Building_In, every photo of both
    lenses registers, and the back lens settles 0.42–0.50° past the half
    turn, within 0.21° of FIORD's model's average. With both lenses
    refined first, the front lens's held-out PSNR gains 0.67 dB on
    Building_In (1.0–1.6 dB from 75° off the axis) and ties on
    MeetingRoom_In. With the factory lenses it loses 0.1–0.3 dB. Details in
    docs/fisheye.md.
- **Insta360 photos through their factory lens; masks past a fisheye's
  field of view.** On three of FIORD's 200° scenes (front lens, 800 px),
  the mask past 100° gains 0.7–1.8 dB of held-out PSNR over masking
  near-black pixels. The factory lens gains another 1.3–1.8 dB over
  FIORD's calibration where that is off, and ties where it is right.
  Details in docs/fisheye.md.
  - **Reading the calibration.** `visloc_io::insv::read_insta360_lenses`
    reads the lenses' factory calibration from an `.insp` photo as well as
    an `.insv` video. It looks through the whole trailer (a photo's is
    indexed differently), and `InsvLens::image_size` gives the
    calibration image's size.
  - **`insta360_lens_cameras PHOTO.insp`** prints each lens as a `--camera`
    row: the `OPENCV_FISHEYE` camera of its half of the photo, fitted to
    the unified model out to 100° (`FisheyeLens::in_photo`,
    `FisheyeLens::kannala_brandt`).
  - **`gsplat_photos --max-angle DEG`** (and `gsplat_photos_cuda`) leaves
    the pixels past DEG off `--camera`'s axis out of training
    (`PhotoSfmConfig::max_angle`), e.g. a 200° fisheye's dark rim and
    corners.
  - **`gsplat_photos_cuda --mask-black N`** is `gsplat_photos`'s.
  - **`gsplat_train_cuda --alpha-mask`** leaves the masked pixels out of
    the loss and the held-out PSNR.
  - **`gsplat_train_cuda --save-eval DIR`** writes the held-out renders.
- **Distortion refinement in bundle adjustment (`ba_refine_extra_params`,
  `gsplat_photos --refine-camera`).** COLMAP's `ba_refine_extra_params` now
  has a counterpart in the native solver. The shared camera's distortion
  parameters (`CameraModel::extra_param_indices`) are refined with its
  focal length and principal point:
  - up to three in the camera node's free slots, the rest in a second
    node, so the thin-prism model's eight fit too;
  - their Jacobian by central differences through the lens, one path for
    every model.

  `gsplat_photos --camera ... --refine-camera` and `gsplat_photos_cuda`
  start from a rough camera and refine it in the mapper's bundle
  adjustments, instead of holding it fixed. A manifest row now keeps a
  model with zero distortion (`SIMPLE_RADIAL ... 0`) instead of reducing it
  to a pinhole. On TUM-VI room1 (195° fisheye), a rough start (focal length
  8% short, principal point centred, zero distortion) held fixed is 11.9 cm
  off. Refined it is 0.48 cm off, against 0.44 cm for the calibration, and
  its radial mapping lands within 0.12 px of the calibration's.
- **The device TSDF through lenses (`gsplat_mesh_cuda --gpu-tsdf`).**
  `CudaGsplat::tsdf_surface` and `mesh::extract_mesh_gpu` now fuse each
  view through its lens, as the CPU TSDF does: voxels project through it
  (`lens_project`), pixels back-project along its rays
  (`lens_pixel_ray`), and the frustum test is the `max_theta` cone. A
  distorted pinhole is fused with its distortion instead of without, and
  fisheye views are no longer refused. `pinhole_geometry_views` in
  `visloc-gsplat-train` is gone with its last caller. On ETH3D's
  delivery_area rig the thin-prism splat meshes to the CPU's F1 within 0.1
  points (16.1% at 5 cm from the median depth, 18.4% from the planar), and
  fusion takes 8.8 s and 14.9 s instead of 115 s and 191 s.
- **Planar depth and the depth-normal loss through lenses.** Both now
  follow each pixel's ray through the view's lens and use the lens's depth
  (`z`, or a fisheye's distance). This covers the planar median depth
  (`CudaGsplat::render_depth` with `planar`, `gsplat_mesh_cuda
  --planar-depth`) and the depth-normal loss (`--normal-weight` in the CUDA
  and wgpu trainers).
  - A distorted pinhole trains the loss with its distortion instead of
    without, and fisheye views are no longer refused.
  - The CUDA library makes a camera's pixel rays on the device
    (`lens_pixel_ray`, Newton's method) and keeps them while the camera
    stays.
  - The wgpu trainer makes them on the host (`lens::unproject`) and keeps
    one buffer per camera.

  On ETH3D's delivery_area rig, planar depth meshes the thin-prism splat
  to 18.5% F1 at 5 cm, up from 16.1% with median depth. The undistorted
  images reach 18.8% with planar depth.
- **Meshes through fisheye and distorted lenses.** The CPU TSDF
  (`visloc_gsplat_core::mesh::Tsdf`) fuses each depth frame through its
  view's lens:
  - A voxel projects through the lens, and its depth is compared in the
    renderers' terms (`z`, or a fisheye's distance).
  - Pixels back-project along the lens's rays (`lens::unproject`, the
    distortion inverted by Newton's method).
  - Frustum culling becomes a `max_theta` cone, and occlusion culling uses
    each block's bounding cone.

  `gsplat_mesh`, `gsplat_photos` and `gsplat_mesh_cuda` (median and
  expected depth) now mesh fisheye splats. A distorted pinhole is meshed
  with its distortion instead of without. On ETH3D's delivery_area rig, the
  splat trained through the thin-prism lens meshes to 16.1% F1 at 5 cm,
  against 17.9% for ETH3D's undistorted images and 11.1% with the
  distortion dropped.
- **SfM through fisheye lenses wider than 180°.** A camera whose image holds
  rays past 90° from its axis (`Camera::sees_behind`: a panorama, or an
  equidistant fisheye whose corners lie past the radius its polynomial
  reaches at 90°) goes through unit rays where a panorama already did:
  two-view verification and initialization, P3P, triangulation, and the
  front-of-camera tests (`Camera::sees_in_front`). Before, those used the
  normalized image plane, which drops every ray at or behind it. Lenses
  narrower than 180° keep the normalized-plane path.
- **SfM through fisheye and distorted lenses.** See
  [docs/fisheye.md](docs/fisheye.md).
  - The COLMAP-port mapper's bundle adjustment projects through the
    camera's lens (`rig_ba_solver::Lens`: any `visloc-core` model, via
    `Camera::project_with_point_jacobian`). Before, it solved every camera
    with a focal length as an undistorted pinhole, silently: a fisheye
    trajectory came out centimetres off.
  - Focal length and principal point refinement works in front of a lens,
    and writes back only those parameters, keeping the distortion
    (`Camera::set_intrinsics`).
  - The rig manifest's sensor rows take any COLMAP model with its
    parameters after the pose, and `run_colmap_port` writes them.
  - `gsplat_photos` and `gsplat_photos_cuda` take `--camera "MODEL W H
    PARAMS..."`, a calibrated camera for the photos. It is scaled to the
    working images and held fixed. SfM and training then follow its lens on
    the images as they are. `sparse/0/cameras.txt` keeps the model.
  - `visloc-core`: `Camera::distortion_free`, `Camera::set_intrinsics` and
    `CameraModel::colmap_param_count`.
- **3DGS through fisheye and distorted lenses (`visloc_gsplat_core::lens`).**
  See [docs/fisheye.md](docs/fisheye.md).
  - A camera's intrinsics carry a `Lens`: the pinhole or the equidistant
    (fisheye) projection, with COLMAP's radial, tangential and thin-prism
    distortion.
  - COLMAP datasets with `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`,
    `OPENCV_FISHEYE`, `RADIAL_FISHEYE`, `SIMPLE_RADIAL_FISHEYE` or
    `THIN_PRISM_FISHEYE` cameras now train on their images as they are.
    Before, the first three loaded with their distortion dropped, and the
    fisheye models did not load.
  - The CPU reference, the f64 gradient oracle, CUDA and wgpu linearise the
    lens at each gaussian, rays past 90° included. A fisheye sorts by
    distance, and its depth channels hold the distance. The backward passes
    get the Jacobian's derivatives from dual numbers.
  - Past the image, where the distortion is not calibrated, a gaussian is
    drawn through the lens's first-order extension rather than culled
    (`lens::extended_lens_coords`), as a pinhole draws one off its frustum.
    Culling there let training park big gaussians just outside every
    view's field, which then darkened any other camera's render.
  - Meshes, the planar median depth and the depth-normal loss still
    back-project through a pinhole. They take a distorted pinhole without
    its distortion, as before, and refuse fisheye views.
  - CUDA ABI 11 (`GscCamera` carries the lens). The wgpu kernels are
    validated with naga in a test that needs no GPU.
  - `visloc-core` and `visloc-io`: COLMAP's `THIN_PRISM_FISHEYE`
    (`CameraModel::ThinPrismFisheye`), with projection, unprojection and
    point Jacobian.
- **CUDA 3DGS rasterizer and trainer (`visloc-gsplat-cuda`, experimental,
  opt-in).** This is a CUDA backend for 3DGS rendering and training, for
  machines where wgpu cannot reach the GPU (WSL2 exposes CUDA but no NVIDIA
  Vulkan driver). It also serves as a reference for CUDA-level
  optimizations.
  - The packaging follows the DPVO kernels: `native/gsplat_cuda/gsplat_cuda.cu`
    is built by `scripts/build_gsplat_cuda.sh` and loaded at runtime. Cargo
    never compiles CUDA, and the crate is `publish = false`.
  - The rasterizer's maths is the wgpu renderer's. Against the CPU f64
    oracle, every parameter gradient is within 4.2e-6 of its largest value;
    the tests use the wgpu tests' tolerances, 3e-3.
  - The training step is the wgpu trainer's: L1 + D-SSIM, Adam with brush's
    aux losses, refine statistics and mean noise. The scene, its moments and
    the images stay on the device.
  - `CudaTrainer` (example `gsplat_train_cuda`) runs the wgpu trainer's
    brush-strategy loop.
  - On south-building (1024x768, 112 train / 16 eval views) it trains 30k
    steps in 691 s on an 8 GB laptop GPU, ending at 1.41M gaussians and
    23.22 dB held-out PSNR. The README's wgpu run reached 22.77 dB with a
    different SfM and the depth-normal loss.
  - Rasterizer optimizations cut a training step from 30.9 to 21 GPU ms;
    before them the run took 978 s.
    - Sub-tile masks: 8x4 sub-tiles per warp, 8 bits in the sorted value.
    - `project_bwd` skips gaussians that blended into nothing.
    - Gradients go to global memory with native float reductions; on this
      GPU the shared-memory float atomic was a CAS loop.
    - Three dense runs spread over 22.71–23.22 dB.
  - `--sparse-sh-adam` (opt-in) updates SH only for blended gaussians,
    halving the optimizer (18 ms per step). In training it saved 5% at
    22.87 dB.
  - Tried and not adopted, each under 1% per step: the two-stage sort, a
    vectorized Adam and dropping the gradient memset.
    `gsplat_cuda_bench --data` now also reports each step's wall time.
  - Meshes from CUDA:
    - The median depth comes from the splat centre, or with
      `--planar-depth` from where the ray meets the splat's plane.
    - `gsplat_mesh_cuda` meshes a trained splat; `gsplat_train_cuda --mesh`
      meshes after training.
    - The TSDF fusion is shared with wgpu
      (`visloc_gsplat_train::mesh::fuse_views`, behind the new `mesh`
      feature).
  - **ETH3D geometry.** Three high-res training scenes, on their
    ground-truth poses at 1600x1067, score a mean F1 of 33.1% at 2 cm
    (terrains 34.1%, playground 43.9%, office 21.4%). visloc-mvs's clouds
    score 80.0% on the same scenes. The mesh options needed are 1 cm
    voxels, no support filter, depths up to 20 m, min weight 1 and planar
    depth; `gsplat_photos`' defaults score 20.8%. See
    `scripts/eth3d_3dgs_mesh.sh`.
- **Depth priors for 3DGS (opt-in).**
  - `visloc-mvs --save-fused-depth` writes each view's depths that fusion's
    source views agree with.
  - `gsplat_train_cuda --depth-prior <dir> [--depth-weight 0.1]` trains the
    splat's expected depth against them, with a relative L1. A map of another
    size is resampled by nearest neighbour.
  - The expected-depth channel's gradients match an exact colour-channel
    identity to 1.8e-7.
  - On the three ETH3D scenes, weight 0.1 raises the 3DGS mesh's mean F1
    at 2 cm from 33.1% to 35.0% (office 21.4% → 25.5%); weights 0.5 and 1.0
    give 34.0% and 35.3%. Accuracy rises by up to 32 points, but
    completeness does not, and the mesh stays far below the MVS cloud that
    supervised it (67.6%).
  - CUDA ABI 5.
- **Depth-normal loss for the CUDA trainer (opt-in).**
  - `gsplat_train_cuda --normal-weight w [--normal-start 7000]` runs the wgpu
    trainer's 2DGS / PGSR-style loss, which `CudaTrainer` refused before.
  - The step blends a normal channel next to the expected depth and
    back-propagates it to the rotations.
  - `CudaGsplat::render_geometry`, `normal_loss` and
    `backward_with_geometry` expose the channels and their gradients.
  - The normal channel's gradients match an exact colour-channel identity
    to 9.0e-7. The loss's gradients match central differences of an f64 CPU
    copy to 2.5e-6.
  - It does not improve the ETH3D meshes. At weight 0.005 the mean F1 at
    2 cm is 32.0% against 33.1%; with the depth prior 0.1 it is 34.0%
    against 35.0% for the prior alone. 0.05 is far too strong (terrains
    22.9%).
  - CUDA ABI 6.
- **TSDF on the GPU for CUDA meshes (`gsplat_mesh_cuda --gpu-tsdf`).**
  - `mesh::extract_mesh_gpu` (`CudaGsplat::tsdf_surface`) fuses the median
    depth and extracts surface nets on the device, with `Tsdf`'s maths,
    options and filters.
  - Every block is allocated before any view is fused. The blocks are then
    fused in slabs along x, so memory stays bounded at any voxel size.
  - On ETH3D at 1 cm it matches the CPU's F1 to 0.2 points in 4–5x less time
    and 1/6 of the memory (terrains: 5.4 s and 1.6 GB against 24.0 s and
    10.3 GB). playground, which ran out of memory on the CPU, meshes in
    12.0 s and 4.2 GB.
  - One view gives the CPU's mesh, and any number of slabs gives the same
    surface bit for bit.
  - `Mesh::remove_small_components` counts triangles per component in a
    vector instead of a hash map.
  - CUDA ABI 7.
- **Descriptor matching on the GPU for photos-to-dataset
  (`gsplat_photos_cuda`, `visloc-gsplat-cuda` feature `photos`).**
  - `PhotoSfmConfig::matcher` lets a `PairMatcher` take
    `build_photo_dataset`'s candidate pairs. The CPU matches if it fails.
  - `matching::CudaPairMatcher` computes each pair's top-2 scores in both
    directions with exact 8-bit dot products (`CudaGsplat::match_top2`). It
    then applies the CPU matcher's filter, so the matches are
    `BruteForceMatcher::match_descriptors_cross_checked`'s, bit for bit.
  - On south-building (128 images, 10.5k SIFT each), matching and
    verification take 18.4 s against 780.7 s for the CPU matcher, with the
    same 2461 verified pairs. The whole SfM takes 148.4 s against 945.0 s,
    with the same model: 128/128 registered, 101,450 points, focal 850.2 px.
  - Verification alone takes 5.3 s once the CPU matcher no longer holds
    every core.
  - `CudaGsplat` is `Send`. CUDA ABI 8.
- **PatchMatch MVS on CUDA (`visloc-mvs --features cuda`, `mvs --cuda`,
  opt-in).**
  - `native/mvs_cuda/mvs_cuda.cu`, built by `scripts/build_mvs_cuda.sh`,
    runs the wgpu shaders as CUDA kernels. `visloc_mvs::CudaPatchMatch`
    loads it at runtime from `VISLOC_MVS_CUDA_LIB`, so Cargo never compiles
    CUDA.
  - On ETH3D (8 GB laptop GPU) it scores what the CPU path does, 16 to 21
    times faster: terrains at 1600 px 84.73% → 85.05% F1 in 209 s instead
    of 4483 s, playground and office at 800 px within 0.02 points in 49 s
    and 28 s instead of 804 s and 468 s.
  - At 1600 px office scores 72.8% and playground 67.7%, in 102 s and
    177 s.
  - With those 1600 px maps, the hybrid meshes (`--depth-maps`) of the same
    splats score 64.2% instead of 62.7% with a plain splat, and 68.1%
    instead of 65.7% with a prior-trained one (mean ETH3D mesh F1 at 2 cm;
    office +3.8 to +4.8 points). With the prior trained on those maps as
    well, the prior-30 hybrid reaches 70.5% (office 59.0% → 66.9%), and
    the splat's own expected-depth mesh 67.7% instead of 63.6%.
  - Three more scenes end to end (pipes, terrace, kicker): the prior-30
    hybrid scores 81.2% (76.8 / 87.3 / 79.5%), 75.8% over six scenes. A
    plain splat can hurt: on pipes its hybrid scores 67.8% against 68.9%
    for the MVS depth alone. courtyard's meshes were too large for the
    evaluator on this machine.
  - The wgpu path's pass preparation moved to `pass_setup.rs`, shared by
    both backends, as are their tests (`pass_setup::checks`).
  - The crate's `unsafe_code` lint is `deny` instead of `forbid`; the
    `cuda` module alone allows it.
- **Meshes from MVS depth with the splat filling its holes
  (`gsplat_mesh_cuda --gpu-tsdf --depth-maps <dir> [--depth-maps-only]`,
  `visloc_gsplat_cuda::mesh::DepthMaps`).**
  - The GPU TSDF fuses per-view depth maps (visloc-mvs `--save-fused-depth`)
    wherever they have a depth, and the splat's depth elsewhere.
    `CudaGsplat::tsdf_surface_with_maps` and `TsdfParams::override_mode` do
    this.
  - With a plain splat, whose image is untouched, the ETH3D mesh F1 at 2 cm
    is 62.7%, against 33.3% for the splat's own mesh and 54.6% for the MVS
    depth alone. With a splat trained with the depth prior at 30 it is
    65.7%.
  - `mesh::read_depth_map` reads and resamples a map; the trainer's
    `load_depth_priors` uses it.
  - `.gitattributes` keeps `*.sh` in LF, so the scripts run under WSL2 from
    a Windows checkout.
  - CUDA ABI 10.
- **Meshes from the expected depth (`gsplat_mesh_cuda --expected-depth`,
  `visloc_gsplat_cuda::mesh::MeshDepth`).**
  - The depth prior and the depth-normal loss train the expected depth, but
    meshes were fused from the median depth.
  - Fusing the expected depth (where the opacity exceeds 0.5) from splats
    trained with a strong prior lifts the ETH3D mesh F1 at 2 cm from 33.3%
    to 63.6% at prior weight 30: terrains 78.1%, playground 60.4%, office
    52.3%. The prior's own MVS clouds score 67.6%.
  - Completeness rises with it, which no loss achieved in the median mesh
    (terrains 26.9% → 70.2%).
  - The cost is the image fit: the final training L1 about doubles at 30.
  - `extract_mesh` / `extract_mesh_gpu` take a `MeshDepth` instead of
    `planar: bool`. CUDA ABI 9.
  - `gsplat_train_cuda --depth-weight-final W`
    (`CudaTrainer::set_depth_weight_final`) moves the prior's weight
    log-linearly to W over the run. With every 8th view held out, 30 → 1
    costs 0.79 dB of held-out PSNR against the plain splat, where a
    constant 30 costs 1.81 dB. Its expected-depth meshes score 54.6% against
    63.2% (plain median mesh: 33.5%).
- **`colmap_resize`** (example of `visloc-gsplat-train`) scales a COLMAP text
  model of pinhole images and the images themselves to a maximum or an
  exact size (`--size` makes cameras of slightly different sizes, as in some
  ETH3D scenes, share one).
- **`scripts/build_eth3d_evaluator.sh`** builds the official ETH3D
  multi-view evaluator on current toolchains (CMake 4, Boost 1.90, PCL 1.15).
  - See [docs/gsplat_cuda.md](docs/gsplat_cuda.md).
- **`visloc_gsplat_train::config`.** `TrainConfig` moved here from the
  wgpu-only `trainer` module, together with the image packing, brush's
  bounds and `population_to_scene`, so that a trainer without wgpu can use
  them. `trainer::TrainConfig` is still exported.
- **`--detector-quality Q`: a lower ORB detector threshold (opt-in).** The
  mapping detector keeps corners stronger than 1% of the image's strongest
  (upstream's quality level); in OpenLORIS-Scene's corridor a few bright
  corners push the walls' texture below that.
  `basalt_euroc_online_slam_demo --detector-quality Q` builds a map with
  another level (`map.json`: `feature_quality_level`), and the localize demo
  and `visloc_map_localize_node` detect the same way
  (`visloc_basalt::mapper::features::detect_keypoints_mapping_with`).
  With the blur at 0.003: corridor1-1 localizes 360 times in corridor1-5's
  map instead of 75, its VIO + map 13 cm off at the median instead of 62 cm;
  office1-7 186 times instead of 52, at 2.66 cm instead of 6.61 cm. About
  20 ms more per image. On EuRoC's bright, textured images, alone, it adds
  fixes in the Vicon rooms but not accuracy (six of the eight queries
  0.04–11 cm worse), so it stays off by default. `map_files::Frontend`
  gathers the examples' equalization, blur and quality level.

- **Blur before ORB (opt-in).** ORB's descriptor compares single pixels,
  and on OpenLORIS-Scene's grainy T265 images noise flips its bits: two
  near-identical views from two corridor sessions shared 22 consistent
  matches. `basalt_euroc_online_slam_demo --smooth-features SIGMA` blurs
  each keyframe image before ORB extraction
  (`visloc_basalt::smoothing::GaussianSmoothing`), `map.json` records it
  (`feature_smoothing_sigma`), and `basalt_euroc_localize_demo` and
  `visloc_map_localize_node` blur their images the same way;
  `basalt_map_merge` refuses maps built differently. At sigma 1.5 the pairs
  share 38 matches, corridor1-5's own images localize 52% of the time
  instead of 33.5%, and corridor1-1 localizes 75 times in its map instead
  of 57 (VIO + map 1.55 → 1.35 m against the map around it). Elsewhere more
  images localize at about the same error: office1-7 52 instead of 5 (6.86
  → 6.61 cm), V2_03 36 instead of 5 (29.3 → 28.8 cm), MH_02 1.97 → 2.03 cm.
  About 10 ms more per image. Off by default.

- **Wheel odometry with its heading from a gyro.** Wheels slip in turns in
  place: on OpenLORIS-Scene's corridor (corridor1-1, 272 s) the wheel
  odometry's heading drifts 44.5° from the ground truth.
  `visloc_localization::GyroWheelOdometry` keeps the wheels' travel and
  turns it by the heading a gyro integrates about the vertical, which the
  accelerometer gives (any IMU, mounted any way); at a standstill the
  heading holds and the gyro's rate is its bias. The heading then stays
  within 5.6°, and 60 s after a known pose the odometry is at most 1.4 m
  off instead of 15.9 m. Wheel odometry + map on corridor1-1 localizes 58
  times instead of 12 and its RMSE falls from 3.5 to 0.95 m; office1-6
  and office1-7 go from 5.11 to 4.79 cm and from 11.69 to 10.60 cm.
  `visloc_wheel_calibration_node` uses it with an `imu` topic
  (`gyro_heading`, on by default), `launch/map_localization.launch.py`
  passes `imu_topic` to it, and `wheel_calibration_replay` takes `--imu`.

- **Extra descriptors per landmark (opt-in).**
  `basalt_euroc_online_slam_demo --extra-descriptors` keeps, besides each
  landmark's medoid descriptor, up to three of its views' descriptors,
  farthest first and at least 24 bits apart
  (`landmark_descriptors_extra.txt`).
  `visloc_localization::BinaryLandmarkIndex::with_extra_descriptors`
  matches a keypoint against the nearest of a landmark's descriptors, with
  the ratio test against the nearest other landmark;
  `basalt_euroc_localize_demo --extra-descriptors N` and
  `visloc_map_localize_node`'s `extra_descriptors` use them. On the eight
  EuRoC queries they find more fixes (MH_04 56 instead of 48) but less
  accurate ones: six queries get 0.02–1.2 cm worse, V2_03 better (29.3 to
  23.6 cm), and the map-wide search up to three times slower. Off by
  default.
- **`evaluate_map_localization.py --mode local`.** Like `fixed`, but the
  map->ground truth transform is fitted for each scored pose on the stretch
  of the map's own trajectory within `--align-radius-m` (5) of it and, by
  the ground truth, facing within 90° of it, turning about the vertical
  only. On a map whose SLAM drifted over a long loop (corridor1-5: 1.33 m
  RMSE, no loop closed) it scores a localization against the map around
  it, on the pass whose landmarks it saw.
- **`wheel_calibration_replay`.** `ros2/visloc-ros2/examples` runs
  `visloc_wheel_calibration_node`'s calibration on recorded wheel odometry,
  VIO and map localizations (TUM files), without ROS, and writes the
  calibrated wheel odometry for `basalt_euroc_localize_demo --odometry`.
- **`basalt_euroc_localize_demo --stale-projection-min-inliers N`.** A
  prediction past the fresh limits is tried too, and its projection search
  counts only with N inliers. With 60, its 14 fixes over the eight EuRoC
  queries are at most 10.4 cm off; MH_04 goes from 22.0 to 21.2 cm and
  V1_03 from 4.80 to 4.24 cm, but V2_03, with 5 fixes in all, from 29.3 to
  39.1 cm. Off by default.

- **`gsplat_photos --imu`: metric models from the camera's IMU.** For
  photos that are frames of a video (file names ending in the frame
  number), the model is scaled to metres right after SfM, so sparse/0, the
  splats and the mesh come out in metres. `visloc_slam::estimate_imu_scale`
  is the Rust port of the per-run visual-inertial alignment validated on the
  50 OverMaps360 walks (docs/insta360_imu_scale.md): it searches the frame
  timing from the gyro and fits scale, gravity, accelerometer bias and a
  rolling-shutter term per run. It matches the Python reference to four
  decimals in seconds. The IMU comes from an Insta360 `.insv`
  (`visloc_io::insv` reads its trailer: IMU and video frame timestamps) or a
  CSV file (`visloc_io::imu_csv`: SI units, EuRoC, OverMaps360). On 136
  frames of an OverMaps360 walk the metric model agrees with the dataset's
  IMU scale within 0.01%. The `imu_metric_scale` example runs the estimator
  on any COLMAP model.
- **`insv_stitch`: an Insta360 video to equirectangular (360°) frames.**
  The `visloc-gsplat-train` example (feature `euroc`) stitches an X5
  `.insv`'s two lenses with their factory calibration, no Insta360 Studio
  involved: every `--step`-th frame to JPEGs, or one video through ffmpeg
  (`--video`), 6,496 x 3,248 by default (`dual_fisheye::stitch_equirect`).
  - The back lens is fitted to SIFT matches along the seam in 16 frames
    (`dual_fisheye::seam_pairs`, `fit_back_lens`): its rotation and its
    optical centre, across the lenses' parallax exactly and along it up to
    one shared shift. The recorded angles give only the turn about the axis
    (the fit lands within 0.06° of it), not the tilt. The back lens's
    optical centre sits 5–21 px from where the calibration puts it relative
    to the front lens's (`FisheyeLens::shifted`); `gsplat_photos --insv`
    keeps the model as it is, which gives it the better metric scale.
  - The shared shift is stitched in (`stitch_equirect`'s `parallax`): the
    scene's typical distance along the seam (1.8–3.0 m outdoors), or the
    lens models' scale where it points away from the axes. `--distance`
    sets a distance instead.
  - On five OverMaps360 walks the two lenses then put the same thing
    0.12–0.18° apart along the seam (median), against 0.85–1.15° with the
    recorded angles stitched far away.
  - Each frame's two lenses are matched along the seam and one morphed
    into the other across a 6° band (`dual_fisheye::SeamDisparity`,
    `stitch_equirect`'s `disparity`), so things nearer or farther than the
    stitching distance show once: census costs and semi-global matching on
    strips rectified about the lenses' baseline. On the five walks' frames
    the fit did not use, the lenses then put the same thing 0.033–0.038°
    apart within 5° of the seam (median), against 0.11–0.19° with one
    distance; about 0.6 s more per frame. `--seam blend` only mixes the
    lenses. Exposure is not matched: in the overlap the lenses agree
    within 1.7% per colour.
  - `--back-lens factory` or a `gsplat_photos --insv` run's
    `back_lens.txt` skip the fit; `--frame` stitches side-by-side images.
- **`gsplat_photos --insv-calibrate`: both Insta360 lenses calibrated
  from the video.** The first reconstruction goes into `<out>/calibration`.
  From it each lens's optical centre and focal lengths are calibrated
  (`<out>/lenses.txt`), and the second reconstruction cuts the faces from
  the calibrated lenses.
  - `dual_fisheye::calibrate_lenses` is a bundle adjustment in the lens
    images' pixels through the unified model (points eliminated,
    soft-L1). It refines the lenses with the back lens's pose, the frames
    and the points. `photos::calibrate_insv_lenses` feeds it a run's
    observations and SIFT matches along the seam, which pin the two
    centres to each other.
  - On three OverMaps360 walks, the median distance to the dataset's
    camera track went 0.148 → 0.109 m and 0.343 → 0.159 m, but
    0.307 → 0.340 m on the park walk. The metric scale moved by up to 5
    points. It is not the default.
  - `--insv-lenses lenses.txt` and `insv_stitch --lenses` reuse a
    calibration (`dual_fisheye::LensCalibration`,
    `PhotoSfmConfig::insv_lenses`, `FisheyeLens::with_focal_scale`).
- **`gsplat_photos --insv`: Insta360 video straight to a metric model.**
  The input is the video's raw dual-fisheye frames, both lenses side by
  side (one ffmpeg command), with no Insta360 Studio export.

  - `visloc_io::insv` reads each lens's factory calibration (unified model)
    from the `.insv` metadata. The video's lens image is the square spanning
    the 200° image circle around the calibrated centre
    (docs/insta360_lens.md).
  - `visloc_gsplat_train::dual_fisheye` cuts each lens into five 90° faces
    within 90° of its axis. A frame's ten faces are solved as one rig, with
    the back lens's pose refined by the mapper. The video's IMU scales the
    result to metres.
  - The IMU frame search starts from the `.insv`'s own frame period.
  - Without `--images`, `gsplat_photos --insv VID.insv --out run` cuts the
    frames itself with ffmpeg (`--frame-step`, default 10;
    `dual_fisheye::extract_dual_fisheye_frames`).
  - The refined back lens goes to `<out>/back_lens.txt`.
  - On 200 frames each of five OverMaps360 walks (CPU, 13–30 min), SfM
    registered 99.4–100% of the faces.
    - Outdoors, the metric scale is within 4% of the dataset's IMU scale
      over the same frames.
    - Indoors (a mall), it is off by 12%, with a weak fit (|g| 9.57).
    - These runs used 640 px faces. With 1024 px faces the mall's fit is
      sound (|g| 9.80) and lands +1.2% over the same frames, while the
      outdoor walks move by at most 0.8 points. Keep `--max-size` at 1024
      or more.
    - The `.insv` image-circle string is read after `p2_` or `n2_`.

### Fixed

- **A fisheye pixel past the lens's reach unprojected to a ray.** An
  `OPENCV_FISHEYE` (or radial fisheye) lens's radius peaks at some angle,
  and the pixels beyond it, such as the corners past a 200° lens's image
  circle, have no ray. `Camera::unit_ray_from_pixel` solved for the angle by
  Newton and took wherever it stopped, sometimes within 40° of the axis. It
  now returns `None` unless the angle projects back to the pixel, as the
  thin-prism model already did. With a lens refined on FIORD (radius peaking
  114° off the axis, inside the 800 px image's corners), `--max-angle 100`
  kept 1.4% of each image past 100°, dark corners included in training.
  Two-lens rigs of those lenses (`--back-images`, above) trained so scored
  1.1–1.4 dB lower on their held-out photos.
- **`--refine-camera` could fold a fisheye's lens on the initial pair.**
  The mapper's first bundle adjustment refines the lens from the initial
  pair's points alone. On one of FIORD's 200° scenes, from 36 points, its
  `k1..k4` bent the radius back past 68°, so the radius shrank as the
  angle grew. The lens stayed so until the final bundle adjustment, the
  mapper dropped every observation past 76°, and the held-out PSNR fell
  to 22.5 dB. The native bundle adjustment now refuses a step whose lens
  stops growing with the angle, along any of eight directions, inside the
  radius 99% of the solve's keypoints reach (`SharedLens::keeping_reach`).
  - That scene then refines like the other two: the refined lens gains
    0.4–0.8 dB of held-out PSNR over the factory lens on all three.
  - On the other two scenes the refined lens is unchanged.
  - Holding the last keypoint as well cost one scene 0.4–0.8 dB: the
    outermost keypoints lie at the image circle's edge, past 100°.
  - Details in docs/fisheye.md.
- **The GPU descriptor matcher found almost nothing on Apple Silicon
  (#465).** Its shaders wrote shared memory one vec4 component per thread,
  and on Metal a component store can compile to a read-modify-write of the
  whole vec4, so concurrent writers lost each other's components (5 matches
  where the CPU found 817). Every shared write is now a whole vec4 by one
  thread; the matches are bit-identical on Vulkan.
- **`VlfeatSiftGpu` panicked on an image without DoG candidates (#453).** A
  flat image (sky, road, a blank frame) read back zero bytes, which were
  rejected as misaligned. It now returns no features, as the CPU extractor
  does.
- **An Insta360 photo's lens centre had x and y swapped.** An `.insp`
  photo's calibration has the photo's axes transposed: its `cx` is the
  row in the lens's square and its `cy` the column.
  `FisheyeLens::in_photo` (behind `insta360_lens_cameras`) read them as
  they stand, so the centre landed 6 px off along each axis on FIORD's
  One RS 1-Inch photos. It now swaps them, and `fx`/`fy` and `p1`/`p2`
  with them.
  - The swapped centres lie within 0.9 px of the image circles' centres.
  - On FIORD's three scenes, the factory runs gain 0.1–0.4 dB of held-out
    PSNR.
  - Videos (`--insv`) are unchanged; they put the centre in the middle of
    the lens image.
  - Details in docs/insta360_lens.md and docs/fisheye.md.
- **Kannala-Brandt fisheyes unprojected rays past 90° to their mirror
  image.** `OPENCV_FISHEYE`, `RADIAL_FISHEYE` and `SIMPLE_RADIAL_FISHEYE`
  built the ray as `(tan θ · x̂, 1)`, which past 90° points ahead of the
  camera, on the opposite side. `normalize_pixel` then handed E and PnP a
  wrong point for every such pixel. The ray is now `(sin θ · x̂, cos θ)`.
  `THIN_PRISM_FISHEYE` was already right. A point straight behind such a
  camera no longer projects to the image centre.
- **The photo mapper fell into wrong solutions on small images.**
  `run_colmap_port` (behind `gsplat_photos` and the EuRoC path) used
  COLMAP's registration threshold of 12 px whatever the image size. On
  752x480 images that let P3P keep poses the correspondences barely pin
  down. The mapper then followed the RANSAC seed into one of two solutions,
  and on ETH3D's rig cameras about one run in three ended a metre off with
  every camera turned around. The threshold now scales with the long side:
  12 px at 1600 px and above (unchanged there), 5.6 px at 752, at least
  4 px (`abs_pose_max_error_for`). Over 8 seeds on three such ETH3D runs,
  4 or 6 px gives 1-4 cm on every seed, where 12 px gave 1-119 cm; 8 px
  already lets some drift to 14 cm.
- **`gsplat_photos_cuda` matched nothing.** With the default VLFeat-compatible
  SIFT, the descriptors live as bytes beside the features, but the external
  (CUDA) matcher was handed the features' empty descriptor lists, so every
  pair verified 0 matches and the mapper got no images. It now gets the
  descriptors the CPU matcher uses.
- **`gsplat_mesh_cuda --depth-maps-only` carved surfaces away through the
  maps' holes.** The GPU TSDF carved free space along every pixel without
  a depth, but a depth map's 0 is unknown, not free: each view's holes
  (two thirds to three quarters of its pixels) erased the surfaces that
  other views had seen. A map's hole now adds nothing and carves nothing.
  With `--depth-maps`, `--min-component` also defaults to 0, since the
  maps' small components are mostly surface, not floaters. Over six ETH3D
  scenes the MVS depth's own mesh scores 70.2% F1 at 2 cm instead of
  61.3% (kicker 47.8% → 75.0%), and the hybrids 71.4% and 76.5% instead of
  70.1% and 75.8%. With the filter off either way, treating holes as
  unknown loses 1.6 and 2.7 points on office and pipes, where carving had
  also removed wrong depths.
- **Wheel calibration: the offset wandered on long runs.** A robot's
  wheels slip in turns in place, and on OpenLORIS-Scene's corridor
  (corridor1-1: 284 s, 232 m) their heading drifted 38° off the VIO's in
  two minutes. Fitted with one heading over the 120 s window, the offset
  absorbed that drift and wandered over −117…29 cm forward and −64…71 cm
  left, and the map's scale from localizations 140 s apart came out 0.485.
  `visloc_localization::WheelCalibration` now gives each 5 s of VIO pairs
  (`segment_s`) and each 30 s of map localizations (`fix_segment_s`) a
  heading and a position of their own, sharing the offset and the scale:
  the offset stays at −4.0…2.8 / 6.7…12.1 cm (the office sessions':
  1–3 / 6–8 cm), the fit residual falls from 62 to 2.3 cm, and the scale
  is the map's 1.0315. `visloc_wheel_calibration_node` takes both as
  parameters. The short office sessions' wheel odometry + map stays within
  0.01 cm.
- **`openloris_to_euroc.py` on large .7z files.** py7zr fed its LZMA decoder
  more input on every call while the decoder still held some, so on the
  T265's incompressible PNGs its memory grew with the archive (to 9 GB, and
  killed, on corridor1-1's 13.9 GB). The converter now feeds it only when
  it needs input: 54 MB instead of 417 MB on a 547 MB test archive.
- **ROS 2 nodes: a missed `/tf_static` stalled them for good.** A RustDDS
  reader sometimes never receives a transient-local writer's samples: a
  probe with the nodes' `/tf_static` reader
  (`ros2/visloc-ros2/examples/tf_static_probe.rs`), started 200 times next
  to freshly started `static_transform_publisher`s, had nothing after 30 s
  in 4 cases, and in one OpenLORIS-Scene replay `visloc_vio_node`
  published no odometry. A second reader in the same node receives them.
  `visloc_vio_node` (`base_frame`) and `visloc_map_localize_node`
  (`extrinsics tf`) now subscribe again while a transform they need is
  missing (`dds::StaticTfListener`): 5 s after the first subscription,
  then with doubling waits, at most six times. With that, 300 of 300
  probes had both transforms.
- **`gsplat_photos --equirect-native` crashed writing its model.** Since
  0.3.0 the sparse model lists each image's observations. With
  `--equirect-native` the tracks observe the panoramas while the model
  holds their cube faces, so the panorama observations were indexed as
  faces, and every run panicked at the end. Each observation now goes to
  the face nearest its ray, at the pixel where the ray crosses it. On the
  synthetic room the faces' observations reproject at a median 0.09 px, as
  with `--equirect`. The two end-to-end tests that cover this (run with
  `--ignored`) misread the observation lines and failed too; they now also
  check the observations' reprojection.
- **Saving a map from a lens wider than the Double Sphere model failed.**
  The map export fits COLMAP's `OPENCV_FISHEYE` to each Double Sphere
  camera. It gave up when an image corner did not unproject, which happens
  with a RealSense T265's fisheyes. The corner is now moved toward the
  principal point until it unprojects, so the fit covers the model's
  domain: 0.03 px over 118° for the T265. EuRoC maps are unchanged.

- **visloc-basalt: `vio_min_triangulation_dist` was ignored.** The VIO's
  landmark baseline gate was hard-coded at 5 cm; it now reads the
  configuration value, as upstream does. The pinned EuRoC file says 5 cm,
  so its results are unchanged.

- **visloc-basalt: `vio_obs_std_dev` and `vio_obs_huber_thresh` were
  ignored.** The visual residuals were weighted with a hard-coded 0.5 px
  standard deviation and 1 px Huber threshold; they now read the
  configuration, as upstream does. The pinned EuRoC file has the same
  values, so its results are unchanged.

- **`basalt_map_merge` wrote merged maps without point tracks.** The
  COLMAP text reader fills only the keyframes' observations, so the
  merged `sparse/points3D.txt` had an empty `TRACK[]` for every point.
  Localization never reads the tracks, but COLMAP and other readers do.
  The saved-map loader of the Basalt examples now rebuilds each
  landmark's track from the keyframes, and the save/load round trip test
  checks the tracks. Maps merged before this fix should be merged again.

- **ROS 2 nodes: relative topics were invisible to ROS 2.** `ros2-client`
  0.11 resolves a relative topic name in the root namespace to the DDS
  topic `rt//odom` instead of `rt/odom`. Only other `ros2-client` nodes
  saw those topics, so a ROS 2 subscriber to `/odom`, `/map_pose`,
  `/pose` or `/path`, or a camera driver publishing `/left/image_raw`,
  never connected; only the absolute `/tf` and `/diagnostics` did. The
  nodes now resolve names under their namespace themselves (`odom` is
  `/odom`, `/robot1/odom` with `-r __ns:=/robot1`).

### Added

- **`visloc_wheel_calibration_node`: wheel odometry calibrated while
  driving.** It fits where a robot's wheel odometry is measured relative to
  `base_link`, `base_link`'s lean and the map's scale
  (`visloc_localization::WheelCalibration`, the Rust port of
  `scripts/calibrate_wheel_offset.py`), and publishes the wheel odometry as
  `base_link`'s pose and `odom -> base_link`. The offset comes from
  `visloc_vio_node`'s odometry once the robot has turned 180° and travelled
  1 m; the lean and the scale from `visloc_map_localize_node`'s
  localizations (the new `map_fix` topic), which carry the map's own gravity
  and scale, and without a VIO the offset too, held near the saved one
  where their turns do not show it. A VIO session's scale is reported but
  not used: on OpenLORIS-Scene it was up to 3.6% off the map's. The node
  refits every 5 s over the last 120 s, continuous wherever a new estimate
  takes over, and a saved `calibration_file` applies from the start. On
  OpenLORIS-Scene through ROS 2, the map's localizations alone take
  office1-6's wheel odometry + map from 8.01 to 5.49 cm; office1-3's
  offline calibration, saved, gives 4.92 cm and office1-7 4.66 cm (12.10 as
  is), where its 24 localizations show only the lean. In the demo, the
  map's lean and scale with office1-3's offset take office1-7 from 5.07 to
  3.64 cm. `launch/map_localization.launch.py` starts it with
  `wheel_odom_topic` (and `wheel_calibration_file` to keep the calibration
  between launches).
- **The ROS 2 nodes on a real robot's wheel odometry.**
  `tools/euroc_to_rosbag2.py --odometry FILE.tum` adds a robot's wheel
  odometry to the bag (`nav_msgs/Odometry` on `/wheel/odom`, `odom ->
  base_link`). On OpenLORIS-Scene, played at real time into
  `visloc_map_localize_node` (`extrinsics:=tf`, `weak_gate_m:=0.05`)
  against the scale-corrected maps of other sessions, it gives 2.47 and 3.25
  cm RMSE with the wheel odometry and 1.51 and 2.22 cm with
  `visloc_vio_node`, within 0.4 cm of the offline demo.
- **`calibrate_wheel_offset.py` levels the body frame.** Planar wheel
  odometry keeps its body level, so a tilt in the extrinsic moves the IMU
  sideways, by 1.6 cm per degree with OpenLORIS-Scene's camera 0.9 m above
  `base_link`. The script now turns the body frame by the smallest rotation
  that makes the VIO's mean up its z axis (`--keep-tilt` skips it): 1.9° on
  OpenLORIS-Scene. Wheel odometry + map falls from 4.4–6.2 to 3.5–5.1 cm
  RMSE on the maps as shipped (five of six runs better), and the wheel
  odometry alone to 0.5–4.3 cm. It also normalizes the quaternion it reads.
- **`scripts/scale_stereo_baseline.py`: the stereo VIO's scale from the
  wheel odometry.** A stereo VIO takes its scale mostly from its camera
  baseline. On OpenLORIS-Scene the T265 calibration shipped with the
  dataset makes the SLAM 1–3.5% long, which is most of the maps' error.
  `calibrate_wheel_offset.py` prints the SLAM's scale against the wheel
  odometry (1.013 and 1.015 on the two mapping sessions; 1.017 and 1.010
  against the ground truth), and the script scales the calibration's
  baseline by the inverse of their mean (0.986). Maps rebuilt with it are
  0.76 and 0.91 cm from the ground truth instead of 2.75 and 1.58 cm, and
  VIO + map falls from 2.8–6.9 cm to 1.2–5.4 cm RMSE on three queries.
  With `--weak-gate-cm 5` too, it is 1.1–2.4 cm.
- **Close weak fixes after the first map pose, as an option.**
  `OdomCorrectionConfig::weak_gate_m` (`basalt_euroc_localize_demo
  --weak-gate-cm`, `visloc_map_localize_node` `weak_gate_m`; off by
  default): once the correction is set, a weak fix within that distance
  (and 2°) of the corrected odometry is used as a fix. Localizations
  matched around the predicted projections can then be weak too. On
  OpenLORIS-Scene office1-7, whose images rarely reach 30 inliers against
  another session's map, VIO + map falls from 6.86 to 3.35 cm (map B) and
  from 7.62 to 3.80 cm (merged map). On the eight EuRoC queries, where
  full fixes are frequent, it changes the error by -0.6 to +0.5 cm, so it
  stays off.
- **`scripts/calibrate_wheel_offset.py`: where the wheel odometry is
  measured.** `--odometry` and the node's `extrinsics tf` place the camera
  relative to the odometry's body. If that body is not the point the wheel
  odometry reports (the turning centre), every turn moves the computed
  camera pose, which looks like drift. The script fits that planar offset
  from a run with both the VIO and the wheel odometry, with no ground truth,
  and prints the corrected `--imu-from-body`. On OpenLORIS-Scene the
  dataset's `base_link` sits 5–8 cm from it. Calibrated from one
  session's VIO, the wheel odometry + map error falls from 7.2–11.9 cm to
  4.4–6.2 cm RMSE over five queries. On office1-7 at 15 inliers it reaches
  2.27 cm, ahead of the VIO's 4.06 cm. Details in
  `docs/openloris_map_localization.md`.

- **Local contrast equalization for maps that must localize under other
  lighting.** `visloc_basalt::contrast::LocalContrast` equalizes an
  image's local contrast (contrast-limited adaptive histogram equalization
  over 8 × 8 tiles; black pixels stay black).
  `basalt_euroc_online_slam_demo --equalize-local-contrast` applies it to
  each keyframe image before ORB extraction (the VIO tracks the raw
  images), `map.json` records it as `feature_local_contrast`, and
  `basalt_euroc_localize_demo` and `visloc_map_localize_node` apply it to
  every query image of such a map; `basalt_map_merge` refuses to merge an
  equalized map with a raw one. On OpenLORIS-Scene the daylight session
  office1-4, which no night map localized before, gets a map pose for
  33.6% of its frames at 2.46 cm RMSE (at `--min-inliers 15`; its best
  image has 23 inliers); the dawn session still gets none. On EuRoC it
  changes little (MH_02 and V2_02 within 0.1 cm) except V2_03 (29.6 to
  39.8 cm), so it stays off by default. Results in
  `docs/openloris_map_localization.md`.

- **Keyframe retrieval for the map-wide search.**
  `visloc_localization::KeyframeRetrieval` ranks a map's keyframes by
  hashed visual words of their landmarks' binary descriptors (Basalt's
  `HashBow` scheme, histogram intersection through an inverted index).
  `BinaryLandmarkIndex::match_among` matches among the landmarks of the best
  ones. `basalt_euroc_localize_demo --retrieval-keyframes N` tries that
  before the whole map. On the EuRoC queries it localizes 0.1–0.9 points
  more images. On a merged OpenLORIS map it localizes 4.6% of office1-6's
  images instead of 2.9%. On a Machine Hall map grown session by session
  to 41,000 landmarks (`docs/basalt_map_localization.md`, "Lifelong use"),
  it cuts MH_05's median time per image from 125 to 74 ms. Off by
  default.

- **Map-based localization on a real wheeled robot (OpenLORIS-Scene).**
  `scripts/openloris_to_euroc.py` converts an OpenLORIS-Scene sequence's
  T265 data to the EuRoC layout with a Basalt calibration:
  - the fisheyes are resampled to a Double Sphere camera;
  - the accelerometer is interpolated to the gyro;
  - the ground truth (`base_link`) is moved to the IMU.

  `basalt_euroc_localize_demo --odometry FILE [--imu-from-body ...]`
  corrects an external odometry, such as the robot's wheel odometry, into
  the map frame instead of the VIO.

  `docs/openloris_map_localization.md` reports the office scene, seven
  sessions over four days:
  - Sessions with the same lighting, up to 2.8 days later with furniture
    moved, get a map pose for 95–99.7% of frames at 2.8–4.1 cm with the VIO.
    With the wheel odometry they get 7–12 cm, because it drifts in in-place
    turns.
  - No session localizes across the day/night lighting change.
  - Maps of opposite driving directions merge through a session that turns
    in place.

- **README: map-based localization animation.** EuRoC MH_02 with the VIO
  corrected into the MH_01 map's frame: the map-frame trajectory (a pose for every
  frame), the keypoints that matched the map and the rays to their
  landmarks (`docs/assets/hero_map_localization.gif`). It is rendered by
  `scripts/render_map_localization_gif.py` (`--trajectory` for a `--vio`
  run) from `basalt_euroc_localize_demo --dump-inliers FILE`, which writes
  each accepted localization's PnP inliers (keypoint pixel and map
  landmark).

- **`basalt_euroc_localize_demo`: matching around the predicted projections
  with `--vio` (and `--odometry`), by default.** Once the VIO has a map
  pose, each image is first matched around the landmarks' projections
  from the pose the corrected VIO predicts, as `visloc_map_localize_node`
  does, with the map-wide search as the fallback (`--projection-radius-px`,
  20; `0` turns it off).
  - Over the eight EuRoC queries, localizing every 5th image, more images
    localize (MH_02 98.4% instead of 81.6%, V2_02 56.4% instead of 21.1%)
    in about half the time.
  - The VIO + map error falls on 6 of the 7 queries with map poses (V1_03 4.60 cm
    instead of 6.66). On MH_05 it rises from 14.4 to 15.9 cm.
  - The prediction is used only within 5 s and 2 m of odometry travel after
    the correction's newest fix (`--projection-fresh-s`, `--projection-fresh-m`).
    After a 27 s gap on MH_03, a drifted prediction had produced fixes
    16–42 cm off.

- **Live monocular-inertial VIO in a phone browser** (`bindings/wasm`). The
  Basalt port's VIO runs as WebAssembly on the phone's camera and motion
  sensors, on iPhone Safari and Android Chrome.
  - **Calibration per phone:** Client Hints model or iPhone screen family,
    with intrinsics from the 35 mm-equivalent focal length. The camera-IMU
    time offset is estimated online and saved per phone (−25 ms until
    measured).
  - **Watchdog:** a diverged estimate is restarted and re-anchored to the
    last good pose.
  - **Page:** AR overlay, map, wasm threads when cross-origin isolated, and
    session recording and upload.
  - **Telemetry:** the page reports clocks, frame and IMU flow, worker
    state, resets, stalls and errors to the serving host
    (`telemetry/<session>.jsonl`). The sync estimate also measures the
    camera-IMU rotation.
  - **iOS:** WebKit's gyroscope axis order (alpha / beta / gamma about x /
    y / z), corrected automatically from the camera where a phone
    disagrees. The main lens is chosen over the virtual multi-camera.
  - **IMU only:** a button that freezes the map and moves it by the motion
    sensors alone, to check the camera-IMU calibration by eye.
  - **Phone defaults:** a 28 px feature grid, a 1 cm triangulation
    baseline, a keyframe allowed every 2 frames with 2 non-keyframe states
    in the window, bias random walks 30x stiffer than the white noise's
    scaling, and the first attitude from 0.5 s of motion samples instead of
    one. On ADVIO, against the EuRoC file's values: mean ATE 2.6 m instead
    of 4.7 m, 2 resets in four walks instead of 8. On three iPhone
    recordings: 93 landmarks in view against 85 before the keyframe change.
  - **Still start:** each start and restart waits (up to 2 s) for the
    phone to be held still, judged from the tracked points' motion after
    the gyroscope's rotation, and takes its attitude from the still frames.
    Basalt assumes a start at rest; on 60 starts over iPhone recordings
    with the phone moving, 27% diverged within 3 s. The page only corrects
    the camera-IMU time offset by 15 ms or more (estimates swing by ±8 ms,
    and each correction restarts tracking). `replay` reports why the
    frontend dropped tracks.
  - **Coasting:** frames without a landmark after some were seen report
    `"coast"`, and the page keeps drawing the IMU's pose ("Coasting on
    IMU") instead of showing "Initializing". ⚙ → "Film at 60 fps" captures
    at 60 fps (half the longest exposure, so less blur in fast moves).
  - **Frontend on the GPU:** the pyramid, the frame-to-frame KLT and the
    grid FAST run as WGSL compute shaders with wgpu (WebGPU in the
    browser, Vulkan or Metal natively; ⚙ → "Track points on the GPU", on
    where WebGPU exists), one submission and one readback a frame; the
    estimator stays on the CPU. From the same tracks it keeps and loses the
    same points as the CPU frontend (4 differ in 28,356; positions agree
    to 0.002 px), and end to end ADVIO stays at 2.65 m. A GPU error falls
    back to the CPU frontend. `replay` and
    `advio` take `--gpu`; `examples/gpu_check.rs` compares the two. The
    worker runs the GPU frontend one frame ahead of the estimator, so the
    GPU's round trip overlaps the estimate (same poses).
  - **60 Hz:** filming at 60 fps and tracking every frame are on by
    default (⚙ → "Film at 60 fps", "…and track every frame at 60 Hz").
    Tracking every frame of a 60 fps camera kept more landmarks in view on
    two iPhone recordings than the same frames at 30 Hz (172 against 94)
    without a reset, once the IMU noise fix below is in. Recordings keep
    every camera frame (`replay --every 2` replays one at 30 Hz), the rig
    carries `frameRateHz`, and `--estimate-every 2` estimates every other
    frame.
    The session's watchdog, still start and coasting are timed in seconds
    instead of frames.
  - **Android phones:** profiles for the Galaxy Note20 Ultra / Note20 and
    the Nothing Phone (4a) Pro (24 mm). The page takes the lowest-numbered
    back camera (Chrome's default on the Note20 Ultra was the ultra-wide),
    defaults the gyroscope's axis order by browser (Chrome on Android now
    reports WebKit's order, Firefox the spec's), waits up to 10 s for the
    first motion events, and reports each frame's brightness and contrast.
  - **Lighter tracking for slower phones:** ⚙ → "Lighter tracking…" (on
    by default in Firefox, which has no WebGPU and took 68–120 ms a frame
    on Android) sets a feature grid of a tenth of the frame's short side, at
    most 3 solver iterations, and the estimator on every other frame. Natively
    it costs 2.4x less a frame on iPhone recordings, with 10 resets against
    15 but ADVIO's mean ATE at 3.7 m against 2.6 m.
  - **Overlay matched to the image:** the AR view is drawn at the pose of
    the image that will be on screen when it appears (a display refresh
    ahead, less the camera's measured latency to the page; ⚙ → "Overlay
    timing" fine-tunes it), carried forward from the last estimate by the
    gyroscope and velocity, the camera's lever arm included, and the
    tracked points, which belong to an older frame, are hidden unless asked
    for, then turned onto the frame on screen.
  - **ADVIO results** (iPhone, browser-like 60 Hz IMU): 1.9–4.1 m ATE over
    142–244 m walks, 98–99% of frames tracking (ARKit: 1.2–2.6 m).
  - **Speed:** about 14–17 ms a frame natively on one thread at 360 x 640.
    The page reports whether the phone offers WebGPU and the main
    thread's time to read a frame back.

- **Launch file, URDF and RViz config for the ROS 2 nodes.**
  `ros2/visloc-ros2/launch/map_localization.launch.py` starts
  `robot_state_publisher`, the VIO (`odom -> base_link`), the map
  localization (extrinsics from `/tf_static`) and optionally RViz with
  `launch/rviz/map_localization.rviz`. `tools/calib_to_urdf.py` writes a
  URDF from a Basalt calibration (`launch/euroc_rig.urdf` for EuRoC).
  `visloc_map_localize_node` publishes the map's landmarks once on
  `map_cloud` (`sensor_msgs/PointCloud2`, transient-local;
  `publish_map_cloud`). Tested with ROS 2 Jazzy: MH_02 through the launch
  file gives 99.6% of frames a map pose at 2.06 cm RMSE, and RViz shows
  the map, path, pose and TF tree.

- **Duplicate-tolerant ratio test.**
  `BinaryLandmarkIndex::with_duplicate_radius` makes the map-wide search
  compare the best landmark with the best one farther than a radius from
  it. Landmarks of one physical point (broken SLAM tracks, sessions of a
  merged map) then no longer reject each other's matches. Off by default.
  Enable it with `basalt_euroc_localize_demo --duplicate-radius-m` or
  `visloc_map_localize_node -p duplicate_radius:=...`. At 10 cm, MH_02
  against MH_01 localizes 83.1% of its images (81.2%), with the worst pose
  7.6 cm off instead of 16.8 cm. V2_02 against V2_01 merged with V2_03 goes
  from 18.4% to 19.3%.

- **Merging maps of several flights.** `basalt_map_merge --base A --add B
  --out AB` registers saved SLAM maps of the same place to the first one
  and writes one map in its frame. `visloc_localization::register_map`
  localizes the added map's keyframes in the base map and keeps the yaw +
  translation that at least 3 of them agree on;
  `visloc_localization::merge_maps` renumbers the added map, moves it and
  fuses its landmarks that coincide with base landmarks. The merged map
  loads like any saved map (`basalt_euroc_localize_demo`,
  `visloc_map_localize_node`); `map.json` lists the sessions and
  `images.csv` gains a `session` column. On EuRoC, MH_04 against MH_01
  merged with MH_05 localizes 55.8% of its images (4.6% against MH_01) and
  the corrected VIO gives 99.7% of frames a pose at 9.5 cm RMSE (73.1% at
  23.0 cm). An added map keeps its own error: MH_05 against MH_01 + MH_04
  gets 99.8% of frames (77.1%) and half the median error, but a worse tail
  where MH_04's part of the map is 25 cm off. Results and limits in
  `docs/basalt_map_localization.md`.

- **Ground robots (ROS 2): `base_link` and wheel odometry.**
  - `visloc_vio_node -p base_frame:=base_link` publishes `odom -> base_link`
    (REP-105) instead of `odom -> imu_link`: odometry, pose, path and `/tf`
    for the base, placed by the `/tf_static` transform between the IMU's
    frame (`body_frame`) and the base. `visloc_ros2::vio::rebase` moves an
    estimate, velocity included, to a frame rigidly attached to the IMU.
  - `visloc_map_localize_node -p extrinsics:=tf` takes the camera's
    position on the robot from `/tf_static` (image `frame_id` to the
    odometry's `child_frame_id`) instead of the map's IMU-from-camera, so
    any gravity-aligned odometry can be corrected into the map frame, e.g.
    wheel odometry on a level floor. New parameter `camera_frame`.
  - `visloc_ros2::tf_tree::StaticTransforms` composes static transform
    chains like tf2; the nodes subscribe to `/tf_static` transient-local,
    as `tf2_ros` does.
  - On ROS 2 Jazzy, with `static_transform_publisher` providing
    `base_link -> imu0 -> cam0`, EuRoC MH_02 against the MH_01 map: the VIO
    publishing `base_link` gives a map pose for 99.5% of frames at
    2.21 cm RMSE; simulated wheel odometry (98.7 cm ATE on its own) 98.3%
    at 3.45 cm.

- **The real-time ROS 2 map localization is validated on all eight EuRoC
  query sequences** (rosbag2 at 1.0x, Fast DDS): within 0.5 points of the
  offline `basalt_euroc_localize_demo --vio` on the share of frames with a
  map pose (V2_03: 77.5%, from 21 s on), with no VIO reset and at most 15
  dropped frames per sequence; table in `ros2/visloc-ros2/README.md`.

- **The ROS 2 nodes are tested against ROS 2 Jazzy** with both
  `rmw_fastrtps_cpp` and `rmw_cyclonedds_cpp`, on one host:
  - `ros2 topic` / `ros2 node` see the nodes and decode their messages.
  - `tf2_echo map imu_link` resolves the REP-105 chain.
  - EuRoC MH_02 played from a rosbag2 into the VIO and map nodes, with
    reliable or best-effort publishers, gets a map pose for 98.7-99.7% of
    frames at 2.06 cm RMSE.
  - New helper scripts, both run with a sourced ROS 2 environment:
    `ros2/visloc-ros2/tools/euroc_to_rosbag2.py` converts a EuRoC sequence
    to a bag, and `tools/bag_to_tum.py` writes a recorded pose topic as a
    TUM trajectory.

- **Real-time map localization (ROS 2).** `visloc_vio_node` and
  `visloc_map_localize_node` now keep up with EuRoC MH_02 replayed over DDS
  at real time against an MH_01 map, with all three processes on 4 cores.
  99.6% of frames get a map pose (was 90.0%), at 2.05 cm RMSE (5.2 cm max).
  The median latency from frame to `map_pose` is 49 ms (was 171 ms) and the
  90th percentile 90 ms (was 271 ms).
  - `visloc_vio_node` runs KLT tracking and the estimator on two threads, as
    upstream Basalt does: `BasaltVioEstimatorAdapter::into_stages` splits
    it into `BasaltVioFrontendStage` / `BasaltVioEstimatorStage`, and
    `VioCore::into_stages` into `VioFrontend` / `VioBackend`. The split
    gives the same estimates as the serial path (tested). New parameter
    `frontend_threads` (default 2).
  - Once the odometry has a map pose, `visloc_map_localize_node` predicts
    each image's pose from the corrected odometry. It then matches each map landmark only
    against the keypoints within `projection_radius` (20 px) of its
    predicted projection, and falls back to the map-wide search if that
    fails. Matching and PnP drop from about 50 ms to 4 ms per image, with
    twice the inliers. New parameters `projection_radius`, `prior_wait_ms`
    and `match_threads`; `/diagnostics` reports `matching`.
  - `visloc_localization::BinaryLandmarkIndex` packs a map's binary
    descriptors once. `match_all` and `match_projected` give the same
    correspondences as `CorrespondenceBuilder` /
    `ProjectionCorrespondenceBuilder` with a `HammingMatcher` (tested).
    `LocalizationPipeline::localize_correspondences` runs the pose
    estimation and quality gate on correspondences built elsewhere.

- **`visloc_map_localize_node` (ROS 2): robot-style localization in a saved
  SLAM map.** Run next to `visloc_vio_node`, it localizes every 5th camera
  image against a map from `basalt_euroc_online_slam_demo --save-map`,
  pairs each fix with the VIO odometry of the same stamp, and publishes
  `/tf` `map -> odom` (REP-105), `map_pose` (body pose in the map, position
  covariance 3 cm + 2.5 cm per metre since the last fix) and
  `/diagnostics`. EuRoC MH_02 replayed over DDS against an MH_01 map: a map
  pose for 99.8% of frames at 2.18 cm RMSE (5.6 cm max) at half speed;
  at real time on 4 cores the VIO node drops 9.5% of frames.
  `visloc_vision::matching::HammingMatcher` (binary descriptors as 0/1
  floats) is now shared by the node and the localize example.

- **Save a VI-SLAM map and localize against it.**
  `basalt_euroc_online_slam_demo --save-map DIR` saves the online
  mapper's final map: keyframe poses, ORB landmarks and their
  descriptors, as a COLMAP text model (`DIR/sparse`) with
  `landmark_descriptors.txt`, `images.csv` and `map.json` (IMU-from-camera
  extrinsics). The new `basalt_euroc_localize_demo` runs each image of a
  EuRoC sequence through the mapper's own ORB frontend, matches it to the
  map by Hamming distance, estimates the pose with P3P RANSAC, accepts it
  with at least 30 inliers (`--min-inliers`), and writes IMU poses in the
  map frame to `localization.tum`.
  - The Double Sphere cameras are saved as COLMAP `OPENCV_FISHEYE` with
    `k1..k4` fitted over the field of view (`fit_opencv_fisheye`; at most
    0.0003 px from the Double Sphere projection over the EuRoC cam0
    calibration's 57° half field of view), so the model also opens in
    COLMAP.
  - Library: `NfrMapper::localization_map` (`visloc-basalt`),
    `write_landmark_descriptors_txt` (`visloc-io`),
    `euroc::read_camera_manifest` / `read_camera_image`.
  - EuRoC, one map per environment (MH_01, V1_01, V2_01) queried with the
    other eight sequences: no accepted pose is more than 25 cm off; MH_02
    localizes 81.2% of its images at 1.8 cm RMSE against ground truth,
    and flights the map barely covers far fewer (MH_04: 4.6%). Below 30
    inliers wrong poses appear (at 20, up to 3.6 m off on V2_02).
    `docs/basalt_map_localization.md`, scored by the new
    `scripts/evaluate_map_localization.py` (`--per-frame --min-inliers`
    scores any threshold from one run).
  - `--vio --calibration FILE --config FILE` runs the Basalt
    stereo-inertial VIO on the query sequence and corrects it into the map
    frame with the fixes of every `--localize-every`-th image
    (`visloc_localization::OdomCorrection`: a gated, gravity-aligned yaw +
    translation offset). Every frame gets a map pose in
    `trajectory_map.tum`; `trajectory_map.csv` adds the time and distance
    since the last fix. MH_02: 99.8% of frames at 2.0 cm RMSE localizing
    every 5th image, at 0.83× real time on 4 cores; on all seven queries
    with map poses no pose within 2 m and 5 s of a fix is more than 12.6 cm off,
    while poses carried tens of metres between rare fixes reach decimetres.

### Changed

- **The labels' logits learn 4x faster: `--semantic-lr` (and
  `SemanticTraining::lr`) defaults to 0.01, was 0.0025.**
  - **Why.** Slow logits leave the cross-entropy to the opacities early on.
    Gaussians of the wrong class turn transparent and are pruned, and each
    pruned one is one fewer grown by the refine.
  - **Bridge_Out.** Steps 1,000–5,000 pruned 3.5x as many gaussians as
    without the labels. That left 9% fewer gaussians and 0.1 dB less PSNR
    (30.91–31.17 dB against 31.17–31.20 dB).
  - **Now.**
    - Bridge_Out: 239,402 and 241,035 gaussians, 31.31 and 31.37 dB, and
      95.0% of the held-out pixels (95.2% before).
    - Building_In: 32.10 and 32.27 dB (32.00 and 32.05), 91.2–91.3%.
    - MeetingRoom_In: 32.54–32.64 dB, 88.1–88.3%.
    - No floaters.
  - **Found by** capping a run without labels at the labelled runs' count
    (31.09 and 31.15 dB) and by training the logits alone (31.21 dB),
    docs/semantic_splat.md, "The logits' learning rate".
  - **For the classes, vote** on a splat trained with the labels rather
    than read its logits. On Bridge_Out the logits give thin classes to
    gaussians that lie partly on them: lamp posts' IoU is 13% from the
    logits against 23% voted.

- **`GpuMatcher::match_pairs` returns a `Result` (#469).** Breaking:
  `match_pairs` and `match_pairs_async` return
  `Result<Vec<Vec<DescriptorMatch>>, SiftGpuError>`. A lost device or a
  failed buffer mapping is `SiftGpuError::Readback`, and a pair whose
  outputs do not fit the device's storage binding limit is
  `SiftGpuError::PairTooLarge`; both panicked before. `VlfeatSiftGpu`'s
  `extract` and `extract_batch` return readback failures too. The photo and
  EuRoC pipelines, the `colmap` example, `unordered_sfm_demo` and the
  browser SfM match a chunk that fails on the GPU on the CPU instead.
- **`VlfeatSiftGpu` keeps its GPU buffers across image sizes (#453).** A
  change of image size used to allocate every buffer again, on nearly every
  frame of a rig whose cameras differ in resolution. The buffers are now
  kept and grown to the largest image, and only the dispatches are
  recorded again. On 40 interleaved rig images (3840x2160 and 1920x1536,
  NVIDIA L40S) the median time per image falls from 140 to 20 ms, with the
  same features and about the same peak memory (2860 MiB against 2852).
  `clear_plan_cache()` frees them.
- **The mapping ORB frontend is about four times faster, with the same
  output.** The corner detector (`detect_keypoints_mapping_with`) runs each
  filter along whole rows, reflecting only at their ends, and checks the
  minimum distance only against the chosen corners in the 3x3 cells around
  a candidate; the blur (`GaussianSmoothing`) sums its taps across a row at
  a time. Every sum keeps its order, so corners, descriptors and maps are
  the same to the bit (a test compares the detector with the direct
  computation). On a T265 image (848x800) with the blur and quality 0.003,
  detection, blur and description take 14 ms instead of 50 ms; on an EuRoC
  image detection takes 4 ms instead of 25 ms. The localize demo's median
  time per image falls from 28.9 to 14.5 ms on MH_02 against MH_01 and
  from 69.9 to 34.6 ms on office1-7 against office1-2's blurred map, with
  byte-identical trajectories. Times quoted elsewhere were measured before.

- **The IMU scale fits the gyro's bias.** A constant gyro bias bends every
  frame-to-frame rotation, and with it the camera-from-IMU fit and the
  frame-timing search.
  - `visloc_slam::estimate_imu_scale` now fits the bias after the first
    timing search, takes it off the gyro, and searches the timing again.
    This is `ImuScaleConfig::gyro_bias`, on by default; turn it off with
    `imu_metric_scale --no-gyro-bias`. The fitted bias is reported in
    `ImuScaleEstimate::gyro_bias`, the `gsplat_photos` log and
    `imu_scale.txt`.
  - On OverMaps360's 9d2fe9b1 the camera's gyro reads 1.8°/s off, which
    left a 1.18° rotation residual. The fit takes it to 0.05° and puts the
    frame timing back (drift 18.5 → 2.7 ms).
  - Over all 50 walks the bias is 0.07–1.69°/s, and the rotation residual
    falls from 0.03–0.60° to 0.01–0.05° on all walks but one. The scale
    moves by 0.06% in the median and 1.1% at most.
  - Against ARKit and GPS the agreement is unchanged (MAE 1.85% and 1.38%,
    from 1.86% and 1.34%; docs/insta360_imu_scale.md, "Gyro bias").

- **`gsplat_photos` frees the working images after SIFT, and `--insv`
  writes its lens faces to disk as it cuts them.**
  - Each SfM point takes its colour from its first keypoint, as before. The
    colours under the keypoints are now kept at SIFT time, so the decoded
    photos, cube faces or lens faces are freed after feature extraction
    instead of being held through SfM.
  - With `--insv` each frame is cut into its ten faces and written as it
    is read, and SIFT reads the faces back (PNG, so the same pixels).
  - The output is byte-identical for photos, `--mask-black`, `--equirect`,
    `--equirect-native` and `--insv` (100 frames).
  - On 4 cores and 15 GB, 100 frames with 1024 px lens faces peak at
    4.5 GB instead of 7.9 GB. 200 frames, killed at 14 GB before, finish
    at 8.6 GB.

- **`gsplat_photos` keeps VLFeat-compatible SIFT descriptors as bytes.**
  - They come out of SIFT as COLMAP's bytes, but were held as f32 vectors
    until matching ended, about 550 bytes a keypoint. They are now kept as
    bytes, 128 a keypoint.
  - They are turned into the same f32 values (`byte / 512` is exact) only
    where they are read: per chunk of pairs for the CPU matcher, per block
    for the GPU bank, per image for the retrieval, and before the
    intrinsics refinement.
  - The output is byte-identical for photos (with and without retrieval
    pairs), `--equirect` and `--insv`.
  - The mall walk's 200 frames with 1024 px lens faces peak at 4.5 GB
    instead of 7.8 GB.
  - At the defaults (1600 px faces, 8,192 keypoints; `--window 2`) the same
    frames peak at 9.5 GB, in 2 h 23 min on 4 CPU cores, most of it
    matching. On a CPU, `--max-size 1024 --max-keypoints 4096` is the
    practical setting (docs/photos_to_splat.md).

- **Weak fixes: localizations with 15–29 inliers can set the first map
  pose.** With `--vio` or `--odometry`, `basalt_euroc_localize_demo` passes
  a map-wide localization turned down only for too few inliers, but with at
  least `--weak-min-inliers` (15), to `OdomCorrection::add_weak_fix`.
  `visloc_map_localize_node` does the same (`weak_min_inliers`). Weak fixes
  count only toward setting the first offset, which then needs three
  agreeing fixes instead of two. Once it is set they are
  `FixOutcome::Unused`. Replaying all eight EuRoC queries through the rule,
  it never uses a fix more than 25 cm off. On OpenLORIS-Scene, office1-7,
  whose images never reach 30 inliers against another session's map, gets
  a map pose for 99.4% of its frames by default (6.86 cm RMSE). The
  daylight office1-4 gets one against an equalized night map (33.4%,
  2.62 cm), where before it needed `--min-inliers 15`. Rerun on all eight
  EuRoC queries, MH_04 gets its first map pose 2.6 s sooner (76.1% of the
  frames instead of 73.6%), V2_03 improves from 29.6 to 23.7 cm, and the
  other six change by at most 0.1 cm.

- **`OdomCorrection` measures travel along the odometry's path.** The gate
  widens with the odometry's travel since the last accepted fix, which was
  measured only between the odometry positions of consecutive fixes. After
  a long gap that is one straight line, and a correct fix on EuRoC V2_03, 65
  s after the previous one, was rejected. `OdomCorrection::observe_odometry`
  now takes every odometry pose and keeps the path length by stamp, so it
  also works when the odometry runs ahead of the localizations. The demo
  and `visloc_map_localize_node` feed it every pose. Replayed on EuRoC
  without weak fixes, it takes V2_03 from 29.1 to 24.6 cm and leaves the
  other seven queries unchanged.

- **Map localization tries every image until the odometry has a map
  pose.** No frame has one before two fixes agree, and with every 5th
  image tried, EuRoC V2_03 never got two. `basalt_euroc_localize_demo`
  (`--vio`, `--odometry`) and `visloc_map_localize_node` now try every image
  until the first map pose, then every `--localize-every`-th /
  `localize_every`-th. V2_03 gets a map pose for 77.8% of its frames, from
  21 s on (29.6 cm RMSE; 5.8 cm within 5 s and 2 m of a fix). On the other
  seven queries the first map pose comes up to 1.2 s sooner and the
  accuracy is unchanged; MH_04 and MH_05, whose starts the map does not
  cover, try about twice as many images. After 30 s without any
  localization (`--initial-window-s` / `initial_window_s`), where the map
  likely does not cover the place, they back off to every
  `localize_every`-th image until one localizes. `--initial-localize-every`
  / `initial_localize_every` (default 1) sets the start-up cadence; 5 is
  the previous behaviour.

- **`visloc_map_localize_node`: the projection prior expires.** The node
  matches around the predicted projections only within `projection_fresh_s`
  (5) seconds and `projection_fresh_m` (2) metres of odometry travel after
  the correction's newest fix. After a longer gap it matches map-wide, since a
  drifted prediction can steer the projection search to a wrong pose that
  agrees with it.

- **visloc-basalt builds and runs on wasm32-unknown-unknown.** Natively
  everything is identical, and the crate's tests pass unchanged.
  - The libstdc++ bucket arithmetic is width independent.
  - The diagnostic probes skip the environment and process id where there
    are none.
  - `mul_add` goes through a `fused_mul_add` that inlines on wasm. The
    software `fmaf` it replaced took 40% of a frame.
  - `BasaltVioEstimator::landmark_positions_world` exposes the window's
    landmarks for display.
  - `BasaltVioEstimatorAdapter::process_tracks_no_trace` runs the estimator
    on tracks from another frontend, and `DirectKltStream::cam0_tracks`
    exposes the tracks' patch transforms.
  - `config.vio_imu_noise_per_step` (opt-in, default off): the IMU noise is
    discretized by each preintegration step's length. Basalt injects the
    per-sample noise into every step, which undercounts the short steps at
    each frame interval's edges: a third of the covariance with one IMU
    sample per frame (60 Hz IMU at 60 fps), whose overconfident IMU took
    ADVIO at 60 fps from 2.65 m to 7.0 m. With it, 4.3 m and 2 resets
    instead of 13; at 30 Hz neutral. A test checks the covariance no
    longer depends on where an interval's samples fall.

- `basalt_euroc_localize_demo` matches map-wide through
  `BinaryLandmarkIndex`: the same matches as before (byte-identical
  output on MH_02), at a median 67 ms per image instead of 86 ms.

- **Basalt KLT frontend about 38% faster, bit-identical.** The five bilinear
  samples of an interpolated gradient now share one neighbourhood read. FAST
  replenishment converts each grid cell to u8 once and rejects pixels with a
  cardinal-pixel test before the full ring test (2.6 to 0.6 ms per frame).
  Each frame's backward searches already sample the source patches the
  next frame's forward searches need, so those patches are cached by camera
  and exact position and reused. MH_02 on 4 cores: 12.3 to 7.6 ms per
  frame, with a byte-identical trajectory.
- **Basalt VIO estimator about 26% faster, bit-identical.** The LM
  iteration's linearization and its model cost decrease (on the retained
  QR payload path) evaluate landmarks in parallel and combine them in the
  original order. MH_02 on 4 cores: 30.0 to 22.1 ms per frame (115 to 76 ms
  at worst), with a byte-identical trajectory.
- `visloc_vio_node` runs 3 LM iterations per frame by default (new
  parameter `max_iterations`, overriding `config.vio_max_iterations`, which
  the EuRoC config sets to 5). On eight EuRoC sequences the VIO's ATE moved
  by at most 0.13 cm, and the estimator got 15-30% faster. `0` keeps the
  config's value.
- `visloc_vio_node`: `frame_queue` defaults to 1 (was 4), and the frontend
  hands frames to the estimator without buffering. When the estimator falls
  behind, the node drops older frames before tracking instead of building
  a backlog. `visloc-ros2` builds `visloc-basalt` with
  `basalt-lm-workspace-reuse`.
- `MapLocalizer` (`visloc-ros2`): `localize(image, prior)`; the new
  `extract` / `features` produce `ImageFeatures` for
  `localize_features(&features, prior)`; `Fix::matching` names the search
  used.

## 0.3.1 - 2026-10-25

Patch release: `--help` on the release binaries, fewer cameras left far off
by `gsplat_photos`' intrinsics refinement, and prebuilt binaries attached
to every release.

### Fixed

- **`--help` on the release binaries' command-line tools.** `visloc_mvs`
  (the `mvs` example), `gsplat_photos` and `colmap_incremental_mapper`
  print their options with `-h` / `--help`. `visloc_mvs` panicked on it
  ("unknown argument --help"), `gsplat_photos` stopped with the same error
  and the mapper panicked on the missing `--manifest`. A missing or unknown
  argument now prints the message (with the usage for `visloc_mvs` and the
  mapper) and exits with status 2 instead of a panic and a backtrace.

### Changed

- **`gsplat_photos` damps its intrinsics refinement from the start.** The
  joint pose + intrinsics BA after the mapper starts Levenberg–Marquardt at
  λ = 1e4 instead of 1e-4. The damping is absolute and a photo set's
  Hessian diagonals are 1e4–1e8, so 1e-4 was no damping: its first steps
  swung cameras by up to a radian and were rejected until λ reached that
  scale.
  - **Three 500-image SmallCity pass-1 subsets**, camera centres against
    the release's COLMAP model (Sim(3) refit on errors under three times
    the median), median / p90 / max:

    | Subset | before | after |
    |---|---|---|
    | 0–499 | 13.6 / 22.6 / 282 cm (4 cameras over 1 m) | 12.1 / 17.3 / 23 cm (none) |
    | 500–999 | 23.8 / 29.2 / 66 cm | 24.9 / 30.8 / 69 cm |
    | 1000–1499 | 8.6 / 16.1 / 22 cm | 7.9 / 15.2 / 21 cm |

  - **ETH3D DSLR, all 13 training scenes** (one camera per scene, 1,600 px,
    the focal started 5% off, ground-truth poses): the same images register
    and the scores barely move. Means over the scenes before → after:
    relative rotation 0.084° → 0.085°, camera-centre median 0.76 → 0.79 cm,
    p90 1.38 → 1.42 cm, maximum 3.99 → 3.86 cm. meadow is the largest
    change either way (median 4.0 → 4.5 cm), then facade (1.71 → 1.61 cm),
    office (maximum 6.4 → 4.3 cm) and playground (0.26 → 0.30 cm); seven
    scenes are unchanged.
  - **Why the first subset needed it:** the mapper starts from images 66
    and 324, a revisit 3.5 m apart, because consecutive photos are too
    close for COLMAP's 16° initial-pair angle. Its model leaves image 66
    32 m off and images 319–352 bent by up to 29 m; the refinement pulls
    them back, and with the damping it no longer stops halfway. Starting
    from nearby photos instead (an 8° initial-pair angle, or a pair at most
    30 photos apart) avoids the bend but leaves a 18 cm median on that
    subset: the revisit pair also closes the loop, so neither is used.

### Added

- **Prebuilt binaries on every release** (`.github/workflows/release.yml`).
  A `v*` tag push creates the GitHub release from that version's CHANGELOG
  section and attaches `gsplat_photos`, `visloc_mvs` (the `mvs` example),
  `colmap_incremental_mapper` and `visloc_web`, built with their GPU
  features, for Linux x86_64 (built on Ubuntu 22.04, glibc 2.35), Windows
  x86_64 and macOS on Apple silicon, as `.tar.gz` / `.zip` with SHA-256
  sums. A manual run (`workflow_dispatch`, input `tag`) builds an existing
  tag, e.g. `v0.3.0`.

## 0.3.0 - 2026-10-24

### Highlights

- **Multi-view stereo: `visloc-mvs` (new crate).** Dense point clouds and
  meshes from a COLMAP model, on the CPU or on the GPU through wgpu. On
  the ETH3D high-res training set (13 scenes at 3,200 px, one NVIDIA L4)
  F1 at 2 cm is 83.7%, against the 83.42% ACMMP reports; against
  ACMMP's released code on the same L4, 83.4% against 83.2% over six
  scenes, in 27.5 minutes against 110. From the photos end to end,
  76.3% against 57.4% for COLMAP 4.2.1 CUDA over the 10 scenes both
  finished, in 58 minutes against 372.
- **City-scale photos to splats.** The README hero is Hierarchical 3DGS
  SmallCity: 5,822 photos, 5,583 registered, 4.7M gaussians and a
  5.9M-triangle mesh from one `gsplat_photos` command in 2.5 h on a Colab
  A100. Along the way: VLFeat-compatible SIFT on the GPU (SmallCity's
  photos in 43.8 s on an RTX 3090), optional GPU bundle adjustment, a
  converged final BA, completion of the largest model, and a COLMAP-port
  mapper more than twice as fast with about half the memory (SmallCity
  peak 9.0 GiB instead of 18.9 GiB).
- **SfM ahead of COLMAP 4.2.1 on ETH3D.** On COLMAP's matches of the 13
  DSLR training scenes the mapper scores a mean AUC@1°/3°/5° of
  53.69 / 79.94 / 86.90 (452 of 454 images), against 51.34 / 76.25 /
  82.88 for COLMAP's incremental mapper and 48.65 / 70.41 / 76.36 for
  its global mapper.
- **360° panoramas.** `gsplat_photos --equirect` (the six cube faces of a
  panorama as one rig) and `--equirect-native` (COLMAP 4.1's
  `EQUIRECTANGULAR` camera); `gsplat_train` trains on equirectangular
  COLMAP models.
- **New ways in.** `visloc_web` runs the photo pipeline from a browser;
  Python bindings (`bindings/python`, package `visloc`); ROS 2 nodes
  (`ros2/visloc-ros2`) for the VIO and localization.
- **Sensors and fusion.** Multi-camera Basalt VIO for divergent rigs
  (opt-in), monocular-inertial VIO (experimental), and a joint GNSS +
  visual-odometry pose graph.
- **Lens models in bundle adjustment.** Pose/structure BA and rig BA use
  each camera's full distortion model; tangential distortion can be
  self-calibrated (opt-in).

### Full change log

The itemized 0.3.0 change log (every Added / Changed / Fixed entry from
2026-10-08 to 2026-10-24, about 1,400 lines) is archived in
[`docs/archive/changelog_0.3.0_full.md`](docs/archive/changelog_0.3.0_full.md).

## 0.2.1 - 2026-10-08

Patch release: camera-distortion correctness fixes. Each fix below corrected
results that were silently wrong, not just errors that were reported.

### Fixed

- **COLMAP camera export with self-calibrated distortion.** A `Pinhole`
  camera carrying the refined radial `[k1, k2]` tail
  (`--refine-distortion`) was written as `PINHOLE` with 6 params, which is
  not valid COLMAP (`PINHOLE` has 4); COLMAP rejects it and other readers
  silently drop the distortion. Every COLMAP writer (text and binary) now
  exports it as `OPENCV` with `p1 = p2 = 0` (the identical projection), and
  fallible writers reject any camera whose parameter count does not match its
  COLMAP model. New public helper `visloc_io::colmap::colmap_camera_record`.
- **OPENCV tangential distortion.** `Camera::project` / `normalize_pixel` /
  `unit_ray_from_pixel` ignored `p1, p2` of an `OpenCv` camera, so a COLMAP
  `OPENCV` model with non-zero tangential terms was silently treated as
  radial-only. Such cameras now use the full Brown-Conrady model (shared with
  `FULL_OPENCV`, k3..k6 = 0); results with `p1 = p2 = 0` are bit-identical to
  before. New `Camera::tangential_distortion`.
- **Pose/structure BA ignored self-calibrated radial distortion.** With
  `refine_intrinsics = false` (e.g. the SfM polish / warm-start passes after
  `--refine-distortion`), `optimize_weighted` built its residuals and
  Jacobians with an undistorted pinhole while the cost it accepts steps
  against used the distorted `Camera::project`, so LM optimised a different
  model than it measured and could stall. The monocular residual and Jacobian
  (serial, parallel, and dense-QR assembly, plus the GNC residual vector) now
  apply `1 + k1·r² + k2·r⁴` with an analytic Jacobian; distortion-free cameras
  keep the historical closed form bit-for-bit.

### Added

- **`BaConfig::shared_focal` / `--shared-focal`** (unordered and sequential
  SfM demos): constrain intrinsics self-calibration to `fx == fy`. Previously
  `fx` and `fy` were always refined independently, so weakly observable
  motion (near-pure forward translation) could settle on a spurious aspect
  ratio. Default off; the joint solve is unchanged when it is not set.

## 0.2.0 - 2026-09-15

### Highlights

- **Real-data SfM at scale.** All ten ETH3D low-resolution many-view scenes:
  **9,996/10,008 cameras registered (99.88%)**, with no mapper run above
  3.32 GiB. Same-input ETH3D Electro 1,200 runs **3.46× faster than COLMAP**
  with **25.2% lower** camera-centre RMSE, and the 38-image courtyard control
  reaches **0.5379 cm vs COLMAP's 1.6166 cm**.
- **Visual-Inertial SLAM (faithful Basalt Rust port).** A tightly-coupled
  stereo-inertial VIO estimator plus offline mapper that matches native
  Basalt's ATE within **0.1%** on all 11 EuRoC sequences and, with EuRoC's
  official calibration, beats measured ORB-SLAM3 on **8 of 11** sequences.
- **COLMAP-compatible unordered SfM** with VLAD retrieval, essential-matrix
  verification, incremental mapping, connected-component models, and COLMAP
  text export for downstream 3DGS / MVS.
- **Online stereo SLAM** with loop closure, SE(3)/Sim(3) pose-graph
  optimization, GNC outlier rejection, and bundle adjustment, benchmarked on
  KITTI, EuRoC, and TUM RGB-D.
- **Opt-in deep frontend** (SuperPoint / LightGlue via in-Rust ONNX Runtime,
  CUDA-accelerated), plus public-data map-reuse localization and GNSS-prior
  tracking demos.

### Full change log

The itemized 0.2.0 change log (every Added / Changed / Fixed entry from
2026-05-07 to 2026-09-15, about 4,800 lines) is archived in
[`docs/archive/changelog_0.2.0_full.md`](docs/archive/changelog_0.2.0_full.md).

## 0.1.0 - 2026-05-07

### Added

- Workspace split into core, vision, IO, localization pipeline, and tracking pipeline crates.
- Core visual localization types: `Frame`, `Keyframe`, `VisualMap`, `Landmark`, `Observation`, `Camera`, `Pose`, and `LocalizationResult`.
- `SO3` / `SE3` pose wrappers and reprojection utilities built on `nalgebra`.
- Brute-force descriptor matching with L2 distance, ratio test, optional cross-checking, and match diagnostics.
- Minimal DLT PnP estimator, PnP RANSAC, pose-estimation diagnostics, and optional Gauss-Newton pose refinement.
- COLMAP text and binary map parsers for `cameras`, `images`, and `points3D`.
- Text parsers for landmark descriptors and query features.
- Localization pipeline over query descriptors and visual-map landmarks.
- Map providers, submap selectors, priors, localization quality gates, and map validation reports.
- Tracking skeleton with motion models, state transitions, and sequence examples.
- Local mapping skeleton with keyframe policy, local map windows, landmark candidates, linear triangulation, staged map updates, and local refinement hooks.
- Online SLAM MVP composition that combines tracking and local mapping without loop closure or global optimization.
- COLMAP text model writer for saving reusable sparse maps.
- Sensor-fusion foundation crate with timestamped frames/poses, GNSS/pose/IMU measurements, covariance types, measurement buffers, frame prior sources, and external localization-prior tracking hooks.
- GNSS-prior tracking example showing radius-submap narrowing before localization.
- COLMAP compatibility notes covering supported sparse model inputs, descriptor handling, writer behavior, and current limitations.
- Root crate prelude and top-level re-exports for common application-facing localization APIs.
- Pre-1.0 to v1.0 migration guide covering recommended imports, localization boundaries, COLMAP descriptor handling, tracking priors, and experimental layers.
- Package metadata and crate-content checks in the local quality gate and CI.
- crates.io package metadata now includes project homepage and repository URLs.
- Workspace member crates now use crate-specific descriptions and docs.rs URLs.
- Publishing guide documenting workspace crate publish order and package-check workflow.
- Examples, integration tests, design docs, local check script, and GitHub Actions CI.

### Not Yet Implemented

- Full Visual SLAM.
- Full SfM.
- Loop closure.
- Dense mapping.
- Full bundle adjustment.
- Full tightly-coupled visual-inertial or GNSS/INS fusion.
