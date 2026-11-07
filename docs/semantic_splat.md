# Semantic splats: SAM 3 classes on the gaussians

A trained splat can carry a class per gaussian, and an object, voted from
per-pixel labels of its photos:

1. **Label the photos.** `scripts/sam3_labels.py` runs
   [SAM 3](https://huggingface.co/facebook/sam3) on each training image with
   a list of text prompts.
2. **Vote.** `gsplat_semantic_cuda` renders every view and adds each splat's
   blending weight to the class of the pixel it lands in. A gaussian takes
   the class with the most weight.
3. **Objects.** With `--instances`, each photo's SAM 3 instances vote the
   same way. The instances of one class that share gaussians across photos
   merge into one 3D object ([Objects](#objects)).
4. **Look.** The web viewer (`docs/euroc_splat`) blends the splat's colours
   into the classes' or the objects' with a slider.

<p align="center">
  <img src="assets/semantic_meetingroom.jpg" alt="FIORD MeetingRoom_In from two viewpoints: the trained splat in its colours (left), painted by the SAM 3 class of each gaussian (middle: table red, chairs green, floor yellow, ceiling blue, walls orange, windows purple, doors cyan, whiteboards magenta, ceiling lights lime), and by object (right: each chair, table, window and light its own colour)" width="900">
</p>

<p align="center"><sub>FIORD's MeetingRoom_In (CC BY 4.0), 106 photos of a 200° fisheye: the splat (left), its gaussians painted by class (middle) and by object (right), in the web viewer.</sub></p>

```bash
# 1. SAM 3 labels: <run>/labels/<image>.png, classes.txt and instances/
python scripts/sam3_labels.py --images run/images --out run/labels --instances \
    --prompts "table,chair,floor,ceiling,wall,window,door,whiteboard,ceiling light,projector"
# 2. The votes: semantic.ply, and the web viewer's files in run/web
gsplat_semantic_cuda --ply run/scene.ply --data run --labels run/labels \
    --out run/semantic.ply --web run/web --instances
# 3. Serve a directory holding docs/euroc_splat and run/web, then open
#    euroc_splat/index.html?url=/web/scene.splat&semantic=/web/semantic.splat&instances=/web/instances.splat&cameras=/web/cameras.json
```

`run` is a `gsplat_photos` output trained by `gsplat_train_cuda` (any COLMAP
model with its images and a `scene.ply`).

## Labels from SAM 3

`sam3_labels.py` takes SAM 3 through Hugging Face transformers
(`Sam3Model`). It needs torch, a transformers with SAM 3, and access to the
gated `facebook/sam3` checkpoint (`hf auth login`).

- **Prompts.** SAM 3 finds every instance of each prompt. A pixel takes the
  prompt of the highest-scoring instance that covers it, and class 0,
  "other", where none does. The image's features are computed once, each
  prompt's text once.
- **No label.** Where the image's alpha masks it out, the pixel gets 255
  and casts no vote: past `--max-angle` and under a `--camera-mask`, as
  `gsplat_photos` wrote them.
- **Output.** `<out>/<image name>.png` (8-bit grey, the class index) and
  `<out>/classes.txt` (`index r g b name` per line). `--preview` also writes
  each photo with its classes over it.
- **Upright.** SAM 3 reads the scene from the image's orientation. FIORD's
  photos are upside down, and without `--rotate 180` SAM 3 labels their
  floor as ceiling and their ceiling as floor. `--rotate` turns each image
  for SAM 3 and turns the labels back.
- **Instances.** `--instances` also writes each pixel's instance, the one
  that gave its class: `<out>/instances/<image name>.png` (0 none, i the
  image's i-th instance, 255 no label) and `.txt` (`id class score` per
  instance, best first).

On an 8 GB laptop GPU in bf16, ten prompts take 0.75 s per 800 px image.

## The vote

`CudaGsplat::vote_labels` renders a view and runs the compositing loop
again: at each pixel with a class, each splat's blending weight there
(`T_i a_i`, the weight its colour gets in the render) is added to that
splat's tally for the pixel's class. The tallies sum over all views on the
device; `votes` copies them back.

- **Through the lens.** The projection is the renderer's, so a fisheye's
  views vote through their lens, past 90° too.
- **The class.** `semantic::label_gaussians` gives each gaussian its class
  with the most weight. Its confidence is that class's share of the
  gaussian's weight. A gaussian no labelled pixel saw gets class 255.
- **Tested.** Against the CPU f64 oracle, with one splat white and the rest
  black, so the oracle's render is that splat's weight per pixel. The
  votes agree to 1.7e-5 over a pinhole and fisheyes
  ([gsplat_cuda.md](gsplat_cuda.md#against-the-cpu-oracle)).

`--out` gets the splat painted by class: the class colour as degree-0 SH,
the geometry and opacity unchanged. It has extra float properties, which
readers that do not know them skip: `class` and `class_confidence`, and
with `--instances` `instance` (-1 for none). `--web <dir>` writes the
viewer's files:

- `scene.splat`, `semantic.splat` and with `--instances` `instances.splat`
  (each object its own colour), the same `--max` gaussians in the same
  order.
- `semantic.json` and `instances.json`, the legends.
- `cameras.json`, the capture viewpoints.

## MeetingRoom_In

The splat is the front lens of FIORD's MeetingRoom_In, with the lens
refined and the tripod masked out
([fisheye.md](fisheye.md#fiord-a-200-insta360-through-its-factory-calibration)):
106 photos at 800 px, 88,309 gaussians.

- **SAM 3:** the ten prompts above with `--rotate 180`, 89 s for the 106
  photos. 20% of the labelled pixels are "other".
- **The vote:** 0.5 s for the 106 views. Every gaussian is seen by a
  labelled pixel. The mean confidence is 0.80.
- **Gaussians by class:** 44% are "other", twice its share of pixels. The
  floor, ceiling and walls are covered by few large gaussians: their median
  volume is 60 times that of the "other" gaussians.

`--check` holds out every 8th view, as training did, and lets only the
other 92 vote. It then renders the painted splat at the 14 held-out views
and compares each pixel's class with SAM 3's label of that photo. A pixel's
class is the class colour nearest its render.

| | Pixel accuracy | mIoU (11 classes) |
|---|---|---|
| Held-out views | 82.6% | 71.3% |

| Class | IoU | Class | IoU |
|---|---|---|---|
| table | 91.6% | window | 31.3% |
| chair | 75.9% | door | 76.8% |
| floor | 83.6% | whiteboard | 94.4% |
| ceiling | 87.9% | ceiling light | 34.3% |
| wall | 63.4% | projector | 92.5% |
| other | 52.0% | | |

Windows and ceiling lights score lowest. These numbers measure agreement
with SAM 3's labels of the held-out photos, not with a ground truth.

## Objects

SAM 3's instances are per photo: the third chair in one photo is not the
third in the next. `--instances` joins them into 3D objects through the
gaussians they share.

- **Members.** Each photo's instance image votes on its own
  (`vote_labels` with the photo's instances as classes, 0 for the pixels of
  none). An instance's members are the gaussians that give it at least half
  of their weight in that photo (`semantic::view_instance_members`).
- **Merging.** Two instances of one class in different photos join when
  their members overlap: each instance's weights normalised to sum 1, the
  sum over shared gaussians of the smaller weight is at least
  `--instance-overlap` (0.5). Joins chain (union-find).
- **Objects.** A gaussian takes the object its memberships give the most
  weight, counting only objects of its own class. Objects seen in fewer
  than `--min-views` photos (3) are dropped (`semantic::merge_instances`).

On MeetingRoom_In the 106 photos hold 3,728 instances, 38 per photo
(median). They merge into 65 objects in 2.6 s: 25 chairs, 6 tables, 12
windows, 9 ceiling lights, 4 walls and others. 56.5% of the gaussians
belong to one: nearly all but the "other" ones (43.5%).

`--check --instances` merges the 92 training views' instances (62 objects)
and compares them with each held-out photo's instances, in weight. An
instance and an object overlap by the weight the instance's pixels give the
object's gaussians, against the weight of the instance's pixels plus that
of all pixels on the object's gaussians (a soft pixel IoU).

| `--instance-overlap` | Objects | Chairs | Mean best IoU | Instances matched at 0.5 | Objects shown that match |
|---|---|---|---|---|---|
| 0.1 | 43 | 15 | 41.8% | 43.2% | 58.5% |
| 0.3 | 56 | 22 | 50.5% | 59.2% | 60.7% |
| **0.5** | 62 | 26 | **52.8%** | **64.4%** | 59.3% |
| 0.6 | 70 | 29 | 51.6% | 61.0% | 51.0% |
| 0.8 | 139 | 64 | 43.4% | 43.0% | 19.5% |

- **The 525 held-out instances.** Each is scored against its best object.
- **Objects shown.** These are the objects whose gaussians take 50 pixels'
  weight or more in a held-out photo; each matches if some instance there
  overlaps it by 0.5.
- **The threshold.** A lower one merges more (15 chairs at 0.1), a higher
  one splits more (64 at 0.8). Both lose matches; 0.5 matches the most.

At 0.5, by class:

| Class | Held-out instances | Mean best IoU | Matched at 0.5 |
|---|---|---|---|
| chair | 221 | 54.6% | 68.8% |
| ceiling light | 92 | 52.8% | 66.3% |
| table | 51 | 62.0% | 74.5% |
| window | 50 | 19.3% | 8.0% |
| wall | 39 | 47.4% | 53.8% |
| whiteboard | 20 | 65.0% | 85.0% |

Windows rarely match. As for the classes, this is agreement with SAM 3's
instances, not a ground truth.

## The viewer

`docs/euroc_splat/index.html` (antimatter15's WebGL viewer) takes these
parameters:

- `?url=scene.splat&semantic=semantic.splat` loads both splats. A slider,
  or the M key, blends the scene's colours into the classes'.
- `&instances=instances.splat` adds the objects; O switches the overlay
  between classes and objects.
- `&blend=1` starts on the overlay.
- The legend comes from the `.json` beside each overlay: the share of the
  splats per class, or the objects per class.
- `&cameras=cameras.json` starts at the first capture viewpoint, and the
  number keys step through the captures.

## Not done

- **The labels in training.** The classes and objects come after training
  and do not shape the gaussians. A gaussian that straddles two objects
  takes one.
