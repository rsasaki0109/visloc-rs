# Learned retrieval for relocalization (7-Scenes)

Relocalization — placing a single query image into a prebuilt map with no
temporal or odometry prior — is gated by *appearance retrieval*: the query must
first find the handful of map keyframes it could match against. That retrieval
is the binding constraint here (unlike VO loop closure, where the candidate is
easy to find and the geometric verifier is the lever). So a better global
descriptor should translate directly into a higher localization rate.

`examples/relocalization_7scenes_demo.rs` gates retrieval with a hand-built
`normalized_mean` (the L2-normalized mean of a frame's SuperPoint descriptors, a
bag-of-features centroid). This benchmark swaps in a **learned** global
descriptor — EigenPlaces, run in-process through the same ONNX the Rust
`GlobalDescriptorOnnxExtractor` loads — via `--global-descriptor-dir`, and
measures the difference. Everything downstream (per-keyframe PnP, the SuperPoint
features, the thresholds) is identical, so the only variable is the retrieval
descriptor.

The dataset is 7-Scenes `chess` (train sequences 1/2/4/6, test 3/5). EigenPlaces
is trained on outdoor Street View imagery, so this is also a cross-domain
transfer test.

## Retrieval recall

Isolating the retrieval stage: for each test query, is a geometrically correct
train keyframe (camera centre within 0.3 m) ranked in the descriptor's top-K?
(`scripts/eval_relocalization_recall.py`, 200 train keyframes, 49 valid queries.)

| recall@K | normalized_mean (incumbent) | EigenPlaces (learned) |
|---------:|:---------------------------:|:---------------------:|
| @1       | 42.9% | **57.1%** |
| @5       | 69.4% | **75.5%** |
| @10      | 79.6% | **87.8%** |
| @20      | 85.7% | **89.8%** |

The learned descriptor wins at every K; the gap is largest at @1 (+14.2 pts),
exactly where a tight retrieval gate operates.

## End-to-end localization

Running the full relocalization pipeline (retrieve top-K, per-keyframe PnP, keep
the best-inlier pose) with each gate, over a sweep of the retrieval depth K:

| `--retrieve-topk` | localized (baseline / learned) | within 5 cm / 5 deg (baseline / learned) |
|:-----------------:|:------------------------------:|:----------------------------------------:|
| 1  | 35.0% / **60.0%** | 27.5% / **35.0%** |
| 3  | 55.0% / **72.5%** | 30.0% / **35.0%** |
| 5  | 65.0% / **82.5%** | 30.0% / **42.5%** |
| 10 | 82.5% / **87.5%** | 42.5% / **47.5%** |

The learned descriptor's edge is largest when the gate is tightest: at top-1 it
nearly doubles the localization rate (35% to 60%), because it ranks the true
keyframe first far more often. As K grows both gates eventually retrieve the
correct keyframe, so the gap narrows — i.e. the learned descriptor reaches a
given recall at a *lower* candidate budget (less per-query matching work).

This is the converse of the KITTI seq02 VO loop-closure finding (see the
`--loop-min-inlier-ratio` flag and its commit), where retrieval is *not* the
bottleneck and a learned descriptor changes nothing end-to-end: the lever there
is geometric verification. Learned place-recognition pays off precisely where
retrieval is the binding constraint, and relocalization is that regime.

## Reproduce

```sh
DS=~/datasets/7scenes/chess

# 1. SuperPoint features (for PnP) and EigenPlaces globals (for the gate).
scripts/export_vpr_onnx.py --out models/eigenplaces_r50_2048.onnx
scripts/export_superpoint_7scenes.py --dataset $DS --seqs 1,2,4,6 --stride 20 --out-dir /tmp/sp_7scenes_chess
scripts/export_superpoint_7scenes.py --dataset $DS --seqs 3,5   --stride 50 --out-dir /tmp/sp_7scenes_chess
scripts/export_vpr_globals_7scenes.py --dataset $DS --seqs 1,2,4,6 --stride 20 --out-dir /tmp/vpr_7scenes_chess
scripts/export_vpr_globals_7scenes.py --dataset $DS --seqs 3,5   --stride 50 --out-dir /tmp/vpr_7scenes_chess

# 2. Retrieval recall (isolated):
scripts/eval_relocalization_recall.py --dataset $DS

# 3. End-to-end loc-rate A/B (per-keyframe PnP):
EX=target/release/examples/relocalization_7scenes_demo
cargo build --release --example relocalization_7scenes_demo --features image-io
$EX --dataset $DS --sp-features-dir /tmp/sp_7scenes_chess --retrieve-topk 5             # baseline
$EX --dataset $DS --sp-features-dir /tmp/sp_7scenes_chess \
    --global-descriptor-dir /tmp/vpr_7scenes_chess --retrieve-topk 5                    # learned
```

The SuperPoint export needs a Python env with `lightglue`; the EigenPlaces
export needs `torch` + `torchvision` and the cached EigenPlaces weights (see
`scripts/export_vpr_onnx.py`). Model `.onnx` files are git-ignored.
