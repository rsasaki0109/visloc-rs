# Multi-session lifelong mapping (7-Scenes)

A lifelong map is built once and then **grown across later visits**: each new
traversal of the scene is folded into the existing map without re-surveying it.
The only thing that lets a later keyframe attach is *appearance retrieval* —
the keyframe must find a relevant prior keyframe to relocalize against — so, as
in single-shot relocalization, retrieval quality is the binding constraint. This
benchmark makes that loop real and measures a learned global descriptor
(EigenPlaces) against the bag-of-features baseline (`normalized_mean`) on it.

`examples/multi_session_kitti_merge_demo.rs` merges *synthetic* pose graphs over
a hand-supplied bridge edge — no images, no features, no retrieval.
`examples/multi_session_lifelong_demo.rs` instead runs the genuine loop on real
7-Scenes `chess` data:

1. **Bootstrap** the metric map from the first session (seq-01) using its
   ground-truth poses — the "surveyed" deployment map. Each SuperPoint keypoint
   is back-projected through the registered Kinect depth and lifted to world.
2. **Integrate each later session** (seq-02, seq-04, seq-06) keyframe by
   keyframe, **with no ground-truth poses**: relocalize each keyframe against the
   map *so far* (learned-retrieval top-K → per-keyframe PnP, the dominant
   accuracy lever from [`relocalization_7scenes_demo`](learned_retrieval_relocalization.md)),
   and if it localizes, lift its keypoints through the **estimated** pose and
   append them — the map grows. Keyframes that fail to relocalize are dropped.
3. **Evaluate** the held-out test sessions (seq-03, seq-05) against both the
   bootstrap-only map and the full grown map.

Ground truth is used only to seed session 0 and to score; it never participates
in the merge. So the number of later-session keyframes each retrieval method
manages to integrate — and how accurately — is a direct measure of its
cross-session relocalization ability.

## Map growth across sessions

The headline is integration: how many of each later session's keyframes attach
to the lifelong map, and the median error of those relocalized merges (vs GT,
scored only). Sweeping the retrieval gate from loose to strict:

| gate (`--retrieve-topk` / `--min-inliers`) | merged keyframes (baseline / learned) | final map size (baseline / learned) | merge reloc median, hard sessions (baseline / learned) |
|:--:|:--:|:--:|:--:|
| 15 / 12 (loose)  | 120 / **126** | 170 / **176** | 0.085 m / **0.070 m** |
| 5 / 15           | 89 / **113**  | 152 / **163** | 0.143 m / **0.089 m** |
| 3 / 20 (strict)  | 93 / **106**  | 143 / **156** | 0.145 m / **0.103 m** |

(150 later-session keyframes attempted = 3 sessions × 50.) The learned
descriptor integrates more keyframes at every gate, and the gap **widens as the
gate tightens** (+6 loose → +24 → +13), exactly the
[single-shot relocalization](learned_retrieval_relocalization.md) pattern
carried into the lifelong setting. Just as important, the learned merges are far
more accurate on the hard re-visit sessions (≈0.09–0.10 m vs ≈0.14–0.15 m
median): the bag-of-features gate surfaces the wrong prior keyframe often enough
that PnP locks onto a worse pose, polluting the grown map.

## Held-out test localization

Localizing seq-03 / seq-05 against the map before and after the lifelong growth
(loose gate, top-15):

| map | retrieval | localized | median trans | within 5 cm / 5 deg |
|:--|:--|:--:|:--:|:--:|
| bootstrap only (50 kf)   | learned  | 80.0% | 0.070 m | 25.0% |
| **lifelong (176 kf)**    | learned  | 85.0% | **0.059 m** | **30.0%** |
| bootstrap only (50 kf)   | baseline | 75.0% | 0.070 m | 25.0% |
| lifelong (170 kf)        | baseline | 90.0% | 0.144 m | 25.0% |

Growing the map helps both fronts reach more of the test set, but the learned
lifelong map is markedly more *accurate* (median 0.059 m vs 0.144 m, 2.4×):
because its merges were cleaner, the points it added land where they should. The
test set here is small (20 strided frames), so read the loc-rate as indicative
and the map-growth / merge-accuracy numbers above as the robust signal.

## Reproduce

```sh
DS=~/datasets/7scenes/chess

# Features + globals for all six sequences (stride 20).
scripts/export_vpr_onnx.py --out models/eigenplaces_r50_2048.onnx
scripts/export_superpoint_7scenes.py --dataset $DS --seqs 1,2,3,4,5,6 --stride 20 --out-dir /tmp/sp_7scenes_chess
scripts/export_vpr_globals_7scenes.py --dataset $DS --seqs 1,2,3,4,5,6 --stride 20 --out-dir /tmp/vpr_7scenes_chess

EX=target/release/examples/multi_session_lifelong_demo
cargo build --release --example multi_session_lifelong_demo --features image-io

# learned EigenPlaces retrieval
$EX --dataset $DS --sp-features-dir /tmp/sp_7scenes_chess \
    --global-descriptor-dir /tmp/vpr_7scenes_chess \
    --sessions 1,2,4,6 --test-seqs 3,5 --retrieve-topk 15 --ratio 0.9 --reproj 6
# bag-of-features baseline (drop --global-descriptor-dir)
$EX --dataset $DS --sp-features-dir /tmp/sp_7scenes_chess \
    --sessions 1,2,4,6 --test-seqs 3,5 --retrieve-topk 15 --ratio 0.9 --reproj 6
```

`scripts/run_multi_session_lifelong.sh` wraps both runs. The SuperPoint export
needs a Python env with `lightglue`; the EigenPlaces export needs `torch` +
`torchvision`. Model `.onnx` files are git-ignored.
