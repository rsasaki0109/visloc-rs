# M5 scale validation

Milestone 5 separates three different claims: real-geometry quality across the
ten ETH3D low-res many-view scenes, bounded/restartable I/O at 10k and 100k,
and one genuinely connected 10k-image environment. Unrelated ETH3D scenes are
never joined into one graph, and supplied poses are read only after a model is
selected and written.

## 10k / 100k I/O gate

The synthetic tier exercises manifests, `O(NK)` pair generation, bounded
shards, interruption, hash-validated resume, and a verify-only replay. It makes
no image-geometry or registration claim.

| Tier | Images / pairs | Shards | Clean wall / RSS | Interrupted → resumed | Verify-only replay | Identity |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 10k | 10,000 / 319,472 | 78 | 0.311 s / 19,720 KiB | 10 reused + 68 written | 78 reused, 0 written | exact index + aggregate SHA |
| 100k | 100,000 / 3,199,472 | 782 | 3.160 s / 33,832 KiB | 100 reused + 682 written | 782 reused, 0 written | exact index + aggregate SHA |

The 100k tier remains about `32N`, peaks at 33.6 MiB during verify-only replay,
and does not materialize an `N²` pair matrix. Clean and interrupted/resumed
runs reproduce the same index and aggregate-shard hashes. Full values and
individual result-file hashes are in
[`m5-scale-io-audit.json`](../benchmarks/electro/m5-scale-io-audit.json).

## Snapshot merge memory A/B

The previous snapshot writer encoded a complete payload `Vec`, copied it into
a second file `Vec`, and the reader loaded the complete file before decoding.
The shipped path computes the payload length from bounded metadata, streams
the exact existing format through an incremental FNV checksum, validates a
snapshot with a 1 MiB checksum buffer, then decodes from the file. The
unordered edge hash now sorts matches within normalized image-pair groups
instead of materializing one global correspondence tuple array.

On the same 730 `sand_box` shards, the legacy helper peaked at **5,611,868
KiB** and took **11.01 s**. The final streaming/grouped helper peaked at
**2,132,012 KiB** and took **11.25 s**: **62.0% less peak RSS**, with the same
merged snapshot SHA-256
`1cf25a…fb232`. Mapping that snapshot retained all three model-file hashes;
mapper peak fell from 4,228,540 KiB to 3,478,220 KiB and wall fell from 501.98
s to 454.15 s. Exact hashes and timing-file identities are in
[`m5-streaming-snapshot-memory-audit.json`](../benchmarks/electro/m5-streaming-snapshot-memory-audit.json).

## All ten ETH3D low-res many-view scenes

The ten supplied collections contain 10,008 images. They are reconstructed and
scored independently: unrelated scenes are never joined to manufacture a 10k
graph. Candidate generation, matching, seed selection, and mapping receive no
supplied pose. The reference model is opened once, after the output model is
selected and written.

| Scene | Registered / supplied | RMSE | Relative RMSE | Mapper peak RSS |
| --- | ---: | ---: | ---: | ---: |
| terrains | 660/660 | 0.58 cm | 0.12% | 1,640,940 KiB |
| delivery_area | 948/948 | 9.22 cm | 0.99% | 2,426,332 KiB |
| forest | 1028/1028 | 1.33 cm | 0.19% | 2,777,044 KiB |
| playground | 955/960 | 6.12 cm | 2.52% | 2,563,276 KiB |
| electro | 1200/1200 | 3.50 cm | 0.55% | 1,459,194 KiB |
| lakeside | 1063/1064 | 0.34 cm | 0.08% | 3,342,288 KiB |
| sand_box | 1112/1112 | 2.35 cm | 0.45% | 3,478,220 KiB |
| storage_room | 795/796 | 0.61 cm | 0.42% | 1,795,320 KiB |
| storage_room_2 | 831/832 | 3.48 cm | 2.57% | 1,045,948 KiB |
| tunnel | 1404/1408 | 14.92 cm | 1.61% | 3,405,036 KiB |
| **Total / maximum** | **9996/10008 (99.88%)** | — | — | **3,478,220 KiB** |

`playground` excludes five explicit source outliers through its hash-bound
staging audit; all 955 staged images register. `storage_room_2` seed 1 produced
only 2/832 registrations, so the predeclared internal ladder advanced to seed
16; no reference score was available during that choice. `tunnel` has a 2.47
cm median and 9.41 cm p95, but its RMSE includes one honest 5.32 m cam4 outlier.

The README animation and still are regenerated from the ten actual
`images.txt` models with
[`generate_eth3d_scale_readme_visuals.py`](../scripts/generate_eth3d_scale_readme_visuals.py).
PCA projection and per-panel normalization affect display only. Full-precision
scores, the tunnel snapshot/model hashes, and selection notes are in
[`m5-eth3d-scale-validation.json`](../benchmarks/electro/m5-eth3d-scale-validation.json).

### Tunnel compact replay A/B

The 2.3 GiB tunnel snapshot contains 30,386 accepted pairs and 36,784,462
accepted correspondences. A lossless full decode peaked at 4,688,360 KiB. The
final mapper reader validates the complete checksum and each raw-index mapping
at a pair boundary, releases audit-only vectors, and peaked at **3,405,036
KiB**: **27.4% less memory** and below the 4 GiB gate. Wall changed from 569.07
s to 565.21 s. Both runs registered 1404/1408 images and produced identical
`cameras.txt`, `images.txt`, and `points3D.txt` SHA-256 values.

## Connected 10k environment

OpenLORIS `corridor1-1` is the connected-environment complement to the ten
independent ETH3D reconstructions. The 13.85 GB source archive is pinned to
commit `cbc03108723d08322b23d0338680bffa9404cce9` and LFS object
`c7ff1a…8415`. Staging selects the first 5,000 frames from each T265 fisheye,
sorts both streams by timestamp, validates the official `sensors.yaml`, and
rectifies the Kannala-Brandt images to 848×800 PINHOLE views while preserving
the supplied focal lengths and principal points. Images, features, snapshots,
and models remain external; the CC BY-ND 4.0 staged derivatives are not
redistributed.

The complete 10,000-image feature bank was extracted once in eight disjoint
1,250-image workers. All workers completed, yielding 1,427,634 keypoints; each
worker peaked below 273 MiB and the slowest wall time was 1:14:35. Tier views
are prefix symlinks into that one bank, so tier growth does not duplicate image
or feature bytes.

The frozen connected-run policy uses temporal offsets 1/2/4/8/16/32, same-time
cross-camera edges, then a deterministic VLAD fill to exactly `7N` candidates.
It uses ratio 0.8, 12 minimum verified matches, one persistent four-thread
matcher, 32-pair restart shards, a 96-correspondence mapper cap, keypoints-only
snapshot replay, 16 seed trials, and sparse eight-iteration BA. No trajectory
or supplied extrinsic enters candidate generation, matching, seed selection,
or mapping.

| Tier | Candidates / verified | Registered | Tracks / observations | Mean reproj | Total wall | Peak phase RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 7,000 / 6,869 | **989 (98.9%)** | 2,586 / 28,918 | 1.574 px | 2:25 | 280,916 KiB |
| 2,500 | 17,500 / 16,321 | 1,223 (48.9%) | 4,226 / 36,644 | 1.766 px | 7:10 | 401,308 KiB |
| 5,000 | 35,000 / 31,521 | 1,212 (24.2%) | 3,318 / 29,317 | 1.072 px | 20:33 | 691,840 KiB |
| 10,000 | 70,000 / 58,879 | **199 (2.0%)** | 873 / 5,184 | 1.301 px | 1:03:45 | 1,869,412 KiB |

The scale result is intentionally mixed rather than presented as a quality
win. RSS stays bounded and the 5k mapper itself peaks at only 470,492 KiB, but
registration plateaus near 1.2k images and falls to 199 at the full tier. The
10k peak is the 2,188-shard merge at 1.78 GiB; its candidate, matcher, and
mapper peaks are 1.14 GiB, 851 MiB, and 501 MiB respectively. A denser 1k graph
was worse: 23,157 verified pairs produced only 2/1,000 registrations because
UnionFind conflict
resolution collapsed the usable tracks; the selected 7N graph reduces its
152 MiB merged snapshot to about 44 MiB and registers 989. A 2.5k global-mapper
diagnostic reaches 1,459 cameras but is rejected at 3,567 px mean reprojection.

Candidate output is `O(NK)`, but this baseline's exact VLAD top-K ranking still
scores every image pair and loads the file-backed feature bank before reducing
it to global descriptors. Consequently its candidate-generation time grows
from 0:52 at 1k to 3:42 at 2.5k, 13:01 at 5k, and 49:49 at 10k. Replacing that
full similarity scan with a bounded ANN/inverted index, and streaming
descriptor-to-global aggregation, are the next speed and memory work. Track
construction also needs
to preserve geometrically consistent alternatives instead of collapsing long
sequence evidence through same-image UnionFind conflicts. This report does not
label the current connected pipeline linear-time or fully registered. Exact
source/config/artifact hashes and all four tier ledgers are in
[`m5-openloris-connected-scale-validation.json`](../benchmarks/electro/m5-openloris-connected-scale-validation.json).
