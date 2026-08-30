# Cross-dataset non-regression preflight and closure (2026-08-30)

This report records the authoritative controls, the initial preflight, and the
follow-up acquisition/attempts. It is not a claim that a substitute frontend
or configuration passed. The CPU frontend is now provisioned in an isolated
external environment, so current results are recorded below with their exact
comparability caveats; no missing historical feature cache is silently
replaced.

## Authoritative commands and recorded controls

South Building's repository reproduction command is:

```sh
scripts/run_colmap_sfm_benchmark.sh --dataset south-building
```

Its documented default control is 128/128 registered and 1.09 cm Sim(3)
camera-centre RMSE against the shipped COLMAP model. The archived same-machine
registry arm additionally records the exact frontend/mapping configuration as
SuperPoint 2048, retrieval top-12, min-matches 30, and `--colmap-style`, with
128/128 and 0.40 cm against that historical reference; these are distinct
controls and are not conflated.

ETH3D terrace and office have no repository runner. Their authoritative
archived commands are the recorded SuperPoint export at max-dimension 3200,
2048 keypoints, CUDA, followed by `unordered_sfm_demo --retrieval-topk 12
--min-matches 30 --colmap-style`. The recorded laser-GT controls are terrace
23/23 at 12.37 cm and office 18/26 at 0.37 cm Sim(3) centre RMSE.

For EuRoC MH_03_medium, the authoritative trajectory command is:

```sh
scripts/run_euroc_loop_closure_benchmark.sh \
  --mav0 /path/to/MH_03_medium/mav0 --frames 2700
```

The recorded open/loop controls are respectively 2.462/2.203 m and
0.464/0.443 m (SE(3)/Sim(3) ATE RMSE). The full documented stack's final
10-pixel initial-residual-gate control is 0.057/0.072 m on MH_03/MH_05 (SE(3));
it is not substituted for the exact loop script. The separate SfM command is
`scripts/run_euroc_sfm_benchmark.sh --mav0 /path/to/MH_03_medium/mav0 --frames
2700`, with the recorded 178,973 tracks, 2,029,024 observations, and
4.08 px -> 1.04 px reprojection control.

## Current external-venv follow-up (2026-08-30)

The isolated environment is
`/media/sasaki/aiueo1/visloc-rs/envs/nonregression_20260830`. It uses Python
3.12.3, CPU `torch==2.3.1+cpu`, `torchvision==0.18.1+cpu`, LightGlue commit
`eb42fee2d71449efb0aa5c10549752b5d75384d8`, OpenCV 4.10.0, Kornia 0.7.3,
and `evo==1.31.1`; the current `pip-freeze.txt` digest is
`dae41bf42ceedf9a214cd040d002f941d88f8dcc036d5ecd1dc637808dad8f9f`.
The environment and SuperPoint weights are outside the repository and the
weights' hashes are recorded in its provenance file.

The South command was run twice with the same durable archive, feature files,
and default flags:

```sh
scripts/run_colmap_sfm_benchmark.sh \
  --dataset south-building \
  --data-root /media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/data \
  --out-dir /media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/runs/south-building-authoritative-venv \
  --python /media/sasaki/aiueo1/visloc-rs/envs/nonregression_20260830/bin/python \
  --device cpu
```

Both default runs loaded 128 images / 262,144 keypoints, verified 750/872
VLAD candidates with 277,354 inliers, and produced exactly the same model
files: **127/128** registered (always missing `P1180163.JPG`), 20,313 tracks,
92,682 observations, 1.409 px mean reprojection, and 0.74 cm RMSE against
the shipped COLMAP reference (median 0.60, max 1.59 cm). This is one fewer
registered image than the documented default 128/128 control, despite the
lower reference RMSE, so it is a registration regression candidate and is not
declared a pass. The separate, explicitly opt-in `--colmap-style` control
recovered 128/128, 20,570 tracks / 93,844 observations, 1.405 px and 0.40 cm;
it does not change default behavior. Logs and both repeat outputs are in
`runs/south-building-authoritative-venv*` and
`runs/south-building-colmap-style-venv`.

Terrace and office were run with the documented camera-0 PINHOLE values,
top-12/min-30, `--colmap-style`, and `--colmap-verification`. The historical
`kp2048@max-dim-3200` feature files used by the archived controls were not
available. Therefore the first rows below are an explicitly labelled external
frontend reproduction: LightGlue's SuperPoint was evaluated on
`load_image(..., resize=3200)`/OpenCV area output and keypoints were mapped back
to original pixels using the half-pixel convention before writing feature
files. The second rows use the current repository helper at original
resolution. Neither is an exact replay of the historical cached-feature arm.

| suite/frontend | loaded keypoints | candidate / verified pairs | registration | tracks / observations | mean reproj | reference Sim(3) RMSE |
|---|---:|---:|---:|---:|---:|---:|
| terrace, external 3200 | 46,615 | 147 / 90 (14,812 inliers) | 23/23 | 3,693 / 10,238 | 1.528 px | 132.07 cm (median 50.31, max 585.72) |
| terrace, current full-res helper | 46,301 | 151 / 92 (14,958 inliers) | 23/23 | 3,698 / 10,348 | 1.511 px | 130.62 cm (median 49.37, max 581.57) |
| office, external 3200 | 35,644 | 170 / 57 (4,904 inliers) | 18/26 | 1,172 / 3,283 | 1.519 px | 1.28 cm (median 0.49, max 4.73) |
| office, current full-res helper | 33,725 | 172 / 59 (4,912 inliers) | 17/26 | 1,168 / 3,275 | 1.522 px | 0.35 cm (median 0.22, max 0.96) |

The exact mapping/score commands and stdout, including camera values and
resource usage, are preserved in each run's `mapping.log`, `score.log`, and
`feature_export.log` under
`/media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/runs/`. The
archived acceptance controls remain terrace 23/23 at 12.37 cm and office
18/26 at 0.37 cm; the table above cannot establish a regression against those
controls because their feature bytes are missing. The downloaded archive
hashes continue to pass `SHA256SUMS`.

EuRoC was not substituted. After installing `evo`, the exact runner still
stops before rectification because no `MH_03_medium/mav0/cam0/sensor.yaml`
exists. The official legacy HTTP and HTTPS sequence URLs both timed out with
`curl` 28 after five seconds; the official landing page redirects successfully
but the Research Collection landing endpoint returned 403. Official metadata
reports a 12,096.15 MB Machine Hall archive, while durable free space was
4.7 GB, and no sequence-specific checksum/direct archive was exposed by the
accessible official endpoints. The complete evidence is in
`euroc_download_attempt_20260830.log`; no third-party archive was used.

## Follow-up acquisition and exact attempts

The initially missing photo archives were obtained without changing the repo
and stored under
`/media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830`:

| suite | source/archive | SHA-256 | extracted input |
|---|---|---|---|
| South Building | `https://github.com/colmap/colmap/releases/download/3.11.1/South-Building.zip` | `d210016bd2de20936a5f02b87fd38a76bf0440c42d045231218372cf9db9a7a1` | 128 JPEGs, 3072×2304, shipped `sparse/` and `database.db` |
| ETH3D terrace | `https://www.eth3d.net/data/terrace_dslr_undistorted.7z` | `586980a32f660bbe1f5107f031246440347ad4ce8d0d0e863e829e818c66d957` | 23 JPEGs; calibration present; dimensions 6203–6206×4133–4136 |
| ETH3D office | `https://www.eth3d.net/data/office_dslr_undistorted.7z` | `316c0c10c79cc173e4b5c26102fed3caedbde7e1865fd896ae15423f0f8cf04c` | 26 JPEGs, 6221×4146; calibration present |

The three archive hashes are verified by the durable root `SHA256SUMS`.
The 7z files were extracted with the already-installed `py7zr`; no system
package was installed. The exact South attempt was:

```sh
./scripts/run_colmap_sfm_benchmark.sh \
  --dataset south-building \
  --data-root /media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/data \
  --out-dir /media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/runs/south-building-authoritative \
  --python python3 --device cpu
```

It recognized all 128 images and the sparse model, derived the documented
3072×2304 pinhole, and stopped before export at
`ModuleNotFoundError: No module named 'torch'`. The durable log is
`.../nonregression_20260830/south_building_run.log`.

Terrace and office have no repository runner; their authoritative archived
commands require `export_superpoint_undistorted.py` followed by the documented
`unordered_sfm_demo --retrieval-topk 12 --min-matches 30 --colmap-style` arm.
The same Python environment has no `torch` or `lightglue`, and no prepared
feature directory is present, so those exact arms were not started. Running a
pure-SIFT or other substitute would not be a valid comparison and was not
reported as one.

EuRoC's official landing page is
`https://ethz-asl.github.io/datasets/euroc-mav/` (Research Collection DOI
`https://doi.org/10.3929/ethz-b-000690084`). No valid `MH_03_medium/mav0`
was available locally and the attempted exact command below stopped in the
rectifier at the missing `cam0/sensor.yaml`:

```sh
sh scripts/run_euroc_loop_closure_benchmark.sh \
  --mav0 /media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/data/euroc/MH_03_medium/mav0 \
  --out-dir /media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/runs/euroc-loop-authoritative \
  --frames 2700 --device cpu
```

The durable log is `.../nonregression_20260830/euroc_run.log`; it records the
missing path and no ATE/RPE was computed. The full EuRoC archive was not
downloaded merely to guess a subarchive or replace the authoritative command.

This paragraph records the pre-acquisition state; the official archive was
subsequently retrieved and validated in the 2026-08-31 section below.

The initial preflight attempts above remain historical blockers; the current
venv follow-up and its comparability limits are summarized in the section
above. South's default registration shortfall is recorded as an unresolved
regression candidate, while terrace/office cannot be judged against their
cached-feature controls and EuRoC has no valid input sequence.

## Initial local preflight findings (superseded where noted)

- `/home/sasaki/datasets/colmap_sfm/south-building` is absent; the follow-up
  archive is now under the durable path above.
- The original `/home/sasaki/datasets/eth3d/{terrace,office}` locations are
  absent; follow-up durable terrace/office archives and extracted inputs are
  now present under the path above.
- No EuRoC `MH_03_medium/mav0` directory or prepared rectified/features
  artifacts were found under the local dataset roots.
- `colmap` remains unavailable on `PATH`; Docker provides the pinned COLMAP
  image used for the separate courtyard control. The isolated external venv
  now provides `torch`, `lightglue`, `evo`, OpenCV, NumPy, Pillow, and py7zr
  for the follow-up runs.

The exact authoritative EuRoC ATE/RPE run remains blocked by missing `mav0` in
this historical preflight; no regression/pass judgment is claimed for it. For
South, terrace, and office,
the measured values and the reasons they are or are not comparable are in the
current follow-up section above. The existing default-off experimental flags
were not enabled in the default arms. The official `mav0` acquisition and
rectification are documented below.

## South provenance closure and exact smallest recovery switch (2026-08-30)

The recorded South controls have two distinct provenance lines. The repository
benchmark script's corrected plain result is the 128/128, 1.09 cm run recorded
by commit `2a36d44` (the earlier 0.58 cm value was explicitly superseded). The
registry's same-machine 128/128, 0.40 cm result is a Windows/CUDA scratchpad
run that included `--colmap-style`, at dirty commit
`fd7d06901c15869133f6d72e7b5dc14f4ef24d41`; it is not an exact plain-default
control. Neither historical record includes a feature-cache hash. A search of
the mounted workspace found no historical South feature files, so the current
CPU frontend/cache cannot be compared byte-for-byte to either result.

The current authoritative default was repeated with the verified 128-image
archive and the isolated CPU environment (`Python 3.12.3`, CPU
`torch 2.3.1+cpu`, LightGlue `eb42fee2d71449efb0aa5c10549752b5d75384d8`). Both
runs produced the identical candidate/verified graph (872/750 pairs,
277,354 inliers), 20,313 tracks and 92,682 observations, 1.409 px mean
reprojection, and 127/128 registration; only `P1180163.JPG` was missing. The
debug run shows the exact failure after plain growth:

```text
image 22: 443 triangulated track correspondences -> 3 PnP inliers (need 12)
635 pair correspondences available; growth exhausted at 127/128
```

The same graph with `--colmap-style` first retries that image while it is still
under-supported (462 -> 3), then succeeds after its repeated refinement
schedule (634 -> 524). A narrower A/B with only the existing
`--post-refinement-registration` pass leaves the plain growth unchanged, but
retries after final refinement with 615 correspondences and 508 inliers and
finishes at 128/128, 20,554 tracks, 93,647 observations, 1.406 px, and 0.73 cm
against the shipped COLMAP reference. The non-debug and debug post-only models
have identical SHA-256 files:

```text
cameras.txt  9358c2c20d03ef2f76e40f5c31565775717dbcc6a4b162ade16ae2b1db482392
images.txt   b8bee737605fe67ed2b5f6d7357d89d28a06a99d62331a7ef7cee612ee808885
points3D.txt 6057e1f8ebe3771c0c0cce73483b228a216b5ed99d37b9a8dbdd3fd312f83eff
```

Consequently, the smallest behavior that explains 127 -> 128 is the already
opt-in post-refinement retry, not a proven frontend or default-production
regression. No default switch was changed. The durable logs are under
`/media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/runs/` in
`south-building-{authoritative,debug-default,debug-colmap-style,post-only}*`.

The historical terrace/office frontend caches are also absent: the registry
records only the `superpoint_2048_cuda_maxdim3200` scratchpad, dirty Windows
revision and dataset tree hash. Current CPU runs therefore remain unlike-cache
diagnostics (terrace 132.07 cm / full-resolution helper 130.62 cm; office
1.28 cm / 0.35 cm), not regression verdicts against the cached 12.37/0.37 cm
controls. For EuRoC, all mounted filesystems were checked: the durable SSD has
4.7 GB free while the root NVMe has about 120 GB; official Machine Hall
metadata reports a 12,096.15 MB archive. The legacy official HTTP/HTTPS URLs
timed out, the Research Collection endpoint returned 403, and no sequence
specific direct archive/checksum was exposed, so the exact runner still lacks
`MH_03_medium/mav0` and no ATE/RPE result is claimed. Evidence is retained in
`euroc_download_attempt_20260830.log`.

## Cache-fixed source A/B and default-policy result (2026-08-30)

To remove frontend/cache ambiguity, the same durable feature directories were
fed to detached binaries built from `2a36d44`, intermediate commits, and the
current tree. The common command was:

```sh
RAYON_NUM_THREADS=4 /usr/bin/time -v <binary> \
  --features-dir <suite>/features --feature-suffix _features.txt \
  --image-suffix .JPG --width <W> --height <H> \
  --fx <fx> --fy <fy> --cx <cx> --cy <cy> \
  --retrieval-topk 12 --min-matches 30 --out-colmap <run>/colmap
```

For terrace, the feature-directory digest (sorted per-file SHA-256 list) is
`ccb41e8d3e9a9c4312fae9b73dc3183a7d0fcb038a49b3c32c0d7bac9f581b20`. The
documented baseline commit `2a36d44`, and detached `c48750a`, `dac1400`, and
`e18dea1` builds, all produced the same graph (147 candidates, 92 verified,
18,561 inliers) and **23/23, 3,614 tracks / 10,161 observations, 1.575 px**;
the baseline score is **1.63 cm** (median 0.99, max 4.45). The first
visibility-pyramid implementation, `9c35f72`, and current `101e5cc` produced
the same graph but only **12/23, 906 / 2,437, 1.465 px**; current score is
**148.78 cm** (median 74.82, max 420.12). The exact logs and model hashes are
under `runs/cache-fixed-ab-terrace-{baseline-2a36d44,c48750a-correct,dac1400,
e18dea1,9c35f72,current-101e5cc}-20260830/`.

The source change is therefore localized to the default next-image ranking
introduced by `9c35f72`: at the time of this 2026-08-30 comparison,
`NextImagePolicy::VisibilityPyramid` was the current library default, whereas
the earlier implementation selected by raw 2D--3D-correspondence count. A
temporary unordered-demo-only raw-count A/B recovered terrace to **23/23,
3,595 / 10,119, 1.574 px, 2.56 cm** and office to **17/26, 1,024 / 2,904,
1.531 px, 0.43 cm**, but reduced South to **123/128, 19,658 / 88,114, 1.411
px, 1.15 cm** (versus current visibility **127/128, 20,313 / 92,682, 1.409
px, 0.74 cm**). Because the simple global policy switch regressed South
registration/score, it was removed from that earlier experiment. The
2026-08-31 follow-up below deliberately revisits the historical default and
records the resulting cross-suite tradeoff. The visibility policy remains
available for explicit selection, and the raw-count policy remains available
as an experiment. This is a proven same-cache terrace default-path
regression, but it needs a cross-suite selection/fallback design rather than
assuming one global ranking is universally superior.

Office's feature-directory digest is
`689a5498d41e324f495bda99c9e320c58c15ce4847b6a56455ffd627213b8396`; the
current visibility run is **17/26, 931 / 2,612, 1.534 px, 0.58 cm**, while
the detached `2a36d44` control is **17/26, 1,024 / 2,904, 1.532 px, 0.43 cm**.
South's feature-directory digest is
`4b9121c171bc9fa96235cbee804959d89111fcc262f7901527acfc31d6dbbef2`; the
detached old control is **124/128, 20,160 / 90,467, 1.405 px, 0.69 cm** and
current visibility is **127/128, 20,313 / 92,682, 1.409 px, 0.74 cm**. These
figures are cache-fixed source comparisons, not claims that the historical
SuperPoint/Windows registry frontend has been reproduced.

## Historical ranking restoration A/B (pre-final-Auto patch, 2026-08-31)

The smallest source-derived patch was applied after the cache-fixed audit:
`NextImagePolicy::default()` and both demos' omitted policy now use the
historical raw correspondence-count comparator. The visibility-pyramid
comparator remains available explicitly through
`--next-image-policy visibility`; `count` selects the restored comparator.
The source audit found the behavior change in `9c35f72`, which replaced the
pre-change `corrs.len()` ordering with a 2x2 visibility-pyramid score followed
by correspondence count. No feature, match, verifier, or track code changed
for this A/B.

The current release binary was built in
`/tmp/visloc-regression-build-current-default-count-20260831/release/examples/`
and all runs used the same durable feature directories and camera/CLI values
as the cache-fixed controls above. The result table is:

| Dataset / policy | Candidates / verified / inliers | Registered | Tracks / observations | Mean reprojection | Sim(3) RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Terrace / count (default) | 147 / 92 / 18,561 | 23/23 | 3,595 / 10,119 | 1.574 px | 2.56 cm |
| Office / count (default) | 170 / 70 / 6,931 | 17/26 | 1,024 / 2,904 | 1.531 px | 0.43 cm |
| South / count (default) | 872 / 750 / 277,354 | 123/128 | 19,658 / 88,114 | 1.411 px | 1.15 cm |
| South / explicit visibility | 872 / 750 / 277,354 | 127/128 | 20,313 / 92,682 | 1.409 px | 0.74 cm |
| Courtyard / count (default) | 703 / 366 / 261,724 | 38/38 | 42,352 / 142,379 | 0.580 px | 5.03 cm |
| Courtyard / explicit visibility | 703 / 366 / 261,724 | 38/38 | 43,852 / 152,432 | 0.579 px | 0.5379 cm |

The count policy restores terrace's 23/23 registration and remains equal to
the detached office control, but it is not a safe unconditional default:
South loses four registered images relative to the current visibility run,
and the courtyard score is about 9.4x worse despite unchanged verified graph
counts. The explicit visibility courtyard run reproduces the durable
0.5379-cm champion. Consequently the requested terrace restoration is
confirmed, while the cross-suite and courtyard no-regression gates remain
open; a future fix needs a principled selection/fallback policy rather than
claiming that one global ranking is universally superior.

The EuRoC 2,700-frame feature export is still resumable and was not included
in this table. Its current status and the frozen archive/rectification hashes
remain in the EuRoC section below.

After the exporter hardening and stride-boundary fix, `sh scripts/check.sh`
completed successfully. This
covered the MSRV check, no-default/default/image-io workspace checks, workspace
clippy with `-D warnings`, all workspace targets, Python tests (243, eight
skipped), registry validation (189 JSON files), generated-doc/link/release
checks, examples, trajectory checks, workspace docs, and package metadata.
The independent `git diff --check` and `cargo fmt --all -- --check` checks also
passed. The running EuRoC feature exporter is an external long-running job,
not part of this finite CI result. The focused exporter safety suite is 5/5;
the full check's Python stage reports 243 tests with eight optional skips.

## Automatic next-image fallback A/B (historical pre-final-Auto policy, 2026-08-31)

At the time of this historical arm, `NextImagePolicy::Auto` was an explicit,
default-off policy in the incremental library and both demos. It ran
`VisibilityPyramid` first. If the resulting
registered fraction is below the fixed 90% threshold, it reruns
`CorrespondenceCount` from the same initial inputs and selects by the
lexicographic tuple `(registered images, valid observations, tracks,
-mean reprojection)`, with visibility winning exact ties. The implementation
clones only the small config value; feature and pair storage stays shared.
The library and omitted CLI default were then `CorrespondenceCount`; the
final omitted-CLI policy is documented in the authoritative section below.

The following runs used the durable same-cache feature directories and
per-dataset calibration values already listed in the ranking A/B above. Each
was run with `--next-image-policy auto`; the run log contains the candidate
metrics and selected policy.

| Dataset | Visibility candidate | Count candidate | Auto result |
| --- | --- | --- | --- |
| Terrace | 12/23, 906 tracks / 2,437 observations, 1.465059 px | 23/23, 3,595 / 10,119, 1.573934 px | count; 2.56 cm |
| Office | 17/26, 931 / 2,612, 1.533506 px | 17/26, 1,024 / 2,904, 1.531406 px | count; 0.43 cm |
| South Building | 127/128, 20,313 / 92,682, 1.409413 px | skipped (visibility >=90%) | visibility; 0.74 cm |
| Courtyard exhaustive | 38/38, 43,852 / 152,432, 0.579134 px | skipped (visibility >=90%) | visibility; 0.5379 cm |

The Auto artifacts are under
`/media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/runs/` with
suffix `-auto-20260831` (the corrected calibrated courtyard run is
`courtyard-current-auto-20260831-retry`). The selected courtyard cameras,
images, and points hashes are respectively
`76fc758375228319300a5c076b6bcc84a88413d69cf3613eb40c980b60e8cc9c`,
`a14ac6b958bf09481bfcc0ae72b59671d8e6405cd880e4711ff14c5c9432852e`, and
`d7b680e6d51a403a4962920e8f0e0615f382646a1ac9836f160ecb44622a3293`, equal
to the explicit-visibility champion. The low-resolution verified snapshot
control was run with
`/tmp/snapshot_colmap_verified_20260830.vps` (SHA-256
`6511181ac3b099cb9a9c8d7525b1746d28b7d5c7459df27e8460fef27f71f82a`) and
selected visibility at **38/38, 20,086 tracks / 66,894 observations,
0.281139 px, 3.42 cm**. Its repeat was byte-identical. Explicit Count on
the same snapshot was **38/38, 20,649 / 68,514, 0.342 px, 8.78 cm**, with
different image/point hashes, so the intended Count-default snapshot byte
identity check deliberately fails for Auto.

The visibility-first 90% rule is a fixed completeness guard, not a claim that
the count candidate is always inferior: when visibility reaches 90% it is
accepted without a second run, which can mask a complete-but-poor count path
by design. The low-resolution control is evidence for the chosen ordering,
while the cross-suite results are the reason Auto remains explicit rather than
becoming the default. Observed whole-process wall time / peak RSS was terrace
1:11 / 252 MB, office 0:49 / 189 MB, South 3:28 / 960 MB, and courtyard
1:28 / 1.07 GB; the latter two skipped the second mapper. Focused Auto
selection/public-API/CLI tests, release check, `cargo fmt --all -- --check`,
and `git diff --check` pass.

## Unified Auto plus recovery/post registration A/B (2026-08-31)

The requested unified, explicitly opt-in command was tested on the frozen
cache directories.  The common flags were:

```text
--mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8
--geometry-guided-conflict-recovery --post-refinement-registration
--final-iterative-refinement --next-image-policy auto
```

Terrace, office, and South used the authoritative file-feature cache with
`--retrieval-topk 12 --min-matches 30 --match-ratio 0.8`; their historical
Legacy verifier was intentionally retained by omitting `--verification-mode
full`.  Their camera values were respectively `(6205,4136,3412.13,3409.71,
3114.27,2060.02)`, `(6221,4146,3437.84,3435.95,3127.19,2066.98)`, and
`(3072,2304,2559.68,2559.68,1536.0,1152.0)` for width, height, fx, fy, cx,
cy.  Courtyard used all 703 imported raw pairs, per-image calibration, and
`--exhaustive --min-matches 20 --match-ratio 0.8 --verification-mode full`:

```text
--feature-extractor files
--features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/artifacts/colmap_highres_exhaustive_allpairs_20260830/features_sixcol
--feature-suffix _features.txt --image-suffix .JPG
--images-dir /home/sasaki/datasets/eth3d/courtyard/images/dslr_images_undistorted
--input-colmap-calibration /home/sasaki/datasets/eth3d/courtyard/dslr_calibration_undistorted
--import-matches-file /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/artifacts/colmap_highres_exhaustive_allpairs_20260830/exhaustive/matches_import.txt
--exhaustive --min-matches 20 --match-ratio 0.8 --verification-mode full
```

Results (all runs also used the common flags above) were:

| Dataset / verifier | Candidate / verified / inlier pairs | Auto selection | Registered | Tracks / observations | Mean reprojection | Reference Sim(3) RMSE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Terrace / Legacy | 147 / 92 / 18,561 | visibility (count skipped) | 23/23 | 3,522 / 9,790 | 1.559 px | 78.54 cm |
| Office / Legacy | 170 / 70 / 6,931 | visibility (count tie-break run) | 20/26 | 1,210 / 3,325 | 1.502 px | 0.53 cm |
| South / Legacy | 872 / 750 / 277,354 | visibility (count skipped) | 128/128 | 21,427 / 98,981 | 1.379 px | 0.92 cm |
| Courtyard / imported Full | 703 / 366 / 261,724 | visibility (count skipped) | 38/38 | 43,852 / 152,432 | 0.579 px | 0.54 cm |

The terrace result is a clear accuracy failure despite complete registration;
therefore this unified recovery/post stack is not a cross-suite non-regression
configuration.  Office registration improves to 20/26, South reaches 128/128
but is 0.92 cm versus its earlier visibility-only 0.74 cm, and courtyard
retains the 0.5379-cm champion.  The full-verifier terrace/office/South probes
are retained in their separate `cache-fixed-unified-*-auto-rpf-20260831`
directories but are not mixed into this Legacy comparison.

The exact selected-model directories are under
`/media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/runs/` with
`cache-fixed-unified-*-auto-rpf-legacy-20260831` (Legacy) and
`cache-fixed-unified-courtyard-auto-rpf-20260831` (courtyard Full).  Repeats
were written with `-repeat`/`-repeat2`; the three model files were byte-identical
for every completed repeat.  In particular, the courtyard hashes remained
`76fc758375228319300a5c076b6bcc84a88413d69cf3613eb40c980b60e8cc9c`,
`a14ac6b958bf09481bfcc0ae72b59671d8e6405cd880e4711ff14c5c9432852e`, and
`d7b680e6d51a403a4962920e8f0e0615f382646a1ac9836f160ecb44622a3293` for
`cameras.txt`, `images.txt`, and `points3D.txt`; terrace, office, and South
repeats likewise matched their first-run hashes.  Auto remains explicit and
the omitted CLI/library default is unchanged.

## EuRoC official archive acquisition and code A/B (2026-08-31)

The previous preflight's missing-`mav0` blocker was resolved without using a
third-party substitute. The ETH Research Collection record's public DSpace
API exposed the official Machine Hall bitstream
`7b2419c1-62b5-4714-b7f8-485e5fe3e5fe`:

```text
https://www.research-collection.ethz.ch/server/api/core/bitstreams/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content
```

It reports `machine_hall.zip`, 12,683,729,426 bytes. The archive was fetched
by bounded HTTP Range requests to the NVMe-only directory
`/home/sasaki/euroc_mh03_official_20260830/` (no repository or SSD files were
overwritten). The completed archive has exactly that size and SHA-256
`5ed7d07903f8d19b6c8808e2ae8a0872b281f6e34ef5497023b8ac58c3de0f6f`; the
same digest is also the public Hugging Face LFS OID for its independently
listed `machine_hall.zip` copy, used only as corroboration. `zipinfo -t`
reported 20 files, 15,492,631,305 uncompressed bytes and no error. The nested
official `machine_hall/MH_03_medium/MH_03_medium.zip` is 1,105,301,873 bytes,
SHA-256
`0f1707dfd6c9cda2c38302f4f7a47abb9a01a622a515dcbd6863730f0990f442`, and
passes `zipinfo -t` (5,420 files, 1,959,432,822 uncompressed bytes).

Only that nested sequence was extracted to the same external directory. It
contains 2,700 cam0 and 2,700 cam1 PNGs, both camera `sensor.yaml` and
`data.csv`, IMU data, and `state_groundtruth_estimate0/data.csv`. The frozen
rectification command was:

```sh
/media/sasaki/aiueo1/visloc-rs/envs/nonregression_20260830/bin/python \
  scripts/rectify_euroc_stereo.py \
  --mav0 /home/sasaki/euroc_mh03_official_20260830/MH_03_medium/mav0 \
  --out-dir /home/sasaki/euroc_mh03_official_20260830/rect_2700 \
  --frames 2700
```

It completed in 34.73 s and produced 2,700 images per rectified camera,
`calib.txt`, and 2,700 timestamps. The rectified calibration is
`752x480`, `fx=436.2443`, `cx=364.4412`, `cy=256.9517`, baseline
`0.110078 m`. The sorted 5,402-file rectified manifest has SHA-256
`b8c854b81bdf947599ec088dd32242fcd78ea1aa4e0af391f7a2628b36eab600`; the
individual calibration/timestamp hashes are retained in `key_hashes.sha256`.
The exact rectification log and archive/range logs are in the external
directory above.

For a code smoke control, detached `2a36d44` and current `101e5cc` binaries
were built in separate target directories. On the first ten frozen frames,
both produced the same per-pair counts (temporal matches 1,495--1,572,
stereo pairs 518--560, PnP inliers 577--616) and exited successfully; their
pose file hashes differ, so this is a smoke check, not the 2,700-frame metric
result. The baseline/current binaries are respectively
`0230a9f00a10a1f70178194e3c5226effbeaeacb2891e408bfd6196161c273db` and
`722401cd368d16e28e8db2bbb8deb45f5a3905f627c272dbd41ff3874d0f182b`.

The authoritative 2,700-frame feature export uses the frozen rectified images,
CPU `torch 2.3.1+cpu`/LightGlue environment, and `max-keypoints=2048` at
`/home/sasaki/euroc_mh03_official_20260830/features_sp2048_cpu_2700`.
The legacy direct-write/size-only exporter PID 3686761 was safely retired with
SIGINT followed by SIGTERM after its last completed frame 1848; its prefix
outputs were structurally validated before use.  The current
`scripts/export_superpoint_lightglue.py` helper now writes every feature and
match file through a same-directory temporary file plus `os.replace`, validates
existing files structurally under `--skip-existing`, and can emit a checked
SHA-256 manifest.  Explicit sequence/mono ranges use source indices and are
selected with `--start-index I --end-index J`; the first frame in a range also
creates the boundary temporal match against its predecessor exactly once.
The atomic tail workers covered `[2200,2450)` and `[2450,2700)`; after retiring
the legacy process, three disjoint workers repaired the gap `[1849,1966)`,
`[1966,2083)`, and `[2083,2200)`.  Their range manifest hashes are, in that
order, `db058e79ce36e2463d3d4b1424f5b46f6eb861ffa0db1dc4ee1570d37a6a78b0`,
`4f7dbaef513907186ccffbe4a99fea7f311cc023f0ebdcfc21451bafdddd7ab5`, and
`89133b294bec33ed7c6563f218c8a643cb5773eb7b99afd0887df95f86d18a76`; the
two earlier tail manifests are `ab17076e4bfc5669a1d556c7618c9733d3e621c9c16c32c76b14ccf09edf7ce6`
and `d151ca478238e228839930c5bb3e8c77990a59e13af0743eb0b108656c198d`.
The canonical validated manifest is
`/home/sasaki/euroc_mh03_official_20260830/manifest_full_2700.json` (JSON
SHA-256 `6c6f9f64551882bd5dafbe98719348879511c10cfeed280dcee25630db97ed38`,
manifest SHA-256 `489d953274540d331603fa072f04996ab20c39c9cddfcecb1d332120a4ab801f`):
it contains exactly 2,700 left features, 2,700 right features, 2,700 stereo
match files, and 2,699 temporal match files (10,799 files total), with zero
temporary files.  The independent structural/hash pass took 11:02.19.

### EuRoC MH_03 same-cache baseline/current ATE-RPE (2026-08-31)

The exact loop benchmark command below was run once from detached baseline
commit `2a36d44` and once from the current worktree (`101e5cc` plus the dirty
tree), always with the same frozen rectification, calibration, 2,700-frame
feature directory, CPU venv, and `--skip-rectify --skip-export`:

```sh
PATH=/media/sasaki/aiueo1/visloc-rs/envs/nonregression_20260830/bin:$PATH \
sh scripts/run_euroc_loop_closure_benchmark.sh \
  --mav0 /home/sasaki/euroc_mh03_official_20260830/MH_03_medium/mav0 \
  --rect-dir /home/sasaki/euroc_mh03_official_20260830/rect_2700 \
  --feat-dir /home/sasaki/euroc_mh03_official_20260830/features_sp2048_cpu_2700 \
  --skip-rectify --skip-export --frames 2700 --device cpu \
  --python /media/sasaki/aiueo1/visloc-rs/envs/nonregression_20260830/bin/python \
  --out-dir /home/sasaki/euroc_mh03_official_20260830/runs/euroc-loop-<baseline-or-current>
```

Each variant produced 2,700 poses and 2,699 pair updates.  ATE is translation
RMSE over all poses (SE(3) / Sim(3) Umeyama, metres).  RPE is the consecutive
one-frame translation RMSE (`evo.main_rpe`, `-a -r trans_part -d 1 -u f`; the
second value also uses `-s`).

| Variant | baseline ATE SE3 / Sim3 | current ATE SE3 / Sim3 | baseline RPE1 SE3 / Sim3 | current RPE1 SE3 / Sim3 |
| --- | ---: | ---: | ---: | ---: |
| open | 2.384514 / 2.153919 | 2.401288 / 2.174040 | 0.068109 / 0.061289 | 0.068052 / 0.061246 |
| loop | 0.498364 / 0.487327 | 0.445577 / 0.439303 | 0.069934 / 0.070975 | 0.070392 / 0.071137 |
| full | 0.084135 / 0.083063 | 0.087830 / 0.084345 | 0.073060 / 0.072916 | 0.073224 / 0.072961 |
| full2v | 0.058594 / 0.050140 | 0.058698 / 0.054061 | 0.071469 / 0.071164 | 0.071357 / 0.071129 |
| full2vh | 0.063683 / 0.053853 | 0.063884 / 0.056473 | 0.071463 / 0.071122 | 0.071362 / 0.071064 |
| full2vhi | 0.059021 / 0.050955 | 0.059811 / 0.053683 | 0.071377 / 0.071079 | 0.071241 / 0.070979 |

No registration loss was observed: all variants retained 2,700/2,700 poses.
The full-pipeline online-BA aggregates were baseline 651,101 tracks /
5,929,255 observations (`full`/`full2v`) and 1,007,103 / 9,953,946
(`full2vh`/`full2vhi`); current was 676,567 / 6,132,761 and 1,046,802 /
10,294,175 respectively.  The current-vs-baseline differences are small and
mixed (current loop improves ATE, while current `full2vhi` Sim(3) is 3.53 mm
worse: 0.053683 m vs 0.050955 m); this is a same-cache code A/B, not a claim
against the older absolute controls.

The EuRoC loop-closure example has no `NextImagePolicy`/`--next-image-policy`
selector; therefore a separate current “Auto” EuRoC ATE/RPE variant is not
applicable and was not fabricated.  Auto remains covered by the documented
ETH3D incremental/SfM A/B above.  Baseline/current run roots and all raw logs
are retained under
`/home/sasaki/euroc_mh03_official_20260830/runs/euroc-loop-{baseline-2a36d44,current}`.
For deterministic evidence, the current `full2vhi` command was repeated in a
fresh `euroc-loop-current-full2vhi-repeat-correct` directory with the same
explicit gap/path/similarity flags; `vo_poses.txt` and `est.tum` hashes were
identical (`b47a3dd8093d9c205fd5b5213a4392c9ea694be831cfc20d1c61667f0ed64743`
and `3190442ff6af2bc35712f24312518449e944b9c10b26d741e75fe9257b23cd3b`), as
were ATE `0.059811/0.053683 m` and RPE1 `0.071241/0.070979 m` (SE3/Sim3).

## Final Auto default closure and same-cache classification (2026-08-31)

The final demo-policy patch was evaluated with the same frozen feature
directories and camera values as the cache-fixed controls.  Both SfM demos
now parse `NextImagePolicy::Auto` when `--next-image-policy` is omitted, while
`IncrementalSfmConfig::default()` remains `CorrespondenceCount` for library
and API compatibility.  Auto runs Visibility first; if any image is missing,
it also runs the historical Count candidate from the same immutable inputs and
selects by registered images, valid observations, tracks, then finite mean
reprojection (visibility wins exact ties).  Only after that selection, and
only when the selected model is incomplete, it runs the existing post
completion pass from a clean state.  The post candidate is adopted only when
the registered-image count strictly increases; ties and lower counts retain
the pre-post model and its bytes.

The no-flag commands were the following (each output directory is unique and
the full command is preserved in its `run.log`):

```sh
target/release/examples/unordered_sfm_demo \
  --features-dir <frozen-suite>/features --feature-suffix _features.txt \
  --image-suffix .JPG --width <W> --height <H> --fx <fx> --fy <fy> \
  --cx <cx> --cy <cy> --retrieval-topk 12 --min-matches 30 \
  --out-colmap <run>/colmap

target/release/examples/unordered_sfm_demo \
  --feature-extractor files \
  --features-dir /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/artifacts/colmap_highres_exhaustive_allpairs_20260830/features_sixcol \
  --feature-suffix _features.txt --image-suffix .JPG \
  --images-dir /home/sasaki/datasets/eth3d/courtyard/images/dslr_images_undistorted \
  --input-colmap-calibration /home/sasaki/datasets/eth3d/courtyard/dslr_calibration_undistorted \
  --import-matches-file /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/artifacts/colmap_highres_exhaustive_allpairs_20260830/exhaustive/matches_import.txt \
  --exhaustive --min-matches 20 --match-ratio 0.8 --verification-mode full \
  --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 \
  --geometry-guided-conflict-recovery --final-iterative-refinement \
  --out-colmap <courtyard-run>/colmap
```

The resulting Auto logs and scores are:

| suite | Auto candidates/decision | final registered | tracks / observations | mean reprojection | reference Sim(3) RMSE |
|---|---|---:|---:|---:|---:|
| South Building | Visibility 127/128 vs Count 123/128; post adopted 127→128 | 128/128 | 20,554 / 93,647 | 1.406 px | 0.73 cm |
| terrace | Visibility 12/23 vs Count 23/23; Count selected, post skipped | 23/23 | 3,595 / 10,119 | 1.574 px | 2.56 cm |
| office | Visibility 17/26 vs Count 17/26; Count selected, post adopted 17→18 | 18/26 | 1,082 / 3,037 | 1.512 px | 0.45 cm |
| courtyard exhaustive | Visibility complete; Count/post skipped | 38/38 | 43,852 / 152,432 | 0.579 px | 0.5379 cm proxy |

The terrace result is the important guard: the earlier explicit
recovery+post experiment produced 78.54 cm, whereas the final omitted-policy
run compares Count first and retains 2.56 cm.  Office gains one registered
camera and lowers reprojection (1.531→1.512 px) relative to the same-cache
Count control; its reference RMSE changes 0.43→0.45 cm, so this is not claimed
as a strict accuracy improvement.  Courtyard's `cameras.txt`, `images.txt`,
and `points3D.txt` are byte-identical to the durable champion, with hashes
`76fc758375228319300a5c076b6bcc84a88413d69cf3613eb40c980b60e8cc9c`,
`a14ac6b958bf09481bfcc0ae72b59671d8e6405cd880e4711ff14c5c9432852e`, and
`d7b680e6d51a403a4962920e8f0e0615f382646a1ac9836f160ecb44622a3293`.
The run roots are
`cache-fixed-auto-default2-{south,terrace,office,courtyard}-20260831` under
`/media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/runs/`; the
effective-config lines in each log show `next_image_policy: Auto`.

Snapshot replay deliberately has a separate compatibility rule.  When a
verified-pair snapshot is imported without an explicit policy, the CLI forces
Count; an explicit `--next-image-policy auto` remains valid and selects the
visibility candidate.  On `/tmp/snapshot_colmap_verified_20260830.vps` (SHA
`6511181ac3b099cb9a9c8d7525b1746d28b7d5c7459df27e8460fef27f71f82a`), the
omitted-policy replay was 38/38, 20,649/68,514, 0.342 px and matched the
Count-control model byte-for-byte:

```text
snapshot-count-default2-identity-20260831/colmap
  cameras a2132068b1a4dbe21f1ad68a23ff05461026c5a84e0b0de14f06311533e5b958
  images  23836ffe18995d83a4e0c7a56375b39aa0d702c59af1c0ec7b5da85c65b04a2e
  points  1a088ea533aaa2891609333dcdc819d1342dbee53523332903b87783e433c81c
```

The explicit Auto override remains reproducible at 20,086/66,894, 0.281 px;
its hashes match the historical visibility model in
`snapshot_colmap_import_model_20260830b`.  Thus snapshot Count identity is
preserved without removing an explicit Auto escape hatch.

Terrace and office are now classified separately for code non-regression
using the detached `2a36d44` same-cache controls, while the historical
SuperPoint cached-feature arms remain unavailable:

| suite | detached baseline → current Count | code non-regression | historical absolute arm |
|---|---|---|---|
| terrace | 23/23, 3,614/10,161, 1.575 px, 1.63 cm → 23/23, 3,595/10,119, 1.574 px, 2.56 cm | registration pass; numeric **inconclusive** (no archived tolerance, +0.93 cm) | **inconclusive**, old feature bytes unavailable |
| office | 17/26, 1,024/2,904, 1.532 px, 0.43 cm → 17/26, 1,024/2,904, 1.531 px, 0.43 cm | **pass** (same registration/support and lower reprojection) | **inconclusive**, old feature bytes unavailable |

For EuRoC, the project-level same-cache rule was fixed before classifying any
variant: all poses and pair updates must be present, and every current ATE/RPE
SE(3)/Sim(3) metric must be no greater than the baseline by more than
`max(5% of baseline, 0.005 m)`.  The 5 mm absolute floor is a single
engineering tolerance for this trajectory evaluator, applied uniformly rather
than tuned per variant; exact repeat hashes are an additional determinism
check.  Under that rule every open, loop, full, full2v, full2vh, and full2vhi
variant passes same-cache non-regression: the largest ATE Sim(3) increase is
3.921 mm (full2v), below the fixed 5 mm floor, and all retain 2,700/2,700
poses and 2,699 updates.  This does not claim improvement over the older
absolute benchmark controls.  The loop runner has no `NextImagePolicy`, so
EuRoC Auto remains N/A rather than an invented variant.

The old section above remains the detailed acquisition and metric record; this
section is the current policy/classification authority after the Auto patch.
