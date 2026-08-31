# Electro 1,200-image quality attribution

This report closes Milestone 2 of the Electro performance roadmap. All
variants consume the same frozen 1,200 images, 12,000 candidate pairs, and
validated 11,625-pair snapshot. Reference poses are used only by the scorer
after mapping.

## Outcome

The selected Electro configuration registers **1,200/1,200** cameras at
**0.03224 m** Sim(3) centre RMSE. On the same candidate workload, COLMAP 3.9.1
registers 1,200/1,200 at 0.04679 m. The selected visloc-rs mapper is therefore
31.1% lower in centre RMSE and 3.31x faster, while remaining below the 4 GiB
mapper-RSS stop limit.

| Same frozen input | Registered | RMSE | Median | Mapper wall | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| visloc-rs cap64 baseline | 1193/1200 | 0.11940 m | 0.10994 m | 513.05 s | 3.79 GiB |
| **visloc-rs cap96 + bounded post pass** | **1200/1200** | **0.03224 m** | **0.01719 m** | **1490.07 s** | **3.83 GiB** |
| COLMAP CPU control | 1200/1200 | 0.04679 m | 0.03156 m | 4929.56 s | 1.20 GiB |

The quality win costs extra refinement time, which becomes the explicit M3
optimization target. It is not presented as an end-to-end win because the
current matcher remains slower than COLMAP and feature-extraction parity has
not yet been measured.

## First divergence: registration

All seven cap64 images absent from visloc-rs but present in COLMAP have strong
candidate and verified-pair support. Their first divergence is the same:
PnP runs before enough structure is available, returns only 3--6 inliers, and
the plain scheduler permanently spends its single trial. At the end of growth,
those images have 52--360 available 2D-3D correspondences.

| Image index | Features | Candidate / verified degree | First PnP | Support at stall |
| ---: | ---: | ---: | ---: | ---: |
| 169 | 2,755 | 15 / 15 | 66 corrs, 3 inliers | 106 |
| 226 | 3,540 | 15 / 14 | 43 corrs, 3 inliers | 101 |
| 417 | 2,431 | 29 / 29 | 301 corrs, 3 inliers | 360 |
| 474 | 2,505 | 15 / 15 | 70 corrs, 3 inliers | 96 |
| 599 | 1,727 | 9 / 5 | 52 corrs, 6 inliers | 52 |
| 938 | 922 | 15 / 15 | 125 corrs, 3 inliers | 142 |
| 946 | 697 | 15 / 15 | 127 corrs, 3 inliers | 141 |

The existing bounded `--post-refinement-registration` path explains and fixes
this divergence without an unbounded retry cycle. It recovers six images at
cap64 and all nine missing growth images at cap96.

## Second divergence: accuracy

Registration completion alone changes cap64 RMSE by less than 0.01%. The
accuracy gap is instead controlled by how many verified correspondences reach
track construction and BA.

| Mapper cap | Stage | Registered | RMSE |
| ---: | --- | ---: | ---: |
| 32 | growth / final | 1180 / 1180 | 0.62395 / 0.65971 m |
| 64 | growth / final | 1193 / 1193 | 0.24135 / 0.11940 m |
| **96** | growth / final+post | 1191 / **1200** | 0.06804 / **0.03224 m** |
| 128 | growth | 1185 | 0.14883 m |
| uncapped | growth | 1197 | 0.47039 m |

Cap96 is the smallest tested cap that passes both the registration and quality
gates. Cap128 and uncapped are two consecutive quality regressions, so the
planned stop condition applies. The non-monotonic result also identifies a
future mapper issue: with the current union-find conflict policy, admitting
more pair correspondences can fragment or contaminate tracks instead of
monotonically adding useful support.

## Preservation and scope

The cap is an explicit resource variable, not a new global default. The
current default path reproduces the courtyard control at 38/38 and 0.005379 m.
A deliberate fixed-cap96 courtyard negative registers 38/38 but regresses to
0.346318 m because that collection has roughly 11.6k features per image versus
about 2.5k on Electro. Consequently, the Electro champion is frozen as an
explicit benchmark configuration; dense-feature collections retain their
uncapped contract. The held-out South Building default run remains 128/128 at
0.73 cm and reproduces all three frozen model hashes exactly.

## Reproducible trace

`scripts/summarize_electro_decision_trace.py` converts mapper debug/timing
stderr into a compact JSON ledger of seed selection, growth PnP decisions,
trial exhaustion, post-refinement PnP, and BA rounds. `--decisions-only`
removes wall-clock fields so deterministic A/B traces can be compared directly.

The complete measured ledger, artifact hashes, preservation controls, and cap
matrix are in
[`benchmarks/electro/quality-attribution.json`](../benchmarks/electro/quality-attribution.json).
