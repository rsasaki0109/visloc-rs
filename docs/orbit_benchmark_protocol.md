# ORBIT SfM benchmark protocol and release audit

Audit date: 2026-07-22.

Authoritative sources:

- [CVPR 2026 paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Sabour_ORBIT_Benchmarking_SfM_in_the_Wild_with_360deg_Video_CVPR_2026_paper.html)
- [camera-ready paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Sabour_ORBIT_Benchmarking_SfM_in_the_Wild_with_360deg_Video_CVPR_2026_paper.pdf)
- [official supplemental archive](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Sabour_ORBIT_Benchmarking_SfM_CVPR_2026_supplemental.zip)

The downloaded paper and supplemental archive are retained under
`E:/visloc_archive/orbit_official_protocol_20260722`. Their SHA-256 digests
are respectively
`27a58f6e7e4c4033d6513710dadb79a4ba88df72b83ee7a6ccb2d3caafb5ad72`
and
`8bcfe32f0665741f464a50e63623b8856b395568c678cb95c61a1398966066ce`.

## Protocol recovered from the official artifacts

- ORBIT contains 100 perspective video clips derived from panoramic 360-degree
  video, with approximately metric ground-truth camera trajectories.
- Missing estimated frames are filled with the previously reported camera
  pose until the next estimate. A method producing no estimate is a failure.
- Estimated and reference trajectories are aligned with Umeyama before ATE
  and relative translation/rotation errors are computed. The paper does not
  state unambiguously in the released text whether benchmark-time Umeyama
  enables its scale term; this must be taken from the evaluator, not guessed.
- The strict binary success criterion is all of `ATE < 0.5`, `RPE-R < 0.4`,
  and `RPE-T < 2.0`. The relaxed criterion doubles all three thresholds.
- The headline comparison includes COLMAP, ORB-SLAM2, ParticleSfM, MonST3R,
  MegaSaM, RoMo+MegaSaM, and VGGT-Long. Report per-clip results, strict and
  relaxed success rates, and distribution/worst-case statistics; the paper
  explicitly warns that means alone hide the highly varied failure modes.
- Challenge slices include low static texture, low light, high camera speed,
  camera rotation, independently moving crowds/objects, objects moving with
  the camera, and fluids.

## Release blocker

The official camera-ready supplemental website still points its Code, Dataset,
and arXiv buttons to the literal placeholder `anon`. The 56,792,508-byte
official supplemental ZIP contains the website, paper/supplement PDFs, images,
and sample videos, but no benchmark dataset, ground-truth trajectories,
evaluator, Python source, JSON, or CSV files. Consequently the 100-clip ORBIT
matrix cannot currently be run or reproduced from the official release.

This is an external artifact blocker, not permission to substitute the sample
videos, infer an evaluator, scrape source videos, or claim an ORBIT result.
Recheck the official paper page/project links before S4 freeze. Once released,
archive the exact dataset/evaluator revision and run the frozen visloc and
baseline configurations without adapting thresholds or clip selection.
