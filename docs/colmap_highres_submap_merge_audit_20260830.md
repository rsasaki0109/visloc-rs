# 高解像度 COLMAP 2サブモデル merge feasibility audit (2026-08-30)

公式高解像度 COLMAP の `Mapper.multiple_models=1` 出力を、独立した
2つの reconstruction が verified cross-component geometry だけで結合できるか、
読み取り専用で調べた。公式 calibration extrinsics/レーザ GT は merge の推定には
使っていない。最後の reference 比較だけ、既存 evaluator の audited calibration
proxy に対して行った。

## 入力と成分

対象 DB は
`/tmp/colmap_official_highres_8192_20260830/database_calibrated_threads8.db`、
submodel は
`/tmp/colmap_official_highres_8192_colmap_mapper_n3_20260830/sparse_multi_txt/{0,1}`
である。`images.txt` の image name を basename 化して照合した結果は次の通り。

| submodel | image membership | points / observations | point error mean / median / p95 / max (px) |
|---|---|---:|---:|
| 0 | `DSC_0286`..`DSC_0308` (23) | 16,714 / 68,797 | 0.6677 / 0.5214 / 1.7346 / 3.6434 |
| 1 | `DSC_0308`..`DSC_0323` (16) | 14,409 / 57,064 | 0.6023 / 0.4749 / 1.5682 / 3.8141 |

共有 camera は **`DSC_0308.JPG` の1枚だけ**である。従って排他的な集合は
submodel 0 が22枚、submodel 1 が15枚、排他的な cross pair は
`22 * 15 = 330` 組になる。

DB の全330組を照合すると、raw match の行が存在するのは以下の4組だけだった。
いずれも `two_view_geometries.rows=0, config=0` で、E/F/H blob はなく、
verified essential geometry ではない。

| pair | raw rows | verified geometry |
|---|---:|---|
| 0305--0309 | 50 | none (`rows=0`, `config=0`) |
| 0306--0310 | 31 | none (`rows=0`, `config=0`) |
| 0307--0309 | 55 | none (`rows=0`, `config=0`) |
| 0307--0311 | 74 | none (`rows=0`, `config=0`) |

つまり、排他的成分同士を結ぶ verified E edge は **0/330**、raw-only edge も
4/330 に留まる。これは matcher の `quadratic_overlap=1` が含めた候補であり、
有効な幾何接続と解釈していない。

## 共有 camera からの cross-anchor 診断

唯一の共有 camera で world-to-camera rotation を合わせた。submodel 1 の local
frame から submodel 0 の frame への回転は

```text
Q = R0(0308)^T R1(0308)
R1(0308) Q^T と R0(0308) の Frobenius error = 6.66e-16
```

であり、共有 pose だけなら回転は完全に一致する。さらに DB の有効 geometry のうち
共有 camera を始点とし、submodel 1 の排他的 camera に向かう3本を
`cv2.recoverPose` 相当の cheirality 選択で診断した。E の符号は positive-depth
解を採用し、OpenCV/COLMAP の camera-`i` から camera-`j` への写像
`x_j = R x_i + t` に対して、camera centre の方向は `C_j^{(i)}=-R^T t` とした。
これは GT ではなく、DB の E、keypoint、intrinsics と各 submodel pose だけを用いた
値である。

| cross-anchor pair | E rows | positive cheirality | model-vs-E rotation | model-vs-E translation direction | normalized epipolar residual median / p95 |
|---|---:|---:|---:|---:|---:|
| 0308--0309 | 1,276 | 1,207 (94.6%) | 0.147° | 1.931° | 1.66e-4 / 6.53e-4 |
| 0308--0310 | 653 | 554 (84.8%) | 1.917° | 11.621° | 7.55e-5 / 4.66e-4 |
| 0308--0312 | 296 | 291 (98.3%) | 2.952° | 6.558° | 7.87e-5 / 3.77e-4 |

方向は完全に一直線ではなく、pairwise angle は 18.17°, 149.67°, 167.69°、
方向行列の singular values は `1.6915, 0.3722, 0.0202`（最大/最小比約83.8）
だった。しかし、3本とも始点が同じ `DSC_0308` であるため、相対 Sim(3) の scale
は消去される。cross product 形式

```text
[d_ij]_x (s Q C1_j + b - C0_i) = 0
```

に共有 camera の関係 `b = C0_shared - s Q C1_shared` を代入すると、理想的な
方向データでは s の係数はゼロになる。実測 E 方向の誤差（特に 0308--0310 の
11.6°）を残差として無理に解くと、9x4 行列の見かけの singular values は
`3.5989, 1.7286, 0.7445, 0.0922`（condition 約39）になるが、最小二乗解は
`s ≈ 0, b ≈ C0_shared`、つまり右成分を共有 camera に潰す退化解になる。これは
scale が観測できたことを意味しない。排他的 cross E が0本なので、非共通の始点を
持つ方向制約もなく、複数辺による正の scale 推定は不可能である。

## 非 GT merge A/B

共有 camera の pose を1つに deduplicateし、submodel 1 の camera centre を

```text
C1' = s Q C1 + (C0_shared - s Q C1_shared)
```

で submodel 0 frame に移した。ここで `s` は以下の2つだけを試した。

1. `s=1`: metric scale を仮定する naive shared-camera merge。
2. `s=2.760958293`: 共有境界の内部 step 長、すなわち
   `median(|0305--0306|, |0306--0307|, |0307--0308|)` を
   `median(|0308--0309|, |0309--0310|)` で割った naive median-step merge。

それぞれ全38 camera centre を proper positive-scale Umeyama で audited calibration
proxy に整合した（proxy は merge の入力には使っていない）。

| model | registered | fit scale | proxy centre RMSE / median / max |
|---|---:|---:|---:|
| submodel 0 alone | 23/38 | 0.645685 | 2.964 / 1.816 / 5.540 cm |
| submodel 1 alone | 16/38 | 1.246450 | 2.466 / 1.523 / 5.217 cm |
| shared-camera merge, `s=1` | 38/38 | 0.891840 | 128.349 / 115.726 / 227.488 cm |
| shared-camera merge, boundary median-step `s=2.760958293` | 38/38 | 0.485911 | 60.623 / 54.405 / 94.321 cm |

median-step merge が単純 s=1 より良いのは scale の方向だけを補正した効果であり、
正しい単一モデルになったことを示さない。submodel 1 には内部的にも
`0308--0309=0.5820`, `0309--0310=0.6094` に対して
`0310--0311=7.1739` の大きな不連続があり、共有 anchor だけではこの basin を
修復できない。

点を単純連結すると 31,123 points / 125,861 observations になるが、これは有効な
merged sparse model ではない。共有 `DSC_0308` の観測は submodel 0 が170、submodel 1
が508で、feature index の共通が8件あり、その8件は全て異なる point ID を指して
いる。同一 camera 内の conflict を解消する cross correspondence/point merge 規則
がないため、点と track をそのまま BA に渡すこと、また公式 pose を補って BA を
回すことはしなかった。独立モデルの各点統計だけを上表に示した。

## 結論と次の候補

- 共有 pose で回転を合わせることはできるが、排他的成分間の verified E が0本で、
  shared-anchor E 3本は scale を拘束しない。方向行列が見かけ上非共線でも、同一始点
  のため相対 Sim(3) の translation/scale を安定に観測できない。
- 非 GT の A/B は 60.62cm（境界 median-step）/128.35cm（s=1）で、各独立モデルの
  2.47--2.96cmを大幅に悪化させる。したがって completeness 38/38だけを根拠に
  submap merge を選ぶべきではない。
- production の一般 submap merge、方向制約 selector、または BA 配線は追加しない。
  最小の有効な次実験は、まず同一 reconstruction に成長させるための、排他的成分間
  verified bridge（少なくとも異なる始点を持つ非共通 cross edge を複数本）を得る
  ことである。実装の scale solve は、その後に cross-edge の非共線性、positive
  cheirality、translation-direction residual、rank/condition をゲートする形にする。

解析は既存 high-resolution artifact を変更せず、公式 extrinsics/GT を fitting に
使わず実施した。submodel 単独の proxy 数値と merge 後の数値は、独立 laser pose が
ローカルに無いという [evaluator audit](evaluator_audit_20260830.md) の caveat に
従い、calibration proxy として解釈する。
