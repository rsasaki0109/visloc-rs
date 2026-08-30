# ETH3D courtyard 評価器・基準フレーム監査 (2026-08-30)

この文書は、courtyard の camera-centre Sim(3) 評価が何を基準にしているかを、
共有ツリーを変更せずに再計算した記録である。公式 extrinsics を mapper の入力には
使っていない。

## 結論

- ローカルの `gt` は独立したレーザーカメラ姿勢ファイルではない。
  `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/gt` は
  `/home/sasaki/datasets/eth3d/courtyard/dslr_calibration_undistorted` への symlink で、
  `gt/images.txt` と公式 calibration `images.txt` はサイズ **7,146,608 bytes**、
  SHA-256 **1afcc917c0538cf7168ca9c045574a786402553712be0ad2ae5affdc52b87f02** の
  同一ファイルである。したがって official-vs-`gt` の 0 cm は同一ファイルを比較した
  恒等値であり、独立したレーザースキャン精度の証拠ではない。
- 現地にある独立な camera-pose GT はこの調査では見つからなかった。ETH3D は
  calibration (`cameras.txt`/`images.txt`) と scan evaluation archive を別に配布するため、
  laser evaluation archive を用意しない限り、以下の「laser-GT」スコアは既存スクリプトが
  `gt` symlink を基準にした **laser-aligned camera calibration proxy** と表記するのが正確である。
- 評価器の数値バグは確認できない。独立 NumPy 実装は、COLMAP の qvec から
  `C=-R^T t` を計算し、proper-rotation/positive-scale Umeyama を再実装した。既存の
  `scripts/score_umeyama_centers.py` と全モデルで表示精度内一致し、共分散の転置を
  変えた二つの実装も RMSE/回転を約 `1e-14` 以内で一致した。
- 真の full COLMAP model はこの proxy に対して **38/38, 1.709 cm RMSE**（median
  1.170 cm, max 4.132 cm）であり、現在の COLMAP-feature champion は **38/38,
  2.842 cm**（2.243 cm, 7.091 cm）である。したがって残る差は evaluator よりも
  frontend の対応・track/PnP の basin・mapper/BA の問題として扱うべきである。
- handover の「sub-cm」は現時点で独立 GT によって実証された閾値ではない。proxy の
  best measured control が 1.709 cm であるため、sub-cm は aspirational target とし、
  まず独立 scan-evaluation pose/GT を取得してプロトコルを固定する必要がある。

## 入力・対応関係

| 名称 | パス | 枚数 | 備考 |
|---|---|---:|---|
| 公式 calibration / `gt` proxy | `/home/sasaki/datasets/eth3d/courtyard/dslr_calibration_undistorted/{cameras,images,points3D}.txt` | 38 | `gt` symlink の実体。4 camera IDs。 |
| full COLMAP reference | `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_oracle_full/sparse_txt` | 38 | 正規化 1600x1066、共有 PINHOLE。 |
| low COLMAP reference | `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_oracle/sparse_txt` | 24 | **cam0 の 0323 + cam1 の23枚だけ**。cam2/cam3は含まない。 |
| COLMAP-feature champion | `/tmp/ba_champion_control_20260829/images.txt` | 38 | mapper control。 |
| own-SIFT historical | `/tmp/repro_31p84_exact_run1_20260829/images.txt` | 38 | 31.84 cm control。 |
| own-SIFT reverse | `/tmp/repro_31p84_reverse_artifact_20260829/images.txt` | 38 | 13.996 cm control。 |
| global edge-scale v5 | `/tmp/eth3d_courtyard_highres_cap8192_global_edge_scales_v5_20260830/images.txt` | 38 | 38/38 でも姿勢が悪い反例。 |

公式 calibration は 38 unique stems で、camera ID 別には `cam0=1`、`cam1=23`、
`cam2=2`、`cam3=12` である。公式の画像名は
`dslr_images_undistorted/DSC_####.JPG`、正規化/mapper の出力は PNG だが、評価時は
`Path(name).stem` で対応し、38件の一意な stem が一致する。low model の実際の共通集合は
次の24件である。

```text
DSC_0286..DSC_0292,
DSC_0302..DSC_0307,
DSC_0310..DSC_0315,
DSC_0319..DSC_0323
```

評価器は推定ファイルに無い画像を補完せず、`est` と `gt` の name intersection のみを
採点する。よって low model の 3.59 cm は 24/38 の registered-subset score であり、
full-scene 38枚のスコアではない。

公式の `cameras.txt` は次の4つの undistorted `PINHOLE` camera blockである。

```text
3 PINHOLE 6208 4134 3408.35 3408.8 3114.7 2070.92
2 PINHOLE 6200 4134 3407.41 3408.08 3112.83 2065.6
1 PINHOLE 6205 4135 3409.58 3409.44 3115.16 2064.73
0 PINHOLE 6198 4129 3411.42 3410.02 3116.72 2062.52
```

full/low reference の `rigs.txt` は両方とも `1 1 CAMERA 1` だけで、sensor pose や
複数センサーの lever-arm は記録していない。公式画像も camera ID ごとの上記寸法と
一致する。したがって、現存メタデータから camera centre に別の rig/sensor 変換を
適用する根拠はない。

## 評価式と独立検算

各 `images.txt` の1行目を `IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME` として読み、
q を正規化する。Hamilton qvec の回転を

```text
R = [[1-2(y²+z²), 2(xy-zw), 2(xz+yw)],
     [2(xy+zw), 1-2(x²+z²), 2(yz-xw)],
     [2(xz-yw), 2(yz+xw), 1-2(x²+y²)]]
C = -Rᵀ t
```

で camera centre に変換した。推定中心 `X` から基準中心 `Y` への
`Y ~= s Q X + b` を、中心化共分散の SVD と determinant correction で解いた。
det が負の場合のみ最後の軸を反転して proper rotation とし、scale は正の
Umeyama scale、誤差は変換後中心の Euclidean norm とした。従って表示 scale は
**estimate → gt** の向きであり、reflection を許す評価ではない。

独立実装による sanity check は次の通り。

- 乱数38点に既知の scale 2.75 の Sim(3) を適用すると、推定 scale
  `2.7500000000000004`、最大中心誤差 `3.2e-15`、回転行列誤差 `6.5e-16`。
- q と -q の回転行列の最大差は `0.0`。
- 既存 scorer と独立 parser の全モデル結果は少なくとも表示6桁で一致した。low model の
  再実行値は `matched=24/38`, `sim3_scale=1.142832`, `rmse_cm=3.59`,
  `median_cm=2.49` である。

## 全体スコア（基準は公式 calibration/`gt` proxy）

単位は cm、scale は estimate→proxy、q は中心誤差の分位点である。

| estimate | common | scale | RMSE | q25 | median | q75 | q90 | q95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official calibration (=gt) | 38/38 | 1.000000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| COLMAP full | 38/38 | 1.261374 | 1.709 | 0.890 | 1.170 | 1.767 | 2.699 | 3.107 | 4.132 |
| COLMAP low | 24/38 | 1.142832 | 3.590 | 2.038 | 2.488 | 3.045 | 4.555 | 5.155 | 11.487 |
| COLMAP-feature champion | 38/38 | 0.577188 | 2.842 | 1.641 | 2.243 | 2.892 | 4.137 | 4.970 | 7.091 |
| own-SIFT historical | 38/38 | 0.559770 | 31.844 | 7.939 | 17.944 | 21.475 | 43.828 | 83.412 | 105.723 |
| own-SIFT reverse | 38/38 | 0.497024 | 13.996 | 4.534 | 6.051 | 7.422 | 11.693 | 23.232 | 68.451 |
| global edge-scale v5 | 38/38 | 3.399940 | 468.396 | 345.196 | 410.243 | 497.421 | 666.671 | 750.006 | 882.728 |

同じ24枚だけに制限した比較（low model の common set）も確認した。

| estimate | common | scale | RMSE | median | max |
|---|---:|---:|---:|---:|---:|
| COLMAP low | 24 | 1.142832 | 3.590 | 2.488 | 11.487 |
| COLMAP full restricted to 24 | 24 | 1.261373 | 1.572 | 1.130 | 3.415 |
| champion restricted to 24 | 24 | 0.577351 | 2.549 | 2.355 | 4.511 |
| own-SIFT historical restricted to 24 | 24 | 0.559526 | 31.079 | 19.523 | 101.566 |
| own-SIFT reverse restricted to 24 | 24 | 0.497226 | 8.686 | 6.523 | 18.724 |

比較の向きを明示するため、公式中心を source、full/low COLMAP中心を destination に
した別 fit も行った。official→full は **1.355 cm RMSE / 0.929 cm median /
3.273 cm max / scale 0.792776**、official→low は common 24枚で **3.141 cm /
2.187 cm / 10.050 cm / scale 0.874967** だった。Sim(3) は非対称な least-squares
fitなので、estimate→official の1.709 cmと数値が異なること自体はバグではない。

## camera ID 別の残差構造

以下は full 38枚を各モデルの一つの Sim(3) で整列した後、公式 camera ID ごとに集計した
値である。`mean/RMSE; q25/q50/q75/q95/max` の順、単位は cm。

| official camera ID | 枚数 | COLMAP full | champion |
|---:|---:|---|---|
| 0 (`0323`) | 1 | 1.664/1.664; 1.664/1.664/1.664/1.664/1.664 | 4.527/4.527; 4.527/4.527/4.527/4.527/4.527 |
| 1 | 23 | 1.400/1.608; 0.855/0.974/1.860/2.894/3.537 | 2.322/2.518; 1.585/1.905/2.787/3.942/4.953 |
| 2 (`0296,0297`) | 2 | 1.151/1.160; 1.076/1.151/1.225/1.284/1.299 | 1.844/1.844; 1.836/1.844/1.852/1.859/1.860 |
| 3 | 12 | 1.692/1.961; 0.943/1.441/1.951/3.526/4.132 | 2.990/3.340; 2.244/2.411/2.919/5.977/7.091 |

full COLMAP の最大 outlier は `DSC_0308=4.132`, `0305=3.537`, `0301=3.031`,
`0304=2.926`, `0303=2.601` cm である。champion は `DSC_0316=7.091`,
`0308=5.066`, `0311=4.953`, `0323=4.527`, `0313=3.970` cm が最大である。
own-SIFT historical の大きな誤差は `0307=105.723`, `0316=84.676`, `0306=83.189`,
`0308=66.967`, `0305=33.911` cm に集中しており、評価器の一様な scale/reflection
ミスでは説明できない。

## laser/rig 解釈と次の優先順位

ETH3D の [公式 documentation](https://www.eth3d.net/documentation) は calibration を
COLMAP形式で提供し、undistorted画像を PINHOLE として扱い、`images.txt` の pose を
global-to-local（原点は projection center）として説明している。[dataset page](https://www.eth3d.net/datasets)
は courtyard の camera画像と `courtyard_dslr_scan_eval` を別 archive として掲載している。
また ETH3D の [データセット説明](https://www.eth3d.net/data/schoeps2017cvpr.pdf) では
DSLR と laser scan の登録・refinement が別段階で行われる。従って公式 calibration は
laser-aligned camera reference として有用だが、ローカルの symlink だけで「独立 laser
GT」と呼ぶのは不正確である。

`rigs.txt` に sensor pose が無く、ローカルに lever-arm/extrinsic 補正表も無いので、
camera centre に追加の rig transform を挿入する production 修正は行わない。既存の
lever-arm fit は oracle 診断としては一部の誤差を下げるが、軌跡半分ごとの offset が
不安定で calibration metadata に対応しないため採用根拠にならない。

したがって次の高レバレッジ項目は evaluator ではなく、(1) COLMAP級の verified
bridge/match を保った track/PnP 成長、(2) その後の mapper/BA が COLMAP basin を壊さない
こと、の順である。global-edge v5 のように 38/38 と低い局所 reprojection だけで
468.396 cm になる反例があるため、登録枚数や reprojection 単独で実験を選ばない。
独立 scan-evaluation pose/GT を取得できるまで、sub-cm は目標値として掲げるに留め、
すべての実験報告では common count と proxy の性質を併記する。

## 再現コマンド

権威スクリプト（low model の確認例）:

```bash
python3 /media/sasaki/aiueo1/visloc-rs/scripts/score_umeyama_centers.py \
  --est /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/colmap_oracle/sparse_txt/images.txt \
  --gt /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/gt/images.txt
```

独立検算は、このスクリプトと同じ入力を別の NumPy parser で読み、qvec→`-R.T@t`
および Umeyama を再計算した。今回の監査は read-only のため、official extrinsics を
mapperへ投入した A/B や、独立 archive が無い状態での production code 変更は実施して
いない。
