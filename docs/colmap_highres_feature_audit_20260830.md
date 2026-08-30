# 高解像度公式 COLMAP SIFT A/B (2026-08-30)

公式 ETH3D courtyard の元解像度画像に対して、Docker 内の公式 COLMAP
SIFT を抽出し、同じ特徴を visloc と COLMAP の mapper に投入した非破壊の
対照実験である。既存の feature directory、画像、repo の未コミット差分は
変更していない。先行の [evaluator audit](evaluator_audit_20260830.md) の
とおり、ローカル `gt` は独立 laser pose ではなく公式 calibration の
symlink なので、以下の `gt proxy` はその camera-centre proxy との比較を表す。

## 入力と provenance

- `colmap`/`pycolmap` のホスト実行ファイルは無かったが、Docker image
  `colmap/colmap:latest` は利用可能だった。実行時の表示は
  `COLMAP 4.2.0.dev0 (Commit Unknown on Unknown with CUDA)`、image ID/digest は
  `sha256:b809882552887b6471094dcadd2f2eb01656b010663564c43a5e7f04c0a08f2f`
  （作成 `2026-07-29T08:19:29Z`）。GPU は検出されなかったため CPU を使った。
- 入力画像は
  `/home/sasaki/datasets/eth3d/courtyard/images/dslr_images_undistorted`
  の38枚。camera dimensions は 6205x4135 (23枚)、6208x4134 (12枚)、
  6200x4134 (2枚)、6198x4129 (1枚) である。calibration は
  `/home/sasaki/datasets/eth3d/courtyard/dslr_calibration_undistorted` の
  4つの `PINHOLE` camera block を使用し、`cameras.txt` の SHA-256 は
  `0cf9d1f1615b89eed8e48a92ef0ee44352f0fbadbb1be0c00b12f15d01c1f83c`、
  `images.txt` は
  `1afcc917c0538cf7168ca9c045574a786402553712be0ad2ae5affdc52b87f02`。
- 公式抽出コマンドは次である（新規 DB `/tmp/colmap_official_highres_8192_20260830/database.db`）。

  ```bash
  docker run --rm \
    -v /home/sasaki/datasets/eth3d/courtyard/images/dslr_images_undistorted:/images:ro \
    -v /tmp/colmap_official_highres_8192_20260830:/out \
    colmap/colmap:latest colmap feature_extractor \
    --database_path /out/database.db --image_path /images \
    --ImageReader.camera_model PINHOLE --ImageReader.single_camera 0 \
    --FeatureExtraction.type SIFT --FeatureExtraction.use_gpu 0 \
    --FeatureExtraction.num_threads 1 \
    --SiftExtraction.max_num_features 8192 \
    --SiftExtraction.first_octave -1 --SiftExtraction.num_octaves 4 \
    --SiftExtraction.octave_resolution 3 \
    --SiftExtraction.peak_threshold 0.00667 \
    --SiftExtraction.edge_threshold 10 \
    --SiftExtraction.max_num_orientations 2 \
    --SiftExtraction.upright 0 --SiftExtraction.estimate_affine_shape 0 \
    --SiftExtraction.domain_size_pooling 0
  ```

  実測は **5.789分**。DB は38画像、keypoint/descriptor blobとも38行、総
  **439,481 rows**（画像ごと **9,663–17,689**）である。`max_num_features`
  は orientation expansion 後の row 数を8192に単純固定する指定ではないため、
  `max_num_orientations=2` と合わせて8192を超える行がある。抽出 DB の SHA-256 は
  `b676d6fcde13ff3e44b38a125348f251b8159b01860e9dbc404eaf0086a7acb1`。

## visloc への変換と実行

DB の各 keypoint 6個の float32 値と descriptor 128個の uint8 値を、DB row
順のまま `x y a11 a12 a21 a22 d0...d127` へ変換した。変換先は
`/tmp/colmap_official_highres_8192_20260830/features_sixcol`、38ファイル、
**439,481 rows**で、全rowについてDB blobと値が一致することを検証した。
manifest は `MANIFEST.tsv`、SHA-256 は
`030b02982e263f3b5e3d94d1edd873b757894b4c20a9609feef264cadffd0d2a` である。

visloc の実行コマンド（`--cross-check` はこのCLIには無く、NN matcher の暗黙
cross-checkを使用）は次である。

```bash
RAYON_NUM_THREADS=1 VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_IMAGES=20,21 \
target/release/examples/unordered_sfm_demo \
  --feature-extractor files \
  --features-dir /tmp/colmap_official_highres_8192_20260830/features_sixcol \
  --feature-suffix _features.txt --image-suffix .JPG \
  --images-dir /home/sasaki/datasets/eth3d/courtyard/images/dslr_images_undistorted \
  --input-colmap-calibration /home/sasaki/datasets/eth3d/courtyard/dslr_calibration_undistorted \
  --exhaustive --pair-stem-window 3 --min-matches 20 --match-ratio 0.8 \
  --guided-matching --verification-mode full --mapper incremental \
  --pnp-max-iterations 100000 --min-pnp-inliers 8 \
  --geometry-guided-conflict-recovery --post-refinement-registration \
  --final-iterative-refinement \
  --out-colmap /tmp/colmap_official_highres_8192_visloc_n3_20260830
```

effective-config hash は `724c924e3ad50488`。実測は **108 candidate、100
verified、137,171 inlier correspondences**、configuration 内訳は
`CALIBRATED=28 UNCALIBRATED=72 PLANAR=1 DEGENERATE=7`。UnionFind 初期値は
49,180 components、399 conflict components、48,781 retained tracks /
140,386 observations。複数 seed のうち最終採用モデルは **26/38**（登録名
`DSC_0298`–`DSC_0323`）、29,639 tracks / 80,632 observations、mean reprojection
**0.451 px** で、`DSC_0297` の PnP が `65 -> 3` inliers（minimum 8）で停止した。
0306/0307/0308/0309 の最終成長ログはそれぞれ `34->33`、`9->8`、`254->245`、
`1221->1103` PnP inliers である。出力 images の SHA-256 は
`b6fe81cec7754ca86e51d0074231543610e5e122c251f601cb32ab0aa35d9e60`、
points3D は `b117f586024f9e205d711d66c3d7eaf7646a6f38d288f1b7cb85391d2e847c7c`。

`gt proxy` への proper positive-scale Umeyama は `matched=26/38`、scale
`1.986774`、RMSE **3.612 cm**、median **3.207 cm**、max **6.603 cm**。
公式 extrinsics は mapper に投入していない。個別の高解像度 bridge 診断は次の通り
（visloc の `Uncalibrated` row、括弧内は E/F/H inlier）。

| pair | raw | accepted | E/F/H |
|---|---:|---:|---:|
| 0307–0308 | 357 | 284 | 91/284/6 |
| 0308–0309 | 699 | 617 | 391/617/130 |
| 0309–0310 | 2,012 | 1,926 | 1,544/1,926/35 |

## 公式 COLMAP matcher/mapper 対照

抽出 DB の一時コピーへ calibration の camera params を設定し、公式 CPU
`sequential_matcher`（overlap 3、ratio 0.8、cross-check、guided matching、
`max_error=4`、random seed 0）を実行した。`FeatureMatching.num_threads=8`、
`SequentialMatching.num_threads=1` で **14.799分**（最大 RSS 約2.3 GB）。元の
抽出 DB は変更せず、post-match DB
`/tmp/colmap_official_highres_8192_20260830/database_calibrated_threads8.db`
（post-match SHA-256 `b6d9ab9dffdac9c1cc921d6f82d7a9aa33fd0bf8032b53feeeb3c6761495d6c8`）を
使用した。結果は **107 match rows / 107 geometry rows**、raw match rows 合計
142,214、geometry `rows` 合計189,112。DB の `config` code 分布は
`0:7, 2:56, 3:14, 6:30` である。

matcher の実コマンドは次である。

```bash
docker run --rm \
  -v /tmp/colmap_official_highres_8192_20260830:/out \
  colmap/colmap:latest colmap sequential_matcher \
  --database_path /out/database_calibrated_threads8.db \
  --FeatureMatching.type SIFT_BRUTEFORCE --FeatureMatching.use_gpu 0 \
  --FeatureMatching.num_threads 8 --FeatureMatching.guided_matching 1 \
  --FeatureMatching.max_num_matches 32768 \
  --SiftMatching.max_ratio 0.8 --SiftMatching.max_distance 0.7 \
  --SiftMatching.cross_check 1 --SiftMatching.cpu_brute_force_matcher 1 \
  --TwoViewGeometry.max_error 4 --TwoViewGeometry.min_num_inliers 15 \
  --TwoViewGeometry.random_seed 0 --SequentialMatching.overlap 3 \
  --SequentialMatching.quadratic_overlap 1 --SequentialMatching.expand_rig_images 0 \
  --SequentialMatching.loop_detection 0 --SequentialMatching.num_threads 1
```

重要な遷移の公式値は次の通り。公式とvislocはguided matching、幾何モデル選択、
row保存規則が異なるため、同じ特徴からの独立対照であり、inlier数の直接同一性は
要求していない。

| pair | COLMAP raw | COLMAP geometry rows |
|---|---:|---:|
| 0307–0308 | 356 | 731 |
| 0308–0309 | 694 | 1,276 |
| 0309–0310 | 1,996 | 2,445 |

公式 mapper の単一モデルコマンドは以下である（official extrinsics無し）。

```bash
docker run --rm \
  -v /home/sasaki/datasets/eth3d/courtyard/images/dslr_images_undistorted:/images:ro \
  -v /tmp/colmap_official_highres_8192_colmap_mapper_n3_20260830:/out \
  colmap/colmap:latest colmap mapper \
  --database_path /out/database.db --image_path /images \
  --output_path /out/sparse \
  --Mapper.multiple_models 0 --Mapper.max_num_models 1 --Mapper.num_threads 8 \
  --Mapper.random_seed 0 --Mapper.extract_colors 0 \
  --Mapper.init_num_trials 200 --Mapper.init_min_num_inliers 100 \
  --Mapper.init_max_error 4 --Mapper.init_min_tri_angle 16 \
  --Mapper.abs_pose_min_num_inliers 30 --Mapper.abs_pose_max_error 12 \
  --Mapper.filter_max_reproj_error 4 --Mapper.filter_min_tri_angle 1.5 \
  --Mapper.ba_use_gpu 0
```

実測は **23/38**（`DSC_0286`–`DSC_0308`）で、16,714 points / 68,797
observations、mean point reprojection/error **0.668 px**、proxy RMSE **2.964 cm**、
median **1.816 cm**、max **5.540 cm**（scale `0.645685`）。公式 mapper も
0308–0309 geometryをDBに持つが、単一モデルの成長は0308で止まり、0309を登録しなかった。

補助的に `Mapper.multiple_models=1,max_num_models=5` を一度だけ実行したところ、
model 0 は上記23枚（16,714 points/68,797 obs）、model 1 は
`DSC_0308`–`DSC_0323` の16枚（14,409 points/57,064 obs、mean 0.602 px）だった。
model 1単独のproxyは RMSE **2.466 cm**、median **1.523 cm**、max **5.217 cm**。
0308が両モデルに重複するが、2モデルは別 gauge のためこれらを結合した38枚の単一
Sim(3)精度とは報告していない。公式mapperのmulti-model実行時間は **0.589分**。

## 比較と判断

| path | registered | tracks/obs | mean reproj | gt-proxy RMSE |
|---|---:|---:|---:|---:|
| own prefix8192 control | 23/38 | — | — | 4.51 cm |
| own full 518,015-row control | 23/38 | — | — | 5.02 cm |
| existing COLMAP-feature champion | 38/38 | — | — | 2.842 cm |
| official supplied calibration model | 38/38 | — | — | 1.709 cm |
| fresh official SIFT → visloc | 26/38 | 29,639/80,632 | 0.451 px | 3.612 cm |
| fresh official SIFT → COLMAP mapper | 23/38 | 16,714/68,797 | 0.668 px | 2.964 cm |

今回の公式抽出は、特徴行・descriptor bytesをDBから完全保存した genuine
high-resolution COLMAP SIFT であり、既存のlow-resolution/過去版artifactとの差を
埋める再現可能な対照になった。一方、公式COLMAPとvislocはともにN=3隣接候補で
0308–0309を十分なraw/verified対応として保持していても単一モデルの成長境界が
一致しない。従って現時点の最大のボトルネックは「公式SIFTのraw特徴不足」だけではなく、
モデル選択後の track/PnP 成長・seed/order・単一モデル接続である。高解像度公式SIFT
単独でも38/38単一再構成には到達せず、既存 champion 38/38/2.842 cm を更新しないため、
新たな heuristic や default変更は行わない。

実験ログは `/tmp/colmap_official_highres_8192_visloc_n3_20260830_retry.log`、
`/tmp/colmap_official_highres_8192_colmap_sequential_match8_20260830.log`、
`/tmp/colmap_official_highres_8192_colmap_mapper_n3_20260830.log`、および
`...mapper_multi_n3...log` に保存した。repo sourceの変更はこの実験では行っていない。
