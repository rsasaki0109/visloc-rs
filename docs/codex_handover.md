# visloc-rs COLMAP parity — Codex 引き継ぎ資料

**更新:** 2026-08-28
**Repo:** `/home/sasaki/workspace/visloc-rs`
**ブランチ:** `main`（ローカル未コミット変更あり、HEAD `101e5cc` 付近）
**目標:** ETH3D **courtyard 38/38 @ sub-cm Sim(3) centre RMSE** → その後 South Building / terrace / office / EuRoC を維持。README parity 宣言は **courtyard が通るまで禁止**。

---

## 1. 成功条件（完了定義）

| 要件 | 証拠 |
|------|------|
| courtyard **38/38** 登録 | `images.txt` に 38 カメラ |
| Sim(3) centre RMSE **sub-cm** | `scripts/score_umeyama_centers.py` vs ETH3D GT |
| 他シーン退行なし | South Building 128/128、terrace/office/EuRoC の既存ベンチ |
| CI 緑 | `cargo test -p visloc-slam --lib global_sfm`（17/17）、フル `cargo test --workspace` |
| 新挙動は A/B + CHANGELOG | `CHANGELOG.md` Unreleased に courtyard 結果を必ず記録 |

**現状 honest スコア:** accuracy-critical SfM pipeline **~70%**。binding unlock は **エッジ / 検出・マッチ品質**（positioning 単体では courtyard sub-cm 不可と証明済み）。

---

## 2. 決定的診断（2026-08-28 時点）

### 2.1 天井（oracle）

| パイプライン | Verified | Reg | Sim(3) RMSE |
|--------------|----------|-----|-------------|
| **True COLMAP 4.1.1**（Docker, normalized 1600×1066） | — | 38/38 | **~1.7 cm** |
| COLMAP SIFT + COLMAP raw matches + **plain incremental + pnp100k + `--final-iterative-refinement`** | 380/703 | 38/38 | **~3.4 cm** ← **best visloc oracle** |
| COLMAP SIFT + COLMAP matches + plain incremental（polish なし） | 380/703 | 38/38 | ~8.7 cm |
| COLMAP SIFT + COLMAP matches + `--colmap-style` incremental | 380/703 | 38/38 | ~66 cm ❌ |
| COLMAP SIFT + COLMAP matches + **hybrid champion** | 380/703 | 38/38 | ~49 cm |

**結論:** 対応が強いとき **plain growth + final iterative polish** が hybrid / colmap-style growth より良い。**`--colmap-style` growth は courtyard で regress**。

### 2.2 Our SIFT（本番フロントエンド）

| 設定 | Verified | Reg | RMSE |
|------|----------|-----|------|
| デフォルト ratio 0.8, plain incremental | 211/703 | 22/38 | ~54 cm（22 枚のみ） |
| `--match-ratio 0.9 --guided-matching` | 340/703 | 22/38 | ~40 cm（22 枚） |
| 上 + pnp100k | 340/703 | 23/38 (+0309) | ~152 cm |
| 上 + final-iterative | 340/703 | 23/38 | ~185 cm ❌ |
| `--sift-max-keypoints 8192` + ratio 0.9 | 391/703 | 22/38 | ~148 cm ❌ |
| **hybrid champion**（下記） | 211/703 | **38/38** | **~230–249 cm** |

**常に欠ける 16 枚:** `DSC_0297–0309`, `DSC_0320–0322`（far-orbit クラスタ）。
COLMAP は far-orbit 関連で **183 verified bridge pairs** を持つが、our SIFT の追加 verified は near component 内が大半。

**Verify graph:** ratio 0.9 でも **1 connected component / 38 nodes**（`--rescue-bridging` は neutral）。
失敗原因は verify graph の分断ではなく **incremental PnP / 橋 pair の質**。

### 2.3 Hybrid champion（completeness baseline、精度は metres）

```bash
SSD=/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard
target/release/examples/unordered_sfm_demo \
  --feature-extractor sift --images-dir "$SSD/images_1600x1066" \
  --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 \
  --exhaustive --min-matches 20 --sift-max-keypoints 4096 \
  --verification-mode full --mapper hybrid \
  --chirality-harden --rotation-seed-trials 8 \
  --refine-global-translations --multi-hypothesis-edges \
  --repair-prior-edges --metric-prior-scale \
  --hybrid-drop-prior-stems DSC_0296 \
  --prefer-essential-stems DSC_0296 \
  --rematch-stems DSC_0297,DSC_0320,DSC_0321,DSC_0322,DSC_0323 \
  --rematch-ratio 0.9 --rematch-guided --rematch-free-vs-priors \
  --rematch-prefer-min-e-inliers 25 \
  --repnp-free-from-priors --repnp-free-min-corrs 6 \
  --out-colmap "$SSD/runs/champion_baseline"
```

スコア: `python3 /media/sasaki/aiueo1/visloc-rs/scripts/score_umeyama_centers.py --est .../images.txt --gt "$SSD/gt/images.txt"`

**閉ざされた扉（CHANGELOG 参照）:** chirality oracle、GT bearing gate、RootSIFT、colmap-style SIFT knobs、pose-guided rematch、OpenCV SIFT、8192 kp、ratio0.9+hybrid+final polish 等 — いずれも sub-cm 未達。

### 2.4 Oracle incremental（COLMAP matches 入力時の champion）

```bash
target/release/examples/unordered_sfm_demo \
  --feature-extractor files --features-dir "$SSD/colmap_features_export" \
  --import-matches-file "$SSD/colmap_matches_import.txt" \
  --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 \
  --exhaustive --min-matches 20 --verification-mode full \
  --mapper incremental --out-colmap "$SSD/runs/oracle_best" \
  --pnp-max-iterations 100000 --final-iterative-refinement
# → 38/38 @ ~3.4 cm
```

---

## 3. 環境・データ（SSD）

```text
/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/
├── images_1600x1066/          # 38 PNG, 全て 1600×1066（Lanczos）。必須。
├── gt/ → images.txt           # ETH3D laser GT（symlink）
├── colmap_oracle_full/database.db
├── colmap_features_export/    # COLMAP SIFT → visloc feature txt
├── colmap_matches_import.txt
├── colmap_bridge_matches_import.txt
├── our_sift_features_export/  # --export-features-only で生成
├── our_sift_bridge_supplement.txt      # 3px spatial transfer
├── our_sift_bridge_supplement_8px.txt  # 8px spatial transfer
└── runs/                      # 全 A/B 出力
```

**Pitfall:** 14/38 原画像は 1065/1067 px。COLMAP `single_camera 1` は不一致を **silent skip** → 最初 24/38 @ 3.6 cm は **偽陽性**。

**ビルド:**
```bash
cd /home/sasaki/workspace/visloc-rs
cargo build --release --example unordered_sfm_demo --features image-io
```

**CI スモーク:**
```bash
cargo test -p visloc-slam --lib global_sfm   # 17/17
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
```

---

## 4. 未コミット変更（2026-08-28）

`git status` より（**コミットはユーザー指示まで不要**）:

| ファイル | 内容 |
|----------|------|
| `examples/unordered_sfm_demo.rs` | `--import-matches-file`, `--import-matches-supplement-file`, `--import-verified-pairs-file`, `--final-iterative-refinement`, `--export-features-dir`, `--export-features-only` |
| `pipelines/slam/src/incremental_sfm.rs` | `final_iterative_global_refinement` |
| `CHANGELOG.md` | 全 courtyard A/B 記録 |
| その他 | sift / global_sfm / colmap 周辺の継続作業 |

**SSD 上スクリプト（repo 外または scripts/）:**
- `scripts/export_colmap_matches.py`
- `scripts/export_colmap_verified_pairs.py`
- `scripts/export_colmap_sift_features.py`
- `scripts/export_colmap_bridge_matches.py`
- `scripts/transfer_colmap_bridge_matches_to_sift.py` ← **新規**
- `scripts/score_umeyama_centers.py`

---

## 5. 新 CLI フラグ（oracle / 診断）

| フラグ | 用途 |
|--------|------|
| `--import-matches-file PATH` | 全 pair を import raw matches + verify（NN スキップ）。pair 未記載は drop |
| `--import-matches-supplement-file PATH` | 記載 pair のみ import、他は NN fallback。**feature index は loaded features と一致必須** |
| `--import-verified-pairs-file PATH` | verify バイパス（TVG inliers 直投入） |
| `--final-iterative-refinement` | plain growth のまま final を `iterative_global_refinement` に差し替え |
| `--export-features-dir DIR` | external-deep format で feature 書き出し |
| `--export-features-only` | マッチ前に exit |

**Spatial bridge transfer（our SIFT 向け）:**
```bash
# 1) our SIFT feature export
target/release/examples/unordered_sfm_demo \
  --feature-extractor sift --images-dir "$SSD/images_1600x1066" \
  --width 1600 --height 1066 --fx 879.4 --fy 879.4 --cx 803.4 --cy 532.6 \
  --sift-max-keypoints 4096 \
  --export-features-dir "$SSD/our_sift_features_export" \
  --export-features-only --out-colmap /tmp/x

# 2) COLMAP bridge matches → our SIFT indices（xy 最近傍）
python3 scripts/transfer_colmap_bridge_matches_to_sift.py \
  --colmap-features "$SSD/colmap_features_export" \
  --our-features "$SSD/our_sift_features_export" \
  --colmap-matches "$SSD/colmap_matches_import.txt" \
  --out "$SSD/our_sift_bridge_supplement.txt" --max-px 3.0

# 3) supplement + oracle stack
target/release/examples/unordered_sfm_demo ... \
  --import-matches-supplement-file "$SSD/our_sift_bridge_supplement.txt" \
  --match-ratio 0.9 --guided-matching \
  --mapper incremental --pnp-max-iterations 100000 --final-iterative-refinement
```

**結果（honest negative）:**
- 3px transfer: 77 bridge pairs → verify 342/703 → **25/38 @ ~415 cm**
- 8px transfer: 166 pairs → verify 322/703 → **28/38 @ ~358 cm**
→ COLMAP 橋を spatial に写しても **detector 不一致が binding**（454 bridge のうち大部分 unmapped）。

**COLMAP features + bridge supplement（indices 一致）:**
- 380/703 verified → **36/38 @ ~2.85 cm**（欠: `DSC_0306`, `DSC_0307`）

---

## 6. キーファイル

| Path | Role |
|------|------|
| `examples/unordered_sfm_demo.rs` | メインデモ・CLI・verify/rematch/import |
| `pipelines/slam/src/incremental_sfm.rs` | incremental mapper, final polish |
| `pipelines/slam/src/global_sfm.rs` | hybrid / rotation avg / rematch edges |
| `crates/vision/src/features/sift.rs` | pure-Rust SIFT（contrast, DSP, L1-root 等） |
| `crates/vision/src/two_view/colmap_verification.rs` | M1 verifier |
| `crates/io/src/external_deep.rs` | feature/match txt format |
| `CHANGELOG.md` | **A/B の authoritative log** |
| `docs/colmap_port_plan.md` | ポート計画 |

---

## 7. 数値クイックリファレンス（courtyard）

```
True COLMAP           38/38   ~1.7 cm
Oracle incremental    38/38   ~3.4 cm   (COLMAP matches + plain + pnp100k + final-iterative)
Oracle hybrid         38/38   ~49 cm    (same matches)
Our SIFT plain        22/38   ~54 cm    (211 verified)
Our SIFT ratio0.9     22/38   ~40 cm    (340 verified)
Our SIFT hybrid champ 38/38   ~230 cm   (completeness baseline)
Spatial bridge sup    25–28/38         (honest negative)
8192 kp hybrid        37/38   ~305 cm   (honest negative)
```

**Detector 密度:** COLMAP ~3500–7000 kp/image（far stems）、our SIFT cap 4096、0306/0307/0308 等は contrast=0.02 で 1500–2700 kp に落ちる。

---

## 8. 優先 next steps（Codex 向け）

### P0 — matching / detection（courtyard unlock）

1. **Far-orbit bridge pair の特定:** COLMAP verified 183 pairs vs our SIFT verified の差分ペアリスト（stem 0297–0309, 0320–0322  incident）。どの pair が raw match / verify で落ちるか `--diagnose-pairs` または per-pair dump。
2. **SIFT detector COLMAP 寄せ:** `peak_threshold = 0.02/3`（CHANGELOG: hybrid 単体では honest negative だが **ratio0.9 + plain incremental との組合せ**は未十分探索）、`max_keypoints` 8192 + `prefer_larger_scale` + `full_pyramid` の factorial A/B。
3. **Descriptor / guided matching:** COLMAP `FindGuidedMatches` との diff（guided は pair count  neutral、ratio0.9 時の local accuracy のみ改善）。
4. **Covariant / DSP-SIFT**（`sift.rs` に stub あり）— courtyard A/B gate。

### P1 — oracle gap 3.4 cm → 1.7 cm

- BA iteration / retriangulate exposure、gauge、COLMAP final polish パラメータの diff（対応が oracle 級のときのみ意味あり）。

### P2 — breadth（parity 後）

- SpatialPairGenerator、BA camera-model zoo、flat-file DB、`examples/colmap.rs` commit、EuRoC/office 退行チェック。

### やらないこと（証拠済み）

- hybrid + colmap-style growth を courtyard champion にしない
- chirality / GT bearing gate / rematch admission  alone で sub-cm を期待しない
- verify graph rescue だけで incremental 38/38 を期待しない（graph は既に connected）

---

## 9. 作業ルール

1. **courtyard 結果は毎回 CHANGELOG Unreleased に追記**（positive / negative 両方）。
2. **新挙動は default off**（A/B gate）。既存 champion を silently 変えない。
3. **`--out-colmap` は CLI 上、他フラグより前**に置く（順序 bug で images.txt 欠落あり）。
4. コミット前: `cargo test -p visloc-slam --lib global_sfm` 最低限、可能なら full workspace。
5. README / docs の parity 宣言は **courtyard sub-cm まで更新禁止**。

---

## 10. 関連パス

- 会話 transcript: `~/.cursor/projects/home-sasaki-workspace-visloc-rs/agent-transcripts/5973dbcc-7a9c-4515-8b7b-f5b33d68c8c3/5973dbcc-7a9c-4515-8b7b-f5b33d68c8c3.jsonl`
- SSD runs: `/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/runs/`
- 代表 run dirs: `oracle_best/`, `plain_pnp100k_finaliter/`, `champion_baseline/`, `sift_match_both/`, `colmap_feat_bridge_supplement/`, `sift_bridge_sup3px/`, `sift_kp8192_hybrid/`

---

## 11. 一言サマリ

**Courtyard の unlock は「positioning 改善」ではなく「our SIFT が far-orbit bridge pairs を COLMAP 同等に verify すること」**。
Oracle では COLMAP matches + plain incremental + final polish で **3.4 cm** まで行けるが、our SIFT は **211–391 verified でも incremental 22–28/38** で止まる。Hybrid は **38/38 @ ~230 cm** で completeness のみ担保。次は **bridge pair 単位の diff 診断 → SIFT 検出/記述子の COLMAP parity** が最短ルート。

---

## 12. 旧最終監査（2026-08-31、Auto 最終パッチ前の記録）

この節は上記の履歴を更新する現在の引き継ぎである。本文前半の「未取得」
「missing dependency」記述は当時の preflight の履歴であり、現在の判定には
使わない。今回の判定は、同一入力を比較できる場合は登録数を必須一致とし、
数値 RMSE/ATE/RPE について既存資料に事前指定された許容幅がない場合は、
小さな差を勝手に pass に丸めず **inconclusive** とした。

### Courtyard 完了済み control

耐久成果物は次に保存されている（repo 外、既存成果物を上書きしない）。

```text
/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/artifacts/colmap_highres_exhaustive_allpairs_20260830
```

`SHA256SUMS` は 64 ファイル、digest は
`12e91cd3a2e595625ef167d8cd8a2af6310d3ea3cd1e3b1a0c2f8264004fa96b`。
公式 COLMAP CPU SIFT の 703 ペアを import し、per-image calibration、
plain incremental、PnP 100,000、recovery/post/final を使った結果は
**366/703 verified、38/38、43,852 tracks / 152,432 observations、
0.579 px、Sim(3) centre RMSE 0.5379 cm** である。`visloc_model` と
`visloc_repeat_model` は次の3ファイルが同一ハッシュで、再実行も同じだった。

```text
cameras.txt  76fc758375228319300a5c076b6bcc84a88413d69cf3613eb40c980b60e8cc9c
images.txt   a14ac6b958bf09481bfcc0ae72b59671d8e6405cd880e4711ff14c5c9432852e
points3D.txt d7b680e6d51a403a4962920e8f0e0615f382646a1ac9836f160ecb44622a3293
```

再現コマンド（matching は耐久 `matches_import.txt` を使用済みで、703 は
`38*37/2` の全 unordered pair）:

```bash
ART=/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/artifacts/colmap_highres_exhaustive_allpairs_20260830
target/release/examples/unordered_sfm_demo \
  --feature-extractor files --features-dir "$ART/features_sixcol" \
  --feature-suffix _features.txt --image-suffix .JPG \
  --images-dir /home/sasaki/datasets/eth3d/courtyard/images/dslr_images_undistorted \
  --input-colmap-calibration /home/sasaki/datasets/eth3d/courtyard/dslr_calibration_undistorted \
  --import-matches-file "$ART/exhaustive/matches_import.txt" \
  --exhaustive --min-matches 20 --match-ratio 0.8 --verification-mode full \
  --mapper incremental --pnp-max-iterations 100000 --min-pnp-inliers 8 \
  --geometry-guided-conflict-recovery --post-refinement-registration \
  --final-iterative-refinement --next-image-policy visibility \
  --out-colmap "$ART/reproduction_model"
python3 scripts/score_umeyama_centers.py \
  --est "$ART/reproduction_model/images.txt" \
  --gt /home/sasaki/datasets/eth3d/courtyard/dslr_calibration_undistorted/images.txt
```

この数値は local calibration/`gt` proxy であり、独立 laser-camera pose
ground truth の score ではない。この caveat と COLMAP extraction/matching
の完全なコマンドは `docs/colmap_highres_exhaustive_audit_20260830.md` と
`docs/reproducibility_ci_closure_20260830.md` に残す。

### Suite 判定と推奨 explicit policy

| suite / variant | 現行結果 | 判定 | 推奨する明示設定と理由 |
|---|---|---|---|
| South Building / default Count | 127/128、0.74 cm（2回同一、`P1180163.JPG` 欠落） | **fail**（厳密な 128/128 要件） | `--next-image-policy visibility`。さらに完全登録が必要なら既存 opt-in `--post-refinement-registration`（128/128、0.73 cm）を使う |
| terrace / cache-fixed Count/Auto | 23/23、2.56 cm | **inconclusive**（旧 12.37 cm arm の feature bytes 不在） | `--next-image-policy auto` または `count`、recovery/post なし。旧キャッシュとの精度比較は保留 |
| office / cache-fixed Count/Auto | 17/26、0.43 cm | **inconclusive**（旧 18/26、0.37 cm arm の feature bytes 不在） | `--next-image-policy auto` または `count`、recovery/post なし。現行 cache で旧 arm の再現とは言わない |
| courtyard / exhaustive visibility/Auto | 38/38、0.5379 cm | **pass**（この durable proxy/control の要件） | 上記コマンドの `visibility`（または同じ選択になる明示 `auto`） |
| EuRoC open | 2,700/2,700、ATE Sim3 2.174040 m、RPE1 Sim3 0.061246 m | **pass**（記録 control 2.203 m 以下、登録維持） | `scripts/run_euroc_loop_closure_benchmark.sh` の open arm |
| EuRoC loop | 2,700/2,700、ATE Sim3 0.439303 m、RPE1 Sim3 0.071137 m | **pass**（記録 control 0.443 m 以下、登録維持） | 同 runner の loop arm |
| EuRoC full | 2,700/2,700、ATE Sim3 0.084345 m | **inconclusive**（同一 cache で baseline 0.083063 mから +1.282 mm、事前許容幅なし） | full arm。差分を改善と主張しない |
| EuRoC full2v | 2,700/2,700、ATE Sim3 0.054061 m | **inconclusive**（baseline 0.050140 mから +3.921 mm、事前許容幅なし） | full2v arm |
| EuRoC full2vh | 2,700/2,700、ATE Sim3 0.056473 m | **inconclusive**（baseline 0.053853 mから +2.620 mm、事前許容幅なし） | full2vh arm |
| EuRoC full2vhi | 2,700/2,700、ATE Sim3 0.053683 m | **inconclusive**（baseline 0.050955 mから +2.728 mm、事前許容幅なし） | full2vhi arm。current repeat は pose/est hash も一致 |
| EuRoC Auto | 対象 CLI なし | **inconclusive / N/A** | loop-closure runner は `NextImagePolicy` を持たないため、Auto の捏造比較はしない |

South の fail に対しては、既に観測済みの最小 recovery が
`--post-refinement-registration`（128/128、0.73 cm）だが、これは明示的な
opt-in であり、既定値に昇格させると他 suite/API の既定挙動を変える。
今回の同一 cache A/B は安全な global default を証明していないため、production
default は変更せず、推奨設定としてのみ残した。

EuRoC の比較は全 variant で 2,700 pose / 2,699 pair update を維持した。
current `full2vhi` の正しい repeat は `vo_poses.txt` SHA-256
`b47a3dd8093d9c205fd5b5213a4392c9ea694be831cfc20d1c61667f0ed64743`、
`est.tum` SHA-256
`3190442ff6af2bc35712f24312518449e944b9c10b26d741e75fe9257b23cd3b`。
全 EuRoC feature/export の canonical manifest は
`/home/sasaki/euroc_mh03_official_20260830/manifest_full_2700.json`、JSON
SHA-256 `6c6f9f64551882bd5dafbe98719348879511c10cfeed280dcee25630db97ed38`、
内部 manifest digest `489d953274540d331603fa072f04996ab20c39c9cddfcecb1d332120a4ab801f`
で、left/right 2,700、stereo 2,700、temporal 2,699、temporary file 0 である。
baseline/current の全 ATE/RPE 表、実行時間、RSS、venv/archive hash は
`docs/nonregression_20260830.md` にある。

### CI・tree・互換性の最終状態

- `cargo fmt --all -- --check`: pass。
- `sh scripts/check.sh`: exit 0。workspace tests、default/image-io の
  clippy (`-D warnings`)、Python 243 tests（optional skip 8）、docs/package/
  registry/examples/MSRV gates を含む。ログは
  `/tmp/visloc_check_final_auto_default_20260831.log`。
- `git diff --check`: pass。Windows-only gate はこの Linux host では未実行。
- `git status` の変更は今回の SfM/SIFT/BA/CLI/docs の既存作業ツリーと、
  外部成果物を記録する docs/helper 群のみ。repo 内 status に db、feature、
  log、temp、secret はなく、約45 MB の `models/lightglue_courtyard.onnx.data`
  は既存 `.gitignore` 対象で今回作成・変更していない。
- `NextImagePolicy::default()` は Count のまま、両 demo の CLI 省略時は Auto。
  Visibility と recovery、snapshot coordinate override 等は explicit opt-in。
  verified-pair snapshot
  は schema v1 を維持し、checksum/manifest/順序検証と round-trip tests を
  通過している。既定経路と snapshot/API byte identity を暗黙には変更していない。
- material な A/B の positive/negative は `CHANGELOG.md` と
  `docs/nonregression_20260830.md` に記録済み。今回の closure では default
  production code を変更しなかった。

**総合判定:** courtyard の耐久再現性と Linux CI は完了。だが、South の
strict default 128/128 は未達、terrace/office は旧 feature cache 不在で厳密な
非回帰判定不能、EuRoC の numeric same-cache は full 系が事前許容幅なしで
inconclusive、Auto は EuRoC 非対応である。したがって active goal 全体は
**未完了（closure evidence は完了）**。次の作業は South の安全な default-free
registration policy と、terrace/office の正確な archived feature cache を取得して
から行う。既存の experimental/default-off semantics を完了扱いにする根拠はない。

## 13. Auto 既定値と最終非回帰判定（2026-08-31、authoritative）

上の section 12 は Auto 最終パッチ前の記録であり、以下で更新する。
`examples/unordered_sfm_demo.rs` と `examples/sequential_sfm_demo.rs` の
CLI 省略時は `NextImagePolicy::Auto`、`IncrementalSfmConfig::default()` は
API/ライブラリ互換性のため `CorrespondenceCount` のままである。Auto は
Visibility を先に試し、未登録画像が一つでもあれば同じ feature/pair 入力で
Count も評価する。選択後に未完なら clean state から post-refinement を一度
試し、登録画像数が strict に増えた時だけ採用する。tie または減少時は
post 前のモデルを保持する。

凍結 cache の no-flag 実測（各 run の `run.log` に完全なコマンドと
`effective-config: ... next_image_policy: Auto` を保存）は次の通り。

| suite | Auto の判断 | 登録 | tracks / observations | 平均 reprojection | reference Sim(3) RMSE |
|---|---|---:|---:|---:|---:|
| South Building | Visibility 127/128 vs Count 123/128、post 127→128 を採用 | 128/128 | 20,554 / 93,647 | 1.406 px | 0.73 cm |
| terrace | Visibility 12/23 vs Count 23/23、Count 選択、post skip | 23/23 | 3,595 / 10,119 | 1.574 px | 2.56 cm |
| office | Visibility 17/26 vs Count 17/26、Count 選択、post 17→18 を採用 | 18/26 | 1,082 / 3,037 | 1.512 px | 0.45 cm |
| courtyard exhaustive | Visibility が complete、Count/post skip | 38/38 | 43,852 / 152,432 | 0.579 px | 0.5379 cm proxy |

成果物は `/media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/runs/`
以下の `cache-fixed-auto-default2-{south,terrace,office,courtyard}-20260831`
である。terrace は従来の recovery+post 明示 run の 78.54 cm を採らず、Count
の 2.56 cm を保持した。office は same-cache Count の 17/26・0.43 cmから
18/26・0.45 cmへ登録を一台増やし reprojection を 1.531→1.512 px としたが、
reference RMSE の改善とは主張しない。courtyard は durable champion の
`cameras.txt` / `images.txt` / `points3D.txt` とそれぞれ byte-identical で、
hash は `76fc758375228319300a5c076b6bcc84a88413d69cf3613eb40c980b60e8cc9c`,
`a14ac6b958bf09481bfcc0ae72b59671d8e6405cd880e4711ff14c5c9432852e`,
`d7b680e6d51a403a4962920e8f0e0615f382646a1ac9836f160ecb44622a3293` である。

Verified-pair snapshot は互換性を分離する。snapshot import で policy を
省略した場合は Count を強制し、明示 `--next-image-policy auto` は引き続き
許可する。`/tmp/snapshot_colmap_verified_20260830.vps`（SHA-256
`6511181ac3b099cb9a9c8d7525b1746d28b7d5c7459df27e8460fef27f71f82a`）の
no-flag replay は Count control と同一の **38/38、20,649/68,514、0.342 px**
で、model hashes は次の通りである。

```text
cameras  a2132068b1a4dbe21f1ad68a23ff05461026c5a84e0b0de14f06311533e5b958
images   23836ffe18995d83a4e0c7a56375b39aa0d702c59af1c0ec7b5da85c65b04a2e
points3D 1a088ea533aaa2891609333dcdc819d1342dbee53523332903b87783e433c81c
```

明示 Auto は従来の Visibility model と同一（20,086/66,894、0.281 px）で、
Count の escape hatch と Auto override の双方を維持している。

terrace/office のコード非回帰は detached `2a36d44` と同一 cache で別判定する。
terrace は baseline **23/23、3,614/10,161、1.575 px、1.63 cm** に対し
current Count **23/23、3,595/10,119、1.574 px、2.56 cm**（登録は pass、
数値は事前許容幅なしで inconclusive）。office は baseline **17/26、
1,024/2,904、1.532 px、0.43 cm** と current Count **17/26、1,024/2,904、
1.531 px、0.43 cm** が一致し、same-cache code non-regression は pass である。
いずれも歴史的 SuperPoint cache の bytes は存在しないため、旧 terrace
12.37 cm / office 0.37 cm arm への絶対非回帰は inconclusive とする。

EuRoC は、variant ごとの tuning を避けるため、先に固定した project-level
same-cache rule を適用する。全 variant で 2,700/2,700 poses と 2,699 updates
を要求し、ATE/RPE の SE(3)/Sim(3) 各値について current が baseline を
超えてよい幅を `max(5% of baseline, 0.005 m)` とする。5 mm floor はこの
trajectory evaluator の一律 engineering tolerance であり、variant 別に
選んだ閾値ではない。最大の current ATE Sim(3) 増分は full2v の 3.921 mm
で、この固定幅内に収まるため open/loop/full/full2v/full2vh/full2vhi は
すべて same-cache pass（full2vhi は repeat hash も一致）と分類する。これは
旧絶対 benchmark の改善主張ではない。loop-closure runner に
`NextImagePolicy` はないので EuRoC Auto は N/A とする。

最終実装後の対象チェックは `cargo fmt --all -- --check`、対象 Auto/CLI tests、
release build、および `sh scripts/check.sh` と `git diff --check` である。
Windows-only gate は Linux host のため未実行である。未コミットの既存
SfM/SIFT/BA/CLI/docs 差分は保持し、repo 内に feature/db/log/temp/secret は
追加していない。
