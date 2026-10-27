# 写真 → SfM / MVS / 3DGS: ローカル PC への引き継ぎ

2026-10-27 時点。ここまでの作業はクラウドで進めてきた。内訳はクラウドのコンテナ（4 スレッド、メモリ 15 GB、GPU なし）と、計測用に借りた GPU ポッド（L4 など）。この先は GPU のあるローカル PC で続ける。LaMAria（VIO と BA）の引き継ぎは別の文書 [lamaria_gpu_handover.md](lamaria_gpu_handover.md) にある。

## 現状

- **リリース**:
  - main のバージョンは 0.3.1。CHANGELOG の [0.3.1](../CHANGELOG.md) 節は切ってある。
  - GitHub のタグと Release は `v0.3.0` まで。`v0.3.1` は打っていない。
  - `v*` タグを push すると、`.github/workflows/release.yml` が Release を作り、Linux・Windows・macOS のバイナリを添付する。
    - 中身は `gsplat_photos`、`visloc_mvs`、`colmap_incremental_mapper`、`visloc_web`。
    - crates.io と PyPI には何も公開していない。
- **成績**:

  | 項目 | 結果 | 詳細 |
  |---|---|---|
  | SfM | ETH3D DSLR 13 シーンの AUC@1°/3°/5° で COLMAP 4.2.1 を上回る。COLMAP の照合で 53.69 / 79.94 / 86.90 | [eth3d_auc_benchmark.md](eth3d_auc_benchmark.md) |
  | MVS（正解姿勢） | ETH3D high-res 13 シーンの F1（2 cm）で 83.7%。ACMMP 83.42%、APD-MVS 86.84% | [mvs.md](mvs.md) |
  | 写真から MVS まで通し | 81.5%（カメラの向きで正解に合わせた場合）。カメラ中心だけで合わせると 76.3%、COLMAP 4.2.1 CUDA は 57.4%（両方が完走した 10 シーン）| [mvs.md](mvs.md) |
  | 大規模 | Hierarchical 3DGS の SmallCity 5,822 枚を `gsplat_photos` 1 コマンドで処理（Colab A100 で 2.5 時間）| README |

- **このあと何をするか**: [next_development_plan.md](next_development_plan.md) を見る。
  - 表の行に、試して効かなかったことも含めて記録してある。
  - 「On hold: GPU work」節に、保留中の GPU 作業がまとめてある。

## 環境構築

- **Rust**: `rust-toolchain.toml` で 1.94.0 に固定してある。rustup を入れればリポジトリ内で自動的に使われる。MSRV は 1.88 で、CI が確かめる。
- **GPU**: すべて wgpu 経由。CUDA は要らない。
  - Linux は Vulkan のドライバがあればよい。NVIDIA なら公式ドライバ。
  - GPU の経路は Cargo の feature で有効にする。
- **ビルド**: `gsplat_photos` は `visloc-gsplat-train` パッケージの example で、feature が要る。ルートで `cargo build --example gsplat_photos` とすると「no example target named …」で失敗する。

  ```bash
  cargo build --release -p visloc-gsplat-train --features "gpu euroc" --example gsplat_photos
  cargo build --release -p visloc-mvs --features gpu --example mvs
  cargo build --release --example colmap_incremental_mapper --example visloc_web
  # 出力は target/release/examples/
  ```

- **Python**: 評価スクリプトに `numpy`、`Pillow`、`py7zr` を使う。ETH3D AUC のベンチマークには `pycolmap>=4.2` も要る。
- **品質ゲート**: `scripts/check.sh` が CI と同じことをすべて走らせる（fmt、clippy、テスト、ドキュメントなど）。普段は変えたパッケージの `cargo test -p …` と `cargo fmt --all --check` で足りる。
- **メモリ**: 15 GB の機械では、SmallCity の 1,499 枚を CPU で通すと、内部パラメータ補正の途中で OOM killer に止められた（RSS 約 11〜14 GB）。500 枚なら問題ない。

## よく使うもの

- **写真から SfM まで**（`--steps 0` で学習なし）:

  ```bash
  target/release/examples/gsplat_photos --images photos/ --out run/ --steps 0 --no-mesh
  # GPU がないとき: --cpu --cpu-sift。全オプションは --help
  ```

  - 出力は `run/sparse/0/`（COLMAP テキスト）と `run/image_names.txt`。
  - `image_names.txt` は作業コピー `frame_NNNNN.png` と元の写真の対応表。
  - マッパーへの入力は `run/port/` に書き出される。`colmap_incremental_mapper` で再生でき、`port/mapper_options.txt` の設定も引き継がれる。
- **計測用の環境変数**:

  | 変数 | 出すもの |
  |---|---|
  | `VISLOC_SFM_TIMING=1` | SfM の各段の時間（`sfm-timing-ba: … iterations=20 accepted=4` など）|
  | `VISLOC_JOINT_BA_TIMING=1` | 内部パラメータ補正 BA の反復ごとの build / solve / cost の時間 |
  | `VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_BA=1 VISLOC_SFM_DEBUG_BA_STEPS=1` | LM の各ステップの採否、コスト、λ |

- **プロファイル**: `perf record --call-graph dwarf -p $(pgrep -x gsplat_photos)`。
  - クラウドの機械では、`perf` のラッパーがカーネルの版に合わず、`/usr/lib/linux-tools-*/perf` を直接使った。
  - `pgrep -f` だとラッパースクリプトや自分のシェルにも一致するので、`-x` を使う。

## 評価の再現

| 評価 | コマンド | 備考 |
|---|---|---|
| ETH3D AUC（照合とマッパー、焦点は正解で固定） | `scripts/benchmark_eth3d_auc.sh WORK [scene ...]` | 結果は [eth3d_auc_benchmark.md](eth3d_auc_benchmark.md) |
| ETH3D の姿勢誤差（`gsplat_photos` 一式、焦点 5% ずれから補正） | `scripts/eth3d_photos_pose_check.sh WORK TAG [scene ...]` | 2 つのビルドや設定を TAG を変えて比べる。GPU なしなら `PHOTOS_ARGS="--cpu --cpu-sift"` |
| SmallCity の参照モデルとの比較 | `scripts/score_smallcity_reference.py` | 下を参照 |
| MVS の F1 | [mvs.md](mvs.md) の手順（ETH3D 公式評価器） | GPU で 13 シーン約 1 時間（L4）|

- **姿勢誤差の比較手順**:
  1. `cargo build` してから `scripts/eth3d_photos_pose_check.sh ~/eth3d new` を走らせる。
  2. 変更前のバイナリでも、`PHOTOS=old/gsplat_photos scripts/eth3d_photos_pose_check.sh ~/eth3d old` を走らせる。
  3. `~/eth3d/results.txt` に、シーンごとに 1 行ずつ結果が並ぶ。
  - 13 シーンの取得には合わせて数 GB の通信がかかる。
  - 2026-10-25 に 13 シーンを測った結果は、CHANGELOG の 0.3.1 節にある（中央値の平均 0.79 cm）。

- **SmallCity の手順**: 正解はなく、リリースの COLMAP モデルを参照にする。

  ```bash
  python3 scripts/score_smallcity_reference.py fetch ref_images.txt              # 参照の姿勢（約 300 MB の通信）
  python3 scripts/score_smallcity_reference.py fetch-images sub500 pass1 0 500   # pass1 の 0〜499 枚目
  target/release/examples/gsplat_photos --images sub500 --out run500 --steps 0 --no-mesh \
      --focal 487 --retrieval 15 --max-keypoints 2048
  python3 scripts/score_smallcity_reference.py score --names run500/image_names.txt \
      --trimmed --profile ref_images.txt run500/sparse/0/images.txt
  ```

  - `--trimmed` は、大きく外れたカメラを除いて位置合わせし直す。1 m 以上外れたカメラの一覧も出す。
  - `--profile` は 100 枚ごとの縮尺と局所誤差を出す。経路の周期で縮尺が揺れていれば、局所的には正しいのにループ全体が歪んでいる。
  - 2026-10-25 時点の 3 サブセット（CPU）の結果（中央値 / p90 / 最大）:

    | サブセット | 結果 |
    |---|---|
    | 0〜499 | 12.1 / 17.3 / 23 cm |
    | 500〜999 | 24.9 / 30.8 / 69 cm |
    | 1000〜1499 | 7.9 / 15.2 / 21 cm |

    500〜999 が悪い理由は [next_development_plan.md](next_development_plan.md) の「SmallCity 500–999」の行にある。

## 次の候補

ローカルの GPU で進められるもの:

1. **ETH3D MVS の差を詰める（GPU）**:
   - APD-MVS との差 3.2 点の大半は、テクスチャの弱い 4 シーン（office、playground、facade、kicker）。
   - r55（strong median の 3 通りを office・playground・terrace で比べる）は準備だけしてあり、まだ走らせていない。
   - [next_development_plan.md](next_development_plan.md) の「On hold: GPU work」と [apd_mvs_code_notes.md](apd_mvs_code_notes.md) を読んでから始める。
2. **フル解像度での SfM 回転精度（GPU）**:
   - playground を 800 px で通したとき、MVS で落ちる点の一部は、SfM の回転誤差（1 枚あたり 0.02〜0.03°、3,200 px で約 1 px）から来ていた。
   - 原因は 3,200 px での SIFT の観測と見ているが、フル解像度の SfM が 15 GB のメモリで回らず、確かめられていない。
3. **配布バイナリの実地確認（CPU）**:
   - Release から落としたバイナリで、写真から SfM・MVS まで通るかを見る。
   - 配布ビルドでしか起きない問題（足りないファイル、GPU のない環境での動作など）を拾う。
4. **SfM の精度（CPU、優先度は低い）**:
   - SmallCity 500〜999 の歪みの残り半分（±1.3%）の原因は分かっていない。
   - EXIF の焦点距離を BA の事前分布に使う案があるが、正しい焦点距離で固定しても 24.9 → 14.5 cm なので、伸びしろは限られる。

## 注意点

- **既定の出力を変えない**: 既定の出力はビット単位で変えないのが方針。速度改善は、変更前後の出力の md5 を比べてから入れてきた（例: playground の `sparse/0/*.txt`）。
- **否定的な結果も残す**: 試して効かなかったことも、CHANGELOG か `docs/` に数値付きで残す。同じことを二度試さないため。
- **Windows の PowerShell**:
  - 5.1 は `&&` が使えない。`;` で区切るか 1 行ずつ打つ。
  - コマンド例の `\` での行継続も使えない。PowerShell では行末をバッククォートにするか、1 行にまとめる。
- **クラウドのセッション（Claude Code on the web）でできないこと**:
  - 作業ブランチ以外への push。タグも push できない（HTTP 403）。
  - GitHub Actions の手動実行。
  - タグ付けと release ワークフローの手動実行は、ローカルから行う。
