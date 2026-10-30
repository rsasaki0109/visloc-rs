# LaMAria: GPU のローカル PC への引き継ぎ

2026-10-31 時点。2026-10-27 まではクラウド（GitHub Actions、GPU なし）で進め、そのあと 8 GB の GPU を積んだローカル PC（Windows の WSL2 Ubuntu、データは外付け SSD の NTFS）で続けた。

## 現状

提出 6 回目（2026-10-30）の成績。手法名は visloc-rs
（<https://lamaria.ethz.ch/method_info.php?id=48>）で、5 位（参考枠の Aria's SLAM を含む）。

| | Short | Medium | Long | Low light | Moving | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| visloc-rs（提出 6） | 69.7 | 56.8 | 54.1 | 60.6 | 30.6 | 54.3 |
| GroundedSLAM（2026-10-29 更新） | 87.0 | 72.7 | 70.8 | 81.9 | 49.8 | 72.4 |
| RedComet | 75.5 | 65.8 | 61.3 | 77.6 | 53.1 | 66.7 |
| FPV-SLAM、両眼（2026-10-28 更新） | 77.3 | 58.8 | 51.0 | 64.6 | 29.6 | 56.3 |

- **手法**: Rust 版 Basalt の VIO のあとに、バッチ VI-BA のチェーンをつなぐ。チェーンの中身は SIFT、VIO で導いた照合、union-find のトラック、ループ補正、録画後のパス、パスで導いた再照合。
  - 機器 1WM093701G1276 の Short の 5 系列だけ、パスで ALIKED のキーポイントを LightGlue で照合する（提出 5 から）。
  - 使うセンサーは SLAM カメラ 2 台と 1 kHz の IMU だけ。各系列で VIO は 1 本だけ走らせる。
  - 詳細:
    - [lamaria_submission.md](lamaria_submission.md)（提出パイプライン）
    - [lamaria_batch_ba.md](lamaria_batch_ba.md)（BA と実験の記録）
    - [lamaria_test_results.md](lamaria_test_results.md)（提出ごとの結果）
- **点を落としている所**: 制御点がすべて 1 m 以内に入った系列では、Short で平均 86.0 点取れている（GroundedSLAM の Short は 87.0）。
  差は、系列の途中から溜まる 1〜5 m のずれでほぼ説明できる。
  - 長い一方向のルートで縮尺と向きがゆっくりずれること（Short・Medium・Long の弱い系列）。
  - 路面電車のドア窓越しに街並みが少ししか見えず、乗車区間を見落とすこと（Moving の 5_5、5_6）。

## GPU の PC でやったこと（2026-10-27〜31）

| やったこと | 結果 | 記録 |
|---|---|---|
| 学習特徴点（ALIKED、DISK、SuperPoint）と LightGlue をチェーンに入れる | パスの ALIKED が効くのは機器 1WM093701G1276 の Short（R 系列）だけ。テストの 5 系列で 57.2 → 61.5（提出 5、平均 +0.2） | [Learned features](lamaria_batch_ba.md#learned-features) |
| 場所認識で再訪問を拾う | パスの再訪問として採用（#360） | [Revisits for place recognition](lamaria_batch_ba.md#revisits-for-place-recognition) |
| ALIKED の照合を VIO の帯なしで（エッセンシャル行列、online の結果の帯） | 学習 R の 2 本では online の帯が +5.1。テストの 5 系列では 61.5 → 60.0（提出 6） | [Learned matches without the VIO's band](lamaria_batch_ba.md#learned-matches-without-the-vios-band) |
| 遠くの点を長く追う・見つけ直す | 頭に付けたカメラでは、遠くの点が視野に続けて入るのは中央値 1 秒。20 秒以上外れた点を見つけ直すと 8 割が誤り | [Far points out of view](lamaria_batch_ba.md#far-points-out-of-view) |
| 乗車区間の見落とし（プラットフォームフィルターを緩める、乗車区間を凍らせて再現） | 5_5 の乗車区間は動くようになるが、学習系列で誤検出して点を落とす。凍らせた乗車区間はチェーンで戻らない | [The platform filter where it stays off](lamaria_batch_ba.md#the-platform-filter-where-it-stays-off) |

## 残っている案

- 今の構成（Basalt の VIO と SIFT のバッチ BA）の調整は、どれも 1 回 ±0.3 点以下。学習系列が少ないので、その大きさでは選び方自体があてにならない（提出 6）。
- **条件付きのプラットフォームフィルター**: IMU の速度変化と VIO の速度が食い違うときだけ条件を緩める。学習系列は変わらないはずだが、効果は学習で確かめられない。当たっても平均 +0.4〜0.8。
- **新しい VIO / 前段**: GroundedSLAM との差（約 18 点）を詰めるには、ずれそのものを減らす新しい部品が要る。数週間規模で、効く保証はない。

## 外部 SSD のデータ配置

- **ファイルシステム**: ext4 などの Linux のものが無難。
  - BA チェーンは `work/<seq>/ba/rematch/features.npz` をシンボリックリンクで張る。exFAT では動かない。
  - NTFS でも、WSL2 の drvfs で uid=1000 に再マウントすれば動いた（`mount -t drvfs D: /mnt/d -o uid=1000,gid=1000`。root の所有のままだと、チェーンのファイルのコピーが chmod で失敗する）。
  - drvfs からの読み込みは、メモリが断片化したり Windows 側のメモリが足りなくなったりすると、`Cannot allocate memory (os error 12)` で失敗する。root で `echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory` を定期的に回す。長い VIO は、画像を ext4 にコピーしてから走らせると確実。
  - 抽出した画像は 1 系列で数万ファイルになる。
- **容量**:
  - テスト 63 系列は ASL で約 862 GB。1 系列の画像は最大 29 GB。
  - 学習系列（12 本）をすべて残し、作業領域も取るなら 2 TB が目安。
  - テストは、系列ごとに取得して処理し、消す運用（`--keep-data` なし）なら 1 系列分の空きで足りる。

置き場所は環境変数で決める。例:

```bash
export LAMARIA_DATA=/media/$USER/lamaria_ssd/lamaria   # 外部 SSD のマウント先
mkdir -p $LAMARIA_DATA/{work,slam,logs,gt,ariacal,eval}
```

| 置き場所 | 中身 |
|---|---|
| `$LAMARIA_DATA/work/` | 提出スクリプトの `--work-dir`。系列ごとの抽出データ、校正、VIO、BA の作業ファイル（`--keep-data` で残る） |
| `$LAMARIA_DATA/slam/` | `--slam-dir`。`<seq>.txt`（提出形式の推定） |
| `$LAMARIA_DATA/logs/` | `--log-dir`。BA チェーンの各段のログと `ba2_estimate.txt` |
| `$LAMARIA_DATA/gt/` | 学習系列の疑似正解 `<seq>_pgt.txt` と CP `<seq>_sparse.json` |
| `$LAMARIA_DATA/ariacal/` | 評価用の Aria 校正 `<seq>.json` |

データの取得元は <https://cvg-data.inf.ethz.ch/lamaria>。

- **提出スクリプトが自動で取得するもの**:
  - `asl_folder/<split>/<seq>.zip`（画像と IMU）
  - `pinhole_calibrations/<split>/<seq>.json`
  - `raw_data/<split>/<seq>.vrs` の先頭 4 MiB（工場の IMU 校正）
- **評価用に自分で取得するもの**:

```bash
SEQS="sequence_1_19 sequence_1_20 sequence_2_11 sequence_2_12 sequence_3_17 sequence_3_18 \
      sequence_4_10 sequence_4_11 sequence_5_11 sequence_5_12 R_11_5cp R_12_10cp"
B=https://cvg-data.inf.ethz.ch/lamaria
for s in $SEQS; do
  curl -sSfo $LAMARIA_DATA/gt/${s}_pgt.txt     $B/ground_truth/pseudo_dense/$s.txt
  curl -sSfo $LAMARIA_DATA/gt/${s}_sparse.json $B/ground_truth/sparse/$s.json
  curl -sSfo $LAMARIA_DATA/ariacal/$s.json      $B/aria_calibrations/training/$s.json
done
```

（sequence_5_11 と 5_12 には疑似正解がなく、CP だけで評価する。取得に失敗した組はそのまま飛ばしてよい。）

## 環境構築

```bash
git clone https://github.com/rsasaki0109/visloc-rs && cd visloc-rs
cargo build --locked --profile release-rt --example basalt_euroc_vio_demo   # Actions と同じ VIO

python3.12 -m venv ~/venv/lamaria && . ~/venv/lamaria/bin/activate
pip install numpy==2.5.3 scipy==1.18.1 opencv-python-headless==5.0.0.93 gtsam==4.3.0 numba==0.68.0
# GPU 用（CUDA の版に合わせる）: torch、LightGlue（https://github.com/cvg/LightGlue）など

git clone https://github.com/cvg/lamaria ~/lamaria-eval   # 評価コード（ここまでは 238b6ca で評価）
```

`aria2c` があれば、zip の取得は 16 接続で速くなる（サーバーは 1 接続を絞る）。

## 学習系列で 1 本回す（基準の再現）

```bash
python scripts/run_lamaria_test_submission.py --split training --sequences sequence_1_20 \
  --vio-exe target/release-rt/examples/basalt_euroc_vio_demo --ba-chain --keep-data \
  --threads 8 --ba-jobs 8 --ba-threads 4 \
  --work-dir $LAMARIA_DATA/work --slam-dir $LAMARIA_DATA/slam --log-dir $LAMARIA_DATA/logs
```

評価は `cvg/lamaria` の 2 本を使う。CP に対する評価がスコアで、疑似正解に対する評価が姿勢の再現率。

```bash
S=sequence_1_20; E=$LAMARIA_DATA/eval/$S; mkdir -p $E && cp $LAMARIA_DATA/slam/$S.txt $E/
cd ~/lamaria-eval
python evaluate_wrt_control_points.py --estimate $E/$S.txt --cp_json_file $LAMARIA_DATA/gt/${S}_sparse.json \
  --device_calibration_json $LAMARIA_DATA/ariacal/$S.json --output_path $E/cp --corresponding_sensor imu
python evaluate_wrt_pgt.py --estimate $E/$S.txt --gt_estimate $LAMARIA_DATA/gt/${S}_pgt.txt \
  --sparse_eval_result $(find $E/cp -name '*.npy' | head -1)
```

基準の値（GitHub Actions、再照合あり、CP Score）。sequence_1_20 は 86.98 になるはず。

| 系列 | CP Score |
|---|---:|
| sequence_1_19 | 82.25 |
| sequence_1_20 | 86.98 |
| sequence_2_11 | 89.02 |
| sequence_2_12 | 51.97 |
| sequence_3_17 | 80.03 |
| sequence_3_18 | 54.28 |
| sequence_5_11 | 66.77 |
| sequence_5_12 | 6.76 |
| 昼間 8 系列の平均 | 64.76 |
| R_12_10cp | 64.30 |

## テストの提出

- **実行**: これまでどおり GitHub Actions の *LaMAria submission*（GPU なし）で回せる。GPU の前段を使う場合はローカルで回す。
  - `--split test` に、系列名を省略するか `--sequences` で指定する。
  - 系列ごとの推定は `--slam-dir` に溜まり、終わると `--work-dir` の隣に `submission.zip` ができる。
- **アップロード**: 手で <https://lamaria.ethz.ch> に上げる。
  - Submissions の visloc-rs の「Update results」を開き、「Test set sequences」を選ぶ。
  - 上げるのは `slam/<seq>.txt` が 63 個入った zip。GitHub の artifact をダウンロードすると zip が二重になっているので、中身の zip を上げる。
  - 更新理由（1〜2 文、具体的に）を求められる。
  - 評価が終わってから 24 時間は、次の評価ができない。
  - zip は提出の番号を付けた名前で残す（`submission6_upload.zip` など）。提出スクリプトは `--slam-dir` の親に毎回 `submission.zip` を書き直す。提出 5 の zip を取り違えて上げ直したことがある（点は 63 系列とも同じで、評価は毎回同じ結果になる）。
  - 手法の説明は、Submissions で手法名を押すと編集できる（評価の制限とは別）。
- 提出ファイルは git に入れない。

## 守ること

- GPS と磁気センサーは使わない。カメラ 2 台と IMU だけ。
- VIO は 1 系列につき 1 本。複数の VIO の結果を組み合わせない。
- テストの正解は見られない。設定の良し悪しは学習系列で決める。テスト系列を使うのは、正解なしで分かる診断（乗車区間の長さなど）だけにする。
  - テストの点を見て設定を戻すのも、テストで選ぶことになるので避ける（提出 6 はそのまま）。
- PR の作成とマージは、リポジトリの持ち主の了承を得てから。
- 文書・コミット・PR に GPU の機種名は書かない（「8 GB の GPU」と書く）。
