# LaMAria: GPU のローカル PC への引き継ぎ

2026-10-27 時点。ここまでの作業はクラウド（GitHub Actions、GPU なし）で進めてきた。この先は GPU のあるローカル PC で、データを外部 SSD に置いて続ける。

## 現状

提出 4 回目（2026-10-27）の成績。手法名は visloc-rs
（<https://lamaria.ethz.ch/method_info.php?id=48>）で、4 位。

| | Short | Medium | Long | Low light | Moving | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| visloc-rs（提出 4） | 68.9 | 56.8 | 54.1 | 60.6 | 30.6 | 54.2 |
| GroundedSLAM | 80.2 | 61.6 | 59.9 | 67.7 | 41.9 | 62.3 |
| RedComet | 75.5 | 65.8 | 61.3 | 77.6 | 53.1 | 66.7 |

- **手法**: Rust 版 Basalt の VIO のあとに、バッチ VI-BA のチェーンをつなぐ。チェーンの中身は SIFT、VIO で導いた照合、union-find のトラック、ループ補正、録画後のパス、パスで導いた再照合。
  - 使うセンサーは SLAM カメラ 2 台と 1 kHz の IMU だけ。各系列で VIO は 1 本だけ走らせる。
  - 詳細:
    - [lamaria_submission.md](lamaria_submission.md)（提出パイプライン）
    - [lamaria_batch_ba.md](lamaria_batch_ba.md)（BA と実験の記録）
    - [lamaria_test_results.md](lamaria_test_results.md)（提出ごとの結果）
- **残っている差**: GroundedSLAM との差 8.1 点の大半は次の 2 つ
  ([lamaria_test_results.md](lamaria_test_results.md#against-groundedslam-sequence-by-sequence))。
  - 長い一方向のルートで縮尺と向きがゆっくりずれること（Short・Long の弱い系列）。
  - 路面電車のドア窓越しに街並みが見えず、乗車区間を見落とすこと（Moving の 5_5、5_6、5_9）。
- **試して効かなかったこと**: カメラと IMU の範囲の手がかりは一通り試して記録してある。
  - 縮尺: 地面の高さ、歩幅、同時刻ステレオ、加速度計の軸別スケール、レバーアーム。
  - 向き: 地球自転、太陽、建物の水平線。
  - Moving: IMU だけで乗車区間をつなぐこと。
  - 記録の場所は [Slow heading drift](lamaria_batch_ba.md#slow-heading-drift) と
    [Finding the rides the VIO missed](lamaria_batch_ba.md#finding-the-rides-the-vio-missed)。

## GPU で次にやること

1. **学習特徴点＋学習照合を BA チェーンに入れる（本命）。**
   - **狙い**: sequence_3_18 では、BA のランドマークが映るはずのキーフレームのうち、SIFT が同じ点を見つけ直せるのは 8.9 % しかない
     （[Where the second sequences' heading drift comes from](lamaria_batch_ba.md#where-the-second-sequences-heading-drift-comes-from)）。
     視点が変わってもトラックが続けば、一方向のルートのずれを長い区間で縛れる。
   - **候補**: SuperPoint、ALIKED、DISK のいずれかと LightGlue の組み合わせ、または XFeat と LighterGlue。
   - **差し込む場所**: `scripts/lamaria_batch_ba.py` の 2 か所。
     - `features`: `features.npz` に `kf_ns`、`offsets`、`xy`、`desc`、`size` を書く。
     - `match`: `matches.npz` に `pairs`、`matches` を書く。VIO で導いたエピポーラの帯は、学習照合のあとの外れ値除去として残してよい。
   - **以前の否定的な結果**: 暗い 4_10 の窓で、XFeat と ALIKED を SIFT と同じ比率検定で照合した。
     キーポイントの位置精度が低く（再投影誤差 0.52〜0.60 px、SIFT は 0.47 px）、結果は悪かった
     （[Low light: features after denoising](lamaria_batch_ba.md#low-light-features-after-denoising)）。
     昼間の系列と学習照合は、まだ試していない。
   - **進め方**: まず sequence_3_18 か R_12_10cp で、同じ点を見つけ直せる割合が 8.9 % から伸びるかを測る。
     伸びたら、下の「基準の値」の学習系列で CP スコアを比べる。
2. **場所認識で再訪問を拾う。**
   - 今の `loops` は、VIO の位置で近くにあるキーフレームを探す方式で、ずれの許容幅（drift gate）がある。
   - NetVLAD、MixVPR、SALAD などの大域記述子なら、ずれても再訪問を見つけられる。
   - ただし一方向のルートには再訪問がないので、効くのは往復のある系列。

## 外部 SSD のデータ配置

- **ファイルシステム**: ext4 などの Linux のものにする。
  - BA チェーンは `work/<seq>/ba/rematch/features.npz` をシンボリックリンクで張る。exFAT や NTFS では動かない。
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
  - 既存の手法 visloc-rs の「Re-upload results」を使う。
  - 上げるのは `slam/<seq>.txt` が 63 個入った zip。GitHub の artifact をダウンロードすると zip が二重になっているので、中身の zip を上げる。
  - 更新理由を求められる。
  - 評価が終わってから 24 時間は、次の評価ができない。
- 提出ファイルは git に入れない。

## 守ること

- GPS と磁気センサーは使わない。カメラ 2 台と IMU だけ。
- VIO は 1 系列につき 1 本。複数の VIO の結果を組み合わせない。
- テストの正解は見られない。設定の良し悪しは学習系列で決める。テスト系列を使うのは、正解なしで分かる診断（乗車区間の長さなど）だけにする。
- PR の作成とマージは、リポジトリの持ち主の了承を得てから。
