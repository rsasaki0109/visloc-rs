# LaMAria: GPU のローカル PC への引き継ぎ

2026-11-02 時点（成績は 2026-11-13 まで。提出 9 は取り下げ）。**2026-11-13 に、リーダーボードの作業をいったん区切った**（[区切り](#区切り2026-11-13)）。2026-10-27 まではクラウド（GitHub Actions、GPU なし）で進め、そのあと 8 GB の GPU を積んだローカル PC（Windows の WSL2 Ubuntu、データは外付け SSD の NTFS）で続けた。

## 区切り（2026-11-13）

- **到達点**: 提出 10 で平均 57.8、6 位。手法の説明も提出 10 の内容にしてある。
- **なぜ低いか**: 壊れるのではなく、ゆっくりずれる。Low light では 5 m 以内のポーズが 99.5 % で FlexVIO と同じだが、1 m 以内の基準点は 49.9 % で、上位は 91〜93 %。
  - チェーンは、出発点の VIO が持ち込んだ向きのずれを引き継ぐ（[The solution keeps its starting point's drift](lamaria_batch_ba.md#the-solution-keeps-its-starting-points-drift)）。
  - 残るずれの大半は、片道のルートで決まらないジャイロの鉛直バイアス（[Where the second sequences' heading drift comes from](lamaria_batch_ba.md#where-the-second-sequences-heading-drift-comes-from)）。機器ごとのジャイロ補正は VIO だけなら効くが、チェーンでは 5 系列の平均が 72.6 → 70.1 に下がり、正解なしで補正を選ぶ方法がない（[Where the device's correction gains](lamaria_batch_ba.md#where-the-devices-correction-gains)）。
  - 学習系列の平均は約 68.6、テストは 57.8。設定は各トラックの 1 本目で選んでいて、テストは 2 本目と同じように振る舞う。
- **上位との差**: すぐ上の RedComet まで 8.9、FlexVIO まで 15.4、SLAM Dunk まで 17.7。提出 1 から 10 までの伸びは 51.5 → 57.8 で、うち 3.4 は Moving の GPS。今の構成でこの差を埋める、中身の決まった案はない。
- **FlexVIO**: 2026-11-10 から 11-13 のあいだの再アップロードで、平均が 58.1 → 73.2 になった。Short から Long への下がり幅は 16.4 → 9.2。匿名で、説明もコードもない。何を変えたのかは分からない。
- **再開の条件**: 上位の手法（SLAM Dunk、FlexVIO、GroundedSLAM、RedComet）のどれかが論文かコードを公開したとき。ずれを何で抑えているかが分かれば、試す中身を決められる。

## 現状

提出 10 回目（サイトの日付で 2026-11-12）の成績。提出 10 は、提出 7 の sequence_4_2 だけを作り直したもの（下記）。以下は提出 7 回目（2026-11-06）からの経緯。手法名は visloc-rs
（<https://lamaria.ethz.ch/method_info.php?id=48>）で、平均では 6 位（参考枠の Aria's SLAM を含む。リーダーボード自体は Short の点で並ぶ）。提出 9 は録画どうしをまとめて解いたもので、取り下げた（下記）。2026-11-10 に提出 7 をアップロードし直し、63 系列とも最初のときと同じ点に戻った。「WIP」になっていた手法の説明も、提出 7 の文面に戻した。ほかの手法の点は 2026-11-13 時点のもの。FlexVIO だけが 2026-11-10 から変わった（それまでは 73.1、62.9、56.7、69.7、28.1 で平均 58.1）。

| | Short | Medium | Long | Low light | Moving | 平均 |
|---|---:|---:|---:|---:|---:|---:|
| visloc-rs（提出 10） | 69.7 | 56.8 | 54.1 | 61.1 | 47.3 | 57.8 |
| visloc-rs（提出 7） | 69.7 | 56.8 | 54.1 | 60.6 | 47.3 | 57.7 |
| visloc-rs（提出 6） | 69.7 | 56.8 | 54.1 | 60.6 | 30.6 | 54.3 |
| visloc-rs（提出 9、取り下げ） | 87.8 | 81.6 | 87.5 | 82.7 | 66.7 | 81.3 |
| Aria's SLAM（参考枠） | 90.7 | 78.5 | 70.9 | 84.2 | 55.0 | 75.9 |
| SLAM Dunk | 89.6 | 78.9 | 71.1 | 85.4 | 52.3 | 75.5 |
| FlexVIO（2026-11-13 時点） | 88.5 | 81.8 | 79.3 | 82.4 | 34.0 | 73.2 |
| GroundedSLAM（2026-10-29 更新） | 87.0 | 72.7 | 70.8 | 81.9 | 49.8 | 72.4 |
| RedComet | 75.5 | 65.8 | 61.3 | 77.6 | 53.1 | 66.7 |
| FPV-SLAM、両眼（2026-10-28 更新） | 77.3 | 58.8 | 51.0 | 64.6 | 29.6 | 56.3 |

提出 7 は、提出 6 の Moving の 8 系列を GPS でつなぎ直したもの（[lamaria_gps.md](lamaria_gps.md)）。路面電車を見落としていた 5_5 は 3.1 → 63.7、5_6 は 2.4 → 35.2 になった。

提出 10 は、提出 7 の sequence_4_2 を、最初の 1 分を埋めて作り直したもの（[The poses before a late start](lamaria_batch_ba.md#the-poses-before-a-late-start)）。4_2 は 55.9 → 59.7、Low light は 60.6 → 61.1、平均は 57.7 → 57.8 になった。ほかの 62 系列は提出 7 のファイルのまま。

提出 9 は、提出 7 の 60 系列を、録画どうしをまとめて解いた推定に差し替えたもの（[lamaria_multisession.md](lamaria_multisession.md)）。差し替えたのは歩行 52 系列と Moving 8 系列。歩行 53 系列の平均は 61.0 → 85.7、Moving は 47.3 → 66.7 になった。点が下がったのは 1_15 の −0.8（94.9 → 94.1）だけ。提出 8 は作ったが、サイトに届かないあいだに提出 9 に置き換えたので、評価されていない。

**提出 9 は取り下げた（2026-11-08）。**
- 公式のルールは、ほかの録画を使うことについて何も書いていない。
- 上位の手法は、1 系列ずつ解いているように見える。Long の点は、Aria's SLAM が Short より 19.8、SLAM Dunk が 18.5、GroundedSLAM が 16.2 低い。長く歩くほどずれがたまるためで、提出 9 だけが 0.3 しか下がらない。
- 条件の違う点数を並べないことにした。
- これからは、各系列の推定はその録画だけから作る。提出 9 の点数と推定は、何かを選んだり調整したりするのに使わない。

- **手法**: Rust 版 Basalt の VIO のあとに、バッチ VI-BA のチェーンをつなぐ。チェーンの中身は SIFT、VIO で導いた照合、union-find のトラック、ループ補正、録画後のパス、パスで導いた再照合。
  - 機器 1WM093701G1276 の Short の 5 系列だけ、パスで ALIKED のキーポイントを LightGlue で照合する（提出 5 から）。
  - 使うセンサーは SLAM カメラ 2 台と 1 kHz の IMU。各系列で VIO は 1 本だけ走らせる。
  - 提出 7 から、Moving の系列だけ、乗車で軌跡が飛んだ所を raw 録画の GPS でつなぎ直す。
  - 提出 9 では、歩行 52 系列と Moving 8 系列を、録画どうしでまとめてオフラインで解き直した。同じ場所を通るほかの録画の地図に位置を合わせ、ポーズグラフを解く方法。取り下げたので、今は使わない。
  - 詳細:
    - [lamaria_submission.md](lamaria_submission.md)（提出パイプライン）
    - [lamaria_batch_ba.md](lamaria_batch_ba.md)（BA と実験の記録）
    - [lamaria_test_results.md](lamaria_test_results.md)（提出ごとの結果）
    - [lamaria_gps.md](lamaria_gps.md)（GPS の精度、向きの補正の否定結果、乗車のつなぎ直し）
    - [lamaria_multisession.md](lamaria_multisession.md)（録画をまとめて解く多セッションのポーズグラフ。提出 9 で使い、取り下げた）
- **点を落としている所**: 制御点がすべて 1 m 以内に入った系列では、Short で平均 86.0 点取れている（GroundedSLAM の Short は 87.0）。
  差は、系列の途中から溜まる 1〜5 m のずれでほぼ説明できる。
  - 長い一方向のルートで縮尺と向きがゆっくりずれること（Short・Medium・Long の弱い系列）。
  - 路面電車のドア窓越しに街並みが少ししか見えず、乗車区間を見落とすこと（Moving の 5_5、5_6）。VIO では今も見落とすが、提出 7 から GPS でつないでいる。
  - 提出 9 では、同じ道を通るほかの録画がこのずれの大半を消したが、取り下げた。1 系列ずつ解く限り、このずれは残っている。

## GPU の PC でやったこと（2026-10-27〜11-13）

| やったこと | 結果 | 記録 |
|---|---|---|
| 学習特徴点（ALIKED、DISK、SuperPoint）と LightGlue をチェーンに入れる | パスの ALIKED が効くのは機器 1WM093701G1276 の Short（R 系列）だけ。テストの 5 系列で 57.2 → 61.5（提出 5、平均 +0.2） | [Learned features](lamaria_batch_ba.md#learned-features) |
| 場所認識で再訪問を拾う | パスの再訪問として採用（#360） | [Revisits for place recognition](lamaria_batch_ba.md#revisits-for-place-recognition) |
| ALIKED の照合を VIO の帯なしで（エッセンシャル行列、online の結果の帯） | 学習 R の 2 本では online の帯が +5.1。テストの 5 系列では 61.5 → 60.0（提出 6） | [Learned matches without the VIO's band](lamaria_batch_ba.md#learned-matches-without-the-vios-band) |
| 遠くの点を長く追う・見つけ直す | 頭に付けたカメラでは、遠くの点が視野に続けて入るのは中央値 1 秒。20 秒以上外れた点を見つけ直すと 8 割が誤り | [Far points out of view](lamaria_batch_ba.md#far-points-out-of-view) |
| 乗車区間の見落とし（プラットフォームフィルターを緩める、乗車区間を凍らせて再現） | 5_5 の乗車区間は動くようになるが、学習系列で誤検出して点を落とす。凍らせた乗車区間はチェーンで戻らない | [The platform filter where it stays off](lamaria_batch_ba.md#the-platform-filter-where-it-stays-off) |
| 機器ごとのジャイロ補正 | チェーンの向きを系列によらずほぼ同じだけ回す。3_18 と 4_11 で効いたのは、残っていたずれと逆向きに回ったから。系列自身の疑似正解で当てた補正でも 2_11 では下がり、チェーン自身のコストでは選べない。採らない | [Where the device's correction gains](lamaria_batch_ba.md#where-the-devices-correction-gains) |
| 学習 12 系列すべてに種 1〜3 | 基準は sequence_2_12 を除く 11 系列で 68.59 ± 0.13。sequence_2_12 は種だけで 47.5〜57.9 動く | [All twelve training sequences under the seeds](lamaria_batch_ba.md#all-twelve-training-sequences-under-the-seeds) |
| 4〜16 秒離れたキーフレームを RoMa で密に照合し、SIFT のキーポイントに吸着させて足す | sequence_2_11 では吸着した対応の 93〜96 % が正しく、正しい対応が 3〜5 割増える（暗い 4_10 ではほぼ増えない）。それでも 2_11 は 89.25 → 88.96、3_18 は 54.27 → 52.29 で、3_18 は向きのばらつきが増える。録画の 3 倍強の時間がかかる。採らない | [Dense matches between distant keyframes](lamaria_batch_ba.md#dense-matches-between-distant-keyframes) |
| sequence_3_18 に残る誤差の内訳（疑似正解で取り除く診断） | 今は 54.27。向きを除くと 69.35、縮尺を除くと 58.09、両方で 86.92。向きは片道のルートで積分したジャイロ任せ。速く動く区間で縮尺がずれるのは 3_18 だけで、ほかの系列には一貫して出ない | [What sequence_3_18 still loses](lamaria_batch_ba.md#what-sequence_3_18-still-loses) |
| 歩行の録画をまとめて解く（同じ場所を通る録画の地図に ALIKED + LightGlue で位置を合わせ、全録画のキーフレームを 4 自由度のポーズグラフで解く。GPS は組の候補だけ） | テストの歩行 49 系列と一緒に解いた学習の歩行 6 系列で、合計 445.3 → 546.8（3_18 は 54.3 → 91.1、4_10 は 59.8 → 87.9）。設定はこの 6 系列で選んだ。提出 9 で平均 57.7 → 81.3。ほかの手法と条件が違うので取り下げた | [lamaria_multisession.md](lamaria_multisession.md) |
| GPS が少ない歩行を 2 段階でつなぐ（50 m までの GPS で近くを探し、見つかったエッジで置き直してもう一度照合） | 学習の 2 系列の GPS をわざと悪くして確かめ、6 系列の合計が 524.9 → 551.0。テストでは 1_16 が 35.7 → 76.1、1_17 が 53.9 → 90.4、2_10 が 73.0 → 88.0（提出 9、取り下げ）。4_7 は GPS がほかの録画から約 1 km 離れていて、つなげない | [Walks with poor GPS](lamaria_multisession.md#walks-with-poor-gps) |
| Moving の録画をグラフに足す（解いた歩行は動かさない） | テストの 8 系列がすべて上がり、トラックは 47.3 → 66.7（提出 9、取り下げ）。GPS とのずれが増えた 5_5 と 5_7 は提出 7 のまま。学習の 5_11 は、見落とした乗車が最初の良い GPS より前にあり、51.5 → 40.9 に下がる | [The Moving recordings](lamaria_multisession.md#the-moving-recordings) |
| GPS をグラフの最適化に入れる | 密なテストのグラフでは下がる（なしで 546.8、σ 30 m で 544.0、3 m で 465.9）。採らない | [GPS in the solve](lamaria_multisession.md#gps-in-the-solve-a-negative-result) |
| 磁気センサーで向きを抑える（2026-11-10） | 疑似正解で校正を当てた最良の条件でも、向きの誤差は 256 秒ごとに中央値 0.4〜2.9°、最大 8°。チェーンのずれ（256 秒で 0.0〜0.7°）より大きい。街中の磁場の乱れで、軟鉄の補正を足しても減らない。採らない | [The magnetometer as a compass](lamaria_batch_ba.md#the-magnetometer-as-a-compass) |
| raw 録画の魚眼画像をチェーンに通す（2026-11-04） | 同じピンホール画像を魚眼の経路に通すだけで、9 系列の平均が 4.2 下がる。広い視野で取り戻すのは 0.6。視野そのものは公平に試せていない | [The raw fisheye images](lamaria_batch_ba.md#the-raw-fisheye-images) |
| 2 つの IMU のジャイロを平均する（2026-11-04） | 9 系列中 6 系列で下がる（4_10 は 58.0 → 8.6）。2 つのジャイロの差のゆらぎ（0.4〜0.7 mrad/s）が、向きのずれを生む鉛直バイアス（0.15 mrad/s）より大きい。採らない | [Both IMUs' gyroscopes](lamaria_batch_ba.md#both-imus-gyroscopes) |
| VIO の外れ値フィルター（`config.vio_filter_outliers`）を入れる（2026-11-10） | VIO だけなら夜の 4_10 が 34.7 → 57.4 に上がるが、そこからチェーンを通すと 9 系列中 7 系列で下がり、平均が 4.0 下がる（種 1 どうしの比較。4_10 は 58.0 → 43.8）。採らない | [The VIO's outlier filter](lamaria_batch_ba.md#the-vios-outlier-filter) |
| マッチャーを FlexGlue や VisMatch のものに替える | FlexGlue は固定ペアで位置が決まる数が 2〜5 倍になるが、学習 8 系列のグラフでは LightGlue を超えない（504〜505 対 510.1）。RoMa v2 は GPU を共有した状態で、最初のマップの分も 1 時間で終わらなかった。採らない | [Other matchers](lamaria_multisession.md#other-matchers-a-negative-result) |
| 乗車の前後で区間ごとに回転を持たせる（GPS で測る。2026-11-11） | 両側の区間が大きい 5_5 と 5_9 では、回転の差は 1.1° と 6.8° で、乗車のない歩行を同じ時刻で切ったときの差に収まる。差が大きく出る区間は、GPS の点が中心から 7〜21 m にしか広がっておらず、回転が決まらない。採らない | [A rotation per piece: not measurable](lamaria_gps.md#a-rotation-per-piece-not-measurable) |
| 気圧計で高さを抑える（2026-11-11） | 評価は水平の誤差だけを採点する。基準点の高さを測量値に置き換えて合わせ直しても、学習 12 系列の平均は +0.09〜+0.20。気圧計は読んでいない | [What a perfect height would add](lamaria_batch_ba.md#what-a-perfect-height-would-add) |
| VIO を途中から始めた録画の、欠けた最初の区間を埋める（時間を逆にした録画に同じ VIO をかけ、遅く始めた VIO につなぐ。2026-11-11） | 学習の歩行 7 系列で、わざと 60 秒遅く始めて確かめた。6 系列が上がる（中央値 +4.2。3_17 は 43.7 → 76.6）。夜の 4_10 は遅い開始そのもので崩れ、埋めても戻らない（27.1 → 26.3）。逆向きの VIO が発散するか、つなぎ目が 0.5 m 以上ずれるときは埋めない。テストで当てはまるのは 4_2（60 秒）と 1_15（10 秒）。テストのドライバーに入れた。テストの 4_2 は 55.9 → 59.7（提出 10）。逆向きの VIO は路面電車が通る所で 10.9 m ずれ、最初の基準点は 1 m 以内に入らなかった | [The poses before a late start](lamaria_batch_ba.md#the-poses-before-a-late-start) |

## 残っている案

- 今の構成（Basalt の VIO と SIFT のバッチ BA）の調整は、どれも 1 回 ±0.3 点以下。学習系列が少ないので、その大きさでは選び方自体があてにならない（提出 6）。
- **評価の進め方**: 改善案は、IMU に雑音を少し足した種 1〜3 の実行どうしで比べる。対象は学習 12 系列のうち sequence_2_12 を除く 11 系列（基準は種 1〜3 の平均 68.59 ± 0.13）。平均を 0.4 以上上げ、過半の系列を上げれば採る。sequence_2_12 は種だけで 47.5〜57.9 動くので、結果は記録するが判断には使わない。3 種で約 11 時間かかる（[All twelve training sequences under the seeds](lamaria_batch_ba.md#all-twelve-training-sequences-under-the-seeds)）。
  - 同じ入力のままの実行は、チェーンの設定を選ぶのに使った 5 系列で約 1.2 点甘く出る。ほかの 7 系列では甘くならない。
- **見落とした乗車**: 提出 7 から GPS でつないでいる（[lamaria_gps.md](lamaria_gps.md)）。以前ここに挙げていた、条件付きでプラットフォームフィルターを緩める案は、これで要らなくなった。乗車（56〜162 秒）の間に GPS の点がない 5_10 は、つなげない。
- **上位との差（SLAM Dunk と約 18 点）**: 中身の決まった案はない。2026-11-13 に作業を区切った（[区切り](#区切り2026-11-13)）。
  - 残っているずれの大半は、片道のルートで決まらないジャイロの鉛直バイアス。カメラと IMU のどちらからも決まらない（[Where the second sequences' heading drift comes from](lamaria_batch_ba.md#where-the-second-sequences-heading-drift-comes-from)）。
  - 向きを外から押さえる手がかりは、GPS も磁気センサーも、チェーンより粗い。
  - 学習済みの密な前段（DROID-SLAM）は、向きを今より悪くした。
  - 上位の手法は実時間で動く（SLAM Dunk 62 FPS、GroundedSLAM 60 FPS）。手法は公開されていない（2026-11-10 時点）。
- **魚眼の経路の損**: 同じピンホール画像を魚眼の経路に通すだけで、9 系列の平均が 4.2 下がる。理由（再サンプリングか、ピンホール向きに選んだチェーンの設定か）が分かれば、広い視野を公平に試せる。
- **1 つの録画の中で同じ場所を通るときの照合**: すでにほぼ拾えている。再訪問はキーフレームの 0〜23 % で、どれも 4 分以内。チェーンの `loops` は、1_20 の 9 か所中 7 か所、3_18 の 3 か所中 2 か所を確かめていて、再訪問のキーフレームの組の 93〜96 % にあたる（[Revisits for place recognition](lamaria_batch_ba.md#revisits-for-place-recognition)）。

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

- 記録にあるセンサーは使ってよい（2026-11-05 から。それまではカメラ 2 台と IMU だけ）。
  - リーダーボードのセンサー表示は mono、bino、imu しかない。GPS などを使ったら、手法の説明に書く。
  - 今の使い道は、Moving の乗車で軌跡が飛んだ所を GPS でつなぎ直すことだけ（[lamaria_gps.md](lamaria_gps.md)）。
  - テストの GPS は、VRS の先頭の索引と GPS の記録だけを取る（`scripts/lamaria_vrs_fetch.py`）。raw データの取得になるので、持ち主の了承を得てから。
- VIO は 1 系列につき 1 本。複数の VIO の結果を組み合わせない。
- テストの正解は見られない。設定の良し悪しは学習系列で決める。テスト系列を使うのは、正解なしで分かる診断（乗車区間の長さなど）だけにする。
  - テストの点を見て設定を戻すのも、テストで選ぶことになるので避ける（提出 6 はそのまま）。
- PR の作成とマージは、リポジトリの持ち主の了承を得てから。
- 文書・コミット・PR に GPU の機種名は書かない（「8 GB の GPU」と書く）。
