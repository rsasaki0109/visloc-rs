# Codex 向け作業指令：バッチパスのフロントエンドを詰める（提出13の次）

ブランチ `claude/determined-lovelace-pwgdae` を pull してから始めること。push もこのブランチにだけ行う。

## 背景（ここまでで分かっていること）

- 目標は LaMAria リーダーボードで SLAM Dunk との差を縮めること。
- 論文のキャリブレーション項（fisheye、カメラ経路、G、加速度計、時刻ずれ）は、13本の学習用録画では全部まとめて入れても −1.30 だった。差の主因は項ではなく、項を載せている土台にある。
- バッチパスの特徴点を RootSIFT 1500点・比率しきい値 0.78 にしたところ、13本平均が 68.9 → 70.18 になった（+1.29、7本が上昇）。これを既定にした（commit 9af669f）。提出13がこの設定。
  - 900点でも 2回とも約 +1.1 で、再現性はある。
  - キーフレームを 1 s 間隔にすると −6.80。0.5 s のままにする。
- SLAM Dunk の素の状態（どの項も入れない）との差は、R_12_10cp が 5 点（75.8 対 81.1）、sequence_1_20 が 8 点（87.2 対 95.1）残っている。
- トラック2（sequence_2_11、2_12）は、RootSIFT にすると 900点でも 1500点でも下がる。
- 詳細は `docs/lamaria_slam_dunk_comparison.md` の「The paper's front end in the pass」以降を読むこと。

## 守ること（必須）

1. **ライセンス:** SLAM Dunk（github.com/jacobvm04/slam-dunk）は AGPL-3.0。
   - ソースコードを閲覧しない。コピーもしない。
   - 参照してよいのは論文（alphaXiv 2611.slam-dunk）と公開 CLI だけ。性能比較のために実行するのは可（`.github/workflows/slam_dunk_reference.yml`）。
2. **テスト用データ:** テスト分割の推定には、その録画自身のデータだけを使う。
3. **設定の選び方:** 設定は学習用の13本で選ぶ。テストのスコアを見て設定を選ばない。
4. **採用の判定:** 同じ VIO から既定チェーンと1対1で比べ、平均 +0.4 以上かつ過半数が上がったときだけ採用する。
   - CI でのぶれは約 0.3。ただし 2_12 と Moving の録画は数点ぶれる。
   - 判定が際どいときは同じ設定でもう一度走らせて確かめる。
5. **リーダーボード:** アップロードは人が手で行う。自動で提出しない。
6. **コミット:**
   - メッセージにモデル名やモデル ID を書かない。
   - 末尾に次の2行を付ける。
     ```
     Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
     Claude-Session: https://claude.ai/code/session_01ALg7o6en9ePwFQ11c3WCYG
     ```
   - ドキュメントだけのコミットには `[skip ci]` を付ける。
7. **個人情報:** 個人のメールアドレスなどを外部サービスやコミットに出さない。

## 評価の回し方（CI）

`.github/workflows/lamaria_training_eval.yml` の `EXPERIMENT_ARGS` を書き換えて push すると、13本の学習用録画で、実験チェーンと既定チェーンを1対1で比べる評価が走る。所要時間は約6時間。

- ドライバのオプションの中に別のオプションを入れたいときは、空白の代わりにカンマを使う。

  ```
  EXPERIMENT_ARGS: "--ba-experiment-pass-features rootsift --ba-experiment-pass-feature-args=--num-features,900 --ba-experiment-pass-match-args=--ratio,0.75"
  ```

- 2本を同時に比べたいときは、片方を `gh workflow run lamaria_training_eval.yml --ref <branch>` で起動する（ブランチ先頭のファイル内容で走る）。その後でファイルを書き換えて push すると、もう片方が走る。
- 結果は artifact の `training-table`（table.md）に入る。表の最終行に判定（Keep / Do not keep）が出る。
- ジョブが失敗したら、まずログで `IMU rectification failed`、接続拒否、タイムアウトを確認する。ETH サーバーの障害なら、同じ設定で再実行するだけでよい。
- 評価が終わったら、`EXPERIMENT_ARGS` を `""` に戻して push する。この push で既定チェーン単体の評価が走るので、それで確かめてから次に進む。

## やってほしいこと（優先順）

各項目について、13本の表、平均、上がった本数、判定を `docs/lamaria_slam_dunk_comparison.md` の該当節に追記する。採用したものはドライバの既定に反映し、テストを足し、`python -m pytest -q scripts/tests` がすべて通る状態にする。

1. **比率しきい値と点数の組み合わせを詰める（RootSIFT のまま）**
   - 試す組み合わせ：比率 0.75 / 0.8、点数 1200 / 2000。
   - まず 1500点で比率 0.75 と 0.8 の2本を同時に流し、良い方の比率で点数を振る。
   - 既定（1500点・0.78）との1対1比較なので、`EXPERIMENT_ARGS` には実験側の設定をすべて書くこと。
2. **essential matrix による検証を RootSIFT のバッチパスに足す**
   - 論文は essential-matrix RANSAC を使っている。
   - 既定ではバッチパスの照合ガイドが `vio`（VIO のエピポーラ帯）になっている。essential にするには、ドライバの `--ba-pass-match essential` が実験チェーンの RootSIFT パスにも効くようにする。
     - 今の `--ba-pass-match` は、既定チェーンと実験チェーンの両方に入る `chain_options` に渡っている。そのため、実験側だけに効く `--ba-experiment-pass-match` を足す必要がある。
     - `scripts/run_lamaria_test_submission.py` の main_options の組み立てを参照すること。
3. **トラック2が下がる原因を調べる**
   - 2_11 と 2_12 は RootSIFT で −3〜−5.5 になる。
   - 前回の CI artifact（run 45751990291 と 45751972024 の training-sequence_2_11 / 2_12）の `*_summary.json` と各ステップのログを比べる。照合数、ループ（再訪問）の数、外れ値の割合の違いを見る。
   - 原因が分かったときだけ対策を試す。トラックごとに設定を切り替える案は、学習データへの過適合になりやすい。採用するなら、根拠をドキュメントに書くこと。
4. **（余力があれば）RootSIFT パスにオプティカルフローのトラックを足す**
   - 論文は短いフローのトラックを使う。
   - 今の `lamaria_ba_chain.py` は、`flow=True` と自前の特徴点を同時に指定すると止まる（`ValueError`）。flow が SIFT のキーポイントを足す設計のため。
   - RootSIFT パス側の作業ディレクトリ（work/rootsift）で flow を作れるようにしてから評価する。

## 終わったら

- 何を試して、それぞれ採用したか見送ったかを、1つずつ短く報告する。
- 採用したものがあれば、テスト提出用の実行（`lamaria_submission.yml`、split=test、sequences 空）を起動する。
  - 終わったら artifact `estimate-*` を集めて63本そろっているか確かめる。
  - Moving の10本は、`scripts/lamaria_gps_steps.py` で GPS つなぎ直しをしてから zip にする（`docs/lamaria_gps.md`）。
  - zip の中身は `slam/<sequence>.txt` を63本。
  - アップロードは人がするので、zip の置き場所を伝えて止まる。
