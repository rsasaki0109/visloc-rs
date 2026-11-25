# Codex への作業指示: 動的障害物に強いマップ照合ローカライゼーション(屋内・Jetson 前提)

計画本体は [`dynamic_obstacle_localization_plan.md`](dynamic_obstacle_localization_plan.md)。
この文書はそのうち **Phase 0(計測)と Phase 1(検出器なしの施策)** を実装・計測するための指示。
Phase 2(検出器)と Phase 3(ROS 2 / Jetson 実機)は今回の範囲外。

## 0. 最初に読むもの

1. `docs/dynamic_obstacle_localization_plan.md`: 目的、失敗モード F1–F5、リソース予算
2. `PLAN.md`: ガードレールと品質ゲート
3. `docs/basalt_map_localization.md`: 照合 → P3P RANSAC → `OdomCorrection` の流れ、予測投影マッチング、weak fix
4. `docs/tum_rgbd_benchmark.md`: `tum_rgbd_save_map` による RGB-D からの地図作成
5. `docs/openloris_map_localization.md`: cafe の地図と問い合わせ手順
6. コード
   - `examples/basalt_euroc_localize_demo.rs`: 問い合わせ側の本体(`Localizer::localize`、`Attempt`、CSV 出力)
   - `pipelines/localization/src/lib.rs`: `LocalizationPipeline`、`LocalizationResult`
   - `pipelines/localization/src/binary_index.rs`: `match_projected`
   - `pipelines/localization/src/odom_correction.rs`: ゲートと置き換え規則
   - `pipelines/basalt/src/dynamic_objects.rs`: VIO 側の既存フィルタ(作りの参考)

## 1. 守るルール(全タスク共通)

- **新しい挙動はすべてオプトインにする**: フラグ、設定、パラメータで有効にする。デフォルトは無効で、無効時の出力(`trajectory*.tum`、CSV の既存列、地図ファイル)はビット単位で同じにする。変更前後で sha256 を取って確かめる。
- **計測だけの追加はポーズに影響させない**: CSV に列を足すのはよいが、既存の列の順序と値は変えない。新しい列は末尾に足すか、別ファイルにする。
- **地図フォーマットは壊さない**: ランドマークの安定度などは、地図ディレクトリ内の別ファイル(サイドカー、例 `landmark_stability.bin` / `.csv`)に置く。ファイルが無い古い地図でも今までどおり動くこと。
- **リソース**: Phase 1 の各施策は、4 コアの開発機で、位置合わせする画像 1 枚あたり +2 ms 以下に収める。ホットループで画像ごとにヒープ確保を増やさない。中央値で計測し、結果に書く。
- **`unsafe` 禁止、MSRV 1.88**。ML や GPU の依存は足さない(Phase 1 は検出器なし)。
- **品質ゲート**: 各コミットの前に `cargo fmt`、`cargo clippy --workspace --all-targets -- -D warnings`、関係する crate のテストを通す。マージ前には `scripts/check.sh` を通す。
- **ドキュメント**: 挙動を変えたら `docs/` の該当ページと `CHANGELOG.md` の Unreleased を更新する。効かなかった施策も数字つきで記録し、消さない(リポジトリの慣習)。
- **ばらつき**: 1 回の走行で結論を出さない。オフライン再生で、シードや開始フレームを変えて 3 回以上走らせ、中央値と範囲を書く。
- **コミット**: 小さく分ける(計測 → データ準備 → 施策ごと)。メッセージは英語で、何を・なぜ、を書く。ブランチは作業用のものを切る。`main` へ直接 push しない。
- **データの扱い**: データセットはリポジトリに入れない。パスは環境変数か引数で渡す(例 `TUM=/data/tum`、`BONN=/data/bonn`、`OPENLORIS=/data/openloris`)。ダウンロードしたファイルの sha256 を結果ドキュメントに書く。

## 2. タスク A: 計測の追加(Phase 0 の前提)

`basalt_euroc_localize_demo` に、ポーズを変えない診断出力を足す。

1. **インライアの広がり**: 画像を `G × G` のグリッド(既定 4、`--diag-grid N`)に分け、インライアが入っている升の数を出す。
2. **見えるはずのランドマーク**: 予測姿勢がある場合、予測カメラの視野内・奥行き正でカメラを向いているランドマークの数と、そのうちマッチしたもの、インライアになったものの数を出す。見えるはずなのにマッチしなかったものの画像上の分布(グリッドの升ごとの数、または欠けている升の数)も出す。
3. **評価用マスク**(任意、`--eval-mask-dir DIR`): フレームごとの 2 値マスク(PNG、非 0 = 動く物体)を読み、インライアのうちマスク上にあるものの数を出す。**評価専用で、照合には使わない**。マスクが無いフレームは空欄にする。
4. 出力先は既存の CSV の末尾列か、別の `diagnostics.csv`(タイムスタンプで結合できる形)。どちらにしたかを docs に書く。
5. 単体テスト: グリッド計数と可視判定を、合成データで確かめる。

**完了条件**: 既存の EuRoC と OpenLORIS の問い合わせで、フラグなしのときに既存の出力が sha256 で一致すること。診断出力の 1 枚あたりの追加時間を計測して書く。

## 3. タスク B: 屋内データの準備

### B1. TUM RGB-D fr3 動的系列(最優先)

- 問い合わせ: `rgbd_dataset_freiburg3_walking_{xyz,static,halfsphere,rpy}`、`..._sitting_{xyz,halfsphere}`
- 地図の候補(両方試して、問い合わせ側の視野をよく覆う方を選ぶ。選んだ理由を書く):
  - `rgbd_dataset_freiburg3_sitting_static`: 人が座っているので、地図に人が入る。F3 の試験にもなる。
  - `rgbd_dataset_freiburg3_long_office_household`: 同じ建物だが場所がずれる可能性がある。視野の重なりを確かめる。
- 地図は `tum_rgbd_save_map` で、`--trajectory` には地図系列の ground truth を使う(地図の誤差を問題から外すため)。
- `basalt_euroc_localize_demo` が TUM の画像を直接読めない場合は、`scripts/openloris_to_euroc.py` と同じ流儀で `scripts/tum_rgbd_to_euroc.py` を書く。TUM の RGB を cam0 の EuRoC レイアウトに変換し、カメラ内部パラメータは docs/tum_rgbd_benchmark.md の fr3 の値を使う。
- 問い合わせは 2 モードで行う。
  1. **単画像モード**(オドメトリなし): F1 と F2 をそのまま見る。
  2. **オドメトリありモード**(`--odometry`): fr3 には IMU が無い。ground truth に、ドリフト(並進 2%、回転 3%/m 程度)とノイズを足した合成オドメトリを使う。シードで再現できるスクリプトにする(`scripts/` に置く)。予測投影マッチングと `OdomCorrection` の挙動(F5 を含む)を見るため。
- 評価用の人マスク: TUM には無い。タスク A の 3 は、B2 か B4 で使う。

### B2. Bonn RGB-D Dynamic

- 地図: 静的系列(`rgbd_bonn_static`、`rgbd_bonn_static_close_far`)
- 問い合わせ: `crowd*`、`person_tracking*`、`moving_*_box*`、`balloon*`、`synchronous*` など
- B1 と同じ 2 モード。ダウンロード元とライセンスを確かめて docs に書く。

### B3. OpenLORIS-Scene cafe

- 既存の cafe1-1 の地図に、cafe1-2 で問い合わせる。手順は `docs/openloris_map_localization.md` のとおり(`--vio` と `--odometry`)。
- market は 41.5 GB あるので、B1–B3 の結果を見てから判断する。今回は不要。

### B4. Gazebo アクター(任意。B1–B3 が終わって時間があれば)

- `ros2/visloc-ros2/gz/make_world.py`(apartment)と `make_depot_world.py`(Depot)に、Gazebo の animated actor を経路つきで足すオプションを付ける。人数 0 / 2 / 5 / 10。
- セグメンテーションカメラで人のマスクを書き出し、タスク A の 3 に渡す。
- **地図作成時に置いてあった箱や椅子を、問い合わせ時には動かす、または消す** 変種を作る(F3)。
- Nav2 ではなく、記録した画像とオドメトリをオフラインで再生して評価する。

## 4. タスク C: Phase 0 のベースライン表 → **ここで一度止めて報告**

`docs/dynamic_obstacle_localization.md`(結果ページ。新規作成)に、次の表を作る。

- 行: データセット × 系列 × モード(単画像 / オドメトリあり)
- 列:
  - 位置が取れた画像の割合
  - 地図姿勢を持つフレームの割合
  - 最長の空白時間
  - 誤差 RMSE / 中央値 / 最大
  - 10 cm 以上ずれて受け付けられた fix の数
  - 補正のリセット回数
  - インライアの升の数(中央値)
  - 見えるはずのランドマークのマッチ率
  - (マスクがあれば)動く物体上のインライアの割合
  - 1 枚あたりの時間
- 対照として、同じ地図に対する静的な系列(人なし)の問い合わせも入れる。

採点は既存の流儀に合わせる。地図から ground truth への変換は、地図側の軌跡で 1 回だけ当て、問い合わせ側では当て直さない(`basalt_map_localization.md` の「Scoring」)。

失敗した区間を見て、それぞれを F1–F5 のどれかに振り分ける。代表的なフレームを数枚、インライアと見えるはずなのに未マッチの点を描いた画像にして載せる(画像は `docs/assets/` に、合計数 MB 以内)。

**報告して止める**。報告する内容:
- 表
- F1–F5 の内訳
- Phase 1 の施策の中で効きそうな順番
- 目標値の提案(計画の「Targets」を数字にしたもの)

人間が確認してから次に進む。

## 5. タスク D: Phase 1 の施策(確認後に着手。1 施策 = 1 フラグ = 1 コミット群)

各施策で、次の順に作業する。

1. 実装。既定は無効。
2. 合成データでの単体テスト。
3. フラグ無効時に、既存の出力がビット単位で同じであることを確認。
4. タスク C の表に 1 列ずつ足して効果を計測。
5. 1 枚あたりの追加時間を計測。
6. docs と CHANGELOG を更新。

施策ごとの仕様は以下。

### D1. インライアの広がり(F2, F5)
- `--min-inlier-cells K`(グリッドはタスク A のもの)。インライアの升の数が K 未満の fix は weak fix に落とす。weak fix は、既存のとおり複数が一致したときだけ使われる。
- 照合の本体は `pipelines/localization` 側に関数として置き、デモと(後で)ROS 2 ノードから呼べるようにする。

### D2. 予測と合わない対応の塊を除く(F2)
- 予測が新しい間(`projection_fresh_s/m` の範囲内)だけ有効。予測姿勢での再投影誤差が大きい対応のうち、画像上で塊になっているもの(グリッドの近傍でつながった成分、人ほどの大きさ)を RANSAC の前に外す。
- `--drop-disagreeing-clusters`。しきい値は引数にし、既定値の根拠を docs に書く。
- 予測が古いときは何もしない(誤った予測で正しい対応を消さないため)。

### D3. 遮蔽と迷子の区別(F1)
- タスク A の 2 の値から「一部の升だけが大きく欠けていて、残りはマッチしている」状態を遮蔽と判定する。
- 遮蔽中は、weak fix の最小インライア数を下げる(見えている部分だけで判断する)。補正をリセットしない。`--occlusion-aware`。
- 判定の偽陽性(本当に迷子なのに遮蔽とみなす)で誤った fix が増えていないかを、必ず表で示す。

### D4. ランドマークの安定度(F3)
- 地図作成時の初期値: 観測したキーフレームの数と時間幅、`basalt_map_merge` の追記で何セッションに現れたか。
- 問い合わせ時の減衰: 予測上見えるはずで、遮蔽されていない(D3 の判定)のに、マッチしなかった回数で減らす。
- 保存はサイドカーファイル。減衰した結果を書き戻すかどうかは、フラグで選べるようにする(`--update-landmark-stability`)。既定は読むだけ。
- 使い方: RANSAC の採点で重みを付けるか、しきい値未満のランドマークを探索から外す(`--min-landmark-stability S`)。両方を試して比べる。
- 評価: 「地図作成時にあった物が問い合わせ時に無い」系列(B1 の sitting_static の地図、B4 の箱の変種)を、同じ地図で 2 回続けて回す。2 回目に改善するかを見る。

### D5. 補正の置き換え規則を厳しくする(F5)
- `OdomCorrection` の「弾かれた fix が 3 回一致したら置き換える」に、各 fix のインライアの升の数(D1)と安定度(D4)の条件を足す。設定は `OdomCorrectionConfig` に足し、既定は今までどおりにする。

### D の最後
- 効いた施策を組み合わせた行を表に足す。
- EuRoC(MH_02 → MH_01、V1_02 → V1_01)と OpenLORIS office で、フラグ有効時の非回帰を確かめる。ばらつきの範囲内ならよい。
- **報告して止める**。

## 6. 範囲外(今回はやらない)

- 検出器(YOLOX)を使う施策(Phase 2)
- `visloc_map_localize_node` へのパラメータ追加と Nav2 での評価(Phase 3)
- Jetson 実機での計測
- README の見出しにある数字の変更

## 7. 報告のフォーマット(止まるたびに)

- やったこと(コミットの一覧)
- 表(Markdown)と、それを再現するコマンド
- 効いたもの、効かなかったもの、その理由の推測
- 時間とメモリのコスト
- 次にやるべきことの提案と、人間に判断してほしいこと
