# visloc-rs COLMAP parity — Codex 引き継ぎ資料

> 続行（2026-09-08）: PR #90はhead `b00d693` のCI9項目
> （run `34179534033`）通過後、`1090b97`へmerge済み。旧branch整理済み。
> ユーザーの復旧連絡後、Luna Maxのtoolアクセス復旧を確認しました。
> 現在 `feat/m8-weak-angle-fixed-landmarks`。Luna Maxが比較driverに固定点リスト入力を実装中。
> [事前契約](openloris_weak_angle_fixed_landmarks.md)はadaptive＋弱角39点XYZ固定の一候補。
> rootの独立2計算法で39 IDs・949観測を照合し、入力hash付きリストを固定しました。
> [選択証跡](../benchmarks/electro/m8-openloris-weak-angle-membership-v1.json)。
> 全観測は残しますが、不確かなXYZ固定がposeを悪化させる可能性もあります。実solveは未実施。
> 以下のLuna Max利用上限は当時の履歴です。全体goalは引き続き未完了です。

> 観測幾何診断完了（2026-09-08）: Luna Maxの`5300634`をrootレビュー・31関連テスト後、
> 初期/legacy/adaptive/Ceres/Ceres反復の5本で実行。19.53–21.29 s、peak RSS約216 MiB。
> 全16群の点数/観測数を別計算法でも確認。Ceres反復はruntime以外一致、全入力hash不変。
> 角度0.1°未満の39点にadaptive/Ceresの点移動量71.86%/73.50%が集中する一方、
> net cost改善の寄与は6.28%/6.54%。軌跡悪化の原因や除去/固定の有効性は未証明。
> [結果と限界](openloris_observation_geometry_diagnostic.md)、
> [全証跡](../benchmarks/electro/m8-openloris-observation-geometry-v1.json)。
> Luna Maxはコード/テストstage後に利用上限で停止。rootが既存実装をcommit・監査・測定。
> 次の新規実装は、Luna Max復旧またはユーザーによる別モデル作業の指定が必要です。
> 現PRの証跡/CI/mergeはroot担当で続行。全体goalは未完了です。

> 続行（2026-09-08）: PR #89は最終head `24c6814` のCI9項目
> （run `34160530991`）通過後、`7e917df`へsquash merge済み。旧local/remote branch整理済み。
> 現在は `feat/m8-observation-geometry-diagnostic`。
> [事前固定の診断契約](openloris_observation_geometry_diagnostic.md)に従い、Luna Maxで
> 初期モデルの視点角度・track長別のread-only集計を実装します。GT/solver/README変更なし。
> Ceresの軌跡悪化は確定しましたが、原因は未確定。全体goalは引き続き未完了です。

> Ceres参照の採点完了（2026-09-08）: publisher `90672d4` をroot独立6テスト後に
> 実行し、出力2回の3ファイルbyte一致、全identity/校正/anchor/正深度/1成分を確認。
> 全1,000画像・500支持frame・4,716点・130,900観測・361,170 keypointを保持。
> post-only GT（308画像、同条件の再採点一致）はRMSE/p95 **0.029571/0.044943 m**で悪化。
> cost113473.879982/平均再投影0.674902 pxは改善しても軌跡改善にはつながりませんでした。
> [最終証跡](../benchmarks/electro/m8-openloris-ceres-reference-solve-v1.json)。
> 独立Ceresでも同じ傾向を示すため、次は観測目的関数/幾何/観測可能性の条件を調べます。
> 唯一原因の断定やGT選択の減衰sweepはしません。Luna Maxが一次資料の研究レビュー中。
> draft PR #89は最終証跡/CI/merge確認へ。C++/測定binaryは変更せず保持。
> 以下の「publisher/GT未完了」は各時点の履歴です。全体M8–M10 goalは未完了のままです。

> Ceres 1k実行完了（2026-09-08）: Luna Maxの`1f31335`をroot独立認証。
> self-test/微分再検証通過、初期130,900残差dumpは監査済みcheckpointとbyte一致。
> 固定条件の実solveを1回実行し77.45 s / peak RSS305,072 KiBでexit0。
> 20反復上限のNO_CONVERGENCE（usable）であり、収束確認済みとは扱いません。
> 実更新13受理/7拒否（Ceresのsuccessful14は初期iteration0込み）。
> 独立最終state監査でcost113473.8799820374、平均再投影0.6749016424 px、
> 全130,900正深度・全500支持frame・全pose/point ID・固定anchorを確認。
> 最大pose中心移動0.02943 m、XYZ移動231.001 m。GT/モデル出力監査はまだ未実施。
> [solve証跡](../benchmarks/electro/m8-openloris-ceres-reference-solve-v1.json)。
> Luna MaxはPython model publisherとテストを実装中。C++はこれ以上変更せず測定binaryを保持。
> [draft PR #89](https://github.com/rsasaki0109/visloc-rs/pull/89)を作成済み。
> 初期head `a8553c1`はCI9項目通過(run34158171089)。solve head `1f31335`は
> run34158566321のCI9項目も通過。publisher/全監査/最終CIが揃うまでdraft・非mergeです。
> 既定Rust/README性能値は不変。全体goalと10k精度/native E2E/各規模ゲートは未完了。

> Ceres微分の独立検証（2026-09-08）: rootが凍結`1d6424f`の実AutoDiff因子と
> 実ProductManifoldを使い、独立Eigen投影の数値微分・別途導いた解析式と照合。
> ambient/XYZ/tangentのFD最大差は5.22e-7/5.05e-7/3.25e-7（許容1e-5）、
> 解析tangent最大差1.14e-13（許容1e-9）で通過。
> [微分証跡](../benchmarks/electro/m8-openloris-ceres-factor-derivatives-v1.json)。
> これは新SolveInMemory/Problem/保存経路の試験ではありません。Luna Maxがそれらと
> synthetic solveテストを作業中。実1k solveは未実施、final sourceレビュー後に再認証します。

> Ceres初期parity通過（2026-09-08）: `1d6424f`をroot独立buildし、自己テストと
> 同一1k全130,900観測の独立照合が通過。残差座標の最大差は9.84e-11 px、
> 深度最大差1.82e-12 m。全ID/order/xy/XYZ・校正・元fixture hashを確認済み。
> [初期parity証跡](../benchmarks/electro/m8-openloris-ceres-initial-parity-v1.json)。
> 初回は一時出力のENOENT誤判定で保存前に失敗。修正と保存成功/no-clobberテストを追加し、
> 失敗実行を証跡に残しました。元入力・既存出力は変更していません。
> 次はLuna Maxで同じAutoDiff因子を使うsolve/state出力と、微分・manifold・固定anchorの
> synthetic testを実装。rootレビュー前に実1k solveは行いません。
> 現在も `feat/m8-frozen-ceres-reference`、PR未作成。最適化・モデル出力・GT採点は未実施。
> これはCOLMAP native性能比較ではなく、README/既定solverの昇格根拠ではありません。

> 最新の続行（2026-09-08）: PR #88はhead `9ba5c3b` のCI8項目
> （run `34154673044`）通過後、`23cbe88`へmerge済み。旧branch整理済み。
> 現在は `feat/m8-frozen-ceres-reference`。既存のRust oracle fixtureを再利用し、
> Luna Maxで独立Ceres参照を実装中です。COLMAP native pipelineの比較とは区別します。
> 開発依存は専用container `visloc-m8-ceres-reference-v1` 内だけへ追加、host変更なし。
> Ceres 2.2.0と全入力hashを固定し、初期残差parityを証明するまでsolveしません。
> 今回のfixtureは既存1k全観測・rig/anchorを保持し、10kへ拡大しません。
> 前turnは2件のPR統合と実測/図修正まで進捗あり。全体goalは引き続きactive。

> 最新の続行（2026-09-08）: PR #87はhead `054b1ef` のCI8項目
> （run `34153720050`）通過後、`d91fc44`へmerge済み。旧local/remote branchも整理済み。
> 現在は `docs/electro-cdf-annotation`。Luna Maxで既存Electro図のCDF方向説明と
> 重なりをnative generatorから修正します。測定値・曲線・点群は変更しません。
> solver側の次の診断は同一frozen 1k入力のCOLMAP/Ceres BA参照。まず厳密な入力・
> residual・rig/anchor固定の一致を証明し、一致できるまで実行しません。
> 既存COLMAP imageはありますがhost pycolmapもcontainer python3もありません。
> 新規インストールや条件の近似は行っていません。全体goalは未完了です。

> README図修正（2026-09-08）: `695ed47` でCDF注記を正しいhigher/leftへ直し、
> 曲線外へ移動。median/p95列もカード内へ収めました。旧assetは元generatorで再現一致、
> 新PNGは2回＋root独立生成で一致。GIFは旧版とbyte一致の24 frameです。
> 曲線/軌跡領域のRGB画素一致と9入力ファイルのhashをroot確認済み。
> [生成証跡](../benchmarks/electro/readme-cdf-annotation-v1.json)。性能値・README本文は不変。

> 最新実測（2026-09-08）: Luna Maxのadaptive scaled LMを `005dbcc` の認証済み
> binaryで7本測定。`75df6e5` / `7e25dee` はtest-only補強で再buildなし。
> 新方式は1kでPCG20/20成功・LM15/20受理、25.83 / 25.54 s、peak RSS
> 84,860 / 84,852 KiB。mean再投影0.675153 pxへ改善した一方、
> RMSE/p95は0.029190 / 0.044521 mへ悪化し、事前のlegacy非回帰条件に未達。
> **adaptive atlasは実行せず、既定solver/READMEへ昇格しません。**
> 全1,000画像・500支持rig frame・4,716点・130,900観測・361,170 keypointを維持。
> 反復とdebug ON/OFFのモデル/数値trace一致、従来/直接解法/scaled対照もPR #85と一致。
> [7本の証跡](../benchmarks/electro/m8-openloris-adaptive-scaled-lm-v1.json)。
> BA57件（1 ignored）＋CLI19件、関連Python46件が通過。CI/PRはこれから。
> 次は減衰/tolerance sweepではなく、同一観測目的関数とCOLMAP側条件の切り分け。
> READMEの既存CDF図注「lower curve」は方向が逆なので、別のnative生成修正で扱います。
> M8–M10全体goalは未完了です。

> 最新の続行（2026-09-08）: PR #86は最終head `b5ffa66` のCI8項目
> （run `34149951882`）通過後、`d339093`へmerge済み。local/remote旧branchも整理済み。
> 現在は `feat/m8-adaptive-scaled-lm-damping`。Luna Maxで受理後の減衰係数を
> 更新品質rhoに応じて変える明示的なA/Bを実装します。拒否時の増加係数とPCG設定は固定。
> 初期の投影不能観測は新モードだけ明示拒否し、候補のprediction無効と線形失敗を分けます。
> 非昇格のscaled controlに勝つだけでは不十分で、1kのlegacy MF軌跡精度・支持・再投影の
> 非回帰を先に確認し、未達ならatlasへ進めません。全体goalは引き続きactiveです。
> 以下のPR/CI待ちは各時点の履歴で、最新状態はこの追記を優先します。

> 続行（2026-09-08）: PR #85は最終head `11f36b8` のCI8項目
> （run `34145940542`）通過後、`d751f6e`へmerge済み。旧local/remote branchも整理済み。
> 現在は `feat/m8-lm-step-quality-diagnostic`。Luna Maxで既定OFFの更新品質診断を
> `97d9fb0`に実装済みです。予測/実際のcost減少と座標を明示した正規化残差を記録し、solverや
> LMの採否は変更しません。元normalやモデルの追加複製は行いません。
> 前turnは実装・測定・監査・PR統合まで進捗あり。全体goalは精度/E2E/各規模ゲートが未達です。

> 更新品質診断の1k対照（2026-09-08）: 従来/列スケーリングのON/OFF全4本と
> scaled ON再実行は、PR #85の各モデル・LM/PCG/scaling traceと完全一致。
> ON再実行の診断20行も一致しました。scaled全候補のcomponentwise backward errorは
> 約8.62e-11〜1.33e-9ですが、10回棄却（9回は投影不能増加、1回はcost増加）。
> 線形残差の小ささだけでは非線形更新の妥当性を保証できません。
> `9016b8f`でビルドした認証済みbinaryを測定し、`d998fc0`は後続のtest-only補強です。
> 実atlas主成分346.91 s / 1,090,824 KiB、末尾38.62 s / 110,400 KiBで完走。
> 両成分とも2 GiB上限内で、全モデルと既存数値traceはPR #85と完全一致しました。
> 主成分は10回PCG上限到達/10回受理。受理時rhoは0.984477〜1.000015。
> [診断の測定証跡](../benchmarks/electro/m8-openloris-lm-step-quality-v1.json)。
> 次はPCG許容誤差を固定して、更新品質に基づく減衰制御を別の明示的なA/Bにします。
> README/既定solverは変更せず、PR/CI確認へ進みます。

> 最新実測（2026-09-08）: 列スケーリング＋scaled LMは `99d899b` に実装済み。
> 1kは2回ともPCG20/20成功・LM10/20受理・モデル/数値trace一致ですが、
> RMSE/p95が0.028550/0.043791 mへ悪化し非昇格です。
> 10k初回は主成分342.46 s / 1,090,964 KiB、末尾37.69 s / 110,512 KiBで完走。
> 合算RMSE/p95は0.387518/0.635967 m、平均再投影0.562992 px。
> 全identity・支持・校正・正深度を維持しましたが、COLMAP RMSE 0.384307 mに未達。
> 両成分の再実行もモデル・LM/PCG/scaling trace完全一致。主成分343.31 s、末尾38.36 s。
> 旧診断18行dumpのbyte一致、新scaled診断の座標表示とON/OFF結果一致も確認済み。
> [測定証跡](../benchmarks/electro/m8-openloris-column-scaled-lm-v1.json)。
> BA46件＋CLI18件、Python46件、clippy/fmtが通過。PR/CIへ進みます。
> 既定solver/READMEは変更しません。M8–M10の最終条件は未達のままです。

> 最新追記（2026-09-08）: PR #84は最終head `fe4a710` のCI8項目
> （run `34140417320`）通過後、`c49e542`へmerge済み。旧branchも整理済み。
> 続行branchは `feat/m8-column-scaled-lm`。Luna Maxで明示的な列スケーリングと
> scaled座標のLM減衰を実装中です。既定経路は変更せず、同一観測の1k対照と
> 実atlas両成分の反復測定で判断します。実装・測定の契約は
> [有界BA文書](openloris_atlas_bounded_ba.md)末尾を参照。
> 以下の「次の診断」「未実装」は各時点の履歴で、最新状態はこの追記を優先します。

> PR #83は最終head `b99f2e1` のCI8項目（run `34137338043`）と独立監査を通過し、
> `a71f40a`へmerge済み。旧local/remote branchも整理済みです。
> 続行branchは `feat/m8-local-schur-block-diagnostic`。主成分のvariable-pose slot 191
> （元frame 192）の6×6ブロックを明示診断し、算術順序・既定solver・全観測を維持したまま
> 正定値性失敗と消去項の数値規模を確認します。診断だけで精度目標を達成したとは扱いません。
> `bd3291a` の局所診断を実測済み。ON2回の18行dump・モデル・数値traceが一致し、
> 主成分/1kのOFF対照もPR #83と一致しました。
> [証跡](../benchmarks/electro/m8-openloris-local-schur-block-diagnostic-v1.json)。
> 巨大な消去項の支配点は13921。低減衰で局所Schurが非正定値となり、局所SPD化後も
> 全体PCG失敗が残ります。次は明示的な列スケーリング＋scaled座標でのLM減衰を、
> 物理減衰の変更としてA/Bします。観測を除去せず、全normal複製なしで実装する計画です。

> 最新追記（2026-09-08）: `feat/m8-bounded-pcg-residual-restart` の
> `ca67e80` で上限1回・総反復数を延長しないPCG再開を実装し、1kの9本を測定済み。
> [証跡](../benchmarks/electro/m8-openloris-bounded-pcg-restart-v1.json)。
> 相対許容誤差1e-8では線形成功3→5、LM受理3→4だが、RMSE改善は約2.76e-8 mで
> ごく小さい。strict設定では再開7回でも最終モデルは従来と同一。10k/README非昇格。
> `c2eeb70` の明示的な共有rig支持・元ID固定anchor対応で、実atlas両成分も完走。
> [6本の証跡](../benchmarks/electro/m8-openloris-matrix-free-atlas-policy-v1.json)。
> 主成分peak RSS約1.03 GiB、再実行のモデル・数値trace一致。合算RMSE/p95は
> 0.388720/0.638174 m、平均再投影0.579509 px。COLMAPのRMSEにはまだ未達。
> 再開あり/なしの最終モデルは同一。主成分の前処理block正定値性失敗が次の診断対象。
> M8–M10のCOLMAP比較・E2E・各規模ゲートは未完了。

> 現在の引き継ぎ先（2026-09-07）:
> [OpenLORIS M8–M10計画](openloris_m8_m10_plan.md) と
> [rig-atlas診断記録](openloris_rig_atlas_diagnostic.md)。以下は9月1日時点の
> 履歴です。**M8は未完了**。現在の比較入力は
> [boundary修復モデル](../benchmarks/electro/m8-openloris-atlas-boundary-repair-v1.json)
> （PR #72でmainへ統合済み）。独立監査で9,998画像姿勢、9,997支持画像、
> 全4,999支持rigフレーム、4,494＋505の連結成分を確認しています。
> 平均再投影誤差は0.633112 px、軌跡RMSE/p95は0.391075/0.643107 m。
> COLMAPの0.384307/0.638669 mの精度ゲートはまだ未達です。
>
> [同一入力のstrict BA](../benchmarks/electro/m8-openloris-atlas-connected-strict-ba-v1.json)
> はBA直前checkpointの全6ファイル一致を確認しましたが、RMSE/p95が
> 0.391408/0.643460 mに悪化したため非昇格。PR #73
> (`feat/m8-connected-atlas-refinement`) の既定OFFの
> `--joint-rig-ba-filter-observations` は実装と44テストが完了しました。
> 除去前の全観測costと、同じ残存観測集合での前後costを別々に検査し、
> 支持画像・フレームと連結性を維持します。変更する局所候補だけを保持し、
> 削除した観測・trackを明示的に数えます。詳細は
> [有界BAの検証契約](openloris_atlas_bounded_ba.md)を参照してください。
>
> [10k filtering試走](../benchmarks/electro/m8-openloris-atlas-connected-filtered-ba-v1.json)
> はRMSE/p95が0.388993/0.638173 m、観測数重み付き平均再投影0.581744 pxへ改善。
> p95だけがCOLMAP基準を満たし、RMSEはまだ未達です。独立監査で
> 957点・14,440観測の削除数一致、全支持/連結性維持を確認しました。
> 両成分のBA直前checkpointと、filter OFFの主成分strict出力も全6ファイル一致。
> 両成分の再実行も各6ファイル一致、実装コミットのCIは8項目通過。
> PR #73は最終CI8項目通過後、`07a4104`へsquash merge済み。旧branchも整理済みです。
> `feat/m8-preserve-optimized-atlas-points`（PR #74）の点保持モードも同一入力で測定済み。
> 全支持/連結性・再実行一致を満たしますが、RMSE/p95は0.389420/0.639357 mへ
> 悪化したため非昇格です。47 example tests / 21 auditor testsが通過。
> PR #74は最終CI8項目通過後、`f92fa3c`へmergeし旧branchも整理済み。
> 現在は `feat/m8-two-sweep-atlas-refinement` で従来filtered(DLT)方式の
> 固定2回適用を実装・測定しました（`c99c381`、50 example / 23 auditor tests通過）。
> 両成分の1回目checkpointと既存filteredモデル、BA前checkpointと修復モデル、
> 1回適用controlはいずれも全6ファイル一致。2回目も支持・連結性を維持しますが、
> RMSE/p95は0.390165/0.633716 mでRMSEが悪化したため非昇格です。
> 2回目の追加削除140点・2,193観測も独立監査と一致。両成分の再実行は
> 最終出力・1回目checkpointとも全6ファイル一致。PR #75は最終CI8項目通過後
> `0989877`へmergeし旧branchも整理済みです。
> 現在は `perf/m8-reuse-atlas-window-selection`。精度変更を混ぜず、同じwindowの
> landmark選択scanの重複だけを削減します。比較用基準runは172.69 s /
> 525,052 KiB、既存filtered主成分と全6ファイル一致（共有機、局所処理のみ）。
> 選択scan共有は `1eacc08` で実装済み、51 example / 23 auditor tests通過。
> PR #76でserial比較中。基準2回は172.69/172.80 s、変更版初回154.39 sで
> 全6ファイル・ログ全文一致。各3回測定と他モードの非回帰を完了してから判断します。
> 各3回の計測が完了。基準中央値172.69 s、変更版154.39 sで局所処理10.6%短縮。
> 全6実行でモデル・ログ全文一致、RSS中央値525,232/525,036 KiB（実質同じ）。
> strict主成分、点保持両成分、2回適用両成分、filtered末尾成分の非回帰が完了。
> 各6ファイルと、2回適用の両成分1巡目checkpointも凍結モデルに完全一致。
> PR #76は最終CI8項目通過（run 34104286558）後、`56bea96`へmergeし旧branchも整理済み。
> [scan共有の測定記録](../benchmarks/electro/m8-openloris-atlas-selected-scan-reuse-v1.json)。
> 現在は `feat/m8-implicit-schur-prototype`。private/test-onlyのimplicit Schur作用素と
> PCGの数値試作を実装済み。既定solver・公開API・mapper CLIは変更せず、global runは
> まだ行いません。真の線形残差、複数センサーのcross項、明示行列との作用素・step一致を
> 検証しています。空pose入力の拒否は、本番の全pose固定/点のみBAの検証とは別です。
> 次の本番組み込みでは非対応factorを行列組立前に拒否し、固定gaugeの小規模/1k比較を
> 先行します。具体的な検証契約は有界BA文書の末尾にあります。
> `469ee9a`＋`8a1485a`の試作10テストを含むBA 21テスト、clippy/fmt、関連Python
> 46テストが通過。手計算fixtureとfull 15×15連立系も独立照合済みです。
> [数値検証記録](../benchmarks/electro/m8-openloris-implicit-schur-prototype-v1.json)。
> PR #77は最終head CI8項目通過（run 34108930129）後、`0460c9c`へmerge済み。
> 旧branchも整理済み。現在は `feat/m8-matrix-free-ba-entry` で追加の選択式APIを
> 実装中です。既定solver/API互換性を保ち、組立前の適格性検査、有限で正のLM設定、
> 真の残差と失敗理由の診断、更新拒否を共通LM処理に組み込みます。
> 追加APIは `1d7427f` で実装済み。共有LM loopからimplicit Schur/PCGを呼び、
> 反復失敗は状態を変えず記録し、λを増やして上限付きで再試行します。
> 単眼・stereo・非零baseline/回転付き2センサーrigの更新を検証し、既存BA 57件
> （benchmark 1件ignored）、GNC 6件、BA namespace 26件が独立実行で通過。
> [実行APIの記録](../benchmarks/electro/m8-openloris-matrix-free-ba-entry-v1.json)。
> [1k比較入力](../benchmarks/electro/m8-openloris-matrix-free-1k-input-v1.json)は
> 元の3ファイルhash、全支持・連結性・校正、RMSE/p95 0.026703/0.041346 mの
> 再採点一致まで確認済み。driver/A/Bは未実施。名前・全keypoint identityを
> 保持するpost-map比較driverが次段階で、10k global runにはまだ進みません。
> PR #78は最終CI8項目（run 34112509926）通過後、`552f79b`へmergeし旧branchを整理済み。
> 現在は `feat/m8-matrix-free-rig-ba-comparison`。1k用single-arm driverを実装中で、
> direct/matrix-freeを別プロセス・同一入力・全観測保持・frame 0のみ固定で測定します。
> 初期設定は両者20 LM反復/λ1e-4/robustなし/serial、PCGは既定128反復/tol1e-12。
> 元画像名・全keypoint・track identityを維持し、保存ERRORは投影から再計算します。
> この局所処理の時間をmapper全体やnative E2Eの高速化実績とは扱いません。
> 10k精度、高速化、省メモリ、各規模の非回帰とM9/M10の最終条件は維持します。
> PR #79でdriverを `c810be9` に実装し10テスト通過、1k比較は実行済みです。
> 同一条件の再実行はモデル3ファイル・数値trace一致、全観測/支持/校正を維持。
> ただしPCG128は20回中3回しかstepを受理せず、上限512でも最終モデルは同じ。
> directとの数値同等性は未達なので10kには非昇格です。
> [比較記録](../benchmarks/electro/m8-openloris-matrix-free-rig-ba-comparison-v1.json)。
> `parallel=false`とは別にdirect内部Rayonが動くため、`RAYON_NUM_THREADS=1`の
> controlも各2回計測。次は同じ初期normal system/λ1e-4と1e10で作用素・真残差を
> explicit/directと照合し、桁落ち/再帰残差のずれと収束不足を切り分けます。
> ログのλは拒否時だけ増加後の値なので、LM0のsolve λは1e-4です。
> PR #79は最終CI8項目（run 34116769919）通過後、`42bf86f`へmergeし旧branch整理済み。
> 現在は `feat/m8-real-normal-system-oracle`。次の変更はtest-onlyの実normal系診断で、
> 公開API/既定経路の変更や10k実行はまだ行いません。入力depthの独立確認では最小
> 0.000110933 m、最大5,207 mの観測があり、スケール差は診断の手掛かりです。
> これを理由に観測を除去したり、PCG失敗の原因を断定したりはしません。
> 実normal系oracleは `3e50dc5` に実装し、1kの2回実行で全診断数値が一致。
> [実測記録](../benchmarks/electro/m8-openloris-real-normal-system-oracle-v1.json)。
> 低λではdirect自身のimplicit真残差も0.01848で基準7.35e-7を超え、
> PCG128/512も未収束。高λでは22反復で成功しdirectと一致します。
> 低λのtrialは880観測が非正深度なので部分costを改善実績と扱いません。
> 既存direct/MF CLIのモデル3ファイル・数値traceはPR #79と一致。
> 次は同じpreconditionerでexplicit lower-Schur PCGとimplicit PCGを比べ、
> 蓄積誤差と収束性を切り分けます。閾値変更・10k昇格はまだ行いません。
> PR #80は最終CI8項目（run 34122318818）通過後、`2b4ef99`へmergeし旧branch整理済み。
> 現在は `feat/m8-explicit-pcg-isolation` で同前処理のPCG切り分けを進めています。
> 切り分けは `cb14bdc` に実装・1k実測済み、2回の全数値一致を確認。
> [実測記録](../benchmarks/electro/m8-openloris-explicit-pcg-isolation-v1.json)。
> λ1e5でexplicit PCGは316反復の真残差再確認に失敗（8.947e-7 > 7.363e-7）、
> implicit512は9.079e-4。dense参照解も同基準未達で、前処理だけが原因とは断定できません。
> 次は物理座標・dampingを変えず、3x3 landmarkブロックのCholesky構成を別armで比較。
> 現1e-12は診断baselineで、ユーザーの最終目的は精度・速度・省メモリです。
> PR #81は最終CI8項目（run 34125303050）通過後、`06ca2e1`へmergeし旧branch整理済み。
> 現在は `feat/m8-cholesky-landmark-elimination`。同じ物理damping・観測集合で
> landmark eliminationの構成だけを変えるA/Bは `05500bf` に実装し2回の数値一致を確認。
> [証跡](../benchmarks/electro/m8-openloris-cholesky-landmark-elimination-v1.json)。
> general inverseの実測非対称性は0。Choleskyもλ1e5で真残差1.215e-5 > 7.363e-7で失敗し、
> test-onlyのまま非昇格です。次はexampleで停止条件を明示し、production general inverseの
> PCG512 relative1e-8 / absolute1e-12を、既存strict設定・directと1k全最適化で比較します。
> これは別設定armであり旧strict gateの合格扱いにはしません。10k・README昇格は未実施。
> relative設定は `7b6056a` で実装・表示修正し、同一binaryで3 arm各2回の計測完了。
> [結果](../benchmarks/electro/m8-openloris-relative-pcg-tolerance-v1.json): 新設定cost118070.554、
> RMSE/p95 0.026608/0.041100 m、20.19/20.29 s、84,564 KiB。directより精度は未達。
> 全支持・identity・depth・calibration保持、各repeat完全一致、旧direct/strictもPR #79一致。
> 次候補はtrue residual再確認失敗時だけ最大1回restartする別arm（総512反復内）。
> まだ未実装。デフォルト変更・N²状態・反復上限のリセットは行いません。
> PR #82は最終head `cd226dd` のCI全8項目（run 34130693082）通過後、
> `35df35f`へmerge済み。旧local/remote branchも削除しました。
> 続きは `feat/m8-bounded-pcg-residual-restart`。まず既定OFFの回復機構を
> 同じ総反復上限で実装・検証し、1k nonlinearのdirect/strict/relative controlと比較します。
> PR #82 binaryの2 GiB制限付きatlas主成分試走は、既知の観測なしsensor画像8987で
> preflight reject（BA未開始、0.94 s /407032 KiB）。[証跡](../benchmarks/electro/m8-openloris-matrix-free-atlas-pilot-v1.json)。
> 同frame4493のもう片側は10観測を持ち、全rig frameは支持されています。画像削除で迂回しません。
> 10k driverには支持rig内の観測なしsensor画像保持と、tail用の明示anchor4495が必要。
> restart機構とは別の明示モードとして既定1k動作を保ち、実装後に両成分を再試走します。

**更新:** 2026-09-01
**Repo:** `/home/sasaki/workspace/visloc-rs`
**ブランチ:** `perf/electro-m4-persistent-matcher`
**現在地:** M0–M3はmainへ統合済み。M4ではdefault-offのpersistent
matcher、single-GEMM cross-check、single-scan top-2、exact RANSAC枝刈り、
owned bounded-memory mergeを実装した。Electro 1200枚・同一12,000 pairの
3回runで **snapshot SHA完全一致、matching中央値439.01 s、worker peak
RSS中央値1.89 GiB** を確認し、COLMAP 471.37 sを1.074倍上回った。CPU8
feature extractionは721.42 sで全2,400 feature/locusファイルが旧bankと
byte一致。candidate/mapperを含む保守的end-to-endは **1,649.48 s対
5,705.05 s（3.46倍高速）**、品質は1200/1200、RMSE 0.03501 m、mapper
peak 1.39 GiBを維持。次はM4のCI/merge/branch整理後、M5の全ETH3D
scene・単一大規模環境・10k/100k I/O stressへ進む。

今後の実装順、数値ゲート、PR境界、停止条件は
[`docs/electro_performance_roadmap.md`](electro_performance_roadmap.md) を正とする。
データセット、shard、resume、成果物の実行規約は
[`docs/large_scale_unordered_sfm_plan.md`](large_scale_unordered_sfm_plan.md) を参照する。

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
試し、登録画像数が strict に増え、かつ平均 reprojection が有限で既存以下の
時だけ採用する。tie、減少、または誤差悪化時は post 前のモデルを保持する。

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

## 14. Office milestone 2 connectivity audit (2026-08-31)

対象は固定した同一 cache
`/media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/runs/office-authoritative-venv/features`
と、全画像共通の PINHOLE `6221x4146, fx=3437.84, fy=3435.95,
cx=3127.19, cy=2066.98` である。Auto no-flag の完全な未登録画像は
`DSC_0236, DSC_0237, DSC_0238, DSC_0239, DSC_0240, DSC_0241,
DSC_0253, DSC_0254`。初期 Visibility/Count はともに 17/26 で、Count
選択後の clean post pass は `DSC_0223` を 52 correspondences / 13 PnP
inliers で追加し、18/26、1,082 tracks / 3,037 observations、1.512 px
となった。再実行でも同じ結果を得た。

### Verified-graph upper-bound evidence

全探索 `.8` + cross-check の診断は 325 candidate pairs、92 verified pairs、
5,686 accepted inliers。検証済みグラフは 23-image component と孤立した
`DSC_0238`–`DSC_0240` で、後者は raw matches が各 24/40/30 ある一方
accepted inliers はすべて 0。従って `.8` の安全な verifier/track graph
だけでは 26/26 を支える幾何辺が存在しない。`.9` + cross-check は
`DSC_0238` を `DSC_0237` 経由で 15 inliers まで増やし、21/26 の実再構成
になったが、`DSC_0239` と `DSC_0240` はなお孤立（安全上の上限は 24-image
connectivity）。`.95` no-cross-check は全26画像を接続したものの、relaxed
bridges の同一画像 conflict が増え、rescue 実測は 18/26・967 tracks /
2,619 observations・1.492 px に悪化したため採用しない。

### A/B and implementation disposition

detached baseline `2a36d44`、current Count、Visibility、Auto、全探索 `.8`
recovery/post、`.9` control の入力 artifacts は同一である。候補 coverage
だけを増やしても `DSC_0238`–`DSC_0240` の verifier failure は解消せず、
track/PnP 側だけで 26/26 を作るのは GT-free では安全でない。今回の最小
実装は Auto post の clean-candidate 採用条件のみであり、Count/Visibility
および `IncrementalSfmConfig::default()`、PnP 閾値、cross-check semantics
は変更していない。`next_image_auto_post_candidate_is_better` の focused
unit tests は、登録増分・finite error・非悪化を確認する。

実測コマンドと durable run paths は CHANGELOG の Office entry と
`docs/nonregression_20260830.md` に残し、`.8` safe graph での次の候補は
検証失敗理由を改善する descriptor/geometry parity の診断である。false
pose を追加する一般緩和は停止条件とした。

## 15. Office milestone 2 high-density frontend rescue (2026-08-31)

### Genuine official COLMAP CPU control

固定 cache の上限を結論にしないため、公式の full-resolution Office 26画像
（各 `6221x4146`）を新規 DB へ再抽出した。mapping へ渡したのは supplied
 calibration の PINHOLE intrinsics (`fx=3437.84, fy=3435.95,
 cx=3127.19, cy=2066.98`) だけで、extrinsics/laser GT は入力していない。
使用コンテナは `colmap/colmap:latest`、`COLMAP 4.2.0.dev0 (Commit Unknown
 on Unknown with CUDA)` だが、全工程で GPU を無効化した。

再現コマンド（`$IMG` は source image directory、`$OUT` は空の新規 directory）:

```sh
docker run --rm -v "$IMG:/input:ro" -v "$OUT:/output" \
  colmap/colmap:latest colmap feature_extractor \
  --database_path /output/database.db --image_path /input \
  --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 \
  --ImageReader.camera_params 3437.84,3435.95,3127.19,2066.98 \
  --FeatureExtraction.use_gpu 0 --SiftExtraction.max_num_features 8192 \
  --SiftExtraction.first_octave -1 --SiftExtraction.num_octaves 4 \
  --SiftExtraction.octave_resolution 3 --SiftExtraction.peak_threshold 0.00667 \
  --SiftExtraction.edge_threshold 10 --SiftExtraction.max_num_orientations 2
docker run --rm -v "$IMG:/input:ro" -v "$OUT:/output" \
  colmap/colmap:latest colmap exhaustive_matcher \
  --database_path /output/database.db --FeatureMatching.use_gpu 0 \
  --SiftMatching.max_ratio 0.8 --SiftMatching.cross_check 1 \
  --FeatureMatching.guided_matching 1
mkdir -p "$OUT/sparse"
docker run --rm -v "$IMG:/input:ro" -v "$OUT:/output" \
  colmap/colmap:latest colmap mapper \
  --database_path /output/database.db --image_path /input \
  --output_path /output/sparse --Mapper.min_num_matches 15 \
  --Mapper.multiple_models 1 --Mapper.max_num_models 50 \
  --Mapper.num_threads 4 --Mapper.ba_use_gpu 0
```

実測は `167,891` keypoint/descriptor rows、`325/325` raw pairs、
`173` non-empty verified pairs、`83,438` accepted inliers だった。画像ごとの
全対 raw/verified graph support は次の通りで、従来の8欠落画像にも安全な
geometry edge が存在する。

| image | raw matches | verified degree | verified inliers |
|---|---:|---:|---:|
| `DSC_0236` | 11,696 | 11 | 14,686 |
| `DSC_0237` | 11,060 | 7 | 15,392 |
| `DSC_0238` | 7,310 | 5 | 10,590 |
| `DSC_0239` | 5,164 | 5 | 6,775 |
| `DSC_0240` | 6,177 | 5 | 8,993 |
| `DSC_0241` | 10,231 | 11 | 14,736 |
| `DSC_0253` | 1,702 | 16 | 4,194 |
| `DSC_0254` | 1,336 | 15 | 3,014 |

公式 COLMAP mapper は **26/26** を登録し、`13,425` points / `44,979`
observations を出力した。supplied calibration extrinsics との診断 score は
center RMSE **0.50 cm**（median `0.25 cm`, max `1.94 cm`）である。これは
公式 mapper の feasibility control であり、extrinsics/GT を mapping に
注入した結果ではない。

### Same official features through visloc

COLMAP DB の keypoint/uint8 descriptor rows を text feature files へ変換し、
同じ入力を visloc の `NN + ratio .8 + exhaustive + full verifier`、per-image
calibration、`pnp100k/min8 + geometry-guided-conflict-recovery +
post-refinement-registration + final-iterative-refinement + Auto` へ渡した。
DB export は外部の検証済み converter で行った。

```sh
python3 /media/sasaki/aiueo1/visloc-rs/scripts/export_colmap_sift_features.py \
  --database "$OUT/database.db" --out-dir "$FEATURES"
target/release/examples/unordered_sfm_demo \
  --feature-extractor files --features-dir "$FEATURES" \
  --feature-suffix _features.txt --image-suffix .JPG \
  --images-dir "$IMG" --input-colmap-calibration "$CAL" \
  --exhaustive --min-matches 30 --match-ratio 0.8 \
  --verification-mode full --mapper incremental \
  --pnp-max-iterations 100000 --min-pnp-inliers 8 \
  --geometry-guided-conflict-recovery --post-refinement-registration \
  --final-iterative-refinement --next-image-policy auto \
  --out-colmap "$VISLOC_OUT/colmap"
```

ここで `$IMG` は source image directory、`$CAL` は per-image calibration
directory、`$OUT` は official COLMAP run、`$FEATURES` は text export、
`$VISLOC_OUT` は空の新規 output directory である。
この run は **121/325 verified**、`45,285` verifier inlier correspondences、
初期 track build `18,704` tracks を経て、最終 **26/26**、`17,772` tracks /
`47,503` observations、mean reprojection **0.380 px**、calibration-reference
center RMSE **0.34 cm**（median `0.22 cm`, max `0.67 cm`）となった。欠落8画像の
PnP は次の通りで、raw support だけでなく登録可能な 2D--3D support も得ている。

| image | PnP correspondences | PnP inliers | ratio |
|---|---:|---:|---:|
| `DSC_0236` | 54 | 50 | 0.926 |
| `DSC_0237` | 2,014 | 1,962 | 0.974 |
| `DSC_0238` | 1,581 | 1,562 | 0.988 |
| `DSC_0239` | 1,577 | 1,572 | 0.997 |
| `DSC_0240` | 1,667 | 1,656 | 0.993 |
| `DSC_0241` | 47 | 36 | 0.766 |
| `DSC_0253` | 121 | 75 | 0.620 |
| `DSC_0254` | 118 | 78 | 0.661 |

従って、凍結 SuperPoint cache の no-flag `17/26 -> post 18/26`、bounded
LightGlue supplement の `21/26` に対し、公式 SIFT は frontend 起因の
observability gap を埋めて `26/26` にする。ただし official SIFT は現行の
SuperPoint/LightGlue extractor の単なる sparse rematch ではなく別 frontend
である。LightGlue の 31 bounded candidate supplement は `88/325` verified /
`7,641` inliers、最終 `21/26` で、`DSC_0238`--`DSC_0240` は raw support 不足
のままだった。

### Artifact and implementation disposition

全成果物（公式 DB、text features、公式/visloc text models、LightGlue
supplement、logs、SHA-256）は次に隔離保存した。

```text
/media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830/
  runs/office-colmap-sift8192-20260831/
```

主要 hash は DB `f904a0dc3cb22e1e019e4dafcd74d77a9b4a9f18daa19c3ab2eeed710135fbde`、
公式 `sparse_txt/images.txt` `9d98833d40ea3d3bc9a0f257c2a764f19dde336db917d999f4c8658c43796702`、
visloc `colmap/images.txt` `93e7262792016c8a3877d26ded38d2892b458cb42067151bd34114460910e8d4`
である。既存の `--import-matches-supplement-file` と file-feature input は、
candidate を限定した高密度 artifact を deterministic に追加する再利用可能な
merge path であり、今回の LightGlue run でも使用した。一方、公式 SIFT の成功は
別 extractor の置換結果であり、未登録画像だけを自動再抽出して選択する品質保証
（同じ row/provenance を保ったまま全 suite で候補を比較する仕組み）はまだない。
したがって今回は Auto の暗黙 trigger や dataset-specific rescue を追加せず、
公式 SIFT result を frontend feasibility ceiling として記録する。次の実装候補は
この既存 supplement interface に、validated feature-manifest/provenance と bounded
per-image rematch selection を一般化して加えることだが、現データだけで既定動作を
変更する根拠にはしない。

## 16. Milestone 3 bounded candidate scheduling (2026-08-31)

候補生成をmatch/verifier前に限定するM3実装を追加した。`PairSource` の
`vlad-mutual` と、numeric stemのlocal windowとVLAD retrievalを決定的に併合して
budgetで切る `vlad-union` を追加し、`visloc_candidate_manifest_v1`（画像名・順序を
束縛したpair-only manifest、重複/範囲検証、同一ディレクトリのatomic rename）を
`--export-candidate-manifest` / `--candidate-manifest` で再利用できるようにした。
通常のpair source/defaultは変更していない。

候補選択は画像順/filename stemと事前VLADだけを使う。凍結したraw match importは
候補選択後のverification replayにのみ使い、GT、official extrinsics、raw/inlier数を
候補順位付けへ渡していない。`scripts/benchmark_courtyard.py` はconfigの名前付き
schedule、manifest hash/row validation、`--allow-incomplete` negative A/B、候補数・
verified/inlier数・tracks/observations・reprojection・mapping elapsedとJSON gateを
提供する。exhaustive 703-pair controlは引き続き既定である。

### Frozen courtyard A/B

共通条件はofficial high-resolution SIFT features、per-image calibration、ratio .8、
cross-check/full verifier、geometry recovery、post/final incremental mapperである。

| schedule | candidates | verified / inliers | registration | tracks / observations | reprojection | centre RMSE |
|---|---:|---:|---:|---:|---:|---:|
| exhaustive control | 703 | 366 / 261,724 | 38/38 | 43,852 / 152,432 | 0.579 px | 0.5379 cm |
| local stem≤3 + VLAD top-8, budget 200 | 200 | 172 / 199,871 | 38/38 | 45,016 / 148,192 | 0.537 px | 0.66 cm |
| VLAD top-8, non-mutual | 188 | 158 / 187,191 | 38/38 | 44,800 / 135,485 | 0.502 px | 14.17 cm |
| VLAD top-8, mutual | 116 | 112 / 156,140 | 13/38 | 16,646 / 50,278 | 0.478 px | 33.97 cm* |
| sequential stem≤3 | 108 | 100 / 136,553 | 23/38 | 25,315 / 77,273 | 0.499 px | 1.08 cm* |
| vocab-tree base top-4 | 104 | 89 / 129,766 | 38/38 | 44,002 / 121,742 | 0.425 px | exploratory |

`*` は登録済みsubsetだけの診断scoreであり、full gateではない。200-pair unionは
703から503 pair（71.6%）を削減しながら38/38・1cm以下を維持したが、最小RMSEの
exhaustive championではない。manifest replay runのhashは
`2d654b9ed124ec1732d65f4f3829dfadba5e3e978b085844b59c1ee5e5ed32c1` である。候補
生成31.4秒、manifestからのmapping57.3秒（RAYON_NUM_THREADS=1）をJSONへ保存した。
このM3結果をOffice/South/terrace/EuRoCの品質主張へ流用せず、各suiteのfrontend/cache
比較は従来の記録を使う。

## 17. Milestone 4 large-scale unordered-SfM plan (2026-08-31)

大規模化はまだ実行せず、候補データセット、公式一次情報、段階的なresource/runbook、
候補budget・shard・atomic resume・component recovery・評価ゲートを
[`docs/large_scale_unordered_sfm_plan.md`](large_scale_unordered_sfm_plan.md) に整理した。
最初の対象は公式ETH3D low-res many-view `electro`（300画像probe → 1,200画像）で、
10k級へ進む前に候補recall・登録率・再現hash・RSS/disk上限を確認する。
この計画はdownload/large runを含まず、courtyard exhaustive 703-pair / 38/38 /
0.005379 m gateとM3 200-pair A/Bを変更しない。

## 18. Electro roadmap M0/M1 closure (2026-09-01)

M0の6依存PRはmainへ統合済みで、README冒頭にはElectro 1,200画像のGIF・PNG・
COLMAP比較表、courtyardのGIF・比較表、SfM demo commandを配置した。Electroの
主張はmapper 9.61倍高速というphase claimに限定し、matching 4.43倍遅いこと、
RSS 3.17倍、登録7画像差、feature extraction未計測のためend-to-end claimでは
ないことを同じ表で明示している。

M1の300画像single-camera probeはK=64、local stem window 3、cap512、10,634
candidateで凍結した。exhaustive 44,850 candidate / 3,883 verifiedに対して
3,878 verified（99.871% recall）、登録200/300対202/300（loss 0.667 percentage
point）で両gateを満たす。独立2 runのcandidate index、feature manifest、merged
snapshot、cameras/images/points3D hashは完全一致した。feature SIGKILLでは4画像の
sidecarを再利用して900 fileがuninterrupted runと一致し、match SIGKILL resumeも
merged snapshotが一致した。同サイズ1-byte破損が当初verify-onlyを通る不具合を
発見し、complete match indexをverify modeでも再検証するよう修正した。

証跡は benchmarks/electro/electro-300-phase-ledger.json と
benchmarks/electro/electro-300-failure-injection.log。次はroadmap M2で、同じ
12,000 pairのElectro 1,200画像における7未登録画像をfirst-divergence ledgerと
cap32/64/96/128/uncapped controlで切り分ける。M1の300画像absolute RMSE 0.533 mは
quality championではなく、M2で改善する対象である。

## 19. Milestone 5 scale closure (2026-09-02)

M2–M4は完了し、Electro 1,200のCPU8 end-to-endは1,649.48秒、COLMAP
5,705.05秒に対して3.46倍高速、1200/1200、centre RMSE 3.50 cm、mapper
peak 1.39 GiBまで到達した。その上でM5を全tierまで実行した。

- 300画像restart/corruption gateはhash一致・fail-closedで完了。
- synthetic 10k/100k I/Oは約32N pair、100k verify-only 33.6 MiB。
- ETH3D low-res 10 sceneは別々に再構成し、合計9,996/10,008（99.88%）、
  mapper最大3.32 GiB。README冒頭のGIF/PNGは実model camera centreから生成。
- snapshot writer/checksum readerをstream化し、pair境界でraw/audit vectorを検証後
  解放。sand_box 730-shard mergeは5.35→2.03 GiB（−62.0%）、tunnel mapperは
  4.47→3.25 GiB（−27.4%）でmodel 3 file byte-identical。
- connected OpenLORIS corridor1-1はcommit
  `cbc03108723d08322b23d0338680bffa9404cce9`、CC BY-ND 4.0。両T265
  fisheye先頭5,000 frameをtimestamp順にrectifyし、画像/派生物は外部のみ。
  10,000-image feature bankは1,427,634 keypoints、8 worker各peak <273 MiB。

OpenLORISの固定7N ladderは以下。全runはGT/extrinsicsをcandidate/match/mapへ
渡さず、temporal offsets 1/2/4/8/16/32 + same-time rig edge + VLAD fill、
ratio .8、min12、persistent 4-thread matcher、32-pair shard、cap96、compact
snapshot、seed16、sparse BA8である。

| tier | candidate / verified | registered | total | peak |
|---|---:|---:|---:|---:|
|1k|7,000 / 6,869|989|2:25|280,916 KiB|
|2.5k|17,500 / 16,321|1,223|7:10|401,308 KiB|
|5k|35,000 / 31,521|1,212|20:33|691,840 KiB|
|10k|70,000 / 58,879|199|1:03:45|1,869,412 KiB|

したがってM5はresource gate pass / connected-quality gate fail。10k peakは
2,188-shard merge 1.78 GiB、candidate 1.14 GiB、matcher 851 MiB、mapper
501 MiB。manifest出力は7NでもVLAD exact rankingは全画像pairをscoreし、10k
candidateだけで49:49かかる。dense 1kは23,157 verifiedでもUnionFind conflictで
2/1000、global 2.5kは1,459登録でもmean reproj 3,567 pxで棄却した。

証跡は
`benchmarks/electro/m5-openloris-connected-scale-validation.json` と
`docs/electro_m5_scale_validation.md`。このM5 branchを全test/PR/merge/branch整理
した次は、別branchで次の順にA/Bする。

1. exact top-K full sortをbounded selectionへ置換しcandidate hash完全一致で
   wallを短縮。その後ANN/inverted indexをbehavioral A/Bとして追加。
2. feature fileからtraining sample/global descriptorを2-pass streamし、candidate
   phaseでdescriptor bank全保持をやめる。
3. same-image conflictでsupportを捨てるUnionFind track builderを、geometry-supported
   alternativeを保持するbounded component builderへ置換。1k/2.5k/5k/10kの凍結
   manifestをすべて再実行し、resourceだけでなくregistered fractionをgateにする。
