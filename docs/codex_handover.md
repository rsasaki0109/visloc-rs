# visloc-rs COLMAP parity — Codex 引き継ぎ資料

## 現在の状態（以下の過去ログより優先）

連続native E2E診断v1（source c80f45f）はterminal success。
unit visloc-native-e2e-v1.service、MainPID0/SubState exited/Result success/exit0、
invocation5118aa2190cc4bc69110d602caa37e25。再実行・再poll不要。
全17stageが初回連続実行でexit0/completed。終了後に全stage artifact checkpointを
独立再検証して17/17 PASS。pipeline wall17005.153s、計測wall17006.100s。
sample aggregate peak RSS1806376KiB（1.72GiB）、cgroup peak2GiB、Swap0、
memory.events max366806、OOM/oom_kill0。メモリ圧迫なしとは主張しない。
最終scoreは9998画像/4999frame、RMSE0.388993m、p950.638173m。
COLMAP比でmapper20.41倍、phase合計比1.13倍、sampled RSS17.0%低いが、
RMSEはCOLMAP0.384307mより1.22%悪く品質gate未達。
またOS cache未制御、COLMAP側は連続wallでなくphase和、単発runのため、
同条件cold E2E性能acceptance・3反復・全SIGKILL restartは未達。
証跡benchmarks/electro/m8-native-e2e-v1.json。pipeline report SHA375ec7b...、
measurement SHA9f889ea...、score SHA74e839b...。
次はCOLMAP IncrementalTriangulator::Createとの差として残るbounded recursive
partitionを固定armで実装する。既存batched Createは最初のinlier集合だけを所有したが、
COLMAPは未所有残差が3観測以上なら別trackを再帰生成する。上限はCSR rowあたり
32近傍/128 ray-pair仮説/4排他的partition、GTはpublication後だけ。まず1kで
登録/RMSE/p95/再投影/time/RSSの一つでも悪化したら上位tierへ進めない。
通過後にcold条件統一、3反復、tier非回帰へ進む。
共有化済み既存model/VPSは上書き禁止。新runはfresh root。

M8 model/match共有化完了: apply_m8_duplicate_inventory.py、session60517 exit0。
固定一覧からsingle-link682filesのみ置換、19277934592bytes（17.95GiB）解放。
全事前hash/inode/link照合と全事後hash/shared inode確認PASS。
証跡benchmarks/electro/m8-model-match-dedup-applied-v1.json。全path/bytes保持。
対象model/VPSは以後上書き禁止、fresh outputまたは独立コピーを使用。
df空き45G（この操作で説明できる解放量は17.95GiBだけ）。16GiB容量guardは収容可能。
次は全cold executorの現状レビュー・固定pins確認・測定開始条件確認。
以下のinventory未適用という記録は過去状態。適用scriptの再実行不要。

容量整理候補を保存: scripts/inventory_m8_duplicates.py（読み取り専用）。
外部一覧m8-model-match-duplicate-inventory-v1.json、repo同名証跡にSHA固定。
206groups/888unique inodes、解放候補19277934592bytes。まだ適用していない。
対象はM8配下1MiB以上のimages.txt/points3D.txt/*.vpsのみ、symlink除外。
同一user FD監査で候補open0、ただし5process権限拒否あり（再確認sd-pam/sshd）。
visloc user service稼働0。将来writer排除/完全quiescence証明ではない。
次は固定一覧の対象レビューと適用前再照合、安全な共有化。inventory test1 PASS。

base-v3の1万特徴を参照実体m5/feature-extract/featuresへhardlink化完了。
scripts/deduplicate_base_v3.py、監査m8-base-v3-dedup-v1.json。
全件事前hash/参照symlink実体確認、事後全hash+共有inode確認済み。
1705103360bytes（約1.59GiB）解放、空き5.5GiB。全cold guard16GiBにはまだ不足。
両bankは共有inodeなので上書き禁止。編集は独立コピー、新runは新root。
sidecar/raw/log/計測証跡は変更なし。過去性能を共有化後の状態で再解釈しない。
helper2tests PASS、apply session90133 exit0。395d537のCI34310596934は直近実行中。

base-v3は正常終了済み（MainPID0/Result success/SubState exited）。再起動不要。
全6worker exit0。出力1万特徴と参照1万特徴を独立再SHA256照合し、集合・全hash一致。
証跡: benchmarks/electro/m8-full-base-extraction-v3-audit.json。
wall4673.225689s、sample aggregate peak RSS1474684KiB、OOM/oom_kill0。
cgroup peak2147483648bytesで上限到達、max19818（圧迫なしとは言わない）。
これはbase単体の新しい完走計測。v2の欠損ledgerや全cold E2E/品質gate達成とは別。
次はこの証跡のレビューとPR反映、全cold用容量確保。以下のbase稼働中記録は過去状態。

最新軽量作業: run_native_pipeline.pyに--resume（通常子exit>0の記録済み失敗限定）。
native_pipeline_resume.pyでplan/pins/成功prefixの全artifact checkpointを検証後、
失敗stageのoutput/log/payload/captureをfailed-attempts/<uuid>/へ退避し再実行。
過去reportも退避、完了済みstageは再実行しない。flockで実行中executorとの競合拒否。
関連14tests PASS：実child exit7→再開成功・完了inode保持・失敗成果物保持、
plan差分/未観測失敗/負のsignal終了/実行中lockを拒否。
SIGKILL/親強制終了/孤児child安全確認は未対応、全目標restart達成ではない。
保守的16GiB guardはresume時も維持。各attempt wallをcold E2E時間と混同しない。
base-v3はMainPID4045651で継続中、worker0=163/1667、OOM0、max5180。
実行中base-v3 script/inputには変更なし。重い実験・buildを追加していない。

最新軽量作業: native_pipeline_checkpoint.pyを追加、全体executorで成功stageの
output/capture/payload/logの内容checkpointとcompleted=trueを保存するよう変更。
ファイル・空directory・file symlink（target文字列+参照先hash）を記録。
directory symlink/特殊fileは拒否。変更/追加/削除/参照先変更を検出。
関連11tests PASS（実executor完了checkpointと事後log変更検出を含む）。
これはresume前提の出力検証のみ。--resume/失敗stage退避/whole-pipeline再開は未実装。
実行中base-v3のscript/inputは変更していない。新しいcheckpointは次の全体run向け。
base-v3は継続中MainPID4045651、直近worker0=84/1667、memory.events oom/oom_kill0。
memory.events max4767（圧迫あり）。終端RSS未確定、同unitを追跡し再起動しない。

実行中: visloc-full-base-extraction-v3.service、2GiB/Swap0/timeout22000。
probe_openloris_base_extraction.py --all-images --workers6 --variant base。
出力dataset/corridor1-1-m8-full-base-extraction-v3、
計測dataset/m8-full-base-extraction-measurement-v3。同unitを追跡、再起動禁止。
目的はv2で欠けた終端aggregate RSS/cgroup/OOM記録の取得。特徴一致だけで完了扱いしない。
既存frozen extractor extract-3ae253aを使用、出力見積約1.8GiB/開始前空き5.7GiB。
他の大型run/buildや入力bank変更を並行しない。完了後独立1万特徴hash監査と
measurement terminalを照合。v2自体の不完全ledgerを後付けPASSに変更しない。
起動head db9c76c。全cold E2Eではない、#138 base resource gap解消のための再計測。

CI cd1c5fa/run34304912157はglobal_sfm disconnected fixtureでFAIL。
ローカル別プロセス反復でも再現。fixture HashMap列挙が対応順を毎回変えていた。
db9c76cでfixtureのみBTreeMap化（本番solver変更なし）、push済み。
global_sfm19tests PASS（build79020 exit0/2m20s）、対象別process100回PASS、fmt PASS。
入力順依存の本番一般性を解決したとは主張しない。新head CIは未確認。

最新容量整理: 完了済みfull-dense-extraction-v1/featuresの2万filesを
dense256x2-full10k-v2/featuresへ全hash確認後hardlink化。
scripts/deduplicate_dense_extraction.py、plan session84735/apply43516ともexit0完了。
4,765,159,424bytes（4.44GiB）解放、空き5.7GiB。
全出力を事後再hashし共有inodeも全件確認。証跡m8-dense-dedup-v1.json、
外部audit dataset/m8-dense-dedup-applied-v1.json。元reportSHA3fabb11f...不変。
feature内容/path/manifest/抽出ログ/計測証跡は保持。inode/metadata共有なので
両feature bankを絶対に上書きしない。編集が必要ならまず独立コピーを作る。
新抽出は必ず新rootに実行。整理後のcache/共有状態で過去の性能比較を再解釈しない。
generic merge helper2tests PASS、専用script py_compile+実全2万検証PASS。
全cold開始guard16GiBには不足（残り約10.3GiB）。次は他の検証済み重複を調べる。

最新: OFF/ON既存phaseログ集計完了、証跡m8-feasible-backtrack-phase-accounting-v1.json。
main normal_equations/linear_solve event各2804で変化なし（factorization数そのものではない）。
linear_solve27.038570→26.998891s、tentative_update_and_cost29.963184→109.656562s。
ON main3430縮小候補/291受理。再solve削減仮説は今回成立せず、棄却を維持。
容量整理: scripts/deduplicate_atlas_parity.pyで完了済みSchur診断3rootの14filesずつを
canonical schur-feasibility-v2と全hash確認後hardlink化。42files、929980416bytes解放。
raw/feature/log/measurement/ONモデルは変更なし。全path/bytes保持、inode metadataは共有。
完了済みrootのモデルを絶対に上書きしないこと。必要なら独立コピーを作ってから利用。
監査dataset/m8-schur-dedup-applied-v1.json（phase証跡内SHA）、planも保存。
dedup helper2tests PASS、実全対象hash検証+終端unit確認+適用完了。空き約1.3GiB。
全cold pipelineの容量には不足。新しい大型runを開始する根拠にはしない。

最新確定: feasible-backtrack ON v1はterminal success/MainPID0/exited。
再poll/再実行不要。実験は棄却。RMSE OFF0.3889930047→ON0.3891842840mで悪化。
p95は0.6381734851→0.6380699612m、登録9998/GT採点9306。
scorer/manifest/GT/transform/aliases/gapの同一性を確認。pre-ba+camera8hash独立一致。
integration区間OFF153.3184523s→ON233.9327779s（単発比較、全wallは比較不可）。
ON計測全wall237.1764653s/sample aggregate RSS560852KiB。
証跡m8-feasible-backtrack-on-v1.json。既定OFFを維持し同設定再実験・GT閾値調整禁止。
採点session19580はexit0終了。runner9c25b16、binary37298b3。
空き386MiB。大型run/build不可。まず保存済みOFF/ON phase timingの内訳を集計し、
無駄な再solve削減仮説を見直す。次の実験前に検証済み成果物の容量整理が必要。
raw/feature/reference銀行は削除しない。全目標・COLMAP品質gateは依然未達。

実行中: visloc-feasible-backtrack-on-v1.service（2GiB/Swap0/timeout1000）。
runner9c25b16 scripts/run_atlas_backtrack_trial.py、同OFF binary37298b3。
出力dataset/corridor1-1-m8-feasible-backtrack-on-v1、
計測dataset/m8-feasible-backtrack-on-measurement-v1。同unit追跡、再起動禁止。
mainでbacktrackログを確認。ON未完走、入力pins対象のscript/JSON/binaryは変更禁止。
OFF stitchを再利用するため速度比較はintegration stage同士のみ（全wall比較禁止）。
pre-ba全3filesとmodel/cameras完全一致を要求、post modelはhash記録し品質未評価扱い。
runner出力validator2tests/既存exact executor3tests PASS。起動時空き683MiB。
事後scorerはscripts/score_openloris_model.py（SHAc0196be50b4b7ee385bd8db437a8fa6cae508fb0d4a9ecc4038981c94455d5d0）
--manifest dataset/corridor1-1-m5/manifests/tier-10000.json
--ground-truth dataset/official-groundtruth/calibration/corridor1-1/groundtruth.txt
--transform-matrix dataset/official-groundtruth/calibration/corridor1-1/trans_matrix.yaml
--model-images newroot/integrated/main/model/images.txt とtailも指定、aliasesなし。
従来scoreはdataset/corridor1-1-m8-atlas-landmarks-v1/connected-filtered-ba-v1/score.json。

最新確定: feasible-backtrack OFF v1はterminal success/MainPID0/exited。
再poll/再実行不要。14参照hashをsha256sum --checkで独立全一致。
main/tail backtrackログ0。wall155.1414069s、sample aggregate RSS560748KiB。
証跡m8-feasible-backtrack-off-v1.json。ON未実行、品質・高速化は未証明。
CI37298b3/run34303656924はterminal success（ON synthetic test含む）。
次は同binaryのONを評価。ただし既存probe/execute_atlasはpost model完全一致を
要求するのでON品質比較にはそのまま使えない。厳密parity runnerのgateを緩めず、
実験専用runnerでpre-ba一致・出力hash・事後品質採点を扱うこと。
現在空き684MiB。新build不要、追加runは出力予算を再確認して一件のみ。

実行中: visloc-feasible-backtrack-off-v1.service（2GiB/Swap0/timeout1000）。
新binary source37298b3、build80812はexit0（1m52s）。
保存schur-probe-binaries.4WAfOy/integrate-37298b3、
SHA10b3698d14bbec10c61d35e4ddc25eae8e8d412fa1ffdd39cc46cb42ae20c05c。
出力dataset/corridor1-1-m8-feasible-backtrack-off-v1、
計測dataset/m8-feasible-backtrack-off-measurement-v1。
probe_atlas_schur_diagnostic.py診断なしで実験OFFの14参照hashを検査中。
同unitを追跡、完走後に独立hash監査。再起動しない。ONは未実行。
起動直前空き983MiB、過去出力約310MBで今回1件のみ収容可と判断。
これを全体cold pipeline/ON比較の容量許可と解釈しない。
CI37298b3/run34303656924は直近rust実行中、他8jobs success。

最新: joint variable pose+point（回転固定）もproduction fixtureへ追加。
session48581はexit0（release2m51s）。OFF/ON各4 tests PASS。
jointでalpha0.5拒否/0.25受理、両更新nonzero・step norm一致、回転exact不変。
4候補尽きた場合のproblem全体exact rollbackも両modeでPASS。
CIに実験flag ONの別プロセステストを追加。実atlas OFF一致とA/Bは次の未完作業。
06a21fdまでpush済み、CI34303430195は直近確認時in_progress。
全体CI成功・性能改善はまだ主張しない。

最新作業: bundle.rsにVISLOC_SFM_BA_FEASIBLE_BACKTRACK=1限定の
4候補joint pose/landmark step縮小を実装（既定OFF、実データ未適用）。
pure rig/legacy sparse/LM/velocity+biasなしに限定。追加全体snapshotなし。
有限cost低下・既存非投影数gate・以前validな全rig観測のvalid維持を要求。
失敗候補尽きたら強制拒否→既存rollback。受理時step normをalpha倍。
session56698/23080はexit0完了、再poll不要。generalized_rig_factor_testsは
OFF/ON別プロセス各4件PASS。production固定pose fixtureでalpha0.5のcost悪化拒否、
alpha0.25受理、別fixtureの4候補尽きた完全rollback、固定pose不変、step norm一致。
有限cost/既存valid観測維持predicateもPASS。joint variable pose+point、固定rotation、
実atlas OFF byte parityは未検証。次はこの不足テストを補い一つの固定armを評価。
session22417 Clippy --lib --release -D warnings PASS（8.29s）。
診断step-detailのfeasibility表示もexhaustion込みgateへ修正済み。
空き973MiB。実データrun禁止、まずテスト契約と容量予算を満たすこと。

2026-09-09 最新確定: feasibility v2はterminal success、再実行不要。
14参照hash一致、115診断行の全観測が更新前projectable。
5内部trackの更新前sensor depthは0.000068〜0.038669m。
証跡m8-schur-feasibility-v2.json。wall209.103801s/RSS550604KiB。
品質改善ではない（RMSE 0.388993m > COLMAP 0.384307mのまま）。
次の候補は点削除ではなく有限回のfeasibility-preserving step縮小。
詳細・棄却条件はopenloris_atlas_bounded_ba.md末尾。未実装。
空き1.1GiB、追加大型runを開始しない。既存データ削除なし。
CI adb8fe6/run34302051261はClippy too_many_argumentsでFAIL。
solve_stepから分離したsolve_step_with_debugの局所allow漏れを修正済み。
cargo clippy -p visloc-slam --lib --release -j1 -- -D warnings PASS（19.55s）。
cargo fmt --all -- --check / git diff --check PASS。新headの全体CIは未確認。
以下のbuild/run稼働中という記録は過去ログ（現在完了済み）。

最新: build39403はexit0完了（1m55s）。保存binary integrate-5def794（既存schur-probe-binaries.4WAfOy内）、
SHA1539a5d848fcc30553471746781d007821da0bc7108373dbd1d92ba61a19a689。
unit visloc-schur-feasibility-v2.serviceを2GiB/Swap0/timeout1000で起動。
出力dataset/corridor1-1-m8-schur-feasibility-v2、計測dataset/m8-schur-feasibility-measurement-v2。
同v2を追跡し14hash比較とbefore深度/投影可否/stepを監査。v1は終了済み。
起動前空き1.4GiB、出力見積約310MB。追加大型runを並行起動しない。

実行中: integration release build session39403（source5def794）。同sessionをpoll。
完了後は新binary名で保存/hash確認し、新probe outputでbefore-depth診断を実行する。

最新: session25100はexit0完了（release2m50s）、generalized_rig_factor_tests全2件PASS。
before-depth診断ログの実データ検証は未実施。次にintegration binaryをbuildする。

最新: feasibility sampleにbefore_sensor_depth/before_projectable/point_stepを追加中。
saved_poses/saved_landmarks（既存rollback状態）を利用し追加全体コピーなし。
受理gate/モデル更新は変更しない。未commit、実ログ未検証。
cargo test -p visloc-slam --lib generalized_rig_factor_tests --release -j1
session25100がビルド中。再起動せずpoll。rustfmt/diff check PASS、空き約1.4GiB。

最新: feasibility v1はterminal success/MainPID0/exited、14参照hash独立全一致。
wall158.972974s/sample aggregate RSS550652KiB。再poll不要。
main115失敗観測行、内部trackは140/407/578/641/12192の5件、各反復最大12件。
tail失敗行0。最大消去寄与4266とは異なる。証跡m8-schur-feasibility-v1.json。
次はこの5trackの更新前後depth/geometryを取得し、受理gateを維持する改善仮説を検討。
内部ID→COLMAP IDの直接joinは禁止。品質改善・因果はまだ未証明。

最新: build99071はexit0完了、5bb84bbをpush済み。
新保存binary schur-probe-binaries.4WAfOy/integrate-5bb84bb、
SHA7194864f91b057878bfec1472f9ddd6ef41e677cf916195fa36219024984372a。
unit visloc-schur-feasibility-v1.serviceを2GiB/Swap0/timeout1000で開始。
出力dataset/corridor1-1-m8-schur-feasibility-v1、計測dataset/m8-schur-feasibility-measurement-v1。
同unitを追跡し、完了後モデル14hashとsfm-debug-ba-rig-infeasibleを監査。
既存OFF/ONは完了済みなので再poll不要。

最新: session72367はexit0終了（release2m24s）、bounded sample test PASS。
generalized_rig_factor_tests全2件もPASS。追加診断は実データではまだ未実行。
空き1.7GiBなので新probeは一度に一件、保守的に出力約310MBとbinary容量を確認する。

最新: first-window step gateをsolver境界で集計（step-detailは拒否のみ出すので
先頭N行をwindow扱いしない）。main20反復15拒否/feasibility14、tail14反復拒否0。
証跡m8-schur-first-window-step-gates-v1.json、commit9c2dff5。
bundle.rsに非投影rig観測の最大16件sampleを追加中。最初の診断windowかつ
feasibility失敗時のみ、仮更新後のobservation index/frame/track/sensor depthを出力。
本体rig_residual_jacobians判定を再利用。新規失敗だけではなく仮更新後失敗のsample。
関連test build session72367稼働中。再起動せずpoll。未commit、実モデル未検証。

最新: Schur ON v1はterminal success/MainPID0/exited。再poll不要。
OFF/ON/凍結参照の14filesを独立hashし全一致。
main診断20行(iter0..19/frame0)、tail14行(iter0..13/frame4495)、OFF0行。
全行finite=true/singular_hll=0。初回適格BAのみの出力範囲を確認。
ON wall152.620315s/RSS550680KiB、OFF176.969935s/RSS560632KiB。
single sequential/cache条件が異なるため速度改善とはしない。
証跡m8-schur-diagnostic-parity-v1.jsonに14hash/測定/診断各行の主要値。
次は最大消去寄与trackと低視差大移動trackの対応を調べる。全pose/全window寄与は未測定。

最新: Schur OFF v1はterminal success/MainPID0/exited、14参照ファイル独立hash全一致。
wall176.969935s、sampled aggregate peak RSS560632KiB。再poll不要。
ON unit visloc-schur-parity-on-v1.serviceを同binary/入力・--diagnosticで起動。
出力dataset/corridor1-1-m8-schur-parity-on-v1、計測dataset/m8-schur-parity-on-measurement-v1。
同ON unitを追跡し、完了後14file一致とSchurログのfirst-window制御を監査。
ON未完了。空き起動前2.0GiB。実行スクリプト/binaryは変更しない。

最新: integration build session76465はexit0完了（2m12s）。
保存binary /home/sasaki/datasets/openloris/schur-probe-binaries.4WAfOy/integrate-dee3141
SHA276c5d90452f938db49bf700591c87f98ceef9c4012e90af4bdb26676dd7782b。
probe_atlas_schur_diagnostic.pyを追加（source/rig検証→stitch/integration参照比較）。
OFF unit visloc-schur-parity-off-v1.serviceを起動、2GiB/Swap0/timeout1000。
出力dataset/corridor1-1-m8-schur-parity-off-v1、計測dataset/m8-schur-parity-off-measurement-v1。
同unitを追跡。OFF PASS後に新ON出力で診断比較、まだON未開始。
Python既存93 tests PASS、新probeはpy_compile確認で実試験進行中。

最新: session19766はexit0完了、release2m57s、Schur関連3 tests PASS。
first-window claim（不適格非消費/並行一意）と非zero解不変性を確認。
次はatlas integrationの実binaryをbuildし、同入力で診断OFF/ONのmodel bytesを比較。
基準binaryはdataset下の保存コピーを維持し上書きしない。

最新: first-window制御をbundle.rsに実装中（未commit）。
VISLOC_SFM_DEBUG_BA_SPARSE_FIRST_WINDOW + 既存VISLOC_SFM_DEBUG_BA/
VISLOC_SFM_DEBUG_BA_STEPS/VISLOC_SFM_DEBUG_BA_SCHUR_SLOTで最初の適格rig sparse BAを選択。
呼出内全反復で同slot使用。プロセス単位AtomicBoolで一度のみ、無効slot/非適格は消費しない。
並行8呼出の一意claimテスト追加。cargo test --release schur_block_debug_tests
session19766がビルド中。再起動せずpollする。rustfmt/diff check PASS。
実model不変性・診断結果はまだ未検証。選択は単一thread atlas前提で再現性確認が必要。

最新: Schur診断fixtureを非zero pose/landmark RHSへ強化。
診断ON/OFFの解exact一致、両更新norm>0、同pose合算後の消去norm一致PASS。
session53281はexit0終了、release2m24s、関連2 tests PASS。再poll不要。
公開済み03e0368のCI34298844551はsuccess（この新Rust変更のCIではない）。
first-window制御と実model byte比較は引き続き次の作業。

最新: session91383はexit0終了。release build 3m07s、追加した診断有無の解一致test PASS。
続いてschur_block_debug_tests全2件PASS。build再poll不要。
現一致fixtureはzero RHSなので非zero RHS・実model byte比較を追加する必要がある。
first-window限定もまだ未実装。診断接続のみを完了とし品質改善を主張しない。

最新: bundle.rsにlegacy sparse→既存Schur診断への接続を実装中（未commit）。
solve_step_with_debug/solve_step_pose_blocks_with_debugは実inverse cache/reduced blockを使用。
既存debug flags/slotのgateを維持。診断有無の解一致testを追加。
`cargo test -p visloc-slam --lib debug_context_maps_variable_slot_after_fixed_pose_and_counts_rig_crosses --release -j 1`
session91383がビルド稼働中。再起動せずpollすること。rustfmt実施/diff check PASS。
最初の1window限定・実model bytes比較は未実装/未検証。空き約2.3GiB。

最新: pose coupling実装箇所を確認。既存collect_schur_block_debug_countsは
同poseのcross合算・solver inverse cache利用に対応するが、emitはmatrix-free側のみ。
atlas実経路solve_step_pose_blocksのinverse cache構築後/factor前へ接続が必要。
index mapのframe/landmark IDを渡し、export point IDは使わない。
診断機能の新規接続はまだ未実装。詳細はopenloris_atlas_bounded_ba.md末尾。

最新: main/tail各移動量上位5点の観測ray角とcamera range/translationを監査。
main上位5は全2観測、pre角0.000252〜0.112018deg。
最大1004.613m移動点はpre距離1043m、post38.4m、観測camera最大移動0.004214m。
低視差と点の大移動は確認したがpose RMSEの原因・Jacobian影響は未証明。
証跡`m8-atlas-top-motion-geometry-v1.json`。rangeは軸方向depthではない。
次はposeへの残差/Jacobian寄与を診断。既存weak-angle freeze棄却armの単純再実行禁止。

最新: ObservationKeyでpre-ba/modelを対応付けた診断完了。
image ID/name一致、全final trackが一意の旧trackのsubset、新規/曖昧track0。
削除957点・総14440観測で既存filter ledger一致。
mainの旧2観測群216828点の平均移動0.043919m、最大1004.612687m。
単純ID join結果とは異なる有効対応だが、軌跡誤差への因果は未証明。
証跡`m8-atlas-observation-key-motion-v1.json`に8入力hash/集計手順/全群統計。
次は大移動点の視差角・深度・pose couplingをGT-freeで調査。閾値変更はまだ行わない。

最新: 品質診断に戻りmapping-v2 main pre-ba/modelのpoint対応を監査。
共通数値ID318222件中、同一観測集合7件、観測集合disjoint318215件。
integrate_rig_atlas_landmarks.rsの出力は順序付け後index+1でIDを再採番する。
従って数値point IDでBA前後をjoinした移動量・観測数変化の集計は無効として棄却。
次はObservationKey（image identity, feature index）の対応で点集団を比較する。
GT未使用、モデル変更なし。再投影低下だけで軌跡改善しない既存Ceres診断に沿う。

最新: pipeline checkpointを既存atomic_jsonへ切替、child起動前にactive_stageを記録。
atomic replace失敗時の旧JSON保持・起動前記録の試験を追加、93 tests PASS。
file fsync+renameでありdirectory fsyncなし。電源断durability/full restartを主張しない。

最新: 全体pipeline CLIはcgroup v2のmemory.max=2147483648/swap.max=0を必須化。
通常シェルからの実CLIはexit1/期待診断/出力なしを確認。91 tests PASS。
制限確認だけでは独立monitorの存在を証明しないためlaunch_native_measurement経由を維持。
ライブラリexecuteの単体テストはCLIガード外。全体実データ実行は未開始。

最新: pipeline reportに実行計画由来のartifact_lifetimesを追加（削除機能なし）。
base/native match最終利用はprefix admission、adaptive matchはrepair admission、
targeted matchはfinal admission。dense/adaptive特徴は最後のmappingまで必要。
実測dense特徴4,437,450,752 bytes、loci327,708,672 bytes。
dense特徴だけで空き2.4GiBを超えるため、早期releaseだけでは現方式を実行できない。
baseのhardlinkを消してもadaptiveが共有するinodeのbytesは解放されない。
90 tests PASS。release設計はrestart保持契約と別途統合が必要。実データ削除なし。

最新: pipeline CLIは全scripts/*.py・electro JSON/TSV・9 binaryのhashを記録し、
stage前後で再検証。変更されたらexit0のstageでも全体FAILで後段停止。
実子プロセスの変更注入を含め90 tests PASS。これは境界検出で物理immutable化ではない。
外部raw/calibration/reference全体の固定と、実行中変更→復元の検出はまだ保証しない。

最新: 全体executorの候補生成binaryを実証跡と照合し分離。
nativeはa7ff5ff/8eeee5c、denseは3ae253a/8cfa9c5。開始前に各hashを固定検証。
9 binaryの実パス/hashを`benchmarks/electro/m8-native-pipeline-binaries-v1.json`に保存。
同specで`run_native_pipeline.py --plan-only`が17 stageを生成することを実確認。
88 tests PASS。長時間の全体実行は開始していない。16GiB guardに対して空き2.4GiB。
全binary/script/inputの凍結検証・容量lifetime・restartは依然残る。

最新: `run_native_pipeline.py`に全抽出→候補→4系統shared matching→3 admission→
prefix/target selection→source/atlasの接続実装を追加。88 Python tests PASS。
実データで全体未実行、restartなし、全script/inputのimmutable provenance凍結も未完。
16GiB freeを保守的な開始条件とする（実測lifetime上限ではない）。現在空き2.4GiB。
自動削除なし。`--plan-only`と8 binaryのpath/sha256 JSONで計画を検査可能。
次はbinary specを凍結し全コマンドと参照互換性をpreflight、容量確保/lifetime設計。
特にnative/dense候補生成に同candidate binaryを使う接続は実データ未検証。
品質gate未達とfull restart、連続cold測定の要求は変わらない。

最新: mapping v2は正常終了（MainPID0 / exited / Result success）。再poll不要。
21 source / 23 nodes / stitch / tail+main integrationが全PASS。
参照86ファイルを独立再hashして全一致、nodes.tsvも新runへの23 bindingに一致。
入力検証込み連続wall926.828秒、sampled aggregate peak RSS572860KiB、
cgroup peak1182564352 bytes（RSSではない）、OOM0。
証跡`benchmarks/electro/m8-native-mapping-bound-v2.json`。
既存frontendからのmapping suffix一回の結果でcold E2E・restart・品質改善ではない。
次はfrontend全段接続と容量/lifetime設計、未達RMSE gateの品質改善。

最新: 補助ファイル許可/未知出力拒否/参照欠落/内容変更の実子プロセス回帰試験PASS、
Python85 tests。修正head `b66d155`をpush済み。
新unit `visloc-native-mapping-bound-v2.service`を起動（2GiB/Swap0/timeout6000s）。
出力`/home/sasaki/datasets/openloris/corridor1-1-m8-native-mapping-bound-v2`、
計測`/home/sasaki/datasets/openloris/m8-native-mapping-bound-measurement-v2`。
このv2を同unitで追跡し、terminalを確認するまで再起動しない。
v1の測定はfail/exit1、46.179秒、sampled aggregate peak RSS376044KiB、OOM0。
最初のsourceはmodel一致だが全pipeline結果ではない。v1は保持。

最新: 上記mapping v1はterminal FAILED（MainPID0 / Result exit-code）。再poll不要。
最初のsource-replay-1950はmapper exit0、参照model hash全一致、31.800秒。
失敗原因は参照audit対象外のcomponents.tsv/retrieval-components.txtを厳密dict比較で
余分と判定したexecutorバグ。既知2補助ファイルだけ許可し全hash記録を保つ修正を実施。
未知出力・参照欠落・hash不一致は引き続き拒否。85 tests PASS、追加回帰テストが次。
v1出力・measurementは失敗証跡として保持。修正後は新v2パスで実行すること。

最新追記: mapping suffixの実データ計測を開始。
unit `visloc-native-mapping-bound-v1.service`、開始確認MainPID3927517。
測定root `/home/sasaki/datasets/openloris/m8-native-mapping-bound-measurement-v1`、
出力root `/home/sasaki/datasets/openloris/corridor1-1-m8-native-mapping-bound-v1`。
再起動せず同unitのterminal状態とreportを確認すること。MemoryMax2G/Swap0、timeout6000s。
入力spec `benchmarks/electro/m8-native-mapping-bound-inputs-v1.json` は事前検証PASS。
全dense新規抽出bankと新targeted admissionを使用、dense snapshot等はretained。
従ってcold E2Eではない。保存出力からの容量見積はsource326MB+integration309MB。
実行head `1153a0479c9e2a676b46f27a16e7589aa229b3b1`、CI34296178485は開始時in_progress。
atlas実子プロセスexit0/7試験を追加しPython85 tests PASS。SfM実行結果とは区別。

最新追記: `run_native_mapping.py`でsource全21→nodes23→atlas結合/BAを接続。
`--inputs`は5入力のpathとsha256（feature bankはmanifest/manifest_sha256）を要求。
3 binaryは証跡hashを検証。`--validate-only`は出力を作らない。
Python 84 tests成功。接続と失敗停止はmock検証で実データ実行は未確認。
これは既存frontendからのmapping suffixであり、cold E2Eではない。
入力hashは指定specへの一致で、spec自体の凍結参照との一致は呼出側の責務。
資源制限は`launch_native_measurement.py`の外側wrapperを必ず使う。
空き実測は内蔵3.0GiB/外部1.6GiB。全抽出DAGは容量/lifetime設計が必要。

最新追記: `c718cf7`のCI run 34295216906はsuccess。
`execute_native_atlas.py`にstitch→tail/main integrationの逐次実行を追加。
binary/input hash記録、出力先拘束、timeout、参照14ファイルhash gateで失敗時停止。
Python 81 tests成功（atlas実行はmock検証であり実SfM実行未確認）。
まだsource phaseとの単一CLI接続、frontend全DAG、入力provenanceの全検証、
restartと連続resource計測は未実装。cold E2E完了・品質改善とはしない。

最新追記（2026-09-09）: dense全10k抽出サービス
`visloc-full-dense-extraction-v1.service`は正常終了（MainPID 0 / exited / exit0）。
再起動・待機pollは不要。20,000 feature/lociファイルを独立に再hashし参照全一致、
6 worker全exit0、計測レポートPASS。抽出＋検証9,789.925秒、50msサンプリングの
合計peak RSS 1,776,952 KiB、OOM 0。証跡`m8-full-dense-extraction-v1.json`。
現branchは`feat/shared-matching-recipe`。空き約1GiBのため大容量stageは要容量確認。
連続native DAG・COLMAP品質/速度比較は未達。base抽出の欠落resource ledgerも
今回のdense測定では埋まらない。サブエージェントは使わない。

最新（2026-09-09）: PR118/119に続きPR120（head`e136ae6`、CI9成功）は
`09e79b8dae067521ee8b8be342c1373c68e090c8`へmerge、旧branch整理済み。
整理記録のPR121（head`8b05461`、CI9成功）は
`fdf94400e98f80b2cb81c45c86b0441a7977d6e9`へmerge、旧branch整理済み。
PR122（head`506510e`、CI9成功）は`8ca18505b11ad871ef1329be44d2b5a5328a4bac`
へmerge、旧branch整理済み。空語彙修正はPR123（head`ee0fa5a`）、CI9成功後
`d5e465e41d776f01d5b197683dc5b9613f823a14`へmerge済み。
現作業branchは`test/native-runner-shared-restart`（PR130の子）。
サブエージェントは使わない。

- native候補生成は現代版`3ae253a`だと3,761ペア差。旧版`a7ff5ff`再buildで
  70,000候補・2,188 matching shards・merged snapshotが全bytes一致。
  score順fillへの仕様変更を確認した。旧版への本番ロールバックではない。
- adaptiveはnativeと同じ候補計画。再生成済みバンクからmatching/mergeを実行し、
  61,286ペアのmerged snapshotが全bytes一致。元の個別shardがないため比較は統合境界。
- targeted7のmatching448 shard・統合snapshotは全bytes一致。
  前段の記録コマンドを同thread実行ログから回収し、repair19 snapshot +
  `--deferred-registration-pair-prefix 59961`で再実行。登録manifestと導出した7 frame、
  そこからの14,319候補は全bytes一致。中間modelのmain images/pointsとcomponents.tsvは
  異なるため、全model再現とはしない（後段は登録一覧のみ使用）。
- denseも80,000候補・v2分割2,500・matching全shard・統合snapshotが全bytes一致。
  現行streaming mergeはpeak18,188KiB（統合単体）。full extraction/E2Eではない。
- 692画像の追加特徴はdenseバンクの同名特徴と全bytes一致。denseを直接supplement入力に
  してadaptive10kバンク全hash一致を確認（resume検証、書込0）。連続実行では重複抽出を省ける。
  base側はSIFT256/1 orientation/contrast0.02で別レシピ。base/D両方の全10k新規抽出は未完了。
- 新規証跡と具体的コマンドは`docs/openloris_frontend_reproduction.md`。
  adaptive/targetedでは退避済みの再生成バンク（外部ext4）を使用。過去と同一I/Oではない。
- 許可済み整理で空き約21 GiBを確保し、新規replay後は約15 GiB。
  整理ログと検証付き重複削除scriptはPR121経由でmainへ。元データ・検証ログは保持。
- 次: この変更のPR/CI/merge、base抽出の小規模一致probeと全10k抽出・
  continuous E2E・未達のCOLMAP軌跡品質改善。phase時間の和をE2Eとしない。
  100k前に共有metadataを含むartifact増加率を監査する。vocabなし時のstreamed候補生成の
  all_pairs(N) fallbackを削除し、明示エラーで停止するよう変更。100k空特徴の実CLIは
  2GiB仮想メモリ制限下でexit1・期待診断・出力なし、1.00s/102,892KiB。
  通常の100k SfM/共有metadataの容量保証ではない。64 example tests通過。
  全goalは未達。READMEに未検証の高速化/品質改善を追加しない。

更新: base抽出の4画像probeは完了。両カメラの時系列に散らした4画像・584特徴が
保存済みbaseと全bytes一致。scriptは`probe_openloris_base_extraction.py`、
証跡は`m8-openloris-base-extraction-probe-v1.json`。実行session64673はexit0完了。
全10k抽出/E2Eではない。Python40 tests/py_compile通過。大規模抽出前に容量を確認し、
base/Dを別レシピで処理する。

更新: dense snapshot全2500の構造監査完了。総1,294,392,088 bytes中
共通envelope777,135,000 bytes、重複分776,824,146 bytes。全envelope SHA一致。
固定ペア数shardに全画像metadataを反復するため、O(N) shardでO(N²)保存/I/Oになる。
次は共有envelope+内容IDに結びつくpair chunkの設計・実装、v1互換/restart検証。
証跡`m8-dense-snapshot-envelope-audit-v1.json`。監査script自身はpayload checksumを検証しない。
Python43 tests通過。全goal未達。PR123旧branch整理と現branchのPRが次。

更新: PR123旧branch整理済み、probe/監査PR124は最終head`9e4f0d4`でCI9成功後
`e5ec9d6b29c625937e79e507ac3374318f4e1d8b`へmerge済み。
共有snapshot形式を実装。明示flag `--shared-snapshot-envelope`のみ、既定v1は変更なし。
実dense2500 shard/64862 pairsを変換し全Snapshotレコード一致。
保存量1,294,392,088→517,615,442 bytes（envelope1個）。変換検証16.90s/7820KiB。
証跡`m8-dense-shared-snapshot-parity-v1.json`、設計/制約`docs/shared_snapshot_envelopes.md`。
lib15/example64 testsとClippy通過。現形式は読書きごとにenvelopeを再検証するため
I/O/CPUのN²解消は未達。次はbounded cache、強制中断/同時publication、worker実出力と
100k stress。変換session79247はexit0完了。全goal未達。

更新: PR125初回CIはlib testの不要cloneによるClippy警告1件で失敗。
修正head`3b3464a`をpush済み、子branchへrebase済み。mergeは最終CI待ち。
mergeの各passに1-entry envelope cacheを実装、load/eviction/終了時検証・pass間破棄。
lib17 tests（並行8 writer/キャッシュeviction・変更検出を含む）通過。
実dense2500 shardのshared/legacy入力を同binaryで統合、双方全SHA一致。
shared10.54s/16644KiB（単発warm-cache、E2Eではない）。証跡shared-merge-cache-v1。
両実行session58380/10776はexit0完了。standalone readはuncached。
writer側とdecoded envelopeの反復CPU処理は未解消。次はprepared writer/borrowed envelope、
強制中断restartとworker実出力/100k。全goal未達。空き約14GiB。

更新: PR125修正head`3b3464a`はCI9成功後`b25c3d8ddedd0653746fd1c9db1089a85affb63d`へmerge。
SharedSnapshotWriter/SharedPairChunkを追加し、persistent workerの明示shared出力へ組込。
画像envelopeを1回だけ準備し、以後pairのみ書く。lib18 tests/example64 tests・tests込みClippy通過。
合成1k/10k/100k writer stress成功、100k 99999 pairs/3125 chunks/8.86s/19324KiB。
保存量は約20MB/200MB/2003MBで線形。seedのペアを合成隣接画像へ複製、manifest/hashは
合成placeholder。writer-onlyで全出力readback・restart・SfM品質は未検証。
証跡`m8-shared-writer-scaling-v1.json`、外部root`corridor1-1-m8-shared-writer-stress-v1`。
session92952はexit0完了。空き約11GiB。次はborrowed reader、restart、実worker parity。
現branchはcacheとprepared writerをまとめてPR化。全goal未達。

更新: PR126はhead`e3312fd`でCI9成功後`70f221d69ce4ac53a9d08b701ae421ba2ea7afed`へmerge。
実workerの8-shard probeを開始。`scripts/probe_shared_match_worker.py`で10k dense特徴の
全manifestと候補hashを検証し、全scheduleから等間隔8 shardを共有形式で再matching。
完了後`compare_verified_pair_snapshots`で従来shardの全レコードと比較する。
現在exec session55995、worker PID3573965で稼働確認（elapsed2:16/CPU2:15、RSS451512KiB）。
出力root`/home/sasaki/datasets/openloris/corridor1-1-m8-shared-worker-probe-v1`。
まだ完了/一致とは判定していない。初期全10k特徴読込・検証中。timeout1800秒。
同sessionをpollし、観測timeoutだけで再起動しない。稼働中は同時build/性能測定を避ける。
Python43 tests/比較example Clippy通過。PR126旧branch整理とprobe結果の記録が次。

更新: PR126旧branchはremote/local整理済み。8-shard probe session55995はexit0完了、
出力membershipと全Snapshotレコードが従来8 shardと一致。peak451184KiB。
証跡`m8-shared-worker-probe-v1.json`。保存binaryはprobe rootのsfm-b02f85e/compare-1180dc1。
続いて全2500 shard/80000候補の共有形式workerを開始、exec session60175。
出力root`/home/sasaki/datasets/openloris/corridor1-1-m8-shared-worker-full-v1`、timeout1800秒。
全件の終了/一致は未判定。完了後report.jsonのexit/membership/比較を確認する。
同sessionをpollし、観測timeoutで再起動しない。保存binary使用、同時build/測定なし。
全goal未達。full worker後もborrowed reader/restart/抽出E2E/COLMAP品質は残る。

更新: PR127はhead`d5b0386`でCI9成功後`a5d51961e6e8eb5d6bce7e2623aa3f30fd58759e`へmerge。
full workerは同session60175/PID3577312で継続、直近1803/2500 shard（elapsed7:41）。
完了後の全出力比較は未確認。保存binary実行中、別build/測定はしない。
新branchでborrowed envelope readerを準備中（src/verified_pair_snapshot.rsとshared.rs）。
pair decoderを分離、解析済みenvelopeをArcで共有し、同一Arcのmetadata再比較を省く。
公開Snapshot API/v1形式は維持。参照共有と不一致拒否のテスト追加、まだ未build/未test。
このWIPは今回のagent変更。worker終了後にlib/example test・tests込みClippyを実行してから
実dense merge全SHA一致と100k読取を確認する。PR127旧branch整理も次。全goal未達。

更新: full worker session60175はexit0完了。全2500 shardのmembership/全レコードが従来一致。
worker594.68s/451060KiB（以前legacy557.35sより速いとは示せない、単発/別run履歴）。
証跡`m8-shared-worker-full-v1.json`。保存binaryと出力保持。再起動/poll不要。
borrowed envelope実装はlib19/example64 tests、tests込みClippy、Python43 tests通過。
新worker全2500 shardをborrowed readerでmergeし最終SHA02cd6475...全一致。
8.19s/16256KiB、証跡`m8-dense-borrowed-merge-v1.json`。session94712もexit0完了。
次はこのbranchのPR/CI/merge、100k読取・強制中断restart。全E2E/軌跡品質は未達。
常駐画像metadataをArcで共有、同一envelopeを再解析/再比較しない。公開owned APIは維持。
測定プロセスなし、空き約10GiB。全goal未達。

更新: PR128 head`7ee56a2`はCI8成功/rust実行中。100k structural readbackを実施し、
全3125 shard/99999 pairs/30599694 matches検証、envelope load1回、8.79s/20524KiB。
1k/10kもpass。validate_files API/validate_snapshot_shards example追加。
チェックサム/raw-inlier関係/画像ID/ペア重複/件数を検証。外部bankと宣言edge hashの検証ではない。
変換converterに明示--resumeを追加。実子プロセスを253 shard時にSIGKILL(-9)して再開、
全2500のdecodedレコードが元と一致、既存253ファイルも全bytes不変。session81409完了。
これは変換batch restartのみ、worker/mapper/full E2E restartではない。
証跡shared-readback-scaling-v1/shared-snapshot-restart-v1。lib20 tests/Clippy通過。
readback session37522も完了。測定稼働なし。次はPR128最終CI/merge、現branchのPR、
native runnerでの共有形式とrestart統合、全抽出E2E/COLMAP品質改善。全goal未達。

更新: PR128はhead`7ee56a2`でCI9成功後`4b51e97a84f3c9d99e70086a8ba20ec9dfe8151c`へmerge、
旧branchはremote/local整理済み。現branchはmainへrebase済み。Python43 testsも通過。

更新: native runnerへ--shared-snapshot-envelope（persistentのみ）を追加。
completed chunkには参照envelopeのfilename/SHAを記録し、resume/merge index検証で
本体と参照の両方を確認。completion log回収も同じ経路。検証call内で共通envelopeは1回hash。
runner tests21件/scripts tests43件通過。欠落/改変/未記録依存・完了記録・既定OFFを検証。
実worker既存2500出力へbinding helper適用、全件同一envelope1個を確認。
証跡`m8-runner-envelope-bindings-v1.json`。新runner経由の実行/強制中断はまだ未検証。
次はPR129最終CI/merge、現branchのPR、新runner実行とresume検証、全E2E/品質改善。
測定プロセスなし。全goal未達。

更新: PR129はhead`ecd3819`でCI9成功後`7fe9e7d885bb0d22f16721ef35cb5aabbd4aef23`へmerge。
旧branchをremote/local整理、現branchをmainへrebase済み。

更新: 実native runner restart試験のv1がmatching前にTypeErrorで停止。
run_match_shardsへのshared_snapshot_envelope引数転送漏れをPR130で修正し、
dispatch回帰test追加。head`e63a1fd`をpush、子branchへrebase済み。最終CI待ち。
v2試験はpass。128実特徴画像/988候補/124 shardをcontrolと中断系で実行。
専用runner+worker process groupを8 shard完了時にSIGKILL(-9)、再開workerは残り116のみ。
全124 shard/統合snapshotがcontrol全bytes一致、既存8ファイル不変、全indexにenvelope binding。
証跡`m8-native-shared-restart-v2.json`、外部root同名。session7249はexit0完了。
v1失敗logも保存、session29544はexit1完了。再poll/再起動不要。
runner tests22通過。次はPR130最終CI/mergeと現branchのPR、全10k runner/E2E・品質改善。
これはmatching+merge restartで、抽出・mapper・全10k restartではない。全goal未達。

更新: PR130修正head`e63a1fd`はCI9成功後`9e270c8250d7f7c0ddd4e52522657cb564aa9b37`へmerge。
旧branchをremote/local整理し、現branchをmainへrebase済み。実restart証跡と更新計画をPR化。
`docs/openloris_m8_m10_plan.md`最新checkpointを09-09へ更新し、既に失敗した品質候補の
再試験を避け、全10k Python runner→連続native DAG→未達軌跡品質→最終tier行列の順を明示。
次は既存dense10k候補/特徴を新artifact rootでrunner prepare/match/merge/completed-resume。
候補はv2、shared snapshotは明示ON。初期入力は凍結済みなので抽出/E2E達成としない。
Python runnerの--prepareはpersistent flag禁止なのでprepareとmatchを別CLIにする。
測定プロセスなし。全goal未達。

### 以前の状態（上記を優先）

PR117は最終head68b1f72のCI9成功後a769432へmerge、branch整理済み。
現branch `diag/m8-pair-admission-recovery`。targeted7追加とrepair19修復・順序を
既存ツールで再生成しsnapshot全bytes一致（証跡targeted7-admission-replay-v1、
repair19-order-replay-v1、repair19-admission-replay-v1）。中間登録結果と修復前snapshotは未再生成。
disk空き89MiBまで低下したため、cargo/rustc稼働なしを確認し、再生成可能な
`target/release/deps`と`target/doc`を `/dev/shm/visloc-regenerable-cache.D87DLG/`へ退避。
移動後空き576MiB。実行バイナリ・入力・計測出力は移動/削除していない。
tmpfsは再起動で消える。次のbuildは依存cache再生成が必要なので容量を再確認する。
全10k再抽出や新バンク一式の追加保存容量は依然不足。全goal未達。

更新: PR116はhead`bef08e15b40e1e020584fc3fd83fc56447964ed6`のCI9件成功後、
`75fea9aba77507862e503fb090395cd5e602bf3d`へsquash merge、旧branch整理済み。
現branchは`feat/m8-adaptive-bank-publication`。
実10kバンク書き出しは全ファイル一致、completed resumeは全件検証・再利用。
completed bankのresumeをSIGKILL後、再resumeも全件一致。新規10k書き出し中断は未検証。
小規模同期SIGKILLテストは部分stagingを残して再開成功、一時ファイルはbank外に保持。
合成100kは書き出し・resume・全内容一致。全試験peak528516KiB、別プロセスresume単独45388KiB。
差の原因未確定。SfM全体の100k/E2E/quality gate完了とは扱わない。
出力は外部`corridor1-1-m8-adaptive-bank-publication-v1`と`synthetic-sift-bank-100k-v1`。
大規模出力前にdisk空きを再確認。既存入力・証跡を削除しない。
次の主要課題はペア追加レシピの再現とnative E2E、未達の10k軌跡品質改善。

- branch: `test/m8-extraction-replay-preflight`。サブエージェントは使わない。
- 抽出session37896は終了コード0で完了。再poll・再起動不要。
  1,250画像/2,500 feature+loci files/585,615,663 bytesが保存済み出力と完全一致。
  51:00.39、peak283,612KiB。他CPU負荷を観測したため単独速度比較には使わない。
- 追加画像選択は692画像の順序を含むJSON全項目が一致。
  結合は元特徴のみを距離判定対象にして1px以内を除外し、追加特徴同士は除去しない。
  全10,000 feature filesの生成hashが一致（変更692、元と同一9,308）。
  base/adaptiveバンクにlociはない。新規バンク書き出し・restartは未実装/未検証。
- 証跡: extraction-shard0-replay-v1、supplement-selection-replay-v1、
  adaptive-bank-replay-audit-v1（`benchmarks/electro/m8-openloris-*.json`）。
- 次: このbranchのPR/最終CI、追加特徴バンクの安全な書き出し、
  ペア追加の入力・順序・モード再現、continuous native E2E。
  10k atlas RMSEのCOLMAP基準未達は未解決。M8–M10全goalは未完了。

## 過去の作業ログ（稼働状態は上記で上書き）

> PR115は最終headc09875aのCI9成功確認後、bbaa344d7d94c099ae8bfcbebba6f2668761d93bへmerge済み。
> 旧branch削除、test/m8-extraction-replay-preflightはmainへrebase済み。
> 抽出session37896は継続中（直近50/1250）。saved binaryなのでgit更新は実行内容へ影響しない。
> 同handleをpoll、観測timeoutで再実行しない。同時build/測定禁止。全goal未達。

> 稼働中: extraction-shard0-replay-v1、exec session37896。1250画像/8threads/3600秒上限。
> 出力 `/home/sasaki/datasets/openloris/corridor1-1-m8-extraction-shard0-replay-v1`、extract.log/time。
> 直近18/1250画像を出力、同handleをpollして継続確認。観測timeoutで再起動しない。
> 期待出力585615663bytes、開始free4184961024bytes。抽出binaryは保存extract-3ae253a。
> 完了後全feature/loci membership/hash比較して証跡JSON更新。1/8shardで全10k/E2E成功ではない。
> 同時build/追加測定は行わない。PR115はheadc09875a CI8成功/rust稼働を最終確認。

> 最新作業branch `test/m8-extraction-replay-preflight`（PR115の子）。PR115最終headc09875aは
> CI8件成功、rust1件実行中のため未merge。親へ追加pushしてCIを再起動しない。
> 容量対応: 再生成可能な`target/debug`約3.3GiBを
> `/dev/shm/visloc-debug-cache.drbnfQ/debug`へ退避。移動完了、disk空き約3.9GiB。
> 入力/証跡/release/保存binary変更なし。tmpfsは再起動で消えるためcache以外を置かない。
> ローカルcargo/rustc稼働なしを確認して移動。以降debug buildは再生成が必要、CARGO_INCREMENTAL=0維持。
> 全特徴4.5GiBの追加保存には依然不足。大規模抽出開始前に容量/出力方針を確定する。全goal未達。

> 最新: `diag/m8-source-command-coverage`。PR114は最終head26f755aのCI9件成功後
> `da96845c943f5889543b5ba8929079a0d2f2a982`へsquash merge済み。再利用は既定OFF。
> 親branch整理済み、現branchはmainへrebase。サブエージェントは使わない。
> source21実行のログSHAは全一致。兄弟time記録にコマンドがあるのは5件。
> 証跡 `m8-openloris-source-command-coverage-v1.json`。他の場所の記録は未探索。
> 記録済み650開始/500frame窓をrig-0466499で新規出力先へ再実行し、過去3モデルbytes一致。
> wall36.60秒、RSS340100KiB。証跡 `m8-openloris-source-replay-650-v1.json`。
> 残り20実行とfrontend/atlas alignment/integrationの一体的再現は未完。測定プロセスなし。
> 次の再実行は保存コマンドのある1950/3200/250/4200窓から、毎回新しい出力先を使う。
> 過去の不明flagsを推定した場合は過去再現と称さず、新規仕様として分ける。全goal未達。
> 1950/3200開始500frameも再実行完了、各3モデルbytes過去一致。30.25秒/22.48秒。
> RSS340220/340224KiB。証跡source-replay-1950/3200-v1。3/21実行を確認済み。
> 次は記録のある250開始750frameと4200開始500frame。測定稼働なし。
> 上記2件も完了、全モデルbytes過去一致。250:51.39秒/340160KiB、4200:43.92秒/340228KiB。
> 4200は保存コマンドのVISLOC_DEFERRED_DEBUG=1保持。5/21実行を再現、測定稼働なし。
> 残り16は調査済み兄弟timeファイルにコマンドなし。推定flagsを過去再現と称さない。
> 追加M8スクリプト探索では起動記録未発見。新仕様として650窓flagsの開始位置だけ0へ変更。
> source0全3モデルbytes過去一致、79.58秒/340288KiB。source-spec-0-v1.json参照。
> 過去argv復元ではない。計6ソース出力再生成（うち5件は保存argv由来）。測定稼働なし。
> source-execution-inventory-v1.jsonで全21実行/23nodeの入力start/countをログ/timeから確定。
> atlas window_startと入力startは別（node21は500表記だが入力250開始750frame）。
> 12/13と18/19は同一実行の別成分。残り15実行はこのinventoryで新仕様を組み全成分検証へ。
> 新仕様250開始500frameも過去3モデルbytes一致、38.74秒/340220KiB。
> source-spec-250x500-v1.json。計7/21実行出力を再生成、残り14、測定稼働なし。
> 新仕様3000/4250の複数成分窓も全6files過去一致。23.27秒/29.82秒、RSS340168/340092KiB。
> node12/13、18/19それぞれ1実行として計上。計9/21実行・11/23nodeを再生成。
> source-spec-3000/4250-v1.json。残り12実行、測定稼働なし。
> source-spec-750-v1も全モデル過去一致、39.63秒/340224KiB。計10/21実行、12/23node。
> 残り11実行。次は1000/1250/1500/2000/2250wide750/2500/2750/3500/3750/4000/4500。
> 入力rangeはinventoryで確認。測定稼働なし。品質/E2E未達。
> source-spec-1000/1250-v1も各3モデル過去一致、35.01秒/28.16秒。
> 計12/21実行・14/23node。残り1500/2000/2250wide750/2500/2750/3500/3750/4000/4500。
> 測定稼働なし。過去argvでなく新共通仕様の出力一致。全goal未達。
> 1500/2000窓も過去bytes一致、27.33秒/27.68秒。2000はcomponent-001のみ（旧/新とも）。
> 計14/21実行・16/23node。残り2250wide750/2500/2750/3500/3750/4000/4500。測定稼働なし。
> 2250x750も過去モデル一致、51.56秒/340088KiB。計15/21実行・17/23node。
> 残り2500/2750/3500/3750/4000/4500。source-spec-2250x750-v1.json。測定稼働なし。
> 2500窓も過去モデル一致、37.28秒/340056KiB。計16/21実行・18/23node。
> 残り2750/3500/3750/4000/4500。source-spec-2500-v1.json。測定稼働なし。
> 2750/3500も全モデル過去一致。36.87秒/32.05秒、RSS339964/340184KiB。
> 計18/21実行・20/23node。残り3750/4000/4500。測定稼働なし。全goal未達。
> 3750/4000/4500も全モデル過去一致。21実行すべての出力再生成完了（保存argv5、新仕様16）。
> 各source-replay/source-spec証跡の計21ファイルのold/new hash map一致を集計確認。
> 次は再生成パスをatlas nodesへ束ねる。offset/component membership保持、alignment/integration再実行。
> 孤立run時間合計を連続E2Eとしない。品質/E2E/restart/100k未達。測定稼働なし。
> regenerated-nodes-v1.tsv/json作成。23nodeのID/offset/順序/成分保持、再生成先へパスのみ置換。
> 計69モデルファイルを既存atlas sourceと再照合し全SHA一致。alignment/integrationは未実行。
> nodes元SHA69b3f668...。空き約962MiB、新規大出力前に確認。測定稼働なし。
> regenerated-atlas-v1/stitch-cfe11c6でL/traversal/newest再実行、2成分imagesが既存atlas完全一致。
> 1.21秒/36924KiB。証跡m8-openloris-regenerated-atlas-v1.json。次は再生成atlasでintegration/BA。
> 出力root `/home/sasaki/datasets/openloris/corridor1-1-m8-regenerated-atlas-v1/atlas`。測定稼働なし。
> 再生成nodes+atlasからtail/main integration+filtered BA完了。全12pre/postモデルfiles過去一致。
> tail11.03秒/78120KiB、main146.73秒/525196KiB。regenerated-integration-v1.json参照。
> 保存integrate-b23a6a4使用、trace ON。frontend除外の段階再現で、連続E2Eではない。
> 次は連続pipeline計測と品質課題。既存RMSE未達不変。同じsource再実行の繰返し不要。測定稼働なし。
> frontend-ledger-audit-v1.json追加。targeted7の448行は1worker53.59秒、adaptive32の2188行も1worker272.62秒。
> worker時間をshard数倍しない。両summaryは特徴抽出除外、candidate/mapping nullでE2E未完。
> 次は抽出/選択merge/候補/overlay生成の依存を連続実行仕様へ。空き約672MiBに注意。
> extraction-shard-audit-v1.json: dense256x2抽出8コマンド記録、10000画像名重複なし/全target存在。
> 内容hash/元binary/同時実行メモリは未証明。extract再実行なし。8wall単純合算/最大RSSでE2Eとしない。
> extraction-content-audit-v1で10000画像/2999280260bytesを全SHA+size照合、欠落/余分/不一致0。
> manifest SHAa3458a3a...はCOLMAP10k frozen契約と一致。入力画像同一性のみ確定、抽出出力/E2E未達。
> extraction-resume-pilot-v1: cam1_000000の抽出feature/loci過去bytes一致（377kp）。resume初期化後hit確認。
> 保存extract-3ae253a SHA8cfa9c53...、image-io build。1画像pilotのみ、10k/restart全体成功ではない。
> full特徴4.5GiBに対し空き669MiB時点。全抽出未着手、既存証跡保持。測定稼働なし。
> extraction-resume-corrupt-v1: private copyのfeature hash記録を0へ変更。resumeは無効判定→再抽出。
> feature/loci/sidecar全3filesが元pilotと完全一致。元証跡変更なし。1画像破損復旧のみ、全restart未達。

> 最新: `perf/m8-bounded-retry-reuse`。PR111/112/113はCI確認後merge済み。
> mainは `17277f9822a78e410bc2ae39de6915504fbccdf3`。サブエージェントは使用しない。
> BA再試行の単一所有normal system再利用を既定OFFで実装。CLIは
> `--ba-reuse-rejected-pose-diagonal`。source0466499の保存binaryで1kを4回完走。
> 全runのモデル3ファイルは過去Legacyとbytes一致。wall OFF3.73/3.94秒、ON3.44/3.48秒。
> RSS OFF81944/81968KiB、ON82008/82200KiB。省メモリ成功とはいわない。
> 証跡 `m8-openloris-1000-retry-reuse-v1.json`、保存root `corridor1-1-m8-retry-reuse-1k-v1`。
> 明示ON native固定状態テスト成功。reject→accept→reject専用coverage、上位tier、
> atlas CLIへの明示設定伝達は未完。実測はfrontend除外でE2Eではない。全goal未達。
> 以下は過去の時点の記録。最新状態はgitと証跡を優先する。

> 同じ保存binaryで独立2.5k control/candidate-a/b/control-repeatもexit0、
> 全モデル3ファイルが過去Legacy対照と完全一致。wall OFF38.50/37.69秒、ON35.63/36.16秒。
> RSS OFF412748/413244KiB、ON412980/413008KiB。省メモリ/10k/E2E優越は未証明。
> 証跡 `m8-openloris-2500-retry-reuse-v1.json`。測定プロセスはすべて終了。
> 合成parityテストに連続「棄却→受理→再棄却」の必須assertを追加し通過。
> ON/OFFの全反復統計・最終状態一致も通過。次は5k/10k。実装既定OFFを維持。
> 派生5kの初回control/candidate-aはexit0、過去Legacyと全モデルbytes一致。
> wall54.58/53.07秒、RSS722484/722808KiB。反復candidate-b/control-repeatは未実行。
> `m8-openloris-5000-retry-reuse-v1.json`に完了/未実行コマンドを区別して記録。
> 現時点で実行中の測定なし。5k独立frontend/E2Eは未検証。
> 5k反復も完了、全4run過去Legacyモデルbytes一致。OFF54.58/74.53秒、ON53.07/64.05秒。
> 時間範囲が重なり変動大、安定高速化は未証明。RSSほぼ横ばい。次は10k、測定稼働なし。
> 完全10k初回control/candidate-aもexit0、全6モデルファイル過去Legacyと完全一致。
> OFF102.32秒/RSS1274584KiB、ON112.67秒/RSS1274380KiB。候補が遅い、反復未実行。
> `m8-openloris-10000-retry-reuse-v1.json`に全コマンド/結果。candidate-b/control-repeatが次。
> 保存binaryは引き続きrig-0466499。測定稼働なし。9936画像基準の品質未達も不変。
> 10k反復完了: OFF102.32/99.99秒、ON112.67/113.79秒。全4run全6モデルbytes過去一致。
> 候補2回とも対照より遅い。既定OFF維持、同条件の追加反復は不要。品質/E2E未達。
> 証跡と計画更新済み。測定プロセスなし。次はPR整理と限定的な遅延原因診断/品質課題。

> 最新: `perf/m8-stream-connectivity-unions`。PR108/109/110は最終headのCI9件成功後merge済み。
> main最新merge `14d9b50661e2df6d8eb0c1743e815adfa6d62f9d`（PR110）。
> atlas内訳計測で主成分164.04秒、solver87.06秒、連結性27.33秒、出力6ファイル過去一致。
> `m8-openloris-atlas-inner-timing-v1.json`参照。精度改善/mapper/E2E成功ではない。
> 連結性検査を画像・frameフラグと逐次unionへ変更。旧実装oracle64変種一致、全52テスト/clippy成功。
> 保存binary `corridor1-1-m8-compact-connectivity-v1/integrate-2c024cf` SHA256
> `04314afeb4a280f411a5ad48e8a1f97ef8a8d0b2f17b68099278b0d68f720a16`。
> tailはexit0、全6ファイル過去一致。連結性0.10246秒対前回0.31470秒、RSS77916KiB。
> mainもexit0・全6ファイル過去一致。連結性9.55577秒対27.32571秒、総151.46秒対164.04秒。
> RSS524744KiB対525228KiB。候補main-repeatもexit0・全6ファイル過去一致。
> 全反復完了。main候補連結性9.56/10.47秒対control27.33/31.70秒、総151.46/158.84対164.04/185.19秒。
> tailも全6ファイル一致、候補連結性0.102/0.106秒対control0.315/0.240秒。
> tail全体時間には変動があり一律短縮を主張しない。全モデル過去一致、全goal精度未達は変わらない。
> Draft PR111作成済み。古いplan/m8-fixed-boundary-baは内容がmainに保持されていることを確認して削除。
> 正確なコマンドを `m8-openloris-compact-connectivity-v1.json` のrepeat_commandsに凍結済み。
> 同時build/測定禁止、既存出力上書き禁止。全goal未達。
> 容量対応で再生成可能な`target/debug/incremental`約4.9GiBのみ
> `/dev/shm/visloc-atlas-build-cache.RcnQnI/incremental`へ退避。入力/証跡は保持。
> 以降buildは`CARGO_INCREMENTAL=0`。空き約3.4GiB、tmpfs退避は再起動で消えるcacheのみ。

> 最新: `feat/m8-covisibility-local-ba`。PR107は最終head6914606のCI9件成功後
> `f7ab2ca`へmerge。共視選択を既定OFFの`--local-ba-covisibility`として追加。
> rig_sfm39テスト、CLI check、lib clippy通過。release source9d7868b、57.05秒。
> 保存binary `corridor1-1-m8-bounded-native-v1/rig-9d7868b` SHA256
> `3dc9d3f9302c73bc58a218cd3ca0239450e877fad0bb9a4998ca596bc2dfcc79`。
> `covisibility-1k/control`はLegacy全3ファイル一致。candidate-a/bも全3ファイル一致。
> 1000画像/500frameすべて支持あり、正深度/固定rig/双方向参照正常、連結成分1。
> ただしRMSE0.025216mと再投影0.703792pxは既存gateを超えるため不採用。
> 証跡 `m8-openloris-1000-covisibility-v1.json`。このまま上位tierへ進めない。
> 全goal未達。支持欠落やrig破損が今回の品質低下を説明する証拠はない。
> strong-boundary候補bをaへ全6ファイルcmp後hardlink化済み、パス/bytes保持。
> 空き約119MiB。新規run名を使い、既存hardlinkモデルは上書き禁止。

> 現branch `test/m8-strong-boundary-10k`。旧strong30/deferred8/direct2/s1/rot5条件をtimeログから復元。
> manifest/snapshot/visloc pose-prior2ファイルSHA確認済み。契約 `m8-openloris-strong-boundary-contract-v1.json`。
> 同binary `rig-7d99e07` の `strong-boundary-10k/control` はexit0、過去9998画像モデル全成分bytes一致。
> structure/deferred分割ログも過去先頭行と一致。candidate-aはexit0、9998画像を維持。
> ただしRMSE/p95 0.775847/1.313240mは対照0.637290/1.170177mより悪化、不採用。
> mapper283.672706秒、VmHWM1480176KiB。主成分2画像/後半28画像は支持なし、後半11frame支持なし。
> candidate-bもexit0、両成分の3モデルファイルSHAがcandidate-aと完全一致。
> mapper260.671910秒、RSS1481112KiB。証跡 `m8-openloris-strong-boundary-result-v1.json`。
> 対照にも同じ30画像/11frameの支持欠落あり。boundaryによる新規欠落ではない。
> 精度と時間が悪化したため不採用。次は局所BA窓の共視ベース選択を単独要因として設計する。
> 360秒上限・単一thread、完走後旧モデルbytes確認→boundary flagのみ追加したcandidate-a/b。
> 事前モデル生成コストは別途必要、native E2Eと混同しない。入力やthreshold変更なし。
> 容量確保のため直前10k boundary-bモデルをbytes一致したboundary-aへhardlink化。
> パス/内容保持、既存出力を上書きしない。空き約395MB（今回対照開始前）。

> 10k mapper-only対照 `tier-10000-boundary/boundary-control` はexit0、過去Legacy全成分bytes一致。
> 9936画像/4968frame、RMSE/p95 0.878064/1.485271m、mapper117.426119秒、VmHWM1274076KiB。
> boundary-aはexit0、9936画像/4968frame、RMSE/p95 0.858292/1.473066mでCOLMAP基準未達。
> 再投影0.791517px、mapper180.290904秒、VmHWM1276928KiB。支持/正深度/固定rig正常。
> 最終track成分は各model内で連結（frame0..4491と4524..4999）、欠落4492..4523の32frame。
> boundary-bもexit0、全成分候補bytes一致。mapper180.291/163.331秒、RSS1276928/1276600KiB。
> 証跡 `m8-openloris-10000-fixed-boundary-v1.json`。登録/軌跡gate未達、全goal未達。
> PR105は最終head `f961eaf` CI9成功後 `9fef464` にmerge。現branch `test/m8-fixed-boundary-10k`。
> 過去 `m8-openloris-10k-first-divergence.json` にstrong structure/deferred weak direct bridgeで
> 9998画像登録（RMSE0.637290m）の証跡あり。入力/コマンドを復元確認してから次のA/Bを決める。
> 同じ `rig-7d99e07`/完全10k入力、frame sliceなし、各run上限360秒、単一thread。
> 入力5000frame/10000画像/64862pairs/7551021 capped matches、初期VmHWM547620KiB。
> 完走後対照を採点しboundary-a/bへ。native E2Eではなく、登録不足も隠さず記録する。

> PR105 https://github.com/rsasaki0109/visloc-rs/pull/105 はhead `aa3938c`、CI34200559679実行中。
> 同じbinary `rig-7d99e07` の独立dense ANN2.5k対照を開始。
> `tier-2500-ann/boundary-control` はexit0・Legacy bytes一致、mapper36.262899秒。
> boundary-aはexit0、全2500画像、RMSE/p95 0.067853/0.134449m、再投影0.778608pxで改善。
> 支持/正深度/固定rig正常、track成分1250frame一つ。mapper50.634806秒で対照より遅い。
> boundary-bもexit0・反復bytes一致。mapper50.635/50.016秒対control36.263秒。
> 証跡 `m8-openloris-2500-fixed-boundary-v1.json`。次は10k。
> 容量確保: 5k legacyと完全一致したpolicy-a/b、support-debug、ba-motion-debug、
> ba-connectivity-debug、anchored-control、boundary-controlの3モデルファイルをhardlink化。
> パス/bytes不変、入力データ変更なし。これらの既存出力は上書きせず新規run名を使用する。
> 全goal未達。ディスク空き約327MB（対照開始前）、10k前に再確認する。

> PR104はhead `109705c` のCI9成功後 `1f98c5c` にmerge（成分固定は既定OFF維持）。
> 現branch `feat/m8-fixed-boundary-ba`、build source `7d99e07dbdbc95bc72d2e28805a75e51dbcd7270`。
> `--ba-fixed-boundary-observations` を既定OFF追加。窓内usable観測を持つtrackのみ、
> 窓外usable観測を固定body poseとしてBAへ含める。成分固定と組み合わせずLegacy対照で測る。
> 窓内1観測+窓外1観測の採用、固定pose/観測不変、窓外のみtrack除外テスト通過。
> clippy/CLI check通過。release45.41秒、保存binary `rig-7d99e07`。
> SHA256 `d87194e92b472b9d74fd8d2e43d41faad5766f330afc3a6e9155c41816277d72`。
> 5k `tier-5000-slice/boundary-control` はexit0、Legacy bytes一致、mapper53.399406秒。
> boundary-aはexit0、全5000画像、RMSE/p95 0.164335/0.271032m、再投影0.819133pxで対照より改善。
> track成分2500frame一つ、支持/正深度/固定rig正常。mapper83.156035秒で対照53.399406秒より遅い。
> boundary-bもexit0、候補反復bytes一致。mapper83.156/91.705秒、RSS724532/724796KiB。
> 成分固定OFF。証跡 `m8-openloris-5000-fixed-boundary-v1.json`。
> 1k control/candidate-a/bはexit0、control Legacy bytes一致、候補反復bytes一致。
> RMSE/p95 0.022320/0.035799m、再投影0.627506px、全1000画像・幾何正常で既存品質gate通過。
> mapper5.575/5.557秒対control4.099秒。証跡 `m8-openloris-1000-fixed-boundary-v1.json`。
> 次は独立dense ANN2.5kと10k、native E2Eは依然未検証。
> 反復確認後、独立tier非回帰/10k品質へ。5kだけで全goal達成やCOLMAP優越を主張しない。
> 実ソルバーのpose_indexはfixed pose除外をコード確認。追加観測/固定poseのRSSは未測定。
> [契約](openloris_fixed_boundary_ba.md)。5k残存singleton1616は二乗誤差2.134%のみ。

> PR103はhead `9f7cfeb` CI9成功後 `a82a4bb` にmerge、旧branch整理済み。
> 現branch `feat/m8-ba-component-anchors`。既定OFFの
> `--ba-anchor-disconnected-components` は既存固定poseのない各BA成分の最小frame IDを固定。
> 選択順不変/既存anchor保持/単独poseテストと有効時native固定状態/rollbackテスト通過。
> clippy/CLI check通過。build `36eb02159f88211fd2e57db75a43d9cadbb494ca`、release52.95秒。
> binary SHA256 `779c6295f5cf62503790da530b9da3c3a1afcf3735e903fed9799501cb49db2b`。
> 5k対照 `tier-5000-slice/anchored-control` はexit0、Legacy bytes一致、mapper53.417580秒。
> 候補component-aはexit0、全5000画像。RMSE0.261284/最大1.773457 mへ改善するが、
> p950.490866 m/再投影0.837364 pxは対照より悪化。全品質gate未達、既定化なし。
> track成分2499/1、孤立frame1616。支持/正深度/固定rig正常。component-bもexit0、反復bytes一致。
> 各追加anchor39。mapper53.641/62.541秒対control53.418秒で高速化主張なし。
> 証跡 `m8-openloris-5000-component-anchors-v1.json`。次は品質の残差要因を調べる。
> [設計と検証契約](openloris_ba_component_anchors.md)。既存mono scale/弱い幾何は別問題。

> 現branch `diag/m8-ba-anchor-connectivity`、親PR102はCI実行中。
> build `4fbb952b600d36bc312c04f97c076f0a3ee1c7f8` に既定OFFの
> `VISLOC_SFM_TRACE_BA_CONNECTIVITY` を追加。実際のBA採用観測のlandmark-starで
> 固定poseへの到達性を判定。密なpose cliqueなし、O(観測+点+pose)診断状態。
> 接続単体テストと有効状態の固定/rollbackテスト、clippy通過。release42.71秒。
> 5k `tier-5000-slice/ba-connectivity-debug` はexit0、Legacyモデルbytes一致。
> binary SHA256 `a2b1523fccf956ca912ba051043e186dc0f7a95e2ce70a041cfb2851079b2fb0`。
> 全20000 pose/BA記録中306件が全固定poseから未接続。変位上位5件は全て未接続。
> 1070/1072のanchor1414のBAも未接続。次は独立成分のgauge固定をdefault-off検証。
> 1416は接続済みでも変位するため十分条件ではない。品質変更はまだない。
> 証跡 `m8-openloris-5000-ba-connectivity-v1.json`。

> PR101は最終head `4ae1a8f` のCI9成功後、`a551e5a` にsquash merge・旧branch整理済み。
> 現branch `diag/m8-ba-pose-motion`、build `49f3021`。既定OFFの
> `VISLOC_SFM_TRACE_BA_POSE_MOTION` は各BAの固定anchor相対距離を前後で記録する。
> 有効状態のbounded policy固定状態/rollbackテストとclippy通過、release57.19秒。
> 5k `tier-5000-slice/ba-motion-debug` はexit0、Legacyモデルbytes一致。
> 1070/1072はanchor1414との距離が同一BA内で約8.9m増加。固定pose距離変化0。
> 次は登録順local BA窓内の観測連結性と固定anchorへの接続を診断する。
> 証跡 `m8-openloris-5000-ba-motion-v1.json`。共通gauge移動だけでは説明できない。
> binary SHA256 `296a740951a94b486daa4ca4c7a2eb9dfc4de0b6b780e0bfbd9dc6092ea88e6d`。
> 前のsupport-debugはexit0・Legacy bytes一致。登録→最終の変位は共通gauge変化も含むので
> BA原因の断定不可。証跡 `m8-openloris-5000-registration-diagnostic-v1.json`。

> 5k外れ値診断: 孤立5frameは同一全体Sim(3)下で二乗誤差69.287%を占める。
> 最大8画像は1068/1070/1071/1072、誤差8.21〜10.09 m。
> 主成分内1416も5.577 mなので孤立解消だけで品質達成とは言えない。
> GTは診断のみ、モデル削除/再整列/閾値変更なし。既存repair経路の条件と
> 登録時→最終trackの支持喪失を次に確認する。証跡 `m8-openloris-5000-slice-outliers-v1.json`。

> 5k slice検証: 全5000画像登録、Legacy/policy2反復はモデルbytes一致。
> 各420 BAは直接法、QR0。GT RMSE/p95 0.477216/0.478983 m、最大10.094671 m。
> 幾何監査は支持点/正深度/固定rig正常だがtrack成分2495/2/2/1 frame。
> 主成分外は791、1068/1071、1070/1072。次は軌跡外れ値との関連を診断。
> 10k由来mapper-only入力で独立5k native E2Eではない。全goal未達。
> 証跡 `m8-openloris-bounded-native-5000-slice-v1.json`。

> Tier再検証: PR #100はCI9項目通過後 `d72bfdd` へmerge、旧branch整理済み。
> 現branch `test/m8-rig-ann-tier-validation`。rig-aware dense ANN 2.5kでLegacy/policy2反復は過去モデルbytes一致。
> 全2500画像/1250frame、独立GT RMSE/p95 0.133927/0.237364 m。各211 BAはすべて直接法、QR0。
> mapper36.36/36.55 s対Legacy35.53 s、速度改善主張なし。証跡 `m8-openloris-bounded-native-2500-ann-v1.json`。
> 次は5k入力契約の確認。独立したdense5kスナップショットはまだ見つからず、10kのsliceを使う場合は
> mapper-only派生入力と明記し、5k native E2Eとは区別する。全goal未達。

> 境界PnP: PR #99はCI9項目通過後 `28bf0e2` へmerge、旧branch整理済み。
> 現branch `diag/m8-boundary-pnp`。build `015ab1b` の診断モデルはLegacy bytes一致。
> frame612/613は25/29対応・2sensorで推定失敗。DLT候補26/11、central report各2は得られるが、
> pooled inliersは両方5で必要6未満。閾値変更なし。各sensor内とrig全体のinlier比較ログを追加中。
> 証跡 `m8-openloris-2500-pnp-hypotheses-v1.json`。次は追加ログのnative parity確認。

> bounded policy 1k: build `b011cde`、Legacyとpolicy2反復はchampion bytes一致。
> policyは各86回すべて直接法、QR0回。mapper4.28/4.51 s対Legacy4.42 s、速度改善主張なし。
> 証跡 `m8-openloris-bounded-native-1k-v1.json`。次tierでは選択経路のcoverageを必ず確認。
> PR #98はCI9項目通過後 `66ec5c2` へmerge、旧branch整理済み。現在のpolicy branchは未PR。

> 次候補: `feat/m8-bounded-native-ba` に `bounded-direct64-qr` をdefault-off実装。
> 解く前に64可変pose以下は従来直接法、65以上はQRを選択。失敗後fallbackなし。
> 共通適用条件、境界/固定状態/rollbackテストとclippy通過。native実測はまだ。
> [設計と検証契約](openloris_bounded_native_ba.md)。小窓合格だけで大規模QR品質を合格にしない。
> 親PR #98はCI実行中。大規模品質・速度・省メモリの全goalは未達。

> 同一状態診断: PR #97はCI9項目通過後 `ee3764e` へmerge、旧branch整理済み。
> 現branch `diag/m8-qr-shared-state`。build `0cd9ec4` のLegacy shadowで全688状態を比較。
> 出力はchampion bytes一致。QR成功571、失敗117（MaxIterations114/ResidualCheck3）。
> 成功時の正規化pose/point差は最大3.083e-6/3.537e-6。ただし同一LM採否は未証明。
> 診断上限の境界テスト含むQR8テスト/clippy通過。PR/CI/mergeは未完了。
> 別々の軌跡の棄却件数だけでは原因を決めない。実データ証跡 `m8-openloris-qr-shared-state-v1.json`。

> QR follow-up: PR #96はCI9項目通過後 `bcbe2d5` へmerge、旧branch整理済み。
> 現branch `perf/m8-qr-validation-scans`、重複全pose検査を除去 (`4072e05`, build `c7c3d8b`)。
> Legacy/QRモデルは変更前bytes一致。QR12.59/13.12 s対旧QR対照19.35 s、Legacy4.60 s。
> 速度は改善したが品質未達・Legacyより遅い。証跡 `m8-openloris-qr-validation-v1.json`。
> このfollow-upのPR/CI/mergeは未完了、全体goalも未達。サブエージェントは使用しない。

> QR核の実装開始（2026-09-08）: PR #95はCI9項目通過後 `2f383f2` にmerge、旧branch整理済み。
> 現branch `feat/m8-implicit-landmark-qr`。3本のHouseholderベクトルによる点消去核をtest-only実装。
> 疎pose行の作用/随伴/右辺/normal action/点逆代入も追加、5テスト/clippy通過。
> native接続・実データ・性能改善は未検証。
> [設計と次の実装手順](openloris_implicit_landmark_qr.md)。全Q/密なトラックJacobianは作らない。
> rig線形化adapter/複数点operatorもtest-only接続済み。既存Jacobian/Huber重みを再利用し、
> 固定pose/rotation/点、回転付きsensor、None/Huber6と減衰0.5/100で旧normal/direct stepと一致。
> test PCG接続・poseごと6×6前処理も実装。6関連テスト通過、真の残差/直接解/反復一致を確認。
> 前処理はfull normal systemを二重保持しないが、normal-form減算の数値限界は残る。
> 共通LMループへtest-only接続済み。QR時は旧normal assemblyをスキップ。
> 3非線形反復のcost低下/反復一致/固定条件と、強制PCG失敗時の状態保持・lambda増加を検証。
> production/native selector `--ba-backend matrix-free-qr` をdefault-off追加。
> QR7テスト（native固定状態/rollback含む）通過。実データ品質・性能は未証明。
> release `f62f09b`、同一binary Legacyはchampion bytes一致、QR2反復もbytes一致。
> QRは再投影0.660774 pxで合格だがRMSE/p95 0.044781/0.064069 mで失敗。
> mapper18.80/19.88 sはLegacy4.06 sより遅い。51/688線形失敗、accepted166。
> evidence `m8-openloris-native-qr-v1.json`保存。大規模展開/既定化なし、PR/CI/merge未完了。
> サブエージェントは使用しない。goal全体は未達。

> Native window診断（2026-09-08）: PR #94はCI9項目通過後 `539ae4f` にmerge、旧branch整理済み。
> `db9b7f0`のscalar contextで両debugモデルはPR94 bytes一致。strict初回失敗40pose、cluster8は30pose。
> cluster8は残差チェック530/反復上限57（strict296/293）。単なる小窓→大窓のメモリ問題ではない。
> 次は[事前契約](openloris_native_rig_matrix_free.md#native-window-context-and-next-bounded-decision)に沿い
> cluster8+既存restart1を `2cc2bd2` で実装、総PCG128内・同じ残差基準で検証済み。
> 2反復bytes一致、507restart/最大128反復。ただしRMSE/p95 0.172115/0.315988 m、raw mean0.812699 pxで失敗。
> 残差・cluster-size・restartの追加sweepなし、既定化/10k展開なし。証跡保存後にPR/CI/mergeが必要。
> 現branch `feat/m8-native-ba-context`。サブエージェントは使用しない。

> 直接作業へ移行（2026-09-08）: ユーザー希望により以後サブエージェントへ依頼しない。
> PR #93はCI9項目通過後 `3136b20` にmerge、旧branch整理済み。
> 設定不変のdebug replayで589線形失敗を分類: ResidualCheckFailed296、MaxIterations293。
> debug出力モデルは通常MF出力と全bytes一致。残差比はvariant別fieldを区別して記録。
> [診断とメモリ制約](openloris_native_rig_matrix_free.md#fixed-profile-linear-failure-classification)。
> 全pose-pair graphを作るIC(0)案は最悪O(N²)のため採らない。
> 固定上限8 poseのcluster Jacobiをdefault-off実装、新規5テスト通過。未採用、許容誤差/品質基準は維持。
> 現branch `docs/m8-native-linear-failure-diagnosis`。release `8ba580c`、同一binary Legacy/strict対照bytes一致。
> cluster8反復もbytes一致だがRMSE/p95 0.223738/0.432400 m、raw mean0.760114 pxで不合格。
> 線形失敗587/688、accepted59。clusterサイズsweep/10k展開/既定化なし。PR/CI/merge未完了。

> Native rig matrix-free検証（2026-09-08）: Luna Max実装 `0298e0c`、
> root独立rig29/API15テスト通過。同一binaryのLegacyは既存championと全モデルbytes一致。
> MFの2反復もbytes一致、1000画像/500frame支持・校正・正深度・xy順序は保持。
> ただしRMSE/p95 **0.140219/0.234580 m**、raw mean **0.725011 px**で全品質基準失敗。
> 688反復中589 PCG失敗、86呼び出し中52はaccepted stepなし。既定化・10k展開なし。
> [結果と次の診断方針](openloris_native_rig_matrix_free.md)、
> [証跡](../benchmarks/electro/m8-openloris-native-rig-matrix-free-v1.json)。
> 現branch `feat/m8-native-rig-matrix-free` のPR/CI/mergeは未完了。READMEの性能主張は変更なし。

> 続行（2026-09-08）: PR #92は最終head `8cda545`のCI9項目
> （run `34184156416`）通過後、`ece2b71`へmerge。旧branch整理済み。
> 現在 `feat/m8-native-rig-matrix-free`。次は[事前契約](openloris_native_rig_matrix_free.md)
> に沿って、既存rig mapperの共通BA呼び出しに明示的matrix-free選択を接続します。
> monocular経路への置き換えや別の局所BAパラメータ診断ではありません。
> 小窓では高速化が未証明のため、従来設定と同じnative入力から品質・時間を比較します。
> Huber6等のcaller設定は維持し、全pose固定時は明示的zero-pose landmark-only経路。
> 大きなSchurへの暗黙fallbackなし。実装・native A/Bはまだ未完了です。

> Frozen Huber BA完了（2026-09-08）: Luna Max `c026f40`をroot独立28テストと
> clippy後にrelease認証。4本ともexit0、None対照はPR #91のモデル/trace一致、
> Huber反復も3ファイル/85行一致。全identity/支持/校正/anchor/正深度を保持。
> Huberは22.00/22.43 s、peak RSS85,156/85,108 KiB。GT308画像はRMSE/p95
> **0.028822/0.044018 m**でlegacy基準を満たさず不採用（adaptive Noneよりは改善）。
> 平均再投影0.673262 pxでも軌跡改善ゲートは閉じません。尺度sweep/atlas/既定化なし。
> [結果](openloris_frozen_huber_ba.md)、
> [全証跡](../benchmarks/electro/m8-openloris-frozen-huber-ba-v1.json)。次はこのPRのCI/merge。
> Luna Maxは次のnative統合候補をread-only監査中。局所BAのパラメータ診断は繰り返さず、
> native速度・メモリ改善へ進める具体的scopeを確認します。全体goalは未完了です。

> 続行（2026-09-08）: PR #91はhead `b2a012f`のCI9項目
> （run `34182134711`）通過後、`e169527`へsquash merge。旧branch整理済み。
> 現在は `feat/m8-frozen-huber-ba`。次の[事前契約](openloris_frozen_huber_ba.md)を
> `9de6fe8`で固定し、Luna Maxがdriver-only Huber-3 opt-inを実装中。
> 同一130,900観測・全XYZ可変のrobust BAは、過去の観測集合を変える
> robust triangulation/Huber-1実験とは区別します。尺度sweep・GT調整なし。
> rootの初期モデル独立集計で3 px超は1,984観測（約1.52%）、最小重み0.750081。
> 影響範囲は限定的で改善は未証明。実solveは未実施、全体goalは未完了です。

> 固定39点A/B完了（2026-09-08）: `5a8f135`をroot独立24driverテスト＋既存rig固定点テスト後に
> release build。legacy/adaptive対照は過去モデル・数値trace一致。fixed39反復も3ファイル/83行一致。
> 全支持/identity/校正/anchor/正深度と固定39 XYZ bit一致を維持したが、post-only GT308画像は
> RMSE/p95 **0.030217/0.045693 m**でlegacy/adaptive双方より悪化。非昇格・atlas未実行。
> 点移動最大32.592 mへ縮小しても軌跡は改善せず、このhard XYZ固定案は棄却。
> 候補26.26/27.09 s、peak RSS84,596/84,884 KiB。native mapper/E2E時間ではありません。
> [結果](openloris_weak_angle_fixed_landmarks.md)、
> [全4本の証跡](../benchmarks/electro/m8-openloris-weak-angle-fixed-landmarks-v1.json)。
> 次はCI/PR/merge確認。Luna Maxは既存robust BA実験の有無を一次資料・repo証跡から調査中
> （編集/solveなし）。固定点数や閾値のsweepはしません。全体goalは未完了です。

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
