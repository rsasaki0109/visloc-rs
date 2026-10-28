#!/usr/bin/env bash
# IMU metric scale (scripts/insta360_imu_scale.py) on the OverMaps360_50
# scenes, against their ARKit scale (docs/insta360_imu_scale.md).
#
#   HF_TOKEN=... scripts/overmaps360_imu_scale.sh WORK_DIR [mapping_id ...]
#
# The dataset (huggingface.co/datasets/OverTheReality/OverMaps360_50,
# CC BY-NC 4.0) is gated: accept its terms on Hugging Face, then pass a read
# token in HF_TOKEN. Per scene: download the perspective model's images.bin
# (0.2-0.3 GB), insta360_imu.csv (about 70 MB), the smartphone poses and
# the phone's GPS log (for a GPS reference scale), write
# WORK_DIR/res/<id>.json and WORK_DIR/log/<id>.txt, then delete the
# downloads. Scenes that already have a result are skipped. Without ids,
# every scene in the dataset manifest. Ends with the summary over
# WORK_DIR/res/*.json. Needs curl and python3 with numpy and scipy.
set -euo pipefail
W=$1
shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BASE=https://huggingface.co/datasets/OverTheReality/OverMaps360_50/resolve/main
: "${HF_TOKEN:?set HF_TOKEN to a Hugging Face read token (the dataset is gated)}"
mkdir -p "$W/res" "$W/log" "$W/data"
fetch() {
  curl -fsSL --retry 5 --retry-all-errors -H "Authorization: Bearer $HF_TOKEN" -o "$2" "$BASE/$1"
}
[ -s "$W/manifest.csv" ] || fetch dataset_manifest.csv "$W/manifest.csv"
ids=("$@")
[ ${#ids[@]} -gt 0 ] || mapfile -t ids < <(tail -n +2 "$W/manifest.csv" | cut -d, -f1)
for id in "${ids[@]}"; do
  [ -s "$W/res/$id.json" ] && continue
  row=$(grep "^$id," "$W/manifest.csv") || { echo "$id: not in the manifest"; continue; }
  # Manifest columns: imu_path (3), location_path (4, the phone's GPS log),
  # perspective_sparse_path (9), smartphone_poses_path (12).
  imu=$(cut -d, -f3 <<<"$row")
  gps=$(cut -d, -f4 <<<"$row")
  sparse=$(cut -d, -f9 <<<"$row")
  poses=$(cut -d, -f12 <<<"$row")
  D=$W/data/$id
  mkdir -p "$D"
  if fetch "$sparse/0/images.bin" "$D/images.bin" && fetch "$imu" "$D/insta360_imu.csv" &&
    fetch "$poses" "$D/$(basename "$poses")"; then
    [ -n "$gps" ] && { fetch "$gps" "$D/backgroundlocationData.txt" || rm -f "$D/backgroundlocationData.txt"; }
    python3 -I "$ROOT/scripts/insta360_imu_scale.py" scene "$D" "$W/res/$id.json" \
      >"$W/log/$id.txt" 2>&1 || echo "$id: failed, see $W/log/$id.txt"
    tail -2 "$W/log/$id.txt" | sed "s/^/${id:0:8} /"
  else
    echo "$id: download failed"
  fi
  rm -rf "${D:?}"
done
python3 -I "$ROOT/scripts/insta360_imu_scale.py" summary "$W"/res/*.json
