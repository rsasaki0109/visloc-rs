#!/usr/bin/env bash
# Publishes the page as a static site on a GitHub Pages branch: the page, the
# two builds from ./build.sh, and the sensor-free demo if web/demo exists
# (ADVIO, CC BY-NC 4.0: the page credits it when the demo plays). The branch
# is replaced on each run.
#
#   ./deploy_pages.sh <git remote or URL> [branch, default gh-pages]
#   ./deploy_pages.sh --dir <directory>   # only write the site there
#
# Static hosts cannot send the COOP/COEP headers wasm threads need;
# web/coi-sw.js adds them in the browser. Telemetry and uploads, which need
# serve.py, are off there.
set -euo pipefail
cd "$(dirname "$0")"

usage="usage: deploy_pages.sh <git remote or URL> [branch] | --dir <directory>"
for pkg in pkg pkg-mt; do
  [[ -f "web/$pkg/visloc_wasm_bg.wasm" ]] || { echo "web/$pkg is missing: run ./build.sh first" >&2; exit 1; }
done

# assemble <directory>
assemble() {
  mkdir -p "$1"
  cp -r web/index.html web/app.js web/worker.js web/profiles.js web/recording.js \
    web/telemetry.js web/coi-sw.js web/vendor web/pkg web/pkg-mt "$1/"
  if [[ -d web/demo ]]; then cp -r web/demo "$1/"; fi
  touch "$1/.nojekyll" # serve wasm-bindgen's snippets/ and anything else as is
}

if [[ "${1:-}" == --dir ]]; then
  assemble "${2:?$usage}"
  exit
fi

remote="${1:?$usage}"
branch="${2:-gh-pages}"
if [[ "$remote" != *[:/]* ]]; then
  remote="$(git remote get-url "$remote")"
fi

site="$(mktemp -d)"
trap 'rm -rf "$site"' EXIT
assemble "$site"

git -C "$site" init -q -b "$branch"
git -C "$site" add -A
git -C "$site" commit -q -m "bindings/wasm page at $(git rev-parse --short HEAD)"
git -C "$site" push -q -f "$remote" "$branch"
echo "pushed $(du -sh "$site" --exclude=.git | cut -f1) to $branch"
