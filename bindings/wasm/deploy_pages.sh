#!/usr/bin/env bash
# Publishes the page as a static site on a GitHub Pages branch: the page, the
# two builds from ./build.sh, and the sensor-free demo if web/demo exists
# (ADVIO, CC BY-NC 4.0: the page credits it when the demo plays). The branch
# is replaced on each run.
#
#   ./deploy_pages.sh <git remote or URL> [branch, default gh-pages]
#
# Static hosts cannot send the COOP/COEP headers wasm threads need;
# web/coi-sw.js adds them in the browser. Telemetry and uploads, which need
# serve.py, are off there.
set -euo pipefail
cd "$(dirname "$0")"

remote="${1:?usage: deploy_pages.sh <git remote or URL> [branch]}"
branch="${2:-gh-pages}"
for pkg in pkg pkg-mt; do
  [[ -f "web/$pkg/visloc_wasm_bg.wasm" ]] || { echo "web/$pkg is missing: run ./build.sh first" >&2; exit 1; }
done
if [[ "$remote" != *[:/]* ]]; then
  remote="$(git remote get-url "$remote")"
fi

site="$(mktemp -d)"
trap 'rm -rf "$site"' EXIT
cp -r web/index.html web/app.js web/worker.js web/profiles.js web/recording.js \
  web/telemetry.js web/coi-sw.js web/vendor web/pkg web/pkg-mt "$site/"
[[ -d web/demo ]] && cp -r web/demo "$site/"
touch "$site/.nojekyll" # serve wasm-bindgen's snippets/ and anything else as is

git -C "$site" init -q -b "$branch"
git -C "$site" add -A
git -C "$site" commit -q -m "bindings/wasm page at $(git rev-parse --short HEAD)"
git -C "$site" push -q -f "$remote" "$branch"
echo "pushed $(du -sh "$site" --exclude=.git | cut -f1) to $branch"
