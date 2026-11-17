#!/usr/bin/env bash
# Publish every workspace crate to crates.io, in dependency order.
#
# Safe to re-run: a crate whose current version is already in the crates.io
# index is skipped, so a run that stopped half-way (network, rate limit,
# timeout) resumes where it left off.
#
# crates.io limits how fast *new* crates may be created (a burst of a few,
# then one every few minutes). The first release creates eighteen crates, so
# on HTTP 429 the script waits until the time crates.io names, then retries.
#
# Usage:
#   CARGO_REGISTRY_TOKEN=... scripts/publish_crates.sh            # publish
#   scripts/publish_crates.sh --dry-run                           # list only
#
# Packaged contents are expected to be verify-built beforehand
# (`cargo package --workspace`, see docs/publishing.md); each crate is then
# uploaded with `--no-verify` so the release job does not build every crate a
# second time.
set -euo pipefail

if ! command -v cargo >/dev/null 2>&1 && [ -d "$HOME/.cargo/bin" ]; then
    PATH="$HOME/.cargo/bin:$PATH"
    export PATH
fi

dry_run=0
if [ "${1:-}" = "--dry-run" ]; then
    dry_run=1
fi

# Publish order (see docs/publishing.md; kept in sync with
# scripts/package_check.sh by scripts/check_release_metadata.sh).
packages="
visloc-core
visloc-dpvo-cuda-runtime
visloc-vision
visloc-localization
visloc-fusion
visloc-io
visloc-tracking
visloc-mapping
visloc-slam
visloc-basalt
visloc-objects
visloc-gsplat-core
visloc-gsplat-render
visloc-sift-gpu
visloc-ba-gpu
visloc-mvs
visloc-gsplat-train
visloc-rs
"

max_attempts="${VISLOC_PUBLISH_MAX_ATTEMPTS:-30}"
default_wait="${VISLOC_PUBLISH_RATE_LIMIT_WAIT:-600}"

version="$(cargo metadata --no-deps --format-version 1 \
    | python3 -c 'import json, sys
pkgs = {p["name"]: p["version"] for p in json.load(sys.stdin)["packages"]}
print(pkgs["visloc-rs"])')"
echo "Workspace version: $version"

# Path of a crate in the crates.io sparse index.
index_path() {
    local name="$1"
    case "${#name}" in
        1) echo "1/$name" ;;
        2) echo "2/$name" ;;
        3) echo "3/${name:0:1}/$name" ;;
        *) echo "${name:0:2}/${name:2:2}/$name" ;;
    esac
}

is_published() {
    local name="$1"
    local body
    # 404 (crate does not exist yet) and network errors both read as "not
    # published"; a real upload then reports the actual problem.
    body="$(curl -fsSL --retry 3 "https://index.crates.io/$(index_path "$name")" 2>/dev/null || true)"
    printf '%s\n' "$body" | grep -q "\"vers\":\"$version\""
}

# Seconds to wait after a 429, from crates.io's "try again after <date>".
rate_limit_wait() {
    local log="$1"
    local after now target
    after="$(sed -n 's/.*try again after \([^.]*GMT\).*/\1/p' "$log" | head -n 1)"
    if [ -n "$after" ] && target="$(date -d "$after" +%s 2>/dev/null)"; then
        now="$(date +%s)"
        if [ "$target" -gt "$now" ]; then
            echo $((target - now + 15))
            return
        fi
        echo 15
        return
    fi
    echo "$default_wait"
}

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

for package in $packages; do
    if is_published "$package"; then
        echo "skip     $package $version (already on crates.io)"
        continue
    fi
    if [ "$dry_run" = "1" ]; then
        echo "publish  $package $version (dry run)"
        continue
    fi

    attempt=1
    while :; do
        echo "publish  $package $version (attempt $attempt)"
        if cargo publish -p "$package" --no-verify 2>&1 | tee "$log"; then
            break
        fi
        if grep -q "already exists" "$log"; then
            echo "skip     $package $version (already on crates.io)"
            break
        fi
        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "error: giving up on $package after $attempt attempts" >&2
            exit 1
        fi
        if grep -qiE "429|too many|rate limit" "$log"; then
            wait="$(rate_limit_wait "$log")"
            echo "rate limited by crates.io; waiting ${wait}s"
        elif [ "$attempt" -ge 3 ]; then
            echo "error: $package failed $attempt times (not a rate limit)" >&2
            exit 1
        else
            wait=30
            echo "publish failed; retrying in ${wait}s"
        fi
        sleep "$wait"
        attempt=$((attempt + 1))
    done
done

if [ "$dry_run" = "1" ]; then
    echo "Dry run: nothing was uploaded."
else
    echo "All crates are on crates.io at $version."
fi
