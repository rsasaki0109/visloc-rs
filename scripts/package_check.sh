#!/usr/bin/env sh
set -eu

packages="
visloc-core
visloc-vision
visloc-localization
visloc-io
visloc-tracking
visloc-mapping
visloc-slam
visloc-fusion
visloc-rs
"

for package in $packages; do
    echo "Listing package contents: $package"
    cargo package -p "$package" --allow-dirty --no-verify --list >/dev/null
done

if [ "${VISLOC_PACKAGE_ALL:-0}" = "1" ]; then
    echo "Packaging every crate. This requires internal visloc-* dependencies to be available from the target registry."
else
    echo "Packaging first publishable crate: visloc-core"
    cargo package -p visloc-core --allow-dirty --no-verify
    echo "Set VISLOC_PACKAGE_ALL=1 to package every crate after internal dependencies are available from the registry."
    exit 0
fi

for package in $packages; do
    echo "Packaging crate: $package"
    cargo package -p "$package" --allow-dirty --no-verify
done
