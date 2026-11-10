#!/bin/bash
# Validates and render-checks shaders/webgpu/default.wgsl in headless Chrome
# (WebGPU), against campello_renderer's web layouts. No Emscripten needed.
# Uses campello_gpu's tests/webgpu/run_in_browser.mjs (Node + Chrome).
#
#   test/webgpu/run_wgsl_check.sh            # campello_gpu checkout next to this repo
#   CAMPELLO_GPU_DIR=/path/to/campello_gpu test/webgpu/run_wgsl_check.sh
#   CHROME_PATH=... CHROME_FLAGS=...        # passed to the runner (e.g. SwiftShader on CI)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
gpu="${CAMPELLO_GPU_DIR:-$repo/../campello_gpu}"
runner="$gpu/tests/webgpu/run_in_browser.mjs"
[ -f "$runner" ] || { echo "run_wgsl_check: $runner not found (set CAMPELLO_GPU_DIR)" >&2; exit 3; }
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cp "$here/pbr_wgsl_test.html" "$repo/shaders/webgpu/default.wgsl" "$work/"
node "$runner" "$work/pbr_wgsl_test.html" --timeout=60
