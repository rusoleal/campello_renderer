#!/bin/bash
# Builds campello_renderer's test suite with Emscripten and runs it in
# headless Chrome (WebGPU), using campello_gpu's tests/webgpu/run_in_browser.mjs.
#
#   source ~/emsdk/emsdk_env.sh            # Emscripten 4.0.10+ (emdawnwebgpu)
#   test/webgpu/run_web_tests.sh [build-dir]
#
# Environment: CAMPELLO_GPU_DIR (campello_gpu checkout, default ../campello_gpu,
# also used as the campello_gpu source), CHROME_PATH / CHROME_FLAGS (runner).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
gpu="${CAMPELLO_GPU_DIR:-$repo/../campello_gpu}"
build="${1:-$repo/build/web}"
command -v emcmake > /dev/null || { echo "run_web_tests: emcmake not found (source emsdk_env.sh)" >&2; exit 3; }
[ -f "$gpu/tests/webgpu/run_in_browser.mjs" ] || { echo "run_web_tests: campello_gpu not found at $gpu" >&2; exit 3; }
emcmake cmake -S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCAMPELLO_RENDERER_BUILD_TEST=ON \
    -DFETCHCONTENT_SOURCE_DIR_EXTERN_CAMPELLO_GPU="$gpu" ${CAMPELLO_WEB_CMAKE_ARGS:-}
cmake --build "$build" --target campello_renderer_test -j8
node "$gpu/tests/webgpu/run_in_browser.mjs" "$build/campello_renderer_test.html" --timeout=300
