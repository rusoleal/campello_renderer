#!/bin/bash
# Builds the test suite for arm64 Android, pushes it to the connected device
# (adb) and runs it there: the Vulkan backend on real hardware. CI has no
# device, so this is how the Vulkan path is tested.
#
#   ./run_android_tests.sh                      # every test
#   ./run_android_tests.sh '*GBuffer*:*Taa*'    # a gtest filter
#
# Environment:
#   ANDROID_HOME          the SDK (default ~/Library/Android/sdk); its NDK and CMake are used
#   ANDROID_NDK           an NDK (default: the newest under $ANDROID_HOME/ndk)
#   CAMPELLO_GPU_SOURCE   a local campello_gpu checkout to build against instead of the
#                         released one (FetchContent source override)
#   ANDROID_SERIAL        the device, when more than one is connected (adb)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILTER="${1:-*}"
ANDROID_HOME="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
if [ -z "${ANDROID_NDK:-}" ]; then
    ANDROID_NDK="$(ls -d "$ANDROID_HOME"/ndk/* 2>/dev/null | sort -V | tail -1)"
fi
CMAKE_DIR="$(ls -d "$ANDROID_HOME"/cmake/* 2>/dev/null | sort -V | tail -1)"
if [ -z "$ANDROID_NDK" ] || [ -z "$CMAKE_DIR" ]; then
    echo "Error: no NDK or CMake under $ANDROID_HOME (set ANDROID_HOME / ANDROID_NDK)." >&2
    exit 1
fi
command -v adb >/dev/null || { echo "Error: adb not found." >&2; exit 1; }
adb get-state >/dev/null 2>&1 || { echo "Error: no device connected (adb devices)." >&2; exit 1; }

BUILD_DIR="$SCRIPT_DIR/build/android/arm64"
REMOTE_DIR=/data/local/tmp/campello_renderer_test

CONFIGURE=(
    -S "$SCRIPT_DIR" -B "$BUILD_DIR" -G Ninja
    -DCMAKE_MAKE_PROGRAM="$CMAKE_DIR/bin/ninja"
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28
    -DCMAKE_BUILD_TYPE=Debug -DCAMPELLO_RENDERER_BUILD_TEST=ON
)
if [ -n "${CAMPELLO_GPU_SOURCE:-}" ]; then
    CONFIGURE+=(-DFETCHCONTENT_SOURCE_DIR_EXTERN_CAMPELLO_GPU="$CAMPELLO_GPU_SOURCE")
else
    CONFIGURE+=(-UFETCHCONTENT_SOURCE_DIR_EXTERN_CAMPELLO_GPU)
fi
"$CMAKE_DIR/bin/cmake" "${CONFIGURE[@]}" >/dev/null
"$CMAKE_DIR/bin/ninja" -C "$BUILD_DIR" campello_renderer_test

LIBCXX="$(find "$ANDROID_NDK/toolchains/llvm/prebuilt" -path '*aarch64-linux-android/libc++_shared.so' | head -1)"
adb shell mkdir -p "$REMOTE_DIR"
adb push \
    "$BUILD_DIR/campello_renderer_test" \
    "$BUILD_DIR/libcampello_renderer.so" \
    "$BUILD_DIR/_deps/extern_gltf-build/libgltf.so" \
    "$BUILD_DIR/_deps/extern_campello_image-build/libcampello_image.so" \
    "$BUILD_DIR/_deps/extern_campello_gpu-build/libcampello_gpu.so" \
    "$LIBCXX" \
    "$REMOTE_DIR/" >/dev/null
echo "Running on $(adb shell getprop ro.product.model | tr -d '\r') (Android $(adb shell getprop ro.build.version.release | tr -d '\r'))"
adb shell "cd $REMOTE_DIR && LD_LIBRARY_PATH=. ./campello_renderer_test --gtest_filter='$FILTER'"
