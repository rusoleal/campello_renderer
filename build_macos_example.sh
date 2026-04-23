#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default to Debug if no argument given
BUILD_TYPE="${1:-Debug}"
BUILD_TYPE_LOWER="$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$SCRIPT_DIR/build/macos/$BUILD_TYPE_LOWER"

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCAMPELLO_RENDERER_BUILD_MACOS_EXAMPLE=ON \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

cmake --build "$BUILD_DIR" --config "$BUILD_TYPE"
