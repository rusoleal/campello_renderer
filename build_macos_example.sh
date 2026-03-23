#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/macos"

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCAMPELLO_RENDERER_BUILD_MACOS_EXAMPLE=ON

cmake --build "$BUILD_DIR" --config Debug

APP="$BUILD_DIR/examples/macos/campello_renderer_macos.app"
open "$APP"
