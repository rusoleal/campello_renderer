#!/bin/bash
# Run macOS example with Metal API Validation enabled (Debug builds).
# Build first with: ./build_macos_example.sh Debug
APP="build/macos/debug/examples/macos/campello_renderer_macos.app"
MTL_DEBUG_LAYER=1 \
MTL_DEBUG_LAYER_ERROR_MODE=assert \
MTL_DEBUG_LAYER_WARNING_MODE=assert \
"$APP/Contents/MacOS/campello_renderer_macos"
